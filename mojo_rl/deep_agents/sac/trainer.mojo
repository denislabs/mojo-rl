"""SACTrainer — storage-framework SAC trainer (CPU gate; GPU stretch).

Assembles the migrated `mojo_rl.nn.storage` SAC blocks into a single
driver-conforming `SACTrainer` that conforms `OffPolicyAgentGpu` so both
the CPU single-env driver (`run_offpolicy_train`) and the batched drivers
type-check. The CPU path is the production path; the GPU-only batched
record / device-kernel-capture methods raise until the GPU storage path
is wired.

Pipeline (per train step, mirrors the proven convergence test):
  state.step_idx = step; state.alpha = exp(alpha_opt.value)
  sample_blk.step(state)                       # fills state.mb_*
  target_y_blk.step(state, actor, tgt1, tgt2)  # writes state.mb_y
  twin_critic_blk.step(state, c1, c1_opt, c2, c2_opt)
  out = actor_loss.forward_backward(actor, actor_opt, c1, c2, mb_s, alpha)
  state.log_prob_mean = out.log_prob_mean
  alpha_blk.step(state, alpha_opt)
  polyak_blk.step(state, pair1, pair2)

α is a HOST scalar on CPU (`state.alpha = exp(alpha_opt.value)`).

Dimensions (OBS / ACT / BATCH) derive from `SAMPLE` so they're specified
once (on the sample block type).
"""

from std.math import exp as fexp, log as flog, tanh as ftanh
from std.random import random_float64
from layout import Layout, LayoutTensor

from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.module import Module
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Xavier, Zero
from mojo_rl.nn.storage.primitives.rsample import RSample
from mojo_rl.nn.storage.optimizer.adam import Adam
from mojo_rl.nn.storage.optimizer.scalar_adam import ScalarAdam
from mojo_rl.nn.storage.core.checkpoint import save_params, load_params

from mojo_rl.nn.core.log_bundle import log_bundle
from mojo_rl.nn.core.metric import LogScalar

from ..data.n_step_replay import GPUNStepBuffer
from ..core.online_target_pair import OnlineTargetPair
from ..training.episode_tracker import EpisodeTracker
from ..training.trainer_block import TrainerState
from ..training.driver_offpolicy import OffPolicyAgentGpu
from ..training.blocks import SampleBlock, TwinCriticStep, PolyakStep
from .target_y_block import TargetYBlock
from .actor_loss import SACActorLoss
from .blocks.alpha_update_step import AlphaUpdateStep
from .metrics import SACMetrics


struct SACTrainer[
    train_target: StaticString,
    SAMPLE: SampleBlock,
    ACTOR: Module,
    CRITIC: Module,
](OffPolicyAgentGpu):
    """Storage-framework SAC trainer. Dimensions (OBS / ACT / BATCH) are
    derived from SAMPLE so the user specifies them once on the sample
    block type — symbolic-equality follows for the TrainerState the
    trainer holds vs the one SAMPLE.step expects."""

    comptime OBS_DIM: Int = Self.SAMPLE.OBS
    comptime ACT_DIM: Int = Self.SAMPLE.ACT
    comptime BATCH: Int = Self.SAMPLE.BATCH

    comptime AGENT_OBS_DIM: Int = Self.OBS_DIM
    comptime AGENT_ACT_DIM: Int = Self.ACT_DIM
    comptime AGENT_TRAIN_TARGET: StaticString = Self.train_target

    var actor: Self.ACTOR
    var pair1: OnlineTargetPair[Self.CRITIC]
    var pair2: OnlineTargetPair[Self.CRITIC]
    var actor_opt: Adam
    var critic1_opt: Adam
    var critic2_opt: Adam
    var alpha_opt: ScalarAdam

    var sample_blk: Self.SAMPLE
    var target_y_blk: TargetYBlock[
        Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM,
    ]
    var twin_critic_blk: TwinCriticStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
    ]
    var actor_loss_blk: SACActorLoss[Self.ACTOR, Self.CRITIC, Self.BATCH]
    var alpha_blk: AlphaUpdateStep[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]
    var polyak_blk: PolyakStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
    ]

    # select-action rsample (separate from the loss graphs' own rsamples).
    var sel: RSample[Self.ACT_DIM]

    var state: TrainerState[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]
    var tracker: EpisodeTracker
    var ctx: Optional[DeviceContext]

    # Owned action-selection scratch Tensors (lazily `.ensure`d per call).
    var _ob_scr: Tensor   # N_ENVS * OBS
    var _ao_scr: Tensor   # N_ENVS * 2*ACT
    var _alp_scr: Tensor  # N_ENVS * (ACT + 1)

    var action_scale: Scalar[DT]
    var learning_starts: Int

    # Host metric accumulators (CPU path; simple scalars like the test).
    var _actor_L_accum: Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    var _alpha_accum: Scalar[DT]
    var _update_count: Int
    var _total_train_steps: Int

    def __init__(out self):
        self.actor = Self.ACTOR()
        self.pair1 = OnlineTargetPair[Self.CRITIC]()
        self.pair2 = OnlineTargetPair[Self.CRITIC]()
        self.actor_opt = Adam(lr=Scalar[DT](3e-4))
        self.critic1_opt = Adam(lr=Scalar[DT](1e-3))
        self.critic2_opt = Adam(lr=Scalar[DT](1e-3))
        self.alpha_opt = ScalarAdam.new(flog(Scalar[DT](0.2)), Scalar[DT](3e-4))
        self.sample_blk = Self.SAMPLE()
        self.target_y_blk = TargetYBlock[
            Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM,
        ]()
        self.twin_critic_blk = TwinCriticStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ]()
        self.actor_loss_blk = SACActorLoss[
            Self.ACTOR, Self.CRITIC, Self.BATCH
        ]()
        self.alpha_blk = AlphaUpdateStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH
        ]()
        self.polyak_blk = PolyakStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ]()
        self.sel = RSample[Self.ACT_DIM]()
        self.state = TrainerState[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]()
        self.tracker = EpisodeTracker(
            window=List[Scalar[DT]](),
            window_size=0,
            idx=0,
            current_return=Scalar[DT](0.0),
            ep_count=0,
        )
        self.ctx = None
        self._ob_scr = Tensor()
        self._ao_scr = Tensor()
        self._alp_scr = Tensor()
        self.action_scale = Scalar[DT](1.0)
        self.learning_starts = 1_000
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum = Scalar[DT](0.0)
        self._update_count = 0
        self._total_train_steps = 0

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        actor_lr: Scalar[DT] = Scalar[DT](3e-4),
        critic_lr: Scalar[DT] = Scalar[DT](1e-3),
        alpha_lr: Scalar[DT] = Scalar[DT](3e-4),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        tau: Scalar[DT] = Scalar[DT](0.005),
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        init_alpha: Scalar[DT] = Scalar[DT](0.2),
        target_entropy: Scalar[DT] = Scalar[DT](-1.0),
        learning_starts: Int = 1_000,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
        max_grad_norm: Scalar[DT] = Scalar[DT](0.0),
        per_alpha: Scalar[DT] = Scalar[DT](0.6),
        per_beta: Scalar[DT] = Scalar[DT](0.4),
        per_epsilon: Scalar[DT] = Scalar[DT](1e-6),
        use_bf16: Bool = False,
        use_ere: Bool = False,
        ere_eta: Scalar[DT] = Scalar[DT](0.996),
        ere_c_min: Int = 1,
        ere_k_max: Int = 1000,
    ) raises -> Self:
        """Unified factory. PER args applied via the SampleBlock trait's
        `configure_per` (no-op for uniform blocks). `ctx` required for
        train_target='gpu'."""
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "SACTrainer: target must be 'cpu' or 'gpu'"
        comptime if Self.train_target == "gpu":
            if not ctx:
                raise Error("SACTrainer.make[target='gpu']: ctx required")

        var t = Self()
        t.ctx = ctx

        t.actor = Self.ACTOR.make[Self.train_target, Xavier](ctx)
        t.pair1 = OnlineTargetPair[Self.CRITIC].make[
            Self.train_target, Xavier
        ](ctx)
        t.pair2 = OnlineTargetPair[Self.CRITIC].make[
            Self.train_target, Xavier
        ](ctx)

        t.actor_opt = Adam(lr=actor_lr)
        t.critic1_opt = Adam(lr=critic_lr)
        t.critic2_opt = Adam(lr=critic_lr)
        comptime if Self.train_target == "gpu":
            t.actor_opt.adopt[Self.train_target, Self.ACTOR](t.actor, ctx)
            t.critic1_opt.adopt[Self.train_target, Self.CRITIC](
                t.pair1.online, ctx
            )
            t.critic2_opt.adopt[Self.train_target, Self.CRITIC](
                t.pair2.online, ctx
            )

        t.target_y_blk = TargetYBlock[
            Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM,
        ].make[Self.train_target](
            action_scale=action_scale, gamma=gamma, ctx=ctx
        )
        t.twin_critic_blk = TwinCriticStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ].make[Self.train_target](ctx=ctx)
        t.actor_loss_blk = SACActorLoss[
            Self.ACTOR, Self.CRITIC, Self.BATCH
        ].make[Self.train_target](ctx=ctx, action_scale=action_scale)
        t.alpha_blk = AlphaUpdateStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH
        ].make(target_entropy=target_entropy)
        t.polyak_blk = PolyakStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ].make(tau=tau)

        t.alpha_opt = ScalarAdam.new(flog(init_alpha), alpha_lr)

        t.sel = RSample[Self.ACT_DIM].make[Self.train_target, Zero](ctx)
        t.sel.action_scale = action_scale

        t.state = TrainerState[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH
        ].make[Self.train_target](ctx=ctx)

        t.tracker = EpisodeTracker.new(
            window_size=window_size, initial_fill=initial_episode_fill
        )

        t.action_scale = action_scale
        t.learning_starts = learning_starts

        # PER / ERE wiring: no-op default for uniform blocks.
        t.sample_blk.configure_per(
            alpha=per_alpha, beta=per_beta, epsilon=per_epsilon
        )
        t.sample_blk.setup(learning_starts, ctx=ctx)
        t.sample_blk.configure_ere(
            enable=use_ere, eta=ere_eta, c_min=ere_c_min, k_max=ere_k_max
        )

        # Pre-size the action scratch for the single-env path.
        comptime if Self.train_target == "cpu":
            t._ob_scr.ensure(Self.OBS_DIM)
            t._ao_scr.ensure(2 * Self.ACT_DIM)
            t._alp_scr.ensure(Self.ACT_DIM + 1)
        return t^

    def set_beta(mut self, beta: Scalar[DT]):
        """PER IS-β anneal hook. No-op for uniform sample blocks."""
        self.sample_blk.set_beta(beta)

    # ─── train_step ────────────────────────────────────────────────────
    def train_step(mut self, step_idx: Int) raises -> Bool:
        self.state.step_idx = step_idx
        self.state.did_step = True
        comptime if Self.train_target == "cpu":
            self.state.alpha = fexp(self.alpha_opt.value)
        else:
            self.state.ctx = self.ctx

        self.sample_blk.step(self.state)
        if not self.state.did_step:
            return False

        self.target_y_blk.step[Self.train_target](
            self.state, self.actor, self.pair1.target_net, self.pair2.target_net
        )
        self.twin_critic_blk.step[Self.train_target](
            self.state,
            self.pair1.online,
            self.critic1_opt,
            self.pair2.online,
            self.critic2_opt,
        )
        var out = self.actor_loss_blk.forward_backward[Self.train_target](
            self.actor,
            self.actor_opt,
            self.pair1.online,
            self.pair2.online,
            self.state.mb_s,
            self.state.alpha,
            self.ctx,
        )
        self.state.log_prob_mean = out.log_prob_mean
        self.state.actor_loss = out.loss
        self.alpha_blk.step[Self.train_target](self.state, self.alpha_opt)
        self.polyak_blk.step[Self.train_target](
            self.state, self.pair1, self.pair2
        )

        # PER tail (no-op for uniform blocks).
        self.sample_blk.update_priorities(self.state)

        # Host bookkeeping.
        self._actor_L_accum += out.loss
        self._critic_L_accum += self.state.critic_loss
        self._alpha_accum += fexp(self.alpha_opt.value)
        self._update_count += 1
        self._total_train_steps += 1
        return True

    def total_train_steps(self) -> Int:
        return self._total_train_steps

    # ─── Action selection ──────────────────────────────────────────────
    def select_action_batched[
        N_ENVS: Int
    ](
        mut self,
        obs: LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.AGENT_OBS_DIM), MutAnyOrigin
        ],
        action: LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.AGENT_ACT_DIM), MutAnyOrigin
        ],
        ao_scratch: LayoutTensor[
            DT, Layout.row_major(N_ENVS, 2 * Self.AGENT_ACT_DIM), MutAnyOrigin
        ],
        alp_scratch: LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.AGENT_ACT_DIM + 1), MutAnyOrigin
        ],
        step_idx: Int,
    ) raises:
        comptime assert N_ENVS > 0, "N_ENVS must be > 0"
        comptime ACT = Self.ACT_DIM
        comptime OBS = Self.OBS_DIM

        # ── Warmup: uniform random action in [-action_scale, +scale].
        if step_idx < self.learning_starts:
            comptime if Self.train_target == "cpu":
                for env in range(N_ENVS):
                    for j in range(ACT):
                        var u = Scalar[DT](2.0 * random_float64() - 1.0)
                        action[env, j] = u * self.action_scale
                return
            else:
                raise Error(
                    "SACTrainer.select_action_batched: GPU warmup not yet"
                    " migrated to storage"
                )

        # ── Policy forward through the STORAGE surface.
        comptime if Self.train_target == "cpu":
            # Bridge LayoutTensor obs → owned Tensor; storage actor.forward
            # wants a Tensor.
            self._ob_scr.ensure(N_ENVS * OBS)
            for env in range(N_ENVS):
                for d in range(OBS):
                    self._ob_scr.data[env * OBS + d] = rebind[Scalar[DT]](
                        obs[env, d]
                    )
            self._ao_scr.ensure(N_ENVS * 2 * ACT)
            self._alp_scr.ensure(N_ENVS * (ACT + 1))
            self.actor.forward["cpu", N_ENVS](
                TensorRefs[Self.ACTOR.ARITY](self._ob_scr), self._ao_scr
            )
            self.sel.forward["cpu", N_ENVS](
                TensorRefs[1](self._ao_scr), self._alp_scr
            )
            for env in range(N_ENVS):
                for j in range(ACT):
                    var a = self._alp_scr.data[env * (ACT + 1) + j]
                    if a > self.action_scale:
                        a = self.action_scale
                    elif a < -self.action_scale:
                        a = -self.action_scale
                    action[env, j] = a
            # silence unused warnings on the driver-owned scratch views.
            _ = ao_scratch
            _ = alp_scratch
        else:
            _ = obs
            _ = action
            _ = ao_scratch
            _ = alp_scratch
            raise Error(
                "SACTrainer.select_action_batched: GPU policy path not yet"
                " migrated to storage"
            )

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        comptime ACT = Self.ACT_DIM
        comptime OBS = Self.OBS_DIM
        comptime if Self.train_target == "cpu":
            self._ob_scr.ensure(OBS)
            self._ao_scr.ensure(2 * ACT)
            for d in range(OBS):
                self._ob_scr.data[d] = obs[d]
            self.actor.forward["cpu", 1](
                TensorRefs[Self.ACTOR.ARITY](self._ob_scr), self._ao_scr
            )
            for j in range(ACT):
                var a = ftanh(self._ao_scr.data[j]) * self.action_scale
                if a > self.action_scale:
                    a = self.action_scale
                elif a < -self.action_scale:
                    a = -self.action_scale
                action_out[j] = a
        else:
            _ = obs
            _ = action_out
            raise Error(
                "SACTrainer.select_greedy_action: GPU path not yet migrated"
                " to storage"
            )

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        """Host-list stochastic action — user-facing entry for smoke tests
        that bypass the driver. Stages obs into the owned scratch and runs
        the warmup/policy path directly."""
        comptime ACT = Self.ACT_DIM
        comptime OBS = Self.OBS_DIM
        comptime if Self.train_target == "cpu":
            if step_idx < self.learning_starts:
                for j in range(ACT):
                    var u = Scalar[DT](2.0 * random_float64() - 1.0)
                    action_out[j] = u * self.action_scale
                return
            self._ob_scr.ensure(OBS)
            self._ao_scr.ensure(2 * ACT)
            self._alp_scr.ensure(ACT + 1)
            for d in range(OBS):
                self._ob_scr.data[d] = obs[d]
            self.actor.forward["cpu", 1](
                TensorRefs[Self.ACTOR.ARITY](self._ob_scr), self._ao_scr
            )
            self.sel.forward["cpu", 1](
                TensorRefs[1](self._ao_scr), self._alp_scr
            )
            for j in range(ACT):
                var a = self._alp_scr.data[j]
                if a > self.action_scale:
                    a = self.action_scale
                elif a < -self.action_scale:
                    a = -self.action_scale
                action_out[j] = a
        else:
            _ = obs
            _ = action_out
            raise Error(
                "SACTrainer.select_action: GPU path not yet migrated to"
                " storage"
            )

    # ─── Record ────────────────────────────────────────────────────────
    def record(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        self.tracker.add_reward(reward)
        self.sample_blk.add(obs, action, reward, next_obs, done, ctx=self.ctx)

    def _replay_add(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        self.sample_blk.add(obs, action, reward, next_obs, done, ctx=self.ctx)

    def _tracker_ptr(self) -> UnsafePointer[EpisodeTracker, MutAnyOrigin]:
        return rebind[UnsafePointer[EpisodeTracker, MutAnyOrigin]](
            UnsafePointer(to=self.tracker)
        )

    # ─── GPU-batched record surface (raise until GPU storage migrated) ─
    def record_batch_gpu[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        prev_obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        reward_dev: DeviceBuffer[DT],
        obs_dev: DeviceBuffer[DT],
        done_dev: DeviceBuffer[DT],
    ) raises:
        raise Error("SACTrainer.record_batch_gpu: GPU path not yet migrated")

    def record_batch_gpu_nstep[
        N_ENVS: Int, NS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut nstep_buf: GPUNStepBuffer[
            NS, Self.AGENT_OBS_DIM, Self.AGENT_ACT_DIM, N_ENVS,
        ],
        prev_obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        reward_dev: DeviceBuffer[DT],
        obs_dev: DeviceBuffer[DT],
        done_dev: DeviceBuffer[DT],
    ) raises:
        raise Error(
            "SACTrainer.record_batch_gpu_nstep: GPU path not yet migrated"
        )

    # ─── Metrics / logging ─────────────────────────────────────────────
    def flush_metrics[
        L: Logger = NoOpLogger
    ](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> SACMetrics:
        """Drain accumulators into a SACMetrics bundle. CPU host scalars
        only; the rich per-batch diags are omitted (the legacy GPU device
        accumulators are not part of the storage CPU gate)."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var bundle = SACMetrics(
            actor_loss=LogScalar[DT](self._actor_L_accum * inv),
            critic_loss=LogScalar[DT](self._critic_L_accum * inv),
            alpha=LogScalar[DT](self._alpha_accum * inv),
            mean_q=LogScalar[DT](Scalar[DT](0.0)),
            mean_target=LogScalar[DT](Scalar[DT](0.0)),
            mean_reward=LogScalar[DT](Scalar[DT](0.0)),
            mean_next_q=LogScalar[DT](Scalar[DT](0.0)),
            mean_done=LogScalar[DT](Scalar[DT](0.0)),
            mean_abs_action=LogScalar[DT](Scalar[DT](0.0)),
            train_steps=LogScalar[DT](Scalar[DT](self._total_train_steps)),
            n_updates=LogScalar[DT](Scalar[DT](self._update_count)),
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum = Scalar[DT](0.0)
        self._update_count = 0
        if Bool(logger):
            log_bundle(logger.value()[], bundle, step)
        return bundle^

    def flush_metrics_through_logger[
        L: Logger
    ](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        _ = self.flush_metrics[L](logger, step)

    def flush_train_log(
        mut self,
    ) -> Tuple[Scalar[DT], Scalar[DT], Scalar[DT], Int]:
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var out = (
            self._actor_L_accum * inv,
            self._critic_L_accum * inv,
            self._alpha_accum * inv,
            self._update_count,
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum = Scalar[DT](0.0)
        self._update_count = 0
        return out

    # ─── Checkpoint (storage save_params for the 3 modules) ────────────
    def save_state(mut self, path: String) raises:
        """Write the actor + twin critic online nets via storage
        `save_params`. Three files: `path`.actor / .critic1 / .critic2.
        Optimizer moments + α are NOT persisted (resume re-warms)."""
        save_params[Self.train_target, Self.ACTOR](
            self.actor, path + ".actor", self.ctx, save_moments=False
        )
        save_params[Self.train_target, Self.CRITIC](
            self.pair1.online, path + ".critic1", self.ctx, save_moments=False
        )
        save_params[Self.train_target, Self.CRITIC](
            self.pair2.online, path + ".critic2", self.ctx, save_moments=False
        )

    def load_state(mut self, path: String) raises:
        load_params[Self.train_target, Self.ACTOR](
            self.actor, path + ".actor", self.ctx
        )
        load_params[Self.train_target, Self.CRITIC](
            self.pair1.online, path + ".critic1", self.ctx
        )
        load_params[Self.train_target, Self.CRITIC](
            self.pair2.online, path + ".critic2", self.ctx
        )
        self.pair1.target_net.polyak_from[Self.train_target](
            self.pair1.online, Scalar[DT](1.0), self.ctx
        )
        self.pair2.target_net.polyak_from[Self.train_target](
            self.pair2.online, Scalar[DT](1.0), self.ctx
        )

    def flush_timer_log(mut self) -> String:
        return String("")

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
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.storage.core.module import Module
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Xavier, Zero
from mojo_rl.nn.storage.primitives.rsample import RSample
from mojo_rl.nn.storage.optimizer.adam import Adam
from mojo_rl.nn.storage.optimizer.scalar_adam import ScalarAdam
from mojo_rl.nn.storage.core.checkpoint import (
    CheckpointWriter, CheckpointReader, _split_lines,
)

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


# ──────────────────────────────────────────────────────────────────────
# GPU device kernels for the batched action-selection path. Mirror the
# DDPG trainer's device body (Philox warmup + copy/clamp) adapted to the
# storage actor/rsample surface (which take owned Tensors, so obs is
# COPIED into the trainer's device scratch before the actor forward).
# ──────────────────────────────────────────────────────────────────────


def _sac_warmup_uniform_kernel[
    N_ENVS: Int, ACT: Int
](
    action_dest: LayoutTensor[DT, Layout.row_major(N_ENVS, ACT), MutAnyOrigin],
    action_scale: Scalar[DT],
    seed: UInt64,
    offset_base: UInt64,
):
    """Per-lane Philox uniform → [N_ENVS, ACT] of Uniform(-scale, +scale)."""
    var i = Int(global_idx.x)
    var total = N_ENVS * ACT
    if i >= total:
        return
    var philox = PhiloxRandom(seed=seed + UInt64(i), offset=offset_base)
    var u = Float32(philox.step_uniform()[0])
    var s = Scalar[DT](2.0) * Scalar[DT](u) - Scalar[DT](1.0)
    action_dest[i // ACT, i % ACT] = s * action_scale


def _sac_copy2d_kernel[
    N_ENVS: Int, D: Int
](
    src: LayoutTensor[DT, Layout.row_major(N_ENVS, D), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(N_ENVS, D), MutAnyOrigin],
):
    """dst[e,d] = src[e,d] — bridge the driver's obs view into the trainer's
    owned device scratch the storage actor.forward consumes."""
    var i = Int(global_idx.x)
    var total = N_ENVS * D
    if i < total:
        dst[i // D, i % D] = rebind[Scalar[DT]](src[i // D, i % D])


def _sac_clamp_action_kernel[
    N_ENVS: Int, ACT: Int, ALP: Int
](
    alp: LayoutTensor[DT, Layout.row_major(N_ENVS, ALP), MutAnyOrigin],
    action: LayoutTensor[DT, Layout.row_major(N_ENVS, ACT), MutAnyOrigin],
    scale: Scalar[DT],
):
    """action[e,j] = clamp(alp[e,j], ±scale) — drop the trailing log-prob
    column of the rsample output and clamp the squashed action."""
    var i = Int(global_idx.x)
    var total = N_ENVS * ACT
    if i < total:
        var e = i // ACT
        var j = i % ACT
        var a = rebind[Scalar[DT]](alp[e, j])
        if a > scale:
            a = scale
        elif a < -scale:
            a = -scale
        action[e, j] = a


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

    # Philox state for the GPU batched warmup kernel (gpu path only).
    var _warmup_rng_seed: UInt64
    var _warmup_rng_offset: UInt64

    # Host metric accumulators (CPU path; simple scalars like the test).
    var _actor_L_accum: Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    var _alpha_accum: Scalar[DT]
    # Diagnostic accumulators (CPU): batch means drained at flush_metrics.
    var _mean_q_accum: Scalar[DT]            # online Q1(s, a) over the batch
    var _mean_target_accum: Scalar[DT]       # Bellman target y
    var _mean_reward_accum: Scalar[DT]       # batch reward
    var _mean_next_q_accum: Scalar[DT]       # min(Q1_t,Q2_t)(s',a') bootstrap
    var _mean_done_accum: Scalar[DT]         # batch done
    var _mean_abs_action_accum: Scalar[DT]   # mean |action|
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
        self._warmup_rng_seed = UInt64(0x5AC_C0FFEE)
        self._warmup_rng_offset = UInt64(0)
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum = Scalar[DT](0.0)
        self._mean_q_accum = Scalar[DT](0.0)
        self._mean_target_accum = Scalar[DT](0.0)
        self._mean_reward_accum = Scalar[DT](0.0)
        self._mean_next_q_accum = Scalar[DT](0.0)
        self._mean_done_accum = Scalar[DT](0.0)
        self._mean_abs_action_accum = Scalar[DT](0.0)
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
        # α is a HOST scalar both on CPU and GPU (the blocks read state.alpha
        # into their Scale node / sac_target_y; device-α capture is deferred).
        self.state.alpha = fexp(self.alpha_opt.value)
        comptime if Self.train_target == "gpu":
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
        # Diagnostic batch means (CPU host data; GPU path leaves these 0.0,
        # same convention as the legacy GPU-SAC diagnostics).
        comptime if Self.train_target == "cpu":
            comptime B = Self.BATCH
            comptime A = Self.ACT_DIM
            var inv_b = Scalar[DT](1.0) / Scalar[DT](B)
            var sq: Scalar[DT] = 0.0
            var sy: Scalar[DT] = 0.0
            var sr: Scalar[DT] = 0.0
            var snq: Scalar[DT] = 0.0
            var sd: Scalar[DT] = 0.0
            for b in range(B):
                sq += self.twin_critic_blk.inner.c1._mb_q.data[b]
                sy += self.state.mb_y.data[b]
                sr += self.state.mb_r.data[b]
                snq += self.target_y_blk.graph.node_output["min_q"]().data[b]
                sd += self.state.mb_d.data[b]
            var saa: Scalar[DT] = 0.0
            for k in range(B * A):
                var av = self.state.mb_a.data[k]
                saa += av if av >= 0 else -av
            self._mean_q_accum += sq * inv_b
            self._mean_target_accum += sy * inv_b
            self._mean_reward_accum += sr * inv_b
            self._mean_next_q_accum += snq * inv_b
            self._mean_done_accum += sd * inv_b
            self._mean_abs_action_accum += saa / Scalar[DT](B * A)
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
                var c = self.ctx.value()
                comptime tot = N_ENVS * ACT
                c.enqueue_function[_sac_warmup_uniform_kernel[N_ENVS, ACT]](
                    action,
                    self.action_scale,
                    self._warmup_rng_seed,
                    self._warmup_rng_offset,
                    grid_dim=(tot + TPB - 1) // TPB,
                    block_dim=TPB,
                )
                self._warmup_rng_offset += UInt64(N_ENVS * ACT * 2)
                return

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
            # Bridge the driver's device obs view → owned device scratch
            # (storage actor.forward consumes an owned Tensor), run actor →
            # rsample on device, then clamp the squashed action out.
            var c = self.ctx.value()
            self._ob_scr.ensure_gpu(c, N_ENVS * OBS)
            self._ao_scr.ensure_gpu(c, N_ENVS * 2 * ACT)
            self._alp_scr.ensure_gpu(c, N_ENVS * (ACT + 1))
            comptime tot_obs = N_ENVS * OBS
            c.enqueue_function[_sac_copy2d_kernel[N_ENVS, OBS]](
                obs,
                self._ob_scr.lt["gpu", Layout.row_major(N_ENVS, OBS)](),
                grid_dim=(tot_obs + TPB - 1) // TPB,
                block_dim=TPB,
            )
            self.actor.forward["gpu", N_ENVS](
                TensorRefs[Self.ACTOR.ARITY](self._ob_scr), self._ao_scr,
                self.ctx,
            )
            self.sel.forward["gpu", N_ENVS](
                TensorRefs[1](self._ao_scr), self._alp_scr, self.ctx
            )
            comptime tot_act = N_ENVS * ACT
            c.enqueue_function[
                _sac_clamp_action_kernel[N_ENVS, ACT, ACT + 1]
            ](
                self._alp_scr.lt["gpu", Layout.row_major(N_ENVS, ACT + 1)](),
                action,
                self.action_scale,
                grid_dim=(tot_act + TPB - 1) // TPB,
                block_dim=TPB,
            )
            _ = ao_scratch
            _ = alp_scratch

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
            # Fresh single-env Tensors (NOT the batched _ob/_ao scratch, whose
            # `n` is sized for N_ENVS during training — `upload` walks `n` host
            # elements, so reusing them here would read past the 1-env fill).
            var c = self.ctx.value()
            var ob = Tensor.alloc(OBS)
            for d in range(OBS):
                ob.data[d] = obs[d]
            ob.upload(c)
            var ao = Tensor.alloc_gpu(c, 2 * ACT)
            self.actor.forward["gpu", 1](
                TensorRefs[Self.ACTOR.ARITY](ob), ao, self.ctx
            )
            ao.download(c)
            for j in range(ACT):
                var a = ftanh(ao.data[j]) * self.action_scale
                if a > self.action_scale:
                    a = self.action_scale
                elif a < -self.action_scale:
                    a = -self.action_scale
                action_out[j] = a

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
            if step_idx < self.learning_starts:
                for j in range(ACT):
                    var u = Scalar[DT](2.0 * random_float64() - 1.0)
                    action_out[j] = u * self.action_scale
                return
            # Fresh single-env Tensors (see select_greedy_action for why the
            # batched scratch can't be reused on the host-list path).
            var c = self.ctx.value()
            var ob = Tensor.alloc(OBS)
            for d in range(OBS):
                ob.data[d] = obs[d]
            ob.upload(c)
            var ao = Tensor.alloc_gpu(c, 2 * ACT)
            var alp = Tensor.alloc_gpu(c, ACT + 1)
            self.actor.forward["gpu", 1](
                TensorRefs[Self.ACTOR.ARITY](ob), ao, self.ctx
            )
            self.sel.forward["gpu", 1](TensorRefs[1](ao), alp, self.ctx)
            alp.download(c)
            for j in range(ACT):
                var a = alp.data[j]
                if a > self.action_scale:
                    a = self.action_scale
                elif a < -self.action_scale:
                    a = -self.action_scale
                action_out[j] = a

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
        """Delegate the N_ENVS device transitions to the sample block's GPU
        replay (one kernel launch)."""
        self.sample_blk.add_batch_gpu[N_ENVS](
            ctx, prev_obs_dev, action_dev, reward_dev, obs_dev, done_dev,
        )

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
        """Push the device transitions through the n-step buffer, then store
        matured n-step transitions into the GPU replay via the sample block."""
        nstep_buf.process(
            ctx, prev_obs_dev, action_dev, reward_dev, obs_dev, done_dev,
        )
        self.sample_blk.store_via_block_gpu[N_ENVS, NS](ctx, nstep_buf)

    # ─── Metrics / logging ─────────────────────────────────────────────
    def flush_metrics[
        L: Logger = NoOpLogger
    ](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> SACMetrics:
        """Drain accumulators into a SACMetrics bundle. On CPU the full
        per-batch diagnostics (mean_q / mean_target / mean_next_q / mean_reward
        / mean_done / mean_abs_action) are real; the GPU path leaves them 0.0."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var bundle = SACMetrics(
            actor_loss=LogScalar[DT](self._actor_L_accum * inv),
            critic_loss=LogScalar[DT](self._critic_L_accum * inv),
            alpha=LogScalar[DT](self._alpha_accum * inv),
            mean_q=LogScalar[DT](self._mean_q_accum * inv),
            mean_target=LogScalar[DT](self._mean_target_accum * inv),
            mean_reward=LogScalar[DT](self._mean_reward_accum * inv),
            mean_next_q=LogScalar[DT](self._mean_next_q_accum * inv),
            mean_done=LogScalar[DT](self._mean_done_accum * inv),
            mean_abs_action=LogScalar[DT](self._mean_abs_action_accum * inv),
            train_steps=LogScalar[DT](Scalar[DT](self._total_train_steps)),
            n_updates=LogScalar[DT](Scalar[DT](self._update_count)),
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum = Scalar[DT](0.0)
        self._mean_q_accum = Scalar[DT](0.0)
        self._mean_target_accum = Scalar[DT](0.0)
        self._mean_reward_accum = Scalar[DT](0.0)
        self._mean_next_q_accum = Scalar[DT](0.0)
        self._mean_done_accum = Scalar[DT](0.0)
        self._mean_abs_action_accum = Scalar[DT](0.0)
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

    # ─── Checkpoint (ONE file: actor + twin critics in a v2 envelope) ──────
    def save_state(mut self, path: String) raises:
        """Write actor + the two ONLINE critics into a SINGLE `nn-ckpt v2`
        file, sections name-prefixed `actor.` / `critic1.` / `critic2.` (one
        shared `CheckpointWriter` → one header → one `open`). Optimizer moments
        + α are NOT persisted (resume re-warms)."""
        var w = CheckpointWriter(save_moments=False)
        w.mode = 0
        self.actor.for_each_param[Self.train_target](w, self.ctx, "actor")
        self.pair1.online.for_each_param[Self.train_target](w, self.ctx, "critic1")
        self.pair2.online.for_each_param[Self.train_target](w, self.ctx, "critic2")
        w.mode = 1
        self.actor.for_each_state[Self.train_target](w, self.ctx, "actor")
        self.pair1.online.for_each_state[Self.train_target](w, self.ctx, "critic1")
        self.pair2.online.for_each_state[Self.train_target](w, self.ctx, "critic2")
        with open(path, "w") as f:
            f.write(w.content)

    def load_state(mut self, path: String) raises:
        """Restore actor + twin online critics from the single envelope (same
        walk order as `save_state`), then hard-copy online → target."""
        var content: String
        with open(path, "r") as f:
            content = String(f.read())
        var lines = _split_lines(content)
        var body = List[String]()
        for li in range(len(lines)):
            if lines[li].startswith("storage-ckpt"):
                continue
            body.append(lines[li])
        var r = CheckpointReader(body^)
        r.mode = 0
        self.actor.for_each_param[Self.train_target](r, self.ctx, "actor")
        self.pair1.online.for_each_param[Self.train_target](r, self.ctx, "critic1")
        self.pair2.online.for_each_param[Self.train_target](r, self.ctx, "critic2")
        r.mode = 1
        self.actor.for_each_state[Self.train_target](r, self.ctx, "actor")
        self.pair1.online.for_each_state[Self.train_target](r, self.ctx, "critic1")
        self.pair2.online.for_each_state[Self.train_target](r, self.ctx, "critic2")
        self.pair1.target_net.polyak_from[Self.train_target](
            self.pair1.online, Scalar[DT](1.0), self.ctx
        )
        self.pair2.target_net.polyak_from[Self.train_target](
            self.pair2.online, Scalar[DT](1.0), self.ctx
        )

    def flush_timer_log(mut self) -> String:
        return String("")

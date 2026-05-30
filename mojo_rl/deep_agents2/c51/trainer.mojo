"""C51Trainer — distributional DQN trainer (Bellemare et al. 2017).

CPU-only initial port. Pipeline body mirrors `dqn/trainer.mojo`:
sample → target-Y (categorical projection) → q-update (cross-entropy)
→ polyak. Differences from DQN:

  - `Q_NET.OUT_DIM == NA · N_ATOMS` (per-atom logits instead of Q-values).
  - Target is a distribution `m [B, N_ATOMS]` (NOT a scalar) — the
    trainer owns its own `_mb_m` scratch instead of using `state.mb_y`.
  - Action selection picks `argmax_a Σ_k softmax(logits[b, a])_k · z_k`
    (expected Q from the distribution), not a plain `argmax` over Q.

Conforms to `OffPolicyDiscreteAgent` so it slots into the existing
`run_offpolicy_discrete_train` driver. PER + N-step plumbing inherited
from the SAMPLE block + target-Y γ^n.
"""

from std.math import exp as fexp
from std.random import random_float64
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core import Module
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn2.core.checkpoint import (
    save_state_v2_body, load_state_v2_body,
    save_state_v2_body_gpu, load_state_v2_body_gpu,
)
from mojo_rl.nn2.core.log_bundle import log_bundle
from mojo_rl.nn2.core.map_params import hard_copy_params
from mojo_rl.nn2.core.metric import LogScalar
from mojo_rl.nn2.core.scratch import Scratch
from mojo_rl.nn2.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.training.timer import Timer
from ..core.checkpoint_helpers import (
    save_optimizer_v2_body, load_optimizer_v2_body,
    save_optimizer_v2_body_gpu, load_optimizer_v2_body_gpu,
    split_lines_v2, read_file_v2, expect_v2_header,
)
from ..core.online_target_pair import OnlineTargetPair
from ..training.episode_tracker import EpisodeTracker
from ..training.trainer_block import TrainerState
from ..training.driver_offpolicy_discrete import OffPolicyDiscreteAgent
from ..training.blocks import SampleBlock, SinglePolyakStep
from .blocks.target_y_step import C51TargetYStep
from .blocks.q_update_step import C51QUpdateStep
from .metrics import C51Metrics


struct C51Trainer[
    train_target: StaticString,
    SAMPLE: SampleBlock,
    Q_NET: Module,
    N_ATOMS: Int = 51,
    NUM_ACTIONS: Int = 2,
    DOUBLE: Bool = False,
](OffPolicyDiscreteAgent):
    """Q_NET.OUT_DIM must equal NUM_ACTIONS · N_ATOMS (per-atom logits).
    Standard Rainbow defaults: N_ATOMS=51, V_min=-10, V_max=+10."""

    comptime OBS_DIM: Int = Self.SAMPLE.OBS
    comptime ACT_DIM: Int = Self.SAMPLE.ACT
    comptime BATCH: Int = Self.SAMPLE.BATCH

    comptime AGENT_TRAIN_TARGET: StaticString = Self.train_target
    comptime AGENT_OBS_DIM: Int = Self.OBS_DIM
    comptime AGENT_NUM_ACTIONS: Int = Self.NUM_ACTIONS

    comptime _T_SAMPLE = 0
    comptime _T_TARGET_Y = 1
    comptime _T_CRITIC = 2
    comptime _T_POLYAK = 3

    var pair: OnlineTargetPair[Self.Q_NET]
    var q_opt: Adam
    var sample_blk: Self.SAMPLE
    var target_y_blk: C51TargetYStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.NUM_ACTIONS,
        Self.N_ATOMS, Self.Q_NET, Self.DOUBLE,
    ]
    var q_update_blk: C51QUpdateStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.NUM_ACTIONS,
        Self.N_ATOMS, Self.Q_NET,
    ]
    var polyak_blk: SinglePolyakStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.Q_NET,
    ]

    var state: TrainerState[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]

    # Target distribution scratch — replaces `state.mb_y`'s [B,1] role.
    var _mb_m: Scratch["mb_m", Self.BATCH * Self.N_ATOMS]

    # Action-selection scratch.
    var _ob1: Scratch["ob1", Self.OBS_DIM, True]
    var _q_logits: Scratch["q_logits", Self.NUM_ACTIONS * Self.N_ATOMS, True]
    var _q_batch: Scratch["q_batch", Self.NUM_ACTIONS * Self.N_ATOMS]

    var tracker: EpisodeTracker
    var ctx: Optional[DeviceContext]

    var epsilon: Scalar[DT]
    var epsilon_decay: Scalar[DT]
    var epsilon_min: Scalar[DT]
    var learning_starts: Int

    var _action_list: List[Scalar[DT]]

    var _loss_accum: Scalar[DT]
    var _update_count: Int
    # Never reset by `flush_*` — emitted as `train_steps` so the
    # downstream monitor can plot cumulative updates over time.
    var _total_train_steps: Int
    var timer: Timer

    def __init__(out self):
        self.pair = OnlineTargetPair[Self.Q_NET]()
        self.q_opt = Adam()
        self.sample_blk = Self.SAMPLE()
        self.target_y_blk = C51TargetYStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.NUM_ACTIONS,
            Self.N_ATOMS, Self.Q_NET, Self.DOUBLE,
        ]()
        self.q_update_blk = C51QUpdateStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.NUM_ACTIONS,
            Self.N_ATOMS, Self.Q_NET,
        ]()
        self.polyak_blk = SinglePolyakStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.Q_NET,
        ]()
        self.state = TrainerState[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ]()
        self._mb_m = Scratch["mb_m", Self.BATCH * Self.N_ATOMS]()
        self._ob1 = Scratch["ob1", Self.OBS_DIM, True]()
        self._q_logits = Scratch[
            "q_logits", Self.NUM_ACTIONS * Self.N_ATOMS, True,
        ]()
        self._q_batch = Scratch[
            "q_batch", Self.NUM_ACTIONS * Self.N_ATOMS,
        ]()
        self.tracker = EpisodeTracker(
            window=List[Scalar[DT]](),
            window_size=0,
            idx=0,
            current_return=Scalar[DT](0.0),
            ep_count=0,
        )
        self.ctx = None
        self.epsilon = Scalar[DT](1.0)
        self.epsilon_decay = Scalar[DT](0.995)
        self.epsilon_min = Scalar[DT](0.01)
        self.learning_starts = 1_000
        self._action_list = List[Scalar[DT]](length=1, fill=Scalar[DT](0.0))
        self._loss_accum = Scalar[DT](0.0)
        self._update_count = 0
        self._total_train_steps = 0
        self.timer = Timer.new()

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        lr: Scalar[DT] = Scalar[DT](1e-4),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        tau: Scalar[DT] = Scalar[DT](0.005),
        epsilon: Scalar[DT] = Scalar[DT](1.0),
        epsilon_decay: Scalar[DT] = Scalar[DT](0.995),
        epsilon_min: Scalar[DT] = Scalar[DT](0.05),
        learning_starts: Int = 1_000,
        target_update_freq: Int = 500,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = Scalar[DT](0.0),
        max_grad_norm: Scalar[DT] = Scalar[DT](0.0),
        per_alpha: Scalar[DT] = Scalar[DT](0.6),
        per_beta: Scalar[DT] = Scalar[DT](0.4),
        per_epsilon: Scalar[DT] = Scalar[DT](1e-6),
        nstep: Int = 1,
        v_min: Scalar[DT] = Scalar[DT](-10.0),
        v_max: Scalar[DT] = Scalar[DT](10.0),
    ) raises -> Self:
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "C51Trainer: train_target must be 'cpu' or 'gpu'"
        comptime assert (
            Self.ACT_DIM == 1
        ), "C51Trainer: SAMPLE.ACT must be 1 (discrete action index)"
        comptime assert (
            Self.Q_NET.IN_DIMS[0] == Self.OBS_DIM
        ), "C51Trainer: Q_NET.IN_DIM must equal SAMPLE.OBS"
        comptime assert (
            Self.Q_NET.OUT_DIM == Self.NUM_ACTIONS * Self.N_ATOMS
        ), "C51Trainer: Q_NET.OUT_DIM must equal NUM_ACTIONS · N_ATOMS"

        var t = Self()
        t.ctx = ctx
        t.epsilon = epsilon
        t.epsilon_decay = epsilon_decay
        t.epsilon_min = epsilon_min
        t.learning_starts = learning_starts

        t.pair = OnlineTargetPair[Self.Q_NET].make[
            target=Self.train_target, INIT=Xavier,
        ](ctx=ctx)
        t.q_opt = Adam.make[target=Self.train_target, M=Self.Q_NET](
            t.pair.online, ctx=ctx,
        )
        t.q_opt.lr = lr
        t.q_opt.max_grad_norm = max_grad_norm

        t.target_y_blk = C51TargetYStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.NUM_ACTIONS,
            Self.N_ATOMS, Self.Q_NET, Self.DOUBLE,
        ].make[Self.train_target](
            gamma=gamma, nstep=nstep,
            v_min=v_min, v_max=v_max, ctx=ctx,
        )

        t.q_update_blk = C51QUpdateStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.NUM_ACTIONS,
            Self.N_ATOMS, Self.Q_NET,
        ].make[Self.train_target](ctx=ctx)

        t.polyak_blk = SinglePolyakStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.Q_NET,
        ].make(tau=tau, update_every=target_update_freq)

        t.state = TrainerState[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ].make[Self.train_target](ctx=ctx)

        t.tracker = EpisodeTracker.new(
            window_size=window_size,
            initial_fill=initial_episode_fill,
        )

        init_scratch_auto[Self, target=Self.train_target](t, ctx)

        t.sample_blk.configure_per(
            alpha=per_alpha, beta=per_beta, epsilon=per_epsilon,
        )
        t.sample_blk.configure_gamma(gamma)
        t.sample_blk.setup(learning_starts, ctx=ctx)

        t.timer.add_section("sample")
        t.timer.add_section("target_y")
        t.timer.add_section("critic")
        t.timer.add_section("polyak")
        return t^

    # ─── Train step ──────────────────────────────────────────────────

    def _train_step_impl[POLICY: AMPPolicy = NoAMP](
        mut self, step_idx: Int,
    ) raises -> Bool:
        from std.time import perf_counter_ns

        self.state.step_idx = step_idx
        self.state.did_step = True

        var t_sample = perf_counter_ns()
        self.sample_blk.step(self.state)
        if not self.state.did_step:
            return False
        self.timer.accumulate(Self._T_SAMPLE, t_sample)

        var t_ty = perf_counter_ns()
        var m_ptr = self._mb_m.target_ptr[Self.train_target]()
        self.target_y_blk.step[Self.train_target, POLICY](
            self.state, self.pair.target_net, self.pair.online, m_ptr,
        )
        self.timer.accumulate(Self._T_TARGET_Y, t_ty)

        var t_crit = perf_counter_ns()
        self.q_update_blk.step[Self.train_target, POLICY](
            self.state, self.pair.online, self.q_opt, m_ptr,
        )
        self.timer.accumulate(Self._T_CRITIC, t_crit)

        var t_poly = perf_counter_ns()
        self.polyak_blk.step[Self.train_target](self.state, self.pair)
        self.timer.accumulate(Self._T_POLYAK, t_poly)

        self.sample_blk.update_priorities(self.state)

        self._loss_accum += self.state.critic_loss
        self._update_count += 1
        self._total_train_steps += 1
        return True

    def train_step(mut self, step_idx: Int) raises -> Bool:
        return self._train_step_impl[NoAMP](step_idx)

    def set_beta(mut self, beta: Scalar[DT]):
        self.sample_blk.set_beta(beta)

    # ─── Record ──────────────────────────────────────────────────────

    def record(
        mut self,
        ref obs: List[Scalar[DT]],
        action_idx: Int,
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        self.tracker.add_reward(reward)
        self._action_list[0] = Scalar[DT](action_idx)
        self.sample_blk.add(
            obs, self._action_list, reward, next_obs, done, ctx=self.ctx,
        )

    def record_batch_cpu[
        N_ENVS: Int,
    ](
        mut self,
        prev_obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        reward_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        next_obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        done_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        comptime assert (
            Self.train_target == "cpu"
        ), "record_batch_cpu: trainer must be cpu"
        comptime OBS = Self.OBS_DIM
        var obs_lane = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
        var act_lane = List[Scalar[DT]](length=1, fill=Scalar[DT](0.0))
        var nxt_lane = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
        for env_idx in range(N_ENVS):
            for d in range(OBS):
                obs_lane[d] = prev_obs_ptr[env_idx * OBS + d]
                nxt_lane[d] = next_obs_ptr[env_idx * OBS + d]
            act_lane[0] = action_ptr[env_idx]
            self.sample_blk.add(
                obs_lane, act_lane, reward_ptr[env_idx], nxt_lane,
                done_ptr[env_idx], ctx=self.ctx,
            )

    # ─── Action selection (expected-Q argmax over softmax·z) ─────────

    def _expected_q_argmax(
        mut self, q_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) -> Int:
        """Compute argmax_a Σ_k softmax(logits[a])_k · z_k from a flat
        NUM_ACTIONS · N_ATOMS logits row (single sample)."""
        comptime NA = Self.NUM_ACTIONS
        comptime NK = Self.N_ATOMS
        var z_p = self.target_y_blk.z_ptr()
        var best_a: Int = 0
        var best_eq: Scalar[DT] = Scalar[DT](0.0)
        for a in range(NA):
            var base = a * NK
            var mx = q_p[base]
            for i in range(1, NK):
                if q_p[base + i] > mx:
                    mx = q_p[base + i]
            var s_exp: Scalar[DT] = Scalar[DT](0.0)
            for i in range(NK):
                s_exp = s_exp + fexp(q_p[base + i] - mx)
            var eq: Scalar[DT] = Scalar[DT](0.0)
            for i in range(NK):
                var p = fexp(q_p[base + i] - mx) / s_exp
                eq = eq + p * z_p[i]
            if a == 0 or eq > best_eq:
                best_eq = eq
                best_a = a
        return best_a

    def select_action_batched[
        N_ENVS: Int,
    ](
        mut self,
        obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        step_idx: Int,
    ) raises:
        comptime NA = Self.NUM_ACTIONS
        comptime NK = Self.N_ATOMS
        comptime OBS = Self.OBS_DIM

        # Warmup: uniform random action.
        if step_idx < self.learning_starts:
            comptime if Self.train_target == "cpu":
                for i in range(N_ENVS):
                    var r = random_float64()
                    action_ptr[i] = Scalar[DT](Int(r * Float64(NA)))
            else:
                comptime assert (
                    N_ENVS == 1
                ), "GPU C51 select_action_batched warmup: N_ENVS>1 not yet supported"
                var ctx = self.ctx.value()
                var r = random_float64()
                self._q_logits.cpu_ptr()[0] = Scalar[DT](
                    Int(r * Float64(NA))
                )
                var action_dev = DeviceBuffer[DT](
                    ctx, action_ptr, 1, owning=False,
                )
                ctx.enqueue_copy(action_dev, self._q_logits.cpu_ptr())
            return

        comptime if Self.train_target == "cpu":
            # Policy: batched forward then per-env expected-Q argmax.
            var q_buf = List[Scalar[DT]](
                length=N_ENVS * NA * NK, fill=Scalar[DT](0.0),
            )
            var q_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                q_buf.unsafe_ptr()
            )
            var obs_t = TileTensor(obs_ptr, row_major[N_ENVS, OBS]())
            var q_t = TileTensor(q_ptr, row_major[N_ENVS, NA * NK]())
            self.pair.online.forward[Self.train_target, N_ENVS](
                obs_t, output=q_t,
            )
            for i in range(N_ENVS):
                var r = random_float64()
                if r < Float64(self.epsilon):
                    action_ptr[i] = Scalar[DT](
                        Int(random_float64() * Float64(NA))
                    )
                else:
                    action_ptr[i] = Scalar[DT](
                        self._expected_q_argmax(q_ptr + i * NA * NK)
                    )
        else:
            comptime assert (
                N_ENVS == 1
            ), "GPU C51 select_action_batched: N_ENVS>1 not yet supported"
            var ctx = self.ctx.value()
            var obs_t = TileTensor(obs_ptr, row_major[1, OBS]())
            var q_t = TileTensor(
                self._q_logits.dev_ptr(), row_major[1, NA * NK](),
            )
            self.pair.online.forward[Self.train_target, 1](
                obs_t, output=q_t,
            )
            ctx.enqueue_copy(
                self._q_logits.cpu_ptr(), self._q_logits.dev.value(),
            )
            ctx.synchronize()
            var r = random_float64()
            var act: Scalar[DT]
            if r < Float64(self.epsilon):
                act = Scalar[DT](Int(random_float64() * Float64(NA)))
            else:
                act = Scalar[DT](
                    self._expected_q_argmax(self._q_logits.cpu_ptr())
                )
            self._q_logits.cpu_ptr()[0] = act
            var action_dev = DeviceBuffer[DT](
                ctx, action_ptr, 1, owning=False,
            )
            ctx.enqueue_copy(action_dev, self._q_logits.cpu_ptr())

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
    ) raises -> Int:
        comptime NA = Self.NUM_ACTIONS
        comptime NK = Self.N_ATOMS
        comptime OBS = Self.OBS_DIM
        var ob1_p = self._ob1.cpu_ptr()
        for d in range(OBS):
            ob1_p[d] = obs[d]
        comptime if Self.train_target == "cpu":
            var ob1_t = TileTensor(ob1_p, row_major[1, OBS]())
            var q_p = self._q_logits.cpu_ptr()
            var q_t = TileTensor(q_p, row_major[1, NA * NK]())
            self.pair.online.forward[Self.train_target, 1](
                ob1_t, output=q_t,
            )
            return self._expected_q_argmax(q_p)
        else:
            var ctx = self.ctx.value()
            ctx.enqueue_copy(self._ob1.dev.value(), ob1_p)
            var ob1_t = TileTensor(
                self._ob1.dev_ptr(), row_major[1, OBS](),
            )
            var q_t = TileTensor(
                self._q_logits.dev_ptr(), row_major[1, NA * NK](),
            )
            self.pair.online.forward[Self.train_target, 1](
                ob1_t, output=q_t,
            )
            ctx.enqueue_copy(
                self._q_logits.cpu_ptr(), self._q_logits.dev.value(),
            )
            ctx.synchronize()
            return self._expected_q_argmax(self._q_logits.cpu_ptr())

    # ─── Episode tracking ────────────────────────────────────────────

    def end_episode(mut self):
        self.tracker.end_episode()
        self.epsilon = self.epsilon * self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    def add_complete_return(mut self, ret: Scalar[DT]):
        self.tracker.add_complete_return(ret)

    def mean_return(self) -> Scalar[DT]:
        return self.tracker.mean_return()

    def ep_count(self) -> Int:
        return self.tracker.ep_count

    # ─── Logging ─────────────────────────────────────────────────────

    def flush_train_log(
        mut self,
    ) -> Tuple[Scalar[DT], Scalar[DT], Int]:
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var out = (
            self._loss_accum * inv,
            self.epsilon,
            self._update_count,
        )
        self._loss_accum = Scalar[DT](0.0)
        self._update_count = 0
        return out

    def total_train_steps(self) -> Int:
        """Cumulative training updates since trainer was made. Not reset
        by `flush_*`."""
        return self._total_train_steps

    def flush_metrics[
        L: Logger = NoOpLogger
    ](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> C51Metrics:
        """Drain accumulators into a C51Metrics bundle. If a logger
        pointer is wired, also emit one log_scalar per metric field.
        Resets per-chunk accumulators on every call; the cumulative
        `_total_train_steps` counter is NOT reset."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var bundle = C51Metrics(
            loss=LogScalar[DT](self._loss_accum * inv),
            epsilon=LogScalar[DT](self.epsilon),
            train_steps=LogScalar[DT](Scalar[DT](self._total_train_steps)),
            n_updates=LogScalar[DT](Scalar[DT](self._update_count)),
        )
        self._loss_accum = Scalar[DT](0.0)
        self._update_count = 0
        if Bool(logger):
            log_bundle(logger.value()[], bundle, step)
        return bundle^

    # ─── Trait-uniform cadence hooks (consumed by the driver) ─────────

    def flush_metrics_through_logger[L: Logger](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        """Trait-uniform passthrough: drains the C51 metric accumulators
        through `flush_metrics` and discards the typed bundle. The
        driver calls this at the user's `diag_every` cadence so no
        chunking is needed."""
        _ = self.flush_metrics[L](logger, step)

    def save_state(mut self, path: String) raises:
        """One-file v2 checkpoint of the C51 module + optimizer.
        Sections: `q_net.*`, `q_opt.*`. Overwrites `path`. CPU-only;
        GPU save/load would need device→host sync first. Replay buffer
        + episode tracker NOT included."""
        var body = String("")
        comptime if Self.train_target == "cpu":
            save_state_v2_body(self.pair.online, body, "q_net")
            save_optimizer_v2_body(self.q_opt, body, "q_opt")
        else:
            var c = self.ctx.value()
            save_state_v2_body_gpu(self.pair.online, body, "q_net", c)
            save_optimizer_v2_body_gpu(self.q_opt, body, "q_opt")
        var content = String("nn2-ckpt v2\n") + body
        with open(path, "w") as f:
            f.write(content)

    def load_state(mut self, path: String) raises:
        """Inverse of `save_state`. Target net is hard-copied from the
        online net after the online params are restored. On GPU the
        device params + Adam moments are restored via host staging; the
        on-disk format is byte-identical to the CPU path."""
        var content = read_file_v2(path)
        var lines = split_lines_v2(content)
        expect_v2_header(lines)
        var idx: Int = 1
        comptime if Self.train_target == "cpu":
            load_state_v2_body(self.pair.online, lines, idx, "q_net")
            load_optimizer_v2_body(self.q_opt, lines, idx, "q_opt")
        else:
            var c = self.ctx.value()
            load_state_v2_body_gpu(self.pair.online, lines, idx, "q_net", c)
            load_optimizer_v2_body_gpu(self.q_opt, lines, idx, "q_opt")
        hard_copy_params[Self.train_target, M=Self.Q_NET](
            self.pair.online, self.pair.target_net, self.ctx,
        )

    def flush_timer_log(mut self) -> String:
        var report = self.timer.format_report()
        self.timer.reset()
        return report

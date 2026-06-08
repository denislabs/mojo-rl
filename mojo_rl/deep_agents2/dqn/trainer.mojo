"""DQNTrainer — DQN family trainer (CPU/GPU × uniform replay).

Discrete-action off-policy trainer for DQN and variants. Pipeline body
mirrors `sac/trainer.mojo`'s SAC pipeline: each train step is a short
sequence of `<block>.step[target, POLICY]` calls against a shared
`TrainerState`, with target-Y and gather/scatter running on-device.

Block decomposition:
  1. `sample_blk: SAMPLE` (SampleBlock trait — uniform / PER / N-step)
  2. `target_y_blk: DQNTargetYStep` (forward-only, owns `Q_target.forward
     → ReduceMax → finalize fuse`; Double branch swaps in argmax+gather)
  3. `q_update_blk: DQNQUpdateStep` (owns gather+MSE+scatter+Q.vjp)
  4. `polyak_blk: SinglePolyakStep` (soft τ-update OR hard copy every N)

Driver-trait conformance: `OffPolicyDiscreteAgent` via `train_step`,
`select_action_batched`, `select_greedy_action`, `record`,
`record_batch_cpu`, `add_complete_return`.

CPU + GPU bodies share one `_train_step_impl[POLICY]` — `train_target`
is a struct comptime param threaded into every block. No D2H/H2D in
the train step on GPU (all gather/scatter is on-device).
"""

from std.random import random_float64
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import mptr
from mojo_rl.nn2.core import Module
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn2.core.checkpoint import (
    save_state_v2_body, load_state_v2_body,
    save_state_v2_body_gpu, load_state_v2_body_gpu,
)
from mojo_rl.nn2.core.log_bundle import log_bundle
from mojo_rl.nn2.core.map_params import hard_copy_params
from mojo_rl.nn2.core.save_scalar import SaveScalar
from mojo_rl.nn2.core.metric import LogScalar
from mojo_rl.nn2.core.scratch import Scratch
from mojo_rl.nn2.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.training.timer import Timer
from ..core.checkpoint_helpers import (
    save_optimizer_v2_body, load_optimizer_v2_body,
    save_optimizer_v2_body_gpu, load_optimizer_v2_body_gpu,
    save_counter_v2_body, load_counter_v2_body,
    split_lines_v2, read_file_v2, expect_v2_header,
)
from ..core.online_target_pair import OnlineTargetPair
from ..training.episode_tracker import EpisodeTracker
from ..training.device_mean_accum import DeviceMeanAccum
from ..training.trainer_block import TrainerState
from ..training.driver_offpolicy_discrete import OffPolicyDiscreteAgent
from ..training.blocks import SampleBlock, SinglePolyakStep
from .blocks.target_y_step import DQNTargetYStep
from .blocks.q_update_step import DQNQUpdateStep
from .metrics import DQNMetrics


struct DQNTrainer[
    train_target: StaticString,
    SAMPLE: SampleBlock,
    Q_NET: Module,
    DOUBLE: Bool = False,
](OffPolicyDiscreteAgent):
    """Dimensions derived from SAMPLE (OBS, ACT=1, BATCH) and Q_NET
    (OUT_DIM = NUM_ACTIONS). The sample block stores discrete action
    indices as a single Scalar[DT] in ACT=1."""

    comptime OBS_DIM: Int = Self.SAMPLE.OBS
    comptime ACT_DIM: Int = Self.SAMPLE.ACT
    comptime BATCH: Int = Self.SAMPLE.BATCH
    comptime NUM_ACTIONS: Int = Self.Q_NET.OUT_DIM

    comptime AGENT_TRAIN_TARGET: StaticString = Self.train_target
    comptime AGENT_OBS_DIM: Int = Self.OBS_DIM
    comptime AGENT_NUM_ACTIONS: Int = Self.NUM_ACTIONS

    comptime _T_SAMPLE = 0
    comptime _T_TARGET_Y = 1
    comptime _T_CRITIC = 2
    comptime _T_POLYAK = 3
    comptime _T_DIAG = 4

    var pair: OnlineTargetPair[Self.Q_NET]
    var q_opt: Adam
    var sample_blk: Self.SAMPLE
    var target_y_blk: DQNTargetYStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.NUM_ACTIONS,
        Self.Q_NET, Self.DOUBLE,
    ]
    var q_update_blk: DQNQUpdateStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.NUM_ACTIONS, Self.Q_NET,
    ]
    var polyak_blk: SinglePolyakStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.Q_NET,
    ]

    var state: TrainerState[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]

    # Action-selection scratch (single-env greedy + GPU N=1 path).
    var _ob1: Scratch["ob1", Self.OBS_DIM, True]
    var _q_select: Scratch["q_select", Self.NUM_ACTIONS, True]

    # Lazily-sized scratch for the GPU batched action path (N_ENVS is a
    # method-comptime param, unknown at construction). Allocated once on the
    # first `select_action_batched[N_ENVS]` call and reused — avoids the
    # per-step `enqueue_create_buffer` that explodes disk on NVIDIA. Mirrors
    # the proven C51 pattern (`c51/trainer.mojo::_ensure_batch_scratch`).
    var _batch_q_dev: Optional[DeviceBuffer[DT]]
    var _batch_q_host: List[Scalar[DT]]
    var _batch_act_host: List[Scalar[DT]]
    var _batch_n: Int

    var tracker: EpisodeTracker
    var ctx: Optional[DeviceContext]

    var epsilon: Scalar[DT]
    var epsilon_decay: Scalar[DT]
    var epsilon_min: Scalar[DT]
    var learning_starts: Int

    var _action_list: List[Scalar[DT]]

    var _loss_accum: Scalar[DT]
    # Per-batch diagnostic accumulators (CPU-only diag walk; see
    # `_train_step_impl`). Drained + reset by `flush_metrics`.
    var _q_accum: Scalar[DT]
    var _target_accum: Scalar[DT]
    var _td_error_accum: Scalar[DT]
    var _reward_accum: Scalar[DT]
    var _done_accum: Scalar[DT]
    # GPU device-resident mirrors (CPU keeps the host scalars above).
    var _q_mean_dev: DeviceMeanAccum
    var _target_mean_dev: DeviceMeanAccum
    var _td_error_mean_dev: DeviceMeanAccum
    var _reward_mean_dev: DeviceMeanAccum
    var _done_mean_dev: DeviceMeanAccum
    var _update_count: Int
    # Never reset by `flush_*` — emitted as `train_steps` so the
    # downstream monitor can plot cumulative updates over time.
    var _total_train_steps: Int
    var timer: Timer

    def __init__(out self):
        self.pair = OnlineTargetPair[Self.Q_NET]()
        self.q_opt = Adam()
        self.sample_blk = Self.SAMPLE()
        self.target_y_blk = DQNTargetYStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.NUM_ACTIONS,
            Self.Q_NET, Self.DOUBLE,
        ]()
        self.q_update_blk = DQNQUpdateStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.NUM_ACTIONS,
            Self.Q_NET,
        ]()
        self.polyak_blk = SinglePolyakStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.Q_NET,
        ]()
        self.state = TrainerState[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ]()
        self._ob1 = Scratch["ob1", Self.OBS_DIM, True]()
        self._q_select = Scratch["q_select", Self.NUM_ACTIONS, True]()
        self._batch_q_dev = None
        self._batch_q_host = List[Scalar[DT]]()
        self._batch_act_host = List[Scalar[DT]]()
        self._batch_n = 0
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
        self._q_accum = Scalar[DT](0.0)
        self._target_accum = Scalar[DT](0.0)
        self._td_error_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        self._q_mean_dev = DeviceMeanAccum()
        self._target_mean_dev = DeviceMeanAccum()
        self._td_error_mean_dev = DeviceMeanAccum()
        self._reward_mean_dev = DeviceMeanAccum()
        self._done_mean_dev = DeviceMeanAccum()
        self._update_count = 0
        self._total_train_steps = 0
        self.timer = Timer.new()

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        lr: Scalar[DT] = Scalar[DT](1e-3),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        tau: Scalar[DT] = Scalar[DT](0.005),
        epsilon: Scalar[DT] = Scalar[DT](1.0),
        epsilon_decay: Scalar[DT] = Scalar[DT](0.995),
        epsilon_min: Scalar[DT] = Scalar[DT](0.01),
        learning_starts: Int = 1_000,
        target_update_freq: Int = 0,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = Scalar[DT](0.0),
        max_grad_norm: Scalar[DT] = Scalar[DT](0.0),
        per_alpha: Scalar[DT] = Scalar[DT](0.6),
        per_beta: Scalar[DT] = Scalar[DT](0.4),
        per_epsilon: Scalar[DT] = Scalar[DT](1e-6),
        nstep: Int = 1,
    ) raises -> Self:
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "DQNTrainer: target must be 'cpu' or 'gpu'"
        comptime assert (
            Self.ACT_DIM == 1
        ), "DQNTrainer: SAMPLE.ACT must be 1 (discrete action index)"
        comptime assert (
            Self.Q_NET.IN_DIMS[0] == Self.OBS_DIM
        ), "DQNTrainer: Q_NET.IN_DIM must equal SAMPLE.OBS"
        comptime if Self.train_target == "gpu":
            if not ctx:
                raise Error("DQNTrainer.make[target='gpu']: ctx required")

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

        t.target_y_blk = DQNTargetYStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.NUM_ACTIONS,
            Self.Q_NET, Self.DOUBLE,
        ].make[Self.train_target](gamma=gamma, nstep=nstep, ctx=ctx)

        t.q_update_blk = DQNQUpdateStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.NUM_ACTIONS,
            Self.Q_NET,
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

        comptime if Self.train_target == "gpu":
            # Device-resident mean accumulators for the GPU diag path.
            t._q_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._target_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._td_error_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._reward_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._done_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)

        # PER hyperparameter wiring: no-op default for uniform blocks.
        t.sample_blk.configure_per(
            alpha=per_alpha, beta=per_beta, epsilon=per_epsilon,
        )
        # N-step γ alignment: no-op default for non-nstep blocks.
        t.sample_blk.configure_gamma(gamma)
        t.sample_blk.setup(learning_starts, ctx=ctx)

        t.timer.add_section("sample")
        t.timer.add_section("target_y")
        t.timer.add_section("critic")
        t.timer.add_section("polyak")
        t.timer.add_section("diag")
        return t^

    # ─── Train step ──────────────────────────────────────────────────

    def _train_step_impl[POLICY: AMPPolicy = NoAMP](
        mut self, step_idx: Int,
    ) raises -> Bool:
        from std.time import perf_counter_ns

        self.state.step_idx = step_idx
        self.state.did_step = True
        comptime if Self.train_target == "gpu":
            self.state.ctx = self.ctx

        # 1. Sample.
        var t_sample = perf_counter_ns()
        self.sample_blk.step(self.state)
        if not self.state.did_step:
            return False
        self.timer.accumulate(Self._T_SAMPLE, t_sample)

        # 2. Target-Y.
        var t_ty = perf_counter_ns()
        self.target_y_blk.step[Self.train_target, POLICY](
            self.state, self.pair.target_net, self.pair.online,
        )
        self.timer.accumulate(Self._T_TARGET_Y, t_ty)

        # 3. Q-update (gather → MSE → scatter → Q.vjp → opt.step).
        var t_crit = perf_counter_ns()
        self.q_update_blk.step[Self.train_target, POLICY](
            self.state, self.pair.online, self.q_opt,
        )
        self.timer.accumulate(Self._T_CRITIC, t_crit)

        # 4. Polyak / hard-copy.
        var t_poly = perf_counter_ns()
        self.polyak_blk.step[Self.train_target](self.state, self.pair)
        self.timer.accumulate(Self._T_POLYAK, t_poly)

        # PER tail (no-op for uniform blocks).
        self.sample_blk.update_priorities(self.state)

        self._loss_accum += self.state.critic_loss

        # Per-batch diagnostic means (CPU-only — GPU train_target would need
        # D2H copies of the mb_* scratches; deferred, mirroring SAC). Q(s,a)
        # is the gathered Q at the taken action, populated by `q_update_blk`;
        # target/reward/done live in the shared TrainerState scratches.
        # `mean_td_error` = mean |Q − y|, the Bellman residual magnitude.
        var t_diag = perf_counter_ns()
        comptime if Self.train_target == "cpu":
            var inv_b: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](Self.BATCH)
            var q_p = self.q_update_blk.inner._mb_q_gath.target_ptr["cpu"]()
            var y_p = self.state.mb_y.target_ptr["cpu"]()
            var r_p = self.state.mb_r.target_ptr["cpu"]()
            var d_p = self.state.mb_d.target_ptr["cpu"]()
            var sum_q: Scalar[DT] = 0.0
            var sum_y: Scalar[DT] = 0.0
            var sum_te: Scalar[DT] = 0.0
            var sum_r: Scalar[DT] = 0.0
            var sum_d: Scalar[DT] = 0.0
            for i in range(Self.BATCH):
                var qi = q_p[i]
                var yi = y_p[i]
                var te = qi - yi
                sum_q += qi
                sum_y += yi
                sum_te += te if te >= Scalar[DT](0.0) else -te
                sum_r += r_p[i]
                sum_d += d_p[i]
            self._q_accum += sum_q * inv_b
            self._target_accum += sum_y * inv_b
            self._td_error_accum += sum_te * inv_b
            self._reward_accum += sum_r * inv_b
            self._done_accum += sum_d * inv_b
        else:
            var q_ptr = self.q_update_blk.inner._mb_q_gath.target_ptr["gpu"]()
            var y_ptr = self.state.mb_y.target_ptr["gpu"]()
            var r_ptr = self.state.mb_r.target_ptr["gpu"]()
            var d_ptr = self.state.mb_d.target_ptr["gpu"]()
            self._q_mean_dev.accumulate_gpu[Self.BATCH](q_ptr)
            self._target_mean_dev.accumulate_gpu[Self.BATCH](y_ptr)
            self._td_error_mean_dev.accumulate_gpu_abs_diff[Self.BATCH](
                q_ptr, y_ptr
            )
            self._reward_mean_dev.accumulate_gpu[Self.BATCH](r_ptr)
            self._done_mean_dev.accumulate_gpu[Self.BATCH](d_ptr)
        self.timer.accumulate(Self._T_DIAG, t_diag)

        self._update_count += 1
        self._total_train_steps += 1
        return True

    def train_step(mut self, step_idx: Int) raises -> Bool:
        return self._train_step_impl[NoAMP](step_idx)

    def set_beta(mut self, beta: Scalar[DT]):
        """PER IS-β anneal hook (callers ramp 0.4 → 1.0). No-op for
        uniform sample blocks."""
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

    # ─── Action selection ────────────────────────────────────────────

    def _ensure_batch_scratch[
        N_ENVS: Int
    ](mut self, ctx: DeviceContext) raises:
        """Lazily (re)allocate the GPU batched-action scratch for N_ENVS:
        a device `[N_ENVS·NA]` Q buffer + host mirrors for the D2H Q
        readback and the H2D action indices. One allocation per distinct
        N_ENVS (cached via `_batch_n`). Mirrors
        `c51/trainer.mojo::_ensure_batch_scratch`."""
        comptime NA = Self.NUM_ACTIONS
        if self._batch_n == N_ENVS:
            return
        self._batch_q_dev = Optional(
            ctx.enqueue_create_buffer[DT](N_ENVS * NA)
        )
        self._batch_q_host = List[Scalar[DT]](
            length=N_ENVS * NA, fill=Scalar[DT](0.0)
        )
        self._batch_act_host = List[Scalar[DT]](
            length=N_ENVS, fill=Scalar[DT](0.0)
        )
        self._batch_n = N_ENVS

    def _q_argmax(
        self, q_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) -> Int:
        """argmax_a q_p[a] over a flat NUM_ACTIONS row (single sample)."""
        comptime NA = Self.NUM_ACTIONS
        var best_a = 0
        var best_q = q_p[0]
        for a in range(1, NA):
            if q_p[a] > best_q:
                best_q = q_p[a]
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
        comptime OBS = Self.OBS_DIM

        # Warmup path (random action). CPU writes action_ptr directly;
        # GPU stages through a host buffer then H2D (action_ptr is
        # device-side, a CPU write to it is UB / Metal crash).
        if step_idx < self.learning_starts:
            comptime if Self.train_target == "cpu":
                for i in range(N_ENVS):
                    var r = random_float64()
                    action_ptr[i] = Scalar[DT](Int(r * Float64(NA)))
            else:
                # GPU warmup: draw N_ENVS uniform action indices on the host
                # and H2D them. At N_ENVS=1 this consumes exactly one
                # random_float64 draw, matching the legacy single-env path's
                # RNG order.
                var ctx = self.ctx.value()
                self._ensure_batch_scratch[N_ENVS](ctx)
                var act_h = self._batch_act_host.unsafe_ptr()
                for i in range(N_ENVS):
                    act_h[i] = Scalar[DT](Int(random_float64() * Float64(NA)))
                var action_dev = DeviceBuffer[DT](
                    ctx, action_ptr, N_ENVS, owning=False,
                )
                ctx.enqueue_copy(action_dev, act_h)
            return

        comptime if Self.train_target == "cpu":
            var q_buf = List[Scalar[DT]](
                length=N_ENVS * NA, fill=Scalar[DT](0.0),
            )
            var q_ptr = mptr(q_buf.unsafe_ptr())
            var obs_t = TileTensor(obs_ptr, row_major[N_ENVS, OBS]())
            var q_t = TileTensor(q_ptr, row_major[N_ENVS, NA]())
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
                    var best_a = 0
                    var best_q = q_ptr[i * NA]
                    for a in range(1, NA):
                        var q = q_ptr[i * NA + a]
                        if q > best_q:
                            best_q = q
                            best_a = a
                    action_ptr[i] = Scalar[DT](best_a)
        else:
            # GPU policy: ONE batched device forward over all N_ENVS obs,
            # then D2H the [N_ENVS, NA] Q values and run epsilon-greedy
            # argmax per env on the host — there is no batched device argmax
            # kernel, and the readback is tiny (N_ENVS·NA floats). At
            # N_ENVS=1 this reproduces the legacy single-env behaviour (one
            # forward, one host argmax) bit-for-bit, including RNG order.
            var ctx = self.ctx.value()
            self._ensure_batch_scratch[N_ENVS](ctx)
            var qdev = self._batch_q_dev.value()
            var qdev_ptr = mptr(qdev.unsafe_ptr())
            var obs_t = TileTensor(obs_ptr, row_major[N_ENVS, OBS]())
            var q_t = TileTensor(qdev_ptr, row_major[N_ENVS, NA]())
            self.pair.online.forward[Self.train_target, N_ENVS](
                obs_t, output=q_t,
            )
            var qh = self._batch_q_host.unsafe_ptr()
            ctx.enqueue_copy(qh, qdev)
            ctx.synchronize()
            var act_h = self._batch_act_host.unsafe_ptr()
            for i in range(N_ENVS):
                var r = random_float64()
                if r < Float64(self.epsilon):
                    act_h[i] = Scalar[DT](Int(random_float64() * Float64(NA)))
                else:
                    act_h[i] = Scalar[DT](self._q_argmax(qh + i * NA))
            var action_dev = DeviceBuffer[DT](
                ctx, action_ptr, N_ENVS, owning=False,
            )
            ctx.enqueue_copy(action_dev, act_h)

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
    ) raises -> Int:
        comptime NA = Self.NUM_ACTIONS
        comptime OBS = Self.OBS_DIM
        var ob1_cpu_p = self._ob1.cpu_ptr()
        for d in range(OBS):
            ob1_cpu_p[d] = obs[d]
        comptime if Self.train_target == "cpu":
            var ob1_t = TileTensor(ob1_cpu_p, row_major[1, OBS]())
            var q_p = self._q_select.cpu_ptr()
            var q_t = TileTensor(q_p, row_major[1, NA]())
            self.pair.online.forward[Self.train_target, 1](ob1_t, output=q_t)
            var best_a = 0
            var best_q = q_p[0]
            for a in range(1, NA):
                if q_p[a] > best_q:
                    best_q = q_p[a]
                    best_a = a
            return best_a
        else:
            var ctx = self.ctx.value()
            ctx.enqueue_copy(self._ob1.dev.value(), ob1_cpu_p)
            var ob1_t = TileTensor(self._ob1.dev_ptr(), row_major[1, OBS]())
            var q_t = TileTensor(self._q_select.dev_ptr(), row_major[1, NA]())
            self.pair.online.forward[Self.train_target, 1](ob1_t, output=q_t)
            ctx.enqueue_copy(
                self._q_select.cpu_ptr(), self._q_select.dev.value(),
            )
            ctx.synchronize()
            var q_p = self._q_select.cpu_ptr()
            var best_a = 0
            var best_q = q_p[0]
            for a in range(1, NA):
                if q_p[a] > best_q:
                    best_q = q_p[a]
                    best_a = a
            return best_a

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
        """Return (mean_loss, epsilon, n_updates) since last flush."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var out = (
            self._loss_accum * inv,
            self.epsilon,
            self._update_count,
        )
        self._loss_accum = Scalar[DT](0.0)
        # Keep the diagnostic accumulators in lock-step with the chunk
        # counter so a later `flush_metrics` doesn't average a multi-chunk
        # sum by a single-chunk `n`. (This legacy tuple API drops them.)
        self._q_accum = Scalar[DT](0.0)
        self._target_accum = Scalar[DT](0.0)
        self._td_error_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
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
    ) raises -> DQNMetrics:
        """Drain accumulators into a DQNMetrics bundle. If a logger
        pointer is wired, also emit one log_scalar per metric field.
        Resets per-chunk accumulators on every call; the cumulative
        `_total_train_steps` counter is NOT reset."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        # Per-batch diag means: device-resident on GPU (folded in by
        # `_train_step_impl`), host scalars on CPU.
        var q_mean: Scalar[DT]
        var target_mean: Scalar[DT]
        var td_error_mean: Scalar[DT]
        var reward_mean: Scalar[DT]
        var done_mean: Scalar[DT]
        comptime if Self.train_target == "gpu":
            q_mean = self._q_mean_dev.read["gpu"]()
            target_mean = self._target_mean_dev.read["gpu"]()
            td_error_mean = self._td_error_mean_dev.read["gpu"]()
            reward_mean = self._reward_mean_dev.read["gpu"]()
            done_mean = self._done_mean_dev.read["gpu"]()
        else:
            q_mean = self._q_accum * inv
            target_mean = self._target_accum * inv
            td_error_mean = self._td_error_accum * inv
            reward_mean = self._reward_accum * inv
            done_mean = self._done_accum * inv
        var bundle = DQNMetrics(
            loss=LogScalar[DT](self._loss_accum * inv),
            epsilon=LogScalar[DT](self.epsilon),
            mean_q=LogScalar[DT](q_mean),
            mean_target=LogScalar[DT](target_mean),
            mean_td_error=LogScalar[DT](td_error_mean),
            mean_reward=LogScalar[DT](reward_mean),
            mean_done=LogScalar[DT](done_mean),
            train_steps=LogScalar[DT](Scalar[DT](self._total_train_steps)),
            n_updates=LogScalar[DT](Scalar[DT](self._update_count)),
        )
        self._loss_accum = Scalar[DT](0.0)
        self._q_accum = Scalar[DT](0.0)
        self._target_accum = Scalar[DT](0.0)
        self._td_error_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        self._update_count = 0
        comptime if Self.train_target == "gpu":
            self._q_mean_dev.reset["gpu"]()
            self._target_mean_dev.reset["gpu"]()
            self._td_error_mean_dev.reset["gpu"]()
            self._reward_mean_dev.reset["gpu"]()
            self._done_mean_dev.reset["gpu"]()
        if Bool(logger):
            log_bundle(logger.value()[], bundle, step)
        return bundle^

    # ─── Trait-uniform cadence hooks (consumed by the driver) ─────────

    def flush_metrics_through_logger[L: Logger](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        """Trait-uniform passthrough: drains the DQN metric accumulators
        through `flush_metrics` and discards the typed bundle. The
        driver calls this at the user's `diag_every` cadence so no
        chunking is needed."""
        _ = self.flush_metrics[L](logger, step)

    def save_state(mut self, path: String) raises:
        """One-file v2 checkpoint of every DQN module + optimizer + the
        ε-greedy exploration state. Sections: `q_net.*`, `q_opt.*`, then
        `eps.{epsilon,epsilon_decay,epsilon_min}`. On GPU the device
        params + Adam moments are downloaded to host first; the on-disk
        format is byte-identical to the CPU path, so a GPU checkpoint
        loads on a CPU trainer (train-on-GPU → eval-on-CPU). The ε state
        is a host scalar in both targets, so it persists identically with
        no device sync — resume continues the decay schedule instead of
        restarting exploration at ε=1."""
        var body = String("")
        comptime if Self.train_target == "cpu":
            save_state_v2_body(self.pair.online, body, "q_net")
            save_optimizer_v2_body(self.q_opt, body, "q_opt")
        else:
            var c = self.ctx.value()
            save_state_v2_body_gpu(self.pair.online, body, "q_net", c)
            save_optimizer_v2_body_gpu(self.q_opt, body, "q_opt")
        SaveScalar[DT](self.epsilon).save(body, "eps.epsilon")
        SaveScalar[DT](self.epsilon_decay).save(body, "eps.epsilon_decay")
        SaveScalar[DT](self.epsilon_min).save(body, "eps.epsilon_min")
        save_counter_v2_body(self._total_train_steps, body, "_total_train_steps")
        var content = String("nn2-ckpt v2\n") + body
        with open(path, "w") as f:
            f.write(content)

    def load_state(mut self, path: String) raises:
        """Inverse of `save_state`. Target net is hard-copied from the
        online net after the online params are restored. On GPU the
        restored host values are uploaded to the device buffers."""
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
        # ε-greedy exploration state (host scalar in both targets).
        var eps_w = SaveScalar[DT](self.epsilon)
        eps_w.load(lines, idx, "eps.epsilon")
        self.epsilon = eps_w.v
        var eps_decay_w = SaveScalar[DT](self.epsilon_decay)
        eps_decay_w.load(lines, idx, "eps.epsilon_decay")
        self.epsilon_decay = eps_decay_w.v
        var eps_min_w = SaveScalar[DT](self.epsilon_min)
        eps_min_w.load(lines, idx, "eps.epsilon_min")
        self.epsilon_min = eps_min_w.v
        load_counter_v2_body(
            self._total_train_steps, lines, idx, "_total_train_steps"
        )
        hard_copy_params[Self.train_target, M=Self.Q_NET](
            self.pair.online, self.pair.target_net, self.ctx,
        )

    def flush_timer_log(mut self) -> String:
        var report = self.timer.format_report()
        self.timer.reset()
        return report

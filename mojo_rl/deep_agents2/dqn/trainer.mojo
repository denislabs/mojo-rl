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
from mojo_rl.nn2.core import Module
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn2.core.log_bundle import log_bundle
from mojo_rl.nn2.core.metric import LogScalar
from mojo_rl.nn2.core.scratch import Scratch
from mojo_rl.nn2.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.training.timer import Timer
from ..core.online_target_pair import OnlineTargetPair
from ..training.episode_tracker import EpisodeTracker
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

    var tracker: EpisodeTracker
    var ctx: Optional[DeviceContext]

    var epsilon: Scalar[DT]
    var epsilon_decay: Scalar[DT]
    var epsilon_min: Scalar[DT]
    var learning_starts: Int

    var _action_list: List[Scalar[DT]]

    var _loss_accum: Scalar[DT]
    var _update_count: Int
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
        ].make[Self.train_target](gamma=gamma, ctx=ctx)

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

        self._loss_accum += self.state.critic_loss
        self._update_count += 1
        return True

    def train_step(mut self, step_idx: Int) raises -> Bool:
        return self._train_step_impl[NoAMP](step_idx)

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
        # GPU stages through _q_select.cpu_ptr() then H2D (action_ptr is
        # device-side, a CPU write to it is UB / Metal crash).
        if step_idx < self.learning_starts:
            comptime if Self.train_target == "cpu":
                for i in range(N_ENVS):
                    var r = random_float64()
                    action_ptr[i] = Scalar[DT](Int(r * Float64(NA)))
            else:
                comptime assert (
                    N_ENVS == 1
                ), "GPU select_action_batched warmup: N_ENVS>1 not yet supported"
                var ctx = self.ctx.value()
                var r = random_float64()
                self._q_select.cpu_ptr()[0] = Scalar[DT](
                    Int(r * Float64(NA))
                )
                var action_dev = DeviceBuffer[DT](
                    ctx, action_ptr, 1, owning=False,
                )
                ctx.enqueue_copy(action_dev, self._q_select.cpu_ptr())
            return

        comptime if Self.train_target == "cpu":
            var q_buf = List[Scalar[DT]](
                length=N_ENVS * NA, fill=Scalar[DT](0.0),
            )
            var q_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                q_buf.unsafe_ptr()
            )
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
            comptime assert (
                N_ENVS == 1
            ), "GPU select_action_batched: N_ENVS>1 not yet supported"
            var ctx = self.ctx.value()
            var obs_t = TileTensor(obs_ptr, row_major[1, OBS]())
            var q_t = TileTensor(self._q_select.dev_ptr(), row_major[1, NA]())
            self.pair.online.forward[Self.train_target, 1](obs_t, output=q_t)
            ctx.enqueue_copy(
                self._q_select.cpu_ptr(), self._q_select.dev.value(),
            )
            ctx.synchronize()
            var qp = self._q_select.cpu_ptr()
            var r = random_float64()
            var act: Scalar[DT]
            if r < Float64(self.epsilon):
                act = Scalar[DT](Int(random_float64() * Float64(NA)))
            else:
                var best_a = 0
                var best_q = qp[0]
                for a in range(1, NA):
                    var q = qp[a]
                    if q > best_q:
                        best_q = q
                        best_a = a
                act = Scalar[DT](best_a)
            self._q_select.cpu_ptr()[0] = act
            var action_dev = DeviceBuffer[DT](
                ctx, action_ptr, 1, owning=False,
            )
            ctx.enqueue_copy(action_dev, self._q_select.cpu_ptr())

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
        self._update_count = 0
        return out

    def flush_metrics[
        L: Logger = NoOpLogger
    ](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> DQNMetrics:
        """Drain accumulators into a DQNMetrics bundle. If a logger
        pointer is wired, also emit one log_scalar per metric field.
        Resets accumulators on every call."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var bundle = DQNMetrics(
            loss=LogScalar[DT](self._loss_accum * inv),
            epsilon=LogScalar[DT](self.epsilon),
            n_updates=LogScalar[DT](Scalar[DT](self._update_count)),
        )
        self._loss_accum = Scalar[DT](0.0)
        self._update_count = 0
        if Bool(logger):
            log_bundle(logger.value()[], bundle, step)
        return bundle^

    def flush_timer_log(mut self) -> String:
        var report = self.timer.format_report()
        self.timer.reset()
        return report

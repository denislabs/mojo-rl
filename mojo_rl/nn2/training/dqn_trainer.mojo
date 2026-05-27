"""DQNTrainer — DQN family trainer: CPU/GPU × uniform/PER replay.

Discrete-action off-policy trainer for DQN and variants. Mirrors the
SACTrainer architecture but replaces the actor + twin critics + alpha
pipeline with a single Q-network (online + target) + epsilon-greedy.

  - `train_target: StaticString` — "cpu" or "gpu" — kernel dispatch.
  - `SAMPLE: SampleBlock` — replay-buffer-owning block (ACT=1 for
    discrete actions stored as Scalar[DT] indices).
  - `Q_NET: Module` — Q-network: IN_DIM=OBS_DIM, OUT_DIM=NUM_ACTIONS.
  - `DOUBLE: Bool` — False = standard DQN (max Q_target for target-Y),
    True = Double DQN (argmax from Q_online, evaluate with Q_target).

Train-step pipeline:
  1. sample_blk.step (sample minibatch)
  2. target-Y:
       standard: y = r + γ · max_a Q_target(s', a) · (1 − done)
       double:   y = r + γ · Q_target(s', argmax_a Q_online(s', a)) · (1 − done)
  3. critic update: MSE(Q_online(s)[a_taken], y) → backward → opt.step
  4. target update: Polyak soft update (or hard copy every N steps)

Driver-trait conformance: `OffPolicyDiscreteAgent` via `train_step`,
`select_action_batched`, `select_greedy_action`, `record`,
`record_batch_cpu`, `add_complete_return`.
"""

from std.math import exp as fexp
from std.random import random_float64
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from ..constants import DT
from ..core import Module
from ..core.scratch import Scratch
from ..core.scratch_walkers import init_scratch_auto
from ..core.online_target_pair import OnlineTargetPair
from ..initializer import Xavier
from ..optimizer.adam import Adam
from ..loss.mse import MSELoss
from .episode_tracker import EpisodeTracker
from .timer import Timer
from .trainer_block import TrainerState
from .driver_offpolicy_discrete import OffPolicyDiscreteAgent
from .blocks import SampleBlock


struct DQNTrainer[
    train_target: StaticString,
    SAMPLE: SampleBlock,
    Q_NET: Module,
    DOUBLE: Bool = False,
](OffPolicyDiscreteAgent):
    """Dimensions derived from SAMPLE (OBS, ACT=1, BATCH) and Q_NET
    (OUT_DIM = NUM_ACTIONS). The sample block stores discrete action
    indices as a single Scalar[DT] in ACT=1.

    When DOUBLE=True, target-Y uses the online net for action selection
    (argmax) and the target net for evaluation, reducing overestimation
    bias (van Hasselt et al. 2016)."""

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
    var mse_loss: MSELoss[1]

    var state: TrainerState[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]

    var _q_all: Scratch["q_all", Self.BATCH * Self.NUM_ACTIONS, True]
    var _q_gathered: Scratch["q_gathered", Self.BATCH, True]
    var _grad_q: Scratch["grad_q", Self.BATCH, True]
    var _grad_obs: Scratch["grad_obs", Self.BATCH * Self.OBS_DIM]

    var _ob1: Scratch["ob1", Self.OBS_DIM, True]
    var _q_select: Scratch["q_select", Self.NUM_ACTIONS, True]

    var tracker: EpisodeTracker
    var ctx: Optional[DeviceContext]

    var gamma: Scalar[DT]
    var tau: Scalar[DT]
    var epsilon: Scalar[DT]
    var epsilon_decay: Scalar[DT]
    var epsilon_min: Scalar[DT]
    var learning_starts: Int
    var target_update_freq: Int

    var _action_list: List[Scalar[DT]]

    var _loss_accum: Scalar[DT]
    var _update_count: Int
    var timer: Timer

    def __init__(out self):
        self.pair = OnlineTargetPair[Self.Q_NET]()
        self.q_opt = Adam()
        self.sample_blk = Self.SAMPLE()
        self.mse_loss = MSELoss[1]()
        self.state = TrainerState[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]()
        self._q_all = Scratch["q_all", Self.BATCH * Self.NUM_ACTIONS, True]()
        self._q_gathered = Scratch["q_gathered", Self.BATCH, True]()
        self._grad_q = Scratch["grad_q", Self.BATCH, True]()
        self._grad_obs = Scratch["grad_obs", Self.BATCH * Self.OBS_DIM]()
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
        self.gamma = Scalar[DT](0.99)
        self.tau = Scalar[DT](0.005)
        self.epsilon = Scalar[DT](1.0)
        self.epsilon_decay = Scalar[DT](0.995)
        self.epsilon_min = Scalar[DT](0.01)
        self.learning_starts = 1_000
        self.target_update_freq = 0
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
        t.gamma = gamma
        t.tau = tau
        t.epsilon = epsilon
        t.epsilon_decay = epsilon_decay
        t.epsilon_min = epsilon_min
        t.learning_starts = learning_starts
        t.target_update_freq = target_update_freq

        t.pair = OnlineTargetPair[Self.Q_NET].make[
            target=Self.train_target, INIT=Xavier
        ](ctx=ctx)
        t.q_opt = Adam.make[target=Self.train_target, M=Self.Q_NET](
            t.pair.online,
            ctx=ctx,
        )
        t.q_opt.lr = lr
        t.q_opt.max_grad_norm = max_grad_norm

        t.mse_loss = MSELoss[1].make[Self.train_target](ctx=ctx)

        t.state = TrainerState[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
        ].make[
            Self.train_target
        ](ctx=ctx)

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

    def _train_step_impl(mut self, step_idx: Int) raises -> Bool:
        self.state.step_idx = step_idx
        self.state.did_step = True
        comptime if Self.train_target == "gpu":
            self.state.ctx = self.ctx

        # 1. Sample minibatch.
        var t_sample = perf_counter_ns()
        self.sample_blk.step(self.state)
        if not self.state.did_step:
            return False
        self.timer.accumulate(Self._T_SAMPLE, t_sample)

        comptime target = Self.train_target
        comptime BATCH = Self.BATCH
        comptime OBS = Self.OBS_DIM
        comptime NA = Self.NUM_ACTIONS

        var q_all_p = self._q_all.target_ptr[target]()
        var q_gath_p = self._q_gathered.target_ptr[target]()
        var grad_q_p = self._grad_q.target_ptr[target]()
        var grad_obs_p = self._grad_obs.target_ptr[target]()
        var mb_s_p = self.state.mb_s.target_ptr[target]()
        var mb_a_p = self.state.mb_a.target_ptr[target]()
        var mb_r_p = self.state.mb_r.target_ptr[target]()
        var mb_sp_p = self.state.mb_sp.target_ptr[target]()
        var mb_d_p = self.state.mb_d.target_ptr[target]()
        var mb_y_p = self.state.mb_y.target_ptr[target]()

        # GPU: D2H mb_r, mb_d, mb_a once — used by target-Y + gather/scatter.
        # Staging buffers pre-allocated via Scratch[..., STAGING=True].
        comptime if target == "gpu":
            var ctx = self.ctx.value()
            ctx.enqueue_copy(
                self.state.mb_r.cpu_ptr(),
                self.state.mb_r.dev.value(),
            )
            ctx.enqueue_copy(
                self.state.mb_d.cpu_ptr(),
                self.state.mb_d.dev.value(),
            )
            ctx.enqueue_copy(
                self.state.mb_a.cpu_ptr(),
                self.state.mb_a.dev.value(),
            )
            ctx.synchronize()
        var h_r = self.state.mb_r.cpu_ptr()
        var h_d = self.state.mb_d.cpu_ptr()
        var h_a = self.state.mb_a.cpu_ptr()
        var h_q = self._q_all.cpu_ptr()
        var h_y = self.state.mb_y.cpu_ptr()

        # 2. Target-Y.
        #   standard: y = r + γ * max_a Q_target(s', a) * (1 - d)
        #   double:   y = r + γ * Q_target(s', argmax_a Q_online(s', a)) * (1 - d)
        var t_ty = perf_counter_ns()
        var sp_t = TileTensor(mb_sp_p, row_major[BATCH, OBS]())
        var q_all_t = TileTensor(q_all_p, row_major[BATCH, NA]())
        comptime if Self.DOUBLE:
            # Double DQN: online selects, target evaluates.
            # Step A: forward Q_online(sp) → _q_all, extract argmax.
            self.pair.online.forward[target, BATCH](sp_t, output=q_all_t)
            var best_actions = List[Int](length=BATCH, fill=0)
            comptime if target == "gpu":
                var ctx = self.ctx.value()
                ctx.enqueue_copy(h_q, self._q_all.dev.value())
                ctx.synchronize()
            for b in range(BATCH):
                var best_a = 0
                var best_q = h_q[b * NA]
                for a in range(1, NA):
                    var q = h_q[b * NA + a]
                    if q > best_q:
                        best_q = q
                        best_a = a
                best_actions[b] = best_a
            # Step B: forward Q_target(sp) → _q_all, gather at argmax.
            self.pair.target_net.forward[target, BATCH](sp_t, output=q_all_t)
            comptime if target == "gpu":
                var ctx = self.ctx.value()
                ctx.enqueue_copy(h_q, self._q_all.dev.value())
                ctx.synchronize()
            for b in range(BATCH):
                var tgt_q = h_q[b * NA + best_actions[b]]
                h_y[b] = h_r[b] + self.gamma * tgt_q * (
                    Scalar[DT](1.0) - h_d[b]
                )
            comptime if target == "gpu":
                self.ctx.value().enqueue_copy(
                    self.state.mb_y.dev.value(),
                    h_y,
                )
        else:
            # Standard DQN: target selects and evaluates.
            self.pair.target_net.forward[target, BATCH](sp_t, output=q_all_t)
            comptime if target == "gpu":
                var ctx = self.ctx.value()
                ctx.enqueue_copy(h_q, self._q_all.dev.value())
                ctx.synchronize()
            for b in range(BATCH):
                var max_q = h_q[b * NA]
                for a in range(1, NA):
                    var q = h_q[b * NA + a]
                    if q > max_q:
                        max_q = q
                h_y[b] = h_r[b] + self.gamma * max_q * (
                    Scalar[DT](1.0) - h_d[b]
                )
            comptime if target == "gpu":
                self.ctx.value().enqueue_copy(
                    self.state.mb_y.dev.value(),
                    h_y,
                )
        self.timer.accumulate(Self._T_TARGET_Y, t_ty)

        # 3. Critic update: MSE(Q_online(s)[a_taken], y).
        var t_crit = perf_counter_ns()
        self.q_opt.zero_grad[target, M=Self.Q_NET](self.pair.online)

        var s_t = TileTensor(mb_s_p, row_major[BATCH, OBS]())
        var q_all_t2 = TileTensor(q_all_p, row_major[BATCH, NA]())
        self.pair.online.forward[target, BATCH](s_t, output=q_all_t2)

        # Gather Q(s, a_taken) → _q_gathered.
        comptime if target == "gpu":
            var ctx = self.ctx.value()
            ctx.enqueue_copy(h_q, self._q_all.dev.value())
            ctx.synchronize()
        var h_gath = self._q_gathered.cpu_ptr()
        for b in range(BATCH):
            var a_idx = Int(h_a[b])
            h_gath[b] = h_q[b * NA + a_idx]
        comptime if target == "gpu":
            self.ctx.value().enqueue_copy(
                self._q_gathered.dev.value(),
                h_gath,
            )

        # MSE forward + backward.
        var q_gath_t = TileTensor(q_gath_p, row_major[BATCH, 1]())
        var y_t = TileTensor(mb_y_p, row_major[BATCH, 1]())
        var loss = self.mse_loss.forward[target, BATCH](q_gath_t, y_t)
        var grad_q_t = TileTensor(grad_q_p, row_major[BATCH, 1]())
        self.mse_loss.vjp[target, BATCH](y_t, grad_q_t)

        # Scatter grad_q → grad_q_all (sparse: only taken-action slot nonzero).
        comptime if target == "gpu":
            var ctx = self.ctx.value()
            ctx.enqueue_copy(
                self._grad_q.cpu_ptr(),
                self._grad_q.dev.value(),
            )
            ctx.synchronize()
        var h_grad = self._grad_q.cpu_ptr()
        for i in range(BATCH * NA):
            h_q[i] = Scalar[DT](0.0)
        for b in range(BATCH):
            var a_idx = Int(h_a[b])
            h_q[b * NA + a_idx] = h_grad[b]
        comptime if target == "gpu":
            self.ctx.value().enqueue_copy(self._q_all.dev.value(), h_q)

        # Q_online.vjp → accumulate param grads.
        var grad_q_all_t = TileTensor(q_all_p, row_major[BATCH, NA]())
        var grad_obs_t = TileTensor(grad_obs_p, row_major[BATCH, OBS]())
        self.pair.online.vjp[target, BATCH](grad_q_all_t, grad_obs_t)
        self.q_opt.step[target, M=Self.Q_NET](self.pair.online)
        self.timer.accumulate(Self._T_CRITIC, t_crit)

        # 4. Target update.
        var t_poly = perf_counter_ns()
        if self.target_update_freq > 0:
            if step_idx % self.target_update_freq == 0:
                self.pair.polyak_step[target](Scalar[DT](1.0), self.ctx)
        else:
            self.pair.polyak_step[target](self.tau, self.ctx)
        self.timer.accumulate(Self._T_POLYAK, t_poly)

        self._loss_accum += loss
        self._update_count += 1
        return True

    def train_step(mut self, step_idx: Int) raises -> Bool:
        return self._train_step_impl(step_idx)

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
            obs,
            self._action_list,
            reward,
            next_obs,
            done,
            ctx=self.ctx,
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
                obs_lane,
                act_lane,
                reward_ptr[env_idx],
                nxt_lane,
                done_ptr[env_idx],
                ctx=self.ctx,
            )

    # ─── Action selection ────────────────────────────────────────────

    def select_action_batched[
        N_ENVS: Int
    ](
        mut self,
        obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        step_idx: Int,
    ) raises:
        comptime NA = Self.NUM_ACTIONS
        comptime OBS = Self.OBS_DIM

        if step_idx < self.learning_starts:
            for i in range(N_ENVS):
                var r = random_float64()
                action_ptr[i] = Scalar[DT](Int(r * Float64(NA)))
            return

        comptime if Self.train_target == "cpu":
            var q_buf = List[Scalar[DT]](
                length=N_ENVS * NA,
                fill=Scalar[DT](0.0),
            )
            var q_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                q_buf.unsafe_ptr()
            )
            var obs_t = TileTensor(obs_ptr, row_major[N_ENVS, OBS]())
            var q_t = TileTensor(q_ptr, row_major[N_ENVS, NA]())
            self.pair.online.forward[Self.train_target, N_ENVS](
                obs_t,
                output=q_t,
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
            # GPU path. For N_ENVS=1 (Tier-1 driver), use _q_select
            # staging scratch (pre-allocated). action_ptr is device-side;
            # the driver owns a DriverScratch with host mirror and handles
            # the D2H itself after this call returns.
            comptime assert (
                N_ENVS == 1
            ), "GPU select_action_batched: N_ENVS>1 not yet supported"
            var ctx = self.ctx.value()
            var obs_t = TileTensor(obs_ptr, row_major[1, OBS]())
            var q_t = TileTensor(
                self._q_select.dev_ptr(),
                row_major[1, NA](),
            )
            self.pair.online.forward[Self.train_target, 1](
                obs_t,
                output=q_t,
            )
            ctx.enqueue_copy(
                self._q_select.cpu_ptr(),
                self._q_select.dev.value(),
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
                ctx,
                action_ptr,
                1,
                owning=False,
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
            var ob1_t = TileTensor(
                self._ob1.dev_ptr(),
                row_major[1, OBS](),
            )
            var q_t = TileTensor(
                self._q_select.dev_ptr(),
                row_major[1, NA](),
            )
            self.pair.online.forward[Self.train_target, 1](ob1_t, output=q_t)
            ctx.enqueue_copy(
                self._q_select.cpu_ptr(), self._q_select.dev.value()
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

    def flush_timer_log(mut self) -> String:
        var report = self.timer.format_report()
        self.timer.reset()
        return report

"""SACTrainerV2R — unified SAC trainer: CPU/GPU × uniform/PER replay.

Replaces the previous three-trainer matrix (SACTrainerV2R / SACTrainerV2RGpu /
SACPerTrainerV2RGpu) with one struct parameterised on:

  - `target: StaticString` — "cpu" or "gpu" — kernel dispatch
  - `SAMPLE: SampleBlock`  — replay-buffer-owning block, picks uniform
                             vs PER vs (future) N-step / sequence

Single pipeline body in `_train_step_impl[target]`. Single `make[target]`
factory using the matmul-stdlib `Optional[DeviceContext]` idiom. Block
choices made at type-instantiation time → zero runtime branches on
target or replay kind beyond the comptime-if already inside each block.

Dual driver-trait conformance: exposes both `train_step` /
`select_action` (satisfies `OffPolicyTrainable`) and `train_step_gpu` /
`select_action_gpu` (satisfies `OffPolicyTrainableGpu`). Each
non-parametric method dispatches into the parametric `_impl[target]`.
Calling the wrong surface (e.g. `train_step` on a `target="gpu"`
instance) raises at runtime. This keeps legacy non-V2R trainers on
their existing traits unchanged until they're sunset; after that we
can collapse to a single parametric trait method.

Bit-equivalent to the previous SACTrainerV2R when
`SAMPLE = UniformSampleCpuStep` + `target = "cpu"` (validated by
the bit-identity gate −169.04118 @ 30k Pendulum seed=42).
"""

from std.math import exp as fexp, log as flog, tanh as ftanh
from std.random import random_float64
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.core.logger import Logger, NoOpLogger
from ..constants import DT
from ..core import Module
from ..core.log_bundle import log_bundle
from ..core.metric import LogScalar
from ..core.online_target_pair import OnlineTargetPair
from ..core.scratch import Scratch
from ..core.scratch_walkers import init_scratch_auto
from ..initializer import Xavier
from ..optimizer.adam import Adam
from ..optimizer.scalar_adam import ScalarAdam
from .episode_tracker import EpisodeTracker
from .sac_metrics import SACMetrics
from .timer import Timer
from .trainer_block import TrainerState
from .driver_cpu import OffPolicyTrainable, OffPolicyTrainableGpu
from .blocks_ref import (
    SampleBlock,
    TargetYStep,
    TwinCriticStep,
    SACActorStep,
    AlphaUpdateStep,
    PolyakStep,
)


struct SACTrainerV2R[
    target: StaticString,
    SAMPLE: SampleBlock,
    ACTOR: Module,
    CRITIC: Module,
](OffPolicyTrainable, OffPolicyTrainableGpu):
    """Dimensions (OBS / ACT / BATCH) are derived from SAMPLE so the
    user specifies them ONCE (on the sample block type), not on both
    the trainer and the block. Symbolic-equality follows: the
    TrainerState the trainer holds is exactly the one SAMPLE.step
    expects, so the Mojo type system stops complaining about
    `OBS_DIM != SAMPLE.OBS` mismatches even though they're numerically
    equal."""

    comptime OBS_DIM: Int = Self.SAMPLE.OBS
    comptime ACT_DIM: Int = Self.SAMPLE.ACT
    comptime BATCH:   Int = Self.SAMPLE.BATCH

    comptime AGENT_OBS_DIM: Int = Self.OBS_DIM
    comptime AGENT_ACT_DIM: Int = Self.ACT_DIM

    # Timer section indices. Order matches `add_section` calls in `make`.
    comptime _T_SAMPLE   = 0
    comptime _T_TARGET_Y = 1
    comptime _T_CRITIC   = 2
    comptime _T_ACTOR    = 3
    comptime _T_ALPHA    = 4
    comptime _T_POLYAK   = 5

    var actor:       Self.ACTOR
    var pair1:       OnlineTargetPair[Self.CRITIC]
    var pair2:       OnlineTargetPair[Self.CRITIC]
    var actor_opt:   Adam
    var critic1_opt: Adam
    var critic2_opt: Adam
    var alpha_opt:   ScalarAdam

    var sample_blk: Self.SAMPLE
    var target_y_blk: TargetYStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
    ]
    var twin_critic_blk: TwinCriticStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
    ]
    var actor_blk: SACActorStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
    ]
    var alpha_blk: AlphaUpdateStep[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]
    var polyak_blk: PolyakStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
    ]

    var state:   TrainerState[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]
    var tracker: EpisodeTracker
    var ctx:     Optional[DeviceContext]

    var _ob1:  Scratch["ob1",  Self.OBS_DIM, True]
    var _ao1:  Scratch["ao1",  2 * Self.ACT_DIM, True]
    var _alp1: Scratch["alp1", Self.ACT_DIM + 1, True]

    var action_scale:    Scalar[DT]
    var learning_starts: Int

    var _actor_L_accum:  Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    var _alpha_accum:    Scalar[DT]
    var _update_count:   Int

    var timer: Timer

    def __init__(out self):
        self.actor = Self.ACTOR()
        self.pair1 = OnlineTargetPair[Self.CRITIC]()
        self.pair2 = OnlineTargetPair[Self.CRITIC]()
        self.actor_opt   = Adam()
        self.critic1_opt = Adam()
        self.critic2_opt = Adam()
        self.alpha_opt = ScalarAdam(
            value=0.0, m=0.0, v=0.0, t=0,
            lr=0.0003, beta1=0.9, beta2=0.999, eps=1e-8,
        )
        self.sample_blk = Self.SAMPLE()
        self.target_y_blk = TargetYStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
        ]()
        self.twin_critic_blk = TwinCriticStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ]()
        self.actor_blk = SACActorStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
        ]()
        self.alpha_blk = AlphaUpdateStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ]()
        self.polyak_blk = PolyakStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ]()
        self.state = TrainerState[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ]()
        self.tracker = EpisodeTracker(
            window=List[Scalar[DT]](), window_size=0, idx=0,
            current_return=Scalar[DT](0.0), ep_count=0,
        )
        self.ctx = None
        self._ob1  = Scratch["ob1",  Self.OBS_DIM, True]()
        self._ao1  = Scratch["ao1",  2 * Self.ACT_DIM, True]()
        self._alp1 = Scratch["alp1", Self.ACT_DIM + 1, True]()
        self.action_scale = Scalar[DT](1.0)
        self.learning_starts = 1_000
        self._actor_L_accum  = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum    = Scalar[DT](0.0)
        self._update_count   = 0
        self.timer = Timer.new()

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
    ) raises -> Self:
        """Unified factory. PER args are applied unconditionally via the
        SampleBlock trait's `configure_per` (no-op default for uniform
        blocks). `ctx` is required for `target="gpu"`."""
        comptime assert Self.target == "cpu" or Self.target == "gpu", (
            "SACTrainerV2R: target must be 'cpu' or 'gpu'"
        )
        comptime if Self.target == "gpu":
            if not ctx:
                raise Error("SACTrainerV2R.make[target='gpu']: ctx required")

        var t = Self()
        t.ctx = ctx

        t.actor = Self.ACTOR.make[target=Self.target, INIT=Xavier](ctx=ctx)
        t.pair1 = OnlineTargetPair[Self.CRITIC].make[
            target=Self.target, INIT=Xavier
        ](ctx=ctx)
        t.pair2 = OnlineTargetPair[Self.CRITIC].make[
            target=Self.target, INIT=Xavier
        ](ctx=ctx)
        t.actor_opt = Adam.make[target=Self.target, M=Self.ACTOR](
            t.actor, ctx=ctx,
        )
        t.critic1_opt = Adam.make[target=Self.target, M=Self.CRITIC](
            t.pair1.online, ctx=ctx,
        )
        t.critic2_opt = Adam.make[target=Self.target, M=Self.CRITIC](
            t.pair2.online, ctx=ctx,
        )
        t.target_y_blk = TargetYStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
            Self.ACTOR, Self.CRITIC,
        ].make[Self.target](
            action_scale=action_scale, gamma=gamma, ctx=ctx,
        )
        t.twin_critic_blk = TwinCriticStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ].make[Self.target](ctx=ctx)
        t.actor_blk = SACActorStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
            Self.ACTOR, Self.CRITIC,
        ].make[Self.target](action_scale=action_scale, ctx=ctx)
        t.state = TrainerState[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ].make[Self.target](ctx=ctx)

        t.actor_opt.lr = actor_lr
        t.actor_opt.max_grad_norm = max_grad_norm
        t.critic1_opt.lr = critic_lr
        t.critic1_opt.max_grad_norm = max_grad_norm
        t.critic2_opt.lr = critic_lr
        t.critic2_opt.max_grad_norm = max_grad_norm
        t.alpha_opt = ScalarAdam.new(flog(init_alpha), alpha_lr)

        t.alpha_blk = AlphaUpdateStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ].make(target_entropy=target_entropy)
        t.polyak_blk = PolyakStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ].make(tau=tau)

        t.tracker = EpisodeTracker.new(
            window_size=window_size, initial_fill=initial_episode_fill
        )

        init_scratch_auto[Self, target=Self.target](t, ctx)

        t.action_scale = action_scale
        t.learning_starts = learning_starts

        # PER hyperparameter wiring: no-op default for uniform blocks.
        t.sample_blk.configure_per(
            alpha=per_alpha, beta=per_beta, epsilon=per_epsilon,
        )
        t.sample_blk.setup(learning_starts, ctx=ctx)

        # Timer sections — index order MUST match the `_T_*` comptime
        # constants above. Six standard SAC train_step phases.
        t.timer.add_section("sample")
        t.timer.add_section("target_y")
        t.timer.add_section("critic")
        t.timer.add_section("actor")
        t.timer.add_section("alpha")
        t.timer.add_section("polyak")
        return t^

    def set_beta(mut self, beta: Scalar[DT]):
        """PER IS-β anneal hook (callers ramp 0.4 → 1.0). No-op for
        uniform sample blocks."""
        self.sample_blk.set_beta(beta)

    # ─── Internal parametric core ─────────────────────────────────────

    def _select_action_impl(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        if step_idx < self.learning_starts:
            for j in range(Self.ACT_DIM):
                var u = Scalar[DT](2.0 * random_float64() - 1.0)
                action_out[j] = u * self.action_scale
            return
        var ob1_cpu_p = self._ob1.cpu_ptr()
        var ao1_cpu_p = self._ao1.cpu_ptr()
        var alp1_cpu_p = self._alp1.cpu_ptr()
        for d in range(Self.OBS_DIM):
            ob1_cpu_p[d] = obs[d]

        comptime if Self.target == "cpu":
            var ob1_t = TileTensor(ob1_cpu_p, row_major[1, Self.OBS_DIM]())
            var ao1_t = TileTensor(
                ao1_cpu_p, row_major[1, 2 * Self.ACT_DIM]()
            )
            self.actor.forward["cpu", 1](ob1_t, output=ao1_t)
            var alp1_t = TileTensor(
                alp1_cpu_p, row_major[1, Self.ACT_DIM + 1](),
            )
            self.actor_blk.inner.rsample.forward["cpu", 1](
                ao1_t, output=alp1_t,
            )
        else:
            var ctx = self.ctx.value()
            ctx.enqueue_copy(self._ob1.dev.value(), ob1_cpu_p)
            var ob1_p = self._ob1.dev_ptr()
            var ao1_p = self._ao1.dev_ptr()
            var alp1_p = self._alp1.dev_ptr()
            var ob1_t = TileTensor(ob1_p, row_major[1, Self.OBS_DIM]())
            var ao1_t = TileTensor(ao1_p, row_major[1, 2 * Self.ACT_DIM]())
            self.actor.forward["gpu", 1](ob1_t, output=ao1_t)
            var alp1_t = TileTensor(
                alp1_p, row_major[1, Self.ACT_DIM + 1](),
            )
            self.actor_blk.inner.rsample.forward["gpu", 1](
                ao1_t, output=alp1_t,
            )
            ctx.enqueue_copy(alp1_cpu_p, self._alp1.dev.value())
            ctx.synchronize()

        for j in range(Self.ACT_DIM):
            var a = alp1_cpu_p[j]
            if a > self.action_scale:
                a = self.action_scale
            elif a < -self.action_scale:
                a = -self.action_scale
            action_out[j] = a

    def _select_greedy_action_impl(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        var ob1_cpu_p = self._ob1.cpu_ptr()
        var ao1_cpu_p = self._ao1.cpu_ptr()
        for d in range(Self.OBS_DIM):
            ob1_cpu_p[d] = obs[d]

        comptime if Self.target == "cpu":
            var ob1_t = TileTensor(ob1_cpu_p, row_major[1, Self.OBS_DIM]())
            var ao1_t = TileTensor(
                ao1_cpu_p, row_major[1, 2 * Self.ACT_DIM]()
            )
            self.actor.forward["cpu", 1](ob1_t, output=ao1_t)
        else:
            var ctx = self.ctx.value()
            ctx.enqueue_copy(self._ob1.dev.value(), ob1_cpu_p)
            var ob1_t = TileTensor(
                self._ob1.dev_ptr(), row_major[1, Self.OBS_DIM](),
            )
            var ao1_t = TileTensor(
                self._ao1.dev_ptr(), row_major[1, 2 * Self.ACT_DIM](),
            )
            self.actor.forward["gpu", 1](ob1_t, output=ao1_t)
            ctx.enqueue_copy(ao1_cpu_p, self._ao1.dev.value())
            ctx.synchronize()

        for j in range(Self.ACT_DIM):
            var mean = ao1_cpu_p[j]
            var a = ftanh(mean) * self.action_scale
            if a > self.action_scale:
                a = self.action_scale
            elif a < -self.action_scale:
                a = -self.action_scale
            action_out[j] = a

    def _train_step_impl(mut self, step_idx: Int) raises -> Bool:
        """Single pipeline body shared across CPU/GPU and uniform/PER.
        Replay-specific behavior lives in `self.sample_blk` (which
        block-internally branches via state.has_per + handles its own
        target). All other blocks parametric on `[Self.target]`."""
        self.state.step_idx = step_idx
        self.state.did_step = True
        self.state.alpha = fexp(self.alpha_opt.value)
        comptime if Self.target == "gpu":
            self.state.ctx = self.ctx

        var t_sample = perf_counter_ns()
        self.sample_blk.step(self.state)
        if not self.state.did_step:
            return False
        self.timer.accumulate(Self._T_SAMPLE, t_sample)

        var t_ty = perf_counter_ns()
        self.target_y_blk.step[Self.target](
            self.state, self.actor,
            self.pair1.target_net, self.pair2.target_net,
        )
        self.timer.accumulate(Self._T_TARGET_Y, t_ty)

        var t_crit = perf_counter_ns()
        self.twin_critic_blk.step[Self.target](
            self.state,
            self.pair1.online, self.critic1_opt,
            self.pair2.online, self.critic2_opt,
        )
        self.timer.accumulate(Self._T_CRITIC, t_crit)

        var t_act = perf_counter_ns()
        self.actor_blk.step[Self.target](
            self.state, self.actor, self.actor_opt,
            self.pair1.online, self.pair2.online,
        )
        self.timer.accumulate(Self._T_ACTOR, t_act)

        var t_alp = perf_counter_ns()
        self.alpha_blk.step(self.state, self.alpha_opt)
        self.timer.accumulate(Self._T_ALPHA, t_alp)

        var t_pol = perf_counter_ns()
        self.polyak_blk.step[Self.target](
            self.state, self.pair1, self.pair2,
        )
        self.timer.accumulate(Self._T_POLYAK, t_pol)

        # PER tail (no-op for uniform blocks).
        self.sample_blk.update_priorities(self.state)

        self._actor_L_accum  += self.state.actor_loss
        self._critic_L_accum += self.state.critic_loss
        self._alpha_accum    += fexp(self.alpha_opt.value)
        self._update_count   += 1
        return True

    def _record_impl(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        self.tracker.add_reward(reward)
        self.sample_blk.add(
            obs, action, reward, next_obs, done, ctx=self.ctx,
        )

    # ─── OffPolicyTrainable (CPU) surface ─────────────────────────────

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        comptime if Self.target != "cpu":
            raise Error(
                "SACTrainerV2R[target='gpu']: use select_action_gpu"
            )
        self._select_action_impl(obs, action_out, step_idx)

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        comptime if Self.target != "cpu":
            raise Error(
                "SACTrainerV2R[target='gpu']: use select_greedy_action_gpu"
            )
        self._select_greedy_action_impl(obs, action_out)

    def train_step(mut self, step_idx: Int) raises -> Bool:
        comptime if Self.target != "cpu":
            raise Error("SACTrainerV2R[target='gpu']: use train_step_gpu")
        return self._train_step_impl(step_idx)

    # ─── OffPolicyTrainableGpu (GPU) surface ──────────────────────────

    def select_action_gpu(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        comptime if Self.target != "gpu":
            raise Error(
                "SACTrainerV2R[target='cpu']: use select_action"
            )
        self._select_action_impl(obs, action_out, step_idx)

    def select_greedy_action_gpu(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        comptime if Self.target != "gpu":
            raise Error(
                "SACTrainerV2R[target='cpu']: use select_greedy_action"
            )
        self._select_greedy_action_impl(obs, action_out)

    def train_step_gpu(mut self, step_idx: Int) raises -> Bool:
        comptime if Self.target != "gpu":
            raise Error("SACTrainerV2R[target='cpu']: use train_step")
        return self._train_step_impl(step_idx)

    # ─── Shared surface (both traits) ─────────────────────────────────

    def record(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        self._record_impl(obs, action, reward, next_obs, done)

    def end_episode(mut self):
        self.tracker.end_episode()

    def mean_return(self) -> Scalar[DT]:
        return self.tracker.mean_return()

    def ep_count(self) -> Int:
        return self.tracker.ep_count

    # ─── Logging surface (parity with legacy SACTrainer) ──────────────

    def flush_train_log(mut self) -> Tuple[
        Scalar[DT], Scalar[DT], Scalar[DT], Int
    ]:
        """Return (mean_actor_loss, mean_critic_loss, mean_alpha, n_updates)
        accumulated since the last flush. Resets accumulators."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var out = (
            self._actor_L_accum * inv,
            self._critic_L_accum * inv,
            self._alpha_accum * inv,
            self._update_count,
        )
        self._actor_L_accum  = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum    = Scalar[DT](0.0)
        self._update_count   = 0
        return out

    def flush_metrics[L: Logger = NoOpLogger](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> SACMetrics:
        """Drain accumulators into a SACMetrics bundle. If a logger
        pointer is wired, also emit one log_scalar per metric field.
        Resets accumulators on every call."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var bundle = SACMetrics(
            actor_loss=LogScalar[DT](self._actor_L_accum * inv),
            critic_loss=LogScalar[DT](self._critic_L_accum * inv),
            alpha=LogScalar[DT](self._alpha_accum * inv),
            n_updates=LogScalar[DT](Scalar[DT](self._update_count)),
        )
        self._actor_L_accum  = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum    = Scalar[DT](0.0)
        self._update_count   = 0
        if Bool(logger):
            log_bundle(logger.value()[], bundle, step)
        return bundle^

    def flush_timer_log(mut self) -> String:
        """Return a per-section wall-time report (one line per sub-step:
        sample / target_y / critic / actor / alpha / polyak) and reset
        the accumulators."""
        var report = self.timer.format_report()
        self.timer.reset()
        return report

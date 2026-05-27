"""SACTrainer — unified SAC trainer: CPU/GPU × uniform/PER replay.

Replaces the previous three-trainer matrix (SACTrainer / SACTrainerGpu /
SACPerTrainerGpu) with one struct parameterised on:

  - `train_target: StaticString` — "cpu" or "gpu" — kernel dispatch.
    Renamed from `target` (Phase 3.5) to disambiguate from the env's
    `ENV_TARGET` introduced by the dual-target unified driver.
  - `SAMPLE: SampleBlock` — replay-buffer-owning block, picks uniform
                             vs PER vs (future) N-step / sequence

Single pipeline body in `_train_step_impl[train_target]`. Single
`make[train_target]` factory using the matmul-stdlib
`Optional[DeviceContext]` idiom. Block choices made at type-instantiation
time → zero runtime branches on target or replay kind beyond the
comptime-if already inside each block.

Dual driver-trait conformance: exposes both `train_step` /
`select_action` (satisfies `OffPolicyTrainable`) and `train_step_gpu` /
`select_action_gpu` (satisfies `OffPolicyTrainableGpu`), plus the
newer `train_step_unified` / `select_action_unified` (satisfies
`OffPolicyAgentUnified`). Each non-parametric method dispatches into
the parametric `_impl[train_target]`. Calling the wrong surface (e.g.
`train_step` on a `train_target="gpu"` instance) raises at runtime.

Bit-equivalent to the previous SACTrainer when
`SAMPLE = UniformSampleCpuStep` + `train_target = "cpu"` (validated by
the bit-identity gate −169.04118 @ 30k Pendulum seed=42).
"""

from std.math import exp as fexp, log as flog, tanh as ftanh
from std.random import random_float64
from std.time import perf_counter_ns
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..data.n_step_replay import GPUNStepBuffer

from mojo_rl.core.logger import Logger, NoOpLogger
from ..constants import DT
from ..core import Module
from ..core.amp import AMPPolicy, NoAMP, Bf16Compute
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
from .driver_cpu import (
    OffPolicyTrainable,
    OffPolicyTrainableGpu,
    OffPolicyTrainableGpuBatched,
)
from .driver_unified import OffPolicyAgentUnified, OffPolicyAgentUnifiedGpu
from .blocks import (
    SampleBlock,
    TargetYStep,
    TwinCriticStep,
    SACActorStep,
    AlphaUpdateStep,
    PolyakStep,
)


# ──────────────────────────────────────────────────────────────────────
# Top-level kernels for the batched GPU action path (copied from legacy
# SACTrainer — Phase B.5b kernels, same semantics).
# ──────────────────────────────────────────────────────────────────────


def _warmup_uniform_kernel[
    N_ENVS: Int, ACT: Int
](
    action_dest: LayoutTensor[
        DT,
        Layout.row_major(N_ENVS, ACT),
        MutAnyOrigin,
    ],
    action_scale: Scalar[DT],
    seed: UInt64,
    offset_base: UInt64,
):
    """Per-lane Philox uniform → [N_ENVS, ACT] of Uniform(-scale, +scale)."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    var total = N_ENVS * ACT
    if i >= total:
        return
    var philox = PhiloxRandom(seed=seed + UInt64(i), offset=offset_base)
    var u = Float32(philox.step_uniform()[0])
    var s = Scalar[DT](2.0) * Scalar[DT](u) - Scalar[DT](1.0)
    var env = i // ACT
    var j = i % ACT
    action_dest[env, j] = s * action_scale


def _action_clamp_kernel[
    N_ENVS: Int, ACT: Int
](
    alp: LayoutTensor[
        DT,
        Layout.row_major(N_ENVS, ACT + 1),
        MutAnyOrigin,
    ],
    action_out: LayoutTensor[
        DT,
        Layout.row_major(N_ENVS, ACT),
        MutAnyOrigin,
    ],
    action_scale: Scalar[DT],
):
    """Extract first ACT lanes of alp (drop log_prob), clamp into action_out."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    var total = N_ENVS * ACT
    if i >= total:
        return
    var env = i // ACT
    var j = i % ACT
    var a = alp[env, j]
    if a > action_scale:
        a = action_scale
    elif a < -action_scale:
        a = -action_scale
    action_out[env, j] = a


struct SACTrainer[
    train_target: StaticString,
    SAMPLE: SampleBlock,
    ACTOR: Module,
    CRITIC: Module,
](
    OffPolicyTrainable,
    OffPolicyTrainableGpu,
    OffPolicyTrainableGpuBatched,
    OffPolicyAgentUnified,
    OffPolicyAgentUnifiedGpu,
):
    """Dimensions (OBS / ACT / BATCH) are derived from SAMPLE so the
    user specifies them ONCE (on the sample block type), not on both
    the trainer and the block. Symbolic-equality follows: the
    TrainerState the trainer holds is exactly the one SAMPLE.step
    expects, so the Mojo type system stops complaining about
    `OBS_DIM != SAMPLE.OBS` mismatches even though they're numerically
    equal."""

    comptime OBS_DIM: Int = Self.SAMPLE.OBS
    comptime ACT_DIM: Int = Self.SAMPLE.ACT
    comptime BATCH: Int = Self.SAMPLE.BATCH

    comptime AGENT_OBS_DIM: Int = Self.OBS_DIM
    comptime AGENT_ACT_DIM: Int = Self.ACT_DIM
    # Trait-visible alias of the struct's `train_target` comptime param,
    # so `OffPolicyAgentUnified` (and any future trait that needs to
    # gate on the trainer's compute target) can see it as a member.
    # Distinct from the env's ENV_TARGET — see driver_unified docs.
    comptime AGENT_TRAIN_TARGET: StaticString = Self.train_target

    # Timer section indices. Order matches `add_section` calls in `make`.
    comptime _T_SAMPLE = 0
    comptime _T_TARGET_Y = 1
    comptime _T_CRITIC = 2
    comptime _T_ACTOR = 3
    comptime _T_ALPHA = 4
    comptime _T_POLYAK = 5

    var actor: Self.ACTOR
    var pair1: OnlineTargetPair[Self.CRITIC]
    var pair2: OnlineTargetPair[Self.CRITIC]
    var actor_opt: Adam
    var critic1_opt: Adam
    var critic2_opt: Adam
    var alpha_opt: ScalarAdam

    var sample_blk: Self.SAMPLE
    var target_y_blk: TargetYStep[
        Self.OBS_DIM,
        Self.ACT_DIM,
        Self.BATCH,
        Self.ACTOR,
        Self.CRITIC,
    ]
    var twin_critic_blk: TwinCriticStep[
        Self.OBS_DIM,
        Self.ACT_DIM,
        Self.BATCH,
        Self.CRITIC,
    ]
    var actor_blk: SACActorStep[
        Self.OBS_DIM,
        Self.ACT_DIM,
        Self.BATCH,
        Self.ACTOR,
        Self.CRITIC,
    ]
    var alpha_blk: AlphaUpdateStep[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]
    var polyak_blk: PolyakStep[
        Self.OBS_DIM,
        Self.ACT_DIM,
        Self.BATCH,
        Self.CRITIC,
    ]

    var state: TrainerState[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]
    var tracker: EpisodeTracker
    var ctx: Optional[DeviceContext]

    var _ob1: Scratch["ob1", Self.OBS_DIM, True]
    var _ao1: Scratch["ao1", 2 * Self.ACT_DIM, True]
    var _alp1: Scratch["alp1", Self.ACT_DIM + 1, True]

    var action_scale: Scalar[DT]
    var learning_starts: Int
    var _use_bf16: Bool
    # Philox state for batched warmup uniform actions (N_ENVS path only).
    var _warmup_rng_seed: UInt64
    var _warmup_rng_offset: UInt64

    var _actor_L_accum: Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    var _alpha_accum: Scalar[DT]
    var _update_count: Int

    var timer: Timer

    def __init__(out self):
        self.actor = Self.ACTOR()
        self.pair1 = OnlineTargetPair[Self.CRITIC]()
        self.pair2 = OnlineTargetPair[Self.CRITIC]()
        self.actor_opt = Adam()
        self.critic1_opt = Adam()
        self.critic2_opt = Adam()
        self.alpha_opt = ScalarAdam(
            value=0.0,
            m=0.0,
            v=0.0,
            t=0,
            lr=0.0003,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
        )
        self.sample_blk = Self.SAMPLE()
        self.target_y_blk = TargetYStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.ACTOR,
            Self.CRITIC,
        ]()
        self.twin_critic_blk = TwinCriticStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.CRITIC,
        ]()
        self.actor_blk = SACActorStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.ACTOR,
            Self.CRITIC,
        ]()
        self.alpha_blk = AlphaUpdateStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
        ]()
        self.polyak_blk = PolyakStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.CRITIC,
        ]()
        self.state = TrainerState[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
        ]()
        self.tracker = EpisodeTracker(
            window=List[Scalar[DT]](),
            window_size=0,
            idx=0,
            current_return=Scalar[DT](0.0),
            ep_count=0,
        )
        self.ctx = None
        self._ob1 = Scratch["ob1", Self.OBS_DIM, True]()
        self._ao1 = Scratch["ao1", 2 * Self.ACT_DIM, True]()
        self._alp1 = Scratch["alp1", Self.ACT_DIM + 1, True]()
        self.action_scale = Scalar[DT](1.0)
        self.learning_starts = 1_000
        self._use_bf16 = False
        self._warmup_rng_seed = UInt64(0xC0FFEE_C0DE)
        self._warmup_rng_offset = UInt64(0)
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum = Scalar[DT](0.0)
        self._update_count = 0
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
        use_bf16: Bool = False,
        use_ere: Bool = False,
        ere_eta: Scalar[DT] = Scalar[DT](0.996),
        ere_c_min: Int = 1,
        ere_k_max: Int = 1000,
    ) raises -> Self:
        """Unified factory. PER args are applied unconditionally via the
        SampleBlock trait's `configure_per` (no-op default for uniform
        blocks). `ctx` is required for `target="gpu"`."""
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "SACTrainer: target must be 'cpu' or 'gpu'"
        comptime if Self.train_target == "gpu":
            if not ctx:
                raise Error("SACTrainer.make[target='gpu']: ctx required")

        var t = Self()
        t.ctx = ctx

        t.actor = Self.ACTOR.make[target=Self.train_target, INIT=Xavier](ctx=ctx)
        t.pair1 = OnlineTargetPair[Self.CRITIC].make[
            target=Self.train_target, INIT=Xavier
        ](ctx=ctx)
        t.pair2 = OnlineTargetPair[Self.CRITIC].make[
            target=Self.train_target, INIT=Xavier
        ](ctx=ctx)
        t.actor_opt = Adam.make[target=Self.train_target, M=Self.ACTOR](
            t.actor,
            ctx=ctx,
        )
        t.critic1_opt = Adam.make[target=Self.train_target, M=Self.CRITIC](
            t.pair1.online,
            ctx=ctx,
        )
        t.critic2_opt = Adam.make[target=Self.train_target, M=Self.CRITIC](
            t.pair2.online,
            ctx=ctx,
        )
        t.target_y_blk = TargetYStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.ACTOR,
            Self.CRITIC,
        ].make[Self.train_target](
            action_scale=action_scale,
            gamma=gamma,
            ctx=ctx,
        )
        t.twin_critic_blk = TwinCriticStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.CRITIC,
        ].make[Self.train_target](ctx=ctx)
        t.actor_blk = SACActorStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.ACTOR,
            Self.CRITIC,
        ].make[Self.train_target](action_scale=action_scale, ctx=ctx)
        t.state = TrainerState[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
        ].make[
            Self.train_target
        ](ctx=ctx)

        t.actor_opt.lr = actor_lr
        t.actor_opt.max_grad_norm = max_grad_norm
        t.critic1_opt.lr = critic_lr
        t.critic1_opt.max_grad_norm = max_grad_norm
        t.critic2_opt.lr = critic_lr
        t.critic2_opt.max_grad_norm = max_grad_norm
        t.alpha_opt = ScalarAdam.new(flog(init_alpha), alpha_lr)

        t.alpha_blk = AlphaUpdateStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
        ].make(target_entropy=target_entropy)
        t.polyak_blk = PolyakStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.CRITIC,
        ].make(tau=tau)

        t.tracker = EpisodeTracker.new(
            window_size=window_size, initial_fill=initial_episode_fill
        )

        init_scratch_auto[Self, target=Self.train_target](t, ctx)

        t.action_scale = action_scale
        t.learning_starts = learning_starts
        t._use_bf16 = use_bf16

        # PER hyperparameter wiring: no-op default for uniform blocks.
        t.sample_blk.configure_per(
            alpha=per_alpha,
            beta=per_beta,
            epsilon=per_epsilon,
        )
        t.sample_blk.setup(learning_starts, ctx=ctx)
        # ERE wiring: no-op default for blocks that don't own GPUReplay.
        # Must come AFTER setup() (GPUReplay is constructed there).
        t.sample_blk.configure_ere(
            enable=use_ere,
            eta=ere_eta,
            c_min=ere_c_min,
            k_max=ere_k_max,
        )

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

        comptime if Self.train_target == "cpu":
            var ob1_t = TileTensor(ob1_cpu_p, row_major[1, Self.OBS_DIM]())
            var ao1_t = TileTensor(ao1_cpu_p, row_major[1, 2 * Self.ACT_DIM]())
            self.actor.forward["cpu", 1](ob1_t, output=ao1_t)
            var alp1_t = TileTensor(
                alp1_cpu_p,
                row_major[1, Self.ACT_DIM + 1](),
            )
            self.actor_blk.inner.rsample.forward["cpu", 1](
                ao1_t,
                output=alp1_t,
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
                alp1_p,
                row_major[1, Self.ACT_DIM + 1](),
            )
            self.actor_blk.inner.rsample.forward["gpu", 1](
                ao1_t,
                output=alp1_t,
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

        comptime if Self.train_target == "cpu":
            var ob1_t = TileTensor(ob1_cpu_p, row_major[1, Self.OBS_DIM]())
            var ao1_t = TileTensor(ao1_cpu_p, row_major[1, 2 * Self.ACT_DIM]())
            self.actor.forward["cpu", 1](ob1_t, output=ao1_t)
        else:
            var ctx = self.ctx.value()
            ctx.enqueue_copy(self._ob1.dev.value(), ob1_cpu_p)
            var ob1_t = TileTensor(
                self._ob1.dev_ptr(),
                row_major[1, Self.OBS_DIM](),
            )
            var ao1_t = TileTensor(
                self._ao1.dev_ptr(),
                row_major[1, 2 * Self.ACT_DIM](),
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

    def _train_step_impl[
        POLICY: AMPPolicy = NoAMP,
    ](mut self, step_idx: Int) raises -> Bool:
        """Single pipeline body shared across CPU/GPU and uniform/PER.
        Replay-specific behavior lives in `self.sample_blk` (which
        block-internally branches via state.has_per + handles its own
        target). All other blocks parametric on `[Self.train_target, POLICY]`."""
        self.state.step_idx = step_idx
        self.state.did_step = True
        self.state.alpha = fexp(self.alpha_opt.value)
        comptime if Self.train_target == "gpu":
            self.state.ctx = self.ctx

        var t_sample = perf_counter_ns()
        self.sample_blk.step(self.state)
        if not self.state.did_step:
            return False
        self.timer.accumulate(Self._T_SAMPLE, t_sample)

        var t_ty = perf_counter_ns()
        self.target_y_blk.step[Self.train_target, POLICY](
            self.state,
            self.actor,
            self.pair1.target_net,
            self.pair2.target_net,
        )
        self.timer.accumulate(Self._T_TARGET_Y, t_ty)

        var t_crit = perf_counter_ns()
        self.twin_critic_blk.step[Self.train_target, POLICY](
            self.state,
            self.pair1.online,
            self.critic1_opt,
            self.pair2.online,
            self.critic2_opt,
        )
        self.timer.accumulate(Self._T_CRITIC, t_crit)

        var t_act = perf_counter_ns()
        self.actor_blk.step[Self.train_target, POLICY](
            self.state,
            self.actor,
            self.actor_opt,
            self.pair1.online,
            self.pair2.online,
        )
        self.timer.accumulate(Self._T_ACTOR, t_act)

        var t_alp = perf_counter_ns()
        self.alpha_blk.step(self.state, self.alpha_opt)
        self.timer.accumulate(Self._T_ALPHA, t_alp)

        var t_pol = perf_counter_ns()
        self.polyak_blk.step[Self.train_target](
            self.state,
            self.pair1,
            self.pair2,
        )
        self.timer.accumulate(Self._T_POLYAK, t_pol)

        # PER tail (no-op for uniform blocks).
        self.sample_blk.update_priorities(self.state)

        self._actor_L_accum += self.state.actor_loss
        self._critic_L_accum += self.state.critic_loss
        self._alpha_accum += fexp(self.alpha_opt.value)
        self._update_count += 1
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
            obs,
            action,
            reward,
            next_obs,
            done,
            ctx=self.ctx,
        )

    # ─── OffPolicyTrainable (CPU) surface ─────────────────────────────

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        comptime if Self.train_target != "cpu":
            raise Error("SACTrainer[target='gpu']: use select_action_gpu")
        self._select_action_impl(obs, action_out, step_idx)

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        comptime if Self.train_target != "cpu":
            raise Error(
                "SACTrainer[target='gpu']: use select_greedy_action_gpu"
            )
        self._select_greedy_action_impl(obs, action_out)

    def train_step(mut self, step_idx: Int) raises -> Bool:
        comptime if Self.train_target != "cpu":
            raise Error("SACTrainer[target='gpu']: use train_step_gpu")
        return self._train_step_impl[NoAMP](step_idx)

    # ─── Tier-1 unified train_step — target-agnostic ─────────────────
    #
    # `train_step_unified` collapses `train_step` / `train_step_gpu`.
    # Picks the right AMP policy (NoAMP / Bf16Compute) and dispatches to
    # `_train_step_impl[POLICY]`. Used by the unified single-env driver
    # (`run_offpolicy_train_unified`); `OffPolicyAgentUnified`-trait
    # surface.
    def train_step_unified(mut self, step_idx: Int) raises -> Bool:
        comptime if Self.train_target == "cpu":
            return self._train_step_impl[NoAMP](step_idx)
        else:
            if self._use_bf16:
                return self._train_step_impl[Bf16Compute](step_idx)
            return self._train_step_impl[NoAMP](step_idx)

    # ─── OffPolicyTrainableGpu (GPU) surface ──────────────────────────

    def select_action_gpu(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        comptime if Self.train_target != "gpu":
            raise Error("SACTrainer[target='cpu']: use select_action")
        self._select_action_impl(obs, action_out, step_idx)

    def select_greedy_action_gpu(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        comptime if Self.train_target != "gpu":
            raise Error("SACTrainer[target='cpu']: use select_greedy_action")
        self._select_greedy_action_impl(obs, action_out)

    def train_step_gpu(mut self, step_idx: Int) raises -> Bool:
        """Auto-routes through `Bf16Compute` when the trainer was built
        with `use_bf16=True`, else `NoAMP`. Both specializations compile;
        only one is exercised per call. Mirrors legacy SACTrainer."""
        comptime if Self.train_target != "gpu":
            raise Error("SACTrainer[target='cpu']: use train_step")
        if self._use_bf16:
            return self._train_step_impl[Bf16Compute](step_idx)
        return self._train_step_impl[NoAMP](step_idx)

    # ─── OffPolicyTrainableGpuBatched (N_ENVS) surface ────────────────

    def select_action_gpu_batched[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        ao_scratch_dev: DeviceBuffer[DT],
        alp_scratch_dev: DeviceBuffer[DT],
        step_idx: Int,
    ) raises:
        """Batched policy step for N_ENVS envs — `OffPolicyTrainableGpuBatched`
        trait surface. Thin pointer-extraction wrapper that delegates to
        `select_action_unified[N_ENVS]`. The `ctx` arg is retained for
        trait conformance; the unified core uses `self.ctx` internally
        (assumed to be the same DeviceContext)."""
        comptime assert (
            Self.train_target == "gpu"
        ), "select_action_gpu_batched: target must be 'gpu'"
        comptime assert N_ENVS > 0, "N_ENVS must be > 0"
        _ = ctx  # unused — unified core threads through self.ctx
        self.select_action_unified[N_ENVS](
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                obs_dev.unsafe_ptr()
            ),
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                action_dev.unsafe_ptr()
            ),
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                ao_scratch_dev.unsafe_ptr()
            ),
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                alp_scratch_dev.unsafe_ptr()
            ),
            step_idx,
        )

    # ─── Tier-1 unified select_action — prototype ────────────────────
    #
    # `select_action_unified[N_ENVS]` collapses the current 3 method
    # spread (`_select_action_impl` for single-env CPU + GPU,
    # `select_action_gpu_batched` for N_ENVS GPU) into one parametric
    # body. The existing methods stay unchanged for back-compat — this
    # is additive only. Once drivers migrate to the unified surface and
    # `OffPolicyTrainable*` traits drop the legacy methods, the existing
    # public wrappers can go away.
    #
    # The CPU/GPU branch is taken on `Self.train_target` (the struct comptime),
    # not a per-method parameter — `target` is already pinned at make
    # time, so callers don't pass it again.
    #
    # Pointer contract: all four pointers (obs / action / ao / alp) must
    # be target-side (host for CPU, device for GPU). N_ENVS=1 GPU
    # drivers use `DriverScratch[..., with_host_mirror=True]` and do
    # the H2D obs / D2H action around this call themselves — the
    # trainer is no longer responsible for that staging.
    #
    # Sizes:
    #   obs_ptr        — N_ENVS * OBS_DIM
    #   action_ptr     — N_ENVS * ACT_DIM         (output)
    #   ao_scratch_ptr — N_ENVS * 2 * ACT_DIM     (actor output cache)
    #   alp_scratch_ptr— N_ENVS * (ACT_DIM + 1)   (action + log_prob)
    def select_action_unified[
        N_ENVS: Int,
    ](
        mut self,
        obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        ao_scratch_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        alp_scratch_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        step_idx: Int,
    ) raises:
        comptime assert N_ENVS > 0, "N_ENVS must be > 0"
        comptime if N_ENVS > 1:
            comptime assert (
                Self.train_target == "gpu"
            ), (
                "select_action_unified[N_ENVS>1]: requires the trainer's"
                " target to be 'gpu'. Tier-2 (batched CPU envs) will lift"
                " this restriction."
            )

        comptime ACT = Self.ACT_DIM
        comptime OBS = Self.OBS_DIM

        # ── Warmup: uniform random action in [-action_scale, +scale].
        if step_idx < self.learning_starts:
            comptime if Self.train_target == "cpu":
                # CPU warmup: random_float64 lane-by-lane. At N_ENVS=1
                # this consumes exactly ACT random_float64 draws in the
                # same order as the existing _select_action_impl — so
                # SAC CPU bit-identity is preserved if a caller swaps in.
                for i in range(N_ENVS * ACT):
                    var u = Scalar[DT](2.0 * random_float64() - 1.0)
                    action_ptr[i] = u * self.action_scale
            else:
                # GPU warmup: same Philox kernel select_action_gpu_batched
                # already uses. Bumps _warmup_rng_offset by 2 draws per
                # lane (matches the existing batched code path).
                var action_lt = LayoutTensor[
                    DT,
                    Layout.row_major(N_ENVS, ACT),
                    MutAnyOrigin,
                ](action_ptr)
                comptime TPB = 128
                comptime total = N_ENVS * ACT
                comptime n_blocks = (total + TPB - 1) // TPB
                comptime warmup_kernel = _warmup_uniform_kernel[N_ENVS, ACT]
                var ctx = self.ctx.value()
                ctx.enqueue_function[warmup_kernel](
                    action_lt,
                    self.action_scale,
                    self._warmup_rng_seed,
                    self._warmup_rng_offset,
                    grid_dim=n_blocks,
                    block_dim=TPB,
                )
                self._warmup_rng_offset += UInt64(N_ENVS * ACT * 2)
            return

        # ── Policy forward — actor + rsample. Identical on CPU and GPU
        # because both `actor.forward` and `rsample.forward` are
        # target-parametric. N_ENVS rolls through transparently.
        var obs_t = TileTensor(obs_ptr, row_major[N_ENVS, OBS]())
        var ao_t = TileTensor(ao_scratch_ptr, row_major[N_ENVS, 2 * ACT]())
        var alp_t = TileTensor(alp_scratch_ptr, row_major[N_ENVS, ACT + 1]())
        self.actor.forward[Self.train_target, N_ENVS](obs_t, output=ao_t)
        self.actor_blk.inner.rsample.forward[Self.train_target, N_ENVS](
            ao_t, output=alp_t
        )

        # ── Clamp + write to action_ptr. CPU loops on host; GPU
        # dispatches the existing clamp kernel.
        comptime if Self.train_target == "cpu":
            for env in range(N_ENVS):
                var src = alp_scratch_ptr + env * (ACT + 1)
                var dst = action_ptr + env * ACT
                for j in range(ACT):
                    var a = src[j]
                    if a > self.action_scale:
                        a = self.action_scale
                    elif a < -self.action_scale:
                        a = -self.action_scale
                    dst[j] = a
        else:
            var alp_lt = LayoutTensor[
                DT,
                Layout.row_major(N_ENVS, ACT + 1),
                MutAnyOrigin,
            ](alp_scratch_ptr)
            var action_lt = LayoutTensor[
                DT,
                Layout.row_major(N_ENVS, ACT),
                MutAnyOrigin,
            ](action_ptr)
            comptime TPB = 128
            comptime total = N_ENVS * ACT
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime clamp_kernel = _action_clamp_kernel[N_ENVS, ACT]
            var ctx = self.ctx.value()
            ctx.enqueue_function[clamp_kernel](
                alp_lt,
                action_lt,
                self.action_scale,
                grid_dim=n_blocks,
                block_dim=TPB,
            )

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
        """Delegates to the sample block's add_batch_gpu. Default raises
        if the block doesn't support batched-add (e.g. CPU blocks or
        single-env n-step wrappers)."""
        self.sample_blk.add_batch_gpu[N_ENVS](
            ctx,
            prev_obs_dev,
            action_dev,
            reward_dev,
            obs_dev,
            done_dev,
        )

    def record_batch_gpu_nstep[
        N_ENVS: Int, NS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut nstep_buf: GPUNStepBuffer[
            NS,
            Self.AGENT_OBS_DIM,
            Self.AGENT_ACT_DIM,
            N_ENVS,
        ],
        prev_obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        reward_dev: DeviceBuffer[DT],
        obs_dev: DeviceBuffer[DT],
        done_dev: DeviceBuffer[DT],
    ) raises:
        """N_ENVS + n-step batched record. The caller owns the
        GPUNStepBuffer[NS, OBS, ACT, N_ENVS] (N_ENVS is method-comptime,
        not struct-comptime — pre-allocating it would break single-env
        trainers). Behaviour:
          1. nstep_buf.process(...) — kernel ring-updates all N_ENVS
             lanes and emits compressed transitions into out_* device
             buffers. Invalid slots zero-padded.
          2. sample_blk.store_via_block_gpu[N_ENVS, NS] — block routes
             nstep_buf.store_into[CAP] through its owned replay
             (uniform or PER overload selected by block type).

        The caller is responsible for ensuring NS matches the trainer's
        target_y γ^N bake, AND that the sample block supports
        store_via_block_gpu (only GPU uniform / PER blocks do today)."""
        nstep_buf.process(
            ctx,
            prev_obs_dev,
            action_dev,
            reward_dev,
            obs_dev,
            done_dev,
        )
        self.sample_blk.store_via_block_gpu[N_ENVS, NS](ctx, nstep_buf)

    def add_complete_return(mut self, ret: Scalar[DT]):
        """Driver hook — push a complete-episode return into the
        rolling-window tracker (used by the N_ENVS GPU driver which
        owns per-env reward accumulators host-side)."""
        self.tracker.add_complete_return(ret)

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

    def flush_train_log(
        mut self,
    ) -> Tuple[Scalar[DT], Scalar[DT], Scalar[DT], Int]:
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
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum = Scalar[DT](0.0)
        self._update_count = 0
        return out

    def flush_metrics[
        L: Logger = NoOpLogger
    ](
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
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum = Scalar[DT](0.0)
        self._update_count = 0
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

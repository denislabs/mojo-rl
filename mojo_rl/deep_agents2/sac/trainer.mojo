"""SACTrainer — unified SAC trainer: CPU/GPU × uniform/PER replay.

Replaces the previous three-trainer matrix (SACTrainer / SACTrainerGpu /
SACPerTrainerGpu) with one struct parameterised on:

  - `train_target: StaticString` — "cpu" or "gpu" — kernel dispatch.
    Renamed from `target` (Phase 3.5) to disambiguate from the env's
    `ENV_TARGET` introduced by the dual-target off-policy driver.
  - `SAMPLE: SampleBlock` — replay-buffer-owning block, picks uniform
                             vs PER vs (future) N-step / sequence

Single pipeline body in `_train_step_impl[train_target]`. Single
`make[train_target]` factory using the matmul-stdlib
`Optional[DeviceContext]` idiom. Block choices made at type-instantiation
time → zero runtime branches on target or replay kind beyond the
comptime-if already inside each block.

Driver-trait conformance: `OffPolicyAgentGpu` via
`train_step` / `select_action_batched` /
`select_greedy_action` / `record_batch_cpu` /
`record_batch_gpu[_nstep]`. One host-list wrapper (`select_action`) is
kept as a user-facing entry point for smoke tests that bypass the
driver — it stages obs/action through scratch and delegates to
`select_action_batched[1]`. There is no per-target raise guard:
`train_target` is a struct comptime param, so the wrong wrapper can't
be invoked.

Bit-equivalent to the previous SACTrainer when
`SAMPLE = UniformSampleCpuStep` + `train_target = "cpu"` (validated by
the bit-identity gate −169.04118 @ 30k Pendulum seed=42). On GPU,
warmup RNG migrated from CPU `random_float64` (legacy `_gpu` path) to
Philox (batched path), so GPU loss values shift slightly between
pre-Option-B and post-Option-B baselines — still convergent, still
finite.
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
from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.core import Module
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP, Bf16Compute
from mojo_rl.nn2.core.checkpoint import (
    save_state_v2_body, load_state_v2_body,
    save_state_v2_body_gpu, load_state_v2_body_gpu,
)
from mojo_rl.nn2.core.log_bundle import log_bundle
from mojo_rl.nn2.core.map_params import hard_copy_params
from mojo_rl.nn2.core.metric import LogScalar
from ..core.checkpoint_helpers import (
    save_optimizer_v2_body, load_optimizer_v2_body,
    save_optimizer_v2_body_gpu, load_optimizer_v2_body_gpu,
    save_scalar_adam_v2_body, load_scalar_adam_v2_body,
    save_scalar_adam_v2_body_gpu, load_scalar_adam_v2_body_gpu,
    split_lines_v2, read_file_v2, expect_v2_header,
)
from ..core.online_target_pair import OnlineTargetPair
from mojo_rl.nn2.core.scratch import Scratch
from mojo_rl.nn2.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.optimizer.scalar_adam import ScalarAdam
from ..training.episode_tracker import EpisodeTracker
from ..training.device_mean_accum import DeviceMeanAccum
from .metrics import SACMetrics
from mojo_rl.nn2.training.timer import Timer
from ..training.trainer_block import TrainerState
from ..training.driver_offpolicy import OffPolicyAgentGpu
from ..training.blocks import SampleBlock, TwinCriticStep, PolyakStep
from .blocks.target_y_step import TargetYStep
from .blocks.actor_step import SACActorStep
from .blocks.alpha_update_step import AlphaUpdateStep


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
](OffPolicyAgentGpu):
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
    # so `OffPolicyAgent` (and any future trait that needs to
    # gate on the trainer's compute target) can see it as a member.
    # Distinct from the env's ENV_TARGET — see driver_offpolicy docs.
    comptime AGENT_TRAIN_TARGET: StaticString = Self.train_target

    # Timer section indices. Order matches `add_section` calls in `make`.
    comptime _T_SAMPLE = 0
    comptime _T_TARGET_Y = 1
    comptime _T_CRITIC = 2
    comptime _T_ACTOR = 3
    comptime _T_ALPHA = 4
    comptime _T_POLYAK = 5
    comptime _T_DIAG = 6

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
    var _q_accum: Scalar[DT]
    var _target_accum: Scalar[DT]
    var _reward_accum: Scalar[DT]
    var _next_q_accum: Scalar[DT]
    var _done_accum: Scalar[DT]
    var _abs_action_accum: Scalar[DT]
    var _update_count: Int
    # Never reset by `flush_*` — emitted as `train_steps` so the
    # downstream monitor can plot cumulative updates over time.
    var _total_train_steps: Int

    # GPU-only device-resident mean accumulators for the per-batch diags
    # (the CPU path uses the `_*_accum` host scalars above). Default-
    # constructed on CPU (no device buffer); made on GPU in `make`.
    var _q_mean_dev: DeviceMeanAccum
    var _reward_mean_dev: DeviceMeanAccum
    var _target_mean_dev: DeviceMeanAccum
    var _next_q_mean_dev: DeviceMeanAccum
    var _done_mean_dev: DeviceMeanAccum
    var _abs_action_mean_dev: DeviceMeanAccum

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
        self._q_accum = Scalar[DT](0.0)
        self._target_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._next_q_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        self._abs_action_accum = Scalar[DT](0.0)
        self._update_count = 0
        self._total_train_steps = 0
        self._q_mean_dev = DeviceMeanAccum()
        self._reward_mean_dev = DeviceMeanAccum()
        self._target_mean_dev = DeviceMeanAccum()
        self._next_q_mean_dev = DeviceMeanAccum()
        self._done_mean_dev = DeviceMeanAccum()
        self._abs_action_mean_dev = DeviceMeanAccum()
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
        # Slice 4b — on GPU the entropy temperature lives in a device buffer
        # (state_dev) updated by a 1-thread kernel; on CPU it stays a host
        # scalar (bit-identity path). `flog(init_alpha)` seeds log_α.
        comptime if Self.train_target == "gpu":
            t.alpha_opt = ScalarAdam.new_device(
                ctx.value(), flog(init_alpha), alpha_lr,
            )
            # Device-resident mean accumulators for the GPU diag path.
            t._q_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._reward_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._target_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._next_q_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._done_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._abs_action_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
        else:
            t.alpha_opt = ScalarAdam.new(flog(init_alpha), alpha_lr)

        # Slice 4d — one-time wiring of the device α buffer into both
        # Scale nodes that consume α (target-y soft-V and actor-loss
        # α·log_prob). After this, neither block bakes α as a per-step
        # host scalar; both read it on-device, and the device ScalarAdam
        # refreshes it each step. The pointer is into alpha_opt's device
        # allocation (stable across the struct move on `return t^`).
        comptime if Self.train_target == "gpu":
            var alpha_p = t.alpha_opt.alpha_dev_ptr()
            t.target_y_blk.set_alpha_ptr(alpha_p)
            t.actor_blk.set_alpha_ptr(alpha_p)

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
        t.timer.add_section("diag")
        return t^

    def set_beta(mut self, beta: Scalar[DT]):
        """PER IS-β anneal hook (callers ramp 0.4 → 1.0). No-op for
        uniform sample blocks."""
        self.sample_blk.set_beta(beta)

    # ─── Internal parametric core ─────────────────────────────────────

    def _train_step_impl[
        POLICY: AMPPolicy = NoAMP,
    ](mut self, step_idx: Int) raises -> Bool:
        """Single pipeline body shared across CPU/GPU and uniform/PER.
        Replay-specific behavior lives in `self.sample_blk` (which
        block-internally branches via state.has_per + handles its own
        target). All other blocks parametric on `[Self.train_target, POLICY]`."""
        self.state.step_idx = step_idx
        self.state.did_step = True
        # CPU bakes the host α scalar into the target-y / actor Scale nodes
        # per step. On GPU α lives on-device (wired once at make) and is
        # refreshed by the device ScalarAdam; `state.alpha` is unused there.
        comptime if Self.train_target == "cpu":
            self.state.alpha = fexp(self.alpha_opt.value)
        else:
            self.state.ctx = self.ctx

        var t_sample = perf_counter_ns()
        self.sample_blk.step(self.state)
        if not self.state.did_step:
            return False
        self.timer.accumulate(Self._T_SAMPLE, t_sample)

        # Post-sample kernel sequence (target_y → critic → actor → alpha →
        # polyak → PER tail). Shared verbatim with the CUDA-graph capture
        # path (`train_device_kernels`) so captured == non-captured kernels.
        self._train_post_sample_kernels[POLICY]()

        # Host bookkeeping (counters + metric accumulators). Shared with the
        # capture path, where the driver calls it once per replayed update.
        self.note_train_update()

        # Per-batch diagnostic means — matches the GPU-SAC legacy bundle
        # at offpolicy_agent.mojo:1958-1976. CPU-only: GPU train_target
        # would need D2H copies of the mb_* scratches; deferred.
        #
        # `mean_q` reads `twin_critic_blk.inner.c1._mb_q`, the Q1(s, a)
        # batch output populated by critic.forward inside
        # `twin_critic_blk.step` and NOT overwritten by `actor_blk.step`
        # (the actor loss has its own internal Q scratch).
        #
        # Timed under `_T_DIAG` so `flush_timer_log` reports the exact
        # cost of the diag walk — useful for sizing the trade-off when
        # the user disables some accumulators or raises diag_every.
        var t_diag = perf_counter_ns()
        comptime if Self.train_target == "cpu":
            var inv_b: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](Self.BATCH)
            var y_p = self.state.mb_y.target_ptr["cpu"]()
            var r_p = self.state.mb_r.target_ptr["cpu"]()
            var d_p = self.state.mb_d.target_ptr["cpu"]()
            var a_p = self.state.mb_a.target_ptr["cpu"]()
            var q_p = self.twin_critic_blk.inner.c1._mb_q.target_ptr["cpu"]()
            # `mean_next_q` reads the `min_q` intermediate of the target-y
            # ComputeGraph — min(Q1_t, Q2_t)(s', a') over the batch, the
            # value the TD bootstrap is built from (matches the legacy
            # GPU-SAC bundle: min of the two target-critic next-Q's). The
            # node output is stable after `target_y_blk.step`'s forward,
            # which ran earlier this train_step.
            var nq_p = self.target_y_blk.inner.graph.node_out_ptr["min_q"]()
            var sum_y: Scalar[DT] = 0.0
            var sum_r: Scalar[DT] = 0.0
            var sum_d: Scalar[DT] = 0.0
            var sum_q: Scalar[DT] = 0.0
            var sum_nq: Scalar[DT] = 0.0
            for i in range(Self.BATCH):
                sum_y += y_p[i]
                sum_r += r_p[i]
                sum_d += d_p[i]
                sum_q += q_p[i]
                sum_nq += nq_p[i]
            var sum_a: Scalar[DT] = 0.0
            for i in range(Self.BATCH * Self.ACT_DIM):
                var av = a_p[i]
                sum_a += av if av >= Scalar[DT](0.0) else -av
            self._q_accum += sum_q * inv_b
            self._target_accum += sum_y * inv_b
            self._reward_accum += sum_r * inv_b
            self._next_q_accum += sum_nq * inv_b
            self._done_accum += sum_d * inv_b
            self._abs_action_accum += sum_a * (
                Scalar[DT](1.0) / Scalar[DT](Self.BATCH * Self.ACT_DIM)
            )
        self.timer.accumulate(Self._T_DIAG, t_diag)
        return True

    # ─── Shared post-sample kernel sequence ───────────────────────────
    #
    # target_y → twin-critic (ACCUMULATE on GPU) → actor → alpha → polyak →
    # PER tail. Called by BOTH `_train_step_impl` (non-captured) and
    # `train_device_kernels` (the CUDA-graph capture closure body), so the
    # two paths enqueue an identical kernel sequence — bit-identity by
    # construction. The `perf_counter_ns` timers are host-only: harmless
    # during a capture run (not a kernel, so not recorded) and simply
    # don't fire on replay.
    def _train_post_sample_kernels[
        POLICY: AMPPolicy = NoAMP,
    ](mut self) raises:
        var t_ty = perf_counter_ns()
        self.target_y_blk.step[Self.train_target, POLICY](
            self.state,
            self.actor,
            self.pair1.target_net,
            self.pair2.target_net,
        )
        self.timer.accumulate(Self._T_TARGET_Y, t_ty)

        var t_crit = perf_counter_ns()
        # Slice 3 — on GPU, accumulate critic loss on-device (no per-step
        # D2H); the metric is read at flush. CPU keeps the live-scalar path.
        self.twin_critic_blk.step[
            Self.train_target, POLICY, ACCUMULATE = Self.train_target == "gpu"
        ](
            self.state,
            self.pair1.online,
            self.critic1_opt,
            self.pair2.online,
            self.critic2_opt,
        )
        self.timer.accumulate(Self._T_CRITIC, t_crit)

        # GPU per-batch diag (Q1 / reward / target / min_q / done / |action|)
        # is NOT folded in here. Those six device reductions used to run every
        # update — pure launch overhead on the eager path, and six extra nodes
        # baked into the CUDA-graph capture. They are now taken as a single
        # snapshot of the last update's device buffers inside `flush_metrics`
        # (`_accumulate_diag_snapshot_gpu`), at the `diag_every` cadence only.
        # The CPU path still sums the host scratches per step in
        # `_train_step_impl` (cheap, no launch overhead). Critic/actor loss
        # accumulators are untouched — they fold into the existing loss kernels
        # (ACCUMULATE=True), so they cost no extra launch.

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
        # CPU: host-scalar grad from state.log_prob_mean. GPU: the device
        # ScalarAdam reads the actor-loss `lp_mean` device buffer directly
        # (no D2H) and refreshes the device α the Scale nodes read.
        comptime if Self.train_target == "cpu":
            self.alpha_blk.step["cpu"](self.state, self.alpha_opt)
        else:
            self.alpha_blk.step["gpu"](
                self.state,
                self.alpha_opt,
                self.actor_blk.lp_mean_dev_ptr(),
                self.ctx,
            )
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

    # ─── Host bookkeeping (counters + metric accumulators) ────────────
    #
    # One logical update's worth of host accounting. Called by
    # `_train_step_impl` and — on the capture path — by the driver once per
    # replayed update, so `_total_train_steps` / `n_updates` stay correct
    # whether the device work ran directly or via graph replay. On GPU the
    # loss/α accumulators are unused (flush reads device accumulators); the
    # `+= state.actor_loss` etc. are harmless `+= 0` sentinels there.
    def note_train_update(mut self):
        self._actor_L_accum += self.state.actor_loss
        self._critic_L_accum += self.state.critic_loss
        comptime if Self.train_target == "cpu":
            self._alpha_accum += fexp(self.alpha_opt.value)
        self._update_count += 1
        self._total_train_steps += 1

    # ─── CUDA-graph capture surface (Slice 7) ─────────────────────────
    #
    # `train_device_kernels` is the pure device-kernel train step — sampling
    # (device RNG → fresh minibatch each replay) + the shared post-sample
    # sequence, with NO host gate, NO counters. It is the body of the
    # capture closure passed to `maybe_capture_replay`. GPU-only; the caller
    # gates on `learning_starts_count()` (buffer ready) and advances host
    # counters via `note_train_update()`.
    def _train_device_kernels_impl[
        POLICY: AMPPolicy = NoAMP,
    ](mut self) raises:
        # The caller (driver) only invokes this once past warmup, when the
        # replay buffer is ready. Pin `state.step_idx = learning_starts` so
        # the sample block's `step_idx < learning_starts` warmup gate passes
        # (this method has no step_idx of its own); the buffer-size gate is
        # satisfied because the driver requires learning_starts >= BATCH for
        # the capture path, and by then count == cumulative_step >= BATCH.
        self.state.step_idx = self.learning_starts
        self.state.did_step = True
        self.state.ctx = self.ctx
        self.sample_blk.step(self.state)
        self._train_post_sample_kernels[POLICY]()

    def train_device_kernels(mut self) raises:
        comptime assert Self.train_target == "gpu", (
            "train_device_kernels is GPU-only (CUDA-graph capture path)"
        )
        if self._use_bf16:
            self._train_device_kernels_impl[Bf16Compute]()
        else:
            self._train_device_kernels_impl[NoAMP]()

    def learning_starts_count(self) -> Int:
        """Cumulative env-step threshold after which the replay buffer is
        warm enough to train — the driver gates the capture path on this."""
        return self.learning_starts

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

    # ─── Tier-2 — batched CPU record (no tracker update) ─────────────
    #
    # The batched-CPU driver maintains per-env return accumulators on
    # the host and pushes complete returns via `add_complete_return`.
    # `record_batch_cpu` is the pure-replay-push counterpart of
    # `_record_impl` minus the tracker.add_reward call — without this,
    # batched mode would conflate rewards across all N envs into the
    # single-env tracker's `current_return`.
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
        ), "record_batch_cpu: trainer's train_target must be 'cpu'"
        comptime OBS = Self.OBS_DIM
        comptime ACT = Self.ACT_DIM
        # Per-lane Lists for sample_blk.add (which takes List args).
        # Re-using the same Lists across lanes avoids re-allocation.
        var obs_lane = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
        var act_lane = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
        var nxt_lane = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
        for env_idx in range(N_ENVS):
            for d in range(OBS):
                obs_lane[d] = prev_obs_ptr[env_idx * OBS + d]
                nxt_lane[d] = next_obs_ptr[env_idx * OBS + d]
            for j in range(ACT):
                act_lane[j] = action_ptr[env_idx * ACT + j]
            self.sample_blk.add(
                obs_lane,
                act_lane,
                reward_ptr[env_idx],
                nxt_lane,
                done_ptr[env_idx],
                ctx=self.ctx,
            )

    # ─── Direct-callable (host-list) surface ─────────────────────────
    #
    # `select_action` is the user-facing host-list entry point for
    # smoke tests that bypass the driver. It stages obs into the
    # trainer's `_ob1` / `_ao1` / `_alp1` scratches and delegates to
    # `select_action_batched[1]`. The trainer is the single source of
    # truth on its `train_target` comptime param; there's no per-target
    # raise guard to enforce because the wrong target literally can't
    # be constructed.

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        """Host-list wrapper. Stages obs into `_ob1` (H2D if GPU),
        delegates to `select_action_batched[1]` which writes the
        clamped action into the first ACT_DIM scalars of `_alp1`
        (action_ptr aliased with alp_scratch_ptr — safe at N=1, both
        warmup and policy paths write before reading per-element).
        On GPU, D2H the action back through `_alp1.cpu_ptr()`."""
        var ob1_cpu_p = self._ob1.cpu_ptr()
        for d in range(Self.OBS_DIM):
            ob1_cpu_p[d] = obs[d]
        comptime if Self.train_target == "gpu":
            self.ctx.value().enqueue_copy(self._ob1.dev.value(), ob1_cpu_p)
        self.select_action_batched[1](
            self._ob1.target_ptr[Self.train_target](),
            self._alp1.target_ptr[Self.train_target](),
            self._ao1.target_ptr[Self.train_target](),
            self._alp1.target_ptr[Self.train_target](),
            step_idx,
        )
        comptime if Self.train_target == "gpu":
            var ctx = self.ctx.value()
            ctx.enqueue_copy(self._alp1.cpu_ptr(), self._alp1.dev.value())
            ctx.synchronize()
        var alp1_cpu_p = self._alp1.cpu_ptr()
        for j in range(Self.ACT_DIM):
            action_out[j] = alp1_cpu_p[j]

    # ─── train_step — target-agnostic ─────────────────────────────────
    #
    # Picks the right AMP policy (NoAMP / Bf16Compute) and dispatches
    # to `_train_step_impl[POLICY]`. Single entry point for the
    # off-policy driver and for direct smoke-test callers.
    def train_step(mut self, step_idx: Int) raises -> Bool:
        comptime if Self.train_target == "cpu":
            return self._train_step_impl[NoAMP](step_idx)
        else:
            if self._use_bf16:
                return self._train_step_impl[Bf16Compute](step_idx)
            return self._train_step_impl[NoAMP](step_idx)

    # ─── Greedy eval — target-agnostic ───────────────────────────────
    #
    # Single host-list greedy path; comptime-branches on
    # `Self.train_target`. CPU runs native; GPU uploads obs, forwards
    # the actor on device, downloads the mean, applies tanh+clamp on
    # host. Used by `run_offpolicy_eval` and direct smoke-test callers.
    def select_greedy_action(
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

    # ─── select_action_batched — single batched entry point ──────────
    #
    # The single entry point for all (target, N_ENVS) combinations.
    # The CPU/GPU split happens via `Self.train_target` (the struct
    # comptime), and N_ENVS rolls through transparently for both
    # warmup and policy paths.
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
    def select_action_batched[
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
        # CPU N_ENVS>1 was gated off in Phase 1 (no caller existed);
        # Tier-2's `run_offpolicy_train_batched_cpu_env` is now that
        # caller, so the assert is removed. The CPU body (warmup loop,
        # actor.forward[cpu, N_ENVS], clamp loop) is already
        # N_ENVS-parametric and was never CPU-N=1-specific.

        comptime ACT = Self.ACT_DIM
        comptime OBS = Self.OBS_DIM

        # ── Warmup: uniform random action in [-action_scale, +scale].
        if step_idx < self.learning_starts:
            comptime if Self.train_target == "cpu":
                # CPU warmup: random_float64 lane-by-lane. At N_ENVS=1
                # this consumes exactly ACT random_float64 draws in the
                # same order the legacy single-env CPU path consumed,
                # preserving SAC CPU bit-identity for the host-list
                # `select_action` wrapper that now delegates here.
                for i in range(N_ENVS * ACT):
                    var u = Scalar[DT](2.0 * random_float64() - 1.0)
                    action_ptr[i] = u * self.action_scale
            else:
                # GPU warmup: Philox kernel, bumps _warmup_rng_offset by
                # 2 draws per lane.
                var action_lt = LayoutTensor[
                    DT,
                    Layout.row_major(N_ENVS, ACT),
                    MutAnyOrigin,
                ](action_ptr)
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

    def total_train_steps(self) -> Int:
        """Cumulative training updates since trainer was made. Not reset
        by `flush_*`. Used as the `train_steps` metric and by external
        schedulers."""
        return self._total_train_steps

    # ─── Per-batch diag snapshot (GPU) ────────────────────────────────
    #
    # Fold the LAST completed update's device minibatch into the six per-batch
    # diag accumulators (Q1 / reward / target / min_q / done / |action|).
    # Relocated out of `_train_post_sample_kernels` so these six reductions
    # neither launch every eager update nor get captured into the CUDA graph —
    # they run once per `diag_every` flush instead. Reading the post-update
    # device buffers yields a single-batch snapshot (the legacy GPU-SAC bundle
    # was likewise a single-batch readback). The accumulators are reset
    # alongside the others at the end of `flush_metrics`, so each flush reports
    # exactly this snapshot.
    def _accumulate_diag_snapshot_gpu(mut self) raises:
        comptime assert Self.train_target == "gpu", (
            "_accumulate_diag_snapshot_gpu is GPU-only"
        )
        var t_diag_gpu = perf_counter_ns()
        var q_ptr = self.twin_critic_blk.inner.c1._mb_q.target_ptr["gpu"]()
        var r_ptr = self.state.mb_r.target_ptr["gpu"]()
        var y_ptr = self.state.mb_y.target_ptr["gpu"]()
        var d_ptr = self.state.mb_d.target_ptr["gpu"]()
        var a_ptr = self.state.mb_a.target_ptr["gpu"]()
        # `min_q` is the target-y ComputeGraph's min(Q1_t, Q2_t)(s', a') node;
        # its device out_ptr is fresh from the last update's target_y forward.
        var nq_ptr = self.target_y_blk.inner.graph.node_out_ptr["min_q"]()
        self._q_mean_dev.accumulate_gpu[Self.BATCH](q_ptr)
        self._reward_mean_dev.accumulate_gpu[Self.BATCH](r_ptr)
        self._target_mean_dev.accumulate_gpu[Self.BATCH](y_ptr)
        self._next_q_mean_dev.accumulate_gpu[Self.BATCH](nq_ptr)
        self._done_mean_dev.accumulate_gpu[Self.BATCH](d_ptr)
        self._abs_action_mean_dev.accumulate_gpu_abs[
            Self.BATCH * Self.ACT_DIM
        ](a_ptr)
        self.timer.accumulate(Self._T_DIAG, t_diag_gpu)

    def flush_metrics[
        L: Logger = NoOpLogger
    ](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> SACMetrics:
        """Drain accumulators into a SACMetrics bundle. If a logger
        pointer is wired, also emit one log_scalar per metric field.
        Resets per-chunk accumulators on every call; the cumulative
        `_total_train_steps` counter is NOT reset."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        # Slice 3 — on GPU the critic loss is accumulated on-device by the
        # twin-critic block (no per-step D2H). Read both critics' (sum,count)
        # accumulators here — flush cadence only — and sum them to recover the
        # `loss1+loss2` convention. `read_accum` already divides by its own
        # per-step count (== _update_count), so it returns the chunk mean
        # directly. CPU keeps the host-scalar accumulator path.
        var critic_mean: Scalar[DT]
        # Slice 4 — on GPU the actor loss is accumulated on-device by the
        # actor-loss block (no per-step D2H) and α lives in the device
        # ScalarAdam buffer; read both here at flush cadence. The α metric
        # is the instantaneous device value (α is slowly varying, so this
        # tracks the window mean to logging precision). CPU keeps the
        # host-scalar accumulator paths.
        var actor_mean: Scalar[DT]
        var alpha_val: Scalar[DT]
        # All per-batch diags are device-resident on GPU (Q1 / reward / target
        # / min_q / done / |action| reductions folded in by
        # `_train_post_sample_kernels`); host scalars on CPU.
        var q_mean: Scalar[DT]
        var reward_mean: Scalar[DT]
        var target_mean: Scalar[DT]
        var next_q_mean: Scalar[DT]
        var done_mean: Scalar[DT]
        var abs_action_mean: Scalar[DT]
        comptime if Self.train_target == "gpu":
            # Take the per-batch diag snapshot now (relocated out of the hot
            # loop). Gated on having trained ≥1 update so pre-warmup flushes
            # read empty accumulators (→ 0), matching the prior behavior.
            if self._total_train_steps > 0:
                self._accumulate_diag_snapshot_gpu()
            var cl1 = self.twin_critic_blk.inner.c1.mse_loss.read_accum["gpu"]()
            var cl2 = self.twin_critic_blk.inner.c2.mse_loss.read_accum["gpu"]()
            critic_mean = cl1 + cl2
            actor_mean = self.actor_blk.read_loss_accum()
            alpha_val = self.alpha_opt.read_alpha()
            q_mean = self._q_mean_dev.read["gpu"]()
            reward_mean = self._reward_mean_dev.read["gpu"]()
            target_mean = self._target_mean_dev.read["gpu"]()
            next_q_mean = self._next_q_mean_dev.read["gpu"]()
            done_mean = self._done_mean_dev.read["gpu"]()
            abs_action_mean = self._abs_action_mean_dev.read["gpu"]()
        else:
            critic_mean = self._critic_L_accum * inv
            actor_mean = self._actor_L_accum * inv
            alpha_val = self._alpha_accum * inv
            q_mean = self._q_accum * inv
            reward_mean = self._reward_accum * inv
            target_mean = self._target_accum * inv
            next_q_mean = self._next_q_accum * inv
            done_mean = self._done_accum * inv
            abs_action_mean = self._abs_action_accum * inv
        var bundle = SACMetrics(
            actor_loss=LogScalar[DT](actor_mean),
            critic_loss=LogScalar[DT](critic_mean),
            alpha=LogScalar[DT](alpha_val),
            mean_q=LogScalar[DT](q_mean),
            mean_target=LogScalar[DT](target_mean),
            mean_reward=LogScalar[DT](reward_mean),
            mean_next_q=LogScalar[DT](next_q_mean),
            mean_done=LogScalar[DT](done_mean),
            mean_abs_action=LogScalar[DT](abs_action_mean),
            train_steps=LogScalar[DT](Scalar[DT](self._total_train_steps)),
            n_updates=LogScalar[DT](Scalar[DT](self._update_count)),
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum = Scalar[DT](0.0)
        self._q_accum = Scalar[DT](0.0)
        self._target_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._next_q_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        self._abs_action_accum = Scalar[DT](0.0)
        self._update_count = 0
        # Slices 3+4 — reset the on-device critic + actor accumulators in
        # lock-step with the host counters so the next chunk's mean uses a
        # fresh window. (α has no accumulator — read instantaneously above.)
        comptime if Self.train_target == "gpu":
            self.twin_critic_blk.inner.c1.mse_loss.reset_accum["gpu"]()
            self.twin_critic_blk.inner.c2.mse_loss.reset_accum["gpu"]()
            self.actor_blk.reset_loss_accum()
            self._q_mean_dev.reset["gpu"]()
            self._reward_mean_dev.reset["gpu"]()
            self._target_mean_dev.reset["gpu"]()
            self._next_q_mean_dev.reset["gpu"]()
            self._done_mean_dev.reset["gpu"]()
            self._abs_action_mean_dev.reset["gpu"]()
        if Bool(logger):
            log_bundle(logger.value()[], bundle, step)
        return bundle^

    # ─── Trait-uniform cadence hooks (consumed by the driver) ─────────

    def flush_metrics_through_logger[L: Logger](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        """Trait-uniform passthrough: drains the SAC metric accumulators
        through `flush_metrics` and discards the typed bundle. The
        driver calls this at the user's `diag_every` cadence so no
        chunking is needed."""
        _ = self.flush_metrics[L](logger, step)

    def save_state(mut self, path: String) raises:
        """One-file v2 checkpoint of every SAC module + optimizer.
        Sections: `actor.*`, `critic1.*`, `critic2.*`, `actor_opt.*`,
        `critic1_opt.*`, `critic2_opt.*`, `alpha_opt.*`. Overwrites
        `path`. CPU-only."""
        var body = String("")
        comptime if Self.train_target == "cpu":
            save_state_v2_body(self.actor, body, "actor")
            save_state_v2_body(self.pair1.online, body, "critic1")
            save_state_v2_body(self.pair2.online, body, "critic2")
            save_optimizer_v2_body(self.actor_opt, body, "actor_opt")
            save_optimizer_v2_body(self.critic1_opt, body, "critic1_opt")
            save_optimizer_v2_body(self.critic2_opt, body, "critic2_opt")
            save_scalar_adam_v2_body(self.alpha_opt, body, "alpha_opt")
        else:
            var c = self.ctx.value()
            save_state_v2_body_gpu(self.actor, body, "actor", c)
            save_state_v2_body_gpu(self.pair1.online, body, "critic1", c)
            save_state_v2_body_gpu(self.pair2.online, body, "critic2", c)
            save_optimizer_v2_body_gpu(self.actor_opt, body, "actor_opt")
            save_optimizer_v2_body_gpu(self.critic1_opt, body, "critic1_opt")
            save_optimizer_v2_body_gpu(self.critic2_opt, body, "critic2_opt")
            save_scalar_adam_v2_body_gpu(self.alpha_opt, body, "alpha_opt")
        var content = String("nn2-ckpt v2\n") + body
        with open(path, "w") as f:
            f.write(content)

    def load_state(mut self, path: String) raises:
        """Inverse of `save_state`. Target critics are hard-copied from
        their online twins after the online params are restored."""
        var content = read_file_v2(path)
        var lines = split_lines_v2(content)
        expect_v2_header(lines)
        var idx: Int = 1
        comptime if Self.train_target == "cpu":
            load_state_v2_body(self.actor, lines, idx, "actor")
            load_state_v2_body(self.pair1.online, lines, idx, "critic1")
            load_state_v2_body(self.pair2.online, lines, idx, "critic2")
            load_optimizer_v2_body(self.actor_opt, lines, idx, "actor_opt")
            load_optimizer_v2_body(self.critic1_opt, lines, idx, "critic1_opt")
            load_optimizer_v2_body(self.critic2_opt, lines, idx, "critic2_opt")
            load_scalar_adam_v2_body(self.alpha_opt, lines, idx, "alpha_opt")
        else:
            var c = self.ctx.value()
            load_state_v2_body_gpu(self.actor, lines, idx, "actor", c)
            load_state_v2_body_gpu(self.pair1.online, lines, idx, "critic1", c)
            load_state_v2_body_gpu(self.pair2.online, lines, idx, "critic2", c)
            load_optimizer_v2_body_gpu(self.actor_opt, lines, idx, "actor_opt")
            load_optimizer_v2_body_gpu(
                self.critic1_opt, lines, idx, "critic1_opt"
            )
            load_optimizer_v2_body_gpu(
                self.critic2_opt, lines, idx, "critic2_opt"
            )
            load_scalar_adam_v2_body_gpu(self.alpha_opt, lines, idx, "alpha_opt")
        hard_copy_params[Self.train_target, M=Self.CRITIC](
            self.pair1.online, self.pair1.target_net, self.ctx,
        )
        hard_copy_params[Self.train_target, M=Self.CRITIC](
            self.pair2.online, self.pair2.target_net, self.ctx,
        )

    def flush_timer_log(mut self) -> String:
        """Return a per-section wall-time report (one line per sub-step:
        sample / target_y / critic / actor / alpha / polyak) and reset
        the accumulators."""
        var report = self.timer.format_report()
        self.timer.reset()
        return report

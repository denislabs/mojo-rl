"""SACTrainer — end-to-end SAC continuous trainer for nn2.

Phase 9B. The user-facing trainer for SAC continuous. Owns *everything*
the algorithm needs: networks, optimizers, loss-blocks, replay, tracker,
all forward/backward scratch, logging accumulators, and hyperparameters.

Exposed surface (user-visible API):
    SACTrainer[ACTOR, CRITIC, OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY].make["cpu"](...)
        builds the whole pipeline.
    trainer.select_action(obs_ptr, action_ptr_out, step_idx)
        warmup (uniform) vs policy (squashed-Gaussian) action, in-place.
    trainer.record(obs_ptr, action_ptr, reward, next_obs_ptr, done)
        push transition to replay, accumulate episode reward.
    trainer.end_episode()
        roll episode return into the tracker window.
    trainer.train_step(step_idx) -> Bool
        one full off-policy update if past warmup; no-op otherwise.
        Returns True when a training step actually ran.
    trainer.mean_return() / .ep_count() / .last_actor_loss() / .last_critic_loss()
        logging accessors.

Algorithm (CleanRL-style continuous SAC):
    target y       = r + γ·(min Q1_t(s', a'), Q2_t(s', a') − α·log_prob(a'|s'))
                     where a' = squashed-Gaussian sample from online actor π(·|s')
                     (NO target actor — SAC samples a' from the online policy).
    critic_i loss  = MSE(Q_i(s, a_buffer), y)            (i=1, 2; independent updates)
    actor loss     = mean_b ( α·log_prob(a|s) − min(Q1(s, a), Q2(s, a)) )
                     where (a, log_prob) = squashed-Gaussian rsample of π(·|s)
                     The composed-form chain is in `SACActorLoss` (Phase 9A).
    α update       = ScalarAdam.step( -( mean_b log_prob + target_entropy ) )
    target update  = Polyak τ-soft-update on both critic pairs.

Pendulum truncation gotcha: `done` from the env is the step-200 time-limit
truncation, NOT a real terminal. The target-y compute hard-codes
`nonterm = 1.0` — the value still bootstraps past `done`. See
`feedback_ppo_pendulum_timelimit_gae`. Real-terminal envs (LunarLander,
Hopper, ...) will eventually need a `gymnasium-style terminated vs
truncated` split through this code path; the current API does not yet
expose it.

CPU only in Phase 9B. The GPU path lands when the first GPU SAC env (e.g.
HalfCheetah on physics3d) ships through nn2.

Method boundaries are split to dodge the Mojo nightly inline-call-
explosion trap (memory: `feedback_mojo_function_inline_call_explosion` —
~20 sequential def-raises calls/function ceiling). `train_step` is a
6-call orchestrator over `_train_compute_target_y`, `_train_critic_update`,
`_train_actor_update`, `_train_alpha_update`, `_train_polyak`. Each
sub-method stays under the threshold.
"""

from std.math import exp as fexp, log as flog, pow as fpow, tanh as ftanh
from std.random import random_float64
from std.random.philox import Random as PhiloxRandom
from std.time import perf_counter_ns
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, TileTensor, TensorLayout, row_major

from ..constants import DT
from ..core import Module
from ..core.amp import AMPPolicy, NoAMP, Bf16Compute
from ..core.online_target_pair import OnlineTargetPair
from ..core.scratch import Scratch
from ..core.scratch_walkers import init_scratch_auto
from ..initializer import Xavier
from ..optimizer.adam import Adam
from ..optimizer.scalar_adam import ScalarAdam
from ..loss.sac_actor_loss_cg import SACActorLossCG as SACActorLoss, SACActorLossOut
from ..loss.critic_update_block import TwinCriticUpdateBlock
from ..data.cpu_replay import CPUReplay
from ..data.gpu_replay import GPUReplay
from ..data.n_step_replay import NStepBuffer, NStepTransition, GPUNStepBuffer
from ..data.per_replay import GPUPrioritizedReplay
from ..random.box_muller import box_muller_normal
from mojo_rl.core.logger import Logger, NoOpLogger
from ..core.log_bundle import log_bundle
from ..core.metric import LogScalar
from .episode_tracker import EpisodeTracker
from .sac_config import SACConfig
from .sac_metrics import SACMetrics
from .driver_cpu import (
    OffPolicyTrainable, OffPolicyTrainableGpu, OffPolicyTrainableGpuBatched,
)
from .target_y_block import TargetYBlock
from .timer import Timer


# ──────────────────────────────────────────────────────────────────────
# Phase B.5b — top-level kernels for the batched GPU action path.
# ──────────────────────────────────────────────────────────────────────


def _warmup_uniform_kernel[N_ENVS: Int, ACT: Int](
    action_dest: LayoutTensor[
        DT, Layout.row_major(N_ENVS, ACT), MutAnyOrigin,
    ],
    action_scale: Scalar[DT],
    seed: UInt64,
    offset_base: UInt64,
):
    """Per-lane Philox uniform → `[N_ENVS, ACT]` filled with
    `Uniform(-action_scale, +action_scale)` samples.

    Each `(env, j)` lane gets its own Philox stream seeded by
    `seed + (env * ACT + j)` at offset `offset_base`. Mirrors B.5
    single-env warmup (host `random_float64` × 2 − 1 × action_scale)
    but device-side so no host→device upload is needed.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    var total = N_ENVS * ACT
    if i >= total:
        return
    var philox = PhiloxRandom(seed=seed + UInt64(i), offset=offset_base)
    var u = Float32(philox.step_uniform()[0])  # uniform in [0, 1)
    var s = Scalar[DT](2.0) * Scalar[DT](u) - Scalar[DT](1.0)
    var env = i // ACT
    var j = i % ACT
    action_dest[env, j] = s * action_scale


def _td_abs_kernel[BATCH: Int](
    q: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    y: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    td_out: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
):
    """Per-lane TD-error magnitude: `td_out[i] = |q[i,0] - y[i,0]|`.

    Phase C.3b. Used after the critic update to refresh PER priorities.
    `q` is the pre-update Q1 prediction (still resident in c1._mb_q
    from the just-finished forward pass) and `y` is the target value
    computed by `target_y_block`. Both are device buffers.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    var diff = q[i, 0] - y[i, 0]
    if diff < Scalar[DT](0.0):
        diff = -diff
    td_out[i] = diff


def _action_clamp_kernel[N_ENVS: Int, ACT: Int](
    alp: LayoutTensor[
        DT, Layout.row_major(N_ENVS, ACT + 1), MutAnyOrigin,
    ],
    action_out: LayoutTensor[
        DT, Layout.row_major(N_ENVS, ACT), MutAnyOrigin,
    ],
    action_scale: Scalar[DT],
):
    """Extract the first `ACT` lanes from `alp[env, :ACT]` (rsample's
    sampled action — the last lane is `log_prob` which we drop here)
    and write the clamped result into `action_out`.

    Mirrors the host-side `tanh`-then-clamp the single-env GPU path
    runs after the D2H of `_alp1.dev`; in batched mode we keep it on
    device so no per-step download is needed.
    """
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
    ACTOR: Module,
    CRITIC: Module,
    OBS_DIM: Int,
    ACT_DIM: Int,
    BATCH: Int,
    REPLAY_CAPACITY: Int,
    N_STEP: Int = 1,
](OffPolicyTrainable, OffPolicyTrainableGpu, OffPolicyTrainableGpuBatched):
    comptime SA_DIM = Self.OBS_DIM + Self.ACT_DIM

    # Trait-conformance aliases (added 2026-05-25). The struct's
    # parametric params `OBS_DIM` / `ACT_DIM` are the source of truth;
    # these aliases publish them under names that don't clash with the
    # struct param namespace so the driver can access them via the
    # `OffPolicyTrainableGpuBatched` trait surface (`A.AGENT_OBS_DIM`,
    # `A.AGENT_ACT_DIM`).
    comptime AGENT_OBS_DIM: Int = Self.OBS_DIM
    comptime AGENT_ACT_DIM: Int = Self.ACT_DIM

    # Timer section indices. Order matches `add_section` calls in `make`.
    comptime _T_SAMPLE    = 0
    comptime _T_TARGET_Y  = 1
    comptime _T_CRITIC    = 2
    comptime _T_ACTOR     = 3
    comptime _T_ALPHA     = 4
    comptime _T_POLYAK    = 5

    # ─── Networks ─────────────────────────────────────────────────────
    var actor: Self.ACTOR
    var pair1: OnlineTargetPair[Self.CRITIC]
    var pair2: OnlineTargetPair[Self.CRITIC]

    # ─── Optimizers ───────────────────────────────────────────────────
    var actor_opt: Adam
    var critic1_opt: Adam
    var critic2_opt: Adam
    var alpha_opt: ScalarAdam

    # ─── Loss objects ─────────────────────────────────────────────────
    var actor_loss: SACActorLoss[Self.ACTOR, Self.CRITIC, Self.BATCH]
    var twin_critic_block: TwinCriticUpdateBlock[
        Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
    ]
    var target_y_block: TargetYBlock[
        Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
    ]

    # ─── Replay + tracker ─────────────────────────────────────────────
    var buf: CPUReplay[Self.OBS_DIM, Self.ACT_DIM, Self.REPLAY_CAPACITY]
    # Phase C.1 — GPU-resident replay. Populated only by the GPU
    # factory; CPU trainers leave this `None`. When set, `record`
    # routes transitions to the device buffer and `train_step["gpu"]`
    # samples directly into the device minibatch buffers (no CPU
    # sample + 4 host→device uploads).
    var buf_gpu: Optional[
        GPUReplay[Self.OBS_DIM, Self.ACT_DIM, Self.REPLAY_CAPACITY]
    ]
    # Phase C.3b — GPU prioritized replay. Mutually exclusive with
    # `buf_gpu`: when `config.use_per=True` the GPU factory swaps
    # `buf_gpu` for this field. `record` and `train_step["gpu"]`
    # branch on which Optional is `Some`.
    var buf_per: Optional[
        GPUPrioritizedReplay[
            Self.OBS_DIM, Self.ACT_DIM, Self.REPLAY_CAPACITY
        ]
    ]
    # Phase C.3b — device-side TD-error scratch (BATCH lanes). Reused
    # across train_steps so we don't reallocate per-step.
    var _td_err_dev: Optional[DeviceBuffer[DT]]
    # Phase C.2b — host-side n-step buffer. When the trainer was built
    # from a SACConfig with `use_n_step=True` AND the struct's `N_STEP`
    # comptime param > 1, `record` routes the (s, a, r, s', done) stream
    # through this buffer first and pushes only the compressed n-step
    # transitions into `buf` / `buf_gpu`. Single-env path only — the
    # GPU N_ENVS-batched n-step buffer (`GPUNStepBuffer`) lives in the
    # driver, not the trainer, since `N_ENVS` is method-comptime not
    # struct-comptime.
    var nstep_cpu: Optional[
        NStepBuffer[Self.N_STEP, Self.OBS_DIM, Self.ACT_DIM]
    ]
    var _use_nstep: Bool
    var tracker: EpisodeTracker

    # ─── Single-step scratch (env interaction) ────────────────────────
    # Phase G.1 — every single-step scratch is a `Scratch[NAME, SIZE,
    # STAGING=True]`. STAGING keeps a CPU mirror alongside the device
    # buffer on GPU init so host-side obs upload + sampler-out download
    # use the same buffer. CPU-only trainers allocate only the CPU half.
    var _ob1: Scratch["ob1", Self.OBS_DIM, True]
    var _ao1: Scratch["ao1", 2 * Self.ACT_DIM, True]
    var _alp1: Scratch["alp1", Self.ACT_DIM + 1, True]

    # Phase C.2b — host scratch for n-step emitted transitions. When
    # `_use_nstep`, `record` copies the n-step buffer's emitted
    # InlineArray fields into these pre-allocated host pointers, then
    # calls the underlying replay `add` against pointers we own (rather
    # than into InlineArray-internal locals whose origins don't match
    # the `MutAnyOrigin`-typed replay surface).
    #
    # STAGING=True even though these are purely host-side: it guarantees
    # the CPU list is materialised on BOTH CPU and GPU `make` targets.
    # Without it, GPU init only fills the device half and the `record`
    # path would call `.cpu_ptr()` against an empty List. The GPU half
    # allocated here (7 floats total) is harmless overhead.
    var _nstep_obs: Scratch["nstep_obs", Self.OBS_DIM, True]
    var _nstep_act: Scratch["nstep_act", Self.ACT_DIM, True]
    var _nstep_nxt: Scratch["nstep_nxt", Self.OBS_DIM, True]

    # ─── Minibatch scratch (training) — only raw replay samples ──────
    # Target-y compute scratch is owned by `target_y_block` (Phase 10F).
    # Critic update scratch is owned by `twin_critic_block` (Phase 10F).
    # STAGING=True keeps the CPU sample buffer + the GPU minibatch in
    # one field; the GPU path samples directly into the device half.
    var _mb_s: Scratch["mb_s", Self.BATCH * Self.OBS_DIM, True]
    var _mb_a: Scratch["mb_a", Self.BATCH * Self.ACT_DIM, True]
    var _mb_r: Scratch["mb_r", Self.BATCH, True]
    var _mb_sp: Scratch["mb_sp", Self.BATCH * Self.OBS_DIM, True]
    var _mb_d: Scratch["mb_d", Self.BATCH, True]
    var _mb_y: Scratch["mb_y", Self.BATCH, True]

    # ─── Hyperparameters ──────────────────────────────────────────────
    var gamma: Scalar[DT]
    var tau: Scalar[DT]
    var action_scale: Scalar[DT]
    var target_entropy: Scalar[DT]
    var learning_starts: Int

    # Phase C.5b — runtime mirror of `SACConfig.use_bf16`. When True,
    # the non-parametric `train_step_gpu` trait wrapper dispatches to
    # `train_step["gpu", Bf16Compute]` instead of `NoAMP`. CPU path
    # always uses NoAMP regardless of this flag (bf16 is a GPU-only
    # optimization in nn2 today).
    var _use_bf16: Bool

    # ─── Phase B.5b — batched-GPU warmup RNG state ───────────────────
    # `select_action_gpu_batched[N_ENVS]` uses Philox on-device for the
    # uniform warmup actions. Offset bumps by `2*N_ENVS*ACT` per call
    # so back-to-back warmup steps draw disjoint streams; this state
    # is independent from RSample's RNG (which governs policy
    # sampling). Single-env paths don't touch this field.
    var _warmup_rng_seed: UInt64
    var _warmup_rng_offset: UInt64

    # ─── Logging accumulators ─────────────────────────────────────────
    var _actor_L_accum: Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    var _alpha_accum: Scalar[DT]
    var _update_count: Int

    # ─── Wall-time introspection ──────────────────────────────────────
    # Per-sub-step accumulator. On GPU the per-section numbers reflect
    # enqueue overhead, not real kernel time — accurate per-block GPU
    # timing requires `ctx.synchronize()` brackets which would wreck
    # throughput, so we don't insert them by default.
    var timer: Timer

    def __init__(out self):
        self.actor = Self.ACTOR()
        self.pair1 = OnlineTargetPair[Self.CRITIC]()
        self.pair2 = OnlineTargetPair[Self.CRITIC]()
        self.actor_opt = Adam()
        self.critic1_opt = Adam()
        self.critic2_opt = Adam()
        self.alpha_opt = ScalarAdam(
            value=0.0, m=0.0, v=0.0, t=0,
            lr=0.0003, beta1=0.9, beta2=0.999, eps=1e-8,
        )
        self.actor_loss = SACActorLoss[
            Self.ACTOR, Self.CRITIC, Self.BATCH
        ]()
        self.twin_critic_block = TwinCriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
        ]()
        self.target_y_block = TargetYBlock[
            Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
        ]()
        self.buf = CPUReplay[
            Self.OBS_DIM, Self.ACT_DIM, Self.REPLAY_CAPACITY
        ](
            obs=UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
            act=UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
            rew=UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
            nxt=UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
            dne=UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
            size=0, pos=0,
        )
        self.buf_gpu = None
        self.buf_per = None
        self._td_err_dev = None
        self.nstep_cpu = None
        self._use_nstep = False
        self.tracker = EpisodeTracker(
            window=List[Scalar[DT]](),
            window_size=0, idx=0,
            current_return=Scalar[DT](0.0), ep_count=0,
        )
        # Phase G.1 — Scratch defaults give empty cpu list + None dev.
        # `init_scratch_auto[Self, target]` in `make[target]` populates
        # the matching storage (STAGING=True scratches also keep the CPU
        # mirror on GPU init for upload/download bookkeeping).
        self._ob1 = Scratch["ob1", Self.OBS_DIM, True]()
        self._ao1 = Scratch["ao1", 2 * Self.ACT_DIM, True]()
        self._alp1 = Scratch["alp1", Self.ACT_DIM + 1, True]()
        self._nstep_obs = Scratch["nstep_obs", Self.OBS_DIM, True]()
        self._nstep_act = Scratch["nstep_act", Self.ACT_DIM, True]()
        self._nstep_nxt = Scratch["nstep_nxt", Self.OBS_DIM, True]()
        self._mb_s = Scratch["mb_s", Self.BATCH * Self.OBS_DIM, True]()
        self._mb_a = Scratch["mb_a", Self.BATCH * Self.ACT_DIM, True]()
        self._mb_r = Scratch["mb_r", Self.BATCH, True]()
        self._mb_sp = Scratch["mb_sp", Self.BATCH * Self.OBS_DIM, True]()
        self._mb_d = Scratch["mb_d", Self.BATCH, True]()
        self._mb_y = Scratch["mb_y", Self.BATCH, True]()
        self.gamma = Scalar[DT](0.99)
        self.tau = Scalar[DT](0.005)
        self.action_scale = Scalar[DT](1.0)
        self.target_entropy = Scalar[DT](-1.0)
        self.learning_starts = 1_000
        self._use_bf16 = False
        self._warmup_rng_seed = UInt64(0xBADBEEF_FEEDFACE)
        self._warmup_rng_offset = UInt64(0)
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum = Scalar[DT](0.0)
        self._update_count = 0
        self.timer = Timer.new()

    @staticmethod
    def _maybe_enable_nstep(mut t: Self, config: SACConfig) raises:
        """Phase C.2b — common n-step opt-in path shared by CPU + GPU
        config factories. Comptime-gated on `N_STEP > 1` (the runtime
        if collapses since `N_STEP` is a struct comptime param), so
        the N_STEP=1 default path remains zero-cost and bit-identical.

        Effects when enabled:
            - Allocate `nstep_cpu = NStepBuffer[N_STEP, OBS, ACT]`
              with `gamma = config.gamma.v` (per-step γ, not γ^N).
            - Re-bake `γ^N_STEP` into target_y_block's `gamma_softv`
              Scale node so the critic bootstrap matches the n-step
              return (rather than the per-step γ baked at the keyword
              factory).

        Phase G.1: the `nstep_obs/act/nxt` host scratches are now
        unconditionally allocated by `init_scratch_auto` in `make[target]`
        (zero-filled CPU lists), so the `record` path can read/write
        through `cpu_ptr()` regardless of whether `_use_nstep` flips.
        """
        if config.use_n_step.v and Self.N_STEP > 1:
            t.nstep_cpu = NStepBuffer[
                Self.N_STEP, Self.OBS_DIM, Self.ACT_DIM
            ].new(gamma=config.gamma.v)
            t._use_nstep = True
            var gamma_n = Scalar[DT](
                fpow(Float64(config.gamma.v), Float64(Self.N_STEP))
            )
            t.target_y_block.graph.set_node_attr[
                "gamma_softv", "multiplier"
            ](gamma_n)

    @staticmethod
    def _init_timer(mut t: Self) raises:
        """Register the 6 standard train_step sections in declaration order.
        Index order MUST match the `_T_*` comptime constants on the struct."""
        t.timer.add_section("sample")
        t.timer.add_section("target_y")
        t.timer.add_section("critic")
        t.timer.add_section("actor")
        t.timer.add_section("alpha")
        t.timer.add_section("polyak")

    @staticmethod
    def make[target: StaticString](config: SACConfig) raises -> Self:
        """Phase A.4 — Config-driven CPU factory. Unpacks the Config
        and forwards to the existing keyword path so both routes share
        the same construction code (bit-identical by construction).

        Phase C.5b: `config.use_bf16` is forwarded but the CPU factory
        ignores it — bf16 mixed-precision is a GPU-only knob in nn2
        today (CPU forward/vjp kernels have no bf16 path).

        Phase C.2b: when `config.use_n_step=True` AND the struct's
        `N_STEP` comptime param > 1, allocates an n-step ring buffer
        and re-bakes `γ^N_STEP` into target-y's bootstrap multiplier.
        With either flag off (default), the trainer is bit-identical
        to pre-C.2b."""
        var t = Self.make[target](
            actor_lr=config.actor_lr.v,
            critic_lr=config.critic_lr.v,
            alpha_lr=config.alpha_lr.v,
            gamma=config.gamma.v,
            tau=config.tau.v,
            action_scale=config.action_scale.v,
            init_alpha=config.init_alpha.v,
            target_entropy=config.target_entropy.v,
            learning_starts=config.learning_starts.v,
            window_size=config.window_size.v,
            initial_episode_fill=config.initial_episode_fill.v,
            max_grad_norm=config.max_grad_norm.v,
        )
        t._use_bf16 = False  # CPU path never routes to bf16
        Self._maybe_enable_nstep(t, config)
        return t^

    @staticmethod
    def make[target: StaticString](
        ctx: DeviceContext, config: SACConfig,
    ) raises -> Self:
        """Phase A.4 — Config-driven GPU factory.

        Phase C.5b: when `config.use_bf16 == True`, the trainer's
        non-parametric `train_step_gpu` wrapper routes through
        `POLICY=Bf16Compute`. Default False → `NoAMP` → bit-identical
        to pre-C.5.

        Phase C.4b: when `config.use_ere == True`, flips the device-
        resident GPUReplay into ERE recency-biased sampling mode by
        calling `buf_gpu.enable_ere(eta, c_min, k_max)` with the
        SACConfig hyperparameters. Default `use_ere=False` is a no-op
        (uniform sampler) → bit-identical to pre-C.4b."""
        var t = Self.make[target](
            ctx,
            actor_lr=config.actor_lr.v,
            critic_lr=config.critic_lr.v,
            alpha_lr=config.alpha_lr.v,
            gamma=config.gamma.v,
            tau=config.tau.v,
            action_scale=config.action_scale.v,
            init_alpha=config.init_alpha.v,
            target_entropy=config.target_entropy.v,
            learning_starts=config.learning_starts.v,
            window_size=config.window_size.v,
            initial_episode_fill=config.initial_episode_fill.v,
            max_grad_norm=config.max_grad_norm.v,
        )
        t._use_bf16 = config.use_bf16.v
        # Phase C.3b — PER overrides ERE: PER's priority-weighted
        # sampling supersedes ERE's recency bias. If both flags are
        # set, we honor PER and silently skip ERE.
        if config.use_per.v:
            t.buf_per = GPUPrioritizedReplay[
                Self.OBS_DIM, Self.ACT_DIM, Self.REPLAY_CAPACITY
            ].new(
                ctx,
                alpha=config.per_alpha.v,
                beta=config.per_beta.v,
                epsilon=config.per_epsilon.v,
                batch_capacity=Self.BATCH,
            )
            t.buf_gpu = None
            t._td_err_dev = ctx.enqueue_create_buffer[DT](Self.BATCH)
        elif config.use_ere.v:
            t.buf_gpu.value().enable_ere(
                eta=config.ere_eta.v,
                c_min=config.ere_c_min.v,
                k_max=config.ere_k_max.v,
            )
        Self._maybe_enable_nstep(t, config)
        return t^

    @staticmethod
    def make[target: StaticString](
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
    ) raises -> Self:
        """CPU factory. Builds nets via Xavier init, 3 Adams + 1 ScalarAdam,
        the Phase 9A actor-loss block, MSE loss, replay buffer, and tracker.
        Allocates all training scratch up front (no per-step allocation).

        `max_grad_norm` (Phase B.3): global L2 grad-norm clip applied to
        the actor + both critic optimizers. Default 0.0 → disabled
        (bit-identical to pre-B.3 behaviour)."""
        comptime assert target == "cpu", (
            "SACTrainer.make[target='gpu'] requires a DeviceContext"
        )
        var t = Self()
        t.actor = Self.ACTOR.make[target="cpu", INIT=Xavier]()
        t.pair1 = OnlineTargetPair[Self.CRITIC].make[
            target="cpu", INIT=Xavier
        ]()
        t.pair2 = OnlineTargetPair[Self.CRITIC].make[
            target="cpu", INIT=Xavier
        ]()
        t.actor_opt = Adam.make[target="cpu", M=Self.ACTOR](t.actor)
        t.actor_opt.lr = actor_lr
        t.actor_opt.max_grad_norm = max_grad_norm
        t.critic1_opt = Adam.make[target="cpu", M=Self.CRITIC](t.pair1.online)
        t.critic1_opt.lr = critic_lr
        t.critic1_opt.max_grad_norm = max_grad_norm
        t.critic2_opt = Adam.make[target="cpu", M=Self.CRITIC](t.pair2.online)
        t.critic2_opt.lr = critic_lr
        t.critic2_opt.max_grad_norm = max_grad_norm
        t.alpha_opt = ScalarAdam.new(flog(init_alpha), alpha_lr)
        t.actor_loss = SACActorLoss[
            Self.ACTOR, Self.CRITIC, Self.BATCH
        ].make["cpu"](action_scale=action_scale)
        t.twin_critic_block = TwinCriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
        ].make["cpu"]()
        t.target_y_block = TargetYBlock[
            Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
        ].make["cpu"](action_scale=action_scale, gamma=gamma)
        t.buf = CPUReplay[
            Self.OBS_DIM, Self.ACT_DIM, Self.REPLAY_CAPACITY
        ].new()
        t.tracker = EpisodeTracker.new(
            window_size=window_size, initial_fill=initial_episode_fill
        )

        # Phase G.1 — one walker call replaces 9 raw `alloc` lines.
        # Every `Scratch[NAME, SIZE, STAGING]` field on the struct gets
        # its CPU list materialized (zero-filled). STAGING-flagged
        # scratches don't allocate the GPU half here (we're CPU-only).
        init_scratch_auto[Self, target="cpu"](t)

        t.gamma = gamma
        t.tau = tau
        t.action_scale = action_scale
        t.target_entropy = target_entropy
        t.learning_starts = learning_starts
        Self._init_timer(t)
        return t^

    @staticmethod
    def make[target: StaticString](
        ctx: DeviceContext,
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
    ) raises -> Self:
        """GPU factory. Builds GPU nets + Adams, GPU-allocated critic
        update block, GPU sub-allocated SACActorLoss + TargetYBlock.

        **Important**: `train_step["gpu"]` and `select_action["gpu"]`
        currently raise because they depend on GPU box_muller / GPU
        squashed_gaussian / GPU RSample, all of which are Block D scope
        in the roadmap. `make["gpu"]` is exposed so callers can build
        the trainer and stage env-side wiring; full GPU training lands
        once those primitives ship."""
        comptime assert target == "gpu", (
            "SACTrainer.make[target='cpu'](ctx) — drop ctx for CPU"
        )
        var t = Self()
        t.actor = Self.ACTOR.make[target="gpu", INIT=Xavier](ctx)
        t.pair1 = OnlineTargetPair[Self.CRITIC].make[
            target="gpu", INIT=Xavier
        ](ctx)
        t.pair2 = OnlineTargetPair[Self.CRITIC].make[
            target="gpu", INIT=Xavier
        ](ctx)
        t.actor_opt = Adam.make[target="gpu", M=Self.ACTOR](t.actor, ctx)
        t.actor_opt.lr = actor_lr
        t.actor_opt.max_grad_norm = max_grad_norm
        t.critic1_opt = Adam.make[target="gpu", M=Self.CRITIC](
            t.pair1.online, ctx
        )
        t.critic1_opt.lr = critic_lr
        t.critic1_opt.max_grad_norm = max_grad_norm
        t.critic2_opt = Adam.make[target="gpu", M=Self.CRITIC](
            t.pair2.online, ctx
        )
        t.critic2_opt.lr = critic_lr
        t.critic2_opt.max_grad_norm = max_grad_norm
        t.alpha_opt = ScalarAdam.new(flog(init_alpha), alpha_lr)
        t.actor_loss = SACActorLoss[
            Self.ACTOR, Self.CRITIC, Self.BATCH
        ].make["gpu"](ctx, action_scale=action_scale)
        t.twin_critic_block = TwinCriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
        ].make["gpu"](ctx)
        t.target_y_block = TargetYBlock[
            Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
        ].make["gpu"](ctx, action_scale=action_scale, gamma=gamma)
        # Phase C.1 — GPU-resident replay. `buf` stays as the null-
        # pointer-initialised CPUReplay from `__init__` (not allocated)
        # so we don't waste ~1.8MB host memory; `buf_gpu` holds the
        # device-side circular store. `record` + `train_step["gpu"]`
        # route through it.
        t.buf_gpu = GPUReplay[
            Self.OBS_DIM, Self.ACT_DIM, Self.REPLAY_CAPACITY
        ].new(ctx, batch_capacity=Self.BATCH)
        t.tracker = EpisodeTracker.new(
            window_size=window_size, initial_fill=initial_episode_fill
        )

        # Phase G.1 — one walker call replaces 9 raw `alloc` lines + 7
        # `enqueue_create_buffer` lines. Every staging-flagged Scratch
        # field allocates BOTH the CPU mirror (for upload/download
        # bookkeeping) AND the device buffer in one pass; CPU-only
        # scratches (nstep_*) get just the host list.
        #
        # Phase C.1 — `_mb_d` device half exists so the GPU
        # `_gather_batch_kernel` can fill it. SAC's GPU target_y
        # currently ignores done (Pendulum truncation hard-codes
        # nonterm=1.0), but keeping the field future-proofs non-
        # truncation envs and matches the gather kernel's surface.
        init_scratch_auto[Self, target="gpu"](t, Optional[DeviceContext](ctx))

        t.gamma = gamma
        t.tau = tau
        t.action_scale = action_scale
        t.target_entropy = target_entropy
        t.learning_starts = learning_starts
        Self._init_timer(t)
        return t^

    # ──────────────────────────────────────────────────────────────────
    # Env-interaction API
    # ──────────────────────────────────────────────────────────────────

    def select_action[target: StaticString = "cpu"](
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        step_idx: Int,
    ) raises:
        """Sample action into `action_out` ([ACT_DIM]).

        Warmup (step_idx < learning_starts): uniform on [-action_scale, +action_scale].
        Else: actor.forward + rsample.forward at BATCH=1, write the
        squashed-Gaussian sample (no log_prob extracted here).

        Output is clamped to ±action_scale (the env's valid range).

        `obs` and `action_out` are always host pointers; on GPU the
        method internally uploads obs and downloads action."""
        if step_idx < self.learning_starts:
            for j in range(Self.ACT_DIM):
                var u = Scalar[DT](2.0 * random_float64() - 1.0)
                action_out[j] = u * self.action_scale
            return

        var ob1_cpu_p = self._ob1.cpu_ptr()
        var ao1_cpu_p = self._ao1.cpu_ptr()
        var alp1_cpu_p = self._alp1.cpu_ptr()

        comptime if target == "cpu":
            for d in range(Self.OBS_DIM):
                ob1_cpu_p[d] = obs[d]
            var ob1_t = TileTensor(ob1_cpu_p, row_major[1, Self.OBS_DIM]())
            var ao1_t = TileTensor(ao1_cpu_p, row_major[1, 2 * Self.ACT_DIM]())
            self.actor.forward["cpu", 1](ob1_t, output=ao1_t)
            var alp1_t = TileTensor(alp1_cpu_p, row_major[1, Self.ACT_DIM + 1]())
            self.actor_loss.rsample.forward["cpu", 1](ao1_t, output=alp1_t)
            for j in range(Self.ACT_DIM):
                var a = alp1_cpu_p[j]
                if a > self.action_scale:
                    a = self.action_scale
                elif a < -self.action_scale:
                    a = -self.action_scale
                action_out[j] = a
        else:
            var ctx = self.target_y_block.ts.ctx.value()
            # Upload obs via the CPU staging mirror, then run actor +
            # rsample on device, then download the alp result.
            for d in range(Self.OBS_DIM):
                ob1_cpu_p[d] = obs[d]
            ctx.enqueue_copy(self._ob1.dev.value(), ob1_cpu_p)
            var ob1_p = self._ob1.dev_ptr()
            var ao1_p = self._ao1.dev_ptr()
            var alp1_p = self._alp1.dev_ptr()
            var ob1_t = TileTensor(ob1_p, row_major[1, Self.OBS_DIM]())
            var ao1_t = TileTensor(ao1_p, row_major[1, 2 * Self.ACT_DIM]())
            self.actor.forward["gpu", 1](ob1_t, output=ao1_t)
            var alp1_t = TileTensor(alp1_p, row_major[1, Self.ACT_DIM + 1]())
            self.actor_loss.rsample.forward["gpu", 1](ao1_t, output=alp1_t)
            ctx.enqueue_copy(alp1_cpu_p, self._alp1.dev.value())
            ctx.synchronize()
            for j in range(Self.ACT_DIM):
                var a = alp1_cpu_p[j]
                if a > self.action_scale:
                    a = self.action_scale
                elif a < -self.action_scale:
                    a = -self.action_scale
                action_out[j] = a

    def select_action(
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        step_idx: Int,
    ) raises:
        """Non-parametric overload — forwards to the CPU path.

        Required by the `OffPolicyTrainable` trait surface used by
        `driver_cpu.run_offpolicy_train_cpu`. Mojo trait conformance is
        strict: methods with comptime params do NOT match non-parametric
        trait signatures even when a default is provided. This thin
        wrapper closes that gap without changing struct layout."""
        self.select_action["cpu"](obs, action_out, step_idx)

    def select_greedy_action(
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Phase B.2 — deterministic greedy action for eval (CPU).

        SAC's actor outputs [mean, log_std] (2*ACT_DIM). The training
        path samples via rsample (tanh(mean + std·N(0,1))); the greedy
        eval path uses the mode of the squashed Gaussian, which is
        tanh(mean), scaled by `action_scale`. No log_std consumed, no
        Gaussian sample, no log_prob produced — purely deterministic.

        Clamps to ±action_scale as a safety net (tanh already bounds to
        ±1 so the clamp is redundant in the well-behaved case, but
        retained to mirror `select_action`'s contract)."""
        var ob1_cpu_p = self._ob1.cpu_ptr()
        var ao1_cpu_p = self._ao1.cpu_ptr()
        for d in range(Self.OBS_DIM):
            ob1_cpu_p[d] = obs[d]
        var ob1_t = TileTensor(ob1_cpu_p, row_major[1, Self.OBS_DIM]())
        var ao1_t = TileTensor(ao1_cpu_p, row_major[1, 2 * Self.ACT_DIM]())
        self.actor.forward["cpu", 1](ob1_t, output=ao1_t)
        for j in range(Self.ACT_DIM):
            var mean = ao1_cpu_p[j]
            var a = ftanh(mean) * self.action_scale
            if a > self.action_scale:
                a = self.action_scale
            elif a < -self.action_scale:
                a = -self.action_scale
            action_out[j] = a

    # ─── Phase B.5 — non-parametric GPU wrappers (trait conformance) ──
    # Mirror the CPU wrappers but route to ["gpu"]. Required because
    # Mojo trait conformance doesn't accept parametric methods (even
    # with defaults) as matching non-parametric trait signatures. The
    # wrappers are pure pass-throughs.

    def select_action_gpu(
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        step_idx: Int,
    ) raises:
        self.select_action["gpu"](obs, action_out, step_idx)

    def train_step_gpu(mut self, step_idx: Int) raises -> Bool:
        """Phase C.5b — auto-routes through `Bf16Compute` when the
        trainer was built from a `SACConfig` with `use_bf16=True`,
        else falls back to `NoAMP`. The runtime branch compiles both
        `train_step["gpu", NoAMP]` and `train_step["gpu", Bf16Compute]`
        specializations; only one is exercised per call."""
        if self._use_bf16:
            return self.train_step["gpu", Bf16Compute](step_idx)
        return self.train_step["gpu", NoAMP](step_idx)

    def select_greedy_action_gpu(
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Phase B.5 — deterministic greedy action for eval (GPU).

        Same math as `select_greedy_action` (tanh(mean) * action_scale)
        but executes the actor forward against device buffers. Upload
        obs → device, forward, download head, apply tanh + clamp on
        host. Single-step path is small (BATCH=1), so the launch
        overhead dominates over the actual compute — host-side tanh on
        ACT_DIM values is cheap and avoids a kernel launch."""
        var ctx = self.target_y_block.ts.ctx.value()
        var ob1_cpu_p = self._ob1.cpu_ptr()
        var ao1_cpu_p = self._ao1.cpu_ptr()
        for d in range(Self.OBS_DIM):
            ob1_cpu_p[d] = obs[d]
        ctx.enqueue_copy(self._ob1.dev.value(), ob1_cpu_p)
        var ob1_p = self._ob1.dev_ptr()
        var ao1_p = self._ao1.dev_ptr()
        var ob1_t = TileTensor(ob1_p, row_major[1, Self.OBS_DIM]())
        var ao1_t = TileTensor(ao1_p, row_major[1, 2 * Self.ACT_DIM]())
        self.actor.forward["gpu", 1](ob1_t, output=ao1_t)
        # Download the actor output (we only consume the mean half host-
        # side, but the whole vector lives in the staging buffer).
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

    # ──────────────────────────────────────────────────────────────────
    # Phase B.5b — batched GPU action + record API. Used by the N_ENVS
    # GPU driver (`run_offpolicy_train_gpu_n_envs`). The driver owns
    # the N_ENVS-sized obs/action/ao/alp scratch buffers (since
    # SACTrainer is not N_ENVS-parametric at the struct level — it can
    # only be N_ENVS-parametric at the method level, so allocating
    # struct-resident N_ENVS-sized scratch isn't possible without
    # breaking single-env trainers).
    # ──────────────────────────────────────────────────────────────────

    def select_action_gpu_batched[N_ENVS: Int](
        mut self,
        ctx: DeviceContext,
        obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        ao_scratch_dev: DeviceBuffer[DT],
        alp_scratch_dev: DeviceBuffer[DT],
        step_idx: Int,
    ) raises:
        """Batched policy step for N_ENVS envs.

        Warmup (`step_idx < learning_starts`): launches
        `_warmup_uniform_kernel` to fill `action_dev` with
        Uniform(-action_scale, +action_scale) lanes via Philox. The
        trainer's `_warmup_rng_offset` advances each call so successive
        warmup steps draw disjoint streams.

        Else: `actor.forward["gpu", N_ENVS](obs → ao_scratch_dev)` →
        `rsample.forward["gpu", N_ENVS](ao → alp_scratch_dev)` →
        `_action_clamp_kernel` extracts + clamps the first ACT_DIM
        lanes of `alp_scratch_dev` into `action_dev`.

        Buffer shapes:
          obs_dev          : [N_ENVS * OBS_DIM]
          action_dev       : [N_ENVS * ACT_DIM]              (out)
          ao_scratch_dev   : [N_ENVS * 2 * ACT_DIM]          (driver-owned)
          alp_scratch_dev  : [N_ENVS * (ACT_DIM + 1)]        (driver-owned)
        """
        comptime assert N_ENVS > 0, "N_ENVS must be > 0"
        var action_lt = LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.ACT_DIM), MutAnyOrigin,
        ](action_dev.unsafe_ptr())

        if step_idx < self.learning_starts:
            comptime TPB = 128
            comptime total = N_ENVS * Self.ACT_DIM
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime warmup_kernel = _warmup_uniform_kernel[
                N_ENVS, Self.ACT_DIM
            ]
            ctx.enqueue_function[warmup_kernel](
                action_lt,
                self.action_scale,
                self._warmup_rng_seed,
                self._warmup_rng_offset,
                grid_dim=n_blocks, block_dim=TPB,
            )
            self._warmup_rng_offset += UInt64(N_ENVS * Self.ACT_DIM * 2)
            return

        var obs_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            obs_dev.unsafe_ptr()
        )
        var ao_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            ao_scratch_dev.unsafe_ptr()
        )
        var alp_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            alp_scratch_dev.unsafe_ptr()
        )
        var obs_t = TileTensor(
            obs_p, row_major[N_ENVS, Self.OBS_DIM]()
        )
        var ao_t = TileTensor(
            ao_p, row_major[N_ENVS, 2 * Self.ACT_DIM]()
        )
        var alp_t = TileTensor(
            alp_p, row_major[N_ENVS, Self.ACT_DIM + 1]()
        )
        self.actor.forward["gpu", N_ENVS](obs_t, output=ao_t)
        self.actor_loss.rsample.forward["gpu", N_ENVS](
            ao_t, output=alp_t,
        )

        var alp_lt = LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.ACT_DIM + 1), MutAnyOrigin,
        ](alp_scratch_dev.unsafe_ptr())
        comptime TPB = 128
        comptime total = N_ENVS * Self.ACT_DIM
        comptime n_blocks = (total + TPB - 1) // TPB
        comptime clamp_kernel = _action_clamp_kernel[N_ENVS, Self.ACT_DIM]
        ctx.enqueue_function[clamp_kernel](
            alp_lt, action_lt, self.action_scale,
            grid_dim=n_blocks, block_dim=TPB,
        )

    def record_batch_gpu[N_ENVS: Int](
        mut self,
        ctx: DeviceContext,
        prev_obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        reward_dev: DeviceBuffer[DT],
        obs_dev: DeviceBuffer[DT],
        done_dev: DeviceBuffer[DT],
    ) raises:
        """Push N_ENVS transitions into the device-resident replay.

        Routes the device-side batched add either to
        `GPUPrioritizedReplay.add_batch[N_ENVS]` (when the trainer was
        built from a SACConfig with `use_per=True`, so `buf_per` is
        Some) or to the uniform `GPUReplay.add_batch[N_ENVS]`.

        Does NOT touch the episode tracker — the N_ENVS driver manages
        its own host-side per-env reward accumulators (so this method
        stays purely about the replay push, no D2H of `reward_dev`).
        """
        if self.buf_per:
            self.buf_per.value().add_batch[N_ENVS](
                ctx,
                prev_obs_dev, action_dev, reward_dev, obs_dev, done_dev,
            )
        else:
            self.buf_gpu.value().add_batch[N_ENVS](
                ctx,
                prev_obs_dev, action_dev, reward_dev, obs_dev, done_dev,
            )

    def record_batch_gpu_nstep[N_ENVS: Int, NS: Int](
        mut self,
        ctx: DeviceContext,
        mut nstep_buf: GPUNStepBuffer[
            NS, Self.OBS_DIM, Self.ACT_DIM, N_ENVS
        ],
        prev_obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        reward_dev: DeviceBuffer[DT],
        obs_dev: DeviceBuffer[DT],
        done_dev: DeviceBuffer[DT],
    ) raises:
        """N_ENVS + n-step batched record. The caller owns the
        `GPUNStepBuffer[NS, OBS, ACT, N_ENVS]` (since `N_ENVS` is
        method-comptime, not struct-comptime — the trainer can't
        pre-allocate it without losing single-env trainer flexibility).
        Behaviour:
            1. `nstep_buf.process(...)` — one kernel ring-updates all
               N_ENVS lanes and emits compressed transitions into
               `out_*` device buffers. Invalid slots zero-padded.
            2. `nstep_buf.store_into[CAP](ctx, buf_per | buf_gpu)` —
               blind-stores all N_ENVS slots into the underlying
               replay. PER overload picks up `max_priority^alpha` for
               each leaf so freshly-stored compressed transitions are
               immediately eligible for prioritised sampling.

        The depth `NS` must match the trainer's `Self.N_STEP` comptime
        so the γ^N bake in `target_y_block` agrees with the n-step
        return. The comptime assert below catches mismatches at
        instantiation time.
        """
        comptime assert NS == Self.N_STEP, (
            "GPUNStepBuffer depth NS must match SACTrainer.N_STEP "
            + "so γ^N_STEP bake in target_y matches the n-step return"
        )
        nstep_buf.process(
            ctx, prev_obs_dev, action_dev, reward_dev, obs_dev, done_dev,
        )
        if self.buf_per:
            nstep_buf.store_into[Self.REPLAY_CAPACITY](
                ctx, self.buf_per.value(),
            )
        else:
            nstep_buf.store_into[Self.REPLAY_CAPACITY](
                ctx, self.buf_gpu.value(),
            )

    def record(
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action: UnsafePointer[Scalar[DT], MutAnyOrigin],
        reward: Scalar[DT],
        next_obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        done: Scalar[DT],
    ) raises:
        """Push (s, a, r, s', done) into replay; accumulate the episode
        reward.

        Phase C.1 — branches on `buf_gpu`. When the trainer was built
        with `make["gpu"]`, `buf_gpu` is `Some` and the transition is
        uploaded into the device-resident replay (5 small D2H + 1 tiny
        kernel per call). CPU trainers route to `self.buf.add`. The
        tracker's per-step reward accumulation is unaffected.

        Phase C.2b — when `_use_nstep`, the (s, a, r, s', done) stream
        is fed through `nstep_cpu` first; only when that buffer emits
        a compressed n-step transition do we push into the replay.
        The tracker still accumulates the *single-step* reward so
        episode-return tracking is independent of n-step compression.
        """
        self.tracker.add_reward(reward)

        if self._use_nstep:
            var tx = self.nstep_cpu.value().add(
                obs, action, reward, next_obs, done > Scalar[DT](0.5),
            )
            if not tx.valid:
                return
            var nstep_obs_p = self._nstep_obs.cpu_ptr()
            var nstep_act_p = self._nstep_act.cpu_ptr()
            var nstep_nxt_p = self._nstep_nxt.cpu_ptr()
            for d in range(Self.OBS_DIM):
                nstep_obs_p[d] = tx.obs[d]
                nstep_nxt_p[d] = tx.next_obs[d]
            for j in range(Self.ACT_DIM):
                nstep_act_p[j] = tx.action[j]
            var done_emit = Scalar[DT](1.0) if tx.done else Scalar[DT](0.0)
            if self.buf_per:
                var ctx = self.target_y_block.ts.ctx.value()
                self.buf_per.value().add(
                    ctx,
                    nstep_obs_p, nstep_act_p,
                    tx.reward, nstep_nxt_p, done_emit,
                )
            elif self.buf_gpu:
                var ctx = self.target_y_block.ts.ctx.value()
                self.buf_gpu.value().add(
                    ctx,
                    nstep_obs_p, nstep_act_p,
                    tx.reward, nstep_nxt_p, done_emit,
                )
            else:
                self.buf.add(
                    nstep_obs_p, nstep_act_p,
                    tx.reward, nstep_nxt_p, done_emit,
                )
            return

        if self.buf_per:
            var ctx = self.target_y_block.ts.ctx.value()
            self.buf_per.value().add(
                ctx, obs, action, reward, next_obs, done,
            )
        elif self.buf_gpu:
            var ctx = self.target_y_block.ts.ctx.value()
            self.buf_gpu.value().add(
                ctx, obs, action, reward, next_obs, done,
            )
        else:
            self.buf.add(obs, action, reward, next_obs, done)

    def end_episode(mut self):
        """Roll the current episode return into the tracker window."""
        self.tracker.end_episode()

    def add_complete_return(mut self, ret: Scalar[DT]):
        """Push an externally-tracked complete-episode return into the
        tracker window. Used by the N_ENVS GPU driver (Phase B.5b) where
        the driver maintains its own per-env reward accumulators; the
        trainer never sees individual rewards via `record_batch_gpu`."""
        self.tracker.add_complete_return(ret)

    # ──────────────────────────────────────────────────────────────────
    # Training step + sub-steps. Each sub-step stays under the Mojo
    # ~20-sequential-def-raises inline-explosion threshold.
    # ──────────────────────────────────────────────────────────────────

    def train_step(mut self, step_idx: Int) raises -> Bool:
        """Non-parametric overload — forwards to CPU path. See
        `select_action` doc for the trait-conformance rationale."""
        return self.train_step["cpu"](step_idx)

    def train_step[
        target: StaticString = "cpu",
        POLICY: AMPPolicy = NoAMP,
    ](mut self, step_idx: Int) raises -> Bool:
        """One full off-policy SAC update if past warmup.

        Returns True if a training step actually ran, False if the call
        was skipped (warmup or under-filled buffer).

        Phase C.1 — on GPU the minibatch is sampled *directly* into the
        device buffers by `GPUReplay.sample[BATCH]`. Pre-C.1, the GPU
        path went CPU sample → 4 host→device uploads each step; that
        upload block is now gone.

        Phase C.5 — `POLICY` (default `NoAMP`) threads into every
        forward/vjp call inside `target_y_block.step`,
        `twin_critic_block.step`, and `actor_loss.forward_backward`.
        Pass `POLICY=Bf16Compute` for bf16 mixed-precision compute on
        supported hardware. CPU path with `POLICY=NoAMP` is bit-
        identical to pre-C.5.
        """
        if step_idx < self.learning_starts:
            return False

        var t_sample = perf_counter_ns()

        # Phase G.1 — pointer resolution. CPU branch threads CPU
        # mirrors; GPU branch threads device pointers from the same
        # Scratch fields' `.dev` half. mb_y has no CPU read site, but
        # the staging mirror is allocated unconditionally so we can
        # take cpu_ptr() once and reassign in the GPU branch.
        var mb_s_ptr = self._mb_s.cpu_ptr()
        var mb_a_ptr = self._mb_a.cpu_ptr()
        var mb_r_ptr = self._mb_r.cpu_ptr()
        var mb_sp_ptr = self._mb_sp.cpu_ptr()
        var mb_y_ptr = self._mb_y.cpu_ptr()

        comptime if target == "cpu":
            if self.buf.size < Self.BATCH:
                return False
            self.buf.sample(
                Self.BATCH,
                mb_s_ptr, mb_a_ptr, mb_r_ptr, mb_sp_ptr,
                self._mb_d.cpu_ptr(),
            )
        else:
            var ctx = self.target_y_block.ts.ctx.value()
            if self.buf_per:
                # Phase C.3b — prioritized sample. Replaces the
                # uniform sample with stratified PER draws and
                # populates `buf_per.weights` / `buf_per.base.indices`
                # for the post-critic priority refresh.
                if self.buf_per.value().base.size < Self.BATCH:
                    return False
                self.buf_per.value().sample[Self.BATCH](
                    ctx,
                    self._mb_s.dev.value(), self._mb_a.dev.value(),
                    self._mb_r.dev.value(), self._mb_sp.dev.value(),
                    self._mb_d.dev.value(),
                )
            else:
                if self.buf_gpu.value().size < Self.BATCH:
                    return False
                self.buf_gpu.value().sample[Self.BATCH](
                    ctx,
                    self._mb_s.dev.value(), self._mb_a.dev.value(),
                    self._mb_r.dev.value(), self._mb_sp.dev.value(),
                    self._mb_d.dev.value(),
                )
            mb_s_ptr = self._mb_s.dev_ptr()
            mb_a_ptr = self._mb_a.dev_ptr()
            mb_r_ptr = self._mb_r.dev_ptr()
            mb_sp_ptr = self._mb_sp.dev_ptr()
            mb_y_ptr = self._mb_y.dev_ptr()
        self.timer.accumulate(Self._T_SAMPLE, t_sample)

        var alpha = fexp(self.alpha_opt.value)

        var t_ty = perf_counter_ns()
        self._train_compute_target_y[target, POLICY](
            alpha, mb_sp_ptr, mb_r_ptr, mb_y_ptr,
        )
        self.timer.accumulate(Self._T_TARGET_Y, t_ty)

        var t_crit = perf_counter_ns()
        var crit_loss = self._train_critic_update[target, POLICY](
            mb_s_ptr, mb_a_ptr, mb_y_ptr,
        )
        self.timer.accumulate(Self._T_CRITIC, t_crit)

        # Phase C.3b — PER priority refresh, GPU-only. Reads Q1 pred
        # still resident in twin_critic_block.c1._mb_q (pre-update
        # forward output) and `mb_y_ptr`, launches the absolute-TD
        # kernel into `_td_err_dev`, then hands those device-side
        # |TD| values to the host sum-tree update path. No-op when
        # PER is disabled.
        comptime if target == "gpu":
            if self.buf_per:
                self._train_per_priority_refresh()

        var t_act = perf_counter_ns()
        var actor_res = self._train_actor_update[target, POLICY](
            alpha, mb_s_ptr,
        )
        self.timer.accumulate(Self._T_ACTOR, t_act)

        var t_alp = perf_counter_ns()
        self._train_alpha_update(actor_res.log_prob_mean)
        self.timer.accumulate(Self._T_ALPHA, t_alp)

        var t_pol = perf_counter_ns()
        self._train_polyak[target]()
        self.timer.accumulate(Self._T_POLYAK, t_pol)

        self._actor_L_accum += actor_res.loss
        self._critic_L_accum += crit_loss
        self._alpha_accum += fexp(self.alpha_opt.value)
        self._update_count += 1
        return True

    def _train_compute_target_y[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        alpha: Scalar[DT],
        mb_sp_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_r_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_y_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        self.target_y_block.step[target, POLICY](
            self.actor, self.pair1.target_net, self.pair2.target_net,
            mb_sp_ptr, mb_r_ptr, alpha, mb_y_ptr,
        )

    def _train_critic_update[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mb_s_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_a_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_y_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises -> Scalar[DT]:
        """Phase C.3c — when `buf_per` is active (PER), pass the
        per-sample IS-weights vector through `twin_critic_block.step`
        so both critics receive a w_i-scaled gradient. Pre-C.3c PER
        used unweighted MSE (sample-side prioritization only); now
        the IS correction flows into the gradient via the new kernel
        in `critic_update_block.mojo`. Null pointer when buf_per is
        None → unweighted MSE → bit-identical to pre-C.3c."""
        var mb_y_t = TileTensor(mb_y_ptr, row_major[Self.BATCH, 1]())
        var weights_p = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        comptime if target == "gpu":
            if self.buf_per:
                weights_p = rebind[
                    UnsafePointer[Scalar[DT], MutAnyOrigin]
                ](self.buf_per.value().weights.unsafe_ptr())
        return self.twin_critic_block.step[target, POLICY](
            self.pair1.online, self.critic1_opt,
            self.pair2.online, self.critic2_opt,
            mb_s_ptr, mb_a_ptr, mb_y_t,
            weights_p=weights_p,
        )

    def _train_actor_update[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        alpha: Scalar[DT],
        mb_s_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises -> SACActorLossOut:
        return self.actor_loss.forward_backward[
            target, OPT=Adam, POLICY=POLICY,
        ](
            self.actor, self.actor_opt, self.pair1.online, self.pair2.online,
            mb_s_ptr, alpha,
        )

    def _train_per_priority_refresh(mut self) raises:
        """Phase C.3b — GPU-only. Launches `_td_abs_kernel` against
        `twin_critic_block.c1._mb_q` (pre-update Q1) and `_mb_y.dev`,
        writing per-lane `|Q1 − y|` into `_td_err_dev`, then forwards
        that to `buf_per.update_priorities[BATCH]` which D2H-copies it,
        recomputes the sum-tree leaves, and bumps `max_priority`.
        """
        var ctx = self.target_y_block.ts.ctx.value()
        var q1_p = self.twin_critic_block.c1._mb_q.dev_ptr()
        var y_p = self._mb_y.dev_ptr()
        var td_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._td_err_dev.value().unsafe_ptr()
        )
        var q_lt = LayoutTensor[
            DT, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](q1_p)
        var y_lt = LayoutTensor[
            DT, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](y_p)
        var td_lt = LayoutTensor[
            DT, Layout.row_major(Self.BATCH), MutAnyOrigin
        ](td_p)
        comptime TPB = 128
        comptime n_blocks = (Self.BATCH + TPB - 1) // TPB
        comptime td_kernel = _td_abs_kernel[Self.BATCH]
        ctx.enqueue_function[td_kernel](
            q_lt, y_lt, td_lt,
            grid_dim=n_blocks, block_dim=TPB,
        )
        self.buf_per.value().update_priorities[Self.BATCH](
            ctx, self._td_err_dev.value(),
        )

    def _train_alpha_update(mut self, log_prob_mean: Scalar[DT]):
        """Auto-tuned entropy temperature: minimize α·(−(H_target + H_curr)).
        ScalarAdam grad = -(log_prob_mean + target_entropy)."""
        self.alpha_opt.step(-(log_prob_mean + self.target_entropy))

    def _train_polyak[target: StaticString](mut self) raises:
        """Polyak τ-soft-update both critic target nets.

        GPU path threads the trainer's existing `DeviceContext` (stored in
        the target_y_block) through to `polyak_update`, so the kernel
        launcher reuses the queue instead of constructing a fresh
        `DeviceContext()` per leaf per step (Apple Metal would otherwise
        exhaust command-queue resources within ~1000 train steps)."""
        comptime if target == "cpu":
            self.pair1.polyak_step[target](self.tau)
            self.pair2.polyak_step[target](self.tau)
        else:
            var ctx = self.target_y_block.ts.ctx.value()
            self.pair1.polyak_step[target](self.tau, ctx)
            self.pair2.polyak_step[target](self.tau, ctx)

    # ──────────────────────────────────────────────────────────────────
    # Logging accessors
    # ──────────────────────────────────────────────────────────────────

    def mean_return(self) -> Scalar[DT]:
        return self.tracker.mean_return()

    def ep_count(self) -> Int:
        return self.tracker.ep_count

    def flush_train_log(mut self) -> Tuple[Scalar[DT], Scalar[DT], Scalar[DT], Int]:
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

    # ─────────────────── Phase A.5 — Logger plumbing ───────────────────
    # `flush_metrics` is the structured-logging analogue of
    # `flush_train_log`. It builds a `SACMetrics` bundle from the same
    # accumulators, emits one `log_scalar` per field (via reflection
    # through `log_bundle`), and resets. The user passes an optional
    # Logger pointer; when None the call is a cheap stack-build (the
    # bundle's `Float64` fields cost nothing) and no emit happens.
    # When the Logger is `NoOpLogger` (default), `log_bundle`'s
    # `comptime if not L.ENABLED: return` elides the walk at compile
    # time — zero runtime overhead even with the pointer wired.

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
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum = Scalar[DT](0.0)
        self._update_count = 0
        if Bool(logger):
            log_bundle(logger.value()[], bundle, step)
        return bundle^

    def flush_timer_log(mut self) -> String:
        """Return a formatted per-section wall-time report (one line per
        sub-step: sample / target_y / critic / actor / alpha / polyak)
        and reset the accumulators."""
        var report = self.timer.format_report()
        self.timer.reset()
        return report



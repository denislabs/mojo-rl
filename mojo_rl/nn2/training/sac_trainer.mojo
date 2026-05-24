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

from std.math import exp as fexp, log as flog, tanh as ftanh
from std.memory import alloc
from std.random import random_float64
from std.random.philox import Random as PhiloxRandom
from std.time import perf_counter_ns
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, TileTensor, TensorLayout, row_major

from ..constants import DT
from ..core import Module
from ..core.amp import AMPPolicy, NoAMP
from ..core.online_target_pair import OnlineTargetPair
from ..initializer import Xavier
from ..optimizer.adam import Adam
from ..optimizer.scalar_adam import ScalarAdam
from ..loss.sac_actor_loss_cg import SACActorLossCG as SACActorLoss, SACActorLossOut
from ..loss.critic_update_block import TwinCriticUpdateBlock
from ..data.cpu_replay import CPUReplay
from ..data.gpu_replay import GPUReplay
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
    runs after the D2H of `_alp1_dev`; in batched mode we keep it on
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
](OffPolicyTrainable, OffPolicyTrainableGpu, OffPolicyTrainableGpuBatched):
    comptime SA_DIM = Self.OBS_DIM + Self.ACT_DIM

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
    var tracker: EpisodeTracker

    # ─── Single-step scratch (env interaction) ────────────────────────
    var _ob1: UnsafePointer[Scalar[DT], MutAnyOrigin]            # [OBS_DIM]
    var _ao1: UnsafePointer[Scalar[DT], MutAnyOrigin]            # [2*ACT_DIM]
    var _alp1: UnsafePointer[Scalar[DT], MutAnyOrigin]           # [ACT_DIM+1]
    # GPU single-step env-interaction scratch (block D).
    var _ob1_dev: Optional[DeviceBuffer[DT]]
    var _ao1_dev: Optional[DeviceBuffer[DT]]
    var _alp1_dev: Optional[DeviceBuffer[DT]]

    # ─── Minibatch scratch (training) — only raw replay samples ──────
    # Target-y compute scratch is owned by `target_y_block` (Phase 10F).
    # Critic update scratch is owned by `twin_critic_block` (Phase 10F).
    var _mb_s: UnsafePointer[Scalar[DT], MutAnyOrigin]           # [BATCH, OBS]
    var _mb_a: UnsafePointer[Scalar[DT], MutAnyOrigin]           # [BATCH, ACT]
    var _mb_r: UnsafePointer[Scalar[DT], MutAnyOrigin]           # [BATCH]
    var _mb_sp: UnsafePointer[Scalar[DT], MutAnyOrigin]          # [BATCH, OBS]
    var _mb_d: UnsafePointer[Scalar[DT], MutAnyOrigin]           # [BATCH]
    var _mb_y: UnsafePointer[Scalar[DT], MutAnyOrigin]           # [BATCH, 1]
    # GPU minibatch scratch (block D). Trainer uploads CPU replay to
    # these after each `buf.sample` and threads device pointers to the
    # GPU step methods.
    var _mb_s_dev: Optional[DeviceBuffer[DT]]
    var _mb_a_dev: Optional[DeviceBuffer[DT]]
    var _mb_r_dev: Optional[DeviceBuffer[DT]]
    var _mb_sp_dev: Optional[DeviceBuffer[DT]]
    var _mb_d_dev: Optional[DeviceBuffer[DT]]
    var _mb_y_dev: Optional[DeviceBuffer[DT]]

    # ─── Hyperparameters ──────────────────────────────────────────────
    var gamma: Scalar[DT]
    var tau: Scalar[DT]
    var action_scale: Scalar[DT]
    var target_entropy: Scalar[DT]
    var learning_starts: Int

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
        self.tracker = EpisodeTracker(
            window=List[Scalar[DT]](),
            window_size=0, idx=0,
            current_return=Scalar[DT](0.0), ep_count=0,
        )
        var null_p = UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0)
        self._ob1 = null_p
        self._ao1 = null_p
        self._alp1 = null_p
        self._mb_s = null_p
        self._mb_a = null_p
        self._mb_r = null_p
        self._mb_sp = null_p
        self._mb_d = null_p
        self._mb_y = null_p
        self._ob1_dev = None
        self._ao1_dev = None
        self._alp1_dev = None
        self._mb_s_dev = None
        self._mb_a_dev = None
        self._mb_r_dev = None
        self._mb_sp_dev = None
        self._mb_d_dev = None
        self._mb_y_dev = None
        self.gamma = Scalar[DT](0.99)
        self.tau = Scalar[DT](0.005)
        self.action_scale = Scalar[DT](1.0)
        self.target_entropy = Scalar[DT](-1.0)
        self.learning_starts = 1_000
        self._warmup_rng_seed = UInt64(0xBADBEEF_FEEDFACE)
        self._warmup_rng_offset = UInt64(0)
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum = Scalar[DT](0.0)
        self._update_count = 0
        self.timer = Timer.new()

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
        the same construction code (bit-identical by construction)."""
        return Self.make[target](
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

    @staticmethod
    def make[target: StaticString](
        ctx: DeviceContext, config: SACConfig,
    ) raises -> Self:
        """Phase A.4 — Config-driven GPU factory."""
        return Self.make[target](
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

        # Allocate all training scratch.
        t._ob1 = alloc[Scalar[DT]](Self.OBS_DIM)
        t._ao1 = alloc[Scalar[DT]](2 * Self.ACT_DIM)
        t._alp1 = alloc[Scalar[DT]](Self.ACT_DIM + 1)
        t._mb_s = alloc[Scalar[DT]](Self.BATCH * Self.OBS_DIM)
        t._mb_a = alloc[Scalar[DT]](Self.BATCH * Self.ACT_DIM)
        t._mb_r = alloc[Scalar[DT]](Self.BATCH)
        t._mb_sp = alloc[Scalar[DT]](Self.BATCH * Self.OBS_DIM)
        t._mb_d = alloc[Scalar[DT]](Self.BATCH)
        t._mb_y = alloc[Scalar[DT]](Self.BATCH)

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

        # CPU scratch is allocated alongside GPU scratch — CPU buffers
        # are where replay samples land before upload, and where env
        # observations land before single-step actor forward.
        t._ob1 = alloc[Scalar[DT]](Self.OBS_DIM)
        t._ao1 = alloc[Scalar[DT]](2 * Self.ACT_DIM)
        t._alp1 = alloc[Scalar[DT]](Self.ACT_DIM + 1)
        t._mb_s = alloc[Scalar[DT]](Self.BATCH * Self.OBS_DIM)
        t._mb_a = alloc[Scalar[DT]](Self.BATCH * Self.ACT_DIM)
        t._mb_r = alloc[Scalar[DT]](Self.BATCH)
        t._mb_sp = alloc[Scalar[DT]](Self.BATCH * Self.OBS_DIM)
        t._mb_d = alloc[Scalar[DT]](Self.BATCH)
        t._mb_y = alloc[Scalar[DT]](Self.BATCH)
        # GPU device buffers.
        t._ob1_dev = ctx.enqueue_create_buffer[DT](Self.OBS_DIM)
        t._ao1_dev = ctx.enqueue_create_buffer[DT](2 * Self.ACT_DIM)
        t._alp1_dev = ctx.enqueue_create_buffer[DT](Self.ACT_DIM + 1)
        t._mb_s_dev = ctx.enqueue_create_buffer[DT](Self.BATCH * Self.OBS_DIM)
        t._mb_a_dev = ctx.enqueue_create_buffer[DT](Self.BATCH * Self.ACT_DIM)
        t._mb_r_dev = ctx.enqueue_create_buffer[DT](Self.BATCH)
        t._mb_sp_dev = ctx.enqueue_create_buffer[DT](Self.BATCH * Self.OBS_DIM)
        # Phase C.1 — done flag minibatch lives on device too so the
        # GPU `_gather_batch_kernel` can fill it. SAC's GPU target_y
        # currently ignores done (Pendulum truncation hard-codes
        # nonterm=1.0), but keeping the field future-proofs non-
        # truncation envs and matches the gather kernel's surface.
        t._mb_d_dev = ctx.enqueue_create_buffer[DT](Self.BATCH)
        t._mb_y_dev = ctx.enqueue_create_buffer[DT](Self.BATCH)

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

        comptime if target == "cpu":
            for d in range(Self.OBS_DIM):
                self._ob1[d] = obs[d]
            var ob1_t = TileTensor(self._ob1, row_major[1, Self.OBS_DIM]())
            var ao1_t = TileTensor(self._ao1, row_major[1, 2 * Self.ACT_DIM]())
            self.actor.forward["cpu", 1](ob1_t, output=ao1_t)
            var alp1_t = TileTensor(self._alp1, row_major[1, Self.ACT_DIM + 1]())
            self.actor_loss.rsample.forward["cpu", 1](ao1_t, output=alp1_t)
            for j in range(Self.ACT_DIM):
                var a = self._alp1[j]
                if a > self.action_scale:
                    a = self.action_scale
                elif a < -self.action_scale:
                    a = -self.action_scale
                action_out[j] = a
        else:
            var ctx = self.target_y_block.ts.ctx.value()
            # Upload obs into _ob1 (CPU) then to _ob1_dev.
            for d in range(Self.OBS_DIM):
                self._ob1[d] = obs[d]
            var ob1_dev = self._ob1_dev.value()
            ctx.enqueue_copy(ob1_dev, self._ob1)
            var ob1_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                ob1_dev.unsafe_ptr()
            )
            var ao1_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self._ao1_dev.value().unsafe_ptr()
            )
            var alp1_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self._alp1_dev.value().unsafe_ptr()
            )
            var ob1_t = TileTensor(ob1_p, row_major[1, Self.OBS_DIM]())
            var ao1_t = TileTensor(ao1_p, row_major[1, 2 * Self.ACT_DIM]())
            self.actor.forward["gpu", 1](ob1_t, output=ao1_t)
            var alp1_t = TileTensor(alp1_p, row_major[1, Self.ACT_DIM + 1]())
            self.actor_loss.rsample.forward["gpu", 1](ao1_t, output=alp1_t)
            # Download alp1 → CPU buffer, then clamp + write to action_out.
            ctx.enqueue_copy(self._alp1, self._alp1_dev.value())
            ctx.synchronize()
            for j in range(Self.ACT_DIM):
                var a = self._alp1[j]
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
        for d in range(Self.OBS_DIM):
            self._ob1[d] = obs[d]
        var ob1_t = TileTensor(self._ob1, row_major[1, Self.OBS_DIM]())
        var ao1_t = TileTensor(self._ao1, row_major[1, 2 * Self.ACT_DIM]())
        self.actor.forward["cpu", 1](ob1_t, output=ao1_t)
        for j in range(Self.ACT_DIM):
            var mean = self._ao1[j]
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
        return self.train_step["gpu"](step_idx)

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
        for d in range(Self.OBS_DIM):
            self._ob1[d] = obs[d]
        var ob1_dev = self._ob1_dev.value()
        ctx.enqueue_copy(ob1_dev, self._ob1)
        var ob1_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            ob1_dev.unsafe_ptr()
        )
        var ao1_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._ao1_dev.value().unsafe_ptr()
        )
        var ob1_t = TileTensor(ob1_p, row_major[1, Self.OBS_DIM]())
        var ao1_t = TileTensor(ao1_p, row_major[1, 2 * Self.ACT_DIM]())
        self.actor.forward["gpu", 1](ob1_t, output=ao1_t)
        # Download just the mean half (first ACT_DIM elements) into _ao1.
        ctx.enqueue_copy(self._ao1, self._ao1_dev.value())
        ctx.synchronize()
        for j in range(Self.ACT_DIM):
            var mean = self._ao1[j]
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

        Thin wrapper over `GPUReplay.add_batch[N_ENVS]`. Does NOT
        touch the episode tracker — the N_ENVS driver manages its
        own host-side per-env reward accumulators (so this method
        stays purely about the replay push, no D2H of `reward_dev`).
        """
        self.buf_gpu.value().add_batch[N_ENVS](
            ctx, prev_obs_dev, action_dev, reward_dev, obs_dev, done_dev,
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
        kernel per call). CPU trainers route to `self.buf.add` as
        before. The CPU path's behaviour is unchanged — bit-identity
        preserved by construction.
        """
        if self.buf_gpu:
            var ctx = self.target_y_block.ts.ctx.value()
            self.buf_gpu.value().add(
                ctx, obs, action, reward, next_obs, done,
            )
        else:
            self.buf.add(obs, action, reward, next_obs, done)
        self.tracker.add_reward(reward)

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

        # Resolve which pointers to thread through the step methods.
        var mb_s_ptr = self._mb_s
        var mb_a_ptr = self._mb_a
        var mb_r_ptr = self._mb_r
        var mb_sp_ptr = self._mb_sp
        var mb_y_ptr = self._mb_y

        comptime if target == "cpu":
            if self.buf.size < Self.BATCH:
                return False
            self.buf.sample(
                Self.BATCH,
                self._mb_s, self._mb_a, self._mb_r, self._mb_sp, self._mb_d,
            )
        else:
            if self.buf_gpu.value().size < Self.BATCH:
                return False
            var ctx = self.target_y_block.ts.ctx.value()
            self.buf_gpu.value().sample[Self.BATCH](
                ctx,
                self._mb_s_dev.value(), self._mb_a_dev.value(),
                self._mb_r_dev.value(), self._mb_sp_dev.value(),
                self._mb_d_dev.value(),
            )
            mb_s_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self._mb_s_dev.value().unsafe_ptr()
            )
            mb_a_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self._mb_a_dev.value().unsafe_ptr()
            )
            mb_r_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self._mb_r_dev.value().unsafe_ptr()
            )
            mb_sp_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self._mb_sp_dev.value().unsafe_ptr()
            )
            mb_y_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self._mb_y_dev.value().unsafe_ptr()
            )
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
        var mb_y_t = TileTensor(mb_y_ptr, row_major[Self.BATCH, 1]())
        return self.twin_critic_block.step[target, POLICY](
            self.pair1.online, self.critic1_opt,
            self.pair2.online, self.critic2_opt,
            mb_s_ptr, mb_a_ptr, mb_y_t,
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



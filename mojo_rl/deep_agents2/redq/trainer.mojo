"""REDQTrainer — Phase R.3 (CPU).

Randomized Ensembled Double Q-learning (Chen et al., ICLR 2021).
Mirrors `SACTrainer`'s shape — same `[train_target, SAMPLE, ACTOR,
CRITIC]` family of comptime params plus `[N, N_MIN, UTD, POLICY_DELAY,
Q_MODE]` for the ensemble knobs — and runs a UTD inner critic loop
per `train_step` call:

    train_step(step_idx):                  # outer = 1 env step
        sample once (gates warmup)
        for inner = 0..UTD-1:
            resample subset                # Fisher-Yates of (N choose N_MIN)
            target-y     (EnsembleTargetYBlock)
            critic       (EnsembleCriticStep)
            polyak       (EnsemblePolyakStep — every inner step, paper-faithful)
            if (_inner_count % POLICY_DELAY) == 0:
                actor    (EnsembleActorStep — mean over N online critics)
                alpha    (AlphaUpdateStep — unchanged from SAC)

The actor/α delayed cadence and the per-inner polyak match the
paper-faithful schedule from `deep_agents/redq/redq.mojo`.

R.3 is CPU-only and ships the OffPolicyAgent trait surface
(select_action_batched / select_greedy_action / record /
record_batch_cpu / train_step / mean_return / add_complete_return)
so the existing CPU off-policy driver can train it unchanged. GPU,
config-driven presets, and the one-file v2 checkpoint follow up in
R.5 / a separate slice.
"""

from std.math import exp as fexp, tanh as ftanh
from std.random import random_float64
from std.random.philox import Random as PhiloxRandom
from std.time import perf_counter_ns
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn2.core.checkpoint import (
    save_state_v2_body,
    load_state_v2_body,
    save_state_v2_body_gpu,
    load_state_v2_body_gpu,
)
from mojo_rl.nn2.core.map_params import hard_copy_params
from ..core.checkpoint_helpers import (
    save_optimizer_v2_body,
    load_optimizer_v2_body,
    save_optimizer_v2_body_gpu,
    load_optimizer_v2_body_gpu,
    save_scalar_adam_v2_body,
    load_scalar_adam_v2_body,
    save_counter_v2_body,
    load_counter_v2_body,
    split_lines_v2,
    read_file_v2,
    expect_v2_header,
)
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.core.scratch import Scratch
from mojo_rl.nn2.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn2.core.metric import LogScalar
from mojo_rl.nn2.core.log_bundle import log_bundle
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.optimizer.scalar_adam import ScalarAdam
from mojo_rl.nn2.training.timer import Timer

from ..training.episode_tracker import EpisodeTracker
from ..training.device_mean_accum import DeviceMeanAccum
from ..training.trainer_block import TrainerState
from ..training.driver_offpolicy import OffPolicyAgent, OffPolicyAgentGpu
from ..data.n_step_replay import GPUNStepBuffer
from ..training.blocks import SampleBlock
from ..sac.blocks.alpha_update_step import AlphaUpdateStep

from .ensemble import CriticEnsemble
from .ensemble_target_y_block import EnsembleTargetYBlock
from .blocks.ensemble_critic_step import EnsembleCriticStep
from .blocks.ensemble_actor_step import EnsembleActorStep
from .blocks.ensemble_polyak_step import EnsemblePolyakStep
from .metrics import REDQMetrics


# ────────────────────────────────────────────────────────────────────
# GPU select_action_batched kernels (mirror SAC's `_warmup_uniform_kernel`
# + `_action_clamp_kernel` so the batched action surface is shape-equivalent
# on CPU and GPU).
# ────────────────────────────────────────────────────────────────────


def _redq_warmup_uniform_kernel[
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


def _redq_action_clamp_kernel[
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


struct REDQTrainer[
    train_target: StaticString,
    SAMPLE: SampleBlock,
    ACTOR: Module,
    CRITIC: Module,
    N: Int,
    N_MIN: Int,
    UTD: Int,
    POLICY_DELAY: Int,
    Q_MODE: Int,
](OffPolicyAgentGpu):
    """REDQ Trainer. Dims (OBS / ACT / BATCH) derived from SAMPLE,
    so the user specifies them ONCE on the sample block type.

    `N` total online/target critics; `N_MIN` random subset size for
    MIN-mode TD target; `UTD` inner critic updates per env step;
    `POLICY_DELAY` actor + α update every K-th inner critic update;
    `Q_MODE` ∈ {0=MIN, 1=AVE} (REM is GPU-only — out of R.3 scope).
    """

    comptime OBS_DIM: Int = Self.SAMPLE.OBS
    comptime ACT_DIM: Int = Self.SAMPLE.ACT
    comptime BATCH: Int = Self.SAMPLE.BATCH

    # OffPolicyAgent trait aliases.
    comptime AGENT_OBS_DIM: Int = Self.OBS_DIM
    comptime AGENT_ACT_DIM: Int = Self.ACT_DIM
    comptime AGENT_TRAIN_TARGET: StaticString = Self.train_target

    # Timer section indices. Order matches `add_section` calls in `make`.
    comptime _T_SAMPLE = 0
    comptime _T_TARGET_Y = 1
    comptime _T_CRITIC = 2
    comptime _T_ACTOR = 3
    comptime _T_ALPHA = 4
    comptime _T_POLYAK = 5
    comptime _T_DIAG = 6

    # ─── Owned state ─────────────────────────────────────────────────
    var actor: Self.ACTOR
    var ensemble: CriticEnsemble[Self.CRITIC, Self.N]
    var actor_opt: Adam
    var alpha_opt: ScalarAdam

    var sample_blk: Self.SAMPLE
    var target_y_blk: EnsembleTargetYBlock[
        Self.ACTOR,
        Self.CRITIC,
        Self.N,
        Self.BATCH,
        Self.OBS_DIM,
        Self.ACT_DIM,
        Self.N_MIN,
        Self.Q_MODE,
    ]
    var critic_blk: EnsembleCriticStep[
        Self.CRITIC,
        Self.N,
        Self.OBS_DIM,
        Self.ACT_DIM,
        Self.BATCH,
    ]
    var actor_blk: EnsembleActorStep[
        Self.ACTOR,
        Self.CRITIC,
        Self.N,
        Self.OBS_DIM,
        Self.ACT_DIM,
        Self.BATCH,
    ]
    var alpha_blk: AlphaUpdateStep[
        Self.OBS_DIM,
        Self.ACT_DIM,
        Self.BATCH,
    ]
    var polyak_blk: EnsemblePolyakStep[
        Self.CRITIC,
        Self.N,
        Self.OBS_DIM,
        Self.ACT_DIM,
        Self.BATCH,
    ]

    var state: TrainerState[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]
    var tracker: EpisodeTracker
    var ctx: Optional[DeviceContext]

    # Single-env action-selection scratch (mirror SAC's _ob1 / _ao1 / _alp1).
    var _ob1: Scratch["ob1", Self.OBS_DIM, True]
    var _ao1: Scratch["ao1", 2 * Self.ACT_DIM, True]
    var _alp1: Scratch["alp1", Self.ACT_DIM + 1, True]

    var action_scale: Scalar[DT]
    var learning_starts: Int

    # GPU Philox warmup state (host counter advanced per warmup batch).
    var _warmup_rng_seed: UInt64
    var _warmup_rng_offset: UInt64

    # UTD bookkeeping. `_inner_count` is the cumulative inner critic
    # update counter modulo POLICY_DELAY drives the actor cadence.
    var _inner_count: Int

    # Metric accumulators (drained on flush).
    var _actor_L_accum: Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    var _alpha_accum: Scalar[DT]
    var _q_accum: Scalar[DT]
    var _target_accum: Scalar[DT]
    var _reward_accum: Scalar[DT]
    var _done_accum: Scalar[DT]
    var _abs_action_accum: Scalar[DT]
    # GPU device-resident mirrors (CPU keeps the host scalars above).
    var _q_mean_dev: DeviceMeanAccum
    var _target_mean_dev: DeviceMeanAccum
    var _reward_mean_dev: DeviceMeanAccum
    var _done_mean_dev: DeviceMeanAccum
    var _abs_action_mean_dev: DeviceMeanAccum
    var _update_count: Int  # inner steps this chunk
    var _actor_update_count: Int  # actor steps this chunk
    var _total_train_steps: Int  # cumulative inner steps (never reset)

    var timer: Timer

    def __init__(out self):
        self.actor = Self.ACTOR()
        self.ensemble = CriticEnsemble[Self.CRITIC, Self.N]()
        self.actor_opt = Adam()
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
        self.target_y_blk = EnsembleTargetYBlock[
            Self.ACTOR,
            Self.CRITIC,
            Self.N,
            Self.BATCH,
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.N_MIN,
            Self.Q_MODE,
        ]()
        self.critic_blk = EnsembleCriticStep[
            Self.CRITIC,
            Self.N,
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
        ]()
        self.actor_blk = EnsembleActorStep[
            Self.ACTOR,
            Self.CRITIC,
            Self.N,
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
        ]()
        self.alpha_blk = AlphaUpdateStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
        ]()
        self.polyak_blk = EnsemblePolyakStep[
            Self.CRITIC,
            Self.N,
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
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
        self._warmup_rng_seed = UInt64(0xC0FFEE_C0DE)
        self._warmup_rng_offset = UInt64(0)
        self._inner_count = 0
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum = Scalar[DT](0.0)
        self._q_accum = Scalar[DT](0.0)
        self._target_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        self._abs_action_accum = Scalar[DT](0.0)
        self._q_mean_dev = DeviceMeanAccum()
        self._target_mean_dev = DeviceMeanAccum()
        self._reward_mean_dev = DeviceMeanAccum()
        self._done_mean_dev = DeviceMeanAccum()
        self._abs_action_mean_dev = DeviceMeanAccum()
        self._update_count = 0
        self._actor_update_count = 0
        self._total_train_steps = 0
        self.timer = Timer.new()

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        actor_lr: Scalar[DT] = Scalar[DT](3e-4),
        critic_lr: Scalar[DT] = Scalar[DT](3e-4),
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
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "REDQTrainer: train_target must be 'cpu' or 'gpu'"
        comptime if Self.train_target == "gpu":
            if not ctx:
                raise Error(
                    "REDQTrainer.make[train_target='gpu']: ctx required"
                )
        comptime assert Self.N >= 2, "REDQ: N must be ≥ 2"
        comptime assert Self.N_MIN >= 1, "REDQ: N_MIN must be ≥ 1"
        comptime assert Self.N_MIN <= Self.N, "REDQ: N_MIN must be ≤ N"
        comptime assert Self.UTD >= 1, "REDQ: UTD must be ≥ 1"
        comptime assert Self.POLICY_DELAY >= 1, "REDQ: POLICY_DELAY must be ≥ 1"

        var t = Self()
        t.ctx = ctx

        t.actor = Self.ACTOR.make[Self.train_target, Xavier](ctx=ctx)
        t.ensemble = CriticEnsemble[Self.CRITIC, Self.N].make[
            Self.train_target,
            Xavier,
        ](ctx=ctx)
        t.actor_opt = Adam.make[Self.train_target, M=Self.ACTOR](
            t.actor,
            ctx=ctx,
        )
        t.alpha_opt = ScalarAdam.new(fexp_to_log(init_alpha), alpha_lr)
        # Apply LR to all N critic Adams (defaults already set inside
        # CriticEnsemble.make; this is the explicit user-tunable knob).
        for i in range(Self.N):
            t.ensemble.opts[i].lr = critic_lr
            t.ensemble.opts[i].max_grad_norm = max_grad_norm

        t.actor_opt.lr = actor_lr
        t.actor_opt.max_grad_norm = max_grad_norm

        t.target_y_blk = EnsembleTargetYBlock[
            Self.ACTOR,
            Self.CRITIC,
            Self.N,
            Self.BATCH,
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.N_MIN,
            Self.Q_MODE,
        ].make[Self.train_target](
            action_scale=action_scale,
            gamma=gamma,
            ctx=ctx,
        )
        t.critic_blk = EnsembleCriticStep[
            Self.CRITIC,
            Self.N,
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
        ].make[Self.train_target](ctx=ctx)
        t.actor_blk = EnsembleActorStep[
            Self.ACTOR,
            Self.CRITIC,
            Self.N,
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
        ].make[Self.train_target](
            action_scale=action_scale,
            ctx=ctx,
        )
        t.alpha_blk = AlphaUpdateStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
        ].make(target_entropy=target_entropy)
        t.polyak_blk = EnsemblePolyakStep[
            Self.CRITIC,
            Self.N,
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
        ].make(tau=tau)

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

        comptime if Self.train_target == "gpu":
            # Device-resident mean accumulators for the GPU diag path.
            t._q_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._target_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._reward_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._done_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._abs_action_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)

        t.action_scale = action_scale
        t.learning_starts = learning_starts

        # Sample block lifecycle.
        t.sample_blk.setup(learning_starts, ctx=ctx)

        # Timer sections (order must match the `_T_*` aliases).
        t.timer.add_section("sample")
        t.timer.add_section("target_y")
        t.timer.add_section("critic")
        t.timer.add_section("actor")
        t.timer.add_section("alpha")
        t.timer.add_section("polyak")
        t.timer.add_section("diag")
        return t^

    # ────────────────────────────────────────────────────────────────
    # Inner step body — one (target_y + critic + polyak + maybe actor).
    # ────────────────────────────────────────────────────────────────

    def _run_inner_step[POLICY: AMPPolicy = NoAMP](mut self) raises:
        """Single UTD inner step. Assumes `self.state` already has a
        fresh minibatch loaded by `sample_blk.step` above."""
        self._inner_count += 1

        # Subset resample (Fisher-Yates over {0..N-1}, length N_MIN).
        # MODE=AVE ignores it but the call is harmless. CPU-side
        # random_float64; the GPU target_y kernel reads the device
        # uploaded mirror set by `target_y_blk.step["gpu"]`.
        self.target_y_blk.resample_subset_idxs()

        # α: host scalar on both CPU and GPU (REDQ doesn't capture under
        # CUDA graphs — host control flow with subset sampling +
        # policy-delay gating — so the SAC device-α plumbing isn't
        # needed). The target-y kernel reads α as a launch argument.
        var t_ty = perf_counter_ns()
        var alpha_val = fexp(self.alpha_opt.value)
        self.state.alpha = alpha_val
        self.target_y_blk.step[Self.train_target, POLICY](
            self.actor,
            self.ensemble,
            self.state.mb_sp.target_ptr[Self.train_target](),
            self.state.mb_r.target_ptr[Self.train_target](),
            self.state.mb_d.target_ptr[Self.train_target](),
            alpha_val,
            self.state.mb_y.target_ptr[Self.train_target](),
        )
        self.timer.accumulate(Self._T_TARGET_Y, t_ty)

        # Critic update (N forward+vjp+Adam.step against shared y).
        var t_crit = perf_counter_ns()
        self.critic_blk.step[Self.train_target, POLICY](
            self.state,
            self.ensemble,
        )
        self.timer.accumulate(Self._T_CRITIC, t_crit)

        # Polyak ALL N targets every inner step (paper-faithful).
        var t_pol = perf_counter_ns()
        self.polyak_blk.step[Self.train_target](
            self.state,
            self.ensemble,
        )
        self.timer.accumulate(Self._T_POLYAK, t_pol)

        # Actor + α every POLICY_DELAY inner critic updates. First fire
        # at _inner_count == POLICY_DELAY, then 2·POLICY_DELAY, …
        # Matches legacy `(critic_update_count % POL_DELAY == 0)`.
        if self._inner_count % Self.POLICY_DELAY == 0:
            var t_act = perf_counter_ns()
            self.actor_blk.step[Self.train_target, POLICY](
                self.state,
                self.actor,
                self.actor_opt,
                self.ensemble,
            )
            self.timer.accumulate(Self._T_ACTOR, t_act)

            # α: ScalarAdam stays a host scalar (state.log_prob_mean is
            # already a host Scalar — populated on both CPU and GPU by
            # EnsembleActorLoss.forward_backward which D2Hs the lp mean
            # at the end of its scalar reduction). So the same CPU
            # AlphaUpdateStep code path drives α on both targets.
            var t_alp = perf_counter_ns()
            self.alpha_blk.step["cpu"](self.state, self.alpha_opt)
            self.timer.accumulate(Self._T_ALPHA, t_alp)

            self._actor_L_accum += self.state.actor_loss
            self._alpha_accum += fexp(self.alpha_opt.value)
            self._actor_update_count += 1

        # Per-batch diagnostic accumulators. CPU walks the host scratches
        # directly; GPU folds the same `[BATCH]` device buffers into
        # device-resident running means (no per-step D2H) read at flush.
        comptime if Self.train_target == "cpu":
            self._accumulate_diag()
        else:
            self._accumulate_diag_gpu()

        self._critic_L_accum += self.state.critic_loss
        self._update_count += 1
        self._total_train_steps += 1

    def _accumulate_diag(mut self):
        """Per-inner-step host-walk over the batch scratches:
        mean_q (last critic's Q), mean_target, mean_reward, mean_done,
        mean_abs_action. Matches the SAC `_T_DIAG` walk shape."""
        var t_diag = perf_counter_ns()
        var inv_b: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](Self.BATCH)
        var y_p = self.state.mb_y.cpu_ptr()
        var r_p = self.state.mb_r.cpu_ptr()
        var d_p = self.state.mb_d.cpu_ptr()
        var a_p = self.state.mb_a.cpu_ptr()
        # Last-critic Q (one CriticUpdateBlock reused across N critics —
        # `_mb_q` carries the last iter's `Q_{N-1}(s, a)`).
        var q_p = self.critic_blk.member_step._mb_q.cpu_ptr()
        var sum_y: Scalar[DT] = 0.0
        var sum_r: Scalar[DT] = 0.0
        var sum_d: Scalar[DT] = 0.0
        var sum_q: Scalar[DT] = 0.0
        for i in range(Self.BATCH):
            sum_y += y_p[i]
            sum_r += r_p[i]
            sum_d += d_p[i]
            sum_q += q_p[i]
        var sum_a: Scalar[DT] = 0.0
        for i in range(Self.BATCH * Self.ACT_DIM):
            var av = a_p[i]
            sum_a += av if av >= Scalar[DT](0.0) else -av
        self._q_accum += sum_q * inv_b
        self._target_accum += sum_y * inv_b
        self._reward_accum += sum_r * inv_b
        self._done_accum += sum_d * inv_b
        self._abs_action_accum += sum_a * (
            Scalar[DT](1.0) / Scalar[DT](Self.BATCH * Self.ACT_DIM)
        )
        self.timer.accumulate(Self._T_DIAG, t_diag)

    def _accumulate_diag_gpu(mut self) raises:
        """GPU mirror of `_accumulate_diag`: device reductions of the same
        batch scratches into device-resident running means. `mean_abs_action`
        uses the abs-reduction over the `[BATCH*ACT_DIM]` action buffer."""
        var t_diag = perf_counter_ns()
        var q_ptr = self.critic_blk.member_step._mb_q.target_ptr["gpu"]()
        var y_ptr = self.state.mb_y.target_ptr["gpu"]()
        var r_ptr = self.state.mb_r.target_ptr["gpu"]()
        var d_ptr = self.state.mb_d.target_ptr["gpu"]()
        var a_ptr = self.state.mb_a.target_ptr["gpu"]()
        self._q_mean_dev.accumulate_gpu[Self.BATCH](q_ptr)
        self._target_mean_dev.accumulate_gpu[Self.BATCH](y_ptr)
        self._reward_mean_dev.accumulate_gpu[Self.BATCH](r_ptr)
        self._done_mean_dev.accumulate_gpu[Self.BATCH](d_ptr)
        self._abs_action_mean_dev.accumulate_gpu_abs[
            Self.BATCH * Self.ACT_DIM
        ](a_ptr)
        self.timer.accumulate(Self._T_DIAG, t_diag)

    # ────────────────────────────────────────────────────────────────
    # train_step — outer (one env step). Runs UTD inner critic updates.
    # ────────────────────────────────────────────────────────────────

    def train_step(mut self, step_idx: Int) raises -> Bool:
        """One outer training tick. Samples + runs UTD inner critic
        updates. Returns True if any inner update ran (False during
        warmup / when buffer too small). The trainer's
        `_total_train_steps` increments by UTD when this returns True."""
        self.state.step_idx = step_idx
        self.state.did_step = True
        # state.ctx routes through into blocks (polyak / target_y / etc.)
        # that need it on GPU. None on CPU is fine.
        self.state.ctx = self.ctx

        # Sample #1 — gates warmup and buffer-readiness.
        var t_s0 = perf_counter_ns()
        self.sample_blk.step(self.state)
        self.timer.accumulate(Self._T_SAMPLE, t_s0)
        if not self.state.did_step:
            return False

        self._run_inner_step[NoAMP]()

        # Inner steps 2..UTD. Re-sample each time.
        for _ in range(Self.UTD - 1):
            var t_s = perf_counter_ns()
            self.state.did_step = True
            self.sample_blk.step(self.state)
            self.timer.accumulate(Self._T_SAMPLE, t_s)
            if not self.state.did_step:
                break  # buffer drained mid-iter (single-env: never)
            self._run_inner_step[NoAMP]()

        return True

    # ────────────────────────────────────────────────────────────────
    # Action-selection surface (OffPolicyAgent trait).
    # ────────────────────────────────────────────────────────────────

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
        """Single batched entry — CPU + GPU. CPU uses host
        `random_float64` for warmup; GPU launches a Philox kernel."""
        comptime assert N_ENVS > 0, "N_ENVS must be > 0"
        comptime ACT = Self.ACT_DIM
        comptime OBS = Self.OBS_DIM

        if step_idx < self.learning_starts:
            comptime if Self.train_target == "cpu":
                # CPU warmup: uniform random in [-scale, +scale].
                for i in range(N_ENVS * ACT):
                    var u = Scalar[DT](2.0 * random_float64() - 1.0)
                    action_ptr[i] = u * self.action_scale
            else:
                # GPU warmup: Philox kernel; advance offset by 2·N·A
                # (each step_uniform consumes 2 raw uint32s per lane).
                var action_lt = LayoutTensor[
                    DT,
                    Layout.row_major(N_ENVS, ACT),
                    MutAnyOrigin,
                ](action_ptr)
                comptime total = N_ENVS * ACT
                comptime n_blocks = (total + TPB - 1) // TPB
                comptime warmup_kernel = _redq_warmup_uniform_kernel[
                    N_ENVS,
                    ACT,
                ]
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

        # Policy forward — actor + rsample. Both target-parametric;
        # N_ENVS rolls through transparently.
        var obs_t = TileTensor(obs_ptr, row_major[N_ENVS, OBS]())
        var ao_t = TileTensor(ao_scratch_ptr, row_major[N_ENVS, 2 * ACT]())
        var alp_t = TileTensor(alp_scratch_ptr, row_major[N_ENVS, ACT + 1]())
        self.actor.forward[Self.train_target, N_ENVS](obs_t, output=ao_t)
        self.actor_blk.inner.rsample.forward[Self.train_target, N_ENVS](
            ao_t,
            output=alp_t,
        )

        # Clamp action.
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
            comptime clamp_kernel = _redq_action_clamp_kernel[N_ENVS, ACT]
            var ctx = self.ctx.value()
            ctx.enqueue_function[clamp_kernel](
                alp_lt,
                action_lt,
                self.action_scale,
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        """Host-list single-env wrapper (smoke-test/eval-loop callers).
        On GPU: H2D obs → select_action_batched[1] writes through
        device scratches → D2H action."""
        var ob1_cpu_p = self._ob1.cpu_ptr()
        for d in range(Self.OBS_DIM):
            ob1_cpu_p[d] = obs[d]
        comptime if Self.train_target == "gpu":
            var ctx = self.ctx.value()
            ctx.enqueue_copy(self._ob1.dev.value(), ob1_cpu_p)
        self.select_action_batched[1](
            self._ob1.target_ptr[Self.train_target](),
            self._alp1.target_ptr[Self.train_target](),  # action_ptr aliasing
            self._ao1.target_ptr[Self.train_target](),
            self._alp1.target_ptr[Self.train_target](),
            step_idx,
        )
        comptime if Self.train_target == "gpu":
            var ctx = self.ctx.value()
            ctx.enqueue_copy(self._alp1.cpu_ptr(), self._alp1.dev.value())
            ctx.synchronize()
        var alp_cpu_p = self._alp1.cpu_ptr()
        for j in range(Self.ACT_DIM):
            action_out[j] = alp_cpu_p[j]

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        """Deterministic eval: `action = tanh(actor.forward(s).mean) ·
        action_scale`, clamped. CPU runs natively; GPU H2Ds obs,
        forwards on device, D2Hs the mean, and clamps on host
        (matches SAC's `select_greedy_action` shape)."""
        var ob1_cpu_p = self._ob1.cpu_ptr()
        var ao1_cpu_p = self._ao1.cpu_ptr()
        for d in range(Self.OBS_DIM):
            ob1_cpu_p[d] = obs[d]
        comptime if Self.train_target == "cpu":
            var ob1_t = TileTensor(
                ob1_cpu_p,
                row_major[1, Self.OBS_DIM](),
            )
            var ao1_t = TileTensor(
                ao1_cpu_p,
                row_major[1, 2 * Self.ACT_DIM](),
            )
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

    # ────────────────────────────────────────────────────────────────
    # Replay-push surface.
    # ────────────────────────────────────────────────────────────────

    def record(
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

    def record_batch_gpu[
        N_ENVS: Int,
    ](
        mut self,
        ctx: DeviceContext,
        prev_obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        reward_dev: DeviceBuffer[DT],
        obs_dev: DeviceBuffer[DT],
        done_dev: DeviceBuffer[DT],
    ) raises:
        """GPU-batched replay push. Forwards to `sample_blk.add_batch_gpu`
        which the ReplayBuffer trait routes through `R`'s `add_batch`
        (no-op/raise on CPU backends; real H2D-free path on GPU
        backends). The trait gate is the only thing exercising this on
        the (env=cpu, train=gpu) hybrid driver path."""
        self.sample_blk.add_batch_gpu[N_ENVS](
            ctx,
            prev_obs_dev,
            action_dev,
            reward_dev,
            obs_dev,
            done_dev,
        )

    def record_batch_gpu_nstep[
        N_ENVS: Int,
        NS: Int,
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
        """N-step batched record — not on R.5's REDQ surface. The
        single-step GPU-batched path (`record_batch_gpu` + uniform
        replay) covers Pendulum-shape envs."""
        raise Error(
            "REDQTrainer.record_batch_gpu_nstep: n-step replay not"
            " supported (R.5 ships uniform 1-step replay only)"
        )

    def learning_starts_count(self) -> Int:
        """OffPolicyAgentGpu trait hook — env-step threshold past which
        training is unlocked. Used by the GPU-env driver to gate the
        capture path (N/A for REDQ — `train_device_kernels` raises)."""
        return self.learning_starts

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
        """Per-lane SAC-style replay push without tracker.add_reward
        (the driver manages per-env return accumulators)."""
        comptime OBS = Self.OBS_DIM
        comptime ACT = Self.ACT_DIM
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

    # ────────────────────────────────────────────────────────────────
    # Tracker passthroughs.
    # ────────────────────────────────────────────────────────────────

    def end_episode(mut self):
        self.tracker.end_episode()

    def mean_return(self) -> Scalar[DT]:
        return self.tracker.mean_return()

    def ep_count(self) -> Int:
        return self.tracker.ep_count

    def add_complete_return(mut self, ret: Scalar[DT]):
        self.tracker.add_complete_return(ret)

    # ────────────────────────────────────────────────────────────────
    # Metrics surface.
    # ────────────────────────────────────────────────────────────────

    def total_train_steps(self) -> Int:
        """Cumulative INNER train steps. One env-step contributes UTD
        of these when past warmup. Used as the `train_steps` metric."""
        return self._total_train_steps

    def flush_metrics[
        L: Logger = NoOpLogger,
    ](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> REDQMetrics:
        """Drain inner-step accumulators into a REDQMetrics bundle.
        Critic_loss / mean_* are averaged over `_update_count` inner
        steps; actor_loss / alpha are averaged over `_actor_update_count`
        actor steps (smaller — actor fires every POLICY_DELAY inner
        steps)."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var na = self._actor_update_count if self._actor_update_count > 0 else 1
        var inv_a = Scalar[DT](1.0) / Scalar[DT](na)

        # Per-batch diag means: device-resident on GPU (folded in by
        # `_accumulate_diag_gpu`), host scalars on CPU.
        var q_mean: Scalar[DT]
        var target_mean: Scalar[DT]
        var reward_mean: Scalar[DT]
        var done_mean: Scalar[DT]
        var abs_action_mean: Scalar[DT]
        comptime if Self.train_target == "gpu":
            q_mean = self._q_mean_dev.read["gpu"]()
            target_mean = self._target_mean_dev.read["gpu"]()
            reward_mean = self._reward_mean_dev.read["gpu"]()
            done_mean = self._done_mean_dev.read["gpu"]()
            abs_action_mean = self._abs_action_mean_dev.read["gpu"]()
        else:
            q_mean = self._q_accum * inv
            target_mean = self._target_accum * inv
            reward_mean = self._reward_accum * inv
            done_mean = self._done_accum * inv
            abs_action_mean = self._abs_action_accum * inv

        var bundle = REDQMetrics(
            actor_loss=LogScalar[DT](self._actor_L_accum * inv_a),
            critic_loss=LogScalar[DT](self._critic_L_accum * inv),
            alpha=LogScalar[DT](self._alpha_accum * inv_a),
            mean_q=LogScalar[DT](q_mean),
            mean_target=LogScalar[DT](target_mean),
            mean_reward=LogScalar[DT](reward_mean),
            mean_next_q=LogScalar[DT](Scalar[DT](0.0)),
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
        self._done_accum = Scalar[DT](0.0)
        self._abs_action_accum = Scalar[DT](0.0)
        self._update_count = 0
        self._actor_update_count = 0
        comptime if Self.train_target == "gpu":
            self._q_mean_dev.reset["gpu"]()
            self._target_mean_dev.reset["gpu"]()
            self._reward_mean_dev.reset["gpu"]()
            self._done_mean_dev.reset["gpu"]()
            self._abs_action_mean_dev.reset["gpu"]()
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
        """Trait-uniform passthrough — drains the bundle via
        `flush_metrics` (which logs to `logger`) and discards the
        typed return."""
        _ = self.flush_metrics[L](logger, step)

    def flush_timer_log(mut self) -> String:
        var report = self.timer.format_report()
        self.timer.reset()
        return report

    # ────────────────────────────────────────────────────────────────
    # Checkpointing — one-file v2 envelope.
    # ────────────────────────────────────────────────────────────────
    #
    # Section layout:
    #   actor.*
    #   critic0.* … critic{N-1}.*           (online twins; targets reconstructed)
    #   actor_opt.*
    #   critic0_opt.* … critic{N-1}_opt.*
    #   alpha_opt.*
    #
    # Target nets are NOT serialized: they're hard-copied from the just-
    # restored online twins inside `load_state`. The replay buffer and
    # episode tracker are NOT serialized either (same convention as SAC).
    # CPU + GPU produce byte-identical files (GPU first D2Hs through the
    # CPU serializer); GPU→CPU interchange therefore works without any
    # format negotiation.

    def save_state(mut self, path: String) raises:
        var body = String("")
        comptime if Self.train_target == "cpu":
            save_state_v2_body(self.actor, body, "actor")
            for i in range(Self.N):
                save_state_v2_body(
                    self.ensemble.pairs[i].online,
                    body,
                    "critic" + String(i),
                )
            save_optimizer_v2_body(self.actor_opt, body, "actor_opt")
            for i in range(Self.N):
                save_optimizer_v2_body(
                    self.ensemble.opts[i],
                    body,
                    "critic" + String(i) + "_opt",
                )
            save_scalar_adam_v2_body(self.alpha_opt, body, "alpha_opt")
        else:
            var c = self.ctx.value()
            save_state_v2_body_gpu(self.actor, body, "actor", c)
            for i in range(Self.N):
                save_state_v2_body_gpu(
                    self.ensemble.pairs[i].online,
                    body,
                    "critic" + String(i),
                    c,
                )
            save_optimizer_v2_body_gpu(self.actor_opt, body, "actor_opt")
            for i in range(Self.N):
                save_optimizer_v2_body_gpu(
                    self.ensemble.opts[i],
                    body,
                    "critic" + String(i) + "_opt",
                )
            # ScalarAdam: REDQ uses the host-only constructor (`.new`)
            # on both CPU and GPU because the actor-loss block D2Hs
            # `log_prob_mean` already (no CUDA-graph capture goal),
            # so `step_device` and `state_dev` are never wired. Use
            # the CPU serializer here — the GPU variants would try
            # to `sync_to_host()` an unallocated `state_dev`.
            save_scalar_adam_v2_body(self.alpha_opt, body, "alpha_opt")
        save_counter_v2_body(self._total_train_steps, body, "_total_train_steps")
        var content = String("nn2-ckpt v2\n") + body
        with open(path, "w") as f:
            f.write(content)

    def load_state(mut self, path: String) raises:
        var content = read_file_v2(path)
        var lines = split_lines_v2(content)
        expect_v2_header(lines)
        var idx: Int = 1
        comptime if Self.train_target == "cpu":
            load_state_v2_body(self.actor, lines, idx, "actor")
            for i in range(Self.N):
                load_state_v2_body(
                    self.ensemble.pairs[i].online,
                    lines,
                    idx,
                    "critic" + String(i),
                )
            load_optimizer_v2_body(
                self.actor_opt,
                lines,
                idx,
                "actor_opt",
            )
            for i in range(Self.N):
                load_optimizer_v2_body(
                    self.ensemble.opts[i],
                    lines,
                    idx,
                    "critic" + String(i) + "_opt",
                )
            load_scalar_adam_v2_body(
                self.alpha_opt,
                lines,
                idx,
                "alpha_opt",
            )
        else:
            var c = self.ctx.value()
            load_state_v2_body_gpu(
                self.actor,
                lines,
                idx,
                "actor",
                c,
            )
            for i in range(Self.N):
                load_state_v2_body_gpu(
                    self.ensemble.pairs[i].online,
                    lines,
                    idx,
                    "critic" + String(i),
                    c,
                )
            load_optimizer_v2_body_gpu(
                self.actor_opt,
                lines,
                idx,
                "actor_opt",
            )
            for i in range(Self.N):
                load_optimizer_v2_body_gpu(
                    self.ensemble.opts[i],
                    lines,
                    idx,
                    "critic" + String(i) + "_opt",
                )
            # See save_state above for the rationale — REDQ uses the
            # CPU ScalarAdam path regardless of train_target.
            load_scalar_adam_v2_body(
                self.alpha_opt,
                lines,
                idx,
                "alpha_opt",
            )
        load_counter_v2_body(
            self._total_train_steps, lines, idx, "_total_train_steps"
        )
        # Re-sync every target net from its just-restored online twin.
        for i in range(Self.N):
            hard_copy_params[Self.train_target, M=Self.CRITIC](
                self.ensemble.pairs[i].online,
                self.ensemble.pairs[i].target_net,
                self.ctx,
            )


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────


def fexp_to_log(alpha: Scalar[DT]) -> Scalar[DT]:
    """Log(α) — for seeding ScalarAdam.value (which holds log_α). Wrapper
    around std.math.log; named to avoid clashing with the `flog` import
    used elsewhere in the module."""
    from std.math import log as _flog

    return _flog(alpha)

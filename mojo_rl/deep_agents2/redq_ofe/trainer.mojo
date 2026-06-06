"""REDQOFETrainer — minimal Phase O.2.b.3 (CPU) orchestrator.

Composes all of the O.2.x blocks into one train_step. **Replay / driver
/ checkpoint plumbing is deliberately out of scope** — those land in
O.2.b.4 (driver wiring, Pendulum smoke) and O.2.b.5 (one-file v2
checkpoint over actor + N critics + SB + AB + PRED + opts). What this
file gates is the orchestration shape: that all five OFE blocks chain
correctly with the reused REDQ polyak + SAC α update, that the
critic / aux / actor losses descend on a synthetic fixed minibatch,
and that the gradient gating contract (RL path → input_only on SB +
AB + critics; aux path → all-mode on SB + AB + PRED) holds end-to-end.

Architectural notes
===================

(a) Shared OFE networks. `state_branch`, `action_branch`, `predictor`
    are OWNED by the trainer (not by individual blocks). They are
    referenced by `feat_blk`, `target_y_blk`, `critic_blk`,
    `actor_blk`, and `aux_blk` — sharing is non-negotiable since all
    five paths must operate on the same OFE params.

(b) Cache aliasing safety. Forward calls on SB / AB during the RL
    pipeline (`feat_blk`, `target_y_blk`, `critic_blk`, `actor_blk`)
    populate their caches, which get clobbered by subsequent
    forwards. The RL path NEVER calls SB.vjp or AB.vjp, so the
    clobber is harmless. The aux step (`aux_blk`) runs SB+AB+PRED
    forward+vjp atomically at the END of train_step, so its caches
    are valid for vjp.

(c) Aux cadence. Legacy redq_ofe.mojo runs aux on its own fresh
    minibatch every critic step. For this minimal trainer the aux
    step runs ONCE per train_step (per env step), on the same
    minibatch the RL UTD loop consumed. This is the simplest cadence
    that gates the orchestration; revisit before Pendulum smoke if
    convergence stalls.

(d) Reused blocks.
      * `EnsemblePolyakStep[CRITIC, N, OBS, ACT, BATCH]` — REDQ's
        polyak is OFE-agnostic (just polyaks N target critics).
      * `AlphaUpdateStep[OBS, ACT, BATCH]` — reads
        `state.log_prob_mean` (host scalar); OFE-agnostic.
    Both come in unchanged from `redq/` / `sac/`.

(e) State.mb_s/mb_a/mb_r/mb_sp/mb_d. These remain the raw replay-
    sampled obs / act / reward / next-obs / done. The OFE blocks
    populate their own φ-scratches; the trainer does NOT extend
    TrainerState. This keeps replay / sample / driver code
    compatible with what REDQ already uses.
"""

from std.math import exp as fexp, tanh as ftanh
from std.random import random_float64
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn2.core.checkpoint import (
    save_state_v2_body,
    load_state_v2_body,
    save_state_v2_body_gpu,
    load_state_v2_body_gpu,
)
from mojo_rl.nn2.core.map_params import hard_copy_params
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.core.scratch import Scratch
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.optimizer.scalar_adam import ScalarAdam

from ..redq.trainer import (
    _redq_warmup_uniform_kernel,
    _redq_action_clamp_kernel,
)

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

from ..training.trainer_block import TrainerState
from ..training.episode_tracker import EpisodeTracker
from ..training.blocks.sample_block import SampleBlock
from ..training.driver_offpolicy import OffPolicyAgent
from mojo_rl.core.logger import Logger
from ..sac.blocks.alpha_update_step import AlphaUpdateStep
from ..redq.ensemble import CriticEnsemble
from ..redq.blocks.ensemble_polyak_step import EnsemblePolyakStep
from ..redq.kernels import REDQ_TARGET_MIN, REDQ_TARGET_AVE

from .feature_step import OFEFeatureStep
from .ensemble_target_y_block_ofe import EnsembleTargetYBlockOFE
from .ensemble_critic_step_ofe import EnsembleCriticStepOFE
from .ensemble_actor_step_ofe import EnsembleActorStepOFE
from .aux_loss_step import OFEAuxLossStep
from .metrics import REDQOFEMetrics


# ────────────────────────────────────────────────────────────────────
# Helper: log(α) — same role as redq/trainer.mojo's `fexp_to_log`.
# ────────────────────────────────────────────────────────────────────


def _log_alpha(alpha: Scalar[DT]) -> Scalar[DT]:
    from std.math import log as _flog

    return _flog(alpha)


# ────────────────────────────────────────────────────────────────────
# Result struct — returned by train_step_inner so the test (and
# future driver) can inspect per-step losses without scraping
# trainer state.
# ────────────────────────────────────────────────────────────────────


@fieldwise_init
struct REDQOFEStepResult(Movable & ImplicitlyDestructible):
    var critic_loss: Scalar[DT]
    var actor_loss: Scalar[DT]
    var log_prob_mean: Scalar[DT]
    var alpha: Scalar[DT]
    var aux_loss: Scalar[DT]
    var did_actor_step: Bool


# ────────────────────────────────────────────────────────────────────
# REDQOFETrainer
# ────────────────────────────────────────────────────────────────────


struct REDQOFETrainer[
    train_target: StaticString,  # "cpu" or "gpu"
    SAMPLE: SampleBlock,  # owns replay buffer; provides setup/add/step
    ACTOR: Module,  # IN=PHI_S_DIM, OUT=2·ACT
    CRITIC: Module,  # IN=PHI_SA_DIM, OUT=1
    SB: Module,  # IN=OBS, OUT=PHI_S_DIM
    AB: Module,  # IN=PHI_S_DIM+ACT, OUT=PHI_SA_DIM
    PRED: Module,  # IN=PHI_SA_DIM, OUT=OBS
    N: Int,
    N_MIN: Int,
    UTD: Int,
    POLICY_DELAY: Int,
    Q_MODE: Int,
](OffPolicyAgent):
    # Dims derived from the sample block — caller specifies them ONCE
    # on the SAMPLE type, the trainer threads them everywhere.
    comptime OBS = Self.SAMPLE.OBS
    comptime ACT = Self.SAMPLE.ACT
    comptime BATCH = Self.SAMPLE.BATCH
    comptime PHI_S_DIM = Self.SB.OUT_DIM

    # OffPolicyAgent trait aliases.
    comptime AGENT_TRAIN_TARGET: StaticString = Self.train_target
    comptime AGENT_OBS_DIM: Int = Self.OBS
    comptime AGENT_ACT_DIM: Int = Self.ACT
    comptime PHI_SA_DIM = Self.AB.OUT_DIM

    # ── Owned networks ────────────────────────────────────────────────
    var actor: Self.ACTOR
    var state_branch: Self.SB
    var action_branch: Self.AB
    var predictor: Self.PRED
    var ensemble: CriticEnsemble[Self.CRITIC, Self.N]

    # ── Owned optimizers ──────────────────────────────────────────────
    var actor_opt: Adam
    var sb_opt: Adam
    var ab_opt: Adam
    var pred_opt: Adam
    var alpha_opt: ScalarAdam  # holds log_α

    # ── Owned blocks ──────────────────────────────────────────────────
    var feat_blk: OFEFeatureStep[Self.SB, Self.OBS, Self.ACT, Self.BATCH]
    var target_y_blk: EnsembleTargetYBlockOFE[
        Self.ACTOR,
        Self.AB,
        Self.CRITIC,
        Self.N,
        Self.BATCH,
        Self.PHI_S_DIM,
        Self.ACT,
        Self.N_MIN,
        Self.Q_MODE,
    ]
    var critic_blk: EnsembleCriticStepOFE[
        Self.AB,
        Self.CRITIC,
        Self.N,
        Self.BATCH,
        Self.PHI_S_DIM,
        Self.ACT,
    ]
    var actor_blk: EnsembleActorStepOFE[
        Self.ACTOR,
        Self.AB,
        Self.CRITIC,
        Self.N,
        Self.BATCH,
        Self.PHI_S_DIM,
        Self.ACT,
    ]
    var aux_blk: OFEAuxLossStep[
        Self.SB,
        Self.AB,
        Self.PRED,
        Self.OBS,
        Self.ACT,
        Self.BATCH,
    ]
    # Reused as-is from REDQ / SAC (OFE-agnostic).
    var polyak_blk: EnsemblePolyakStep[
        Self.CRITIC,
        Self.N,
        Self.OBS,
        Self.ACT,
        Self.BATCH,
    ]
    var alpha_blk: AlphaUpdateStep[Self.OBS, Self.ACT, Self.BATCH]
    var sample_blk: Self.SAMPLE

    # ── Trainer state ─────────────────────────────────────────────────
    var state: TrainerState[Self.OBS, Self.ACT, Self.BATCH]
    var tracker: EpisodeTracker

    # DeviceContext: None on CPU, Some(ctx) on GPU. The ctx threads
    # through every block call so per-leaf DeviceContext() creation
    # doesn't exhaust Apple Metal's queue pool
    # (feedback_apple_metal_devicecontext_per_call).
    var ctx: Optional[DeviceContext]

    # ── Single-env action-selection scratches ────────────────────────
    var _ob1: Scratch["ob1", Self.OBS, True]
    var _phi_s1: Scratch["phi_s1", Self.PHI_S_DIM, True]
    var _ao1: Scratch["ao1", 2 * Self.ACT, True]
    var _alp1: Scratch["alp1", Self.ACT + 1, True]

    # Lazy-grown φ(s) scratch for the batched action surface.
    # Sized to `last_seen_N_ENVS * PHI_S_DIM`. Initially empty; first
    # `select_action_batched` call grows it. Used only on the
    # warmup-passed branch (the warmup branch never runs the OFE
    # forward, so no allocation is needed during warmup).
    var _phi_s_batched_cpu: List[Scalar[DT]]
    var _phi_s_batched_cap: Int

    # GPU mirror — lazy-grown DeviceBuffer for the batched φ(s)
    # scratch on the GPU action-selection path. None on CPU.
    var _phi_s_batched_dev: Optional[DeviceBuffer[DT]]
    var _phi_s_batched_dev_cap: Int

    # Philox warmup counters — advanced by `select_action_batched`
    # on each warmup call.
    var _warmup_rng_seed: UInt64
    var _warmup_rng_offset: UInt64

    # ── Hyperparams (kept as fields so the test can inspect them) ─────
    var action_scale: Scalar[DT]
    var tau: Scalar[DT]
    var gamma: Scalar[DT]
    var target_entropy: Scalar[DT]
    var learning_starts: Int
    var _inner_count: Int
    var _total_train_steps: Int

    # ── Per-flush-window accumulators ─────────────────────────────────
    var _acc_critic_loss: Scalar[DT]
    var _acc_actor_loss: Scalar[DT]
    var _acc_alpha: Scalar[DT]
    var _acc_lp_mean: Scalar[DT]
    var _acc_aux_loss: Scalar[DT]
    var _acc_n_updates: Int
    var _acc_n_actor_updates: Int

    # ── Defaultable ───────────────────────────────────────────────────

    def __init__(out self):
        self.actor = Self.ACTOR()
        self.state_branch = Self.SB()
        self.action_branch = Self.AB()
        self.predictor = Self.PRED()
        self.ensemble = CriticEnsemble[Self.CRITIC, Self.N]()
        self.actor_opt = Adam()
        self.sb_opt = Adam()
        self.ab_opt = Adam()
        self.pred_opt = Adam()
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
        self.feat_blk = OFEFeatureStep[
            Self.SB,
            Self.OBS,
            Self.ACT,
            Self.BATCH,
        ]()
        self.target_y_blk = EnsembleTargetYBlockOFE[
            Self.ACTOR,
            Self.AB,
            Self.CRITIC,
            Self.N,
            Self.BATCH,
            Self.PHI_S_DIM,
            Self.ACT,
            Self.N_MIN,
            Self.Q_MODE,
        ]()
        self.critic_blk = EnsembleCriticStepOFE[
            Self.AB,
            Self.CRITIC,
            Self.N,
            Self.BATCH,
            Self.PHI_S_DIM,
            Self.ACT,
        ]()
        self.actor_blk = EnsembleActorStepOFE[
            Self.ACTOR,
            Self.AB,
            Self.CRITIC,
            Self.N,
            Self.BATCH,
            Self.PHI_S_DIM,
            Self.ACT,
        ]()
        self.aux_blk = OFEAuxLossStep[
            Self.SB,
            Self.AB,
            Self.PRED,
            Self.OBS,
            Self.ACT,
            Self.BATCH,
        ]()
        self.polyak_blk = EnsemblePolyakStep[
            Self.CRITIC,
            Self.N,
            Self.OBS,
            Self.ACT,
            Self.BATCH,
        ]()
        self.alpha_blk = AlphaUpdateStep[
            Self.OBS,
            Self.ACT,
            Self.BATCH,
        ]()
        self.sample_blk = Self.SAMPLE()
        self.state = TrainerState[Self.OBS, Self.ACT, Self.BATCH]()
        self.tracker = EpisodeTracker(
            window=List[Scalar[DT]](),
            window_size=0,
            idx=0,
            current_return=Scalar[DT](0.0),
            ep_count=0,
        )
        self.ctx = None
        self._ob1 = Scratch["ob1", Self.OBS, True]()
        self._phi_s1 = Scratch["phi_s1", Self.PHI_S_DIM, True]()
        self._ao1 = Scratch["ao1", 2 * Self.ACT, True]()
        self._alp1 = Scratch["alp1", Self.ACT + 1, True]()
        self._phi_s_batched_cpu = List[Scalar[DT]]()
        self._phi_s_batched_cap = 0
        self._phi_s_batched_dev = None
        self._phi_s_batched_dev_cap = 0
        self._warmup_rng_seed = UInt64(0xC0FFEE_C0DE)
        self._warmup_rng_offset = UInt64(0)
        self.action_scale = Scalar[DT](1.0)
        self.tau = Scalar[DT](0.005)
        self.gamma = Scalar[DT](0.99)
        self.target_entropy = -Scalar[DT](Self.ACT)
        self.learning_starts = 1_000
        self._inner_count = 0
        self._total_train_steps = 0
        self._acc_critic_loss = Scalar[DT](0.0)
        self._acc_actor_loss = Scalar[DT](0.0)
        self._acc_alpha = Scalar[DT](0.0)
        self._acc_lp_mean = Scalar[DT](0.0)
        self._acc_aux_loss = Scalar[DT](0.0)
        self._acc_n_updates = 0
        self._acc_n_actor_updates = 0

    # ── Factory ───────────────────────────────────────────────────────

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        actor_lr: Scalar[DT] = Scalar[DT](3e-4),
        critic_lr: Scalar[DT] = Scalar[DT](3e-4),
        ofe_lr: Scalar[DT] = Scalar[DT](3e-4),
        alpha_lr: Scalar[DT] = Scalar[DT](3e-4),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        tau: Scalar[DT] = Scalar[DT](0.005),
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        init_alpha: Scalar[DT] = Scalar[DT](0.2),
        target_entropy: Scalar[DT] = Scalar[DT](-1.0),
        learning_starts: Int = 1_000,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
    ) raises -> Self:
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "REDQ-OFE: train_target must be 'cpu' or 'gpu'"
        comptime if Self.train_target == "gpu":
            if not ctx:
                raise Error(
                    "REDQOFETrainer.make[train_target='gpu']: ctx required"
                )
        comptime assert Self.N >= 2, "REDQ-OFE: N must be ≥ 2"
        comptime assert Self.N_MIN >= 1, "REDQ-OFE: N_MIN ≥ 1"
        comptime assert Self.N_MIN <= Self.N, "REDQ-OFE: N_MIN ≤ N"
        comptime assert Self.UTD >= 1, "REDQ-OFE: UTD ≥ 1"
        comptime assert Self.POLICY_DELAY >= 1, "REDQ-OFE: POLICY_DELAY ≥ 1"
        comptime assert (
            Self.Q_MODE == REDQ_TARGET_MIN or Self.Q_MODE == REDQ_TARGET_AVE
        ), "REDQ-OFE: Q_MODE must be MIN (0) or AVE (1)"
        comptime assert (
            Self.PRED.OUT_DIM == Self.OBS
        ), "REDQ-OFE: predictor OUT must equal OBS"

        var t = Self()
        t.ctx = ctx

        # Networks.
        t.actor = Self.ACTOR.make[Self.train_target, Xavier](ctx=ctx)
        t.state_branch = Self.SB.make[Self.train_target, Xavier](ctx=ctx)
        t.action_branch = Self.AB.make[Self.train_target, Xavier](ctx=ctx)
        t.predictor = Self.PRED.make[Self.train_target, Xavier](ctx=ctx)
        t.ensemble = CriticEnsemble[Self.CRITIC, Self.N].make[
            Self.train_target,
            Xavier,
        ](ctx=ctx)

        # Optimizers.
        t.actor_opt = Adam.make[Self.train_target, M=Self.ACTOR](
            t.actor, ctx=ctx,
        )
        t.sb_opt = Adam.make[Self.train_target, M=Self.SB](
            t.state_branch, ctx=ctx,
        )
        t.ab_opt = Adam.make[Self.train_target, M=Self.AB](
            t.action_branch, ctx=ctx,
        )
        t.pred_opt = Adam.make[Self.train_target, M=Self.PRED](
            t.predictor, ctx=ctx,
        )
        t.actor_opt.lr = actor_lr
        t.sb_opt.lr = ofe_lr
        t.ab_opt.lr = ofe_lr
        t.pred_opt.lr = ofe_lr
        for i in range(Self.N):
            t.ensemble.opts[i].lr = critic_lr
        t.alpha_opt = ScalarAdam.new(_log_alpha(init_alpha), alpha_lr)

        # Blocks.
        t.feat_blk = OFEFeatureStep[
            Self.SB,
            Self.OBS,
            Self.ACT,
            Self.BATCH,
        ].make[Self.train_target](ctx=ctx)
        t.target_y_blk = EnsembleTargetYBlockOFE[
            Self.ACTOR,
            Self.AB,
            Self.CRITIC,
            Self.N,
            Self.BATCH,
            Self.PHI_S_DIM,
            Self.ACT,
            Self.N_MIN,
            Self.Q_MODE,
        ].make[Self.train_target](
            action_scale=action_scale,
            gamma=gamma,
            ctx=ctx,
        )
        t.critic_blk = EnsembleCriticStepOFE[
            Self.AB,
            Self.CRITIC,
            Self.N,
            Self.BATCH,
            Self.PHI_S_DIM,
            Self.ACT,
        ].make[Self.train_target](ctx=ctx)
        t.actor_blk = EnsembleActorStepOFE[
            Self.ACTOR,
            Self.AB,
            Self.CRITIC,
            Self.N,
            Self.BATCH,
            Self.PHI_S_DIM,
            Self.ACT,
        ].make[Self.train_target](
            action_scale=action_scale, ctx=ctx,
        )
        t.aux_blk = OFEAuxLossStep[
            Self.SB,
            Self.AB,
            Self.PRED,
            Self.OBS,
            Self.ACT,
            Self.BATCH,
        ].make[Self.train_target](ctx=ctx)
        t.polyak_blk = EnsemblePolyakStep[
            Self.CRITIC,
            Self.N,
            Self.OBS,
            Self.ACT,
            Self.BATCH,
        ].make(tau=tau)
        t.alpha_blk = AlphaUpdateStep[
            Self.OBS,
            Self.ACT,
            Self.BATCH,
        ].make(target_entropy=target_entropy)
        t.sample_blk = Self.SAMPLE()
        t.sample_blk.setup(learning_starts, ctx=ctx)

        # State + tracker.
        t.state = TrainerState[
            Self.OBS,
            Self.ACT,
            Self.BATCH,
        ].make[Self.train_target](ctx=ctx)
        t.tracker = EpisodeTracker.new(window_size, initial_episode_fill)

        # Action-selection scratches — STAGING=True so host mirrors
        # exist on both CPU and GPU (the action-selection helpers
        # H2D the obs, run the device forward, then D2H the action).
        comptime if Self.train_target == "cpu":
            t._ob1 = Scratch["ob1", Self.OBS, True].make_cpu()
            t._phi_s1 = Scratch[
                "phi_s1", Self.PHI_S_DIM, True,
            ].make_cpu()
            t._ao1 = Scratch["ao1", 2 * Self.ACT, True].make_cpu()
            t._alp1 = Scratch["alp1", Self.ACT + 1, True].make_cpu()
        else:
            var c = ctx.value()
            t._ob1 = Scratch["ob1", Self.OBS, True].make_gpu(c)
            t._phi_s1 = Scratch[
                "phi_s1", Self.PHI_S_DIM, True,
            ].make_gpu(c)
            t._ao1 = Scratch["ao1", 2 * Self.ACT, True].make_gpu(c)
            t._alp1 = Scratch["alp1", Self.ACT + 1, True].make_gpu(c)
        t._phi_s_batched_cpu = List[Scalar[DT]]()
        t._phi_s_batched_cap = 0
        t._phi_s_batched_dev = None
        t._phi_s_batched_dev_cap = 0
        t._warmup_rng_seed = UInt64(0xC0FFEE_C0DE)
        t._warmup_rng_offset = UInt64(0)

        # Hyperparam fields.
        t.action_scale = action_scale
        t.tau = tau
        t.gamma = gamma
        t.target_entropy = target_entropy
        t.learning_starts = learning_starts
        t._inner_count = 0
        t._total_train_steps = 0
        t._acc_critic_loss = Scalar[DT](0.0)
        t._acc_actor_loss = Scalar[DT](0.0)
        t._acc_alpha = Scalar[DT](0.0)
        t._acc_lp_mean = Scalar[DT](0.0)
        t._acc_aux_loss = Scalar[DT](0.0)
        t._acc_n_updates = 0
        t._acc_n_actor_updates = 0

        return t^

    # ── Direct state-write helpers (used by tests / drivers in
    # downstream slices). The driver wiring slice will replace these
    # with a SampleBlock-driven path. ───────────────────────────────

    def write_minibatch_cpu(
        mut self,
        obs: List[Scalar[DT]],
        act: List[Scalar[DT]],
        rew: List[Scalar[DT]],
        next_obs: List[Scalar[DT]],
        done: List[Scalar[DT]],
    ) raises:
        """Fill `state.mb_*` from caller-provided host lists. Lengths
        must match BATCH·{OBS, ACT, 1, OBS, 1}."""
        if len(obs) != Self.BATCH * Self.OBS:
            raise Error("write_minibatch_cpu: obs length mismatch")
        if len(act) != Self.BATCH * Self.ACT:
            raise Error("write_minibatch_cpu: act length mismatch")
        if len(rew) != Self.BATCH:
            raise Error("write_minibatch_cpu: rew length mismatch")
        if len(next_obs) != Self.BATCH * Self.OBS:
            raise Error("write_minibatch_cpu: next_obs length mismatch")
        if len(done) != Self.BATCH:
            raise Error("write_minibatch_cpu: done length mismatch")
        var mb_s_p = self.state.mb_s.cpu_ptr()
        var mb_a_p = self.state.mb_a.cpu_ptr()
        var mb_r_p = self.state.mb_r.cpu_ptr()
        var mb_sp_p = self.state.mb_sp.cpu_ptr()
        var mb_d_p = self.state.mb_d.cpu_ptr()
        for i in range(Self.BATCH * Self.OBS):
            mb_s_p[i] = obs[i]
            mb_sp_p[i] = next_obs[i]
        for i in range(Self.BATCH * Self.ACT):
            mb_a_p[i] = act[i]
        for b in range(Self.BATCH):
            mb_r_p[b] = rew[b]
            mb_d_p[b] = done[b]

    # ──────────────────────────────────────────────────────────────────
    # The orchestration. Three entry points:
    #   `_one_inner_tick`  → ONE RL tick on the current `state.mb_*`
    #   `train_step_inner` → UTD ticks on a SHARED minibatch + aux.
    #                        Used by the fixed-batch overfit gate
    #                        (test_redq_ofe_trainer_cpu) so the
    #                        existing test contract is preserved.
    #   `train_step`       → PRODUCTION path. Samples per inner tick
    #                        (REDQ paper-faithful cadence) + aux at
    #                        the end.
    # ──────────────────────────────────────────────────────────────────

    def _one_inner_tick[
        POLICY: AMPPolicy = NoAMP
    ](mut self,) raises -> Tuple[Scalar[DT], Bool, Scalar[DT], Scalar[DT]]:
        """Run ONE inner critic tick on the CURRENT `state.mb_*`:
        feature pre-pass → target_y → critic update → polyak → (if
        cadence fires) actor + α. Increments `_inner_count`. Returns
        `(critic_loss, did_actor_step, actor_loss, lp_mean)`. Caller
        is responsible for sampling into `state.mb_*` before the
        call."""
        self._inner_count += 1
        self.target_y_blk.resample_subset_idxs()
        var alpha_val = fexp(self.alpha_opt.value)
        self.state.alpha = alpha_val

        var mb_a_p = self.state.mb_a.target_ptr[Self.train_target]()
        var mb_r_p = self.state.mb_r.target_ptr[Self.train_target]()
        var mb_d_p = self.state.mb_d.target_ptr[Self.train_target]()
        var mb_y_p = self.state.mb_y.target_ptr[Self.train_target]()

        # (1) Feature pre-pass.
        self.feat_blk.step[Self.train_target](
            self.state_branch, self.state,
        )
        var phi_s_p = self.feat_blk.phi_s_ptr[Self.train_target]()
        var phi_sp_p = self.feat_blk.phi_sp_ptr[Self.train_target]()

        # (2) Target y.
        self.target_y_blk.step[Self.train_target](
            self.actor,
            self.action_branch,
            self.ensemble,
            phi_sp_p,
            mb_r_p,
            mb_d_p,
            alpha_val,
            mb_y_p,
        )

        # (3) Critic update.
        var cl = self.critic_blk.step[Self.train_target](
            self.action_branch,
            self.ensemble,
            phi_s_p,
            mb_a_p,
            mb_y_p,
        )

        # (4) Polyak every inner tick (paper-faithful).
        self.polyak_blk.step[Self.train_target](
            self.state, self.ensemble,
        )

        # (5) Actor + α every POLICY_DELAY.
        var did_actor: Bool = False
        var actor_loss: Scalar[DT] = Scalar[DT](0.0)
        var lp_mean: Scalar[DT] = Scalar[DT](0.0)
        if self._inner_count % Self.POLICY_DELAY == 0:
            var res = self.actor_blk.forward_backward[Self.train_target](
                self.actor,
                self.actor_opt,
                self.action_branch,
                self.ensemble,
                phi_s_p,
                alpha_val,
            )
            self.state.actor_loss = res.loss
            self.state.log_prob_mean = res.log_prob_mean
            actor_loss = res.loss
            lp_mean = res.log_prob_mean
            did_actor = True
            # α stays a host scalar — `state.log_prob_mean` is host-
            # populated by the actor-step's host-side reduction on
            # both CPU and GPU.
            self.alpha_blk.step["cpu"](self.state, self.alpha_opt)

        return (cl, did_actor, actor_loss, lp_mean)

    def _run_aux_step[
        POLICY: AMPPolicy = NoAMP
    ](mut self,) raises -> Scalar[DT]:
        """One aux loss step on the CURRENT `state.mb_*`. Forward+vjp
        on SB+AB+PRED is atomic so prior RL forwards clobbering caches
        is harmless. Returns the MSE loss."""
        return self.aux_blk.step[Self.train_target](
            self.state_branch,
            self.action_branch,
            self.predictor,
            self.sb_opt,
            self.ab_opt,
            self.pred_opt,
            self.state,
        )

    def train_step_inner[
        POLICY: AMPPolicy = NoAMP
    ](mut self,) raises -> REDQOFEStepResult:
        """UTD inner ticks on a SHARED `state.mb_*` + ONE aux step.
        Used by the fixed-batch overfit gate (the caller fills
        `state.mb_*` once via `write_minibatch_cpu` before calling).
        Production code should use `train_step` instead, which
        re-samples per inner tick (paper-faithful)."""
        var critic_loss_acc: Scalar[DT] = Scalar[DT](0.0)
        var actor_loss_last: Scalar[DT] = Scalar[DT](0.0)
        var lp_mean_last: Scalar[DT] = Scalar[DT](0.0)
        var did_actor_step: Bool = False
        for _ in range(Self.UTD):
            var tick = self._one_inner_tick[POLICY]()
            critic_loss_acc += tick[0]
            if tick[1]:
                did_actor_step = True
                actor_loss_last = tick[2]
                lp_mean_last = tick[3]
        var aux_loss = self._run_aux_step[POLICY]()
        return REDQOFEStepResult(
            critic_loss=critic_loss_acc,
            actor_loss=actor_loss_last,
            log_prob_mean=lp_mean_last,
            alpha=fexp(self.alpha_opt.value),
            aux_loss=aux_loss,
            did_actor_step=did_actor_step,
        )

    # ── Read-only accessors used by tests / drivers ─────────────────

    def alpha_value(self) -> Scalar[DT]:
        return fexp(self.alpha_opt.value)

    def inner_count(self) -> Int:
        return self._inner_count

    def total_train_steps(self) -> Int:
        return self._total_train_steps

    # ──────────────────────────────────────────────────────────────────
    # Outer train_step — drives the env loop. Calls sample_blk.step to
    # populate state.mb_* from the replay buffer (gates warmup +
    # buffer-readiness), then runs the OFE pipeline.
    # ──────────────────────────────────────────────────────────────────

    def train_step(mut self, step_idx: Int) raises -> Bool:
        """Outer train step — paper-faithful cadence. Sample → inner
        tick → (sample → inner tick) × UTD-1 → ONE aux step.

        Each inner critic tick uses a FRESH minibatch — matches REDQ's
        reference schedule from Chen et al. 2021 (and the legacy
        `redq/trainer.mojo` cadence). The aux step at the end runs on
        whichever minibatch the LAST inner tick used; the OFE
        feature pre-pass is hoisted INTO each inner tick (it has to
        be — φ(s) feeds the target_y + critic + actor blocks).

        Returns True if any inner update ran (False during warmup /
        when buffer < BATCH). `_total_train_steps` increments by the
        number of inner ticks that actually fired (could be < UTD if
        the buffer drains mid-loop — single-env: never)."""
        self.state.step_idx = step_idx
        self.state.did_step = True
        # Thread the trainer's ctx into state so blocks that need it
        # (polyak GPU, sample_blk GPU, etc.) can read it through.
        self.state.ctx = self.ctx

        # First sample gates warmup + buffer readiness.
        self.sample_blk.step(self.state)
        if not self.state.did_step:
            return False

        # Tick 1 on the first sample.
        var critic_loss_acc: Scalar[DT] = Scalar[DT](0.0)
        var actor_loss_last: Scalar[DT] = Scalar[DT](0.0)
        var lp_mean_last: Scalar[DT] = Scalar[DT](0.0)
        var did_actor_step: Bool = False
        var ticks_fired = 0

        var tick0 = self._one_inner_tick[NoAMP]()
        critic_loss_acc += tick0[0]
        if tick0[1]:
            did_actor_step = True
            actor_loss_last = tick0[2]
            lp_mean_last = tick0[3]
        ticks_fired += 1

        # Inner ticks 2..UTD — fresh sample per tick.
        for _ in range(Self.UTD - 1):
            self.state.did_step = True
            self.sample_blk.step(self.state)
            if not self.state.did_step:
                break  # buffer drained mid-iter (single-env: never)
            var tick = self._one_inner_tick[NoAMP]()
            critic_loss_acc += tick[0]
            if tick[1]:
                did_actor_step = True
                actor_loss_last = tick[2]
                lp_mean_last = tick[3]
            ticks_fired += 1

        # Aux loss step — runs ONCE per outer call, on the LAST
        # sampled minibatch.
        var aux_loss = self._run_aux_step[NoAMP]()

        self._total_train_steps += ticks_fired

        # Drain into accumulators (consumed by the next flush_metrics).
        self._acc_critic_loss += critic_loss_acc
        self._acc_alpha += fexp(self.alpha_opt.value)
        self._acc_aux_loss += aux_loss
        self._acc_n_updates += 1
        if did_actor_step:
            self._acc_actor_loss += actor_loss_last
            self._acc_lp_mean += lp_mean_last
            self._acc_n_actor_updates += 1

        return True

    # ──────────────────────────────────────────────────────────────────
    # Action-selection surface (single-env CPU).
    # ──────────────────────────────────────────────────────────────────

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        """Stochastic single-env action selection. Warmup (step <
        learning_starts) → uniform random in [-action_scale, +scale].
        Otherwise → state_branch(obs) → φ(s); actor(φ(s)) → ao;
        rsample(ao) → action; clamp."""

        if step_idx < self.learning_starts:
            for j in range(Self.ACT):
                var u = Scalar[DT](2.0 * random_float64() - 1.0)
                action_out[j] = u * self.action_scale
            return

        # Stage obs into _ob1 host mirror.
        var ob1_cpu_p = self._ob1.cpu_ptr()
        for d in range(Self.OBS):
            ob1_cpu_p[d] = obs[d]
        comptime if Self.train_target == "gpu":
            self.ctx.value().enqueue_copy(
                self._ob1.dev.value(), ob1_cpu_p,
            )

        # Device/host TileTensor views using `target_ptr`.
        var ob1_p = self._ob1.target_ptr[Self.train_target]()
        var phi_s1_p = self._phi_s1.target_ptr[Self.train_target]()
        var ao1_p = self._ao1.target_ptr[Self.train_target]()
        var alp1_p = self._alp1.target_ptr[Self.train_target]()

        var ob1_t = TileTensor(ob1_p, row_major[1, Self.OBS]())
        var phi_s1_t = TileTensor(
            phi_s1_p, row_major[1, Self.PHI_S_DIM](),
        )
        var ao1_t = TileTensor(ao1_p, row_major[1, 2 * Self.ACT]())
        var alp1_t = TileTensor(alp1_p, row_major[1, Self.ACT + 1]())

        # φ(s) = state_branch.forward(obs).
        self.state_branch.forward[Self.train_target, 1](
            ob1_t, output=phi_s1_t,
        )
        # actor.forward(φ(s)) → ao.
        self.actor.forward[Self.train_target, 1](
            phi_s1_t, output=ao1_t,
        )
        # rsample(ao) via the actor_blk's rsample primitive.
        self.actor_blk.rsample.forward[Self.train_target, 1](
            ao1_t, output=alp1_t,
        )

        # D2H alp1 on GPU; read host-side for clamp + emit.
        comptime if Self.train_target == "gpu":
            var ctx = self.ctx.value()
            ctx.enqueue_copy(self._alp1.cpu_ptr(), self._alp1.dev.value())
            ctx.synchronize()
        var alp1_cpu_p = self._alp1.cpu_ptr()
        for j in range(Self.ACT):
            var a = alp1_cpu_p[j]
            if a > self.action_scale:
                a = self.action_scale
            elif a < -self.action_scale:
                a = -self.action_scale
            action_out[j] = a

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        """Deterministic eval: tanh(actor.mean) · action_scale, clamped.
        Skips the rsample noise — uses the actor's mean head only."""
        var ob1_cpu_p = self._ob1.cpu_ptr()
        for d in range(Self.OBS):
            ob1_cpu_p[d] = obs[d]
        comptime if Self.train_target == "gpu":
            self.ctx.value().enqueue_copy(
                self._ob1.dev.value(), ob1_cpu_p,
            )

        var ob1_p = self._ob1.target_ptr[Self.train_target]()
        var phi_s1_p = self._phi_s1.target_ptr[Self.train_target]()
        var ao1_p = self._ao1.target_ptr[Self.train_target]()

        var ob1_t = TileTensor(ob1_p, row_major[1, Self.OBS]())
        var phi_s1_t = TileTensor(
            phi_s1_p, row_major[1, Self.PHI_S_DIM](),
        )
        var ao1_t = TileTensor(ao1_p, row_major[1, 2 * Self.ACT]())

        self.state_branch.forward[Self.train_target, 1](
            ob1_t, output=phi_s1_t,
        )
        self.actor.forward[Self.train_target, 1](
            phi_s1_t, output=ao1_t,
        )

        comptime if Self.train_target == "gpu":
            var ctx = self.ctx.value()
            ctx.enqueue_copy(self._ao1.cpu_ptr(), self._ao1.dev.value())
            ctx.synchronize()
        var ao1_cpu_p = self._ao1.cpu_ptr()
        for j in range(Self.ACT):
            var mean = ao1_cpu_p[j]
            var a = ftanh(mean) * self.action_scale
            if a > self.action_scale:
                a = self.action_scale
            elif a < -self.action_scale:
                a = -self.action_scale
            action_out[j] = a

    # ──────────────────────────────────────────────────────────────────
    # Replay-push + episode-tracker surface.
    # ──────────────────────────────────────────────────────────────────

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

    def end_episode(mut self):
        self.tracker.end_episode()

    def add_complete_return(mut self, ret: Scalar[DT]):
        self.tracker.add_complete_return(ret)

    def mean_return(self) -> Scalar[DT]:
        return self.tracker.mean_return()

    def ep_count(self) -> Int:
        return self.tracker.ep_count

    def flush_metrics(mut self) -> REDQOFEMetrics:
        """Drain per-flush-window accumulators into a `REDQOFEMetrics`
        snapshot and reset. Means use sum/count (0.0 sentinel if no
        updates fired this window — no NaN poisoning).

        Bundle fields are documented on `REDQOFEMetrics`. The driver
        / agent typically calls this on a `diag_every` cadence."""
        var inv_n: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](
            self._acc_n_updates
        ) if self._acc_n_updates > 0 else Scalar[DT](0.0)
        var inv_a: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](
            self._acc_n_actor_updates
        ) if self._acc_n_actor_updates > 0 else Scalar[DT](0.0)
        var m = REDQOFEMetrics(
            critic_loss=self._acc_critic_loss * inv_n,
            actor_loss=self._acc_actor_loss * inv_a,
            alpha=self._acc_alpha * inv_n,
            log_prob_mean=self._acc_lp_mean * inv_a,
            aux_loss=self._acc_aux_loss * inv_n,
            n_updates=self._acc_n_updates,
            n_actor_updates=self._acc_n_actor_updates,
        )
        self._acc_critic_loss = Scalar[DT](0.0)
        self._acc_actor_loss = Scalar[DT](0.0)
        self._acc_alpha = Scalar[DT](0.0)
        self._acc_lp_mean = Scalar[DT](0.0)
        self._acc_aux_loss = Scalar[DT](0.0)
        self._acc_n_updates = 0
        self._acc_n_actor_updates = 0
        return m^

    def flush_metrics_through_logger[L: Logger](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        """Trait-uniform cadence hook (overrides the no-op default) so the
        off-policy driver streams REDQ-OFE metrics at its `diag_every`
        cadence. `REDQOFEMetrics` holds plain host scalars (not `LogScalar`),
        so the fields are emitted explicitly rather than via `log_bundle`.
        Values are correct on both CPU and GPU — the accumulators are filled
        unconditionally in `train_step` (REDQ-OFE runs host control flow,
        no CUDA-graph capture)."""
        var m = self.flush_metrics()
        if Bool(logger):
            var lg = logger.value()
            lg[].log_scalar("critic_loss", Float64(m.critic_loss), step)
            lg[].log_scalar("actor_loss", Float64(m.actor_loss), step)
            lg[].log_scalar("alpha", Float64(m.alpha), step)
            lg[].log_scalar("log_prob_mean", Float64(m.log_prob_mean), step)
            lg[].log_scalar("aux_loss", Float64(m.aux_loss), step)
            lg[].log_scalar("n_updates", Float64(m.n_updates), step)
            lg[].log_scalar(
                "n_actor_updates", Float64(m.n_actor_updates), step
            )

    # ──────────────────────────────────────────────────────────────────
    # OffPolicyAgent trait methods — batched action surface + per-lane
    # replay push. The single-env helpers (`select_action`, `record`)
    # above stay available for non-driver callers.
    # ──────────────────────────────────────────────────────────────────

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
        """Single batched entry for the off-policy driver.
        Caller supplies obs / ao / alp / action buffers sized for
        `N_ENVS`. The internal `_phi_s_batched_*` scratch is grown
        lazily to `N_ENVS · PHI_S_DIM` on first call (host List on
        CPU, DeviceBuffer on GPU). CPU + GPU paths share the same
        forward pipeline (state_branch → actor → rsample → clamp);
        only warmup random + final clamp use different mechanisms
        per target."""
        comptime assert N_ENVS > 0, "N_ENVS must be > 0"

        # Warmup → uniform random in [-action_scale, +action_scale].
        if step_idx < self.learning_starts:
            comptime if Self.train_target == "cpu":
                for i in range(N_ENVS * Self.ACT):
                    var u = Scalar[DT](2.0 * random_float64() - 1.0)
                    action_ptr[i] = u * self.action_scale
            else:
                # GPU warmup: Philox kernel. Advance the host offset
                # by 2·N·A each call (step_uniform consumes 2 raw
                # uint32 lanes).
                var action_lt = LayoutTensor[
                    DT, Layout.row_major(N_ENVS, Self.ACT),
                    MutAnyOrigin,
                ](action_ptr)
                comptime total_w = N_ENVS * Self.ACT
                comptime n_blocks_w = (total_w + TPB - 1) // TPB
                comptime warmup_kernel = _redq_warmup_uniform_kernel[
                    N_ENVS, Self.ACT,
                ]
                var ctx = self.ctx.value()
                ctx.enqueue_function[warmup_kernel](
                    action_lt,
                    self.action_scale,
                    self._warmup_rng_seed,
                    self._warmup_rng_offset,
                    grid_dim=n_blocks_w, block_dim=TPB,
                )
                self._warmup_rng_offset += UInt64(N_ENVS * Self.ACT * 2)
            return

        # Lazy-grow φ(s) scratch on the appropriate target.
        var needed = N_ENVS * Self.PHI_S_DIM
        var phi_s_p: UnsafePointer[Scalar[DT], MutAnyOrigin]
        comptime if Self.train_target == "cpu":
            if self._phi_s_batched_cap < needed:
                self._phi_s_batched_cpu = List[Scalar[DT]](
                    length=needed, fill=Scalar[DT](0.0),
                )
                self._phi_s_batched_cap = needed
            phi_s_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self._phi_s_batched_cpu.unsafe_ptr()
            )
        else:
            if self._phi_s_batched_dev_cap < needed:
                self._phi_s_batched_dev = (
                    self.ctx.value().enqueue_create_buffer[DT](needed)
                )
                self._phi_s_batched_dev_cap = needed
            phi_s_p = self._phi_s_batched_dev.value().unsafe_ptr()

        # (1) state_branch(obs_ptr) → phi_s_p.
        var obs_t = TileTensor(obs_ptr, row_major[N_ENVS, Self.OBS]())
        var phi_s_t = TileTensor(
            phi_s_p, row_major[N_ENVS, Self.PHI_S_DIM](),
        )
        self.state_branch.forward[Self.train_target, N_ENVS](
            obs_t, output=phi_s_t,
        )

        # (2) actor(phi_s) → ao.
        var ao_t = TileTensor(
            ao_scratch_ptr,
            row_major[N_ENVS, 2 * Self.ACT](),
        )
        self.actor.forward[Self.train_target, N_ENVS](
            phi_s_t, output=ao_t,
        )

        # (3) rsample(ao) → alp = (action | log_prob).
        var alp_t = TileTensor(
            alp_scratch_ptr,
            row_major[N_ENVS, Self.ACT + 1](),
        )
        self.actor_blk.rsample.forward[Self.train_target, N_ENVS](
            ao_t, output=alp_t,
        )

        # (4) Clamp into action_ptr (drop the log_prob slot).
        comptime if Self.train_target == "cpu":
            for env_idx in range(N_ENVS):
                var src = alp_scratch_ptr + env_idx * (Self.ACT + 1)
                var dst = action_ptr + env_idx * Self.ACT
                for j in range(Self.ACT):
                    var a = src[j]
                    if a > self.action_scale:
                        a = self.action_scale
                    elif a < -self.action_scale:
                        a = -self.action_scale
                    dst[j] = a
        else:
            var alp_lt = LayoutTensor[
                DT, Layout.row_major(N_ENVS, Self.ACT + 1),
                MutAnyOrigin,
            ](alp_scratch_ptr)
            var action_lt = LayoutTensor[
                DT, Layout.row_major(N_ENVS, Self.ACT),
                MutAnyOrigin,
            ](action_ptr)
            comptime total_c = N_ENVS * Self.ACT
            comptime n_blocks_c = (total_c + TPB - 1) // TPB
            comptime clamp_kernel = _redq_action_clamp_kernel[
                N_ENVS, Self.ACT,
            ]
            self.ctx.value().enqueue_function[clamp_kernel](
                alp_lt, action_lt, self.action_scale,
                grid_dim=n_blocks_c, block_dim=TPB,
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
        """Per-lane replay push WITHOUT touching the episode tracker
        (the driver manages per-env return accumulators via
        `add_complete_return`). Matches REDQ's `record_batch_cpu`."""
        var obs_lane = List[Scalar[DT]](
            length=Self.OBS,
            fill=Scalar[DT](0.0),
        )
        var act_lane = List[Scalar[DT]](
            length=Self.ACT,
            fill=Scalar[DT](0.0),
        )
        var nxt_lane = List[Scalar[DT]](
            length=Self.OBS,
            fill=Scalar[DT](0.0),
        )
        for env_idx in range(N_ENVS):
            for d in range(Self.OBS):
                obs_lane[d] = prev_obs_ptr[env_idx * Self.OBS + d]
                nxt_lane[d] = next_obs_ptr[env_idx * Self.OBS + d]
            for j in range(Self.ACT):
                act_lane[j] = action_ptr[env_idx * Self.ACT + j]
            self.sample_blk.add(
                obs_lane,
                act_lane,
                reward_ptr[env_idx],
                nxt_lane,
                done_ptr[env_idx],
                ctx=self.ctx,
            )

    # ──────────────────────────────────────────────────────────────────
    # One-file `nn2-ckpt v2` checkpoint.
    #
    # Section order:
    #   actor / critic{0..N-1} / state_branch / action_branch /
    #     predictor / actor_opt / critic{0..N-1}_opt / sb_opt /
    #     ab_opt / pred_opt / alpha_opt
    #
    # Target critics are NOT serialized — hard-copied from their just-
    # restored online twins inside `load_state` (matches REDQ's
    # convention). Replay buffer + episode tracker NOT serialized
    # (matches SAC + REDQ).
    # ──────────────────────────────────────────────────────────────────

    def save_state(mut self, path: String) raises:
        var body = String("")
        comptime if Self.train_target == "cpu":
            save_state_v2_body(self.actor, body, "actor")
            for i in range(Self.N):
                save_state_v2_body(
                    self.ensemble.pairs[i].online,
                    body, "critic" + String(i),
                )
            save_state_v2_body(self.state_branch, body, "state_branch")
            save_state_v2_body(self.action_branch, body, "action_branch")
            save_state_v2_body(self.predictor, body, "predictor")
            save_optimizer_v2_body(self.actor_opt, body, "actor_opt")
            for i in range(Self.N):
                save_optimizer_v2_body(
                    self.ensemble.opts[i],
                    body, "critic" + String(i) + "_opt",
                )
            save_optimizer_v2_body(self.sb_opt, body, "sb_opt")
            save_optimizer_v2_body(self.ab_opt, body, "ab_opt")
            save_optimizer_v2_body(self.pred_opt, body, "pred_opt")
        else:
            var c = self.ctx.value()
            save_state_v2_body_gpu(self.actor, body, "actor", c)
            for i in range(Self.N):
                save_state_v2_body_gpu(
                    self.ensemble.pairs[i].online,
                    body, "critic" + String(i), c,
                )
            save_state_v2_body_gpu(
                self.state_branch, body, "state_branch", c,
            )
            save_state_v2_body_gpu(
                self.action_branch, body, "action_branch", c,
            )
            save_state_v2_body_gpu(self.predictor, body, "predictor", c)
            save_optimizer_v2_body_gpu(self.actor_opt, body, "actor_opt")
            for i in range(Self.N):
                save_optimizer_v2_body_gpu(
                    self.ensemble.opts[i],
                    body, "critic" + String(i) + "_opt",
                )
            save_optimizer_v2_body_gpu(self.sb_opt, body, "sb_opt")
            save_optimizer_v2_body_gpu(self.ab_opt, body, "ab_opt")
            save_optimizer_v2_body_gpu(self.pred_opt, body, "pred_opt")
        # ScalarAdam: REDQ-OFE uses ScalarAdam.new (host-only), so
        # the CPU serializer applies regardless of train_target.
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
                    lines, idx, "critic" + String(i),
                )
            load_state_v2_body(
                self.state_branch, lines, idx, "state_branch",
            )
            load_state_v2_body(
                self.action_branch, lines, idx, "action_branch",
            )
            load_state_v2_body(self.predictor, lines, idx, "predictor")
            load_optimizer_v2_body(
                self.actor_opt, lines, idx, "actor_opt",
            )
            for i in range(Self.N):
                load_optimizer_v2_body(
                    self.ensemble.opts[i],
                    lines, idx, "critic" + String(i) + "_opt",
                )
            load_optimizer_v2_body(self.sb_opt, lines, idx, "sb_opt")
            load_optimizer_v2_body(self.ab_opt, lines, idx, "ab_opt")
            load_optimizer_v2_body(
                self.pred_opt, lines, idx, "pred_opt",
            )
        else:
            var c = self.ctx.value()
            load_state_v2_body_gpu(self.actor, lines, idx, "actor", c)
            for i in range(Self.N):
                load_state_v2_body_gpu(
                    self.ensemble.pairs[i].online,
                    lines, idx, "critic" + String(i), c,
                )
            load_state_v2_body_gpu(
                self.state_branch, lines, idx, "state_branch", c,
            )
            load_state_v2_body_gpu(
                self.action_branch, lines, idx, "action_branch", c,
            )
            load_state_v2_body_gpu(
                self.predictor, lines, idx, "predictor", c,
            )
            load_optimizer_v2_body_gpu(
                self.actor_opt, lines, idx, "actor_opt",
            )
            for i in range(Self.N):
                load_optimizer_v2_body_gpu(
                    self.ensemble.opts[i],
                    lines, idx, "critic" + String(i) + "_opt",
                )
            load_optimizer_v2_body_gpu(
                self.sb_opt, lines, idx, "sb_opt",
            )
            load_optimizer_v2_body_gpu(
                self.ab_opt, lines, idx, "ab_opt",
            )
            load_optimizer_v2_body_gpu(
                self.pred_opt, lines, idx, "pred_opt",
            )
        load_scalar_adam_v2_body(
            self.alpha_opt, lines, idx, "alpha_opt",
        )
        load_counter_v2_body(
            self._total_train_steps, lines, idx, "_total_train_steps"
        )
        # Re-sync every target net from its just-restored online twin.
        for i in range(Self.N):
            hard_copy_params[target=Self.train_target, M=Self.CRITIC](
                self.ensemble.pairs[i].online,
                self.ensemble.pairs[i].target_net,
                self.ctx,
            )

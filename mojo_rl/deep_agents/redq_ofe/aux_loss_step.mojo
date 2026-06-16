"""OFEAuxLossStep — auxiliary next-state prediction loss for REDQ-OFE.

Phase O.2.a (CPU). Owns the scratch buffers + a `Concat[PHI_S_DIM, ACT]`
glue module; takes refs (via `mut` args) to:

    state_branch, action_branch, predictor          # OFE networks
    sb_opt, ab_opt, pred_opt                        # their Adams
    state: TrainerState[OBS, ACT, BATCH]            # mb_s / mb_a / mb_sp

…and runs the full aux pipeline in one `step` call:

    forward chain
    -------------
    φ(s)    = state_branch(mb_s)                    [BATCH × PHI_S_DIM]
    sa_in   = concat(φ(s), mb_a)                    [BATCH × PHI_S_DIM+ACT]
    φ(s,a)  = action_branch(sa_in)                  [BATCH × PHI_SA_DIM]
    pred    = predictor(φ(s,a))                     [BATCH × OBS]

    loss & gradient
    ---------------
    loss        = (1/(BATCH·OBS)) · Σ (pred − mb_sp)^2
    grad_pred   = 2·(pred − mb_sp) / (BATCH·OBS)

    backward chain (mode="all" everywhere — aux IS the path that
    trains OFE params)
    -----------------------------------------------------------
    predictor.vjp     → grad_φ(s,a)
    action_branch.vjp → grad_sa_in
    concat.vjp        → grad_φ(s), grad_action_dummy (discarded)
    state_branch.vjp  → grad_obs_dummy (discarded)

    opt steps
    ---------
    pred_opt.step(predictor)
    ab_opt.step(action_branch)
    sb_opt.step(state_branch)

Returns the scalar MSE loss for diagnostics. The OFE param updates
happen as a side effect of the three Adam `step` calls.

Design notes
============

(a) Networks + Adams are NOT owned. They live on the trainer (since
    state_branch + action_branch are also consumed by actor / critic
    forwards on the RL side — sharing is non-negotiable). OFEAuxLossStep
    is the work unit that does the gradient bookkeeping for the aux path.

(b) `mode="all"` on every vjp call. The OFE aux loss is the ONLY
    backward pass that should accumulate OFE param grads — the RL
    forward path will run state_branch / action_branch with
    `mode="input_only"` (or via a `StopGradParams` wrapper, depending
    on the trainer wiring chosen in O.2.b).

(c) Indep minibatch vs RL minibatch — the legacy redq_ofe sampled a
    fresh minibatch for the aux step (separate from the critic
    minibatch). Phase O.2.a takes the trainer state as-is; the trainer
    (O.2.b) gets to decide whether to re-sample before calling
    `aux.step(...)` or share with the critic step. Either way the
    block is reusable.

(d) CPU only. The trainer integration (O.2.b) is where the GPU port
    lands — the kernels.mojo file already documents the math, and the
    nn building blocks (LayerNorm, SkipConcat, Linear, Concat) all
    have GPU paths from prior phases.
"""

from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.scratch import Scratch
from mojo_rl.nn.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn.core.target_storage import TargetStorage, assert_tag_for
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.primitives.concat import Concat

from ..training.trainer_block import TrainerState
from .kernels import (
    aux_mse_grad_cpu, aux_mse_loss_cpu, aux_mse_grad_gpu,
)


struct OFEAuxLossStep[
    SB: Module,        # OFE State Branch
    AB: Module,        # OFE Action Branch
    PRED: Module,      # OFE Predictor Head (Linear[PHI_SA_DIM, OBS])
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_

    # Derived dims (comptime).
    comptime PHI_S_DIM = Self.SB.OUT_DIM
    comptime SA_IN_DIM = Self.PHI_S_DIM + Self.ACT
    comptime PHI_SA_DIM = Self.AB.OUT_DIM

    var concat: Concat[Self.PHI_S_DIM, Self.ACT]

    # ── Forward-pass scratches ─────────────────────────────────────────
    var phi_s: Scratch["ofe_phi_s",     Self.BATCH * Self.PHI_S_DIM]
    var sa_in: Scratch["ofe_sa_in",     Self.BATCH * Self.SA_IN_DIM]
    var phi_sa: Scratch["ofe_phi_sa",   Self.BATCH * Self.PHI_SA_DIM]
    var pred: Scratch["ofe_pred",       Self.BATCH * Self.OBS]

    # ── Backward-pass scratches ────────────────────────────────────────
    var g_pred:    Scratch["ofe_g_pred",    Self.BATCH * Self.OBS]
    var g_phi_sa:  Scratch["ofe_g_phi_sa",  Self.BATCH * Self.PHI_SA_DIM]
    var g_sa_in:   Scratch["ofe_g_sa_in",   Self.BATCH * Self.SA_IN_DIM]
    var g_phi_s:   Scratch["ofe_g_phi_s",   Self.BATCH * Self.PHI_S_DIM]
    # Discarded: grads flowing into the action input of the concat
    # (action came from the replay buffer — no upstream to send the
    # gradient to) and into the obs input of the state branch.
    var g_act_dummy: Scratch["ofe_g_act_dummy", Self.BATCH * Self.ACT]
    var g_obs_dummy: Scratch["ofe_g_obs_dummy", Self.BATCH * Self.OBS]

    var ts: TargetStorage

    # ── Comptime validations ───────────────────────────────────────────

    def __init__(out self):
        comptime assert Self.SB.IN_DIMS[0] == Self.OBS, (
            "OFEAuxLossStep: state branch IN must match OBS"
        )
        comptime assert Self.AB.IN_DIMS[0] == Self.SA_IN_DIM, (
            "OFEAuxLossStep: action branch IN must equal PHI_S_DIM + ACT"
        )
        comptime assert Self.PRED.IN_DIMS[0] == Self.PHI_SA_DIM, (
            "OFEAuxLossStep: predictor IN must equal action branch OUT"
        )
        comptime assert Self.PRED.OUT_DIM == Self.OBS, (
            "OFEAuxLossStep: predictor OUT must equal OBS"
        )
        self.concat = Concat[Self.PHI_S_DIM, Self.ACT]()
        self.phi_s = Scratch["ofe_phi_s", Self.BATCH * Self.PHI_S_DIM]()
        self.sa_in = Scratch["ofe_sa_in", Self.BATCH * Self.SA_IN_DIM]()
        self.phi_sa = Scratch["ofe_phi_sa", Self.BATCH * Self.PHI_SA_DIM]()
        self.pred = Scratch["ofe_pred", Self.BATCH * Self.OBS]()
        self.g_pred = Scratch["ofe_g_pred", Self.BATCH * Self.OBS]()
        self.g_phi_sa = Scratch[
            "ofe_g_phi_sa", Self.BATCH * Self.PHI_SA_DIM,
        ]()
        self.g_sa_in = Scratch[
            "ofe_g_sa_in", Self.BATCH * Self.SA_IN_DIM,
        ]()
        self.g_phi_s = Scratch[
            "ofe_g_phi_s", Self.BATCH * Self.PHI_S_DIM,
        ]()
        self.g_act_dummy = Scratch[
            "ofe_g_act_dummy", Self.BATCH * Self.ACT,
        ]()
        self.g_obs_dummy = Scratch[
            "ofe_g_obs_dummy", Self.BATCH * Self.OBS,
        ]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "OFEAuxLossStep: target must be 'cpu' or 'gpu'"
        )
        comptime if target == "gpu":
            if not ctx:
                raise Error(
                    "OFEAuxLossStep.make[target='gpu']: ctx required"
                )
        var blk = Self()
        blk.concat = Concat[Self.PHI_S_DIM, Self.ACT].make[
            target, INIT=Zero,
        ](ctx=ctx)
        blk.ts = TargetStorage.make[target](ctx=ctx)
        init_scratch_auto[Self, target](blk, ctx)
        return blk^

    # ──────────────────────────────────────────────────────────────────
    # The aux step (CPU).
    # ──────────────────────────────────────────────────────────────────

    def step[
        target: StaticString = "cpu",
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut state_branch: Self.SB,
        mut action_branch: Self.AB,
        mut predictor: Self.PRED,
        mut sb_opt: Adam,
        mut ab_opt: Adam,
        mut pred_opt: Adam,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises -> Scalar[DT]:
        """Run one full aux training step. Returns the MSE loss (host
        scalar, diagnostics only). On GPU the loss is computed via
        D2H of pred + target — REDQ-OFE doesn't capture under CUDA
        graphs (host control flow), so the per-step D2H is fine."""
        comptime assert target == "cpu" or target == "gpu", (
            "OFEAuxLossStep.step: target must be 'cpu' or 'gpu'"
        )
        assert_tag_for["OFEAuxLossStep", target](self.ts.target_tag)

        # ── Local raw pointers + TileTensor views ──────────────────────
        var obs_p = state.mb_s.target_ptr[target]()
        var act_p = state.mb_a.target_ptr[target]()
        var nobs_p = state.mb_sp.target_ptr[target]()

        var phi_s_p = self.phi_s.target_ptr[target]()
        var sa_in_p = self.sa_in.target_ptr[target]()
        var phi_sa_p = self.phi_sa.target_ptr[target]()
        var pred_p = self.pred.target_ptr[target]()
        var g_pred_p = self.g_pred.target_ptr[target]()
        var g_phi_sa_p = self.g_phi_sa.target_ptr[target]()
        var g_sa_in_p = self.g_sa_in.target_ptr[target]()
        var g_phi_s_p = self.g_phi_s.target_ptr[target]()
        var g_act_dummy_p = self.g_act_dummy.target_ptr[target]()
        var g_obs_dummy_p = self.g_obs_dummy.target_ptr[target]()

        var obs_t = TileTensor(obs_p, row_major[Self.BATCH, Self.OBS]())
        # Variadic-Concat hetero-shape workaround
        # (feedback_mojo_variadic_hetero_shape_workaround.md): build both
        # *inputs* TileTensors with the SAME comptime Layout
        # (row_major[BATCH, PHI_S_DIM]) — leaf recovers the per-input
        # shape via typed_view[BATCH, DIMS[i]]. Same for *grad_inputs* in
        # the backward call (g_phi_s, g_act_dummy).
        var act_t = TileTensor(act_p, row_major[Self.BATCH, Self.PHI_S_DIM]())
        var phi_s_t = TileTensor(
            phi_s_p, row_major[Self.BATCH, Self.PHI_S_DIM](),
        )
        var sa_in_t = TileTensor(
            sa_in_p, row_major[Self.BATCH, Self.SA_IN_DIM](),
        )
        var phi_sa_t = TileTensor(
            phi_sa_p, row_major[Self.BATCH, Self.PHI_SA_DIM](),
        )
        var pred_t = TileTensor(pred_p, row_major[Self.BATCH, Self.OBS]())
        var g_pred_t = TileTensor(
            g_pred_p, row_major[Self.BATCH, Self.OBS](),
        )
        var g_phi_sa_t = TileTensor(
            g_phi_sa_p, row_major[Self.BATCH, Self.PHI_SA_DIM](),
        )
        var g_sa_in_t = TileTensor(
            g_sa_in_p, row_major[Self.BATCH, Self.SA_IN_DIM](),
        )
        var g_phi_s_t = TileTensor(
            g_phi_s_p, row_major[Self.BATCH, Self.PHI_S_DIM](),
        )
        # Hetero-shape workaround mirror for backward (see above).
        var g_act_dummy_t = TileTensor(
            g_act_dummy_p, row_major[Self.BATCH, Self.PHI_S_DIM](),
        )
        var g_obs_dummy_t = TileTensor(
            g_obs_dummy_p, row_major[Self.BATCH, Self.OBS](),
        )

        # ── Forward chain ──────────────────────────────────────────────
        state_branch.forward[target, Self.BATCH, POLICY=POLICY](
            obs_t, output=phi_s_t,
        )
        self.concat.forward[target, Self.BATCH, POLICY=POLICY](
            TensorPack[2].of(phi_s_t, act_t), output=sa_in_t,
        )
        action_branch.forward[target, Self.BATCH, POLICY=POLICY](
            sa_in_t, output=phi_sa_t,
        )
        predictor.forward[target, Self.BATCH, POLICY=POLICY](
            phi_sa_t, output=pred_t,
        )

        # ── Loss + MSE gradient ────────────────────────────────────────
        var loss: Scalar[DT]
        comptime if target == "cpu":
            loss = aux_mse_loss_cpu[Self.BATCH, Self.OBS](
                pred_p, nobs_p,
            )
            aux_mse_grad_cpu[Self.BATCH, Self.OBS](
                pred_p, nobs_p, g_pred_p,
            )
        else:
            # Device grad kernel + D2H of pred + target for the
            # host-side loss reduction (cheap; REDQ-OFE doesn't
            # capture under CUDA graphs).
            var ctx = self.ts.ctx.value()
            aux_mse_grad_gpu[Self.BATCH, Self.OBS](
                ctx, pred_p, nobs_p, g_pred_p,
            )
            var pred_host = ctx.enqueue_create_host_buffer[DT](
                Self.BATCH * Self.OBS
            )
            var nobs_host = ctx.enqueue_create_host_buffer[DT](
                Self.BATCH * Self.OBS
            )
            ctx.enqueue_copy(
                pred_host, self.pred.dev.value(),
            )
            ctx.enqueue_copy(
                nobs_host, state.mb_sp.dev.value(),
            )
            ctx.synchronize()
            var pred_h_p = pred_host.unsafe_ptr()
            var nobs_h_p = nobs_host.unsafe_ptr()
            loss = aux_mse_loss_cpu[Self.BATCH, Self.OBS](
                pred_h_p, nobs_h_p,
            )

        # ── Zero grads on all three OFE networks ──────────────────────
        sb_opt.zero_grad[target, M=Self.SB](state_branch)
        ab_opt.zero_grad[target, M=Self.AB](action_branch)
        pred_opt.zero_grad[target, M=Self.PRED](predictor)

        # ── Backward chain (mode='all' — aux IS the path that trains
        # the OFE params) ──────────────────────────────────────────────
        predictor.vjp[target, Self.BATCH, POLICY=POLICY](
            g_pred_t, g_phi_sa_t,
        )
        action_branch.vjp[target, Self.BATCH, POLICY=POLICY](
            g_phi_sa_t, g_sa_in_t,
        )
        # Concat splits grad_sa_in into (grad_phi_s, grad_action).
        # grad_action is discarded — the buffer-sampled action has no
        # upstream that the OFE training can propagate to.
        self.concat.vjp[target, Self.BATCH, POLICY=POLICY](g_sa_in_t, TensorPack[2].of(g_phi_s_t, g_act_dummy_t))
        state_branch.vjp[target, Self.BATCH, POLICY=POLICY](
            g_phi_s_t, g_obs_dummy_t,
        )

        # ── Adam steps ─────────────────────────────────────────────────
        pred_opt.step[target, M=Self.PRED](predictor)
        ab_opt.step[target, M=Self.AB](action_branch)
        sb_opt.step[target, M=Self.SB](state_branch)

        return loss

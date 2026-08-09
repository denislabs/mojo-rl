"""OFEAuxLossStep — auxiliary next-state prediction loss for REDQ-OFE (STORAGE).

Owns the scratch buffers + a `Concat[PHI_S_DIM, ACT]` glue module; takes
refs (via `mut` args) to:

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

    backward chain (aux IS the path that trains OFE params)
    -------------------------------------------------------
    predictor.vjp     → grad_φ(s,a)
    action_branch.vjp → grad_sa_in
    concat.vjp        → grad_φ(s), grad_action_dummy (discarded)
    state_branch.vjp  → grad_obs_dummy (discarded)

    opt steps
    ---------
    pred_opt.step(predictor)
    ab_opt.step(action_branch)
    sb_opt.step(state_branch)

Returns the scalar MSE loss for diagnostics.

Design notes
============
(a) Networks + Adams are NOT owned. They live on the trainer (since
    state_branch + action_branch are also consumed by actor / critic
    forwards on the RL side — sharing is non-negotiable).
(b) The storage `Module.vjp` has NO `mode` param — it always computes
    BOTH param + input grads. The aux loss is the ONLY backward pass
    that should accumulate OFE param grads; the RL forwards SHARE the
    same OFE params but never call `.vjp` on them (forward-only), so no
    RL gradient ever reaches the OFE params. The aux step zero_grads the
    three OFE nets before its own forward/vjp, so the discarded RL-path
    forward caches are irrelevant (aux runs its own forward+vjp atomically).
(c) Uses the variadic storage `Concat[PHI_S_DIM, ACT]` (the N=2 instance)
    as the (φ(s), a) glue — a real consumer of the variadic primitive.

STORAGE migration (Stage 5): legacy `Scratch`/`TargetStorage`/`mptr`/
TileTensor gone — scratch are owned `nn.storage.Tensor`s; all forwards/vjps
use the storage Module surface over `TensorRefs`; the loss/grad use the
storage `kernels.mojo` functions. CPU + GPU.
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.initializer import Zero
from mojo_rl.nn.core.call import call_forward, call_vjp
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.primitives.concat import Concat

from ..training.trainer_block import TrainerState
from .kernels import aux_mse_grad_cpu, aux_mse_loss_cpu, aux_mse_grad_gpu


struct OFEAuxLossStep[
    SB: Module,        # OFE State Branch  (IN=OBS, OUT=PHI_S_DIM)
    AB: Module,        # OFE Action Branch (IN=PHI_S_DIM+ACT, OUT=PHI_SA_DIM)
    PRED: Module,      # OFE Predictor Head (Linear[PHI_SA_DIM, OBS])
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
](Defaultable & Movable & Deinitable):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_

    # Derived dims (comptime).
    comptime PHI_S_DIM = Self.SB.OUT_DIM
    comptime SA_IN_DIM = Self.PHI_S_DIM + Self.ACT
    comptime PHI_SA_DIM = Self.AB.OUT_DIM

    var concat: Concat[Self.PHI_S_DIM, Self.ACT]

    # ── Concat input / grad packs (ONE owner → ONE origin, satisfies the
    # §B0 TensorRefs constraint that all variadic-pack inputs share an
    # origin). `cat_in[0]` = φ(s) (state-branch output), `cat_in[1]` = a
    # copy of mb_a; `cat_gin[0]` = grad_φ(s), `cat_gin[1]` = grad_action
    # (discarded). ─────────────────────────────────────────────────────
    var cat_in: TensorPack[2]
    var cat_gin: TensorPack[2]

    # ── Forward-pass scratches ─────────────────────────────────────────
    var sa_in: Tensor    # [BATCH, SA_IN_DIM]
    var phi_sa: Tensor   # [BATCH, PHI_SA_DIM]
    var pred: Tensor     # [BATCH, OBS]

    # ── Backward-pass scratches ────────────────────────────────────────
    var g_pred: Tensor      # [BATCH, OBS]
    var g_phi_sa: Tensor    # [BATCH, PHI_SA_DIM]
    var g_sa_in: Tensor     # [BATCH, SA_IN_DIM]
    # Discarded: grad into the state-branch obs input.
    var g_obs_dummy: Tensor  # [BATCH, OBS]

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
        self.cat_in = TensorPack[2]()
        self.cat_gin = TensorPack[2]()
        self.sa_in = Tensor()
        self.phi_sa = Tensor()
        self.pred = Tensor()
        self.g_pred = Tensor()
        self.g_phi_sa = Tensor()
        self.g_sa_in = Tensor()
        self.g_obs_dummy = Tensor()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "OFEAuxLossStep: target must be 'cpu' or 'gpu'"
        )
        comptime if target == "gpu":
            if not ctx:
                raise Error("OFEAuxLossStep.make[target='gpu']: ctx required")
        var blk = Self()
        blk.concat = Concat[Self.PHI_S_DIM, Self.ACT].make[target, INIT=Zero](
            ctx=ctx
        )
        comptime if target == "cpu":
            blk.cat_in[0].ensure(Self.BATCH * Self.PHI_S_DIM)
            blk.cat_in[1].ensure(Self.BATCH * Self.ACT)
            blk.cat_gin[0].ensure(Self.BATCH * Self.PHI_S_DIM)
            blk.cat_gin[1].ensure(Self.BATCH * Self.ACT)
            blk.sa_in = Tensor.alloc(Self.BATCH * Self.SA_IN_DIM)
            blk.phi_sa = Tensor.alloc(Self.BATCH * Self.PHI_SA_DIM)
            blk.pred = Tensor.alloc(Self.BATCH * Self.OBS)
            blk.g_pred = Tensor.alloc(Self.BATCH * Self.OBS)
            blk.g_phi_sa = Tensor.alloc(Self.BATCH * Self.PHI_SA_DIM)
            blk.g_sa_in = Tensor.alloc(Self.BATCH * Self.SA_IN_DIM)
            blk.g_obs_dummy = Tensor.alloc(Self.BATCH * Self.OBS)
        else:
            var c = ctx.value()
            blk.cat_in[0].ensure_gpu(c, Self.BATCH * Self.PHI_S_DIM)
            blk.cat_in[1].ensure_gpu(c, Self.BATCH * Self.ACT)
            blk.cat_gin[0].ensure_gpu(c, Self.BATCH * Self.PHI_S_DIM)
            blk.cat_gin[1].ensure_gpu(c, Self.BATCH * Self.ACT)
            blk.sa_in = Tensor.alloc_gpu(c, Self.BATCH * Self.SA_IN_DIM)
            blk.phi_sa = Tensor.alloc_gpu(c, Self.BATCH * Self.PHI_SA_DIM)
            blk.pred = Tensor.alloc_gpu(c, Self.BATCH * Self.OBS)
            blk.g_pred = Tensor.alloc_gpu(c, Self.BATCH * Self.OBS)
            blk.g_phi_sa = Tensor.alloc_gpu(c, Self.BATCH * Self.PHI_SA_DIM)
            blk.g_sa_in = Tensor.alloc_gpu(c, Self.BATCH * Self.SA_IN_DIM)
            blk.g_obs_dummy = Tensor.alloc_gpu(c, Self.BATCH * Self.OBS)
        return blk^

    # ──────────────────────────────────────────────────────────────────
    # The aux step.
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
        scalar, diagnostics only)."""
        comptime assert target == "cpu" or target == "gpu", (
            "OFEAuxLossStep.step: target must be 'cpu' or 'gpu'"
        )
        var ctx = state.ctx

        # ── Zero grads on all three OFE networks BEFORE forward/vjp. ───
        sb_opt.zero_grad[target, M=Self.SB](state_branch, ctx)
        ab_opt.zero_grad[target, M=Self.AB](action_branch, ctx)
        pred_opt.zero_grad[target, M=Self.PRED](predictor, ctx)

        # ── Stage mb_a into the concat-input pool (so both Concat inputs
        # share ONE origin — the §B0 constraint). ──────────────────────
        comptime AT = Self.BATCH * Self.ACT
        comptime if target == "cpu":
            for k in range(AT):
                self.cat_in[1].data[k] = state.mb_a.data[k]
        else:
            var c = ctx.value()
            c.enqueue_copy(self.cat_in[1].dev.value(), state.mb_a.dev.value())

        # ── Forward chain ──────────────────────────────────────────────
        # φ(s) → cat_in[0]; concat(φ(s), mb_a copy) → sa_in.
        call_forward[target, Self.BATCH, POLICY=POLICY](
            state_branch, TensorRefs[Self.SB.ARITY](state.mb_s), self.cat_in[0], ctx
        )
        call_forward[target, Self.BATCH, POLICY=POLICY](
            self.concat,
            TensorRefs[2](self.cat_in[0], self.cat_in[1]), self.sa_in, ctx
        )
        call_forward[target, Self.BATCH, POLICY=POLICY](
            action_branch, TensorRefs[Self.AB.ARITY](self.sa_in), self.phi_sa, ctx
        )
        call_forward[target, Self.BATCH, POLICY=POLICY](
            predictor, TensorRefs[Self.PRED.ARITY](self.phi_sa), self.pred, ctx
        )

        # ── Loss + MSE gradient ────────────────────────────────────────
        var loss: Scalar[DT]
        comptime if target == "cpu":
            loss = aux_mse_loss_cpu[Self.BATCH, Self.OBS](self.pred, state.mb_sp)
            aux_mse_grad_cpu[Self.BATCH, Self.OBS](
                self.pred, state.mb_sp, self.g_pred
            )
        else:
            var c = ctx.value()
            aux_mse_grad_gpu[Self.BATCH, Self.OBS](
                c, self.pred, state.mb_sp, self.g_pred
            )
            # D2H pred + target for the host-side diagnostic loss reduction.
            self.pred.download(c)
            state.mb_sp.download(c)
            loss = aux_mse_loss_cpu[Self.BATCH, Self.OBS](
                self.pred, state.mb_sp
            )

        # ── Backward chain ─────────────────────────────────────────────
        call_vjp[target, Self.BATCH, POLICY=POLICY](
            predictor,
            TensorRefs[Self.PRED.ARITY](self.phi_sa),
            self.g_pred,
            TensorRefs[Self.PRED.ARITY](self.g_phi_sa),
            ctx,
        )
        call_vjp[target, Self.BATCH, POLICY=POLICY](
            action_branch,
            TensorRefs[Self.AB.ARITY](self.sa_in),
            self.g_phi_sa,
            TensorRefs[Self.AB.ARITY](self.g_sa_in),
            ctx,
        )
        # Concat splits grad_sa_in into (grad_phi_s, grad_action). grad_action
        # (cat_gin[1]) is discarded — the buffer-sampled action has no upstream.
        call_vjp[target, Self.BATCH, POLICY=POLICY](
            self.concat,
            TensorRefs[2](self.cat_in[0], self.cat_in[1]),
            self.g_sa_in,
            TensorRefs[2](self.cat_gin[0], self.cat_gin[1]),
            ctx,
        )
        call_vjp[target, Self.BATCH, POLICY=POLICY](
            state_branch,
            TensorRefs[Self.SB.ARITY](state.mb_s),
            self.cat_gin[0],
            TensorRefs[Self.SB.ARITY](self.g_obs_dummy),
            ctx,
        )

        # ── Adam steps ─────────────────────────────────────────────────
        pred_opt.step[target, M=Self.PRED](predictor, ctx)
        ab_opt.step[target, M=Self.AB](action_branch, ctx)
        sb_opt.step[target, M=Self.SB](state_branch, ctx)

        return loss

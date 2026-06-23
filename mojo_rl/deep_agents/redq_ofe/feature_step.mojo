"""OFEFeatureStep — pre-pass that populates φ(s) and φ(s') scratches (STORAGE).

Runs the OFE state-branch on the sampled minibatch obs (`state.mb_s`) and
next-obs (`state.mb_sp`) to fill two owned `nn.storage.Tensor`s:

    phi_s  : [BATCH, PHI_S_DIM]   ← state_branch.forward(mb_s)
    phi_sp : [BATCH, PHI_S_DIM]   ← state_branch.forward(mb_sp)

Downstream OFE-aware RL blocks (`EnsembleTargetYBlockOFE`,
`EnsembleCriticStepOFE`, `EnsembleActorStepOFE`) read these owned Tensors
(passed by `mut` ref by the trainer). `phi_s(a)` and `phi_s'(a')` are *not*
precomputed here — they depend on the sampled action (from the buffer or
from the actor), so the consuming block runs the action-branch on the fly.

Gradient policy
===============
Forward only. The OFE state-branch parameters are trained EXCLUSIVELY via
`OFEAuxLossStep`. On the RL path no gradient ever flows back to state-branch
params — the RL blocks NEVER call `state_branch.vjp` on these scratches.

This means the state-branch *cache* populated by this forward pass gets
clobbered later in the train_step (e.g. by the aux step's own forward, which
runs an independent forward+vjp pair atomically). The RL path never reads the
cache because it never vjps the state-branch, so the clobber is harmless.

STORAGE migration (Stage 5): legacy `Scratch`/`TargetStorage`/`mptr`/
TileTensor gone — scratch are owned `nn.storage.Tensor`s (alloc on target);
`state_branch.forward` uses the storage Module surface (`forward[target, B](
TensorRefs, mut out, ctx)`).
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs

from ..training.trainer_block import TrainerState


struct OFEFeatureStep[
    SB: Module,
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
](Defaultable & Movable & ImplicitlyDeletable):
    """Pre-pass populating φ(s) and φ(s') scratches.

    Comptime params:
      SB    : OFE state-branch module (IN=OBS, OUT=PHI_S_DIM).
      OBS_  : raw observation dim (matches replay buffer / mb_s width).
      ACT_  : action dim (carried so TrainerState typing matches the
              trainer's).
      BATCH_: minibatch size.
    """

    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime PHI_S_DIM = Self.SB.OUT_DIM

    var phi_s: Tensor    # [BATCH, PHI_S_DIM]
    var phi_sp: Tensor   # [BATCH, PHI_S_DIM]

    def __init__(out self):
        comptime assert Self.SB.IN_DIMS[0] == Self.OBS, (
            "OFEFeatureStep: state branch IN must equal OBS"
        )
        self.phi_s = Tensor()
        self.phi_sp = Tensor()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "OFEFeatureStep: target must be 'cpu' or 'gpu'"
        )
        var blk = Self()
        comptime if target == "cpu":
            blk.phi_s = Tensor.alloc(Self.BATCH * Self.PHI_S_DIM)
            blk.phi_sp = Tensor.alloc(Self.BATCH * Self.PHI_S_DIM)
        else:
            var c = ctx.value()
            blk.phi_s = Tensor.alloc_gpu(c, Self.BATCH * Self.PHI_S_DIM)
            blk.phi_sp = Tensor.alloc_gpu(c, Self.BATCH * Self.PHI_S_DIM)
        return blk^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut state_branch: Self.SB,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        """Populate `phi_s` and `phi_sp` from `state.mb_s` / `state.mb_sp`."""
        var ctx = state.ctx
        state_branch.forward[target, Self.BATCH, POLICY=POLICY](
            TensorRefs[Self.SB.ARITY](state.mb_s), self.phi_s, ctx
        )
        state_branch.forward[target, Self.BATCH, POLICY=POLICY](
            TensorRefs[Self.SB.ARITY](state.mb_sp), self.phi_sp, ctx
        )

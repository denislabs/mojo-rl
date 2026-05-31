"""OFEFeatureStep — pre-pass that populates φ(s) and φ(s') scratches.

Phase O.2.b.1 (CPU). Runs the OFE state-branch on the sampled
minibatch obs (`state.mb_s`) and next-obs (`state.mb_sp`) to fill two
owned scratches:

    phi_s  : [BATCH, PHI_S_DIM]   ← state_branch.forward(mb_s)
    phi_sp : [BATCH, PHI_S_DIM]   ← state_branch.forward(mb_sp)

Downstream OFE-aware RL blocks (`EnsembleTargetYBlockOFE`,
`EnsembleCriticStepOFE`, `EnsembleActorStepOFE`) read these via the
`phi_s_ptr()` / `phi_sp_ptr()` accessors. `phi_s(a)` and `phi_s'(a')`
are *not* precomputed here — they depend on the sampled action (from
the buffer or from the actor), so the consuming block runs the
action-branch on the fly.

Gradient policy
===============
Forward only (Module.forward has no `mode`; the gradient path is what
gets gated). The OFE state-branch parameters are trained EXCLUSIVELY
via `OFEAuxLossStep`. On the RL path no gradient ever flows back to
state-branch params — the RL blocks NEVER call `state_branch.vjp` on
these scratches.

This means the state-branch *cache* populated by this forward pass
gets clobbered later in the train_step (e.g. by the aux step's own
forward, which runs an independent forward+vjp pair atomically). The
RL path never reads the cache because it never vjps the state-branch,
so the clobber is harmless. The aux step's forward+vjp must remain
atomic — that's the contract `OFEAuxLossStep` already satisfies.
"""

from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.core.scratch import Scratch
from mojo_rl.nn2.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn2.core.target_storage import TargetStorage, assert_tag_for

from ..training.trainer_block import TrainerState


struct OFEFeatureStep[
    SB: Module,
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
](Defaultable & Movable & ImplicitlyDestructible):
    """Pre-pass populating φ(s) and φ(s') scratches.

    Comptime params:
      SB    : OFE state-branch module (IN=OBS, OUT=PHI_S_DIM).
      OBS_  : raw observation dim (matches replay buffer / mb_s width).
      ACT_  : action dim (carried so TrainerState typing matches the
              trainer's). Unused inside this step but needed for the
              shared `TrainerState[OBS, ACT, BATCH]` parameterization.
      BATCH_: minibatch size.
    """

    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime PHI_S_DIM = Self.SB.OUT_DIM

    var phi_s:  Scratch["ofe_feat_phi_s",  Self.BATCH * Self.PHI_S_DIM]
    var phi_sp: Scratch["ofe_feat_phi_sp", Self.BATCH * Self.PHI_S_DIM]

    var ts: TargetStorage

    def __init__(out self):
        comptime assert Self.SB.IN_DIMS[0] == Self.OBS, (
            "OFEFeatureStep: state branch IN must equal OBS"
        )
        self.phi_s = Scratch[
            "ofe_feat_phi_s", Self.BATCH * Self.PHI_S_DIM,
        ]()
        self.phi_sp = Scratch[
            "ofe_feat_phi_sp", Self.BATCH * Self.PHI_S_DIM,
        ]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "OFEFeatureStep: target must be 'cpu' or 'gpu'"
        )
        var blk = Self()
        blk.ts = TargetStorage.make[target](ctx=ctx)
        init_scratch_auto[Self, target](blk, ctx)
        return blk^

    # ── Accessors used by downstream OFE-aware blocks ─────────────────

    def phi_s_ptr[target: StaticString](
        self,
    ) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self.phi_s.target_ptr[target]()

    def phi_sp_ptr[target: StaticString](
        self,
    ) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self.phi_sp.target_ptr[target]()

    # ── The feature pre-pass ──────────────────────────────────────────

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut state_branch: Self.SB,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
    ) raises:
        """Populate `phi_s` and `phi_sp` from `state.mb_s` / `state.mb_sp`.

        The TrainerState comes from the trainer that owns this block.
        We only touch `mb_s` and `mb_sp` (both width OBS) — the action
        slot is consumed by other OFE blocks (critic step, actor
        step), not here."""
        assert_tag_for["OFEFeatureStep", target](self.ts.target_tag)

        var obs_p = state.mb_s.target_ptr[target]()
        var nobs_p = state.mb_sp.target_ptr[target]()
        var phi_s_p = self.phi_s.target_ptr[target]()
        var phi_sp_p = self.phi_sp.target_ptr[target]()

        var obs_t = TileTensor(obs_p, row_major[Self.BATCH, Self.OBS]())
        var nobs_t = TileTensor(nobs_p, row_major[Self.BATCH, Self.OBS]())
        var phi_s_t = TileTensor(
            phi_s_p, row_major[Self.BATCH, Self.PHI_S_DIM](),
        )
        var phi_sp_t = TileTensor(
            phi_sp_p, row_major[Self.BATCH, Self.PHI_S_DIM](),
        )

        state_branch.forward[target, Self.BATCH, POLICY=POLICY](
            obs_t, output=phi_s_t,
        )
        state_branch.forward[target, Self.BATCH, POLICY=POLICY](
            nobs_t, output=phi_sp_t,
        )

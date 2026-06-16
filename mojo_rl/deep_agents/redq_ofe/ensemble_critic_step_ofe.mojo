"""EnsembleCriticStepOFE — N-critic gradient step against shared target y.

Phase O.2.b.2 (CPU). Mirrors `EnsembleCriticStep` from `redq/`: loops
over N online critics in `CriticEnsemble[CRITIC, N]` and runs one
`CriticUpdateBlock.step` per critic. The OFE delta is the input:

  - Non-OFE REDQ: critic takes `concat(mb_s, mb_a)` of dim OBS+ACT
  - OFE REDQ:     critic takes `φ(s, a) = action_branch(concat(φ(s),
                  mb_a))` of dim PHI_SA_DIM

So this block runs the action-branch forward ONCE on
`concat(phi_s, mb_a)` to produce `φ(s, a)` in `_mb_phi_sa`, then
loops the N critic updates against that same `phi_sa` (every critic
sees the same φ(s,a) — matches the legacy REDQ-OFE data flow).

Like `EnsembleCriticStep` we hold ONE `CriticUpdateBlock` and reuse
it across all N members; its scratches are overwritten between
iterations but never read across them.

Gradient policy
===============
`CriticUpdateBlock.step` writes `grad_φ(s, a)` into its
`_mb_grad_sa` scratch and stops there — it doesn't propagate back
into `action_branch` or `state_branch`. That's correct: OFE params
only train via the aux loss path (`OFEAuxLossStep`). The action-
branch forward here populates its cache, but no `action_branch.vjp`
ever follows it on the RL path.

R.2 is CPU-only for this slice; GPU lands alongside the full
REDQOFETrainer GPU port. The `concat(phi_s, mb_a)` step is a plain
CPU loop (same pattern as `EnsembleCriticStep`'s inline `concat_sa`
helper); the GPU port will add a `concat_phi_sa_gpu` kernel.
"""

from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.scratch import Scratch
from mojo_rl.nn.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn.core.target_storage import TargetStorage, assert_tag_for

from ..loss.critic_update_block import CriticUpdateBlock
from ..redq.ensemble import CriticEnsemble
from ..training.off_policy_critic import concat_sa, concat_sa_gpu


struct EnsembleCriticStepOFE[
    AB: Module,            # IN=PHI_S_DIM+ACT, OUT=PHI_SA_DIM
    CRITIC: Module,        # IN=PHI_SA_DIM, OUT=1
    N: Int,
    BATCH_: Int,
    PHI_S_DIM_: Int,
    ACT_: Int,
](Defaultable & Movable & ImplicitlyDeletable):
    comptime BATCH = Self.BATCH_
    comptime PHI_S_DIM = Self.PHI_S_DIM_
    comptime ACT = Self.ACT_
    comptime SA_IN_DIM = Self.PHI_S_DIM + Self.ACT
    comptime PHI_SA_DIM = Self.AB.OUT_DIM

    var member_step: CriticUpdateBlock[
        Self.CRITIC, Self.BATCH, Self.PHI_SA_DIM,
    ]
    var _mb_sa_in: Scratch[
        "ocstep_mb_sa_in", Self.BATCH * Self.SA_IN_DIM,
    ]
    var _mb_phi_sa: Scratch[
        "ocstep_mb_phi_sa", Self.BATCH * Self.PHI_SA_DIM,
    ]
    var ts: TargetStorage

    def __init__(out self):
        self.member_step = CriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.PHI_SA_DIM,
        ]()
        self._mb_sa_in = Scratch[
            "ocstep_mb_sa_in", Self.BATCH * Self.SA_IN_DIM,
        ]()
        self._mb_phi_sa = Scratch[
            "ocstep_mb_phi_sa", Self.BATCH * Self.PHI_SA_DIM,
        ]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "EnsembleCriticStepOFE: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.AB.IN_DIMS[0] == Self.SA_IN_DIM, (
            "EnsembleCriticStepOFE: AB.IN must equal PHI_S_DIM + ACT"
        )
        comptime assert Self.CRITIC.IN_DIMS[0] == Self.PHI_SA_DIM, (
            "EnsembleCriticStepOFE: CRITIC.IN must equal AB.OUT"
        )
        var blk = Self()
        blk.member_step = CriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.PHI_SA_DIM,
        ].make[target](ctx=ctx)
        blk.ts = TargetStorage.make[target](ctx=ctx)
        init_scratch_auto[Self, target](blk, ctx)
        return blk^

    def step[
        target: StaticString = "cpu",
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut action_branch: Self.AB,
        mut ensemble: CriticEnsemble[Self.CRITIC, Self.N],
        mb_phi_s_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_a_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_y_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises -> Scalar[DT]:
        """One ensemble-wide critic gradient step. Reads φ(s) (from the
        feature step), mb_a, and the shared target mb_y; returns the
        sum of per-critic losses."""
        comptime assert target == "cpu" or target == "gpu", (
            "EnsembleCriticStepOFE.step: target must be 'cpu' or 'gpu'"
        )
        assert_tag_for["EnsembleCriticStepOFE", target](self.ts.target_tag)

        # 1. concat(φ(s), mb_a) → _mb_sa_in [BATCH, PHI_S_DIM + ACT].
        # `concat_sa(_gpu)` takes the first-input width as `OBS` —
        # passing PHI_S_DIM in that slot is correct (the helper
        # is width-agnostic).
        var sa_in_p = self._mb_sa_in.target_ptr[target]()
        comptime if target == "cpu":
            concat_sa[Self.PHI_S_DIM, Self.ACT, Self.BATCH](
                mb_phi_s_ptr, mb_a_ptr, sa_in_p,
            )
        else:
            concat_sa_gpu[Self.PHI_S_DIM, Self.ACT, Self.BATCH](
                self.ts.ctx.value(), mb_phi_s_ptr, mb_a_ptr, sa_in_p,
            )

        # 2. action_branch.forward(sa_in) → φ(s, a) [BATCH, PHI_SA_DIM].
        var sa_in_t = TileTensor(
            sa_in_p, row_major[Self.BATCH, Self.SA_IN_DIM](),
        )
        var phi_sa_p = self._mb_phi_sa.target_ptr[target]()
        var phi_sa_t = TileTensor(
            phi_sa_p, row_major[Self.BATCH, Self.PHI_SA_DIM](),
        )
        action_branch.forward[target, Self.BATCH, POLICY](
            sa_in_t, output=phi_sa_t,
        )

        # 3. Loop N online critics, each gets the same φ(s, a) → mb_y.
        var mb_y_t = TileTensor(mb_y_ptr, row_major[Self.BATCH, 1]())
        var loss_sum: Scalar[DT] = Scalar[DT](0.0)
        for i in range(Self.N):
            var loss = self.member_step.step[target, POLICY](
                ensemble.pairs[i].online,
                ensemble.opts[i],
                phi_sa_t,
                mb_y_t,
            )
            loss_sum += loss
        return loss_sum

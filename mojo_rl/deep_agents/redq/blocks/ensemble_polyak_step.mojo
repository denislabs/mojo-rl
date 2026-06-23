"""EnsemblePolyakStep — soft-update all N target nets in one step.

Phase R.3. SAC's `PolyakStep` polyaks 2 pairs; REDQ's ensemble has
N. The block is a thin loop over `CriticEnsemble.soft_update_all`
which itself iterates the N owned `OnlineTargetPair`s — so this is
mostly the TrainerState-aware façade matching SAC's
`PolyakStep.step` signature shape (parameterless-aside-from-tau,
reads `state.ctx` so the trainer doesn't pass it explicitly).

REDQ is paper-faithful "polyak every inner critic step" (not every
N inner steps) — the trainer calls `step` from inside its UTD loop
every iteration.
"""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module

from ..ensemble import CriticEnsemble
from ...training.trainer_block import TrainerState


struct EnsemblePolyakStep[
    CRITIC: Module,
    N_: Int,
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
](Defaultable & Movable & ImplicitlyDeletable):
    comptime N = Self.N_
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_

    var tau: Scalar[DT]

    def __init__(out self):
        self.tau = Scalar[DT](0.005)

    @staticmethod
    def make(tau: Scalar[DT] = Scalar[DT](0.005)) -> Self:
        var b = Self()
        b.tau = tau
        return b^

    def step[
        target: StaticString,
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut ensemble: CriticEnsemble[Self.CRITIC, Self.N],
    ) raises:
        """τ-polyak every target_net toward its online twin. Pulls
        the trainer's `DeviceContext` from `state.ctx` (None on CPU,
        Some(ctx) on GPU) — matches SAC's PolyakStep convention."""
        ensemble.soft_update_all[target](self.tau, state.ctx)

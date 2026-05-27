"""PolyakStep — twin-critic target polyak update.

Holds τ as a small hyperparam. Reads no state; runs the polyak step on
both pair1 and pair2 (online → target soft copy).
"""

from ...constants import DT
from ...core.module import Module
from ...core.online_target_pair import OnlineTargetPair
from ..trainer_block import TrainerState


struct PolyakStep[
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
    CRITIC: Module,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime Pair = OnlineTargetPair[Self.CRITIC]

    var tau: Scalar[DT]

    def __init__(out self):
        self.tau = Scalar[DT](0.005)

    @staticmethod
    def make(tau: Scalar[DT]) -> Self:
        var b = Self()
        b.tau = tau
        return b^

    def step[
        target: StaticString
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut pair1: Self.Pair,
        mut pair2: Self.Pair,
    ) raises:
        pair1.polyak_step[target](self.tau, state.ctx)
        pair2.polyak_step[target](self.tau, state.ctx)

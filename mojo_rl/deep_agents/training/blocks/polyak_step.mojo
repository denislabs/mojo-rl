"""PolyakStep / SinglePolyakStep — target-net update blocks.

PolyakStep: twin-critic soft-update (SAC / TD3). Holds τ; soft-updates
both pair1 and pair2 on every train step.

SinglePolyakStep: single-pair update (DQN family). Holds τ + an
`update_every: Int` mode switch:
  - `update_every == 0` → soft τ-update every train step (Polyak).
  - `update_every > 0`  → hard copy (online → target) every N steps,
    skip otherwise. Implemented as `polyak_step(tau=1.0)` since
    `target = 1·online + 0·target = online` is bit-exact a hard copy.

Reading step_idx via `state.step_idx` keeps the block's `step` surface
parameterless (matches the rest of the SAC pipeline blocks).
"""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from ...core.online_target_pair import OnlineTargetPair
from ..trainer_block import TrainerState


struct PolyakStep[
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
    CRITIC: Module,
](Defaultable & Movable & ImplicitlyDeletable):
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


struct SinglePolyakStep[
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
    CRITIC: Module,
](Defaultable & Movable & ImplicitlyDeletable):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime Pair = OnlineTargetPair[Self.CRITIC]

    var tau: Scalar[DT]
    var update_every: Int

    def __init__(out self):
        self.tau = Scalar[DT](0.005)
        self.update_every = 0

    @staticmethod
    def make(
        tau: Scalar[DT] = Scalar[DT](0.005),
        update_every: Int = 0,
    ) -> Self:
        var b = Self()
        b.tau = tau
        b.update_every = update_every
        return b^

    def step[
        target: StaticString
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut pair: Self.Pair,
    ) raises:
        if self.update_every > 0:
            if state.step_idx % self.update_every == 0:
                pair.polyak_step[target](Scalar[DT](1.0), state.ctx)
        else:
            pair.polyak_step[target](self.tau, state.ctx)

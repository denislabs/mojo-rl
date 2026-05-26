"""DDPGActorStep — DDPG actor (DPG) gradient step (owns DDPGActorLoss).

Writes state.actor_loss.
"""

from ...constants import DT
from ...core.module import Module
from ...optimizer.adam import Adam
from ...loss.ddpg_actor_loss import DDPGActorLoss
from ..trainer_block import TrainerState


struct DDPGActorStep[
    OBS_: Int, ACT_: Int, BATCH_: Int, ACTOR: Module, CRITIC: Module,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime Inner = DDPGActorLoss[Self.ACTOR, Self.CRITIC, Self.BATCH]

    var inner: Self.Inner

    def __init__(out self):
        self.inner = Self.Inner()

    @staticmethod
    def make[target: StaticString]() raises -> Self:
        comptime assert target == "cpu", (
            "DDPGActorStep.make[target='gpu'] not yet supported"
        )
        var b = Self()
        b.inner = Self.Inner.make[target]()
        return b^

    def step[target: StaticString](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut actor: Self.ACTOR,
        mut actor_opt: Adam,
        mut critic: Self.CRITIC,
    ) raises:
        var loss = self.inner.forward_backward[target, OPT=Adam](
            actor, actor_opt, critic, state.mb_s.cpu_ptr(),
        )
        state.actor_loss = loss

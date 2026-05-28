"""DDPGActorStep — DDPG actor (DPG) gradient step (owns DDPGActorLoss).

Writes state.actor_loss.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.optimizer.adam import Adam
from ..actor_loss import DDPGActorLoss
from ...training.trainer_block import TrainerState


struct DDPGActorStep[
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
    ACTOR: Module,
    CRITIC: Module,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime Inner = DDPGActorLoss[Self.ACTOR, Self.CRITIC, Self.BATCH]

    var inner: Self.Inner

    def __init__(out self):
        self.inner = Self.Inner()

    @staticmethod
    def make[
        target: StaticString
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var b = Self()
        b.inner = Self.Inner.make[target](ctx)
        return b^

    def step[
        target: StaticString
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut actor: Self.ACTOR,
        mut actor_opt: Adam,
        mut critic: Self.CRITIC,
    ) raises:
        var loss = self.inner.forward_backward[target, OPT=Adam](
            actor,
            actor_opt,
            critic,
            state.mb_s.target_ptr[target](),
        )
        state.actor_loss = loss

"""DDPGPolyakStep — polyak update for one actor pair + one critic pair.

Used by DDPG. (TD3 uses TD3DelayedActorPolyakStep — gated 3-pair polyak.)
"""

from ...constants import DT
from ...core.module import Module
from ...core.online_target_pair import OnlineTargetPair
from ..trainer_block import TrainerState


struct DDPGPolyakStep[
    OBS_: Int, ACT_: Int, BATCH_: Int, ACTOR: Module, CRITIC: Module,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime APair = OnlineTargetPair[Self.ACTOR]
    comptime CPair = OnlineTargetPair[Self.CRITIC]

    var tau: Scalar[DT]

    def __init__(out self):
        self.tau = Scalar[DT](0.005)

    @staticmethod
    def make(tau: Scalar[DT]) -> Self:
        var b = Self()
        b.tau = tau
        return b^

    def step[target: StaticString](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut actor_pair: Self.APair,
        mut critic_pair: Self.CPair,
    ) raises:
        comptime if target == "cpu":
            actor_pair.polyak_step["cpu"](self.tau)
            critic_pair.polyak_step["cpu"](self.tau)
        else:
            actor_pair.polyak_step["gpu"](self.tau, state.ctx)
            critic_pair.polyak_step["gpu"](self.tau, state.ctx)

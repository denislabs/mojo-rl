"""DDPGPolyakStep — polyak update for one actor pair + one critic pair.

Used by DDPG. (TD3 uses TD3DelayedActorPolyakStep — gated 3-pair polyak.)
"""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.module import Module
from ...core.online_target_pair import OnlineTargetPair
from ...training.trainer_block import TrainerState


struct DDPGPolyakStep[
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
    ACTOR: Module,
    CRITIC: Module,
](Defaultable & Movable & ImplicitlyDeletable):
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

    def step[
        target: StaticString
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut actor_pair: Self.APair,
        mut critic_pair: Self.CPair,
    ) raises:
        actor_pair.polyak_step[target](self.tau, state.ctx)
        critic_pair.polyak_step[target](self.tau, state.ctx)

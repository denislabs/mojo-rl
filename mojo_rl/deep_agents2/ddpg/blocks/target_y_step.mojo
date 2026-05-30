"""DDPGTargetYStep — wraps DDPGTargetYBlock (owns the inner block).

Reads state.mb_sp, state.mb_r → writes state.mb_y.
"""

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from ..target_y_block import DDPGTargetYBlock
from ...training.trainer_block import TrainerState


struct DDPGTargetYStep[
    OBS_: Int, ACT_: Int, BATCH_: Int, ACTOR: Module, CRITIC: Module,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime Inner = DDPGTargetYBlock[
        Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS, Self.ACT,
    ]

    var inner: Self.Inner

    def __init__(out self):
        self.inner = Self.Inner()

    @staticmethod
    def make[target: StaticString](
        action_scale: Scalar[DT], gamma: Scalar[DT],
    ) raises -> Self:
        comptime assert target == "cpu", (
            "DDPGTargetYStep.make[target='gpu'] not yet supported"
        )
        var b = Self()
        b.inner = Self.Inner.make[target](
            action_scale=action_scale, gamma=gamma,
        )
        return b^

    def step[target: StaticString](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut actor_t: Self.ACTOR,
        mut critic_t: Self.CRITIC,
    ) raises:
        self.inner.step[target](
            actor_t, critic_t,
            state.mb_sp.cpu_ptr(), state.mb_r.cpu_ptr(),
            state.mb_d.cpu_ptr(),
            state.mb_y.cpu_ptr(),
        )

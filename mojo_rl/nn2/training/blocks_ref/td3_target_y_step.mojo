"""TD3TargetYStep — wraps TD3TargetYBlock (target smoothing on a').

Reads state.mb_sp, state.mb_r → writes state.mb_y.
"""

from ...constants import DT
from ...core.module import Module
from ..td3_target_y_block import TD3TargetYBlock
from ..trainer_block import TrainerState


struct TD3TargetYStep[
    OBS_: Int, ACT_: Int, BATCH_: Int, ACTOR: Module, CRITIC: Module,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime Inner = TD3TargetYBlock[
        Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS, Self.ACT,
    ]

    var inner: Self.Inner

    def __init__(out self):
        self.inner = Self.Inner()

    @staticmethod
    def make[target: StaticString](
        action_scale: Scalar[DT], gamma: Scalar[DT],
        noise_std: Scalar[DT], noise_clip: Scalar[DT],
    ) raises -> Self:
        comptime assert target == "cpu", (
            "TD3TargetYStep.make[target='gpu'] not yet supported"
        )
        var b = Self()
        b.inner = Self.Inner.make[target](
            action_scale=action_scale, gamma=gamma,
            noise_std=noise_std, noise_clip=noise_clip,
        )
        return b^

    def step[target: StaticString](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut actor_t: Self.ACTOR,
        mut critic1_t: Self.CRITIC,
        mut critic2_t: Self.CRITIC,
    ) raises:
        self.inner.step[target](
            actor_t, critic1_t, critic2_t,
            state.mb_sp.cpu_ptr(), state.mb_r.cpu_ptr(),
            state.mb_y.cpu_ptr(),
        )

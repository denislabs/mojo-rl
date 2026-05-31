"""TD3TargetYStep — wraps TD3TargetYBlock (target smoothing on a').

Reads state.mb_sp, state.mb_r, state.mb_d → writes state.mb_y.

CPU + GPU. The inner `TD3TargetYBlock` carries a full GPU path (FullGraph
forward + device Philox target-policy noise + `apply_terminal_mask`); this
wrapper routes the optional `DeviceContext` and reads the minibatch
scratches through `target_ptr[target]()`.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module
from ..target_y_block import TD3TargetYBlock
from ...training.trainer_block import TrainerState


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
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx` is required for `target='gpu'`."""
        comptime assert target == "cpu" or target == "gpu", (
            "TD3TargetYStep: target must be 'cpu' or 'gpu'"
        )
        var b = Self()
        comptime if target == "cpu":
            b.inner = Self.Inner.make[target](
                action_scale=action_scale, gamma=gamma,
                noise_std=noise_std, noise_clip=noise_clip,
            )
        else:
            b.inner = Self.Inner.make[target](
                ctx.value(),
                action_scale=action_scale, gamma=gamma,
                noise_std=noise_std, noise_clip=noise_clip,
            )
        return b^

    def step[target: StaticString, POLICY: AMPPolicy = NoAMP](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut actor_t: Self.ACTOR,
        mut critic1_t: Self.CRITIC,
        mut critic2_t: Self.CRITIC,
    ) raises:
        self.inner.step[target, POLICY](
            actor_t, critic1_t, critic2_t,
            state.mb_sp.target_ptr[target](), state.mb_r.target_ptr[target](),
            state.mb_d.target_ptr[target](),
            state.mb_y.target_ptr[target](),
        )

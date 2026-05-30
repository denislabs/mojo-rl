"""DDPGTargetYStep — wraps DDPGTargetYBlock (owns the inner block).

Reads state.mb_sp, state.mb_r, state.mb_d → writes state.mb_y.

CPU + GPU. The inner `DDPGTargetYBlock` already carries a full GPU path
(FullGraph forward + `apply_terminal_mask`); this wrapper just routes the
optional `DeviceContext` and reads the minibatch scratches through
`target_ptr[target]()`.
"""

from std.gpu.host import DeviceContext

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
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx` is required for `target='gpu'`
        (the inner block's GPU make takes a concrete `DeviceContext`)."""
        comptime assert target == "cpu" or target == "gpu", (
            "DDPGTargetYStep: target must be 'cpu' or 'gpu'"
        )
        var b = Self()
        comptime if target == "cpu":
            b.inner = Self.Inner.make[target](
                action_scale=action_scale, gamma=gamma,
            )
        else:
            b.inner = Self.Inner.make[target](
                ctx.value(), action_scale=action_scale, gamma=gamma,
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
            state.mb_sp.target_ptr[target](), state.mb_r.target_ptr[target](),
            state.mb_d.target_ptr[target](),
            state.mb_y.target_ptr[target](),
        )

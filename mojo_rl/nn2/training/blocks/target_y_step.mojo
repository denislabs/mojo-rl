"""TargetYStep — SAC target y compute (owns inner TargetYBlock graph).

Computes y = r + γ·(min(Q1', Q2')(s', a') − α·log_pi(a'|s')) where
a' ~ tanh(actor(s')). Writes into state.mb_y.
"""

from std.gpu.host import DeviceContext

from ...constants import DT
from ...core.amp import AMPPolicy, NoAMP
from ...core.module import Module
from ..target_y_block import TargetYBlock
from ..trainer_block import TrainerState


struct TargetYStep[
    OBS_: Int, ACT_: Int, BATCH_: Int, ACTOR: Module, CRITIC: Module,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime Inner = TargetYBlock[
        Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS, Self.ACT,
    ]

    var inner: Self.Inner

    def __init__(out self):
        self.inner = Self.Inner()

    @staticmethod
    def make[target: StaticString = "cpu"](
        action_scale: Scalar[DT], gamma: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified make — matmul-style `Optional[DeviceContext]`."""
        var b = Self()
        comptime if target == "cpu":
            b.inner = Self.Inner.make[target](
                action_scale=action_scale, gamma=gamma,
            )
        else:
            if not ctx:
                raise Error(
                    "TargetYStep.make[target='gpu']: ctx required"
                )
            b.inner = Self.Inner.make[target](
                ctx.value(), action_scale=action_scale, gamma=gamma,
            )
        return b^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut actor: Self.ACTOR,
        mut tgt1: Self.CRITIC,
        mut tgt2: Self.CRITIC,
    ) raises:
        self.inner.step[target, POLICY](
            actor, tgt1, tgt2,
            state.mb_sp.target_ptr[target](),
            state.mb_r.target_ptr[target](),
            state.alpha,
            state.mb_y.target_ptr[target](),
        )

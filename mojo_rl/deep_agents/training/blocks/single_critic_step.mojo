"""SingleCriticStep — single-critic update (sa concat) for DDPG.

Wraps `CriticUpdateBlock[CRITIC, BATCH, SA_DIM]`. Builds sa = concat(s, a)
internally (the critic forward takes the pre-concatenated [B, OBS+ACT]). Owns
the sa scratch since the SA shape is block-specific (TrainerState carries only
OBS/ACT).

STORAGE migration (Stage 5): the single-critic sibling of TwinCriticStep —
storage `Tensor` concat scratch, no TargetStorage/Scratch/TileTensor; passes the
storage TrainerState minibatch Tensors straight into the migrated
`CriticUpdateBlock`. CPU + GPU.
"""

from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.storage.core.module import Module
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.optimizer.adam import Adam
from ...loss.critic_update_block import CriticUpdateBlock
from ...training.off_policy_critic import concat_sa, concat_sa_gpu
from ..trainer_block import TrainerState


struct SingleCriticStep[
    OBS_: Int,
    ACT_: Int,
    BATCH_: Int,
    CRITIC: Module,
](Defaultable & Movable & ImplicitlyDeletable):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime SA = Self.OBS + Self.ACT
    comptime Inner = CriticUpdateBlock[Self.CRITIC, Self.BATCH, Self.SA]

    var inner: Self.Inner
    var _mb_sa: Tensor

    def __init__(out self):
        self.inner = Self.Inner()
        self._mb_sa = Tensor()

    @staticmethod
    def make[
        target: StaticString = "cpu"
    ](ctx: Optional[DeviceContext] = None,) raises -> Self:
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "SingleCriticStep: target must be 'cpu' or 'gpu'"
        var b = Self()
        b.inner = Self.Inner.make[target](ctx=ctx)
        comptime if target == "cpu":
            b._mb_sa = Tensor.alloc(Self.BATCH * Self.SA)
        else:
            b._mb_sa = Tensor.alloc_gpu(ctx.value(), Self.BATCH * Self.SA)
        return b^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
        ACCUMULATE: Bool = False,
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut critic: Self.CRITIC,
        mut critic_opt: Adam,
    ) raises:
        comptime if target == "cpu":
            concat_sa[Self.OBS, Self.ACT, Self.BATCH](
                state.mb_s.lt["cpu", Layout.row_major(Self.BATCH, Self.OBS)](),
                state.mb_a.lt["cpu", Layout.row_major(Self.BATCH, Self.ACT)](),
                self._mb_sa.lt["cpu", Layout.row_major(Self.BATCH, Self.SA)](),
            )
        else:
            concat_sa_gpu[Self.OBS, Self.ACT, Self.BATCH](
                state.ctx.value(),
                state.mb_s.lt["gpu", Layout.row_major(Self.BATCH, Self.OBS)](),
                state.mb_a.lt["gpu", Layout.row_major(Self.BATCH, Self.ACT)](),
                self._mb_sa.lt["gpu", Layout.row_major(Self.BATCH, Self.SA)](),
            )

        var weights: Optional[
            LayoutTensor[DT, Layout.row_major(Self.BATCH), MutAnyOrigin]
        ] = None
        var td_residuals: Optional[
            LayoutTensor[DT, Layout.row_major(Self.BATCH), MutAnyOrigin]
        ] = None
        if state.has_per:
            weights = state.mb_w.lt[target, Layout.row_major(Self.BATCH)]()
            td_residuals = state.td_residuals.lt[
                target, Layout.row_major(Self.BATCH)
            ]()

        var loss = self.inner.step[target, POLICY, ACCUMULATE](
            critic,
            critic_opt,
            self._mb_sa,
            state.mb_y,
            weights=weights,
            td_residuals=td_residuals,
            ctx=state.ctx,
        )
        state.critic_loss = loss

"""DQNTargetYStep — wraps DQNTargetYBlock (owns the inner block).

Reads `state.mb_sp / mb_r / mb_d` → writes `state.mb_y`. Threads
`pair.target_net` (and `pair.online` when DOUBLE=True) into the inner
block. Analogue of `sac/blocks/target_y_step.mojo`.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module
from ..target_y_block import DQNTargetYBlock
from ...training.trainer_block import TrainerState


struct DQNTargetYStep[
    OBS_: Int, ACT_: Int, BATCH_: Int, NA_: Int,
    Q_NET: Module, DOUBLE: Bool = False,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_  # = 1 for DQN (action index)
    comptime BATCH = Self.BATCH_
    comptime NA = Self.NA_
    comptime Inner = DQNTargetYBlock[
        Self.Q_NET, Self.BATCH, Self.OBS, Self.NA, Self.DOUBLE,
    ]

    var inner: Self.Inner

    def __init__(out self):
        self.inner = Self.Inner()

    @staticmethod
    def make[target: StaticString = "cpu"](
        gamma: Scalar[DT] = Scalar[DT](0.99),
        nstep: Int = 1,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        var b = Self()
        b.inner = Self.Inner.make[target](gamma=gamma, nstep=nstep, ctx=ctx)
        return b^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut q_target: Self.Q_NET,
        mut q_online: Self.Q_NET,
    ) raises:
        """`q_online` is ignored on the standard path (DOUBLE=False) but
        always threaded so the trainer call site is uniform."""
        self.inner.step[target, POLICY](
            q_target,
            q_online,
            state.mb_sp.target_ptr[target](),
            state.mb_r.target_ptr[target](),
            state.mb_d.target_ptr[target](),
            state.mb_y.target_ptr[target](),
        )

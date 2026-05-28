"""DQNQUpdateStep — wraps DQNQUpdateBlock (owns the inner block).

Reads `state.mb_s / mb_a / mb_y` → writes `state.critic_loss`. Analogue
of `training/blocks/single_critic_step.mojo` but for a single-Q-net
discrete agent (gather instead of `sa` concat).
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.optimizer.adam import Adam
from ..q_update_block import DQNQUpdateBlock
from ...training.trainer_block import TrainerState


struct DQNQUpdateStep[
    OBS_: Int, ACT_: Int, BATCH_: Int, NA_: Int, Q_NET: Module,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_  # = 1 for DQN
    comptime BATCH = Self.BATCH_
    comptime NA = Self.NA_
    comptime Inner = DQNQUpdateBlock[
        Self.Q_NET, Self.BATCH, Self.OBS, Self.NA,
    ]

    var inner: Self.Inner

    def __init__(out self):
        self.inner = Self.Inner()

    @staticmethod
    def make[target: StaticString = "cpu"](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        var b = Self()
        b.inner = Self.Inner.make[target](ctx=ctx)
        return b^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut q_online: Self.Q_NET,
        mut q_opt: Adam,
    ) raises:
        """PER hook: when `state.has_per`, forward IS weights into the
        update + capture per-sample signed TD residuals so the sample
        block can refresh sum-tree priorities. When unset, both
        pointers stay null and the inner block falls back to the
        uniform path (bit-identical to pre-PER)."""
        var weights_p = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0,
        )
        var td_res_p = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0,
        )
        if state.has_per:
            weights_p = state.mb_w.target_ptr[target]()
            td_res_p = state.td_residuals.target_ptr[target]()
        var loss = self.inner.step[target, POLICY](
            q_online,
            q_opt,
            state.mb_s.target_ptr[target](),
            state.mb_a.target_ptr[target](),
            state.mb_y.target_ptr[target](),
            weights_p=weights_p,
            td_residuals_p=td_res_p,
        )
        state.critic_loss = loss

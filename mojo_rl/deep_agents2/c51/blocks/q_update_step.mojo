"""C51QUpdateStep — wraps C51QUpdateBlock.

Reads `state.mb_s / mb_a` + caller-supplied `mb_m_ptr` (target
distribution) → writes `state.critic_loss`.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.optimizer.adam import Adam
from ..q_update_block import C51QUpdateBlock
from ...training.trainer_block import TrainerState


struct C51QUpdateStep[
    OBS_: Int, ACT_: Int, BATCH_: Int, NA_: Int, N_ATOMS_: Int, Q_NET: Module,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime NA = Self.NA_
    comptime N_ATOMS = Self.N_ATOMS_
    comptime Inner = C51QUpdateBlock[
        Self.Q_NET, Self.BATCH, Self.OBS, Self.NA, Self.N_ATOMS,
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
        mb_m_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """PER hook: when `state.has_per`, forward IS weights into the
        update and capture per-sample CE residuals."""
        var weights_p: Optional[
            UnsafePointer[Scalar[DT], MutAnyOrigin]
        ] = None
        var td_res_p: Optional[
            UnsafePointer[Scalar[DT], MutAnyOrigin]
        ] = None
        if state.has_per:
            weights_p = state.mb_w.target_ptr[target]()
            td_res_p = state.td_residuals.target_ptr[target]()
        var loss = self.inner.step[target, POLICY](
            q_online,
            q_opt,
            state.mb_s.target_ptr[target](),
            state.mb_a.target_ptr[target](),
            mb_m_ptr,
            weights_p=weights_p,
            td_residuals_p=td_res_p,
        )
        state.critic_loss = loss

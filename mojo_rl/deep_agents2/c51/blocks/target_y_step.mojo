"""C51TargetYStep — wraps C51TargetYBlock.

Reads `state.mb_sp / mb_r / mb_d` → writes the [B, N_ATOMS] target
distribution into the caller-supplied `mb_m_ptr`. The trainer owns
that buffer (TrainerState's `mb_y` is [B, 1] — too small for C51).
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module
from ..target_y_block import C51TargetYBlock
from ...training.trainer_block import TrainerState


struct C51TargetYStep[
    OBS_: Int, ACT_: Int, BATCH_: Int, NA_: Int, N_ATOMS_: Int,
    Q_NET: Module, DOUBLE: Bool = False,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_  # = 1 for C51 (action index)
    comptime BATCH = Self.BATCH_
    comptime NA = Self.NA_
    comptime N_ATOMS = Self.N_ATOMS_
    comptime Inner = C51TargetYBlock[
        Self.Q_NET, Self.BATCH, Self.OBS, Self.NA, Self.N_ATOMS, Self.DOUBLE,
    ]

    var inner: Self.Inner

    def __init__(out self):
        self.inner = Self.Inner()

    @staticmethod
    def make[target: StaticString = "cpu"](
        gamma: Scalar[DT] = Scalar[DT](0.99),
        nstep: Int = 1,
        v_min: Scalar[DT] = Scalar[DT](-10.0),
        v_max: Scalar[DT] = Scalar[DT](10.0),
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        var b = Self()
        b.inner = Self.Inner.make[target](
            gamma=gamma, nstep=nstep,
            v_min=v_min, v_max=v_max, ctx=ctx,
        )
        return b^

    def z_ptr(mut self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self.inner.z_ptr()

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut q_target: Self.Q_NET,
        mut q_online: Self.Q_NET,
        mb_m_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        self.inner.step[target, POLICY](
            q_target,
            q_online,
            state.mb_sp.target_ptr[target](),
            state.mb_r.target_ptr[target](),
            state.mb_d.target_ptr[target](),
            mb_m_ptr,
        )

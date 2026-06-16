"""TwinCriticStep — twin-critic gradient step (owns inner TwinCriticUpdateBlock).

Reads state.mb_s, state.mb_a, state.mb_y → writes state.critic_loss.
Reusable across SAC, TD3, MBPO.
"""

from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.optimizer.adam import Adam
from ...loss.critic_update_block import TwinCriticUpdateBlock
from ..trainer_block import TrainerState


struct TwinCriticStep[
    OBS_: Int, ACT_: Int, BATCH_: Int, CRITIC: Module,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime Inner = TwinCriticUpdateBlock[
        Self.CRITIC, Self.BATCH, Self.OBS, Self.ACT,
    ]

    var inner: Self.Inner

    def __init__(out self):
        self.inner = Self.Inner()

    @staticmethod
    def make[target: StaticString = "cpu"](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified make — matmul-style `Optional[DeviceContext]`. Inner
        block now accepts the optional directly (no boundary shim)."""
        var b = Self()
        b.inner = Self.Inner.make[target](ctx=ctx)
        return b^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
        ACCUMULATE: Bool = False,
    ](
        mut self,
        mut state: TrainerState[Self.OBS, Self.ACT, Self.BATCH],
        mut critic1: Self.CRITIC,
        mut critic1_opt: Adam,
        mut critic2: Self.CRITIC,
        mut critic2_opt: Adam,
    ) raises:
        var mb_y_t = TileTensor(
            state.mb_y.target_ptr[target](), row_major[Self.BATCH, 1](),
        )
        # PER hook: when state.has_per is set, forward IS weights into the
        # update and capture per-sample signed TD residuals. When unset,
        # both pointers stay None and the inner block falls back to the
        # uniform path (bit-identical to pre-PER).
        var weights_p: Optional[
            UnsafePointer[Scalar[DT], MutAnyOrigin]
        ] = None
        var td_res_p: Optional[
            UnsafePointer[Scalar[DT], MutAnyOrigin]
        ] = None
        if state.has_per:
            weights_p = state.mb_w.target_ptr[target]()
            td_res_p  = state.td_residuals.target_ptr[target]()
        var loss = self.inner.step[target, POLICY, ACCUMULATE](
            critic1, critic1_opt, critic2, critic2_opt,
            state.mb_s.target_ptr[target](),
            state.mb_a.target_ptr[target](),
            mb_y_t,
            weights_p=weights_p,
            td_residuals_p=td_res_p,
        )
        # With ACCUMULATE (GPU) the per-batch loss is reduced on-device into
        # the critics' accumulators; `loss` is a 0 sentinel here and the
        # real metric is read at flush. Otherwise `loss` is the live scalar.
        state.critic_loss = loss

"""SingleCriticStep — DDPG critic update (single critic, sa concat).

Wraps `CriticUpdateBlock[CRITIC, BATCH, SA_DIM]`. Builds sa = concat(s, a)
internally (DDPG critic forward takes sa). Owns the sa scratch since the
SA shape is block-specific (TrainerState only carries OBS/ACT).

CPU + GPU. The CPU path is bit-identical to the pre-Phase-4 block; the
GPU path mirrors `TwinCriticStep`/`TwinCriticUpdateBlock`: the concat
runs via `concat_sa_gpu`, and the inner `CriticUpdateBlock` (which
already has a full GPU path) takes over from there. With `ACCUMULATE`
(GPU) the per-batch loss is reduced on-device into the critic's
accumulator — the host reads it at flush cadence via
`inner.mse_loss.read_accum`, so the per-step path stays D2H-free /
CUDA-graph capturable.
"""

from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.scratch import Scratch
from mojo_rl.nn.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn.core.target_storage import TargetStorage, assert_tag_for
from mojo_rl.nn.optimizer.adam import Adam
from ...loss.critic_update_block import CriticUpdateBlock
from ...training.off_policy_critic import concat_sa, concat_sa_gpu
from ..trainer_block import TrainerState


struct SingleCriticStep[
    OBS_: Int, ACT_: Int, BATCH_: Int, CRITIC: Module,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime BATCH = Self.BATCH_
    comptime SA = Self.OBS + Self.ACT
    comptime Inner = CriticUpdateBlock[Self.CRITIC, Self.BATCH, Self.SA]

    var inner: Self.Inner
    var _mb_sa: Scratch["mb_sa", Self.BATCH * Self.SA]
    # ctx + tag for the GPU concat (CPU: ctx is None, tag is the cpu tag).
    var ts: TargetStorage

    def __init__(out self):
        self.inner = Self.Inner()
        self._mb_sa = Scratch["mb_sa", Self.BATCH * Self.SA]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString = "cpu"](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "SingleCriticStep: target must be 'cpu' or 'gpu'"
        )
        var b = Self()
        b.inner = Self.Inner.make[target](ctx=ctx)
        b.ts = TargetStorage.make[target](ctx=ctx)
        init_scratch_auto[Self, target](b, ctx)
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
        assert_tag_for["SingleCriticStep", target](self.ts.target_tag)

        # Build sa = concat(s, a) on the matching backend.
        var sa_p = self._mb_sa.target_ptr[target]()
        comptime if target == "cpu":
            concat_sa[Self.OBS, Self.ACT, Self.BATCH](
                state.mb_s.target_ptr[target](),
                state.mb_a.target_ptr[target](),
                sa_p,
            )
        else:
            concat_sa_gpu[Self.OBS, Self.ACT, Self.BATCH](
                self.ts.ctx.value(),
                state.mb_s.target_ptr[target](),
                state.mb_a.target_ptr[target](),
                sa_p,
            )
        var sa_t = TileTensor(sa_p, row_major[Self.BATCH, Self.SA]())
        var y_t = TileTensor(
            state.mb_y.target_ptr[target](), row_major[Self.BATCH, 1]()
        )

        # PER hook (mirrors TwinCriticStep): forward IS weights + capture
        # signed TD residuals when state.has_per is set. None otherwise →
        # uniform path, bit-identical to pre-PER.
        var weights_p: Optional[
            UnsafePointer[Scalar[DT], MutAnyOrigin]
        ] = None
        var td_res_p: Optional[
            UnsafePointer[Scalar[DT], MutAnyOrigin]
        ] = None
        if state.has_per:
            weights_p = state.mb_w.target_ptr[target]()
            td_res_p = state.td_residuals.target_ptr[target]()

        var loss = self.inner.step[target, POLICY, ACCUMULATE](
            critic, critic_opt, sa_t, y_t,
            weights_p=weights_p,
            td_residuals_p=td_res_p,
        )
        # With ACCUMULATE (GPU) `loss` is a 0 sentinel and the real metric
        # is read from the critic's device accumulator at flush. Otherwise
        # it's the live scalar.
        state.critic_loss = loss

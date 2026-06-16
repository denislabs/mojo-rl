"""LossBlockBundle[*BLOCKS: LossBlock] — variadic container (Block E-3).

Parallel to `OptimizerBundle` for loss blocks. Trainers with multiple
loss blocks collapse N parallel fields + N parallel makes into one
bundle field.

Usage (DreamerV3 example, illustrative):

    comptime Blocks = LossBlockBundle[
        WorldModelLoss[...],
        RewardHeadLoss[...],
        DoneHeadLoss[...],
        ActorLoss[...],
        CriticLoss[...],
    ]
    var blocks = Blocks.make_default["cpu"]()
    blocks.items[0] = WorldModelLoss[...].make["cpu"](...)
    blocks.items[1] = RewardHeadLoss[...].make["cpu"](...)
    # ...
    # Domain-specific step calls at the trainer level:
    blocks.items[0].step["cpu", OPT=Adam](world_model, ws_opt, mb_seq, ...)

The bundle does not dispatch `step` itself (signatures vary per block).
It exists for storage consolidation + lifecycle uniformity.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.core.target_storage import TargetStorage, assert_tag_for
from .loss_block import LossBlock


struct LossBlockBundle[*BLOCKS: LossBlock](
    Defaultable & Movable & ImplicitlyDestructible
):
    comptime N = Self.BLOCKS.size

    var items: Tuple[*Self.BLOCKS]
    var ts: TargetStorage

    def __init__(out self):
        comptime assert Self.N >= 1, "LossBlockBundle: at least one block"
        self.items = Tuple[*Self.BLOCKS]()
        self.ts = TargetStorage.make_uninit()

    def __init__(out self, var *blocks: *Self.BLOCKS):
        """Variadic consume — accepts pre-built loss blocks (CPU)."""
        comptime assert Self.N >= 1, "LossBlockBundle: at least one block"
        self.items = Tuple(*blocks^)
        self.ts = TargetStorage.make_cpu()

    def __init__(out self, ctx: DeviceContext, var *blocks: *Self.BLOCKS) raises:
        """GPU constructor — bundle tag set to gpu + ctx stored."""
        comptime assert Self.N >= 1, "LossBlockBundle: at least one block"
        self.items = Tuple(*blocks^)
        self.ts = TargetStorage.make_gpu(ctx)

    @staticmethod
    def make_default[target: StaticString]() raises -> Self:
        """CPU factory: default-init each block. Caller assigns real
        per-instance blocks into `items[i]` post-construction."""
        comptime assert target == "cpu", (
            "LossBlockBundle.make_default[target='gpu'] requires a DeviceContext"
        )
        var b = Self()
        b.ts = TargetStorage.make_cpu()
        return b^

    @staticmethod
    def make_default[target: StaticString](ctx: DeviceContext) raises -> Self:
        """GPU factory: default-init each block + record ctx."""
        comptime assert target == "gpu", (
            "LossBlockBundle.make_default[target='cpu'](ctx) — drop ctx for CPU"
        )
        var b = Self()
        b.ts = TargetStorage.make_gpu(ctx)
        return b^

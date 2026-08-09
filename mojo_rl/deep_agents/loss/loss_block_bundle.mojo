"""LossBlockBundle[*BLOCKS: LossBlock] — variadic container (Block E-3).

Trainers with multiple loss blocks collapse N parallel fields + N parallel
makes into one bundle field. The bundle does NOT dispatch `step` (signatures
vary per block) — it exists for storage consolidation + lifecycle uniformity.

STORAGE migration (Stage 5): the legacy `TargetStorage` tag/ctx is dropped — in
the storage design `target` is a comptime parameter everywhere and each loss
block carries its own device state, so the bundle is a plain owning `Tuple`.
`make_default[target]` (CPU) / `make_default[target](ctx)` (GPU) kept for call-
site parity; the caller assigns real per-instance blocks into `items[i]`.
"""

from std.gpu.host import DeviceContext

from .loss_block import LossBlock


struct LossBlockBundle[*BLOCKS: LossBlock](
    Defaultable & Movable & ImplicitlyDeletable
):
    comptime N = Self.BLOCKS.length

    var items: Tuple[*Self.BLOCKS]

    def __init__(out self):
        comptime assert Self.N >= 1, "LossBlockBundle: at least one block"
        self.items = Tuple[*Self.BLOCKS]()

    def __init__(out self, var *blocks: *Self.BLOCKS):
        """Variadic consume — accepts pre-built loss blocks."""
        comptime assert Self.N >= 1, "LossBlockBundle: at least one block"
        self.items = Tuple(*blocks^)

    @staticmethod
    def make_default[target: StaticString]() raises -> Self:
        """CPU factory: default-init each block. Caller assigns real
        per-instance blocks into `items[i]` post-construction."""
        comptime assert target == "cpu", (
            "LossBlockBundle.make_default[target='gpu'] requires a DeviceContext"
        )
        return Self()

    @staticmethod
    def make_default[target: StaticString](ctx: DeviceContext) raises -> Self:
        """GPU factory: default-init each block. `ctx` unused (each block owns
        its device state via its own make[gpu]); kept for call-site parity."""
        comptime assert target == "gpu", (
            "LossBlockBundle.make_default[target='cpu'](ctx) — drop ctx for CPU"
        )
        return Self()

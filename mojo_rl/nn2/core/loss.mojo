"""Loss trait — (logits, targets) → scalar + grad_logits.

Phase 2.4: methods take `target: StaticString` as comptime method param.
Tensor args use generic `MutOrigin` (see module.mojo for rationale).
"""

from std.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout

from ..constants import DT
from .amp import AMPPolicy, NoAMP


trait Loss(Defaultable & Movable & ImplicitlyDestructible):
    comptime OUT_DIM: Int

    @staticmethod
    def make[target: StaticString]() raises -> Self:
        ...

    @staticmethod
    def make[target: StaticString](ctx: DeviceContext) raises -> Self:
        ...

    def forward[
        target: StaticString,
        BATCH: Int,
        LL: TensorLayout,
        LT: TensorLayout,
        OL: MutOrigin,
        OT: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        logits: TileTensor[DT, LL, OL],
        targets: TileTensor[DT, LT, OT],
    ) raises -> Scalar[DT]:
        ...

    def backward[
        target: StaticString,
        BATCH: Int,
        LT: TensorLayout,
        LG: TensorLayout,
        OT: MutOrigin,
        OG: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        targets: TileTensor[DT, LT, OT],
        mut grad_logits: TileTensor[DT, LG, OG],
    ) raises:
        ...

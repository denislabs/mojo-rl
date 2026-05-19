"""Module trait — uniform tree-walk API for leaves and combinators.

Phase 2.4: `target` is a comptime method param. Modules carry a runtime
`_target_tag` set by `make[target, INIT]`, asserted by every method.

Tensor args use generic `MutOrigin` so callers can pass `TileTensor`
views built directly from `DeviceBuffer` (narrow origin) without an
intermediate `MutAnyOrigin` widening step. Impl bodies that pipe pointers
into kernels accept the generic origin and rebind to `MutAnyOrigin` only
at the kernel-launch boundary.

Trait requirements:
  - `Defaultable`: zero-arg `__init__()` yields empty placeholders.
  - `IN_DIM`, `OUT_DIM`: comptime ints.
  - `make[target, INIT]()` / `make[target, INIT](ctx)`: static factories.
  - `forward`, `backward`, `for_each_param`: see signatures below.
"""

from std.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout

from ..constants import DT
from .param_visitor import ParamVisitor
from .initializer import Initializer


trait Module(Defaultable & Movable & ImplicitlyDestructible):
    comptime IN_DIM: Int
    comptime OUT_DIM: Int

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        ...

    @staticmethod
    def make[target: StaticString, INIT: Initializer](ctx: DeviceContext) raises -> Self:
        ...

    def forward[
        target: StaticString,
        BATCH: Int,
        LIN: TensorLayout,
        LOUT: TensorLayout,
        OIN: MutOrigin,
        OOUT: MutOrigin,
    ](
        mut self,
        input: TileTensor[DT, LIN, OIN],
        mut output: TileTensor[DT, LOUT, OOUT],
    ) raises:
        ...

    def backward[
        target: StaticString,
        BATCH: Int,
        LGO: TensorLayout,
        LGI: TensorLayout,
        OGO: MutOrigin,
        OGI: MutOrigin,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, OGO],
        mut grad_input: TileTensor[DT, LGI, OGI],
    ) raises:
        ...

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](
        mut self,
        prefix: String,
        mut visitor: V,
    ) raises:
        ...

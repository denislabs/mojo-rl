"""TernaryModule trait — 3-input → 1-output building block. Phase 10C.

Sibling of `Module` (1→1) and `BinaryModule` (2→1). No concrete impl
yet — scaffolded ahead of ComputeGraph v2 (Phase 10D) so the graph's
node-kind dispatch can treat 1/2/3-input nodes uniformly from day one.
Likely first uses: 3-way `min` / `max`, three-branch fusions in
MuZero-style heads.

Shape:
  - `IN0_DIM`, `IN1_DIM`, `IN2_DIM`, `OUT_DIM`
  - forward:  `(in0, in1, in2) → output`
  - backward: `grad_output → (grad_in0, grad_in1, grad_in2)`

Same explicit per-arg `TensorLayout` + `MutOrigin` generics + default
buffer surface as BinaryModule.
"""

from std.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout

from ..constants import DT
from .param_visitor import ParamVisitor
from .initializer import Initializer
from .amp import AMPPolicy, NoAMP


trait TernaryModule(Defaultable & Movable & ImplicitlyDestructible):
    comptime IN0_DIM: Int
    comptime IN1_DIM: Int
    comptime IN2_DIM: Int
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
        L0: TensorLayout,
        L1: TensorLayout,
        L2: TensorLayout,
        LOUT: TensorLayout,
        O0: MutOrigin,
        O1: MutOrigin,
        O2: MutOrigin,
        OOUT: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        in0: TileTensor[DT, L0, O0],
        in1: TileTensor[DT, L1, O1],
        in2: TileTensor[DT, L2, O2],
        mut output: TileTensor[DT, LOUT, OOUT],
    ) raises:
        ...

    def backward[
        target: StaticString,
        BATCH: Int,
        LGO: TensorLayout,
        LG0: TensorLayout,
        LG1: TensorLayout,
        LG2: TensorLayout,
        OGO: MutOrigin,
        OG0: MutOrigin,
        OG1: MutOrigin,
        OG2: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, OGO],
        mut grad_in0: TileTensor[DT, LG0, OG0],
        mut grad_in1: TileTensor[DT, LG1, OG1],
        mut grad_in2: TileTensor[DT, LG2, OG2],
    ) raises:
        ...

    def backward_input[
        target: StaticString,
        BATCH: Int,
        LGO: TensorLayout,
        LG0: TensorLayout,
        LG1: TensorLayout,
        LG2: TensorLayout,
        OGO: MutOrigin,
        OG0: MutOrigin,
        OG1: MutOrigin,
        OG2: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, OGO],
        mut grad_in0: TileTensor[DT, LG0, OG0],
        mut grad_in1: TileTensor[DT, LG1, OG1],
        mut grad_in2: TileTensor[DT, LG2, OG2],
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

    def set_inference(mut self, value: Bool):
        ...

    # ── Buffer surface (default no-op + null pointers) ────────────────

    def ensure_buffers[BATCH: Int](mut self) raises:
        pass

    def out_ptr(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0)

    def grad_in0_ptr(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0)

    def grad_in1_ptr(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0)

    def grad_in2_ptr(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0)

    def grad_out_ptr(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0)

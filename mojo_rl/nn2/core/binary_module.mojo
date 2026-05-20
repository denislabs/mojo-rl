"""BinaryModule trait — 2-input → 1-output building block. Phase 10C.

Sibling of `Module` (1→1). Unlike the packed-tensor convention used by
the Phase 8.4 `Sub` / `ElemMin` (single input `[a | b]` of width `2*DIM`),
`BinaryModule` takes two *separate* input tiles. This is the natural
shape for ComputeGraph v2 (Phase 10D): the graph routes outputs of two
predecessor nodes into a binary node's `in0` / `in1` slots directly,
no pre-pack / post-unpack scratch.

Shape:
  - `IN0_DIM`, `IN1_DIM`, `OUT_DIM` — per-side feature widths
  - forward:  `(in0: [B, IN0_DIM], in1: [B, IN1_DIM]) → output: [B, OUT_DIM]`
  - backward: `grad_output → (grad_in0, grad_in1)`

Tensor args use generic `MutOrigin` per arg so callers can pass tiles
built from any source buffer (DeviceBuffer, List, etc.) without
explicit `MutAnyOrigin` widening.

Phase 10A buffer surface (CG v2 wiring): every BinaryModule owns three
output / grad buffers (one out, one grad_out, two grad_ins). Default
impls return null pointers + no-op `ensure_buffers` so concrete impls
opt in lazily. Identical pattern to the Module trait extension.
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT
from .param_visitor import ParamVisitor
from .initializer import Initializer
from .amp import AMPPolicy, NoAMP


trait BinaryModule(Defaultable & Movable & ImplicitlyDestructible):
    comptime IN0_DIM: Int
    comptime IN1_DIM: Int
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
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        in0: TileTensor[dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...],
        in1: TileTensor[dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
    ) raises:
        ...

    def backward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        grad_output: TileTensor[dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...],
        mut grad_in0: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut grad_in1: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
    ) raises:
        ...

    def backward_input[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        grad_output: TileTensor[dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...],
        mut grad_in0: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut grad_in1: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
    ) raises:
        """grad_input-only backward (no grad_w accumulation).

        For param-less ops (Sub, ElemMin, ...) this is a delegate to
        `backward`. Kept in the trait so combinators can dispatch
        uniformly when a frozen-grad path is needed.
        """
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

    # ──────────────────────────────────────────────────────────────────
    # Phase 10A-style buffer surface for ComputeGraph v2 wiring.
    #
    # Each BinaryModule owns four List[Scalar[DT]] buffers:
    #   - _out_buf       [BATCH, OUT_DIM]   forward writes here
    #   - _grad_in0_buf  [BATCH, IN0_DIM]   backward writes here (lhs)
    #   - _grad_in1_buf  [BATCH, IN1_DIM]   backward writes here (rhs)
    #   - _grad_out_buf  [BATCH, OUT_DIM]   downstream consumers
    #                                        scatter-add into; backward
    #                                        reads as grad_output
    #
    # Default impls below: no-op + null pointers. Concrete impls opt
    # in by overriding (same idiom as Module Phase 10A).
    # ──────────────────────────────────────────────────────────────────

    def ensure_buffers[BATCH: Int](mut self) raises:
        """Lazy-grow owned out / grad buffers to BATCH samples. Default no-op."""
        pass

    def out_ptr(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Pointer to owned [BATCH, OUT_DIM] output buffer. Default null."""
        return UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0)

    def grad_in0_ptr(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Pointer to owned [BATCH, IN0_DIM] grad-input-0 buffer. Default null."""
        return UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0)

    def grad_in1_ptr(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Pointer to owned [BATCH, IN1_DIM] grad-input-1 buffer. Default null."""
        return UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0)

    def grad_out_ptr(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Pointer to owned [BATCH, OUT_DIM] grad-output buffer. Default null."""
        return UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0)

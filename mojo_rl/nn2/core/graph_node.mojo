"""GraphNode trait — uniform interface over unary / binary graph nodes.

Phase 10D. ComputeGraph v2 holds a variadic pack `*NODES: GraphNode`,
walks them in topological order during forward, and in reverse during
backward. Mojo variadic packs require a single trait, so unary and
binary node wrappers share this interface.

Per-node fields:
  - NAME             unique name within the graph
  - IN0_NAME         predecessor for input slot 0 ("input" = graph input)
  - IN1_NAME         predecessor for input slot 1 ("" for unary nodes)
  - IN0_DIM/IN1_DIM  per-slot feature widths (IN1_DIM = 0 for unary)
  - OUT_DIM          output feature width
  - KIND             0 = external input slot, 1 = unary, 2 = binary
                     (3 reserved for ternary)

Per-node lifecycle:
  - `make_via[target, INIT](ctx?)` — recursive factory
  - `ensure_buffers_via[BATCH]`    — lazy-grow owned out / grad bufs
  - `out_ptr_via`, `grad_out_ptr_via`, `grad_in0_ptr_via`, `grad_in1_ptr_via`

Per-node compute:
  - `forward_via[target, BATCH](in0_ptr, in1_ptr)`
       reads inputs from raw pointers (owned by caller / predecessors),
       writes output into this node's `_out_buf`. Unary ignores `in1_ptr`.
  - `backward_via[target, BATCH]()`
       reads this node's `_grad_out_buf`, writes `_grad_in0_buf`
       (and `_grad_in1_buf` for binary). The graph scatter-adds those
       into predecessors' `_grad_out_buf`s after the call returns.

Scatter-add is done by the graph (not the node) so the trait stays
agnostic about predecessor identity.
"""

from std.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout

from ..constants import DT
from .param_visitor import ParamVisitor
from .initializer import Initializer
from .amp import AMPPolicy, NoAMP


trait GraphNode(Defaultable & Movable & ImplicitlyDestructible):
    comptime NAME: StaticString
    comptime IN0_NAME: StaticString
    comptime IN1_NAME: StaticString
    comptime IN0_DIM: Int
    comptime IN1_DIM: Int
    comptime OUT_DIM: Int
    comptime KIND: Int  # 0 = input slot, 1 = unary, 2 = binary, 3 = ternary

    # Set the externally-supplied input pointer for InputSlot (KIND=0).
    # Unary/binary nodes implement this as a no-op — they don't consume
    # an external pointer (their input comes from a predecessor node).
    def set_input_via(
        mut self,
        ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        pass

    # Phase 3 — Bind an externally-owned Module instance to an
    # ExternalUnaryNode / ExternalBinaryNode. The pointer is type-erased
    # to `UnsafePointer[Scalar[DT]]` at the trait surface; the external-
    # node implementations `rebind` it back to the typed pointer at the
    # dispatch site (Self.M is known there). InputSlot / UnaryNode /
    # BinaryNode inherit the no-op default — they own their op.
    def set_external_via(
        mut self,
        ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        pass

    @staticmethod
    def make_via[target: StaticString, INIT: Initializer]() raises -> Self:
        ...

    @staticmethod
    def make_via[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        ...

    def ensure_buffers_via[BATCH: Int](mut self) raises:
        ...

    def out_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        ...

    def grad_out_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        ...

    def grad_in0_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        ...

    def grad_in1_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        ...

    def forward_via[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        in0_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        in1_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        ...

    def backward_via[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](mut self) raises:
        ...

    def for_each_param_via[
        target: StaticString,
        V: ParamVisitor,
    ](
        mut self,
        prefix: String,
        mut visitor: V,
    ) raises:
        ...

    def set_op_attr_via[ATTR: StaticString](
        mut self, value: Scalar[DT],
    ):
        """Forward an attribute mutation to the inner op. Default
        no-op for InputSlot (no `.op` field). UnaryNode/BinaryNode
        override to call `self.op.set_attr[ATTR](value)`."""
        pass

"""GraphNode trait — uniform interface over unary / binary / ternary /
quaternary graph nodes.

Phase 10D + I.2.5. ComputeGraph v2 holds a variadic pack `*NODES:
GraphNode`, walks them in topological order during forward, and in
reverse during backward. Mojo variadic packs require a single trait,
so all node wrappers share this interface.

Per-node fields:
  - NAME             unique name within the graph
  - IN0_NAME/.../IN3_NAME  predecessor names per input slot (default "" past ARITY-1)
  - IN0_DIM/.../IN3_DIM    per-slot feature widths (0 past ARITY-1)
  - OUT_DIM          output feature width
  - KIND             0 = external input slot, 1 = unary, 2 = binary,
                     3 = ternary, 4 = quaternary

Per-node lifecycle:
  - `make_via[target, INIT](ctx?)` — recursive factory
  - `ensure_buffers_via[BATCH]`    — lazy-grow owned out / grad bufs
  - `out_ptr_via`, `grad_out_ptr_via`, `grad_in{0,1,2,3}_ptr_via`

Per-node compute:
  - `forward_via[target, BATCH](in0_ptr, in1_ptr, in2_ptr, in3_ptr)`
       reads inputs from raw pointers (owned by predecessors), writes
       output into this node's `_out_buf`. Unary ignores in1+; binary
       ignores in2+; ternary ignores in3.
  - `vjp_via[target, BATCH]()`
       reads this node's `_grad_out_buf`, writes `_grad_in{0..ARITY-1}_buf`.
       The graph scatter-adds those into predecessors' `_grad_out_buf`s
       after the call returns.

Scatter-add is done by the graph (not the node) so the trait stays
agnostic about predecessor identity.

I.2.5 raised the trait surface from ARITY≤2 to ARITY≤4 so quaternary
loss leaves (e.g. `PPOObjective4(actor_out, action, old_log_prob,
advantage)`) can be expressed natively in a ComputeGraph without the
aux-packing workaround. The node wrapper struct decls (Node /
ExternalNode in graph_nodes.mojo) use Mojo's struct comptime variadic
`*in_names: StaticString` to keep the user-facing API elegant; the
trait surface keeps fixed in0/in1/in2/in3 ptr params + comptime-if
ARITY branches for low-risk dispatch.
"""

from std.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout

from ..constants import DT
from .param_visitor import ParamVisitor
from .initializer import Initializer
from .amp import AMPPolicy, NoAMP


# ──────────────────────────────────────────────────────────────────────
# I.2.6 — `_node_in_dims_from_ladder` builds the IN_DIMS InlineArray
# for a GraphNode from the per-input dim ladder. Sized at KIND (the
# node's effective arity — KIND=0 for InputSlot gives an empty array;
# compute nodes get sized at Op.ARITY). Used as the default for
# `GraphNode.IN_DIMS` so existing wrappers (Node, ExternalNode,
# InputSlot) get IN_DIMS for free without changing their declarations.
# ──────────────────────────────────────────────────────────────────────


def _node_in_dims_from_ladder[
    KIND: Int, D0: Int, D1: Int, D2: Int, D3: Int,
]() -> InlineArray[Int, KIND]:
    var d = InlineArray[Int, KIND](fill=0)
    comptime if KIND >= 1:
        d[0] = D0
    comptime if KIND >= 2:
        d[1] = D1
    comptime if KIND >= 3:
        d[2] = D2
    comptime if KIND >= 4:
        d[3] = D3
    return d


trait GraphNode(Defaultable & Movable & ImplicitlyDestructible):
    comptime NAME: StaticString
    comptime IN0_NAME: StaticString
    comptime IN1_NAME: StaticString = ""   # default — InputSlot / unary inherit
    comptime IN2_NAME: StaticString = ""   # default — < ternary inherit
    comptime IN3_NAME: StaticString = ""   # default — < quaternary inherit
    comptime IN0_DIM: Int
    comptime IN1_DIM: Int = 0              # default — InputSlot / unary inherit
    comptime IN2_DIM: Int = 0              # default — < ternary inherit
    comptime IN3_DIM: Int = 0              # default — < quaternary inherit
    comptime OUT_DIM: Int
    comptime KIND: Int  # 0 = input slot, 1 = unary, 2 = binary, 3 = ternary, 4 = quaternary
    # I.2.6 variadic per-input dim accessor. Default derives from the
    # ladder so InputSlot / Node / ExternalNode get IN_DIMS for free.
    # Sized at KIND (InputSlots get empty array; ARITY-N nodes get N).
    comptime IN_DIMS: InlineArray[Int, Self.KIND] = (
        _node_in_dims_from_ladder[
            Self.KIND, Self.IN0_DIM, Self.IN1_DIM, Self.IN2_DIM, Self.IN3_DIM,
        ]()
    )

    # Set the externally-supplied input pointer for InputSlot (KIND=0).
    # Unary/binary nodes implement this as a no-op — they don't consume
    # an external pointer (their input comes from a predecessor node).
    def set_input_via(
        mut self,
        ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        pass

    # Bind an externally-owned Module instance to an ExternalNode. The
    # pointer is type-erased to `UnsafePointer[Scalar[DT]]` at the trait
    # surface; the external-node implementations `rebind` it back to
    # the typed pointer at the dispatch site (Self.M is known there).
    # InputSlot / Node inherit the no-op default — they own their op
    # (or have none, for InputSlot).
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

    def grad_in2_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Default: null. Ternary+ nodes override to return their
        grad_in2_buf pointer."""
        return UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0)

    def grad_in3_ptr_via(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Default: null. Quaternary nodes override to return their
        grad_in3_buf pointer."""
        return UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0)

    def forward_via[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        in0_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        in1_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        in2_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin] = UnsafePointer[
            Scalar[DT], MutAnyOrigin
        ](unsafe_from_address=0),
        in3_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin] = UnsafePointer[
            Scalar[DT], MutAnyOrigin
        ](unsafe_from_address=0),
    ) raises:
        ...

    def vjp_via[
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
        no-op for InputSlot (no `.op` field). Node overrides to call
        `self.op.set_attr[ATTR](value)`."""
        pass

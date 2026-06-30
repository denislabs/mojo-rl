"""GraphDecl + InputSlot / Node / ExternalNode — name-based node decls for the
storage `ComputeGraph`.

This restores the legacy ComputeGraph DX (edges wired by NAME, baked into the
node declarations) on top of the storage-passing internals. Instead of a runtime
`List[List[Int]]` edge list hand-built per block, the graph is declared as:

```

comptime Graph = ComputeGraph[
    InputSlot["s", OBS],                            # external input
    ExternalNode["actor", ACTOR, "s"],              # supplied at forward
    Node["rsample", RSample[ACT], "actor"],         # owned node
    Node["action", Slice[...], "rsample"],
    Node["concat", Concat2[OBS, ACT], "s", "action"],
    ExternalNode["q1", CRITIC, "concat"],
    ...
]
```

Each node carries its NAME and its predecessors' NAMES; the graph resolves the
pool-slot indices at COMPILE TIME (`ComputeGraph._slot_of[name]()`). No runtime
edge list, no index bookkeeping.

The three decls (vs the legacy five) are MUCH thinner than `nn/combinators/
graph_nodes.mojo`: they own NO buffers and NO pointer accessors — every
activation/grad slab lives in the graph's owning `TensorPack` pool (the storage
§B0 design). A decl is pure comptime metadata + (for `Node`) one owned `Op`.

  - `InputSlot[NAME, DIM]`   KIND=0, ARITY=0. A named external input. No compute;
    its pool slot is seeded by `graph.set_input[NAME, B](tensor)` (a COPY, not a
    cached pointer). Its grad slot accumulates the input-gradient, read back via
    `graph.grad_input[NAME]()`.
  - `Node[NAME, Op, *IN_NAMES]`   KIND=1. Owns `Op: Module`; delegates every
    `Module` method to it. `IN_NAMES` (one per `Op.ARITY` input) name the
    predecessor slots.
  - `ExternalNode[NAME, M, *IN_NAMES]`   KIND=2, `IsExternal`. Owns NOTHING (like
    the old `ExternalRef`): the module is threaded into `forward`/`vjp` as a
    tracked `mut *externals` ref, in node order. Its own `forward`/`vjp` raise —
    the graph dispatches the slot to the threaded external. This is the
    load-bearing GPU fix (a stored wildcard pointer disables arg-exclusivity and
    miscompiles the delegated matmul — see external_ref.mojo).

All three conform to BOTH `Module` (so `ComputeGraph` calls forward/vjp/walkers
on the concrete `children[i]`) and the standalone `GraphDecl` bound (graph
identity + length-erased `IN_DIMS_L`/`IN_NAMES_L`). `GraphDecl` no longer
inherits `Module` — under Mojo 1.0 a `*DECLS: GraphDecl` pack with heterogeneous
arities cannot merge `Module`'s `InlineArray[Int, Self.ARITY]` `IN_DIMS`
("conflicting types"); the `_L` `List` members are length-erased so the pack
unifies. The graph branches on `KIND` to skip inputs / route externals.
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn.constants import DT
from ..core.module import Module
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from ..core.graph_visitor import DisplayStep


trait IsExternal:
    """Marker: a `ComputeGraph` node (`ExternalNode`) whose module is supplied
    at forward time (threaded via `mut *externals`), NOT owned by the graph. The
    graph dispatches the slot to the matching threaded external instead of
    calling the node's own forward/vjp.

    WHY threaded, not stored (the load-bearing GPU fix): a struct field of type
    `Pointer[M, MutAnyOrigin]` carries a WILDCARD origin, which disables
    argument-exclusivity enforcement while the field is live. On GPU that lets
    the delegated module's matmul mis-bind its kernel buffers after any
    intervening matmul, producing structured garbage on the 2nd+ forward (owned
    by-value nodes immune; CPU immune). See
    docs/BUG_REPORT_gpu_matmul_wildcard_pointer_miscompile.md."""

    pass


def names_to_inline_array[
    N: Int, *ITEMS: StaticString
]() -> InlineArray[StaticString, N]:
    """Build an `InlineArray[StaticString, N]` from a comptime variadic of
    `StaticString` — the predecessor-name list for `Node`/`ExternalNode`. N=0
    (InputSlot) yields an empty array. Uncapped."""
    var d = InlineArray[StaticString, N](fill=StaticString(""))
    comptime for k in range(N):
        d[k] = ITEMS[k]
    return d


def names_to_list[N: Int, *ITEMS: StaticString]() -> List[StaticString]:
    """`*ITEMS` → `List[StaticString]`. The uniform-typed `IN_NAMES_L` the
    `GraphDecl` bound exposes (a `List`'s type does NOT encode its length, so
    a heterogeneous-arity `*DECLS: GraphDecl` pack unifies cleanly — unlike an
    `InlineArray[..., Self.ARITY]`, which Mojo 1.0 refuses to merge across the
    pack with "conflicting types")."""
    var d = List[StaticString]()
    comptime for k in range(N):
        d.append(ITEMS[k])
    return d^


def in_dims_to_list[ARITY: Int](arr: InlineArray[Int, ARITY]) -> List[Int]:
    """`InlineArray[Int, ARITY]` (an op's `Module.IN_DIMS`) → `List[Int]` — the
    uniform-typed `IN_DIMS_L` the `GraphDecl` bound exposes (see
    `names_to_list`)."""
    var d = List[Int]()
    comptime for k in range(ARITY):
        d.append(arr[k])
    return d^


trait GraphDecl(Defaultable & Movable & ImplicitlyDeletable):
    """A `ComputeGraph` node declaration's graph-identity surface.

    `NAME` is unique within the graph; `KIND` is 0 (input) / 1 (owned) / 2
    (external); `IN_NAMES_L[k]` names the predecessor feeding input slot `k`.

    NOTE (Mojo 1.0): this trait deliberately does NOT inherit `Module`, and it
    exposes the per-input metadata as length-erased `List`s (`IN_DIMS_L`,
    `IN_NAMES_L`) rather than `InlineArray[..., Self.ARITY]`. A variadic
    `*DECLS: GraphDecl` pack with heterogeneous arities would otherwise fail to
    merge the `Self.ARITY`-typed associated constants ("trait composition has
    conflicting types for IN_DIMS"). The concrete decls still conform to
    `Module` independently (so `ComputeGraph` calls forward/vjp/walkers on the
    concrete `children[i]`); the graph only reaches this trait's surface
    type-level (`Self.DECLS[i].<member>`)."""

    comptime NAME: StaticString
    comptime KIND: Int
    comptime ARITY: Int
    comptime OUT_DIM: Int
    comptime ACT_DT: DType = DT
    comptime IN_DIMS_L: List[Int]
    comptime IN_NAMES_L: List[StaticString]

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Construct + initialize the decl (delegates to the wrapped op's
        `Module.make` for `Node`; trivial for input/external slots)."""
        ...

    # ── Module method surface ────────────────────────────────────────────
    # GraphDecl no longer inherits `Module` (the parametric `IN_DIMS` field
    # breaks heterogeneous-arity pack merging — see the trait docstring), but
    # `ComputeGraph` still calls these on the concrete `children[i]` through the
    # `*DECLS: GraphDecl` bound. They mirror `Module`'s signatures exactly (none
    # reference the conflicting `InlineArray[Int, Self.ARITY]` — only `Self.ARITY`
    # as a plain `Int` and `Self.ACT_DT`); the concrete decls satisfy them via
    # their independent `Module` conformance.
    def forward[
        target: StaticString, B: Int, o: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ...

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[Self.ARITY, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ...

    # Walker / attr surface — DEFAULT no-ops (GraphDecl is standalone, so it
    # supplies these defaults that `Module` used to provide). Param/state-less
    # decls (`InputSlot`, `ExternalNode`) inherit the no-op; `Node` overrides
    # all of them to recurse into its wrapped owned op.
    def for_each_param[target: StaticString, V: ParamVisitor](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        pass

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        pass

    def for_each_state[target: StaticString, V: ParamVisitor](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        pass

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        pass

    def set_attr_buf[ATTR: StaticString](mut self, buf: DeviceBuffer[DT]):
        pass

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        pass

    @staticmethod
    def display_label_via() -> String:
        """The wrapped op's display label (for `ComputeGraph.describe`) —
        `_via` because the decl's own `Module.display_label` would report the
        wrapper, not the op."""
        ...

    @staticmethod
    def display_steps_via() -> List[DisplayStep]:
        """The wrapped op's inner display steps (containers expand)."""
        ...


# ──────────────────────────────────────────────────────────────────────
# InputSlot — a named external input (KIND=0, ARITY=0). No compute; its
# pool slot is seeded by the graph's `set_input[NAME, B]`. forward/vjp
# are never called by the graph (KIND==0 is skipped in both walks); they
# raise so a mis-wire surfaces. Walkers inherit the Module no-op default
# (no Param fields).
# ──────────────────────────────────────────────────────────────────────


struct InputSlot[slot_name: StaticString, DIM_: Int, ADT: DType = DT](
    GraphDecl
):
    comptime NAME = Self.slot_name
    comptime KIND = 0
    comptime ARITY = 0
    comptime IN_DIMS = InlineArray[Int, 0]()
    comptime IN_NAMES = InlineArray[StaticString, 0]()
    comptime IN_DIMS_L = List[Int]()
    comptime IN_NAMES_L = List[StaticString]()
    comptime OUT_DIM = Self.DIM_
    # No wrapped op — the input slot carries no activation dtype of its own.
    # ADT (default DT) lets a bf16 graph set the slot's flow dtype to match the
    # nodes' ACT_DT (ComputeGraph asserts all decls share one ACT_DT). forward/
    # vjp never run (KIND==0 is skipped).
    comptime ACT_DT = Self.ADT

    @staticmethod
    def display_label_via() -> String:
        return String("input")

    @staticmethod
    def display_steps_via() -> List[DisplayStep]:
        return List[DisplayStep]()

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self()

    def forward[
        target: StaticString,
        B: Int,
        o: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        raise Error(
            "InputSlot.forward called — the graph seeds input slots via"
            " set_input and skips KIND==0 in the forward walk. Wiring bug."
        )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[Self.ARITY, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        raise Error(
            "InputSlot.vjp called — input grad lives in the graph's grad pool"
            " (read via grad_input[NAME]); KIND==0 is skipped. Wiring bug."
        )


# ──────────────────────────────────────────────────────────────────────
# Node — owns an `Op: Module` of any arity; delegates the whole Module
# surface to it. `*IN_NAMES` (one per Op.ARITY) name the predecessors.
# ──────────────────────────────────────────────────────────────────────


struct Node[
    node_name: StaticString,
    Op: Module,
    *in_names: StaticString,
](GraphDecl):
    comptime NAME = Self.node_name
    comptime KIND = 1
    comptime ARITY = Self.Op.ARITY
    comptime IN_DIMS = Self.Op.IN_DIMS
    comptime OUT_DIM = Self.Op.OUT_DIM
    comptime IN_NAMES = names_to_inline_array[Self.ARITY, *Self.in_names]()
    comptime IN_DIMS_L = in_dims_to_list[Self.Op.ARITY](Self.Op.IN_DIMS)
    comptime IN_NAMES_L = names_to_list[Self.ARITY, *Self.in_names]()
    # Activation dtype is the owned op's.
    comptime ACT_DT = Self.Op.ACT_DT

    @staticmethod
    def display_label_via() -> String:
        return Self.Op.display_label()

    @staticmethod
    def display_steps_via() -> List[DisplayStep]:
        return Self.Op.display_steps()

    var op: Self.Op

    def __init__(out self):
        comptime assert (
            Self.in_names.size == Self.Op.ARITY
        ), "Node: number of in_names must match Op.ARITY"
        self.op = Self.Op()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var n = Self()
        n.op = Self.Op.make[target, INIT](ctx)
        return n^

    def forward[
        target: StaticString,
        B: Int,
        o: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        # Self.ARITY/ACT_DT == Op's (definitionally), distinct to the checker —
        # rebind the whole pack + the mut output to the op's child types.
        comptime ci = Self.Op.ACT_DT
        comptime cn = Self.Op.ARITY
        self.op.forward[target, B, o, POLICY=POLICY](
            rebind[TensorRefs[cn, o, ci]](inputs),
            rebind[TensorImpl[ci]](out),
            ctx,
        )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[Self.ARITY, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime ci = Self.Op.ACT_DT
        comptime cn = Self.Op.ARITY
        self.op.vjp[target, B, ofi, ogi, POLICY=POLICY](
            rebind[TensorRefs[cn, ofi, ci]](forward_input),
            rebind[TensorImpl[ci]](grad_output),
            rebind[TensorRefs[cn, ogi, ci]](grad_inputs),
            ctx,
        )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        self.op.for_each_param[target](visitor, ctx, prefix)

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        self.op.for_each_state[target](visitor, ctx, prefix)

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.op.zero_grad[target](ctx)

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT], ctx: Optional[DeviceContext]
    ) raises:
        self.op.polyak_from[target](src.op, tau, ctx)

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        self.op.set_attr[ATTR](value)

    def set_attr_buf[
        ATTR: StaticString
    ](mut self, buf: DeviceBuffer[DT]):
        self.op.set_attr_buf[ATTR](buf)


# ──────────────────────────────────────────────────────────────────────
# ExternalNode — a named graph slot whose module is supplied at forward
# time (threaded as a tracked `mut *externals` ref, in node order). Owns
# NOTHING; carries M's metadata for pool sizing + edge typing. forward/
# vjp raise — the graph dispatches the slot to the threaded external.
# Walkers are the Module no-op default (the real owner walks its params).
# Subsumes the old `ExternalRef` and adds NAME + predecessor names.
# ──────────────────────────────────────────────────────────────────────


struct ExternalNode[
    node_name: StaticString,
    M: Module,
    *in_names: StaticString,
](GraphDecl, IsExternal):
    comptime NAME = Self.node_name
    comptime KIND = 2
    comptime ARITY = Self.M.ARITY
    comptime IN_DIMS = Self.M.IN_DIMS
    comptime OUT_DIM = Self.M.OUT_DIM
    comptime IN_NAMES = names_to_inline_array[Self.ARITY, *Self.in_names]()
    comptime IN_DIMS_L = in_dims_to_list[Self.M.ARITY](Self.M.IN_DIMS)
    comptime IN_NAMES_L = names_to_list[Self.ARITY, *Self.in_names]()
    # Activation dtype mirrors the threaded module's (for pool sizing / edge
    # typing); forward/vjp raise — the graph dispatches to the external.
    comptime ACT_DT = Self.M.ACT_DT

    @staticmethod
    def display_label_via() -> String:
        return Self.M.display_label()

    @staticmethod
    def display_steps_via() -> List[DisplayStep]:
        return Self.M.display_steps()

    def __init__(out self):
        comptime assert (
            Self.in_names.size == Self.M.ARITY
        ), "ExternalNode: number of in_names must match M.ARITY"

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self()

    def forward[
        target: StaticString,
        B: Int,
        o: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        raise Error(
            "ExternalNode.forward called directly — the graph must dispatch"
            " this slot to a threaded external (mut *externals). Wiring bug."
        )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[Self.ARITY, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        raise Error(
            "ExternalNode.vjp called directly — the graph must dispatch this"
            " slot to a threaded external (mut *externals). Wiring bug."
        )

"""ExternalRef[M] — a comptime MARKER for an externally-owned graph node.

The storage-clean replacement for the legacy `ExternalNode`/`set_external`
mechanism. Unlike the legacy node (and unlike an earlier pointer-storing draft),
`ExternalRef` stores **nothing** — no module, no pointer. It is a pure
compile-time placeholder inside `ComputeGraph[NUM_IN, *NODES]` that contributes
its `ARITY` / `IN_DIMS` / `OUT_DIM` (for pool sizing + edge typing) and marks the
slot as "supplied at forward time".

WHY no stored pointer (the load-bearing reason): a struct field of type
`Pointer[M, MutAnyOrigin]` carries a WILDCARD origin, which — per the Mojo
lifetimes manual — disables argument-exclusivity enforcement for as long as the
field is live (the whole graph). On GPU that lets the delegated module's matmul
mis-bind its kernel buffers after any intervening matmul, producing
structured-garbage on the 2nd+ forward (owned by-value nodes are immune; CPU is
immune). See docs/BUG_REPORT_gpu_matmul_wildcard_pointer_miscompile.md.

The fix: the trainer (which owns the actor/critics) threads them as TRACKED `mut`
ref arguments straight into `ComputeGraph.forward`/`vjp` (`mut *externals`). The
graph dispatches each `ExternalRef` slot to the matching external by order — so
the matmul runs with a tracked origin, exactly like an owned node. `IsExternal`
is the marker trait the graph filters on (`conforms_to`).

Because the graph NEVER calls this marker's own `forward`/`vjp` (it dispatches to
the threaded external instead), those bodies raise — calling them is a wiring
bug. The walkers are no-ops (the external module's real owner walks its params).
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from ..core.module import Module
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


trait IsExternal:
    """Marker: a `ComputeGraph` node whose module is supplied at forward time
    (threaded via `mut *externals`), NOT owned by the graph. The graph filters
    on this via `conforms_to` and dispatches the slot to the matching external
    instead of calling the node's own forward/vjp."""

    pass


struct ExternalRef[M: Module](Module, IsExternal):
    comptime ARITY: Int = Self.M.ARITY
    comptime IN_DIMS = Self.M.IN_DIMS
    comptime OUT_DIM = Self.M.OUT_DIM

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        # Pure marker — nothing to allocate; the real module is threaded into
        # ComputeGraph.forward/vjp by the trainer at call time.
        return Self()

    def forward[
        target: StaticString,
        B: Int,
        o: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        raise Error(
            "ExternalRef.forward called directly — the graph must dispatch this"
            " slot to a threaded external (mut *externals). This is a wiring"
            " bug."
        )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        raise Error(
            "ExternalRef.vjp called directly — the graph must dispatch this"
            " slot to a threaded external (mut *externals). This is a wiring"
            " bug."
        )

    # ── Walkers: NO-OP — the bound module is owned + walked by its real owner.
    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        pass

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](
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

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT], ctx: Optional[DeviceContext]
    ) raises:
        pass

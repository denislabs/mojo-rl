"""Param[NAME, DECAY, SIZE] — single trainable tensor + its gradient.

Bundles the (`value`, `grad`, `value_dev`, `grad_dev`) field cluster that
every parameterised nn2 leaf currently declares as four separate fields
into one composable field. Combined with `core/walkers.mojo`, eliminates
the per-leaf `for_each_param` body.

A leaf that previously declared
    var weight: List[Scalar[DT]]
    var bias:   List[Scalar[DT]]
    var grad_w: List[Scalar[DT]]
    var grad_b: List[Scalar[DT]]
    var weight_dev: Optional[DeviceBuffer[DT]]
    var bias_dev:   Optional[DeviceBuffer[DT]]
    var grad_w_dev: Optional[DeviceBuffer[DT]]
    var grad_b_dev: Optional[DeviceBuffer[DT]]

now declares
    var weight: Param["weight", True,  Self.IN * Self.OUT]
    var bias:   Param["bias",   False, Self.OUT]

Reflection walks the leaf's fields, picks the `Param`-typed ones (filtered
by `conforms_to(_, IsParam)`), and dispatches the visitor. Each
`(NAME, DECAY, SIZE)` triple is a distinct type so reflection sees them
distinctly.

The wrapper handles CPU + GPU dual storage symmetrically; the leaf passes
`target` through to `Param.visit_with` / `Param.zero_grad` etc., and each
helper `comptime if target == "cpu"` branches.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from ..constants import DT
from .param_visitor import ParamVisitor


# ──────────────────────────────────────────────────────────────────────
# IsParam — marker trait so reflection can filter Param-typed fields.
# The minimal surface is `visit_with`: every Param-aware helper goes
# through `core/walkers.mojo` which dispatches via this trait.
# ──────────────────────────────────────────────────────────────────────


trait IsParam(Movable & ImplicitlyDestructible):
    """Marker — a field-type that the param-walker should visit."""

    def param_name(self) -> StaticString:
        ...

    def param_decay(self) -> Bool:
        ...

    def visit_with[V: ParamVisitor, target: StaticString](
        mut self, full_name: String, mut visitor: V,
    ) raises:
        ...

    def zero_grad_with[target: StaticString](mut self) raises:
        ...


# ──────────────────────────────────────────────────────────────────────
# Param[NAME, APPLY_DECAY, SIZE] — flat 1D trainable tensor + gradient.
#
# Each (NAME, APPLY_DECAY, SIZE) triple is a distinct type. Holds CPU
# storage (`List`) AND GPU storage (`Optional[DeviceBuffer]`) — only the
# matching set is populated; selection is via the owning leaf's
# `TargetStorage.target_tag`. The `Param` wrapper itself doesn't carry
# a tag (kept stateless to keep field count low); helpers take
# `[target]` from the call site.
# ──────────────────────────────────────────────────────────────────────


struct Param[NAME: StaticString, APPLY_DECAY: Bool, SIZE: Int](IsParam):
    var value: List[Scalar[DT]]
    var grad: List[Scalar[DT]]
    var value_dev: Optional[DeviceBuffer[DT]]
    var grad_dev: Optional[DeviceBuffer[DT]]

    def __init__(out self):
        self.value = List[Scalar[DT]]()
        self.grad = List[Scalar[DT]]()
        self.value_dev = None
        self.grad_dev = None

    # ----- Factories -------------------------------------------------------

    @staticmethod
    def make_cpu() raises -> Self:
        """CPU param — allocate fp32 storage, zero-fill grad."""
        var p = Self()
        p.value = List[Scalar[DT]](length=Self.SIZE, fill=0.0)
        p.grad  = List[Scalar[DT]](length=Self.SIZE, fill=0.0)
        return p^

    @staticmethod
    def make_gpu(ctx: DeviceContext) raises -> Self:
        """GPU param — allocate device buffers, zero-fill grad."""
        var p = Self()
        var v_dev = ctx.enqueue_create_buffer[DT](Self.SIZE)
        var g_dev = ctx.enqueue_create_buffer[DT](Self.SIZE)
        g_dev.enqueue_fill(0.0)
        p.value_dev = v_dev^
        p.grad_dev  = g_dev^
        return p^

    # ----- Pointer accessors (used by the owning leaf's INIT + kernels) ---

    def value_unsafe_ptr_cpu(
        ref self,
    ) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.value.unsafe_ptr()
        )

    def grad_unsafe_ptr_cpu(
        ref self,
    ) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.grad.unsafe_ptr()
        )

    # NOTE: Shaped TileTensor accessors (e.g. `weight_tile[IN, OUT]()`)
    # are intentionally not provided. Mojo nightly rejects typed-return
    # signatures that compute `row_major[Self.SIZE]()` in the return-
    # type position. The owning leaf builds TileTensors inline from
    # `param.value` (CPU) or `param.value_dev.value()` (GPU); the
    # leaf already knows the correct (IN, OUT) shape at its call site.

    # ----- IsParam interface ----------------------------------------------

    def param_name(self) -> StaticString:
        return Self.NAME

    def param_decay(self) -> Bool:
        return Self.APPLY_DECAY

    def visit_with[V: ParamVisitor, target: StaticString](
        mut self, full_name: String, mut visitor: V,
    ) raises:
        """Dispatch the visitor with 1D [SIZE] TileTensors over this
        Param's storage. Mirrors the existing per-leaf
        `visitor.visit(name, w_tile, gw_tile, n, decay)` shape."""
        comptime if target == "cpu":
            var v_tt = TileTensor(self.value, row_major[Self.SIZE]())
            var g_tt = TileTensor(self.grad,  row_major[Self.SIZE]())
            visitor.visit(
                full_name, v_tt, g_tt, Self.SIZE, Self.APPLY_DECAY,
            )
        else:
            var v_tt = TileTensor(
                self.value_dev.value(), row_major[Self.SIZE](),
            )
            var g_tt = TileTensor(
                self.grad_dev.value(),  row_major[Self.SIZE](),
            )
            visitor.visit(
                full_name, v_tt, g_tt, Self.SIZE, Self.APPLY_DECAY,
            )

    def zero_grad_with[target: StaticString](mut self) raises:
        """Zero the gradient buffer on the active target."""
        comptime if target == "cpu":
            var g_ptr = self.grad.unsafe_ptr()
            for k in range(Self.SIZE):
                g_ptr[k] = Scalar[DT](0.0)
        else:
            self.grad_dev.value().enqueue_fill(0.0)

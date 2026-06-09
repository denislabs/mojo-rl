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
from .saveable import Saveable
from .tensor import Tensor


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


struct Param[NAME: StaticString, APPLY_DECAY: Bool, SIZE: Int](IsParam, Saveable):
    """Two `Tensor`s — value + grad — sharing the unified storage core (S5).
    External surface (`val`/`grd` fields + the accessor methods below)
    matches what leaves use: `p.val.dev` (device buffer), `p.val.cpu`
    (host List), `value_unsafe_ptr_cpu()` / `grad_unsafe_ptr_cpu()`."""
    var val: Tensor[Self.NAME, Self.SIZE]
    var grd: Tensor[Self.NAME, Self.SIZE]

    def __init__(out self):
        self.val = Tensor[Self.NAME, Self.SIZE]()
        self.grd = Tensor[Self.NAME, Self.SIZE]()

    # ----- Factories -------------------------------------------------------

    @staticmethod
    def make_cpu() raises -> Self:
        """CPU param — allocate fp32 storage, zero-fill value + grad."""
        var p = Self()
        p.val = Tensor[Self.NAME, Self.SIZE].make_cpu()
        p.grd = Tensor[Self.NAME, Self.SIZE].make_cpu()
        return p^

    @staticmethod
    def make_gpu(ctx: DeviceContext) raises -> Self:
        """GPU param — allocate device buffers, zero-fill grad."""
        var p = Self()
        p.val = Tensor[Self.NAME, Self.SIZE].make_gpu(ctx)
        p.grd = Tensor[Self.NAME, Self.SIZE].make_gpu(ctx)
        p.grd.dev.value().enqueue_fill(0.0)
        return p^

    # ----- Pointer accessors (used by the owning leaf's INIT + kernels) ---

    def value_unsafe_ptr_cpu(
        ref self,
    ) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self.val.cpu_ptr()

    def grad_unsafe_ptr_cpu(
        ref self,
    ) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self.grd.cpu_ptr()

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
            var v_tt = TileTensor(self.val.cpu, row_major[Self.SIZE]())
            var g_tt = TileTensor(self.grd.cpu, row_major[Self.SIZE]())
            visitor.visit(
                full_name, v_tt, g_tt, Self.SIZE, Self.APPLY_DECAY,
            )
        else:
            var v_tt = TileTensor(
                self.val.dev.value(), row_major[Self.SIZE](),
            )
            var g_tt = TileTensor(
                self.grd.dev.value(),  row_major[Self.SIZE](),
            )
            visitor.visit(
                full_name, v_tt, g_tt, Self.SIZE, Self.APPLY_DECAY,
            )

    def zero_grad_with[target: StaticString](mut self) raises:
        """Zero the gradient buffer on the active target."""
        comptime if target == "cpu":
            for k in range(Self.SIZE):
                self.grd.cpu[k] = Scalar[DT](0.0)
        else:
            self.grd.dev.value().enqueue_fill(0.0)

    # ----- Saveable interface (CPU only) ---------------------------------
    # Format: a section header line then SIZE value lines.
    #     <prefix>#size=<SIZE>
    #     v0
    #     v1
    #     ...
    # The header lets `load` validate that the saved Param's size matches
    # the in-memory Param's compile-time `SIZE` (catches topology drift
    # between save and load).
    #
    # GPU Params: trainer is responsible for downloading device → host
    # storage (into `self.value`) before calling `save`, and uploading
    # back after `load`. Mirrors v1's CPU-only scope.

    def save(self, mut out: String, prefix: String) raises:
        out += prefix + "#size=" + String(Self.SIZE) + "\n"
        for k in range(Self.SIZE):
            out += String(self.val.cpu[k]) + "\n"

    def load(
        mut self, lines: List[String], mut idx: Int, prefix: String,
    ) raises:
        if idx >= len(lines):
            raise Error(
                "Param.load: out of input. Expected section header `"
                + prefix + "#size=" + String(Self.SIZE)
                + "` at idx " + String(idx)
            )
        var header = lines[idx]
        var expected = prefix + "#size=" + String(Self.SIZE)
        if header != expected:
            raise Error(
                "Param.load: section-header mismatch at idx " + String(idx)
                + ". Expected `" + expected + "`, got `" + header + "`"
            )
        idx += 1
        for k in range(Self.SIZE):
            if idx >= len(lines):
                raise Error(
                    "Param.load: short read at element " + String(k)
                    + " of " + String(Self.SIZE) + " for `" + prefix + "`"
                )
            self.val.cpu[k] = Scalar[DT](atof(lines[idx]))
            idx += 1

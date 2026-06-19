"""Param[NAME, DECAY, SIZE] + ParamVisitor — storage-native params.

The user's idea: "make ParamO a Tensor." A param is just `Tensor`s, each carrying
CPU + GPU storage — so params are STORAGES, unified with activations. The
optimizer walks them via `ParamVisitor`, which receives the `Tensor`s + `target`
+ `ctx` and updates the active buffer (`.data` on CPU, `.dev` via a kernel on
GPU). No separate CPU-only param type.

A Param owns FOUR Tensors: `val` + `grd` (always) plus `m` + `v` — the per-param
optimizer moment state (Adam). `m`/`v` stay EMPTY (lazy, zero-cost) until a
stateful optimizer's `visit` calls `ensure`/`ensure_gpu` on them on its first
step; SGD ignores them. Co-locating moment state with the param (rather than a
flat side-table in the optimizer) keeps the storage design stateless at the
visitor and rides the param walk for checkpointing (Stage 4).
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from .tensor import Tensor


trait ParamVisitor(ImplicitlyDeletable):
    def visit[target: StaticString, N: Int](
        mut self,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        ...


# ──────────────────────────────────────────────────────────────────────
# IsParam — marker trait so reflection (core/walkers.mojo) can filter the
# Param-typed fields of a leaf and dispatch the visitor / zero_grad. The
# `Module.for_each_param` / `zero_grad` trait DEFAULTS reflection-walk
# these, so a Param-bearing leaf no longer needs to hand-write the walk —
# forgetting it can no longer silently skip params in the optimizer /
# checkpoint walks (the S1 footgun fix on the storage ABI).
# ──────────────────────────────────────────────────────────────────────


trait IsParam(Movable & ImplicitlyDeletable):
    """Marker — a field-type the param-walker should visit."""

    def param_name(self) -> StaticString:
        ...

    def param_decay(self) -> Bool:
        ...

    def visit_with[target: StaticString, V: ParamVisitor](
        mut self, mut visitor: V, ctx: Optional[DeviceContext]
    ) raises:
        ...

    def zero_grad[target: StaticString](
        mut self, ctx: Optional[DeviceContext]
    ) raises:
        ...


struct Param[NAME: StaticString, APPLY_DECAY: Bool, SIZE: Int](IsParam):
    var val: Tensor
    var grd: Tensor
    var m: Tensor   # optimizer 1st-moment state (Adam) — lazy, empty for SGD
    var v: Tensor   # optimizer 2nd-moment state (Adam) — lazy, empty for SGD

    def __init__(out self):
        self.val = Tensor()
        self.grd = Tensor()
        self.m = Tensor()
        self.v = Tensor()

    @staticmethod
    def make[
        target: StaticString
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Allocate val/grd. `val` keeps a CPU list for host init; the owning
        leaf's `INIT.init_weight[target]` fills it and (on GPU) uploads, which
        allocates `val.dev`. On GPU `grd.dev` is allocated + zeroed here."""
        var p = Self()
        p.val = Tensor.alloc(Self.SIZE)
        p.grd = Tensor.alloc(Self.SIZE)
        comptime if target == "gpu":
            var c = ctx.value()
            p.grd.ensure_gpu(c, Self.SIZE)
            p.grd.dev.value().enqueue_fill(Scalar[DT](0))
        return p^

    def param_name(self) -> StaticString:
        return Self.NAME

    def param_decay(self) -> Bool:
        return Self.APPLY_DECAY

    def visit_with[target: StaticString, V: ParamVisitor](
        mut self, mut visitor: V, ctx: Optional[DeviceContext]
    ) raises:
        visitor.visit[target, Self.SIZE](
            self.val, self.grd, self.m, self.v, Self.APPLY_DECAY, ctx
        )

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        comptime if target == "cpu":
            for k in range(Self.SIZE):
                self.grd.data[k] = Scalar[DT](0)
        else:
            self.grd.dev.value().enqueue_fill(Scalar[DT](0))

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

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from .tensor import Tensor


trait ParamVisitor(Deinitable):
    def visit[target: StaticString, N: Int](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        """`name` is the dotted path of this param/state in the module tree
        (e.g. "0.weight"); composed by the walker from the combinator child
        indices + the field's param_name/state_name. Empty when walked with the
        default prefix. Optimizer visitors ignore it; checkpoint / named_params
        visitors use it."""
        ...


# ──────────────────────────────────────────────────────────────────────
# ParamWalkable — the parameter surface an OPTIMIZER needs, which is
# strictly less than being a `Module`.
# ──────────────────────────────────────────────────────────────────────


trait ParamWalkable(Movable & Deinitable):
    """Everything `Adam`/`SGD` actually touch on a model: the param walk and
    the grad zero. Nothing else — not `forward`, not `ARITY`, not `make`.

    `adopt` / `step` / `zero_grad` / `clip_grads` were all bound to `Module`
    for no reason beyond convenience; each one only ever calls
    `for_each_param` (and `zero_grad`). `ComputeGraph` has both and CANNOT be
    a `Module` — it is addressed by slot NAME rather than by a `TensorRefs`
    input pack, which is why `TwoInputGraph` exists at all — so every
    graph-driven trainer (ACT, lewm) was locked out of the grouped arena and
    fell back to launching one Adam kernel per parameter. In the ACT profile
    that is 16,445 launches, **10.5% of every kernel launch in the run for
    1.0% of the kernel time** (`docs/GPU_STEP_PERF.md`).

    `Module` INHERITS this trait and satisfies both members with the defaults
    it already had, so every existing model is a `ParamWalkable` and no
    conformer changes. `ComputeGraph` declares it directly.
    """

    def for_each_param[target: StaticString, V: ParamVisitor](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        """Dispatch `visitor` at every trainable Param, in a stable order."""
        ...

    def zero_grad[target: StaticString](
        mut self, ctx: Optional[DeviceContext]
    ) raises:
        """Zero every Param's gradient buffer."""
        ...

    def for_each_state[target: StaticString, V: ParamVisitor](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        """Dispatch `visitor` at every persisted State (BatchNorm running
        statistics and friends). Not the optimizer's business, but the
        checkpoint walk runs it right after `for_each_param` and both walks
        want the same bound."""
        ...


# ──────────────────────────────────────────────────────────────────────
# IsParam — marker trait so reflection (core/walkers.mojo) can filter the
# Param-typed fields of a leaf and dispatch the visitor / zero_grad. The
# `Module.for_each_param` / `zero_grad` trait DEFAULTS reflection-walk
# these, so a Param-bearing leaf no longer needs to hand-write the walk —
# forgetting it can no longer silently skip params in the optimizer /
# checkpoint walks (the S1 footgun fix on the storage ABI).
# ──────────────────────────────────────────────────────────────────────


trait IsParam(Movable & Deinitable):
    """Marker — a field-type the param-walker should visit."""

    def param_name(self) -> StaticString:
        ...

    def param_decay(self) -> Bool:
        ...

    def visit_with[target: StaticString, V: ParamVisitor](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        full_name: String = String(""),
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
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        full_name: String = String(""),
    ) raises:
        visitor.visit[target, Self.SIZE](
            full_name, self.val, self.grd, self.m, self.v,
            Self.APPLY_DECAY, ctx,
        )

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        comptime if target == "cpu":
            for k in range(Self.SIZE):
                self.grd.data[k] = Scalar[DT](0)
        else:
            self.grd.dev.value().enqueue_fill(Scalar[DT](0))


# ──────────────────────────────────────────────────────────────────────
# ParamVersionBump — a no-op ParamVisitor that bumps each param VALUE's
# `version` (host-side, no kernel). The optimizer runs it once per `step`
# (through the existing `for_each_param` walk, so it recurses through
# combinators for free) to signal "weights changed this step". AMP leaves
# read `weight.val.version` to invalidate their cached bf16 weight — recast
# iff the version advanced. This is the bug the legacy AMP never closed: it
# cached the bf16 weight but NO caller ever invalidated it (the net trained
# against a frozen cast). Covers BOTH optimizer paths uniformly — the
# per-param walk AND the arena grouped step (which bypasses `visit`).
# ──────────────────────────────────────────────────────────────────────
struct ParamVersionBump(ParamVisitor):
    def __init__(out self):
        pass

    def visit[target: StaticString, N: Int](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        param.version += 1

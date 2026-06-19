"""Tensor[NAME, SIZE, dtype, STAGING] — the unified storage primitive (S5).

ONE dtype-generic storage cluster `{cpu: List, dev: DeviceBuffer,
hbuf: HostBuffer}` underpinning every non-trivial field cluster in nn.
The four *roles* differ only by which reflection walker visits them and
whether the size is known at compile time:

  | Role    | #Tensors | size     | optimizer-walk | checkpoint-walk |
  |---------|----------|----------|----------------|-----------------|
  | Param   | 2        | comptime | yes            | yes             |
  | State   | 1        | comptime | no             | yes             |
  | Scratch | 1        | comptime | no             | no              |
  | Cache   | 1        | runtime  | no             | no              |

- `Tensor` itself conforms `IsScratch` → a bare `Tensor`/`Scratch`/`Cache`
  field is picked up by `init_scratch_auto` and initialised on the target.
- `Scratch[NAME, SIZE, STAGING]` is a parametric alias of `Tensor`.
- `Cache[NAME, dtype]` is a `SIZE=0` `Tensor` that lazy-grows at forward
  time via `ensure_gpu` / `ensure_cpu` (folds the old `ensure_*_buffer`
  helpers + `cache_dev_n` capacity field into the type).
- `State[NAME, SIZE, dtype]` (defined here) wraps one `Tensor` + conforms
  `IsState` + `Saveable` → checkpointed but NOT optimized. Replaces the
  decay-exempt-`Param`-with-dead-grad hack for BatchNorm running stats.
- `Param` (param.mojo) holds two `Tensor`s (value + grad).

`dtype` defaults to `DT`, so existing call sites stay terse; bf16 AMP
scratch and integer RNG counters use `dtype=DType.bfloat16` /
`dtype=DType.uint32`.

`STAGING=True` keeps a CPU `List` mirror AND a pinned `HostBuffer`
alongside the device buffer for host upload/download bookkeeping.
"""

from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor

from ..constants import DT
from .module import mptr


# ──────────────────────────────────────────────────────────────────────
# Role markers. `IsScratch` (was scratch.mojo) lives here so `Tensor` can
# conform it without a circular import; scratch.mojo re-exports it.
# `IsState` is the checkpoint-only role marker (visited by the checkpoint
# walker, skipped by the optimizer walker).
# ──────────────────────────────────────────────────────────────────────


trait IsScratch(Movable & ImplicitlyDeletable):
    """Marker — a field-type the scratch-walker should initialise."""

    def scratch_name(self) -> StaticString:
        ...

    def scratch_size(self) -> Int:
        ...

    def init_with[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        ...


# ──────────────────────────────────────────────────────────────────────
# Tensor — the storage core.
# ──────────────────────────────────────────────────────────────────────


struct Tensor[
    NAME: StaticString,
    SIZE: Int,
    dtype: DType = DT,
    STAGING: Bool = False,
](IsScratch):
    var cpu: List[Scalar[Self.dtype]]
    var dev: Optional[DeviceBuffer[Self.dtype]]
    var hbuf: Optional[HostBuffer[Self.dtype]]
    var cap: Int  # current device allocation length (for dynamic Cache)

    def __init__(out self):
        self.cpu = List[Scalar[Self.dtype]]()
        self.dev = None
        self.hbuf = None
        self.cap = 0

    # ----- factories ------------------------------------------------------

    # `make_cpu` / `make_gpu` are thin wrappers over the single `init_with`
    # allocation codepath (was: each re-spelled the List / device-buffer /
    # staging block). One place to get allocation right; the factories just
    # pin the target.
    @staticmethod
    def make_cpu() raises -> Self:
        var t = Self()
        t.init_with["cpu"](None)
        return t^

    @staticmethod
    def make_gpu(ctx: DeviceContext) raises -> Self:
        var t = Self()
        t.init_with["gpu"](ctx)
        return t^

    def init_with[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        """Walker entry point (IsScratch). SIZE==0 (Cache) is a no-op here
        — the leaf lazy-allocates via `ensure_*` at forward time."""
        comptime if Self.SIZE > 0:
            comptime if target == "cpu":
                self.cpu = List[Scalar[Self.dtype]](
                    length=Self.SIZE, fill=Scalar[Self.dtype](0)
                )
            else:
                self.dev = ctx.value().enqueue_create_buffer[Self.dtype](
                    Self.SIZE
                )
                self.cap = Self.SIZE
                comptime if Self.STAGING:
                    self.cpu = List[Scalar[Self.dtype]](
                        length=Self.SIZE, fill=Scalar[Self.dtype](0)
                    )
                    self.hbuf = ctx.value().enqueue_create_host_buffer[
                        Self.dtype
                    ](Self.SIZE)

    # ----- dynamic (Cache) lazy grow -------------------------------------

    def ensure_cpu(mut self, n: Int):
        """Lazy-grow the CPU list to >= n, zero-filled. Cache role."""
        if len(self.cpu) < n:
            self.cpu = List[Scalar[Self.dtype]](
                length=n, fill=Scalar[Self.dtype](0)
            )

    def ensure_gpu(mut self, ctx: DeviceContext, n: Int) raises:
        """Lazy-(re)allocate the device buffer to >= n. Cache role —
        replaces the old `ensure_gpu_buffer` helper + `*_dev_n` field. When
        `STAGING`, a pinned host buffer of matching length is (re)allocated
        alongside it for H2D/D2H bookkeeping (folds the old `*_hbuf` field)."""
        if self.cap < n:
            self.dev = ctx.enqueue_create_buffer[Self.dtype](n)
            comptime if Self.STAGING:
                self.hbuf = ctx.enqueue_create_host_buffer[Self.dtype](n)
            self.cap = n

    # ----- IsScratch ------------------------------------------------------

    def scratch_name(self) -> StaticString:
        return Self.NAME

    def scratch_size(self) -> Int:
        return Self.SIZE

    # ----- pointer / buffer accessors ------------------------------------

    def cpu_ptr(ref self) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        return mptr(self.cpu.unsafe_ptr())

    def dev_ptr(self) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        return mptr(self.dev.value().unsafe_ptr())

    def host_ptr(self) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        """Pinned host-staging pointer (STAGING). For H2D/D2H bookkeeping."""
        return mptr(self.hbuf.value().unsafe_ptr())

    def target_ptr[
        target: StaticString
    ](self) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        comptime if target == "cpu":
            return self.cpu_ptr()
        elif target == "gpu":
            return self.dev_ptr()
        else:
            comptime assert False, "target must be 'cpu' or 'gpu'"

    # ----- shaped typed view (removes manual ptr->LayoutTensor rebuild) ---

    def lt[
        layout: Layout
    ](mut self) -> LayoutTensor[Self.dtype, layout, MutAnyOrigin]:
        """Typed device view of the buffer at the given comptime layout.
        Origin-linking ctor (no `dev_ptr()`/`mptr`/`unsafe_ptr` round-trip);
        see `lt_target` for why the `MutAnyOrigin` unifier stays."""
        return LayoutTensor[Self.dtype, layout, MutAnyOrigin](self.dev.value())

    def lt_target[
        target: StaticString, layout: Layout
    ](mut self) -> LayoutTensor[Self.dtype, layout, MutAnyOrigin]:
        """Typed view of the target buffer at the given comptime layout.

        This is the origin-LINKING constructor (`LayoutTensor(buffer)` —
        no `.unsafe_ptr()` / `mptr` round-trip), so the pointer plumbing
        is gone. The remaining `MutAnyOrigin` here is DELIBERATE and
        load-bearing, not laziness — two reasons it can't be a tracked
        `origin_of(self)`:
          1. The one method dispatches over `target`: the CPU branch
             builds from `Span(self.cpu)` (origin `origin_of(self.cpu)`)
             and the GPU branch from `ref` to `self.dev.value()` (a
             different origin). The buffer ctors INFER origin from their
             argument — you can't force a common `origin_of(self)` — so a
             single return type must use the `MutAnyOrigin` unifier.
          2. The dominant consumer is `Module.forward`/`vjp`, whose
             variadic `TensorPack` surface is `MutAnyOrigin` (§B0). A
             tracked view would need an explicit launder at every such
             call site, moving the wildcard rather than removing it.
        i.e. this is the named CPU/GPU-storage + Module boundary."""
        comptime if target == "cpu":
            return LayoutTensor[Self.dtype, layout, MutAnyOrigin](self.cpu)
        elif target == "gpu":
            return LayoutTensor[Self.dtype, layout, MutAnyOrigin](
                self.dev.value()
            )
        else:
            comptime assert False, "target must be 'cpu' or 'gpu'"

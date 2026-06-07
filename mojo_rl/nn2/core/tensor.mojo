"""Tensor[NAME, SIZE, dtype, STAGING] — the unified storage primitive (S5).

ONE dtype-generic storage cluster `{cpu: List, dev: DeviceBuffer,
hbuf: HostBuffer}` underpinning every non-trivial field cluster in nn2.
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


# ──────────────────────────────────────────────────────────────────────
# Role markers. `IsScratch` (was scratch.mojo) lives here so `Tensor` can
# conform it without a circular import; scratch.mojo re-exports it.
# `IsState` is the checkpoint-only role marker (visited by the checkpoint
# walker, skipped by the optimizer walker).
# ──────────────────────────────────────────────────────────────────────


trait IsScratch(Movable & ImplicitlyDestructible):
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

    @staticmethod
    def make_cpu() raises -> Self:
        var t = Self()
        comptime if Self.SIZE > 0:
            t.cpu = List[Scalar[Self.dtype]](
                length=Self.SIZE, fill=Scalar[Self.dtype](0)
            )
        return t^

    @staticmethod
    def make_gpu(ctx: DeviceContext) raises -> Self:
        var t = Self()
        comptime if Self.SIZE > 0:
            t.dev = ctx.enqueue_create_buffer[Self.dtype](Self.SIZE)
            t.cap = Self.SIZE
            comptime if Self.STAGING:
                t.cpu = List[Scalar[Self.dtype]](
                    length=Self.SIZE, fill=Scalar[Self.dtype](0)
                )
                t.hbuf = ctx.enqueue_create_host_buffer[Self.dtype](Self.SIZE)
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

    def ensure_cpu(mut self, n: Int) raises:
        """Lazy-grow the CPU list to >= n, zero-filled. Cache role."""
        if len(self.cpu) < n:
            self.cpu = List[Scalar[Self.dtype]](
                length=n, fill=Scalar[Self.dtype](0)
            )

    def ensure_gpu(mut self, ctx: DeviceContext, n: Int) raises:
        """Lazy-(re)allocate the device buffer to >= n. Cache role —
        replaces the old `ensure_gpu_buffer` helper + `*_dev_n` field."""
        if self.cap < n:
            self.dev = ctx.enqueue_create_buffer[Self.dtype](n)
            self.cap = n

    # ----- IsScratch ------------------------------------------------------

    def scratch_name(self) -> StaticString:
        return Self.NAME

    def scratch_size(self) -> Int:
        return Self.SIZE

    # ----- pointer / buffer accessors ------------------------------------

    def cpu_ptr(ref self) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]](
            self.cpu.unsafe_ptr()
        )

    def dev_ptr(self) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        return rebind[UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]](
            self.dev.value().unsafe_ptr()
        )

    def host_ptr(self) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        """Pinned host-staging pointer (STAGING). For H2D/D2H bookkeeping."""
        return rebind[UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]](
            self.hbuf.value().unsafe_ptr()
        )

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
    ](self) -> LayoutTensor[Self.dtype, layout, MutAnyOrigin]:
        """Typed device view of the buffer at the given comptime layout."""
        return LayoutTensor[Self.dtype, layout, MutAnyOrigin](self.dev_ptr())

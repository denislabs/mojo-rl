"""Tensor[dt] — the storage cell for the storage-passing design (CPU + GPU).

Owns a CPU `List` AND an optional GPU `DeviceBuffer`; the active one is picked
by the method's `target`. `dt` defaults to `DT` (fp32), so the surface stays
`Tensor`; AMP scratch is `Tensor[DType.bfloat16]`. The KEY idea is unchanged:
leaves/orchestrators pass `ref`/`mut Tensor` (the STORAGE), and each method
builds its typed view INTERNALLY — `TileTensor(self.data, …)` on CPU, or
`self.lt["gpu", layout]()` (a device `LayoutTensor`) on GPU. The only erasure on
the GPU path is the kernel-arg `MutAnyOrigin` (the GPU ABI).
"""

from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT


struct TensorImpl[dt: DType = DT](Defaultable & Movable & ImplicitlyDeletable):
    var data: List[Scalar[Self.dt]]
    var dev: Optional[DeviceBuffer[Self.dt]]
    var n: Int  # logical length (tracks the device buffer too)
    # Persistent pinned host staging buffer for H2D/D2H. Lazily (re)allocated
    # by `ensure_host` and REUSED across upload/download so a per-step hot loop
    # doesn't churn (and leak) pinned host buffers. `hcap` is its capacity.
    var hbuf: Optional[HostBuffer[Self.dt]]
    var hcap: Int

    def __init__(out self):
        self.data = List[Scalar[Self.dt]]()
        self.dev = None
        self.n = 0
        self.hbuf = None
        self.hcap = 0

    # ----- unified CPU/GPU allocator -------------------------------------
    @staticmethod
    def make[
        target: StaticString
    ](n: Int, ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified allocator (zero-filled, length `n`) — dispatches to
        `alloc` / `alloc_gpu`. `ctx` is ignored on CPU and required on GPU.
        Lets `[target]`-generic code allocate without a `comptime if` branch
        at every site (the CPU/GPU-path unification the leaves want)."""
        comptime if target == "cpu":
            return Self.alloc(n)
        elif target == "gpu":
            if not ctx:
                raise Error("Tensor.make[target='gpu']: ctx required")
            return Self.alloc_gpu(ctx.value(), n)
        else:
            comptime assert False, "target must be 'cpu' or 'gpu'"

    def ensure[
        target: StaticString
    ](mut self, n: Int, ctx: Optional[DeviceContext] = None) raises:
        """Unified lazy-(re)allocate to >= `n` — dispatches to the target's
        `ensure` / `ensure_gpu`. The `[target]`-generic companion to `make`."""
        comptime if target == "cpu":
            self.ensure(n)
        elif target == "gpu":
            self.ensure_gpu(ctx.value(), n)
        else:
            comptime assert False, "target must be 'cpu' or 'gpu'"

    # ----- CPU -----------------------------------------------------------
    @staticmethod
    def alloc(n: Int) raises -> Self:
        var t = Self()
        t.data = List[Scalar[Self.dt]](length=n, fill=Scalar[Self.dt](0))
        t.n = n
        return t^

    def ensure(mut self, n: Int):
        """Lazy-grow the CPU list to >= n, zero-filled."""
        if len(self.data) < n:
            self.data = List[Scalar[Self.dt]](length=n, fill=Scalar[Self.dt](0))
            self.n = n

    # ----- GPU -----------------------------------------------------------
    @staticmethod
    def alloc_gpu(ctx: DeviceContext, n: Int) raises -> Self:
        var t = Self()
        t.dev = ctx.enqueue_create_buffer[Self.dt](n)
        t.dev.value().enqueue_fill(Scalar[Self.dt](0))
        t.n = n
        return t^

    def ensure_gpu(mut self, ctx: DeviceContext, n: Int) raises:
        """Lazy-(re)allocate the device buffer to >= n."""
        if not self.dev or self.n < n:
            self.dev = ctx.enqueue_create_buffer[Self.dt](n)
            self.n = n

    def ensure_host(mut self, ctx: DeviceContext, n: Int) raises:
        """Lazy-(re)allocate the pinned host staging buffer to >= n. Reused
        across upload/download so the hot loop doesn't churn pinned buffers."""
        if not self.hbuf or self.hcap < n:
            self.hbuf = ctx.enqueue_create_host_buffer[Self.dt](n)
            self.hcap = n

    def lt[
        target: StaticString, layout: Layout
    ](mut self) -> LayoutTensor[Self.dt, layout, MutAnyOrigin]:
        """Typed device view at `layout`. Origin-linking ctor (no
        `.unsafe_ptr()`); `MutAnyOrigin` is the GPU kernel-ABI boundary."""
        comptime if target == "cpu":
            return LayoutTensor[Self.dt, layout, MutAnyOrigin](self.data)
        elif target == "gpu":
            return LayoutTensor[Self.dt, layout, MutAnyOrigin](self.dev.value())
        else:
            comptime assert False, "target must be 'cpu' or 'gpu'"

    def upload(mut self, ctx: DeviceContext) raises:
        """CPU `data` → device buffer (via the persistent host staging buffer).
        The device buffer is (re)allocated to `self.n` (original semantics —
        callers may grow `self.n` before calling); the pinned host buffer is
        REUSED across calls instead of allocated fresh each time."""
        self.dev = ctx.enqueue_create_buffer[Self.dt](self.n)
        self.ensure_host(ctx, self.n)
        var hb = self.hbuf.value()
        ctx.synchronize()
        for i in range(self.n):
            hb[i] = self.data[i]
        ctx.enqueue_copy(self.dev.value(), hb)
        ctx.synchronize()

    def download_enqueue(mut self, ctx: DeviceContext) raises:
        """Enqueue the D2H copy into the persistent host buffer WITHOUT
        synchronizing. Pair with a later `ctx.synchronize()` + per-tensor
        `download_finalize()` to batch several D2H copies behind ONE sync."""
        self.ensure_host(ctx, self.n)
        ctx.enqueue_copy(self.hbuf.value(), self.dev.value())

    def download_finalize(mut self):
        """Copy the staged host buffer into CPU `data` (call AFTER the sync)."""
        var hb = self.hbuf.value()
        if len(self.data) < self.n:
            self.data = List[Scalar[Self.dt]](
                length=self.n, fill=Scalar[Self.dt](0)
            )
        for i in range(self.n):
            self.data[i] = hb[i]

    def download(mut self, ctx: DeviceContext) raises:
        """Device buffer → CPU `data` (via the persistent host staging buffer).
        The pinned host buffer is reused across calls — the previous design
        allocated a fresh one every call, churning/leaking it in a hot loop."""
        self.download_enqueue(ctx)
        ctx.synchronize()
        self.download_finalize()


# The fp32 surface type is a concrete alias, so bare `Tensor` is never an
# unbound-parameter ambiguity; AMP scratch uses `TensorImpl[DType.bfloat16]`.
comptime Tensor = TensorImpl[DT]

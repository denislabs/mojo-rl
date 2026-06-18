"""Tensor[dt] — the storage cell for the storage-passing design (CPU + GPU).

Owns a CPU `List` AND an optional GPU `DeviceBuffer`; the active one is picked
by the method's `target`. `dt` defaults to `DT` (fp32), so the surface stays
`Tensor`; AMP scratch is `Tensor[DType.bfloat16]`. The KEY idea is unchanged:
leaves/orchestrators pass `ref`/`mut Tensor` (the STORAGE), and each method
builds its typed view INTERNALLY — `TileTensor(self.data, …)` on CPU, or
`self.lt_gpu[layout]()` (a device `LayoutTensor`) on GPU. The only erasure on
the GPU path is the kernel-arg `MutAnyOrigin` (the GPU ABI).
"""

from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT


struct TensorImpl[dt: DType = DT](Defaultable & Movable & ImplicitlyDeletable):
    var data: List[Scalar[Self.dt]]
    var dev: Optional[DeviceBuffer[Self.dt]]
    var n: Int  # logical length (tracks the device buffer too)

    def __init__(out self):
        self.data = List[Scalar[Self.dt]]()
        self.dev = None
        self.n = 0

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

    def lt_gpu[
        layout: Layout
    ](mut self) -> LayoutTensor[Self.dt, layout, MutAnyOrigin]:
        """Typed device view at `layout`. Origin-linking ctor (no
        `.unsafe_ptr()`); `MutAnyOrigin` is the GPU kernel-ABI boundary."""
        return LayoutTensor[Self.dt, layout, MutAnyOrigin](self.dev.value())

    def upload(mut self, ctx: DeviceContext) raises:
        """CPU `data` → device buffer (via a pinned host staging buffer)."""
        self.dev = ctx.enqueue_create_buffer[Self.dt](self.n)
        var hb = ctx.enqueue_create_host_buffer[Self.dt](self.n)
        ctx.synchronize()
        for i in range(self.n):
            hb[i] = self.data[i]
        ctx.enqueue_copy(self.dev.value(), hb)
        ctx.synchronize()

    def download(mut self, ctx: DeviceContext) raises:
        """Device buffer → CPU `data`."""
        var hb = ctx.enqueue_create_host_buffer[Self.dt](self.n)
        ctx.enqueue_copy(hb, self.dev.value())
        ctx.synchronize()
        if len(self.data) < self.n:
            self.data = List[Scalar[Self.dt]](length=self.n, fill=Scalar[Self.dt](0))
        for i in range(self.n):
            self.data[i] = hb[i]


# The fp32 surface type is a concrete alias, so bare `Tensor` is never an
# unbound-parameter ambiguity; AMP scratch uses `TensorImpl[DType.bfloat16]`.
comptime Tensor = TensorImpl[DT]

"""Tensor[dt] — the storage cell for the storage-passing design (CPU + GPU).

Owns a CPU `List` AND an optional GPU `DeviceBuffer`; the active one is picked
by the method's `target`. `dt` defaults to `DT` (fp32), so the surface stays
`Tensor`; AMP scratch is `Tensor[DType.bfloat16]`. The KEY idea is unchanged:
leaves/orchestrators pass `ref`/`mut Tensor` (the STORAGE), and each method
builds its typed view INTERNALLY — `TileTensor(self.data, …)` on CPU, or
`self.lt["gpu", layout]()` (a device `LayoutTensor`) on GPU. The only erasure on
the GPU path is the kernel-arg `MutAnyOrigin` (the GPU ABI).
"""

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor, RuntimeLayout

from mojo_rl.nn.constants import DT


struct TensorImpl[dt: DType = DT](Defaultable & Movable & Deinitable):
    var data: List[Scalar[Self.dt]]
    var dev: Optional[DeviceBuffer[Self.dt]]
    var n: Int  # logical length (tracks the device buffer too)
    # Persistent pinned host staging buffer for H2D/D2H. Lazily (re)allocated
    # by `ensure_host` and REUSED across upload/download so a per-step hot loop
    # doesn't churn (and leak) pinned host buffers. `hcap` is its capacity.
    var hbuf: Optional[HostBuffer[Self.dt]]
    var hcap: Int
    # Monotonic write-version of the VALUES, bumped by the optimizer once per
    # step (on the param-value tensors it updates — see `ParamVersionBump`). AMP
    # leaves read it to invalidate a cached low-precision weight copy: recast iff
    # `val.version` advanced since the last cast (so the bf16 weight is cast ONCE
    # per optimizer step, not once per forward — the Phase-1 economics fix).
    # Inert on activation tensors (never bumped, never read there).
    var version: Int

    def __init__(out self):
        self.data = List[Scalar[Self.dt]]()
        self.dev = None
        self.n = 0
        self.hbuf = None
        self.hcap = 0
        self.version = 0

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

    @staticmethod
    def view_gpu(
        ctx: DeviceContext,
        ptr: Pointer[Scalar[Self.dt], MutAnyOrigin],
        n: Int,
    ) raises -> Self:
        """BORROWING device view — wrap an EXTERNALLY-owned device buffer (e.g.
        an MCTS planner's search buffer, reached as a `LayoutTensor.ptr`) in a
        non-owning `Tensor` so it can be fed straight to `forward`/`vjp` with NO
        copy. The caller owns the memory and MUST keep it alive for the view's
        lifetime; `owning=False` means this `Tensor` never frees it (it dies as
        a plain handle, the storage outlives it via the owner).

        This is the sanctioned boundary between the storage `Module` surface and
        the raw device-buffer interop of `planners/tree_search` — the adapters
        in `deep_agents/zero/mcts_adapters*` build one of these per net input and
        output, then call `net.forward["gpu", B](TensorRefs(in_view), out_view)`.
        The net's whole forward/vjp then runs on the safe storage surface; only
        this thin wrap touches a raw pointer (irreducible GPU ABI).

        GPU-only: the CPU `data` List owns its storage and cannot alias external
        memory, so CPU adapters copy into an owned `Tensor` instead (no analog —
        like `lt_at`).

        ⚠️ A single non-owning view is fine as a kernel operand, but TWO
        simultaneous non-owning views as operands of ONE kernel miscompile on
        Metal (deterministic prefix-drop — the exclusivity/wildcard-origin class
        of the prior ExternalRef GPU bug). For the adapter's input+output case,
        prefer `copy_from_device`/`copy_to_device` (owned scratch + a small D2D
        copy at the boundary; the external buffer is only ever a copy ENDPOINT,
        never a kernel operand)."""
        var t = Self()
        t.dev = DeviceBuffer[Self.dt](ctx, ptr, n, owning=False)
        t.n = n
        return t^

    def copy_from_device(
        mut self,
        ctx: DeviceContext,
        src: Pointer[Scalar[Self.dt], MutAnyOrigin],
        n: Int,
    ) raises:
        """Device→device copy an EXTERNALLY-owned buffer INTO this Tensor's own
        device buffer (lazily allocated to >= n). The external pointer (e.g. an
        MCTS planner's `LayoutTensor.ptr`) is wrapped in a transient non-owning
        `DeviceBuffer` used ONLY as a copy endpoint — never a kernel operand —
        so the net's forward/vjp then runs entirely on owned storage. This is
        the robust adapter-boundary INPUT bridge (see `view_gpu`'s warning)."""
        self.ensure_gpu(ctx, n)
        var src_buf = DeviceBuffer[Self.dt](ctx, src, n, owning=False)
        ctx.enqueue_copy(self.dev.value(), src_buf)

    def copy_to_device(
        mut self,
        ctx: DeviceContext,
        dst: Pointer[Scalar[Self.dt], MutAnyOrigin],
        n: Int,
    ) raises:
        """Device→device copy this Tensor's own device buffer OUT to an
        EXTERNALLY-owned buffer (the planner side). Mirror of
        `copy_from_device`; the adapter-boundary OUTPUT bridge."""
        var dst_buf = DeviceBuffer[Self.dt](ctx, dst, n, owning=False)
        ctx.enqueue_copy(dst_buf, self.dev.value())

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

    def lt_dyn[
        target: StaticString, layout: Layout
    ](mut self, rl: RuntimeLayout[layout]) -> LayoutTensor[
        Self.dt, layout, MutAnyOrigin
    ]:
        """The same view, shaped at RUN TIME — `layout` carries UNKNOWN
        extents and `rl` carries the real ones.

        This is `lt`'s counterpart for physics3d's dynamic leg (assessment
        §12.4): one kernel body serves a comptime `Layout.row_major(BATCH,
        NV*NV)` from the GPU path and a `Layout.row_major[2]()` + this from
        the CPU path, because `LM: Layout` accepts both. Same origin-linking
        ctor as `lt` — no `.unsafe_ptr()`, so the borrow is not severed."""
        comptime if target == "cpu":
            return LayoutTensor[Self.dt, layout, MutAnyOrigin](self.data, rl)
        elif target == "gpu":
            return LayoutTensor[Self.dt, layout, MutAnyOrigin](
                self.dev.value(), rl
            )
        else:
            comptime assert False, "target must be 'cpu' or 'gpu'"

    def lt_at[
        target: StaticString, layout: Layout
    ](mut self, offset: Int) raises -> LayoutTensor[
        Self.dt, layout, MutAnyOrigin
    ]:
        """Typed GPU view at element `offset` into the device buffer — a
        stacked-ensemble / per-step sub-view — WITHOUT `.unsafe_ptr()`. The
        sanctioned replacement for
        `LayoutTensor[..MutAnyOrigin](buf.dev.value().unsafe_ptr() + offset)`:
        a memory-sharing `create_sub_buffer` (offset + `layout.size()`, in
        elements) feeds the same explicit-`MutAnyOrigin` DeviceBuffer ctor `lt`
        uses, so the returned static `layout` matches the kernel ABI exactly
        (callers/kernels identical to `lt`, just offset).

        GPU-only: the per-member offset views that needed raw pointers are all on
        the GPU kernel path; CPU branches index the owning `self.data` List
        directly (no ABI erasure, so no helper needed). The parent buffer owns
        the storage — the sub-buffer handle dying after this returns is safe
        (the `MutAnyOrigin` cast erases its origin; the memory outlives via the
        parent)."""
        comptime assert target == "gpu", (
            "lt_at is GPU-only (offset sub-view); CPU indexes self.data directly"
        )
        comptime sz = layout.size()
        var sub = self.dev.value().create_sub_buffer[Self.dt](offset, sz)
        return LayoutTensor[Self.dt, layout, MutAnyOrigin](sub)

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

    def upload_resident(mut self, ctx: DeviceContext) raises:
        """CPU `data[:n]` → the EXISTING device buffer WITHOUT reallocating it
        (the buffer must already exist and be sized >= `n`; lazily allocates on
        the first call only). Unlike `upload` — which recreates `self.dev` every
        call (changing the pointer) — this reuses the buffer, so a CUDA-graph
        that captured this buffer stays valid across replays: only the CONTENTS
        change. Use as the EAGER per-step input refresh under graph capture."""
        self.ensure_gpu(ctx, self.n)
        self.ensure_host(ctx, self.n)
        var hb = self.hbuf.value()
        ctx.synchronize()
        for i in range(self.n):
            hb[i] = self.data[i]
        ctx.enqueue_copy(self.dev.value(), hb)

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

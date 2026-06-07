"""BinaryElementwise[DIM, OP: BinaryElementOp] — single template for all
elementwise binary ops (Add / Sub / ElemMin / future Mul / Div / Where …).

Phase 4.5. Mirror of `Elementwise[DIM, OP]` but with two inputs +
two grad inputs. Each leaf binary collapses to a one-line alias:

    alias BinarySub[DIM: Int]     = BinaryElementwise[DIM, BinarySubOp]
    alias BinaryElemMin[DIM: Int] = BinaryElementwise[DIM, BinaryElemMinOp]

Cache strategy is chosen by `Self.OP.owns_cache`:

  - `Self.OP.owns_cache = True`  (BinaryElemMin):
        forward writes per-element carry (e.g. min mask) to an OWNED
        cache buffer (`cache: List` CPU + `cache_dev: DeviceBuffer` GPU).
        Backward reads the cache when computing grad_in0 / grad_in1.

  - `Self.OP.owns_cache = False` (BinarySub):
        no cache allocated; backward needs only grad_output. The cache
        fields still exist on the struct (Mojo nightly can't drop fields
        conditionally) but are never touched.

(Add — the elementwise N-arity counterpart — lives in `add.mojo` as a
variadic primitive, not as a `BinaryElementOp`.)

CPU paths use SIMD via `Self.OP.forward_simd[W]` / `cache_simd[W]` /
`backward_simd_x[W]` / `backward_simd_y[W]` with `CPU_SIMD_W = 8`. GPU
paths use one shared kernel per (cached / uncached) × (forward / backward)
quadrant, parameterised on `OP` via comptime.

BACKWARD-ORDER INVARIANT: binary leaves never alias either input pointer
into their grad-output, so the SAC-style "param-grad before grad_input"
invariant doesn't apply here.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from ..constants import DT, CPU_SIMD_W, TPB
from ..core import Initializer, AMPPolicy, NoAMP, Cache, TensorPack
from ..core.binary_element_op import BinaryElementOp
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import (
    require_ctx,
    TargetStorage,
    assert_tag_for,
)


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — four flavours: forward (cached / uncached) × backward
# (cached / uncached). OP supplies the per-element math via static
# methods; cache reads/writes are comptime-gated by `Self.OP.owns_cache`
# at dispatch time (the kernel selected here matches the OP's cache mode).
# ──────────────────────────────────────────────────────────────────────


def _be_forward_kernel_uncached[
    N: Int, OP: BinaryElementOp,
](
    in0: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    in1: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        var x = rebind[Scalar[DT]](in0[idx])
        var y = rebind[Scalar[DT]](in1[idx])
        output[idx] = OP.forward_scalar(x, y)


def _be_forward_kernel_cached[
    N: Int, OP: BinaryElementOp,
](
    in0: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    in1: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        var x = rebind[Scalar[DT]](in0[idx])
        var y = rebind[Scalar[DT]](in1[idx])
        output[idx] = OP.forward_scalar(x, y)
        cache[idx] = OP.cache_scalar(x, y)


def _be_backward_kernel_uncached[
    N: Int, OP: BinaryElementOp,
](
    grad_output: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    grad_in0: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    grad_in1: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        var go = rebind[Scalar[DT]](grad_output[idx])
        var zero: Scalar[DT] = 0.0   # `c` is unused for owns_cache=False ops
        grad_in0[idx] = OP.backward_scalar_x(zero, go)
        grad_in1[idx] = OP.backward_scalar_y(zero, go)


def _be_backward_kernel_cached[
    N: Int, OP: BinaryElementOp,
](
    grad_output: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    grad_in0: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    grad_in1: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        var c = rebind[Scalar[DT]](cache[idx])
        var go = rebind[Scalar[DT]](grad_output[idx])
        grad_in0[idx] = OP.backward_scalar_x(c, go)
        grad_in1[idx] = OP.backward_scalar_y(c, go)


# ──────────────────────────────────────────────────────────────────────
# BinaryElementwise[DIM, OP] — owns: TargetStorage + (when
# OP.owns_cache=True) a cache buffer. Uncached ops carry the cache
# fields anyway (Mojo nightly can't drop fields conditionally) but
# never touch them.
# ──────────────────────────────────────────────────────────────────────


struct BinaryElementwise[DIM: Int, OP: BinaryElementOp](Module):
    comptime ARITY: Int = 2
    comptime IN_DIMS = InlineArray[Int, 2](fill=Self.DIM)
    comptime IN0_DIM = Self.DIM
    comptime OUT_DIM = Self.DIM

    @staticmethod
    def display_label() -> String:
        return Self.OP.display_label()

    var ts: TargetStorage

    # Cache (used only when Self.OP.owns_cache = True). S5 dynamic Cache
    # role — one Tensor lazy-grown at forward; was List + Optional
    # DeviceBuffer + capacity-Int.
    var cache: Cache["be_cache"]

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()
        self.cache = Cache["be_cache"]()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. INIT ignored (BinaryElementwise is
        parameterless) but accepted for uniformity so combinators that
        call `child.make[target, INIT]` recurse cleanly."""
        comptime assert target == "cpu" or target == "gpu", (
            "BinaryElementwise: target must be 'cpu' or 'gpu'"
        )
        var e = Self()
        comptime if target == "cpu":
            e.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["BinaryElementwise.make[target='gpu']"](ctx)
            e.ts = TargetStorage.make_gpu(ctx_v)
            # cache is lazy (S5 Cache) — grown at forward via ensure_gpu.
        return e^

    # ------------------------------------------------------------------
    # Forward — SIMD CPU loop or one GPU kernel launch, comptime-branched
    # on Self.OP.owns_cache.
    # ------------------------------------------------------------------

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorPack[Self.ARITY],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["BinaryElementwise", target](self.ts.target_tag)

        comptime if target == "cpu":
            comptime N = BATCH * Self.DIM
            # is centralized in `of()`, not re-spelled per input here.
            var i0_p = inputs.ptr[0]()
            var i1_p = inputs.ptr[1]()
            var o_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)

            comptime if Self.OP.owns_cache:
                self.cache.ensure_cpu(N)
                var c_p = self.cache.cpu_ptr()
                var k = 0
                while k + CPU_SIMD_W <= N:
                    var xv = i0_p.load[width=CPU_SIMD_W](k)
                    var yv = i1_p.load[width=CPU_SIMD_W](k)
                    o_p.store(k, Self.OP.forward_simd[CPU_SIMD_W](xv, yv))
                    c_p.store(k, Self.OP.cache_simd[CPU_SIMD_W](xv, yv))
                    k += CPU_SIMD_W
                while k < N:
                    var x = i0_p[k]
                    var y = i1_p[k]
                    o_p[k] = Self.OP.forward_scalar(x, y)
                    c_p[k] = Self.OP.cache_scalar(x, y)
                    k += 1
            else:
                var k = 0
                while k + CPU_SIMD_W <= N:
                    var xv = i0_p.load[width=CPU_SIMD_W](k)
                    var yv = i1_p.load[width=CPU_SIMD_W](k)
                    o_p.store(k, Self.OP.forward_simd[CPU_SIMD_W](xv, yv))
                    k += CPU_SIMD_W
                while k < N:
                    o_p[k] = Self.OP.forward_scalar(i0_p[k], i1_p[k])
                    k += 1
        else:
            comptime N = BATCH * Self.DIM
            comptime layout = Layout.row_major(N)
            # the per-input `rebind` + manual `LayoutTensor` rebuild.
            var i0_lt = inputs.lt[0, layout]()
            var i1_lt = inputs.lt[1, layout]()
            var o_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
            var o_lt = LayoutTensor[DT, layout, MutAnyOrigin](o_p)
            comptime n_blocks = (N + TPB - 1) // TPB

            comptime if Self.OP.owns_cache:
                self.cache.ensure_gpu(self.ts.ctx.value(), N)
                var c_lt = LayoutTensor[DT, layout, MutAnyOrigin](
                    self.cache.dev.value()
                )
                comptime kernel = _be_forward_kernel_cached[N, Self.OP]
                self.ts.ctx.value().enqueue_function[kernel](
                    i0_lt, i1_lt, o_lt, c_lt,
                    grid_dim=n_blocks, block_dim=TPB,
                )
            else:
                comptime kernel = _be_forward_kernel_uncached[N, Self.OP]
                self.ts.ctx.value().enqueue_function[kernel](
                    i0_lt, i1_lt, o_lt,
                    grid_dim=n_blocks, block_dim=TPB,
                )

    # ------------------------------------------------------------------
    # Backward — compute (gi0, gi1) from go (and optionally cached carry).
    # `mode` is a no-op (no params).
    # ------------------------------------------------------------------

    def vjp[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        grad_inputs: TensorPack[Self.ARITY],
    ) raises:
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["BinaryElementwise", target](self.ts.target_tag)

        comptime if target == "cpu":
            comptime N = BATCH * Self.DIM
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
            var gi0_p = grad_inputs.ptr[0]()
            var gi1_p = grad_inputs.ptr[1]()

            comptime if Self.OP.owns_cache:
                var c_p = self.cache.cpu_ptr()
                var k = 0
                while k + CPU_SIMD_W <= N:
                    var c = c_p.load[width=CPU_SIMD_W](k)
                    var g = go_p.load[width=CPU_SIMD_W](k)
                    gi0_p.store(k, Self.OP.backward_simd_x[CPU_SIMD_W](c, g))
                    gi1_p.store(k, Self.OP.backward_simd_y[CPU_SIMD_W](c, g))
                    k += CPU_SIMD_W
                while k < N:
                    gi0_p[k] = Self.OP.backward_scalar_x(c_p[k], go_p[k])
                    gi1_p[k] = Self.OP.backward_scalar_y(c_p[k], go_p[k])
                    k += 1
            else:
                # `c` is unused for owns_cache=False ops. Pass a zero
                # vector to keep the SIMD/scalar surface uniform.
                var zero_v = SIMD[DT, CPU_SIMD_W](0.0)
                var k = 0
                while k + CPU_SIMD_W <= N:
                    var g = go_p.load[width=CPU_SIMD_W](k)
                    gi0_p.store(k, Self.OP.backward_simd_x[CPU_SIMD_W](zero_v, g))
                    gi1_p.store(k, Self.OP.backward_simd_y[CPU_SIMD_W](zero_v, g))
                    k += CPU_SIMD_W
                var zero: Scalar[DT] = 0.0
                while k < N:
                    gi0_p[k] = Self.OP.backward_scalar_x(zero, go_p[k])
                    gi1_p[k] = Self.OP.backward_scalar_y(zero, go_p[k])
                    k += 1
        else:
            comptime N = BATCH * Self.DIM
            comptime layout = Layout.row_major(N)
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
            var go_lt = LayoutTensor[DT, layout, MutAnyOrigin](go_p)
            var gi0_lt = grad_inputs.lt[0, layout]()
            var gi1_lt = grad_inputs.lt[1, layout]()
            comptime n_blocks = (N + TPB - 1) // TPB

            comptime if Self.OP.owns_cache:
                var c_lt = LayoutTensor[DT, layout, MutAnyOrigin](
                    self.cache.dev.value()
                )
                comptime kernel = _be_backward_kernel_cached[N, Self.OP]
                self.ts.ctx.value().enqueue_function[kernel](
                    go_lt, c_lt, gi0_lt, gi1_lt,
                    grid_dim=n_blocks, block_dim=TPB,
                )
            else:
                comptime kernel = _be_backward_kernel_uncached[N, Self.OP]
                self.ts.ctx.value().enqueue_function[kernel](
                    go_lt, gi0_lt, gi1_lt,
                    grid_dim=n_blocks, block_dim=TPB,
                )

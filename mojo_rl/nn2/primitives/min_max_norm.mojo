"""MinMaxNorm[DIM] — per-sample (x - min) / (max - min) scaling.

Phase 2 of `nn2/PORTING_PLAN.md`. Used by MuZero (paper appendix
Training, see muzero-general models.py:138-145) and EZ-V2 to keep the
representation network's output bounded, with gradient flowing through
the rescaling so the rep network learns to produce well-spread outputs.

Math (per sample of dim N):
    m = min(x), M = max(x), s = clamp(M - m, ≥ ε)
    y_j = (x_j - m) / s

Backward (given grad_y, compute grad_x):
    G  = Σ grad_y
    Gy = Σ grad_y · y
    grad_x[argmax] = (grad_y[argmax] - Gy) / s
    grad_x[argmin] = (Gy + grad_y[argmin] - G) / s
    grad_x[i ∉ {argmin, argmax}] = grad_y[i] / s
    grad_x = 0 in the degenerate (M - m < ε) case.

Sum-zero invariant: Σ grad_x = 0 (gradient is shift-invariant, since
y is shift-invariant in x).

Cache: leaf-owned copy of the input row, so backward can re-derive
min/max/argmin/argmax without indexing the orchestrator's input slab.
CPU + GPU paths. GPU layout: one block per sample, threads parallelise
DIM. Block reductions handle min / max / argmin / argmax / G / Gy.
"""

from std.gpu import thread_idx, block_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP, Cache
from ..core.module import Module, typed_view, typed_view_mut
from ..core.tensor_pack import TensorPack
from ..core.target_storage import (
    require_ctx,
    TargetStorage,
    assert_tag_for,
)


comptime MMN_EPS: Scalar[DT] = 1e-5
comptime MMN_TPB: Int = 128


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — one block per sample, threads stride over DIM.
# Block reductions: min, max for stats; min over argmin/argmax sentinels
# for index resolution; sum for G = Σ grad_y and Gy = Σ grad_y · y.
# ──────────────────────────────────────────────────────────────────────


def _min_max_norm_forward_kernel[
    BATCH: Int, DIM: Int,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if b >= BATCH:
        return

    var pos_inf = Scalar[DT](1e30)
    var neg_inf = Scalar[DT](-1e30)
    var my_min = pos_inf
    var my_max = neg_inf
    var idx = t
    while idx < DIM:
        var v = rebind[Scalar[DT]](input[b, idx])
        if v < my_min:
            my_min = v
        if v > my_max:
            my_max = v
        idx += MMN_TPB

    var min_val = block.min[block_size=MMN_TPB, broadcast=True](val=my_min)
    var max_val = block.max[block_size=MMN_TPB, broadcast=True](val=my_max)

    var s = max_val - min_val
    if s < MMN_EPS:
        s = MMN_EPS
    var inv_s = Scalar[DT](1.0) / s

    idx = t
    while idx < DIM:
        var x = rebind[Scalar[DT]](input[b, idx])
        cache[b, idx] = x
        output[b, idx] = (x - min_val) * inv_s
        idx += MMN_TPB


def _min_max_norm_backward_kernel[
    BATCH: Int, DIM: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if b >= BATCH:
        return

    var pos_inf = Scalar[DT](1e30)
    var neg_inf = Scalar[DT](-1e30)
    var my_min = pos_inf
    var my_max = neg_inf
    var my_argmin: Int = 0
    var my_argmax: Int = 0
    var idx = t
    while idx < DIM:
        var v = rebind[Scalar[DT]](cache[b, idx])
        if v < my_min:
            my_min = v
            my_argmin = idx
        if v > my_max:
            my_max = v
            my_argmax = idx
        idx += MMN_TPB

    var min_val = block.min[block_size=MMN_TPB, broadcast=True](val=my_min)
    var max_val = block.max[block_size=MMN_TPB, broadcast=True](val=my_max)

    # Resolve argmin/argmax: threads that don't hold the block min/max
    # emit a sentinel (DIM), then block.min finds the smallest valid index.
    var SENTINEL = DIM
    var my_argmin_s: Int = my_argmin if my_min == min_val else SENTINEL
    var my_argmax_s: Int = my_argmax if my_max == max_val else SENTINEL
    var argmin = block.min[block_size=MMN_TPB, broadcast=True](
        val=Scalar[DType.int32](my_argmin_s)
    )
    var argmax = block.min[block_size=MMN_TPB, broadcast=True](
        val=Scalar[DType.int32](my_argmax_s)
    )

    var s = max_val - min_val
    var degenerate = s < MMN_EPS
    if degenerate:
        s = MMN_EPS
    var inv_s = Scalar[DT](1.0) / s

    if degenerate:
        idx = t
        while idx < DIM:
            grad_input[b, idx] = Scalar[DT](0.0)
            idx += MMN_TPB
        return

    var my_G = Scalar[DT](0.0)
    var my_Gy = Scalar[DT](0.0)
    idx = t
    while idx < DIM:
        var x = rebind[Scalar[DT]](cache[b, idx])
        var y = (x - min_val) * inv_s
        var dy = rebind[Scalar[DT]](grad_output[b, idx])
        my_G += dy
        my_Gy += dy * y
        idx += MMN_TPB
    var G = block.sum[block_size=MMN_TPB, broadcast=True](val=my_G)
    var Gy = block.sum[block_size=MMN_TPB, broadcast=True](val=my_Gy)

    var dy_argmin = rebind[Scalar[DT]](grad_output[b, Int(argmin)])
    var dy_argmax = rebind[Scalar[DT]](grad_output[b, Int(argmax)])

    idx = t
    while idx < DIM:
        var dy = rebind[Scalar[DT]](grad_output[b, idx])
        var dx: Scalar[DT]
        if Int32(idx) == argmin and Int32(idx) == argmax:
            dx = Scalar[DT](0.0)
        elif Int32(idx) == argmin:
            dx = (Gy + dy_argmin - G) * inv_s
        elif Int32(idx) == argmax:
            dx = (dy_argmax - Gy) * inv_s
        else:
            dx = dy * inv_s
        grad_input[b, idx] = dx
        idx += MMN_TPB


struct MinMaxNorm[DIM: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.DIM)
    comptime OUT_DIM = Self.DIM

    # Cache: per-sample copy of x, re-scanned for min/max/argmin/argmax
    # in vjp. Cheaper than caching indices (no int-as-float fragility).
    var cache_x: Cache["cache_x"]
    var ts: TargetStorage

    def __init__(out self):
        self.cache_x = Cache["cache_x"]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. INIT ignored (no params)."""
        comptime assert target == "cpu" or target == "gpu", (
            "MinMaxNorm: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.DIM > 1, "MinMaxNorm: DIM must be > 1"
        var n = Self()
        comptime if target == "cpu":
            n.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["MinMaxNorm.make[target='gpu']"](ctx)
            n.ts = TargetStorage.make_gpu(ctx_v)
        return n^

    def _ensure_cache_gpu(mut self, batch: Int) raises:
        var ctx = self.ts.ctx.value()
        self.cache_x.ensure_gpu(ctx, batch * Self.DIM)
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
        assert_tag_for["MinMaxNorm", target](self.ts.target_tag)
        var input = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            self.cache_x.ensure_cpu(BATCH * Self.DIM)
            var cache_v = TileTensor(
                self.cache_x.cpu, row_major[BATCH, Self.DIM](),
            )
            for b in range(BATCH):
                var x0: Scalar[DT] = input[b, 0]
                var min_val = x0
                var max_val = x0
                cache_v[b, 0] = x0
                for i in range(1, Self.DIM):
                    var v: Scalar[DT] = input[b, i]
                    cache_v[b, i] = v
                    if v < min_val:
                        min_val = v
                    if v > max_val:
                        max_val = v
                var s = max_val - min_val
                if s < MMN_EPS:
                    s = MMN_EPS
                var inv_s = Scalar[DT](1.0) / s
                for i in range(Self.DIM):
                    output_v[b, i] = (cache_v[b, i] - min_val) * inv_s
        else:
            self._ensure_cache_gpu(BATCH)
            comptime layout_2d = Layout.row_major(BATCH, Self.DIM)
            var in_p = input.ptr
            var out_p = output_v.ptr
            var in_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](in_p)
            var out_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](out_p)
            var cache_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](
                self.cache_x.dev.value()
            )
            comptime kernel = _min_max_norm_forward_kernel[BATCH, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                in_lt, out_lt, cache_lt,
                grid_dim=BATCH, block_dim=MMN_TPB,
            )

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
        assert_tag_for["MinMaxNorm", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()

        comptime if target == "cpu":
            var cache_v = TileTensor(
                self.cache_x.cpu, row_major[BATCH, Self.DIM](),
            )
            for b in range(BATCH):
                var x0: Scalar[DT] = cache_v[b, 0]
                var min_val = x0
                var max_val = x0
                var argmin = 0
                var argmax = 0
                for i in range(1, Self.DIM):
                    var v: Scalar[DT] = cache_v[b, i]
                    if v < min_val:
                        min_val = v
                        argmin = i
                    if v > max_val:
                        max_val = v
                        argmax = i
                var raw_s = max_val - min_val
                var degenerate = raw_s < MMN_EPS
                if degenerate:
                    for i in range(Self.DIM):
                        grad_input_v[b, i] = Scalar[DT](0.0)
                    continue
                var inv_s = Scalar[DT](1.0) / raw_s
                var g_sum: Scalar[DT] = 0.0
                var gy_sum: Scalar[DT] = 0.0
                for i in range(Self.DIM):
                    var y = (cache_v[b, i] - min_val) * inv_s
                    var dy: Scalar[DT] = grad_output_v[b, i]
                    g_sum += dy
                    gy_sum += dy * y
                var dy_argmin: Scalar[DT] = grad_output_v[b, argmin]
                var dy_argmax: Scalar[DT] = grad_output_v[b, argmax]
                for i in range(Self.DIM):
                    var dy: Scalar[DT] = grad_output_v[b, i]
                    var dx: Scalar[DT]
                    if i == argmin and i == argmax:
                        dx = Scalar[DT](0.0)
                    elif i == argmin:
                        dx = (gy_sum + dy_argmin - g_sum) * inv_s
                    elif i == argmax:
                        dx = (dy_argmax - gy_sum) * inv_s
                    else:
                        dx = dy * inv_s
                    grad_input_v[b, i] = dx
        else:
            comptime layout_2d = Layout.row_major(BATCH, Self.DIM)
            var go_p = grad_output_v.ptr
            var gi_p = grad_input_v.ptr
            var go_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](go_p)
            var gi_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](gi_p)
            var cache_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](
                self.cache_x.dev.value()
            )
            comptime kernel = _min_max_norm_backward_kernel[BATCH, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, cache_lt, gi_lt,
                grid_dim=BATCH, block_dim=MMN_TPB,
            )

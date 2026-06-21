"""D0 microbench: storage LayerNorm / RMSNorm forward GPU kernel —
baseline (scalar strided, DT accum) vs optimized (VEC-wide vectorized loads,
accum_type). Self-contained A/B: both kernels run in the SAME process at the
SAME shapes, so one run gives the real speedup on this GPU.

Both variants keep the identical launch config (grid=BATCH, block=128) so the
ONLY difference measured is vectorization + accum dtype — not the block shape.

The norm forward is memory-bound: ~3 reads of input (sum / var / normalize) +
2 writes (output, cache_xhat). Effective GB/s = 5·BATCH·DIM·4 / time, an
approximation, but identical for both variants so the ratio is exact.

Run (NVIDIA — required for the perf sign-off in
docs/STORAGE_NN_GPU_KERNEL_OPTIMIZATION.md):
    pixi run -e nvidia mojo run -I . benchmarks/bench_storage_norm_gpu.mojo
Apple (sanity only — Metal perf is not the sign-off):
    pixi run -e apple  mojo run -I . benchmarks/bench_storage_norm_gpu.mojo
"""

from std.math import sqrt
from std.gpu import thread_idx, block_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext
from std.time import perf_counter_ns
from std.utils import Index
from std.utils.numerics import get_accum_type
from layout import Layout, LayoutTensor

comptime DT = DType.float32
comptime TPB = 128
comptime ACC = get_accum_type[DT]()
comptime LN_EPS: Scalar[DT] = 1e-5


# ───────────── baseline: scalar strided, DT accumulation ─────────────
def _ln_fwd_scalar[BATCH: Int, DIM: Int](
    input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    beta: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    cache_xhat: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_inv_std: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
):
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if b >= BATCH:
        return
    var inv_dim: Scalar[DT] = 1.0 / Float32(DIM)
    var my_sum: Scalar[DT] = 0.0
    var idx = t
    while idx < DIM:
        my_sum += rebind[Scalar[DT]](input[b, idx])
        idx += TPB
    var mean_val = block.sum[block_size=TPB, broadcast=True](val=my_sum) * inv_dim
    var my_var: Scalar[DT] = 0.0
    idx = t
    while idx < DIM:
        var diff = rebind[Scalar[DT]](input[b, idx]) - mean_val
        my_var += diff * diff
        idx += TPB
    var var_val = block.sum[block_size=TPB, broadcast=True](val=my_var) * inv_dim
    var inv_std: Scalar[DT] = 1.0 / sqrt(var_val + LN_EPS)
    if t == 0:
        cache_inv_std[b] = inv_std
    idx = t
    while idx < DIM:
        var x = rebind[Scalar[DT]](input[b, idx])
        var x_hat = (x - mean_val) * inv_std
        cache_xhat[b, idx] = x_hat
        output[b, idx] = (
            rebind[Scalar[DT]](gamma[idx]) * x_hat
            + rebind[Scalar[DT]](beta[idx])
        )
        idx += TPB


# ───────────── optimized: VEC-wide vectorized, accum_type ─────────────
def _ln_fwd_vec[BATCH: Int, DIM: Int](
    input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    beta: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    cache_xhat: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_inv_std: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
):
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if b >= BATCH:
        return
    comptime VEC = 4 if DIM % 4 == 0 else 1
    var inv_dim = Scalar[ACC](1.0) / Scalar[ACC](DIM)
    var my_sum = Scalar[ACC](0)
    var idx = t * VEC
    while idx < DIM:
        my_sum += input.load[width=VEC](b, idx).cast[ACC]().reduce_add()
        idx += TPB * VEC
    var mean_val = block.sum[block_size=TPB, broadcast=True](val=my_sum) * inv_dim
    var my_var = Scalar[ACC](0)
    idx = t * VEC
    while idx < DIM:
        var diff = input.load[width=VEC](b, idx).cast[ACC]() - mean_val
        my_var += (diff * diff).reduce_add()
        idx += TPB * VEC
    var var_val = block.sum[block_size=TPB, broadcast=True](val=my_var) * inv_dim
    var inv_std = Scalar[ACC](1.0) / sqrt(var_val + LN_EPS.cast[ACC]())
    if t == 0:
        cache_inv_std[b] = inv_std.cast[DT]()
    idx = t * VEC
    while idx < DIM:
        var x = input.load[width=VEC](b, idx).cast[ACC]()
        var x_hat = (x - mean_val) * inv_std
        cache_xhat.store[width=VEC](b, idx, x_hat.cast[DT]())
        var g = gamma.load[width=VEC](Index(idx)).cast[ACC]()
        var bt = beta.load[width=VEC](Index(idx)).cast[ACC]()
        output.store[width=VEC](b, idx, (g * x_hat + bt).cast[DT]())
        idx += TPB * VEC


def _time[
    BATCH: Int, DIM: Int, VECTORIZED: Bool, WARMUP: Int, ITERS: Int
](ctx: DeviceContext, label: StaticString) raises:
    var inp = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    var out = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    var gam = ctx.enqueue_create_buffer[DT](DIM)
    var bet = ctx.enqueue_create_buffer[DT](DIM)
    var xh = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    var iv = ctx.enqueue_create_buffer[DT](BATCH)
    _ = inp.enqueue_fill(Scalar[DT](0.013))
    _ = gam.enqueue_fill(Scalar[DT](1.0))
    _ = bet.enqueue_fill(Scalar[DT](0.0))

    comptime l2d = Layout.row_major(BATCH, DIM)
    comptime ld = Layout.row_major(DIM)
    comptime lb = Layout.row_major(BATCH)
    var it = LayoutTensor[DT, l2d, MutAnyOrigin](inp.unsafe_ptr())
    var ot = LayoutTensor[DT, l2d, MutAnyOrigin](out.unsafe_ptr())
    var gt = LayoutTensor[DT, ld, MutAnyOrigin](gam.unsafe_ptr())
    var btt = LayoutTensor[DT, ld, MutAnyOrigin](bet.unsafe_ptr())
    var xt = LayoutTensor[DT, l2d, MutAnyOrigin](xh.unsafe_ptr())
    var ivt = LayoutTensor[DT, lb, MutAnyOrigin](iv.unsafe_ptr())

    comptime kern = _ln_fwd_vec[BATCH, DIM] if VECTORIZED else _ln_fwd_scalar[
        BATCH, DIM
    ]
    comptime for _ in range(WARMUP):
        ctx.enqueue_function[kern](
            it, ot, gt, btt, xt, ivt, grid_dim=BATCH, block_dim=TPB
        )
    ctx.synchronize()
    var t0 = perf_counter_ns()
    comptime for _ in range(ITERS):
        ctx.enqueue_function[kern](
            it, ot, gt, btt, xt, ivt, grid_dim=BATCH, block_dim=TPB
        )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var us = Float64(t1 - t0) / Float64(ITERS) / 1000.0
    var gb = 5.0 * Float64(BATCH) * Float64(DIM) * 4.0 / 1e9
    var gbps = gb / (us / 1e6)
    print("  ", label, " B=", BATCH, " D=", DIM, " | ", us, "us/iter ", gbps, "GB/s")


def _ab[BATCH: Int, DIM: Int, WARMUP: Int, ITERS: Int](ctx: DeviceContext) raises:
    _time[BATCH, DIM, False, WARMUP, ITERS](ctx, "scalar")
    _time[BATCH, DIM, True, WARMUP, ITERS](ctx, "vec   ")


def main() raises:
    var ctx = DeviceContext()
    print("LayerNorm forward GPU — baseline(scalar) vs optimized(vec) [fp32]")
    print("=" * 66)
    _ab[4096, 256, 10, 100](ctx)
    _ab[4096, 512, 10, 100](ctx)
    _ab[4096, 1024, 10, 100](ctx)
    _ab[1024, 4096, 10, 100](ctx)
    _ab[16384, 256, 10, 100](ctx)
    print("=" * 66)
    print("vec/scalar GB/s ratio = realized speedup. If vec is already near")
    print("peak HBM bandwidth, the kernel is done; if not, more headroom.")

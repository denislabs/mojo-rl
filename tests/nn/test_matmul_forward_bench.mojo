"""Benchmark: MatMul forward — hand-written MMA vs max_matmul.

The 339μs kernel is MatMul[5376, 256].eval_gpu using hand-written MMA
with 672 k-tiles (K=5376, MMA_K=8 → 672 iterations with barrier each).

Fix: Replace with max_matmul (linalg) like conv2d already does.

Run:
    pixi run -e nvidia mojo run -I . tests/nn/test_matmul_forward_bench.mojo
"""

from std.random import seed, random_float64
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tile_tensor import lt_to_tt
from linalg.matmul import matmul as max_matmul
from std.runtime.asyncrt import DeviceContextPtr

from mojo_rl.nn.constants import dtype, TPB, MMA_BLOCK_THREADS
from mojo_rl.nn.autodiff.primitives.matmul import MatMul


def main() raises:
    seed(42)
    print("=" * 70)
    print("MatMul FORWARD: hand-written MMA vs max_matmul")
    print("=" * 70)
    print()

    comptime BATCH = 64
    comptime N_ITERS = 1000

    with DeviceContext() as ctx:
        # Test multiple layer sizes
        _bench[BATCH, 5376, 256, N_ITERS, "LinBNReLU[5376→256]"](ctx)
        _bench[BATCH, 256, 128, N_ITERS, "LinBNReLU[256→128] "](ctx)
        _bench[BATCH, 128, 7, N_ITERS, "Linear[128→7] policy"](ctx)
        _bench[BATCH, 128, 1, N_ITERS, "Linear[128→1] value "](ctx)

    print("=" * 70)


def _bench[
    BATCH: Int,
    IN: Int,
    OUT: Int,
    N_ITERS: Int,
    label: StringLiteral,
](ctx: DeviceContext) raises:
    comptime MM = MatMul[IN, OUT]
    comptime num_k_tiles = (IN + 7) // 8  # MMA_K = 8

    var input_buf = ctx.enqueue_create_buffer[dtype](BATCH * IN)
    var params_buf = ctx.enqueue_create_buffer[dtype](MM.PARAM_SIZE)
    var out_buf = ctx.enqueue_create_buffer[dtype](BATCH * OUT)
    var cache_buf = ctx.enqueue_create_buffer[dtype](BATCH * MM.CACHE_SIZE)
    var out_buf2 = ctx.enqueue_create_buffer[dtype](BATCH * OUT)

    var hb = ctx.enqueue_create_host_buffer[dtype](max(BATCH * IN, IN * OUT))
    for i in range(BATCH * IN):
        hb.unsafe_ptr()[i] = Scalar[dtype](random_float64(-1.0, 1.0).cast[dtype]())
    ctx.enqueue_copy(input_buf, hb)
    for i in range(MM.PARAM_SIZE):
        hb.unsafe_ptr()[i] = Scalar[dtype](random_float64(-0.5, 0.5).cast[dtype]())
    ctx.enqueue_copy(params_buf, hb)
    ctx.enqueue_memset(out_buf, 0)
    ctx.enqueue_memset(out_buf2, 0)
    ctx.enqueue_memset(cache_buf, 0)
    ctx.synchronize()

    var input_lt = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](input_buf.unsafe_ptr())
    var params_lt = LayoutTensor[dtype, Layout.row_major(MM.PARAM_SIZE), MutAnyOrigin](params_buf.unsafe_ptr())
    var out_lt = LayoutTensor[dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin](out_buf.unsafe_ptr())
    var cache_lt = LayoutTensor[dtype, Layout.row_major(BATCH, MM.CACHE_SIZE), MutAnyOrigin](cache_buf.unsafe_ptr())

    # max_matmul tensors: output = input @ W
    var input_mm = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](input_buf.unsafe_ptr())
    var W_mm = LayoutTensor[dtype, Layout.row_major(IN, OUT), MutAnyOrigin](params_buf.unsafe_ptr())
    var out_mm = LayoutTensor[dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin](out_buf2.unsafe_ptr())

    var ws = UnsafePointer[Scalar[dtype], MutAnyOrigin]()

    # Warmup
    for _ in range(5):
        MM.eval_gpu[BATCH](ctx, out_lt, input_lt, params_lt, cache_lt, ws)
        max_matmul[target="gpu"](lt_to_tt(out_mm), lt_to_tt(input_mm), lt_to_tt(W_mm), DeviceContextPtr(ctx))
    ctx.synchronize()

    # Benchmark MMA
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(N_ITERS):
        MM.eval_gpu[BATCH](ctx, out_lt, input_lt, params_lt, cache_lt, ws)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var mma_us = Float64(t1 - t0) / 1000.0 / Float64(N_ITERS)

    # Benchmark max_matmul
    ctx.synchronize()
    var t2 = perf_counter_ns()
    for _ in range(N_ITERS):
        max_matmul[target="gpu"](lt_to_tt(out_mm), lt_to_tt(input_mm), lt_to_tt(W_mm), DeviceContextPtr(ctx))
    ctx.synchronize()
    var t3 = perf_counter_ns()
    var mm_us = Float64(t3 - t2) / 1000.0 / Float64(N_ITERS)

    # Verify
    var out_mma_hb = ctx.enqueue_create_host_buffer[dtype](BATCH * OUT)
    var out_mm_hb = ctx.enqueue_create_host_buffer[dtype](BATCH * OUT)
    ctx.enqueue_copy(out_mma_hb, out_buf)
    ctx.enqueue_copy(out_mm_hb, out_buf2)
    ctx.synchronize()
    var max_diff: Float64 = 0
    for i in range(BATCH * OUT):
        var d = abs(Float64(out_mma_hb.unsafe_ptr()[i]) - Float64(out_mm_hb.unsafe_ptr()[i]))
        if d > max_diff:
            max_diff = d

    var speedup = mma_us / mm_us
    print(
        "  "
        + label
        + " [B="
        + String(BATCH)
        + ", K="
        + String(IN)
        + " → "
        + String(num_k_tiles)
        + " k-tiles]:"
    )
    print(
        "    MMA kernel:   "
        + String(mma_us)[byte=:10]
        + " μs  grid=("
        + String((OUT + 31) // 32)
        + ","
        + String((BATCH + 31) // 32)
        + ")"
    )
    print(
        "    max_matmul:   "
        + String(mm_us)[byte=:10]
        + " μs"
    )
    print(
        "    speedup:      "
        + String(speedup)[byte=:6]
        + "x  diff="
        + String(max_diff)
    )
    print()

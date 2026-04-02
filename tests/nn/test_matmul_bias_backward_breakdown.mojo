"""Benchmark: FusedMatMulBias backward for AlphaZero ConnectFour layers.

Identifies which layer produces the 339μs autodiff_primitive in nsys.

Layers in AlphaZero ConnectFour PredNet (backward order):
  - Linear[128, 7]    (policy head)
  - Linear[128, 1]    (value head)
  - LinearBNReLU[256, 128]
  - LinearBNReLU[5376, 256]  ← likely the 339μs one
  - (then conv layers)

Run:
    pixi run -e nvidia mojo run -I . tests/nn/test_matmul_bias_backward_breakdown.mojo
"""

from std.random import seed, random_float64
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.autodiff.fused.matmul_bias import FusedMatMulBias


def bench_layer[
    BATCH: Int,
    IN: Int,
    OUT: Int,
    N_ITERS: Int,
    label: StringLiteral,
](ctx: DeviceContext) raises:
    comptime F = FusedMatMulBias[IN, OUT]

    # Allocate
    var go_buf = ctx.enqueue_create_buffer[dtype](BATCH * OUT)
    var gi_buf = ctx.enqueue_create_buffer[dtype](BATCH * IN)
    var params_buf = ctx.enqueue_create_buffer[dtype](F.PARAM_SIZE)
    var cache_buf = ctx.enqueue_create_buffer[dtype](BATCH * F.CACHE_SIZE)
    var gp_buf = ctx.enqueue_create_buffer[dtype](F.PARAM_SIZE)

    # Fill random
    var go_hb = ctx.enqueue_create_host_buffer[dtype](BATCH * OUT)
    var params_hb = ctx.enqueue_create_host_buffer[dtype](F.PARAM_SIZE)
    var cache_hb = ctx.enqueue_create_host_buffer[dtype](BATCH * IN)
    for i in range(BATCH * OUT):
        go_hb.unsafe_ptr()[i] = Scalar[dtype](random_float64(-1.0, 1.0).cast[dtype]())
    for i in range(F.PARAM_SIZE):
        params_hb.unsafe_ptr()[i] = Scalar[dtype](random_float64(-0.5, 0.5).cast[dtype]())
    for i in range(BATCH * IN):
        cache_hb.unsafe_ptr()[i] = Scalar[dtype](random_float64(-1.0, 1.0).cast[dtype]())
    ctx.enqueue_copy(go_buf, go_hb)
    ctx.enqueue_copy(params_buf, params_hb)
    ctx.enqueue_copy(cache_buf, cache_hb)
    ctx.enqueue_memset(gi_buf, 0)
    ctx.enqueue_memset(gp_buf, 0)
    ctx.synchronize()

    var go_lt = LayoutTensor[dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin](go_buf.unsafe_ptr())
    var gi_lt = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](gi_buf.unsafe_ptr())
    var params_lt = LayoutTensor[dtype, Layout.row_major(F.PARAM_SIZE), MutAnyOrigin](params_buf.unsafe_ptr())
    var cache_lt = LayoutTensor[dtype, Layout.row_major(BATCH, F.CACHE_SIZE), MutAnyOrigin](cache_buf.unsafe_ptr())
    var gp_lt = LayoutTensor[dtype, Layout.row_major(F.PARAM_SIZE), MutAnyOrigin](gp_buf.unsafe_ptr())

    # No workspace needed for FusedMatMulBias
    var ws = UnsafePointer[Scalar[dtype], MutAnyOrigin]()

    # Warmup
    for _ in range(5):
        F.vjp_gpu[BATCH](ctx, go_lt, gi_lt, params_lt, cache_lt, gp_lt, ws)
    ctx.synchronize()

    # Benchmark
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(N_ITERS):
        F.vjp_gpu[BATCH](ctx, go_lt, gi_lt, params_lt, cache_lt, gp_lt, ws)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var us = Float64(t1 - t0) / 1000.0 / Float64(N_ITERS)

    print(
        "  "
        + label
        + " [B="
        + String(BATCH)
        + ", "
        + String(IN)
        + "→"
        + String(OUT)
        + "]: "
        + String(us)[byte=:10]
        + " μs  (dW params="
        + String(IN * OUT)
        + ")"
    )


def main() raises:
    seed(42)
    print("=" * 70)
    print("FusedMatMulBias Backward — AlphaZero ConnectFour Layers")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        # AlphaZero ConnectFour layers (backward order, BATCH=64)
        bench_layer[64, 128, 7, 1000, "Linear[128,7] policy"](ctx)
        bench_layer[64, 128, 1, 1000, "Linear[128,1] value "](ctx)
        bench_layer[64, 256, 128, 1000, "LinBNReLU[256,128] "](ctx)
        bench_layer[64, 5376, 256, 1000, "LinBNReLU[5376,256]"](ctx)

    print()
    print("=" * 70)

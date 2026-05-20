"""Benchmark FusedConv2DActivation (Conv2DReLU/Tanh/Sigmoid/Mish) on CPU.

Same shapes as the AlphaZero TTT CNN benchmark to keep numbers comparable
to the BN-fused variants we already benched. Reports forward + backward
time per layer at BATCH=64.

Run:
    pixi run mojo run -I . benchmarks/benchmark_conv2d_act_cpu.mojo
"""

from std.memory import alloc, memset
from std.random import seed, random_float64
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.model import Conv2DReLU, Conv2DTanh, Conv2DSigmoid, Model


@always_inline
def fill_random(p: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int):
    for i in range(n):
        p[i] = Scalar[dtype](random_float64(-1.0, 1.0))


def bench[
    M: Model, BATCH: Int, FWD_ITERS: Int, BWD_ITERS: Int
](label: String) raises:
    comptime IN_DIM = M.IN_DIM
    comptime OUT_DIM = M.OUT_DIM
    comptime PARAM_SIZE = M.PARAM_SIZE
    comptime STATE_SIZE = M.STATE_SIZE
    comptime CACHE_SIZE = M.CACHE_SIZE

    print("─" * 70)
    print(label)
    print(
        "  IN_DIM=",
        IN_DIM,
        " OUT_DIM=",
        OUT_DIM,
        " PS=",
        PARAM_SIZE,
        " BATCH=",
        BATCH,
    )

    var input_buf = alloc[Scalar[dtype]](BATCH * IN_DIM)
    var output_buf = alloc[Scalar[dtype]](BATCH * OUT_DIM)
    var grad_in_buf = alloc[Scalar[dtype]](BATCH * IN_DIM)
    var grad_out_buf = alloc[Scalar[dtype]](BATCH * OUT_DIM)
    var params_buf = alloc[Scalar[dtype]](PARAM_SIZE)
    var grads_buf = alloc[Scalar[dtype]](PARAM_SIZE)
    var state_buf = alloc[Scalar[dtype]](max(1, STATE_SIZE))
    var cache_buf = alloc[Scalar[dtype]](BATCH * max(1, CACHE_SIZE))

    fill_random(input_buf, BATCH * IN_DIM)
    fill_random(grad_out_buf, BATCH * OUT_DIM)
    memset(output_buf, 0, BATCH * OUT_DIM)
    memset(grad_in_buf, 0, BATCH * IN_DIM)
    memset(grads_buf, 0, PARAM_SIZE)
    memset(cache_buf, 0, BATCH * max(1, CACHE_SIZE))

    var params_lt = LayoutTensor[
        dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
    ](params_buf)
    var state_lt = LayoutTensor[
        dtype, Layout.row_major(STATE_SIZE), MutAnyOrigin
    ](state_buf)
    M.initialize_params[Kaiming[], dtype](params_lt)

    var input_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin
    ](input_buf)
    var output_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ](output_buf)
    var grad_in_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin
    ](grad_in_buf)
    var grad_out_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ](grad_out_buf)
    var cache_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin
    ](cache_buf)
    var grads_lt = LayoutTensor[
        dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
    ](grads_buf)

    # Warmup
    M.forward[BATCH, dtype](
        input_lt, output_lt, params_lt, state_lt, cache_lt
    )
    M.backward[BATCH, dtype](
        grad_out_lt, grad_in_lt, params_lt, state_lt, cache_lt, grads_lt
    )

    var t0 = perf_counter_ns()
    for _ in range(FWD_ITERS):
        M.forward[BATCH, dtype](
            input_lt, output_lt, params_lt, state_lt, cache_lt
        )
    var t1 = perf_counter_ns()
    var fwd_ms = Float64(t1 - t0) / 1e6 / Float64(FWD_ITERS)

    var t2 = perf_counter_ns()
    for _ in range(BWD_ITERS):
        M.backward[BATCH, dtype](
            grad_out_lt, grad_in_lt, params_lt, state_lt, cache_lt, grads_lt
        )
    var t3 = perf_counter_ns()
    var bwd_ms = Float64(t3 - t2) / 1e6 / Float64(BWD_ITERS)

    print("  forward (train) :", String(fwd_ms), "ms")
    print("  backward        :", String(bwd_ms), "ms")

    input_buf.free()
    output_buf.free()
    grad_in_buf.free()
    grad_out_buf.free()
    params_buf.free()
    grads_buf.free()
    state_buf.free()
    cache_buf.free()


def main() raises:
    seed(42)
    comptime BATCH = 64
    comptime F = 128

    print("=" * 70)
    print(" Conv2DReLU/Tanh/Sigmoid CPU benchmark, BATCH=", BATCH)
    print("=" * 70)
    print()

    bench[Conv2DReLU[3, F, 3, 1, 1, 3, 3], BATCH, 30, 15](
        "Conv2DReLU L1: 3 -> F, 3x3 same (AZ TTT-CNN sized)"
    )
    bench[Conv2DReLU[F, F, 3, 1, 1, 3, 3], BATCH, 20, 10](
        "Conv2DReLU L2/L3: F -> F, 3x3 same (AZ TTT-CNN sized)"
    )
    bench[Conv2DReLU[F, F, 3, 1, 0, 3, 3], BATCH, 30, 15](
        "Conv2DReLU L4: F -> F, 3x3 -> 1x1 valid (AZ TTT-CNN sized)"
    )
    bench[Conv2DTanh[F, F, 3, 1, 1, 3, 3], BATCH, 20, 10](
        "Conv2DTanh: F -> F, 3x3 same (sanity)"
    )

    print()
    print("=" * 70)
    print(" Done.")
    print("=" * 70)

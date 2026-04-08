"""POC: Benchmarking matmul approaches for NormedLinear.

Compares:
  1. vendor_blas.matmul (cuBLAS)
  2. linalg.matmul.matmul with target="gpu" (Modular's multistage GEMM)
  3. Our custom MMA kernel (via NormedLinear forward)

Run:
    pixi run -e nvidia mojo run -I . tests/nn/test_max_matmul_poc.mojo
"""

from std.random import seed, random_float64
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype

# Max kernels
import linalg.matmul.vendor.blas as vendor_blas
from linalg.matmul import matmul as max_matmul
from std.runtime.asyncrt import DeviceContextPtr


def main() raises:
    seed(42)

    # Dimensions matching MPPI: BATCH=17152, IN=512, OUT=512
    comptime BATCH = 17152
    comptime IN_DIM = 512
    comptime OUT_DIM = 512
    comptime N_ITERS = 100

    var flops = 2.0 * Float64(BATCH) * Float64(IN_DIM) * Float64(OUT_DIM)

    print("=" * 60)
    print("Max Kernels Matmul Benchmark")
    print("=" * 60)
    print(
        "Matrix: ["
        + String(BATCH)
        + ", "
        + String(IN_DIM)
        + "] x ["
        + String(IN_DIM)
        + ", "
        + String(OUT_DIM)
        + "]"
    )
    print("Iterations: " + String(N_ITERS))
    print()

    with DeviceContext() as ctx:
        # Allocate GPU buffers
        var a_buf = ctx.enqueue_create_buffer[dtype](BATCH * IN_DIM)
        var b_buf = ctx.enqueue_create_buffer[dtype](IN_DIM * OUT_DIM)
        var c_buf1 = ctx.enqueue_create_buffer[dtype](BATCH * OUT_DIM)
        var c_buf2 = ctx.enqueue_create_buffer[dtype](BATCH * OUT_DIM)

        # Initialize with random data
        var a_host = ctx.enqueue_create_host_buffer[dtype](BATCH * IN_DIM)
        var b_host = ctx.enqueue_create_host_buffer[dtype](IN_DIM * OUT_DIM)
        for i in range(BATCH * IN_DIM):
            a_host[i] = Scalar[dtype](random_float64() * 0.1)
        for i in range(IN_DIM * OUT_DIM):
            b_host[i] = Scalar[dtype](random_float64() * 0.1)
        ctx.enqueue_copy(a_buf, a_host)
        ctx.enqueue_copy(b_buf, b_host)
        ctx.enqueue_memset(c_buf1, 0)
        ctx.enqueue_memset(c_buf2, 0)
        ctx.synchronize()

        var a_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin
        ](a_buf.unsafe_ptr())
        var b_tensor = LayoutTensor[
            dtype, Layout.row_major(IN_DIM, OUT_DIM), MutAnyOrigin
        ](b_buf.unsafe_ptr())
        var c1_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ](c_buf1.unsafe_ptr())
        var c2_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ](c_buf2.unsafe_ptr())

        # ── Warmup both ──
        print("Warming up...")
        vendor_blas.matmul(ctx, c1_tensor, a_tensor, b_tensor, c_row_major=True)
        max_matmul[target="gpu"](c2_tensor, a_tensor, b_tensor, DeviceContextPtr(ctx))
        ctx.synchronize()

        # ── Benchmark 1: vendor BLAS (cuBLAS) ──
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_ITERS):
            vendor_blas.matmul(
                ctx, c1_tensor, a_tensor, b_tensor, c_row_major=True
            )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var blas_us = Float64(t1 - t0) / 1000.0 / Float64(N_ITERS)
        var blas_tflops = flops / (blas_us * 1e-6) / 1e12

        print(
            "1. vendor BLAS (cuBLAS):      "
            + String(blas_us)[byte=:8]
            + " μs  |  "
            + String(blas_tflops)[byte=:6]
            + " TFLOPS"
        )

        # ── Benchmark 2: linalg.matmul with target="gpu" (multistage GEMM) ──
        ctx.synchronize()
        var t2 = perf_counter_ns()
        for _ in range(N_ITERS):
            max_matmul[target="gpu"](c2_tensor, a_tensor, b_tensor, DeviceContextPtr(ctx))
        ctx.synchronize()
        var t3 = perf_counter_ns()
        var gemm_us = Float64(t3 - t2) / 1000.0 / Float64(N_ITERS)
        var gemm_tflops = flops / (gemm_us * 1e-6) / 1e12

        print(
            "2. linalg.matmul (GPU GEMM):  "
            + String(gemm_us)[byte=:8]
            + " μs  |  "
            + String(gemm_tflops)[byte=:6]
            + " TFLOPS"
        )

        print(
            "3. Our custom MMA:            ~308     μs  |  ~29    TFLOPS (from"
            " nsys profile)"
        )

        print()
        print(
            "Speedup BLAS vs custom MMA: " + String(308.0 / blas_us)[byte=:4] + "x"
        )
        print(
            "Speedup GEMM vs custom MMA: " + String(308.0 / gemm_us)[byte=:4] + "x"
        )
        if gemm_us < blas_us:
            print(">>> multistage GEMM is FASTER than cuBLAS! <<<")
        else:
            print(
                ">>> cuBLAS is faster (GEMM/BLAS ratio: "
                + String(gemm_us / blas_us)[byte=:4]
                + "x) <<<"
            )

        # ── Verify both produce same results ──
        print()
        print("Verifying consistency...")
        var c1_host = ctx.enqueue_create_host_buffer[dtype](BATCH * OUT_DIM)
        var c2_host = ctx.enqueue_create_host_buffer[dtype](BATCH * OUT_DIM)
        ctx.enqueue_copy(c1_host, c_buf1)
        ctx.enqueue_copy(c2_host, c_buf2)
        ctx.synchronize()

        var max_diff: Float64 = 0.0
        for i in range(min(1000, BATCH * OUT_DIM)):
            var diff = abs(Float64(c1_host[i]) - Float64(c2_host[i]))
            if diff > max_diff:
                max_diff = diff
        print("Max abs diff (first 1000 elements): " + String(max_diff)[byte=:10])

        print()
        print("=" * 60)
        print("Benchmark complete!")
        print("=" * 60)

"""Test MMA matmul kernel vs CPU reference.

On NVIDIA: tests tensor core MMA kernel (gpu_matmul dispatches to mma_matmul_kernel).
On Apple: tests tiled scalar kernel (gpu_matmul dispatches to tiled_matmul_kernel).

Run with:
    pixi run -e apple mojo run tests/test_mma_matmul.mojo   # Apple Silicon
    pixi run -e nvidia mojo run tests/test_mma_matmul.mojo  # NVIDIA GPU
"""

from std.random import seed, random_float64
from std.time import perf_counter_ns
from std.sys import is_nvidia_gpu

from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from nn.constants import dtype, TILE
from nn.gpu.matmul import gpu_matmul, tiled_matmul_kernel


# =============================================================================
# CPU reference matmul
# =============================================================================


fn cpu_matmul[
    M: Int, K: Int, N: Int
](
    a: LayoutTensor[dtype, Layout.row_major(M, K), MutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(K, N), MutAnyOrigin],
    mut c: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
):
    for i in range(M):
        for j in range(N):
            var acc: Scalar[dtype] = 0
            for k in range(K):
                acc += rebind[Scalar[dtype]](a[i, k]) * rebind[Scalar[dtype]](
                    b[k, j]
                )
            c[i, j] = acc


# =============================================================================
# Test: small matrix (64×64)
# =============================================================================


fn test_matmul[M: Int, K: Int, N: Int](ctx: DeviceContext) raises:
    print("  Matrix: [" + String(M) + " x " + String(K) + "] @ [" + String(K) + " x " + String(N) + "]")

    # Allocate host memory
    comptime a_size = M * K
    comptime b_size = K * N
    comptime c_size = M * N

    var a_host = ctx.enqueue_create_host_buffer[dtype](a_size)
    var b_host = ctx.enqueue_create_host_buffer[dtype](b_size)
    var c_ref_host = ctx.enqueue_create_host_buffer[dtype](c_size)

    # Fill with random data
    seed(42)
    for i in range(a_size):
        a_host[i] = (random_float64() * 2.0 - 1.0).cast[dtype]()
    for i in range(b_size):
        b_host[i] = (random_float64() * 2.0 - 1.0).cast[dtype]()
    for i in range(c_size):
        c_ref_host[i] = Scalar[dtype](0)

    # CPU reference
    var a_lt = LayoutTensor[dtype, Layout.row_major(M, K), MutAnyOrigin](
        a_host.unsafe_ptr()
    )
    var b_lt = LayoutTensor[dtype, Layout.row_major(K, N), MutAnyOrigin](
        b_host.unsafe_ptr()
    )
    var c_ref_lt = LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin](
        c_ref_host.unsafe_ptr()
    )
    cpu_matmul[M, K, N](a_lt, b_lt, c_ref_lt)

    # GPU matmul
    var a_dev = ctx.enqueue_create_buffer[dtype](a_size)
    var b_dev = ctx.enqueue_create_buffer[dtype](b_size)
    var c_dev = ctx.enqueue_create_buffer[dtype](c_size)

    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    # Zero output
    var c_zero = ctx.enqueue_create_host_buffer[dtype](c_size)
    for i in range(c_size):
        c_zero[i] = Scalar[dtype](0)
    ctx.enqueue_copy(c_dev, c_zero)

    var out_lt = LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin](
        c_dev.unsafe_ptr()
    )
    var a_dev_lt = LayoutTensor[dtype, Layout.row_major(M, K), ImmutAnyOrigin](
        a_dev.unsafe_ptr()
    )
    var b_dev_lt = LayoutTensor[dtype, Layout.row_major(K, N), ImmutAnyOrigin](
        b_dev.unsafe_ptr()
    )

    # Warmup
    gpu_matmul[dtype, M, N, K, TILE](ctx, out_lt, a_dev_lt, b_dev_lt)
    ctx.synchronize()

    # Benchmark
    comptime NUM_ITERS = 100
    var start = perf_counter_ns()
    for _ in range(NUM_ITERS):
        gpu_matmul[dtype, M, N, K, TILE](ctx, out_lt, a_dev_lt, b_dev_lt)
    ctx.synchronize()
    var elapsed_ns = perf_counter_ns() - start
    var avg_us = elapsed_ns / (NUM_ITERS * 1000)

    # Copy result back
    var c_result = ctx.enqueue_create_host_buffer[dtype](c_size)
    ctx.enqueue_copy(c_result, c_dev)
    ctx.synchronize()

    # Compare
    var max_err: Float64 = 0
    var sum_err: Float64 = 0
    for i in range(c_size):
        var diff = abs(
            c_result[i].cast[DType.float64]()
            - c_ref_host[i].cast[DType.float64]()
        )
        if diff > max_err:
            max_err = diff
        sum_err += diff
    var avg_err = sum_err / Float64(c_size)

    var passed = max_err < 0.01
    var status = "PASS" if passed else "FAIL"
    print(
        "  ["
        + status
        + "] max_err="
        + String(max_err)
        + " avg_err="
        + String(avg_err)
        + " time="
        + String(avg_us)
        + "us/iter"
    )

    if not passed:
        # Print first few mismatches
        var shown = 0
        for i in range(c_size):
            var diff = abs(
                c_result[i].cast[DType.float64]()
                - c_ref_host[i].cast[DType.float64]()
            )
            if diff > 0.001 and shown < 5:
                var row = i // N
                var col = i % N
                print(
                    "    mismatch at ("
                    + String(row)
                    + ","
                    + String(col)
                    + "): gpu="
                    + String(c_result[i])
                    + " cpu="
                    + String(c_ref_host[i])
                )
                shown += 1


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 60)
    comptime if is_nvidia_gpu():
        print("MMA Matmul Test (NVIDIA — tensor core path)")
    else:
        print("MMA Matmul Test (Apple/other — tiled scalar fallback)")
    print("=" * 60)

    var ctx = DeviceContext()

    # Test various matrix sizes relevant to RL
    print("\n--- Small (CartPole-sized) ---")
    test_matmul[32, 64, 64](ctx)       # hidden × hidden
    test_matmul[32, 4, 64](ctx)        # input → hidden
    test_matmul[32, 64, 2](ctx)        # hidden → output

    print("\n--- Medium (LunarLander-sized) ---")
    test_matmul[64, 8, 64](ctx)        # input → hidden
    test_matmul[64, 64, 64](ctx)       # hidden × hidden
    test_matmul[64, 64, 4](ctx)        # hidden → output

    print("\n--- Larger (HalfCheetah-sized) ---")
    test_matmul[128, 17, 256](ctx)     # input → hidden
    test_matmul[128, 256, 256](ctx)    # hidden × hidden
    test_matmul[128, 256, 6](ctx)      # hidden → output

    print("\n--- Large (stress test) ---")
    test_matmul[256, 256, 256](ctx)
    test_matmul[512, 512, 512](ctx)

    print("\nDone!")

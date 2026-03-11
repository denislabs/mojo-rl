"""Benchmark matrix multiplication implementations on Apple Silicon.

Compares:
1. Original 16x16 tiled matmul (matmul.mojo)
2. Apple-optimized 8x8 tiled matmul
3. Register-blocked 2x2 kernel
4. FP16 kernel

Run with:
    pixi run -e apple mojo run tests/benchmark_matmul_apple.mojo
"""

from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from std.time import perf_counter_ns
from std.random import random_float64

from mojo_rl.nn.gpu.matmul import tiled_matmul_kernel
from mojo_rl.nn.gpu.matmul_apple import (
    matmul_apple_kernel,
    matmul_apple_reg2x2_kernel,
    matmul_fp16_apple_kernel,
    TILE_APPLE,
)
from mojo_rl.nn.constants import dtype, TILE


fn benchmark_size[M: Int, N: Int, K: Int](ctx: DeviceContext) raises:
    """Benchmark all kernels for a given matrix size."""
    print("\n" + "=" * 70)
    print("Matrix size: ", M, " x ", K, " @ ", K, " x ", N)
    print("=" * 70)

    # Allocate FP32 buffers
    var a = ctx.enqueue_create_buffer[dtype](M * K)
    var b = ctx.enqueue_create_buffer[dtype](K * N)
    var out_16x16 = ctx.enqueue_create_buffer[dtype](M * N)
    var out_8x8 = ctx.enqueue_create_buffer[dtype](M * N)
    var out_reg2x2 = ctx.enqueue_create_buffer[dtype](M * N)

    # Allocate FP16 buffers
    var a_fp16 = ctx.enqueue_create_buffer[DType.float16](M * K)
    var b_fp16 = ctx.enqueue_create_buffer[DType.float16](K * N)
    var out_fp16 = ctx.enqueue_create_buffer[DType.float16](M * N)

    # Initialize with random data
    with a.map_to_host() as a_host, b.map_to_host() as b_host:
        for i in range(M * K):
            a_host[i] = Scalar[dtype](random_float64(-1.0, 1.0))
        for i in range(K * N):
            b_host[i] = Scalar[dtype](random_float64(-1.0, 1.0))

    # Copy to FP16 buffers
    with a.map_to_host() as a32, a_fp16.map_to_host() as a16:
        for i in range(M * K):
            a16[i] = Float16(a32[i])
    with b.map_to_host() as b32, b_fp16.map_to_host() as b16:
        for i in range(K * N):
            b16[i] = Float16(b32[i])

    out_16x16.enqueue_fill(0)
    out_8x8.enqueue_fill(0)
    out_reg2x2.enqueue_fill(0)
    out_fp16.enqueue_fill(Float16(0))
    ctx.synchronize()

    var warmup = 5
    var iterations = 20

    print("\nRunning benchmarks...")
    print("-" * 70)

    # ========== 16x16 Tiled Kernel ==========
    var out_tensor_16 = LayoutTensor[
        dtype, Layout.row_major(M, N), MutAnyOrigin
    ](out_16x16)
    var a_tensor = LayoutTensor[dtype, Layout.row_major(M, K), ImmutAnyOrigin](
        a
    )
    var b_tensor = LayoutTensor[dtype, Layout.row_major(K, N), ImmutAnyOrigin](
        b
    )

    comptime grid_16 = ((N + TILE - 1) // TILE, (M + TILE - 1) // TILE)
    comptime block_16 = (TILE, TILE)
    comptime kernel_16 = tiled_matmul_kernel[dtype, M, N, K, TILE]

    for _ in range(warmup):
        ctx.enqueue_function[kernel_16, kernel_16](
            out_tensor_16,
            a_tensor,
            b_tensor,
            grid_dim=grid_16,
            block_dim=block_16,
        )
        ctx.synchronize()

    var total_ns_16: UInt = 0
    for _ in range(iterations):
        var start = perf_counter_ns()
        ctx.enqueue_function[kernel_16, kernel_16](
            out_tensor_16,
            a_tensor,
            b_tensor,
            grid_dim=grid_16,
            block_dim=block_16,
        )
        ctx.synchronize()
        total_ns_16 += perf_counter_ns() - start

    var time_16x16 = Float64(total_ns_16) / Float64(iterations) / 1_000_000.0

    # ========== 8x8 Apple Kernel ==========
    var out_tensor_8 = LayoutTensor[
        dtype, Layout.row_major(M, N), MutAnyOrigin
    ](out_8x8)
    comptime grid_8 = (
        (N + TILE_APPLE - 1) // TILE_APPLE,
        (M + TILE_APPLE - 1) // TILE_APPLE,
    )
    comptime block_8 = (TILE_APPLE, TILE_APPLE)
    comptime kernel_8 = matmul_apple_kernel[dtype, M, N, K, TILE_APPLE]

    for _ in range(warmup):
        ctx.enqueue_function[kernel_8, kernel_8](
            out_tensor_8, a_tensor, b_tensor, grid_dim=grid_8, block_dim=block_8
        )
        ctx.synchronize()

    var total_ns_8: UInt = 0
    for _ in range(iterations):
        var start = perf_counter_ns()
        ctx.enqueue_function[kernel_8, kernel_8](
            out_tensor_8, a_tensor, b_tensor, grid_dim=grid_8, block_dim=block_8
        )
        ctx.synchronize()
        total_ns_8 += perf_counter_ns() - start

    var time_8x8 = Float64(total_ns_8) / Float64(iterations) / 1_000_000.0

    # ========== Reg2x2 Kernel ==========
    var out_tensor_reg = LayoutTensor[
        dtype, Layout.row_major(M, N), MutAnyOrigin
    ](out_reg2x2)
    comptime BLOCK_TILE = TILE_APPLE * 2
    comptime grid_reg = (
        (N + BLOCK_TILE - 1) // BLOCK_TILE,
        (M + BLOCK_TILE - 1) // BLOCK_TILE,
    )
    comptime block_reg = (TILE_APPLE, TILE_APPLE)
    comptime kernel_reg = matmul_apple_reg2x2_kernel[dtype, M, N, K, TILE_APPLE]

    for _ in range(warmup):
        ctx.enqueue_function[kernel_reg, kernel_reg](
            out_tensor_reg,
            a_tensor,
            b_tensor,
            grid_dim=grid_reg,
            block_dim=block_reg,
        )
        ctx.synchronize()

    var total_ns_reg: UInt = 0
    for _ in range(iterations):
        var start = perf_counter_ns()
        ctx.enqueue_function[kernel_reg, kernel_reg](
            out_tensor_reg,
            a_tensor,
            b_tensor,
            grid_dim=grid_reg,
            block_dim=block_reg,
        )
        ctx.synchronize()
        total_ns_reg += perf_counter_ns() - start

    var time_reg2x2 = Float64(total_ns_reg) / Float64(iterations) / 1_000_000.0

    # ========== FP16 Kernel ==========
    var out_tensor_fp16 = LayoutTensor[
        DType.float16, Layout.row_major(M, N), MutAnyOrigin
    ](out_fp16)
    var a_tensor_fp16 = LayoutTensor[
        DType.float16, Layout.row_major(M, K), ImmutAnyOrigin
    ](a_fp16)
    var b_tensor_fp16 = LayoutTensor[
        DType.float16, Layout.row_major(K, N), ImmutAnyOrigin
    ](b_fp16)

    comptime kernel_fp16 = matmul_fp16_apple_kernel[M, N, K, TILE_APPLE]

    for _ in range(warmup):
        ctx.enqueue_function[kernel_fp16, kernel_fp16](
            out_tensor_fp16,
            a_tensor_fp16,
            b_tensor_fp16,
            grid_dim=grid_8,
            block_dim=block_8,
        )
        ctx.synchronize()

    var total_ns_fp16: UInt = 0
    for _ in range(iterations):
        var start = perf_counter_ns()
        ctx.enqueue_function[kernel_fp16, kernel_fp16](
            out_tensor_fp16,
            a_tensor_fp16,
            b_tensor_fp16,
            grid_dim=grid_8,
            block_dim=block_8,
        )
        ctx.synchronize()
        total_ns_fp16 += perf_counter_ns() - start

    var time_fp16 = Float64(total_ns_fp16) / Float64(iterations) / 1_000_000.0

    # ========== Results ==========
    var ops = 2.0 * Float64(M) * Float64(N) * Float64(K)
    var gflops_16 = ops / (time_16x16 * 1e6)
    var gflops_8 = ops / (time_8x8 * 1e6)
    var gflops_reg = ops / (time_reg2x2 * 1e6)
    var gflops_fp16 = ops / (time_fp16 * 1e6)

    print("Kernel                  | Time (ms) | GFLOPS  | Speedup vs 16x16")
    print("-" * 70)
    print(
        "16x16 Tiled (FP32)      | ",
        time_16x16,
        " | ",
        gflops_16,
        " | 1.00x",
    )
    print(
        "8x8 Apple (FP32)        | ",
        time_8x8,
        " | ",
        gflops_8,
        " | ",
        time_16x16 / time_8x8,
        "x",
    )
    print(
        "8x8 Reg2x2 (FP32)       | ",
        time_reg2x2,
        " | ",
        gflops_reg,
        " | ",
        time_16x16 / time_reg2x2,
        "x",
    )
    print(
        "8x8 Apple (FP16)        | ",
        time_fp16,
        " | ",
        gflops_fp16,
        " | ",
        time_16x16 / time_fp16,
        "x",
    )

    # Verify correctness (sample check)
    print("\nVerifying correctness...")
    var all_match = True
    with out_16x16.map_to_host() as h1, out_8x8.map_to_host() as h2:
        for i in range(min(100, M * N)):
            var diff = abs(Float64(h1[i]) - Float64(h2[i]))
            if diff > 1e-3:
                print("Mismatch at ", i, ": ", h1[i], " vs ", h2[i])
                all_match = False
                break
    if all_match:
        print("FP32 results match!")

    # Check FP16 (with larger tolerance)
    with out_16x16.map_to_host() as h32, out_fp16.map_to_host() as h16:
        for i in range(min(100, M * N)):
            var diff = abs(Float64(h32[i]) - Float64(h16[i]))
            if diff > 0.1:  # Larger tolerance for FP16
                print("FP16 mismatch at ", i, ": ", h32[i], " vs ", h16[i])
                all_match = False
                break
    if all_match:
        print("FP16 results match (within tolerance)!")


def main() raises:
    print("=" * 70)
    print("          MATMUL BENCHMARK ON APPLE SILICON")
    print("=" * 70)

    with DeviceContext() as ctx:
        # Small matrices (256x256)
        benchmark_size[256, 256, 256](ctx)

        # Medium matrices (512x512)
        benchmark_size[512, 512, 512](ctx)

        # Large matrices (1024x1024)
        benchmark_size[1024, 1024, 1024](ctx)

        # Very large matrices (2048x2048)
        benchmark_size[2048, 2048, 2048](ctx)

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)

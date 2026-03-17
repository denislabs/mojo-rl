"""POC: Benchmarking conv2D backward approaches.

The backward pass (dW, dx) is 87% of GPU time in DQN CNN training.

dW computation is a matmul: dW = masked_grad.T @ col
  - masked_grad: (out_channels, BATCH * spatial_out) — grad with act backward applied
  - col:         (BATCH * spatial_out, col_size)      — im2col cache
  - dW:          (out_channels, col_size)

dx computation is: dcol = W.T @ masked_grad, then col2im scatter

This POC tests whether max_matmul can accelerate the dW backward.

Conv layer dimensions (Atari DQN):
  Conv1: [4→32, 8×8, s=4]  dW = (32, 102400) @ (102400, 256) → (32, 256)
  Conv2: [32→64, 4×4, s=2] dW = (64, 20736)  @ (20736, 512)  → (64, 512)
  Conv3: [64→64, 3×3, s=1] dW = (64, 12544)  @ (12544, 576)  → (64, 576)

Run:
    pixi run -e nvidia mojo run -I . tests/nn/test_conv2d_backward_poc.mojo
"""

from std.random import seed, random_float64
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.memory import AddressSpace
from std.gpu.primitives import lane_id
from std.sys import is_nvidia_gpu, has_nvidia_gpu_accelerator
from std.gpu.compute.mma import mma
from layout import Layout, LayoutTensor
from std.utils import IndexList

from mojo_rl.nn.constants import (
    dtype,
    TPB,
    MMA_BLOCK_M,
    MMA_BLOCK_N,
    MMA_BLOCK_THREADS,
    MMA_K,
    MMA_M,
    MMA_N,
    MMA_WARPS_M,
    MMA_WARPS_N,
)

from linalg.matmul import matmul as max_matmul


fn main() raises:
    seed(42)

    # ── Test all 3 Atari DQN conv layers ──
    print("=" * 70)
    print("Conv2D Backward dW Benchmark: max_matmul vs custom MMA")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        # Conv1: dW = (32, K) @ (K, 256) where K = BATCH * 400
        benchmark_dW[32, 256, 256 * 400](ctx, "Conv1 [4→32, 8×8, s=4]")
        print()

        # Conv2: dW = (64, K) @ (K, 512) where K = BATCH * 81
        benchmark_dW[64, 512, 256 * 81](ctx, "Conv2 [32→64, 4×4, s=2]")
        print()

        # Conv3: dW = (64, K) @ (K, 576) where K = BATCH * 49
        benchmark_dW[64, 576, 256 * 49](ctx, "Conv3 [64→64, 3×3, s=1]")

    print()
    print("=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


fn benchmark_dW[
    M: Int,  # out_channels
    N: Int,  # col_size
    K: Int,  # BATCH * spatial_out
](ctx: DeviceContext, name: StringLiteral) raises:
    """Benchmark dW = A @ B where A is (M, K), B is (K, N), C is (M, N).

    This represents: dW = masked_grad_reshaped @ col
    where masked_grad is reshaped to (out_channels, BATCH*spatial_out)
    and col is (BATCH*spatial_out, col_size).
    """
    comptime N_ITERS = 100

    var flops = 2.0 * Float64(M) * Float64(N) * Float64(K)

    print(
        "  " + name + ": dW = ("
        + String(M) + ", " + String(K) + ") @ ("
        + String(K) + ", " + String(N) + ") → ("
        + String(M) + ", " + String(N) + ")"
    )

    # Allocate
    var a_buf = ctx.enqueue_create_buffer[dtype](M * K)
    var b_buf = ctx.enqueue_create_buffer[dtype](K * N)
    var c_buf1 = ctx.enqueue_create_buffer[dtype](M * N)
    var c_buf2 = ctx.enqueue_create_buffer[dtype](M * N)

    # Init with random data
    var a_host = ctx.enqueue_create_host_buffer[dtype](M * K)
    var b_host = ctx.enqueue_create_host_buffer[dtype](K * N)
    for i in range(M * K):
        a_host[i] = Scalar[dtype](random_float64() * 0.1)
    for i in range(K * N):
        b_host[i] = Scalar[dtype](random_float64() * 0.1)
    ctx.enqueue_copy(a_buf, a_host)
    ctx.enqueue_copy(b_buf, b_host)
    ctx.enqueue_memset(c_buf1, 0)
    ctx.enqueue_memset(c_buf2, 0)
    ctx.synchronize()

    var a_tensor = LayoutTensor[
        dtype, Layout.row_major(M, K), MutAnyOrigin
    ](a_buf.unsafe_ptr())
    var b_tensor = LayoutTensor[
        dtype, Layout.row_major(K, N), MutAnyOrigin
    ](b_buf.unsafe_ptr())
    var c1_tensor = LayoutTensor[
        dtype, Layout.row_major(M, N), MutAnyOrigin
    ](c_buf1.unsafe_ptr())
    var c2_tensor = LayoutTensor[
        dtype, Layout.row_major(M, N), MutAnyOrigin
    ](c_buf2.unsafe_ptr())

    # ── Warmup ──
    max_matmul[target="gpu"](c1_tensor, a_tensor, b_tensor, ctx)
    ctx.synchronize()

    # ── Benchmark 1: max_matmul ──
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(N_ITERS):
        max_matmul[target="gpu"](c1_tensor, a_tensor, b_tensor, ctx)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var max_us = Float64(t1 - t0) / 1000.0 / Float64(N_ITERS)
    var max_gflops = flops / (max_us * 1e-6) / 1e9

    print(
        "    max_matmul:     "
        + String(max_us)[:8]
        + " μs  |  "
        + String(max_gflops)[:7]
        + " GFLOPS"
    )

    # ── Benchmark 2: custom MMA kernel (same as backward_dW_kernel_mma) ──
    # This simulates the current dW kernel: tiles over K_TOTAL with MMA blocks
    comptime dW_grid_x = (N + MMA_BLOCK_N - 1) // MMA_BLOCK_N
    comptime dW_grid_y = (M + MMA_BLOCK_M - 1) // MMA_BLOCK_M

    @always_inline
    fn dW_mma_wrapper(
        c: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
        a: LayoutTensor[dtype, Layout.row_major(K, M), MutAnyOrigin],  # transposed for cache.T
        b: LayoutTensor[dtype, Layout.row_major(K, N), MutAnyOrigin],
    ):
        # Simulates backward_dW_kernel_mma: dW = A.T @ B
        # A is (K, M) stored as cache.T, B is (K, N) stored as grad_output
        comptime if is_nvidia_gpu():
            var tid = Int(thread_idx.x)
            var warp_id = tid // 32
            var warp_m = warp_id // MMA_WARPS_N
            var warp_n = warp_id % MMA_WARPS_N

            var block_row = Int(block_idx.y) * MMA_BLOCK_M
            var block_col = Int(block_idx.x) * MMA_BLOCK_N

            var a_smem = LayoutTensor[
                dtype, Layout.row_major(MMA_BLOCK_M, MMA_K), MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ].stack_allocation()
            var b_smem = LayoutTensor[
                dtype, Layout.row_major(MMA_K, MMA_BLOCK_N), MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ].stack_allocation()

            var acc = SIMD[DType.float32, 4](0)
            var lid = lane_id()
            var group_id = lid >> 2
            var group_lane = lid % 4

            comptime num_k_tiles = (K + MMA_K - 1) // MMA_K

            for k_tile in range(num_k_tiles):
                var k_off = k_tile * MMA_K

                var a_r = tid // MMA_K
                var a_c = tid % MMA_K
                if k_off + a_c < K and block_row + a_r < M:
                    a_smem[a_r, a_c] = a[k_off + a_c, block_row + a_r]  # transposed load
                else:
                    a_smem[a_r, a_c] = 0

                var br = tid // MMA_BLOCK_N
                var bc = tid % MMA_BLOCK_N
                if k_off + br < K and block_col + bc < N:
                    b_smem[br, bc] = b[k_off + br, block_col + bc]
                else:
                    b_smem[br, bc] = 0

                barrier()

                var warp_row = warp_m * MMA_M
                var a_frag = SIMD[DType.float32, 4](
                    rebind[Scalar[DType.float32]](a_smem[warp_row + Int(group_id), Int(group_lane)]),
                    rebind[Scalar[DType.float32]](a_smem[warp_row + Int(group_id) + 8, Int(group_lane)]),
                    rebind[Scalar[DType.float32]](a_smem[warp_row + Int(group_id), Int(group_lane) + 4]),
                    rebind[Scalar[DType.float32]](a_smem[warp_row + Int(group_id) + 8, Int(group_lane) + 4]),
                )
                var warp_col = warp_n * MMA_N
                var b_frag = SIMD[DType.float32, 2](
                    rebind[Scalar[DType.float32]](b_smem[Int(group_lane), warp_col + Int(group_id)]),
                    rebind[Scalar[DType.float32]](b_smem[Int(group_lane) + 4, warp_col + Int(group_id)]),
                )

                mma(acc, a_frag, b_frag, acc)
                barrier()

            var r0 = block_row + warp_m * MMA_M + Int(group_id)
            var r1 = r0 + 8
            var c0 = block_col + warp_n * MMA_N + Int(group_lane * 2)
            var c1 = c0 + 1

            if r0 < M and c0 < N:
                c[r0, c0] = rebind[Scalar[dtype]](acc[0])
            if r0 < M and c1 < N:
                c[r0, c1] = rebind[Scalar[dtype]](acc[1])
            if r1 < M and c0 < N:
                c[r1, c0] = rebind[Scalar[dtype]](acc[2])
            if r1 < M and c1 < N:
                c[r1, c1] = rebind[Scalar[dtype]](acc[3])

    # For the MMA kernel, A needs to be in (K, M) layout (transposed)
    # We reinterpret a_buf as (K, M) since data is already there
    var a_transposed = LayoutTensor[
        dtype, Layout.row_major(K, M), MutAnyOrigin
    ](a_buf.unsafe_ptr())

    ctx.enqueue_function[dW_mma_wrapper, dW_mma_wrapper](
        c2_tensor, a_transposed, b_tensor,
        grid_dim=(dW_grid_x, dW_grid_y),
        block_dim=(MMA_BLOCK_THREADS, 1),
    )
    ctx.synchronize()

    ctx.synchronize()
    var t2 = perf_counter_ns()
    for _ in range(N_ITERS):
        ctx.enqueue_function[dW_mma_wrapper, dW_mma_wrapper](
            c2_tensor, a_transposed, b_tensor,
            grid_dim=(dW_grid_x, dW_grid_y),
            block_dim=(MMA_BLOCK_THREADS, 1),
        )
    ctx.synchronize()
    var t3 = perf_counter_ns()
    var mma_us = Float64(t3 - t2) / 1000.0 / Float64(N_ITERS)
    var mma_gflops = flops / (mma_us * 1e-6) / 1e9

    print(
        "    custom MMA:     "
        + String(mma_us)[:8]
        + " μs  |  "
        + String(mma_gflops)[:7]
        + " GFLOPS"
    )

    # Speedup
    var speedup = mma_us / max_us
    print(
        "    >>> max_matmul is "
        + String(speedup)[:5]
        + "x " + ("faster" if speedup > 1.0 else "slower")
    )

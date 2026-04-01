"""Benchmark: Implicit GEMM Conv2D vs current explicit im2col + max_matmul.

Compares NVIDIA forward pass approaches:
  A. Current: explicit im2col → batched_matmul → transpose+bias (3 kernels)
  B. Implicit GEMM 2x2: fused im2col+matmul with 2x2 register tiling (1 kernel)
  C. Implicit GEMM MMA: fused im2col+matmul with tensor core MMA (1 kernel)

All three produce identical output (verified). The implicit GEMM kernels
compute im2col indices on the fly during shared memory tile loading,
eliminating the separate im2col buffer write to global memory.

Run:
    pixi run -e nvidia mojo run -I . tests/nn/test_implicit_gemm_bench.mojo
"""

from std.random import seed, random_float64
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype, TPB, MMA_BLOCK_THREADS
from mojo_rl.nn.autodiff.primitives.conv2d import Conv2D


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def max_abs_diff_host(
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
) -> Float64:
    var mx: Float64 = 0
    for i in range(n):
        var d = abs(Float64(a[i]) - Float64(b[i]))
        if d > mx:
            mx = d
    return mx


# ─────────────────────────────────────────────────────────────────────
# Benchmark harness
# ─────────────────────────────────────────────────────────────────────


def bench_config[
    BATCH: Int,
    IC: Int,
    OC: Int,
    KS: Int,
    STRIDE: Int,
    PAD: Int,
    IN_H: Int,
    IN_W: Int,
    N_ITERS: Int,
    label: StringLiteral,
](ctx: DeviceContext) raises:
    comptime C = Conv2D[IC, OC, KS, STRIDE, PAD, IN_H, IN_W]

    var flops = (
        2.0
        * Float64(BATCH)
        * Float64(OC)
        * Float64(C.out_h)
        * Float64(C.out_w)
        * Float64(IC)
        * Float64(KS)
        * Float64(KS)
    )

    print(
        "── "
        + label
        + ": ["
        + String(BATCH)
        + ", "
        + String(IC)
        + "→"
        + String(OC)
        + ", "
        + String(KS)
        + "×"
        + String(KS)
        + ", s="
        + String(STRIDE)
        + ", p="
        + String(PAD)
        + "] "
        + String(IN_H)
        + "×"
        + String(IN_W)
        + " → "
        + String(C.out_h)
        + "×"
        + String(C.out_w)
        + " ──"
    )

    # ── Allocate ──
    var input_buf = ctx.enqueue_create_buffer[dtype](BATCH * C.IN_DIM)
    var params_buf = ctx.enqueue_create_buffer[dtype](C.PARAM_SIZE)
    var cache_buf = ctx.enqueue_create_buffer[dtype](BATCH * C.CACHE_SIZE)
    var out_buf_current = ctx.enqueue_create_buffer[dtype](BATCH * C.OUT_DIM)
    var out_buf_2x2 = ctx.enqueue_create_buffer[dtype](BATCH * C.OUT_DIM)
    var out_buf_mma = ctx.enqueue_create_buffer[dtype](BATCH * C.OUT_DIM)
    comptime ws_size = max(BATCH * C.OUT_DIM, BATCH * C.CACHE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[dtype](ws_size)

    # ── Fill random data ──
    var input_hb = ctx.enqueue_create_host_buffer[dtype](BATCH * C.IN_DIM)
    var params_hb = ctx.enqueue_create_host_buffer[dtype](C.PARAM_SIZE)
    for i in range(BATCH * C.IN_DIM):
        input_hb.unsafe_ptr()[i] = Scalar[dtype](random_float64(-1.0, 1.0).cast[dtype]())
    for i in range(C.PARAM_SIZE):
        params_hb.unsafe_ptr()[i] = Scalar[dtype](random_float64(-0.5, 0.5).cast[dtype]())
    ctx.enqueue_copy(input_buf, input_hb)
    ctx.enqueue_copy(params_buf, params_hb)
    ctx.enqueue_memset(cache_buf, 0)
    ctx.enqueue_memset(out_buf_current, 0)
    ctx.enqueue_memset(out_buf_2x2, 0)
    ctx.enqueue_memset(out_buf_mma, 0)
    ctx.enqueue_memset(workspace_buf, 0)
    ctx.synchronize()

    # ── LayoutTensors ──
    var input_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.IN_DIM), MutAnyOrigin,
    ](input_buf.unsafe_ptr())
    var input_immut = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.IN_DIM), ImmutAnyOrigin,
    ](input_buf.unsafe_ptr())
    var params_lt = LayoutTensor[
        dtype, Layout.row_major(C.PARAM_SIZE), MutAnyOrigin,
    ](params_buf.unsafe_ptr())
    var params_immut = LayoutTensor[
        dtype, Layout.row_major(C.PARAM_SIZE), ImmutAnyOrigin,
    ](params_buf.unsafe_ptr())
    var cache_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.CACHE_SIZE), MutAnyOrigin,
    ](cache_buf.unsafe_ptr())
    var out_current = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.OUT_DIM), MutAnyOrigin,
    ](out_buf_current.unsafe_ptr())
    var out_2x2 = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.OUT_DIM), MutAnyOrigin,
    ](out_buf_2x2.unsafe_ptr())
    var out_mma = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.OUT_DIM), MutAnyOrigin,
    ](out_buf_mma.unsafe_ptr())

    # ── Grid dims for implicit GEMM kernels ──
    comptime grid_x = (C.spatial_out + 31) // 32
    comptime grid_y = (C.out_channels + 31) // 32

    # ── 2x2 kernel wrapper (no cache write — forward-only benchmark) ──
    @always_inline
    def wrapper_2x2(
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, C.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, C.IN_DIM), ImmutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(C.PARAM_SIZE), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, C.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        C.eval_kernel_2x2[BATCH](output, input, params, cache)

    # ── MMA kernel wrapper ──
    @always_inline
    def wrapper_mma(
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, C.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, C.IN_DIM), ImmutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(C.PARAM_SIZE), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, C.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        C.eval_kernel_mma[BATCH](output, input, params, cache)

    # ── Warmup ──
    for _ in range(3):
        C.eval_gpu[BATCH](ctx, out_current, input_lt, params_lt, cache_lt, workspace_buf.unsafe_ptr())
        ctx.enqueue_function[wrapper_2x2, wrapper_2x2](
            out_2x2, input_immut, params_immut, cache_lt,
            grid_dim=(grid_x, grid_y, BATCH),
            block_dim=(MMA_BLOCK_THREADS, 1),
        )
        ctx.enqueue_function[wrapper_mma, wrapper_mma](
            out_mma, input_immut, params_immut, cache_lt,
            grid_dim=(grid_x, grid_y, BATCH),
            block_dim=(MMA_BLOCK_THREADS, 1),
        )
    ctx.synchronize()

    # ══════════════════════════════════════════════════════════════
    # Benchmark A: Current (im2col + batched_matmul + transpose+bias)
    # ══════════════════════════════════════════════════════════════
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(N_ITERS):
        C.eval_gpu[BATCH](ctx, out_current, input_lt, params_lt, cache_lt, workspace_buf.unsafe_ptr())
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var current_us = Float64(t1 - t0) / 1000.0 / Float64(N_ITERS)

    # ══════════════════════════════════════════════════════════════
    # Benchmark B: Implicit GEMM 2x2
    # ══════════════════════════════════════════════════════════════
    ctx.synchronize()
    var t2 = perf_counter_ns()
    for _ in range(N_ITERS):
        ctx.enqueue_function[wrapper_2x2, wrapper_2x2](
            out_2x2, input_immut, params_immut, cache_lt,
            grid_dim=(grid_x, grid_y, BATCH),
            block_dim=(MMA_BLOCK_THREADS, 1),
        )
    ctx.synchronize()
    var t3 = perf_counter_ns()
    var ig2x2_us = Float64(t3 - t2) / 1000.0 / Float64(N_ITERS)

    # ══════════════════════════════════════════════════════════════
    # Benchmark C: Implicit GEMM MMA
    # ══════════════════════════════════════════════════════════════
    ctx.synchronize()
    var t4 = perf_counter_ns()
    for _ in range(N_ITERS):
        ctx.enqueue_function[wrapper_mma, wrapper_mma](
            out_mma, input_immut, params_immut, cache_lt,
            grid_dim=(grid_x, grid_y, BATCH),
            block_dim=(MMA_BLOCK_THREADS, 1),
        )
    ctx.synchronize()
    var t5 = perf_counter_ns()
    var igmma_us = Float64(t5 - t4) / 1000.0 / Float64(N_ITERS)

    # ── Verify correctness ──
    var out_current_hb = ctx.enqueue_create_host_buffer[dtype](BATCH * C.OUT_DIM)
    var out_2x2_hb = ctx.enqueue_create_host_buffer[dtype](BATCH * C.OUT_DIM)
    var out_mma_hb = ctx.enqueue_create_host_buffer[dtype](BATCH * C.OUT_DIM)
    ctx.enqueue_copy(out_current_hb, out_buf_current)
    ctx.enqueue_copy(out_2x2_hb, out_buf_2x2)
    ctx.enqueue_copy(out_mma_hb, out_buf_mma)
    ctx.synchronize()

    var diff_2x2 = max_abs_diff_host(
        out_current_hb.unsafe_ptr(), out_2x2_hb.unsafe_ptr(), BATCH * C.OUT_DIM
    )
    var diff_mma = max_abs_diff_host(
        out_current_hb.unsafe_ptr(), out_mma_hb.unsafe_ptr(), BATCH * C.OUT_DIM
    )

    # ── Print results ──
    var fastest = min(current_us, min(ig2x2_us, igmma_us))
    var current_gflops = flops / (current_us * 1e-6) / 1e9
    var ig2x2_gflops = flops / (ig2x2_us * 1e-6) / 1e9
    var igmma_gflops = flops / (igmma_us * 1e-6) / 1e9

    print(
        "  current (im2col+mm): "
        + String(current_us)[byte=:10]
        + " μs  "
        + String(current_gflops)[byte=:8]
        + " GFLOPS  ("
        + String(current_us / fastest)[byte=:5]
        + "x)"
    )
    print(
        "  implicit GEMM 2x2:  "
        + String(ig2x2_us)[byte=:10]
        + " μs  "
        + String(ig2x2_gflops)[byte=:8]
        + " GFLOPS  ("
        + String(ig2x2_us / fastest)[byte=:5]
        + "x)  diff="
        + String(diff_2x2)
    )
    print(
        "  implicit GEMM MMA:  "
        + String(igmma_us)[byte=:10]
        + " μs  "
        + String(igmma_gflops)[byte=:8]
        + " GFLOPS  ("
        + String(igmma_us / fastest)[byte=:5]
        + "x)  diff="
        + String(diff_mma)
    )
    print()


def main() raises:
    seed(42)
    print("=" * 70)
    print("Implicit GEMM Conv2D Benchmark (NVIDIA)")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        bench_config[32, 4, 32, 8, 4, 0, 84, 84, 500, "Atari conv1"](ctx)
        bench_config[32, 32, 64, 4, 2, 0, 20, 20, 500, "Atari conv2"](ctx)
        bench_config[32, 64, 64, 3, 1, 0, 9, 9, 500, "Atari conv3"](ctx)
        bench_config[64, 128, 128, 3, 1, 1, 6, 7, 500, "AZ ConnectFour"](ctx)
        bench_config[64, 64, 64, 3, 1, 1, 3, 3, 500, "AZ TicTacToe"](ctx)
        bench_config[128, 4, 32, 8, 4, 0, 84, 84, 500, "Atari conv1 B=128"](ctx)

    print("=" * 70)
    print("Done!")
    print("=" * 70)

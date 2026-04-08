"""Breakdown: Time each step of the NVIDIA conv2D forward pipeline.

Measures separately:
  1. im2col kernel alone
  2. batched_matmul alone (pre-filled im2col)
  3. transpose+bias kernel alone
  4. Full pipeline (all 3)

This tells us exactly where the time goes and what's worth optimizing.

Run:
    pixi run -e nvidia mojo run -I . tests/nn/test_conv2d_breakdown.mojo
"""

from std.random import seed, random_float64
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor
from layout.tile_tensor import lt_to_tt
from linalg.matmul import matmul as max_matmul
from std.runtime.asyncrt import DeviceContextPtr

from mojo_rl.nn.constants import dtype, TPB, MMA_BLOCK_THREADS
from mojo_rl.nn.autodiff.primitives.conv2d import Conv2D


def bench_breakdown[
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
    comptime K_TOTAL = BATCH * C.spatial_out
    comptime KS2 = KS * KS

    print(
        "── "
        + label
        + ": B="
        + String(BATCH)
        + " ["
        + String(IC)
        + "→"
        + String(OC)
        + ", "
        + String(KS)
        + "×"
        + String(KS)
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
    print(
        "  col_size="
        + String(C.col_size)
        + "  spatial_out="
        + String(C.spatial_out)
        + "  CACHE="
        + String(C.CACHE_SIZE)
        + "  OUT="
        + String(C.OUT_DIM)
    )

    # ── Allocate ──
    var input_buf = ctx.enqueue_create_buffer[dtype](BATCH * C.IN_DIM)
    var params_buf = ctx.enqueue_create_buffer[dtype](C.PARAM_SIZE)
    var cache_buf = ctx.enqueue_create_buffer[dtype](BATCH * C.CACHE_SIZE)
    var out_buf = ctx.enqueue_create_buffer[dtype](BATCH * C.OUT_DIM)
    var ws_buf = ctx.enqueue_create_buffer[dtype](K_TOTAL * C.out_channels)

    # Fill random
    var input_hb = ctx.enqueue_create_host_buffer[dtype](BATCH * C.IN_DIM)
    var params_hb = ctx.enqueue_create_host_buffer[dtype](C.PARAM_SIZE)
    for i in range(BATCH * C.IN_DIM):
        input_hb.unsafe_ptr()[i] = Scalar[dtype](random_float64(-1.0, 1.0).cast[dtype]())
    for i in range(C.PARAM_SIZE):
        params_hb.unsafe_ptr()[i] = Scalar[dtype](random_float64(-0.5, 0.5).cast[dtype]())
    ctx.enqueue_copy(input_buf, input_hb)
    ctx.enqueue_copy(params_buf, params_hb)
    ctx.enqueue_memset(cache_buf, 0)
    ctx.enqueue_memset(out_buf, 0)
    ctx.enqueue_memset(ws_buf, 0)
    ctx.synchronize()

    # ── Tensors ──
    var input_immut = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.IN_DIM), ImmutAnyOrigin,
    ](input_buf.unsafe_ptr())
    var params_lt = LayoutTensor[
        dtype, Layout.row_major(C.PARAM_SIZE), MutAnyOrigin,
    ](params_buf.unsafe_ptr())
    var cache_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.CACHE_SIZE), MutAnyOrigin,
    ](cache_buf.unsafe_ptr())
    var out_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.OUT_DIM), MutAnyOrigin,
    ](out_buf.unsafe_ptr())

    # Matmul tensors
    var col_flat = LayoutTensor[
        dtype, Layout.row_major(K_TOTAL, C.col_size), MutAnyOrigin,
    ](cache_buf.unsafe_ptr())
    var W_mat = LayoutTensor[
        dtype, Layout.row_major(C.out_channels, C.col_size), MutAnyOrigin,
    ](params_buf.unsafe_ptr())
    var out_temp = LayoutTensor[
        dtype, Layout.row_major(K_TOTAL, C.out_channels), MutAnyOrigin,
    ](ws_buf.unsafe_ptr())
    var bias_lt = LayoutTensor[
        dtype, Layout.row_major(C.out_channels), ImmutAnyOrigin,
    ](params_buf.unsafe_ptr() + C.out_channels * C.col_size)

    # ── im2col kernel ──
    comptime im2col_elems = BATCH * C.CACHE_SIZE
    comptime im2col_blocks = (im2col_elems + TPB - 1) // TPB

    @always_inline
    def im2col_kernel(
        cache_out: LayoutTensor[
            dtype, Layout.row_major(BATCH, C.CACHE_SIZE), MutAnyOrigin,
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, C.IN_DIM), ImmutAnyOrigin,
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= im2col_elems:
            return
        var b = idx // C.CACHE_SIZE
        var pos = idx % C.CACHE_SIZE
        var s = pos // C.col_size
        var k = pos % C.col_size
        var oh = s // C.out_w
        var ow = s % C.out_w
        var ch = k // KS2
        var rem_k = k % KS2
        var kh = rem_k // KS
        var kw = rem_k % KS
        var ih = oh * STRIDE - PAD + kh
        var iw = ow * STRIDE - PAD + kw
        var val: Scalar[dtype] = 0
        if ih >= 0 and ih < IN_H and iw >= 0 and iw < IN_W:
            val = rebind[Scalar[dtype]](
                input[b, ch * IN_H * IN_W + ih * IN_W + iw]
            )
        cache_out[b, pos] = val

    # ── transpose+bias kernel ──
    comptime out_elems = BATCH * C.OUT_DIM
    comptime out_blocks = (out_elems + TPB - 1) // TPB

    @always_inline
    def transpose_bias_kernel(
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, C.OUT_DIM), MutAnyOrigin,
        ],
        temp: LayoutTensor[
            dtype, Layout.row_major(K_TOTAL, C.out_channels), MutAnyOrigin,
        ],
        bias: LayoutTensor[
            dtype, Layout.row_major(C.out_channels), ImmutAnyOrigin,
        ],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= out_elems:
            return
        var b = idx // C.OUT_DIM
        var out_pos = idx % C.OUT_DIM
        var oc = out_pos // C.spatial_out
        var s = out_pos % C.spatial_out
        output[b, out_pos] = rebind[Scalar[dtype]](
            temp[b * C.spatial_out + s, oc]
        ) + rebind[Scalar[dtype]](bias[oc])

    # ── Warmup all paths ──
    for _ in range(5):
        ctx.enqueue_function[im2col_kernel, im2col_kernel](
            cache_lt, input_immut,
            grid_dim=(im2col_blocks,), block_dim=(TPB,),
        )
        max_matmul[target="gpu", transpose_b=True](lt_to_tt(out_temp), lt_to_tt(col_flat), lt_to_tt(W_mat), DeviceContextPtr(ctx))
        ctx.enqueue_function[transpose_bias_kernel, transpose_bias_kernel](
            out_lt, out_temp, bias_lt,
            grid_dim=(out_blocks,), block_dim=(TPB,),
        )
    ctx.synchronize()

    # ════════════════════════════════════════════════════════════
    # Step 1: im2col only
    # ════════════════════════════════════════════════════════════
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(N_ITERS):
        ctx.enqueue_function[im2col_kernel, im2col_kernel](
            cache_lt, input_immut,
            grid_dim=(im2col_blocks,), block_dim=(TPB,),
        )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var im2col_us = Float64(t1 - t0) / 1000.0 / Float64(N_ITERS)

    # ════════════════════════════════════════════════════════════
    # Step 2: batched_matmul only (im2col already filled)
    # ════════════════════════════════════════════════════════════
    ctx.synchronize()
    var t2 = perf_counter_ns()
    for _ in range(N_ITERS):
        max_matmul[target="gpu", transpose_b=True](lt_to_tt(out_temp), lt_to_tt(col_flat), lt_to_tt(W_mat), DeviceContextPtr(ctx))
    ctx.synchronize()
    var t3 = perf_counter_ns()
    var matmul_us = Float64(t3 - t2) / 1000.0 / Float64(N_ITERS)

    # ════════════════════════════════════════════════════════════
    # Step 3: transpose+bias only
    # ════════════════════════════════════════════════════════════
    ctx.synchronize()
    var t4 = perf_counter_ns()
    for _ in range(N_ITERS):
        ctx.enqueue_function[transpose_bias_kernel, transpose_bias_kernel](
            out_lt, out_temp, bias_lt,
            grid_dim=(out_blocks,), block_dim=(TPB,),
        )
    ctx.synchronize()
    var t5 = perf_counter_ns()
    var transpose_us = Float64(t5 - t4) / 1000.0 / Float64(N_ITERS)

    # ════════════════════════════════════════════════════════════
    # Full pipeline
    # ════════════════════════════════════════════════════════════
    ctx.synchronize()
    var t6 = perf_counter_ns()
    for _ in range(N_ITERS):
        ctx.enqueue_function[im2col_kernel, im2col_kernel](
            cache_lt, input_immut,
            grid_dim=(im2col_blocks,), block_dim=(TPB,),
        )
        max_matmul[target="gpu", transpose_b=True](lt_to_tt(out_temp), lt_to_tt(col_flat), lt_to_tt(W_mat), DeviceContextPtr(ctx))
        ctx.enqueue_function[transpose_bias_kernel, transpose_bias_kernel](
            out_lt, out_temp, bias_lt,
            grid_dim=(out_blocks,), block_dim=(TPB,),
        )
    ctx.synchronize()
    var t7 = perf_counter_ns()
    var full_us = Float64(t7 - t6) / 1000.0 / Float64(N_ITERS)

    var sum_parts = im2col_us + matmul_us + transpose_us

    # ── im2col bandwidth stats ──
    # im2col writes BATCH * CACHE_SIZE floats + reads ~same from input
    var im2col_bytes = Float64(BATCH * C.CACHE_SIZE) * 4.0 * 2.0  # read + write
    var im2col_bw = im2col_bytes / (im2col_us * 1e-6) / 1e9  # GB/s

    # transpose reads+writes BATCH * OUT_DIM + reads OC bias
    var transpose_bytes = Float64(BATCH * C.OUT_DIM) * 4.0 * 2.0
    var transpose_bw = transpose_bytes / (transpose_us * 1e-6) / 1e9

    print(
        "  im2col:        "
        + String(im2col_us)[byte=:8]
        + " μs  ("
        + String(im2col_us / full_us * 100.0)[byte=:5]
        + "%)  "
        + String(im2col_bw)[byte=:6]
        + " GB/s"
    )
    print(
        "  matmul:        "
        + String(matmul_us)[byte=:8]
        + " μs  ("
        + String(matmul_us / full_us * 100.0)[byte=:5]
        + "%)"
    )
    print(
        "  transpose+b:   "
        + String(transpose_us)[byte=:8]
        + " μs  ("
        + String(transpose_us / full_us * 100.0)[byte=:5]
        + "%)  "
        + String(transpose_bw)[byte=:6]
        + " GB/s"
    )
    print(
        "  sum of parts:  "
        + String(sum_parts)[byte=:8]
        + " μs"
    )
    print(
        "  full pipeline:  "
        + String(full_us)[byte=:8]
        + " μs  (overhead: "
        + String((full_us - sum_parts) / full_us * 100.0)[byte=:5]
        + "%)"
    )
    print()


def main() raises:
    seed(42)
    print("=" * 70)
    print("Conv2D Forward Pipeline Breakdown (NVIDIA)")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        bench_breakdown[32, 4, 32, 8, 4, 0, 84, 84, 1000, "Atari conv1"](ctx)
        bench_breakdown[32, 32, 64, 4, 2, 0, 20, 20, 1000, "Atari conv2"](ctx)
        bench_breakdown[32, 64, 64, 3, 1, 0, 9, 9, 1000, "Atari conv3"](ctx)
        bench_breakdown[64, 128, 128, 3, 1, 1, 6, 7, 1000, "AZ ConnectFour"](ctx)
        bench_breakdown[64, 64, 64, 3, 1, 1, 3, 3, 1000, "AZ TicTacToe"](ctx)
        bench_breakdown[128, 4, 32, 8, 4, 0, 84, 84, 1000, "Atari conv1 B=128"](ctx)

    print("=" * 70)

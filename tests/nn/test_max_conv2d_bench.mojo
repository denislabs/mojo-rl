"""Benchmark: Max Kernels conv2d vs Our Conv2D across multiple configurations.

Tests configurations used by different agents:
  1-3. Atari conv layers at B=32 (DQN, Rainbow, PPO CNN, MuZero)
  4.   AlphaZero ConnectFour ResBlock [128→128, 3×3, s=1, p=1] 6×7 at B=64
  5.   AlphaZero TicTacToe [64→64, 3×3, s=1, p=1] 3×3 at B=64
  6.   Atari conv1 at B=128
  7-12. AlphaZero ConnectFour at B=256/512/1024 (production batch sizes)
        - Initial conv (3→128), ResBlock conv (128→128), last conv (128→128 p=0)
        - nsys profiling shows conv kernels take ~50% of GPU time at B=512

Compares:
  A. conv_gpu (Max's auto-dispatching, cuDNN-backed)
  B. conv2d_gpu_naive_nhwc_rscf (Max's pure Mojo naive kernel)
  C. Our Conv2D (im2col + max_matmul via batched_matmul)

Run:
    pixi run -e nvidia mojo run -I . tests/nn/test_max_conv2d_bench.mojo
"""

from std.random import seed, random_float64
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor
from std.utils import IndexList

from mojo_rl.nn.constants import dtype, TPB, MMA_BLOCK_THREADS
from mojo_rl.nn.autodiff.primitives.conv2d import Conv2D
from layout.tile_tensor import lt_to_tt

# Max conv kernels
from nn.conv.conv import conv_gpu, conv2d_gpu_naive_nhwc_rscf


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
    comptime OUT_H = (IN_H + 2 * PAD - KS) // STRIDE + 1
    comptime OUT_W = (IN_W + 2 * PAD - KS) // STRIDE + 1
    comptime OurConv = Conv2D[IC, OC, KS, STRIDE, PAD, IN_H, IN_W]

    var flops = (
        2.0
        * Float64(BATCH)
        * Float64(OC)
        * Float64(OUT_H)
        * Float64(OUT_W)
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
        + String(OUT_H)
        + "×"
        + String(OUT_W)
        + " ──"
    )
    print(
        "  FLOPs: "
        + String(flops / 1e6)[byte=:8]
        + " MFLOP  |  col_size="
        + String(OurConv.col_size)
        + "  spatial_out="
        + String(OurConv.spatial_out)
    )

    # ── Allocate Max (NHWC) buffers ──
    var input_buf_nhwc = ctx.enqueue_create_buffer[dtype](
        BATCH * IN_H * IN_W * IC
    )
    var filter_buf_rscf = ctx.enqueue_create_buffer[dtype](
        KS * KS * IC * OC
    )
    var out_buf_max = ctx.enqueue_create_buffer[dtype](
        BATCH * OUT_H * OUT_W * OC
    )
    var out_buf_naive = ctx.enqueue_create_buffer[dtype](
        BATCH * OUT_H * OUT_W * OC
    )

    # ── Allocate Our (flattened CHW) buffers ──
    var input_buf_chw = ctx.enqueue_create_buffer[dtype](
        BATCH * OurConv.IN_DIM
    )
    var params_buf = ctx.enqueue_create_buffer[dtype](OurConv.PARAM_SIZE)
    var cache_buf = ctx.enqueue_create_buffer[dtype](
        BATCH * OurConv.CACHE_SIZE
    )
    var out_buf_ours = ctx.enqueue_create_buffer[dtype](
        BATCH * OurConv.OUT_DIM
    )

    # ── Initialize random data ──
    var input_host_nhwc = ctx.enqueue_create_host_buffer[dtype](
        BATCH * IN_H * IN_W * IC
    )
    var input_host_chw = ctx.enqueue_create_host_buffer[dtype](
        BATCH * OurConv.IN_DIM
    )
    var filter_host_rscf = ctx.enqueue_create_host_buffer[dtype](
        KS * KS * IC * OC
    )
    var params_host = ctx.enqueue_create_host_buffer[dtype](
        OurConv.PARAM_SIZE
    )

    for i in range(BATCH * IN_H * IN_W * IC):
        input_host_nhwc[i] = Scalar[dtype](random_float64() * 0.1)

    # NHWC → CHW conversion
    for b in range(BATCH):
        for h in range(IN_H):
            for w in range(IN_W):
                for c in range(IC):
                    var nhwc_idx = (
                        b * IN_H * IN_W * IC + h * IN_W * IC + w * IC + c
                    )
                    var chw_idx = (
                        b * (IC * IN_H * IN_W)
                        + c * IN_H * IN_W
                        + h * IN_W
                        + w
                    )
                    input_host_chw[chw_idx] = input_host_nhwc[nhwc_idx]

    # Fill filter RSCF
    for i in range(KS * KS * IC * OC):
        filter_host_rscf[i] = Scalar[dtype](random_float64() * 0.1)

    # RSCF → Our W layout
    for oc in range(OC):
        for c in range(IC):
            for kh in range(KS):
                for kw in range(KS):
                    var rscf_idx = (
                        kh * KS * IC * OC + kw * IC * OC + c * OC + oc
                    )
                    var our_idx = (
                        oc * (IC * KS * KS) + c * KS * KS + kh * KS + kw
                    )
                    params_host[our_idx] = filter_host_rscf[rscf_idx]

    # Zero bias
    for i in range(OC):
        params_host[OC * IC * KS * KS + i] = 0

    # Copy to GPU
    ctx.enqueue_copy(input_buf_nhwc, input_host_nhwc)
    ctx.enqueue_copy(input_buf_chw, input_host_chw)
    ctx.enqueue_copy(filter_buf_rscf, filter_host_rscf)
    ctx.enqueue_copy(params_buf, params_host)
    ctx.enqueue_memset(out_buf_max, 0)
    ctx.enqueue_memset(out_buf_naive, 0)
    ctx.enqueue_memset(out_buf_ours, 0)
    ctx.enqueue_memset(cache_buf, 0)

    # Workspace for our Conv2D (used as temp buffer in eval_gpu)
    comptime ws_size = max(BATCH * OurConv.OUT_DIM, BATCH * OurConv.CACHE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[dtype](ws_size)
    ctx.enqueue_memset(workspace_buf, 0)
    ctx.synchronize()

    # ── LayoutTensors ──
    var input_nhwc = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_H, IN_W, IC), MutAnyOrigin,
    ](input_buf_nhwc.unsafe_ptr())
    var filter_rscf = LayoutTensor[
        dtype, Layout.row_major(KS, KS, IC, OC), MutAnyOrigin,
    ](filter_buf_rscf.unsafe_ptr())
    var out_max = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_H, OUT_W, OC), MutAnyOrigin,
    ](out_buf_max.unsafe_ptr())
    var out_naive_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_H, OUT_W, OC), MutAnyOrigin,
    ](out_buf_naive.unsafe_ptr())
    var input_chw = LayoutTensor[
        dtype, Layout.row_major(BATCH, OurConv.IN_DIM), MutAnyOrigin,
    ](input_buf_chw.unsafe_ptr())
    var params_t = LayoutTensor[
        dtype, Layout.row_major(OurConv.PARAM_SIZE), MutAnyOrigin,
    ](params_buf.unsafe_ptr())
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OurConv.CACHE_SIZE), MutAnyOrigin,
    ](cache_buf.unsafe_ptr())
    var out_ours = LayoutTensor[
        dtype, Layout.row_major(BATCH, OurConv.OUT_DIM), MutAnyOrigin,
    ](out_buf_ours.unsafe_ptr())

    # ── Naive kernel wrapper ──
    comptime BS = 16
    comptime grid_x_naive = (OUT_W + BS - 1) // BS
    comptime grid_y_naive = (OUT_H + BS - 1) // BS

    @always_inline
    def naive_kernel_wrapper(
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, IN_H, IN_W, IC), MutAnyOrigin
        ],
        filter: LayoutTensor[
            dtype, Layout.row_major(KS, KS, IC, OC), MutAnyOrigin
        ],
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_H, OUT_W, OC), MutAnyOrigin
        ],
    ):
        conv2d_gpu_naive_nhwc_rscf[block_size=BS, maybe_epilogue_func=None](
            input,
            filter,
            output,
            IndexList[2](STRIDE, STRIDE),
            IndexList[2](1, 1),
            IndexList[2](PAD, PAD),
            num_groups=1,
        )

    # ── Warmup (3 iterations each) ──
    for _ in range(3):
        conv_gpu(
            lt_to_tt(input_nhwc), lt_to_tt(filter_rscf), lt_to_tt(out_max),
            IndexList[2](STRIDE, STRIDE),
            IndexList[2](1, 1),
            IndexList[4](PAD, PAD, PAD, PAD),
            num_groups=1, ctx=ctx,
        )
        ctx.enqueue_function[naive_kernel_wrapper, naive_kernel_wrapper](
            input_nhwc, filter_rscf, out_naive_lt,
            grid_dim=(grid_x_naive, grid_y_naive, BATCH),
            block_dim=(BS, BS),
        )
        OurConv.eval_gpu[BATCH](ctx, out_ours, input_chw, params_t, cache_t, workspace_buf.unsafe_ptr())
    ctx.synchronize()

    # ══════════════════════════════════════════════════════════════
    # Benchmark A: conv_gpu (cuDNN-backed)
    # ══════════════════════════════════════════════════════════════
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(N_ITERS):
        conv_gpu(
            lt_to_tt(input_nhwc), lt_to_tt(filter_rscf), lt_to_tt(out_max),
            IndexList[2](STRIDE, STRIDE),
            IndexList[2](1, 1),
            IndexList[4](PAD, PAD, PAD, PAD),
            num_groups=1, ctx=ctx,
        )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var max_us = Float64(t1 - t0) / 1000.0 / Float64(N_ITERS)
    var max_gflops = flops / (max_us * 1e-6) / 1e9

    # ══════════════════════════════════════════════════════════════
    # Benchmark B: conv2d_gpu_naive
    # ══════════════════════════════════════════════════════════════
    ctx.synchronize()
    var t2 = perf_counter_ns()
    for _ in range(N_ITERS):
        ctx.enqueue_function[naive_kernel_wrapper, naive_kernel_wrapper](
            input_nhwc, filter_rscf, out_naive_lt,
            grid_dim=(grid_x_naive, grid_y_naive, BATCH),
            block_dim=(BS, BS),
        )
    ctx.synchronize()
    var t3 = perf_counter_ns()
    var naive_us = Float64(t3 - t2) / 1000.0 / Float64(N_ITERS)
    var naive_gflops = flops / (naive_us * 1e-6) / 1e9

    # ══════════════════════════════════════════════════════════════
    # Benchmark C: Our Conv2D (im2col + batched_matmul)
    # ══════════════════════════════════════════════════════════════
    ctx.synchronize()
    var t4 = perf_counter_ns()
    for _ in range(N_ITERS):
        OurConv.eval_gpu[BATCH](ctx, out_ours, input_chw, params_t, cache_t, workspace_buf.unsafe_ptr())
    ctx.synchronize()
    var t5 = perf_counter_ns()
    var ours_us = Float64(t5 - t4) / 1000.0 / Float64(N_ITERS)
    var ours_gflops = flops / (ours_us * 1e-6) / 1e9

    # ── Verify correctness ──
    var out_max_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * OUT_H * OUT_W * OC
    )
    var out_ours_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * OurConv.OUT_DIM
    )
    ctx.enqueue_copy(out_max_host, out_buf_max)
    ctx.enqueue_copy(out_ours_host, out_buf_ours)
    ctx.synchronize()

    # Compare NHWC (Max) vs CHW (ours)
    var max_diff: Float64 = 0.0
    for b in range(min(BATCH, 4)):
        for oh in range(OUT_H):
            for ow in range(OUT_W):
                for oc in range(OC):
                    var nhwc_idx = (
                        b * OUT_H * OUT_W * OC
                        + oh * OUT_W * OC
                        + ow * OC
                        + oc
                    )
                    var chw_idx = (
                        b * OurConv.OUT_DIM
                        + oc * OUT_H * OUT_W
                        + oh * OUT_W
                        + ow
                    )
                    var diff = abs(
                        Float64(out_max_host[nhwc_idx])
                        - Float64(out_ours_host[chw_idx])
                    )
                    if diff > max_diff:
                        max_diff = diff

    # ── Print results ──
    var fastest = min(max_us, min(naive_us, ours_us))
    print(
        "  conv_gpu (cuDNN): "
        + String(max_us)[byte=:10]
        + " μs  "
        + String(max_gflops)[byte=:8]
        + " GFLOPS  ("
        + String(max_us / fastest)[byte=:5]
        + "x)"
    )
    print(
        "  naive NHWC:       "
        + String(naive_us)[byte=:10]
        + " μs  "
        + String(naive_gflops)[byte=:8]
        + " GFLOPS  ("
        + String(naive_us / fastest)[byte=:5]
        + "x)"
    )
    print(
        "  ours (im2col+mm): "
        + String(ours_us)[byte=:10]
        + " μs  "
        + String(ours_gflops)[byte=:8]
        + " GFLOPS  ("
        + String(ours_us / fastest)[byte=:5]
        + "x)"
    )
    print(
        "  max diff (conv_gpu vs ours): "
        + String(max_diff)
    )
    print()


def main() raises:
    seed(42)
    print("=" * 70)
    print("Conv2D Benchmark: Max Kernels vs Our Custom (forward only)")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        # ── Config 1: Atari conv1 (DQN, Rainbow, PPO CNN, MuZero) ──
        bench_config[32, 4, 32, 8, 4, 0, 84, 84, 200, "Atari conv1"](ctx)

        # ── Config 2: Atari conv2 ──
        bench_config[32, 32, 64, 4, 2, 0, 20, 20, 200, "Atari conv2"](ctx)

        # ── Config 3: Atari conv3 ──
        bench_config[32, 64, 64, 3, 1, 0, 9, 9, 200, "Atari conv3"](ctx)

        # ── Config 4: AlphaZero ConnectFour ResBlock (F=128) B=64 ──
        bench_config[64, 128, 128, 3, 1, 1, 6, 7, 200, "AZ CF B=64"](ctx)

        # ── Config 5: AlphaZero TicTacToe (small) ──
        bench_config[64, 64, 64, 3, 1, 1, 3, 3, 200, "AZ TicTacToe"](ctx)

        # ── Config 6: Larger batch Atari conv1 (PPO) ──
        bench_config[128, 4, 32, 8, 4, 0, 84, 84, 200, "Atari conv1 B=128"](ctx)

        # ══════════════════════════════════════════════════════════════
        # AlphaZero ConnectFour at B=512 (production batch size)
        # These are the configs that dominate ~50% of GPU time at B=512
        # ══════════════════════════════════════════════════════════════

        # ── Config 7: AZ CF initial conv (3→128, 3×3, p=1, 6×7) ──
        bench_config[512, 3, 128, 3, 1, 1, 6, 7, 200, "AZ CF init B=512"](ctx)

        # ── Config 8: AZ CF ResBlock conv (128→128, 3×3, p=1, 6×7) ──
        bench_config[512, 128, 128, 3, 1, 1, 6, 7, 200, "AZ CF ResBlk B=512"](ctx)

        # ── Config 9: AZ CF last conv (128→128, 3×3, p=0, 6×7 → 4×5) ──
        bench_config[512, 128, 128, 3, 1, 0, 6, 7, 200, "AZ CF last B=512"](ctx)

        # ── Config 10: AZ CF ResBlock at B=512 sim=16 (effective B=8192) ──
        # With 16 MCTS sims, each sim evaluates B=512 positions
        # but network sees batches of 512 at a time
        bench_config[512, 128, 128, 3, 1, 1, 6, 7, 500, "AZ CF ResBlk B=512 x500"](ctx)

        # ── Config 11: AZ CF at B=256 (intermediate) ──
        bench_config[256, 128, 128, 3, 1, 1, 6, 7, 200, "AZ CF ResBlk B=256"](ctx)

        # ── Config 12: AZ CF at B=1024 (upper bound) ──
        bench_config[1024, 128, 128, 3, 1, 1, 6, 7, 200, "AZ CF ResBlk B=1024"](ctx)

    print("=" * 70)
    print("Done!")
    print("=" * 70)

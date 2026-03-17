"""POC: Benchmarking conv2D approaches.

Compares:
  1. nn.conv.conv_gpu (Max's auto-dispatching GPU conv2d)
  2. nn.conv.conv2d_gpu_naive_nhwc_rscf (Max's pure Mojo GPU conv2d)
  3. Our custom Conv2D (im2col + MMA/2x2 tiled matmul)

Layout notes:
  - Max conv expects NHWC input (batch, height, width, channels)
  - Max conv expects RSCF filter (kernel_h, kernel_w, in_channels, out_channels)
  - Our impl uses flattened (BATCH, C*H*W) with im2col

Run:
    pixi run -e nvidia mojo run -I . tests/nn/test_max_conv2d_poc.mojo
"""

from std.random import seed, random_float64
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor
from std.utils import IndexList

from mojo_rl.nn.constants import dtype, TPB, MMA_BLOCK_THREADS
from mojo_rl.nn.autodiff.primitives.conv2d import Conv2D

# Max conv kernels
from nn.conv import conv_gpu, conv2d_gpu_naive_nhwc_rscf


fn main() raises:
    seed(42)

    # ── Dimensions matching Atari DQN first layer: Conv2D[4, 32, 8, 4, 0, 84, 84]
    comptime BATCH = 32
    comptime IC = 4  # input channels (stacked frames)
    comptime OC = 32  # output channels
    comptime KS = 8  # kernel size
    comptime STRIDE = 4
    comptime PAD = 0
    comptime IN_H = 84
    comptime IN_W = 84
    comptime OUT_H = (IN_H + 2 * PAD - KS) // STRIDE + 1  # = 20
    comptime OUT_W = (IN_W + 2 * PAD - KS) // STRIDE + 1  # = 20
    comptime N_ITERS = 100

    # Our Conv2D type
    comptime OurConv = Conv2D[IC, OC, KS, STRIDE, PAD, IN_H, IN_W]

    # FLOPs: 2 * BATCH * OC * OUT_H * OUT_W * IC * KS * KS
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

    print("=" * 60)
    print("Conv2D Benchmark: Max Kernels vs Our Custom")
    print("=" * 60)
    print(
        "Conv2D: ["
        + String(BATCH)
        + ", "
        + String(IC)
        + ", "
        + String(IN_H)
        + ", "
        + String(IN_W)
        + "] kernel="
        + String(KS)
        + " stride="
        + String(STRIDE)
        + " -> ["
        + String(BATCH)
        + ", "
        + String(OC)
        + ", "
        + String(OUT_H)
        + ", "
        + String(OUT_W)
        + "]"
    )
    print("Our Conv2D dims: IN_DIM=" + String(OurConv.IN_DIM) + " OUT_DIM=" + String(OurConv.OUT_DIM))
    print("  PARAM_SIZE=" + String(OurConv.PARAM_SIZE) + " CACHE_SIZE=" + String(OurConv.CACHE_SIZE))
    print("Iterations: " + String(N_ITERS))
    print()

    with DeviceContext() as ctx:
        # ══════════════════════════════════════════════════════════════
        # Allocate for Max kernels (NHWC layout)
        # ══════════════════════════════════════════════════════════════
        var input_buf_nhwc = ctx.enqueue_create_buffer[dtype](
            BATCH * IN_H * IN_W * IC
        )
        var filter_buf_rscf = ctx.enqueue_create_buffer[dtype](
            KS * KS * IC * OC
        )
        var out_buf_gpu = ctx.enqueue_create_buffer[dtype](
            BATCH * OUT_H * OUT_W * OC
        )
        var out_buf_naive = ctx.enqueue_create_buffer[dtype](
            BATCH * OUT_H * OUT_W * OC
        )

        # ══════════════════════════════════════════════════════════════
        # Allocate for our Conv2D (flattened CHW layout)
        # ══════════════════════════════════════════════════════════════
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

        # ── Initialize random data on host ──
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

        # Fill NHWC input with random data
        for i in range(BATCH * IN_H * IN_W * IC):
            input_host_nhwc[i] = Scalar[dtype](random_float64() * 0.1)

        # Convert NHWC -> CHW for our kernel (same data, different layout)
        # NHWC index: b*H*W*C + h*W*C + w*C + c
        # CHW index:  b*(C*H*W) + c*H*W + h*W + w
        for b in range(BATCH):
            for h in range(IN_H):
                for w in range(IN_W):
                    for c in range(IC):
                        var nhwc_idx = b * IN_H * IN_W * IC + h * IN_W * IC + w * IC + c
                        var chw_idx = b * (IC * IN_H * IN_W) + c * IN_H * IN_W + h * IN_W + w
                        input_host_chw[chw_idx] = input_host_nhwc[nhwc_idx]

        # Fill filter: RSCF (KS, KS, IC, OC) for Max
        for i in range(KS * KS * IC * OC):
            filter_host_rscf[i] = Scalar[dtype](random_float64() * 0.1)

        # Convert RSCF -> our layout: W(OC, col_size) where col_size = IC*KS*KS
        # RSCF index: r*S*C*F + s*C*F + c*F + f  (r=kh, s=kw, c=ic, f=oc)
        # Our W index: oc * col_size + (c*KS*KS + kh*KS + kw)
        for oc in range(OC):
            for c in range(IC):
                for kh in range(KS):
                    for kw in range(KS):
                        var rscf_idx = kh * KS * IC * OC + kw * IC * OC + c * OC + oc
                        var our_idx = oc * (IC * KS * KS) + c * KS * KS + kh * KS + kw
                        params_host[our_idx] = filter_host_rscf[rscf_idx]

        # Zero bias in our params
        for i in range(OC):
            params_host[OC * IC * KS * KS + i] = 0

        # Copy to GPU
        ctx.enqueue_copy(input_buf_nhwc, input_host_nhwc)
        ctx.enqueue_copy(input_buf_chw, input_host_chw)
        ctx.enqueue_copy(filter_buf_rscf, filter_host_rscf)
        ctx.enqueue_copy(params_buf, params_host)
        ctx.enqueue_memset(out_buf_gpu, 0)
        ctx.enqueue_memset(out_buf_naive, 0)
        ctx.enqueue_memset(out_buf_ours, 0)
        ctx.enqueue_memset(cache_buf, 0)
        ctx.synchronize()

        # ── Create LayoutTensors ──
        # Max NHWC tensors
        var input_nhwc = LayoutTensor[
            dtype, Layout.row_major(BATCH, IN_H, IN_W, IC), MutAnyOrigin,
        ](input_buf_nhwc.unsafe_ptr())
        var filter_rscf = LayoutTensor[
            dtype, Layout.row_major(KS, KS, IC, OC), MutAnyOrigin,
        ](filter_buf_rscf.unsafe_ptr())
        var out_gpu = LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_H, OUT_W, OC), MutAnyOrigin,
        ](out_buf_gpu.unsafe_ptr())
        var out_naive = LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_H, OUT_W, OC), MutAnyOrigin,
        ](out_buf_naive.unsafe_ptr())

        # Our flattened tensors
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

        # ── Warmup ──
        print("Warming up...")

        # conv_gpu
        conv_gpu(
            input_nhwc, filter_rscf, out_gpu,
            IndexList[2](STRIDE, STRIDE),
            IndexList[2](1, 1),
            IndexList[4](PAD, PAD, PAD, PAD),
            num_groups=1, ctx=ctx,
        )

        # naive kernel
        comptime BS = 16
        comptime grid_x_naive = (OUT_W + BS - 1) // BS
        comptime grid_y_naive = (OUT_H + BS - 1) // BS

        @always_inline
        fn naive_kernel_wrapper(
            input: LayoutTensor[dtype, Layout.row_major(BATCH, IN_H, IN_W, IC), MutAnyOrigin],
            filter: LayoutTensor[dtype, Layout.row_major(KS, KS, IC, OC), MutAnyOrigin],
            output: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_H, OUT_W, OC), MutAnyOrigin],
        ):
            conv2d_gpu_naive_nhwc_rscf[block_size=BS, maybe_epilogue_func=None](
                input, filter, output,
                IndexList[2](STRIDE, STRIDE),
                IndexList[2](1, 1),
                IndexList[2](PAD, PAD),
                num_groups=1,
            )

        ctx.enqueue_function[naive_kernel_wrapper, naive_kernel_wrapper](
            input_nhwc, filter_rscf, out_naive,
            grid_dim=(grid_x_naive, grid_y_naive, BATCH),
            block_dim=(BS, BS),
        )

        # Our Conv2D
        OurConv.eval_gpu[BATCH](ctx, out_ours, input_chw, params_t, cache_t)

        ctx.synchronize()
        print("Warmup done!")
        print()

        # ══════════════════════════════════════════════════════════════
        # Benchmark 1: conv_gpu
        # ══════════════════════════════════════════════════════════════
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_ITERS):
            conv_gpu(
                input_nhwc, filter_rscf, out_gpu,
                IndexList[2](STRIDE, STRIDE),
                IndexList[2](1, 1),
                IndexList[4](PAD, PAD, PAD, PAD),
                num_groups=1, ctx=ctx,
            )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var gpu_us = Float64(t1 - t0) / 1000.0 / Float64(N_ITERS)
        var gpu_gflops = flops / (gpu_us * 1e-6) / 1e9

        print(
            "1. conv_gpu (auto):     "
            + String(gpu_us)[:8]
            + " μs  |  "
            + String(gpu_gflops)[:6]
            + " GFLOPS"
        )

        # ══════════════════════════════════════════════════════════════
        # Benchmark 2: conv2d_gpu_naive
        # ══════════════════════════════════════════════════════════════
        ctx.synchronize()
        var t2 = perf_counter_ns()
        for _ in range(N_ITERS):
            ctx.enqueue_function[naive_kernel_wrapper, naive_kernel_wrapper](
                input_nhwc, filter_rscf, out_naive,
                grid_dim=(grid_x_naive, grid_y_naive, BATCH),
                block_dim=(BS, BS),
            )
        ctx.synchronize()
        var t3 = perf_counter_ns()
        var naive_us = Float64(t3 - t2) / 1000.0 / Float64(N_ITERS)
        var naive_gflops = flops / (naive_us * 1e-6) / 1e9

        print(
            "2. conv2d_gpu_naive:    "
            + String(naive_us)[:8]
            + " μs  |  "
            + String(naive_gflops)[:6]
            + " GFLOPS"
        )

        # ══════════════════════════════════════════════════════════════
        # Benchmark 3: Our Conv2D (im2col + MMA/2x2)
        # ══════════════════════════════════════════════════════════════
        ctx.synchronize()
        var t4 = perf_counter_ns()
        for _ in range(N_ITERS):
            OurConv.eval_gpu[BATCH](ctx, out_ours, input_chw, params_t, cache_t)
        ctx.synchronize()
        var t5 = perf_counter_ns()
        var ours_us = Float64(t5 - t4) / 1000.0 / Float64(N_ITERS)
        var ours_gflops = flops / (ours_us * 1e-6) / 1e9

        print(
            "3. Our Conv2D (MMA):    "
            + String(ours_us)[:8]
            + " μs  |  "
            + String(ours_gflops)[:6]
            + " GFLOPS"
        )

        # ── Verify consistency ──
        print()
        print("Verifying consistency...")

        # Read outputs back
        var out_gpu_host = ctx.enqueue_create_host_buffer[dtype](BATCH * OUT_H * OUT_W * OC)
        var out_ours_host = ctx.enqueue_create_host_buffer[dtype](BATCH * OurConv.OUT_DIM)
        ctx.enqueue_copy(out_gpu_host, out_buf_gpu)
        ctx.enqueue_copy(out_ours_host, out_buf_ours)
        ctx.synchronize()

        # Compare: Max output is NHWC, ours is flattened CHW
        # NHWC: b*OH*OW*OC + oh*OW*OC + ow*OC + oc
        # CHW:  b*OUT_DIM + oc*OH*OW + oh*OW + ow
        var max_diff: Float64 = 0.0
        for b in range(min(BATCH, 4)):
            for oh in range(OUT_H):
                for ow in range(OUT_W):
                    for oc in range(OC):
                        var nhwc_idx = b * OUT_H * OUT_W * OC + oh * OUT_W * OC + ow * OC + oc
                        var chw_idx = b * OurConv.OUT_DIM + oc * OUT_H * OUT_W + oh * OUT_W + ow
                        var diff = abs(Float64(out_gpu_host[nhwc_idx]) - Float64(out_ours_host[chw_idx]))
                        if diff > max_diff:
                            max_diff = diff

        print("Max diff (conv_gpu vs ours, first 4 batches): " + String(max_diff)[:10])

        # Sample values
        print()
        print("Sample values (batch=0, oc=0, oh=0, ow=0..2):")
        for ow in range(3):
            var nhwc_idx = 0 * OUT_H * OUT_W * OC + 0 * OUT_W * OC + ow * OC + 0
            var chw_idx = 0 * OurConv.OUT_DIM + 0 * OUT_H * OUT_W + 0 * OUT_W + ow
            print(
                "  ow=" + String(ow)
                + ": conv_gpu=" + String(Float64(out_gpu_host[nhwc_idx]))[:8]
                + "  ours=" + String(Float64(out_ours_host[chw_idx]))[:8]
            )

        # Summary
        print()
        print("=" * 60)
        var fastest = min(gpu_us, min(naive_us, ours_us))
        print(
            "conv_gpu: "
            + String(gpu_us / fastest)[:5]
            + "x  |  naive: "
            + String(naive_us / fastest)[:5]
            + "x  |  ours: "
            + String(ours_us / fastest)[:5]
            + "x"
        )
        print("=" * 60)

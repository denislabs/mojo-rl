"""POC: Benchmarking conv2D approaches.

Compares:
  1. nn.conv.conv_gpu (Max's GPU conv2d - auto-dispatches cuDNN/naive)
  2. nn.conv.conv_cudnn (cuDNN directly)
  3. Our custom 2x2-tiled kernel (via FusedConv2DActivation forward)

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
from layout import Layout, LayoutTensor
from collections import IndexList

from mojo_rl.nn.constants import dtype

# Max conv kernels
from nn.conv import conv_gpu, conv_cudnn


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
    print("Max Kernels Conv2D Benchmark")
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
    print("Iterations: " + String(N_ITERS))
    print()

    with DeviceContext() as ctx:
        # ── Allocate NHWC tensors (what Max expects) ──
        # Input: (BATCH, IN_H, IN_W, IC) - NHWC
        var input_buf = ctx.enqueue_create_buffer[dtype](BATCH * IN_H * IN_W * IC)
        # Filter: (KS, KS, IC, OC) - RSCF
        var filter_buf = ctx.enqueue_create_buffer[dtype](KS * KS * IC * OC)
        # Output: (BATCH, OUT_H, OUT_W, OC) - NHWC
        var out_buf1 = ctx.enqueue_create_buffer[dtype](
            BATCH * OUT_H * OUT_W * OC
        )
        var out_buf2 = ctx.enqueue_create_buffer[dtype](
            BATCH * OUT_H * OUT_W * OC
        )

        # Initialize with random data
        var input_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH * IN_H * IN_W * IC
        )
        var filter_host = ctx.enqueue_create_host_buffer[dtype](
            KS * KS * IC * OC
        )
        for i in range(BATCH * IN_H * IN_W * IC):
            input_host[i] = Scalar[dtype](random_float64() * 0.1)
        for i in range(KS * KS * IC * OC):
            filter_host[i] = Scalar[dtype](random_float64() * 0.1)
        ctx.enqueue_copy(input_buf, input_host)
        ctx.enqueue_copy(filter_buf, filter_host)
        ctx.enqueue_memset(out_buf1, 0)
        ctx.enqueue_memset(out_buf2, 0)
        ctx.synchronize()

        # ── Create LayoutTensors with NHWC/RSCF layouts ──
        var input_tensor = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, IN_H, IN_W, IC),
            MutAnyOrigin,
        ](input_buf.unsafe_ptr())

        var filter_tensor = LayoutTensor[
            dtype,
            Layout.row_major(KS, KS, IC, OC),
            MutAnyOrigin,
        ](filter_buf.unsafe_ptr())

        var out1_tensor = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, OUT_H, OUT_W, OC),
            MutAnyOrigin,
        ](out_buf1.unsafe_ptr())

        var out2_tensor = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, OUT_H, OUT_W, OC),
            MutAnyOrigin,
        ](out_buf2.unsafe_ptr())

        var stride = IndexList[2](STRIDE, STRIDE)
        var dilation = IndexList[2](1, 1)
        var padding = IndexList[4](PAD, PAD, PAD, PAD)
        var padding_2 = IndexList[2](PAD, PAD)

        # ── Warmup ──
        print("Warming up...")
        conv_gpu[2](
            input_tensor,
            filter_tensor,
            out1_tensor,
            stride,
            dilation,
            padding,
            num_groups=1,
            ctx=ctx,
        )
        conv_cudnn(
            input_tensor,
            filter_tensor,
            out2_tensor,
            stride,
            dilation,
            padding_2,
            num_groups=1,
            ctx=ctx,
        )
        ctx.synchronize()
        print("Warmup done!")
        print()

        # ── Benchmark 1: conv_gpu (auto-dispatching) ──
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_ITERS):
            conv_gpu[2](
                input_tensor,
                filter_tensor,
                out1_tensor,
                stride,
                dilation,
                padding,
                num_groups=1,
                ctx=ctx,
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

        # ── Benchmark 2: conv_cudnn (direct cuDNN) ──
        ctx.synchronize()
        var t2 = perf_counter_ns()
        for _ in range(N_ITERS):
            conv_cudnn(
                input_tensor,
                filter_tensor,
                out2_tensor,
                stride,
                dilation,
                padding_2,
                num_groups=1,
                ctx=ctx,
            )
        ctx.synchronize()
        var t3 = perf_counter_ns()
        var cudnn_us = Float64(t3 - t2) / 1000.0 / Float64(N_ITERS)
        var cudnn_gflops = flops / (cudnn_us * 1e-6) / 1e9

        print(
            "2. conv_cudnn:          "
            + String(cudnn_us)[:8]
            + " μs  |  "
            + String(cudnn_gflops)[:6]
            + " GFLOPS"
        )

        # ── Verify consistency ──
        print()
        print("Verifying consistency...")
        var out1_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH * OUT_H * OUT_W * OC
        )
        var out2_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH * OUT_H * OUT_W * OC
        )
        ctx.enqueue_copy(out1_host, out_buf1)
        ctx.enqueue_copy(out2_host, out_buf2)
        ctx.synchronize()

        var max_diff: Float64 = 0.0
        for i in range(min(1000, BATCH * OUT_H * OUT_W * OC)):
            var diff = abs(Float64(out1_host[i]) - Float64(out2_host[i]))
            if diff > max_diff:
                max_diff = diff
        print(
            "Max abs diff (first 1000 elements): " + String(max_diff)[:10]
        )

        # Show some output values to verify non-zero
        print()
        print("Sample output values (conv_gpu):")
        for i in range(5):
            print("  [" + String(i) + "] = " + String(Float64(out1_host[i])))

        print()
        print("=" * 60)
        if cudnn_us < gpu_us:
            print(
                ">>> cuDNN is faster (ratio: "
                + String(gpu_us / cudnn_us)[:4]
                + "x) <<<"
            )
        else:
            print(
                ">>> conv_gpu is faster (ratio: "
                + String(cudnn_us / gpu_us)[:4]
                + "x) <<<"
            )
        print("=" * 60)

"""POC: Benchmarking conv2D approaches.

Compares:
  1. nn.conv.conv2d_gpu_naive_nhwc_rscf (Max's pure Mojo GPU conv2d)
  2. Our custom 2x2-tiled kernel (baseline reference)

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

from mojo_rl.nn.constants import dtype, TPB

# Max conv kernels (pure Mojo — no cuDNN dependency)
from nn.conv import conv2d_gpu_naive_nhwc_rscf


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
    print("Max Kernels Conv2D Benchmark (no cuDNN)")
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
        var input_buf = ctx.enqueue_create_buffer[dtype](
            BATCH * IN_H * IN_W * IC
        )
        # Filter: (KS, KS, IC, OC) - RSCF
        var filter_buf = ctx.enqueue_create_buffer[dtype](KS * KS * IC * OC)
        # Output: (BATCH, OUT_H, OUT_W, OC) - NHWC
        var out_buf1 = ctx.enqueue_create_buffer[dtype](
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

        # ── Warmup ──
        print("Warming up...")

        # conv2d_gpu_naive_nhwc_rscf kernel thread mapping (from source):
        #   n = block_idx.z
        #   h = block_idx.y * block_dim.y + thread_idx.y
        #   w = block_idx.x * block_dim.x + thread_idx.x
        #   loops over C_out internally
        # So: block_dim = (BS, BS), grid = (ceil(OUT_W/BS), ceil(OUT_H/BS), BATCH)
        comptime BS = 16  # block_size for 2D thread block (16x16 = 256 threads)
        comptime grid_x = (OUT_W + BS - 1) // BS
        comptime grid_y = (OUT_H + BS - 1) // BS

        @always_inline
        fn conv_kernel_wrapper(
            input: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, IN_H, IN_W, IC),
                MutAnyOrigin,
            ],
            filter: LayoutTensor[
                dtype,
                Layout.row_major(KS, KS, IC, OC),
                MutAnyOrigin,
            ],
            output: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, OUT_H, OUT_W, OC),
                MutAnyOrigin,
            ],
        ):
            conv2d_gpu_naive_nhwc_rscf[
                block_size=BS,
                maybe_epilogue_func=None,
            ](
                input,
                filter,
                output,
                IndexList[2](STRIDE, STRIDE),
                IndexList[2](1, 1),
                IndexList[2](PAD, PAD),
                num_groups=1,
            )

        ctx.enqueue_function[conv_kernel_wrapper, conv_kernel_wrapper](
            input_tensor,
            filter_tensor,
            out1_tensor,
            grid_dim=(grid_x, grid_y, BATCH),
            block_dim=(BS, BS),
        )
        ctx.synchronize()
        print("Warmup done!")
        print()

        # ── Benchmark: conv2d_gpu_naive_nhwc_rscf ──
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_ITERS):
            ctx.enqueue_function[conv_kernel_wrapper, conv_kernel_wrapper](
                input_tensor,
                filter_tensor,
                out1_tensor,
                grid_dim=(grid_x, grid_y, BATCH),
                block_dim=(BS, BS),
            )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var naive_us = Float64(t1 - t0) / 1000.0 / Float64(N_ITERS)
        var naive_gflops = flops / (naive_us * 1e-6) / 1e9

        print(
            "1. conv2d_gpu_naive:    "
            + String(naive_us)[:8]
            + " μs  |  "
            + String(naive_gflops)[:6]
            + " GFLOPS"
        )

        # ── Verify output is non-zero ──
        print()
        print("Verifying output...")
        var out1_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH * OUT_H * OUT_W * OC
        )
        ctx.enqueue_copy(out1_host, out_buf1)
        ctx.synchronize()

        print("Sample output values:")
        for i in range(5):
            print("  [" + String(i) + "] = " + String(Float64(out1_host[i])))

        var non_zero = 0
        for i in range(BATCH * OUT_H * OUT_W * OC):
            if Float64(out1_host[i]) != 0.0:
                non_zero += 1
        print(
            "Non-zero outputs: "
            + String(non_zero)
            + " / "
            + String(BATCH * OUT_H * OUT_W * OC)
        )

        print()
        print("=" * 60)
        print("Benchmark complete!")
        print("=" * 60)

"""Benchmark stream-based dispatch for fused ops forward + backward.

Compares:
1. ctx.enqueue_function (eval_gpu + vjp_gpu) — current default
2. compile_function + stream.enqueue_function (on_stream variants)

This measures the real-world impact on fused matmul+bias+relu ops,
which are the hottest path in RL training.

Run with:
    pixi run -e nvidia mojo run -I . benchmarks/benchmark_stream_dispatch.mojo
    pixi run -e apple mojo run -I . benchmarks/benchmark_stream_dispatch.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream
from std.gpu import thread_idx, block_idx, block_dim
from std.sys import has_nvidia_gpu_accelerator
from layout import Layout, LayoutTensor
from std.time import perf_counter_ns
from std.random import random_float64

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.autodiff.fused import (
    FusedMatMulBiasActivation,
    FusedMatMulBias,
    ReLUActivation,
)


def format_time(ns: UInt) -> String:
    var us = Float64(ns) / 1_000.0
    if us < 1000.0:
        return String.write(us) + " us"
    return String.write(us / 1000.0) + " ms"


def benchmark_fused_matmul_bias_relu[
    BATCH: Int, IN_DIM: Int, OUT_DIM: Int
](ctx: DeviceContext) raises:
    """Benchmark FusedMatMulBiasActivation[IN, OUT, ReLU] forward+backward."""
    comptime FusedOp = FusedMatMulBiasActivation[IN_DIM, OUT_DIM, ReLUActivation]
    comptime PARAM_SIZE = FusedOp.PARAM_SIZE
    comptime CACHE_SIZE = FusedOp.CACHE_SIZE

    print("\n" + "=" * 70)
    print(
        "FusedMatMulBiasReLU — BATCH=",
        BATCH,
        " IN=",
        IN_DIM,
        " OUT=",
        OUT_DIM,
    )
    print("  PARAM_SIZE=", PARAM_SIZE, " CACHE_SIZE=", CACHE_SIZE)
    print("=" * 70)

    # Allocate buffers
    var input_buf = ctx.enqueue_create_buffer[dtype](BATCH * IN_DIM)
    var output_buf = ctx.enqueue_create_buffer[dtype](BATCH * OUT_DIM)
    var params_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)
    var cache_buf = ctx.enqueue_create_buffer[dtype](BATCH * CACHE_SIZE)
    var grad_out_buf = ctx.enqueue_create_buffer[dtype](BATCH * OUT_DIM)
    var grad_in_buf = ctx.enqueue_create_buffer[dtype](BATCH * IN_DIM)
    var grad_params_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)

    # Initialize with random data
    with input_buf.map_to_host() as h:
        for i in range(BATCH * IN_DIM):
            h[i] = Scalar[dtype](random_float64(-1.0, 1.0))
    with params_buf.map_to_host() as h:
        for i in range(PARAM_SIZE):
            h[i] = Scalar[dtype](random_float64(-0.1, 0.1))
    with grad_out_buf.map_to_host() as h:
        for i in range(BATCH * OUT_DIM):
            h[i] = Scalar[dtype](random_float64(-1.0, 1.0))
    output_buf.enqueue_fill(Scalar[dtype](0.0))
    cache_buf.enqueue_fill(Scalar[dtype](0.0))
    grad_in_buf.enqueue_fill(Scalar[dtype](0.0))
    grad_params_buf.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    # Create tensor views
    var input_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin
    ](input_buf)
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ](output_buf)
    var params_t = LayoutTensor[
        dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
    ](params_buf)
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin
    ](cache_buf)
    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ](grad_out_buf)
    var grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin
    ](grad_in_buf)
    var grad_params_t = LayoutTensor[
        dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
    ](grad_params_buf)

    # Workspace (unused for this op but required by signature)
    var ws_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin]()

    var warmup = 50
    var iterations = 500

    # =========================================================================
    # Path A: ctx.enqueue_function (eval_gpu + vjp_gpu)
    # =========================================================================

    # Warmup
    for _ in range(warmup):
        FusedOp.eval_gpu[BATCH](ctx, output_t, input_t, params_t, cache_t, ws_ptr)
        FusedOp.vjp_gpu[BATCH](
            ctx, grad_out_t, grad_in_t, params_t, cache_t, grad_params_t, ws_ptr
        )
        ctx.synchronize()

    # Forward only
    var total_fwd_ctx: UInt = 0
    for _ in range(iterations):
        var start = perf_counter_ns()
        FusedOp.eval_gpu[BATCH](ctx, output_t, input_t, params_t, cache_t, ws_ptr)
        ctx.synchronize()
        total_fwd_ctx += perf_counter_ns() - start

    # Backward only
    var total_bwd_ctx: UInt = 0
    for _ in range(iterations):
        var start = perf_counter_ns()
        FusedOp.vjp_gpu[BATCH](
            ctx, grad_out_t, grad_in_t, params_t, cache_t, grad_params_t, ws_ptr
        )
        ctx.synchronize()
        total_bwd_ctx += perf_counter_ns() - start

    # Forward + backward combined
    var total_both_ctx: UInt = 0
    for _ in range(iterations):
        var start = perf_counter_ns()
        FusedOp.eval_gpu[BATCH](ctx, output_t, input_t, params_t, cache_t, ws_ptr)
        FusedOp.vjp_gpu[BATCH](
            ctx, grad_out_t, grad_in_t, params_t, cache_t, grad_params_t, ws_ptr
        )
        ctx.synchronize()
        total_both_ctx += perf_counter_ns() - start

    var avg_fwd_ctx = total_fwd_ctx // UInt(iterations)
    var avg_bwd_ctx = total_bwd_ctx // UInt(iterations)
    var avg_both_ctx = total_both_ctx // UInt(iterations)

    print("\n  --- ctx.enqueue_function (current) ---")
    print("  Forward:           ", format_time(avg_fwd_ctx))
    print("  Backward:          ", format_time(avg_bwd_ctx))
    print("  Forward+Backward:  ", format_time(avg_both_ctx))

    # =========================================================================
    # Path B: compile_function + stream (on_stream variants)
    # =========================================================================

    comptime if has_nvidia_gpu_accelerator():
        var stream = ctx.create_stream()

        # Warmup
        for _ in range(warmup):
            FusedOp.eval_gpu_on_stream[BATCH](
                ctx, stream, output_t, input_t, params_t, cache_t, ws_ptr
            )
            FusedOp.vjp_gpu_on_stream[BATCH](
                ctx, stream, grad_out_t, grad_in_t, params_t, cache_t,
                grad_params_t, ws_ptr,
            )
            ctx.synchronize()

        # Forward only
        var total_fwd_stream: UInt = 0
        for _ in range(iterations):
            var start = perf_counter_ns()
            FusedOp.eval_gpu_on_stream[BATCH](
                ctx, stream, output_t, input_t, params_t, cache_t, ws_ptr
            )
            ctx.synchronize()
            total_fwd_stream += perf_counter_ns() - start

        # Backward only
        var total_bwd_stream: UInt = 0
        for _ in range(iterations):
            var start = perf_counter_ns()
            FusedOp.vjp_gpu_on_stream[BATCH](
                ctx, stream, grad_out_t, grad_in_t, params_t, cache_t,
                grad_params_t, ws_ptr,
            )
            ctx.synchronize()
            total_bwd_stream += perf_counter_ns() - start

        # Forward + backward combined
        var total_both_stream: UInt = 0
        for _ in range(iterations):
            var start = perf_counter_ns()
            FusedOp.eval_gpu_on_stream[BATCH](
                ctx, stream, output_t, input_t, params_t, cache_t, ws_ptr
            )
            FusedOp.vjp_gpu_on_stream[BATCH](
                ctx, stream, grad_out_t, grad_in_t, params_t, cache_t,
                grad_params_t, ws_ptr,
            )
            ctx.synchronize()
            total_both_stream += perf_counter_ns() - start

        var avg_fwd_stream = total_fwd_stream // UInt(iterations)
        var avg_bwd_stream = total_bwd_stream // UInt(iterations)
        var avg_both_stream = total_both_stream // UInt(iterations)

        print("\n  --- compile_function + stream (new) ---")
        print("  Forward:           ", format_time(avg_fwd_stream))
        print("  Backward:          ", format_time(avg_bwd_stream))
        print("  Forward+Backward:  ", format_time(avg_both_stream))

        print("\n  --- Speedup ---")
        if avg_fwd_stream > 0:
            print(
                "  Forward:           ",
                Float64(avg_fwd_ctx) / Float64(avg_fwd_stream),
                "x",
            )
        if avg_bwd_stream > 0:
            print(
                "  Backward:          ",
                Float64(avg_bwd_ctx) / Float64(avg_bwd_stream),
                "x",
            )
        if avg_both_stream > 0:
            print(
                "  Forward+Backward:  ",
                Float64(avg_both_ctx) / Float64(avg_both_stream),
                "x",
            )

        # Kernels per fwd+bwd: 1 fwd + 3 bwd = 4 launches
        # Savings per launch = ctx_avg - stream_avg
        if avg_both_ctx > avg_both_stream:
            var savings_per_step_us = Float64(
                avg_both_ctx - avg_both_stream
            ) / 1000.0
            print(
                "\n  Savings per fwd+bwd step: ",
                savings_per_step_us,
                " us",
            )
            print(
                "  At 1000 train steps:       ",
                savings_per_step_us / 1000.0,
                " ms saved",
            )
    else:
        print(
            "\n  --- compile_function + stream: SKIPPED (Metal:"
            " unimplemented) ---",
        )


def main() raises:
    print("=" * 70)
    print("STREAM DISPATCH BENCHMARK — Fused Op Forward + Backward")
    print("=" * 70)
    print(
        "Compares ctx.enqueue_function vs compile_function + stream dispatch"
    )

    with DeviceContext() as ctx:
        # RL-typical sizes
        benchmark_fused_matmul_bias_relu[64, 128, 256](ctx)  # small MLP layer
        benchmark_fused_matmul_bias_relu[64, 256, 256](ctx)  # hidden layer
        benchmark_fused_matmul_bias_relu[256, 128, 256](ctx)  # larger batch

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

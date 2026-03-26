"""Test: AutoFused backward_gpu_on_stream full path.

Verifies that the stream backward path through AutoFused →
_auto_fused_backward_gpu_on_stream → FusedOp.vjp_gpu_on_stream
compiles and produces correct results.

Run with:
    pixi run -e nvidia mojo run -I . benchmarks/test_autofused_stream.mojo
    pixi run -e apple mojo run -I . benchmarks/test_autofused_stream.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream
from std.sys import has_nvidia_gpu_accelerator
from layout import Layout, LayoutTensor
from std.time import perf_counter_ns
from std.random import random_float64

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.autodiff.auto_fused import AutoFused
from mojo_rl.nn.autodiff.primitives import MatMul, BiasAdd, ReLUOp


def main() raises:
    print("=== Test: AutoFused backward_gpu_on_stream ===\n")

    # Simple 2-layer MLP: MatMul+Bias+ReLU → MatMul+Bias
    comptime Net = AutoFused[
        MatMul[8, 16], BiasAdd[16], ReLUOp[16],
        MatMul[16, 4], BiasAdd[4],
    ]

    comptime BATCH = 32
    comptime IN_DIM = Net.IN_DIM
    comptime OUT_DIM = Net.OUT_DIM
    comptime PARAM_SIZE = Net.PARAM_SIZE
    comptime CACHE_SIZE = Net.CACHE_SIZE
    comptime WS_SIZE = Net.WORKSPACE_SIZE_PER_SAMPLE * BATCH

    print("Net: IN=", IN_DIM, " OUT=", OUT_DIM)
    print("PARAM_SIZE=", PARAM_SIZE, " CACHE_SIZE=", CACHE_SIZE)
    print("WS_SIZE=", WS_SIZE)

    with DeviceContext() as ctx:
        # Allocate buffers
        var input_buf = ctx.enqueue_create_buffer[dtype](BATCH * IN_DIM)
        var output_buf = ctx.enqueue_create_buffer[dtype](BATCH * OUT_DIM)
        var params_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)
        var cache_buf = ctx.enqueue_create_buffer[dtype](BATCH * CACHE_SIZE)
        var grad_out_buf = ctx.enqueue_create_buffer[dtype](BATCH * OUT_DIM)
        var grad_in_buf = ctx.enqueue_create_buffer[dtype](BATCH * IN_DIM)
        var grads_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)
        var ws_buf = ctx.enqueue_create_buffer[dtype](WS_SIZE)

        # Init
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
        grads_buf.enqueue_fill(Scalar[dtype](0.0))
        ws_buf.enqueue_fill(Scalar[dtype](0.0))
        ctx.synchronize()

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
        var grads_t = LayoutTensor[
            dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
        ](grads_buf)

        # --- Forward (ctx path) ---
        Net.forward_gpu[BATCH](
            ctx, output_t, input_t, params_t, cache_t, ws_buf
        )
        ctx.synchronize()
        print("\nForward OK")

        # --- Backward (ctx path) ---
        Net.backward_gpu[BATCH](
            ctx, grad_in_t, grad_out_t, params_t, cache_t, grads_t, ws_buf
        )
        ctx.synchronize()

        # Capture reference grads
        var ref_grads = ctx.enqueue_create_host_buffer[dtype](PARAM_SIZE)
        ctx.enqueue_copy(ref_grads, grads_buf)
        var ref_grad_in = ctx.enqueue_create_host_buffer[dtype](BATCH * IN_DIM)
        ctx.enqueue_copy(ref_grad_in, grad_in_buf)
        ctx.synchronize()
        print("Backward (ctx) OK — grads[0]=", ref_grads[0])

        comptime if has_nvidia_gpu_accelerator():
            # --- Backward (stream path) ---
            # Reset grads
            grads_buf.enqueue_fill(Scalar[dtype](0.0))
            grad_in_buf.enqueue_fill(Scalar[dtype](0.0))
            ctx.synchronize()

            var stream = ctx.create_stream()
            Net.backward_gpu_on_stream[BATCH](
                ctx, stream, grad_in_t, grad_out_t, params_t, cache_t,
                grads_t, ws_buf,
            )
            ctx.synchronize()

            var stream_grads = ctx.enqueue_create_host_buffer[dtype](
                PARAM_SIZE
            )
            ctx.enqueue_copy(stream_grads, grads_buf)
            var stream_grad_in = ctx.enqueue_create_host_buffer[dtype](
                BATCH * IN_DIM
            )
            ctx.enqueue_copy(stream_grad_in, grad_in_buf)
            ctx.synchronize()

            print(
                "Backward (stream) OK — grads[0]=",
                stream_grads[0],
            )

            # Verify results match
            var max_diff_grads = Scalar[dtype](0.0)
            for i in range(PARAM_SIZE):
                var diff = abs(ref_grads[i] - stream_grads[i])
                if diff > max_diff_grads:
                    max_diff_grads = diff

            var max_diff_gi = Scalar[dtype](0.0)
            for i in range(BATCH * IN_DIM):
                var diff = abs(ref_grad_in[i] - stream_grad_in[i])
                if diff > max_diff_gi:
                    max_diff_gi = diff

            print("\nMax diff grads:    ", max_diff_grads)
            print("Max diff grad_in:  ", max_diff_gi)
            if max_diff_grads < 1e-5 and max_diff_gi < 1e-5:
                print("PASS — stream backward matches ctx backward")
            else:
                print("FAIL — results diverge!")

            # --- Benchmark ---
            var warmup = 50
            var iters = 500

            for _ in range(warmup):
                Net.backward_gpu[BATCH](
                    ctx, grad_in_t, grad_out_t, params_t, cache_t, grads_t,
                    ws_buf,
                )
                ctx.synchronize()

            var total_ctx: UInt = 0
            for _ in range(iters):
                var start = perf_counter_ns()
                Net.backward_gpu[BATCH](
                    ctx, grad_in_t, grad_out_t, params_t, cache_t, grads_t,
                    ws_buf,
                )
                ctx.synchronize()
                total_ctx += perf_counter_ns() - start

            for _ in range(warmup):
                Net.backward_gpu_on_stream[BATCH](
                    ctx, stream, grad_in_t, grad_out_t, params_t, cache_t,
                    grads_t, ws_buf,
                )
                ctx.synchronize()

            var total_stream: UInt = 0
            for _ in range(iters):
                var start = perf_counter_ns()
                Net.backward_gpu_on_stream[BATCH](
                    ctx, stream, grad_in_t, grad_out_t, params_t, cache_t,
                    grads_t, ws_buf,
                )
                ctx.synchronize()
                total_stream += perf_counter_ns() - start

            var avg_ctx = Float64(total_ctx // UInt(iters)) / 1000.0
            var avg_stream = Float64(total_stream // UInt(iters)) / 1000.0

            print("\n--- Benchmark: AutoFused 2-layer backward ---")
            print("  ctx.enqueue:  ", avg_ctx, " us")
            print("  stream:       ", avg_stream, " us")
            if avg_stream > 0.0:
                print("  Speedup:      ", avg_ctx / avg_stream, "x")
        else:
            print("\n(Stream test skipped — Metal: unimplemented)")

    print("\n=== DONE ===")

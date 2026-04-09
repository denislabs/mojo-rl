"""Benchmark: MatMul vs FusedMatMulBias backward for the 339μs kernel.

The slow kernel is MatMul[5376, 256].vjp_gpu called from
LinearBatchNormReLU[5376, 256].backward_gpu. This uses hand-written
MMA kernels. Compare against:
  1. MatMul vjp_gpu (hand-written MMA) — the slow one
  2. FusedMatMulBias vjp_gpu (also hand-written MMA)
  3. max_matmul (linalg) for the same matmul dimensions

Run:
    pixi run -e nvidia mojo run -I . tests/nn/test_matmul_vjp_bench.mojo
"""

from std.random import seed, random_float64
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tile_tensor import lt_to_tt
from linalg.matmul import matmul as max_matmul
from std.runtime.asyncrt import DeviceContextPtr

from mojo_rl.nn.constants import dtype, TPB, MMA_BLOCK_THREADS
from mojo_rl.nn.autodiff.primitives.matmul import MatMul
from mojo_rl.nn.autodiff.fused.matmul_bias import FusedMatMulBias


def main() raises:
    seed(42)
    print("=" * 70)
    print("MatMul backward: hand-written MMA vs max_matmul")
    print("=" * 70)
    print()

    comptime BATCH = 64
    comptime IN = 5376
    comptime OUT = 256
    comptime N_ITERS = 1000

    comptime MM = MatMul[IN, OUT]
    comptime FMB = FusedMatMulBias[IN, OUT]

    with DeviceContext() as ctx:
        # Allocate
        var go_buf = ctx.enqueue_create_buffer[dtype](BATCH * OUT)
        var gi_buf = ctx.enqueue_create_buffer[dtype](BATCH * IN)
        var params_buf = ctx.enqueue_create_buffer[dtype](MM.PARAM_SIZE)
        var cache_buf = ctx.enqueue_create_buffer[dtype](BATCH * IN)
        var gp_buf = ctx.enqueue_create_buffer[dtype](MM.PARAM_SIZE)
        var fmb_gp_buf = ctx.enqueue_create_buffer[dtype](FMB.PARAM_SIZE)

        # For max_matmul comparison
        # dx: (BATCH, OUT) @ (OUT, IN) = (BATCH, IN)  → grad_output @ W.T
        # dW: (IN, BATCH) @ (BATCH, OUT) = (IN, OUT)  → cache.T @ grad_output
        var dx_out_buf = ctx.enqueue_create_buffer[dtype](BATCH * IN)
        var dW_out_buf = ctx.enqueue_create_buffer[dtype](IN * OUT)

        # Fill
        var hb = ctx.enqueue_create_host_buffer[dtype](max(BATCH * IN, IN * OUT))
        for i in range(BATCH * OUT):
            hb.unsafe_ptr()[i] = Scalar[dtype](random_float64(-1.0, 1.0).cast[dtype]())
        ctx.enqueue_copy(go_buf, hb)
        for i in range(BATCH * IN):
            hb.unsafe_ptr()[i] = Scalar[dtype](random_float64(-1.0, 1.0).cast[dtype]())
        ctx.enqueue_copy(cache_buf, hb)
        for i in range(MM.PARAM_SIZE):
            hb.unsafe_ptr()[i] = Scalar[dtype](random_float64(-0.5, 0.5).cast[dtype]())
        ctx.enqueue_copy(params_buf, hb)
        ctx.enqueue_memset(gi_buf, 0)
        ctx.enqueue_memset(gp_buf, 0)
        ctx.enqueue_memset(fmb_gp_buf, 0)
        ctx.enqueue_memset(dx_out_buf, 0)
        ctx.enqueue_memset(dW_out_buf, 0)
        ctx.synchronize()

        # Tensors
        var go_lt = LayoutTensor[dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin](go_buf.unsafe_ptr())
        var gi_lt = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](gi_buf.unsafe_ptr())
        var params_lt = LayoutTensor[dtype, Layout.row_major(MM.PARAM_SIZE), MutAnyOrigin](params_buf.unsafe_ptr())
        var cache_lt = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](cache_buf.unsafe_ptr())
        var gp_lt = LayoutTensor[dtype, Layout.row_major(MM.PARAM_SIZE), MutAnyOrigin](gp_buf.unsafe_ptr())
        var fmb_gp_lt = LayoutTensor[dtype, Layout.row_major(FMB.PARAM_SIZE), MutAnyOrigin](fmb_gp_buf.unsafe_ptr())

        # max_matmul tensors
        # dx: grad_output(BATCH, OUT) @ W(IN, OUT).T → (BATCH, IN)
        var go_mm = LayoutTensor[dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin](go_buf.unsafe_ptr())
        var W_mm = LayoutTensor[dtype, Layout.row_major(IN, OUT), MutAnyOrigin](params_buf.unsafe_ptr())
        var dx_mm = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](dx_out_buf.unsafe_ptr())
        # dW: cache.T(IN, BATCH) @ grad_output(BATCH, OUT) → need to reshape
        # Actually: cache(BATCH, IN).T @ grad_output(BATCH, OUT) = (IN, OUT)
        # max_matmul now supports transpose_a
        # Instead: dW.T = grad_output.T @ cache → (OUT, BATCH) @ (BATCH, IN) = (OUT, IN)
        # Then transpose. Or just benchmark dx part.
        var cache_mm = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](cache_buf.unsafe_ptr())
        var dW_mm = LayoutTensor[dtype, Layout.row_major(IN, OUT), MutAnyOrigin](dW_out_buf.unsafe_ptr())

        var ws = UnsafePointer[Scalar[dtype], MutAnyOrigin]()

        # FMB needs separate param/grad buffers (different PARAM_SIZE due to bias)
        var fmb_params_buf = ctx.enqueue_create_buffer[dtype](FMB.PARAM_SIZE)
        ctx.enqueue_memset(fmb_params_buf, 0)
        # Copy W part from params
        var fmb_params_hb = ctx.enqueue_create_host_buffer[dtype](FMB.PARAM_SIZE)
        for i in range(IN * OUT):
            fmb_params_hb.unsafe_ptr()[i] = hb.unsafe_ptr()[i]
        ctx.enqueue_copy(fmb_params_buf, fmb_params_hb)
        ctx.synchronize()
        var fmb_params_lt = LayoutTensor[dtype, Layout.row_major(FMB.PARAM_SIZE), MutAnyOrigin](fmb_params_buf.unsafe_ptr())

        # Warmup
        for _ in range(5):
            MM.vjp_gpu[BATCH](ctx, go_lt, gi_lt, params_lt, cache_lt, gp_lt, ws)
            FMB.vjp_gpu[BATCH](ctx, go_lt, gi_lt, fmb_params_lt, cache_lt, fmb_gp_lt, ws)
            max_matmul[target="gpu", transpose_b=True](lt_to_tt(dx_mm), lt_to_tt(go_mm), lt_to_tt(W_mm), DeviceContextPtr(ctx))
        ctx.synchronize()

        # ── Benchmark 1: MatMul.vjp_gpu (hand-written MMA) ──
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_ITERS):
            MM.vjp_gpu[BATCH](ctx, go_lt, gi_lt, params_lt, cache_lt, gp_lt, ws)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var mm_us = Float64(t1 - t0) / 1000.0 / Float64(N_ITERS)

        # ── Benchmark 2: FusedMatMulBias.vjp_gpu ──
        ctx.synchronize()
        var t2 = perf_counter_ns()
        for _ in range(N_ITERS):
            FMB.vjp_gpu[BATCH](ctx, go_lt, gi_lt, fmb_params_lt, cache_lt, fmb_gp_lt, ws)
        ctx.synchronize()
        var t3 = perf_counter_ns()
        var fmb_us = Float64(t3 - t2) / 1000.0 / Float64(N_ITERS)

        # ── Benchmark 3: max_matmul for dx only ──
        ctx.synchronize()
        var t4 = perf_counter_ns()
        for _ in range(N_ITERS):
            max_matmul[target="gpu", transpose_b=True](lt_to_tt(dx_mm), lt_to_tt(go_mm), lt_to_tt(W_mm), DeviceContextPtr(ctx))
        ctx.synchronize()
        var t5 = perf_counter_ns()
        var linalg_dx_us = Float64(t5 - t4) / 1000.0 / Float64(N_ITERS)

        # ── Benchmark 4: MatMul dx kernel alone ──
        var go_immut = LayoutTensor[dtype, Layout.row_major(BATCH, OUT), ImmutAnyOrigin](go_buf.unsafe_ptr())
        var W_immut = LayoutTensor[dtype, Layout.row_major(IN, OUT), ImmutAnyOrigin](params_buf.unsafe_ptr())

        comptime dx_grid_x = (IN + 31) // 32
        comptime dx_grid_y = (BATCH + 31) // 32

        @always_inline
        def dx_wrapper(
            gi: LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin],
            go: LayoutTensor[dtype, Layout.row_major(BATCH, OUT), ImmutAnyOrigin],
            W: LayoutTensor[dtype, Layout.row_major(IN, OUT), ImmutAnyOrigin],
        ):
            MM.backward_dx_kernel_mma[BATCH](gi, go, W)

        ctx.synchronize()
        var t6 = perf_counter_ns()
        for _ in range(N_ITERS):
            ctx.enqueue_function[dx_wrapper, dx_wrapper](
                gi_lt, go_immut, W_immut,
                grid_dim=(dx_grid_x, dx_grid_y),
                block_dim=(MMA_BLOCK_THREADS, 1),
            )
        ctx.synchronize()
        var t7 = perf_counter_ns()
        var mma_dx_us = Float64(t7 - t6) / 1000.0 / Float64(N_ITERS)

        # ── Benchmark 5: MatMul dW kernel alone ──
        var cache_immut = LayoutTensor[dtype, Layout.row_major(BATCH, IN), ImmutAnyOrigin](cache_buf.unsafe_ptr())
        var dW_lt = LayoutTensor[dtype, Layout.row_major(IN, OUT), MutAnyOrigin](gp_buf.unsafe_ptr())

        comptime dW_grid_x = (OUT + 31) // 32
        comptime dW_grid_y = (IN + 31) // 32

        @always_inline
        def dW_wrapper(
            dW: LayoutTensor[dtype, Layout.row_major(IN, OUT), MutAnyOrigin],
            cache: LayoutTensor[dtype, Layout.row_major(BATCH, IN), ImmutAnyOrigin],
            go: LayoutTensor[dtype, Layout.row_major(BATCH, OUT), ImmutAnyOrigin],
        ):
            MM.backward_dW_kernel_mma[BATCH](dW, cache, go)

        ctx.synchronize()
        var t8 = perf_counter_ns()
        for _ in range(N_ITERS):
            ctx.enqueue_function[dW_wrapper, dW_wrapper](
                dW_lt, cache_immut, go_immut,
                grid_dim=(dW_grid_x, dW_grid_y),
                block_dim=(MMA_BLOCK_THREADS, 1),
            )
        ctx.synchronize()
        var t9 = perf_counter_ns()
        var mma_dW_us = Float64(t9 - t8) / 1000.0 / Float64(N_ITERS)

        print("MatMul[5376, 256] backward, BATCH=64:")
        print()
        print("  MatMul.vjp_gpu (full):       " + String(mm_us)[byte=:10] + " μs")
        print("  FusedMatMulBias.vjp_gpu:     " + String(fmb_us)[byte=:10] + " μs")
        print()
        print("  MMA dx kernel alone:         " + String(mma_dx_us)[byte=:10] + " μs  grid=(" + String(dx_grid_x) + "," + String(dx_grid_y) + ")")
        print("  MMA dW kernel alone:         " + String(mma_dW_us)[byte=:10] + " μs  grid=(" + String(dW_grid_x) + "," + String(dW_grid_y) + ")")
        print("  max_matmul dx only:          " + String(linalg_dx_us)[byte=:10] + " μs")
        print()
        print("  MMA dx+dW sum:               " + String(mma_dx_us + mma_dW_us)[byte=:10] + " μs")
        print("  Speedup (max_matmul dx vs MMA dx): " + String(mma_dx_us / linalg_dx_us)[byte=:5] + "x slower")

    print()
    print("=" * 70)

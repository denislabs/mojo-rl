"""Breakdown: Time each step of the NVIDIA conv2D backward pipeline.

The backward (vjp_gpu) has 6 steps:
  1. transpose_grad: (BATCH, OC*S) → (OC, BATCH*S)
  2. matmul_dW: grad_reshaped @ col_flat = dW  (via linalg_matmul)
  3. transpose_W: W(OC, col_size) → W.T(col_size, OC)
  4. matmul_dX: W.T @ grad_reshaped = dcol  (via linalg_matmul)
  5. col2im: scatter dcol → grad_input (loop over kernel_size²)
  6. db: reduce grad across spatial dims

Run:
    pixi run -e nvidia mojo run -I . tests/nn/test_conv2d_backward_breakdown.mojo
"""

from std.random import seed, random_float64
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor
from layout.tile_tensor import lt_to_tt
from linalg.matmul import matmul as max_matmul

from mojo_rl.nn.constants import dtype, TPB, MMA_BLOCK_THREADS
from mojo_rl.nn.autodiff.primitives.conv2d import Conv2D


def bench_backward[
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
        "  K_TOTAL="
        + String(K_TOTAL)
        + "  col_size="
        + String(C.col_size)
        + "  IN_DIM="
        + String(C.IN_DIM)
        + "  OUT_DIM="
        + String(C.OUT_DIM)
    )

    # ── Allocate ──
    var grad_output_buf = ctx.enqueue_create_buffer[dtype](BATCH * C.OUT_DIM)
    var grad_input_buf = ctx.enqueue_create_buffer[dtype](BATCH * C.IN_DIM)
    var params_buf = ctx.enqueue_create_buffer[dtype](C.PARAM_SIZE)
    var cache_buf = ctx.enqueue_create_buffer[dtype](BATCH * C.CACHE_SIZE)
    var grad_params_buf = ctx.enqueue_create_buffer[dtype](C.PARAM_SIZE)
    # Workspace: dcol (col_size * K_TOTAL) + W.T (col_size * OC)
    comptime ws_size = C.col_size * K_TOTAL + C.col_size * OC
    var ws_buf = ctx.enqueue_create_buffer[dtype](ws_size)

    # Fill random
    var go_hb = ctx.enqueue_create_host_buffer[dtype](BATCH * C.OUT_DIM)
    var params_hb = ctx.enqueue_create_host_buffer[dtype](C.PARAM_SIZE)
    var cache_hb = ctx.enqueue_create_host_buffer[dtype](BATCH * C.CACHE_SIZE)
    for i in range(BATCH * C.OUT_DIM):
        go_hb.unsafe_ptr()[i] = Scalar[dtype](random_float64(-1.0, 1.0).cast[dtype]())
    for i in range(C.PARAM_SIZE):
        params_hb.unsafe_ptr()[i] = Scalar[dtype](random_float64(-0.5, 0.5).cast[dtype]())
    for i in range(BATCH * C.CACHE_SIZE):
        cache_hb.unsafe_ptr()[i] = Scalar[dtype](random_float64(-1.0, 1.0).cast[dtype]())
    ctx.enqueue_copy(grad_output_buf, go_hb)
    ctx.enqueue_copy(params_buf, params_hb)
    ctx.enqueue_copy(cache_buf, cache_hb)
    ctx.enqueue_memset(grad_input_buf, 0)
    ctx.enqueue_memset(grad_params_buf, 0)
    ctx.enqueue_memset(ws_buf, 0)
    ctx.synchronize()

    # ── Tensors ──
    var grad_output_immut = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.OUT_DIM), ImmutAnyOrigin,
    ](grad_output_buf.unsafe_ptr())
    var grad_input_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.IN_DIM), MutAnyOrigin,
    ](grad_input_buf.unsafe_ptr())
    var params_immut = LayoutTensor[
        dtype, Layout.row_major(C.PARAM_SIZE), ImmutAnyOrigin,
    ](params_buf.unsafe_ptr())
    var col_flat = LayoutTensor[
        dtype, Layout.row_major(K_TOTAL, C.col_size), MutAnyOrigin,
    ](cache_buf.unsafe_ptr())
    var dW = LayoutTensor[
        dtype, Layout.row_major(C.out_channels, C.col_size), MutAnyOrigin,
    ](grad_params_buf.unsafe_ptr())

    # grad_reshaped reuses grad_input memory
    var grad_reshaped = LayoutTensor[
        dtype, Layout.row_major(OC, K_TOTAL), MutAnyOrigin,
    ](grad_input_buf.unsafe_ptr())

    # Workspace views
    var dcol = LayoutTensor[
        dtype, Layout.row_major(C.col_size, K_TOTAL), MutAnyOrigin,
    ](ws_buf.unsafe_ptr())
    comptime w_t_offset = C.col_size * K_TOTAL
    var w_t_bwd = LayoutTensor[
        dtype, Layout.row_major(C.col_size, OC), MutAnyOrigin,
    ](ws_buf.unsafe_ptr() + w_t_offset)

    # ── Step 1: transpose_grad ──
    comptime grad_elems = OC * K_TOTAL
    comptime grad_blocks = (grad_elems + TPB - 1) // TPB

    @always_inline
    def transpose_grad_kernel(
        dst: LayoutTensor[dtype, Layout.row_major(OC, K_TOTAL), MutAnyOrigin],
        src: LayoutTensor[dtype, Layout.row_major(BATCH, C.OUT_DIM), ImmutAnyOrigin],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= grad_elems:
            return
        var oc = idx // K_TOTAL
        var bs = idx % K_TOTAL
        var b = bs // C.spatial_out
        var s = bs % C.spatial_out
        dst[oc, bs] = src[b, oc * C.spatial_out + s]

    # ── Step 3: transpose_W ──
    comptime w_elems = OC * C.col_size
    comptime w_blocks = (w_elems + TPB - 1) // TPB

    @always_inline
    def transpose_w_kernel(
        dst: LayoutTensor[dtype, Layout.row_major(C.col_size, OC), MutAnyOrigin],
        src: LayoutTensor[dtype, Layout.row_major(C.PARAM_SIZE), ImmutAnyOrigin],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= w_elems:
            return
        var k = idx // OC
        var oc = idx % OC
        dst[k, oc] = src[oc * C.col_size + k]

    # ── Step 5: col2im_gather ──
    comptime total_dx = BATCH * C.IN_DIM
    comptime grid_dx = (total_dx + TPB - 1) // TPB

    @always_inline
    def col2im_kernel(
        grad_in: LayoutTensor[dtype, Layout.row_major(BATCH, C.IN_DIM), MutAnyOrigin],
        dcol_t: LayoutTensor[dtype, Layout.row_major(C.col_size, K_TOTAL), MutAnyOrigin],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= total_dx:
            return
        var b = idx // C.IN_DIM
        var in_pos = idx % C.IN_DIM
        var c = in_pos // (IN_H * IN_W)
        var rem = in_pos % (IN_H * IN_W)
        var ih = rem // IN_W
        var iw = rem % IN_W

        var acc: Scalar[dtype] = 0
        for kh in range(KS):
            for kw in range(KS):
                var oh_num = ih + PAD - kh
                var ow_num = iw + PAD - kw
                if (
                    oh_num >= 0
                    and oh_num % STRIDE == 0
                    and ow_num >= 0
                    and ow_num % STRIDE == 0
                ):
                    var oh = oh_num // STRIDE
                    var ow = ow_num // STRIDE
                    if oh < C.out_h and ow < C.out_w:
                        var s = oh * C.out_w + ow
                        var c_k = c * KS * KS + kh * KS + kw
                        acc += rebind[Scalar[dtype]](
                            dcol_t[c_k, b * C.spatial_out + s]
                        )
        grad_in[b, in_pos] = acc

    # ── Step 6: db ──
    @always_inline
    def db_kernel(
        db: LayoutTensor[dtype, Layout.row_major(OC), MutAnyOrigin],
        grad_out: LayoutTensor[dtype, Layout.row_major(BATCH, C.OUT_DIM), ImmutAnyOrigin],
    ):
        var oc = Int(block_idx.x)
        if oc >= OC:
            return
        var tid = Int(thread_idx.x)
        var smem = LayoutTensor[
            dtype, Layout.row_major(TPB), MutAnyOrigin,
            address_space = __mlir_attr.`#std.gpu<address_space shared>`,
        ].stack_allocation()

        var local_sum: Scalar[dtype] = 0
        var total_spatial = BATCH * C.spatial_out
        var i = tid
        while i < total_spatial:
            var b = i // C.spatial_out
            var s = i % C.spatial_out
            local_sum += rebind[Scalar[dtype]](
                grad_out[b, oc * C.spatial_out + s]
            )
            i += TPB
        smem[tid] = local_sum

        from std.gpu import barrier
        barrier()

        # Tree reduction
        var stride = TPB // 2
        while stride > 0:
            if tid < stride:
                smem[tid] = rebind[Scalar[dtype]](smem[tid]) + rebind[Scalar[dtype]](smem[tid + stride])
            barrier()
            stride //= 2

        if tid == 0:
            db[oc] = smem[0]

    var db_lt = LayoutTensor[
        dtype, Layout.row_major(OC), MutAnyOrigin,
    ](grad_params_buf.unsafe_ptr() + OC * C.col_size)

    # ── Warmup ──
    for _ in range(5):
        ctx.enqueue_function[transpose_grad_kernel, transpose_grad_kernel](
            grad_reshaped, grad_output_immut,
            grid_dim=(grad_blocks,), block_dim=(TPB,),
        )
        max_matmul[target="gpu"](lt_to_tt(dW), lt_to_tt(grad_reshaped), lt_to_tt(col_flat), ctx)
        ctx.enqueue_function[transpose_w_kernel, transpose_w_kernel](
            w_t_bwd, params_immut,
            grid_dim=(w_blocks,), block_dim=(TPB,),
        )
        max_matmul[target="gpu"](lt_to_tt(dcol), lt_to_tt(w_t_bwd), lt_to_tt(grad_reshaped), ctx)
        ctx.enqueue_function[col2im_kernel, col2im_kernel](
            grad_input_lt, dcol,
            grid_dim=(grid_dx,), block_dim=(TPB,),
        )
        ctx.enqueue_function[db_kernel, db_kernel](
            db_lt, grad_output_immut,
            grid_dim=(OC,), block_dim=(TPB,),
        )
    ctx.synchronize()

    # ════════════════════════════════════════════════════════════
    # Benchmark each step independently
    # ════════════════════════════════════════════════════════════

    # Step 1: transpose_grad
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(N_ITERS):
        ctx.enqueue_function[transpose_grad_kernel, transpose_grad_kernel](
            grad_reshaped, grad_output_immut,
            grid_dim=(grad_blocks,), block_dim=(TPB,),
        )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var tg_us = Float64(t1 - t0) / 1000.0 / Float64(N_ITERS)

    # Step 2: matmul dW
    ctx.synchronize()
    var t2 = perf_counter_ns()
    for _ in range(N_ITERS):
        max_matmul[target="gpu"](lt_to_tt(dW), lt_to_tt(grad_reshaped), lt_to_tt(col_flat), ctx)
    ctx.synchronize()
    var t3 = perf_counter_ns()
    var mm_dw_us = Float64(t3 - t2) / 1000.0 / Float64(N_ITERS)

    # Step 3: transpose W
    ctx.synchronize()
    var t4 = perf_counter_ns()
    for _ in range(N_ITERS):
        ctx.enqueue_function[transpose_w_kernel, transpose_w_kernel](
            w_t_bwd, params_immut,
            grid_dim=(w_blocks,), block_dim=(TPB,),
        )
    ctx.synchronize()
    var t5 = perf_counter_ns()
    var tw_us = Float64(t5 - t4) / 1000.0 / Float64(N_ITERS)

    # Step 4: matmul dX
    ctx.synchronize()
    var t6 = perf_counter_ns()
    for _ in range(N_ITERS):
        max_matmul[target="gpu"](lt_to_tt(dcol), lt_to_tt(w_t_bwd), lt_to_tt(grad_reshaped), ctx)
    ctx.synchronize()
    var t7 = perf_counter_ns()
    var mm_dx_us = Float64(t7 - t6) / 1000.0 / Float64(N_ITERS)

    # Step 5: col2im
    ctx.synchronize()
    var t8 = perf_counter_ns()
    for _ in range(N_ITERS):
        ctx.enqueue_function[col2im_kernel, col2im_kernel](
            grad_input_lt, dcol,
            grid_dim=(grid_dx,), block_dim=(TPB,),
        )
    ctx.synchronize()
    var t9 = perf_counter_ns()
    var c2i_us = Float64(t9 - t8) / 1000.0 / Float64(N_ITERS)

    # Step 6: db
    ctx.synchronize()
    var t10 = perf_counter_ns()
    for _ in range(N_ITERS):
        ctx.enqueue_function[db_kernel, db_kernel](
            db_lt, grad_output_immut,
            grid_dim=(OC,), block_dim=(TPB,),
        )
    ctx.synchronize()
    var t11 = perf_counter_ns()
    var db_us = Float64(t11 - t10) / 1000.0 / Float64(N_ITERS)

    # Full pipeline
    ctx.synchronize()
    var tf0 = perf_counter_ns()
    for _ in range(N_ITERS):
        ctx.enqueue_function[transpose_grad_kernel, transpose_grad_kernel](
            grad_reshaped, grad_output_immut,
            grid_dim=(grad_blocks,), block_dim=(TPB,),
        )
        max_matmul[target="gpu"](lt_to_tt(dW), lt_to_tt(grad_reshaped), lt_to_tt(col_flat), ctx)
        ctx.enqueue_function[transpose_w_kernel, transpose_w_kernel](
            w_t_bwd, params_immut,
            grid_dim=(w_blocks,), block_dim=(TPB,),
        )
        max_matmul[target="gpu"](lt_to_tt(dcol), lt_to_tt(w_t_bwd), lt_to_tt(grad_reshaped), ctx)
        ctx.enqueue_function[col2im_kernel, col2im_kernel](
            grad_input_lt, dcol,
            grid_dim=(grid_dx,), block_dim=(TPB,),
        )
        ctx.enqueue_function[db_kernel, db_kernel](
            db_lt, grad_output_immut,
            grid_dim=(OC,), block_dim=(TPB,),
        )
    ctx.synchronize()
    var tf1 = perf_counter_ns()
    var full_us = Float64(tf1 - tf0) / 1000.0 / Float64(N_ITERS)

    var sum_parts = tg_us + mm_dw_us + tw_us + mm_dx_us + c2i_us + db_us
    var matmul_total = mm_dw_us + mm_dx_us
    var overhead_total = tg_us + tw_us + c2i_us + db_us

    # ── Print ──
    print(
        "  1. transpose_grad:  "
        + String(tg_us)[byte=:8]
        + " μs  ("
        + String(tg_us / full_us * 100.0)[byte=:5]
        + "%)"
    )
    print(
        "  2. matmul dW:       "
        + String(mm_dw_us)[byte=:8]
        + " μs  ("
        + String(mm_dw_us / full_us * 100.0)[byte=:5]
        + "%)"
    )
    print(
        "  3. transpose W:     "
        + String(tw_us)[byte=:8]
        + " μs  ("
        + String(tw_us / full_us * 100.0)[byte=:5]
        + "%)"
    )
    print(
        "  4. matmul dX:       "
        + String(mm_dx_us)[byte=:8]
        + " μs  ("
        + String(mm_dx_us / full_us * 100.0)[byte=:5]
        + "%)"
    )
    print(
        "  5. col2im:          "
        + String(c2i_us)[byte=:8]
        + " μs  ("
        + String(c2i_us / full_us * 100.0)[byte=:5]
        + "%)"
    )
    print(
        "  6. db:              "
        + String(db_us)[byte=:8]
        + " μs  ("
        + String(db_us / full_us * 100.0)[byte=:5]
        + "%)"
    )
    print(
        "  ──────────────────────────────────────"
    )
    print(
        "  matmul total:       "
        + String(matmul_total)[byte=:8]
        + " μs  ("
        + String(matmul_total / full_us * 100.0)[byte=:5]
        + "%)"
    )
    print(
        "  overhead total:     "
        + String(overhead_total)[byte=:8]
        + " μs  ("
        + String(overhead_total / full_us * 100.0)[byte=:5]
        + "%)"
    )
    print(
        "  full pipeline:      "
        + String(full_us)[byte=:8]
        + " μs"
    )
    print()


def main() raises:
    seed(42)
    print("=" * 70)
    print("Conv2D BACKWARD Pipeline Breakdown (NVIDIA)")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        # AlphaZero ConnectFour — the main target from nsys profile
        bench_backward[64, 128, 128, 3, 1, 1, 6, 7, 1000, "AZ ConnectFour"](ctx)

        # Other configs for comparison
        bench_backward[32, 4, 32, 8, 4, 0, 84, 84, 1000, "Atari conv1"](ctx)
        bench_backward[32, 32, 64, 4, 2, 0, 20, 20, 1000, "Atari conv2"](ctx)
        bench_backward[32, 64, 64, 3, 1, 0, 9, 9, 1000, "Atari conv3"](ctx)
        bench_backward[64, 64, 64, 3, 1, 1, 3, 3, 1000, "AZ TicTacToe"](ctx)

    print("=" * 70)

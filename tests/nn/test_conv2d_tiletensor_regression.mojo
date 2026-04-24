"""Regression test: Conv2D eval_gpu/vjp_gpu vs eval_gpu_tt/vjp_gpu_tt.

Runs the existing Apple-path forward + backward kernels and the new
TileTensor variants side-by-side on identical inputs. Diffs output, cache,
grad_input, grad_W, grad_b — all must match to within fp32 rounding.

Run:
    pixi run -e apple mojo run -I . tests/nn/test_conv2d_tiletensor_regression.mojo
"""

from std.random import seed, random_float64
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.autodiff.primitives.conv2d import Conv2D


def max_abs_diff(
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
) -> Scalar[dtype]:
    var mx: Scalar[dtype] = 0
    for i in range(n):
        var d = a[i] - b[i]
        if d < 0:
            d = -d
        if d > mx:
            mx = d
    return mx


def run_regression[
    BATCH: Int,
    IC: Int,
    OC: Int,
    KS: Int,
    STRIDE: Int,
    PAD: Int,
    IN_H: Int,
    IN_W: Int,
](ctx: DeviceContext) raises:
    comptime C = Conv2D[IC, OC, KS, STRIDE, PAD, IN_H, IN_W]
    comptime W_SIZE = OC * C.col_size

    print(
        "── ["
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
        + " batch="
        + String(BATCH)
        + " ──"
    )

    # ── Host-side random inputs ──
    var input_host = ctx.enqueue_create_host_buffer[dtype](BATCH * C.IN_DIM)
    var params_host = ctx.enqueue_create_host_buffer[dtype](C.PARAM_SIZE)
    var grad_out_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * C.OUT_DIM
    )
    for i in range(BATCH * C.IN_DIM):
        input_host.unsafe_ptr()[i] = Scalar[dtype](
            random_float64(-1.0, 1.0).cast[dtype]()
        )
    for i in range(C.PARAM_SIZE):
        params_host.unsafe_ptr()[i] = Scalar[dtype](
            random_float64(-0.5, 0.5).cast[dtype]()
        )
    for i in range(BATCH * C.OUT_DIM):
        grad_out_host.unsafe_ptr()[i] = Scalar[dtype](
            random_float64(-1.0, 1.0).cast[dtype]()
        )

    # ── Shared device buffers ──
    var input_buf = ctx.enqueue_create_buffer[dtype](BATCH * C.IN_DIM)
    var params_buf = ctx.enqueue_create_buffer[dtype](C.PARAM_SIZE)
    var grad_out_buf = ctx.enqueue_create_buffer[dtype](BATCH * C.OUT_DIM)
    comptime ws_size = BATCH * C.OP_WORKSPACE_PER_SAMPLE
    var workspace_buf = ctx.enqueue_create_buffer[dtype](ws_size)
    ctx.enqueue_copy(input_buf, input_host)
    ctx.enqueue_copy(params_buf, params_host)
    ctx.enqueue_copy(grad_out_buf, grad_out_host)
    ctx.enqueue_memset(workspace_buf, 0)

    # ── Per-variant device buffers ──
    var out_old_buf = ctx.enqueue_create_buffer[dtype](BATCH * C.OUT_DIM)
    var out_new_buf = ctx.enqueue_create_buffer[dtype](BATCH * C.OUT_DIM)
    var cache_old_buf = ctx.enqueue_create_buffer[dtype](BATCH * C.CACHE_SIZE)
    var cache_new_buf = ctx.enqueue_create_buffer[dtype](BATCH * C.CACHE_SIZE)
    var grad_in_old_buf = ctx.enqueue_create_buffer[dtype](BATCH * C.IN_DIM)
    var grad_in_new_buf = ctx.enqueue_create_buffer[dtype](BATCH * C.IN_DIM)
    var grad_params_old_buf = ctx.enqueue_create_buffer[dtype](C.PARAM_SIZE)
    var grad_params_new_buf = ctx.enqueue_create_buffer[dtype](C.PARAM_SIZE)
    ctx.enqueue_memset(out_old_buf, 0)
    ctx.enqueue_memset(out_new_buf, 0)
    ctx.enqueue_memset(cache_old_buf, 0)
    ctx.enqueue_memset(cache_new_buf, 0)
    ctx.enqueue_memset(grad_in_old_buf, 0)
    ctx.enqueue_memset(grad_in_new_buf, 0)
    ctx.enqueue_memset(grad_params_old_buf, 0)
    ctx.enqueue_memset(grad_params_new_buf, 0)

    # ── LayoutTensor views ──
    var input_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.IN_DIM), MutAnyOrigin
    ](input_buf)
    var params_lt = LayoutTensor[
        dtype, Layout.row_major(C.PARAM_SIZE), MutAnyOrigin
    ](params_buf)
    var grad_out_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.OUT_DIM), MutAnyOrigin
    ](grad_out_buf)
    var out_old_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.OUT_DIM), MutAnyOrigin
    ](out_old_buf)
    var out_new_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.OUT_DIM), MutAnyOrigin
    ](out_new_buf)
    var cache_old_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.CACHE_SIZE), MutAnyOrigin
    ](cache_old_buf)
    var cache_new_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.CACHE_SIZE), MutAnyOrigin
    ](cache_new_buf)
    var grad_in_old_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.IN_DIM), MutAnyOrigin
    ](grad_in_old_buf)
    var grad_in_new_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.IN_DIM), MutAnyOrigin
    ](grad_in_new_buf)
    var grad_params_old_lt = LayoutTensor[
        dtype, Layout.row_major(C.PARAM_SIZE), MutAnyOrigin
    ](grad_params_old_buf)
    var grad_params_new_lt = LayoutTensor[
        dtype, Layout.row_major(C.PARAM_SIZE), MutAnyOrigin
    ](grad_params_new_buf)

    # ── Forward: eval_gpu vs eval_gpu_tt ──
    C.eval_gpu[BATCH](
        ctx,
        out_old_lt,
        input_lt,
        params_lt,
        cache_old_lt,
        workspace_buf.unsafe_ptr(),
    )
    C.eval_gpu_tt[BATCH](
        ctx,
        out_new_lt,
        input_lt,
        params_lt,
        cache_new_lt,
        workspace_buf.unsafe_ptr(),
    )

    # ── Backward: vjp_gpu vs vjp_gpu_tt ──
    C.vjp_gpu[BATCH](
        ctx,
        grad_out_lt,
        grad_in_old_lt,
        params_lt,
        cache_old_lt,
        grad_params_old_lt,
        workspace_buf.unsafe_ptr(),
    )
    C.vjp_gpu_tt[BATCH](
        ctx,
        grad_out_lt,
        grad_in_new_lt,
        params_lt,
        cache_new_lt,
        grad_params_new_lt,
        workspace_buf.unsafe_ptr(),
    )

    # ── Read back ──
    var out_old_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * C.OUT_DIM
    )
    var out_new_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * C.OUT_DIM
    )
    var cache_old_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * C.CACHE_SIZE
    )
    var cache_new_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * C.CACHE_SIZE
    )
    var gin_old_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * C.IN_DIM
    )
    var gin_new_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * C.IN_DIM
    )
    var gp_old_host = ctx.enqueue_create_host_buffer[dtype](C.PARAM_SIZE)
    var gp_new_host = ctx.enqueue_create_host_buffer[dtype](C.PARAM_SIZE)
    ctx.enqueue_copy(out_old_host, out_old_buf)
    ctx.enqueue_copy(out_new_host, out_new_buf)
    ctx.enqueue_copy(cache_old_host, cache_old_buf)
    ctx.enqueue_copy(cache_new_host, cache_new_buf)
    ctx.enqueue_copy(gin_old_host, grad_in_old_buf)
    ctx.enqueue_copy(gin_new_host, grad_in_new_buf)
    ctx.enqueue_copy(gp_old_host, grad_params_old_buf)
    ctx.enqueue_copy(gp_new_host, grad_params_new_buf)
    ctx.synchronize()

    # ── Diffs ──
    var out_diff = max_abs_diff(
        out_old_host.unsafe_ptr(), out_new_host.unsafe_ptr(), BATCH * C.OUT_DIM
    )
    var cache_diff = max_abs_diff(
        cache_old_host.unsafe_ptr(),
        cache_new_host.unsafe_ptr(),
        BATCH * C.CACHE_SIZE,
    )
    var gin_diff = max_abs_diff(
        gin_old_host.unsafe_ptr(), gin_new_host.unsafe_ptr(), BATCH * C.IN_DIM
    )
    var dW_diff = max_abs_diff(
        gp_old_host.unsafe_ptr(), gp_new_host.unsafe_ptr(), W_SIZE
    )
    var db_diff = max_abs_diff(
        gp_old_host.unsafe_ptr() + W_SIZE, gp_new_host.unsafe_ptr() + W_SIZE, OC
    )

    var fwd_pass = out_diff < 1e-5 and cache_diff < 1e-5
    var bwd_pass = gin_diff < 1e-5 and dW_diff < 1e-5 and db_diff < 1e-5
    print(
        "  fwd: out="
        + String(out_diff)
        + " cache="
        + String(cache_diff)
        + "  "
        + ("PASS" if fwd_pass else "FAIL")
    )
    print(
        "  bwd: grad_in="
        + String(gin_diff)
        + " dW="
        + String(dW_diff)
        + " db="
        + String(db_diff)
        + "  "
        + ("PASS" if bwd_pass else "FAIL")
    )


def main() raises:
    seed(42)
    print("=" * 65)
    print("Conv2D regression: eval_gpu/vjp_gpu vs eval_gpu_tt/vjp_gpu_tt")
    print("=" * 65)

    with DeviceContext() as ctx:
        # Small
        run_regression[4, 1, 2, 3, 1, 1, 6, 6](ctx)
        # Medium
        run_regression[8, 4, 16, 3, 1, 1, 10, 10](ctx)
        # Atari conv1
        run_regression[4, 4, 32, 8, 4, 0, 84, 84](ctx)

    print("=" * 65)

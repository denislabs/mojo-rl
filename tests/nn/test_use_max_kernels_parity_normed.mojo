"""GPU parity test — NormedLinear (phase 4a).

NormedLinear = Linear + LayerNorm + Mish, fused. Validates the
max_matmul + post-kernel-norm pattern. Tolerances are looser than plain
Linear because LayerNorm propagates GEMM noise through the variance.

Tolerance: pass if abs_err < 5e-3 OR rel_err < 1e-2. On Apple, both paths
route to the same kernel — compile-check only.

Usage:
    pixi run -e nvidia mojo run -I . tests/nn/test_use_max_kernels_parity_normed.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import abs
from std.sys import has_nvidia_gpu_accelerator
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Model, NormedLinear


def _init_params(ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int):
    for i in range(n):
        var v = Float64((i * 2654435761) % 1000) / 1000.0 - 0.5
        ptr[i] = Scalar[dtype](v)


def _init_input(ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int):
    for i in range(n):
        ptr[i] = Scalar[dtype](0.1 + Float64(i % 13) / 13.0 * 0.8)


def _init_grad_out(ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int):
    for i in range(n):
        ptr[i] = Scalar[dtype](
            0.5 + Float64(i % 7) / 14.0 - Float64(i % 3) / 6.0
        )


def _compare(
    name: String,
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    abs_tol: Float64,
    rel_tol: Float64,
) -> Int:
    var max_abs: Float64 = 0.0
    var max_rel: Float64 = 0.0
    var fails = 0
    for i in range(n):
        var x = Float64(a[i])
        var y = Float64(b[i])
        var err = abs(x - y)
        var denom = abs(x) + abs(y)
        var rel: Float64 = 0.0
        if denom > 1e-7:
            rel = err / denom
        if err > max_abs:
            max_abs = err
        if rel > max_rel:
            max_rel = rel
        if err > abs_tol and rel > rel_tol:
            fails += 1
            if fails <= 3:
                print(
                    "    ", name, "[", i, "]: off=", x, "on=", y,
                    "abs=", err, "rel=", rel,
                )
    if fails == 0:
        print("  [PASS]", name, ": max_abs=", max_abs, "max_rel=", max_rel)
    else:
        print("  [FAIL]", name, ":", fails, "/", n,
              "max_abs=", max_abs, "max_rel=", max_rel)
    return fails


def parity_check[
    OFF: Model, ON: Model, BS: Int = 4
](
    ctx: DeviceContext,
    name: String,
    fwd_abs_tol: Float64 = 5e-3,
    fwd_rel_tol: Float64 = 1e-2,
    bwd_abs_tol: Float64 = 5e-3,
    bwd_rel_tol: Float64 = 1e-2,
) raises -> Int:
    comptime assert OFF.IN_DIM == ON.IN_DIM
    comptime assert OFF.OUT_DIM == ON.OUT_DIM
    comptime assert OFF.PARAM_SIZE == ON.PARAM_SIZE
    comptime assert OFF.CACHE_SIZE == ON.CACHE_SIZE

    comptime IN = OFF.IN_DIM
    comptime OUT = OFF.OUT_DIM
    comptime PS = OFF.PARAM_SIZE
    comptime CS = OFF.CACHE_SIZE
    comptime WS_OFF = OFF.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_ON = ON.WORKSPACE_SIZE_PER_SAMPLE

    print("Parity:", name, "(IN=", IN, "OUT=", OUT, "BS=", BS, "PS=", PS, ")")

    var params_host = ctx.enqueue_create_host_buffer[dtype](PS)
    _init_params(params_host.unsafe_ptr(), PS)
    var input_host = ctx.enqueue_create_host_buffer[dtype](BS * IN)
    _init_input(input_host.unsafe_ptr(), BS * IN)
    var grad_out_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
    _init_grad_out(grad_out_host.unsafe_ptr(), BS * OUT)

    var params_off = ctx.enqueue_create_buffer[dtype](PS)
    var params_on = ctx.enqueue_create_buffer[dtype](PS)
    ctx.enqueue_copy(params_off, params_host)
    ctx.enqueue_copy(params_on, params_host)
    var input_off = ctx.enqueue_create_buffer[dtype](BS * IN)
    var input_on = ctx.enqueue_create_buffer[dtype](BS * IN)
    ctx.enqueue_copy(input_off, input_host)
    ctx.enqueue_copy(input_on, input_host)
    var output_off = ctx.enqueue_create_buffer[dtype](BS * OUT)
    var output_on = ctx.enqueue_create_buffer[dtype](BS * OUT)
    var cache_off = ctx.enqueue_create_buffer[dtype](BS * CS if CS > 0 else 1)
    var cache_on = ctx.enqueue_create_buffer[dtype](BS * CS if CS > 0 else 1)
    var ws_off = ctx.enqueue_create_buffer[dtype](
        BS * WS_OFF if WS_OFF > 0 else 1
    )
    var ws_on = ctx.enqueue_create_buffer[dtype](
        BS * WS_ON if WS_ON > 0 else 1
    )
    var state_buf = ctx.enqueue_create_buffer[dtype](1)

    var inp_off_t = LayoutTensor[
        dtype, Layout.row_major(BS, IN), MutAnyOrigin
    ](input_off.unsafe_ptr())
    var out_off_t = LayoutTensor[
        dtype, Layout.row_major(BS, OUT), MutAnyOrigin
    ](output_off.unsafe_ptr())
    var p_off_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](
        params_off.unsafe_ptr()
    )
    var s_off_t = LayoutTensor[
        dtype, Layout.row_major(OFF.STATE_SIZE), MutAnyOrigin
    ](state_buf.unsafe_ptr())
    var c_off_t = LayoutTensor[
        dtype, Layout.row_major(BS, CS), MutAnyOrigin
    ](cache_off.unsafe_ptr())
    OFF.forward_gpu[BS](
        ctx, out_off_t, inp_off_t, p_off_t, s_off_t, c_off_t, ws_off
    )

    var inp_on_t = LayoutTensor[
        dtype, Layout.row_major(BS, ON.IN_DIM), MutAnyOrigin
    ](input_on.unsafe_ptr())
    var out_on_t = LayoutTensor[
        dtype, Layout.row_major(BS, ON.OUT_DIM), MutAnyOrigin
    ](output_on.unsafe_ptr())
    var p_on_t = LayoutTensor[
        dtype, Layout.row_major(ON.PARAM_SIZE), MutAnyOrigin
    ](params_on.unsafe_ptr())
    var s_on_t = LayoutTensor[
        dtype, Layout.row_major(ON.STATE_SIZE), MutAnyOrigin
    ](state_buf.unsafe_ptr())
    var c_on_t = LayoutTensor[
        dtype, Layout.row_major(BS, ON.CACHE_SIZE), MutAnyOrigin
    ](cache_on.unsafe_ptr())
    ON.forward_gpu[BS](
        ctx, out_on_t, inp_on_t, p_on_t, s_on_t, c_on_t, ws_on
    )

    var out_off_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
    var out_on_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
    ctx.enqueue_copy(out_off_host, output_off)
    ctx.enqueue_copy(out_on_host, output_on)
    ctx.synchronize()

    var fails = _compare(
        "forward",
        out_off_host.unsafe_ptr(),
        out_on_host.unsafe_ptr(),
        BS * OUT,
        fwd_abs_tol,
        fwd_rel_tol,
    )

    var grad_out_off_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    var grad_out_on_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    ctx.enqueue_copy(grad_out_off_buf, grad_out_host)
    ctx.enqueue_copy(grad_out_on_buf, grad_out_host)
    var grad_in_off_buf = ctx.enqueue_create_buffer[dtype](BS * IN)
    var grad_in_on_buf = ctx.enqueue_create_buffer[dtype](BS * IN)
    ctx.enqueue_memset(grad_in_off_buf, 0)
    ctx.enqueue_memset(grad_in_on_buf, 0)
    var grads_off = ctx.enqueue_create_buffer[dtype](PS)
    var grads_on = ctx.enqueue_create_buffer[dtype](PS)
    ctx.enqueue_memset(grads_off, 0)
    ctx.enqueue_memset(grads_on, 0)

    var go_off_t = LayoutTensor[
        dtype, Layout.row_major(BS, OUT), MutAnyOrigin
    ](grad_out_off_buf.unsafe_ptr())
    var gi_off_t = LayoutTensor[
        dtype, Layout.row_major(BS, IN), MutAnyOrigin
    ](grad_in_off_buf.unsafe_ptr())
    var gp_off_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](
        grads_off.unsafe_ptr()
    )
    OFF.backward_gpu[BS](
        ctx, gi_off_t, go_off_t, p_off_t, s_off_t, c_off_t, gp_off_t, ws_off
    )

    var go_on_t = LayoutTensor[
        dtype, Layout.row_major(BS, ON.OUT_DIM), MutAnyOrigin
    ](grad_out_on_buf.unsafe_ptr())
    var gi_on_t = LayoutTensor[
        dtype, Layout.row_major(BS, ON.IN_DIM), MutAnyOrigin
    ](grad_in_on_buf.unsafe_ptr())
    var gp_on_t = LayoutTensor[
        dtype, Layout.row_major(ON.PARAM_SIZE), MutAnyOrigin
    ](grads_on.unsafe_ptr())
    ON.backward_gpu[BS](
        ctx, gi_on_t, go_on_t, p_on_t, s_on_t, c_on_t, gp_on_t, ws_on
    )

    var grad_in_off_host = ctx.enqueue_create_host_buffer[dtype](BS * IN)
    var grad_in_on_host = ctx.enqueue_create_host_buffer[dtype](BS * IN)
    var grads_off_host = ctx.enqueue_create_host_buffer[dtype](PS)
    var grads_on_host = ctx.enqueue_create_host_buffer[dtype](PS)
    ctx.enqueue_copy(grad_in_off_host, grad_in_off_buf)
    ctx.enqueue_copy(grad_in_on_host, grad_in_on_buf)
    ctx.enqueue_copy(grads_off_host, grads_off)
    ctx.enqueue_copy(grads_on_host, grads_on)
    ctx.synchronize()

    fails += _compare(
        "grad_input",
        grad_in_off_host.unsafe_ptr(),
        grad_in_on_host.unsafe_ptr(),
        BS * IN,
        bwd_abs_tol,
        bwd_rel_tol,
    )
    fails += _compare(
        "grad_params",
        grads_off_host.unsafe_ptr(),
        grads_on_host.unsafe_ptr(),
        PS,
        bwd_abs_tol,
        bwd_rel_tol,
    )
    return fails


def main() raises:
    var ctx = DeviceContext()

    comptime if not has_nvidia_gpu_accelerator():
        print(
            "WARNING: not running on NVIDIA — both flag values route to the"
            " same kernel. Compile-check only."
        )

    print("=== USE_MAX_KERNELS parity (NormedLinear) ===")
    var total_fails = 0

    print("--- NormedLinear (Linear + LN + Mish) ---")
    total_fails += parity_check[
        NormedLinear[8, 16, USE_MAX_KERNELS=False],
        NormedLinear[8, 16, USE_MAX_KERNELS=True],
        BS=8,
    ](ctx, "NormedLinear[8,16] BS=8")

    total_fails += parity_check[
        NormedLinear[16, 16, USE_MAX_KERNELS=False],
        NormedLinear[16, 16, USE_MAX_KERNELS=True],
        BS=16,
    ](ctx, "NormedLinear[16,16] BS=16")

    total_fails += parity_check[
        NormedLinear[32, 32, USE_MAX_KERNELS=False],
        NormedLinear[32, 32, USE_MAX_KERNELS=True],
        BS=16,
    ](ctx, "NormedLinear[32,32] BS=16")

    total_fails += parity_check[
        NormedLinear[64, 64, USE_MAX_KERNELS=False],
        NormedLinear[64, 64, USE_MAX_KERNELS=True],
        BS=32,
    ](ctx, "NormedLinear[64,64] BS=32 (TDMPC2 typical)")

    total_fails += parity_check[
        NormedLinear[256, 256, USE_MAX_KERNELS=False],
        NormedLinear[256, 256, USE_MAX_KERNELS=True],
        BS=64,
    ](ctx, "NormedLinear[256,256] BS=64 (RL agent hidden)")

    print("======================================================")
    if total_fails == 0:
        print("=== USE_MAX_KERNELS parity (NormedLinear): ALL PASS ===")
    else:
        print(
            "=== USE_MAX_KERNELS parity (NormedLinear): FAILURES =",
            total_fails,
            "===",
        )

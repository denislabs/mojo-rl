"""GPU parity test: Linear / MatMul with USE_MAX_KERNELS=False vs =True.

Builds the same model twice — once with the custom MMA kernel, once with
linalg.matmul (max_matmul) — initializes both with identical parameters,
runs the same input through both, and compares forward outputs and backward
gradients (grad_input + grad_params).

If the two paths produce numerically equivalent results within float32 GEMM
tolerance, max_matmul is correct on the tested shapes. If they diverge, the
report identifies which path (forward / dx / dW) is responsible.

On Apple, both paths route to the same 2x2 tiled kernel (the comptime if
that selects max_matmul requires has_nvidia_gpu_accelerator()), so the test
is a no-op there — useful only as a compile-check. Run on NVIDIA for a real
A/B.

Usage:
    pixi run -e nvidia mojo run -I . tests/nn/test_use_max_kernels_parity.mojo
    pixi run -e apple  mojo run -I . tests/nn/test_use_max_kernels_parity.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import abs
from std.sys import has_nvidia_gpu_accelerator
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import (
    Model,
    Linear,
    LinearReLU,
    LinearTanh,
    LinearSigmoid,
    LinearMish,
    LinearSwish,
    Conv2DReLU,
    Conv2DTanh,
    Conv2DSigmoid,
    Conv2DMish,
)
from mojo_rl.nn.autodiff import MatMul, AutoFused, BiasAdd, Conv2D


def _init_params(ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int):
    """Deterministic dense init in [-0.5, 0.5] — exercises non-zero rows
    everywhere so accumulation differences would surface.
    """
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
    """Element-wise compare with abs-or-rel tolerance.

    An element passes if either the absolute error is below `abs_tol` (covers
    the float32 GEMM noise floor — different reduction orders between two
    correct implementations differ by ~1e-3 abs on tensor-core fp32) or the
    relative error is below `rel_tol` (covers larger-magnitude values where
    noise scales with magnitude). Reports max_abs and max_rel on every run
    so the noise floor stays visible even when all elements pass.
    """
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
        # Pass = within abs OR within rel. Fail only if both exceed.
        if err > abs_tol and rel > rel_tol:
            fails += 1
            if fails <= 3:
                print(
                    "    ",
                    name,
                    "[",
                    i,
                    "]: off=",
                    x,
                    "on=",
                    y,
                    "abs=",
                    err,
                    "rel=",
                    rel,
                )
    if fails == 0:
        print("  [PASS]", name, ": max_abs=", max_abs, "max_rel=", max_rel)
    else:
        print(
            "  [FAIL]",
            name,
            ":",
            fails,
            "/",
            n,
            "max_abs=",
            max_abs,
            "max_rel=",
            max_rel,
        )
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
    """OFF and ON must describe the same shape but with different
    USE_MAX_KERNELS values. Returns total fail count.

    Tolerances default to float32 tensor-core GEMM noise levels: an element
    passes if either abs_err < abs_tol OR rel_err < rel_tol. The defaults
    (5e-3 abs, 1e-2 rel) accommodate ~1e-3 abs noise observed between
    custom-MMA and linalg.matmul on small/medium shapes.
    """
    # Flag must not change layout — OFF and ON must agree on dims.
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

    print(
        "Parity:",
        name,
        "(IN=",
        IN,
        "OUT=",
        OUT,
        "BS=",
        BS,
        "PS=",
        PS,
        ")",
    )

    # ── Shared input + params (host) ─────────────────────────
    var params_host = ctx.enqueue_create_host_buffer[dtype](PS)
    _init_params(params_host.unsafe_ptr(), PS)

    var input_host = ctx.enqueue_create_host_buffer[dtype](BS * IN)
    _init_input(input_host.unsafe_ptr(), BS * IN)

    var grad_out_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
    _init_grad_out(grad_out_host.unsafe_ptr(), BS * OUT)

    # ── GPU buffers — duplicated per flag ────────────────────
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
    var cache_off = ctx.enqueue_create_buffer[dtype](
        BS * CS if CS > 0 else 1
    )
    var cache_on = ctx.enqueue_create_buffer[dtype](
        BS * CS if CS > 0 else 1
    )

    var ws_off = ctx.enqueue_create_buffer[dtype](
        BS * WS_OFF if WS_OFF > 0 else 1
    )
    var ws_on = ctx.enqueue_create_buffer[dtype](
        BS * WS_ON if WS_ON > 0 else 1
    )

    # State buffer (dummy — DiffOps are stateless, STATE_SIZE = 0)
    var state_buf = ctx.enqueue_create_buffer[dtype](1)

    # ── Forward (OFF) ────────────────────────────────────────
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

    # ── Forward (ON) ─────────────────────────────────────────
    # Note: tensor layouts must reference ON's own comptime dims (not OFF's),
    # even though they evaluate equal — the compiler treats `OFF.IN_DIM` and
    # `ON.IN_DIM` as distinct expressions when matching layout parameters.
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

    # Download outputs
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

    # ── Backward — uploaded grad_output ─────────────────────
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
        ctx,
        gi_off_t,
        go_off_t,
        p_off_t,
        s_off_t,
        c_off_t,
        gp_off_t,
        ws_off,
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
        ctx,
        gi_on_t,
        go_on_t,
        p_on_t,
        s_on_t,
        c_on_t,
        gp_on_t,
        ws_on,
    )

    # Download gradients
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
            " same kernel. This run only verifies compilation; for a real A/B"
            " run with `pixi run -e nvidia ...`."
        )

    print("=== USE_MAX_KERNELS parity (forward + backward) ===")

    var total_fails = 0

    # ── Plain MatMul standalone ─────────────────────────────
    # Bare MatMul (no AutoFused). Tests the primitive's GPU path directly.
    print("--- MatMul (standalone) ---")
    total_fails += parity_check[
        AutoFused[MatMul[8, 4, USE_MAX_KERNELS=False]],
        AutoFused[MatMul[8, 4, USE_MAX_KERNELS=True]],
    ](ctx, "MatMul[8,4]")

    total_fails += parity_check[
        AutoFused[MatMul[32, 32, USE_MAX_KERNELS=False]],
        AutoFused[MatMul[32, 32, USE_MAX_KERNELS=True]],
    ](ctx, "MatMul[32,32]")

    total_fails += parity_check[
        AutoFused[MatMul[128, 1, USE_MAX_KERNELS=False]],
        AutoFused[MatMul[128, 1, USE_MAX_KERNELS=True]],
    ](ctx, "MatMul[128,1] (AlphaZero value head shape)")

    # ── Linear (MatMul + BiasAdd, fused as FusedMatMulBias) ──
    print("--- Linear (Fused M+B) ---")
    total_fails += parity_check[
        Linear[8, 4, USE_MAX_KERNELS=False],
        Linear[8, 4, USE_MAX_KERNELS=True],
    ](ctx, "Linear[8,4]")

    total_fails += parity_check[
        Linear[16, 16, USE_MAX_KERNELS=False],
        Linear[16, 16, USE_MAX_KERNELS=True],
    ](ctx, "Linear[16,16]")

    total_fails += parity_check[
        Linear[32, 32, USE_MAX_KERNELS=False],
        Linear[32, 32, USE_MAX_KERNELS=True],
    ](ctx, "Linear[32,32]")

    # AlphaZero TTT-like shapes — small dims, small batch (the suspected
    # regression case in project_alphazero_ttt_nvidia_regression).
    print("--- Linear (AlphaZero TTT-like small shapes) ---")
    total_fails += parity_check[
        Linear[27, 64, USE_MAX_KERNELS=False],
        Linear[27, 64, USE_MAX_KERNELS=True],
        BS=16,
    ](ctx, "Linear[27,64] BS=16 (TTT trunk)")

    total_fails += parity_check[
        Linear[64, 9, USE_MAX_KERNELS=False],
        Linear[64, 9, USE_MAX_KERNELS=True],
        BS=16,
    ](ctx, "Linear[64,9] BS=16 (TTT policy head)")

    total_fails += parity_check[
        Linear[128, 1, USE_MAX_KERNELS=False],
        Linear[128, 1, USE_MAX_KERNELS=True],
        BS=16,
    ](ctx, "Linear[128,1] BS=16 (value head, the suspect)")

    # ── Larger shapes — typical RL agent dims ────────────────
    print("--- Linear (larger RL-agent shapes) ---")
    total_fails += parity_check[
        Linear[256, 256, USE_MAX_KERNELS=False],
        Linear[256, 256, USE_MAX_KERNELS=True],
        BS=128,
    ](ctx, "Linear[256,256] BS=128")

    total_fails += parity_check[
        Linear[256, 17, USE_MAX_KERNELS=False],
        Linear[256, 17, USE_MAX_KERNELS=True],
        BS=128,
    ](ctx, "Linear[256,17] BS=128 (HalfCheetah action head)")

    # ── Phase 2: Linear + activation (FusedMatMulBiasActivation) ─
    print("--- LinearReLU (Fused M+B+ReLU) ---")
    total_fails += parity_check[
        LinearReLU[27, 64, USE_MAX_KERNELS=False],
        LinearReLU[27, 64, USE_MAX_KERNELS=True],
        BS=16,
    ](ctx, "LinearReLU[27,64] BS=16 (TTT trunk layer 1)")

    total_fails += parity_check[
        LinearReLU[64, 64, USE_MAX_KERNELS=False],
        LinearReLU[64, 64, USE_MAX_KERNELS=True],
        BS=16,
    ](ctx, "LinearReLU[64,64] BS=16 (TTT trunk layer 2 close approx)")

    total_fails += parity_check[
        LinearReLU[256, 256, USE_MAX_KERNELS=False],
        LinearReLU[256, 256, USE_MAX_KERNELS=True],
        BS=128,
    ](ctx, "LinearReLU[256,256] BS=128 (RL agent hidden)")

    print("--- LinearTanh (Fused M+B+Tanh) ---")
    total_fails += parity_check[
        LinearTanh[64, 64, USE_MAX_KERNELS=False],
        LinearTanh[64, 64, USE_MAX_KERNELS=True],
        BS=16,
    ](ctx, "LinearTanh[64,64] BS=16")

    total_fails += parity_check[
        LinearTanh[256, 17, USE_MAX_KERNELS=False],
        LinearTanh[256, 17, USE_MAX_KERNELS=True],
        BS=128,
    ](ctx, "LinearTanh[256,17] BS=128 (HalfCheetah continuous actor)")

    print("--- LinearSigmoid / LinearMish / LinearSwish ---")
    total_fails += parity_check[
        LinearSigmoid[64, 32, USE_MAX_KERNELS=False],
        LinearSigmoid[64, 32, USE_MAX_KERNELS=True],
        BS=16,
    ](ctx, "LinearSigmoid[64,32] BS=16")

    total_fails += parity_check[
        LinearMish[128, 128, USE_MAX_KERNELS=False],
        LinearMish[128, 128, USE_MAX_KERNELS=True],
        BS=32,
    ](ctx, "LinearMish[128,128] BS=32 (TDMPC2-ish)")

    total_fails += parity_check[
        LinearSwish[128, 128, USE_MAX_KERNELS=False],
        LinearSwish[128, 128, USE_MAX_KERNELS=True],
        BS=32,
    ](ctx, "LinearSwish[128,128] BS=32 (MBPO actor)")

    # ── Phase 3a: Conv2D primitive ─────────────────────────
    print("--- Conv2D (im2col + matmul) ---")
    # Tiny shape — AlphaZero-like board CNN
    total_fails += parity_check[
        AutoFused[
            Conv2D[2, 4, 3, 1, 1, 5, 5, USE_MAX_KERNELS=False]
        ],
        AutoFused[
            Conv2D[2, 4, 3, 1, 1, 5, 5, USE_MAX_KERNELS=True]
        ],
        BS=8,
    ](ctx, "Conv2D[2->4,3x3,5x5] BS=8 (AlphaZero board)")

    # MNIST-like first layer
    total_fails += parity_check[
        AutoFused[
            Conv2D[1, 32, 3, 1, 1, 28, 28, USE_MAX_KERNELS=False]
        ],
        AutoFused[
            Conv2D[1, 32, 3, 1, 1, 28, 28, USE_MAX_KERNELS=True]
        ],
        BS=16,
    ](ctx, "Conv2D[1->32,3x3,28x28] BS=16 (MNIST first layer)")

    # CIFAR-like first layer
    total_fails += parity_check[
        AutoFused[
            Conv2D[3, 32, 3, 1, 1, 32, 32, USE_MAX_KERNELS=False]
        ],
        AutoFused[
            Conv2D[3, 32, 3, 1, 1, 32, 32, USE_MAX_KERNELS=True]
        ],
        BS=16,
    ](ctx, "Conv2D[3->32,3x3,32x32] BS=16 (CIFAR first layer)")

    # Deeper conv with stride=1, 64→64 channels
    total_fails += parity_check[
        AutoFused[
            Conv2D[64, 64, 3, 1, 1, 16, 16, USE_MAX_KERNELS=False]
        ],
        AutoFused[
            Conv2D[64, 64, 3, 1, 1, 16, 16, USE_MAX_KERNELS=True]
        ],
        BS=16,
    ](ctx, "Conv2D[64->64,3x3,16x16] BS=16 (typical hidden conv)")

    # Stride-2 downsample (Atari-style first layer)
    total_fails += parity_check[
        AutoFused[
            Conv2D[4, 32, 8, 4, 0, 84, 84, USE_MAX_KERNELS=False]
        ],
        AutoFused[
            Conv2D[4, 32, 8, 4, 0, 84, 84, USE_MAX_KERNELS=True]
        ],
        BS=8,
    ](ctx, "Conv2D[4->32,8x8 s4,84x84] BS=8 (Atari NatureCNN)")

    # ── Phase 3b: Conv2D + activation fusion ────────────────
    print("--- Conv2DReLU (Fused C+B+ReLU) ---")
    total_fails += parity_check[
        Conv2DReLU[2, 4, 3, 1, 1, 5, 5, USE_MAX_KERNELS=False],
        Conv2DReLU[2, 4, 3, 1, 1, 5, 5, USE_MAX_KERNELS=True],
        BS=8,
    ](ctx, "Conv2DReLU[2->4,3x3,5x5] BS=8 (AlphaZero board)")

    total_fails += parity_check[
        Conv2DReLU[1, 32, 3, 1, 1, 28, 28, USE_MAX_KERNELS=False],
        Conv2DReLU[1, 32, 3, 1, 1, 28, 28, USE_MAX_KERNELS=True],
        BS=16,
    ](ctx, "Conv2DReLU[1->32,3x3,28x28] BS=16 (MNIST first layer)")

    total_fails += parity_check[
        Conv2DReLU[3, 32, 3, 1, 1, 32, 32, USE_MAX_KERNELS=False],
        Conv2DReLU[3, 32, 3, 1, 1, 32, 32, USE_MAX_KERNELS=True],
        BS=16,
    ](ctx, "Conv2DReLU[3->32,3x3,32x32] BS=16 (CIFAR first layer)")

    total_fails += parity_check[
        Conv2DReLU[64, 64, 3, 1, 1, 16, 16, USE_MAX_KERNELS=False],
        Conv2DReLU[64, 64, 3, 1, 1, 16, 16, USE_MAX_KERNELS=True],
        BS=16,
    ](ctx, "Conv2DReLU[64->64,3x3,16x16] BS=16 (typical hidden conv)")

    print("--- Conv2DTanh / Conv2DSigmoid / Conv2DMish ---")
    total_fails += parity_check[
        Conv2DTanh[3, 16, 3, 1, 1, 32, 32, USE_MAX_KERNELS=False],
        Conv2DTanh[3, 16, 3, 1, 1, 32, 32, USE_MAX_KERNELS=True],
        BS=16,
    ](ctx, "Conv2DTanh[3->16,3x3,32x32] BS=16")

    total_fails += parity_check[
        Conv2DSigmoid[8, 16, 3, 1, 1, 8, 8, USE_MAX_KERNELS=False],
        Conv2DSigmoid[8, 16, 3, 1, 1, 8, 8, USE_MAX_KERNELS=True],
        BS=16,
    ](ctx, "Conv2DSigmoid[8->16,3x3,8x8] BS=16")

    total_fails += parity_check[
        Conv2DMish[16, 32, 3, 1, 1, 16, 16, USE_MAX_KERNELS=False],
        Conv2DMish[16, 32, 3, 1, 1, 16, 16, USE_MAX_KERNELS=True],
        BS=16,
    ](ctx, "Conv2DMish[16->32,3x3,16x16] BS=16")

    print("======================================================")
    if total_fails == 0:
        print("=== USE_MAX_KERNELS parity: ALL PASS ===")
    else:
        print(
            "=== USE_MAX_KERNELS parity: FAILURES =",
            total_fails,
            "===",
        )

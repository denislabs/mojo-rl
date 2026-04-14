"""Integer correctness test for custom MMA kernels.

Small integer inputs (1-7) are exactly representable in TF32/FP32.
Products and sums stay within exact FP32 range (< 2^24).
Therefore: exact match expected. Any mismatch = kernel bug, not precision.

Tests:
  1. Full Conv2D pipeline: CPU eval vs GPU eval_gpu with integers
  2. Full Conv2D backward: CPU vjp vs GPU vjp_gpu with integers
  3. Isolated MMA kernels vs CPU reference (NVIDIA only):
     - conv_matmul_fwd_mma: C = A @ B.T (transpose_b)
     - conv_matmul_dW_mma:  C = A @ B   (standard)
     - conv_matmul_dx_mma:  C = A.T @ B  (transpose_a)
  4. Full Linear pipeline: FusedMatMulBiasReLU CPU vs GPU with integers

Usage:
    pixi run -e nvidia mojo run -I . tests/nn/test_mma_integer.mojo
    pixi run -e apple mojo run -I . tests/nn/test_mma_integer.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import abs
from std.memory import alloc
from layout import Layout, LayoutTensor
from std.sys import has_nvidia_gpu_accelerator

from mojo_rl.nn.constants import (
    dtype,
    MMA_BLOCK_M,
    MMA_BLOCK_N,
    MMA_BLOCK_THREADS,
)
from mojo_rl.nn.autodiff.primitives.conv2d import Conv2D
from mojo_rl.nn.training import NetworkState, GPUNetworkState
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.model import (
    Model,
    Sequential,
    LinearReLU,
    Linear,
    Conv2DReLU,
    FlattenLayer,
)


# =========================================================================
# Helpers
# =========================================================================


def compare(
    name: String,
    cpu_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    gpu_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    count: Int,
) -> Bool:
    """Compare two buffers, report mismatches. Returns True if exact match."""
    var errors = 0
    var max_abs: Float64 = 0.0
    for i in range(count):
        var c = Float64(cpu_ptr[i])
        var g = Float64(gpu_ptr[i])
        var diff = abs(c - g)
        if diff > max_abs:
            max_abs = diff
        if diff > 0:
            errors += 1
            if errors <= 5:
                print("  [", i, "]: cpu=", c, "gpu=", g, "diff=", diff)
    if errors == 0:
        print("[PASS]", name, ": exact match (", count, "elements)")
    else:
        print("[FAIL]", name, ":", errors, "/", count, "max_abs=", max_abs)
    return errors == 0


# =========================================================================
# Test 1: Full Conv2D pipeline (CPU vs GPU) with integer weights/inputs
# =========================================================================


def test_conv2d_pipeline[
    IC: Int, OC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, BATCH: Int
](ctx: DeviceContext) raises:
    comptime C = Conv2D[IC, OC, K, S, P, H, W]
    print(
        "Conv2D[IC=",
        IC,
        ",OC=",
        OC,
        ",K=",
        K,
        ",H=",
        H,
        ",W=",
        W,
        "] BATCH=",
        BATCH,
    )
    print(
        "  col_size=",
        C.col_size,
        " spatial_out=",
        C.spatial_out,
        " IN_DIM=",
        C.IN_DIM,
        " OUT_DIM=",
        C.OUT_DIM,
    )

    # ── CPU buffers ──
    var input_ptr = alloc[Scalar[dtype]](BATCH * C.IN_DIM)
    var params_ptr = alloc[Scalar[dtype]](C.PARAM_SIZE)
    var output_cpu = alloc[Scalar[dtype]](BATCH * C.OUT_DIM)
    var cache_cpu = alloc[Scalar[dtype]](BATCH * C.CACHE_SIZE)
    var output_gpu = alloc[Scalar[dtype]](BATCH * C.OUT_DIM)

    # Fill with small integers (1-7)
    for i in range(BATCH * C.IN_DIM):
        input_ptr[i] = Scalar[dtype]((i % 5) + 1)
    for i in range(C.PARAM_SIZE):
        params_ptr[i] = Scalar[dtype]((i % 7) + 1)
    for i in range(BATCH * C.OUT_DIM):
        output_cpu[i] = 0
    for i in range(BATCH * C.CACHE_SIZE):
        cache_cpu[i] = 0

    # ── CPU forward ──
    var in_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.IN_DIM), MutAnyOrigin
    ](input_ptr)
    var out_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.OUT_DIM), MutAnyOrigin
    ](output_cpu)
    var p_lt = LayoutTensor[
        dtype, Layout.row_major(C.PARAM_SIZE), MutAnyOrigin
    ](params_ptr)
    var ca_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.CACHE_SIZE), MutAnyOrigin
    ](cache_cpu)

    C.eval[BATCH](in_lt, out_lt, p_lt, ca_lt)

    # ── GPU forward ──
    var in_buf = ctx.enqueue_create_buffer[dtype](BATCH * C.IN_DIM)
    var p_buf = ctx.enqueue_create_buffer[dtype](C.PARAM_SIZE)
    var out_buf = ctx.enqueue_create_buffer[dtype](BATCH * C.OUT_DIM)
    var ca_buf = ctx.enqueue_create_buffer[dtype](BATCH * C.CACHE_SIZE)
    comptime WS_SIZE = BATCH * C.OP_WORKSPACE_PER_SAMPLE
    var ws_buf = ctx.enqueue_create_buffer[dtype](WS_SIZE if WS_SIZE > 0 else 1)

    # Upload input + params
    var in_host = ctx.enqueue_create_host_buffer[dtype](BATCH * C.IN_DIM)
    for i in range(BATCH * C.IN_DIM):
        in_host[i] = input_ptr[i]
    ctx.enqueue_copy(in_buf, in_host)

    var p_host = ctx.enqueue_create_host_buffer[dtype](C.PARAM_SIZE)
    for i in range(C.PARAM_SIZE):
        p_host[i] = params_ptr[i]
    ctx.enqueue_copy(p_buf, p_host)

    var in_g = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.IN_DIM), MutAnyOrigin
    ](in_buf.unsafe_ptr())
    var out_g = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.OUT_DIM), MutAnyOrigin
    ](out_buf.unsafe_ptr())
    var p_g = LayoutTensor[dtype, Layout.row_major(C.PARAM_SIZE), MutAnyOrigin](
        p_buf.unsafe_ptr()
    )
    var ca_g = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.CACHE_SIZE), MutAnyOrigin
    ](ca_buf.unsafe_ptr())

    C.eval_gpu[BATCH](ctx, out_g, in_g, p_g, ca_g, ws_buf.unsafe_ptr())

    # Download + compare forward
    var out_host = ctx.enqueue_create_host_buffer[dtype](BATCH * C.OUT_DIM)
    ctx.enqueue_copy(out_host, out_buf)
    ctx.synchronize()

    for i in range(BATCH * C.OUT_DIM):
        output_gpu[i] = out_host[i]

    var fwd_ok = compare("forward", output_cpu, output_gpu, BATCH * C.OUT_DIM)

    # ── Backward ──
    var grad_out_ptr = alloc[Scalar[dtype]](BATCH * C.OUT_DIM)
    for i in range(BATCH * C.OUT_DIM):
        grad_out_ptr[i] = Scalar[dtype]((i % 3) + 1)

    # CPU backward
    var grad_in_cpu = alloc[Scalar[dtype]](BATCH * C.IN_DIM)
    var grad_p_cpu = alloc[Scalar[dtype]](C.PARAM_SIZE)
    for i in range(BATCH * C.IN_DIM):
        grad_in_cpu[i] = 0
    for i in range(C.PARAM_SIZE):
        grad_p_cpu[i] = 0

    var go_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.OUT_DIM), MutAnyOrigin
    ](grad_out_ptr)
    var gi_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.IN_DIM), MutAnyOrigin
    ](grad_in_cpu)
    var gp_lt = LayoutTensor[
        dtype, Layout.row_major(C.PARAM_SIZE), MutAnyOrigin
    ](grad_p_cpu)

    C.vjp[BATCH](go_lt, gi_lt, p_lt, ca_lt, gp_lt)

    # GPU backward
    var go_buf = ctx.enqueue_create_buffer[dtype](BATCH * C.OUT_DIM)
    var gi_buf = ctx.enqueue_create_buffer[dtype](BATCH * C.IN_DIM)
    var gp_buf = ctx.enqueue_create_buffer[dtype](C.PARAM_SIZE)

    var go_host = ctx.enqueue_create_host_buffer[dtype](BATCH * C.OUT_DIM)
    for i in range(BATCH * C.OUT_DIM):
        go_host[i] = grad_out_ptr[i]
    ctx.enqueue_copy(go_buf, go_host)

    # Zero grad_input and grad_params on GPU
    var zi_host = ctx.enqueue_create_host_buffer[dtype](BATCH * C.IN_DIM)
    for i in range(BATCH * C.IN_DIM):
        zi_host[i] = 0
    ctx.enqueue_copy(gi_buf, zi_host)

    var zp_host = ctx.enqueue_create_host_buffer[dtype](C.PARAM_SIZE)
    for i in range(C.PARAM_SIZE):
        zp_host[i] = 0
    ctx.enqueue_copy(gp_buf, zp_host)

    var go_g = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.OUT_DIM), MutAnyOrigin
    ](go_buf.unsafe_ptr())
    var gi_g = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.IN_DIM), MutAnyOrigin
    ](gi_buf.unsafe_ptr())
    var gp_g = LayoutTensor[
        dtype, Layout.row_major(C.PARAM_SIZE), MutAnyOrigin
    ](gp_buf.unsafe_ptr())

    C.vjp_gpu[BATCH](ctx, go_g, gi_g, p_g, ca_g, gp_g, ws_buf.unsafe_ptr())

    # Download + compare backward
    var gi_host_out = ctx.enqueue_create_host_buffer[dtype](BATCH * C.IN_DIM)
    var gp_host_out = ctx.enqueue_create_host_buffer[dtype](C.PARAM_SIZE)
    ctx.enqueue_copy(gi_host_out, gi_buf)
    ctx.enqueue_copy(gp_host_out, gp_buf)
    ctx.synchronize()

    var grad_in_gpu = alloc[Scalar[dtype]](BATCH * C.IN_DIM)
    var grad_p_gpu = alloc[Scalar[dtype]](C.PARAM_SIZE)
    for i in range(BATCH * C.IN_DIM):
        grad_in_gpu[i] = gi_host_out[i]
    for i in range(C.PARAM_SIZE):
        grad_p_gpu[i] = gp_host_out[i]

    var gp_ok = compare("grad_params", grad_p_cpu, grad_p_gpu, C.PARAM_SIZE)
    var gi_ok = compare(
        "grad_input", grad_in_cpu, grad_in_gpu, BATCH * C.IN_DIM
    )

    # Cleanup
    input_ptr.free()
    params_ptr.free()
    output_cpu.free()
    output_gpu.free()
    cache_cpu.free()
    grad_out_ptr.free()
    grad_in_cpu.free()
    grad_p_cpu.free()
    grad_in_gpu.free()
    grad_p_gpu.free()


# =========================================================================
# Test 2: Isolated MMA kernel vs CPU reference (NVIDIA only)
# =========================================================================


def test_isolated_mma_fwd[
    IC: Int, OC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, BATCH: Int
](ctx: DeviceContext) raises:
    """Test conv_matmul_fwd_mma: C(K_TOTAL, OC) = A(K_TOTAL, col_size) @ B.T(OC, col_size).
    """
    comptime C = Conv2D[IC, OC, K, S, P, H, W]
    comptime K_TOTAL = BATCH * C.spatial_out
    comptime M = K_TOTAL
    comptime N = OC
    comptime KK = C.col_size

    # CPU reference: out[i,j] = sum_k( A[i,k] * B[j,k] )
    var a_ptr = alloc[Scalar[dtype]](M * KK)
    var b_ptr = alloc[Scalar[dtype]](N * KK)
    var c_ref = alloc[Scalar[dtype]](M * N)
    var c_gpu = alloc[Scalar[dtype]](M * N)

    for i in range(M * KK):
        a_ptr[i] = Scalar[dtype]((i % 5) + 1)
    for i in range(N * KK):
        b_ptr[i] = Scalar[dtype]((i % 7) + 1)

    # CPU matmul: C = A @ B.T
    for i in range(M):
        for j in range(N):
            var acc: Float64 = 0
            for k in range(KK):
                acc += Float64(a_ptr[i * KK + k]) * Float64(b_ptr[j * KK + k])
            c_ref[i * N + j] = Scalar[dtype](acc)

    # GPU kernel
    var a_buf = ctx.enqueue_create_buffer[dtype](M * KK)
    var b_buf = ctx.enqueue_create_buffer[dtype](N * KK)
    var c_buf = ctx.enqueue_create_buffer[dtype](M * N)

    var ah = ctx.enqueue_create_host_buffer[dtype](M * KK)
    var bh = ctx.enqueue_create_host_buffer[dtype](N * KK)
    for i in range(M * KK):
        ah[i] = a_ptr[i]
    for i in range(N * KK):
        bh[i] = b_ptr[i]
    ctx.enqueue_copy(a_buf, ah)
    ctx.enqueue_copy(b_buf, bh)

    var a_lt = LayoutTensor[
        dtype, Layout.row_major(K_TOTAL, C.col_size), MutAnyOrigin
    ](a_buf.unsafe_ptr())
    var b_lt = LayoutTensor[
        dtype, Layout.row_major(C.out_channels, C.col_size), MutAnyOrigin
    ](b_buf.unsafe_ptr())
    var c_lt = LayoutTensor[
        dtype, Layout.row_major(K_TOTAL, C.out_channels), MutAnyOrigin
    ](c_buf.unsafe_ptr())

    comptime grid_x = (N + MMA_BLOCK_N - 1) // MMA_BLOCK_N
    comptime grid_y = (M + MMA_BLOCK_M - 1) // MMA_BLOCK_M

    @always_inline
    def fwd_wrapper(
        c: LayoutTensor[
            dtype, Layout.row_major(K_TOTAL, C.out_channels), MutAnyOrigin
        ],
        a: LayoutTensor[
            dtype, Layout.row_major(K_TOTAL, C.col_size), MutAnyOrigin
        ],
        b: LayoutTensor[
            dtype, Layout.row_major(C.out_channels, C.col_size), MutAnyOrigin
        ],
    ):
        C.conv_matmul_fwd_mma[K_TOTAL, dtype](c, a, b)

    ctx.enqueue_function[fwd_wrapper, fwd_wrapper](
        c_lt,
        a_lt,
        b_lt,
        grid_dim=(grid_x, grid_y),
        block_dim=(MMA_BLOCK_THREADS, 1),
    )

    var ch = ctx.enqueue_create_host_buffer[dtype](M * N)
    ctx.enqueue_copy(ch, c_buf)
    ctx.synchronize()

    for i in range(M * N):
        c_gpu[i] = ch[i]

    _ = compare(
        "isolated fwd_mma (C=A@B.T) ["
        + String(M)
        + "x"
        + String(KK)
        + "] @ ["
        + String(N)
        + "x"
        + String(KK)
        + "].T",
        c_ref,
        c_gpu,
        M * N,
    )

    a_ptr.free()
    b_ptr.free()
    c_ref.free()
    c_gpu.free()


def test_isolated_mma_dW[
    IC: Int, OC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, BATCH: Int
](ctx: DeviceContext) raises:
    """Test conv_matmul_dW_mma: dW(OC, col_size) = grad(OC, K_TOTAL) @ col(K_TOTAL, col_size).
    """
    comptime C = Conv2D[IC, OC, K, S, P, H, W]
    comptime K_TOTAL = BATCH * C.spatial_out
    comptime M = OC
    comptime N = C.col_size
    comptime KK = K_TOTAL

    var a_ptr = alloc[Scalar[dtype]](M * KK)
    var b_ptr = alloc[Scalar[dtype]](KK * N)
    var c_ref = alloc[Scalar[dtype]](M * N)
    var c_gpu = alloc[Scalar[dtype]](M * N)

    for i in range(M * KK):
        a_ptr[i] = Scalar[dtype]((i % 3) + 1)
    for i in range(KK * N):
        b_ptr[i] = Scalar[dtype]((i % 5) + 1)

    # CPU matmul: C = A @ B
    for i in range(M):
        for j in range(N):
            var acc: Float64 = 0
            for k in range(KK):
                acc += Float64(a_ptr[i * KK + k]) * Float64(b_ptr[k * N + j])
            c_ref[i * N + j] = Scalar[dtype](acc)

    var a_buf = ctx.enqueue_create_buffer[dtype](M * KK)
    var b_buf = ctx.enqueue_create_buffer[dtype](KK * N)
    var c_buf = ctx.enqueue_create_buffer[dtype](M * N)

    var ah = ctx.enqueue_create_host_buffer[dtype](M * KK)
    var bh = ctx.enqueue_create_host_buffer[dtype](KK * N)
    for i in range(M * KK):
        ah[i] = a_ptr[i]
    for i in range(KK * N):
        bh[i] = b_ptr[i]
    ctx.enqueue_copy(a_buf, ah)
    ctx.enqueue_copy(b_buf, bh)

    var a_lt = LayoutTensor[
        dtype, Layout.row_major(C.out_channels, K_TOTAL), MutAnyOrigin
    ](a_buf.unsafe_ptr())
    var b_lt = LayoutTensor[
        dtype, Layout.row_major(K_TOTAL, C.col_size), MutAnyOrigin
    ](b_buf.unsafe_ptr())
    var c_lt = LayoutTensor[
        dtype, Layout.row_major(C.out_channels, C.col_size), MutAnyOrigin
    ](c_buf.unsafe_ptr())

    comptime grid_x = (N + MMA_BLOCK_N - 1) // MMA_BLOCK_N
    comptime grid_y = (M + MMA_BLOCK_M - 1) // MMA_BLOCK_M

    @always_inline
    def dw_wrapper(
        c: LayoutTensor[
            dtype, Layout.row_major(C.out_channels, C.col_size), MutAnyOrigin
        ],
        a: LayoutTensor[
            dtype, Layout.row_major(C.out_channels, K_TOTAL), MutAnyOrigin
        ],
        b: LayoutTensor[
            dtype, Layout.row_major(K_TOTAL, C.col_size), MutAnyOrigin
        ],
    ):
        C.conv_matmul_dW_mma[K_TOTAL, dtype](c, a, b)

    ctx.enqueue_function[dw_wrapper, dw_wrapper](
        c_lt,
        a_lt,
        b_lt,
        grid_dim=(grid_x, grid_y),
        block_dim=(MMA_BLOCK_THREADS, 1),
    )

    var ch = ctx.enqueue_create_host_buffer[dtype](M * N)
    ctx.enqueue_copy(ch, c_buf)
    ctx.synchronize()

    for i in range(M * N):
        c_gpu[i] = ch[i]

    _ = compare(
        "isolated dW_mma (C=A@B) ["
        + String(M)
        + "x"
        + String(KK)
        + "] @ ["
        + String(KK)
        + "x"
        + String(N)
        + "]",
        c_ref,
        c_gpu,
        M * N,
    )

    a_ptr.free()
    b_ptr.free()
    c_ref.free()
    c_gpu.free()


def test_isolated_mma_dx[
    IC: Int, OC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, BATCH: Int
](ctx: DeviceContext) raises:
    """Test conv_matmul_dx_mma: dcol(col_size, K_TOTAL) = W.T(col_size, OC) @ grad(OC, K_TOTAL).

    W stored as (OC, col_size), used transposed.
    """
    comptime C = Conv2D[IC, OC, K, S, P, H, W]
    comptime K_TOTAL = BATCH * C.spatial_out
    comptime M = C.col_size
    comptime N = K_TOTAL
    comptime KK = OC

    # W stored as (OC, col_size) — will be transposed by kernel
    var w_ptr = alloc[Scalar[dtype]](KK * M)
    var g_ptr = alloc[Scalar[dtype]](KK * N)
    var c_ref = alloc[Scalar[dtype]](M * N)
    var c_gpu = alloc[Scalar[dtype]](M * N)

    for i in range(KK * M):
        w_ptr[i] = Scalar[dtype]((i % 7) + 1)
    for i in range(KK * N):
        g_ptr[i] = Scalar[dtype]((i % 3) + 1)

    # CPU reference: dcol[m, n] = sum_k( W.T[m, k] * grad[k, n] ) = sum_k( W[k, m] * grad[k, n] )
    for i in range(M):
        for j in range(N):
            var acc: Float64 = 0
            for k in range(KK):
                acc += Float64(w_ptr[k * M + i]) * Float64(g_ptr[k * N + j])
            c_ref[i * N + j] = Scalar[dtype](acc)

    var w_buf = ctx.enqueue_create_buffer[dtype](KK * M)
    var g_buf = ctx.enqueue_create_buffer[dtype](KK * N)
    var c_buf = ctx.enqueue_create_buffer[dtype](M * N)

    var wh = ctx.enqueue_create_host_buffer[dtype](KK * M)
    var gh = ctx.enqueue_create_host_buffer[dtype](KK * N)
    for i in range(KK * M):
        wh[i] = w_ptr[i]
    for i in range(KK * N):
        gh[i] = g_ptr[i]
    ctx.enqueue_copy(w_buf, wh)
    ctx.enqueue_copy(g_buf, gh)

    var c_lt = LayoutTensor[
        dtype, Layout.row_major(C.col_size, K_TOTAL), MutAnyOrigin
    ](c_buf.unsafe_ptr())
    var w_lt = LayoutTensor[
        dtype, Layout.row_major(C.out_channels, C.col_size), MutAnyOrigin
    ](w_buf.unsafe_ptr())
    var g_lt = LayoutTensor[
        dtype, Layout.row_major(C.out_channels, K_TOTAL), MutAnyOrigin
    ](g_buf.unsafe_ptr())

    comptime grid_x = (N + MMA_BLOCK_N - 1) // MMA_BLOCK_N
    comptime grid_y = (M + MMA_BLOCK_M - 1) // MMA_BLOCK_M

    @always_inline
    def dx_wrapper(
        c: LayoutTensor[
            dtype, Layout.row_major(C.col_size, K_TOTAL), MutAnyOrigin
        ],
        w: LayoutTensor[
            dtype, Layout.row_major(C.out_channels, C.col_size), MutAnyOrigin
        ],
        g: LayoutTensor[
            dtype, Layout.row_major(C.out_channels, K_TOTAL), MutAnyOrigin
        ],
    ):
        C.conv_matmul_dx_mma[K_TOTAL, dtype](c, w, g)

    ctx.enqueue_function[dx_wrapper, dx_wrapper](
        c_lt,
        w_lt,
        g_lt,
        grid_dim=(grid_x, grid_y),
        block_dim=(MMA_BLOCK_THREADS, 1),
    )

    var ch = ctx.enqueue_create_host_buffer[dtype](M * N)
    ctx.enqueue_copy(ch, c_buf)
    ctx.synchronize()

    for i in range(M * N):
        c_gpu[i] = ch[i]

    _ = compare(
        "isolated dx_mma (C=W.T@grad) ["
        + String(KK)
        + "x"
        + String(M)
        + "].T @ ["
        + String(KK)
        + "x"
        + String(N)
        + "]",
        c_ref,
        c_gpu,
        M * N,
    )

    w_ptr.free()
    g_ptr.free()
    c_ref.free()
    c_gpu.free()


# =========================================================================
# Test 3: Linear MMA (existing matmul_bias_act, for reference)
# =========================================================================


def test_linear_pipeline[
    IN: Int, OUT: Int, BATCH: Int
](ctx: DeviceContext,) raises:
    """Test Linear (FusedMatMulBiasReLU) CPU vs GPU with positive integer inputs.

    With positive inputs, ReLU is identity, so output = input @ W + b exactly.
    """
    comptime M = LinearReLU[IN, OUT]
    print("LinearReLU[", IN, ",", OUT, "] BATCH=", BATCH)

    var cpu_state = NetworkState[M, Adam[]]()

    # Set params to small positive integers
    var pv = cpu_state.params_view()
    for i in range(M.PARAM_SIZE):
        pv[i] = Scalar[dtype]((i % 7) + 1)

    var gpu = GPUNetworkState[M, Adam[], dtype](ctx)
    gpu.upload_from(cpu_state, ctx)

    # Positive integer input (so ReLU = identity)
    var input_ptr = alloc[Scalar[dtype]](BATCH * IN)
    for i in range(BATCH * IN):
        input_ptr[i] = Scalar[dtype]((i % 5) + 1)

    # CPU forward
    var cpu_out = alloc[Scalar[dtype]](BATCH * OUT)
    var cpu_cache = alloc[Scalar[dtype]](BATCH * M.CACHE_SIZE)
    for i in range(BATCH * OUT):
        cpu_out[i] = 0
    for i in range(BATCH * M.CACHE_SIZE):
        cpu_cache[i] = 0

    var in_lt = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](
        input_ptr
    )
    var out_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
    ](cpu_out)
    var ca_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
    ](cpu_cache)
    M.forward[BATCH](in_lt, out_lt, cpu_state.params_view(), ca_lt)

    # GPU forward
    var in_buf = ctx.enqueue_create_buffer[dtype](BATCH * IN)
    var out_buf = ctx.enqueue_create_buffer[dtype](BATCH * OUT)
    var ca_buf = ctx.enqueue_create_buffer[dtype](BATCH * M.CACHE_SIZE)
    comptime WS = BATCH * M.WORKSPACE_SIZE_PER_SAMPLE
    var ws_buf = ctx.enqueue_create_buffer[dtype](WS if WS > 0 else 1)

    var ih = ctx.enqueue_create_host_buffer[dtype](BATCH * IN)
    for i in range(BATCH * IN):
        ih[i] = input_ptr[i]
    ctx.enqueue_copy(in_buf, ih)

    var in_g = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](
        in_buf.unsafe_ptr()
    )
    var out_g = LayoutTensor[dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin](
        out_buf.unsafe_ptr()
    )
    var ca_g = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
    ](ca_buf.unsafe_ptr())

    M.forward_gpu[BATCH](ctx, out_g, in_g, gpu.params_view(), ca_g, ws_buf)

    var oh = ctx.enqueue_create_host_buffer[dtype](BATCH * OUT)
    ctx.enqueue_copy(oh, out_buf)
    ctx.synchronize()

    var gpu_out = alloc[Scalar[dtype]](BATCH * OUT)
    for i in range(BATCH * OUT):
        gpu_out[i] = oh[i]

    _ = compare("forward", cpu_out, gpu_out, BATCH * OUT)

    input_ptr.free()
    cpu_out.free()
    cpu_cache.free()
    gpu_out.free()


# =========================================================================
# Main
# =========================================================================


def main() raises:
    var ctx = DeviceContext()
    print("=== MMA Integer Correctness Test ===")
    print("(Small integers → exact TF32/FP32 → any diff = kernel bug)\n")

    print("--- Conv2D full pipeline (CPU vs GPU) ---")
    # Small conv (lots of zero-padding in MMA tiles)
    test_conv2d_pipeline[2, 4, 3, 1, 1, 5, 5, 2](ctx)
    print()
    # Larger conv (better tile utilization)
    test_conv2d_pipeline[4, 8, 3, 1, 1, 8, 8, 4](ctx)
    print()

    print("--- Linear pipeline (CPU vs GPU) ---")
    test_linear_pipeline[8, 4, 4](ctx)
    print()
    test_linear_pipeline[16, 8, 4](ctx)
    print()

    comptime if has_nvidia_gpu_accelerator():
        print("--- Isolated MMA kernels (NVIDIA only) ---")
        # Small dims (heavy zero-padding in MMA tiles)
        test_isolated_mma_fwd[2, 4, 3, 1, 1, 5, 5, 2](ctx)
        test_isolated_mma_dW[2, 4, 3, 1, 1, 5, 5, 2](ctx)
        test_isolated_mma_dx[2, 4, 3, 1, 1, 5, 5, 2](ctx)
        print()
        # Larger dims
        test_isolated_mma_fwd[4, 8, 3, 1, 1, 8, 8, 4](ctx)
        test_isolated_mma_dW[4, 8, 3, 1, 1, 8, 8, 4](ctx)
        test_isolated_mma_dx[4, 8, 3, 1, 1, 8, 8, 4](ctx)

    print("\n=== Done ===")

"""Progressive scale-up test to localize BatchNorm divergence.

Test A: single Conv2DBatchNormReLU[3,32,3,1,1,32,32] at BATCH=128 — CIFAR layer 0 dims.
Test B: two-layer stack (adds Conv2DBatchNormReLU[32,32,...]).
Test C: full 6-layer Conv stack (no FC head, no loss, no optimizer).

Inputs random ~N(0,1). Reports max|output|, max|grad_input|, max|grad_gamma|,
max|grad_beta|, max|grad_W|, max|grad_bias|. No fitness/training — just bound checks.
"""

from std.math import sqrt, log
from std.random import seed, random_float64
from std.memory import alloc, memset, UnsafePointer
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model.conv2d_bn_relu import Conv2DBatchNormReLU
from mojo_rl.nn.model.pool_layer import MaxPoolLayer
from mojo_rl.nn.model.sequential import Sequential
from mojo_rl.nn.initializer.initializers import Kaiming


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def _randn() -> Float64:
    # Box-Muller transform.
    var u1 = random_float64()
    var u2 = random_float64()
    # Guard against log(0)
    if u1 < 1e-12:
        u1 = 1e-12
    return sqrt(-2.0 * log(u1)) * _cos(6.283185307179586 * u2)


def _cos(x: Float64) -> Float64:
    # Use polynomial approximation via stdlib.
    from std.math import cos as stdlibcos
    return stdlibcos(x)


def _report_stats(name: String, buf: UnsafePointer[Scalar[dtype], ...], n: Int):
    var mx: Float64 = 0.0
    var mn: Float64 = 0.0
    var nan_ct: Int = 0
    var inf_ct: Int = 0
    for i in range(n):
        var v = Float64(buf[i])
        # NaN check: v != v
        if v != v:
            nan_ct += 1
            continue
        # inf check
        if v > 1e30 or v < -1e30:
            inf_ct += 1
            continue
        var av = v if v >= 0.0 else -v
        if av > mx:
            mx = av
        if v < mn:
            mn = v
    print(
        "  "
        + name
        + ": max|.|="
        + String(mx)
        + " min="
        + String(mn)
        + " nan="
        + String(nan_ct)
        + " inf="
        + String(inf_ct)
    )


def test_A_single_layer() raises:
    print("=" * 65)
    print("TEST A — single Conv2DBatchNormReLU[3,32,3,1,1,32,32] @ BATCH=128")
    print("=" * 65)

    comptime Layer = Conv2DBatchNormReLU[3, 32, 3, 1, 1, 32, 32]
    comptime BATCH = 128
    comptime IN_DIM = Layer.IN_DIM              # 3072
    comptime OUT_DIM = Layer.OUT_DIM            # 32768
    comptime PS = Layer.PARAM_SIZE
    comptime CS = Layer.CACHE_SIZE
    comptime WS = BATCH * Layer.WORKSPACE_SIZE_PER_SAMPLE

    print(
        "  IN_DIM="
        + String(IN_DIM)
        + " OUT_DIM="
        + String(OUT_DIM)
        + " PARAM_SIZE="
        + String(PS)
        + " CACHE_SIZE="
        + String(CS)
        + " WORKSPACE="
        + String(WS)
    )

    var ctx = DeviceContext()
    seed(42)

    # Kaiming-init params then override BN stats
    var gpu_params = ctx.enqueue_create_buffer[dtype](PS)
    var p_host = ctx.enqueue_create_host_buffer[dtype](PS)
    # Kaiming host-init
    var p_host_lt = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](
        p_host.unsafe_ptr()
    )
    Layer.initialize_params[Kaiming[]](p_host_lt)
    ctx.enqueue_copy(gpu_params, p_host)

    # Input ~N(0,1)
    var in_host = ctx.enqueue_create_host_buffer[dtype](BATCH * IN_DIM)
    for i in range(BATCH * IN_DIM):
        in_host[i] = Scalar[dtype](_randn())
    var gpu_input = ctx.enqueue_create_buffer[dtype](BATCH * IN_DIM)
    ctx.enqueue_copy(gpu_input, in_host)

    var gpu_output = ctx.enqueue_create_buffer[dtype](BATCH * OUT_DIM)
    var gpu_cache = ctx.enqueue_create_buffer[dtype](BATCH * CS)
    var gpu_ws = ctx.enqueue_create_buffer[dtype](WS if WS > 0 else 1)
    gpu_output.enqueue_fill(Scalar[dtype](0.0))
    gpu_cache.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    var out_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin](
        gpu_output.unsafe_ptr()
    )
    var in_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin](
        gpu_input.unsafe_ptr()
    )
    var p_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](
        gpu_params.unsafe_ptr()
    )
    var c_t = LayoutTensor[dtype, Layout.row_major(BATCH, CS), MutAnyOrigin](
        gpu_cache.unsafe_ptr()
    )
    var s_t = LayoutTensor[dtype, Layout.row_major(Layer.STATE_SIZE), MutAnyOrigin](
        UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=0)
    )

    Layer.forward_gpu[BATCH](ctx, out_t, in_t, p_t, s_t, c_t, gpu_ws)
    ctx.synchronize()

    var out_dl = ctx.enqueue_create_host_buffer[dtype](BATCH * OUT_DIM)
    ctx.enqueue_copy(out_dl, gpu_output)
    ctx.synchronize()
    print("[A] forward:")
    _report_stats("output", out_dl.unsafe_ptr(), BATCH * OUT_DIM)

    # grad_output ~N(0,1)
    var go_host = ctx.enqueue_create_host_buffer[dtype](BATCH * OUT_DIM)
    for i in range(BATCH * OUT_DIM):
        go_host[i] = Scalar[dtype](_randn())
    var gpu_go = ctx.enqueue_create_buffer[dtype](BATCH * OUT_DIM)
    ctx.enqueue_copy(gpu_go, go_host)
    var gpu_gi = ctx.enqueue_create_buffer[dtype](BATCH * IN_DIM)
    var gpu_gp = ctx.enqueue_create_buffer[dtype](PS)
    gpu_gi.enqueue_fill(Scalar[dtype](0.0))
    gpu_gp.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    var go_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin](
        gpu_go.unsafe_ptr()
    )
    var gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin](
        gpu_gi.unsafe_ptr()
    )
    var gp_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](
        gpu_gp.unsafe_ptr()
    )

    Layer.backward_gpu[BATCH](ctx, gi_t, go_t, p_t, s_t, c_t, gp_t, gpu_ws)
    ctx.synchronize()

    var gi_dl = ctx.enqueue_create_host_buffer[dtype](BATCH * IN_DIM)
    ctx.enqueue_copy(gi_dl, gpu_gi)
    var gp_dl = ctx.enqueue_create_host_buffer[dtype](PS)
    ctx.enqueue_copy(gp_dl, gpu_gp)
    ctx.synchronize()
    print("[A] backward:")
    _report_stats("grad_input", gi_dl.unsafe_ptr(), BATCH * IN_DIM)
    _report_stats(
        "grad_W",
        gp_dl.unsafe_ptr(),
        Layer.CONV_W_SIZE,
    )
    _report_stats(
        "grad_bias",
        gp_dl.unsafe_ptr() + Layer.BIAS_OFF,
        Layer.out_channels,
    )
    _report_stats(
        "grad_gamma",
        gp_dl.unsafe_ptr() + Layer.GAMMA_OFF,
        Layer.out_channels,
    )
    _report_stats(
        "grad_beta",
        gp_dl.unsafe_ptr() + Layer.BETA_OFF,
        Layer.out_channels,
    )


def test_B_two_layer() raises:
    print()
    print("=" * 65)
    print("TEST B — 2x Conv2DBatchNormReLU (3→32, 32→32) @ BATCH=128, 32x32")
    print("=" * 65)

    comptime Net = Sequential[
        Conv2DBatchNormReLU[3, 32, 3, 1, 1, 32, 32],
        Conv2DBatchNormReLU[32, 32, 3, 1, 1, 32, 32],
    ]
    comptime BATCH = 128
    comptime IN_DIM = Net.IN_DIM
    comptime OUT_DIM = Net.OUT_DIM
    comptime PS = Net.PARAM_SIZE
    comptime CS = Net.CACHE_SIZE
    comptime WS = BATCH * Net.WORKSPACE_SIZE_PER_SAMPLE
    print(
        "  IN_DIM="
        + String(IN_DIM)
        + " OUT_DIM="
        + String(OUT_DIM)
        + " PS="
        + String(PS)
        + " CS="
        + String(CS)
        + " WS="
        + String(WS)
    )

    var ctx = DeviceContext()
    seed(42)

    var gpu_params = ctx.enqueue_create_buffer[dtype](PS)
    var p_host = ctx.enqueue_create_host_buffer[dtype](PS)
    var p_host_lt = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](
        p_host.unsafe_ptr()
    )
    Net.initialize_params[Kaiming[]](p_host_lt)
    ctx.enqueue_copy(gpu_params, p_host)

    var in_host = ctx.enqueue_create_host_buffer[dtype](BATCH * IN_DIM)
    for i in range(BATCH * IN_DIM):
        in_host[i] = Scalar[dtype](_randn())
    var gpu_input = ctx.enqueue_create_buffer[dtype](BATCH * IN_DIM)
    ctx.enqueue_copy(gpu_input, in_host)

    var gpu_output = ctx.enqueue_create_buffer[dtype](BATCH * OUT_DIM)
    var gpu_cache = ctx.enqueue_create_buffer[dtype](BATCH * CS)
    var gpu_ws = ctx.enqueue_create_buffer[dtype](WS if WS > 0 else 1)
    gpu_output.enqueue_fill(Scalar[dtype](0.0))
    gpu_cache.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    var out_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin](
        gpu_output.unsafe_ptr()
    )
    var in_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin](
        gpu_input.unsafe_ptr()
    )
    var p_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](
        gpu_params.unsafe_ptr()
    )
    var c_t = LayoutTensor[dtype, Layout.row_major(BATCH, CS), MutAnyOrigin](
        gpu_cache.unsafe_ptr()
    )
    var s_t = LayoutTensor[dtype, Layout.row_major(Net.STATE_SIZE), MutAnyOrigin](
        UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=0)
    )

    Net.forward_gpu[BATCH](ctx, out_t, in_t, p_t, s_t, c_t, gpu_ws)
    ctx.synchronize()

    var out_dl = ctx.enqueue_create_host_buffer[dtype](BATCH * OUT_DIM)
    ctx.enqueue_copy(out_dl, gpu_output)
    ctx.synchronize()
    print("[B] forward:")
    _report_stats("output", out_dl.unsafe_ptr(), BATCH * OUT_DIM)

    var go_host = ctx.enqueue_create_host_buffer[dtype](BATCH * OUT_DIM)
    for i in range(BATCH * OUT_DIM):
        go_host[i] = Scalar[dtype](_randn())
    var gpu_go = ctx.enqueue_create_buffer[dtype](BATCH * OUT_DIM)
    ctx.enqueue_copy(gpu_go, go_host)
    var gpu_gi = ctx.enqueue_create_buffer[dtype](BATCH * IN_DIM)
    var gpu_gp = ctx.enqueue_create_buffer[dtype](PS)
    gpu_gi.enqueue_fill(Scalar[dtype](0.0))
    gpu_gp.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    var go_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin](
        gpu_go.unsafe_ptr()
    )
    var gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin](
        gpu_gi.unsafe_ptr()
    )
    var gp_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](
        gpu_gp.unsafe_ptr()
    )

    Net.backward_gpu[BATCH](ctx, gi_t, go_t, p_t, s_t, c_t, gp_t, gpu_ws)
    ctx.synchronize()

    var gi_dl = ctx.enqueue_create_host_buffer[dtype](BATCH * IN_DIM)
    ctx.enqueue_copy(gi_dl, gpu_gi)
    var gp_dl = ctx.enqueue_create_host_buffer[dtype](PS)
    ctx.enqueue_copy(gp_dl, gpu_gp)
    ctx.synchronize()
    print("[B] backward:")
    _report_stats("grad_input", gi_dl.unsafe_ptr(), BATCH * IN_DIM)
    _report_stats("grads_all", gp_dl.unsafe_ptr(), PS)


def test_C_deep_no_head() raises:
    print()
    print("=" * 65)
    print("TEST C — full 6x Conv2DBatchNormReLU + 3 MaxPool (no head)")
    print("=" * 65)

    comptime Net = Sequential[
        Conv2DBatchNormReLU[3, 32, 3, 1, 1, 32, 32],
        Conv2DBatchNormReLU[32, 32, 3, 1, 1, 32, 32],
        MaxPoolLayer[32, 32, 32, 2],
        Conv2DBatchNormReLU[32, 64, 3, 1, 1, 16, 16],
        Conv2DBatchNormReLU[64, 64, 3, 1, 1, 16, 16],
        MaxPoolLayer[64, 16, 16, 2],
        Conv2DBatchNormReLU[64, 128, 3, 1, 1, 8, 8],
        Conv2DBatchNormReLU[128, 128, 3, 1, 1, 8, 8],
        MaxPoolLayer[128, 8, 8, 2],
    ]
    comptime BATCH = 128
    comptime IN_DIM = Net.IN_DIM              # 3072
    comptime OUT_DIM = Net.OUT_DIM            # 128*4*4 = 2048
    comptime PS = Net.PARAM_SIZE
    comptime CS = Net.CACHE_SIZE
    comptime WS_PS = Net.WORKSPACE_SIZE_PER_SAMPLE
    print(
        "  IN_DIM="
        + String(IN_DIM)
        + " OUT_DIM="
        + String(OUT_DIM)
        + " PS="
        + String(PS)
        + " CS="
        + String(CS)
        + " WS_per_sample="
        + String(WS_PS)
    )

    var ctx = DeviceContext()
    seed(42)

    var gpu_params = ctx.enqueue_create_buffer[dtype](PS)
    var p_host = ctx.enqueue_create_host_buffer[dtype](PS)
    var p_host_lt = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](
        p_host.unsafe_ptr()
    )
    Net.initialize_params[Kaiming[]](p_host_lt)
    ctx.enqueue_copy(gpu_params, p_host)

    var in_host = ctx.enqueue_create_host_buffer[dtype](BATCH * IN_DIM)
    for i in range(BATCH * IN_DIM):
        in_host[i] = Scalar[dtype](_randn())
    var gpu_input = ctx.enqueue_create_buffer[dtype](BATCH * IN_DIM)
    ctx.enqueue_copy(gpu_input, in_host)

    var gpu_output = ctx.enqueue_create_buffer[dtype](BATCH * OUT_DIM)
    var gpu_cache = ctx.enqueue_create_buffer[dtype](BATCH * CS)
    var gpu_ws = ctx.enqueue_create_buffer[dtype](BATCH * WS_PS)
    gpu_output.enqueue_fill(Scalar[dtype](0.0))
    gpu_cache.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    var out_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin](
        gpu_output.unsafe_ptr()
    )
    var in_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin](
        gpu_input.unsafe_ptr()
    )
    var p_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](
        gpu_params.unsafe_ptr()
    )
    var c_t = LayoutTensor[dtype, Layout.row_major(BATCH, CS), MutAnyOrigin](
        gpu_cache.unsafe_ptr()
    )
    var s_t = LayoutTensor[dtype, Layout.row_major(Net.STATE_SIZE), MutAnyOrigin](
        UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=0)
    )

    Net.forward_gpu[BATCH](ctx, out_t, in_t, p_t, s_t, c_t, gpu_ws)
    ctx.synchronize()

    var out_dl = ctx.enqueue_create_host_buffer[dtype](BATCH * OUT_DIM)
    ctx.enqueue_copy(out_dl, gpu_output)
    ctx.synchronize()
    print("[C] forward:")
    _report_stats("output", out_dl.unsafe_ptr(), BATCH * OUT_DIM)

    var go_host = ctx.enqueue_create_host_buffer[dtype](BATCH * OUT_DIM)
    for i in range(BATCH * OUT_DIM):
        go_host[i] = Scalar[dtype](_randn())
    var gpu_go = ctx.enqueue_create_buffer[dtype](BATCH * OUT_DIM)
    ctx.enqueue_copy(gpu_go, go_host)
    var gpu_gi = ctx.enqueue_create_buffer[dtype](BATCH * IN_DIM)
    var gpu_gp = ctx.enqueue_create_buffer[dtype](PS)
    gpu_gi.enqueue_fill(Scalar[dtype](0.0))
    gpu_gp.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    var go_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin](
        gpu_go.unsafe_ptr()
    )
    var gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin](
        gpu_gi.unsafe_ptr()
    )
    var gp_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](
        gpu_gp.unsafe_ptr()
    )

    # NOTE: the MaxPool2D.backward_kernel_impl assumes grad_input is zero-init
    # but neither its kernel nor its vjp_gpu launcher zero it. In Sequential,
    # the inter workspace region reused as grad_input for MaxPool still holds
    # the forward activations from the forward pass (post-ReLU, so ≥ 0).
    # Uncomment the next two lines to confirm divergence disappears:
    # gpu_ws.enqueue_fill(Scalar[dtype](0.0))
    # ctx.synchronize()

    Net.backward_gpu[BATCH](ctx, gi_t, go_t, p_t, s_t, c_t, gp_t, gpu_ws)
    ctx.synchronize()

    var gi_dl = ctx.enqueue_create_host_buffer[dtype](BATCH * IN_DIM)
    ctx.enqueue_copy(gi_dl, gpu_gi)
    var gp_dl = ctx.enqueue_create_host_buffer[dtype](PS)
    ctx.enqueue_copy(gp_dl, gpu_gp)
    ctx.synchronize()
    print("[C] backward:")
    _report_stats("grad_input", gi_dl.unsafe_ptr(), BATCH * IN_DIM)
    _report_stats("grads_all", gp_dl.unsafe_ptr(), PS)

    # Per-layer param grad breakdown (only conv+BN layers have params)
    # Layer indices that are Conv2DBatchNormReLU: 0, 1, 3, 4, 6, 7 — but we need
    # offsets from Sequential._param_offset; compute manually below.

    # Channels at each BN layer:
    # L0: 32, L1: 32, L3: 64, L4: 64, L6: 128, L7: 128
    # Call each BN layer type:
    comptime L0 = Conv2DBatchNormReLU[3, 32, 3, 1, 1, 32, 32]
    comptime L1 = Conv2DBatchNormReLU[32, 32, 3, 1, 1, 32, 32]
    comptime L3 = Conv2DBatchNormReLU[32, 64, 3, 1, 1, 16, 16]
    comptime L4 = Conv2DBatchNormReLU[64, 64, 3, 1, 1, 16, 16]
    comptime L6 = Conv2DBatchNormReLU[64, 128, 3, 1, 1, 8, 8]
    comptime L7 = Conv2DBatchNormReLU[128, 128, 3, 1, 1, 8, 8]

    from mojo_rl.nn.constants import gpu_align
    # Manual offsets — pooling layers have PARAM_SIZE=0 (no padding)
    var off_L0 = 0
    var off_L1 = gpu_align(L0.PARAM_SIZE)
    var off_L3 = off_L1 + gpu_align(L1.PARAM_SIZE)  # L2 pool = 0
    var off_L4 = off_L3 + gpu_align(L3.PARAM_SIZE)
    var off_L6 = off_L4 + gpu_align(L4.PARAM_SIZE)  # L5 pool = 0
    var off_L7 = off_L6 + gpu_align(L6.PARAM_SIZE)

    print("  === per-layer param grad breakdown ===")
    print("  L0 (3->32, 32x32):")
    _report_stats("dW", gp_dl.unsafe_ptr() + off_L0 + L0.W_OFF, L0.CONV_W_SIZE)
    _report_stats("db", gp_dl.unsafe_ptr() + off_L0 + L0.BIAS_OFF, L0.out_channels)
    _report_stats("dgamma", gp_dl.unsafe_ptr() + off_L0 + L0.GAMMA_OFF, L0.out_channels)
    _report_stats("dbeta", gp_dl.unsafe_ptr() + off_L0 + L0.BETA_OFF, L0.out_channels)

    print("  L1 (32->32, 32x32):")
    _report_stats("dW", gp_dl.unsafe_ptr() + off_L1 + L1.W_OFF, L1.CONV_W_SIZE)
    _report_stats("db", gp_dl.unsafe_ptr() + off_L1 + L1.BIAS_OFF, L1.out_channels)
    _report_stats("dgamma", gp_dl.unsafe_ptr() + off_L1 + L1.GAMMA_OFF, L1.out_channels)
    _report_stats("dbeta", gp_dl.unsafe_ptr() + off_L1 + L1.BETA_OFF, L1.out_channels)

    print("  L3 (32->64, 16x16):")
    _report_stats("dW", gp_dl.unsafe_ptr() + off_L3 + L3.W_OFF, L3.CONV_W_SIZE)
    _report_stats("db", gp_dl.unsafe_ptr() + off_L3 + L3.BIAS_OFF, L3.out_channels)
    _report_stats("dgamma", gp_dl.unsafe_ptr() + off_L3 + L3.GAMMA_OFF, L3.out_channels)
    _report_stats("dbeta", gp_dl.unsafe_ptr() + off_L3 + L3.BETA_OFF, L3.out_channels)

    print("  L4 (64->64, 16x16):")
    _report_stats("dW", gp_dl.unsafe_ptr() + off_L4 + L4.W_OFF, L4.CONV_W_SIZE)
    _report_stats("db", gp_dl.unsafe_ptr() + off_L4 + L4.BIAS_OFF, L4.out_channels)
    _report_stats("dgamma", gp_dl.unsafe_ptr() + off_L4 + L4.GAMMA_OFF, L4.out_channels)
    _report_stats("dbeta", gp_dl.unsafe_ptr() + off_L4 + L4.BETA_OFF, L4.out_channels)

    print("  L6 (64->128, 8x8):")
    _report_stats("dW", gp_dl.unsafe_ptr() + off_L6 + L6.W_OFF, L6.CONV_W_SIZE)
    _report_stats("db", gp_dl.unsafe_ptr() + off_L6 + L6.BIAS_OFF, L6.out_channels)
    _report_stats("dgamma", gp_dl.unsafe_ptr() + off_L6 + L6.GAMMA_OFF, L6.out_channels)
    _report_stats("dbeta", gp_dl.unsafe_ptr() + off_L6 + L6.BETA_OFF, L6.out_channels)

    print("  L7 (128->128, 8x8):")
    _report_stats("dW", gp_dl.unsafe_ptr() + off_L7 + L7.W_OFF, L7.CONV_W_SIZE)
    _report_stats("db", gp_dl.unsafe_ptr() + off_L7 + L7.BIAS_OFF, L7.out_channels)
    _report_stats("dgamma", gp_dl.unsafe_ptr() + off_L7 + L7.GAMMA_OFF, L7.out_channels)
    _report_stats("dbeta", gp_dl.unsafe_ptr() + off_L7 + L7.BETA_OFF, L7.out_channels)


def main() raises:
    test_A_single_layer()
    test_B_two_layer()
    test_C_deep_no_head()

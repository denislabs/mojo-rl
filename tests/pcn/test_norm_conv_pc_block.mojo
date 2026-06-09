"""NormConvPCBlock validation — finite differences (CPU).

Validates the fused conv + input-side per-channel RMSNorm block (no separate
norm level). Transform μ = Conv(RMSNorm_ch(ACT(x_below))). FD checks:
  1. pull_back → act_derivative_mul must equal J^T ε (numerical via predict).
  2. weight_grad must equal dE/dW for E = ½‖x_above − μ‖² (conv weights+bias).

ACT=PCReLU with positive-biased inputs (ReLU active, no kinks → clean FD).

Run:
    pixi run mojo run -I . tests/pcn/test_norm_conv_pc_block.mojo
"""

from std.memory import alloc
from std.math import sin
from layout import Layout, LayoutTensor

from mojo_rl.nn.initializer import Xavier
from mojo_rl.experimental.pcn.pc_norm_conv_block import NormConvPCBlock
from mojo_rl.experimental.pcn import PCReLU

comptime dtype = DType.float32
comptime CB = NormConvPCBlock[2, 3, 3, 1, 1, 4, 4, PCReLU]
comptime BATCH = 2
comptime IN = CB.IN_DIM
comptime OUT = CB.OUT_DIM
comptime PSZ = CB.PARAM_SIZE


def _predict(
    x: LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin],
    params: LayoutTensor[dtype, Layout.row_major(PSZ), MutAnyOrigin],
    mut mu: LayoutTensor[dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin],
    mut a: LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin],
):
    CB.predict[BATCH, dtype](x, params, mu, a)


def main() raises:
    print("NormConvPCBlock validation (finite differences)\n")
    print("  IN=", IN, " OUT=", OUT, " PARAM_SIZE=", PSZ)

    var x_buf = alloc[Scalar[dtype]](BATCH * IN)
    var y_buf = alloc[Scalar[dtype]](BATCH * OUT)
    var params_buf = alloc[Scalar[dtype]](PSZ)
    var mu_buf = alloc[Scalar[dtype]](BATCH * OUT)
    var a_buf = alloc[Scalar[dtype]](BATCH * IN)
    var eps_buf = alloc[Scalar[dtype]](BATCH * OUT)
    var z_buf = alloc[Scalar[dtype]](BATCH * IN)
    var zeff_buf = alloc[Scalar[dtype]](BATCH * IN)
    var grads_buf = alloc[Scalar[dtype]](PSZ)

    for i in range(BATCH * IN):
        x_buf[i] = Scalar[dtype](sin(Float32(i) * 0.7 + 0.3) * 0.5 + 1.0)  # >0
    for i in range(BATCH * OUT):
        y_buf[i] = Scalar[dtype](sin(Float32(i) * 1.1 + 0.6))

    var x = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](x_buf)
    var y = LayoutTensor[dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin](y_buf)
    var params = LayoutTensor[dtype, Layout.row_major(PSZ), MutAnyOrigin](params_buf)
    var mu = LayoutTensor[dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin](mu_buf)
    var a = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](a_buf)
    var eps = LayoutTensor[dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin](eps_buf)
    var z = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](z_buf)
    var zeff = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](zeff_buf)
    var grads = LayoutTensor[dtype, Layout.row_major(PSZ), MutAnyOrigin](grads_buf)
    CB.initialize_params[Xavier[3], dtype](params)

    _predict(x, params, mu, a)
    CB.eps_compute[BATCH, dtype](y, mu, eps)
    CB.pull_back[BATCH, dtype](eps, params, z)
    CB.act_derivative_mul[BATCH, dtype](x, z, zeff)

    var h: Float64 = 1e-3
    var mu_p = alloc[Scalar[dtype]](BATCH * OUT)
    var mu_m = alloc[Scalar[dtype]](BATCH * OUT)
    var a_tmp = alloc[Scalar[dtype]](BATCH * IN)
    var mu_p_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin](mu_p)
    var mu_m_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin](mu_m)
    var a_tmp_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](a_tmp)

    var max_jvp_rel: Float64 = 0.0
    for b in range(BATCH):
        for j in range(IN):
            var saved = x_buf[b * IN + j]
            x_buf[b * IN + j] = Scalar[dtype](Float64(saved) + h)
            _predict(x, params, mu_p_t, a_tmp_t)
            x_buf[b * IN + j] = Scalar[dtype](Float64(saved) - h)
            _predict(x, params, mu_m_t, a_tmp_t)
            x_buf[b * IN + j] = saved
            var jtv: Float64 = 0.0
            for i in range(OUT):
                var dmu = (Float64(mu_p[b * OUT + i]) - Float64(mu_m[b * OUT + i])) / (2.0 * h)
                jtv += Float64(eps_buf[b * OUT + i]) * dmu
            var an = Float64(zeff_buf[b * IN + j])
            var rel = abs(jtv - an) / (abs(jtv) + abs(an) + 1e-6)
            if rel > max_jvp_rel:
                max_jvp_rel = rel
    print("\n  [pull_back∘act_deriv vs J^Tε]  max rel error =", max_jvp_rel)
    var jvp_ok = max_jvp_rel < 5e-2

    CB.weight_grad[BATCH, dtype](eps, a, grads)
    var max_g_rel: Float64 = 0.0
    for i in range(PSZ):
        var saved = params_buf[i]
        params_buf[i] = Scalar[dtype](Float64(saved) + h)
        _predict(x, params, mu_p_t, a_tmp_t)
        var e_p: Float64 = 0.0
        for k in range(BATCH * OUT):
            var d = Float64(y_buf[k]) - Float64(mu_p[k])
            e_p += 0.5 * d * d
        params_buf[i] = Scalar[dtype](Float64(saved) - h)
        _predict(x, params, mu_m_t, a_tmp_t)
        var e_m: Float64 = 0.0
        for k in range(BATCH * OUT):
            var d = Float64(y_buf[k]) - Float64(mu_m[k])
            e_m += 0.5 * d * d
        params_buf[i] = saved
        var g_fd = (e_p - e_m) / (2.0 * h)
        var g_an = Float64(grads_buf[i])
        var rel = abs(g_fd - g_an) / (abs(g_fd) + abs(g_an) + 1e-6)
        if rel > max_g_rel:
            max_g_rel = rel
    print("  [weight_grad vs FD dE/dW]      max rel error =", max_g_rel)
    var g_ok = max_g_rel < 5e-2

    print("")
    if jvp_ok and g_ok:
        print("✅ PASS — NormConvPCBlock matches finite differences")
    else:
        print("❌ FAIL")
        raise Error("NormConvPCBlock validation failed")

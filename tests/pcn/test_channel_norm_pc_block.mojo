"""ChannelNormPCBlock validation — finite differences (CPU).

Same FD checks as the global NormPCBlock test, for the per-channel variant:
  1. pull_back → act_derivative_mul must equal J^T ε for μ = γ_c⊙RMSNorm_c(x).
  2. weight_grad must equal dE/dγ for E = ½‖x_above − μ‖².

Run:
    pixi run mojo run -I . tests/pcn/test_channel_norm_pc_block.mojo
"""

from std.memory import alloc
from std.math import sin
from layout import Layout, LayoutTensor

from mojo_rl.experimental.pcn.pc_channel_norm_block import ChannelNormPCBlock

comptime dtype = DType.float32
comptime C = 2
comptime SP = 4
comptime DIM = C * SP  # 8
comptime BATCH = 3
comptime CB = ChannelNormPCBlock[C, SP]


def _predict_into(
    x: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    params: LayoutTensor[dtype, Layout.row_major(C), MutAnyOrigin],
    mut mu: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    mut a: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    CB.predict[BATCH, dtype](x, params, mu, a)


def main() raises:
    print("ChannelNormPCBlock validation (finite differences)\n")
    print("  C=", C, " spatial=", SP, " DIM=", DIM, " PARAM_SIZE=", CB.PARAM_SIZE)

    var x_buf = alloc[Scalar[dtype]](BATCH * DIM).as_unsafe_any_origin()
    var g_buf = alloc[Scalar[dtype]](C).as_unsafe_any_origin()
    var y_buf = alloc[Scalar[dtype]](BATCH * DIM).as_unsafe_any_origin()
    var mu_buf = alloc[Scalar[dtype]](BATCH * DIM).as_unsafe_any_origin()
    var a_buf = alloc[Scalar[dtype]](BATCH * DIM).as_unsafe_any_origin()
    var eps_buf = alloc[Scalar[dtype]](BATCH * DIM).as_unsafe_any_origin()
    var z_buf = alloc[Scalar[dtype]](BATCH * DIM).as_unsafe_any_origin()
    var zeff_buf = alloc[Scalar[dtype]](BATCH * DIM).as_unsafe_any_origin()
    var grads_buf = alloc[Scalar[dtype]](C).as_unsafe_any_origin()

    for i in range(BATCH * DIM):
        x_buf[i] = Scalar[dtype](sin(Float32(i) * 0.7 + 0.3) * 1.5 + 0.2)
        y_buf[i] = Scalar[dtype](sin(Float32(i) * 1.1 + 1.0))
    for c in range(C):
        g_buf[c] = Scalar[dtype](0.6 + 0.4 * sin(Float32(c) * 1.7))

    var x = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](x_buf)
    var g = LayoutTensor[dtype, Layout.row_major(C), MutAnyOrigin](g_buf)
    var mu = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](mu_buf)
    var a = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](a_buf)
    var eps = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](eps_buf)
    var z = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](z_buf)
    var zeff = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](zeff_buf)
    var grads = LayoutTensor[dtype, Layout.row_major(C), MutAnyOrigin](grads_buf)
    var y = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](y_buf)

    _predict_into(x, g, mu, a)
    CB.eps_compute[BATCH, dtype](y, mu, eps)
    CB.pull_back[BATCH, dtype](eps, g, z)
    CB.act_derivative_mul[BATCH, dtype](x, z, zeff)

    var h: Float64 = 1e-3
    var mu_p = alloc[Scalar[dtype]](BATCH * DIM).as_unsafe_any_origin()
    var mu_m = alloc[Scalar[dtype]](BATCH * DIM).as_unsafe_any_origin()
    var a_tmp = alloc[Scalar[dtype]](BATCH * DIM).as_unsafe_any_origin()
    var mu_p_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](mu_p)
    var mu_m_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](mu_m)
    var a_tmp_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](a_tmp)

    var max_jvp_rel: Float64 = 0.0
    for b in range(BATCH):
        for j in range(DIM):
            var saved = x_buf[b * DIM + j]
            x_buf[b * DIM + j] = Scalar[dtype](Float64(saved) + h)
            _predict_into(x, g, mu_p_t, a_tmp_t)
            x_buf[b * DIM + j] = Scalar[dtype](Float64(saved) - h)
            _predict_into(x, g, mu_m_t, a_tmp_t)
            x_buf[b * DIM + j] = saved
            var jtv: Float64 = 0.0
            for i in range(DIM):
                var dmu = (Float64(mu_p[b * DIM + i]) - Float64(mu_m[b * DIM + i])) / (2.0 * h)
                jtv += Float64(eps_buf[b * DIM + i]) * dmu
            var an = Float64(zeff_buf[b * DIM + j])
            var rel = abs(jtv - an) / (abs(jtv) + abs(an) + 1e-6)
            if rel > max_jvp_rel:
                max_jvp_rel = rel
    print("\n  [pull_back∘act_deriv vs J^Tε]  max rel error =", max_jvp_rel)
    var jvp_ok = max_jvp_rel < 5e-2

    CB.weight_grad[BATCH, dtype](eps, a, grads)
    var max_g_rel: Float64 = 0.0
    for c in range(C):
        var saved = g_buf[c]
        g_buf[c] = Scalar[dtype](Float64(saved) + h)
        _predict_into(x, g, mu_p_t, a_tmp_t)
        var e_p: Float64 = 0.0
        for k in range(BATCH * DIM):
            var d = Float64(y_buf[k]) - Float64(mu_p[k])
            e_p += 0.5 * d * d
        g_buf[c] = Scalar[dtype](Float64(saved) - h)
        _predict_into(x, g, mu_m_t, a_tmp_t)
        var e_m: Float64 = 0.0
        for k in range(BATCH * DIM):
            var d = Float64(y_buf[k]) - Float64(mu_m[k])
            e_m += 0.5 * d * d
        g_buf[c] = saved
        var g_fd = (e_p - e_m) / (2.0 * h)
        var g_an = Float64(grads_buf[c])
        var rel = abs(g_fd - g_an) / (abs(g_fd) + abs(g_an) + 1e-6)
        if rel > max_g_rel:
            max_g_rel = rel
    print("  [weight_grad vs FD dE/dγ]      max rel error =", max_g_rel)
    var g_ok = max_g_rel < 5e-2

    print("")
    if jvp_ok and g_ok:
        print("✅ PASS — ChannelNormPCBlock matches finite differences")
    else:
        print("❌ FAIL")
        raise Error("ChannelNormPCBlock validation failed")

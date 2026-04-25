"""Minimal test to localize BatchNorm bug: Conv2DBatchNormReLU fused path.

Compares Conv2DBatchNormReLU GPU forward vs. CPU forward (same struct), and
vs. an independent CPU reference that composes Conv2D + BatchNorm2D + ReLU
manually (using the BatchNorm2D struct directly, so mean/var/x_hat/scale/shift
is a different code path than the fused BN kernel).

This test is intentionally tiny (3→4 ch, 4x4 spatial, BATCH=4) so compile is
fast (~30s on Apple). No training. Fixed deterministic inputs.

Hypotheses tested (in order):
  H3: fused forward vs reference forward — should match within fp32 noise
  H1: running_mean/running_var sanity after one forward
  H2: gamma/beta/W/bias gradient magnitudes after one backward
"""

from std.math import sqrt
from std.memory import alloc, memset, UnsafePointer
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model.conv2d_bn_relu import Conv2DBatchNormReLU
from mojo_rl.nn.model.batch_norm_2d import BatchNorm2D


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def test_bn_single_layer() raises:
    print("=" * 60)
    print("TEST: Conv2DBatchNormReLU single-layer fused vs reference")
    print("=" * 60)

    comptime IC = 3
    comptime OC = 4
    comptime K = 3
    comptime S = 1
    comptime P = 1
    comptime H = 4
    comptime W = 4
    comptime BATCH = 4

    comptime Fused = Conv2DBatchNormReLU[IC, OC, K, S, P, H, W]

    comptime IN_DIM = Fused.IN_DIM                   # IC*H*W = 48
    comptime OUT_DIM = Fused.OUT_DIM                 # OC*H*W = 64 (stride=1, pad=1)
    comptime PS = Fused.PARAM_SIZE
    comptime CS = Fused.CACHE_SIZE
    comptime COL = Fused.col_size                    # IC*K*K = 27
    comptime SP = Fused.spatial_out                  # H*W = 16
    comptime WS = BATCH * Fused.WORKSPACE_SIZE_PER_SAMPLE

    print("IN_DIM=", IN_DIM, "OUT_DIM=", OUT_DIM, "PARAM_SIZE=", PS, "CACHE_SIZE=", CS)
    print("col_size=", COL, "spatial_out=", SP, "workspace=", WS)

    var ctx = DeviceContext()

    # --------------- Deterministic input ---------------
    # Host-side input_data we'll reuse for CPU path AND upload to GPU.
    var input_data = alloc[Scalar[dtype]](BATCH * IN_DIM)
    for i in range(BATCH * IN_DIM):
        # Small signed pattern around zero, std ~0.5
        input_data[i] = Scalar[dtype](Float64((i * 37) % 17) * 0.1 - 0.8)

    # --------------- Init params deterministically ---------------
    # Skip Kaiming (non-deterministic); use a simple explicit pattern so
    # CPU and GPU test the *same* weights.
    var params_host = alloc[Scalar[dtype]](PS)
    # conv_W: shape [OC, col_size] flattened
    for oc in range(OC):
        for k in range(COL):
            var v = Float64(((oc * 13 + k * 7) % 11)) * 0.05 - 0.25
            params_host[oc * COL + k] = Scalar[dtype](v)
    # conv_bias
    for oc in range(OC):
        params_host[Fused.BIAS_OFF + oc] = Scalar[dtype](Float64(oc) * 0.01)
    # gamma=1, beta=0, rmean=0, rvar=1
    for oc in range(OC):
        params_host[Fused.GAMMA_OFF + oc] = Scalar[dtype](1.0)
        params_host[Fused.BETA_OFF + oc] = Scalar[dtype](0.0)
        params_host[Fused.RMEAN_OFF + oc] = Scalar[dtype](0.0)
        params_host[Fused.RVAR_OFF + oc] = Scalar[dtype](1.0)

    # Save params before (running stats get modified by forward)
    var params_save = alloc[Scalar[dtype]](PS)
    for i in range(PS):
        params_save[i] = params_host[i]

    # =====================================================================
    # PATH A — Fused CPU forward (straight-line Conv→BN→ReLU in one call)
    # =====================================================================
    var cpu_params = alloc[Scalar[dtype]](PS)
    for i in range(PS):
        cpu_params[i] = params_host[i]
    var cpu_out = alloc[Scalar[dtype]](BATCH * OUT_DIM)
    var cpu_cache = alloc[Scalar[dtype]](BATCH * CS)
    memset(cpu_out, 0, BATCH * OUT_DIM)
    memset(cpu_cache, 0, BATCH * CS)

    var cpu_in_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin](input_data)
    var cpu_out_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin](cpu_out)
    var cpu_p_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](cpu_params)
    var cpu_c_t = LayoutTensor[dtype, Layout.row_major(BATCH, CS), MutAnyOrigin](cpu_cache)
    var cpu_s_t = LayoutTensor[dtype, Layout.row_major(Fused.STATE_SIZE), MutAnyOrigin](
        UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=0)
    )

    Fused.forward[BATCH](cpu_in_t, cpu_out_t, cpu_p_t, cpu_s_t, cpu_c_t)

    # Snapshot CPU running stats
    var cpu_rmean = alloc[Scalar[dtype]](OC)
    var cpu_rvar = alloc[Scalar[dtype]](OC)
    for oc in range(OC):
        cpu_rmean[oc] = cpu_params[Fused.RMEAN_OFF + oc]
        cpu_rvar[oc] = cpu_params[Fused.RVAR_OFF + oc]

    # =====================================================================
    # PATH B — Fused GPU forward
    # =====================================================================
    var gpu_input = ctx.enqueue_create_buffer[dtype](BATCH * IN_DIM)
    var gpu_output = ctx.enqueue_create_buffer[dtype](BATCH * OUT_DIM)
    var gpu_params = ctx.enqueue_create_buffer[dtype](PS)
    var gpu_cache = ctx.enqueue_create_buffer[dtype](BATCH * CS)
    var gpu_ws = ctx.enqueue_create_buffer[dtype](WS if WS > 0 else 1)

    var in_host = ctx.enqueue_create_host_buffer[dtype](BATCH * IN_DIM)
    for i in range(BATCH * IN_DIM):
        in_host[i] = input_data[i]
    ctx.enqueue_copy(gpu_input, in_host)

    var p_host = ctx.enqueue_create_host_buffer[dtype](PS)
    for i in range(PS):
        p_host[i] = params_save[i]
    ctx.enqueue_copy(gpu_params, p_host)

    gpu_output.enqueue_fill(Scalar[dtype](0.0))
    gpu_cache.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    var gpu_in_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin](gpu_input.unsafe_ptr())
    var gpu_out_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin](gpu_output.unsafe_ptr())
    var gpu_p_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](gpu_params.unsafe_ptr())
    var gpu_c_t = LayoutTensor[dtype, Layout.row_major(BATCH, CS), MutAnyOrigin](gpu_cache.unsafe_ptr())
    var gpu_s_t = LayoutTensor[dtype, Layout.row_major(Fused.STATE_SIZE), MutAnyOrigin](
        UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=0)
    )

    Fused.forward_gpu[BATCH](ctx, gpu_out_t, gpu_in_t, gpu_p_t, gpu_s_t, gpu_c_t, gpu_ws)
    ctx.synchronize()

    # Download GPU outputs + updated params + cache
    var gpu_out_dl = ctx.enqueue_create_host_buffer[dtype](BATCH * OUT_DIM)
    ctx.enqueue_copy(gpu_out_dl, gpu_output)
    var gpu_p_dl = ctx.enqueue_create_host_buffer[dtype](PS)
    ctx.enqueue_copy(gpu_p_dl, gpu_params)
    var gpu_c_dl = ctx.enqueue_create_host_buffer[dtype](BATCH * CS)
    ctx.enqueue_copy(gpu_c_dl, gpu_cache)
    ctx.synchronize()

    # Compare GPU-fused vs CPU-fused outputs
    var max_out_diff: Float64 = 0.0
    var max_out_idx: Int = 0
    for i in range(BATCH * OUT_DIM):
        var d = _abs(Float64(gpu_out_dl[i]) - Float64(cpu_out[i]))
        if d > max_out_diff:
            max_out_diff = d
            max_out_idx = i
    print("[fused GPU vs fused CPU] max |out diff| =", max_out_diff, " at idx", max_out_idx)
    if max_out_diff > 1e-4:
        print("  sample mismatches (first 8):")
        var printed = 0
        for i in range(BATCH * OUT_DIM):
            if _abs(Float64(gpu_out_dl[i]) - Float64(cpu_out[i])) > 1e-4:
                if printed < 8:
                    print("    idx", i, " GPU=", Float64(gpu_out_dl[i]), " CPU=", Float64(cpu_out[i]))
                    printed += 1

    # Compare x_hat region in cache
    var max_xhat_diff: Float64 = 0.0
    for b in range(BATCH):
        for i in range(OUT_DIM):
            var g = Float64(gpu_c_dl[b * CS + Fused.XHAT_OFF + i])
            var c = Float64(cpu_cache[b * CS + Fused.XHAT_OFF + i])
            var d = _abs(g - c)
            if d > max_xhat_diff:
                max_xhat_diff = d
    print("[fused GPU vs fused CPU] max |x_hat diff| =", max_xhat_diff)

    # Compare inv_std region in cache (replicated per sample)
    var max_inv_diff: Float64 = 0.0
    for b in range(BATCH):
        for oc in range(OC):
            var g = Float64(gpu_c_dl[b * CS + Fused.INVSTD_OFF + oc])
            var c = Float64(cpu_cache[b * CS + Fused.INVSTD_OFF + oc])
            var d = _abs(g - c)
            if d > max_inv_diff:
                max_inv_diff = d
    print("[fused GPU vs fused CPU] max |inv_std diff| =", max_inv_diff)

    # Compare running stats
    var max_rmean_diff: Float64 = 0.0
    var max_rvar_diff: Float64 = 0.0
    for oc in range(OC):
        var dm = _abs(Float64(gpu_p_dl[Fused.RMEAN_OFF + oc]) - Float64(cpu_rmean[oc]))
        var dv = _abs(Float64(gpu_p_dl[Fused.RVAR_OFF + oc]) - Float64(cpu_rvar[oc]))
        if dm > max_rmean_diff:
            max_rmean_diff = dm
        if dv > max_rvar_diff:
            max_rvar_diff = dv
    print("[fused GPU vs fused CPU] max |rmean diff| =", max_rmean_diff)
    print("[fused GPU vs fused CPU] max |rvar diff|  =", max_rvar_diff)

    # Print running stats for absolute sanity (H1)
    print()
    print("H1 check — running stats on CPU after one forward (expect rmean~0, rvar~0.9+0.1*batch_var):")
    for oc in range(OC):
        print("  c", oc, " rmean=", Float64(cpu_rmean[oc]), " rvar=", Float64(cpu_rvar[oc]))
    print()
    print("H1 check — running stats on GPU after one forward:")
    for oc in range(OC):
        print("  c", oc,
              " rmean=", Float64(gpu_p_dl[Fused.RMEAN_OFF + oc]),
              " rvar=", Float64(gpu_p_dl[Fused.RVAR_OFF + oc]))

    # =====================================================================
    # PATH C — Independent reference: manual Conv + BatchNorm2D + ReLU (all CPU)
    # Uses BatchNorm2D struct (different code path than Conv2DBatchNormReLU CPU).
    # =====================================================================
    # Step 1: manual conv with same W/bias as fused
    var ref_pre = alloc[Scalar[dtype]](BATCH * OUT_DIM)  # pre-BN activations
    memset(ref_pre, 0, BATCH * OUT_DIM)
    for b in range(BATCH):
        for oc in range(OC):
            for oh in range(H):
                for ow in range(W):
                    var s_ = oh * W + ow
                    var acc = Float64(params_save[Fused.BIAS_OFF + oc])
                    for c in range(IC):
                        for kh in range(K):
                            for kw in range(K):
                                var ih = oh * S - P + kh
                                var iw = ow * S - P + kw
                                if ih >= 0 and ih < H and iw >= 0 and iw < W:
                                    var w_idx = oc * COL + c * K * K + kh * K + kw
                                    var in_idx = c * H * W + ih * W + iw
                                    acc += Float64(params_save[w_idx]) * Float64(input_data[b * IN_DIM + in_idx])
                    ref_pre[b * OUT_DIM + oc * SP + s_] = Scalar[dtype](acc)

    # Step 2: BatchNorm2D CPU forward on ref_pre.
    # BatchNorm2D params layout: [gamma(C) | beta(C) | rmean(C) | rvar(C)]
    comptime BNRef = BatchNorm2D[OC, H, W]
    var bn_params = alloc[Scalar[dtype]](BNRef.PARAM_SIZE)
    for oc in range(OC):
        bn_params[BNRef.GAMMA_OFF + oc] = params_save[Fused.GAMMA_OFF + oc]
        bn_params[BNRef.BETA_OFF + oc] = params_save[Fused.BETA_OFF + oc]
        bn_params[BNRef.RMEAN_OFF + oc] = params_save[Fused.RMEAN_OFF + oc]
        bn_params[BNRef.RVAR_OFF + oc] = params_save[Fused.RVAR_OFF + oc]

    var bn_out = alloc[Scalar[dtype]](BATCH * OUT_DIM)
    memset(bn_out, 0, BATCH * OUT_DIM)
    var bn_cache = alloc[Scalar[dtype]](BATCH * BNRef.CACHE_SIZE)
    memset(bn_cache, 0, BATCH * BNRef.CACHE_SIZE)

    var ref_pre_t = LayoutTensor[dtype, Layout.row_major(BATCH, BNRef.IN_DIM), MutAnyOrigin](ref_pre)
    var bn_out_t = LayoutTensor[dtype, Layout.row_major(BATCH, BNRef.OUT_DIM), MutAnyOrigin](bn_out)
    var bn_p_t = LayoutTensor[dtype, Layout.row_major(BNRef.PARAM_SIZE), MutAnyOrigin](bn_params)
    var bn_c_t = LayoutTensor[dtype, Layout.row_major(BATCH, BNRef.CACHE_SIZE), MutAnyOrigin](bn_cache)
    var bn_s_t = LayoutTensor[dtype, Layout.row_major(BNRef.STATE_SIZE), MutAnyOrigin](
        UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=0)
    )
    BNRef.forward[BATCH](ref_pre_t, bn_out_t, bn_p_t, bn_s_t, bn_c_t)

    # Step 3: apply ReLU
    var ref_out = alloc[Scalar[dtype]](BATCH * OUT_DIM)
    for i in range(BATCH * OUT_DIM):
        var v = Float64(bn_out[i])
        ref_out[i] = Scalar[dtype](v if v > 0.0 else 0.0)

    # Compare fused CPU vs independent reference
    var max_fused_ref: Float64 = 0.0
    var max_fused_ref_idx: Int = 0
    for i in range(BATCH * OUT_DIM):
        var d = _abs(Float64(cpu_out[i]) - Float64(ref_out[i]))
        if d > max_fused_ref:
            max_fused_ref = d
            max_fused_ref_idx = i
    print()
    print("[fused CPU vs independent Conv+BN2D+ReLU] max |out diff| =", max_fused_ref,
          " at idx", max_fused_ref_idx)
    if max_fused_ref > 1e-4:
        print("  sample mismatches (first 8):")
        var printed = 0
        for i in range(BATCH * OUT_DIM):
            if _abs(Float64(cpu_out[i]) - Float64(ref_out[i])) > 1e-4:
                if printed < 8:
                    print("    idx", i, " fused=", Float64(cpu_out[i]),
                          " ref=", Float64(ref_out[i]))
                    printed += 1

    # Compare BN2D running stats to fused CPU running stats
    var max_rmean_ref: Float64 = 0.0
    var max_rvar_ref: Float64 = 0.0
    for oc in range(OC):
        var dm = _abs(Float64(bn_params[BNRef.RMEAN_OFF + oc]) - Float64(cpu_rmean[oc]))
        var dv = _abs(Float64(bn_params[BNRef.RVAR_OFF + oc]) - Float64(cpu_rvar[oc]))
        if dm > max_rmean_ref:
            max_rmean_ref = dm
        if dv > max_rvar_ref:
            max_rvar_ref = dv
    print("[fused CPU vs BN2D reference] max |rmean diff| =", max_rmean_ref)
    print("[fused CPU vs BN2D reference] max |rvar diff|  =", max_rvar_ref)

    # =====================================================================
    # H2 check — one backward with random-ish grad_output
    # =====================================================================
    print()
    print("H2 check — single backward pass, gradient magnitudes:")

    var grad_out_data = alloc[Scalar[dtype]](BATCH * OUT_DIM)
    for i in range(BATCH * OUT_DIM):
        grad_out_data[i] = Scalar[dtype](Float64((i * 11) % 7) * 0.1 - 0.3)

    # ---- CPU backward (fused) ----
    var cpu_grad_in = alloc[Scalar[dtype]](BATCH * IN_DIM)
    var cpu_grads = alloc[Scalar[dtype]](PS)
    memset(cpu_grad_in, 0, BATCH * IN_DIM)
    memset(cpu_grads, 0, PS)
    var go_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin](grad_out_data)
    var cgi_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin](cpu_grad_in)
    var cgp_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](cpu_grads)
    Fused.backward[BATCH](go_t, cgi_t, cpu_p_t, cpu_s_t, cpu_c_t, cgp_t)

    # ---- GPU backward ----
    var gpu_go = ctx.enqueue_create_buffer[dtype](BATCH * OUT_DIM)
    var gpu_gi = ctx.enqueue_create_buffer[dtype](BATCH * IN_DIM)
    var gpu_gp = ctx.enqueue_create_buffer[dtype](PS)

    var go_host = ctx.enqueue_create_host_buffer[dtype](BATCH * OUT_DIM)
    for i in range(BATCH * OUT_DIM):
        go_host[i] = grad_out_data[i]
    ctx.enqueue_copy(gpu_go, go_host)
    gpu_gi.enqueue_fill(Scalar[dtype](0.0))
    gpu_gp.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    var gpu_go_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin](gpu_go.unsafe_ptr())
    var gpu_gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin](gpu_gi.unsafe_ptr())
    var gpu_gp_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](gpu_gp.unsafe_ptr())

    Fused.backward_gpu[BATCH](ctx, gpu_gi_t, gpu_go_t, gpu_p_t, gpu_s_t, gpu_c_t, gpu_gp_t, gpu_ws)
    ctx.synchronize()

    var gpu_gi_dl = ctx.enqueue_create_host_buffer[dtype](BATCH * IN_DIM)
    ctx.enqueue_copy(gpu_gi_dl, gpu_gi)
    var gpu_gp_dl = ctx.enqueue_create_host_buffer[dtype](PS)
    ctx.enqueue_copy(gpu_gp_dl, gpu_gp)
    ctx.synchronize()

    # Report max abs diff of grad_input
    var max_gi_diff: Float64 = 0.0
    for i in range(BATCH * IN_DIM):
        var d = _abs(Float64(gpu_gi_dl[i]) - Float64(cpu_grad_in[i]))
        if d > max_gi_diff:
            max_gi_diff = d
    print("[fused GPU vs fused CPU] max |grad_input diff| =", max_gi_diff)

    # Per-region grad diffs: W, bias, gamma, beta, rmean (should be 0), rvar (should be 0)
    var max_w_diff: Float64 = 0.0
    for i in range(Fused.CONV_W_SIZE):
        var d = _abs(Float64(gpu_gp_dl[i]) - Float64(cpu_grads[i]))
        if d > max_w_diff:
            max_w_diff = d
    var max_b_diff: Float64 = 0.0
    for i in range(OC):
        var d = _abs(Float64(gpu_gp_dl[Fused.BIAS_OFF + i]) - Float64(cpu_grads[Fused.BIAS_OFF + i]))
        if d > max_b_diff:
            max_b_diff = d
    var max_g_diff: Float64 = 0.0
    var max_g_mag_cpu: Float64 = 0.0
    var max_g_mag_gpu: Float64 = 0.0
    for i in range(OC):
        var d = _abs(Float64(gpu_gp_dl[Fused.GAMMA_OFF + i]) - Float64(cpu_grads[Fused.GAMMA_OFF + i]))
        if d > max_g_diff:
            max_g_diff = d
        var mc = _abs(Float64(cpu_grads[Fused.GAMMA_OFF + i]))
        if mc > max_g_mag_cpu:
            max_g_mag_cpu = mc
        var mg = _abs(Float64(gpu_gp_dl[Fused.GAMMA_OFF + i]))
        if mg > max_g_mag_gpu:
            max_g_mag_gpu = mg
    var max_be_diff: Float64 = 0.0
    var max_be_mag_cpu: Float64 = 0.0
    var max_be_mag_gpu: Float64 = 0.0
    for i in range(OC):
        var d = _abs(Float64(gpu_gp_dl[Fused.BETA_OFF + i]) - Float64(cpu_grads[Fused.BETA_OFF + i]))
        if d > max_be_diff:
            max_be_diff = d
        var mc = _abs(Float64(cpu_grads[Fused.BETA_OFF + i]))
        if mc > max_be_mag_cpu:
            max_be_mag_cpu = mc
        var mg = _abs(Float64(gpu_gp_dl[Fused.BETA_OFF + i]))
        if mg > max_be_mag_gpu:
            max_be_mag_gpu = mg

    print("[fused GPU vs fused CPU] max |dW diff|    =", max_w_diff)
    print("[fused GPU vs fused CPU] max |dbias diff| =", max_b_diff)
    print("[fused GPU vs fused CPU] max |dgamma diff|=", max_g_diff)
    print("[fused GPU vs fused CPU] max |dbeta diff| =", max_be_diff)
    print("H2 magnitudes (expect O(1) for tiny input):")
    print("  max |dgamma| CPU =", max_g_mag_cpu, " GPU =", max_g_mag_gpu)
    print("  max |dbeta|  CPU =", max_be_mag_cpu, " GPU =", max_be_mag_gpu)

    # Per-channel dgamma dump
    print()
    print("Per-channel dgamma/dbeta (CPU / GPU):")
    for oc in range(OC):
        print("  c", oc,
              " dgamma CPU=", Float64(cpu_grads[Fused.GAMMA_OFF + oc]),
              " GPU=", Float64(gpu_gp_dl[Fused.GAMMA_OFF + oc]),
              " dbeta CPU=", Float64(cpu_grads[Fused.BETA_OFF + oc]),
              " GPU=", Float64(gpu_gp_dl[Fused.BETA_OFF + oc]))

    # Cleanup
    input_data.free()
    params_host.free()
    params_save.free()
    cpu_params.free()
    cpu_out.free()
    cpu_cache.free()
    cpu_rmean.free()
    cpu_rvar.free()
    ref_pre.free()
    bn_params.free()
    bn_out.free()
    bn_cache.free()
    ref_out.free()
    grad_out_data.free()
    cpu_grad_in.free()
    cpu_grads.free()

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


def main() raises:
    test_bn_single_layer()

"""Gradient check and CPU/GPU parity for BatchNorm1D.

Tests:
  1. CPU finite-difference gradient check (input + params).
  2. CPU vs GPU forward parity (training mode + inference mode).
  3. CPU vs GPU backward parity.
  4. Running-stats EMA: successive training forwards drive running stats
     toward the true batch stats, and inference-mode forward uses them.

Usage:
    pixi run -e apple mojo run -I . tests/nn/test_batch_norm_1d.mojo
    pixi run -e nvidia mojo run -I . tests/nn/test_batch_norm_1d.mojo
"""

from std.math import sqrt
from std.memory import alloc, memset
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import BatchNorm1D


def test_cpu_gradcheck() raises:
    """Finite-difference vs analytical backward (CPU)."""
    print("=" * 60)
    print("TEST: BatchNorm1D CPU gradient check (dim=6, batch=4)")
    print("=" * 60)

    comptime DIM = 6
    comptime BATCH = 4
    comptime BN = BatchNorm1D[DIM]
    comptime PS = BN.PARAM_SIZE  # 4*DIM = 24
    comptime CS = BN.CACHE_SIZE  # 3*DIM = 18

    var input_data = alloc[Scalar[dtype]](BATCH * DIM)
    for i in range(BATCH * DIM):
        input_data[i] = Scalar[dtype](Float64(i % 7) * 0.3 - 0.9)

    var params = alloc[Scalar[dtype]](PS)
    for f in range(DIM):
        params[f] = Scalar[dtype](1.0 + Float64(f) * 0.1)  # gamma varies
        params[DIM + f] = Scalar[dtype](Float64(f) * 0.05)  # beta varies
        params[2 * DIM + f] = Scalar[dtype](0.0)
        params[3 * DIM + f] = Scalar[dtype](1.0)

    # Forward (training)
    var output_data = alloc[Scalar[dtype]](BATCH * DIM)
    var cache_data = alloc[Scalar[dtype]](BATCH * CS)
    memset(output_data, 0, BATCH * DIM)
    memset(cache_data, 0, BATCH * CS)

    var inp = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](input_data)
    var out = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](output_data)
    var p = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](params)
    var c = LayoutTensor[dtype, Layout.row_major(BATCH, CS), MutAnyOrigin](cache_data)

    BN.forward[BATCH](inp, out, p, c)

    # Backward with dL/dy = 1 (L = sum(output))
    var grad_out = alloc[Scalar[dtype]](BATCH * DIM)
    for i in range(BATCH * DIM):
        grad_out[i] = Scalar[dtype](1.0)

    var grad_in = alloc[Scalar[dtype]](BATCH * DIM)
    var grad_params = alloc[Scalar[dtype]](PS)
    memset(grad_in, 0, BATCH * DIM)
    memset(grad_params, 0, PS)

    var go = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](grad_out)
    var gi = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](grad_in)
    var gp = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](grad_params)

    BN.backward[BATCH](go, gi, p, c, gp)

    # FD check for grad_input. Running stats are reset before each forward
    # so the EMA doesn't accumulate across FD probes.
    var eps_fd = Float64(1e-3)
    var max_diff_input: Float64 = 0.0
    for idx in range(BATCH * DIM):
        var orig = Float64(input_data[idx])

        input_data[idx] = Scalar[dtype](orig + eps_fd)
        for f in range(DIM):
            params[2 * DIM + f] = Scalar[dtype](0.0)
            params[3 * DIM + f] = Scalar[dtype](1.0)
        var out_plus = alloc[Scalar[dtype]](BATCH * DIM)
        var cache_plus = alloc[Scalar[dtype]](BATCH * CS)
        memset(out_plus, 0, BATCH * DIM)
        memset(cache_plus, 0, BATCH * CS)
        var op_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](out_plus)
        var cp_t = LayoutTensor[dtype, Layout.row_major(BATCH, CS), MutAnyOrigin](cache_plus)
        BN.forward[BATCH](inp, op_t, p, cp_t)
        var loss_plus: Float64 = 0.0
        for j in range(BATCH * DIM):
            loss_plus += Float64(out_plus[j])

        input_data[idx] = Scalar[dtype](orig - eps_fd)
        for f in range(DIM):
            params[2 * DIM + f] = Scalar[dtype](0.0)
            params[3 * DIM + f] = Scalar[dtype](1.0)
        var out_minus = alloc[Scalar[dtype]](BATCH * DIM)
        var cache_minus = alloc[Scalar[dtype]](BATCH * CS)
        memset(out_minus, 0, BATCH * DIM)
        memset(cache_minus, 0, BATCH * CS)
        var om_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](out_minus)
        var cm_t = LayoutTensor[dtype, Layout.row_major(BATCH, CS), MutAnyOrigin](cache_minus)
        BN.forward[BATCH](inp, om_t, p, cm_t)
        var loss_minus: Float64 = 0.0
        for j in range(BATCH * DIM):
            loss_minus += Float64(out_minus[j])

        input_data[idx] = Scalar[dtype](orig)

        var fd = (loss_plus - loss_minus) / (2.0 * eps_fd)
        var anal = Float64(grad_in[idx])
        var d = fd - anal
        if d < 0:
            d = -d
        if d > max_diff_input:
            max_diff_input = d

        out_plus.free()
        out_minus.free()
        cache_plus.free()
        cache_minus.free()

    print("Max |fd - analytical| for input:", max_diff_input)
    if max_diff_input < 0.01:
        print("PASS: Input gradient check")
    else:
        print("FAIL: Input gradient check (threshold 0.01)")

    # FD check for gamma + beta (skip running stats)
    var max_diff_params: Float64 = 0.0
    for pidx in range(2 * DIM):
        var orig = Float64(params[pidx])

        params[pidx] = Scalar[dtype](orig + eps_fd)
        for f in range(DIM):
            params[2 * DIM + f] = Scalar[dtype](0.0)
            params[3 * DIM + f] = Scalar[dtype](1.0)
        var out_pp = alloc[Scalar[dtype]](BATCH * DIM)
        var cache_pp = alloc[Scalar[dtype]](BATCH * CS)
        memset(out_pp, 0, BATCH * DIM)
        memset(cache_pp, 0, BATCH * CS)
        var opp_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](out_pp)
        var cpp_t = LayoutTensor[dtype, Layout.row_major(BATCH, CS), MutAnyOrigin](cache_pp)
        BN.forward[BATCH](inp, opp_t, p, cpp_t)
        var lp: Float64 = 0.0
        for j in range(BATCH * DIM):
            lp += Float64(out_pp[j])

        params[pidx] = Scalar[dtype](orig - eps_fd)
        for f in range(DIM):
            params[2 * DIM + f] = Scalar[dtype](0.0)
            params[3 * DIM + f] = Scalar[dtype](1.0)
        var out_pm = alloc[Scalar[dtype]](BATCH * DIM)
        var cache_pm = alloc[Scalar[dtype]](BATCH * CS)
        memset(out_pm, 0, BATCH * DIM)
        memset(cache_pm, 0, BATCH * CS)
        var opm_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](out_pm)
        var cpm_t = LayoutTensor[dtype, Layout.row_major(BATCH, CS), MutAnyOrigin](cache_pm)
        BN.forward[BATCH](inp, opm_t, p, cpm_t)
        var lm: Float64 = 0.0
        for j in range(BATCH * DIM):
            lm += Float64(out_pm[j])

        params[pidx] = Scalar[dtype](orig)

        var fd = (lp - lm) / (2.0 * eps_fd)
        var anal = Float64(grad_params[pidx])
        var d = fd - anal
        if d < 0:
            d = -d
        if d > max_diff_params:
            max_diff_params = d

        out_pp.free()
        out_pm.free()
        cache_pp.free()
        cache_pm.free()

    print("Max |fd - analytical| for params:", max_diff_params)
    if max_diff_params < 0.01:
        print("PASS: Param gradient check")
    else:
        print("FAIL: Param gradient check (threshold 0.01)")

    input_data.free()
    params.free()
    output_data.free()
    cache_data.free()
    grad_out.free()
    grad_in.free()
    grad_params.free()


def test_cpu_vs_gpu() raises:
    """Forward + backward parity: CPU vs GPU."""
    print()
    print("=" * 60)
    print("TEST: BatchNorm1D CPU vs GPU (dim=32, batch=16)")
    print("=" * 60)

    var ctx = DeviceContext()

    comptime DIM = 32
    comptime BATCH = 16
    comptime BN = BatchNorm1D[DIM]
    comptime PS = BN.PARAM_SIZE
    comptime CS = BN.CACHE_SIZE

    var input_data = alloc[Scalar[dtype]](BATCH * DIM)
    for i in range(BATCH * DIM):
        input_data[i] = Scalar[dtype](Float64(i % 13) * 0.2 - 1.2)

    var params_init = alloc[Scalar[dtype]](PS)
    for f in range(DIM):
        params_init[f] = Scalar[dtype](1.0 + Float64(f) * 0.05)
        params_init[DIM + f] = Scalar[dtype](Float64(f) * 0.03)
        params_init[2 * DIM + f] = Scalar[dtype](0.1 * Float64(f % 4))
        params_init[3 * DIM + f] = Scalar[dtype](1.0 + 0.1 * Float64(f % 3))

    # --- CPU forward (training) ---
    var cpu_out = alloc[Scalar[dtype]](BATCH * DIM)
    var cpu_cache = alloc[Scalar[dtype]](BATCH * CS)
    var cpu_params = alloc[Scalar[dtype]](PS)
    for i in range(PS):
        cpu_params[i] = params_init[i]
    memset(cpu_out, 0, BATCH * DIM)
    memset(cpu_cache, 0, BATCH * CS)

    var inp_cpu = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](input_data)
    var out_cpu = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](cpu_out)
    var pcpu_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](cpu_params)
    var ccpu_t = LayoutTensor[dtype, Layout.row_major(BATCH, CS), MutAnyOrigin](cpu_cache)

    BN.forward[BATCH](inp_cpu, out_cpu, pcpu_t, ccpu_t)

    # --- GPU forward (training) ---
    var gpu_input = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
    var gpu_output = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
    var gpu_params = ctx.enqueue_create_buffer[dtype](PS)
    var gpu_cache = ctx.enqueue_create_buffer[dtype](BATCH * CS)
    var gpu_workspace = ctx.enqueue_create_buffer[dtype](1)

    ctx.enqueue_copy(gpu_input, input_data)
    ctx.enqueue_copy(gpu_params, params_init)
    ctx.enqueue_memset(gpu_output, Scalar[dtype](0.0))
    ctx.enqueue_memset(gpu_cache, Scalar[dtype](0.0))

    var ginp_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](gpu_input.unsafe_ptr())
    var gout_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](gpu_output.unsafe_ptr())
    var gpar_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](gpu_params.unsafe_ptr())
    var gcac_t = LayoutTensor[dtype, Layout.row_major(BATCH, CS), MutAnyOrigin](gpu_cache.unsafe_ptr())

    BN.forward_gpu[BATCH](ctx, gout_t, ginp_t, gpar_t, gcac_t, gpu_workspace)
    ctx.synchronize()

    var gpu_out_h = alloc[Scalar[dtype]](BATCH * DIM)
    var gpu_params_h = alloc[Scalar[dtype]](PS)
    ctx.enqueue_copy(gpu_out_h, gpu_output)
    ctx.enqueue_copy(gpu_params_h, gpu_params)
    ctx.synchronize()

    var max_fwd_diff: Float64 = 0.0
    for i in range(BATCH * DIM):
        var d = Float64(cpu_out[i]) - Float64(gpu_out_h[i])
        if d < 0:
            d = -d
        if d > max_fwd_diff:
            max_fwd_diff = d
    print("Max |cpu - gpu| forward:", max_fwd_diff)
    if max_fwd_diff < 1e-4:
        print("PASS: Forward parity")
    else:
        print("FAIL: Forward parity (threshold 1e-4)")

    var max_rstat_diff: Float64 = 0.0
    for i in range(2 * DIM, 4 * DIM):
        var d = Float64(cpu_params[i]) - Float64(gpu_params_h[i])
        if d < 0:
            d = -d
        if d > max_rstat_diff:
            max_rstat_diff = d
    print("Max |cpu - gpu| running stats:", max_rstat_diff)
    if max_rstat_diff < 1e-4:
        print("PASS: Running stats parity")
    else:
        print("FAIL: Running stats parity (threshold 1e-4)")

    # --- Backward ---
    var grad_out_data = alloc[Scalar[dtype]](BATCH * DIM)
    for i in range(BATCH * DIM):
        grad_out_data[i] = Scalar[dtype](0.5 + Float64(i % 5) * 0.2)

    # CPU backward
    var cpu_gin = alloc[Scalar[dtype]](BATCH * DIM)
    var cpu_gp = alloc[Scalar[dtype]](PS)
    memset(cpu_gin, 0, BATCH * DIM)
    memset(cpu_gp, 0, PS)
    var go_cpu = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](grad_out_data)
    var gi_cpu = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](cpu_gin)
    var gp_cpu = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](cpu_gp)
    BN.backward[BATCH](go_cpu, gi_cpu, pcpu_t, ccpu_t, gp_cpu)

    # GPU backward (reuse gpu_cache + gpu_params from training forward)
    var gpu_grad_out = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
    var gpu_grad_in = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
    var gpu_grad_params = ctx.enqueue_create_buffer[dtype](PS)
    ctx.enqueue_copy(gpu_grad_out, grad_out_data)
    ctx.enqueue_memset(gpu_grad_in, Scalar[dtype](0.0))
    ctx.enqueue_memset(gpu_grad_params, Scalar[dtype](0.0))

    var ggo_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](gpu_grad_out.unsafe_ptr())
    var ggi_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](gpu_grad_in.unsafe_ptr())
    var ggp_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](gpu_grad_params.unsafe_ptr())

    BN.backward_gpu[BATCH](ctx, ggi_t, ggo_t, gpar_t, gcac_t, ggp_t, gpu_workspace)
    ctx.synchronize()

    var gpu_gin_h = alloc[Scalar[dtype]](BATCH * DIM)
    var gpu_gp_h = alloc[Scalar[dtype]](PS)
    ctx.enqueue_copy(gpu_gin_h, gpu_grad_in)
    ctx.enqueue_copy(gpu_gp_h, gpu_grad_params)
    ctx.synchronize()

    var max_gin_diff: Float64 = 0.0
    for i in range(BATCH * DIM):
        var d = Float64(cpu_gin[i]) - Float64(gpu_gin_h[i])
        if d < 0:
            d = -d
        if d > max_gin_diff:
            max_gin_diff = d
    print("Max |cpu - gpu| grad_input:", max_gin_diff)
    if max_gin_diff < 1e-3:
        print("PASS: Backward grad_input parity")
    else:
        print("FAIL: Backward grad_input parity (threshold 1e-3)")

    var max_gp_diff: Float64 = 0.0
    for i in range(2 * DIM):  # only gamma + beta
        var d = Float64(cpu_gp[i]) - Float64(gpu_gp_h[i])
        if d < 0:
            d = -d
        if d > max_gp_diff:
            max_gp_diff = d
    print("Max |cpu - gpu| grad_params (gamma+beta):", max_gp_diff)
    if max_gp_diff < 1e-3:
        print("PASS: Backward grad_params parity")
    else:
        print("FAIL: Backward grad_params parity (threshold 1e-3)")

    input_data.free()
    params_init.free()
    cpu_out.free()
    cpu_cache.free()
    cpu_params.free()
    gpu_out_h.free()
    gpu_params_h.free()
    grad_out_data.free()
    cpu_gin.free()
    cpu_gp.free()
    gpu_gin_h.free()
    gpu_gp_h.free()


def test_inference_mode_cpu_vs_gpu() raises:
    """Inference forward (no cache, uses running stats): CPU vs GPU."""
    print()
    print("=" * 60)
    print("TEST: BatchNorm1D inference mode CPU vs GPU (dim=16, batch=8)")
    print("=" * 60)

    var ctx = DeviceContext()

    comptime DIM = 16
    comptime BATCH = 8
    comptime BN = BatchNorm1D[DIM]
    comptime PS = BN.PARAM_SIZE

    # Use non-trivial running stats (simulate post-training)
    var params = alloc[Scalar[dtype]](PS)
    for f in range(DIM):
        params[f] = Scalar[dtype](1.0 + Float64(f) * 0.02)  # gamma
        params[DIM + f] = Scalar[dtype](Float64(f) * 0.01)   # beta
        params[2 * DIM + f] = Scalar[dtype](0.1 + Float64(f) * 0.03)  # rmean
        params[3 * DIM + f] = Scalar[dtype](0.5 + Float64(f) * 0.02)  # rvar

    var input_data = alloc[Scalar[dtype]](BATCH * DIM)
    for i in range(BATCH * DIM):
        input_data[i] = Scalar[dtype](Float64(i % 11) * 0.15)

    # CPU inference forward
    var cpu_out = alloc[Scalar[dtype]](BATCH * DIM)
    memset(cpu_out, 0, BATCH * DIM)
    var inp_cpu = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](input_data)
    var oc_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](cpu_out)
    var pc_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](params)
    BN.forward[BATCH](inp_cpu, oc_t, pc_t)  # inference overload (no cache)

    # GPU inference forward
    var gpu_in = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
    var gpu_out = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
    var gpu_params = ctx.enqueue_create_buffer[dtype](PS)
    var gpu_ws = ctx.enqueue_create_buffer[dtype](1)
    ctx.enqueue_copy(gpu_in, input_data)
    ctx.enqueue_copy(gpu_params, params)
    ctx.enqueue_memset(gpu_out, Scalar[dtype](0.0))

    var gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](gpu_in.unsafe_ptr())
    var go_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](gpu_out.unsafe_ptr())
    var gp_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](gpu_params.unsafe_ptr())

    BN.forward_gpu_no_cache[BATCH](ctx, go_t, gi_t, gp_t, gpu_ws)
    ctx.synchronize()

    var gpu_out_h = alloc[Scalar[dtype]](BATCH * DIM)
    ctx.enqueue_copy(gpu_out_h, gpu_out)
    ctx.synchronize()

    var max_diff: Float64 = 0.0
    for i in range(BATCH * DIM):
        var d = Float64(cpu_out[i]) - Float64(gpu_out_h[i])
        if d < 0:
            d = -d
        if d > max_diff:
            max_diff = d
    print("Max |cpu - gpu| inference forward:", max_diff)
    if max_diff < 1e-4:
        print("PASS: Inference forward parity")
    else:
        print("FAIL: Inference forward parity (threshold 1e-4)")

    params.free()
    input_data.free()
    cpu_out.free()
    gpu_out_h.free()


def test_running_stats_ema() raises:
    """After many training forwards with constant input, running stats should
    converge toward true batch stats (mean, var)."""
    print()
    print("=" * 60)
    print("TEST: BatchNorm1D running-stats EMA convergence")
    print("=" * 60)

    comptime DIM = 4
    comptime BATCH = 8
    comptime BN = BatchNorm1D[DIM]
    comptime PS = BN.PARAM_SIZE
    comptime CS = BN.CACHE_SIZE
    comptime N_STEPS = 200

    var input_data = alloc[Scalar[dtype]](BATCH * DIM)
    # Feature-wise target stats: feature f has mean=f, variance=(f+1)
    # Construct: x[b,f] = f + sqrt(f+1) * z_b where z_b is zero-mean unit var.
    # Use a simple deterministic pattern that approximates this per feature.
    var means_true = alloc[Scalar[dtype]](DIM)
    var vars_true = alloc[Scalar[dtype]](DIM)
    for f in range(DIM):
        var sum_: Float64 = 0.0
        var sum_sq: Float64 = 0.0
        for b in range(BATCH):
            var z = Float64((b * 7 + f * 3) % 11) / 10.0 - 0.5  # in [-0.5, 0.5]
            var x = Float64(f) + (Float64(f) + 1.0) * z
            input_data[b * DIM + f] = Scalar[dtype](x)
            sum_ += x
            sum_sq += x * x
        var mean_f = sum_ / Float64(BATCH)
        var var_f = sum_sq / Float64(BATCH) - mean_f * mean_f
        means_true[f] = Scalar[dtype](mean_f)
        vars_true[f] = Scalar[dtype](var_f)

    var params = alloc[Scalar[dtype]](PS)
    # Manual init: gamma=1, beta=0, running_mean=0, running_var=1
    for f in range(DIM):
        params[f] = Scalar[dtype](1.0)
        params[DIM + f] = Scalar[dtype](0.0)
        params[2 * DIM + f] = Scalar[dtype](0.0)
        params[3 * DIM + f] = Scalar[dtype](1.0)

    var out = alloc[Scalar[dtype]](BATCH * DIM)
    var cache = alloc[Scalar[dtype]](BATCH * CS)
    var inp_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](input_data)
    var out_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](out)
    var p_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](params)
    var c_t = LayoutTensor[dtype, Layout.row_major(BATCH, CS), MutAnyOrigin](cache)

    for _ in range(N_STEPS):
        memset(out, 0, BATCH * DIM)
        memset(cache, 0, BATCH * CS)
        BN.forward[BATCH](inp_t, out_t, p_t, c_t)

    var max_rmean_err: Float64 = 0.0
    var max_rvar_err: Float64 = 0.0
    for f in range(DIM):
        var rm = Float64(params[2 * DIM + f])
        var rv = Float64(params[3 * DIM + f])
        var dm = rm - Float64(means_true[f])
        var dv = rv - Float64(vars_true[f])
        if dm < 0:
            dm = -dm
        if dv < 0:
            dv = -dv
        if dm > max_rmean_err:
            max_rmean_err = dm
        if dv > max_rvar_err:
            max_rvar_err = dv

    print("Max |running_mean - true_mean|:", max_rmean_err)
    print("Max |running_var  - true_var |:", max_rvar_err)
    if max_rmean_err < 0.01 and max_rvar_err < 0.01:
        print("PASS: EMA convergence")
    else:
        print("FAIL: EMA convergence (threshold 0.01)")

    input_data.free()
    means_true.free()
    vars_true.free()
    params.free()
    out.free()
    cache.free()


def main() raises:
    test_cpu_gradcheck()
    test_cpu_vs_gpu()
    test_inference_mode_cpu_vs_gpu()
    test_running_stats_ema()

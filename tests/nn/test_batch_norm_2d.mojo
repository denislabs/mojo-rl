"""Gradient check for BatchNorm2D: finite-difference vs analytical backward.

Tests both standalone BatchNorm2D and Conv2D+BN+ReLU pipeline.
"""

from std.math import exp, log, sqrt
from std.memory import alloc, memset
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training import NetworkState
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.model import (
    BatchNorm2D,
    Conv2DLayer,
    ReLU,
    Linear,
    Sequential,
    Parallel,
    FlattenLayer,
)
from mojo_rl.nn.optimizer import Adam


def test_bn_gradient_check() raises:
    """Finite-difference gradient check for BatchNorm2D."""
    print("=" * 60)
    print("TEST: BatchNorm2D gradient check (C=4, H=2, W=2)")
    print("=" * 60)

    comptime C = 4
    comptime H = 2
    comptime W = 2
    comptime BN = BatchNorm2D[C, H, W]
    comptime BATCH = 3
    comptime DIM = C * H * W  # 16
    comptime PS = BN.PARAM_SIZE  # 4*C = 16
    comptime CS = BN.CACHE_SIZE

    # Random input
    var input_data = alloc[Scalar[dtype]](BATCH * DIM)
    for i in range(BATCH * DIM):
        input_data[i] = Scalar[dtype](Float64(i % 7) * 0.3 - 0.9)

    # Init params: gamma=1, beta=0, rmean=0, rvar=1
    var params = alloc[Scalar[dtype]](PS)
    for c in range(C):
        params[c] = Scalar[dtype](1.0)          # gamma
        params[C + c] = Scalar[dtype](0.0)       # beta
        params[2*C + c] = Scalar[dtype](0.0)     # running_mean
        params[3*C + c] = Scalar[dtype](1.0)     # running_var

    # Forward
    var output_data = alloc[Scalar[dtype]](BATCH * DIM)
    var cache_data = alloc[Scalar[dtype]](BATCH * CS)
    memset(output_data, 0, BATCH * DIM)
    memset(cache_data, 0, BATCH * CS)

    var inp = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](input_data)
    var out = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](output_data)
    var p = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](params)
    var c = LayoutTensor[dtype, Layout.row_major(BATCH, CS), MutAnyOrigin](cache_data)

    BN.forward[BATCH](inp, out, p, c)

    print("Output sample 0 (first 8):", end="")
    for i in range(8):
        print("", Float64(Int(Float64(output_data[i]) * 1000)) / 1000.0, end="")
    print()

    # Backward with unit gradient
    var grad_out = alloc[Scalar[dtype]](BATCH * DIM)
    for i in range(BATCH * DIM):
        grad_out[i] = Scalar[dtype](1.0)  # dL/dy = 1 everywhere

    var grad_in = alloc[Scalar[dtype]](BATCH * DIM)
    var grad_params = alloc[Scalar[dtype]](PS)
    memset(grad_in, 0, BATCH * DIM)
    memset(grad_params, 0, PS)

    var go = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](grad_out)
    var gi = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](grad_in)
    var gp = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](grad_params)

    BN.backward[BATCH](go, gi, p, c, gp)

    # Finite-difference check for grad_input
    var eps_fd = Float64(1e-3)
    var max_diff_input: Float64 = 0.0
    for idx in range(BATCH * DIM):
        var orig = Float64(input_data[idx])

        # f(x + eps)
        input_data[idx] = Scalar[dtype](orig + eps_fd)
        var out_plus = alloc[Scalar[dtype]](BATCH * DIM)
        memset(out_plus, 0, BATCH * DIM)
        # Need fresh cache for each forward
        var cache_plus = alloc[Scalar[dtype]](BATCH * CS)
        memset(cache_plus, 0, BATCH * CS)
        var inp_p = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](input_data)
        var out_p = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](out_plus)
        var c_p = LayoutTensor[dtype, Layout.row_major(BATCH, CS), MutAnyOrigin](cache_plus)
        # Reset running stats for clean forward
        params[2*C] = 0; params[2*C+1] = 0; params[2*C+2] = 0; params[2*C+3] = 0
        params[3*C] = 1; params[3*C+1] = 1; params[3*C+2] = 1; params[3*C+3] = 1
        BN.forward[BATCH](inp_p, out_p, p, c_p)
        var loss_plus: Float64 = 0.0
        for j in range(BATCH * DIM):
            loss_plus += Float64(out_plus[j])  # L = sum(output), so dL/dy = 1

        # f(x - eps)
        input_data[idx] = Scalar[dtype](orig - eps_fd)
        var out_minus = alloc[Scalar[dtype]](BATCH * DIM)
        memset(out_minus, 0, BATCH * DIM)
        var cache_minus = alloc[Scalar[dtype]](BATCH * CS)
        memset(cache_minus, 0, BATCH * CS)
        var inp_m = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](input_data)
        var out_m = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](out_minus)
        var c_m = LayoutTensor[dtype, Layout.row_major(BATCH, CS), MutAnyOrigin](cache_minus)
        params[2*C] = 0; params[2*C+1] = 0; params[2*C+2] = 0; params[2*C+3] = 0
        params[3*C] = 1; params[3*C+1] = 1; params[3*C+2] = 1; params[3*C+3] = 1
        BN.forward[BATCH](inp_m, out_m, p, c_m)
        var loss_minus: Float64 = 0.0
        for j in range(BATCH * DIM):
            loss_minus += Float64(out_minus[j])

        input_data[idx] = Scalar[dtype](orig)

        var fd_grad = (loss_plus - loss_minus) / (2.0 * eps_fd)
        var analytical_grad = Float64(grad_in[idx])
        var diff = fd_grad - analytical_grad
        if diff < 0:
            diff = -diff
        if diff > max_diff_input:
            max_diff_input = diff

        out_plus.free()
        out_minus.free()
        cache_plus.free()
        cache_minus.free()

    print("Max |fd_grad - analytical_grad| for input:", max_diff_input)
    if max_diff_input < 0.01:
        print("PASS: Input gradient check")
    else:
        print("FAIL: Input gradient check (threshold 0.01)")

    # Finite-difference check for gamma and beta params
    var max_diff_params: Float64 = 0.0
    for pidx in range(2 * C):  # Only gamma and beta (not running stats)
        var orig = Float64(params[pidx])

        params[pidx] = Scalar[dtype](orig + eps_fd)
        # Reset running stats
        for rc in range(C):
            params[2*C + rc] = 0; params[3*C + rc] = 1
        var out_pp = alloc[Scalar[dtype]](BATCH * DIM)
        var cache_pp = alloc[Scalar[dtype]](BATCH * CS)
        memset(out_pp, 0, BATCH * DIM)
        memset(cache_pp, 0, BATCH * CS)
        var out_pp_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](out_pp)
        var c_pp = LayoutTensor[dtype, Layout.row_major(BATCH, CS), MutAnyOrigin](cache_pp)
        BN.forward[BATCH](inp, out_pp_t, p, c_pp)
        var lp: Float64 = 0.0
        for j in range(BATCH * DIM):
            lp += Float64(out_pp[j])

        params[pidx] = Scalar[dtype](orig - eps_fd)
        for rc in range(C):
            params[2*C + rc] = 0; params[3*C + rc] = 1
        var out_pm = alloc[Scalar[dtype]](BATCH * DIM)
        var cache_pm = alloc[Scalar[dtype]](BATCH * CS)
        memset(out_pm, 0, BATCH * DIM)
        memset(cache_pm, 0, BATCH * CS)
        var out_pm_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](out_pm)
        var c_pm = LayoutTensor[dtype, Layout.row_major(BATCH, CS), MutAnyOrigin](cache_pm)
        BN.forward[BATCH](inp, out_pm_t, p, c_pm)
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

    print("Max |fd_grad - analytical_grad| for params:", max_diff_params)
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


def test_cnn_bn_learns() raises:
    """Test: can a small Conv+BN+ReLU network learn on CPU?"""
    print()
    print("=" * 60)
    print("TEST: Small Conv+BN+ReLU network learning (C4 board)")
    print("=" * 60)

    # Small network: Conv+BN+ReLU → Flatten → Linear → Parallel
    comptime SmallNet = Sequential[
        Conv2DLayer[3, 16, 3, 1, 1, 6, 7],
        BatchNorm2D[16, 6, 7],
        ReLU[16 * 6 * 7],
        FlattenLayer[16 * 6 * 7],
        Linear[16 * 6 * 7, 8],  # 7 policy + 1 value
    ]
    comptime Opt = Adam[LR=0.001]
    comptime BATCH = 8
    comptime OBS = 126
    comptime OUT = 8  # 7 + 1

    print("PARAM_SIZE:", SmallNet.PARAM_SIZE)

    var state = NetworkState[SmallNet, Opt]()
    state.initialize[Kaiming[]]()

    # Create fake data
    var obs = alloc[Scalar[dtype]](BATCH * OBS)
    memset(obs, 0, BATCH * OBS)
    # Fill plane 2 (empty)
    for b in range(BATCH):
        for i in range(42):
            obs[b * OBS + 84 + i] = Scalar[dtype](1.0)
    # Add some pieces for variety
    obs[1 * OBS + 3] = Scalar[dtype](1.0)  # piece at col 3
    obs[1 * OBS + 84 + 3] = Scalar[dtype](0.0)
    obs[2 * OBS + 0] = Scalar[dtype](1.0)
    obs[2 * OBS + 84 + 0] = Scalar[dtype](0.0)

    # Target: action 3 for all samples, value +1
    var target_pol = alloc[Scalar[dtype]](BATCH * 7)
    var target_val = alloc[Scalar[dtype]](BATCH)
    memset(target_pol, 0, BATCH * 7)
    for b in range(BATCH):
        target_pol[b * 7 + 3] = Scalar[dtype](1.0)
        target_val[b] = Scalar[dtype](1.0)

    var pred = alloc[Scalar[dtype]](BATCH * OUT)
    var cache = alloc[Scalar[dtype]](BATCH * SmallNet.CACHE_SIZE)
    var grad_out = alloc[Scalar[dtype]](BATCH * OUT)
    var grad_in = alloc[Scalar[dtype]](BATCH * OBS)

    var obs_t = LayoutTensor[dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin](obs)
    var pred_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin](pred)
    var cache_t = LayoutTensor[dtype, Layout.row_major(BATCH, SmallNet.CACHE_SIZE), MutAnyOrigin](cache)

    # Training loop
    var init_loss: Float64 = 0.0
    for step in range(100):
        memset(pred, 0, BATCH * OUT)
        memset(cache, 0, BATCH * SmallNet.CACHE_SIZE)

        SmallNet.forward[BATCH](obs_t, pred_t, state.params_view(), cache_t)

        # Compute CE loss + gradient for policy
        var batch_loss: Float64 = 0.0
        var inv_batch = Scalar[dtype](1.0 / Float64(BATCH))
        for b in range(BATCH):
            var max_l: Float64 = -1e18
            for a in range(7):
                var l = Float64(pred[b * OUT + a])
                if l > max_l:
                    max_l = l
            var sum_e: Float64 = 0.0
            for a in range(7):
                sum_e += exp(Float64(pred[b * OUT + a]) - max_l)
            for a in range(7):
                var prob = exp(Float64(pred[b * OUT + a]) - max_l) / sum_e
                var target = Float64(target_pol[b * 7 + a])
                if target > 0.01 and prob > 1e-8:
                    batch_loss -= target * log(prob)
                grad_out[b * OUT + a] = Scalar[dtype]((prob - Float64(target_pol[b * 7 + a])) * Float64(inv_batch))
            # Value MSE gradient
            var raw_v = Float64(pred[b * OUT + 7])
            var ev_p = exp(raw_v)
            var ev_n = exp(-raw_v)
            var tanh_v = (ev_p - ev_n) / (ev_p + ev_n)
            var dtanh = 1.0 - tanh_v * tanh_v
            grad_out[b * OUT + 7] = Scalar[dtype](2.0 * (tanh_v - Float64(target_val[b])) * dtanh * Float64(inv_batch))
        batch_loss /= Float64(BATCH)

        if step == 0:
            init_loss = batch_loss

        # Backward
        state.zero_grads()
        var go_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin](grad_out)
        memset(grad_in, 0, BATCH * OBS)
        var gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin](grad_in)
        var grads_v = state.grads_view()
        SmallNet.backward[BATCH](go_t, gi_t, state.params_view(), cache_t, grads_v)

        state.optimizer_step()

        if step % 25 == 0 or step == 99:
            print("  Step", step, "| loss:", Float64(Int(batch_loss * 1000)) / 1000.0)

    # Final forward
    memset(pred, 0, BATCH * OUT)
    SmallNet.forward[BATCH](obs_t, pred_t, state.params_view())
    var final_loss: Float64 = 0.0
    for b in range(BATCH):
        var max_l: Float64 = -1e18
        for a in range(7):
            var l = Float64(pred[b * OUT + a])
            if l > max_l:
                max_l = l
        var sum_e: Float64 = 0.0
        for a in range(7):
            sum_e += exp(Float64(pred[b * OUT + a]) - max_l)
        for a in range(7):
            var prob = exp(Float64(pred[b * OUT + a]) - max_l) / sum_e
            var target = Float64(target_pol[b * 7 + a])
            if target > 0.01 and prob > 1e-8:
                final_loss -= target * log(prob)
    final_loss /= Float64(BATCH)

    print("Initial loss:", Float64(Int(init_loss * 1000)) / 1000.0,
          "| Final loss:", Float64(Int(final_loss * 1000)) / 1000.0)
    if final_loss < init_loss * 0.5:
        print("PASS: Conv+BN+ReLU network learned")
    else:
        print("FAIL: Conv+BN+ReLU network did not learn")

    obs.free()
    target_pol.free()
    target_val.free()
    pred.free()
    cache.free()
    grad_out.free()
    grad_in.free()


def main() raises:
    test_bn_gradient_check()
    test_cnn_bn_learns()

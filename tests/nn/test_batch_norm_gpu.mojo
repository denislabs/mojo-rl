"""GPU vs CPU consistency test for BatchNorm2D.

Runs the same forward and backward on CPU and GPU, compares outputs.
This catches GPU-specific bugs (wrong kernel launch, stride issues, etc).
"""

from std.math import sqrt
from std.memory import alloc, memset, UnsafePointer
from std.gpu.host import DeviceContext
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
    FlattenLayer,
    Parallel,
)
from mojo_rl.nn.optimizer import Adam


def test_bn_gpu_vs_cpu() raises:
    """Compare BatchNorm2D forward+backward: GPU vs CPU."""
    print("=" * 60)
    print("TEST: BatchNorm2D GPU vs CPU (C=16, H=6, W=7)")
    print("=" * 60)

    var ctx = DeviceContext()

    comptime C = 16
    comptime H = 6
    comptime W = 7
    comptime BN = BatchNorm2D[C, H, W]
    comptime BATCH = 8
    comptime DIM = C * H * W  # 672
    comptime PS = BN.PARAM_SIZE
    comptime CS = BN.CACHE_SIZE

    # Create input data
    var input_data = alloc[Scalar[dtype]](BATCH * DIM)
    for i in range(BATCH * DIM):
        input_data[i] = Scalar[dtype](Float64(i % 13) * 0.2 - 1.2)

    # Init params
    var params = alloc[Scalar[dtype]](PS)
    for c in range(C):
        params[c] = Scalar[dtype](1.0 + Float64(c) * 0.1)  # gamma varies
        params[C + c] = Scalar[dtype](Float64(c) * 0.05)    # beta varies
        params[2*C + c] = Scalar[dtype](0.0)
        params[3*C + c] = Scalar[dtype](1.0)

    # ── CPU Forward ──
    var cpu_output = alloc[Scalar[dtype]](BATCH * DIM)
    var cpu_cache = alloc[Scalar[dtype]](BATCH * CS)
    memset(cpu_output, 0, BATCH * DIM)
    memset(cpu_cache, 0, BATCH * CS)

    # Save params before (running stats get modified)
    var params_save = alloc[Scalar[dtype]](PS)
    for i in range(PS):
        params_save[i] = params[i]

    var inp_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](input_data)
    var cpu_out_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](cpu_output)
    var cpu_params_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](params)
    var cpu_cache_t = LayoutTensor[dtype, Layout.row_major(BATCH, CS), MutAnyOrigin](cpu_cache)
    var cpu_state_t = LayoutTensor[dtype, Layout.row_major(BN.STATE_SIZE), MutAnyOrigin](
        UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=0)
    )

    BN.forward[BATCH](inp_t, cpu_out_t, cpu_params_t, cpu_state_t, cpu_cache_t)

    # Save CPU running stats
    var cpu_rmean = alloc[Scalar[dtype]](C)
    var cpu_rvar = alloc[Scalar[dtype]](C)
    for c in range(C):
        cpu_rmean[c] = params[2*C + c]
        cpu_rvar[c] = params[3*C + c]

    # ── GPU Forward ──
    # Restore params (running stats were modified by CPU forward)
    for i in range(PS):
        params[i] = params_save[i]

    var gpu_input = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
    var gpu_output = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
    var gpu_params = ctx.enqueue_create_buffer[dtype](PS)
    var gpu_cache = ctx.enqueue_create_buffer[dtype](BATCH * CS)
    var gpu_ws = ctx.enqueue_create_buffer[dtype](1)

    # Upload
    var input_host = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
    for i in range(BATCH * DIM):
        input_host[i] = input_data[i]
    ctx.enqueue_copy(gpu_input, input_host)

    var params_host = ctx.enqueue_create_host_buffer[dtype](PS)
    for i in range(PS):
        params_host[i] = params[i]
    ctx.enqueue_copy(gpu_params, params_host)

    gpu_output.enqueue_fill(Scalar[dtype](0.0))
    gpu_cache.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    var gpu_inp_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](gpu_input.unsafe_ptr())
    var gpu_out_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](gpu_output.unsafe_ptr())
    var gpu_params_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](gpu_params.unsafe_ptr())
    var gpu_cache_t = LayoutTensor[dtype, Layout.row_major(BATCH, CS), MutAnyOrigin](gpu_cache.unsafe_ptr())
    var gpu_state_t = LayoutTensor[dtype, Layout.row_major(BN.STATE_SIZE), MutAnyOrigin](
        UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=0)
    )

    BN.forward_gpu[BATCH](ctx, gpu_out_t, gpu_inp_t, gpu_params_t, gpu_state_t, gpu_cache_t, gpu_ws)
    ctx.synchronize()

    # Download GPU output
    var gpu_out_host = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
    ctx.enqueue_copy(gpu_out_host, gpu_output)
    ctx.synchronize()

    # Compare forward outputs
    var max_fwd_diff: Float64 = 0.0
    for i in range(BATCH * DIM):
        var diff = Float64(gpu_out_host[i]) - Float64(cpu_output[i])
        if diff < 0:
            diff = -diff
        if diff > max_fwd_diff:
            max_fwd_diff = diff

    print("Forward max |GPU - CPU|:", max_fwd_diff)
    if max_fwd_diff < 0.01:
        print("PASS: Forward consistency")
    else:
        print("FAIL: Forward consistency")
        # Print first diverging elements
        for i in range(min(BATCH * DIM, 20)):
            if Float64(gpu_out_host[i]) - Float64(cpu_output[i]) > 0.001 or Float64(cpu_output[i]) - Float64(gpu_out_host[i]) > 0.001:
                print("  [", i, "] GPU:", Float64(gpu_out_host[i]), "CPU:", Float64(cpu_output[i]))

    # Compare running stats
    var gpu_params_dl = ctx.enqueue_create_host_buffer[dtype](PS)
    ctx.enqueue_copy(gpu_params_dl, gpu_params)
    ctx.synchronize()

    var max_rmean_diff: Float64 = 0.0
    var max_rvar_diff: Float64 = 0.0
    for c in range(C):
        var d1 = Float64(gpu_params_dl[2*C + c]) - Float64(cpu_rmean[c])
        if d1 < 0:
            d1 = -d1
        if d1 > max_rmean_diff:
            max_rmean_diff = d1
        var d2 = Float64(gpu_params_dl[3*C + c]) - Float64(cpu_rvar[c])
        if d2 < 0:
            d2 = -d2
        if d2 > max_rvar_diff:
            max_rvar_diff = d2

    print("Running mean max |GPU - CPU|:", max_rmean_diff)
    print("Running var max |GPU - CPU|:", max_rvar_diff)

    # ── Backward ──
    # Create gradient output (all ones)
    var grad_out_data = alloc[Scalar[dtype]](BATCH * DIM)
    for i in range(BATCH * DIM):
        grad_out_data[i] = Scalar[dtype](1.0)

    # CPU backward
    var cpu_grad_in = alloc[Scalar[dtype]](BATCH * DIM)
    var cpu_grad_params = alloc[Scalar[dtype]](PS)
    memset(cpu_grad_in, 0, BATCH * DIM)
    memset(cpu_grad_params, 0, PS)
    var go_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](grad_out_data)
    var cpu_gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](cpu_grad_in)
    var cpu_gp_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](cpu_grad_params)
    BN.backward[BATCH](go_t, cpu_gi_t, cpu_params_t, cpu_state_t, cpu_cache_t, cpu_gp_t)

    # GPU backward
    var gpu_grad_out = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
    var gpu_grad_in = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
    var gpu_grads = ctx.enqueue_create_buffer[dtype](PS)

    var go_host = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
    for i in range(BATCH * DIM):
        go_host[i] = Scalar[dtype](1.0)
    ctx.enqueue_copy(gpu_grad_out, go_host)
    gpu_grad_in.enqueue_fill(Scalar[dtype](0.0))
    gpu_grads.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    var gpu_go_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](gpu_grad_out.unsafe_ptr())
    var gpu_gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](gpu_grad_in.unsafe_ptr())
    var gpu_gp_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](gpu_grads.unsafe_ptr())

    BN.backward_gpu[BATCH](ctx, gpu_gi_t, gpu_go_t, gpu_params_t, gpu_state_t, gpu_cache_t, gpu_gp_t, gpu_ws)
    ctx.synchronize()

    # Download and compare
    var gpu_gi_host = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
    ctx.enqueue_copy(gpu_gi_host, gpu_grad_in)
    var gpu_gp_host = ctx.enqueue_create_host_buffer[dtype](PS)
    ctx.enqueue_copy(gpu_gp_host, gpu_grads)
    ctx.synchronize()

    var max_gi_diff: Float64 = 0.0
    for i in range(BATCH * DIM):
        var diff = Float64(gpu_gi_host[i]) - Float64(cpu_grad_in[i])
        if diff < 0:
            diff = -diff
        if diff > max_gi_diff:
            max_gi_diff = diff

    var max_gp_diff: Float64 = 0.0
    for i in range(PS):
        var diff = Float64(gpu_gp_host[i]) - Float64(cpu_grad_params[i])
        if diff < 0:
            diff = -diff
        if diff > max_gp_diff:
            max_gp_diff = diff

    print("Backward grad_input max |GPU - CPU|:", max_gi_diff)
    print("Backward grad_params max |GPU - CPU|:", max_gp_diff)
    if max_gi_diff < 0.01 and max_gp_diff < 0.01:
        print("PASS: Backward consistency")
    else:
        print("FAIL: Backward consistency")

    # ── No-cache forward (inference) ──
    # Restore params
    for i in range(PS):
        params[i] = params_save[i]
    for i in range(PS):
        params_host[i] = params[i]
    ctx.enqueue_copy(gpu_params, params_host)

    var cpu_nc_output = alloc[Scalar[dtype]](BATCH * DIM)
    memset(cpu_nc_output, 0, BATCH * DIM)
    var cpu_nc_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](cpu_nc_output)
    # Restore params for CPU
    var cpu_nc_params = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](params)
    BN.forward[BATCH](inp_t, cpu_nc_t, cpu_nc_params, cpu_state_t)

    gpu_output.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()
    BN.forward_gpu_no_cache[BATCH](ctx, gpu_out_t, gpu_inp_t, gpu_params_t, gpu_state_t, gpu_ws)
    ctx.synchronize()
    ctx.enqueue_copy(gpu_out_host, gpu_output)
    ctx.synchronize()

    var max_nc_diff: Float64 = 0.0
    for i in range(BATCH * DIM):
        var diff = Float64(gpu_out_host[i]) - Float64(cpu_nc_output[i])
        if diff < 0:
            diff = -diff
        if diff > max_nc_diff:
            max_nc_diff = diff

    print("No-cache forward max |GPU - CPU|:", max_nc_diff)
    if max_nc_diff < 0.01:
        print("PASS: No-cache forward consistency")
    else:
        print("FAIL: No-cache forward consistency")

    # Cleanup
    input_data.free()
    params.free()
    params_save.free()
    cpu_output.free()
    cpu_cache.free()
    cpu_rmean.free()
    cpu_rvar.free()
    grad_out_data.free()
    cpu_grad_in.free()
    cpu_grad_params.free()
    cpu_nc_output.free()


def test_full_cnn_gpu_training() raises:
    """Test: can a Conv+BN+ReLU network learn on GPU?"""
    print()
    print("=" * 60)
    print("TEST: Full CNN+BN GPU training (ConnectFour-sized)")
    print("=" * 60)

    var ctx = DeviceContext()

    # Minimal CNN matching ConnectFour structure (smaller filters for speed)
    comptime Net = Sequential[
        Conv2DLayer[3, 8, 3, 1, 1, 6, 7],
        BatchNorm2D[8, 6, 7],
        ReLU[8 * 6 * 7],
        FlattenLayer[8 * 6 * 7],
        Linear[8 * 6 * 7, 8],  # 7 policy + 1 value
    ]
    comptime Opt = Adam[LR=0.001]
    comptime BATCH = 16
    comptime OBS = 126
    comptime OUT = 8

    print("PARAM_SIZE:", Net.PARAM_SIZE, "CACHE_SIZE:", Net.CACHE_SIZE)

    var state = NetworkState[Net, Opt]()
    state.initialize[Kaiming[]]()

    from mojo_rl.nn.training import Network, GPUNetworkState

    var gpu = GPUNetworkState[Net, Opt](ctx)
    gpu.upload_from(state, ctx)
    ctx.synchronize()

    # Create data on GPU
    var obs_buf = ctx.enqueue_create_buffer[dtype](BATCH * OBS)
    var pred_buf = ctx.enqueue_create_buffer[dtype](BATCH * OUT)
    var cache_buf = ctx.enqueue_create_buffer[dtype](BATCH * Net.CACHE_SIZE)
    var grad_out_buf = ctx.enqueue_create_buffer[dtype](BATCH * OUT)
    var grad_in_buf = ctx.enqueue_create_buffer[dtype](BATCH * OBS)
    comptime WS_SIZE = BATCH * Net.WORKSPACE_SIZE_PER_SAMPLE
    var ws_buf = ctx.enqueue_create_buffer[dtype](WS_SIZE if WS_SIZE > 0 else 1)

    # Fill obs: empty board + some pieces
    var obs_host = ctx.enqueue_create_host_buffer[dtype](BATCH * OBS)
    for i in range(BATCH * OBS):
        obs_host[i] = Scalar[dtype](0.0)
    for b in range(BATCH):
        for i in range(42):
            obs_host[b * OBS + 84 + i] = Scalar[dtype](1.0)
        # Add variation
        if b % 3 == 1:
            obs_host[b * OBS + 3] = Scalar[dtype](1.0)
            obs_host[b * OBS + 84 + 3] = Scalar[dtype](0.0)
    ctx.enqueue_copy(obs_buf, obs_host)

    # Fill gradient: target action 3 for policy, +1 for value
    var grad_host = ctx.enqueue_create_host_buffer[dtype](BATCH * OUT)
    for b in range(BATCH):
        for a in range(7):
            # Simplified CE-like gradient
            if a == 3:
                grad_host[b * OUT + a] = Scalar[dtype](-0.8 / Float64(BATCH))
            else:
                grad_host[b * OUT + a] = Scalar[dtype](0.133 / Float64(BATCH))
        grad_host[b * OUT + 7] = Scalar[dtype](-0.1 / Float64(BATCH))
    ctx.enqueue_copy(grad_out_buf, grad_host)
    ctx.synchronize()

    # Training loop on GPU
    var obs_t = LayoutTensor[dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin](obs_buf.unsafe_ptr())
    var pred_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin](pred_buf.unsafe_ptr())
    var cache_t = LayoutTensor[dtype, Layout.row_major(BATCH, Net.CACHE_SIZE), MutAnyOrigin](cache_buf.unsafe_ptr())
    var go_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin](grad_out_buf.unsafe_ptr())
    var gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin](grad_in_buf.unsafe_ptr())

    var pred_dl = ctx.enqueue_create_host_buffer[dtype](BATCH * OUT)

    for step in range(50):
        # Forward with cache
        ctx.enqueue_memset(pred_buf, 0)
        ctx.enqueue_memset(cache_buf, 0)
        Net.forward_gpu[BATCH](ctx, pred_t, obs_t, gpu.params_view(), gpu.model_state_view(), cache_t, ws_buf)

        # Backward
        gpu.zero_grads(ctx)
        ctx.enqueue_memset(grad_in_buf, 0)
        var grads_v = gpu.grads_view()
        Net.backward_gpu[BATCH](ctx, gi_t, go_t, gpu.params_view(), gpu.model_state_view(), cache_t, grads_v, ws_buf)

        # Optimizer
        gpu.optimizer_step(ctx)

        if step % 10 == 0 or step == 49:
            ctx.enqueue_copy(pred_dl, pred_buf)
            ctx.synchronize()
            # Check if action 3 logit is increasing
            var avg_a3: Float64 = 0.0
            var avg_other: Float64 = 0.0
            for b in range(BATCH):
                avg_a3 += Float64(pred_dl[b * OUT + 3])
                for a in range(7):
                    if a != 3:
                        avg_other += Float64(pred_dl[b * OUT + a])
            avg_a3 /= Float64(BATCH)
            avg_other /= Float64(BATCH * 6)
            print("  Step", step, "| action3:", Float64(Int(avg_a3 * 100)) / 100.0,
                  "| others:", Float64(Int(avg_other * 100)) / 100.0,
                  "| diff:", Float64(Int((avg_a3 - avg_other) * 100)) / 100.0)

    # Final check
    ctx.enqueue_copy(pred_dl, pred_buf)
    ctx.synchronize()
    var final_a3: Float64 = 0.0
    var final_other: Float64 = 0.0
    for b in range(BATCH):
        final_a3 += Float64(pred_dl[b * OUT + 3])
        for a in range(7):
            if a != 3:
                final_other += Float64(pred_dl[b * OUT + a])
    final_a3 /= Float64(BATCH)
    final_other /= Float64(BATCH * 6)

    if final_a3 > final_other:
        print("PASS: GPU CNN+BN learned (action3 > others)")
    else:
        print("FAIL: GPU CNN+BN did not learn (action3 <= others)")


def main() raises:
    test_bn_gpu_vs_cpu()
    test_full_cnn_gpu_training()

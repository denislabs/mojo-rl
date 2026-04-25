"""GPU numerical gradcheck for ResBlock layers.

Tests both ResBlockConv2D and ResBlockConv2DBN on GPU using host-side
perturbation (full params upload per finite-diff step). Checks both
param gradients and input gradients.

For BN-containing models, BN running stats params (rmean, rvar) are
skipped in the gradcheck since they don't receive gradients, and BN
batch stats shift during perturbation introduces inherent noise.

Usage:
    pixi run -e apple mojo run -I . tests/nn/test_resblock_gpu_gradcheck.mojo
    pixi run -e nvidia mojo run -I . tests/nn/test_resblock_gpu_gradcheck.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import abs, sqrt
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.training import NetworkState, GPUNetworkState
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.model import (
    Model,
    ResBlockConv2D,
)
from mojo_rl.nn.model.resblock_conv2d_bn import ResBlockConv2DBN


def gpu_gradcheck[M: Model, BS: Int = 4](
    ctx: DeviceContext,
    name: String,
    eps: Float64 = 1e-3,
    max_params: Int = 300,
    max_inputs: Int = 200,
    param_tol: Float64 = 0.01,
    input_tol: Float64 = 0.01,
    skip_param_range_start: Int = -1,
    skip_param_range_end: Int = -1,
) raises:
    """GPU numerical gradcheck with host-side perturbation.

    Checks both param gradients and input gradients. Optionally skips
    a range of param indices (e.g., BN running stats that shouldn't
    receive gradients).
    """
    comptime IN = M.IN_DIM
    comptime OUT = M.OUT_DIM
    comptime PS = M.PARAM_SIZE
    comptime CS = M.CACHE_SIZE
    comptime WS = M.WORKSPACE_SIZE_PER_SAMPLE

    print("GPU Gradcheck:", name, "(IN=", IN, "OUT=", OUT, "PS=", PS, ")")

    var cpu_state = NetworkState[M, Adam[]]()
    cpu_state.initialize[Xavier[]]()
    var gpu = GPUNetworkState[M, Adam[], dtype](ctx)
    gpu.upload_from(cpu_state, ctx)

    var workspace = ctx.enqueue_create_buffer[dtype](BS * WS if WS > 0 else 1)

    # Input
    var input_host = ctx.enqueue_create_host_buffer[dtype](BS * IN)
    for i in range(BS * IN):
        input_host[i] = Scalar[dtype](0.1 + Float64(i % 13) / 13.0 * 0.8)
    var input_buf = ctx.enqueue_create_buffer[dtype](BS * IN)
    ctx.enqueue_copy(input_buf, input_host)

    # Grad output
    var grad_out_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
    for i in range(BS * OUT):
        grad_out_host[i] = Scalar[dtype](
            0.5 + Float64(i % 7) / 14.0 - Float64(i % 3) / 6.0
        )
    var grad_out_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    ctx.enqueue_copy(grad_out_buf, grad_out_host)

    # --- Analytical backward on GPU ---
    var cache_buf = ctx.enqueue_create_buffer[dtype](BS * CS if CS > 0 else 1)
    var output_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)

    var input_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](
        input_buf.unsafe_ptr()
    )
    var output_t = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](
        output_buf.unsafe_ptr()
    )
    var cache_t = LayoutTensor[dtype, Layout.row_major(BS, CS), MutAnyOrigin](
        cache_buf.unsafe_ptr()
    )

    M.forward_gpu[BS](ctx, output_t, input_t, gpu.params_view(), gpu.model_state_view(), cache_t, workspace)

    var grad_in_buf = ctx.enqueue_create_buffer[dtype](BS * IN)
    ctx.enqueue_memset(grad_in_buf, 0)
    var grad_out_t = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](
        grad_out_buf.unsafe_ptr()
    )
    var grad_in_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](
        grad_in_buf.unsafe_ptr()
    )

    gpu.zero_grads(ctx)
    var grads = gpu.grads_view()
    M.backward_gpu[BS](
        ctx, grad_in_t, grad_out_t, gpu.params_view(), gpu.model_state_view(), cache_t, grads, workspace,
    )

    # Read analytical grads, params, grad_input to host
    var ana_grads_host = ctx.enqueue_create_host_buffer[dtype](PS)
    var params_host = ctx.enqueue_create_host_buffer[dtype](PS)
    var ana_grad_in_host = ctx.enqueue_create_host_buffer[dtype](BS * IN)
    ctx.enqueue_copy(ana_grads_host, gpu.grads_buf)
    ctx.enqueue_copy(params_host, gpu.params_buf)
    ctx.enqueue_copy(ana_grad_in_host, grad_in_buf)

    var grad_out_host2 = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
    ctx.enqueue_copy(grad_out_host2, grad_out_buf)
    ctx.synchronize()

    # --- Finite differences via host-side perturbation ---
    var params_copy = ctx.enqueue_create_host_buffer[dtype](PS)
    var input_copy = ctx.enqueue_create_host_buffer[dtype](BS * IN)
    var out_plus_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    var out_minus_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    var out_plus_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
    var out_minus_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
    var cache_tmp = ctx.enqueue_create_buffer[dtype](BS * CS if CS > 0 else 1)
    ctx.synchronize()

    # === Check param gradients ===
    var p_step = PS // max_params
    if p_step < 1:
        p_step = 1

    var p_max_rel: Float64 = 0.0
    var p_checked = 0
    var p_fail = 0

    for p_idx in range(0, PS, p_step):
        # Skip BN running stats if requested
        if p_idx >= skip_param_range_start and p_idx < skip_param_range_end:
            continue

        for i in range(PS):
            params_copy[i] = params_host[i]
        var orig = Float64(params_host[p_idx])

        # f(p + eps)
        params_copy[p_idx] = Scalar[dtype](orig + eps)
        ctx.enqueue_copy(gpu.params_buf, params_copy)
        var out_plus_t = LayoutTensor[
            dtype, Layout.row_major(BS, OUT), MutAnyOrigin
        ](out_plus_buf.unsafe_ptr())
        var cache_tmp_t = LayoutTensor[
            dtype, Layout.row_major(BS, CS), MutAnyOrigin
        ](cache_tmp.unsafe_ptr())
        M.forward_gpu[BS](
            ctx, out_plus_t, input_t, gpu.params_view(), gpu.model_state_view(), cache_tmp_t, workspace
        )
        ctx.enqueue_copy(out_plus_host, out_plus_buf)

        # f(p - eps)
        params_copy[p_idx] = Scalar[dtype](orig - eps)
        ctx.enqueue_copy(gpu.params_buf, params_copy)
        var out_minus_t = LayoutTensor[
            dtype, Layout.row_major(BS, OUT), MutAnyOrigin
        ](out_minus_buf.unsafe_ptr())
        var cache_m_t = LayoutTensor[
            dtype, Layout.row_major(BS, CS), MutAnyOrigin
        ](cache_tmp.unsafe_ptr())
        M.forward_gpu[BS](
            ctx, out_minus_t, input_t, gpu.params_view(), gpu.model_state_view(), cache_m_t, workspace
        )
        ctx.enqueue_copy(out_minus_host, out_minus_buf)

        # Restore
        params_copy[p_idx] = Scalar[dtype](orig)
        ctx.enqueue_copy(gpu.params_buf, params_copy)
        ctx.synchronize()

        var num_grad: Float64 = 0.0
        for j in range(BS * OUT):
            var go = Float64(grad_out_host2[j])
            var fp = Float64(out_plus_host[j])
            var fm = Float64(out_minus_host[j])
            num_grad += go * (fp - fm) / (2.0 * eps)

        var ana_grad = Float64(ana_grads_host[p_idx])
        var err = abs(ana_grad - num_grad)
        var denom = abs(ana_grad) + abs(num_grad)
        var rel: Float64 = 0.0
        if denom > 1e-5:
            rel = err / denom

        if rel > p_max_rel:
            p_max_rel = rel
        if rel > param_tol and denom > 1e-4:
            p_fail += 1
            if p_fail <= 5:
                print(
                    "    PARAM p[", p_idx, "]: ana=",
                    ana_grad, "num=", num_grad, "rel=", rel,
                )
        p_checked += 1

    if p_fail == 0:
        print("  [PASS] params: max_rel=", p_max_rel, "(", p_checked, "checked)")
    else:
        print("  [FAIL] params:", p_fail, "/", p_checked, "max_rel=", p_max_rel)

    # === Check input gradients ===
    # Perturb input elements instead of params
    var i_step = (BS * IN) // max_inputs
    if i_step < 1:
        i_step = 1

    var i_max_rel: Float64 = 0.0
    var i_checked = 0
    var i_fail = 0

    # Restore original params for input gradcheck
    ctx.enqueue_copy(gpu.params_buf, params_host)
    ctx.synchronize()

    for i_idx in range(0, BS * IN, i_step):
        for i in range(BS * IN):
            input_copy[i] = input_host[i]
        var orig = Float64(input_host[i_idx])

        # f(x + eps)
        input_copy[i_idx] = Scalar[dtype](orig + eps)
        ctx.enqueue_copy(input_buf, input_copy)
        var out_plus_t = LayoutTensor[
            dtype, Layout.row_major(BS, OUT), MutAnyOrigin
        ](out_plus_buf.unsafe_ptr())
        var cache_tmp_t = LayoutTensor[
            dtype, Layout.row_major(BS, CS), MutAnyOrigin
        ](cache_tmp.unsafe_ptr())
        M.forward_gpu[BS](
            ctx, out_plus_t, input_t, gpu.params_view(), gpu.model_state_view(), cache_tmp_t, workspace
        )
        ctx.enqueue_copy(out_plus_host, out_plus_buf)

        # f(x - eps)
        input_copy[i_idx] = Scalar[dtype](orig - eps)
        ctx.enqueue_copy(input_buf, input_copy)
        var out_minus_t = LayoutTensor[
            dtype, Layout.row_major(BS, OUT), MutAnyOrigin
        ](out_minus_buf.unsafe_ptr())
        var cache_m_t = LayoutTensor[
            dtype, Layout.row_major(BS, CS), MutAnyOrigin
        ](cache_tmp.unsafe_ptr())
        M.forward_gpu[BS](
            ctx, out_minus_t, input_t, gpu.params_view(), gpu.model_state_view(), cache_m_t, workspace
        )
        ctx.enqueue_copy(out_minus_host, out_minus_buf)

        # Restore input
        input_copy[i_idx] = Scalar[dtype](orig)
        ctx.enqueue_copy(input_buf, input_copy)
        ctx.synchronize()

        var num_grad: Float64 = 0.0
        for j in range(BS * OUT):
            var go = Float64(grad_out_host2[j])
            var fp = Float64(out_plus_host[j])
            var fm = Float64(out_minus_host[j])
            num_grad += go * (fp - fm) / (2.0 * eps)

        var ana_grad = Float64(ana_grad_in_host[i_idx])
        var err = abs(ana_grad - num_grad)
        var denom = abs(ana_grad) + abs(num_grad)
        var rel: Float64 = 0.0
        if denom > 1e-5:
            rel = err / denom

        if rel > i_max_rel:
            i_max_rel = rel
        if rel > input_tol and denom > 1e-4:
            i_fail += 1
            if i_fail <= 5:
                print(
                    "    INPUT i[", i_idx, "]: ana=",
                    ana_grad, "num=", num_grad, "rel=", rel,
                )
        i_checked += 1

    # Restore original input for next test
    ctx.enqueue_copy(input_buf, input_host)
    ctx.synchronize()

    if i_fail == 0:
        print("  [PASS] inputs: max_rel=", i_max_rel, "(", i_checked, "checked)")
    else:
        print("  [FAIL] inputs:", i_fail, "/", i_checked, "max_rel=", i_max_rel)

    print()


def main() raises:
    print("=== ResBlock GPU Gradcheck ===")
    print()

    var ctx = DeviceContext()

    # ── ResBlockConv2D (no BN -- should be clean) ────────────
    gpu_gradcheck[ResBlockConv2D[4, 3, 1, 5, 5]](
        ctx, "ResBlockConv2D[4ch,3x3,5x5]",
    )

    # Larger: 8 channels
    gpu_gradcheck[ResBlockConv2D[8, 3, 1, 5, 5]](
        ctx, "ResBlockConv2D[8ch,3x3,5x5]",
    )

    # ── ResBlockConv2DBN (with BN -- relaxed tolerance) ──────
    # Skip BN2 running stats (rmean + rvar = 2*channels params at end of param buffer)
    # BN2 params layout: [gamma(C) | beta(C) | rmean(C) | rvar(C)]
    # BN2 starts at CONV1_PS + CONV2_PS
    # rmean starts at BN2_OFF + 2*C, rvar at BN2_OFF + 3*C
    comptime RB_BN = ResBlockConv2DBN[4, 3, 1, 5, 5]
    comptime BN2_RMEAN_START = RB_BN.CONV1_PS + RB_BN.CONV2_PS + 2 * 4  # skip rmean+rvar
    comptime BN2_END = RB_BN.PARAM_SIZE

    gpu_gradcheck[ResBlockConv2DBN[4, 3, 1, 5, 5]](
        ctx,
        "ResBlockConv2DBN[4ch,3x3,5x5]",
        param_tol=0.05,  # BN batch stats add noise
        input_tol=0.05,
        skip_param_range_start=BN2_RMEAN_START,
        skip_param_range_end=BN2_END,
    )

    # Larger batch to reduce BN noise
    gpu_gradcheck[ResBlockConv2DBN[4, 3, 1, 5, 5], 16](
        ctx,
        "ResBlockConv2DBN[4ch,3x3,5x5] BS=16",
        param_tol=0.03,
        input_tol=0.03,
        skip_param_range_start=BN2_RMEAN_START,
        skip_param_range_end=BN2_END,
    )

    print("=== Done ===")

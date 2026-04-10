"""Sanity test for the AlphaZero Connect Four architecture.

Verifies that forward pass produces valid output dimensions and
backward pass produces non-zero gradients for all parameter groups.

Usage:
    pixi run mojo run -I . tests/nn/test_alphazero_architecture.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import abs, sqrt
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.training import NetworkState, GPUNetworkState
from mojo_rl.nn.optimizer import Adam, AdamW
from mojo_rl.nn.initializer import Xavier, Kaiming
from mojo_rl.deep_agents.alphazero import (
    AlphaZeroConnectFourFusedResNetConfig,
    AlphaZeroTicTacToeConfig,
)


def test_forward_backward[Config: AlphaZeroConfig](
    ctx: DeviceContext, name: String
) raises:
    """Test forward + backward on GPU for an AlphaZero config."""
    comptime M = Config.PredModel
    comptime BS = 4  # Small batch for testing
    comptime IN = M.IN_DIM
    comptime OUT = M.OUT_DIM
    comptime PS = M.PARAM_SIZE
    comptime CS = M.CACHE_SIZE

    print("Testing:", name)
    print("  IN_DIM:", IN, "OUT_DIM:", OUT, "PARAM_SIZE:", PS, "CACHE_SIZE:", CS)

    # Initialize on CPU, then upload to GPU
    var cpu_state = NetworkState[M, Config.OptType]()
    cpu_state.initialize[Kaiming[]]()
    var gpu = GPUNetworkState[M, Config.OptType, dtype](ctx)
    gpu.upload_from(cpu_state, ctx)

    # Workspace
    comptime WS = M.WORKSPACE_SIZE_PER_SAMPLE
    var workspace = ctx.enqueue_create_buffer[dtype](BS * WS if WS > 0 else 1)

    # Input: random-ish values in [0, 1]
    var input_host = ctx.enqueue_create_host_buffer[dtype](BS * IN)
    for i in range(BS * IN):
        input_host[i] = Scalar[dtype](Float64(i % 7) / 7.0)
    var input_buf = ctx.enqueue_create_buffer[dtype](BS * IN)
    ctx.enqueue_copy(input_buf, input_host)

    # Output + cache buffers
    var output_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    var cache_buf = ctx.enqueue_create_buffer[dtype](BS * CS)

    # Forward
    var input_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](
        input_buf.unsafe_ptr()
    )
    var output_t = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](
        output_buf.unsafe_ptr()
    )
    var cache_t = LayoutTensor[dtype, Layout.row_major(BS, CS), MutAnyOrigin](
        cache_buf.unsafe_ptr()
    )

    M.forward_gpu[BS](
        ctx, output_t, input_t, gpu.params_view(), cache_t, workspace
    )

    # Read output to host
    var output_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
    ctx.enqueue_copy(output_host, output_buf)
    ctx.synchronize()

    # Check output shape: first 7 = policy logits, last 1 = value
    comptime ACT = Config.action_dim
    var all_finite = True
    var policy_sum: Float64 = 0.0
    var value_sum: Float64 = 0.0
    for b in range(BS):
        for a in range(ACT):
            var v = Float64(output_host[b * OUT + a])
            if v != v:  # NaN check
                all_finite = False
            policy_sum += v * v
        var val = Float64(output_host[b * OUT + ACT])
        if val != val:
            all_finite = False
        value_sum += val * val

    if all_finite:
        print("  [PASS] Forward: all outputs finite")
    else:
        print("  [FAIL] Forward: NaN in output!")
    print(
        "  Policy L2:",
        sqrt(policy_sum / Float64(BS)),
        "Value L2:",
        sqrt(value_sum / Float64(BS)),
    )

    # Backward with unit gradient
    var grad_out_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    var grad_out_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
    for i in range(BS * OUT):
        grad_out_host[i] = Scalar[dtype](1.0 / Float64(BS))
    ctx.enqueue_copy(grad_out_buf, grad_out_host)

    var grad_in_buf = ctx.enqueue_create_buffer[dtype](BS * IN)
    ctx.enqueue_memset(grad_in_buf, 0)

    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BS, M.OUT_DIM), MutAnyOrigin
    ](grad_out_buf.unsafe_ptr())
    var grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BS, M.IN_DIM), MutAnyOrigin
    ](grad_in_buf.unsafe_ptr())

    gpu.zero_grads(ctx)
    var grads = gpu.grads_view()
    M.backward_gpu[BS](
        ctx,
        grad_in_t,
        grad_out_t,
        gpu.params_view(),
        cache_t,
        grads,
        workspace,
    )

    # Read gradients to host
    var grads_host = ctx.enqueue_create_host_buffer[dtype](PS)
    ctx.enqueue_copy(grads_host, gpu.grads_buf)
    ctx.synchronize()

    # Check gradients: count zero vs nonzero
    var n_zero = 0
    var n_nonzero = 0
    var n_nan = 0
    var grad_norm: Float64 = 0.0
    var grad_max: Float64 = 0.0
    for i in range(PS):
        var g = Float64(grads_host[i])
        if g != g:
            n_nan += 1
        elif abs(g) < 1e-12:
            n_zero += 1
        else:
            n_nonzero += 1
            grad_norm += g * g
            if abs(g) > grad_max:
                grad_max = abs(g)

    print(
        "  Gradients: nonzero=",
        n_nonzero,
        "zero=",
        n_zero,
        "nan=",
        n_nan,
        "/ total=",
        PS,
    )
    print("  Grad norm:", sqrt(grad_norm), "max:", grad_max)

    if n_nan > 0:
        print("  [FAIL] NaN gradients detected!")
    elif n_nonzero == 0:
        print("  [FAIL] All gradients are zero — backward pass broken!")
    elif Float64(n_zero) / Float64(PS) > 0.5:
        print(
            "  [WARN] >50% zero gradients — possible dead layers (",
            Float64(n_zero) / Float64(PS) * 100.0,
            "%)",
        )
    else:
        print(
            "  [PASS] Backward: gradients flow through",
            Float64(n_nonzero) / Float64(PS) * 100.0,
            "% of params",
        )

    # Check grad_input (should be nonzero if trunk is connected)
    var grad_in_host = ctx.enqueue_create_host_buffer[dtype](BS * IN)
    ctx.enqueue_copy(grad_in_host, grad_in_buf)
    ctx.synchronize()
    var gi_nonzero = 0
    for i in range(BS * IN):
        if abs(Float64(grad_in_host[i])) > 1e-12:
            gi_nonzero += 1
    print(
        "  Grad input: nonzero=",
        gi_nonzero,
        "/",
        BS * IN,
    )

    print()


def gpu_finite_diff_check[
    M: Model,
    BS: Int,
](
    ctx: DeviceContext,
    gpu: GPUNetworkState[M, Adam[], dtype],
    input_buf: DeviceBuffer[dtype],
    grad_out_buf: DeviceBuffer[dtype],
    workspace: DeviceBuffer[dtype],
    eps: Float64 = 1e-3,
    max_params_to_check: Int = 200,
) raises -> Tuple[Float64, Float64, Int, Int]:
    """GPU numerical gradient check via finite differences.

    Perturbs each param on GPU, re-runs forward, compares numerical vs
    analytical gradient. Returns (max_abs_err, max_rel_err, num_checked, num_fail).
    """
    comptime IN = M.IN_DIM
    comptime OUT = M.OUT_DIM
    comptime PS = M.PARAM_SIZE
    comptime CS = M.CACHE_SIZE

    var input_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](
        input_buf.unsafe_ptr()
    )

    # 1. Analytical backward: forward → backward → read gradients
    var cache_buf = ctx.enqueue_create_buffer[dtype](BS * CS)
    var output_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BS, OUT), MutAnyOrigin
    ](output_buf.unsafe_ptr())
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BS, CS), MutAnyOrigin
    ](cache_buf.unsafe_ptr())

    M.forward_gpu[BS](
        ctx, output_t, input_t, gpu.params_view(), cache_t, workspace
    )

    var grad_in_buf = ctx.enqueue_create_buffer[dtype](BS * IN)
    ctx.enqueue_memset(grad_in_buf, 0)
    var grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BS, IN), MutAnyOrigin
    ](grad_in_buf.unsafe_ptr())
    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BS, OUT), MutAnyOrigin
    ](grad_out_buf.unsafe_ptr())

    gpu.zero_grads(ctx)
    var grads = gpu.grads_view()
    M.backward_gpu[BS](
        ctx, grad_in_t, grad_out_t, gpu.params_view(), cache_t,
        grads, workspace,
    )

    # Read analytical gradients + params to host
    var ana_grads_host = ctx.enqueue_create_host_buffer[dtype](PS)
    var params_host = ctx.enqueue_create_host_buffer[dtype](PS)
    var grad_out_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
    ctx.enqueue_copy(ana_grads_host, gpu.grads_buf)
    ctx.enqueue_copy(params_host, gpu.params_buf)
    ctx.enqueue_copy(grad_out_host, grad_out_buf)
    ctx.synchronize()

    # 2. Finite differences for sampled parameters
    var step = PS // max_params_to_check
    if step < 1:
        step = 1

    var max_abs: Float64 = 0.0
    var max_rel: Float64 = 0.0
    var num_checked = 0
    var num_fail = 0

    var out_plus_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    var out_minus_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    var cache_tmp = ctx.enqueue_create_buffer[dtype](BS * CS)
    var out_plus_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
    var out_minus_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)

    # Perturb a single param on GPU via a tiny kernel
    from std.gpu import thread_idx

    @always_inline
    def _write_one_param(
        buf: LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin],
        val: Scalar[dtype],
        idx: Int,
    ):
        if Int(thread_idx.x) == 0:
            buf[idx] = val

    var params_t = gpu.params_view()

    for p_idx in range(0, PS, step):
        var orig = params_host[p_idx]

        # f(p + eps)
        ctx.enqueue_function[_write_one_param, _write_one_param](
            params_t, Scalar[dtype](Float64(orig) + eps), p_idx,
            grid_dim=(1,), block_dim=(1,),
        )
        var out_plus_t = LayoutTensor[
            dtype, Layout.row_major(BS, OUT), MutAnyOrigin
        ](out_plus_buf.unsafe_ptr())
        var cache_tmp_t = LayoutTensor[
            dtype, Layout.row_major(BS, CS), MutAnyOrigin
        ](cache_tmp.unsafe_ptr())
        M.forward_gpu[BS](
            ctx, out_plus_t, input_t, gpu.params_view(), cache_tmp_t, workspace
        )
        ctx.enqueue_copy(out_plus_host, out_plus_buf)

        # f(p - eps)
        ctx.enqueue_function[_write_one_param, _write_one_param](
            params_t, Scalar[dtype](Float64(orig) - eps), p_idx,
            grid_dim=(1,), block_dim=(1,),
        )
        var out_minus_t = LayoutTensor[
            dtype, Layout.row_major(BS, OUT), MutAnyOrigin
        ](out_minus_buf.unsafe_ptr())
        var cache_minus_t = LayoutTensor[
            dtype, Layout.row_major(BS, CS), MutAnyOrigin
        ](cache_tmp.unsafe_ptr())
        M.forward_gpu[BS](
            ctx, out_minus_t, input_t, gpu.params_view(), cache_minus_t, workspace
        )
        ctx.enqueue_copy(out_minus_host, out_minus_buf)

        # Restore
        ctx.enqueue_function[_write_one_param, _write_one_param](
            params_t, orig, p_idx,
            grid_dim=(1,), block_dim=(1,),
        )
        ctx.synchronize()

        # Numerical gradient = sum_j(grad_output_j * (f_plus_j - f_minus_j) / (2*eps))
        var num_grad: Float64 = 0.0
        for j in range(BS * OUT):
            var go = Float64(grad_out_host[j])
            var fp = Float64(out_plus_host[j])
            var fm = Float64(out_minus_host[j])
            num_grad += go * (fp - fm) / (2.0 * eps)

        var ana_grad = Float64(ana_grads_host[p_idx])
        var err = abs(ana_grad - num_grad)
        var denom = abs(ana_grad) + abs(num_grad)
        var rel: Float64 = 0.0
        if denom > 1e-5:
            rel = err / denom

        if err > max_abs:
            max_abs = err
        if rel > max_rel:
            max_rel = rel
        if rel > 0.01 and denom > 1e-4:
            num_fail += 1
            if num_fail <= 5:
                print(
                    "    MISMATCH param[", p_idx, "]: analytical=",
                    ana_grad, "numerical=", num_grad,
                    "rel_err=", rel,
                )
        num_checked += 1

    return (max_abs, max_rel, num_checked, num_fail)


def test_gradcheck[Config: AlphaZeroConfig](
    ctx: DeviceContext, name: String
) raises:
    """GPU numerical gradient check for an AlphaZero config."""
    comptime M = Config.PredModel
    comptime BS = 4
    comptime IN = M.IN_DIM
    comptime OUT = M.OUT_DIM
    comptime PS = M.PARAM_SIZE
    comptime WS = M.WORKSPACE_SIZE_PER_SAMPLE

    print("Gradcheck:", name, "(PS=", PS, ")")

    # Init
    var cpu_state = NetworkState[M, Adam[]]()
    cpu_state.initialize[Xavier[]]()
    var gpu = GPUNetworkState[M, Adam[], dtype](ctx)
    gpu.upload_from(cpu_state, ctx)

    var workspace = ctx.enqueue_create_buffer[dtype](BS * WS if WS > 0 else 1)

    # Input
    var input_host = ctx.enqueue_create_host_buffer[dtype](BS * IN)
    for i in range(BS * IN):
        # Use values that avoid ReLU dead zones
        input_host[i] = Scalar[dtype](0.1 + Float64(i % 13) / 13.0 * 0.8)
    var input_buf = ctx.enqueue_create_buffer[dtype](BS * IN)
    ctx.enqueue_copy(input_buf, input_host)

    # Grad output (random-ish, not all same to catch cross-output bugs)
    var grad_out_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
    for i in range(BS * OUT):
        grad_out_host[i] = Scalar[dtype](
            0.5 + Float64(i % 7) / 14.0 - Float64(i % 3) / 6.0
        )
    var grad_out_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    ctx.enqueue_copy(grad_out_buf, grad_out_host)
    ctx.synchronize()

    var result = gpu_finite_diff_check[M, BS](
        ctx, gpu, input_buf, grad_out_buf, workspace,
        eps=1e-3,
        max_params_to_check=300,
    )

    var max_abs = result[0]
    var max_rel = result[1]
    var num_checked = result[2]
    var num_fail = result[3]

    if num_fail == 0 and max_rel < 0.02:
        print(
            "  [PASS] max_rel_err=", max_rel,
            "max_abs_err=", max_abs,
            "(", num_checked, "params checked)",
        )
    elif num_fail == 0:
        print(
            "  [WARN] max_rel_err=", max_rel,
            "(", num_checked, "params checked) — slightly high but no hard failures",
        )
    else:
        print(
            "  [FAIL]", num_fail, "params with rel_err > 1% out of", num_checked,
            "checked. max_rel=", max_rel,
        )
    print()


def test_gradcheck_model[
    M: Model,
    BS: Int = 4,
](
    ctx: DeviceContext, name: String,
    eps: Float64 = 1e-3,
    max_params: Int = 300,
) raises:
    """GPU gradcheck for an arbitrary Model (no config needed)."""
    comptime IN = M.IN_DIM
    comptime OUT = M.OUT_DIM
    comptime PS = M.PARAM_SIZE
    comptime WS = M.WORKSPACE_SIZE_PER_SAMPLE

    print("Gradcheck:", name, "(PS=", PS, "IN=", IN, "OUT=", OUT, ")")

    var cpu_state = NetworkState[M, Adam[]]()
    cpu_state.initialize[Xavier[]]()
    var gpu = GPUNetworkState[M, Adam[], dtype](ctx)
    gpu.upload_from(cpu_state, ctx)

    var workspace = ctx.enqueue_create_buffer[dtype](BS * WS if WS > 0 else 1)

    var input_host = ctx.enqueue_create_host_buffer[dtype](BS * IN)
    for i in range(BS * IN):
        input_host[i] = Scalar[dtype](0.1 + Float64(i % 13) / 13.0 * 0.8)
    var input_buf = ctx.enqueue_create_buffer[dtype](BS * IN)
    ctx.enqueue_copy(input_buf, input_host)

    var grad_out_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
    for i in range(BS * OUT):
        grad_out_host[i] = Scalar[dtype](
            0.5 + Float64(i % 7) / 14.0 - Float64(i % 3) / 6.0
        )
    var grad_out_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    ctx.enqueue_copy(grad_out_buf, grad_out_host)
    ctx.synchronize()

    var result = gpu_finite_diff_check[M, BS](
        ctx, gpu, input_buf, grad_out_buf, workspace,
        eps=eps, max_params_to_check=max_params,
    )

    var max_abs = result[0]
    var max_rel = result[1]
    var num_checked = result[2]
    var num_fail = result[3]

    if num_fail == 0 and max_rel < 0.02:
        print(
            "  [PASS] max_rel_err=", max_rel,
            "max_abs_err=", max_abs,
            "(", num_checked, "params checked)",
        )
    elif num_fail == 0:
        print(
            "  [WARN] max_rel_err=", max_rel,
            "(", num_checked, "params checked) — slightly high but no hard failures",
        )
    else:
        print(
            "  [FAIL]", num_fail, "params with rel_err > 1% out of", num_checked,
            "checked. max_rel=", max_rel,
        )
    print()


from mojo_rl.deep_agents.alphazero.configs import AlphaZeroConfig
from mojo_rl.nn.model import (
    Model, Sequential, Parallel, Linear, LinearReLU,
    FlattenLayer, Conv2DReLU, Conv2DBatchNormReLU, ReLU,
)


def test_forward_only[M: Model, BS: Int = 4](
    ctx: DeviceContext, name: String
) raises:
    """Minimal forward-only test to isolate crash location."""
    comptime IN = M.IN_DIM
    comptime OUT = M.OUT_DIM
    comptime PS = M.PARAM_SIZE
    comptime CS = M.CACHE_SIZE
    comptime WS = M.WORKSPACE_SIZE_PER_SAMPLE

    print("Forward-only:", name, "(IN=", IN, "OUT=", OUT, "PS=", PS, "WS=", WS, ")")

    var cpu_state = NetworkState[M, Adam[]]()
    cpu_state.initialize[Kaiming[]]()
    var gpu = GPUNetworkState[M, Adam[], dtype](ctx)
    gpu.upload_from(cpu_state, ctx)

    var workspace = ctx.enqueue_create_buffer[dtype](BS * WS if WS > 0 else 1)
    var input_host = ctx.enqueue_create_host_buffer[dtype](BS * IN)
    for i in range(BS * IN):
        input_host[i] = Scalar[dtype](Float64(i % 7) / 7.0)
    var input_buf = ctx.enqueue_create_buffer[dtype](BS * IN)
    ctx.enqueue_copy(input_buf, input_host)

    var output_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    var cache_buf = ctx.enqueue_create_buffer[dtype](BS * CS if CS > 0 else 1)

    var input_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](
        input_buf.unsafe_ptr()
    )
    var output_t = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](
        output_buf.unsafe_ptr()
    )
    var cache_t = LayoutTensor[dtype, Layout.row_major(BS, CS), MutAnyOrigin](
        cache_buf.unsafe_ptr()
    )

    M.forward_gpu[BS](
        ctx, output_t, input_t, gpu.params_view(), cache_t, workspace
    )
    ctx.synchronize()
    print("  [PASS]")


def main() raises:
    print("=== AlphaZero Architecture Tests ===")
    print()

    var ctx = DeviceContext()

    # ── Isolation tests: find exactly what crashes on NVIDIA ──
    print("--- Isolation tests ---")

    # A. Conv2DBatchNormReLU 3x3 (known to work)
    comptime Conv3x3 = Conv2DBatchNormReLU[3, 128, 3, 1, 1, 6, 7]
    test_forward_only[Conv3x3](ctx, "Conv2DBatchNormReLU 3x3 (3->128)")

    # B. Conv2DBatchNormReLU 1x1 ALONE (the new layer)
    comptime Conv1x1 = Conv2DBatchNormReLU[128, 32, 1, 1, 0, 6, 7]
    test_forward_only[Conv1x1](ctx, "Conv2DBatchNormReLU 1x1 (128->32)")

    # C. Sequential with 1x1 conv + flatten + linear (single head, no Parallel)
    comptime SingleHead = Sequential[
        Conv2DBatchNormReLU[128, 32, 1, 1, 0, 6, 7],
        FlattenLayer[32 * 6 * 7],
        Linear[32 * 6 * 7, 7],
    ]
    test_forward_only[SingleHead](ctx, "Sequential[Conv1x1+BN+ReLU, Flatten, Linear]")

    # C2. ValueHead Sequential alone (4 layers with LinearReLU)
    comptime ValueHead = Sequential[
        Conv2DBatchNormReLU[128, 32, 1, 1, 0, 6, 7],
        FlattenLayer[32 * 6 * 7],
        LinearReLU[32 * 6 * 7, 128],
        Linear[128, 1],
    ]
    test_forward_only[ValueHead](ctx, "ValueHead Sequential alone")

    # D. Parallel with two simple heads (no conv, from trunk output dim)
    comptime SimplePar = Parallel[Linear[5376, 7], Linear[5376, 1]]
    test_forward_only[SimplePar](ctx, "Parallel[Linear, Linear] from 5376-dim")

    # E1. Parallel with ONLY policy head (1 branch)
    comptime PolicyOnly = Parallel[
        Sequential[
            Conv2DBatchNormReLU[128, 32, 1, 1, 0, 6, 7],
            FlattenLayer[32 * 6 * 7],
            Linear[32 * 6 * 7, 7],
        ],
    ]
    test_forward_only[PolicyOnly](ctx, "Parallel[PolicyHead only]")

    # E2. Parallel with ONLY value head (1 branch)
    comptime ValueOnly = Parallel[ValueHead]
    test_forward_only[ValueOnly](ctx, "Parallel[ValueHead only]")

    # E3. Parallel with two IDENTICAL policy heads (both 3-layer)
    comptime TwoPolicyPar = Parallel[
        Sequential[
            Conv2DBatchNormReLU[128, 32, 1, 1, 0, 6, 7],
            FlattenLayer[32 * 6 * 7],
            Linear[32 * 6 * 7, 7],
        ],
        Sequential[
            Conv2DBatchNormReLU[128, 32, 1, 1, 0, 6, 7],
            FlattenLayer[32 * 6 * 7],
            Linear[32 * 6 * 7, 7],
        ],
    ]
    test_forward_only[TwoPolicyPar](ctx, "Parallel[PolicyHead, PolicyHead] (identical branches)")

    # E. Parallel with conv heads (policy + value)
    comptime ConvPar = Parallel[
        Sequential[
            Conv2DBatchNormReLU[128, 32, 1, 1, 0, 6, 7],
            FlattenLayer[32 * 6 * 7],
            Linear[32 * 6 * 7, 7],
        ],
        ValueHead,
    ]
    test_forward_only[ConvPar](ctx, "Parallel[PolicyHead, ValueHead] (the dual-head)")

    print("\n--- Full tests ---")

    # ── Sanity tests (fast) ──────────────────────────────────
    test_forward_backward[AlphaZeroTicTacToeConfig[]](ctx, "TicTacToe MLP")
    test_forward_backward[AlphaZeroConnectFourFusedResNetConfig[NUM_BLOCKS=1]](
        ctx, "ConnectFour FusedResNet (1 block, quick)"
    )

    # ── Numerical gradient checks ────────────────────────────

    # 1. TicTacToe MLP (no BN — should be clean)
    test_gradcheck[AlphaZeroTicTacToeConfig[]](ctx, "TicTacToe MLP")

    # 2. BN-free Parallel[Sequential, Sequential] — tests combinator logic
    #    without BN noise. Mimics the dual-head structure.
    comptime NoBN_DualHead = Sequential[
        LinearReLU[126, 64],  # Shared trunk (like flatten+FC)
        Parallel[
            # Policy head: FC → FC
            Sequential[LinearReLU[64, 32], Linear[32, 7]],
            # Value head: FC → FC → FC
            Sequential[LinearReLU[64, 32], LinearReLU[32, 16], Linear[16, 1]],
        ],
    ]
    test_gradcheck_model[NoBN_DualHead](
        ctx, "BN-free Parallel[Sequential, Sequential] dual-head"
    )

    # 3. Conv+Parallel without BN — tests Conv2D in heads
    comptime Conv_DualHead = Sequential[
        Conv2DReLU[3, 16, 3, 1, 1, 6, 7],  # Conv trunk (no BN)
        Parallel[
            # Policy: Conv1x1(no BN) → Flatten → FC
            Sequential[
                Conv2DReLU[16, 4, 1, 1, 0, 6, 7],
                FlattenLayer[4 * 6 * 7],
                Linear[4 * 6 * 7, 7],
            ],
            # Value: Conv1x1(no BN) → Flatten → FC → FC
            Sequential[
                Conv2DReLU[16, 4, 1, 1, 0, 6, 7],
                FlattenLayer[4 * 6 * 7],
                LinearReLU[4 * 6 * 7, 16],
                Linear[16, 1],
            ],
        ],
    ]
    test_gradcheck_model[Conv_DualHead](
        ctx, "Conv+Parallel dual-head (no BN — isolates conv grad path)"
    )

    # 4. Full FusedResNet 1-block with larger batch (BS=32) — BN noise should shrink
    test_gradcheck_model[
        AlphaZeroConnectFourFusedResNetConfig[NUM_BLOCKS=1].PredModel, 32
    ](
        ctx,
        "FusedResNet 1-block BS=32 (larger batch reduces BN noise)",
        max_params=200,
    )

    print("=== Done ===")

"""Minimal GPU gradcheck isolation tests for NVIDIA.

Progressively tests from single Linear up to Parallel to find where
NVIDIA max_matmul diverges from Apple.

Usage:
    pixi run -e nvidia mojo run -I . tests/nn/test_nvidia_gradcheck_isolate.mojo
    pixi run -e apple mojo run -I . tests/nn/test_nvidia_gradcheck_isolate.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import thread_idx, block_idx, block_dim
from std.math import abs, sqrt
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.training import NetworkState, GPUNetworkState
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.model import (
    Model, Sequential, Parallel, Linear, LinearReLU, ReLU,
)


def gradcheck[M: Model, BS: Int = 4](
    ctx: DeviceContext, name: String, eps: Float64 = 1e-3, max_check: Int = 200,
) raises:
    """Compact GPU gradcheck: perturb via host upload to avoid kernel issues."""
    comptime IN = M.IN_DIM
    comptime OUT = M.OUT_DIM
    comptime PS = M.PARAM_SIZE
    comptime CS = M.CACHE_SIZE
    comptime WS = M.WORKSPACE_SIZE_PER_SAMPLE

    print("Gradcheck:", name, "(IN=", IN, "OUT=", OUT, "PS=", PS, ")")

    var cpu_state = NetworkState[M, Adam[]]()
    cpu_state.initialize[Xavier[]]()
    var gpu = GPUNetworkState[M, Adam[], dtype](ctx)
    gpu.upload_from(cpu_state, ctx)

    var workspace = ctx.enqueue_create_buffer[dtype](BS * WS if WS > 0 else 1)

    # Input (deterministic)
    var input_host = ctx.enqueue_create_host_buffer[dtype](BS * IN)
    for i in range(BS * IN):
        input_host[i] = Scalar[dtype](0.1 + Float64(i % 13) / 13.0 * 0.8)
    var input_buf = ctx.enqueue_create_buffer[dtype](BS * IN)
    ctx.enqueue_copy(input_buf, input_host)

    # Grad output (varied)
    var grad_out_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
    for i in range(BS * OUT):
        grad_out_host[i] = Scalar[dtype](
            0.5 + Float64(i % 7) / 14.0 - Float64(i % 3) / 6.0
        )
    var grad_out_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    ctx.enqueue_copy(grad_out_buf, grad_out_host)

    # --- Analytical backward ---
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

    var ana_grads_host = ctx.enqueue_create_host_buffer[dtype](PS)
    ctx.enqueue_copy(ana_grads_host, gpu.grads_buf)

    # Read params to host for perturbation
    var params_host = ctx.enqueue_create_host_buffer[dtype](PS)
    ctx.enqueue_copy(params_host, gpu.params_buf)
    ctx.synchronize()

    # --- Finite differences via HOST-SIDE perturbation ---
    # Instead of GPU kernel write, upload entire params buffer from host.
    # This avoids any GPU cache/write visibility issues.
    var params_copy = ctx.enqueue_create_host_buffer[dtype](PS)
    var out_plus_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    var out_minus_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    var out_plus_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
    var out_minus_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
    var cache_tmp = ctx.enqueue_create_buffer[dtype](BS * CS if CS > 0 else 1)
    var grad_out_host2 = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
    ctx.enqueue_copy(grad_out_host2, grad_out_buf)
    ctx.synchronize()

    var step = PS // max_check
    if step < 1:
        step = 1

    var max_abs: Float64 = 0.0
    var max_rel: Float64 = 0.0
    var num_checked = 0
    var num_fail = 0

    for p_idx in range(0, PS, step):
        # Copy full params to host, perturb one, upload back
        for i in range(PS):
            params_copy[i] = params_host[i]

        var orig = Float64(params_host[p_idx])

        # f(p + eps): full host→device upload
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

        # f(p - eps): full host→device upload
        params_copy[p_idx] = Scalar[dtype](orig - eps)
        ctx.enqueue_copy(gpu.params_buf, params_copy)
        var out_minus_t = LayoutTensor[
            dtype, Layout.row_major(BS, OUT), MutAnyOrigin
        ](out_minus_buf.unsafe_ptr())
        var cache_minus_t = LayoutTensor[
            dtype, Layout.row_major(BS, CS), MutAnyOrigin
        ](cache_tmp.unsafe_ptr())
        M.forward_gpu[BS](
            ctx, out_minus_t, input_t, gpu.params_view(), gpu.model_state_view(), cache_minus_t, workspace
        )
        ctx.enqueue_copy(out_minus_host, out_minus_buf)

        # Restore original
        params_copy[p_idx] = Scalar[dtype](orig)
        ctx.enqueue_copy(gpu.params_buf, params_copy)
        ctx.synchronize()

        # Numerical gradient
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

        if err > max_abs:
            max_abs = err
        if rel > max_rel:
            max_rel = rel
        if rel > 0.01 and denom > 1e-4:
            num_fail += 1
            if num_fail <= 3:
                print(
                    "    MISMATCH p[", p_idx, "]: ana=",
                    ana_grad, "num=", num_grad, "rel=", rel,
                )
        num_checked += 1

    if num_fail == 0:
        print("  [PASS] max_rel=", max_rel, "(", num_checked, "checked)")
    else:
        print(
            "  [FAIL]", num_fail, "/", num_checked,
            "max_rel=", max_rel,
        )
    print()


def main() raises:
    print("=== NVIDIA Gradcheck Isolation ===")
    print()
    var ctx = DeviceContext()

    # Level 1: Single layers
    gradcheck[Linear[8, 4]](ctx, "Linear[8,4]")
    gradcheck[LinearReLU[8, 4]](ctx, "LinearReLU[8,4]")
    gradcheck[Linear[128, 1]](ctx, "Linear[128,1] (small output)")

    # Level 2: Small Sequential
    gradcheck[Sequential[LinearReLU[8, 6], Linear[6, 3]]](
        ctx, "Sequential[LinearReLU[8,6], Linear[6,3]]"
    )

    # Level 3: Parallel with same-sized branches
    gradcheck[Parallel[Linear[8, 4], Linear[8, 4]]](
        ctx, "Parallel[Linear[8,4], Linear[8,4]] (identical)"
    )

    # Level 4: Parallel with different-sized branches (the failing case)
    gradcheck[Parallel[Linear[8, 4], Linear[8, 1]]](
        ctx, "Parallel[Linear[8,4], Linear[8,1]] (different)"
    )

    # Level 5: Parallel with Sequential branches
    gradcheck[
        Parallel[
            Sequential[LinearReLU[8, 6], Linear[6, 4]],
            Sequential[LinearReLU[8, 6], Linear[6, 1]],
        ]
    ](ctx, "Parallel[Seq(8→6→4), Seq(8→6→1)] (different Sequential)")

    # Level 6: Larger — matches TicTacToe structure
    gradcheck[
        Sequential[
            LinearReLU[27, 128],
            LinearReLU[128, 128],
            Parallel[Linear[128, 9], Linear[128, 1]],
        ]
    ](ctx, "TicTacToe MLP architecture")

    # Level 7: BN-free Parallel with larger FC branches (was 154/304 fail in old test)
    gradcheck[
        Sequential[
            LinearReLU[126, 64],
            Parallel[
                Sequential[LinearReLU[64, 32], Linear[32, 7]],
                Sequential[LinearReLU[64, 32], LinearReLU[32, 16], Linear[16, 1]],
            ],
        ]
    ](ctx, "BN-free Parallel[Seq, Seq] dual-head (larger)")

    # Level 8: Conv2D heads (was crashing / 84/321 fail in old test)
    from mojo_rl.nn.model import Conv2DReLU, FlattenLayer
    gradcheck[
        Sequential[
            Conv2DReLU[3, 16, 3, 1, 1, 6, 7],
            Parallel[
                Sequential[
                    Conv2DReLU[16, 4, 1, 1, 0, 6, 7],
                    FlattenLayer[4 * 6 * 7],
                    Linear[4 * 6 * 7, 7],
                ],
                Sequential[
                    Conv2DReLU[16, 4, 1, 1, 0, 6, 7],
                    FlattenLayer[4 * 6 * 7],
                    LinearReLU[4 * 6 * 7, 16],
                    Linear[16, 1],
                ],
            ],
        ]
    ](ctx, "Conv+Parallel dual-head (no BN)")

    # Level 9: Full AlphaZero FusedResNet 1-block (was 197/201 fail in old test)
    from mojo_rl.deep_agents.alphazero import AlphaZeroConnectFourFusedResNetConfig
    comptime FusedResNet1 = AlphaZeroConnectFourFusedResNetConfig[NUM_BLOCKS=1].PredModel
    gradcheck[FusedResNet1, 4](ctx, "FusedResNet 1-block (full architecture)")

    # ── CPU vs GPU comparison ─────────────────────────────────
    print("--- CPU vs GPU forward comparison ---")
    cpu_vs_gpu_forward[
        Sequential[
            LinearReLU[126, 64],
            Parallel[
                Sequential[LinearReLU[64, 32], Linear[32, 7]],
                Sequential[LinearReLU[64, 32], LinearReLU[32, 16], Linear[16, 1]],
            ],
        ]
    ](ctx, "BN-free dual-head")

    cpu_vs_gpu_forward[
        Sequential[
            LinearReLU[27, 128],
            LinearReLU[128, 128],
            Parallel[Linear[128, 9], Linear[128, 1]],
        ]
    ](ctx, "TicTacToe MLP")

    print("=== Done ===")


def cpu_vs_gpu_forward[M: Model, BS: Int = 4](
    ctx: DeviceContext, name: String
) raises:
    """Compare CPU and GPU forward pass outputs for the same params/input."""
    comptime IN = M.IN_DIM
    comptime OUT = M.OUT_DIM
    comptime PS = M.PARAM_SIZE
    comptime CS = M.CACHE_SIZE
    comptime WS = M.WORKSPACE_SIZE_PER_SAMPLE

    print("CPU vs GPU:", name, "(IN=", IN, "OUT=", OUT, ")")

    # Initialize on CPU
    var cpu_state = NetworkState[M, Adam[]]()
    cpu_state.initialize[Xavier[]]()

    # Create input
    var input_arr = List[Scalar[dtype]](capacity=BS * IN)
    for i in range(BS * IN):
        input_arr.append(Scalar[dtype](0.1 + Float64(i % 13) / 13.0 * 0.8))

    # CPU forward
    var cpu_out = List[Scalar[dtype]](capacity=BS * OUT)
    for i in range(BS * OUT):
        cpu_out.append(Scalar[dtype](0.0))
    var cpu_cache = List[Scalar[dtype]](capacity=BS * CS + 1)
    for i in range(BS * CS + 1):
        cpu_cache.append(Scalar[dtype](0.0))

    var input_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](
        input_arr.unsafe_ptr()
    )
    var cpu_out_t = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](
        cpu_out.unsafe_ptr()
    )
    var cpu_cache_t = LayoutTensor[dtype, Layout.row_major(BS, CS), MutAnyOrigin](
        cpu_cache.unsafe_ptr()
    )

    M.forward[BS](input_t, cpu_out_t, cpu_state.params_view(), cpu_state.model_state_view(), cpu_cache_t)

    # GPU forward with same params
    var gpu = GPUNetworkState[M, Adam[], dtype](ctx)
    gpu.upload_from(cpu_state, ctx)

    var workspace = ctx.enqueue_create_buffer[dtype](BS * WS if WS > 0 else 1)
    var input_host = ctx.enqueue_create_host_buffer[dtype](BS * IN)
    for i in range(BS * IN):
        input_host[i] = input_arr[i]
    var input_buf = ctx.enqueue_create_buffer[dtype](BS * IN)
    ctx.enqueue_copy(input_buf, input_host)

    var output_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    var cache_buf = ctx.enqueue_create_buffer[dtype](BS * CS if CS > 0 else 1)

    var gpu_input_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](
        input_buf.unsafe_ptr()
    )
    var gpu_out_t = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](
        output_buf.unsafe_ptr()
    )
    var gpu_cache_t = LayoutTensor[dtype, Layout.row_major(BS, CS), MutAnyOrigin](
        cache_buf.unsafe_ptr()
    )

    M.forward_gpu[BS](
        ctx, gpu_out_t, gpu_input_t, gpu.params_view(), gpu.model_state_view(), gpu_cache_t, workspace
    )

    var gpu_out_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
    ctx.enqueue_copy(gpu_out_host, output_buf)
    ctx.synchronize()

    # Compare
    var max_abs: Float64 = 0.0
    var max_rel: Float64 = 0.0
    var n_mismatch = 0
    for i in range(BS * OUT):
        var c = Float64(cpu_out[i])
        var g = Float64(gpu_out_host[i])
        var err = abs(c - g)
        var denom = abs(c) + abs(g)
        var rel: Float64 = 0.0
        if denom > 1e-8:
            rel = err / denom
        if err > max_abs:
            max_abs = err
        if rel > max_rel:
            max_rel = rel
        if rel > 0.01:
            n_mismatch += 1
            if n_mismatch <= 3:
                print(
                    "    OUT[", i, "]: cpu=", c, "gpu=", g,
                    "rel=", rel,
                )

    if n_mismatch == 0:
        print("  [PASS] max_rel=", max_rel, "max_abs=", max_abs)
    else:
        print("  [FAIL]", n_mismatch, "/", BS * OUT, "max_rel=", max_rel)
    print()

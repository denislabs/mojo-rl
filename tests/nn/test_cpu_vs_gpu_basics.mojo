"""CPU vs GPU consistency — leaf layers, activations, combinators.

Split from test_cpu_vs_gpu.mojo to keep NVIDIA comptime budget manageable
(too many specializations of cpu_vs_gpu_check[M] in one file blow up nvcc).

Usage:
    pixi run -e apple mojo run -I . tests/nn/test_cpu_vs_gpu_basics.mojo
    pixi run -e nvidia mojo run -I . tests/nn/test_cpu_vs_gpu_basics.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import abs
from std.memory import alloc, memset
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training import NetworkState, GPUNetworkState
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.model import (
    Model,
    Sequential,
    Parallel,
    Linear,
    LinearReLU,
    LinearTanh,
    ReLU,
    Tanh,
    Sigmoid,
    LayerNorm,
    Residual,
    Repeat,
    SkipConcat,
    DualPath,
    FanOut,
)


def cpu_vs_gpu_check[M: Model, BS: Int = 4](
    ctx: DeviceContext,
    name: String,
    fwd_tol: Float64 = 1e-4,
    bwd_tol: Float64 = 1e-3,
) raises:
    """Compare CPU forward/backward against GPU for the same model and inputs."""
    comptime IN = M.IN_DIM
    comptime OUT = M.OUT_DIM
    comptime PS = M.PARAM_SIZE
    comptime CS = M.CACHE_SIZE
    comptime WS = M.WORKSPACE_SIZE_PER_SAMPLE

    print("CPU vs GPU:", name, "(IN=", IN, "OUT=", OUT, "PS=", PS, ")")

    var cpu_state = NetworkState[M, Adam[]]()
    cpu_state.initialize[Xavier[]]()

    var gpu = GPUNetworkState[M, Adam[], dtype](ctx)
    gpu.upload_from(cpu_state, ctx)

    var input_ptr = alloc[Scalar[dtype]](BS * IN)
    for i in range(BS * IN):
        (input_ptr + i)[] = Scalar[dtype](0.1 + Float64(i % 13) / 13.0 * 0.8)

    var cpu_output_ptr = alloc[Scalar[dtype]](BS * OUT)
    var cpu_cache_ptr = alloc[Scalar[dtype]](BS * CS if CS > 0 else 1)
    var cpu_input_t = LayoutTensor[
        dtype, Layout.row_major(BS, IN), MutAnyOrigin
    ](input_ptr)
    var cpu_output_t = LayoutTensor[
        dtype, Layout.row_major(BS, OUT), MutAnyOrigin
    ](cpu_output_ptr)
    var cpu_cache_t = LayoutTensor[
        dtype, Layout.row_major(BS, CS), MutAnyOrigin
    ](cpu_cache_ptr)

    var cpu_model_state = cpu_state.model_state_view()
    M.forward[BS](cpu_input_t, cpu_output_t, cpu_state.params_view(), cpu_model_state, cpu_cache_t)

    var gpu_input_host = ctx.enqueue_create_host_buffer[dtype](BS * IN)
    for i in range(BS * IN):
        gpu_input_host[i] = (input_ptr + i)[]
    var gpu_input_buf = ctx.enqueue_create_buffer[dtype](BS * IN)
    ctx.enqueue_copy(gpu_input_buf, gpu_input_host)

    var gpu_output_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    var gpu_cache_buf = ctx.enqueue_create_buffer[dtype](
        BS * CS if CS > 0 else 1
    )
    var workspace = ctx.enqueue_create_buffer[dtype](
        BS * WS if WS > 0 else 1
    )

    var gpu_input_t = LayoutTensor[
        dtype, Layout.row_major(BS, IN), MutAnyOrigin
    ](gpu_input_buf.unsafe_ptr())
    var gpu_output_t = LayoutTensor[
        dtype, Layout.row_major(BS, OUT), MutAnyOrigin
    ](gpu_output_buf.unsafe_ptr())
    var gpu_cache_t = LayoutTensor[
        dtype, Layout.row_major(BS, CS), MutAnyOrigin
    ](gpu_cache_buf.unsafe_ptr())

    var gpu_model_state = gpu.model_state_view()
    M.forward_gpu[BS](
        ctx,
        gpu_output_t,
        gpu_input_t,
        gpu.params_view(),
        gpu_model_state,
        gpu_cache_t,
        workspace,
    )

    var gpu_output_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
    ctx.enqueue_copy(gpu_output_host, gpu_output_buf)
    ctx.synchronize()

    var fwd_max_abs: Float64 = 0.0
    var fwd_max_rel: Float64 = 0.0
    var fwd_fail = 0
    for i in range(BS * OUT):
        var cpu_val = Float64((cpu_output_ptr + i)[])
        var gpu_val = Float64(gpu_output_host[i])
        var err = abs(cpu_val - gpu_val)
        var denom = abs(cpu_val) + abs(gpu_val)
        var rel: Float64 = 0.0
        if denom > 1e-7:
            rel = err / denom
        if err > fwd_max_abs:
            fwd_max_abs = err
        if rel > fwd_max_rel:
            fwd_max_rel = rel
        if rel > fwd_tol and denom > 1e-6:
            fwd_fail += 1
            if fwd_fail <= 3:
                print(
                    "    FWD[", i, "]: cpu=", cpu_val, "gpu=", gpu_val,
                    "rel=", rel,
                )

    if fwd_fail == 0:
        print("  [PASS] forward: max_abs=", fwd_max_abs, "max_rel=", fwd_max_rel)
    else:
        print(
            "  [FAIL] forward:", fwd_fail, "/", BS * OUT,
            "max_abs=", fwd_max_abs, "max_rel=", fwd_max_rel,
        )

    var grad_out_ptr = alloc[Scalar[dtype]](BS * OUT)
    for i in range(BS * OUT):
        (grad_out_ptr + i)[] = Scalar[dtype](
            0.5 + Float64(i % 7) / 14.0 - Float64(i % 3) / 6.0
        )

    cpu_state.zero_grads()
    var cpu_grad_in_ptr = alloc[Scalar[dtype]](BS * IN)
    memset(cpu_grad_in_ptr, 0, BS * IN)
    var cpu_bwd_grad_out_ptr = alloc[Scalar[dtype]](BS * OUT)
    for i in range(BS * OUT):
        (cpu_bwd_grad_out_ptr + i)[] = (grad_out_ptr + i)[]
    var cpu_grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BS, OUT), MutAnyOrigin
    ](cpu_bwd_grad_out_ptr)
    var cpu_grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BS, IN), MutAnyOrigin
    ](cpu_grad_in_ptr)

    var cpu_grads = cpu_state.grads_view()
    M.backward[BS](
        cpu_grad_out_t, cpu_grad_in_t, cpu_state.params_view(),
        cpu_model_state, cpu_cache_t, cpu_grads,
    )

    gpu.zero_grads(ctx)
    var gpu_grad_out_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
    for i in range(BS * OUT):
        gpu_grad_out_host[i] = (grad_out_ptr + i)[]
    var gpu_grad_out_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    ctx.enqueue_copy(gpu_grad_out_buf, gpu_grad_out_host)

    var gpu_grad_in_buf = ctx.enqueue_create_buffer[dtype](BS * IN)
    ctx.enqueue_memset(gpu_grad_in_buf, 0)

    var gpu_grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BS, OUT), MutAnyOrigin
    ](gpu_grad_out_buf.unsafe_ptr())
    var gpu_grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BS, IN), MutAnyOrigin
    ](gpu_grad_in_buf.unsafe_ptr())

    var gpu_grads = gpu.grads_view()
    M.backward_gpu[BS](
        ctx, gpu_grad_in_t, gpu_grad_out_t, gpu.params_view(),
        gpu_model_state, gpu_cache_t, gpu_grads, workspace,
    )

    var gpu_grads_host = ctx.enqueue_create_host_buffer[dtype](
        PS if PS > 0 else 1
    )
    if PS > 0:
        ctx.enqueue_copy(gpu_grads_host, gpu.grads_buf)
    var gpu_grad_in_host = ctx.enqueue_create_host_buffer[dtype](BS * IN)
    ctx.enqueue_copy(gpu_grad_in_host, gpu_grad_in_buf)
    ctx.synchronize()

    if PS > 0:
        var gp_max_abs: Float64 = 0.0
        var gp_max_rel: Float64 = 0.0
        var gp_fail = 0
        for i in range(PS):
            var cpu_g = Float64((cpu_state.grads + i)[])
            var gpu_g = Float64(gpu_grads_host[i])
            var err = abs(cpu_g - gpu_g)
            var denom = abs(cpu_g) + abs(gpu_g)
            var rel: Float64 = 0.0
            if denom > 1e-7:
                rel = err / denom
            if err > gp_max_abs:
                gp_max_abs = err
            if rel > gp_max_rel:
                gp_max_rel = rel
            if rel > bwd_tol and denom > 1e-6:
                gp_fail += 1
                if gp_fail <= 3:
                    print(
                        "    GRAD_P[", i, "]: cpu=", cpu_g, "gpu=", gpu_g,
                        "rel=", rel,
                    )

        if gp_fail == 0:
            print(
                "  [PASS] grad_params: max_abs=", gp_max_abs,
                "max_rel=", gp_max_rel,
            )
        else:
            print(
                "  [FAIL] grad_params:", gp_fail, "/", PS,
                "max_abs=", gp_max_abs, "max_rel=", gp_max_rel,
            )
    else:
        print("  [SKIP] grad_params: PARAM_SIZE=0")

    var gi_max_abs: Float64 = 0.0
    var gi_max_rel: Float64 = 0.0
    var gi_fail = 0
    for i in range(BS * IN):
        var cpu_g = Float64((cpu_grad_in_ptr + i)[])
        var gpu_g = Float64(gpu_grad_in_host[i])
        var err = abs(cpu_g - gpu_g)
        var denom = abs(cpu_g) + abs(gpu_g)
        var rel: Float64 = 0.0
        if denom > 1e-7:
            rel = err / denom
        if err > gi_max_abs:
            gi_max_abs = err
        if rel > gi_max_rel:
            gi_max_rel = rel
        if rel > bwd_tol and denom > 1e-6:
            gi_fail += 1
            if gi_fail <= 3:
                print(
                    "    GRAD_IN[", i, "]: cpu=", cpu_g, "gpu=", gpu_g,
                    "rel=", rel,
                )

    if gi_fail == 0:
        print(
            "  [PASS] grad_input: max_abs=", gi_max_abs,
            "max_rel=", gi_max_rel,
        )
    else:
        print(
            "  [FAIL] grad_input:", gi_fail, "/", BS * IN,
            "max_abs=", gi_max_abs, "max_rel=", gi_max_rel,
        )

    print()


def main() raises:
    print("=== NN CPU vs GPU Consistency — Basics ===")
    print()

    var ctx = DeviceContext()

    print("--- Leaf layers ---")
    cpu_vs_gpu_check[Linear[8, 4]](ctx, "Linear[8,4]")
    cpu_vs_gpu_check[Linear[32, 1]](ctx, "Linear[32,1] (small output)")
    cpu_vs_gpu_check[LinearReLU[16, 8]](ctx, "LinearReLU[16,8]")
    cpu_vs_gpu_check[LinearTanh[8, 4]](ctx, "LinearTanh[8,4]")
    cpu_vs_gpu_check[LayerNorm[16]](ctx, "LayerNorm[16]")

    print("--- Activations ---")
    cpu_vs_gpu_check[ReLU[8]](ctx, "ReLU[8]")
    cpu_vs_gpu_check[Tanh[8]](ctx, "Tanh[8]")
    cpu_vs_gpu_check[Sigmoid[8]](ctx, "Sigmoid[8]")

    print("--- Combinators ---")
    cpu_vs_gpu_check[Sequential[LinearReLU[8, 6], Linear[6, 4]]](
        ctx, "Sequential[LinearReLU, Linear]"
    )
    cpu_vs_gpu_check[Parallel[Linear[8, 4], Linear[8, 1]]](
        ctx, "Parallel[Linear(8->4), Linear(8->1)]"
    )
    cpu_vs_gpu_check[Residual[LinearReLU[8, 8]]](
        ctx, "Residual[LinearReLU[8,8]]"
    )
    cpu_vs_gpu_check[Repeat[2, LinearReLU[8, 8]]](
        ctx, "Repeat[2, LinearReLU[8,8]]"
    )
    cpu_vs_gpu_check[SkipConcat[Linear[8, 4]]](
        ctx, "SkipConcat[Linear[8,4]]"
    )
    cpu_vs_gpu_check[DualPath[Linear[8, 4], Linear[8, 1]]](
        ctx, "DualPath[Linear(8->4), Linear(8->1)]"
    )
    cpu_vs_gpu_check[FanOut[Linear[8, 4], 2]](
        ctx, "FanOut[Linear[8,4], N=2]"
    )

    print("--- MLP dual-head ---")
    comptime MLP_DualHead = Sequential[
        LinearReLU[27, 64],
        Parallel[Linear[64, 9], Linear[64, 1]],
    ]
    cpu_vs_gpu_check[MLP_DualHead](ctx, "MLP dual-head (TicTacToe-like)")

    print("=== Done ===")

"""MuZero CPU vs GPU parity test — network-level forward + backward.

The audit (docs/MUZERO_GPU_AUDIT_2026-05-22.md) recommends a single-step
parity test to isolate where GPU MuZero diverges from CPU MuZero. This
file is Stage 1 of that diagnostic: it verifies that each MuZero network
(rep, dyn, pred) produces identical forward outputs AND identical
backward gradients on CPU vs GPU, for the EXACT model types declared in
``MuZeroMLPConfig`` (LinearMish + Linear + MinMaxNorm + Parallel + Sequential).

Decision tree:
- All three pass → forward/backward kernels are bit-faithful. The GPU
  convergence bug is in update-level code (CE gradient kernels, K-step
  unroll wiring, 0.5 dual-consumer split, 1/K dyn scale, two-hot encoding,
  scalar transform). Investigate ``update_gpu`` itself next.
- Any fail → kernel bug. Smallest failing layer is the bisection target.

Reuses the ``cpu_vs_gpu_check`` pattern from
``tests/nn/test_cpu_vs_gpu_tdmpc2.mojo``.

Usage:
    pixi run -e apple mojo run -I . tests/deep_agents/test_muzero_cpu_gpu_parity.mojo
"""

from std.gpu.host import DeviceContext
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
    Linear,
    LinearMish,
    Parallel,
    MinMaxNorm,
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
    M.forward[BS](
        cpu_input_t,
        cpu_output_t,
        cpu_state.params_view(),
        cpu_model_state,
        cpu_cache_t,
    )

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
                    "    FWD[",
                    i,
                    "]: cpu=",
                    cpu_val,
                    "gpu=",
                    gpu_val,
                    "rel=",
                    rel,
                )

    if fwd_fail == 0:
        print(
            "  [PASS] forward: max_abs=",
            fwd_max_abs,
            "max_rel=",
            fwd_max_rel,
        )
    else:
        print(
            "  [FAIL] forward:",
            fwd_fail,
            "/",
            BS * OUT,
            "max_abs=",
            fwd_max_abs,
            "max_rel=",
            fwd_max_rel,
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
        cpu_grad_out_t,
        cpu_grad_in_t,
        cpu_state.params_view(),
        cpu_model_state,
        cpu_cache_t,
        cpu_grads,
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
        ctx,
        gpu_grad_in_t,
        gpu_grad_out_t,
        gpu.params_view(),
        gpu_model_state,
        gpu_cache_t,
        gpu_grads,
        workspace,
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
                        "    GRAD_P[",
                        i,
                        "]: cpu=",
                        cpu_g,
                        "gpu=",
                        gpu_g,
                        "rel=",
                        rel,
                    )

        if gp_fail == 0:
            print(
                "  [PASS] grad_params: max_abs=",
                gp_max_abs,
                "max_rel=",
                gp_max_rel,
            )
        else:
            print(
                "  [FAIL] grad_params:",
                gp_fail,
                "/",
                PS,
                "max_abs=",
                gp_max_abs,
                "max_rel=",
                gp_max_rel,
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
                    "    GRAD_IN[",
                    i,
                    "]: cpu=",
                    cpu_g,
                    "gpu=",
                    gpu_g,
                    "rel=",
                    rel,
                )

    if gi_fail == 0:
        print(
            "  [PASS] grad_input: max_abs=",
            gi_max_abs,
            "max_rel=",
            gi_max_rel,
        )
    else:
        print(
            "  [FAIL] grad_input:",
            gi_fail,
            "/",
            BS * IN,
            "max_abs=",
            gi_max_abs,
            "max_rel=",
            gi_max_rel,
        )

    print()


def main() raises:
    print("=== MuZero CPU vs GPU Network Parity ===")
    print()
    print(
        "Stage 1 diagnostic — verifies each MuZero network has bit-faithful"
    )
    print(
        "forward + backward across CPU/GPU. Uses the exact model types from"
    )
    print(
        "MuZeroMLPConfig (LinearMish, Linear, MinMaxNorm, Parallel, Sequential)."
    )
    print()

    var ctx = DeviceContext()

    # CartPole config dims (matches examples/cartpole/cartpole_muzero_gpu.mojo)
    comptime OBS = 4
    comptime ACT = 2
    comptime LATENT = 128
    comptime HIDDEN = 128
    comptime BINS = 51
    comptime DYN_IN = LATENT + ACT
    comptime DYN_OUT = LATENT + BINS
    comptime PRED_OUT = ACT + BINS

    print("--- Bisect: individual layer types ---")
    cpu_vs_gpu_check[LinearMish[OBS, HIDDEN]](
        ctx, "LinearMish[4,128] (rep first layer)"
    )
    cpu_vs_gpu_check[LinearMish[HIDDEN, HIDDEN]](
        ctx, "LinearMish[128,128] (rep/dyn middle layer)"
    )
    cpu_vs_gpu_check[Linear[HIDDEN, LATENT]](
        ctx, "Linear[128,128] (rep final / pre-MinMaxNorm)"
    )
    cpu_vs_gpu_check[Linear[HIDDEN, ACT]](
        ctx, "Linear[128,2] (pred policy head)"
    )
    cpu_vs_gpu_check[Linear[HIDDEN, BINS]](
        ctx, "Linear[128,51] (pred value head / dyn reward head)"
    )
    cpu_vs_gpu_check[MinMaxNorm[LATENT]](
        ctx, "MinMaxNorm[128] (rep + dyn output norm)"
    )

    print("--- Composites: Parallel and Sequential heads ---")
    # Pred head structure: Parallel[Linear[H,ACT], Linear[H,BINS]]
    cpu_vs_gpu_check[Parallel[Linear[HIDDEN, ACT], Linear[HIDDEN, BINS]]](
        ctx,
        "Parallel[Linear[128,2], Linear[128,51]] (pred Parallel head)",
    )
    # Dyn head structure: Parallel[Sequential[Linear[H,LATENT], MinMaxNorm[LATENT]], Linear[H,BINS]]
    cpu_vs_gpu_check[
        Parallel[
            Sequential[Linear[HIDDEN, LATENT], MinMaxNorm[LATENT]],
            Linear[HIDDEN, BINS],
        ]
    ](
        ctx,
        "Parallel[Sequential[Linear+MinMaxNorm], Linear] (dyn Parallel head)",
    )

    print("--- Full MuZero networks (as declared in MuZeroMLPConfig) ---")

    # RepModel: 3 LinearMish-ish layers then MinMaxNorm
    comptime RepModel = Sequential[
        LinearMish[OBS, HIDDEN],
        LinearMish[HIDDEN, HIDDEN],
        Linear[HIDDEN, LATENT],
        MinMaxNorm[LATENT],
    ]
    cpu_vs_gpu_check[RepModel](ctx, "Full RepModel")

    # DynModel
    comptime DynModel = Sequential[
        LinearMish[DYN_IN, HIDDEN],
        LinearMish[HIDDEN, HIDDEN],
        Parallel[
            Sequential[Linear[HIDDEN, LATENT], MinMaxNorm[LATENT]],
            Linear[HIDDEN, BINS],
        ],
    ]
    cpu_vs_gpu_check[DynModel](ctx, "Full DynModel")

    # PredModel
    comptime PredModel = Sequential[
        LinearMish[LATENT, HIDDEN],
        Parallel[
            Linear[HIDDEN, ACT],
            Linear[HIDDEN, BINS],
        ],
    ]
    cpu_vs_gpu_check[PredModel](ctx, "Full PredModel")

    print("=== Done ===")
    print()
    print(
        "Interpretation: if every check above is [PASS], the kernels are"
        " bit-faithful"
    )
    print(
        "and the GPU MuZero convergence bug is in ``update_gpu`` itself —"
        " specifically"
    )
    print(
        "in the CE gradient kernels, K-step unroll wiring, 0.5/1/K scaling,"
        " or"
    )
    print("the two-hot encoding / scalar transform. Stage 2 would then be a")
    print("test that hand-builds a batch and steps update_gpu vs update.")

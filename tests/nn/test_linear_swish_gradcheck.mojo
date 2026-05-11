"""LinearSwish-focused gradcheck — diagnoses MBPO dynamics-ensemble NaN bug.

Why this test exists:
    MBPO HalfCheetah's dyn_holdout_loss is locked at 1e10 (the sentinel value
    of `var best_holdout = Float64(1e10)` in mbpo_agent.mojo:1189). The
    early-stopping comparison `if holdout_loss < best_holdout` returns False
    for NaN under IEEE 754, so the loss is silently NaN.

    `DefaultMBPOConfig.DynamicsModel` is a stack of LinearSwish layers, and
    LinearSwish is not covered by any existing test in tests/nn/. The recent
    state-buffer refactor (commits 75521e5, bc7db5d, ed1f633) added a new
    `state` argument to every layer's forward/backward, plus c5809b5 fixed
    a workspace under-allocation bug — any of those could have left the
    Swish forward/backward path broken.

This test exercises three things at the EXACT dimensions the dynamics
ensemble uses (DYN_IN=23, DYN_HIDDEN=200, DYN_OUT=36, BATCH=256), so a
green run rules out LinearSwish itself as the NaN source:

  1. CPU finite-difference gradcheck on a single LinearSwish layer.
     Catches: any analytical-vs-numerical mismatch in CPU forward/backward.

  2. CPU↔GPU consistency: forward output, param gradients, input gradients.
     Catches: GPU kernel divergence from CPU reference (the most likely
     class of bug after the recent refactor).

  3. Full 5-layer dynamics MLP (4×LinearSwish + Linear), CPU↔GPU consistency.
     Catches: bugs that only show up through Sequential composition,
     workspace partitioning, cache offsets, or AutoFused dispatch.

Each test also explicitly flags NaN/Inf in the GPU output — those are the
failure mode actually observed in MBPO training, so we want to surface
them clearly even when relative error is uninformative.

Usage:
    pixi run -e nvidia mojo run -I . tests/nn/test_linear_swish_gradcheck.mojo
    pixi run -e apple  mojo run -I . tests/nn/test_linear_swish_gradcheck.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import abs as math_abs
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
    LinearSwish,
)


# =============================================================================
# Helpers
# =============================================================================


def _is_nan(x: Float64) -> Bool:
    return x != x


def _is_inf(x: Float64) -> Bool:
    # +inf or -inf
    return (not _is_nan(x)) and (x * 0.0 != 0.0)


def _count_bad(
    ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    mut nans: Int,
    mut infs: Int,
):
    """Count NaNs and Infs over n elements; updates nans/infs in place."""
    nans = 0
    infs = 0
    for i in range(n):
        var v = Float64(ptr[i])
        if _is_nan(v):
            nans += 1
        elif _is_inf(v):
            infs += 1


def _print_first_few(
    label: String,
    ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    k: Int = 8,
):
    print("    ", label, "(first", k, "of", n, "):")
    var lim = k if k < n else n
    for i in range(lim):
        print("      [", i, "] =", Float64(ptr[i]))


# =============================================================================
# CPU finite-difference gradcheck on a single LinearSwish layer
# =============================================================================


def cpu_gradcheck_linear_swish[
    IN: Int, OUT: Int, BS: Int = 4
](
    name: String,
    eps: Float64 = 1e-3,
    tol: Float64 = 0.05,
    max_params: Int = 200,
    max_inputs: Int = 100,
    min_denom: Float64 = 2e-3,
) raises -> Int:
    """CPU finite-diff gradcheck on a single LinearSwish[IN, OUT].

    Returns number of failures (0 = pass). Same finite-diff pattern as
    tests/nn/test_layer_gradcheck.mojo: f(p+eps) - f(p-eps) / (2*eps),
    then dot with grad_output and compare to analytical grad accumulated
    by backward.
    """
    comptime M = LinearSwish[IN, OUT]
    comptime PS = M.PARAM_SIZE
    comptime CS = M.CACHE_SIZE

    print("CPU FD gradcheck:", name, "(IN=", IN, "OUT=", OUT, "PS=", PS, ")")

    var state = NetworkState[M, Adam[]]()
    state.initialize[Xavier[]]()

    var input_ptr = alloc[Scalar[dtype]](BS * IN)
    var grad_out_ptr = alloc[Scalar[dtype]](BS * OUT)
    var bwd_grad_out_ptr = alloc[Scalar[dtype]](BS * OUT)
    var output_ptr = alloc[Scalar[dtype]](BS * OUT)
    var cache_ptr = alloc[Scalar[dtype]](BS * CS if CS > 0 else 1)
    var grad_in_ptr = alloc[Scalar[dtype]](BS * IN)
    var out_plus_ptr = alloc[Scalar[dtype]](BS * OUT)
    var out_minus_ptr = alloc[Scalar[dtype]](BS * OUT)
    var fd_cache_ptr = alloc[Scalar[dtype]](BS * CS if CS > 0 else 1)

    # Mix positive and negative inputs so Swish is exercised in both regimes
    # (sigmoid saturates one-sided, which would hide a sign error otherwise).
    for i in range(BS * IN):
        var v = -0.6 + Float64(i % 13) / 13.0 * 1.2
        (input_ptr + i)[] = Scalar[dtype](v)

    for i in range(BS * OUT):
        (grad_out_ptr + i)[] = Scalar[dtype](
            0.5 + Float64(i % 7) / 14.0 - Float64(i % 3) / 6.0
        )

    var input_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](input_ptr)
    var output_t = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](output_ptr)
    var cache_t = LayoutTensor[dtype, Layout.row_major(BS, CS), MutAnyOrigin](cache_ptr)
    var bwd_grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BS, OUT), MutAnyOrigin
    ](bwd_grad_out_ptr)
    var grad_in_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](grad_in_ptr)
    var out_plus_t = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](out_plus_ptr)
    var out_minus_t = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](out_minus_ptr)
    var fd_cache_t = LayoutTensor[dtype, Layout.row_major(BS, CS), MutAnyOrigin](fd_cache_ptr)

    var ms = state.model_state_view()
    M.forward[BS](input_t, output_t, state.params_view(), ms, cache_t)

    # NaN/Inf sanity on forward output before doing any FD comparison.
    var fwd_nans = 0
    var fwd_infs = 0
    _count_bad(output_ptr, BS * OUT, fwd_nans, fwd_infs)
    if fwd_nans > 0 or fwd_infs > 0:
        print(
            "  [FAIL] forward output has",
            fwd_nans,
            "NaN /",
            fwd_infs,
            "Inf — gradcheck aborted",
        )
        _print_first_few("output", output_ptr, BS * OUT)
        return 1

    state.zero_grads()
    memset(grad_in_ptr, 0, BS * IN)
    var grads = state.grads_view()
    for i in range(BS * OUT):
        (bwd_grad_out_ptr + i)[] = (grad_out_ptr + i)[]
    M.backward[BS](
        bwd_grad_out_t, grad_in_t, state.params_view(), ms, cache_t, grads
    )

    # Param FD check
    var p_fail = 0
    var p_max_rel: Float64 = 0.0
    var p_checked = 0
    if PS > 0:
        var p_step = PS // max_params
        if p_step < 1:
            p_step = 1
        for p_idx in range(0, PS, p_step):
            var orig = (state.params + p_idx)[]
            (state.params + p_idx)[] = Scalar[dtype](Float64(orig) + eps)
            M.forward[BS](input_t, out_plus_t, state.params_view(), ms, fd_cache_t)
            (state.params + p_idx)[] = Scalar[dtype](Float64(orig) - eps)
            M.forward[BS](input_t, out_minus_t, state.params_view(), ms, fd_cache_t)
            (state.params + p_idx)[] = orig

            var num_grad: Float64 = 0.0
            for j in range(BS * OUT):
                num_grad += (
                    Float64((grad_out_ptr + j)[])
                    * (Float64((out_plus_ptr + j)[]) - Float64((out_minus_ptr + j)[]))
                    / (2.0 * eps)
                )
            var ana_grad = Float64((state.grads + p_idx)[])
            var err = math_abs(ana_grad - num_grad)
            var denom = math_abs(ana_grad) + math_abs(num_grad)
            var rel: Float64 = 0.0
            if denom > 1e-5:
                rel = err / denom
            if rel > p_max_rel:
                p_max_rel = rel
            if rel > tol and denom > min_denom:
                p_fail += 1
                if p_fail <= 3:
                    print(
                        "    PARAM p[", p_idx,
                        "]: ana=", ana_grad, "num=", num_grad, "rel=", rel,
                    )
            p_checked += 1

    # Input FD check
    var i_fail = 0
    var i_max_rel: Float64 = 0.0
    var i_checked = 0
    var i_total = BS * IN
    var i_step = i_total // max_inputs
    if i_step < 1:
        i_step = 1
    for i_idx in range(0, i_total, i_step):
        var orig = (input_ptr + i_idx)[]
        (input_ptr + i_idx)[] = Scalar[dtype](Float64(orig) + eps)
        M.forward[BS](input_t, out_plus_t, state.params_view(), ms, fd_cache_t)
        (input_ptr + i_idx)[] = Scalar[dtype](Float64(orig) - eps)
        M.forward[BS](input_t, out_minus_t, state.params_view(), ms, fd_cache_t)
        (input_ptr + i_idx)[] = orig

        var num_grad: Float64 = 0.0
        for j in range(BS * OUT):
            num_grad += (
                Float64((grad_out_ptr + j)[])
                * (Float64((out_plus_ptr + j)[]) - Float64((out_minus_ptr + j)[]))
                / (2.0 * eps)
            )
        var ana_grad = Float64((grad_in_ptr + i_idx)[])
        var err = math_abs(ana_grad - num_grad)
        var denom = math_abs(ana_grad) + math_abs(num_grad)
        var rel: Float64 = 0.0
        if denom > 1e-5:
            rel = err / denom
        if rel > i_max_rel:
            i_max_rel = rel
        if rel > tol and denom > 1e-4:
            i_fail += 1
            if i_fail <= 3:
                print(
                    "    INPUT i[", i_idx,
                    "]: ana=", ana_grad, "num=", num_grad, "rel=", rel,
                )
        i_checked += 1

    var fails = p_fail + i_fail
    if PS > 0:
        if p_fail == 0:
            print("  [PASS] params: max_rel=", p_max_rel, "(", p_checked, "checked)")
        else:
            print("  [FAIL] params:", p_fail, "/", p_checked, "max_rel=", p_max_rel)
    if i_fail == 0:
        print("  [PASS] inputs: max_rel=", i_max_rel, "(", i_checked, "checked)")
    else:
        print("  [FAIL] inputs:", i_fail, "/", i_checked, "max_rel=", i_max_rel)
    print()
    return fails


# =============================================================================
# CPU vs GPU consistency for any Model (LinearSwish or Sequential thereof)
# =============================================================================


def cpu_vs_gpu_check[M: Model, BS: Int = 4](
    ctx: DeviceContext,
    name: String,
    fwd_tol: Float64 = 1e-3,
    bwd_tol: Float64 = 5e-3,
    min_denom: Float64 = 1e-4,
) raises -> Int:
    """Compare CPU vs GPU forward, param grads, and input grads.

    Returns number of failed checks (0 = pass). Reports NaN/Inf counts
    explicitly so the actual MBPO failure mode (Swish forward producing NaN)
    is impossible to overlook.
    """
    comptime IN = M.IN_DIM
    comptime OUT = M.OUT_DIM
    comptime PS = M.PARAM_SIZE
    comptime CS = M.CACHE_SIZE
    comptime WS = M.WORKSPACE_SIZE_PER_SAMPLE

    print(
        "CPU vs GPU:", name,
        "(IN=", IN, "OUT=", OUT, "PS=", PS, "CS=", CS, "WS/s=", WS, "BS=", BS, ")",
    )

    var fails = 0

    # CPU init
    var cpu_state = NetworkState[M, Adam[]]()
    cpu_state.initialize[Xavier[]]()

    # GPU mirror of the same params
    var gpu = GPUNetworkState[M, Adam[], dtype](ctx)
    gpu.upload_from(cpu_state, ctx)

    # Deterministic input — mix signs to exercise Swish on both regimes.
    var input_ptr = alloc[Scalar[dtype]](BS * IN)
    for i in range(BS * IN):
        var v = -0.6 + Float64(i % 13) / 13.0 * 1.2
        (input_ptr + i)[] = Scalar[dtype](v)

    # CPU forward
    var cpu_output_ptr = alloc[Scalar[dtype]](BS * OUT)
    var cpu_cache_ptr = alloc[Scalar[dtype]](BS * CS if CS > 0 else 1)
    var cpu_input_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](input_ptr)
    var cpu_output_t = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](cpu_output_ptr)
    var cpu_cache_t = LayoutTensor[dtype, Layout.row_major(BS, CS), MutAnyOrigin](cpu_cache_ptr)
    var cpu_ms = cpu_state.model_state_view()
    M.forward[BS](cpu_input_t, cpu_output_t, cpu_state.params_view(), cpu_ms, cpu_cache_t)

    var cpu_nans = 0
    var cpu_infs = 0
    _count_bad(cpu_output_ptr, BS * OUT, cpu_nans, cpu_infs)
    if cpu_nans > 0 or cpu_infs > 0:
        print("  [FAIL] CPU forward NaN=", cpu_nans, " Inf=", cpu_infs)
        _print_first_few("cpu_output", cpu_output_ptr, BS * OUT)
        fails += 1

    # GPU forward
    var gpu_input_host = ctx.enqueue_create_host_buffer[dtype](BS * IN)
    for i in range(BS * IN):
        gpu_input_host[i] = (input_ptr + i)[]
    var gpu_input_buf = ctx.enqueue_create_buffer[dtype](BS * IN)
    ctx.enqueue_copy(gpu_input_buf, gpu_input_host)

    var gpu_output_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    var gpu_cache_buf = ctx.enqueue_create_buffer[dtype](BS * CS if CS > 0 else 1)
    var workspace = ctx.enqueue_create_buffer[dtype](BS * WS if WS > 0 else 1)

    var gpu_input_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](
        gpu_input_buf.unsafe_ptr()
    )
    var gpu_output_t = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](
        gpu_output_buf.unsafe_ptr()
    )
    var gpu_cache_t = LayoutTensor[dtype, Layout.row_major(BS, CS), MutAnyOrigin](
        gpu_cache_buf.unsafe_ptr()
    )
    var gpu_ms = gpu.model_state_view()
    M.forward_gpu[BS](
        ctx,
        gpu_output_t,
        gpu_input_t,
        gpu.params_view(),
        gpu_ms,
        gpu_cache_t,
        workspace,
    )

    var gpu_output_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
    ctx.enqueue_copy(gpu_output_host, gpu_output_buf)
    ctx.synchronize()

    # GPU forward NaN/Inf (the MBPO failure mode)
    var gpu_nans = 0
    var gpu_infs = 0
    for i in range(BS * OUT):
        var v = Float64(gpu_output_host[i])
        if _is_nan(v):
            gpu_nans += 1
        elif _is_inf(v):
            gpu_infs += 1
    if gpu_nans > 0 or gpu_infs > 0:
        print("  [FAIL] GPU forward NaN=", gpu_nans, " Inf=", gpu_infs)
        var lim = 8 if (BS * OUT) > 8 else (BS * OUT)
        for i in range(lim):
            print("      gpu_output[", i, "] =", Float64(gpu_output_host[i]))
        fails += 1

    # Forward CPU vs GPU
    var fwd_max_abs: Float64 = 0.0
    var fwd_max_rel: Float64 = 0.0
    var fwd_fail = 0
    for i in range(BS * OUT):
        var c = Float64((cpu_output_ptr + i)[])
        var g = Float64(gpu_output_host[i])
        if _is_nan(c) or _is_nan(g):
            continue  # already reported above
        var err = math_abs(c - g)
        var denom = math_abs(c) + math_abs(g)
        var rel: Float64 = 0.0
        if denom > 1e-7:
            rel = err / denom
        if err > fwd_max_abs:
            fwd_max_abs = err
        if rel > fwd_max_rel:
            fwd_max_rel = rel
        if rel > fwd_tol and denom > min_denom:
            fwd_fail += 1
            if fwd_fail <= 3:
                print("    FWD[", i, "]: cpu=", c, "gpu=", g, "rel=", rel)

    if fwd_fail == 0 and gpu_nans == 0 and gpu_infs == 0 and cpu_nans == 0 and cpu_infs == 0:
        print("  [PASS] forward: max_abs=", fwd_max_abs, "max_rel=", fwd_max_rel)
    else:
        print(
            "  [FAIL] forward:", fwd_fail, "/", BS * OUT,
            "max_abs=", fwd_max_abs, "max_rel=", fwd_max_rel,
        )
        fails += 1

    # ── grad_output (same for both sides) ────────────────────
    var grad_out_ptr = alloc[Scalar[dtype]](BS * OUT)
    for i in range(BS * OUT):
        (grad_out_ptr + i)[] = Scalar[dtype](
            0.5 + Float64(i % 7) / 14.0 - Float64(i % 3) / 6.0
        )

    # CPU backward
    cpu_state.zero_grads()
    var cpu_grad_in_ptr = alloc[Scalar[dtype]](BS * IN)
    memset(cpu_grad_in_ptr, 0, BS * IN)
    var cpu_bwd_go_ptr = alloc[Scalar[dtype]](BS * OUT)
    for i in range(BS * OUT):
        (cpu_bwd_go_ptr + i)[] = (grad_out_ptr + i)[]
    var cpu_grad_out_t = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](cpu_bwd_go_ptr)
    var cpu_grad_in_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](cpu_grad_in_ptr)
    var cpu_grads = cpu_state.grads_view()
    M.backward[BS](
        cpu_grad_out_t, cpu_grad_in_t, cpu_state.params_view(), cpu_ms, cpu_cache_t, cpu_grads
    )

    # GPU backward
    gpu.zero_grads(ctx)
    var gpu_go_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
    for i in range(BS * OUT):
        gpu_go_host[i] = (grad_out_ptr + i)[]
    var gpu_go_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    ctx.enqueue_copy(gpu_go_buf, gpu_go_host)

    var gpu_gi_buf = ctx.enqueue_create_buffer[dtype](BS * IN)
    ctx.enqueue_memset(gpu_gi_buf, 0)

    var gpu_go_t = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](
        gpu_go_buf.unsafe_ptr()
    )
    var gpu_gi_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](
        gpu_gi_buf.unsafe_ptr()
    )
    var gpu_grads = gpu.grads_view()
    M.backward_gpu[BS](
        ctx,
        gpu_gi_t,
        gpu_go_t,
        gpu.params_view(),
        gpu_ms,
        gpu_cache_t,
        gpu_grads,
        workspace,
    )

    var gpu_grads_host = ctx.enqueue_create_host_buffer[dtype](PS if PS > 0 else 1)
    if PS > 0:
        ctx.enqueue_copy(gpu_grads_host, gpu.grads_buf)
    var gpu_gi_host = ctx.enqueue_create_host_buffer[dtype](BS * IN)
    ctx.enqueue_copy(gpu_gi_host, gpu_gi_buf)
    ctx.synchronize()

    # NaN/Inf in GPU grads
    if PS > 0:
        var gp_nans = 0
        var gp_infs = 0
        for i in range(PS):
            var v = Float64(gpu_grads_host[i])
            if _is_nan(v):
                gp_nans += 1
            elif _is_inf(v):
                gp_infs += 1
        if gp_nans > 0 or gp_infs > 0:
            print("  [FAIL] GPU grad_params NaN=", gp_nans, " Inf=", gp_infs)
            fails += 1

    var gi_nans = 0
    var gi_infs = 0
    for i in range(BS * IN):
        var v = Float64(gpu_gi_host[i])
        if _is_nan(v):
            gi_nans += 1
        elif _is_inf(v):
            gi_infs += 1
    if gi_nans > 0 or gi_infs > 0:
        print("  [FAIL] GPU grad_input NaN=", gi_nans, " Inf=", gi_infs)
        fails += 1

    # Param grad CPU vs GPU
    if PS > 0:
        var gp_max_abs: Float64 = 0.0
        var gp_max_rel: Float64 = 0.0
        var gp_fail = 0
        for i in range(PS):
            var c = Float64((cpu_state.grads + i)[])
            var g = Float64(gpu_grads_host[i])
            if _is_nan(c) or _is_nan(g):
                continue
            var err = math_abs(c - g)
            var denom = math_abs(c) + math_abs(g)
            var rel: Float64 = 0.0
            if denom > 1e-7:
                rel = err / denom
            if err > gp_max_abs:
                gp_max_abs = err
            if rel > gp_max_rel:
                gp_max_rel = rel
            if rel > bwd_tol and denom > min_denom:
                gp_fail += 1
                if gp_fail <= 3:
                    print("    GRAD_P[", i, "]: cpu=", c, "gpu=", g, "rel=", rel)
        if gp_fail == 0:
            print("  [PASS] grad_params: max_abs=", gp_max_abs, "max_rel=", gp_max_rel)
        else:
            print(
                "  [FAIL] grad_params:", gp_fail, "/", PS,
                "max_abs=", gp_max_abs, "max_rel=", gp_max_rel,
            )
            fails += 1

    # Grad input CPU vs GPU
    var gi_max_abs: Float64 = 0.0
    var gi_max_rel: Float64 = 0.0
    var gi_fail = 0
    for i in range(BS * IN):
        var c = Float64((cpu_grad_in_ptr + i)[])
        var g = Float64(gpu_gi_host[i])
        if _is_nan(c) or _is_nan(g):
            continue
        var err = math_abs(c - g)
        var denom = math_abs(c) + math_abs(g)
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
                print("    GRAD_IN[", i, "]: cpu=", c, "gpu=", g, "rel=", rel)
    if gi_fail == 0:
        print("  [PASS] grad_input: max_abs=", gi_max_abs, "max_rel=", gi_max_rel)
    else:
        print(
            "  [FAIL] grad_input:", gi_fail, "/", BS * IN,
            "max_abs=", gi_max_abs, "max_rel=", gi_max_rel,
        )
        fails += 1

    print()
    return fails


# =============================================================================
# main
# =============================================================================


def main() raises:
    print("=" * 70)
    print("LinearSwish gradcheck — MBPO dynamics-ensemble NaN diagnosis")
    print("=" * 70)
    print()

    var ctx = DeviceContext()
    var total_fail = 0

    # ── Step 1: CPU finite-diff gradcheck ───────────────────────
    # If any of these fail, the analytical CPU backward for Swish is broken
    # — bug is in SwishOp.vjp or the AutoFused 3-op CPU path.
    print("--- Step 1: CPU finite-difference gradcheck ---")
    print()
    total_fail += cpu_gradcheck_linear_swish[8, 4, BS=4]("LinearSwish[8,4]  small")
    total_fail += cpu_gradcheck_linear_swish[16, 8, BS=4]("LinearSwish[16,8] small")
    # MBPO dynamics first layer: DYN_IN=23, DYN_HIDDEN=200
    total_fail += cpu_gradcheck_linear_swish[23, 200, BS=4]("LinearSwish[23,200]  MBPO L0")
    # MBPO dynamics middle layer: 200 -> 200
    total_fail += cpu_gradcheck_linear_swish[200, 200, BS=4]("LinearSwish[200,200] MBPO L1-3")

    # ── Step 2: CPU vs GPU on a single LinearSwish layer ────────
    # If CPU gradcheck passed but this fails, the bug is in the GPU path
    # (FusedMatMulBiasActivation eval_kernel_mma / eval_kernel_2x2 or vjp_gpu).
    print("--- Step 2: CPU vs GPU, single LinearSwish ---")
    print()
    total_fail += cpu_vs_gpu_check[LinearSwish[8, 4], BS=4](ctx, "LinearSwish[8,4]")
    total_fail += cpu_vs_gpu_check[LinearSwish[16, 8], BS=4](ctx, "LinearSwish[16,8]")
    # Match MBPO's DYN_HIDDEN=200 — note 200 is NOT a multiple of MMA_BLOCK_N=32,
    # so the MMA kernel runs ceil(200/32)=7 column-blocks with the last partial.
    total_fail += cpu_vs_gpu_check[LinearSwish[23, 200], BS=4](ctx, "LinearSwish[23,200]  MBPO L0, BS=4")
    total_fail += cpu_vs_gpu_check[LinearSwish[200, 200], BS=4](ctx, "LinearSwish[200,200] MBPO L1, BS=4")
    # Real MBPO train batch — exposes batch-tile-edge effects.
    total_fail += cpu_vs_gpu_check[LinearSwish[23, 200], BS=256](
        ctx, "LinearSwish[23,200]  MBPO L0, BS=256"
    )
    total_fail += cpu_vs_gpu_check[LinearSwish[200, 200], BS=256](
        ctx, "LinearSwish[200,200] MBPO L1, BS=256"
    )

    # ── Step 3: Full 5-layer dynamics MLP (Sequential) ──────────
    # If steps 1+2 passed but this fails, the bug is in Sequential's
    # workspace partitioning, cache offsets, or per-layer dispatch — i.e.
    # the c5809b5 workspace fix or the bc7db5d state-buffer wiring on
    # the parent container, not in the leaf Swish layer.
    print("--- Step 3: CPU vs GPU, full dynamics MLP (Sequential) ---")
    print()
    # Exact replica of DefaultMBPOConfig.DynamicsModel (HalfCheetah dims):
    # DYN_IN = obs(17) + act(6) = 23, DYN_HIDDEN = 200, DYN_OUT = 2*(1+17) = 36.
    comptime DynMLP = Sequential[
        LinearSwish[23, 200],
        LinearSwish[200, 200],
        LinearSwish[200, 200],
        LinearSwish[200, 200],
        Linear[200, 36],
    ]
    total_fail += cpu_vs_gpu_check[DynMLP, BS=4](ctx, "DynMLP (4xLinearSwish + Linear)  BS=4")
    total_fail += cpu_vs_gpu_check[DynMLP, BS=256](
        ctx, "DynMLP (4xLinearSwish + Linear)  BS=256 (MBPO train batch)"
    )

    # ── Summary ─────────────────────────────────────────────────
    print("=" * 70)
    if total_fail == 0:
        print("ALL TESTS PASSED — LinearSwish forward/backward is sound.")
        print("If MBPO still diverges, the NaN does NOT come from the Swish path.")
        print("Look elsewhere: dynamics input scaler, NLL kernel race condition")
        print("(kernels.mojo:3556), learnable logvar bounds Adam update, or")
        print("synthetic-rollout sampling.")
    else:
        print("FAILURES:", total_fail)
        print("The earliest failing step localizes the bug:")
        print("  Step 1 fails  -> SwishOp.vjp (CPU analytical backward)")
        print("  Step 2 fails  -> FusedMatMulBiasActivation GPU kernels (Swish path)")
        print("  Step 3 fails  -> Sequential composition / workspace / state wiring")
    print("=" * 70)

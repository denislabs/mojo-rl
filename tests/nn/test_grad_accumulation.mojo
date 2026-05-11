"""Multi-call backward gradient accumulation regression test.

Validates that calling `backward_gpu` multiple times into the same
`grad_params` buffer correctly accumulates gradients across calls, rather
than overwriting (the bug fixed in docs/AUTODIFF_GRAD_ACCUMULATION.md).

The bug: GPU vjp kernels aliased `dW`/`db` to slices of `grad_params` and
wrote `dW[r,c] = acc` (overwrite) instead of `dW[r,c] = dW[r,c] + acc`
(accumulate). When a single update calls backward more than once into the
same grad view (MuZero K-step unroll, DreamerV3 RSSM BPTT, TD-MPC2
world-model BPTT), only the LAST call's contribution survived. Earlier
calls were silently discarded.

Test design:
    G1  = grad_params after ONE backward call with grad_output=Y.
    G_N = grad_params after N backward calls (same Y, same cache, no zero
          between calls) into the same buffer.
    Assert G_N ≈ N * G1.

A bug-bearing layer (overwrite) would produce G_N ≈ G1 — caught and reported
with the bug-signature ratio so failures are diagnostic, not opaque.

Usage:
    pixi run -e apple mojo run -I . tests/nn/test_grad_accumulation.mojo
    pixi run -e nvidia mojo run -I . tests/nn/test_grad_accumulation.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import abs
from std.memory import alloc
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
    LinearSigmoid,
    LinearMish,
    LinearSwish,
    LayerNorm,
    NormedLinear,
    NoisyLinear,
    BatchNorm1D,
    BatchNorm2D,
    Conv2DReLU,
    Conv2DLayer,
    Conv2DMish,
    Conv2DBatchNormReLU,
    LinearBatchNormReLU,
    FlattenLayer,
    Residual,
    ResBlockConv2D,
)
from mojo_rl.nn.model.resblock_conv2d_bn import ResBlockConv2DBN


def multi_call_check[M: Model, BS: Int = 4, N_CALLS: Int = 3](
    ctx: DeviceContext,
    name: String,
    tol: Float64 = 5e-3,
) raises:
    """Assert N backward calls accumulate to N × the single-call grad.

    Reports max_abs / max_rel and (on failure) how many params match the
    bug-signature G_N ≈ G1 — i.e. overwrite, not accumulate.
    """
    comptime IN = M.IN_DIM
    comptime OUT = M.OUT_DIM
    comptime PS = M.PARAM_SIZE
    comptime CS = M.CACHE_SIZE
    comptime WS = M.WORKSPACE_SIZE_PER_SAMPLE

    print(
        "Accum:",
        name,
        "(IN=",
        IN,
        "OUT=",
        OUT,
        "PS=",
        PS,
        "N=",
        N_CALLS,
        ")",
    )

    if PS == 0:
        print("  [SKIP] PARAM_SIZE=0 (no params to accumulate)")
        print()
        return

    # ── Init params on CPU, upload to GPU ────────────────────
    var cpu_state = NetworkState[M, Adam[]]()
    cpu_state.initialize[Xavier[]]()
    var gpu = GPUNetworkState[M, Adam[], dtype](ctx)
    gpu.upload_from(cpu_state, ctx)

    # ── Deterministic input ──────────────────────────────────
    var input_host = ctx.enqueue_create_host_buffer[dtype](BS * IN)
    for i in range(BS * IN):
        input_host[i] = Scalar[dtype](
            0.1 + Float64(i % 13) / 13.0 * 0.8
        )
    var input_buf = ctx.enqueue_create_buffer[dtype](BS * IN)
    ctx.enqueue_copy(input_buf, input_host)

    var output_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    var cache_buf = ctx.enqueue_create_buffer[dtype](
        BS * CS if CS > 0 else 1
    )
    var workspace = ctx.enqueue_create_buffer[dtype](
        BS * WS if WS > 0 else 1
    )

    var input_t = LayoutTensor[
        dtype, Layout.row_major(BS, IN), MutAnyOrigin
    ](input_buf.unsafe_ptr())
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BS, OUT), MutAnyOrigin
    ](output_buf.unsafe_ptr())
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BS, CS), MutAnyOrigin
    ](cache_buf.unsafe_ptr())

    # ── Forward once: cache is read-only input to all backward calls ──
    var model_state = gpu.model_state_view()
    M.forward_gpu[BS](
        ctx,
        output_t,
        input_t,
        gpu.params_view(),
        model_state,
        cache_t,
        workspace,
    )

    # ── Build grad_output once on host. Re-uploaded before EVERY backward
    # call because some layers (e.g. NormedLinear) mutate grad_output as
    # scratch during the backward pass.
    var go_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
    for i in range(BS * OUT):
        go_host[i] = Scalar[dtype](
            0.5 + Float64(i % 7) / 14.0 - Float64(i % 3) / 6.0
        )
    var grad_out_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    var grad_in_buf = ctx.enqueue_create_buffer[dtype](BS * IN)

    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BS, OUT), MutAnyOrigin
    ](grad_out_buf.unsafe_ptr())
    var grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BS, IN), MutAnyOrigin
    ](grad_in_buf.unsafe_ptr())

    # ── Pass A: single backward call → snapshot G1 ───────────
    gpu.zero_grads(ctx)
    ctx.enqueue_copy(grad_out_buf, go_host)
    ctx.enqueue_memset(grad_in_buf, 0)
    var grads_view_a = gpu.grads_view()
    M.backward_gpu[BS](
        ctx,
        grad_in_t,
        grad_out_t,
        gpu.params_view(),
        model_state,
        cache_t,
        grads_view_a,
        workspace,
    )
    var g1_host = ctx.enqueue_create_host_buffer[dtype](PS)
    ctx.enqueue_copy(g1_host, gpu.grads_buf)
    ctx.synchronize()
    var g1_arr = alloc[Scalar[dtype]](PS)
    for i in range(PS):
        (g1_arr + i)[] = g1_host[i]

    # ── Pass B: N backward calls into the same buffer → snapshot G_N ──
    gpu.zero_grads(ctx)
    var grads_view_b = gpu.grads_view()
    for _ in range(N_CALLS):
        # Restore grad_output and zero grad_input every call. grad_params
        # is intentionally NOT zeroed — that's what we're testing.
        ctx.enqueue_copy(grad_out_buf, go_host)
        ctx.enqueue_memset(grad_in_buf, 0)
        M.backward_gpu[BS](
            ctx,
            grad_in_t,
            grad_out_t,
            gpu.params_view(),
            model_state,
            cache_t,
            grads_view_b,
            workspace,
        )
    var gn_host = ctx.enqueue_create_host_buffer[dtype](PS)
    ctx.enqueue_copy(gn_host, gpu.grads_buf)
    ctx.synchronize()

    # ── Compare: G_N must ≈ N * G1 ───────────────────────────
    var max_abs: Float64 = 0.0
    var max_rel: Float64 = 0.0
    var fail = 0
    var bug_sig_overwrite = 0  # G_N ≈ G1, not N*G1 → indicates the bug
    for i in range(PS):
        var g1 = Float64((g1_arr + i)[])
        var gn = Float64(gn_host[i])
        var expected = Float64(N_CALLS) * g1
        var err = abs(expected - gn)
        var denom = abs(expected) + abs(gn)
        var rel: Float64 = 0.0
        if denom > 1e-7:
            rel = err / denom
        if err > max_abs:
            max_abs = err
        if rel > max_rel:
            max_rel = rel
        if rel > tol and denom > 1e-6:
            fail += 1
            # Bug-signature: did G_N collapse to G1? (The classic overwrite
            # symptom — only the last call's contribution survived.)
            var bug_err = abs(g1 - gn)
            var bug_denom = abs(g1) + abs(gn)
            if bug_denom > 1e-7 and (bug_err / bug_denom) < 0.1:
                bug_sig_overwrite += 1
            if fail <= 3:
                print(
                    "    p[",
                    i,
                    "]: G1=",
                    g1,
                    "G_N=",
                    gn,
                    "expected=",
                    expected,
                    "rel=",
                    rel,
                )

    if fail == 0:
        print(
            "  [PASS] max_abs=",
            max_abs,
            "max_rel=",
            max_rel,
        )
    else:
        print(
            "  [FAIL]",
            fail,
            "/",
            PS,
            "max_abs=",
            max_abs,
            "max_rel=",
            max_rel,
            "(overwrite-signature G_N~=G1:",
            bug_sig_overwrite,
            ")",
        )

    g1_arr.free()
    print()


def main() raises:
    print("=== NN Multi-Call Backward Accumulation Regression ===")
    print()

    var ctx = DeviceContext()

    # ── Fused matmul+bias+activation (matmul_bias_act.mojo) ──
    # Already fixed for the MuZero audit. This pins the fix.
    print("--- Fused Linear+Activation (matmul_bias_act.mojo) ---")
    multi_call_check[Linear[8, 4]](ctx, "Linear[8,4]")
    multi_call_check[Linear[32, 16]](ctx, "Linear[32,16]")
    multi_call_check[LinearReLU[16, 8]](ctx, "LinearReLU[16,8]")
    multi_call_check[LinearTanh[8, 4]](ctx, "LinearTanh[8,4]")
    multi_call_check[LinearSigmoid[8, 4]](ctx, "LinearSigmoid[8,4]")
    multi_call_check[LinearMish[8, 4]](ctx, "LinearMish[8,4]")
    multi_call_check[LinearSwish[8, 4]](ctx, "LinearSwish[8,4]")

    # ── Conv2D variants (fused/conv2d_act.mojo + primitives/conv2d.mojo) ──
    # Phase 1 of the grad-accumulation rollout.
    print("--- Conv2D ---")
    multi_call_check[Conv2DReLU[2, 4, 3, 1, 1, 5, 5]](
        ctx, "Conv2DReLU[2,4,3x3,5x5]"
    )
    multi_call_check[Conv2DMish[2, 4, 3, 1, 1, 5, 5]](
        ctx, "Conv2DMish[2,4,3x3,5x5]"
    )
    multi_call_check[Conv2DLayer[2, 4, 3, 1, 1, 5, 5]](
        ctx, "Conv2DLayer[2,4,3x3,5x5]"
    )

    # ── Norm layers (primitives/layer_norm.mojo) ─────────────
    # NormedLinear has both LayerNorm γ/β and Linear W/b grads — exercises
    # both fused matmul AND norm grad-slice writes in one layer.
    print("--- Norm layers ---")
    multi_call_check[LayerNorm[16]](ctx, "LayerNorm[16]")
    multi_call_check[LayerNorm[64]](ctx, "LayerNorm[64]")
    multi_call_check[NormedLinear[8, 16]](ctx, "NormedLinear[8,16]")
    multi_call_check[NormedLinear[16, 8]](ctx, "NormedLinear[16,8]")

    # ── Combinators: composite layers must propagate accumulation ──
    # If any leaf-layer write is overwrite, a Sequential or Residual will
    # fail here even if the leaf passes its own row above. (Defensive.)
    print("--- Combinators ---")
    multi_call_check[Sequential[LinearReLU[8, 6], Linear[6, 4]]](
        ctx, "Sequential[LinearReLU[8,6], Linear[6,4]]"
    )
    multi_call_check[Sequential[LinearMish[8, 16], LinearMish[16, 8]]](
        ctx, "Sequential[LinearMish, LinearMish]"
    )
    multi_call_check[Residual[LinearReLU[8, 8]]](
        ctx, "Residual[LinearReLU[8,8]]"
    )
    multi_call_check[Parallel[Linear[8, 4], Linear[8, 4]]](
        ctx, "Parallel[Linear[8,4], Linear[8,4]]"
    )

    # ── BatchNorm + fused BN-act variants ────────────────────
    # Custom backward kernels write directly to grads.ptr[GAMMA_OFF/BETA_OFF].
    # Audit showed they already use += pattern, but verify empirically.
    print("--- BatchNorm + fused variants ---")
    multi_call_check[BatchNorm1D[16]](ctx, "BatchNorm1D[16]")
    multi_call_check[BatchNorm2D[4, 5, 5]](ctx, "BatchNorm2D[4ch,5x5]")
    multi_call_check[Conv2DBatchNormReLU[2, 4, 3, 1, 1, 5, 5]](
        ctx, "Conv2DBatchNormReLU[2,4,3x3,5x5]"
    )
    multi_call_check[LinearBatchNormReLU[16, 8]](
        ctx, "LinearBatchNormReLU[16,8]"
    )

    # ── NoisyLinear (DQN exploration layer) ──────────────────
    # 4 param slices: μ_W, σ_W, μ_b, σ_b. Custom kernel; audit looked safe.
    print("--- NoisyLinear ---")
    multi_call_check[NoisyLinear[8, 4]](ctx, "NoisyLinear[8,4]")
    multi_call_check[NoisyLinear[16, 8]](ctx, "NoisyLinear[16,8]")

    # ── ResBlocks (delegate to inner Conv2D — should propagate fix) ──
    print("--- ResBlocks ---")
    multi_call_check[ResBlockConv2D[4, 3, 1, 5, 5]](
        ctx, "ResBlockConv2D[4ch,3x3,5x5]"
    )
    multi_call_check[ResBlockConv2DBN[4, 3, 1, 5, 5]](
        ctx, "ResBlockConv2DBN[4ch,3x3,5x5]"
    )

    # ── Realistic BPTT-shaped composite (MuZero pred head pattern) ──
    # Conv trunk + flatten + dual heads — the same shape that surfaced
    # the original bug in MuZero K-step unroll.
    print("--- Realistic (MuZero-pred shape) ---")
    comptime ConvHead = Sequential[
        Conv2DReLU[2, 4, 3, 1, 1, 5, 5],
        FlattenLayer[4 * 5 * 5],
        LinearReLU[100, 16],
        Linear[16, 4],
    ]
    multi_call_check[ConvHead](ctx, "Conv2DReLU + Flatten + LinearReLU + Linear")

    # ── TDMPC2 trunk shape (NormedLinear stack) ──────────────
    # World-model BPTT is the original bug-driver for NormedLinear.
    print("--- Realistic (TDMPC2 trunk shape) ---")
    comptime TDMPC2Trunk = Sequential[
        NormedLinear[8, 16],
        NormedLinear[16, 16],
        NormedLinear[16, 8],
    ]
    multi_call_check[TDMPC2Trunk](ctx, "NormedLinear x3 (TDMPC2 trunk)")

    print("=== Done ===")

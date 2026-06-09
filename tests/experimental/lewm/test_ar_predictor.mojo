"""ARPredictor — Tokenwise AdaLN-zero stack validation.

Validates the stacked AdaLN-zero predictor pattern that LeWM's `ARPredictor`
uses (`references/le-wm-main/module.py:244-286`). Multiple residual blocks
share the same conditioning signal `c` and each contributes a small
correction to `x` via an AdaLN-gated branch.

One ARPredictor block here:

    raw_mod[k] = c @ W_adaLN[k] + b_adaLN[k]                     # (B, T, 3D)
    shift_k, scale_k, gate_k = chunk(raw_mod[k], 3)
    ln_x       = LayerNormNoAffine(x_{k-1})                       # per-token
    mod_x      = ln_x * (1 + scale_k) + shift_k                   # per-token
    inner_k    = mod_x @ W_inner[k] + b_inner[k]                  # per-token
    x_k        = x_{k-1} + gate_k * inner_k                        # per-token

with `N_BLOCKS` stacked. The SiLU on `c` from `references/.../module.py:99`
is moved INSIDE each block (so each block has its own `silu(c) → W_adaLN`
mini-MLP), matching the reference's `nn.Sequential(SiLU, Linear)` pattern
that lives inside every `ConditionalBlock`.

The inner module is a placeholder `Linear[D, D]` per branch — real
attention (`MultiHeadAttention[D, h, T, causal=True]`) or FFN swaps in by
replacing the inner `MatMul + BiasAdd` calls with `Sequential[QKV, SDPA,
out_proj]`. The AdaLN-zero wrapping is independent of the inner module's
shape (input `(B*T, D)` → output `(B*T, D)`).

Tests:
  1. **Zero-init identity**: with every block's `W_adaLN = b_adaLN = 0`,
     every block contributes zero (gate=0). Stack is exactly identity:
     output == x_input bitwise.
  2. **Backward gradcheck**: at non-zero adaLN init, finite-difference
     grads w.r.t. x_input and c_input match the analytical chain through
     all blocks within 1e-3.

The orchestration is callable directly from Phase 3 training code — no
Model-trait conformance commitment yet.

Toy config: B=2, T=3, D=4, N_BLOCKS=2.

Run:
    pixi run mojo run -I . tests/experimental/lewm/test_ar_predictor.mojo
"""

from std.math import abs
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.autodiff.primitives import (
    MatMul,
    BiasAdd,
    SwishOp,
    ModulateOp,
    GateOp,
    LayerNormNoAffineOp,
)


# =============================================================================
# Single-block forward (operates per-token, treats effective batch = B*T)
# =============================================================================
def block_forward[
    BT: Int, D: Int
](
    x_in: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    c_in: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    adaln_w: LayoutTensor[dtype, Layout.row_major(D * 3 * D), MutAnyOrigin],
    adaln_b: LayoutTensor[dtype, Layout.row_major(3 * D), MutAnyOrigin],
    inner_w: LayoutTensor[dtype, Layout.row_major(D * D), MutAnyOrigin],
    inner_b: LayoutTensor[dtype, Layout.row_major(D), MutAnyOrigin],
    # Outputs
    mut x_out: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    # Caches (per-token, BT samples)
    mut silu_cache: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut matmul_ad_cache: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut bias_ad_cache: LayoutTensor[dtype, Layout.row_major(BT, 0), MutAnyOrigin],
    mut ln_cache: LayoutTensor[dtype, Layout.row_major(BT, D + 1), MutAnyOrigin],
    mut mod_cache: LayoutTensor[dtype, Layout.row_major(BT, 2 * D), MutAnyOrigin],
    mut matmul_in_cache: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut bias_in_cache: LayoutTensor[dtype, Layout.row_major(BT, 0), MutAnyOrigin],
    mut gate_cache: LayoutTensor[dtype, Layout.row_major(BT, 2 * D), MutAnyOrigin],
    # Cached intermediates (used by backward)
    mut raw_mod: LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin],
    # Scratch (transient)
    mut silu_c: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut adaln_matmul_out: LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin],
    mut ln_out: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut mod_inp: LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin],
    mut mod_x: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut inner_matmul_out: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut inner_out: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut gate_inp: LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin],
) raises:
    var empty_params = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=Int(0)))

    SwishOp[D].eval[BT](c_in, silu_c, empty_params, silu_cache)
    MatMul[D, 3 * D].eval[BT](silu_c, adaln_matmul_out, adaln_w, matmul_ad_cache)
    BiasAdd[3 * D].eval[BT](adaln_matmul_out, raw_mod, adaln_b, bias_ad_cache)

    LayerNormNoAffineOp[D].eval[BT](x_in, ln_out, empty_params, ln_cache)

    # Pack [ln_x | scale | shift] for ModulateOp.
    # raw_mod chunked: shift=raw_mod[:, :D], scale=raw_mod[:, D:2D], gate=raw_mod[:, 2D:3D]
    for b in range(BT):
        for i in range(D):
            mod_inp[b, i] = ln_out[b, i]
            mod_inp[b, D + i] = raw_mod[b, D + i]            # scale
            mod_inp[b, 2 * D + i] = raw_mod[b, i]            # shift

    ModulateOp[D].eval[BT](mod_inp, mod_x, empty_params, mod_cache)
    MatMul[D, D].eval[BT](mod_x, inner_matmul_out, inner_w, matmul_in_cache)
    BiasAdd[D].eval[BT](inner_matmul_out, inner_out, inner_b, bias_in_cache)

    # Pack [x | gate | branch_out] for GateOp.
    for b in range(BT):
        for i in range(D):
            gate_inp[b, i] = x_in[b, i]
            gate_inp[b, D + i] = raw_mod[b, 2 * D + i]       # gate
            gate_inp[b, 2 * D + i] = inner_out[b, i]

    GateOp[D].eval[BT](gate_inp, x_out, empty_params, gate_cache)


# =============================================================================
# Single-block backward — adds c-grad contributions into a shared accumulator.
# =============================================================================
def block_backward[
    BT: Int, D: Int
](
    grad_x_out: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    # Params (read-only)
    adaln_w: LayoutTensor[dtype, Layout.row_major(D * 3 * D), MutAnyOrigin],
    adaln_b: LayoutTensor[dtype, Layout.row_major(3 * D), MutAnyOrigin],
    inner_w: LayoutTensor[dtype, Layout.row_major(D * D), MutAnyOrigin],
    inner_b: LayoutTensor[dtype, Layout.row_major(D), MutAnyOrigin],
    # Caches from forward
    silu_cache: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    matmul_ad_cache: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    bias_ad_cache: LayoutTensor[dtype, Layout.row_major(BT, 0), MutAnyOrigin],
    ln_cache: LayoutTensor[dtype, Layout.row_major(BT, D + 1), MutAnyOrigin],
    mod_cache: LayoutTensor[dtype, Layout.row_major(BT, 2 * D), MutAnyOrigin],
    matmul_in_cache: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    bias_in_cache: LayoutTensor[dtype, Layout.row_major(BT, 0), MutAnyOrigin],
    gate_cache: LayoutTensor[dtype, Layout.row_major(BT, 2 * D), MutAnyOrigin],
    # Outputs (grad_x_in, grad_c_in are ACCUMULATED across blocks; param-grads are owned by this block)
    mut grad_x_in: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut grad_c_in: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut grad_adaln_w: LayoutTensor[dtype, Layout.row_major(D * 3 * D), MutAnyOrigin],
    mut grad_adaln_b: LayoutTensor[dtype, Layout.row_major(3 * D), MutAnyOrigin],
    mut grad_inner_w: LayoutTensor[dtype, Layout.row_major(D * D), MutAnyOrigin],
    mut grad_inner_b: LayoutTensor[dtype, Layout.row_major(D), MutAnyOrigin],
    # Scratch (transient grad buffers — caller reuses across blocks)
    mut grad_gate_inp: LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin],
    mut grad_inner_out: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut grad_inner_matmul: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut grad_mod_x: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut grad_mod_inp: LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin],
    mut grad_ln_out: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut grad_ln_in: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut grad_raw_mod: LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin],
    mut grad_adaln_matmul: LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin],
    mut grad_silu_c: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut grad_c_this_block: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
) raises:
    var empty_params = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=Int(0)))
    var empty_grad_params = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=Int(0)))

    GateOp[D].vjp[BT](
        grad_x_out, grad_gate_inp, empty_params, gate_cache, empty_grad_params
    )

    # Split grad_gate_inp → grad_x_residual (accum into grad_x_in), grad_gate, grad_inner_out
    for b in range(BT):
        for i in range(D):
            grad_x_in[b, i] = grad_x_in[b, i] + grad_gate_inp[b, i]
            grad_raw_mod[b, 2 * D + i] = grad_gate_inp[b, D + i]      # gate
            grad_inner_out[b, i] = grad_gate_inp[b, 2 * D + i]

    BiasAdd[D].vjp[BT](
        grad_inner_out, grad_inner_matmul, inner_b, bias_in_cache, grad_inner_b
    )
    MatMul[D, D].vjp[BT](
        grad_inner_matmul, grad_mod_x, inner_w, matmul_in_cache, grad_inner_w
    )

    ModulateOp[D].vjp[BT](
        grad_mod_x, grad_mod_inp, empty_params, mod_cache, empty_grad_params
    )

    # Split grad_mod_inp → grad_ln_out, grad_scale, grad_shift
    for b in range(BT):
        for i in range(D):
            grad_ln_out[b, i] = grad_mod_inp[b, i]
            grad_raw_mod[b, D + i] = grad_mod_inp[b, D + i]           # scale
            grad_raw_mod[b, i] = grad_mod_inp[b, 2 * D + i]           # shift

    LayerNormNoAffineOp[D].vjp[BT](
        grad_ln_out, grad_ln_in, empty_params, ln_cache, empty_grad_params
    )

    # grad_x_in += grad_ln_in (gradient through LN path)
    for b in range(BT):
        for i in range(D):
            grad_x_in[b, i] = grad_x_in[b, i] + grad_ln_in[b, i]

    BiasAdd[3 * D].vjp[BT](
        grad_raw_mod, grad_adaln_matmul, adaln_b, bias_ad_cache, grad_adaln_b
    )
    MatMul[D, 3 * D].vjp[BT](
        grad_adaln_matmul, grad_silu_c, adaln_w, matmul_ad_cache, grad_adaln_w
    )
    SwishOp[D].vjp[BT](
        grad_silu_c, grad_c_this_block, empty_params, silu_cache, empty_grad_params
    )

    # Accumulate this block's grad_c into shared grad_c_in.
    for b in range(BT):
        for i in range(D):
            grad_c_in[b, i] = grad_c_in[b, i] + grad_c_this_block[b, i]


# =============================================================================
# Two-block AR-predictor stack — packs all the buffers and chains the blocks.
# Block params are stored back-to-back, indexed by `k`.
# =============================================================================
comptime BATCH = 2
comptime T = 3
comptime D = 4
comptime N_BLOCKS = 2
comptime BT = BATCH * T

comptime ADALN_W_PER = D * 3 * D
comptime ADALN_B_PER = 3 * D
comptime INNER_W_PER = D * D
comptime INNER_B_PER = D

# Per-block cache sizes (per token).
comptime _C_SILU = D
comptime _C_MMAD = D
comptime _C_BAD = 0
comptime _C_LN = D + 1
comptime _C_MOD = 2 * D
comptime _C_MMIN = D
comptime _C_BIN = 0
comptime _C_GATE = 2 * D
comptime _C_RAWMOD = 3 * D
# Sum needed per block (per token) — not strictly used as one cache; each op
# has its own slot. We allocate them as separate InlineArrays below.


def ar_predictor_forward(
    x_input_t: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    c_input_t: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut x_after_t: List[LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin]],
    mut adaln_w_ts: List[LayoutTensor[dtype, Layout.row_major(ADALN_W_PER), MutAnyOrigin]],
    mut adaln_b_ts: List[LayoutTensor[dtype, Layout.row_major(ADALN_B_PER), MutAnyOrigin]],
    mut inner_w_ts: List[LayoutTensor[dtype, Layout.row_major(INNER_W_PER), MutAnyOrigin]],
    mut inner_b_ts: List[LayoutTensor[dtype, Layout.row_major(INNER_B_PER), MutAnyOrigin]],
    mut silu_cs: List[LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin]],
    mut matmul_ad_cs: List[LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin]],
    mut bias_ad_cs: List[LayoutTensor[dtype, Layout.row_major(BT, 0), MutAnyOrigin]],
    mut ln_cs: List[LayoutTensor[dtype, Layout.row_major(BT, D + 1), MutAnyOrigin]],
    mut mod_cs: List[LayoutTensor[dtype, Layout.row_major(BT, 2 * D), MutAnyOrigin]],
    mut matmul_in_cs: List[LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin]],
    mut bias_in_cs: List[LayoutTensor[dtype, Layout.row_major(BT, 0), MutAnyOrigin]],
    mut gate_cs: List[LayoutTensor[dtype, Layout.row_major(BT, 2 * D), MutAnyOrigin]],
    mut raw_mods: List[LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin]],
    # Shared scratch (overwritten each block — backward doesn't need them since
    # the caches above are sufficient).
    mut silu_c_buf: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut adaln_matmul_buf: LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin],
    mut ln_buf: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut mod_inp_buf: LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin],
    mut mod_x_buf: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut inner_matmul_buf: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut inner_out_buf: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut gate_inp_buf: LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin],
) raises:
    # Block 0 uses x_input; subsequent blocks use x_after[k-1].
    for k in range(N_BLOCKS):
        var x_prev_t: LayoutTensor[
            dtype, Layout.row_major(BT, D), MutAnyOrigin
        ]
        if k == 0:
            x_prev_t = x_input_t
        else:
            x_prev_t = x_after_t[k - 1]

        block_forward[BT, D](
            x_prev_t, c_input_t,
            adaln_w_ts[k], adaln_b_ts[k], inner_w_ts[k], inner_b_ts[k],
            x_after_t[k],
            silu_cs[k], matmul_ad_cs[k], bias_ad_cs[k], ln_cs[k], mod_cs[k],
            matmul_in_cs[k], bias_in_cs[k], gate_cs[k],
            raw_mods[k],
            silu_c_buf, adaln_matmul_buf, ln_buf, mod_inp_buf, mod_x_buf,
            inner_matmul_buf, inner_out_buf, gate_inp_buf,
        )


# =============================================================================
# Backward — iterate blocks in reverse, accumulating grad_x and grad_c.
# =============================================================================
def ar_predictor_backward(
    grad_output_t: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut grad_x_input_t: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut grad_c_input_t: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    # Block params + their grads
    adaln_w_ts: List[LayoutTensor[dtype, Layout.row_major(ADALN_W_PER), MutAnyOrigin]],
    adaln_b_ts: List[LayoutTensor[dtype, Layout.row_major(ADALN_B_PER), MutAnyOrigin]],
    inner_w_ts: List[LayoutTensor[dtype, Layout.row_major(INNER_W_PER), MutAnyOrigin]],
    inner_b_ts: List[LayoutTensor[dtype, Layout.row_major(INNER_B_PER), MutAnyOrigin]],
    mut g_adaln_w_ts: List[LayoutTensor[dtype, Layout.row_major(ADALN_W_PER), MutAnyOrigin]],
    mut g_adaln_b_ts: List[LayoutTensor[dtype, Layout.row_major(ADALN_B_PER), MutAnyOrigin]],
    mut g_inner_w_ts: List[LayoutTensor[dtype, Layout.row_major(INNER_W_PER), MutAnyOrigin]],
    mut g_inner_b_ts: List[LayoutTensor[dtype, Layout.row_major(INNER_B_PER), MutAnyOrigin]],
    # Caches from forward
    silu_cs: List[LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin]],
    matmul_ad_cs: List[LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin]],
    bias_ad_cs: List[LayoutTensor[dtype, Layout.row_major(BT, 0), MutAnyOrigin]],
    ln_cs: List[LayoutTensor[dtype, Layout.row_major(BT, D + 1), MutAnyOrigin]],
    mod_cs: List[LayoutTensor[dtype, Layout.row_major(BT, 2 * D), MutAnyOrigin]],
    matmul_in_cs: List[LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin]],
    bias_in_cs: List[LayoutTensor[dtype, Layout.row_major(BT, 0), MutAnyOrigin]],
    gate_cs: List[LayoutTensor[dtype, Layout.row_major(BT, 2 * D), MutAnyOrigin]],
    # Per-block grad-x flowing back into the previous block.
    mut grad_x_flows: List[LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin]],
    # Reusable scratch (per-block, overwritten).
    mut s_grad_gate_inp: LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin],
    mut s_grad_inner_out: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut s_grad_inner_matmul: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut s_grad_mod_x: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut s_grad_mod_inp: LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin],
    mut s_grad_ln_out: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut s_grad_ln_in: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut s_grad_raw_mod: LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin],
    mut s_grad_adaln_matmul: LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin],
    mut s_grad_silu_c: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
    mut s_grad_c_block: LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin],
) raises:
    # grad_x_flows[N-1] is the grad of the final block's output = grad_output_t.
    # After backward through block k, grad_x_flows[k-1] is filled in.
    # grad_x_input_t = grad_x_flows[-1] at end (the grad flowing into block 0's x input).
    # grad_c_input_t accumulates contributions from every block.
    for b in range(BT):
        for i in range(D):
            grad_x_flows[N_BLOCKS - 1][b, i] = grad_output_t[b, i]
            grad_c_input_t[b, i] = Scalar[dtype](0)

    # Backward through blocks in reverse.
    for kk in range(N_BLOCKS):
        var k = N_BLOCKS - 1 - kk
        # Output grad for this block.
        var g_x_out_t = grad_x_flows[k]
        # Block 0's x-input grad goes into grad_x_input_t; otherwise into grad_x_flows[k-1].
        var g_x_in_t: LayoutTensor[
            dtype, Layout.row_major(BT, D), MutAnyOrigin
        ]
        if k == 0:
            g_x_in_t = grad_x_input_t
        else:
            g_x_in_t = grad_x_flows[k - 1]
        # Initialize destination to zero (block_backward accumulates).
        for b in range(BT):
            for i in range(D):
                g_x_in_t[b, i] = Scalar[dtype](0)

        block_backward[BT, D](
            g_x_out_t,
            adaln_w_ts[k], adaln_b_ts[k], inner_w_ts[k], inner_b_ts[k],
            silu_cs[k], matmul_ad_cs[k], bias_ad_cs[k], ln_cs[k], mod_cs[k],
            matmul_in_cs[k], bias_in_cs[k], gate_cs[k],
            g_x_in_t, grad_c_input_t,
            g_adaln_w_ts[k], g_adaln_b_ts[k],
            g_inner_w_ts[k], g_inner_b_ts[k],
            s_grad_gate_inp, s_grad_inner_out, s_grad_inner_matmul,
            s_grad_mod_x, s_grad_mod_inp, s_grad_ln_out, s_grad_ln_in,
            s_grad_raw_mod, s_grad_adaln_matmul, s_grad_silu_c,
            s_grad_c_block,
        )


# =============================================================================
# Helper: allocate buffers + run forward, returning everything needed for the
# tests. We re-instantiate per test to keep state contained.
# =============================================================================
def main() raises:
    print("=== ARPredictor — Tokenwise AdaLN-zero stack ===")
    print()
    print(
        "  BATCH=", BATCH,
        " T=", T,
        " D=", D,
        " N_BLOCKS=", N_BLOCKS,
        " effective batch (BT) =", BT,
    )

    # -------- Inputs --------
    var x_input = InlineArray[Scalar[dtype], BT * D](uninitialized=True)
    var c_input = InlineArray[Scalar[dtype], BT * D](uninitialized=True)
    for i in range(BT * D):
        x_input[i] = Scalar[dtype](0.23 * Float64(i % 7) - 0.4)
        c_input[i] = Scalar[dtype](0.17 * Float64(i % 11) + 0.15)

    var x_input_t = LayoutTensor[
        dtype, Layout.row_major(BT, D), MutAnyOrigin
    ](x_input.unsafe_ptr())
    var c_input_t = LayoutTensor[
        dtype, Layout.row_major(BT, D), MutAnyOrigin
    ](c_input.unsafe_ptr())

    # -------- Per-block params + caches + intermediates --------
    var adaln_ws = List[InlineArray[Scalar[dtype], ADALN_W_PER]]()
    var adaln_bs = List[InlineArray[Scalar[dtype], ADALN_B_PER]]()
    var inner_ws = List[InlineArray[Scalar[dtype], INNER_W_PER]]()
    var inner_bs = List[InlineArray[Scalar[dtype], INNER_B_PER]]()

    var x_after = List[InlineArray[Scalar[dtype], BT * D]]()
    var silu_c_caches = List[InlineArray[Scalar[dtype], BT * D]]()
    var mm_ad_caches = List[InlineArray[Scalar[dtype], BT * D]]()
    var bias_ad_caches = List[InlineArray[Scalar[dtype], BT * 1]]()
    var ln_caches = List[InlineArray[Scalar[dtype], BT * (D + 1)]]()
    var mod_caches = List[InlineArray[Scalar[dtype], BT * 2 * D]]()
    var mm_in_caches = List[InlineArray[Scalar[dtype], BT * D]]()
    var bias_in_caches = List[InlineArray[Scalar[dtype], BT * 1]]()
    var gate_caches = List[InlineArray[Scalar[dtype], BT * 2 * D]]()
    var raw_mods_arr = List[InlineArray[Scalar[dtype], BT * 3 * D]]()

    for k in range(N_BLOCKS):
        var aw = InlineArray[Scalar[dtype], ADALN_W_PER](uninitialized=True)
        var ab = InlineArray[Scalar[dtype], ADALN_B_PER](uninitialized=True)
        var iw = InlineArray[Scalar[dtype], INNER_W_PER](uninitialized=True)
        var ib = InlineArray[Scalar[dtype], INNER_B_PER](uninitialized=True)
        # Small, varied init across blocks.
        for i in range(ADALN_W_PER):
            aw[i] = Scalar[dtype](0.08 * Float64((i + 7 * k) % 13) - 0.32)
        for i in range(ADALN_B_PER):
            ab[i] = Scalar[dtype](0.05 * Float64((i + 3 * k) % 5) - 0.1)
        for i in range(INNER_W_PER):
            iw[i] = Scalar[dtype](0.18 * Float64((i + 5 * k) % 5) - 0.25)
        for i in range(INNER_B_PER):
            ib[i] = Scalar[dtype](0.06 + 0.01 * Float64(k))
        adaln_ws.append(aw^)
        adaln_bs.append(ab^)
        inner_ws.append(iw^)
        inner_bs.append(ib^)

        x_after.append(InlineArray[Scalar[dtype], BT * D](uninitialized=True))
        silu_c_caches.append(InlineArray[Scalar[dtype], BT * D](uninitialized=True))
        mm_ad_caches.append(InlineArray[Scalar[dtype], BT * D](uninitialized=True))
        bias_ad_caches.append(InlineArray[Scalar[dtype], BT * 1](uninitialized=True))
        ln_caches.append(InlineArray[Scalar[dtype], BT * (D + 1)](uninitialized=True))
        mod_caches.append(InlineArray[Scalar[dtype], BT * 2 * D](uninitialized=True))
        mm_in_caches.append(InlineArray[Scalar[dtype], BT * D](uninitialized=True))
        bias_in_caches.append(InlineArray[Scalar[dtype], BT * 1](uninitialized=True))
        gate_caches.append(InlineArray[Scalar[dtype], BT * 2 * D](uninitialized=True))
        raw_mods_arr.append(InlineArray[Scalar[dtype], BT * 3 * D](uninitialized=True))

    # Build LayoutTensor lists (separate from InlineArray lists; they share memory).
    var adaln_w_ts = List[LayoutTensor[dtype, Layout.row_major(ADALN_W_PER), MutAnyOrigin]]()
    var adaln_b_ts = List[LayoutTensor[dtype, Layout.row_major(ADALN_B_PER), MutAnyOrigin]]()
    var inner_w_ts = List[LayoutTensor[dtype, Layout.row_major(INNER_W_PER), MutAnyOrigin]]()
    var inner_b_ts = List[LayoutTensor[dtype, Layout.row_major(INNER_B_PER), MutAnyOrigin]]()
    var x_after_t = List[LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin]]()
    var silu_cs_t = List[LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin]]()
    var mm_ad_cs_t = List[LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin]]()
    var bias_ad_cs_t = List[LayoutTensor[dtype, Layout.row_major(BT, 0), MutAnyOrigin]]()
    var ln_cs_t = List[LayoutTensor[dtype, Layout.row_major(BT, D + 1), MutAnyOrigin]]()
    var mod_cs_t = List[LayoutTensor[dtype, Layout.row_major(BT, 2 * D), MutAnyOrigin]]()
    var mm_in_cs_t = List[LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin]]()
    var bias_in_cs_t = List[LayoutTensor[dtype, Layout.row_major(BT, 0), MutAnyOrigin]]()
    var gate_cs_t = List[LayoutTensor[dtype, Layout.row_major(BT, 2 * D), MutAnyOrigin]]()
    var raw_mods_t = List[LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin]]()
    for k in range(N_BLOCKS):
        adaln_w_ts.append(LayoutTensor[dtype, Layout.row_major(ADALN_W_PER), MutAnyOrigin](adaln_ws[k].unsafe_ptr()))
        adaln_b_ts.append(LayoutTensor[dtype, Layout.row_major(ADALN_B_PER), MutAnyOrigin](adaln_bs[k].unsafe_ptr()))
        inner_w_ts.append(LayoutTensor[dtype, Layout.row_major(INNER_W_PER), MutAnyOrigin](inner_ws[k].unsafe_ptr()))
        inner_b_ts.append(LayoutTensor[dtype, Layout.row_major(INNER_B_PER), MutAnyOrigin](inner_bs[k].unsafe_ptr()))
        x_after_t.append(LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin](x_after[k].unsafe_ptr()))
        silu_cs_t.append(LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin](silu_c_caches[k].unsafe_ptr()))
        mm_ad_cs_t.append(LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin](mm_ad_caches[k].unsafe_ptr()))
        bias_ad_cs_t.append(LayoutTensor[dtype, Layout.row_major(BT, 0), MutAnyOrigin](bias_ad_caches[k].unsafe_ptr()))
        ln_cs_t.append(LayoutTensor[dtype, Layout.row_major(BT, D + 1), MutAnyOrigin](ln_caches[k].unsafe_ptr()))
        mod_cs_t.append(LayoutTensor[dtype, Layout.row_major(BT, 2 * D), MutAnyOrigin](mod_caches[k].unsafe_ptr()))
        mm_in_cs_t.append(LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin](mm_in_caches[k].unsafe_ptr()))
        bias_in_cs_t.append(LayoutTensor[dtype, Layout.row_major(BT, 0), MutAnyOrigin](bias_in_caches[k].unsafe_ptr()))
        gate_cs_t.append(LayoutTensor[dtype, Layout.row_major(BT, 2 * D), MutAnyOrigin](gate_caches[k].unsafe_ptr()))
        raw_mods_t.append(LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin](raw_mods_arr[k].unsafe_ptr()))

    # -------- Shared scratch buffers --------
    var silu_buf = InlineArray[Scalar[dtype], BT * D](uninitialized=True)
    var ad_mm_buf = InlineArray[Scalar[dtype], BT * 3 * D](uninitialized=True)
    var ln_buf = InlineArray[Scalar[dtype], BT * D](uninitialized=True)
    var mod_inp_buf = InlineArray[Scalar[dtype], BT * 3 * D](uninitialized=True)
    var mod_x_buf = InlineArray[Scalar[dtype], BT * D](uninitialized=True)
    var inner_mm_buf = InlineArray[Scalar[dtype], BT * D](uninitialized=True)
    var inner_out_buf = InlineArray[Scalar[dtype], BT * D](uninitialized=True)
    var gate_inp_buf = InlineArray[Scalar[dtype], BT * 3 * D](uninitialized=True)
    var silu_buf_t = LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin](silu_buf.unsafe_ptr())
    var ad_mm_buf_t = LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin](ad_mm_buf.unsafe_ptr())
    var ln_buf_t = LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin](ln_buf.unsafe_ptr())
    var mod_inp_buf_t = LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin](mod_inp_buf.unsafe_ptr())
    var mod_x_buf_t = LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin](mod_x_buf.unsafe_ptr())
    var inner_mm_buf_t = LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin](inner_mm_buf.unsafe_ptr())
    var inner_out_buf_t = LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin](inner_out_buf.unsafe_ptr())
    var gate_inp_buf_t = LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin](gate_inp_buf.unsafe_ptr())

    # -------- Test 1: zero-init identity --------
    print()
    print("--- Test 1: zero-init identity ---")
    # Save original adaLN and zero them out for this test.
    for k in range(N_BLOCKS):
        for i in range(ADALN_W_PER):
            adaln_ws[k][i] = Scalar[dtype](0.0)
        for i in range(ADALN_B_PER):
            adaln_bs[k][i] = Scalar[dtype](0.0)

    ar_predictor_forward(
        x_input_t, c_input_t, x_after_t,
        adaln_w_ts, adaln_b_ts, inner_w_ts, inner_b_ts,
        silu_cs_t, mm_ad_cs_t, bias_ad_cs_t, ln_cs_t, mod_cs_t,
        mm_in_cs_t, bias_in_cs_t, gate_cs_t,
        raw_mods_t,
        silu_buf_t, ad_mm_buf_t, ln_buf_t, mod_inp_buf_t, mod_x_buf_t,
        inner_mm_buf_t, inner_out_buf_t, gate_inp_buf_t,
    )

    var max_diff_id = Float64(0.0)
    for i in range(BT * D):
        var d = abs(Float64(x_after[N_BLOCKS - 1][i]) - Float64(x_input[i]))
        if d > max_diff_id:
            max_diff_id = d
    if max_diff_id == 0.0:
        print("  [PASS] stacked zero-init identity: bitwise (max_diff =", max_diff_id, ")")
    else:
        print("  [FAIL] stacked zero-init identity: max_diff =", max_diff_id)

    # -------- Test 2: backward gradcheck at non-zero init --------
    print()
    print("--- Test 2: backward gradcheck (x_input, c_input) ---")
    # Restore non-zero adaLN params.
    for k in range(N_BLOCKS):
        for i in range(ADALN_W_PER):
            adaln_ws[k][i] = Scalar[dtype](
                0.08 * Float64((i + 7 * k) % 13) - 0.32
            )
        for i in range(ADALN_B_PER):
            adaln_bs[k][i] = Scalar[dtype](
                0.05 * Float64((i + 3 * k) % 5) - 0.1
            )

    # Forward at non-zero init.
    ar_predictor_forward(
        x_input_t, c_input_t, x_after_t,
        adaln_w_ts, adaln_b_ts, inner_w_ts, inner_b_ts,
        silu_cs_t, mm_ad_cs_t, bias_ad_cs_t, ln_cs_t, mod_cs_t,
        mm_in_cs_t, bias_in_cs_t, gate_cs_t,
        raw_mods_t,
        silu_buf_t, ad_mm_buf_t, ln_buf_t, mod_inp_buf_t, mod_x_buf_t,
        inner_mm_buf_t, inner_out_buf_t, gate_inp_buf_t,
    )

    # grad_output = ones (so L = sum(output))
    var grad_out_arr = InlineArray[Scalar[dtype], BT * D](uninitialized=True)
    for i in range(BT * D):
        grad_out_arr[i] = Scalar[dtype](1.0)
    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BT, D), MutAnyOrigin
    ](grad_out_arr.unsafe_ptr())

    # grad buffers
    var grad_x_input = InlineArray[Scalar[dtype], BT * D](uninitialized=True)
    var grad_c_input = InlineArray[Scalar[dtype], BT * D](uninitialized=True)
    var grad_x_input_t = LayoutTensor[
        dtype, Layout.row_major(BT, D), MutAnyOrigin
    ](grad_x_input.unsafe_ptr())
    var grad_c_input_t = LayoutTensor[
        dtype, Layout.row_major(BT, D), MutAnyOrigin
    ](grad_c_input.unsafe_ptr())

    # Per-block grad-x flow buffers.
    var grad_x_flows = List[InlineArray[Scalar[dtype], BT * D]]()
    var grad_x_flows_t = List[LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin]]()
    var g_adaln_w = List[InlineArray[Scalar[dtype], ADALN_W_PER]]()
    var g_adaln_b = List[InlineArray[Scalar[dtype], ADALN_B_PER]]()
    var g_inner_w = List[InlineArray[Scalar[dtype], INNER_W_PER]]()
    var g_inner_b = List[InlineArray[Scalar[dtype], INNER_B_PER]]()
    var g_adaln_w_ts = List[LayoutTensor[dtype, Layout.row_major(ADALN_W_PER), MutAnyOrigin]]()
    var g_adaln_b_ts = List[LayoutTensor[dtype, Layout.row_major(ADALN_B_PER), MutAnyOrigin]]()
    var g_inner_w_ts = List[LayoutTensor[dtype, Layout.row_major(INNER_W_PER), MutAnyOrigin]]()
    var g_inner_b_ts = List[LayoutTensor[dtype, Layout.row_major(INNER_B_PER), MutAnyOrigin]]()
    for k in range(N_BLOCKS):
        grad_x_flows.append(InlineArray[Scalar[dtype], BT * D](uninitialized=True))
        g_adaln_w.append(InlineArray[Scalar[dtype], ADALN_W_PER](uninitialized=True))
        g_adaln_b.append(InlineArray[Scalar[dtype], ADALN_B_PER](uninitialized=True))
        g_inner_w.append(InlineArray[Scalar[dtype], INNER_W_PER](uninitialized=True))
        g_inner_b.append(InlineArray[Scalar[dtype], INNER_B_PER](uninitialized=True))
        for i in range(ADALN_W_PER):
            g_adaln_w[k][i] = Scalar[dtype](0.0)
        for i in range(ADALN_B_PER):
            g_adaln_b[k][i] = Scalar[dtype](0.0)
        for i in range(INNER_W_PER):
            g_inner_w[k][i] = Scalar[dtype](0.0)
        for i in range(INNER_B_PER):
            g_inner_b[k][i] = Scalar[dtype](0.0)
    for k in range(N_BLOCKS):
        grad_x_flows_t.append(LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin](grad_x_flows[k].unsafe_ptr()))
        g_adaln_w_ts.append(LayoutTensor[dtype, Layout.row_major(ADALN_W_PER), MutAnyOrigin](g_adaln_w[k].unsafe_ptr()))
        g_adaln_b_ts.append(LayoutTensor[dtype, Layout.row_major(ADALN_B_PER), MutAnyOrigin](g_adaln_b[k].unsafe_ptr()))
        g_inner_w_ts.append(LayoutTensor[dtype, Layout.row_major(INNER_W_PER), MutAnyOrigin](g_inner_w[k].unsafe_ptr()))
        g_inner_b_ts.append(LayoutTensor[dtype, Layout.row_major(INNER_B_PER), MutAnyOrigin](g_inner_b[k].unsafe_ptr()))

    # Backward scratch
    var sgg = InlineArray[Scalar[dtype], BT * 3 * D](uninitialized=True)
    var sgi_out = InlineArray[Scalar[dtype], BT * D](uninitialized=True)
    var sgi_mm = InlineArray[Scalar[dtype], BT * D](uninitialized=True)
    var sgmx = InlineArray[Scalar[dtype], BT * D](uninitialized=True)
    var sgmi = InlineArray[Scalar[dtype], BT * 3 * D](uninitialized=True)
    var sgln_out = InlineArray[Scalar[dtype], BT * D](uninitialized=True)
    var sgln_in = InlineArray[Scalar[dtype], BT * D](uninitialized=True)
    var sgrm = InlineArray[Scalar[dtype], BT * 3 * D](uninitialized=True)
    var sgam = InlineArray[Scalar[dtype], BT * 3 * D](uninitialized=True)
    var sgsc = InlineArray[Scalar[dtype], BT * D](uninitialized=True)
    var sgcb = InlineArray[Scalar[dtype], BT * D](uninitialized=True)
    var sgg_t = LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin](sgg.unsafe_ptr())
    var sgi_out_t = LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin](sgi_out.unsafe_ptr())
    var sgi_mm_t = LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin](sgi_mm.unsafe_ptr())
    var sgmx_t = LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin](sgmx.unsafe_ptr())
    var sgmi_t = LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin](sgmi.unsafe_ptr())
    var sgln_out_t = LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin](sgln_out.unsafe_ptr())
    var sgln_in_t = LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin](sgln_in.unsafe_ptr())
    var sgrm_t = LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin](sgrm.unsafe_ptr())
    var sgam_t = LayoutTensor[dtype, Layout.row_major(BT, 3 * D), MutAnyOrigin](sgam.unsafe_ptr())
    var sgsc_t = LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin](sgsc.unsafe_ptr())
    var sgcb_t = LayoutTensor[dtype, Layout.row_major(BT, D), MutAnyOrigin](sgcb.unsafe_ptr())

    ar_predictor_backward(
        grad_out_t, grad_x_input_t, grad_c_input_t,
        adaln_w_ts, adaln_b_ts, inner_w_ts, inner_b_ts,
        g_adaln_w_ts, g_adaln_b_ts, g_inner_w_ts, g_inner_b_ts,
        silu_cs_t, mm_ad_cs_t, bias_ad_cs_t, ln_cs_t, mod_cs_t,
        mm_in_cs_t, bias_in_cs_t, gate_cs_t,
        grad_x_flows_t,
        sgg_t, sgi_out_t, sgi_mm_t, sgmx_t, sgmi_t, sgln_out_t, sgln_in_t,
        sgrm_t, sgam_t, sgsc_t, sgcb_t,
    )

    # FD on x_input and c_input.
    # 2-block stacked FD accumulates fp32 noise — use both abs + rel tolerance.
    var max_abs_x = Float64(0.0)
    var max_abs_c = Float64(0.0)
    var max_rel_x = Float64(0.0)
    var max_rel_c = Float64(0.0)
    var max_grad_x = Float64(0.0)
    var max_grad_c = Float64(0.0)
    var eps = Float64(1e-4)

    for idx in range(BT * D):
        var orig = x_input[idx]
        x_input[idx] = orig + Scalar[dtype](eps)
        ar_predictor_forward(
            x_input_t, c_input_t, x_after_t,
            adaln_w_ts, adaln_b_ts, inner_w_ts, inner_b_ts,
            silu_cs_t, mm_ad_cs_t, bias_ad_cs_t, ln_cs_t, mod_cs_t,
            mm_in_cs_t, bias_in_cs_t, gate_cs_t,
            raw_mods_t,
            silu_buf_t, ad_mm_buf_t, ln_buf_t, mod_inp_buf_t, mod_x_buf_t,
            inner_mm_buf_t, inner_out_buf_t, gate_inp_buf_t,
        )
        var fp = Float64(0.0)
        for j in range(BT * D):
            fp += Float64(x_after[N_BLOCKS - 1][j])

        x_input[idx] = orig - Scalar[dtype](eps)
        ar_predictor_forward(
            x_input_t, c_input_t, x_after_t,
            adaln_w_ts, adaln_b_ts, inner_w_ts, inner_b_ts,
            silu_cs_t, mm_ad_cs_t, bias_ad_cs_t, ln_cs_t, mod_cs_t,
            mm_in_cs_t, bias_in_cs_t, gate_cs_t,
            raw_mods_t,
            silu_buf_t, ad_mm_buf_t, ln_buf_t, mod_inp_buf_t, mod_x_buf_t,
            inner_mm_buf_t, inner_out_buf_t, gate_inp_buf_t,
        )
        var fm = Float64(0.0)
        for j in range(BT * D):
            fm += Float64(x_after[N_BLOCKS - 1][j])
        x_input[idx] = orig
        var num_g = (fp - fm) / (2.0 * eps)
        var ana_g = Float64(grad_x_input[idx])
        var err = abs(ana_g - num_g)
        if err > max_abs_x:
            max_abs_x = err
        var denom = abs(ana_g) + abs(num_g)
        if denom > 1e-6:
            var rel = err / denom
            if rel > max_rel_x:
                max_rel_x = rel
        if abs(ana_g) > max_grad_x:
            max_grad_x = abs(ana_g)

    for idx in range(BT * D):
        var orig = c_input[idx]
        c_input[idx] = orig + Scalar[dtype](eps)
        ar_predictor_forward(
            x_input_t, c_input_t, x_after_t,
            adaln_w_ts, adaln_b_ts, inner_w_ts, inner_b_ts,
            silu_cs_t, mm_ad_cs_t, bias_ad_cs_t, ln_cs_t, mod_cs_t,
            mm_in_cs_t, bias_in_cs_t, gate_cs_t,
            raw_mods_t,
            silu_buf_t, ad_mm_buf_t, ln_buf_t, mod_inp_buf_t, mod_x_buf_t,
            inner_mm_buf_t, inner_out_buf_t, gate_inp_buf_t,
        )
        var fp = Float64(0.0)
        for j in range(BT * D):
            fp += Float64(x_after[N_BLOCKS - 1][j])

        c_input[idx] = orig - Scalar[dtype](eps)
        ar_predictor_forward(
            x_input_t, c_input_t, x_after_t,
            adaln_w_ts, adaln_b_ts, inner_w_ts, inner_b_ts,
            silu_cs_t, mm_ad_cs_t, bias_ad_cs_t, ln_cs_t, mod_cs_t,
            mm_in_cs_t, bias_in_cs_t, gate_cs_t,
            raw_mods_t,
            silu_buf_t, ad_mm_buf_t, ln_buf_t, mod_inp_buf_t, mod_x_buf_t,
            inner_mm_buf_t, inner_out_buf_t, gate_inp_buf_t,
        )
        var fm = Float64(0.0)
        for j in range(BT * D):
            fm += Float64(x_after[N_BLOCKS - 1][j])
        c_input[idx] = orig
        var num_g = (fp - fm) / (2.0 * eps)
        var ana_g = Float64(grad_c_input[idx])
        var err = abs(ana_g - num_g)
        if err > max_abs_c:
            max_abs_c = err
        var denom = abs(ana_g) + abs(num_g)
        if denom > 1e-6:
            var rel = err / denom
            if rel > max_rel_c:
                max_rel_c = rel
        if abs(ana_g) > max_grad_c:
            max_grad_c = abs(ana_g)

    print(
        "    grad_x: max|ana| =", max_grad_x,
        " max|err| =", max_abs_x,
        " max_rel_err =", max_rel_x,
    )
    print(
        "    grad_c: max|ana| =", max_grad_c,
        " max|err| =", max_abs_c,
        " max_rel_err =", max_rel_c,
    )

    # 2-block FD noise floor ≈ 1e-3 abs, 1e-2 rel for D=4 chain.
    if max_rel_x < 1e-2 and max_rel_c < 1e-2:
        print("  [PASS] AR predictor gradcheck (relative tolerance 1e-2)")
    else:
        print("  [FAIL] AR predictor gradcheck")
    print()
    print("=== Done ===")

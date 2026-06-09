"""Conditional Transformer Block — AdaLN-zero composition test.

Validates the AdaLN-zero residual pattern from LeWM's `ConditionalBlock`
(reference: `references/le-wm-main/module.py:88-111`). Exercises one residual
branch with a Linear inner module (placeholder for attention or FFN — both
share the same AdaLN-zero wrapping).

Branch:
    c_silu  = SiLU(c)                                       # (B, D)
    raw_mod = c_silu @ W_adaLN + b_adaLN                    # (B, 3*D)
    shift, scale, gate = chunk(raw_mod, 3, dim=-1)          # each (B, D)
    ln_x   = LayerNormNoAffine(x)                            # (B, D)
    mod_x  = ln_x * (1 + scale) + shift                      # (B, D)
    inner  = mod_x @ W_inner + b_inner                       # (B, D)
    out    = x + gate * inner                                # (B, D)

Tests:
  1. Zero-init invariance: with W_adaLN = 0, b_adaLN = 0, all of
     (shift, scale, gate) = 0. modulate is identity, gate kills the branch:
     out = x bitwise (the "zero-initialized residual" property that lets
     AdaLN-zero blocks stack stably from random init).
  2. Backward consistency: at non-zero adaLN init, finite-difference grad
     w.r.t. x matches analytical grad_x produced by the autodiff chain.

The two-branch ConditionalTransformerBlock is the same pattern applied twice
(MSA + MLP) — both branches consume the same `c` via a shared 6*D adaLN
projection. Once this single-branch test passes, the two-branch wrapping is
direct composition.

Run:
    pixi run mojo run -I . tests/experimental/lewm/test_conditional_block.mojo
"""

from std.memory import alloc
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
# Forward pass — orchestrates the AdaLN-zero block end-to-end on CPU.
#
# Buffers are passed in; this function only does compute. Callers manage
# memory so the test can reuse buffers between forward/backward and FD probes.
# =============================================================================
def adaln_zero_forward[
    BATCH: Int, D: Int
](
    # Inputs
    x: LayoutTensor[dtype, Layout.row_major(BATCH, D), MutAnyOrigin],
    c: LayoutTensor[dtype, Layout.row_major(BATCH, D), MutAnyOrigin],
    # Params (adaLN_W, adaLN_b, inner_W, inner_b)
    adaln_w: LayoutTensor[dtype, Layout.row_major(D * 3 * D), MutAnyOrigin],
    adaln_b: LayoutTensor[dtype, Layout.row_major(3 * D), MutAnyOrigin],
    inner_w: LayoutTensor[dtype, Layout.row_major(D * D), MutAnyOrigin],
    inner_b: LayoutTensor[dtype, Layout.row_major(D), MutAnyOrigin],
    # Output
    mut out: LayoutTensor[dtype, Layout.row_major(BATCH, D), MutAnyOrigin],
    # Caches (one per primitive op; sizes match the op's CACHE_SIZE per sample)
    mut silu_cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ],
    mut matmul_adaln_cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ],
    mut bias_adaln_cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, 0), MutAnyOrigin
    ],
    mut ln_cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, D + 1), MutAnyOrigin
    ],
    mut modulate_cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, 2 * D), MutAnyOrigin
    ],
    mut matmul_inner_cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ],
    mut bias_inner_cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, 0), MutAnyOrigin
    ],
    mut gate_cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, 2 * D), MutAnyOrigin
    ],
    # Intermediates (reused by backward)
    mut c_silu_out: LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ],
    mut adaln_matmul_out: LayoutTensor[
        dtype, Layout.row_major(BATCH, 3 * D), MutAnyOrigin
    ],
    mut raw_mod: LayoutTensor[
        dtype, Layout.row_major(BATCH, 3 * D), MutAnyOrigin
    ],
    mut ln_out: LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ],
    mut modulate_input: LayoutTensor[
        dtype, Layout.row_major(BATCH, 3 * D), MutAnyOrigin
    ],
    mut mod_x: LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ],
    mut inner_matmul_out: LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ],
    mut inner_out: LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ],
    mut gate_input: LayoutTensor[
        dtype, Layout.row_major(BATCH, 3 * D), MutAnyOrigin
    ],
) raises:
    # Empty-param view for ops with PARAM_SIZE=0
    var empty_params = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=Int(0)))

    # 1. c_silu = SiLU(c)
    SwishOp[D].eval[BATCH](c, c_silu_out, empty_params, silu_cache)

    # 2. adaln_matmul_out = c_silu @ W_adaLN
    MatMul[D, 3 * D].eval[BATCH](
        c_silu_out, adaln_matmul_out, adaln_w, matmul_adaln_cache
    )

    # 3. raw_mod = adaln_matmul_out + b_adaLN
    BiasAdd[3 * D].eval[BATCH](
        adaln_matmul_out, raw_mod, adaln_b, bias_adaln_cache
    )

    # 4. ln_out = LayerNormNoAffine(x)
    LayerNormNoAffineOp[D].eval[BATCH](
        x, ln_out, empty_params, ln_cache
    )

    # 5. modulate_input = concat(ln_out, scale, shift)
    #    raw_mod layout: [shift_0..D, scale_D..2D, gate_2D..3D]
    #    chunk: shift = raw_mod[:, 0:D], scale = raw_mod[:, D:2D],
    #           gate = raw_mod[:, 2D:3D]
    #    Modulate expects: [x_0..D, scale_D..2D, shift_2D..3D]
    for b in range(BATCH):
        for i in range(D):
            modulate_input[b, i] = ln_out[b, i]
            modulate_input[b, D + i] = raw_mod[b, D + i]            # scale
            modulate_input[b, 2 * D + i] = raw_mod[b, i]            # shift

    # 6. mod_x = Modulate(modulate_input)
    ModulateOp[D].eval[BATCH](
        modulate_input, mod_x, empty_params, modulate_cache
    )

    # 7. inner_matmul_out = mod_x @ W_inner
    MatMul[D, D].eval[BATCH](
        mod_x, inner_matmul_out, inner_w, matmul_inner_cache
    )

    # 8. inner_out = inner_matmul_out + b_inner
    BiasAdd[D].eval[BATCH](
        inner_matmul_out, inner_out, inner_b, bias_inner_cache
    )

    # 9. gate_input = concat(x, gate, inner_out)
    for b in range(BATCH):
        for i in range(D):
            gate_input[b, i] = x[b, i]
            gate_input[b, D + i] = raw_mod[b, 2 * D + i]            # gate
            gate_input[b, 2 * D + i] = inner_out[b, i]

    # 10. out = Gate(gate_input)
    GateOp[D].eval[BATCH](
        gate_input, out, empty_params, gate_cache
    )


# =============================================================================
# Backward pass — reverses the forward chain and produces grad_x.
# Param grads are computed but discarded (we only validate grad_x in this test).
# =============================================================================
def adaln_zero_backward[
    BATCH: Int, D: Int
](
    grad_out: LayoutTensor[dtype, Layout.row_major(BATCH, D), MutAnyOrigin],
    # Params (read-only)
    adaln_w: LayoutTensor[dtype, Layout.row_major(D * 3 * D), MutAnyOrigin],
    adaln_b: LayoutTensor[dtype, Layout.row_major(3 * D), MutAnyOrigin],
    inner_w: LayoutTensor[dtype, Layout.row_major(D * D), MutAnyOrigin],
    inner_b: LayoutTensor[dtype, Layout.row_major(D), MutAnyOrigin],
    # Caches (from forward)
    silu_cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ],
    matmul_adaln_cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ],
    bias_adaln_cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, 0), MutAnyOrigin
    ],
    ln_cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, D + 1), MutAnyOrigin
    ],
    modulate_cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, 2 * D), MutAnyOrigin
    ],
    matmul_inner_cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ],
    bias_inner_cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, 0), MutAnyOrigin
    ],
    gate_cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, 2 * D), MutAnyOrigin
    ],
    # Outputs
    mut grad_x: LayoutTensor[dtype, Layout.row_major(BATCH, D), MutAnyOrigin],
    mut grad_c: LayoutTensor[dtype, Layout.row_major(BATCH, D), MutAnyOrigin],
    # Param grads (written but not checked here)
    mut grad_adaln_w: LayoutTensor[
        dtype, Layout.row_major(D * 3 * D), MutAnyOrigin
    ],
    mut grad_adaln_b: LayoutTensor[
        dtype, Layout.row_major(3 * D), MutAnyOrigin
    ],
    mut grad_inner_w: LayoutTensor[
        dtype, Layout.row_major(D * D), MutAnyOrigin
    ],
    mut grad_inner_b: LayoutTensor[dtype, Layout.row_major(D), MutAnyOrigin],
    # Scratch — gradient buffers reused across the chain.
    mut grad_gate_input: LayoutTensor[
        dtype, Layout.row_major(BATCH, 3 * D), MutAnyOrigin
    ],
    mut grad_inner_out: LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ],
    mut grad_inner_matmul: LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ],
    mut grad_mod_x: LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ],
    mut grad_modulate_input: LayoutTensor[
        dtype, Layout.row_major(BATCH, 3 * D), MutAnyOrigin
    ],
    mut grad_ln_out: LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ],
    mut grad_raw_mod: LayoutTensor[
        dtype, Layout.row_major(BATCH, 3 * D), MutAnyOrigin
    ],
    mut grad_adaln_matmul: LayoutTensor[
        dtype, Layout.row_major(BATCH, 3 * D), MutAnyOrigin
    ],
    mut grad_c_silu: LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ],
) raises:
    var empty_params = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=Int(0)))
    var empty_grad_params = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=Int(0)))

    # 10' Gate.vjp: grad_out → grad_gate_input
    GateOp[D].vjp[BATCH](
        grad_out, grad_gate_input, empty_params, gate_cache, empty_grad_params
    )

    # Split grad_gate_input → (grad_x_residual, grad_gate_chunk, grad_inner_out)
    for b in range(BATCH):
        for i in range(D):
            grad_x[b, i] = grad_gate_input[b, i]                     # residual x
            grad_raw_mod[b, 2 * D + i] = grad_gate_input[b, D + i]   # gate
            grad_inner_out[b, i] = grad_gate_input[b, 2 * D + i]

    # 8' BiasAdd.vjp: grad_inner_out → grad_inner_matmul; grad_inner_b
    BiasAdd[D].vjp[BATCH](
        grad_inner_out,
        grad_inner_matmul,
        inner_b,
        bias_inner_cache,
        grad_inner_b,
    )

    # 7' MatMul.vjp: grad_inner_matmul → grad_mod_x; grad_inner_w
    MatMul[D, D].vjp[BATCH](
        grad_inner_matmul,
        grad_mod_x,
        inner_w,
        matmul_inner_cache,
        grad_inner_w,
    )

    # 6' Modulate.vjp: grad_mod_x → grad_modulate_input
    ModulateOp[D].vjp[BATCH](
        grad_mod_x,
        grad_modulate_input,
        empty_params,
        modulate_cache,
        empty_grad_params,
    )

    # Split grad_modulate_input → (grad_ln_out, grad_scale_chunk, grad_shift_chunk)
    for b in range(BATCH):
        for i in range(D):
            grad_ln_out[b, i] = grad_modulate_input[b, i]
            grad_raw_mod[b, D + i] = grad_modulate_input[b, D + i]    # scale
            grad_raw_mod[b, i] = grad_modulate_input[b, 2 * D + i]    # shift

    # 4' LayerNormNoAffine.vjp: grad_ln_out → grad_x (add into residual)
    var grad_ln_in_arr = InlineArray[Scalar[dtype], BATCH * D](
        uninitialized=True
    )
    var grad_ln_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](grad_ln_in_arr.unsafe_ptr())
    LayerNormNoAffineOp[D].vjp[BATCH](
        grad_ln_out,
        grad_ln_in_t,
        empty_params,
        ln_cache,
        empty_grad_params,
    )
    for b in range(BATCH):
        for i in range(D):
            grad_x[b, i] = grad_x[b, i] + grad_ln_in_t[b, i]

    # 3' BiasAdd.vjp on adaLN: grad_raw_mod → grad_adaln_matmul; grad_adaln_b
    BiasAdd[3 * D].vjp[BATCH](
        grad_raw_mod,
        grad_adaln_matmul,
        adaln_b,
        bias_adaln_cache,
        grad_adaln_b,
    )

    # 2' MatMul.vjp on adaLN: grad_adaln_matmul → grad_c_silu; grad_adaln_w
    MatMul[D, 3 * D].vjp[BATCH](
        grad_adaln_matmul,
        grad_c_silu,
        adaln_w,
        matmul_adaln_cache,
        grad_adaln_w,
    )

    # 1' SwishOp.vjp: grad_c_silu → grad_c
    SwishOp[D].vjp[BATCH](
        grad_c_silu, grad_c, empty_params, silu_cache, empty_grad_params
    )


# =============================================================================
# Test 1 — zero-init invariance
# =============================================================================

def test_zero_init_identity() raises:
    """At zero adaLN init, the block must be exactly identity."""
    comptime BATCH = 2
    comptime D = 4

    # Allocate everything
    var x_arr = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    var c_arr = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    for i in range(BATCH * D):
        x_arr[i] = Scalar[dtype](0.31 * Float64(i % 7) - 0.5)
        c_arr[i] = Scalar[dtype](0.17 * Float64(i % 11) + 0.2)

    var adaln_w_arr = InlineArray[Scalar[dtype], D * 3 * D](uninitialized=True)
    var adaln_b_arr = InlineArray[Scalar[dtype], 3 * D](uninitialized=True)
    for i in range(D * 3 * D):
        adaln_w_arr[i] = Scalar[dtype](0.0)      # zero-init
    for i in range(3 * D):
        adaln_b_arr[i] = Scalar[dtype](0.0)      # zero-init

    var inner_w_arr = InlineArray[Scalar[dtype], D * D](uninitialized=True)
    var inner_b_arr = InlineArray[Scalar[dtype], D](uninitialized=True)
    for i in range(D * D):
        inner_w_arr[i] = Scalar[dtype](0.2 * Float64(i % 5) - 0.3)
    for i in range(D):
        inner_b_arr[i] = Scalar[dtype](0.1)

    # Caches
    var silu_c = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    var mm_ad_c = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    var bias_ad_c = InlineArray[Scalar[dtype], 1](uninitialized=True)
    var ln_c = InlineArray[Scalar[dtype], BATCH * (D + 1)](uninitialized=True)
    var mod_c = InlineArray[Scalar[dtype], BATCH * 2 * D](uninitialized=True)
    var mm_in_c = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    var bias_in_c = InlineArray[Scalar[dtype], 1](uninitialized=True)
    var gate_c = InlineArray[Scalar[dtype], BATCH * 2 * D](uninitialized=True)

    # Intermediates
    var c_silu = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    var ad_mm = InlineArray[Scalar[dtype], BATCH * 3 * D](uninitialized=True)
    var raw_mod = InlineArray[Scalar[dtype], BATCH * 3 * D](uninitialized=True)
    var ln_out = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    var mod_inp = InlineArray[Scalar[dtype], BATCH * 3 * D](uninitialized=True)
    var mod_x = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    var in_mm = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    var in_out = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    var gate_inp = InlineArray[Scalar[dtype], BATCH * 3 * D](uninitialized=True)
    var out = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)

    var x_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](x_arr.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](c_arr.unsafe_ptr())
    var adaln_w_t = LayoutTensor[
        dtype, Layout.row_major(D * 3 * D), MutAnyOrigin
    ](adaln_w_arr.unsafe_ptr())
    var adaln_b_t = LayoutTensor[
        dtype, Layout.row_major(3 * D), MutAnyOrigin
    ](adaln_b_arr.unsafe_ptr())
    var inner_w_t = LayoutTensor[
        dtype, Layout.row_major(D * D), MutAnyOrigin
    ](inner_w_arr.unsafe_ptr())
    var inner_b_t = LayoutTensor[
        dtype, Layout.row_major(D), MutAnyOrigin
    ](inner_b_arr.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](out.unsafe_ptr())
    var silu_c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](silu_c.unsafe_ptr())
    var mm_ad_c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](mm_ad_c.unsafe_ptr())
    var bias_ad_c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 0), MutAnyOrigin
    ](bias_ad_c.unsafe_ptr())
    var ln_c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D + 1), MutAnyOrigin
    ](ln_c.unsafe_ptr())
    var mod_c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 2 * D), MutAnyOrigin
    ](mod_c.unsafe_ptr())
    var mm_in_c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](mm_in_c.unsafe_ptr())
    var bias_in_c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 0), MutAnyOrigin
    ](bias_in_c.unsafe_ptr())
    var gate_c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 2 * D), MutAnyOrigin
    ](gate_c.unsafe_ptr())
    var c_silu_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](c_silu.unsafe_ptr())
    var ad_mm_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 3 * D), MutAnyOrigin
    ](ad_mm.unsafe_ptr())
    var raw_mod_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 3 * D), MutAnyOrigin
    ](raw_mod.unsafe_ptr())
    var ln_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](ln_out.unsafe_ptr())
    var mod_inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 3 * D), MutAnyOrigin
    ](mod_inp.unsafe_ptr())
    var mod_x_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](mod_x.unsafe_ptr())
    var in_mm_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](in_mm.unsafe_ptr())
    var in_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](in_out.unsafe_ptr())
    var gate_inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 3 * D), MutAnyOrigin
    ](gate_inp.unsafe_ptr())

    adaln_zero_forward[BATCH, D](
        x_t, c_t, adaln_w_t, adaln_b_t, inner_w_t, inner_b_t, out_t,
        silu_c_t, mm_ad_c_t, bias_ad_c_t, ln_c_t, mod_c_t, mm_in_c_t,
        bias_in_c_t, gate_c_t,
        c_silu_t, ad_mm_t, raw_mod_t, ln_out_t, mod_inp_t, mod_x_t,
        in_mm_t, in_out_t, gate_inp_t,
    )

    # Verify out == x bitwise
    var max_diff = Float64(0.0)
    for i in range(BATCH * D):
        var d = abs(Float64(out[i]) - Float64(x_arr[i]))
        if d > max_diff:
            max_diff = d

    if max_diff == 0.0:
        print(
            "  [PASS] zero-init identity: out == x bitwise (max_diff =",
            max_diff,
            ")",
        )
    else:
        print(
            "  [FAIL] zero-init identity: out != x  (max_diff =",
            max_diff,
            ")",
        )


# =============================================================================
# Test 2 — backward gradcheck on x with non-zero adaLN init.
# =============================================================================

def test_grad_x_finite_diff() raises:
    """At non-zero adaLN init, dout/dx matches finite-difference."""
    comptime BATCH = 2
    comptime D = 4

    var x_arr = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    var c_arr = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    for i in range(BATCH * D):
        x_arr[i] = Scalar[dtype](0.23 * Float64(i % 7) - 0.4)
        c_arr[i] = Scalar[dtype](0.13 * Float64(i % 11) + 0.15)

    # Non-trivial adaLN: small but non-zero.
    var adaln_w_arr = InlineArray[Scalar[dtype], D * 3 * D](uninitialized=True)
    for i in range(D * 3 * D):
        adaln_w_arr[i] = Scalar[dtype](0.07 * Float64(i % 13) - 0.3)
    var adaln_b_arr = InlineArray[Scalar[dtype], 3 * D](uninitialized=True)
    for i in range(3 * D):
        adaln_b_arr[i] = Scalar[dtype](0.05 * Float64(i % 5) - 0.1)

    var inner_w_arr = InlineArray[Scalar[dtype], D * D](uninitialized=True)
    for i in range(D * D):
        inner_w_arr[i] = Scalar[dtype](0.18 * Float64(i % 5) - 0.25)
    var inner_b_arr = InlineArray[Scalar[dtype], D](uninitialized=True)
    for i in range(D):
        inner_b_arr[i] = Scalar[dtype](0.07)

    # Caches + intermediates allocations.
    var silu_c = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    var mm_ad_c = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    var bias_ad_c = InlineArray[Scalar[dtype], 1](uninitialized=True)
    var ln_c = InlineArray[Scalar[dtype], BATCH * (D + 1)](uninitialized=True)
    var mod_c = InlineArray[Scalar[dtype], BATCH * 2 * D](uninitialized=True)
    var mm_in_c = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    var bias_in_c = InlineArray[Scalar[dtype], 1](uninitialized=True)
    var gate_c = InlineArray[Scalar[dtype], BATCH * 2 * D](uninitialized=True)

    var c_silu = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    var ad_mm = InlineArray[Scalar[dtype], BATCH * 3 * D](uninitialized=True)
    var raw_mod = InlineArray[Scalar[dtype], BATCH * 3 * D](uninitialized=True)
    var ln_out = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    var mod_inp = InlineArray[Scalar[dtype], BATCH * 3 * D](uninitialized=True)
    var mod_x = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    var in_mm = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    var in_out = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    var gate_inp = InlineArray[Scalar[dtype], BATCH * 3 * D](uninitialized=True)
    var out = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)

    # Bind LayoutTensors once
    var x_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](x_arr.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](c_arr.unsafe_ptr())
    var adaln_w_t = LayoutTensor[
        dtype, Layout.row_major(D * 3 * D), MutAnyOrigin
    ](adaln_w_arr.unsafe_ptr())
    var adaln_b_t = LayoutTensor[
        dtype, Layout.row_major(3 * D), MutAnyOrigin
    ](adaln_b_arr.unsafe_ptr())
    var inner_w_t = LayoutTensor[
        dtype, Layout.row_major(D * D), MutAnyOrigin
    ](inner_w_arr.unsafe_ptr())
    var inner_b_t = LayoutTensor[
        dtype, Layout.row_major(D), MutAnyOrigin
    ](inner_b_arr.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](out.unsafe_ptr())
    var silu_c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](silu_c.unsafe_ptr())
    var mm_ad_c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](mm_ad_c.unsafe_ptr())
    var bias_ad_c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 0), MutAnyOrigin
    ](bias_ad_c.unsafe_ptr())
    var ln_c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D + 1), MutAnyOrigin
    ](ln_c.unsafe_ptr())
    var mod_c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 2 * D), MutAnyOrigin
    ](mod_c.unsafe_ptr())
    var mm_in_c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](mm_in_c.unsafe_ptr())
    var bias_in_c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 0), MutAnyOrigin
    ](bias_in_c.unsafe_ptr())
    var gate_c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 2 * D), MutAnyOrigin
    ](gate_c.unsafe_ptr())
    var c_silu_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](c_silu.unsafe_ptr())
    var ad_mm_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 3 * D), MutAnyOrigin
    ](ad_mm.unsafe_ptr())
    var raw_mod_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 3 * D), MutAnyOrigin
    ](raw_mod.unsafe_ptr())
    var ln_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](ln_out.unsafe_ptr())
    var mod_inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 3 * D), MutAnyOrigin
    ](mod_inp.unsafe_ptr())
    var mod_x_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](mod_x.unsafe_ptr())
    var in_mm_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](in_mm.unsafe_ptr())
    var in_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](in_out.unsafe_ptr())
    var gate_inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 3 * D), MutAnyOrigin
    ](gate_inp.unsafe_ptr())

    # Forward
    adaln_zero_forward[BATCH, D](
        x_t, c_t, adaln_w_t, adaln_b_t, inner_w_t, inner_b_t, out_t,
        silu_c_t, mm_ad_c_t, bias_ad_c_t, ln_c_t, mod_c_t, mm_in_c_t,
        bias_in_c_t, gate_c_t,
        c_silu_t, ad_mm_t, raw_mod_t, ln_out_t, mod_inp_t, mod_x_t,
        in_mm_t, in_out_t, gate_inp_t,
    )

    # Backward with grad_out = ones (so L = sum(out))
    var grad_out_arr = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    for i in range(BATCH * D):
        grad_out_arr[i] = Scalar[dtype](1.0)
    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](grad_out_arr.unsafe_ptr())

    var grad_x = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    var grad_c = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    var g_adaln_w = InlineArray[Scalar[dtype], D * 3 * D](uninitialized=True)
    var g_adaln_b = InlineArray[Scalar[dtype], 3 * D](uninitialized=True)
    var g_inner_w = InlineArray[Scalar[dtype], D * D](uninitialized=True)
    var g_inner_b = InlineArray[Scalar[dtype], D](uninitialized=True)
    for i in range(D * 3 * D):
        g_adaln_w[i] = Scalar[dtype](0.0)
    for i in range(3 * D):
        g_adaln_b[i] = Scalar[dtype](0.0)
    for i in range(D * D):
        g_inner_w[i] = Scalar[dtype](0.0)
    for i in range(D):
        g_inner_b[i] = Scalar[dtype](0.0)

    var g_gate_inp = InlineArray[Scalar[dtype], BATCH * 3 * D](uninitialized=True)
    var g_inner_out = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    var g_inner_matmul = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    var g_mod_x = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    var g_mod_inp = InlineArray[Scalar[dtype], BATCH * 3 * D](uninitialized=True)
    var g_ln_out = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)
    var g_raw_mod = InlineArray[Scalar[dtype], BATCH * 3 * D](uninitialized=True)
    var g_adaln_matmul = InlineArray[Scalar[dtype], BATCH * 3 * D](uninitialized=True)
    var g_c_silu = InlineArray[Scalar[dtype], BATCH * D](uninitialized=True)

    var grad_x_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](grad_x.unsafe_ptr())
    var grad_c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](grad_c.unsafe_ptr())
    var g_adaln_w_t = LayoutTensor[
        dtype, Layout.row_major(D * 3 * D), MutAnyOrigin
    ](g_adaln_w.unsafe_ptr())
    var g_adaln_b_t = LayoutTensor[
        dtype, Layout.row_major(3 * D), MutAnyOrigin
    ](g_adaln_b.unsafe_ptr())
    var g_inner_w_t = LayoutTensor[
        dtype, Layout.row_major(D * D), MutAnyOrigin
    ](g_inner_w.unsafe_ptr())
    var g_inner_b_t = LayoutTensor[
        dtype, Layout.row_major(D), MutAnyOrigin
    ](g_inner_b.unsafe_ptr())
    var g_gate_inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 3 * D), MutAnyOrigin
    ](g_gate_inp.unsafe_ptr())
    var g_inner_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](g_inner_out.unsafe_ptr())
    var g_inner_matmul_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](g_inner_matmul.unsafe_ptr())
    var g_mod_x_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](g_mod_x.unsafe_ptr())
    var g_mod_inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 3 * D), MutAnyOrigin
    ](g_mod_inp.unsafe_ptr())
    var g_ln_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](g_ln_out.unsafe_ptr())
    var g_raw_mod_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 3 * D), MutAnyOrigin
    ](g_raw_mod.unsafe_ptr())
    var g_adaln_matmul_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 3 * D), MutAnyOrigin
    ](g_adaln_matmul.unsafe_ptr())
    var g_c_silu_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D), MutAnyOrigin
    ](g_c_silu.unsafe_ptr())

    adaln_zero_backward[BATCH, D](
        grad_out_t,
        adaln_w_t, adaln_b_t, inner_w_t, inner_b_t,
        silu_c_t, mm_ad_c_t, bias_ad_c_t, ln_c_t, mod_c_t,
        mm_in_c_t, bias_in_c_t, gate_c_t,
        grad_x_t, grad_c_t,
        g_adaln_w_t, g_adaln_b_t, g_inner_w_t, g_inner_b_t,
        g_gate_inp_t, g_inner_out_t, g_inner_matmul_t, g_mod_x_t,
        g_mod_inp_t, g_ln_out_t, g_raw_mod_t, g_adaln_matmul_t, g_c_silu_t,
    )

    # FD on x: f = sum(out_i). num_grad[x_idx] = (f(x+eps) - f(x-eps)) / (2*eps)
    var max_abs_x = Float64(0.0)
    var max_abs_c = Float64(0.0)
    var eps = Float64(1e-4)

    for idx in range(BATCH * D):
        var orig = x_arr[idx]
        x_arr[idx] = orig + Scalar[dtype](eps)
        adaln_zero_forward[BATCH, D](
            x_t, c_t, adaln_w_t, adaln_b_t, inner_w_t, inner_b_t, out_t,
            silu_c_t, mm_ad_c_t, bias_ad_c_t, ln_c_t, mod_c_t, mm_in_c_t,
            bias_in_c_t, gate_c_t,
            c_silu_t, ad_mm_t, raw_mod_t, ln_out_t, mod_inp_t, mod_x_t,
            in_mm_t, in_out_t, gate_inp_t,
        )
        var fplus = Float64(0.0)
        for j in range(BATCH * D):
            fplus += Float64(out[j])

        x_arr[idx] = orig - Scalar[dtype](eps)
        adaln_zero_forward[BATCH, D](
            x_t, c_t, adaln_w_t, adaln_b_t, inner_w_t, inner_b_t, out_t,
            silu_c_t, mm_ad_c_t, bias_ad_c_t, ln_c_t, mod_c_t, mm_in_c_t,
            bias_in_c_t, gate_c_t,
            c_silu_t, ad_mm_t, raw_mod_t, ln_out_t, mod_inp_t, mod_x_t,
            in_mm_t, in_out_t, gate_inp_t,
        )
        var fminus = Float64(0.0)
        for j in range(BATCH * D):
            fminus += Float64(out[j])
        x_arr[idx] = orig

        var num_g = (fplus - fminus) / (2.0 * eps)
        var ana_g = Float64(grad_x[idx])
        var err = abs(ana_g - num_g)
        if err > max_abs_x:
            max_abs_x = err

    # FD on c
    for idx in range(BATCH * D):
        var orig = c_arr[idx]
        c_arr[idx] = orig + Scalar[dtype](eps)
        adaln_zero_forward[BATCH, D](
            x_t, c_t, adaln_w_t, adaln_b_t, inner_w_t, inner_b_t, out_t,
            silu_c_t, mm_ad_c_t, bias_ad_c_t, ln_c_t, mod_c_t, mm_in_c_t,
            bias_in_c_t, gate_c_t,
            c_silu_t, ad_mm_t, raw_mod_t, ln_out_t, mod_inp_t, mod_x_t,
            in_mm_t, in_out_t, gate_inp_t,
        )
        var fplus = Float64(0.0)
        for j in range(BATCH * D):
            fplus += Float64(out[j])

        c_arr[idx] = orig - Scalar[dtype](eps)
        adaln_zero_forward[BATCH, D](
            x_t, c_t, adaln_w_t, adaln_b_t, inner_w_t, inner_b_t, out_t,
            silu_c_t, mm_ad_c_t, bias_ad_c_t, ln_c_t, mod_c_t, mm_in_c_t,
            bias_in_c_t, gate_c_t,
            c_silu_t, ad_mm_t, raw_mod_t, ln_out_t, mod_inp_t, mod_x_t,
            in_mm_t, in_out_t, gate_inp_t,
        )
        var fminus = Float64(0.0)
        for j in range(BATCH * D):
            fminus += Float64(out[j])
        c_arr[idx] = orig

        var num_g = (fplus - fminus) / (2.0 * eps)
        var ana_g = Float64(grad_c[idx])
        var err = abs(ana_g - num_g)
        if err > max_abs_c:
            max_abs_c = err

    if max_abs_x < 1e-3 and max_abs_c < 1e-3:
        print(
            "  [PASS] conditional-block gradcheck: max_abs_grad_x =",
            max_abs_x,
            "  max_abs_grad_c =",
            max_abs_c,
        )
    else:
        print(
            "  [FAIL] conditional-block gradcheck: max_abs_grad_x =",
            max_abs_x,
            "  max_abs_grad_c =",
            max_abs_c,
        )


def main() raises:
    print("=== ConditionalTransformerBlock (AdaLN-zero) — Phase 1 wiring ===")
    print()
    print("--- zero-init identity ---")
    test_zero_init_identity()
    print()
    print("--- backward gradcheck (x, c) ---")
    test_grad_x_finite_diff()
    print()
    print("=== Conditional block test done ===")

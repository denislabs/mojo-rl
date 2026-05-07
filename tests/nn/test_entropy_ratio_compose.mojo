"""Stage 2 composition test — entropy-ratio shape via chained DiffOps.

Reference TD-MPC2 computes (world_model.py:166-184):

    log_prob_pre  = gaussian_logprob(eps, log_std)              # depends on log_std
    scaled_lp     = log_prob_pre * ACT
    log_prob_post = log_prob_pre - sum(log(1 - tanh²(u)))       # depends on log_std AND mean
    entropy_scale = scaled_lp / (log_prob_post + 1e-8)
    scaled_entropy = -log_prob_post * entropy_scale

PyTorch's autograd traces every path (B + C + D + E in the audit). To
faithfully reproduce paths D + E in our manual backward we need the
divide and the two-input multiply to compose with their proper VJPs.

This test composes three DiffOps:
    ratio = DivideOp([a' | b])                  # = a' / (b + ε)
    prod  = ElemMulTwoInputOp([a | ratio])      # = a · ratio
    y     = NegateOp(prod)                      # = -a · a' / (b + ε)

and compares the resulting (grad_a, grad_a', grad_b) against the analytic
gradient of `y = -a · a' / (b + ε)`. With a, a', b distinct, all three grad
paths are independently exercised.

If this passes we have the building blocks for Stage 3: assemble the full
TDMPC2 entropy graph (with `a ≡ b ≡ log_prob_post` and `a' ≡ scaled_lp`)
and let autograd handle the algebraic cancellation that paths D+E provide.
"""

from std.math import abs as _abs
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.autodiff import DivideOp, ElemMulTwoInputOp
from mojo_rl.nn.autodiff.primitives import NegateOp


def _abs_f64(x: Float64) -> Float64:
    return -x if x < 0.0 else x


def test_entropy_ratio_compose() raises:
    comptime DIM = 3
    comptime BATCH = 2
    comptime EPS_F: Float64 = 1e-8

    comptime Div = DivideOp[DIM, EPS_F]
    comptime Mul = ElemMulTwoInputOp[DIM]
    comptime Neg = NegateOp[DIM]

    # ── Inputs ─────────────────────────────────────────────────────────────
    var a_arr = InlineArray[Scalar[dtype], BATCH * DIM](uninitialized=True)
    var ap_arr = InlineArray[Scalar[dtype], BATCH * DIM](uninitialized=True)
    var b_arr = InlineArray[Scalar[dtype], BATCH * DIM](uninitialized=True)
    a_arr[0] = -3.0
    a_arr[1] = -8.0
    a_arr[2] = 1.0
    a_arr[3] = -2.0
    a_arr[4] = -5.0
    a_arr[5] = 4.0
    ap_arr[0] = -50.0
    ap_arr[1] = -48.0
    ap_arr[2] = 6.0
    ap_arr[3] = -12.0
    ap_arr[4] = -30.0
    ap_arr[5] = 24.0
    b_arr[0] = -3.0  # NOTE: in TDMPC2 a == b (both are log_prob_post). Test that case + others.
    b_arr[1] = -10.0
    b_arr[2] = 1.5
    b_arr[3] = -2.5
    b_arr[4] = -5.0
    b_arr[5] = 4.0

    # ── Forward: ratio = a' / (b+eps) via DivideOp ─────────────────────────
    var div_in_arr = InlineArray[Scalar[dtype], BATCH * 2 * DIM](
        uninitialized=True
    )
    for k in range(BATCH):
        for i in range(DIM):
            div_in_arr[k * 2 * DIM + i] = ap_arr[k * DIM + i]
            div_in_arr[k * 2 * DIM + DIM + i] = b_arr[k * DIM + i]
    var div_out_arr = InlineArray[Scalar[dtype], BATCH * DIM](uninitialized=True)
    var div_cache_arr = InlineArray[Scalar[dtype], BATCH * 2 * DIM](
        uninitialized=True
    )
    var empty_params = InlineArray[Scalar[dtype], 1](uninitialized=True)

    var div_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 2 * DIM), MutAnyOrigin
    ](div_in_arr.unsafe_ptr())
    var div_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](div_out_arr.unsafe_ptr())
    var div_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 2 * DIM), MutAnyOrigin
    ](div_cache_arr.unsafe_ptr())
    var pT = LayoutTensor[dtype, Layout.row_major(0), MutAnyOrigin](
        empty_params.unsafe_ptr()
    )
    Div.eval[BATCH, dtype](div_in_t, div_out_t, pT, div_cache_t)

    # ── Forward: prod = a * ratio via ElemMulTwoInputOp ────────────────────
    var mul_in_arr = InlineArray[Scalar[dtype], BATCH * 2 * DIM](
        uninitialized=True
    )
    for k in range(BATCH):
        for i in range(DIM):
            mul_in_arr[k * 2 * DIM + i] = a_arr[k * DIM + i]
            mul_in_arr[k * 2 * DIM + DIM + i] = div_out_arr[k * DIM + i]
    var mul_out_arr = InlineArray[Scalar[dtype], BATCH * DIM](uninitialized=True)
    var mul_cache_arr = InlineArray[Scalar[dtype], BATCH * 2 * DIM](
        uninitialized=True
    )
    var mul_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 2 * DIM), MutAnyOrigin
    ](mul_in_arr.unsafe_ptr())
    var mul_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](mul_out_arr.unsafe_ptr())
    var mul_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 2 * DIM), MutAnyOrigin
    ](mul_cache_arr.unsafe_ptr())
    Mul.eval[BATCH, dtype](mul_in_t, mul_out_t, pT, mul_cache_t)

    # ── Forward: y = -prod via NegateOp ────────────────────────────────────
    var y_arr = InlineArray[Scalar[dtype], BATCH * DIM](uninitialized=True)
    var neg_cache = InlineArray[Scalar[dtype], 1](uninitialized=True)
    var y_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](y_arr.unsafe_ptr())
    var neg_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 0), MutAnyOrigin
    ](neg_cache.unsafe_ptr())
    Neg.eval[BATCH, dtype](mul_out_t, y_t, pT, neg_cache_t)

    # Verify forward y_b,i = -a_b,i * a'_b,i / (b_b,i + eps)
    var max_fwd_err: Float64 = 0.0
    for b in range(BATCH):
        for i in range(DIM):
            var a_v = Float64(a_arr[b * DIM + i])
            var ap_v = Float64(ap_arr[b * DIM + i])
            var b_v = Float64(b_arr[b * DIM + i])
            var expected = -a_v * ap_v / (b_v + EPS_F)
            var got = Float64(y_arr[b * DIM + i])
            var err = _abs_f64(got - expected)
            if err > max_fwd_err:
                max_fwd_err = err
    print("forward max_err =", max_fwd_err)
    if max_fwd_err > 1e-3:
        raise Error("forward composition failed")

    # ── Backward chain: given upstream grad_y, propagate back ──────────────
    var grad_y_arr = InlineArray[Scalar[dtype], BATCH * DIM](uninitialized=True)
    grad_y_arr[0] = 1.0
    grad_y_arr[1] = -2.0
    grad_y_arr[2] = 0.5
    grad_y_arr[3] = 1.5
    grad_y_arr[4] = -1.0
    grad_y_arr[5] = 0.7
    var grad_y_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](grad_y_arr.unsafe_ptr())

    # NegateOp backward: grad_prod = -grad_y
    var grad_prod_arr = InlineArray[Scalar[dtype], BATCH * DIM](
        uninitialized=True
    )
    var grad_prod_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](grad_prod_arr.unsafe_ptr())
    var grad_params_arr = InlineArray[Scalar[dtype], 1](uninitialized=True)
    var grad_params_t = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](grad_params_arr.unsafe_ptr())
    Neg.vjp[BATCH, dtype](
        grad_y_t, grad_prod_t, pT, neg_cache_t, grad_params_t
    )

    # ElemMulTwoInputOp backward: splits grad into [grad_a | grad_ratio]
    var grad_mul_in_arr = InlineArray[Scalar[dtype], BATCH * 2 * DIM](
        uninitialized=True
    )
    for k in range(BATCH * 2 * DIM):
        grad_mul_in_arr[k] = Scalar[dtype](0.0)
    var grad_mul_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 2 * DIM), MutAnyOrigin
    ](grad_mul_in_arr.unsafe_ptr())
    Mul.vjp[BATCH, dtype](
        grad_prod_t, grad_mul_in_t, pT, mul_cache_t, grad_params_t
    )

    # Extract grad_a and grad_ratio
    var grad_a_arr = InlineArray[Scalar[dtype], BATCH * DIM](uninitialized=True)
    var grad_ratio_arr = InlineArray[Scalar[dtype], BATCH * DIM](
        uninitialized=True
    )
    for k in range(BATCH):
        for i in range(DIM):
            grad_a_arr[k * DIM + i] = grad_mul_in_arr[k * 2 * DIM + i]
            grad_ratio_arr[k * DIM + i] = grad_mul_in_arr[k * 2 * DIM + DIM + i]
    var grad_ratio_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](grad_ratio_arr.unsafe_ptr())

    # DivideOp backward: splits grad_ratio into [grad_a' | grad_b]
    var grad_div_in_arr = InlineArray[Scalar[dtype], BATCH * 2 * DIM](
        uninitialized=True
    )
    for k in range(BATCH * 2 * DIM):
        grad_div_in_arr[k] = Scalar[dtype](0.0)
    var grad_div_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 2 * DIM), MutAnyOrigin
    ](grad_div_in_arr.unsafe_ptr())
    Div.vjp[BATCH, dtype](
        grad_ratio_t, grad_div_in_t, pT, div_cache_t, grad_params_t
    )

    # ── Compare against analytical gradient of y = -a · a' / (b+ε) ────────
    # grad_a = -go · a' / (b+ε)
    # grad_a' = -go · a / (b+ε)
    # grad_b  = +go · a · a' / (b+ε)²
    var max_a_rel: Float64 = 0.0
    var max_ap_rel: Float64 = 0.0
    var max_b_rel: Float64 = 0.0
    for k in range(BATCH):
        for i in range(DIM):
            var a_v = Float64(a_arr[k * DIM + i])
            var ap_v = Float64(ap_arr[k * DIM + i])
            var b_v = Float64(b_arr[k * DIM + i])
            var go = Float64(grad_y_arr[k * DIM + i])
            var bp = b_v + EPS_F

            var exp_ga = -go * ap_v / bp
            var exp_gap = -go * a_v / bp
            var exp_gb = go * a_v * ap_v / (bp * bp)

            var got_ga = Float64(grad_a_arr[k * DIM + i])
            var got_gap = Float64(grad_div_in_arr[k * 2 * DIM + i])
            var got_gb = Float64(grad_div_in_arr[k * 2 * DIM + DIM + i])

            var d_a = _abs_f64(got_ga - exp_ga) / (_abs_f64(exp_ga) + 1e-9)
            var d_ap = _abs_f64(got_gap - exp_gap) / (_abs_f64(exp_gap) + 1e-9)
            var d_b = _abs_f64(got_gb - exp_gb) / (_abs_f64(exp_gb) + 1e-9)

            if d_a > max_a_rel: max_a_rel = d_a
            if d_ap > max_ap_rel: max_ap_rel = d_ap
            if d_b > max_b_rel: max_b_rel = d_b

    print(
        "compose backward rel-err: grad_a max=", max_a_rel,
        " grad_a' max=", max_ap_rel,
        " grad_b max=", max_b_rel,
    )
    if max_a_rel > 1e-4 or max_ap_rel > 1e-4 or max_b_rel > 1e-4:
        raise Error("entropy-ratio composition gradient mismatch")

    print(
        "OK — DivideOp + ElemMulTwoInputOp + NegateOp compose to faithfully"
        " produce the (grad_a, grad_a', grad_b) of y = -a·a'/(b+ε)."
    )


def main() raises:
    test_entropy_ratio_compose()

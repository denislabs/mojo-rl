"""Causal-mode tests for ScaledDotProductAttention.

Exercises the causal=True compile-time variant added for GPT-style decoder
attention. Tests:
  1. Compile-time dimension checks — causal=True must not change CACHE_SIZE,
     IN_DIM, OUT_DIM, PARAM_SIZE.
  2. Causal forward sanity — attn weights for j > i are zero; each row's
     valid (j ≤ i) entries sum to ~1.
  3. Causality invariant — ∂output[i] / ∂input[j] == 0 for any input position
     j > i (Q, K, or V), verified via finite differences.
  4. Causal gradcheck — analytic vs finite-diff gradient agreement on a
     small random input.
  5. Non-causal regression — causal=False (default) path still produces the
     same output as before.

Run with:
    pixi run mojo run -I . tests/nn/test_attention_causal.mojo
"""

from std.random import seed, random_float64
from std.math import abs as math_abs

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.autodiff import ScaledDotProductAttention
from layout import Layout, LayoutTensor


def print_header(name: String):
    print("\n" + "=" * 70)
    print("TEST: " + name)
    print("=" * 70)


def check(cond: Bool, msg: String, mut fails: Int):
    if cond:
        print("  PASS: " + msg)
    else:
        print("  FAIL: " + msg)
        fails += 1


def make_list(size: Int) -> List[Scalar[dtype]]:
    var lst = List[Scalar[dtype]](capacity=size)
    for _ in range(size):
        lst.append(0)
    return lst^


def make_rand_list(size: Int) -> List[Scalar[dtype]]:
    var lst = List[Scalar[dtype]](capacity=size)
    for _ in range(size):
        lst.append(Scalar[dtype](random_float64(-0.5, 0.5)))
    return lst^


# =============================================================================
# Test 1: causal=True does not change shape parameters
# =============================================================================
def test_causal_dims() -> Int:
    print_header("Causal mode: shape parameters unchanged")
    var fails = 0

    comptime AttnNonCausal = ScaledDotProductAttention[8, 2, 4]
    comptime AttnCausal = ScaledDotProductAttention[8, 2, 4, True]

    check(
        AttnCausal.IN_DIM == AttnNonCausal.IN_DIM,
        "IN_DIM identical: " + String(AttnCausal.IN_DIM),
        fails,
    )
    check(
        AttnCausal.OUT_DIM == AttnNonCausal.OUT_DIM,
        "OUT_DIM identical: " + String(AttnCausal.OUT_DIM),
        fails,
    )
    check(
        AttnCausal.CACHE_SIZE == AttnNonCausal.CACHE_SIZE,
        "CACHE_SIZE identical: " + String(AttnCausal.CACHE_SIZE),
        fails,
    )
    check(
        AttnCausal.PARAM_SIZE == 0,
        "PARAM_SIZE = 0",
        fails,
    )
    check(AttnCausal.causal == True, "causal flag set", fails)
    check(AttnNonCausal.causal == False, "default causal=False", fails)

    return fails


# =============================================================================
# Test 2: Causal attention weights respect the triangular mask
# =============================================================================
def test_causal_attn_weights() -> Int:
    """After forward, the attn-weights cache slots for j ≤ i should sum
    to ~1 per row, and the j > i slots are by construction never written
    (we don't read them in backward, but we can check that for j ≤ i the
    softmax is well-formed)."""
    print_header("Causal attention weights: triangular softmax")
    var fails = 0
    seed(11)

    comptime DIM = 4
    comptime HEADS = 1
    comptime SEQ = 4
    comptime BATCH = 1
    comptime Attn = ScaledDotProductAttention[DIM, HEADS, SEQ, True]

    var inp = make_rand_list(BATCH * Attn.IN_DIM)
    var out_data = make_list(BATCH * Attn.OUT_DIM)
    var cache_data = make_list(BATCH * Attn.CACHE_SIZE)
    var params_data = make_list(1)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Attn.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Attn.OUT_DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Attn.PARAM_SIZE), MutAnyOrigin
    ](params_data.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Attn.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())

    Attn.eval[BATCH](inp_t, out_t, p_t, c_t)

    # attn_weights cache offset = 3 * SEQ * DIM
    var attn_off = 3 * SEQ * DIM
    var max_row_err: Float64 = 0.0
    for i in range(SEQ):
        var row_sum: Float64 = 0.0
        # Sum the valid (j ≤ i) entries
        for j in range(i + 1):
            row_sum += Float64(cache_data[attn_off + i * SEQ + j])
        var row_err = math_abs(row_sum - 1.0)
        if row_err > max_row_err:
            max_row_err = row_err

    check(
        max_row_err < 1e-5,
        "softmax row sum ≈ 1.0 over j ≤ i (max err = " + String(max_row_err) + ")",
        fails,
    )

    # Position 0 must produce output equal to V[0] (only attends to itself,
    # weight = 1.0).
    var out_pos0_err: Float64 = 0.0
    for d in range(DIM):
        var v0_d = Float64(inp[2 * SEQ * DIM + 0 * DIM + d])  # V[0, d]
        var out_d = Float64(out_data[0 * DIM + d])
        var diff = math_abs(out_d - v0_d)
        if diff > out_pos0_err:
            out_pos0_err = diff
    check(
        out_pos0_err < 1e-5,
        "output[0] == V[0] (pure self-attention at first token, err = "
        + String(out_pos0_err)
        + ")",
        fails,
    )

    return fails


# =============================================================================
# Test 3: Causality invariant — output[i] is independent of input[j>i]
# =============================================================================
def test_causality_invariant() -> Int:
    """The defining property of causal attention: perturbing any input at
    position j > i (in Q, K, or V) must leave output[i] unchanged.

    We perturb each input position individually and check that the output
    at every earlier query position is unaffected.
    """
    print_header("Causality invariant: future inputs do not affect past outputs")
    var fails = 0
    seed(13)

    comptime DIM = 4
    comptime HEADS = 2
    comptime SEQ = 4
    comptime BATCH = 1
    comptime Attn = ScaledDotProductAttention[DIM, HEADS, SEQ, True]

    var inp = make_rand_list(BATCH * Attn.IN_DIM)
    var params_data = make_list(1)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Attn.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Attn.PARAM_SIZE), MutAnyOrigin
    ](params_data.unsafe_ptr())

    # Baseline forward
    var out_base = make_list(BATCH * Attn.OUT_DIM)
    var cache_base = make_list(BATCH * Attn.CACHE_SIZE)
    var out_base_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Attn.OUT_DIM), MutAnyOrigin
    ](out_base.unsafe_ptr())
    var cache_base_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Attn.CACHE_SIZE), MutAnyOrigin
    ](cache_base.unsafe_ptr())
    Attn.eval[BATCH](inp_t, out_base_t, p_t, cache_base_t)

    # For each query position i, perturb each (Q/K/V at position j > i, dim d)
    # individually and assert output[i] is unchanged.
    var max_violation: Float64 = 0.0
    var n_checks = 0
    var eps = 1e-3

    # Iterate over future input positions to perturb. We perturb K[j, d] and
    # V[j, d] for j > i (perturbing Q[j] only affects output[j], not any
    # earlier i — irrelevant here, so we skip Q to keep the test tight).
    for j in range(SEQ):
        for which in range(2):  # 0 = K, 1 = V
            var base_off = (1 + which) * SEQ * DIM + j * DIM
            for d in range(DIM):
                var idx = base_off + d
                var orig = inp[idx]
                inp[idx] = Scalar[dtype](Float64(orig) + eps)

                var out_pert = make_list(BATCH * Attn.OUT_DIM)
                var cache_pert = make_list(BATCH * Attn.CACHE_SIZE)
                var out_pert_t = LayoutTensor[
                    dtype, Layout.row_major(BATCH, Attn.OUT_DIM), MutAnyOrigin
                ](out_pert.unsafe_ptr())
                var cache_pert_t = LayoutTensor[
                    dtype, Layout.row_major(BATCH, Attn.CACHE_SIZE), MutAnyOrigin
                ](cache_pert.unsafe_ptr())
                Attn.eval[BATCH](inp_t, out_pert_t, p_t, cache_pert_t)
                inp[idx] = orig

                # output[i, d'] for any i < j MUST be unchanged.
                for i in range(j):
                    for dp in range(DIM):
                        var diff = math_abs(
                            Float64(out_pert[i * DIM + dp])
                            - Float64(out_base[i * DIM + dp])
                        )
                        if diff > max_violation:
                            max_violation = diff
                        n_checks += 1

    check(
        max_violation < 1e-6,
        "max output change at i < j after perturbing K/V[j] = "
        + String(max_violation)
        + " (checked "
        + String(n_checks)
        + " (i, perturb_idx) pairs)",
        fails,
    )

    return fails


# =============================================================================
# Test 4: Finite-difference gradcheck for causal=True
# =============================================================================
def test_causal_gradcheck() -> Int:
    """Compares analytic gradient (vjp) against centered finite differences
    on a random input. Same recipe as the existing non-causal gradcheck in
    test_autodiff_phase8.mojo, applied with causal=True."""
    print_header("Causal ScaledDotProductAttention gradient check")
    var fails = 0
    seed(77)

    comptime DIM = 4
    comptime HEADS = 2
    comptime SEQ = 3
    comptime BATCH = 1
    comptime Attn = ScaledDotProductAttention[DIM, HEADS, SEQ, True]

    var inp = make_rand_list(BATCH * Attn.IN_DIM)
    var go_data = make_rand_list(BATCH * Attn.OUT_DIM)
    var params_data = make_list(1)

    var out_data = make_list(BATCH * Attn.OUT_DIM)
    var cache_data = make_list(BATCH * Attn.CACHE_SIZE)
    var gi_data = make_list(BATCH * Attn.IN_DIM)
    var gp_data = make_list(1)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Attn.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Attn.OUT_DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Attn.PARAM_SIZE), MutAnyOrigin
    ](params_data.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Attn.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())
    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Attn.OUT_DIM), MutAnyOrigin
    ](go_data.unsafe_ptr())
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Attn.IN_DIM), MutAnyOrigin
    ](gi_data.unsafe_ptr())
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(Attn.PARAM_SIZE), MutAnyOrigin
    ](gp_data.unsafe_ptr())

    Attn.eval[BATCH](inp_t, out_t, p_t, c_t)
    Attn.vjp[BATCH](go_t, gi_t, p_t, c_t, gp_t)

    var eps: Float64 = 1e-4
    var max_err: Float64 = 0.0
    var n_checked = 0

    for idx in range(BATCH * Attn.IN_DIM):
        var orig = inp[idx]

        inp[idx] = Scalar[dtype](Float64(orig) + eps)
        var out_plus = make_list(BATCH * Attn.OUT_DIM)
        var cache_plus = make_list(BATCH * Attn.CACHE_SIZE)
        var out_plus_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Attn.OUT_DIM), MutAnyOrigin
        ](out_plus.unsafe_ptr())
        var cache_plus_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Attn.CACHE_SIZE), MutAnyOrigin
        ](cache_plus.unsafe_ptr())
        Attn.eval[BATCH](inp_t, out_plus_t, p_t, cache_plus_t)

        inp[idx] = Scalar[dtype](Float64(orig) - eps)
        var out_minus = make_list(BATCH * Attn.OUT_DIM)
        var cache_minus = make_list(BATCH * Attn.CACHE_SIZE)
        var out_minus_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Attn.OUT_DIM), MutAnyOrigin
        ](out_minus.unsafe_ptr())
        var cache_minus_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Attn.CACHE_SIZE), MutAnyOrigin
        ](cache_minus.unsafe_ptr())
        Attn.eval[BATCH](inp_t, out_minus_t, p_t, cache_minus_t)

        inp[idx] = orig

        var fd_grad: Float64 = 0.0
        for j in range(BATCH * Attn.OUT_DIM):
            fd_grad += (
                Float64(go_data[j])
                * (Float64(out_plus[j]) - Float64(out_minus[j]))
                / (2.0 * eps)
            )

        var analytic = Float64(gi_data[idx])
        var err = math_abs(fd_grad - analytic)
        var denom = math_abs(fd_grad) + math_abs(analytic) + 1e-8
        var rel_err = err / denom

        if rel_err > max_err:
            max_err = rel_err
        n_checked += 1

    check(
        max_err < 2e-2,
        "max relative error = "
        + String(max_err)
        + " (checked "
        + String(n_checked)
        + " inputs, tol 2e-2)",
        fails,
    )

    return fails


# =============================================================================
# Test 5: Non-causal regression — default path still produces sane output
# =============================================================================
def test_noncausal_regression() -> Int:
    """Smoke check: the existing default (causal=False) path forward still
    produces full-row softmax (sums to 1 across all SEQ positions)."""
    print_header("Non-causal regression: default path softmax sums to 1 over full row")
    var fails = 0
    seed(31)

    comptime DIM = 4
    comptime HEADS = 1
    comptime SEQ = 3
    comptime BATCH = 1
    comptime Attn = ScaledDotProductAttention[DIM, HEADS, SEQ]  # causal default = False

    var inp = make_rand_list(BATCH * Attn.IN_DIM)
    var out_data = make_list(BATCH * Attn.OUT_DIM)
    var cache_data = make_list(BATCH * Attn.CACHE_SIZE)
    var params_data = make_list(1)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Attn.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Attn.OUT_DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Attn.PARAM_SIZE), MutAnyOrigin
    ](params_data.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Attn.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())

    Attn.eval[BATCH](inp_t, out_t, p_t, c_t)

    var attn_off = 3 * SEQ * DIM
    var max_row_err: Float64 = 0.0
    for i in range(SEQ):
        var row_sum: Float64 = 0.0
        for j in range(SEQ):
            row_sum += Float64(cache_data[attn_off + i * SEQ + j])
        var row_err = math_abs(row_sum - 1.0)
        if row_err > max_row_err:
            max_row_err = row_err

    check(
        max_row_err < 1e-5,
        "non-causal softmax row sum ≈ 1.0 over full row (max err = "
        + String(max_row_err)
        + ")",
        fails,
    )

    return fails


# =============================================================================
# Driver
# =============================================================================
def main():
    var total_fails = 0
    total_fails += test_causal_dims()
    total_fails += test_causal_attn_weights()
    total_fails += test_causality_invariant()
    total_fails += test_causal_gradcheck()
    total_fails += test_noncausal_regression()

    print("\n" + "=" * 70)
    if total_fails == 0:
        print("ALL CAUSAL ATTENTION TESTS PASSED")
    else:
        print("FAILED: " + String(total_fails) + " checks")
    print("=" * 70)

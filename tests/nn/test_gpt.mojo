"""Tests for the GPT composite (causal char-level transformer).

Verifies:
  1. Shape parameters propagate through Embedding + pos-embed + n × TransformerBlock + LN + LM head.
  2. Forward runs without crashing on a small config and produces non-NaN logits.
  3. Causality invariant on the full GPT — perturbing the one-hot input at
     position t > i must not change the logits at any position i ≤ t-1
     (defining property of a causal LM).
  4. Finite-difference gradcheck end-to-end on a tiny config.

Run:
    pixi run mojo run -I . tests/nn/test_gpt.mojo
"""

from std.random import seed, random_float64
from std.math import abs as math_abs

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.composites import GPT
from layout import Layout, LayoutTensor


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


def check(cond: Bool, msg: String, mut fails: Int):
    if cond:
        print("  PASS: " + msg)
    else:
        print("  FAIL: " + msg)
        fails += 1


def print_header(name: String):
    print("\n" + "=" * 70)
    print("TEST: " + name)
    print("=" * 70)


# Build a (BATCH, seq_len * vocab) one-hot tensor from token indices.
# token_ids has shape [BATCH * seq_len]; row-major outer-batch-then-token.
def make_one_hot(
    token_ids: List[Int], vocab: Int, batch: Int, seq_len: Int
) -> List[Scalar[dtype]]:
    var oh = make_list(batch * seq_len * vocab)
    for b in range(batch):
        for t in range(seq_len):
            var token = token_ids[b * seq_len + t]
            oh[(b * seq_len + t) * vocab + token] = Scalar[dtype](1.0)
    return oh^


def random_token_ids(batch: Int, seq_len: Int, vocab: Int) -> List[Int]:
    var ids = List[Int](capacity=batch * seq_len)
    for _ in range(batch * seq_len):
        var u = random_float64(0.0, 1.0)
        var idx = Int(u * Float64(vocab))
        if idx >= vocab:
            idx = vocab - 1
        ids.append(idx)
    return ids^


# =============================================================================
# Test 1: shape parameters
# =============================================================================
def test_dims() -> Int:
    print_header("GPT: shape parameters")
    var fails = 0

    comptime V = 16
    comptime S = 4
    comptime D = 8
    comptime H = 2
    comptime N = 2

    comptime Model = GPT[V, S, D, H, N]

    check(
        Model.IN_DIM == S * V,
        "IN_DIM = seq_len * vocab = " + String(Model.IN_DIM),
        fails,
    )
    check(
        Model.OUT_DIM == S * V,
        "OUT_DIM = seq_len * vocab = " + String(Model.OUT_DIM),
        fails,
    )
    check(Model.PARAM_SIZE > 0, "PARAM_SIZE > 0: " + String(Model.PARAM_SIZE), fails)

    return fails


# =============================================================================
# Test 2: forward smoke
# =============================================================================
def test_forward_runs() -> Int:
    print_header("GPT: forward smoke (random one-hot inputs)")
    var fails = 0
    seed(11)

    comptime V = 12
    comptime S = 5
    comptime D = 8
    comptime H = 2
    comptime N = 2
    comptime BATCH = 2
    comptime Model = GPT[V, S, D, H, N]

    var ids = random_token_ids(BATCH, S, V)
    var inp = make_one_hot(ids, V, BATCH, S)
    var params = make_rand_list(Model.PARAM_SIZE)
    # Scale params down so the stack doesn't saturate the LayerNorms.
    for i in range(Model.PARAM_SIZE):
        params[i] = Scalar[dtype](Float64(params[i]) * 0.1)

    var out_data = make_list(BATCH * Model.OUT_DIM)
    var cache_data = make_list(BATCH * Model.CACHE_SIZE)
    var state = make_list(Model.STATE_SIZE if Model.STATE_SIZE > 0 else 1)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.OUT_DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Model.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var s_t = LayoutTensor[
        dtype, Layout.row_major(Model.STATE_SIZE), MutAnyOrigin
    ](state.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())

    Model.forward[BATCH, dtype](inp_t, out_t, p_t, s_t, c_t)

    var has_nan = False
    var any_nonzero = False
    for i in range(BATCH * Model.OUT_DIM):
        var v = Float64(out_data[i])
        if v != v:
            has_nan = True
        if math_abs(v) > 1e-6:
            any_nonzero = True
    check(not has_nan, "logits contain no NaN", fails)
    check(any_nonzero, "logits are non-trivial (non-zero)", fails)

    return fails


# =============================================================================
# Test 3: causality invariant on the full GPT
# =============================================================================
def test_causality() -> Int:
    """Defining property of a causal LM: perturbing the input at position t
    must not change the logits at any position i < t."""
    print_header("GPT: causality invariant (logits at i < t unaffected by input[t])")
    var fails = 0
    seed(13)

    comptime V = 8
    comptime S = 4
    comptime D = 8
    comptime H = 2
    comptime N = 1  # one block is enough to verify causality threading
    comptime BATCH = 1
    comptime Model = GPT[V, S, D, H, N]

    var ids = random_token_ids(BATCH, S, V)
    var inp_base = make_one_hot(ids, V, BATCH, S)
    var params = make_rand_list(Model.PARAM_SIZE)
    for i in range(Model.PARAM_SIZE):
        params[i] = Scalar[dtype](Float64(params[i]) * 0.1)
    var state = make_list(Model.STATE_SIZE if Model.STATE_SIZE > 0 else 1)

    # Baseline
    var out_base = make_list(BATCH * Model.OUT_DIM)
    var cache_base = make_list(BATCH * Model.CACHE_SIZE)
    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.IN_DIM), MutAnyOrigin
    ](inp_base.unsafe_ptr())
    var out_base_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.OUT_DIM), MutAnyOrigin
    ](out_base.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Model.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var s_t = LayoutTensor[
        dtype, Layout.row_major(Model.STATE_SIZE), MutAnyOrigin
    ](state.unsafe_ptr())
    var c_base_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.CACHE_SIZE), MutAnyOrigin
    ](cache_base.unsafe_ptr())
    Model.forward[BATCH, dtype](inp_t, out_base_t, p_t, s_t, c_base_t)

    # Swap the token at position t (for each t > 0) to a different token,
    # then verify logits at positions < t are unchanged.
    var max_violation: Float64 = 0.0
    var n_checks = 0
    for t in range(1, S):
        # Build a perturbed input by changing the one-hot at position t to a
        # different token. Pick (current+1) mod V as the new token.
        var inp_pert = List[Scalar[dtype]](capacity=BATCH * Model.IN_DIM)
        for i in range(BATCH * Model.IN_DIM):
            inp_pert.append(inp_base[i])
        var orig_tok = ids[t]
        var new_tok = (orig_tok + 1) % V
        var pos_offset = t * V  # batch=0, position t
        inp_pert[pos_offset + orig_tok] = Scalar[dtype](0.0)
        inp_pert[pos_offset + new_tok] = Scalar[dtype](1.0)

        var out_pert = make_list(BATCH * Model.OUT_DIM)
        var cache_pert = make_list(BATCH * Model.CACHE_SIZE)
        var inp_pert_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Model.IN_DIM), MutAnyOrigin
        ](inp_pert.unsafe_ptr())
        var out_pert_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Model.OUT_DIM), MutAnyOrigin
        ](out_pert.unsafe_ptr())
        var c_pert_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Model.CACHE_SIZE), MutAnyOrigin
        ](cache_pert.unsafe_ptr())
        Model.forward[BATCH, dtype](inp_pert_t, out_pert_t, p_t, s_t, c_pert_t)

        # Logits at positions [0, t) must match exactly.
        for i in range(t):
            for v in range(V):
                var b_idx = i * V + v
                var d = math_abs(
                    Float64(out_base[b_idx]) - Float64(out_pert[b_idx])
                )
                if d > max_violation:
                    max_violation = d
                n_checks += 1

    check(
        max_violation < 1e-5,
        "max logit change at i < t after swapping input[t] = "
        + String(max_violation)
        + " (checked "
        + String(n_checks)
        + " (i, vocab) pairs)",
        fails,
    )

    return fails


# =============================================================================
# Test 4: end-to-end gradcheck (tiny dims)
# =============================================================================
def test_gradcheck() -> Int:
    print_header("GPT: end-to-end finite-difference gradcheck (input)")
    var fails = 0
    seed(101)

    # Very small config — gradcheck cost is O(IN_DIM * OUT_DIM).
    comptime V = 4
    comptime S = 3
    comptime D = 4
    comptime H = 2
    comptime N = 1
    comptime BATCH = 1
    comptime Model = GPT[V, S, D, H, N]

    # Use a soft (non-one-hot) input: gradcheck differentiates the input,
    # which is the one-hot tensor here. Smooth perturbations in any direction
    # are valid for gradient-checking the embedding's matmul-style backward.
    var inp = make_rand_list(BATCH * Model.IN_DIM)
    var params = make_rand_list(Model.PARAM_SIZE)
    for i in range(Model.PARAM_SIZE):
        params[i] = Scalar[dtype](Float64(params[i]) * 0.1)
    var go = make_rand_list(BATCH * Model.OUT_DIM)
    var state = make_list(Model.STATE_SIZE if Model.STATE_SIZE > 0 else 1)

    var out_data = make_list(BATCH * Model.OUT_DIM)
    var cache_data = make_list(BATCH * Model.CACHE_SIZE)
    var gi_data = make_list(BATCH * Model.IN_DIM)
    var gp_data = make_list(Model.PARAM_SIZE)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.OUT_DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Model.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var s_t = LayoutTensor[
        dtype, Layout.row_major(Model.STATE_SIZE), MutAnyOrigin
    ](state.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())
    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.OUT_DIM), MutAnyOrigin
    ](go.unsafe_ptr())
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Model.IN_DIM), MutAnyOrigin
    ](gi_data.unsafe_ptr())
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(Model.PARAM_SIZE), MutAnyOrigin
    ](gp_data.unsafe_ptr())

    Model.forward[BATCH, dtype](inp_t, out_t, p_t, s_t, c_t)
    Model.backward[BATCH, dtype](go_t, gi_t, p_t, s_t, c_t, gp_t)

    var eps: Float64 = 1e-3
    var max_in_err: Float64 = 0.0

    for idx in range(BATCH * Model.IN_DIM):
        var orig = inp[idx]

        inp[idx] = Scalar[dtype](Float64(orig) + eps)
        var out_plus = make_list(BATCH * Model.OUT_DIM)
        var cache_plus = make_list(BATCH * Model.CACHE_SIZE)
        var op_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Model.OUT_DIM), MutAnyOrigin
        ](out_plus.unsafe_ptr())
        var cp_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Model.CACHE_SIZE), MutAnyOrigin
        ](cache_plus.unsafe_ptr())
        Model.forward[BATCH, dtype](inp_t, op_t, p_t, s_t, cp_t)

        inp[idx] = Scalar[dtype](Float64(orig) - eps)
        var out_minus = make_list(BATCH * Model.OUT_DIM)
        var cache_minus = make_list(BATCH * Model.CACHE_SIZE)
        var om_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Model.OUT_DIM), MutAnyOrigin
        ](out_minus.unsafe_ptr())
        var cm_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Model.CACHE_SIZE), MutAnyOrigin
        ](cache_minus.unsafe_ptr())
        Model.forward[BATCH, dtype](inp_t, om_t, p_t, s_t, cm_t)

        inp[idx] = orig

        var fd: Float64 = 0.0
        for j in range(BATCH * Model.OUT_DIM):
            fd += (
                Float64(go[j])
                * (Float64(out_plus[j]) - Float64(out_minus[j]))
                / (2.0 * eps)
            )
        var an = Float64(gi_data[idx])
        var err = math_abs(fd - an)
        if math_abs(fd) < 1e-4 and math_abs(an) < 1e-4:
            continue
        var denom = math_abs(fd) + math_abs(an) + 1e-8
        var rel = err / denom
        if rel > max_in_err:
            max_in_err = rel

    check(
        max_in_err < 5e-2,
        "max relative error on grad_input = " + String(max_in_err) + " (tol 5e-2)",
        fails,
    )

    return fails


def main():
    var total_fails = 0
    total_fails += test_dims()
    total_fails += test_forward_runs()
    total_fails += test_causality()
    total_fails += test_gradcheck()

    print("\n" + "=" * 70)
    if total_fails == 0:
        print("ALL GPT TESTS PASSED")
    else:
        print("FAILED: " + String(total_fails) + " checks")
    print("=" * 70)

"""Smoke tests for the Tokenwise combinator.

Verifies:
  1. Compile-time shape parameters are correct.
  2. Forward equivalence: Tokenwise[seq_len, Inner] applied to a (BATCH, S*D)
     input produces per-token Inner outputs, equivalent to running Inner at
     batch size BATCH * seq_len with shared weights.
  3. Backward equivalence: gradients match running Inner directly at
     BATCH * seq_len.

Run:
    pixi run mojo run -I . tests/nn/test_tokenwise.mojo
"""

from std.random import seed, random_float64
from std.math import abs as math_abs

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Linear, Tokenwise
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


def test_tokenwise_dims() -> Int:
    print_header("Tokenwise: shape parameters")
    var fails = 0

    comptime SEQ = 3
    comptime IN = 4
    comptime OUT = 5
    comptime InnerLin = Linear[IN, OUT]
    comptime TW = Tokenwise[SEQ, InnerLin]

    check(
        TW.IN_DIM == SEQ * IN,
        "IN_DIM = SEQ*IN = " + String(TW.IN_DIM),
        fails,
    )
    check(
        TW.OUT_DIM == SEQ * OUT,
        "OUT_DIM = SEQ*OUT = " + String(TW.OUT_DIM),
        fails,
    )
    check(
        TW.PARAM_SIZE == InnerLin.PARAM_SIZE,
        "PARAM_SIZE shared with Inner = " + String(TW.PARAM_SIZE),
        fails,
    )
    check(
        TW.CACHE_SIZE == SEQ * InnerLin.CACHE_SIZE,
        "CACHE_SIZE = SEQ * Inner.CACHE_SIZE = " + String(TW.CACHE_SIZE),
        fails,
    )

    return fails


def test_tokenwise_vs_inner_forward() -> Int:
    """Forward Tokenwise on (BATCH, S*D_in) must equal Inner on (BATCH*S, D_in)
    bitwise (same memory layout, same params)."""
    print_header("Tokenwise forward equivalence with Inner at batch=BATCH*S")
    var fails = 0
    seed(42)

    comptime SEQ = 3
    comptime IN = 4
    comptime OUT = 5
    comptime BATCH = 2
    comptime InnerLin = Linear[IN, OUT]
    comptime TW = Tokenwise[SEQ, InnerLin]

    var inp = make_rand_list(BATCH * TW.IN_DIM)
    var params = make_rand_list(TW.PARAM_SIZE)

    # --- Run via Tokenwise ---
    var tw_out = make_list(BATCH * TW.OUT_DIM)
    var tw_cache = make_list(BATCH * TW.CACHE_SIZE)
    var tw_state = make_list(1)

    var tw_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, TW.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var tw_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, TW.OUT_DIM), MutAnyOrigin
    ](tw_out.unsafe_ptr())
    var tw_p_t = LayoutTensor[
        dtype, Layout.row_major(TW.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var tw_s_t = LayoutTensor[
        dtype, Layout.row_major(TW.STATE_SIZE), MutAnyOrigin
    ](tw_state.unsafe_ptr())
    var tw_c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, TW.CACHE_SIZE), MutAnyOrigin
    ](tw_cache.unsafe_ptr())

    TW.forward[BATCH, dtype](tw_in_t, tw_out_t, tw_p_t, tw_s_t, tw_c_t)

    # --- Run Inner directly at expanded batch ---
    var ref_out = make_list(BATCH * SEQ * OUT)
    var ref_cache = make_list(BATCH * SEQ * InnerLin.CACHE_SIZE)
    var ref_state = make_list(1)

    var ref_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH * SEQ, IN), MutAnyOrigin
    ](inp.unsafe_ptr())
    var ref_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH * SEQ, OUT), MutAnyOrigin
    ](ref_out.unsafe_ptr())
    var ref_p_t = LayoutTensor[
        dtype, Layout.row_major(InnerLin.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var ref_s_t = LayoutTensor[
        dtype, Layout.row_major(InnerLin.STATE_SIZE), MutAnyOrigin
    ](ref_state.unsafe_ptr())
    var ref_c_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH * SEQ, InnerLin.CACHE_SIZE),
        MutAnyOrigin,
    ](ref_cache.unsafe_ptr())

    InnerLin.forward[BATCH * SEQ, dtype](
        ref_in_t, ref_out_t, ref_p_t, ref_s_t, ref_c_t
    )

    # Outputs must match bitwise (same dtype, same memory layout)
    var max_err: Float64 = 0.0
    for i in range(BATCH * TW.OUT_DIM):
        var err = math_abs(Float64(tw_out[i]) - Float64(ref_out[i]))
        if err > max_err:
            max_err = err
    check(
        max_err < 1e-7,
        "Tokenwise vs Inner forward max diff = " + String(max_err),
        fails,
    )

    return fails


def test_tokenwise_backward() -> Int:
    """Tokenwise backward should accumulate gradients identically to running
    Inner at BATCH*S, since shared weights ⇒ summed gradient over all tokens."""
    print_header("Tokenwise backward gradient equivalence")
    var fails = 0
    seed(99)

    comptime SEQ = 3
    comptime IN = 4
    comptime OUT = 5
    comptime BATCH = 2
    comptime InnerLin = Linear[IN, OUT]
    comptime TW = Tokenwise[SEQ, InnerLin]

    var inp = make_rand_list(BATCH * TW.IN_DIM)
    var params = make_rand_list(TW.PARAM_SIZE)
    var go = make_rand_list(BATCH * TW.OUT_DIM)

    # --- TW path ---
    var tw_out = make_list(BATCH * TW.OUT_DIM)
    var tw_cache = make_list(BATCH * TW.CACHE_SIZE)
    var tw_state = make_list(1)
    var tw_gi = make_list(BATCH * TW.IN_DIM)
    var tw_gp = make_list(TW.PARAM_SIZE)

    var tw_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, TW.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var tw_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, TW.OUT_DIM), MutAnyOrigin
    ](tw_out.unsafe_ptr())
    var tw_p_t = LayoutTensor[
        dtype, Layout.row_major(TW.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var tw_s_t = LayoutTensor[
        dtype, Layout.row_major(TW.STATE_SIZE), MutAnyOrigin
    ](tw_state.unsafe_ptr())
    var tw_c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, TW.CACHE_SIZE), MutAnyOrigin
    ](tw_cache.unsafe_ptr())
    var tw_go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, TW.OUT_DIM), MutAnyOrigin
    ](go.unsafe_ptr())
    var tw_gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, TW.IN_DIM), MutAnyOrigin
    ](tw_gi.unsafe_ptr())
    var tw_gp_t = LayoutTensor[
        dtype, Layout.row_major(TW.PARAM_SIZE), MutAnyOrigin
    ](tw_gp.unsafe_ptr())

    TW.forward[BATCH, dtype](tw_in_t, tw_out_t, tw_p_t, tw_s_t, tw_c_t)
    TW.backward[BATCH, dtype](
        tw_go_t, tw_gi_t, tw_p_t, tw_s_t, tw_c_t, tw_gp_t
    )

    # --- Inner path at BATCH*SEQ ---
    var ref_out = make_list(BATCH * SEQ * OUT)
    var ref_cache = make_list(BATCH * SEQ * InnerLin.CACHE_SIZE)
    var ref_state = make_list(1)
    var ref_gi = make_list(BATCH * SEQ * IN)
    var ref_gp = make_list(InnerLin.PARAM_SIZE)

    var ref_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH * SEQ, IN), MutAnyOrigin
    ](inp.unsafe_ptr())
    var ref_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH * SEQ, OUT), MutAnyOrigin
    ](ref_out.unsafe_ptr())
    var ref_p_t = LayoutTensor[
        dtype, Layout.row_major(InnerLin.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var ref_s_t = LayoutTensor[
        dtype, Layout.row_major(InnerLin.STATE_SIZE), MutAnyOrigin
    ](ref_state.unsafe_ptr())
    var ref_c_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH * SEQ, InnerLin.CACHE_SIZE),
        MutAnyOrigin,
    ](ref_cache.unsafe_ptr())
    var ref_go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH * SEQ, OUT), MutAnyOrigin
    ](go.unsafe_ptr())
    var ref_gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH * SEQ, IN), MutAnyOrigin
    ](ref_gi.unsafe_ptr())
    var ref_gp_t = LayoutTensor[
        dtype, Layout.row_major(InnerLin.PARAM_SIZE), MutAnyOrigin
    ](ref_gp.unsafe_ptr())

    InnerLin.forward[BATCH * SEQ, dtype](
        ref_in_t, ref_out_t, ref_p_t, ref_s_t, ref_c_t
    )
    InnerLin.backward[BATCH * SEQ, dtype](
        ref_go_t, ref_gi_t, ref_p_t, ref_s_t, ref_c_t, ref_gp_t
    )

    var max_gi_err: Float64 = 0.0
    for i in range(BATCH * TW.IN_DIM):
        var err = math_abs(Float64(tw_gi[i]) - Float64(ref_gi[i]))
        if err > max_gi_err:
            max_gi_err = err
    check(
        max_gi_err < 1e-7,
        "grad_input bitwise match: max diff = " + String(max_gi_err),
        fails,
    )

    var max_gp_err: Float64 = 0.0
    for i in range(TW.PARAM_SIZE):
        var err = math_abs(Float64(tw_gp[i]) - Float64(ref_gp[i]))
        if err > max_gp_err:
            max_gp_err = err
    check(
        max_gp_err < 1e-7,
        "grad_params bitwise match: max diff = " + String(max_gp_err),
        fails,
    )

    return fails


def main():
    var total_fails = 0
    total_fails += test_tokenwise_dims()
    total_fails += test_tokenwise_vs_inner_forward()
    total_fails += test_tokenwise_backward()

    print("\n" + "=" * 70)
    if total_fails == 0:
        print("ALL TOKENWISE TESTS PASSED")
    else:
        print("FAILED: " + String(total_fails) + " checks")
    print("=" * 70)

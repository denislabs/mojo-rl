"""Tests for the TransformerBlock + MultiHeadAttention composites.

Verifies:
  1. Shape parameters propagate correctly through Tokenwise + Sequential +
     Residual + AutoDiffChain[ScaledDotProductAttention].
  2. Forward runs without crashing on a small block (sanity).
  3. Causal vs non-causal blocks differ on the same input (sanity that the
     causal flag actually threads through to the attention op).
  4. Finite-difference gradcheck on the full block end-to-end (small dims).

Run:
    pixi run mojo run -I . tests/nn/test_transformer_block.mojo
"""

from std.random import seed, random_float64
from std.math import abs as math_abs

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.composites import (
    MultiHeadAttention,
    TransformerFFN,
    TransformerBlock,
)
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


# =============================================================================
# Test 1: dimension checks
# =============================================================================
def test_dims() -> Int:
    print_header("TransformerBlock: shape parameters")
    var fails = 0

    comptime DIM = 8
    comptime HEADS = 2
    comptime SEQ = 4
    comptime FF = 16
    comptime SD = SEQ * DIM

    comptime MHA = MultiHeadAttention[DIM, HEADS, SEQ]
    comptime FFN_T = TransformerFFN[SEQ, DIM, FF]
    comptime Block = TransformerBlock[DIM, HEADS, SEQ, FF]
    comptime BlockCausal = TransformerBlock[DIM, HEADS, SEQ, FF, True]

    check(MHA.IN_DIM == SD, "MHA.IN_DIM = SEQ*DIM = " + String(MHA.IN_DIM), fails)
    check(MHA.OUT_DIM == SD, "MHA.OUT_DIM = SEQ*DIM = " + String(MHA.OUT_DIM), fails)
    check(FFN_T.IN_DIM == SD, "FFN.IN_DIM = SEQ*DIM = " + String(FFN_T.IN_DIM), fails)
    check(FFN_T.OUT_DIM == SD, "FFN.OUT_DIM = SEQ*DIM = " + String(FFN_T.OUT_DIM), fails)
    check(Block.IN_DIM == SD, "Block.IN_DIM = SEQ*DIM = " + String(Block.IN_DIM), fails)
    check(Block.OUT_DIM == SD, "Block.OUT_DIM = SEQ*DIM = " + String(Block.OUT_DIM), fails)
    check(
        BlockCausal.IN_DIM == SD,
        "BlockCausal.IN_DIM = SEQ*DIM = " + String(BlockCausal.IN_DIM),
        fails,
    )
    # Causal block has identical PARAM_SIZE / CACHE_SIZE — only attention's
    # internal compute differs.
    check(
        BlockCausal.PARAM_SIZE == Block.PARAM_SIZE,
        "Causal vs non-causal PARAM_SIZE identical: " + String(Block.PARAM_SIZE),
        fails,
    )
    check(
        BlockCausal.CACHE_SIZE == Block.CACHE_SIZE,
        "Causal vs non-causal CACHE_SIZE identical: " + String(Block.CACHE_SIZE),
        fails,
    )

    # Param count sanity:
    #   MHA: QKV proj (D * 3D + 3D) + out proj (D*D + D)  = 4D² + 4D
    #   FFN: D * FF + FF + FF * D + D                     = 2 D FF + FF + D
    #   2 LayerNorms: 2 * 2 * D                            = 4 D
    var expected_mha_params = 4 * DIM * DIM + 4 * DIM
    var expected_ffn_params = 2 * DIM * FF + FF + DIM
    var expected_ln_params = 4 * DIM
    var expected_block_params = expected_mha_params + expected_ffn_params + expected_ln_params

    # Block params may include AutoFused/Sequential alignment padding; assert
    # at least the expected lower bound.
    check(
        Block.PARAM_SIZE >= expected_block_params,
        "Block PARAM_SIZE >= expected ("
        + String(expected_block_params)
        + "), got "
        + String(Block.PARAM_SIZE),
        fails,
    )

    return fails


# =============================================================================
# Test 2: forward smoke + causal differs from non-causal
# =============================================================================
def test_forward_runs() -> Int:
    print_header("TransformerBlock: forward smoke (causal vs non-causal)")
    var fails = 0
    seed(7)

    comptime DIM = 8
    comptime HEADS = 2
    comptime SEQ = 4
    comptime FF = 16
    comptime BATCH = 2

    comptime Block = TransformerBlock[DIM, HEADS, SEQ, FF]
    comptime BlockCausal = TransformerBlock[DIM, HEADS, SEQ, FF, True]

    var inp = make_rand_list(BATCH * Block.IN_DIM)
    var params = make_rand_list(Block.PARAM_SIZE)
    var state = make_list(Block.STATE_SIZE if Block.STATE_SIZE > 0 else 1)

    # Non-causal forward
    var out_nc = make_list(BATCH * Block.OUT_DIM)
    var cache_nc = make_list(BATCH * Block.CACHE_SIZE)
    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Block.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_nc_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Block.OUT_DIM), MutAnyOrigin
    ](out_nc.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Block.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var s_t = LayoutTensor[
        dtype, Layout.row_major(Block.STATE_SIZE), MutAnyOrigin
    ](state.unsafe_ptr())
    var c_nc_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Block.CACHE_SIZE), MutAnyOrigin
    ](cache_nc.unsafe_ptr())
    Block.forward[BATCH, dtype](inp_t, out_nc_t, p_t, s_t, c_nc_t)

    # Causal forward — same underlying params/input/state buffers, but
    # rebound to the BlockCausal-typed LayoutTensor view (the two block types
    # have identical PARAM_SIZE/STATE_SIZE values but are nominally distinct).
    var out_c = make_list(BATCH * BlockCausal.OUT_DIM)
    var cache_c = make_list(BATCH * BlockCausal.CACHE_SIZE)
    var inp_c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, BlockCausal.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, BlockCausal.OUT_DIM), MutAnyOrigin
    ](out_c.unsafe_ptr())
    var p_c_t = LayoutTensor[
        dtype, Layout.row_major(BlockCausal.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var s_c_t = LayoutTensor[
        dtype, Layout.row_major(BlockCausal.STATE_SIZE), MutAnyOrigin
    ](state.unsafe_ptr())
    var c_c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, BlockCausal.CACHE_SIZE), MutAnyOrigin
    ](cache_c.unsafe_ptr())
    BlockCausal.forward[BATCH, dtype](inp_c_t, out_c_t, p_c_t, s_c_t, c_c_t)

    # Outputs should differ at non-first positions (causal mask hides future).
    # We don't expect them to differ at position 0 by much but should differ
    # somewhere. Check max absolute difference across the batch.
    var max_diff: Float64 = 0.0
    for i in range(BATCH * Block.OUT_DIM):
        var d = math_abs(Float64(out_nc[i]) - Float64(out_c[i]))
        if d > max_diff:
            max_diff = d
    check(
        max_diff > 1e-4,
        "Causal output differs from non-causal (max abs diff = "
        + String(max_diff)
        + ")",
        fails,
    )

    # Sanity: no NaN/Inf in either output.
    var has_nan = False
    for i in range(BATCH * Block.OUT_DIM):
        var v = Float64(out_nc[i])
        if v != v:  # NaN check
            has_nan = True
        v = Float64(out_c[i])
        if v != v:
            has_nan = True
    check(not has_nan, "outputs contain no NaN", fails)

    return fails


# =============================================================================
# Test 3: gradcheck end-to-end (tiny config)
# =============================================================================
def test_gradcheck() -> Int:
    """Finite-diff gradcheck on the full TransformerBlock with random weights
    and random upstream gradient."""
    print_header("TransformerBlock: end-to-end finite-difference gradcheck")
    var fails = 0
    seed(123)

    # Keep tiny — the inner loops scale with BATCH * SEQ * DIM and a 2x forward
    # per parameter.
    comptime DIM = 4
    comptime HEADS = 2
    comptime SEQ = 3
    comptime FF = 8
    comptime BATCH = 1
    comptime Block = TransformerBlock[DIM, HEADS, SEQ, FF]

    var inp = make_rand_list(BATCH * Block.IN_DIM)
    var params = make_rand_list(Block.PARAM_SIZE)
    var go = make_rand_list(BATCH * Block.OUT_DIM)
    var state = make_list(Block.STATE_SIZE if Block.STATE_SIZE > 0 else 1)

    # Analytic forward + backward
    var out_data = make_list(BATCH * Block.OUT_DIM)
    var cache_data = make_list(BATCH * Block.CACHE_SIZE)
    var gi_data = make_list(BATCH * Block.IN_DIM)
    var gp_data = make_list(Block.PARAM_SIZE)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Block.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Block.OUT_DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Block.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var s_t = LayoutTensor[
        dtype, Layout.row_major(Block.STATE_SIZE), MutAnyOrigin
    ](state.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Block.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())
    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Block.OUT_DIM), MutAnyOrigin
    ](go.unsafe_ptr())
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Block.IN_DIM), MutAnyOrigin
    ](gi_data.unsafe_ptr())
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(Block.PARAM_SIZE), MutAnyOrigin
    ](gp_data.unsafe_ptr())

    Block.forward[BATCH, dtype](inp_t, out_t, p_t, s_t, c_t)
    Block.backward[BATCH, dtype](go_t, gi_t, p_t, s_t, c_t, gp_t)

    var eps: Float64 = 1e-4
    var max_in_err: Float64 = 0.0

    # Gradcheck w.r.t. inputs.
    for idx in range(BATCH * Block.IN_DIM):
        var orig = inp[idx]

        inp[idx] = Scalar[dtype](Float64(orig) + eps)
        var out_plus = make_list(BATCH * Block.OUT_DIM)
        var cache_plus = make_list(BATCH * Block.CACHE_SIZE)
        var out_plus_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Block.OUT_DIM), MutAnyOrigin
        ](out_plus.unsafe_ptr())
        var c_plus_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Block.CACHE_SIZE), MutAnyOrigin
        ](cache_plus.unsafe_ptr())
        Block.forward[BATCH, dtype](inp_t, out_plus_t, p_t, s_t, c_plus_t)

        inp[idx] = Scalar[dtype](Float64(orig) - eps)
        var out_minus = make_list(BATCH * Block.OUT_DIM)
        var cache_minus = make_list(BATCH * Block.CACHE_SIZE)
        var out_minus_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Block.OUT_DIM), MutAnyOrigin
        ](out_minus.unsafe_ptr())
        var c_minus_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Block.CACHE_SIZE), MutAnyOrigin
        ](cache_minus.unsafe_ptr())
        Block.forward[BATCH, dtype](inp_t, out_minus_t, p_t, s_t, c_minus_t)

        inp[idx] = orig

        var fd: Float64 = 0.0
        for j in range(BATCH * Block.OUT_DIM):
            fd += (
                Float64(go[j])
                * (Float64(out_plus[j]) - Float64(out_minus[j]))
                / (2.0 * eps)
            )
        var an = Float64(gi_data[idx])
        var err = math_abs(fd - an)
        var denom = math_abs(fd) + math_abs(an) + 1e-8
        var rel = err / denom
        if rel > max_in_err:
            max_in_err = rel

    check(
        max_in_err < 5e-2,
        "max relative error on grad_input = " + String(max_in_err) + " (tol 5e-2)",
        fails,
    )

    # Param gradcheck: every parameter index, larger eps to reduce fp32 cancellation noise.
    # Use combined absolute+relative tolerance — for params with tiny true
    # gradients (|grad| < ABS_FLOOR), FP cancellation in (out_plus - out_minus)
    # dominates and pure relative error becomes meaningless.
    var p_eps: Float64 = 1e-3
    # Absolute floor: skip params whose true gradient magnitude is below the
    # fp32 FD cancellation noise floor (≈ output_mag / (2*eps) * fp32_eps ≈
    # 5e-5 at this scale). Use 2e-4 as a safe margin; below that, the FD
    # estimate measures roundoff, not gradient.
    var ABS_FLOOR: Float64 = 2e-4
    var max_p_err: Float64 = 0.0
    var max_p_idx: Int = -1
    var max_p_fd: Float64 = 0.0
    var max_p_an: Float64 = 0.0
    for pidx in range(Block.PARAM_SIZE):
        var orig = params[pidx]

        params[pidx] = Scalar[dtype](Float64(orig) + p_eps)
        var out_plus = make_list(BATCH * Block.OUT_DIM)
        var cache_plus = make_list(BATCH * Block.CACHE_SIZE)
        var out_plus_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Block.OUT_DIM), MutAnyOrigin
        ](out_plus.unsafe_ptr())
        var c_plus_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Block.CACHE_SIZE), MutAnyOrigin
        ](cache_plus.unsafe_ptr())
        Block.forward[BATCH, dtype](inp_t, out_plus_t, p_t, s_t, c_plus_t)

        params[pidx] = Scalar[dtype](Float64(orig) - p_eps)
        var out_minus = make_list(BATCH * Block.OUT_DIM)
        var cache_minus = make_list(BATCH * Block.CACHE_SIZE)
        var out_minus_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Block.OUT_DIM), MutAnyOrigin
        ](out_minus.unsafe_ptr())
        var c_minus_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Block.CACHE_SIZE), MutAnyOrigin
        ](cache_minus.unsafe_ptr())
        Block.forward[BATCH, dtype](inp_t, out_minus_t, p_t, s_t, c_minus_t)

        params[pidx] = orig

        var fd: Float64 = 0.0
        for j in range(BATCH * Block.OUT_DIM):
            fd += (
                Float64(go[j])
                * (Float64(out_plus[j]) - Float64(out_minus[j]))
                / (2.0 * p_eps)
            )
        var an = Float64(gp_data[pidx])
        var err = math_abs(fd - an)
        # Combined abs+rel: skip params whose true gradient is below the FP
        # cancellation noise floor. With BATCH*OUT_DIM ~ 96 summands and fp32
        # outputs of O(1), centered FD at eps=1e-3 has noise ~1e-4.
        if math_abs(fd) < ABS_FLOOR and math_abs(an) < ABS_FLOOR:
            continue
        var denom = math_abs(fd) + math_abs(an) + 1e-8
        var rel = err / denom
        if rel > max_p_err:
            max_p_err = rel
            max_p_idx = pidx
            max_p_fd = fd
            max_p_an = an

    print(
        "  worst param idx="
        + String(max_p_idx)
        + " fd="
        + String(max_p_fd)
        + " an="
        + String(max_p_an)
    )
    check(
        max_p_err < 5e-2,
        "max relative error on grad_params (full sweep, eps=1e-3) = "
        + String(max_p_err)
        + " (tol 5e-2)",
        fails,
    )

    return fails


def main():
    var total_fails = 0
    total_fails += test_dims()
    total_fails += test_forward_runs()
    total_fails += test_gradcheck()

    print("\n" + "=" * 70)
    if total_fails == 0:
        print("ALL TRANSFORMER BLOCK TESTS PASSED")
    else:
        print("FAILED: " + String(total_fails) + " checks")
    print("=" * 70)

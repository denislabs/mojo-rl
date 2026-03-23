"""Phase 8 verification tests for Attention & Transformer Primitives.

Tests: ScaledDotProductAttention, transformer composites.

Run with:
    pixi run mojo run -I . tests/test_autodiff_phase8.mojo
"""

from std.random import seed, random_float64
from std.math import abs as math_abs, sqrt, exp

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.autodiff import (
    AutoDiffChain,
    Dense,
    DenseReLU,
    MatMul,
    BiasAdd,
    ReLUOp,
    LayerNormOp,
    ScaledDotProductAttention,
    Residual,
    Repeat,
)
from mojo_rl.nn.model.sequential import Sequential
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


def max_diff(a: List[Scalar[dtype]], b: List[Scalar[dtype]], n: Int) -> Float64:
    var md: Float64 = 0
    for i in range(n):
        var d = math_abs(Float64(a[i]) - Float64(b[i]))
        if d > md:
            md = d
    return md


# =============================================================================
# Test 1: Compile-time dimension checks
# =============================================================================


def test_attention_dims() -> Int:
    print_header("ScaledDotProductAttention dimension checks")
    var fails = 0

    # Single head, dim=4, seq_len=3
    comptime Attn1 = ScaledDotProductAttention[4, 1, 3]
    check(
        Attn1.IN_DIM == 3 * 4 * 3,
        "IN_DIM = seq*dim*3 = " + String(Attn1.IN_DIM),
        fails,
    )
    check(
        Attn1.OUT_DIM == 3 * 4,
        "OUT_DIM = seq*dim = " + String(Attn1.OUT_DIM),
        fails,
    )
    check(Attn1.PARAM_SIZE == 0, "PARAM_SIZE = 0", fails)
    check(Attn1.head_dim == 4, "head_dim = dim/n_heads = 4", fails)

    # Multi-head: dim=8, n_heads=2, seq_len=4
    comptime Attn2 = ScaledDotProductAttention[8, 2, 4]
    check(
        Attn2.IN_DIM == 4 * 8 * 3,
        "IN_DIM = 4*8*3 = " + String(Attn2.IN_DIM),
        fails,
    )
    check(
        Attn2.OUT_DIM == 4 * 8,
        "OUT_DIM = 4*8 = " + String(Attn2.OUT_DIM),
        fails,
    )
    check(Attn2.head_dim == 4, "head_dim = 8/2 = 4", fails)
    check(
        Attn2.CACHE_SIZE == 3 * 4 * 8 + 2 * 4 * 4,
        "CACHE_SIZE = 3*seq*dim + n_heads*seq^2 = " + String(Attn2.CACHE_SIZE),
        fails,
    )

    return fails


# =============================================================================
# Test 2: Single-head attention forward matches manual computation
# =============================================================================


def test_single_head_forward() -> Int:
    print_header("Single-head attention forward correctness")
    var fails = 0
    seed(42)

    # dim=2, n_heads=1, seq_len=2, batch=1
    comptime DIM = 2
    comptime SEQ = 2
    comptime BATCH = 1
    comptime Attn = ScaledDotProductAttention[DIM, 1, SEQ]

    # Input: [Q0, Q1, K0, K1, V0, V1] each 2D
    # Q0 = [1, 0], Q1 = [0, 1]
    # K0 = [1, 0], K1 = [0, 1]
    # V0 = [1, 2], V1 = [3, 4]
    var inp = make_list(BATCH * Attn.IN_DIM)
    # Q
    inp[0] = 1
    inp[1] = 0  # Q0
    inp[2] = 0
    inp[3] = 1  # Q1
    # K
    inp[4] = 1
    inp[5] = 0  # K0
    inp[6] = 0
    inp[7] = 1  # K1
    # V
    inp[8] = 1
    inp[9] = 2  # V0
    inp[10] = 3
    inp[11] = 4  # V1

    var out_data = make_list(BATCH * Attn.OUT_DIM)
    var cache_data = make_list(BATCH * Attn.CACHE_SIZE)
    var params_data = make_list(1)  # PARAM_SIZE=0, but need at least 1

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

    # Manual computation:
    # scores[0,0] = Q0.K0 / sqrt(2) = 1/sqrt(2) ≈ 0.7071
    # scores[0,1] = Q0.K1 / sqrt(2) = 0/sqrt(2) = 0
    # scores[1,0] = Q1.K0 / sqrt(2) = 0
    # scores[1,1] = Q1.K1 / sqrt(2) = 1/sqrt(2) ≈ 0.7071
    var scale = 1.0 / sqrt(2.0)

    # softmax([0.7071, 0]) ≈ [0.6457, 0.3543]
    var s00 = exp(1.0 * scale)
    var s01 = exp(0.0 * scale)
    var sum0 = s00 + s01
    var a00 = s00 / sum0
    var a01 = s01 / sum0

    # softmax([0, 0.7071]) ≈ [0.3543, 0.6457]
    var s10 = exp(0.0 * scale)
    var s11 = exp(1.0 * scale)
    var sum1 = s10 + s11
    var a10 = s10 / sum1
    var a11 = s11 / sum1

    # out[0] = a00 * V0 + a01 * V1
    var expected_00 = a00 * 1.0 + a01 * 3.0
    var expected_01 = a00 * 2.0 + a01 * 4.0
    # out[1] = a10 * V0 + a11 * V1
    var expected_10 = a10 * 1.0 + a11 * 3.0
    var expected_11 = a10 * 2.0 + a11 * 4.0

    var tol: Float64 = 1e-5
    check(
        math_abs(Float64(out_data[0]) - expected_00) < tol,
        "out[0,0] = "
        + String(Float64(out_data[0]))
        + " expected "
        + String(expected_00),
        fails,
    )
    check(
        math_abs(Float64(out_data[1]) - expected_01) < tol,
        "out[0,1] = "
        + String(Float64(out_data[1]))
        + " expected "
        + String(expected_01),
        fails,
    )
    check(
        math_abs(Float64(out_data[2]) - expected_10) < tol,
        "out[1,0] = "
        + String(Float64(out_data[2]))
        + " expected "
        + String(expected_10),
        fails,
    )
    check(
        math_abs(Float64(out_data[3]) - expected_11) < tol,
        "out[1,1] = "
        + String(Float64(out_data[3]))
        + " expected "
        + String(expected_11),
        fails,
    )

    return fails


# =============================================================================
# Test 3: Multi-head attention produces correct output shape
# =============================================================================


def test_multi_head_forward() -> Int:
    print_header("Multi-head attention forward (dim=4, heads=2, seq=3)")
    var fails = 0
    seed(123)

    comptime DIM = 4
    comptime HEADS = 2
    comptime SEQ = 3
    comptime BATCH = 2
    comptime Attn = ScaledDotProductAttention[DIM, HEADS, SEQ]

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

    # Check output is not all zeros (sanity check)
    var has_nonzero = False
    for i in range(BATCH * Attn.OUT_DIM):
        if math_abs(Float64(out_data[i])) > 1e-10:
            has_nonzero = True
            break
    check(has_nonzero, "output is non-zero", fails)

    # Verify attention weights sum to 1 for each query position
    var attn_offset = Attn._attn_cache_offset()
    for b in range(BATCH):
        for h in range(HEADS):
            for i in range(SEQ):
                var row_sum: Float64 = 0.0
                for j in range(SEQ):
                    var idx = (
                        b * Attn.CACHE_SIZE
                        + attn_offset
                        + h * SEQ * SEQ
                        + i * SEQ
                        + j
                    )
                    row_sum += Float64(cache_data[idx])
                check(
                    math_abs(row_sum - 1.0) < 1e-5,
                    "attn weights sum=1 (b="
                    + String(b)
                    + " h="
                    + String(h)
                    + " i="
                    + String(i)
                    + ") sum="
                    + String(row_sum),
                    fails,
                )

    return fails


# =============================================================================
# Test 4: Finite difference gradient check
# =============================================================================


def test_attention_grad() -> Int:
    print_header("ScaledDotProductAttention gradient check (finite diff)")
    var fails = 0
    seed(77)

    comptime DIM = 4
    comptime HEADS = 2
    comptime SEQ = 2
    comptime BATCH = 1
    comptime Attn = ScaledDotProductAttention[DIM, HEADS, SEQ]

    var inp = make_rand_list(BATCH * Attn.IN_DIM)
    var go_data = make_rand_list(BATCH * Attn.OUT_DIM)
    var params_data = make_list(1)

    # Analytical gradients
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

    # Finite difference for input gradients
    var eps: Float64 = 1e-4
    var max_err: Float64 = 0.0
    var n_checked = 0

    for idx in range(BATCH * Attn.IN_DIM):
        var orig = inp[idx]

        # f(x + eps)
        inp[idx] = Scalar[dtype](Float64(orig) + eps)
        var out_plus = make_list(BATCH * Attn.OUT_DIM)
        var cache_plus = make_list(BATCH * Attn.CACHE_SIZE)
        var inp_t2 = LayoutTensor[
            dtype, Layout.row_major(BATCH, Attn.IN_DIM), MutAnyOrigin
        ](inp.unsafe_ptr())
        var out_t2 = LayoutTensor[
            dtype, Layout.row_major(BATCH, Attn.OUT_DIM), MutAnyOrigin
        ](out_plus.unsafe_ptr())
        var c_t2 = LayoutTensor[
            dtype, Layout.row_major(BATCH, Attn.CACHE_SIZE), MutAnyOrigin
        ](cache_plus.unsafe_ptr())
        Attn.eval[BATCH](inp_t2, out_t2, p_t, c_t2)

        # f(x - eps)
        inp[idx] = Scalar[dtype](Float64(orig) - eps)
        var out_minus = make_list(BATCH * Attn.OUT_DIM)
        var cache_minus = make_list(BATCH * Attn.CACHE_SIZE)
        var inp_t3 = LayoutTensor[
            dtype, Layout.row_major(BATCH, Attn.IN_DIM), MutAnyOrigin
        ](inp.unsafe_ptr())
        var out_t3 = LayoutTensor[
            dtype, Layout.row_major(BATCH, Attn.OUT_DIM), MutAnyOrigin
        ](out_minus.unsafe_ptr())
        var c_t3 = LayoutTensor[
            dtype, Layout.row_major(BATCH, Attn.CACHE_SIZE), MutAnyOrigin
        ](cache_minus.unsafe_ptr())
        Attn.eval[BATCH](inp_t3, out_t3, p_t, c_t3)

        inp[idx] = orig

        # FD gradient: sum_j go[j] * (f_plus[j] - f_minus[j]) / (2*eps)
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
# Test 5: AutoDiffChain composition with attention
# =============================================================================


def test_chain_with_attention() -> Int:
    print_header("AutoDiffChain composition: MatMul -> Attention")
    var fails = 0
    seed(99)

    # Project from dim=4 to dim*3=12 (QKV), then attention
    comptime DIM = 4
    comptime SEQ = 2
    comptime HEADS = 2
    comptime Proj = MatMul[SEQ * DIM, SEQ * DIM * 3]
    comptime Attn = ScaledDotProductAttention[DIM, HEADS, SEQ]
    comptime Chain = AutoDiffChain[Proj, Attn]

    check(
        Chain.IN_DIM == SEQ * DIM,
        "Chain IN_DIM = " + String(Chain.IN_DIM),
        fails,
    )
    check(
        Chain.OUT_DIM == SEQ * DIM,
        "Chain OUT_DIM = " + String(Chain.OUT_DIM),
        fails,
    )
    check(
        Chain.PARAM_SIZE == Proj.PARAM_SIZE,
        "Chain PARAM_SIZE = " + String(Chain.PARAM_SIZE),
        fails,
    )

    # Forward pass
    comptime BATCH = 1
    var inp = make_rand_list(BATCH * Chain.IN_DIM)
    var params = make_rand_list(Chain.PARAM_SIZE)
    var out_data = make_list(BATCH * Chain.OUT_DIM)
    var cache_data = make_list(BATCH * Chain.CACHE_SIZE)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Chain.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Chain.OUT_DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Chain.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Chain.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())

    Chain.forward[BATCH](inp_t, out_t, p_t, c_t)

    var has_nonzero = False
    for i in range(BATCH * Chain.OUT_DIM):
        if math_abs(Float64(out_data[i])) > 1e-10:
            has_nonzero = True
            break
    check(has_nonzero, "Chain forward produces non-zero output", fails)

    # Backward pass
    var go_data = make_rand_list(BATCH * Chain.OUT_DIM)
    var gi_data = make_list(BATCH * Chain.IN_DIM)
    var gp_data = make_list(Chain.PARAM_SIZE)

    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Chain.OUT_DIM), MutAnyOrigin
    ](go_data.unsafe_ptr())
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Chain.IN_DIM), MutAnyOrigin
    ](gi_data.unsafe_ptr())
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(Chain.PARAM_SIZE), MutAnyOrigin
    ](gp_data.unsafe_ptr())

    Chain.backward[BATCH](go_t, gi_t, p_t, c_t, gp_t)

    var has_nonzero_gi = False
    for i in range(BATCH * Chain.IN_DIM):
        if math_abs(Float64(gi_data[i])) > 1e-10:
            has_nonzero_gi = True
            break
    check(has_nonzero_gi, "Chain backward produces non-zero grad_input", fails)

    var has_nonzero_gp = False
    for i in range(Chain.PARAM_SIZE):
        if math_abs(Float64(gp_data[i])) > 1e-10:
            has_nonzero_gp = True
            break
    check(has_nonzero_gp, "Chain backward produces non-zero grad_params", fails)

    return fails


# =============================================================================
# Test 6: Residual + Attention (Transformer block pattern)
# =============================================================================


def test_residual_attention() -> Int:
    print_header(
        "Residual[AutoDiffChain[Proj, Attention]] — transformer block pattern"
    )
    var fails = 0
    seed(42)

    comptime DIM = 4
    comptime SEQ = 2
    comptime HEADS = 2
    comptime SD = SEQ * DIM  # flattened sequence dim
    comptime AttnBlock = AutoDiffChain[
        MatMul[SD, SD * 3],
        BiasAdd[SD * 3],
        ScaledDotProductAttention[DIM, HEADS, SEQ],
    ]
    comptime ResAttn = Residual[AttnBlock]

    check(
        ResAttn.IN_DIM == SD,
        "ResAttn IN_DIM = " + String(ResAttn.IN_DIM),
        fails,
    )
    check(
        ResAttn.OUT_DIM == SD,
        "ResAttn OUT_DIM = " + String(ResAttn.OUT_DIM),
        fails,
    )

    comptime BATCH = 2
    var inp = make_rand_list(BATCH * ResAttn.IN_DIM)
    var params = make_rand_list(ResAttn.PARAM_SIZE)
    var out_data = make_list(BATCH * ResAttn.OUT_DIM)
    var cache_data = make_list(BATCH * ResAttn.CACHE_SIZE)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, ResAttn.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, ResAttn.OUT_DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(ResAttn.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, ResAttn.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())

    ResAttn.forward[BATCH](inp_t, out_t, p_t, c_t)

    # output should differ from input (inner block adds something)
    var diff: Float64 = 0
    for i in range(BATCH * SD):
        diff += math_abs(Float64(out_data[i]) - Float64(inp[i]))
    check(
        diff > 1e-6,
        "Residual output differs from input (diff=" + String(diff) + ")",
        fails,
    )

    # Backward
    var go_data = make_rand_list(BATCH * ResAttn.OUT_DIM)
    var gi_data = make_list(BATCH * ResAttn.IN_DIM)
    var gp_data = make_list(ResAttn.PARAM_SIZE)

    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, ResAttn.OUT_DIM), MutAnyOrigin
    ](go_data.unsafe_ptr())
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, ResAttn.IN_DIM), MutAnyOrigin
    ](gi_data.unsafe_ptr())
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(ResAttn.PARAM_SIZE), MutAnyOrigin
    ](gp_data.unsafe_ptr())

    ResAttn.backward[BATCH](go_t, gi_t, p_t, c_t, gp_t)

    var has_gi = False
    for i in range(BATCH * ResAttn.IN_DIM):
        if math_abs(Float64(gi_data[i])) > 1e-10:
            has_gi = True
            break
    check(has_gi, "Residual backward produces non-zero grad_input", fails)

    return fails


# =============================================================================
# Test 7: Transformer composites compile and run
# =============================================================================


def test_transformer_composites() -> Int:
    print_header("Transformer composites: Sequential[ResAttn, ResFFN] compiles")
    var fails = 0
    seed(55)

    comptime DIM = 4
    comptime SEQ = 2
    comptime HEADS = 2
    comptime FF_DIM = 8
    comptime SD = SEQ * DIM

    # Attention sub-block: project to QKV, attend, project back
    comptime AttnInner = AutoDiffChain[
        MatMul[SD, SD * 3],
        BiasAdd[SD * 3],
        ScaledDotProductAttention[DIM, HEADS, SEQ],
    ]
    comptime ResAttn = Residual[AttnInner]

    # Fdef sub-block
    comptime FFNInner = AutoDiffChain[
        MatMul[SD, FF_DIM],
        BiasAdd[FF_DIM],
        ReLUOp[FF_DIM],
        MatMul[FF_DIM, SD],
        BiasAdd[SD],
    ]
    comptime ResFdef = Residual[FFNInner]

    # Full transformer layer
    comptime TransformerLayer = Sequential[ResAttn, ResFFN]

    check(
        TransformerLayer.IN_DIM == SD,
        "TransformerLayer IN_DIM = " + String(TransformerLayer.IN_DIM),
        fails,
    )
    check(
        TransformerLayer.OUT_DIM == SD,
        "TransformerLayer OUT_DIM = " + String(TransformerLayer.OUT_DIM),
        fails,
    )

    comptime BATCH = 1
    var inp = make_rand_list(BATCH * TransformerLayer.IN_DIM)
    var params = make_rand_list(TransformerLayer.PARAM_SIZE)
    var out_data = make_list(BATCH * TransformerLayer.OUT_DIM)
    var cache_data = make_list(BATCH * TransformerLayer.CACHE_SIZE)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, TransformerLayer.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, TransformerLayer.OUT_DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(TransformerLayer.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, TransformerLayer.CACHE_SIZE),
        MutAnyOrigin,
    ](cache_data.unsafe_ptr())

    TransformerLayer.forward[BATCH](inp_t, out_t, p_t, c_t)

    var has_output = False
    for i in range(BATCH * TransformerLayer.OUT_DIM):
        if math_abs(Float64(out_data[i])) > 1e-10:
            has_output = True
            break
    check(has_output, "TransformerLayer produces non-zero output", fails)

    # Backward
    var go_data = make_rand_list(BATCH * TransformerLayer.OUT_DIM)
    var gi_data = make_list(BATCH * TransformerLayer.IN_DIM)
    var gp_data = make_list(TransformerLayer.PARAM_SIZE)

    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, TransformerLayer.OUT_DIM), MutAnyOrigin
    ](go_data.unsafe_ptr())
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, TransformerLayer.IN_DIM), MutAnyOrigin
    ](gi_data.unsafe_ptr())
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(TransformerLayer.PARAM_SIZE), MutAnyOrigin
    ](gp_data.unsafe_ptr())

    TransformerLayer.backward[BATCH](go_t, gi_t, p_t, c_t, gp_t)

    var has_gi = False
    for i in range(BATCH * TransformerLayer.IN_DIM):
        if math_abs(Float64(gi_data[i])) > 1e-10:
            has_gi = True
            break
    check(has_gi, "TransformerLayer backward produces grad_input", fails)

    return fails


# =============================================================================
# Test 8: Repeat[N, TransformerLayer] compiles and runs
# =============================================================================


def test_stacked_transformer() -> Int:
    print_header("Repeat[2, TransformerLayer] — stacked transformer")
    var fails = 0
    seed(77)

    comptime DIM = 4
    comptime SEQ = 2
    comptime HEADS = 2
    comptime FF_DIM = 8
    comptime SD = SEQ * DIM

    comptime AttnInner = AutoDiffChain[
        MatMul[SD, SD * 3],
        BiasAdd[SD * 3],
        ScaledDotProductAttention[DIM, HEADS, SEQ],
    ]
    comptime ResAttn = Residual[AttnInner]
    comptime FFNInner = AutoDiffChain[
        MatMul[SD, FF_DIM],
        BiasAdd[FF_DIM],
        ReLUOp[FF_DIM],
        MatMul[FF_DIM, SD],
        BiasAdd[SD],
    ]
    comptime ResFdef = Residual[FFNInner]
    comptime Layer = Sequential[ResAttn, ResFFN]

    # Stack 2 layers with shared weights
    comptime Encoder = Repeat[2, Layer]

    check(
        Encoder.IN_DIM == SD,
        "Encoder IN_DIM = " + String(Encoder.IN_DIM),
        fails,
    )
    check(
        Encoder.OUT_DIM == SD,
        "Encoder OUT_DIM = " + String(Encoder.OUT_DIM),
        fails,
    )
    check(
        Encoder.PARAM_SIZE == Layer.PARAM_SIZE,
        "Encoder PARAM_SIZE = Layer PARAM_SIZE (shared weights) = "
        + String(Encoder.PARAM_SIZE),
        fails,
    )
    check(
        Encoder.CACHE_SIZE == Layer.CACHE_SIZE * 2,
        "Encoder CACHE_SIZE = 2 * Layer CACHE_SIZE = "
        + String(Encoder.CACHE_SIZE),
        fails,
    )

    comptime BATCH = 1
    var inp = make_rand_list(BATCH * Encoder.IN_DIM)
    var params = make_rand_list(Encoder.PARAM_SIZE)
    var out_data = make_list(BATCH * Encoder.OUT_DIM)
    var cache_data = make_list(BATCH * Encoder.CACHE_SIZE)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Encoder.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Encoder.OUT_DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Encoder.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Encoder.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())

    Encoder.forward[BATCH](inp_t, out_t, p_t, c_t)

    var has_output = False
    for i in range(BATCH * Encoder.OUT_DIM):
        if math_abs(Float64(out_data[i])) > 1e-10:
            has_output = True
            break
    check(
        has_output,
        "Repeat[2, TransformerLayer] produces non-zero output",
        fails,
    )

    # Backward
    var go_data = make_rand_list(BATCH * Encoder.OUT_DIM)
    var gi_data = make_list(BATCH * Encoder.IN_DIM)
    var gp_data = make_list(Encoder.PARAM_SIZE)

    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Encoder.OUT_DIM), MutAnyOrigin
    ](go_data.unsafe_ptr())
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Encoder.IN_DIM), MutAnyOrigin
    ](gi_data.unsafe_ptr())
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(Encoder.PARAM_SIZE), MutAnyOrigin
    ](gp_data.unsafe_ptr())

    Encoder.backward[BATCH](go_t, gi_t, p_t, c_t, gp_t)

    var has_gi = False
    for i in range(BATCH * Encoder.IN_DIM):
        if math_abs(Float64(gi_data[i])) > 1e-10:
            has_gi = True
            break
    check(
        has_gi,
        "Repeat[2, TransformerLayer] backward produces grad_input",
        fails,
    )

    return fails


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("Phase 8: Attention & Transformer Primitives — Test Suite")
    print("=" * 70)

    var total_fails = 0
    total_fails += test_attention_dims()
    total_fails += test_single_head_forward()
    total_fails += test_multi_head_forward()
    total_fails += test_attention_grad()
    total_fails += test_chain_with_attention()
    total_fails += test_residual_attention()
    total_fails += test_transformer_composites()
    total_fails += test_stacked_transformer()

    print("\n" + "=" * 70)
    if total_fails == 0:
        print("ALL TESTS PASSED")
    else:
        print(String(total_fails) + " TESTS FAILED")
    print("=" * 70)

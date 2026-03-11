"""Phase 6 verification tests for regularization & structural primitives.

Tests: DropoutOp, Flatten, Embedding.

Per op: (1) forward correctness, (2) backward correctness / gradient check.

Run with:
    pixi run mojo run tests/test_autodiff_phase6.mojo
"""

from std.random import seed, random_float64
from std.math import abs as math_abs

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.autodiff import (
    AutoDiffChain,
    DropoutOp,
    Flatten,
    Embedding,
    MatMul,
    BiasAdd,
    ReLUOp,
)
from layout import Layout, LayoutTensor


fn print_header(name: String):
    print("\n" + "=" * 70)
    print("TEST: " + name)
    print("=" * 70)


fn check(cond: Bool, msg: String, mut fails: Int):
    if cond:
        print("  PASS: " + msg)
    else:
        print("  FAIL: " + msg)
        fails += 1


fn make_list(size: Int) -> List[Scalar[dtype]]:
    var lst = List[Scalar[dtype]](capacity=size)
    for _ in range(size):
        lst.append(0)
    return lst^


fn make_rand_list(size: Int) -> List[Scalar[dtype]]:
    var lst = List[Scalar[dtype]](capacity=size)
    for _ in range(size):
        lst.append(Scalar[dtype](random_float64(-1.0, 1.0)))
    return lst^


fn max_diff(a: List[Scalar[dtype]], b: List[Scalar[dtype]], n: Int) -> Float64:
    var md: Float64 = 0
    for i in range(n):
        var d = math_abs(Float64(a[i]) - Float64(b[i]))
        if d > md:
            md = d
    return md


# =============================================================================
# Test 1: DropoutOp — basic properties
# =============================================================================


fn test_dropout_basic() -> Int:
    print_header("DropoutOp basic properties + rate 0 = identity")
    var fails = 0

    comptime DIM = 8
    comptime BATCH = 4

    # Rate 0 dropout (0/10) = identity with scale 1.0
    comptime Op0 = DropoutOp[DIM, 0, 10]
    check(Op0.PARAM_SIZE == 0, "PARAM_SIZE == 0", fails)
    check(Op0.CACHE_SIZE == DIM, "CACHE_SIZE == dim", fails)
    check(Op0.IN_DIM == DIM, "IN_DIM == dim", fails)
    check(Op0.OUT_DIM == DIM, "OUT_DIM == dim", fails)

    seed(42)
    var inp = make_rand_list(BATCH * DIM)
    var params = make_list(0)
    var out_data = make_list(BATCH * DIM)
    var cache_data = make_list(BATCH * DIM)

    var inp_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        inp.unsafe_ptr()
    )
    var out_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        out_data.unsafe_ptr()
    )
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Op0.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var c_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        cache_data.unsafe_ptr()
    )

    Op0.eval[BATCH](inp_t, out_t, p_t, c_t)

    # Rate 0: all elements should pass through (scale = 1.0)
    var identity_ok = True
    for i in range(BATCH * DIM):
        if math_abs(Float64(out_data[i]) - Float64(inp[i])) > 1e-5:
            identity_ok = False
    check(identity_ok, "Rate 0 = identity (output == input)", fails)

    return fails


# =============================================================================
# Test 2: DropoutOp — mask and backward consistency
# =============================================================================


fn test_dropout_mask() -> Int:
    print_header("DropoutOp mask consistency + backward")
    var fails = 0

    comptime DIM = 16
    comptime BATCH = 4
    comptime Op = DropoutOp[DIM, 5, 10]  # 50% dropout

    seed(42)
    var inp = make_rand_list(BATCH * DIM)
    var params = make_list(0)
    var out_data = make_list(BATCH * DIM)
    var cache_data = make_list(BATCH * DIM)

    var inp_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        inp.unsafe_ptr()
    )
    var out_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        out_data.unsafe_ptr()
    )
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var c_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        cache_data.unsafe_ptr()
    )

    Op.eval[BATCH](inp_t, out_t, p_t, c_t)

    # Check that some elements are zeroed and some are scaled
    var n_zero = 0
    var n_nonzero = 0
    var scale = 2.0  # 1/(1-0.5) = 2.0
    var scale_ok = True
    for i in range(BATCH * DIM):
        var out_val = Float64(out_data[i])
        var cache_val = Float64(cache_data[i])
        if math_abs(cache_val) < 1e-7:
            n_zero += 1
            if math_abs(out_val) > 1e-7:
                scale_ok = False
        else:
            n_nonzero += 1
            var expected = Float64(inp[i]) * scale
            if math_abs(out_val - expected) > 1e-4:
                scale_ok = False
    check(
        n_zero > 0,
        "Some elements dropped (n_zero=" + String(n_zero) + ")",
        fails,
    )
    check(
        n_nonzero > 0,
        "Some elements kept (n_nonzero=" + String(n_nonzero) + ")",
        fails,
    )
    check(scale_ok, "Kept elements scaled by 1/(1-rate)=2.0", fails)

    # Backward: grad_input = grad_output * mask (same mask from cache)
    var go_data = make_rand_list(BATCH * DIM)
    var gi_data = make_list(BATCH * DIM)
    var gp_data = make_list(0)

    var go_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        go_data.unsafe_ptr()
    )
    var gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        gi_data.unsafe_ptr()
    )
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin
    ](gp_data.unsafe_ptr())

    Op.vjp[BATCH](go_t, gi_t, p_t, c_t, gp_t)

    var bwd_ok = True
    for i in range(BATCH * DIM):
        var expected = Float64(go_data[i]) * Float64(cache_data[i])
        var got = Float64(gi_data[i])
        if math_abs(expected - got) > 1e-5:
            bwd_ok = False
    check(bwd_ok, "Backward: grad_input = grad_output * mask", fails)

    return fails


# =============================================================================
# Test 3: DropoutOp — rate 1 = zero output
# =============================================================================


fn test_dropout_rate_one() -> Int:
    print_header("DropoutOp rate=1 -> zero output")
    var fails = 0

    comptime DIM = 8
    comptime BATCH = 2
    # Rate = 10/10 = 100% dropout -> all elements dropped
    comptime Op = DropoutOp[DIM, 10, 10]

    seed(42)
    var inp = make_rand_list(BATCH * DIM)
    var params = make_list(0)
    var out_data = make_list(BATCH * DIM)
    var cache_data = make_list(BATCH * DIM)

    var inp_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        inp.unsafe_ptr()
    )
    var out_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        out_data.unsafe_ptr()
    )
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var c_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        cache_data.unsafe_ptr()
    )

    Op.eval[BATCH](inp_t, out_t, p_t, c_t)

    var all_zero = True
    for i in range(BATCH * DIM):
        if math_abs(Float64(out_data[i])) > 1e-7:
            all_zero = False
    check(all_zero, "Rate 1.0 = all zeros output", fails)

    return fails


# =============================================================================
# Test 4: Flatten — identity forward/backward
# =============================================================================


fn test_flatten() -> Int:
    print_header("Flatten forward/backward identity")
    var fails = 0

    comptime DIM = 6
    comptime BATCH = 3
    comptime Op = Flatten[DIM]

    check(Op.PARAM_SIZE == 0, "PARAM_SIZE == 0", fails)
    check(Op.CACHE_SIZE == 0, "CACHE_SIZE == 0", fails)
    check(Op.IN_DIM == DIM, "IN_DIM == dim", fails)
    check(Op.OUT_DIM == DIM, "OUT_DIM == dim", fails)

    seed(44)
    var inp = make_rand_list(BATCH * DIM)
    var params = make_list(0)
    var out_data = make_list(BATCH * DIM)
    var cache_data = make_list(0)

    var inp_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        inp.unsafe_ptr()
    )
    var out_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        out_data.unsafe_ptr()
    )
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Op.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())

    Op.eval[BATCH](inp_t, out_t, p_t, c_t)

    var fwd_ok = True
    for i in range(BATCH * DIM):
        if math_abs(Float64(out_data[i]) - Float64(inp[i])) > 1e-7:
            fwd_ok = False
    check(fwd_ok, "Forward: output == input (identity)", fails)

    # Backward
    var go_data = make_rand_list(BATCH * DIM)
    var gi_data = make_list(BATCH * DIM)
    var gp_data = make_list(0)

    var go_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        go_data.unsafe_ptr()
    )
    var gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](
        gi_data.unsafe_ptr()
    )
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin
    ](gp_data.unsafe_ptr())

    Op.vjp[BATCH](go_t, gi_t, p_t, c_t, gp_t)

    var bwd_ok = True
    for i in range(BATCH * DIM):
        if math_abs(Float64(gi_data[i]) - Float64(go_data[i])) > 1e-7:
            bwd_ok = False
    check(bwd_ok, "Backward: grad_input == grad_output (identity)", fails)

    return fails


# =============================================================================
# Test 5: Flatten composition with AutoDiffChain
# =============================================================================


fn test_flatten_composition() -> Int:
    print_header("Flatten in AutoDiffChain composition")
    var fails = 0

    comptime DIM = 4
    comptime OUT = 3
    comptime BATCH = 2

    # Chain: MatMul[4,4] -> Flatten[4] -> MatMul[4,3] -> BiasAdd[3]
    comptime Chain = AutoDiffChain[
        MatMul[DIM, DIM], Flatten[DIM], MatMul[DIM, OUT], BiasAdd[OUT]
    ]

    check(Chain.IN_DIM == DIM, "Chain IN_DIM == 4", fails)
    check(Chain.OUT_DIM == OUT, "Chain OUT_DIM == 3", fails)

    # Also check that Flatten doesn't add params/cache
    comptime ChainNoFlatten = AutoDiffChain[
        MatMul[DIM, DIM], MatMul[DIM, OUT], BiasAdd[OUT]
    ]
    check(
        Chain.PARAM_SIZE == ChainNoFlatten.PARAM_SIZE,
        "Flatten adds 0 params",
        fails,
    )
    check(
        Chain.CACHE_SIZE == ChainNoFlatten.CACHE_SIZE,
        "Flatten adds 0 cache",
        fails,
    )

    return fails


# =============================================================================
# Test 6: Embedding — forward correctness
# =============================================================================


fn test_embedding_forward() -> Int:
    print_header("Embedding forward correctness")
    var fails = 0

    comptime VOCAB = 5
    comptime EMBED = 3
    comptime BATCH = 2
    comptime Op = Embedding[VOCAB, EMBED]

    check(Op.PARAM_SIZE == VOCAB * EMBED, "PARAM_SIZE == vocab * embed", fails)
    check(Op.CACHE_SIZE == VOCAB, "CACHE_SIZE == vocab", fails)
    check(Op.IN_DIM == VOCAB, "IN_DIM == vocab", fails)
    check(Op.OUT_DIM == EMBED, "OUT_DIM == embed", fails)

    seed(45)

    # Create embedding table (params)
    var params = make_rand_list(VOCAB * EMBED)

    # Create one-hot inputs: batch 0 = index 2, batch 1 = index 4
    var inp = make_list(BATCH * VOCAB)
    inp[0 * VOCAB + 2] = Scalar[dtype](1.0)  # batch 0 -> index 2
    inp[1 * VOCAB + 4] = Scalar[dtype](1.0)  # batch 1 -> index 4

    var out_data = make_list(BATCH * EMBED)
    var cache_data = make_list(BATCH * VOCAB)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, VOCAB), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, EMBED), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var c_t = LayoutTensor[dtype, Layout.row_major(BATCH, VOCAB), MutAnyOrigin](
        cache_data.unsafe_ptr()
    )

    Op.eval[BATCH](inp_t, out_t, p_t, c_t)

    # Check: output[0] should be row 2 of W, output[1] should be row 4 of W
    var fwd_ok = True
    for j in range(EMBED):
        var expected_0 = Float64(params[2 * EMBED + j])
        var got_0 = Float64(out_data[0 * EMBED + j])
        if math_abs(expected_0 - got_0) > 1e-5:
            fwd_ok = False

        var expected_1 = Float64(params[4 * EMBED + j])
        var got_1 = Float64(out_data[1 * EMBED + j])
        if math_abs(expected_1 - got_1) > 1e-5:
            fwd_ok = False
    check(fwd_ok, "One-hot input selects correct embedding rows", fails)

    return fails


# =============================================================================
# Test 7: Embedding — backward gradient scatter
# =============================================================================


fn test_embedding_backward() -> Int:
    print_header("Embedding backward gradient scatter")
    var fails = 0

    comptime VOCAB = 5
    comptime EMBED = 3
    comptime BATCH = 2
    comptime Op = Embedding[VOCAB, EMBED]

    seed(46)
    var params = make_rand_list(VOCAB * EMBED)

    # One-hot: batch 0 = index 1, batch 1 = index 3
    var inp = make_list(BATCH * VOCAB)
    inp[0 * VOCAB + 1] = Scalar[dtype](1.0)
    inp[1 * VOCAB + 3] = Scalar[dtype](1.0)

    var out_data = make_list(BATCH * EMBED)
    var cache_data = make_list(BATCH * VOCAB)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, VOCAB), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, EMBED), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var c_t = LayoutTensor[dtype, Layout.row_major(BATCH, VOCAB), MutAnyOrigin](
        cache_data.unsafe_ptr()
    )

    Op.eval[BATCH](inp_t, out_t, p_t, c_t)

    # Backward
    var go_data = make_rand_list(BATCH * EMBED)
    var gi_data = make_list(BATCH * VOCAB)
    var gp_data = make_list(VOCAB * EMBED)

    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, EMBED), MutAnyOrigin
    ](go_data.unsafe_ptr())
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, VOCAB), MutAnyOrigin
    ](gi_data.unsafe_ptr())
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin
    ](gp_data.unsafe_ptr())

    Op.vjp[BATCH](go_t, gi_t, p_t, c_t, gp_t)

    # Check grad_params: dW should have gradients scattered to rows 1 and 3
    # dW[1, j] = go[0, j] (from batch 0 one-hot at index 1)
    # dW[3, j] = go[1, j] (from batch 1 one-hot at index 3)
    # All other rows should be 0
    var scatter_ok = True
    for v in range(VOCAB):
        for j in range(EMBED):
            var dw_val = Float64(gp_data[v * EMBED + j])
            if v == 1:
                var expected = Float64(go_data[0 * EMBED + j])
                if math_abs(dw_val - expected) > 1e-5:
                    scatter_ok = False
            elif v == 3:
                var expected = Float64(go_data[1 * EMBED + j])
                if math_abs(dw_val - expected) > 1e-5:
                    scatter_ok = False
            else:
                if math_abs(dw_val) > 1e-7:
                    scatter_ok = False
    check(scatter_ok, "dW gradient scattered to correct rows only", fails)

    # Check grad_input: should be grad_output @ W.T
    var gi_ok = True
    for b in range(BATCH):
        for v in range(VOCAB):
            var expected: Float64 = 0.0
            for j in range(EMBED):
                expected += Float64(go_data[b * EMBED + j]) * Float64(
                    params[v * EMBED + j]
                )
            var got = Float64(gi_data[b * VOCAB + v])
            if math_abs(expected - got) > 1e-4:
                gi_ok = False
    check(gi_ok, "grad_input = grad_output @ W.T", fails)

    return fails


# =============================================================================
# Test 8: Embedding finite-difference gradient check
# =============================================================================


fn test_embedding_fd() -> Int:
    print_header("Embedding finite-difference gradient check")
    var fails = 0

    comptime VOCAB = 4
    comptime EMBED = 3
    comptime BATCH = 2
    comptime Op = Embedding[VOCAB, EMBED]

    seed(47)
    var params = make_rand_list(VOCAB * EMBED)

    # One-hot: batch 0 = index 0, batch 1 = index 2
    var inp = make_list(BATCH * VOCAB)
    inp[0 * VOCAB + 0] = Scalar[dtype](1.0)
    inp[1 * VOCAB + 2] = Scalar[dtype](1.0)

    var go_data = make_rand_list(BATCH * EMBED)

    # Finite-difference check on params
    var eps: Float64 = 1e-4
    var max_err: Float64 = 0.0

    # Analytic gradients
    var out_data = make_list(BATCH * EMBED)
    var cache_data = make_list(BATCH * VOCAB)
    var gi_data = make_list(BATCH * VOCAB)
    var gp_data = make_list(VOCAB * EMBED)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, VOCAB), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, EMBED), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var c_t = LayoutTensor[dtype, Layout.row_major(BATCH, VOCAB), MutAnyOrigin](
        cache_data.unsafe_ptr()
    )
    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, EMBED), MutAnyOrigin
    ](go_data.unsafe_ptr())
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, VOCAB), MutAnyOrigin
    ](gi_data.unsafe_ptr())
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin
    ](gp_data.unsafe_ptr())

    Op.eval[BATCH](inp_t, out_t, p_t, c_t)
    Op.vjp[BATCH](go_t, gi_t, p_t, c_t, gp_t)

    # Numerical gradient check on params
    for idx in range(VOCAB * EMBED):
        var orig = params[idx]

        # f(p + eps)
        params[idx] = orig + Scalar[dtype](eps)
        var out_plus = make_list(BATCH * EMBED)
        var cache_plus = make_list(BATCH * VOCAB)
        var p_plus = LayoutTensor[
            dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin
        ](params.unsafe_ptr())
        var out_plus_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, EMBED), MutAnyOrigin
        ](out_plus.unsafe_ptr())
        var c_plus = LayoutTensor[
            dtype, Layout.row_major(BATCH, VOCAB), MutAnyOrigin
        ](cache_plus.unsafe_ptr())
        Op.eval[BATCH](inp_t, out_plus_t, p_plus, c_plus)

        # f(p - eps)
        params[idx] = orig - Scalar[dtype](eps)
        var out_minus = make_list(BATCH * EMBED)
        var cache_minus = make_list(BATCH * VOCAB)
        var p_minus = LayoutTensor[
            dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin
        ](params.unsafe_ptr())
        var out_minus_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, EMBED), MutAnyOrigin
        ](out_minus.unsafe_ptr())
        var c_minus = LayoutTensor[
            dtype, Layout.row_major(BATCH, VOCAB), MutAnyOrigin
        ](cache_minus.unsafe_ptr())
        Op.eval[BATCH](inp_t, out_minus_t, p_minus, c_minus)

        params[idx] = orig

        var num_grad: Float64 = 0.0
        for j in range(BATCH * EMBED):
            num_grad += (
                Float64(go_data[j])
                * (Float64(out_plus[j]) - Float64(out_minus[j]))
                / (2.0 * eps)
            )

        var analytic = Float64(gp_data[idx])
        var err = math_abs(analytic - num_grad)
        if err > max_err:
            max_err = err

    print("  Max param gradient error: " + String(max_err))
    check(max_err < 1e-3, "Param gradient check (tol 1e-3)", fails)

    return fails


# =============================================================================
# Test 9: DropoutOp in AutoDiffChain
# =============================================================================


fn test_dropout_chain() -> Int:
    print_header(
        "DropoutOp in AutoDiffChain[MatMul, BiasAdd, DropoutOp, ReLUOp]"
    )
    var fails = 0

    comptime IN = 3
    comptime OUT = 4
    comptime BATCH = 2

    comptime Chain = AutoDiffChain[
        MatMul[IN, OUT], BiasAdd[OUT], DropoutOp[OUT, 0, 10], ReLUOp[OUT]
    ]

    check(Chain.IN_DIM == IN, "Chain IN_DIM == 3", fails)
    check(Chain.OUT_DIM == OUT, "Chain OUT_DIM == 4", fails)

    # Rate 0 dropout: should behave identically to chain without dropout
    comptime ChainNoDrop = AutoDiffChain[
        MatMul[IN, OUT], BiasAdd[OUT], ReLUOp[OUT]
    ]

    seed(48)
    var params = make_rand_list(Chain.PARAM_SIZE)
    var inp = make_rand_list(BATCH * IN)

    # Forward with dropout (rate 0)
    var out1 = make_list(BATCH * OUT)
    var cache1 = make_list(BATCH * Chain.CACHE_SIZE)
    var inp_t1 = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](
        inp.unsafe_ptr()
    )
    var out_t1 = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
    ](out1.unsafe_ptr())
    var p_t1 = LayoutTensor[
        dtype, Layout.row_major(Chain.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var c_t1 = LayoutTensor[
        dtype, Layout.row_major(BATCH, Chain.CACHE_SIZE), MutAnyOrigin
    ](cache1.unsafe_ptr())
    Chain.forward[BATCH](inp_t1, out_t1, p_t1, c_t1)

    # Forward without dropout
    var out2 = make_list(BATCH * OUT)
    var cache2 = make_list(BATCH * ChainNoDrop.CACHE_SIZE)
    var inp_t2 = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](
        inp.unsafe_ptr()
    )
    var out_t2 = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
    ](out2.unsafe_ptr())
    var p_t2 = LayoutTensor[
        dtype, Layout.row_major(ChainNoDrop.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var c_t2 = LayoutTensor[
        dtype, Layout.row_major(BATCH, ChainNoDrop.CACHE_SIZE), MutAnyOrigin
    ](cache2.unsafe_ptr())
    ChainNoDrop.forward[BATCH](inp_t2, out_t2, p_t2, c_t2)

    var md = max_diff(out1, out2, BATCH * OUT)
    print("  Forward diff (rate 0 dropout vs no dropout): " + String(md))
    check(md < 1e-5, "Rate 0 dropout chain matches no-dropout chain", fails)

    return fails


# =============================================================================
# Main
# =============================================================================


fn main():
    print("=" * 70)
    print("Phase 6: Regularization & Structural Primitives")
    print("=" * 70)

    var total_fails = 0
    total_fails += test_dropout_basic()
    total_fails += test_dropout_mask()
    total_fails += test_dropout_rate_one()
    total_fails += test_flatten()
    total_fails += test_flatten_composition()
    total_fails += test_embedding_forward()
    total_fails += test_embedding_backward()
    total_fails += test_embedding_fd()
    total_fails += test_dropout_chain()

    print("\n" + "=" * 70)
    if total_fails == 0:
        print("ALL PHASE 6 TESTS PASSED")
    else:
        print(String(total_fails) + " TESTS FAILED")
    print("=" * 70)

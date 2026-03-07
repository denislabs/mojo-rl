"""Phase 3 verification tests for additional DiffOp primitives.

Tests: MishOp, Scale, ElemMul, ReduceSum, ReduceMean, SoftmaxOp,
       LayerNormOp, RMSNormOp.

Per op: (1) forward vs manual computation, (2) finite-difference gradient check.

Run with:
    pixi run mojo run tests/test_autodiff_phase3.mojo
"""

from std.random import seed, random_float64
from std.math import abs as math_abs, exp, log, tanh, sqrt

from nn.constants import dtype
from nn.autodiff import (
    AutoDiffChain,
    MishOp,
    Scale,
    ElemMul,
    ReduceSum,
    ReduceMean,
    SoftmaxOp,
    LayerNormOp,
    RMSNormOp,
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
# Finite-difference gradient check helper
# =============================================================================

fn finite_diff_check[
    Op: MishOp
](
    inp: List[Scalar[dtype]],
    params: List[Scalar[dtype]],
    grad_output: List[Scalar[dtype]],
    batch: Int,
    in_dim: Int,
    out_dim: Int,
    param_size: Int,
    cache_size: Int,
    eps: Float64 = 1e-4,
    tol: Float64 = 1e-3,
) -> Float64:
    """Not used directly — we use the generic version below."""
    return 0.0


fn fd_grad_check_generic[
    BATCH: Int, IN_DIM: Int, OUT_DIM: Int, PARAM_SIZE: Int, CACHE_SIZE: Int
](
    eval_fn: fn (
        LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
        mut LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ],
        LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
        mut LayoutTensor[
            dtype, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin
        ],
    ) -> None,
    vjp_fn: fn (
        LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
        mut LayoutTensor[
            dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin
        ],
        LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
        LayoutTensor[dtype, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
        mut LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    ) -> None,
    mut inp: List[Scalar[dtype]],
    params: List[Scalar[dtype]],
    go_data: List[Scalar[dtype]],
    eps: Float64 = 1e-4,
) -> Float64:
    """Finite-difference gradient check for input gradients.

    Returns the max absolute error between analytic and numerical gradients.
    """
    var n_inp = BATCH * IN_DIM

    # --- Analytic gradients ---
    var out_data = make_list(BATCH * OUT_DIM)
    var cache_data = make_list(BATCH * CACHE_SIZE)
    var gi_data = make_list(n_inp)
    var gp_data = make_list(PARAM_SIZE)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin](
        params.unsafe_ptr()
    )
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())
    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ](go_data.unsafe_ptr())
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin
    ](gi_data.unsafe_ptr())
    var gp_t = LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin](
        gp_data.unsafe_ptr()
    )

    eval_fn(inp_t, out_t, p_t, c_t)
    vjp_fn(go_t, gi_t, p_t, c_t, gp_t)

    # --- Numerical gradients (input) ---
    var max_err: Float64 = 0.0
    for idx in range(n_inp):
        var orig = inp[idx]

        # f(x + eps)
        inp[idx] = orig + Scalar[dtype](eps)
        var out_plus = make_list(BATCH * OUT_DIM)
        var cache_plus = make_list(BATCH * CACHE_SIZE)
        var inp_p = LayoutTensor[
            dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin
        ](inp.unsafe_ptr())
        var out_p = LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ](out_plus.unsafe_ptr())
        var c_p = LayoutTensor[
            dtype, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin
        ](cache_plus.unsafe_ptr())
        eval_fn(inp_p, out_p, p_t, c_p)

        # f(x - eps)
        inp[idx] = orig - Scalar[dtype](eps)
        var out_minus = make_list(BATCH * OUT_DIM)
        var cache_minus = make_list(BATCH * CACHE_SIZE)
        var inp_m = LayoutTensor[
            dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin
        ](inp.unsafe_ptr())
        var out_m = LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ](out_minus.unsafe_ptr())
        var c_m = LayoutTensor[
            dtype, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin
        ](cache_minus.unsafe_ptr())
        eval_fn(inp_m, out_m, p_t, c_m)

        inp[idx] = orig

        # numerical grad = sum_j(go[j] * (f_plus[j] - f_minus[j]) / (2*eps))
        var num_grad: Float64 = 0.0
        for j in range(BATCH * OUT_DIM):
            num_grad += Float64(go_data[j]) * (
                Float64(out_plus[j]) - Float64(out_minus[j])
            ) / (2.0 * eps)

        var analytic = Float64(gi_data[idx])
        var err = math_abs(analytic - num_grad)
        if err > max_err:
            max_err = err

    return max_err


# =============================================================================
# Test 1: MishOp
# =============================================================================

fn test_mish() -> Int:
    print_header("MishOp forward + gradient check")
    var fails = 0

    comptime DIM = 4
    comptime BATCH = 2
    comptime Op = MishOp[DIM]

    check(Op.PARAM_SIZE == 0, "PARAM_SIZE == 0", fails)
    check(Op.CACHE_SIZE == DIM, "CACHE_SIZE == dim", fails)

    seed(42)
    var inp = make_rand_list(BATCH * DIM)
    var params = make_list(0)
    var go_data = make_rand_list(BATCH * DIM)

    # Forward check: y = x * tanh(ln(1 + exp(x)))
    var out_data = make_list(BATCH * DIM)
    var cache_data = make_list(BATCH * DIM)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin](
        params.unsafe_ptr()
    )
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](cache_data.unsafe_ptr())

    Op.eval[BATCH](inp_t, out_t, p_t, c_t)

    var fwd_ok = True
    for b in range(BATCH):
        for i in range(DIM):
            var x = Float64(inp[b * DIM + i])
            var sp = log(1.0 + exp(x))
            var expected = x * tanh(sp)
            var got = Float64(out_data[b * DIM + i])
            if math_abs(expected - got) > 1e-4:
                fwd_ok = False
    check(fwd_ok, "Forward matches manual computation", fails)

    # Gradient check
    var max_err = fd_grad_check_generic[BATCH, DIM, DIM, Op.PARAM_SIZE, DIM](
        Op.eval[BATCH],
        Op.vjp[BATCH],
        inp,
        params,
        go_data,
    )
    print("  Max gradient error: " + String(max_err))
    check(max_err < 1e-3, "Gradient check (tol 1e-3)", fails)

    return fails


# =============================================================================
# Test 2: Scale
# =============================================================================

fn test_scale() -> Int:
    print_header("Scale forward + gradient check")
    var fails = 0

    comptime DIM = 4
    comptime BATCH = 2
    comptime Op = Scale[DIM, 1, 3]  # scale = 1/3

    check(Op.PARAM_SIZE == 0, "PARAM_SIZE == 0", fails)
    check(Op.CACHE_SIZE == 0, "CACHE_SIZE == 0", fails)

    seed(43)
    var inp = make_rand_list(BATCH * DIM)
    var params = make_list(0)
    var go_data = make_rand_list(BATCH * DIM)

    var out_data = make_list(BATCH * DIM)
    var cache_data = make_list(0)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin](
        params.unsafe_ptr()
    )
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Op.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())

    Op.eval[BATCH](inp_t, out_t, p_t, c_t)

    var scale_val = 1.0 / 3.0
    var fwd_ok = True
    for idx in range(BATCH * DIM):
        var expected = Float64(inp[idx]) * scale_val
        var got = Float64(out_data[idx])
        if math_abs(expected - got) > 1e-5:
            fwd_ok = False
    check(fwd_ok, "Forward matches x * (1/3)", fails)

    # Gradient check
    var max_err = fd_grad_check_generic[
        BATCH, DIM, DIM, Op.PARAM_SIZE, Op.CACHE_SIZE
    ](
        Op.eval[BATCH],
        Op.vjp[BATCH],
        inp,
        params,
        go_data,
    )
    print("  Max gradient error: " + String(max_err))
    check(max_err < 1e-3, "Gradient check (tol 1e-3)", fails)

    return fails


# =============================================================================
# Test 3: ElemMul
# =============================================================================

fn test_elem_mul() -> Int:
    print_header("ElemMul forward + gradient check")
    var fails = 0

    comptime DIM = 4
    comptime BATCH = 2
    comptime Op = ElemMul[DIM]

    check(Op.PARAM_SIZE == DIM, "PARAM_SIZE == dim", fails)
    check(Op.CACHE_SIZE == DIM, "CACHE_SIZE == dim", fails)

    seed(44)
    var inp = make_rand_list(BATCH * DIM)
    var params = make_rand_list(DIM)  # gamma
    var go_data = make_rand_list(BATCH * DIM)

    var out_data = make_list(BATCH * DIM)
    var cache_data = make_list(BATCH * DIM)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[dtype, Layout.row_major(DIM), MutAnyOrigin](
        params.unsafe_ptr()
    )
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](cache_data.unsafe_ptr())

    Op.eval[BATCH](inp_t, out_t, p_t, c_t)

    var fwd_ok = True
    for b in range(BATCH):
        for i in range(DIM):
            var expected = Float64(inp[b * DIM + i]) * Float64(params[i])
            var got = Float64(out_data[b * DIM + i])
            if math_abs(expected - got) > 1e-5:
                fwd_ok = False
    check(fwd_ok, "Forward matches x * gamma", fails)

    # Gradient check for input
    var max_err = fd_grad_check_generic[BATCH, DIM, DIM, DIM, DIM](
        Op.eval[BATCH],
        Op.vjp[BATCH],
        inp,
        params,
        go_data,
    )
    print("  Max input gradient error: " + String(max_err))
    check(max_err < 1e-3, "Input gradient check (tol 1e-3)", fails)

    return fails


# =============================================================================
# Test 4: ReduceSum
# =============================================================================

fn test_reduce_sum() -> Int:
    print_header("ReduceSum forward + gradient check")
    var fails = 0

    comptime DIM = 5
    comptime BATCH = 3
    comptime Op = ReduceSum[DIM]

    check(Op.IN_DIM == DIM, "IN_DIM == dim", fails)
    check(Op.OUT_DIM == 1, "OUT_DIM == 1", fails)
    check(Op.PARAM_SIZE == 0, "PARAM_SIZE == 0", fails)

    seed(45)
    var inp = make_rand_list(BATCH * DIM)
    var params = make_list(0)
    var go_data = make_rand_list(BATCH * 1)

    var out_data = make_list(BATCH * 1)
    var cache_data = make_list(0)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin](
        params.unsafe_ptr()
    )
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Op.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())

    Op.eval[BATCH](inp_t, out_t, p_t, c_t)

    var fwd_ok = True
    for b in range(BATCH):
        var expected: Float64 = 0.0
        for i in range(DIM):
            expected += Float64(inp[b * DIM + i])
        if math_abs(expected - Float64(out_data[b])) > 1e-4:
            fwd_ok = False
    check(fwd_ok, "Forward matches manual sum", fails)

    # Gradient check
    var max_err = fd_grad_check_generic[
        BATCH, DIM, 1, Op.PARAM_SIZE, Op.CACHE_SIZE
    ](
        Op.eval[BATCH],
        Op.vjp[BATCH],
        inp,
        params,
        go_data,
    )
    print("  Max gradient error: " + String(max_err))
    check(max_err < 1e-3, "Gradient check (tol 1e-3)", fails)

    return fails


# =============================================================================
# Test 5: ReduceMean
# =============================================================================

fn test_reduce_mean() -> Int:
    print_header("ReduceMean forward + gradient check")
    var fails = 0

    comptime DIM = 5
    comptime BATCH = 3
    comptime Op = ReduceMean[DIM]

    seed(46)
    var inp = make_rand_list(BATCH * DIM)
    var params = make_list(0)
    var go_data = make_rand_list(BATCH * 1)

    var out_data = make_list(BATCH * 1)
    var cache_data = make_list(0)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin](
        params.unsafe_ptr()
    )
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Op.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())

    Op.eval[BATCH](inp_t, out_t, p_t, c_t)

    var fwd_ok = True
    for b in range(BATCH):
        var s: Float64 = 0.0
        for i in range(DIM):
            s += Float64(inp[b * DIM + i])
        var expected = s / Float64(DIM)
        if math_abs(expected - Float64(out_data[b])) > 1e-4:
            fwd_ok = False
    check(fwd_ok, "Forward matches manual mean", fails)

    var max_err = fd_grad_check_generic[
        BATCH, DIM, 1, Op.PARAM_SIZE, Op.CACHE_SIZE
    ](
        Op.eval[BATCH],
        Op.vjp[BATCH],
        inp,
        params,
        go_data,
    )
    print("  Max gradient error: " + String(max_err))
    check(max_err < 1e-3, "Gradient check (tol 1e-3)", fails)

    return fails


# =============================================================================
# Test 6: SoftmaxOp
# =============================================================================

fn test_softmax() -> Int:
    print_header("SoftmaxOp forward + gradient check")
    var fails = 0

    comptime DIM = 4
    comptime BATCH = 2
    comptime Op = SoftmaxOp[DIM]

    check(Op.PARAM_SIZE == 0, "PARAM_SIZE == 0", fails)
    check(Op.CACHE_SIZE == DIM, "CACHE_SIZE == dim", fails)

    seed(47)
    var inp = make_rand_list(BATCH * DIM)
    var params = make_list(0)
    var go_data = make_rand_list(BATCH * DIM)

    var out_data = make_list(BATCH * DIM)
    var cache_data = make_list(BATCH * DIM)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin](
        params.unsafe_ptr()
    )
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](cache_data.unsafe_ptr())

    Op.eval[BATCH](inp_t, out_t, p_t, c_t)

    # Check: outputs sum to 1 and are positive
    var fwd_ok = True
    for b in range(BATCH):
        var s: Float64 = 0.0
        for i in range(DIM):
            var v = Float64(out_data[b * DIM + i])
            if v < 0:
                fwd_ok = False
            s += v
        if math_abs(s - 1.0) > 1e-4:
            fwd_ok = False
    check(fwd_ok, "Softmax outputs sum to 1 and are positive", fails)

    # Gradient check
    var max_err = fd_grad_check_generic[BATCH, DIM, DIM, Op.PARAM_SIZE, DIM](
        Op.eval[BATCH],
        Op.vjp[BATCH],
        inp,
        params,
        go_data,
    )
    print("  Max gradient error: " + String(max_err))
    check(max_err < 1e-3, "Gradient check (tol 1e-3)", fails)

    return fails


# =============================================================================
# Test 7: LayerNormOp
# =============================================================================

fn test_layer_norm() -> Int:
    print_header("LayerNormOp forward + gradient check")
    var fails = 0

    comptime DIM = 4
    comptime BATCH = 2
    comptime Op = LayerNormOp[DIM]

    check(Op.PARAM_SIZE == 2 * DIM, "PARAM_SIZE == 2*dim", fails)
    check(Op.CACHE_SIZE == DIM + 1, "CACHE_SIZE == dim + 1", fails)

    seed(48)
    var inp = make_rand_list(BATCH * DIM)
    # gamma = 1, beta = 0 (identity affine transform for easy verification)
    var params = make_list(2 * DIM)
    for i in range(DIM):
        params[i] = 1.0  # gamma = 1
        params[DIM + i] = 0.0  # beta = 0
    var go_data = make_rand_list(BATCH * DIM)

    var out_data = make_list(BATCH * DIM)
    var cache_data = make_list(BATCH * (DIM + 1))

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Op.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())

    Op.eval[BATCH](inp_t, out_t, p_t, c_t)

    # With gamma=1, beta=0: output should have mean ~0, var ~1
    var fwd_ok = True
    for b in range(BATCH):
        var mean: Float64 = 0.0
        for i in range(DIM):
            mean += Float64(out_data[b * DIM + i])
        mean /= Float64(DIM)
        var var_val: Float64 = 0.0
        for i in range(DIM):
            var diff = Float64(out_data[b * DIM + i]) - mean
            var_val += diff * diff
        var_val /= Float64(DIM)
        if math_abs(mean) > 1e-4 or math_abs(var_val - 1.0) > 1e-3:
            fwd_ok = False
    check(fwd_ok, "Output has mean ~0, var ~1 (gamma=1, beta=0)", fails)

    # Gradient check with non-trivial gamma/beta
    seed(49)
    var params2 = make_rand_list(2 * DIM)
    var max_err = fd_grad_check_generic[
        BATCH, DIM, DIM, Op.PARAM_SIZE, Op.CACHE_SIZE
    ](
        Op.eval[BATCH],
        Op.vjp[BATCH],
        inp,
        params2,
        go_data,
    )
    print("  Max input gradient error: " + String(max_err))
    check(max_err < 1e-3, "Input gradient check (tol 1e-3)", fails)

    return fails


# =============================================================================
# Test 8: RMSNormOp
# =============================================================================

fn test_rms_norm() -> Int:
    print_header("RMSNormOp forward + gradient check")
    var fails = 0

    comptime DIM = 4
    comptime BATCH = 2
    comptime Op = RMSNormOp[DIM]

    check(Op.PARAM_SIZE == DIM, "PARAM_SIZE == dim", fails)
    check(Op.CACHE_SIZE == DIM + 1, "CACHE_SIZE == dim + 1", fails)

    seed(50)
    var inp = make_rand_list(BATCH * DIM)
    # gamma = 1 for easy verification
    var params = make_list(DIM)
    for i in range(DIM):
        params[i] = 1.0
    var go_data = make_rand_list(BATCH * DIM)

    var out_data = make_list(BATCH * DIM)
    var cache_data = make_list(BATCH * (DIM + 1))

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[dtype, Layout.row_major(DIM), MutAnyOrigin](
        params.unsafe_ptr()
    )
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, Op.CACHE_SIZE), MutAnyOrigin
    ](cache_data.unsafe_ptr())

    Op.eval[BATCH](inp_t, out_t, p_t, c_t)

    # With gamma=1: output = x / rms(x), so mean(output^2) should be ~1
    var fwd_ok = True
    for b in range(BATCH):
        var mean_sq: Float64 = 0.0
        for i in range(DIM):
            var v = Float64(out_data[b * DIM + i])
            mean_sq += v * v
        mean_sq /= Float64(DIM)
        if math_abs(mean_sq - 1.0) > 1e-3:
            fwd_ok = False
    check(fwd_ok, "Output has mean(x^2) ~1 (gamma=1)", fails)

    # Gradient check
    seed(51)
    var params2 = make_rand_list(DIM)
    var max_err = fd_grad_check_generic[
        BATCH, DIM, DIM, DIM, Op.CACHE_SIZE
    ](
        Op.eval[BATCH],
        Op.vjp[BATCH],
        inp,
        params2,
        go_data,
    )
    print("  Max input gradient error: " + String(max_err))
    check(max_err < 1e-3, "Input gradient check (tol 1e-3)", fails)

    return fails


# =============================================================================
# Test 9: AutoDiffChain composition test
# =============================================================================

fn test_chain_composition() -> Int:
    print_header("AutoDiffChain composition with new ops")
    var fails = 0

    # Chain: MatMul[3,4] -> SoftmaxOp[4] should compile and have correct dims
    comptime Chain1 = AutoDiffChain[MatMul[3, 4], SoftmaxOp[4]]
    check(Chain1.IN_DIM == 3, "MatMul->Softmax IN_DIM == 3", fails)
    check(Chain1.OUT_DIM == 4, "MatMul->Softmax OUT_DIM == 4", fails)

    # Chain: MatMul[3,4] -> LayerNormOp[4]
    comptime Chain2 = AutoDiffChain[MatMul[3, 4], LayerNormOp[4]]
    check(Chain2.IN_DIM == 3, "MatMul->LayerNorm IN_DIM == 3", fails)
    check(Chain2.OUT_DIM == 4, "MatMul->LayerNorm OUT_DIM == 4", fails)
    # PARAM_SIZE = MatMul(3*4=12) + LayerNorm(2*4=8) = 20
    check(
        Chain2.PARAM_SIZE == 20, "MatMul->LayerNorm PARAM_SIZE == 20", fails
    )

    # Chain: MatMul[3,4] -> RMSNormOp[4] -> MishOp[4]
    comptime Chain3 = AutoDiffChain[MatMul[3, 4], RMSNormOp[4], MishOp[4]]
    check(Chain3.IN_DIM == 3, "MatMul->RMSNorm->Mish IN_DIM == 3", fails)
    check(Chain3.OUT_DIM == 4, "MatMul->RMSNorm->Mish OUT_DIM == 4", fails)
    # PARAM_SIZE = MatMul(12) + RMSNorm(4) + Mish(0) = 16
    check(
        Chain3.PARAM_SIZE == 16,
        "MatMul->RMSNorm->Mish PARAM_SIZE == 16",
        fails,
    )

    # Chain with ReduceMean changes output dimension
    comptime Chain4 = AutoDiffChain[MatMul[3, 4], ReduceMean[4]]
    check(Chain4.IN_DIM == 3, "MatMul->ReduceMean IN_DIM == 3", fails)
    check(Chain4.OUT_DIM == 1, "MatMul->ReduceMean OUT_DIM == 1", fails)

    return fails


# =============================================================================
# main
# =============================================================================

fn main():
    print("=" * 70)
    print("Phase 3 AutoDiff Primitives — Verification Tests")
    print("=" * 70)

    var total_fails = 0
    total_fails += test_mish()
    total_fails += test_scale()
    total_fails += test_elem_mul()
    total_fails += test_reduce_sum()
    total_fails += test_reduce_mean()
    total_fails += test_softmax()
    total_fails += test_layer_norm()
    total_fails += test_rms_norm()
    total_fails += test_chain_composition()

    print("\n" + "=" * 70)
    if total_fails == 0:
        print("ALL TESTS PASSED")
    else:
        print(String(total_fails) + " FAILURE(S)")
    print("=" * 70)

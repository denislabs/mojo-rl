"""Phase 4 verification tests for combinator Models.

Tests: Residual, Parallel, Repeat.

Per combinator: (1) dimension checks, (2) forward correctness,
(3) finite-difference gradient check, (4) composition & training convergence.

Run with:
    pixi run mojo run tests/test_autodiff_phase4.mojo
"""

from std.random import seed, random_float64
from std.math import abs as math_abs

from nn.constants import dtype
from nn.autodiff import (
    AutoDiffChain,
    Dense,
    DenseReLU,
    MatMul,
    BiasAdd,
    ReLUOp,
    FusedMatMulBias,
    FusedMatMulBiasReLU,
    Residual,
    Parallel,
    Repeat,
)
from nn.model.sequential import Sequential
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
# Generic finite-difference gradient check for Model trait
# =============================================================================

fn fd_grad_check_model[
    BATCH: Int, IN_DIM: Int, OUT_DIM: Int, PARAM_SIZE: Int, CACHE_SIZE: Int
](
    fwd_fn: fn (
        LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
        mut LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ],
        LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
        mut LayoutTensor[
            dtype, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin
        ],
    ) -> None,
    bwd_fn: fn (
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
    """FD gradient check for input gradients. Returns max abs error."""
    var n_inp = BATCH * IN_DIM

    # Analytic gradients
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

    fwd_fn(inp_t, out_t, p_t, c_t)
    bwd_fn(go_t, gi_t, p_t, c_t, gp_t)

    # Numerical gradients (input)
    var max_err: Float64 = 0.0
    for idx in range(n_inp):
        var orig = inp[idx]

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
        fwd_fn(inp_p, out_p, p_t, c_p)

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
        fwd_fn(inp_m, out_m, p_t, c_m)

        inp[idx] = orig

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


# FD check for parameter gradients
fn fd_param_grad_check[
    BATCH: Int, IN_DIM: Int, OUT_DIM: Int, PARAM_SIZE: Int, CACHE_SIZE: Int
](
    fwd_fn: fn (
        LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
        mut LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ],
        LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
        mut LayoutTensor[
            dtype, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin
        ],
    ) -> None,
    bwd_fn: fn (
        LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
        mut LayoutTensor[
            dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin
        ],
        LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
        LayoutTensor[dtype, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin],
        mut LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
    ) -> None,
    inp: List[Scalar[dtype]],
    mut params: List[Scalar[dtype]],
    go_data: List[Scalar[dtype]],
    eps: Float64 = 1e-4,
) -> Float64:
    """FD gradient check for parameter gradients. Returns max abs error."""

    # Analytic gradients
    var out_data = make_list(BATCH * OUT_DIM)
    var cache_data = make_list(BATCH * CACHE_SIZE)
    var gi_data = make_list(BATCH * IN_DIM)
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

    fwd_fn(inp_t, out_t, p_t, c_t)
    bwd_fn(go_t, gi_t, p_t, c_t, gp_t)

    # Numerical param gradients
    var max_err: Float64 = 0.0
    for idx in range(PARAM_SIZE):
        var orig = params[idx]

        params[idx] = orig + Scalar[dtype](eps)
        var out_plus = make_list(BATCH * OUT_DIM)
        var cache_plus = make_list(BATCH * CACHE_SIZE)
        var p_plus = LayoutTensor[
            dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
        ](params.unsafe_ptr())
        var out_p = LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ](out_plus.unsafe_ptr())
        var c_p = LayoutTensor[
            dtype, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin
        ](cache_plus.unsafe_ptr())
        fwd_fn(inp_t, out_p, p_plus, c_p)

        params[idx] = orig - Scalar[dtype](eps)
        var out_minus = make_list(BATCH * OUT_DIM)
        var cache_minus = make_list(BATCH * CACHE_SIZE)
        var p_minus = LayoutTensor[
            dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
        ](params.unsafe_ptr())
        var out_m = LayoutTensor[
            dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
        ](out_minus.unsafe_ptr())
        var c_m = LayoutTensor[
            dtype, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin
        ](cache_minus.unsafe_ptr())
        fwd_fn(inp_t, out_m, p_minus, c_m)

        params[idx] = orig

        var num_grad: Float64 = 0.0
        for j in range(BATCH * OUT_DIM):
            num_grad += Float64(go_data[j]) * (
                Float64(out_plus[j]) - Float64(out_minus[j])
            ) / (2.0 * eps)

        var analytic = Float64(gp_data[idx])
        var err = math_abs(analytic - num_grad)
        if err > max_err:
            max_err = err

    return max_err


# =============================================================================
# Test 1: Residual dimension checks and forward
# =============================================================================

fn test_residual_dims_and_forward() -> Int:
    print_header("Residual[Dense[4,4]] — dimensions + forward")
    var fails = 0

    comptime R = Residual[Dense[4, 4]]

    check(R.IN_DIM == 4, "IN_DIM == 4", fails)
    check(R.OUT_DIM == 4, "OUT_DIM == 4", fails)
    # Dense[4,4] PARAM_SIZE = 4*4 + 4 = 20
    check(R.PARAM_SIZE == 20, "PARAM_SIZE == 20 (4*4 W + 4 b)", fails)
    check(
        R.CACHE_SIZE == Dense[4, 4].CACHE_SIZE,
        "CACHE_SIZE matches inner",
        fails,
    )

    seed(100)
    comptime BATCH = 2
    var inp = make_rand_list(BATCH * R.IN_DIM)
    var params = make_rand_list(R.PARAM_SIZE)

    # Compute reference: Dense(x) + x manually
    # Use the FD helper's pattern: create tensors with Model's own comptime dims
    var dense_out = make_list(BATCH * R.OUT_DIM)
    var dense_cache = make_list(BATCH * Dense[4, 4].CACHE_SIZE)
    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, R.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var dense_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, R.OUT_DIM), MutAnyOrigin
    ](dense_out.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(R.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var dc_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, Dense[4, 4].CACHE_SIZE),
        MutAnyOrigin,
    ](dense_cache.unsafe_ptr())
    Dense[4, 4].forward[BATCH](inp_t, dense_out_t, p_t, dc_t)

    # Add skip: expected = dense_out + inp
    var expected = make_list(BATCH * R.OUT_DIM)
    for i in range(BATCH * R.OUT_DIM):
        expected[i] = dense_out[i] + inp[i]

    # Residual forward
    var res_out = make_list(BATCH * R.OUT_DIM)
    var res_cache = make_list(BATCH * R.CACHE_SIZE)
    var res_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, R.OUT_DIM), MutAnyOrigin
    ](res_out.unsafe_ptr())
    var rc_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, R.CACHE_SIZE), MutAnyOrigin
    ](res_cache.unsafe_ptr())
    R.forward[BATCH](inp_t, res_out_t, p_t, rc_t)

    var md = max_diff(expected, res_out, BATCH * R.OUT_DIM)
    print("  Max forward diff: " + String(md))
    check(md < 1e-5, "Forward matches Dense(x) + x", fails)

    return fails


# =============================================================================
# Test 2: Residual gradient check
# =============================================================================

fn test_residual_grad() -> Int:
    print_header("Residual[Dense[4,4]] — gradient check")
    var fails = 0

    comptime R = Residual[Dense[4, 4]]
    comptime BATCH = 2

    seed(101)
    var inp = make_rand_list(BATCH * R.IN_DIM)
    var params = make_rand_list(R.PARAM_SIZE)
    var go_data = make_rand_list(BATCH * R.OUT_DIM)

    # Input gradient check
    var max_err = fd_grad_check_model[
        BATCH, R.IN_DIM, R.OUT_DIM, R.PARAM_SIZE, R.CACHE_SIZE
    ](R.forward[BATCH], R.backward[BATCH], inp, params, go_data)
    print("  Max input gradient error: " + String(max_err))
    check(max_err < 1e-3, "Input gradient check (tol 1e-3)", fails)

    # Param gradient check
    var max_perr = fd_param_grad_check[
        BATCH, R.IN_DIM, R.OUT_DIM, R.PARAM_SIZE, R.CACHE_SIZE
    ](R.forward[BATCH], R.backward[BATCH], inp, params, go_data)
    print("  Max param gradient error: " + String(max_perr))
    check(max_perr < 1e-3, "Param gradient check (tol 1e-3)", fails)

    return fails


# =============================================================================
# Test 3: Parallel dimension checks and forward
# =============================================================================

fn test_parallel_dims_and_forward() -> Int:
    print_header("Parallel[Dense[2,3], Dense[2,2]] — dimensions + forward")
    var fails = 0

    comptime P = Parallel[Dense[2, 3], Dense[2, 2]]

    check(P.IN_DIM == 2, "IN_DIM == 2", fails)
    check(P.OUT_DIM == 5, "OUT_DIM == 3 + 2 = 5", fails)
    # Dense[2,3] params = 2*3+3 = 9, Dense[2,2] params = 2*2+2 = 6
    check(P.PARAM_SIZE == 15, "PARAM_SIZE == 9 + 6 = 15", fails)

    seed(102)
    comptime BATCH = 2
    var inp = make_rand_list(BATCH * P.IN_DIM)
    var params = make_rand_list(P.PARAM_SIZE)

    # Compute reference: concat(A(x), B(x))
    comptime A = Dense[2, 3]
    comptime B = Dense[2, 2]

    var out_a = make_list(BATCH * A.OUT_DIM)
    var out_b = make_list(BATCH * B.OUT_DIM)
    var cache_a = make_list(BATCH * A.CACHE_SIZE)
    var cache_b = make_list(BATCH * B.CACHE_SIZE)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, P.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_a_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, A.OUT_DIM), MutAnyOrigin
    ](out_a.unsafe_ptr())
    var out_b_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, B.OUT_DIM), MutAnyOrigin
    ](out_b.unsafe_ptr())
    var pa_t = LayoutTensor[
        dtype, Layout.row_major(A.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var pb_t = LayoutTensor[
        dtype, Layout.row_major(B.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr() + A.PARAM_SIZE)
    var ca_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, A.CACHE_SIZE), MutAnyOrigin
    ](cache_a.unsafe_ptr())
    var cb_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, B.CACHE_SIZE), MutAnyOrigin
    ](cache_b.unsafe_ptr())

    # Rebind inp for B since P.IN_DIM and B.IN_DIM are same value but different types
    var inp_b = rebind[
        LayoutTensor[
            dtype, Layout.row_major(BATCH, B.IN_DIM), MutAnyOrigin
        ]
    ](inp_t)
    A.forward[BATCH](inp_t, out_a_t, pa_t, ca_t)
    B.forward[BATCH](inp_b, out_b_t, pb_t, cb_t)

    # Build expected concat
    var expected = make_list(BATCH * P.OUT_DIM)
    for b_idx in range(BATCH):
        for i in range(A.OUT_DIM):
            expected[b_idx * P.OUT_DIM + i] = out_a[b_idx * A.OUT_DIM + i]
        for i in range(B.OUT_DIM):
            expected[b_idx * P.OUT_DIM + A.OUT_DIM + i] = out_b[
                b_idx * B.OUT_DIM + i
            ]

    # Parallel forward
    var par_out = make_list(BATCH * P.OUT_DIM)
    var par_cache = make_list(BATCH * P.CACHE_SIZE)
    var par_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, P.OUT_DIM), MutAnyOrigin
    ](par_out.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(P.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var pc_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, P.CACHE_SIZE), MutAnyOrigin
    ](par_cache.unsafe_ptr())
    P.forward[BATCH](inp_t, par_out_t, p_t, pc_t)

    var md = max_diff(expected, par_out, BATCH * P.OUT_DIM)
    print("  Max forward diff: " + String(md))
    check(md < 1e-5, "Forward matches concat(A(x), B(x))", fails)

    return fails


# =============================================================================
# Test 4: Parallel gradient check
# =============================================================================

fn test_parallel_grad() -> Int:
    print_header("Parallel[Dense[2,3], Dense[2,2]] — gradient check")
    var fails = 0

    comptime P = Parallel[Dense[2, 3], Dense[2, 2]]
    comptime BATCH = 2

    seed(103)
    var inp = make_rand_list(BATCH * P.IN_DIM)
    var params = make_rand_list(P.PARAM_SIZE)
    var go_data = make_rand_list(BATCH * P.OUT_DIM)

    var max_err = fd_grad_check_model[
        BATCH, P.IN_DIM, P.OUT_DIM, P.PARAM_SIZE, P.CACHE_SIZE
    ](P.forward[BATCH], P.backward[BATCH], inp, params, go_data)
    print("  Max input gradient error: " + String(max_err))
    check(max_err < 1e-3, "Input gradient check (tol 1e-3)", fails)

    var max_perr = fd_param_grad_check[
        BATCH, P.IN_DIM, P.OUT_DIM, P.PARAM_SIZE, P.CACHE_SIZE
    ](P.forward[BATCH], P.backward[BATCH], inp, params, go_data)
    print("  Max param gradient error: " + String(max_perr))
    check(max_perr < 1e-3, "Param gradient check (tol 1e-3)", fails)

    return fails


# =============================================================================
# Test 5: Repeat dimension checks and forward
# =============================================================================

fn test_repeat_dims_and_forward() -> Int:
    print_header("Repeat[3, Dense[4,4]] — dimensions + forward")
    var fails = 0

    comptime R3 = Repeat[3, Dense[4, 4]]
    comptime D = Dense[4, 4]

    check(R3.IN_DIM == 4, "IN_DIM == 4", fails)
    check(R3.OUT_DIM == 4, "OUT_DIM == 4", fails)
    check(R3.PARAM_SIZE == D.PARAM_SIZE, "PARAM_SIZE shared with Dense", fails)
    check(
        R3.CACHE_SIZE == 3 * D.CACHE_SIZE,
        "CACHE_SIZE == 3 * inner cache",
        fails,
    )

    seed(104)
    comptime BATCH = 2
    var inp = make_rand_list(BATCH * R3.IN_DIM)
    var params = make_rand_list(R3.PARAM_SIZE)

    # Compute reference: apply Dense 3 times manually
    var buf1 = make_list(BATCH * D.OUT_DIM)
    var buf2 = make_list(BATCH * D.OUT_DIM)
    var buf3 = make_list(BATCH * D.OUT_DIM)
    var c1 = make_list(BATCH * D.CACHE_SIZE)
    var c2 = make_list(BATCH * D.CACHE_SIZE)
    var c3 = make_list(BATCH * D.CACHE_SIZE)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, R3.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(R3.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())

    var b1_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D.OUT_DIM), MutAnyOrigin
    ](buf1.unsafe_ptr())
    var c1_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D.CACHE_SIZE), MutAnyOrigin
    ](c1.unsafe_ptr())
    D.forward[BATCH](inp_t, b1_t, p_t, c1_t)

    var b2_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D.OUT_DIM), MutAnyOrigin
    ](buf2.unsafe_ptr())
    var c2_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D.CACHE_SIZE), MutAnyOrigin
    ](c2.unsafe_ptr())
    D.forward[BATCH](b1_t, b2_t, p_t, c2_t)

    var b3_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D.OUT_DIM), MutAnyOrigin
    ](buf3.unsafe_ptr())
    var c3_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D.CACHE_SIZE), MutAnyOrigin
    ](c3.unsafe_ptr())
    D.forward[BATCH](b2_t, b3_t, p_t, c3_t)

    # Repeat forward
    var rep_out = make_list(BATCH * R3.OUT_DIM)
    var rep_cache = make_list(BATCH * R3.CACHE_SIZE)
    var rep_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, R3.OUT_DIM), MutAnyOrigin
    ](rep_out.unsafe_ptr())
    var rc_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, R3.CACHE_SIZE), MutAnyOrigin
    ](rep_cache.unsafe_ptr())
    R3.forward[BATCH](inp_t, rep_out_t, p_t, rc_t)

    var md = max_diff(buf3, rep_out, BATCH * R3.OUT_DIM)
    print("  Max forward diff: " + String(md))
    check(md < 1e-5, "Forward matches 3x manual application", fails)

    return fails


# =============================================================================
# Test 6: Repeat gradient check (shared weight accumulation)
# =============================================================================

fn test_repeat_grad() -> Int:
    print_header("Repeat[3, Dense[4,4]] — gradient check (shared weights)")
    var fails = 0

    comptime R3 = Repeat[3, Dense[4, 4]]
    comptime BATCH = 2

    seed(105)
    var inp = make_rand_list(BATCH * R3.IN_DIM)
    var params = make_rand_list(R3.PARAM_SIZE)
    var go_data = make_rand_list(BATCH * R3.OUT_DIM)

    var max_err = fd_grad_check_model[
        BATCH, R3.IN_DIM, R3.OUT_DIM, R3.PARAM_SIZE, R3.CACHE_SIZE
    ](R3.forward[BATCH], R3.backward[BATCH], inp, params, go_data)
    print("  Max input gradient error: " + String(max_err))
    check(max_err < 1e-3, "Input gradient check (tol 1e-3)", fails)

    # Param gradient check — verifies shared weight grad accumulation
    var max_perr = fd_param_grad_check[
        BATCH, R3.IN_DIM, R3.OUT_DIM, R3.PARAM_SIZE, R3.CACHE_SIZE
    ](R3.forward[BATCH], R3.backward[BATCH], inp, params, go_data)
    print("  Max param gradient error: " + String(max_perr))
    check(
        max_perr < 2e-3,
        "Param gradient check (shared weights, tol 2e-3)",
        fails,
    )

    return fails


# =============================================================================
# Test 7: Composition — Sequential[DenseReLU, Residual[Dense], Dense]
# =============================================================================

fn test_composition_training() -> Int:
    print_header("Composition: Sequential + Residual — XOR convergence")
    var fails = 0

    # Model: DenseReLU[2,8] -> Residual[Dense[8,8]] -> Dense[8,1]
    comptime M = Sequential[
        DenseReLU[2, 8], Residual[Dense[8, 8]], Dense[8, 1]
    ]

    check(M.IN_DIM == 2, "Composed IN_DIM == 2", fails)
    check(M.OUT_DIM == 1, "Composed OUT_DIM == 1", fails)
    print(
        "  PARAM_SIZE = "
        + String(M.PARAM_SIZE)
        + ", CACHE_SIZE = "
        + String(M.CACHE_SIZE)
    )

    # XOR data
    comptime BATCH = 4
    var inp_data = List[Scalar[dtype]](capacity=8)
    # [0,0], [0,1], [1,0], [1,1]
    inp_data.append(0)
    inp_data.append(0)
    inp_data.append(0)
    inp_data.append(1)
    inp_data.append(1)
    inp_data.append(0)
    inp_data.append(1)
    inp_data.append(1)

    var target_data = List[Scalar[dtype]](capacity=4)
    target_data.append(0)
    target_data.append(1)
    target_data.append(1)
    target_data.append(0)

    seed(200)
    var params = make_rand_list(M.PARAM_SIZE)
    # Scale params small
    for i in range(M.PARAM_SIZE):
        params[i] = params[i] * 0.1

    var grads = make_list(M.PARAM_SIZE)

    comptime LR: Float64 = 0.01
    comptime EPOCHS = 3000

    var final_loss: Float64 = 999.0

    for epoch in range(EPOCHS):
        # Zero grads
        for i in range(M.PARAM_SIZE):
            grads[i] = 0

        var out_data = make_list(BATCH * M.OUT_DIM)
        var cache_data = make_list(BATCH * M.CACHE_SIZE)

        var inp_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
        ](inp_data.unsafe_ptr())
        var out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
        ](out_data.unsafe_ptr())
        var p_t = LayoutTensor[
            dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
        ](params.unsafe_ptr())
        var c_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
        ](cache_data.unsafe_ptr())

        M.forward[BATCH](inp_t, out_t, p_t, c_t)

        # MSE loss + backward
        var loss: Float64 = 0.0
        var go_data = make_list(BATCH * M.OUT_DIM)
        for b_idx in range(BATCH):
            var diff = Float64(out_data[b_idx]) - Float64(
                target_data[b_idx]
            )
            loss += diff * diff
            go_data[b_idx] = Scalar[dtype](
                2.0 * diff / Float64(BATCH)
            )
        loss /= Float64(BATCH)

        var gi_data = make_list(BATCH * M.IN_DIM)
        var go_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
        ](go_data.unsafe_ptr())
        var gi_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
        ](gi_data.unsafe_ptr())
        var g_t = LayoutTensor[
            dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
        ](grads.unsafe_ptr())

        M.backward[BATCH](go_t, gi_t, p_t, c_t, g_t)

        # SGD update
        for i in range(M.PARAM_SIZE):
            params[i] = params[i] - Scalar[dtype](LR) * grads[i]

        final_loss = loss

    print("  Final loss: " + String(final_loss))
    check(final_loss < 0.05, "XOR converges (loss < 0.05)", fails)

    return fails


# =============================================================================
# Test 8: Nesting — Residual[Sequential[DenseReLU, Dense]]
# =============================================================================

fn test_nested_residual() -> Int:
    print_header(
        "Residual[Sequential[DenseReLU[4,4], Dense[4,4]]] — grad check"
    )
    var fails = 0

    comptime R = Residual[Sequential[DenseReLU[4, 4], Dense[4, 4]]]
    comptime BATCH = 2

    check(R.IN_DIM == 4, "IN_DIM == 4", fails)
    check(R.OUT_DIM == 4, "OUT_DIM == 4", fails)

    seed(106)
    var inp = make_rand_list(BATCH * R.IN_DIM)
    var params = make_rand_list(R.PARAM_SIZE)
    var go_data = make_rand_list(BATCH * R.OUT_DIM)

    var max_err = fd_grad_check_model[
        BATCH, R.IN_DIM, R.OUT_DIM, R.PARAM_SIZE, R.CACHE_SIZE
    ](R.forward[BATCH], R.backward[BATCH], inp, params, go_data)
    print("  Max input gradient error: " + String(max_err))
    check(max_err < 1e-3, "Input gradient check (tol 1e-3)", fails)

    var max_perr = fd_param_grad_check[
        BATCH, R.IN_DIM, R.OUT_DIM, R.PARAM_SIZE, R.CACHE_SIZE
    ](R.forward[BATCH], R.backward[BATCH], inp, params, go_data)
    print("  Max param gradient error: " + String(max_perr))
    check(max_perr < 1e-3, "Param gradient check (tol 1e-3)", fails)

    return fails


# =============================================================================
# Test 9: Repeat[1, ...] is identity to inner
# =============================================================================

fn test_repeat_one() -> Int:
    print_header("Repeat[1, Dense[4,4]] — matches Dense directly")
    var fails = 0

    comptime R1 = Repeat[1, Dense[4, 4]]
    comptime D = Dense[4, 4]
    comptime BATCH = 2

    check(R1.PARAM_SIZE == D.PARAM_SIZE, "PARAM_SIZE matches", fails)
    check(R1.CACHE_SIZE == D.CACHE_SIZE, "CACHE_SIZE matches", fails)

    seed(107)
    var inp = make_rand_list(BATCH * R1.IN_DIM)
    var params = make_rand_list(R1.PARAM_SIZE)

    var out_r1 = make_list(BATCH * R1.OUT_DIM)
    var cache_r1 = make_list(BATCH * R1.CACHE_SIZE)
    var out_d = make_list(BATCH * D.OUT_DIM)
    var cache_d = make_list(BATCH * D.CACHE_SIZE)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, R1.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(R1.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())

    var or1_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, R1.OUT_DIM), MutAnyOrigin
    ](out_r1.unsafe_ptr())
    var cr1_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, R1.CACHE_SIZE), MutAnyOrigin
    ](cache_r1.unsafe_ptr())
    R1.forward[BATCH](inp_t, or1_t, p_t, cr1_t)

    var od_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D.OUT_DIM), MutAnyOrigin
    ](out_d.unsafe_ptr())
    var cd_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, D.CACHE_SIZE), MutAnyOrigin
    ](cache_d.unsafe_ptr())
    D.forward[BATCH](inp_t, od_t, p_t, cd_t)

    var md = max_diff(out_r1, out_d, BATCH * R1.OUT_DIM)
    print("  Max diff: " + String(md))
    check(md < 1e-6, "Repeat[1] == Dense forward", fails)

    return fails


# =============================================================================
# Test 10: 3-branch Parallel dimension checks and forward
# =============================================================================

fn test_parallel_3branch_dims_and_forward() -> Int:
    print_header(
        "Parallel[Dense[2,3], Dense[2,2], Dense[2,1]] — dimensions + forward"
    )
    var fails = 0

    comptime P3 = Parallel[Dense[2, 3], Dense[2, 2], Dense[2, 1]]

    check(P3.IN_DIM == 2, "IN_DIM == 2", fails)
    check(P3.OUT_DIM == 6, "OUT_DIM == 3 + 2 + 1 = 6", fails)
    # Dense[2,3] params = 2*3+3 = 9, Dense[2,2] = 2*2+2 = 6, Dense[2,1] = 2*1+1 = 3
    check(P3.PARAM_SIZE == 18, "PARAM_SIZE == 9 + 6 + 3 = 18", fails)

    seed(108)
    comptime BATCH = 2
    var inp = make_rand_list(BATCH * P3.IN_DIM)
    var params = make_rand_list(P3.PARAM_SIZE)

    # Compute reference: concat(A(x), B(x), C(x))
    comptime A = Dense[2, 3]
    comptime B = Dense[2, 2]
    comptime C = Dense[2, 1]

    var out_a = make_list(BATCH * A.OUT_DIM)
    var out_b = make_list(BATCH * B.OUT_DIM)
    var out_c = make_list(BATCH * C.OUT_DIM)
    var cache_a = make_list(BATCH * A.CACHE_SIZE)
    var cache_b = make_list(BATCH * B.CACHE_SIZE)
    var cache_c = make_list(BATCH * C.CACHE_SIZE)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, P3.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())

    var out_a_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, A.OUT_DIM), MutAnyOrigin
    ](out_a.unsafe_ptr())
    var out_b_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, B.OUT_DIM), MutAnyOrigin
    ](out_b.unsafe_ptr())
    var out_c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.OUT_DIM), MutAnyOrigin
    ](out_c.unsafe_ptr())

    var pa_t = LayoutTensor[
        dtype, Layout.row_major(A.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var pb_t = LayoutTensor[
        dtype, Layout.row_major(B.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr() + A.PARAM_SIZE)
    var pc_t = LayoutTensor[
        dtype, Layout.row_major(C.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr() + A.PARAM_SIZE + B.PARAM_SIZE)

    var ca_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, A.CACHE_SIZE), MutAnyOrigin
    ](cache_a.unsafe_ptr())
    var cb_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, B.CACHE_SIZE), MutAnyOrigin
    ](cache_b.unsafe_ptr())
    var cc_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, C.CACHE_SIZE), MutAnyOrigin
    ](cache_c.unsafe_ptr())

    var inp_b = rebind[
        LayoutTensor[
            dtype, Layout.row_major(BATCH, B.IN_DIM), MutAnyOrigin
        ]
    ](inp_t)
    var inp_c = rebind[
        LayoutTensor[
            dtype, Layout.row_major(BATCH, C.IN_DIM), MutAnyOrigin
        ]
    ](inp_t)

    A.forward[BATCH](inp_t, out_a_t, pa_t, ca_t)
    B.forward[BATCH](inp_b, out_b_t, pb_t, cb_t)
    C.forward[BATCH](inp_c, out_c_t, pc_t, cc_t)

    # Build expected concat
    var expected = make_list(BATCH * P3.OUT_DIM)
    for b_idx in range(BATCH):
        for i in range(A.OUT_DIM):
            expected[b_idx * P3.OUT_DIM + i] = out_a[b_idx * A.OUT_DIM + i]
        for i in range(B.OUT_DIM):
            expected[b_idx * P3.OUT_DIM + A.OUT_DIM + i] = out_b[
                b_idx * B.OUT_DIM + i
            ]
        for i in range(C.OUT_DIM):
            expected[
                b_idx * P3.OUT_DIM + A.OUT_DIM + B.OUT_DIM + i
            ] = out_c[b_idx * C.OUT_DIM + i]

    # Parallel forward
    var par_out = make_list(BATCH * P3.OUT_DIM)
    var par_cache = make_list(BATCH * P3.CACHE_SIZE)
    var par_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, P3.OUT_DIM), MutAnyOrigin
    ](par_out.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(P3.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var pcache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, P3.CACHE_SIZE), MutAnyOrigin
    ](par_cache.unsafe_ptr())
    P3.forward[BATCH](inp_t, par_out_t, p_t, pcache_t)

    var md = max_diff(expected, par_out, BATCH * P3.OUT_DIM)
    print("  Max forward diff: " + String(md))
    check(md < 1e-5, "Forward matches concat(A(x), B(x), C(x))", fails)

    return fails


# =============================================================================
# Test 11: 3-branch Parallel gradient check
# =============================================================================

fn test_parallel_3branch_grad() -> Int:
    print_header(
        "Parallel[Dense[2,3], Dense[2,2], Dense[2,1]] — gradient check"
    )
    var fails = 0

    comptime P3 = Parallel[Dense[2, 3], Dense[2, 2], Dense[2, 1]]
    comptime BATCH = 2

    seed(109)
    var inp = make_rand_list(BATCH * P3.IN_DIM)
    var params = make_rand_list(P3.PARAM_SIZE)
    var go_data = make_rand_list(BATCH * P3.OUT_DIM)

    var max_err = fd_grad_check_model[
        BATCH, P3.IN_DIM, P3.OUT_DIM, P3.PARAM_SIZE, P3.CACHE_SIZE
    ](P3.forward[BATCH], P3.backward[BATCH], inp, params, go_data)
    print("  Max input gradient error: " + String(max_err))
    check(max_err < 1e-3, "Input gradient check (tol 1e-3)", fails)

    var max_perr = fd_param_grad_check[
        BATCH, P3.IN_DIM, P3.OUT_DIM, P3.PARAM_SIZE, P3.CACHE_SIZE
    ](P3.forward[BATCH], P3.backward[BATCH], inp, params, go_data)
    print("  Max param gradient error: " + String(max_perr))
    check(max_perr < 1e-3, "Param gradient check (tol 1e-3)", fails)

    return fails


# =============================================================================
# main
# =============================================================================

fn main():
    print("=" * 70)
    print("Phase 4 AutoDiff Combinators — Verification Tests")
    print("=" * 70)

    var total_fails = 0
    total_fails += test_residual_dims_and_forward()
    total_fails += test_residual_grad()
    total_fails += test_parallel_dims_and_forward()
    total_fails += test_parallel_grad()
    total_fails += test_parallel_3branch_dims_and_forward()
    total_fails += test_parallel_3branch_grad()
    total_fails += test_repeat_dims_and_forward()
    total_fails += test_repeat_grad()
    total_fails += test_composition_training()
    total_fails += test_nested_residual()
    total_fails += test_repeat_one()

    print("\n" + "=" * 70)
    if total_fails == 0:
        print("ALL TESTS PASSED")
    else:
        print(String(total_fails) + " FAILURE(S)")
    print("=" * 70)

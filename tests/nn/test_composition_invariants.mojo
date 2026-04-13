"""Level 5: Composition invariant tests.

Verifies algebraic properties of combinators by running the composed model
and comparing against manual step-by-step computation with the same params.

Properties tested:
    Residual[A]:     y == x + A(x)
    SkipConcat[A]:   y == cat(x, A(x))
    Repeat[1, A]:    y == A(x)
    Repeat[2, A]:    y == A(A(x))  (shared weights)
    Sequential[A,B]: y == B(A(x))
    Parallel[A,B]:   y == cat(A(x), B(x))

Usage:
    pixi run mojo run -I . tests/nn/test_composition_invariants.mojo
"""

from std.math import abs
from std.memory import alloc, memset
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training import NetworkState
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.model import (
    Model,
    Sequential,
    Parallel,
    Linear,
    LinearReLU,
    ReLU,
    Residual,
    Repeat,
    SkipConcat,
    DualPath,
    SplitApply,
    FanOut,
)


def _align4(x: Int) -> Int:
    return (x + 3) & ~3


def _compare(
    name: String,
    a_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    tol: Float64 = 1e-6,
) raises:
    """Compare two float buffers element-wise."""
    var max_abs: Float64 = 0.0
    var max_rel: Float64 = 0.0
    var n_fail = 0
    for i in range(n):
        var va = Float64((a_ptr + i)[])
        var vb = Float64((b_ptr + i)[])
        var err = abs(va - vb)
        var denom = abs(va) + abs(vb)
        if err > max_abs:
            max_abs = err
        var rel: Float64 = 0.0
        if denom > 1e-8:
            rel = err / denom
        if rel > max_rel:
            max_rel = rel
        if err > tol and denom > 1e-7:
            n_fail += 1
    if n_fail == 0:
        print("  [PASS]", name, "max_abs=", max_abs, "max_rel=", max_rel)
    else:
        print(
            "  [FAIL]", name, n_fail, "/", n, "max_abs=", max_abs, "max_rel=", max_rel,
        )


def test_residual() raises:
    """Residual[A]: output == x + A(x)."""
    print("Residual[LinearReLU[8,8]]: y == x + Inner(x)")
    comptime Inner = LinearReLU[8, 8]
    comptime Res = Residual[Inner]
    comptime BS = 4
    comptime DIM = Inner.IN_DIM  # 8

    # Initialize composed model
    var state = NetworkState[Res, Adam[]]()
    state.initialize[Xavier[]]()

    # Input
    var input_ptr = alloc[Scalar[dtype]](BS * DIM)
    for i in range(BS * DIM):
        (input_ptr + i)[] = Scalar[dtype](0.1 + Float64(i % 11) / 11.0 * 0.8)
    var input_t = LayoutTensor[dtype, Layout.row_major(BS, DIM), MutAnyOrigin](input_ptr)

    # Forward through Residual
    var res_out = alloc[Scalar[dtype]](BS * DIM)
    var res_cache = alloc[Scalar[dtype]](BS * Res.CACHE_SIZE if Res.CACHE_SIZE > 0 else 1)
    var res_out_t = LayoutTensor[dtype, Layout.row_major(BS, DIM), MutAnyOrigin](res_out)
    var res_cache_t = LayoutTensor[dtype, Layout.row_major(BS, Res.CACHE_SIZE), MutAnyOrigin](res_cache)
    Res.forward[BS](input_t, res_out_t, state.params_view(), res_cache_t)

    # Forward through Inner alone (same params -- Residual.PARAM_SIZE == Inner.PARAM_SIZE)
    var inner_out = alloc[Scalar[dtype]](BS * DIM)
    var inner_cache = alloc[Scalar[dtype]](BS * Inner.CACHE_SIZE if Inner.CACHE_SIZE > 0 else 1)
    var inner_out_t = LayoutTensor[dtype, Layout.row_major(BS, DIM), MutAnyOrigin](inner_out)
    var inner_cache_t = LayoutTensor[dtype, Layout.row_major(BS, Inner.CACHE_SIZE), MutAnyOrigin](inner_cache)
    Inner.forward[BS](input_t, inner_out_t, state.params_view(), inner_cache_t)

    # Check: res_out == input + inner_out
    var expected = alloc[Scalar[dtype]](BS * DIM)
    for i in range(BS * DIM):
        (expected + i)[] = (input_ptr + i)[] + (inner_out + i)[]

    _compare("y == x + Inner(x)", res_out, expected, BS * DIM)

    input_ptr.free()
    res_out.free()
    res_cache.free()
    inner_out.free()
    inner_cache.free()
    expected.free()
    print()


def test_skip_concat() raises:
    """SkipConcat[A]: output == cat(x, A(x))."""
    print("SkipConcat[Linear[8,4]]: y == cat(x, Inner(x))")
    comptime Inner = Linear[8, 4]
    comptime SC = SkipConcat[Inner]
    comptime BS = 4
    comptime IN = Inner.IN_DIM  # 8
    comptime INNER_OUT = Inner.OUT_DIM  # 4
    comptime OUT = SC.OUT_DIM  # 12

    var state = NetworkState[SC, Adam[]]()
    state.initialize[Xavier[]]()

    var input_ptr = alloc[Scalar[dtype]](BS * IN)
    for i in range(BS * IN):
        (input_ptr + i)[] = Scalar[dtype](0.1 + Float64(i % 11) / 11.0 * 0.8)
    var input_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](input_ptr)

    # SkipConcat forward
    var sc_out = alloc[Scalar[dtype]](BS * OUT)
    var sc_cache = alloc[Scalar[dtype]](BS * SC.CACHE_SIZE if SC.CACHE_SIZE > 0 else 1)
    var sc_out_t = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](sc_out)
    var sc_cache_t = LayoutTensor[dtype, Layout.row_major(BS, SC.CACHE_SIZE), MutAnyOrigin](sc_cache)
    SC.forward[BS](input_t, sc_out_t, state.params_view(), sc_cache_t)

    # Inner forward (same params)
    var inner_out = alloc[Scalar[dtype]](BS * INNER_OUT)
    var inner_cache = alloc[Scalar[dtype]](BS * Inner.CACHE_SIZE if Inner.CACHE_SIZE > 0 else 1)
    var inner_out_t = LayoutTensor[dtype, Layout.row_major(BS, INNER_OUT), MutAnyOrigin](inner_out)
    var inner_cache_t = LayoutTensor[dtype, Layout.row_major(BS, Inner.CACHE_SIZE), MutAnyOrigin](inner_cache)
    Inner.forward[BS](input_t, inner_out_t, state.params_view(), inner_cache_t)

    # Check: sc_out == cat(x, inner_out) per row
    var expected = alloc[Scalar[dtype]](BS * OUT)
    for b in range(BS):
        for j in range(IN):
            (expected + b * OUT + j)[] = (input_ptr + b * IN + j)[]
        for j in range(INNER_OUT):
            (expected + b * OUT + IN + j)[] = (inner_out + b * INNER_OUT + j)[]

    _compare("y == cat(x, Inner(x))", sc_out, expected, BS * OUT)

    input_ptr.free()
    sc_out.free()
    sc_cache.free()
    inner_out.free()
    inner_cache.free()
    expected.free()
    print()


def test_repeat_1() raises:
    """Repeat[1, A]: output == A(x)."""
    print("Repeat[1, LinearReLU[8,8]]: y == Inner(x)")
    comptime Inner = LinearReLU[8, 8]
    comptime Rep = Repeat[1, Inner]
    comptime BS = 4
    comptime DIM = 8

    var state = NetworkState[Rep, Adam[]]()
    state.initialize[Xavier[]]()

    var input_ptr = alloc[Scalar[dtype]](BS * DIM)
    for i in range(BS * DIM):
        (input_ptr + i)[] = Scalar[dtype](0.1 + Float64(i % 11) / 11.0 * 0.8)
    var input_t = LayoutTensor[dtype, Layout.row_major(BS, DIM), MutAnyOrigin](input_ptr)

    # Repeat[1] forward
    var rep_out = alloc[Scalar[dtype]](BS * DIM)
    var rep_cache = alloc[Scalar[dtype]](BS * Rep.CACHE_SIZE if Rep.CACHE_SIZE > 0 else 1)
    var rep_out_t = LayoutTensor[dtype, Layout.row_major(BS, DIM), MutAnyOrigin](rep_out)
    var rep_cache_t = LayoutTensor[dtype, Layout.row_major(BS, Rep.CACHE_SIZE), MutAnyOrigin](rep_cache)
    Rep.forward[BS](input_t, rep_out_t, state.params_view(), rep_cache_t)

    # Inner forward (same params)
    var inner_out = alloc[Scalar[dtype]](BS * DIM)
    var inner_cache = alloc[Scalar[dtype]](BS * Inner.CACHE_SIZE if Inner.CACHE_SIZE > 0 else 1)
    var inner_out_t = LayoutTensor[dtype, Layout.row_major(BS, DIM), MutAnyOrigin](inner_out)
    var inner_cache_t = LayoutTensor[dtype, Layout.row_major(BS, Inner.CACHE_SIZE), MutAnyOrigin](inner_cache)
    Inner.forward[BS](input_t, inner_out_t, state.params_view(), inner_cache_t)

    _compare("y == Inner(x)", rep_out, inner_out, BS * DIM)

    input_ptr.free()
    rep_out.free()
    rep_cache.free()
    inner_out.free()
    inner_cache.free()
    print()


def test_repeat_2() raises:
    """Repeat[2, A, shared=True]: output == A(A(x))."""
    print("Repeat[2, LinearReLU[8,8]]: y == Inner(Inner(x))")
    comptime Inner = LinearReLU[8, 8]
    comptime Rep = Repeat[2, Inner]
    comptime BS = 4
    comptime DIM = 8

    var state = NetworkState[Rep, Adam[]]()
    state.initialize[Xavier[]]()

    var input_ptr = alloc[Scalar[dtype]](BS * DIM)
    for i in range(BS * DIM):
        (input_ptr + i)[] = Scalar[dtype](0.1 + Float64(i % 11) / 11.0 * 0.8)
    var input_t = LayoutTensor[dtype, Layout.row_major(BS, DIM), MutAnyOrigin](input_ptr)

    # Repeat[2] forward
    var rep_out = alloc[Scalar[dtype]](BS * DIM)
    var rep_cache = alloc[Scalar[dtype]](BS * Rep.CACHE_SIZE if Rep.CACHE_SIZE > 0 else 1)
    var rep_out_t = LayoutTensor[dtype, Layout.row_major(BS, DIM), MutAnyOrigin](rep_out)
    var rep_cache_t = LayoutTensor[dtype, Layout.row_major(BS, Rep.CACHE_SIZE), MutAnyOrigin](rep_cache)
    Rep.forward[BS](input_t, rep_out_t, state.params_view(), rep_cache_t)

    # Manual: inter = Inner(x), then output = Inner(inter) -- same params (shared)
    var inter = alloc[Scalar[dtype]](BS * DIM)
    var c1 = alloc[Scalar[dtype]](BS * Inner.CACHE_SIZE if Inner.CACHE_SIZE > 0 else 1)
    var inter_t = LayoutTensor[dtype, Layout.row_major(BS, DIM), MutAnyOrigin](inter)
    var c1_t = LayoutTensor[dtype, Layout.row_major(BS, Inner.CACHE_SIZE), MutAnyOrigin](c1)
    Inner.forward[BS](input_t, inter_t, state.params_view(), c1_t)

    var manual_out = alloc[Scalar[dtype]](BS * DIM)
    var c2 = alloc[Scalar[dtype]](BS * Inner.CACHE_SIZE if Inner.CACHE_SIZE > 0 else 1)
    var manual_out_t = LayoutTensor[dtype, Layout.row_major(BS, DIM), MutAnyOrigin](manual_out)
    var c2_t = LayoutTensor[dtype, Layout.row_major(BS, Inner.CACHE_SIZE), MutAnyOrigin](c2)
    Inner.forward[BS](inter_t, manual_out_t, state.params_view(), c2_t)

    _compare("y == Inner(Inner(x))", rep_out, manual_out, BS * DIM)

    input_ptr.free()
    rep_out.free()
    rep_cache.free()
    inter.free()
    c1.free()
    manual_out.free()
    c2.free()
    print()


def test_sequential() raises:
    """Sequential[A, B]: output == B(A(x))."""
    print("Sequential[LinearReLU[8,6], Linear[6,4]]: y == B(A(x))")
    comptime A = LinearReLU[8, 6]
    comptime B = Linear[6, 4]
    comptime Seq = Sequential[A, B]
    comptime BS = 4

    var state = NetworkState[Seq, Adam[]]()
    state.initialize[Xavier[]]()

    var input_ptr = alloc[Scalar[dtype]](BS * A.IN_DIM)
    for i in range(BS * A.IN_DIM):
        (input_ptr + i)[] = Scalar[dtype](0.1 + Float64(i % 11) / 11.0 * 0.8)
    var input_t = LayoutTensor[dtype, Layout.row_major(BS, A.IN_DIM), MutAnyOrigin](input_ptr)

    # Sequential forward
    var seq_out = alloc[Scalar[dtype]](BS * Seq.OUT_DIM)
    var seq_cache = alloc[Scalar[dtype]](BS * Seq.CACHE_SIZE if Seq.CACHE_SIZE > 0 else 1)
    var seq_out_t = LayoutTensor[dtype, Layout.row_major(BS, Seq.OUT_DIM), MutAnyOrigin](seq_out)
    var seq_cache_t = LayoutTensor[dtype, Layout.row_major(BS, Seq.CACHE_SIZE), MutAnyOrigin](seq_cache)
    Seq.forward[BS](input_t, seq_out_t, state.params_view(), seq_cache_t)

    # Extract sub-params: A at offset 0, B at offset align4(A.PS)
    comptime A_OFF = 0
    comptime B_OFF = _align4(A.PARAM_SIZE)
    var pa = LayoutTensor[dtype, Layout.row_major(A.PARAM_SIZE), MutAnyOrigin](state.params + A_OFF)
    var pb = LayoutTensor[dtype, Layout.row_major(B.PARAM_SIZE), MutAnyOrigin](state.params + B_OFF)

    # Manual: inter = A(x), out = B(inter)
    var inter = alloc[Scalar[dtype]](BS * A.OUT_DIM)
    var ca = alloc[Scalar[dtype]](BS * A.CACHE_SIZE if A.CACHE_SIZE > 0 else 1)
    var inter_t = LayoutTensor[dtype, Layout.row_major(BS, A.OUT_DIM), MutAnyOrigin](inter)
    var ca_t = LayoutTensor[dtype, Layout.row_major(BS, A.CACHE_SIZE), MutAnyOrigin](ca)
    A.forward[BS](input_t, inter_t, pa, ca_t)

    var manual_out = alloc[Scalar[dtype]](BS * B.OUT_DIM)
    var cb = alloc[Scalar[dtype]](BS * B.CACHE_SIZE if B.CACHE_SIZE > 0 else 1)
    var manual_out_t = LayoutTensor[dtype, Layout.row_major(BS, B.OUT_DIM), MutAnyOrigin](manual_out)
    var cb_t = LayoutTensor[dtype, Layout.row_major(BS, B.CACHE_SIZE), MutAnyOrigin](cb)
    B.forward[BS](inter_t, manual_out_t, pb, cb_t)

    _compare("y == B(A(x))", seq_out, manual_out, BS * Seq.OUT_DIM)

    input_ptr.free()
    seq_out.free()
    seq_cache.free()
    inter.free()
    ca.free()
    manual_out.free()
    cb.free()
    print()


def test_parallel() raises:
    """Parallel[A, B]: output == cat(A(x), B(x))."""
    print("Parallel[Linear[8,4], Linear[8,1]]: y == cat(A(x), B(x))")
    comptime A = Linear[8, 4]
    comptime B = Linear[8, 1]
    comptime Par = Parallel[A, B]
    comptime BS = 4
    comptime IN = A.IN_DIM  # 8

    var state = NetworkState[Par, Adam[]]()
    state.initialize[Xavier[]]()

    var input_ptr = alloc[Scalar[dtype]](BS * IN)
    for i in range(BS * IN):
        (input_ptr + i)[] = Scalar[dtype](0.1 + Float64(i % 11) / 11.0 * 0.8)
    var input_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](input_ptr)

    # Parallel forward
    var par_out = alloc[Scalar[dtype]](BS * Par.OUT_DIM)
    var par_cache = alloc[Scalar[dtype]](BS * Par.CACHE_SIZE if Par.CACHE_SIZE > 0 else 1)
    var par_out_t = LayoutTensor[dtype, Layout.row_major(BS, Par.OUT_DIM), MutAnyOrigin](par_out)
    var par_cache_t = LayoutTensor[dtype, Layout.row_major(BS, Par.CACHE_SIZE), MutAnyOrigin](par_cache)
    Par.forward[BS](input_t, par_out_t, state.params_view(), par_cache_t)

    # Extract sub-params: Parallel uses align4 accumulation
    # offset[0] = 0, offset[1] = align4(0 + A.PS) = align4(A.PS)
    comptime A_OFF = 0
    comptime B_OFF = _align4(A.PARAM_SIZE)
    var pa = LayoutTensor[dtype, Layout.row_major(A.PARAM_SIZE), MutAnyOrigin](state.params + A_OFF)
    var pb = LayoutTensor[dtype, Layout.row_major(B.PARAM_SIZE), MutAnyOrigin](state.params + B_OFF)

    # Manual: a_out = A(x), b_out = B(x)
    var a_out = alloc[Scalar[dtype]](BS * A.OUT_DIM)
    var a_cache = alloc[Scalar[dtype]](BS * A.CACHE_SIZE if A.CACHE_SIZE > 0 else 1)
    var a_out_t = LayoutTensor[dtype, Layout.row_major(BS, A.OUT_DIM), MutAnyOrigin](a_out)
    var a_cache_t = LayoutTensor[dtype, Layout.row_major(BS, A.CACHE_SIZE), MutAnyOrigin](a_cache)
    A.forward[BS](input_t, a_out_t, pa, a_cache_t)

    var b_out = alloc[Scalar[dtype]](BS * B.OUT_DIM)
    var b_cache = alloc[Scalar[dtype]](BS * B.CACHE_SIZE if B.CACHE_SIZE > 0 else 1)
    var b_out_t = LayoutTensor[dtype, Layout.row_major(BS, B.OUT_DIM), MutAnyOrigin](b_out)
    var b_cache_t = LayoutTensor[dtype, Layout.row_major(BS, B.CACHE_SIZE), MutAnyOrigin](b_cache)
    B.forward[BS](input_t, b_out_t, pb, b_cache_t)

    # Check: par_out[b, :] == cat(a_out[b, :], b_out[b, :])
    var expected = alloc[Scalar[dtype]](BS * Par.OUT_DIM)
    for b in range(BS):
        for j in range(A.OUT_DIM):
            (expected + b * Par.OUT_DIM + j)[] = (a_out + b * A.OUT_DIM + j)[]
        for j in range(B.OUT_DIM):
            (expected + b * Par.OUT_DIM + A.OUT_DIM + j)[] = (b_out + b * B.OUT_DIM + j)[]

    _compare("y == cat(A(x), B(x))", par_out, expected, BS * Par.OUT_DIM)

    input_ptr.free()
    par_out.free()
    par_cache.free()
    a_out.free()
    a_cache.free()
    b_out.free()
    b_cache.free()
    expected.free()
    print()


def test_split_apply() raises:
    """SplitApply[L, R, s]: output == cat(L(x[:s]), R(x[s:]))."""
    print("SplitApply[Linear[4,3], Linear[4,2], 4]: y == cat(L(x[:4]), R(x[4:]))")
    comptime L = Linear[4, 3]
    comptime R = Linear[4, 2]
    comptime SA = SplitApply[L, R, 4]
    comptime BS = 4
    comptime IN = SA.IN_DIM  # 8
    comptime SPLIT = 4

    var state = NetworkState[SA, Adam[]]()
    state.initialize[Xavier[]]()

    var input_ptr = alloc[Scalar[dtype]](BS * IN)
    for i in range(BS * IN):
        (input_ptr + i)[] = Scalar[dtype](0.1 + Float64(i % 11) / 11.0 * 0.8)
    var input_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](input_ptr)

    # SplitApply forward
    var sa_out = alloc[Scalar[dtype]](BS * SA.OUT_DIM)
    var sa_cache = alloc[Scalar[dtype]](BS * SA.CACHE_SIZE if SA.CACHE_SIZE > 0 else 1)
    var sa_out_t = LayoutTensor[dtype, Layout.row_major(BS, SA.OUT_DIM), MutAnyOrigin](sa_out)
    var sa_cache_t = LayoutTensor[dtype, Layout.row_major(BS, SA.CACHE_SIZE), MutAnyOrigin](sa_cache)
    SA.forward[BS](input_t, sa_out_t, state.params_view(), sa_cache_t)

    # Extract sub-params
    comptime L_OFF = 0
    comptime R_OFF = _align4(L.PARAM_SIZE)
    var pl = LayoutTensor[dtype, Layout.row_major(L.PARAM_SIZE), MutAnyOrigin](state.params + L_OFF)
    var pr = LayoutTensor[dtype, Layout.row_major(R.PARAM_SIZE), MutAnyOrigin](state.params + R_OFF)

    # Manual: split input, apply L and R separately
    var left_in = alloc[Scalar[dtype]](BS * SPLIT)
    var right_in = alloc[Scalar[dtype]](BS * (IN - SPLIT))
    for b in range(BS):
        for j in range(SPLIT):
            (left_in + b * SPLIT + j)[] = (input_ptr + b * IN + j)[]
        for j in range(IN - SPLIT):
            (right_in + b * (IN - SPLIT) + j)[] = (input_ptr + b * IN + SPLIT + j)[]

    var left_in_t = LayoutTensor[dtype, Layout.row_major(BS, SPLIT), MutAnyOrigin](left_in)
    var l_out = alloc[Scalar[dtype]](BS * L.OUT_DIM)
    var lc = alloc[Scalar[dtype]](BS * L.CACHE_SIZE if L.CACHE_SIZE > 0 else 1)
    var l_out_t = LayoutTensor[dtype, Layout.row_major(BS, L.OUT_DIM), MutAnyOrigin](l_out)
    var lc_t = LayoutTensor[dtype, Layout.row_major(BS, L.CACHE_SIZE), MutAnyOrigin](lc)
    L.forward[BS](left_in_t, l_out_t, pl, lc_t)

    var right_in_t = LayoutTensor[dtype, Layout.row_major(BS, IN - SPLIT), MutAnyOrigin](right_in)
    var r_out = alloc[Scalar[dtype]](BS * R.OUT_DIM)
    var rc = alloc[Scalar[dtype]](BS * R.CACHE_SIZE if R.CACHE_SIZE > 0 else 1)
    var r_out_t = LayoutTensor[dtype, Layout.row_major(BS, R.OUT_DIM), MutAnyOrigin](r_out)
    var rc_t = LayoutTensor[dtype, Layout.row_major(BS, R.CACHE_SIZE), MutAnyOrigin](rc)
    R.forward[BS](right_in_t, r_out_t, pr, rc_t)

    # Expected: cat(l_out, r_out) per row
    var expected = alloc[Scalar[dtype]](BS * SA.OUT_DIM)
    for b in range(BS):
        for j in range(L.OUT_DIM):
            (expected + b * SA.OUT_DIM + j)[] = (l_out + b * L.OUT_DIM + j)[]
        for j in range(R.OUT_DIM):
            (expected + b * SA.OUT_DIM + L.OUT_DIM + j)[] = (r_out + b * R.OUT_DIM + j)[]

    _compare("y == cat(L(x[:4]), R(x[4:]))", sa_out, expected, BS * SA.OUT_DIM)

    input_ptr.free()
    sa_out.free()
    sa_cache.free()
    left_in.free()
    right_in.free()
    l_out.free()
    lc.free()
    r_out.free()
    rc.free()
    expected.free()
    print()


def test_fanout() raises:
    """FanOut[A, 2]: output == cat(A_0(x), A_1(x)) with independent params."""
    print("FanOut[Linear[8,4], 2]: y == cat(copy0(x), copy1(x))")
    comptime Inner = Linear[8, 4]
    comptime FO = FanOut[Inner, 2]
    comptime BS = 4
    comptime IN = Inner.IN_DIM  # 8

    var state = NetworkState[FO, Adam[]]()
    state.initialize[Xavier[]]()

    var input_ptr = alloc[Scalar[dtype]](BS * IN)
    for i in range(BS * IN):
        (input_ptr + i)[] = Scalar[dtype](0.1 + Float64(i % 11) / 11.0 * 0.8)
    var input_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](input_ptr)

    # FanOut forward
    var fo_out = alloc[Scalar[dtype]](BS * FO.OUT_DIM)
    var fo_cache = alloc[Scalar[dtype]](BS * FO.CACHE_SIZE if FO.CACHE_SIZE > 0 else 1)
    var fo_out_t = LayoutTensor[dtype, Layout.row_major(BS, FO.OUT_DIM), MutAnyOrigin](fo_out)
    var fo_cache_t = LayoutTensor[dtype, Layout.row_major(BS, FO.CACHE_SIZE), MutAnyOrigin](fo_cache)
    FO.forward[BS](input_t, fo_out_t, state.params_view(), fo_cache_t)

    # Extract params for copy 0 and copy 1
    # FanOut: (N-1) * aligned(PS) + PS = align4(PS) + PS for N=2
    comptime P0_OFF = 0
    comptime P1_OFF = _align4(Inner.PARAM_SIZE)
    var p0 = LayoutTensor[dtype, Layout.row_major(Inner.PARAM_SIZE), MutAnyOrigin](state.params + P0_OFF)
    var p1 = LayoutTensor[dtype, Layout.row_major(Inner.PARAM_SIZE), MutAnyOrigin](state.params + P1_OFF)

    # Manual: run each copy
    var out0 = alloc[Scalar[dtype]](BS * Inner.OUT_DIM)
    var c0 = alloc[Scalar[dtype]](BS * Inner.CACHE_SIZE if Inner.CACHE_SIZE > 0 else 1)
    var out0_t = LayoutTensor[dtype, Layout.row_major(BS, Inner.OUT_DIM), MutAnyOrigin](out0)
    var c0_t = LayoutTensor[dtype, Layout.row_major(BS, Inner.CACHE_SIZE), MutAnyOrigin](c0)
    Inner.forward[BS](input_t, out0_t, p0, c0_t)

    var out1 = alloc[Scalar[dtype]](BS * Inner.OUT_DIM)
    var c1 = alloc[Scalar[dtype]](BS * Inner.CACHE_SIZE if Inner.CACHE_SIZE > 0 else 1)
    var out1_t = LayoutTensor[dtype, Layout.row_major(BS, Inner.OUT_DIM), MutAnyOrigin](out1)
    var c1_t = LayoutTensor[dtype, Layout.row_major(BS, Inner.CACHE_SIZE), MutAnyOrigin](c1)
    Inner.forward[BS](input_t, out1_t, p1, c1_t)

    var expected = alloc[Scalar[dtype]](BS * FO.OUT_DIM)
    for b in range(BS):
        for j in range(Inner.OUT_DIM):
            (expected + b * FO.OUT_DIM + j)[] = (out0 + b * Inner.OUT_DIM + j)[]
        for j in range(Inner.OUT_DIM):
            (expected + b * FO.OUT_DIM + Inner.OUT_DIM + j)[] = (out1 + b * Inner.OUT_DIM + j)[]

    _compare("y == cat(copy0(x), copy1(x))", fo_out, expected, BS * FO.OUT_DIM)

    input_ptr.free()
    fo_out.free()
    fo_cache.free()
    out0.free()
    c0.free()
    out1.free()
    c1.free()
    expected.free()
    print()


def test_dim_invariants() raises:
    """Verify compile-time dimension invariants for composed models."""
    print("Dimension invariants:")
    comptime A = LinearReLU[8, 6]
    comptime B = Linear[6, 4]

    # Sequential: IN == first.IN, OUT == last.OUT
    comptime Seq = Sequential[A, B]
    comptime assert Seq.IN_DIM == A.IN_DIM, "Sequential IN_DIM"
    comptime assert Seq.OUT_DIM == B.OUT_DIM, "Sequential OUT_DIM"
    print("  [PASS] Sequential: IN_DIM==A.IN, OUT_DIM==B.OUT")

    # Parallel: IN == branches.IN (all same), OUT == sum
    comptime P = Parallel[Linear[8, 4], Linear[8, 1]]
    comptime assert P.IN_DIM == 8, "Parallel IN_DIM"
    comptime assert P.OUT_DIM == 5, "Parallel OUT_DIM == 4 + 1"
    print("  [PASS] Parallel: IN_DIM==8, OUT_DIM==4+1=5")

    # Residual: IN == OUT == Inner.IN == Inner.OUT
    comptime R = Residual[LinearReLU[8, 8]]
    comptime assert R.IN_DIM == 8, "Residual IN"
    comptime assert R.OUT_DIM == 8, "Residual OUT"
    comptime assert R.PARAM_SIZE == LinearReLU[8, 8].PARAM_SIZE, "Residual PS"
    print("  [PASS] Residual: IN==OUT==8, PS==Inner.PS")

    # Repeat: IN == OUT, PS == Inner.PS (shared)
    comptime Rep = Repeat[3, LinearReLU[8, 8]]
    comptime assert Rep.IN_DIM == 8, "Repeat IN"
    comptime assert Rep.OUT_DIM == 8, "Repeat OUT"
    comptime assert Rep.PARAM_SIZE == LinearReLU[8, 8].PARAM_SIZE, "Repeat shared PS"
    comptime assert Rep.CACHE_SIZE == 3 * LinearReLU[8, 8].CACHE_SIZE, "Repeat cache"
    print("  [PASS] Repeat[3]: IN==OUT==8, PS==Inner.PS, CS==3*Inner.CS")

    # SkipConcat: OUT == IN + Inner.OUT
    comptime SC = SkipConcat[Linear[8, 4]]
    comptime assert SC.IN_DIM == 8, "SkipConcat IN"
    comptime assert SC.OUT_DIM == 12, "SkipConcat OUT == 8 + 4"
    print("  [PASS] SkipConcat: OUT==IN+Inner.OUT=12")

    # SplitApply: IN == L.IN + R.IN, OUT == L.OUT + R.OUT
    comptime SA = SplitApply[Linear[4, 3], Linear[4, 2], 4]
    comptime assert SA.IN_DIM == 8, "SplitApply IN"
    comptime assert SA.OUT_DIM == 5, "SplitApply OUT"
    print("  [PASS] SplitApply: IN==4+4=8, OUT==3+2=5")

    # FanOut: IN == Inner.IN, OUT == N * Inner.OUT
    comptime FO = FanOut[Linear[8, 4], 3]
    comptime assert FO.IN_DIM == 8, "FanOut IN"
    comptime assert FO.OUT_DIM == 12, "FanOut OUT == 3 * 4"
    print("  [PASS] FanOut[3]: IN==8, OUT==3*4=12")

    print()


def main() raises:
    print("=== Composition Invariant Tests ===")
    print()

    test_dim_invariants()
    test_residual()
    test_skip_concat()
    test_repeat_1()
    test_repeat_2()
    test_sequential()
    test_parallel()
    test_split_apply()
    test_fanout()

    print("=== Done ===")

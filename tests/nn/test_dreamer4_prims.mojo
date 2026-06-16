"""Dreamer 4 block primitives — CPU correctness (Phase 1).

SwiGLU: scalar reference for forward + finite-diff gradcheck.
SpaceTimeTranspose: exact permutation forward, self-inverse (fwd∘vjp),
and a known-grid spot check.
"""

from std.memory import alloc
from std.math import abs, exp
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.primitives.swiglu import SwiGLU
from mojo_rl.nn.primitives.space_time_transpose import SpaceTimeTranspose


comptime EPS: Float64 = 1e-3
comptime TOL: Float64 = 1e-2


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def _spread(i: Int, seed: Float64) -> Scalar[DT]:
    var x = seed + 0.7 * Float64(i)
    var t = x - 6.2831853 * Float64(Int(x / 6.2831853))
    return Scalar[DT](0.5 * (t - (t * t * t) / 6.0))


def test_swiglu_forward_reference() raises:
    print("test_swiglu_forward_reference ...")
    comptime HIDDEN = 4
    comptime BATCH = 2
    comptime IN_N = BATCH * 2 * HIDDEN
    comptime OUT_N = BATCH * HIDDEN
    var op = SwiGLU[HIDDEN].make[target="cpu", INIT=Zero]()
    var x = _alloc(IN_N)
    var y = _alloc(OUT_N)
    for i in range(IN_N):
        x[i] = _spread(i, 0.6)
    var xt = TileTensor(x, row_major[BATCH, 2 * HIDDEN]())
    var yt = TileTensor(y, row_major[BATCH, HIDDEN]())
    op.forward["cpu", BATCH](xt, output=yt)

    var max_err: Float64 = 0.0
    for b in range(BATCH):
        for k in range(HIDDEN):
            var u = Float64(x[b * 2 * HIDDEN + k])
            var v = Float64(x[b * 2 * HIDDEN + HIDDEN + k])
            var s = 1.0 / (1.0 + exp(-v))
            var refv = u * (v * s)
            var d = abs(Float64(y[b * HIDDEN + k]) - refv)
            if d > max_err:
                max_err = d
    print("   max fwd err =", max_err)
    assert_true(max_err < 1e-5, "swiglu forward reference")
    print("  ok")


def test_swiglu_gradcheck() raises:
    print("test_swiglu_gradcheck ...")
    comptime HIDDEN = 5
    comptime BATCH = 2
    comptime IN_N = BATCH * 2 * HIDDEN
    comptime OUT_N = BATCH * HIDDEN
    var op = SwiGLU[HIDDEN].make[target="cpu", INIT=Zero]()
    var x = _alloc(IN_N)
    var y = _alloc(OUT_N)
    var go = _alloc(OUT_N)
    var gi = _alloc(IN_N)
    for i in range(IN_N):
        x[i] = _spread(i, 1.1)
    for i in range(OUT_N):
        go[i] = _spread(i, 3.3)
    var xt = TileTensor(x, row_major[BATCH, 2 * HIDDEN]())
    var yt = TileTensor(y, row_major[BATCH, HIDDEN]())
    op.forward["cpu", BATCH](xt, output=yt)
    var got = TileTensor(go, row_major[BATCH, HIDDEN]())
    var git = TileTensor(gi, row_major[BATCH, 2 * HIDDEN]())
    op.vjp["cpu", BATCH](got, git)

    var max_err: Float64 = 0.0
    for kk in range(IN_N):
        var orig = x[kk]
        x[kk] = orig + Scalar[DT](EPS)
        op.forward["cpu", BATCH](xt, output=yt)
        var lp: Float64 = 0.0
        for i in range(OUT_N):
            lp += Float64(y[i]) * Float64(go[i])
        x[kk] = orig - Scalar[DT](EPS)
        op.forward["cpu", BATCH](xt, output=yt)
        var lm: Float64 = 0.0
        for i in range(OUT_N):
            lm += Float64(y[i]) * Float64(go[i])
        x[kk] = orig
        var fd = (lp - lm) / (2.0 * EPS)
        var d = abs(Float64(gi[kk]) - fd)
        if d > max_err:
            max_err = d
    print("   max|analytic - FD| =", max_err)
    assert_true(max_err < TOL, "swiglu gradcheck")
    print("  ok")


def test_stt_permutation_and_self_inverse() raises:
    print("test_stt_permutation_and_self_inverse ...")
    comptime T = 3
    comptime S = 4
    comptime D = 2
    comptime BATCH = 2
    comptime N = BATCH * T * S * D
    var op = SpaceTimeTranspose[T, S, D].make[target="cpu", INIT=Zero]()
    var x = _alloc(N)
    var y = _alloc(N)
    for i in range(N):
        x[i] = _spread(i, 2.0)
    var xt = TileTensor(x, row_major[BATCH, T * S * D]())
    var yt = TileTensor(y, row_major[BATCH, T * S * D]())
    op.forward["cpu", BATCH](xt, output=yt)

    # exact permutation: out[(s,t)] == in[(t,s)]
    var max_err: Float64 = 0.0
    for b in range(BATCH):
        for t in range(T):
            for s in range(S):
                for d in range(D):
                    var got = Float64(y[b * T * S * D + (s * T + t) * D + d])
                    var refv = Float64(x[b * T * S * D + (t * S + s) * D + d])
                    var e = abs(got - refv)
                    if e > max_err:
                        max_err = e
    assert_true(max_err < 1e-7, "stt forward permutation")

    # self-inverse: vjp(forward(x)) == x  (grad_output := forward output)
    var gi = _alloc(N)
    var git = TileTensor(gi, row_major[BATCH, T * S * D]())
    op.vjp["cpu", BATCH](yt, git)
    var max_round: Float64 = 0.0
    for i in range(N):
        var e = abs(Float64(gi[i]) - Float64(x[i]))
        if e > max_round:
            max_round = e
    print("   perm err =", max_err, "  roundtrip err =", max_round)
    assert_true(max_round < 1e-7, "stt self-inverse roundtrip")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Dreamer 4 block primitives — CPU (Phase 1)")
    print("=" * 70)
    test_swiglu_forward_reference()
    test_swiglu_gradcheck()
    test_stt_permutation_and_self_inverse()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

"""Dreamer 4 block-causal transformer stack — CPU end-to-end (Phase 1).

Builds Dreamer4Stack (depth>1, time attention every block) as a pure
combinator, runs forward + vjp at BATCH = B*T, and finite-diff gradchecks the
input — exercising space attention (modality-masked), time attention (causal
over T, latents only), SwiGLU FFN, and the residual wiring end-to-end.
"""

from std.memory import alloc
from std.math import abs
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.deep_agents2.dreamer4.blocks import Dreamer4Stack


comptime EPS: Float64 = 1e-3
comptime TOL: Float64 = 3e-2


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def _spread(i: Int, seed: Float64) -> Scalar[DT]:
    var x = seed + 0.7 * Float64(i)
    var t = x - 6.2831853 * Float64(Int(x / 6.2831853))
    return Scalar[DT](0.5 * (t - (t * t * t) / 6.0))


def main() raises:
    print("=" * 70)
    print("Dreamer4Stack — CPU end-to-end (Phase 1)")
    print("=" * 70)
    comptime D = 4
    comptime NH = 2
    comptime T = 3
    comptime S = 5
    comptime L = 2
    comptime HID = 8
    comptime DEPTH = 2
    comptime B = 2
    comptime BATCH = B * T
    comptime N = BATCH * S * D

    var stack = Dreamer4Stack[D, NH, T, S, L, HID, DEPTH, "encoder"].make[
        target="cpu", INIT=Xavier
    ]()

    var x = _alloc(N)
    var y = _alloc(N)
    var go = _alloc(N)
    var gi = _alloc(N)
    for i in range(N):
        x[i] = _spread(i, 0.8)
    for i in range(N):
        go[i] = _spread(i, 3.7)
    var xt = TileTensor(x, row_major[BATCH, S * D]())
    var yt = TileTensor(y, row_major[BATCH, S * D]())
    stack.forward["cpu", BATCH](xt, output=yt)

    # output must be finite (sanity).
    var any_nonzero = False
    for i in range(N):
        if Float64(y[i]) != 0.0:
            any_nonzero = True
    assert_true(any_nonzero, "stack output is all-zero (init/forward broken)")

    var got = TileTensor(go, row_major[BATCH, S * D]())
    var git = TileTensor(gi, row_major[BATCH, S * D]())
    stack.vjp["cpu", BATCH](got, git)

    print("gradcheck (whole stack) ...")
    var max_err: Float64 = 0.0
    for k in range(N):
        var orig = x[k]
        x[k] = orig + Scalar[DT](EPS)
        stack.forward["cpu", BATCH](xt, output=yt)
        var lp: Float64 = 0.0
        for i in range(N):
            lp += Float64(y[i]) * Float64(go[i])
        x[k] = orig - Scalar[DT](EPS)
        stack.forward["cpu", BATCH](xt, output=yt)
        var lm: Float64 = 0.0
        for i in range(N):
            lm += Float64(y[i]) * Float64(go[i])
        x[k] = orig
        var fd = (lp - lm) / (2.0 * EPS)
        var d = abs(Float64(gi[k]) - fd)
        if d > max_err:
            max_err = d
    print("   max|analytic - FD| =", max_err)
    assert_true(max_err < TOL, "Dreamer4Stack gradcheck")

    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

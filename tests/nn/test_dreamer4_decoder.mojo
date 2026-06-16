"""Dreamer4Decoder — CPU end-to-end (Phase 1).

Builds the tokenizer decoder (up_proj → append queries → +pos → decoder
transformer → slice patches → head → sigmoid) and checks: output in [0,1]
(sigmoid), output not all-equal (sanity), and an input gradcheck through the
whole decoder.
"""

from std.memory import alloc
from std.math import abs
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Xavier
from mojo_rl.deep_agents.dreamer4.blocks import Dreamer4Decoder


comptime EPS: Float64 = 1e-3
comptime TOL: Float64 = 4e-2


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def _spread(i: Int, seed: Float64) -> Scalar[DT]:
    var x = seed + 0.7 * Float64(i)
    var t = x - 6.2831853 * Float64(Int(x / 6.2831853))
    return Scalar[DT](0.5 * (t - (t * t * t) / 6.0))


def main() raises:
    print("=" * 70)
    print("Dreamer4Decoder — CPU end-to-end (Phase 1)")
    print("=" * 70)
    comptime D_BOT = 3
    comptime D = 4
    comptime NH = 2
    comptime T = 3
    comptime L = 2
    comptime NP = 4          # patches per frame
    comptime DP = 5          # patch_size^2 * C
    comptime HID = 8
    comptime DEPTH = 2
    comptime B = 2
    comptime BATCH = B * T
    comptime IN_N = BATCH * L * D_BOT
    comptime OUT_N = BATCH * NP * DP

    var dec = Dreamer4Decoder[D_BOT, D, NH, T, L, NP, DP, HID, DEPTH].make[
        target="cpu", INIT=Xavier
    ]()

    var x = _alloc(IN_N)
    var y = _alloc(OUT_N)
    var go = _alloc(OUT_N)
    var gi = _alloc(IN_N)
    for i in range(IN_N):
        x[i] = _spread(i, 0.5)
    for i in range(OUT_N):
        go[i] = _spread(i, 2.9)
    var xt = TileTensor(x, row_major[BATCH, L * D_BOT]())
    var yt = TileTensor(y, row_major[BATCH, NP * DP]())
    dec.forward["cpu", BATCH](xt, output=yt)

    # sigmoid output range + non-degenerate.
    print("checking output range [0,1] ...")
    var lo: Float64 = 1e30
    var hi: Float64 = -1e30
    for i in range(OUT_N):
        var v = Float64(y[i])
        if v < lo:
            lo = v
        if v > hi:
            hi = v
    print("   min =", lo, "  max =", hi)
    assert_true(lo >= 0.0 and hi <= 1.0, "decoder output must be in [0,1]")
    assert_true(hi - lo > 1e-4, "decoder output is degenerate (all equal)")

    var got = TileTensor(go, row_major[BATCH, NP * DP]())
    var git = TileTensor(gi, row_major[BATCH, L * D_BOT]())
    dec.vjp["cpu", BATCH](got, git)

    print("gradcheck (whole decoder) ...")
    var max_err: Float64 = 0.0
    for k in range(IN_N):
        var orig = x[k]
        x[k] = orig + Scalar[DT](EPS)
        dec.forward["cpu", BATCH](xt, output=yt)
        var lp: Float64 = 0.0
        for i in range(OUT_N):
            lp += Float64(y[i]) * Float64(go[i])
        x[k] = orig - Scalar[DT](EPS)
        dec.forward["cpu", BATCH](xt, output=yt)
        var lm: Float64 = 0.0
        for i in range(OUT_N):
            lm += Float64(y[i]) * Float64(go[i])
        x[k] = orig
        var fd = (lp - lm) / (2.0 * EPS)
        var d = abs(Float64(gi[k]) - fd)
        if d > max_err:
            max_err = d
    print("   max|analytic - FD| =", max_err)
    assert_true(max_err < TOL, "Dreamer4Decoder gradcheck")

    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

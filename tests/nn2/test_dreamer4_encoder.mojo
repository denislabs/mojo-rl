"""Dreamer4Encoder — CPU end-to-end (Phase 1).

patch tokens (NP·DP) → z (L·D_BOT). Checks: z in [-1,1] (tanh bottleneck),
MAE actually drops some patches (mask has both kept and dropped), and input
+ mask_token gradchecks (RNG frozen — advance_rng not called — so the mask is
identical across the finite-diff forwards).
"""

from std.memory import alloc
from std.math import abs
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.deep_agents2.dreamer4.encoder import Dreamer4Encoder


comptime EPS: Float64 = 1e-3
comptime TOL: Float64 = 5e-2


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def _spread(i: Int, seed: Float64) -> Scalar[DT]:
    var x = seed + 0.7 * Float64(i)
    var t = x - 6.2831853 * Float64(Int(x / 6.2831853))
    return Scalar[DT](0.5 * (t - (t * t * t) / 6.0))


def main() raises:
    print("=" * 70)
    print("Dreamer4Encoder — CPU end-to-end (Phase 1)")
    print("=" * 70)
    comptime DP = 5
    comptime D = 4
    comptime NH = 2
    comptime T = 3
    comptime L = 2
    comptime NP = 4
    comptime D_BOT = 3
    comptime HID = 8
    comptime DEPTH = 2
    comptime B = 2
    comptime BATCH = B * T
    comptime IN_N = BATCH * NP * DP
    comptime OUT_N = BATCH * L * D_BOT

    var enc = Dreamer4Encoder[
        DP, D, NH, T, L, NP, D_BOT, HID, DEPTH, 0.3, 0.6, 12345
    ].make[target="cpu", INIT=Xavier]()

    var x = _alloc(IN_N)
    var y = _alloc(OUT_N)
    var go = _alloc(OUT_N)
    var gi = _alloc(IN_N)
    for i in range(IN_N):
        x[i] = _spread(i, 0.4)
    for i in range(OUT_N):
        go[i] = _spread(i, 3.1)
    var xt = TileTensor(x, row_major[BATCH, NP * DP]())
    var yt = TileTensor(y, row_major[BATCH, L * D_BOT]())
    enc.forward["cpu", BATCH](xt, output=yt)

    # bottleneck tanh range.
    print("checking z range [-1,1] + MAE mask ...")
    var lo: Float64 = 1e30
    var hi: Float64 = -1e30
    for i in range(OUT_N):
        var v = Float64(y[i])
        if v < lo:
            lo = v
        if v > hi:
            hi = v
    assert_true(lo >= -1.0 and hi <= 1.0, "z must be in [-1,1]")

    # MAE: mask must have both kept (1) and dropped (0).
    var mp = enc.mae_mask_ptr()
    var n_kept = 0
    var n_drop = 0
    for i in range(BATCH * NP):
        if Float64(mp[i]) > 0.5:
            n_kept += 1
        else:
            n_drop += 1
    print("   z in [", lo, ",", hi, "]  kept =", n_kept, " dropped =", n_drop)
    assert_true(n_kept > 0 and n_drop > 0, "MAE must drop some and keep some")

    enc.zero_grad["cpu"]()
    var got = TileTensor(go, row_major[BATCH, L * D_BOT]())
    var git = TileTensor(gi, row_major[BATCH, NP * DP]())
    enc.vjp["cpu", BATCH](got, git)

    print("input gradcheck ...")
    var max_in: Float64 = 0.0
    for k in range(IN_N):
        var orig = x[k]
        x[k] = orig + Scalar[DT](EPS)
        enc.forward["cpu", BATCH](xt, output=yt)
        var lp: Float64 = 0.0
        for i in range(OUT_N):
            lp += Float64(y[i]) * Float64(go[i])
        x[k] = orig - Scalar[DT](EPS)
        enc.forward["cpu", BATCH](xt, output=yt)
        var lm: Float64 = 0.0
        for i in range(OUT_N):
            lm += Float64(y[i]) * Float64(go[i])
        x[k] = orig
        var fd = (lp - lm) / (2.0 * EPS)
        var d = abs(Float64(gi[k]) - fd)
        if d > max_in:
            max_in = d
    print("   max input |analytic - FD| =", max_in)
    assert_true(max_in < TOL, "encoder input gradcheck")

    # mask_token param gradcheck.
    print("mask_token gradcheck ...")
    var tp = enc.mae.mask_token.value_unsafe_ptr_cpu()
    var gtok = enc.mae.mask_token.grad.unsafe_ptr()
    var max_p: Float64 = 0.0
    for k in range(D):
        var orig = tp[k]
        tp[k] = orig + Scalar[DT](EPS)
        enc.forward["cpu", BATCH](xt, output=yt)
        var lp: Float64 = 0.0
        for i in range(OUT_N):
            lp += Float64(y[i]) * Float64(go[i])
        tp[k] = orig - Scalar[DT](EPS)
        enc.forward["cpu", BATCH](xt, output=yt)
        var lm: Float64 = 0.0
        for i in range(OUT_N):
            lm += Float64(y[i]) * Float64(go[i])
        tp[k] = orig
        var fd = (lp - lm) / (2.0 * EPS)
        var d = abs(Float64(gtok[k]) - fd)
        if d > max_p:
            max_p = d
    print("   max mask_token |analytic - FD| =", max_p)
    assert_true(max_p < TOL, "encoder mask_token gradcheck")

    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

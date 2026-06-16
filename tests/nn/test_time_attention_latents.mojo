"""TimeAttentionLatents — CPU forward/vjp (Phase 1).

Finite-diff gradcheck on the leaf input (validates the gather → causal MHA
over T → scatter path end-to-end), plus the structural invariants: non-latent
outputs are exactly 0 (so the enclosing Residual leaves them unchanged) and
non-latent input grads are exactly 0.
"""

from std.memory import alloc
from std.math import abs
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.primitives.time_attention_latents import TimeAttentionLatents


comptime EPS: Float64 = 1e-3
comptime TOL: Float64 = 2e-2


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def _spread(i: Int, seed: Float64) -> Scalar[DT]:
    var x = seed + 0.7 * Float64(i)
    var t = x - 6.2831853 * Float64(Int(x / 6.2831853))
    return Scalar[DT](0.5 * (t - (t * t * t) / 6.0))


def main() raises:
    print("=" * 70)
    print("TimeAttentionLatents — CPU (Phase 1)")
    print("=" * 70)
    comptime D = 4
    comptime NH = 2
    comptime T = 3
    comptime S = 5
    comptime L = 2
    comptime B = 2
    comptime BATCH = B * T
    comptime IN_N = BATCH * S * D
    comptime OUT_N = BATCH * S * D

    var op = TimeAttentionLatents[D, NH, T, S, L].make[
        target="cpu", INIT=Xavier
    ]()
    var x = _alloc(IN_N)
    var y = _alloc(OUT_N)
    var go = _alloc(OUT_N)
    var gi = _alloc(IN_N)
    for i in range(IN_N):
        x[i] = _spread(i, 1.3)
    for i in range(OUT_N):
        go[i] = _spread(i, 4.1)
    var xt = TileTensor(x, row_major[BATCH, S * D]())
    var yt = TileTensor(y, row_major[BATCH, S * D]())
    op.forward["cpu", BATCH](xt, output=yt)
    var got = TileTensor(go, row_major[BATCH, S * D]())
    var git = TileTensor(gi, row_major[BATCH, S * D]())
    op.vjp["cpu", BATCH](got, git)

    # invariant 1: non-latent outputs are exactly 0.
    print("invariant: non-latent outputs == 0 ...")
    var max_nl_out: Float64 = 0.0
    for b in range(B):
        for t in range(T):
            for s in range(S):
                if s >= L:
                    for d in range(D):
                        var v = abs(Float64(y[(b * T + t) * S * D + s * D + d]))
                        if v > max_nl_out:
                            max_nl_out = v
    print("   max =", max_nl_out)
    assert_true(max_nl_out == 0.0, "non-latent outputs must be 0")

    # invariant 2: non-latent input grads are exactly 0.
    print("invariant: non-latent input grads == 0 ...")
    var max_nl_g: Float64 = 0.0
    for b in range(B):
        for t in range(T):
            for s in range(S):
                if s >= L:
                    for d in range(D):
                        var v = abs(Float64(gi[(b * T + t) * S * D + s * D + d]))
                        if v > max_nl_g:
                            max_nl_g = v
    print("   max =", max_nl_g)
    assert_true(max_nl_g == 0.0, "non-latent input grads must be 0")

    # finite-diff gradcheck on the input.
    print("gradcheck ...")
    var max_err: Float64 = 0.0
    for k in range(IN_N):
        var orig = x[k]
        x[k] = orig + Scalar[DT](EPS)
        op.forward["cpu", BATCH](xt, output=yt)
        var lp: Float64 = 0.0
        for i in range(OUT_N):
            lp += Float64(y[i]) * Float64(go[i])
        x[k] = orig - Scalar[DT](EPS)
        op.forward["cpu", BATCH](xt, output=yt)
        var lm: Float64 = 0.0
        for i in range(OUT_N):
            lm += Float64(y[i]) * Float64(go[i])
        x[k] = orig
        var fd = (lp - lm) / (2.0 * EPS)
        var d = abs(Float64(gi[k]) - fd)
        if d > max_err:
            max_err = d
    print("   max|analytic - FD| =", max_err)
    assert_true(max_err < TOL, "time-attention gradcheck")

    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

"""The two reductions §12.15 got wrong, gated where they can be SEEN.

`test_fb_trainer_gpu_parity.mojo` cannot see either of these, by construction:
it initialises with `Deterministic`, so `F1 == F2`, so `mt1 == mt2`, so
min and mean coincide and every pessimism penalty is a no-op. Its numbers were
bit-identical across the fix — vacuously, not reassuringly. These gates make
the twins DIFFER and the signs MIX, which is the only regime where either bug
is visible.

  [1] `pessimism_blend_t` is BFM-Zero's `get_targets_uncertainty` at P = 2:
      penalty 0.0 must be exactly the mean, 0.5 exactly the entrywise min, and
      the two must DIFFER wherever the twins disagree. The identity
      "0.5 IS the twin-min" is what the whole FB-target argument rests on, so
      it is pinned rather than asserted in a comment.

  [2] `mean_abs_into_t` is `Q.abs().mean()`, NOT `|Q.mean()|`. The gate uses a
      sign-balanced input where the signed mean is ~0 and the mean absolute
      value is large — the regime that switched the CPR style term off.

Run:
    pixi run mojo run -I . tests/deep_agents/test_fb_pessimism_and_scale.mojo
"""

from std.math import abs
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.deep_agents.fb.kernels import (
    pessimism_blend_t, pessimism_row_weights_t, mean_abs_into_t, mean_into_t,
)

comptime N: Int = 64
# `DT` is float32: the MEAN costs one rounding, so it gets a float32 band.
# The MIN gets none — the reformulated blend returns `lo` bit-exactly at
# p = 0.5, and anything else there means the cancellation is back.
comptime MEAN_TOL: Float64 = 1e-5
comptime MIN_TOL: Float64 = 0.0


def test_pessimism_blend() raises:
    print("[1] pessimism_blend: 0.0 is the mean, 0.5 is the twin-min ...")
    var m1 = Tensor.alloc(N)
    var m2 = Tensor.alloc(N)
    for i in range(N):
        # deliberately straddling: m1 above m2 on half the entries, below on
        # the other half, so a min/mean mix-up cannot hide behind an ordering
        m1.data[i] = Scalar[DT](0.5 * Float64(i) - 8.0)
        m2.data[i] = Scalar[DT](-0.3 * Float64(i) + 5.0)

    var mean_out = Tensor.alloc(N)
    var min_out = Tensor.alloc(N)
    pessimism_blend_t["cpu", N](mean_out, m1, m2, Scalar[DT](1.0), Scalar[DT](0.0), None)
    pessimism_blend_t["cpu", N](min_out, m1, m2, Scalar[DT](1.0), Scalar[DT](0.5), None)

    var worst_mean = Float64(0)
    var worst_min = Float64(0)
    var n_differ = 0
    for i in range(N):
        var a = Float64(m1.data[i])
        var b = Float64(m2.data[i])
        var want_mean = 0.5 * (a + b)
        var want_min = a if a < b else b
        var e1 = abs(Float64(mean_out.data[i]) - want_mean)
        var e2 = abs(Float64(min_out.data[i]) - want_min)
        if e1 > worst_mean:
            worst_mean = e1
        if e2 > worst_min:
            worst_min = e2
        if abs(Float64(mean_out.data[i]) - Float64(min_out.data[i])) > 1e-9:
            n_differ += 1
    print("      worst |blend(0.0) - mean| =", worst_mean)
    print("      worst |blend(0.5) - min|  =", worst_min)
    print("      entries where 0.0 and 0.5 differ:", n_differ, "/", N)
    assert_true(
        worst_mean < MEAN_TOL,
        "penalty 0.0 is not the mean: " + String(worst_mean),
    )
    assert_true(
        worst_min <= MIN_TOL,
        "penalty 0.5 is not BIT-EXACTLY the min (" + String(worst_min)
        + ") — the blend has gone back to a cancelling form",
    )
    # THE ANTI-VACUITY CLAUSE. If the twins agreed everywhere the two checks
    # above would both pass on a single wrong implementation.
    assert_true(
        n_differ > N // 2,
        "the probe's twins barely disagree (" + String(n_differ) + " of "
        + String(N) + ") — min and mean nearly coincide here, so this gate"
        " would pass on either one",
    )


def test_gamma_multiplies_the_blend() raises:
    print("[2] gamma scales the blended target ...")
    var m1 = Tensor.alloc(N)
    var m2 = Tensor.alloc(N)
    for i in range(N):
        m1.data[i] = Scalar[DT](1.0 + 0.1 * Float64(i))
        m2.data[i] = Scalar[DT](2.0 - 0.05 * Float64(i))
    var g1 = Tensor.alloc(N)
    var gg = Tensor.alloc(N)
    pessimism_blend_t["cpu", N](g1, m1, m2, Scalar[DT](1.0), Scalar[DT](0.0), None)
    pessimism_blend_t["cpu", N](gg, m1, m2, Scalar[DT](0.98), Scalar[DT](0.0), None)
    var worst = Float64(0)
    for i in range(N):
        var e = abs(Float64(gg.data[i]) - 0.98 * Float64(g1.data[i]))
        if e > worst:
            worst = e
    print("      worst |blend(gamma) - gamma*blend(1)| =", worst)
    # float32 on values of order 10: one rounding is ~1e-6 absolute here, so
    # this band is the dtype, not a tuned number.
    assert_true(
        worst < MEAN_TOL, "gamma is not a clean scale: " + String(worst)
    )


def test_mean_abs_is_not_abs_mean() raises:
    print("[3] mean(|x|) is not |mean(x)| ...")
    var x = Tensor.alloc(N)
    # sign-balanced: the signed mean cancels to ~0, the absolute mean does not
    for i in range(N):
        var v = 3.0 + 0.01 * Float64(i)
        x.data[i] = Scalar[DT](v if i % 2 == 0 else -v)

    var acc_signed = Tensor.alloc(1)
    var acc_abs = Tensor.alloc(1)
    mean_into_t["cpu", N](x, acc_signed, None)
    mean_abs_into_t["cpu", N](x, acc_abs, None)
    var signed = abs(Float64(acc_signed.data[0]))
    var absmean = Float64(acc_abs.data[0])
    print("      |mean(x)| =", signed, "   mean(|x|) =", absmean)

    var want = Float64(0)
    for i in range(N):
        want += abs(Float64(x.data[i]))
    want /= Float64(N)
    assert_true(
        abs(absmean - want) < MEAN_TOL,
        "mean_abs_into_t is wrong: " + String(absmean) + " want " + String(want),
    )
    # the whole point: on this input the two differ by orders of magnitude, so
    # weighting by the wrong one is not a rounding difference
    assert_true(
        absmean > 100.0 * signed,
        "the probe is not sign-balanced enough to discriminate (|mean| "
        + String(signed) + " vs mean|.| " + String(absmean) + ")",
    )
    # and Jensen's direction must hold on ANY input
    assert_true(
        absmean >= signed - MEAN_TOL, "mean(|x|) < |mean(x)| is impossible"
    )
    print("      ratio mean(|x|)/|mean(x)| =", absmean / signed)


def test_weights_are_the_blends_derivative() raises:
    """The actor's Q_fb VALUE and its GRADIENT must come from one rule.

    `pessimism_blend_t` reduces the twin to a value; `pessimism_row_weights_t`
    says how much of the gradient each twin gets. They are written as separate
    kernels, so nothing but this stops them drifting apart — and a value that
    is `min` paired with a gradient that is the mean would train the actor
    against an objective it is not being scored on, silently.

    The identity: `blend(a, b, 1, p) == w1*a + w2*b`, exactly, for every p.
    """
    print("[4] the row weights ARE the blend's derivative ...")
    var a = Tensor.alloc(N)
    var b = Tensor.alloc(N)
    for i in range(N):
        # straddling, so `min` picks each side about half the time
        a.data[i] = Scalar[DT](0.5 * Float64(i) - 8.0)
        b.data[i] = Scalar[DT](-0.3 * Float64(i) + 5.0)
    var w1 = Tensor.alloc(N)
    var w2 = Tensor.alloc(N)
    var blended = Tensor.alloc(N)
    for pi in range(3):
        var p = 0.0 if pi == 0 else (0.25 if pi == 1 else 0.5)
        pessimism_blend_t["cpu", N](blended, a, b, Scalar[DT](1.0), Scalar[DT](p), None)
        pessimism_row_weights_t["cpu", N](w1, w2, a, b, Scalar[DT](p), None)
        var worst = Float64(0)
        var n_min = 0
        for i in range(N):
            var recon = (
                Float64(w1.data[i]) * Float64(a.data[i])
                + Float64(w2.data[i]) * Float64(b.data[i])
            )
            var e = abs(recon - Float64(blended.data[i]))
            if e > worst:
                worst = e
            if Float64(w1.data[i]) > 0.99:
                n_min += 1
        print("      p =", p, "  worst |w.x - blend| =", worst,
              "  rows where twin1 took it all:", n_min)
        assert_true(
            worst < MEAN_TOL,
            "at p=" + String(p) + " the weights do not reconstruct the blend ("
            + String(worst) + ") — value and gradient have drifted apart",
        )
    # at p = 0.5 the split must be hard: all-or-nothing, never shared
    pessimism_row_weights_t["cpu", N](w1, w2, a, b, Scalar[DT](0.5), None)
    for i in range(N):
        var x = Float64(w1.data[i])
        assert_true(
            abs(x) < 1e-12 or abs(x - 1.0) < 1e-12,
            "at penalty 0.5 each row's gradient must go ENTIRELY to the min"
            " twin; got weight " + String(x),
        )


def main() raises:
    print("=== FB pessimism + scale_reg reductions ===")
    test_pessimism_blend()
    test_gamma_multiplies_the_blend()
    test_mean_abs_is_not_abs_mean()
    test_weights_are_the_blends_derivative()
    print("=== all passed ===")

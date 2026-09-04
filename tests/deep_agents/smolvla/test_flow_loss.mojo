# +--------------------------------------------------------------------------+ #
# | The flow-matching objective: the time law, the interpolant, the loss
# +--------------------------------------------------------------------------+ #
"""Three things that are shape-identical to their wrong versions.

    pixi run -e apple mojo run -I . \\
        tests/deep_agents/smolvla/test_flow_loss.mojo

Every defect this file guards against produces a training run that starts,
converges, and ends up somewhere else:

  1. **A uniform `t` instead of `Beta(1.5, 1)`.** Both are floats in (0, 1).
     Beta(1.5, 1) has mean 0.6 and concentrates near the noisy end, which is
     where the velocity is hard to predict; uniform spends a third of its
     samples where the problem is nearly trivial.
  2. **`u_t = actions - noise`**, or `x_t = (1-t)*noise + t*actions`. Same
     shapes, same magnitudes, a model that drives the chunk the wrong way.
  3. **The loss covering all 32 padded action columns.** 26 of them are
     `u_t = noise - 0 = noise`: unpredictable by construction. The network
     would spend a large share of its gradient budget failing to fit noise,
     and the loss curve would look plausible the whole time.

Each is gated by a property that only the right version has, and leg [1]
additionally REJECTS the uniform sampler rather than merely accepting the beta
one — an empirical CDF that nothing fails is not evidence.

⚠ Leg [1] is statistical. 200,000 draws puts the binomial standard error at
about 0.0011 per decile, and the band is 0.006 — a bit over 5 sigma, so a
false failure is not something this suite will see. MEASURED, the worst
decile gap is 1.2e-03 for the real sampler and 1.5e-01 for a uniform one —
the rejection clears the band by 25x, which is the point: these two
distributions are not close, and nothing here rests on a fine margin.
"""

from std.math import abs, exp, log, sqrt
from std.random import random_float64
from std.testing import assert_true, assert_equal
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.deep_agents.smolvla.flow_loss import (
    sample_time, sample_times, sample_noise, build_xt_ut, flow_mse, mean_err,
    FM_ALPHA, FM_SCALE, FM_OFFSET,
)

comptime B = 2
comptime CHUNK = 4
comptime ADIM = 8           # the padded width
comptime ADIM_REAL = 3      # the "robot"
comptime ROW = CHUNK * ADIM
comptime TOT = B * ROW
comptime N_DRAW = 200000
comptime CDF_BAND = 0.006


def _ref_loss(mut vt: Tensor, mut ut: Tensor) -> Float64:
    """The loss straight from the definition, in Float64.

    ⚠ The finite differences below differentiate THIS, not `mean_err`. A first
    attempt differenced `mean_err` itself, whose `err` slab is fp32, and 15 of
    64 components missed a 1e-6 band by up to 3.9e-05 — the reference's own
    quantisation, not a wrong gradient
    (cf. `_a_tolerance_below_the_float32_noise_floor`).

    ⚠ And because the loss is QUADRATIC in v, a central difference of this is
    exact up to roundoff: the O(h^2) truncation term carries the third
    derivative, which is zero. That is why the band below is not negotiated.
    """
    var acc = 0.0
    for i in range(TOT):
        if i % ADIM < ADIM_REAL:
            var e = Float64(vt.data[i]) - Float64(ut.data[i])
            acc += e * e
    return acc / Float64(B * CHUNK * ADIM_REAL)


def _cdf_gap(ref draws: List[Float64], q: Float64) -> Float64:
    """|empirical CDF - x^alpha| for the UNSCALED beta variate at `q`."""
    var hit = 0
    for i in range(len(draws)):
        # undo scale/offset to get the beta variate back
        if (draws[i] - FM_OFFSET) / FM_SCALE <= q:
            hit += 1
    return abs(Float64(hit) / Float64(len(draws)) - exp(FM_ALPHA * log(q)))


def main() raises:
    print("=" * 70)
    print("SmolVLA flow-matching objective")
    print("=" * 70)

    # ── [1] the time law ─────────────────────────────────────────────────
    var draws = List[Float64]()
    var lo = 2.0
    var hi = -1.0
    var sum = 0.0
    for _ in range(N_DRAW):
        var t = sample_time()
        draws.append(t)
        sum += t
        if t < lo:
            lo = t
        if t > hi:
            hi = t
    var mean = sum / Float64(N_DRAW)
    # Beta(a, 1) has mean a/(a+1); scale and offset move it affinely.
    var want_mean = (FM_ALPHA / (FM_ALPHA + 1.0)) * FM_SCALE + FM_OFFSET
    print("  [1] t ~ Beta(", FM_ALPHA, ", 1)*", FM_SCALE, "+", FM_OFFSET)
    print("      draws", N_DRAW, " mean", mean, " want", want_mean,
          "  range [", lo, ",", hi, "]")
    assert_true(
        abs(mean - want_mean) < 0.004,
        "the sampled mean is not Beta(alpha, 1)'s",
    )
    assert_true(
        lo >= FM_OFFSET and hi <= FM_SCALE + FM_OFFSET,
        "a timestep fell outside [offset, scale+offset]",
    )

    var worst = 0.0
    var checked = 0
    for k in range(1, 10):
        var q = Float64(k) / 10.0
        var gap = _cdf_gap(draws, q)
        checked += 1
        if gap > worst:
            worst = gap
        print("      CDF at", q, ": |empirical - q^alpha| =", gap)
    assert_equal(checked, 9, "every decile must be probed")
    assert_true(
        worst < CDF_BAND,
        "the empirical CDF is not x^alpha — the time law is not Beta(alpha, 1)",
    )

    # ⚠ REJECTION. Without this the leg above only says "some distribution
    # passed"; a band loose enough for 200k samples is loose enough to accept
    # a lot. A uniform draw is the specific wrong thing most likely to be
    # written by hand, and it must FAIL by a wide margin.
    var unif = List[Float64]()
    for _ in range(N_DRAW):
        unif.append(random_float64() * FM_SCALE + FM_OFFSET)
    var uworst = 0.0
    for k in range(1, 10):
        var gap = _cdf_gap(unif, Float64(k) / 10.0)
        if gap > uworst:
            uworst = gap
    print("      uniform t, same test: worst gap", uworst, " (band",
          CDF_BAND, "-> must FAIL)")
    assert_true(
        uworst > CDF_BAND * 5.0,
        "a UNIFORM sampler passes this CDF check, so the check proves nothing",
    )

    # per-sample times, not one shared one
    var ts = sample_times(B)
    assert_equal(len(ts), B, "sample_times must give one t per batch element")
    assert_true(ts[0] != ts[1], "a batch got the same timestep twice")

    # ── [2] the interpolant's endpoints ──────────────────────────────────
    var noise = Tensor.alloc(TOT)
    var acts = Tensor.alloc(TOT)
    sample_noise(noise, TOT)
    for i in range(TOT):
        acts.data[i] = Scalar[DT](((i * 29) % 17) - 8) * 0.13
    var times = Tensor.alloc(B)
    var xt = Tensor.alloc(TOT)
    var ut = Tensor.alloc(TOT)

    # t = 1 -> x_t IS the noise, exactly.
    for b in range(B):
        times.data[b] = Scalar[DT](1.0)
    build_xt_ut["cpu", B, ROW](noise, acts, times, xt, ut, None)
    var e1 = 0
    var eu = 0
    for i in range(TOT):
        if xt.data[i] != noise.data[i]:
            e1 += 1
        if ut.data[i] != noise.data[i] - acts.data[i]:
            eu += 1
    print("  [2] t=1: x_t vs noise: compared", TOT, " differing", e1,
          " |  u_t vs noise-actions: differing", eu)
    assert_true(e1 == 0, "at t=1 the interpolant must BE the noise —"
                         " the convention is flipped")
    assert_true(eu == 0, "u_t is not noise - actions")

    # t = 0 -> x_t IS the actions, exactly.
    for b in range(B):
        times.data[b] = Scalar[DT](0.0)
    build_xt_ut["cpu", B, ROW](noise, acts, times, xt, ut, None)
    var e0 = 0
    for i in range(TOT):
        if xt.data[i] != acts.data[i]:
            e0 += 1
    print("      t=0: x_t vs actions: compared", TOT, " differing", e0)
    assert_true(e0 == 0, "at t=0 the interpolant must BE the action chunk")

    # a genuinely per-sample t: sample 0 at t=1, sample 1 at t=0.
    times.data[0] = Scalar[DT](1.0)
    times.data[1] = Scalar[DT](0.0)
    build_xt_ut["cpu", B, ROW](noise, acts, times, xt, ut, None)
    var mix = 0
    for j in range(ROW):
        if xt.data[j] != noise.data[j]:
            mix += 1
        if xt.data[ROW + j] != acts.data[ROW + j]:
            mix += 1
    print("      per-sample t (b0=1, b1=0): compared", 2 * ROW,
          " differing", mix)
    assert_true(
        mix == 0,
        "one timestep is being applied to the whole batch — the reference"
        " broadcasts time[:, None, None], one t PER SAMPLE",
    )

    # ── [3] the loss, its gradient, and the padded columns ───────────────
    var vt = Tensor.alloc(TOT)
    for i in range(TOT):
        vt.data[i] = Scalar[DT](((i * 41) % 23) - 11) * 0.08
    for b in range(B):
        times.data[b] = Scalar[DT](0.4)
    build_xt_ut["cpu", B, ROW](noise, acts, times, xt, ut, None)
    var gv = Tensor.alloc(TOT)
    var err = Tensor.alloc(TOT)
    flow_mse["cpu", B, CHUNK, ADIM, ADIM_REAL](vt, ut, gv, err, None)
    var loss = mean_err["cpu", B, CHUNK, ADIM, ADIM_REAL](err, None)

    # `mean_err` against the definition, before anything is differenced.
    var lref = _ref_loss(vt, ut)
    print("  [3] mean_err", loss, " vs the definition", lref, " diff",
          abs(loss - lref))
    assert_true(
        abs(loss - lref) < 1.0e-5,
        "mean_err is not the mean over real columns of (v - u)^2",
    )

    # every gradient component against a central difference of the Float64
    # reference. The loss is quadratic, so this is exact up to roundoff.
    var H = 1.0e-3
    var gbad = 0
    var gworst = 0.0
    for i in range(TOT):
        var keep = vt.data[i]
        # ⚠ Divide by the step ACTUALLY TAKEN, not the one asked for. `vt` is
        # fp32, so `keep + H` and `keep - H` each round, and their true
        # separation differs from 2H by ~1e-7. Dividing by the nominal 2H
        # leaves a relative error of 5e-05 — which showed up as 11 of 64
        # components missing a 1e-6 band by up to 4.4e-06, entirely an
        # artefact of the reference and not of the gradient.
        vt.data[i] = Scalar[DT](Float64(keep) + H)
        var ap = Float64(vt.data[i])
        var lp = _ref_loss(vt, ut)
        vt.data[i] = Scalar[DT](Float64(keep) - H)
        var am = Float64(vt.data[i])
        var lm = _ref_loss(vt, ut)
        vt.data[i] = keep
        var fd = (lp - lm) / (ap - am)
        var d = abs(Float64(gv.data[i]) - fd)
        if d > gworst:
            gworst = d
        if d > 1.0e-6:
            gbad += 1
    print("      dL/dv vs central differences: compared", TOT, " differing",
          gbad, " worst abs", gworst)
    assert_true(gbad == 0, "dL/dv disagrees with a central difference")

    # ⚠ THE padded-column property. A gradient of 1e-12 would pass leg [3]'s
    # band and still be wrong: those columns must be structurally dead.
    var padded = 0
    var padnz = 0
    var padmoves = 0
    for i in range(TOT):
        if i % ADIM >= ADIM_REAL:
            padded += 1
            if gv.data[i] != Scalar[DT](0):
                padnz += 1
            # and the LOSS must not move when a padded prediction moves
            var keep = vt.data[i]
            vt.data[i] = keep + Scalar[DT](100.0)
            flow_mse["cpu", B, CHUNK, ADIM, ADIM_REAL](vt, ut, gv, err, None)
            if mean_err["cpu", B, CHUNK, ADIM, ADIM_REAL](err, None) != loss:
                padmoves += 1
            vt.data[i] = keep
    flow_mse["cpu", B, CHUNK, ADIM, ADIM_REAL](vt, ut, gv, err, None)
    print("  [4] padded columns:", padded, "of", TOT, " nonzero grad", padnz,
          " loss moved by a +100 kick", padmoves)
    assert_equal(
        padded, TOT * (ADIM - ADIM_REAL) // ADIM, "padded column count"
    )
    assert_true(padnz == 0, "a padded action column has a nonzero gradient —"
                            " the network is being trained to predict noise")
    assert_true(
        padmoves == 0,
        "the loss depends on a padded action column: the"
        " losses[:, :, :real] slice is missing",
    )

    # and the real columns MUST move, or leg [4] is passing vacuously
    var realmoves = 0
    var reals = 0
    for i in range(TOT):
        if i % ADIM < ADIM_REAL:
            reals += 1
            var keep = vt.data[i]
            vt.data[i] = keep + Scalar[DT](1.0)
            flow_mse["cpu", B, CHUNK, ADIM, ADIM_REAL](vt, ut, gv, err, None)
            if mean_err["cpu", B, CHUNK, ADIM, ADIM_REAL](err, None) != loss:
                realmoves += 1
            vt.data[i] = keep
    flow_mse["cpu", B, CHUNK, ADIM, ADIM_REAL](vt, ut, gv, err, None)
    print("      control: real columns", reals, " loss moved", realmoves,
          "(must be all of them)")
    assert_true(
        realmoves == reals,
        "a REAL action column does not affect the loss — leg [4] is vacuous",
    )

    # ── [5] GPU parity ───────────────────────────────────────────────────
    var c = DeviceContext()
    var gn = Tensor.alloc(TOT)
    var ga = Tensor.alloc(TOT)
    var gt = Tensor.alloc(B)
    var gvt = Tensor.alloc(TOT)
    for i in range(TOT):
        gn.data[i] = noise.data[i]
        ga.data[i] = acts.data[i]
        gvt.data[i] = vt.data[i]
    for b in range(B):
        gt.data[b] = Scalar[DT](0.4)
    gn.upload(c)
    ga.upload(c)
    gt.upload(c)
    gvt.upload(c)
    var gxt = Tensor.alloc(TOT)
    var gut = Tensor.alloc(TOT)
    gxt.upload(c)
    gut.upload(c)
    build_xt_ut["gpu", B, ROW](gn, ga, gt, gxt, gut, Optional(c))
    var ggv = Tensor.alloc(TOT)
    var gerr = Tensor.alloc(TOT)
    ggv.upload(c)
    gerr.upload(c)
    flow_mse["gpu", B, CHUNK, ADIM, ADIM_REAL](gvt, gut, ggv, gerr,
                                               Optional(c))
    var gloss = mean_err["gpu", B, CHUNK, ADIM, ADIM_REAL](gerr, Optional(c))
    c.synchronize()
    gxt.download(c)
    gut.download(c)
    ggv.download(c)

    var pbad = 0
    var pworst = 0.0
    for i in range(TOT):
        var a = abs(Float64(gxt.data[i]) - Float64(xt.data[i]))
        var b2 = abs(Float64(gut.data[i]) - Float64(ut.data[i]))
        var d = abs(Float64(ggv.data[i]) - Float64(gv.data[i]))
        if a > pworst: pworst = a
        if b2 > pworst: pworst = b2
        if d > pworst: pworst = d
        if a > 1.0e-6 or b2 > 1.0e-6 or d > 1.0e-6:
            pbad += 1
    print("  [5] GPU vs CPU (x_t, u_t, dL/dv): compared", 3 * TOT,
          " differing", pbad, " worst", pworst, " | loss", gloss, "vs", loss)
    assert_true(pbad == 0, "the GPU path disagrees with the CPU one")
    assert_true(abs(gloss - loss) < 1.0e-6, "the GPU loss disagrees")

    print()
    print("PASSED — the time law, both endpoints, the gradient, and the"
          " padded columns")

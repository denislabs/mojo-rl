"""G8 — SWM Phase 3 gate (P2): what the v1 cocycle loss actually does.

The whole v2 rests on one claim: minimising the cocycle cannot help, because it
is either DESTRUCTIVE (free morphisms — it crushes the frustrated dimension) or
INERT (orthogonal morphisms — the loss is constant on the topological class).
This gate measures both, and the inert half is exact algebra rather than a
trend, so it is checked as an identity and not as a tolerance.

  `L = ||H - I||_F^2 = 4 - 2 tr H`. On the `det = -1` component of O(2) every
  `H` is a reflection, so `tr H = 0` and `L = 4` IDENTICALLY. Any tangent
  direction of the product manifold keeps `H` a reflection, so the tangent
  gradient is exactly zero.

Validates:
  - TANGENT GRADIENT, three ways. Mobius: raw gradient large, tangent
    projection at the float floor. Frustrated orientable (`det = +1`, `H != I`):
    tangent gradient plainly NON-zero — the control that makes the zero above a
    measurement rather than an artefact of a quantity that is always small.
    Flat orientable (`H = I`): both zero, and it is named as the degenerate
    case it is.
  - On LEARNED encodings, fitted on the same representations model A uses:
      A   det H = -1, residual at the noise floor, uniform
      B   translations: the odd/even PARITY gap
      C   free GL(2) + cocycle: |det H| collapses toward 0 as lambda grows, the
          minimum singular value shrinks, and the local residual rises
      C'  O(2) + cocycle: indistinguishable from A
  - GUARDRAIL: C at lambda = 10 must differ from A by a lot. Without it,
    "C' is inert" could equally be "the cocycle term was never wired in".

Run:
    pixi run mojo run -I . tests/experimental/swm/test_swmh_ablations.mojo
"""

from std.math import abs, sqrt, cos, sin
from std.testing import assert_true

from mojo_rl.experimental.swm.so_d import SqMat
from mojo_rl.experimental.swm.rng import Rng
from mojo_rl.experimental.swm.procrustes import (
    PairBatch,
    procrustes_o_d,
    mean_squared_residual,
)
from mojo_rl.experimental.swm.ablations import (
    holonomy_product,
    cocycle_tangent_norm,
    min_singular_value,
    fit_free_with_cocycle,
    fit_orthogonal_with_cocycle,
    fit_translations,
)
from mojo_rl.experimental.swm.swm_trainer import SwmPhase3, Phase3Config
from mojo_rl.experimental.swm.envs.mobius_ring import MobiusConfig

comptime DT = DType.float64
comptime N = 12
comptime TrainerT = SwmPhase3[12, 6, 16, 32, 8, DT]


def rot2(t: Float64) -> SqMat[2, DT]:
    var m = SqMat[2, DT]()
    m[0, 0] = Scalar[DT](cos(t))
    m[0, 1] = Scalar[DT](-sin(t))
    m[1, 0] = Scalar[DT](sin(t))
    m[1, 1] = Scalar[DT](cos(t))
    return m^


def planted_ring(kind: Int) raises -> List[SqMat[2, DT]]:
    """0 = Mobius, 1 = flat orientable (H = I), 2 = frustrated orientable."""
    var rng = Rng(3)
    var rs = List[SqMat[2, DT]]()
    var total = Float64(0)
    for i in range(N):
        var t = rng.uniform_range(-0.6, 0.6)
        if i == N - 1 and kind != 2:
            t = -total
        else:
            total += t
        var m = rot2(t)
        if kind == 0 and i == N - 1:
            var refl = SqMat[2, DT].identity()
            refl[1, 1] = Scalar[DT](-1)
            m = refl * m
        rs.append(m^)
    return rs^


def closure_errors(
    rs: List[SqMat[2, DT]],
    seq_u: List[Scalar[DT]],
    n_ep: Int,
    n_frames: Int,
    laps: Int,
) -> List[Float64]:
    """Mean ||predicted - observed|| after k full laps, k = 1..laps."""
    var out = List[Float64](length=laps, fill=0)
    for ep in range(n_ep):
        var x = List[Float64](length=2, fill=0)
        for c in range(2):
            x[c] = Float64(seq_u[(ep * n_frames + 0) * 2 + c])
        for k in range(1, laps + 1):
            for i in range(N):
                var nx = List[Float64](length=2, fill=0)
                for r in range(2):
                    var s = Float64(0)
                    for c in range(2):
                        s += Float64(rs[i][r, c]) * x[c]
                    nx[r] = s
                for c in range(2):
                    x[c] = nx[c]
            var d = Float64(0)
            for c in range(2):
                var e = x[c] - Float64(seq_u[(ep * n_frames + k * N) * 2 + c])
                d += e * e
            out[k - 1] += sqrt(d)
    for k in range(laps):
        out[k] /= Float64(n_ep)
    return out^


def closure_errors_translation(
    ts: List[Scalar[DT]],
    seq_u: List[Scalar[DT]],
    n_ep: Int,
    n_frames: Int,
    laps: Int,
) -> List[Float64]:
    var out = List[Float64](length=laps, fill=0)
    for ep in range(n_ep):
        var x = List[Float64](length=2, fill=0)
        for c in range(2):
            x[c] = Float64(seq_u[(ep * n_frames + 0) * 2 + c])
        for k in range(1, laps + 1):
            for i in range(N):
                for c in range(2):
                    x[c] += Float64(ts[i * 2 + c])
            var d = Float64(0)
            for c in range(2):
                var e = x[c] - Float64(seq_u[(ep * n_frames + k * N) * 2 + c])
                d += e * e
            out[k - 1] += sqrt(d)
    for k in range(laps):
        out[k] /= Float64(n_ep)
    return out^


def mean_residual(
    batches: List[PairBatch[2, DT]], rs: List[SqMat[2, DT]]
) -> Float64:
    var t = Float64(0)
    for e in range(N):
        t += Float64(mean_squared_residual[2, DT](batches[e], rs[e]))
    return t / Float64(N)


def main() raises:
    var checks = 0

    # =====================================================================
    # 1. The exact claim: the cocycle loss has NO tangent gradient on the
    #    det = -1 component.
    # =====================================================================
    var labels: List[String] = [
        "mobius (det=-1)", "flat orientable (H=I)", "frustrated orientable"
    ]
    var tangents = List[Float64]()
    var raws = List[Float64]()
    for kind in range(3):
        var rs = planted_ring(kind)
        var h = holonomy_product[2, DT](rs)
        var nrm = cocycle_tangent_norm[2, DT](rs)
        raws.append(nrm[0])
        tangents.append(nrm[1])
        print(
            "  ", labels[kind], ": det H =", h.det(),
            " |H-I| =", h.dist_to_identity(),
            " raw grad =", nrm[0], " TANGENT =", nrm[1],
        )

    checks += 4
    assert_true(
        raws[0] > 1.0,
        "the RAW cocycle gradient on the Mobius ring must be large — if it is "
        + "not, the zero tangent below is vacuous. got " + String(raws[0]),
    )
    assert_true(
        tangents[0] < 1e-10,
        "the cocycle loss must have ZERO tangent gradient on the det = -1 "
        + "component (it is constant there), got " + String(tangents[0]),
    )
    assert_true(
        tangents[2] > 1.0,
        "CONTROL FAILED: with CONTINUOUS frustration (det = +1, H != I) the "
        + "tangent gradient must be non-zero, otherwise the measurement above "
        + "is an artefact of a quantity that is always small. got "
        + String(tangents[2]),
    )
    assert_true(
        tangents[0] < 1e-6 * tangents[2],
        "the inert/active contrast must be many orders of magnitude, got "
        + String(tangents[0]) + " vs " + String(tangents[2]),
    )

    # =====================================================================
    # 2. The arms, fitted on LEARNED encodings.
    # =====================================================================
    var cfg = Phase3Config.default()
    cfg.seed = 20260904
    var model = TrainerT.train(MobiusConfig.default_mobius(), cfg)
    var roll = TrainerT.encode_rollouts(
        model, MobiusConfig.default_mobius(), cfg, 24
    )
    var pairs_per_edge = roll.batches[0].count()
    checks += 1
    assert_true(
        pairs_per_edge >= 24,
        "too few pairs per edge to fit anything: " + String(pairs_per_edge),
    )

    # ---- A ---------------------------------------------------------------
    var a_rs = List[SqMat[2, DT]]()
    for e in range(N):
        a_rs.append(procrustes_o_d[2, DT](roll.batches[e]))
    var a_det = Float64(holonomy_product[2, DT](a_rs).det())
    var a_res = mean_residual(roll.batches, a_rs)
    var a_lce = closure_errors(a_rs, roll.seq_u, roll.n_episodes, roll.n_frames, cfg.laps)
    checks += 1
    assert_true(
        a_det < -0.99,
        "arm A on learned encodings must give det H = -1, got " + String(a_det),
    )
    print("  A  (O(2), holonomy read)      det H =", a_det,
          " resid =", a_res, " closure k=1:", a_lce[0])

    # ---- B: translations -> parity ---------------------------------------
    var b_ts = fit_translations[DT](roll.batches)
    var b_lce = closure_errors_translation(
        b_ts, roll.seq_u, roll.n_episodes, roll.n_frames, cfg.laps
    )
    var odd_b = b_lce[0]
    var even_b = b_lce[1]
    var gap = odd_b / even_b
    # The §1.2 table's "7-10x odd/even parity gap" does NOT transfer verbatim to
    # learned encodings, and the reason is exact rather than a tolerance issue.
    # In the oracle-frame prototype the observations are centred, so the fitted
    # translations are exactly ZERO: model B predicts "nothing moves", which is
    # RIGHT after an even number of laps (H^2 = I) and wrong after an odd one —
    # hence a clean parity ratio with B nearly matching A at even k. A learned
    # frame carries an offset (measured |mean|/|std| ~ 0.22), so the fitted
    # translations are non-zero, accumulate, and B fails at EVERY lap. Its
    # odd/even ratio therefore compresses (1.8x) while its failure against A
    # gets much worse (13-27x). That is a stronger refutation of the constant
    # sheaf, not a weaker one, so this gate asserts the B-vs-A gap and reports
    # the parity ratio rather than asserting the prototype's number.
    var ba_worst = b_lce[0] / a_lce[0]
    for k in range(cfg.laps):
        var r = b_lce[k] / a_lce[k]
        if r < ba_worst:
            ba_worst = r
    checks += 2
    assert_true(
        ba_worst > 5.0,
        "ablation B must be far worse than A at EVERY lap count; worst ratio "
        + String(ba_worst),
    )
    assert_true(
        gap > 1.5,
        "ablation B must still show an odd/even asymmetry on learned "
        + "encodings: odd/even = " + String(gap),
    )
    print("  A  closure k=1..3:", a_lce[0], a_lce[1], a_lce[2])
    print("  B  closure k=1..3:", b_lce[0], b_lce[1], b_lce[2],
          " odd/even =", gap, " worst B/A =", ba_worst)

    # ---- C: free GL(2) + cocycle -> collapse ------------------------------
    var lams: List[Float64] = [0.1, 1.0, 10.0]
    var prev_absdet = 1.0
    for i in range(len(lams)):
        var c_rs = fit_free_with_cocycle[DT](roll.batches, lams[i], 1500)
        var c_det = Float64(holonomy_product[2, DT](c_rs).det())
        var c_res = mean_residual(roll.batches, c_rs)
        var smin = 1.0
        for e in range(N):
            var sv = min_singular_value[DT](c_rs[e])
            if sv < smin:
                smin = sv
        var c_lce = closure_errors(
            c_rs, roll.seq_u, roll.n_episodes, roll.n_frames, cfg.laps
        )
        print("  C  (free GL(2)+cocycle) l=", lams[i], " det H =", c_det,
              " min sv =", smin, " resid =", c_res, " closure k=1:", c_lce[0])
        checks += 2
        assert_true(
            abs(c_det) < prev_absdet + 1e-9,
            "|det H| must not GROW as the cocycle weight rises: "
            + String(abs(c_det)) + " after " + String(prev_absdet),
        )
        assert_true(
            c_res > a_res,
            "arm C must pay in local residual (lambda=" + String(lams[i])
            + "): " + String(c_res) + " vs A " + String(a_res),
        )
        prev_absdet = abs(c_det)
        if i == len(lams) - 1:
            checks += 2
            assert_true(
                abs(c_det) < 0.5,
                "at lambda=10 the free-morphism arm must have crushed the "
                + "frustrated dimension (|det H| -> 0), got " + String(abs(c_det)),
            )
            assert_true(
                smin < 0.9,
                "GUARDRAIL: the minimum singular value must have shrunk, "
                + "otherwise the cocycle term is not wired in. got "
                + String(smin),
            )

    # ---- C': O(2) + cocycle -> inert --------------------------------------
    # C' comes out BIT-IDENTICAL to A, which is normally the signature of an
    # unchanged binary. Here it is the correct answer, and it is not taken on
    # trust: arm C above is driven by the SAME `cocycle_grad`, and it visibly
    # collapsed (det H -1.0 -> -0.024, min singular value 1.0 -> 0.64). So the
    # term is wired; it simply has nothing to push against once the transports
    # are orthogonal and sit at the Procrustes optimum, because both the fit
    # gradient and the cocycle's TANGENT gradient are zero there.
    var cp_identical = 0
    for i in range(2):
        var lam = 1.0 if i == 0 else 10.0
        var cp_rs = fit_orthogonal_with_cocycle[DT](roll.batches, lam, 1500)
        var cp_det = Float64(holonomy_product[2, DT](cp_rs).det())
        var cp_res = mean_residual(roll.batches, cp_rs)
        print("  C' (O(2)+cocycle)      l=", lam, " det H =", cp_det,
              " resid =", cp_res, " (A resid =", a_res, ")")
        checks += 2
        assert_true(
            cp_det < -0.99,
            "arm C' must keep det H = -1 — the loss is constant on that "
            + "component, so it cannot move it. got " + String(cp_det),
        )
        assert_true(
            abs(cp_res - a_res) < 0.25 * a_res + 1e-12,
            "arm C' must be INDISTINGUISHABLE from A in residual: "
            + String(cp_res) + " vs " + String(a_res),
        )
        if cp_res == a_res:
            cp_identical += 1

    print()
    print("C' arms bit-identical to A:", cp_identical, "/ 2",
          " (correct: both gradients vanish there; arm C proves the term is wired)")
    print("pairs per edge      :", pairs_per_edge, " edges:", N)
    print("assertions compared :", checks)
    print("PASS: G8 the cocycle loss is destructive (C) or inert (C'), never useful")

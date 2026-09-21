"""FB loss gate — gradients, and the two collapses the losses exist to prevent.

`docs/BFM_ZERO_SHOT_RL.md` §11 lists two silent failures that live here. Both
have the same signature: the loss curve looks healthy and the representation is
empty. Neither is detectable from a training run, so both are gated by
construction:

  [3] **the anchor term.** `-2·E[F·B(s')]` is the expansion of the square, not a
      regulariser. Without it, `F = 0` is a perfect global minimum of the
      residual term — the bootstrapped target is then also zero and the loss is
      exactly 0. The gate asserts the discriminating fact: at `F = 0` the
      gradient wrt `F` is EXACTLY zero without the anchor and non-zero with it.
      A test that merely compared loss VALUES would not distinguish "the anchor
      shifts the loss by a constant" from "the anchor creates a descent
      direction", and only the second is the point.

  [4] **B collapse.** `L_ortho` must score a rank-1 `B` (every row the same
      direction) strictly worse than a spread-out one. Checked as an ordering
      between two concrete `B`s rather than against a magic threshold.

      ⚠ [2] and [4] are NECESSARY AND NOT SUFFICIENT, and the first G1 run is
      the proof: both passed for months against an `L_ortho` whose minimiser
      was a rank-d/2 collapse. [2] checks the gradient against the loss, and
      the LOSS was the bug; [4] compares two hand-built points and never
      visits the minimiser. Only following the gradient can see it, which is
      `test_fb_ortho_fixed_point.mojo`.

[1] and [2] are central finite differences of every input of both losses.

Run:
    pixi run mojo run -I . tests/deep_agents/test_fb_loss.mojo
"""

from std.math import abs, sqrt
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.deep_agents.fb.loss import (
    fb_measure_loss,
    fb_ortho_loss,
    pairwise_matrix,
)


comptime D: Int = 5
comptime BATCH: Int = 4
comptime EPS: Float64 = 1e-3
comptime FD_TOL: Float64 = 2e-2


def _fill(mut t: Tensor, n: Int, a: Float64, b: Float64, c: Float64):
    for i in range(n):
        t.data[i] = Scalar[DT](
            a * Float64(i % 7) + b * Float64((i * 3) % 5) + c
        )


def _mk(n: Int, a: Float64, b: Float64, c: Float64) raises -> Tensor:
    var t = Tensor.alloc(n)
    _fill(t, n, a, b, c)
    return t^


def _measure_only_loss(
    ref f: Tensor, ref bg: Tensor, ref mt: Tensor, with_anchor: Bool,
) raises -> Float64:
    var gf = Tensor.alloc(BATCH * D)
    var gbg = Tensor.alloc(BATCH * D)
    return fb_measure_loss[D, BATCH](f, bg, mt, gf, gbg, with_anchor)


def test_measure_loss_gradients() raises:
    print("[1] fb_measure_loss vs central finite differences ...")
    var f = _mk(BATCH * D, 0.13, -0.21, 0.05)
    var bg = _mk(BATCH * D, -0.17, 0.29, -0.2)
    var mt = _mk(BATCH * BATCH, 0.07, 0.03, -0.1)
    var gf = Tensor.alloc(BATCH * D)
    var gbg = Tensor.alloc(BATCH * D)
    var base = fb_measure_loss[D, BATCH](f, bg, mt, gf, gbg, True)
    print("      L_FB =", base)

    var worst_f = Float64(0)
    var worst_b = Float64(0)

    for idx in range(BATCH * D):
        var keep = f.data[idx]
        f.data[idx] = Scalar[DT](Float64(keep) + EPS)
        var lp = _measure_only_loss(f, bg, mt, True)
        f.data[idx] = Scalar[DT](Float64(keep) - EPS)
        var lm = _measure_only_loss(f, bg, mt, True)
        f.data[idx] = keep
        var fd = (lp - lm) / (2.0 * EPS)
        var an = Float64(gf.data[idx])
        var den = abs(an) if abs(an) > 0.1 else 0.1
        var rel = abs(fd - an) / den
        if rel > worst_f:
            worst_f = rel

    # ONE B tensor now, so a single perturbation moves the matrix, its
    # diagonal and the anchor together — which is exactly the derivative the
    # trainer applies.
    for idx in range(BATCH * D):
        var keep = bg.data[idx]
        bg.data[idx] = Scalar[DT](Float64(keep) + EPS)
        var lp = _measure_only_loss(f, bg, mt, True)
        bg.data[idx] = Scalar[DT](Float64(keep) - EPS)
        var lm = _measure_only_loss(f, bg, mt, True)
        bg.data[idx] = keep
        var fd = (lp - lm) / (2.0 * EPS)
        var an = Float64(gbg.data[idx])
        var den = abs(an) if abs(an) > 0.1 else 0.1
        var rel = abs(fd - an) / den
        if rel > worst_b:
            worst_b = rel

    print("      worst rel err: dF", worst_f, " dB(goal)", worst_b)
    assert_true(worst_f < FD_TOL, "dF: " + String(worst_f))
    assert_true(worst_b < FD_TOL, "dB(goal): " + String(worst_b))


def test_ortho_loss_gradients() raises:
    print("[2] fb_ortho_loss vs central finite differences ...")
    var b = _mk(BATCH * D, 0.19, -0.23, 0.31)
    var g = Tensor.alloc(BATCH * D)
    var base = fb_ortho_loss[D, BATCH](b, g)
    print("      L_ortho =", base)

    var sink = Tensor.alloc(BATCH * D)
    var worst = Float64(0)

    # `B` is both inputs of the pairwise dot, so a single perturbation moves
    # BOTH sides — which is exactly the derivative the trainer needs.
    for idx in range(BATCH * D):
        var keep = b.data[idx]
        b.data[idx] = Scalar[DT](Float64(keep) + EPS)
        var lp = fb_ortho_loss[D, BATCH](b, sink)
        b.data[idx] = Scalar[DT](Float64(keep) - EPS)
        var lm = fb_ortho_loss[D, BATCH](b, sink)
        b.data[idx] = keep
        var fd = (lp - lm) / (2.0 * EPS)
        var an = Float64(g.data[idx])
        var den = abs(an) if abs(an) > 0.1 else 0.1
        var rel = abs(fd - an) / den
        if rel > worst:
            worst = rel

    print("      worst rel err: dB(s+)", worst)
    assert_true(worst < FD_TOL, "dB(s+): " + String(worst))


def test_anchor_term_is_load_bearing() raises:
    """At `F = 0` the anchor must be the ONLY source of gradient.

    Without it, `F = 0` with a zero target is an exact global minimum: loss 0,
    gradient 0, nothing to descend. That is the "decreasing loss, empty
    representation" failure, reproduced here rather than described.
    """
    print("[3] the -2·E[F·B(s')] term creates the descent direction ...")
    var f0 = Tensor.alloc(BATCH * D)          # F = 0
    for i in range(BATCH * D):
        f0.data[i] = Scalar[DT](0)
    var bg = _mk(BATCH * D, -0.17, 0.29, -0.2)
    var mt = Tensor.alloc(BATCH * BATCH)      # target also 0 (F=0 bootstrapped)
    for i in range(BATCH * BATCH):
        mt.data[i] = Scalar[DT](0)

    var gf_no = Tensor.alloc(BATCH * D)
    var gbg_no = Tensor.alloc(BATCH * D)
    var l_no = fb_measure_loss[D, BATCH](f0, bg, mt, gf_no, gbg_no, False)

    var gf_yes = Tensor.alloc(BATCH * D)
    var gbg_yes = Tensor.alloc(BATCH * D)
    var l_yes = fb_measure_loss[D, BATCH](f0, bg, mt, gf_yes, gbg_yes, True)

    var gn_no = Float64(0)
    var gn_yes = Float64(0)
    for i in range(BATCH * D):
        gn_no += abs(Float64(gf_no.data[i]))
        gn_yes += abs(Float64(gf_yes.data[i]))

    print("      F=0:  loss", l_no, "->", l_yes,
          " |dF|_1", gn_no, "->", gn_yes)

    assert_true(
        abs(l_no) < 1e-9,
        "without the anchor, F=0 should be an EXACT minimum (loss 0), got "
        + String(l_no) + " — the ablation is not reproducing the collapse it"
        " is meant to demonstrate",
    )
    assert_true(
        gn_no < 1e-9,
        "without the anchor the gradient at F=0 must vanish; got "
        + String(gn_no),
    )
    assert_true(
        gn_yes > 1e-3,
        "WITH the anchor the gradient at F=0 must NOT vanish; got "
        + String(gn_yes) + ". If it does, the anchor term is not wired in and"
        " training will converge to the empty representation.",
    )

    # And the gradient must point somewhere useful. The anchor is now the
    # DIAGONAL of the same matrix, `-2·mean_i(M_ii - Mt_ii)`, so at F = 0 with
    # a zero target the off-diagonal residual contributes nothing and
    # dL/dF_i = -2/BATCH · B(goal)_i exactly — a step against it increases
    # F_i·B(goal)_i, which is the anchor's whole purpose.
    var worst = Float64(0)
    for i in range(BATCH * D):
        var want = -2.0 / Float64(BATCH) * Float64(bg.data[i])
        var e = abs(Float64(gf_yes.data[i]) - want)
        if e > worst:
            worst = e
    assert_true(
        worst < 1e-5,
        "the anchor gradient at F=0 is not -2/BATCH·B(goal): worst "
        + String(worst),
    )


def test_ortho_penalises_collapse() raises:
    """A rank-1 `B` must score strictly worse than a spread-out one.

    Uses two `B`s of the SAME Frobenius norm, so the ordering cannot be
    explained by one simply being larger — which is what the `-2·E[||B||^2]`
    term would otherwise reward.
    """
    print("[4] L_ortho ranks a collapsed B worse than a spread one ...")
    # Collapsed: every row the same direction.
    var collapsed = Tensor.alloc(BATCH * D)
    for i in range(BATCH):
        for k in range(D):
            collapsed.data[i * D + k] = Scalar[DT](1.0 if k == 0 else 0.0)
    # Spread: rows on distinct axes.
    var spread = Tensor.alloc(BATCH * D)
    for i in range(BATCH):
        for k in range(D):
            spread.data[i * D + k] = Scalar[DT](1.0 if k == (i % D) else 0.0)

    var nc = Float64(0)
    var ns = Float64(0)
    for i in range(BATCH * D):
        nc += Float64(collapsed.data[i]) * Float64(collapsed.data[i])
        ns += Float64(spread.data[i]) * Float64(spread.data[i])
    assert_true(
        abs(nc - ns) < 1e-6,
        "the two probe Bs have different Frobenius norms (" + String(nc)
        + " vs " + String(ns) + "), so the comparison below measures scale"
        " rather than collapse",
    )

    var g1 = Tensor.alloc(BATCH * D)
    var l_col = fb_ortho_loss[D, BATCH](collapsed, g1)
    var l_spr = fb_ortho_loss[D, BATCH](spread, g1)
    print("      L_ortho collapsed", l_col, " spread", l_spr)
    assert_true(
        l_col > l_spr + 1e-6,
        "L_ortho scored the collapsed B (" + String(l_col) + ") no worse than"
        " the spread one (" + String(l_spr) + ") — it is not preventing the"
        " collapse it exists for",
    )


def main() raises:
    test_measure_loss_gradients()
    test_ortho_loss_gradients()
    test_anchor_term_is_load_bearing()
    test_ortho_penalises_collapse()
    print("\n[PASS] FB loss gate")

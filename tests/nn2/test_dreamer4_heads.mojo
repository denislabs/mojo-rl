"""Dreamer 4 MTP policy + reward heads — overfit via dist losses (Phase 3.5).

    pixi run mojo run -I . tests/nn2/test_dreamer4_heads.mojo

The heads are plain nn2 Sequential MLPs producing NMTP distance-major logit
blocks. This wires them to the categorical (dists_discrete) and symexp-twohot
(twohot) losses and overfits fixed per-(sample, distance) targets:
  - policy: minimise Σ −logp(a) → argmax logits recovers every target action;
  - reward: minimise twohot CE → twohot_pred recovers every target reward.
Proves the heads + dist-loss backward + Adam path are correct end-to-end.
"""

from std.memory import alloc
from std.math import sin, abs
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.deep_agents2.dreamer4.heads import (
    Dreamer4PolicyHead, Dreamer4RewardHead,
)
from mojo_rl.deep_agents2.dreamerv3.dists_discrete import cat_fwd, cat_bwd, cat_argmax
from mojo_rl.deep_agents2.dreamerv3.twohot import (
    symexp_twohot_bins, twohot_loss, twohot_loss_backward, twohot_pred,
)


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


comptime D_IN = 8
comptime HID = 32
comptime NACT = 4
comptime NBINS = 41
comptime NMTP = 3            # small L for the test (production L+1=9)
comptime B = 4
comptime PLOG = NMTP * NACT  # policy logits per sample
comptime RLOG = NMTP * NBINS # reward logits per sample
comptime ZERO = Scalar[DT](0.0)


def main() raises:
    print("=" * 70)
    print("Dreamer 4 MTP heads — overfit policy + reward (Phase 3.5)")
    print("=" * 70)

    comptime PH = Dreamer4PolicyHead[D_IN, HID, NACT, NMTP]
    comptime RH = Dreamer4RewardHead[D_IN, HID, NBINS, NMTP]
    var ph = PH.make[target="cpu", INIT=Xavier]()
    var rh = RH.make[target="cpu", INIT=Xavier]()
    var ap = Adam.make["cpu", M=PH](ph)
    var ar = Adam.make["cpu", M=RH](rh)
    ap.lr = Scalar[DT](3e-3)
    ar.lr = Scalar[DT](3e-3)

    # fixed input embeddings (stand in for h_t)
    var h = _alloc(B * D_IN)
    for i in range(B * D_IN):
        h[i] = Scalar[DT](0.5 * sin(0.3 + 0.7 * Float64(i)))
    var ht = TileTensor(h, row_major[B, D_IN]())

    # per-(sample, distance) targets
    var tgt_a = List[Int]()
    var tgt_r = _alloc(B * NMTP)
    for b in range(B):
        for n in range(NMTP):
            tgt_a.append((b + 2 * n) % NACT)
            tgt_r[b * NMTP + n] = Scalar[DT](0.5 * Float64((b - n)))

    var bins = _alloc(NBINS)
    symexp_twohot_bins[NBINS](bins, lo=Scalar[DT](-9.0))

    var plog = _alloc(B * PLOG)
    var rlog = _alloc(B * RLOG)
    var gpl = _alloc(B * PLOG)
    var grl = _alloc(B * RLOG)
    var gh = _alloc(B * D_IN)              # grad wrt h (discarded here)
    var sm = _alloc(NACT)
    var pp = _alloc(NACT)

    var plt = TileTensor(plog, row_major[B, PLOG]())
    var rlt = TileTensor(rlog, row_major[B, RLOG]())
    var gplt = TileTensor(gpl, row_major[B, PLOG]())
    var grlt = TileTensor(grl, row_major[B, RLOG]())
    var ght = TileTensor(gh, row_major[B, D_IN]())

    var first_p: Float64 = 0.0
    var last_p: Float64 = 0.0
    var first_r: Float64 = 0.0
    var last_r: Float64 = 0.0

    for step in range(400):
        # ── policy ──────────────────────────────────────────────────────
        ap.zero_grad["cpu"](ph)
        ph.forward["cpu", B](ht, output=plt)
        var loss_p: Float64 = 0.0
        for i in range(B * PLOG):
            gpl[i] = ZERO
        for b in range(B):
            for n in range(NMTP):
                var base = b * PLOG + n * NACT
                var k = tgt_a[b * NMTP + n]
                var lp_ent = cat_fwd[NACT](plog, base, ZERO, k, sm, pp)
                loss_p += -Float64(lp_ent[0])
                cat_bwd[NACT](
                    sm, pp, ZERO, k, Scalar[DT](-1.0), ZERO, gpl, base
                )
        ph.vjp["cpu", B](gplt, ght)
        ap.step["cpu"](ph)

        # ── reward ──────────────────────────────────────────────────────
        ar.zero_grad["cpu"](rh)
        rh.forward["cpu", B](ht, output=rlt)
        var loss_r: Float64 = 0.0
        for i in range(B * RLOG):
            grl[i] = ZERO
        for b in range(B):
            for n in range(NMTP):
                var base = b * RLOG + n * NBINS
                var tr = tgt_r[b * NMTP + n]
                loss_r += Float64(twohot_loss[NBINS](rlog, base, bins, tr))
                twohot_loss_backward[NBINS](
                    rlog, base, bins, tr, Scalar[DT](1.0), grl
                )
        rh.vjp["cpu", B](grlt, ght)
        ar.step["cpu"](rh)

        if step == 0:
            first_p = loss_p
            first_r = loss_r
        last_p = loss_p
        last_r = loss_r
        if step % 80 == 0:
            print("   step", step, " policy NLL =", loss_p, " reward CE =", loss_r)

    print("   policy NLL", first_p, "->", last_p)
    print("   reward CE ", first_r, "->", last_r)

    # ── final: every target action recovered by argmax; reward pred close ─
    ph.forward["cpu", B](ht, output=plt)
    rh.forward["cpu", B](ht, output=rlt)
    var n_correct = 0
    var max_rew_err: Float64 = 0.0
    for b in range(B):
        for n in range(NMTP):
            var pbase = b * PLOG + n * NACT
            if cat_argmax[NACT](plog, pbase) == tgt_a[b * NMTP + n]:
                n_correct += 1
            var rbase = b * RLOG + n * NBINS
            var pred = twohot_pred[NBINS](rlog, rbase, bins)
            var err = abs(Float64(pred) - Float64(tgt_r[b * NMTP + n]))
            if err > max_rew_err:
                max_rew_err = err
    print("   action accuracy =", n_correct, "/", B * NMTP)
    print("   max reward pred err =", max_rew_err)

    # Policy NLL → ~0 (categorical CE floor is 0). Reward CE floors at the
    # two-hot target entropy (never 0), so check a substantial drop + that the
    # decoded reward (twohot_pred) tracks the target to within a bin or so.
    assert_true(last_p < 0.05 * first_p, "policy NLL must collapse")
    assert_true(last_r < 0.2 * first_r, "reward CE must drop substantially")
    assert_true(n_correct == B * NMTP, "all target actions recovered")
    assert_true(max_rew_err < 0.2, "reward predictions track targets")

    print("=" * 70)
    print("ALL PASSED — Dreamer 4 MTP heads (Phase 3.5)")
    print("=" * 70)

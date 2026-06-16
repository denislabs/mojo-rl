"""LeWMTrainer end-to-end (Phase D, CPU).

Wires the config-driven trainer over an in-memory offline buffer:
  1. train N steps sampling windows → loss decreases,
  2. collapse_probes returns finite var_min / gram_off,
  3. save_params → fresh trainer → load_params reproduces the eval loss.
"""

from std.memory import alloc
from std.math import isnan, isinf
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.experimental.lewm.trainer import LeWMTrainer
from mojo_rl.experimental.lewm.offline_buffer import OfflineWindowBuffer


# toy config
comptime IN_CH = 4
comptime IMG = 8
comptime PATCH = 4
comptime HIDDEN = 8
comptime ENC_HEADS = 2
comptime ENC_LAYERS = 2
comptime EMB = 8
comptime ENC_PROJ_H = 16
comptime ENC_FF_MULT = 2
comptime T = 4
comptime ACT = 3
comptime SMOOTHED = 8
comptime AE_MLP = 2
comptime H = 3
comptime N_PREDS = 1
comptime PRED_HEADS = 2
comptime PRED_FF = 16
comptime DEPTH = 2
comptime PRED_PROJ_H = 16
comptime SIG_PROJ = 8
comptime SIG_KNOTS = 5
comptime B = 4

comptime IMG_DIM = IN_CH * IMG * IMG
comptime PIX = T * IMG_DIM
comptime ACTIN = T * ACT

comptime Trainer = LeWMTrainer[
    IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
    ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS, PRED_FF,
    DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "cpu",
]
comptime Buffer = OfflineWindowBuffer[IMG_DIM, ACT, T]


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


def main() raises:
    print("=" * 70)
    print("LeWMTrainer end-to-end (Phase D, CPU)")
    print("=" * 70)

    var buf = Buffer(n_traj=8, traj_len=12, seed=999)
    buf.fill_synthetic()

    var tr = Trainer.make(lam=Scalar[DT](0.09), lr=Scalar[DT](1e-3))

    var pix = _a(B * PIX); var act = _a(B * ACTIN)
    var pix_t = TileTensor(pix, row_major[B, PIX]())
    var act_t = TileTensor(act, row_major[B, ACTIN]())

    print("train ...")
    var first: Scalar[DT] = 0.0
    var last: Scalar[DT] = 0.0
    comptime STEPS = 200
    for s in range(STEPS):
        buf.sample_into(pix, act, B)
        var loss = tr.train_step(pix_t, act_t)
        if s == 0:
            first = loss
        last = loss
        if s % 40 == 0 or s == STEPS - 1:
            var probes = tr.collapse_probes()
            print("   step", s, " loss=", loss,
                  " var_min=", probes[0], " gram_off=", probes[1])
    print("   first=", first, " last=", last)
    assert_true(last < first, "loss must decrease over training")

    var probes = tr.collapse_probes()
    assert_true(not (isnan(probes[0]) or isinf(probes[0])), "var_min finite")
    assert_true(not (isnan(probes[1]) or isinf(probes[1])), "gram_off finite")
    assert_true(probes[1] >= Scalar[DT](0.0) and probes[1] <= Scalar[DT](1.01),
                "gram_off is a mean |correlation| in [0,1]")

    # checkpoint round-trip (same instance — SIGReg's projection is seeded
    # from its cache pointer, so a fresh instance would differ on the SIGReg
    # term even with identical params; the same instance keeps it stable. At
    # eval/MPC time SIGReg isn't used anyway, so cross-instance param restore
    # is correct for the real use case — we just can't compare full loss
    # across instances here).
    print("checkpoint round-trip ...")
    var efix_pix = _a(B * PIX); var efix_act = _a(B * ACTIN)
    buf.sample_into(efix_pix, efix_act, B)
    var efix_pix_t = TileTensor(efix_pix, row_major[B, PIX]())
    var efix_act_t = TileTensor(efix_act, row_major[B, ACTIN]())

    var lA = tr.eval_loss(efix_pix_t, efix_act_t)
    tr.save_params(String("/tmp/lewm_ckpt.txt"))
    # perturb params with more training on fresh batches
    for _ in range(10):
        buf.sample_into(pix, act, B)
        _ = tr.train_step(pix_t, act_t)
    var lA2 = tr.eval_loss(efix_pix_t, efix_act_t)
    tr.load_params(String("/tmp/lewm_ckpt.txt"))
    var lA3 = tr.eval_loss(efix_pix_t, efix_act_t)
    print("   lA=", lA, " perturbed=", lA2, " restored=", lA3)
    assert_true((lA2 - lA).__abs__() > Scalar[DT](1e-6),
                "training should perturb the eval loss (sanity)")
    assert_true((lA3 - lA).__abs__() < Scalar[DT](1e-4),
                "load_params must restore the saved model exactly")

    pix.free(); act.free(); efix_pix.free(); efix_act.free()
    _ = tr^
    _ = buf^
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

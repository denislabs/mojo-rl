"""lewm_shuffled_eval test (CPU, toy) — continuous-action-compatible.

The shuffle-based action-awareness eval (legacy H6) needs no action
sampling, so it works for continuous actions (PushT). This validates the
pipeline on the toy synthetic setup: expert + shuffled scores finite and
positive, frac_shuffled_worse in [0,1].

Run:  pixi run mojo run -I . tests/experimental/lewm/test_shuffled_eval.mojo
"""

from std.memory import alloc
from std.math import isnan, isinf
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.experimental.lewm.trainer import LeWMTrainer
from mojo_rl.experimental.lewm.offline_buffer import OfflineWindowBuffer
from mojo_rl.experimental.lewm.eval import lewm_shuffled_eval


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
comptime B = 6

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
    print("lewm_shuffled_eval (CPU, toy)")
    print("=" * 70)

    var buf = Buffer(n_traj=8, traj_len=12, seed=909)
    buf.fill_synthetic()
    var tr = Trainer.make(lam=Scalar[DT](0.09), lr=Scalar[DT](1e-3))

    var pix = _a(B * PIX); var act = _a(B * ACTIN)
    var pix_t = TileTensor(pix, row_major[B, PIX]())
    var act_t = TileTensor(act, row_major[B, ACTIN]())
    print("train 120 steps ...")
    for _ in range(120):
        buf.sample_into(pix, act, B)
        _ = tr.train_step(pix_t, act_t)

    buf.sample_into(pix, act, B)
    print("shuffled eval ...")
    var r = lewm_shuffled_eval[
        IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
        ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS,
        PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "cpu",
    ](tr, pix_t, act)

    assert_true(not (isnan(r[0]) or isinf(r[0])), "expert finite")
    assert_true(not (isnan(r[1]) or isinf(r[1])), "shuffled_mean finite")
    assert_true(r[0] > 0.0 and r[1] > 0.0, "scores positive (MSE)")
    assert_true(r[3] >= 0.0 and r[3] <= 1.0, "frac_shuffled_worse in [0,1]")

    pix.free(); act.free()
    _ = tr^; _ = buf^
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

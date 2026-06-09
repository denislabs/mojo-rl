"""LeWM2 teacher-forced eval test (CPU, toy).

Trains a toy world model on the synthetic offline buffer, then runs the
action-awareness eval (expert vs random vs CEM, reusing the shared
planners). Validates the eval pipeline end-to-end:
  - all scores finite,
  - random_min <= random_mean (definitional sanity of the shooter),
  - CEM runs and returns a finite score.
Prints the §10.9 ratios. (On synthetic data, action-awareness ordering
isn't guaranteed for a tiny under-trained model, so we don't assert it —
the real signal is the GPU Pong run.)

Run:  pixi run mojo run -I . tests/experimental/lewm2/test_eval.mojo
"""

from std.memory import alloc
from std.math import isnan, isinf
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.experimental.lewm2.trainer import LeWMTrainer
from mojo_rl.experimental.lewm2.offline_buffer import OfflineWindowBuffer
from mojo_rl.experimental.lewm2.eval import lewm2_action_awareness_eval


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
    print("LeWM2 teacher-forced eval (CPU, toy)")
    print("=" * 70)

    var buf = Buffer(n_traj=8, traj_len=12, seed=321)
    buf.fill_synthetic()
    var tr = Trainer.make(lam=Scalar[DT](0.09), lr=Scalar[DT](1e-3))

    var pix = _a(B * PIX); var act = _a(B * ACTIN)
    var pix_t = TileTensor(pix, row_major[B, PIX]())
    var act_t = TileTensor(act, row_major[B, ACTIN]())

    print("train 150 steps ...")
    for _ in range(150):
        buf.sample_into(pix, act, B)
        _ = tr.train_step(pix_t, act_t)

    # eval on a fresh sampled window
    buf.sample_into(pix, act, B)
    print("eval ...")
    var r = lewm2_action_awareness_eval[
        IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
        ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS,
        PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "cpu",
    ](tr, pix, act, num_random=16, cem_iters=3, cem_samples=32, cem_topk=8)

    var expert = r[0]; var rand_mean = r[1]
    var rand_min = r[2]; var cem = r[3]

    assert_true(not (isnan(expert) or isinf(expert)), "expert finite")
    assert_true(not (isnan(rand_mean) or isinf(rand_mean)), "rand_mean finite")
    assert_true(not (isnan(rand_min) or isinf(rand_min)), "rand_min finite")
    assert_true(not (isnan(cem) or isinf(cem)), "cem finite")
    assert_true(rand_min <= rand_mean + 1e-9,
                "random_min must be <= random_mean")
    assert_true(expert > 0.0 and rand_min > 0.0, "scores positive (MSE)")

    pix.free(); act.free()
    _ = tr^
    _ = buf^
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

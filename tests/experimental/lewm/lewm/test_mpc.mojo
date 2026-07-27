"""LeWM autoregressive MPC eval test (CPU, toy).

Trains a toy world model, then runs the latent-rollout MPC eval
(horizon=2 → NEEDED=4 actions) reusing the shared planners. Validates the
rollout pipeline end-to-end: replicate-start → slide/predict/store loop →
mpc_score, scored for expert vs random vs CEM. Asserts finiteness +
definitional ordering (the action-awareness gate is the GPU Pong run).

Run:  pixi run mojo run -I . tests/experimental/lewm/test_mpc.mojo
"""

from std.memory import alloc
from std.math import isnan, isinf
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.experimental.lewm.trainer import LeWMTrainer
from mojo_rl.experimental.lewm.offline_buffer import OfflineWindowBuffer
from mojo_rl.experimental.lewm.mpc import lewm_mpc_eval


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
comptime MPC_HORIZON = 2          # NEEDED = H + horizon - 1 = 4 == T

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
    return alloc[Scalar[DT]](n).as_unsafe_any_origin()


def main() raises:
    print("=" * 70)
    print("LeWM autoregressive MPC eval (CPU, toy)")
    print("=" * 70)

    var buf = Buffer(n_traj=8, traj_len=12, seed=555)
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
    print("MPC eval ...")
    var r = lewm_mpc_eval[
        IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
        ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS,
        PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, MPC_HORIZON,
        "cpu",
    ](tr, pix_t, act_t, act, num_random=16, cem_iters=3, cem_samples=32,
      cem_topk=8)

    assert_true(not (isnan(r[0]) or isinf(r[0])), "expert finite")
    assert_true(not (isnan(r[2]) or isinf(r[2])), "random_min finite")
    assert_true(not (isnan(r[3]) or isinf(r[3])), "cem finite")
    assert_true(r[2] <= r[1] + 1e-9, "random_min <= random_mean")
    assert_true(r[0] > 0.0 and r[2] > 0.0, "scores positive (MSE)")

    pix.free(); act.free()
    _ = tr^; _ = buf^
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

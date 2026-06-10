"""LeWMTrainer with the CLS-token encoder — trains end to end (toy, CPU).

Instantiates LeWMTrainer with ENC = LeWMEncoderCLS (the trailing encoder
type param) and trains on synthetic windows: the full CLS path (CLS encoder
→ loss graph → Adam) must run and the loss must decrease. Validates the
retrain wiring before the NVIDIA paper-width CLS run; the standard
(mean-pooled) trainer is unchanged (test_trainer covers it).

Run:  pixi run mojo run -I . tests/experimental/lewm2/test_trainer_cls.mojo
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.experimental.lewm2.trainer import LeWMTrainer
from mojo_rl.experimental.lewm2.encoder import LeWMEncoderCLS
from mojo_rl.experimental.lewm2.offline_buffer import OfflineWindowBuffer


comptime IN_CH = 4
comptime IMG = 8
comptime PATCH = 4
comptime N_PATCHES = (IMG // PATCH) * (IMG // PATCH)
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

comptime EncCLS = LeWMEncoderCLS[
    IN_CH, IMG, PATCH, N_PATCHES, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB,
    ENC_PROJ_H, ENC_FF_MULT,
]
comptime Trainer = LeWMTrainer[
    IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
    ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS, PRED_FF,
    DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "cpu", 0, EncCLS,
]
comptime Buffer = OfflineWindowBuffer[IMG_DIM, ACT, T]


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


def main() raises:
    print("=" * 70)
    print("LeWMTrainer + CLS-token encoder (toy, CPU)")
    print("=" * 70)

    var buf = Buffer(n_traj=8, traj_len=12, seed=777)
    buf.fill_synthetic()
    # max_grad_norm=1.0 exercises the graph grad-clip path (Adam.step_graph
    # → clip_grads_graph_cpu) — the fix for the CLS readout's mid-training
    # gradient explosion. Training must still decrease with it on.
    var tr = Trainer.make(
        lam=Scalar[DT](0.09), lr=Scalar[DT](1e-3),
        max_grad_norm=Scalar[DT](1.0),
        weight_decay=Scalar[DT](1e-3),
    )

    var pix = _a(B * PIX); var act = _a(B * ACTIN)
    var pix_t = TileTensor(pix, row_major[B, PIX]())
    var act_t = TileTensor(act, row_major[B, ACTIN]())

    print("train 150 steps (CLS encoder) ...")
    var first: Scalar[DT] = 0.0
    var last: Scalar[DT] = 0.0
    for s in range(150):
        buf.sample_into(pix, act, B)
        var l = tr.train_step(pix_t, act_t)
        if s == 0:
            first = l
        last = l
    print("   loss", first, "→", last)
    assert_true(last < first, "CLS-encoder loss decreases")

    var probes = tr.collapse_probes()
    print("   var_min=", probes[0], " gram_off=", probes[1])

    pix.free(); act.free()
    _ = tr^; _ = buf^
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

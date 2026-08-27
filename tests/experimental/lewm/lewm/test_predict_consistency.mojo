"""LeWMPredictor ↔ loss-graph consistency (CPU, toy).

The autoregressive MPC path runs the predictor on arbitrary latent context
via a SEPARATE LeWMPredictGraph whose params are name-synced from the
trainer. This test proves the sync + the predict graph are correct:

  feed the predictor `latent_ctx = (loss-graph emb's first H frames)` plus
  the SAME actions ⇒ its `pred` must EQUAL the loss graph's `pred` node
  (both compute BiasAdd → ARPredictor → PredProj over identical inputs with
  identical synced params).

If they match (~1e-5), the predictor-from-latents path is correct and ready
for the latent rollout.

Run:  pixi run mojo run -I . tests/experimental/lewm/test_predict_consistency.mojo
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.experimental.lewm.trainer import LeWMTrainer
from mojo_rl.experimental.lewm.predict_graph import LeWMPredictor
from mojo_rl.experimental.lewm.offline_buffer import OfflineWindowBuffer


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
comptime HE = H * EMB
comptime TE = T * EMB

comptime Trainer = LeWMTrainer[
    IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
    ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS, PRED_FF,
    DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "cpu",
]
comptime Predictor = LeWMPredictor[
    EMB, T, ACT, SMOOTHED, AE_MLP, H, PRED_HEADS, PRED_FF, DEPTH,
    PRED_PROJ_H, B, "cpu",
]
comptime Buffer = OfflineWindowBuffer[IMG_DIM, ACT, T]


def _a(n: Int) -> Pointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n).as_unsafe_any_origin()


def main() raises:
    print("=" * 70)
    print("LeWMPredictor ↔ loss-graph consistency (CPU, toy)")
    print("=" * 70)

    var buf = Buffer(n_traj=8, traj_len=12, seed=77)
    buf.fill_synthetic()
    var tr = Trainer.make(lam=Scalar[DT](0.09), lr=Scalar[DT](1e-3))

    var pix = _a(B * PIX); var act = _a(B * ACTIN)
    var pix_t = TileTensor(pix, row_major[B, PIX]())
    var act_t = TileTensor(act, row_major[B, ACTIN]())
    print("train 50 steps ...")
    for _ in range(50):
        buf.sample_into(pix, act, B)
        _ = tr.train_step(pix_t, act_t)

    # one window: run the loss graph forward, capture its pred + emb
    buf.sample_into(pix, act, B)
    var pred_loss = _a(B * HE)
    var tgt_scratch = _a(B * HE)
    tr.forward_into(pix_t, act_t, pred_loss, tgt_scratch)
    var emb_host = _a(B * TE)
    tr.read_node_into["emb"](emb_host, B * TE)

    # latent_ctx = first H frames of each row's emb
    var latent_ctx = _a(B * HE)
    for b in range(B):
        for i in range(HE):
            latent_ctx[b * HE + i] = emb_host[b * TE + i]

    # predictor: sync trained weights by name, run on the same latent + actions
    var pr = Predictor.make()
    pr.sync_from_named(tr.export_named_params())
    var pred_pr = _a(B * HE)
    var lc_t = TileTensor(latent_ctx, row_major[B, HE]())
    var pred_pr_t = TileTensor(pred_pr, row_major[B, HE]())
    pr.forward(lc_t, act_t, pred_pr_t)

    var maxd: Scalar[DT] = 0.0
    for i in range(B * HE):
        var d = (pred_pr[i] - pred_loss[i]).__abs__()
        if d > maxd:
            maxd = d
    print("   max|predictor.pred - lossgraph.pred| =", maxd)
    assert_true(maxd < Scalar[DT](1e-4),
                "predictor-from-latents must reproduce the loss graph pred")

    # ── EVAL-MODE consistency (BN running stats; planning path) ────────
    # Training-mode parity above holds even WITHOUT state sync (both BNs
    # see the same batch ⇒ same batch stats). Eval mode is the real test:
    # the predictor's BN must use the trainer's EMA running stats (warmed
    # well off their 0/1 defaults by the 50 train steps; momentum 0.1 →
    # time constant 10 batches), carried by export_named_params /
    # sync_from_named via the graph for_each_state walk. If state were
    # NOT synced, pr2's BN would normalize with 0/1 defaults and the
    # parity below would fail. This is the closed-loop planner's config.
    tr.set_bn_training(False)
    var pr2 = Predictor.make()
    pr2.sync_from_named(tr.export_named_params())
    pr2.set_bn_training(False)

    tr.forward_into(pix_t, act_t, pred_loss, tgt_scratch)
    tr.read_node_into["emb"](emb_host, B * TE)
    for b in range(B):
        for i in range(HE):
            latent_ctx[b * HE + i] = emb_host[b * TE + i]
    pr2.forward(lc_t, act_t, pred_pr_t)

    var maxd_e: Scalar[DT] = 0.0
    for i in range(B * HE):
        var d = (pred_pr[i] - pred_loss[i]).__abs__()
        if d > maxd_e:
            maxd_e = d
    print("   max|predictor.pred - lossgraph.pred| (EVAL mode) =", maxd_e)
    assert_true(maxd_e < Scalar[DT](1e-5),
                "eval-mode predictor must reproduce the loss graph pred"
                " (BN running stats synced via for_each_state)")

    pix.free(); act.free(); pred_loss.free(); tgt_scratch.free()
    emb_host.free(); latent_ctx.free(); pred_pr.free()
    _ = tr^; _ = pr^; _ = pr2^; _ = buf^
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

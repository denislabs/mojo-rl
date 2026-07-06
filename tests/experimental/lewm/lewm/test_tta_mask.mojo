"""AdaJEPA TTA grad-mask test (Phase 0, CPU).

Validates `_MaskGradV` / `LeWMTrainer.train_step_masked`
(docs/ADAJEPA_LEWM_TTA_PLAN.md §4) on the toy config:

  1. masked step with an EMPTY keep list on the FRESH trainer → every param
     bit-identical (fresh-Adam + wd=0 invariant: zero grad + zero moments
     ⇒ exactly zero update),
  2. masked step keeping the predictor side (`pred_raw.`, `pred.`, `x_pe.`,
     `act_emb.`) → every `emb.` (encoder) param is BIT-IDENTICAL, and at
     least one kept param changed,
  3. positive control: a plain `train_step` DOES change encoder params
     (guards against a false pass where `emb.` never moves anyway).

Ordering matters: the empty-keep check MUST run first. Once a param has
taken a real gradient step, its Adam moments are nonzero and a later
zero-grad step still moves it (momentum decay) — which is also why a TTA
episode must use one constant mask set for its whole lifetime.

Params only — BatchNorm running stats (state) legitimately drift with any
training-mode forward and are exempt from the mask by design.
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.experimental.lewm.trainer import LeWMTrainer, _NamedExportVisitor
from mojo_rl.experimental.lewm.offline_buffer import OfflineWindowBuffer


# toy config (same as test_trainer.mojo)
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


def _export_params(mut tr: Trainer) raises -> Dict[String, List[Scalar[DT]]]:
    """Params only (no BN running-stats state, unlike export_named_params)."""
    var v = _NamedExportVisitor()
    tr.graph.for_each_param["cpu", _NamedExportVisitor](v, tr.ctx)
    return v^.take()


def _diff_counts(
    mut before: Dict[String, List[Scalar[DT]]],
    mut after: Dict[String, List[Scalar[DT]]],
    kept_prefixes: List[String],
) raises -> Tuple[Int, Int, Int, Int]:
    """(kept_total, kept_changed, masked_total, masked_changed); a param
    counts as changed if ANY element differs bitwise."""
    var kept_total = 0
    var kept_changed = 0
    var masked_total = 0
    var masked_changed = 0
    for e in after.items():
        var kept = False
        for p in kept_prefixes:
            if e.key.startswith(p):
                kept = True
                break
        ref old = before[e.key]
        var changed = False
        for i in range(len(e.value)):
            if e.value[i] != old[i]:
                changed = True
                break
        if kept:
            kept_total += 1
            if changed:
                kept_changed += 1
        else:
            masked_total += 1
            if changed:
                masked_changed += 1
    return (kept_total, kept_changed, masked_total, masked_changed)


def main() raises:
    print("=" * 70)
    print("AdaJEPA TTA grad-mask test (Phase 0, CPU)")
    print("=" * 70)

    var buf = Buffer(n_traj=8, traj_len=12, seed=999)
    buf.fill_synthetic()
    # wd=0 + fresh Adam are the mask invariant's preconditions.
    var tr = Trainer.make(lam=Scalar[DT](0.09), lr=Scalar[DT](1e-2))

    var pix = alloc[Scalar[DT]](B * PIX)
    var act = alloc[Scalar[DT]](B * ACTIN)
    var pix_t = TileTensor(pix, row_major[B, PIX]())
    var act_t = TileTensor(act, row_major[B, ACTIN]())
    buf.sample_into(
        pix.as_unsafe_any_origin(), act.as_unsafe_any_origin(), B
    )

    # 1. empty keep list on the FRESH trainer → everything frozen. (Must be
    # first: after any real step, kept params carry nonzero Adam moments and
    # a zero-grad step still moves them via momentum decay.)
    var none = List[String]()
    var before1 = _export_params(tr)
    _ = tr.train_step_masked(pix_t, act_t, none)
    var after1 = _export_params(tr)
    var c1 = _diff_counts(before1, after1, none)
    print("   empty-keep: ", c1[3], "/", c1[2], " params changed")
    assert_true(
        c1[3] == 0, "empty keep list must freeze every param bit-exact"
    )

    # 2. predictor-side subset: encoder frozen bit-exact, predictor moves.
    var keep: List[String] = [
        String("pred_raw."),
        String("pred."),
        String("x_pe."),
        String("act_emb."),
    ]
    var before = _export_params(tr)
    _ = tr.train_step_masked(pix_t, act_t, keep)
    var after = _export_params(tr)
    var c = _diff_counts(before, after, keep)
    print("   kept:   ", c[1], "/", c[0], " params changed")
    print("   masked: ", c[3], "/", c[2], " params changed")
    assert_true(c[0] > 0, "keep prefixes must match params (typo guard)")
    assert_true(c[2] > 0, "mask must cover params (encoder exists)")
    assert_true(c[1] > 0, "masked step must update kept (predictor) params")
    assert_true(
        c[3] == 0, "masked (encoder) params must stay bit-identical"
    )

    # 3. positive control: unmasked step DOES move encoder params.
    var before3 = _export_params(tr)
    _ = tr.train_step(pix_t, act_t)
    var after3 = _export_params(tr)
    var enc_only: List[String] = [String("emb.")]
    var c3 = _diff_counts(before3, after3, enc_only)
    print("   control (train_step): ", c3[1], "/", c3[0], " emb. changed")
    assert_true(
        c3[1] > 0, "plain train_step must update encoder params (control)"
    )

    pix.free()
    act.free()
    _ = tr^
    _ = buf^
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

"""LeWM checkpoint v3 migration test (CPU, toy dims).

Validates the LeWMTrainer save/load migration from the legacy positional
flat text to nn.core.checkpoint v3 binary:

  1. v3 roundtrip: params AND BN running stats bit-identical across a
     fresh trainer instance; `last_load_had_state` True,
  2. Adam-moments hazard + cure: a v3 ckpt saved mid-training carries
     moments; after load, a zero-grad (empty-keep masked) step MOVES params
     (momentum decay) — `reset_opt_moments()` restores the fresh-optimizer
     invariant and the same step becomes a bit-exact no-op
     (docs/ADAJEPA_LEWM_TTA_PLAN.md §4),
  3. legacy flat-text read-compat: an old-format file still loads (params
     only, `last_load_had_state` False).
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.experimental.lewm.trainer import (
    LeWMTrainer,
    _SaveVisitor,
    _NamedExportVisitor,
)
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

comptime V3_PATH: String = "/tmp/lewm_test_ckpt_v3.ckpt"
comptime FLAT_PATH: String = "/tmp/lewm_test_ckpt_flat.txt"


def _dicts_equal(
    mut a: Dict[String, List[Scalar[DT]]],
    mut b: Dict[String, List[Scalar[DT]]],
) raises -> Int:
    """Bit-exact equality; raises on any mismatch, returns entries checked."""
    var n = 0
    for e in a.items():
        ref other = b[e.key]
        for i in range(len(e.value)):
            if e.value[i] != other[i]:
                raise Error("mismatch in '" + e.key + "' at " + String(i))
        n += 1
    return n


def _export_params(mut tr: Trainer) raises -> Dict[String, List[Scalar[DT]]]:
    """Params only — BN running stats legitimately drift with any
    training-mode forward and are not part of the moments invariant."""
    var v = _NamedExportVisitor()
    tr.graph.for_each_param["cpu", _NamedExportVisitor](v, tr.ctx)
    return v^.take()


def _count_moved(
    mut before: Dict[String, List[Scalar[DT]]],
    mut after: Dict[String, List[Scalar[DT]]],
) raises -> Int:
    var moved = 0
    for e in after.items():
        ref old = before[e.key]
        for i in range(len(e.value)):
            if e.value[i] != old[i]:
                moved += 1
                break
    return moved


def main() raises:
    print("=" * 70)
    print("LeWM checkpoint v3 migration test (CPU)")
    print("=" * 70)

    var buf = Buffer(n_traj=8, traj_len=12, seed=999)
    buf.fill_synthetic()
    var pix = alloc[Scalar[DT]](B * PIX)
    var act = alloc[Scalar[DT]](B * ACTIN)
    var pix_t = TileTensor(pix, row_major[B, PIX]())
    var act_t = TileTensor(act, row_major[B, ACTIN]())

    # ── trainer A: a few real steps (moments + BN stats populated), save ─
    var tra = Trainer.make(lam=Scalar[DT](0.09), lr=Scalar[DT](1e-2))
    for _ in range(5):
        buf.sample_into(
            pix.as_unsafe_any_origin(), act.as_unsafe_any_origin(), B
        )
        _ = tra.train_step(pix_t, act_t)
    tra.save_params(V3_PATH)
    var a_all = tra.export_named_params()  # params + BN state

    # ── 1. v3 roundtrip into a fresh instance ───────────────────────────
    var trb = Trainer.make(lam=Scalar[DT](0.09), lr=Scalar[DT](1e-2))
    trb.load_params(V3_PATH)
    assert_true(trb.last_load_had_state, "v3 load must report state")
    var b_all = trb.export_named_params()
    var n1 = _dicts_equal(a_all, b_all)
    print("   v3 roundtrip:", n1, "params+state entries bit-identical")

    # ── 2. moments hazard + reset_opt_moments cure (params only — BN
    # running stats drift with any training-mode forward by design) ──────
    var none = List[String]()
    var before_h = _export_params(trb)
    _ = trb.train_step_masked(pix_t, act_t, none)
    var after_h = _export_params(trb)
    var moved = _count_moved(before_h, after_h)
    print("   loaded-moments hazard: ", moved, " params moved on zero-grad")
    assert_true(
        moved > 0,
        "loaded moments must move params on a zero-grad step (hazard)",
    )
    trb.load_params(V3_PATH)  # restore params (reloads moments too)
    trb.reset_opt_moments()
    var before_c = _export_params(trb)
    _ = trb.train_step_masked(pix_t, act_t, none)
    var after_c = _export_params(trb)
    var moved_c = _count_moved(before_c, after_c)
    print("   after reset_opt_moments: ", moved_c, " params moved")
    assert_true(
        moved_c == 0,
        "reset_opt_moments must make a zero-grad step a bit-exact no-op",
    )

    # ── 3. legacy flat-text read-compat (params only) ───────────────────
    var sv = _SaveVisitor()
    tra.graph.for_each_param["cpu", _SaveVisitor](sv, tra.ctx)
    var s = String()
    s += String(len(sv.vals)) + "\n"
    for i in range(len(sv.vals)):
        s += String(Float64(sv.vals[i])) + "\n"
    with open(FLAT_PATH, "w") as f:
        f.write(s)
    var trc = Trainer.make(lam=Scalar[DT](0.09), lr=Scalar[DT](1e-2))
    trc.load_params(FLAT_PATH)
    assert_true(
        not trc.last_load_had_state, "flat load must report no state"
    )
    var va = _NamedExportVisitor()
    tra.graph.for_each_param["cpu", _NamedExportVisitor](va, tra.ctx)
    var a_params = va^.take()
    var vc = _NamedExportVisitor()
    trc.graph.for_each_param["cpu", _NamedExportVisitor](vc, trc.ctx)
    var c_params = vc^.take()
    var n3 = _dicts_equal(a_params, c_params)
    print("   legacy flat read-compat:", n3, "params bit-identical")

    pix.free()
    act.free()
    _ = tra^
    _ = trb^
    _ = trc^
    _ = buf^
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

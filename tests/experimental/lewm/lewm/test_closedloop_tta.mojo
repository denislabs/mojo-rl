"""LeWM closed-loop MPC with AdaJEPA TTA — wiring test (toy, Apple GPU).

Runs the full closed-loop harness (toy dims, untrained WM, real PushTEnv)
with `tta_enabled=True` and enough cycles for the window buffer to fill
(T=6) and several adapt steps to fire. Asserts:

  1. the loop runs end-to-end with adaptation on and returns finite metrics,
  2. the wm's params + BN state are BIT-IDENTICAL after the call
     (snapshot/restore = fresh-model-per-episode),
  3. a frozen run on the restored wm still works (no poisoned state).

The WM is random so nothing converges — this validates the plumbing
(buffer fill → masked steps → predictor re-sync → restore), not the
method. The real E1 frozen-vs-adapt comparison is the trained-ckpt run
(docs/ADAJEPA_LEWM_TTA_PLAN.md §6).

Run:  pixi run -e apple mojo run -I . tests/experimental/lewm/lewm/test_closedloop_tta.mojo
"""

from max.gpu.host import DeviceContext
from std.math import isnan, isinf
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.experimental.lewm.trainer import LeWMTrainer
from mojo_rl.experimental.lewm.closedloop import run_lewm_closedloop


# toy WM (RGB, tiny) — same as test_closedloop.mojo
comptime IN_CH = 3
comptime IMG = 16
comptime PATCH = 4
comptime HIDDEN = 8
comptime ENC_HEADS = 2
comptime ENC_LAYERS = 2
comptime EMB = 8
comptime ENC_PROJ_H = 16
comptime ENC_FF_MULT = 2
comptime T = 6
comptime ACT = 10           # frameskip(5) × action_dim(2)
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
comptime MPC_HORIZON = 2     # NEEDED = H + horizon - 1 = 4

comptime Trainer = LeWMTrainer[
    IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
    ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS, PRED_FF,
    DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "gpu",
]


def main() raises:
    print("=" * 70)
    print("LeWM closed-loop + AdaJEPA TTA — wiring test (toy, GPU)")
    print("=" * 70)
    var ctx = DeviceContext()
    # TTA preconditions: fresh Adam, wd=0 (make defaults), clip on.
    var wm = Trainer.make(
        lam=Scalar[DT](0.09),
        lr=Scalar[DT](1e-3),
        max_grad_norm=Scalar[DT](1.0),
        ctx=ctx,
    )
    var before = wm.export_named_params()  # params + BN state

    # T=6 → buffer fills at end of cycle 5; adapt fires cycles 5..8.
    print("running closed loop with TTA (untrained WM — wiring only) ...")
    var r = run_lewm_closedloop[
        IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
        ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS,
        PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, MPC_HORIZON,
        "gpu", 0, 2, 16,   # PRED_DIM_HEAD=0, ACT_DIM=2, VIZ=16
    ](
        wm,
        n_cycles=9,
        scale_x=100.0, scale_y=100.0,
        cem_iters=3, cem_samples=16, cem_topk=4, init_std=0.2,
        ctx=ctx,
        verbose=True,
        tta_enabled=True,
        tta_steps=1,
    )
    print("   success_rate=", r[0], " mean_cov=", r[1])
    assert_true(not (isnan(r[1]) or isinf(r[1])), "mean_cov finite")
    assert_true(r[0] >= 0.0 and r[0] <= 1.0, "success_rate in [0,1]")
    assert_true(r[1] >= 0.0 and r[1] <= 1.0, "mean_cov in [0,1]")

    # 2. fresh-model-per-episode: params + BN state restored bit-exact.
    var after = wm.export_named_params()
    var n_checked = 0
    for e in after.items():
        ref old = before[e.key]
        for i in range(len(e.value)):
            if e.value[i] != old[i]:
                raise Error(
                    "param/state '" + e.key + "' not restored after TTA run"
                )
        n_checked += 1
    print("   restore check:", n_checked, "params+state entries bit-identical")
    assert_true(n_checked > 0, "restore check must cover entries")

    # 3. restored wm still drives a frozen run.
    var r2 = run_lewm_closedloop[
        IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
        ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS,
        PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, MPC_HORIZON,
        "gpu", 0, 2, 16,
    ](
        wm,
        n_cycles=2,
        scale_x=100.0, scale_y=100.0,
        cem_iters=2, cem_samples=8, cem_topk=2, init_std=0.2,
        ctx=ctx,
        verbose=False,
    )
    print("   frozen-after-restore mean_cov=", r2[1])
    assert_true(not (isnan(r2[1]) or isinf(r2[1])), "post-restore run finite")

    _ = wm^
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

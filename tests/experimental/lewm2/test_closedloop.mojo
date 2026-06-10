"""LeWM2 closed-loop MPC harness — wiring test (toy, Apple GPU).

Runs the FULL closed-loop control loop at toy scale on the real PushTEnv
(rendered at 16²) with a tiny UNTRAINED world model: predictor sync →
render→encode start latent → goal-image latent → ContinuousCEM plan →
denormalize (delta) → env.step → coverage → viz strip. The WM is random so
it won't solve the task; this asserts the harness RUNS end to end and
returns finite metrics — the real solve is the NVIDIA paper-WM run.

Run:  pixi run -e apple mojo run -I . tests/experimental/lewm2/test_closedloop.mojo
"""

from std.gpu.host import DeviceContext
from std.math import isnan, isinf
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.experimental.lewm2.trainer import LeWMTrainer
from mojo_rl.experimental.lewm2.encoder import LeWMEncoderCLS
from mojo_rl.experimental.lewm2.closedloop import run_lewm2_closedloop


# toy WM (RGB, tiny) — PushT renders 3 channels
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

# CLS-encoder variant — compile-checks the EncCLS path through both
# LeWMTrainer and run_lewm2_closedloop (Gate C uses exactly this wiring).
comptime N_PATCHES = (IMG // PATCH) * (IMG // PATCH)
comptime EncCLS = LeWMEncoderCLS[
    IN_CH, IMG, PATCH, N_PATCHES, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB,
    ENC_PROJ_H, ENC_FF_MULT,
]
comptime TrainerCLS = LeWMTrainer[
    IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
    ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS, PRED_FF,
    DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "gpu", 0, EncCLS,
]


def main() raises:
    print("=" * 70)
    print("LeWM2 closed-loop MPC harness — wiring test (toy, GPU)")
    print("=" * 70)
    var ctx = DeviceContext()
    var wm = Trainer.make(lam=Scalar[DT](0.09), lr=Scalar[DT](1e-3), ctx=ctx)

    print("running closed loop (untrained WM — wiring only) ...")
    var r = run_lewm2_closedloop[
        IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
        ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS,
        PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, MPC_HORIZON,
        "gpu", 0, 2, 16,   # PRED_DIM_HEAD=0, ACT_DIM=2, VIZ=16
    ](
        wm,
        n_cycles=4,
        scale_x=142.0, scale_y=148.0,
        cem_iters=3, cem_samples=16, cem_topk=4, init_std=0.2,
        viz_path=String("/tmp/lewm2_closedloop_toy.ppm"),
        ctx=ctx,
        verbose=True,
    )
    print("   success_rate=", r[0], " mean_cov=", r[1])
    assert_true(not (isnan(r[1]) or isinf(r[1])), "mean_cov finite")
    assert_true(r[0] >= 0.0 and r[0] <= 1.0, "success_rate in [0,1]")
    assert_true(r[1] >= 0.0 and r[1] <= 1.0, "mean_cov in [0,1]")

    print("running closed loop with CLS encoder (wiring only) ...")
    var wm_cls = TrainerCLS.make(
        lam=Scalar[DT](0.09), lr=Scalar[DT](1e-3), ctx=ctx
    )
    var rc = run_lewm2_closedloop[
        IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
        ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS,
        PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, MPC_HORIZON,
        "gpu", 0, 2, 16, EncCLS,   # trailing ENC = CLS encoder
    ](
        wm_cls,
        n_cycles=4,
        scale_x=142.0, scale_y=148.0,
        cem_iters=3, cem_samples=16, cem_topk=4, init_std=0.2,
        ctx=ctx,
        verbose=False,
    )
    print("   CLS success_rate=", rc[0], " mean_cov=", rc[1])
    assert_true(not (isnan(rc[1]) or isinf(rc[1])), "CLS mean_cov finite")
    assert_true(rc[1] >= 0.0 and rc[1] <= 1.0, "CLS mean_cov in [0,1]")

    _ = wm^
    _ = wm_cls^
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

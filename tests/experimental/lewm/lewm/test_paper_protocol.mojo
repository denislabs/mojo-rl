"""LeWM2 paper-protocol eval harness — wiring test (toy, Apple GPU).

Runs the FULL paper-protocol loop at toy scale with a tiny UNTRAINED world
model: PushTEnv.set_state round-trip → start/goal state pairs fabricated by
rolling the env → goal-state encode → CEM plan → execute future blocks →
swm-style success/pos_diff metrics. The WM is random so success is not
expected; this asserts the harness RUNS end to end with finite metrics —
the real eval is the NVIDIA paper-WM run.

Run:  pixi run -e apple mojo run -I . tests/experimental/lewm/test_paper_protocol.mojo
"""

from std.memory import alloc
from std.math import isnan, isinf
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.experimental.lewm.trainer import LeWMTrainer
from mojo_rl.experimental.lewm.encoder import LeWMEncoderCLS
from mojo_rl.experimental.lewm.paper_protocol import run_lewm_paper_protocol
from mojo_rl.envs.pusht import PushTEnv, PushTAction


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

# CLS-encoder variant — compile-checks the EncCLS path through
# run_lewm_paper_protocol (the NVIDIA example uses exactly this wiring).
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
    print("LeWM2 paper-protocol harness — wiring test (toy, GPU)")
    print("=" * 70)

    # ── PushTEnv.set_state round-trip ───────────────────────────────────
    var e = PushTEnv[DT](seed=3)
    _ = e.reset()
    _ = e.set_state(
        Scalar[DT](123.0), Scalar[DT](234.0),
        Scalar[DT](345.0), Scalar[DT](222.0), Scalar[DT](1.25),
    )
    var ap = e.agent_pos()
    var bp = e.block_pose()
    assert_true((Float64(ap[0]) - 123.0).__abs__() < 1e-4, "agent x set")
    assert_true((Float64(ap[1]) - 234.0).__abs__() < 1e-4, "agent y set")
    assert_true((Float64(bp[0]) - 345.0).__abs__() < 1e-4, "block x set")
    assert_true((Float64(bp[1]) - 222.0).__abs__() < 1e-4, "block y set")
    assert_true((Float64(bp[2]) - 1.25).__abs__() < 1e-4, "block angle set")
    assert_true(not e.is_done(), "set_state clears done")
    print("   set_state round-trip OK")

    # ── fabricate (start, goal) pairs by rolling the env ───────────────
    var starts = alloc[Scalar[DT]](B * 5)
    var goals = alloc[Scalar[DT]](B * 5)
    for b in range(B):
        var env = PushTEnv[DT](seed=UInt64(10 + b))
        _ = env.reset()
        var ap0 = env.agent_pos()
        var bp0 = env.block_pose()
        starts[b * 5 + 0] = ap0[0]; starts[b * 5 + 1] = ap0[1]
        starts[b * 5 + 2] = bp0[0]; starts[b * 5 + 3] = bp0[1]
        starts[b * 5 + 4] = bp0[2]
        for s in range(10):
            var apc = env.agent_pos()
            _ = env.step(PushTAction[DT](
                apc[0] + Scalar[DT](8.0 - Float64(s)),
                apc[1] + Scalar[DT](5.0),
            ))
        var ap1 = env.agent_pos()
        var bp1 = env.block_pose()
        goals[b * 5 + 0] = ap1[0]; goals[b * 5 + 1] = ap1[1]
        goals[b * 5 + 2] = bp1[0]; goals[b * 5 + 3] = bp1[1]
        goals[b * 5 + 4] = bp1[2]

    # ── full protocol loop (untrained WM — wiring only) ────────────────
    var ctx = DeviceContext()
    var wm = Trainer.make(lam=Scalar[DT](0.09), lr=Scalar[DT](1e-3), ctx=ctx)
    print("running paper protocol (untrained WM — wiring only) ...")
    var r = run_lewm_paper_protocol[
        IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
        ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS,
        PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, MPC_HORIZON,
        "gpu", 0, 2, 16,   # PRED_DIM_HEAD=0, ACT_DIM=2, VIZ=16
    ](
        wm, starts, goals,
        eval_budget=20,    # 2 plans × (horizon 2 × frameskip 5)
        scale_x=100.0, scale_y=100.0,
        cem_iters=3, cem_samples=16, cem_topk=4, init_std=0.2,
        viz_path=String("/tmp/lewm_protocol_toy.ppm"),
        ctx=ctx,
        verbose=True,
    )
    print("   success_rate=", r[0], " mean_pos_diff=", r[1])
    assert_true(not (isnan(r[1]) or isinf(r[1])), "pos_diff finite")
    assert_true(r[0] >= 0.0 and r[0] <= 1.0, "success_rate in [0,1]")
    assert_true(r[1] >= 0.0, "pos_diff non-negative")

    print("running paper protocol with CLS encoder (wiring only) ...")
    var wm_cls = TrainerCLS.make(
        lam=Scalar[DT](0.09), lr=Scalar[DT](1e-3), ctx=ctx
    )
    var rc = run_lewm_paper_protocol[
        IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
        ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS,
        PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, MPC_HORIZON,
        "gpu", 0, 2, 16, EncCLS,   # trailing ENC = CLS encoder
    ](
        wm_cls, starts, goals,
        eval_budget=10,
        scale_x=100.0, scale_y=100.0,
        cem_iters=2, cem_samples=8, cem_topk=2, init_std=0.2,
        ctx=ctx,
        verbose=False,
    )
    print("   CLS success_rate=", rc[0], " mean_pos_diff=", rc[1])
    assert_true(not (isnan(rc[1]) or isinf(rc[1])), "CLS pos_diff finite")

    starts.free(); goals.free()
    _ = wm^
    _ = wm_cls^
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

"""LeWM paper protocol + AdaJEPA TTA — wiring test (toy, Apple GPU).

Runs the full paper-protocol loop (CLS encoder, MPC_HORIZON=1 — the
AdaJEPA plan-execute-adapt-replan shape the E1 driver uses) with
`tta_enabled=True` on a tiny untrained WM. Budget 50 at horizon 1 gives
10 replans; the T=6 window buffer fills after 6, so several adapt steps
fire (visible as `tta:` lines). Asserts:

  1. the loop runs end-to-end with adaptation on and returns finite
     metrics,
  2. the wm's params + BN state are BIT-IDENTICAL after the call
     (snapshot/restore = fresh-model-per-episode).

The WM is random so success is not expected — this validates the
plumbing (per-block pushes, adapt, predictor re-sync, goal re-encode,
restore). The real E1 is the NVIDIA recipe-WM run
(examples/lewm/lewm_pusht_paper_protocol_tta_gpu.mojo).

Run:  pixi run -e apple mojo run -I . tests/experimental/lewm/lewm/test_paper_protocol_tta.mojo
"""

from std.memory import alloc
from std.math import isnan, isinf
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.experimental.lewm.trainer import LeWMTrainer
from mojo_rl.experimental.lewm.encoder import LeWMEncoderCLS
from mojo_rl.experimental.lewm.paper_protocol import run_lewm_paper_protocol
from mojo_rl.envs.pusht import PushTEnv, PushTAction


# toy WM (RGB, tiny) — same as test_paper_protocol.mojo
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
comptime MPC_HORIZON = 2     # plan 2 chunks; run 1 executes both (j>0
                             # frame-render push path), run 2 executes 1
                             # (the E1 lookahead-with-receding-horizon shape)

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
    print("LeWM paper protocol + AdaJEPA TTA — wiring test (toy, GPU)")
    print("=" * 70)

    # fabricate (start, goal) pairs by rolling the env
    var starts = alloc[Scalar[DT]](B * 5).as_unsafe_any_origin()
    var goals = alloc[Scalar[DT]](B * 5).as_unsafe_any_origin()
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

    var ctx = DeviceContext()
    # TTA preconditions: fresh Adam, wd=0 (make defaults), clip on.
    var wm = TrainerCLS.make(
        lam=Scalar[DT](0.09),
        lr=Scalar[DT](1e-3),
        max_grad_norm=Scalar[DT](1.0),
        ctx=ctx,
    )
    var before = wm.export_named_params()  # params + BN state

    print("running TTA, execute ALL blocks (j>0 render-push path) ...")
    var r = run_lewm_paper_protocol[
        IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
        ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS,
        PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, MPC_HORIZON,
        "gpu", 0, 2, 16, EncCLS,   # PRED_DIM_HEAD=0, ACT_DIM=2, VIZ=16
    ](
        wm, starts, goals,
        eval_budget=80,    # 8 plans × 2 blocks; buffer full after 3 plans
        scale_x=100.0, scale_y=100.0,
        cem_iters=3, cem_samples=16, cem_topk=4, init_std=0.2,
        ctx=ctx,
        verbose=True,
        tta_enabled=True,
        tta_steps=1,
    )
    print("   success_rate=", r[0], " mean_pos_diff=", r[1])
    assert_true(not (isnan(r[1]) or isinf(r[1])), "pos_diff finite")
    assert_true(r[0] >= 0.0 and r[0] <= 1.0, "success_rate in [0,1]")
    assert_true(r[1] >= 0.0, "pos_diff non-negative")

    print("running TTA, plan 2 / execute 1 (E1 receding-horizon shape) ...")
    var r2 = run_lewm_paper_protocol[
        IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
        ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS,
        PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, MPC_HORIZON,
        "gpu", 0, 2, 16, EncCLS,
    ](
        wm, starts, goals,
        eval_budget=50,    # 10 replans × 1 block; adapt fires from #7
        scale_x=100.0, scale_y=100.0,
        cem_iters=3, cem_samples=16, cem_topk=4, init_std=0.2,
        ctx=ctx,
        verbose=True,
        execute_blocks=1,
        tta_enabled=True,
        tta_steps=1,
    )
    print("   exec1 success_rate=", r2[0], " mean_pos_diff=", r2[1])
    assert_true(not (isnan(r2[1]) or isinf(r2[1])), "exec1 pos_diff finite")

    # fresh-model-per-episode: params + BN state restored bit-exact
    var after = wm.export_named_params()
    var n_checked = 0
    var n_bad = 0
    for e in after.items():
        var old = before[e.key].copy()
        var mx: Float64 = 0.0
        var nd = 0
        for i in range(len(e.value)):
            if e.value[i] != old[i]:
                nd += 1
                var d = Float64(e.value[i] - old[i])
                if d < 0.0:
                    d = -d
                if d > mx:
                    mx = d
        if nd > 0:
            n_bad += 1
            print("   MISMATCH", e.key, ": ", nd, "/", len(e.value),
                  " elems, max |Δ| =", mx)
        n_checked += 1
    print("   restore check:", n_checked, "entries,", n_bad, "mismatching")
    assert_true(n_bad == 0, "params+state must restore bit-identical")
    assert_true(n_checked > 0, "restore check must cover entries")

    starts.free(); goals.free()
    _ = wm^
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)

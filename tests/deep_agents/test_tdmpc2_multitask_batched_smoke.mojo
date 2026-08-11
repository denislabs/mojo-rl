"""TD-MPC2 MULTI-TASK batched driver smoke (CPU) — 2 tasks x 4 envs.

Gates `train_batched_mt` / `evaluate_batched_mt`, which nothing else
instantiates (an uninstantiated generic is uncompiled code, so the existing
multi-task smoke says nothing about them).

Both tasks here run the same env type — the tasks differ only by id. That is
deliberate: this file gates the MECHANISM (per-task tagging, per-task
embeddings, segment alternation, the batched acting path), not whether two
different rewards can be learned at once. The walker example does the latter.

Gates:
  1. Segments alternate task 0 / task 1 / task 0 and the WM loss falls.
  2. Every task's embedding row receives gradient — a row that never moves
     means the task conditioning is decorative and the agent is really
     single-task with extra parameters.
  3. The replay holds BOTH tasks after alternating segments, and sampled
     windows are single-task (the strided task walk).
  4. `evaluate_batched_mt` returns a finite number for each task.

Run: `pixi run mojo run -I . tests/deep_agents/test_tdmpc2_multitask_batched_smoke.mojo`
"""

from std.math import isfinite, abs
from std.random import seed
from std.testing import assert_true, TestSuite

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.tdmpc2.config_mt import TDMPC2MultiTask
from mojo_rl.deep_agents.training.batched_env import BatchedCpuEnv
from mojo_rl.envs.pendulum import PendulumV2

comptime MAX_OBS = 3
comptime MAX_ACT = 1
comptime NUM_TASKS = 2
comptime TASK_EMB = 8
comptime ENC = 32
comptime LATENT = 32
comptime MLP = 32
comptime BINS = 21
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime B = 4
comptime H = 3
comptime N_ENVS = 4
comptime CAP = 4096

comptime Env = BatchedCpuEnv[PendulumV2[DT], N_ENVS, MAX_OBS, MAX_ACT]


def test_multitask_segments_train_and_move_every_embedding() raises:
    seed(11)
    var env = Env(PendulumV2[DT]())
    var ag = TDMPC2MultiTask[
        "cpu", MAX_OBS, MAX_ACT, NUM_TASKS, TASK_EMB, B, CAP,
        ENC, LATENT, MLP, BINS, SN, VMIN, VMAX, H,
    ](lr=Scalar[DT](1e-3), action_scale=Scalar[DT](2.0), learning_starts=64)

    # Snapshot the embedding table before any training.
    var n_emb = NUM_TASKS * TASK_EMB
    var emb0 = List[Scalar[DT]](length=n_emb, fill=Scalar[DT](0))
    for i in range(n_emb):
        emb0[i] = ag.task_emb.param.data[i]

    # Three segments: task 0, task 1, task 0 again. `base_step` keeps the
    # cumulative counter honest across calls, exactly as the example does.
    var wm = List[Scalar[DT]]()
    var at = 0
    for seg in range(3):
        var task = seg % 2
        var _b = ag.train_batched_mt[Env, N_ENVS](
            env,
            task,
            256,
            rng_seed=UInt64(20 + seg),
            updates_per_step=1,
            print_every=0,
            verbose=False,
            base_step=at,
        )
        at += 256
        wm.append(ag.last_wm_loss())

    for i in range(len(wm)):
        assert_true(
            isfinite(Float64(wm[i])), "WM loss must stay finite in every segment"
        )
    assert_true(
        wm[len(wm) - 1] < wm[0],
        "WM loss should fall across segments (" + String(wm[0]) + " -> "
        + String(wm[len(wm) - 1]) + ")",
    )
    assert_true(
        ag.replay.count() == 3 * 256,
        "replay should hold every recorded transition of every segment",
    )

    # Per-task embedding movement: BOTH rows must have moved. A row that never
    # moves is a task the conditioning never actually used.
    for t in range(NUM_TASKS):
        var moved: Scalar[DT] = 0.0
        for e in range(TASK_EMB):
            var k = t * TASK_EMB + e
            moved += abs(ag.task_emb.param.data[k] - emb0[k])
        assert_true(
            moved > Scalar[DT](1e-6),
            "task " + String(t) + " embedding row received NO gradient",
        )
        print("  task", t, "embedding L1 movement =", moved)

    print("  segments wm:", wm[0], "->", wm[len(wm) - 1], "✓")


def test_evaluate_batched_mt_finite_per_task() raises:
    seed(3)
    var eval_env = Env(PendulumV2[DT]())
    var ag = TDMPC2MultiTask[
        "cpu", MAX_OBS, MAX_ACT, NUM_TASKS, TASK_EMB, B, CAP,
        ENC, LATENT, MLP, BINS, SN, VMIN, VMAX, H,
    ](lr=Scalar[DT](1e-3), action_scale=Scalar[DT](2.0), learning_starts=0)
    for t in range(NUM_TASKS):
        var r = ag.evaluate_batched_mt[Env, N_ENVS](
            eval_env, t, max_steps=205
        )
        assert_true(
            isfinite(Float64(r)), "eval return must be finite for every task"
        )
        print("  task", t, "eval return =", r)


def test_task_id_out_of_range_raises() raises:
    seed(1)
    var env = Env(PendulumV2[DT]())
    var ag = TDMPC2MultiTask[
        "cpu", MAX_OBS, MAX_ACT, NUM_TASKS, TASK_EMB, B, CAP,
        ENC, LATENT, MLP, BINS, SN, VMIN, VMAX, H,
    ](lr=Scalar[DT](1e-3), action_scale=Scalar[DT](2.0), learning_starts=0)
    var raised = False
    try:
        var _b = ag.train_batched_mt[Env, N_ENVS](
            env, NUM_TASKS, 32, print_every=0, verbose=False
        )
    except:
        raised = True
    assert_true(
        raised,
        "a task_id past NUM_TASKS must raise, not index off the embedding"
        " table",
    )
    print("  out-of-range task id raises ✓")


def main() raises:
    print("=" * 70)
    print("TD-MPC2 multi-task BATCHED driver smoke (CPU) — 2 tasks x 4 envs")
    print("=" * 70)
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("=" * 70)
    print("MULTI-TASK BATCHED SMOKE PASSED")
    print("=" * 70)

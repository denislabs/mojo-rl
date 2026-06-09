"""TD-MPC2 multi-task smoke (CPU) — item C, §14.3.

Builds the multi-task agent via the `TDMPC2MultiTask[...]` preset (MAX_OBS=4,
MAX_ACT=1, NUM_TASKS=2, TASK_EMB=8) and trains on synthetic two-task data. Two
tasks alternate; task 0's obs live in [0:3] with [3]=0 (Pendulum-shaped), task 1
uses all 4 dims (InvertedPendulum-shaped). Checks the WM loss is finite and
decreases (the BPTT + task-conditioning learning stack is wired), and that the
embedding table moved from its init (gradient flowed into it).

Run: `pixi run mojo run -I . tests/deep_agents2/test_tdmpc2_multitask_smoke.mojo`
"""

from std.memory import alloc
from std.random import random_float64, seed
from std.math import isfinite, abs
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.tdmpc2.config_mt import TDMPC2MultiTask

comptime MAX_OBS = 4
comptime MAX_ACT = 1
comptime NUM_TASKS = 2
comptime TASK_EMB = 8
comptime B = 4
comptime CAP = 4096
comptime ENC = 64
comptime LATENT = 64
comptime MLP = 64
comptime BINS = 51


def main() raises:
    print("=" * 70)
    print("TD-MPC2 multi-task smoke (CPU) — Pendulum-shaped + InvPend-shaped")
    print("=" * 70)
    seed(11)

    var ag = TDMPC2MultiTask[
        "cpu", MAX_OBS, MAX_ACT, NUM_TASKS, TASK_EMB, B, CAP,
        ENC, LATENT, MLP, BINS,
    ](lr=Scalar[DT](1e-3), learning_starts=64)

    # snapshot the embedding table to detect movement after training.
    var emb0 = alloc[Scalar[DT]](NUM_TASKS * TASK_EMB)
    for i in range(NUM_TASKS * TASK_EMB):
        emb0[i] = ag.task_emb.param[i]

    var obsbuf = alloc[Scalar[DT]](MAX_OBS)
    var actbuf = alloc[Scalar[DT]](MAX_ACT)

    comptime TOTAL = 400
    comptime LEARN_START = 64
    comptime TRAIN_EVERY = 6
    var first_wm: Scalar[DT] = 0.0
    var last_wm: Scalar[DT] = 0.0
    var n_train = 0

    for step in range(TOTAL):
        var task = step % NUM_TASKS
        ag.set_task(task)
        # synthetic obs: task 0 fills [0:3], leaves [3]=0; task 1 fills all 4.
        var ndim = 3 if task == 0 else 4
        for i in range(MAX_OBS):
            obsbuf[i] = Scalar[DT](0.0)
        for i in range(ndim):
            obsbuf[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
        if step < LEARN_START:
            actbuf[0] = Scalar[DT](random_float64() * 2.0 - 1.0)
        else:
            ag.select_action(obsbuf, actbuf, explore=True)
        var r = Scalar[DT](random_float64() - 0.5)
        ag.record(obsbuf, actbuf, r, Scalar[DT](0.0))
        if step >= LEARN_START and step % TRAIN_EVERY == 0:
            if ag.train_step():
                var wm = ag.last_wm_loss()
                assert_true(isfinite(wm), "WM finite")
                if n_train == 0:
                    first_wm = wm
                last_wm = wm
                n_train += 1

    # embedding moved?
    var moved: Scalar[DT] = 0.0
    for i in range(NUM_TASKS * TASK_EMB):
        moved += abs(ag.task_emb.param[i] - emb0[i])

    print("  trained", n_train, "steps; WM:", first_wm, "->", last_wm)
    print("  embedding L1 movement:", moved)
    assert_true(n_train > 0, "should have trained")
    assert_true(last_wm < first_wm, "WM loss should decrease")
    assert_true(moved > Scalar[DT](1e-6), "embedding table must receive gradient")
    print("=" * 70)
    print("MULTI-TASK SMOKE PASSED — task-conditioned WM trains + embedding learns")
    print("=" * 70)
    obsbuf.free(); actbuf.free(); emb0.free()

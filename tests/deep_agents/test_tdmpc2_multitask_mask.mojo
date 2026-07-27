"""TD-MPC2 multi-task action-masking test (synthetic, CPU) — item C, §14.3.

The lighthouse (Pendulum + InvertedPendulum) is MAX_ACT=1, so its action mask is
≡1 — the masking machinery is a no-op there. This test exercises it directly:
MAX_ACT=2, NUM_TASKS=2, with task 1 masking action dim 1 (mask = [1, 0]). It
asserts that acting for task 1 always zeroes dim 1 (both explore + greedy) while
dim 0 stays active, and that task 0 (mask [1,1]) leaves both dims active. This is
the per-task action masking applied at acting/record time (env wrapper +
`agent_mt.select_action`); in-graph masking is deferred-experimental (see
`policy_step_mt.mojo`).

Run: `pixi run mojo run -I . tests/deep_agents/test_tdmpc2_multitask_mask.mojo`
"""

from std.random import random_float64, seed
from std.math import abs, max
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.tdmpc2.config_mt import TDMPC2MultiTask

comptime MAX_OBS = 4
comptime MAX_ACT = 2
comptime NUM_TASKS = 2
comptime TASK_EMB = 8
comptime B = 4
comptime CAP = 2048


def main() raises:
    print("=" * 70)
    print("TD-MPC2 multi-task action-mask test (synthetic) — task 1 masks dim 1")
    print("=" * 70)
    seed(3)
    var ag = TDMPC2MultiTask[
        "cpu", MAX_OBS, MAX_ACT, NUM_TASKS, TASK_EMB, B, CAP, 32, 32, 32, 51,
    ](lr=Scalar[DT](1e-3))

    # task 0: both dims active; task 1: dim 1 masked off.
    var m0 = List[Scalar[DT]](length=MAX_ACT, fill=Scalar[DT](0))
    m0[0] = Scalar[DT](1.0); m0[1] = Scalar[DT](1.0)
    var m1 = List[Scalar[DT]](length=MAX_ACT, fill=Scalar[DT](0))
    m1[0] = Scalar[DT](1.0); m1[1] = Scalar[DT](0.0)
    ag.set_action_mask(0, m0)
    ag.set_action_mask(1, m1)

    var obsbuf = List[Scalar[DT]](length=MAX_OBS, fill=Scalar[DT](0))
    var actbuf = List[Scalar[DT]](length=MAX_ACT, fill=Scalar[DT](0))

    var t1_dim1_max: Scalar[DT] = 0.0
    var t1_dim0_abs_sum: Scalar[DT] = 0.0
    var t0_dim1_abs_sum: Scalar[DT] = 0.0

    for it in range(40):
        for i in range(MAX_OBS):
            obsbuf[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
        # task 1 (dim 1 masked): explore + greedy.
        ag.set_task(1)
        ag.select_action(obsbuf, actbuf, explore=True)
        t1_dim1_max = max(t1_dim1_max, abs(actbuf[1]))
        t1_dim0_abs_sum += abs(actbuf[0])
        ag.select_greedy_action(obsbuf, actbuf)
        t1_dim1_max = max(t1_dim1_max, abs(actbuf[1]))
        # task 0 (both active).
        ag.set_task(0)
        ag.select_action(obsbuf, actbuf, explore=True)
        t0_dim1_abs_sum += abs(actbuf[1])

    print("  task1 dim1 max|a| =", t1_dim1_max, " (must be 0)")
    print("  task1 dim0 Σ|a| =", t1_dim0_abs_sum, " (must be > 0)")
    print("  task0 dim1 Σ|a| =", t0_dim1_abs_sum, " (must be > 0)")
    assert_true(t1_dim1_max == Scalar[DT](0.0), "task-1 masked dim 1 must be 0")
    assert_true(t1_dim0_abs_sum > Scalar[DT](0.0), "task-1 active dim 0 must vary")
    assert_true(t0_dim1_abs_sum > Scalar[DT](0.0), "task-0 dim 1 must be active")
    print("=" * 70)
    print("ACTION-MASK PASSED — per-task masking zeroes unused dims at acting")
    print("=" * 70)

"""Per-task policy-loss scale — a DEVIATION from TD-MPC2, off by default.

TD-MPC2 normalizes the policy loss by ONE running scale across every task
(`references/tdmpc2-main/tdmpc2/tdmpc2.py:34` — a single `RunningScale`, even
for MT80). `set_per_task_pi_scale(True)` gives each task its own, so each
task's policy gradient becomes invariant to its OWN Q spread rather than to the
mixed-batch spread.

Motivation (`docs/TDMPC2_MULTITASK_VALIDATION.md`): on walker stand+walk+run the
shared scale was set by the two solved tasks (Q ~98) while run sat at ~16 and
collapsed to the standing floor — with a MATCHED run-weighted gradient budget
(104k vs 99k), so it was not a data problem.

The gates, in the order that matters:

  1. OFF is BIT-IDENTICAL run to run, and leaves the per-task table untouched.
     This is the one that protects every result measured before 2026-08-13 —
     a deviation shipped ON by default, or leaking when off, would silently
     invalidate the reference comparison.
  2. ON actually changes something. A flag that is wired to nothing passes
     gate 1 perfectly.
  3. ON is finite and stable on GPU, where the reweighted backward seed is
     built host-side and uploaded rather than filled by a kernel.

⚠ The batch below deliberately gives the three tasks reward magnitudes an order
of magnitude apart (10 / 1 / 0.15). Equal-magnitude tasks would make per-task
and shared scaling nearly identical and gate 2 would pass on noise.

Run: `pixi run mojo run -I . tests/deep_agents/test_tdmpc2_per_task_pi_scale.mojo`
     `pixi run -e apple mojo run -I . tests/deep_agents/test_tdmpc2_per_task_pi_scale.mojo`
"""

from std.math import abs, isfinite
from std.random import seed
from std.testing import assert_true, assert_almost_equal, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.tdmpc2.agent_mt import TDMPC2MultiTaskAgent

comptime MAX_OBS = 3
comptime ENC = 32
comptime MAX_ACT = 1
comptime LATENT = 32
comptime MLP = 32
comptime BINS = 21
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime B = 8
comptime H = 3
comptime CAP = 2048
comptime NUM_TASKS = 3
comptime TASK_EMB = 8

comptime AgCPU = TDMPC2MultiTaskAgent[
    "cpu", MAX_OBS, ENC, MAX_ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H,
    CAP, NUM_TASKS, TASK_EMB,
]
comptime AgGPU = TDMPC2MultiTaskAgent[
    "gpu", MAX_OBS, ENC, MAX_ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H,
    CAP, NUM_TASKS, TASK_EMB,
]


comptime N_TXN = 600


def _txn(
    i: Int, mut obs: List[Scalar[DT]], mut act: List[Scalar[DT]]
) -> Scalar[DT]:
    """Transition `i`: fills obs/act, returns the reward. Rewards differ by
    ~70x across tasks — equal magnitudes would make per-task and shared
    scaling nearly identical and the ON/OFF gate would pass on noise."""
    var t = i % NUM_TASKS
    for d in range(MAX_OBS):
        obs[d] = Scalar[DT](0.1 * Float64((i * 7 + d) % 13) - 0.6)
    act[0] = Scalar[DT](0.2 * Float64(i % 5) - 0.4)
    return Scalar[DT](10.0) if t == 0 else (
        Scalar[DT](1.0) if t == 1 else Scalar[DT](0.15)
    )


def _run_cpu(enable: Bool) raises -> List[Scalar[DT]]:
    seed(11)
    var ag = AgCPU.make(
        lr=Scalar[DT](1e-3), action_scale=Scalar[DT](2.0), learning_starts=0
    )
    ag.set_per_task_pi_scale(enable)
    var obs = List[Scalar[DT]](length=MAX_OBS, fill=Scalar[DT](0))
    var act = List[Scalar[DT]](length=MAX_ACT, fill=Scalar[DT](0))
    for i in range(N_TXN):
        ag.set_task(i % NUM_TASKS)
        var r = _txn(i, obs, act)
        ag.record(obs, act, r, Scalar[DT](0.0))
    for _ in range(40):
        _ = ag.train_step()
    var o = List[Scalar[DT]]()
    o.append(ag.last_pi_loss())
    o.append(ag.pi_scale())
    for t in range(NUM_TASKS):        # [2..2+NUM_TASKS) = the per-task table
        o.append(ag.task_pi_scale(t))
    return o^


def test_off_is_deterministic_and_leaves_the_table_untouched() raises:
    """The gate that protects the reference comparison."""
    var a = _run_cpu(False)
    var b = _run_cpu(False)
    print("  OFF  pi =", a[0], " shared =", a[1], " task2 =", a[2])
    assert_almost_equal(
        a[0], b[0], atol=1e-9,
        msg="OFF is not reproducible — the A/B below would be meaningless",
    )
    # ⚠ EVERY row, not just one — a loop that asserts on a fixed index is
    # vacuous and passes whatever the other rows do.
    for t in range(NUM_TASKS):
        assert_almost_equal(
            a[2 + t], Scalar[DT](1.0), atol=1e-9,
            msg=(
                "OFF must leave EVERY per-task row at its 1.0 init — a table"
                " that updates while the flag is off means the deviation is"
                " leaking into runs that claim to reproduce the reference"
            ),
        )
    print("  OFF  per-task table:", a[2], a[3], a[4], "(all must be 1.0)")


def test_on_changes_the_policy_update() raises:
    """A flag wired to nothing passes the OFF gate perfectly."""
    var off = _run_cpu(False)
    var on = _run_cpu(True)
    print("  ON   pi =", on[0], " shared =", on[1], " table:", on[2], on[3], on[4])
    assert_true(
        isfinite(Float64(on[0])), "ON produced a non-finite policy loss"
    )
    assert_true(
        abs(Float64(on[2] - 1.0)) > 1e-9
        or abs(Float64(on[3] - 1.0)) > 1e-9
        or abs(Float64(on[4] - 1.0)) > 1e-9,
        "ON did not update the per-task table — the per-task percentile pass"
        " never ran",
    )
    assert_true(
        abs(Float64(on[0] - off[0])) > 1e-9,
        "ON changed NOTHING against OFF — the reweighted backward seed is not"
        " reaching the policy update",
    )
    print("  |pi_on - pi_off| =", abs(Float64(on[0] - off[0])))


def test_gpu_enabled_path_is_finite() raises:
    """GPU builds the reweighted seed HOST-side and uploads it, where the OFF
    path fills it with a kernel — a different code path, so gated separately."""
    seed(11)
    var ctx = DeviceContext()
    var ag = AgGPU.make(
        lr=Scalar[DT](1e-3), action_scale=Scalar[DT](2.0), learning_starts=0,
        ctx=ctx,
    )
    ag.set_per_task_pi_scale(True)
    var obs = List[Scalar[DT]](length=MAX_OBS, fill=Scalar[DT](0))
    var act = List[Scalar[DT]](length=MAX_ACT, fill=Scalar[DT](0))
    for i in range(N_TXN):
        ag.set_task(i % NUM_TASKS)
        var r = _txn(i, obs, act)
        ag.record(obs, act, r, Scalar[DT](0.0))
    for _ in range(10):
        _ = ag.train_step()
    print("  GPU  pi =", ag.last_pi_loss(), " task2 =", ag.task_pi_scale(2))
    assert_true(
        isfinite(Float64(ag.last_pi_loss())),
        "GPU per-task path produced a non-finite policy loss",
    )
    assert_true(
        isfinite(Float64(ag.last_wm_loss())),
        "GPU per-task path produced a non-finite world-model loss",
    )
    assert_true(
        abs(Float64(ag.task_pi_scale(2) - 1.0)) > 1e-9,
        "GPU per-task table never updated",
    )


def main() raises:
    print("=" * 70)
    print("TD-MPC2 per-task policy-loss scale (deviation, default OFF)")
    print("=" * 70)
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("=" * 70)
    print("PER-TASK PI_SCALE GATE PASSED")
    print("=" * 70)

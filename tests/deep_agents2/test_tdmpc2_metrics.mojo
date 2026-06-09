"""TD-MPC2 metrics (CPU) — per-component losses + flush bundle.

Trains a few steps, then flush_metrics drains the diag-window accumulators
into a TDMPC2Metrics bundle. Checks: components finite; wm_loss ==
consistency + reward + value; flush resets the window (a second flush with
no training in between yields the same wm_loss components averaged over the
prior window vs a fresh window).

Run: `pixi run mojo run -I . tests/deep_agents2/test_tdmpc2_metrics.mojo`
"""

from std.memory import alloc
from std.math import isfinite, abs
from std.random import random_float64, seed
from std.testing import assert_true, assert_almost_equal, TestSuite

from mojo_rl.nn2.constants import DT
from mojo_rl.core.logger import NoOpLogger
from mojo_rl.deep_agents2.tdmpc2.agent import TDMPC2Agent

comptime OBS = 3
comptime ENC = 32
comptime ACT = 1
comptime LATENT = 32
comptime MLP = 32
comptime BINS = 21
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime B = 8
comptime H = 3
comptime CAP = 2000

comptime Ag = TDMPC2Agent[
    "cpu", OBS, ENC, ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H, CAP
]


def test_metrics() raises:
    seed(0)
    var ag = Ag.make(lr=Scalar[DT](1e-3), learning_starts=0)

    # fill replay + train a few steps
    var ob = alloc[Scalar[DT]](OBS)
    var ac = alloc[Scalar[DT]](ACT)
    for _ in range(200):
        for i in range(OBS):
            ob[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
        ac[0] = Scalar[DT](random_float64() * 2.0 - 1.0)
        ag.record(ob, ac, Scalar[DT](random_float64() - 1.0), Scalar[DT](0.0))
    for _ in range(8):
        _ = ag.train_step()

    var none: Optional[UnsafePointer[NoOpLogger, MutAnyOrigin]] = None
    var m = ag.flush_metrics[NoOpLogger](none, 0)

    print(
        "  cons=", m.consistency_loss, " rew=", m.reward_loss,
        " val=", m.value_loss, " wm=", m.wm_loss, " pi=", m.pi_loss,
        " pi_scale=", m.pi_scale,
    )
    print(
        "  q_mean=", m.q_mean, " q_min=", m.q_min, " q_max=", m.q_max,
        " td_mean=", m.td_target_mean, " td_min=", m.td_target_min,
        " td_max=", m.td_target_max,
    )
    assert_true(isfinite(m.consistency_loss), "consistency finite")
    assert_true(isfinite(m.reward_loss), "reward finite")
    assert_true(isfinite(m.value_loss), "value finite")
    assert_true(isfinite(m.pi_loss), "pi finite")
    assert_true(m.pi_scale > Scalar[DT](0.0), "pi_scale positive")
    assert_true(isfinite(m.q_mean), "q_mean finite")
    assert_true(isfinite(m.td_target_mean), "td_target_mean finite")
    assert_true(m.q_max >= m.q_min, "q_max >= q_min")
    assert_true(m.td_target_max >= m.td_target_min, "td_max >= td_min")
    assert_almost_equal(
        m.wm_loss, m.consistency_loss + m.reward_loss + m.value_loss,
        atol=1e-5, msg="wm_loss == cons + rew + val",
    )
    # flush reset the window → a second flush with no training reads zeros.
    var m2 = ag.flush_metrics[NoOpLogger](none, 1)
    assert_true(m2.wm_loss == Scalar[DT](0.0), "window reset after flush")

    ob.free(); ac.free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

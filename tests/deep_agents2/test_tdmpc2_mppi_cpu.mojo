"""TD-MPC2 MPPI (CPU) integration — nn2 world model ↔ shared MPPICPU planner.

Builds a CPU world model (dynamics / reward / policy / target-Q) + the
TDMPC2RolloutCallbackCPU, runs one MPPICPU.plan, and checks the returned
action is finite and in [-action_scale, action_scale]. Validates the
callback bridge (List[Float64] ↔ nn2 forward, two-hot reward/Q decode,
action scaling); the planner's optimization itself is covered by its own
stub-model tests.

Run: `pixi run mojo run -I . tests/deep_agents2/test_tdmpc2_mppi_cpu.mojo`
"""

from std.math import isfinite
from std.random import random_float64, seed
from std.testing import assert_true, TestSuite

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.planners.trajectory.mppi import MPPICPU
from mojo_rl.deep_agents2.tdmpc2.nets import (
    TDMPC2Dynamics, TDMPC2Reward, TDMPC2QNet, TDMPC2Policy,
)
from mojo_rl.deep_agents2.tdmpc2.callback import TDMPC2RolloutCallbackCPU

comptime ACT = 2
comptime LATENT = 16
comptime MLP = 16
comptime BINS = 11
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
# small planning config
comptime HORIZON = 3
comptime NUM_SAMPLES = 64
comptime NUM_PI_TRAJS = 8
comptime NUM_ITERS = 3
comptime NUM_ELITES = 16


def test_mppi_cpu_plan() raises:
    seed(0)
    comptime DynT = TDMPC2Dynamics[LATENT, ACT, MLP, SN]
    comptime RewT = TDMPC2Reward[LATENT, ACT, MLP, BINS]
    comptime QNetT = TDMPC2QNet[LATENT, ACT, MLP, BINS]
    comptime PolicyT = TDMPC2Policy[LATENT, ACT, MLP]
    comptime CB = TDMPC2RolloutCallbackCPU[ACT, LATENT, MLP, BINS, SN, VMIN, VMAX]
    comptime Planner = MPPICPU[
        LATENT, ACT, HORIZON, NUM_SAMPLES, NUM_PI_TRAJS, NUM_ITERS, NUM_ELITES
    ]

    var dyn = DynT.make["cpu", INIT=Kaiming]()
    var rew = RewT.make["cpu", INIT=Kaiming]()
    var pol = PolicyT.make["cpu", INIT=Kaiming]()
    var qt = List[QNetT]()
    qt.append(QNetT.make["cpu", INIT=Kaiming]())
    qt.append(QNetT.make["cpu", INIT=Kaiming]())

    var action_scale = 1.0
    var cb = CB.make(dyn, rew, pol, qt, action_scale, 0, 1)
    var planner = Planner()

    var z0 = List[Float64](length=LATENT, fill=0.0)
    for i in range(LATENT):
        z0[i] = random_float64() * 2.0 - 1.0

    var act = planner.plan[CB](
        cb, z0, gamma=0.99, temperature=0.5, action_scale=action_scale,
        deterministic=True,
    )

    assert_true(len(act) == ACT, "action length == ACT")
    for j in range(ACT):
        assert_true(isfinite(act[j]), "action finite")
        assert_true(
            act[j] >= -action_scale - 1e-5 and act[j] <= action_scale + 1e-5,
            "action within [-scale, scale]",
        )
    print("  MPPI action =", act[0], act[1])

    # Lifetime extenders: the callback holds raw pointers to these modules;
    # keep them alive past plan() (feedback_mojo_set_external_lifetime).
    _ = dyn^
    _ = rew^
    _ = pol^
    _ = qt^
    _ = cb^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

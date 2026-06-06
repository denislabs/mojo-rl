"""TD-MPC2 checkpoint round-trip (CPU).

agent A (seed 1) saves; a fresh agent B (seed 2, different init → different
action) loads A's checkpoint and must reproduce A's greedy action
dimension-by-dimension. Validates save_state/load_state restores every
module (encoder/dynamics/reward/online+target Q/policy) exactly.

Run: `pixi run mojo run -I . tests/deep_agents2/test_tdmpc2_checkpoint.mojo`
"""

from std.memory import alloc
from std.math import abs
from std.random import seed
from std.testing import assert_true, assert_almost_equal, TestSuite

from mojo_rl.nn2.constants import DT
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
comptime PATH = "tdmpc2_ckpt_test.ckpt"

comptime Ag = TDMPC2Agent[
    "cpu", OBS, ENC, ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H, CAP
]


def _greedy(mut ag: Ag, obs: UnsafePointer[Scalar[DT], MutAnyOrigin]) raises -> List[Scalar[DT]]:
    var act = alloc[Scalar[DT]](ACT)
    ag.select_greedy_action(obs, act)
    var out = List[Scalar[DT]](length=ACT, fill=0.0)
    for j in range(ACT):
        out[j] = act[j]
    act.free()
    return out^


def test_checkpoint_roundtrip() raises:
    var probe = alloc[Scalar[DT]](OBS)
    for d in range(OBS):
        probe[d] = Scalar[DT](0.2 * Float64(d) - 0.3)

    seed(1)
    var a = Ag.make(action_scale=Scalar[DT](2.0))
    var act_a = _greedy(a, probe)
    a.save_state(PATH)

    seed(2)
    var b = Ag.make(action_scale=Scalar[DT](2.0))
    var act_b0 = _greedy(b, probe)
    # different init → different action (sanity that the test isn't trivial)
    var differ = False
    for j in range(ACT):
        if abs(Float64(act_b0[j] - act_a[j])) > 1e-4:
            differ = True
    assert_true(differ, "fresh agent should differ before load")

    b.load_state(PATH)
    var act_b1 = _greedy(b, probe)
    for j in range(ACT):
        print("  dim", j, " A=", act_a[j], " B(after load)=", act_b1[j])
        assert_almost_equal(
            act_b1[j], act_a[j], atol=1e-5,
            msg="loaded agent must reproduce saved agent's action",
        )
    probe.free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

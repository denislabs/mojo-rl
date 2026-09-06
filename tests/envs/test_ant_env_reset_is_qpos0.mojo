"""The Ant env resets from MuJoCo's `qpos0` (z 0.75, ankles 0) and still runs.

`<custom><numeric name="init_qpos">` in ant.xml (z 0.55, ankles ±1 rad) used
to be the reset pose; the parser stopped applying it on 2026-09-06 because
MuJoCo and Gymnasium never did (`test_qpos0_vs_mujoco`). This pins what the
env does with that: the first observation's height is 0.75 ± the 0.1 reset
noise, and three episodes of 300 random steps produce finite observations
and rewards and do not terminate on the first step — the ankles start
outside their range and MuJoCo's own first step is a limit shove, which the
`healthy` band [0.2, 1.0] has to survive, as it does in Gymnasium.

Run: pixi run mojo run -I . tests/envs/test_ant_env_reset_is_qpos0.mojo
"""
from std.math import abs
from std.random import seed, random_float64
from std.testing import assert_true, TestSuite

from mojo_rl.envs.ant import Ant
from mojo_rl.envs.ant.ant_xml import AntModel
from mojo_rl.core.cont_action import ContAction


def test_reset_height_is_mujoco_qpos0() raises:
    print("=== Ant reset: obs[0] (torso z) is qpos0's 0.75 +- noise ===")
    seed(7)
    var env = Ant[DType.float64]()
    for ep in range(5):
        var obs = env.reset()
        var z = Float64(obs[0])
        print("  episode", ep, "reset z =", z)
        assert_true(
            abs(z - 0.75) <= 0.1 + 1e-9,
            "reset z " + String(z) + " is not 0.75 +- 0.1 — the reset pose is"
            " not MuJoCo's qpos0",
        )
    print("  PASS")


def test_random_rollout_survives_the_first_step() raises:
    print("=== Ant rollout: 3 episodes x 300 random steps from qpos0 ===")
    seed(11)
    var env = Ant[DType.float64]()
    for ep in range(3):
        var obs = env.reset()
        var steps = 0
        var total = 0.0
        for t in range(300):
            var a = ContAction[AntModel.ACTION_DIM]()
            for j in range(AntModel.ACTION_DIM):
                a.data[j] = random_float64() * 2.0 - 1.0
            var r = env.step(a)
            obs = r[0]
            var rew = Float64(r[1])
            assert_true(rew == rew, "NaN reward at step " + String(t))
            for k in range(AntModel.OBS_DIM):
                var v = Float64(obs[k])
                assert_true(v == v and abs(v) < 1e6, "non-finite obs at step " + String(t))
            total += rew
            steps = t + 1
            if r[2]:
                break
        print("  episode", ep, "steps", steps, "return", total)
        assert_true(steps > 1, "episode terminated on its first step — the limit shove at qpos0 leaves the healthy band")
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

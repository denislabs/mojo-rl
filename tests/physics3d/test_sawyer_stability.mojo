"""Test: Sawyer Reach physics stability with random actions.

Verifies the arm doesn't diverge to NaN over 500 steps of random actions.
"""

from std.testing import assert_true, TestSuite
from std.math import isnan
from std.random import seed, random_float64
from mojo_rl.envs.metaworld import SawyerReach
from mojo_rl.core import ContAction


def test_sawyer_no_nan() raises:
    """Run 500 steps of random actions and verify no NaN in arm qpos."""
    print("=== Sawyer Stability Test (500 steps) ===")
    seed(42)

    var env = SawyerReach()
    _ = env.reset()

    var max_qpos: Float64 = 0
    var nan_step = -1
    comptime ACTION_DIM = 4

    for step in range(500):
        var action = ContAction[ACTION_DIM]()
        for i in range(ACTION_DIM):
            action.data[i] = random_float64(-1.0, 1.0)

        _ = env.step(action)

        # Check arm qpos (first 9) for NaN — ignore object free joint
        for i in range(9):
            var q = Float64(env.data.qpos[i])
            if isnan(q):
                nan_step = step
                print("NaN detected at step", step, "qpos[", i, "]")
                break
            if abs(q) > max_qpos:
                max_qpos = abs(q)

        if nan_step >= 0:
            break

        if step % 100 == 0:
            var hx = Float64(env.data.xpos[24 * 3 + 0])
            var hy = Float64(env.data.xpos[24 * 3 + 1])
            var hz = Float64(env.data.xpos[24 * 3 + 2])
            var mx = Float64(env.data.mocap_pos[32 * 3 + 0])
            var my = Float64(env.data.mocap_pos[32 * 3 + 1])
            var mz = Float64(env.data.mocap_pos[32 * 3 + 2])
            var max_vel: Float64 = 0
            for i in range(9):
                var v = abs(Float64(env.data.qvel[i]))
                if v > max_vel:
                    max_vel = v
            print(
                "Step", step,
                " hand=(", hx, ",", hy, ",", hz, ")",
                " mocap=(", mx, ",", my, ",", mz, ")",
                " max_vel=", max_vel,
            )

    assert_true(nan_step == -1, "Physics diverged to NaN!")
    print("PASS: No NaN after 500 steps, max |qpos| =", max_qpos)


def main() raises:
    test_sawyer_no_nan()

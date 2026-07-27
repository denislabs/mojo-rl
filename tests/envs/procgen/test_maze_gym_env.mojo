"""`MazeGymEnv` guard: `BoxDiscreteActionEnv` conformance + obs pipeline.

Fast check that the training adapter produces a well-formed 3×84×84 normalized
observation and steps through the gym API. Full learning is exercised by
`examples/procgen/maze_rainbow_training_cpu.mojo`. See `docs/PROCGEN_PORT.md`.
"""

from std.testing import assert_equal, assert_true, TestSuite
from mojo_rl.core.env_traits import BoxDiscreteActionEnv
from mojo_rl.envs.procgen.games import MazeGymEnv

comptime ASSET_ROOT = String("assets/procgen/")


def _drive[E: BoxDiscreteActionEnv](mut env: E) -> Int:
    # Exercises the trait surface generically (compile-time conformance proof).
    var obs = env.reset_obs_list()
    var n = len(obs)
    _ = env.step_obs(4)
    _ = env.was_terminated()
    _ = env.num_actions()
    _ = env.obs_dim()
    return n


def test_gym_env_conformance_and_obs() raises:
    var env = MazeGymEnv[DType.float32](
        ASSET_ROOT, rand_seed=0, num_levels=1, start_level=0
    )
    assert_equal(env.obs_dim(), 3 * 84 * 84)
    assert_equal(env.num_actions(), 15)

    var obs = env.reset_obs_list()
    assert_equal(len(obs), 3 * 84 * 84)

    # Normalized to [0,1], and not a flat frame.
    var mn = Float32(2.0)
    var mx = Float32(-1.0)
    for i in range(len(obs)):
        var v = obs[i]
        assert_true(v >= 0.0 and v <= 1.0, "obs out of [0,1]")
        if v < mn:
            mn = v
        if v > mx:
            mx = v
    assert_true(mx - mn > 0.1, "obs is flat")

    # A stand action: well-formed obs, no reward, not done on step 1.
    var res = env.step_obs(4)
    assert_equal(len(res[0]), 3 * 84 * 84)
    assert_equal(Int(res[1]), 0)
    assert_true(not res[2], "unexpected early termination")
    assert_true(not env.was_terminated())

    # Generic trait-bound drive (conformance).
    assert_equal(_drive(env), 3 * 84 * 84)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

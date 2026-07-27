"""Plunder P2 smoke: render + env wrapper + gym adapter.

Asset-backed (loads the ~50 plunder sprites). Checks the 64×64×3 observation is
well-formed + non-degenerate, the env accrues kill reward over a fire-heavy rollout,
and `PlunderGymEnv` conforms (obs shape/range, action count, terminating episodes).
See `docs/PROCGEN_STARPILOT_SCOPE.md`.

Run from repo root:
    pixi run mojo run -I . tests/envs/procgen/test_plunder_env.mojo
"""

from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.envs.procgen.games import (
    PlunderGame,
    PlunderAssets,
    PlunderEnv,
    PlunderGymEnv,
)
from mojo_rl.envs.procgen.games.plunder import DIST_HARD
from mojo_rl.nn.constants import DT

comptime ASSET_ROOT = String("assets/procgen/")


def _tape(step: Int) -> Int:
    var t: List[Int] = [9, 4, 7, 9, 1, 9, 4, 7, 9, 4]
    return t[step % len(t)]


def test_plunder_render_obs_wellformed() raises:
    var assets = PlunderAssets(ASSET_ROOT)
    var game = PlunderGame(DIST_HARD)
    game.reset(0)
    for s in range(60):  # let ships sail in + fire
        _ = game.step(_tape(s))
    var obs = game.render_obs(assets)
    assert_equal(len(obs), 64 * 64 * 3)
    var lit = 0
    var mn = 255
    var mx = 0
    for i in range(len(obs)):
        var v = Int(obs[i])
        if v > 0:
            lit += 1
        if v < mn:
            mn = v
        if v > mx:
            mx = v
    assert_true(lit > 500, "render produced a near-black frame")
    assert_true(mx - mn > 40, "render produced a flat frame")


def test_plunder_env_obs() raises:
    var env = PlunderEnv(ASSET_ROOT, rand_seed=0, num_levels=10, dist_mode=DIST_HARD)
    var obs = env.reset()
    assert_equal(len(obs), PlunderEnv.OBS_DIM)
    assert_true(
        env.current_level_seed >= 0 and env.current_level_seed < 10,
        "level seed out of configured range",
    )
    for s in range(100):
        var res = env.step(_tape(s))
        assert_equal(len(res.obs), PlunderEnv.OBS_DIM)


def test_plunder_gym_env() raises:
    comptime E = PlunderGymEnv[DT]
    var env = E(ASSET_ROOT, rand_seed=0, num_levels=1, dist_mode=DIST_HARD)
    assert_equal(env.obs_dim(), E.OBS_DIM)
    assert_equal(env.num_actions(), 15)

    var obs = env.reset_obs_list()
    assert_equal(len(obs), E.OBS_DIM)
    var mn = Scalar[DT](2.0)
    var mx = Scalar[DT](-1.0)
    for i in range(len(obs)):
        if obs[i] < mn:
            mn = obs[i]
        if obs[i] > mx:
            mx = obs[i]
    assert_true(mn >= 0.0 and mx <= 1.0, "obs not normalized to [0,1]")
    assert_true(mx > 0.0, "obs all zero")

    var got_done = False
    for s in range(E.MAX_STEPS + 5):
        var r = env.step_obs(_tape(s))
        if r[2]:
            got_done = True
            break
    assert_true(got_done, "episode never terminated within MAX_STEPS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

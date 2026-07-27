"""Dodgeball P2 smoke: render + env wrapper + gym adapter.

Asset-backed. Checks the 64×64×3 obs is well-formed + non-degenerate, the env
steps a jump rollout, and `DodgeballGymEnv` conforms. See `docs/PROCGEN_CLIMBER_SCOPE.md`.
"""

from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.envs.procgen.games import (
    DodgeballGame, DodgeballAssets, DodgeballEnv, DodgeballGymEnv
)
from mojo_rl.envs.procgen.games.dodgeball import DIST_HARD
from mojo_rl.nn.constants import DT

comptime ASSET_ROOT = String("assets/procgen/")


def _tape(step: Int) -> Int:
    var t: List[Int] = [8, 7, 5, 8, 7, 4, 8, 1, 8, 5]
    return t[step % len(t)]


def test_dodgeball_render_obs_wellformed() raises:
    var assets = DodgeballAssets(ASSET_ROOT)
    var game = DodgeballGame(DIST_HARD)
    game.reset(0)
    for s in range(30):
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


def test_dodgeball_env_obs() raises:
    var env = DodgeballEnv(ASSET_ROOT, rand_seed=0, num_levels=10, dist_mode=DIST_HARD)
    var obs = env.reset()
    assert_equal(len(obs), DodgeballEnv.OBS_DIM)
    assert_true(env.current_level_seed >= 0 and env.current_level_seed < 10, "level seed range")
    for s in range(100):
        var res = env.step(_tape(s))
        assert_equal(len(res.obs), DodgeballEnv.OBS_DIM)


def test_dodgeball_gym_env() raises:
    comptime E = DodgeballGymEnv[DT]
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
    assert_true(mn >= 0.0 and mx <= 1.0, "obs normalized")
    assert_true(mx > 0.0, "obs all zero")
    var got_done = False
    for s in range(E.MAX_STEPS + 5):
        var r = env.step_obs(_tape(s))
        if r[2]:
            got_done = True
            break
    assert_true(got_done, "episode never terminated")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

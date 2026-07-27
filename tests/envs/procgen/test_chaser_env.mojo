"""Chaser P2 smoke: render + env wrapper + gym adapter.

Asset-backed (loads the chaser sprites), so slower to JIT than the parity tests.
Checks the 64×64×3 observation is well-formed and non-degenerate, that a scripted
episode through `ChaserEnv` accumulates orb reward, and that `ChaserGymEnv`
conforms (obs shape/range, action count, terminating episodes).

Run from repo root:
    pixi run mojo run -I . tests/envs/procgen/test_chaser_env.mojo
"""

from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.envs.procgen.games import (
    ChaserGame,
    ChaserAssets,
    ChaserEnv,
    ChaserGymEnv,
    ChaserAction,
    DIST_EASY,
)
from mojo_rl.nn.constants import DT

comptime ASSET_ROOT = String("assets/procgen/")


def _tape(step: Int) -> Int:
    var t: List[Int] = [7, 7, 5, 5, 1, 1, 3, 3, 8, 6, 2, 0, 4, 7, 5, 3, 1]
    return t[step % len(t)]


def test_chaser_render_obs_wellformed() raises:
    var assets = ChaserAssets(ASSET_ROOT)
    var game = ChaserGame(DIST_EASY)
    game.reset(7)
    var obs = game.render_obs(assets)
    assert_equal(len(obs), 64 * 64 * 3)
    # Non-degenerate: background + walls + orbs + sprites fill the frame.
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


def test_chaser_env_episode() raises:
    # Dense reward: a scripted 200-step rollout should collect several orbs.
    var env = ChaserEnv(ASSET_ROOT, rand_seed=0, num_levels=10, dist_mode=DIST_EASY)
    var obs = env.reset()
    assert_equal(len(obs), ChaserEnv.OBS_DIM)
    assert_true(
        env.current_level_seed >= 0 and env.current_level_seed < 10,
        "level seed out of configured range",
    )
    var total: Float32 = 0.0
    for s in range(200):
        var res = env.step(_tape(s))
        assert_equal(len(res.obs), ChaserEnv.OBS_DIM)
        total += res.reward
    assert_true(total > 0.0, "no orb reward collected over 200 steps")


def test_chaser_gym_env() raises:
    comptime E = ChaserGymEnv[DT]
    var env = E(ASSET_ROOT, rand_seed=0, num_levels=1, dist_mode=DIST_EASY)
    assert_equal(env.obs_dim(), E.OBS_DIM)
    assert_equal(env.num_actions(), 15)

    var obs = env.reset_obs_list()
    assert_equal(len(obs), E.OBS_DIM)
    # Normalized to [0, 1].
    var mn = Scalar[DT](2.0)
    var mx = Scalar[DT](-1.0)
    for i in range(len(obs)):
        if obs[i] < mn:
            mn = obs[i]
        if obs[i] > mx:
            mx = obs[i]
    assert_true(mn >= 0.0 and mx <= 1.0, "obs not normalized to [0,1]")
    assert_true(mx > 0.0, "obs all zero")

    # Steps advance; episode must terminate within MAX_STEPS.
    var got_done = False
    for s in range(E.MAX_STEPS + 5):
        var r = env.step_obs(_tape(s))
        if r[2]:
            got_done = True
            break
    assert_true(got_done, "episode never terminated within MAX_STEPS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

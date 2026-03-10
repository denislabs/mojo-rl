"""Smoke test for AtariEnv — BoxDiscreteActionEnv conformance."""

from envs.atari.atari_env import AtariEnv, AtariEnvState, AtariAction
from envs.atari.environment import load_rom
from envs.atari.games.pong import PongDef
from envs.atari.flags import RAM_SIZE, OBS_WIDTH, OBS_HEIGHT


fn test_ram_mode() raises:
    """Test RAM mode (OBS_MODE=0): 128-dim observations."""
    print("Test: AtariEnv RAM mode...")

    var rom = load_rom(
        "/Users/denislaboureyras/opt/anaconda3/lib/python3.9/site-packages/ale_py/roms/pong.bin"
    )
    var env = AtariEnv[PongDef, 0](rom.data, rom.size)

    # Check obs dimension
    assert_equal(env.obs_dim(), RAM_SIZE, "obs_dim should be 128")
    assert_equal(env.num_actions(), 6, "Pong has 6 actions")

    # Reset and check obs
    var obs = env.reset_obs_list()
    assert_equal(len(obs), RAM_SIZE, "reset_obs_list should return 128 floats")

    # Check all values in [0, 1]
    var all_in_range = True
    for i in range(len(obs)):
        if obs[i] < 0.0 or obs[i] > 1.0:
            all_in_range = False
            break
    assert_true(all_in_range, "All RAM obs should be in [0, 1]")

    # Step with NOOP and check obs changes are valid
    var result = env.step_obs(0)  # NOOP
    assert_equal(len(result[0].copy()), RAM_SIZE, "step_obs should return 128 floats")

    # Step several times
    for i in range(10):
        _ = env.step_obs(i % 6)

    # Check state
    _ = env.get_state()

    # Test action_from_index
    var action = env.action_from_index(3)
    assert_equal(action.action_idx, 3, "action_from_index should preserve index")

    env.close()
    print("  PASSED")


fn test_pixel_mode() raises:
    """Test pixel mode (OBS_MODE=1): 4×84×84 = 28224-dim observations."""
    print("Test: AtariEnv pixel mode...")

    var rom = load_rom(
        "/Users/denislaboureyras/opt/anaconda3/lib/python3.9/site-packages/ale_py/roms/pong.bin"
    )
    var env = AtariEnv[PongDef, 1](rom.data, rom.size)

    comptime EXPECTED_DIM: Int = 4 * OBS_WIDTH * OBS_HEIGHT  # 28224

    # Check obs dimension
    assert_equal(env.obs_dim(), EXPECTED_DIM, "pixel obs_dim should be 28224")

    # Reset and check obs
    var obs = env.reset_obs_list()
    assert_equal(len(obs), EXPECTED_DIM, "reset_obs_list should return 28224 floats")

    # After reset, all 4 frames should be identical (same initial frame)
    # Check first frame equals second frame
    var frames_equal = True
    for i in range(OBS_WIDTH * OBS_HEIGHT):
        if obs[i] != obs[i + OBS_WIDTH * OBS_HEIGHT]:
            frames_equal = False
            break
    assert_true(frames_equal, "After reset, stacked frames should be identical")

    # Check values in [0, 1]
    var all_in_range = True
    for i in range(len(obs)):
        if obs[i] < 0.0 or obs[i] > 1.0:
            all_in_range = False
            break
    assert_true(all_in_range, "All pixel obs should be in [0, 1]")

    # Step and verify
    var result = env.step_obs(0)
    assert_equal(len(result[0]), EXPECTED_DIM, "step_obs pixel should return 28224 floats")

    # Step a few more times
    for i in range(5):
        _ = env.step_obs(i % 6)

    env.close()
    print("  PASSED")


fn test_trait_conformance() raises:
    """Test that AtariEnv works through BoxDiscreteActionEnv interface."""
    print("Test: BoxDiscreteActionEnv conformance...")

    var rom = load_rom(
        "/Users/denislaboureyras/opt/anaconda3/lib/python3.9/site-packages/ale_py/roms/pong.bin"
    )
    var env = AtariEnv[PongDef, 0](rom.data, rom.size)

    # These are the BoxDiscreteActionEnv methods
    _ = env.reset_obs_list()  # ContinuousStateEnv
    _ = env.get_obs_list()    # ContinuousStateEnv
    _ = env.obs_dim()         # ContinuousStateEnv
    _ = env.num_actions()     # DiscreteActionEnv
    _ = env.action_from_index(0) # DiscreteActionEnv
    _ = env.step_obs(0)       # BoxDiscreteActionEnv

    # Env base trait methods
    _ = env.reset()
    _ = env.step(AtariAction(action_idx=1))
    _ = env.get_state()
    env.close()

    print("  PASSED")


# ========================================================================
# Helpers
# ========================================================================

fn assert_equal(a: Int, b: Int, msg: String) raises:
    if a != b:
        raise Error("ASSERTION FAILED: " + msg + " (got " + String(a) + " != " + String(b) + ")")

fn assert_true(val: Bool, msg: String) raises:
    if not val:
        raise Error("ASSERTION FAILED: " + msg)


fn main() raises:
    print("=== AtariEnv Smoke Tests ===")
    test_ram_mode()
    test_pixel_mode()
    test_trait_conformance()
    print("=== All AtariEnv tests passed! ===")

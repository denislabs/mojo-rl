"""Test planar MuJoCo environments."""

from envs.mujoco_planar import (
    HopperPlanar,
    HopperPlanarConstants,
    Walker2dPlanar,
    Walker2dPlanarConstants,
    HalfCheetahPlanar,
    HalfCheetahPlanarConstants,
)


fn test_hopper() raises:
    """Test HopperPlanar environment."""
    print("\nTesting HopperPlanar...")

    print("  Constants:")
    print("    NUM_BODIES:", HopperPlanarConstants.NUM_BODIES)
    print("    NUM_JOINTS:", HopperPlanarConstants.NUM_JOINTS)
    print("    OBS_DIM:", HopperPlanarConstants.OBS_DIM)
    print("    ACTION_DIM:", HopperPlanarConstants.ACTION_DIM)

    var env = HopperPlanar()
    var obs = env.reset()

    print("  Initial observation (", len(obs), "D):")
    print("    Height (z):", obs[0])
    print("    Torso angle:", obs[1])

    # Take a few steps with random actions
    var total_reward = 0.0
    for step in range(10):
        var action = List[Float64]()
        action.append(0.0)  # hip
        action.append(0.0)  # knee
        action.append(0.0)  # ankle

        var result = env.step(action)
        var reward = result[1]
        var done = result[2]
        total_reward += reward

        if done:
            print("  Episode ended at step", step + 1)
            break

    print("  Total reward after 10 steps:", total_reward)
    print("  HopperPlanar test passed!")


fn test_walker() raises:
    """Test Walker2dPlanar environment."""
    print("\nTesting Walker2dPlanar...")

    print("  Constants:")
    print("    NUM_BODIES:", Walker2dPlanarConstants.NUM_BODIES)
    print("    NUM_JOINTS:", Walker2dPlanarConstants.NUM_JOINTS)
    print("    OBS_DIM:", Walker2dPlanarConstants.OBS_DIM)
    print("    ACTION_DIM:", Walker2dPlanarConstants.ACTION_DIM)

    var env = Walker2dPlanar()
    var obs = env.reset()

    print("  Initial observation (", len(obs), "D):")
    print("    Height (z):", obs[0])
    print("    Torso angle:", obs[1])

    var total_reward = 0.0
    for step in range(10):
        var action = List[Float64]()
        for _ in range(6):
            action.append(0.0)

        var result = env.step(action)
        var reward = result[1]
        var done = result[2]
        total_reward += reward

        if done:
            print("  Episode ended at step", step + 1)
            break

    print("  Total reward after 10 steps:", total_reward)
    print("  Walker2dPlanar test passed!")


fn test_cheetah() raises:
    """Test HalfCheetahPlanar environment."""
    print("\nTesting HalfCheetahPlanar...")

    print("  Constants:")
    print("    NUM_BODIES:", HalfCheetahPlanarConstants.NUM_BODIES)
    print("    NUM_JOINTS:", HalfCheetahPlanarConstants.NUM_JOINTS)
    print("    OBS_DIM:", HalfCheetahPlanarConstants.OBS_DIM)
    print("    ACTION_DIM:", HalfCheetahPlanarConstants.ACTION_DIM)

    var env = HalfCheetahPlanar()
    var obs = env.reset()

    print("  Initial observation (", len(obs), "D):")
    print("    Height (z):", obs[0])
    print("    Torso angle:", obs[1])

    var total_reward = 0.0
    for step in range(10):
        var action = List[Float64]()
        for _ in range(6):
            action.append(0.1)  # Small forward action

        var result = env.step(action)
        var reward = result[1]
        var done = result[2]
        total_reward += reward

        if done:
            print("  Episode ended at step", step + 1)
            break

    print("  Total reward after 10 steps:", total_reward)
    print("  HalfCheetahPlanar test passed!")


fn main() raises:
    print("Testing MuJoCo Planar Environments")
    print("==================================")

    test_hopper()
    test_walker()
    test_cheetah()

    print("\nAll MuJoCo planar environment tests passed!")

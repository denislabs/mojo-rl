"""Test native MuJoCo 3D environments."""

from envs.mujoco_native import (
    Hopper3D,
    Hopper3DConstants,
    Walker2d3D,
    Walker2d3DConstants,
    HalfCheetah3D,
    HalfCheetah3DConstants,
)


fn test_hopper3d() raises:
    """Test Hopper3D environment."""
    print("\nTesting Hopper3D...")

    print("  Constants:")
    print("    NUM_BODIES:", Hopper3DConstants.NUM_BODIES)
    print("    NUM_JOINTS:", Hopper3DConstants.NUM_JOINTS)
    print("    OBS_DIM:", Hopper3DConstants.OBS_DIM)
    print("    ACTION_DIM:", Hopper3DConstants.ACTION_DIM)

    var env = Hopper3D()
    var obs = env.reset()

    print("  Initial observation (", len(obs), "D):")
    print("    Height (z):", obs[0])
    print("    Torso pitch:", obs[1])

    var total_reward = 0.0
    for step in range(10):
        var action = List[Float64]()
        action.append(0.1)
        action.append(-0.1)
        action.append(0.0)

        var result = env.step(action^)
        var reward = result[1]
        var done = result[2]
        total_reward += reward

        if done:
            print("  Episode ended at step", step + 1)
            break

    print("  Total reward after 10 steps:", total_reward)
    print("  Hopper3D test passed!")


fn test_walker2d3d() raises:
    """Test Walker2d3D environment."""
    print("\nTesting Walker2d3D...")

    print("  Constants:")
    print("    NUM_BODIES:", Walker2d3DConstants.NUM_BODIES)
    print("    NUM_JOINTS:", Walker2d3DConstants.NUM_JOINTS)
    print("    OBS_DIM:", Walker2d3DConstants.OBS_DIM)
    print("    ACTION_DIM:", Walker2d3DConstants.ACTION_DIM)

    var env = Walker2d3D()
    var obs = env.reset()

    print("  Initial observation (", len(obs), "D):")
    print("    Height (z):", obs[0])
    print("    Torso pitch:", obs[1])

    var total_reward = 0.0
    for step in range(10):
        var action = List[Float64]()
        for _ in range(6):
            action.append(0.0)

        var result = env.step(action^)
        var reward = result[1]
        var done = result[2]
        total_reward += reward

        if done:
            print("  Episode ended at step", step + 1)
            break

    print("  Total reward after 10 steps:", total_reward)
    print("  Walker2d3D test passed!")


fn test_cheetah3d() raises:
    """Test HalfCheetah3D environment."""
    print("\nTesting HalfCheetah3D...")

    print("  Constants:")
    print("    NUM_BODIES:", HalfCheetah3DConstants.NUM_BODIES)
    print("    NUM_JOINTS:", HalfCheetah3DConstants.NUM_JOINTS)
    print("    OBS_DIM:", HalfCheetah3DConstants.OBS_DIM)
    print("    ACTION_DIM:", HalfCheetah3DConstants.ACTION_DIM)

    var env = HalfCheetah3D()
    var obs = env.reset()

    print("  Initial observation (", len(obs), "D):")
    print("    Height (z):", obs[0])
    print("    Torso pitch:", obs[1])

    var total_reward = 0.0
    for step in range(10):
        var action = List[Float64]()
        for _ in range(6):
            action.append(0.1)

        var result = env.step(action^)
        var reward = result[1]
        var done = result[2]
        total_reward += reward

        if done:
            print("  Episode ended at step", step + 1)
            break

    print("  Total reward after 10 steps:", total_reward)
    print("  HalfCheetah3D test passed!")


fn main() raises:
    print("Testing Native MuJoCo 3D Environments")
    print("=====================================")

    test_hopper3d()
    test_walker2d3d()
    test_cheetah3d()

    print("\nAll native MuJoCo 3D environment tests passed!")

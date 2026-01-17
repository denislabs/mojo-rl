"""Test LunarLanderV2 using the new PhysicsState architecture."""

from gpu.host import DeviceContext

from envs.lunar_lander_v2 import LunarLanderV2
from envs.lunar_lander_physics import LunarLanderPhysics
from physics_gpu import dtype


fn test_basic_functionality() raises:
    """Test basic environment functionality."""
    print("=" * 60)
    print("LunarLanderV2: Basic Functionality Test")
    print("=" * 60)

    var env = LunarLanderV2[1](seed=42)

    # Test initial observation
    var obs = env.get_observation(0)
    print("Initial observation:")
    print("  x:", obs[0], "y:", obs[1])
    print("  vx:", obs[2], "vy:", obs[3])
    print("  angle:", obs[4], "omega:", obs[5])
    print("  left_leg:", obs[6], "right_leg:", obs[7])

    # Test a few steps
    print("\nStepping through actions:")
    var total_reward = Scalar[dtype](0)

    for step in range(20):
        var action = step % 4
        var result = env.step(0, action)
        var reward = result[0]
        var done = result[1]
        total_reward = total_reward + reward

        if step % 5 == 0:
            obs = env.get_observation(0)
            print("Step", step, "action", action, "reward", reward, "y", obs[1])

        if done:
            print("Episode done at step", step)
            break

    print("Total reward:", total_reward)
    print("\n✓ TEST PASSED: Basic functionality works!")


fn test_cpu_gpu_equivalence() raises:
    """Test that CPU and GPU produce identical results."""
    print("\n" + "=" * 60)
    print("LunarLanderV2: CPU vs GPU Equivalence Test")
    print("=" * 60)

    comptime BATCH: Int = 1
    comptime NUM_STEPS: Int = 100

    var cpu_env = LunarLanderV2[BATCH](seed=12345)
    var gpu_env = LunarLanderV2[BATCH](seed=12345)

    var ctx = DeviceContext()

    var actions = InlineArray[Int, 4](0, 1, 2, 3)

    var max_obs_error = Scalar[dtype](0)
    var max_reward_error = Scalar[dtype](0)
    var done_mismatch = 0

    print("Step | Max Obs Err  | Max Reward Err | Done Match")
    print("-" * 60)

    for step in range(NUM_STEPS):
        var action = actions[step % 4]

        # Sync RNG state
        var shared_counter = cpu_env.rng_counter

        # CPU step
        cpu_env.rng_counter = shared_counter
        var cpu_result = cpu_env.step(0, action)
        var cpu_reward = cpu_result[0]
        var cpu_done = cpu_result[1]

        # GPU step
        gpu_env.rng_counter = shared_counter
        var gpu_result = gpu_env.step_gpu(0, action, ctx)
        var gpu_reward = gpu_result[0]
        var gpu_done = gpu_result[1]

        # Compare observations
        var cpu_obs = cpu_env.get_observation(0)
        var gpu_obs = gpu_env.get_observation(0)

        var step_obs_error = Scalar[dtype](0)
        for i in range(8):
            var obs_err = cpu_obs[i] - gpu_obs[i]
            if obs_err < Scalar[dtype](0):
                obs_err = -obs_err
            if obs_err > step_obs_error:
                step_obs_error = obs_err

        # Compare rewards
        var step_reward_error = cpu_reward - gpu_reward
        if step_reward_error < Scalar[dtype](0):
            step_reward_error = -step_reward_error

        # Compare done flags
        var step_done_match = cpu_done == gpu_done
        if not step_done_match:
            done_mismatch += 1

        if step_obs_error > max_obs_error:
            max_obs_error = step_obs_error
        if step_reward_error > max_reward_error:
            max_reward_error = step_reward_error

        if step % 10 == 0:
            var done_str = "YES" if step_done_match else "NO"
            print(step, "   |", step_obs_error, "|", step_reward_error, "|", done_str)

        # Reset both if done
        if cpu_done or gpu_done:
            var reset_counter = cpu_env.rng_counter
            cpu_env.rng_counter = reset_counter
            cpu_env.reset(0)
            gpu_env.rng_counter = reset_counter
            gpu_env.reset(0)

    print("-" * 60)
    print("Final Maximum Errors:")
    print("  Observation:", max_obs_error)
    print("  Reward:     ", max_reward_error)
    print("  Done mismatches:", done_mismatch)

    var tolerance = Scalar[dtype](1e-5)
    if max_obs_error < tolerance and max_reward_error < tolerance and done_mismatch == 0:
        print("\n✓ TEST PASSED: CPU and GPU LunarLanderV2 match!")
    else:
        print("\n✗ TEST FAILED: CPU and GPU results differ!")


fn test_v1_v2_equivalence() raises:
    """Test that V2 produces same results as original LunarLanderPhysics."""
    print("\n" + "=" * 60)
    print("LunarLanderV2 vs LunarLanderPhysics Equivalence Test")
    print("=" * 60)

    comptime BATCH: Int = 1
    comptime NUM_STEPS: Int = 50

    var v1_env = LunarLanderPhysics[BATCH](seed=99999)
    var v2_env = LunarLanderV2[BATCH](seed=99999)

    var actions = InlineArray[Int, 4](0, 1, 2, 3)

    var max_obs_error = Scalar[dtype](0)
    var max_reward_error = Scalar[dtype](0)

    print("Step | V1 x      | V2 x      | Diff")
    print("-" * 60)

    for step in range(NUM_STEPS):
        var action = actions[step % 4]

        # Sync RNG
        var shared_counter = v1_env.rng_counter

        v1_env.rng_counter = shared_counter
        var v1_result = v1_env.step(0, action)
        var v1_reward = v1_result[0]
        var v1_done = v1_result[1]

        v2_env.rng_counter = shared_counter
        var v2_result = v2_env.step(0, action)
        var v2_reward = v2_result[0]
        var v2_done = v2_result[1]

        # Compare observations
        var v1_obs = v1_env.get_observation(0)
        var v2_obs = v2_env.get_observation(0)

        var step_obs_error = Scalar[dtype](0)
        for i in range(8):
            var obs_err = v1_obs[i] - v2_obs[i]
            if obs_err < Scalar[dtype](0):
                obs_err = -obs_err
            if obs_err > step_obs_error:
                step_obs_error = obs_err

        var step_reward_error = v1_reward - v2_reward
        if step_reward_error < Scalar[dtype](0):
            step_reward_error = -step_reward_error

        if step_obs_error > max_obs_error:
            max_obs_error = step_obs_error
        if step_reward_error > max_reward_error:
            max_reward_error = step_reward_error

        if step % 10 == 0:
            print(step, "  |", v1_obs[0], "|", v2_obs[0], "|", step_obs_error)

        if v1_done or v2_done:
            var reset_counter = v1_env.rng_counter
            v1_env.rng_counter = reset_counter
            v1_env.reset(0)
            v2_env.rng_counter = reset_counter
            v2_env.reset(0)

    print("-" * 60)
    print("Maximum Errors:")
    print("  Observation:", max_obs_error)
    print("  Reward:     ", max_reward_error)

    var tolerance = Scalar[dtype](1e-5)
    if max_obs_error < tolerance and max_reward_error < tolerance:
        print("\n✓ TEST PASSED: V2 matches original LunarLanderPhysics!")
    else:
        print("\n✗ TEST FAILED: V2 differs from original!")


fn main() raises:
    """Run all LunarLanderV2 tests."""
    print("\n")
    print("=" * 60)
    print("    LUNAR LANDER V2 TESTS")
    print("=" * 60)

    test_basic_functionality()
    test_cpu_gpu_equivalence()
    test_v1_v2_equivalence()

    print("\n" + "=" * 60)
    print("All LunarLanderV2 tests completed!")
    print("=" * 60)

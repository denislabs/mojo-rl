"""Test LunarLander CPU-GPU equivalence using physics_gpu module."""

from gpu.host import DeviceContext

from envs.lunar_lander_physics import LunarLanderPhysics
from physics_gpu import dtype


fn test_cpu_gpu_equivalence() raises:
    """Test that CPU and GPU LunarLander produce identical results."""
    print("=" * 60)
    print("LunarLander Physics: CPU vs GPU Equivalence Test")
    print("=" * 60)

    # Use BATCH=1 for clean comparison
    comptime BATCH: Int = 1
    comptime NUM_STEPS: Int = 100

    # Create two environments with same seed
    var cpu_env = LunarLanderPhysics[BATCH](seed=12345)
    var gpu_env = LunarLanderPhysics[BATCH](seed=12345)

    # GPU context
    var ctx = DeviceContext()

    # Test actions sequence
    var actions = InlineArray[Int, 4](0, 1, 2, 3)

    var max_obs_error = Scalar[dtype](0)
    var max_reward_error = Scalar[dtype](0)
    var done_mismatch = 0

    print("Step | Max Obs Err  | Max Reward Err | Done Match")
    print("-" * 60)

    for step in range(NUM_STEPS):
        var action = actions[step % 4]

        # IMPORTANT: Sync Philox state BEFORE stepping so both use same random values
        # Both envs should have same seed and counter
        var shared_counter = cpu_env.rng_counter

        # Step CPU
        cpu_env.rng_counter = shared_counter
        var cpu_result = cpu_env.step(0, action)
        var cpu_reward = cpu_result[0]
        var cpu_done = cpu_result[1]

        # Step GPU with same Philox counter
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

        # Print every 10 steps
        if step % 10 == 0:
            var done_str = "YES" if step_done_match else "NO"
            print(
                step,
                "   |",
                step_obs_error,
                "|",
                step_reward_error,
                "|",
                done_str,
            )

        # Reset both if done (with same Philox state)
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

    # Check tolerance
    var tolerance = Scalar[dtype](1e-5)
    if (
        max_obs_error < tolerance
        and max_reward_error < tolerance
        and done_mismatch == 0
    ):
        print("\n TEST PASSED: CPU and GPU LunarLander match!")
    else:
        print("\n TEST FAILED: CPU and GPU results differ!")


fn test_basic_functionality() raises:
    """Test basic environment functionality."""
    print("\n" + "=" * 60)
    print("LunarLander Physics: Basic Functionality Test")
    print("=" * 60)

    var env = LunarLanderPhysics[1](seed=42)

    # Test initial observation
    var obs = env.get_observation(0)
    print("Initial observation:")
    print("  x:", obs[0], "y:", obs[1])
    print("  vx:", obs[2], "vy:", obs[3])
    print("  angle:", obs[4], "omega:", obs[5])
    print("  left_leg:", obs[6], "right_leg:", obs[7])

    # Test a few steps with different actions
    print("\nStepping through actions:")
    var total_reward = Scalar[dtype](0)

    for step in range(20):
        var action = step % 4  # Cycle through actions
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
    print("\n TEST PASSED: Basic functionality works!")


fn main() raises:
    """Run all LunarLander tests."""
    print("\n")
    print("=" * 60)
    print("    LUNAR LANDER PHYSICS GPU TESTS")
    print("=" * 60)

    test_basic_functionality()
    test_cpu_gpu_equivalence()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

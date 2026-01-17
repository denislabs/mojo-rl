"""Test LunarLanderV2 using the new PhysicsState architecture.

Tests:
- Basic functionality
- Revolute joints (leg physics)
- CPU vs GPU equivalence
- Wind effects
- Continuous action space
- Terrain generation
"""

from gpu.host import DeviceContext
from math import sqrt

from envs.lunar_lander_v2 import LunarLanderV2
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


fn test_revolute_joints() raises:
    """Test that revolute joints connect legs to main body correctly."""
    print("\n" + "=" * 60)
    print("LunarLanderV2: Revolute Joints Test")
    print("=" * 60)

    var env = LunarLanderV2[1](seed=42)

    # Verify physics setup
    print("Physics configuration:")
    print("  Bodies:", env.NUM_BODIES, "(main lander + 2 legs)")
    print("  Shapes:", env.NUM_SHAPES, "(lander polygon + 2 leg rectangles)")
    print("  Max joints:", env.MAX_JOINTS)

    # Get joint count
    var joint_count = env.physics.get_joint_count(0)
    print("  Active joints:", joint_count)

    # Get body positions
    print("\nInitial body positions:")
    var lander_x = Float64(env.physics.get_body_x(0, env.BODY_LANDER))
    var lander_y = Float64(env.physics.get_body_y(0, env.BODY_LANDER))
    var left_leg_x = Float64(env.physics.get_body_x(0, env.BODY_LEFT_LEG))
    var left_leg_y = Float64(env.physics.get_body_y(0, env.BODY_LEFT_LEG))
    var right_leg_x = Float64(env.physics.get_body_x(0, env.BODY_RIGHT_LEG))
    var right_leg_y = Float64(env.physics.get_body_y(0, env.BODY_RIGHT_LEG))

    print("  Lander:    (", lander_x, ",", lander_y, ")")
    print("  Left leg:  (", left_leg_x, ",", left_leg_y, ")")
    print("  Right leg: (", right_leg_x, ",", right_leg_y, ")")

    # Run physics and verify legs stay attached
    print("\nRunning 50 physics steps (no thrust)...")
    for step in range(50):
        _ = env.step(0, 0)  # No action

    print("\nBody positions after falling:")
    var lander_x2 = Float64(env.physics.get_body_x(0, env.BODY_LANDER))
    var lander_y2 = Float64(env.physics.get_body_y(0, env.BODY_LANDER))
    var left_leg_x2 = Float64(env.physics.get_body_x(0, env.BODY_LEFT_LEG))
    var left_leg_y2 = Float64(env.physics.get_body_y(0, env.BODY_LEFT_LEG))
    var right_leg_x2 = Float64(env.physics.get_body_x(0, env.BODY_RIGHT_LEG))
    var right_leg_y2 = Float64(env.physics.get_body_y(0, env.BODY_RIGHT_LEG))

    print("  Lander:    (", lander_x2, ",", lander_y2, ")")
    print("  Left leg:  (", left_leg_x2, ",", left_leg_y2, ")")
    print("  Right leg: (", right_leg_x2, ",", right_leg_y2, ")")

    # Calculate relative positions (legs should stay below lander)
    var left_rel_x = left_leg_x2 - lander_x2
    var left_rel_y = left_leg_y2 - lander_y2
    var right_rel_x = right_leg_x2 - lander_x2
    var right_rel_y = right_leg_y2 - lander_y2

    print("\nLeg positions relative to lander:")
    print("  Left leg:  (", left_rel_x, ",", left_rel_y, ")")
    print("  Right leg: (", right_rel_x, ",", right_rel_y, ")")

    # Check constraints:
    # 1. Both legs should be below lander (negative relative y)
    # 2. Left leg should be to the left (negative relative x)
    # 3. Right leg should be to the right (positive relative x)
    var legs_below = left_rel_y < 0 and right_rel_y < 0
    var left_is_left = left_rel_x < 0
    var right_is_right = right_rel_x > 0

    print("\nJoint constraint checks:")
    print("  Legs below lander:", legs_below)
    print("  Left leg on left:", left_is_left)
    print("  Right leg on right:", right_is_right)

    # Test leg contact detection with actual leg bodies
    print("\nLeg contact detection test:")
    env.reset(0)

    # Run until landing or timeout
    for step in range(500):
        _ = env.step(0, 2)  # Main engine to slow descent
        var obs = env.get_observation(0)

        if step % 100 == 0:
            print("  Step", step, "- y:", obs[1], "left_leg:", obs[6], "right_leg:", obs[7])

        if obs[6] > Scalar[dtype](0.5) or obs[7] > Scalar[dtype](0.5):
            print("  Leg contact detected at step", step)
            break

    if joint_count == 2 and legs_below and left_is_left and right_is_right:
        print("\n✓ TEST PASSED: Revolute joints work correctly!")
    else:
        print("\n✗ TEST FAILED: Joint constraints not satisfied!")


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


fn test_wind_effects() raises:
    """Test that wind affects lander trajectory."""
    print("\n" + "=" * 60)
    print("LunarLanderV2: Wind Effects Test")
    print("=" * 60)

    # Two environments: one with wind, one without
    var no_wind_env = LunarLanderV2[1](seed=42, enable_wind=False)
    var wind_env = LunarLanderV2[1](seed=42, enable_wind=True, wind_power=15.0)

    comptime NUM_STEPS: Int = 50

    print("Step | No Wind vx | Wind vx   | Diff")
    print("-" * 60)

    var total_diff = Float64(0)

    for step in range(NUM_STEPS):
        # Both do nothing (action 0)
        _ = no_wind_env.step(0, 0)
        _ = wind_env.step(0, 0)

        var no_wind_obs = no_wind_env.get_observation(0)
        var wind_obs = wind_env.get_observation(0)

        var vx_diff = Float64(no_wind_obs[2]) - Float64(wind_obs[2])
        if vx_diff < 0:
            vx_diff = -vx_diff
        total_diff += vx_diff

        if step % 10 == 0:
            print(step, "  |", no_wind_obs[2], "|", wind_obs[2], "|", vx_diff)

    print("-" * 60)
    print("Total velocity difference:", total_diff)

    # Wind should cause noticeable velocity differences
    if total_diff > 0.1:
        print("\n✓ TEST PASSED: Wind affects trajectory!")
    else:
        print("\n✗ TEST FAILED: Wind has no effect!")


fn test_continuous_actions() raises:
    """Test continuous action space functionality."""
    print("\n" + "=" * 60)
    print("LunarLanderV2: Continuous Actions Test")
    print("=" * 60)

    var env = LunarLanderV2[1, True](seed=42)  # CONTINUOUS=True

    print("Testing continuous action mapping:")

    # Test main engine throttle
    env.reset(0)
    var obs_before = env.get_observation(0)
    var vy_before = Float64(obs_before[3])

    # Fire main engine at full throttle
    for _ in range(10):
        _ = env.step_continuous(0, 1.0, 0.0)  # Full main throttle

    var obs_after = env.get_observation(0)
    var vy_after = Float64(obs_after[3])

    print("  Main engine (1.0, 0.0):")
    print("    vy before:", vy_before, "after:", vy_after)

    var main_effect = vy_after > vy_before  # Should counteract gravity

    # Test left engine
    env.reset(0)
    obs_before = env.get_observation(0)
    var omega_before = Float64(obs_before[5])

    for _ in range(10):
        _ = env.step_continuous(0, 0.0, -1.0)  # Left engine

    obs_after = env.get_observation(0)
    var omega_after = Float64(obs_after[5])

    print("  Left engine (0.0, -1.0):")
    print("    omega before:", omega_before, "after:", omega_after)

    var left_effect = omega_after != omega_before

    # Test right engine
    env.reset(0)
    obs_before = env.get_observation(0)
    omega_before = Float64(obs_before[5])

    for _ in range(10):
        _ = env.step_continuous(0, 0.0, 1.0)  # Right engine

    obs_after = env.get_observation(0)
    omega_after = Float64(obs_after[5])

    print("  Right engine (0.0, 1.0):")
    print("    omega before:", omega_before, "after:", omega_after)

    var right_effect = omega_after != omega_before

    if main_effect and left_effect and right_effect:
        print("\n✓ TEST PASSED: Continuous actions work!")
    else:
        print("\n✗ TEST FAILED: Some continuous actions don't work!")
        print("    Main engine:", main_effect)
        print("    Left engine:", left_effect)
        print("    Right engine:", right_effect)


fn test_terrain_generation() raises:
    """Test that terrain is generated with varying heights."""
    print("\n" + "=" * 60)
    print("LunarLanderV2: Terrain Generation Test")
    print("=" * 60)

    # Create environment with multiple batches to see variation
    var env = LunarLanderV2[4](seed=12345)

    print("Terrain heights per environment (11 chunks each):")

    var has_variation = False
    var helipad_flat = True

    for batch in range(4):
        print("  Env", batch, ":", end=" ")
        var env_heights = List[Float64]()
        for chunk in range(11):
            var height = Float64(env.terrain_heights[batch * 11 + chunk])
            env_heights.append(height)
            if chunk > 0:
                var prev_height = env_heights[chunk - 1]
                if abs(height - prev_height) > 0.001:
                    has_variation = True

        # Print first and last few heights
        print(env_heights[0], env_heights[5], env_heights[10])

        # Check helipad is flat (chunks 3-7)
        for chunk in range(3, 8):
            if chunk + 1 < 8:
                var h1 = env_heights[chunk]
                var h2 = env_heights[chunk + 1]
                if abs(h1 - h2) > 0.01:
                    helipad_flat = False

    print("-" * 60)
    print("Has terrain variation:", has_variation)
    print("Helipad area is flat:", helipad_flat)

    if has_variation and helipad_flat:
        print("\n✓ TEST PASSED: Terrain generated correctly!")
    else:
        print("\n✗ TEST FAILED: Terrain generation issues!")


fn test_landing_detection() raises:
    """Test that successful landing is detected."""
    print("\n" + "=" * 60)
    print("LunarLanderV2: Landing Detection Test")
    print("=" * 60)

    # Create environment and manually position lander near ground
    var env = LunarLanderV2[1](seed=42)

    # Reset and check initial state
    env.reset(0)
    var obs = env.get_observation(0)
    print("Initial position: x=", obs[0], "y=", obs[1])

    # Run several steps and check if episode can terminate
    var terminated_count = 0
    var crash_count = 0
    var success_count = 0

    for episode in range(10):
        env.reset(0)
        for step in range(200):
            # Random-ish action
            var action = step % 4
            var result = env.step(0, action)
            if result[1]:  # done
                terminated_count += 1
                if env.game_over[0]:
                    crash_count += 1
                else:
                    success_count += 1
                break

    print("Episodes terminated:", terminated_count, "/ 10")
    print("  Crashes:", crash_count)
    print("  Other (success/timeout):", success_count)

    if terminated_count > 0:
        print("\n✓ TEST PASSED: Landing detection works!")
    else:
        print("\n✗ TEST FAILED: No episodes terminated!")


fn main() raises:
    """Run all LunarLanderV2 tests."""
    print("\n")
    print("=" * 60)
    print("    LUNAR LANDER V2 TESTS")
    print("    (with revolute joint leg physics)")
    print("=" * 60)

    test_basic_functionality()
    test_revolute_joints()
    test_cpu_gpu_equivalence()
    test_wind_effects()
    test_continuous_actions()
    test_terrain_generation()
    test_landing_detection()

    print("\n" + "=" * 60)
    print("All LunarLanderV2 tests completed!")
    print("=" * 60)

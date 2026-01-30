"""Diagnostic tests for HalfCheetah3D environment.

Tests to identify why training rewards oscillate:
1. Observation consistency - same state should give same observation
2. Reward calculation - verify reward makes sense
3. Action effects - verify motors actually move the joints
4. Physics stability - check for numerical issues

Run with:
    pixi run mojo run tests/test_half_cheetah_3d_diagnostic.mojo
"""

from random import seed
from math import sqrt

from envs.half_cheetah_3d import HalfCheetah3D
from envs.half_cheetah_3d.constants3d import HC3DConstantsCPU


fn abs_f64(x: Float64) -> Float64:
    return x if x >= 0 else -x


fn abs_scalar[D: DType](x: Scalar[D]) -> Scalar[D]:
    return x if x >= 0 else -x


# =============================================================================
# Test 1: Observation Consistency
# =============================================================================


fn test_observation_consistency() raises:
    """Test that the same state produces the same observation."""
    print("\n" + "=" * 60)
    print("TEST 1: OBSERVATION CONSISTENCY")
    print("=" * 60)

    var env = HalfCheetah3D(seed=42)

    # Reset and get observation
    var obs1 = env.reset_obs_list()
    print("First observation after reset:")
    for i in range(5):
        print("  obs[" + String(i) + "]:", obs1[i])

    # Get observation again without stepping
    var obs2 = env.get_obs_list()
    print("\nSecond call to get_obs_list (no step):")
    for i in range(5):
        print("  obs[" + String(i) + "]:", obs2[i])

    # Check they match
    var max_diff: Float64 = 0.0
    for i in range(len(obs1)):
        var diff = abs_f64(Float64(obs1[i]) - Float64(obs2[i]))
        if diff > max_diff:
            max_diff = diff

    print("\nMax difference:", max_diff)
    if max_diff > 1e-6:
        print("FAIL: Observations should be identical without stepping")
        raise Error("Observation consistency test failed")

    print("PASSED: Observation consistency test")


# =============================================================================
# Test 2: Reward Sanity Check
# =============================================================================


fn test_reward_calculation() raises:
    """Test that rewards make sense."""
    print("\n" + "=" * 60)
    print("TEST 2: REWARD CALCULATION")
    print("=" * 60)

    var env = HalfCheetah3D(seed=42)
    _ = env.reset_obs_list()

    print("Testing reward with zero actions (should be near 0 due to no movement):")

    # Take a few steps with zero actions
    var zero_actions = List[Scalar[DType.float32]]()
    for _ in range(6):
        zero_actions.append(Scalar[DType.float32](0.0))

    var total_reward: Float64 = 0.0
    for step in range(10):
        var result = env.step_continuous_vec(zero_actions)
        var reward = Float64(result[1])
        total_reward += reward
        if step < 3:
            print("  Step", step, "reward:", reward)

    print("Total reward (10 steps, zero actions):", total_reward)

    # Test with extreme actions
    print("\nTesting reward with maximum actions (should have ctrl_cost):")
    _ = env.reset_obs_list()

    var max_actions = List[Scalar[DType.float32]]()
    for _ in range(6):
        max_actions.append(Scalar[DType.float32](1.0))

    total_reward = 0.0
    for step in range(10):
        var result = env.step_continuous_vec(max_actions)
        var reward = Float64(result[1])
        total_reward += reward
        if step < 3:
            print("  Step", step, "reward:", reward)

    print("Total reward (10 steps, max actions):", total_reward)
    print()

    # Expected: zero actions should have better reward than max actions
    # because of control cost
    print("PASSED: Reward calculation test (manual inspection)")


# =============================================================================
# Test 3: Motor Effect on Joints
# =============================================================================


fn test_motor_effect() raises:
    """Test that motor actions affect joint angles."""
    print("\n" + "=" * 60)
    print("TEST 3: MOTOR EFFECT ON JOINTS")
    print("=" * 60)

    var env = HalfCheetah3D(seed=42)
    var obs_initial = env.reset_obs_list()

    print("Initial joint angles (obs[2:8]):")
    for i in range(2, 8):
        print("  Joint", i - 2, ":", obs_initial[i])

    # Apply torque to first joint (back thigh)
    var actions = List[Scalar[DType.float32]]()
    actions.append(Scalar[DType.float32](1.0))  # Max torque on back thigh
    for _ in range(5):
        actions.append(Scalar[DType.float32](0.0))

    print("\nApplying max torque to back thigh joint for 50 steps...")

    for _ in range(50):
        _ = env.step_continuous_vec(actions)

    var obs_after = env.get_obs_list()
    print("\nJoint angles after 50 steps:")
    for i in range(2, 8):
        var change = Float64(obs_after[i]) - Float64(obs_initial[i])
        print(
            "  Joint",
            i - 2,
            ":",
            obs_after[i],
            "(change:",
            String(change)[:8],
            ")",
        )

    # Check if first joint changed significantly
    var back_thigh_change = abs_f64(
        Float64(obs_after[2]) - Float64(obs_initial[2])
    )
    print("\nBack thigh angle change:", back_thigh_change)

    if back_thigh_change < 0.01:
        print("WARNING: Joint barely moved despite full torque!")
        print("This could indicate a problem with motor control")
    else:
        print("Joint moved as expected")

    print("PASSED: Motor effect test (manual inspection)")


# =============================================================================
# Test 4: Physics Stability
# =============================================================================


fn test_physics_stability() raises:
    """Test for numerical instability in physics."""
    print("\n" + "=" * 60)
    print("TEST 4: PHYSICS STABILITY")
    print("=" * 60)

    var env = HalfCheetah3D(seed=42)
    _ = env.reset_obs_list()

    # Run with random actions and check for NaN/Inf
    var has_nan = False
    var has_extreme = False
    var max_obs_value: Float64 = 0.0
    var min_obs_value: Float64 = 0.0

    print("Running 1000 steps with random actions...")

    for step in range(1000):
        # Random actions
        var actions = List[Scalar[DType.float32]]()
        for j in range(6):
            # Simple pseudo-random
            var rand_val = Float64(step * 17 + j * 7) % 1.0 * 2.0 - 1.0
            actions.append(Scalar[DType.float32](rand_val))

        var result = env.step_continuous_vec(actions)
        var obs = result[0].copy()
        var reward = result[1]
        var done = result[2]

        # Check for NaN/Inf
        for i in range(len(obs)):
            var val = Float64(obs[i])
            if val != val:  # NaN check
                has_nan = True
                print("NaN detected at step", step, "obs[", i, "]")
                break

            if abs_f64(val) > 1e6:
                has_extreme = True
                print(
                    "Extreme value at step", step, "obs[", i, "] =", val
                )

            if val > max_obs_value:
                max_obs_value = val
            if val < min_obs_value:
                min_obs_value = val

        if has_nan:
            break

        if done:
            # Reset and continue
            _ = env.reset_obs_list()

    print("\nResults:")
    print("  NaN detected:", has_nan)
    print("  Extreme values (>1e6):", has_extreme)
    print("  Max observation value:", max_obs_value)
    print("  Min observation value:", min_obs_value)

    if has_nan:
        print("FAIL: NaN values detected - physics is unstable!")
        raise Error("Physics stability test failed")

    if has_extreme:
        print("WARNING: Extreme values detected - may cause training issues")
    else:
        print("Physics appears stable")

    print("PASSED: Physics stability test")


# =============================================================================
# Test 5: Reward Variance Analysis
# =============================================================================


fn test_reward_variance() raises:
    """Analyze reward variance across episodes."""
    print("\n" + "=" * 60)
    print("TEST 5: REWARD VARIANCE ANALYSIS")
    print("=" * 60)

    var env = HalfCheetah3D(seed=42)

    var num_episodes = 10
    var steps_per_episode = 100
    var rewards = List[Float64]()

    print("Running", num_episodes, "episodes of", steps_per_episode, "steps each...")
    print()

    for episode in range(num_episodes):
        _ = env.reset_obs_list()
        var episode_reward: Float64 = 0.0

        for step in range(steps_per_episode):
            # Use consistent actions for fair comparison
            var actions = List[Scalar[DType.float32]]()
            for i in range(6):
                # Sine wave pattern for locomotion-like motion
                var t = Float64(step) * 0.1 + Float64(i) * 0.5
                var action_val = Float64(
                    0.5 * (Float64(step % 20) / 10.0 - 1.0)
                )
                actions.append(Scalar[DType.float32](action_val))

            var result = env.step_continuous_vec(actions)
            episode_reward += Float64(result[1])

        rewards.append(episode_reward)
        print("Episode", episode + 1, "reward:", String(episode_reward)[:10])

    # Compute statistics
    var mean_reward: Float64 = 0.0
    for i in range(len(rewards)):
        mean_reward += rewards[i]
    mean_reward /= Float64(len(rewards))

    var variance: Float64 = 0.0
    var min_reward = rewards[0]
    var max_reward = rewards[0]
    for i in range(len(rewards)):
        variance += (rewards[i] - mean_reward) * (rewards[i] - mean_reward)
        if rewards[i] < min_reward:
            min_reward = rewards[i]
        if rewards[i] > max_reward:
            max_reward = rewards[i]
    variance /= Float64(len(rewards))
    var std_dev = sqrt(variance)

    print()
    print("Statistics:")
    print("  Mean reward:", mean_reward)
    print("  Std dev:", std_dev)
    print("  Min:", min_reward)
    print("  Max:", max_reward)
    print("  Range:", max_reward - min_reward)

    # High variance could indicate physics issues
    if std_dev > abs_f64(mean_reward) * 0.5:
        print("\nWARNING: High reward variance relative to mean")
        print("This could cause training instability")

    print("\nPASSED: Reward variance analysis (manual inspection)")


# =============================================================================
# Test 6: Forward Velocity Tracking
# =============================================================================


fn test_forward_velocity() raises:
    """Test that forward velocity is computed correctly."""
    print("\n" + "=" * 60)
    print("TEST 6: FORWARD VELOCITY TRACKING")
    print("=" * 60)

    var env = HalfCheetah3D(seed=42)
    var obs = env.reset_obs_list()

    print("Observation layout:")
    print("  obs[0]: torso_z =", obs[0])
    print("  obs[1]: torso_pitch =", obs[1])
    print("  obs[8]: vel_x =", obs[8])
    print("  obs[9]: vel_z =", obs[9])
    print()

    # Apply forward-pushing actions
    print("Applying forward motion for 100 steps...")

    var prev_vel_x = Float64(obs[8])
    var max_vel_x: Float64 = 0.0

    for step in range(100):
        # Coordinated leg motion for forward locomotion
        var actions = List[Scalar[DType.float32]]()
        # Try to create forward motion
        var phase = Float64(step) * 0.2
        actions.append(Scalar[DType.float32](0.5))   # back thigh
        actions.append(Scalar[DType.float32](-0.5))  # back shin
        actions.append(Scalar[DType.float32](0.0))   # back foot
        actions.append(Scalar[DType.float32](-0.5))  # front thigh
        actions.append(Scalar[DType.float32](0.5))   # front shin
        actions.append(Scalar[DType.float32](0.0))   # front foot

        var result = env.step_continuous_vec(actions)
        obs = result[0].copy()

        var vel_x = Float64(obs[8])
        if vel_x > max_vel_x:
            max_vel_x = vel_x

        if step % 20 == 0:
            print(
                "  Step",
                step,
                "| vel_x:",
                String(vel_x)[:8],
                "| z:",
                String(obs[0])[:6],
            )

        prev_vel_x = vel_x

    print()
    print("Max forward velocity achieved:", max_vel_x)

    if max_vel_x < 0.1:
        print("WARNING: Very low forward velocity - motor control may be weak")

    print("PASSED: Forward velocity tracking (manual inspection)")


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    print("=" * 60)
    print("HALFCHEETAH3D DIAGNOSTIC TESTS")
    print("=" * 60)

    seed(42)

    var passed = 0
    var failed = 0

    try:
        test_observation_consistency()
        passed += 1
    except e:
        print("FAILED:", e)
        failed += 1

    try:
        test_reward_calculation()
        passed += 1
    except e:
        print("FAILED:", e)
        failed += 1

    try:
        test_motor_effect()
        passed += 1
    except e:
        print("FAILED:", e)
        failed += 1

    try:
        test_physics_stability()
        passed += 1
    except e:
        print("FAILED:", e)
        failed += 1

    try:
        test_reward_variance()
        passed += 1
    except e:
        print("FAILED:", e)
        failed += 1

    try:
        test_forward_velocity()
        passed += 1
    except e:
        print("FAILED:", e)
        failed += 1

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Passed:", passed)
    print("Failed:", failed)

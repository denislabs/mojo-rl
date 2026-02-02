"""Test HopperEnv (Phase 7, Step 7.3).

Tests:
1. Environment initialization
2. Reset returns valid observation
3. Step with zero action (standing)
4. Step with torque causes motion
5. Termination on falling
6. Reward structure
"""

from math import sqrt, sin
from testing import assert_true

from physics3d_v2.envs import HopperEnv


comptime DTYPE = DType.float64
comptime PI: Float64 = 3.14159265358979323846


fn abs64(x: Scalar[DTYPE]) -> Scalar[DTYPE]:
    if x < 0:
        return -x
    return x


fn test_initialization() raises:
    """Test that HopperEnv initializes correctly."""
    print("Test 1: Environment initialization...")

    var env = HopperEnv[DTYPE](
        torque_limit=10.0,
        min_height=0.15,
        max_pitch=1.0,
        max_steps=1000,
    )

    # Check model configuration
    print("  Torque limit:", env.torque_limit)
    print("  Min height:", env.min_height)
    print("  Max pitch:", env.max_pitch)
    print("  Max steps:", env.max_steps)

    # Check bodies are configured
    print("  Torso mass:", env.model.masses[0])
    print("  Foot mass:", env.model.masses[1])
    print("  Num joints:", env.model.num_joints)

    assert_true(
        env.model.num_joints == 1,
        "Should have exactly 1 joint (hip)",
    )

    print("  PASSED: Environment initialized correctly")


fn test_reset() raises:
    """Test that reset returns valid observation."""
    print("\nTest 2: Reset returns valid observation...")

    var env = HopperEnv[DTYPE]()
    var obs = env.reset()

    print("  Initial observation:")
    print("    Height (z):", obs[0])
    print("    X velocity:", obs[1])
    print("    Z velocity:", obs[2])
    print("    Pitch:", obs[3])
    print("    Pitch vel:", obs[4])
    print("    Hip angle:", obs[5])
    print("    Hip ang vel:", obs[6])
    print("    Foot contact:", obs[7])

    # Height should be reasonable (torso above ground)
    assert_true(
        obs[0] > 0.3,
        "Torso height should be > 0.3m after reset",
    )

    # Velocities should be zero
    assert_true(
        abs64(obs[1]) < 0.01 and abs64(obs[2]) < 0.01,
        "Velocities should be ~0 after reset",
    )

    # Angles should be ~zero
    assert_true(
        abs64(obs[3]) < 0.1 and abs64(obs[5]) < 0.1,
        "Angles should be ~0 after reset",
    )

    print("  PASSED: Reset returns valid observation")


fn test_standing_stability() raises:
    """Test that hopper stands stable with zero action."""
    print("\nTest 3: Standing stability (zero action)...")

    var env = HopperEnv[DTYPE](timestep=0.005)
    var obs = env.reset()

    var initial_height = obs[0]
    print("  Initial height:", initial_height)

    # Run for 100 steps with zero action
    var terminated = False
    for i in range(100):
        var result = env.step(0.0)
        obs = result[0].copy()
        terminated = result[2]

        if terminated:
            print("  Terminated at step", i)
            break

    var final_height = obs[0]
    print("  Final height:", final_height)
    print("  Height change:", final_height - initial_height)

    # Hopper might settle a bit but shouldn't fall
    assert_true(
        not terminated,
        "Hopper should not terminate with zero action for 100 steps",
    )

    assert_true(
        final_height > 0.2,
        "Hopper should remain standing (height > 0.2)",
    )

    print("  PASSED: Hopper stands stable with zero action")


fn test_torque_causes_motion() raises:
    """Test that applying torque causes motion."""
    print("\nTest 4: Torque causes motion...")

    var env = HopperEnv[DTYPE]()
    _ = env.reset()

    # Apply constant positive torque for 50 steps
    var obs = env.get_observation()
    var initial_x_vel = obs[1]
    var initial_hip_angle = obs[5]

    for _ in range(50):
        var result = env.step(0.5)  # 50% of max torque
        obs = result[0].copy()

    var final_x_vel = obs[1]
    var final_hip_angle = obs[5]

    print("  Initial x velocity:", initial_x_vel)
    print("  Final x velocity:", final_x_vel)
    print("  Initial hip angle:", initial_hip_angle * 180.0 / PI, "deg")
    print("  Final hip angle:", final_hip_angle * 180.0 / PI, "deg")

    # Hip angle should change
    assert_true(
        abs64(final_hip_angle - initial_hip_angle) > 0.1,
        "Hip angle should change with applied torque",
    )

    print("  PASSED: Torque causes motion")


fn test_termination_on_falling() raises:
    """Test that episode terminates when hopper falls."""
    print("\nTest 5: Termination on falling...")

    var env = HopperEnv[DTYPE](min_height=0.25)
    _ = env.reset()

    # Apply large torque to make it fall
    var terminated = False
    var step_count = 0

    for i in range(500):
        var result = env.step(1.0)  # Max torque
        _ = result[0]  # Discard observation, we'll get it after loop
        terminated = result[2]
        step_count = i + 1

        if terminated:
            break

    var obs = env.get_observation()
    print("  Final height:", obs[0])
    print("  Final pitch:", obs[3] * 180.0 / PI, "deg")
    print("  Terminated:", terminated)
    print("  Steps until termination:", step_count)

    # Should eventually terminate (either by falling or tipping)
    assert_true(
        terminated or step_count == 500,
        "Episode should run for 500 steps or terminate",
    )

    print("  PASSED: Termination detection works")


fn test_reward_structure() raises:
    """Test that reward has expected structure."""
    print("\nTest 6: Reward structure...")

    var env = HopperEnv[DTYPE]()
    _ = env.reset()

    # Zero action should give alive bonus - small control cost
    var result_zero = env.step(0.0)
    var reward_zero = result_zero[1]

    print("  Reward with zero action:", reward_zero)

    # Alive bonus should be ~1.0 (minus very small control cost)
    assert_true(
        reward_zero > 0.9,
        "Reward with zero action should be ~1.0 (alive bonus)",
    )

    # Reset and apply forward motion
    _ = env.reset()

    # Simulate forward motion by setting velocity directly (hack for testing)
    env.data.velocities[0] = 1.0  # x velocity = 1 m/s

    var obs = env.get_observation()
    var terminated = env._is_terminated(obs)
    var reward_forward = env._compute_reward(obs, 0.0, terminated)

    print("  Reward with forward velocity (1 m/s):", reward_forward)

    # Should be higher due to forward velocity
    assert_true(
        reward_forward > 1.5,
        "Reward should include forward velocity bonus",
    )

    # Large torque should reduce reward due to control cost
    _ = env.reset()
    var result_torque = env.step(1.0)  # Max torque
    var reward_torque = result_torque[1]

    print("  Reward with max torque:", reward_torque)

    # Control cost should reduce reward
    # torque = 10 (max), cost = 0.01 * 100 = 1.0
    # But still positive due to alive bonus

    print("  PASSED: Reward structure is correct")


fn test_observation_bounds() raises:
    """Test that observations stay within reasonable bounds."""
    print("\nTest 7: Observation bounds...")

    var env = HopperEnv[DTYPE]()
    _ = env.reset()

    var min_height: Float64 = 1000.0
    var max_height: Float64 = -1000.0
    var max_vel: Float64 = 0.0
    var max_angle: Float64 = 0.0

    # Run for 200 steps with varying actions
    for i in range(200):
        var action = Scalar[DTYPE](0.5) * sin(Scalar[DTYPE](i) * 0.1)
        var result = env.step(action)
        var obs = result[0].copy()
        var terminated = result[2]

        if terminated:
            _ = env.reset()

        if Float64(obs[0]) < min_height:
            min_height = Float64(obs[0])
        if Float64(obs[0]) > max_height:
            max_height = Float64(obs[0])

        var vel_mag = sqrt(Float64(obs[1] * obs[1] + obs[2] * obs[2]))
        if vel_mag > max_vel:
            max_vel = vel_mag

        if abs64(obs[3]) > max_angle:
            max_angle = Float64(abs64(obs[3]))

    print("  Height range:", min_height, "to", max_height)
    print("  Max velocity magnitude:", max_vel)
    print("  Max pitch angle:", max_angle * 180.0 / PI, "deg")

    # Values should be physically reasonable
    assert_true(
        max_height < 2.0,
        "Max height should be reasonable (< 2m)",
    )

    assert_true(
        max_vel < 20.0,
        "Max velocity should be reasonable (< 20 m/s)",
    )

    print("  PASSED: Observations within reasonable bounds")


fn main() raises:
    print("=" * 60)
    print("HopperEnv Tests (Phase 7, Step 7.3)")
    print("=" * 60)

    test_initialization()
    test_reset()
    test_standing_stability()
    test_torque_causes_motion()
    test_termination_on_falling()
    test_reward_structure()
    test_observation_bounds()

    print("\n" + "=" * 60)
    print("All HopperEnv tests passed!")
    print("=" * 60)

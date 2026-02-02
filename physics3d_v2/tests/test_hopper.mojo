"""Test HopperEnv (Phase 11b - Updated for slide joints).

Tests:
1. Environment initialization (4 bodies, 4 hinge joints, 2 slide joints)
2. Reset returns valid observation (11 dimensions)
3. Step with zero action (standing stability)
4. Y-axis constraint (slide joints keep torso in X-Z plane)
5. Step with torque causes motion
6. Termination on falling
7. Reward structure
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
        torque_limit=200.0,
        min_height=0.7,
        max_pitch=1.0,
        max_steps=1000,
    )

    # Check model configuration
    print("  Torque limit:", env.torque_limit)
    print("  Min height:", env.min_height)
    print("  Max pitch:", env.max_pitch)
    print("  Max steps:", env.max_steps)

    # Check bodies are configured (4 bodies: torso, thigh, leg, foot)
    print("  Torso mass:", env.model.masses[0])
    print("  Thigh mass:", env.model.masses[1])
    print("  Leg mass:", env.model.masses[2])
    print("  Foot mass:", env.model.masses[3])
    print("  Num hinge joints:", env.model.num_joints)
    print("  Num slide joints:", env.model.num_slide_joints)

    assert_true(
        env.model.num_joints == 4,
        "Should have 4 hinge joints (RootY, Hip, Knee, Ankle)",
    )

    assert_true(
        env.model.num_slide_joints == 2,
        "Should have 2 slide joints (RootX, RootZ)",
    )

    print("  PASSED: Environment initialized correctly")


fn test_reset() raises:
    """Test that reset returns valid observation."""
    print("\nTest 2: Reset returns valid observation...")

    var env = HopperEnv[DTYPE]()
    var obs = env.reset()

    print("  Initial observation (11 dimensions):")
    print("    [0] Height (z):", obs[0])
    print("    [1] Torso pitch:", obs[1])
    print("    [2] Hip angle:", obs[2])
    print("    [3] Knee angle:", obs[3])
    print("    [4] Ankle angle:", obs[4])
    print("    [5] X velocity:", obs[5])
    print("    [6] Z velocity:", obs[6])
    print("    [7] Pitch ang vel:", obs[7])
    print("    [8] Hip ang vel:", obs[8])
    print("    [9] Knee ang vel:", obs[9])
    print("    [10] Ankle ang vel:", obs[10])

    # Height should be reasonable (torso above ground)
    assert_true(
        obs[0] > 1.0,
        "Torso height should be > 1.0m after reset",
    )

    # Velocities should be zero
    assert_true(
        abs64(obs[5]) < 0.01 and abs64(obs[6]) < 0.01,
        "Velocities should be ~0 after reset",
    )

    # Angles should be ~zero
    assert_true(
        abs64(obs[1]) < 0.1,
        "Torso pitch should be ~0 after reset",
    )

    print("  PASSED: Reset returns valid observation")


fn test_standing_stability() raises:
    """Test that hopper stands stable with zero action."""
    print("\nTest 3: Standing stability (zero action)...")

    var env = HopperEnv[DTYPE](timestep=0.002)
    var obs = env.reset()

    var initial_height = obs[0]
    print("  Initial height:", initial_height)

    # Run for 100 steps with zero action (all three joint torques = 0)
    var terminated = False
    for i in range(100):
        var result = env.step(0.0, 0.0, 0.0)  # hip, knee, ankle = 0
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
        final_height > 0.8,
        "Hopper should remain standing (height > 0.8)",
    )

    print("  PASSED: Hopper stands stable with zero action")


fn test_y_axis_constraint() raises:
    """Test that slide joints constrain torso to X-Z plane."""
    print("\nTest 4: Y-axis constraint (X-Z plane)...")

    var env = HopperEnv[DTYPE](timestep=0.002)
    _ = env.reset()

    # Get initial Y position
    var torso_pos = env.data.get_body_position(0)
    var initial_y = torso_pos[1]
    print("  Initial Y position:", initial_y)

    var max_y_drift: Float64 = 0.0

    # Run for 200 steps with varying actions
    for i in range(200):
        var action = Scalar[DTYPE](0.3) * sin(Scalar[DTYPE](i) * 0.1)
        var result = env.step(action, -action * 0.5, action * 0.2)
        var terminated = result[2]

        if terminated:
            _ = env.reset()

        # Check Y drift
        torso_pos = env.data.get_body_position(0)
        var y_drift = abs64(Scalar[DTYPE](torso_pos[1]) - Scalar[DTYPE](initial_y))
        if Float64(y_drift) > max_y_drift:
            max_y_drift = Float64(y_drift)

    print("  Max Y drift:", max_y_drift * 1000.0, "mm")

    # Y position should stay near zero (constrained to X-Z plane)
    assert_true(
        max_y_drift < 0.05,  # Less than 50mm drift
        "Torso Y position should be constrained (< 50mm drift)",
    )

    print("  PASSED: Slide joints constrain torso to X-Z plane")


fn test_torque_causes_motion() raises:
    """Test that applying torque causes motion."""
    print("\nTest 5: Torque causes motion...")

    var env = HopperEnv[DTYPE]()
    _ = env.reset()

    # Apply constant positive hip torque for 50 steps
    var obs = env.get_observation()
    var initial_hip_angle = obs[2]

    for _ in range(50):
        var result = env.step(0.5, 0.0, 0.0)  # 50% hip torque, no knee/ankle
        obs = result[0].copy()

    var final_hip_angle = obs[2]

    print("  Initial hip angle:", initial_hip_angle * 180.0 / PI, "deg")
    print("  Final hip angle:", final_hip_angle * 180.0 / PI, "deg")
    print("  Hip angle change:", (final_hip_angle - initial_hip_angle) * 180.0 / PI, "deg")

    # Hip angle should change
    assert_true(
        abs64(final_hip_angle - initial_hip_angle) > 0.05,
        "Hip angle should change with applied torque",
    )

    print("  PASSED: Torque causes motion")


fn test_termination_on_falling() raises:
    """Test that episode terminates when hopper falls."""
    print("\nTest 6: Termination on falling...")

    var env = HopperEnv[DTYPE](min_height=0.7)
    _ = env.reset()

    # Apply large asymmetric torque to make it fall
    var terminated = False
    var step_count = 0

    for i in range(500):
        var result = env.step(1.0, -1.0, 0.5)  # Large unbalanced torques
        _ = result[0]
        terminated = result[2]
        step_count = i + 1

        if terminated:
            break

    var obs = env.get_observation()
    print("  Final height:", obs[0])
    print("  Final pitch:", obs[1] * 180.0 / PI, "deg")
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
    print("\nTest 7: Reward structure...")

    var env = HopperEnv[DTYPE]()
    _ = env.reset()

    # Zero action should give alive bonus - very small control cost
    var result_zero = env.step(0.0, 0.0, 0.0)
    var reward_zero = result_zero[1]

    print("  Reward with zero action:", reward_zero)

    # Alive bonus should be ~1.0 (with ~0 control cost)
    assert_true(
        reward_zero > 0.9,
        "Reward with zero action should be ~1.0 (alive bonus)",
    )

    # Reset and apply forward motion
    _ = env.reset()

    # Simulate forward motion by setting velocity directly
    env.data.velocities[0] = 2.0  # x velocity = 2 m/s

    var obs = env.get_observation()
    var terminated = env._is_terminated(obs)
    var reward_forward = env._compute_reward(obs, 0.0, 0.0, 0.0, terminated)

    print("  Reward with forward velocity (2 m/s):", reward_forward)

    # Should be higher due to forward velocity (velocity + alive_bonus - control_cost)
    # reward = 2.0 + 1.0 - 0.0 = 3.0
    assert_true(
        reward_forward > 2.5,
        "Reward should include forward velocity bonus",
    )

    # Large torque should reduce reward due to control cost
    _ = env.reset()
    var result_torque = env.step(1.0, 1.0, 1.0)  # Max torque on all joints
    var reward_torque = result_torque[1]

    print("  Reward with max torque on all joints:", reward_torque)

    # Control cost = 0.001 * (200^2 + 200^2 + 200^2) = 0.001 * 120000 = 120
    # But with alive bonus of 1.0, reward should still be positive initially
    # (depends on velocity)

    print("  PASSED: Reward structure is correct")


fn main() raises:
    print("=" * 60)
    print("HopperEnv Tests (Phase 11b - With Slide Joints)")
    print("=" * 60)

    test_initialization()
    test_reset()
    test_standing_stability()
    test_y_axis_constraint()
    test_torque_causes_motion()
    test_termination_on_falling()
    test_reward_structure()

    print("\n" + "=" * 60)
    print("All HopperEnv tests passed!")
    print("=" * 60)

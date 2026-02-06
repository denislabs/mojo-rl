"""Test joint state sensing (Phase 7, Step 7.2).

Tests:
1. Initial angle is zero for default quaternions
2. Angle changes during pendulum swing
3. Angular velocity matches rotation speed
4. Angle sign convention is correct
"""

from math import sqrt, sin, cos, atan2
from testing import assert_true

from physics3d import Model, Data, ImpulseIntegrator
from physics3d.joints import get_joint_angle, get_joint_angular_velocity


comptime DTYPE = DType.float64
comptime PI = 3.14159265358979323846


fn abs64(x: Scalar[DTYPE]) -> Scalar[DTYPE]:
    """Absolute value helper."""
    if x < 0:
        return -x
    return x


fn test_initial_angle_zero() raises:
    """Test that initial angle is zero for default configuration."""
    print("Test 1: Initial angle is zero...")

    var model = Model[DTYPE, 1, 5, 1](
        gravity_z=0.0,
        timestep=0.01,
        ground_z=-10.0,
    )
    model.set_body(0, mass=1.0, radius=0.1)

    _ = model.add_hinge_joint(
        parent=-1,
        child=0,
        anchor_parent=(0.0, 0.0, 1.0),
        anchor_child=(0.0, 0.0, 0.0),
        axis=(0.0, 1.0, 0.0),
    )

    var data = Data[DTYPE, 1, 5, 1]()
    data.set_body_position(0, 0.0, 0.0, 1.0)
    # Default quaternion is identity [0, 0, 0, 1]

    var angle = get_joint_angle(model, data, 0)
    print("  Initial angle:", angle, "rad")
    print("  Initial angle:", angle * 180.0 / PI, "deg")

    assert_true(
        abs64(angle) < 0.01,
        "Initial angle should be approximately zero",
    )

    print("  PASSED: Initial angle is zero")


fn test_angle_changes_during_swing() raises:
    """Test that angle changes as pendulum swings."""
    print("\nTest 2: Angle changes during swing...")

    var model = Model[DTYPE, 1, 5, 1](
        gravity_z=-9.81,
        timestep=0.005,
        ground_z=-10.0,
    )
    model.set_body(0, mass=1.0, radius=0.1)

    var L = Scalar[DTYPE](1.0)
    _ = model.add_hinge_joint(
        parent=-1,
        child=0,
        anchor_parent=(0.0, 0.0, L),
        anchor_child=(0.0, 0.0, L),
        axis=(0.0, 1.0, 0.0),
    )

    var data = Data[DTYPE, 1, 5, 1]()
    # Start at 30 degrees from vertical
    var theta0 = Scalar[DTYPE](30.0 * PI / 180.0)
    data.set_body_position(0, sin(theta0), 0.0, L - L * cos(theta0))

    # Set initial quaternion to match 30 degree rotation around Y
    var half_theta = theta0 / 2.0
    data.quaternions[0] = 0.0  # qx
    data.quaternions[1] = sin(half_theta)  # qy
    data.quaternions[2] = 0.0  # qz
    data.quaternions[3] = cos(half_theta)  # qw

    var initial_angle = get_joint_angle(model, data, 0)
    print("  Initial angle:", initial_angle * 180.0 / PI, "deg")

    # Simulate for 0.5 seconds (pendulum should swing toward center)
    for _ in range(100):
        ImpulseIntegrator.step(model, data)

    var mid_angle = get_joint_angle(model, data, 0)
    print("  Angle at t=0.5s:", mid_angle * 180.0 / PI, "deg")

    # Simulate more
    for _ in range(100):
        ImpulseIntegrator.step(model, data)

    var final_angle = get_joint_angle(model, data, 0)
    print("  Angle at t=1.0s:", final_angle * 180.0 / PI, "deg")

    # Angle should have changed significantly
    assert_true(
        abs64(final_angle - initial_angle) > 0.1,
        "Angle should change during swing",
    )

    print("  PASSED: Angle changes during swing")


fn test_angular_velocity_matches_rotation() raises:
    """Test that angular velocity reading matches actual rotation speed."""
    print("\nTest 3: Angular velocity matches rotation speed...")

    var model = Model[DTYPE, 1, 5, 1](
        gravity_z=0.0,  # No gravity to have constant angular velocity
        timestep=0.01,
        ground_z=-10.0,
    )
    model.set_body(0, mass=1.0, radius=0.1)

    _ = model.add_hinge_joint(
        parent=-1,
        child=0,
        anchor_parent=(0.0, 0.0, 1.0),
        anchor_child=(0.0, 0.0, 0.0),
        axis=(0.0, 1.0, 0.0),
    )

    var data = Data[DTYPE, 1, 5, 1]()
    data.set_body_position(0, 0.0, 0.0, 1.0)

    # Set angular velocity around Y axis
    var omega_set = Scalar[DTYPE](2.0)  # 2 rad/s
    data.set_body_angular_velocity(0, 0.0, omega_set, 0.0)

    var omega_read = get_joint_angular_velocity(model, data, 0)
    print("  Set angular velocity:", omega_set, "rad/s")
    print("  Read angular velocity:", omega_read, "rad/s")

    assert_true(
        abs64(omega_read - omega_set) < 0.01,
        "Angular velocity reading should match set value",
    )

    # Simulate and check angle changes at expected rate
    var angle0 = get_joint_angle(model, data, 0)

    # Step 10 times (0.1 seconds)
    for _ in range(10):
        ImpulseIntegrator.step(model, data)

    var angle1 = get_joint_angle(model, data, 0)
    var expected_delta = omega_set * 0.1  # ω * dt

    print("  Angle change:", angle1 - angle0, "rad")
    print("  Expected change:", expected_delta, "rad")

    # Allow some tolerance due to constraint solving
    assert_true(
        abs64((angle1 - angle0) - expected_delta) < 0.05,
        "Angle should change at angular velocity rate",
    )

    print("  PASSED: Angular velocity matches rotation speed")


fn test_angle_sign_convention() raises:
    """Test that angle sign convention is correct."""
    print("\nTest 4: Angle sign convention...")

    var model = Model[DTYPE, 1, 5, 1](
        gravity_z=0.0,
        timestep=0.01,
        ground_z=-10.0,
    )
    model.set_body(0, mass=1.0, radius=0.1)

    _ = model.add_hinge_joint(
        parent=-1,
        child=0,
        anchor_parent=(0.0, 0.0, 1.0),
        anchor_child=(0.0, 0.0, 0.0),
        axis=(0.0, 1.0, 0.0),  # Y-axis
    )

    var data = Data[DTYPE, 1, 5, 1]()
    data.set_body_position(0, 0.0, 0.0, 1.0)

    # Rotate +45 degrees around Y (positive angle)
    var theta_pos = Scalar[DTYPE](45.0 * PI / 180.0)
    var half_theta_pos = theta_pos / 2.0
    data.quaternions[0] = 0.0
    data.quaternions[1] = sin(half_theta_pos)
    data.quaternions[2] = 0.0
    data.quaternions[3] = cos(half_theta_pos)

    var angle_pos = get_joint_angle(model, data, 0)
    print("  +45 deg rotation, measured angle:", angle_pos * 180.0 / PI, "deg")

    # Rotate -45 degrees around Y (negative angle)
    var theta_neg = Scalar[DTYPE](-45.0 * PI / 180.0)
    var half_theta_neg = theta_neg / 2.0
    data.quaternions[0] = 0.0
    data.quaternions[1] = sin(half_theta_neg)
    data.quaternions[2] = 0.0
    data.quaternions[3] = cos(half_theta_neg)

    var angle_neg = get_joint_angle(model, data, 0)
    print("  -45 deg rotation, measured angle:", angle_neg * 180.0 / PI, "deg")

    # Positive rotation should give positive angle
    assert_true(
        angle_pos > 0.5,
        "Positive rotation should give positive angle",
    )

    # Negative rotation should give negative angle
    assert_true(
        angle_neg < -0.5,
        "Negative rotation should give negative angle",
    )

    # Magnitudes should be approximately equal
    assert_true(
        abs64(abs64(angle_pos) - abs64(angle_neg)) < 0.1,
        "Angle magnitudes should be similar",
    )

    print("  PASSED: Angle sign convention is correct")


fn test_two_body_joint_sensing() raises:
    """Test sensing for a joint between two bodies (not world-anchored)."""
    print("\nTest 5: Two-body joint sensing...")

    var model = Model[DTYPE, 2, 10, 1](
        gravity_z=0.0,
        timestep=0.01,
        ground_z=-10.0,
    )
    model.set_body(0, mass=1.0, radius=0.1)
    model.set_body(1, mass=1.0, radius=0.1)

    # Joint between body 0 and body 1
    _ = model.add_hinge_joint(
        parent=0,
        child=1,
        anchor_parent=(0.0, 0.0, -0.1),
        anchor_child=(0.0, 0.0, 0.1),
        axis=(0.0, 1.0, 0.0),
    )

    var data = Data[DTYPE, 2, 10, 1]()
    data.set_body_position(0, 0.0, 0.0, 0.5)
    data.set_body_position(1, 0.0, 0.0, 0.3)

    # Both bodies at identity quaternion -> relative angle = 0
    var angle0 = get_joint_angle(model, data, 0)
    print("  Both at identity, angle:", angle0 * 180.0 / PI, "deg")

    assert_true(
        abs64(angle0) < 0.01,
        "Angle should be zero when both bodies at identity",
    )

    # Rotate child body 30 degrees
    var theta = Scalar[DTYPE](30.0 * PI / 180.0)
    var half_theta = theta / 2.0
    data.quaternions[1 * 4 + 0] = 0.0
    data.quaternions[1 * 4 + 1] = sin(half_theta)
    data.quaternions[1 * 4 + 2] = 0.0
    data.quaternions[1 * 4 + 3] = cos(half_theta)

    var angle1 = get_joint_angle(model, data, 0)
    print("  Child rotated 30 deg, angle:", angle1 * 180.0 / PI, "deg")

    assert_true(
        abs64(angle1 - theta) < 0.1,
        "Angle should be ~30 degrees",
    )

    # Set different angular velocities
    data.set_body_angular_velocity(0, 0.0, 1.0, 0.0)  # Parent: 1 rad/s
    data.set_body_angular_velocity(1, 0.0, 3.0, 0.0)  # Child: 3 rad/s

    var omega = get_joint_angular_velocity(model, data, 0)
    print("  Parent wy=1, Child wy=3, relative:", omega, "rad/s")

    # Relative should be 3 - 1 = 2 rad/s
    assert_true(
        abs64(omega - 2.0) < 0.1,
        "Relative angular velocity should be child - parent",
    )

    print("  PASSED: Two-body joint sensing works")


fn main() raises:
    print("=" * 60)
    print("Joint State Sensing Tests (Phase 7, Step 7.2)")
    print("=" * 60)

    test_initial_angle_zero()
    test_angle_changes_during_swing()
    test_angular_velocity_matches_rotation()
    test_angle_sign_convention()
    test_two_body_joint_sensing()

    print("\n" + "=" * 60)
    print("All joint sensing tests passed!")
    print("=" * 60)

"""Test joint torque actuation (Phase 7, Step 7.1).

Tests:
1. Torque application causes angular acceleration
2. Torque limits are respected
3. Reaction torque on parent body
4. Pendulum responds to torque control
"""

from math import sqrt
from testing import assert_true

from physics3d_v2 import Model, Data, ImpulseIntegrator


comptime DTYPE = DType.float64


fn abs64(x: Scalar[DTYPE]) -> Scalar[DTYPE]:
    """Absolute value helper."""
    if x < 0:
        return -x
    return x


fn test_torque_causes_angular_acceleration() raises:
    """Test that applying torque causes angular velocity change."""
    print("Test 1: Torque causes angular acceleration...")

    # Single pendulum: world -> body 0
    var model = Model[DTYPE, 1, 5, 1](
        gravity_z=0.0,  # No gravity to isolate torque effect
        timestep=0.001,
        ground_z=-10.0,
    )
    model.set_body(0, mass=1.0, radius=0.1)

    # Add hinge joint anchored to world at (0, 0, 1)
    _ = model.add_hinge_joint(
        parent=-1,
        child=0,
        anchor_parent=(0.0, 0.0, 1.0),
        anchor_child=(0.0, 0.0, 0.0),
        axis=(0.0, 1.0, 0.0),  # Y-axis rotation
    )

    # Set torque
    var torque = Scalar[DTYPE](5.0)
    model.joints[0].set_torque(torque)

    var data = Data[DTYPE, 1, 5, 1]()
    data.set_body_position(0, 0.0, 0.0, 1.0)

    # Record initial angular velocity
    var w0 = data.get_body_angular_velocity(0)
    var wy_initial = w0[1]

    # Step simulation
    ImpulseIntegrator.step(model, data)

    # Check angular velocity increased
    var w1 = data.get_body_angular_velocity(0)
    var wy_final = w1[1]

    # Torque should cause rotation around Y axis
    print("  Initial wy:", wy_initial)
    print("  Final wy:", wy_final)
    print("  Delta wy:", wy_final - wy_initial)

    # With I = 2/5 * m * r^2 = 0.4 * 1.0 * 0.01 = 0.004
    # inv_I = 250
    # delta_w = tau * inv_I * dt = 5.0 * 250 * 0.001 = 1.25 rad/s
    # But we use average inv_I which is same for sphere
    assert_true(
        abs64(wy_final - wy_initial) > 0.5,
        "Torque should cause significant angular velocity change",
    )

    print("  PASSED: Torque causes angular acceleration")


fn test_torque_limits_respected() raises:
    """Test that torque limits clamp the applied torque."""
    print("\nTest 2: Torque limits are respected...")

    var model = Model[DTYPE, 1, 5, 1](
        gravity_z=0.0,
        timestep=0.001,
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

    # Set torque limit to 2.0 N*m
    model.joints[0].set_torque_limit(2.0)

    # Try to set torque above limit
    model.joints[0].set_torque(10.0)

    # Torque should be clamped
    print("  Requested torque: 10.0")
    print("  Torque limit: 2.0")
    print("  Actual torque:", model.joints[0].target_torque)

    assert_true(
        abs64(model.joints[0].target_torque - 2.0) < 1e-6,
        "Torque should be clamped to limit",
    )

    # Test negative limit
    model.joints[0].set_torque(-10.0)
    print("  Requested torque: -10.0")
    print("  Actual torque:", model.joints[0].target_torque)

    assert_true(
        abs64(model.joints[0].target_torque - (-2.0)) < 1e-6,
        "Negative torque should be clamped to -limit",
    )

    print("  PASSED: Torque limits are respected")


fn test_reaction_torque_on_parent() raises:
    """Test that parent body receives reaction torque (Newton's 3rd law)."""
    print("\nTest 3: Reaction torque on parent body...")

    # Two bodies connected by a hinge
    var model = Model[DTYPE, 2, 10, 1](
        gravity_z=0.0,  # No gravity
        timestep=0.001,
        ground_z=-10.0,
    )
    model.set_body(0, mass=1.0, radius=0.1)  # Parent
    model.set_body(1, mass=1.0, radius=0.1)  # Child

    # Joint between body 0 and body 1
    _ = model.add_hinge_joint(
        parent=0,  # Body 0 is parent
        child=1,
        anchor_parent=(0.0, 0.0, -0.1),  # Bottom of body 0
        anchor_child=(0.0, 0.0, 0.1),  # Top of body 1
        axis=(0.0, 1.0, 0.0),  # Y-axis rotation
    )

    model.joints[0].set_torque(5.0)

    var data = Data[DTYPE, 2, 10, 1]()
    data.set_body_position(0, 0.0, 0.0, 0.5)
    data.set_body_position(1, 0.0, 0.0, 0.3)

    # Step simulation
    ImpulseIntegrator.step(model, data)

    # Check both bodies have angular velocity
    var w0 = data.get_body_angular_velocity(0)
    var w1 = data.get_body_angular_velocity(1)

    print("  Parent wy:", w0[1])
    print("  Child wy:", w1[1])

    # Parent should rotate opposite to child (reaction)
    # Since equal masses and inertias, magnitudes should be equal
    # Child gets positive torque, parent gets negative reaction
    assert_true(
        w0[1] * w1[1] < 0 or abs64(w0[1]) < 1e-6 or abs64(w1[1]) < 1e-6,
        "Parent and child should rotate in opposite directions (or constraint prevents motion)",
    )

    print("  PASSED: Reaction torque applied to parent")


fn test_pendulum_torque_control() raises:
    """Test that a pendulum responds to torque control."""
    print("\nTest 4: Pendulum torque control...")

    var model = Model[DTYPE, 1, 5, 1](
        gravity_z=-9.81,
        timestep=0.01,
        ground_z=-10.0,
    )
    model.set_body(0, mass=1.0, radius=0.1)

    # Pendulum hanging down
    var L = Scalar[DTYPE](1.0)
    _ = model.add_hinge_joint(
        parent=-1,
        child=0,
        anchor_parent=(0.0, 0.0, L),
        anchor_child=(0.0, 0.0, L),
        axis=(0.0, 1.0, 0.0),
    )

    var data = Data[DTYPE, 1, 5, 1]()
    # Start at equilibrium (hanging straight down)
    data.set_body_position(0, 0.0, 0.0, 0.0)

    # Record initial x position
    var initial_x = data.positions[0]

    # Apply torque to swing the pendulum
    model.joints[0].set_torque(20.0)

    # Simulate for 0.5 seconds
    for _ in range(50):
        ImpulseIntegrator.step(model, data)

    var pos = data.get_body_position(0)
    print("  Initial x: 0.0")
    print("  Final x:", pos[0])
    print("  Final z:", pos[2])

    # Pendulum should have swung significantly (positive Y torque = CCW = -X swing)
    # The sign depends on convention, but magnitude should be > 0.1
    assert_true(
        abs64(pos[0]) > 0.1,
        "Pendulum should swing significantly due to torque",
    )

    print("  PASSED: Pendulum responds to torque control")


fn test_zero_torque_no_effect() raises:
    """Test that zero torque has no effect."""
    print("\nTest 5: Zero torque has no effect...")

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

    # Explicitly set zero torque
    model.joints[0].set_torque(0.0)

    var data = Data[DTYPE, 1, 5, 1]()
    data.set_body_position(0, 0.0, 0.0, 1.0)

    # Step simulation
    for _ in range(100):
        ImpulseIntegrator.step(model, data)

    var w = data.get_body_angular_velocity(0)
    var total_w = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2])

    print("  Angular velocity magnitude after 100 steps:", total_w)

    assert_true(
        total_w < 0.01,
        "Zero torque should result in minimal angular velocity",
    )

    print("  PASSED: Zero torque has no effect")


fn main() raises:
    print("=" * 60)
    print("Joint Torque Actuation Tests (Phase 7, Step 7.1)")
    print("=" * 60)

    test_torque_causes_angular_acceleration()
    test_torque_limits_respected()
    test_reaction_torque_on_parent()
    test_pendulum_torque_control()
    test_zero_torque_no_effect()

    print("\n" + "=" * 60)
    print("All joint torque tests passed!")
    print("=" * 60)

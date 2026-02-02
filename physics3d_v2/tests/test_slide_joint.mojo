"""Phase 11b Validation: Slide (Prismatic) Joint Test.

Tests slide joints that allow 1 DOF translation along a specified axis.

Slide Joint Constraints:
- 2 perpendicular position constraints (no movement perpendicular to axis)
- 3 angular constraints (no rotation allowed)
- 1 free DOF: translation along axis

Expected behavior:
- Body moves only along the slide axis
- No drift in perpendicular directions
- No rotation of the body
- Position/velocity sensing accuracy

Run with:
    cd mojo-rl
    pixi run mojo run physics3d_v2/tests/test_slide_joint.mojo
"""

from math import sqrt, sin, cos
from physics3d_v2.types import Model, Data
from physics3d_v2.integrator import PGSIntegrator
from physics3d_v2.joints import get_slide_joint_position, get_slide_joint_velocity

# Configuration
comptime NUM_BODIES: Int = 1
comptime MAX_CONTACTS: Int = 5
comptime MAX_JOINTS: Int = 0  # No hinge joints
comptime MAX_SLIDE_JOINTS: Int = 1
comptime DTYPE = DType.float64

# Physics constants
comptime G: Float64 = 9.81


fn abs_val(x: Float64) -> Float64:
    """Absolute value."""
    if x < 0:
        return -x
    return x


fn max_val(a: Float64, b: Float64) -> Float64:
    """Maximum of two values."""
    if a > b:
        return a
    return b


fn test_slide_motion_x_axis() -> Bool:
    """Test that a body with X-axis slide joint only moves along X.

    Setup: Body at (1, 0, 0.5) with slide joint along X-axis
    Expected: Body falls under gravity but stays on X-axis (Y=0, Z decreases)
    Wait - with a slide joint along X, gravity (Z direction) is perpendicular,
    so the body should NOT fall. Let's test with gravity along X instead,
    or just verify that perpendicular motion is constrained.

    Actually, for a slide joint along X:
    - Free motion: X
    - Constrained: Y, Z, and all rotations

    With gravity in -Z, the body cannot fall because Z is constrained.
    """
    print("=" * 60)
    print("Phase 11b Validation: Slide Joint X-Axis Motion Test")
    print("=" * 60)

    var mass: Float64 = 1.0
    var radius: Float64 = 0.1
    var dt: Float64 = 0.001
    var initial_x: Float64 = 1.0
    var initial_z: Float64 = 0.5

    print("\nSetup:")
    print("  Mass:", mass, "kg")
    print("  Initial position: (", initial_x, ", 0,", initial_z, ")")
    print("  Slide axis: X (1, 0, 0)")
    print("  Gravity: -Z (perpendicular to slide axis)")

    # Create model with slide joint
    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS](
        gravity_z=Scalar[DTYPE](-G),
        timestep=Scalar[DTYPE](dt),
        ground_z=Scalar[DTYPE](-10.0),  # Ground far below
        restitution=Scalar[DTYPE](0.0),
    )
    model.set_body(0, mass=Scalar[DTYPE](mass), radius=Scalar[DTYPE](radius))

    # Add slide joint along X-axis anchored to world
    model.add_slide_joint(
        parent=-1,  # World anchor
        child=0,
        anchor_parent=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](initial_z)),
        anchor_child=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](0.0)),
        axis=(Scalar[DTYPE](1.0), Scalar[DTYPE](0.0), Scalar[DTYPE](0.0)),  # X-axis
    )

    # Initialize data
    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS]()
    data.set_body_position(0, Scalar[DTYPE](initial_x), Scalar[DTYPE](0.0), Scalar[DTYPE](initial_z))
    data.set_body_velocity(0, Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](0.0))

    # Simulate
    var max_steps = 1000
    var max_y_drift: Float64 = 0.0
    var max_z_drift: Float64 = 0.0

    print("\nSimulating", max_steps, "steps...")

    for step in range(max_steps):
        PGSIntegrator.step(model, data)

        var x = Float64(data.positions[0])
        var y = Float64(data.positions[1])
        var z = Float64(data.positions[2])

        var y_drift = abs_val(y - 0.0)
        var z_drift = abs_val(z - initial_z)

        max_y_drift = max_val(max_y_drift, y_drift)
        max_z_drift = max_val(max_z_drift, z_drift)

        if step % 200 == 0:
            var t = Float64(step) * dt
            print("  t =", t, "s: pos = (", x, ",", y, ",", z, ")")

    print("\nResults:")
    print("  Max Y drift:", max_y_drift * 1000.0, "mm")
    print("  Max Z drift:", max_z_drift * 1000.0, "mm")

    # Perpendicular motion should be constrained to within 10mm
    var passed = max_y_drift < 0.01 and max_z_drift < 0.01

    print()
    if passed:
        print("PASSED: Perpendicular motion constrained within 10mm")
    else:
        print("FAILED: Perpendicular drift exceeds 10mm")

    print("=" * 60)
    return passed


fn test_slide_rotation_locked() -> Bool:
    """Test that slide joint locks all rotations.

    Setup: Body with initial angular velocity
    Expected: Angular velocity should be damped to zero
    """
    print("\n")
    print("=" * 60)
    print("Phase 11b Validation: Slide Joint Rotation Lock Test")
    print("=" * 60)

    var mass: Float64 = 1.0
    var radius: Float64 = 0.1
    var dt: Float64 = 0.001
    var initial_x: Float64 = 0.0
    var initial_z: Float64 = 0.5

    print("\nSetup:")
    print("  Initial angular velocity: (1, 0.5, 0.3) rad/s")
    print("  Slide axis: X")
    print("  Expected: Angular velocity damped to zero")

    # Create model
    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS](
        gravity_z=Scalar[DTYPE](0.0),  # No gravity
        timestep=Scalar[DTYPE](dt),
        ground_z=Scalar[DTYPE](-10.0),
        restitution=Scalar[DTYPE](0.0),
    )
    model.set_body(0, mass=Scalar[DTYPE](mass), radius=Scalar[DTYPE](radius))

    model.add_slide_joint(
        parent=-1,
        child=0,
        anchor_parent=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](initial_z)),
        anchor_child=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](0.0)),
        axis=(Scalar[DTYPE](1.0), Scalar[DTYPE](0.0), Scalar[DTYPE](0.0)),
    )

    # Initialize with angular velocity
    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS]()
    data.set_body_position(0, Scalar[DTYPE](initial_x), Scalar[DTYPE](0.0), Scalar[DTYPE](initial_z))
    data.set_body_velocity(0, Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](0.0))
    data.angular_velocities[0] = Scalar[DTYPE](1.0)   # wx
    data.angular_velocities[1] = Scalar[DTYPE](0.5)   # wy
    data.angular_velocities[2] = Scalar[DTYPE](0.3)   # wz

    # Simulate
    var max_steps = 500

    print("\nSimulating", max_steps, "steps...")

    for step in range(max_steps):
        PGSIntegrator.step(model, data)

        if step % 100 == 0:
            var wx = Float64(data.angular_velocities[0])
            var wy = Float64(data.angular_velocities[1])
            var wz = Float64(data.angular_velocities[2])
            var w_mag = sqrt(wx * wx + wy * wy + wz * wz)
            var t = Float64(step) * dt
            print("  t =", t, "s: |omega| =", w_mag, "rad/s")

    # Check final angular velocity
    var wx = Float64(data.angular_velocities[0])
    var wy = Float64(data.angular_velocities[1])
    var wz = Float64(data.angular_velocities[2])
    var final_w_mag = sqrt(wx * wx + wy * wy + wz * wz)

    print("\nResults:")
    print("  Initial angular velocity magnitude:", sqrt(1.0 + 0.25 + 0.09), "rad/s")
    print("  Final angular velocity magnitude:", final_w_mag, "rad/s")

    # Angular velocity should be significantly reduced
    var passed = final_w_mag < 0.1  # Less than 0.1 rad/s

    print()
    if passed:
        print("PASSED: Angular velocity damped to < 0.1 rad/s")
    else:
        print("FAILED: Angular velocity not properly constrained")

    print("=" * 60)
    return passed


fn test_slide_position_sensing() -> Bool:
    """Test slide joint position sensing accuracy.

    Setup: Body at various X positions with X-axis slide joint
    Expected: get_slide_joint_position returns correct displacement
    """
    print("\n")
    print("=" * 60)
    print("Phase 11b Validation: Slide Joint Position Sensing Test")
    print("=" * 60)

    var mass: Float64 = 1.0
    var radius: Float64 = 0.1
    var dt: Float64 = 0.001
    var anchor_z: Float64 = 0.5

    print("\nSetup:")
    print("  Anchor point: (0, 0,", anchor_z, ")")
    print("  Slide axis: X")

    # Create model
    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS](
        gravity_z=Scalar[DTYPE](0.0),  # No gravity
        timestep=Scalar[DTYPE](dt),
        ground_z=Scalar[DTYPE](-10.0),
        restitution=Scalar[DTYPE](0.0),
    )
    model.set_body(0, mass=Scalar[DTYPE](mass), radius=Scalar[DTYPE](radius))

    model.add_slide_joint(
        parent=-1,
        child=0,
        anchor_parent=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](anchor_z)),
        anchor_child=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](0.0)),
        axis=(Scalar[DTYPE](1.0), Scalar[DTYPE](0.0), Scalar[DTYPE](0.0)),
    )

    # Test at various positions
    var test_positions = List[Float64]()
    test_positions.append(0.0)
    test_positions.append(1.0)
    test_positions.append(-0.5)
    test_positions.append(2.5)

    var max_error: Float64 = 0.0

    print("\nTesting position sensing:")

    for i in range(len(test_positions)):
        var test_x = test_positions[i]

        var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS]()
        data.set_body_position(0, Scalar[DTYPE](test_x), Scalar[DTYPE](0.0), Scalar[DTYPE](anchor_z))

        var sensed_pos = Float64(get_slide_joint_position(model, data, 0))
        var expected_pos = test_x  # Displacement from anchor along X
        var error = abs_val(sensed_pos - expected_pos)
        max_error = max_val(max_error, error)

        print("  Body at x =", test_x, ": sensed =", sensed_pos, ", expected =", expected_pos, ", error =", error)

    print("\nResults:")
    print("  Max sensing error:", max_error * 1000.0, "mm")

    var passed = max_error < 0.001  # Within 1mm

    print()
    if passed:
        print("PASSED: Position sensing accurate within 1mm")
    else:
        print("FAILED: Position sensing error exceeds 1mm")

    print("=" * 60)
    return passed


fn test_slide_velocity_sensing() -> Bool:
    """Test slide joint velocity sensing accuracy.

    Setup: Body moving along X-axis with X-axis slide joint
    Expected: get_slide_joint_velocity returns correct velocity
    """
    print("\n")
    print("=" * 60)
    print("Phase 11b Validation: Slide Joint Velocity Sensing Test")
    print("=" * 60)

    var mass: Float64 = 1.0
    var radius: Float64 = 0.1
    var dt: Float64 = 0.001
    var anchor_z: Float64 = 0.5

    print("\nSetup:")
    print("  Anchor point: (0, 0,", anchor_z, ")")
    print("  Slide axis: X")

    # Create model
    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS](
        gravity_z=Scalar[DTYPE](0.0),  # No gravity
        timestep=Scalar[DTYPE](dt),
        ground_z=Scalar[DTYPE](-10.0),
        restitution=Scalar[DTYPE](0.0),
    )
    model.set_body(0, mass=Scalar[DTYPE](mass), radius=Scalar[DTYPE](radius))

    model.add_slide_joint(
        parent=-1,
        child=0,
        anchor_parent=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](anchor_z)),
        anchor_child=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](0.0)),
        axis=(Scalar[DTYPE](1.0), Scalar[DTYPE](0.0), Scalar[DTYPE](0.0)),
    )

    # Test at various velocities
    var test_velocities = List[Float64]()
    test_velocities.append(0.0)
    test_velocities.append(1.0)
    test_velocities.append(-2.0)
    test_velocities.append(0.5)

    var max_error: Float64 = 0.0

    print("\nTesting velocity sensing:")

    for i in range(len(test_velocities)):
        var test_vx = test_velocities[i]

        var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS]()
        data.set_body_position(0, Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](anchor_z))
        data.set_body_velocity(0, Scalar[DTYPE](test_vx), Scalar[DTYPE](0.0), Scalar[DTYPE](0.0))

        var sensed_vel = Float64(get_slide_joint_velocity(model, data, 0))
        var expected_vel = test_vx  # Velocity along X
        var error = abs_val(sensed_vel - expected_vel)
        max_error = max_val(max_error, error)

        print("  Body vx =", test_vx, "m/s: sensed =", sensed_vel, ", expected =", expected_vel, ", error =", error)

    print("\nResults:")
    print("  Max sensing error:", max_error, "m/s")

    var passed = max_error < 0.001  # Within 1mm/s

    print()
    if passed:
        print("PASSED: Velocity sensing accurate within 1mm/s")
    else:
        print("FAILED: Velocity sensing error exceeds 1mm/s")

    print("=" * 60)
    return passed


fn test_slide_with_force() -> Bool:
    """Test slide joint with applied force along axis.

    Setup: Body with slide joint along X, apply force in X direction
    Expected: Body accelerates along X, stays at Y=0, Z=anchor_z
    """
    print("\n")
    print("=" * 60)
    print("Phase 11b Validation: Slide Joint Force Actuation Test")
    print("=" * 60)

    var mass: Float64 = 1.0
    var radius: Float64 = 0.1
    var dt: Float64 = 0.001
    var anchor_z: Float64 = 0.5
    var applied_force: Float64 = 10.0  # N

    print("\nSetup:")
    print("  Mass:", mass, "kg")
    print("  Applied force:", applied_force, "N along X")
    print("  Expected acceleration:", applied_force / mass, "m/s^2")

    # Create model
    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS](
        gravity_z=Scalar[DTYPE](0.0),  # No gravity
        timestep=Scalar[DTYPE](dt),
        ground_z=Scalar[DTYPE](-10.0),
        restitution=Scalar[DTYPE](0.0),
    )
    model.set_body(0, mass=Scalar[DTYPE](mass), radius=Scalar[DTYPE](radius))

    var joint_idx = model.add_slide_joint(
        parent=-1,
        child=0,
        anchor_parent=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](anchor_z)),
        anchor_child=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](0.0)),
        axis=(Scalar[DTYPE](1.0), Scalar[DTYPE](0.0), Scalar[DTYPE](0.0)),
    )

    # Set target force
    model.slide_joints[joint_idx].target_force = Scalar[DTYPE](applied_force)

    # Initialize
    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS]()
    data.set_body_position(0, Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](anchor_z))
    data.set_body_velocity(0, Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](0.0))

    # Simulate
    var max_steps = 100
    var max_y_drift: Float64 = 0.0
    var max_z_drift: Float64 = 0.0

    print("\nSimulating", max_steps, "steps...")

    for step in range(max_steps):
        PGSIntegrator.step(model, data)

        var x = Float64(data.positions[0])
        var y = Float64(data.positions[1])
        var z = Float64(data.positions[2])
        var vx = Float64(data.velocities[0])

        max_y_drift = max_val(max_y_drift, abs_val(y))
        max_z_drift = max_val(max_z_drift, abs_val(z - anchor_z))

        if step % 20 == 0:
            var t = Float64(step) * dt
            print("  t =", t, "s: x =", x, "m, vx =", vx, "m/s")

    # Final state
    var final_x = Float64(data.positions[0])
    var final_vx = Float64(data.velocities[0])
    var sim_time = Float64(max_steps) * dt

    # Expected values (kinematics: v = a*t, x = 0.5*a*t^2)
    var expected_a = applied_force / mass
    var expected_vx = expected_a * sim_time
    var expected_x = 0.5 * expected_a * sim_time * sim_time

    print("\nResults:")
    print("  Final x:", final_x, "m (expected:", expected_x, "m)")
    print("  Final vx:", final_vx, "m/s (expected:", expected_vx, "m/s)")
    print("  Max Y drift:", max_y_drift * 1000.0, "mm")
    print("  Max Z drift:", max_z_drift * 1000.0, "mm")

    # Check position and velocity (with 20% tolerance for solver effects)
    var x_error = abs_val(final_x - expected_x) / expected_x
    var vx_error = abs_val(final_vx - expected_vx) / expected_vx

    var passed = x_error < 0.2 and vx_error < 0.2 and max_y_drift < 0.01 and max_z_drift < 0.01

    print()
    if passed:
        print("PASSED: Force actuation working correctly")
    else:
        print("FAILED: Force actuation not working as expected")
        if x_error >= 0.2:
            print("  - Position error:", x_error * 100.0, "%")
        if vx_error >= 0.2:
            print("  - Velocity error:", vx_error * 100.0, "%")
        if max_y_drift >= 0.01:
            print("  - Y drift too large")
        if max_z_drift >= 0.01:
            print("  - Z drift too large")

    print("=" * 60)
    return passed


fn main():
    """Run all slide joint validation tests."""
    print("\n")
    print("=" * 60)
    print("    PHYSICS3D v2 - Slide Joint Validation Suite    ")
    print("=" * 60)
    print()

    var all_passed = True

    # Test perpendicular motion constraint
    if not test_slide_motion_x_axis():
        all_passed = False

    # Test rotation lock
    if not test_slide_rotation_locked():
        all_passed = False

    # Test position sensing
    if not test_slide_position_sensing():
        all_passed = False

    # Test velocity sensing
    if not test_slide_velocity_sensing():
        all_passed = False

    # Test force actuation
    if not test_slide_with_force():
        all_passed = False

    print("\n")
    print("=" * 60)
    if all_passed:
        print("              ALL SLIDE JOINT TESTS PASSED                 ")
    else:
        print("              SOME TESTS FAILED                            ")
    print("=" * 60)
    print()

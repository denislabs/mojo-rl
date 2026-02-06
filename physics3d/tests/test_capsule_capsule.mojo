"""Phase 8 Test: Capsule-Capsule Collision.

Tests capsule-capsule collision detection and response:
1. Parallel capsules colliding (cylinder-cylinder contact)
2. Perpendicular capsules (cross contact)
3. Capsule endpoints colliding (cap-cap contact)
4. Two capsules falling and bouncing
"""

from math import sqrt, sin, cos, pi
from physics3d import Model, Data, ImpulseIntegrator
from physics3d.gpu.constants import GEOM_CAPSULE


fn test_parallel_capsules() -> Bool:
    """Test two parallel capsules colliding.

    Two horizontal capsules side by side should collide along their
    cylindrical surfaces.
    """
    print("Test: Parallel capsules collision")

    comptime DTYPE = DType.float64
    comptime NUM_BODIES = 2
    comptime MAX_CONTACTS = 10

    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=0.0,  # No gravity
        timestep=0.001,
        ground_z=-10.0,
        restitution=0.5,
        friction=0.0,
    )

    var radius: Scalar[DTYPE] = 0.1
    var half_len: Scalar[DTYPE] = 0.3

    # Both capsules horizontal along X
    model.set_body_capsule(0, mass=1.0, radius=radius, half_length=half_len)
    model.set_body_capsule(1, mass=1.0, radius=radius, half_length=half_len)

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()

    # Rotate both to be horizontal (along X)
    var angle = Scalar[DTYPE](pi) / Scalar[DTYPE](2.0)
    var half_angle = angle / Scalar[DTYPE](2.0)
    var qy = sin(half_angle)
    var qw = cos(half_angle)

    # Capsule 0: moving up in Y
    data.set_body_position(0, 0.0, -0.3, 0.5)
    data.quaternions[0 * 4 + 1] = qy
    data.quaternions[0 * 4 + 3] = qw
    data.set_body_velocity(0, 0.0, 1.0, 0.0)

    # Capsule 1: moving down in Y
    data.set_body_position(1, 0.0, 0.3, 0.5)
    data.quaternions[1 * 4 + 1] = qy
    data.quaternions[1 * 4 + 3] = qw
    data.set_body_velocity(1, 0.0, -1.0, 0.0)

    # Simulate collision
    for _ in range(500):
        ImpulseIntegrator.step(model, data)

    var y0 = data.positions[0 * 3 + 1]
    var y1 = data.positions[1 * 3 + 1]
    var vy0 = data.velocities[0 * 3 + 1]
    var vy1 = data.velocities[1 * 3 + 1]

    print("  Capsule 0: y =", y0, ", vy =", vy0)
    print("  Capsule 1: y =", y1, ", vy =", vy1)

    # After collision, they should have bounced (velocities reversed)
    var bounced = (vy0 < 0.0) and (vy1 > 0.0)
    # Minimum separation should be 2*radius = 0.2
    var separation = y1 - y0
    var separated = separation >= 2.0 * radius - 0.01

    print("  Separation:", separation, "(expected >=", 2.0 * radius, ")")

    var passed = bounced and separated
    if passed:
        print("  PASSED")
    else:
        print("  FAILED (bounced:", bounced, ", separated:", separated, ")")
    return passed


fn test_perpendicular_capsules() -> Bool:
    """Test two perpendicular capsules colliding (cross shape).

    One horizontal (X-axis) and one horizontal (Y-axis) capsule crossing.
    """
    print("Test: Perpendicular capsules collision (cross)")

    comptime DTYPE = DType.float64
    comptime NUM_BODIES = 2
    comptime MAX_CONTACTS = 10

    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=0.0,
        timestep=0.001,
        ground_z=-10.0,
        restitution=0.5,
        friction=0.0,
    )

    var radius: Scalar[DTYPE] = 0.1
    var half_len: Scalar[DTYPE] = 0.3

    model.set_body_capsule(0, mass=1.0, radius=radius, half_length=half_len)
    model.set_body_capsule(1, mass=1.0, radius=radius, half_length=half_len)

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()

    # Capsule 0: along X axis (rotate 90 deg around Y)
    var angle_y = Scalar[DTYPE](pi) / Scalar[DTYPE](2.0)
    data.set_body_position(0, 0.0, 0.0, 0.0)
    data.quaternions[0 * 4 + 1] = sin(angle_y / 2.0)
    data.quaternions[0 * 4 + 3] = cos(angle_y / 2.0)

    # Capsule 1: along Y axis (rotate 90 deg around X)
    var angle_x = Scalar[DTYPE](pi) / Scalar[DTYPE](2.0)
    data.set_body_position(1, 0.0, 0.0, 0.5)
    data.quaternions[1 * 4 + 0] = sin(angle_x / 2.0)
    data.quaternions[1 * 4 + 3] = cos(angle_x / 2.0)

    # Capsule 1 moving down
    data.set_body_velocity(1, 0.0, 0.0, -1.0)

    # Simulate collision
    for _ in range(500):
        ImpulseIntegrator.step(model, data)

    var z0 = data.positions[0 * 3 + 2]
    var z1 = data.positions[1 * 3 + 2]
    var vz1 = data.velocities[1 * 3 + 2]

    print("  Capsule 0 z:", z0)
    print("  Capsule 1 z:", z1, ", vz:", vz1)

    # After collision, capsule 1 should have bounced up
    var bounced = vz1 > 0.0
    # Capsule 1 should be above capsule 0 by at least 2*radius
    var above = z1 > z0 + 2.0 * radius - 0.01

    var passed = bounced or above  # Either bounced or settled above
    if passed:
        print("  PASSED")
    else:
        print("  FAILED")
    return passed


fn test_capsule_cap_collision() -> Bool:
    """Test two capsules colliding at their endpoints (cap-cap).

    Two vertical capsules with their caps approaching.
    """
    print("Test: Capsule cap-cap collision")

    comptime DTYPE = DType.float64
    comptime NUM_BODIES = 2
    comptime MAX_CONTACTS = 10

    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=0.0,
        timestep=0.001,
        ground_z=-10.0,
        restitution=0.5,
        friction=0.0,
    )

    var radius: Scalar[DTYPE] = 0.1
    var half_len: Scalar[DTYPE] = 0.2

    model.set_body_capsule(0, mass=1.0, radius=radius, half_length=half_len)
    model.set_body_capsule(1, mass=1.0, radius=radius, half_length=half_len)

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()

    # Both capsules vertical (default orientation)
    # Capsule 0 at bottom, top cap at z = 0.5 + 0.2 = 0.7
    data.set_body_position(0, 0.0, 0.0, 0.5)
    data.set_body_velocity(0, 0.0, 0.0, 1.0)  # Moving up

    # Capsule 1 at top, bottom cap at z = 1.5 - 0.2 = 1.3
    data.set_body_position(1, 0.0, 0.0, 1.5)
    data.set_body_velocity(1, 0.0, 0.0, -1.0)  # Moving down

    # Initial gap: 1.3 - 0.7 - 2*radius = 0.4

    # Simulate collision
    for _ in range(500):
        ImpulseIntegrator.step(model, data)

    var z0 = data.positions[0 * 3 + 2]
    var z1 = data.positions[1 * 3 + 2]
    var vz0 = data.velocities[0 * 3 + 2]
    var vz1 = data.velocities[1 * 3 + 2]

    print("  Capsule 0: z =", z0, ", vz =", vz0)
    print("  Capsule 1: z =", z1, ", vz =", vz1)

    # After collision, velocities should be reversed
    var bounced = (vz0 < 0.0) and (vz1 > 0.0)
    # Minimum separation: cap0_top to cap1_bottom = 2*(half_len + radius)
    var min_sep = Scalar[DTYPE](2.0) * (half_len + radius)
    var actual_sep = z1 - z0
    var separated = actual_sep >= min_sep - 0.05

    print("  Separation:", actual_sep, "(expected >=", min_sep, ")")

    var passed = bounced and separated
    if passed:
        print("  PASSED")
    else:
        print("  FAILED (bounced:", bounced, ", separated:", separated, ")")
    return passed


fn test_capsules_fall_and_stack() -> Bool:
    """Test two capsules falling and stacking with gravity.

    Two vertical capsules should fall and stack on ground.
    """
    print("Test: Capsules falling and stacking")

    comptime DTYPE = DType.float64
    comptime NUM_BODIES = 2
    comptime MAX_CONTACTS = 10

    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81,
        timestep=0.001,
        ground_z=0.0,
        restitution=0.0,  # No bounce for stable stacking
        friction=0.5,
    )

    var radius: Scalar[DTYPE] = 0.1
    var half_len: Scalar[DTYPE] = 0.15

    model.set_body_capsule(0, mass=1.0, radius=radius, half_length=half_len)
    model.set_body_capsule(1, mass=1.0, radius=radius, half_length=half_len)

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()

    # Capsule 0 lower (will hit ground first)
    data.set_body_position(0, 0.0, 0.0, 1.0)
    # Capsule 1 higher (will land on capsule 0)
    data.set_body_position(1, 0.0, 0.0, 2.0)

    # Expected final positions:
    # Capsule 0: z = half_len + radius = 0.25
    # Capsule 1: z = capsule0_top + half_len + radius
    #          = (0.25 + half_len + radius) + half_len + radius
    #          = 0.25 + 0.25 + 0.25 = 0.75
    # Wait, let me recalculate:
    # Capsule 0 center z = half_len + radius = 0.15 + 0.1 = 0.25
    # Capsule 0 top cap center = 0.25 + half_len = 0.4
    # Capsule 1 bottom cap center = capsule1_z - half_len
    # Contact at cap0_top surface to cap1_bottom surface:
    # cap0_top_surface = 0.4 + radius = 0.5
    # cap1_bottom_surface = cap1_z - half_len - radius
    # For contact: cap1_z - 0.15 - 0.1 = 0.5 => cap1_z = 0.75

    var expected_z0 = half_len + radius  # 0.25
    var expected_z1 = (
        Scalar[DTYPE](2.0) * (half_len + radius) + half_len + radius
    )  # 0.75

    # Simulate for long enough to settle
    for _ in range(2000):
        ImpulseIntegrator.step(model, data)

    var z0 = data.positions[0 * 3 + 2]
    var z1 = data.positions[1 * 3 + 2]

    print("  Capsule 0: z =", z0, "(expected:", expected_z0, ")")
    print("  Capsule 1: z =", z1, "(expected:", expected_z1, ")")

    var error0 = abs(z0 - expected_z0)
    var error1 = abs(z1 - expected_z1)

    # Allow 2cm tolerance for stacking
    var passed = (error0 < 0.02) and (error1 < 0.02)
    if passed:
        print("  PASSED")
    else:
        print("  FAILED (error0:", error0, ", error1:", error1, ")")
    return passed


fn main():
    print("=" * 60)
    print("Phase 8: Capsule-Capsule Collision Tests")
    print("=" * 60)

    var passed = 0
    var total = 4

    if test_parallel_capsules():
        passed += 1
    print()

    if test_perpendicular_capsules():
        passed += 1
    print()

    if test_capsule_cap_collision():
        passed += 1
    print()

    if test_capsules_fall_and_stack():
        passed += 1
    print()

    print("=" * 60)
    print("Results:", passed, "/", total, "tests passed")
    if passed == total:
        print("All capsule-capsule tests PASSED!")
    else:
        print("Some tests FAILED")
    print("=" * 60)

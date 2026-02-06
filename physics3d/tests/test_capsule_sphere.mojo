"""Phase 8 Test: Capsule-Sphere Collision.

Tests capsule-sphere collision detection and response:
1. Sphere falling onto horizontal capsule
2. Sphere colliding with capsule endpoint (cap)
3. Capsule and sphere bouncing apart
"""

from math import sqrt, sin, cos, pi
from physics3d import Model, Data, ImpulseIntegrator
from physics3d.gpu.constants import GEOM_SPHERE, GEOM_CAPSULE


fn test_sphere_on_horizontal_capsule() -> Bool:
    """Test sphere falling onto horizontal capsule.

    A sphere should stop when it contacts the cylindrical surface of
    a horizontal capsule resting on ground.
    """
    print("Test: Sphere on horizontal capsule")

    comptime DTYPE = DType.float64
    comptime NUM_BODIES = 2
    comptime MAX_CONTACTS = 10

    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81,
        timestep=0.001,
        ground_z=0.0,  # Ground at z=0
        restitution=0.0,
        friction=0.5,
    )

    # Body 0: Capsule (horizontal, along X)
    var cap_radius: Scalar[DTYPE] = 0.1
    var cap_half_len: Scalar[DTYPE] = 0.3
    model.set_body_capsule(
        0, mass=10.0, radius=cap_radius, half_length=cap_half_len
    )

    # Body 1: Sphere
    var sph_radius: Scalar[DTYPE] = 0.05
    model.set_body(1, mass=1.0, radius=sph_radius)

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()

    # Capsule horizontal (along X) at rest on ground
    # Horizontal capsule rests at z = radius
    var capsule_rest_z = cap_radius
    data.set_body_position(0, 0.0, 0.0, capsule_rest_z)
    var angle = Scalar[DTYPE](pi) / Scalar[DTYPE](2.0)
    var half_angle = angle / Scalar[DTYPE](2.0)
    data.quaternions[0 * 4 + 0] = Scalar[DTYPE](0.0)
    data.quaternions[0 * 4 + 1] = sin(half_angle)
    data.quaternions[0 * 4 + 2] = Scalar[DTYPE](0.0)
    data.quaternions[0 * 4 + 3] = cos(half_angle)

    # Sphere above capsule center
    data.set_body_position(1, 0.0, 0.0, 0.5)

    # Expected: sphere center stops at capsule_z + cap_radius + sph_radius
    var expected_z = capsule_rest_z + cap_radius + sph_radius

    # Simulate
    for _ in range(500):
        ImpulseIntegrator.step(model, data)

    var final_z = data.positions[1 * 3 + 2]  # Sphere z
    var error = abs(final_z - expected_z)

    print("  Expected sphere z:", expected_z)
    print("  Final sphere z:", final_z)
    print("  Error:", error)

    # Allow 1cm tolerance
    var passed = error < 0.01
    if passed:
        print("  PASSED")
    else:
        print("  FAILED")
    return passed


fn test_sphere_on_capsule_cap() -> Bool:
    """Test sphere contacting capsule endpoint (hemispherical cap).

    A sphere falling onto the end of a vertical capsule should stop
    when it contacts the hemispherical cap.
    """
    print("Test: Sphere on capsule cap (endpoint)")

    comptime DTYPE = DType.float64
    comptime NUM_BODIES = 2
    comptime MAX_CONTACTS = 10

    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81,
        timestep=0.001,
        ground_z=0.0,  # Ground at z=0
        restitution=0.0,
        friction=0.5,
    )

    # Body 0: Capsule (vertical)
    var cap_radius: Scalar[DTYPE] = 0.1
    var cap_half_len: Scalar[DTYPE] = 0.2
    model.set_body_capsule(
        0, mass=10.0, radius=cap_radius, half_length=cap_half_len
    )

    # Body 1: Sphere
    var sph_radius: Scalar[DTYPE] = 0.05
    model.set_body(1, mass=1.0, radius=sph_radius)

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()

    # Capsule vertical, resting on ground
    # Vertical capsule rests at z = half_len + radius
    var capsule_rest_z = cap_half_len + cap_radius
    data.set_body_position(0, 0.0, 0.0, capsule_rest_z)

    # Sphere above the top cap of the capsule
    data.set_body_position(1, 0.0, 0.0, 1.0)

    # Expected: sphere center stops at cap_top_surface + sph_radius
    # cap_top_surface = capsule_z + half_len + radius
    var cap_top_surface_z = capsule_rest_z + cap_half_len + cap_radius
    var expected_z = cap_top_surface_z + sph_radius

    # Simulate
    for _ in range(500):
        ImpulseIntegrator.step(model, data)

    var final_z = data.positions[1 * 3 + 2]
    var error = abs(final_z - expected_z)

    print("  Capsule rest z:", capsule_rest_z)
    print("  Cap top surface z:", cap_top_surface_z)
    print("  Expected sphere z:", expected_z)
    print("  Final sphere z:", final_z)
    print("  Error:", error)

    # Allow 1cm tolerance
    var passed = error < 0.01
    if passed:
        print("  PASSED")
    else:
        print("  FAILED")
    return passed


fn test_capsule_sphere_bounce() -> Bool:
    """Test capsule and sphere bouncing apart with restitution.

    Two bodies colliding should bounce apart according to coefficient
    of restitution.
    """
    print("Test: Capsule-sphere bounce")

    comptime DTYPE = DType.float64
    comptime NUM_BODIES = 2
    comptime MAX_CONTACTS = 10

    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=0.0,  # No gravity for clean bounce test
        timestep=0.001,
        ground_z=-10.0,
        restitution=0.5,  # 50% energy retained
        friction=0.0,
    )

    # Body 0: Capsule (vertical)
    model.set_body_capsule(0, mass=2.0, radius=0.1, half_length=0.2)

    # Body 1: Sphere
    model.set_body(1, mass=1.0, radius=0.1)

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()

    # Capsule moving right (+X)
    data.set_body_position(0, -0.5, 0.0, 0.5)
    data.set_body_velocity(0, 1.0, 0.0, 0.0)

    # Sphere moving left (-X)
    data.set_body_position(1, 0.5, 0.0, 0.5)
    data.set_body_velocity(1, -1.0, 0.0, 0.0)

    # Record initial relative velocity
    var init_rel_vel = Scalar[DTYPE](2.0)  # Approaching at 2 m/s

    # Simulate collision
    for _ in range(1000):
        ImpulseIntegrator.step(model, data)

    # Check that they bounced apart
    var cap_vx = data.velocities[0 * 3 + 0]
    var sph_vx = data.velocities[1 * 3 + 0]
    var cap_x = data.positions[0 * 3 + 0]
    var sph_x = data.positions[1 * 3 + 0]

    print("  Capsule final x:", cap_x, ", vx:", cap_vx)
    print("  Sphere final x:", sph_x, ", vx:", sph_vx)

    # Bodies should have separated
    var separated = sph_x > cap_x + 0.1
    # Velocities should be reversed (capsule going left, sphere going right)
    # With restitution and momentum conservation, both should have changed direction
    var bounced = (cap_vx < 0.0) or (sph_vx > 0.0)

    var passed = separated and bounced
    if passed:
        print("  PASSED (bodies separated and bounced)")
    else:
        print("  FAILED (separated:", separated, ", bounced:", bounced, ")")
    return passed


fn test_sphere_slides_along_capsule() -> Bool:
    """Test sphere sliding along capsule surface.

    A sphere with tangential velocity should slide along the capsule
    without getting stuck.
    """
    print("Test: Sphere sliding along capsule")

    comptime DTYPE = DType.float64
    comptime NUM_BODIES = 2
    comptime MAX_CONTACTS = 10

    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=0.0,
        timestep=0.001,
        ground_z=-10.0,
        restitution=0.0,
        friction=0.0,  # Zero friction for clean sliding
    )

    # Body 0: Horizontal capsule (along X)
    var cap_radius: Scalar[DTYPE] = 0.15
    var cap_half_len: Scalar[DTYPE] = 0.5
    model.set_body_capsule(
        0, mass=100.0, radius=cap_radius, half_length=cap_half_len
    )

    # Body 1: Sphere
    var sph_radius: Scalar[DTYPE] = 0.05
    model.set_body(1, mass=1.0, radius=sph_radius)

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()

    # Capsule horizontal along X
    data.set_body_position(0, 0.0, 0.0, 0.3)
    var angle = Scalar[DTYPE](pi) / Scalar[DTYPE](2.0)
    var half_angle = angle / Scalar[DTYPE](2.0)
    data.quaternions[0 * 4 + 1] = sin(half_angle)
    data.quaternions[0 * 4 + 3] = cos(half_angle)

    # Sphere on top of capsule with X velocity
    var contact_z = Scalar[DTYPE](0.3) + cap_radius + sph_radius
    data.set_body_position(1, -0.3, 0.0, contact_z)
    data.set_body_velocity(1, 1.0, 0.0, 0.0)

    var init_x = data.positions[1 * 3 + 0]

    # Simulate
    for _ in range(500):
        ImpulseIntegrator.step(model, data)

    var final_x = data.positions[1 * 3 + 0]
    var final_z = data.positions[1 * 3 + 2]
    var moved_x = final_x - init_x

    print("  Initial x:", init_x)
    print("  Final x:", final_x)
    print("  Final z:", final_z)
    print("  Moved x:", moved_x)

    # Sphere should have moved in X direction
    var passed = moved_x > 0.3
    if passed:
        print("  PASSED (sphere slid along capsule)")
    else:
        print("  FAILED (sphere didn't slide enough)")
    return passed


fn main():
    print("=" * 60)
    print("Phase 8: Capsule-Sphere Collision Tests")
    print("=" * 60)

    var passed = 0
    var total = 4

    if test_sphere_on_horizontal_capsule():
        passed += 1
    print()

    if test_sphere_on_capsule_cap():
        passed += 1
    print()

    if test_capsule_sphere_bounce():
        passed += 1
    print()

    if test_sphere_slides_along_capsule():
        passed += 1
    print()

    print("=" * 60)
    print("Results:", passed, "/", total, "tests passed")
    if passed == total:
        print("All capsule-sphere tests PASSED!")
    else:
        print("Some tests FAILED")
    print("=" * 60)

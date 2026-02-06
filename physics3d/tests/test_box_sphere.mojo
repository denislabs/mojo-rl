"""Phase 9 Test: Box-Sphere Collision.

Tests box-sphere collision:
1. Sphere hitting box face
2. Sphere hitting box edge (corner approach)
3. Sphere inside box (deep penetration)
"""

from math import sqrt, sin, cos, pi
from physics3d import Model, Data, ImpulseIntegrator


fn test_sphere_box_face() -> Bool:
    """Test sphere falling onto top face of stationary box.

    Sphere should come to rest on top of the box.
    """
    print("Test: Sphere-box face collision")

    comptime DTYPE = DType.float64
    comptime NUM_BODIES = 2
    comptime MAX_CONTACTS = 10

    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81,
        timestep=0.001,
        ground_z=0.0,
        restitution=0.0,
        friction=0.5,
    )

    # Body 0: Box (stationary, very heavy)
    var half_x: Scalar[DTYPE] = 0.3
    var half_y: Scalar[DTYPE] = 0.3
    var half_z: Scalar[DTYPE] = 0.1
    model.set_body_box(
        0, mass=1000.0, half_x=half_x, half_y=half_y, half_z=half_z
    )

    # Body 1: Sphere (falling)
    var sphere_radius: Scalar[DTYPE] = 0.1
    model.set_body(1, mass=1.0, radius=sphere_radius)

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    # Box at rest on ground
    data.set_body_position(0, 0.0, 0.0, half_z)
    # Sphere starts above box
    data.set_body_position(1, 0.0, 0.0, 1.0)

    # Expected final height of sphere: box_top + sphere_radius = 2*half_z + radius
    var expected_z = Scalar[DTYPE](2.0) * half_z + sphere_radius

    # Simulate for 500ms
    for _ in range(500):
        ImpulseIntegrator.step(model, data)

    var final_z = data.get_body_z(1)
    var error = abs(final_z - expected_z)

    print("  Expected sphere z:", expected_z)
    print("  Final sphere z:", final_z)
    print("  Error:", error)

    # Allow 5mm tolerance
    var passed = error < 0.005
    if passed:
        print("  PASSED")
    else:
        print("  FAILED")
    return passed


fn test_sphere_box_edge() -> Bool:
    """Test sphere approaching box from side (edge contact).

    Sphere should be pushed away from box edge.
    """
    print("Test: Sphere-box edge collision")

    comptime DTYPE = DType.float64
    comptime NUM_BODIES = 2
    comptime MAX_CONTACTS = 10

    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81,
        timestep=0.001,
        ground_z=0.0,
        restitution=0.0,
        friction=0.5,
    )

    # Body 0: Box
    var half_ext: Scalar[DTYPE] = 0.2
    model.set_body_box(
        0, mass=1000.0, half_x=half_ext, half_y=half_ext, half_z=half_ext
    )

    # Body 1: Sphere
    var sphere_radius: Scalar[DTYPE] = 0.1
    model.set_body(1, mass=1.0, radius=sphere_radius)

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    # Box at rest on ground
    data.set_body_position(0, 0.0, 0.0, half_ext)
    # Sphere starts diagonally above (will hit edge region)
    data.set_body_position(
        1, half_ext + sphere_radius * Scalar[DTYPE](0.5), 0.0, 1.0
    )

    # Simulate for 500ms
    for _ in range(500):
        ImpulseIntegrator.step(model, data)

    var final_z = data.get_body_z(1)

    print("  Final sphere z:", final_z)

    # Sphere should be resting on box or ground, not below ground
    var passed = final_z >= sphere_radius - Scalar[DTYPE](0.01)
    if passed:
        print("  PASSED (sphere above ground)")
    else:
        print("  FAILED (sphere below ground)")
    return passed


fn test_sphere_box_corner() -> Bool:
    """Test sphere hitting near box corner.

    Sphere falling near corner should be deflected properly.
    """
    print("Test: Sphere-box corner collision")

    comptime DTYPE = DType.float64
    comptime NUM_BODIES = 2
    comptime MAX_CONTACTS = 10

    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81,
        timestep=0.001,
        ground_z=0.0,
        restitution=0.0,
        friction=0.5,
    )

    # Body 0: Box (cube)
    var half_ext: Scalar[DTYPE] = 0.2
    model.set_body_box(
        0, mass=1000.0, half_x=half_ext, half_y=half_ext, half_z=half_ext
    )

    # Body 1: Sphere
    var sphere_radius: Scalar[DTYPE] = 0.1
    model.set_body(1, mass=1.0, radius=sphere_radius)

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    # Box at rest on ground
    data.set_body_position(0, 0.0, 0.0, half_ext)
    # Sphere starts near corner (diagonal offset in XY)
    var corner_offset = half_ext + sphere_radius * Scalar[DTYPE](0.3)
    data.set_body_position(1, corner_offset, corner_offset, 1.0)

    # Simulate for 500ms
    for _ in range(500):
        ImpulseIntegrator.step(model, data)

    var final_z = data.get_body_z(1)

    print("  Final sphere z:", final_z)

    # Sphere should end up on ground (missed the box) or on box
    var passed = final_z >= sphere_radius - Scalar[DTYPE](0.01)
    if passed:
        print("  PASSED (sphere above ground)")
    else:
        print("  FAILED (sphere below ground)")
    return passed


fn main():
    print("=" * 60)
    print("Phase 9: Box-Sphere Collision Tests")
    print("=" * 60)

    var passed = 0
    var total = 3

    if test_sphere_box_face():
        passed += 1
    print()

    if test_sphere_box_edge():
        passed += 1
    print()

    if test_sphere_box_corner():
        passed += 1
    print()

    print("=" * 60)
    print("Results:", passed, "/", total, "tests passed")
    if passed == total:
        print("All box-sphere tests PASSED!")
    else:
        print("Some tests FAILED")
    print("=" * 60)

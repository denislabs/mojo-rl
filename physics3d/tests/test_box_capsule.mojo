"""Phase 9 Test: Box-Capsule Collision.

Tests box-capsule collision:
1. Capsule falling onto box face
2. Capsule parallel to box edge
3. Capsule at angle to box
"""

from math import sqrt, sin, cos, pi
from physics3d import Model, Data, ImpulseIntegrator


fn test_capsule_box_face() -> Bool:
    """Test vertical capsule falling onto top face of box.

    Capsule should come to rest on top of the box.
    """
    print("Test: Capsule-box face collision")

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

    # Body 0: Box (stationary, heavy)
    var box_hx: Scalar[DTYPE] = 0.3
    var box_hy: Scalar[DTYPE] = 0.3
    var box_hz: Scalar[DTYPE] = 0.1
    model.set_body_box(
        0, mass=1000.0, half_x=box_hx, half_y=box_hy, half_z=box_hz
    )

    # Body 1: Capsule (falling)
    var cap_radius: Scalar[DTYPE] = 0.05
    var cap_half_len: Scalar[DTYPE] = 0.1
    model.set_body_capsule(
        1, mass=1.0, radius=cap_radius, half_length=cap_half_len
    )

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    # Box at rest on ground
    data.set_body_position(0, 0.0, 0.0, box_hz)
    # Capsule starts above box (vertical orientation)
    data.set_body_position(1, 0.0, 0.0, 1.0)

    # Expected: capsule rests on box with bottom cap touching
    # Box top = 2*box_hz = 0.2
    # Capsule center should be at: box_top + half_len + radius = 0.2 + 0.1 + 0.05 = 0.35
    var expected_z = Scalar[DTYPE](2.0) * box_hz + cap_half_len + cap_radius

    # Simulate for 500ms
    for _ in range(500):
        ImpulseIntegrator.step(model, data)

    var final_z = data.get_body_z(1)
    var error = abs(final_z - expected_z)

    print("  Expected capsule z:", expected_z)
    print("  Final capsule z:", final_z)
    print("  Error:", error)

    # Allow 5mm tolerance
    var passed = error < 0.005
    if passed:
        print("  PASSED")
    else:
        print("  FAILED")
    return passed


fn test_capsule_box_parallel() -> Bool:
    """Test horizontal capsule parallel to box edge.

    Capsule should rest on box with cylinder touching face.
    """
    print("Test: Capsule-box parallel to edge")

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
    var box_hz: Scalar[DTYPE] = 0.1
    model.set_body_box(0, mass=1000.0, half_x=0.3, half_y=0.3, half_z=box_hz)

    # Body 1: Capsule
    var cap_radius: Scalar[DTYPE] = 0.05
    var cap_half_len: Scalar[DTYPE] = 0.15
    model.set_body_capsule(
        1, mass=1.0, radius=cap_radius, half_length=cap_half_len
    )

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    # Box at rest on ground
    data.set_body_position(0, 0.0, 0.0, box_hz)
    # Capsule starts above, horizontal (along X axis)
    data.set_body_position(1, 0.0, 0.0, 1.0)

    # Rotate capsule 90 degrees around Y to make it horizontal (along X)
    var angle = Scalar[DTYPE](pi) / Scalar[DTYPE](2.0)
    var half_angle = angle / Scalar[DTYPE](2.0)
    data.quaternions[1 * 4 + 0] = Scalar[DTYPE](0.0)
    data.quaternions[1 * 4 + 1] = sin(half_angle)
    data.quaternions[1 * 4 + 2] = Scalar[DTYPE](0.0)
    data.quaternions[1 * 4 + 3] = cos(half_angle)

    # Expected: horizontal capsule rests on box
    # Box top = 2*box_hz = 0.2
    # Capsule center should be at: box_top + radius = 0.2 + 0.05 = 0.25
    var expected_z = Scalar[DTYPE](2.0) * box_hz + cap_radius

    # Simulate for 500ms
    for _ in range(500):
        ImpulseIntegrator.step(model, data)

    var final_z = data.get_body_z(1)
    var error = abs(final_z - expected_z)

    print("  Expected capsule z:", expected_z)
    print("  Final capsule z:", final_z)
    print("  Error:", error)

    # Allow 5mm tolerance
    var passed = error < 0.005
    if passed:
        print("  PASSED")
    else:
        print("  FAILED")
    return passed


fn test_capsule_box_angled() -> Bool:
    """Test tilted capsule falling onto box.

    Capsule at 45 degrees should settle on box.
    """
    print("Test: Capsule-box angled collision")

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
    var box_hz: Scalar[DTYPE] = 0.1
    model.set_body_box(0, mass=1000.0, half_x=0.4, half_y=0.4, half_z=box_hz)

    # Body 1: Capsule
    var cap_radius: Scalar[DTYPE] = 0.05
    var cap_half_len: Scalar[DTYPE] = 0.1
    model.set_body_capsule(
        1, mass=1.0, radius=cap_radius, half_length=cap_half_len
    )

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    # Box at rest on ground
    data.set_body_position(0, 0.0, 0.0, box_hz)
    # Capsule starts above, tilted 45 degrees
    data.set_body_position(1, 0.0, 0.0, 1.0)

    # Rotate capsule 45 degrees around Y
    var angle = Scalar[DTYPE](pi) / Scalar[DTYPE](4.0)
    var half_angle = angle / Scalar[DTYPE](2.0)
    data.quaternions[1 * 4 + 0] = Scalar[DTYPE](0.0)
    data.quaternions[1 * 4 + 1] = sin(half_angle)
    data.quaternions[1 * 4 + 2] = Scalar[DTYPE](0.0)
    data.quaternions[1 * 4 + 3] = cos(half_angle)

    # Simulate for 500ms
    for _ in range(500):
        ImpulseIntegrator.step(model, data)

    var final_z = data.get_body_z(1)
    var box_top = Scalar[DTYPE](2.0) * box_hz

    print("  Box top:", box_top)
    print("  Final capsule z:", final_z)

    # Capsule should be above box top (not fallen through)
    var passed = final_z > box_top
    if passed:
        print("  PASSED (capsule above box)")
    else:
        print("  FAILED (capsule through box)")
    return passed


fn main():
    print("=" * 60)
    print("Phase 9: Box-Capsule Collision Tests")
    print("=" * 60)

    var passed = 0
    var total = 3

    if test_capsule_box_face():
        passed += 1
    print()

    if test_capsule_box_parallel():
        passed += 1
    print()

    if test_capsule_box_angled():
        passed += 1
    print()

    print("=" * 60)
    print("Results:", passed, "/", total, "tests passed")
    if passed == total:
        print("All box-capsule tests PASSED!")
    else:
        print("Some tests FAILED")
    print("=" * 60)

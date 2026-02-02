"""Phase 8 Test: Capsule-Plane Collision.

Tests capsule collision with ground plane:
1. Vertical capsule falls and stops at correct height
2. Horizontal capsule stops at radius above ground
3. Tilted capsule stops at lowest endpoint + radius
4. Capsule at rest does not drift
"""

from math import sqrt, sin, cos, pi
from physics3d_v2 import Model, Data, ImpulseIntegrator
from physics3d_v2.gpu.constants import GEOM_CAPSULE


fn test_vertical_capsule_plane() -> Bool:
    """Test vertical capsule falling to ground.

    A vertical capsule (axis along Z) should stop when its bottom cap
    touches the ground. Expected stop height = half_length + radius.
    """
    print("Test: Vertical capsule-plane collision")

    comptime DTYPE = DType.float64
    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 5

    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81,
        timestep=0.001,  # Small timestep for accuracy
        ground_z=0.0,
        restitution=0.0,  # No bounce
        friction=0.5,
    )

    # Capsule: radius=0.1, half_length=0.2 (total length 0.4 + caps)
    var radius: Scalar[DTYPE] = 0.1
    var half_length: Scalar[DTYPE] = 0.2
    model.set_body_capsule(0, mass=1.0, radius=radius, half_length=half_length)

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    # Start at height 1.0 with identity quaternion (vertical capsule)
    data.set_body_position(0, 0.0, 0.0, 1.0)

    # Expected final height: half_length + radius = 0.2 + 0.1 = 0.3
    var expected_z = half_length + radius

    # Simulate for 500ms
    for _ in range(500):
        ImpulseIntegrator.step(model, data)

    var final_z = data.get_body_z(0)
    var error = abs(final_z - expected_z)

    print("  Expected z:", expected_z)
    print("  Final z:", final_z)
    print("  Error:", error)

    # Allow 5mm tolerance
    var passed = error < 0.005
    if passed:
        print("  PASSED")
    else:
        print("  FAILED")
    return passed


fn test_horizontal_capsule_plane() -> Bool:
    """Test horizontal capsule falling to ground.

    A horizontal capsule (axis along X) should stop when its surface
    touches the ground. Expected stop height = radius.
    """
    print("Test: Horizontal capsule-plane collision")

    comptime DTYPE = DType.float64
    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 5

    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81,
        timestep=0.001,
        ground_z=0.0,
        restitution=0.0,
        friction=0.5,
    )

    var radius: Scalar[DTYPE] = 0.1
    var half_length: Scalar[DTYPE] = 0.2
    model.set_body_capsule(0, mass=1.0, radius=radius, half_length=half_length)

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    # Start at height 1.0
    data.set_body_position(0, 0.0, 0.0, 1.0)

    # Set quaternion to rotate 90 degrees around Y axis (capsule along X)
    # q = (0, sin(pi/4), 0, cos(pi/4)) = (0, 0.7071, 0, 0.7071)
    var angle = Scalar[DTYPE](pi) / Scalar[DTYPE](2.0)
    var half_angle = angle / Scalar[DTYPE](2.0)
    data.quaternions[0 * 4 + 0] = Scalar[DTYPE](0.0)  # qx
    data.quaternions[0 * 4 + 1] = sin(half_angle)  # qy
    data.quaternions[0 * 4 + 2] = Scalar[DTYPE](0.0)  # qz
    data.quaternions[0 * 4 + 3] = cos(half_angle)  # qw

    # Expected final height: just radius (horizontal capsule)
    var expected_z = radius

    # Simulate for 500ms
    for _ in range(500):
        ImpulseIntegrator.step(model, data)

    var final_z = data.get_body_z(0)
    var error = abs(final_z - expected_z)

    print("  Expected z:", expected_z)
    print("  Final z:", final_z)
    print("  Error:", error)

    # Allow 5mm tolerance
    var passed = error < 0.005
    if passed:
        print("  PASSED")
    else:
        print("  FAILED")
    return passed


fn test_tilted_capsule_plane() -> Bool:
    """Test tilted capsule falling to ground.

    A capsule tilted 45 degrees should stop when its lowest point
    (endpoint + radius) touches the ground.
    """
    print("Test: Tilted capsule-plane collision")

    comptime DTYPE = DType.float64
    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 5

    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81,
        timestep=0.001,
        ground_z=0.0,
        restitution=0.0,
        friction=0.5,
    )

    var radius: Scalar[DTYPE] = 0.1
    var half_length: Scalar[DTYPE] = 0.2
    model.set_body_capsule(0, mass=1.0, radius=radius, half_length=half_length)

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    # Start at height 1.0
    data.set_body_position(0, 0.0, 0.0, 1.0)

    # Set quaternion to rotate 45 degrees around Y axis
    var angle = Scalar[DTYPE](pi) / Scalar[DTYPE](4.0)  # 45 degrees
    var half_angle = angle / Scalar[DTYPE](2.0)
    data.quaternions[0 * 4 + 0] = Scalar[DTYPE](0.0)
    data.quaternions[0 * 4 + 1] = sin(half_angle)
    data.quaternions[0 * 4 + 2] = Scalar[DTYPE](0.0)
    data.quaternions[0 * 4 + 3] = cos(half_angle)

    # At 45 degrees tilt, the lowest endpoint is at:
    # z_endpoint = center_z - half_length * cos(45) = center_z - half_length * 0.7071
    # Expected center z = half_length * cos(45) + radius
    var cos_45 = cos(angle)
    var expected_z = half_length * cos_45 + radius

    # Simulate for 500ms
    for _ in range(500):
        ImpulseIntegrator.step(model, data)

    var final_z = data.get_body_z(0)
    var error = abs(final_z - expected_z)

    print("  Expected z:", expected_z)
    print("  Final z:", final_z)
    print("  Error:", error)

    # Allow 1cm tolerance (tilted capsule settling is less precise)
    var passed = error < 0.01
    if passed:
        print("  PASSED")
    else:
        print("  FAILED")
    return passed


fn test_capsule_resting() -> Bool:
    """Test capsule at rest doesn't drift.

    A capsule placed exactly at rest position should not drift over time.
    """
    print("Test: Capsule at rest (no drift)")

    comptime DTYPE = DType.float64
    comptime NUM_BODIES = 1
    comptime MAX_CONTACTS = 5

    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81,
        timestep=0.01,
        ground_z=0.0,
        restitution=0.0,
        friction=0.5,
    )

    var radius: Scalar[DTYPE] = 0.1
    var half_length: Scalar[DTYPE] = 0.2
    model.set_body_capsule(0, mass=1.0, radius=radius, half_length=half_length)

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    # Start exactly at rest height (vertical capsule)
    var rest_z = half_length + radius
    data.set_body_position(0, 0.0, 0.0, rest_z)
    data.set_body_velocity(0, 0.0, 0.0, 0.0)

    # Simulate for 1000 steps
    for _ in range(1000):
        ImpulseIntegrator.step(model, data)

    var final_z = data.get_body_z(0)
    var drift = abs(final_z - rest_z)

    print("  Initial z:", rest_z)
    print("  Final z:", final_z)
    print("  Drift:", drift)

    # Allow 1mm drift
    var passed = drift < 0.001
    if passed:
        print("  PASSED")
    else:
        print("  FAILED")
    return passed


fn main():
    print("=" * 60)
    print("Phase 8: Capsule-Plane Collision Tests")
    print("=" * 60)

    var passed = 0
    var total = 4

    if test_vertical_capsule_plane():
        passed += 1
    print()

    if test_horizontal_capsule_plane():
        passed += 1
    print()

    if test_tilted_capsule_plane():
        passed += 1
    print()

    if test_capsule_resting():
        passed += 1
    print()

    print("=" * 60)
    print("Results:", passed, "/", total, "tests passed")
    if passed == total:
        print("All capsule-plane tests PASSED!")
    else:
        print("Some tests FAILED")
    print("=" * 60)

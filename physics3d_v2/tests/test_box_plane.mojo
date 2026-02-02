"""Phase 9 Test: Box-Plane Collision.

Tests box collision with ground plane:
1. Axis-aligned box falls and stops at correct height (half_z)
2. Tilted box stops at lowest vertex
3. Box at rest does not drift
"""

from math import sqrt, sin, cos, pi
from physics3d_v2 import Model, Data, ImpulseIntegrator


fn test_axis_aligned_box_plane() -> Bool:
    """Test axis-aligned box falling to ground.

    An axis-aligned box should stop when its bottom face touches the ground.
    Expected stop height = half_z.
    """
    print("Test: Axis-aligned box-plane collision")

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

    # Box: half-extents 0.1 x 0.1 x 0.2
    var half_x: Scalar[DTYPE] = 0.1
    var half_y: Scalar[DTYPE] = 0.1
    var half_z: Scalar[DTYPE] = 0.2
    model.set_body_box(0, mass=1.0, half_x=half_x, half_y=half_y, half_z=half_z)

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    # Start at height 1.0 with identity quaternion (axis-aligned box)
    data.set_body_position(0, 0.0, 0.0, 1.0)

    # Expected final height: half_z = 0.2
    var expected_z = half_z

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


fn test_tilted_box_plane() -> Bool:
    """Test tilted box falling to ground.

    A box tilted 45 degrees around Y axis should stop when its lowest vertex
    touches the ground.
    """
    print("Test: Tilted box-plane collision")

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

    # Cube: half-extent 0.1 in all directions
    var half_ext: Scalar[DTYPE] = 0.1
    model.set_body_box(0, mass=1.0, half_x=half_ext, half_y=half_ext, half_z=half_ext)

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    # Start at height 1.0
    data.set_body_position(0, 0.0, 0.0, 1.0)

    # Set quaternion to rotate 45 degrees around Y axis
    var angle = Scalar[DTYPE](pi) / Scalar[DTYPE](4.0)  # 45 degrees
    var half_angle = angle / Scalar[DTYPE](2.0)
    data.quaternions[0 * 4 + 0] = Scalar[DTYPE](0.0)  # qx
    data.quaternions[0 * 4 + 1] = sin(half_angle)  # qy
    data.quaternions[0 * 4 + 2] = Scalar[DTYPE](0.0)  # qz
    data.quaternions[0 * 4 + 3] = cos(half_angle)  # qw

    # At 45 degrees tilt around Y, the lowest vertex is at distance:
    # sqrt(hx^2 + hz^2) = sqrt(0.01 + 0.01) = sqrt(0.02) = 0.1414...
    # below the center (diagonal of the box in XZ plane)
    var expected_z = sqrt(half_ext * half_ext + half_ext * half_ext)

    # Simulate for 500ms
    for _ in range(500):
        ImpulseIntegrator.step(model, data)

    var final_z = data.get_body_z(0)
    var error = abs(final_z - expected_z)

    print("  Expected z:", expected_z)
    print("  Final z:", final_z)
    print("  Error:", error)

    # Allow 1cm tolerance (tilted box settling is less precise)
    var passed = error < 0.01
    if passed:
        print("  PASSED")
    else:
        print("  FAILED")
    return passed


fn test_box_resting() -> Bool:
    """Test box at rest doesn't drift.

    A box placed exactly at rest position should not drift over time.
    """
    print("Test: Box at rest (no drift)")

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

    var half_z: Scalar[DTYPE] = 0.2
    model.set_body_box(0, mass=1.0, half_x=0.1, half_y=0.1, half_z=half_z)

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    # Start exactly at rest height (axis-aligned box)
    var rest_z = half_z
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
    print("Phase 9: Box-Plane Collision Tests")
    print("=" * 60)

    var passed = 0
    var total = 3

    if test_axis_aligned_box_plane():
        passed += 1
    print()

    if test_tilted_box_plane():
        passed += 1
    print()

    if test_box_resting():
        passed += 1
    print()

    print("=" * 60)
    print("Results:", passed, "/", total, "tests passed")
    if passed == total:
        print("All box-plane tests PASSED!")
    else:
        print("Some tests FAILED")
    print("=" * 60)

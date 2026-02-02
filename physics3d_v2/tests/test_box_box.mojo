"""Phase 9 Test: Box-Box Collision.

Tests box-box collision:
1. Face-face contact (stacked boxes)
2. Edge-edge contact
3. Stacked boxes stability
"""

from math import sqrt, sin, cos, pi
from physics3d_v2 import Model, Data, ImpulseIntegrator


fn test_box_box_face_face() -> Bool:
    """Test box falling onto another box (face-face contact).

    Small box should rest on top of large box.
    """
    print("Test: Box-box face-face collision")

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

    # Body 0: Large box (heavy, stationary)
    var big_hx: Scalar[DTYPE] = 0.3
    var big_hy: Scalar[DTYPE] = 0.3
    var big_hz: Scalar[DTYPE] = 0.1
    model.set_body_box(0, mass=1000.0, half_x=big_hx, half_y=big_hy, half_z=big_hz)

    # Body 1: Small box (falling)
    var small_hx: Scalar[DTYPE] = 0.1
    var small_hy: Scalar[DTYPE] = 0.1
    var small_hz: Scalar[DTYPE] = 0.1
    model.set_body_box(1, mass=1.0, half_x=small_hx, half_y=small_hy, half_z=small_hz)

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    # Big box at rest on ground
    data.set_body_position(0, 0.0, 0.0, big_hz)
    # Small box starts above
    data.set_body_position(1, 0.0, 0.0, 1.0)

    # Expected: small box rests on big box
    # Big box top = 2*big_hz = 0.2
    # Small box center = big_box_top + small_hz = 0.2 + 0.1 = 0.3
    var expected_z = Scalar[DTYPE](2.0) * big_hz + small_hz

    # Simulate for 500ms
    for _ in range(500):
        ImpulseIntegrator.step(model, data)

    var final_z = data.get_body_z(1)
    var error = abs(final_z - expected_z)

    print("  Expected small box z:", expected_z)
    print("  Final small box z:", final_z)
    print("  Error:", error)

    # Allow 5mm tolerance
    var passed = error < 0.005
    if passed:
        print("  PASSED")
    else:
        print("  FAILED")
    return passed


fn test_box_box_edge_edge() -> Bool:
    """Test box falling with edge contacting another box's edge.

    One box rotated 45 degrees, falling onto aligned box.
    """
    print("Test: Box-box edge-edge collision")

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

    # Body 0: Base box (large, heavy)
    var base_h: Scalar[DTYPE] = 0.1
    model.set_body_box(0, mass=1000.0, half_x=0.3, half_y=0.3, half_z=base_h)

    # Body 1: Top box (rotated)
    var top_h: Scalar[DTYPE] = 0.1
    model.set_body_box(1, mass=1.0, half_x=top_h, half_y=top_h, half_z=top_h)

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    # Base box on ground
    data.set_body_position(0, 0.0, 0.0, base_h)
    # Top box starts above, rotated 45 degrees around Z
    data.set_body_position(1, 0.0, 0.0, 1.0)

    # Rotate 45 degrees around Z axis
    var angle = Scalar[DTYPE](pi) / Scalar[DTYPE](4.0)
    var half_angle = angle / Scalar[DTYPE](2.0)
    data.quaternions[1 * 4 + 0] = Scalar[DTYPE](0.0)
    data.quaternions[1 * 4 + 1] = Scalar[DTYPE](0.0)
    data.quaternions[1 * 4 + 2] = sin(half_angle)
    data.quaternions[1 * 4 + 3] = cos(half_angle)

    # Simulate for 500ms
    for _ in range(500):
        ImpulseIntegrator.step(model, data)

    var final_z = data.get_body_z(1)
    var base_top = Scalar[DTYPE](2.0) * base_h

    print("  Base box top:", base_top)
    print("  Final top box z:", final_z)

    # Top box should be resting above base box
    var passed = final_z > base_top
    if passed:
        print("  PASSED (box above base)")
    else:
        print("  FAILED (box fell through)")
    return passed


fn test_stacked_boxes_stability() -> Bool:
    """Test stability of stacked boxes.

    Three boxes stacked should remain stable without drift.
    """
    print("Test: Stacked boxes stability")

    comptime DTYPE = DType.float64
    comptime NUM_BODIES = 3
    comptime MAX_CONTACTS = 15

    var model = Model[DTYPE, NUM_BODIES, MAX_CONTACTS](
        gravity_z=-9.81,
        timestep=0.01,
        ground_z=0.0,
        restitution=0.0,
        friction=0.5,
    )

    # All boxes are cubes
    var half_ext: Scalar[DTYPE] = 0.1
    for i in range(NUM_BODIES):
        model.set_body_box(i, mass=1.0, half_x=half_ext, half_y=half_ext, half_z=half_ext)

    var data = Data[DTYPE, NUM_BODIES, MAX_CONTACTS]()
    # Stack boxes perfectly
    data.set_body_position(0, 0.0, 0.0, half_ext)  # Bottom
    data.set_body_position(1, 0.0, 0.0, Scalar[DTYPE](3.0) * half_ext)  # Middle
    data.set_body_position(2, 0.0, 0.0, Scalar[DTYPE](5.0) * half_ext)  # Top

    # Record initial positions
    var initial_z0 = data.get_body_z(0)
    var initial_z1 = data.get_body_z(1)
    var initial_z2 = data.get_body_z(2)

    # Simulate for 500 steps
    for _ in range(500):
        ImpulseIntegrator.step(model, data)

    var final_z0 = data.get_body_z(0)
    var final_z1 = data.get_body_z(1)
    var final_z2 = data.get_body_z(2)

    var drift0 = abs(final_z0 - initial_z0)
    var drift1 = abs(final_z1 - initial_z1)
    var drift2 = abs(final_z2 - initial_z2)
    var max_drift = max(drift0, max(drift1, drift2))

    print("  Initial positions:", initial_z0, initial_z1, initial_z2)
    print("  Final positions:", final_z0, final_z1, final_z2)
    print("  Max drift:", max_drift)

    # Allow 5mm drift
    var passed = max_drift < 0.005
    if passed:
        print("  PASSED (minimal drift)")
    else:
        print("  FAILED (excessive drift)")
    return passed


fn main():
    print("=" * 60)
    print("Phase 9: Box-Box Collision Tests")
    print("=" * 60)

    var passed = 0
    var total = 3

    if test_box_box_face_face():
        passed += 1
    print()

    if test_box_box_edge_edge():
        passed += 1
    print()

    if test_stacked_boxes_stability():
        passed += 1
    print()

    print("=" * 60)
    print("Results:", passed, "/", total, "tests passed")
    if passed == total:
        print("All box-box tests PASSED!")
    else:
        print("Some tests FAILED")
    print("=" * 60)

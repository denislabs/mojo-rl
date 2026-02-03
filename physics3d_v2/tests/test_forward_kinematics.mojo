"""Tests for forward kinematics in the Generalized Coordinates engine.

Tests:
1. FK identity: qpos=0 -> xpos matches body_pos
2. FK HINGE 90deg: Body rotated correctly
3. FK chain: Multiple bodies in kinematic tree

Run with:
    pixi run mojo run physics3d_v2/generalized/tests/test_forward_kinematics.mojo
"""

from math import sqrt, pi
from builtin.math import abs
from physics3d_v2.types import ModelGC, DataGC
from physics3d_v2.kinematics.forward_kinematics import forward_kinematics
from physics3d_v2.kinematics.quat_math import quat_rotate


fn test_fk_identity() -> Bool:
    """Test that qpos=0 gives xpos at pivot + body_pos offset."""
    print("Test FK identity (qpos=0)...")

    # Create a single body with a hinge joint
    # NQ=1, NV=1, NBODY=1, NJOINT=1, MAX_CONTACTS=5
    var model = ModelGC[DType.float64, 1, 1, 1, 1, 5]()

    # Set body with world as parent
    model.set_body(0, mass=1.0, inertia=(0.1, 0.1, 0.1), radius=0.1)
    model.set_body_parent(0, -1)  # Parent is world

    # Body's CoM is offset (0, 0, -0.5) from the pivot
    # So at qpos=0, body should be at pivot + offset
    model.set_body_local_frame(0, pos=(0.0, 0.0, -0.5))

    # Add hinge joint at (0, 0, 1) with Y axis
    _ = model.add_hinge_joint(
        body_id=0,
        pos=(0.0, 0.0, 1.0),  # Pivot at height 1
        axis=(0.0, 1.0, 0.0),
    )

    # Create data with qpos=0
    var data = DataGC[DType.float64, 1, 1, 1, 1, 5]()
    data.qpos[0] = Float64(0.0)  # Zero angle

    # Run forward kinematics
    forward_kinematics(model, data)

    # Check xpos: pivot (0,0,1) + offset (0,0,-0.5) = (0,0,0.5)
    var x = data.xpos[0]
    var y = data.xpos[1]
    var z = data.xpos[2]

    var expected_x = Float64(0.0)
    var expected_y = Float64(0.0)
    var expected_z = Float64(0.5)

    var tol = Float64(1e-10)
    var pass_test = True

    if abs(x - expected_x) > tol:
        print("  FAIL: x =", x, "expected", expected_x)
        pass_test = False
    if abs(y - expected_y) > tol:
        print("  FAIL: y =", y, "expected", expected_y)
        pass_test = False
    if abs(z - expected_z) > tol:
        print("  FAIL: z =", z, "expected", expected_z)
        pass_test = False

    if pass_test:
        print("  PASS: xpos =", x, y, z)
    return pass_test


fn test_fk_hinge_90deg() -> Bool:
    """Test that a 90-degree hinge rotation is computed correctly."""
    print("Test FK HINGE 90deg...")

    # Pendulum: body at (0, 0, -1) relative to pivot at (0, 0, 1)
    # When rotated 90deg around Y axis, body should be at (1, 0, 1)
    var model = ModelGC[DType.float64, 1, 1, 1, 1, 5]()

    model.set_body(0, mass=1.0, inertia=(0.1, 0.1, 0.1), radius=0.1)
    model.set_body_parent(0, -1)

    # Body CoM at (0, 0, -1) in local frame (will be rotated by joint)
    model.set_body_local_frame(0, pos=(0.0, 0.0, -1.0))

    # Hinge at world origin, Y axis
    _ = model.add_hinge_joint(
        body_id=0,
        pos=(0.0, 0.0, 0.0),  # Pivot at origin
        axis=(0.0, 1.0, 0.0),  # Rotate around Y
    )

    var data = DataGC[DType.float64, 1, 1, 1, 1, 5]()
    data.qpos[0] = pi / 2.0  # 90 degrees

    forward_kinematics(model, data)

    var x = data.xpos[0]
    var y = data.xpos[1]
    var z = data.xpos[2]

    # After 90deg rotation around Y:
    # Original: (0, 0, -1)
    # Rotated:  (1, 0, 0) (approximately, accounting for pivot)
    # Actually pivot is at origin, body_pos is (0,0,-1)
    # So body ends up at (1, 0, 0) - no, wait...
    # Rotation around Y: (x, y, z) -> (z, y, -x)
    # (0, 0, -1) -> (-1, 0, 0) - that's -90 degrees
    # For +90 degrees: (0, 0, -1) -> (x', y', z')
    # x' = cos(90)*x - sin(90)*z = 0 - (-1)*1 = 1? No, rotation matrices...
    # Ry(θ) = [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
    # [0, 0, -1] * Ry(90) = [0*0 + 0*0 + (-1)*1, 0, 0*(-1) + 0*0 + (-1)*0]
    # Actually let's just compute: sin(90)=1, cos(90)=0
    # x' = cos(θ)*x + sin(θ)*z = 0*0 + 1*(-1) = -1
    # y' = y = 0
    # z' = -sin(θ)*x + cos(θ)*z = -1*0 + 0*(-1) = 0
    # So expected: (-1, 0, 0)

    var expected_x = Float64(-1.0)
    var expected_y = Float64(0.0)
    var expected_z = Float64(0.0)

    var tol = Float64(1e-6)
    var pass_test = True

    if abs(x - expected_x) > tol:
        print("  FAIL: x =", x, "expected", expected_x)
        pass_test = False
    if abs(y - expected_y) > tol:
        print("  FAIL: y =", y, "expected", expected_y)
        pass_test = False
    if abs(z - expected_z) > tol:
        print("  FAIL: z =", z, "expected", expected_z)
        pass_test = False

    if pass_test:
        print("  PASS: xpos =", x, y, z)
    return pass_test


fn test_fk_double_pendulum() -> Bool:
    """Test FK for a double pendulum (kinematic chain)."""
    print("Test FK double pendulum...")

    # Double pendulum:
    # Joint 0 at world origin (0,0,0), body 0's CoM at (0, 0, -0.5) relative to pivot
    # Joint 1 at body 0's CoM, body 1's CoM at (0, 0, -0.5) relative to joint 1
    #
    # At qpos=0:
    # - Body 0: pivot(0,0,0) + offset(0,0,-0.5) = (0,0,-0.5)
    # - Body 1: body0 + pivot_offset(0) + joint1_offset + body1_offset
    #           The joint is at body 0's frame, so relative to body 0

    # NQ=2, NV=2, NBODY=2, NJOINT=2
    var model = ModelGC[DType.float64, 2, 2, 2, 2, 5]()

    # Body 0: CoM at offset -0.5 below its joint pivot
    model.set_body(0, mass=1.0, inertia=(0.1, 0.1, 0.1), radius=0.1)
    model.set_body_parent(0, -1)  # World
    model.set_body_local_frame(0, pos=(0.0, 0.0, -0.5))

    # Body 1: CoM at offset -0.5 below its joint pivot
    model.set_body(1, mass=1.0, inertia=(0.1, 0.1, 0.1), radius=0.1)
    model.set_body_parent(1, 0)  # Parent is body 0
    model.set_body_local_frame(1, pos=(0.0, 0.0, -0.5))

    # Joint 0: hinge at origin (0,0,0), Y axis
    _ = model.add_hinge_joint(
        body_id=0,
        pos=(0.0, 0.0, 0.0),  # Pivot at world origin
        axis=(0.0, 1.0, 0.0),
    )

    # Joint 1: hinge at body 0's CoM position (relative to body 0)
    # Since body 0's local frame is at -0.5 from its pivot,
    # placing joint 1 at (0,0,0) in body 0's frame means it's at body 0's CoM
    _ = model.add_hinge_joint(
        body_id=1,
        pos=(0.0, 0.0, 0.0),  # Relative to body 0
        axis=(0.0, 1.0, 0.0),
    )

    var data = DataGC[DType.float64, 2, 2, 2, 2, 5]()
    data.qpos[0] = Float64(0.0)  # First joint angle
    data.qpos[1] = Float64(0.0)  # Second joint angle

    forward_kinematics(model, data)

    # Body 0: pivot(0,0,0) + offset(0,0,-0.5) = (0,0,-0.5)
    var x0 = data.xpos[0]
    var y0 = data.xpos[1]
    var z0 = data.xpos[2]

    # Body 1: joint1 is at body0_xpos (0,0,-0.5)
    #         body1_xpos = joint1_pivot + body1_offset = (0,0,-0.5) + (0,0,-0.5) = (0,0,-1)
    var x1 = data.xpos[3]
    var y1 = data.xpos[4]
    var z1 = data.xpos[5]

    var tol = Float64(1e-6)
    var pass_test = True

    if abs(x0 - 0.0) > tol or abs(y0 - 0.0) > tol or abs(z0 - (-0.5)) > tol:
        print("  FAIL: body0 xpos =", x0, y0, z0, "expected (0, 0, -0.5)")
        pass_test = False
    else:
        print("  PASS: body0 xpos =", x0, y0, z0)

    if abs(x1 - 0.0) > tol or abs(y1 - 0.0) > tol or abs(z1 - (-1.0)) > tol:
        print("  FAIL: body1 xpos =", x1, y1, z1, "expected (0, 0, -1)")
        pass_test = False
    else:
        print("  PASS: body1 xpos =", x1, y1, z1)

    return pass_test


fn main():
    print("=== Forward Kinematics Tests ===\n")

    var all_pass = True

    if not test_fk_identity():
        all_pass = False

    if not test_fk_hinge_90deg():
        all_pass = False

    if not test_fk_double_pendulum():
        all_pass = False

    print("")
    if all_pass:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")

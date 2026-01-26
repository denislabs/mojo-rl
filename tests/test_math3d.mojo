"""Tests for the math3d module."""

from math3d import Vec3, Quat, Mat3, Mat4
from math import sqrt


fn check(condition: Bool, msg: String) raises:
    """Check condition and raise if false."""
    if not condition:
        raise Error(msg)


fn test_vec3() raises:
    """Test Vec3 operations."""
    print("Testing Vec3...")

    # Construction
    var v1 = Vec3(1.0, 2.0, 3.0)
    var v2 = Vec3(4.0, 5.0, 6.0)

    # Arithmetic
    var sum = v1 + v2
    print("  v1 + v2 =", sum.x, sum.y, sum.z)
    check(sum.x == 5.0 and sum.y == 7.0 and sum.z == 9.0, "addition failed")

    var diff = v2 - v1
    check(diff.x == 3.0 and diff.y == 3.0 and diff.z == 3.0, "subtraction failed")

    var scaled = v1 * 2.0
    check(scaled.x == 2.0 and scaled.y == 4.0 and scaled.z == 6.0, "scalar mul failed")

    # Dot product
    var d = v1.dot(v2)
    check(d == 32.0, "dot product failed")  # 1*4 + 2*5 + 3*6 = 32

    # Cross product
    var cross = Vec3(1.0, 0.0, 0.0).cross(Vec3(0.0, 1.0, 0.0))
    check(cross.z == 1.0, "cross product failed")

    # Length
    var len_v = Vec3(3.0, 4.0, 0.0).length()
    check(abs(len_v - 5.0) < 1e-10, "length failed")

    # Normalize
    var normalized = Vec3(3.0, 4.0, 0.0).normalized()
    check(abs(normalized.length() - 1.0) < 1e-10, "normalize failed")

    print("  Vec3 tests PASSED")


fn test_quat() raises:
    """Test Quat operations."""
    print("Testing Quat...")

    # Identity
    var q_id = Quat.identity()
    check(q_id.w == 1.0 and q_id.x == 0.0, "identity failed")

    # Rotation around Z axis by 90 degrees
    var pi_2 = 1.5707963267948966
    var q = Quat.from_axis_angle(Vec3.unit_z(), pi_2)

    # Rotate X-axis vector -> should become Y-axis
    var v = Vec3(1.0, 0.0, 0.0)
    var rotated = q.rotate_vec(v)
    print("  Rotated X by 90 deg around Z:", rotated.x, rotated.y, rotated.z)
    check(abs(rotated.x) < 1e-10 and abs(rotated.y - 1.0) < 1e-10, "rotation failed")

    # Conjugate is inverse for unit quaternion
    var q_conj = q.conjugate()
    var q_prod = q * q_conj
    check(abs(q_prod.w - 1.0) < 1e-10 and abs(q_prod.x) < 1e-10, "conjugate failed")

    print("  Quat tests PASSED")


fn test_mat3() raises:
    """Test Mat3 operations."""
    print("Testing Mat3...")

    # Identity
    var I = Mat3.identity()
    check(I.m00 == 1.0 and I.m11 == 1.0 and I.m22 == 1.0, "identity failed")

    # Rotation Z by 90 degrees
    var pi_2 = 1.5707963267948966
    var Rz = Mat3.rotation_z(pi_2)

    # Rotate X-axis vector -> should become Y-axis
    var v = Vec3(1.0, 0.0, 0.0)
    var rotated = Rz * v
    print("  Rotated X by 90 deg around Z:", rotated.x, rotated.y, rotated.z)
    check(abs(rotated.x) < 1e-10 and abs(rotated.y - 1.0) < 1e-10, "mat3 rotation failed")

    # Determinant of identity
    var det = I.determinant()
    check(abs(det - 1.0) < 1e-10, "determinant failed")

    # Mat3 from Quat and back
    var q = Quat.from_axis_angle(Vec3.unit_y(), 0.5)
    var M = Mat3.from_quat(q)
    var q2 = M.to_quat()
    check(q.approx_eq(q2) or q.approx_eq(-q2), "mat3<->quat conversion failed")

    print("  Mat3 tests PASSED")


fn test_mat4() raises:
    """Test Mat4 operations."""
    print("Testing Mat4...")

    # Identity
    var I = Mat4.identity()
    check(I.m00 == 1.0 and I.m33 == 1.0, "identity failed")

    # Translation
    var T = Mat4.from_translation(Vec3(1.0, 2.0, 3.0))
    var p = T.transform_point(Vec3.zero())
    check(p.x == 1.0 and p.y == 2.0 and p.z == 3.0, "translation failed")

    # Vector should not be affected by translation
    var v = T.transform_vector(Vec3(1.0, 0.0, 0.0))
    check(v.x == 1.0 and v.y == 0.0 and v.z == 0.0, "vector transform failed")

    # Matrix multiplication
    var T2 = Mat4.from_translation(Vec3(3.0, 0.0, 0.0))
    var combined = T @ T2
    var p2 = combined.transform_point(Vec3.zero())
    check(p2.x == 4.0 and p2.y == 2.0 and p2.z == 3.0, "matrix multiply failed")

    # Inverse
    var T_inv = T.inverse()
    var p3 = T_inv.transform_point(Vec3(1.0, 2.0, 3.0))
    check(abs(p3.x) < 1e-10 and abs(p3.y) < 1e-10 and abs(p3.z) < 1e-10, "inverse failed")

    print("  Mat4 tests PASSED")


fn main() raises:
    print("=== Math3D Tests ===\n")

    test_vec3()
    test_quat()
    test_mat3()
    test_mat4()

    print("\n=== All Math3D Tests PASSED ===")

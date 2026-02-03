"""Quaternion mathematics for forward kinematics.

Quaternion convention: [x, y, z, w] (scalar last, same as MuJoCo).
Identity quaternion: [0, 0, 0, 1]

Functions:
- quat_mul: Quaternion multiplication
- quat_conjugate: Quaternion conjugate (inverse for unit quaternions)
- quat_rotate: Rotate a vector by a quaternion
- quat_normalize: Normalize a quaternion to unit length
- axis_angle_to_quat: Convert axis-angle to quaternion
"""

from math import sqrt, sin, cos


# =============================================================================
# Quaternion Multiplication
# =============================================================================


fn quat_mul[
    DTYPE: DType
](
    ax: Scalar[DTYPE],
    ay: Scalar[DTYPE],
    az: Scalar[DTYPE],
    aw: Scalar[DTYPE],
    bx: Scalar[DTYPE],
    by: Scalar[DTYPE],
    bz: Scalar[DTYPE],
    bw: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Multiply two quaternions: result = a * b.

    Args:
        ax, ay, az, aw: First quaternion [x, y, z, w].
        bx, by, bz, bw: Second quaternion [x, y, z, w].

    Returns:
        Product quaternion [x, y, z, w].
    """
    # Hamilton product for quaternions in [x, y, z, w] format
    var rx = aw * bx + ax * bw + ay * bz - az * by
    var ry = aw * by - ax * bz + ay * bw + az * bx
    var rz = aw * bz + ax * by - ay * bx + az * bw
    var rw = aw * bw - ax * bx - ay * by - az * bz
    return (rx, ry, rz, rw)


# =============================================================================
# Quaternion Conjugate
# =============================================================================


fn quat_conjugate[
    DTYPE: DType
](
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Compute quaternion conjugate (inverse for unit quaternions).

    Args:
        qx, qy, qz, qw: Quaternion [x, y, z, w].

    Returns:
        Conjugate quaternion [-x, -y, -z, w].
    """
    return (-qx, -qy, -qz, qw)


# =============================================================================
# Quaternion Rotation of Vector
# =============================================================================


fn quat_rotate[
    DTYPE: DType
](
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    vx: Scalar[DTYPE],
    vy: Scalar[DTYPE],
    vz: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Rotate a vector by a quaternion.

    Computes: q * v * q^(-1) using the efficient formula:
    v' = v + 2*w*(w x v) + 2*(w x (w x v))
    where q = [qx, qy, qz, qw]

    Args:
        qx, qy, qz, qw: Unit quaternion [x, y, z, w].
        vx, vy, vz: Vector to rotate.

    Returns:
        Rotated vector (rx, ry, rz).
    """
    # Compute 2 * (q_xyz x v)
    var tx = Scalar[DTYPE](2) * (qy * vz - qz * vy)
    var ty = Scalar[DTYPE](2) * (qz * vx - qx * vz)
    var tz = Scalar[DTYPE](2) * (qx * vy - qy * vx)

    # Result = v + w*t + (q_xyz x t)
    var rx = vx + qw * tx + (qy * tz - qz * ty)
    var ry = vy + qw * ty + (qz * tx - qx * tz)
    var rz = vz + qw * tz + (qx * ty - qy * tx)

    return (rx, ry, rz)


# =============================================================================
# Quaternion Normalization
# =============================================================================


fn quat_normalize[
    DTYPE: DType
](
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Normalize a quaternion to unit length.

    Args:
        qx, qy, qz, qw: Quaternion [x, y, z, w].

    Returns:
        Normalized unit quaternion.
    """
    var length_sq = qx * qx + qy * qy + qz * qz + qw * qw
    var inv_length = Scalar[DTYPE](1.0) / sqrt(length_sq + Scalar[DTYPE](1e-12))
    return (qx * inv_length, qy * inv_length, qz * inv_length, qw * inv_length)


# =============================================================================
# Axis-Angle to Quaternion
# =============================================================================


fn axis_angle_to_quat[
    DTYPE: DType
](
    ax: Scalar[DTYPE],
    ay: Scalar[DTYPE],
    az: Scalar[DTYPE],
    angle: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Convert axis-angle representation to quaternion.

    Args:
        ax, ay, az: Rotation axis (should be normalized).
        angle: Rotation angle in radians.

    Returns:
        Quaternion [x, y, z, w] = [sin(θ/2)*axis, cos(θ/2)].
    """
    var half_angle = angle * Scalar[DTYPE](0.5)
    var s = sin(half_angle)
    var c = cos(half_angle)

    return (ax * s, ay * s, az * s, c)


# =============================================================================
# Quaternion to Axis-Angle (for debugging/visualization)
# =============================================================================


fn quat_to_axis_angle[
    DTYPE: DType
](
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Convert quaternion to axis-angle representation.

    Args:
        qx, qy, qz, qw: Quaternion [x, y, z, w].

    Returns:
        (axis_x, axis_y, axis_z, angle) where angle is in radians.
    """
    from math import acos, atan2

    # Ensure qw is in valid range for acos
    var w_clamped = qw
    if w_clamped > Scalar[DTYPE](1):
        w_clamped = Scalar[DTYPE](1)
    elif w_clamped < Scalar[DTYPE](-1):
        w_clamped = Scalar[DTYPE](-1)

    # Compute angle
    var angle = Scalar[DTYPE](2) * acos(w_clamped)

    # Compute axis (handle near-zero angle case)
    var sin_half = sqrt(qx * qx + qy * qy + qz * qz)
    if sin_half < Scalar[DTYPE](1e-10):
        # Near-identity rotation, axis is arbitrary
        return (Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1), angle)

    var inv_sin = Scalar[DTYPE](1) / sin_half
    return (qx * inv_sin, qy * inv_sin, qz * inv_sin, angle)


# =============================================================================
# Quaternion Integration (for integrating angular velocity)
# =============================================================================


fn quat_integrate[
    DTYPE: DType
](
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    wx: Scalar[DTYPE],
    wy: Scalar[DTYPE],
    wz: Scalar[DTYPE],
    dt: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Integrate quaternion with angular velocity.

    Uses first-order approximation: q(t+dt) = q(t) + 0.5 * dt * omega * q(t)
    where omega is the quaternion [wx, wy, wz, 0].

    Args:
        qx, qy, qz, qw: Current orientation quaternion.
        wx, wy, wz: Angular velocity in world frame.
        dt: Time step.

    Returns:
        Updated (normalized) quaternion.
    """
    # Compute qdot = 0.5 * omega * q
    # where omega = [wx, wy, wz, 0]
    var half_dt = Scalar[DTYPE](0.5) * dt

    var qdot_x = half_dt * (qw * wx + qy * wz - qz * wy)
    var qdot_y = half_dt * (qw * wy - qx * wz + qz * wx)
    var qdot_z = half_dt * (qw * wz + qx * wy - qy * wx)
    var qdot_w = half_dt * (-qx * wx - qy * wy - qz * wz)

    # Integrate
    var new_qx = qx + qdot_x
    var new_qy = qy + qdot_y
    var new_qz = qz + qdot_z
    var new_qw = qw + qdot_w

    # Normalize
    return quat_normalize(new_qx, new_qy, new_qz, new_qw)

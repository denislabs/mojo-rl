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

from std.math import sqrt, sin, cos, acos, atan2


# =============================================================================
# Quaternion Multiplication
# =============================================================================


def quat_mul[
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
        ax: First quaternion x.
        ay: First quaternion y.
        az: First quaternion z.
        aw: First quaternion w.
        bx: Second quaternion x.
        by: Second quaternion y.
        bz: Second quaternion z.
        bw: Second quaternion w.

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


def quat_conjugate[
    DTYPE: DType
](
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Compute quaternion conjugate (inverse for unit quaternions).

    Args:
        qx: Quaternion x.
        qy: Quaternion y.
        qz: Quaternion z.
        qw: Quaternion w.

    Returns:
        Conjugate quaternion [-x, -y, -z, w].
    """
    return (-qx, -qy, -qz, qw)


# =============================================================================
# Quaternion Rotation of Vector
# =============================================================================


def quat_rotate[
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
        qx: Unit quaternion x.
        qy: Unit quaternion y.
        qz: Unit quaternion z.
        qw: Unit quaternion w.
        vx: Vector to rotate x.
        vy: Vector to rotate y.
        vz: Vector to rotate z.

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


def quat_rotate_inverse[
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
    """Inverse-rotate a vector by a quaternion: q^(-1) * v * q.

    Equivalent to quat_rotate with conjugate quaternion (-qx, -qy, -qz, qw).
    """
    return quat_rotate[DTYPE](-qx, -qy, -qz, qw, vx, vy, vz)


# =============================================================================
# Quaternion Normalization
# =============================================================================


def quat_normalize[
    DTYPE: DType
](
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Normalize a quaternion to unit length.

    Args:
        qx: Quaternion x.
        qy: Quaternion y.
        qz: Quaternion z.
        qw: Quaternion w.

    Returns:
        Normalized unit quaternion.
    """
    var length_sq = qx * qx + qy * qy + qz * qz + qw * qw
    var inv_length = Scalar[DTYPE](1.0) / sqrt(length_sq + Scalar[DTYPE](1e-12))
    return (qx * inv_length, qy * inv_length, qz * inv_length, qw * inv_length)


# =============================================================================
# Axis-Angle to Quaternion
# =============================================================================


def axis_angle_to_quat[
    DTYPE: DType
](
    ax: Scalar[DTYPE],
    ay: Scalar[DTYPE],
    az: Scalar[DTYPE],
    angle: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Convert axis-angle representation to quaternion.

    Args:
        ax: Rotation x-component of axis (should be normalized).
        ay: Rotation y-component of axis (should be normalized).
        az: Rotation z-component of axis (should be normalized).
        angle: Rotation angle in radians.

    Returns:
        Quaternion [x, y, z, w] = [sin(θ/2)*axis, cos(θ/2)].
    """
    comptime assert (
        DTYPE.is_floating_point()
    ), "DTYPE must be a floating point type"
    var half = angle.cast[DTYPE]() * 0.5
    var s = Scalar[DTYPE](sin(half))
    var c = Scalar[DTYPE](cos(half))

    return (ax * s, ay * s, az * s, c)


# =============================================================================
# Quaternion to Axis-Angle (for debugging/visualization)
# =============================================================================


def quat_to_axis_angle[
    DTYPE: DType
](
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Convert quaternion to axis-angle representation.

    Args:
        qx: Quaternion x.
        qy: Quaternion y.
        qz: Quaternion z.
        qw: Quaternion w.

    Returns:
        (axis_x, axis_y, axis_z, angle) where angle is in radians.
    """

    # Ensure qw is in valid range for acos
    var w_f64 = Float64(qw).clamp(-1.0, 1.0)

    # Compute angle
    var angle = Scalar[DTYPE](2.0 * acos(w_f64))

    # Compute axis (handle near-zero angle case)
    var sin_half = sqrt(Float64(qx * qx + qy * qy + qz * qz))
    if sin_half < 1e-10:
        # Near-identity rotation, axis is arbitrary
        return (Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1), angle)

    var inv_sin = Scalar[DTYPE](1.0 / sin_half)
    return (qx * inv_sin, qy * inv_sin, qz * inv_sin, angle)


# =============================================================================
# Quaternion Integration (for integrating angular velocity)
# =============================================================================


def quat_integrate[
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
    """Integrate quaternion with angular velocity, EXACTLY as MuJoCo does.

    `q(t+dt) = normalize(q(t)) * axisAngle(omega/|omega|, dt*|omega|)` —
    a RIGHT multiplication, i.e. omega is in the BODY-LOCAL frame, matching
    MuJoCo's free/ball-joint qvel convention.

    Verbatim `mju_quatIntegrate` (engine_util_spatial.c:241): normalize the
    axis, scale the angle by dt, build the rotation quaternion, normalize the
    current quat, right-multiply. The rotation applied is the exact
    exponential map, NOT a truncated series.

    ⚠ This was a FIRST-ORDER approximation (`q + 0.5*dt*q*omega`, then
    normalize) until 2026-07-30, which is a different integrator from
    MuJoCo's for any free- or ball-jointed model. Cost: the root quaternion
    was the single largest state discrepancy vs MuJoCo at EVERY step of a
    dm_control humanoid rollout (1.1e-8 after one step, where the 21 hinge
    DOFs sat at 4e-10), and it compounds. Only free-rooted models were
    affected, and their gates were loose enough to miss it — the Ant Euler
    gate budgets 1e-4 on qpos.

    Args:
        qx: Current orientation quaternion x.
        qy: Current orientation quaternion y.
        qz: Current orientation quaternion z.
        qw: Current orientation quaternion w.
        wx: Angular velocity x in body-local frame.
        wy: Angular velocity y in body-local frame.
        wz: Angular velocity z in body-local frame.
        dt: Time step.

    Returns:
        Updated (normalized) quaternion.
    """
    comptime assert (
        DTYPE.is_floating_point()
    ), "DTYPE must be a floating point type"
    # mju_normalize3 on the angular velocity; below mjMINVAL MuJoCo leaves the
    # angle at 0, which makes the rotation the identity.
    var wn = sqrt(wx * wx + wy * wy + wz * wz)
    var q = quat_normalize(qx, qy, qz, qw)
    if wn < Scalar[DTYPE](1e-15):
        return q

    var angle = dt * wn
    var half = Scalar[DTYPE](0.5) * angle
    var s = Scalar[DTYPE](sin(half)) / wn  # fold the axis normalization in
    var c = Scalar[DTYPE](cos(half))

    var rx = wx * s
    var ry = wy * s
    var rz = wz * s

    # q * qrot — right multiplication (local frame).
    return quat_mul(q[0], q[1], q[2], q[3], rx, ry, rz, c)


# =============================================================================
# GPU Quaternion Operations (InlineArray return for GPU compatibility)
# =============================================================================


@always_inline
def gpu_quat_mul[
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
) -> InlineArray[Scalar[DTYPE], 4]:
    """Quaternion multiplication a * b (GPU version with InlineArray return)."""
    var result = InlineArray[Scalar[DTYPE], 4](uninitialized=True)
    result[0] = aw * bx + ax * bw + ay * bz - az * by
    result[1] = aw * by - ax * bz + ay * bw + az * bx
    result[2] = aw * bz + ax * by - ay * bx + az * bw
    result[3] = aw * bw - ax * bx - ay * by - az * bz
    return result^


@always_inline
def gpu_quat_rotate[
    DTYPE: DType
](
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    vx: Scalar[DTYPE],
    vy: Scalar[DTYPE],
    vz: Scalar[DTYPE],
) -> InlineArray[Scalar[DTYPE], 3]:
    """Rotate vector v by quaternion q: q * v * q^-1 (GPU version)."""
    var t_x = Scalar[DTYPE](2) * (qy * vz - qz * vy)
    var t_y = Scalar[DTYPE](2) * (qz * vx - qx * vz)
    var t_z = Scalar[DTYPE](2) * (qx * vy - qy * vx)

    var result = InlineArray[Scalar[DTYPE], 3](uninitialized=True)
    result[0] = vx + qw * t_x + (qy * t_z - qz * t_y)
    result[1] = vy + qw * t_y + (qz * t_x - qx * t_z)
    result[2] = vz + qw * t_z + (qx * t_y - qy * t_x)
    return result^


@always_inline
def gpu_axis_angle_to_quat[
    DTYPE: DType
](
    axis_x: Scalar[DTYPE],
    axis_y: Scalar[DTYPE],
    axis_z: Scalar[DTYPE],
    angle: Scalar[DTYPE],
) -> InlineArray[Scalar[DTYPE], 4]:
    """Convert axis-angle to quaternion (GPU version)."""
    comptime assert (
        DTYPE.is_floating_point()
    ), "DTYPE must be a floating point type"
    var half = angle.cast[DTYPE]() * 0.5
    var s = Scalar[DTYPE](sin(half))
    var c = Scalar[DTYPE](cos(half))

    # ⚠ THE ZERO-AXIS GUARD IS A BRANCH, NOT AN EPSILON UNDER THE SQRT. This
    # was `1.0 / sqrt(len_sq + 1e-10)`, which for the unit axis every caller
    # actually passes returns `1 - 5e-11` instead of `1` — see
    # `gpu_quat_normalize` for the full story and the measurement.
    var len_sq = axis_x * axis_x + axis_y * axis_y + axis_z * axis_z
    var result = InlineArray[Scalar[DTYPE], 4](uninitialized=True)
    # Degenerate axis keeps the old formula, for the reason in
    # `gpu_quat_normalize` — this change is about the epsilon, not about what
    # a zero axis should mean.
    var inv_len: Scalar[DTYPE]
    if len_sq < Scalar[DTYPE](1e-6):
        inv_len = Scalar[DTYPE](1.0) / sqrt(len_sq + Scalar[DTYPE](1e-10))
    else:
        inv_len = Scalar[DTYPE](1.0) / sqrt(len_sq)

    result[0] = axis_x * inv_len * s
    result[1] = axis_y * inv_len * s
    result[2] = axis_z * inv_len * s
    result[3] = c
    return result^


@always_inline
def gpu_quat_normalize[
    DTYPE: DType
](
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
) -> InlineArray[Scalar[DTYPE], 4]:
    """Normalize quaternion (GPU version).

    ⚠ THE DEGENERATE GUARD IS A BRANCH, NOT AN EPSILON UNDER THE SQRT. This
    was `1.0 / sqrt(norm_sq + 1e-10)`, which POISONS THE COMMON CASE to
    protect a case that never happens: for an already-unit quaternion it
    returns `1/sqrt(1 + 1e-10)` = 0.99999999995, so every quaternion this
    function touched came out 5e-11 SHORT of unit, and every vector rotated by
    one was scaled by `|q|^2` = 1 - 1e-10.

    That is not rounding — it is a deterministic one-directional bias, and it
    was the cause of BOTH of the ~1e-10 residuals that had been filed as
    separate mysteries:

      * task #48 — quadruped's forward kinematics reproducing MuJoCo to only
        ~1e-10 where other models manage 1e-15, "showing up identically in
        cvel, qfrc_bias and tendon_invweight0". Every body quaternion in the
        chain is normalized here.
      * task #49 — capsule contact normals ~1e-10 off MuJoCo's while every
        sphere pair was exact. A capsule's world AXIS is its local quaternion
        rotated by the body's; spheres have no axis to corrupt, which is
        exactly why the error looked type-specific.

    The observed body quaternion was 0.99999999995 to every digit of
    `1/sqrt(1+1e-10)`. Branch on the degenerate case instead, so the
    arithmetic is exact for the input every caller actually passes.
    """
    var norm_sq = qx * qx + qy * qy + qz * qz + qw * qw
    var result = InlineArray[Scalar[DTYPE], 4](uninitialized=True)
    # ⚠ THE DEGENERATE BRANCH KEEPS THE OLD FORMULA ON PURPOSE, so that this
    # change is a pure precision fix and nothing else. Below 1e-6 the result is
    # bit-identical to what it always was; above it, exact.
    #
    # It is NOT bit-identical to return identity there instead — an A/B moved
    # `test_equality_tendon_fields`'s humanoid fingerprint by 0.83, which means
    # SOMETHING IN THAT MODEL NORMALIZES A NEAR-ZERO QUATERNION and takes this
    # path. (It is not what caused that file's golden to move 2.6%; the
    # precision fix did, with this branch held at the old formula.) Returning
    # `(0,0,0,0)` for a zero-norm input is not a rotation at all and is very
    # likely wrong, but which caller reaches it and what it should do there is
    # a separate question from the epsilon. Filed rather than guessed.
    var inv_norm: Scalar[DTYPE]
    if norm_sq < Scalar[DTYPE](1e-6):
        inv_norm = Scalar[DTYPE](1.0) / sqrt(norm_sq + Scalar[DTYPE](1e-10))
    else:
        inv_norm = Scalar[DTYPE](1.0) / sqrt(norm_sq)

    result[0] = qx * inv_norm
    result[1] = qy * inv_norm
    result[2] = qz * inv_norm
    result[3] = qw * inv_norm
    return result^

"""3D Hinge (Revolute) Joint.

A hinge joint constrains two bodies to rotate around a single axis,
like a door hinge or elbow joint. It has 1 degree of freedom (the rotation angle).

This is the most common joint type in MuJoCo locomotion environments.

GPU support: The GPU-compatible functions use only scalar operations,
no Vec3/Quat struct instantiation, following the physics2d pattern.
"""

from math import cos, sin, sqrt, acos, atan2
from layout import LayoutTensor, Layout

from ..math_gpu import atan2_gpu

from ..constants import (
    dtype,
    BODY_STATE_SIZE_3D,
    JOINT_DATA_SIZE_3D,
    IDX_PX,
    IDX_PY,
    IDX_PZ,
    IDX_QW,
    IDX_QX,
    IDX_QY,
    IDX_QZ,
    IDX_VX,
    IDX_VY,
    IDX_VZ,
    IDX_WX,
    IDX_WY,
    IDX_WZ,
    IDX_TX,
    IDX_TY,
    IDX_TZ,
    IDX_INV_MASS,
    IDX_IXX,
    IDX_IYY,
    IDX_IZZ,
    JOINT3D_TYPE,
    JOINT3D_BODY_A,
    JOINT3D_BODY_B,
    JOINT3D_ANCHOR_AX,
    JOINT3D_ANCHOR_AY,
    JOINT3D_ANCHOR_AZ,
    JOINT3D_ANCHOR_BX,
    JOINT3D_ANCHOR_BY,
    JOINT3D_ANCHOR_BZ,
    JOINT3D_AXIS_X,
    JOINT3D_AXIS_Y,
    JOINT3D_AXIS_Z,
    JOINT3D_POSITION,
    JOINT3D_VELOCITY,
    JOINT3D_MOTOR_TARGET,
    JOINT3D_MOTOR_KP,
    JOINT3D_MOTOR_KD,
    JOINT3D_MAX_FORCE,
    JOINT3D_LOWER_LIMIT,
    JOINT3D_UPPER_LIMIT,
    JOINT3D_FLAGS,
    JOINT3D_IMPULSE_X,
    JOINT3D_IMPULSE_Y,
    JOINT3D_IMPULSE_Z,
    JOINT3D_MOTOR_IMPULSE,
    JOINT_HINGE,
    JOINT3D_FLAG_LIMIT_ENABLED,
    JOINT3D_FLAG_MOTOR_ENABLED,
)

from math3d import Vec3, Quat


struct Hinge3D:
    """3D Hinge (Revolute) Joint Constraint Solver.

    Constrains two bodies to:
    1. Keep their anchor points coincident (3 linear constraints)
    2. Keep their axes aligned (2 angular constraints)

    The remaining DOF allows rotation around the shared axis.

    Key insight: The inertia tensor is stored in body-local frame, but constraint
    solving happens in world frame. We must transform the inverse inertia tensor
    to world frame: I_world^-1 = R * I_local^-1 * R^T
    """

    # =========================================================================
    # World-Frame Inverse Inertia Tensor Computation
    # =========================================================================

    @always_inline
    @staticmethod
    fn compute_world_inv_inertia(
        qw: Scalar[dtype], qx: Scalar[dtype], qy: Scalar[dtype], qz: Scalar[dtype],
        inv_ixx: Scalar[dtype], inv_iyy: Scalar[dtype], inv_izz: Scalar[dtype],
    ) -> Tuple[
        Scalar[dtype], Scalar[dtype], Scalar[dtype],  # Row 0: m00, m01, m02
        Scalar[dtype], Scalar[dtype], Scalar[dtype],  # Row 1: m10, m11, m12
        Scalar[dtype], Scalar[dtype], Scalar[dtype],  # Row 2: m20, m21, m22
    ]:
        """Compute world-frame inverse inertia tensor from quaternion and local diagonal inverse inertia.

        I_world^-1 = R * I_local^-1 * R^T

        For diagonal I_local^-1 = diag(inv_ixx, inv_iyy, inv_izz), this expands to:
        I_world^-1 = inv_ixx * r0⊗r0 + inv_iyy * r1⊗r1 + inv_izz * r2⊗r2

        Where r0, r1, r2 are the columns of the rotation matrix R, and ⊗ denotes outer product.
        """
        var two = Scalar[dtype](2.0)
        var one = Scalar[dtype](1.0)

        # Rotation matrix from quaternion (column-major: r0, r1, r2 are columns)
        # R = [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
        var r00 = one - two * (qy * qy + qz * qz)
        var r10 = two * (qx * qy + qw * qz)
        var r20 = two * (qx * qz - qw * qy)

        var r01 = two * (qx * qy - qw * qz)
        var r11 = one - two * (qx * qx + qz * qz)
        var r21 = two * (qy * qz + qw * qx)

        var r02 = two * (qx * qz + qw * qy)
        var r12 = two * (qy * qz - qw * qx)
        var r22 = one - two * (qx * qx + qy * qy)

        # I_world^-1 = inv_ixx * r0⊗r0 + inv_iyy * r1⊗r1 + inv_izz * r2⊗r2
        # (r⊗r)_ij = r_i * r_j
        var m00 = inv_ixx * r00 * r00 + inv_iyy * r01 * r01 + inv_izz * r02 * r02
        var m01 = inv_ixx * r00 * r10 + inv_iyy * r01 * r11 + inv_izz * r02 * r12
        var m02 = inv_ixx * r00 * r20 + inv_iyy * r01 * r21 + inv_izz * r02 * r22
        var m11 = inv_ixx * r10 * r10 + inv_iyy * r11 * r11 + inv_izz * r12 * r12
        var m12 = inv_ixx * r10 * r20 + inv_iyy * r11 * r21 + inv_izz * r12 * r22
        var m22 = inv_ixx * r20 * r20 + inv_iyy * r21 * r21 + inv_izz * r22 * r22

        # Matrix is symmetric: m10 = m01, m20 = m02, m21 = m12
        return (m00, m01, m02, m01, m11, m12, m02, m12, m22)

    @always_inline
    @staticmethod
    fn compute_skew_inv_inertia_skew(
        rx: Scalar[dtype], ry: Scalar[dtype], rz: Scalar[dtype],
        inv_i00: Scalar[dtype], inv_i01: Scalar[dtype], inv_i02: Scalar[dtype],
        inv_i11: Scalar[dtype], inv_i12: Scalar[dtype], inv_i22: Scalar[dtype],
    ) -> Tuple[
        Scalar[dtype], Scalar[dtype], Scalar[dtype],  # Row 0
        Scalar[dtype], Scalar[dtype], Scalar[dtype],  # Row 1
        Scalar[dtype], Scalar[dtype], Scalar[dtype],  # Row 2
    ]:
        """Compute skew(r)^T * I^-1 * skew(r) for the effective mass matrix.

        skew(r) = [[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]]

        This is the angular contribution to the effective mass matrix K.
        """
        # skew(r)^T = [[0, rz, -ry], [-rz, 0, rx], [ry, -rx, 0]]
        # First compute: temp = I^-1 * skew(r)
        # temp_00 = inv_i00*0 + inv_i01*rz + inv_i02*(-ry) = inv_i01*rz - inv_i02*ry
        # temp_01 = inv_i00*(-rz) + inv_i01*0 + inv_i02*rx = -inv_i00*rz + inv_i02*rx
        # temp_02 = inv_i00*ry + inv_i01*(-rx) + inv_i02*0 = inv_i00*ry - inv_i01*rx
        # ... and so on for other rows

        var t00 = inv_i01 * rz - inv_i02 * ry
        var t01 = -inv_i00 * rz + inv_i02 * rx
        var t02 = inv_i00 * ry - inv_i01 * rx

        var t10 = inv_i11 * rz - inv_i12 * ry
        var t11 = -inv_i01 * rz + inv_i12 * rx
        var t12 = inv_i01 * ry - inv_i11 * rx

        var t20 = inv_i12 * rz - inv_i22 * ry
        var t21 = -inv_i02 * rz + inv_i22 * rx
        var t22 = inv_i02 * ry - inv_i12 * rx

        # Now compute: skew(r)^T * temp
        # Result_00 = 0*t00 + rz*t10 + (-ry)*t20 = rz*t10 - ry*t20
        # Result_01 = 0*t01 + rz*t11 + (-ry)*t21 = rz*t11 - ry*t21
        # Result_02 = 0*t02 + rz*t12 + (-ry)*t22 = rz*t12 - ry*t22
        # Result_10 = (-rz)*t00 + 0*t10 + rx*t20 = -rz*t00 + rx*t20
        # ... etc

        var k00 = rz * t10 - ry * t20
        var k01 = rz * t11 - ry * t21
        var k02 = rz * t12 - ry * t22
        var k10 = -rz * t00 + rx * t20
        var k11 = -rz * t01 + rx * t21
        var k12 = -rz * t02 + rx * t22
        var k20 = ry * t00 - rx * t10
        var k21 = ry * t01 - rx * t11
        var k22 = ry * t02 - rx * t12

        return (k00, k01, k02, k10, k11, k12, k20, k21, k22)

    @always_inline
    @staticmethod
    fn invert_3x3_symmetric(
        m00: Scalar[dtype], m01: Scalar[dtype], m02: Scalar[dtype],
        m11: Scalar[dtype], m12: Scalar[dtype], m22: Scalar[dtype],
    ) -> Tuple[
        Scalar[dtype], Scalar[dtype], Scalar[dtype],  # inv_00, inv_01, inv_02
        Scalar[dtype], Scalar[dtype], Scalar[dtype],  # inv_11, inv_12, inv_22
    ]:
        """Invert a 3x3 symmetric matrix using adjugate method.

        Returns only the upper triangle (symmetric).
        """
        var eps = Scalar[dtype](1e-10)
        var one = Scalar[dtype](1.0)

        # Determinant of symmetric matrix
        # det = m00*(m11*m22 - m12*m12) - m01*(m01*m22 - m12*m02) + m02*(m01*m12 - m11*m02)
        var det = (
            m00 * (m11 * m22 - m12 * m12)
            - m01 * (m01 * m22 - m12 * m02)
            + m02 * (m01 * m12 - m11 * m02)
        )

        # Prevent division by zero
        if det > -eps and det < eps:
            if det >= Scalar[dtype](0.0):
                det = eps
            else:
                det = -eps

        var inv_det = one / det

        # Adjugate (cofactor matrix transposed) for symmetric matrix
        var adj00 = m11 * m22 - m12 * m12
        var adj01 = m02 * m12 - m01 * m22
        var adj02 = m01 * m12 - m02 * m11
        var adj11 = m00 * m22 - m02 * m02
        var adj12 = m01 * m02 - m00 * m12
        var adj22 = m00 * m11 - m01 * m01

        return (
            adj00 * inv_det, adj01 * inv_det, adj02 * inv_det,
            adj11 * inv_det, adj12 * inv_det, adj22 * inv_det,
        )

    @always_inline
    @staticmethod
    fn apply_inv_inertia(
        inv_i00: Scalar[dtype], inv_i01: Scalar[dtype], inv_i02: Scalar[dtype],
        inv_i11: Scalar[dtype], inv_i12: Scalar[dtype], inv_i22: Scalar[dtype],
        tx: Scalar[dtype], ty: Scalar[dtype], tz: Scalar[dtype],
    ) -> Tuple[Scalar[dtype], Scalar[dtype], Scalar[dtype]]:
        """Apply inverse inertia tensor to a torque/angular impulse vector.

        Returns I^-1 * t
        """
        var rx = inv_i00 * tx + inv_i01 * ty + inv_i02 * tz
        var ry = inv_i01 * tx + inv_i11 * ty + inv_i12 * tz
        var rz = inv_i02 * tx + inv_i12 * ty + inv_i22 * tz
        return (rx, ry, rz)

    # =========================================================================
    # Joint Initialization (CPU only - uses Vec3)
    # =========================================================================

    @staticmethod
    fn init_joint[
        BATCH: Int,
        STATE_SIZE: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        joint_idx: Int,
        body_a: Int,
        body_b: Int,
        anchor_a: Vec3,  # Local anchor on body A
        anchor_b: Vec3,  # Local anchor on body B
        axis: Vec3,  # Joint axis (in body A's local frame)
        lower_limit: Scalar[dtype] = -3.14159,
        upper_limit: Scalar[dtype] = 3.14159,
        motor_kp: Scalar[dtype] = 100.0,
        motor_kd: Scalar[dtype] = 10.0,
        max_force: Scalar[dtype] = 100.0,
    ):
        """Initialize a hinge joint between two bodies."""
        var joint_off = JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE_3D

        # Joint type
        state[env, joint_off + JOINT3D_TYPE] = Scalar[dtype](JOINT_HINGE)

        # Body indices
        state[env, joint_off + JOINT3D_BODY_A] = Scalar[dtype](body_a)
        state[env, joint_off + JOINT3D_BODY_B] = Scalar[dtype](body_b)

        # Local anchors
        state[env, joint_off + JOINT3D_ANCHOR_AX] = Scalar[dtype](anchor_a.x)
        state[env, joint_off + JOINT3D_ANCHOR_AY] = Scalar[dtype](anchor_a.y)
        state[env, joint_off + JOINT3D_ANCHOR_AZ] = Scalar[dtype](anchor_a.z)
        state[env, joint_off + JOINT3D_ANCHOR_BX] = Scalar[dtype](anchor_b.x)
        state[env, joint_off + JOINT3D_ANCHOR_BY] = Scalar[dtype](anchor_b.y)
        state[env, joint_off + JOINT3D_ANCHOR_BZ] = Scalar[dtype](anchor_b.z)

        # Joint axis (normalized)
        var axis_norm = axis.normalized()
        state[env, joint_off + JOINT3D_AXIS_X] = Scalar[dtype](axis_norm.x)
        state[env, joint_off + JOINT3D_AXIS_Y] = Scalar[dtype](axis_norm.y)
        state[env, joint_off + JOINT3D_AXIS_Z] = Scalar[dtype](axis_norm.z)

        # Joint state
        state[env, joint_off + JOINT3D_POSITION] = Scalar[dtype](0.0)
        state[env, joint_off + JOINT3D_VELOCITY] = Scalar[dtype](0.0)

        # Motor parameters
        state[env, joint_off + JOINT3D_MOTOR_TARGET] = Scalar[dtype](0.0)
        state[env, joint_off + JOINT3D_MOTOR_KP] = Scalar[dtype](motor_kp)
        state[env, joint_off + JOINT3D_MOTOR_KD] = Scalar[dtype](motor_kd)
        state[env, joint_off + JOINT3D_MAX_FORCE] = Scalar[dtype](max_force)

        # Limits
        state[env, joint_off + JOINT3D_LOWER_LIMIT] = Scalar[dtype](lower_limit)
        state[env, joint_off + JOINT3D_UPPER_LIMIT] = Scalar[dtype](upper_limit)

        # Enable limits and motor by default
        state[env, joint_off + JOINT3D_FLAGS] = Scalar[dtype](
            JOINT3D_FLAG_LIMIT_ENABLED | JOINT3D_FLAG_MOTOR_ENABLED
        )

        # Clear accumulated impulses
        state[env, joint_off + JOINT3D_IMPULSE_X] = Scalar[dtype](0.0)
        state[env, joint_off + JOINT3D_IMPULSE_Y] = Scalar[dtype](0.0)
        state[env, joint_off + JOINT3D_IMPULSE_Z] = Scalar[dtype](0.0)
        state[env, joint_off + JOINT3D_MOTOR_IMPULSE] = Scalar[dtype](0.0)

    # =========================================================================
    # GPU-Compatible Joint State Extraction (Scalar-only)
    # =========================================================================

    @always_inline
    @staticmethod
    fn get_joint_angle_gpu[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        joint_idx: Int,
    ) -> Scalar[dtype]:
        """Compute current joint angle from body orientations (GPU-compatible).

        Uses only scalar operations, no Vec3/Quat structs.
        """
        var joint_off = JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE_3D

        var body_a = Int(state[env, joint_off + JOINT3D_BODY_A])
        var body_b = Int(state[env, joint_off + JOINT3D_BODY_B])

        # Get body orientations as quaternion components
        var body_a_off = BODIES_OFFSET + body_a * BODY_STATE_SIZE_3D
        var body_b_off = BODIES_OFFSET + body_b * BODY_STATE_SIZE_3D

        var qa_w = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QW])
        var qa_x = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QX])
        var qa_y = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QY])
        var qa_z = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QZ])

        var qb_w = rebind[Scalar[dtype]](state[env, body_b_off + IDX_QW])
        var qb_x = rebind[Scalar[dtype]](state[env, body_b_off + IDX_QX])
        var qb_y = rebind[Scalar[dtype]](state[env, body_b_off + IDX_QY])
        var qb_z = rebind[Scalar[dtype]](state[env, body_b_off + IDX_QZ])

        # Joint axis in body A's frame
        var axis_x = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_X])
        var axis_y = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_Y])
        var axis_z = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_Z])

        # Relative quaternion: qrel = qa^-1 * qb
        # conjugate(qa) = (qa_w, -qa_x, -qa_y, -qa_z)
        var qa_conj_w = qa_w
        var qa_conj_x = -qa_x
        var qa_conj_y = -qa_y
        var qa_conj_z = -qa_z

        # Quaternion multiplication: qrel = qa_conj * qb
        var qrel_w = (
            qa_conj_w * qb_w
            - qa_conj_x * qb_x
            - qa_conj_y * qb_y
            - qa_conj_z * qb_z
        )
        var qrel_x = (
            qa_conj_w * qb_x
            + qa_conj_x * qb_w
            + qa_conj_y * qb_z
            - qa_conj_z * qb_y
        )
        var qrel_y = (
            qa_conj_w * qb_y
            - qa_conj_x * qb_z
            + qa_conj_y * qb_w
            + qa_conj_z * qb_x
        )
        var qrel_z = (
            qa_conj_w * qb_z
            + qa_conj_x * qb_y
            - qa_conj_y * qb_x
            + qa_conj_z * qb_w
        )

        # Extract rotation angle around the joint axis
        var dot_xyz_axis = qrel_x * axis_x + qrel_y * axis_y + qrel_z * axis_z
        var angle = Scalar[dtype](2.0) * atan2_gpu[dtype](dot_xyz_axis, qrel_w)

        return angle

    @always_inline
    @staticmethod
    fn get_joint_velocity_gpu[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        joint_idx: Int,
    ) -> Scalar[dtype]:
        """Compute current joint angular velocity (GPU-compatible)."""
        var joint_off = JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE_3D

        var body_a = Int(state[env, joint_off + JOINT3D_BODY_A])
        var body_b = Int(state[env, joint_off + JOINT3D_BODY_B])

        var body_a_off = BODIES_OFFSET + body_a * BODY_STATE_SIZE_3D
        var body_b_off = BODIES_OFFSET + body_b * BODY_STATE_SIZE_3D

        # Get body angular velocities
        var wa_x = rebind[Scalar[dtype]](state[env, body_a_off + IDX_WX])
        var wa_y = rebind[Scalar[dtype]](state[env, body_a_off + IDX_WY])
        var wa_z = rebind[Scalar[dtype]](state[env, body_a_off + IDX_WZ])

        var wb_x = rebind[Scalar[dtype]](state[env, body_b_off + IDX_WX])
        var wb_y = rebind[Scalar[dtype]](state[env, body_b_off + IDX_WY])
        var wb_z = rebind[Scalar[dtype]](state[env, body_b_off + IDX_WZ])

        # Get orientation of body A to transform axis to world frame
        var qa_w = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QW])
        var qa_x = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QX])
        var qa_y = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QY])
        var qa_z = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QZ])

        # Joint axis in local frame
        var axis_local_x = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_X])
        var axis_local_y = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_Y])
        var axis_local_z = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_Z])

        # Transform axis to world frame using quaternion rotation
        # v' = v + 2*qw*(q_xyz x v) + 2*(q_xyz x (q_xyz x v))
        var cx = qa_y * axis_local_z - qa_z * axis_local_y
        var cy = qa_z * axis_local_x - qa_x * axis_local_z
        var cz = qa_x * axis_local_y - qa_y * axis_local_x
        var ccx = qa_y * cz - qa_z * cy
        var ccy = qa_z * cx - qa_x * cz
        var ccz = qa_x * cy - qa_y * cx
        var two = Scalar[dtype](2.0)
        var axis_world_x = axis_local_x + two * qa_w * cx + two * ccx
        var axis_world_y = axis_local_y + two * qa_w * cy + two * ccy
        var axis_world_z = axis_local_z + two * qa_w * cz + two * ccz

        # Relative angular velocity projected onto joint axis
        var rel_omega_x = wb_x - wa_x
        var rel_omega_y = wb_y - wa_y
        var rel_omega_z = wb_z - wa_z
        var joint_vel = (
            rel_omega_x * axis_world_x
            + rel_omega_y * axis_world_y
            + rel_omega_z * axis_world_z
        )

        return joint_vel

    # =========================================================================
    # GPU-Compatible Direct Torque Application (Scalar-only)
    # =========================================================================

    @always_inline
    @staticmethod
    fn apply_direct_torque_gpu[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        joint_idx: Int,
        torque: Scalar[dtype],
    ):
        """Apply direct torque to joint (GPU-compatible)."""
        var joint_off = JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE_3D

        var body_a = Int(state[env, joint_off + JOINT3D_BODY_A])
        var body_b = Int(state[env, joint_off + JOINT3D_BODY_B])

        var body_a_off = BODIES_OFFSET + body_a * BODY_STATE_SIZE_3D
        var body_b_off = BODIES_OFFSET + body_b * BODY_STATE_SIZE_3D

        # Get orientation of body A to get world-space axis
        var qa_w = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QW])
        var qa_x = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QX])
        var qa_y = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QY])
        var qa_z = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QZ])

        # Joint axis in local frame
        var axis_local_x = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_X])
        var axis_local_y = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_Y])
        var axis_local_z = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_Z])

        # Transform axis to world frame using quaternion rotation
        var cx = qa_y * axis_local_z - qa_z * axis_local_y
        var cy = qa_z * axis_local_x - qa_x * axis_local_z
        var cz = qa_x * axis_local_y - qa_y * axis_local_x
        var ccx = qa_y * cz - qa_z * cy
        var ccy = qa_z * cx - qa_x * cz
        var ccz = qa_x * cy - qa_y * cx
        var two = Scalar[dtype](2.0)
        var axis_world_x = axis_local_x + two * qa_w * cx + two * ccx
        var axis_world_y = axis_local_y + two * qa_w * cy + two * ccy
        var axis_world_z = axis_local_z + two * qa_w * cz + two * ccz

        # Apply torque vector: tau = axis * torque
        var tau_x = axis_world_x * torque
        var tau_y = axis_world_y * torque
        var tau_z = axis_world_z * torque

        # Apply equal and opposite torques to the bodies
        state[env, body_a_off + IDX_TX] = state[env, body_a_off + IDX_TX] - tau_x
        state[env, body_a_off + IDX_TY] = state[env, body_a_off + IDX_TY] - tau_y
        state[env, body_a_off + IDX_TZ] = state[env, body_a_off + IDX_TZ] - tau_z

        state[env, body_b_off + IDX_TX] = state[env, body_b_off + IDX_TX] + tau_x
        state[env, body_b_off + IDX_TY] = state[env, body_b_off + IDX_TY] + tau_y
        state[env, body_b_off + IDX_TZ] = state[env, body_b_off + IDX_TZ] + tau_z

    # =========================================================================
    # GPU-Compatible Velocity Constraint Solving (Scalar-only)
    # =========================================================================

    @always_inline
    @staticmethod
    fn solve_velocity_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        joint_idx: Int,
        dt: Scalar[dtype],
    ):
        """Solve velocity constraints for hinge joint (GPU-compatible).

        Uses a stable Jacobi-style approach where each axis is processed
        independently with a scalar effective mass. This is more robust than
        the full 3x3 matrix approach for real-time physics.
        """
        var joint_off = JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE_3D

        # Skip if not a hinge joint
        var joint_type = Int(state[env, joint_off + JOINT3D_TYPE])
        if joint_type != JOINT_HINGE:
            return

        var body_a = Int(state[env, joint_off + JOINT3D_BODY_A])
        var body_b = Int(state[env, joint_off + JOINT3D_BODY_B])

        var body_a_off = BODIES_OFFSET + body_a * BODY_STATE_SIZE_3D
        var body_b_off = BODIES_OFFSET + body_b * BODY_STATE_SIZE_3D

        # Get body properties
        var inv_ma = rebind[Scalar[dtype]](state[env, body_a_off + IDX_INV_MASS])
        var inv_mb = rebind[Scalar[dtype]](state[env, body_b_off + IDX_INV_MASS])

        # Get inertia (diagonal, local frame)
        var ixx_a = rebind[Scalar[dtype]](state[env, body_a_off + IDX_IXX])
        var iyy_a = rebind[Scalar[dtype]](state[env, body_a_off + IDX_IYY])
        var izz_a = rebind[Scalar[dtype]](state[env, body_a_off + IDX_IZZ])
        var ixx_b = rebind[Scalar[dtype]](state[env, body_b_off + IDX_IXX])
        var iyy_b = rebind[Scalar[dtype]](state[env, body_b_off + IDX_IYY])
        var izz_b = rebind[Scalar[dtype]](state[env, body_b_off + IDX_IZZ])

        var eps = Scalar[dtype](1e-10)
        var one = Scalar[dtype](1.0)
        var three = Scalar[dtype](3.0)

        # Use averaged scalar inertia for stability (common in game physics)
        var avg_inv_i_a = (one / (ixx_a + eps) + one / (iyy_a + eps) + one / (izz_a + eps)) / three
        var avg_inv_i_b = (one / (ixx_b + eps) + one / (iyy_b + eps) + one / (izz_b + eps)) / three

        # Get orientations
        var qa_w = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QW])
        var qa_x = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QX])
        var qa_y = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QY])
        var qa_z = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QZ])
        var qb_w = rebind[Scalar[dtype]](state[env, body_b_off + IDX_QW])
        var qb_x = rebind[Scalar[dtype]](state[env, body_b_off + IDX_QX])
        var qb_y = rebind[Scalar[dtype]](state[env, body_b_off + IDX_QY])
        var qb_z = rebind[Scalar[dtype]](state[env, body_b_off + IDX_QZ])

        # Get velocities
        var va_x = rebind[Scalar[dtype]](state[env, body_a_off + IDX_VX])
        var va_y = rebind[Scalar[dtype]](state[env, body_a_off + IDX_VY])
        var va_z = rebind[Scalar[dtype]](state[env, body_a_off + IDX_VZ])
        var vb_x = rebind[Scalar[dtype]](state[env, body_b_off + IDX_VX])
        var vb_y = rebind[Scalar[dtype]](state[env, body_b_off + IDX_VY])
        var vb_z = rebind[Scalar[dtype]](state[env, body_b_off + IDX_VZ])
        var wa_x = rebind[Scalar[dtype]](state[env, body_a_off + IDX_WX])
        var wa_y = rebind[Scalar[dtype]](state[env, body_a_off + IDX_WY])
        var wa_z = rebind[Scalar[dtype]](state[env, body_a_off + IDX_WZ])
        var wb_x = rebind[Scalar[dtype]](state[env, body_b_off + IDX_WX])
        var wb_y = rebind[Scalar[dtype]](state[env, body_b_off + IDX_WY])
        var wb_z = rebind[Scalar[dtype]](state[env, body_b_off + IDX_WZ])

        # Local anchors
        var anchor_a_local_x = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_ANCHOR_AX])
        var anchor_a_local_y = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_ANCHOR_AY])
        var anchor_a_local_z = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_ANCHOR_AZ])
        var anchor_b_local_x = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_ANCHOR_BX])
        var anchor_b_local_y = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_ANCHOR_BY])
        var anchor_b_local_z = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_ANCHOR_BZ])

        # Transform anchors to world frame (quaternion rotation)
        var two = Scalar[dtype](2.0)

        # ra = qa.rotate_vec(anchor_a_local)
        var ca_x = qa_y * anchor_a_local_z - qa_z * anchor_a_local_y
        var ca_y = qa_z * anchor_a_local_x - qa_x * anchor_a_local_z
        var ca_z = qa_x * anchor_a_local_y - qa_y * anchor_a_local_x
        var cca_x = qa_y * ca_z - qa_z * ca_y
        var cca_y = qa_z * ca_x - qa_x * ca_z
        var cca_z = qa_x * ca_y - qa_y * ca_x
        var ra_x = anchor_a_local_x + two * qa_w * ca_x + two * cca_x
        var ra_y = anchor_a_local_y + two * qa_w * ca_y + two * cca_y
        var ra_z = anchor_a_local_z + two * qa_w * ca_z + two * cca_z

        # rb = qb.rotate_vec(anchor_b_local)
        var cb_x = qb_y * anchor_b_local_z - qb_z * anchor_b_local_y
        var cb_y = qb_z * anchor_b_local_x - qb_x * anchor_b_local_z
        var cb_z = qb_x * anchor_b_local_y - qb_y * anchor_b_local_x
        var ccb_x = qb_y * cb_z - qb_z * cb_y
        var ccb_y = qb_z * cb_x - qb_x * cb_z
        var ccb_z = qb_x * cb_y - qb_y * cb_x
        var rb_x = anchor_b_local_x + two * qb_w * cb_x + two * ccb_x
        var rb_y = anchor_b_local_y + two * qb_w * cb_y + two * ccb_y
        var rb_z = anchor_b_local_z + two * qb_w * cb_z + two * ccb_z

        # Velocity at anchor points: v_anchor = v + w × r
        var va_anchor_x = va_x + (wa_y * ra_z - wa_z * ra_y)
        var va_anchor_y = va_y + (wa_z * ra_x - wa_x * ra_z)
        var va_anchor_z = va_z + (wa_x * ra_y - wa_y * ra_x)
        var vb_anchor_x = vb_x + (wb_y * rb_z - wb_z * rb_y)
        var vb_anchor_y = vb_y + (wb_z * rb_x - wb_x * rb_z)
        var vb_anchor_z = vb_z + (wb_x * rb_y - wb_y * rb_x)

        # Relative velocity (should be zero for joint constraint)
        var cdot_x = vb_anchor_x - va_anchor_x
        var cdot_y = vb_anchor_y - va_anchor_y
        var cdot_z = vb_anchor_z - va_anchor_z

        # Simple proportional velocity correction (very stable)
        # Apply a fraction of the relative velocity as a correction, scaled by mass
        var correction_factor = Scalar[dtype](0.8)  # Apply 80% of the error per iteration

        # Total inverse mass for proportional distribution
        var total_inv_mass = inv_ma + inv_mb + eps

        # Simple impulse: directly oppose the relative velocity
        # This is equivalent to assuming diagonal effective mass = 1/total_inv_mass
        var impulse_x = -cdot_x * correction_factor / total_inv_mass
        var impulse_y = -cdot_y * correction_factor / total_inv_mass
        var impulse_z = -cdot_z * correction_factor / total_inv_mass

        # Clamp impulse magnitude to prevent numerical explosion
        var impulse_sq = impulse_x * impulse_x + impulse_y * impulse_y + impulse_z * impulse_z
        var max_impulse = Scalar[dtype](1.0)  # Conservative max impulse
        if impulse_sq > max_impulse * max_impulse:
            var scale = max_impulse / sqrt(impulse_sq)
            impulse_x = impulse_x * scale
            impulse_y = impulse_y * scale
            impulse_z = impulse_z * scale

        # Apply linear impulse
        var new_va_x = va_x - impulse_x * inv_ma
        var new_va_y = va_y - impulse_y * inv_ma
        var new_va_z = va_z - impulse_z * inv_ma
        var new_vb_x = vb_x + impulse_x * inv_mb
        var new_vb_y = vb_y + impulse_y * inv_mb
        var new_vb_z = vb_z + impulse_z * inv_mb

        # Apply angular impulse using averaged scalar inertia
        # delta_omega = avg_inv_I * (r × impulse)
        var ra_cross_impulse_x = ra_y * impulse_z - ra_z * impulse_y
        var ra_cross_impulse_y = ra_z * impulse_x - ra_x * impulse_z
        var ra_cross_impulse_z = ra_x * impulse_y - ra_y * impulse_x

        # Apply angular impulses with full strength
        # w_a' = w_a - avg_inv_I_a * (r_a × λ)
        var new_wa_x = wa_x - avg_inv_i_a * ra_cross_impulse_x
        var new_wa_y = wa_y - avg_inv_i_a * ra_cross_impulse_y
        var new_wa_z = wa_z - avg_inv_i_a * ra_cross_impulse_z

        var rb_cross_impulse_x = rb_y * impulse_z - rb_z * impulse_y
        var rb_cross_impulse_y = rb_z * impulse_x - rb_x * impulse_z
        var rb_cross_impulse_z = rb_x * impulse_y - rb_y * impulse_x

        # w_b' = w_b + avg_inv_I_b * (r_b × λ)
        var new_wb_x = wb_x + avg_inv_i_b * rb_cross_impulse_x
        var new_wb_y = wb_y + avg_inv_i_b * rb_cross_impulse_y
        var new_wb_z = wb_z + avg_inv_i_b * rb_cross_impulse_z

        # Write back velocities
        state[env, body_a_off + IDX_VX] = new_va_x
        state[env, body_a_off + IDX_VY] = new_va_y
        state[env, body_a_off + IDX_VZ] = new_va_z
        state[env, body_a_off + IDX_WX] = new_wa_x
        state[env, body_a_off + IDX_WY] = new_wa_y
        state[env, body_a_off + IDX_WZ] = new_wa_z

        state[env, body_b_off + IDX_VX] = new_vb_x
        state[env, body_b_off + IDX_VY] = new_vb_y
        state[env, body_b_off + IDX_VZ] = new_vb_z
        state[env, body_b_off + IDX_WX] = new_wb_x
        state[env, body_b_off + IDX_WY] = new_wb_y
        state[env, body_b_off + IDX_WZ] = new_wb_z

        # Handle motor constraint
        var flags = Int(state[env, joint_off + JOINT3D_FLAGS])
        if flags & JOINT3D_FLAG_MOTOR_ENABLED:
            Self._apply_motor_gpu[
                BATCH,
                NUM_BODIES,
                MAX_JOINTS,
                STATE_SIZE,
                BODIES_OFFSET,
                JOINTS_OFFSET,
            ](state, env, joint_idx, dt)

        # Handle joint angle limits (velocity constraint)
        if flags & JOINT3D_FLAG_LIMIT_ENABLED:
            Self._apply_limits_gpu[
                BATCH,
                NUM_BODIES,
                MAX_JOINTS,
                STATE_SIZE,
                BODIES_OFFSET,
                JOINTS_OFFSET,
            ](state, env, joint_idx)

    @always_inline
    @staticmethod
    fn _apply_motor_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        joint_idx: Int,
        dt: Scalar[dtype],
    ):
        """Apply PD motor control to joint (GPU-compatible)."""
        var joint_off = JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE_3D

        # Get motor parameters
        var target = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_MOTOR_TARGET])
        var kp = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_MOTOR_KP])
        var kd = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_MOTOR_KD])
        var max_force = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_MAX_FORCE])

        # Get current joint state
        var current_angle = Self.get_joint_angle_gpu[
            BATCH, STATE_SIZE, BODIES_OFFSET, JOINTS_OFFSET
        ](state, env, joint_idx)
        var current_vel = Self.get_joint_velocity_gpu[
            BATCH, STATE_SIZE, BODIES_OFFSET, JOINTS_OFFSET
        ](state, env, joint_idx)

        # PD control: torque = kp * (target - current) - kd * velocity
        var error = target - current_angle
        var torque = kp * error - kd * current_vel

        # Clamp to max force
        if torque > max_force:
            torque = max_force
        if torque < -max_force:
            torque = -max_force

        # Get body indices and properties
        var body_a = Int(state[env, joint_off + JOINT3D_BODY_A])
        var body_b = Int(state[env, joint_off + JOINT3D_BODY_B])

        var body_a_off = BODIES_OFFSET + body_a * BODY_STATE_SIZE_3D
        var body_b_off = BODIES_OFFSET + body_b * BODY_STATE_SIZE_3D

        # Get orientation of body A to get world-space axis
        var qa_w = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QW])
        var qa_x = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QX])
        var qa_y = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QY])
        var qa_z = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QZ])

        var axis_local_x = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_X])
        var axis_local_y = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_Y])
        var axis_local_z = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_Z])

        # Transform axis to world frame
        var two = Scalar[dtype](2.0)
        var cx = qa_y * axis_local_z - qa_z * axis_local_y
        var cy = qa_z * axis_local_x - qa_x * axis_local_z
        var cz = qa_x * axis_local_y - qa_y * axis_local_x
        var ccx = qa_y * cz - qa_z * cy
        var ccy = qa_z * cx - qa_x * cz
        var ccz = qa_x * cy - qa_y * cx
        var axis_world_x = axis_local_x + two * qa_w * cx + two * ccx
        var axis_world_y = axis_local_y + two * qa_w * cy + two * ccy
        var axis_world_z = axis_local_z + two * qa_w * cz + two * ccz

        # Compute angular impulse
        var impulse_x = axis_world_x * torque * dt
        var impulse_y = axis_world_y * torque * dt
        var impulse_z = axis_world_z * torque * dt

        # Get inverse inertia (simplified diagonal average)
        var eps = Scalar[dtype](1e-10)
        var one = Scalar[dtype](1.0)
        var three = Scalar[dtype](3.0)
        var inv_ia_avg = (
            one / (rebind[Scalar[dtype]](state[env, body_a_off + IDX_IXX]) + eps)
            + one / (rebind[Scalar[dtype]](state[env, body_a_off + IDX_IYY]) + eps)
            + one / (rebind[Scalar[dtype]](state[env, body_a_off + IDX_IZZ]) + eps)
        ) / three

        var inv_ib_avg = (
            one / (rebind[Scalar[dtype]](state[env, body_b_off + IDX_IXX]) + eps)
            + one / (rebind[Scalar[dtype]](state[env, body_b_off + IDX_IYY]) + eps)
            + one / (rebind[Scalar[dtype]](state[env, body_b_off + IDX_IZZ]) + eps)
        ) / three

        # Apply equal and opposite angular impulses
        state[env, body_a_off + IDX_WX] = (
            state[env, body_a_off + IDX_WX] - impulse_x * inv_ia_avg
        )
        state[env, body_a_off + IDX_WY] = (
            state[env, body_a_off + IDX_WY] - impulse_y * inv_ia_avg
        )
        state[env, body_a_off + IDX_WZ] = (
            state[env, body_a_off + IDX_WZ] - impulse_z * inv_ia_avg
        )

        state[env, body_b_off + IDX_WX] = (
            state[env, body_b_off + IDX_WX] + impulse_x * inv_ib_avg
        )
        state[env, body_b_off + IDX_WY] = (
            state[env, body_b_off + IDX_WY] + impulse_y * inv_ib_avg
        )
        state[env, body_b_off + IDX_WZ] = (
            state[env, body_b_off + IDX_WZ] + impulse_z * inv_ib_avg
        )

    @always_inline
    @staticmethod
    fn _apply_limits_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        joint_idx: Int,
    ):
        """Apply joint angle limits (GPU-compatible velocity constraint).

        When the joint is at or beyond its limit, prevents velocity that would
        push it further past the limit by zeroing out the velocity component
        along the joint axis. Also applies a restoring impulse if past the limit.
        """
        var joint_off = JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE_3D

        # Get joint limits
        var lower_limit = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_LOWER_LIMIT])
        var upper_limit = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_UPPER_LIMIT])

        # Get current joint angle and velocity
        var current_angle = Hinge3D.get_joint_angle_gpu[
            BATCH, STATE_SIZE, BODIES_OFFSET, JOINTS_OFFSET
        ](state, env, joint_idx)
        var current_vel = Hinge3D.get_joint_velocity_gpu[
            BATCH, STATE_SIZE, BODIES_OFFSET, JOINTS_OFFSET
        ](state, env, joint_idx)

        # Calculate required impulse to enforce limits
        # Simple approach: just stop velocity that would push past limit
        var limit_impulse = Scalar[dtype](0.0)
        var zero = Scalar[dtype](0.0)

        if current_angle <= lower_limit:
            # At lower limit - stop any negative velocity
            if current_vel < zero:
                limit_impulse = -current_vel
        elif current_angle >= upper_limit:
            # At upper limit - stop any positive velocity
            if current_vel > zero:
                limit_impulse = -current_vel

        # Apply limit impulse if needed
        if limit_impulse != zero:
            var body_a = Int(state[env, joint_off + JOINT3D_BODY_A])
            var body_b = Int(state[env, joint_off + JOINT3D_BODY_B])

            var body_a_off = BODIES_OFFSET + body_a * BODY_STATE_SIZE_3D
            var body_b_off = BODIES_OFFSET + body_b * BODY_STATE_SIZE_3D

            # Get orientation of body A to transform axis to world frame
            var qa_w = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QW])
            var qa_x = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QX])
            var qa_y = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QY])
            var qa_z = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QZ])

            var axis_local_x = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_X])
            var axis_local_y = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_Y])
            var axis_local_z = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_Z])

            # Transform axis to world frame
            var two = Scalar[dtype](2.0)
            var cx = qa_y * axis_local_z - qa_z * axis_local_y
            var cy = qa_z * axis_local_x - qa_x * axis_local_z
            var cz = qa_x * axis_local_y - qa_y * axis_local_x
            var ccx = qa_y * cz - qa_z * cy
            var ccy = qa_z * cx - qa_x * cz
            var ccz = qa_x * cy - qa_y * cx
            var axis_world_x = axis_local_x + two * qa_w * cx + two * ccx
            var axis_world_y = axis_local_y + two * qa_w * cy + two * ccy
            var axis_world_z = axis_local_z + two * qa_w * cz + two * ccz

            # Get inverse inertia (simplified diagonal average)
            var eps = Scalar[dtype](1e-10)
            var one = Scalar[dtype](1.0)
            var three = Scalar[dtype](3.0)
            var inv_ia_avg = (
                one / (rebind[Scalar[dtype]](state[env, body_a_off + IDX_IXX]) + eps)
                + one / (rebind[Scalar[dtype]](state[env, body_a_off + IDX_IYY]) + eps)
                + one / (rebind[Scalar[dtype]](state[env, body_a_off + IDX_IZZ]) + eps)
            ) / three

            var inv_ib_avg = (
                one / (rebind[Scalar[dtype]](state[env, body_b_off + IDX_IXX]) + eps)
                + one / (rebind[Scalar[dtype]](state[env, body_b_off + IDX_IYY]) + eps)
                + one / (rebind[Scalar[dtype]](state[env, body_b_off + IDX_IZZ]) + eps)
            ) / three

            # Effective inertia for angular constraint around axis
            var eff_inertia = inv_ia_avg + inv_ib_avg
            if eff_inertia < eps:
                eff_inertia = eps

            # Compute angular impulse magnitude
            var impulse_mag = limit_impulse / eff_inertia

            # Angular impulse vector along axis
            var impulse_x = axis_world_x * impulse_mag
            var impulse_y = axis_world_y * impulse_mag
            var impulse_z = axis_world_z * impulse_mag

            # Apply equal and opposite angular impulses
            state[env, body_a_off + IDX_WX] = (
                state[env, body_a_off + IDX_WX] - impulse_x * inv_ia_avg
            )
            state[env, body_a_off + IDX_WY] = (
                state[env, body_a_off + IDX_WY] - impulse_y * inv_ia_avg
            )
            state[env, body_a_off + IDX_WZ] = (
                state[env, body_a_off + IDX_WZ] - impulse_z * inv_ia_avg
            )

            state[env, body_b_off + IDX_WX] = (
                state[env, body_b_off + IDX_WX] + impulse_x * inv_ib_avg
            )
            state[env, body_b_off + IDX_WY] = (
                state[env, body_b_off + IDX_WY] + impulse_y * inv_ib_avg
            )
            state[env, body_b_off + IDX_WZ] = (
                state[env, body_b_off + IDX_WZ] + impulse_z * inv_ib_avg
            )

    # =========================================================================
    # GPU-Compatible Position Constraint Solving (Scalar-only)
    # =========================================================================

    @always_inline
    @staticmethod
    fn solve_position_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        joint_idx: Int,
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ):
        """Solve position constraints to correct drift (GPU-compatible).

        Uses simplified scalar effective mass for stability.
        Only applies linear position correction - no angular correction to avoid instability.
        """
        var joint_off = JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE_3D

        var joint_type = Int(state[env, joint_off + JOINT3D_TYPE])
        if joint_type != JOINT_HINGE:
            return

        var body_a = Int(state[env, joint_off + JOINT3D_BODY_A])
        var body_b = Int(state[env, joint_off + JOINT3D_BODY_B])

        var body_a_off = BODIES_OFFSET + body_a * BODY_STATE_SIZE_3D
        var body_b_off = BODIES_OFFSET + body_b * BODY_STATE_SIZE_3D

        # Get positions and orientations
        var pa_x = rebind[Scalar[dtype]](state[env, body_a_off + IDX_PX])
        var pa_y = rebind[Scalar[dtype]](state[env, body_a_off + IDX_PY])
        var pa_z = rebind[Scalar[dtype]](state[env, body_a_off + IDX_PZ])
        var pb_x = rebind[Scalar[dtype]](state[env, body_b_off + IDX_PX])
        var pb_y = rebind[Scalar[dtype]](state[env, body_b_off + IDX_PY])
        var pb_z = rebind[Scalar[dtype]](state[env, body_b_off + IDX_PZ])

        var qa_w = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QW])
        var qa_x = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QX])
        var qa_y = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QY])
        var qa_z = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QZ])
        var qb_w = rebind[Scalar[dtype]](state[env, body_b_off + IDX_QW])
        var qb_x = rebind[Scalar[dtype]](state[env, body_b_off + IDX_QX])
        var qb_y = rebind[Scalar[dtype]](state[env, body_b_off + IDX_QY])
        var qb_z = rebind[Scalar[dtype]](state[env, body_b_off + IDX_QZ])

        # Get mass and inertia properties
        var inv_ma = rebind[Scalar[dtype]](state[env, body_a_off + IDX_INV_MASS])
        var inv_mb = rebind[Scalar[dtype]](state[env, body_b_off + IDX_INV_MASS])

        var ixx_a = rebind[Scalar[dtype]](state[env, body_a_off + IDX_IXX])
        var iyy_a = rebind[Scalar[dtype]](state[env, body_a_off + IDX_IYY])
        var izz_a = rebind[Scalar[dtype]](state[env, body_a_off + IDX_IZZ])
        var ixx_b = rebind[Scalar[dtype]](state[env, body_b_off + IDX_IXX])
        var iyy_b = rebind[Scalar[dtype]](state[env, body_b_off + IDX_IYY])
        var izz_b = rebind[Scalar[dtype]](state[env, body_b_off + IDX_IZZ])

        var eps = Scalar[dtype](1e-10)
        var one = Scalar[dtype](1.0)
        var three = Scalar[dtype](3.0)

        # Use averaged scalar inertia for stability
        var avg_inv_i_a = (one / (ixx_a + eps) + one / (iyy_a + eps) + one / (izz_a + eps)) / three
        var avg_inv_i_b = (one / (ixx_b + eps) + one / (iyy_b + eps) + one / (izz_b + eps)) / three

        # Local anchors
        var anchor_a_local_x = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_ANCHOR_AX])
        var anchor_a_local_y = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_ANCHOR_AY])
        var anchor_a_local_z = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_ANCHOR_AZ])
        var anchor_b_local_x = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_ANCHOR_BX])
        var anchor_b_local_y = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_ANCHOR_BY])
        var anchor_b_local_z = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_ANCHOR_BZ])

        # Transform anchors to world frame (quaternion rotation)
        var two = Scalar[dtype](2.0)

        var ca_x = qa_y * anchor_a_local_z - qa_z * anchor_a_local_y
        var ca_y = qa_z * anchor_a_local_x - qa_x * anchor_a_local_z
        var ca_z = qa_x * anchor_a_local_y - qa_y * anchor_a_local_x
        var cca_x = qa_y * ca_z - qa_z * ca_y
        var cca_y = qa_z * ca_x - qa_x * ca_z
        var cca_z = qa_x * ca_y - qa_y * ca_x
        var ra_x = anchor_a_local_x + two * qa_w * ca_x + two * cca_x
        var ra_y = anchor_a_local_y + two * qa_w * ca_y + two * cca_y
        var ra_z = anchor_a_local_z + two * qa_w * ca_z + two * cca_z

        var cb_x = qb_y * anchor_b_local_z - qb_z * anchor_b_local_y
        var cb_y = qb_z * anchor_b_local_x - qb_x * anchor_b_local_z
        var cb_z = qb_x * anchor_b_local_y - qb_y * anchor_b_local_x
        var ccb_x = qb_y * cb_z - qb_z * cb_y
        var ccb_y = qb_z * cb_x - qb_x * cb_z
        var ccb_z = qb_x * cb_y - qb_y * cb_x
        var rb_x = anchor_b_local_x + two * qb_w * cb_x + two * ccb_x
        var rb_y = anchor_b_local_y + two * qb_w * cb_y + two * ccb_y
        var rb_z = anchor_b_local_z + two * qb_w * cb_z + two * ccb_z

        # World anchors
        var anchor_a_world_x = pa_x + ra_x
        var anchor_a_world_y = pa_y + ra_y
        var anchor_a_world_z = pa_z + ra_z
        var anchor_b_world_x = pb_x + rb_x
        var anchor_b_world_y = pb_y + rb_y
        var anchor_b_world_z = pb_z + rb_z

        # Position error
        var cx = anchor_b_world_x - anchor_a_world_x
        var cy = anchor_b_world_y - anchor_a_world_y
        var cz = anchor_b_world_z - anchor_a_world_z
        var error_sq = cx * cx + cy * cy + cz * cz
        var error_mag = sqrt(error_sq)

        if error_mag < slop:
            var flags = Int(state[env, joint_off + JOINT3D_FLAGS])
            if flags & JOINT3D_FLAG_LIMIT_ENABLED:
                Self._apply_limits_position_gpu[
                    BATCH, NUM_BODIES, MAX_JOINTS, STATE_SIZE, BODIES_OFFSET, JOINTS_OFFSET
                ](state, env, joint_idx, baumgarte)
            return

        # Simple proportional position correction
        var total_inv_mass = inv_ma + inv_mb + eps

        # Apply position correction proportional to error
        var corr_x = -baumgarte * cx / total_inv_mass
        var corr_y = -baumgarte * cy / total_inv_mass
        var corr_z = -baumgarte * cz / total_inv_mass

        # Clamp correction magnitude
        var corr_sq = corr_x * corr_x + corr_y * corr_y + corr_z * corr_z
        var max_corr = Scalar[dtype](0.2)  # Conservative max position correction
        if corr_sq > max_corr * max_corr:
            var scale = max_corr / sqrt(corr_sq)
            corr_x = corr_x * scale
            corr_y = corr_y * scale
            corr_z = corr_z * scale

        # Apply linear position correction only (no angular correction for stability)
        state[env, body_a_off + IDX_PX] = pa_x - inv_ma * corr_x
        state[env, body_a_off + IDX_PY] = pa_y - inv_ma * corr_y
        state[env, body_a_off + IDX_PZ] = pa_z - inv_ma * corr_z

        state[env, body_b_off + IDX_PX] = pb_x + inv_mb * corr_x
        state[env, body_b_off + IDX_PY] = pb_y + inv_mb * corr_y
        state[env, body_b_off + IDX_PZ] = pb_z + inv_mb * corr_z

        # Handle joint angle limit position correction
        var flags = Int(state[env, joint_off + JOINT3D_FLAGS])
        if flags & JOINT3D_FLAG_LIMIT_ENABLED:
            Self._apply_limits_position_gpu[
                BATCH, NUM_BODIES, MAX_JOINTS, STATE_SIZE, BODIES_OFFSET, JOINTS_OFFSET
            ](state, env, joint_idx, baumgarte)

    @always_inline
    @staticmethod
    fn _apply_limits_position_gpu[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        joint_idx: Int,
        baumgarte: Scalar[dtype],
    ):
        """Apply joint angle limits (GPU-compatible position correction).

        When the joint angle is beyond its limits, rotates bodies to bring
        it back within the allowed range.
        """
        var joint_off = JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE_3D

        # Get joint limits
        var lower_limit = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_LOWER_LIMIT])
        var upper_limit = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_UPPER_LIMIT])

        # Get current joint angle
        var current_angle = Hinge3D.get_joint_angle_gpu[
            BATCH, STATE_SIZE, BODIES_OFFSET, JOINTS_OFFSET
        ](state, env, joint_idx)

        # Calculate angle correction needed
        var angle_error = Scalar[dtype](0.0)
        var zero = Scalar[dtype](0.0)

        if current_angle < lower_limit:
            angle_error = lower_limit - current_angle  # Positive to rotate back up
        elif current_angle > upper_limit:
            angle_error = upper_limit - current_angle  # Negative to rotate back down

        if angle_error == zero:
            return

        # Apply correction with Baumgarte stabilization (soft constraint)
        var correction = angle_error * baumgarte

        var body_a = Int(state[env, joint_off + JOINT3D_BODY_A])
        var body_b = Int(state[env, joint_off + JOINT3D_BODY_B])

        var body_a_off = BODIES_OFFSET + body_a * BODY_STATE_SIZE_3D
        var body_b_off = BODIES_OFFSET + body_b * BODY_STATE_SIZE_3D

        # Get current orientations
        var qa_w = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QW])
        var qa_x = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QX])
        var qa_y = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QY])
        var qa_z = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QZ])

        var qb_w = rebind[Scalar[dtype]](state[env, body_b_off + IDX_QW])
        var qb_x = rebind[Scalar[dtype]](state[env, body_b_off + IDX_QX])
        var qb_y = rebind[Scalar[dtype]](state[env, body_b_off + IDX_QY])
        var qb_z = rebind[Scalar[dtype]](state[env, body_b_off + IDX_QZ])

        # Get joint axis in body A's frame
        var axis_local_x = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_X])
        var axis_local_y = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_Y])
        var axis_local_z = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_Z])

        # Transform axis to world frame using body A's orientation
        var two = Scalar[dtype](2.0)
        var cx = qa_y * axis_local_z - qa_z * axis_local_y
        var cy = qa_z * axis_local_x - qa_x * axis_local_z
        var cz = qa_x * axis_local_y - qa_y * axis_local_x
        var ccx = qa_y * cz - qa_z * cy
        var ccy = qa_z * cx - qa_x * cz
        var ccz = qa_x * cy - qa_y * cx
        var axis_world_x = axis_local_x + two * qa_w * cx + two * ccx
        var axis_world_y = axis_local_y + two * qa_w * cy + two * ccy
        var axis_world_z = axis_local_z + two * qa_w * cz + two * ccz

        # Get inverse inertia (simplified diagonal average)
        var eps = Scalar[dtype](1e-10)
        var one = Scalar[dtype](1.0)
        var three = Scalar[dtype](3.0)
        var inv_ia_avg = (
            one / (rebind[Scalar[dtype]](state[env, body_a_off + IDX_IXX]) + eps)
            + one / (rebind[Scalar[dtype]](state[env, body_a_off + IDX_IYY]) + eps)
            + one / (rebind[Scalar[dtype]](state[env, body_a_off + IDX_IZZ]) + eps)
        ) / three

        var inv_ib_avg = (
            one / (rebind[Scalar[dtype]](state[env, body_b_off + IDX_IXX]) + eps)
            + one / (rebind[Scalar[dtype]](state[env, body_b_off + IDX_IYY]) + eps)
            + one / (rebind[Scalar[dtype]](state[env, body_b_off + IDX_IZZ]) + eps)
        ) / three

        # Effective inertia for angular constraint
        var eff_inertia = inv_ia_avg + inv_ib_avg
        if eff_inertia < eps:
            return

        # Compute rotation correction for each body (proportional to inverse inertia)
        var half = Scalar[dtype](0.5)
        var half_correction_a = half * correction * (inv_ia_avg / eff_inertia)
        var half_correction_b = half * correction * (inv_ib_avg / eff_inertia)

        # Rotation increment quaternion: q_delta = (cos(theta/2), sin(theta/2)*axis)
        # For small angles: cos(theta/2) ≈ 1, sin(theta/2) ≈ theta/2
        # Body A rotates negative (to decrease relative angle)
        var sin_half_a = -half_correction_a  # Negative rotation for body A
        var sin_half_b = half_correction_b   # Positive rotation for body B

        # Create rotation quaternions (small angle approximation, w ≈ 1)
        var dqa_w = one
        var dqa_x = sin_half_a * axis_world_x
        var dqa_y = sin_half_a * axis_world_y
        var dqa_z = sin_half_a * axis_world_z

        var dqb_w = one
        var dqb_x = sin_half_b * axis_world_x
        var dqb_y = sin_half_b * axis_world_y
        var dqb_z = sin_half_b * axis_world_z

        # Quaternion multiply: new_q = dq * q
        # dq * q = (dw*w - dx*x - dy*y - dz*z,
        #           dw*x + dx*w + dy*z - dz*y,
        #           dw*y - dx*z + dy*w + dz*x,
        #           dw*z + dx*y - dy*x + dz*w)

        # Apply to body A
        var new_qa_w = dqa_w * qa_w - dqa_x * qa_x - dqa_y * qa_y - dqa_z * qa_z
        var new_qa_x = dqa_w * qa_x + dqa_x * qa_w + dqa_y * qa_z - dqa_z * qa_y
        var new_qa_y = dqa_w * qa_y - dqa_x * qa_z + dqa_y * qa_w + dqa_z * qa_x
        var new_qa_z = dqa_w * qa_z + dqa_x * qa_y - dqa_y * qa_x + dqa_z * qa_w

        # Apply to body B
        var new_qb_w = dqb_w * qb_w - dqb_x * qb_x - dqb_y * qb_y - dqb_z * qb_z
        var new_qb_x = dqb_w * qb_x + dqb_x * qb_w + dqb_y * qb_z - dqb_z * qb_y
        var new_qb_y = dqb_w * qb_y - dqb_x * qb_z + dqb_y * qb_w + dqb_z * qb_x
        var new_qb_z = dqb_w * qb_z + dqb_x * qb_y - dqb_y * qb_x + dqb_z * qb_w

        # Normalize quaternions
        var norm_a = sqrt(new_qa_w * new_qa_w + new_qa_x * new_qa_x + new_qa_y * new_qa_y + new_qa_z * new_qa_z)
        var norm_b = sqrt(new_qb_w * new_qb_w + new_qb_x * new_qb_x + new_qb_y * new_qb_y + new_qb_z * new_qb_z)

        if norm_a > eps:
            new_qa_w = new_qa_w / norm_a
            new_qa_x = new_qa_x / norm_a
            new_qa_y = new_qa_y / norm_a
            new_qa_z = new_qa_z / norm_a

        if norm_b > eps:
            new_qb_w = new_qb_w / norm_b
            new_qb_x = new_qb_x / norm_b
            new_qb_y = new_qb_y / norm_b
            new_qb_z = new_qb_z / norm_b

        # Write back corrected orientations
        state[env, body_a_off + IDX_QW] = new_qa_w
        state[env, body_a_off + IDX_QX] = new_qa_x
        state[env, body_a_off + IDX_QY] = new_qa_y
        state[env, body_a_off + IDX_QZ] = new_qa_z

        state[env, body_b_off + IDX_QW] = new_qb_w
        state[env, body_b_off + IDX_QX] = new_qb_x
        state[env, body_b_off + IDX_QY] = new_qb_y
        state[env, body_b_off + IDX_QZ] = new_qb_z

    @always_inline
    @staticmethod
    fn enforce_limits_all_joints_single_env[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        joint_count: Int,
    ):
        """Hard-enforce joint limits on all joints (GPU-compatible).

        This is called after integration to ensure joints never exceed their limits.
        If a joint is past its limit, it is immediately clamped and the velocity
        component along the axis is zeroed.
        """
        for j in range(joint_count):
            Hinge3D._enforce_limit_single_joint[
                BATCH, NUM_BODIES, MAX_JOINTS, STATE_SIZE, BODIES_OFFSET, JOINTS_OFFSET
            ](state, env, j)

    @always_inline
    @staticmethod
    fn _enforce_limit_single_joint[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        joint_idx: Int,
    ):
        """Hard-enforce limits on a single joint."""
        var joint_off = JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE_3D

        var joint_type = Int(state[env, joint_off + JOINT3D_TYPE])
        if joint_type != JOINT_HINGE:
            return

        var flags = Int(state[env, joint_off + JOINT3D_FLAGS])
        if not (flags & JOINT3D_FLAG_LIMIT_ENABLED):
            return

        # Get joint limits
        var lower_limit = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_LOWER_LIMIT])
        var upper_limit = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_UPPER_LIMIT])

        # Get current joint angle
        var current_angle = Hinge3D.get_joint_angle_gpu[
            BATCH, STATE_SIZE, BODIES_OFFSET, JOINTS_OFFSET
        ](state, env, joint_idx)

        # Check if within limits
        var target_angle = current_angle
        var clamped = False

        if current_angle < lower_limit:
            target_angle = lower_limit
            clamped = True
        elif current_angle > upper_limit:
            target_angle = upper_limit
            clamped = True

        if not clamped:
            return

        # Need to rotate body B relative to body A to achieve target_angle
        var correction = target_angle - current_angle

        var body_a = Int(state[env, joint_off + JOINT3D_BODY_A])
        var body_b = Int(state[env, joint_off + JOINT3D_BODY_B])

        var body_a_off = BODIES_OFFSET + body_a * BODY_STATE_SIZE_3D
        var body_b_off = BODIES_OFFSET + body_b * BODY_STATE_SIZE_3D

        # Get body B's orientation
        var qb_w = rebind[Scalar[dtype]](state[env, body_b_off + IDX_QW])
        var qb_x = rebind[Scalar[dtype]](state[env, body_b_off + IDX_QX])
        var qb_y = rebind[Scalar[dtype]](state[env, body_b_off + IDX_QY])
        var qb_z = rebind[Scalar[dtype]](state[env, body_b_off + IDX_QZ])

        # Get body A's orientation for axis transformation
        var qa_w = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QW])
        var qa_x = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QX])
        var qa_y = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QY])
        var qa_z = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QZ])

        # Get joint axis in body A's frame
        var axis_local_x = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_X])
        var axis_local_y = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_Y])
        var axis_local_z = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_Z])

        # Transform axis to world frame using body A's orientation
        var two = Scalar[dtype](2.0)
        var cx = qa_y * axis_local_z - qa_z * axis_local_y
        var cy = qa_z * axis_local_x - qa_x * axis_local_z
        var cz = qa_x * axis_local_y - qa_y * axis_local_x
        var ccx = qa_y * cz - qa_z * cy
        var ccy = qa_z * cx - qa_x * cz
        var ccz = qa_x * cy - qa_y * cx
        var axis_world_x = axis_local_x + two * qa_w * cx + two * ccx
        var axis_world_y = axis_local_y + two * qa_w * cy + two * ccy
        var axis_world_z = axis_local_z + two * qa_w * cz + two * ccz

        # Create rotation quaternion to correct body B
        # For small angles: q = (1, axis * theta/2) normalized
        # For larger angles, use proper formula
        var half_angle = correction * Scalar[dtype](0.5)
        var sin_half = sin(half_angle)
        var cos_half = cos(half_angle)

        var dq_w = cos_half
        var dq_x = sin_half * axis_world_x
        var dq_y = sin_half * axis_world_y
        var dq_z = sin_half * axis_world_z

        # Apply rotation: new_qb = dq * qb
        var new_qb_w = dq_w * qb_w - dq_x * qb_x - dq_y * qb_y - dq_z * qb_z
        var new_qb_x = dq_w * qb_x + dq_x * qb_w + dq_y * qb_z - dq_z * qb_y
        var new_qb_y = dq_w * qb_y - dq_x * qb_z + dq_y * qb_w + dq_z * qb_x
        var new_qb_z = dq_w * qb_z + dq_x * qb_y - dq_y * qb_x + dq_z * qb_w

        # Normalize
        var norm = sqrt(new_qb_w * new_qb_w + new_qb_x * new_qb_x + new_qb_y * new_qb_y + new_qb_z * new_qb_z)
        var eps = Scalar[dtype](1e-10)
        if norm > eps:
            new_qb_w = new_qb_w / norm
            new_qb_x = new_qb_x / norm
            new_qb_y = new_qb_y / norm
            new_qb_z = new_qb_z / norm

        # Write back corrected orientation
        state[env, body_b_off + IDX_QW] = new_qb_w
        state[env, body_b_off + IDX_QX] = new_qb_x
        state[env, body_b_off + IDX_QY] = new_qb_y
        state[env, body_b_off + IDX_QZ] = new_qb_z

        # Also zero out velocity component along the axis to prevent bouncing
        var wb_x = rebind[Scalar[dtype]](state[env, body_b_off + IDX_WX])
        var wb_y = rebind[Scalar[dtype]](state[env, body_b_off + IDX_WY])
        var wb_z = rebind[Scalar[dtype]](state[env, body_b_off + IDX_WZ])

        var wa_x = rebind[Scalar[dtype]](state[env, body_a_off + IDX_WX])
        var wa_y = rebind[Scalar[dtype]](state[env, body_a_off + IDX_WY])
        var wa_z = rebind[Scalar[dtype]](state[env, body_a_off + IDX_WZ])

        # Relative angular velocity
        var rel_wx = wb_x - wa_x
        var rel_wy = wb_y - wa_y
        var rel_wz = wb_z - wa_z

        # Project onto axis
        var rel_along_axis = rel_wx * axis_world_x + rel_wy * axis_world_y + rel_wz * axis_world_z

        # Only remove component that pushes past limit
        var zero = Scalar[dtype](0.0)
        if (current_angle < lower_limit and rel_along_axis < zero) or (current_angle > upper_limit and rel_along_axis > zero):
            # Remove the velocity component along the axis from body B
            state[env, body_b_off + IDX_WX] = wb_x - rel_along_axis * axis_world_x
            state[env, body_b_off + IDX_WY] = wb_y - rel_along_axis * axis_world_y
            state[env, body_b_off + IDX_WZ] = wb_z - rel_along_axis * axis_world_z

    # =========================================================================
    # GPU Single-Environment Helpers (Scalar-only)
    # =========================================================================

    @always_inline
    @staticmethod
    fn solve_velocity_all_joints_single_env[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        joint_count: Int,
        dt: Scalar[dtype],
    ):
        """Solve velocity constraints for all joints (GPU-compatible)."""
        for j in range(joint_count):
            Hinge3D.solve_velocity_gpu[
                BATCH, NUM_BODIES, MAX_JOINTS, STATE_SIZE, BODIES_OFFSET, JOINTS_OFFSET
            ](state, env, j, dt)

    @always_inline
    @staticmethod
    fn solve_position_all_joints_single_env[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        joint_count: Int,
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ):
        """Solve position constraints for all joints (GPU-compatible)."""
        for j in range(joint_count):
            Hinge3D.solve_position_gpu[
                BATCH, NUM_BODIES, MAX_JOINTS, STATE_SIZE, BODIES_OFFSET, JOINTS_OFFSET
            ](state, env, j, baumgarte, slop)

    @always_inline
    @staticmethod
    fn apply_direct_torques_single_env[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
        ACTION_DIM: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        actions: LayoutTensor[
            dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
        ],
        max_torque: Scalar[dtype],
    ):
        """Apply direct torques from action buffer (GPU-compatible)."""

        @parameter
        for j in range(ACTION_DIM):
            var action = rebind[Scalar[dtype]](actions[env, j])
            var one = Scalar[dtype](1.0)
            if action > one:
                action = one
            if action < -one:
                action = -one
            var torque = action * max_torque

            Hinge3D.apply_direct_torque_gpu[
                BATCH, STATE_SIZE, BODIES_OFFSET, JOINTS_OFFSET
            ](state, env, j, torque)

    # =========================================================================
    # CPU-Only Functions (Still use Vec3/Quat for convenience)
    # =========================================================================

    @staticmethod
    fn get_joint_angle[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        joint_idx: Int,
    ) -> Scalar[dtype]:
        """Compute current joint angle (CPU version)."""
        var joint_off = JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE_3D

        var body_a = Int(state[env, joint_off + JOINT3D_BODY_A])
        var body_b = Int(state[env, joint_off + JOINT3D_BODY_B])

        var body_a_off = BODIES_OFFSET + body_a * BODY_STATE_SIZE_3D
        var body_b_off = BODIES_OFFSET + body_b * BODY_STATE_SIZE_3D

        var qa = Quat(
            rebind[Scalar[dtype]](state[env, body_a_off + IDX_QW]),
            rebind[Scalar[dtype]](state[env, body_a_off + IDX_QX]),
            rebind[Scalar[dtype]](state[env, body_a_off + IDX_QY]),
            rebind[Scalar[dtype]](state[env, body_a_off + IDX_QZ]),
        )

        var qb = Quat(
            rebind[Scalar[dtype]](state[env, body_b_off + IDX_QW]),
            rebind[Scalar[dtype]](state[env, body_b_off + IDX_QX]),
            rebind[Scalar[dtype]](state[env, body_b_off + IDX_QY]),
            rebind[Scalar[dtype]](state[env, body_b_off + IDX_QZ]),
        )

        var axis = Vec3(
            rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_X]),
            rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_Y]),
            rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_Z]),
        )

        var qrel = qa.conjugate() * qb
        var qrel_xyz = Vec3(qrel.x, qrel.y, qrel.z)
        var angle = 2.0 * atan2(qrel_xyz.dot(axis), qrel.w)

        return Scalar[dtype](angle)

    @staticmethod
    fn get_joint_velocity[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        joint_idx: Int,
    ) -> Scalar[dtype]:
        """Compute current joint velocity (CPU version)."""
        var joint_off = JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE_3D

        var body_a = Int(state[env, joint_off + JOINT3D_BODY_A])
        var body_b = Int(state[env, joint_off + JOINT3D_BODY_B])

        var body_a_off = BODIES_OFFSET + body_a * BODY_STATE_SIZE_3D
        var body_b_off = BODIES_OFFSET + body_b * BODY_STATE_SIZE_3D

        var wa = Vec3(
            rebind[Scalar[dtype]](state[env, body_a_off + IDX_WX]),
            rebind[Scalar[dtype]](state[env, body_a_off + IDX_WY]),
            rebind[Scalar[dtype]](state[env, body_a_off + IDX_WZ]),
        )

        var wb = Vec3(
            rebind[Scalar[dtype]](state[env, body_b_off + IDX_WX]),
            rebind[Scalar[dtype]](state[env, body_b_off + IDX_WY]),
            rebind[Scalar[dtype]](state[env, body_b_off + IDX_WZ]),
        )

        var qa = Quat(
            rebind[Scalar[dtype]](state[env, body_a_off + IDX_QW]),
            rebind[Scalar[dtype]](state[env, body_a_off + IDX_QX]),
            rebind[Scalar[dtype]](state[env, body_a_off + IDX_QY]),
            rebind[Scalar[dtype]](state[env, body_a_off + IDX_QZ]),
        )

        var axis_local = Vec3(
            rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_X]),
            rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_Y]),
            rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_AXIS_Z]),
        )
        var axis_world = qa.rotate_vec(axis_local)

        var rel_omega = wb - wa
        var joint_vel = rel_omega.dot(axis_world)

        return Scalar[dtype](joint_vel)

    @staticmethod
    fn apply_direct_torque[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        joint_idx: Int,
        torque: Scalar[dtype],
    ):
        """Apply direct torque to joint (CPU version)."""
        Hinge3D.apply_direct_torque_gpu[
            BATCH, STATE_SIZE, BODIES_OFFSET, JOINTS_OFFSET
        ](state, env, joint_idx, torque)

    @staticmethod
    fn solve_velocity[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        joint_idx: Int,
        dt: Scalar[dtype],
    ):
        """Solve velocity constraints (CPU version)."""
        Hinge3D.solve_velocity_gpu[
            BATCH, NUM_BODIES, MAX_JOINTS, STATE_SIZE, BODIES_OFFSET, JOINTS_OFFSET
        ](state, env, joint_idx, dt)

    @staticmethod
    fn solve_position[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        joint_idx: Int,
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ):
        """Solve position constraints (CPU version)."""
        Hinge3D.solve_position_gpu[
            BATCH, NUM_BODIES, MAX_JOINTS, STATE_SIZE, BODIES_OFFSET, JOINTS_OFFSET
        ](state, env, joint_idx, baumgarte, slop)

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
    """

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
        """Solve velocity constraints for hinge joint (GPU-compatible)."""
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

        # Get inverse inertia (diagonal, local frame - simplified)
        var ixx_a = rebind[Scalar[dtype]](state[env, body_a_off + IDX_IXX])
        var iyy_a = rebind[Scalar[dtype]](state[env, body_a_off + IDX_IYY])
        var izz_a = rebind[Scalar[dtype]](state[env, body_a_off + IDX_IZZ])
        var ixx_b = rebind[Scalar[dtype]](state[env, body_b_off + IDX_IXX])
        var iyy_b = rebind[Scalar[dtype]](state[env, body_b_off + IDX_IYY])
        var izz_b = rebind[Scalar[dtype]](state[env, body_b_off + IDX_IZZ])

        var eps = Scalar[dtype](1e-10)
        var one = Scalar[dtype](1.0)
        var three = Scalar[dtype](3.0)
        var inv_ia_x = one / (ixx_a + eps)
        var inv_ia_y = one / (iyy_a + eps)
        var inv_ia_z = one / (izz_a + eps)
        var inv_ib_x = one / (ixx_b + eps)
        var inv_ib_y = one / (iyy_b + eps)
        var inv_ib_z = one / (izz_b + eps)

        # Get orientations
        var qa_w = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QW])
        var qa_x = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QX])
        var qa_y = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QY])
        var qa_z = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QZ])
        var qb_w = rebind[Scalar[dtype]](state[env, body_b_off + IDX_QW])
        var qb_x = rebind[Scalar[dtype]](state[env, body_b_off + IDX_QX])
        var qb_y = rebind[Scalar[dtype]](state[env, body_b_off + IDX_QY])
        var qb_z = rebind[Scalar[dtype]](state[env, body_b_off + IDX_QZ])

        # Get positions
        var pa_x = rebind[Scalar[dtype]](state[env, body_a_off + IDX_PX])
        var pa_y = rebind[Scalar[dtype]](state[env, body_a_off + IDX_PY])
        var pa_z = rebind[Scalar[dtype]](state[env, body_a_off + IDX_PZ])
        var pb_x = rebind[Scalar[dtype]](state[env, body_b_off + IDX_PX])
        var pb_y = rebind[Scalar[dtype]](state[env, body_b_off + IDX_PY])
        var pb_z = rebind[Scalar[dtype]](state[env, body_b_off + IDX_PZ])

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

        # Compute effective mass for point-to-point constraint (simplified)
        var eff_mass_linear = inv_ma + inv_mb

        # Add angular contribution (simplified diagonal approximation)
        var ra_cross_sq = ra_x * ra_x + ra_y * ra_y + ra_z * ra_z
        var rb_cross_sq = rb_x * rb_x + rb_y * rb_y + rb_z * rb_z
        var avg_inv_ia = (inv_ia_x + inv_ia_y + inv_ia_z) / three
        var avg_inv_ib = (inv_ib_x + inv_ib_y + inv_ib_z) / three

        eff_mass_linear = eff_mass_linear + ra_cross_sq * avg_inv_ia + rb_cross_sq * avg_inv_ib

        if eff_mass_linear < eps:
            eff_mass_linear = eps

        # Compute and apply impulse (simplified scalar version)
        var inv_eff_mass = one / eff_mass_linear
        var impulse_x = cdot_x * (-inv_eff_mass)
        var impulse_y = cdot_y * (-inv_eff_mass)
        var impulse_z = cdot_z * (-inv_eff_mass)

        # Apply linear impulse
        var new_va_x = va_x - impulse_x * inv_ma
        var new_va_y = va_y - impulse_y * inv_ma
        var new_va_z = va_z - impulse_z * inv_ma
        var new_vb_x = vb_x + impulse_x * inv_mb
        var new_vb_y = vb_y + impulse_y * inv_mb
        var new_vb_z = vb_z + impulse_z * inv_mb

        # Apply angular impulse (simplified): ra × impulse
        var ra_cross_impulse_x = ra_y * impulse_z - ra_z * impulse_y
        var ra_cross_impulse_y = ra_z * impulse_x - ra_x * impulse_z
        var ra_cross_impulse_z = ra_x * impulse_y - ra_y * impulse_x
        var rb_cross_impulse_x = rb_y * impulse_z - rb_z * impulse_y
        var rb_cross_impulse_y = rb_z * impulse_x - rb_x * impulse_z
        var rb_cross_impulse_z = rb_x * impulse_y - rb_y * impulse_x

        var new_wa_x = wa_x - ra_cross_impulse_x * avg_inv_ia
        var new_wa_y = wa_y - ra_cross_impulse_y * avg_inv_ia
        var new_wa_z = wa_z - ra_cross_impulse_z * avg_inv_ia
        var new_wb_x = wb_x + rb_cross_impulse_x * avg_inv_ib
        var new_wb_y = wb_y + rb_cross_impulse_y * avg_inv_ib
        var new_wb_z = wb_z + rb_cross_impulse_z * avg_inv_ib

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

        # Handle motor constraint (now using GPU-compatible atan2_gpu)
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
        """Solve position constraints to correct drift (GPU-compatible)."""
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
        var error_x = anchor_b_world_x - anchor_a_world_x
        var error_y = anchor_b_world_y - anchor_a_world_y
        var error_z = anchor_b_world_z - anchor_a_world_z
        var error_sq = error_x * error_x + error_y * error_y + error_z * error_z
        var error_mag = sqrt(error_sq)

        if error_mag < slop:
            return

        # Get mass properties
        var inv_ma = rebind[Scalar[dtype]](state[env, body_a_off + IDX_INV_MASS])
        var inv_mb = rebind[Scalar[dtype]](state[env, body_b_off + IDX_INV_MASS])

        # Simplified mass for position correction
        var total_inv_mass = inv_ma + inv_mb
        var eps = Scalar[dtype](1e-10)
        if total_inv_mass < eps:
            return

        # Position correction
        var correction_x = error_x * baumgarte
        var correction_y = error_y * baumgarte
        var correction_z = error_z * baumgarte
        var delta_a_x = correction_x * (inv_ma / total_inv_mass)
        var delta_a_y = correction_y * (inv_ma / total_inv_mass)
        var delta_a_z = correction_z * (inv_ma / total_inv_mass)
        var delta_b_x = correction_x * (inv_mb / total_inv_mass)
        var delta_b_y = correction_y * (inv_mb / total_inv_mass)
        var delta_b_z = correction_z * (inv_mb / total_inv_mass)

        # Apply position corrections
        state[env, body_a_off + IDX_PX] = pa_x + delta_a_x
        state[env, body_a_off + IDX_PY] = pa_y + delta_a_y
        state[env, body_a_off + IDX_PZ] = pa_z + delta_a_z

        state[env, body_b_off + IDX_PX] = pb_x - delta_b_x
        state[env, body_b_off + IDX_PY] = pb_y - delta_b_y
        state[env, body_b_off + IDX_PZ] = pb_z - delta_b_z

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

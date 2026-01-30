"""Passive Joint Dynamics (Spring-Damper Forces).

This module implements MuJoCo-style passive joint forces:
- Spring force: qfrc_spring = -stiffness * (qpos - qpos_ref)
- Damper force: qfrc_damper = -damping * qvel

These forces resist joint motion and velocity, providing natural stabilization
that allows the use of realistic torque magnitudes without joints separating.

GPU support: All functions use only scalar operations, no Vec3/Quat struct
instantiation, following the physics2d/3d pattern.
"""

from math import sqrt
from layout import LayoutTensor, Layout

from .constants import (
    dtype,
    BODY_STATE_SIZE_3D,
    JOINT_DATA_SIZE_3D,
    IDX_QW,
    IDX_QX,
    IDX_QY,
    IDX_QZ,
    IDX_WX,
    IDX_WY,
    IDX_WZ,
    IDX_TX,
    IDX_TY,
    IDX_TZ,
    JOINT3D_TYPE,
    JOINT3D_BODY_A,
    JOINT3D_BODY_B,
    JOINT3D_AXIS_X,
    JOINT3D_AXIS_Y,
    JOINT3D_AXIS_Z,
    JOINT3D_STIFFNESS,
    JOINT3D_DAMPING,
    JOINT3D_ARMATURE,
    JOINT3D_REFERENCE_POS,
    JOINT_HINGE,
)

from .math_gpu import atan2_gpu


struct PassiveJointForces:
    """Passive spring-damper force computation for joints.

    Implements MuJoCo-style passive joint dynamics:
    1. Spring force: resists deviation from reference position
    2. Damper force: resists joint velocity

    The combined effect is:
    tau_passive = -stiffness * (angle - ref_angle) - damping * angular_velocity

    This torque is applied equal and opposite to the two bodies connected
    by the joint, along the joint axis in world frame.
    """

    # =========================================================================
    # Passive Torque Computation (GPU-compatible, scalar-only)
    # =========================================================================

    @always_inline
    @staticmethod
    fn compute_passive_torque_gpu[
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
        """Compute passive spring-damper torque for a joint (GPU-compatible).

        Returns the scalar torque magnitude along the joint axis.
        Positive torque rotates body_b relative to body_a in the positive direction.
        """
        var joint_off = JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE_3D

        # Skip if not a hinge joint
        var joint_type = Int(state[env, joint_off + JOINT3D_TYPE])
        if joint_type != JOINT_HINGE:
            return Scalar[dtype](0.0)

        # Get passive dynamics parameters
        var stiffness = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_STIFFNESS])
        var damping = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_DAMPING])
        var ref_pos = rebind[Scalar[dtype]](state[env, joint_off + JOINT3D_REFERENCE_POS])

        # Early exit if no passive forces configured
        var eps = Scalar[dtype](1e-10)
        if stiffness < eps and damping < eps:
            return Scalar[dtype](0.0)

        # Get current joint angle
        var current_angle = Self._get_joint_angle_gpu[
            BATCH, STATE_SIZE, BODIES_OFFSET, JOINTS_OFFSET
        ](state, env, joint_idx)

        # Get current joint velocity
        var current_vel = Self._get_joint_velocity_gpu[
            BATCH, STATE_SIZE, BODIES_OFFSET, JOINTS_OFFSET
        ](state, env, joint_idx)

        # Compute passive torque: tau = -k*(x - x_ref) - c*v
        var spring_torque = -stiffness * (current_angle - ref_pos)
        var damper_torque = -damping * current_vel
        var passive_torque = spring_torque + damper_torque

        return passive_torque

    @always_inline
    @staticmethod
    fn apply_passive_forces_gpu[
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
    ):
        """Apply passive spring-damper forces to joint bodies (GPU-compatible).

        Computes the passive torque and applies it equal and opposite to
        the two bodies connected by the joint.
        """
        var joint_off = JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE_3D

        # Skip if not a hinge joint
        var joint_type = Int(state[env, joint_off + JOINT3D_TYPE])
        if joint_type != JOINT_HINGE:
            return

        # Compute passive torque magnitude
        var passive_torque = Self.compute_passive_torque_gpu[
            BATCH, STATE_SIZE, BODIES_OFFSET, JOINTS_OFFSET
        ](state, env, joint_idx)

        # Skip if negligible
        var eps = Scalar[dtype](1e-10)
        if passive_torque > -eps and passive_torque < eps:
            return

        # Get body indices
        var body_a = Int(state[env, joint_off + JOINT3D_BODY_A])
        var body_b = Int(state[env, joint_off + JOINT3D_BODY_B])

        var body_a_off = BODIES_OFFSET + body_a * BODY_STATE_SIZE_3D
        var body_b_off = BODIES_OFFSET + body_b * BODY_STATE_SIZE_3D

        # Get world-space axis from body_a orientation
        var qa_w = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QW])
        var qa_x = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QX])
        var qa_y = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QY])
        var qa_z = rebind[Scalar[dtype]](state[env, body_a_off + IDX_QZ])

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

        # Compute torque vector: tau = axis * torque
        var tau_x = axis_world_x * passive_torque
        var tau_y = axis_world_y * passive_torque
        var tau_z = axis_world_z * passive_torque

        # Apply equal and opposite torques to the bodies
        # Body A gets negative torque (opposes relative rotation)
        state[env, body_a_off + IDX_TX] = state[env, body_a_off + IDX_TX] - tau_x
        state[env, body_a_off + IDX_TY] = state[env, body_a_off + IDX_TY] - tau_y
        state[env, body_a_off + IDX_TZ] = state[env, body_a_off + IDX_TZ] - tau_z

        # Body B gets positive torque
        state[env, body_b_off + IDX_TX] = state[env, body_b_off + IDX_TX] + tau_x
        state[env, body_b_off + IDX_TY] = state[env, body_b_off + IDX_TY] + tau_y
        state[env, body_b_off + IDX_TZ] = state[env, body_b_off + IDX_TZ] + tau_z

    # =========================================================================
    # Helper Functions (GPU-compatible, scalar-only)
    # =========================================================================

    @always_inline
    @staticmethod
    fn _get_joint_angle_gpu[
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
        """Get current joint angle from body orientations (GPU-compatible).

        Duplicated from Hinge3D to avoid circular dependencies.
        """
        var joint_off = JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE_3D

        var body_a = Int(state[env, joint_off + JOINT3D_BODY_A])
        var body_b = Int(state[env, joint_off + JOINT3D_BODY_B])

        var body_a_off = BODIES_OFFSET + body_a * BODY_STATE_SIZE_3D
        var body_b_off = BODIES_OFFSET + body_b * BODY_STATE_SIZE_3D

        # Get body orientations
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
        var qa_conj_w = qa_w
        var qa_conj_x = -qa_x
        var qa_conj_y = -qa_y
        var qa_conj_z = -qa_z

        # Quaternion multiplication
        var qrel_w = qa_conj_w * qb_w - qa_conj_x * qb_x - qa_conj_y * qb_y - qa_conj_z * qb_z
        var qrel_x = qa_conj_w * qb_x + qa_conj_x * qb_w + qa_conj_y * qb_z - qa_conj_z * qb_y
        var qrel_y = qa_conj_w * qb_y - qa_conj_x * qb_z + qa_conj_y * qb_w + qa_conj_z * qb_x
        var qrel_z = qa_conj_w * qb_z + qa_conj_x * qb_y - qa_conj_y * qb_x + qa_conj_z * qb_w

        # Extract rotation angle around the joint axis
        var dot_xyz_axis = qrel_x * axis_x + qrel_y * axis_y + qrel_z * axis_z
        var angle = Scalar[dtype](2.0) * atan2_gpu[dtype](dot_xyz_axis, qrel_w)

        return angle

    @always_inline
    @staticmethod
    fn _get_joint_velocity_gpu[
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
        """Get current joint angular velocity (GPU-compatible).

        Duplicated from Hinge3D to avoid circular dependencies.
        """
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

        # Transform axis to world frame
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

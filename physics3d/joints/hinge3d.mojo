"""3D Hinge (Revolute) Joint.

A hinge joint constrains two bodies to rotate around a single axis,
like a door hinge or elbow joint. It has 1 degree of freedom (the rotation angle).

This is the most common joint type in MuJoCo locomotion environments.
"""

from math import cos, sin, sqrt, acos, atan2
from layout import LayoutTensor, Layout

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
    # Joint Initialization
    # =========================================================================

    @staticmethod
    fn init_joint[
        BATCH: Int,
        STATE_SIZE: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        joint_idx: Int,
        body_a: Int,
        body_b: Int,
        anchor_a: Vec3,  # Local anchor on body A
        anchor_b: Vec3,  # Local anchor on body B
        axis: Vec3,  # Joint axis (in body A's local frame)
        lower_limit: Float64 = -3.14159,
        upper_limit: Float64 = 3.14159,
        motor_kp: Float64 = 100.0,
        motor_kd: Float64 = 10.0,
        max_force: Float64 = 100.0,
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
    # Joint State Extraction
    # =========================================================================

    @staticmethod
    fn get_joint_angle[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        joint_idx: Int,
    ) -> Scalar[dtype]:
        """Compute current joint angle from body orientations."""
        var joint_off = JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE_3D

        var body_a = Int(state[env, joint_off + JOINT3D_BODY_A])
        var body_b = Int(state[env, joint_off + JOINT3D_BODY_B])

        # Get body orientations as quaternions
        var body_a_off = BODIES_OFFSET + body_a * BODY_STATE_SIZE_3D
        var body_b_off = BODIES_OFFSET + body_b * BODY_STATE_SIZE_3D

        var qa = Quat(
            Float64(state[env, body_a_off + IDX_QW]),
            Float64(state[env, body_a_off + IDX_QX]),
            Float64(state[env, body_a_off + IDX_QY]),
            Float64(state[env, body_a_off + IDX_QZ]),
        )

        var qb = Quat(
            Float64(state[env, body_b_off + IDX_QW]),
            Float64(state[env, body_b_off + IDX_QX]),
            Float64(state[env, body_b_off + IDX_QY]),
            Float64(state[env, body_b_off + IDX_QZ]),
        )

        # Joint axis in body A's frame
        var axis = Vec3(
            Float64(state[env, joint_off + JOINT3D_AXIS_X]),
            Float64(state[env, joint_off + JOINT3D_AXIS_Y]),
            Float64(state[env, joint_off + JOINT3D_AXIS_Z]),
        )

        # Relative quaternion: qrel = qa^-1 * qb
        var qrel = qa.conjugate() * qb

        # Extract rotation angle around the joint axis
        # Project qrel onto axis using: angle = 2 * atan2(dot(qrel.xyz, axis), qrel.w)
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
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        joint_idx: Int,
    ) -> Scalar[dtype]:
        """Compute current joint angular velocity."""
        var joint_off = JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE_3D

        var body_a = Int(state[env, joint_off + JOINT3D_BODY_A])
        var body_b = Int(state[env, joint_off + JOINT3D_BODY_B])

        # Get body angular velocities
        var body_a_off = BODIES_OFFSET + body_a * BODY_STATE_SIZE_3D
        var body_b_off = BODIES_OFFSET + body_b * BODY_STATE_SIZE_3D

        var wa = Vec3(
            Float64(state[env, body_a_off + IDX_WX]),
            Float64(state[env, body_a_off + IDX_WY]),
            Float64(state[env, body_a_off + IDX_WZ]),
        )

        var wb = Vec3(
            Float64(state[env, body_b_off + IDX_WX]),
            Float64(state[env, body_b_off + IDX_WY]),
            Float64(state[env, body_b_off + IDX_WZ]),
        )

        # Get orientation of body A to transform axis to world frame
        var qa = Quat(
            Float64(state[env, body_a_off + IDX_QW]),
            Float64(state[env, body_a_off + IDX_QX]),
            Float64(state[env, body_a_off + IDX_QY]),
            Float64(state[env, body_a_off + IDX_QZ]),
        )

        # Joint axis in local frame, transform to world
        var axis_local = Vec3(
            Float64(state[env, joint_off + JOINT3D_AXIS_X]),
            Float64(state[env, joint_off + JOINT3D_AXIS_Y]),
            Float64(state[env, joint_off + JOINT3D_AXIS_Z]),
        )
        var axis_world = qa.rotate_vec(axis_local)

        # Relative angular velocity projected onto joint axis
        var rel_omega = wb - wa
        var joint_vel = rel_omega.dot(axis_world)

        return Scalar[dtype](joint_vel)

    # =========================================================================
    # Velocity Constraint Solving
    # =========================================================================

    @staticmethod
    fn solve_velocity[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        joint_idx: Int,
        dt: Scalar[dtype],
    ):
        """Solve velocity constraints for hinge joint.

        Constrains:
        1. Anchor point velocities (3 linear)
        2. Angular velocity perpendicular to axis (2 angular)
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
        var inv_ma = state[env, body_a_off + IDX_INV_MASS]
        var inv_mb = state[env, body_b_off + IDX_INV_MASS]

        # Get inverse inertia (diagonal, local frame - simplified)
        var inv_ia = Vec3(
            Float64(1.0 / (state[env, body_a_off + IDX_IXX] + 1e-10)),
            Float64(1.0 / (state[env, body_a_off + IDX_IYY] + 1e-10)),
            Float64(1.0 / (state[env, body_a_off + IDX_IZZ] + 1e-10)),
        )
        var inv_ib = Vec3(
            Float64(1.0 / (state[env, body_b_off + IDX_IXX] + 1e-10)),
            Float64(1.0 / (state[env, body_b_off + IDX_IYY] + 1e-10)),
            Float64(1.0 / (state[env, body_b_off + IDX_IZZ] + 1e-10)),
        )

        # Get orientations
        var qa = Quat(
            Float64(state[env, body_a_off + IDX_QW]),
            Float64(state[env, body_a_off + IDX_QX]),
            Float64(state[env, body_a_off + IDX_QY]),
            Float64(state[env, body_a_off + IDX_QZ]),
        )
        var qb = Quat(
            Float64(state[env, body_b_off + IDX_QW]),
            Float64(state[env, body_b_off + IDX_QX]),
            Float64(state[env, body_b_off + IDX_QY]),
            Float64(state[env, body_b_off + IDX_QZ]),
        )

        # Get positions
        var pa = Vec3(
            Float64(state[env, body_a_off + IDX_PX]),
            Float64(state[env, body_a_off + IDX_PY]),
            Float64(state[env, body_a_off + IDX_PZ]),
        )
        var pb = Vec3(
            Float64(state[env, body_b_off + IDX_PX]),
            Float64(state[env, body_b_off + IDX_PY]),
            Float64(state[env, body_b_off + IDX_PZ]),
        )

        # Get velocities
        var va = Vec3(
            Float64(state[env, body_a_off + IDX_VX]),
            Float64(state[env, body_a_off + IDX_VY]),
            Float64(state[env, body_a_off + IDX_VZ]),
        )
        var vb = Vec3(
            Float64(state[env, body_b_off + IDX_VX]),
            Float64(state[env, body_b_off + IDX_VY]),
            Float64(state[env, body_b_off + IDX_VZ]),
        )
        var wa = Vec3(
            Float64(state[env, body_a_off + IDX_WX]),
            Float64(state[env, body_a_off + IDX_WY]),
            Float64(state[env, body_a_off + IDX_WZ]),
        )
        var wb = Vec3(
            Float64(state[env, body_b_off + IDX_WX]),
            Float64(state[env, body_b_off + IDX_WY]),
            Float64(state[env, body_b_off + IDX_WZ]),
        )

        # Local anchors
        var anchor_a_local = Vec3(
            Float64(state[env, joint_off + JOINT3D_ANCHOR_AX]),
            Float64(state[env, joint_off + JOINT3D_ANCHOR_AY]),
            Float64(state[env, joint_off + JOINT3D_ANCHOR_AZ]),
        )
        var anchor_b_local = Vec3(
            Float64(state[env, joint_off + JOINT3D_ANCHOR_BX]),
            Float64(state[env, joint_off + JOINT3D_ANCHOR_BY]),
            Float64(state[env, joint_off + JOINT3D_ANCHOR_BZ]),
        )

        # Transform anchors to world frame
        var ra = qa.rotate_vec(anchor_a_local)
        var rb = qb.rotate_vec(anchor_b_local)

        # Velocity at anchor points: v_anchor = v + w × r
        var va_anchor = va + wa.cross(ra)
        var vb_anchor = vb + wb.cross(rb)

        # Relative velocity (should be zero for joint constraint)
        var cdot = vb_anchor - va_anchor

        # Compute effective mass for point-to-point constraint (simplified)
        # K = inv_ma + inv_mb + [ra×]^T * inv_Ia * [ra×] + [rb×]^T * inv_Ib * [rb×]
        # This is a 3x3 matrix, but we use a scalar approximation
        var eff_mass_linear = Float64(inv_ma + inv_mb)

        # Add angular contribution (simplified diagonal approximation)
        var ra_cross_sq = ra.length_squared()
        var rb_cross_sq = rb.length_squared()
        var avg_inv_ia = (inv_ia.x + inv_ia.y + inv_ia.z) / 3.0
        var avg_inv_ib = (inv_ib.x + inv_ib.y + inv_ib.z) / 3.0

        eff_mass_linear += ra_cross_sq * avg_inv_ia + rb_cross_sq * avg_inv_ib

        if eff_mass_linear < 1e-10:
            eff_mass_linear = 1e-10

        # Compute and apply impulse (simplified scalar version)
        var inv_eff_mass = 1.0 / eff_mass_linear
        var impulse = cdot * (-inv_eff_mass)

        # Apply linear impulse
        var new_va = va - impulse * Float64(inv_ma)
        var new_vb = vb + impulse * Float64(inv_mb)

        # Apply angular impulse (simplified)
        var angular_impulse_a = ra.cross(impulse) * avg_inv_ia
        var angular_impulse_b = rb.cross(impulse) * avg_inv_ib

        var new_wa = wa - angular_impulse_a
        var new_wb = wb + angular_impulse_b

        # Write back velocities
        state[env, body_a_off + IDX_VX] = Scalar[dtype](new_va.x)
        state[env, body_a_off + IDX_VY] = Scalar[dtype](new_va.y)
        state[env, body_a_off + IDX_VZ] = Scalar[dtype](new_va.z)
        state[env, body_a_off + IDX_WX] = Scalar[dtype](new_wa.x)
        state[env, body_a_off + IDX_WY] = Scalar[dtype](new_wa.y)
        state[env, body_a_off + IDX_WZ] = Scalar[dtype](new_wa.z)

        state[env, body_b_off + IDX_VX] = Scalar[dtype](new_vb.x)
        state[env, body_b_off + IDX_VY] = Scalar[dtype](new_vb.y)
        state[env, body_b_off + IDX_VZ] = Scalar[dtype](new_vb.z)
        state[env, body_b_off + IDX_WX] = Scalar[dtype](new_wb.x)
        state[env, body_b_off + IDX_WY] = Scalar[dtype](new_wb.y)
        state[env, body_b_off + IDX_WZ] = Scalar[dtype](new_wb.z)

        # Handle motor constraint
        var flags = Int(state[env, joint_off + JOINT3D_FLAGS])
        if flags & JOINT3D_FLAG_MOTOR_ENABLED:
            Self._apply_motor[
                BATCH, NUM_BODIES, MAX_JOINTS, STATE_SIZE, BODIES_OFFSET, JOINTS_OFFSET
            ](state, env, joint_idx, dt)

    @staticmethod
    fn _apply_motor[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        joint_idx: Int,
        dt: Scalar[dtype],
    ):
        """Apply PD motor control to joint."""
        var joint_off = JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE_3D

        # Get motor parameters
        var target = state[env, joint_off + JOINT3D_MOTOR_TARGET]
        var kp = state[env, joint_off + JOINT3D_MOTOR_KP]
        var kd = state[env, joint_off + JOINT3D_MOTOR_KD]
        var max_force = state[env, joint_off + JOINT3D_MAX_FORCE]

        # Get current joint state
        var current_angle = Self.get_joint_angle[
            BATCH, STATE_SIZE, BODIES_OFFSET, JOINTS_OFFSET
        ](state, env, joint_idx)
        var current_vel = Self.get_joint_velocity[
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
        var qa = Quat(
            Float64(state[env, body_a_off + IDX_QW]),
            Float64(state[env, body_a_off + IDX_QX]),
            Float64(state[env, body_a_off + IDX_QY]),
            Float64(state[env, body_a_off + IDX_QZ]),
        )

        var axis_local = Vec3(
            Float64(state[env, joint_off + JOINT3D_AXIS_X]),
            Float64(state[env, joint_off + JOINT3D_AXIS_Y]),
            Float64(state[env, joint_off + JOINT3D_AXIS_Z]),
        )
        var axis_world = qa.rotate_vec(axis_local)

        # Compute angular impulse
        var torque_vec = axis_world * Float64(torque)
        var impulse_vec = torque_vec * Float64(dt)

        # Get inverse inertia (simplified diagonal average)
        var inv_ia_avg = (
            Float64(1.0 / (state[env, body_a_off + IDX_IXX] + 1e-10))
            + Float64(1.0 / (state[env, body_a_off + IDX_IYY] + 1e-10))
            + Float64(1.0 / (state[env, body_a_off + IDX_IZZ] + 1e-10))
        ) / 3.0

        var inv_ib_avg = (
            Float64(1.0 / (state[env, body_b_off + IDX_IXX] + 1e-10))
            + Float64(1.0 / (state[env, body_b_off + IDX_IYY] + 1e-10))
            + Float64(1.0 / (state[env, body_b_off + IDX_IZZ] + 1e-10))
        ) / 3.0

        # Apply equal and opposite angular impulses
        var delta_wa = impulse_vec * (-inv_ia_avg)
        var delta_wb = impulse_vec * inv_ib_avg

        state[env, body_a_off + IDX_WX] = state[env, body_a_off + IDX_WX] + Scalar[dtype](
            delta_wa.x
        )
        state[env, body_a_off + IDX_WY] = state[env, body_a_off + IDX_WY] + Scalar[dtype](
            delta_wa.y
        )
        state[env, body_a_off + IDX_WZ] = state[env, body_a_off + IDX_WZ] + Scalar[dtype](
            delta_wa.z
        )

        state[env, body_b_off + IDX_WX] = state[env, body_b_off + IDX_WX] + Scalar[dtype](
            delta_wb.x
        )
        state[env, body_b_off + IDX_WY] = state[env, body_b_off + IDX_WY] + Scalar[dtype](
            delta_wb.y
        )
        state[env, body_b_off + IDX_WZ] = state[env, body_b_off + IDX_WZ] + Scalar[dtype](
            delta_wb.z
        )

    # =========================================================================
    # Position Constraint Solving (Baumgarte Stabilization)
    # =========================================================================

    @staticmethod
    fn solve_position[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        joint_idx: Int,
        baumgarte: Scalar[dtype],
        slop: Scalar[dtype],
    ):
        """Solve position constraints to correct drift."""
        var joint_off = JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE_3D

        var joint_type = Int(state[env, joint_off + JOINT3D_TYPE])
        if joint_type != JOINT_HINGE:
            return

        var body_a = Int(state[env, joint_off + JOINT3D_BODY_A])
        var body_b = Int(state[env, joint_off + JOINT3D_BODY_B])

        var body_a_off = BODIES_OFFSET + body_a * BODY_STATE_SIZE_3D
        var body_b_off = BODIES_OFFSET + body_b * BODY_STATE_SIZE_3D

        # Get positions and orientations
        var pa = Vec3(
            Float64(state[env, body_a_off + IDX_PX]),
            Float64(state[env, body_a_off + IDX_PY]),
            Float64(state[env, body_a_off + IDX_PZ]),
        )
        var pb = Vec3(
            Float64(state[env, body_b_off + IDX_PX]),
            Float64(state[env, body_b_off + IDX_PY]),
            Float64(state[env, body_b_off + IDX_PZ]),
        )

        var qa = Quat(
            Float64(state[env, body_a_off + IDX_QW]),
            Float64(state[env, body_a_off + IDX_QX]),
            Float64(state[env, body_a_off + IDX_QY]),
            Float64(state[env, body_a_off + IDX_QZ]),
        )
        var qb = Quat(
            Float64(state[env, body_b_off + IDX_QW]),
            Float64(state[env, body_b_off + IDX_QX]),
            Float64(state[env, body_b_off + IDX_QY]),
            Float64(state[env, body_b_off + IDX_QZ]),
        )

        # Local anchors
        var anchor_a_local = Vec3(
            Float64(state[env, joint_off + JOINT3D_ANCHOR_AX]),
            Float64(state[env, joint_off + JOINT3D_ANCHOR_AY]),
            Float64(state[env, joint_off + JOINT3D_ANCHOR_AZ]),
        )
        var anchor_b_local = Vec3(
            Float64(state[env, joint_off + JOINT3D_ANCHOR_BX]),
            Float64(state[env, joint_off + JOINT3D_ANCHOR_BY]),
            Float64(state[env, joint_off + JOINT3D_ANCHOR_BZ]),
        )

        # World anchors
        var anchor_a_world = pa + qa.rotate_vec(anchor_a_local)
        var anchor_b_world = pb + qb.rotate_vec(anchor_b_local)

        # Position error
        var error = anchor_b_world - anchor_a_world
        var error_mag = error.length()

        if error_mag < Float64(slop):
            return

        # Get mass properties
        var inv_ma = Float64(state[env, body_a_off + IDX_INV_MASS])
        var inv_mb = Float64(state[env, body_b_off + IDX_INV_MASS])

        # Simplified mass for position correction
        var total_inv_mass = inv_ma + inv_mb
        if total_inv_mass < 1e-10:
            return

        # Position correction
        var correction = error * Float64(baumgarte)
        var delta_a = correction * (inv_ma / total_inv_mass)
        var delta_b = correction * (-inv_mb / total_inv_mass)

        # Apply position corrections
        state[env, body_a_off + IDX_PX] = Scalar[dtype](pa.x - delta_a.x)
        state[env, body_a_off + IDX_PY] = Scalar[dtype](pa.y - delta_a.y)
        state[env, body_a_off + IDX_PZ] = Scalar[dtype](pa.z - delta_a.z)

        state[env, body_b_off + IDX_PX] = Scalar[dtype](pb.x + delta_b.x)
        state[env, body_b_off + IDX_PY] = Scalar[dtype](pb.y + delta_b.y)
        state[env, body_b_off + IDX_PZ] = Scalar[dtype](pb.z + delta_b.z)

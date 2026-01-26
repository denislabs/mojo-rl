"""ArticulatedChain - Kinematic chain representation for 2D articulated bodies.

Provides utilities for defining and manipulating articulated body chains
such as those used in Hopper, Walker2d, and HalfCheetah planar environments.
"""

from math import cos, sin, sqrt
from layout import LayoutTensor, Layout

from physics_gpu.constants import (
    dtype,
    BODY_STATE_SIZE,
    JOINT_DATA_SIZE,
    IDX_X,
    IDX_Y,
    IDX_ANGLE,
    IDX_VX,
    IDX_VY,
    IDX_OMEGA,
    IDX_FX,
    IDX_FY,
    IDX_TAU,
    IDX_MASS,
    IDX_INV_MASS,
    IDX_INV_INERTIA,
    IDX_SHAPE,
    JOINT_TYPE,
    JOINT_BODY_A,
    JOINT_BODY_B,
    JOINT_ANCHOR_AX,
    JOINT_ANCHOR_AY,
    JOINT_ANCHOR_BX,
    JOINT_ANCHOR_BY,
    JOINT_REF_ANGLE,
    JOINT_LOWER_LIMIT,
    JOINT_UPPER_LIMIT,
    JOINT_MAX_MOTOR_TORQUE,
    JOINT_MOTOR_SPEED,
    JOINT_FLAGS,
    JOINT_REVOLUTE,
    JOINT_FLAG_LIMIT_ENABLED,
    JOINT_FLAG_MOTOR_ENABLED,
)

from .constants import (
    CHAIN_BODY_COUNT,
    CHAIN_JOINT_COUNT,
    CHAIN_ROOT_IDX,
    CHAIN_HEADER_SIZE,
    LINK_DATA_SIZE,
    LINK_PARENT_IDX,
    LINK_JOINT_IDX,
    LINK_LENGTH,
    LINK_WIDTH,
    DEFAULT_KP,
    DEFAULT_KD,
    DEFAULT_MAX_TORQUE,
)


struct LinkDef:
    """Definition of a link in an articulated chain.

    Used to construct chains programmatically before simulation.
    """

    var name: String
    var parent_idx: Int  # -1 for root
    var length: Float64
    var width: Float64
    var mass: Float64
    var local_anchor: Tuple[Float64, Float64]  # Where joint attaches on parent
    var joint_axis: Float64  # Reference angle for joint (radians)
    var joint_lower: Float64  # Lower angle limit
    var joint_upper: Float64  # Upper angle limit

    fn __init__(
        out self,
        name: String,
        parent_idx: Int,
        length: Float64,
        width: Float64,
        mass: Float64,
        local_anchor: Tuple[Float64, Float64] = (0.0, 0.0),
        joint_axis: Float64 = 0.0,
        joint_lower: Float64 = -1.57,
        joint_upper: Float64 = 1.57,
    ):
        self.name = name
        self.parent_idx = parent_idx
        self.length = length
        self.width = width
        self.mass = mass
        self.local_anchor = local_anchor
        self.joint_axis = joint_axis
        self.joint_lower = joint_lower
        self.joint_upper = joint_upper


fn compute_link_inertia(mass: Float64, length: Float64, width: Float64) -> Float64:
    """Compute moment of inertia for a rectangular link (rod).

    I = (1/12) * m * (l^2 + w^2) for rectangle about center
    """
    return (1.0 / 12.0) * mass * (length * length + width * width)


struct ArticulatedChain[
    NUM_BODIES: Int,
    NUM_JOINTS: Int,
]:
    """Articulated chain utilities for 2D multi-body systems.

    Provides methods for:
    - Forward kinematics (computing link positions from joint angles)
    - Jacobian computation (for inverse dynamics)
    - Motor torque application
    - Joint angle/velocity extraction
    """

    # =========================================================================
    # Forward Kinematics
    # =========================================================================

    @staticmethod
    fn compute_link_transforms[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        parent_indices: StaticTuple[Int, NUM_BODIES],
    ):
        """Compute world-space transforms for all links from joint angles.

        Assumes root body (index 0) transform is already set.
        Propagates transforms through the kinematic chain.
        """
        # Process each link in order (assuming topological sort)
        for i in range(1, NUM_BODIES):
            var parent_idx = parent_indices[i]
            if parent_idx < 0:
                continue  # Root body, already positioned

            var parent_off = BODIES_OFFSET + parent_idx * BODY_STATE_SIZE
            var child_off = BODIES_OFFSET + i * BODY_STATE_SIZE

            # Get parent transform
            var px = state[env, parent_off + IDX_X]
            var py = state[env, parent_off + IDX_Y]
            var p_angle = state[env, parent_off + IDX_ANGLE]

            # Get joint connecting parent to child (joint i-1 connects body i-1 to body i)
            var joint_idx = i - 1
            if joint_idx >= 0 and joint_idx < NUM_JOINTS:
                var joint_off = JOINTS_OFFSET + joint_idx * JOINT_DATA_SIZE

                # Get joint anchors (local coordinates)
                var anchor_ax = state[env, joint_off + JOINT_ANCHOR_AX]
                var anchor_ay = state[env, joint_off + JOINT_ANCHOR_AY]
                var anchor_bx = state[env, joint_off + JOINT_ANCHOR_BX]
                var anchor_by = state[env, joint_off + JOINT_ANCHOR_BY]
                var ref_angle = state[env, joint_off + JOINT_REF_ANGLE]

                # Transform parent anchor to world
                var cos_p = cos(p_angle)
                var sin_p = sin(p_angle)
                var world_anchor_x = px + anchor_ax * cos_p - anchor_ay * sin_p
                var world_anchor_y = py + anchor_ax * sin_p + anchor_ay * cos_p

                # Child angle = parent angle + joint angle + reference angle
                # Note: Actual joint angle is computed from motor/constraint state
                var child_angle = state[env, child_off + IDX_ANGLE]

                # Position child so its anchor coincides with parent anchor
                var cos_c = cos(child_angle)
                var sin_c = sin(child_angle)
                var child_x = world_anchor_x - (anchor_bx * cos_c - anchor_by * sin_c)
                var child_y = world_anchor_y - (anchor_bx * sin_c + anchor_by * cos_c)

                state[env, child_off + IDX_X] = child_x
                state[env, child_off + IDX_Y] = child_y

    # =========================================================================
    # Joint Angle/Velocity Extraction
    # =========================================================================

    @staticmethod
    fn get_joint_angles[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        out angles: StaticTuple[Scalar[dtype], NUM_JOINTS],
    ):
        """Extract joint angles from body states.

        Joint angle = angle_b - angle_a - reference_angle
        """
        for j in range(NUM_JOINTS):
            var joint_off = JOINTS_OFFSET + j * JOINT_DATA_SIZE

            var body_a = Int(state[env, joint_off + JOINT_BODY_A])
            var body_b = Int(state[env, joint_off + JOINT_BODY_B])
            var ref_angle = state[env, joint_off + JOINT_REF_ANGLE]

            var angle_a = state[env, BODIES_OFFSET + body_a * BODY_STATE_SIZE + IDX_ANGLE]
            var angle_b = state[env, BODIES_OFFSET + body_b * BODY_STATE_SIZE + IDX_ANGLE]

            angles[j] = angle_b - angle_a - ref_angle

    @staticmethod
    fn get_joint_velocities[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        out velocities: StaticTuple[Scalar[dtype], NUM_JOINTS],
    ):
        """Extract joint angular velocities from body states.

        Joint velocity = omega_b - omega_a
        """
        for j in range(NUM_JOINTS):
            var joint_off = JOINTS_OFFSET + j * JOINT_DATA_SIZE

            var body_a = Int(state[env, joint_off + JOINT_BODY_A])
            var body_b = Int(state[env, joint_off + JOINT_BODY_B])

            var omega_a = state[env, BODIES_OFFSET + body_a * BODY_STATE_SIZE + IDX_OMEGA]
            var omega_b = state[env, BODIES_OFFSET + body_b * BODY_STATE_SIZE + IDX_OMEGA]

            velocities[j] = omega_b - omega_a

    # =========================================================================
    # Motor Control
    # =========================================================================

    @staticmethod
    fn apply_motor_torques[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        actions: StaticTuple[Scalar[dtype], NUM_JOINTS],
        max_torque: Scalar[dtype] = Scalar[dtype](DEFAULT_MAX_TORQUE),
    ):
        """Apply motor torques to joints based on action inputs.

        Actions are interpreted as target torques, clamped to max_torque.
        """
        for j in range(NUM_JOINTS):
            var joint_off = JOINTS_OFFSET + j * JOINT_DATA_SIZE

            var body_a = Int(state[env, joint_off + JOINT_BODY_A])
            var body_b = Int(state[env, joint_off + JOINT_BODY_B])

            # Clamp action to max torque
            var torque = actions[j]
            if torque > max_torque:
                torque = max_torque
            if torque < -max_torque:
                torque = -max_torque

            # Apply equal and opposite torques to connected bodies
            var body_a_off = BODIES_OFFSET + body_a * BODY_STATE_SIZE
            var body_b_off = BODIES_OFFSET + body_b * BODY_STATE_SIZE

            state[env, body_a_off + IDX_TAU] = state[env, body_a_off + IDX_TAU] - torque
            state[env, body_b_off + IDX_TAU] = state[env, body_b_off + IDX_TAU] + torque

    @staticmethod
    fn apply_pd_control[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
        target_angles: StaticTuple[Scalar[dtype], NUM_JOINTS],
        kp: Scalar[dtype] = Scalar[dtype](DEFAULT_KP),
        kd: Scalar[dtype] = Scalar[dtype](DEFAULT_KD),
        max_torque: Scalar[dtype] = Scalar[dtype](DEFAULT_MAX_TORQUE),
    ):
        """Apply PD control to joints.

        torque = kp * (target - current) - kd * velocity
        """
        for j in range(NUM_JOINTS):
            var joint_off = JOINTS_OFFSET + j * JOINT_DATA_SIZE

            var body_a = Int(state[env, joint_off + JOINT_BODY_A])
            var body_b = Int(state[env, joint_off + JOINT_BODY_B])
            var ref_angle = state[env, joint_off + JOINT_REF_ANGLE]

            # Current joint state
            var angle_a = state[env, BODIES_OFFSET + body_a * BODY_STATE_SIZE + IDX_ANGLE]
            var angle_b = state[env, BODIES_OFFSET + body_b * BODY_STATE_SIZE + IDX_ANGLE]
            var omega_a = state[env, BODIES_OFFSET + body_a * BODY_STATE_SIZE + IDX_OMEGA]
            var omega_b = state[env, BODIES_OFFSET + body_b * BODY_STATE_SIZE + IDX_OMEGA]

            var current_angle = angle_b - angle_a - ref_angle
            var current_velocity = omega_b - omega_a

            # PD control
            var error = target_angles[j] - current_angle
            var torque = kp * error - kd * current_velocity

            # Clamp to max torque
            if torque > max_torque:
                torque = max_torque
            if torque < -max_torque:
                torque = -max_torque

            # Apply equal and opposite torques
            var body_a_off = BODIES_OFFSET + body_a * BODY_STATE_SIZE
            var body_b_off = BODIES_OFFSET + body_b * BODY_STATE_SIZE

            state[env, body_a_off + IDX_TAU] = state[env, body_a_off + IDX_TAU] - torque
            state[env, body_b_off + IDX_TAU] = state[env, body_b_off + IDX_TAU] + torque

    # =========================================================================
    # Observation Helpers
    # =========================================================================

    @staticmethod
    fn get_root_state[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
    ) -> Tuple[
        Scalar[dtype],
        Scalar[dtype],
        Scalar[dtype],
        Scalar[dtype],
        Scalar[dtype],
        Scalar[dtype],
    ]:
        """Get root body state (x, y, angle, vx, vy, omega)."""
        var root_off = BODIES_OFFSET

        return (
            state[env, root_off + IDX_X],
            state[env, root_off + IDX_Y],
            state[env, root_off + IDX_ANGLE],
            state[env, root_off + IDX_VX],
            state[env, root_off + IDX_VY],
            state[env, root_off + IDX_OMEGA],
        )

    @staticmethod
    fn get_center_of_mass[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
        env: Int,
    ) -> Tuple[Scalar[dtype], Scalar[dtype]]:
        """Compute center of mass position for the entire chain."""
        var total_mass = Scalar[dtype](0.0)
        var com_x = Scalar[dtype](0.0)
        var com_y = Scalar[dtype](0.0)

        for i in range(NUM_BODIES):
            var body_off = BODIES_OFFSET + i * BODY_STATE_SIZE
            var mass = state[env, body_off + IDX_MASS]
            var x = state[env, body_off + IDX_X]
            var y = state[env, body_off + IDX_Y]

            com_x += mass * x
            com_y += mass * y
            total_mass += mass

        if total_mass > Scalar[dtype](1e-10):
            com_x /= total_mass
            com_y /= total_mass

        return (com_x, com_y)

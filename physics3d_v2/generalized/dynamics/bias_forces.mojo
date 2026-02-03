"""Bias forces computation for Generalized Coordinates engine.

Computes the bias forces b(q, qvel) = C(q, qvel) + g(q) where:
- C(q, qvel): Coriolis and centrifugal forces
- g(q): Gravitational forces

For simple HINGE chains (pendulums), the gravity term dominates:
- bias[i] = m * g * L * sin(theta) for each joint

Reference: Featherstone, "Rigid Body Dynamics Algorithms"
"""

from math import sin, cos
from ..types import ModelGC, DataGC
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..kinematics.quat_math import quat_rotate


# =============================================================================
# Bias Forces for HINGE-only Chains
# =============================================================================


fn compute_bias_forces[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    V_SIZE: Int,
](
    model: ModelGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    data: DataGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    mut bias: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """Compute bias forces b(q, qvel) = C(q, qvel) + g(q).

    For each joint, computes the gravitational torque/force and Coriolis terms.
    For HINGE joints in simple chains, this is primarily gravitational torque.

    Args:
        model: Static model configuration with gravity.
        data: Current state (qpos, qvel, xpos, xquat from FK).
        bias: Output bias force vector (NV elements).

    Note: These forces oppose motion, so qacc = M^-1 * (qfrc - bias).
    """
    # Initialize to zero
    for i in range(NV):
        bias[i] = Scalar[DTYPE](0)

    # Get gravity
    var gx = model.gravity[0]
    var gy = model.gravity[1]
    var gz = model.gravity[2]

    # For each joint, compute gravitational contribution
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var body = joint.body_id
        var dof_idx = joint.dof_adr

        if joint.jnt_type == JNT_HINGE:
            # Gravitational torque = r x (m * g)
            # where r is from joint axis to body CoM

            # Get joint position in world frame
            var parent = model.body_parent[body]
            var joint_pos_x = joint.pos_x
            var joint_pos_y = joint.pos_y
            var joint_pos_z = joint.pos_z

            var jpos_world_x = joint_pos_x
            var jpos_world_y = joint_pos_y
            var jpos_world_z = joint_pos_z

            if parent >= 0:
                var parent_px = data.xpos[parent * 3 + 0]
                var parent_py = data.xpos[parent * 3 + 1]
                var parent_pz = data.xpos[parent * 3 + 2]
                var parent_qx = data.xquat[parent * 4 + 0]
                var parent_qy = data.xquat[parent * 4 + 1]
                var parent_qz = data.xquat[parent * 4 + 2]
                var parent_qw = data.xquat[parent * 4 + 3]

                var rotated = quat_rotate(
                    parent_qx, parent_qy, parent_qz, parent_qw,
                    joint_pos_x, joint_pos_y, joint_pos_z
                )
                jpos_world_x = parent_px + rotated[0]
                jpos_world_y = parent_py + rotated[1]
                jpos_world_z = parent_pz + rotated[2]

            # Get joint axis in world frame
            var axis_x = joint.axis_x
            var axis_y = joint.axis_y
            var axis_z = joint.axis_z

            if parent >= 0:
                var parent_qx = data.xquat[parent * 4 + 0]
                var parent_qy = data.xquat[parent * 4 + 1]
                var parent_qz = data.xquat[parent * 4 + 2]
                var parent_qw = data.xquat[parent * 4 + 3]
                var axis_world = quat_rotate(
                    parent_qx, parent_qy, parent_qz, parent_qw,
                    axis_x, axis_y, axis_z
                )
                axis_x = axis_world[0]
                axis_y = axis_world[1]
                axis_z = axis_world[2]

            # Compute gravitational torque from body and all descendants
            var tau_gravity = Scalar[DTYPE](0)

            # Body contribution
            var body_px = data.xpos[body * 3 + 0]
            var body_py = data.xpos[body * 3 + 1]
            var body_pz = data.xpos[body * 3 + 2]
            var mass = model.body_mass[body]

            # Vector from joint to body CoM
            var r_x = body_px - jpos_world_x
            var r_y = body_py - jpos_world_y
            var r_z = body_pz - jpos_world_z

            # Gravity force
            var fg_x = mass * gx
            var fg_y = mass * gy
            var fg_z = mass * gz

            # Torque = r x F
            var tau_x = r_y * fg_z - r_z * fg_y
            var tau_y = r_z * fg_x - r_x * fg_z
            var tau_z = r_x * fg_y - r_y * fg_x

            # Project onto joint axis
            tau_gravity = tau_gravity + (tau_x * axis_x + tau_y * axis_y + tau_z * axis_z)

            # Add contributions from descendant bodies
            for desc_body in range(body + 1, NBODY):
                if _is_descendant(model, desc_body, body):
                    var desc_px = data.xpos[desc_body * 3 + 0]
                    var desc_py = data.xpos[desc_body * 3 + 1]
                    var desc_pz = data.xpos[desc_body * 3 + 2]
                    var desc_mass = model.body_mass[desc_body]

                    var desc_r_x = desc_px - jpos_world_x
                    var desc_r_y = desc_py - jpos_world_y
                    var desc_r_z = desc_pz - jpos_world_z

                    var desc_fg_x = desc_mass * gx
                    var desc_fg_y = desc_mass * gy
                    var desc_fg_z = desc_mass * gz

                    var desc_tau_x = desc_r_y * desc_fg_z - desc_r_z * desc_fg_y
                    var desc_tau_y = desc_r_z * desc_fg_x - desc_r_x * desc_fg_z
                    var desc_tau_z = desc_r_x * desc_fg_y - desc_r_y * desc_fg_x

                    tau_gravity = tau_gravity + (
                        desc_tau_x * axis_x + desc_tau_y * axis_y + desc_tau_z * axis_z
                    )

            # Store bias force (note: sign convention - bias opposes motion)
            bias[dof_idx] = -tau_gravity

        elif joint.jnt_type == JNT_SLIDE:
            # Gravitational force along slide axis

            # Get joint axis in world frame
            var parent = model.body_parent[body]
            var axis_x = joint.axis_x
            var axis_y = joint.axis_y
            var axis_z = joint.axis_z

            if parent >= 0:
                var parent_qx = data.xquat[parent * 4 + 0]
                var parent_qy = data.xquat[parent * 4 + 1]
                var parent_qz = data.xquat[parent * 4 + 2]
                var parent_qw = data.xquat[parent * 4 + 3]
                var axis_world = quat_rotate(
                    parent_qx, parent_qy, parent_qz, parent_qw,
                    axis_x, axis_y, axis_z
                )
                axis_x = axis_world[0]
                axis_y = axis_world[1]
                axis_z = axis_world[2]

            # Total mass of body and descendants
            var total_mass = model.body_mass[body]
            for desc_body in range(body + 1, NBODY):
                if _is_descendant(model, desc_body, body):
                    total_mass = total_mass + model.body_mass[desc_body]

            # Gravity force component along axis
            var fg_dot_axis = total_mass * (gx * axis_x + gy * axis_y + gz * axis_z)

            bias[dof_idx] = -fg_dot_axis

        elif joint.jnt_type == JNT_FREE:
            # FREE joint: direct gravity forces and torques
            var total_mass = model.body_mass[body]

            for desc_body in range(body + 1, NBODY):
                if _is_descendant(model, desc_body, body):
                    total_mass = total_mass + model.body_mass[desc_body]

            # Linear gravity force
            bias[dof_idx + 0] = -total_mass * gx
            bias[dof_idx + 1] = -total_mass * gy
            bias[dof_idx + 2] = -total_mass * gz

            # Angular gravity torque (usually zero for symmetric bodies at CoM)
            bias[dof_idx + 3] = Scalar[DTYPE](0)
            bias[dof_idx + 4] = Scalar[DTYPE](0)
            bias[dof_idx + 5] = Scalar[DTYPE](0)

        elif joint.jnt_type == JNT_BALL:
            # BALL joint: gravity doesn't directly create torque at CoM
            # (but would if CoM offset from joint)
            bias[dof_idx + 0] = Scalar[DTYPE](0)
            bias[dof_idx + 1] = Scalar[DTYPE](0)
            bias[dof_idx + 2] = Scalar[DTYPE](0)


# =============================================================================
# Helper Functions
# =============================================================================


fn _is_descendant[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
](
    model: ModelGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    body: Int,
    ancestor: Int,
) -> Bool:
    """Check if body is a descendant of ancestor in the kinematic tree."""
    var current = body
    while current >= 0:
        if model.body_parent[current] == ancestor:
            return True
        current = model.body_parent[current]
    return False


# =============================================================================
# Coriolis Forces (for higher-order accuracy)
# =============================================================================


fn compute_coriolis_forces[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
](
    model: ModelGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    data: DataGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    mut coriolis: InlineArray[Scalar[DTYPE], NV],
):
    """Compute Coriolis and centrifugal forces.

    For simple HINGE chains at low velocities, these are often negligible
    compared to gravity. This is a placeholder for more accurate dynamics.

    Args:
        model: Static model configuration.
        data: Current state with qvel.
        coriolis: Output Coriolis force vector.
    """
    # Initialize to zero (simplified - ignore Coriolis for now)
    for i in range(NV):
        coriolis[i] = Scalar[DTYPE](0)

    # TODO: Implement Coriolis terms for higher accuracy
    # C(q, qvel) involves velocity-dependent coupling between joints
    # For a double pendulum:
    # C[0] = -m2*l1*l2*sin(q1-q0)*qvel[1]^2 - m2*l2*(l1*sin(q1-q0)*qvel[0]*qvel[1])
    # C[1] = m2*l1*l2*sin(q1-q0)*qvel[0]^2

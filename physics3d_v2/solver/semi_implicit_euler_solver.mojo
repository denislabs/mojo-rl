"""Semi-implicit Euler solver for Generalized Coordinates engine.


"""

from math import sqrt
from ..types import ModelGC, DataGC, _max_one
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from ..kinematics.quat_math import quat_normalize, quat_integrate, quat_rotate
from ..dynamics.mass_matrix import compute_mass_matrix, solve_linear_diagonal
from ..dynamics.bias_forces import compute_bias_forces


fn normalize_qpos_quaternions[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
](
    model: ModelGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    mut data: DataGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
):
    """Normalize quaternions in qpos for BALL and FREE joints."""
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var qpos_adr = joint.qpos_adr

        if joint.jnt_type == JNT_FREE:
            # Quaternion at qpos_adr + 3..6
            var qx = data.qpos[qpos_adr + 3]
            var qy = data.qpos[qpos_adr + 4]
            var qz = data.qpos[qpos_adr + 5]
            var qw = data.qpos[qpos_adr + 6]

            var normalized = quat_normalize(qx, qy, qz, qw)
            data.qpos[qpos_adr + 3] = normalized[0]
            data.qpos[qpos_adr + 4] = normalized[1]
            data.qpos[qpos_adr + 5] = normalized[2]
            data.qpos[qpos_adr + 6] = normalized[3]

        elif joint.jnt_type == JNT_BALL:
            # Quaternion at qpos_adr + 0..3
            var qx = data.qpos[qpos_adr + 0]
            var qy = data.qpos[qpos_adr + 1]
            var qz = data.qpos[qpos_adr + 2]
            var qw = data.qpos[qpos_adr + 3]

            var normalized = quat_normalize(qx, qy, qz, qw)
            data.qpos[qpos_adr + 0] = normalized[0]
            data.qpos[qpos_adr + 1] = normalized[1]
            data.qpos[qpos_adr + 2] = normalized[2]
            data.qpos[qpos_adr + 3] = normalized[3]


fn enforce_joint_limits[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
](
    model: ModelGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    mut data: DataGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
):
    """Enforce joint position limits for HINGE and SLIDE joints.

    When a joint exceeds its limit:
    1. Clamp position to the limit
    2. Zero velocity if moving further into the limit

    This provides hard constraint behavior similar to MuJoCo's joint limits.
    """
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var qpos_adr = joint.qpos_adr
        var dof_adr = joint.dof_adr

        # Only enforce limits for HINGE and SLIDE joints
        if joint.jnt_type == JNT_HINGE or joint.jnt_type == JNT_SLIDE:
            var pos = data.qpos[qpos_adr]
            var vel = data.qvel[dof_adr]
            var range_min = joint.range_min
            var range_max = joint.range_max

            # Check lower limit
            if pos < range_min:
                data.qpos[qpos_adr] = range_min
                # Zero velocity if moving into the limit
                if vel < Scalar[DTYPE](0):
                    data.qvel[dof_adr] = Scalar[DTYPE](0)

            # Check upper limit
            elif pos > range_max:
                data.qpos[qpos_adr] = range_max
                # Zero velocity if moving into the limit
                if vel > Scalar[DTYPE](0):
                    data.qvel[dof_adr] = Scalar[DTYPE](0)


fn detect_ground_contacts[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
](
    model: ModelGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    mut data: DataGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
):
    """Detect contacts between bodies and ground plane.

    For capsules, checks both endpoints (center ± half_length along axis).
    The capsule axis is determined by the body's world orientation.
    """
    data.num_contacts = 0
    var ground_z = model.ground_z

    for body in range(NBODY):
        var px = data.xpos[body * 3 + 0]
        var py = data.xpos[body * 3 + 1]
        var pz = data.xpos[body * 3 + 2]
        var radius = model.body_radius[body]
        var half_length = model.body_half_length[body]

        # Get body orientation
        var qx = data.xquat[body * 4 + 0]
        var qy = data.xquat[body * 4 + 1]
        var qz = data.xquat[body * 4 + 2]
        var qw = data.xquat[body * 4 + 3]

        # Capsule axis in local frame is (0, 0, 1) - along Z
        # Transform to world frame
        var axis_world = quat_rotate(qx, qy, qz, qw,
            Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1))
        var axis_x = axis_world[0]
        var axis_y = axis_world[1]
        var axis_z = axis_world[2]

        # For spheres (half_length = 0), just check center - radius
        if half_length <= Scalar[DTYPE](0.0001):
            var dist = pz - radius - ground_z
            if dist < Scalar[DTYPE](0):
                if data.num_contacts < MAX_CONTACTS:
                    var idx = data.num_contacts
                    data.contacts[idx].body_a = body
                    data.contacts[idx].body_b = -1  # Ground
                    data.contacts[idx].pos_x = px
                    data.contacts[idx].pos_y = py
                    data.contacts[idx].pos_z = ground_z
                    data.contacts[idx].normal_x = Scalar[DTYPE](0)
                    data.contacts[idx].normal_y = Scalar[DTYPE](0)
                    data.contacts[idx].normal_z = Scalar[DTYPE](1)
                    data.contacts[idx].dist = dist
                    data.num_contacts += 1
        else:
            # Capsule: check both endpoints
            # Endpoint 1: center + half_length * axis
            var e1_x = px + half_length * axis_x
            var e1_y = py + half_length * axis_y
            var e1_z = pz + half_length * axis_z
            var dist1 = e1_z - radius - ground_z

            # Endpoint 2: center - half_length * axis
            var e2_x = px - half_length * axis_x
            var e2_y = py - half_length * axis_y
            var e2_z = pz - half_length * axis_z
            var dist2 = e2_z - radius - ground_z

            # Check endpoint 1
            if dist1 < Scalar[DTYPE](0):
                if data.num_contacts < MAX_CONTACTS:
                    var idx = data.num_contacts
                    data.contacts[idx].body_a = body
                    data.contacts[idx].body_b = -1  # Ground
                    data.contacts[idx].pos_x = e1_x
                    data.contacts[idx].pos_y = e1_y
                    data.contacts[idx].pos_z = ground_z
                    data.contacts[idx].normal_x = Scalar[DTYPE](0)
                    data.contacts[idx].normal_y = Scalar[DTYPE](0)
                    data.contacts[idx].normal_z = Scalar[DTYPE](1)
                    data.contacts[idx].dist = dist1
                    data.num_contacts += 1

            # Check endpoint 2
            if dist2 < Scalar[DTYPE](0):
                if data.num_contacts < MAX_CONTACTS:
                    var idx = data.num_contacts
                    data.contacts[idx].body_a = body
                    data.contacts[idx].body_b = -1  # Ground
                    data.contacts[idx].pos_x = e2_x
                    data.contacts[idx].pos_y = e2_y
                    data.contacts[idx].pos_z = ground_z
                    data.contacts[idx].normal_x = Scalar[DTYPE](0)
                    data.contacts[idx].normal_y = Scalar[DTYPE](0)
                    data.contacts[idx].normal_z = Scalar[DTYPE](1)
                    data.contacts[idx].dist = dist2
                    data.num_contacts += 1


fn compute_contact_forces[
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
    mut qfrc_contact: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """Compute joint-space contact forces from Cartesian contacts.

    Uses a simplified spring-damper model for ground contacts.
    Includes Coulomb friction for tangential forces.
    Forces are projected into joint space using the Jacobian transpose.
    """
    var stiffness = Scalar[DTYPE](5000.0)  # Ground stiffness
    var damping = Scalar[DTYPE](100.0)  # Ground damping
    var friction_coef = model.friction  # Friction coefficient

    for c in range(data.num_contacts):
        var contact = data.contacts[c]
        var body = contact.body_a

        if contact.dist >= Scalar[DTYPE](0):
            continue  # No penetration

        # Penetration depth (positive)
        var depth = -contact.dist

        # Body velocity at contact point
        var vx = data.xvel[body * 3 + 0]
        var vy = data.xvel[body * 3 + 1]
        var vz = data.xvel[body * 3 + 2]

        # Spring-damper normal force (in world z direction for ground)
        var normal_force = stiffness * depth - damping * vz
        if normal_force < Scalar[DTYPE](0):
            normal_force = Scalar[DTYPE](0)

        # Tangential velocity (in XY plane for ground contact)
        var v_tangent_x = vx
        var v_tangent_y = vy
        var v_tangent_mag = sqrt(v_tangent_x * v_tangent_x + v_tangent_y * v_tangent_y)

        # Coulomb friction force (opposes tangential velocity)
        var max_friction = friction_coef * normal_force
        var friction_x: Scalar[DTYPE] = Scalar[DTYPE](0)
        var friction_y: Scalar[DTYPE] = Scalar[DTYPE](0)

        if v_tangent_mag > Scalar[DTYPE](1e-6):
            # Kinetic friction: F = -mu * N * v_hat
            friction_x = -max_friction * (v_tangent_x / v_tangent_mag)
            friction_y = -max_friction * (v_tangent_y / v_tangent_mag)

        # Total contact force in world frame
        var total_fx = friction_x
        var total_fy = friction_y
        var total_fz = normal_force

        # Project to joint space using Jacobian transpose
        # For each joint affecting this body, compute torque/force contribution

        for j in range(model.num_joints):
            var joint = model.joints[j]

            # Check if this joint affects the contacted body
            if not _joint_affects_body(model, j, body):
                continue

            var dof_idx = joint.dof_adr

            if joint.jnt_type == JNT_HINGE:
                # Compute torque: tau = r x F, projected onto joint axis
                # r = contact position - joint position
                var parent = model.body_parent[joint.body_id]

                # Get joint position in world
                var jpos_x = joint.pos_x
                var jpos_y = joint.pos_y
                var jpos_z = joint.pos_z

                if parent >= 0:
                    var parent_px = data.xpos[parent * 3 + 0]
                    var parent_py = data.xpos[parent * 3 + 1]
                    var parent_pz = data.xpos[parent * 3 + 2]
                    var parent_qx = data.xquat[parent * 4 + 0]
                    var parent_qy = data.xquat[parent * 4 + 1]
                    var parent_qz = data.xquat[parent * 4 + 2]
                    var parent_qw = data.xquat[parent * 4 + 3]

                    var rotated = quat_rotate(
                        parent_qx,
                        parent_qy,
                        parent_qz,
                        parent_qw,
                        jpos_x,
                        jpos_y,
                        jpos_z,
                    )
                    jpos_x = parent_px + rotated[0]
                    jpos_y = parent_py + rotated[1]
                    jpos_z = parent_pz + rotated[2]

                # Lever arm from joint to contact
                var rx = contact.pos_x - jpos_x
                var ry = contact.pos_y - jpos_y
                var rz = contact.pos_z - jpos_z

                # Torque = r x F (using total force with friction)
                var tau_x = ry * total_fz - rz * total_fy
                var tau_y = rz * total_fx - rx * total_fz
                var tau_z = rx * total_fy - ry * total_fx

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
                        parent_qx,
                        parent_qy,
                        parent_qz,
                        parent_qw,
                        axis_x,
                        axis_y,
                        axis_z,
                    )
                    axis_x = axis_world[0]
                    axis_y = axis_world[1]
                    axis_z = axis_world[2]

                # Project torque onto axis
                var tau_joint = tau_x * axis_x + tau_y * axis_y + tau_z * axis_z
                qfrc_contact[dof_idx] = qfrc_contact[dof_idx] + tau_joint

            elif joint.jnt_type == JNT_SLIDE:
                # Force along axis (now includes friction)
                var axis_x = joint.axis_x
                var axis_y = joint.axis_y
                var axis_z = joint.axis_z

                var parent = model.body_parent[joint.body_id]
                if parent >= 0:
                    var parent_qx = data.xquat[parent * 4 + 0]
                    var parent_qy = data.xquat[parent * 4 + 1]
                    var parent_qz = data.xquat[parent * 4 + 2]
                    var parent_qw = data.xquat[parent * 4 + 3]
                    var axis_world = quat_rotate(
                        parent_qx,
                        parent_qy,
                        parent_qz,
                        parent_qw,
                        axis_x,
                        axis_y,
                        axis_z,
                    )
                    axis_x = axis_world[0]
                    axis_y = axis_world[1]
                    axis_z = axis_world[2]

                # Project total force (with friction) onto axis
                var f_joint = total_fx * axis_x + total_fy * axis_y + total_fz * axis_z
                qfrc_contact[dof_idx] = qfrc_contact[dof_idx] + f_joint

            elif joint.jnt_type == JNT_FREE:
                # Direct force and torque (with friction)
                qfrc_contact[dof_idx + 0] = (
                    qfrc_contact[dof_idx + 0] + total_fx
                )
                qfrc_contact[dof_idx + 1] = (
                    qfrc_contact[dof_idx + 1] + total_fy
                )
                qfrc_contact[dof_idx + 2] = (
                    qfrc_contact[dof_idx + 2] + total_fz
                )


fn _joint_affects_body[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
](
    model: ModelGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    joint_idx: Int,
    body_idx: Int,
) -> Bool:
    """Check if a joint affects a body (body is the joint's body or a descendant).
    """
    var joint_body = model.joints[joint_idx].body_id

    if body_idx == joint_body:
        return True

    # Check if body_idx is a descendant of joint_body
    var current = body_idx
    while current >= 0:
        if model.body_parent[current] == joint_body:
            return True
        current = model.body_parent[current]

    return False

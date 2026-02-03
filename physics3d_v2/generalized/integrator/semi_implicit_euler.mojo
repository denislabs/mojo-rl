"""Semi-implicit Euler integrator for Generalized Coordinates engine.

Implements the main simulation step:
1. Forward kinematics: qpos -> xpos, xquat
2. Collision detection (optional)
3. Compute dynamics: mass matrix M(q), bias forces b(q, qvel)
4. Solve: qacc = M^-1 * (qfrc - bias)
5. Integrate: qvel += qacc * dt, qpos += qvel * dt
6. Normalize quaternions in qpos

Semi-implicit Euler is symplectic and provides good energy conservation.
"""

from math import sqrt
from ..types import ModelGC, DataGC, _max_one
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..kinematics.forward_kinematics import forward_kinematics, compute_body_velocities
from ..kinematics.quat_math import quat_normalize, quat_integrate, quat_rotate
from ..dynamics.mass_matrix import compute_mass_matrix, solve_linear_diagonal
from ..dynamics.bias_forces import compute_bias_forces


# =============================================================================
# Main Step Function
# =============================================================================


fn step_gc[
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
    """Execute one simulation step using semi-implicit Euler.

    Args:
        model: Static model configuration (masses, joints, gravity).
        data: Mutable simulation state (qpos, qvel, qfrc).

    After this function:
    - data.qvel is updated with new velocities
    - data.qpos is updated with new positions
    - data.xpos, data.xquat contain world-space poses
    """
    var dt = model.timestep

    # 1. Forward kinematics: compute xpos, xquat from qpos
    forward_kinematics(model, data)

    # 2. Compute body velocities from qvel
    compute_body_velocities(model, data)

    # 3. Compute mass matrix M(q)
    # Use the same size as the DataGC arrays for consistency
    comptime M_SIZE = _max_one[NV * NV]()
    comptime V_SIZE = _max_one[NV]()

    var M = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    for i in range(M_SIZE):
        M[i] = Scalar[DTYPE](0)
    compute_mass_matrix[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, M_SIZE](model, data, M)

    # 4. Compute bias forces b(q, qvel)
    var bias = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(V_SIZE):
        bias[i] = Scalar[DTYPE](0)
    compute_bias_forces[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE](model, data, bias)

    # 5. Compute net force: f_net = qfrc - bias
    var f_net = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(V_SIZE):
        f_net[i] = Scalar[DTYPE](0)
    for i in range(NV):
        f_net[i] = data.qfrc[i] - bias[i]

    # 6. Solve M * qacc = f_net
    # Using diagonal approximation for simplicity
    solve_linear_diagonal[DTYPE, NV, M_SIZE, V_SIZE](M, f_net, data.qacc)

    # 7. Semi-implicit Euler integration
    # First update velocities, then positions
    for i in range(NV):
        data.qvel[i] = data.qvel[i] + data.qacc[i] * dt

    for i in range(NQ):
        data.qpos[i] = data.qpos[i] + data.qvel[i] * dt

    # 8. Normalize quaternions in qpos (for BALL and FREE joints)
    _normalize_qpos_quaternions(model, data)


# =============================================================================
# Step with Collision Detection
# =============================================================================


fn step_gc_with_contacts[
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
    """Execute one simulation step with collision detection.

    Same as step_gc but includes ground plane collision detection.
    Contact forces are applied as additional joint-space forces.

    Args:
        model: Static model configuration.
        data: Mutable simulation state.
    """
    var dt = model.timestep

    # 1. Forward kinematics
    forward_kinematics(model, data)
    compute_body_velocities(model, data)

    # 2. Collision detection with ground
    _detect_ground_contacts(model, data)

    # 3. Compute dynamics
    comptime M_SIZE = _max_one[NV * NV]()
    comptime V_SIZE = _max_one[NV]()

    var M = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    for i in range(M_SIZE):
        M[i] = Scalar[DTYPE](0)
    compute_mass_matrix[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, M_SIZE](model, data, M)

    var bias = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(V_SIZE):
        bias[i] = Scalar[DTYPE](0)
    compute_bias_forces[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE](model, data, bias)

    # 4. Compute contact forces in joint space
    var qfrc_contact = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(V_SIZE):
        qfrc_contact[i] = Scalar[DTYPE](0)
    _compute_contact_forces(model, data, qfrc_contact)

    # 5. Net force
    var f_net = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(V_SIZE):
        f_net[i] = Scalar[DTYPE](0)
    for i in range(NV):
        f_net[i] = data.qfrc[i] + qfrc_contact[i] - bias[i]

    # 6. Solve M * qacc = f_net
    solve_linear_diagonal[DTYPE, NV, M_SIZE, V_SIZE](M, f_net, data.qacc)

    # 7. Integration
    for i in range(NV):
        data.qvel[i] = data.qvel[i] + data.qacc[i] * dt

    for i in range(NQ):
        data.qpos[i] = data.qpos[i] + data.qvel[i] * dt

    # 8. Normalize quaternions
    _normalize_qpos_quaternions(model, data)


# =============================================================================
# Helper Functions
# =============================================================================


fn _normalize_qpos_quaternions[
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


fn _detect_ground_contacts[
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
    """Detect contacts between bodies and ground plane."""
    data.num_contacts = 0
    var ground_z = model.ground_z

    for body in range(NBODY):
        var px = data.xpos[body * 3 + 0]
        var py = data.xpos[body * 3 + 1]
        var pz = data.xpos[body * 3 + 2]
        var radius = model.body_radius[body]

        # Check if body penetrates ground
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


fn _compute_contact_forces[
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
    Forces are projected into joint space using the Jacobian transpose.
    """
    var stiffness = Scalar[DTYPE](5000.0)  # Ground stiffness
    var damping = Scalar[DTYPE](100.0)  # Ground damping

    for c in range(data.num_contacts):
        var contact = data.contacts[c]
        var body = contact.body_a

        if contact.dist >= Scalar[DTYPE](0):
            continue  # No penetration

        # Penetration depth (positive)
        var depth = -contact.dist

        # Normal velocity
        var vz = data.xvel[body * 3 + 2]

        # Spring-damper force (in world z direction)
        var fn = stiffness * depth - damping * vz
        if fn < Scalar[DTYPE](0):
            fn = Scalar[DTYPE](0)

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
                # r = body position - joint position
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
                        parent_qx, parent_qy, parent_qz, parent_qw,
                        jpos_x, jpos_y, jpos_z
                    )
                    jpos_x = parent_px + rotated[0]
                    jpos_y = parent_py + rotated[1]
                    jpos_z = parent_pz + rotated[2]

                # Lever arm from joint to contact
                var rx = contact.pos_x - jpos_x
                var ry = contact.pos_y - jpos_y
                var rz = contact.pos_z - jpos_z

                # Force is in z direction (normal)
                var fx = fn * contact.normal_x
                var fy = fn * contact.normal_y
                var fz = fn * contact.normal_z

                # Torque = r x F
                var tau_x = ry * fz - rz * fy
                var tau_y = rz * fx - rx * fz
                var tau_z = rx * fy - ry * fx

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

                # Project torque onto axis
                var tau_joint = tau_x * axis_x + tau_y * axis_y + tau_z * axis_z
                qfrc_contact[dof_idx] = qfrc_contact[dof_idx] + tau_joint

            elif joint.jnt_type == JNT_SLIDE:
                # Force along axis
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
                        parent_qx, parent_qy, parent_qz, parent_qw,
                        axis_x, axis_y, axis_z
                    )
                    axis_x = axis_world[0]
                    axis_y = axis_world[1]
                    axis_z = axis_world[2]

                # Force in world frame
                var fx = fn * contact.normal_x
                var fy = fn * contact.normal_y
                var fz = fn * contact.normal_z

                # Project onto axis
                var f_joint = fx * axis_x + fy * axis_y + fz * axis_z
                qfrc_contact[dof_idx] = qfrc_contact[dof_idx] + f_joint

            elif joint.jnt_type == JNT_FREE:
                # Direct force and torque
                qfrc_contact[dof_idx + 0] = qfrc_contact[dof_idx + 0] + fn * contact.normal_x
                qfrc_contact[dof_idx + 1] = qfrc_contact[dof_idx + 1] + fn * contact.normal_y
                qfrc_contact[dof_idx + 2] = qfrc_contact[dof_idx + 2] + fn * contact.normal_z


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
    """Check if a joint affects a body (body is the joint's body or a descendant)."""
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

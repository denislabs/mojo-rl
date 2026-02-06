"""Joint Constraint Solver for Physics3D v2.

Implements velocity and position constraint solving for hinge and slide joints.

Physics:
- Hinge joint constrains 5 DOF (3 linear + 2 angular)
- Slide joint constrains 5 DOF (2 perpendicular + 3 angular)
- Position constraint: Anchor points must coincide (hinge) or be on slide axis (slide)
- Angular constraint: Rotation only around hinge axis (hinge) or locked (slide)

Reference: Adapted from physics3d/solvers/joint_solver3d.mojo
"""

from math import sqrt
from ..types import Model, Data
from .hinge_joint import HingeJoint
from .slide_joint import SlideJoint
from utils.numerics import isnan

# =============================================================================
# Helper Functions
# =============================================================================


fn _quat_rotate[
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

    q * v * q^-1 = v + 2*w*(w x v) + 2*(w x (w x v))
    where q = [qx, qy, qz, qw] (scalar last convention)
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


fn _get_world_anchor[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int, MAX_SLIDE_JOINTS: Int = 0
](
    data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    body_idx: Int,
    local_x: Scalar[DTYPE],
    local_y: Scalar[DTYPE],
    local_z: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Transform local anchor point to world space."""
    if body_idx < 0:
        # World anchor - local coordinates are already world coordinates
        return (local_x, local_y, local_z)

    # Get body position
    var px = data.positions[body_idx * 3 + 0]
    var py = data.positions[body_idx * 3 + 1]
    var pz = data.positions[body_idx * 3 + 2]

    # Get body quaternion
    var qx = data.quaternions[body_idx * 4 + 0]
    var qy = data.quaternions[body_idx * 4 + 1]
    var qz = data.quaternions[body_idx * 4 + 2]
    var qw = data.quaternions[body_idx * 4 + 3]

    # Rotate local anchor to world frame
    var rotated = _quat_rotate(qx, qy, qz, qw, local_x, local_y, local_z)

    return (px + rotated[0], py + rotated[1], pz + rotated[2])


fn _get_world_axis[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int, MAX_SLIDE_JOINTS: Int = 0
](
    data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    body_idx: Int,
    local_x: Scalar[DTYPE],
    local_y: Scalar[DTYPE],
    local_z: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Transform local axis to world space (normalized)."""
    if body_idx < 0:
        # World axis - already in world frame
        return (local_x, local_y, local_z)

    # Get body quaternion
    var qx = data.quaternions[body_idx * 4 + 0]
    var qy = data.quaternions[body_idx * 4 + 1]
    var qz = data.quaternions[body_idx * 4 + 2]
    var qw = data.quaternions[body_idx * 4 + 3]

    # Rotate local axis to world frame
    var rotated = _quat_rotate(qx, qy, qz, qw, local_x, local_y, local_z)

    # Normalize
    var length_sq = (
        rotated[0] * rotated[0]
        + rotated[1] * rotated[1]
        + rotated[2] * rotated[2]
    )
    var inv_length = Scalar[DTYPE](1.0) / sqrt(length_sq + Scalar[DTYPE](1e-10))

    return (
        rotated[0] * inv_length,
        rotated[1] * inv_length,
        rotated[2] * inv_length,
    )


# =============================================================================
# Joint State Sensing (Observation)
# =============================================================================


fn _quat_conjugate[
    DTYPE: DType
](
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Compute quaternion conjugate (inverse for unit quaternions)."""
    return (-qx, -qy, -qz, qw)


fn _quat_multiply[
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
    """Multiply two quaternions: result = a * b."""
    var rx = aw * bx + ax * bw + ay * bz - az * by
    var ry = aw * by - ax * bz + ay * bw + az * bx
    var rz = aw * bz + ax * by - ay * bx + az * bw
    var rw = aw * bw - ax * bx - ay * by - az * bz
    return (rx, ry, rz, rw)


fn get_joint_angle[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int, MAX_SLIDE_JOINTS: Int = 0
](
    model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    joint_idx: Int,
) -> Scalar[DTYPE]:
    """Compute the current hinge angle in radians.

    The angle is measured as the rotation of the child body relative to the
    parent body (or world) around the hinge axis. Zero angle corresponds to
    the initial configuration when both bodies have identity quaternions.

    Args:
        model: Static model configuration with joints.
        data: Current simulation state.
        joint_idx: Index of the joint to query.

    Returns:
        Angle in radians. Positive is counterclockwise around the axis.
    """
    var joint = model.joints[joint_idx]
    var body_a = joint.parent_body
    var body_b = joint.child_body

    # Get quaternions (identity for world)
    var qa_x: Scalar[DTYPE] = 0
    var qa_y: Scalar[DTYPE] = 0
    var qa_z: Scalar[DTYPE] = 0
    var qa_w: Scalar[DTYPE] = 1

    if body_a >= 0:
        qa_x = data.quaternions[body_a * 4 + 0]
        qa_y = data.quaternions[body_a * 4 + 1]
        qa_z = data.quaternions[body_a * 4 + 2]
        qa_w = data.quaternions[body_a * 4 + 3]

    var qb_x = data.quaternions[body_b * 4 + 0]
    var qb_y = data.quaternions[body_b * 4 + 1]
    var qb_z = data.quaternions[body_b * 4 + 2]
    var qb_w = data.quaternions[body_b * 4 + 3]

    # Compute relative quaternion: q_rel = q_a^(-1) * q_b
    var qa_conj = _quat_conjugate(qa_x, qa_y, qa_z, qa_w)
    var q_rel = _quat_multiply(
        qa_conj[0], qa_conj[1], qa_conj[2], qa_conj[3], qb_x, qb_y, qb_z, qb_w
    )

    # Get hinge axis in parent frame (or world if parent=-1)
    var ax = joint.axis_x
    var ay = joint.axis_y
    var az = joint.axis_z

    # Project quaternion rotation onto hinge axis
    # For a rotation around axis, the quaternion is:
    #   q = [sin(θ/2) * axis, cos(θ/2)]
    # So the angle component along axis is:
    #   sin(θ/2) = q_xyz · axis / |q_xyz|
    #   cos(θ/2) = q_w
    # And θ = 2 * atan2(sin(θ/2), cos(θ/2))

    # Dot product of quaternion vector part with axis
    var sin_half_theta = q_rel[0] * ax + q_rel[1] * ay + q_rel[2] * az
    var cos_half_theta = q_rel[3]

    # Use atan2 for full angle range
    from math import atan2

    var angle = Scalar[DTYPE](2.0) * atan2(sin_half_theta, cos_half_theta)

    return angle


fn get_joint_angular_velocity[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int, MAX_SLIDE_JOINTS: Int = 0
](
    model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    joint_idx: Int,
) -> Scalar[DTYPE]:
    """Compute the angular velocity around the hinge axis.

    This is the component of relative angular velocity projected onto
    the hinge axis direction.

    Args:
        model: Static model configuration with joints.
        data: Current simulation state.
        joint_idx: Index of the joint to query.

    Returns:
        Angular velocity in rad/s. Positive is counterclockwise around axis.
    """
    var joint = model.joints[joint_idx]
    var body_a = joint.parent_body
    var body_b = joint.child_body

    # Get angular velocities
    var wa_x: Scalar[DTYPE] = 0
    var wa_y: Scalar[DTYPE] = 0
    var wa_z: Scalar[DTYPE] = 0

    if body_a >= 0:
        wa_x = data.angular_velocities[body_a * 3 + 0]
        wa_y = data.angular_velocities[body_a * 3 + 1]
        wa_z = data.angular_velocities[body_a * 3 + 2]

    var wb_x = data.angular_velocities[body_b * 3 + 0]
    var wb_y = data.angular_velocities[body_b * 3 + 1]
    var wb_z = data.angular_velocities[body_b * 3 + 2]

    # Relative angular velocity
    var rel_wx = wb_x - wa_x
    var rel_wy = wb_y - wa_y
    var rel_wz = wb_z - wa_z

    # Get world-space hinge axis
    var axis = _get_world_axis(
        data, body_a, joint.axis_x, joint.axis_y, joint.axis_z
    )

    # Project onto hinge axis
    var omega_hinge = rel_wx * axis[0] + rel_wy * axis[1] + rel_wz * axis[2]

    return omega_hinge


# =============================================================================
# Joint Torque Application (Actuation)
# =============================================================================


fn apply_joint_torques[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int, MAX_SLIDE_JOINTS: Int = 0
](
    model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    dt: Scalar[DTYPE],
):
    """Apply actuator torques to angular velocities.

    This should be called early in the physics step, before constraint solving.
    Torques are applied around the hinge axis to both parent and child bodies
    (action-reaction pair).

    Args:
        model: Static model configuration with joints.
        data: Mutable simulation state.
        dt: Timestep for integration.
    """
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var body_a = joint.parent_body
        var body_b = joint.child_body

        # Get torque, clamped to limits
        var torque = joint.target_torque
        if torque > joint.torque_limit:
            torque = joint.torque_limit
        elif torque < -joint.torque_limit:
            torque = -joint.torque_limit

        # Skip if no torque
        if torque * torque < Scalar[DTYPE](1e-12):
            continue

        # Get world-space hinge axis
        var axis = _get_world_axis(
            data, body_a, joint.axis_x, joint.axis_y, joint.axis_z
        )

        # Apply torque to child body: Δω = τ × axis × inv_I × dt
        # For a hinge, torque is scalar around the axis
        var inv_I_b = (
            model.inv_inertias[body_b * 3 + 0]
            + model.inv_inertias[body_b * 3 + 1]
            + model.inv_inertias[body_b * 3 + 2]
        ) / Scalar[DTYPE](3.0)

        var delta_w = torque * inv_I_b * dt
        data.angular_velocities[body_b * 3 + 0] += delta_w * axis[0]
        data.angular_velocities[body_b * 3 + 1] += delta_w * axis[1]
        data.angular_velocities[body_b * 3 + 2] += delta_w * axis[2]

        # Apply reaction torque to parent (Newton's third law)
        if body_a >= 0:
            var inv_I_a = (
                model.inv_inertias[body_a * 3 + 0]
                + model.inv_inertias[body_a * 3 + 1]
                + model.inv_inertias[body_a * 3 + 2]
            ) / Scalar[DTYPE](3.0)

            var delta_w_a = torque * inv_I_a * dt
            data.angular_velocities[body_a * 3 + 0] -= delta_w_a * axis[0]
            data.angular_velocities[body_a * 3 + 1] -= delta_w_a * axis[1]
            data.angular_velocities[body_a * 3 + 2] -= delta_w_a * axis[2]


# =============================================================================
# Velocity Constraint Solving
# =============================================================================


fn _solve_single_joint_velocity[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int, MAX_SLIDE_JOINTS: Int = 0
](
    model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    joint_idx: Int,
):
    """Solve velocity constraints for a single hinge joint.

    Constrains:
    1. Anchor points to have same velocity (point-to-point)
    2. Angular velocities to differ only around hinge axis

    If is_free_dof=True, skip constraint solving (MuJoCo-style root joint).
    """
    var joint = model.joints[joint_idx]

    # Skip constraint solving for free DOF joints
    if joint.is_free_dof:
        return

    var body_a = joint.parent_body
    var body_b = joint.child_body

    # Get world-space anchors
    var anchor_a = _get_world_anchor(
        data,
        body_a,
        joint.anchor_parent_x,
        joint.anchor_parent_y,
        joint.anchor_parent_z,
    )
    var anchor_b = _get_world_anchor(
        data,
        body_b,
        joint.anchor_child_x,
        joint.anchor_child_y,
        joint.anchor_child_z,
    )

    # Get world-space hinge axis
    var axis = _get_world_axis(
        data, body_a, joint.axis_x, joint.axis_y, joint.axis_z
    )

    # --- Point-to-point velocity constraint (3 DOF) ---

    # Get velocities
    var va_x: Scalar[DTYPE] = 0
    var va_y: Scalar[DTYPE] = 0
    var va_z: Scalar[DTYPE] = 0
    var wa_x: Scalar[DTYPE] = 0
    var wa_y: Scalar[DTYPE] = 0
    var wa_z: Scalar[DTYPE] = 0
    var pa_x: Scalar[DTYPE] = 0
    var pa_y: Scalar[DTYPE] = 0
    var pa_z: Scalar[DTYPE] = 0

    if body_a >= 0:
        va_x = data.velocities[body_a * 3 + 0]
        va_y = data.velocities[body_a * 3 + 1]
        va_z = data.velocities[body_a * 3 + 2]
        wa_x = data.angular_velocities[body_a * 3 + 0]
        wa_y = data.angular_velocities[body_a * 3 + 1]
        wa_z = data.angular_velocities[body_a * 3 + 2]
        pa_x = data.positions[body_a * 3 + 0]
        pa_y = data.positions[body_a * 3 + 1]
        pa_z = data.positions[body_a * 3 + 2]

    var vb_x = data.velocities[body_b * 3 + 0]
    var vb_y = data.velocities[body_b * 3 + 1]
    var vb_z = data.velocities[body_b * 3 + 2]
    var wb_x = data.angular_velocities[body_b * 3 + 0]
    var wb_y = data.angular_velocities[body_b * 3 + 1]
    var wb_z = data.angular_velocities[body_b * 3 + 2]
    var pb_x = data.positions[body_b * 3 + 0]
    var pb_y = data.positions[body_b * 3 + 1]
    var pb_z = data.positions[body_b * 3 + 2]

    # Lever arms from body centers to anchors
    var ra_x = anchor_a[0] - pa_x
    var ra_y = anchor_a[1] - pa_y
    var ra_z = anchor_a[2] - pa_z
    var rb_x = anchor_b[0] - pb_x
    var rb_y = anchor_b[1] - pb_y
    var rb_z = anchor_b[2] - pb_z

    # Velocity at anchor A: v_a + w_a x r_a
    var vel_anchor_a_x = va_x + (wa_y * ra_z - wa_z * ra_y)
    var vel_anchor_a_y = va_y + (wa_z * ra_x - wa_x * ra_z)
    var vel_anchor_a_z = va_z + (wa_x * ra_y - wa_y * ra_x)

    # Velocity at anchor B: v_b + w_b x r_b
    var vel_anchor_b_x = vb_x + (wb_y * rb_z - wb_z * rb_y)
    var vel_anchor_b_y = vb_y + (wb_z * rb_x - wb_x * rb_z)
    var vel_anchor_b_z = vb_z + (wb_x * rb_y - wb_y * rb_x)

    # Velocity error
    var dv_x = vel_anchor_a_x - vel_anchor_b_x
    var dv_y = vel_anchor_a_y - vel_anchor_b_y
    var dv_z = vel_anchor_a_z - vel_anchor_b_z

    # Compute effective mass including rotational contribution
    # K = inv_mass_a + inv_mass_b + r_a^2 * inv_inertia_a + r_b^2 * inv_inertia_b
    var inv_mass_a: Scalar[DTYPE] = 0
    var rot_contrib_a: Scalar[DTYPE] = 0
    if body_a >= 0:
        inv_mass_a = model.inv_masses[body_a]
        var ra_sq = ra_x * ra_x + ra_y * ra_y + ra_z * ra_z
        var avg_inv_inertia_a = (
            model.inv_inertias[body_a * 3 + 0]
            + model.inv_inertias[body_a * 3 + 1]
            + model.inv_inertias[body_a * 3 + 2]
        ) / Scalar[DTYPE](3.0)
        rot_contrib_a = ra_sq * avg_inv_inertia_a

    var inv_mass_b = model.inv_masses[body_b]
    var rb_sq = rb_x * rb_x + rb_y * rb_y + rb_z * rb_z
    var avg_inv_inertia_b = (
        model.inv_inertias[body_b * 3 + 0]
        + model.inv_inertias[body_b * 3 + 1]
        + model.inv_inertias[body_b * 3 + 2]
    ) / Scalar[DTYPE](3.0)
    var rot_contrib_b = rb_sq * avg_inv_inertia_b

    var K = inv_mass_a + inv_mass_b + rot_contrib_a + rot_contrib_b
    if K < Scalar[DTYPE](1e-10):
        return

    # Impulse to correct velocity error (with relaxation for stability)
    var relaxation = Scalar[DTYPE](0.8)
    var impulse_x = -relaxation * dv_x / K
    var impulse_y = -relaxation * dv_y / K
    var impulse_z = -relaxation * dv_z / K

    # Apply linear impulse
    if body_a >= 0:
        data.velocities[body_a * 3 + 0] += impulse_x * inv_mass_a
        data.velocities[body_a * 3 + 1] += impulse_y * inv_mass_a
        data.velocities[body_a * 3 + 2] += impulse_z * inv_mass_a

        # Apply angular impulse from linear: tau = r x f
        var tau_a_x = ra_y * impulse_z - ra_z * impulse_y
        var tau_a_y = ra_z * impulse_x - ra_x * impulse_z
        var tau_a_z = ra_x * impulse_y - ra_y * impulse_x
        data.angular_velocities[body_a * 3 + 0] += (
            tau_a_x * model.inv_inertias[body_a * 3 + 0]
        )
        data.angular_velocities[body_a * 3 + 1] += (
            tau_a_y * model.inv_inertias[body_a * 3 + 1]
        )
        data.angular_velocities[body_a * 3 + 2] += (
            tau_a_z * model.inv_inertias[body_a * 3 + 2]
        )

    data.velocities[body_b * 3 + 0] -= impulse_x * inv_mass_b
    data.velocities[body_b * 3 + 1] -= impulse_y * inv_mass_b
    data.velocities[body_b * 3 + 2] -= impulse_z * inv_mass_b

    var tau_b_x = rb_y * impulse_z - rb_z * impulse_y
    var tau_b_y = rb_z * impulse_x - rb_x * impulse_z
    var tau_b_z = rb_x * impulse_y - rb_y * impulse_x
    data.angular_velocities[body_b * 3 + 0] -= (
        tau_b_x * model.inv_inertias[body_b * 3 + 0]
    )
    data.angular_velocities[body_b * 3 + 1] -= (
        tau_b_y * model.inv_inertias[body_b * 3 + 1]
    )
    data.angular_velocities[body_b * 3 + 2] -= (
        tau_b_z * model.inv_inertias[body_b * 3 + 2]
    )

    # --- Angular constraint (2 DOF) - restrict rotation to hinge axis ---

    # Re-read angular velocities (they may have changed)
    if body_a >= 0:
        wa_x = data.angular_velocities[body_a * 3 + 0]
        wa_y = data.angular_velocities[body_a * 3 + 1]
        wa_z = data.angular_velocities[body_a * 3 + 2]
    else:
        wa_x = Scalar[DTYPE](0)
        wa_y = Scalar[DTYPE](0)
        wa_z = Scalar[DTYPE](0)

    wb_x = data.angular_velocities[body_b * 3 + 0]
    wb_y = data.angular_velocities[body_b * 3 + 1]
    wb_z = data.angular_velocities[body_b * 3 + 2]

    # Relative angular velocity
    var rel_omega_x = wa_x - wb_x
    var rel_omega_y = wa_y - wb_y
    var rel_omega_z = wa_z - wb_z

    # Component along hinge axis (this is allowed)
    var omega_dot_axis = (
        rel_omega_x * axis[0] + rel_omega_y * axis[1] + rel_omega_z * axis[2]
    )
    var omega_along_x = axis[0] * omega_dot_axis
    var omega_along_y = axis[1] * omega_dot_axis
    var omega_along_z = axis[2] * omega_dot_axis

    # Component perpendicular to axis (should be zero)
    var omega_perp_x = rel_omega_x - omega_along_x
    var omega_perp_y = rel_omega_y - omega_along_y
    var omega_perp_z = rel_omega_z - omega_along_z

    var omega_perp_sq = (
        omega_perp_x * omega_perp_x
        + omega_perp_y * omega_perp_y
        + omega_perp_z * omega_perp_z
    )
    if omega_perp_sq < Scalar[DTYPE](1e-12):
        return

    # Compute angular effective mass (simplified)
    var k_angular: Scalar[DTYPE] = 0
    if body_a >= 0:
        k_angular += model.inv_inertias[body_a * 3 + 0]
        k_angular += model.inv_inertias[body_a * 3 + 1]
        k_angular += model.inv_inertias[body_a * 3 + 2]
    k_angular += model.inv_inertias[body_b * 3 + 0]
    k_angular += model.inv_inertias[body_b * 3 + 1]
    k_angular += model.inv_inertias[body_b * 3 + 2]

    if k_angular < Scalar[DTYPE](1e-10):
        return

    # Angular impulse to cancel perpendicular component (with damping factor)
    var damping = Scalar[DTYPE](0.5)
    var ang_impulse_x = -omega_perp_x * damping / k_angular
    var ang_impulse_y = -omega_perp_y * damping / k_angular
    var ang_impulse_z = -omega_perp_z * damping / k_angular

    # Apply angular impulse
    if body_a >= 0:
        data.angular_velocities[body_a * 3 + 0] += (
            ang_impulse_x * model.inv_inertias[body_a * 3 + 0]
        )
        data.angular_velocities[body_a * 3 + 1] += (
            ang_impulse_y * model.inv_inertias[body_a * 3 + 1]
        )
        data.angular_velocities[body_a * 3 + 2] += (
            ang_impulse_z * model.inv_inertias[body_a * 3 + 2]
        )

    data.angular_velocities[body_b * 3 + 0] -= (
        ang_impulse_x * model.inv_inertias[body_b * 3 + 0]
    )
    data.angular_velocities[body_b * 3 + 1] -= (
        ang_impulse_y * model.inv_inertias[body_b * 3 + 1]
    )
    data.angular_velocities[body_b * 3 + 2] -= (
        ang_impulse_z * model.inv_inertias[body_b * 3 + 2]
    )


fn solve_joint_velocity_constraints[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int, MAX_SLIDE_JOINTS: Int = 0
](
    model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    iterations: Int = 10,
):
    """Solve velocity constraints for all joints.

    Args:
        model: Static model configuration with joints.
        data: Mutable simulation state.
        iterations: Number of solver iterations.
    """
    for _ in range(iterations):
        for j in range(model.num_joints):
            _solve_single_joint_velocity(model, data, j)


# =============================================================================
# Position Constraint Solving
# =============================================================================


fn _solve_single_joint_position[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int, MAX_SLIDE_JOINTS: Int = 0
](
    model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    joint_idx: Int,
    baumgarte: Scalar[DTYPE],
):
    """Solve position constraint for a single hinge joint.

    Uses Baumgarte stabilization to correct anchor point drift.
    If is_free_dof=True, skip constraint solving (MuJoCo-style root joint).
    """
    var joint = model.joints[joint_idx]

    # Skip constraint solving for free DOF joints
    if joint.is_free_dof:
        return

    var body_a = joint.parent_body
    var body_b = joint.child_body

    # Get world-space anchors
    var anchor_a = _get_world_anchor(
        data,
        body_a,
        joint.anchor_parent_x,
        joint.anchor_parent_y,
        joint.anchor_parent_z,
    )
    var anchor_b = _get_world_anchor(
        data,
        body_b,
        joint.anchor_child_x,
        joint.anchor_child_y,
        joint.anchor_child_z,
    )

    # Position error
    var error_x = anchor_a[0] - anchor_b[0]
    var error_y = anchor_a[1] - anchor_b[1]
    var error_z = anchor_a[2] - anchor_b[2]

    var error_len_sq = error_x * error_x + error_y * error_y + error_z * error_z
    if error_len_sq < Scalar[DTYPE](1e-12):
        return

    # Compute effective mass
    var inv_mass_a: Scalar[DTYPE] = 0
    if body_a >= 0:
        inv_mass_a = model.inv_masses[body_a]
    var inv_mass_b = model.inv_masses[body_b]

    var total_inv_mass = inv_mass_a + inv_mass_b
    if total_inv_mass < Scalar[DTYPE](1e-10):
        return

    # Position correction (Baumgarte stabilization)
    var correction_x = -baumgarte * error_x / total_inv_mass
    var correction_y = -baumgarte * error_y / total_inv_mass
    var correction_z = -baumgarte * error_z / total_inv_mass

    # Apply position correction
    if body_a >= 0:
        data.positions[body_a * 3 + 0] += correction_x * inv_mass_a
        data.positions[body_a * 3 + 1] += correction_y * inv_mass_a
        data.positions[body_a * 3 + 2] += correction_z * inv_mass_a

    data.positions[body_b * 3 + 0] -= correction_x * inv_mass_b
    data.positions[body_b * 3 + 1] -= correction_y * inv_mass_b
    data.positions[body_b * 3 + 2] -= correction_z * inv_mass_b


fn solve_joint_position_constraints[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int, MAX_SLIDE_JOINTS: Int = 0
](
    model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    baumgarte: Scalar[DTYPE] = 0.2,
    iterations: Int = 5,
):
    """Solve position constraints for all joints.

    Uses Baumgarte stabilization to correct anchor point drift.

    Args:
        model: Static model configuration with joints.
        data: Mutable simulation state.
        baumgarte: Stabilization factor (0.1-0.5 typical).
        iterations: Number of solver iterations.
    """
    for _ in range(iterations):
        for j in range(model.num_joints):
            _solve_single_joint_position(model, data, j, baumgarte)


# =============================================================================
# Slide Joint State Sensing (Observation)
# =============================================================================


fn get_slide_joint_position[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int, MAX_SLIDE_JOINTS: Int = 0
](
    model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    joint_idx: Int,
) -> Scalar[DTYPE]:
    """Compute the current slide position (displacement along axis).

    The position is measured as the displacement of the child anchor
    relative to the parent anchor projected onto the slide axis.

    Args:
        model: Static model configuration with slide joints.
        data: Current simulation state.
        joint_idx: Index of the slide joint to query.

    Returns:
        Position along slide axis in meters.
    """
    var joint = model.slide_joints[joint_idx]
    var body_a = joint.parent_body
    var body_b = joint.child_body

    # Get world-space anchors
    var anchor_a = _get_world_anchor(
        data,
        body_a,
        joint.anchor_parent_x,
        joint.anchor_parent_y,
        joint.anchor_parent_z,
    )
    var anchor_b = _get_world_anchor(
        data,
        body_b,
        joint.anchor_child_x,
        joint.anchor_child_y,
        joint.anchor_child_z,
    )

    # Get world-space slide axis
    var axis = _get_world_axis(
        data, body_a, joint.axis_x, joint.axis_y, joint.axis_z
    )

    # Displacement from parent anchor to child anchor
    var dx = anchor_b[0] - anchor_a[0]
    var dy = anchor_b[1] - anchor_a[1]
    var dz = anchor_b[2] - anchor_a[2]

    # Project onto slide axis
    var position = dx * axis[0] + dy * axis[1] + dz * axis[2]

    return position


fn get_slide_joint_velocity[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int, MAX_SLIDE_JOINTS: Int = 0
](
    model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    joint_idx: Int,
) -> Scalar[DTYPE]:
    """Compute the velocity along the slide axis.

    This is the component of relative velocity projected onto the slide axis.

    Args:
        model: Static model configuration with slide joints.
        data: Current simulation state.
        joint_idx: Index of the slide joint to query.

    Returns:
        Velocity along slide axis in m/s.
    """
    var joint = model.slide_joints[joint_idx]
    var body_a = joint.parent_body
    var body_b = joint.child_body

    # Get velocities
    var va_x: Scalar[DTYPE] = 0
    var va_y: Scalar[DTYPE] = 0
    var va_z: Scalar[DTYPE] = 0

    if body_a >= 0:
        va_x = data.velocities[body_a * 3 + 0]
        va_y = data.velocities[body_a * 3 + 1]
        va_z = data.velocities[body_a * 3 + 2]

    var vb_x = data.velocities[body_b * 3 + 0]
    var vb_y = data.velocities[body_b * 3 + 1]
    var vb_z = data.velocities[body_b * 3 + 2]

    # Relative velocity
    var rel_vx = vb_x - va_x
    var rel_vy = vb_y - va_y
    var rel_vz = vb_z - va_z

    # Get world-space slide axis
    var axis = _get_world_axis(
        data, body_a, joint.axis_x, joint.axis_y, joint.axis_z
    )

    # Project onto slide axis
    var velocity = rel_vx * axis[0] + rel_vy * axis[1] + rel_vz * axis[2]

    return velocity


# =============================================================================
# Slide Joint Force Application (Actuation)
# =============================================================================


fn apply_slide_joint_forces[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int, MAX_SLIDE_JOINTS: Int = 0
](
    model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    dt: Scalar[DTYPE],
):
    """Apply actuator forces to linear velocities along slide axis.

    This should be called early in the physics step, before constraint solving.
    Forces are applied along the slide axis to both parent and child bodies
    (action-reaction pair).

    Args:
        model: Static model configuration with slide joints.
        data: Mutable simulation state.
        dt: Timestep for integration.
    """
    for j in range(model.num_slide_joints):
        var joint = model.slide_joints[j]
        var body_a = joint.parent_body
        var body_b = joint.child_body

        # Get force, clamped to limits
        var force = joint.target_force
        if force > joint.force_limit:
            force = joint.force_limit
        elif force < -joint.force_limit:
            force = -joint.force_limit

        # Skip if no force
        if force * force < Scalar[DTYPE](1e-12):
            continue

        # Get world-space slide axis
        var axis = _get_world_axis(
            data, body_a, joint.axis_x, joint.axis_y, joint.axis_z
        )

        # Apply force to child body: Δv = F × inv_m × dt
        var inv_m_b = model.inv_masses[body_b]
        var delta_v = force * inv_m_b * dt
        data.velocities[body_b * 3 + 0] += delta_v * axis[0]
        data.velocities[body_b * 3 + 1] += delta_v * axis[1]
        data.velocities[body_b * 3 + 2] += delta_v * axis[2]

        # Apply reaction force to parent (Newton's third law)
        if body_a >= 0:
            var inv_m_a = model.inv_masses[body_a]
            var delta_v_a = force * inv_m_a * dt
            data.velocities[body_a * 3 + 0] -= delta_v_a * axis[0]
            data.velocities[body_a * 3 + 1] -= delta_v_a * axis[1]
            data.velocities[body_a * 3 + 2] -= delta_v_a * axis[2]


# =============================================================================
# Slide Joint Velocity Constraint Solving
# =============================================================================


fn _compute_slide_basis[
    DTYPE: DType
](
    axis_x: Scalar[DTYPE],
    axis_y: Scalar[DTYPE],
    axis_z: Scalar[DTYPE],
) -> Tuple[
    Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]],
    Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]],
]:
    """Compute two vectors perpendicular to the slide axis.

    Returns t1 and t2 such that (axis, t1, t2) form an orthonormal basis.
    """
    # Find a vector not parallel to axis
    var t1_x: Scalar[DTYPE]
    var t1_y: Scalar[DTYPE]
    var t1_z: Scalar[DTYPE]

    # Use cross product with either (1,0,0) or (0,1,0) to get perpendicular
    if abs(axis_x) < Scalar[DTYPE](0.9):
        # Cross with (1,0,0): (0, az, -ay)
        t1_x = Scalar[DTYPE](0)
        t1_y = axis_z
        t1_z = -axis_y
    else:
        # Cross with (0,1,0): (-az, 0, ax)
        t1_x = -axis_z
        t1_y = Scalar[DTYPE](0)
        t1_z = axis_x

    # Normalize t1
    var len_sq = t1_x * t1_x + t1_y * t1_y + t1_z * t1_z
    var inv_len = Scalar[DTYPE](1.0) / sqrt(len_sq + Scalar[DTYPE](1e-10))
    t1_x = t1_x * inv_len
    t1_y = t1_y * inv_len
    t1_z = t1_z * inv_len

    # t2 = axis × t1
    var t2_x = axis_y * t1_z - axis_z * t1_y
    var t2_y = axis_z * t1_x - axis_x * t1_z
    var t2_z = axis_x * t1_y - axis_y * t1_x

    return ((t1_x, t1_y, t1_z), (t2_x, t2_y, t2_z))


fn _solve_single_slide_joint_velocity[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int, MAX_SLIDE_JOINTS: Int = 0
](
    model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    joint_idx: Int,
):
    """Solve velocity constraints for a single slide joint.

    Constrains:
    1. Velocity perpendicular to slide axis (2 DOF)
    2. All angular velocities to be equal (3 DOF)

    If is_free_dof=True, skip constraint solving (MuJoCo-style root joint).
    """
    var joint = model.slide_joints[joint_idx]

    # Skip constraint solving for free DOF joints
    if joint.is_free_dof:
        return

    var body_a = joint.parent_body
    var body_b = joint.child_body

    # Get world-space anchors
    var anchor_a = _get_world_anchor(
        data,
        body_a,
        joint.anchor_parent_x,
        joint.anchor_parent_y,
        joint.anchor_parent_z,
    )
    var anchor_b = _get_world_anchor(
        data,
        body_b,
        joint.anchor_child_x,
        joint.anchor_child_y,
        joint.anchor_child_z,
    )

    # Get world-space slide axis and perpendicular basis
    var axis = _get_world_axis(
        data, body_a, joint.axis_x, joint.axis_y, joint.axis_z
    )
    var basis = _compute_slide_basis(axis[0], axis[1], axis[2])
    var t1 = basis[0]
    var t2 = basis[1]

    # --- Perpendicular velocity constraint (2 DOF) ---

    # Get velocities
    var va_x: Scalar[DTYPE] = 0
    var va_y: Scalar[DTYPE] = 0
    var va_z: Scalar[DTYPE] = 0
    var wa_x: Scalar[DTYPE] = 0
    var wa_y: Scalar[DTYPE] = 0
    var wa_z: Scalar[DTYPE] = 0
    var pa_x: Scalar[DTYPE] = 0
    var pa_y: Scalar[DTYPE] = 0
    var pa_z: Scalar[DTYPE] = 0

    if body_a >= 0:
        va_x = data.velocities[body_a * 3 + 0]
        va_y = data.velocities[body_a * 3 + 1]
        va_z = data.velocities[body_a * 3 + 2]
        wa_x = data.angular_velocities[body_a * 3 + 0]
        wa_y = data.angular_velocities[body_a * 3 + 1]
        wa_z = data.angular_velocities[body_a * 3 + 2]
        pa_x = data.positions[body_a * 3 + 0]
        pa_y = data.positions[body_a * 3 + 1]
        pa_z = data.positions[body_a * 3 + 2]

    var vb_x = data.velocities[body_b * 3 + 0]
    var vb_y = data.velocities[body_b * 3 + 1]
    var vb_z = data.velocities[body_b * 3 + 2]
    var wb_x = data.angular_velocities[body_b * 3 + 0]
    var wb_y = data.angular_velocities[body_b * 3 + 1]
    var wb_z = data.angular_velocities[body_b * 3 + 2]
    var pb_x = data.positions[body_b * 3 + 0]
    var pb_y = data.positions[body_b * 3 + 1]
    var pb_z = data.positions[body_b * 3 + 2]

    # Lever arms from body centers to anchors
    var ra_x = anchor_a[0] - pa_x
    var ra_y = anchor_a[1] - pa_y
    var ra_z = anchor_a[2] - pa_z
    var rb_x = anchor_b[0] - pb_x
    var rb_y = anchor_b[1] - pb_y
    var rb_z = anchor_b[2] - pb_z

    # Velocity at anchor A: v_a + w_a x r_a
    var vel_anchor_a_x = va_x + (wa_y * ra_z - wa_z * ra_y)
    var vel_anchor_a_y = va_y + (wa_z * ra_x - wa_x * ra_z)
    var vel_anchor_a_z = va_z + (wa_x * ra_y - wa_y * ra_x)

    # Velocity at anchor B: v_b + w_b x r_b
    var vel_anchor_b_x = vb_x + (wb_y * rb_z - wb_z * rb_y)
    var vel_anchor_b_y = vb_y + (wb_z * rb_x - wb_x * rb_z)
    var vel_anchor_b_z = vb_z + (wb_x * rb_y - wb_y * rb_x)

    # Relative velocity
    var dv_x = vel_anchor_a_x - vel_anchor_b_x
    var dv_y = vel_anchor_a_y - vel_anchor_b_y
    var dv_z = vel_anchor_a_z - vel_anchor_b_z

    # Project relative velocity onto perpendicular directions (should be zero)
    var dv_t1 = dv_x * t1[0] + dv_y * t1[1] + dv_z * t1[2]
    var dv_t2 = dv_x * t2[0] + dv_y * t2[1] + dv_z * t2[2]

    # Compute effective mass
    var inv_mass_a: Scalar[DTYPE] = 0
    var rot_contrib_a: Scalar[DTYPE] = 0
    if body_a >= 0:
        inv_mass_a = model.inv_masses[body_a]
        var ra_sq = ra_x * ra_x + ra_y * ra_y + ra_z * ra_z
        var avg_inv_inertia_a = (
            model.inv_inertias[body_a * 3 + 0]
            + model.inv_inertias[body_a * 3 + 1]
            + model.inv_inertias[body_a * 3 + 2]
        ) / Scalar[DTYPE](3.0)
        rot_contrib_a = ra_sq * avg_inv_inertia_a

    var inv_mass_b = model.inv_masses[body_b]
    var rb_sq = rb_x * rb_x + rb_y * rb_y + rb_z * rb_z
    var avg_inv_inertia_b = (
        model.inv_inertias[body_b * 3 + 0]
        + model.inv_inertias[body_b * 3 + 1]
        + model.inv_inertias[body_b * 3 + 2]
    ) / Scalar[DTYPE](3.0)
    var rot_contrib_b = rb_sq * avg_inv_inertia_b

    var K = inv_mass_a + inv_mass_b + rot_contrib_a + rot_contrib_b
    if K < Scalar[DTYPE](1e-10):
        return

    # Impulse to correct perpendicular velocity error
    var relaxation = Scalar[DTYPE](0.8)
    var impulse_t1 = -relaxation * dv_t1 / K
    var impulse_t2 = -relaxation * dv_t2 / K

    # Convert to world-space impulse
    var impulse_x = impulse_t1 * t1[0] + impulse_t2 * t2[0]
    var impulse_y = impulse_t1 * t1[1] + impulse_t2 * t2[1]
    var impulse_z = impulse_t1 * t1[2] + impulse_t2 * t2[2]

    # Apply linear impulse
    if body_a >= 0:
        data.velocities[body_a * 3 + 0] += impulse_x * inv_mass_a
        data.velocities[body_a * 3 + 1] += impulse_y * inv_mass_a
        data.velocities[body_a * 3 + 2] += impulse_z * inv_mass_a

        # Apply angular impulse from linear: tau = r x f
        var tau_a_x = ra_y * impulse_z - ra_z * impulse_y
        var tau_a_y = ra_z * impulse_x - ra_x * impulse_z
        var tau_a_z = ra_x * impulse_y - ra_y * impulse_x
        data.angular_velocities[body_a * 3 + 0] += (
            tau_a_x * model.inv_inertias[body_a * 3 + 0]
        )
        data.angular_velocities[body_a * 3 + 1] += (
            tau_a_y * model.inv_inertias[body_a * 3 + 1]
        )
        data.angular_velocities[body_a * 3 + 2] += (
            tau_a_z * model.inv_inertias[body_a * 3 + 2]
        )

    data.velocities[body_b * 3 + 0] -= impulse_x * inv_mass_b
    data.velocities[body_b * 3 + 1] -= impulse_y * inv_mass_b
    data.velocities[body_b * 3 + 2] -= impulse_z * inv_mass_b

    var tau_b_x = rb_y * impulse_z - rb_z * impulse_y
    var tau_b_y = rb_z * impulse_x - rb_x * impulse_z
    var tau_b_z = rb_x * impulse_y - rb_y * impulse_x
    data.angular_velocities[body_b * 3 + 0] -= (
        tau_b_x * model.inv_inertias[body_b * 3 + 0]
    )
    data.angular_velocities[body_b * 3 + 1] -= (
        tau_b_y * model.inv_inertias[body_b * 3 + 1]
    )
    data.angular_velocities[body_b * 3 + 2] -= (
        tau_b_z * model.inv_inertias[body_b * 3 + 2]
    )

    # --- Angular constraint (3 DOF) - lock all rotation ---

    # Re-read angular velocities (they may have changed)
    if body_a >= 0:
        wa_x = data.angular_velocities[body_a * 3 + 0]
        wa_y = data.angular_velocities[body_a * 3 + 1]
        wa_z = data.angular_velocities[body_a * 3 + 2]
    else:
        wa_x = Scalar[DTYPE](0)
        wa_y = Scalar[DTYPE](0)
        wa_z = Scalar[DTYPE](0)

    wb_x = data.angular_velocities[body_b * 3 + 0]
    wb_y = data.angular_velocities[body_b * 3 + 1]
    wb_z = data.angular_velocities[body_b * 3 + 2]

    # Relative angular velocity (should all be zero for slide joint)
    var rel_omega_x = wa_x - wb_x
    var rel_omega_y = wa_y - wb_y
    var rel_omega_z = wa_z - wb_z

    var omega_sq = (
        rel_omega_x * rel_omega_x
        + rel_omega_y * rel_omega_y
        + rel_omega_z * rel_omega_z
    )
    if omega_sq < Scalar[DTYPE](1e-12):
        return

    # Compute angular effective mass (simplified)
    var k_angular: Scalar[DTYPE] = 0
    if body_a >= 0:
        k_angular += model.inv_inertias[body_a * 3 + 0]
        k_angular += model.inv_inertias[body_a * 3 + 1]
        k_angular += model.inv_inertias[body_a * 3 + 2]
    k_angular += model.inv_inertias[body_b * 3 + 0]
    k_angular += model.inv_inertias[body_b * 3 + 1]
    k_angular += model.inv_inertias[body_b * 3 + 2]

    if k_angular < Scalar[DTYPE](1e-10):
        return

    # Angular impulse to cancel relative rotation (with damping factor)
    var damping = Scalar[DTYPE](0.5)
    var ang_impulse_x = -rel_omega_x * damping / k_angular
    var ang_impulse_y = -rel_omega_y * damping / k_angular
    var ang_impulse_z = -rel_omega_z * damping / k_angular

    # Apply angular impulse
    if body_a >= 0:
        data.angular_velocities[body_a * 3 + 0] += (
            ang_impulse_x * model.inv_inertias[body_a * 3 + 0]
        )
        data.angular_velocities[body_a * 3 + 1] += (
            ang_impulse_y * model.inv_inertias[body_a * 3 + 1]
        )
        data.angular_velocities[body_a * 3 + 2] += (
            ang_impulse_z * model.inv_inertias[body_a * 3 + 2]
        )

    data.angular_velocities[body_b * 3 + 0] -= (
        ang_impulse_x * model.inv_inertias[body_b * 3 + 0]
    )
    data.angular_velocities[body_b * 3 + 1] -= (
        ang_impulse_y * model.inv_inertias[body_b * 3 + 1]
    )
    data.angular_velocities[body_b * 3 + 2] -= (
        ang_impulse_z * model.inv_inertias[body_b * 3 + 2]
    )


fn solve_slide_joint_velocity_constraints[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int, MAX_SLIDE_JOINTS: Int = 0
](
    model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    iterations: Int = 10,
):
    """Solve velocity constraints for all slide joints.

    Args:
        model: Static model configuration with slide joints.
        data: Mutable simulation state.
        iterations: Number of solver iterations.
    """
    for _ in range(iterations):
        for j in range(model.num_slide_joints):
            _solve_single_slide_joint_velocity(model, data, j)


# =============================================================================
# Slide Joint Position Constraint Solving
# =============================================================================


fn _solve_single_slide_joint_position[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int, MAX_SLIDE_JOINTS: Int = 0
](
    model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    joint_idx: Int,
    baumgarte: Scalar[DTYPE],
):
    """Solve position constraint for a single slide joint.

    Uses Baumgarte stabilization to correct perpendicular drift.
    If is_free_dof=True, skip constraint solving (MuJoCo-style root joint).
    """
    var joint = model.slide_joints[joint_idx]

    # Skip constraint solving for free DOF joints
    if joint.is_free_dof:
        return

    var body_a = joint.parent_body
    var body_b = joint.child_body

    # Get world-space anchors
    var anchor_a = _get_world_anchor(
        data,
        body_a,
        joint.anchor_parent_x,
        joint.anchor_parent_y,
        joint.anchor_parent_z,
    )
    var anchor_b = _get_world_anchor(
        data,
        body_b,
        joint.anchor_child_x,
        joint.anchor_child_y,
        joint.anchor_child_z,
    )

    # Get world-space slide axis and perpendicular basis
    var axis = _get_world_axis(
        data, body_a, joint.axis_x, joint.axis_y, joint.axis_z
    )
    var basis = _compute_slide_basis(axis[0], axis[1], axis[2])
    var t1 = basis[0]
    var t2 = basis[1]

    # Position error (from parent anchor to child anchor)
    var error_x = anchor_b[0] - anchor_a[0]
    var error_y = anchor_b[1] - anchor_a[1]
    var error_z = anchor_b[2] - anchor_a[2]

    # Project error onto perpendicular directions (should be zero)
    var error_t1 = error_x * t1[0] + error_y * t1[1] + error_z * t1[2]
    var error_t2 = error_x * t2[0] + error_y * t2[1] + error_z * t2[2]

    var error_len_sq = error_t1 * error_t1 + error_t2 * error_t2
    if error_len_sq < Scalar[DTYPE](1e-12):
        return

    # Compute effective mass
    var inv_mass_a: Scalar[DTYPE] = 0
    if body_a >= 0:
        inv_mass_a = model.inv_masses[body_a]
    var inv_mass_b = model.inv_masses[body_b]

    var total_inv_mass = inv_mass_a + inv_mass_b
    if total_inv_mass < Scalar[DTYPE](1e-10):
        return

    # Position correction (Baumgarte stabilization)
    # Correct perpendicular error only
    var correction_t1 = -baumgarte * error_t1 / total_inv_mass
    var correction_t2 = -baumgarte * error_t2 / total_inv_mass

    # Convert to world-space correction
    var correction_x = correction_t1 * t1[0] + correction_t2 * t2[0]
    var correction_y = correction_t1 * t1[1] + correction_t2 * t2[1]
    var correction_z = correction_t1 * t1[2] + correction_t2 * t2[2]

    # Apply position correction
    if body_a >= 0:
        data.positions[body_a * 3 + 0] -= correction_x * inv_mass_a
        data.positions[body_a * 3 + 1] -= correction_y * inv_mass_a
        data.positions[body_a * 3 + 2] -= correction_z * inv_mass_a

    data.positions[body_b * 3 + 0] += correction_x * inv_mass_b
    data.positions[body_b * 3 + 1] += correction_y * inv_mass_b
    data.positions[body_b * 3 + 2] += correction_z * inv_mass_b


fn solve_slide_joint_position_constraints[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int, MAX_SLIDE_JOINTS: Int = 0
](
    model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS],
    baumgarte: Scalar[DTYPE] = 0.2,
    iterations: Int = 5,
):
    """Solve position constraints for all slide joints.

    Uses Baumgarte stabilization to correct perpendicular drift.

    Args:
        model: Static model configuration with slide joints.
        data: Mutable simulation state.
        baumgarte: Stabilization factor (0.1-0.5 typical).
        iterations: Number of solver iterations.
    """
    for _ in range(iterations):
        for j in range(model.num_slide_joints):
            _solve_single_slide_joint_position(model, data, j, baumgarte)


# =============================================================================
# GPU Joint Solver Functions
# =============================================================================

from layout import LayoutTensor, Layout
from ..gpu.constants import (
    body_offset,
    joint_offset,
    slide_joint_offset,
    metadata_offset,
    BODY_STATE_SIZE,
    BODY_IDX_PX,
    BODY_IDX_PY,
    BODY_IDX_PZ,
    BODY_IDX_QX,
    BODY_IDX_QY,
    BODY_IDX_QZ,
    BODY_IDX_QW,
    BODY_IDX_VX,
    BODY_IDX_VY,
    BODY_IDX_VZ,
    BODY_IDX_WX,
    BODY_IDX_WY,
    BODY_IDX_WZ,
    JOINT_STATE_SIZE,
    JOINT_IDX_PARENT,
    JOINT_IDX_CHILD,
    JOINT_IDX_ANCHOR_PX,
    JOINT_IDX_ANCHOR_PY,
    JOINT_IDX_ANCHOR_PZ,
    JOINT_IDX_ANCHOR_CX,
    JOINT_IDX_ANCHOR_CY,
    JOINT_IDX_ANCHOR_CZ,
    JOINT_IDX_AXIS_X,
    JOINT_IDX_AXIS_Y,
    JOINT_IDX_AXIS_Z,
    JOINT_IDX_TARGET_TORQUE,
    JOINT_IDX_TORQUE_LIMIT,
    JOINT_IDX_IS_FREE_DOF,
    JOINT_IDX_QPOS,
    JOINT_IDX_QVEL,
    # Slide joint constants
    SLIDE_JOINT_STATE_SIZE,
    SLIDE_IDX_PARENT,
    SLIDE_IDX_CHILD,
    SLIDE_IDX_ANCHOR_PX,
    SLIDE_IDX_ANCHOR_PY,
    SLIDE_IDX_ANCHOR_PZ,
    SLIDE_IDX_ANCHOR_CX,
    SLIDE_IDX_ANCHOR_CY,
    SLIDE_IDX_ANCHOR_CZ,
    SLIDE_IDX_AXIS_X,
    SLIDE_IDX_AXIS_Y,
    SLIDE_IDX_AXIS_Z,
    SLIDE_IDX_IMPULSE_P1,
    SLIDE_IDX_IMPULSE_P2,
    SLIDE_IDX_IMPULSE_AX,
    SLIDE_IDX_IMPULSE_AY,
    SLIDE_IDX_IMPULSE_AZ,
    SLIDE_IDX_TARGET_FORCE,
    SLIDE_IDX_FORCE_LIMIT,
    SLIDE_IDX_IS_FREE_DOF,
    SLIDE_IDX_QPOS,
    SLIDE_IDX_QVEL,
    META_IDX_NUM_CONTACTS,
    META_IDX_NUM_JOINTS,
    META_IDX_PADDING_2,
    META_IDX_PADDING_3,
    MODEL_BODY_SIZE,
    MODEL_IDX_INV_MASS,
    MODEL_IDX_INV_IXX,
    MODEL_IDX_INV_IYY,
    MODEL_IDX_INV_IZZ,
)


@always_inline
fn _quat_rotate_gpu[
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
    """Rotate a vector by a quaternion (GPU version)."""
    var tx = Scalar[DTYPE](2) * (qy * vz - qz * vy)
    var ty = Scalar[DTYPE](2) * (qz * vx - qx * vz)
    var tz = Scalar[DTYPE](2) * (qx * vy - qy * vx)
    var rx = vx + qw * tx + (qy * tz - qz * ty)
    var ry = vy + qw * ty + (qz * tx - qx * tz)
    var rz = vz + qw * tz + (qx * ty - qy * tx)
    return (rx, ry, rz)


@always_inline
fn apply_joint_torques_gpu[
    DTYPE: DType,
    NUM_BODIES: Int,
    MAX_CONTACTS: Int,
    MAX_JOINTS: Int,
    MAX_SLIDE_JOINTS: Int,
    STATE_SIZE: Int,
    BATCH: Int,
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[
        DTYPE, Layout.row_major(NUM_BODIES, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    dt: Scalar[DTYPE],
):
    """Apply actuator torques to angular velocities on GPU.

    This should be called early in the physics step, before constraint solving.
    """
    for j in range(MAX_JOINTS):
        var j_off = joint_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS](j)

        # Get torque and limit from state buffer
        var torque = rebind[Scalar[DTYPE]](
            state[env, j_off + JOINT_IDX_TARGET_TORQUE]
        )
        var torque_limit = rebind[Scalar[DTYPE]](
            state[env, j_off + JOINT_IDX_TORQUE_LIMIT]
        )

        # Clamp torque to limits
        if torque > torque_limit:
            torque = torque_limit
        elif torque < -torque_limit:
            torque = -torque_limit

        # Skip if no torque
        if torque * torque < Scalar[DTYPE](1e-12):
            continue

        # Read body indices from state buffer and convert using Int()
        # Our GPU debug test confirmed Int(f) works for exact integers
        var parent_f = rebind[Scalar[DTYPE]](
            state[env, j_off + JOINT_IDX_PARENT]
        )
        var child_f = rebind[Scalar[DTYPE]](state[env, j_off + JOINT_IDX_CHILD])
        var body_a: Int = Int(parent_f)
        var body_b: Int = Int(child_f)

        # Get axis from joint state (in parent/world frame)
        var axis_x = rebind[Scalar[DTYPE]](state[env, j_off + JOINT_IDX_AXIS_X])
        var axis_y = rebind[Scalar[DTYPE]](state[env, j_off + JOINT_IDX_AXIS_Y])
        var axis_z = rebind[Scalar[DTYPE]](state[env, j_off + JOINT_IDX_AXIS_Z])

        # If parent is a body, rotate axis to world frame
        if body_a >= 0:
            var b_off_a = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS](
                body_a
            )
            var qa_x = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_QX])
            var qa_y = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_QY])
            var qa_z = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_QZ])
            var qa_w = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_QW])
            var rot_axis = _quat_rotate_gpu(
                qa_x, qa_y, qa_z, qa_w, axis_x, axis_y, axis_z
            )
            axis_x = rot_axis[0]
            axis_y = rot_axis[1]
            axis_z = rot_axis[2]

        # Apply torque to child body
        var b_off_b = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS](body_b)
        var avg_inv_i_b = (
            rebind[Scalar[DTYPE]](model[body_b, MODEL_IDX_INV_IXX])
            + rebind[Scalar[DTYPE]](model[body_b, MODEL_IDX_INV_IYY])
            + rebind[Scalar[DTYPE]](model[body_b, MODEL_IDX_INV_IZZ])
        ) / Scalar[DTYPE](3.0)

        var delta_w = torque * avg_inv_i_b * dt
        var wb_x = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_WX])
        var wb_y = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_WY])
        var wb_z = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_WZ])

        state[env, b_off_b + BODY_IDX_WX] = wb_x + delta_w * axis_x
        state[env, b_off_b + BODY_IDX_WY] = wb_y + delta_w * axis_y
        state[env, b_off_b + BODY_IDX_WZ] = wb_z + delta_w * axis_z

        # Apply reaction torque to parent (if not world-anchored)
        if body_a >= 0:
            var b_off_a = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS](
                body_a
            )
            var avg_inv_i_a = (
                rebind[Scalar[DTYPE]](model[body_a, MODEL_IDX_INV_IXX])
                + rebind[Scalar[DTYPE]](model[body_a, MODEL_IDX_INV_IYY])
                + rebind[Scalar[DTYPE]](model[body_a, MODEL_IDX_INV_IZZ])
            ) / Scalar[DTYPE](3.0)

            var delta_w_a = torque * avg_inv_i_a * dt
            var wa_x = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_WX])
            var wa_y = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_WY])
            var wa_z = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_WZ])

            state[env, b_off_a + BODY_IDX_WX] = wa_x - delta_w_a * axis_x
            state[env, b_off_a + BODY_IDX_WY] = wa_y - delta_w_a * axis_y
            state[env, b_off_a + BODY_IDX_WZ] = wa_z - delta_w_a * axis_z


@always_inline
fn solve_joint_velocity_constraints_gpu[
    DTYPE: DType,
    NUM_BODIES: Int,
    MAX_CONTACTS: Int,
    MAX_JOINTS: Int,
    MAX_SLIDE_JOINTS: Int,
    STATE_SIZE: Int,
    BATCH: Int,
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[
        DTYPE, Layout.row_major(NUM_BODIES, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    iterations: Int,
):
    """Solve joint velocity constraints on GPU."""
    # The solver will iterate over all MAX_JOINTS slots; invalid joints should be skipped

    for _ in range(iterations):
        for j in range(MAX_JOINTS):
            var j_off = joint_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS](j)

            # Read actual body indices from joint state buffer
            # IMPORTANT: Must use rebind to get the actual Scalar value from LayoutTensor
            var parent_f = rebind[Scalar[DTYPE]](
                state[env, j_off + JOINT_IDX_PARENT]
            )
            var child_f = rebind[Scalar[DTYPE]](
                state[env, j_off + JOINT_IDX_CHILD]
            )

            # Convert to Int - the values are stored as floats for GPU compatibility
            var body_a: Int = Int(parent_f)
            var body_b: Int = Int(child_f)

            # Skip if child body is invalid (indicates uninitialized joint slot)
            if body_b < 0 or body_b >= NUM_BODIES:
                continue

            # Skip free DOF joints (Phase 11f) - they don't apply constraints
            var is_free_dof = rebind[Scalar[DTYPE]](
                state[env, j_off + JOINT_IDX_IS_FREE_DOF]
            )
            if is_free_dof > Scalar[DTYPE](0.5):
                continue

            # Get anchor points from joint state
            var anchor_px = rebind[Scalar[DTYPE]](
                state[env, j_off + JOINT_IDX_ANCHOR_PX]
            )
            var anchor_py = rebind[Scalar[DTYPE]](
                state[env, j_off + JOINT_IDX_ANCHOR_PY]
            )
            var anchor_pz = rebind[Scalar[DTYPE]](
                state[env, j_off + JOINT_IDX_ANCHOR_PZ]
            )
            var anchor_cx = rebind[Scalar[DTYPE]](
                state[env, j_off + JOINT_IDX_ANCHOR_CX]
            )
            var anchor_cy = rebind[Scalar[DTYPE]](
                state[env, j_off + JOINT_IDX_ANCHOR_CY]
            )
            var anchor_cz = rebind[Scalar[DTYPE]](
                state[env, j_off + JOINT_IDX_ANCHOR_CZ]
            )

            # Get world-space anchor for parent
            var wa_x = anchor_px
            var wa_y = anchor_py
            var wa_z = anchor_pz
            var pa_x: Scalar[DTYPE] = 0
            var pa_y: Scalar[DTYPE] = 0
            var pa_z: Scalar[DTYPE] = 0
            var va_x: Scalar[DTYPE] = 0
            var va_y: Scalar[DTYPE] = 0
            var va_z: Scalar[DTYPE] = 0
            var wa_wx: Scalar[DTYPE] = 0
            var wa_wy: Scalar[DTYPE] = 0
            var wa_wz: Scalar[DTYPE] = 0

            if body_a >= 0:
                var b_off_a = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS](
                    body_a
                )
                pa_x = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_PX])
                pa_y = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_PY])
                pa_z = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_PZ])
                var qa_x = rebind[Scalar[DTYPE]](
                    state[env, b_off_a + BODY_IDX_QX]
                )
                var qa_y = rebind[Scalar[DTYPE]](
                    state[env, b_off_a + BODY_IDX_QY]
                )
                var qa_z = rebind[Scalar[DTYPE]](
                    state[env, b_off_a + BODY_IDX_QZ]
                )
                var qa_w = rebind[Scalar[DTYPE]](
                    state[env, b_off_a + BODY_IDX_QW]
                )
                va_x = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_VX])
                va_y = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_VY])
                va_z = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_VZ])
                wa_wx = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_WX])
                wa_wy = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_WY])
                wa_wz = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_WZ])
                var rot_a = _quat_rotate_gpu(
                    qa_x, qa_y, qa_z, qa_w, anchor_px, anchor_py, anchor_pz
                )
                wa_x = pa_x + rot_a[0]
                wa_y = pa_y + rot_a[1]
                wa_z = pa_z + rot_a[2]

            # Get world-space anchor for child
            var b_off_b = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS](
                body_b
            )
            var pb_x = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_PX])
            var pb_y = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_PY])
            var pb_z = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_PZ])
            var qb_x = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_QX])
            var qb_y = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_QY])
            var qb_z = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_QZ])
            var qb_w = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_QW])
            var vb_x = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_VX])
            var vb_y = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_VY])
            var vb_z = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_VZ])
            var wb_x = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_WX])
            var wb_y = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_WY])
            var wb_z = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_WZ])

            var rot_b = _quat_rotate_gpu(
                qb_x, qb_y, qb_z, qb_w, anchor_cx, anchor_cy, anchor_cz
            )
            var wb_ax = pb_x + rot_b[0]
            var wb_ay = pb_y + rot_b[1]
            var wb_az = pb_z + rot_b[2]

            # Lever arms
            var ra_x = wa_x - pa_x
            var ra_y = wa_y - pa_y
            var ra_z = wa_z - pa_z
            var rb_x = wb_ax - pb_x
            var rb_y = wb_ay - pb_y
            var rb_z = wb_az - pb_z

            # Velocity at anchor A
            var vel_a_x = va_x + (wa_wy * ra_z - wa_wz * ra_y)
            var vel_a_y = va_y + (wa_wz * ra_x - wa_wx * ra_z)
            var vel_a_z = va_z + (wa_wx * ra_y - wa_wy * ra_x)

            # Velocity at anchor B
            var vel_b_x = vb_x + (wb_y * rb_z - wb_z * rb_y)
            var vel_b_y = vb_y + (wb_z * rb_x - wb_x * rb_z)
            var vel_b_z = vb_z + (wb_x * rb_y - wb_y * rb_x)

            # Velocity error
            var dv_x = vel_a_x - vel_b_x
            var dv_y = vel_a_y - vel_b_y
            var dv_z = vel_a_z - vel_b_z

            # Compute effective mass with rotational contribution
            var inv_mass_a: Scalar[DTYPE] = 0
            var rot_contrib_a: Scalar[DTYPE] = 0
            if body_a >= 0:
                inv_mass_a = rebind[Scalar[DTYPE]](
                    model[body_a, MODEL_IDX_INV_MASS]
                )
                var ra_sq = ra_x * ra_x + ra_y * ra_y + ra_z * ra_z
                var avg_inv_i_a = (
                    rebind[Scalar[DTYPE]](model[body_a, MODEL_IDX_INV_IXX])
                    + rebind[Scalar[DTYPE]](model[body_a, MODEL_IDX_INV_IYY])
                    + rebind[Scalar[DTYPE]](model[body_a, MODEL_IDX_INV_IZZ])
                ) / Scalar[DTYPE](3.0)
                rot_contrib_a = ra_sq * avg_inv_i_a

            var inv_mass_b = rebind[Scalar[DTYPE]](
                model[body_b, MODEL_IDX_INV_MASS]
            )
            var rb_sq = rb_x * rb_x + rb_y * rb_y + rb_z * rb_z
            var avg_inv_i_b = (
                rebind[Scalar[DTYPE]](model[body_b, MODEL_IDX_INV_IXX])
                + rebind[Scalar[DTYPE]](model[body_b, MODEL_IDX_INV_IYY])
                + rebind[Scalar[DTYPE]](model[body_b, MODEL_IDX_INV_IZZ])
            ) / Scalar[DTYPE](3.0)
            var rot_contrib_b = rb_sq * avg_inv_i_b

            var K = inv_mass_a + inv_mass_b + rot_contrib_a + rot_contrib_b
            if K < Scalar[DTYPE](1e-10):
                continue

            # Impulse with relaxation
            var relaxation = Scalar[DTYPE](0.8)
            var impulse_x = -relaxation * dv_x / K
            var impulse_y = -relaxation * dv_y / K
            var impulse_z = -relaxation * dv_z / K

            # Apply to body A
            if body_a >= 0:
                var b_off_a = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS](
                    body_a
                )
                state[env, b_off_a + BODY_IDX_VX] = (
                    va_x + impulse_x * inv_mass_a
                )
                state[env, b_off_a + BODY_IDX_VY] = (
                    va_y + impulse_y * inv_mass_a
                )
                state[env, b_off_a + BODY_IDX_VZ] = (
                    va_z + impulse_z * inv_mass_a
                )

                var tau_a_x = ra_y * impulse_z - ra_z * impulse_y
                var tau_a_y = ra_z * impulse_x - ra_x * impulse_z
                var tau_a_z = ra_x * impulse_y - ra_y * impulse_x
                var inv_ixx_a = rebind[Scalar[DTYPE]](
                    model[body_a, MODEL_IDX_INV_IXX]
                )
                var inv_iyy_a = rebind[Scalar[DTYPE]](
                    model[body_a, MODEL_IDX_INV_IYY]
                )
                var inv_izz_a = rebind[Scalar[DTYPE]](
                    model[body_a, MODEL_IDX_INV_IZZ]
                )
                state[env, b_off_a + BODY_IDX_WX] = wa_wx + tau_a_x * inv_ixx_a
                state[env, b_off_a + BODY_IDX_WY] = wa_wy + tau_a_y * inv_iyy_a
                state[env, b_off_a + BODY_IDX_WZ] = wa_wz + tau_a_z * inv_izz_a

            # Apply to body B
            state[env, b_off_b + BODY_IDX_VX] = vb_x - impulse_x * inv_mass_b
            state[env, b_off_b + BODY_IDX_VY] = vb_y - impulse_y * inv_mass_b
            state[env, b_off_b + BODY_IDX_VZ] = vb_z - impulse_z * inv_mass_b

            var tau_b_x = rb_y * impulse_z - rb_z * impulse_y
            var tau_b_y = rb_z * impulse_x - rb_x * impulse_z
            var tau_b_z = rb_x * impulse_y - rb_y * impulse_x
            var inv_ixx_b = rebind[Scalar[DTYPE]](
                model[body_b, MODEL_IDX_INV_IXX]
            )
            var inv_iyy_b = rebind[Scalar[DTYPE]](
                model[body_b, MODEL_IDX_INV_IYY]
            )
            var inv_izz_b = rebind[Scalar[DTYPE]](
                model[body_b, MODEL_IDX_INV_IZZ]
            )
            state[env, b_off_b + BODY_IDX_WX] = wb_x - tau_b_x * inv_ixx_b
            state[env, b_off_b + BODY_IDX_WY] = wb_y - tau_b_y * inv_iyy_b
            state[env, b_off_b + BODY_IDX_WZ] = wb_z - tau_b_z * inv_izz_b


@always_inline
fn solve_joint_position_constraints_gpu[
    DTYPE: DType,
    NUM_BODIES: Int,
    MAX_CONTACTS: Int,
    MAX_JOINTS: Int,
    MAX_SLIDE_JOINTS: Int,
    STATE_SIZE: Int,
    BATCH: Int,
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[
        DTYPE, Layout.row_major(NUM_BODIES, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    baumgarte: Scalar[DTYPE],
    iterations: Int,
):
    """Solve joint position constraints on GPU using Baumgarte stabilization."""

    for _ in range(iterations):
        for j in range(MAX_JOINTS):
            var j_off = joint_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS](j)

            # Read actual body indices from joint state buffer
            # IMPORTANT: Must use rebind to get the actual Scalar value from LayoutTensor
            var parent_f = rebind[Scalar[DTYPE]](
                state[env, j_off + JOINT_IDX_PARENT]
            )
            var child_f = rebind[Scalar[DTYPE]](
                state[env, j_off + JOINT_IDX_CHILD]
            )

            # Convert to Int - the values are stored as floats for GPU compatibility
            var body_a: Int = Int(parent_f)
            var body_b: Int = Int(child_f)
            # Skip if child body is invalid (indicates uninitialized joint slot)
            if body_b < 0 or body_b >= NUM_BODIES:
                continue

            # Skip free DOF joints (Phase 11f) - they don't apply constraints
            var is_free_dof = rebind[Scalar[DTYPE]](
                state[env, j_off + JOINT_IDX_IS_FREE_DOF]
            )
            if is_free_dof > Scalar[DTYPE](0.5):
                continue

            # Get anchor points
            var anchor_px = rebind[Scalar[DTYPE]](
                state[env, j_off + JOINT_IDX_ANCHOR_PX]
            )
            var anchor_py = rebind[Scalar[DTYPE]](
                state[env, j_off + JOINT_IDX_ANCHOR_PY]
            )
            var anchor_pz = rebind[Scalar[DTYPE]](
                state[env, j_off + JOINT_IDX_ANCHOR_PZ]
            )
            var anchor_cx = rebind[Scalar[DTYPE]](
                state[env, j_off + JOINT_IDX_ANCHOR_CX]
            )
            var anchor_cy = rebind[Scalar[DTYPE]](
                state[env, j_off + JOINT_IDX_ANCHOR_CY]
            )
            var anchor_cz = rebind[Scalar[DTYPE]](
                state[env, j_off + JOINT_IDX_ANCHOR_CZ]
            )

            # Get world-space anchor for parent
            var wa_x = anchor_px
            var wa_y = anchor_py
            var wa_z = anchor_pz

            if body_a >= 0:
                var b_off_a = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS](
                    body_a
                )
                var pa_x = rebind[Scalar[DTYPE]](
                    state[env, b_off_a + BODY_IDX_PX]
                )
                var pa_y = rebind[Scalar[DTYPE]](
                    state[env, b_off_a + BODY_IDX_PY]
                )
                var pa_z = rebind[Scalar[DTYPE]](
                    state[env, b_off_a + BODY_IDX_PZ]
                )
                var qa_x = rebind[Scalar[DTYPE]](
                    state[env, b_off_a + BODY_IDX_QX]
                )
                var qa_y = rebind[Scalar[DTYPE]](
                    state[env, b_off_a + BODY_IDX_QY]
                )
                var qa_z = rebind[Scalar[DTYPE]](
                    state[env, b_off_a + BODY_IDX_QZ]
                )
                var qa_w = rebind[Scalar[DTYPE]](
                    state[env, b_off_a + BODY_IDX_QW]
                )
                var rot_a = _quat_rotate_gpu(
                    qa_x, qa_y, qa_z, qa_w, anchor_px, anchor_py, anchor_pz
                )
                wa_x = pa_x + rot_a[0]
                wa_y = pa_y + rot_a[1]
                wa_z = pa_z + rot_a[2]

            # Get world-space anchor for child
            var b_off_b = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS](
                body_b
            )
            var pb_x = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_PX])
            var pb_y = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_PY])
            var pb_z = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_PZ])
            var qb_x = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_QX])
            var qb_y = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_QY])
            var qb_z = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_QZ])
            var qb_w = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_QW])

            var rot_b = _quat_rotate_gpu(
                qb_x, qb_y, qb_z, qb_w, anchor_cx, anchor_cy, anchor_cz
            )
            var wb_x = pb_x + rot_b[0]
            var wb_y = pb_y + rot_b[1]
            var wb_z = pb_z + rot_b[2]

            # Position error
            var error_x = wa_x - wb_x
            var error_y = wa_y - wb_y
            var error_z = wa_z - wb_z

            var error_sq = (
                error_x * error_x + error_y * error_y + error_z * error_z
            )
            if error_sq < Scalar[DTYPE](1e-12):
                continue

            # Compute effective mass
            var inv_mass_a: Scalar[DTYPE] = 0
            if body_a >= 0:
                inv_mass_a = rebind[Scalar[DTYPE]](
                    model[body_a, MODEL_IDX_INV_MASS]
                )
            var inv_mass_b = rebind[Scalar[DTYPE]](
                model[body_b, MODEL_IDX_INV_MASS]
            )

            var total_inv_mass = inv_mass_a + inv_mass_b
            if total_inv_mass < Scalar[DTYPE](1e-10):
                continue

            # Position correction
            var correction_x = -baumgarte * error_x / total_inv_mass
            var correction_y = -baumgarte * error_y / total_inv_mass
            var correction_z = -baumgarte * error_z / total_inv_mass

            # Apply to body A
            if body_a >= 0:
                var b_off_a = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS](
                    body_a
                )
                var pa_x = rebind[Scalar[DTYPE]](
                    state[env, b_off_a + BODY_IDX_PX]
                )
                var pa_y = rebind[Scalar[DTYPE]](
                    state[env, b_off_a + BODY_IDX_PY]
                )
                var pa_z = rebind[Scalar[DTYPE]](
                    state[env, b_off_a + BODY_IDX_PZ]
                )
                state[env, b_off_a + BODY_IDX_PX] = (
                    pa_x + correction_x * inv_mass_a
                )
                state[env, b_off_a + BODY_IDX_PY] = (
                    pa_y + correction_y * inv_mass_a
                )
                state[env, b_off_a + BODY_IDX_PZ] = (
                    pa_z + correction_z * inv_mass_a
                )

            # Apply to body B
            state[env, b_off_b + BODY_IDX_PX] = pb_x - correction_x * inv_mass_b
            state[env, b_off_b + BODY_IDX_PY] = pb_y - correction_y * inv_mass_b
            state[env, b_off_b + BODY_IDX_PZ] = pb_z - correction_z * inv_mass_b


# =============================================================================
# GPU Slide Joint Solvers
# =============================================================================


@always_inline
fn solve_slide_joint_velocity_constraints_gpu[
    DTYPE: DType,
    NUM_BODIES: Int,
    MAX_CONTACTS: Int,
    MAX_JOINTS: Int,
    MAX_SLIDE_JOINTS: Int,
    STATE_SIZE: Int,
    BATCH: Int,
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[
        DTYPE, Layout.row_major(NUM_BODIES, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    iterations: Int,
):
    """Solve slide joint velocity constraints on GPU.

    Constrains:
    1. Velocity perpendicular to slide axis (2 DOF)
    2. All angular velocities (3 DOF)
    """
    for _ in range(iterations):
        for sj in range(MAX_SLIDE_JOINTS):
            var sj_off = slide_joint_offset[
                NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS
            ](sj)

            # Read body indices
            var parent_f = rebind[Scalar[DTYPE]](
                state[env, sj_off + SLIDE_IDX_PARENT]
            )
            var child_f = rebind[Scalar[DTYPE]](
                state[env, sj_off + SLIDE_IDX_CHILD]
            )
            var body_a = Int(parent_f)
            var body_b = Int(child_f)

            # Skip invalid joint slots
            if body_b < 0 or body_b >= NUM_BODIES:
                continue

            # Skip free DOF joints (Phase 11f) - they don't apply constraints
            var is_free_dof = rebind[Scalar[DTYPE]](
                state[env, sj_off + SLIDE_IDX_IS_FREE_DOF]
            )
            if is_free_dof > Scalar[DTYPE](0.5):
                continue

            # Get slide axis from joint state
            var axis_x = rebind[Scalar[DTYPE]](
                state[env, sj_off + SLIDE_IDX_AXIS_X]
            )
            var axis_y = rebind[Scalar[DTYPE]](
                state[env, sj_off + SLIDE_IDX_AXIS_Y]
            )
            var axis_z = rebind[Scalar[DTYPE]](
                state[env, sj_off + SLIDE_IDX_AXIS_Z]
            )

            # Compute perpendicular basis (inline for GPU)
            var abs_ax = abs(axis_x)
            var abs_ay = abs(axis_y)
            var abs_az = abs(axis_z)
            var ref_x: Scalar[DTYPE]
            var ref_y: Scalar[DTYPE]
            var ref_z: Scalar[DTYPE]

            if abs_ax < abs_ay and abs_ax < abs_az:
                ref_x = Scalar[DTYPE](1.0)
                ref_y = Scalar[DTYPE](0.0)
                ref_z = Scalar[DTYPE](0.0)
            elif abs_ay < abs_az:
                ref_x = Scalar[DTYPE](0.0)
                ref_y = Scalar[DTYPE](1.0)
                ref_z = Scalar[DTYPE](0.0)
            else:
                ref_x = Scalar[DTYPE](0.0)
                ref_y = Scalar[DTYPE](0.0)
                ref_z = Scalar[DTYPE](1.0)

            # t1 = normalize(ref - (ref·axis)*axis)
            var dot = ref_x * axis_x + ref_y * axis_y + ref_z * axis_z
            var t1_x = ref_x - dot * axis_x
            var t1_y = ref_y - dot * axis_y
            var t1_z = ref_z - dot * axis_z
            var t1_len = sqrt(t1_x * t1_x + t1_y * t1_y + t1_z * t1_z)
            if t1_len > Scalar[DTYPE](1e-10):
                t1_x = t1_x / t1_len
                t1_y = t1_y / t1_len
                t1_z = t1_z / t1_len

            # t2 = axis x t1
            var t2_x = axis_y * t1_z - axis_z * t1_y
            var t2_y = axis_z * t1_x - axis_x * t1_z
            var t2_z = axis_x * t1_y - axis_y * t1_x

            # Get velocities and positions
            var va_x: Scalar[DTYPE] = 0
            var va_y: Scalar[DTYPE] = 0
            var va_z: Scalar[DTYPE] = 0
            var wa_x: Scalar[DTYPE] = 0
            var wa_y: Scalar[DTYPE] = 0
            var wa_z: Scalar[DTYPE] = 0
            var b_off_a: Int = 0
            var inv_mass_a: Scalar[DTYPE] = 0

            if body_a >= 0:
                b_off_a = body_offset[
                    NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS
                ](body_a)
                va_x = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_VX])
                va_y = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_VY])
                va_z = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_VZ])
                wa_x = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_WX])
                wa_y = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_WY])
                wa_z = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_WZ])
                inv_mass_a = rebind[Scalar[DTYPE]](
                    model[body_a, MODEL_IDX_INV_MASS]
                )

            var b_off_b = body_offset[
                NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS
            ](body_b)
            var vb_x = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_VX])
            var vb_y = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_VY])
            var vb_z = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_VZ])
            var wb_x = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_WX])
            var wb_y = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_WY])
            var wb_z = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_WZ])
            var inv_mass_b = rebind[Scalar[DTYPE]](
                model[body_b, MODEL_IDX_INV_MASS]
            )

            # --- Perpendicular velocity constraint ---
            # Relative velocity
            var dv_x = va_x - vb_x
            var dv_y = va_y - vb_y
            var dv_z = va_z - vb_z

            # Project onto perpendicular directions
            var dv_t1 = dv_x * t1_x + dv_y * t1_y + dv_z * t1_z
            var dv_t2 = dv_x * t2_x + dv_y * t2_y + dv_z * t2_z

            var K = inv_mass_a + inv_mass_b
            if K < Scalar[DTYPE](1e-10):
                continue

            # Compute impulses to zero perpendicular velocity
            var relaxation: Scalar[DTYPE] = 0.8
            var impulse_t1 = -relaxation * dv_t1 / K
            var impulse_t2 = -relaxation * dv_t2 / K

            # Convert to world-space impulse
            var impulse_x = impulse_t1 * t1_x + impulse_t2 * t2_x
            var impulse_y = impulse_t1 * t1_y + impulse_t2 * t2_y
            var impulse_z = impulse_t1 * t1_z + impulse_t2 * t2_z

            # Apply linear impulse
            if body_a >= 0:
                state[env, b_off_a + BODY_IDX_VX] = va_x + impulse_x * inv_mass_a
                state[env, b_off_a + BODY_IDX_VY] = va_y + impulse_y * inv_mass_a
                state[env, b_off_a + BODY_IDX_VZ] = va_z + impulse_z * inv_mass_a

            state[env, b_off_b + BODY_IDX_VX] = vb_x - impulse_x * inv_mass_b
            state[env, b_off_b + BODY_IDX_VY] = vb_y - impulse_y * inv_mass_b
            state[env, b_off_b + BODY_IDX_VZ] = vb_z - impulse_z * inv_mass_b

            # --- Angular velocity constraint (lock all rotation) ---
            # Relative angular velocity
            var dw_x = wa_x - wb_x
            var dw_y = wa_y - wb_y
            var dw_z = wa_z - wb_z

            # Simple damping to zero relative angular velocity
            var ang_relaxation: Scalar[DTYPE] = 0.5
            if body_a >= 0:
                state[env, b_off_a + BODY_IDX_WX] = wa_x - ang_relaxation * dw_x
                state[env, b_off_a + BODY_IDX_WY] = wa_y - ang_relaxation * dw_y
                state[env, b_off_a + BODY_IDX_WZ] = wa_z - ang_relaxation * dw_z
            else:
                # Parent is world: child angular velocity should be zero
                state[env, b_off_b + BODY_IDX_WX] = wb_x * (
                    Scalar[DTYPE](1.0) - ang_relaxation
                )
                state[env, b_off_b + BODY_IDX_WY] = wb_y * (
                    Scalar[DTYPE](1.0) - ang_relaxation
                )
                state[env, b_off_b + BODY_IDX_WZ] = wb_z * (
                    Scalar[DTYPE](1.0) - ang_relaxation
                )


@always_inline
fn solve_slide_joint_position_constraints_gpu[
    DTYPE: DType,
    NUM_BODIES: Int,
    MAX_CONTACTS: Int,
    MAX_JOINTS: Int,
    MAX_SLIDE_JOINTS: Int,
    STATE_SIZE: Int,
    BATCH: Int,
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[
        DTYPE, Layout.row_major(NUM_BODIES, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    baumgarte: Scalar[DTYPE],
    iterations: Int,
):
    """Solve slide joint position constraints on GPU.

    Corrects:
    1. Position perpendicular to slide axis (2 DOF)
    """
    for _ in range(iterations):
        for sj in range(MAX_SLIDE_JOINTS):
            var sj_off = slide_joint_offset[
                NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS
            ](sj)

            # Read body indices
            var parent_f = rebind[Scalar[DTYPE]](
                state[env, sj_off + SLIDE_IDX_PARENT]
            )
            var child_f = rebind[Scalar[DTYPE]](
                state[env, sj_off + SLIDE_IDX_CHILD]
            )
            var body_a = Int(parent_f)
            var body_b = Int(child_f)

            # Skip invalid joint slots
            if body_b < 0 or body_b >= NUM_BODIES:
                continue

            # Skip free DOF joints (Phase 11f) - they don't apply constraints
            var is_free_dof = rebind[Scalar[DTYPE]](
                state[env, sj_off + SLIDE_IDX_IS_FREE_DOF]
            )
            if is_free_dof > Scalar[DTYPE](0.5):
                continue

            # Get anchor points
            var anchor_px = rebind[Scalar[DTYPE]](
                state[env, sj_off + SLIDE_IDX_ANCHOR_PX]
            )
            var anchor_py = rebind[Scalar[DTYPE]](
                state[env, sj_off + SLIDE_IDX_ANCHOR_PY]
            )
            var anchor_pz = rebind[Scalar[DTYPE]](
                state[env, sj_off + SLIDE_IDX_ANCHOR_PZ]
            )
            var anchor_cx = rebind[Scalar[DTYPE]](
                state[env, sj_off + SLIDE_IDX_ANCHOR_CX]
            )
            var anchor_cy = rebind[Scalar[DTYPE]](
                state[env, sj_off + SLIDE_IDX_ANCHOR_CY]
            )
            var anchor_cz = rebind[Scalar[DTYPE]](
                state[env, sj_off + SLIDE_IDX_ANCHOR_CZ]
            )

            # Get slide axis
            var axis_x = rebind[Scalar[DTYPE]](
                state[env, sj_off + SLIDE_IDX_AXIS_X]
            )
            var axis_y = rebind[Scalar[DTYPE]](
                state[env, sj_off + SLIDE_IDX_AXIS_Y]
            )
            var axis_z = rebind[Scalar[DTYPE]](
                state[env, sj_off + SLIDE_IDX_AXIS_Z]
            )

            # Get child body position
            var b_off_b = body_offset[
                NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS
            ](body_b)
            var pb_x = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_PX])
            var pb_y = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_PY])
            var pb_z = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_PZ])

            # Compute world-space anchor for child (assuming identity rotation for simplicity)
            var wa_b_x = pb_x + anchor_cx
            var wa_b_y = pb_y + anchor_cy
            var wa_b_z = pb_z + anchor_cz

            # World-space anchor for parent
            var wa_a_x = anchor_px
            var wa_a_y = anchor_py
            var wa_a_z = anchor_pz
            var inv_mass_a: Scalar[DTYPE] = 0
            var b_off_a: Int = 0

            if body_a >= 0:
                b_off_a = body_offset[
                    NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, MAX_SLIDE_JOINTS
                ](body_a)
                var pa_x = rebind[Scalar[DTYPE]](
                    state[env, b_off_a + BODY_IDX_PX]
                )
                var pa_y = rebind[Scalar[DTYPE]](
                    state[env, b_off_a + BODY_IDX_PY]
                )
                var pa_z = rebind[Scalar[DTYPE]](
                    state[env, b_off_a + BODY_IDX_PZ]
                )
                wa_a_x = pa_x + anchor_px
                wa_a_y = pa_y + anchor_py
                wa_a_z = pa_z + anchor_pz
                inv_mass_a = rebind[Scalar[DTYPE]](
                    model[body_a, MODEL_IDX_INV_MASS]
                )

            var inv_mass_b = rebind[Scalar[DTYPE]](
                model[body_b, MODEL_IDX_INV_MASS]
            )

            # Position error (anchors should be collinear along axis)
            var error_x = wa_b_x - wa_a_x
            var error_y = wa_b_y - wa_a_y
            var error_z = wa_b_z - wa_a_z

            # Project error along slide axis (this component is allowed)
            var error_along_axis = (
                error_x * axis_x + error_y * axis_y + error_z * axis_z
            )

            # Perpendicular error (should be zero)
            var perp_error_x = error_x - error_along_axis * axis_x
            var perp_error_y = error_y - error_along_axis * axis_y
            var perp_error_z = error_z - error_along_axis * axis_z

            var error_sq = (
                perp_error_x * perp_error_x
                + perp_error_y * perp_error_y
                + perp_error_z * perp_error_z
            )
            if error_sq < Scalar[DTYPE](1e-12):
                continue

            var total_inv_mass = inv_mass_a + inv_mass_b
            if total_inv_mass < Scalar[DTYPE](1e-10):
                continue

            # Position correction
            var correction_x = -baumgarte * perp_error_x / total_inv_mass
            var correction_y = -baumgarte * perp_error_y / total_inv_mass
            var correction_z = -baumgarte * perp_error_z / total_inv_mass

            # Apply corrections
            if body_a >= 0:
                var pa_x = rebind[Scalar[DTYPE]](
                    state[env, b_off_a + BODY_IDX_PX]
                )
                var pa_y = rebind[Scalar[DTYPE]](
                    state[env, b_off_a + BODY_IDX_PY]
                )
                var pa_z = rebind[Scalar[DTYPE]](
                    state[env, b_off_a + BODY_IDX_PZ]
                )
                state[env, b_off_a + BODY_IDX_PX] = (
                    pa_x + correction_x * inv_mass_a
                )
                state[env, b_off_a + BODY_IDX_PY] = (
                    pa_y + correction_y * inv_mass_a
                )
                state[env, b_off_a + BODY_IDX_PZ] = (
                    pa_z + correction_z * inv_mass_a
                )

            state[env, b_off_b + BODY_IDX_PX] = pb_x - correction_x * inv_mass_b
            state[env, b_off_b + BODY_IDX_PY] = pb_y - correction_y * inv_mass_b
            state[env, b_off_b + BODY_IDX_PZ] = pb_z - correction_z * inv_mass_b

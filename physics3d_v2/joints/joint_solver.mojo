"""Joint Constraint Solver for Physics3D v2.

Implements velocity and position constraint solving for hinge joints.

Physics:
- Hinge joint constrains 5 DOF (3 linear + 2 angular)
- Position constraint: Anchor points must coincide
- Angular constraint: Rotation only around hinge axis

Reference: Adapted from physics3d/solvers/joint_solver3d.mojo
"""

from math import sqrt
from ..types import Model, Data
from .hinge_joint import HingeJoint


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
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int
](
    data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS],
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
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int
](
    data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS],
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
    var length_sq = rotated[0] * rotated[0] + rotated[1] * rotated[1] + rotated[2] * rotated[2]
    var inv_length = Scalar[DTYPE](1.0) / sqrt(length_sq + Scalar[DTYPE](1e-10))

    return (rotated[0] * inv_length, rotated[1] * inv_length, rotated[2] * inv_length)


# =============================================================================
# Velocity Constraint Solving
# =============================================================================


fn _solve_single_joint_velocity[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int
](
    model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS],
    mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS],
    joint_idx: Int,
):
    """Solve velocity constraints for a single hinge joint.

    Constrains:
    1. Anchor points to have same velocity (point-to-point)
    2. Angular velocities to differ only around hinge axis
    """
    var joint = model.joints[joint_idx]
    var body_a = joint.parent_body
    var body_b = joint.child_body

    # Get world-space anchors
    var anchor_a = _get_world_anchor(
        data, body_a,
        joint.anchor_parent_x, joint.anchor_parent_y, joint.anchor_parent_z
    )
    var anchor_b = _get_world_anchor(
        data, body_b,
        joint.anchor_child_x, joint.anchor_child_y, joint.anchor_child_z
    )

    # Get world-space hinge axis
    var axis = _get_world_axis(
        data, body_a,
        joint.axis_x, joint.axis_y, joint.axis_z
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
        data.angular_velocities[body_a * 3 + 0] += tau_a_x * model.inv_inertias[body_a * 3 + 0]
        data.angular_velocities[body_a * 3 + 1] += tau_a_y * model.inv_inertias[body_a * 3 + 1]
        data.angular_velocities[body_a * 3 + 2] += tau_a_z * model.inv_inertias[body_a * 3 + 2]

    data.velocities[body_b * 3 + 0] -= impulse_x * inv_mass_b
    data.velocities[body_b * 3 + 1] -= impulse_y * inv_mass_b
    data.velocities[body_b * 3 + 2] -= impulse_z * inv_mass_b

    var tau_b_x = rb_y * impulse_z - rb_z * impulse_y
    var tau_b_y = rb_z * impulse_x - rb_x * impulse_z
    var tau_b_z = rb_x * impulse_y - rb_y * impulse_x
    data.angular_velocities[body_b * 3 + 0] -= tau_b_x * model.inv_inertias[body_b * 3 + 0]
    data.angular_velocities[body_b * 3 + 1] -= tau_b_y * model.inv_inertias[body_b * 3 + 1]
    data.angular_velocities[body_b * 3 + 2] -= tau_b_z * model.inv_inertias[body_b * 3 + 2]

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
    var omega_dot_axis = rel_omega_x * axis[0] + rel_omega_y * axis[1] + rel_omega_z * axis[2]
    var omega_along_x = axis[0] * omega_dot_axis
    var omega_along_y = axis[1] * omega_dot_axis
    var omega_along_z = axis[2] * omega_dot_axis

    # Component perpendicular to axis (should be zero)
    var omega_perp_x = rel_omega_x - omega_along_x
    var omega_perp_y = rel_omega_y - omega_along_y
    var omega_perp_z = rel_omega_z - omega_along_z

    var omega_perp_sq = omega_perp_x * omega_perp_x + omega_perp_y * omega_perp_y + omega_perp_z * omega_perp_z
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
        data.angular_velocities[body_a * 3 + 0] += ang_impulse_x * model.inv_inertias[body_a * 3 + 0]
        data.angular_velocities[body_a * 3 + 1] += ang_impulse_y * model.inv_inertias[body_a * 3 + 1]
        data.angular_velocities[body_a * 3 + 2] += ang_impulse_z * model.inv_inertias[body_a * 3 + 2]

    data.angular_velocities[body_b * 3 + 0] -= ang_impulse_x * model.inv_inertias[body_b * 3 + 0]
    data.angular_velocities[body_b * 3 + 1] -= ang_impulse_y * model.inv_inertias[body_b * 3 + 1]
    data.angular_velocities[body_b * 3 + 2] -= ang_impulse_z * model.inv_inertias[body_b * 3 + 2]


fn solve_joint_velocity_constraints[
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int
](
    model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS],
    mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS],
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
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int
](
    model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS],
    mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS],
    joint_idx: Int,
    baumgarte: Scalar[DTYPE],
):
    """Solve position constraint for a single hinge joint.

    Uses Baumgarte stabilization to correct anchor point drift.
    """
    var joint = model.joints[joint_idx]
    var body_a = joint.parent_body
    var body_b = joint.child_body

    # Get world-space anchors
    var anchor_a = _get_world_anchor(
        data, body_a,
        joint.anchor_parent_x, joint.anchor_parent_y, joint.anchor_parent_z
    )
    var anchor_b = _get_world_anchor(
        data, body_b,
        joint.anchor_child_x, joint.anchor_child_y, joint.anchor_child_z
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
    DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int
](
    model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS],
    mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS],
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
# GPU Joint Solver Functions
# =============================================================================

from layout import LayoutTensor, Layout
from ..gpu.constants import (
    body_offset,
    joint_offset,
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
    META_IDX_NUM_JOINTS,
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
fn solve_joint_velocity_constraints_gpu[
    DTYPE: DType,
    NUM_BODIES: Int,
    MAX_CONTACTS: Int,
    MAX_JOINTS: Int,
    STATE_SIZE: Int,
    BATCH: Int,
](
    env: Int,
    state: LayoutTensor[DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(NUM_BODIES, MODEL_BODY_SIZE), MutAnyOrigin],
    iterations: Int,
):
    """Solve joint velocity constraints on GPU."""
    # Use compile-time MAX_JOINTS directly to avoid any Float->Int conversion issues
    # The solver will iterate over all MAX_JOINTS slots; invalid joints should be skipped

    for _ in range(iterations):
        for j in range(MAX_JOINTS):
            var j_off = joint_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](j)

            # For now, hardcode body indices to make GPU work
            # TODO: Investigate GPU conditional/conversion issues
            # Known issue: Using conditionals or Int() on buffer reads causes incorrect behavior
            var body_a = -1  # Assume world-anchored
            var body_b = j   # Assume joint j connects to body j

            # Get anchor points from joint state
            var anchor_px = rebind[Scalar[DTYPE]](state[env, j_off + JOINT_IDX_ANCHOR_PX])
            var anchor_py = rebind[Scalar[DTYPE]](state[env, j_off + JOINT_IDX_ANCHOR_PY])
            var anchor_pz = rebind[Scalar[DTYPE]](state[env, j_off + JOINT_IDX_ANCHOR_PZ])
            var anchor_cx = rebind[Scalar[DTYPE]](state[env, j_off + JOINT_IDX_ANCHOR_CX])
            var anchor_cy = rebind[Scalar[DTYPE]](state[env, j_off + JOINT_IDX_ANCHOR_CY])
            var anchor_cz = rebind[Scalar[DTYPE]](state[env, j_off + JOINT_IDX_ANCHOR_CZ])

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
                var b_off_a = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](body_a)
                pa_x = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_PX])
                pa_y = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_PY])
                pa_z = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_PZ])
                var qa_x = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_QX])
                var qa_y = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_QY])
                var qa_z = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_QZ])
                var qa_w = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_QW])
                va_x = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_VX])
                va_y = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_VY])
                va_z = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_VZ])
                wa_wx = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_WX])
                wa_wy = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_WY])
                wa_wz = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_WZ])
                var rot_a = _quat_rotate_gpu(qa_x, qa_y, qa_z, qa_w, anchor_px, anchor_py, anchor_pz)
                wa_x = pa_x + rot_a[0]
                wa_y = pa_y + rot_a[1]
                wa_z = pa_z + rot_a[2]

            # Get world-space anchor for child
            var b_off_b = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](body_b)
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

            var rot_b = _quat_rotate_gpu(qb_x, qb_y, qb_z, qb_w, anchor_cx, anchor_cy, anchor_cz)
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
                inv_mass_a = rebind[Scalar[DTYPE]](model[body_a, MODEL_IDX_INV_MASS])
                var ra_sq = ra_x * ra_x + ra_y * ra_y + ra_z * ra_z
                var avg_inv_i_a = (
                    rebind[Scalar[DTYPE]](model[body_a, MODEL_IDX_INV_IXX])
                    + rebind[Scalar[DTYPE]](model[body_a, MODEL_IDX_INV_IYY])
                    + rebind[Scalar[DTYPE]](model[body_a, MODEL_IDX_INV_IZZ])
                ) / Scalar[DTYPE](3.0)
                rot_contrib_a = ra_sq * avg_inv_i_a

            var inv_mass_b = rebind[Scalar[DTYPE]](model[body_b, MODEL_IDX_INV_MASS])
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
                var b_off_a = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](body_a)
                state[env, b_off_a + BODY_IDX_VX] = va_x + impulse_x * inv_mass_a
                state[env, b_off_a + BODY_IDX_VY] = va_y + impulse_y * inv_mass_a
                state[env, b_off_a + BODY_IDX_VZ] = va_z + impulse_z * inv_mass_a

                var tau_a_x = ra_y * impulse_z - ra_z * impulse_y
                var tau_a_y = ra_z * impulse_x - ra_x * impulse_z
                var tau_a_z = ra_x * impulse_y - ra_y * impulse_x
                var inv_ixx_a = rebind[Scalar[DTYPE]](model[body_a, MODEL_IDX_INV_IXX])
                var inv_iyy_a = rebind[Scalar[DTYPE]](model[body_a, MODEL_IDX_INV_IYY])
                var inv_izz_a = rebind[Scalar[DTYPE]](model[body_a, MODEL_IDX_INV_IZZ])
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
            var inv_ixx_b = rebind[Scalar[DTYPE]](model[body_b, MODEL_IDX_INV_IXX])
            var inv_iyy_b = rebind[Scalar[DTYPE]](model[body_b, MODEL_IDX_INV_IYY])
            var inv_izz_b = rebind[Scalar[DTYPE]](model[body_b, MODEL_IDX_INV_IZZ])
            state[env, b_off_b + BODY_IDX_WX] = wb_x - tau_b_x * inv_ixx_b
            state[env, b_off_b + BODY_IDX_WY] = wb_y - tau_b_y * inv_iyy_b
            state[env, b_off_b + BODY_IDX_WZ] = wb_z - tau_b_z * inv_izz_b


@always_inline
fn solve_joint_position_constraints_gpu[
    DTYPE: DType,
    NUM_BODIES: Int,
    MAX_CONTACTS: Int,
    MAX_JOINTS: Int,
    STATE_SIZE: Int,
    BATCH: Int,
](
    env: Int,
    state: LayoutTensor[DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(NUM_BODIES, MODEL_BODY_SIZE), MutAnyOrigin],
    baumgarte: Scalar[DTYPE],
    iterations: Int,
):
    """Solve joint position constraints on GPU using Baumgarte stabilization."""
    # Use compile-time MAX_JOINTS directly to avoid any Float->Int conversion issues

    for _ in range(iterations):
        for j in range(MAX_JOINTS):
            var j_off = joint_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](j)

            # Get body indices
            # For now, assume first joint (j=0) is world-anchored to body 0
            # This is a simplification for debugging - will generalize later
            var body_a = -1  # World anchor (hardcoded for j=0)
            var body_b = j   # Assume joint j connects to body j

            # Get anchor points
            var anchor_px = rebind[Scalar[DTYPE]](state[env, j_off + JOINT_IDX_ANCHOR_PX])
            var anchor_py = rebind[Scalar[DTYPE]](state[env, j_off + JOINT_IDX_ANCHOR_PY])
            var anchor_pz = rebind[Scalar[DTYPE]](state[env, j_off + JOINT_IDX_ANCHOR_PZ])
            var anchor_cx = rebind[Scalar[DTYPE]](state[env, j_off + JOINT_IDX_ANCHOR_CX])
            var anchor_cy = rebind[Scalar[DTYPE]](state[env, j_off + JOINT_IDX_ANCHOR_CY])
            var anchor_cz = rebind[Scalar[DTYPE]](state[env, j_off + JOINT_IDX_ANCHOR_CZ])

            # Get world-space anchor for parent
            var wa_x = anchor_px
            var wa_y = anchor_py
            var wa_z = anchor_pz

            if body_a >= 0:
                var b_off_a = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](body_a)
                var pa_x = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_PX])
                var pa_y = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_PY])
                var pa_z = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_PZ])
                var qa_x = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_QX])
                var qa_y = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_QY])
                var qa_z = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_QZ])
                var qa_w = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_QW])
                var rot_a = _quat_rotate_gpu(qa_x, qa_y, qa_z, qa_w, anchor_px, anchor_py, anchor_pz)
                wa_x = pa_x + rot_a[0]
                wa_y = pa_y + rot_a[1]
                wa_z = pa_z + rot_a[2]

            # Get world-space anchor for child
            var b_off_b = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](body_b)
            var pb_x = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_PX])
            var pb_y = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_PY])
            var pb_z = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_PZ])
            var qb_x = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_QX])
            var qb_y = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_QY])
            var qb_z = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_QZ])
            var qb_w = rebind[Scalar[DTYPE]](state[env, b_off_b + BODY_IDX_QW])

            var rot_b = _quat_rotate_gpu(qb_x, qb_y, qb_z, qb_w, anchor_cx, anchor_cy, anchor_cz)
            var wb_x = pb_x + rot_b[0]
            var wb_y = pb_y + rot_b[1]
            var wb_z = pb_z + rot_b[2]

            # Position error
            var error_x = wa_x - wb_x
            var error_y = wa_y - wb_y
            var error_z = wa_z - wb_z

            var error_sq = error_x * error_x + error_y * error_y + error_z * error_z
            if error_sq < Scalar[DTYPE](1e-12):
                continue

            # Compute effective mass
            var inv_mass_a: Scalar[DTYPE] = 0
            if body_a >= 0:
                inv_mass_a = rebind[Scalar[DTYPE]](model[body_a, MODEL_IDX_INV_MASS])
            var inv_mass_b = rebind[Scalar[DTYPE]](model[body_b, MODEL_IDX_INV_MASS])

            var total_inv_mass = inv_mass_a + inv_mass_b
            if total_inv_mass < Scalar[DTYPE](1e-10):
                continue

            # Position correction
            var correction_x = -baumgarte * error_x / total_inv_mass
            var correction_y = -baumgarte * error_y / total_inv_mass
            var correction_z = -baumgarte * error_z / total_inv_mass

            # Apply to body A
            if body_a >= 0:
                var b_off_a = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](body_a)
                var pa_x = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_PX])
                var pa_y = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_PY])
                var pa_z = rebind[Scalar[DTYPE]](state[env, b_off_a + BODY_IDX_PZ])
                state[env, b_off_a + BODY_IDX_PX] = pa_x + correction_x * inv_mass_a
                state[env, b_off_a + BODY_IDX_PY] = pa_y + correction_y * inv_mass_a
                state[env, b_off_a + BODY_IDX_PZ] = pa_z + correction_z * inv_mass_a

            # Apply to body B
            state[env, b_off_b + BODY_IDX_PX] = pb_x - correction_x * inv_mass_b
            state[env, b_off_b + BODY_IDX_PY] = pb_y - correction_y * inv_mass_b
            state[env, b_off_b + BODY_IDX_PZ] = pb_z - correction_z * inv_mass_b

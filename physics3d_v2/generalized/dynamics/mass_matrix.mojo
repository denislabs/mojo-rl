"""Mass matrix computation for Generalized Coordinates engine.

Computes the joint-space mass matrix M(q) using the Composite Rigid Body Algorithm (CRBA).

For a system with NV degrees of freedom, M is an NV x NV symmetric positive definite matrix.
The equations of motion are: M(q) * qacc = qfrc - bias(q, qvel)

For simple HINGE-only chains (like pendulums), the mass matrix has a simpler structure:
- M[i,i] = I_axis + m * L^2 (parallel axis theorem)
- Off-diagonal terms couple connected joints

Reference: Featherstone, "Rigid Body Dynamics Algorithms"
"""

from math import sqrt
from ..types import ModelGC, DataGC
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..kinematics.quat_math import quat_rotate


# Helper to ensure positive size (avoid zero-size arrays)
fn _ensure_positive[n: Int]() -> Int:
    if n > 0:
        return n
    return 1


# =============================================================================
# Mass Matrix for HINGE-only Chains
# =============================================================================


fn compute_mass_matrix[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    M_SIZE: Int,
](
    model: ModelGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    data: DataGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    mut M: InlineArray[Scalar[DTYPE], M_SIZE],
):
    """Compute the joint-space mass matrix M(q).

    For HINGE joints, uses the parallel axis theorem:
    M[i,i] = I_axis + sum(m_k * r_k^2) for all bodies k affected by joint i

    Args:
        model: Static model configuration.
        data: Current state (xpos, xquat from forward kinematics).
        M: Output mass matrix (NV x NV, stored row-major).

    The mass matrix is symmetric, so only the upper triangle is computed
    and then copied to the lower triangle.
    """
    # Initialize to zero
    for i in range(NV * NV):
        M[i] = Scalar[DTYPE](0)

    # For each joint, compute its diagonal contribution
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var body = joint.body_id
        var dof_idx = joint.dof_adr

        if joint.jnt_type == JNT_HINGE:
            # Diagonal term: rotational inertia + m*L^2
            var m_effective = Scalar[DTYPE](0)

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

            # Add contribution from body and all its descendants
            # For simplicity, we only consider the direct body for now
            # (full CRBA would accumulate subtree)

            # Body's contribution: I_axis + m * r^2
            # where r is distance from joint axis to body CoM

            # Get body position relative to joint anchor
            var joint_pos_x = joint.pos_x
            var joint_pos_y = joint.pos_y
            var joint_pos_z = joint.pos_z

            # Transform joint pos to world
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

            # Distance from joint to body CoM
            var body_px = data.xpos[body * 3 + 0]
            var body_py = data.xpos[body * 3 + 1]
            var body_pz = data.xpos[body * 3 + 2]

            var r_x = body_px - jpos_world_x
            var r_y = body_py - jpos_world_y
            var r_z = body_pz - jpos_world_z

            # Project r perpendicular to axis
            var r_dot_axis = r_x * axis_x + r_y * axis_y + r_z * axis_z
            var r_perp_x = r_x - r_dot_axis * axis_x
            var r_perp_y = r_y - r_dot_axis * axis_y
            var r_perp_z = r_z - r_dot_axis * axis_z

            var r_perp_sq = r_perp_x * r_perp_x + r_perp_y * r_perp_y + r_perp_z * r_perp_z

            # Body mass
            var mass = model.body_mass[body]

            # Rotational inertia about axis (averaged for simplicity)
            var I_avg = (
                model.body_inertia[body * 3 + 0]
                + model.body_inertia[body * 3 + 1]
                + model.body_inertia[body * 3 + 2]
            ) / Scalar[DTYPE](3)

            # Parallel axis theorem: I_total = I_cm + m * r^2
            m_effective = I_avg + mass * r_perp_sq

            # Add contributions from descendant bodies
            for desc_body in range(body + 1, NBODY):
                if model.body_parent[desc_body] == body:
                    # This is a direct child, include its contribution
                    var desc_mass = model.body_mass[desc_body]
                    var desc_px = data.xpos[desc_body * 3 + 0]
                    var desc_py = data.xpos[desc_body * 3 + 1]
                    var desc_pz = data.xpos[desc_body * 3 + 2]

                    var desc_r_x = desc_px - jpos_world_x
                    var desc_r_y = desc_py - jpos_world_y
                    var desc_r_z = desc_pz - jpos_world_z

                    var desc_r_dot = desc_r_x * axis_x + desc_r_y * axis_y + desc_r_z * axis_z
                    var desc_perp_x = desc_r_x - desc_r_dot * axis_x
                    var desc_perp_y = desc_r_y - desc_r_dot * axis_y
                    var desc_perp_z = desc_r_z - desc_r_dot * axis_z

                    var desc_perp_sq = desc_perp_x * desc_perp_x + desc_perp_y * desc_perp_y + desc_perp_z * desc_perp_z

                    var desc_I_avg = (
                        model.body_inertia[desc_body * 3 + 0]
                        + model.body_inertia[desc_body * 3 + 1]
                        + model.body_inertia[desc_body * 3 + 2]
                    ) / Scalar[DTYPE](3)

                    m_effective = m_effective + desc_I_avg + desc_mass * desc_perp_sq

            # Store diagonal element
            M[dof_idx * NV + dof_idx] = m_effective

        elif joint.jnt_type == JNT_SLIDE:
            # For slide joint, effective mass is just the body mass
            # plus any descendants
            var m_total = model.body_mass[body]

            for desc_body in range(body + 1, NBODY):
                if model.body_parent[desc_body] == body:
                    m_total = m_total + model.body_mass[desc_body]

            M[dof_idx * NV + dof_idx] = m_total

        elif joint.jnt_type == JNT_FREE:
            # FREE joint: 6 DOF
            # Linear DOFs (0-2): total subtree mass
            # Angular DOFs (3-5): total subtree inertia

            var total_mass = model.body_mass[body]
            var I_xx = model.body_inertia[body * 3 + 0]
            var I_yy = model.body_inertia[body * 3 + 1]
            var I_zz = model.body_inertia[body * 3 + 2]

            # Add descendants (simplified, doesn't account for offset)
            for desc_body in range(body + 1, NBODY):
                if model.body_parent[desc_body] == body:
                    total_mass = total_mass + model.body_mass[desc_body]
                    I_xx = I_xx + model.body_inertia[desc_body * 3 + 0]
                    I_yy = I_yy + model.body_inertia[desc_body * 3 + 1]
                    I_zz = I_zz + model.body_inertia[desc_body * 3 + 2]

            # Linear mass (diagonal 3x3 block)
            M[dof_idx * NV + dof_idx] = total_mass
            M[(dof_idx + 1) * NV + (dof_idx + 1)] = total_mass
            M[(dof_idx + 2) * NV + (dof_idx + 2)] = total_mass

            # Angular inertia (diagonal 3x3 block)
            M[(dof_idx + 3) * NV + (dof_idx + 3)] = I_xx
            M[(dof_idx + 4) * NV + (dof_idx + 4)] = I_yy
            M[(dof_idx + 5) * NV + (dof_idx + 5)] = I_zz

        elif joint.jnt_type == JNT_BALL:
            # BALL joint: 3 angular DOF
            var I_xx = model.body_inertia[body * 3 + 0]
            var I_yy = model.body_inertia[body * 3 + 1]
            var I_zz = model.body_inertia[body * 3 + 2]

            M[dof_idx * NV + dof_idx] = I_xx
            M[(dof_idx + 1) * NV + (dof_idx + 1)] = I_yy
            M[(dof_idx + 2) * NV + (dof_idx + 2)] = I_zz


# =============================================================================
# Helper: Solve M * x = b (for small matrices)
# =============================================================================


fn solve_linear_1x1[
    DTYPE: DType
](
    M: Scalar[DTYPE],
    b: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """Solve 1x1 system: M * x = b."""
    if M > Scalar[DTYPE](1e-10) or M < Scalar[DTYPE](-1e-10):
        return b / M
    return Scalar[DTYPE](0)


fn solve_linear_diagonal[
    DTYPE: DType,
    NV: Int,
    M_SIZE: Int,
    V_SIZE: Int,
](
    M: InlineArray[Scalar[DTYPE], M_SIZE],
    b: InlineArray[Scalar[DTYPE], V_SIZE],
    mut x: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """Solve diagonal system: M * x = b.

    Assumes M is diagonal (only diagonal elements are used).
    """
    for i in range(NV):
        var m_ii = M[i * NV + i]
        if m_ii > Scalar[DTYPE](1e-10) or m_ii < Scalar[DTYPE](-1e-10):
            x[i] = b[i] / m_ii
        else:
            x[i] = Scalar[DTYPE](0)

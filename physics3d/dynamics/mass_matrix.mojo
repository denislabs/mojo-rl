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
from layout import LayoutTensor, Layout

from ..types import Model, Data
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..kinematics.quat_math import quat_rotate, gpu_quat_rotate
from ..gpu.constants import (
    xpos_offset,
    xquat_offset,
    model_body_offset,
    model_joint_offset,
    model_metadata_offset,
    BODY_IDX_PARENT,
    BODY_IDX_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_POS_X,
    JOINT_IDX_POS_Y,
    JOINT_IDX_POS_Z,
    JOINT_IDX_AXIS_X,
    JOINT_IDX_AXIS_Y,
    JOINT_IDX_AXIS_Z,
    MODEL_META_IDX_NJOINT,
)

from ..joint_types import (
    JNT_FREE,
    JNT_BALL,
    JNT_SLIDE,
    JNT_HINGE,
)


# Helper to ensure positive size (avoid zero-size arrays)
fn _ensure_positive[n: Int]() -> Int:
    if n > 0:
        return n
    return 1


fn _is_descendant[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
](
    model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    body: Int,
    ancestor: Int,
) -> Bool:
    """Check if body is a descendant of ancestor in the kinematic tree.

    Traverses the parent chain from body upwards to see if ancestor is found.
    """
    var current = body
    while current >= 0:
        if model.body_parent[current] == ancestor:
            return True
        current = model.body_parent[current]
    return False


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
    model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
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
            # m_effective will be computed below
            var m_effective: Scalar[DTYPE]

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
                    parent_qx,
                    parent_qy,
                    parent_qz,
                    parent_qw,
                    joint_pos_x,
                    joint_pos_y,
                    joint_pos_z,
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

            var r_perp_sq = (
                r_perp_x * r_perp_x + r_perp_y * r_perp_y + r_perp_z * r_perp_z
            )

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

            # Add contributions from ALL descendant bodies (not just direct children)
            for desc_body in range(body + 1, NBODY):
                if _is_descendant(model, desc_body, body):
                    # This body is in the subtree, include its contribution
                    var desc_mass = model.body_mass[desc_body]
                    var desc_px = data.xpos[desc_body * 3 + 0]
                    var desc_py = data.xpos[desc_body * 3 + 1]
                    var desc_pz = data.xpos[desc_body * 3 + 2]

                    var desc_r_x = desc_px - jpos_world_x
                    var desc_r_y = desc_py - jpos_world_y
                    var desc_r_z = desc_pz - jpos_world_z

                    var desc_r_dot = (
                        desc_r_x * axis_x
                        + desc_r_y * axis_y
                        + desc_r_z * axis_z
                    )
                    var desc_perp_x = desc_r_x - desc_r_dot * axis_x
                    var desc_perp_y = desc_r_y - desc_r_dot * axis_y
                    var desc_perp_z = desc_r_z - desc_r_dot * axis_z

                    var desc_perp_sq = (
                        desc_perp_x * desc_perp_x
                        + desc_perp_y * desc_perp_y
                        + desc_perp_z * desc_perp_z
                    )

                    var desc_I_avg = (
                        model.body_inertia[desc_body * 3 + 0]
                        + model.body_inertia[desc_body * 3 + 1]
                        + model.body_inertia[desc_body * 3 + 2]
                    ) / Scalar[DTYPE](3)

                    m_effective = (
                        m_effective + desc_I_avg + desc_mass * desc_perp_sq
                    )

            # Store diagonal element
            M[dof_idx * NV + dof_idx] = m_effective

        elif joint.jnt_type == JNT_SLIDE:
            # For slide joint, effective mass is the body mass
            # plus ALL descendants (not just direct children)
            var m_total = model.body_mass[body]

            for desc_body in range(body + 1, NBODY):
                if _is_descendant(model, desc_body, body):
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

            # Add ALL descendants (not just direct children)
            for desc_body in range(body + 1, NBODY):
                if _is_descendant(model, desc_body, body):
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
# Full Mass Matrix via CRBA (Composite Rigid Body Algorithm)
# =============================================================================


fn compute_mass_matrix_full[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    M_SIZE: Int,
    CDOF_SIZE: Int,
    CRB_SIZE: Int,
](
    model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    cdof: InlineArray[Scalar[DTYPE], CDOF_SIZE],
    crb: InlineArray[Scalar[DTYPE], CRB_SIZE],
    mut M: InlineArray[Scalar[DTYPE], M_SIZE],
):
    """Compute the full NV×NV mass matrix using direct sum over bodies.

    M[i,j] = sum over bodies k in subtree of deeper(body_i, body_j):
        m_k * (v_k_i · v_k_j) + omega_i · I_k_world · omega_j

    where v_k_i is the linear velocity of body k's CoM due to unit DOF i velocity,
    computed as: v_k_i = cdof_i_lin + cdof_i_ang × (pos_k - pos_body_i)

    This direct formulation avoids reference-point transformation issues.

    Args:
        model: Static model configuration.
        data: Current simulation state.
        cdof: Spatial motion axes per DOF (6*NV), from compute_cdof().
        crb: Composite rigid body inertia per body (10*NBODY), from compute_composite_inertia().
              Only the per-body inertia is used (not the accumulated composite).
        M: Output mass matrix (NV×NV, stored row-major).
    """
    # Zero M
    for i in range(NV * NV):
        M[i] = Scalar[DTYPE](0)

    # Build dof_to_body mapping
    comptime NV_SAFE = _ensure_positive[NV]()
    var dof_body = InlineArray[Int, NV_SAFE](uninitialized=True)
    for i in range(NV):
        dof_body[i] = 0

    for j in range(model.num_joints):
        var joint = model.joints[j]
        var body = joint.body_id
        var dof_adr = joint.dof_adr
        var ndof = 1
        if joint.jnt_type == JNT_FREE:
            ndof = 6
        elif joint.jnt_type == JNT_BALL:
            ndof = 3
        for d in range(ndof):
            dof_body[dof_adr + d] = body

    # Pre-compute per-body world-frame inertia tensor (just 3x3 rotational inertia)
    # Using body quaternions to rotate from local diagonal to world frame
    comptime NB_SAFE = _ensure_positive[NBODY]()
    comptime I_WORLD_SIZE = _ensure_positive[NBODY * 6]()
    var I_world = InlineArray[Scalar[DTYPE], I_WORLD_SIZE](uninitialized=True)
    for b in range(NBODY):
        var Ixx_l = model.body_inertia[b * 3 + 0]
        var Iyy_l = model.body_inertia[b * 3 + 1]
        var Izz_l = model.body_inertia[b * 3 + 2]

        var qx = data.xquat[b * 4 + 0]
        var qy = data.xquat[b * 4 + 1]
        var qz = data.xquat[b * 4 + 2]
        var qw = data.xquat[b * 4 + 3]

        var r00 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qy * qy + qz * qz)
        var r10 = Scalar[DTYPE](2) * (qx * qy + qw * qz)
        var r20 = Scalar[DTYPE](2) * (qx * qz - qw * qy)
        var r01 = Scalar[DTYPE](2) * (qx * qy - qw * qz)
        var r11 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qx * qx + qz * qz)
        var r21 = Scalar[DTYPE](2) * (qy * qz + qw * qx)
        var r02 = Scalar[DTYPE](2) * (qx * qz + qw * qy)
        var r12 = Scalar[DTYPE](2) * (qy * qz - qw * qx)
        var r22 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qx * qx + qy * qy)

        # I_world = R @ diag(Ixx, Iyy, Izz) @ R^T, store [xx, yy, zz, xy, xz, yz]
        I_world[b * 6 + 0] = (
            Ixx_l * r00 * r00 + Iyy_l * r01 * r01 + Izz_l * r02 * r02
        )
        I_world[b * 6 + 1] = (
            Ixx_l * r10 * r10 + Iyy_l * r11 * r11 + Izz_l * r12 * r12
        )
        I_world[b * 6 + 2] = (
            Ixx_l * r20 * r20 + Iyy_l * r21 * r21 + Izz_l * r22 * r22
        )
        I_world[b * 6 + 3] = (
            Ixx_l * r00 * r10 + Iyy_l * r01 * r11 + Izz_l * r02 * r12
        )
        I_world[b * 6 + 4] = (
            Ixx_l * r00 * r20 + Iyy_l * r01 * r21 + Izz_l * r02 * r22
        )
        I_world[b * 6 + 5] = (
            Ixx_l * r10 * r20 + Iyy_l * r11 * r21 + Izz_l * r12 * r22
        )

    # Compute M[i,j] for all pairs using direct body summation
    for i in range(NV):
        var body_i = dof_body[i]
        var ai0 = cdof[i * 6 + 0]
        var ai1 = cdof[i * 6 + 1]
        var ai2 = cdof[i * 6 + 2]
        var li0 = cdof[i * 6 + 3]
        var li1 = cdof[i * 6 + 4]
        var li2 = cdof[i * 6 + 5]

        for j in range(i, NV):
            var body_j = dof_body[j]
            var aj0 = cdof[j * 6 + 0]
            var aj1 = cdof[j * 6 + 1]
            var aj2 = cdof[j * 6 + 2]
            var lj0 = cdof[j * 6 + 3]
            var lj1 = cdof[j * 6 + 4]
            var lj2 = cdof[j * 6 + 5]

            # Determine common subtree: bodies affected by BOTH DOF i and DOF j
            # A body k is affected by DOF i if k == body_i or k is a descendant of body_i
            # For M[i,j], we sum over all bodies in the intersection of both subtrees
            var mij = Scalar[DTYPE](0)

            for k in range(NBODY):
                # Check if body k is in the subtree of body_i
                var in_subtree_i = (k == body_i) or _is_descendant(
                    model, k, body_i
                )
                if not in_subtree_i:
                    continue

                # Check if body k is in the subtree of body_j
                var in_subtree_j = (k == body_j) or _is_descendant(
                    model, k, body_j
                )
                if not in_subtree_j:
                    continue

                var mk = model.body_mass[k]
                var pk0 = data.xpos[k * 3 + 0]
                var pk1 = data.xpos[k * 3 + 1]
                var pk2 = data.xpos[k * 3 + 2]

                # Velocity of body k due to DOF i:
                # v_k_i = cdof_i_lin + cdof_i_ang × (pos_k - pos_body_i)
                var di0 = pk0 - data.xpos[body_i * 3 + 0]
                var di1 = pk1 - data.xpos[body_i * 3 + 1]
                var di2 = pk2 - data.xpos[body_i * 3 + 2]
                var vki0 = li0 + ai1 * di2 - ai2 * di1
                var vki1 = li1 + ai2 * di0 - ai0 * di2
                var vki2 = li2 + ai0 * di1 - ai1 * di0

                # Velocity of body k due to DOF j:
                var dj0 = pk0 - data.xpos[body_j * 3 + 0]
                var dj1 = pk1 - data.xpos[body_j * 3 + 1]
                var dj2 = pk2 - data.xpos[body_j * 3 + 2]
                var vkj0 = lj0 + aj1 * dj2 - aj2 * dj1
                var vkj1 = lj1 + aj2 * dj0 - aj0 * dj2
                var vkj2 = lj2 + aj0 * dj1 - aj1 * dj0

                # Linear momentum contribution: m_k * v_k_i · v_k_j
                mij = mij + mk * (vki0 * vkj0 + vki1 * vkj1 + vki2 * vkj2)

                # Rotational inertia contribution: omega_i · I_k_world · omega_j
                var Ik_xx = I_world[k * 6 + 0]
                var Ik_yy = I_world[k * 6 + 1]
                var Ik_zz = I_world[k * 6 + 2]
                var Ik_xy = I_world[k * 6 + 3]
                var Ik_xz = I_world[k * 6 + 4]
                var Ik_yz = I_world[k * 6 + 5]

                # I_k @ omega_j
                var Iaj0 = Ik_xx * aj0 + Ik_xy * aj1 + Ik_xz * aj2
                var Iaj1 = Ik_xy * aj0 + Ik_yy * aj1 + Ik_yz * aj2
                var Iaj2 = Ik_xz * aj0 + Ik_yz * aj1 + Ik_zz * aj2

                mij = mij + ai0 * Iaj0 + ai1 * Iaj1 + ai2 * Iaj2

            M[i * NV + j] = mij
            if i != j:
                M[j * NV + i] = mij


# =============================================================================
# LDL Factorization and Solve for SPD matrices
# =============================================================================


fn ldl_factor[
    DTYPE: DType,
    NV: Int,
    M_SIZE: Int,
    V_SIZE: Int,
](
    M: InlineArray[Scalar[DTYPE], M_SIZE],
    mut L: InlineArray[Scalar[DTYPE], M_SIZE],
    mut D: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """In-place LDL factorization of NV×NV SPD matrix M.

    Computes M = L * D * L^T where:
    - L is unit lower triangular (L[i,i] = 1)
    - D is diagonal

    Args:
        M: Input NV×NV matrix (row-major).
        L: Output lower triangular matrix (row-major).
        D: Output diagonal vector (NV entries).
    """
    for i in range(NV * NV):
        L[i] = Scalar[DTYPE](0)
    for i in range(NV):
        D[i] = Scalar[DTYPE](0)
        L[i * NV + i] = Scalar[DTYPE](1)

    for j in range(NV):
        var d_j = M[j * NV + j]
        for k in range(j):
            d_j = d_j - L[j * NV + k] * L[j * NV + k] * D[k]
        D[j] = d_j

        if d_j > Scalar[DTYPE](1e-14) or d_j < Scalar[DTYPE](-1e-14):
            for i in range(j + 1, NV):
                var l_ij = M[i * NV + j]
                for k in range(j):
                    l_ij = l_ij - L[i * NV + k] * L[j * NV + k] * D[k]
                L[i * NV + j] = l_ij / d_j


fn ldl_solve[
    DTYPE: DType,
    NV: Int,
    M_SIZE: Int,
    V_SIZE: Int,
](
    L: InlineArray[Scalar[DTYPE], M_SIZE],
    D: InlineArray[Scalar[DTYPE], V_SIZE],
    b: InlineArray[Scalar[DTYPE], V_SIZE],
    mut x: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """Solve M * x = b using precomputed LDL factors.

    Solves L * D * L^T * x = b in three steps:
    1. Forward substitution: L * y = b
    2. Diagonal solve: D * z = y
    3. Backward substitution: L^T * x = z
    """
    var y = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        var s = b[i]
        for j in range(i):
            s = s - L[i * NV + j] * y[j]
        y[i] = s

    var z = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        if D[i] > Scalar[DTYPE](1e-14) or D[i] < Scalar[DTYPE](-1e-14):
            z[i] = y[i] / D[i]
        else:
            z[i] = Scalar[DTYPE](0)

    for i in range(NV - 1, -1, -1):
        var s = z[i]
        for j in range(i + 1, NV):
            s = s - L[j * NV + i] * x[j]
        x[i] = s


fn compute_M_inv_from_ldl[
    DTYPE: DType,
    NV: Int,
    M_SIZE: Int,
    V_SIZE: Int,
](
    L: InlineArray[Scalar[DTYPE], M_SIZE],
    D: InlineArray[Scalar[DTYPE], V_SIZE],
    mut M_inv: InlineArray[Scalar[DTYPE], M_SIZE],
):
    """Compute full dense M^-1 from LDL factors by solving M * col = e_j."""
    var e = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var col = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

    for j in range(NV):
        for i in range(NV):
            e[i] = Scalar[DTYPE](0)
        e[j] = Scalar[DTYPE](1)

        ldl_solve[DTYPE, NV, M_SIZE, V_SIZE](L, D, e, col)

        for i in range(NV):
            M_inv[i * NV + j] = col[i]


# =============================================================================
# GPU: Full Mass Matrix + LDL
# =============================================================================


@always_inline
fn compute_mass_matrix_full_gpu[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    STATE_SIZE: Int,
    MODEL_SIZE: Int,
    BATCH: Int,
    WS_SIZE: Int,
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    workspace: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ],
):
    """Compute full NV×NV mass matrix on GPU. Reads cdof, writes M to workspace.
    """
    from ..gpu.constants import ws_cdof_offset, ws_M_offset

    # Derive pointers from workspace (MutAnyOrigin)
    comptime cdof_idx = ws_cdof_offset()
    comptime M_idx = ws_M_offset[NV, NBODY]()

    var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
    var num_joints = Int(
        rebind[Scalar[DTYPE]](model[0, model_meta_off + MODEL_META_IDX_NJOINT])
    )

    for i in range(NV * NV):
        workspace[env, M_idx + i] = 0

    comptime NV_SAFE = _ensure_positive[NV]()
    var dof_body = InlineArray[Int, NV_SAFE](uninitialized=True)
    for i in range(NV):
        dof_body[i] = 0

    for j in range(num_joints):
        var joint_off = model_joint_offset[NBODY](j)
        var jnt_type = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
        )
        var body_id = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_BODY_ID])
        )
        var dof_adr = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR])
        )

        var ndof = 1
        if jnt_type == JNT_FREE:
            ndof = 6
        elif jnt_type == JNT_BALL:
            ndof = 3
        for d in range(ndof):
            dof_body[dof_adr + d] = body_id

    var xpos_off = xpos_offset[NQ, NV, NBODY]()
    var xquat_off = xquat_offset[NQ, NV, NBODY]()

    # Pre-compute per-body world-frame inertia tensor
    comptime I_WORLD_SIZE = _ensure_positive[NBODY * 6]()
    var I_world = InlineArray[Scalar[DTYPE], I_WORLD_SIZE](uninitialized=True)
    for b in range(NBODY):
        var body_off = model_body_offset(b)
        var Ixx_l = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IXX])
        var Iyy_l = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IYY])
        var Izz_l = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IZZ])

        var qx = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 0])
        var qy = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 1])
        var qz = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 2])
        var qw = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 3])

        var r00 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qy * qy + qz * qz)
        var r10 = Scalar[DTYPE](2) * (qx * qy + qw * qz)
        var r20 = Scalar[DTYPE](2) * (qx * qz - qw * qy)
        var r01 = Scalar[DTYPE](2) * (qx * qy - qw * qz)
        var r11 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qx * qx + qz * qz)
        var r21 = Scalar[DTYPE](2) * (qy * qz + qw * qx)
        var r02 = Scalar[DTYPE](2) * (qx * qz + qw * qy)
        var r12 = Scalar[DTYPE](2) * (qy * qz - qw * qx)
        var r22 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qx * qx + qy * qy)

        I_world[b * 6 + 0] = (
            Ixx_l * r00 * r00 + Iyy_l * r01 * r01 + Izz_l * r02 * r02
        )
        I_world[b * 6 + 1] = (
            Ixx_l * r10 * r10 + Iyy_l * r11 * r11 + Izz_l * r12 * r12
        )
        I_world[b * 6 + 2] = (
            Ixx_l * r20 * r20 + Iyy_l * r21 * r21 + Izz_l * r22 * r22
        )
        I_world[b * 6 + 3] = (
            Ixx_l * r00 * r10 + Iyy_l * r01 * r11 + Izz_l * r02 * r12
        )
        I_world[b * 6 + 4] = (
            Ixx_l * r00 * r20 + Iyy_l * r01 * r21 + Izz_l * r02 * r22
        )
        I_world[b * 6 + 5] = (
            Ixx_l * r10 * r20 + Iyy_l * r11 * r21 + Izz_l * r12 * r22
        )

    # Compute M[i,j] using direct body summation
    for i in range(NV):
        var body_i = dof_body[i]
        var ai0 = workspace[env, cdof_idx + i * 6 + 0]
        var ai1 = workspace[env, cdof_idx + i * 6 + 1]
        var ai2 = workspace[env, cdof_idx + i * 6 + 2]
        var li0 = workspace[env, cdof_idx + i * 6 + 3]
        var li1 = workspace[env, cdof_idx + i * 6 + 4]
        var li2 = workspace[env, cdof_idx + i * 6 + 5]

        for j in range(i, NV):
            var body_j = dof_body[j]
            var aj0 = workspace[env, cdof_idx + j * 6 + 0]
            var aj1 = workspace[env, cdof_idx + j * 6 + 1]
            var aj2 = workspace[env, cdof_idx + j * 6 + 2]
            var lj0 = workspace[env, cdof_idx + j * 6 + 3]
            var lj1 = workspace[env, cdof_idx + j * 6 + 4]
            var lj2 = workspace[env, cdof_idx + j * 6 + 5]

            var mij: workspace.element_type = 0

            for k in range(NBODY):
                var in_subtree_i = (k == body_i) or _is_descendant_gpu[
                    DTYPE, NBODY, MODEL_SIZE
                ](model, k, body_i)
                if not in_subtree_i:
                    continue
                var in_subtree_j = (k == body_j) or _is_descendant_gpu[
                    DTYPE, NBODY, MODEL_SIZE
                ](model, k, body_j)
                if not in_subtree_j:
                    continue

                var body_off_k = model_body_offset(k)
                var mk = rebind[Scalar[DTYPE]](
                    model[0, body_off_k + BODY_IDX_MASS]
                )
                var pk0 = rebind[Scalar[DTYPE]](
                    state[env, xpos_off + k * 3 + 0]
                )
                var pk1 = rebind[Scalar[DTYPE]](
                    state[env, xpos_off + k * 3 + 1]
                )
                var pk2 = rebind[Scalar[DTYPE]](
                    state[env, xpos_off + k * 3 + 2]
                )

                var pi0 = rebind[Scalar[DTYPE]](
                    state[env, xpos_off + body_i * 3 + 0]
                )
                var pi1 = rebind[Scalar[DTYPE]](
                    state[env, xpos_off + body_i * 3 + 1]
                )
                var pi2 = rebind[Scalar[DTYPE]](
                    state[env, xpos_off + body_i * 3 + 2]
                )
                var di0 = pk0 - pi0
                var di1 = pk1 - pi1
                var di2 = pk2 - pi2
                var vki0 = li0 + ai1 * di2 - ai2 * di1
                var vki1 = li1 + ai2 * di0 - ai0 * di2
                var vki2 = li2 + ai0 * di1 - ai1 * di0

                var pj0 = rebind[Scalar[DTYPE]](
                    state[env, xpos_off + body_j * 3 + 0]
                )
                var pj1 = rebind[Scalar[DTYPE]](
                    state[env, xpos_off + body_j * 3 + 1]
                )
                var pj2 = rebind[Scalar[DTYPE]](
                    state[env, xpos_off + body_j * 3 + 2]
                )
                var dj0 = pk0 - pj0
                var dj1 = pk1 - pj1
                var dj2 = pk2 - pj2
                var vkj0 = lj0 + aj1 * dj2 - aj2 * dj1
                var vkj1 = lj1 + aj2 * dj0 - aj0 * dj2
                var vkj2 = lj2 + aj0 * dj1 - aj1 * dj0

                mij = mij + mk * (vki0 * vkj0 + vki1 * vkj1 + vki2 * vkj2)

                var Ik_xx = I_world[k * 6 + 0]
                var Ik_yy = I_world[k * 6 + 1]
                var Ik_zz = I_world[k * 6 + 2]
                var Ik_xy = I_world[k * 6 + 3]
                var Ik_xz = I_world[k * 6 + 4]
                var Ik_yz = I_world[k * 6 + 5]

                var Iaj0 = Ik_xx * aj0 + Ik_xy * aj1 + Ik_xz * aj2
                var Iaj1 = Ik_xy * aj0 + Ik_yy * aj1 + Ik_yz * aj2
                var Iaj2 = Ik_xz * aj0 + Ik_yz * aj1 + Ik_zz * aj2

                mij = mij + ai0 * Iaj0 + ai1 * Iaj1 + ai2 * Iaj2

            workspace[env, M_idx + i * NV + j] = mij
            if i != j:
                workspace[env, M_idx + j * NV + i] = mij


@always_inline
fn ldl_factor_gpu[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    BATCH: Int,
    WS_SIZE: Int,
](
    env: Int,
    workspace: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ],
):
    """LDL factorization on GPU. Reads M, writes L and D to workspace."""
    from ..gpu.constants import ws_M_offset, ws_L_offset, ws_D_offset

    comptime M_idx = ws_M_offset[NV, NBODY]()
    comptime L_idx = ws_L_offset[NV, NBODY]()
    comptime D_idx = ws_D_offset[NV, NBODY]()

    for i in range(NV * NV):
        workspace[env, L_idx + i] = 0
    for i in range(NV):
        workspace[env, D_idx + i] = 0
        workspace[env, L_idx + i * NV + i] = 1

    for j in range(NV):
        var d_j = workspace[env, M_idx + j * NV + j]
        for k in range(j):
            d_j = (
                d_j
                - workspace[env, L_idx + j * NV + k]
                * workspace[env, L_idx + j * NV + k]
                * workspace[env, D_idx + k]
            )
        workspace[env, D_idx + j] = d_j

        if d_j > Scalar[DTYPE](1e-14) or d_j < Scalar[DTYPE](-1e-14):
            for i in range(j + 1, NV):
                var l_ij = workspace[env, M_idx + i * NV + j]
                for k in range(j):
                    l_ij = (
                        l_ij
                        - workspace[env, L_idx + i * NV + k]
                        * workspace[env, L_idx + j * NV + k]
                        * workspace[env, D_idx + k]
                    )
                workspace[env, L_idx + i * NV + j] = l_ij / d_j


@always_inline
fn ldl_solve_gpu[
    DTYPE: DType,
    NV: Int,
    M_SIZE: Int,
    V_SIZE: Int,
](
    L: InlineArray[Scalar[DTYPE], M_SIZE],
    D: InlineArray[Scalar[DTYPE], V_SIZE],
    b: InlineArray[Scalar[DTYPE], V_SIZE],
    mut x: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """LDL solve (GPU-compatible, same algorithm as CPU)."""
    var y = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        var s = b[i]
        for j in range(i):
            s = s - L[i * NV + j] * y[j]
        y[i] = s

    var z = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        if D[i] > Scalar[DTYPE](1e-14) or D[i] < Scalar[DTYPE](-1e-14):
            z[i] = y[i] / D[i]
        else:
            z[i] = Scalar[DTYPE](0)

    for i in range(NV - 1, -1, -1):
        var s = z[i]
        for j in range(i + 1, NV):
            s = s - L[j * NV + i] * x[j]
        x[i] = s


@always_inline
fn ldl_solve_workspace_gpu[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    BATCH: Int,
    WS_SIZE: Int,
](
    env: Int,
    workspace: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ],
):
    """LDL solve on GPU. Reads L, D, f_net from workspace, writes qacc to workspace.
    """
    from ..gpu.constants import (
        ws_L_offset,
        ws_D_offset,
        ws_fnet_offset,
        ws_qacc_ws_offset,
    )

    comptime L_idx = ws_L_offset[NV, NBODY]()
    comptime D_idx = ws_D_offset[NV, NBODY]()
    comptime b_idx = ws_fnet_offset[NV, NBODY]()
    comptime x_idx = ws_qacc_ws_offset[NV, NBODY]()

    # Forward substitution: y = L^(-1) * b
    comptime V_SIZE = _ensure_positive[NV]()
    var y = InlineArray[workspace.element_type, V_SIZE](uninitialized=True)
    for i in range(NV):
        var s = workspace[env, b_idx + i]
        for j in range(i):
            s = s - workspace[env, L_idx + i * NV + j] * y[j]
        y[i] = s

    # Diagonal solve: z = D^(-1) * y
    var z = InlineArray[workspace.element_type, V_SIZE](uninitialized=True)
    for i in range(NV):
        var d_i = workspace[env, D_idx + i]
        if d_i > 1e-14 or d_i < -1e-14:
            z[i] = y[i] / d_i
        else:
            z[i] = 0

    # Backward substitution: x = L^(-T) * z
    for i in range(NV - 1, -1, -1):
        var s = z[i]
        for j in range(i + 1, NV):
            s = (
                s
                - workspace[env, L_idx + j * NV + i] * workspace[env, x_idx + j]
            )
        workspace[env, x_idx + i] = s


@always_inline
fn compute_M_inv_from_ldl_gpu[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    BATCH: Int,
    WS_SIZE: Int,
](
    env: Int,
    workspace: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ],
):
    """Compute full dense M^-1 from LDL factors in workspace.

    Reads L, D from workspace. Writes M_inv to workspace.
    Uses small local InlineArrays (e, col) for the column solve.
    """
    from ..gpu.constants import ws_L_offset, ws_D_offset, ws_m_inv_offset

    comptime L_idx = ws_L_offset[NV, NBODY]()
    comptime D_idx = ws_D_offset[NV, NBODY]()
    comptime M_inv_idx = ws_m_inv_offset[NV, NBODY]()

    comptime V_SIZE = _ensure_positive[NV]()
    var e = InlineArray[workspace.element_type, V_SIZE](uninitialized=True)
    var col = InlineArray[workspace.element_type, V_SIZE](uninitialized=True)

    for j in range(NV):
        for i in range(NV):
            e[i] = 0
        e[j] = 1

        # Forward substitution: y = L^(-1) * e
        var y = InlineArray[workspace.element_type, V_SIZE](uninitialized=True)
        for i in range(NV):
            var s = e[i]
            for k in range(i):
                s = s - workspace[env, L_idx + i * NV + k] * y[k]
            y[i] = s

        # Diagonal solve: z = D^(-1) * y
        var z = InlineArray[workspace.element_type, V_SIZE](uninitialized=True)
        for i in range(NV):
            var d_i = workspace[env, D_idx + i]
            if d_i > 1e-14 or d_i < -1e-14:
                z[i] = y[i] / d_i
            else:
                z[i] = 0

        # Backward substitution: col = L^(-T) * z
        for i in range(NV - 1, -1, -1):
            var s = z[i]
            for k in range(i + 1, NV):
                s = s - workspace[env, L_idx + k * NV + i] * col[k]
            col[i] = s

        for i in range(NV):
            workspace[env, M_inv_idx + i * NV + j] = col[i]


# =============================================================================
# Helper: Solve M * x = b (for small matrices)
# =============================================================================


fn solve_linear_1x1[
    DTYPE: DType
](M: Scalar[DTYPE], b: Scalar[DTYPE],) -> Scalar[DTYPE]:
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


# =============================================================================
# GPU Mass Matrix Kernel (Diagonal approximation)
# =============================================================================


@always_inline
fn _is_descendant_gpu[
    DTYPE: DType,
    NBODY: Int,
    MODEL_SIZE: Int,
](
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    body: Int,
    ancestor: Int,
) -> Bool:
    """GPU-compatible check if body is a descendant of ancestor."""
    var current = body
    while current >= 0:
        var body_off = model_body_offset(current)
        var parent = Int(
            rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_PARENT])
        )
        if parent == ancestor:
            return True
        current = parent
    return False


@always_inline
fn compute_mass_matrix_diagonal_gpu[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    STATE_SIZE: Int,
    MODEL_SIZE: Int,
    V_SIZE: Int,
    BATCH: Int,
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    mut M_diag: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """Compute diagonal of mass matrix M(q) (GPU version).

    For efficiency on GPU, we compute only diagonal elements.
    """
    var xpos_off = xpos_offset[NQ, NV, NBODY]()
    var xquat_off = xquat_offset[NQ, NV, NBODY]()

    var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
    var num_joints = Int(
        rebind[Scalar[DTYPE]](model[0, model_meta_off + MODEL_META_IDX_NJOINT])
    )

    # Initialize to zero
    for i in range(NV):
        M_diag[i] = Scalar[DTYPE](0)

    for j in range(num_joints):
        var joint_off = model_joint_offset[NBODY](j)

        var jnt_type = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
        )
        var body_id = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_BODY_ID])
        )
        var dof_adr = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR])
        )

        var body_off = model_body_offset(body_id)
        var parent = Int(
            rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_PARENT])
        )
        var mass = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_MASS])
        var I_xx = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IXX])
        var I_yy = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IYY])
        var I_zz = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IZZ])
        var I_avg = (I_xx + I_yy + I_zz) / Scalar[DTYPE](3)

        var jpos_x = rebind[Scalar[DTYPE]](
            model[0, joint_off + JOINT_IDX_POS_X]
        )
        var jpos_y = rebind[Scalar[DTYPE]](
            model[0, joint_off + JOINT_IDX_POS_Y]
        )
        var jpos_z = rebind[Scalar[DTYPE]](
            model[0, joint_off + JOINT_IDX_POS_Z]
        )
        var axis_x = rebind[Scalar[DTYPE]](
            model[0, joint_off + JOINT_IDX_AXIS_X]
        )
        var axis_y = rebind[Scalar[DTYPE]](
            model[0, joint_off + JOINT_IDX_AXIS_Y]
        )
        var axis_z = rebind[Scalar[DTYPE]](
            model[0, joint_off + JOINT_IDX_AXIS_Z]
        )

        if jnt_type == JNT_HINGE:
            var jpos_world_x = jpos_x
            var jpos_world_y = jpos_y
            var jpos_world_z = jpos_z

            if parent >= 0:
                var ppx = rebind[Scalar[DTYPE]](
                    state[env, xpos_off + parent * 3 + 0]
                )
                var ppy = rebind[Scalar[DTYPE]](
                    state[env, xpos_off + parent * 3 + 1]
                )
                var ppz = rebind[Scalar[DTYPE]](
                    state[env, xpos_off + parent * 3 + 2]
                )
                var pqx = rebind[Scalar[DTYPE]](
                    state[env, xquat_off + parent * 4 + 0]
                )
                var pqy = rebind[Scalar[DTYPE]](
                    state[env, xquat_off + parent * 4 + 1]
                )
                var pqz = rebind[Scalar[DTYPE]](
                    state[env, xquat_off + parent * 4 + 2]
                )
                var pqw = rebind[Scalar[DTYPE]](
                    state[env, xquat_off + parent * 4 + 3]
                )

                var rotated = gpu_quat_rotate(
                    pqx, pqy, pqz, pqw, jpos_x, jpos_y, jpos_z
                )
                jpos_world_x = ppx + rotated[0]
                jpos_world_y = ppy + rotated[1]
                jpos_world_z = ppz + rotated[2]

            var axis_world_x = axis_x
            var axis_world_y = axis_y
            var axis_world_z = axis_z
            if parent >= 0:
                var pqx = rebind[Scalar[DTYPE]](
                    state[env, xquat_off + parent * 4 + 0]
                )
                var pqy = rebind[Scalar[DTYPE]](
                    state[env, xquat_off + parent * 4 + 1]
                )
                var pqz = rebind[Scalar[DTYPE]](
                    state[env, xquat_off + parent * 4 + 2]
                )
                var pqw = rebind[Scalar[DTYPE]](
                    state[env, xquat_off + parent * 4 + 3]
                )
                var rotated = gpu_quat_rotate(
                    pqx, pqy, pqz, pqw, axis_x, axis_y, axis_z
                )
                axis_world_x = rotated[0]
                axis_world_y = rotated[1]
                axis_world_z = rotated[2]

            var body_px = rebind[Scalar[DTYPE]](
                state[env, xpos_off + body_id * 3 + 0]
            )
            var body_py = rebind[Scalar[DTYPE]](
                state[env, xpos_off + body_id * 3 + 1]
            )
            var body_pz = rebind[Scalar[DTYPE]](
                state[env, xpos_off + body_id * 3 + 2]
            )

            var rx = body_px - jpos_world_x
            var ry = body_py - jpos_world_y
            var rz = body_pz - jpos_world_z

            var r_dot_axis = (
                rx * axis_world_x + ry * axis_world_y + rz * axis_world_z
            )
            var r_perp_x = rx - r_dot_axis * axis_world_x
            var r_perp_y = ry - r_dot_axis * axis_world_y
            var r_perp_z = rz - r_dot_axis * axis_world_z
            var r_perp_sq = (
                r_perp_x * r_perp_x + r_perp_y * r_perp_y + r_perp_z * r_perp_z
            )

            var m_effective = I_avg + mass * r_perp_sq

            # Add contributions from ALL descendant bodies (matching CPU version)
            for desc_body in range(body_id + 1, NBODY):
                if _is_descendant_gpu[DTYPE, NBODY, MODEL_SIZE](
                    model, desc_body, body_id
                ):
                    var desc_body_off = model_body_offset(desc_body)
                    var desc_mass = rebind[Scalar[DTYPE]](
                        model[0, desc_body_off + BODY_IDX_MASS]
                    )
                    var desc_I_xx = rebind[Scalar[DTYPE]](
                        model[0, desc_body_off + BODY_IDX_IXX]
                    )
                    var desc_I_yy = rebind[Scalar[DTYPE]](
                        model[0, desc_body_off + BODY_IDX_IYY]
                    )
                    var desc_I_zz = rebind[Scalar[DTYPE]](
                        model[0, desc_body_off + BODY_IDX_IZZ]
                    )
                    var desc_I_avg = (
                        desc_I_xx + desc_I_yy + desc_I_zz
                    ) / Scalar[DTYPE](3)

                    var desc_px = rebind[Scalar[DTYPE]](
                        state[env, xpos_off + desc_body * 3 + 0]
                    )
                    var desc_py = rebind[Scalar[DTYPE]](
                        state[env, xpos_off + desc_body * 3 + 1]
                    )
                    var desc_pz = rebind[Scalar[DTYPE]](
                        state[env, xpos_off + desc_body * 3 + 2]
                    )

                    var desc_rx = desc_px - jpos_world_x
                    var desc_ry = desc_py - jpos_world_y
                    var desc_rz = desc_pz - jpos_world_z

                    var desc_r_dot = (
                        desc_rx * axis_world_x
                        + desc_ry * axis_world_y
                        + desc_rz * axis_world_z
                    )
                    var desc_perp_x = desc_rx - desc_r_dot * axis_world_x
                    var desc_perp_y = desc_ry - desc_r_dot * axis_world_y
                    var desc_perp_z = desc_rz - desc_r_dot * axis_world_z
                    var desc_perp_sq = (
                        desc_perp_x * desc_perp_x
                        + desc_perp_y * desc_perp_y
                        + desc_perp_z * desc_perp_z
                    )

                    m_effective = (
                        m_effective + desc_I_avg + desc_mass * desc_perp_sq
                    )

            M_diag[dof_adr] = m_effective

        elif jnt_type == JNT_SLIDE:
            # Accumulate mass from body and ALL descendants
            var total_mass = mass
            for desc_body in range(body_id + 1, NBODY):
                if _is_descendant_gpu[DTYPE, NBODY, MODEL_SIZE](
                    model, desc_body, body_id
                ):
                    var desc_body_off = model_body_offset(desc_body)
                    var desc_mass = rebind[Scalar[DTYPE]](
                        model[0, desc_body_off + BODY_IDX_MASS]
                    )
                    total_mass = total_mass + desc_mass
            M_diag[dof_adr] = total_mass

        elif jnt_type == JNT_FREE:
            M_diag[dof_adr + 0] = mass
            M_diag[dof_adr + 1] = mass
            M_diag[dof_adr + 2] = mass
            M_diag[dof_adr + 3] = I_xx
            M_diag[dof_adr + 4] = I_yy
            M_diag[dof_adr + 5] = I_zz

        elif jnt_type == JNT_BALL:
            M_diag[dof_adr + 0] = I_xx
            M_diag[dof_adr + 1] = I_yy
            M_diag[dof_adr + 2] = I_zz

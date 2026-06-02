"""Mass matrix computation for Generalized Coordinates engine.

Computes the joint-space mass matrix M(q) using the Composite Rigid Body Algorithm (CRBA).

For a system with NV degrees of freedom, M is an NV x NV symmetric positive definite matrix.
The equations of motion are: M(q) * qacc = qfrc - bias(q, qvel)

For simple HINGE-only chains (like pendulums), the mass matrix has a simpler structure:
- M[i,i] = I_axis + m * L^2 (parallel axis theorem)
- Off-diagonal terms couple connected joints

Reference: Featherstone, "Rigid Body Dynamics Algorithms"
"""

from std.math import sqrt
from layout import LayoutTensor, Layout

from ..types import Model, Data, ConeType
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..kinematics.quat_math import (
    quat_rotate,
    quat_mul,
    gpu_quat_rotate,
    gpu_quat_mul,
)
from ..gpu.constants import (
    xpos_offset,
    xquat_offset,
    xipos_offset,
    model_body_offset,
    model_joint_offset,
    model_metadata_offset,
    ws_cdof_offset,
    ws_L_offset,
    ws_M_offset,
    ws_D_offset,
    BODY_IDX_PARENT,
    BODY_IDX_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    BODY_IDX_IPOS_X,
    BODY_IDX_IPOS_Y,
    BODY_IDX_IPOS_Z,
    BODY_IDX_IQUAT_X,
    BODY_IDX_IQUAT_Y,
    BODY_IDX_IQUAT_Z,
    BODY_IDX_IQUAT_W,
    BODY_IDX_ROOTID,
    subtree_com_offset,
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
def _ensure_positive[n: Int]() -> Int:
    if n > 0:
        return n
    return 1


def _is_descendant[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int = 0,
    MAX_EQUALITY: Int = 0,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    MAX_TENDON: Int = 0,
    NSITE: Int = 0,
](
    model: Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        MAX_EQUALITY,
        CONE_TYPE,
        MAX_TENDON,
        NSITE,
    ],
    body: Int,
    ancestor: Int,
) -> Bool:
    """Check if body is a descendant of ancestor in the kinematic tree.

    Traverses the parent chain from body upwards to see if ancestor is found.
    """
    var current = body
    while current > 0:
        if model.body_parent[current] == ancestor:
            return True
        current = model.body_parent[current]
    return False


# =============================================================================
# Mass Matrix for HINGE-only Chains
# =============================================================================


def compute_mass_matrix[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    M_SIZE: Int,
    NGEOM: Int = 0,
    MAX_EQUALITY: Int = 0,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    MAX_TENDON: Int = 0,
    NSITE: Int = 0,
](
    model: Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        MAX_EQUALITY,
        CONE_TYPE,
        MAX_TENDON,
        NSITE,
    ],
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
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

            # Distance from joint to body CoM (use xipos = CoM world position)
            var body_px = data.xipos[body * 3 + 0]
            var body_py = data.xipos[body * 3 + 1]
            var body_pz = data.xipos[body * 3 + 2]

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
                    var desc_px = data.xipos[desc_body * 3 + 0]
                    var desc_py = data.xipos[desc_body * 3 + 1]
                    var desc_pz = data.xipos[desc_body * 3 + 2]

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
# Body Inverse Weights (MuJoCo mj_setConst body_invweight0)
# =============================================================================


def compute_body_invweight0[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int = 0,
    MAX_EQUALITY: Int = 0,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    MAX_TENDON: Int = 0,
    NSITE: Int = 0,
](
    mut model: Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        MAX_EQUALITY,
        CONE_TYPE,
        MAX_TENDON,
        NSITE,
    ],
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
):
    """Compute body_invweight0 from mass matrix and body CoM Jacobians.

    Follows MuJoCo's mj_setConst (engine_setconst.c:620-661):
    For each body, compute the body CoM Jacobian J (6×NV), then
    A = J * M^{-1} * J^T. The inverse weights are:
      invweight0[2*i]   = avg(A[0,0], A[1,1], A[2,2])  (translation)
      invweight0[2*i+1] = avg(A[3,3], A[4,4], A[5,5])  (rotation)

    Requires: forward_kinematics and compute_cdof already called.
    """
    # Compute cdof, mass matrix, and LDL factorization
    var cdof = List[Scalar[DTYPE]](capacity=NV * 6)
    var crb = List[Scalar[DTYPE]](capacity=NBODY * 10)
    var M = List[Scalar[DTYPE]](capacity=NV * NV)
    var L = List[Scalar[DTYPE]](capacity=NV * NV)
    var D = List[Scalar[DTYPE]](capacity=NV)
    for _ in range(NV * 6):
        cdof.append(Scalar[DTYPE](0))
    for _ in range(NBODY * 10):
        crb.append(Scalar[DTYPE](0))
    for _ in range(NV * NV):
        M.append(Scalar[DTYPE](0))
        L.append(Scalar[DTYPE](0))
    for _ in range(NV):
        D.append(Scalar[DTYPE](0))

    from .jacobian import compute_cdof, compute_composite_inertia

    compute_cdof(model, data, cdof)
    compute_composite_inertia(model, data, crb)
    compute_mass_matrix_full[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        MAX_EQUALITY,
        CONE_TYPE,
        MAX_TENDON,
        NSITE,
    ](model, data, cdof, crb, M)

    # Add armature to diagonal before factoring
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var dof_adr = joint.dof_adr
        var arm = joint.armature
        var ndof = 1
        if joint.jnt_type == JNT_FREE:
            ndof = 6
        elif joint.jnt_type == JNT_BALL:
            ndof = 3
        for d in range(ndof):
            M[(dof_adr + d) * NV + (dof_adr + d)] += arm

    ldl_factor[DTYPE, NV](M, L, D)

    # Build dof_to_body mapping
    var dof_body = List[Int](capacity=NV)
    for _ in range(NV):
        dof_body.append(0)

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

    # World body: zero weights
    model.body_invweight0[0] = Scalar[DTYPE](0)
    model.body_invweight0[1] = Scalar[DTYPE](0)

    # For each non-world body, compute invweight0
    for i in range(NBODY):
        # Build 6×NV body CoM Jacobian for body i
        # Row k of J maps joint velocities to body i's CoM spatial velocity
        # Rows 0-2: linear (translation), Rows 3-5: angular (rotation)
        #
        # For each DOF d on the kinematic chain from root to body i:
        #   J_lin[d] = cdof_lin[d] + cdof_ang[d] × (xipos[i] - xipos[dof_body[d]])
        #   J_ang[d] = cdof_ang[d]

        # Target body CoM position
        var ti_x = data.xipos[i * 3 + 0]
        var ti_y = data.xipos[i * 3 + 1]
        var ti_z = data.xipos[i * 3 + 2]

        # We only need the 6 diagonal elements of A = J * M^{-1} * J^T
        # For diagonal A[k,k] = dot(J_row_k, M^{-1} * J_row_k)
        # We solve M * x_k = J_row_k for each k, then A[k,k] = dot(J_row_k, x_k)
        var A_diag = InlineArray[Scalar[DTYPE], 6](uninitialized=True)
        for k in range(6):
            A_diag[k] = Scalar[DTYPE](0)

        # Build J rows and solve systems
        # Process all 6 rows
        for k in range(6):
            var J_row = List[Scalar[DTYPE]](capacity=NV)
            for _ in range(NV):
                J_row.append(Scalar[DTYPE](0))

            # Fill J_row[d] for each DOF that affects body i
            for d in range(NV):
                var b = dof_body[d]
                # Check if DOF d affects body i (d's body is i or ancestor of i)
                var affects = False
                if b == i:
                    affects = True
                else:
                    var current = i
                    while current > 0:
                        if model.body_parent[current] == b:
                            affects = True
                            break
                        current = model.body_parent[current]

                if not affects:
                    continue

                var ang_x = cdof[d * 6 + 0]
                var ang_y = cdof[d * 6 + 1]
                var ang_z = cdof[d * 6 + 2]
                var lin_x = cdof[d * 6 + 3]
                var lin_y = cdof[d * 6 + 4]
                var lin_z = cdof[d * 6 + 5]

                # Offset from DOF's body CoM to target body CoM
                var dx = ti_x - data.xipos[b * 3 + 0]
                var dy = ti_y - data.xipos[b * 3 + 1]
                var dz = ti_z - data.xipos[b * 3 + 2]

                if k == 0:
                    # J_lin_x = cdof_lin_x + (ang_y*dz - ang_z*dy)
                    J_row[d] = lin_x + ang_y * dz - ang_z * dy
                elif k == 1:
                    # J_lin_y = cdof_lin_y + (ang_z*dx - ang_x*dz)
                    J_row[d] = lin_y + ang_z * dx - ang_x * dz
                elif k == 2:
                    # J_lin_z = cdof_lin_z + (ang_x*dy - ang_y*dx)
                    J_row[d] = lin_z + ang_x * dy - ang_y * dx
                elif k == 3:
                    # J_ang_x
                    J_row[d] = ang_x
                elif k == 4:
                    # J_ang_y
                    J_row[d] = ang_y
                else:
                    # J_ang_z
                    J_row[d] = ang_z

            # Solve M * x = J_row (convert to List for ldl_solve)
            var J_row_list = List[Scalar[DTYPE]](capacity=NV)
            var x_list = List[Scalar[DTYPE]](capacity=NV)
            for d in range(NV):
                J_row_list.append(J_row[d])
                x_list.append(Scalar[DTYPE](0))
            ldl_solve[DTYPE, NV](L, D, J_row_list, x_list)

            # A[k,k] = dot(J_row, x)
            var dot_val = Scalar[DTYPE](0)
            for d in range(NV):
                dot_val += J_row[d] * x_list[d]
            A_diag[k] = dot_val

        # Translation: average of A[0,0], A[1,1], A[2,2]
        var tran = (A_diag[0] + A_diag[1] + A_diag[2]) / Scalar[DTYPE](3)
        # Rotation: average of A[3,3], A[4,4], A[5,5]
        var rot = (A_diag[3] + A_diag[4] + A_diag[5]) / Scalar[DTYPE](3)

        # Fallback: if one is near-zero, use the other (MuJoCo behavior)
        if tran < Scalar[DTYPE](1e-10) and rot > Scalar[DTYPE](1e-10):
            tran = rot
        elif rot < Scalar[DTYPE](1e-10) and tran > Scalar[DTYPE](1e-10):
            rot = tran

        model.body_invweight0[2 * i] = tran
        model.body_invweight0[2 * i + 1] = rot

    # Compute dof_invweight0: diagonal of M^{-1}
    # For each DOF d, solve M * x = e_d, then dof_invweight0[d] = x[d]
    var e_dof = List[Scalar[DTYPE]](capacity=NV)
    var x_dof = List[Scalar[DTYPE]](capacity=NV)
    for _ in range(NV):
        e_dof.append(Scalar[DTYPE](0))
        x_dof.append(Scalar[DTYPE](0))
    for d in range(NV):
        for i in range(NV):
            e_dof[i] = Scalar[DTYPE](0)
            x_dof[i] = Scalar[DTYPE](0)
        e_dof[d] = Scalar[DTYPE](1)
        ldl_solve[DTYPE, NV](L, D, e_dof, x_dof)
        model.dof_invweight0[d] = x_dof[d]


# =============================================================================
# LDL Factorization and Solve for SPD matrices
# =============================================================================


def ldl_factor[
    DTYPE: DType, NV: Int
](
    M: List[Scalar[DTYPE]],
    mut L: List[Scalar[DTYPE]],
    mut D: List[Scalar[DTYPE]],
):
    """LDL factorization using heap-allocated List storage."""
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


def ldl_solve[
    DTYPE: DType, NV: Int
](
    L: List[Scalar[DTYPE]],
    D: List[Scalar[DTYPE]],
    b: List[Scalar[DTYPE]],
    mut x: List[Scalar[DTYPE]],
):
    """Solve M*x=b using LDL factors stored in heap-allocated Lists.

    Drop-in replacement for ldl_solve.
    """
    var y = List[Scalar[DTYPE]](capacity=NV)
    for _ in range(NV):
        y.append(Scalar[DTYPE](0))
    for i in range(NV):
        var s = b[i]
        for j in range(i):
            s = s - L[i * NV + j] * y[j]
        y[i] = s

    var z = List[Scalar[DTYPE]](capacity=NV)
    for _ in range(NV):
        z.append(Scalar[DTYPE](0))
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


def compute_M_inv_from_ldl[
    DTYPE: DType, NV: Int
](
    L: List[Scalar[DTYPE]],
    D: List[Scalar[DTYPE]],
    mut M_inv: List[Scalar[DTYPE]],
):
    """Compute full M^-1 from LDL factors stored in heap-allocated Lists."""
    var e = List[Scalar[DTYPE]](capacity=NV)
    var col = List[Scalar[DTYPE]](capacity=NV)
    for _ in range(NV):
        e.append(Scalar[DTYPE](0))
        col.append(Scalar[DTYPE](0))

    for j in range(NV):
        for i in range(NV):
            e[i] = Scalar[DTYPE](0)
        e[j] = Scalar[DTYPE](1)

        ldl_solve[DTYPE, NV](L, D, e, col)

        for i in range(NV):
            M_inv[i * NV + j] = col[i]


# =============================================================================
# Full Mass Matrix via CRBA (Composite Rigid Body Algorithm)
# =============================================================================


def compute_mass_matrix_full[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int = 0,
    MAX_EQUALITY: Int = 0,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    MAX_TENDON: Int = 0,
    NSITE: Int = 0,
](
    model: Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        MAX_EQUALITY,
        CONE_TYPE,
        MAX_TENDON,
        NSITE,
    ],
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
    cdof: List[Scalar[DTYPE]],
    crb: List[Scalar[DTYPE]],
    mut M: List[Scalar[DTYPE]],
):
    """Compute the full NV×NV mass matrix, storing result in a heap-allocated List.

    Drop-in replacement for compute_mass_matrix_full for scalability.
    """
    for i in range(NV * NV):
        M[i] = Scalar[DTYPE](0)

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

    comptime I_WORLD_SIZE = _ensure_positive[NBODY * 6]()
    var I_world = InlineArray[Scalar[DTYPE], I_WORLD_SIZE](uninitialized=True)
    for b in range(NBODY):
        var Ixx_l = model.body_inertia[b * 3 + 0]
        var Iyy_l = model.body_inertia[b * 3 + 1]
        var Izz_l = model.body_inertia[b * 3 + 2]

        var bqx = data.xquat[b * 4 + 0]
        var bqy = data.xquat[b * 4 + 1]
        var bqz = data.xquat[b * 4 + 2]
        var bqw = data.xquat[b * 4 + 3]
        var iqx = model.body_iquat[b * 4 + 0]
        var iqy = model.body_iquat[b * 4 + 1]
        var iqz = model.body_iquat[b * 4 + 2]
        var iqw = model.body_iquat[b * 4 + 3]
        var iq = quat_mul(bqx, bqy, bqz, bqw, iqx, iqy, iqz, iqw)
        var qx = iq[0]
        var qy = iq[1]
        var qz = iq[2]
        var qw = iq[3]

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

            var mij = Scalar[DTYPE](0)

            for k in range(NBODY):
                var in_subtree_i = (k == body_i) or _is_descendant(
                    model, k, body_i
                )
                if not in_subtree_i:
                    continue

                var in_subtree_j = (k == body_j) or _is_descendant(
                    model, k, body_j
                )
                if not in_subtree_j:
                    continue

                var mk = model.body_mass[k]
                var pk0 = data.xipos[k * 3 + 0]
                var pk1 = data.xipos[k * 3 + 1]
                var pk2 = data.xipos[k * 3 + 2]

                # Transport cdof velocity to body k's xipos.
                # Reference = subtree_com[rootid] or xipos[body] (legacy).
                var di0: Scalar[DTYPE]
                var di1: Scalar[DTYPE]
                var di2: Scalar[DTYPE]
                var dj0: Scalar[DTYPE]
                var dj1: Scalar[DTYPE]
                var dj2: Scalar[DTYPE]
                if data.has_subtree_com:
                    var root_i = model.body_rootid[body_i]
                    di0 = pk0 - data.subtree_com[root_i * 3 + 0]
                    di1 = pk1 - data.subtree_com[root_i * 3 + 1]
                    di2 = pk2 - data.subtree_com[root_i * 3 + 2]
                    var root_j = model.body_rootid[body_j]
                    dj0 = pk0 - data.subtree_com[root_j * 3 + 0]
                    dj1 = pk1 - data.subtree_com[root_j * 3 + 1]
                    dj2 = pk2 - data.subtree_com[root_j * 3 + 2]
                else:
                    di0 = pk0 - data.xipos[body_i * 3 + 0]
                    di1 = pk1 - data.xipos[body_i * 3 + 1]
                    di2 = pk2 - data.xipos[body_i * 3 + 2]
                    dj0 = pk0 - data.xipos[body_j * 3 + 0]
                    dj1 = pk1 - data.xipos[body_j * 3 + 1]
                    dj2 = pk2 - data.xipos[body_j * 3 + 2]

                var vki0 = li0 + ai1 * di2 - ai2 * di1
                var vki1 = li1 + ai2 * di0 - ai0 * di2
                var vki2 = li2 + ai0 * di1 - ai1 * di0

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

            M[i * NV + j] = mij
            if i != j:
                M[j * NV + i] = mij


# =============================================================================
# Sparse Mass Matrix (CSR format, MuJoCo-compatible)
# =============================================================================


struct SparseMassMatrix[
    DTYPE: DType,
    NV: Int,
    NM: Int,  # Non-zeros in lower triangle (incl. diagonal). Use NV*(NV+1)/2 as
    # safe maximum for any single kinematic chain. Smaller for branched trees.
]:
    """Sparse mass matrix in Compressed Sparse Row (CSR) format.

    Mirrors MuJoCo's qM/M sparse storage. Only the lower triangle (including
    diagonal) is stored. Sparsity pattern: M[i,j] != 0 (i >= j) iff dof_j's
    body is an ancestor of (or equal to) dof_i's body in the kinematic tree.

    After ldl_factor_sparse():
    - values[off-diagonal positions]: L[i,j] factors
    - values[diagonal positions]:     D[i] values
    - diag_inv[i]:                    1/D[i]  (stored separately for fast solve)

    Usage:
        # Determine NM for your model (or use NV*(NV+1)/2 as safe maximum):
        #   NM = NV * (NV + 1) / 2   # worst case: fully-connected chain
        var sM = SparseMassMatrix[DTYPE, NV, NM]()
        build_sparse_pattern(model, sM)                     # once at setup
        compute_mass_matrix_sparse(model, data, cdof, crb, sM)  # each step
        ldl_factor_sparse(sM)                               # each step
        ldl_solve_sparse(sM, b, x)                          # each solve
    """

    # CSR sparsity structure (set once by build_sparse_pattern)
    var row_nnz: InlineArray[Int, _ensure_positive[Self.NV]()]  # nnz per row
    var row_adr: InlineArray[
        Int, _ensure_positive[Self.NV]()
    ]  # start address in values/col_ind
    var col_ind: InlineArray[
        Int, _ensure_positive[Self.NM]()
    ]  # column indices (sorted asc per row)
    var actual_nnz: Int  # actual non-zeros (may be <= NM)

    # Values: M entries before factorization, LDL entries after
    var values: InlineArray[Scalar[Self.DTYPE], _ensure_positive[Self.NM]()]

    # 1/D[i] — set by ldl_factor_sparse for fast solve
    var diag_inv: InlineArray[Scalar[Self.DTYPE], _ensure_positive[Self.NV]()]

    def __init__(out self):
        self.row_nnz = InlineArray[Int, _ensure_positive[Self.NV]()](fill=0)
        self.row_adr = InlineArray[Int, _ensure_positive[Self.NV]()](fill=0)
        self.col_ind = InlineArray[Int, _ensure_positive[Self.NM]()](fill=0)
        self.actual_nnz = 0
        self.values = InlineArray[
            Scalar[Self.DTYPE], _ensure_positive[Self.NM]()
        ](fill=Scalar[Self.DTYPE](0))
        self.diag_inv = InlineArray[
            Scalar[Self.DTYPE], _ensure_positive[Self.NV]()
        ](fill=Scalar[Self.DTYPE](0))

    @always_inline
    def diag_pos(self, row: Int) -> Int:
        """Flat index of the diagonal entry M[row, row] in values/col_ind."""
        return self.row_adr[row] + self.row_nnz[row] - 1

    @always_inline
    def find_col(self, row: Int, col: Int) -> Int:
        """Return flat index of M[row, col], or -1 if not in pattern."""
        var adr = self.row_adr[row]
        var nnz = self.row_nnz[row]
        for k in range(nnz):
            var c = self.col_ind[adr + k]
            if c == col:
                return adr + k
            elif c > col:
                break
        return -1


def build_sparse_pattern[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NM: Int,
    NGEOM: Int = 0,
    MAX_EQUALITY: Int = 0,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    MAX_TENDON: Int = 0,
    NSITE: Int = 0,
](
    model: Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        MAX_EQUALITY,
        CONE_TYPE,
        MAX_TENDON,
        NSITE,
    ],
    mut sM: SparseMassMatrix[DTYPE, NV, NM],
):
    """Build the CSR sparsity pattern from the kinematic tree.

    Sets row_nnz, row_adr, col_ind, actual_nnz. Call once after model joints
    are configured (e.g., during model setup / mj_setConst equivalent).

    Pattern: M[i,j] != 0 (lower triangle, i >= j) iff body(dof_j) is an
    ancestor of (or equal to) body(dof_i). Equivalent to MuJoCo's precomputed
    M_rownnz, M_rowadr, M_colind.

    Args:
        model: Static model configuration (joints must be set up).
        sM:    SparseMassMatrix to fill; NM must be >= actual non-zeros.
    """
    comptime NV_SAFE = _ensure_positive[NV]()
    var dof_body = InlineArray[Int, NV_SAFE](fill=0)
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var ndof = 1
        if joint.jnt_type == JNT_FREE:
            ndof = 6
        elif joint.jnt_type == JNT_BALL:
            ndof = 3
        for d in range(ndof):
            dof_body[joint.dof_adr + d] = joint.body_id

    var total_nnz = 0
    for i in range(NV):
        var body_i = dof_body[i]
        sM.row_adr[i] = total_nnz
        var row_count = 0
        for j in range(i + 1):  # lower triangle + diagonal (j <= i)
            var body_j = dof_body[j]
            # M[i,j] != 0 iff body_j is ancestor of body_i (or they are equal)
            var connected = (body_j == body_i) or _is_descendant(
                model, body_i, body_j
            )
            if connected and total_nnz < NM:
                sM.col_ind[total_nnz] = j
                total_nnz += 1
                row_count += 1
        sM.row_nnz[i] = row_count

    sM.actual_nnz = total_nnz


def count_sparse_nnz[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int = 0,
    MAX_EQUALITY: Int = 0,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    MAX_TENDON: Int = 0,
    NSITE: Int = 0,
](
    model: Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        MAX_EQUALITY,
        CONE_TYPE,
        MAX_TENDON,
        NSITE,
    ],
) -> Int:
    """Count the actual number of non-zeros in the lower triangle for this model.

    Use the returned value as NM when instantiating SparseMassMatrix for
    minimal memory usage. For a single kinematic chain, this equals NV*(NV+1)/2.
    For branched trees, it is smaller.

    Example:
        # At compile time, use NV*(NV+1)/2 as safe upper bound:
        comptime NM = NV * (NV + 1) / 2
        # At runtime, you can verify: count_sparse_nnz(model) <= NM
    """
    comptime NV_SAFE = _ensure_positive[NV]()
    var dof_body = InlineArray[Int, NV_SAFE](fill=0)
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var ndof = 1
        if joint.jnt_type == JNT_FREE:
            ndof = 6
        elif joint.jnt_type == JNT_BALL:
            ndof = 3
        for d in range(ndof):
            dof_body[joint.dof_adr + d] = joint.body_id

    var total = 0
    for i in range(NV):
        var body_i = dof_body[i]
        for j in range(i + 1):
            var body_j = dof_body[j]
            if (body_j == body_i) or _is_descendant(model, body_i, body_j):
                total += 1
    return total


def compute_mass_matrix_sparse[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NM: Int,
    CDOF_SIZE: Int,
    CRB_SIZE: Int,
    NGEOM: Int = 0,
    MAX_EQUALITY: Int = 0,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    MAX_TENDON: Int = 0,
    NSITE: Int = 0,
](
    model: Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        MAX_EQUALITY,
        CONE_TYPE,
        MAX_TENDON,
        NSITE,
    ],
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
    cdof: List[Scalar[DTYPE]],
    crb: List[Scalar[DTYPE]],
    mut sM: SparseMassMatrix[DTYPE, NV, NM],
):
    """Compute the sparse mass matrix M(q) using the CSR pattern in sM.

    Fills sM.values with the mass matrix entries for all non-zero positions
    defined by the sparsity pattern. build_sparse_pattern() must be called
    first to set up row_nnz, row_adr, col_ind.

    Uses the same direct spatial algebra as compute_mass_matrix_full() but
    only stores/computes entries in the sparsity pattern — no wasted work on
    structurally-zero off-diagonal blocks (in branched models).

    Args:
        model: Static model configuration.
        data:  Current simulation state (xpos, xquat, xipos from FK).
        cdof:  Spatial motion axes per DOF (6*NV), from compute_cdof().
        crb:   Composite rigid body inertia (10*NBODY), from compute_composite_inertia().
        sM:    SparseMassMatrix with pattern already set.
    """
    # Zero values
    for k in range(sM.actual_nnz):
        sM.values[k] = Scalar[DTYPE](0)

    # Build dof_body mapping
    comptime NV_SAFE = _ensure_positive[NV]()
    var dof_body = InlineArray[Int, NV_SAFE](fill=0)
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var ndof = 1
        if joint.jnt_type == JNT_FREE:
            ndof = 6
        elif joint.jnt_type == JNT_BALL:
            ndof = 3
        for d in range(ndof):
            dof_body[joint.dof_adr + d] = joint.body_id

    # Pre-compute per-body world-frame inertia tensors [xx, yy, zz, xy, xz, yz]
    comptime I_WORLD_SIZE = _ensure_positive[NBODY * 6]()
    var I_world = InlineArray[Scalar[DTYPE], I_WORLD_SIZE](uninitialized=True)
    for b in range(NBODY):
        var Ixx_l = model.body_inertia[b * 3 + 0]
        var Iyy_l = model.body_inertia[b * 3 + 1]
        var Izz_l = model.body_inertia[b * 3 + 2]

        var bqx = data.xquat[b * 4 + 0]
        var bqy = data.xquat[b * 4 + 1]
        var bqz = data.xquat[b * 4 + 2]
        var bqw = data.xquat[b * 4 + 3]
        var iqx = model.body_iquat[b * 4 + 0]
        var iqy = model.body_iquat[b * 4 + 1]
        var iqz = model.body_iquat[b * 4 + 2]
        var iqw = model.body_iquat[b * 4 + 3]
        var iq = quat_mul(bqx, bqy, bqz, bqw, iqx, iqy, iqz, iqw)
        var qx = iq[0]
        var qy = iq[1]
        var qz = iq[2]
        var qw = iq[3]

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

    # Fill M[i,j] for each non-zero (i, j) in the sparsity pattern
    for i in range(NV):
        var body_i = dof_body[i]
        var ai0 = cdof[i * 6 + 0]
        var ai1 = cdof[i * 6 + 1]
        var ai2 = cdof[i * 6 + 2]
        var li0 = cdof[i * 6 + 3]
        var li1 = cdof[i * 6 + 4]
        var li2 = cdof[i * 6 + 5]

        var adr_i = sM.row_adr[i]
        var nnz_i = sM.row_nnz[i]

        for k_idx in range(nnz_i):
            var j = sM.col_ind[adr_i + k_idx]
            var body_j = dof_body[j]
            var aj0 = cdof[j * 6 + 0]
            var aj1 = cdof[j * 6 + 1]
            var aj2 = cdof[j * 6 + 2]
            var lj0 = cdof[j * 6 + 3]
            var lj1 = cdof[j * 6 + 4]
            var lj2 = cdof[j * 6 + 5]

            var mij = Scalar[DTYPE](0)

            # Sum over bodies k in subtree(body_i) ∩ subtree(body_j).
            # Since M[i,j] is non-zero, body_j is an ancestor of body_i,
            # so the intersection is subtree(body_i).
            for k in range(NBODY):
                var in_subtree_i = (k == body_i) or _is_descendant(
                    model, k, body_i
                )
                if not in_subtree_i:
                    continue
                var in_subtree_j = (k == body_j) or _is_descendant(
                    model, k, body_j
                )
                if not in_subtree_j:
                    continue

                var mk = model.body_mass[k]
                var pk0 = data.xipos[k * 3 + 0]
                var pk1 = data.xipos[k * 3 + 1]
                var pk2 = data.xipos[k * 3 + 2]

                # Transport cdof velocity to body k's xipos.
                # Reference = subtree_com[rootid] or xipos[body] (legacy).
                var di0: Scalar[DTYPE]
                var di1: Scalar[DTYPE]
                var di2: Scalar[DTYPE]
                var dj0: Scalar[DTYPE]
                var dj1: Scalar[DTYPE]
                var dj2: Scalar[DTYPE]
                if data.has_subtree_com:
                    var root_i = model.body_rootid[body_i]
                    di0 = pk0 - data.subtree_com[root_i * 3 + 0]
                    di1 = pk1 - data.subtree_com[root_i * 3 + 1]
                    di2 = pk2 - data.subtree_com[root_i * 3 + 2]
                    var root_j = model.body_rootid[body_j]
                    dj0 = pk0 - data.subtree_com[root_j * 3 + 0]
                    dj1 = pk1 - data.subtree_com[root_j * 3 + 1]
                    dj2 = pk2 - data.subtree_com[root_j * 3 + 2]
                else:
                    di0 = pk0 - data.xipos[body_i * 3 + 0]
                    di1 = pk1 - data.xipos[body_i * 3 + 1]
                    di2 = pk2 - data.xipos[body_i * 3 + 2]
                    dj0 = pk0 - data.xipos[body_j * 3 + 0]
                    dj1 = pk1 - data.xipos[body_j * 3 + 1]
                    dj2 = pk2 - data.xipos[body_j * 3 + 2]

                var vki0 = li0 + ai1 * di2 - ai2 * di1
                var vki1 = li1 + ai2 * di0 - ai0 * di2
                var vki2 = li2 + ai0 * di1 - ai1 * di0
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

            sM.values[adr_i + k_idx] = mij


def ldl_factor_sparse[
    DTYPE: DType,
    NV: Int,
    NM: Int,
](mut sM: SparseMassMatrix[DTYPE, NV, NM],):
    """In-place backward sparse LDL factorization — matches MuJoCo's mj_factorI.

    Processes rows from NV-1 down to 0 (leaf-to-root in the kinematic tree).
    This is a BACKWARD outer-product elimination that exploits the prefix-alignment
    property of the tree-ordered CSR structure to achieve zero fill-in.

    Prefix-alignment property: for any row i that appears in row k's column list
    (i is an ancestor of k), the first rownnz[i] column indices of row k are
    exactly row i's column indices. This holds for topologically-ordered kinematic
    trees (parent DOFs always have lower index than child DOFs).

    No fill-in guarantee: two DOFs i, j in different branches of the tree can never
    both be ancestors of the same DOF k. So the sparse structure remains unchanged
    throughout the factorization.

    After this call:
    - sM.values[diagonal]:    D[k]            (unchanged scaling factor)
    - sM.values[off-diagonal]: L[k,i] / D[k]  (stored as unit lower-tri factor / D)
    - sM.diag_inv[k]:          1 / D[k]

    The paired solve is ldl_solve_sparse() which uses the same convention.
    Equivalent to MuJoCo's mj_factorI + mj_factorM.
    """
    for k in range(NV - 1, -1, -1):  # backward: leaf to root
        var adr_k = sM.row_adr[k]
        var nnz_k = sM.row_nnz[k]
        var diag_k = adr_k + nnz_k - 1  # diagonal is the last entry in row k

        var D_k = sM.values[diag_k]
        if D_k < Scalar[DTYPE](1e-14):
            D_k = Scalar[DTYPE](1e-14)
        var invD_k = Scalar[DTYPE](1) / D_k
        sM.diag_inv[k] = invD_k

        # Update all ancestor rows of k using the outer-product formula:
        #   row_i -= mat[k,i] * invD[k] * row_k[0:rownnz[i]]
        # where row_k[0:rownnz[i]] == row_i's columns (prefix alignment).
        for adr_off in range(adr_k, diag_k):  # off-diagonal entries of row k
            var i = sM.col_ind[adr_off]  # i is an ancestor of k
            var scale = -sM.values[adr_off] * invD_k
            var adr_i = sM.row_adr[i]
            var nnz_i = sM.row_nnz[i]
            # The first nnz_i entries of row k share the same column indices as row i.
            for t in range(nnz_i):
                sM.values[adr_i + t] += scale * sM.values[adr_k + t]

        # Divide off-diagonals of row k by D[k]: store L[k,i] / D[k]
        for adr_off in range(adr_k, diag_k):
            sM.values[adr_off] *= invD_k


def ldl_solve_sparse[
    DTYPE: DType,
    NV: Int,
    NM: Int,
](
    sM: SparseMassMatrix[DTYPE, NV, NM],
    b: List[Scalar[DTYPE]],
    mut x: List[Scalar[DTYPE]],
):
    """Solve M * x = b using the sparse LDL factorization stored in sM.

    Equivalent to MuJoCo's mj_solveLD. Three phases matching the backward
    factorization convention (off-diagonals store L[k,i]/D[k]):

    Phase 1 (backward): x <- L^{-T} * b
        For i = NV-1 down to 0:
            x_i = x[i]
            For each off-diagonal (i, j): x[j] -= Lbar[i,j] * x_i

    Phase 2 (diagonal): x <- D^{-1} * x
        x[i] *= diag_inv[i]

    Phase 3 (forward): x <- L^{-1} * x
        For i = 0 to NV-1:
            For each off-diagonal (i, j): x[i] -= Lbar[i,j] * x[j]

    ldl_factor_sparse() must be called before this function.
    """
    # Initialize x = b
    for i in range(NV):
        x[i] = b[i]

    # --- Phase 1: Backward  x <- L^{-T} * x ---
    for i_rev in range(NV):
        var i = NV - 1 - i_rev
        var x_i = x[i]
        var adr_i = sM.row_adr[i]
        var nnz_i = sM.row_nnz[i]
        for t in range(nnz_i - 1):  # off-diagonal entries only (j < i)
            var j = sM.col_ind[adr_i + t]
            x[j] -= sM.values[adr_i + t] * x_i

    # --- Phase 2: Diagonal  x <- D^{-1} * x ---
    for i in range(NV):
        x[i] *= sM.diag_inv[i]

    # --- Phase 3: Forward  x <- L^{-1} * x ---
    for i in range(NV):
        var adr_i = sM.row_adr[i]
        var nnz_i = sM.row_nnz[i]
        for t in range(nnz_i - 1):  # off-diagonal entries only (j < i)
            var j = sM.col_ind[adr_i + t]
            x[i] -= sM.values[adr_i + t] * x[j]


def build_sparse_pattern_gpu[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NM: Int,
    MODEL_SIZE: Int,
](
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    mut row_nnz: InlineArray[Int, _ensure_positive[NV]()],
    mut row_adr: InlineArray[Int, _ensure_positive[NV]()],
    mut col_ind: InlineArray[Int, _ensure_positive[NM]()],
) -> Int:
    """Build sparse CSR pattern for M(q) from model buffer data.

    GPU-compatible version of build_sparse_pattern: reads joint/body topology
    from the model LayoutTensor instead of the CPU Model struct. Intended to be
    called at the start of each GPU step kernel so the pattern is always
    available as register-resident InlineArrays without any extra device memory.

    M[i,j] != 0 (for j <= i) iff body(dof_j) is an ancestor of (or equal to)
    body(dof_i) in the kinematic tree.

    Returns:
        Actual_nnz — the number of non-zero entries stored.
    """
    var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
    var num_joints = Int(
        rebind[Scalar[DTYPE]](model[0, model_meta_off + MODEL_META_IDX_NJOINT])
    )

    # Build dof_body mapping
    comptime NV_SAFE = _ensure_positive[NV]()
    var dof_body = InlineArray[Int, NV_SAFE](fill=0)
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

    # Build lower-triangle CSR: M[i,j] != 0 iff body_j is ancestor of body_i
    var actual_nnz = 0
    for i in range(NV):
        row_adr[i] = actual_nnz
        row_nnz[i] = 0
        var body_i = dof_body[i]
        for j in range(i + 1):  # j <= i (lower triangle including diagonal)
            var body_j = dof_body[j]
            # body_j is an ancestor-or-equal of body_i if body_i is in
            # the subtree rooted at body_j
            var connected = (body_j == body_i) or _is_descendant_gpu[
                DTYPE, NBODY, MODEL_SIZE
            ](model, body_i, body_j)
            if connected:
                col_ind[actual_nnz] = j
                actual_nnz += 1
                row_nnz[i] += 1

    return actual_nnz


def sparse_to_dense[
    DTYPE: DType,
    NV: Int,
    NM: Int,
](sM: SparseMassMatrix[DTYPE, NV, NM], mut M: List[Scalar[DTYPE]],):
    """Expand sparse lower-triangle mass matrix to full dense NV×NV matrix.

    Equivalent to MuJoCo's mj_fullM: expands qM (sparse) to a dense matrix.
    Reads sM.values (which must contain M values, NOT LDL factors).

    Args:
        sM: SparseMassMatrix with values containing M entries (before factorization).
        M:  Output dense NV×NV row-major matrix (M_SIZE = NV*NV).
    """
    for k in range(NV * NV):
        M[k] = Scalar[DTYPE](0)

    for i in range(NV):
        var adr_i = sM.row_adr[i]
        var nnz_i = sM.row_nnz[i]
        for t in range(nnz_i):
            var j = sM.col_ind[adr_i + t]
            var val = sM.values[adr_i + t]
            M[i * NV + j] = val
            if i != j:
                M[j * NV + i] = val  # symmetric fill


# =============================================================================
# GPU: Full Mass Matrix + LDL
# =============================================================================


@always_inline
def compute_mass_matrix_full_gpu[
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
    var xipos_off = xipos_offset[NQ, NV, NBODY]()
    var stcom_off_mm = subtree_com_offset[NQ, NV, NBODY, MAX_CONTACTS]()

    # Pre-compute per-body world-frame inertia tensor
    comptime I_WORLD_SIZE = _ensure_positive[NBODY * 6]()
    var I_world = InlineArray[Scalar[DTYPE], I_WORLD_SIZE](uninitialized=True)
    for b in range(NBODY):
        var body_off = model_body_offset(b)
        var Ixx_l = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IXX])
        var Iyy_l = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IYY])
        var Izz_l = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IZZ])

        # Compose xquat with body_iquat for inertia rotation
        var bqx = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 0])
        var bqy = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 1])
        var bqz = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 2])
        var bqw = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 3])
        var iqx = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IQUAT_X])
        var iqy = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IQUAT_Y])
        var iqz = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IQUAT_Z])
        var iqw = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IQUAT_W])
        var iq = gpu_quat_mul(bqx, bqy, bqz, bqw, iqx, iqy, iqz, iqw)
        var qx = iq[0]
        var qy = iq[1]
        var qz = iq[2]
        var qw = iq[3]

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

    # Pre-compute subtree membership: subtree_mask[k * NBODY + b] = True iff
    # body k is in the subtree rooted at body b (i.e., k == b or k descends from b).
    # This replaces O(depth) parent-chain walks with O(1) lookups in the inner loop.
    comptime MASK_SIZE = _ensure_positive[NBODY * NBODY]()
    var subtree_mask = InlineArray[Bool, MASK_SIZE](fill=False)
    for k in range(NBODY):
        subtree_mask[k * NBODY + k] = True  # body is always in its own subtree
        # Walk parent chain from k upward, marking ancestors
        var current = k
        while current > 0:
            var body_off_c = model_body_offset(current)
            var parent = Int(
                rebind[Scalar[DTYPE]](model[0, body_off_c + BODY_IDX_PARENT])
            )
            subtree_mask[k * NBODY + parent] = True
            current = parent

    # Compute M[i,j] using direct body summation with subtree mask lookup
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
                # O(1) subtree check replaces O(depth) parent-chain walk
                if not subtree_mask[k * NBODY + body_i]:
                    continue
                if not subtree_mask[k * NBODY + body_j]:
                    continue

                var body_off_k = model_body_offset(k)
                var mk = rebind[Scalar[DTYPE]](
                    model[0, body_off_k + BODY_IDX_MASS]
                )
                var pk0 = rebind[Scalar[DTYPE]](
                    state[env, xipos_off + k * 3 + 0]
                )
                var pk1 = rebind[Scalar[DTYPE]](
                    state[env, xipos_off + k * 3 + 1]
                )
                var pk2 = rebind[Scalar[DTYPE]](
                    state[env, xipos_off + k * 3 + 2]
                )

                # Velocity transport: use subtree_com[rootid] as reference
                var ri_off = model_body_offset(body_i)
                var ri_root = Int(rebind[Scalar[DTYPE]](model[0, ri_off + BODY_IDX_ROOTID]))
                var pi0 = rebind[Scalar[DTYPE]](state[env, stcom_off_mm + ri_root * 3 + 0])
                var pi1 = rebind[Scalar[DTYPE]](state[env, stcom_off_mm + ri_root * 3 + 1])
                var pi2 = rebind[Scalar[DTYPE]](state[env, stcom_off_mm + ri_root * 3 + 2])
                var di0 = pk0 - pi0
                var di1 = pk1 - pi1
                var di2 = pk2 - pi2
                var vki0 = li0 + ai1 * di2 - ai2 * di1
                var vki1 = li1 + ai2 * di0 - ai0 * di2
                var vki2 = li2 + ai0 * di1 - ai1 * di0

                var rj_off = model_body_offset(body_j)
                var rj_root = Int(rebind[Scalar[DTYPE]](model[0, rj_off + BODY_IDX_ROOTID]))
                var pj0 = rebind[Scalar[DTYPE]](state[env, stcom_off_mm + rj_root * 3 + 0])
                var pj1 = rebind[Scalar[DTYPE]](state[env, stcom_off_mm + rj_root * 3 + 1])
                var pj2 = rebind[Scalar[DTYPE]](state[env, stcom_off_mm + rj_root * 3 + 2])
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
def compute_mass_matrix_full_gpu_mt[
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
    tid: Int,
    n_threads: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    workspace: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ],
):
    """Compute full NV×NV mass matrix on GPU — multi-threaded variant.

    Same algorithm as compute_mass_matrix_full_gpu but distributes work across
    n_threads threads. Each thread handles rows i where i % n_threads == tid.
    All threads redundantly compute I_world and subtree_mask (read-only from
    model/state data) to avoid extra barriers.
    """

    # Derive pointers from workspace (MutAnyOrigin)
    comptime cdof_idx = ws_cdof_offset()
    comptime M_idx = ws_M_offset[NV, NBODY]()

    var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
    var num_joints = Int(
        rebind[Scalar[DTYPE]](model[0, model_meta_off + MODEL_META_IDX_NJOINT])
    )

    # Zero M — distributed across threads
    for i in range(tid, NV * NV, n_threads):
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
    var xipos_off = xipos_offset[NQ, NV, NBODY]()
    var stcom_off_mm = subtree_com_offset[NQ, NV, NBODY, MAX_CONTACTS]()

    # Pre-compute per-body world-frame inertia tensor (all threads redundantly)
    comptime I_WORLD_SIZE = _ensure_positive[NBODY * 6]()
    var I_world = InlineArray[Scalar[DTYPE], I_WORLD_SIZE](uninitialized=True)
    for b in range(NBODY):
        var body_off = model_body_offset(b)
        var Ixx_l = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IXX])
        var Iyy_l = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IYY])
        var Izz_l = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IZZ])

        # Compose xquat with body_iquat for inertia rotation
        var bqx = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 0])
        var bqy = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 1])
        var bqz = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 2])
        var bqw = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 3])
        var iqx = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IQUAT_X])
        var iqy = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IQUAT_Y])
        var iqz = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IQUAT_Z])
        var iqw = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IQUAT_W])
        var iq = gpu_quat_mul(bqx, bqy, bqz, bqw, iqx, iqy, iqz, iqw)
        var qx = iq[0]
        var qy = iq[1]
        var qz = iq[2]
        var qw = iq[3]

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

    # Pre-compute subtree membership (all threads redundantly)
    comptime MASK_SIZE = _ensure_positive[NBODY * NBODY]()
    var subtree_mask = InlineArray[Bool, MASK_SIZE](fill=False)
    for k in range(NBODY):
        subtree_mask[k * NBODY + k] = True
        var current = k
        while current > 0:
            var body_off_c = model_body_offset(current)
            var parent = Int(
                rebind[Scalar[DTYPE]](model[0, body_off_c + BODY_IDX_PARENT])
            )
            subtree_mask[k * NBODY + parent] = True
            current = parent

    # Compute M[i,j] — each thread handles rows where i % n_threads == tid
    for i in range(tid, NV, n_threads):
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
                if not subtree_mask[k * NBODY + body_i]:
                    continue
                if not subtree_mask[k * NBODY + body_j]:
                    continue

                var body_off_k = model_body_offset(k)
                var mk = rebind[Scalar[DTYPE]](
                    model[0, body_off_k + BODY_IDX_MASS]
                )
                var pk0 = rebind[Scalar[DTYPE]](
                    state[env, xipos_off + k * 3 + 0]
                )
                var pk1 = rebind[Scalar[DTYPE]](
                    state[env, xipos_off + k * 3 + 1]
                )
                var pk2 = rebind[Scalar[DTYPE]](
                    state[env, xipos_off + k * 3 + 2]
                )

                # Velocity transport: use subtree_com[rootid] as reference
                var ri_off = model_body_offset(body_i)
                var ri_root = Int(rebind[Scalar[DTYPE]](model[0, ri_off + BODY_IDX_ROOTID]))
                var pi0 = rebind[Scalar[DTYPE]](state[env, stcom_off_mm + ri_root * 3 + 0])
                var pi1 = rebind[Scalar[DTYPE]](state[env, stcom_off_mm + ri_root * 3 + 1])
                var pi2 = rebind[Scalar[DTYPE]](state[env, stcom_off_mm + ri_root * 3 + 2])
                var di0 = pk0 - pi0
                var di1 = pk1 - pi1
                var di2 = pk2 - pi2
                var vki0 = li0 + ai1 * di2 - ai2 * di1
                var vki1 = li1 + ai2 * di0 - ai0 * di2
                var vki2 = li2 + ai0 * di1 - ai1 * di0

                var rj_off = model_body_offset(body_j)
                var rj_root = Int(rebind[Scalar[DTYPE]](model[0, rj_off + BODY_IDX_ROOTID]))
                var pj0 = rebind[Scalar[DTYPE]](state[env, stcom_off_mm + rj_root * 3 + 0])
                var pj1 = rebind[Scalar[DTYPE]](state[env, stcom_off_mm + rj_root * 3 + 1])
                var pj2 = rebind[Scalar[DTYPE]](state[env, stcom_off_mm + rj_root * 3 + 2])
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
def ldl_factor_gpu[
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

        if d_j > 1e-14 or d_j < -1e-14:
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
def ldl_solve_gpu[
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
def ldl_solve_workspace_gpu[
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
def compute_M_inv_from_ldl_gpu[
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


@always_inline
def compute_M_inv_from_ldl_gpu_mt[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    BATCH: Int,
    WS_SIZE: Int,
](
    env: Int,
    tid: Int,
    n_threads: Int,
    workspace: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ],
):
    """Multi-threaded dense M^-1 from LDL factors. Each column j of M^-1 is an
    independent triangular solve, so thread `tid` handles columns
    j where j % n_threads == tid. Bit-identical to compute_M_inv_from_ldl_gpu
    (same per-column arithmetic). Caller must barrier() before (LDL factors
    ready) and after (all columns written). Uses the idle STEP_THREADS threads
    in the RK4 stage kernel instead of computing all NV columns on tid 0.
    """
    from ..gpu.constants import ws_L_offset, ws_D_offset, ws_m_inv_offset

    comptime L_idx = ws_L_offset[NV, NBODY]()
    comptime D_idx = ws_D_offset[NV, NBODY]()
    comptime M_inv_idx = ws_m_inv_offset[NV, NBODY]()

    comptime V_SIZE = _ensure_positive[NV]()
    var e = InlineArray[workspace.element_type, V_SIZE](uninitialized=True)
    var col = InlineArray[workspace.element_type, V_SIZE](uninitialized=True)

    for j in range(tid, NV, n_threads):
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
# GPU: Sparse Mass Matrix + LDL (matching MuJoCo mj_factorI / mj_solveLD)
# =============================================================================


@always_inline
def compute_mass_matrix_sparse_gpu[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NM: Int,
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
    row_nnz: InlineArray[Int, _ensure_positive[NV]()],
    row_adr: InlineArray[Int, _ensure_positive[NV]()],
    col_ind: InlineArray[Int, _ensure_positive[NM]()],
):
    """Compute sparse mass matrix values on GPU.

    Fills the NM sparse entries of M(q) using the pre-built sparsity pattern.
    Values are stored at ws_M_offset[NV, NBODY]() in workspace (first NM
    of the NV*NV-allocated slot).

    cdof must already be in workspace (written by compute_cdof_gpu).
    The sparsity pattern (row_nnz, row_adr, col_ind) must be pre-built on CPU
    via build_sparse_pattern() and passed here as InlineArrays — they are
    identical for every environment and every timestep.

    Compared to the dense compute_mass_matrix_full_gpu, this skips all
    structurally-zero off-diagonal blocks (e.g., the cross-leg entries in
    HalfCheetah), reducing work from O(NV²) to O(NM) where NM ≤ NV*(NV+1)/2.
    """
    comptime cdof_idx = ws_cdof_offset()
    comptime M_idx = ws_M_offset[NV, NBODY]()

    var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
    var num_joints = Int(
        rebind[Scalar[DTYPE]](model[0, model_meta_off + MODEL_META_IDX_NJOINT])
    )

    # Zero NM sparse values
    for k in range(NM):
        workspace[env, M_idx + k] = 0

    # Build dof_body mapping
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

    var xquat_off = xquat_offset[NQ, NV, NBODY]()
    var xipos_off = xipos_offset[NQ, NV, NBODY]()
    var stcom_off_mm = subtree_com_offset[NQ, NV, NBODY, MAX_CONTACTS]()

    # Pre-compute per-body world-frame inertia tensors [xx, yy, zz, xy, xz, yz]
    comptime I_WORLD_SIZE = _ensure_positive[NBODY * 6]()
    var I_world = InlineArray[Scalar[DTYPE], I_WORLD_SIZE](uninitialized=True)
    for b in range(NBODY):
        var body_off = model_body_offset(b)
        var Ixx_l = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IXX])
        var Iyy_l = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IYY])
        var Izz_l = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IZZ])

        var bqx = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 0])
        var bqy = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 1])
        var bqz = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 2])
        var bqw = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 3])
        var iqx = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IQUAT_X])
        var iqy = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IQUAT_Y])
        var iqz = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IQUAT_Z])
        var iqw = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IQUAT_W])
        var iq = gpu_quat_mul(bqx, bqy, bqz, bqw, iqx, iqy, iqz, iqw)
        var qx = iq[0]
        var qy = iq[1]
        var qz = iq[2]
        var qw = iq[3]

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

    # Fill M[i,j] for each non-zero (i, j) in the sparsity pattern
    for i in range(NV):
        var body_i = dof_body[i]
        var ai0 = workspace[env, cdof_idx + i * 6 + 0]
        var ai1 = workspace[env, cdof_idx + i * 6 + 1]
        var ai2 = workspace[env, cdof_idx + i * 6 + 2]
        var li0 = workspace[env, cdof_idx + i * 6 + 3]
        var li1 = workspace[env, cdof_idx + i * 6 + 4]
        var li2 = workspace[env, cdof_idx + i * 6 + 5]

        var adr_i = row_adr[i]
        var nnz_i = row_nnz[i]

        for k_idx in range(nnz_i):
            var j = col_ind[adr_i + k_idx]
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
                    state[env, xipos_off + k * 3 + 0]
                )
                var pk1 = rebind[Scalar[DTYPE]](
                    state[env, xipos_off + k * 3 + 1]
                )
                var pk2 = rebind[Scalar[DTYPE]](
                    state[env, xipos_off + k * 3 + 2]
                )

                # Velocity transport: use subtree_com[rootid] as reference
                var ri_off = model_body_offset(body_i)
                var ri_root = Int(rebind[Scalar[DTYPE]](model[0, ri_off + BODY_IDX_ROOTID]))
                var pi0 = rebind[Scalar[DTYPE]](state[env, stcom_off_mm + ri_root * 3 + 0])
                var pi1 = rebind[Scalar[DTYPE]](state[env, stcom_off_mm + ri_root * 3 + 1])
                var pi2 = rebind[Scalar[DTYPE]](state[env, stcom_off_mm + ri_root * 3 + 2])
                var di0 = pk0 - pi0
                var di1 = pk1 - pi1
                var di2 = pk2 - pi2
                var vki0 = li0 + ai1 * di2 - ai2 * di1
                var vki1 = li1 + ai2 * di0 - ai0 * di2
                var vki2 = li2 + ai0 * di1 - ai1 * di0

                var rj_off = model_body_offset(body_j)
                var rj_root = Int(rebind[Scalar[DTYPE]](model[0, rj_off + BODY_IDX_ROOTID]))
                var pj0 = rebind[Scalar[DTYPE]](state[env, stcom_off_mm + rj_root * 3 + 0])
                var pj1 = rebind[Scalar[DTYPE]](state[env, stcom_off_mm + rj_root * 3 + 1])
                var pj2 = rebind[Scalar[DTYPE]](state[env, stcom_off_mm + rj_root * 3 + 2])
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

            workspace[env, M_idx + adr_i + k_idx] = mij


@always_inline
def ldl_factor_sparse_gpu[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NM: Int,
    BATCH: Int,
    WS_SIZE: Int,
](
    env: Int,
    workspace: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ],
    row_nnz: InlineArray[Int, _ensure_positive[NV]()],
    row_adr: InlineArray[Int, _ensure_positive[NV]()],
    col_ind: InlineArray[Int, _ensure_positive[NM]()],
):
    """Backward sparse LDL factorization on GPU — matches MuJoCo's mj_factorI.

    Reads NM sparse M values from ws_M_offset, factors in-place (backward
    outer-product, leaf-to-root). Writes 1/D[k] to ws_D_offset[k].

    After this call, ws_M_offset holds the factored lower triangle:
      - diagonal entries: D[k]  (unchanged pivot)
      - off-diagonal:     L[k,i] / D[k]
    ws_D_offset holds diag_inv[k] = 1 / D[k].

    Zero fill-in guaranteed by the prefix-alignment property of kinematic trees.
    """
    comptime M_idx = ws_M_offset[NV, NBODY]()
    comptime D_idx = ws_D_offset[NV, NBODY]()

    for k in range(NV - 1, -1, -1):  # backward: leaf to root
        var adr_k = row_adr[k]
        var nnz_k = row_nnz[k]
        var diag_k = adr_k + nnz_k - 1  # diagonal is the last entry in row k

        var D_k = workspace[env, M_idx + diag_k]
        if D_k < 1e-14:
            D_k = 1e-14
        var invD_k = Scalar[DTYPE](1) / rebind[Scalar[DTYPE]](D_k)
        workspace[env, D_idx + k] = invD_k

        # Update each ancestor row i of k: row_i -= L[k,i]*invD[k] * row_k[0:nnz_i]
        for adr_off in range(adr_k, diag_k):
            var i = col_ind[adr_off]
            var scale = (
                -rebind[Scalar[DTYPE]](workspace[env, M_idx + adr_off]) * invD_k
            )
            var adr_i = row_adr[i]
            var nnz_i = row_nnz[i]
            for t in range(nnz_i):
                workspace[env, M_idx + adr_i + t] = (
                    workspace[env, M_idx + adr_i + t]
                    + scale * workspace[env, M_idx + adr_k + t]
                )

        # Divide off-diagonals of row k by D[k]: store L[k,i] / D[k]
        for adr_off in range(adr_k, diag_k):
            workspace[env, M_idx + adr_off] = (
                workspace[env, M_idx + adr_off] * invD_k
            )


@always_inline
def ldl_solve_sparse_gpu[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NM: Int,
    BATCH: Int,
    WS_SIZE: Int,
](
    env: Int,
    workspace: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ],
    row_nnz: InlineArray[Int, _ensure_positive[NV]()],
    row_adr: InlineArray[Int, _ensure_positive[NV]()],
    col_ind: InlineArray[Int, _ensure_positive[NM]()],
):
    """Sparse LDL solve on GPU — matches MuJoCo's mj_solveLD.

    Reads f_net (RHS) from ws_fnet_offset, writes qacc (solution) to
    ws_qacc_ws_offset. Reads the factored sparse M from ws_M_offset and
    diag_inv from ws_D_offset (written by ldl_factor_sparse_gpu).

    Three phases (off-diagonal entries store L[k,i]/D[k]):
      Phase 1 (backward):  x <- L^{-T} * b
      Phase 2 (diagonal):  x <- D^{-1} * x
      Phase 3 (forward):   x <- L^{-1} * x
    """
    from ..gpu.constants import ws_fnet_offset, ws_qacc_ws_offset

    comptime M_idx = ws_M_offset[NV, NBODY]()
    comptime D_idx = ws_D_offset[NV, NBODY]()
    comptime b_idx = ws_fnet_offset[NV, NBODY]()
    comptime x_idx = ws_qacc_ws_offset[NV, NBODY]()

    comptime NV_SAFE = _ensure_positive[NV]()
    var x = InlineArray[Scalar[DTYPE], NV_SAFE](uninitialized=True)

    # Initialize x = b
    for i in range(NV):
        x[i] = rebind[Scalar[DTYPE]](workspace[env, b_idx + i])

    # Phase 1: Backward  x <- L^{-T} * x
    for i_rev in range(NV):
        var i = NV - 1 - i_rev
        var x_i = x[i]
        var adr_i = row_adr[i]
        var nnz_i = row_nnz[i]
        for t in range(nnz_i - 1):  # off-diagonal entries only (j < i)
            var j = col_ind[adr_i + t]
            x[j] = (
                x[j]
                - rebind[Scalar[DTYPE]](workspace[env, M_idx + adr_i + t]) * x_i
            )

    # Phase 2: Diagonal  x <- D^{-1} * x
    for i in range(NV):
        x[i] = x[i] * rebind[Scalar[DTYPE]](workspace[env, D_idx + i])

    # Phase 3: Forward  x <- L^{-1} * x
    for i in range(NV):
        var adr_i = row_adr[i]
        var nnz_i = row_nnz[i]
        for t in range(nnz_i - 1):  # off-diagonal entries only (j < i)
            var j = col_ind[adr_i + t]
            x[i] = (
                x[i]
                - rebind[Scalar[DTYPE]](workspace[env, M_idx + adr_i + t])
                * x[j]
            )

    for i in range(NV):
        workspace[env, x_idx + i] = x[i]


@always_inline
def compute_M_inv_from_sparse_ldl_gpu[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NM: Int,
    BATCH: Int,
    WS_SIZE: Int,
](
    env: Int,
    workspace: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ],
    row_nnz: InlineArray[Int, _ensure_positive[NV]()],
    row_adr: InlineArray[Int, _ensure_positive[NV]()],
    col_ind: InlineArray[Int, _ensure_positive[NM]()],
):
    """Compute full NV×NV M_inv from sparse LDL factors on GPU.

    Reads factored sparse M from ws_M_offset, diag_inv from ws_D_offset.
    Writes M_inv (NV×NV) to ws_m_inv_offset.

    Solves M * e_j = e_j for each basis vector j using the 3-phase sparse
    solve, then stores the result as column j of M_inv. M_inv is dense
    (needed for constraint-level forces that require the full inverse).
    """
    from ..gpu.constants import ws_m_inv_offset

    comptime M_idx = ws_M_offset[NV, NBODY]()
    comptime D_idx = ws_D_offset[NV, NBODY]()
    comptime M_inv_idx = ws_m_inv_offset[NV, NBODY]()

    comptime NV_SAFE = _ensure_positive[NV]()
    var x = InlineArray[Scalar[DTYPE], NV_SAFE](uninitialized=True)

    for j in range(NV):
        # Initialize x = e_j (unit vector)
        for i in range(NV):
            x[i] = Scalar[DTYPE](0)
        x[j] = Scalar[DTYPE](1)

        # Phase 1: Backward  x <- L^{-T} * x
        for i_rev in range(NV):
            var i = NV - 1 - i_rev
            var x_i = x[i]
            var adr_i = row_adr[i]
            var nnz_i = row_nnz[i]
            for t in range(nnz_i - 1):
                var m = col_ind[adr_i + t]
                x[m] = (
                    x[m]
                    - rebind[Scalar[DTYPE]](workspace[env, M_idx + adr_i + t])
                    * x_i
                )

        # Phase 2: Diagonal  x <- D^{-1} * x
        for i in range(NV):
            x[i] = x[i] * rebind[Scalar[DTYPE]](workspace[env, D_idx + i])

        # Phase 3: Forward  x <- L^{-1} * x
        for i in range(NV):
            var adr_i = row_adr[i]
            var nnz_i = row_nnz[i]
            for t in range(nnz_i - 1):
                var m = col_ind[adr_i + t]
                x[i] = (
                    x[i]
                    - rebind[Scalar[DTYPE]](workspace[env, M_idx + adr_i + t])
                    * x[m]
                )

        # Store column j of M_inv
        for i in range(NV):
            workspace[env, M_inv_idx + i * NV + j] = x[i]


# =============================================================================
# Helper: Solve M * x = b (for small matrices)
# =============================================================================


def solve_linear_1x1[
    DTYPE: DType
](M: Scalar[DTYPE], b: Scalar[DTYPE],) -> Scalar[DTYPE]:
    """Solve 1x1 system: M * x = b."""
    if M > Scalar[DTYPE](1e-10) or M < Scalar[DTYPE](-1e-10):
        return b / M
    return Scalar[DTYPE](0)


def solve_linear_diagonal[
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
def _is_descendant_gpu[
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
    while current > 0:
        var body_off = model_body_offset(current)
        var parent = Int(
            rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_PARENT])
        )
        if parent == ancestor:
            return True
        current = parent
    return False


@always_inline
def compute_mass_matrix_diagonal_gpu[
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
    var xipos_off = xipos_offset[NQ, NV, NBODY]()
    var stcom_off_mm = subtree_com_offset[NQ, NV, NBODY, MAX_CONTACTS]()

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
                state[env, xipos_off + body_id * 3 + 0]
            )
            var body_py = rebind[Scalar[DTYPE]](
                state[env, xipos_off + body_id * 3 + 1]
            )
            var body_pz = rebind[Scalar[DTYPE]](
                state[env, xipos_off + body_id * 3 + 2]
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
                        state[env, xipos_off + desc_body * 3 + 0]
                    )
                    var desc_py = rebind[Scalar[DTYPE]](
                        state[env, xipos_off + desc_body * 3 + 1]
                    )
                    var desc_pz = rebind[Scalar[DTYPE]](
                        state[env, xipos_off + desc_body * 3 + 2]
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

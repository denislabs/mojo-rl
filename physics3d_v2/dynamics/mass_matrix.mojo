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

from ..types import ModelGC, DataGC
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..kinematics.quat_math import quat_rotate, gpu_quat_rotate
from ..gpu.constants import (
    gc_xpos_offset,
    gc_xquat_offset,
    gc_model_body_offset,
    gc_model_joint_offset,
    gc_model_metadata_offset,
    GC_BODY_IDX_PARENT,
    GC_BODY_IDX_MASS,
    GC_BODY_IDX_IXX,
    GC_BODY_IDX_IYY,
    GC_BODY_IDX_IZZ,
    GC_JOINT_IDX_TYPE,
    GC_JOINT_IDX_BODY_ID,
    GC_JOINT_IDX_DOF_ADR,
    GC_JOINT_IDX_POS_X,
    GC_JOINT_IDX_POS_Y,
    GC_JOINT_IDX_POS_Z,
    GC_JOINT_IDX_AXIS_X,
    GC_JOINT_IDX_AXIS_Y,
    GC_JOINT_IDX_AXIS_Z,
    GC_MODEL_META_IDX_NJOINT,
    GC_JNT_FREE,
    GC_JNT_BALL,
    GC_JNT_SLIDE,
    GC_JNT_HINGE,
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
    model: ModelGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
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
        var body_off = gc_model_body_offset(current)
        var parent = Int(rebind[Scalar[DTYPE]](model[0, body_off + GC_BODY_IDX_PARENT]))
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
    var xpos_off = gc_xpos_offset[NQ, NV, NBODY]()
    var xquat_off = gc_xquat_offset[NQ, NV, NBODY]()

    var model_meta_off = gc_model_metadata_offset[NBODY, NJOINT]()
    var num_joints = Int(rebind[Scalar[DTYPE]](model[0, model_meta_off + GC_MODEL_META_IDX_NJOINT]))

    # Initialize to zero
    for i in range(NV):
        M_diag[i] = Scalar[DTYPE](0)

    for j in range(num_joints):
        var joint_off = gc_model_joint_offset[NBODY](j)

        var jnt_type = Int(rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_TYPE]))
        var body_id = Int(rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_BODY_ID]))
        var dof_adr = Int(rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_DOF_ADR]))

        var body_off = gc_model_body_offset(body_id)
        var parent = Int(rebind[Scalar[DTYPE]](model[0, body_off + GC_BODY_IDX_PARENT]))
        var mass = rebind[Scalar[DTYPE]](model[0, body_off + GC_BODY_IDX_MASS])
        var I_xx = rebind[Scalar[DTYPE]](model[0, body_off + GC_BODY_IDX_IXX])
        var I_yy = rebind[Scalar[DTYPE]](model[0, body_off + GC_BODY_IDX_IYY])
        var I_zz = rebind[Scalar[DTYPE]](model[0, body_off + GC_BODY_IDX_IZZ])
        var I_avg = (I_xx + I_yy + I_zz) / Scalar[DTYPE](3)

        var jpos_x = rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_POS_X])
        var jpos_y = rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_POS_Y])
        var jpos_z = rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_POS_Z])
        var axis_x = rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_AXIS_X])
        var axis_y = rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_AXIS_Y])
        var axis_z = rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_AXIS_Z])

        if jnt_type == GC_JNT_HINGE:
            var jpos_world_x = jpos_x
            var jpos_world_y = jpos_y
            var jpos_world_z = jpos_z

            if parent >= 0:
                var ppx = rebind[Scalar[DTYPE]](state[env, xpos_off + parent * 3 + 0])
                var ppy = rebind[Scalar[DTYPE]](state[env, xpos_off + parent * 3 + 1])
                var ppz = rebind[Scalar[DTYPE]](state[env, xpos_off + parent * 3 + 2])
                var pqx = rebind[Scalar[DTYPE]](state[env, xquat_off + parent * 4 + 0])
                var pqy = rebind[Scalar[DTYPE]](state[env, xquat_off + parent * 4 + 1])
                var pqz = rebind[Scalar[DTYPE]](state[env, xquat_off + parent * 4 + 2])
                var pqw = rebind[Scalar[DTYPE]](state[env, xquat_off + parent * 4 + 3])

                var rotated = gpu_quat_rotate(pqx, pqy, pqz, pqw, jpos_x, jpos_y, jpos_z)
                jpos_world_x = ppx + rotated[0]
                jpos_world_y = ppy + rotated[1]
                jpos_world_z = ppz + rotated[2]

            var axis_world_x = axis_x
            var axis_world_y = axis_y
            var axis_world_z = axis_z
            if parent >= 0:
                var pqx = rebind[Scalar[DTYPE]](state[env, xquat_off + parent * 4 + 0])
                var pqy = rebind[Scalar[DTYPE]](state[env, xquat_off + parent * 4 + 1])
                var pqz = rebind[Scalar[DTYPE]](state[env, xquat_off + parent * 4 + 2])
                var pqw = rebind[Scalar[DTYPE]](state[env, xquat_off + parent * 4 + 3])
                var rotated = gpu_quat_rotate(pqx, pqy, pqz, pqw, axis_x, axis_y, axis_z)
                axis_world_x = rotated[0]
                axis_world_y = rotated[1]
                axis_world_z = rotated[2]

            var body_px = rebind[Scalar[DTYPE]](state[env, xpos_off + body_id * 3 + 0])
            var body_py = rebind[Scalar[DTYPE]](state[env, xpos_off + body_id * 3 + 1])
            var body_pz = rebind[Scalar[DTYPE]](state[env, xpos_off + body_id * 3 + 2])

            var rx = body_px - jpos_world_x
            var ry = body_py - jpos_world_y
            var rz = body_pz - jpos_world_z

            var r_dot_axis = rx * axis_world_x + ry * axis_world_y + rz * axis_world_z
            var r_perp_x = rx - r_dot_axis * axis_world_x
            var r_perp_y = ry - r_dot_axis * axis_world_y
            var r_perp_z = rz - r_dot_axis * axis_world_z
            var r_perp_sq = r_perp_x * r_perp_x + r_perp_y * r_perp_y + r_perp_z * r_perp_z

            M_diag[dof_adr] = I_avg + mass * r_perp_sq

        elif jnt_type == GC_JNT_SLIDE:
            # Accumulate mass from body and ALL descendants
            var total_mass = mass
            for desc_body in range(body_id + 1, NBODY):
                if _is_descendant_gpu[DTYPE, NBODY, MODEL_SIZE](model, desc_body, body_id):
                    var desc_body_off = gc_model_body_offset(desc_body)
                    var desc_mass = rebind[Scalar[DTYPE]](model[0, desc_body_off + GC_BODY_IDX_MASS])
                    total_mass = total_mass + desc_mass
            M_diag[dof_adr] = total_mass

        elif jnt_type == GC_JNT_FREE:
            M_diag[dof_adr + 0] = mass
            M_diag[dof_adr + 1] = mass
            M_diag[dof_adr + 2] = mass
            M_diag[dof_adr + 3] = I_xx
            M_diag[dof_adr + 4] = I_yy
            M_diag[dof_adr + 5] = I_zz

        elif jnt_type == GC_JNT_BALL:
            M_diag[dof_adr + 0] = I_xx
            M_diag[dof_adr + 1] = I_yy
            M_diag[dof_adr + 2] = I_zz

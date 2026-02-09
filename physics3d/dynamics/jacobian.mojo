"""Jacobian computation for Generalized Coordinates engine.

Provides two key functions for constraint-based contact solving:

1. compute_cdof() - Spatial motion axis per DOF (6 floats per DOF)
   Maps a unit joint velocity to a spatial (angular, linear) velocity.
   Reference: MuJoCo engine_core_smooth.c:298-349, engine_util_spatial.c:446-458

2. compute_contact_jacobian_row() - One row of the contact Jacobian
   Maps joint velocities to contact-normal velocity for a single contact.
   Reference: MuJoCo engine_core_util.c:177-227

Both have CPU and GPU variants.
"""

from math import sqrt
from layout import LayoutTensor, Layout
from ..gpu.constants import ws_cdof_offset
from ..types import Model, Data, _max_one
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..kinematics.quat_math import quat_rotate
from ..gpu.constants import (
    xpos_offset,
    model_body_offset,
    model_joint_offset,
    model_metadata_offset,
    BODY_IDX_PARENT,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_DOF_ADR,
    MODEL_META_IDX_NJOINT,
)
from ..gpu.constants import (
    xpos_offset,
    xquat_offset,
    model_body_offset,
    model_joint_offset,
    model_metadata_offset,
    ws_cdof_offset,
    BODY_IDX_PARENT,
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
    JNT_HINGE,
    JNT_SLIDE,
)
from ..kinematics.quat_math import gpu_quat_rotate
from ..joint_types import (
    JNT_FREE,
    JNT_BALL,
    JNT_HINGE,
    JNT_SLIDE,
)

# =============================================================================
# CPU Functions
# =============================================================================


fn compute_cdof[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    CDOF_SIZE: Int,
](
    model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    mut cdof: InlineArray[Scalar[DTYPE], CDOF_SIZE],
):
    """Compute spatial motion axis (cdof) for each DOF.

    cdof[6*i : 6*i+6] = [ang_x, ang_y, ang_z, lin_x, lin_y, lin_z]

    For HINGE: angular part = axis_world, linear part = axis_world x offset
        where offset = xpos[body] - joint_anchor_world
    For SLIDE: angular part = (0,0,0), linear part = axis_world
    For FREE translation DOFs: angular = (0,0,0), linear = unit axis
    For FREE rotation DOFs: angular = unit axis, linear = unit axis x offset

    Reference: MuJoCo mju_dofCom() in engine_util_spatial.c:446-458

    Args:
        model: Static model configuration.
        data: Current simulation state (xpos, xquat must be computed).
        cdof: Output array of 6*NV spatial motion axes.
    """
    # Zero out
    for i in range(CDOF_SIZE):
        cdof[i] = Scalar[DTYPE](0)

    for j in range(model.num_joints):
        var joint = model.joints[j]
        var body = joint.body_id
        var dof_adr = joint.dof_adr
        var parent = model.body_parent[body]

        # Body world position (subtree root for computing offset)
        var bx = data.xpos[body * 3 + 0]
        var by = data.xpos[body * 3 + 1]
        var bz = data.xpos[body * 3 + 2]

        if joint.jnt_type == JNT_HINGE:
            # Get joint axis and position in world frame
            var axis_x = joint.axis_x
            var axis_y = joint.axis_y
            var axis_z = joint.axis_z

            var jpos_x = joint.pos_x
            var jpos_y = joint.pos_y
            var jpos_z = joint.pos_z

            if parent >= 0:
                var pqx = data.xquat[parent * 4 + 0]
                var pqy = data.xquat[parent * 4 + 1]
                var pqz = data.xquat[parent * 4 + 2]
                var pqw = data.xquat[parent * 4 + 3]

                var axis_world = quat_rotate(
                    pqx, pqy, pqz, pqw, axis_x, axis_y, axis_z
                )
                axis_x = axis_world[0]
                axis_y = axis_world[1]
                axis_z = axis_world[2]

                var ppx = data.xpos[parent * 3 + 0]
                var ppy = data.xpos[parent * 3 + 1]
                var ppz = data.xpos[parent * 3 + 2]

                var jp = quat_rotate(pqx, pqy, pqz, pqw, jpos_x, jpos_y, jpos_z)
                jpos_x = ppx + jp[0]
                jpos_y = ppy + jp[1]
                jpos_z = ppz + jp[2]

            # offset = body_com - joint_anchor
            var ox = bx - jpos_x
            var oy = by - jpos_y
            var oz = bz - jpos_z

            # angular part = axis
            cdof[dof_adr * 6 + 0] = axis_x
            cdof[dof_adr * 6 + 1] = axis_y
            cdof[dof_adr * 6 + 2] = axis_z
            # linear part = axis x offset
            cdof[dof_adr * 6 + 3] = axis_y * oz - axis_z * oy
            cdof[dof_adr * 6 + 4] = axis_z * ox - axis_x * oz
            cdof[dof_adr * 6 + 5] = axis_x * oy - axis_y * ox

        elif joint.jnt_type == JNT_SLIDE:
            # Get joint axis in world frame
            var axis_x = joint.axis_x
            var axis_y = joint.axis_y
            var axis_z = joint.axis_z

            if parent >= 0:
                var pqx = data.xquat[parent * 4 + 0]
                var pqy = data.xquat[parent * 4 + 1]
                var pqz = data.xquat[parent * 4 + 2]
                var pqw = data.xquat[parent * 4 + 3]

                var axis_world = quat_rotate(
                    pqx, pqy, pqz, pqw, axis_x, axis_y, axis_z
                )
                axis_x = axis_world[0]
                axis_y = axis_world[1]
                axis_z = axis_world[2]

            # angular part = 0
            # linear part = axis
            cdof[dof_adr * 6 + 3] = axis_x
            cdof[dof_adr * 6 + 4] = axis_y
            cdof[dof_adr * 6 + 5] = axis_z

        elif joint.jnt_type == JNT_FREE:
            # Translation DOFs (dof_adr + 0,1,2): pure linear motion
            cdof[(dof_adr + 0) * 6 + 3] = Scalar[DTYPE](1)  # x translation
            cdof[(dof_adr + 1) * 6 + 4] = Scalar[DTYPE](1)  # y translation
            cdof[(dof_adr + 2) * 6 + 5] = Scalar[DTYPE](1)  # z translation

            # Rotation DOFs (dof_adr + 3,4,5): angular + linear
            # angular part = unit axes
            cdof[(dof_adr + 3) * 6 + 0] = Scalar[DTYPE](1)  # x rotation
            cdof[(dof_adr + 4) * 6 + 1] = Scalar[DTYPE](1)  # y rotation
            cdof[(dof_adr + 5) * 6 + 2] = Scalar[DTYPE](1)  # z rotation

            # For FREE joints, joint anchor is at body origin, so offset = 0
            # linear part = axis x offset = 0


fn compute_contact_jacobian_row[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    V_SIZE: Int,
    CDOF_SIZE: Int,
](
    model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    cdof: InlineArray[Scalar[DTYPE], CDOF_SIZE],
    contact_body_a: Int,
    contact_body_b: Int,
    contact_pos_x: Scalar[DTYPE],
    contact_pos_y: Scalar[DTYPE],
    contact_pos_z: Scalar[DTYPE],
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    mut J_row: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """Compute one row of the contact Jacobian.

    Maps joint velocities to contact velocity along a given direction
    (normal or tangent) for a contact between body_a and body_b.

    For body-body contacts (body_b >= 0), the Jacobian is bilateral:
    J_row[i] = J_a[i] - J_b[i], where J_a and J_b are the contributions
    from body_a and body_b respectively. When a joint affects both bodies
    (shared ancestor), the contributions cancel — physically correct.

    For ground contacts (body_b = -1), only body_a contributes.

    Reference: MuJoCo mj_jac() in engine_core_util.c:177-227

    Args:
        model: Static model configuration.
        data: Current simulation state.
        cdof: Spatial motion axes per DOF (from compute_cdof).
        contact_body_a: Index of body A in contact.
        contact_body_b: Index of body B (-1 for ground).
        contact_pos_x/y/z: Contact point in world frame.
        dir_x/y/z: Direction vector (normal or tangent).
        J_row: Output Jacobian row (NV entries).
    """
    for i in range(V_SIZE):
        J_row[i] = Scalar[DTYPE](0)

    for j in range(model.num_joints):
        var joint = model.joints[j]
        var dof_adr = joint.dof_adr

        # Check if this joint affects either contact body
        var affects_a = _joint_affects_body(model, j, contact_body_a)
        var affects_b = (contact_body_b >= 0) and _joint_affects_body(
            model, j, contact_body_b
        )

        if not affects_a and not affects_b:
            continue

        var num_dof = 1
        if joint.jnt_type == JNT_FREE:
            num_dof = 6
        elif joint.jnt_type == JNT_BALL:
            num_dof = 3

        # Reference body = joint's body (must match cdof computation)
        var ref_body = joint.body_id
        var ref_x = data.xpos[ref_body * 3 + 0]
        var ref_y = data.xpos[ref_body * 3 + 1]
        var ref_z = data.xpos[ref_body * 3 + 2]

        var rx = contact_pos_x - ref_x
        var ry = contact_pos_y - ref_y
        var rz = contact_pos_z - ref_z

        for d in range(num_dof):
            var dof_idx = dof_adr + d

            # Get cdof components for this DOF
            var ang_x = cdof[dof_idx * 6 + 0]
            var ang_y = cdof[dof_idx * 6 + 1]
            var ang_z = cdof[dof_idx * 6 + 2]
            var lin_x = cdof[dof_idx * 6 + 3]
            var lin_y = cdof[dof_idx * 6 + 4]
            var lin_z = cdof[dof_idx * 6 + 5]

            # ang x r
            var cross_x = ang_y * rz - ang_z * ry
            var cross_y = ang_z * rx - ang_x * rz
            var cross_z = ang_x * ry - ang_y * rx

            var jt_x = lin_x + cross_x
            var jt_y = lin_y + cross_y
            var jt_z = lin_z + cross_z

            # Project onto direction
            var val = jt_x * dir_x + jt_y * dir_y + jt_z * dir_z

            # Body A contributes positively, body B negatively
            if affects_a:
                J_row[dof_idx] = J_row[dof_idx] + val
            if affects_b:
                J_row[dof_idx] = J_row[dof_idx] - val


fn _joint_affects_body[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
](
    model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    joint_idx: Int,
    body_idx: Int,
) -> Bool:
    """Check if a joint affects a body (body is the joint's body or a descendant).
    """
    var joint_body = model.joints[joint_idx].body_id

    if body_idx == joint_body:
        return True

    var current = body_idx
    while current >= 0:
        if model.body_parent[current] == joint_body:
            return True
        current = model.body_parent[current]

    return False


# =============================================================================
# Composite Rigid Body Inertia (CRBA helper)
# =============================================================================


fn compute_composite_inertia[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    CRB_SIZE: Int,
](
    model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    mut crb: InlineArray[Scalar[DTYPE], CRB_SIZE],
):
    """Compute composite rigid body inertia for each body.

    Each body's composite inertia is initialized from its own spatial inertia,
    then accumulated bottom-up: crb[parent] += transform(crb[child]).

    Storage: 10 floats per body:
        [mass, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    where (cx, cy, cz) is the CoM offset from the body frame origin,
    and Ixx..Iyz is the rotational inertia about the CoM.

    For the CRBA, the spatial inertia encodes:
    - mass: total mass of the composite body
    - CoM offset: mass-weighted center relative to body frame origin
    - Inertia: rotational inertia about CoM

    Args:
        model: Static model configuration.
        data: Current state (xpos, xquat from forward kinematics).
        crb: Output composite inertia (10 * NBODY floats).
    """
    # Initialize each body's own spatial inertia
    # The inertia tensor must be rotated from body-local to world frame
    # since cdof vectors are in world frame.
    for b in range(NBODY):
        var mass = model.body_mass[b]
        # CoM offset is 0 since xpos is already the body CoM
        crb[b * 10 + 0] = mass
        crb[b * 10 + 1] = Scalar[DTYPE](0)  # cx
        crb[b * 10 + 2] = Scalar[DTYPE](0)  # cy
        crb[b * 10 + 3] = Scalar[DTYPE](0)  # cz

        # Rotate inertia tensor from body-local to world frame:
        # I_world = R @ diag(Ixx, Iyy, Izz) @ R^T
        # where R columns are body basis vectors in world frame.
        var Ixx_local = model.body_inertia[b * 3 + 0]
        var Iyy_local = model.body_inertia[b * 3 + 1]
        var Izz_local = model.body_inertia[b * 3 + 2]

        # Get body quaternion (world orientation)
        var qx = data.xquat[b * 4 + 0]
        var qy = data.xquat[b * 4 + 1]
        var qz = data.xquat[b * 4 + 2]
        var qw = data.xquat[b * 4 + 3]

        # Compute rotation matrix columns from quaternion
        # col0 = R @ [1,0,0], col1 = R @ [0,1,0], col2 = R @ [0,0,1]
        var r00 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qy * qy + qz * qz)
        var r10 = Scalar[DTYPE](2) * (qx * qy + qw * qz)
        var r20 = Scalar[DTYPE](2) * (qx * qz - qw * qy)

        var r01 = Scalar[DTYPE](2) * (qx * qy - qw * qz)
        var r11 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qx * qx + qz * qz)
        var r21 = Scalar[DTYPE](2) * (qy * qz + qw * qx)

        var r02 = Scalar[DTYPE](2) * (qx * qz + qw * qy)
        var r12 = Scalar[DTYPE](2) * (qy * qz - qw * qx)
        var r22 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qx * qx + qy * qy)

        # I_world[a,b] = Ixx * col0[a]*col0[b] + Iyy * col1[a]*col1[b] + Izz * col2[a]*col2[b]
        crb[b * 10 + 4] = (
            Ixx_local * r00 * r00
            + Iyy_local * r01 * r01
            + Izz_local * r02 * r02
        )  # Ixx_world
        crb[b * 10 + 5] = (
            Ixx_local * r10 * r10
            + Iyy_local * r11 * r11
            + Izz_local * r12 * r12
        )  # Iyy_world
        crb[b * 10 + 6] = (
            Ixx_local * r20 * r20
            + Iyy_local * r21 * r21
            + Izz_local * r22 * r22
        )  # Izz_world
        crb[b * 10 + 7] = (
            Ixx_local * r00 * r10
            + Iyy_local * r01 * r11
            + Izz_local * r02 * r12
        )  # Ixy_world
        crb[b * 10 + 8] = (
            Ixx_local * r00 * r20
            + Iyy_local * r01 * r21
            + Izz_local * r02 * r22
        )  # Ixz_world
        crb[b * 10 + 9] = (
            Ixx_local * r10 * r20
            + Iyy_local * r11 * r21
            + Izz_local * r12 * r22
        )  # Iyz_world

    # Bottom-up accumulation: for each body (from leaves to root),
    # add its composite inertia to its parent.
    # We need to transform the child's spatial inertia to the parent frame.
    for b in range(NBODY - 1, 0, -1):
        var parent = model.body_parent[b]
        if parent < 0:
            continue

        var child_mass = crb[b * 10 + 0]
        if child_mass < Scalar[DTYPE](1e-20):
            continue

        var child_cx = crb[b * 10 + 1]
        var child_cy = crb[b * 10 + 2]
        var child_cz = crb[b * 10 + 3]
        var child_Ixx = crb[b * 10 + 4]
        var child_Iyy = crb[b * 10 + 5]
        var child_Izz = crb[b * 10 + 6]
        var child_Ixy = crb[b * 10 + 7]
        var child_Ixz = crb[b * 10 + 8]
        var child_Iyz = crb[b * 10 + 9]

        # The offset from parent's origin to child's origin (in world frame)
        var dx = data.xpos[b * 3 + 0] - data.xpos[parent * 3 + 0]
        var dy = data.xpos[b * 3 + 1] - data.xpos[parent * 3 + 1]
        var dz = data.xpos[b * 3 + 2] - data.xpos[parent * 3 + 2]

        # Total offset from parent origin to child's composite CoM
        var total_cx = dx + child_cx
        var total_cy = dy + child_cy
        var total_cz = dz + child_cz

        var parent_mass = crb[parent * 10 + 0]
        var parent_cx = crb[parent * 10 + 1]
        var parent_cy = crb[parent * 10 + 2]
        var parent_cz = crb[parent * 10 + 3]

        # New combined mass
        var new_mass = parent_mass + child_mass

        # New combined CoM (mass-weighted average)
        var new_cx = Scalar[DTYPE](0)
        var new_cy = Scalar[DTYPE](0)
        var new_cz = Scalar[DTYPE](0)
        if new_mass > Scalar[DTYPE](1e-20):
            new_cx = (
                parent_mass * parent_cx + child_mass * total_cx
            ) / new_mass
            new_cy = (
                parent_mass * parent_cy + child_mass * total_cy
            ) / new_mass
            new_cz = (
                parent_mass * parent_cz + child_mass * total_cz
            ) / new_mass

        # Parallel axis theorem for combining inertias
        # I_combined = I_parent_about_new_com + I_child_about_new_com
        # For each sub-body: I_about_new_com = I_about_own_com + m * ||d||^2 * I3 - m * d⊗d
        # where d is the vector from new CoM to sub-body CoM.

        # Parent contribution: offset from new CoM to parent CoM
        var dp_x = parent_cx - new_cx
        var dp_y = parent_cy - new_cy
        var dp_z = parent_cz - new_cz
        var dp_sq = dp_x * dp_x + dp_y * dp_y + dp_z * dp_z

        var new_Ixx = crb[parent * 10 + 4] + parent_mass * (dp_sq - dp_x * dp_x)
        var new_Iyy = crb[parent * 10 + 5] + parent_mass * (dp_sq - dp_y * dp_y)
        var new_Izz = crb[parent * 10 + 6] + parent_mass * (dp_sq - dp_z * dp_z)
        var new_Ixy = crb[parent * 10 + 7] - parent_mass * dp_x * dp_y
        var new_Ixz = crb[parent * 10 + 8] - parent_mass * dp_x * dp_z
        var new_Iyz = crb[parent * 10 + 9] - parent_mass * dp_y * dp_z

        # Child contribution: offset from new CoM to child composite CoM
        var dc_x = total_cx - new_cx
        var dc_y = total_cy - new_cy
        var dc_z = total_cz - new_cz
        var dc_sq = dc_x * dc_x + dc_y * dc_y + dc_z * dc_z

        new_Ixx = new_Ixx + child_Ixx + child_mass * (dc_sq - dc_x * dc_x)
        new_Iyy = new_Iyy + child_Iyy + child_mass * (dc_sq - dc_y * dc_y)
        new_Izz = new_Izz + child_Izz + child_mass * (dc_sq - dc_z * dc_z)
        new_Ixy = new_Ixy + child_Ixy - child_mass * dc_x * dc_y
        new_Ixz = new_Ixz + child_Ixz - child_mass * dc_x * dc_z
        new_Iyz = new_Iyz + child_Iyz - child_mass * dc_y * dc_z

        # Store combined
        crb[parent * 10 + 0] = new_mass
        crb[parent * 10 + 1] = new_cx
        crb[parent * 10 + 2] = new_cy
        crb[parent * 10 + 3] = new_cz
        crb[parent * 10 + 4] = new_Ixx
        crb[parent * 10 + 5] = new_Iyy
        crb[parent * 10 + 6] = new_Izz
        crb[parent * 10 + 7] = new_Ixy
        crb[parent * 10 + 8] = new_Ixz
        crb[parent * 10 + 9] = new_Iyz


@always_inline
fn compute_composite_inertia_gpu[
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
    """Compute composite rigid body inertia on GPU. Writes crb to workspace."""
    from ..gpu.constants import (
        xpos_offset,
        xquat_offset,
        model_body_offset,
        ws_crb_offset,
        BODY_IDX_PARENT,
        BODY_IDX_MASS,
        BODY_IDX_IXX,
        BODY_IDX_IYY,
        BODY_IDX_IZZ,
    )

    # Derive crb pointer from workspace (MutAnyOrigin)
    comptime crb_idx = ws_crb_offset[NV]()

    var xpos_off = xpos_offset[NQ, NV, NBODY]()
    var xquat_off = xquat_offset[NQ, NV, NBODY]()

    # Initialize each body's own spatial inertia (rotated to world frame)
    for b in range(NBODY):
        var body_off = model_body_offset(b)
        var mass = model[0, body_off + BODY_IDX_MASS]
        var Ixx_local = model[0, body_off + BODY_IDX_IXX]
        var Iyy_local = model[0, body_off + BODY_IDX_IYY]
        var Izz_local = model[0, body_off + BODY_IDX_IZZ]

        # Get body quaternion (world orientation)
        var qx = state[env, xquat_off + b * 4 + 0]
        var qy = state[env, xquat_off + b * 4 + 1]
        var qz = state[env, xquat_off + b * 4 + 2]
        var qw = state[env, xquat_off + b * 4 + 3]

        # Rotation matrix columns from quaternion
        var r00 = 1 - 2 * (qy * qy + qz * qz)
        var r10 = 2 * (qx * qy + qw * qz)
        var r20 = 2 * (qx * qz - qw * qy)

        var r01 = 2 * (qx * qy - qw * qz)
        var r11 = 1 - 2 * (qx * qx + qz * qz)
        var r21 = 2 * (qy * qz + qw * qx)

        var r02 = 2 * (qx * qz + qw * qy)
        var r12 = 2 * (qy * qz - qw * qx)
        var r22 = 1 - 2 * (qx * qx + qy * qy)

        workspace[env, crb_idx + b * 10 + 0] = mass
        workspace[env, crb_idx + b * 10 + 1] = 0
        workspace[env, crb_idx + b * 10 + 2] = 0
        workspace[env, crb_idx + b * 10 + 3] = 0
        # I_world = R @ diag(Ixx, Iyy, Izz) @ R^T
        workspace[env, crb_idx + b * 10 + 4] = (
            Ixx_local * r00 * r00
            + Iyy_local * r01 * r01
            + Izz_local * r02 * r02
        )
        workspace[env, crb_idx + b * 10 + 5] = (
            Ixx_local * r10 * r10
            + Iyy_local * r11 * r11
            + Izz_local * r12 * r12
        )
        workspace[env, crb_idx + b * 10 + 6] = (
            Ixx_local * r20 * r20
            + Iyy_local * r21 * r21
            + Izz_local * r22 * r22
        )
        workspace[env, crb_idx + b * 10 + 7] = (
            Ixx_local * r00 * r10
            + Iyy_local * r01 * r11
            + Izz_local * r02 * r12
        )
        workspace[env, crb_idx + b * 10 + 8] = (
            Ixx_local * r00 * r20
            + Iyy_local * r01 * r21
            + Izz_local * r02 * r22
        )
        workspace[env, crb_idx + b * 10 + 9] = (
            Ixx_local * r10 * r20
            + Iyy_local * r11 * r21
            + Izz_local * r12 * r22
        )

    # Bottom-up accumulation
    for b in range(NBODY - 1, 0, -1):
        var body_off = model_body_offset(b)
        var parent = Int(model[0, body_off + BODY_IDX_PARENT])
        if parent < 0:
            continue

        var child_mass = workspace[env, crb_idx + b * 10 + 0]
        if child_mass < 1e-20:
            continue

        var child_cx = workspace[env, crb_idx + b * 10 + 1]
        var child_cy = workspace[env, crb_idx + b * 10 + 2]
        var child_cz = workspace[env, crb_idx + b * 10 + 3]
        var child_Ixx = workspace[env, crb_idx + b * 10 + 4]
        var child_Iyy = workspace[env, crb_idx + b * 10 + 5]
        var child_Izz = workspace[env, crb_idx + b * 10 + 6]
        var child_Ixy = workspace[env, crb_idx + b * 10 + 7]
        var child_Ixz = workspace[env, crb_idx + b * 10 + 8]
        var child_Iyz = workspace[env, crb_idx + b * 10 + 9]

        var dx = (
            state[env, xpos_off + b * 3 + 0]
            - state[env, xpos_off + parent * 3 + 0]
        )
        var dy = (
            state[env, xpos_off + b * 3 + 1]
            - state[env, xpos_off + parent * 3 + 1]
        )
        var dz = (
            state[env, xpos_off + b * 3 + 2]
            - state[env, xpos_off + parent * 3 + 2]
        )

        var total_cx = dx + child_cx
        var total_cy = dy + child_cy
        var total_cz = dz + child_cz

        var parent_mass = workspace[env, crb_idx + parent * 10 + 0]
        var parent_cx = workspace[env, crb_idx + parent * 10 + 1]
        var parent_cy = workspace[env, crb_idx + parent * 10 + 2]
        var parent_cz = workspace[env, crb_idx + parent * 10 + 3]

        var new_mass = parent_mass + child_mass

        var new_cx: workspace.element_type = 0
        var new_cy: workspace.element_type = 0
        var new_cz: workspace.element_type = 0
        if new_mass > 1e-20:
            new_cx = (
                parent_mass * parent_cx + child_mass * total_cx
            ) / new_mass
            new_cy = (
                parent_mass * parent_cy + child_mass * total_cy
            ) / new_mass
            new_cz = (
                parent_mass * parent_cz + child_mass * total_cz
            ) / new_mass

        var dp_x = parent_cx - new_cx
        var dp_y = parent_cy - new_cy
        var dp_z = parent_cz - new_cz
        var dp_sq = dp_x * dp_x + dp_y * dp_y + dp_z * dp_z

        var new_Ixx = workspace[
            env, crb_idx + parent * 10 + 4
        ] + parent_mass * (dp_sq - dp_x * dp_x)
        var new_Iyy = workspace[
            env, crb_idx + parent * 10 + 5
        ] + parent_mass * (dp_sq - dp_y * dp_y)
        var new_Izz = workspace[
            env, crb_idx + parent * 10 + 6
        ] + parent_mass * (dp_sq - dp_z * dp_z)
        var new_Ixy = (
            workspace[env, crb_idx + parent * 10 + 7]
            - parent_mass * dp_x * dp_y
        )
        var new_Ixz = (
            workspace[env, crb_idx + parent * 10 + 8]
            - parent_mass * dp_x * dp_z
        )
        var new_Iyz = (
            workspace[env, crb_idx + parent * 10 + 9]
            - parent_mass * dp_y * dp_z
        )

        var dc_x = total_cx - new_cx
        var dc_y = total_cy - new_cy
        var dc_z = total_cz - new_cz
        var dc_sq = dc_x * dc_x + dc_y * dc_y + dc_z * dc_z

        new_Ixx = new_Ixx + child_Ixx + child_mass * (dc_sq - dc_x * dc_x)
        new_Iyy = new_Iyy + child_Iyy + child_mass * (dc_sq - dc_y * dc_y)
        new_Izz = new_Izz + child_Izz + child_mass * (dc_sq - dc_z * dc_z)
        new_Ixy = new_Ixy + child_Ixy - child_mass * dc_x * dc_y
        new_Ixz = new_Ixz + child_Ixz - child_mass * dc_x * dc_z
        new_Iyz = new_Iyz + child_Iyz - child_mass * dc_y * dc_z

        workspace[env, crb_idx + parent * 10 + 0] = new_mass
        workspace[env, crb_idx + parent * 10 + 1] = new_cx
        workspace[env, crb_idx + parent * 10 + 2] = new_cy
        workspace[env, crb_idx + parent * 10 + 3] = new_cz
        workspace[env, crb_idx + parent * 10 + 4] = new_Ixx
        workspace[env, crb_idx + parent * 10 + 5] = new_Iyy
        workspace[env, crb_idx + parent * 10 + 6] = new_Izz
        workspace[env, crb_idx + parent * 10 + 7] = new_Ixy
        workspace[env, crb_idx + parent * 10 + 8] = new_Ixz
        workspace[env, crb_idx + parent * 10 + 9] = new_Iyz


# =============================================================================
# GPU Functions
# =============================================================================


@always_inline
fn compute_cdof_gpu[
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
    """Compute spatial motion axis (cdof) for each DOF on GPU.

    Writes cdof to workspace buffer instead of InlineArray.
    """

    # Derive cdof pointer from workspace (MutAnyOrigin)
    comptime cdof_idx = ws_cdof_offset()

    # Zero out
    for i in range(NV * 6):
        workspace[env, cdof_idx + i] = 0

    var xpos_off = xpos_offset[NQ, NV, NBODY]()
    var xquat_off = xquat_offset[NQ, NV, NBODY]()
    var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
    var num_joints = Int(model[0, model_meta_off + MODEL_META_IDX_NJOINT])

    for j in range(num_joints):
        var joint_off = model_joint_offset[NBODY](j)
        var jnt_type = Int(model[0, joint_off + JOINT_IDX_TYPE])
        var body = Int(model[0, joint_off + JOINT_IDX_BODY_ID])
        var dof_adr = Int(model[0, joint_off + JOINT_IDX_DOF_ADR])
        var body_off = model_body_offset(body)
        var parent = Int(model[0, body_off + BODY_IDX_PARENT])
        var bx = state[env, xpos_off + body * 3 + 0]
        var by = state[env, xpos_off + body * 3 + 1]
        var bz = state[env, xpos_off + body * 3 + 2]
        if jnt_type == JNT_HINGE:
            var axis_x = rebind[Scalar[DTYPE]](
                model[0, joint_off + JOINT_IDX_AXIS_X]
            )
            var axis_y = rebind[Scalar[DTYPE]](
                model[0, joint_off + JOINT_IDX_AXIS_Y]
            )
            var axis_z = rebind[Scalar[DTYPE]](
                model[0, joint_off + JOINT_IDX_AXIS_Z]
            )

            var jpos_x = rebind[Scalar[DTYPE]](
                model[0, joint_off + JOINT_IDX_POS_X]
            )
            var jpos_y = rebind[Scalar[DTYPE]](
                model[0, joint_off + JOINT_IDX_POS_Y]
            )
            var jpos_z = rebind[Scalar[DTYPE]](
                model[0, joint_off + JOINT_IDX_POS_Z]
            )

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

                var a_w = gpu_quat_rotate(
                    pqx, pqy, pqz, pqw, axis_x, axis_y, axis_z
                )
                axis_x = a_w[0]
                axis_y = a_w[1]
                axis_z = a_w[2]

                var ppx = rebind[Scalar[DTYPE]](
                    state[env, xpos_off + parent * 3 + 0]
                )
                var ppy = rebind[Scalar[DTYPE]](
                    state[env, xpos_off + parent * 3 + 1]
                )
                var ppz = rebind[Scalar[DTYPE]](
                    state[env, xpos_off + parent * 3 + 2]
                )

                var jp = gpu_quat_rotate(
                    pqx, pqy, pqz, pqw, jpos_x, jpos_y, jpos_z
                )
                jpos_x = ppx + jp[0]
                jpos_y = ppy + jp[1]
                jpos_z = ppz + jp[2]

            var ox = bx - jpos_x
            var oy = by - jpos_y
            var oz = bz - jpos_z

            workspace[env, cdof_idx + dof_adr * 6 + 0] = axis_x
            workspace[env, cdof_idx + dof_adr * 6 + 1] = axis_y
            workspace[env, cdof_idx + dof_adr * 6 + 2] = axis_z
            workspace[env, cdof_idx + dof_adr * 6 + 3] = (
                axis_y * oz - axis_z * oy
            )
            workspace[env, cdof_idx + dof_adr * 6 + 4] = (
                axis_z * ox - axis_x * oz
            )
            workspace[env, cdof_idx + dof_adr * 6 + 5] = (
                axis_x * oy - axis_y * ox
            )

        elif jnt_type == JNT_SLIDE:
            var axis_x = rebind[Scalar[DTYPE]](
                model[0, joint_off + JOINT_IDX_AXIS_X]
            )
            var axis_y = rebind[Scalar[DTYPE]](
                model[0, joint_off + JOINT_IDX_AXIS_Y]
            )
            var axis_z = rebind[Scalar[DTYPE]](
                model[0, joint_off + JOINT_IDX_AXIS_Z]
            )

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

                var a_w = gpu_quat_rotate(
                    pqx, pqy, pqz, pqw, axis_x, axis_y, axis_z
                )
                axis_x = a_w[0]
                axis_y = a_w[1]
                axis_z = a_w[2]

            workspace[env, cdof_idx + dof_adr * 6 + 3] = axis_x
            workspace[env, cdof_idx + dof_adr * 6 + 4] = axis_y
            workspace[env, cdof_idx + dof_adr * 6 + 5] = axis_z

        elif jnt_type == JNT_FREE:
            workspace[env, cdof_idx + (dof_adr + 0) * 6 + 3] = Scalar[DTYPE](1)
            workspace[env, cdof_idx + (dof_adr + 1) * 6 + 4] = Scalar[DTYPE](1)
            workspace[env, cdof_idx + (dof_adr + 2) * 6 + 5] = Scalar[DTYPE](1)
            workspace[env, cdof_idx + (dof_adr + 3) * 6 + 0] = Scalar[DTYPE](1)
            workspace[env, cdof_idx + (dof_adr + 4) * 6 + 1] = Scalar[DTYPE](1)
            workspace[env, cdof_idx + (dof_adr + 5) * 6 + 2] = Scalar[DTYPE](1)


@always_inline
fn compute_contact_jacobian_row_gpu[
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
    contact_body_a: Int,
    contact_body_b: Int,
    contact_pos_x: Scalar[DTYPE],
    contact_pos_y: Scalar[DTYPE],
    contact_pos_z: Scalar[DTYPE],
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    mut J_row: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """Compute one row of the contact Jacobian on GPU.

    Reads cdof from workspace. J_row remains InlineArray (small, V_SIZE).
    Bilateral: J_row[i] = J_a[i] - J_b[i] for body-body contacts.
    For ground contacts (body_b = -1), only body_a contributes.
    """

    # Derive cdof pointer from workspace (read-only)
    comptime cdof_idx = ws_cdof_offset()

    for i in range(V_SIZE):
        J_row[i] = 0

    var xpos_off = xpos_offset[NQ, NV, NBODY]()
    var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
    var num_joints = Int(
        rebind[Scalar[DTYPE]](model[0, model_meta_off + MODEL_META_IDX_NJOINT])
    )

    for j_idx in range(num_joints):
        var joint_off = model_joint_offset[NBODY](j_idx)
        var jnt_type = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
        )
        var joint_body = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_BODY_ID])
        )
        var dof_adr = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR])
        )

        # Check if this joint affects body_a
        var affects_a = False
        if contact_body_a == joint_body:
            affects_a = True
        else:
            var current = contact_body_a
            while current >= 0:
                var current_body_off = model_body_offset(current)
                var current_parent = Int(
                    rebind[Scalar[DTYPE]](
                        model[0, current_body_off + BODY_IDX_PARENT]
                    )
                )
                if current_parent == joint_body:
                    affects_a = True
                    break
                current = current_parent

        # Check if this joint affects body_b (only if body_b >= 0)
        var affects_b = False
        if contact_body_b >= 0:
            if contact_body_b == joint_body:
                affects_b = True
            else:
                var current_b = contact_body_b
                while current_b >= 0:
                    var current_body_off_b = model_body_offset(current_b)
                    var current_parent_b = Int(
                        rebind[Scalar[DTYPE]](
                            model[0, current_body_off_b + BODY_IDX_PARENT]
                        )
                    )
                    if current_parent_b == joint_body:
                        affects_b = True
                        break
                    current_b = current_parent_b

        if not affects_a and not affects_b:
            continue

        var num_dof = 1
        if jnt_type == JNT_FREE:
            num_dof = 6
        elif jnt_type == JNT_BALL:
            num_dof = 3

        # Reference body = joint's body (must match cdof computation)
        var b_x = rebind[Scalar[DTYPE]](
            state[env, xpos_off + joint_body * 3 + 0]
        )
        var b_y = rebind[Scalar[DTYPE]](
            state[env, xpos_off + joint_body * 3 + 1]
        )
        var b_z = rebind[Scalar[DTYPE]](
            state[env, xpos_off + joint_body * 3 + 2]
        )

        var rx = contact_pos_x - b_x
        var ry = contact_pos_y - b_y
        var rz = contact_pos_z - b_z

        for d in range(num_dof):
            var dof_idx = dof_adr + d

            var ang_x = workspace[env, cdof_idx + dof_idx * 6 + 0]
            var ang_y = workspace[env, cdof_idx + dof_idx * 6 + 1]
            var ang_z = workspace[env, cdof_idx + dof_idx * 6 + 2]
            var lin_x = workspace[env, cdof_idx + dof_idx * 6 + 3]
            var lin_y = workspace[env, cdof_idx + dof_idx * 6 + 4]
            var lin_z = workspace[env, cdof_idx + dof_idx * 6 + 5]

            # J_trans = cdof_lin + cdof_ang x r
            var cross_x = ang_y * rz - ang_z * ry
            var cross_y = ang_z * rx - ang_x * rz
            var cross_z = ang_x * ry - ang_y * rx

            var jt_x = lin_x + cross_x
            var jt_y = lin_y + cross_y
            var jt_z = lin_z + cross_z

            var val = jt_x * dir_x + jt_y * dir_y + jt_z * dir_z

            # Body A contributes positively, body B negatively
            if affects_a:
                J_row[dof_idx] += rebind[Scalar[DTYPE]](val)
            if affects_b:
                J_row[dof_idx] -= rebind[Scalar[DTYPE]](val)

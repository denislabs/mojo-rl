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

from std.math import sqrt
from layout import LayoutTensor, Layout
from ..gpu.constants import ws_cdof_offset
from ..types import Model, Data, _max_one
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..kinematics.quat_math import quat_rotate, quat_mul, axis_angle_to_quat
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
    qpos_offset,
    xpos_offset,
    xquat_offset,
    xipos_offset,
    model_body_offset,
    model_joint_offset,
    model_metadata_offset,
    ws_cdof_offset,
    BODY_IDX_PARENT,
    BODY_IDX_POS_X,
    BODY_IDX_POS_Y,
    BODY_IDX_POS_Z,
    BODY_IDX_QUAT_X,
    BODY_IDX_QUAT_Y,
    BODY_IDX_QUAT_Z,
    BODY_IDX_QUAT_W,
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
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_POS_X,
    JOINT_IDX_POS_Y,
    JOINT_IDX_POS_Z,
    JOINT_IDX_AXIS_X,
    JOINT_IDX_AXIS_Y,
    JOINT_IDX_AXIS_Z,
    JOINT_IDX_QPOS0,
    MODEL_META_IDX_NJOINT,
)
from ..joint_types import (
    JNT_FREE,
    JNT_HINGE,
    JNT_SLIDE,
)
from ..kinematics.quat_math import (
    gpu_quat_rotate,
    gpu_quat_mul,
    gpu_axis_angle_to_quat,
)
from ..joint_types import (
    JNT_FREE,
    JNT_BALL,
    JNT_HINGE,
    JNT_SLIDE,
)

# =============================================================================
# CPU Functions
# =============================================================================


def compute_subtree_com[
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
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        NGEOM, MAX_EQUALITY, CONE_TYPE, MAX_TENDON, NSITE,
    ],
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
    mut subtree_com: List[Scalar[DTYPE]],
):
    """Compute subtree center of mass for each body (MuJoCo mj_comPos).

    subtree_com[3*b : 3*b+3] = mass-weighted average of xipos for body b
    and all its descendants. Used as the spatial reference point for cdof
    and composite inertia (CRBA).
    """
    # Initialize with mass * xipos for each body
    var stmass = List[Scalar[DTYPE]](capacity=NBODY)
    for b in range(NBODY):
        stmass.append(model.body_mass[b])
        subtree_com[b * 3 + 0] = model.body_mass[b] * data.xipos[b * 3 + 0]
        subtree_com[b * 3 + 1] = model.body_mass[b] * data.xipos[b * 3 + 1]
        subtree_com[b * 3 + 2] = model.body_mass[b] * data.xipos[b * 3 + 2]

    # Bottom-up accumulation
    for b in range(NBODY - 1, 0, -1):
        var p = model.body_parent[b]
        stmass[p] = stmass[p] + stmass[b]
        subtree_com[p * 3 + 0] = subtree_com[p * 3 + 0] + subtree_com[b * 3 + 0]
        subtree_com[p * 3 + 1] = subtree_com[p * 3 + 1] + subtree_com[b * 3 + 1]
        subtree_com[p * 3 + 2] = subtree_com[p * 3 + 2] + subtree_com[b * 3 + 2]

    # Normalize by subtree mass
    for b in range(NBODY):
        if stmass[b] > Scalar[DTYPE](1e-10):
            subtree_com[b * 3 + 0] = subtree_com[b * 3 + 0] / stmass[b]
            subtree_com[b * 3 + 1] = subtree_com[b * 3 + 1] / stmass[b]
            subtree_com[b * 3 + 2] = subtree_com[b * 3 + 2] / stmass[b]
        else:
            subtree_com[b * 3 + 0] = data.xipos[b * 3 + 0]
            subtree_com[b * 3 + 1] = data.xipos[b * 3 + 1]
            subtree_com[b * 3 + 2] = data.xipos[b * 3 + 2]


def compute_cdof[
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
    mut cdof: List[Scalar[DTYPE]],
    subtree_com: List[Scalar[DTYPE]] = List[Scalar[DTYPE]](),
):
    """Compute spatial motion axis (cdof) for each DOF.

    cdof[6*i : 6*i+6] = [ang_x, ang_y, ang_z, lin_x, lin_y, lin_z]

    For HINGE: angular part = axis_world, linear part = axis_world x offset
        where offset = subtree_com[rootid[body]] - joint_anchor_world
    For SLIDE: angular part = (0,0,0), linear part = axis_world
    For FREE translation DOFs: angular = (0,0,0), linear = unit axis
    For FREE rotation DOFs: angular = unit axis,
        linear = axis x (subtree_com - body_xpos)  (MuJoCo mj_comPos)

    IMPORTANT: Joint axes are rotated using the body's accumulated orientation
    BEFORE that joint's own rotation is applied (matching MuJoCo's xaxis
    computation in mj_kinematics). For bodies with multiple joints, each joint
    sees the orientation accumulated from previous joints only.

    Reference: MuJoCo engine_core_smooth.c mj_kinematics, engine_util_spatial.c

    Args:
        model: Static model configuration.
        data: Current simulation state (xpos, xquat, xipos must be computed).
        cdof: Output array of 6*NV spatial motion axes.
    """
    # Zero out
    for i in range(NV * 6):
        cdof[i] = Scalar[DTYPE](0)

    # Process per-body (like MuJoCo FK), tracking accumulated orientation
    # Skip worldbody at 0 (no joints)
    for body in range(1, NBODY):
        var parent = model.body_parent[body]

        # Get parent's world orientation (worldbody=0 has identity)
        var acc_qx = data.xquat[parent * 4 + 0]
        var acc_qy = data.xquat[parent * 4 + 1]
        var acc_qz = data.xquat[parent * 4 + 2]
        var acc_qw = data.xquat[parent * 4 + 3]

        # Apply body_quat: acc = parent_quat * body_quat
        var bq_x = model.body_quat[body * 4 + 0]
        var bq_y = model.body_quat[body * 4 + 1]
        var bq_z = model.body_quat[body * 4 + 2]
        var bq_w = model.body_quat[body * 4 + 3]
        var pre_q = quat_mul(
            acc_qx,
            acc_qy,
            acc_qz,
            acc_qw,
            bq_x,
            bq_y,
            bq_z,
            bq_w,
        )
        acc_qx = pre_q[0]
        acc_qy = pre_q[1]
        acc_qz = pre_q[2]
        acc_qw = pre_q[3]

        # Reference point for cdof offset computation.
        # When subtree_com is available, use subtree_com[rootid[body]]
        # (MuJoCo mj_comPos convention). Otherwise fall back to xipos[body].
        var has_stcom = len(subtree_com) >= NBODY * 3
        var ref_x: Scalar[DTYPE]
        var ref_y: Scalar[DTYPE]
        var ref_z: Scalar[DTYPE]
        if has_stcom:
            var root = model.body_rootid[body]
            ref_x = subtree_com[root * 3 + 0]
            ref_y = subtree_com[root * 3 + 1]
            ref_z = subtree_com[root * 3 + 2]
        else:
            ref_x = data.xipos[body * 3 + 0]
            ref_y = data.xipos[body * 3 + 1]
            ref_z = data.xipos[body * 3 + 2]

        # Compute xpos_initial for this body (matching MuJoCo FK):
        # xpos_initial = xpos[parent]_final + R(xquat[parent]_final) * body_pos
        # This is the body position BEFORE any joint corrections (off-center rotation
        # or slide displacements). MuJoCo computes xanchor from this initial position.
        # We must NOT use data.xpos[body] (which is xpos_final after corrections).
        var bpos_world = quat_rotate(
            data.xquat[parent * 4 + 0],
            data.xquat[parent * 4 + 1],
            data.xquat[parent * 4 + 2],
            data.xquat[parent * 4 + 3],
            model.body_pos[body * 3 + 0],
            model.body_pos[body * 3 + 1],
            model.body_pos[body * 3 + 2],
        )
        # Running position — tracks xpos as joints are applied (like MuJoCo FK)
        var cx = data.xpos[parent * 3 + 0] + bpos_world[0]
        var cy = data.xpos[parent * 3 + 1] + bpos_world[1]
        var cz = data.xpos[parent * 3 + 2] + bpos_world[2]

        # Process all joints for this body in order
        for j in range(model.num_joints):
            var joint = model.joints[j]
            if joint.body_id != body:
                continue

            var dof_adr = joint.dof_adr

            if joint.jnt_type == JNT_HINGE:
                # Get joint axis and position (body-relative)
                var axis_lx = joint.axis_x
                var axis_ly = joint.axis_y
                var axis_lz = joint.axis_z

                var jpos_lx = joint.pos_x
                var jpos_ly = joint.pos_y
                var jpos_lz = joint.pos_z

                # Rotate axis to world using accumulated (pre-this-joint) orientation
                var axis_world = quat_rotate(
                    acc_qx,
                    acc_qy,
                    acc_qz,
                    acc_qw,
                    axis_lx,
                    axis_ly,
                    axis_lz,
                )
                var ax = axis_world[0]
                var ay = axis_world[1]
                var az = axis_world[2]

                # Joint anchor = running_xpos + rotate(jnt_pos, acc_quat)
                # Uses cx/cy/cz (xpos before this joint's correction), not xpos_final
                var jp = quat_rotate(
                    acc_qx,
                    acc_qy,
                    acc_qz,
                    acc_qw,
                    jpos_lx,
                    jpos_ly,
                    jpos_lz,
                )
                var anc_x = cx + jp[0]
                var anc_y = cy + jp[1]
                var anc_z = cz + jp[2]

                # offset = reference_point - joint_anchor
                # MuJoCo: subtree_com[rootid] - xanchor
                var ox = ref_x - anc_x
                var oy = ref_y - anc_y
                var oz = ref_z - anc_z

                # angular part = axis
                cdof[dof_adr * 6 + 0] = ax
                cdof[dof_adr * 6 + 1] = ay
                cdof[dof_adr * 6 + 2] = az
                # linear part = axis x offset
                cdof[dof_adr * 6 + 3] = ay * oz - az * oy
                cdof[dof_adr * 6 + 4] = az * ox - ax * oz
                cdof[dof_adr * 6 + 5] = ax * oy - ay * ox

                # Update accumulated orientation with this hinge rotation
                var angle = (
                    data.qpos[joint.qpos_adr] - model.qpos0[joint.qpos_adr]
                )
                var hinge_quat = axis_angle_to_quat(ax, ay, az, angle)
                var new_q = quat_mul(
                    hinge_quat[0],
                    hinge_quat[1],
                    hinge_quat[2],
                    hinge_quat[3],
                    acc_qx,
                    acc_qy,
                    acc_qz,
                    acc_qw,
                )
                acc_qx = new_q[0]
                acc_qy = new_q[1]
                acc_qz = new_q[2]
                acc_qw = new_q[3]

                # Off-center correction: update running xpos (MuJoCo: xpos = anchor - R(new_xquat)*jnt_pos)
                var vec = quat_rotate(
                    acc_qx,
                    acc_qy,
                    acc_qz,
                    acc_qw,
                    jpos_lx,
                    jpos_ly,
                    jpos_lz,
                )
                cx = anc_x - vec[0]
                cy = anc_y - vec[1]
                cz = anc_z - vec[2]

            elif joint.jnt_type == JNT_SLIDE:
                # Get joint axis (body-relative)
                var axis_lx = joint.axis_x
                var axis_ly = joint.axis_y
                var axis_lz = joint.axis_z

                # Rotate axis to world using accumulated (pre-this-joint) orientation
                var axis_world = quat_rotate(
                    acc_qx,
                    acc_qy,
                    acc_qz,
                    acc_qw,
                    axis_lx,
                    axis_ly,
                    axis_lz,
                )

                # angular part = 0 (slide doesn't rotate)
                # linear part = axis
                cdof[dof_adr * 6 + 3] = axis_world[0]
                cdof[dof_adr * 6 + 4] = axis_world[1]
                cdof[dof_adr * 6 + 5] = axis_world[2]

                # Slide: update running xpos by displacement
                var disp = (
                    data.qpos[joint.qpos_adr] - model.qpos0[joint.qpos_adr]
                )
                cx += disp * axis_world[0]
                cy += disp * axis_world[1]
                cz += disp * axis_world[2]

            elif joint.jnt_type == JNT_FREE:
                # Translation DOFs (dof_adr + 0,1,2): pure linear motion
                cdof[(dof_adr + 0) * 6 + 3] = Scalar[DTYPE](1)  # x
                cdof[(dof_adr + 1) * 6 + 4] = Scalar[DTYPE](1)  # y
                cdof[(dof_adr + 2) * 6 + 5] = Scalar[DTYPE](1)  # z

                # Rotation DOFs (dof_adr + 3,4,5): angular + linear
                # MuJoCo (mj_comPos): axes = columns of xmat[body] (body frame
                # in world coordinates), NOT fixed global axes.
                # xmat columns from xquat[body]:
                var bqx = data.xquat[body * 4 + 0]
                var bqy = data.xquat[body * 4 + 1]
                var bqz = data.xquat[body * 4 + 2]
                var bqw = data.xquat[body * 4 + 3]
                # Column 0 of rotation matrix (body x-axis in world)
                var ax0_x = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (bqy * bqy + bqz * bqz)
                var ax0_y = Scalar[DTYPE](2) * (bqx * bqy + bqw * bqz)
                var ax0_z = Scalar[DTYPE](2) * (bqx * bqz - bqw * bqy)
                # Column 1 (body y-axis in world)
                var ax1_x = Scalar[DTYPE](2) * (bqx * bqy - bqw * bqz)
                var ax1_y = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (bqx * bqx + bqz * bqz)
                var ax1_z = Scalar[DTYPE](2) * (bqy * bqz + bqw * bqx)
                # Column 2 (body z-axis in world)
                var ax2_x = Scalar[DTYPE](2) * (bqx * bqz + bqw * bqy)
                var ax2_y = Scalar[DTYPE](2) * (bqy * bqz - bqw * bqx)
                var ax2_z = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (bqx * bqx + bqy * bqy)

                # offset = subtree_com[rootid] - xpos[body]
                var off_x = ref_x - data.xpos[body * 3 + 0]
                var off_y = ref_y - data.xpos[body * 3 + 1]
                var off_z = ref_z - data.xpos[body * 3 + 2]

                # DOF 3: body x-axis rotation
                cdof[(dof_adr + 3) * 6 + 0] = ax0_x
                cdof[(dof_adr + 3) * 6 + 1] = ax0_y
                cdof[(dof_adr + 3) * 6 + 2] = ax0_z
                cdof[(dof_adr + 3) * 6 + 3] = ax0_y * off_z - ax0_z * off_y
                cdof[(dof_adr + 3) * 6 + 4] = ax0_z * off_x - ax0_x * off_z
                cdof[(dof_adr + 3) * 6 + 5] = ax0_x * off_y - ax0_y * off_x

                # DOF 4: body y-axis rotation
                cdof[(dof_adr + 4) * 6 + 0] = ax1_x
                cdof[(dof_adr + 4) * 6 + 1] = ax1_y
                cdof[(dof_adr + 4) * 6 + 2] = ax1_z
                cdof[(dof_adr + 4) * 6 + 3] = ax1_y * off_z - ax1_z * off_y
                cdof[(dof_adr + 4) * 6 + 4] = ax1_z * off_x - ax1_x * off_z
                cdof[(dof_adr + 4) * 6 + 5] = ax1_x * off_y - ax1_y * off_x

                # DOF 5: body z-axis rotation
                cdof[(dof_adr + 5) * 6 + 0] = ax2_x
                cdof[(dof_adr + 5) * 6 + 1] = ax2_y
                cdof[(dof_adr + 5) * 6 + 2] = ax2_z
                cdof[(dof_adr + 5) * 6 + 3] = ax2_y * off_z - ax2_z * off_y
                cdof[(dof_adr + 5) * 6 + 4] = ax2_z * off_x - ax2_x * off_z
                cdof[(dof_adr + 5) * 6 + 5] = ax2_x * off_y - ax2_y * off_x

                # FREE joint sets orientation from qpos directly
                var free_qx = data.qpos[joint.qpos_adr + 3]
                var free_qy = data.qpos[joint.qpos_adr + 4]
                var free_qz = data.qpos[joint.qpos_adr + 5]
                var free_qw = data.qpos[joint.qpos_adr + 6]
                acc_qx = free_qx
                acc_qy = free_qy
                acc_qz = free_qz
                acc_qw = free_qw


def compute_contact_jacobian_row[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    V_SIZE: Int,
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

    For body-body contacts (body_b > 0), the Jacobian is bilateral:
    J_row[i] = J_a[i] - J_b[i], where J_a and J_b are the contributions
    from body_a and body_b respectively. When a joint affects both bodies
    (shared ancestor), the contributions cancel — physically correct.

    For ground contacts (body_b = 0, worldbody), only body_a contributes.

    Reference: MuJoCo mj_jac() in engine_core_util.c:177-227

    Args:
        model: Static model configuration.
        data: Current simulation state.
        cdof: Spatial motion axes per DOF (from compute_cdof).
        contact_body_a: Index of body A in contact.
        contact_body_b: Index of body B (-1 for ground).
        contact_pos_x: Contact point x in world frame.
        contact_pos_y: Contact point y in world frame.
        contact_pos_z: Contact point z in world frame.
        dir_x: Direction vector x (normal or tangent).
        dir_y: Direction vector y (normal or tangent).
        dir_z: Direction vector z (normal or tangent).
        J_row: Output Jacobian row (NV entries).
    """
    for i in range(V_SIZE):
        J_row[i] = Scalar[DTYPE](0)

    for j in range(model.num_joints):
        var joint = model.joints[j]
        var dof_adr = joint.dof_adr

        # Check if this joint affects either contact body
        var affects_a = _joint_affects_body(model, j, contact_body_a)
        var affects_b = (contact_body_b > 0) and _joint_affects_body(
            model, j, contact_body_b
        )

        if not affects_a and not affects_b:
            continue

        var num_dof = 1
        if joint.jnt_type == JNT_FREE:
            num_dof = 6
        elif joint.jnt_type == JNT_BALL:
            num_dof = 3

        # Reference point must match cdof computation
        var ref_body = joint.body_id
        var ref_x: Scalar[DTYPE]
        var ref_y: Scalar[DTYPE]
        var ref_z: Scalar[DTYPE]
        if data.has_subtree_com:
            var root = model.body_rootid[ref_body]
            ref_x = data.subtree_com[root * 3 + 0]
            ref_y = data.subtree_com[root * 3 + 1]
            ref_z = data.subtree_com[root * 3 + 2]
        else:
            ref_x = data.xipos[ref_body * 3 + 0]
            ref_y = data.xipos[ref_body * 3 + 1]
            ref_z = data.xipos[ref_body * 3 + 2]

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


def _joint_affects_body[
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
    joint_idx: Int,
    body_idx: Int,
) -> Bool:
    """Check if a joint affects a body (body is the joint's body or a descendant).
    """
    var joint_body = model.joints[joint_idx].body_id

    if body_idx == joint_body:
        return True

    var current = body_idx
    while current > 0:
        if model.body_parent[current] == joint_body:
            return True
        current = model.body_parent[current]

    return False


# =============================================================================
# Composite Rigid Body Inertia (CRBA helper)
# =============================================================================


def compute_composite_inertia[
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
    mut crb: List[Scalar[DTYPE]],
):
    """Compute composite rigid body inertia for each body.

    Each body's composite inertia is initialized from its own spatial inertia
    at the subtree_com[rootid] reference point, then accumulated bottom-up.

    Storage: 10 floats per body:
        [mass, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    where (cx, cy, cz) is the CoM offset from the reference point,
    and Ixx..Iyz is the rotational inertia about the reference point.

    When subtree_com is provided, all bodies in the same kinematic tree
    share the same reference point (subtree_com[rootid]), matching MuJoCo.
    Without subtree_com, falls back to xipos[body] (legacy).

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
        # crb stays at xipos (not shifted to subtree_com) since the
        # direct summation mass matrix doesn't consume crb.
        crb[b * 10 + 0] = mass
        crb[b * 10 + 1] = Scalar[DTYPE](0)
        crb[b * 10 + 2] = Scalar[DTYPE](0)
        crb[b * 10 + 3] = Scalar[DTYPE](0)

        # Rotate inertia tensor from inertia frame to world frame:
        # I_world = R @ diag(Ixx, Iyy, Izz) @ R^T
        # where R = xquat * body_iquat (inertia frame in world)
        var Ixx_local = model.body_inertia[b * 3 + 0]
        var Iyy_local = model.body_inertia[b * 3 + 1]
        var Izz_local = model.body_inertia[b * 3 + 2]

        # Compose xquat with body_iquat for inertia rotation
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

        var child_mass = crb[b * 10 + 0]
        if child_mass < Scalar[DTYPE](1e-20):
            continue

        var child_cx = crb[b * 10 + 1]
        var child_cy = crb[b * 10 + 2]
        var child_cz = crb[b * 10 + 3]

        var dx = data.xipos[b * 3 + 0] - data.xipos[parent * 3 + 0]
        var dy = data.xipos[b * 3 + 1] - data.xipos[parent * 3 + 1]
        var dz = data.xipos[b * 3 + 2] - data.xipos[parent * 3 + 2]

        var total_cx = dx + child_cx
        var total_cy = dy + child_cy
        var total_cz = dz + child_cz

        var parent_mass = crb[parent * 10 + 0]
        var parent_cx = crb[parent * 10 + 1]
        var parent_cy = crb[parent * 10 + 2]
        var parent_cz = crb[parent * 10 + 3]

        var new_mass = parent_mass + child_mass

        var new_cx: Scalar[DTYPE] = 0
        var new_cy: Scalar[DTYPE] = 0
        var new_cz: Scalar[DTYPE] = 0
        if new_mass > Scalar[DTYPE](1e-20):
            new_cx = (parent_mass * parent_cx + child_mass * total_cx) / new_mass
            new_cy = (parent_mass * parent_cy + child_mass * total_cy) / new_mass
            new_cz = (parent_mass * parent_cz + child_mass * total_cz) / new_mass

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

        var dc_x = total_cx - new_cx
        var dc_y = total_cy - new_cy
        var dc_z = total_cz - new_cz
        var dc_sq = dc_x * dc_x + dc_y * dc_y + dc_z * dc_z

        new_Ixx = new_Ixx + crb[b * 10 + 4] + child_mass * (dc_sq - dc_x * dc_x)
        new_Iyy = new_Iyy + crb[b * 10 + 5] + child_mass * (dc_sq - dc_y * dc_y)
        new_Izz = new_Izz + crb[b * 10 + 6] + child_mass * (dc_sq - dc_z * dc_z)
        new_Ixy = new_Ixy + crb[b * 10 + 7] - child_mass * dc_x * dc_y
        new_Ixz = new_Ixz + crb[b * 10 + 8] - child_mass * dc_x * dc_z
        new_Iyz = new_Iyz + crb[b * 10 + 9] - child_mass * dc_y * dc_z

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
def compute_composite_inertia_gpu[
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

        var xipos_off = xipos_offset[NQ, NV, NBODY]()
        var dx = (
            state[env, xipos_off + b * 3 + 0]
            - state[env, xipos_off + parent * 3 + 0]
        )
        var dy = (
            state[env, xipos_off + b * 3 + 1]
            - state[env, xipos_off + parent * 3 + 1]
        )
        var dz = (
            state[env, xipos_off + b * 3 + 2]
            - state[env, xipos_off + parent * 3 + 2]
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
def compute_cdof_gpu[
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

    Uses per-body processing with accumulated orientation (matching MuJoCo's
    mj_kinematics xaxis computation). Each joint's axis is rotated by the
    accumulated transform BEFORE that joint's own rotation.

    Writes cdof to workspace buffer instead of InlineArray.
    """

    comptime cdof_idx = ws_cdof_offset()

    # Zero out
    for i in range(NV * 6):
        workspace[env, cdof_idx + i] = 0

    var qpos_off = qpos_offset[NQ, NV]()
    var xpos_off = xpos_offset[NQ, NV, NBODY]()
    var xquat_off = xquat_offset[NQ, NV, NBODY]()
    var xi_off = xipos_offset[NQ, NV, NBODY]()
    var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
    var num_joints = Int(model[0, model_meta_off + MODEL_META_IDX_NJOINT])

    # Process per-body, tracking accumulated orientation
    # Skip worldbody at 0 (no joints)
    for body in range(1, NBODY):
        var body_off = model_body_offset(body)
        var parent = Int(
            rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_PARENT])
        )

        # Get parent's world orientation (worldbody=0 has identity)
        var acc_qx = rebind[Scalar[DTYPE]](
            state[env, xquat_off + parent * 4 + 0]
        )
        var acc_qy = rebind[Scalar[DTYPE]](
            state[env, xquat_off + parent * 4 + 1]
        )
        var acc_qz = rebind[Scalar[DTYPE]](
            state[env, xquat_off + parent * 4 + 2]
        )
        var acc_qw = rebind[Scalar[DTYPE]](
            state[env, xquat_off + parent * 4 + 3]
        )

        # Apply body_quat: acc = parent_quat * body_quat
        var bq_x = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_QUAT_X])
        var bq_y = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_QUAT_Y])
        var bq_z = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_QUAT_Z])
        var bq_w = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_QUAT_W])
        var pre_q = gpu_quat_mul(
            acc_qx,
            acc_qy,
            acc_qz,
            acc_qw,
            bq_x,
            bq_y,
            bq_z,
            bq_w,
        )
        acc_qx = pre_q[0]
        acc_qy = pre_q[1]
        acc_qz = pre_q[2]
        acc_qw = pre_q[3]

        # Body CoM world position
        var com_x = rebind[Scalar[DTYPE]](state[env, xi_off + body * 3 + 0])
        var com_y = rebind[Scalar[DTYPE]](state[env, xi_off + body * 3 + 1])
        var com_z = rebind[Scalar[DTYPE]](state[env, xi_off + body * 3 + 2])

        # Compute xpos_initial: xpos[parent] + R(xquat[parent]) * body_pos
        # Use parent's FINAL orientation (data.xquat[parent]) to rotate body_pos.
        # Do NOT use data.xpos[body] (xpos_final after off-center correction).
        var body_pos_x = rebind[Scalar[DTYPE]](
            model[0, body_off + BODY_IDX_POS_X]
        )
        var body_pos_y = rebind[Scalar[DTYPE]](
            model[0, body_off + BODY_IDX_POS_Y]
        )
        var body_pos_z = rebind[Scalar[DTYPE]](
            model[0, body_off + BODY_IDX_POS_Z]
        )
        var par_qx = rebind[Scalar[DTYPE]](
            state[env, xquat_off + parent * 4 + 0]
        )
        var par_qy = rebind[Scalar[DTYPE]](
            state[env, xquat_off + parent * 4 + 1]
        )
        var par_qz = rebind[Scalar[DTYPE]](
            state[env, xquat_off + parent * 4 + 2]
        )
        var par_qw = rebind[Scalar[DTYPE]](
            state[env, xquat_off + parent * 4 + 3]
        )
        var bpos_w = gpu_quat_rotate(
            par_qx, par_qy, par_qz, par_qw, body_pos_x, body_pos_y, body_pos_z
        )
        # Running position — tracks xpos as joints are applied
        var cx = (
            rebind[Scalar[DTYPE]](state[env, xpos_off + parent * 3 + 0])
            + bpos_w[0]
        )
        var cy = (
            rebind[Scalar[DTYPE]](state[env, xpos_off + parent * 3 + 1])
            + bpos_w[1]
        )
        var cz = (
            rebind[Scalar[DTYPE]](state[env, xpos_off + parent * 3 + 2])
            + bpos_w[2]
        )

        # Process all joints for this body in order
        for j in range(num_joints):
            var joint_off = model_joint_offset[NBODY](j)
            var joint_body = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_BODY_ID])
            )
            if joint_body != body:
                continue

            var jnt_type = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
            )
            var dof_adr = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR])
            )

            if jnt_type == JNT_HINGE:
                var axis_lx = rebind[Scalar[DTYPE]](
                    model[0, joint_off + JOINT_IDX_AXIS_X]
                )
                var axis_ly = rebind[Scalar[DTYPE]](
                    model[0, joint_off + JOINT_IDX_AXIS_Y]
                )
                var axis_lz = rebind[Scalar[DTYPE]](
                    model[0, joint_off + JOINT_IDX_AXIS_Z]
                )
                var jpos_lx = rebind[Scalar[DTYPE]](
                    model[0, joint_off + JOINT_IDX_POS_X]
                )
                var jpos_ly = rebind[Scalar[DTYPE]](
                    model[0, joint_off + JOINT_IDX_POS_Y]
                )
                var jpos_lz = rebind[Scalar[DTYPE]](
                    model[0, joint_off + JOINT_IDX_POS_Z]
                )

                # Rotate axis using accumulated (pre-this-joint) orientation
                var a_w = gpu_quat_rotate(
                    acc_qx,
                    acc_qy,
                    acc_qz,
                    acc_qw,
                    axis_lx,
                    axis_ly,
                    axis_lz,
                )
                var ax = a_w[0]
                var ay = a_w[1]
                var az = a_w[2]

                # Joint anchor = running_xpos + rotate(jnt_pos, acc_quat)
                var jp = gpu_quat_rotate(
                    acc_qx,
                    acc_qy,
                    acc_qz,
                    acc_qw,
                    jpos_lx,
                    jpos_ly,
                    jpos_lz,
                )
                var anc_x = cx + jp[0]
                var anc_y = cy + jp[1]
                var anc_z = cz + jp[2]

                # offset = body_com - joint_anchor (GPU: legacy path)
                # TODO: update to subtree_com when GPU workspace supports it
                var ox = com_x - anc_x
                var oy = com_y - anc_y
                var oz = com_z - anc_z

                workspace[env, cdof_idx + dof_adr * 6 + 0] = ax
                workspace[env, cdof_idx + dof_adr * 6 + 1] = ay
                workspace[env, cdof_idx + dof_adr * 6 + 2] = az
                workspace[env, cdof_idx + dof_adr * 6 + 3] = ay * oz - az * oy
                workspace[env, cdof_idx + dof_adr * 6 + 4] = az * ox - ax * oz
                workspace[env, cdof_idx + dof_adr * 6 + 5] = ax * oy - ay * ox

                # Update accumulated orientation with hinge rotation
                var qpos_adr_val = Int(
                    rebind[Scalar[DTYPE]](
                        model[0, joint_off + JOINT_IDX_QPOS_ADR]
                    )
                )
                var qpos0_val = rebind[Scalar[DTYPE]](
                    model[0, joint_off + JOINT_IDX_QPOS0]
                )
                var angle = (
                    rebind[Scalar[DTYPE]](state[env, qpos_off + qpos_adr_val])
                    - qpos0_val
                )
                var hinge_q = gpu_axis_angle_to_quat(ax, ay, az, angle)
                var new_q = gpu_quat_mul(
                    hinge_q[0],
                    hinge_q[1],
                    hinge_q[2],
                    hinge_q[3],
                    acc_qx,
                    acc_qy,
                    acc_qz,
                    acc_qw,
                )
                acc_qx = new_q[0]
                acc_qy = new_q[1]
                acc_qz = new_q[2]
                acc_qw = new_q[3]

                # Off-center correction: update running xpos
                var vec = gpu_quat_rotate(
                    acc_qx, acc_qy, acc_qz, acc_qw, jpos_lx, jpos_ly, jpos_lz
                )
                cx = anc_x - vec[0]
                cy = anc_y - vec[1]
                cz = anc_z - vec[2]

            elif jnt_type == JNT_SLIDE:
                var axis_lx = rebind[Scalar[DTYPE]](
                    model[0, joint_off + JOINT_IDX_AXIS_X]
                )
                var axis_ly = rebind[Scalar[DTYPE]](
                    model[0, joint_off + JOINT_IDX_AXIS_Y]
                )
                var axis_lz = rebind[Scalar[DTYPE]](
                    model[0, joint_off + JOINT_IDX_AXIS_Z]
                )

                # Rotate axis using accumulated orientation
                var a_w = gpu_quat_rotate(
                    acc_qx,
                    acc_qy,
                    acc_qz,
                    acc_qw,
                    axis_lx,
                    axis_ly,
                    axis_lz,
                )

                workspace[env, cdof_idx + dof_adr * 6 + 3] = a_w[0]
                workspace[env, cdof_idx + dof_adr * 6 + 4] = a_w[1]
                workspace[env, cdof_idx + dof_adr * 6 + 5] = a_w[2]

                # Slide: update running xpos by displacement
                var qpos_adr_val2 = Int(
                    rebind[Scalar[DTYPE]](
                        model[0, joint_off + JOINT_IDX_QPOS_ADR]
                    )
                )
                var qpos0_val2 = rebind[Scalar[DTYPE]](
                    model[0, joint_off + JOINT_IDX_QPOS0]
                )
                var disp = (
                    rebind[Scalar[DTYPE]](state[env, qpos_off + qpos_adr_val2])
                    - qpos0_val2
                )
                cx += disp * a_w[0]
                cy += disp * a_w[1]
                cz += disp * a_w[2]

            elif jnt_type == JNT_FREE:
                workspace[env, cdof_idx + (dof_adr + 0) * 6 + 3] = Scalar[
                    DTYPE
                ](1)
                workspace[env, cdof_idx + (dof_adr + 1) * 6 + 4] = Scalar[
                    DTYPE
                ](1)
                workspace[env, cdof_idx + (dof_adr + 2) * 6 + 5] = Scalar[
                    DTYPE
                ](1)
                workspace[env, cdof_idx + (dof_adr + 3) * 6 + 0] = Scalar[
                    DTYPE
                ](1)
                workspace[env, cdof_idx + (dof_adr + 4) * 6 + 1] = Scalar[
                    DTYPE
                ](1)
                workspace[env, cdof_idx + (dof_adr + 5) * 6 + 2] = Scalar[
                    DTYPE
                ](1)

                # FREE joint sets orientation from qpos
                var qpos_adr_val = Int(
                    rebind[Scalar[DTYPE]](
                        model[0, joint_off + JOINT_IDX_QPOS_ADR]
                    )
                )
                acc_qx = rebind[Scalar[DTYPE]](
                    state[env, qpos_off + qpos_adr_val + 3]
                )
                acc_qy = rebind[Scalar[DTYPE]](
                    state[env, qpos_off + qpos_adr_val + 4]
                )
                acc_qz = rebind[Scalar[DTYPE]](
                    state[env, qpos_off + qpos_adr_val + 5]
                )
                acc_qw = rebind[Scalar[DTYPE]](
                    state[env, qpos_off + qpos_adr_val + 6]
                )


@always_inline
def compute_contact_jacobian_row_gpu[
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
    For ground contacts (body_b = 0, worldbody), only body_a contributes.
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
            while current > 0:
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

        # Check if this joint affects body_b (only if body_b > 0, i.e. not ground)
        var affects_b = False
        if contact_body_b > 0:
            if contact_body_b == joint_body:
                affects_b = True
            else:
                var current_b = contact_body_b
                while current_b > 0:
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

        # Reference body = joint's body CoM (must match cdof computation)
        var xipos_off = xipos_offset[NQ, NV, NBODY]()
        var b_x = rebind[Scalar[DTYPE]](
            state[env, xipos_off + joint_body * 3 + 0]
        )
        var b_y = rebind[Scalar[DTYPE]](
            state[env, xipos_off + joint_body * 3 + 1]
        )
        var b_z = rebind[Scalar[DTYPE]](
            state[env, xipos_off + joint_body * 3 + 2]
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


@always_inline
def compute_angular_jacobian_row_gpu[
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
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    mut J_row: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """Compute angular-only Jacobian row on GPU for torsional/rolling friction.

    Like compute_contact_jacobian_row_gpu but only uses the angular component
    of cdof (no cross product with position offset, no linear component).
    J[dof] = cdof_angular[dof] . dir (bilateral: body_a - body_b).
    """
    comptime cdof_idx = ws_cdof_offset()

    for i in range(V_SIZE):
        J_row[i] = 0

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
            while current > 0:
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

        # Check if this joint affects body_b (only if body_b > 0, i.e. not ground)
        var affects_b = False
        if contact_body_b > 0:
            if contact_body_b == joint_body:
                affects_b = True
            else:
                var current_b = contact_body_b
                while current_b > 0:
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

        for d in range(num_dof):
            var dof_idx = dof_adr + d

            # Angular-only: just dot product of angular cdof with direction
            var ang_x = workspace[env, cdof_idx + dof_idx * 6 + 0]
            var ang_y = workspace[env, cdof_idx + dof_idx * 6 + 1]
            var ang_z = workspace[env, cdof_idx + dof_idx * 6 + 2]

            var val = ang_x * dir_x + ang_y * dir_y + ang_z * dir_z

            if affects_a:
                J_row[dof_idx] += rebind[Scalar[DTYPE]](val)
            if affects_b:
                J_row[dof_idx] -= rebind[Scalar[DTYPE]](val)

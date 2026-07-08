"""Forward Kinematics for Generalized Coordinates engine.

Computes body world positions (xpos, xquat) from joint positions (qpos).

Algorithm (MuJoCo-style):
1. For each body in topological order:
   a. Start with parent's world pose (or world frame if parent=-1)
   b. Transform to joint anchor position
   c. Apply joint rotation/translation from qpos
   d. Add body's local frame offset

For HINGE joints:
- The joint defines a rotation axis at a pivot point
- The body's local frame is offset from this pivot
- Rotating the joint rotates the body around the pivot

Joint transformations:
- FREE: xpos = qpos[0:3], xquat = qpos[3:7]
- BALL: rotation from qpos quaternion at anchor
- SLIDE: translation along axis
- HINGE: rotation around axis at anchor
"""

from layout import LayoutTensor, Layout
from std.gpu import barrier

from .quat_math import (
    quat_mul,
    quat_rotate,
    quat_normalize,
    axis_angle_to_quat,
    gpu_quat_mul,
    gpu_quat_rotate,
    gpu_quat_normalize,
    gpu_axis_angle_to_quat,
)
from ..types import Model, Data, ConeType
from ..joint_types import JNT_FREE, JNT_BALL, JNT_SLIDE, JNT_HINGE
from ..gpu.constants import (
    qpos_offset,
    qvel_offset,
    xpos_offset,
    xquat_offset,
    xipos_offset,
    xvel_offset,
    xangvel_offset,
    site_xpos_offset,
    model_body_offset,
    model_joint_offset,
    model_metadata_offset,
    model_site_offset,
    BODY_IDX_PARENT,
    BODY_IDX_POS_X,
    BODY_IDX_POS_Y,
    BODY_IDX_POS_Z,
    BODY_IDX_QUAT_X,
    BODY_IDX_QUAT_Y,
    BODY_IDX_QUAT_Z,
    BODY_IDX_QUAT_W,
    BODY_IDX_IPOS_X,
    BODY_IDX_IPOS_Y,
    BODY_IDX_IPOS_Z,
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
    SITE_IDX_BODY,
    SITE_IDX_POS_X,
    SITE_IDX_POS_Y,
    SITE_IDX_POS_Z,
)

from ..joint_types import (
    JNT_FREE,
    JNT_BALL,
    JNT_SLIDE,
    JNT_HINGE,
)


# =============================================================================
# Forward Kinematics - Main Function
# =============================================================================


def forward_kinematics[
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
    mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
):
    """Compute body world positions from joint positions.

    Traverses the kinematic tree in topological order, computing each body's
    world position and orientation from its parent and joint transformations.

    Args:
        model: Static model configuration (kinematic tree, body frames).
        data: Mutable state with qpos (input) and xpos/xquat (output).

    After this function:
    - data.xpos contains world positions for all bodies
    - data.xquat contains world orientations for all bodies
    """
    # Initialize worldbody at index 0 (identity pose)
    data.xpos[0] = Scalar[DTYPE](0)
    data.xpos[1] = Scalar[DTYPE](0)
    data.xpos[2] = Scalar[DTYPE](0)
    data.xquat[0] = Scalar[DTYPE](0)
    data.xquat[1] = Scalar[DTYPE](0)
    data.xquat[2] = Scalar[DTYPE](0)
    data.xquat[3] = Scalar[DTYPE](1)
    data.xipos[0] = Scalar[DTYPE](0)
    data.xipos[1] = Scalar[DTYPE](0)
    data.xipos[2] = Scalar[DTYPE](0)

    # Process each body in order (skip worldbody at 0)
    for body in range(1, NBODY):
        # Mocap bodies: use externally-set position directly (not from kinematic tree)
        if model.body_mocap[body]:
            data.xpos[body * 3 + 0] = data.mocap_pos[body * 3 + 0]
            data.xpos[body * 3 + 1] = data.mocap_pos[body * 3 + 1]
            data.xpos[body * 3 + 2] = data.mocap_pos[body * 3 + 2]
            data.xquat[body * 4 + 0] = data.mocap_quat[body * 4 + 0]
            data.xquat[body * 4 + 1] = data.mocap_quat[body * 4 + 1]
            data.xquat[body * 4 + 2] = data.mocap_quat[body * 4 + 2]
            data.xquat[body * 4 + 3] = data.mocap_quat[body * 4 + 3]
            data.xipos[body * 3 + 0] = data.mocap_pos[body * 3 + 0]
            data.xipos[body * 3 + 1] = data.mocap_pos[body * 3 + 1]
            data.xipos[body * 3 + 2] = data.mocap_pos[body * 3 + 2]
            continue

        var parent = model.body_parent[body]

        # Get parent's world pose (parent is always valid, worldbody=0 has identity)
        var parent_px = data.xpos[parent * 3 + 0]
        var parent_py = data.xpos[parent * 3 + 1]
        var parent_pz = data.xpos[parent * 3 + 2]
        var parent_qx = data.xquat[parent * 4 + 0]
        var parent_qy = data.xquat[parent * 4 + 1]
        var parent_qz = data.xquat[parent * 4 + 2]
        var parent_qw = data.xquat[parent * 4 + 3]

        # Count joints for this body
        var joint_count = 0
        for j in range(model.num_joints):
            if model.joints[j].body_id == body:
                joint_count += 1

        var has_joint = joint_count > 0

        # Output position and orientation
        var px: Scalar[DTYPE]
        var py: Scalar[DTYPE]
        var pz: Scalar[DTYPE]
        var qx: Scalar[DTYPE]
        var qy: Scalar[DTYPE]
        var qz: Scalar[DTYPE]
        var qw: Scalar[DTYPE]

        if not has_joint:
            # No joint - body is rigidly attached to parent
            # Just apply the body's local transform
            var local_px = model.body_pos[body * 3 + 0]
            var local_py = model.body_pos[body * 3 + 1]
            var local_pz = model.body_pos[body * 3 + 2]

            var rotated_local = quat_rotate(
                parent_qx,
                parent_qy,
                parent_qz,
                parent_qw,
                local_px,
                local_py,
                local_pz,
            )
            px = parent_px + rotated_local[0]
            py = parent_py + rotated_local[1]
            pz = parent_pz + rotated_local[2]

            var local_qx = model.body_quat[body * 4 + 0]
            var local_qy = model.body_quat[body * 4 + 1]
            var local_qz = model.body_quat[body * 4 + 2]
            var local_qw = model.body_quat[body * 4 + 3]

            var combined = quat_mul(
                parent_qx,
                parent_qy,
                parent_qz,
                parent_qw,
                local_qx,
                local_qy,
                local_qz,
                local_qw,
            )
            qx = combined[0]
            qy = combined[1]
            qz = combined[2]
            qw = combined[3]
        else:
            # Body has one or more joints - MuJoCo-style FK
            #
            # Convention (matching MuJoCo):
            #   1. Body origin = parent_xpos + rotate(body_pos, parent_quat)
            #   2. Joint anchor = body_origin + rotate(jnt_pos, body_quat)
            #   3. Hinge: body orbits around anchor, orientation changes
            #   4. Slide: body translates along axis
            #   5. When jnt_pos=(0,0,0): body stays at body_origin, only rotates

            # Step 1: Body pre-joint world position
            var local_px = model.body_pos[body * 3 + 0]
            var local_py = model.body_pos[body * 3 + 1]
            var local_pz = model.body_pos[body * 3 + 2]

            var body_origin = quat_rotate(
                parent_qx,
                parent_qy,
                parent_qz,
                parent_qw,
                local_px,
                local_py,
                local_pz,
            )

            var cur_px = parent_px + body_origin[0]
            var cur_py = parent_py + body_origin[1]
            var cur_pz = parent_pz + body_origin[2]

            # Pre-joint orientation = parent * body_quat
            var local_qx = model.body_quat[body * 4 + 0]
            var local_qy = model.body_quat[body * 4 + 1]
            var local_qz = model.body_quat[body * 4 + 2]
            var local_qw = model.body_quat[body * 4 + 3]

            var pre_q = quat_mul(
                parent_qx,
                parent_qy,
                parent_qz,
                parent_qw,
                local_qx,
                local_qy,
                local_qz,
                local_qw,
            )
            var cur_qx = pre_q[0]
            var cur_qy = pre_q[1]
            var cur_qz = pre_q[2]
            var cur_qw = pre_q[3]

            # Step 2: Process ALL joints for this body (in order)
            for j in range(model.num_joints):
                if model.joints[j].body_id != body:
                    continue

                var joint = model.joints[j]
                var jnt_type = joint.jnt_type
                var qpos_adr = joint.qpos_adr

                if jnt_type == JNT_FREE:
                    # FREE joint: position and orientation directly from qpos.
                    # MuJoCo qpos layout: [tx, ty, tz, qw, qx, qy, qz]
                    # Our internal quaternion convention: (x, y, z, w)
                    cur_px = data.qpos[qpos_adr + 0]
                    cur_py = data.qpos[qpos_adr + 1]
                    cur_pz = data.qpos[qpos_adr + 2]
                    cur_qw = data.qpos[qpos_adr + 3]  # MuJoCo qpos[3] = qw
                    cur_qx = data.qpos[qpos_adr + 4]  # MuJoCo qpos[4] = qx
                    cur_qy = data.qpos[qpos_adr + 5]  # MuJoCo qpos[5] = qy
                    cur_qz = data.qpos[qpos_adr + 6]  # MuJoCo qpos[6] = qz

                    # Normalize quaternion
                    var normalized = quat_normalize(
                        cur_qx, cur_qy, cur_qz, cur_qw
                    )
                    cur_qx = normalized[0]
                    cur_qy = normalized[1]
                    cur_qz = normalized[2]
                    cur_qw = normalized[3]

                elif jnt_type == JNT_HINGE:
                    # HINGE joint: rotation around axis at anchor
                    # jnt_pos is relative to body (MuJoCo convention)
                    var angle = data.qpos[qpos_adr] - model.qpos0[qpos_adr]

                    var jpos_x = joint.pos_x
                    var jpos_y = joint.pos_y
                    var jpos_z = joint.pos_z

                    # Joint anchor = cur_pos + rotate(jnt_pos, cur_quat)
                    var anchor_off = quat_rotate(
                        cur_qx,
                        cur_qy,
                        cur_qz,
                        cur_qw,
                        jpos_x,
                        jpos_y,
                        jpos_z,
                    )
                    var anchor_x = cur_px + anchor_off[0]
                    var anchor_y = cur_py + anchor_off[1]
                    var anchor_z = cur_pz + anchor_off[2]

                    # Joint axis in body frame → world frame
                    var axis_x = joint.axis_x
                    var axis_y = joint.axis_y
                    var axis_z = joint.axis_z

                    var axis_world = quat_rotate(
                        cur_qx,
                        cur_qy,
                        cur_qz,
                        cur_qw,
                        axis_x,
                        axis_y,
                        axis_z,
                    )

                    # Create rotation quaternion from axis-angle
                    var hinge_quat = axis_angle_to_quat(
                        axis_world[0], axis_world[1], axis_world[2], angle
                    )

                    # Compose rotation: cur_quat = hinge_quat * cur_quat
                    var new_quat = quat_mul(
                        hinge_quat[0],
                        hinge_quat[1],
                        hinge_quat[2],
                        hinge_quat[3],
                        cur_qx,
                        cur_qy,
                        cur_qz,
                        cur_qw,
                    )
                    cur_qx = new_quat[0]
                    cur_qy = new_quat[1]
                    cur_qz = new_quat[2]
                    cur_qw = new_quat[3]

                    # Body orbits around anchor:
                    # new_pos = anchor + rotate(-jnt_pos, new_quat)
                    var neg_off = quat_rotate(
                        cur_qx,
                        cur_qy,
                        cur_qz,
                        cur_qw,
                        -jpos_x,
                        -jpos_y,
                        -jpos_z,
                    )
                    cur_px = anchor_x + neg_off[0]
                    cur_py = anchor_y + neg_off[1]
                    cur_pz = anchor_z + neg_off[2]

                elif jnt_type == JNT_SLIDE:
                    # SLIDE joint: translate along axis (MuJoCo: qpos - qpos0)
                    var displacement = (
                        data.qpos[qpos_adr] - model.qpos0[qpos_adr]
                    )

                    # Joint axis in body frame → world frame
                    var axis_x = joint.axis_x
                    var axis_y = joint.axis_y
                    var axis_z = joint.axis_z

                    var axis_world = quat_rotate(
                        cur_qx,
                        cur_qy,
                        cur_qz,
                        cur_qw,
                        axis_x,
                        axis_y,
                        axis_z,
                    )

                    # Add displacement along world axis
                    cur_px = cur_px + axis_world[0] * displacement
                    cur_py = cur_py + axis_world[1] * displacement
                    cur_pz = cur_pz + axis_world[2] * displacement

                elif jnt_type == JNT_BALL:
                    # BALL joint: rotation from quaternion
                    # Same orbit logic as hinge
                    var ball_qx = data.qpos[qpos_adr + 0]
                    var ball_qy = data.qpos[qpos_adr + 1]
                    var ball_qz = data.qpos[qpos_adr + 2]
                    var ball_qw = data.qpos[qpos_adr + 3]

                    # Normalize
                    var normalized = quat_normalize(
                        ball_qx, ball_qy, ball_qz, ball_qw
                    )
                    ball_qx = normalized[0]
                    ball_qy = normalized[1]
                    ball_qz = normalized[2]
                    ball_qw = normalized[3]

                    var jpos_x = joint.pos_x
                    var jpos_y = joint.pos_y
                    var jpos_z = joint.pos_z

                    # Joint anchor
                    var anchor_off = quat_rotate(
                        cur_qx,
                        cur_qy,
                        cur_qz,
                        cur_qw,
                        jpos_x,
                        jpos_y,
                        jpos_z,
                    )
                    var anchor_x = cur_px + anchor_off[0]
                    var anchor_y = cur_py + anchor_off[1]
                    var anchor_z = cur_pz + anchor_off[2]

                    # Compose rotation
                    var new_quat = quat_mul(
                        ball_qx,
                        ball_qy,
                        ball_qz,
                        ball_qw,
                        cur_qx,
                        cur_qy,
                        cur_qz,
                        cur_qw,
                    )
                    cur_qx = new_quat[0]
                    cur_qy = new_quat[1]
                    cur_qz = new_quat[2]
                    cur_qw = new_quat[3]

                    # Body orbits around anchor
                    var neg_off = quat_rotate(
                        cur_qx,
                        cur_qy,
                        cur_qz,
                        cur_qw,
                        -jpos_x,
                        -jpos_y,
                        -jpos_z,
                    )
                    cur_px = anchor_x + neg_off[0]
                    cur_py = anchor_y + neg_off[1]
                    cur_pz = anchor_z + neg_off[2]

            # Final body world pose
            px = cur_px
            py = cur_py
            pz = cur_pz
            qx = cur_qx
            qy = cur_qy
            qz = cur_qz
            qw = cur_qw

        # Store computed world pose
        data.xpos[body * 3 + 0] = px
        data.xpos[body * 3 + 1] = py
        data.xpos[body * 3 + 2] = pz
        data.xquat[body * 4 + 0] = qx
        data.xquat[body * 4 + 1] = qy
        data.xquat[body * 4 + 2] = qz
        data.xquat[body * 4 + 3] = qw

        # Compute xipos = xpos + rotate(body_ipos, xquat)
        var ipos_x = model.body_ipos[body * 3 + 0]
        var ipos_y = model.body_ipos[body * 3 + 1]
        var ipos_z = model.body_ipos[body * 3 + 2]
        var rotated_ipos = quat_rotate(qx, qy, qz, qw, ipos_x, ipos_y, ipos_z)
        data.xipos[body * 3 + 0] = px + rotated_ipos[0]
        data.xipos[body * 3 + 1] = py + rotated_ipos[1]
        data.xipos[body * 3 + 2] = pz + rotated_ipos[2]

    # Compute site world positions: site_xpos = xpos[body] + rotate(site_pos, xquat[body])
    for site_idx in range(NSITE):
        var s_body = model.site_body[site_idx]
        var sp_x = model.site_pos[site_idx * 3 + 0]
        var sp_y = model.site_pos[site_idx * 3 + 1]
        var sp_z = model.site_pos[site_idx * 3 + 2]
        var bqx = data.xquat[s_body * 4 + 0]
        var bqy = data.xquat[s_body * 4 + 1]
        var bqz = data.xquat[s_body * 4 + 2]
        var bqw = data.xquat[s_body * 4 + 3]
        var rot = quat_rotate(bqx, bqy, bqz, bqw, sp_x, sp_y, sp_z)
        data.site_xpos[site_idx * 3 + 0] = data.xpos[s_body * 3 + 0] + rot[0]
        data.site_xpos[site_idx * 3 + 1] = data.xpos[s_body * 3 + 1] + rot[1]
        data.site_xpos[site_idx * 3 + 2] = data.xpos[s_body * 3 + 2] + rot[2]


# =============================================================================
# Compute Body Velocities from Joint Velocities
# =============================================================================


def compute_body_velocities[
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
    mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
):
    """Compute body world velocities from joint velocities.

    Uses the Jacobian relationship: body_vel = J * qvel
    For simple kinematic chains, this can be computed directly.

    Args:
        model: Static model configuration.
        data: Mutable state with qvel (input) and xvel/xangvel (output).
    """
    # Initialize all velocities to zero
    for i in range(NBODY * 3):
        data.xvel[i] = Scalar[DTYPE](0)
        data.xangvel[i] = Scalar[DTYPE](0)

    # Process each body (skip worldbody at 0, already zero)
    for body in range(1, NBODY):
        var parent = model.body_parent[body]

        # Start with parent's velocity
        var vx = data.xvel[parent * 3 + 0]
        var vy = data.xvel[parent * 3 + 1]
        var vz = data.xvel[parent * 3 + 2]
        var wx = data.xangvel[parent * 3 + 0]
        var wy = data.xangvel[parent * 3 + 1]
        var wz = data.xangvel[parent * 3 + 2]

        # Add velocity from parent's rotation about this body's offset
        var rx = data.xipos[body * 3 + 0] - data.xipos[parent * 3 + 0]
        var ry = data.xipos[body * 3 + 1] - data.xipos[parent * 3 + 1]
        var rz = data.xipos[body * 3 + 2] - data.xipos[parent * 3 + 2]

        # v = parent_v + parent_w x r
        vx = vx + (wy * rz - wz * ry)
        vy = vy + (wz * rx - wx * rz)
        vz = vz + (wx * ry - wy * rx)

        # Apply joint velocities
        for j in range(model.num_joints):
            var joint = model.joints[j]
            if joint.body_id != body:
                continue

            var jnt_type = joint.jnt_type
            var dof_adr = joint.dof_adr

            if jnt_type == JNT_FREE:
                # FREE joint: linear velocity is world-frame in qvel; the
                # ANGULAR part is BODY-LOCAL (MuJoCo convention — the free
                # joint's rotational dof axes are the body-frame axes, see
                # mj_comPos/compute_cdof) and must be rotated to world,
                # exactly like the BALL branch below.
                vx = data.qvel[dof_adr + 0]
                vy = data.qvel[dof_adr + 1]
                vz = data.qvel[dof_adr + 2]
                var fqx = data.xquat[body * 4 + 0]
                var fqy = data.xquat[body * 4 + 1]
                var fqz = data.xquat[body * 4 + 2]
                var fqw = data.xquat[body * 4 + 3]
                var w_world = quat_rotate(
                    fqx,
                    fqy,
                    fqz,
                    fqw,
                    data.qvel[dof_adr + 3],
                    data.qvel[dof_adr + 4],
                    data.qvel[dof_adr + 5],
                )
                wx = w_world[0]
                wy = w_world[1]
                wz = w_world[2]

            elif jnt_type == JNT_BALL:
                # BALL joint: angular velocity from qvel
                # Transform from local to world frame
                var qbx = data.xquat[body * 4 + 0]
                var qby = data.xquat[body * 4 + 1]
                var qbz = data.xquat[body * 4 + 2]
                var qbw = data.xquat[body * 4 + 3]

                var local_wx = data.qvel[dof_adr + 0]
                var local_wy = data.qvel[dof_adr + 1]
                var local_wz = data.qvel[dof_adr + 2]

                var world_w = quat_rotate(
                    qbx, qby, qbz, qbw, local_wx, local_wy, local_wz
                )
                wx = wx + world_w[0]
                wy = wy + world_w[1]
                wz = wz + world_w[2]

            elif jnt_type == JNT_SLIDE:
                # SLIDE joint: velocity along axis
                var vel = data.qvel[dof_adr]

                # Get world-space axis
                var axis_x = joint.axis_x
                var axis_y = joint.axis_y
                var axis_z = joint.axis_z

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
                vx = vx + axis_world[0] * vel
                vy = vy + axis_world[1] * vel
                vz = vz + axis_world[2] * vel

            elif jnt_type == JNT_HINGE:
                # HINGE joint: angular velocity around axis
                var omega = data.qvel[dof_adr]

                # Get world-space axis
                var axis_x = joint.axis_x
                var axis_y = joint.axis_y
                var axis_z = joint.axis_z

                # Rotate axis from body frame to world frame
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
                wx = wx + axis_world[0] * omega
                wy = wy + axis_world[1] * omega
                wz = wz + axis_world[2] * omega

        # Store computed velocities
        data.xvel[body * 3 + 0] = vx
        data.xvel[body * 3 + 1] = vy
        data.xvel[body * 3 + 2] = vz
        data.xangvel[body * 3 + 0] = wx
        data.xangvel[body * 3 + 1] = wy
        data.xangvel[body * 3 + 2] = wz


# =============================================================================
# GPU Forward Kinematics — single body (shared by serial + level-parallel)
# =============================================================================


@no_inline
def fk_body_gpu[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    STATE_SIZE: Int,
    MODEL_SIZE: Int,
    BATCH: Int,
](
    env: Int,
    body: Int,
    num_joints: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
):
    """Compute one body's world pose (xpos/xquat/xipos) from its parent.

    Extracted verbatim from the per-body loop body of forward_kinematics_gpu
    so the serial walk and the level-parallel variant share identical
    arithmetic (bit-identical). Requires the parent body's pose already
    written to `state` (guaranteed by topological / level ordering).
    """
    var qpos_off = qpos_offset[NQ, NV]()
    var xpos_off = xpos_offset[NQ, NV, NBODY]()
    var xquat_off = xquat_offset[NQ, NV, NBODY]()
    var body_off = model_body_offset(body)
    var parent = Int(
        rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_PARENT])
    )

    var body_pos_x = rebind[Scalar[DTYPE]](
        model[0, body_off + BODY_IDX_POS_X]
    )
    var body_pos_y = rebind[Scalar[DTYPE]](
        model[0, body_off + BODY_IDX_POS_Y]
    )
    var body_pos_z = rebind[Scalar[DTYPE]](
        model[0, body_off + BODY_IDX_POS_Z]
    )
    var body_quat_x = rebind[Scalar[DTYPE]](
        model[0, body_off + BODY_IDX_QUAT_X]
    )
    var body_quat_y = rebind[Scalar[DTYPE]](
        model[0, body_off + BODY_IDX_QUAT_Y]
    )
    var body_quat_z = rebind[Scalar[DTYPE]](
        model[0, body_off + BODY_IDX_QUAT_Z]
    )
    var body_quat_w = rebind[Scalar[DTYPE]](
        model[0, body_off + BODY_IDX_QUAT_W]
    )

    # Get parent's world pose (parent is always valid, worldbody=0 has identity)
    var cur_px = rebind[Scalar[DTYPE]](
        state[env, xpos_off + parent * 3 + 0]
    )
    var cur_py = rebind[Scalar[DTYPE]](
        state[env, xpos_off + parent * 3 + 1]
    )
    var cur_pz = rebind[Scalar[DTYPE]](
        state[env, xpos_off + parent * 3 + 2]
    )
    var cur_qx = rebind[Scalar[DTYPE]](
        state[env, xquat_off + parent * 4 + 0]
    )
    var cur_qy = rebind[Scalar[DTYPE]](
        state[env, xquat_off + parent * 4 + 1]
    )
    var cur_qz = rebind[Scalar[DTYPE]](
        state[env, xquat_off + parent * 4 + 2]
    )
    var cur_qw = rebind[Scalar[DTYPE]](
        state[env, xquat_off + parent * 4 + 3]
    )

    # Count joints for this body
    var has_joint = False
    for j in range(num_joints):
        var joint_off = model_joint_offset[NBODY](j)
        var joint_body = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_BODY_ID])
        )
        if joint_body == body:
            has_joint = True
            break

    if not has_joint:
        # No joint - body is rigidly attached to parent
        # Just apply the body's local transform
        var rotated_local = gpu_quat_rotate(
            cur_qx,
            cur_qy,
            cur_qz,
            cur_qw,
            body_pos_x,
            body_pos_y,
            body_pos_z,
        )
        var world_px = cur_px + rotated_local[0]
        var world_py = cur_py + rotated_local[1]
        var world_pz = cur_pz + rotated_local[2]

        var combined = gpu_quat_mul(
            cur_qx,
            cur_qy,
            cur_qz,
            cur_qw,
            body_quat_x,
            body_quat_y,
            body_quat_z,
            body_quat_w,
        )
        var norm_q = gpu_quat_normalize(
            combined[0], combined[1], combined[2], combined[3]
        )

        state[env, xpos_off + body * 3 + 0] = world_px
        state[env, xpos_off + body * 3 + 1] = world_py
        state[env, xpos_off + body * 3 + 2] = world_pz
        state[env, xquat_off + body * 4 + 0] = norm_q[0]
        state[env, xquat_off + body * 4 + 1] = norm_q[1]
        state[env, xquat_off + body * 4 + 2] = norm_q[2]
        state[env, xquat_off + body * 4 + 3] = norm_q[3]

        # Compute xipos = xpos + rotate(body_ipos, xquat)
        var xipos_off = xipos_offset[NQ, NV, NBODY]()
        var ipos_x = rebind[Scalar[DTYPE]](
            model[0, body_off + BODY_IDX_IPOS_X]
        )
        var ipos_y = rebind[Scalar[DTYPE]](
            model[0, body_off + BODY_IDX_IPOS_Y]
        )
        var ipos_z = rebind[Scalar[DTYPE]](
            model[0, body_off + BODY_IDX_IPOS_Z]
        )
        var rot_ipos = gpu_quat_rotate(
            norm_q[0],
            norm_q[1],
            norm_q[2],
            norm_q[3],
            ipos_x,
            ipos_y,
            ipos_z,
        )
        state[env, xipos_off + body * 3 + 0] = world_px + rot_ipos[0]
        state[env, xipos_off + body * 3 + 1] = world_py + rot_ipos[1]
        state[env, xipos_off + body * 3 + 2] = world_pz + rot_ipos[2]
    else:
        # Body has one or more joints - MuJoCo-style FK
        #
        # Convention (matching MuJoCo):
        #   1. Body origin = parent_xpos + rotate(body_pos, parent_quat)
        #   2. Joint anchor = body_origin + rotate(jnt_pos, body_quat)
        #   3. Hinge: body orbits around anchor, orientation changes
        #   4. Slide: body translates along axis
        #   5. When jnt_pos=(0,0,0): body stays at body_origin, only rotates

        # Step 1: Body pre-joint world position
        var body_origin = gpu_quat_rotate(
            cur_qx,
            cur_qy,
            cur_qz,
            cur_qw,
            body_pos_x,
            body_pos_y,
            body_pos_z,
        )
        cur_px = cur_px + body_origin[0]
        cur_py = cur_py + body_origin[1]
        cur_pz = cur_pz + body_origin[2]

        # Pre-joint orientation = parent * body_quat
        var pre_q = gpu_quat_mul(
            cur_qx,
            cur_qy,
            cur_qz,
            cur_qw,
            body_quat_x,
            body_quat_y,
            body_quat_z,
            body_quat_w,
        )
        cur_qx = pre_q[0]
        cur_qy = pre_q[1]
        cur_qz = pre_q[2]
        cur_qw = pre_q[3]

        # Step 2: Process all joints for this body in order
        for j in range(num_joints):
            var joint_off = model_joint_offset[NBODY](j)
            var joint_body = Int(
                rebind[Scalar[DTYPE]](
                    model[0, joint_off + JOINT_IDX_BODY_ID]
                )
            )

            if joint_body != body:
                continue

            var jnt_type = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
            )
            var qpos_adr = Int(
                rebind[Scalar[DTYPE]](
                    model[0, joint_off + JOINT_IDX_QPOS_ADR]
                )
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

            if jnt_type == JNT_FREE:
                # FREE joint: position and orientation directly from qpos.
                # MuJoCo qpos layout: [tx, ty, tz, qw, qx, qy, qz]
                # Our internal quaternion convention: (x, y, z, w)
                cur_px = rebind[Scalar[DTYPE]](
                    state[env, qpos_off + qpos_adr + 0]
                )
                cur_py = rebind[Scalar[DTYPE]](
                    state[env, qpos_off + qpos_adr + 1]
                )
                cur_pz = rebind[Scalar[DTYPE]](
                    state[env, qpos_off + qpos_adr + 2]
                )
                cur_qw = rebind[Scalar[DTYPE]](
                    state[env, qpos_off + qpos_adr + 3]
                )  # MuJoCo qpos[3] = qw
                cur_qx = rebind[Scalar[DTYPE]](
                    state[env, qpos_off + qpos_adr + 4]
                )  # MuJoCo qpos[4] = qx
                cur_qy = rebind[Scalar[DTYPE]](
                    state[env, qpos_off + qpos_adr + 5]
                )  # MuJoCo qpos[5] = qy
                cur_qz = rebind[Scalar[DTYPE]](
                    state[env, qpos_off + qpos_adr + 6]
                )  # MuJoCo qpos[6] = qz

                var normalized = gpu_quat_normalize(
                    cur_qx, cur_qy, cur_qz, cur_qw
                )
                cur_qx = normalized[0]
                cur_qy = normalized[1]
                cur_qz = normalized[2]
                cur_qw = normalized[3]

            elif jnt_type == JNT_HINGE:
                # HINGE joint: rotation around anchor (body-relative jnt_pos)
                var qpos0_val = rebind[Scalar[DTYPE]](
                    model[0, joint_off + JOINT_IDX_QPOS0]
                )
                var angle = (
                    rebind[Scalar[DTYPE]](state[env, qpos_off + qpos_adr])
                    - qpos0_val
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

                # Joint anchor = cur_pos + rotate(jnt_pos, cur_quat)
                var anchor_off = gpu_quat_rotate(
                    cur_qx,
                    cur_qy,
                    cur_qz,
                    cur_qw,
                    jpos_x,
                    jpos_y,
                    jpos_z,
                )
                var anchor_x = cur_px + anchor_off[0]
                var anchor_y = cur_py + anchor_off[1]
                var anchor_z = cur_pz + anchor_off[2]

                # Transform axis to world using current orientation
                var axis_world = gpu_quat_rotate(
                    cur_qx,
                    cur_qy,
                    cur_qz,
                    cur_qw,
                    axis_x,
                    axis_y,
                    axis_z,
                )

                # Create rotation quaternion from axis-angle
                var hinge_quat = gpu_axis_angle_to_quat(
                    axis_world[0], axis_world[1], axis_world[2], angle
                )

                # Compose rotation: cur_quat = hinge_quat * cur_quat
                var new_quat = gpu_quat_mul(
                    hinge_quat[0],
                    hinge_quat[1],
                    hinge_quat[2],
                    hinge_quat[3],
                    cur_qx,
                    cur_qy,
                    cur_qz,
                    cur_qw,
                )
                cur_qx = new_quat[0]
                cur_qy = new_quat[1]
                cur_qz = new_quat[2]
                cur_qw = new_quat[3]

                # Body orbits around anchor
                var neg_off = gpu_quat_rotate(
                    cur_qx,
                    cur_qy,
                    cur_qz,
                    cur_qw,
                    -jpos_x,
                    -jpos_y,
                    -jpos_z,
                )
                cur_px = anchor_x + neg_off[0]
                cur_py = anchor_y + neg_off[1]
                cur_pz = anchor_z + neg_off[2]

            elif jnt_type == JNT_SLIDE:
                # SLIDE joint: translate along axis (MuJoCo: qpos - qpos0)
                var qpos0_val = rebind[Scalar[DTYPE]](
                    model[0, joint_off + JOINT_IDX_QPOS0]
                )
                var displacement = (
                    rebind[Scalar[DTYPE]](state[env, qpos_off + qpos_adr])
                    - qpos0_val
                )

                # Transform axis to world using current orientation
                var axis_world = gpu_quat_rotate(
                    cur_qx,
                    cur_qy,
                    cur_qz,
                    cur_qw,
                    axis_x,
                    axis_y,
                    axis_z,
                )

                # Add displacement along world axis
                cur_px = cur_px + axis_world[0] * displacement
                cur_py = cur_py + axis_world[1] * displacement
                cur_pz = cur_pz + axis_world[2] * displacement

            elif jnt_type == JNT_BALL:
                # BALL joint: rotation around anchor
                var ball_qx = rebind[Scalar[DTYPE]](
                    state[env, qpos_off + qpos_adr + 0]
                )
                var ball_qy = rebind[Scalar[DTYPE]](
                    state[env, qpos_off + qpos_adr + 1]
                )
                var ball_qz = rebind[Scalar[DTYPE]](
                    state[env, qpos_off + qpos_adr + 2]
                )
                var ball_qw = rebind[Scalar[DTYPE]](
                    state[env, qpos_off + qpos_adr + 3]
                )

                var normalized = gpu_quat_normalize(
                    ball_qx, ball_qy, ball_qz, ball_qw
                )
                ball_qx = normalized[0]
                ball_qy = normalized[1]
                ball_qz = normalized[2]
                ball_qw = normalized[3]

                var jpos_x = rebind[Scalar[DTYPE]](
                    model[0, joint_off + JOINT_IDX_POS_X]
                )
                var jpos_y = rebind[Scalar[DTYPE]](
                    model[0, joint_off + JOINT_IDX_POS_Y]
                )
                var jpos_z = rebind[Scalar[DTYPE]](
                    model[0, joint_off + JOINT_IDX_POS_Z]
                )

                # Joint anchor
                var anchor_off = gpu_quat_rotate(
                    cur_qx,
                    cur_qy,
                    cur_qz,
                    cur_qw,
                    jpos_x,
                    jpos_y,
                    jpos_z,
                )
                var anchor_x = cur_px + anchor_off[0]
                var anchor_y = cur_py + anchor_off[1]
                var anchor_z = cur_pz + anchor_off[2]

                # Compose rotation
                var new_quat = gpu_quat_mul(
                    ball_qx,
                    ball_qy,
                    ball_qz,
                    ball_qw,
                    cur_qx,
                    cur_qy,
                    cur_qz,
                    cur_qw,
                )
                cur_qx = new_quat[0]
                cur_qy = new_quat[1]
                cur_qz = new_quat[2]
                cur_qw = new_quat[3]

                # Body orbits around anchor
                var neg_off = gpu_quat_rotate(
                    cur_qx,
                    cur_qy,
                    cur_qz,
                    cur_qw,
                    -jpos_x,
                    -jpos_y,
                    -jpos_z,
                )
                cur_px = anchor_x + neg_off[0]
                cur_py = anchor_y + neg_off[1]
                cur_pz = anchor_z + neg_off[2]

        # Final body world pose
        var world_px = cur_px
        var world_py = cur_py
        var world_pz = cur_pz
        var norm_q = gpu_quat_normalize(cur_qx, cur_qy, cur_qz, cur_qw)

        state[env, xpos_off + body * 3 + 0] = world_px
        state[env, xpos_off + body * 3 + 1] = world_py
        state[env, xpos_off + body * 3 + 2] = world_pz
        state[env, xquat_off + body * 4 + 0] = norm_q[0]
        state[env, xquat_off + body * 4 + 1] = norm_q[1]
        state[env, xquat_off + body * 4 + 2] = norm_q[2]
        state[env, xquat_off + body * 4 + 3] = norm_q[3]

        # Compute xipos = xpos + rotate(body_ipos, xquat)
        var xipos_off = xipos_offset[NQ, NV, NBODY]()
        var ipos_x = rebind[Scalar[DTYPE]](
            model[0, body_off + BODY_IDX_IPOS_X]
        )
        var ipos_y = rebind[Scalar[DTYPE]](
            model[0, body_off + BODY_IDX_IPOS_Y]
        )
        var ipos_z = rebind[Scalar[DTYPE]](
            model[0, body_off + BODY_IDX_IPOS_Z]
        )
        var rot_ipos = gpu_quat_rotate(
            norm_q[0],
            norm_q[1],
            norm_q[2],
            norm_q[3],
            ipos_x,
            ipos_y,
            ipos_z,
        )
        state[env, xipos_off + body * 3 + 0] = world_px + rot_ipos[0]
        state[env, xipos_off + body * 3 + 1] = world_py + rot_ipos[1]
        state[env, xipos_off + body * 3 + 2] = world_pz + rot_ipos[2]


# =============================================================================
# GPU Forward Kinematics Kernel
# =============================================================================


@always_inline
def forward_kinematics_gpu[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    STATE_SIZE: Int,
    MODEL_SIZE: Int,
    BATCH: Int,
    NGEOM: Int = 0,
    NEQUALITY: Int = 0,
    NTENDON: Int = 0,
    NSITE: Int = 0,
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
):
    """Compute body world positions and orientations from qpos (GPU version).

    Traverses the kinematic tree from root to leaves, computing xpos and xquat
    for each body based on joint transformations.

    IMPORTANT: This iterates per-body (not per-joint) to properly accumulate
    transformations for bodies with multiple joints (e.g., rootx + rootz + rooty).
    """
    var qpos_off = qpos_offset[NQ, NV]()
    var xpos_off = xpos_offset[NQ, NV, NBODY]()
    var xquat_off = xquat_offset[NQ, NV, NBODY]()

    var meta_off = model_metadata_offset[NBODY, NJOINT]()
    var num_joints = Int(
        rebind[Scalar[DTYPE]](model[0, meta_off + MODEL_META_IDX_NJOINT])
    )

    # Initialize worldbody at index 0 (identity pose)
    state[env, xpos_off + 0] = Scalar[DTYPE](0)
    state[env, xpos_off + 1] = Scalar[DTYPE](0)
    state[env, xpos_off + 2] = Scalar[DTYPE](0)
    state[env, xquat_off + 0] = Scalar[DTYPE](0)
    state[env, xquat_off + 1] = Scalar[DTYPE](0)
    state[env, xquat_off + 2] = Scalar[DTYPE](0)
    state[env, xquat_off + 3] = Scalar[DTYPE](1)
    var xipos_off_init = xipos_offset[NQ, NV, NBODY]()
    state[env, xipos_off_init + 0] = Scalar[DTYPE](0)
    state[env, xipos_off_init + 1] = Scalar[DTYPE](0)
    state[env, xipos_off_init + 2] = Scalar[DTYPE](0)

    # Process each body in order (skip worldbody at 0)
    for body in range(1, NBODY):
        fk_body_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, STATE_SIZE, MODEL_SIZE, BATCH
        ](env, body, num_joints, state, model)

    # Compute site world positions (GPU): site_xpos = xpos[body] + rotate(site_pos, xquat[body])

    comptime if NSITE > 0:
        var site_xpos_off = site_xpos_offset[NQ, NV, NBODY, MAX_CONTACTS]()
        for site_idx in range(NSITE):
            var site_base = model_site_offset[
                NBODY, NJOINT, NGEOM, NEQUALITY, NTENDON
            ](site_idx)
            var s_body = Int(
                rebind[Scalar[DTYPE]](model[0, site_base + SITE_IDX_BODY])
            )
            var sp_x = rebind[Scalar[DTYPE]](
                model[0, site_base + SITE_IDX_POS_X]
            )
            var sp_y = rebind[Scalar[DTYPE]](
                model[0, site_base + SITE_IDX_POS_Y]
            )
            var sp_z = rebind[Scalar[DTYPE]](
                model[0, site_base + SITE_IDX_POS_Z]
            )
            var bqx = rebind[Scalar[DTYPE]](
                state[env, xquat_off + s_body * 4 + 0]
            )
            var bqy = rebind[Scalar[DTYPE]](
                state[env, xquat_off + s_body * 4 + 1]
            )
            var bqz = rebind[Scalar[DTYPE]](
                state[env, xquat_off + s_body * 4 + 2]
            )
            var bqw = rebind[Scalar[DTYPE]](
                state[env, xquat_off + s_body * 4 + 3]
            )
            var rot = gpu_quat_rotate(bqx, bqy, bqz, bqw, sp_x, sp_y, sp_z)
            state[env, site_xpos_off + site_idx * 3 + 0] = (
                rebind[Scalar[DTYPE]](state[env, xpos_off + s_body * 3 + 0])
                + rot[0]
            )
            state[env, site_xpos_off + site_idx * 3 + 1] = (
                rebind[Scalar[DTYPE]](state[env, xpos_off + s_body * 3 + 1])
                + rot[1]
            )
            state[env, site_xpos_off + site_idx * 3 + 2] = (
                rebind[Scalar[DTYPE]](state[env, xpos_off + s_body * 3 + 2])
                + rot[2]
            )


# =============================================================================
# GPU Forward Kinematics — level-parallel (cooperative across STEP_THREADS)
# =============================================================================


@always_inline
def forward_kinematics_gpu_mt[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    STATE_SIZE: Int,
    MODEL_SIZE: Int,
    BATCH: Int,
](
    env: Int,
    tid: Int,
    n_threads: Int,
    valid_env: Bool,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
):
    """Level-parallel forward kinematics (threads of one env cooperate).

    Bit-identical to forward_kinematics_gpu (with NSITE=0, matching how the RK4
    stage kernel calls it): each body's pose is computed by the SAME fk_body_gpu
    helper — only the *distribution* across threads differs. Bodies are processed
    level by level (root depth first); bodies at the same tree depth are
    independent (none is another's parent), so distributing them across threads
    and writing in any order yields identical arithmetic.

    Barrier discipline: every barrier() is reached by ALL threads in the block
    (valid or invalid env, every packed env) — the barrier count depends only on
    the model tree depth (identical across threads), so there is no deadlock.
    Per-body writes are guarded by valid_env; the barriers are unconditional.

    NOTE: computes body poses only (xpos/xquat/xipos). Sites (NSITE>0) are NOT
    handled here — the RK4 stage path uses NSITE=0, so this matches the serial
    call there exactly. Do not substitute this for forward_kinematics_gpu where
    site positions are required.
    """
    var qpos_off = qpos_offset[NQ, NV]()
    var xpos_off = xpos_offset[NQ, NV, NBODY]()
    var xquat_off = xquat_offset[NQ, NV, NBODY]()
    var meta_off = model_metadata_offset[NBODY, NJOINT]()
    var num_joints = Int(
        rebind[Scalar[DTYPE]](model[0, meta_off + MODEL_META_IDX_NJOINT])
    )

    # Body tree depth (level), derived from body_parent in one forward sweep.
    # Bodies are stored topologically (parent index < child index), so level[b]
    # = level[parent]+1 is well-defined in increasing-b order. All threads build
    # the same array (model-only reads) → identical max_level → identical
    # barrier count across the whole block.
    var level = InlineArray[Int, NBODY](fill=0)
    var max_level = 0
    for b in range(1, NBODY):
        var b_off = model_body_offset(b)
        var p = Int(
            rebind[Scalar[DTYPE]](model[0, b_off + BODY_IDX_PARENT])
        )
        level[b] = level[p] + 1
        if level[b] > max_level:
            max_level = level[b]

    # Worldbody (index 0): identity pose. One writer, then publish via barrier.
    if valid_env and tid == 0:
        state[env, xpos_off + 0] = Scalar[DTYPE](0)
        state[env, xpos_off + 1] = Scalar[DTYPE](0)
        state[env, xpos_off + 2] = Scalar[DTYPE](0)
        state[env, xquat_off + 0] = Scalar[DTYPE](0)
        state[env, xquat_off + 1] = Scalar[DTYPE](0)
        state[env, xquat_off + 2] = Scalar[DTYPE](0)
        state[env, xquat_off + 3] = Scalar[DTYPE](1)
        var xipos_off_init = xipos_offset[NQ, NV, NBODY]()
        state[env, xipos_off_init + 0] = Scalar[DTYPE](0)
        state[env, xipos_off_init + 1] = Scalar[DTYPE](0)
        state[env, xipos_off_init + 2] = Scalar[DTYPE](0)
    barrier()

    # Process bodies level by level; within a level, stripe across threads.
    for lvl in range(1, max_level + 1):
        if valid_env:
            for body in range(1 + tid, NBODY, n_threads):
                if level[body] == lvl:
                    fk_body_gpu[
                        DTYPE,
                        NQ,
                        NV,
                        NBODY,
                        NJOINT,
                        STATE_SIZE,
                        MODEL_SIZE,
                        BATCH,
                    ](env, body, num_joints, state, model)
        barrier()


# =============================================================================
# GPU Compute Body Velocities — single body (shared by serial + level-parallel)
# =============================================================================


@no_inline
def vel_body_gpu[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    STATE_SIZE: Int,
    MODEL_SIZE: Int,
    BATCH: Int,
](
    env: Int,
    body: Int,
    num_joints: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
):
    """Compute one body's world velocity (xvel/xangvel) from its parent.

    Extracted verbatim from the per-body loop of compute_body_velocities_gpu so
    the serial walk and the level-parallel variant share identical arithmetic
    (bit-identical). Requires the parent body's velocity already written.
    """
    var qvel_off = qvel_offset[NQ, NV]()
    var xvel_off = xvel_offset[NQ, NV, NBODY]()
    var xangvel_off = xangvel_offset[NQ, NV, NBODY]()
    var xquat_off = xquat_offset[NQ, NV, NBODY]()
    var body_off = model_body_offset(body)
    var parent = Int(
        rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_PARENT])
    )

    # Start with parent's velocity
    var vx = rebind[Scalar[DTYPE]](state[env, xvel_off + parent * 3 + 0])
    var vy = rebind[Scalar[DTYPE]](state[env, xvel_off + parent * 3 + 1])
    var vz = rebind[Scalar[DTYPE]](state[env, xvel_off + parent * 3 + 2])
    var wx = rebind[Scalar[DTYPE]](state[env, xangvel_off + parent * 3 + 0])
    var wy = rebind[Scalar[DTYPE]](state[env, xangvel_off + parent * 3 + 1])
    var wz = rebind[Scalar[DTYPE]](state[env, xangvel_off + parent * 3 + 2])

    # Add velocity from parent's rotation about this body's offset
    var xipos_off = xipos_offset[NQ, NV, NBODY]()
    var body_px = rebind[Scalar[DTYPE]](
        state[env, xipos_off + body * 3 + 0]
    )
    var body_py = rebind[Scalar[DTYPE]](
        state[env, xipos_off + body * 3 + 1]
    )
    var body_pz = rebind[Scalar[DTYPE]](
        state[env, xipos_off + body * 3 + 2]
    )
    var parent_px = rebind[Scalar[DTYPE]](
        state[env, xipos_off + parent * 3 + 0]
    )
    var parent_py = rebind[Scalar[DTYPE]](
        state[env, xipos_off + parent * 3 + 1]
    )
    var parent_pz = rebind[Scalar[DTYPE]](
        state[env, xipos_off + parent * 3 + 2]
    )

    var rx = body_px - parent_px
    var ry = body_py - parent_py
    var rz = body_pz - parent_pz

    # v = parent_v + parent_w x r
    vx = vx + (wy * rz - wz * ry)
    vy = vy + (wz * rx - wx * rz)
    vz = vz + (wx * ry - wy * rx)

    # Apply joint velocities - accumulate for all joints on this body
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
        var axis_x = rebind[Scalar[DTYPE]](
            model[0, joint_off + JOINT_IDX_AXIS_X]
        )
        var axis_y = rebind[Scalar[DTYPE]](
            model[0, joint_off + JOINT_IDX_AXIS_Y]
        )
        var axis_z = rebind[Scalar[DTYPE]](
            model[0, joint_off + JOINT_IDX_AXIS_Z]
        )

        if jnt_type == JNT_FREE:
            # FREE joint: linear velocity is world-frame in qvel; the ANGULAR
            # part is BODY-LOCAL (MuJoCo convention) and must be rotated to
            # world — mirrors the CPU compute_body_velocities fix.
            vx = rebind[Scalar[DTYPE]](state[env, qvel_off + dof_adr + 0])
            vy = rebind[Scalar[DTYPE]](state[env, qvel_off + dof_adr + 1])
            vz = rebind[Scalar[DTYPE]](state[env, qvel_off + dof_adr + 2])
            var fqx = rebind[Scalar[DTYPE]](
                state[env, xquat_off + body * 4 + 0]
            )
            var fqy = rebind[Scalar[DTYPE]](
                state[env, xquat_off + body * 4 + 1]
            )
            var fqz = rebind[Scalar[DTYPE]](
                state[env, xquat_off + body * 4 + 2]
            )
            var fqw = rebind[Scalar[DTYPE]](
                state[env, xquat_off + body * 4 + 3]
            )
            var w_world = gpu_quat_rotate(
                fqx,
                fqy,
                fqz,
                fqw,
                rebind[Scalar[DTYPE]](state[env, qvel_off + dof_adr + 3]),
                rebind[Scalar[DTYPE]](state[env, qvel_off + dof_adr + 4]),
                rebind[Scalar[DTYPE]](state[env, qvel_off + dof_adr + 5]),
            )
            wx = w_world[0]
            wy = w_world[1]
            wz = w_world[2]

        elif jnt_type == JNT_BALL:
            # BALL joint: add angular velocity from qvel
            wx = wx + rebind[Scalar[DTYPE]](
                state[env, qvel_off + dof_adr + 0]
            )
            wy = wy + rebind[Scalar[DTYPE]](
                state[env, qvel_off + dof_adr + 1]
            )
            wz = wz + rebind[Scalar[DTYPE]](
                state[env, qvel_off + dof_adr + 2]
            )

        elif jnt_type == JNT_SLIDE:
            # SLIDE joint: add velocity along axis
            var vel = rebind[Scalar[DTYPE]](state[env, qvel_off + dof_adr])

            # Rotate axis from body frame to world frame
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

            vx = vx + rotated[0] * vel
            vy = vy + rotated[1] * vel
            vz = vz + rotated[2] * vel

        elif jnt_type == JNT_HINGE:
            # HINGE joint: add angular velocity around axis
            var omega = rebind[Scalar[DTYPE]](
                state[env, qvel_off + dof_adr]
            )

            # Rotate axis from body frame to world frame
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

            wx = wx + rotated[0] * omega
            wy = wy + rotated[1] * omega
            wz = wz + rotated[2] * omega

    # Store computed velocities
    state[env, xvel_off + body * 3 + 0] = vx
    state[env, xvel_off + body * 3 + 1] = vy
    state[env, xvel_off + body * 3 + 2] = vz
    state[env, xangvel_off + body * 3 + 0] = wx
    state[env, xangvel_off + body * 3 + 1] = wy
    state[env, xangvel_off + body * 3 + 2] = wz


# =============================================================================
# GPU Compute Body Velocities Kernel
# =============================================================================


@always_inline
def compute_body_velocities_gpu[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    STATE_SIZE: Int,
    MODEL_SIZE: Int,
    BATCH: Int,
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
):
    """Compute body world velocities from qvel (GPU version).

    IMPORTANT: This iterates per-body (not per-joint) to properly accumulate
    velocities for bodies with multiple joints.
    """
    var qvel_off = qvel_offset[NQ, NV]()
    var xpos_off = xpos_offset[NQ, NV, NBODY]()
    var xquat_off = xquat_offset[NQ, NV, NBODY]()
    var xvel_off = xvel_offset[NQ, NV, NBODY]()
    var xangvel_off = xangvel_offset[NQ, NV, NBODY]()

    var meta_off = model_metadata_offset[NBODY, NJOINT]()
    var num_joints = Int(
        rebind[Scalar[DTYPE]](model[0, meta_off + MODEL_META_IDX_NJOINT])
    )

    # Initialize velocities to zero
    for body in range(NBODY):
        state[env, xvel_off + body * 3 + 0] = Scalar[DTYPE](0)
        state[env, xvel_off + body * 3 + 1] = Scalar[DTYPE](0)
        state[env, xvel_off + body * 3 + 2] = Scalar[DTYPE](0)
        state[env, xangvel_off + body * 3 + 0] = Scalar[DTYPE](0)
        state[env, xangvel_off + body * 3 + 1] = Scalar[DTYPE](0)
        state[env, xangvel_off + body * 3 + 2] = Scalar[DTYPE](0)

    # Process each body in order (skip worldbody at 0, already zero)
    for body in range(1, NBODY):
        vel_body_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, STATE_SIZE, MODEL_SIZE, BATCH
        ](env, body, num_joints, state, model)


# =============================================================================
# GPU Compute Body Velocities — level-parallel (cooperative across STEP_THREADS)
# =============================================================================


@always_inline
def compute_body_velocities_gpu_mt[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    STATE_SIZE: Int,
    MODEL_SIZE: Int,
    BATCH: Int,
](
    env: Int,
    tid: Int,
    n_threads: Int,
    valid_env: Bool,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
):
    """Level-parallel body velocities (threads of one env cooperate).

    Within float32 tolerance of compute_body_velocities_gpu (~1e-9, under the
    1e-6 parallel-path bar): each body uses the SAME vel_body_gpu helper; only
    the distribution across threads differs. Bodies are processed by tree depth
    (root first); same-level bodies are independent.
    Worldbody (0) velocity stays 0 (root); bodies 1..NBODY-1 are overwritten by
    vel_body_gpu, so only body 0 needs zeroing. Barriers are block-wide and
    unconditional (count = 1 + max_level, model-only) → no deadlock; per-body
    writes guarded by valid_env.
    """
    var xvel_off = xvel_offset[NQ, NV, NBODY]()
    var xangvel_off = xangvel_offset[NQ, NV, NBODY]()
    var meta_off = model_metadata_offset[NBODY, NJOINT]()
    var num_joints = Int(
        rebind[Scalar[DTYPE]](model[0, meta_off + MODEL_META_IDX_NJOINT])
    )

    var level = InlineArray[Int, NBODY](fill=0)
    var max_level = 0
    for b in range(1, NBODY):
        var b_off = model_body_offset(b)
        var p = Int(
            rebind[Scalar[DTYPE]](model[0, b_off + BODY_IDX_PARENT])
        )
        level[b] = level[p] + 1
        if level[b] > max_level:
            max_level = level[b]

    # Worldbody (index 0): zero velocity (root). One writer, publish via barrier.
    if valid_env and tid == 0:
        state[env, xvel_off + 0] = Scalar[DTYPE](0)
        state[env, xvel_off + 1] = Scalar[DTYPE](0)
        state[env, xvel_off + 2] = Scalar[DTYPE](0)
        state[env, xangvel_off + 0] = Scalar[DTYPE](0)
        state[env, xangvel_off + 1] = Scalar[DTYPE](0)
        state[env, xangvel_off + 2] = Scalar[DTYPE](0)
    barrier()

    for lvl in range(1, max_level + 1):
        if valid_env:
            for body in range(1 + tid, NBODY, n_threads):
                if level[body] == lvl:
                    vel_body_gpu[
                        DTYPE,
                        NQ,
                        NV,
                        NBODY,
                        NJOINT,
                        STATE_SIZE,
                        MODEL_SIZE,
                        BATCH,
                    ](env, body, num_joints, state, model)
        barrier()

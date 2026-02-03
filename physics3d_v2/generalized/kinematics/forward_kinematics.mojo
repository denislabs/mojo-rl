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

from .quat_math import quat_mul, quat_rotate, quat_normalize, axis_angle_to_quat
from ..types import ModelGC, DataGC
from ..joint_types import JNT_FREE, JNT_BALL, JNT_SLIDE, JNT_HINGE


# =============================================================================
# Forward Kinematics - Main Function
# =============================================================================


fn forward_kinematics[
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
    # Process each body in order (assuming topological ordering)
    for body in range(NBODY):
        var parent = model.body_parent[body]

        # Start with parent's world pose (or identity for world)
        var parent_px: Scalar[DTYPE]
        var parent_py: Scalar[DTYPE]
        var parent_pz: Scalar[DTYPE]
        var parent_qx: Scalar[DTYPE]
        var parent_qy: Scalar[DTYPE]
        var parent_qz: Scalar[DTYPE]
        var parent_qw: Scalar[DTYPE]

        if parent < 0:
            # World parent - start at origin with identity orientation
            parent_px = Scalar[DTYPE](0)
            parent_py = Scalar[DTYPE](0)
            parent_pz = Scalar[DTYPE](0)
            parent_qx = Scalar[DTYPE](0)
            parent_qy = Scalar[DTYPE](0)
            parent_qz = Scalar[DTYPE](0)
            parent_qw = Scalar[DTYPE](1)
        else:
            # Get parent's world pose
            parent_px = data.xpos[parent * 3 + 0]
            parent_py = data.xpos[parent * 3 + 1]
            parent_pz = data.xpos[parent * 3 + 2]
            parent_qx = data.xquat[parent * 4 + 0]
            parent_qy = data.xquat[parent * 4 + 1]
            parent_qz = data.xquat[parent * 4 + 2]
            parent_qw = data.xquat[parent * 4 + 3]

        # Find joint for this body (if any)
        var has_joint = False
        var joint_idx = 0
        for j in range(model.num_joints):
            if model.joints[j].body_id == body:
                has_joint = True
                joint_idx = j
                break

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
                parent_qx, parent_qy, parent_qz, parent_qw,
                local_px, local_py, local_pz
            )
            px = parent_px + rotated_local[0]
            py = parent_py + rotated_local[1]
            pz = parent_pz + rotated_local[2]

            var local_qx = model.body_quat[body * 4 + 0]
            var local_qy = model.body_quat[body * 4 + 1]
            var local_qz = model.body_quat[body * 4 + 2]
            var local_qw = model.body_quat[body * 4 + 3]

            var combined = quat_mul(
                parent_qx, parent_qy, parent_qz, parent_qw,
                local_qx, local_qy, local_qz, local_qw
            )
            qx = combined[0]
            qy = combined[1]
            qz = combined[2]
            qw = combined[3]
        else:
            var joint = model.joints[joint_idx]
            var jnt_type = joint.jnt_type
            var qpos_adr = joint.qpos_adr

            if jnt_type == JNT_FREE:
                # FREE joint: position and orientation directly from qpos
                px = data.qpos[qpos_adr + 0]
                py = data.qpos[qpos_adr + 1]
                pz = data.qpos[qpos_adr + 2]
                qx = data.qpos[qpos_adr + 3]
                qy = data.qpos[qpos_adr + 4]
                qz = data.qpos[qpos_adr + 5]
                qw = data.qpos[qpos_adr + 6]

                # Normalize quaternion
                var normalized = quat_normalize(qx, qy, qz, qw)
                qx = normalized[0]
                qy = normalized[1]
                qz = normalized[2]
                qw = normalized[3]

            elif jnt_type == JNT_HINGE:
                # HINGE joint: rotation around axis at pivot point
                var angle = data.qpos[qpos_adr]

                # Joint pivot position (in parent frame, or world if parent=-1)
                var pivot_local_x = joint.pos_x
                var pivot_local_y = joint.pos_y
                var pivot_local_z = joint.pos_z

                # Transform pivot to world
                var pivot_world = quat_rotate(
                    parent_qx, parent_qy, parent_qz, parent_qw,
                    pivot_local_x, pivot_local_y, pivot_local_z
                )
                var pivot_x = parent_px + pivot_world[0]
                var pivot_y = parent_py + pivot_world[1]
                var pivot_z = parent_pz + pivot_world[2]

                # Joint axis (in parent frame)
                var axis_x = joint.axis_x
                var axis_y = joint.axis_y
                var axis_z = joint.axis_z

                # Transform axis to world
                var axis_world = quat_rotate(
                    parent_qx, parent_qy, parent_qz, parent_qw,
                    axis_x, axis_y, axis_z
                )

                # Create rotation quaternion from axis-angle
                var hinge_quat = axis_angle_to_quat(
                    axis_world[0], axis_world[1], axis_world[2], angle
                )

                # Body's local position (offset from pivot)
                var local_px = model.body_pos[body * 3 + 0]
                var local_py = model.body_pos[body * 3 + 1]
                var local_pz = model.body_pos[body * 3 + 2]

                # The local position is relative to pivot, so rotate it by hinge angle
                var rotated_offset = quat_rotate(
                    hinge_quat[0], hinge_quat[1], hinge_quat[2], hinge_quat[3],
                    local_px, local_py, local_pz
                )

                # Final position = pivot + rotated offset
                px = pivot_x + rotated_offset[0]
                py = pivot_y + rotated_offset[1]
                pz = pivot_z + rotated_offset[2]

                # Final orientation = parent * hinge * local
                var local_qx = model.body_quat[body * 4 + 0]
                var local_qy = model.body_quat[body * 4 + 1]
                var local_qz = model.body_quat[body * 4 + 2]
                var local_qw = model.body_quat[body * 4 + 3]

                # First combine parent with hinge
                var parent_hinge = quat_mul(
                    parent_qx, parent_qy, parent_qz, parent_qw,
                    hinge_quat[0], hinge_quat[1], hinge_quat[2], hinge_quat[3]
                )

                # Then combine with local orientation
                var combined = quat_mul(
                    parent_hinge[0], parent_hinge[1], parent_hinge[2], parent_hinge[3],
                    local_qx, local_qy, local_qz, local_qw
                )
                qx = combined[0]
                qy = combined[1]
                qz = combined[2]
                qw = combined[3]

            elif jnt_type == JNT_SLIDE:
                # SLIDE joint: translate along axis
                var displacement = data.qpos[qpos_adr]

                # Joint axis in parent frame
                var axis_x = joint.axis_x
                var axis_y = joint.axis_y
                var axis_z = joint.axis_z

                # Transform axis to world
                var axis_world = quat_rotate(
                    parent_qx, parent_qy, parent_qz, parent_qw,
                    axis_x, axis_y, axis_z
                )

                # Body's local position
                var local_px = model.body_pos[body * 3 + 0]
                var local_py = model.body_pos[body * 3 + 1]
                var local_pz = model.body_pos[body * 3 + 2]

                # Transform local to world
                var local_world = quat_rotate(
                    parent_qx, parent_qy, parent_qz, parent_qw,
                    local_px, local_py, local_pz
                )

                # Final position = parent + local + displacement * axis
                px = parent_px + local_world[0] + axis_world[0] * displacement
                py = parent_py + local_world[1] + axis_world[1] * displacement
                pz = parent_pz + local_world[2] + axis_world[2] * displacement

                # Orientation unchanged for slide joint
                var local_qx = model.body_quat[body * 4 + 0]
                var local_qy = model.body_quat[body * 4 + 1]
                var local_qz = model.body_quat[body * 4 + 2]
                var local_qw = model.body_quat[body * 4 + 3]

                var combined = quat_mul(
                    parent_qx, parent_qy, parent_qz, parent_qw,
                    local_qx, local_qy, local_qz, local_qw
                )
                qx = combined[0]
                qy = combined[1]
                qz = combined[2]
                qw = combined[3]

            elif jnt_type == JNT_BALL:
                # BALL joint: rotation at anchor
                var ball_qx = data.qpos[qpos_adr + 0]
                var ball_qy = data.qpos[qpos_adr + 1]
                var ball_qz = data.qpos[qpos_adr + 2]
                var ball_qw = data.qpos[qpos_adr + 3]

                # Normalize
                var normalized = quat_normalize(ball_qx, ball_qy, ball_qz, ball_qw)
                ball_qx = normalized[0]
                ball_qy = normalized[1]
                ball_qz = normalized[2]
                ball_qw = normalized[3]

                # Joint pivot
                var pivot_local_x = joint.pos_x
                var pivot_local_y = joint.pos_y
                var pivot_local_z = joint.pos_z

                var pivot_world = quat_rotate(
                    parent_qx, parent_qy, parent_qz, parent_qw,
                    pivot_local_x, pivot_local_y, pivot_local_z
                )
                var pivot_x = parent_px + pivot_world[0]
                var pivot_y = parent_py + pivot_world[1]
                var pivot_z = parent_pz + pivot_world[2]

                # Body offset from pivot
                var local_px = model.body_pos[body * 3 + 0]
                var local_py = model.body_pos[body * 3 + 1]
                var local_pz = model.body_pos[body * 3 + 2]

                # Rotate offset by ball joint
                var rotated_offset = quat_rotate(
                    ball_qx, ball_qy, ball_qz, ball_qw,
                    local_px, local_py, local_pz
                )

                px = pivot_x + rotated_offset[0]
                py = pivot_y + rotated_offset[1]
                pz = pivot_z + rotated_offset[2]

                # Combine orientations
                var local_qx = model.body_quat[body * 4 + 0]
                var local_qy = model.body_quat[body * 4 + 1]
                var local_qz = model.body_quat[body * 4 + 2]
                var local_qw = model.body_quat[body * 4 + 3]

                var parent_ball = quat_mul(
                    parent_qx, parent_qy, parent_qz, parent_qw,
                    ball_qx, ball_qy, ball_qz, ball_qw
                )
                var combined = quat_mul(
                    parent_ball[0], parent_ball[1], parent_ball[2], parent_ball[3],
                    local_qx, local_qy, local_qz, local_qw
                )
                qx = combined[0]
                qy = combined[1]
                qz = combined[2]
                qw = combined[3]
            else:
                # Default: no joint transformation
                px = parent_px
                py = parent_py
                pz = parent_pz
                qx = parent_qx
                qy = parent_qy
                qz = parent_qz
                qw = parent_qw

        # Store computed world pose
        data.xpos[body * 3 + 0] = px
        data.xpos[body * 3 + 1] = py
        data.xpos[body * 3 + 2] = pz
        data.xquat[body * 4 + 0] = qx
        data.xquat[body * 4 + 1] = qy
        data.xquat[body * 4 + 2] = qz
        data.xquat[body * 4 + 3] = qw


# =============================================================================
# Compute Body Velocities from Joint Velocities
# =============================================================================


fn compute_body_velocities[
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

    # Process each body
    for body in range(NBODY):
        var parent = model.body_parent[body]

        # Start with parent's velocity (if any)
        var vx: Scalar[DTYPE] = Scalar[DTYPE](0)
        var vy: Scalar[DTYPE] = Scalar[DTYPE](0)
        var vz: Scalar[DTYPE] = Scalar[DTYPE](0)
        var wx: Scalar[DTYPE] = Scalar[DTYPE](0)
        var wy: Scalar[DTYPE] = Scalar[DTYPE](0)
        var wz: Scalar[DTYPE] = Scalar[DTYPE](0)

        if parent >= 0:
            vx = data.xvel[parent * 3 + 0]
            vy = data.xvel[parent * 3 + 1]
            vz = data.xvel[parent * 3 + 2]
            wx = data.xangvel[parent * 3 + 0]
            wy = data.xangvel[parent * 3 + 1]
            wz = data.xangvel[parent * 3 + 2]

            # Add velocity from parent's rotation about this body's offset
            var rx = data.xpos[body * 3 + 0] - data.xpos[parent * 3 + 0]
            var ry = data.xpos[body * 3 + 1] - data.xpos[parent * 3 + 1]
            var rz = data.xpos[body * 3 + 2] - data.xpos[parent * 3 + 2]

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
                # FREE joint: direct velocity from qvel
                vx = data.qvel[dof_adr + 0]
                vy = data.qvel[dof_adr + 1]
                vz = data.qvel[dof_adr + 2]
                wx = data.qvel[dof_adr + 3]
                wy = data.qvel[dof_adr + 4]
                wz = data.qvel[dof_adr + 5]

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

                var world_w = quat_rotate(qbx, qby, qbz, qbw, local_wx, local_wy, local_wz)
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

                if parent >= 0:
                    var parent_qx = data.xquat[parent * 4 + 0]
                    var parent_qy = data.xquat[parent * 4 + 1]
                    var parent_qz = data.xquat[parent * 4 + 2]
                    var parent_qw = data.xquat[parent * 4 + 3]
                    var axis_world = quat_rotate(
                        parent_qx, parent_qy, parent_qz, parent_qw,
                        axis_x, axis_y, axis_z
                    )
                    vx = vx + axis_world[0] * vel
                    vy = vy + axis_world[1] * vel
                    vz = vz + axis_world[2] * vel
                else:
                    vx = vx + axis_x * vel
                    vy = vy + axis_y * vel
                    vz = vz + axis_z * vel

            elif jnt_type == JNT_HINGE:
                # HINGE joint: angular velocity around axis
                var omega = data.qvel[dof_adr]

                # Get world-space axis
                var axis_x = joint.axis_x
                var axis_y = joint.axis_y
                var axis_z = joint.axis_z

                # For hinge, axis is in parent frame
                if parent >= 0:
                    var parent_qx = data.xquat[parent * 4 + 0]
                    var parent_qy = data.xquat[parent * 4 + 1]
                    var parent_qz = data.xquat[parent * 4 + 2]
                    var parent_qw = data.xquat[parent * 4 + 3]
                    var axis_world = quat_rotate(
                        parent_qx, parent_qy, parent_qz, parent_qw,
                        axis_x, axis_y, axis_z
                    )
                    wx = wx + axis_world[0] * omega
                    wy = wy + axis_world[1] * omega
                    wz = wz + axis_world[2] * omega
                else:
                    wx = wx + axis_x * omega
                    wy = wy + axis_y * omega
                    wz = wz + axis_z * omega

        # Store computed velocities
        data.xvel[body * 3 + 0] = vx
        data.xvel[body * 3 + 1] = vy
        data.xvel[body * 3 + 2] = vz
        data.xangvel[body * 3 + 0] = wx
        data.xangvel[body * 3 + 1] = wy
        data.xangvel[body * 3 + 2] = wz

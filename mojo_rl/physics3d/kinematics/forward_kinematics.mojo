"""Forward kinematics over per-field tensors (migration P2, single-source).

The per-field port of `forward_kinematics_gpu`/`fk_body_gpu`: ONE formula
body (`_fk_body`, arithmetic verbatim from `fk_body_gpu`) consumed by
BOTH targets through `forward_kinematics[target]` — the CPU path
loops envs over `.lt["cpu"]` views of the same tensors the GPU kernel binds.
This is the single-source pattern the migration converges on: no flat slab,
no offset math — operands are exactly the fields FK touches (qpos, body
records, joint records [, site records] -> xpos, xquat, xipos [, site_xpos];
6 operands, 8 with sites — vs 24-op budget).

Differences vs the slab version (deliberate):
- `num_joints` comes from the comptime `NJOINT` instead of a runtime read of
  model metadata (they are equal by construction — the joints record tensor
  has exactly NJOINT records), so FK no longer binds the metadata region.
- Iteration order, joint-type branches and every arithmetic expression are
  UNCHANGED -> same-target outputs are bit-exact vs the slab kernel (gated
  by tests/physics3d/test_fk_fields.mojo).
"""

from std.gpu import thread_idx, block_idx, block_dim
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from .quat_math import (
    gpu_quat_mul,
    gpu_quat_rotate,
    gpu_quat_normalize,
    gpu_axis_angle_to_quat,
)
from ..joint_types import JNT_FREE, JNT_BALL, JNT_SLIDE, JNT_HINGE
from ..fields import Data, Model, Dims
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_MOCAP,
    MODEL_JOINT_SIZE,
    MODEL_SITE_SIZE,
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
    SITE_IDX_BODY,
    SITE_IDX_POS_X,
    SITE_IDX_POS_Y,
    SITE_IDX_POS_Z,
)

comptime FK_TPB: Int = 64


@always_inline
def _fk_body[
    DTYPE: DType,
    NQ: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
](
    env: Int,
    body: Int,
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    xpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    xipos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
):
    """One body's world pose from its parent (arithmetic verbatim from
    `fk_body_gpu`; only the addressing is per-field). Requires the parent's
    pose already written (topological order)."""
    # Mocap body: world pose is an EXTERNAL input, preset into xpos/xquat/xipos
    # before the step (facade `_sync_mocap_to_fields`). Skip the parent-chain FK
    # so the target persists across substeps — verbatim semantics of the legacy
    # `forward_kinematics` mocap override. Flag is 0 for every non-mocap env, so
    # this is an exact no-op there.
    if rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_MOCAP]) != 0:
        return

    var parent = Int(rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_PARENT]))

    var body_pos_x = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_POS_X])
    var body_pos_y = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_POS_Y])
    var body_pos_z = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_POS_Z])
    var body_quat_x = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_QUAT_X])
    var body_quat_y = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_QUAT_Y])
    var body_quat_z = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_QUAT_Z])
    var body_quat_w = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_QUAT_W])

    # Parent world pose (worldbody=0 has identity, written by the env init).
    var cur_px = rebind[Scalar[DTYPE]](xpos[env, parent * 3 + 0])
    var cur_py = rebind[Scalar[DTYPE]](xpos[env, parent * 3 + 1])
    var cur_pz = rebind[Scalar[DTYPE]](xpos[env, parent * 3 + 2])
    var cur_qx = rebind[Scalar[DTYPE]](xquat[env, parent * 4 + 0])
    var cur_qy = rebind[Scalar[DTYPE]](xquat[env, parent * 4 + 1])
    var cur_qz = rebind[Scalar[DTYPE]](xquat[env, parent * 4 + 2])
    var cur_qw = rebind[Scalar[DTYPE]](xquat[env, parent * 4 + 3])

    # Count joints for this body
    var has_joint = False
    for j in range(NJOINT):
        var joint_body = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID])
        )
        if joint_body == body:
            has_joint = True
            break

    if not has_joint:
        # No joint - body is rigidly attached to parent
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

        xpos[env, body * 3 + 0] = world_px
        xpos[env, body * 3 + 1] = world_py
        xpos[env, body * 3 + 2] = world_pz
        xquat[env, body * 4 + 0] = norm_q[0]
        xquat[env, body * 4 + 1] = norm_q[1]
        xquat[env, body * 4 + 2] = norm_q[2]
        xquat[env, body * 4 + 3] = norm_q[3]

        # xipos = xpos + rotate(body_ipos, xquat)
        var ipos_x = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_IPOS_X])
        var ipos_y = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_IPOS_Y])
        var ipos_z = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_IPOS_Z])
        var rot_ipos = gpu_quat_rotate(
            norm_q[0],
            norm_q[1],
            norm_q[2],
            norm_q[3],
            ipos_x,
            ipos_y,
            ipos_z,
        )
        xipos[env, body * 3 + 0] = world_px + rot_ipos[0]
        xipos[env, body * 3 + 1] = world_py + rot_ipos[1]
        xipos[env, body * 3 + 2] = world_pz + rot_ipos[2]
    else:
        # Body has one or more joints - MuJoCo-style FK (see fk_body_gpu).
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

        for j in range(NJOINT):
            var joint_body = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID])
            )
            if joint_body != body:
                continue

            var jnt_type = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE])
            )
            var qpos_adr = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS_ADR])
            )
            var axis_x = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_AXIS_X])
            var axis_y = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_AXIS_Y])
            var axis_z = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_AXIS_Z])

            if jnt_type == JNT_FREE:
                # [tx, ty, tz, qw, qx, qy, qz] (MuJoCo layout)
                cur_px = rebind[Scalar[DTYPE]](qpos[env, qpos_adr + 0])
                cur_py = rebind[Scalar[DTYPE]](qpos[env, qpos_adr + 1])
                cur_pz = rebind[Scalar[DTYPE]](qpos[env, qpos_adr + 2])
                cur_qw = rebind[Scalar[DTYPE]](qpos[env, qpos_adr + 3])
                cur_qx = rebind[Scalar[DTYPE]](qpos[env, qpos_adr + 4])
                cur_qy = rebind[Scalar[DTYPE]](qpos[env, qpos_adr + 5])
                cur_qz = rebind[Scalar[DTYPE]](qpos[env, qpos_adr + 6])

                var normalized = gpu_quat_normalize(
                    cur_qx, cur_qy, cur_qz, cur_qw
                )
                cur_qx = normalized[0]
                cur_qy = normalized[1]
                cur_qz = normalized[2]
                cur_qw = normalized[3]

            elif jnt_type == JNT_HINGE:
                var qpos0_val = rebind[Scalar[DTYPE]](
                    joints[j, JOINT_IDX_QPOS0]
                )
                var angle = (
                    rebind[Scalar[DTYPE]](qpos[env, qpos_adr]) - qpos0_val
                )

                var jpos_x = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_POS_X])
                var jpos_y = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_POS_Y])
                var jpos_z = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_POS_Z])

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

                var axis_world = gpu_quat_rotate(
                    cur_qx,
                    cur_qy,
                    cur_qz,
                    cur_qw,
                    axis_x,
                    axis_y,
                    axis_z,
                )

                var hinge_quat = gpu_axis_angle_to_quat(
                    axis_world[0], axis_world[1], axis_world[2], angle
                )

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
                var qpos0_val = rebind[Scalar[DTYPE]](
                    joints[j, JOINT_IDX_QPOS0]
                )
                var displacement = (
                    rebind[Scalar[DTYPE]](qpos[env, qpos_adr]) - qpos0_val
                )

                var axis_world = gpu_quat_rotate(
                    cur_qx,
                    cur_qy,
                    cur_qz,
                    cur_qw,
                    axis_x,
                    axis_y,
                    axis_z,
                )

                cur_px = cur_px + axis_world[0] * displacement
                cur_py = cur_py + axis_world[1] * displacement
                cur_pz = cur_pz + axis_world[2] * displacement

            elif jnt_type == JNT_BALL:
                var ball_qx = rebind[Scalar[DTYPE]](qpos[env, qpos_adr + 0])
                var ball_qy = rebind[Scalar[DTYPE]](qpos[env, qpos_adr + 1])
                var ball_qz = rebind[Scalar[DTYPE]](qpos[env, qpos_adr + 2])
                var ball_qw = rebind[Scalar[DTYPE]](qpos[env, qpos_adr + 3])

                var normalized = gpu_quat_normalize(
                    ball_qx, ball_qy, ball_qz, ball_qw
                )
                ball_qx = normalized[0]
                ball_qy = normalized[1]
                ball_qz = normalized[2]
                ball_qw = normalized[3]

                var jpos_x = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_POS_X])
                var jpos_y = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_POS_Y])
                var jpos_z = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_POS_Z])

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

        xpos[env, body * 3 + 0] = world_px
        xpos[env, body * 3 + 1] = world_py
        xpos[env, body * 3 + 2] = world_pz
        xquat[env, body * 4 + 0] = norm_q[0]
        xquat[env, body * 4 + 1] = norm_q[1]
        xquat[env, body * 4 + 2] = norm_q[2]
        xquat[env, body * 4 + 3] = norm_q[3]

        # xipos = xpos + rotate(body_ipos, xquat)
        var ipos_x = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_IPOS_X])
        var ipos_y = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_IPOS_Y])
        var ipos_z = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_IPOS_Z])
        var rot_ipos = gpu_quat_rotate(
            norm_q[0],
            norm_q[1],
            norm_q[2],
            norm_q[3],
            ipos_x,
            ipos_y,
            ipos_z,
        )
        xipos[env, body * 3 + 0] = world_px + rot_ipos[0]
        xipos[env, body * 3 + 1] = world_py + rot_ipos[1]
        xipos[env, body * 3 + 2] = world_pz + rot_ipos[2]


@always_inline
def _fk_env[
    DTYPE: DType,
    NQ: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
](
    env: Int,
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    xpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    xipos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
):
    """Full FK for one env: worldbody identity + topological body walk."""
    xpos[env, 0] = Scalar[DTYPE](0)
    xpos[env, 1] = Scalar[DTYPE](0)
    xpos[env, 2] = Scalar[DTYPE](0)
    xquat[env, 0] = Scalar[DTYPE](0)
    xquat[env, 1] = Scalar[DTYPE](0)
    xquat[env, 2] = Scalar[DTYPE](0)
    xquat[env, 3] = Scalar[DTYPE](1)
    xipos[env, 0] = Scalar[DTYPE](0)
    xipos[env, 1] = Scalar[DTYPE](0)
    xipos[env, 2] = Scalar[DTYPE](0)

    for body in range(1, NBODY):
        _fk_body[DTYPE, NQ, NBODY, NJOINT, BATCH](
            env, body, qpos, bodies, joints, xpos, xquat, xipos
        )


@always_inline
def _fk_sites[
    DTYPE: DType,
    NBODY: Int,
    NSITE: Int,
    BATCH: Int,
](
    env: Int,
    sites: LayoutTensor[
        DTYPE, Layout.row_major(NSITE, MODEL_SITE_SIZE), MutAnyOrigin
    ],
    xpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    site_xpos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NSITE * 3), MutAnyOrigin
    ],
):
    """site_xpos = xpos[body] + rotate(site_pos, xquat[body])."""
    for site_idx in range(NSITE):
        _fk_site[DTYPE, NBODY, NSITE, BATCH](
            env, site_idx, sites, xpos, xquat, site_xpos
        )


@always_inline
def _fk_site[
    DTYPE: DType,
    NBODY: Int,
    NSITE: Int,
    BATCH: Int,
](
    env: Int,
    site_idx: Int,
    sites: LayoutTensor[
        DTYPE, Layout.row_major(NSITE, MODEL_SITE_SIZE), MutAnyOrigin
    ],
    xpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    site_xpos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NSITE * 3), MutAnyOrigin
    ],
):
    """One site's world position (extracted verbatim from the
    `_fk_sites` loop body so the serial and _mt schedules share the
    identical arithmetic)."""
    var s_body = Int(rebind[Scalar[DTYPE]](sites[site_idx, SITE_IDX_BODY]))
    var sp_x = rebind[Scalar[DTYPE]](sites[site_idx, SITE_IDX_POS_X])
    var sp_y = rebind[Scalar[DTYPE]](sites[site_idx, SITE_IDX_POS_Y])
    var sp_z = rebind[Scalar[DTYPE]](sites[site_idx, SITE_IDX_POS_Z])
    var bqx = rebind[Scalar[DTYPE]](xquat[env, s_body * 4 + 0])
    var bqy = rebind[Scalar[DTYPE]](xquat[env, s_body * 4 + 1])
    var bqz = rebind[Scalar[DTYPE]](xquat[env, s_body * 4 + 2])
    var bqw = rebind[Scalar[DTYPE]](xquat[env, s_body * 4 + 3])
    var rot = gpu_quat_rotate(bqx, bqy, bqz, bqw, sp_x, sp_y, sp_z)
    site_xpos[env, site_idx * 3 + 0] = (
        rebind[Scalar[DTYPE]](xpos[env, s_body * 3 + 0]) + rot[0]
    )
    site_xpos[env, site_idx * 3 + 1] = (
        rebind[Scalar[DTYPE]](xpos[env, s_body * 3 + 1]) + rot[1]
    )
    site_xpos[env, site_idx * 3 + 2] = (
        rebind[Scalar[DTYPE]](xpos[env, s_body * 3 + 2]) + rot[2]
    )


# ── Launchable kernels ────────────────────────────────────────────────────
def _fk_fields_kernel[
    DTYPE: DType,
    NQ: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
](
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    xpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    xipos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _fk_env[DTYPE, NQ, NBODY, NJOINT, BATCH](
        env, qpos, bodies, joints, xpos, xquat, xipos
    )


def _fk_fields_sites_kernel[
    DTYPE: DType,
    NQ: Int,
    NBODY: Int,
    NJOINT: Int,
    NSITE: Int,
    BATCH: Int,
](
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    sites: LayoutTensor[
        DTYPE, Layout.row_major(NSITE, MODEL_SITE_SIZE), MutAnyOrigin
    ],
    xpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    xipos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    site_xpos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NSITE * 3), MutAnyOrigin
    ],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _fk_env[DTYPE, NQ, NBODY, NJOINT, BATCH](
        env, qpos, bodies, joints, xpos, xquat, xipos
    )
    _fk_sites[DTYPE, NBODY, NSITE, BATCH](
        env, sites, xpos, xquat, site_xpos
    )


# ── Cooperative (_mt) kernels: one block per env, N_THREADS cooperate ─────
# Schedule ported verbatim from the legacy `forward_kinematics_gpu_mt`
# (kinematics/forward_kinematics.mojo): bodies are processed level by level
# (tree depth, derived from parents in one forward sweep); bodies within a
# level are striped across threads; one barrier per level. The per-body
# arithmetic is the SAME `_fk_body` helper the serial kernel calls,
# so outputs are bit-exact vs the serial fields kernel. Unlike the legacy
# mt (which lived inside a packed 2D stage block), the grid here is exact —
# one block per env, no invalid envs — so the legacy `valid_env` guards are
# dropped; barriers stay unconditional (every thread reaches every one).


@always_inline
def _fk_env_mt[
    DTYPE: DType,
    NQ: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
    N_THREADS: Int,
](
    env: Int,
    tid: Int,
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    xpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    xipos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
):
    """Level-parallel FK for one env (all block threads must call this —
    it contains barriers)."""
    # Body tree depth (level): model-only reads, identical in every thread
    # -> identical max_level -> identical barrier count.
    var level = InlineArray[Int, NBODY](fill=0)
    var max_level = 0
    for b in range(1, NBODY):
        var p = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_PARENT]))
        level[b] = level[p] + 1
        if level[b] > max_level:
            max_level = level[b]

    # Worldbody (index 0): identity pose. One writer, publish via barrier.
    if tid == 0:
        xpos[env, 0] = Scalar[DTYPE](0)
        xpos[env, 1] = Scalar[DTYPE](0)
        xpos[env, 2] = Scalar[DTYPE](0)
        xquat[env, 0] = Scalar[DTYPE](0)
        xquat[env, 1] = Scalar[DTYPE](0)
        xquat[env, 2] = Scalar[DTYPE](0)
        xquat[env, 3] = Scalar[DTYPE](1)
        xipos[env, 0] = Scalar[DTYPE](0)
        xipos[env, 1] = Scalar[DTYPE](0)
        xipos[env, 2] = Scalar[DTYPE](0)
    barrier()

    # Process bodies level by level; within a level, stripe across threads.
    for lvl in range(1, max_level + 1):
        for body in range(1 + tid, NBODY, N_THREADS):
            if level[body] == lvl:
                _fk_body[DTYPE, NQ, NBODY, NJOINT, BATCH](
                    env, body, qpos, bodies, joints, xpos, xquat, xipos
                )
        barrier()


def _fk_fields_mt_kernel[
    DTYPE: DType,
    NQ: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
    N_THREADS: Int,
](
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    xpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    xipos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
):
    var env = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    _fk_env_mt[DTYPE, NQ, NBODY, NJOINT, BATCH, N_THREADS](
        env, tid, qpos, bodies, joints, xpos, xquat, xipos
    )


def _fk_fields_sites_mt_kernel[
    DTYPE: DType,
    NQ: Int,
    NBODY: Int,
    NJOINT: Int,
    NSITE: Int,
    BATCH: Int,
    N_THREADS: Int,
](
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    sites: LayoutTensor[
        DTYPE, Layout.row_major(NSITE, MODEL_SITE_SIZE), MutAnyOrigin
    ],
    xpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    xipos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    site_xpos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NSITE * 3), MutAnyOrigin
    ],
):
    var env = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    _fk_env_mt[DTYPE, NQ, NBODY, NJOINT, BATCH, N_THREADS](
        env, tid, qpos, bodies, joints, xpos, xquat, xipos
    )
    # Body poses published by the level loop's final barrier; sites are
    # independent per site -> stripe across threads (same `_fk_site`
    # helper as the serial kernel; no legacy sites-mt exists, the legacy mt
    # runs NSITE=0 only — striping independent writes stays bit-exact).
    for site_idx in range(tid, NSITE, N_THREADS):
        _fk_site[DTYPE, NBODY, NSITE, BATCH](
            env, site_idx, sites, xpos, xquat, site_xpos
        )


# ── Public single-source dispatcher ───────────────────────────────────────
def forward_kinematics[
    target: StaticString,
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int = 0,
    NEQUALITY: Int = 0,
    NTENDON: Int = 0,
    NSITE: Int = 0,
    NEXCLUDE: Int = 0,
    NMESH_VERTS: Int = 0,
    BATCH: Int = 1,
    PARALLEL: Bool = False,
    # Appended, not grouped with NEXCLUDE — see `fields.Model`.
    NPAIR: Int = 0,
](
    mut d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH],
    mut m: Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQUALITY, ntendon=NTENDON, nsite=NSITE, nexclude=NEXCLUDE, nmesh_verts=NMESH_VERTS, npair=NPAIR]],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Forward kinematics qpos -> xpos/xquat/xipos (+ site_xpos) over
    per-field tensors. CPU: loop over envs; GPU: one thread per env, or —
    with PARALLEL=True — one block per env with NV cooperating threads
    (level-parallel, bit-exact vs the serial kernel). CPU ignores
    PARALLEL."""
    comptime L_QPOS = Layout.row_major(BATCH, NQ)
    comptime L_XPOS = Layout.row_major(BATCH, NBODY * 3)
    comptime L_XQUAT = Layout.row_major(BATCH, NBODY * 4)
    comptime L_BODY = Layout.row_major(NBODY, MODEL_BODY_SIZE)
    comptime L_JOINT = Layout.row_major(NJOINT, MODEL_JOINT_SIZE)

    comptime if target == "cpu":
        var qpos_v = d.qpos.lt["cpu", L_QPOS]()
        var bodies_v = m.bodies.lt["cpu", L_BODY]()
        var joints_v = m.joints.lt["cpu", L_JOINT]()
        var xpos_v = d.xpos.lt["cpu", L_XPOS]()
        var xquat_v = d.xquat.lt["cpu", L_XQUAT]()
        var xipos_v = d.xipos.lt["cpu", L_XPOS]()
        for e in range(BATCH):
            _fk_env[DTYPE, NQ, NBODY, NJOINT, BATCH](
                e, qpos_v, bodies_v, joints_v, xpos_v, xquat_v, xipos_v
            )
        comptime if NSITE > 0:
            comptime L_SITE_REC = Layout.row_major(NSITE, MODEL_SITE_SIZE)
            comptime L_SITE_X = Layout.row_major(BATCH, NSITE * 3)
            var sites_v = m.sites.lt["cpu", L_SITE_REC]()
            var sitex_v = d.site_xpos.lt["cpu", L_SITE_X]()
            for e in range(BATCH):
                _fk_sites[DTYPE, NBODY, NSITE, BATCH](
                    e, sites_v, xpos_v, xquat_v, sitex_v
                )
    elif PARALLEL:
        # Cooperative within-env schedule (legacy STEP_THREADS = NV).
        var c = ctx.value()
        comptime MT_T = NV
        comptime if NSITE > 0:
            comptime L_SITE_REC = Layout.row_major(NSITE, MODEL_SITE_SIZE)
            comptime L_SITE_X = Layout.row_major(BATCH, NSITE * 3)
            c.enqueue_function[
                _fk_fields_sites_mt_kernel[
                    DTYPE, NQ, NBODY, NJOINT, NSITE, BATCH, MT_T
                ]
            ](
                d.qpos.lt["gpu", L_QPOS](),
                m.bodies.lt["gpu", L_BODY](),
                m.joints.lt["gpu", L_JOINT](),
                m.sites.lt["gpu", L_SITE_REC](),
                d.xpos.lt["gpu", L_XPOS](),
                d.xquat.lt["gpu", L_XQUAT](),
                d.xipos.lt["gpu", L_XPOS](),
                d.site_xpos.lt["gpu", L_SITE_X](),
                grid_dim=(BATCH,),
                block_dim=(MT_T,),
            )
        else:
            c.enqueue_function[
                _fk_fields_mt_kernel[DTYPE, NQ, NBODY, NJOINT, BATCH, MT_T]
            ](
                d.qpos.lt["gpu", L_QPOS](),
                m.bodies.lt["gpu", L_BODY](),
                m.joints.lt["gpu", L_JOINT](),
                d.xpos.lt["gpu", L_XPOS](),
                d.xquat.lt["gpu", L_XQUAT](),
                d.xipos.lt["gpu", L_XPOS](),
                grid_dim=(BATCH,),
                block_dim=(MT_T,),
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + FK_TPB - 1) // FK_TPB
        comptime if NSITE > 0:
            comptime L_SITE_REC = Layout.row_major(NSITE, MODEL_SITE_SIZE)
            comptime L_SITE_X = Layout.row_major(BATCH, NSITE * 3)
            c.enqueue_function[
                _fk_fields_sites_kernel[DTYPE, NQ, NBODY, NJOINT, NSITE, BATCH]
            ](
                d.qpos.lt["gpu", L_QPOS](),
                m.bodies.lt["gpu", L_BODY](),
                m.joints.lt["gpu", L_JOINT](),
                m.sites.lt["gpu", L_SITE_REC](),
                d.xpos.lt["gpu", L_XPOS](),
                d.xquat.lt["gpu", L_XQUAT](),
                d.xipos.lt["gpu", L_XPOS](),
                d.site_xpos.lt["gpu", L_SITE_X](),
                grid_dim=(BLOCKS,),
                block_dim=(FK_TPB,),
            )
        else:
            c.enqueue_function[
                _fk_fields_kernel[DTYPE, NQ, NBODY, NJOINT, BATCH]
            ](
                d.qpos.lt["gpu", L_QPOS](),
                m.bodies.lt["gpu", L_BODY](),
                m.joints.lt["gpu", L_JOINT](),
                d.xpos.lt["gpu", L_XPOS](),
                d.xquat.lt["gpu", L_XQUAT](),
                d.xipos.lt["gpu", L_XPOS](),
                grid_dim=(BLOCKS,),
                block_dim=(FK_TPB,),
            )


# ══════════════════════════════════════════════════════════════════════════
# Body velocities (xvel/xangvel from qvel) — per-field port of
# vel_body_gpu / compute_body_velocities_gpu (arithmetic verbatim).
# Operands: qvel, xquat, xipos + body/joint records -> xvel, xangvel (7).
# ══════════════════════════════════════════════════════════════════════════


@always_inline
def _vel_body[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
](
    env: Int,
    body: Int,
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    xpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    xipos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    xvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin],
    xangvel: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
):
    """One body's world velocity from its parent. Requires the parent's
    velocity already written (topological order).

    Each joint contributes about its OWN world axis and anchor — MuJoCo's
    `d.xaxis[j]` / `d.xanchor[j]`, built in `mj_kinematics` from the frame
    that has already absorbed every PRECEDING joint of the same body. This
    walks that same running (position, quaternion) frame, so the joint-loop
    structure below is deliberately the mirror image of `_fk_body`'s (and of
    `dynamics/cdof.mojo`, which needs the identical walk to place `cdof`).
    Keep the three in step.
    """
    var parent = Int(rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_PARENT]))

    var vx = rebind[Scalar[DTYPE]](xvel[env, parent * 3 + 0])
    var vy = rebind[Scalar[DTYPE]](xvel[env, parent * 3 + 1])
    var vz = rebind[Scalar[DTYPE]](xvel[env, parent * 3 + 2])
    var wx = rebind[Scalar[DTYPE]](xangvel[env, parent * 3 + 0])
    var wy = rebind[Scalar[DTYPE]](xangvel[env, parent * 3 + 1])
    var wz = rebind[Scalar[DTYPE]](xangvel[env, parent * 3 + 2])

    var body_px = rebind[Scalar[DTYPE]](xipos[env, body * 3 + 0])
    var body_py = rebind[Scalar[DTYPE]](xipos[env, body * 3 + 1])
    var body_pz = rebind[Scalar[DTYPE]](xipos[env, body * 3 + 2])
    var parent_px = rebind[Scalar[DTYPE]](xipos[env, parent * 3 + 0])
    var parent_py = rebind[Scalar[DTYPE]](xipos[env, parent * 3 + 1])
    var parent_pz = rebind[Scalar[DTYPE]](xipos[env, parent * 3 + 2])

    var rx = body_px - parent_px
    var ry = body_py - parent_py
    var rz = body_pz - parent_pz

    # v = parent_v + parent_w x r
    vx = vx + (wy * rz - wz * ry)
    vy = vy + (wz * rx - wx * rz)
    vz = vz + (wx * ry - wy * rx)

    # This body's CoM: the point every joint's linear contribution is taken
    # about, since the propagation above carries velocities at the CoM.
    var com_x = body_px
    var com_y = body_py
    var com_z = body_pz

    var ipos_x = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_IPOS_X])
    var ipos_y = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_IPOS_Y])
    var ipos_z = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_IPOS_Z])

    # ── Running joint frame (see the docstring). Seed = this body's frame
    # BEFORE any of its joints: parent pose composed with the body's own
    # fixed pos/quat. Ignoring the fixed `quat=` here is what broke
    # humanoid's `lower_waist` (quat="1.000 0 -.002 0") even though it
    # carries a single joint.
    var body_pos_x = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_POS_X])
    var body_pos_y = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_POS_Y])
    var body_pos_z = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_POS_Z])
    var body_quat_x = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_QUAT_X])
    var body_quat_y = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_QUAT_Y])
    var body_quat_z = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_QUAT_Z])
    var body_quat_w = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_QUAT_W])

    var par_qx = rebind[Scalar[DTYPE]](xquat[env, parent * 4 + 0])
    var par_qy = rebind[Scalar[DTYPE]](xquat[env, parent * 4 + 1])
    var par_qz = rebind[Scalar[DTYPE]](xquat[env, parent * 4 + 2])
    var par_qw = rebind[Scalar[DTYPE]](xquat[env, parent * 4 + 3])

    var body_origin = gpu_quat_rotate(
        par_qx, par_qy, par_qz, par_qw, body_pos_x, body_pos_y, body_pos_z
    )
    var cur_px = (
        rebind[Scalar[DTYPE]](xpos[env, parent * 3 + 0]) + body_origin[0]
    )
    var cur_py = (
        rebind[Scalar[DTYPE]](xpos[env, parent * 3 + 1]) + body_origin[1]
    )
    var cur_pz = (
        rebind[Scalar[DTYPE]](xpos[env, parent * 3 + 2]) + body_origin[2]
    )

    var pre_q = gpu_quat_mul(
        par_qx,
        par_qy,
        par_qz,
        par_qw,
        body_quat_x,
        body_quat_y,
        body_quat_z,
        body_quat_w,
    )
    var cur_qx = pre_q[0]
    var cur_qy = pre_q[1]
    var cur_qz = pre_q[2]
    var cur_qw = pre_q[3]

    for j in range(NJOINT):
        var joint_body = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID])
        )
        if joint_body != body:
            continue

        var jnt_type = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var dof_adr = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR])
        )
        var qpos_adr = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS_ADR])
        )
        var axis_x = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_AXIS_X])
        var axis_y = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_AXIS_Y])
        var axis_z = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_AXIS_Z])
        var jpos_x = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_POS_X])
        var jpos_y = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_POS_Y])
        var jpos_z = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_POS_Z])

        # World axis + anchor in the CURRENT frame == MuJoCo's `d.xaxis[j]`
        # and `d.xanchor[j]`, both taken before this joint is applied
        # (engine_core_smooth.c, mj_kinematics).
        var aw = gpu_quat_rotate(
            cur_qx, cur_qy, cur_qz, cur_qw, axis_x, axis_y, axis_z
        )
        var anchor_off = gpu_quat_rotate(
            cur_qx, cur_qy, cur_qz, cur_qw, jpos_x, jpos_y, jpos_z
        )
        var anc_x = cur_px + anchor_off[0]
        var anc_y = cur_py + anchor_off[1]
        var anc_z = cur_pz + anchor_off[2]

        # Lever from this joint's anchor to the body CoM.
        var lev_x = com_x - anc_x
        var lev_y = com_y - anc_y
        var lev_z = com_z - anc_z

        if jnt_type == JNT_FREE:
            # linear world-frame from qvel; angular is BODY-LOCAL, rotate
            vx = rebind[Scalar[DTYPE]](qvel[env, dof_adr + 0])
            vy = rebind[Scalar[DTYPE]](qvel[env, dof_adr + 1])
            vz = rebind[Scalar[DTYPE]](qvel[env, dof_adr + 2])
            var fqx = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 0])
            var fqy = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 1])
            var fqz = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 2])
            var fqw = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 3])
            var w_world = gpu_quat_rotate(
                fqx,
                fqy,
                fqz,
                fqw,
                rebind[Scalar[DTYPE]](qvel[env, dof_adr + 3]),
                rebind[Scalar[DTYPE]](qvel[env, dof_adr + 4]),
                rebind[Scalar[DTYPE]](qvel[env, dof_adr + 5]),
            )
            wx = w_world[0]
            wy = w_world[1]
            wz = w_world[2]

            # Angular -> linear coupling, exactly as the BALL/HINGE branches
            # below do it. MuJoCo's free-joint qvel[0:3] is the velocity of
            # the body FRAME ORIGIN (it matches `mj_objectVelocity` with
            # mjOBJ_XBODY bit-for-bit), but every other velocity in this
            # function is carried at the body CoM — the propagation above is
            # `v = parent_v + parent_w x (xipos_b - xipos_parent)`. So the
            # free root has to be moved from its origin to its CoM by
            # `w x (R_b * ipos_b)`, the free joint's lever arm (its anchor IS
            # the body frame origin, hence no `- jnt_pos` term here).
            #
            # Omitting it made `Data.xvel` hold the ORIGIN velocity for a
            # free-rooted body and the CoM velocity for every other body, so
            # `sensors.subtree_linvel` — which mass-averages xvel and is
            # specified at the CoM — was wrong for every free-rooted model by
            # a rigid offset of w x r. Invisible on the planar domains
            # (cheetah/walker/hopper are slide+hinge rooted, and their
            # `subtreelinvel` gates pass to 1e-10); dm_control's humanoid is
            # the first free-rooted model to gate this term, and it showed up
            # as a flat ~0.07 m/s error in `com_velocity` from step 0.
            # A free joint is always the sole joint of its body (MuJoCo
            # forbids combining it), so `lev` — which is taken in the
            # PRE-joint frame — does not apply; the free anchor is the
            # body's own post-joint origin, i.e. `xipos - xpos`.
            var flev = gpu_quat_rotate(fqx, fqy, fqz, fqw, ipos_x, ipos_y, ipos_z)
            vx = vx + (wy * flev[2] - wz * flev[1])
            vy = vy + (wz * flev[0] - wx * flev[2])
            vz = vz + (wx * flev[1] - wy * flev[0])

            # Advance the frame (unreachable in practice, per the note above).
            cur_px = rebind[Scalar[DTYPE]](qpos[env, qpos_adr + 0])
            cur_py = rebind[Scalar[DTYPE]](qpos[env, qpos_adr + 1])
            cur_pz = rebind[Scalar[DTYPE]](qpos[env, qpos_adr + 2])
            var fnorm = gpu_quat_normalize(
                rebind[Scalar[DTYPE]](qpos[env, qpos_adr + 4]),
                rebind[Scalar[DTYPE]](qpos[env, qpos_adr + 5]),
                rebind[Scalar[DTYPE]](qpos[env, qpos_adr + 6]),
                rebind[Scalar[DTYPE]](qpos[env, qpos_adr + 3]),
            )
            cur_qx = fnorm[0]
            cur_qy = fnorm[1]
            cur_qz = fnorm[2]
            cur_qw = fnorm[3]

        elif jnt_type == JNT_BALL:
            # ⚠ UNGATED: no model in the repo uses a ball joint, so nothing
            # here is covered by test_body_velocities_vs_mujoco. MuJoCo takes
            # ball qvel in the CHILD body frame — `mj_comPos` builds the three
            # rotational cdof rows from the columns of `d.xmat[body]`, with the
            # comment "I_3 rotation in child frame (assume no subsequent
            # rotations)" — which is why this rotates by the body quat rather
            # than by the running frame. The frame advance below also mirrors
            # `_fk_body`, which composes the ball quat as a WORLD rotation
            # where MuJoCo composes it as a LOCAL one; the two disagree, and
            # `_fk_body` is the one to check first. Verify both against MuJoCo
            # before trusting a ball-jointed model.
            var bqx = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 0])
            var bqy = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 1])
            var bqz = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 2])
            var bqw = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 3])
            var bw = gpu_quat_rotate(
                bqx,
                bqy,
                bqz,
                bqw,
                rebind[Scalar[DTYPE]](qvel[env, dof_adr + 0]),
                rebind[Scalar[DTYPE]](qvel[env, dof_adr + 1]),
                rebind[Scalar[DTYPE]](qvel[env, dof_adr + 2]),
            )
            wx = wx + bw[0]
            wy = wy + bw[1]
            wz = wz + bw[2]

            vx = vx + (bw[1] * lev_z - bw[2] * lev_y)
            vy = vy + (bw[2] * lev_x - bw[0] * lev_z)
            vz = vz + (bw[0] * lev_y - bw[1] * lev_x)

            var bnorm = gpu_quat_normalize(
                rebind[Scalar[DTYPE]](qpos[env, qpos_adr + 0]),
                rebind[Scalar[DTYPE]](qpos[env, qpos_adr + 1]),
                rebind[Scalar[DTYPE]](qpos[env, qpos_adr + 2]),
                rebind[Scalar[DTYPE]](qpos[env, qpos_adr + 3]),
            )
            var bq_new = gpu_quat_mul(
                bnorm[0],
                bnorm[1],
                bnorm[2],
                bnorm[3],
                cur_qx,
                cur_qy,
                cur_qz,
                cur_qw,
            )
            cur_qx = bq_new[0]
            cur_qy = bq_new[1]
            cur_qz = bq_new[2]
            cur_qw = bq_new[3]
            var bneg = gpu_quat_rotate(
                cur_qx, cur_qy, cur_qz, cur_qw, -jpos_x, -jpos_y, -jpos_z
            )
            cur_px = anc_x + bneg[0]
            cur_py = anc_y + bneg[1]
            cur_pz = anc_z + bneg[2]

        elif jnt_type == JNT_SLIDE:
            var vel = rebind[Scalar[DTYPE]](qvel[env, dof_adr])
            vx = vx + aw[0] * vel
            vy = vy + aw[1] * vel
            vz = vz + aw[2] * vel

            var qpos0_val = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS0])
            var disp = (
                rebind[Scalar[DTYPE]](qpos[env, qpos_adr]) - qpos0_val
            )
            cur_px = cur_px + aw[0] * disp
            cur_py = cur_py + aw[1] * disp
            cur_pz = cur_pz + aw[2] * disp

        elif jnt_type == JNT_HINGE:
            var omega = rebind[Scalar[DTYPE]](qvel[env, dof_adr])
            wx = wx + aw[0] * omega
            wy = wy + aw[1] * omega
            wz = wz + aw[2] * omega

            # ANGULAR -> LINEAR COUPLING. Spinning about the joint axis also
            # translates this body's CoM, by omega_j x (CoM - anchor). Missing
            # until 2026-07-29, which left `xvel` wrong for every body below a
            # hinge (~7% on walker2d) while `xangvel` stayed exact. The SLIDE
            # branch always had its linear term, which is why only hinge/ball
            # chains were affected. Gated by
            # tests/physics3d/test_body_velocities_vs_mujoco.mojo.
            vx = vx + (aw[1] * lev_z - aw[2] * lev_y) * omega
            vy = vy + (aw[2] * lev_x - aw[0] * lev_z) * omega
            vz = vz + (aw[0] * lev_y - aw[1] * lev_x) * omega

            # Advance the frame past this joint, mirroring `_fk_body`.
            var qpos0_val = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS0])
            var angle = (
                rebind[Scalar[DTYPE]](qpos[env, qpos_adr]) - qpos0_val
            )
            var hinge_quat = gpu_axis_angle_to_quat(aw[0], aw[1], aw[2], angle)
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
            var neg_off = gpu_quat_rotate(
                cur_qx, cur_qy, cur_qz, cur_qw, -jpos_x, -jpos_y, -jpos_z
            )
            cur_px = anc_x + neg_off[0]
            cur_py = anc_y + neg_off[1]
            cur_pz = anc_z + neg_off[2]

    xvel[env, body * 3 + 0] = vx
    xvel[env, body * 3 + 1] = vy
    xvel[env, body * 3 + 2] = vz
    xangvel[env, body * 3 + 0] = wx
    xangvel[env, body * 3 + 1] = wy
    xangvel[env, body * 3 + 2] = wz


@always_inline
def _body_velocities_env[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
](
    env: Int,
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    xpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    xipos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    xvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin],
    xangvel: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
):
    for body in range(NBODY):
        xvel[env, body * 3 + 0] = Scalar[DTYPE](0)
        xvel[env, body * 3 + 1] = Scalar[DTYPE](0)
        xvel[env, body * 3 + 2] = Scalar[DTYPE](0)
        xangvel[env, body * 3 + 0] = Scalar[DTYPE](0)
        xangvel[env, body * 3 + 1] = Scalar[DTYPE](0)
        xangvel[env, body * 3 + 2] = Scalar[DTYPE](0)

    for body in range(1, NBODY):
        _vel_body[DTYPE, NQ, NV, NBODY, NJOINT, BATCH](
            env, body, qpos, qvel, xpos, xquat, xipos, bodies, joints, xvel,
            xangvel,
        )


def _body_velocities_fields_kernel[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
](
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    xpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    xipos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    xvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin],
    xangvel: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _body_velocities_env[DTYPE, NQ, NV, NBODY, NJOINT, BATCH](
        env, qpos, qvel, xpos, xquat, xipos, bodies, joints, xvel, xangvel
    )


# ── Cooperative (_mt) kernel — schedule from the legacy
# `compute_body_velocities_gpu_mt`: worldbody zeroed by tid 0, then bodies
# level by level, striped across threads, one barrier per level. Per-body
# arithmetic is the SAME `_vel_body` helper as the serial kernel.
# Only body 0 needs zeroing (bodies 1..NBODY-1 are overwritten). Grid is
# exact (one block per env) -> legacy valid_env guards dropped.
def _body_velocities_fields_mt_kernel[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
    N_THREADS: Int,
](
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    xpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    xipos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    xvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin],
    xangvel: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
):
    var env = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var level = InlineArray[Int, NBODY](fill=0)
    var max_level = 0
    for b in range(1, NBODY):
        var p = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_PARENT]))
        level[b] = level[p] + 1
        if level[b] > max_level:
            max_level = level[b]

    # Worldbody (index 0): zero velocity (root). One writer, then barrier.
    if tid == 0:
        xvel[env, 0] = Scalar[DTYPE](0)
        xvel[env, 1] = Scalar[DTYPE](0)
        xvel[env, 2] = Scalar[DTYPE](0)
        xangvel[env, 0] = Scalar[DTYPE](0)
        xangvel[env, 1] = Scalar[DTYPE](0)
        xangvel[env, 2] = Scalar[DTYPE](0)
    barrier()

    for lvl in range(1, max_level + 1):
        for body in range(1 + tid, NBODY, N_THREADS):
            if level[body] == lvl:
                _vel_body[DTYPE, NQ, NV, NBODY, NJOINT, BATCH](
                    env, body, qpos, qvel, xpos, xquat, xipos, bodies, joints,
                    xvel, xangvel,
                )
        barrier()


def compute_body_velocities[
    target: StaticString,
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int = 0,
    NEQUALITY: Int = 0,
    NTENDON: Int = 0,
    NSITE: Int = 0,
    NEXCLUDE: Int = 0,
    NMESH_VERTS: Int = 0,
    BATCH: Int = 1,
    PARALLEL: Bool = False,
    # Appended, not grouped with NEXCLUDE — see `fields.Model`.
    NPAIR: Int = 0,
](
    mut d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH],
    mut m: Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQUALITY, ntendon=NTENDON, nsite=NSITE, nexclude=NEXCLUDE, nmesh_verts=NMESH_VERTS, npair=NPAIR]],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Body world velocities qvel -> xvel/xangvel (needs FK products), both
    targets, one body. PARALLEL=True (GPU only): level-parallel cooperative
    kernel, bit-exact vs serial. CPU ignores PARALLEL."""
    comptime L_NQ = Layout.row_major(BATCH, NQ)
    comptime L_NV = Layout.row_major(BATCH, NV)
    comptime L_B3 = Layout.row_major(BATCH, NBODY * 3)
    comptime L_B4 = Layout.row_major(BATCH, NBODY * 4)
    comptime L_BODY = Layout.row_major(NBODY, MODEL_BODY_SIZE)
    comptime L_JOINT = Layout.row_major(NJOINT, MODEL_JOINT_SIZE)

    comptime if target == "cpu":
        var qpos_v = d.qpos.lt["cpu", L_NQ]()
        var qvel_v = d.qvel.lt["cpu", L_NV]()
        var xpos_v = d.xpos.lt["cpu", L_B3]()
        var xquat_v = d.xquat.lt["cpu", L_B4]()
        var xipos_v = d.xipos.lt["cpu", L_B3]()
        var bodies_v = m.bodies.lt["cpu", L_BODY]()
        var joints_v = m.joints.lt["cpu", L_JOINT]()
        var xvel_v = d.xvel.lt["cpu", L_B3]()
        var xangvel_v = d.xangvel.lt["cpu", L_B3]()
        for e in range(BATCH):
            _body_velocities_env[DTYPE, NQ, NV, NBODY, NJOINT, BATCH](
                e, qpos_v, qvel_v, xpos_v, xquat_v, xipos_v, bodies_v,
                joints_v, xvel_v, xangvel_v,
            )
    elif PARALLEL:
        var c = ctx.value()
        comptime MT_T = NV
        c.enqueue_function[
            _body_velocities_fields_mt_kernel[
                DTYPE, NQ, NV, NBODY, NJOINT, BATCH, MT_T
            ]
        ](
            d.qpos.lt["gpu", L_NQ](),
            d.qvel.lt["gpu", L_NV](),
            d.xpos.lt["gpu", L_B3](),
            d.xquat.lt["gpu", L_B4](),
            d.xipos.lt["gpu", L_B3](),
            m.bodies.lt["gpu", L_BODY](),
            m.joints.lt["gpu", L_JOINT](),
            d.xvel.lt["gpu", L_B3](),
            d.xangvel.lt["gpu", L_B3](),
            grid_dim=(BATCH,),
            block_dim=(MT_T,),
        )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + FK_TPB - 1) // FK_TPB
        c.enqueue_function[
            _body_velocities_fields_kernel[DTYPE, NQ, NV, NBODY, NJOINT, BATCH]
        ](
            d.qpos.lt["gpu", L_NQ](),
            d.qvel.lt["gpu", L_NV](),
            d.xpos.lt["gpu", L_B3](),
            d.xquat.lt["gpu", L_B4](),
            d.xipos.lt["gpu", L_B3](),
            m.bodies.lt["gpu", L_BODY](),
            m.joints.lt["gpu", L_JOINT](),
            d.xvel.lt["gpu", L_B3](),
            d.xangvel.lt["gpu", L_B3](),
            grid_dim=(BLOCKS,),
            block_dim=(FK_TPB,),
        )

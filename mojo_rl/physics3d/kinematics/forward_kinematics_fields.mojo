"""Forward kinematics over per-field tensors (migration P2, single-source).

The per-field port of `forward_kinematics_gpu`/`fk_body_gpu`: ONE formula
body (`_fk_body_fields`, arithmetic verbatim from `fk_body_gpu`) consumed by
BOTH targets through `forward_kinematics_fields[target]` — the CPU path
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
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from .quat_math import (
    gpu_quat_mul,
    gpu_quat_rotate,
    gpu_quat_normalize,
    gpu_axis_angle_to_quat,
)
from ..joint_types import JNT_FREE, JNT_BALL, JNT_SLIDE, JNT_HINGE
from ..fields import DataFields, ModelFields
from ..gpu.constants import (
    MODEL_BODY_SIZE,
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
def _fk_body_fields[
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
def _fk_env_fields[
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
        _fk_body_fields[DTYPE, NQ, NBODY, NJOINT, BATCH](
            env, body, qpos, bodies, joints, xpos, xquat, xipos
        )


@always_inline
def _fk_sites_fields[
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
    _fk_env_fields[DTYPE, NQ, NBODY, NJOINT, BATCH](
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
    _fk_env_fields[DTYPE, NQ, NBODY, NJOINT, BATCH](
        env, qpos, bodies, joints, xpos, xquat, xipos
    )
    _fk_sites_fields[DTYPE, NBODY, NSITE, BATCH](
        env, sites, xpos, xquat, site_xpos
    )


# ── Public single-source dispatcher ───────────────────────────────────────
def forward_kinematics_fields[
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
](
    mut d: DataFields[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH],
    mut m: ModelFields[
        DTYPE,
        NV,
        NBODY,
        NJOINT,
        NGEOM,
        NEQUALITY,
        NTENDON,
        NSITE,
        NEXCLUDE,
        NMESH_VERTS,
    ],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Forward kinematics qpos -> xpos/xquat/xipos (+ site_xpos) over
    per-field tensors. CPU: loop over envs; GPU: one thread per env. Both
    run the same `_fk_env_fields` body."""
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
            _fk_env_fields[DTYPE, NQ, NBODY, NJOINT, BATCH](
                e, qpos_v, bodies_v, joints_v, xpos_v, xquat_v, xipos_v
            )
        comptime if NSITE > 0:
            comptime L_SITE_REC = Layout.row_major(NSITE, MODEL_SITE_SIZE)
            comptime L_SITE_X = Layout.row_major(BATCH, NSITE * 3)
            var sites_v = m.sites.lt["cpu", L_SITE_REC]()
            var sitex_v = d.site_xpos.lt["cpu", L_SITE_X]()
            for e in range(BATCH):
                _fk_sites_fields[DTYPE, NBODY, NSITE, BATCH](
                    e, sites_v, xpos_v, xquat_v, sitex_v
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

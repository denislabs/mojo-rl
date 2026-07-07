"""CRBA mass matrix over per-field tensors (migration P2, single-source).

Per-field port of `compute_mass_matrix_full_gpu` (dynamics/mass_matrix.mojo)
— arithmetic verbatim. Reads `scratch.cdof`, writes `scratch.M` (owned
tensors, replacing the ws_cdof/ws_M regions). Per-thread scratch (dof_body,
world-frame inertia, subtree mask) stays in InlineArrays.

Operands: xquat, xipos, subtree_com + body/joint records + cdof -> M
(7 operands). `num_joints` is the comptime NJOINT (no metadata read)."""

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from ..kinematics.quat_math import gpu_quat_mul
from ..joint_types import JNT_FREE, JNT_BALL
from ..fields import DataFields, ModelFields, DynamicsScratch
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    BODY_IDX_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    BODY_IDX_PARENT,
    BODY_IDX_IQUAT_X,
    BODY_IDX_IQUAT_Y,
    BODY_IDX_IQUAT_Z,
    BODY_IDX_IQUAT_W,
    BODY_IDX_ROOTID,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_DOF_ADR,
)

comptime MM_TPB: Int = 64


@always_inline
def _ensure_positive[N: Int]() -> Int:
    return N if N > 0 else 1


@always_inline
def _mass_matrix_env_fields[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
](
    env: Int,
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    xipos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    subtree_com: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    cdof: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * 6), MutAnyOrigin],
    M: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
):
    """Full NV x NV mass matrix for one env (arithmetic verbatim from
    compute_mass_matrix_full_gpu)."""
    for i in range(NV * NV):
        M[env, i] = 0

    comptime NV_SAFE = _ensure_positive[NV]()
    var dof_body = InlineArray[Int, NV_SAFE](uninitialized=True)
    for i in range(NV):
        dof_body[i] = 0

    for j in range(NJOINT):
        var jnt_type = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var body_id = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID])
        )
        var dof_adr = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR])
        )

        var ndof = 1
        if jnt_type == JNT_FREE:
            ndof = 6
        elif jnt_type == JNT_BALL:
            ndof = 3
        for d in range(ndof):
            dof_body[dof_adr + d] = body_id

    # Per-body world-frame inertia tensor
    comptime I_WORLD_SIZE = _ensure_positive[NBODY * 6]()
    var I_world = InlineArray[Scalar[DTYPE], I_WORLD_SIZE](uninitialized=True)
    for b in range(NBODY):
        var Ixx_l = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IXX])
        var Iyy_l = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IYY])
        var Izz_l = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IZZ])

        var bqx = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 0])
        var bqy = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 1])
        var bqz = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 2])
        var bqw = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 3])
        var iqx = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IQUAT_X])
        var iqy = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IQUAT_Y])
        var iqz = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IQUAT_Z])
        var iqw = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IQUAT_W])
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

    # Subtree membership mask (O(1) lookups in the inner loop)
    comptime MASK_SIZE = _ensure_positive[NBODY * NBODY]()
    var subtree_mask = InlineArray[Bool, MASK_SIZE](fill=False)
    for k in range(NBODY):
        subtree_mask[k * NBODY + k] = True
        var current = k
        while current > 0:
            var parent = Int(
                rebind[Scalar[DTYPE]](bodies[current, BODY_IDX_PARENT])
            )
            subtree_mask[k * NBODY + parent] = True
            current = parent

    # M[i,j] via direct body summation with subtree mask lookup
    for i in range(NV):
        var body_i = dof_body[i]
        var ai0 = cdof[env, i * 6 + 0]
        var ai1 = cdof[env, i * 6 + 1]
        var ai2 = cdof[env, i * 6 + 2]
        var li0 = cdof[env, i * 6 + 3]
        var li1 = cdof[env, i * 6 + 4]
        var li2 = cdof[env, i * 6 + 5]

        for j in range(i, NV):
            var body_j = dof_body[j]
            var aj0 = cdof[env, j * 6 + 0]
            var aj1 = cdof[env, j * 6 + 1]
            var aj2 = cdof[env, j * 6 + 2]
            var lj0 = cdof[env, j * 6 + 3]
            var lj1 = cdof[env, j * 6 + 4]
            var lj2 = cdof[env, j * 6 + 5]

            var mij: M.element_type = 0

            for k in range(NBODY):
                if not subtree_mask[k * NBODY + body_i]:
                    continue
                if not subtree_mask[k * NBODY + body_j]:
                    continue

                var mk = rebind[Scalar[DTYPE]](bodies[k, BODY_IDX_MASS])
                var pk0 = rebind[Scalar[DTYPE]](xipos[env, k * 3 + 0])
                var pk1 = rebind[Scalar[DTYPE]](xipos[env, k * 3 + 1])
                var pk2 = rebind[Scalar[DTYPE]](xipos[env, k * 3 + 2])

                var ri_root = Int(
                    rebind[Scalar[DTYPE]](bodies[body_i, BODY_IDX_ROOTID])
                )
                var pi0 = rebind[Scalar[DTYPE]](
                    subtree_com[env, ri_root * 3 + 0]
                )
                var pi1 = rebind[Scalar[DTYPE]](
                    subtree_com[env, ri_root * 3 + 1]
                )
                var pi2 = rebind[Scalar[DTYPE]](
                    subtree_com[env, ri_root * 3 + 2]
                )
                var di0 = pk0 - pi0
                var di1 = pk1 - pi1
                var di2 = pk2 - pi2
                var vki0 = li0 + ai1 * di2 - ai2 * di1
                var vki1 = li1 + ai2 * di0 - ai0 * di2
                var vki2 = li2 + ai0 * di1 - ai1 * di0

                var rj_root = Int(
                    rebind[Scalar[DTYPE]](bodies[body_j, BODY_IDX_ROOTID])
                )
                var pj0 = rebind[Scalar[DTYPE]](
                    subtree_com[env, rj_root * 3 + 0]
                )
                var pj1 = rebind[Scalar[DTYPE]](
                    subtree_com[env, rj_root * 3 + 1]
                )
                var pj2 = rebind[Scalar[DTYPE]](
                    subtree_com[env, rj_root * 3 + 2]
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

            M[env, i * NV + j] = mij
            if i != j:
                M[env, j * NV + i] = mij


def _mass_matrix_fields_kernel[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
](
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    xipos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    subtree_com: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    cdof: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * 6), MutAnyOrigin],
    M: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _mass_matrix_env_fields[DTYPE, NV, NBODY, NJOINT, BATCH](
        env, xquat, xipos, subtree_com, bodies, joints, cdof, M
    )


def compute_mass_matrix_fields[
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
    mut scratch: DynamicsScratch[DTYPE, NV, NBODY, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """CRBA mass matrix from FK products + cdof, both targets, one body.
    Reads `scratch.cdof`, writes `scratch.M`."""
    comptime L_B3 = Layout.row_major(BATCH, NBODY * 3)
    comptime L_B4 = Layout.row_major(BATCH, NBODY * 4)
    comptime L_BODY = Layout.row_major(NBODY, MODEL_BODY_SIZE)
    comptime L_JOINT = Layout.row_major(NJOINT, MODEL_JOINT_SIZE)
    comptime L_CDOF = Layout.row_major(BATCH, NV * 6)
    comptime L_M = Layout.row_major(BATCH, NV * NV)

    comptime if target == "cpu":
        var xquat_v = d.xquat.lt["cpu", L_B4]()
        var xipos_v = d.xipos.lt["cpu", L_B3]()
        var stcom_v = d.subtree_com.lt["cpu", L_B3]()
        var bodies_v = m.bodies.lt["cpu", L_BODY]()
        var joints_v = m.joints.lt["cpu", L_JOINT]()
        var cdof_v = scratch.cdof.lt["cpu", L_CDOF]()
        var M_v = scratch.M.lt["cpu", L_M]()
        for e in range(BATCH):
            _mass_matrix_env_fields[DTYPE, NV, NBODY, NJOINT, BATCH](
                e, xquat_v, xipos_v, stcom_v, bodies_v, joints_v, cdof_v, M_v
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + MM_TPB - 1) // MM_TPB
        c.enqueue_function[
            _mass_matrix_fields_kernel[DTYPE, NV, NBODY, NJOINT, BATCH]
        ](
            d.xquat.lt["gpu", L_B4](),
            d.xipos.lt["gpu", L_B3](),
            d.subtree_com.lt["gpu", L_B3](),
            m.bodies.lt["gpu", L_BODY](),
            m.joints.lt["gpu", L_JOINT](),
            scratch.cdof.lt["gpu", L_CDOF](),
            scratch.M.lt["gpu", L_M](),
            grid_dim=(BLOCKS,),
            block_dim=(MM_TPB,),
        )

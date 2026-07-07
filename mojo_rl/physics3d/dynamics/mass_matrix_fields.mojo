"""CRBA mass matrix over per-field tensors (migration P2, single-source).

Per-field port of `compute_mass_matrix_full_gpu` (dynamics/mass_matrix.mojo)
— arithmetic verbatim. Reads `scratch.cdof`, writes `scratch.M` (owned
tensors, replacing the ws_cdof/ws_M regions). Per-thread scratch (dof_body,
world-frame inertia, subtree mask) stays in InlineArrays.

Operands: xquat, xipos, subtree_com + body/joint records + cdof -> M
(7 operands). `num_joints` is the comptime NJOINT (no metadata read)."""

from std.gpu import thread_idx, block_idx, block_dim, barrier
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
def _mm_setup_env_fields[
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
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    mut dof_body: InlineArray[Int, _ensure_positive[NV]()],
    mut I_world: InlineArray[Scalar[DTYPE], _ensure_positive[NBODY * 6]()],
    mut subtree_mask: InlineArray[Bool, _ensure_positive[NBODY * NBODY]()],
):
    """CRBA setup (dof->body map, per-body world inertia, subtree mask).
    Extracted verbatim from `_mass_matrix_env_fields` so the serial and _mt
    schedules share identical arithmetic (model + FK-state reads only ->
    every thread computes the same values)."""
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
    for k in range(NBODY):
        subtree_mask[k * NBODY + k] = True
        var current = k
        while current > 0:
            var parent = Int(
                rebind[Scalar[DTYPE]](bodies[current, BODY_IDX_PARENT])
            )
            subtree_mask[k * NBODY + parent] = True
            current = parent


@always_inline
def _mm_row_env_fields[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    BATCH: Int,
](
    env: Int,
    i: Int,
    xipos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    subtree_com: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    cdof: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * 6), MutAnyOrigin],
    M: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    dof_body: InlineArray[Int, _ensure_positive[NV]()],
    I_world: InlineArray[Scalar[DTYPE], _ensure_positive[NBODY * 6]()],
    subtree_mask: InlineArray[Bool, _ensure_positive[NBODY * NBODY]()],
):
    """One CRBA row i (M[i,j] for j>=i + symmetric writes). Extracted
    verbatim from the `_mass_matrix_env_fields` row loop so serial and _mt
    share identical arithmetic."""
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
    compute_mass_matrix_full_gpu; setup/row bodies now live in the shared
    `_mm_setup_env_fields` / `_mm_row_env_fields` helpers — pure refactor,
    gated bit-exact by tests/physics3d/test_fk_fields.mojo)."""
    for i in range(NV * NV):
        M[env, i] = 0

    comptime NV_SAFE = _ensure_positive[NV]()
    comptime I_WORLD_SIZE = _ensure_positive[NBODY * 6]()
    comptime MASK_SIZE = _ensure_positive[NBODY * NBODY]()
    var dof_body = InlineArray[Int, NV_SAFE](uninitialized=True)
    var I_world = InlineArray[Scalar[DTYPE], I_WORLD_SIZE](uninitialized=True)
    var subtree_mask = InlineArray[Bool, MASK_SIZE](fill=False)
    _mm_setup_env_fields[DTYPE, NV, NBODY, NJOINT, BATCH](
        env, xquat, bodies, joints, dof_body, I_world, subtree_mask
    )

    # M[i,j] via direct body summation with subtree mask lookup
    for i in range(NV):
        _mm_row_env_fields[DTYPE, NV, NBODY, BATCH](
            env, i, xipos, subtree_com, bodies, cdof, M,
            dof_body, I_world, subtree_mask,
        )


# ── Cooperative (_mt) kernel — schedule from the legacy
# `compute_mass_matrix_full_gpu_mt` (dynamics/mass_matrix.mojo): every
# thread redundantly computes the setup (model/FK-state reads only), then
# rows are striped i = tid, tid+n, ... Per-row arithmetic is the SAME
# `_mm_row_env_fields` helper as the serial kernel -> bit-exact. One
# barrier after the distributed zero-init (the legacy variant relied on
# caller-side barriers; here the kernel is standalone). Grid is exact ->
# no valid_env guards. NOTE: this ports the DENSE full-CRBA schedule, not
# `compute_mass_matrix_treewalk_gpu_mt` — the treewalk is a different
# algorithm (composite-inertia ancestor walk), only float-tolerance-equal
# to the dense serial kernel, so it cannot satisfy the bit-exact gate.
# The treewalk lives in `_mass_matrix_treewalk_fields_mt_kernel` below,
# behind the dispatcher's TREEWALK=True (gated bit-exact vs the LEGACY
# treewalk instead).
def _mass_matrix_fields_mt_kernel[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
    N_THREADS: Int,
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
    var env = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    # Zero M — distributed across threads, published before the row sweep
    # (every entry is overwritten by the sweep; the barrier just orders the
    # dead zero-stores before the live writes).
    for i in range(tid, NV * NV, N_THREADS):
        M[env, i] = 0
    barrier()

    # Setup redundantly per thread (identical values in every thread).
    comptime NV_SAFE = _ensure_positive[NV]()
    comptime I_WORLD_SIZE = _ensure_positive[NBODY * 6]()
    comptime MASK_SIZE = _ensure_positive[NBODY * NBODY]()
    var dof_body = InlineArray[Int, NV_SAFE](uninitialized=True)
    var I_world = InlineArray[Scalar[DTYPE], I_WORLD_SIZE](uninitialized=True)
    var subtree_mask = InlineArray[Bool, MASK_SIZE](fill=False)
    _mm_setup_env_fields[DTYPE, NV, NBODY, NJOINT, BATCH](
        env, xquat, bodies, joints, dof_body, I_world, subtree_mask
    )

    # Each thread handles rows i where i % N_THREADS == tid.
    for i in range(tid, NV, N_THREADS):
        _mm_row_env_fields[DTYPE, NV, NBODY, BATCH](
            env, i, xipos, subtree_com, bodies, cdof, M,
            dof_body, I_world, subtree_mask,
        )


# ── Tree-walk CRBA (cooperative) — legacy PRODUCTION mass matrix ─────────
# Verbatim port of `compute_mass_matrix_treewalk_gpu_mt`
# (dynamics/mass_matrix.mojo:2017): per-DOF row, walk ancestor DOFs, using
# a composite spatial inertia about P = subtree_com[rootid] (O(NV·depth)
# vs the dense O(NV²·NBODY)). Same launch shape as the dense _mt kernel:
# grid=(BATCH,), block=(N_THREADS,) — exact grid, so the legacy valid_env
# guards collapse to unconditional; barrier placement is identical
# (distributed zero -> barrier -> striped row loop -> barrier). Setup
# (dof maps + composite) is per-thread redundant, exactly like legacy.
#
# NOTE on the legacy composite-inertia skip: when the legacy integrator
# selects the treewalk (USE_TREEWALK_MM, rk4_integrator.mojo:1536-1541) it
# SKIPS `compute_composite_inertia_gpu` (legacy step 5) — the treewalk
# builds its own composite in registers and never reads the `crb` slot
# (RNE later overwrites it with cvel). The fields path has NO standalone
# composite-inertia pass at all (the dense fields kernel also builds its
# per-body I_world in registers), so there is nothing to skip here — this
# kernel simply does not do that dead work, mirroring legacy production.
def _mass_matrix_treewalk_fields_mt_kernel[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
    N_THREADS: Int,
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
    var env = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    comptime NV_S = _ensure_positive[NV]()
    comptime NB_S = _ensure_positive[NBODY]()
    comptime CMP_S = _ensure_positive[NBODY * 10]()
    var dof_body = InlineArray[Int, NV_S](fill=0)
    var dof_parent = InlineArray[Int, NV_S](fill=-1)
    var body_first = InlineArray[Int, NB_S](fill=-1)
    var body_last = InlineArray[Int, NB_S](fill=-1)
    var comp = InlineArray[Scalar[DTYPE], CMP_S](fill=0)

    # --- dof_body + per-body dof range ---
    for j in range(NJOINT):
        var jb = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID]))
        var dadr = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR]))
        var jt = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var ndof = 1
        if jt == JNT_FREE:
            ndof = 6
        elif jt == JNT_BALL:
            ndof = 3
        for d in range(ndof):
            dof_body[dadr + d] = jb
        if body_first[jb] < 0 or dadr < body_first[jb]:
            body_first[jb] = dadr
        if dadr + ndof - 1 > body_last[jb]:
            body_last[jb] = dadr + ndof - 1

    # --- dof_parent: within body = d-1; at body's first dof = last dof of the
    #     nearest ancestor body that has DOFs (else -1) ---
    for d in range(NV):
        var b = dof_body[d]
        if d > body_first[b]:
            dof_parent[d] = d - 1
        else:
            var p = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_PARENT]))
            while p > 0:
                if body_last[p] >= 0:
                    dof_parent[d] = body_last[p]
                    break
                p = Int(rebind[Scalar[DTYPE]](bodies[p, BODY_IDX_PARENT]))

    # --- per-body composite contribution about P = stcom[rootid] ---
    for b in range(NBODY):
        var mass = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_MASS])
        # rotated body inertia (world-aligned, about body COM)
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
        var Iw_xx = Ixx_l * r00 * r00 + Iyy_l * r01 * r01 + Izz_l * r02 * r02
        var Iw_yy = Ixx_l * r10 * r10 + Iyy_l * r11 * r11 + Izz_l * r12 * r12
        var Iw_zz = Ixx_l * r20 * r20 + Iyy_l * r21 * r21 + Izz_l * r22 * r22
        var Iw_xy = Ixx_l * r00 * r10 + Iyy_l * r01 * r11 + Izz_l * r02 * r12
        var Iw_xz = Ixx_l * r00 * r20 + Iyy_l * r01 * r21 + Izz_l * r02 * r22
        var Iw_yz = Ixx_l * r10 * r20 + Iyy_l * r11 * r21 + Izz_l * r12 * r22
        # d = xipos[b] - stcom[rootid[b]]
        var rootb = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_ROOTID]))
        var dx = rebind[Scalar[DTYPE]](
            xipos[env, b * 3 + 0]
        ) - rebind[Scalar[DTYPE]](subtree_com[env, rootb * 3 + 0])
        var dy = rebind[Scalar[DTYPE]](
            xipos[env, b * 3 + 1]
        ) - rebind[Scalar[DTYPE]](subtree_com[env, rootb * 3 + 1])
        var dz = rebind[Scalar[DTYPE]](
            xipos[env, b * 3 + 2]
        ) - rebind[Scalar[DTYPE]](subtree_com[env, rootb * 3 + 2])
        var dd = dx * dx + dy * dy + dz * dz
        comp[b * 10 + 0] = mass
        comp[b * 10 + 1] = mass * dx
        comp[b * 10 + 2] = mass * dy
        comp[b * 10 + 3] = mass * dz
        comp[b * 10 + 4] = Iw_xx + mass * (dd - dx * dx)
        comp[b * 10 + 5] = Iw_yy + mass * (dd - dy * dy)
        comp[b * 10 + 6] = Iw_zz + mass * (dd - dz * dz)
        comp[b * 10 + 7] = Iw_xy - mass * dx * dy
        comp[b * 10 + 8] = Iw_xz - mass * dx * dz
        comp[b * 10 + 9] = Iw_yz - mass * dy * dz

    # leaf→root accumulate (common P within a tree → additive; stop at roots)
    for b in range(NBODY - 1, 0, -1):
        var p = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_PARENT]))
        if p > 0:
            for e in range(10):
                comp[p * 10 + e] = comp[p * 10 + e] + comp[b * 10 + e]

    # zero M (distributed)
    for idx in range(tid, NV * NV, N_THREADS):
        M[env, idx] = Scalar[DTYPE](0)
    barrier()

    # per-DOF row, distributed: f_i = comp[body_i]·cdof_i, walk ancestor DOFs
    for i in range(tid, NV, N_THREADS):
        var bi = dof_body[i]
        var ai0 = cdof[env, i * 6 + 0]
        var ai1 = cdof[env, i * 6 + 1]
        var ai2 = cdof[env, i * 6 + 2]
        var li0 = cdof[env, i * 6 + 3]
        var li1 = cdof[env, i * 6 + 4]
        var li2 = cdof[env, i * 6 + 5]
        var Mc = comp[bi * 10 + 0]
        var hx = comp[bi * 10 + 1]
        var hy = comp[bi * 10 + 2]
        var hz = comp[bi * 10 + 3]
        var Cxx = comp[bi * 10 + 4]
        var Cyy = comp[bi * 10 + 5]
        var Czz = comp[bi * 10 + 6]
        var Cxy = comp[bi * 10 + 7]
        var Cxz = comp[bi * 10 + 8]
        var Cyz = comp[bi * 10 + 9]
        # f_ang = Ic_rot·a_i + hc×l_i
        var fa0 = Cxx * ai0 + Cxy * ai1 + Cxz * ai2 + (hy * li2 - hz * li1)
        var fa1 = Cxy * ai0 + Cyy * ai1 + Cyz * ai2 + (hz * li0 - hx * li2)
        var fa2 = Cxz * ai0 + Cyz * ai1 + Czz * ai2 + (hx * li1 - hy * li0)
        # f_lin = Mc·l_i + a_i×hc
        var fl0 = Mc * li0 + (ai1 * hz - ai2 * hy)
        var fl1 = Mc * li1 + (ai2 * hx - ai0 * hz)
        var fl2 = Mc * li2 + (ai0 * hy - ai1 * hx)
        var j = i
        while j >= 0:
            var aj0 = cdof[env, j * 6 + 0]
            var aj1 = cdof[env, j * 6 + 1]
            var aj2 = cdof[env, j * 6 + 2]
            var lj0 = cdof[env, j * 6 + 3]
            var lj1 = cdof[env, j * 6 + 4]
            var lj2 = cdof[env, j * 6 + 5]
            var mij = (
                aj0 * fa0 + aj1 * fa1 + aj2 * fa2
                + lj0 * fl0 + lj1 * fl1 + lj2 * fl2
            )
            M[env, i * NV + j] = mij
            if i != j:
                M[env, j * NV + i] = mij
            j = dof_parent[j]
    barrier()


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
    PARALLEL: Bool = False,
    TREEWALK: Bool = False,
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
    Reads `scratch.cdof`, writes `scratch.M`. PARALLEL=True (GPU only):
    cooperative row-striped kernel, bit-exact vs serial. CPU ignores
    PARALLEL. TREEWALK=True (requires PARALLEL): the cooperative TREE-WALK
    CRBA — the legacy PRODUCTION GPU algorithm (O(NV·depth), verbatim port
    of `compute_mass_matrix_treewalk_gpu_mt`) — float-tolerance-equal to
    the dense kernels, NOT bit-exact vs them. CPU ignores TREEWALK too
    (mirrors legacy production: CPU dense, GPU treewalk)."""
    comptime assert not (TREEWALK and not PARALLEL), (
        "compute_mass_matrix_fields: TREEWALK requires PARALLEL (the"
        " treewalk is inherently cooperative; CPU and serial GPU stay dense)"
    )
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
    elif PARALLEL and TREEWALK:
        var c = ctx.value()
        comptime MT_T = NV
        c.enqueue_function[
            _mass_matrix_treewalk_fields_mt_kernel[
                DTYPE, NV, NBODY, NJOINT, BATCH, MT_T
            ]
        ](
            d.xquat.lt["gpu", L_B4](),
            d.xipos.lt["gpu", L_B3](),
            d.subtree_com.lt["gpu", L_B3](),
            m.bodies.lt["gpu", L_BODY](),
            m.joints.lt["gpu", L_JOINT](),
            scratch.cdof.lt["gpu", L_CDOF](),
            scratch.M.lt["gpu", L_M](),
            grid_dim=(BATCH,),
            block_dim=(MT_T,),
        )
    elif PARALLEL:
        var c = ctx.value()
        comptime MT_T = NV
        c.enqueue_function[
            _mass_matrix_fields_mt_kernel[
                DTYPE, NV, NBODY, NJOINT, BATCH, MT_T
            ]
        ](
            d.xquat.lt["gpu", L_B4](),
            d.xipos.lt["gpu", L_B3](),
            d.subtree_com.lt["gpu", L_B3](),
            m.bodies.lt["gpu", L_BODY](),
            m.joints.lt["gpu", L_JOINT](),
            scratch.cdof.lt["gpu", L_CDOF](),
            scratch.M.lt["gpu", L_M](),
            grid_dim=(BATCH,),
            block_dim=(MT_T,),
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

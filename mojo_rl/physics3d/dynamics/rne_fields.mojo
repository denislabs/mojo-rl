"""RNE bias forces over per-field tensors (migration P2, single-source).

Per-field port of `compute_bias_forces_rne_gpu` + `rne_fwd_body`
(dynamics/bias_forces.mojo) — arithmetic verbatim. Computes
b(q, qvel) = C(q, qvel)·qvel + g(q) into `scratch.bias`.

Operands (12): qvel, xquat, xipos, subtree_com (data) + bodies, joints,
meta (model; gravity lives in the meta record) + cdof, crb, rne_cacc,
rne_cfrc, bias (scratch). As in the legacy code, the `crb` scratch tensor
doubles as per-body cvel storage during RNE (b*6 indexing within the
NBODY*10 tensor); `cinert` stays a per-thread InlineArray."""

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from ..kinematics.quat_math import gpu_quat_mul
from ..joint_types import JNT_FREE, JNT_BALL
from ..fields import DataFields, ModelFields, DynamicsScratch
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_META_IDX_GRAVITY_X,
    MODEL_META_IDX_GRAVITY_Y,
    MODEL_META_IDX_GRAVITY_Z,
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

comptime RNE_TPB: Int = 64


@always_inline
def _max_one[N: Int]() -> Int:
    return N if N > 0 else 1


@always_inline
def _rne_fwd_body_fields[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
](
    env: Int,
    b: Int,
    gx: Scalar[DTYPE],
    gy: Scalar[DTYPE],
    gz: Scalar[DTYPE],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    cdof: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * 6), MutAnyOrigin],
    cvel: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 10), MutAnyOrigin
    ],
    cacc: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 6), MutAnyOrigin
    ],
):
    """Forward-pass cvel/cacc for one body (verbatim from rne_fwd_body;
    `cvel` is the crb scratch tensor, b*6 indexing)."""
    var parent = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_PARENT]))

    var cv_wx = rebind[Scalar[DTYPE]](cvel[env, parent * 6 + 0])
    var cv_wy = rebind[Scalar[DTYPE]](cvel[env, parent * 6 + 1])
    var cv_wz = rebind[Scalar[DTYPE]](cvel[env, parent * 6 + 2])
    var cv_vx = rebind[Scalar[DTYPE]](cvel[env, parent * 6 + 3])
    var cv_vy = rebind[Scalar[DTYPE]](cvel[env, parent * 6 + 4])
    var cv_vz = rebind[Scalar[DTYPE]](cvel[env, parent * 6 + 5])

    if parent == 0:
        cacc[env, b * 6 + 0] = Scalar[DTYPE](0)
        cacc[env, b * 6 + 1] = Scalar[DTYPE](0)
        cacc[env, b * 6 + 2] = Scalar[DTYPE](0)
        cacc[env, b * 6 + 3] = -gx
        cacc[env, b * 6 + 4] = -gy
        cacc[env, b * 6 + 5] = -gz
    else:
        for k in range(6):
            cacc[env, b * 6 + k] = cacc[env, parent * 6 + k]

    for j in range(NJOINT):
        var jnt_body = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID])
        )
        if jnt_body != b:
            continue

        var jnt_type = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var dof_adr = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR])
        )

        if jnt_type == JNT_FREE:
            # Translation DOFs: cdof_dot = 0, just update cvel
            for d in range(3):
                var dof = dof_adr + d
                var qdot = rebind[Scalar[DTYPE]](qvel[env, dof])
                cv_wx = cv_wx + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 0]
                ) * qdot
                cv_wy = cv_wy + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 1]
                ) * qdot
                cv_wz = cv_wz + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 2]
                ) * qdot
                cv_vx = cv_vx + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 3]
                ) * qdot
                cv_vy = cv_vy + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 4]
                ) * qdot
                cv_vz = cv_vz + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 5]
                ) * qdot

            # Rotation DOFs: all 3 cdof_dots use pre-rotation cvel
            for d in range(3, 6):
                var dof = dof_adr + d
                var qdot = rebind[Scalar[DTYPE]](qvel[env, dof])
                var s_ang_x = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 0])
                var s_ang_y = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 1])
                var s_ang_z = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 2])
                var s_lin_x = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 3])
                var s_lin_y = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 4])
                var s_lin_z = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 5])

                var cdot_ang_x = cv_wy * s_ang_z - cv_wz * s_ang_y
                var cdot_ang_y = cv_wz * s_ang_x - cv_wx * s_ang_z
                var cdot_ang_z = cv_wx * s_ang_y - cv_wy * s_ang_x
                var cdot_lin_x = (cv_wy * s_lin_z - cv_wz * s_lin_y) + (
                    cv_vy * s_ang_z - cv_vz * s_ang_y
                )
                var cdot_lin_y = (cv_wz * s_lin_x - cv_wx * s_lin_z) + (
                    cv_vz * s_ang_x - cv_vx * s_ang_z
                )
                var cdot_lin_z = (cv_wx * s_lin_y - cv_wy * s_lin_x) + (
                    cv_vx * s_ang_y - cv_vy * s_ang_x
                )

                cacc[env, b * 6 + 0] = cacc[env, b * 6 + 0] + cdot_ang_x * qdot
                cacc[env, b * 6 + 1] = cacc[env, b * 6 + 1] + cdot_ang_y * qdot
                cacc[env, b * 6 + 2] = cacc[env, b * 6 + 2] + cdot_ang_z * qdot
                cacc[env, b * 6 + 3] = cacc[env, b * 6 + 3] + cdot_lin_x * qdot
                cacc[env, b * 6 + 4] = cacc[env, b * 6 + 4] + cdot_lin_y * qdot
                cacc[env, b * 6 + 5] = cacc[env, b * 6 + 5] + cdot_lin_z * qdot

            # Update cvel AFTER all 3 rotational cdof_dots
            for d in range(3, 6):
                var dof = dof_adr + d
                var qdot = rebind[Scalar[DTYPE]](qvel[env, dof])
                cv_wx = cv_wx + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 0]
                ) * qdot
                cv_wy = cv_wy + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 1]
                ) * qdot
                cv_wz = cv_wz + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 2]
                ) * qdot
                cv_vx = cv_vx + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 3]
                ) * qdot
                cv_vy = cv_vy + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 4]
                ) * qdot
                cv_vz = cv_vz + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 5]
                ) * qdot

        elif jnt_type == JNT_BALL:
            for d in range(3):
                var dof = dof_adr + d
                var qdot = rebind[Scalar[DTYPE]](qvel[env, dof])
                var s_ang_x = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 0])
                var s_ang_y = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 1])
                var s_ang_z = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 2])
                var s_lin_x = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 3])
                var s_lin_y = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 4])
                var s_lin_z = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 5])

                var cdot_ang_x = cv_wy * s_ang_z - cv_wz * s_ang_y
                var cdot_ang_y = cv_wz * s_ang_x - cv_wx * s_ang_z
                var cdot_ang_z = cv_wx * s_ang_y - cv_wy * s_ang_x
                var cdot_lin_x = (cv_wy * s_lin_z - cv_wz * s_lin_y) + (
                    cv_vy * s_ang_z - cv_vz * s_ang_y
                )
                var cdot_lin_y = (cv_wz * s_lin_x - cv_wx * s_lin_z) + (
                    cv_vz * s_ang_x - cv_vx * s_ang_z
                )
                var cdot_lin_z = (cv_wx * s_lin_y - cv_wy * s_lin_x) + (
                    cv_vx * s_ang_y - cv_vy * s_ang_x
                )

                cacc[env, b * 6 + 0] = cacc[env, b * 6 + 0] + cdot_ang_x * qdot
                cacc[env, b * 6 + 1] = cacc[env, b * 6 + 1] + cdot_ang_y * qdot
                cacc[env, b * 6 + 2] = cacc[env, b * 6 + 2] + cdot_ang_z * qdot
                cacc[env, b * 6 + 3] = cacc[env, b * 6 + 3] + cdot_lin_x * qdot
                cacc[env, b * 6 + 4] = cacc[env, b * 6 + 4] + cdot_lin_y * qdot
                cacc[env, b * 6 + 5] = cacc[env, b * 6 + 5] + cdot_lin_z * qdot

            for d in range(3):
                var dof = dof_adr + d
                var qdot = rebind[Scalar[DTYPE]](qvel[env, dof])
                cv_wx = cv_wx + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 0]
                ) * qdot
                cv_wy = cv_wy + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 1]
                ) * qdot
                cv_wz = cv_wz + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 2]
                ) * qdot
                cv_vx = cv_vx + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 3]
                ) * qdot
                cv_vy = cv_vy + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 4]
                ) * qdot
                cv_vz = cv_vz + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 5]
                ) * qdot

        else:
            # HINGE or SLIDE (1 DOF)
            var dof = dof_adr
            var qdot = rebind[Scalar[DTYPE]](qvel[env, dof])
            var s_ang_x = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 0])
            var s_ang_y = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 1])
            var s_ang_z = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 2])
            var s_lin_x = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 3])
            var s_lin_y = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 4])
            var s_lin_z = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 5])

            var cdot_ang_x = cv_wy * s_ang_z - cv_wz * s_ang_y
            var cdot_ang_y = cv_wz * s_ang_x - cv_wx * s_ang_z
            var cdot_ang_z = cv_wx * s_ang_y - cv_wy * s_ang_x
            var cdot_lin_x = (cv_wy * s_lin_z - cv_wz * s_lin_y) + (
                cv_vy * s_ang_z - cv_vz * s_ang_y
            )
            var cdot_lin_y = (cv_wz * s_lin_x - cv_wx * s_lin_z) + (
                cv_vz * s_ang_x - cv_vx * s_ang_z
            )
            var cdot_lin_z = (cv_wx * s_lin_y - cv_wy * s_lin_x) + (
                cv_vx * s_ang_y - cv_vy * s_ang_x
            )

            cacc[env, b * 6 + 0] = cacc[env, b * 6 + 0] + cdot_ang_x * qdot
            cacc[env, b * 6 + 1] = cacc[env, b * 6 + 1] + cdot_ang_y * qdot
            cacc[env, b * 6 + 2] = cacc[env, b * 6 + 2] + cdot_ang_z * qdot
            cacc[env, b * 6 + 3] = cacc[env, b * 6 + 3] + cdot_lin_x * qdot
            cacc[env, b * 6 + 4] = cacc[env, b * 6 + 4] + cdot_lin_y * qdot
            cacc[env, b * 6 + 5] = cacc[env, b * 6 + 5] + cdot_lin_z * qdot

            cv_wx = cv_wx + s_ang_x * qdot
            cv_wy = cv_wy + s_ang_y * qdot
            cv_wz = cv_wz + s_ang_z * qdot
            cv_vx = cv_vx + s_lin_x * qdot
            cv_vy = cv_vy + s_lin_y * qdot
            cv_vz = cv_vz + s_lin_z * qdot

    cvel[env, b * 6 + 0] = cv_wx
    cvel[env, b * 6 + 1] = cv_wy
    cvel[env, b * 6 + 2] = cv_wz
    cvel[env, b * 6 + 3] = cv_vx
    cvel[env, b * 6 + 4] = cv_vy
    cvel[env, b * 6 + 5] = cv_vz


@always_inline
def _rne_env_fields[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
](
    env: Int,
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
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
    meta: LayoutTensor[DTYPE, Layout.row_major(MODEL_META_SIZE), MutAnyOrigin],
    cdof: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * 6), MutAnyOrigin],
    crb: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 10), MutAnyOrigin
    ],
    rne_cacc: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 6), MutAnyOrigin
    ],
    rne_cfrc: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 6), MutAnyOrigin
    ],
    bias: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
):
    """Full RNE for one env (verbatim from compute_bias_forces_rne_gpu)."""
    for i in range(NV):
        bias[env, i] = 0

    var gx = rebind[Scalar[DTYPE]](meta[MODEL_META_IDX_GRAVITY_X])
    var gy = rebind[Scalar[DTYPE]](meta[MODEL_META_IDX_GRAVITY_Y])
    var gz = rebind[Scalar[DTYPE]](meta[MODEL_META_IDX_GRAVITY_Z])

    comptime BODY6_SIZE = _max_one[NBODY * 6]()
    for i in range(BODY6_SIZE):
        rne_cacc[env, i] = Scalar[DTYPE](0)
    for i in range(BODY6_SIZE):
        rne_cfrc[env, i] = Scalar[DTYPE](0)
    comptime CINERT_GPU_SIZE = _max_one[NBODY * 10]()
    var cinert_g = InlineArray[Scalar[DTYPE], CINERT_GPU_SIZE](
        uninitialized=True
    )
    for i in range(CINERT_GPU_SIZE):
        cinert_g[i] = Scalar[DTYPE](0)

    # Step 0: cinert — spatial inertia at subtree_com (mj_inertCom)
    for b in range(NBODY):
        var Ixx_local = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IXX])
        var Iyy_local = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IYY])
        var Izz_local = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IZZ])

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

        var mass_b = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_MASS])
        cinert_g[b * 10 + 0] = (
            Ixx_local * r00 * r00 + Iyy_local * r01 * r01 + Izz_local * r02 * r02
        )
        cinert_g[b * 10 + 1] = (
            Ixx_local * r10 * r10 + Iyy_local * r11 * r11 + Izz_local * r12 * r12
        )
        cinert_g[b * 10 + 2] = (
            Ixx_local * r20 * r20 + Iyy_local * r21 * r21 + Izz_local * r22 * r22
        )
        cinert_g[b * 10 + 3] = (
            Ixx_local * r00 * r10 + Iyy_local * r01 * r11 + Izz_local * r02 * r12
        )
        cinert_g[b * 10 + 4] = (
            Ixx_local * r00 * r20 + Iyy_local * r01 * r21 + Izz_local * r02 * r22
        )
        cinert_g[b * 10 + 5] = (
            Ixx_local * r10 * r20 + Iyy_local * r11 * r21 + Izz_local * r12 * r22
        )
        var rootid_b = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_ROOTID]))
        var dx_b = rebind[Scalar[DTYPE]](
            xipos[env, b * 3 + 0]
        ) - rebind[Scalar[DTYPE]](subtree_com[env, rootid_b * 3 + 0])
        var dy_b = rebind[Scalar[DTYPE]](
            xipos[env, b * 3 + 1]
        ) - rebind[Scalar[DTYPE]](subtree_com[env, rootid_b * 3 + 1])
        var dz_b = rebind[Scalar[DTYPE]](
            xipos[env, b * 3 + 2]
        ) - rebind[Scalar[DTYPE]](subtree_com[env, rootid_b * 3 + 2])
        cinert_g[b * 10 + 0] = cinert_g[b * 10 + 0] + mass_b * (
            dy_b * dy_b + dz_b * dz_b
        )
        cinert_g[b * 10 + 1] = cinert_g[b * 10 + 1] + mass_b * (
            dx_b * dx_b + dz_b * dz_b
        )
        cinert_g[b * 10 + 2] = cinert_g[b * 10 + 2] + mass_b * (
            dx_b * dx_b + dy_b * dy_b
        )
        cinert_g[b * 10 + 3] = cinert_g[b * 10 + 3] - mass_b * dx_b * dy_b
        cinert_g[b * 10 + 4] = cinert_g[b * 10 + 4] - mass_b * dx_b * dz_b
        cinert_g[b * 10 + 5] = cinert_g[b * 10 + 5] - mass_b * dy_b * dz_b
        cinert_g[b * 10 + 6] = mass_b * dx_b
        cinert_g[b * 10 + 7] = mass_b * dy_b
        cinert_g[b * 10 + 8] = mass_b * dz_b
        cinert_g[b * 10 + 9] = mass_b

    # Per-body spatial velocity stored in the crb tensor (b*6 indexing)
    for i in range(NBODY * 6):
        crb[env, i] = 0

    # Step 1: Forward pass — cvel and cacc (root to leaves)
    for b in range(1, NBODY):
        _rne_fwd_body_fields[DTYPE, NV, NBODY, NJOINT, BATCH](
            env, b, gx, gy, gz, qvel, bodies, joints, cdof, crb, rne_cacc
        )

    # Step 2: Spatial forces per body: cfrc = I*cacc + cvel x* (I*cvel)
    for b in range(NBODY):
        var wx = rebind[Scalar[DTYPE]](crb[env, b * 6 + 0])
        var wy = rebind[Scalar[DTYPE]](crb[env, b * 6 + 1])
        var wz = rebind[Scalar[DTYPE]](crb[env, b * 6 + 2])
        var vx = rebind[Scalar[DTYPE]](crb[env, b * 6 + 3])
        var vy = rebind[Scalar[DTYPE]](crb[env, b * 6 + 4])
        var vz = rebind[Scalar[DTYPE]](crb[env, b * 6 + 5])

        var ci0 = cinert_g[b * 10 + 0]
        var ci1 = cinert_g[b * 10 + 1]
        var ci2 = cinert_g[b * 10 + 2]
        var ci3 = cinert_g[b * 10 + 3]
        var ci4 = cinert_g[b * 10 + 4]
        var ci5 = cinert_g[b * 10 + 5]
        var ci6 = cinert_g[b * 10 + 6]
        var ci7 = cinert_g[b * 10 + 7]
        var ci8 = cinert_g[b * 10 + 8]
        var ci9 = cinert_g[b * 10 + 9]
        var ax = rne_cacc[env, b * 6 + 0]
        var ay = rne_cacc[env, b * 6 + 1]
        var az = rne_cacc[env, b * 6 + 2]
        var alx = rne_cacc[env, b * 6 + 3]
        var aly = rne_cacc[env, b * 6 + 4]
        var alz = rne_cacc[env, b * 6 + 5]

        var Ia0 = ci0 * ax + ci3 * ay + ci4 * az - ci8 * aly + ci7 * alz
        var Ia1 = ci3 * ax + ci1 * ay + ci5 * az + ci8 * alx - ci6 * alz
        var Ia2 = ci4 * ax + ci5 * ay + ci2 * az - ci7 * alx + ci6 * aly
        var Ia3 = ci8 * ay - ci7 * az + ci9 * alx
        var Ia4 = ci6 * az - ci8 * ax + ci9 * aly
        var Ia5 = ci7 * ax - ci6 * ay + ci9 * alz

        var Iv0 = ci0 * wx + ci3 * wy + ci4 * wz - ci8 * vy + ci7 * vz
        var Iv1 = ci3 * wx + ci1 * wy + ci5 * wz + ci8 * vx - ci6 * vz
        var Iv2 = ci4 * wx + ci5 * wy + ci2 * wz - ci7 * vx + ci6 * vy
        var Iv3 = ci8 * wy - ci7 * wz + ci9 * vx
        var Iv4 = ci6 * wz - ci8 * wx + ci9 * vy
        var Iv5 = ci7 * wx - ci6 * wy + ci9 * vz

        var xf0 = wy * Iv2 - wz * Iv1 + vy * Iv5 - vz * Iv4
        var xf1 = wz * Iv0 - wx * Iv2 + vz * Iv3 - vx * Iv5
        var xf2 = wx * Iv1 - wy * Iv0 + vx * Iv4 - vy * Iv3
        var xf3 = wy * Iv5 - wz * Iv4
        var xf4 = wz * Iv3 - wx * Iv5
        var xf5 = wx * Iv4 - wy * Iv3

        rne_cfrc[env, b * 6 + 0] = Ia0 + xf0
        rne_cfrc[env, b * 6 + 1] = Ia1 + xf1
        rne_cfrc[env, b * 6 + 2] = Ia2 + xf2
        rne_cfrc[env, b * 6 + 3] = Ia3 + xf3
        rne_cfrc[env, b * 6 + 4] = Ia4 + xf4
        rne_cfrc[env, b * 6 + 5] = Ia5 + xf5

    # Step 3: Backward pass — simple addition
    for b in range(NBODY - 1, 0, -1):
        var parent = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_PARENT]))
        if parent > 0:
            for k in range(6):
                rne_cfrc[env, parent * 6 + k] = (
                    rne_cfrc[env, parent * 6 + k] + rne_cfrc[env, b * 6 + k]
                )

    # Step 4: Project to joint space: bias[d] = cdof[d] . cfrc[body_of_dof]
    for j in range(NJOINT):
        var jnt_type = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var body = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID]))
        var dof_adr = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR])
        )
        var num_dof = 1
        if jnt_type == JNT_FREE:
            num_dof = 6
        elif jnt_type == JNT_BALL:
            num_dof = 3

        for d in range(num_dof):
            var dof = dof_adr + d
            bias[env, dof] = 0
            for k in range(6):
                bias[env, dof] = (
                    bias[env, dof]
                    + cdof[env, dof * 6 + k] * rne_cfrc[env, body * 6 + k]
                )


def _rne_fields_kernel[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
](
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
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
    meta: LayoutTensor[DTYPE, Layout.row_major(MODEL_META_SIZE), MutAnyOrigin],
    cdof: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * 6), MutAnyOrigin],
    crb: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 10), MutAnyOrigin
    ],
    rne_cacc: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 6), MutAnyOrigin
    ],
    rne_cfrc: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 6), MutAnyOrigin
    ],
    bias: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _rne_env_fields[DTYPE, NV, NBODY, NJOINT, BATCH](
        env, qvel, xquat, xipos, subtree_com, bodies, joints, meta,
        cdof, crb, rne_cacc, rne_cfrc, bias,
    )


def compute_bias_forces_rne_fields[
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
    """RNE bias forces, both targets, one body. Reads FK products + qvel +
    `scratch.cdof`; writes `scratch.bias` (+ crb/rne_cacc/rne_cfrc temps)."""
    comptime L_NV = Layout.row_major(BATCH, NV)
    comptime L_B3 = Layout.row_major(BATCH, NBODY * 3)
    comptime L_B4 = Layout.row_major(BATCH, NBODY * 4)
    comptime L_BODY = Layout.row_major(NBODY, MODEL_BODY_SIZE)
    comptime L_JOINT = Layout.row_major(NJOINT, MODEL_JOINT_SIZE)
    comptime L_META = Layout.row_major(MODEL_META_SIZE)
    comptime L_CDOF = Layout.row_major(BATCH, NV * 6)
    comptime L_CRB = Layout.row_major(BATCH, NBODY * 10)
    comptime L_B6 = Layout.row_major(BATCH, NBODY * 6)

    comptime if target == "cpu":
        var qvel_v = d.qvel.lt["cpu", L_NV]()
        var xquat_v = d.xquat.lt["cpu", L_B4]()
        var xipos_v = d.xipos.lt["cpu", L_B3]()
        var stcom_v = d.subtree_com.lt["cpu", L_B3]()
        var bodies_v = m.bodies.lt["cpu", L_BODY]()
        var joints_v = m.joints.lt["cpu", L_JOINT]()
        var meta_v = m.meta.lt["cpu", L_META]()
        var cdof_v = scratch.cdof.lt["cpu", L_CDOF]()
        var crb_v = scratch.crb.lt["cpu", L_CRB]()
        var cacc_v = scratch.rne_cacc.lt["cpu", L_B6]()
        var cfrc_v = scratch.rne_cfrc.lt["cpu", L_B6]()
        var bias_v = scratch.bias.lt["cpu", L_NV]()
        for e in range(BATCH):
            _rne_env_fields[DTYPE, NV, NBODY, NJOINT, BATCH](
                e, qvel_v, xquat_v, xipos_v, stcom_v, bodies_v, joints_v,
                meta_v, cdof_v, crb_v, cacc_v, cfrc_v, bias_v,
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + RNE_TPB - 1) // RNE_TPB
        c.enqueue_function[
            _rne_fields_kernel[DTYPE, NV, NBODY, NJOINT, BATCH]
        ](
            d.qvel.lt["gpu", L_NV](),
            d.xquat.lt["gpu", L_B4](),
            d.xipos.lt["gpu", L_B3](),
            d.subtree_com.lt["gpu", L_B3](),
            m.bodies.lt["gpu", L_BODY](),
            m.joints.lt["gpu", L_JOINT](),
            m.meta.lt["gpu", L_META](),
            scratch.cdof.lt["gpu", L_CDOF](),
            scratch.crb.lt["gpu", L_CRB](),
            scratch.rne_cacc.lt["gpu", L_B6](),
            scratch.rne_cfrc.lt["gpu", L_B6](),
            scratch.bias.lt["gpu", L_NV](),
            grid_dim=(BLOCKS,),
            block_dim=(RNE_TPB,),
        )

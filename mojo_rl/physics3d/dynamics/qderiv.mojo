"""RNE velocity derivative over per-field tensors (migration P2 / Stage-I,
single-source). Per-field port of `compute_rne_vel_derivative`
(dynamics/velocity_derivatives.mojo) — the dense NV×NV
`qDeriv = d(qfrc_bias)/d(qvel)` (Coriolis/centrifugal velocity sensitivity)
that the fields `ImplicitIntegrator` SUBTRACTS to form the
non-symmetric `M_hat = M + armature - dt*qDeriv`.

Arithmetic is VERBATIM from the legacy CPU function (subtree-COM convention,
matching MuJoCo `mjd_rne_vel_dense`); only the accessors change: model/data
reads come from the packed `Model`/`Data` tensors, `cdof` from
`DynamicsScratch`, and the big cross-body intermediates
(cinert/cdof_sc/cvel_sc/cdof_dot/Dcvel/Dcdofdot/Dcacc/Dcfrcbody) live in the
owned `ImplicitScratch` tensors rather than per-thread InlineArrays — so the
same per-env function serves CPU and GPU without blowing GPU local memory on
wide models (humanoid Dcdofdot = NV*6*NV ≈ 4.4k floats).

The result is SUBTRACTED into `iscratch.qderiv` (caller pre-loads the passive
damping diagonal, exactly like the legacy integrator).
"""

from std.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from ..kinematics.quat_math import quat_mul
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..fields import Data, Model, DynamicsScratch, ImplicitScratch, Dims, DimsLike, AsStatic
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_IDX_NJOINT,
    BODY_IDX_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    BODY_IDX_PARENT,
    BODY_IDX_IQUAT_X,
    BODY_IDX_IQUAT_Y,
    BODY_IDX_IQUAT_Z,
    BODY_IDX_IQUAT_W,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_DOF_ADR,
)

comptime QD_TPB: Int = 32


# =============================================================================
# Spatial derivative helpers — verbatim from velocity_derivatives.mojo
# (tiny fixed InlineArrays, GPU-safe). Self-contained so this module survives
# the legacy sunset.
# =============================================================================


@always_inline
def _mjd_crossMotion_vel[
    DTYPE: DType,
](mut D: InlineArray[Scalar[DTYPE], 36], v: InlineArray[Scalar[DTYPE], 6],):
    """d(crossMotion(vel, v))/d(vel), 6x6 row-major."""
    for i in range(36):
        D[i] = Scalar[DTYPE](0)
    D[0 + 2] = -v[1]
    D[0 + 1] = v[2]
    D[6 + 2] = v[0]
    D[6 + 0] = -v[2]
    D[12 + 1] = -v[0]
    D[12 + 0] = v[1]
    D[18 + 2] = -v[4]
    D[18 + 1] = v[5]
    D[18 + 5] = -v[1]
    D[18 + 4] = v[2]
    D[24 + 2] = v[3]
    D[24 + 0] = -v[5]
    D[24 + 5] = v[0]
    D[24 + 3] = -v[2]
    D[30 + 1] = -v[3]
    D[30 + 0] = v[4]
    D[30 + 4] = -v[0]
    D[30 + 3] = v[1]


@always_inline
def _mjd_crossForce_vel[
    DTYPE: DType,
](mut D: InlineArray[Scalar[DTYPE], 36], f: InlineArray[Scalar[DTYPE], 6],):
    """d(crossForce(vel, f))/d(vel), 6x6 row-major."""
    for i in range(36):
        D[i] = Scalar[DTYPE](0)
    D[0 + 2] = -f[1]
    D[0 + 1] = f[2]
    D[0 + 5] = -f[4]
    D[0 + 4] = f[5]
    D[6 + 2] = f[0]
    D[6 + 0] = -f[2]
    D[6 + 5] = f[3]
    D[6 + 3] = -f[5]
    D[12 + 1] = -f[0]
    D[12 + 0] = f[1]
    D[12 + 4] = -f[3]
    D[12 + 3] = f[4]
    D[18 + 2] = -f[4]
    D[18 + 1] = f[5]
    D[24 + 2] = f[3]
    D[24 + 0] = -f[5]
    D[30 + 1] = -f[3]
    D[30 + 0] = f[4]


@always_inline
def _mjd_crossForce_frc[
    DTYPE: DType,
](mut D: InlineArray[Scalar[DTYPE], 36], vel: InlineArray[Scalar[DTYPE], 6],):
    """d(crossForce(vel, f))/d(f), 6x6 row-major."""
    for i in range(36):
        D[i] = Scalar[DTYPE](0)
    D[0 + 1] = -vel[2]
    D[0 + 2] = vel[1]
    D[0 + 4] = -vel[5]
    D[0 + 5] = vel[4]
    D[6 + 0] = vel[2]
    D[6 + 2] = -vel[0]
    D[6 + 3] = vel[5]
    D[6 + 5] = -vel[3]
    D[12 + 0] = -vel[1]
    D[12 + 1] = vel[0]
    D[12 + 3] = -vel[4]
    D[12 + 4] = vel[3]
    D[18 + 4] = -vel[2]
    D[18 + 5] = vel[1]
    D[24 + 3] = vel[2]
    D[24 + 5] = -vel[0]
    D[30 + 3] = -vel[1]
    D[30 + 4] = vel[0]


@always_inline
def _mjd_mulInertVec_vel[
    DTYPE: DType,
](
    mut D: InlineArray[Scalar[DTYPE], 36],
    cinert: InlineArray[Scalar[DTYPE], 10],
):
    """d(mulInertVec(cinert, v))/d(v), 6x6 row-major."""
    for i in range(36):
        D[i] = Scalar[DTYPE](0)
    D[0 + 0] = cinert[0]
    D[0 + 1] = cinert[3]
    D[0 + 2] = cinert[4]
    D[0 + 4] = -cinert[8]
    D[0 + 5] = cinert[7]
    D[6 + 0] = cinert[3]
    D[6 + 1] = cinert[1]
    D[6 + 2] = cinert[5]
    D[6 + 3] = cinert[8]
    D[6 + 5] = -cinert[6]
    D[12 + 0] = cinert[4]
    D[12 + 1] = cinert[5]
    D[12 + 2] = cinert[2]
    D[12 + 3] = -cinert[7]
    D[12 + 4] = cinert[6]
    D[18 + 1] = cinert[8]
    D[18 + 2] = -cinert[7]
    D[18 + 3] = cinert[9]
    D[24 + 2] = cinert[6]
    D[24 + 0] = -cinert[8]
    D[24 + 4] = cinert[9]
    D[30 + 0] = cinert[7]
    D[30 + 1] = -cinert[6]
    D[30 + 5] = cinert[9]


@always_inline
def _mulInertVec[
    DTYPE: DType,
](
    mut res: InlineArray[Scalar[DTYPE], 6],
    cinert: InlineArray[Scalar[DTYPE], 10],
    v: InlineArray[Scalar[DTYPE], 6],
):
    """res = cinert * v (spatial inertia × spatial vector)."""
    res[0] = (
        cinert[0] * v[0]
        + cinert[3] * v[1]
        + cinert[4] * v[2]
        - cinert[8] * v[4]
        + cinert[7] * v[5]
    )
    res[1] = (
        cinert[3] * v[0]
        + cinert[1] * v[1]
        + cinert[5] * v[2]
        + cinert[8] * v[3]
        - cinert[6] * v[5]
    )
    res[2] = (
        cinert[4] * v[0]
        + cinert[5] * v[1]
        + cinert[2] * v[2]
        - cinert[7] * v[3]
        + cinert[6] * v[4]
    )
    res[3] = cinert[8] * v[1] - cinert[7] * v[2] + cinert[9] * v[3]
    res[4] = cinert[6] * v[2] - cinert[8] * v[0] + cinert[9] * v[4]
    res[5] = cinert[7] * v[0] - cinert[6] * v[1] + cinert[9] * v[5]


@always_inline
def _matmul_6x6_x_6x6[
    DTYPE: DType,
](
    mut result: InlineArray[Scalar[DTYPE], 36],
    A: InlineArray[Scalar[DTYPE], 36],
    B: InlineArray[Scalar[DTYPE], 36],
):
    """result = A @ B, both 6x6 row-major."""
    for i in range(6):
        for k in range(6):
            var s = Scalar[DTYPE](0)
            for j in range(6):
                s += A[i * 6 + j] * B[j * 6 + k]
            result[i * 6 + k] = s


# =============================================================================
# Per-env RNE velocity derivative (verbatim algorithm; fields accessors)
# =============================================================================


@always_inline
def _rne_vel_derivative_env[
    DTYPE: DType,
    D: DimsLike,
    L_BODIES: Layout,
    L_JOINTS: Layout,
    L_XIPOS: Layout,
    L_XQUAT: Layout,
    L_QVEL: Layout,
    L_CDOF: Layout,
    L_CINERT: Layout,
    L_CVEL_SC: Layout,
    L_DCVEL: Layout,
    L_DCDOFDOT: Layout,
    L_QDERIV: Layout,
](
    env: Int,
    njoint: Int,
    dims: D,
    bodies: LayoutTensor[
        DTYPE, L_BODIES, MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, L_JOINTS, MutAnyOrigin
    ],
    xipos: LayoutTensor[DTYPE, L_XIPOS, MutAnyOrigin],
    xquat: LayoutTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, L_QVEL, MutAnyOrigin],
    cdof: LayoutTensor[DTYPE, L_CDOF, MutAnyOrigin],
    cinert: LayoutTensor[
        DTYPE, L_CINERT, MutAnyOrigin
    ],
    cdof_sc: LayoutTensor[DTYPE, L_CDOF, MutAnyOrigin],
    cvel_sc: LayoutTensor[
        DTYPE, L_CVEL_SC, MutAnyOrigin
    ],
    cdof_dot: LayoutTensor[DTYPE, L_CDOF, MutAnyOrigin],
    dcvel: LayoutTensor[
        DTYPE, L_DCVEL, MutAnyOrigin
    ],
    dcdofdot: LayoutTensor[
        DTYPE, L_DCDOFDOT, MutAnyOrigin
    ],
    dcacc: LayoutTensor[
        DTYPE, L_DCVEL, MutAnyOrigin
    ],
    dcfrcbody: LayoutTensor[
        DTYPE, L_DCVEL, MutAnyOrigin
    ],
    qderiv: LayoutTensor[DTYPE, L_QDERIV, MutAnyOrigin],
):
    """d(qfrc_bias)/d(qvel) SUBTRACTED into qderiv for one env. Persistent
    intermediates are zeroed here (scratch is reused across envs/steps)."""
    var nv = dims.get_nv()
    var nbody = dims.get_nbody()
    # Zero the reused scratch slices for this env.
    for i in range(nbody * 10):
        cinert[env, i] = 0
    for i in range(nv * 6):
        cdof_sc[env, i] = 0
        cdof_dot[env, i] = 0
    for i in range(nbody * 6):
        cvel_sc[env, i] = 0
    for i in range(nbody * 6 * nv):
        dcvel[env, i] = 0
        dcacc[env, i] = 0
        dcfrcbody[env, i] = 0
    for i in range(nv * 6 * nv):
        dcdofdot[env, i] = 0

    # ── Step 0: subtree COM + reexpress cinert / cdof_sc / cvel / cdof_dot ──
    var total_mass = Scalar[DTYPE](0)
    var com_x = Scalar[DTYPE](0)
    var com_y = Scalar[DTYPE](0)
    var com_z = Scalar[DTYPE](0)
    for b in range(nbody):
        var m = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_MASS])
        total_mass += m
        com_x += m * rebind[Scalar[DTYPE]](xipos[env, b * 3 + 0])
        com_y += m * rebind[Scalar[DTYPE]](xipos[env, b * 3 + 1])
        com_z += m * rebind[Scalar[DTYPE]](xipos[env, b * 3 + 2])
    if total_mass > Scalar[DTYPE](0):
        com_x = com_x / total_mass
        com_y = com_y / total_mass
        com_z = com_z / total_mass

    # cinert at subtree COM
    for b in range(nbody):
        var mass = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_MASS])
        var dx = rebind[Scalar[DTYPE]](xipos[env, b * 3 + 0]) - com_x
        var dy = rebind[Scalar[DTYPE]](xipos[env, b * 3 + 1]) - com_y
        var dz = rebind[Scalar[DTYPE]](xipos[env, b * 3 + 2]) - com_z

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
        var iq = quat_mul(bqx, bqy, bqz, bqw, iqx, iqy, iqz, iqw)
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

        var Iw_xx = (
            Ixx_local * r00 * r00
            + Iyy_local * r01 * r01
            + Izz_local * r02 * r02
        )
        var Iw_yy = (
            Ixx_local * r10 * r10
            + Iyy_local * r11 * r11
            + Izz_local * r12 * r12
        )
        var Iw_zz = (
            Ixx_local * r20 * r20
            + Iyy_local * r21 * r21
            + Izz_local * r22 * r22
        )
        var Iw_xy = (
            Ixx_local * r00 * r10
            + Iyy_local * r01 * r11
            + Izz_local * r02 * r12
        )
        var Iw_xz = (
            Ixx_local * r00 * r20
            + Iyy_local * r01 * r21
            + Izz_local * r02 * r22
        )
        var Iw_yz = (
            Ixx_local * r10 * r20
            + Iyy_local * r11 * r21
            + Izz_local * r12 * r22
        )

        var d_sq = dx * dx + dy * dy + dz * dz
        cinert[env, b * 10 + 0] = Iw_xx + mass * (d_sq - dx * dx)
        cinert[env, b * 10 + 1] = Iw_yy + mass * (d_sq - dy * dy)
        cinert[env, b * 10 + 2] = Iw_zz + mass * (d_sq - dz * dz)
        cinert[env, b * 10 + 3] = Iw_xy - mass * dx * dy
        cinert[env, b * 10 + 4] = Iw_xz - mass * dx * dz
        cinert[env, b * 10 + 5] = Iw_yz - mass * dy * dz
        cinert[env, b * 10 + 6] = mass * dx
        cinert[env, b * 10 + 7] = mass * dy
        cinert[env, b * 10 + 8] = mass * dz
        cinert[env, b * 10 + 9] = mass

    # dof_bodyid lookup
    var dof_bodyid = InlineArray[Int, D.CAP_NV if D.CAP_NV > 0 else 1](uninitialized=True)
    for i in range(nv):
        dof_bodyid[i] = 0
    for j in range(njoint):
        var jtype = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var body_id = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID]))
        var dof_adr = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR]))
        var num_dof = 1
        if jtype == JNT_FREE:
            num_dof = 6
        elif jtype == JNT_BALL:
            num_dof = 3
        for d in range(num_dof):
            dof_bodyid[dof_adr + d] = body_id

    # cdof_sc: shift lin to subtree COM
    for d in range(nv):
        var body = dof_bodyid[d]
        var ax = rebind[Scalar[DTYPE]](cdof[env, d * 6 + 0])
        var ay = rebind[Scalar[DTYPE]](cdof[env, d * 6 + 1])
        var az = rebind[Scalar[DTYPE]](cdof[env, d * 6 + 2])
        cdof_sc[env, d * 6 + 0] = ax
        cdof_sc[env, d * 6 + 1] = ay
        cdof_sc[env, d * 6 + 2] = az
        var sx = com_x - rebind[Scalar[DTYPE]](xipos[env, body * 3 + 0])
        var sy = com_y - rebind[Scalar[DTYPE]](xipos[env, body * 3 + 1])
        var sz = com_z - rebind[Scalar[DTYPE]](xipos[env, body * 3 + 2])
        cdof_sc[env, d * 6 + 3] = (
            rebind[Scalar[DTYPE]](cdof[env, d * 6 + 3]) + ay * sz - az * sy
        )
        cdof_sc[env, d * 6 + 4] = (
            rebind[Scalar[DTYPE]](cdof[env, d * 6 + 4]) + az * sx - ax * sz
        )
        cdof_sc[env, d * 6 + 5] = (
            rebind[Scalar[DTYPE]](cdof[env, d * 6 + 5]) + ax * sy - ay * sx
        )

    # body_dofadr / body_dofnum lookup
    var body_dofadr = InlineArray[Int, D.CAP_NBODY](uninitialized=True)
    var body_dofnum = InlineArray[Int, D.CAP_NBODY](uninitialized=True)
    for b in range(nbody):
        body_dofadr[b] = -1
        body_dofnum[b] = 0
    for j in range(njoint):
        var jtype = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var body = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID]))
        var dof_adr = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR]))
        var num_dof = 1
        if jtype == JNT_FREE:
            num_dof = 6
        elif jtype == JNT_BALL:
            num_dof = 3
        if body_dofadr[body] < 0:
            body_dofadr[body] = dof_adr
        body_dofnum[body] = body_dofnum[body] + num_dof

    # cvel_sc + cdof_dot (per body, accumulate over the body's DOFs)
    for b in range(nbody):
        var parent = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_PARENT]))
        var cv_wx = Scalar[DTYPE](0)
        var cv_wy = Scalar[DTYPE](0)
        var cv_wz = Scalar[DTYPE](0)
        var cv_vx = Scalar[DTYPE](0)
        var cv_vy = Scalar[DTYPE](0)
        var cv_vz = Scalar[DTYPE](0)
        if parent >= 0:
            cv_wx = rebind[Scalar[DTYPE]](cvel_sc[env, parent * 6 + 0])
            cv_wy = rebind[Scalar[DTYPE]](cvel_sc[env, parent * 6 + 1])
            cv_wz = rebind[Scalar[DTYPE]](cvel_sc[env, parent * 6 + 2])
            cv_vx = rebind[Scalar[DTYPE]](cvel_sc[env, parent * 6 + 3])
            cv_vy = rebind[Scalar[DTYPE]](cvel_sc[env, parent * 6 + 4])
            cv_vz = rebind[Scalar[DTYPE]](cvel_sc[env, parent * 6 + 5])

        for j in range(njoint):
            var body_id = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID])
            )
            if body_id != b:
                continue
            var jtype = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
            var dof_adr = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR])
            )

            if jtype == JNT_FREE:
                for d in range(3):
                    var dof = dof_adr + d
                    var qv = rebind[Scalar[DTYPE]](qvel[env, dof])
                    cv_wx += rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 0]) * qv
                    cv_wy += rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 1]) * qv
                    cv_wz += rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 2]) * qv
                    cv_vx += rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 3]) * qv
                    cv_vy += rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 4]) * qv
                    cv_vz += rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 5]) * qv
                for d in range(3, 6):
                    var dof = dof_adr + d
                    var s_ax = rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 0])
                    var s_ay = rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 1])
                    var s_az = rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 2])
                    var s_lx = rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 3])
                    var s_ly = rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 4])
                    var s_lz = rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 5])
                    cdof_dot[env, dof * 6 + 0] = cv_wy * s_az - cv_wz * s_ay
                    cdof_dot[env, dof * 6 + 1] = cv_wz * s_ax - cv_wx * s_az
                    cdof_dot[env, dof * 6 + 2] = cv_wx * s_ay - cv_wy * s_ax
                    cdof_dot[env, dof * 6 + 3] = (
                        cv_wy * s_lz - cv_wz * s_ly
                    ) + (cv_vy * s_az - cv_vz * s_ay)
                    cdof_dot[env, dof * 6 + 4] = (
                        cv_wz * s_lx - cv_wx * s_lz
                    ) + (cv_vz * s_ax - cv_vx * s_az)
                    cdof_dot[env, dof * 6 + 5] = (
                        cv_wx * s_ly - cv_wy * s_lx
                    ) + (cv_vx * s_ay - cv_vy * s_ax)
                    var qv = rebind[Scalar[DTYPE]](qvel[env, dof])
                    cv_wx += s_ax * qv
                    cv_wy += s_ay * qv
                    cv_wz += s_az * qv
                    cv_vx += s_lx * qv
                    cv_vy += s_ly * qv
                    cv_vz += s_lz * qv

            elif jtype == JNT_BALL:
                for d in range(3):
                    var dof = dof_adr + d
                    var s_ax = rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 0])
                    var s_ay = rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 1])
                    var s_az = rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 2])
                    var s_lx = rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 3])
                    var s_ly = rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 4])
                    var s_lz = rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 5])
                    cdof_dot[env, dof * 6 + 0] = cv_wy * s_az - cv_wz * s_ay
                    cdof_dot[env, dof * 6 + 1] = cv_wz * s_ax - cv_wx * s_az
                    cdof_dot[env, dof * 6 + 2] = cv_wx * s_ay - cv_wy * s_ax
                    cdof_dot[env, dof * 6 + 3] = (
                        cv_wy * s_lz - cv_wz * s_ly
                    ) + (cv_vy * s_az - cv_vz * s_ay)
                    cdof_dot[env, dof * 6 + 4] = (
                        cv_wz * s_lx - cv_wx * s_lz
                    ) + (cv_vz * s_ax - cv_vx * s_az)
                    cdof_dot[env, dof * 6 + 5] = (
                        cv_wx * s_ly - cv_wy * s_lx
                    ) + (cv_vx * s_ay - cv_vy * s_ax)
                for d in range(3):
                    var dof = dof_adr + d
                    var qv = rebind[Scalar[DTYPE]](qvel[env, dof])
                    cv_wx += rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 0]) * qv
                    cv_wy += rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 1]) * qv
                    cv_wz += rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 2]) * qv
                    cv_vx += rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 3]) * qv
                    cv_vy += rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 4]) * qv
                    cv_vz += rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 5]) * qv

            else:
                var dof = dof_adr
                var s_ax = rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 0])
                var s_ay = rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 1])
                var s_az = rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 2])
                var s_lx = rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 3])
                var s_ly = rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 4])
                var s_lz = rebind[Scalar[DTYPE]](cdof_sc[env, dof * 6 + 5])
                cdof_dot[env, dof * 6 + 0] = cv_wy * s_az - cv_wz * s_ay
                cdof_dot[env, dof * 6 + 1] = cv_wz * s_ax - cv_wx * s_az
                cdof_dot[env, dof * 6 + 2] = cv_wx * s_ay - cv_wy * s_ax
                cdof_dot[env, dof * 6 + 3] = (cv_wy * s_lz - cv_wz * s_ly) + (
                    cv_vy * s_az - cv_vz * s_ay
                )
                cdof_dot[env, dof * 6 + 4] = (cv_wz * s_lx - cv_wx * s_lz) + (
                    cv_vz * s_ax - cv_vx * s_az
                )
                cdof_dot[env, dof * 6 + 5] = (cv_wx * s_ly - cv_wy * s_lx) + (
                    cv_vx * s_ay - cv_vy * s_ax
                )
                var qv = rebind[Scalar[DTYPE]](qvel[env, dof])
                cv_wx += s_ax * qv
                cv_wy += s_ay * qv
                cv_wz += s_az * qv
                cv_vx += s_lx * qv
                cv_vy += s_ly * qv
                cv_vz += s_lz * qv

        cvel_sc[env, b * 6 + 0] = cv_wx
        cvel_sc[env, b * 6 + 1] = cv_wy
        cvel_sc[env, b * 6 + 2] = cv_wz
        cvel_sc[env, b * 6 + 3] = cv_vx
        cvel_sc[env, b * 6 + 4] = cv_vy
        cvel_sc[env, b * 6 + 5] = cv_vz

    # ── Step 1: Dcvel + Dcdofdot ─────────────────────────────────────────
    for b in range(nbody):
        var parent = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_PARENT]))
        if parent >= 0:
            for idx in range(6 * nv):
                dcvel[env, b * 6 * nv + idx] = dcvel[env, parent * 6 * nv + idx]
        if body_dofadr[b] < 0:
            continue
        for j in range(njoint):
            var body_id = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID])
            )
            if body_id != b:
                continue
            var jtype = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
            var dof_adr = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR])
            )

            if jtype == JNT_FREE:
                for d in range(3):
                    var dof = dof_adr + d
                    for kk in range(6):
                        dcvel[env, b * 6 * nv + kk * nv + dof] += rebind[
                            Scalar[DTYPE]
                        ](cdof_sc[env, dof * 6 + kk])
                for d in range(3):
                    var dof = dof_adr + 3 + d
                    var cdof_v = InlineArray[Scalar[DTYPE], 6](
                        uninitialized=True
                    )
                    for kk in range(6):
                        cdof_v[kk] = rebind[Scalar[DTYPE]](
                            cdof_sc[env, dof * 6 + kk]
                        )
                    var mat = InlineArray[Scalar[DTYPE], 36](uninitialized=True)
                    _mjd_crossMotion_vel(mat, cdof_v)
                    for ii in range(6):
                        for kk in range(nv):
                            var s = Scalar[DTYPE](0)
                            for jj in range(6):
                                s += mat[ii * 6 + jj] * rebind[Scalar[DTYPE]](
                                    dcvel[env, b * 6 * nv + jj * nv + kk]
                                )
                            dcdofdot[env, dof * 6 * nv + ii * nv + kk] = s
                    for kk in range(6):
                        dcvel[env, b * 6 * nv + kk * nv + dof] += rebind[
                            Scalar[DTYPE]
                        ](cdof_sc[env, dof * 6 + kk])

            elif jtype == JNT_BALL:
                for d in range(3):
                    var dof = dof_adr + d
                    var cdof_v = InlineArray[Scalar[DTYPE], 6](
                        uninitialized=True
                    )
                    for kk in range(6):
                        cdof_v[kk] = rebind[Scalar[DTYPE]](
                            cdof_sc[env, dof * 6 + kk]
                        )
                    var mat = InlineArray[Scalar[DTYPE], 36](uninitialized=True)
                    _mjd_crossMotion_vel(mat, cdof_v)
                    for ii in range(6):
                        for kk in range(nv):
                            var s = Scalar[DTYPE](0)
                            for jj in range(6):
                                s += mat[ii * 6 + jj] * rebind[Scalar[DTYPE]](
                                    dcvel[env, b * 6 * nv + jj * nv + kk]
                                )
                            dcdofdot[env, dof * 6 * nv + ii * nv + kk] = s
                    for kk in range(6):
                        dcvel[env, b * 6 * nv + kk * nv + dof] += rebind[
                            Scalar[DTYPE]
                        ](cdof_sc[env, dof * 6 + kk])

            else:
                var dof = dof_adr
                var cdof_v = InlineArray[Scalar[DTYPE], 6](uninitialized=True)
                for kk in range(6):
                    cdof_v[kk] = rebind[Scalar[DTYPE]](
                        cdof_sc[env, dof * 6 + kk]
                    )
                var mat = InlineArray[Scalar[DTYPE], 36](uninitialized=True)
                _mjd_crossMotion_vel(mat, cdof_v)
                for ii in range(6):
                    for kk in range(nv):
                        var s = Scalar[DTYPE](0)
                        for jj in range(6):
                            s += mat[ii * 6 + jj] * rebind[Scalar[DTYPE]](
                                dcvel[env, b * 6 * nv + jj * nv + kk]
                            )
                        dcdofdot[env, dof * 6 * nv + ii * nv + kk] = s
                for kk in range(6):
                    dcvel[env, b * 6 * nv + kk * nv + dof] += rebind[
                        Scalar[DTYPE]
                    ](cdof_sc[env, dof * 6 + kk])

    # ── Step 2: forward pass — Dcacc + Dcfrcbody ─────────────────────────
    for b in range(nbody):
        var parent = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_PARENT]))
        if parent >= 0:
            for idx in range(6 * nv):
                dcacc[env, b * 6 * nv + idx] = dcacc[env, parent * 6 * nv + idx]

        if body_dofadr[b] >= 0:
            var dof_start = body_dofadr[b]
            var dof_end = dof_start + body_dofnum[b]
            for j_dof in range(dof_start, dof_end):
                for k in range(6):
                    dcacc[env, b * 6 * nv + k * nv + j_dof] += rebind[
                        Scalar[DTYPE]
                    ](cdof_dot[env, j_dof * 6 + k])
                var qvel_j = rebind[Scalar[DTYPE]](qvel[env, j_dof])
                for idx in range(6 * nv):
                    dcacc[env, b * 6 * nv + idx] += (
                        rebind[Scalar[DTYPE]](dcdofdot[env, j_dof * 6 * nv + idx])
                        * qvel_j
                    )

        var ci = InlineArray[Scalar[DTYPE], 10](uninitialized=True)
        for k in range(10):
            ci[k] = rebind[Scalar[DTYPE]](cinert[env, b * 10 + k])

        var dmul = InlineArray[Scalar[DTYPE], 36](uninitialized=True)
        _mjd_mulInertVec_vel(dmul, ci)

        for ii in range(6):
            for kk in range(nv):
                var s = Scalar[DTYPE](0)
                for jj in range(6):
                    s += dmul[ii * 6 + jj] * rebind[Scalar[DTYPE]](
                        dcacc[env, b * 6 * nv + jj * nv + kk]
                    )
                dcfrcbody[env, b * 6 * nv + ii * nv + kk] = s

        var cv = InlineArray[Scalar[DTYPE], 6](uninitialized=True)
        for k in range(6):
            cv[k] = rebind[Scalar[DTYPE]](cvel_sc[env, b * 6 + k])
        var tmp6 = InlineArray[Scalar[DTYPE], 6](uninitialized=True)
        _mulInertVec(tmp6, ci, cv)

        var mat = InlineArray[Scalar[DTYPE], 36](uninitialized=True)
        _mjd_crossForce_vel(mat, tmp6)
        var mat1 = InlineArray[Scalar[DTYPE], 36](uninitialized=True)
        _mjd_crossForce_frc(mat1, cv)
        var mat2 = InlineArray[Scalar[DTYPE], 36](uninitialized=True)
        _matmul_6x6_x_6x6(mat2, mat1, dmul)
        for k in range(36):
            mat[k] += mat2[k]

        for ii in range(6):
            for kk in range(nv):
                var s = Scalar[DTYPE](0)
                for jj in range(6):
                    s += mat[ii * 6 + jj] * rebind[Scalar[DTYPE]](
                        dcvel[env, b * 6 * nv + jj * nv + kk]
                    )
                dcfrcbody[env, b * 6 * nv + ii * nv + kk] += s

    # ── Step 3: backward pass — accumulate to parents ────────────────────
    for b in range(nbody - 1, 0, -1):
        var parent = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_PARENT]))
        if parent >= 0:
            for idx in range(6 * nv):
                dcfrcbody[env, parent * 6 * nv + idx] += rebind[Scalar[DTYPE]](
                    dcfrcbody[env, b * 6 * nv + idx]
                )

    # ── Step 4: project to joint space — SUBTRACT into qderiv ────────────
    for i in range(nv):
        var body_i = dof_bodyid[i]
        for k in range(nv):
            var s = Scalar[DTYPE](0)
            for comp in range(6):
                s += rebind[Scalar[DTYPE]](cdof_sc[env, i * 6 + comp]) * rebind[
                    Scalar[DTYPE]
                ](dcfrcbody[env, body_i * 6 * nv + comp * nv + k])
            qderiv[env, i * nv + k] -= s


# ── launchable kernel (serial: one thread per env) ────────────────────────
def _rne_vel_derivative_fields_kernel[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
](
    njoint_arg: Int64,
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    xipos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin],
    xquat: LayoutTensor[DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    cdof: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * 6), MutAnyOrigin],
    cinert: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 10), MutAnyOrigin
    ],
    cdof_sc: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * 6), MutAnyOrigin],
    cvel_sc: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 6), MutAnyOrigin
    ],
    cdof_dot: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * 6), MutAnyOrigin],
    dcvel: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 6 * NV), MutAnyOrigin
    ],
    dcdofdot: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV * 6 * NV), MutAnyOrigin
    ],
    dcacc: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 6 * NV), MutAnyOrigin
    ],
    dcfrcbody: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 6 * NV), MutAnyOrigin
    ],
    qderiv: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
):
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var njoint = Int(njoint_arg)
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _rne_vel_derivative_env[DTYPE](
        env, njoint, Dims[nv=NV, nbody=NBODY, njoint=NJOINT](), bodies, joints, xipos, xquat, qvel, cdof, cinert,
        cdof_sc, cvel_sc, cdof_dot, dcvel, dcdofdot, dcacc, dcfrcbody, qderiv,
    )


def compute_rne_vel_derivative[

    target: StaticString,
    DTYPE: DType,
    BATCH: Int,
    # Appended, not grouped with NEXCLUDE — see `fields.Model`.

    D: DimsLike,
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    mut scratch: DynamicsScratch[DTYPE, D, BATCH],
    # ⚠ `Dims[nv=NV, nbody=NBODY]` rather than a `D: DimsLike` parameter on
    # this function: the body reads NV/NBODY throughout and every caller
    # passes them positionally, so taking `D` here would pull those callers
    # into 1c. Two `Dims[...]` with equal arguments are ONE type, so this
    # matches whatever the caller built — and when the sweep gives this
    # function a real `D`, the adapter is deleted, not rewired.
    mut iscratch: ImplicitScratch[DTYPE, D, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """`d(qfrc_bias)/d(qvel)` SUBTRACTED into iscratch.qderiv (caller pre-loads
    the damping diagonal). Reads FK products (d.xipos/xquat), d.qvel, and
    scratch.cdof; both targets. NJOINT>0 required (njoint read from meta)."""
    var njoint = Int(m.meta.data[MODEL_META_IDX_NJOINT])

    comptime L_BODY = Layout.row_major(D.NBODY, MODEL_BODY_SIZE)
    comptime L_JOINT = Layout.row_major(D.NJOINT, MODEL_JOINT_SIZE)
    comptime L_B3 = Layout.row_major(BATCH, D.NBODY * 3)
    comptime L_B4 = Layout.row_major(BATCH, D.NBODY * 4)
    comptime L_NV = Layout.row_major(BATCH, D.NV)
    comptime L_NV6 = Layout.row_major(BATCH, D.NV * 6)
    comptime L_B10 = Layout.row_major(BATCH, D.NBODY * 10)
    comptime L_B6 = Layout.row_major(BATCH, D.NBODY * 6)
    comptime L_DCVEL = Layout.row_major(BATCH, D.NBODY * 6 * D.NV)
    comptime L_DCDOF = Layout.row_major(BATCH, D.NV * 6 * D.NV)
    comptime L_QD = Layout.row_major(BATCH, D.NV * D.NV)

    comptime if target == "cpu":
        var bodies_v = m.bodies.lt["cpu", L_BODY]()
        var joints_v = m.joints.lt["cpu", L_JOINT]()
        var xipos_v = d.xipos.lt["cpu", L_B3]()
        var xquat_v = d.xquat.lt["cpu", L_B4]()
        var qvel_v = d.qvel.lt["cpu", L_NV]()
        var cdof_v = scratch.cdof.lt["cpu", L_NV6]()
        var cinert_v = iscratch.cinert.lt["cpu", L_B10]()
        var cdof_sc_v = iscratch.cdof_sc.lt["cpu", L_NV6]()
        var cvel_sc_v = iscratch.cvel_sc.lt["cpu", L_B6]()
        var cdof_dot_v = iscratch.cdof_dot.lt["cpu", L_NV6]()
        var dcvel_v = iscratch.dcvel.lt["cpu", L_DCVEL]()
        var dcdofdot_v = iscratch.dcdofdot.lt["cpu", L_DCDOF]()
        var dcacc_v = iscratch.dcacc.lt["cpu", L_DCVEL]()
        var dcfrcbody_v = iscratch.dcfrcbody.lt["cpu", L_DCVEL]()
        var qderiv_v = iscratch.qderiv.lt["cpu", L_QD]()
        for e in range(BATCH):
            _rne_vel_derivative_env[DTYPE](
                e, njoint, AsStatic[D](), bodies_v, joints_v, xipos_v, xquat_v, qvel_v,
                cdof_v, cinert_v, cdof_sc_v, cvel_sc_v, cdof_dot_v, dcvel_v,
                dcdofdot_v, dcacc_v, dcfrcbody_v, qderiv_v,
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + QD_TPB - 1) // QD_TPB
        c.enqueue_function[
            _rne_vel_derivative_fields_kernel[
                DTYPE, D.NV, D.NBODY, D.NJOINT, BATCH
            ]
        ](
            Int64(njoint),
            m.bodies.lt["gpu", L_BODY](),
            m.joints.lt["gpu", L_JOINT](),
            d.xipos.lt["gpu", L_B3](),
            d.xquat.lt["gpu", L_B4](),
            d.qvel.lt["gpu", L_NV](),
            scratch.cdof.lt["gpu", L_NV6](),
            iscratch.cinert.lt["gpu", L_B10](),
            iscratch.cdof_sc.lt["gpu", L_NV6](),
            iscratch.cvel_sc.lt["gpu", L_B6](),
            iscratch.cdof_dot.lt["gpu", L_NV6](),
            iscratch.dcvel.lt["gpu", L_DCVEL](),
            iscratch.dcdofdot.lt["gpu", L_DCDOF](),
            iscratch.dcacc.lt["gpu", L_DCVEL](),
            iscratch.dcfrcbody.lt["gpu", L_DCVEL](),
            iscratch.qderiv.lt["gpu", L_QD](),
            grid_dim=(BLOCKS,),
            block_dim=(QD_TPB,),
        )

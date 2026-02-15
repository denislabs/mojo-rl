"""RNE velocity derivative: d(bias_forces)/d(qvel).

Computes the dense NV×NV matrix qDeriv = d(qfrc_bias)/d(qvel), which captures
how Coriolis/centrifugal forces change with velocity. Used by ImplicitIntegrator
to form M_hat = M - dt*qDeriv for improved numerical stability.

Algorithm follows MuJoCo engine_derivative.c:mjd_rne_vel_dense(), adapted to
our at-COM convention. We shift quantities to body-origin convention internally
to match MuJoCo's simple propagation rules, then project back.

Reference: MuJoCo engine_derivative.c:321-461 (dense version)
Reference: Analytical derivative of Featherstone's RNE algorithm

Key data structures (all in body-origin convention):
  cinert[b*10 : b*10+10]  = spatial inertia at body origin
      [Ixx, Iyy, Izz, Ixy, Ixz, Iyz, m*cx, m*cy, m*cz, mass]
  cvel_origin[b*6 : b*6+6] = spatial velocity at body origin
      [wx, wy, wz, vx, vy, vz]
  cdof_origin[d*6 : d*6+6] = spatial motion axis at body origin
      [ang_x, ang_y, ang_z, lin_x, lin_y, lin_z]

Scratch space sizes (compile-time, for InlineArrays):
  CINERT_SIZE  = NBODY * 10
  CVEL_SIZE    = NBODY * 6
  CDOF_O_SIZE  = NV * 6
  DCVEL_SIZE   = NBODY * 6 * NV
  DCDOFDOT_SIZE = NV * 6 * NV
  DCACC_SIZE   = NBODY * 6 * NV
  DCFRC_SIZE   = NBODY * 6 * NV
"""

from layout import LayoutTensor, Layout

from ..types import Model, Data, _max_one
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..kinematics.quat_math import quat_rotate, quat_mul
from ..gpu.constants import (
    xpos_offset,
    xquat_offset,
    xipos_offset,
    xvel_offset,
    xangvel_offset,
    qvel_offset,
    model_metadata_offset,
    model_joint_offset,
    model_body_offset,
    ws_cdof_offset,
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
    MODEL_META_IDX_NJOINT,
    ws_implicit_qderiv_offset,
    ws_implicit_cdof_origin_offset,
    ws_implicit_cvel_origin_offset,
    ws_implicit_cinert_offset,
    ws_implicit_cdof_dot_offset,
    ws_implicit_dcvel_offset,
    ws_implicit_dcdofdot_offset,
    ws_implicit_dcacc_offset,
    ws_implicit_dcfrcbody_offset,
)


# =============================================================================
# Spatial derivative helpers (matching MuJoCo engine_derivative.c:34-211)
# =============================================================================


@always_inline
fn _mjd_crossMotion_vel[
    DTYPE: DType,
](mut D: InlineArray[Scalar[DTYPE], 36], v: InlineArray[Scalar[DTYPE], 6],):
    """Derivative of crossMotion(vel, v) w.r.t. vel. D is 6x6 row-major.

    crossMotion(vel, v) computes vel x_motion v (spatial cross for motion vectors).
    This function returns the 6x6 Jacobian D such that:
      d(crossMotion(vel, v))/d(vel) = D

    Reference: MuJoCo engine_derivative.c:65-97
    """
    for i in range(36):
        D[i] = Scalar[DTYPE](0)

    # res[0] = -vel[2]*v[1] + vel[1]*v[2]
    D[0 + 2] = -v[1]
    D[0 + 1] = v[2]

    # res[1] = vel[2]*v[0] - vel[0]*v[2]
    D[6 + 2] = v[0]
    D[6 + 0] = -v[2]

    # res[2] = -vel[1]*v[0] + vel[0]*v[1]
    D[12 + 1] = -v[0]
    D[12 + 0] = v[1]

    # res[3] = -vel[2]*v[4] + vel[1]*v[5] - vel[5]*v[1] + vel[4]*v[2]
    D[18 + 2] = -v[4]
    D[18 + 1] = v[5]
    D[18 + 5] = -v[1]
    D[18 + 4] = v[2]

    # res[4] = vel[2]*v[3] - vel[0]*v[5] + vel[5]*v[0] - vel[3]*v[2]
    D[24 + 2] = v[3]
    D[24 + 0] = -v[5]
    D[24 + 5] = v[0]
    D[24 + 3] = -v[2]

    # res[5] = -vel[1]*v[3] + vel[0]*v[4] - vel[4]*v[0] + vel[3]*v[1]
    D[30 + 1] = -v[3]
    D[30 + 0] = v[4]
    D[30 + 4] = -v[0]
    D[30 + 3] = v[1]


@always_inline
fn _mjd_crossForce_vel[
    DTYPE: DType,
](mut D: InlineArray[Scalar[DTYPE], 36], f: InlineArray[Scalar[DTYPE], 6],):
    """Derivative of crossForce(vel, f) w.r.t. vel. D is 6x6 row-major.

    Reference: MuJoCo engine_derivative.c:101-133
    """
    for i in range(36):
        D[i] = Scalar[DTYPE](0)

    # res[0] = -vel[2]*f[1] + vel[1]*f[2] - vel[5]*f[4] + vel[4]*f[5]
    D[0 + 2] = -f[1]
    D[0 + 1] = f[2]
    D[0 + 5] = -f[4]
    D[0 + 4] = f[5]

    # res[1] = vel[2]*f[0] - vel[0]*f[2] + vel[5]*f[3] - vel[3]*f[5]
    D[6 + 2] = f[0]
    D[6 + 0] = -f[2]
    D[6 + 5] = f[3]
    D[6 + 3] = -f[5]

    # res[2] = -vel[1]*f[0] + vel[0]*f[1] - vel[4]*f[3] + vel[3]*f[4]
    D[12 + 1] = -f[0]
    D[12 + 0] = f[1]
    D[12 + 4] = -f[3]
    D[12 + 3] = f[4]

    # res[3] = -vel[2]*f[4] + vel[1]*f[5]
    D[18 + 2] = -f[4]
    D[18 + 1] = f[5]

    # res[4] = vel[2]*f[3] - vel[0]*f[5]
    D[24 + 2] = f[3]
    D[24 + 0] = -f[5]

    # res[5] = -vel[1]*f[3] + vel[0]*f[4]
    D[30 + 1] = -f[3]
    D[30 + 0] = f[4]


@always_inline
fn _mjd_crossForce_frc[
    DTYPE: DType,
](mut D: InlineArray[Scalar[DTYPE], 36], vel: InlineArray[Scalar[DTYPE], 6],):
    """Derivative of crossForce(vel, f) w.r.t. f. D is 6x6 row-major.

    Reference: MuJoCo engine_derivative.c:137-169
    """
    for i in range(36):
        D[i] = Scalar[DTYPE](0)

    # res[0] = -vel[2]*f[1] + vel[1]*f[2] - vel[5]*f[4] + vel[4]*f[5]
    D[0 + 1] = -vel[2]
    D[0 + 2] = vel[1]
    D[0 + 4] = -vel[5]
    D[0 + 5] = vel[4]

    # res[1] = vel[2]*f[0] - vel[0]*f[2] + vel[5]*f[3] - vel[3]*f[5]
    D[6 + 0] = vel[2]
    D[6 + 2] = -vel[0]
    D[6 + 3] = vel[5]
    D[6 + 5] = -vel[3]

    # res[2] = -vel[1]*f[0] + vel[0]*f[1] - vel[4]*f[3] + vel[3]*f[4]
    D[12 + 0] = -vel[1]
    D[12 + 1] = vel[0]
    D[12 + 3] = -vel[4]
    D[12 + 4] = vel[3]

    # res[3] = -vel[2]*f[4] + vel[1]*f[5]
    D[18 + 4] = -vel[2]
    D[18 + 5] = vel[1]

    # res[4] = vel[2]*f[3] - vel[0]*f[5]
    D[24 + 3] = vel[2]
    D[24 + 5] = -vel[0]

    # res[5] = -vel[1]*f[3] + vel[0]*f[4]
    D[30 + 3] = -vel[1]
    D[30 + 4] = vel[0]


@always_inline
fn _mjd_mulInertVec_vel[
    DTYPE: DType,
](
    mut D: InlineArray[Scalar[DTYPE], 36],
    cinert: InlineArray[Scalar[DTYPE], 10],
):
    """Derivative of mulInertVec(cinert, v) w.r.t. v. D is 6x6 row-major.

    cinert = [Ixx, Iyy, Izz, Ixy, Ixz, Iyz, m*cx, m*cy, m*cz, mass]

    Reference: MuJoCo engine_derivative.c:173-211
    """
    for i in range(36):
        D[i] = Scalar[DTYPE](0)

    # res[0] = i[0]*v[0] + i[3]*v[1] + i[4]*v[2] - i[8]*v[4] + i[7]*v[5]
    D[0 + 0] = cinert[0]
    D[0 + 1] = cinert[3]
    D[0 + 2] = cinert[4]
    D[0 + 4] = -cinert[8]
    D[0 + 5] = cinert[7]

    # res[1] = i[3]*v[0] + i[1]*v[1] + i[5]*v[2] + i[8]*v[3] - i[6]*v[5]
    D[6 + 0] = cinert[3]
    D[6 + 1] = cinert[1]
    D[6 + 2] = cinert[5]
    D[6 + 3] = cinert[8]
    D[6 + 5] = -cinert[6]

    # res[2] = i[4]*v[0] + i[5]*v[1] + i[2]*v[2] - i[7]*v[3] + i[6]*v[4]
    D[12 + 0] = cinert[4]
    D[12 + 1] = cinert[5]
    D[12 + 2] = cinert[2]
    D[12 + 3] = -cinert[7]
    D[12 + 4] = cinert[6]

    # res[3] = i[8]*v[1] - i[7]*v[2] + i[9]*v[3]
    D[18 + 1] = cinert[8]
    D[18 + 2] = -cinert[7]
    D[18 + 3] = cinert[9]

    # res[4] = i[6]*v[2] - i[8]*v[0] + i[9]*v[4]
    D[24 + 2] = cinert[6]
    D[24 + 0] = -cinert[8]
    D[24 + 4] = cinert[9]

    # res[5] = i[7]*v[0] - i[6]*v[1] + i[9]*v[5]
    D[30 + 0] = cinert[7]
    D[30 + 1] = -cinert[6]
    D[30 + 5] = cinert[9]


@always_inline
fn _mulInertVec[
    DTYPE: DType,
](
    mut res: InlineArray[Scalar[DTYPE], 6],
    cinert: InlineArray[Scalar[DTYPE], 10],
    v: InlineArray[Scalar[DTYPE], 6],
):
    """Compute spatial inertia × spatial vector: res = cinert * v.

    cinert = [Ixx, Iyy, Izz, Ixy, Ixz, Iyz, m*cx, m*cy, m*cz, mass]

    Reference: MuJoCo engine_util_spatial.c mju_mulInertVec
    """
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


# =============================================================================
# Dense matrix helpers (small NV, all compile-time sizes)
# =============================================================================


@always_inline
fn _matmul_6x6_x_6xN[
    DTYPE: DType,
    N: Int,
    SIZE_6N: Int,
](
    mut result: InlineArray[Scalar[DTYPE], SIZE_6N],
    A: InlineArray[Scalar[DTYPE], 36],
    B: InlineArray[Scalar[DTYPE], SIZE_6N],
):
    """Compute result = A @ B where A is 6x6 and B is 6xN (row-major)."""
    for i in range(6):
        for k in range(N):
            var s: Scalar[DTYPE] = 0
            for j in range(6):
                s += A[i * 6 + j] * B[j * N + k]
            result[i * N + k] = s


@always_inline
fn _matmul_6x6_x_6x6[
    DTYPE: DType,
](
    mut result: InlineArray[Scalar[DTYPE], 36],
    A: InlineArray[Scalar[DTYPE], 36],
    B: InlineArray[Scalar[DTYPE], 36],
):
    """Compute result = A @ B where both are 6x6 (row-major)."""
    for i in range(6):
        for k in range(6):
            var s: Scalar[DTYPE] = 0
            for j in range(6):
                s += A[i * 6 + j] * B[j * 6 + k]
            result[i * 6 + k] = s


# =============================================================================
# Main: RNE velocity derivative (dense)
# =============================================================================


fn compute_rne_vel_derivative[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    M_SIZE: Int,
    CDOF_SIZE: Int,
    NGEOM: Int = 0,
    MAX_EQUALITY: Int = 0,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
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
    ],
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    cdof: InlineArray[Scalar[DTYPE], CDOF_SIZE],
    mut qDeriv: InlineArray[Scalar[DTYPE], M_SIZE],
) where DTYPE.is_floating_point():
    """Compute d(qfrc_bias)/d(qvel) and subtract from qDeriv.

    This is the dense version matching MuJoCo's mjd_rne_vel_dense().
    Internally converts to body-origin convention for clean propagation,
    then projects back to joint space.

    The result is SUBTRACTED from qDeriv (matching MuJoCo convention):
      qDeriv[i,j] -= d(bias[i])/d(qvel[j])

    For passive systems, call with qDeriv initialized to the diagonal
    damping terms: qDeriv[i,i] = -damping[i].

    Args:
        model: Static model configuration.
        data: Current state (xpos, xquat, xvel, xangvel, xipos from FK).
        cdof: Spatial motion axes per DOF at body COM (6*NV, from compute_cdof).
        qDeriv: NV×NV matrix. RNE derivative is SUBTRACTED from this.
    """
    # Compile-time sizes
    comptime V_SIZE = _max_one[NV]()
    comptime BODY6_SIZE = _max_one[NBODY * 6]()
    comptime CINERT_SIZE = _max_one[NBODY * 10]()
    comptime CDOF_O_SIZE = _max_one[NV * 6]()
    comptime DCVEL_SIZE = _max_one[NBODY * 6 * NV]()
    comptime DCDOFDOT_SIZE = _max_one[NV * 6 * NV]()

    # =========================================================================
    # Step 0: Precompute body-origin quantities
    # =========================================================================

    # --- cdof at body origin ---
    # cdof_origin[d] = cdof_com[d] shifted from xipos to xpos:
    #   ang: same, lin: += ang x (xpos - xipos)
    var cdof_origin = InlineArray[Scalar[DTYPE], CDOF_O_SIZE](
        uninitialized=True
    )
    for d in range(NV):
        cdof_origin[d * 6 + 0] = cdof[d * 6 + 0]
        cdof_origin[d * 6 + 1] = cdof[d * 6 + 1]
        cdof_origin[d * 6 + 2] = cdof[d * 6 + 2]
        # Will be set per-joint below
        cdof_origin[d * 6 + 3] = cdof[d * 6 + 3]
        cdof_origin[d * 6 + 4] = cdof[d * 6 + 4]
        cdof_origin[d * 6 + 5] = cdof[d * 6 + 5]

    for j in range(model.num_joints):
        var joint = model.joints[j]
        var body = joint.body_id
        var dof_adr = joint.dof_adr

        # Offset from body origin to body COM (in world frame)
        var cx = data.xipos[body * 3 + 0] - data.xpos[body * 3 + 0]
        var cy = data.xipos[body * 3 + 1] - data.xpos[body * 3 + 1]
        var cz = data.xipos[body * 3 + 2] - data.xpos[body * 3 + 2]

        # dx = xpos - xipos = -c
        var dx = -cx
        var dy = -cy
        var dz = -cz

        var num_dof = 1
        if joint.jnt_type == JNT_FREE:
            num_dof = 6
        elif joint.jnt_type == JNT_BALL:
            num_dof = 3

        for d in range(num_dof):
            var dof = dof_adr + d
            var ax = cdof[dof * 6 + 0]
            var ay = cdof[dof * 6 + 1]
            var az = cdof[dof * 6 + 2]

            # lin_origin = lin_com + ang x (xpos - xipos) = lin_com + ang x dx
            cdof_origin[dof * 6 + 3] = cdof[dof * 6 + 3] + ay * dz - az * dy
            cdof_origin[dof * 6 + 4] = cdof[dof * 6 + 4] + az * dx - ax * dz
            cdof_origin[dof * 6 + 5] = cdof[dof * 6 + 5] + ax * dy - ay * dx

    # --- cvel at body origin ---
    # v_origin = v_COM + w x (xpos - xipos) = xvel + xangvel x (xpos - xipos)
    var cvel_origin = InlineArray[Scalar[DTYPE], BODY6_SIZE](uninitialized=True)
    for i in range(BODY6_SIZE):
        cvel_origin[i] = Scalar[DTYPE](0)

    for b in range(NBODY):
        var wx = data.xangvel[b * 3 + 0]
        var wy = data.xangvel[b * 3 + 1]
        var wz = data.xangvel[b * 3 + 2]

        var dx = data.xpos[b * 3 + 0] - data.xipos[b * 3 + 0]
        var dy = data.xpos[b * 3 + 1] - data.xipos[b * 3 + 1]
        var dz = data.xpos[b * 3 + 2] - data.xipos[b * 3 + 2]

        cvel_origin[b * 6 + 0] = wx
        cvel_origin[b * 6 + 1] = wy
        cvel_origin[b * 6 + 2] = wz
        cvel_origin[b * 6 + 3] = data.xvel[b * 3 + 0] + wy * dz - wz * dy
        cvel_origin[b * 6 + 4] = data.xvel[b * 3 + 1] + wz * dx - wx * dz
        cvel_origin[b * 6 + 5] = data.xvel[b * 3 + 2] + wx * dy - wy * dx

    # --- cinert at body origin (10-element spatial inertia) ---
    # I_origin = I_COM + m*(c·c*I - c*c^T)  (parallel axis theorem)
    # h = m * c  (first moment)
    var cinert = InlineArray[Scalar[DTYPE], CINERT_SIZE](uninitialized=True)
    for i in range(CINERT_SIZE):
        cinert[i] = Scalar[DTYPE](0)

    for b in range(NBODY):
        var mass = model.body_mass[b]

        # COM offset from body origin (in world frame)
        var cx = data.xipos[b * 3 + 0] - data.xpos[b * 3 + 0]
        var cy = data.xipos[b * 3 + 1] - data.xpos[b * 3 + 1]
        var cz = data.xipos[b * 3 + 2] - data.xpos[b * 3 + 2]

        # World-frame inertia at COM (recompute from body data)
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

        # Rotation matrix from quaternion
        var r00 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qy * qy + qz * qz)
        var r10 = Scalar[DTYPE](2) * (qx * qy + qw * qz)
        var r20 = Scalar[DTYPE](2) * (qx * qz - qw * qy)
        var r01 = Scalar[DTYPE](2) * (qx * qy - qw * qz)
        var r11 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qx * qx + qz * qz)
        var r21 = Scalar[DTYPE](2) * (qy * qz + qw * qx)
        var r02 = Scalar[DTYPE](2) * (qx * qz + qw * qy)
        var r12 = Scalar[DTYPE](2) * (qy * qz - qw * qx)
        var r22 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qx * qx + qy * qy)

        # I_world at COM: R @ diag(Ixx,Iyy,Izz) @ R^T
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

        # Parallel axis: I_origin = I_COM + m*(c·c*I - c*c^T)
        var c_sq = cx * cx + cy * cy + cz * cz
        cinert[b * 10 + 0] = Iw_xx + mass * (c_sq - cx * cx)  # Ixx
        cinert[b * 10 + 1] = Iw_yy + mass * (c_sq - cy * cy)  # Iyy
        cinert[b * 10 + 2] = Iw_zz + mass * (c_sq - cz * cz)  # Izz
        cinert[b * 10 + 3] = Iw_xy - mass * cx * cy  # Ixy
        cinert[b * 10 + 4] = Iw_xz - mass * cx * cz  # Ixz
        cinert[b * 10 + 5] = Iw_yz - mass * cy * cz  # Iyz
        cinert[b * 10 + 6] = mass * cx  # h_x = m*cx
        cinert[b * 10 + 7] = mass * cy  # h_y = m*cy
        cinert[b * 10 + 8] = mass * cz  # h_z = m*cz
        cinert[b * 10 + 9] = mass

    # --- Build body_dofadr and body_dofnum lookup ---
    var body_dofadr = InlineArray[Int, NBODY](uninitialized=True)
    var body_dofnum = InlineArray[Int, NBODY](uninitialized=True)
    for b in range(NBODY):
        body_dofadr[b] = -1
        body_dofnum[b] = 0

    for j in range(model.num_joints):
        var joint = model.joints[j]
        var body = joint.body_id
        var dof_adr = joint.dof_adr
        var num_dof = 1
        if joint.jnt_type == JNT_FREE:
            num_dof = 6
        elif joint.jnt_type == JNT_BALL:
            num_dof = 3

        if body_dofadr[body] < 0:
            body_dofadr[body] = dof_adr
        body_dofnum[body] = body_dofnum[body] + num_dof

    # --- Build dof_bodyid lookup ---
    comptime DOFBODY_SIZE = _max_one[NV]()
    var dof_bodyid = InlineArray[Int, DOFBODY_SIZE](uninitialized=True)
    for i in range(NV):
        dof_bodyid[i] = 0
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var dof_adr = joint.dof_adr
        var num_dof = 1
        if joint.jnt_type == JNT_FREE:
            num_dof = 6
        elif joint.jnt_type == JNT_BALL:
            num_dof = 3
        for d in range(num_dof):
            dof_bodyid[dof_adr + d] = joint.body_id

    # --- Precompute cdof_dot at body origin ---
    # cdof_dot[d] = crossMotion(cvel_origin[parent_of_body_of_d], cdof_origin[d])
    var cdof_dot = InlineArray[Scalar[DTYPE], CDOF_O_SIZE](uninitialized=True)
    for i in range(CDOF_O_SIZE):
        cdof_dot[i] = Scalar[DTYPE](0)

    for j in range(model.num_joints):
        var joint = model.joints[j]
        var body = joint.body_id
        var parent = model.body_parent[body]
        var dof_adr = joint.dof_adr
        var num_dof = 1
        if joint.jnt_type == JNT_FREE:
            num_dof = 6
        elif joint.jnt_type == JNT_BALL:
            num_dof = 3

        # Get parent velocity at body origin (or zero for world)
        var wp_x: Scalar[DTYPE] = 0
        var wp_y: Scalar[DTYPE] = 0
        var wp_z: Scalar[DTYPE] = 0
        var vp_x: Scalar[DTYPE] = 0
        var vp_y: Scalar[DTYPE] = 0
        var vp_z: Scalar[DTYPE] = 0
        if parent >= 0:
            wp_x = cvel_origin[parent * 6 + 0]
            wp_y = cvel_origin[parent * 6 + 1]
            wp_z = cvel_origin[parent * 6 + 2]
            vp_x = cvel_origin[parent * 6 + 3]
            vp_y = cvel_origin[parent * 6 + 4]
            vp_z = cvel_origin[parent * 6 + 5]

        for d in range(num_dof):
            var dof = dof_adr + d
            var s_ang_x = cdof_origin[dof * 6 + 0]
            var s_ang_y = cdof_origin[dof * 6 + 1]
            var s_ang_z = cdof_origin[dof * 6 + 2]
            var s_lin_x = cdof_origin[dof * 6 + 3]
            var s_lin_y = cdof_origin[dof * 6 + 4]
            var s_lin_z = cdof_origin[dof * 6 + 5]

            # crossMotion(cvel_parent, cdof):
            # cdot_ang = w_p x s_ang
            cdof_dot[dof * 6 + 0] = wp_y * s_ang_z - wp_z * s_ang_y
            cdof_dot[dof * 6 + 1] = wp_z * s_ang_x - wp_x * s_ang_z
            cdof_dot[dof * 6 + 2] = wp_x * s_ang_y - wp_y * s_ang_x
            # cdot_lin = w_p x s_lin + v_p x s_ang
            cdof_dot[dof * 6 + 3] = (wp_y * s_lin_z - wp_z * s_lin_y) + (
                vp_y * s_ang_z - vp_z * s_ang_y
            )
            cdof_dot[dof * 6 + 4] = (wp_z * s_lin_x - wp_x * s_lin_z) + (
                vp_z * s_ang_x - vp_x * s_ang_z
            )
            cdof_dot[dof * 6 + 5] = (wp_x * s_lin_y - wp_y * s_lin_x) + (
                vp_x * s_ang_y - vp_y * s_ang_x
            )

    # =========================================================================
    # Step 1: Compute Dcvel and Dcdofdot
    # (MuJoCo: mjd_comVel_vel_dense)
    #
    # Dcvel[body, comp, k] = d(cvel_origin[body][comp]) / d(qvel[k])
    # Layout: Dcvel[body*6*NV + comp*NV + k]
    #
    # Dcdofdot[dof, comp, k] = d(cdof_dot[dof][comp]) / d(qvel[k])
    # Layout: Dcdofdot[dof*6*NV + comp*NV + k]
    # =========================================================================
    var Dcvel = InlineArray[Scalar[DTYPE], DCVEL_SIZE](uninitialized=True)
    for i in range(DCVEL_SIZE):
        Dcvel[i] = Scalar[DTYPE](0)

    var Dcdofdot = InlineArray[Scalar[DTYPE], DCDOFDOT_SIZE](uninitialized=True)
    for i in range(DCDOFDOT_SIZE):
        Dcdofdot[i] = Scalar[DTYPE](0)

    # Temporary 6x6 matrices
    var mat = InlineArray[Scalar[DTYPE], 36](uninitialized=True)
    # Forward pass over bodies
    for b in range(NBODY):
        var parent = model.body_parent[b]
        if parent < 0:
            # Root body: Dcvel = 0 (no qvel dependence for world)
            continue

        # Dcvel[body] = Dcvel[parent]  (simple copy in body-origin convention)
        for idx in range(6 * NV):
            Dcvel[b * 6 * NV + idx] = Dcvel[parent * 6 * NV + idx]

        # Process DOFs of this body
        if body_dofadr[b] < 0:
            continue  # No DOFs for this body

        var dof_start = body_dofadr[b]
        var dof_end = dof_start + body_dofnum[b]

        for j_dof in range(dof_start, dof_end):
            # Get joint type for this DOF
            var jnt_type = JNT_HINGE  # default
            for jj in range(model.num_joints):
                var joint_jj = model.joints[jj]
                if joint_jj.body_id == b:
                    jnt_type = joint_jj.jnt_type
                    break

            if jnt_type == JNT_FREE and j_dof == dof_start:
                # FREE joint: first 3 DOFs are translation
                # Dcdofdot = 0 for translation DOFs (already zeroed)

                # Dcvel += cdof_origin * D(qvel) for translation DOFs
                for k in range(6):
                    Dcvel[b * 6 * NV + k * NV + j_dof + 0] += cdof_origin[
                        (j_dof + 0) * 6 + k
                    ]
                    Dcvel[b * 6 * NV + k * NV + j_dof + 1] += cdof_origin[
                        (j_dof + 1) * 6 + k
                    ]
                    Dcvel[b * 6 * NV + k * NV + j_dof + 2] += cdof_origin[
                        (j_dof + 2) * 6 + k
                    ]

                # Now handle rotation DOFs (j_dof+3, j_dof+4, j_dof+5)
                for rot_d in range(3):
                    var dof_rot = j_dof + 3 + rot_d
                    # Dcdofdot[dof_rot] = crossMotion_vel(cdof_origin[dof_rot]) @ Dcvel[body]
                    var cdof_v = InlineArray[Scalar[DTYPE], 6](
                        uninitialized=True
                    )
                    for kk in range(6):
                        cdof_v[kk] = cdof_origin[dof_rot * 6 + kk]
                    _mjd_crossMotion_vel(mat, cdof_v)

                    # Dcdofdot[dof_rot] = mat @ Dcvel[body]  (6xNV = 6x6 @ 6xNV)
                    for ii in range(6):
                        for kk in range(NV):
                            var s: Scalar[DTYPE] = 0
                            for jj in range(6):
                                s += (
                                    mat[ii * 6 + jj]
                                    * Dcvel[b * 6 * NV + jj * NV + kk]
                                )
                            Dcdofdot[dof_rot * 6 * NV + ii * NV + kk] = s

                    # Dcvel += cdof_origin * D(qvel)
                    for kk in range(6):
                        Dcvel[b * 6 * NV + kk * NV + dof_rot] += cdof_origin[
                            dof_rot * 6 + kk
                        ]

                # Skip the 5 extra DOFs already processed (loop will advance j_dof)
                # But since we process all DOFs of the body in this loop iteration,
                # we need to break to avoid double processing
                break

            elif jnt_type == JNT_BALL and j_dof == dof_start:
                # BALL joint: 3 rotation DOFs
                for rot_d in range(3):
                    var dof_rot = j_dof + rot_d
                    # Dcdofdot[dof_rot] = crossMotion_vel(cdof_origin[dof_rot]) @ Dcvel[body]
                    var cdof_v = InlineArray[Scalar[DTYPE], 6](
                        uninitialized=True
                    )
                    for kk in range(6):
                        cdof_v[kk] = cdof_origin[dof_rot * 6 + kk]
                    _mjd_crossMotion_vel(mat, cdof_v)

                    for ii in range(6):
                        for kk in range(NV):
                            var s: Scalar[DTYPE] = 0
                            for jj in range(6):
                                s += (
                                    mat[ii * 6 + jj]
                                    * Dcvel[b * 6 * NV + jj * NV + kk]
                                )
                            Dcdofdot[dof_rot * 6 * NV + ii * NV + kk] = s

                    # Dcvel += cdof_origin * D(qvel)
                    for kk in range(6):
                        Dcvel[b * 6 * NV + kk * NV + dof_rot] += cdof_origin[
                            dof_rot * 6 + kk
                        ]
                break

            else:
                # HINGE or SLIDE: single DOF
                # Dcdofdot = crossMotion_vel(cdof_origin) @ Dcvel[body]
                var cdof_v = InlineArray[Scalar[DTYPE], 6](uninitialized=True)
                for kk in range(6):
                    cdof_v[kk] = cdof_origin[j_dof * 6 + kk]
                _mjd_crossMotion_vel(mat, cdof_v)

                for ii in range(6):
                    for kk in range(NV):
                        var s: Scalar[DTYPE] = 0
                        for jj in range(6):
                            s += (
                                mat[ii * 6 + jj]
                                * Dcvel[b * 6 * NV + jj * NV + kk]
                            )
                        Dcdofdot[j_dof * 6 * NV + ii * NV + kk] = s

                # Dcvel += cdof_origin * D(qvel)
                for kk in range(6):
                    Dcvel[b * 6 * NV + kk * NV + j_dof] += cdof_origin[
                        j_dof * 6 + kk
                    ]

    # =========================================================================
    # Step 2: Forward pass - compute Dcacc and Dcfrcbody
    # (MuJoCo: mjd_rne_vel_dense, lines 399-434)
    # =========================================================================
    var Dcacc = InlineArray[Scalar[DTYPE], DCVEL_SIZE](uninitialized=True)
    for i in range(DCVEL_SIZE):
        Dcacc[i] = Scalar[DTYPE](0)

    var Dcfrcbody = InlineArray[Scalar[DTYPE], DCVEL_SIZE](uninitialized=True)
    for i in range(DCVEL_SIZE):
        Dcfrcbody[i] = Scalar[DTYPE](0)

    # Temporary arrays for 6x6 @ 6xNV products
    var dmul = InlineArray[Scalar[DTYPE], 36](uninitialized=True)
    var mat1 = InlineArray[Scalar[DTYPE], 36](uninitialized=True)
    var mat2 = InlineArray[Scalar[DTYPE], 36](uninitialized=True)
    var tmp6 = InlineArray[Scalar[DTYPE], 6](uninitialized=True)

    for b in range(1, NBODY):
        var parent = model.body_parent[b]

        # Dcacc[b] = Dcacc[parent]
        for idx in range(6 * NV):
            Dcacc[b * 6 * NV + idx] = Dcacc[parent * 6 * NV + idx]

        # Dcacc += D(cdof_dot * qvel)
        if body_dofadr[b] >= 0:
            var dof_start = body_dofadr[b]
            var dof_end = dof_start + body_dofnum[b]
            for j_dof in range(dof_start, dof_end):
                # Dcacc += cdof_dot * D(qvel): column j_dof gets cdof_dot[j_dof]
                for k in range(6):
                    Dcacc[b * 6 * NV + k * NV + j_dof] += cdof_dot[
                        j_dof * 6 + k
                    ]

                # Dcacc += D(cdof_dot) * qvel[j_dof]
                var qvel_j = data.qvel[j_dof]
                for idx in range(6 * NV):
                    Dcacc[b * 6 * NV + idx] += (
                        Dcdofdot[j_dof * 6 * NV + idx] * qvel_j
                    )

        # --- Dcfrcbody = D(cinert * cacc + cvel x* (cinert * cvel)) ---

        # Extract per-body cinert
        var ci = InlineArray[Scalar[DTYPE], 10](uninitialized=True)
        for k in range(10):
            ci[k] = cinert[b * 10 + k]

        # dmul = D(mulInertVec) / D(vel)  (= the inertia matrix itself, 6x6)
        _mjd_mulInertVec_vel(dmul, ci)

        # Dcfrcbody[b] = dmul @ Dcacc[b]  (6xNV = 6x6 @ 6xNV)
        for ii in range(6):
            for kk in range(NV):
                var s: Scalar[DTYPE] = 0
                for jj in range(6):
                    s += dmul[ii * 6 + jj] * Dcacc[b * 6 * NV + jj * NV + kk]
                Dcfrcbody[b * 6 * NV + ii * NV + kk] = s

        # mat = D(crossForce)/D(vel) + D(crossForce)/D(frc) @ D(mulInertVec)/D(vel)
        # where the crossForce is: cvel x* (cinert * cvel)

        # First: tmp6 = cinert * cvel_origin[b]
        var cv = InlineArray[Scalar[DTYPE], 6](uninitialized=True)
        for k in range(6):
            cv[k] = cvel_origin[b * 6 + k]
        _mulInertVec(tmp6, ci, cv)

        # mat = D(crossForce(cvel, tmp6)) / D(cvel)
        _mjd_crossForce_vel(mat, tmp6)

        # mat1 = D(crossForce(cvel, frc)) / D(frc)
        _mjd_crossForce_frc(mat1, cv)

        # mat2 = mat1 @ dmul  (D(crossForce)/D(frc) @ D(mulInertVec)/D(vel))
        _matmul_6x6_x_6x6(mat2, mat1, dmul)

        # mat += mat2
        for k in range(36):
            mat[k] += mat2[k]

        # Dcfrcbody[b] += mat @ Dcvel[b]  (use body 0 as temp in MuJoCo,
        # but we just accumulate directly)
        for ii in range(6):
            for kk in range(NV):
                var s: Scalar[DTYPE] = 0
                for jj in range(6):
                    s += mat[ii * 6 + jj] * Dcvel[b * 6 * NV + jj * NV + kk]
                Dcfrcbody[b * 6 * NV + ii * NV + kk] += s

    # =========================================================================
    # Step 3: Backward pass - accumulate Dcfrcbody to parents
    # Simple addition (body-origin convention)
    # =========================================================================
    for b in range(NBODY - 1, 0, -1):
        var parent = model.body_parent[b]
        if parent >= 0:
            for idx in range(6 * NV):
                Dcfrcbody[parent * 6 * NV + idx] += Dcfrcbody[b * 6 * NV + idx]

    # =========================================================================
    # Step 4: Project to joint space
    # qDeriv[i, k] -= Σ_comp cdof_origin[i][comp] * Dcfrcbody[body_of_i, comp, k]
    #
    # Note: We use cdof_origin (not cdof at COM) because the forces are
    # at body origins.
    # =========================================================================
    for i in range(NV):
        var body_i = dof_bodyid[i]
        for k in range(NV):
            var s: Scalar[DTYPE] = 0
            for comp in range(6):
                s += (
                    cdof_origin[i * 6 + comp]
                    * Dcfrcbody[body_i * 6 * NV + comp * NV + k]
                )
            qDeriv[i * NV + k] -= s


# =============================================================================
# GPU version: RNE velocity derivative
# =============================================================================


@always_inline
fn compute_rne_vel_derivative_gpu[
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
    NGEOM: Int = 0,
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    workspace: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ],
    implicit_base: Int,
):
    """Compute d(qfrc_bias)/d(qvel) on GPU, subtract from qDeriv in workspace.

    Same algorithm as the CPU version but reads from state/model LayoutTensors
    and uses workspace arrays for large derivative matrices.

    Args:
        env: Environment index.
        state: State buffer.
        model: Model buffer.
        workspace: Workspace buffer.
        implicit_base: Offset to the implicit extra section in workspace.
    """
    from ..kinematics.quat_math import gpu_quat_mul

    # Workspace offsets for implicit extra arrays
    var qd_off = ws_implicit_qderiv_offset(implicit_base)
    var co_off = ws_implicit_cdof_origin_offset[NV](implicit_base)
    var cv_off = ws_implicit_cvel_origin_offset[NV](implicit_base)
    var ci_off = ws_implicit_cinert_offset[NV, NBODY](implicit_base)
    var cd_off = ws_implicit_cdof_dot_offset[NV, NBODY](implicit_base)
    var dcv_off = ws_implicit_dcvel_offset[NV, NBODY](implicit_base)
    var dcd_off = ws_implicit_dcdofdot_offset[NV, NBODY](implicit_base)
    var dca_off = ws_implicit_dcacc_offset[NV, NBODY](implicit_base)
    var dcf_off = ws_implicit_dcfrcbody_offset[NV, NBODY](implicit_base)

    # State offsets
    var xp_off = xpos_offset[NQ, NV, NBODY]()
    var xq_off = xquat_offset[NQ, NV, NBODY]()
    var xi_off = xipos_offset[NQ, NV, NBODY]()
    var xv_off = xvel_offset[NQ, NV, NBODY]()
    var xa_off = xangvel_offset[NQ, NV, NBODY]()
    var qv_off = qvel_offset[NQ, NV]()

    # cdof offset in workspace (from compute_cdof_gpu)
    comptime cdof_idx = ws_cdof_offset()

    var model_meta_off = model_metadata_offset[NBODY, NJOINT]()

    # =========================================================================
    # Step 0: Precompute body-origin quantities
    # =========================================================================

    # --- cdof at body origin ---
    # Copy cdof from workspace, then shift from COM to body origin
    for d in range(NV):
        for k in range(6):
            workspace[env, co_off + d * 6 + k] = workspace[
                env, cdof_idx + d * 6 + k
            ]

    for j in range(NJOINT):
        var joint_off = model_joint_offset[NBODY](j)
        var jnt_type = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
        )
        var body = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_BODY_ID])
        )
        var dof_adr = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR])
        )

        # Offset from body origin to body COM (world frame)
        var cx = rebind[Scalar[DTYPE]](
            state[env, xi_off + body * 3 + 0]
        ) - rebind[Scalar[DTYPE]](state[env, xp_off + body * 3 + 0])
        var cy = rebind[Scalar[DTYPE]](
            state[env, xi_off + body * 3 + 1]
        ) - rebind[Scalar[DTYPE]](state[env, xp_off + body * 3 + 1])
        var cz = rebind[Scalar[DTYPE]](
            state[env, xi_off + body * 3 + 2]
        ) - rebind[Scalar[DTYPE]](state[env, xp_off + body * 3 + 2])
        var dx = -cx
        var dy = -cy
        var dz = -cz

        var num_dof = 1
        if jnt_type == JNT_FREE:
            num_dof = 6
        elif jnt_type == JNT_BALL:
            num_dof = 3

        for d in range(num_dof):
            var dof = dof_adr + d
            var ax = rebind[Scalar[DTYPE]](
                workspace[env, cdof_idx + dof * 6 + 0]
            )
            var ay = rebind[Scalar[DTYPE]](
                workspace[env, cdof_idx + dof * 6 + 1]
            )
            var az = rebind[Scalar[DTYPE]](
                workspace[env, cdof_idx + dof * 6 + 2]
            )
            workspace[env, co_off + dof * 6 + 3] = (
                rebind[Scalar[DTYPE]](workspace[env, cdof_idx + dof * 6 + 3])
                + ay * dz
                - az * dy
            )
            workspace[env, co_off + dof * 6 + 4] = (
                rebind[Scalar[DTYPE]](workspace[env, cdof_idx + dof * 6 + 4])
                + az * dx
                - ax * dz
            )
            workspace[env, co_off + dof * 6 + 5] = (
                rebind[Scalar[DTYPE]](workspace[env, cdof_idx + dof * 6 + 5])
                + ax * dy
                - ay * dx
            )

    # --- cvel at body origin ---
    for b in range(NBODY):
        var wx = rebind[Scalar[DTYPE]](state[env, xa_off + b * 3 + 0])
        var wy = rebind[Scalar[DTYPE]](state[env, xa_off + b * 3 + 1])
        var wz = rebind[Scalar[DTYPE]](state[env, xa_off + b * 3 + 2])
        var ddx = rebind[Scalar[DTYPE]](
            state[env, xp_off + b * 3 + 0]
        ) - rebind[Scalar[DTYPE]](state[env, xi_off + b * 3 + 0])
        var ddy = rebind[Scalar[DTYPE]](
            state[env, xp_off + b * 3 + 1]
        ) - rebind[Scalar[DTYPE]](state[env, xi_off + b * 3 + 1])
        var ddz = rebind[Scalar[DTYPE]](
            state[env, xp_off + b * 3 + 2]
        ) - rebind[Scalar[DTYPE]](state[env, xi_off + b * 3 + 2])

        workspace[env, cv_off + b * 6 + 0] = wx
        workspace[env, cv_off + b * 6 + 1] = wy
        workspace[env, cv_off + b * 6 + 2] = wz
        workspace[env, cv_off + b * 6 + 3] = (
            rebind[Scalar[DTYPE]](state[env, xv_off + b * 3 + 0])
            + wy * ddz
            - wz * ddy
        )
        workspace[env, cv_off + b * 6 + 4] = (
            rebind[Scalar[DTYPE]](state[env, xv_off + b * 3 + 1])
            + wz * ddx
            - wx * ddz
        )
        workspace[env, cv_off + b * 6 + 5] = (
            rebind[Scalar[DTYPE]](state[env, xv_off + b * 3 + 2])
            + wx * ddy
            - wy * ddx
        )

    # --- cinert at body origin (10-element spatial inertia) ---
    for b in range(NBODY):
        var body_off = model_body_offset(b)
        var mass = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_MASS])
        var cx = rebind[Scalar[DTYPE]](state[env, xi_off + b * 3 + 0]) - rebind[
            Scalar[DTYPE]
        ](state[env, xp_off + b * 3 + 0])
        var cy = rebind[Scalar[DTYPE]](state[env, xi_off + b * 3 + 1]) - rebind[
            Scalar[DTYPE]
        ](state[env, xp_off + b * 3 + 1])
        var cz = rebind[Scalar[DTYPE]](state[env, xi_off + b * 3 + 2]) - rebind[
            Scalar[DTYPE]
        ](state[env, xp_off + b * 3 + 2])

        var Ixx_local = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IXX])
        var Iyy_local = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IYY])
        var Izz_local = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IZZ])

        # Compose xquat with body_iquat
        var bqx = rebind[Scalar[DTYPE]](state[env, xq_off + b * 4 + 0])
        var bqy = rebind[Scalar[DTYPE]](state[env, xq_off + b * 4 + 1])
        var bqz = rebind[Scalar[DTYPE]](state[env, xq_off + b * 4 + 2])
        var bqw = rebind[Scalar[DTYPE]](state[env, xq_off + b * 4 + 3])
        var iqx = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IQUAT_X])
        var iqy = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IQUAT_Y])
        var iqz = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IQUAT_Z])
        var iqw = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IQUAT_W])
        var iq = gpu_quat_mul(bqx, bqy, bqz, bqw, iqx, iqy, iqz, iqw)
        var qx = iq[0]
        var qy = iq[1]
        var qz = iq[2]
        var qw = iq[3]

        # Rotation matrix from quaternion
        var r00 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qy * qy + qz * qz)
        var r10 = Scalar[DTYPE](2) * (qx * qy + qw * qz)
        var r20 = Scalar[DTYPE](2) * (qx * qz - qw * qy)
        var r01 = Scalar[DTYPE](2) * (qx * qy - qw * qz)
        var r11 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qx * qx + qz * qz)
        var r21 = Scalar[DTYPE](2) * (qy * qz + qw * qx)
        var r02 = Scalar[DTYPE](2) * (qx * qz + qw * qy)
        var r12 = Scalar[DTYPE](2) * (qy * qz - qw * qx)
        var r22 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qx * qx + qy * qy)

        # I_world at COM
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

        # Parallel axis theorem
        var c_sq = cx * cx + cy * cy + cz * cz
        workspace[env, ci_off + b * 10 + 0] = Iw_xx + mass * (c_sq - cx * cx)
        workspace[env, ci_off + b * 10 + 1] = Iw_yy + mass * (c_sq - cy * cy)
        workspace[env, ci_off + b * 10 + 2] = Iw_zz + mass * (c_sq - cz * cz)
        workspace[env, ci_off + b * 10 + 3] = Iw_xy - mass * cx * cy
        workspace[env, ci_off + b * 10 + 4] = Iw_xz - mass * cx * cz
        workspace[env, ci_off + b * 10 + 5] = Iw_yz - mass * cy * cz
        workspace[env, ci_off + b * 10 + 6] = mass * cx
        workspace[env, ci_off + b * 10 + 7] = mass * cy
        workspace[env, ci_off + b * 10 + 8] = mass * cz
        workspace[env, ci_off + b * 10 + 9] = mass

    # --- Build body_dofadr, body_dofnum, dof_bodyid lookups ---
    comptime V_SIZE = _max_one[NV]()
    var body_dofadr = InlineArray[Int, NBODY](uninitialized=True)
    var body_dofnum = InlineArray[Int, NBODY](uninitialized=True)
    for b in range(NBODY):
        body_dofadr[b] = -1
        body_dofnum[b] = 0

    var dof_bodyid = InlineArray[Int, V_SIZE](uninitialized=True)
    for i in range(NV):
        dof_bodyid[i] = 0

    for j in range(NJOINT):
        var joint_off = model_joint_offset[NBODY](j)
        var jnt_type = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
        )
        var body = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_BODY_ID])
        )
        var dof_adr = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR])
        )
        var num_dof = 1
        if jnt_type == JNT_FREE:
            num_dof = 6
        elif jnt_type == JNT_BALL:
            num_dof = 3
        if body_dofadr[body] < 0:
            body_dofadr[body] = dof_adr
        body_dofnum[body] = body_dofnum[body] + num_dof
        for d in range(num_dof):
            dof_bodyid[dof_adr + d] = body

    # --- Precompute cdof_dot at body origin ---
    for i in range(NV * 6):
        workspace[env, cd_off + i] = Scalar[DTYPE](0)

    for j in range(NJOINT):
        var joint_off = model_joint_offset[NBODY](j)
        var jnt_type = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
        )
        var body = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_BODY_ID])
        )
        var dof_adr = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR])
        )
        var body_off = model_body_offset(body)
        var parent = Int(
            rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_PARENT])
        )
        var num_dof = 1
        if jnt_type == JNT_FREE:
            num_dof = 6
        elif jnt_type == JNT_BALL:
            num_dof = 3

        # Get parent velocity at body origin (or zero for world)
        var wp_x: Scalar[DTYPE] = 0
        var wp_y: Scalar[DTYPE] = 0
        var wp_z: Scalar[DTYPE] = 0
        var vp_x: Scalar[DTYPE] = 0
        var vp_y: Scalar[DTYPE] = 0
        var vp_z: Scalar[DTYPE] = 0
        if parent >= 0:
            wp_x = rebind[Scalar[DTYPE]](
                workspace[env, cv_off + parent * 6 + 0]
            )
            wp_y = rebind[Scalar[DTYPE]](
                workspace[env, cv_off + parent * 6 + 1]
            )
            wp_z = rebind[Scalar[DTYPE]](
                workspace[env, cv_off + parent * 6 + 2]
            )
            vp_x = rebind[Scalar[DTYPE]](
                workspace[env, cv_off + parent * 6 + 3]
            )
            vp_y = rebind[Scalar[DTYPE]](
                workspace[env, cv_off + parent * 6 + 4]
            )
            vp_z = rebind[Scalar[DTYPE]](
                workspace[env, cv_off + parent * 6 + 5]
            )

        for d in range(num_dof):
            var dof = dof_adr + d
            var s_ang_x = rebind[Scalar[DTYPE]](
                workspace[env, co_off + dof * 6 + 0]
            )
            var s_ang_y = rebind[Scalar[DTYPE]](
                workspace[env, co_off + dof * 6 + 1]
            )
            var s_ang_z = rebind[Scalar[DTYPE]](
                workspace[env, co_off + dof * 6 + 2]
            )
            var s_lin_x = rebind[Scalar[DTYPE]](
                workspace[env, co_off + dof * 6 + 3]
            )
            var s_lin_y = rebind[Scalar[DTYPE]](
                workspace[env, co_off + dof * 6 + 4]
            )
            var s_lin_z = rebind[Scalar[DTYPE]](
                workspace[env, co_off + dof * 6 + 5]
            )

            workspace[env, cd_off + dof * 6 + 0] = (
                wp_y * s_ang_z - wp_z * s_ang_y
            )
            workspace[env, cd_off + dof * 6 + 1] = (
                wp_z * s_ang_x - wp_x * s_ang_z
            )
            workspace[env, cd_off + dof * 6 + 2] = (
                wp_x * s_ang_y - wp_y * s_ang_x
            )
            workspace[env, cd_off + dof * 6 + 3] = (
                wp_y * s_lin_z - wp_z * s_lin_y
            ) + (vp_y * s_ang_z - vp_z * s_ang_y)
            workspace[env, cd_off + dof * 6 + 4] = (
                wp_z * s_lin_x - wp_x * s_lin_z
            ) + (vp_z * s_ang_x - vp_x * s_ang_z)
            workspace[env, cd_off + dof * 6 + 5] = (
                wp_x * s_lin_y - wp_y * s_lin_x
            ) + (vp_x * s_ang_y - vp_y * s_ang_x)

    # =========================================================================
    # Step 1: Compute Dcvel and Dcdofdot
    # =========================================================================

    # Zero Dcvel and Dcdofdot
    for i in range(NBODY * 6 * NV):
        workspace[env, dcv_off + i] = Scalar[DTYPE](0)
    for i in range(NV * 6 * NV):
        workspace[env, dcd_off + i] = Scalar[DTYPE](0)

    var mat = InlineArray[Scalar[DTYPE], 36](uninitialized=True)

    for b in range(NBODY):
        var body_off = model_body_offset(b)
        var parent = Int(
            rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_PARENT])
        )
        if parent < 0:
            continue

        # Dcvel[body] = Dcvel[parent]
        for idx in range(6 * NV):
            workspace[env, dcv_off + b * 6 * NV + idx] = workspace[
                env, dcv_off + parent * 6 * NV + idx
            ]

        if body_dofadr[b] < 0:
            continue

        var dof_start = body_dofadr[b]
        var dof_end = dof_start + body_dofnum[b]

        for j_dof in range(dof_start, dof_end):
            # Get joint type for this body
            var jnt_type_b = JNT_HINGE
            for jj in range(NJOINT):
                var jj_off = model_joint_offset[NBODY](jj)
                var jj_body = Int(
                    rebind[Scalar[DTYPE]](model[0, jj_off + JOINT_IDX_BODY_ID])
                )
                if jj_body == b:
                    jnt_type_b = Int(
                        rebind[Scalar[DTYPE]](model[0, jj_off + JOINT_IDX_TYPE])
                    )
                    break

            if jnt_type_b == JNT_FREE and j_dof == dof_start:
                # Translation DOFs: Dcvel += cdof_origin for each
                for k in range(6):
                    for td in range(3):
                        workspace[
                            env, dcv_off + b * 6 * NV + k * NV + j_dof + td
                        ] = rebind[Scalar[DTYPE]](
                            workspace[
                                env, dcv_off + b * 6 * NV + k * NV + j_dof + td
                            ]
                        ) + rebind[
                            Scalar[DTYPE]
                        ](
                            workspace[env, co_off + (j_dof + td) * 6 + k]
                        )

                # Rotation DOFs (j_dof+3..j_dof+5)
                for rot_d in range(3):
                    var dof_rot = j_dof + 3 + rot_d
                    var cdof_v = InlineArray[Scalar[DTYPE], 6](
                        uninitialized=True
                    )
                    for kk in range(6):
                        cdof_v[kk] = rebind[Scalar[DTYPE]](
                            workspace[env, co_off + dof_rot * 6 + kk]
                        )
                    _mjd_crossMotion_vel(mat, cdof_v)

                    for ii in range(6):
                        for kk in range(NV):
                            var s: Scalar[DTYPE] = 0
                            for jj in range(6):
                                s += mat[ii * 6 + jj] * rebind[Scalar[DTYPE]](
                                    workspace[
                                        env, dcv_off + b * 6 * NV + jj * NV + kk
                                    ]
                                )
                            workspace[
                                env, dcd_off + dof_rot * 6 * NV + ii * NV + kk
                            ] = s

                    for kk in range(6):
                        workspace[
                            env, dcv_off + b * 6 * NV + kk * NV + dof_rot
                        ] = rebind[Scalar[DTYPE]](
                            workspace[
                                env, dcv_off + b * 6 * NV + kk * NV + dof_rot
                            ]
                        ) + rebind[
                            Scalar[DTYPE]
                        ](
                            workspace[env, co_off + dof_rot * 6 + kk]
                        )
                break

            elif jnt_type_b == JNT_BALL and j_dof == dof_start:
                for rot_d in range(3):
                    var dof_rot = j_dof + rot_d
                    var cdof_v = InlineArray[Scalar[DTYPE], 6](
                        uninitialized=True
                    )
                    for kk in range(6):
                        cdof_v[kk] = rebind[Scalar[DTYPE]](
                            workspace[env, co_off + dof_rot * 6 + kk]
                        )
                    _mjd_crossMotion_vel(mat, cdof_v)

                    for ii in range(6):
                        for kk in range(NV):
                            var s: Scalar[DTYPE] = 0
                            for jj in range(6):
                                s += mat[ii * 6 + jj] * rebind[Scalar[DTYPE]](
                                    workspace[
                                        env, dcv_off + b * 6 * NV + jj * NV + kk
                                    ]
                                )
                            workspace[
                                env, dcd_off + dof_rot * 6 * NV + ii * NV + kk
                            ] = s

                    for kk in range(6):
                        workspace[
                            env, dcv_off + b * 6 * NV + kk * NV + dof_rot
                        ] = rebind[Scalar[DTYPE]](
                            workspace[
                                env, dcv_off + b * 6 * NV + kk * NV + dof_rot
                            ]
                        ) + rebind[
                            Scalar[DTYPE]
                        ](
                            workspace[env, co_off + dof_rot * 6 + kk]
                        )
                break

            else:
                # HINGE or SLIDE: single DOF
                var cdof_v = InlineArray[Scalar[DTYPE], 6](uninitialized=True)
                for kk in range(6):
                    cdof_v[kk] = rebind[Scalar[DTYPE]](
                        workspace[env, co_off + j_dof * 6 + kk]
                    )
                _mjd_crossMotion_vel(mat, cdof_v)

                for ii in range(6):
                    for kk in range(NV):
                        var s: Scalar[DTYPE] = 0
                        for jj in range(6):
                            s += mat[ii * 6 + jj] * rebind[Scalar[DTYPE]](
                                workspace[
                                    env, dcv_off + b * 6 * NV + jj * NV + kk
                                ]
                            )
                        workspace[
                            env, dcd_off + j_dof * 6 * NV + ii * NV + kk
                        ] = s

                for kk in range(6):
                    workspace[
                        env, dcv_off + b * 6 * NV + kk * NV + j_dof
                    ] = rebind[Scalar[DTYPE]](
                        workspace[env, dcv_off + b * 6 * NV + kk * NV + j_dof]
                    ) + rebind[
                        Scalar[DTYPE]
                    ](
                        workspace[env, co_off + j_dof * 6 + kk]
                    )

    # =========================================================================
    # Step 2: Forward pass - compute Dcacc and Dcfrcbody
    # =========================================================================

    # Zero Dcacc and Dcfrcbody
    for i in range(NBODY * 6 * NV):
        workspace[env, dca_off + i] = Scalar[DTYPE](0)
        workspace[env, dcf_off + i] = Scalar[DTYPE](0)

    var dmul = InlineArray[Scalar[DTYPE], 36](uninitialized=True)
    var mat1 = InlineArray[Scalar[DTYPE], 36](uninitialized=True)
    var mat2 = InlineArray[Scalar[DTYPE], 36](uninitialized=True)
    var tmp6 = InlineArray[Scalar[DTYPE], 6](uninitialized=True)

    for b in range(1, NBODY):
        var body_off = model_body_offset(b)
        var parent = Int(
            rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_PARENT])
        )

        # Dcacc[b] = Dcacc[parent]
        for idx in range(6 * NV):
            workspace[env, dca_off + b * 6 * NV + idx] = workspace[
                env, dca_off + parent * 6 * NV + idx
            ]

        # Dcacc += D(cdof_dot * qvel)
        if body_dofadr[b] >= 0:
            var dof_start = body_dofadr[b]
            var dof_end = dof_start + body_dofnum[b]
            for j_dof in range(dof_start, dof_end):
                # Dcacc += cdof_dot * D(qvel): column j_dof gets cdof_dot[j_dof]
                for k in range(6):
                    workspace[
                        env, dca_off + b * 6 * NV + k * NV + j_dof
                    ] = rebind[Scalar[DTYPE]](
                        workspace[env, dca_off + b * 6 * NV + k * NV + j_dof]
                    ) + rebind[
                        Scalar[DTYPE]
                    ](
                        workspace[env, cd_off + j_dof * 6 + k]
                    )

                # Dcacc += D(cdof_dot) * qvel[j_dof]
                var qvel_j = rebind[Scalar[DTYPE]](state[env, qv_off + j_dof])
                for idx in range(6 * NV):
                    workspace[env, dca_off + b * 6 * NV + idx] = (
                        rebind[Scalar[DTYPE]](
                            workspace[env, dca_off + b * 6 * NV + idx]
                        )
                        + rebind[Scalar[DTYPE]](
                            workspace[env, dcd_off + j_dof * 6 * NV + idx]
                        )
                        * qvel_j
                    )

        # --- Dcfrcbody = D(cinert * cacc + cvel x* (cinert * cvel)) ---
        var ci = InlineArray[Scalar[DTYPE], 10](uninitialized=True)
        for k in range(10):
            ci[k] = rebind[Scalar[DTYPE]](workspace[env, ci_off + b * 10 + k])

        _mjd_mulInertVec_vel(dmul, ci)

        # Dcfrcbody[b] = dmul @ Dcacc[b]
        for ii in range(6):
            for kk in range(NV):
                var s: Scalar[DTYPE] = 0
                for jj in range(6):
                    s += dmul[ii * 6 + jj] * rebind[Scalar[DTYPE]](
                        workspace[env, dca_off + b * 6 * NV + jj * NV + kk]
                    )
                workspace[env, dcf_off + b * 6 * NV + ii * NV + kk] = s

        # mat = D(crossForce)/D(vel) + D(crossForce)/D(frc) @ D(mulInertVec)/D(vel)
        var cv = InlineArray[Scalar[DTYPE], 6](uninitialized=True)
        for k in range(6):
            cv[k] = rebind[Scalar[DTYPE]](workspace[env, cv_off + b * 6 + k])
        _mulInertVec(tmp6, ci, cv)
        _mjd_crossForce_vel(mat, tmp6)
        _mjd_crossForce_frc(mat1, cv)
        _matmul_6x6_x_6x6(mat2, mat1, dmul)
        for k in range(36):
            mat[k] += mat2[k]

        # Dcfrcbody[b] += mat @ Dcvel[b]
        for ii in range(6):
            for kk in range(NV):
                var s: Scalar[DTYPE] = 0
                for jj in range(6):
                    s += mat[ii * 6 + jj] * rebind[Scalar[DTYPE]](
                        workspace[env, dcv_off + b * 6 * NV + jj * NV + kk]
                    )
                workspace[env, dcf_off + b * 6 * NV + ii * NV + kk] = (
                    rebind[Scalar[DTYPE]](
                        workspace[env, dcf_off + b * 6 * NV + ii * NV + kk]
                    )
                    + s
                )

    # =========================================================================
    # Step 3: Backward pass - accumulate Dcfrcbody to parents
    # =========================================================================
    for b in range(NBODY - 1, 0, -1):
        var body_off = model_body_offset(b)
        var parent = Int(
            rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_PARENT])
        )
        if parent >= 0:
            for idx in range(6 * NV):
                workspace[env, dcf_off + parent * 6 * NV + idx] = rebind[
                    Scalar[DTYPE]
                ](workspace[env, dcf_off + parent * 6 * NV + idx]) + rebind[
                    Scalar[DTYPE]
                ](
                    workspace[env, dcf_off + b * 6 * NV + idx]
                )

    # =========================================================================
    # Step 4: Project to joint space — subtract from qDeriv
    # =========================================================================
    for i in range(NV):
        var body_i = dof_bodyid[i]
        for k in range(NV):
            var s: Scalar[DTYPE] = 0
            for comp in range(6):
                s += rebind[Scalar[DTYPE]](
                    workspace[env, co_off + i * 6 + comp]
                ) * rebind[Scalar[DTYPE]](
                    workspace[env, dcf_off + body_i * 6 * NV + comp * NV + k]
                )
            workspace[env, qd_off + i * NV + k] = (
                rebind[Scalar[DTYPE]](workspace[env, qd_off + i * NV + k]) - s
            )

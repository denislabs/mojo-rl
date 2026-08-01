"""Acceleration-stage site sensors — MuJoCo `accelerometer`, `force`, `torque`.

All three read the `mj_rnePostConstraint` products (`dynamics/rne_post.mojo`)
and transport them from the subtree CoM of the body's kinematic root to the
site, then rotate into the site frame — `mju_transformSpatial`:

    flg_force = 0 (motion):  ang' = ang            lin' = lin - dif x ang
    flg_force = 1 (force):   trq' = trq - dif x f  f'   = f
    dif = site_xpos - subtree_com[rootid]

    accelerometer -> objectAcceleration(site).lin, with the
                     `vel_ang x vel_lin` correction MuJoCo adds afterwards
                     (engine_core_util.c:872) — the site is a POINT on a
                     rotating body, so its linear acceleration is not the
                     transported spatial acceleration alone.
    force         -> transformSpatial(cfrc_int[body], flg_force=1)[3:6]
    torque        -> same tmp, [0:3]

Sign convention for force/torque: `cfrc_int` is the interaction force
between the body and its PARENT. For a toe standing on the floor this comes
out NEGATIVE in z (the leg pulls down on the shin), which is what MuJoCo
reports too — dm_control's quadruped feeds it through `arcsinh` and lets the
policy sort it out.

THE SITE FRAME is `xquat[body] * site_quat`, composed by
`kinematics/site_frame.site_world_quat_list`. `Data` deliberately stores no
`site_xmat` — it is one quaternion multiply and every consumer already holds
the body quaternion, so materialising a `[BATCH, NSITE*9]` tensor and writing
it in four forward-kinematics paths would buy nothing the dynamics reads. See
that module for the reasoning.

⚠ This USED TO substitute the body's quaternion for the site's, which is exact
only for an identity-oriented site. Every model that reached here was of that
form, so the substitution was invisible; manipulator's rotated box touch zones
are what forced the site quaternion into the model record, and this was fixed
with it. Fixing it moved NO existing gate, which is the evidence that the old
scope really was as narrow as it claimed.
"""

from ..kinematics.site_frame import site_world_quat_list
from ..kinematics.quat_math import quat_rotate_inverse
from ..gpu.constants import MODEL_BODY_SIZE, BODY_IDX_ROOTID


@always_inline
def _site_transport(
    v0: Float64, v1: Float64, v2: Float64,
    v3: Float64, v4: Float64, v5: Float64,
    dx: Float64, dy: Float64, dz: Float64,
    flg_force: Bool,
) -> Tuple[Float64, Float64, Float64, Float64, Float64, Float64]:
    """`mju_transformSpatial` translation half (no rotation), `d = new - old`.
    Returns the packed 6-vector in MuJoCo order (rotational first)."""
    if flg_force:
        return (
            v0 - (dy * v5 - dz * v4),
            v1 - (dz * v3 - dx * v5),
            v2 - (dx * v4 - dy * v3),
            v3,
            v4,
            v5,
        )
    return (
        v0,
        v1,
        v2,
        v3 - (dy * v2 - dz * v1),
        v4 - (dz * v0 - dx * v2),
        v5 - (dx * v1 - dy * v0),
    )


def site_accelerometer[
    DTYPE: DType
](
    cvel: List[Scalar[DTYPE]],
    cacc: List[Scalar[DTYPE]],
    subtree_com: List[Scalar[DTYPE]],
    site_xpos: List[Scalar[DTYPE]],
    xquat: List[Scalar[DTYPE]],
    m_bodies: List[Scalar[DTYPE]],
    m_sites: List[Scalar[DTYPE]],
    body: Int,
    site: Int,
) raises -> Tuple[Float64, Float64, Float64]:
    """`sensordata` for `<accelerometer site=...>` — linear acceleration of
    the site point, in the site frame.

    All lists are the single-env `Data` / `Model` host buffers (`.data`).
    Requires the integrator to have run with `RNE_POST=True`.
    """
    var root = Int(m_bodies[body * MODEL_BODY_SIZE + BODY_IDX_ROOTID])
    var dx = Float64(site_xpos[site * 3 + 0]) - Float64(
        subtree_com[root * 3 + 0]
    )
    var dy = Float64(site_xpos[site * 3 + 1]) - Float64(
        subtree_com[root * 3 + 1]
    )
    var dz = Float64(site_xpos[site * 3 + 2]) - Float64(
        subtree_com[root * 3 + 2]
    )

    var v = _site_transport(
        Float64(cvel[body * 6 + 0]), Float64(cvel[body * 6 + 1]),
        Float64(cvel[body * 6 + 2]), Float64(cvel[body * 6 + 3]),
        Float64(cvel[body * 6 + 4]), Float64(cvel[body * 6 + 5]),
        dx, dy, dz, False,
    )
    var a = _site_transport(
        Float64(cacc[body * 6 + 0]), Float64(cacc[body * 6 + 1]),
        Float64(cacc[body * 6 + 2]), Float64(cacc[body * 6 + 3]),
        Float64(cacc[body * 6 + 4]), Float64(cacc[body * 6 + 5]),
        dx, dy, dz, False,
    )

    # The SITE's world frame, not the body's — `xquat[body] * site_quat`.
    var sq = site_world_quat_list[DTYPE](m_sites, xquat, body, site)
    var qx = Scalar[DTYPE](sq[0])
    var qy = Scalar[DTYPE](sq[1])
    var qz = Scalar[DTYPE](sq[2])
    var qw = Scalar[DTYPE](sq[3])

    # Rotate into the site frame BEFORE the correction — MuJoCo builds both
    # `vel` and `res` in the local frame and only then adds vel_ang x vel_lin.
    var vl = quat_rotate_inverse[DType.float64](
        Float64(qx), Float64(qy), Float64(qz), Float64(qw), v[0], v[1], v[2]
    )
    var vv = quat_rotate_inverse[DType.float64](
        Float64(qx), Float64(qy), Float64(qz), Float64(qw), v[3], v[4], v[5]
    )
    var al = quat_rotate_inverse[DType.float64](
        Float64(qx), Float64(qy), Float64(qz), Float64(qw), a[3], a[4], a[5]
    )

    return (
        al[0] + (vl[1] * vv[2] - vl[2] * vv[1]),
        al[1] + (vl[2] * vv[0] - vl[0] * vv[2]),
        al[2] + (vl[0] * vv[1] - vl[1] * vv[0]),
    )


def site_force_torque[
    DTYPE: DType
](
    cfrc_int: List[Scalar[DTYPE]],
    subtree_com: List[Scalar[DTYPE]],
    site_xpos: List[Scalar[DTYPE]],
    xquat: List[Scalar[DTYPE]],
    m_bodies: List[Scalar[DTYPE]],
    m_sites: List[Scalar[DTYPE]],
    body: Int,
    site: Int,
) raises -> Tuple[Float64, Float64, Float64, Float64, Float64, Float64]:
    """`sensordata` for `<force site=...>` then `<torque site=...>` on the
    same site — the body/parent interaction wrench in the site frame.

    Returns force first, then torque (the opposite of MuJoCo's packed order,
    which puts the rotational half first; callers here want the force far
    more often).
    """
    var root = Int(m_bodies[body * MODEL_BODY_SIZE + BODY_IDX_ROOTID])
    var dx = Float64(site_xpos[site * 3 + 0]) - Float64(
        subtree_com[root * 3 + 0]
    )
    var dy = Float64(site_xpos[site * 3 + 1]) - Float64(
        subtree_com[root * 3 + 1]
    )
    var dz = Float64(site_xpos[site * 3 + 2]) - Float64(
        subtree_com[root * 3 + 2]
    )

    var t = _site_transport(
        Float64(cfrc_int[body * 6 + 0]), Float64(cfrc_int[body * 6 + 1]),
        Float64(cfrc_int[body * 6 + 2]), Float64(cfrc_int[body * 6 + 3]),
        Float64(cfrc_int[body * 6 + 4]), Float64(cfrc_int[body * 6 + 5]),
        dx, dy, dz, True,
    )

    # The SITE's world frame, not the body's — `xquat[body] * site_quat`.
    var sq = site_world_quat_list[DTYPE](m_sites, xquat, body, site)
    var qx = sq[0]
    var qy = sq[1]
    var qz = sq[2]
    var qw = sq[3]

    var trq = quat_rotate_inverse[DType.float64](
        qx, qy, qz, qw, t[0], t[1], t[2]
    )
    var frc = quat_rotate_inverse[DType.float64](
        qx, qy, qz, qw, t[3], t[4], t[5]
    )

    return (frc[0], frc[1], frc[2], trq[0], trq[1], trq[2])

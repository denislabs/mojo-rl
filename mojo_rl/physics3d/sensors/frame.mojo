"""The frame sensors — pos, quat, axes and velocities (AUD-23).

One family, one helper. MuJoCo evaluates `framepos`, `framequat` and the
three axis sensors from the same pair of quantities: the world POSITION and
the world ORIENTATION of whatever object the sensor names
(`engine_sensor.c:681-736`, via `get_xpos_xmat` and `get_xquat` at
`:226-278`). Everything else is a projection of those two.

`framelinvel` and `frameangvel` add a third quantity — the object's BODY, so
the body's spatial velocity can be transported to the object's point
(`mj_objectVelocity`, engine_core_util.c:835). They are in the same file
because they take the same `objtype` dispatch and the same reference frame,
and splitting them would mean two spellings of that dispatch.

⚠⚠ FIVE OBJECT TYPES, AND `body` IS NOT THE BODY'S OWN FRAME. MuJoCo's
`frameobj_map` (xml/generated/mjcf_map.h:318) admits body, xbody, geom, site
and camera, and the first two are DIFFERENT FRAMES OF THE SAME BODY:

    objtype="xbody"   the body frame          xpos,  xquat
    objtype="body"    the INERTIAL frame      xipos, xquat * body_iquat

They coincide only when the centre of mass sits at the body origin with the
principal axes aligned. On a capsule limb with `fromto`, they do not, and the
error is a fixed offset and a fixed rotation — a number that tracks the body
perfectly and is in the wrong place. The spelling reads backwards and is
MuJoCo's; do not "fix" it.

⚠ THE QUATERNION ORDER FLIPS AT THE `sensordata` BOUNDARY. This tree stores
quaternions (x, y, z, w) — `BODY_IDX_QUAT_*`, `GEOM_IDX_QUAT_*`,
`SITE_IDX_QUAT_*` and `Data.xquat` all agree on that. MuJoCo's `sensordata`
holds (w, x, y, z), because every `mjtNum quat[4]` in the reference does.
`frame_quat_sensor` returns the MuJoCo order, which is the order the caller
writes; the helper below returns the tree's order, which is the order every
other caller in `physics3d` expects. The two are one line apart and the
failure is silent — a rotated-by-something reading of plausible magnitude.

⚠ THE AXIS SENSORS READ A COLUMN, NOT A ROW. `xmat[offset+0], xmat[offset+3],
xmat[offset+6]` with `xmat` row-major is column `offset` — the object frame's
own x/y/z axis expressed in world coordinates. Reading the row would give the
world axis in object coordinates, which is the INVERSE rotation and agrees
with the right answer exactly when the rotation is symmetric.

⚠ CAMERAS ARE NOT SERVED. The four resolvable object types are body, xbody,
geom and site; `<camera>` has no name lookup in this parser, so a frame
sensor naming one stays ADDRESSED and is counted under AUD-23 like any other
gap. That decision is made per ROW in `_fill_sensors`, not per element name —
the first place in the loader where `served` depends on an attribute.
"""

from ..constants import (
    SENSOBJ_BODY,
    SENSOBJ_XBODY,
    SENSOBJ_GEOM,
    SENSOBJ_SITE,
)
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_IQUAT_X,
    BODY_IDX_IQUAT_Y,
    BODY_IDX_IQUAT_Z,
    BODY_IDX_IQUAT_W,
    MODEL_GEOM_SIZE,
    GEOM_IDX_BODY,
    GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y,
    GEOM_IDX_POS_Z,
    GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_Y,
    GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_W,
    MODEL_SITE_SIZE,
    SITE_IDX_BODY,
    SITE_IDX_QUAT_X,
    SITE_IDX_QUAT_Y,
    SITE_IDX_QUAT_Z,
    SITE_IDX_QUAT_W,
)
from ..kinematics.quat_math import gpu_quat_mul, gpu_quat_rotate
from ..kinematics.xmat import quat_xmat_elem
from .frame_vel import point_velocity_world


@always_inline
def frame_object_pose[
    DTYPE: DType
](
    xpos: List[Scalar[DTYPE]],
    xquat: List[Scalar[DTYPE]],
    xipos: List[Scalar[DTYPE]],
    site_xpos: List[Scalar[DTYPE]],
    m_bodies: List[Scalar[DTYPE]],
    m_geoms: List[Scalar[DTYPE]],
    m_sites: List[Scalar[DTYPE]],
    objtype: Int,
    objid: Int,
) -> Tuple[Float64, Float64, Float64, Float64, Float64, Float64, Float64]:
    """World (pos, quat) of a frame-sensor object. Quaternion is (x, y, z, w).

    MuJoCo's `get_xpos_xmat` + `get_xquat`, merged: this tree does not
    materialise `xmat`, `geom_xmat` or `site_xmat`, so the orientation is
    carried as the quaternion the two of them agree on and turned into matrix
    columns by the caller that needs them (`quat_xmat_elem`).

    ⚠ AN UNRECOGNISED `objtype` RETURNS THE IDENTITY AT THE ORIGIN rather than
    raising. Nothing reaches here with one: `_fill_sensors` resolves the four
    supported types and leaves every other row UNSERVED, and the eval pass
    skips unserved rows. The fallback exists so this function is total.
    """
    if objtype == SENSOBJ_XBODY:
        # The body's own frame — `d->xpos` / `d->xquat`, unmodified.
        return (
            Float64(xpos[objid * 3 + 0]),
            Float64(xpos[objid * 3 + 1]),
            Float64(xpos[objid * 3 + 2]),
            Float64(xquat[objid * 4 + 0]),
            Float64(xquat[objid * 4 + 1]),
            Float64(xquat[objid * 4 + 2]),
            Float64(xquat[objid * 4 + 3]),
        )

    if objtype == SENSOBJ_BODY:
        # ⚠ THE INERTIAL FRAME. `d->xipos` is already the CoM in world
        # coordinates (the FK writes it), so only the orientation composes.
        var bb = objid * MODEL_BODY_SIZE
        var q = gpu_quat_mul(
            Float64(xquat[objid * 4 + 0]),
            Float64(xquat[objid * 4 + 1]),
            Float64(xquat[objid * 4 + 2]),
            Float64(xquat[objid * 4 + 3]),
            Float64(m_bodies[bb + BODY_IDX_IQUAT_X]),
            Float64(m_bodies[bb + BODY_IDX_IQUAT_Y]),
            Float64(m_bodies[bb + BODY_IDX_IQUAT_Z]),
            Float64(m_bodies[bb + BODY_IDX_IQUAT_W]),
        )
        return (
            Float64(xipos[objid * 3 + 0]),
            Float64(xipos[objid * 3 + 1]),
            Float64(xipos[objid * 3 + 2]),
            q[0], q[1], q[2], q[3],
        )

    if objtype == SENSOBJ_GEOM:
        # `geom_xpos` / `geom_xquat`, composed here rather than called so the
        # two reads share one `base`. The `body == 0` shortcut is the one
        # those two take: the worldbody frame is the identity.
        var gb = objid * MODEL_GEOM_SIZE
        var b = Int(m_geoms[gb + GEOM_IDX_BODY])
        var lx = Float64(m_geoms[gb + GEOM_IDX_POS_X])
        var ly = Float64(m_geoms[gb + GEOM_IDX_POS_Y])
        var lz = Float64(m_geoms[gb + GEOM_IDX_POS_Z])
        var gx = Float64(m_geoms[gb + GEOM_IDX_QUAT_X])
        var gy = Float64(m_geoms[gb + GEOM_IDX_QUAT_Y])
        var gz = Float64(m_geoms[gb + GEOM_IDX_QUAT_Z])
        var gw = Float64(m_geoms[gb + GEOM_IDX_QUAT_W])
        if b == 0:
            return (lx, ly, lz, gx, gy, gz, gw)
        var bx = Float64(xquat[b * 4 + 0])
        var by = Float64(xquat[b * 4 + 1])
        var bz = Float64(xquat[b * 4 + 2])
        var bw = Float64(xquat[b * 4 + 3])
        var rot = gpu_quat_rotate(bx, by, bz, bw, lx, ly, lz)
        var q = gpu_quat_mul(bx, by, bz, bw, gx, gy, gz, gw)
        return (
            Float64(xpos[b * 3 + 0]) + rot[0],
            Float64(xpos[b * 3 + 1]) + rot[1],
            Float64(xpos[b * 3 + 2]) + rot[2],
            q[0], q[1], q[2], q[3],
        )

    if objtype == SENSOBJ_SITE:
        # ⚠ `site_xpos` IS MATERIALISED AND `site_xmat` IS NOT — the
        # asymmetry `kinematics/site_frame.mojo` documents. So the position
        # is a read and the orientation is a compose.
        var sb = objid * MODEL_SITE_SIZE
        var b = Int(m_sites[sb + SITE_IDX_BODY])
        var q = gpu_quat_mul(
            Float64(xquat[b * 4 + 0]),
            Float64(xquat[b * 4 + 1]),
            Float64(xquat[b * 4 + 2]),
            Float64(xquat[b * 4 + 3]),
            Float64(m_sites[sb + SITE_IDX_QUAT_X]),
            Float64(m_sites[sb + SITE_IDX_QUAT_Y]),
            Float64(m_sites[sb + SITE_IDX_QUAT_Z]),
            Float64(m_sites[sb + SITE_IDX_QUAT_W]),
        )
        return (
            Float64(site_xpos[objid * 3 + 0]),
            Float64(site_xpos[objid * 3 + 1]),
            Float64(site_xpos[objid * 3 + 2]),
            q[0], q[1], q[2], q[3],
        )

    # Unreachable — see the docstring.
    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


@always_inline
def frame_object_body[
    DTYPE: DType
](
    m_geoms: List[Scalar[DTYPE]],
    m_sites: List[Scalar[DTYPE]],
    objtype: Int,
    objid: Int,
) -> Int:
    """The body a frame-sensor object rides on — `mj_objectVelocity`'s
    `bodyid` (engine_core_util.c:840-872).

    ⚠ MuJoCo ZEROES THE VELOCITY OF A DOF-LESS BODY and returns early
    (`body_dofnum[body_weldid[bodyid]] == 0`, `:880`). There is no such guard
    here and none is needed: `Data.xvel` / `xangvel` are PROPAGATED from the
    world through the joint chain, so a body with no dofs above it already
    holds exactly zero. The guard in the reference exists because `cvel` is
    accumulated differently, not because the answer differs.
    """
    if objtype == SENSOBJ_GEOM:
        return Int(m_geoms[objid * MODEL_GEOM_SIZE + GEOM_IDX_BODY])
    if objtype == SENSOBJ_SITE:
        return Int(m_sites[objid * MODEL_SITE_SIZE + SITE_IDX_BODY])
    # BODY and XBODY are the same body in two frames.
    return objid


@always_inline
def frame_vel_sensor[
    DTYPE: DType
](
    xvel: List[Scalar[DTYPE]],
    xangvel: List[Scalar[DTYPE]],
    xipos: List[Scalar[DTYPE]],
    body: Int,
    px: Float64, py: Float64, pz: Float64,
    has_ref: Bool,
    ref_body: Int,
    rpx: Float64, rpy: Float64, rpz: Float64,
    rqx: Float64, rqy: Float64, rqz: Float64, rqw: Float64,
) -> Tuple[Float64, Float64, Float64, Float64, Float64, Float64]:
    """`mj_objectVelocity(..., flg_local=0)` at `(px,py,pz)`, as (ang, lin).

    Returns MuJoCo's packed order — angular first, then linear — because that
    is the order `engine_sensor.c:936` slices, and the two are three floats
    apart. (`site_frame_velocity` next door returns the OPPOSITE order for its
    own callers' convenience, which is exactly the kind of thing that gets
    read wrong; hence this paragraph.)

    ⚠ THE ROTATING-REFERENCE CORRECTION IS NOT OPTIONAL AND IT IS EASY TO
    DROP. Relative to a frame that is itself turning, the object's linear
    velocity is `v - v_ref - w_ref x (p - p_ref)`, and MuJoCo writes that
    third term as `(p - p_ref) x w_ref` added in
    (`mju_cross(cross, rvec, xvel_ref)` at `:923` — `xvel_ref` as a 3-pointer
    is its ANGULAR half). Omitting it is silent whenever the reference frame
    happens not to be rotating, which is most of the time.
    """
    var w = (
        Float64(xangvel[body * 3 + 0]),
        Float64(xangvel[body * 3 + 1]),
        Float64(xangvel[body * 3 + 2]),
    )
    var v = point_velocity_world[DTYPE](
        xvel, xangvel, xipos, body,
        Scalar[DTYPE](px), Scalar[DTYPE](py), Scalar[DTYPE](pz),
    )
    var ax = w[0]
    var ay = w[1]
    var az = w[2]
    var lx = Float64(v[0])
    var ly = Float64(v[1])
    var lz = Float64(v[2])
    if not has_ref:
        return (ax, ay, az, lx, ly, lz)

    var rw = (
        Float64(xangvel[ref_body * 3 + 0]),
        Float64(xangvel[ref_body * 3 + 1]),
        Float64(xangvel[ref_body * 3 + 2]),
    )
    var rv = point_velocity_world[DTYPE](
        xvel, xangvel, xipos, ref_body,
        Scalar[DTYPE](rpx), Scalar[DTYPE](rpy), Scalar[DTYPE](rpz),
    )
    var dax = ax - rw[0]
    var day = ay - rw[1]
    var daz = az - rw[2]
    var dlx = lx - Float64(rv[0])
    var dly = ly - Float64(rv[1])
    var dlz = lz - Float64(rv[2])

    # `rvec x w_ref`, added to the linear half.
    var rx = px - rpx
    var ry = py - rpy
    var rz = pz - rpz
    dlx += ry * rw[2] - rz * rw[1]
    dly += rz * rw[0] - rx * rw[2]
    dlz += rx * rw[1] - ry * rw[0]

    var a = _rot_transpose_mul(rqx, rqy, rqz, rqw, dax, day, daz)
    var l = _rot_transpose_mul(rqx, rqy, rqz, rqw, dlx, dly, dlz)
    return (a[0], a[1], a[2], l[0], l[1], l[2])


@always_inline
def _rot_transpose_mul(
    qx: Float64, qy: Float64, qz: Float64, qw: Float64,
    vx: Float64, vy: Float64, vz: Float64,
) -> Tuple[Float64, Float64, Float64]:
    """`R(q)^T v` — MuJoCo's `mju_mulMatTVec3(res, xmat_ref, v)`.

    Written as a rotation by the CONJUGATE rather than as three dot products
    against `quat_xmat_elem`, which is the same map and four multiplies
    cheaper. `R(q)^T == R(q^-1)` for a unit quaternion, and every quaternion
    reaching here is a product of unit quaternions.
    """
    var r = gpu_quat_rotate(-qx, -qy, -qz, qw, vx, vy, vz)
    return (r[0], r[1], r[2])


@always_inline
def frame_pos_sensor(
    px: Float64, py: Float64, pz: Float64,
    has_ref: Bool,
    rpx: Float64, rpy: Float64, rpz: Float64,
    rqx: Float64, rqy: Float64, rqz: Float64, rqw: Float64,
) -> Tuple[Float64, Float64, Float64]:
    """`mjSENS_FRAMEPOS` (engine_sensor.c:688-706).

    Global when there is no reference frame; otherwise the offset from the
    reference's origin, EXPRESSED IN THE REFERENCE'S AXES — subtract then
    rotate, in that order. Rotating then subtracting gives a different vector
    whenever the reference is itself rotated.
    """
    if not has_ref:
        return (px, py, pz)
    return _rot_transpose_mul(rqx, rqy, rqz, rqw, px - rpx, py - rpy, pz - rpz)


@always_inline
def frame_axis_sensor(
    qx: Float64, qy: Float64, qz: Float64, qw: Float64,
    axis: Int,
    has_ref: Bool,
    rqx: Float64, rqy: Float64, rqz: Float64, rqw: Float64,
) -> Tuple[Float64, Float64, Float64]:
    """`mjSENS_FRAME{X,Y,Z}AXIS`. `axis` is 0, 1 or 2.

    The object frame's own `axis`-th unit vector, in world coordinates — i.e.
    COLUMN `axis` of the rotation matrix (see the module note), which is
    `quat_xmat_elem` at indices `axis`, `axis+3`, `axis+6`.
    """
    var ax = quat_xmat_elem(qx, qy, qz, qw, axis + 0)
    var ay = quat_xmat_elem(qx, qy, qz, qw, axis + 3)
    var az = quat_xmat_elem(qx, qy, qz, qw, axis + 6)
    if not has_ref:
        return (ax, ay, az)
    return _rot_transpose_mul(rqx, rqy, rqz, rqw, ax, ay, az)


@always_inline
def frame_quat_sensor(
    qx: Float64, qy: Float64, qz: Float64, qw: Float64,
    has_ref: Bool,
    rqx: Float64, rqy: Float64, rqz: Float64, rqw: Float64,
) -> Tuple[Float64, Float64, Float64, Float64]:
    """`mjSENS_FRAMEQUAT` (engine_sensor.c:716-735). Returns (w, x, y, z).

    ⚠ THE RETURN IS IN MuJoCo'S ORDER, NOT THE TREE'S. Everything above this
    line is (x, y, z, w); `sensordata` is (w, x, y, z). The flip happens here,
    once, at the boundary — see the module note.

    The relative form is `conj(ref) * obj`, which is MuJoCo's `mju_negQuat`
    followed by `mju_mulQuat`. ⚠ `mju_negQuat` CONJUGATES; it does not negate
    all four components. A true negation is the SAME rotation and would leave
    this test passing while the name lied.
    """
    if not has_ref:
        return (qw, qx, qy, qz)
    var q = gpu_quat_mul(-rqx, -rqy, -rqz, rqw, qx, qy, qz, qw)
    return (q[3], q[0], q[1], q[2])

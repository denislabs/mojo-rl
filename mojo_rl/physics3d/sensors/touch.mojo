"""Touch sensor — MuJoCo `<touch site="..."/>`.

Port of `engine_sensor.c`'s `mjSENS_TOUCH` case. The sensor sums the NORMAL
force of every active contact that (a) involves the site's body and (b) whose
contact point projects into the site's volume along the contact normal:

    for each contact j with efc_address >= 0:
        if site_body not in {body(geom0), body(geom1)}: skip
        f = mj_contactForce(j)[0]           # normal component, contact frame
        if f <= 0: skip
        ray = normalize(frame_normal * f)   # == the unit normal
        if site_body == body1: ray = -ray   # point INTO the sensor body
        if rayGeom(site_xpos, site_xmat, site_size, contact_pos, ray, type) >= 0:
            sensordata += f

Note the ray starts at the CONTACT POINT and is cast along the normal, and the
zone being intersected is the SITE. A contact inside the site volume always
registers; one outside registers only if the normal points through the site.

Used by dm_control's hopper, whose OBSERVATION carries
`np.log1p(sensordata[['touch_toe', 'touch_heel']])` — so this feeds the policy
input, not only a reward term.

SCOPE: sphere and BOX zones, plus ellipsoid zones MEASURED AS a sphere of
radius size[0]. Hopper's sites are spheres; finger's `touchtop`/`touchbottom`
are ellipsoids (`size=".025 .03 .025"`) and take the approximation, which is
exact there because the in-plane semi-axes are equal and the model is planar —
`test_finger_vs_dm_control::test_touch_site_sphere_approximation_is_exact`
pins both facts. Any OTHER site type raises rather than being silently
treated as a sphere; a capsule zone needs its own ray test.

Box zones landed 2026-08-01 with manipulator, all five of whose `<touch>`
sensors are boxes. They are the first zone whose answer depends on the site's
ORIENTATION, which is why they had to wait for the site quaternion to reach
the model record (`SITE_IDX_QUAT_*`). Two of manipulator's zones carry
`euler="0 15 0"`, so treating them as axis-aligned is not a small error on the
two pads that decide whether a grasp registers.

PRECONDITION: contact records must be POST-SOLVE, i.e. read after the
integrator has run the constraint solve for this step. `CONTACT_IDX_FORCE_N`
is zero before that, so a hook that reads it too early gets a silent all-zero
sensor rather than an error.
"""

from std.math import sqrt

from ..fields import Data
from ..constants import GEOM_SPHERE, GEOM_ELLIPSOID, GEOM_BOX
from ..kinematics.quat_math import gpu_quat_mul, gpu_quat_rotate
from ..gpu.constants import (
    CONTACT_SIZE,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_POS_X,
    CONTACT_IDX_POS_Y,
    CONTACT_IDX_POS_Z,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_FORCE_N,
    METADATA_SIZE,
    META_IDX_NUM_CONTACTS,
    MODEL_SITE_SIZE,
    SITE_IDX_BODY,
    SITE_IDX_TYPE,
    SITE_IDX_SIZE_0,
    SITE_IDX_SIZE_1,
    SITE_IDX_SIZE_2,
    SITE_IDX_QUAT_X,
    SITE_IDX_QUAT_Y,
    SITE_IDX_QUAT_Z,
    SITE_IDX_QUAT_W,
)


def touch_sphere_site[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    MAX_CONTACTS: Int,
    NSITE: Int,
](
    d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
    m_sites: List[Scalar[DTYPE]],
    site: Int,
    scale: Float64,
) raises -> Float64:
    """`sensordata` for one `<touch>` sensor, single-env (BATCH=1) CPU path.

    `scale` multiplies every normal force before summing. `CONTACT_IDX_FORCE_N`
    is already in `mj_contactForce`'s units (verified on a settling drop
    against MuJoCo's own `sensordata`), so callers matching MuJoCo pass 1.0.
    The parameter exists for callers wanting impulses or a normalised signal.
    """
    var sbase = site * MODEL_SITE_SIZE
    var stype = Int(m_sites[sbase + SITE_IDX_TYPE])
    # ELLIPSOID is measured as a SPHERE of radius size[0]. That is an
    # approximation, and it is deliberate: it is what this sensor has always
    # done for finger's `touchtop`/`touchbottom`, which are
    # `type="ellipsoid" size=".025 .03 .025"` and used to reach here as
    # GEOM_SPHERE because `_geom_type_from_str` had no `ellipsoid` case and
    # silently defaulted to sphere. Making ellipsoid a real geom type (bug 26)
    # turned that silence into a raise, which would have made finger's
    # observation extraction fail rather than be slightly approximate — so the
    # approximation is now EXPLICIT, and the condition under which it is exact
    # stays pinned by `test_finger_vs_dm_control::
    # test_touch_site_sphere_approximation_is_exact` (equal in-plane semi-axes,
    # planar model). A zone that needs the real ellipsoid needs its own
    # narrow phase; box zones landed 2026-08-01 for manipulator.
    if stype != GEOM_SPHERE and stype != GEOM_ELLIPSOID and stype != GEOM_BOX:
        raise Error(
            String(
                "physics3d touch sensor: site ",
                site,
                " has type ",
                stype,
                "; only sphere zones (type ",
                GEOM_SPHERE,
                "), box zones (type ",
                GEOM_BOX,
                ") and ellipsoid zones measured as a sphere of radius"
                " size[0] (type ",
                GEOM_ELLIPSOID,
                ") are implemented. A capsule zone needs its own ray test —",
                " see sensors/touch.mojo.",
            )
        )

    var sbody = Int(m_sites[sbase + SITE_IDX_BODY])
    var radius = Float64(m_sites[sbase + SITE_IDX_SIZE_0])
    var sx = Float64(d.site_xpos.data[site * 3 + 0])
    var sy = Float64(d.site_xpos.data[site * 3 + 1])
    var sz = Float64(d.site_xpos.data[site * 3 + 2])

    # Box half-extents and the site's WORLD orientation, needed only by the
    # box branch. `site_xmat` has no equivalent in `Data` — see
    # `kinematics/site_frame.mojo` for why it is composed rather than stored.
    var hx = Float64(m_sites[sbase + SITE_IDX_SIZE_0])
    var hy = Float64(m_sites[sbase + SITE_IDX_SIZE_1])
    var hz = Float64(m_sites[sbase + SITE_IDX_SIZE_2])
    var wq = gpu_quat_mul[DType.float64](
        Float64(d.xquat.data[sbody * 4 + 0]),
        Float64(d.xquat.data[sbody * 4 + 1]),
        Float64(d.xquat.data[sbody * 4 + 2]),
        Float64(d.xquat.data[sbody * 4 + 3]),
        Float64(m_sites[sbase + SITE_IDX_QUAT_X]),
        Float64(m_sites[sbase + SITE_IDX_QUAT_Y]),
        Float64(m_sites[sbase + SITE_IDX_QUAT_Z]),
        Float64(m_sites[sbase + SITE_IDX_QUAT_W]),
    )

    var ncon = Int(d.meta.data[META_IDX_NUM_CONTACTS])
    if ncon > MAX_CONTACTS:
        ncon = MAX_CONTACTS

    var total = 0.0
    for c in range(ncon):
        var base = c * CONTACT_SIZE
        var ba = Int(d.contacts.data[base + CONTACT_IDX_BODY_A])
        var bb = Int(d.contacts.data[base + CONTACT_IDX_BODY_B])
        if sbody != ba and sbody != bb:
            continue

        # NB: `fn` is a removed Mojo keyword — do not rename this back.
        var f_normal = (
            Float64(d.contacts.data[base + CONTACT_IDX_FORCE_N]) * scale
        )
        if f_normal <= 0.0:
            continue

        var nx = Float64(d.contacts.data[base + CONTACT_IDX_NX])
        var ny = Float64(d.contacts.data[base + CONTACT_IDX_NY])
        var nz = Float64(d.contacts.data[base + CONTACT_IDX_NZ])
        # MuJoCo flips the ray when the sensorized body is body2, so it always
        # points INTO the sensing body.
        if sbody == bb:
            nx = -nx
            ny = -ny
            nz = -nz

        var px = Float64(d.contacts.data[base + CONTACT_IDX_POS_X])
        var py = Float64(d.contacts.data[base + CONTACT_IDX_POS_Y])
        var pz = Float64(d.contacts.data[base + CONTACT_IDX_POS_Z])

        var hit = False
        if stype == GEOM_BOX:
            hit = _ray_hits_box(
                sx, sy, sz,
                wq[0], wq[1], wq[2], wq[3],
                hx, hy, hz,
                px, py, pz, nx, ny, nz,
            )
        else:
            hit = _ray_hits_sphere(
                sx, sy, sz, radius, px, py, pz, nx, ny, nz
            )
        if hit:
            total += f_normal

    return total


def _ray_hits_box(
    cx: Float64,
    cy: Float64,
    cz: Float64,
    qx: Float64,
    qy: Float64,
    qz: Float64,
    qw: Float64,
    hx: Float64,
    hy: Float64,
    hz: Float64,
    px: Float64,
    py: Float64,
    pz: Float64,
    dx: Float64,
    dy: Float64,
    dz: Float64,
) -> Bool:
    """`mju_rayGeom(..., mjGEOM_BOX) >= 0` — port of `ray_box`
    (`engine_ray.c:389`).

    `(c, q)` is the box's WORLD pose and `(hx, hy, hz)` its half-extents;
    `(p, d)` is the ray. MuJoCo maps both into the box frame (`ray_map`), then
    for each axis with a non-degenerate direction component solves
    `lpnt[i] + x*lvec[i] = ±size[i]` and accepts the root when the crossing
    point falls inside that face's rectangle. Returns whether ANY accepted
    root exists — the sensor only needs the sign, not the distance.

    A ray ORIGINATING INSIDE the box hits: the exit face gives a positive
    root. That matters more than the entry case here, because a contact point
    on a grasped object usually lies within the touch zone rather than outside
    it. MuJoCo's bounding-sphere early-out is skipped — it is a pure
    performance guard, redundant with the face loop.
    """
    # ray_map: into the box frame, i.e. rotate by the conjugate.
    var lp = gpu_quat_rotate[DType.float64](
        -qx, -qy, -qz, qw, px - cx, py - cy, pz - cz
    )
    var lv = gpu_quat_rotate[DType.float64](-qx, -qy, -qz, qw, dx, dy, dz)

    var size = InlineArray[Float64, 3](fill=0.0)
    size[0] = hx
    size[1] = hy
    size[2] = hz
    # `iface[i]` = the two axes spanning the face normal to axis i:
    # {1,2}, {0,2}, {0,1}.
    var iface0 = InlineArray[Int, 3](fill=0)
    var iface1 = InlineArray[Int, 3](fill=0)
    iface0[0] = 1
    iface1[0] = 2
    iface0[1] = 0
    iface1[1] = 2
    iface0[2] = 0
    iface1[2] = 1

    for i in range(3):
        if abs(lv[i]) <= 1e-15:  # mjMINVAL
            continue
        for k in range(2):
            var side = Float64(-1.0) if k == 0 else Float64(1.0)
            var sol = (side * size[i] - lp[i]) / lv[i]
            if sol < 0.0:
                continue
            var a0 = iface0[i]
            var a1 = iface1[i]
            var p0 = lp[a0] + sol * lv[a0]
            var p1 = lp[a1] + sol * lv[a1]
            if abs(p0) <= size[a0] and abs(p1) <= size[a1]:
                return True
    return False


def _ray_hits_sphere(
    cx: Float64,
    cy: Float64,
    cz: Float64,
    radius: Float64,
    px: Float64,
    py: Float64,
    pz: Float64,
    dx: Float64,
    dy: Float64,
    dz: Float64,
) -> Bool:
    """`mju_rayGeom(..., mjGEOM_SPHERE) >= 0` for a unit-ish direction.

    Ray origin (p) to sphere (c, radius). MuJoCo returns the distance to the
    first intersection at NON-NEGATIVE range, so a ray starting inside the
    sphere counts (the origin itself is at distance 0).
    """
    var ox = px - cx
    var oy = py - cy
    var oz = pz - cz
    var oo = ox * ox + oy * oy + oz * oz
    if oo <= radius * radius:
        return True  # origin inside the zone

    var dd = dx * dx + dy * dy + dz * dz
    if dd < 1e-18:
        return False
    var od = ox * dx + oy * dy + oz * dz
    if od >= 0.0:
        return False  # sphere is behind the ray
    var disc = od * od - dd * (oo - radius * radius)
    if disc < 0.0:
        return False
    # Both roots are positive here (od < 0 and the origin is outside), so the
    # nearer one is a valid non-negative hit.
    _ = sqrt(disc)
    return True

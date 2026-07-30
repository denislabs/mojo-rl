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

SCOPE: sphere zones, plus ellipsoid zones MEASURED AS a sphere of radius
size[0]. A sphere's zone test is orientation-free, which matters because our
site records carry no quaternion — `site_xmat` has no equivalent here.
Hopper's sites are spheres; finger's `touchtop`/`touchbottom` are ellipsoids
(`size=".025 .03 .025"`) and take the approximation, which is exact there
because the in-plane semi-axes are equal and the model is planar —
`test_finger_vs_dm_control::test_touch_site_sphere_approximation_is_exact`
pins both facts. Any OTHER site type raises rather than being silently
treated as a sphere; extend it (and add site quats to the record) when a
domain needs capsule/box zones, or a real ellipsoid one.

PRECONDITION: contact records must be POST-SOLVE, i.e. read after the
integrator has run the constraint solve for this step. `CONTACT_IDX_FORCE_N`
is zero before that, so a hook that reads it too early gets a silent all-zero
sensor rather than an error.
"""

from std.math import sqrt

from ..fields import Data
from ..constants import GEOM_SPHERE, GEOM_ELLIPSOID
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
    # planar model). A zone that needs the real ellipsoid needs the site
    # quaternion in the model record, which is also what capsule/box need.
    if stype != GEOM_SPHERE and stype != GEOM_ELLIPSOID:
        raise Error(
            String(
                "physics3d touch sensor: site ",
                site,
                " has type ",
                stype,
                "; only sphere zones (type ",
                GEOM_SPHERE,
                ") and ellipsoid zones measured as a sphere of radius"
                " size[0] (type ",
                GEOM_ELLIPSOID,
                ") are implemented. Capsule/box zones need the site",
                " quaternion in the model record — see sensors/touch.mojo.",
            )
        )

    var sbody = Int(m_sites[sbase + SITE_IDX_BODY])
    var radius = Float64(m_sites[sbase + SITE_IDX_SIZE_0])
    var sx = Float64(d.site_xpos.data[site * 3 + 0])
    var sy = Float64(d.site_xpos.data[site * 3 + 1])
    var sz = Float64(d.site_xpos.data[site * 3 + 2])

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

        if _ray_hits_sphere(sx, sy, sz, radius, px, py, pz, nx, ny, nz):
            total += f_normal

    return total


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

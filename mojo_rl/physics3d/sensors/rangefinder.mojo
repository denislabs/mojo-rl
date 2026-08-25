"""`<rangefinder>` — distance from a site along its own +Z, or -1 for no hit.

`engine_sensor.c:601`. For a SITE-attached rangefinder the whole sensor is four
lines of the reference:

    rvec   = site_xmat column 2          (the site's own +Z, in world)
    origin = site_xpos
    dist   = mj_ray(m, d, origin, rvec, NULL, 1, site_bodyid, &geomid, NULL)

⚠⚠ THE RAY POINTS ALONG +Z, NOT -Z. A CAMERA looks down its own -Z
(`mjCCamera`) and a rangefinder fires along its +Z; the two conventions sit
three hundred lines apart in the same reference and are opposite. Getting it
backwards gives a sensor that reads the scenery BEHIND the robot, which on a
symmetric arena is a plausible-looking signal that never converges.

⚠ `bodyexclude` IS THE SITE'S BODY AND NOTHING MORE. A rangefinder on a
quadruped's torso excludes the torso and still sees its own legs — MuJoCo's
behaviour, reproduced rather than improved, and it is why `escape`'s readings
are not simply "distance to terrain".

⚠ STATICS ARE INCLUDED (`flg_static = 1`) and no group is filtered
(`geomgroup = NULL`), so the floor and the terrain both occlude. The only
exclusions left are the sensor's own body and INVISIBILITY — see
`ray/model.mojo`, where the latter is a precomputed flag precisely because it
is the difference between a rangefinder reading terrain and reading a
decoration.

⚠ **NO HIT IS -1, NOT INFINITY, NOT THE CUTOFF** — and -1 is a SENTINEL, not
a distance. ⚠⚠ AN EARLIER VERSION OF THIS NOTE SAID `tanh(-1)` MAKES IT A
NEGATIVE READING. THAT IS WRONG. dm_control's `Physics.rangefinder`
(`quadruped.py:204`) reads

    np.where(rf_readings == -1.0, 1.0, np.tanh(rf_readings))

so a miss is replaced by **1.0** — the same value a very distant hit
saturates to — BEFORE the `tanh` is applied to the rest. There is no divide by
a scale either. The consequence for a consumer is the opposite of what that
note claimed: a miss reads as MAXIMUM range, and it is the caller's job to
apply that substitution, not this function's. Returning -1 here is what makes
the substitution possible; returning a large number would make it impossible.

COST. `ray_model` is a linear scan over every geom, so N rangefinders on a
model of G geoms cost N*G ray/geom queries per step — and any geom that is a
MESH costs its whole triangle soup on top. `geom_world_poses` is therefore
hoisted out: composing it once per step rather than once per sensor is the
difference between 1 and N passes over the geom table.
"""

from mojo_rl.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric

from ..fields import Data, Model
from ..fields.dims import DimsLike
from ..gpu.constants import (
    MODEL_SITE_SIZE,
    SITE_IDX_BODY,
)
from ..kinematics.site_frame import site_world_quat_list
from ..ray.model import ray_model

def rangefinder_site[
    DTYPE: DType, D: DimsLike
](
    d: Data[DTYPE, D, 1],
    m: Model[DTYPE, D],
    m_sites: List[Scalar[DTYPE]],
    site: Int,
    geom_xpos: List[Vec3Generic[DTYPE]],
    geom_xquat: List[QuatGeneric[DTYPE]],
) raises -> Float64 where DTYPE.is_floating_point():
    """`sensordata` for one site-attached `<rangefinder>`, single-env CPU.

    `geom_xpos`/`geom_xquat` come from `geom_world_poses(d, m)`, called ONCE
    per step by the caller — see the module docstring on cost.

    Returns the distance in METRES, because the direction handed to `ray_model`
    is a unit vector and `t` is in units of `|vec|`. -1.0 means the ray hit
    nothing.
    """
    var sb = site * MODEL_SITE_SIZE
    var body = Int(m_sites[sb + SITE_IDX_BODY])

    var origin = Vec3Generic[DTYPE](
        d.site_xpos.data[site * 3 + 0],
        d.site_xpos.data[site * 3 + 1],
        d.site_xpos.data[site * 3 + 2],
    )

    # `site_world_quat_list` returns (x, y, z, w); `Quat` takes (w, x, y, z).
    var q4 = site_world_quat_list[DTYPE](m_sites, d.xquat.data, body, site)
    var sq = QuatGeneric[DTYPE](
        Scalar[DTYPE](q4[3]),
        Scalar[DTYPE](q4[0]),
        Scalar[DTYPE](q4[1]),
        Scalar[DTYPE](q4[2]),
    )
    # `site_xmat` column 2 == the site frame's +Z taken to world.
    var rvec = sq.rotate_vec(Vec3Generic[DTYPE](0, 0, 1))

    var hit = ray_model[DTYPE](
        m.geoms.data,
        m.dims.get_ngeom(),
        m.bodies.data,
        geom_xpos,
        geom_xquat,
        m.mesh_meta.data,
        m.mesh_tris.data,
        m.hfield_meta.data,
        d.hfield_data.data,
        origin,
        rvec,
        body,
    )
    return Float64(hit.t)

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
MESH costs its whole triangle soup on top. ⚠ An earlier form hoisted the geom
world poses into a `List` shared by all N sensors; `ray_model` composes them
inline now, because a GPU thread owns one RAY and cannot hold a scene.

⚠ `mut d` / `mut m` AND THE `mut` ON `Env.get_state` ARE THE SAME REQUIREMENT.
`TensorImpl.lt_dyn` needs a mutable container to hand out a `LayoutTensor`,
Mojo forbids caching one in a struct field (`AnyOrigin`), and there is no
non-mutating constructor. Nothing here writes; see `core/env.mojo` on why the
trait carries the marker anyway.
"""

from mojo_rl.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric

from ..fields import Data, Model, DYN1, DYN2, rl1, rl2
from ..fields.dims import DimsLike
from ..gpu.constants import (
    MODEL_SITE_SIZE,
    SITE_IDX_BODY,
    MODEL_GEOM_SIZE,
    MODEL_BODY_SIZE,
    MODEL_MESH_META_SIZE,
    MAX_GPU_MESHES,
    MODEL_HFIELD_META_SIZE,
    MAX_GPU_HFIELDS,
)
from ..kinematics.site_frame import site_world_quat_list
from ..ray.model import ray_model

@always_inline
def _pos(n: Int) -> Int:
    """`_at_least_one` — every tensor allocates one element even when unused."""
    return n if n > 0 else 1


def rangefinder_site[
    DTYPE: DType, D: DimsLike, BATCH: Int = 1
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    site: Int,
    env: Int = 0,
) raises -> Float64 where DTYPE.is_floating_point():
    """`sensordata` for one site-attached `<rangefinder>`, in METRES.

    -1.0 means the ray hit nothing. ⚠ A SENTINEL, not a distance:
    `dm_control`'s `Physics.rangefinder` replaces it with 1.0 BEFORE the
    `tanh`, so a miss reads as MAXIMUM range. Applying that is the caller's
    job — returning -1 is what makes it possible.
    """
    var sb = site * MODEL_SITE_SIZE
    var body = Int(m.sites.data[sb + SITE_IDX_BODY])
    var nb = m.dims.get_nbody()
    var origin = Vec3Generic[DTYPE](
        d.site_xpos.data[env * m.dims.get_nsite() * 3 + site * 3 + 0],
        d.site_xpos.data[env * m.dims.get_nsite() * 3 + site * 3 + 1],
        d.site_xpos.data[env * m.dims.get_nsite() * 3 + site * 3 + 2],
    )
    var q4 = site_world_quat_list[DTYPE](
        m.sites.data, d.xquat.data, body, site
    )
    var sq = QuatGeneric[DTYPE](
        Scalar[DTYPE](q4[3]),
        Scalar[DTYPE](q4[0]),
        Scalar[DTYPE](q4[1]),
        Scalar[DTYPE](q4[2]),
    )
    # ⚠ +Z. A rangefinder fires along the site's own +Z; a CAMERA looks down
    # its -Z. See the module docstring.
    var rvec = sq.rotate_vec(Vec3Generic[DTYPE](0, 0, 1))

    var ng = m.dims.get_ngeom()
    var hfn = _pos(m.dims.get_nhfield_data())
    var hit = ray_model[DTYPE](
        m.geoms.lt_dyn["cpu", DYN2](rl2(ng, MODEL_GEOM_SIZE)),
        ng,
        m.bodies.lt_dyn["cpu", DYN2](rl2(nb, MODEL_BODY_SIZE)),
        d.xpos.lt_dyn["cpu", DYN2](rl2(BATCH, nb * 3)),
        d.xquat.lt_dyn["cpu", DYN2](rl2(BATCH, nb * 4)),
        env,
        m.mesh_meta.lt_dyn["cpu", DYN1](
            rl1(MAX_GPU_MESHES * MODEL_MESH_META_SIZE)
        ),
        m.mesh_tris.lt_dyn["cpu", DYN1](
            rl1(_pos(m.dims.get_nmesh_tri() * 9))
        ),
        m.hfield_meta.lt_dyn["cpu", DYN1](
            rl1(MAX_GPU_HFIELDS * MODEL_HFIELD_META_SIZE)
        ),
        d.hfield_data.lt_dyn["cpu", DYN1](rl1(BATCH * hfn)),
        m.dims.get_nhfield_data(),
        origin,
        rvec,
        body,
    )
    return Float64(hit.t)

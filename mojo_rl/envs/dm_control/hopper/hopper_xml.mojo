"""`dm_control` `hopper` model — port of `dm_control/suite/hopper.xml`.

Verbatim apart from the three `<include>` lines.

Two things this model is the first to need:

  * `<default class="hopper"><site type="sphere" size="0.05"/></default>`.
    Both touch sites are declared BY CLASS — the elements themselves are bare
    `<site name="touch_toe" pos=".17 0 0"/>` — and the body picks the class up
    via `childclass="hopper"`. Site default-class inheritance was added with
    this port; without it the sites had no type or size, so the touch sensor's
    zone was a degenerate point and both sensors read a flat zero.

  * `<touch>` sensors. See `physics3d/sensors/touch.mojo`.

`<default class="free">` overrides the `hopper` class for the three root DOFs
(`limited="false" damping="0" armature="0" stiffness="0"`), which is a NESTED
class overriding its parent — the case that broke on cartpole and is gated
here again.

Note the floor is at `pos="48 0 0"` with `size="50 1 .2"`, i.e. it spans
x in [-2, 98]: the hopper starts near one end and hops forward along +x.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML

from mojo_rl.envs.dm_control.hopper.hopper_dims import DM_HOPPER_DIMS





comptime pmh = DM_HOPPER_DIMS

# obs = position (qpos[1:], nq-1 = 6) + velocity (nv = 7) + touch (2) = 15
comptime DMHopperModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/hopper.xml",
    nbody=pmh.NBODY, njoint=pmh.NJOINT, nq=pmh.NQ, nv=pmh.NV,
    ngeom=pmh.NGEOM, nact=pmh.NACT, ntex=pmh.NTEX, nmat=pmh.NMAT,
    nlight=pmh.NLIGHT, ncam=pmh.NCAM, nsite=pmh.NSITE,
    max_contacts=16,
    obs_dim_override=15,
    timestep=pmh.TIMESTEP,
]

# Body indices in worldbody DFS order (0 = world).
comptime TORSO_BODY_IDX: Int = 1
comptime FOOT_BODY_IDX: Int = 5

# Site indices in worldbody DFS order.
comptime TOUCH_TOE_SITE_IDX: Int = 0
comptime TOUCH_HEEL_SITE_IDX: Int = 1

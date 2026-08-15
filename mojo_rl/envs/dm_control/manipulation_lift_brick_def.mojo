"""`manipulation/lift_brick_features` as a COMPTIME model def.

Phase 7's fourth task, and the one that shows the family is now two axes
rather than a list: it is `Lift` (the task class of `lift_large_box`) with the
Duplo (the prop of `reach_duplo`).

    nq 16 (9 robot + 7 free joint)   nv 15   ngeom 62 (21 robot + 41 brick)
    nbody 18 (17 robot + the brick)  nsite 37   nsensor 11   nexclude 4

⚠ nsite 37, NOT `reach_duplo`'s 29. `_DuploWithVertexSites` adds eight
`vertex_*` sites at the corners of the brick's `bounding_box`, which is what
`_get_height_of_lowest_vertex` minimises over. Everything else is
`reach_duplo`'s model exactly — same 62 geoms, same masks, same 9 colliders,
same 0.004647 stud radius drawn at `initialize_episode_mjcf`.

⚠ `obs_dim_override=55` — 42 robot + 13 prop. `Phyics3dEnv` sizes the
observation buffer from the MODEL, so the default `nq - skip + nv` formula (31)
would truncate the observation with nothing raised.

See `manipulation_reach_duplo_def` for the brick's contact masks and the stud
radius, and `manipulation_reach_def` for the elliptic cone / noslip / condim-4
measurements that every task in this family shares.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.physics3d.types import ConeType

from .manipulation_lift_brick_xml import lift_brick_xml
from mojo_rl.envs.dm_control.manipulation_lift_brick_dims import (
    LIFT_BRICK_DIMS,
)

comptime pm = LIFT_BRICK_DIMS

comptime LiftBrickModel = ModelDefFromXML[
    xml=lift_brick_xml,
    xml_path="mojo_rl/envs/dm_control/assets/manipulation/lift_brick.xml",
    nbody=pm.NBODY,
    njoint=pm.NJOINT,
    nq=pm.NQ,
    nv=pm.NV,
    ngeom=pm.NGEOM,
    nact=pm.NACT,
    ntex=pm.NTEX,
    nmat=pm.NMAT,
    nlight=pm.NLIGHT,
    ncam=pm.NCAM,
    nsite=pm.NSITE,
    neq=pm.NEQ,
    # ⚠ Every one of these is taken from `pm`, not defaulted — each default
    # silently disables a feature rather than failing. See
    # `manipulation_reach_def` for the individual consequences.
    nexclude=pm.NEXCLUDE,
    npair=pm.NPAIR,
    max_tendon=pm.NTENDON,
    max_condim=pm.MAX_CONDIM,
    max_equality=pm.NEQ * 6,
    max_contacts=128,
    obs_dim_override=55,
    obs_qpos_skip=0,
    timestep=pm.TIMESTEP,
    cone_type=ConeType.ELLIPTIC,
    noslip_iter=pm.NOSLIP_ITER,
]

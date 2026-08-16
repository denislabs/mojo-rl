"""`manipulation/place_brick_features` as a COMPTIME model def.

`Place` — the third and last task CLASS in the manipulation family, after
`Reach` and `Lift`, and the first with TWO props:

    nq 16 (9 robot + 7 free joint)      nv 15      njnt 10
    ngeom 104 = 21 robot + 41 brick + 1 pillar + 41 CRADLE BRICK
    nbody 20  = 17 robot + brick + pedestal + cradle brick
    nsite 48      nsensor 16      nexclude 4

⚠ THE CRADLE IS A SECOND DUPLO, and it is FIXED — `props.Duplo()` with no
observable options, attached to the pedestal rather than to the arena. It
brings 41 more geoms, 17 more sites and 5 more sensors (all four of its frame
sensors exist in the model and NONE of them is an enabled observable), which is
why this is the largest single-brick model in the family.

⚠⚠ THE PEDESTAL IS NOT ON A FREE JOINT. `Place.__init__` calls
`arena.attach(pedestal)`, not `add_free_entity` — so nq/nv are `reach_duplo`'s
exactly, and the pedestal is placed at reset by writing its attachment frame's
`body_pos`, a MODEL constant. Counting bodies and finding 20 while njnt stays
10 is the tell.

⚠ THE CRADLE GEOMS ARE `condim="6"`. Not here — `place_brick`'s cradle is a SECOND Duplo, whose geoms are condim 3 like the first. The note is kept because the two `place_*` tasks share this file's shape and `place_cradle` does have condim-6 geoms. Task #55's elliptic
condim-4/6 rows are what makes that solvable; `max_condim` comes from `pm`, and
the default of 3 would silently drop the torsional and rolling rows.

⚠ `obs_dim_override=58` — 42 robot + 13 prop + 3 pedestal. `Phyics3dEnv` sizes
the observation buffer from the MODEL, so the default `nq - skip + nv` formula
(31) would truncate the observation with nothing raised.

See `manipulation_reach_duplo_def` for the brick's contact masks and the stud
radius drawn at `initialize_episode_mjcf`, and `manipulation_reach_def` for the
elliptic cone / noslip measurements every task in this family shares.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.physics3d.types import ConeType

from mojo_rl.envs.dm_control.manipulation_place_brick_dims import (
    PLACE_BRICK_DIMS,
)

comptime pm = PLACE_BRICK_DIMS

comptime PlaceBrickModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/manipulation/place_brick.xml",
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
    obs_dim_override=58,
    obs_qpos_skip=0,
    timestep=pm.TIMESTEP,
    cone_type=ConeType.ELLIPTIC,
    noslip_iter=pm.NOSLIP_ITER,
]

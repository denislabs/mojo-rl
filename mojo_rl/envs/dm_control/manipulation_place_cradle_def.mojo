"""`manipulation/place_cradle_features` as a COMPTIME model def.

`Place` — the third and last task CLASS in the manipulation family, after
`Reach` and `Lift`, and the first with TWO props:

    nq 16 (9 robot + 7 free joint)      nv 15      njnt 10
    ngeom 66  = 21 robot + 41 brick + 1 pillar + 3 cradle spheres
    nbody 20  = 17 robot + brick + pedestal + cradle
    nsite 31      nsensor 11      nexclude 4

⚠⚠ THE PEDESTAL IS NOT ON A FREE JOINT. `Place.__init__` calls
`arena.attach(pedestal)`, not `add_free_entity` — so nq/nv are `reach_duplo`'s
exactly, and the pedestal is placed at reset by writing its attachment frame's
`body_pos`, a MODEL constant. Counting bodies and finding 20 while njnt stays
10 is the tell.

⚠ THE CRADLE GEOMS ARE `condim="6"`. `SphereCradle` is three spheres arranged into a concave dish, and it declares `condim="6"` on all three so a brick dropped in it can neither slide nor spin out. Task #55's elliptic
condim-4/6 rows are what makes that solvable; `max_condim` comes from `pm`, and
the default of 3 would silently drop the torsional and rolling rows.

⚠ `obs_dim_override=58` — 42 robot + 13 prop + 3 pedestal. `Phyics3dEnv` sizes
the observation buffer from the MODEL, so the default `nq - skip + nv` formula
(31) would truncate the observation with nothing raised.

See `manipulation_reach_duplo_def` for the brick's contact masks and the stud
radius drawn at `initialize_episode_mjcf`, and `manipulation_reach_def` for the
elliptic cone / noslip measurements every task in this family shares.
"""

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType

from .manipulation_place_cradle_xml import place_cradle_xml

comptime pm = parse_xml(place_cradle_xml)

comptime PlaceCradleModel = ModelDefFromXML[
    xml=place_cradle_xml,
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

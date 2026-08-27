"""`manipulation/reach_duplo_features` as a COMPTIME model def.

Phase 7's third task, and the first whose prop is an ARTICULATED ASSEMBLY
rather than one primitive:

    nq 16 (9 robot + 7 free joint)   nv 15   ngeom 62 (21 robot + 41 brick)
    nbody 18 (17 robot + the brick)  nsite 29   nsensor 11   nexclude 4

Structurally it is `lift_large_box_features` with the box swapped for a 2x4
Duplo — same counts everywhere except geoms and sites, same 55-number
observation (42 robot + 13 free prop), same free joint at qpos 9 / dof 9.
What is new is entirely in the brick.

⚠⚠ THE BRICK'S 41 GEOMS ARE NOT 41 COLLIDERS. `duplo2x4.xml` gives every
default class its own `contype`/`conaffinity` bitmask, and only two of the six
classes can touch anything outside the brick:

    class          contype  conaffinity   collides with the arm / ground?
    base                 3            2   YES
    stud                 5            4   YES
    wall                 4            0   no
    flange               4            0   no
    tube                 4            0   no
    stud-capsule         0            0   no  (the `easy_align` alternative)

The arm and the ground are the compiler default `contype=1 conaffinity=1`, and
a pair collides when `contype_a & conaffinity_b` or `contype_b & conaffinity_a`
is nonzero. `4 & 1 == 0` both ways, so walls, flanges and tubes are INTERNAL
geometry — they exist to let two bricks click together, and the masks are what
stops the brick colliding with itself 800 ways. Nine geoms (the base box and
eight stud cylinders) are all that a reach episode can ever contact.

That is why `max_contacts` stays at 128 here: swapping a 1-geom box for a
41-geom brick adds 8 potential pairs against the arm, not 40.

⚠ THE STUD RADIUS IN THE BAKED XML IS 0.004647, NOT the 0.0047 that
`duplo2x4.xml` declares. `props.Duplo.initialize_episode_mjcf` DRAWS it every
episode and composer recompiles — with the default `variation=0.0` the draw is
deterministic and always lands on the class's lower quartile. The generator
bakes AFTER a reset for exactly this reason; see `manipulation_ref._load`.

⚠ `obs_dim_override=55` — 42 robot + 13 prop. `Phyics3dEnv` sizes the
observation buffer from the MODEL, so the default `nq - skip + nv` formula (31)
would truncate the observation with nothing raised.

Everything else — the elliptic cone, `noslip_iterations=5`, condim 4, the mesh
collision path — is `reach_site_features`' story unchanged; see
`manipulation_reach_def` for the measurements behind each.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.physics3d.types import ConeType

from mojo_rl.envs.dm_control.manipulation_reach_duplo_dims import (
    REACH_DUPLO_DIMS,
)

comptime pm = REACH_DUPLO_DIMS

comptime ReachDuploModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/manipulation/reach_duplo.xml",
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

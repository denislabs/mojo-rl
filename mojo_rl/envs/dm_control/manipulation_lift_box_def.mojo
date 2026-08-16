"""`manipulation/lift_large_box_features` as a COMPTIME model def.

The second Phase 7 task and the first with a PROP. Structurally it is
`reach_site_features` plus one free-jointed box:

    nq 16 (9 robot + 7 free joint)   nv 15   ngeom 22 (21 robot + 1 box)
    nbody 18 (17 robot + the box)    nsite 20   nsensor 10

⚠ THE PROP IS ONE GEOM. `lift_large_box` is the only prop task whose prop is a
single primitive; every other one is a Duplo brick (~40 stud geoms) or a
pedestal. That is why it comes first: it introduces the free-prop OBSERVATION,
the prop PLACER and the per-episode target height without also introducing 40
geoms of collision.

⚠ `obs_dim_override=55` — 42 robot + 13 prop. `Phyics3dEnv` sizes the
observation buffer from the MODEL, not from the config hook, so the default
`nq - skip + nv` formula (31) would truncate the observation to its first 31
entries with nothing raised.

⚠ `max_contacts` is 128, matching `reach_site_features`. The box adds one
convex geom against a mesh hand; the measured worst case on the reach model
was 48 over 60 in-range poses and the box cannot multiply that — but this has
NOT been swept for this model, and an undersized buffer silently drops
contacts rather than raising.

Everything else — the elliptic cone, `noslip_iterations=5`, condim 4, the
mesh collision path — is `reach_site_features`' story unchanged; see
`manipulation_reach_def` for the measurements behind each.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.physics3d.types import ConeType

from mojo_rl.envs.dm_control.manipulation_lift_box_dims import (
    LIFT_LARGE_BOX_DIMS,
)

comptime pm = LIFT_LARGE_BOX_DIMS

comptime LiftLargeBoxModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/manipulation/lift_large_box.xml",
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

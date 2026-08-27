"""`manipulation/stack_3_bricks_features` as a COMPTIME model def.

Phase 7's tenth task: `stack_2_bricks` with a third brick. Same fixed base,
same identity order, one more brick and one more stacked pair.

    nq 23   nv 21   njnt 11   ngeom 267   nbody 23   nsite 113

⚠ `randomize_order` IS FALSE, so `desired_order` is `arange(target_height)`
every episode and `_add_or_remove_freejoints` always strips the freejoint from
brick 0. The model is therefore STABLE across resets — measured over 4 — and
needs none of `manipulation_stack_random`'s relabeling.

⚠ SIX/FOUR DUPLOS, NOT THREE/TWO. Every brick has a translucent contactless
hint twin attached immediately after it, so the real bricks are bodies
17, 19 and 21 and the hints sit between them. See `manipulation_stack2_def` for why
`nmocap` is 0 regardless.

⚠ `obs_dim_override=81` — 42 robot + 3 x 13, and NO `desired_order`
prefix. `Phyics3dEnv` sizes the observation buffer from the MODEL, so the
default formula would truncate.

⚠ `max_contacts` is 128. The reference raises `nconmax` to 400 only for
`num_bricks > 3`.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.physics3d.types import ConeType

from mojo_rl.envs.dm_control.manipulation_stack_3_bricks_dims import (
    STACK_3_BRICKS_DIMS,
)

comptime pm = STACK_3_BRICKS_DIMS

comptime Stack3BricksModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/manipulation/stack_3_bricks.xml",
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
    # ⚠ Every one of these is taken from `pm`, not defaulted — see
    # `manipulation_reach_def` for the individual consequences.
    nexclude=pm.NEXCLUDE,
    npair=pm.NPAIR,
    max_tendon=pm.NTENDON,
    max_condim=pm.MAX_CONDIM,
    max_equality=pm.NEQ * 6,
    max_contacts=128,
    obs_dim_override=81,
    obs_qpos_skip=0,
    timestep=pm.TIMESTEP,
    cone_type=ConeType.ELLIPTIC,
    noslip_iter=pm.NOSLIP_ITER,
]

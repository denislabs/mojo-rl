"""`manipulation/stack_2_bricks_features` as a COMPTIME model def.

Phase 7's seventh task and the first of the `Stack` / `Reassemble` family:

    nq 16 (9 robot + ONE free joint)   nv 15   njnt 10
    ngeom 185 = 21 robot + 4 x 41      nbody 21      nsite 79
    nsensor 21      nexclude 4

⚠⚠ ONE FREE JOINT FOR TWO BRICKS. `bricks.py::_add_or_remove_freejoints` runs
from `initialize_episode_mjcf` and REMOVES the freejoint from the brick at
`desired_order[0]` whenever `moveable_base` is False. With
`randomize_order=False` that is always brick 0, so the model is stable across
episodes — this task's whole portability rests on that. Measured over 4 resets:
`nq` 16 and the free joint is `duplo2x4_2/` every time.

⚠ THE BAKED XML MUST THEREFORE BE THE POST-RESET TREE, or brick 0 keeps a
freejoint it does not have and `nq` is 23 instead of 16. `manipulation_ref.
_load` resets once per cached env; see its docstring.

⚠⚠ FOUR DUPLOS, NOT TWO. Every brick has a translucent CONTACTLESS twin used
as a visual goal hint, and they INTERLEAVE — bodies 17/19 are the real bricks
and 18/20 the hints. Reading `duplo2x4_1/` as the second brick is a plausible
mistake that would observe a body nothing can touch.

⚠ `nmocap` IS 0, despite `_hintify` setting `body.mocap = 'true'` on every
body it finds. `duplo2x4.xml` declares NO `<body>` elements — every geom hangs
directly off its worldbody — so `find_all('body')` returns nothing and the flag
lands on nothing. The hints are ordinary static contactless geometry, which is
why positioning them (`_build_stack`) is renderer-only and not ported.

⚠ `obs_dim_override=68` — 42 robot + 2 x 13. `Phyics3dEnv` sizes the
observation buffer from the MODEL, so the default formula (31) would truncate.

⚠ `max_contacts` is 128 here as elsewhere. The reference raises `nconmax` to
400 only for `num_bricks > 3`; two bricks stay inside MuJoCo's defaults.

See `manipulation_reach_duplo_def` for the brick's contact masks and the stud
radius drawn at `initialize_episode_mjcf`, and `manipulation_reach_def` for the
elliptic cone / noslip measurements the whole family shares.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.physics3d.types import ConeType

from mojo_rl.envs.dm_control.manipulation_stack2_dims import (
    STACK_2_BRICKS_DIMS,
)

comptime pm = STACK_2_BRICKS_DIMS

comptime Stack2BricksModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/manipulation/stack_2_bricks.xml",
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
    obs_dim_override=68,
    obs_qpos_skip=0,
    timestep=pm.TIMESTEP,
    cone_type=ConeType.ELLIPTIC,
    noslip_iter=pm.NOSLIP_ITER,
]

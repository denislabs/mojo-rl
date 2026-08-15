"""`manipulation/stack_3_bricks_random_order_features` as a COMPTIME model def.

Phase 7's eighth task, and the FIRST whose model changes every episode in the
reference:

    nq 23 (9 robot + TWO free joints)   nv 21   njnt 11
    ngeom 267 = 21 robot + 6 x 41       nbody 23      nsite 113

⚠⚠ THE REFERENCE PERMUTES WHICH BRICK IS FIXED, AND THIS MODEL DOES NOT.
`_add_or_remove_freejoints` strips the freejoint from `desired_order[0]`, and
`randomize_order=True` redraws that index every episode. Measured over 30
resets: all three choices occur (14 / 7 / 9), with `nq` 23 throughout — so the
COUNTS are stable and only the assignment moves. A port that checked `nq`
across resets would conclude the model is static and be wrong.

⚠⚠ THIS MODEL IS BAKED WITH BRICK 2 (`duplo2x4_4/`, body 21) FIXED, and the
task is made correct by RELABELING rather than by re-baking. The three bricks
are dynamically identical — every geom's type, condim, contype/conaffinity,
size, pos, quat, friction, solref, solimp and margin, the body's mass, inertia,
ipos and iquat, and every site's type, pos and size are bit-identical; the only
difference is `rgba`, which is not in the `_features` observation. So an
episode that wants reference brick `r` at the base is simulated by putting OUR
fixed brick there and permuting the labels. See `manipulation_stack3r_config`.

⚠ THE QPOS LAYOUT DEPENDS ON WHICH BRICK IS FIXED, which is exactly why the
relabeling has to be a permutation of BODIES and not of qpos slices. In THIS
model the free bricks are `duplo2x4/` at qpos 9 / dof 9 and `duplo2x4_2/` at
qpos 16 / dof 15; a bake that had fixed a different brick would number them
differently.

⚠ SIX DUPLOS, NOT THREE — every brick has a translucent contactless hint twin,
and they interleave: bodies 17/19/21 are the real bricks and 18/20/22 the
hints. See `manipulation_stack2_def` for why `nmocap` is 0 anyway.

⚠ `obs_dim_override=84` — 3 `desired_order` + 42 robot + 3 x 13. The
`desired_order` task observable sorts FIRST; `Phyics3dEnv` sizes the buffer
from the MODEL, so the default formula (44) would truncate.

⚠ `max_contacts` is 128. The reference raises `nconmax` to 400 only for
`num_bricks > 3`; three bricks stay inside MuJoCo's defaults.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.physics3d.types import ConeType

from .manipulation_stack3r_xml import stack_3_random_xml
from mojo_rl.envs.dm_control.manipulation_stack3r_dims import (
    STACK_3_RANDOM_DIMS,
)

comptime pm = STACK_3_RANDOM_DIMS

comptime Stack3RandomModel = ModelDefFromXML[
    xml=stack_3_random_xml,
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
    obs_dim_override=84,
    obs_qpos_skip=0,
    timestep=pm.TIMESTEP,
    cone_type=ConeType.ELLIPTIC,
    noslip_iter=pm.NOSLIP_ITER,
]

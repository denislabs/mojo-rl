"""`manipulation/stack_2_of_3_bricks_random_order_features` as a model def.

⚠⚠ THE SAME BAKED XML AS `stack_3_bricks_random_order_features`, byte for
byte, and this file imports that module's string rather than committing a
second 77 KB copy. `_stack(num_bricks=3, moveable_base=False,
randomize_order=True)` builds an identical model whatever `target_height` is —
verified by generating both and diffing the payload — and the bake-time draw
left the same brick (2, `duplo2x4_4/`) welded down in both.

    nq 23   nv 21   njnt 11   ngeom 267   nbody 23   nsite 113

⚠ `obs_dim_override=83`, NOT 84. `target_height` is 2 here, so `desired_order`
is two numbers rather than three; the robot's 42 and the three bricks' 39 are
unchanged. That single number is the ONLY difference between this model def and
`Stack3RandomModel`, which is why they cannot simply be the same one —
`Phyics3dEnv` sizes the observation buffer from the MODEL.

See `manipulation_stack3r_def` for the freejoint permutation this family has,
and `manipulation_stack_random` for the relabeling that answers it.
"""

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType

from .manipulation_stack3r_xml import stack_3_random_xml

comptime pm = parse_xml(stack_3_random_xml)

comptime Stack2of3Model = ModelDefFromXML[
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
    nexclude=pm.NEXCLUDE,
    npair=pm.NPAIR,
    max_tendon=pm.NTENDON,
    max_condim=pm.MAX_CONDIM,
    max_equality=pm.NEQ * 6,
    max_contacts=128,
    obs_dim_override=83,
    obs_qpos_skip=0,
    timestep=pm.TIMESTEP,
    cone_type=ConeType.ELLIPTIC,
    noslip_iter=pm.NOSLIP_ITER,
]

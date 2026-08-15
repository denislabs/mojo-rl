"""`manipulation/reassemble_3_bricks_fixed_order_features` as a COMPTIME model
def.

    nq 23   nv 21   njnt 11   ngeom 267   nbody 23   nsite 113

⚠⚠ IT REUSES `stack_3_bricks`'s BAKED XML, AND THAT IS A MEASUREMENT, NOT AN
ASSUMPTION. `mjcf.export_with_assets` produces a BYTE-IDENTICAL 76311-character
document for the two tasks, because the model is decided entirely by
`_Common.__init__` (three bricks, three hint twins, the Jaco arm) plus which
brick `initialize_episode_mjcf` strips the freejoint from — and both tasks
strip brick 0's. `Stack` does it because `desired_order[0]` is 0 with
`randomize_order=False`; `Reassemble` because `initial_order[0]` is 0 with
`randomize_initial_order=False`. The gate asserts the byte identity against the
reference's own export rather than trusting this paragraph, so if either task
ever changes shape upstream it fails instead of drifting.

⚠ THE TASKS ARE STILL DIFFERENT — same model, different everything else. The
reward pairs are `desired_order = [0, 2, 1]` and not the identity, `close_coef`
is 0 and not 0.1, and the reset builds an ASSEMBLED stack instead of scattering
and settling one. Sharing the XML shares nothing else.

⚠ `obs_dim_override=81` — 42 robot + 3 x 13, and NO `desired_order` prefix:
`randomize_desired_order` is False, so that task observable does not exist.
`Phyics3dEnv` sizes the observation buffer from the MODEL, so the default
formula would truncate.

⚠⚠ `max_contacts` IS 256 HERE AND 128 FOR `stack_3_bricks`, because the two
tasks start in different scenes. A scattered three-brick reset reports 48
contacts; an ASSEMBLED three-brick stack reports 82, since every stacked pair
puts eight studs into eight holes. The reference raises its own `nconmax` only
for `num_bricks > 3` and so leaves this task on defaults, but our narrow phase
also emits within-margin records the reference does not count, and the duplo's
studs declare `margin=1e-4`. 256 is headroom over the measured 82 rather than a
number carried over from a task that never assembles anything.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.physics3d.types import ConeType

from .manipulation_stack_3_bricks_xml import stack_3_bricks_xml
from mojo_rl.envs.dm_control.manipulation_stack_3_bricks_dims import (
    STACK_3_BRICKS_DIMS,
)

comptime pm = STACK_3_BRICKS_DIMS

comptime Reassemble3Model = ModelDefFromXML[
    xml=stack_3_bricks_xml,
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
    max_contacts=256,
    obs_dim_override=81,
    obs_qpos_skip=0,
    timestep=pm.TIMESTEP,
    cone_type=ConeType.ELLIPTIC,
    noslip_iter=pm.NOSLIP_ITER,
]

"""`libero_kitchen_scene6` on the batched GPU env — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run gen-libero-envs
CI checks it with: pixi run gen-libero-envs --check

    families/<family>.family          the slot table
    scenes/<family>.xml               composed from it
    libero_envs/<family>_dims.mojo    MuJoCo's counts (gen-dims)
    placement/<family>.mojo           the device reset's table
    THIS FILE                         model def, config, env

The config is `libero_osc_config.LiberoOscConfig` — OSC_POSE at 20 Hz, the
task layer's hooks over this family's table; its header is the documentation.
`max_contacts` is `budgets.LIBERO_KITCHEN_SCENE6_MAX_CONTACTS` (112, measured
peak 76). 2 free slots. ELLIPTIC cone: robosuite's `base.xml`.
"""

from noeira.physics3d.parser import ModelDefFromXML
from noeira.physics3d.types import ConeType
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.tasks.task_hooks import TASK_GOAL_WORDS
from noeira.tasks.libero_osc_config import LiberoOscConfig
from noeira.tasks.placement.libero_kitchen_scene6 import LiberoKitchenScene6Placement
from noeira.tasks.libero_envs.budgets import LIBERO_KITCHEN_SCENE6_MAX_CONTACTS
from noeira.tasks.libero_envs.libero_kitchen_scene6_dims import LIBERO_KITCHEN_SCENE6_DIMS

comptime _pm = LIBERO_KITCHEN_SCENE6_DIMS

comptime LIBERO_KITCHEN_SCENE6_OBS_DIM: Int = (
    _pm.NQ + _pm.NV + LiberoKitchenScene6Placement.N_FREE + TASK_GOAL_WORDS
)
"""`task_hooks.write_task_obs`'s layout: qpos, qvel, one active word per
free slot, the goal words."""

comptime LiberoKitchenScene6Model = ModelDefFromXML[
    xml_path="noeira/tasks/scenes/libero_kitchen_scene6.xml",
    nbody=_pm.NBODY,
    njoint=_pm.NJOINT,
    nq=_pm.NQ,
    nv=_pm.NV,
    ngeom=_pm.NGEOM,
    nact=_pm.NACT,
    ntex=_pm.NTEX,
    nmat=_pm.NMAT,
    nlight=_pm.NLIGHT,
    ncam=_pm.NCAM,
    nsite=_pm.NSITE,
    nsensor=_pm.NSENSOR, nsensordata=_pm.NSENSORDATA,
    neq=_pm.NEQ,
    nexclude=_pm.NEXCLUDE,
    npair=_pm.NPAIR,
    timestep=_pm.TIMESTEP,
    cone_type=ConeType.ELLIPTIC,
    max_condim=_pm.MAX_CONDIM,
    max_contacts=LIBERO_KITCHEN_SCENE6_MAX_CONTACTS,
    obs_dim_override=LIBERO_KITCHEN_SCENE6_OBS_DIM,
    action_dim_override=7,
]

comptime LiberoKitchenScene6OscConfig = LiberoOscConfig[LiberoKitchenScene6Placement]

comptime LiberoKitchenScene6OscEnv = Phyics3dBatchedEnv[
    LiberoKitchenScene6Model, LiberoKitchenScene6OscConfig, _, CRBA_TREEWALK=True
]

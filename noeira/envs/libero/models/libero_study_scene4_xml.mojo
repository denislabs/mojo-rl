"""`libero_study_scene4` on the batched GPU env — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run gen-libero-envs
CI checks it with: pixi run gen-libero-envs --check

    families/<family>.family          the slot table
    scenes/<family>.xml               composed from it
    models/<family>_dims.mojo    MuJoCo's counts (gen-dims)
    placement/<family>.mojo           the device reset's table
    THIS FILE                         model def, config, env

The config is `libero_osc_config.LiberoOscConfig` — OSC_POSE at 20 Hz, the
task layer's hooks over this family's table; its header is the documentation.
`max_contacts` is `budgets.LIBERO_STUDY_SCENE4_MAX_CONTACTS` (48, measured
peak 12). 3 free slots. ELLIPTIC cone: robosuite's `base.xml`.
"""

from noeira.physics3d.parser import ModelDefFromXML
from noeira.physics3d.types import ConeType
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.tasks.task_hooks import TASK_GOAL_WORDS
from noeira.envs.libero.osc_config import LiberoOscConfig
from noeira.envs.libero.placement.libero_study_scene4 import LiberoStudyScene4Placement
from noeira.envs.libero.models.budgets import LIBERO_STUDY_SCENE4_MAX_CONTACTS
from noeira.envs.libero.models.libero_study_scene4_dims import LIBERO_STUDY_SCENE4_DIMS

comptime _pm = LIBERO_STUDY_SCENE4_DIMS

comptime LIBERO_STUDY_SCENE4_OBS_DIM: Int = (
    _pm.NQ + _pm.NV + LiberoStudyScene4Placement.N_FREE + TASK_GOAL_WORDS
)
"""`task_hooks.write_task_obs`'s layout: qpos, qvel, one active word per
free slot, the goal words."""

comptime LiberoStudyScene4Model = ModelDefFromXML[
    xml_path="noeira/envs/libero/scenes/libero_study_scene4.xml",
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
    max_contacts=LIBERO_STUDY_SCENE4_MAX_CONTACTS,
    obs_dim_override=LIBERO_STUDY_SCENE4_OBS_DIM,
    action_dim_override=7,
]

comptime LiberoStudyScene4OscConfig = LiberoOscConfig[LiberoStudyScene4Placement]

comptime LiberoStudyScene4OscEnv = Phyics3dBatchedEnv[
    LiberoStudyScene4Model, LiberoStudyScene4OscConfig, _, CRBA_TREEWALK=True
]

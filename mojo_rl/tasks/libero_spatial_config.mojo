"""`libero_spatial` on the batched GPU env, driven by OSC_POSE.

The config is `libero_osc_config.LiberoOscConfig` over this family's generated
placement table; everything it does, and why, is documented there. The model
def, its budget and its observation width are `libero_spatial_xml.mojo`.
"""

from mojo_rl.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from mojo_rl.tasks.libero_osc_config import LiberoOscConfig
from mojo_rl.tasks.placement.libero_spatial import LiberoSpatialPlacement
from mojo_rl.tasks.libero_spatial_xml import LiberoSpatialModel


comptime LiberoSpatialOscConfig = LiberoOscConfig[LiberoSpatialPlacement]

comptime LiberoSpatialOscEnv = Phyics3dBatchedEnv[
    LiberoSpatialModel, LiberoSpatialOscConfig, _, CRBA_TREEWALK=True
]

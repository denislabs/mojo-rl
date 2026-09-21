"""`libero_spatial` on the batched GPU env, driven by OSC_POSE.

The config is `libero_osc_config.LiberoOscConfig` over this family's generated
placement table; everything it does, and why, is documented there. The model
def, its budget and its observation width are `libero_spatial_xml.mojo`.
"""

from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.tasks.libero_osc_config import LiberoOscConfig
from noeira.tasks.placement.libero_spatial import LiberoSpatialPlacement
from noeira.tasks.libero_spatial_xml import LiberoSpatialModel


comptime LiberoSpatialOscConfig = LiberoOscConfig[LiberoSpatialPlacement]

comptime LiberoSpatialOscEnv = Phyics3dBatchedEnv[
    LiberoSpatialModel, LiberoSpatialOscConfig, _, CRBA_TREEWALK=True
]

"""`libero_object` on the batched GPU env, driven by OSC_POSE.

The config is `libero_osc_config.LiberoOscConfig` over this family's generated
placement table; everything it does, and why, is documented there. The model
def, its budget and its observation width are `libero_object_xml.mojo`.
"""

from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.tasks.libero_osc_config import LiberoOscConfig
from noeira.tasks.placement.libero_object import LiberoObjectPlacement
from noeira.tasks.libero_object_xml import LiberoObjectModel


comptime LiberoObjectOscConfig = LiberoOscConfig[LiberoObjectPlacement]

comptime LiberoObjectOscEnv = Phyics3dBatchedEnv[
    LiberoObjectModel, LiberoObjectOscConfig, _, CRBA_TREEWALK=True
]

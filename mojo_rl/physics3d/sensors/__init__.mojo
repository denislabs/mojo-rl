"""MuJoCo `<sensor>` equivalents.

The engine has no sensor framework yet (gap G1 in docs/DM_CONTROL_PORT.md):
there is no `SensorData` record, no `Data.sensordata` tensor and no evaluation
pass. Sensors are added here one at a time as ports need them, each as a plain
function over `Data` + the packed model records, so a config hook can call it
directly.

When the full framework lands, these become the per-type kernels behind it.
"""

from .subtree import subtree_linvel, walk_to_root
from .touch import touch_sphere_site

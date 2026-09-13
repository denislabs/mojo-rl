"""MuJoCo `<sensor>` equivalents.

The framework landed in the 3.12 audit's step 6 (AUD-23) and these are now the
per-type kernels behind it: `<sensor>` is parsed into a `SensorData` table
(`parser/flat_model.mojo`), the model carries a record buffer, `Data` carries
`sensordata`, and `eval.mojo` is the `mj_sensorPos` / `Vel` / `Acc` front end
that addresses each kernel by sensor rather than by hand-counted site index.

Each kernel remains a plain function over `Data` + the packed model records, so
a config hook can still call one directly — the env hooks that predate the
framework do exactly that, and they are what the framework is replacing.
"""

from .frame import (
    frame_object_pose,
    frame_pos_sensor,
    frame_axis_sensor,
    frame_quat_sensor,
)
from .frame_vel import site_frame_velocity, site_frame_velocity_gpu
from .site_acc import (
    site_accelerometer,
    site_accelerometer_gpu,
    site_force_torque,
    site_force_torque_gpu,
)
from .subtree import (
    subtree_linvel, subtree_linvel_gpu, subtree_angmom_gpu, walk_to_root
)
from .touch import touch_sphere_site, touch_sphere_site_gpu
from .rangefinder import rangefinder_site
from .eval import (
    sensor_pos,
    sensor_vel,
    sensor_acc,
)

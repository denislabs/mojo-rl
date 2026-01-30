"""Physics3D v2 kinematics - Forward kinematics.

For a free body with FREE joint, forward kinematics is trivial:
world position = qpos[0:3], world quaternion = qpos[3:7].

This becomes more complex with joint hierarchies (later phases).
"""

from .types import Data


fn update_kinematics[DTYPE: DType](mut data: Data[DTYPE]):
    """Update world-frame quantities from generalized coordinates.

    For FREE joint:
    - xpos = qpos[0:3] (position)
    - xquat = qpos[3:7] (orientation)
    """
    data.xpos_x = data.qpos[0]
    data.xpos_y = data.qpos[1]
    data.xpos_z = data.qpos[2]
    data.xquat_x = data.qpos[3]
    data.xquat_y = data.qpos[4]
    data.xquat_z = data.qpos[5]
    data.xquat_w = data.qpos[6]

"""Kinematics: quaternion math + the fields forward kinematics
(`forward_kinematics` — body world poses xpos/xquat/xipos from qpos).
"""

from .quat_math import (
    quat_mul,
    quat_conjugate,
    quat_rotate,
    quat_normalize,
    axis_angle_to_quat,
    quat_integrate,
)
# Legacy struct-Model/Data FK deleted at the G4 fields sunset; the fields
# FK is `forward_kinematics` (imported directly by module).

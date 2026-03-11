"""Forward kinematics for Generalized Coordinates engine.

This module computes body world positions (xpos, xquat) from joint positions (qpos).
"""

from .quat_math import (
    quat_mul,
    quat_conjugate,
    quat_rotate,
    quat_normalize,
    axis_angle_to_quat,
    quat_integrate,
)
from .forward_kinematics import forward_kinematics, compute_body_velocities

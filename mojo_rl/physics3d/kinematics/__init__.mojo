"""Kinematics: quaternion math, MuJoCo-compatible `xmat` accessors, and the
fields forward kinematics (`forward_kinematics` — body world poses
xpos/xquat/xipos from qpos).
"""

from .quat_math import (
    quat_mul,
    quat_conjugate,
    quat_rotate,
    quat_normalize,
    axis_angle_to_quat,
    quat_integrate,
)
from .xmat import (
    xmat_elem,
    quat_xmat_elem,
    XMAT_XX,
    XMAT_XY,
    XMAT_XZ,
    XMAT_YX,
    XMAT_YY,
    XMAT_YZ,
    XMAT_ZX,
    XMAT_ZY,
    XMAT_ZZ,
)
# Legacy struct-Model/Data FK deleted at the G4 fields sunset; the fields
# FK is `forward_kinematics` (imported directly by module).

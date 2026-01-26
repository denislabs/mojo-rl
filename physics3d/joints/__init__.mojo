"""3D Joint System for articulated bodies.

This module provides joint types commonly used in MuJoCo-style environments:
- Hinge3D: 1-DOF revolute joint rotating around a single axis
- Ball3D: 3-DOF spherical joint allowing rotation on all axes
- Free3D: 6-DOF joint for floating base (no constraints)
- Motor3D: PD motor controller for joint actuation
"""

from .hinge3d import Hinge3D
from .ball3d import Ball3D
from .motor3d import Motor3D, PDController

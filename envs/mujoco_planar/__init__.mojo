"""MuJoCo-style Planar Locomotion Environments.

2D (planar) versions of MuJoCo locomotion environments using physics2d.
These provide quick wins by reusing existing 2D physics infrastructure.

Environments:
    - HopperPlanar: 4 bodies, 3 joints, 3D action
    - Walker2dPlanar: 7 bodies, 6 joints, 6D action
    - HalfCheetahPlanar: 7 bodies, 6 joints, 6D action
"""

from .hopper_planar import HopperPlanar, HopperPlanarConstants
from .walker_planar import Walker2dPlanar, Walker2dPlanarConstants
from .cheetah_planar import HalfCheetahPlanar, HalfCheetahPlanarConstants

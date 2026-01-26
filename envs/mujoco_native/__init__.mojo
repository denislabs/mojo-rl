"""MuJoCo-style Native 3D Locomotion Environments.

Full 3D implementations of MuJoCo locomotion environments using physics3d.
These provide accurate 3D physics simulation for complex locomotion tasks.

Environments:
    - Hopper3D: 4 bodies, 3 joints, 3D action (single-leg hopping)
    - Walker2d3D: 7 bodies, 6 joints, 6D action (bipedal walking)
    - HalfCheetah3D: 7 bodies, 6 joints, 6D action (quadruped running)
    - Ant3D: 13 bodies, 8 joints, 8D action (quadruped with spherical joints)
    - Humanoid3D: 13 bodies, 17 joints, 17D action (bipedal humanoid)
"""

from .base import MuJoCoEnvBase3D, MuJoCoConstants, Body3DState, JointState, ContactInfo
from .hopper3d import Hopper3D, Hopper3DConstants
from .walker2d3d import Walker2d3D, Walker2d3DConstants
from .cheetah3d import HalfCheetah3D, HalfCheetah3DConstants

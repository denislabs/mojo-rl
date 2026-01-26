"""3D Collision Detection.

This module provides collision detection for common shapes used
in MuJoCo-style environments:
- Sphere vs Plane (ground contact)
- Capsule vs Plane (limb/foot contact)
- Contact data structure for constraint solving
"""

from .contact3d import Contact3D, ContactManifold
from .sphere_plane import SpherePlaneCollision, SpherePlaneCollisionGPU
from .capsule_plane import CapsulePlaneCollision, CapsulePlaneCollisionGPU

"""3D Physics GPU Kernels.

Provides GPU-accelerated physics simulation for batched environments.
"""

from .physics_step3d import (
    PhysicsWorld3D,
    Physics3DStepKernel,
)

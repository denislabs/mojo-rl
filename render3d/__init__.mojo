"""3D Rendering Module.

GPU-accelerated 3D renderer using SDL3's GPU API with Metal (MSL) shaders
for Blinn-Phong lit environment visualization.
"""

from .camera3d import Camera3D
from .renderer3d import Renderer3D, Color3D
from .gpu_types import MeshHandle

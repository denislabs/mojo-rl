"""3D Rendering Module.

Software wireframe renderer for MuJoCo-style environment visualization.
Uses SDL2 for display and draws 3D shapes projected to 2D.
"""

from .camera3d import Camera3D
from .shapes3d import (
    WireframeSphere,
    WireframeCapsule,
    WireframeBox,
    WireframeLine,
)
from .renderer3d import Renderer3D, Color3D

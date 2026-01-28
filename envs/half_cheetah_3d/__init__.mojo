"""HalfCheetah3D Environment Module.

A 3D HalfCheetah environment using the physics3D engine with full GPU support.
Implements both BoxContinuousActionEnv and GPUContinuousEnv traits.
"""

from .constants3d import HC3DConstants
from .cheetah_3d import HalfCheetah3D
from .renderer import HalfCheetah3DRenderer, CheetahColors
from .state import HalfCheetah3DState
from .action import HalfCheetah3DAction

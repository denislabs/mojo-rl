"""Hopper - MuJoCo-style Hopper using Generalized Coordinates physics.

This environment uses the physics3d Generalized Coordinates engine with
DefaultIntegrator for constraint-based contact solving.

Usage:
    from envs.hopper import Hopper, HopperState, HopperAction

    var env = Hopper()
    var state = env.reset()

    var action = HopperAction(thigh=0.5, leg=-0.2, foot=0.1)
    var result = env.step(action)

Rendering:
    from envs.hopper import Hopper, HopperRenderer
    from math3d import Vec3, Quat

    var env = Hopper()
    var renderer = HopperRenderer()
    renderer.init()

    # Get body positions and orientations
    var torso_pos = env.get_torso_position()
    var torso_quat = env.get_torso_quaternion()
    # ... get other body positions/quaternions

    # Render frame
    renderer.render(
        Vec3(torso_pos[0], torso_pos[1], torso_pos[2]),
        Quat(torso_quat[0], torso_quat[1], torso_quat[2], torso_quat[3]),
        # ... other bodies
    )
"""

from .hopper import Hopper
from .state import HopperState
from .action import HopperAction
from .constants import (
    HopperConstants,
    HopperConstantsCPU,
    HopperConstantsGPU,
)
from .renderer import HopperRenderer, HopperColors
from .curriculum import HopperCurriculum

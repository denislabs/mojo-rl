"""HopperGC - MuJoCo-style Hopper using Generalized Coordinates physics.

This environment uses the physics3d_v2 Generalized Coordinates engine with
SemiImplicitEulerIntegrator for more accurate, energy-conserving simulation.

Usage:
    from envs.hopper_gc import HopperGC, HopperGCState, HopperGCAction

    var env = HopperGC()
    var state = env.reset()

    var action = HopperGCAction(thigh=0.5, leg=-0.2, foot=0.1)
    var result = env.step(action)

Rendering:
    from envs.hopper_gc import HopperGC, HopperGCRenderer
    from math3d import Vec3, Quat

    var env = HopperGC()
    var renderer = HopperGCRenderer()
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

from .hopper_gc import HopperGC
from .state import HopperGCState
from .action import HopperGCAction
from .constants_gc import HopperGCConstants, HopperGCConstantsCPU, HopperGCConstantsGPU
from .renderer import HopperGCRenderer, HopperGCColors

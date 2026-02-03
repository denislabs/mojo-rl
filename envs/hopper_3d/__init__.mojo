"""Hopper3D Environment Module.

A 3D Hopper environment using the physics3d_v2 engine.
Implements BoxContinuousActionEnv and RenderableEnv traits.

Example usage:
    from envs.hopper_3d import Hopper3D, Hopper3DState, Hopper3DAction

    var env = Hopper3D()

    # Reset and get initial observation
    var obs = env.reset_obs_list()

    # Step with continuous actions
    var action = List[Scalar[DType.float64]]()
    action.append(0.5)   # hip
    action.append(-0.3)  # knee
    action.append(0.1)   # ankle
    var result = env.step_continuous_vec(action)
    var next_obs = result[0]
    var reward = result[1]
    var done = result[2]

    # With rendering
    _ = env.init_renderer()
    while not done and not env.check_renderer_quit():
        env.render_frame()
        result = env.step_continuous_vec(action)
        done = result[2]
        env.renderer_delay(16)  # ~60 FPS
    env.close_renderer()
"""

from .constants3d import Hopper3DConstants, Hopper3DConstantsCPU, Hopper3DConstantsGPU
from .hopper_3d import Hopper3D
from .renderer import Hopper3DRenderer, HopperColors
from .state import Hopper3DState
from .action import Hopper3DAction

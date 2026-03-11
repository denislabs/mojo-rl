"""HalfCheetah Environment Package.

MuJoCo-style Half Cheetah using the physics3d Generalized Coordinates engine.

The Half Cheetah is a 2D planar robot (movement in XZ plane, rotation around Y axis)
consisting of a torso with two leg chains (front and back).

Components:
- HalfCheetah: Main environment struct implementing BoxContinuousActionEnv
- ObsState[17]: 17D observation state (8 qpos + 9 qvel) — from mojo_rl.core
- ContAction[6]: 6D continuous action (joint torques) — from mojo_rl.core
Example usage:
    from mojo_rl.envs.half_cheetah import HalfCheetah
    from mojo_rl.core import ContAction

    var env = HalfCheetah()
    var state = env.reset()

    # Random action (6D)
    var action = ContAction[6]()
    var result = env.step(action)
"""

from .half_cheetah import HalfCheetah
from .curriculum import HalfCheetahCurriculum
from .half_cheetah_xml import HalfCheetahModel
from .half_cheetah_config import HalfCheetahConfig

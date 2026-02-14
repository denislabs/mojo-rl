"""HalfCheetah Environment Package.

MuJoCo-style Half Cheetah using the physics3d Generalized Coordinates engine.

The Half Cheetah is a 2D planar robot (movement in XZ plane, rotation around Y axis)
consisting of a torso with two leg chains (front and back).

Components:
- HalfCheetah: Main environment struct implementing BoxContinuousActionEnv
- ObsState[17]: 17D observation state (8 qpos + 9 qvel) — from core
- ContAction[6]: 6D continuous action (joint torques) — from core
- HalfCheetahRenderer: 3D visualization using render3d

Example usage:
    from envs.half_cheetah import HalfCheetah
    from core import ContAction

    var env = HalfCheetah()
    var state = env.reset()

    # Random action (6D)
    var action = ContAction[6]()
    var result = env.step(action)
"""

from .half_cheetah import HalfCheetah
from .renderer import (
    HalfCheetahRenderer,
)
from .curriculum import HalfCheetahCurriculum
from .half_cheetah_def import (
    # Robot definition
    HalfCheetahModel,
    HalfCheetahBodies,
    HalfCheetahJoints,
    HalfCheetahActuators,
    # Params struct
    HalfCheetahParams,
    HalfCheetahParamsCPU,
    HalfCheetahParamsGPU,
    # Body indices
    BODY_TORSO,
    BODY_BTHIGH,
    BODY_BSHIN,
    BODY_BFOOT,
    BODY_FTHIGH,
    BODY_FSHIN,
    BODY_FFOOT,
    # Joint indices
    JOINT_ROOTX,
    JOINT_ROOTZ,
    JOINT_ROOTY,
    JOINT_BTHIGH,
    JOINT_BSHIN,
    JOINT_BFOOT,
    JOINT_FTHIGH,
    JOINT_FSHIN,
    JOINT_FFOOT,
)

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
from .renderer import HalfCheetahRenderer, HalfCheetahColors
from .curriculum import HalfCheetahCurriculum
from .half_cheetah_def import (
    # Robot definition
    HalfCheetahRobot,
    HalfCheetahBodies,
    HalfCheetahJoints,
    # Params struct (new name)
    HalfCheetahParams,
    HalfCheetahParamsCPU,
    HalfCheetahParamsGPU,
    # Backward-compat aliases
    HalfCheetahConstants,
    HalfCheetahConstantsCPU,
    HalfCheetahConstantsGPU,
    # Body indices
    BODY_TORSO,
    BODY_BTHIGH,
    BODY_BSHIN,
    BODY_BFOOT,
    BODY_FTHIGH,
    BODY_FSHIN,
    BODY_FFOOT,
    BODY_HEAD,
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
    JOINT_HEAD,
    # Body geometry
    CAPSULE_RADIUS,
    TORSO_HALF_LENGTH,
    HEAD_HALF_LENGTH,
    HEAD_POS_X,
    HEAD_POS_Y,
    HEAD_POS_Z,
    HEAD_AXIS_ANGLE,
    BTHIGH_HALF_LENGTH,
    BSHIN_HALF_LENGTH,
    BFOOT_HALF_LENGTH,
    FTHIGH_HALF_LENGTH,
    FSHIN_HALF_LENGTH,
    FFOOT_HALF_LENGTH,
    # Body masses
    TORSO_MASS,
    BTHIGH_MASS,
    BSHIN_MASS,
    BFOOT_MASS,
    FTHIGH_MASS,
    FSHIN_MASS,
    FFOOT_MASS,
    # Dimensions
    NQ,
    NV,
    NBODY,
    NJOINT,
    MAX_CONTACTS,
    OBS_DIM,
    ACTION_DIM,
    # Physics parameters
    DT,
    FRAME_SKIP,
    EFFECTIVE_DT,
    GRAVITY_Z,
    GROUND_Z,
    MAX_STEPS,
    INIT_HEIGHT,
    FRICTION,
    RESTITUTION,
    # Reward parameters
    FORWARD_REWARD_WEIGHT,
    CTRL_COST_WEIGHT,
    # Reset
    RESET_NOISE_SCALE,
    # Gear ratios
    BTHIGH_GEAR,
    BSHIN_GEAR,
    BFOOT_GEAR,
    FTHIGH_GEAR,
    FSHIN_GEAR,
    FFOOT_GEAR,
    # Joint limits
    BTHIGH_LOWER,
    BTHIGH_UPPER,
    BSHIN_LOWER,
    BSHIN_UPPER,
    BFOOT_LOWER,
    BFOOT_UPPER,
    FTHIGH_LOWER,
    FTHIGH_UPPER,
    FSHIN_LOWER,
    FSHIN_UPPER,
    FFOOT_LOWER,
    FFOOT_UPPER,
)

"""Hopper Environment Package.

MuJoCo-style Hopper using the physics3d Generalized Coordinates engine.

The Hopper is a 2D planar robot (movement in XZ plane, rotation around Y axis)
consisting of a torso with three body segments (thigh, leg, foot).

Components:
- Hopper: Main environment struct implementing BoxContinuousActionEnv
- ObsState[11]: 11D observation state (5 qpos + 6 qvel) -- from core
- ContAction[3]: 3D continuous action (joint torques) -- from core
- HopperRenderer: 3D visualization using render3d

Example usage:
    from envs.hopper import Hopper
    from core import ContAction

    var env = Hopper()
    var state = env.reset()

    # Random action (3D)
    var action = ContAction[3]()
    var result = env.step(action)
"""

from .hopper import Hopper
from .renderer import HopperRenderer
from .curriculum import HopperCurriculum
from .hopper_def import (
    # Robot definition
    HopperRobot,
    HopperBodies,
    HopperJoints,
    # Params struct (new name)
    HopperParams,
    HopperParamsCPU,
    HopperParamsGPU,
    # Backward-compat aliases
    HopperConstants,
    HopperConstantsCPU,
    HopperConstantsGPU,
    # Body type aliases
    HopperTorso,
    HopperThigh,
    HopperLeg,
    HopperFoot,
    # Body indices
    BODY_TORSO,
    BODY_THIGH,
    BODY_LEG,
    BODY_FOOT,
    # Joint indices
    JOINT_ROOTX,
    JOINT_ROOTZ,
    JOINT_ROOTY,
    JOINT_THIGH,
    JOINT_LEG,
    JOINT_FOOT,
    # Body geometry
    TORSO_RADIUS,
    TORSO_HALF_LENGTH,
    THIGH_RADIUS,
    THIGH_HALF_LENGTH,
    LEG_RADIUS,
    LEG_HALF_LENGTH,
    FOOT_RADIUS,
    FOOT_HALF_LENGTH,
    # Body masses
    TORSO_MASS,
    THIGH_MASS,
    LEG_MASS,
    FOOT_MASS,
    # Dimensions
    NQ,
    NV,
    NBODY,
    NJOINT,
    MAX_CONTACTS,
    OBS_DIM,
    ACTION_DIM,
    NUM_BODIES,
    # Physics parameters
    DT,
    FRAME_SKIP,
    EFFECTIVE_DT,
    GRAVITY_Z,
    GROUND_Z,
    MAX_STEPS,
    INITIAL_Z,
    FRICTION,
    RESTITUTION,
    # Reward parameters
    FORWARD_REWARD_WEIGHT,
    CTRL_COST_WEIGHT,
    HEALTHY_REWARD,
    # Reset
    RESET_NOISE_SCALE,
    TORQUE_LIMIT,
    # Joint limits
    THIGH_JOINT_MIN,
    THIGH_JOINT_MAX,
    LEG_JOINT_MIN,
    LEG_JOINT_MAX,
    FOOT_JOINT_MIN,
    FOOT_JOINT_MAX,
)

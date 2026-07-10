"""Hopper Environment Package.

MuJoCo-style Hopper using the physics3d Generalized Coordinates engine.

The Hopper is a 2D planar robot (movement in XZ plane, rotation around Y axis)
consisting of a torso with three body segments (thigh, leg, foot).

Components:
- Hopper: Main environment struct implementing BoxContinuousActionEnv
- ObsState[11]: 11D observation state (5 qpos + 6 qvel) -- from mojo_rl.core
- ContAction[3]: 3D continuous action (joint torques) -- from mojo_rl.core

Example usage:
    from mojo_rl.envs.hopper import Hopper
    from mojo_rl.core import ContAction

    var env = Hopper()
    var state = env.reset()

    # Random action (3D)
    var action = ContAction[3]()
    var result = env.step(action)
"""

from .hopper import Hopper
from .curriculum import HopperCurriculum
from .hopper_xml import HopperModel
from .hopper_def import (
    HopperParams,
    BODY_TORSO,
    BODY_THIGH,
    BODY_LEG,
    BODY_FOOT,
    JOINT_ROOTX,
    JOINT_ROOTZ,
    JOINT_ROOTY,
    JOINT_THIGH,
    JOINT_LEG,
    JOINT_FOOT,
    TORSO_RADIUS,
    TORSO_HALF_LENGTH,
    THIGH_RADIUS,
    THIGH_HALF_LENGTH,
    LEG_RADIUS,
    LEG_HALF_LENGTH,
    FOOT_RADIUS,
    FOOT_HALF_LENGTH,
    TORSO_MASS,
    THIGH_MASS,
    LEG_MASS,
    FOOT_MASS,
    NQ,
    NV,
    NBODY,
    NJOINT,
    MAX_CONTACTS,
    OBS_DIM,
    ACTION_DIM,
    NUM_BODIES,
    DT,
    FRAME_SKIP,
    EFFECTIVE_DT,
    MAX_STEPS,
    INITIAL_Z,
    FRICTION,
    RESTITUTION,
    FORWARD_REWARD_WEIGHT,
    CTRL_COST_WEIGHT,
    HEALTHY_REWARD,
    RESET_NOISE_SCALE,
    TORQUE_LIMIT,
    THIGH_JOINT_MIN,
    THIGH_JOINT_MAX,
    LEG_JOINT_MIN,
    LEG_JOINT_MAX,
    FOOT_JOINT_MIN,
    FOOT_JOINT_MAX,
)
from .hopper_config import HopperConfig

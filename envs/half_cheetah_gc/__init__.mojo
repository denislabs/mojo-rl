"""HalfCheetahGC Environment Package.

MuJoCo-style Half Cheetah using the physics3d_v2 Generalized Coordinates engine.

The Half Cheetah is a 2D planar robot (movement in XZ plane, rotation around Y axis)
consisting of a torso with two leg chains (front and back).

Components:
- HalfCheetahGC: Main environment struct implementing BoxContinuousActionEnv
- HalfCheetahGCState: 17D observation state (8 qpos + 9 qvel)
- HalfCheetahGCAction: 6D continuous action (joint torques)
- HalfCheetahGCRenderer: 3D visualization using render3d

Example usage:
    from envs.half_cheetah_gc import HalfCheetahGC, HalfCheetahGCAction

    var env = HalfCheetahGC()
    var state = env.reset()

    # Random action
    var action = HalfCheetahGCAction(0.5, -0.3, 0.1, 0.2, -0.4, 0.0)
    var result = env.step(action)
    var next_state = result[0]
    var reward = result[1]
    var done = result[2]
"""

from .half_cheetah_gc import HalfCheetahGC
from .state import HalfCheetahGCState
from .action import HalfCheetahGCAction
from .renderer import HalfCheetahGCRenderer, HalfCheetahGCColors
from .constants_gc import (
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
    # GC dimensions
    NQ,
    NV,
    NBODY,
    NJOINT,
    MAX_CONTACTS,
    OBS_DIM,
    ACTION_DIM,
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
    # Body geometry
    CAPSULE_RADIUS,
    TORSO_HALF_LENGTH,
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
    # Reward parameters
    FORWARD_REWARD_WEIGHT,
    CTRL_COST_WEIGHT,
    # Reset
    RESET_NOISE_SCALE,
)

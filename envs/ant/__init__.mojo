"""Ant Environment Package.

MuJoCo-style Ant using the physics3d Generalized Coordinates engine.

The Ant is a 3D quadruped (free joint root) with 4 legs, each having
a hip hinge and ankle hinge joint. 13 bodies, 9 joints, 8 actuators.

Components:
- Ant: Main environment struct implementing BoxContinuousActionEnv
- ObsState[27]: 27D observation state (13 qpos + 14 qvel) -- from core
- ContAction[8]: 8D continuous action (joint torques) -- from core

Example usage:
    from envs.ant import Ant
    from core import ContAction

    var env = Ant()
    var state = env.reset()

    # Random action (8D)
    var action = ContAction[8]()
    var result = env.step(action)
"""

from .ant import Ant
from .ant_config import AntConfig
from .curriculum import AntCurriculum
from .ant_def import (
    # Model definition
    AntModel,
    AntBodies,
    AntJoints,
    AntActuators,
    # Params struct
    AntParams,
    AntParamsCPU,
    AntParamsGPU,
    # Body type aliases
    AntTorso,
    AntFrontLeftLeg,
    AntAux1,
    AntAnkle1Body,
    AntFrontRightLeg,
    AntAux2,
    AntAnkle2Body,
    AntBackLeg,
    AntAux3,
    AntAnkle3Body,
    AntRightBackLeg,
    AntAux4,
    AntAnkle4Body,
    # Body indices
    BODY_WORLDBODY,
    BODY_TORSO,
    BODY_FRONT_LEFT_LEG,
    BODY_AUX_1,
    BODY_ANKLE_1,
    BODY_FRONT_RIGHT_LEG,
    BODY_AUX_2,
    BODY_ANKLE_2,
    BODY_BACK_LEG,
    BODY_AUX_3,
    BODY_ANKLE_3,
    BODY_RIGHT_BACK_LEG,
    BODY_AUX_4,
    BODY_ANKLE_4,
    # Joint indices
    JOINT_ROOT,
    JOINT_HIP_1,
    JOINT_ANKLE_1,
    JOINT_HIP_2,
    JOINT_ANKLE_2,
    JOINT_HIP_3,
    JOINT_ANKLE_3,
    JOINT_HIP_4,
    JOINT_ANKLE_4,
    # Dimensions
    NQ,
    NV,
    NBODY,
    NJOINT,
    NGEOM,
    MAX_CONTACTS,
    OBS_DIM,
    ACTION_DIM,
)

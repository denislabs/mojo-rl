"""`dm_control` `humanoid` domain.

Tasks: stand, walk, run, run_pure_state.
Reference: references/dm_control-main/dm_control/suite/humanoid.py + .xml
"""

from .humanoid import (
    DMHumanoidStand,
    DMHumanoidStandBatched,
    DMHumanoidWalk,
    DMHumanoidWalkBatched,
    DMHumanoidRun,
    DMHumanoidRunBatched,
    DMHumanoidRunPureState,
    DMHumanoidRunPureStateBatched,
)
from .humanoid_config import (
    DMHumanoidConfig,
    STAND_HEIGHT,
    WALK_SPEED,
    RUN_SPEED,
)
from .humanoid_xml import (
    DMHumanoidModel,
    DMHumanoidPureModel,
    dm_humanoid_xml,
    HUMANOID_OBS_DIM,
    HUMANOID_PURE_OBS_DIM,
    TORSO_BODY_IDX,
    HEAD_BODY_IDX,
    RIGHT_FOOT_BODY_IDX,
    LEFT_FOOT_BODY_IDX,
    RIGHT_HAND_BODY_IDX,
    LEFT_HAND_BODY_IDX,
    N_EXTREMITIES,
    extremity_body_indices,
    ROOT_QPOS_SIZE,
)

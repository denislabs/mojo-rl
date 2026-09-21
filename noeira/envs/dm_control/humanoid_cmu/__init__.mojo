"""`dm_control` `humanoid_CMU` domain.

Tasks: stand, walk, run.
Reference: references/dm_control-main/dm_control/suite/humanoid_CMU.py + .xml

The CMU skeleton: 32 bodies, 57 joints (1 free + 56 hinge), 56 motors. Its
task class is `humanoid`'s with a different body — the reward functions of the
two files are byte-identical — but the accessors underneath are not
interchangeable. See `humanoid_cmu_config`'s module docstring for the three
places they diverge.
"""

from .humanoid_cmu import (
    DMHumanoidCMUStand,
    DMHumanoidCMUStandBatched,
    DMHumanoidCMUWalk,
    DMHumanoidCMUWalkBatched,
    DMHumanoidCMURun,
    DMHumanoidCMURunBatched,
)
from .humanoid_cmu_config import (
    DMHumanoidCMUConfig,
    STAND_HEIGHT,
    WALK_SPEED,
    RUN_SPEED,
    N_ACTUATORS,
)
from .humanoid_cmu_xml import (
    DMHumanoidCMUModel,
    HUMANOID_CMU_OBS_DIM,
    THORAX_BODY_IDX,
    HEAD_BODY_IDX,
    LEFT_HAND_BODY_IDX,
    LEFT_FOOT_BODY_IDX,
    RIGHT_HAND_BODY_IDX,
    RIGHT_FOOT_BODY_IDX,
    N_EXTREMITIES,
    extremity_body_indices,
    ROOT_QPOS_SIZE,
)

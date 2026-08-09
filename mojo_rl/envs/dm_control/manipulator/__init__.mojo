"""`dm_control` `manipulator` domain — all four tasks (Tier C).

    bring_ball   bring_peg   insert_ball   insert_peg

FOUR MODELS, ONE TASK CLASS. `make_model(use_peg, insert)` deletes the prop
bodies each task does not use, which renumbers every body / geom / site after
the arm, so the models cannot be flags on one another; `Bring` itself is a
single task class parameterised by the same two flags, so the config is
`DMManipulatorConfig[USE_PEG, INSERT]`. See `manipulator_xml` for the segment
assembly and the per-variant index tables.

What this domain forced into the engine, none of which existed before it:
  * a site QUATERNION in the model record (`SITE_IDX_QUAT_*`) — its box touch
    zones are the first orientation-dependent site
  * BOX zones in `sensors/touch.mojo`
  * `<inertial>` as a child element in the runtime parser
  * site `pos` / orientation resolved from a DEFAULT CLASS
  * plane ORIENTATION in the narrow phase (bug 34) and the contact DIRECTION
    invariant (bugs 35, 36), both of which its `ncon` assertion caught

and, for the insert tasks, the first COLLIDING mocap body: `cup` and `slot`
are `class="obstacle"` obstacles the prop has to hit, where every mocap body
ported before them was a `contype=0` decoration or a geomless weld anchor.
"""

from .manipulator_xml import (
    dm_manipulator_bring_ball_xml,
    dm_manipulator_bring_peg_xml,
    dm_manipulator_insert_ball_xml,
    dm_manipulator_insert_peg_xml,
    DMManipulatorBringBallModel,
    DMManipulatorBringPegModel,
    DMManipulatorInsertBallModel,
    DMManipulatorInsertPegModel,
    MANIPULATOR_OBS_DIM,
    arm_joint_obs_order,
    touch_site_order,
    target_body_idx,
    receptacle_body_idx,
    site_object,
    site_object_pinch,
    site_object_grasp,
    site_object_tip,
    site_target,
    site_target_tip,
    HAND_BODY_IDX,
    OBJECT_BODY_IDX,
    OBJECT_QADR_X,
    OBJECT_QADR_Z,
    OBJECT_QADR_Y,
    SITE_GRASP,
    SITE_PINCH,
    N_ARM_SITES,
    NARM_JOINTS,
    BALL_BODY_IDX,
    TARGET_BODY_IDX,
    SITE_BALL,
    SITE_TARGET_BALL,
    BALL_QADR_X,
    BALL_QADR_Z,
    BALL_QADR_Y,
    mbp,
    mbpg,
    mib,
    mip,
)
from .manipulator_config import (
    DMManipulatorConfig,
    DMManipulatorBringBallConfig,
    DMManipulatorBringPegConfig,
    DMManipulatorInsertBallConfig,
    DMManipulatorInsertPegConfig,
    CLOSE,
)
from .manipulator import (
    DMManipulatorBringBall,
    DMManipulatorBringPeg,
    DMManipulatorInsertBall,
    DMManipulatorInsertPeg,
)

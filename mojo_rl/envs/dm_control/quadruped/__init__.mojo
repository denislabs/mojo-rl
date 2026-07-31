"""dm_control `quadruped` domain — walk and run (Tier C).

IN PROGRESS. The stripped model (see `quadruped_xml`) parses to MuJoCo's exact
counts and its twelve `<general>` actuators resolve correctly, but the domain
is not yet runnable: there is no env config, no activation state in `Data`,
no accelerometer / force-torque sensors, and no ellipsoid narrow phase for the
torso. See `mojo_rl/envs/ROADMAP.md` for the remaining gap list.
"""

from .quadruped_xml import (
    dm_quadruped_walk_xml,
    dm_quadruped_run_xml,
    DMQuadrupedWalkModel,
    DMQuadrupedRunModel,
    QUADRUPED_OBS_DIM,
    QUADRUPED_WALK_SPEED,
    QUADRUPED_RUN_SPEED,
    TORSO_BODY_IDX,
    qwp,
    qrp,
)

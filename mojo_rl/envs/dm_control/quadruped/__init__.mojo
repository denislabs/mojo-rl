"""dm_control `quadruped` domain — walk and run (Tier C).

    walk = Phyics3dEnv[DMQuadrupedWalkModel, DMQuadrupedWalkConfig]
    run  = Phyics3dEnv[DMQuadrupedRunModel,  DMQuadrupedRunConfig]

The two differ only in the target speed and the floor's half-extent; see
`quadruped_config` for the observation and reward, and `quadruped_xml` for
what `make_model(terrain=False, rangefinders=False, walls_and_ball=False)`
strips.

The first domain here whose observation is mostly `<sensor>` reads — 34 of
its 78 numbers — including the three that need `mj_rnePostConstraint`
(`physics3d/dynamics/rne_post.mojo`). `escape` and `fetch` stay descoped:
they need heightfields, rangefinders and a ball.
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
    TORSO_SITE_IDX,
    TOE_BODY_0,
    TOE_BODY_STRIDE,
    TOE_SITE_0,
    N_HINGE,
    HINGE_QPOS_0,
    HINGE_DOF_0,
    qwp,
    qrp,
)
from .quadruped import DMQuadrupedWalk, DMQuadrupedRun
from .quadruped_config import (
    DMQuadrupedConfig,
    DMQuadrupedWalkConfig,
    DMQuadrupedRunConfig,
    QUADRUPED_MAX_STEPS,
    QUADRUPED_FRAME_SKIP,
)

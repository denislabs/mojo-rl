"""`dm_control` `quadruped` domain — walk, run and fetch.

    walk  = Phyics3dEnv[DMQuadrupedWalkModel,  DMQuadrupedWalkConfig]
    run   = Phyics3dEnv[DMQuadrupedRunModel,   DMQuadrupedRunConfig]
    fetch = Phyics3dEnv[DMQuadrupedFetchModel, DMQuadrupedFetchConfig]

The two differ only in the target speed and the floor's half-extent; see
`quadruped_config` for the observation and reward, and `quadruped_xml` for
what `make_model(terrain=False, rangefinders=False, walls_and_ball=False)`
strips.

The first domain here whose observation is mostly `<sensor>` reads — 34 of
its 78 numbers — including the three that need `mj_rnePostConstraint`
(`physics3d/dynamics/rne_post.mojo`).

`fetch` keeps what walk/run strip — four TILTED PLANE walls, a condim-6 ball
and a target site — and its site ids are therefore NOT walk/run's: `target` is
declared before the torso body and shifts every other site by one. `escape`
stays descoped; it needs heightfields and rangefinders.
"""

from .quadruped_xml import (
    DMQuadrupedWalkModel,
    DMQuadrupedRunModel,
    DMQuadrupedFetchModel,
    QUADRUPED_OBS_DIM,
    QUADRUPED_FETCH_OBS_DIM,
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
    FETCH_TARGET_SITE_IDX,
    FETCH_WORKSPACE_SITE_IDX,
    FETCH_TORSO_SITE_IDX,
    FETCH_TOE_SITE_0,
    FETCH_BALL_BODY_IDX,
    FETCH_BALL_QPOS_0,
    FETCH_BALL_DOF_0,
    qwp,
    qrp,
    qfp,
)
from .quadruped import (
    DMQuadrupedWalk,
    DMQuadrupedRun,
    DMQuadrupedFetch,
    DMQuadrupedEscape,
    DMQuadrupedWalkBatched,
    DMQuadrupedRunBatched,
)
from .quadruped_config import (
    DMQuadrupedConfig,
    DMQuadrupedWalkConfig,
    DMQuadrupedRunConfig,
    QUADRUPED_MAX_STEPS,
    QUADRUPED_FRAME_SKIP,
)
from .quadruped_fetch_config import DMQuadrupedFetchConfig

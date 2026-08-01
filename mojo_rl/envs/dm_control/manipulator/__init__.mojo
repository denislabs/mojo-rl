"""dm_control `manipulator` domain — `bring_ball` (Tier C).

MODEL LAYER ONLY so far. `manipulator_xml` carries the compiled model for the
`bring_ball` task and `tests/dm_control/test_manipulator_vs_dm_control.mojo`
gates its constants against MuJoCo; the env facade (reset, observation,
reward) is not written yet, and neither are the three peg/insert variants,
which are SEPARATE models rather than task flags (`make_model` deletes the
prop bodies each task does not use, which renumbers everything after the arm).

What this domain forced into the engine, none of which existed before it:
  * a site QUATERNION in the model record (`SITE_IDX_QUAT_*`) — its box touch
    zones are the first orientation-dependent site
  * BOX zones in `sensors/touch.mojo`
  * `<inertial>` as a child element in the runtime parser
  * site `pos` / orientation resolved from a DEFAULT CLASS

Still open before the env can run: whether the elliptic cone's SEQUENTIAL
equality post-pass is accurate enough for the `coupling` tendon equality that
keeps the two fingers symmetric — the pyramidal path moved those rows into the
Newton system for exactly this reason.
"""

from .manipulator_xml import (
    dm_manipulator_bring_ball_xml,
    DMManipulatorBringBallModel,
    MANIPULATOR_OBS_DIM,
    mbp,
)

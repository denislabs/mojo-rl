"""`dm_control` `dog` domain — stand, walk, trot and run (Phase 4).

`fetch` is Phase 5 and is not here yet.

`dog.py::make_model` deletes the ball, target, two cameras and four walls for
these four tasks and rewrites the floor half-extent to `move_speed * 15`, so
there are THREE models: stand and walk share 15 (stand uses `_WALK_SPEED`),
trot is 45, run is 135.

⚠ THIS PORT CARRIES A LABELLED DEVIATION — the 162 STL mesh geoms are baked
away into explicit `<inertial>` elements. See `dog_xml.mojo` for the argument
and `tests/dm_control/dog_ref.py::check_bake` for the proof; the parity test
runs that proof as its layer 0.
"""

from .dog_xml import (
    dm_dog_stand_walk_xml,
    dm_dog_trot_xml,
    dm_dog_run_xml,
    DMDogStandWalkModel,
    DMDogTrotModel,
    DMDogRunModel,
    DOG_OBS_DIM,
    DOG_WALK_SPEED,
    DOG_TROT_SPEED,
    DOG_RUN_SPEED,
    DOG_MIN_UPRIGHT_COSINE,
    DOG_STAND_HEIGHT_FRACTION,
    DOG_FRAME_SKIP,
    DOG_MAX_STEPS,
    DOG_N_HINGE,
    DOG_HINGE_QPOS_0,
    DOG_HINGE_DOF_0,
    DOG_TORSO_BODY_IDX,
    DOG_PELVIS_BODY_IDX,
    DOG_SKULL_BODY_IDX,
    DOG_SITE_HEAD,
    DOG_SITE_PALM_L,
    DOG_SITE_PALM_R,
    DOG_SITE_SOLE_L,
    DOG_SITE_SOLE_R,
    DOG_SITE_FOOT_ANCHOR_L,
    DOG_SITE_FOOT_ANCHOR_R,
    DOG_SITE_HAND_ANCHOR_L,
    DOG_SITE_HAND_ANCHOR_R,
    DOG_SITE_UPPER_BITE,
    DOG_SITE_LOWER_BITE,
    DOG_BODY_FOOT_ANCHOR_L,
    DOG_BODY_FOOT_ANCHOR_R,
    DOG_BODY_HAND_ANCHOR_L,
    DOG_BODY_HAND_ANCHOR_R,
    DOG_STAND_HEIGHT_TORSO,
    DOG_STAND_HEIGHT_PELVIS,
    DOG_BODY_WEIGHT,
    dsp,
    dtp,
    drp,
)
from .dog_config import DMDogStandConfig, DMDogMoveConfig
from .dog import DMDogStand, DMDogWalk, DMDogTrot, DMDogRun, DMDogFetch
from .dog import (
    DMDogStandBatched,
    DMDogWalkBatched,
    DMDogTrotBatched,
    DMDogRunBatched,
)
from .dog_fetch_xml import DMDogFetchModel, DOG_FETCH_OBS_DIM, dm_dog_fetch_xml
from .dog_fetch_config import DMDogFetchConfig

"""`dm_control` `stacker` domain — stack_2 and stack_4 (Tier C).

TWO MODELS, ONE TASK CLASS. `make_model(n_boxes)` deletes `box{n..3}`, which
renumbers the target body / geom / site after them, so the models cannot be a
flag on one another; `Stack` itself is one task class parameterised by the same
count, so the config is `DMStackerConfig[N_BOXES]`.

The arm is `manipulator`'s arm, but the FILE is not: upstream keeps two XMLs
that agree on the arena and the arm verbatim and disagree on the geom default's
`solref` (.005 vs .01), so each domain mirrors its own file and only the
non-XML pieces — the index permutations and the reset-time arm geometry — are
shared, from `dm_control/planar_arm.mojo`.

What this domain exercises that `manipulator` did not: BOX props, and with them
box/box and box/capsule contacts as the task's central mechanic rather than an
incidental one. Every prop here is a .022 cube that has to rest stably on the
floor, on the arm, and on another cube, which makes task #42 — our narrow phase
emitting ONE point per colliding geom PAIR where MuJoCo emits up to four —
load bearing rather than a terminal-state detail. See the parity tests, which
measure the gap instead of assuming its size.
"""

from .stacker_xml import (
    DMStacker2Model,
    DMStacker4Model,
    STACK_2_OBS_DIM,
    STACK_4_OBS_DIM,
    BOX_BODY_0,
    BOX_SITE_0,
    BOX_QADR_0,
    BOX_SIZE,
    box_body_idx,
    box_site_idx,
    box_vel_qadr,
    target_body_idx,
    target_site_idx,
    stacker_obs_dim,
    s2,
    s4,
)
from .stacker_config import (
    DMStackerConfig,
    DMStacker2Config,
    DMStacker4Config,
    CLOSE,
    BOX_BOUND_RAD,
)
from .stacker import DMStacker2, DMStacker4

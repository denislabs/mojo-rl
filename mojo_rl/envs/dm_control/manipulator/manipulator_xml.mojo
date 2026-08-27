"""`dm_control` `manipulator` models — port of `dm_control/suite/manipulator.xml`.

FOUR MODELS, NOT ONE. No task uses `manipulator.xml` as written:
`make_model(use_peg, insert)` DELETES the prop bodies the chosen task does not
need, and which bodies go changes every body / geom / site index after the arm.
So the four tasks are four separate models rather than four flags on one:

    bring_ball   ball + target_ball
    bring_peg    peg  + target_peg
    insert_ball  ball + target_ball + cup    (the receptacle)
    insert_peg   peg  + target_peg  + slot

They share the arena, the arm, the tendons, the equality, the sensors and the
actuators verbatim, so those live in the `MANIP_*` segments below and each
model is a concatenation. Writing the arm out four times would let the four
copies drift, and a drift in the SHARED half would look like a variant-specific
physics bug.

⚠ SEGMENT ORDER IS NOT FREE. `make_model` removes bodies from the parsed tree
and leaves the survivors in their original document order, which is

    ball, peg, slot, cup, target_ball, target_peg

so a receptacle always comes BEFORE the target, never after. Assembling
`... + TARGET + RECEPTACLE + ...` would compile, run, and renumber two bodies,
five geoms and three sites against MuJoCo.

Verbatim apart from the `<include>` lines (spliced by `merge_mjcf`), the
per-task deletions, and the render-only `<asset>`/`<visual>` blocks described
under SUBSTITUTIONS.

WHAT THIS DOMAIN NEEDS THAT NO EARLIER ONE DID
----------------------------------------------
  - ORIENTED SITES. `thumb_touch` and `finger_touch` inherit
    `euler="0 15 0"` from `class="hand"`, i.e. quat [.99144, 0, .13053, 0].
    Every previously ported site was either axis-aligned or a sphere, so the
    site record's missing quaternion (and `Data`'s missing `site_xmat`) had
    never been load-bearing. A box zone is orientation-dependent, so it is
    here.
  - BOX touch zones. All five `<touch>` sensors read `type="box"` sites;
    `sensors/touch.mojo` had sphere zones only (with ellipsoid measured AS a
    sphere, which is exact for finger).
  - `<inertial>` as a CHILD ELEMENT. `pinch site` is a massless marker body
    carrying `<inertial mass="1e-6" diaginertia="1e-12 1e-12 1e-12"/>` and no
    geom. Both parsers read `mass`/`diaginertia` off the `<body>` tag only, so
    without it the body's mass defaults rather than being 1e-6 — a ~6e-5
    relative shift in the hand's composite inertia, which is nowhere near a
    1e-9 gate.
  - ELLIPTIC cone WITH a fixed-tendon equality. The `coupling` equality is
    what keeps `finger` and `thumb` symmetric under the single `grasp`
    actuator, and the elliptic path still solves equality rows in a
    SEQUENTIAL post-pass — the same split that cost standing quadruped 45% of
    its qacc before it was moved into the pyramidal Newton system. Measured
    rather than assumed; see the parity tests' population split.
  - A `<motor>` on a TENDON transmission (`grasp`, gear 2). fish has a
    `<position tendon=...>`, so the transmission itself is not new, but not on
    a plain motor.
  - A COLLIDING MOCAP BODY (the two receptacles). Every mocap body ported
    before this was inert — reacher's and finger's targets are `contype=0`
    decorations, SawyerReach's is a weld anchor with no geom at all. `slot`
    and `cup` are `class="obstacle"` with `friction="0"` and default collision
    masks: the peg has to actually hit them. Nothing in the engine special-
    cases a mocap body's geoms (the narrow phase derives geom world poses from
    `xpos`/`xquat`, which `_sync_mocap_to_fields` presets), so this works — but
    it had never been exercised.

SUBSTITUTIONS
-------------
  * The TARGET and, for the insert tasks, the RECEPTACLE become MOCAP BODIES —
    the same workaround reacher and finger needed (gap G4).
    `Bring.initialize_episode` randomises both every episode by writing
    `model.body_pos[...]` and `model.body_quat[...]`, and `fields.Model` is a
    single SHARED, UNBATCHED tensor set, so a model write is a write for every
    env in the batch. A mocap body is the sanctioned alternative: FK skips it
    and the facade presets its world pose from `d.mocap_pos` / `d.mocap_quat`,
    which are per-env `[BATCH, NBODY*k]` state. Neither body has joints, so
    neither contributes a DOF either way and the only thing that changes is
    WHERE the pose lives.
    ⚠ `d.mocap_pos` / `d.mocap_quat` are ZERO after `reset_data` — unlike
    MuJoCo's `mj_resetData`, which seeds them from `body_pos`/`body_quat`. A
    mocap body whose pose no reset hook writes therefore sits at the origin
    with a DEGENERATE (all-zero) quaternion, not at its XML pose. Every config
    here writes both.
  * The model-local `<asset>` (a `background` texture + material) and
    `<visual>` (shadowclip / shadowsize) blocks are dropped, and the
    `background` geom's `material="background"` with them. Both are purely
    cosmetic. The geom ITSELF stays: it is `contype=1 conaffinity=1`, so it
    collides, and it occupies geom index 3.
  * `<default><general ctrllimited="true"/></default>` is written as
    `<motor ctrllimited="true"/>`. `<motor>` is MJCF shorthand for
    `<general>` and MuJoCo applies the `general` default class to it; our
    defaults are keyed by element name, so the shorthand would miss it. Every
    actuator here declares its own `ctrlrange`, and MuJoCo's `autolimits`
    (on by default) infers `ctrllimited` from that anyway, so the attribute is
    redundant in the reference too — the rewrite keeps it honest rather than
    load-bearing.

ORDERING. Our geom/site order is XML text order; MuJoCo's is sorted by body
id. They coincide for all four variants — the four world geoms and `arm_root`
precede the first body, and the props' own sub-bodies never interleave a geom
between a parent's geoms — but the parity tests pin all four orders explicitly
rather than trusting it. The one place the two orders DIVERGE is `palm_touch`,
which is declared after the `pinch site` body yet belongs to `hand`; that swap
is in the ARM, so it is identical in all four variants.

⚠ The prop's slide joints carry `ref` matching their body `pos`
(`ball_x` ref=".4" in a body at x=.4; `peg_x` ref="-.4" at x=-.4), so the
joint VALUE is the world coordinate and `qpos0` is NOT zero. Per bug 18, a
mis-scaled `ref` skews every constraint inverse weight, since those are built
at qpos0.

⚠ `<body name="pinch site">` has a SPACE in its name attribute. Nothing here
looks bodies up by name, but `mj_name2id` on the MuJoCo side needs the space.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.physics3d.types import ConeType


# The arm's index tables are SHARED with `stacker`, whose arm is this one
# verbatim. Re-exported from here so the names this module has always published
# keep resolving; the definitions live in `dm_control/planar_arm.mojo`.
from ..planar_arm import (
    NARM_JOINTS,
    HAND_BODY_IDX,
    SITE_GRASP,
    SITE_PINCH,
    SITE_PALM_TOUCH,
    SITE_THUMB_TOUCH,
    SITE_THUMBTIP_TOUCH,
    SITE_FINGER_TOUCH,
    SITE_FINGERTIP_TOUCH,
    N_ARM_SITES,
    arm_joint_obs_order,
    touch_site_order,
)
from mojo_rl.envs.dm_control.manipulator.manipulator_dims import (
    DM_MANIPULATOR_BRING_BALL_DIMS,
    DM_MANIPULATOR_BRING_PEG_DIMS,
    DM_MANIPULATOR_INSERT_BALL_DIMS,
    DM_MANIPULATOR_INSERT_PEG_DIMS,
)


# ── shared segments ─────────────────────────────────────────────────────────
# Options, defaults, arena and arm — identical in all four tasks. Ends with the
# `<!-- props -->` marker so a prop segment concatenates straight on.



# Receptacles. `euler` here is only the DEFAULT pose: both are mocap bodies, so
# `custom_reset_cpu` overwrites `mocap_quat` every episode with
# `uniform(-pi/3, pi/3)` about y, exactly as `initialize_episode` overwrites
# `model.body_quat`.






# ── shared indices (identical in all four variants) ─────────────────────────
#
# The observation is the same 44 floats for every task: only WHICH bodies,
# joints and sites it reads changes, and the arm half never does.
#
# obs = arm_pos (8 joints x sin/cos = 16) + arm_vel (8) + touch (5)
#     + hand_pos (4) + object_pos (4) + object_vel (3) + target_pos (4) = 44
comptime MANIPULATOR_OBS_DIM: Int = 44

# The prop is always the first body after the arm, and always carries the
# three DOFs that follow the arm's eight.
comptime OBJECT_BODY_IDX: Int = 10
comptime OBJECT_QADR_X: Int = 8
comptime OBJECT_QADR_Z: Int = 9
comptime OBJECT_QADR_Y: Int = 10


# ── per-variant indices ─────────────────────────────────────────────────────
#
# Everything below the arm shifts with which prop bodies survive `make_model`.
# Written as FUNCTIONS of the two task flags rather than four sets of named
# constants, so the config struct — which is parameterised by exactly those two
# flags — can read them directly and the four variants cannot drift apart.
#
# Body layout, world = 0 and the arm occupying 1..9 in every variant:
#
#            bring_ball   bring_peg   insert_ball   insert_peg
#   prop         10          10           10            10
#   pommel        -          11            -            11
#   receptacle    -           -           11            12
#   target       11          12           12            13
def target_body_idx(use_peg: Bool, insert: Bool) -> Int:
    """Body index of `target_ball` / `target_peg`."""
    if use_peg:
        return 13 if insert else 12
    return 12 if insert else 11


def receptacle_body_idx(use_peg: Bool) -> Int:
    """Body index of `cup` / `slot`. Meaningless unless `insert`."""
    return 12 if use_peg else 11


# Site layout, the seven arm sites occupying 0..6 in every variant:
#
#              bring_ball   bring_peg    insert_ball   insert_peg
#   prop           7         7,8,9,10        7          7,8,9,10
#   receptacle     -            -            8           11,12
#   target         8        11,12,13,14      9         13,14,15,16
#
# The prop's four sites are, in order: <name>, <name>_pinch, <name>_grasp,
# <name>_tip — and the target carries the same four with a `target_` prefix.
def site_object(use_peg: Bool) -> Int:
    """`ball` / `peg` — the site the bring reward measures from."""
    return N_ARM_SITES


def site_object_pinch(use_peg: Bool) -> Int:
    """`peg_pinch`. Peg tasks only."""
    return N_ARM_SITES + 1


def site_object_grasp(use_peg: Bool) -> Int:
    """`peg_grasp`. Peg tasks only."""
    return N_ARM_SITES + 2


def site_object_tip(use_peg: Bool) -> Int:
    """`peg_tip`. Peg tasks only."""
    return N_ARM_SITES + 3


def _n_object_sites(use_peg: Bool) -> Int:
    return 4 if use_peg else 1


def _n_receptacle_sites(use_peg: Bool) -> Int:
    """`slot` + `slot_end`, versus `cup` alone."""
    return 2 if use_peg else 1


def site_target(use_peg: Bool, insert: Bool) -> Int:
    """`target_ball` / `target_peg` — the site the bring reward measures to."""
    var s = N_ARM_SITES + _n_object_sites(use_peg)
    if insert:
        s += _n_receptacle_sites(use_peg)
    return s


def site_target_tip(use_peg: Bool, insert: Bool) -> Int:
    """`target_peg_tip`. Peg tasks only; +3 past `target_peg`."""
    return site_target(use_peg, insert) + 3


# ── the four models ─────────────────────────────────────────────────────────





comptime mbp = DM_MANIPULATOR_BRING_BALL_DIMS

comptime mbpg = DM_MANIPULATOR_BRING_PEG_DIMS

comptime mib = DM_MANIPULATOR_INSERT_BALL_DIMS

comptime mip = DM_MANIPULATOR_INSERT_PEG_DIMS

# Legacy names for the bring_ball indices, kept because the task-agnostic
# spellings above read poorly at the bring_ball call sites that predate them.
comptime BALL_BODY_IDX: Int = OBJECT_BODY_IDX
comptime TARGET_BODY_IDX: Int = 11
comptime SITE_BALL: Int = N_ARM_SITES
comptime SITE_TARGET_BALL: Int = N_ARM_SITES + 1
comptime BALL_QADR_X: Int = OBJECT_QADR_X
comptime BALL_QADR_Z: Int = OBJECT_QADR_Z
comptime BALL_QADR_Y: Int = OBJECT_QADR_Y


comptime DMManipulatorBringBallModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/manipulator_bring_ball.xml",
    nbody=mbp.NBODY, njoint=mbp.NJOINT, nq=mbp.NQ, nv=mbp.NV,
    ngeom=mbp.NGEOM, nact=mbp.NACT, ntex=mbp.NTEX, nmat=mbp.NMAT,
    nlight=mbp.NLIGHT, ncam=mbp.NCAM, nsite=mbp.NSITE,
    max_tendon=mbp.NTENDON,
    cone_type=ConeType.ELLIPTIC,
    # A grasped ball touches both palm capsules, both finger links and both
    # thumb links at once, and can additionally rest on the floor or a wall.
    max_contacts=16,
    obs_dim_override=MANIPULATOR_OBS_DIM,
    obs_qpos_skip=0,
    timestep=mbp.TIMESTEP,
]

comptime DMManipulatorBringPegModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/manipulator_bring_peg.xml",
    nbody=mbpg.NBODY, njoint=mbpg.NJOINT, nq=mbpg.NQ, nv=mbpg.NV,
    ngeom=mbpg.NGEOM, nact=mbpg.NACT, ntex=mbpg.NTEX, nmat=mbpg.NMAT,
    nlight=mbpg.NLIGHT, ncam=mbpg.NCAM, nsite=mbpg.NSITE,
    max_tendon=mbpg.NTENDON,
    cone_type=ConeType.ELLIPTIC,
    # The peg is THREE colliding geoms (blade, guard, pommel) against the same
    # eleven arm geoms plus the floor and two walls, so it reaches much further
    # into the contact table than the ball does. MEASURED by sweeping MuJoCo
    # over the grasp: 21 simultaneous contacts at a hard closed-hand pose,
    # against the ball's 9. 32 leaves half again on top of that, and the parity
    # test's `our ncon == MuJoCo ncon` assertion fails loudly rather than
    # truncating silently if it is ever short.
    max_contacts=32,
    obs_dim_override=MANIPULATOR_OBS_DIM,
    obs_qpos_skip=0,
    timestep=mbpg.TIMESTEP,
]

comptime DMManipulatorInsertBallModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/manipulator_insert_ball.xml",
    nbody=mib.NBODY, njoint=mib.NJOINT, nq=mib.NQ, nv=mib.NV,
    ngeom=mib.NGEOM, nact=mib.NACT, ntex=mib.NTEX, nmat=mib.NMAT,
    nlight=mib.NLIGHT, ncam=mib.NCAM, nsite=mib.NSITE,
    max_tendon=mib.NTENDON,
    cone_type=ConeType.ELLIPTIC,
    # bring_ball's 16 plus the four `cup` capsules, which the ball can touch
    # simultaneously once it is seated.
    max_contacts=20,
    obs_dim_override=MANIPULATOR_OBS_DIM,
    obs_qpos_skip=0,
    timestep=mib.TIMESTEP,
]

comptime DMManipulatorInsertPegModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/manipulator_insert_peg.xml",
    nbody=mip.NBODY, njoint=mip.NJOINT, nq=mip.NQ, nv=mip.NV,
    ngeom=mip.NGEOM, nact=mip.NACT, ntex=mip.NTEX, nmat=mip.NMAT,
    nlight=mip.NLIGHT, ncam=mip.NCAM, nsite=mip.NSITE,
    max_tendon=mip.NTENDON,
    cone_type=ConeType.ELLIPTIC,
    # The worst case in the port: three peg geoms against three `slot` boxes is
    # nine pairs on its own, on top of bring_peg's measured 21 hand contacts.
    max_contacts=40,
    obs_dim_override=MANIPULATOR_OBS_DIM,
    obs_qpos_skip=0,
    timestep=mip.TIMESTEP,
]

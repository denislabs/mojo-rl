"""`dm_control` `stacker` models — port of `dm_control/suite/stacker.xml`.

TWO MODELS, NOT ONE. As with `manipulator`, no task uses `stacker.xml` as
written: `make_model(n_boxes)` DELETES `box{n_boxes..3}` from the parsed tree,
and since the boxes come before the target, deleting them renumbers the target
body, its geom, its site, and nothing else. So

    stack_2   box0, box1               + target
    stack_4   box0, box1, box2, box3   + target

The arena, the arm, the tendons, the equality, the sensors and the actuators are
identical between the two and live in the `STACK_*` segments below, each model
being a concatenation.

⚠ THE ARM IS THE `manipulator` ARM, BUT THE FILE IS NOT. `stacker.xml` and
`manipulator.xml` are two upstream files that agree on the arena and the arm
verbatim and disagree on exactly one thing: the geom default's contact softness,

    manipulator   <geom friction=".7" solimp="0.9 0.97 0.001" solref=".005 1"/>
    stacker       <geom friction=".7" solimp="0.9 0.97 0.001" solref=".01 1"/>

which doubles this domain's contact time constant. Both copies are therefore
kept, each mirroring its own upstream file, so an upstream edit to one cannot
leak into the other. What the two domains DO share — the index permutations and
the reset-time arm geometry, none of which is XML — lives in
`dm_control/planar_arm.mojo`.

WHAT THIS DOMAIN NEEDS THAT `manipulator` DID NOT
-------------------------------------------------
  - BOX PROPS, and therefore box/box and box/capsule contacts as the task's
    CENTRAL mechanic rather than an incidental one. `manipulator`'s props are a
    sphere and a capsule triple; every one of stacker's is a .022 cube that has
    to rest stably on the floor, on the arm, and on another cube. Task #42
    (our narrow phase emits ONE point per colliding geom PAIR where MuJoCo's
    `mjc_CapsuleBox` / `mjc_BoxBox` emit up to two / four) is consequently load
    bearing here in a way it was not for insert_peg, where it only cost a
    terminal-state pivot. `box_plane` already reports up to four points, so a
    box resting on the FLOOR is faithful; a box resting on another box is not.
    The parity tests measure this rather than assume it.
  - FOUR free-floating props at once (stack_4), against manipulator's one.

SUBSTITUTIONS (identical in kind to `manipulator`'s)
----------------------------------------------------
  * The TARGET becomes a MOCAP BODY. `Stack.initialize_episode` randomises it
    every episode by writing `model.body_pos['target', 'x']` and
    `['target', 'z']`, and `fields.Model` is a single SHARED, UNBATCHED tensor
    set, so a model write is a write for every env in the batch. A mocap body is
    the sanctioned alternative (gap G4): FK skips it and the facade presets its
    world pose from `d.mocap_pos` / `d.mocap_quat`, which are per-env state. The
    target has no joints, so it contributes no DOF either way.
    ⚠ `d.mocap_pos` / `d.mocap_quat` are ZERO after `reset_data`, unlike
    MuJoCo's `mj_resetData` which seeds them from `body_pos`/`body_quat`. A
    mocap body no reset hook writes therefore sits at the origin with a
    DEGENERATE all-zero quaternion, not at its XML pose. The config writes both.
    Unlike manipulator's, this target is never ROTATED — `initialize_episode`
    writes only the two position components — so its quaternion stays identity.
  * The model-local `<asset>` (a `background` texture + material) and `<visual>`
    (shadowclip / shadowsize) blocks are dropped, and the `background` geom's
    `material="background"` with them. All three are purely cosmetic. The geom
    ITSELF stays: it is `contype=1 conaffinity=1`, so it collides, and it
    occupies geom index 3.
    (Dropping the `<visual>` block also drops the stray `>` that follows its
    closing tag in the reference — `</visual>>`. It is legal XML, since a bare
    `>` is permitted in character data, and it parses as text belonging to
    `<mujoco>`; it is upstream noise, not a construct we need to support.)
  * `<default><general ctrllimited="true"/></default>` is written as
    `<motor ctrllimited="true"/>`. `<motor>` is MJCF shorthand for `<general>`
    and MuJoCo applies the `general` default class to it; our defaults are keyed
    by element name, so the shorthand would miss it. Every actuator here
    declares its own `ctrlrange` and MuJoCo's `autolimits` infers `ctrllimited`
    from that anyway, so the attribute is redundant in the reference too.

⚠ Each box's slide joints carry `ref` matching their body `pos` (`box0_x`
ref=".5" in a body at x=.5), so the joint VALUE is the world coordinate and
`qpos0` is NOT zero. That is what lets `initialize_episode` write
`qpos['box0_x'] = uniform(.1, .3)` and mean world x directly. Per bug 18, a
mis-scaled `ref` skews every constraint inverse weight, since those are built at
qpos0 — so the four boxes' four DIFFERENT x refs are worth checking, not just
the shape of the table.

⚠ `<body name="pinch site">` has a SPACE in its name attribute. Nothing here
looks bodies up by name, but `mj_name2id` on the MuJoCo side needs the space.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.physics3d.types import ConeType

from ..planar_arm import NARM_JOINTS, N_ARM_SITES
from mojo_rl.envs.dm_control.stacker.stacker_dims import (
    DM_STACKER_2_DIMS,
    DM_STACKER_4_DIMS,
)


# ── shared segments ─────────────────────────────────────────────────────────
# Options, defaults, arena and arm — identical in both tasks. Ends with the
# `<!-- props -->` marker so a box segment concatenates straight on.

# The four boxes. Identical apart from name, body `pos` and the matching `ref`
# on the two slide joints — which is why `ref` is worth reading twice: box0 is
# at x=.5 with ref=".5", box3 at x=.2 with ref=".2", and a copy-paste that left
# all four at ".5" would put three boxes' qpos0 in the wrong place without
# changing where they START.




# `initialize_episode` overwrites x and z every episode; y stays at the .001 the
# XML gives it, which is what puts the ghost marginally in front of the boxes.



# ── indices ─────────────────────────────────────────────────────────────────
#
# The arm occupies bodies 1..9, joints 0..7 and sites 0..6 in both models; only
# the boxes and the target move with `n_boxes`.
#
#   bodies    world 0 | arm 1..9 | box_i = 10 + i | target = 10 + n
#   joints    arm 0..7 | box_i's x, z, y = 8 + 3i, 9 + 3i, 10 + 3i
#   sites     arm 0..6 | box_i = 7 + i | target = 7 + n
comptime BOX_BODY_0: Int = 10
comptime BOX_SITE_0: Int = N_ARM_SITES
comptime BOX_QADR_0: Int = NARM_JOINTS

# Half-extent of every box geom AND of the target geom. The reward reads it off
# the TARGET (`geom_size['target', 0]`), which is the same .022.
comptime BOX_SIZE: Float64 = 0.022


def box_body_idx(i: Int) -> Int:
    return BOX_BODY_0 + i


def box_site_idx(i: Int) -> Int:
    return BOX_SITE_0 + i


def target_body_idx(n_boxes: Int) -> Int:
    return BOX_BODY_0 + n_boxes


def target_site_idx(n_boxes: Int) -> Int:
    return BOX_SITE_0 + n_boxes


# ⚠ `Stack` builds its box joint names as
#     for name in box_names: for dim in 'xyz': ...
# so `box_vel` is ordered x, y, z — while the MODEL declares x, z, y. Entries
# 1 and 2 of every box's velocity triple are therefore TRANSPOSED against the
# qvel layout. `manipulator` iterates 'xzy' and needs no such permutation, so
# this is a difference between two otherwise parallel domains, not a shared
# convention. A box moving in one axis at a time hides it; the parity tests
# drive all three at once.
def box_vel_qadr(i: Int, k: Int) -> Int:
    """`qvel` index of the k'th entry of box `i`'s observed velocity triple."""
    var base = BOX_QADR_0 + 3 * i
    if k == 0:
        return base + 0  # x
    if k == 1:
        return base + 2  # y  (the hinge, declared THIRD)
    return base + 1  # z  (declared SECOND)


def stacker_obs_dim(n_boxes: Int) -> Int:
    """`arm_pos(16) + arm_vel(8) + touch(5) + hand_pos(4)
    + box_pos(4n) + box_vel(3n) + target_pos(2)`."""
    return 2 * NARM_JOINTS + NARM_JOINTS + 5 + 4 + 7 * n_boxes + 2


# ── the two models ──────────────────────────────────────────────────────────



comptime s2 = DM_STACKER_2_DIMS

comptime s4 = DM_STACKER_4_DIMS

comptime STACK_2_OBS_DIM: Int = stacker_obs_dim(2)
comptime STACK_4_OBS_DIM: Int = stacker_obs_dim(4)


comptime DMStacker2Model = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/stacker_2.xml",
    nbody=s2.NBODY, njoint=s2.NJOINT, nq=s2.NQ, nv=s2.NV,
    ngeom=s2.NGEOM, nact=s2.NACT, ntex=s2.NTEX, nmat=s2.NMAT,
    nlight=s2.NLIGHT, ncam=s2.NCAM, nsite=s2.NSITE,
    max_tendon=s2.NTENDON,
    cone_type=ConeType.ELLIPTIC,
    # A cube resting on the floor is four points on its own (`box_plane` reports
    # all four), so two boxes down flat is eight before the arm touches
    # anything. A grasped cube adds both palms, both finger links and both thumb
    # links; a box on a box adds their pair. MEASURED against MuJoCo by the
    # parity tests, whose `our ncon == MuJoCo ncon` assertion fails loudly
    # rather than truncating silently if this is ever short.
    max_contacts=24,
    obs_dim_override=STACK_2_OBS_DIM,
    obs_qpos_skip=0,
    timestep=s2.TIMESTEP,
]

comptime DMStacker4Model = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/stacker_4.xml",
    nbody=s4.NBODY, njoint=s4.NJOINT, nq=s4.NQ, nv=s4.NV,
    ngeom=s4.NGEOM, nact=s4.NACT, ntex=s4.NTEX, nmat=s4.NMAT,
    nlight=s4.NLIGHT, ncam=s4.NCAM, nsite=s4.NSITE,
    max_tendon=s4.NTENDON,
    cone_type=ConeType.ELLIPTIC,
    # Four boxes flat on the floor is sixteen points before anything else
    # happens, and they can additionally stack, touch each other side by side,
    # and be held. Twice stack_2's budget.
    max_contacts=48,
    obs_dim_override=STACK_4_OBS_DIM,
    obs_qpos_skip=0,
    timestep=s4.TIMESTEP,
]

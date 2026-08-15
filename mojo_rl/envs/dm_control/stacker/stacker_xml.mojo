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
from mojo_rl.physics3d.parser.xml_parser import merge_mjcf
from mojo_rl.physics3d.types import ConeType

from ..common_xml import dm_visual_xml, dm_skybox_xml, dm_materials_xml
from ..planar_arm import NARM_JOINTS, N_ARM_SITES
from mojo_rl.envs.dm_control.stacker.stacker_dims import (
    DM_STACKER_2_DIMS,
    DM_STACKER_4_DIMS,
)


# ── shared segments ─────────────────────────────────────────────────────────
# Options, defaults, arena and arm — identical in both tasks. Ends with the
# `<!-- props -->` marker so a box segment concatenates straight on.
comptime STACK_HEAD = """
<mujoco model="planar stacker">

  <option timestep="0.001" cone="elliptic"/>

  <default>
    <geom friction=".7" solimp="0.9 0.97 0.001" solref=".01 1"/>
    <joint solimplimit="0 0.99 0.01" solreflimit=".005 1"/>
    <motor ctrllimited="true"/>
    <tendon width="0.01"/>
    <site size=".003 .003 .003" material="site" group="3"/>

    <default class="arm">
      <geom type="capsule" material="self" density="500"/>
      <joint type="hinge" pos="0 0 0" axis="0 -1 0" limited="true"/>
      <default class="hand">
        <joint damping=".5" range="-10 60"/>
        <geom size=".008"/>
        <site  type="box" size=".018 .005 .005" pos=".022 0 -.002" euler="0 15 0" group="4"/>
        <default class="fingertip">
          <geom type="sphere" size=".008" material="effector"/>
          <joint damping=".01" stiffness=".01" range="-40 20"/>
          <site  size=".012 .005 .008" pos=".003 0 .003" group="4" euler="0 0 0"/>
        </default>
      </default>
    </default>

    <default class="object">
      <geom material="self"/>
    </default>

    <default class="task">
      <site rgba="0 0 0 0"/>
    </default>

    <default class="obstacle">
      <geom material="decoration" friction="0"/>
    </default>

    <default class="ghost">
      <geom material="target" contype="0" conaffinity="0"/>
    </default>
  </default>

  <worldbody>
    <!-- Arena -->
    <light name="light" directional="true" diffuse=".6 .6 .6" pos="0 0 1" specular=".3 .3 .3"/>
    <geom name="floor" type="plane" pos="0 0 0" size=".4 .2 10" material="grid"/>
    <geom name="wall1" type="plane" pos="-.682843 0 .282843" size=".4 .2 10" material="grid" zaxis="1 0 1"/>
    <geom name="wall2" type="plane" pos=".682843 0 .282843" size=".4 .2 10" material="grid" zaxis="-1 0 1"/>
    <geom name="background" type="plane" pos="0 .2 .5" size="1 .5 10" zaxis="0 -1 0"/>
    <camera name="fixed" pos="0 -16 .4" xyaxes="1 0 0 0 0 1" fovy="4"/>

    <!-- Arm -->
    <geom name="arm_root" type="cylinder" fromto="0 -.022 .4 0 .022 .4" size=".024"
          material="decoration" contype="0" conaffinity="0"/>
    <body name="upper_arm" pos="0 0 .4" childclass="arm">
      <joint name="arm_root" damping="2" limited="false"/>
      <geom  name="upper_arm"  size=".02" fromto="0 0 0 0 0 .18"/>
      <body  name="middle_arm" pos="0 0 .18" childclass="arm">
        <joint name="arm_shoulder" damping="1.5" range="-160 160"/>
        <geom  name="middle_arm"  size=".017" fromto="0 0 0 0 0 .15"/>
        <body  name="lower_arm" pos="0 0 .15">
          <joint name="arm_elbow" damping="1" range="-160 160"/>
          <geom  name="lower_arm" size=".014" fromto="0 0 0 0 0 .12"/>
          <body  name="hand" pos="0 0 .12">
            <joint name="arm_wrist" damping=".5" range="-140 140" />
            <geom  name="hand" size=".011" fromto="0 0 0 0 0 .03"/>
            <geom  name="palm1"  fromto="0 0 .03  .03 0 .045" class="hand"/>
            <geom  name="palm2"  fromto="0 0 .03 -.03 0 .045" class="hand"/>
            <site  name="grasp" pos="0 0 .065"/>
            <body  name="pinch site" pos="0 0 .090">
              <site  name="pinch"/>
              <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-12 1e-12 1e-12"/>
              <camera name="hand" pos="0 -.3 0" xyaxes="1 0 0 0 0 1" mode="track"/>
            </body>
            <site  name="palm_touch" type="box" group="4" size=".025 .005 .008" pos="0 0 .043"/>

            <body name="thumb" pos=".03 0 .045" euler="0 -90 0" childclass="hand">
              <joint name="thumb"/>
              <geom  name="thumb1"  fromto="0 0 0 .02 0 -.01" size=".007"/>
              <geom  name="thumb2"  fromto=".02 0 -.01 .04 0 -.01" size=".007"/>
              <site  name="thumb_touch" group="4"/>
              <body  name="thumbtip" pos=".05 0 -.01" childclass="fingertip">
                <joint name="thumbtip"/>
                <geom  name="thumbtip1" pos="-.003 0 0" />
                <geom  name="thumbtip2" pos=".003 0 0" />
                <site  name="thumbtip_touch" group="4"/>
              </body>
            </body>

            <body name="finger" pos="-.03 0 .045" euler="0 90 180" childclass="hand">
              <joint name="finger"/>
              <geom  name="finger1"  fromto="0 0 0 .02 0 -.01" size=".007" />
              <geom  name="finger2"  fromto=".02 0 -.01 .04 0 -.01" size=".007"/>
              <site  name="finger_touch"/>
              <body  name="fingertip" pos=".05 0 -.01" childclass="fingertip">
                <joint name="fingertip"/>
                <geom  name="fingertip1" pos="-.003 0 0" />
                <geom  name="fingertip2" pos=".003 0 0" />
                <site  name="fingertip_touch"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- props -->
"""

# The four boxes. Identical apart from name, body `pos` and the matching `ref`
# on the two slide joints — which is why `ref` is worth reading twice: box0 is
# at x=.5 with ref=".5", box3 at x=.2 with ref=".2", and a copy-paste that left
# all four at ".5" would put three boxes' qpos0 in the wrong place without
# changing where they START.
comptime STACK_BOX_0 = """
    <body name="box0" pos=".5 0 .4" childclass="object">
      <joint name="box0_x" type="slide" axis="1 0 0" ref=".5"/>
      <joint name="box0_z" type="slide" axis="0 0 1" ref=".4"/>
      <joint name="box0_y" type="hinge" axis="0 1 0"/>
      <geom  name="box0" type="box" size=".022 .022 .022" />
      <site  name="box0" type="sphere"/>
    </body>
"""

comptime STACK_BOX_1 = """
    <body name="box1" pos=".4 0 .4" childclass="object">
      <joint name="box1_x" type="slide" axis="1 0 0" ref=".4"/>
      <joint name="box1_z" type="slide" axis="0 0 1" ref=".4"/>
      <joint name="box1_y" type="hinge" axis="0 1 0"/>
      <geom  name="box1" type="box" size=".022 .022 .022" />
      <site  name="box1" type="sphere"/>
    </body>
"""

comptime STACK_BOX_2 = """
    <body name="box2" pos=".3 0 .4" childclass="object">
      <joint name="box2_x" type="slide" axis="1 0 0" ref=".3"/>
      <joint name="box2_z" type="slide" axis="0 0 1" ref=".4"/>
      <joint name="box2_y" type="hinge" axis="0 1 0"/>
      <geom  name="box2" type="box" size=".022 .022 .022" />
      <site  name="box2" type="sphere"/>
    </body>
"""

comptime STACK_BOX_3 = """
    <body name="box3" pos=".2 0 .4" childclass="object">
      <joint name="box3_x" type="slide" axis="1 0 0" ref=".2"/>
      <joint name="box3_z" type="slide" axis="0 0 1" ref=".4"/>
      <joint name="box3_y" type="hinge" axis="0 1 0"/>
      <geom  name="box3" type="box" size=".022 .022 .022" />
      <site  name="box3" type="sphere"/>
    </body>
"""

# `initialize_episode` overwrites x and z every episode; y stays at the .001 the
# XML gives it, which is what puts the ghost marginally in front of the boxes.
comptime STACK_TARGET = """
    <!-- targets -->
    <body name="target" pos="0 .001 .022" childclass="ghost" mocap="true">
      <geom  name="target" type="box" size=".022 .022 .022" />
      <site  name="target" type="sphere"/>
    </body>
"""

comptime STACK_TAIL = """
  </worldbody>

  <tendon>
    <fixed name="grasp">
      <joint joint="thumb"  coef=".5"/>
      <joint joint="finger" coef=".5"/>
    </fixed>
    <fixed name="coupling">
      <joint joint="thumb"  coef="-.5"/>
      <joint joint="finger" coef=".5"/>
    </fixed>
  </tendon>

  <equality>
    <tendon name="coupling" tendon1="coupling" solimp="0.95 0.99 0.001" solref=".005 .5"/>
  </equality>

  <sensor>
    <touch name="palm_touch" site="palm_touch"/>
    <touch name="finger_touch" site="finger_touch"/>
    <touch name="thumb_touch" site="thumb_touch"/>
    <touch name="fingertip_touch" site="fingertip_touch"/>
    <touch name="thumbtip_touch" site="thumbtip_touch"/>
  </sensor>

  <actuator>
    <motor name="root"     joint="arm_root"     ctrlrange="-1 1"  gear="12"/>
    <motor name="shoulder" joint="arm_shoulder" ctrlrange="-1 1"  gear="8"/>
    <motor name="elbow"    joint="arm_elbow"    ctrlrange="-1 1"  gear="4"/>
    <motor name="wrist"    joint="arm_wrist"    ctrlrange="-1 1"  gear="2"/>
    <motor name="grasp"    tendon="grasp"       ctrlrange="-1 1"  gear="2"/>
  </actuator>

</mujoco>
"""


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

comptime dm_stacker_2_xml = merge_mjcf(
    dm_visual_xml,
    dm_skybox_xml,
    dm_materials_xml,
    STACK_HEAD + STACK_BOX_0 + STACK_BOX_1 + STACK_TARGET + STACK_TAIL,
)

comptime dm_stacker_4_xml = merge_mjcf(
    dm_visual_xml,
    dm_skybox_xml,
    dm_materials_xml,
    STACK_HEAD
    + STACK_BOX_0
    + STACK_BOX_1
    + STACK_BOX_2
    + STACK_BOX_3
    + STACK_TARGET
    + STACK_TAIL,
)

comptime s2 = DM_STACKER_2_DIMS

comptime s4 = DM_STACKER_4_DIMS

comptime STACK_2_OBS_DIM: Int = stacker_obs_dim(2)
comptime STACK_4_OBS_DIM: Int = stacker_obs_dim(4)


comptime DMStacker2Model = ModelDefFromXML[
    xml=dm_stacker_2_xml,
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
    xml=dm_stacker_4_xml,
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

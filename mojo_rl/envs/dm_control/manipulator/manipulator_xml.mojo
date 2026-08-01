"""dm_control `manipulator` model — port of `dm_control/suite/manipulator.xml`.

Scoped to the `bring_ball` task, which is the one `@SUITE.add('benchmarking')`
tags. `make_model(use_peg=False, insert=False)` keeps `ball` + `target_ball`
and DELETES the `peg`, `target_peg`, `cup` and `slot` bodies outright, so they
are simply absent below rather than commented out — the deletion changes every
body/geom/site index after the arm, and carrying dead bodies would silently
shift them. `bring_peg` / `insert_ball` / `insert_peg` are separate models for
the same reason and are not ported here.

Verbatim apart from the `<include>` lines (spliced by `merge_mjcf`), the
deleted props, and the render-only `<asset>`/`<visual>` blocks described under
SUBSTITUTIONS.

WHAT THIS MODEL NEEDS THAT NO EARLIER DOMAIN DID
------------------------------------------------
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
    rather than assumed; see the parity test's population split.
  - A `<motor>` on a TENDON transmission (`grasp`, gear 2). fish has a
    `<position tendon=...>`, so the transmission itself is not new, but not on
    a plain motor.

SUBSTITUTIONS
-------------
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
id. They coincide here — the four world geoms and `arm_root` precede the first
body, and no later body interleaves — but the parity test pins all four orders
explicitly rather than trusting it.

⚠ `ball_x` and `ball_z` carry `ref=".4"`, so the ball's qpos0 is NOT zero:
`qpos0 = [0,0,0,0,0,0,0,0, .4, .4, 0]`. Per bug 18, a mis-scaled `ref` skews
every constraint inverse weight, since those are built at qpos0.

⚠ `<body name="pinch site">` has a SPACE in its name attribute. Nothing here
looks bodies up by name, but `mj_name2id` on the MuJoCo side needs the space.
"""

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.parser.xml_parser import merge_mjcf
from mojo_rl.physics3d.types import ConeType

from ..common_xml import dm_visual_xml, dm_skybox_xml, dm_materials_xml


comptime _manipulator_body = """
<mujoco model="planar manipulator">

  <option timestep="0.001" cone="elliptic"/>

  <default>
    <geom friction=".7" solimp="0.9 0.97 0.001" solref=".005 1"/>
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
    <body name="ball" pos=".4 0 .4" childclass="object">
      <joint name="ball_x" type="slide" axis="1 0 0" ref=".4"/>
      <joint name="ball_z" type="slide" axis="0 0 1" ref=".4"/>
      <joint name="ball_y" type="hinge" axis="0 1 0"/>
      <geom  name="ball" type="sphere" size=".022" />
      <site  name="ball" type="sphere"/>
    </body>

    <!-- targets -->
    <body name="target_ball" pos=".4 .001 .4" childclass="ghost">
      <geom  name="target_ball" type="sphere" size=".02" />
      <site  name="target_ball" type="sphere"/>
    </body>

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


comptime dm_manipulator_bring_ball_xml = merge_mjcf(
    dm_visual_xml, dm_skybox_xml, dm_materials_xml, _manipulator_body
)

comptime mbp = parse_xml(dm_manipulator_bring_ball_xml)

# obs = arm_pos (8 joints x sin/cos = 16) + arm_vel (8) + touch (5)
#     + hand_pos (4) + object_pos (4) + object_vel (3) + target_pos (4) = 44
comptime MANIPULATOR_OBS_DIM: Int = 44

comptime DMManipulatorBringBallModel = ModelDefFromXML[
    xml=dm_manipulator_bring_ball_xml,
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

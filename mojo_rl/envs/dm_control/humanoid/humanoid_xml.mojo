"""`dm_control` `humanoid` model — port of `dm_control/suite/humanoid.xml`.

Verbatim apart from the `<include>` lines and the `<sensor>` block.

THE SENSOR BLOCK IS DROPPED, and that is not a shortcut. `humanoid.xml`
declares 18 `touch`, 6 `force`, 6 `torque`, an accelerometer, a velocimeter
and a gyro — and NONE of the four tasks reads any of them. `grep sensordata
humanoid.py` returns exactly one line, `torso_subtreelinvel`, which we compute
directly from `Data.xvel` via `sensors.subtree_linvel`. (`merge_mjcf` does not
carry a `<sensor>` section anyway, so keeping the block would have been
decorative.) Every SITE stays, so `nsite` still matches MuJoCo's 25 and a
future `touch` port has its zones.

WHAT THIS MODEL EXERCISES that no earlier ported domain does:

  * `<freejoint name="root"/>`. MJCF sugar for `<joint type="free">`; our
    scanners look for the literal `"<joint"` in ~20 places, so `merge_mjcf`
    now normalizes the alias textually before anything scans. Without it the
    torso welds to the world and nq/nv come out 7/6 short — silently, since an
    unrecognized element is not an error.
  * JOINT SPRINGS. Every `<joint>` inherits `stiffness="1"` from
    `<default class="body">`, with 5/10/20 on the big joints and 3/6 on the
    ankles. The integrators have always assembled `fnet = qfrc - bias -
    damping - stiffness - frictionloss`, but our Gym humanoid sets
    `stiffness="0"` everywhere, so this is the first model that actually
    loads that term.
  * THREE-DEEP nested default classes (`body` > `big_joint` >
    `big_stiff_joint`) plus `childclass="body"` on the torso.

Body order is the tree DFS in both engines, so our indices match MuJoCo's
here — unlike the GEOM order, which still differs (see `point_mass_xml`).
"""

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.parser.xml_parser import merge_mjcf

from ..common_xml import dm_visual_xml, dm_skybox_xml, dm_materials_xml


comptime _humanoid_body = """
  <statistic extent="2" center="0 0 1"/>

  <option timestep=".005"/>

  <default>
    <motor ctrlrange="-1 1" ctrllimited="true"/>
    <default class="body">
      <geom  type="capsule" condim="1" friction=".7" solimp=".9 .99 .003" solref=".015 1" material="self"/>
      <joint type="hinge" damping=".2" stiffness="1" armature=".01" limited="true" solimplimit="0 .99 .01"/>
      <default class="big_joint">
        <joint damping="5" stiffness="10"/>
        <default class="big_stiff_joint">
          <joint stiffness="20"/>
        </default>
      </default>
      <site size=".04" group="3"/>
      <default class="force-torque">
        <site type="box" size=".01 .01 .02" rgba="1 0 0 1" />
      </default>
      <default class="touch">
        <site type="capsule" rgba="0 0 1 .3"/>
      </default>
    </default>
  </default>

  <worldbody>
    <geom name="floor" type="plane" conaffinity="1" size="100 100 .2" material="grid"/>
    <body name="torso" pos="0 0 1.5" childclass="body">
      <light name="top" pos="0 0 2" mode="trackcom"/>
      <camera name="back" pos="-3 0 1" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
      <camera name="side" pos="0 -3 1" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
      <freejoint name="root"/>
      <site name="root" class="force-torque"/>
      <geom name="torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
      <geom name="upper_waist" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
      <site name="torso" class="touch" type="box" pos="0 0 -.05" size=".075 .14 .13"/>
      <body name="head" pos="0 0 .19">
        <geom name="head" type="sphere" size=".09"/>
        <site name="head" class="touch" type="sphere" size=".091"/>
        <camera name="egocentric" pos=".09 0 0" xyaxes="0 -1 0 .1 0 1" fovy="80"/>
      </body>
      <body name="lower_waist" pos="-.01 0 -.260" quat="1.000 0 -.002 0">
        <geom name="lower_waist" fromto="0 -.06 0 0 .06 0" size=".06"/>
        <site name="lower_waist" class="touch" size=".061 .06" zaxis="0 1 0"/>
        <joint name="abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="big_stiff_joint"/>
        <joint name="abdomen_y" pos="0 0 .065" axis="0 1 0" range="-75 30" class="big_joint"/>
        <body name="pelvis" pos="0 0 -.165" quat="1.000 0 -.002 0">
          <joint name="abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="big_joint"/>
          <geom name="butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
          <site name="butt" class="touch" size=".091 .07" pos="-.02 0 0" zaxis="0 1 0"/>
          <body name="right_thigh" pos="0 -.1 -.04">
            <site name="right_hip" class="force-torque"/>
            <joint name="right_hip_x" axis="1 0 0" range="-25 5"   class="big_joint"/>
            <joint name="right_hip_z" axis="0 0 1" range="-60 35"  class="big_joint"/>
            <joint name="right_hip_y" axis="0 1 0" range="-110 20" class="big_stiff_joint"/>
            <geom name="right_thigh" fromto="0 0 0 0 .01 -.34" size=".06"/>
            <site name="right_thigh" class="touch" pos="0 .005 -.17" size=".061 .17" zaxis="0 -1 34"/>
            <body name="right_shin" pos="0 .01 -.403">
              <site name="right_knee" class="force-torque" pos="0 0 .02"/>
              <joint name="right_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="right_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <site name="right_shin" class="touch" pos="0 0 -.15" size=".05 .15"/>
              <body name="right_foot" pos="0 0 -.39">
                <site name="right_ankle" class="force-torque"/>
                <joint name="right_ankle_y" pos="0 0 .08" axis="0 1 0"   range="-50 50" stiffness="6"/>
                <joint name="right_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="right_right_foot" fromto="-.07 -.02 0 .14 -.04 0" size=".027"/>
                <geom name="left_right_foot" fromto="-.07 0 0 .14  .02 0" size=".027"/>
                <site name="right_right_foot" class="touch" pos=".035 -.03 0" size=".03 .11" zaxis="21 -2 0"/>
                <site name="left_right_foot" class="touch" pos=".035 .01 0" size=".03 .11" zaxis="21 2 0"/>
              </body>
            </body>
          </body>
          <body name="left_thigh" pos="0 .1 -.04">
            <site name="left_hip" class="force-torque"/>
            <joint name="left_hip_x" axis="-1 0 0" range="-25 5"  class="big_joint"/>
            <joint name="left_hip_z" axis="0 0 -1" range="-60 35" class="big_joint"/>
            <joint name="left_hip_y" axis="0 1 0" range="-120 20" class="big_stiff_joint"/>
            <geom name="left_thigh" fromto="0 0 0 0 -.01 -.34" size=".06"/>
            <site name="left_thigh" class="touch" pos="0 -.005 -.17" size=".061 .17" zaxis="0 1 34"/>
            <body name="left_shin" pos="0 -.01 -.403">
              <site name="left_knee" class="force-torque" pos="0 0 .02"/>
              <joint name="left_knee" pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
              <geom name="left_shin" fromto="0 0 0 0 0 -.3"  size=".049"/>
              <site name="left_shin" class="touch"  pos="0 0 -.15" size=".05 .15"/>
              <body name="left_foot" pos="0 0 -.39">
                <site name="left_ankle" class="force-torque"/>
                <joint name="left_ankle_y" pos="0 0 .08" axis="0 1 0"   range="-50 50" stiffness="6"/>
                <joint name="left_ankle_x" pos="0 0 .04" axis="1 0 .5" range="-50 50" stiffness="3"/>
                <geom name="left_left_foot" fromto="-.07 .02 0 .14 .04 0" size=".027"/>
                <geom name="right_left_foot" fromto="-.07 0 0 .14  -.02 0" size=".027"/>
                <site name="right_left_foot" class="touch" pos=".035 -.01 0" size=".03 .11" zaxis="21 -2 0"/>
                <site name="left_left_foot" class="touch" pos=".035 .03 0" size=".03 .11" zaxis="21 2 0"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="right_upper_arm" pos="0 -.17 .06">
        <joint name="right_shoulder1" axis="2 1 1"  range="-85 60"/>
        <joint name="right_shoulder2" axis="0 -1 1" range="-85 60"/>
        <geom name="right_upper_arm" fromto="0 0 0 .16 -.16 -.16" size=".04 .16"/>
        <site name="right_upper_arm" class="touch" pos=".08 -.08 -.08" size=".041 .14" zaxis="1 -1 -1"/>
        <body name="right_lower_arm" pos=".18 -.18 -.18">
          <joint name="right_elbow" axis="0 -1 1" range="-90 50" stiffness="0"/>
          <geom name="right_lower_arm" fromto=".01 .01 .01 .17 .17 .17" size=".031"/>
          <site name="right_lower_arm" class="touch" pos=".09 .09 .09" size=".032 .14" zaxis="1 1 1"/>
          <body name="right_hand" pos=".18 .18 .18">
            <geom name="right_hand" type="sphere" size=".04"/>
            <site name="right_hand" class="touch" type="sphere" size=".041"/>
          </body>
        </body>
      </body>
      <body name="left_upper_arm" pos="0 .17 .06">
        <joint name="left_shoulder1" axis="2 -1 1" range="-60 85"/>
        <joint name="left_shoulder2" axis="0 1 1" range="-60 85"/>
        <geom name="left_upper_arm" fromto="0 0 0 .16 .16 -.16" size=".04 .16"/>
        <site name="left_upper_arm" class="touch" pos=".08 .08 -.08" size=".041 .14" zaxis="1 1 -1"/>
        <body name="left_lower_arm" pos=".18 .18 -.18">
          <joint name="left_elbow" axis="0 -1 -1" range="-90 50" stiffness="0"/>
          <geom name="left_lower_arm" fromto=".01 -.01 .01 .17 -.17 .17" size=".031"/>
          <site name="left_lower_arm" class="touch" pos=".09 -.09 .09" size=".032 .14" zaxis="1 -1 1"/>
          <body name="left_hand" pos=".18 -.18 .18">
            <geom name="left_hand" type="sphere" size=".04"/>
            <site name="left_hand" class="touch" type="sphere" size=".041"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="abdomen_y"       gear="40"  joint="abdomen_y"/>
    <motor name="abdomen_z"       gear="40"  joint="abdomen_z"/>
    <motor name="abdomen_x"       gear="40"  joint="abdomen_x"/>
    <motor name="right_hip_x"     gear="40"  joint="right_hip_x"/>
    <motor name="right_hip_z"     gear="40"  joint="right_hip_z"/>
    <motor name="right_hip_y"     gear="120" joint="right_hip_y"/>
    <motor name="right_knee"      gear="80"  joint="right_knee"/>
    <motor name="right_ankle_x"   gear="20"  joint="right_ankle_x"/>
    <motor name="right_ankle_y"   gear="20"  joint="right_ankle_y"/>
    <motor name="left_hip_x"      gear="40"  joint="left_hip_x"/>
    <motor name="left_hip_z"      gear="40"  joint="left_hip_z"/>
    <motor name="left_hip_y"      gear="120" joint="left_hip_y"/>
    <motor name="left_knee"       gear="80"  joint="left_knee"/>
    <motor name="left_ankle_x"    gear="20"  joint="left_ankle_x"/>
    <motor name="left_ankle_y"    gear="20"  joint="left_ankle_y"/>
    <motor name="right_shoulder1" gear="20"  joint="right_shoulder1"/>
    <motor name="right_shoulder2" gear="20"  joint="right_shoulder2"/>
    <motor name="right_elbow"     gear="40"  joint="right_elbow"/>
    <motor name="left_shoulder1"  gear="20"  joint="left_shoulder1"/>
    <motor name="left_shoulder2"  gear="20"  joint="left_shoulder2"/>
    <motor name="left_elbow"      gear="40"  joint="left_elbow"/>
  </actuator>
"""


comptime dm_humanoid_xml = merge_mjcf(
    dm_skybox_xml, dm_visual_xml, dm_materials_xml, _humanoid_body
)

comptime pmh = parse_xml(dm_humanoid_xml)

# Shared model parameters — the two obs layouts differ only in `OBS_DIM`.
#
#   feature obs = joint_angles (21) + head_height (1) + extremities (12)
#               + torso_vertical (3) + com_velocity (3) + velocity (27) = 67
#   pure state  = position (28) + velocity (27)                         = 55
comptime HUMANOID_OBS_DIM: Int = 67
comptime HUMANOID_PURE_OBS_DIM: Int = 55

comptime DMHumanoidModel = ModelDefFromXML[
    xml=dm_humanoid_xml,
    nbody=pmh.NBODY, njoint=pmh.NJOINT, nq=pmh.NQ, nv=pmh.NV,
    ngeom=pmh.NGEOM, nact=pmh.NACT, ntex=pmh.NTEX, nmat=pmh.NMAT,
    nlight=pmh.NLIGHT, ncam=pmh.NCAM, nsite=pmh.NSITE,
    max_contacts=32,
    obs_dim_override=HUMANOID_OBS_DIM,
    timestep=pmh.TIMESTEP,
]

comptime DMHumanoidPureModel = ModelDefFromXML[
    xml=dm_humanoid_xml,
    nbody=pmh.NBODY, njoint=pmh.NJOINT, nq=pmh.NQ, nv=pmh.NV,
    ngeom=pmh.NGEOM, nact=pmh.NACT, ntex=pmh.NTEX, nmat=pmh.NMAT,
    nlight=pmh.NLIGHT, ncam=pmh.NCAM, nsite=pmh.NSITE,
    max_contacts=32,
    obs_dim_override=HUMANOID_PURE_OBS_DIM,
    timestep=pmh.TIMESTEP,
]

# Body indices — tree DFS, identical to MuJoCo's (asserted in the parity test).
comptime TORSO_BODY_IDX: Int = 1
comptime HEAD_BODY_IDX: Int = 2
comptime RIGHT_FOOT_BODY_IDX: Int = 7
comptime LEFT_FOOT_BODY_IDX: Int = 10
comptime RIGHT_HAND_BODY_IDX: Int = 13
comptime LEFT_HAND_BODY_IDX: Int = 16

comptime N_EXTREMITIES: Int = 4


def extremity_body_indices() -> List[Int]:
    """Bodies whose egocentric offsets form `Physics.extremities()`, IN ORDER.

    The reference iterates `for side in ('left_', 'right_')` then
    `for limb in ('hand', 'foot')`, so the observation order is left_hand,
    left_foot, right_hand, right_foot. Getting it wrong permutes 12
    observation slots without changing the shape — nothing but a value check
    would catch it, which is why the order lives here with the reason attached
    rather than being open-coded at the two call sites.

    A function rather than a `comptime` list: a comptime `List` is not
    `ImplicitlyCopyable`, so it cannot be materialized into a runtime loop.
    """
    return [
        LEFT_HAND_BODY_IDX,
        LEFT_FOOT_BODY_IDX,
        RIGHT_HAND_BODY_IDX,
        RIGHT_FOOT_BODY_IDX,
    ]

# The free root joint occupies qpos[0:7]; `joint_angles()` is qpos[7:].
comptime ROOT_QPOS_SIZE: Int = 7

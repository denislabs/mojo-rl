"""dm_control `quadruped` models — port of `dm_control/suite/quadruped.xml`.

The reference does NOT load that file as written. `quadruped.make_model()`
(suite/quadruped.py:55) parses it with lxml and DELETES elements per task:

    walk / run:  make_model(floor_size=_DEFAULT_TIME_LIMIT * speed)
                 -> terrain=False, rangefinders=False, walls_and_ball=False

so both tasks get the same stripped model, differing only in the floor plane's
half-extent (walk 20*0.5 = 10, run 20*5 = 100). What the strip removes:

  - the four `wall_*` plane geoms,
  - the `target` site,
  - the whole `ball` body (freejoint `ball_root`, `ball` geom, `ball_light`),
  - the `terrain` hfield geom,
  - all twenty `<rangefinder>` SENSORS.

Note what it does NOT remove, and which this module therefore keeps verbatim:
the twenty `rf_*` SITES (only the sensors go), the `terrain` hfield ASSET, and
the `ball` texture/material assets. None of the three touches the dynamics, but
dropping the sites would shift every site index off MuJoCo's — and the
`force`/`torque` sensors are addressed by site.

WHAT THIS MODEL NEEDS THAT THE OTHER PORTS DID NOT
--------------------------------------------------
1. `<general>` actuators with `dyntype="filter"`. Every one of the twelve is

       <general ctrllimited="true" gainprm="1000" biasprm="0 -1000"
                biastype="affine" dyntype="filter" dynprm=".1"/>

   i.e. a position servo (force = 1000*(act - length)) whose setpoint `act` is
   a first-order lag of `ctrl` with a 0.1 s time constant. That ACTIVATION
   STATE is a new piece of `Data` — and it is observable: `egocentric_state()`
   concatenates `data.act` onto the hinge qpos/qvel.

2. Actuator transmission through a fixed tendon for eight of the twelve
   (`lift_*`, `extend_*`), which point_mass already exercised, and directly
   through a joint for the four `yaw_*`.

3. `<equality><tendon>` on the four `coupling_*` tendons, which constrains
   .333*(pitch + knee + ankle) to zero per leg.

4. `accelerometer` and `force`/`torque` sensors — the first two that need
   MuJoCo's `mj_rnePostConstraint` pass rather than a kinematic read.

The `<default>` tree is deep and load-bearing: `class="body"` supplies capsule
type/size/condim/density and every joint's damping/armature/limited, and the
actuator classes (`yaw_act`, `lift_act`, `extend_act`) supply nothing but
`ctrlrange` on top of a bare `<general>` default. Nothing here is spelled out
inline that the reference leaves to a class.
"""

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.parser.xml_parser import merge_mjcf
from mojo_rl.physics3d.types import ConeType

from ..common_xml import dm_visual_xml, dm_skybox_xml, dm_materials_xml


# --- everything before the floor geom ---------------------------------------
comptime _QUADRUPED_HEAD = """
<mujoco model="quadruped">

  <visual>
    <rgba rangefinder="1 1 0.1 0.1"/>
    <map znear=".005" zfar="20"/>
  </visual>

  <asset>
    <hfield name="terrain" ncol="201" nrow="201" size="30 30 5 .1"/>
  </asset>

  <option timestep=".005"/>

  <default>
    <geom solimp=".9 .99 .003" solref=".01 1"/>
    <default class="body">
      <geom  type="capsule" size=".08" condim="1" material="self" density="500"/>
      <joint type="hinge" damping="30" armature=".01"
             limited="true" solimplimit="0 .99 .01"/>
      <default class="hip">
        <default class="yaw">
          <joint axis="0 0 1" range="-50 50"/>
        </default>
        <default class="pitch">
          <joint axis="0 1 0" range="-20 60"/>
        </default>
        <geom fromto="0 0 0 .3 0 .11"/>
      </default>
      <default class="knee">
        <joint axis="0 1 0" range="-60 50"/>
        <geom size=".065" fromto="0 0 0 .25 0 -.25"/>
      </default>
      <default class="ankle">
        <joint axis="0 1 0" range="-45 55"/>
        <geom size=".055" fromto="0 0 0 0 0 -.25"/>
      </default>
      <default class="toe">
        <geom type="sphere" size=".08" material="effector" friction="1.5"/>
        <site type="sphere" size=".084" material="site"  group="4"/>
      </default>
    </default>
    <default class="rangefinder">
      <site type="capsule" size=".005 .1" material="site" group="4"/>
    </default>
    <default class="wall">
      <geom type="plane" material="decoration"/>
    </default>

    <default class="coupling">
      <equality solimp="0.95 0.99 0.01" solref=".005 .5"/>
    </default>

    <general ctrllimited="true" gainprm="1000" biasprm="0 -1000" biastype="affine" dyntype="filter" dynprm=".1"/>
    <default class="yaw_act">
      <general ctrlrange="-1 1"/>
    </default>
    <default class="lift_act">
      <general ctrlrange="-1 1.1"/>
    </default>
    <default class="extend_act">
      <general ctrlrange="-.8 .8"/>
    </default>
  </default>

  <asset>
    <texture name="ball" builtin="checker" mark="cross" width="151" height="151"
             rgb1="0.1 0.1 0.1" rgb2="0.9 0.9 0.9" markrgb="1 1 1"/>
    <material name="ball" texture="ball" />
  </asset>


  <worldbody>
"""

# --- the floor plane, whose half-extent is the ONLY per-task difference ------
comptime _QUADRUPED_TAIL = """
    <camera name="global"  pos="-10 10 10" xyaxes="-1 -1 0 1 0 1" mode="trackcom"/>
    <body name="torso" childclass="body" pos="0 0 .57">
      <freejoint name="root"/>

      <camera name="x"  pos="-1.7 0 1" xyaxes="0 -1 0 .75 0 1" mode="trackcom"/>
      <camera name="y"  pos="0 4 2" xyaxes="-1 0 0 0 -.5 1" mode="trackcom"/>
      <camera name="egocentric"  pos=".3 0 .11" xyaxes="0 -1 0 .4 0 1" fovy="60"/>
      <light name="light" pos="0 0 4" mode="trackcom"/>

      <geom name="eye_r" type="cylinder" size=".05"  fromto=".1 -.07 .12 .31 -.07 .08" mass="0"/>
      <site name="pupil_r" type="sphere" size=".033"  pos=".3 -.07 .08" zaxis="1 0 0" material="eye"/>
      <geom name="eye_l" type="cylinder" size=".05"  fromto=".1 .07 .12 .31 .07 .08" mass="0"/>
      <site name="pupil_l" type="sphere" size=".033"  pos=".3 .07 .08" zaxis="1 0 0" material="eye"/>
      <site name="workspace" type="sphere" size=".3 .3 .3"  material="site" pos=".8 0 -.2" group="3"/>

      <site name="rf_00" class="rangefinder" fromto=".41 -.02  .11 .34 0 .115"/>
      <site name="rf_01" class="rangefinder" fromto=".41 -.01  .11 .34 0 .115"/>
      <site name="rf_02" class="rangefinder" fromto=".41   0   .11 .34 0 .115"/>
      <site name="rf_03" class="rangefinder" fromto=".41  .01  .11 .34 0 .115"/>
      <site name="rf_04" class="rangefinder" fromto=".41  .02  .11 .34 0 .115"/>
      <site name="rf_10" class="rangefinder" fromto=".41 -.02  .1  .36 0 .11"/>
      <site name="rf_11" class="rangefinder" fromto=".41 -.02  .1  .36 0 .11"/>
      <site name="rf_12" class="rangefinder" fromto=".41   0   .1  .36 0 .11"/>
      <site name="rf_13" class="rangefinder" fromto=".41  .01  .1  .36 0 .11"/>
      <site name="rf_14" class="rangefinder" fromto=".41  .02  .1  .36 0 .11"/>
      <site name="rf_20" class="rangefinder" fromto=".41 -.02  .09 .38 0 .105"/>
      <site name="rf_21" class="rangefinder" fromto=".41 -.01  .09 .38 0 .105"/>
      <site name="rf_22" class="rangefinder" fromto=".41   0   .09 .38 0 .105"/>
      <site name="rf_23" class="rangefinder" fromto=".41  .01  .09 .38 0 .105"/>
      <site name="rf_24" class="rangefinder" fromto=".41  .02  .09 .38 0 .105"/>
      <site name="rf_30" class="rangefinder" fromto=".41 -.02  .08 .4  0 .1"/>
      <site name="rf_31" class="rangefinder" fromto=".41 -.01  .08 .4  0 .1"/>
      <site name="rf_32" class="rangefinder" fromto=".41   0   .08 .4  0 .1"/>
      <site name="rf_33" class="rangefinder" fromto=".41  .01  .08 .4  0 .1"/>
      <site name="rf_34" class="rangefinder" fromto=".41  .02  .08 .4  0 .1"/>

      <geom name="torso" type="ellipsoid" size=".3 .27 .2" density="1000"/>
      <site name="torso_touch" type="box" size=".26 .26 .26" rgba="0 0 1 0"/>
      <site name="torso" size=".05" rgba="1 0 0 1" />

      <body name="hip_front_left" pos=".2 .2 0" euler="0 0 45" childclass="hip">
        <joint name="yaw_front_left" class="yaw"/>
        <joint name="pitch_front_left" class="pitch"/>
        <geom name="thigh_front_left"/>
        <body name="knee_front_left" pos=".3 0 .11" childclass="knee">
          <joint name="knee_front_left"/>
          <geom name="shin_front_left"/>
          <body name="ankle_front_left" pos=".25 0 -.25" childclass="ankle">
            <joint name="ankle_front_left"/>
            <geom name="foot_front_left"/>
            <body name="toe_front_left" pos="0 0 -.3" childclass="toe">
              <geom name="toe_front_left"/>
              <site name="toe_front_left"/>
            </body>
          </body>
        </body>
      </body>

      <body name="hip_front_right" pos=".2 -.2 0" euler="0 0 -45" childclass="hip">
        <joint name="yaw_front_right" class="yaw"/>
        <joint name="pitch_front_right" class="pitch"/>
        <geom name="thigh_front_right"/>
        <body name="knee_front_right" pos=".3 0 .11" childclass="knee">
          <joint name="knee_front_right"/>
          <geom name="shin_front_right"/>
          <body name="ankle_front_right" pos=".25 0 -.25" childclass="ankle">
            <joint name="ankle_front_right"/>
            <geom name="foot_front_right"/>
            <body name="toe_front_right" pos="0 0 -.3" childclass="toe">
              <geom name="toe_front_right"/>
              <site name="toe_front_right"/>
            </body>
          </body>
        </body>
      </body>

      <body name="hip_back_right" pos="-.2 -.2 0" euler="0 0 -135" childclass="hip">
        <joint name="yaw_back_right" class="yaw"/>
        <joint name="pitch_back_right" class="pitch"/>
        <geom name="thigh_back_right"/>
        <body name="knee_back_right" pos=".3 0 .11" childclass="knee">
          <joint name="knee_back_right"/>
          <geom name="shin_back_right"/>
          <body name="ankle_back_right" pos=".25 0 -.25" childclass="ankle">
            <joint name="ankle_back_right"/>
            <geom name="foot_back_right"/>
            <body name="toe_back_right" pos="0 0 -.3" childclass="toe">
              <geom name="toe_back_right"/>
              <site name="toe_back_right"/>
            </body>
          </body>
        </body>
      </body>

      <body name="hip_back_left" pos="-.2 .2 0" euler="0 0 135" childclass="hip">
        <joint name="yaw_back_left" class="yaw"/>
        <joint name="pitch_back_left" class="pitch"/>
        <geom name="thigh_back_left"/>
        <body name="knee_back_left" pos=".3 0 .11" childclass="knee">
          <joint name="knee_back_left"/>
          <geom name="shin_back_left"/>
          <body name="ankle_back_left" pos=".25 0 -.25" childclass="ankle">
            <joint name="ankle_back_left"/>
            <geom name="foot_back_left"/>
            <body name="toe_back_left" pos="0 0 -.3" childclass="toe">
              <geom name="toe_back_left"/>
              <site name="toe_back_left"/>
            </body>
          </body>
        </body>
      </body>
    </body>

  </worldbody>

  <tendon>
    <fixed name="coupling_front_left">
      <joint joint="pitch_front_left"      coef=".333"/>
      <joint joint="knee_front_left"       coef=".333"/>
      <joint joint="ankle_front_left"      coef=".333"/>
    </fixed>
    <fixed name="coupling_front_right">
      <joint joint="pitch_front_right"      coef=".333"/>
      <joint joint="knee_front_right"       coef=".333"/>
      <joint joint="ankle_front_right"      coef=".333"/>
    </fixed>
    <fixed name="coupling_back_right">
      <joint joint="pitch_back_right"      coef=".333"/>
      <joint joint="knee_back_right"       coef=".333"/>
      <joint joint="ankle_back_right"      coef=".333"/>
    </fixed>
    <fixed name="coupling_back_left">
      <joint joint="pitch_back_left"      coef=".333"/>
      <joint joint="knee_back_left"       coef=".333"/>
      <joint joint="ankle_back_left"      coef=".333"/>
    </fixed>

    <fixed name="extend_front_left">
      <joint joint="pitch_front_left"      coef=".25"/>
      <joint joint="knee_front_left"       coef="-.5"/>
      <joint joint="ankle_front_left"      coef=".25"/>
    </fixed>
    <fixed name="lift_front_left">
      <joint joint="pitch_front_left"      coef=".5"/>
      <joint joint="ankle_front_left"      coef="-.5"/>
    </fixed>

    <fixed name="extend_front_right">
      <joint joint="pitch_front_right"     coef=".25"/>
      <joint joint="knee_front_right"      coef="-.5"/>
      <joint joint="ankle_front_right"     coef=".25"/>
    </fixed>
    <fixed name="lift_front_right">
      <joint joint="pitch_front_right"     coef=".5"/>
      <joint joint="ankle_front_right"     coef="-.5"/>
    </fixed>

    <fixed name="extend_back_right">
      <joint joint="pitch_back_right"     coef=".25"/>
      <joint joint="knee_back_right"      coef="-.5"/>
      <joint joint="ankle_back_right"     coef=".25"/>
    </fixed>
    <fixed name="lift_back_right">
      <joint joint="pitch_back_right"     coef=".5"/>
      <joint joint="ankle_back_right"     coef="-.5"/>
    </fixed>

    <fixed name="extend_back_left">
      <joint joint="pitch_back_left"      coef=".25"/>
      <joint joint="knee_back_left"       coef="-.5"/>
      <joint joint="ankle_back_left"      coef=".25"/>
    </fixed>
    <fixed name="lift_back_left">
      <joint joint="pitch_back_left"     coef=".5"/>
      <joint joint="ankle_back_left"     coef="-.5"/>
    </fixed>
  </tendon>

  <equality>
    <tendon name="coupling_front_left" tendon1="coupling_front_left" class="coupling"/>
    <tendon name="coupling_front_right" tendon1="coupling_front_right" class="coupling"/>
    <tendon name="coupling_back_right" tendon1="coupling_back_right" class="coupling"/>
    <tendon name="coupling_back_left" tendon1="coupling_back_left" class="coupling"/>
  </equality>

  <actuator>
    <general name="yaw_front_left" class="yaw_act" joint="yaw_front_left"/>
    <general name="lift_front_left" class="lift_act" tendon="lift_front_left"/>
    <general name="extend_front_left" class="extend_act" tendon="extend_front_left"/>
    <general name="yaw_front_right" class="yaw_act" joint="yaw_front_right"/>
    <general name="lift_front_right" class="lift_act" tendon="lift_front_right"/>
    <general name="extend_front_right" class="extend_act" tendon="extend_front_right"/>
    <general name="yaw_back_right" class="yaw_act" joint="yaw_back_right"/>
    <general name="lift_back_right" class="lift_act" tendon="lift_back_right"/>
    <general name="extend_back_right" class="extend_act" tendon="extend_back_right"/>
    <general name="yaw_back_left" class="yaw_act" joint="yaw_back_left"/>
    <general name="lift_back_left" class="lift_act" tendon="lift_back_left"/>
    <general name="extend_back_left" class="extend_act" tendon="extend_back_left"/>
  </actuator>

  <sensor>
    <accelerometer name="imu_accel" site="torso"/>
    <gyro name="imu_gyro" site="torso"/>
    <velocimeter name="velocimeter" site="torso"/>
    <force name="force_toe_front_left" site="toe_front_left"/>
    <force name="force_toe_front_right" site="toe_front_right"/>
    <force name="force_toe_back_right" site="toe_back_right"/>
    <force name="force_toe_back_left" site="toe_back_left"/>
    <torque name="torque_toe_front_left" site="toe_front_left"/>
    <torque name="torque_toe_front_right" site="toe_front_right"/>
    <torque name="torque_toe_back_right" site="toe_back_right"/>
    <torque name="torque_toe_back_left" site="toe_back_left"/>
    <subtreecom name="center_of_mass" body="torso"/>
  </sensor>

</mujoco>
"""


# `f'{floor_size} {floor_size} .5'` with floor_size = _DEFAULT_TIME_LIMIT *
# speed. walk: 20 * 0.5 = 10.0 (a Python float, hence "10.0"); run: 20 * 5 =
# 100 (both ints, hence "100"). The text differs; the number does not.
comptime _quadruped_walk_body = (
    _QUADRUPED_HEAD
    + """    <geom name="floor" type="plane" size="10.0 10.0 .5" material="grid"/>
"""
    + _QUADRUPED_TAIL
)

comptime _quadruped_run_body = (
    _QUADRUPED_HEAD
    + """    <geom name="floor" type="plane" size="100 100 .5" material="grid"/>
"""
    + _QUADRUPED_TAIL
)


comptime dm_quadruped_walk_xml = merge_mjcf(
    dm_skybox_xml, dm_visual_xml, dm_materials_xml, _quadruped_walk_body
)
comptime dm_quadruped_run_xml = merge_mjcf(
    dm_skybox_xml, dm_visual_xml, dm_materials_xml, _quadruped_run_body
)

comptime qwp = parse_xml(dm_quadruped_walk_xml)
comptime qrp = parse_xml(dm_quadruped_run_xml)


# obs = egocentric_state (16 hinge qpos + 16 hinge qvel + 12 act = 44)
#     + torso_velocity (3) + torso_upright (1) + imu (6) + force_torque (24)
#     = 78
comptime QUADRUPED_OBS_DIM: Int = 78


comptime DMQuadrupedWalkModel = ModelDefFromXML[
    xml=dm_quadruped_walk_xml,
    nbody=qwp.NBODY, njoint=qwp.NJOINT, nq=qwp.NQ, nv=qwp.NV,
    ngeom=qwp.NGEOM, nact=qwp.NACT, ntex=qwp.NTEX, nmat=qwp.NMAT,
    nlight=qwp.NLIGHT, ncam=qwp.NCAM, nsite=qwp.NSITE,
    max_tendon=qwp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    # Four toes on the floor, plus the torso ellipsoid when it falls over.
    max_contacts=16,
    obs_dim_override=QUADRUPED_OBS_DIM,
    obs_qpos_skip=0,
    timestep=qwp.TIMESTEP,
]

comptime DMQuadrupedRunModel = ModelDefFromXML[
    xml=dm_quadruped_run_xml,
    nbody=qrp.NBODY, njoint=qrp.NJOINT, nq=qrp.NQ, nv=qrp.NV,
    ngeom=qrp.NGEOM, nact=qrp.NACT, ntex=qrp.NTEX, nmat=qrp.NMAT,
    nlight=qrp.NLIGHT, ncam=qrp.NCAM, nsite=qrp.NSITE,
    max_tendon=qrp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=16,
    obs_dim_override=QUADRUPED_OBS_DIM,
    obs_qpos_skip=0,
    timestep=qrp.TIMESTEP,
]


# --- Task constants, transcribed from suite/quadruped.py --------------------
comptime QUADRUPED_RUN_SPEED: Float64 = 5.0
comptime QUADRUPED_WALK_SPEED: Float64 = 0.5

# The torso body carries the IMU/velocimeter site; the four toes carry the
# force/torque sites. Indices are OURS and are pinned by the parity test
# (`test_rne_post_sensors_vs_mujoco` proves our body and site ORDER equals
# MuJoCo's for this XML, so `mj_name2id` values are valid here).
comptime TORSO_BODY_IDX: Int = 1
comptime TORSO_SITE_IDX: Int = 24

# Toes in SENSOR-ID order — front-left, front-right, back-right, back-left —
# which is how `physics.force_torque()` lays them out. NOT the reference's
# `_TOES` list, which is FL, BL, BR, FR and is only used by `toe_positions()`.
# Each leg is four bodies (hip, knee, ankle, toe), declared in the same order,
# so the toes are evenly spaced. Sites are contiguous because the four toe
# sites are the last four declared.
comptime TOE_BODY_0: Int = 5
comptime TOE_BODY_STRIDE: Int = 4
comptime TOE_SITE_0: Int = 25

# Joint 0 is the free root; joints 1..16 are the leg hinges, so their qpos and
# dof blocks are contiguous. `egocentric_state` reads exactly these.
comptime N_HINGE: Int = 16
comptime HINGE_QPOS_0: Int = 7
comptime HINGE_DOF_0: Int = 6

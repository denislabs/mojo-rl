"""`dm_control` `hopper` model — port of `dm_control/suite/hopper.xml`.

Verbatim apart from the three `<include>` lines.

Two things this model is the first to need:

  * `<default class="hopper"><site type="sphere" size="0.05"/></default>`.
    Both touch sites are declared BY CLASS — the elements themselves are bare
    `<site name="touch_toe" pos=".17 0 0"/>` — and the body picks the class up
    via `childclass="hopper"`. Site default-class inheritance was added with
    this port; without it the sites had no type or size, so the touch sensor's
    zone was a degenerate point and both sensors read a flat zero.

  * `<touch>` sensors. See `physics3d/sensors/touch.mojo`.

`<default class="free">` overrides the `hopper` class for the three root DOFs
(`limited="false" damping="0" armature="0" stiffness="0"`), which is a NESTED
class overriding its parent — the case that broke on cartpole and is gated
here again.

Note the floor is at `pos="48 0 0"` with `size="50 1 .2"`, i.e. it spans
x in [-2, 98]: the hopper starts near one end and hops forward along +x.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.physics3d.parser.xml_parser import merge_mjcf

from ..common_xml import dm_visual_xml, dm_skybox_xml, dm_materials_xml
from mojo_rl.envs.dm_control.hopper.hopper_dims import DM_HOPPER_DIMS


comptime _hopper_body = """
<mujoco model="planar hopper">
  <statistic extent="2" center="0 0 .5"/>

  <default>
    <default class="hopper">
      <joint type="hinge" axis="0 1 0" limited="true" damping=".05" armature=".2"/>
      <geom type="capsule" material="self"/>
      <site type="sphere" size="0.05" group="3"/>
    </default>
    <default class="free">
      <joint limited="false" damping="0" armature="0" stiffness="0"/>
    </default>
    <motor ctrlrange="-1 1" ctrllimited="true"/>
  </default>

  <option timestep="0.005"/>

  <worldbody>
    <camera name="cam0" pos="0 -2.8 0.8" euler="90 0 0" mode="trackcom"/>
    <camera name="back" pos="-2 -.2 1.2" xyaxes="0.2 -1 0 .5 0 2" mode="trackcom"/>
    <geom name="floor" type="plane" conaffinity="1" pos="48 0 0" size="50 1 .2" material="grid"/>
    <body name="torso" pos="0 0 1" childclass="hopper">
      <light name="top" pos="0 0 2" mode="trackcom"/>
      <joint name="rootx" type="slide" axis="1 0 0" class="free"/>
      <joint name="rootz" type="slide" axis="0 0 1" class="free"/>
      <joint name="rooty" type="hinge" axis="0 1 0" class="free"/>
      <geom name="torso" fromto="0 0 -.05 0 0 .2" size="0.0653"/>
      <geom name="nose" fromto=".08 0 .13 .15 0 .14" size="0.03"/>
      <body name="pelvis" pos="0 0 -.05">
        <joint name="waist" range="-30 30"/>
        <geom name="pelvis" fromto="0 0 0 0 0 -.15" size="0.065"/>
        <body name="thigh" pos="0 0 -.2">
          <joint name="hip" range="-170 10"/>
          <geom name="thigh" fromto="0 0 0 0 0 -.33" size="0.04"/>
          <body name="calf" pos="0 0 -.33">
            <joint name="knee" range="5 150"/>
            <geom name="calf" fromto="0 0 0 0 0 -.32" size="0.03"/>
            <body name="foot" pos="0 0 -.32">
              <joint name="ankle" range="-45 45"/>
              <geom name="foot" fromto="-.08 0 0 .17 0 0" size="0.04"/>
              <site name="touch_toe" pos=".17 0 0"/>
              <site name="touch_heel" pos="-.08 0 0"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <sensor>
    <subtreelinvel name="torso_subtreelinvel" body="torso"/>
    <touch name="touch_toe" site="touch_toe"/>
    <touch name="touch_heel" site="touch_heel"/>
  </sensor>

  <actuator>
    <motor name="waist" joint="waist" gear="30"/>
    <motor name="hip" joint="hip" gear="40"/>
    <motor name="knee" joint="knee" gear="30"/>
    <motor name="ankle" joint="ankle" gear="10"/>
  </actuator>
</mujoco>
"""


comptime dm_hopper_xml = merge_mjcf(
    dm_skybox_xml, dm_visual_xml, dm_materials_xml, _hopper_body
)

comptime pmh = DM_HOPPER_DIMS

# obs = position (qpos[1:], nq-1 = 6) + velocity (nv = 7) + touch (2) = 15
comptime DMHopperModel = ModelDefFromXML[
    xml=dm_hopper_xml,
    xml_path="mojo_rl/envs/dm_control/assets/hopper.xml",
    nbody=pmh.NBODY, njoint=pmh.NJOINT, nq=pmh.NQ, nv=pmh.NV,
    ngeom=pmh.NGEOM, nact=pmh.NACT, ntex=pmh.NTEX, nmat=pmh.NMAT,
    nlight=pmh.NLIGHT, ncam=pmh.NCAM, nsite=pmh.NSITE,
    max_contacts=16,
    obs_dim_override=15,
    timestep=pmh.TIMESTEP,
]

# Body indices in worldbody DFS order (0 = world).
comptime TORSO_BODY_IDX: Int = 1
comptime FOOT_BODY_IDX: Int = 5

# Site indices in worldbody DFS order.
comptime TOUCH_TOE_SITE_IDX: Int = 0
comptime TOUCH_HEEL_SITE_IDX: Int = 1

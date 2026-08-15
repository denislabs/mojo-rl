"""`dm_control` `walker` model — port of `dm_control/suite/walker.xml`.

Body XML is the reference file verbatim apart from the three `<include>`
lines, which `merge_mjcf` splices in instead (same order as the reference).

Unlike cartpole this model is NOT built procedurally — walker.py loads the
file as-is for all three tasks, so there is a single model here.

Two MJCF features this file needs that older ports did not:

  * `zaxis="1 0 0"` on the foot capsules — "point local +Z along this vector",
    MuJoCo's `mjuu_z2quat`. Previously unparsed, so the feet came out oriented
    along +Z (upright pegs instead of forward-pointing soles), which changes
    both their inertia tensors and their contact geometry.
  * a top-level unnamed `<default>` **plus** a named `<default class="walker">`,
    with `childclass="walker"` on the torso. Bare `<joint name="right_hip"
    range="-20 100"/>` elements take their type (hinge, MuJoCo's default),
    axis, damping, armature and solimplimit from those two blocks combined.

`<option timestep="0.0025"/>` states no integrator, so MuJoCo's Euler default
applies — as with pendulum, and unlike cartpole's explicit RK4.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.physics3d.parser.xml_parser import merge_mjcf

from ..common_xml import dm_visual_xml, dm_skybox_xml, dm_materials_xml
from mojo_rl.envs.dm_control.walker.walker_dims import DM_WALKER_DIMS


comptime _walker_body = """
<mujoco model="planar walker">
  <option timestep="0.0025"/>

  <statistic extent="2" center="0 0 1"/>

  <default>
    <joint damping=".1" armature="0.01" limited="true" solimplimit="0 .99 .01"/>
    <geom contype="1" conaffinity="0" friction=".7 .1 .1"/>
    <motor ctrlrange="-1 1" ctrllimited="true"/>
    <site size="0.01"/>
    <default class="walker">
      <geom material="self" type="capsule"/>
      <joint axis="0 -1 0"/>
    </default>
  </default>

  <worldbody>
    <geom name="floor" type="plane" conaffinity="1" pos="248 0 0" size="250 .8 .2" material="grid" zaxis="0 0 1"/>
    <body name="torso" pos="0 0 1.3" childclass="walker">
      <light name="light" pos="0 0 2" mode="trackcom"/>
      <camera name="side" pos="0 -2 .7" euler="60 0 0" mode="trackcom"/>
      <camera name="back" pos="-2 0 .5" xyaxes="0 -1 0 1 0 3" mode="trackcom"/>
      <joint name="rootz" axis="0 0 1" type="slide" limited="false" armature="0" damping="0"/>
      <joint name="rootx" axis="1 0 0" type="slide" limited="false" armature="0" damping="0"/>
      <joint name="rooty" axis="0 1 0" type="hinge" limited="false" armature="0" damping="0"/>
      <geom name="torso" size="0.07 0.3"/>
      <body name="right_thigh" pos="0 -.05 -0.3">
        <joint name="right_hip" range="-20 100"/>
        <geom name="right_thigh" pos="0 0 -0.225" size="0.05 0.225"/>
        <body name="right_leg" pos="0 0 -0.7">
          <joint name="right_knee" pos="0 0 0.25" range="-150 0"/>
          <geom name="right_leg" size="0.04 0.25"/>
          <body name="right_foot" pos="0.06 0 -0.25">
            <joint name="right_ankle" pos="-0.06 0 0" range="-45 45"/>
            <geom name="right_foot" zaxis="1 0 0" size="0.05 0.1"/>
          </body>
        </body>
      </body>
      <body name="left_thigh" pos="0 .05 -0.3" >
        <joint name="left_hip" range="-20 100"/>
        <geom name="left_thigh" pos="0 0 -0.225" size="0.05 0.225"/>
        <body name="left_leg" pos="0 0 -0.7">
          <joint name="left_knee" pos="0 0 0.25" range="-150 0"/>
          <geom name="left_leg" size="0.04 0.25"/>
          <body name="left_foot" pos="0.06 0 -0.25">
            <joint name="left_ankle" pos="-0.06 0 0" range="-45 45"/>
            <geom name="left_foot" zaxis="1 0 0" size="0.05 0.1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <sensor>
    <subtreelinvel name="torso_subtreelinvel" body="torso"/>
  </sensor>

  <actuator>
    <motor name="right_hip" joint="right_hip" gear="100"/>
    <motor name="right_knee" joint="right_knee" gear="50"/>
    <motor name="right_ankle" joint="right_ankle" gear="20"/>
    <motor name="left_hip" joint="left_hip" gear="100"/>
    <motor name="left_knee" joint="left_knee" gear="50"/>
    <motor name="left_ankle" joint="left_ankle" gear="20"/>
  </actuator>
</mujoco>
"""


comptime dm_walker_xml = merge_mjcf(
    dm_visual_xml, dm_skybox_xml, dm_materials_xml, _walker_body
)

comptime pmw = DM_WALKER_DIMS

# obs = orientations (nbody-1 bodies x [xx, xz]) + height (1) + velocity (nv)
#     = 7*2 + 1 + 9 = 24
comptime DMWalkerModel = ModelDefFromXML[
    xml=dm_walker_xml,
    xml_path="mojo_rl/envs/dm_control/assets/walker.xml",
    nbody=pmw.NBODY, njoint=pmw.NJOINT, nq=pmw.NQ, nv=pmw.NV,
    ngeom=pmw.NGEOM, nact=pmw.NACT, ntex=pmw.NTEX, nmat=pmw.NMAT,
    nlight=pmw.NLIGHT, ncam=pmw.NCAM, nsite=pmw.NSITE,
    max_contacts=16,
    obs_dim_override=24,
    timestep=pmw.TIMESTEP,
]

# Body indices in worldbody DFS order.
comptime TORSO_BODY_IDX: Int = 1

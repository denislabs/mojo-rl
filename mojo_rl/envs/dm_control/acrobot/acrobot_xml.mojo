"""`dm_control` `acrobot` model — port of `dm_control/suite/acrobot.xml`.

Verbatim apart from the three `<include>` lines, which `merge_mjcf` replaces
with the shared fragments (the reference's include order is skybox-last but
`common.read_model` splices them all before compile, so order is cosmetic).

Two things about this model are load-bearing:

  * `<flag constraint="disable"/>`. The lower arm sweeps a full metre below
    the floor plane, so the model only makes sense with the constraint solver
    off. Parser support for the flag was added on 2026-07-29 alongside this
    port; before that the arms bounced off the floor.

  * `<default><geom type="capsule" mass="1"/></default>` supplies BOTH geoms'
    type and mass, and `<joint damping=".05"/>` both joints' damping. The
    elements themselves carry only names and `fromto`/`axis`, so unnamed
    default inheritance for structural attributes has to work.

Sites are indexed in worldbody DFS order, so `target` (declared before the
arm chain) is site 0 and `tip` (inside `lower_arm`) is site 1.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.physics3d.parser.xml_parser import merge_mjcf

from ..common_xml import dm_visual_xml, dm_skybox_xml, dm_materials_xml
from mojo_rl.envs.dm_control.acrobot.acrobot_dims import DM_ACROBOT_DIMS


comptime _acrobot_body = """
<mujoco model="acrobot">
  <default>
    <joint damping=".05"/>
    <geom type="capsule" mass="1"/>
  </default>

  <option timestep="0.01" integrator="RK4">
    <flag constraint="disable" energy="enable"/>
  </option>

  <worldbody>
    <light name="light" pos="0 0 6"/>
    <geom name="floor" size="3 3 .2" type="plane" material="grid"/>
    <site name="target" type="sphere" pos="0 0 4" size="0.2" material="target" group="3"/>
    <camera name="fixed" pos="0 -6 2" zaxis="0 -1 0"/>
    <body name="upper_arm" pos="0 0 2">
      <joint name="shoulder" type="hinge" axis="0 1 0"/>
      <geom name="upper_arm_decoration" material="decoration" type="cylinder" fromto="0 -.06 0 0 .06 0" size="0.051" mass="0"/>
      <geom name="upper_arm" fromto="0 0 0 0 0 1" size="0.05" material="self"/>
      <body name="lower_arm" pos="0 0 1">
        <joint name="elbow" type="hinge" axis="0 1 0"/>
        <geom name="lower_arm" fromto="0 0 0 0 0 1" size="0.049" material="self"/>
        <site name="tip" pos="0 0 1" size="0.01"/>
      </body>
    </body>
  </worldbody>

   <actuator>
    <motor name="elbow" joint="elbow" gear="2" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


comptime dm_acrobot_xml = merge_mjcf(
    dm_visual_xml, dm_skybox_xml, dm_materials_xml, _acrobot_body
)

comptime pma = DM_ACROBOT_DIMS

# obs = orientations (2 bodies x xz, then 2 bodies x zz = 4) + velocity (2) = 6
comptime DMAcrobotModel = ModelDefFromXML[
    xml=dm_acrobot_xml,
    nbody=pma.NBODY, njoint=pma.NJOINT, nq=pma.NQ, nv=pma.NV,
    ngeom=pma.NGEOM, nact=pma.NACT, ntex=pma.NTEX, nmat=pma.NMAT,
    nlight=pma.NLIGHT, ncam=pma.NCAM, nsite=pma.NSITE,
    max_contacts=1,
    obs_dim_override=6,
    timestep=pma.TIMESTEP,
]

# Body indices in worldbody DFS order (0 = world).
comptime UPPER_ARM_BODY_IDX: Int = 1
comptime LOWER_ARM_BODY_IDX: Int = 2

# Site indices in worldbody DFS order.
comptime TARGET_SITE_IDX: Int = 0
comptime TIP_SITE_IDX: Int = 1

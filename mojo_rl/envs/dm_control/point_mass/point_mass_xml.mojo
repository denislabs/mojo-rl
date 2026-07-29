"""dm_control `point_mass` model — port of `dm_control/suite/point_mass.xml`.

Verbatim apart from the `<include>` lines and ONE deliberate substitution.

THE SUBSTITUTION. The reference drives the mass through two fixed tendons:

    <tendon>
      <fixed name="t1"><joint joint="root_x" coef="1"/>
                       <joint joint="root_y" coef="0"/></fixed>
      <fixed name="t2"><joint joint="root_x" coef="0"/>
                       <joint joint="root_y" coef="1"/></fixed>
    </tendon>
    <actuator>
      <motor name="t1" tendon="t1"/>
      <motor name="t2" tendon="t2"/>
    </actuator>

A fixed tendon's actuator moment arm on joint j is just its `coef`, so with
the identity coefficient matrix above this is exactly two joint motors —
`t1` drives `root_x`, `t2` drives `root_y`, both at the default `gear=".1"`.
Our engine has no tendon transmission (an actuator resolves to one joint), so
the model below writes that equivalence out directly. This is safe ONLY for
the `easy` task: `hard` overwrites `model.wrap_prm` each episode with a random
mixing matrix, which is genuinely a tendon feature and is why `point_mass-hard`
sits in Tier B. `tests/dm_control/test_point_mass_vs_dm_control.mojo` drives
MuJoCo from the UNMODIFIED reference XML, so the substitution is proved rather
than assumed.

A tendon-transmission actuator used to parse into `joint_id = -1` silently;
`ModelDefFromXML` now raises on that (added with this port).

Note the joints are `limited="true" range="-.29 .29"` SLIDE joints, so the
degree->radian conversion must not touch them — see the `<compiler angle>`
notes in docs/DM_CONTROL_PORT.md.
"""

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.parser.xml_parser import merge_mjcf

from ..common_xml import dm_visual_xml, dm_skybox_xml, dm_materials_xml


comptime _point_mass_body = """
<mujoco model="planar point mass">
  <option timestep="0.02">
    <flag contact="disable"/>
  </option>

  <default>
    <joint type="hinge" axis="0 0 1" limited="true" range="-.29 .29" damping="1"/>
    <motor gear=".1" ctrlrange="-1 1" ctrllimited="true"/>
  </default>

  <worldbody>
    <light name="light" pos="0 0 1"/>
    <camera name="fixed" pos="0 0 .75" quat="1 0 0 0"/>
    <geom name="ground" type="plane" pos="0 0 0" size=".3 .3 .1" material="grid"/>
    <geom name="wall_x" type="plane" pos="-.3 0 .02" zaxis="1 0 0"  size=".02 .3 .02" material="decoration"/>
    <geom name="wall_y" type="plane" pos="0 -.3 .02" zaxis="0 1 0"  size=".3 .02 .02" material="decoration"/>
    <geom name="wall_neg_x" type="plane" pos=".3 0 .02" zaxis="-1 0 0"  size=".02 .3 .02" material="decoration"/>
    <geom name="wall_neg_y" type="plane" pos="0 .3 .02" zaxis="0 -1 0"  size=".3 .02 .02" material="decoration"/>

    <body name="pointmass" pos="0 0 .01">
      <camera name="cam0" pos="0 -0.3 0.3" xyaxes="1 0 0 0 0.7 0.7"/>
      <joint name="root_x" type="slide"  pos="0 0 0" axis="1 0 0" />
      <joint name="root_y" type="slide"  pos="0 0 0" axis="0 1 0" />
      <geom name="pointmass" type="sphere" size=".01" material="self" mass=".3"/>
    </body>

    <geom name="target" pos="0 0 .01" material="target" type="sphere" size=".015"/>
  </worldbody>

  <actuator>
    <motor name="t1" joint="root_x"/>
    <motor name="t2" joint="root_y"/>
  </actuator>
</mujoco>
"""


comptime dm_point_mass_xml = merge_mjcf(
    dm_skybox_xml, dm_visual_xml, dm_materials_xml, _point_mass_body
)

comptime pmp = parse_xml(dm_point_mass_xml)

# obs = position (qpos, 2) + velocity (qvel, 2) = 4
comptime DMPointMassModel = ModelDefFromXML[
    xml=dm_point_mass_xml,
    nbody=pmp.NBODY, njoint=pmp.NJOINT, nq=pmp.NQ, nv=pmp.NV,
    ngeom=pmp.NGEOM, nact=pmp.NACT, ntex=pmp.NTEX, nmat=pmp.NMAT,
    nlight=pmp.NLIGHT, ncam=pmp.NCAM, nsite=pmp.NSITE,
    max_contacts=1,
    obs_dim_override=4,
    timestep=pmp.TIMESTEP,
]

# Geom indices in OUR ordering, which is worldbody text (DFS) order.
#
# CAUTION: this is NOT MuJoCo's geom order. MuJoCo sorts geoms by body id, so
# in mjModel the five world geoms come first, then `target` (also world), then
# `pointmass` on body 1 — i.e. target=5, pointmass=6, the reverse of ours.
# Our parser numbers them as it walks the XML, so `pointmass` (declared inside
# the body) precedes `target` (declared after it). The two orders coincide for
# every previously ported domain because their world geoms all appear before
# any body; point_mass is the first model to interleave. The parity test pins
# both orders explicitly instead of assuming they agree.
comptime POINTMASS_GEOM_IDX: Int = 5
comptime TARGET_GEOM_IDX: Int = 6

# `named.model.geom_size['target', 0]` — the target sphere's radius. Geom sizes
# are not carried in a form the reward hook reads, so it is lifted from the XML
# here and asserted against `model.geom_size` in the parity test.
comptime TARGET_SIZE: Float64 = 0.015

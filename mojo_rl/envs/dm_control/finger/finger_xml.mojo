"""`dm_control` `finger` model — port of `dm_control/suite/finger.xml`.

Verbatim apart from the `<include>` lines and ONE deliberate substitution, the
same one reacher needed (gap G4).

THE SUBSTITUTION — the target becomes a MOCAP BODY. The reference declares it
as a plain worldbody site:

    <site name="target" type="sphere" size=".03" pos="0 0 .4" material="target"/>

and `Turn.initialize_episode` then rewrites `model.site_pos['target', ['x','z']]`
every episode from a fresh random angle. `fields.Model` is a single SHARED,
UNBATCHED tensor set, so a model write is a write for every env in the batch.
A mocap body is the sanctioned workaround: FK skips mocap bodies and the facade
presets their world pose from `d.mocap_pos`, which is per-env `[BATCH, NBODY*3]`
state — so the target moves per episode without the model moving. The site rides
its body, so the `framepos` sensor reads back correctly.

    <body name="target" mocap="true" pos="0 0 .4">
      <site name="target" type="sphere" size=".03" material="target"/>
    </body>

This is physically inert in both versions: the site carries no geom, and a
jointless body contributes no DOF. It adds one body (index 4, appended after
the arm chain and spinner so those keep the reference's own 1..3).

WHAT RESET WRITES THAT WE CARRY AS CONFIG COMPTIMES INSTEAD (all constant per
task, so none of them needs a per-episode model write):
  * `site_size['target', 0]` — .07 (turn_easy) / .03 (turn_hard). Feeds only
    the reward radius via `dist_to_target`, never a contact.
  * `dof_damping['hinge']` — `Spin.initialize_episode` drops it from the XML's
    .5 to .03. The XML below keeps the reference's .5; the spin config applies
    .03. This one is NOT cosmetic — it changes the spinner's dynamics.
  * `site_rgba['target'/'tip', 3] = 0` in Spin — pure visuals, dropped.

`<option cone="elliptic" iterations="200">` — the cone is passed through as
`cone_type=ConeType.ELLIPTIC` (a `ModelDefFromXML` parameter; the parser does
not read the attribute). MuJoCo's 200 solver iterations already match our
Newton default, so `iterations` needs no plumbing. `<flag gravity="disable"/>`
IS parsed and zeroes the gravity vector.

⚠ `joint proximal` carries `ref="-90"`, so its qpos0 is NOT zero — the one
place in this model where the reference configuration differs from all-zeros.

GEOM ORDER, as always: ours is XML text order, MuJoCo's is sorted by body id.
The parity test pins both.
"""

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.parser.xml_parser import merge_mjcf
from mojo_rl.physics3d.types import ConeType

from ..common_xml import dm_visual_xml, dm_skybox_xml, dm_materials_xml


comptime _finger_body = """
<mujoco model="finger">
  <option timestep="0.01" cone="elliptic" iterations="200">
    <flag gravity="disable"/>
  </option>

  <default>
    <geom solimp="0 0.9 0.01" solref=".02 1"/>
    <joint type="hinge" axis="0 -1 0"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
    <default class="finger">
      <joint damping="2.5" limited="true"/>
      <site type="ellipsoid" size=".025 .03 .025" material="site" group="3"/>
    </default>
  </default>

  <worldbody>
    <light name="light" directional="true" diffuse=".6 .6 .6" pos="0 0 2" specular=".3 .3 .3"/>
    <geom name="ground" type="plane" pos="0 0 0" size=".6 .2 10" material="grid"/>
    <camera name="cam0" pos="0 -1 .8" xyaxes="1 0 0 0 1 2"/>
    <camera name="cam1" pos="0 -1 .4" xyaxes="1 0 0 0 0 1" />

    <body name="proximal" pos="-.2 0 .4" childclass="finger">
      <geom name="proximal_decoration" type="cylinder" fromto="0 -.033 0 0 .033 0" size=".034" material="decoration"/>
      <joint name="proximal" range="-110 110" ref="-90"/>
      <geom name="proximal" type="capsule" material="self" size=".03" fromto="0 0 0 0 0 -.17"/>
      <body name="distal" pos="0 0 -.18" childclass="finger">
        <joint name="distal" range="-110 110"/>
        <geom name="distal" type="capsule" size=".028" material="self" fromto="0 0 0 0 0 -.16" contype="0" conaffinity="0"/>
        <geom name="fingertip" type="capsule" size=".03" material="effector" fromto="0 0 -.13 0 0 -.161"/>
        <site name="touchtop" pos=".01 0 -.17"/>
        <site name="touchbottom" pos="-.01 0 -.17"/>
      </body>
    </body>

    <body name="spinner" pos=".2 0 .4">
      <joint name="hinge" frictionloss=".1" damping=".5"/>
      <geom name="cap1" type="capsule" size=".04 .09" material="self" pos=".02 0 0"/>
      <geom name="cap2" type="capsule" size=".04 .09" material="self" pos="-.02 0 0"/>
      <site name="tip" type="sphere"  size=".02" pos="0 0 .13" material="target"/>
      <geom name="spinner_decoration" type="cylinder" fromto="0 -.045 0 0 .045 0" size=".02" material="decoration"/>
    </body>

    <body name="target" mocap="true" pos="0 0 .4">
      <site name="target" type="sphere" size=".03" material="target"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="proximal" joint="proximal" gear="30"/>
    <motor name="distal" joint="distal" gear="15"/>
  </actuator>

  <sensor>
    <jointpos name="proximal" joint="proximal"/>
    <jointpos name="distal" joint="distal"/>
    <jointvel name="proximal_velocity" joint="proximal"/>
    <jointvel name="distal_velocity" joint="distal"/>
    <jointvel name="hinge_velocity" joint="hinge"/>
    <framepos name="tip" objtype="site" objname="tip"/>
    <framepos name="target" objtype="site" objname="target"/>
    <framepos name="spinner" objtype="xbody" objname="spinner"/>
    <touch name="touchtop" site="touchtop"/>
    <touch name="touchbottom" site="touchbottom"/>
    <framepos name="touchtop_pos" objtype="site" objname="touchtop"/>
    <framepos name="touchbottom_pos" objtype="site" objname="touchbottom"/>
  </sensor>
</mujoco>
"""


# `Spin.initialize_episode` writes `dof_damping['hinge'] = .03` (the XML says
# .5). That is a real dynamics change and `fields.Model` is shared+unbatched,
# so it cannot be a per-episode write — spin compiles from its OWN XML with the
# value already substituted. The substitution is asserted in the parity test,
# because a silent no-op here would leave spin running the turn dynamics.
comptime _finger_body_spin = String(_finger_body).replace(
    '<joint name="hinge" frictionloss=".1" damping=".5"/>',
    '<joint name="hinge" frictionloss=".1" damping=".03"/>',
)

comptime dm_finger_xml = merge_mjcf(
    dm_skybox_xml, dm_visual_xml, dm_materials_xml, _finger_body
)

comptime dm_finger_spin_xml = merge_mjcf(
    dm_skybox_xml, dm_visual_xml, dm_materials_xml, _finger_body_spin
)

comptime pmf = parse_xml(dm_finger_xml)
comptime pmfs = parse_xml(dm_finger_spin_xml)

# obs (spin) = position (4) + velocity (3) + touch (2) = 9
comptime DMFingerSpinModel = ModelDefFromXML[
    xml=dm_finger_spin_xml,
    nbody=pmf.NBODY, njoint=pmf.NJOINT, nq=pmf.NQ, nv=pmf.NV,
    ngeom=pmf.NGEOM, nact=pmf.NACT, ntex=pmf.NTEX, nmat=pmf.NMAT,
    nlight=pmf.NLIGHT, ncam=pmf.NCAM, nsite=pmf.NSITE,
    max_contacts=8,
    obs_dim_override=9,
    timestep=pmf.TIMESTEP,
    cone_type = ConeType.ELLIPTIC,
]

# obs (turn) = position (4) + velocity (3) + touch (2) + target_position (2)
#            + dist_to_target (1) = 12
comptime DMFingerTurnModel = ModelDefFromXML[
    xml=dm_finger_xml,
    nbody=pmf.NBODY, njoint=pmf.NJOINT, nq=pmf.NQ, nv=pmf.NV,
    ngeom=pmf.NGEOM, nact=pmf.NACT, ntex=pmf.NTEX, nmat=pmf.NMAT,
    nlight=pmf.NLIGHT, ncam=pmf.NCAM, nsite=pmf.NSITE,
    max_contacts=8,
    obs_dim_override=12,
    timestep=pmf.TIMESTEP,
    cone_type = ConeType.ELLIPTIC,
]

# Body indices in worldbody DFS order (0 = world); `target` appended last.
comptime PROXIMAL_BODY_IDX: Int = 1
comptime DISTAL_BODY_IDX: Int = 2
comptime SPINNER_BODY_IDX: Int = 3
comptime TARGET_BODY_IDX: Int = 4

# Site indices in XML text order — pinned by the parity test.
comptime TOUCHTOP_SITE_IDX: Int = 0
comptime TOUCHBOTTOM_SITE_IDX: Int = 1
comptime TIP_SITE_IDX: Int = 2
comptime TARGET_SITE_IDX: Int = 3

# qpos / qvel addresses (three hinges, in XML order).
comptime PROXIMAL_ADR: Int = 0
comptime DISTAL_ADR: Int = 1
comptime HINGE_ADR: Int = 2

# `radius = model.geom_size['cap1'].sum()` in Turn.initialize_episode — the
# arm length at which the target is placed around the hinge (.04 + .09).
comptime SPINNER_RADIUS: Float64 = 0.13

# Target site z when Turn writes only x/z: the body sits at the hinge height.
comptime TARGET_Z: Float64 = 0.4

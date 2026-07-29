"""dm_control `cheetah` model — port of `dm_control/suite/cheetah.xml`.

Body XML is the reference file verbatim apart from the three `<include>`
lines, which `merge_mjcf` splices in instead (same order as the reference).

The features this model needs that earlier domains did not:

  * `euler="0 <deg> 0"` on seven geoms — head and every leg segment. MuJoCo
    resolves it per `ResolveOrientation` with the compiler's `eulerseq`
    (default "xyz", lowercase => moving axes => post-multiply). Added to the
    parser 2026-07-29.
  * `<compiler settotalmass="14"/>`, which rescales every body mass after the
    geom-derived inertia pass.
  * NO `angle` attribute on `<compiler>`, so MuJoCo's default of DEGREE
    applies. That default was wrong in our parser until 2026-07-29 — the joint
    ranges here ("-30 60", "-230 50", ...) would otherwise be read as radians
    and the legs would swing unconstrained.
  * joint `stiffness` (a passive spring), 8 on the class and 60-240 per joint.

`<option timestep="0.01"/>` states no integrator => MuJoCo's Euler default,
and cheetah.py passes no `control_timestep`, so one env step is one physics
step.
"""

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.parser.xml_parser import merge_mjcf

from ..common_xml import dm_visual_xml, dm_skybox_xml, dm_materials_xml


comptime _cheetah_body = """
<mujoco model="cheetah">
  <compiler settotalmass="14"/>

  <default>
    <default class="cheetah">
      <joint limited="true" damping=".01" armature=".1" stiffness="8" type="hinge" axis="0 1 0"/>
      <geom contype="1" conaffinity="1" condim="3" friction=".4 .1 .1" material="self"/>
    </default>
    <default class="free">
      <joint limited="false" damping="0" armature="0" stiffness="0"/>
    </default>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>

  <statistic center="0 0 .7" extent="2"/>

  <option timestep="0.01"/>

  <worldbody>
    <geom name="ground" type="plane" conaffinity="1" pos="98 0 0" size="100 .8 .5" material="grid"/>
    <body name="torso" pos="0 0 .7" childclass="cheetah">
      <light name="light" pos="0 0 2" mode="trackcom"/>
      <camera name="side" pos="0 -3 0" quat="0.707 0.707 0 0" mode="trackcom"/>
      <camera name="back" pos="-1.8 -1.3 0.8" xyaxes="0.45 -0.9 0 0.3 0.15 0.94" mode="trackcom"/>
      <joint name="rootx" type="slide" axis="1 0 0" class="free"/>
      <joint name="rootz" type="slide" axis="0 0 1" class="free"/>
      <joint name="rooty" type="hinge" axis="0 1 0" class="free"/>
      <geom name="torso" type="capsule" fromto="-.5 0 0 .5 0 0" size="0.046"/>
      <geom name="head" type="capsule" pos=".6 0 .1" euler="0 50 0" size="0.046 .15"/>
      <body name="bthigh" pos="-.5 0 0">
        <joint name="bthigh" range="-30 60" stiffness="240" damping="6"/>
        <geom name="bthigh" type="capsule" pos=".1 0 -.13" euler="0 -218 0" size="0.046 .145"/>
        <body name="bshin" pos=".16 0 -.25">
          <joint name="bshin" range="-50 50" stiffness="180" damping="4.5"/>
          <geom name="bshin" type="capsule" pos="-.14 0 -.07" euler="0 -116 0" size="0.046 .15"/>
          <body name="bfoot" pos="-.28 0 -.14">
            <joint name="bfoot" range="-230 50" stiffness="120" damping="3"/>
            <geom name="bfoot" type="capsule" pos=".03 0 -.097" euler="0 -15 0" size="0.046 .094"/>
          </body>
        </body>
      </body>
      <body name="fthigh" pos=".5 0 0">
        <joint name="fthigh" range="-57 .40" stiffness="180" damping="4.5"/>
        <geom name="fthigh" type="capsule" pos="-.07 0 -.12" euler="0 30 0" size="0.046 .133"/>
        <body name="fshin" pos="-.14 0 -.24">
          <joint name="fshin" range="-70 50" stiffness="120" damping="3"/>
          <geom name="fshin" type="capsule" pos=".065 0 -.09" euler="0 -34 0" size="0.046 .106"/>
          <body name="ffoot" pos=".13 0 -.18">
            <joint name="ffoot" range="-28 28" stiffness="60" damping="1.5"/>
            <geom name="ffoot" type="capsule" pos=".045 0 -.07" euler="0 -34 0" size="0.046 .07"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <sensor>
    <subtreelinvel name="torso_subtreelinvel" body="torso"/>
  </sensor>

  <actuator>
    <motor name="bthigh" joint="bthigh" gear="120" />
    <motor name="bshin" joint="bshin" gear="90" />
    <motor name="bfoot" joint="bfoot" gear="60" />
    <motor name="fthigh" joint="fthigh" gear="90" />
    <motor name="fshin" joint="fshin" gear="60" />
    <motor name="ffoot" joint="ffoot" gear="30" />
  </actuator>
</mujoco>
"""


comptime dm_cheetah_xml = merge_mjcf(
    dm_skybox_xml, dm_visual_xml, dm_materials_xml, _cheetah_body
)

comptime pmc = parse_xml(dm_cheetah_xml)

# obs = position (qpos[1:], nq-1 = 8) + velocity (nv = 9) = 17
comptime DMCheetahModel = ModelDefFromXML[
    xml=dm_cheetah_xml,
    nbody=pmc.NBODY, njoint=pmc.NJOINT, nq=pmc.NQ, nv=pmc.NV,
    ngeom=pmc.NGEOM, nact=pmc.NACT, ntex=pmc.NTEX, nmat=pmc.NMAT,
    nlight=pmc.NLIGHT, ncam=pmc.NCAM, nsite=pmc.NSITE,
    max_contacts=16,
    obs_dim_override=17,
    timestep=pmc.TIMESTEP,
]

# Body indices in worldbody DFS order.
comptime TORSO_BODY_IDX: Int = 1

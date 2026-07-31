"""dm_control `ball_in_cup` model — port of `dm_control/suite/ball_in_cup.xml`.

VERBATIM apart from the three `<include>` lines, which `merge_mjcf` splices
in. Unlike `point_mass`, nothing here is substituted: this is the first ported
domain whose `<tendon>` is load-bearing and expressed directly.

WHAT THIS MODEL NEEDED THAT DID NOT EXIST
-----------------------------------------
`<spatial>` tendons. The string is a two-site polyline from the ball's site to
the cup's, with `limited="true" range="0 0.3"` — so it is inextensible past
30 cm and does nothing at all below that. Before this port the engine had only
FIXED (joint-coefficient) tendons, in two disjoint representations neither of
which could express a site-routed length:

  - `dynamics/tendon.mojo`      — length + dense moment arm (mj_tendon)
  - `constraints/tendon_limit.mojo` — the `mjCNSTR_LIMIT_TENDON` row
  - `full_parser` `_fill_tendons` — `<tendon>` had NO runtime parsing at all;
    `fields_build` hardcoded `ntendon = 0`, so every tendon record was dead.

The limit is built as a ROW OF THE SAME SYSTEM as the contacts, not as a
post-pass. ball_in_cup is precisely the shape that made the sequential split
visible on finger (commit 04a7c508): a caught ball rests on the cup capsules
while the string is taut, on shared dofs.

ORDERING. Our geom/site/body/joint numbering coincides with MuJoCo's here
(geoms: ground, cup_part_0..4, ball; sites: cup, target, ball; bodies: world,
cup, ball) because every world geom precedes the first body — the interleaving
that bit `point_mass` does not occur. The parity test pins all four orders
explicitly rather than trusting that.

CONE. `<option>` is absent, so MuJoCo's defaults apply: timestep 0.002,
Newton solver, Euler integrator, PYRAMIDAL cone. The pyramidal cone matters —
tendon limit rows are built on the pyramidal edge list only, and
`ModelDefFromXML` raises if a model asks for elliptic with a limited tendon.
"""

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.parser.xml_parser import merge_mjcf
from mojo_rl.physics3d.types import ConeType

from ..common_xml import dm_visual_xml, dm_skybox_xml, dm_materials_xml


comptime _ball_in_cup_body = """
<mujoco model="ball in cup">

  <default>
    <motor ctrllimited="true" ctrlrange="-1 1" gear="5"/>
    <default class="cup">
      <joint type="slide" damping="3" stiffness="20"/>
      <geom type="capsule" size=".008" material="self"/>
    </default>
  </default>

  <worldbody>
    <light name="light" directional="true" diffuse=".6 .6 .6" pos="0 0 2" specular=".3 .3 .3"/>
    <geom name="ground" type="plane" pos="0 0 0" size=".6 .2 10" material="grid"/>
    <camera name="cam0" pos="0 -1 .8" xyaxes="1 0 0 0 1 2"/>
    <camera name="cam1" pos="0 -1 .4" xyaxes="1 0 0 0 0 1" />

    <body name="cup" pos="0 0 .6" childclass="cup">
      <joint name="cup_x" axis="1 0 0"/>
      <joint name="cup_z" axis="0 0 1"/>
      <geom name="cup_part_0" fromto="-.05 0 0 -.05 0 -.075" />
      <geom name="cup_part_1" fromto="-.05 0 -.075 -.025 0 -.1" />
      <geom name="cup_part_2" fromto="-.025 0 -.1 .025 0 -.1" />
      <geom name="cup_part_3" fromto=".025 0 -.1 .05 0 -.075" />
      <geom name="cup_part_4" fromto=".05 0 -.075 .05 0 0" />
      <site name="cup" pos="0 0 -.108" size=".005"/>
      <site name="target" type="box" pos="0 0 -.05" size=".05 .006 .05" group="4"/>
    </body>

    <body name="ball" pos="0 0 .2">
      <joint name="ball_x" type="slide" axis="1 0 0"/>
      <joint name="ball_z" type="slide" axis="0 0 1"/>
      <geom name="ball" type="sphere" size=".025" material="effector"/>
      <site name="ball" size=".005"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="x" joint="cup_x"/>
    <motor name="z" joint="cup_z"/>
  </actuator>

  <tendon>
    <spatial name="string" limited="true" range="0 0.3" width="0.003">
      <site site="ball"/>
      <site site="cup"/>
    </spatial>
  </tendon>

</mujoco>
"""


comptime dm_ball_in_cup_xml = merge_mjcf(
    dm_skybox_xml, dm_visual_xml, dm_materials_xml, _ball_in_cup_body
)

comptime bicp = parse_xml(dm_ball_in_cup_xml)

# obs = position (qpos, 4) + velocity (qvel, 4) = 8
comptime DMBallInCupModel = ModelDefFromXML[
    xml=dm_ball_in_cup_xml,
    nbody=bicp.NBODY, njoint=bicp.NJOINT, nq=bicp.NQ, nv=bicp.NV,
    ngeom=bicp.NGEOM, nact=bicp.NACT, ntex=bicp.NTEX, nmat=bicp.NMAT,
    nlight=bicp.NLIGHT, ncam=bicp.NCAM, nsite=bicp.NSITE,
    max_tendon=bicp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    # The ball can touch several cup capsules at once while it settles; 8 is
    # comfortably above the 3-4 MuJoCo reports at rest in the cup.
    max_contacts=8,
    obs_dim_override=8,
    obs_qpos_skip=0,
    timestep=bicp.TIMESTEP,
]

# --- Indices, in OUR ordering (== MuJoCo's here; the parity test pins both).
comptime BALL_BODY_IDX: Int = 2
comptime CUP_SITE_IDX: Int = 0
comptime TARGET_SITE_IDX: Int = 1
comptime BALL_SITE_IDX: Int = 2
comptime BALL_GEOM_IDX: Int = 6
comptime CUP_GEOM_FIRST: Int = 1  # cup_part_0
comptime CUP_GEOM_LAST: Int = 5  # cup_part_4

# `site_size['target', [0, 2]]` and `geom_size['ball', 0]`, which
# `Physics.in_target` differences. Asserted against the model tensors in the
# parity test rather than trusted.
comptime TARGET_HALF_X: Float64 = 0.05
comptime TARGET_HALF_Z: Float64 = 0.05
comptime BALL_RADIUS: Float64 = 0.025

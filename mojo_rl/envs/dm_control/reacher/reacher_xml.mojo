"""dm_control `reacher` model — port of `dm_control/suite/reacher.xml`.

Verbatim apart from the `<include>` lines and ONE deliberate substitution.

THE SUBSTITUTION — the target becomes a MOCAP BODY. The reference declares it
as a plain worldbody geom:

    <geom name="target" pos="0 0 .01" material="target" type="sphere" size=".05"/>

and then rewrites `model.geom_pos['target']` at every reset with a fresh polar
coordinate. Our `fields.Model` is a single SHARED, UNBATCHED tensor set, so a
model write is a write for every env in the batch — per-episode model mutation
(gap G4) is not expressible. A mocap body is the sanctioned workaround: FK
SKIPS mocap bodies and the facade presets their world pose from `d.mocap_pos`,
which is per-env `[BATCH, NBODY*3]` state. So the target moves per episode
without the model moving at all, and `geom_xpos(target)` reads back correctly
because the geom rides its body.

    <body name="target" mocap="true" pos="0 0 .01">
      <geom name="target" material="target" type="sphere" size=".05"/>
    </body>

This costs nothing physically: `<flag contact="disable"/>` is set model-wide,
the body carries no joint, and a jointless body contributes no DOF, so the
target is inert in both the reference and here. It does add one body to NBODY
(index 4, appended after the arm chain so the arm keeps indices 1..3), which
the parity test accounts for explicitly.

`geom_size['target', 0]` is the OTHER thing reset writes — `.05` for `easy`,
`.015` for `hard`. It feeds only the reward radius, never a contact, so the
port carries it as a config comptime (`DMReacherConfig.TARGET_SIZE`) rather
than a per-episode model write. The XML keeps the reference's `.05`.

GEOM ORDER, as always: ours is XML text order, MuJoCo's is sorted by body id.
Here MuJoCo puts `target` (a world geom) at 6, ahead of arm/hand/finger; ours
puts it at 9, behind them. The parity test pins both.
"""

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.parser.xml_parser import merge_mjcf

from ..common_xml import dm_visual_xml, dm_skybox_xml, dm_materials_xml


comptime _reacher_body = """
<mujoco model="two-link planar reacher">
  <option timestep="0.02">
    <flag contact="disable"/>
  </option>

  <default>
    <joint type="hinge" axis="0 0 1" damping="0.01"/>
    <motor gear=".05" ctrlrange="-1 1" ctrllimited="true"/>
  </default>

  <worldbody>
    <light name="light" pos="0 0 1"/>
    <camera name="fixed" pos="0 0 .75" quat="1 0 0 0"/>
    <geom name="ground" type="plane" pos="0 0 0" size=".3 .3 10" material="grid"/>
    <geom name="wall_x" type="plane" pos="-.3 0 .02" zaxis="1 0 0"  size=".02 .3 .02" material="decoration"/>
    <geom name="wall_y" type="plane" pos="0 -.3 .02" zaxis="0 1 0"  size=".3 .02 .02" material="decoration"/>
    <geom name="wall_neg_x" type="plane" pos=".3 0 .02" zaxis="-1 0 0"  size=".02 .3 .02" material="decoration"/>
    <geom name="wall_neg_y" type="plane" pos="0 .3 .02" zaxis="0 -1 0"  size=".3 .02 .02" material="decoration"/>

    <geom name="root" type="cylinder" fromto="0 0 0 0 0 0.02" size=".011" material="decoration"/>
    <body name="arm" pos="0 0 .01">
      <geom name="arm" type="capsule" fromto="0 0 0 0.12 0 0" size=".01" material="self"/>
      <joint name="shoulder"/>
      <body name="hand" pos=".12 0 0">
        <geom name="hand" type="capsule" fromto="0 0 0 0.1 0 0" size=".01" material="self"/>
        <joint name="wrist" limited="true" range="-160 160"/>
        <body name="finger" pos=".12 0 0">
          <camera name="hand" pos="0 0 .2" mode="track"/>
          <geom name="finger" type="sphere" size=".01" material="effector"/>
        </body>
      </body>
    </body>

    <body name="target" mocap="true" pos="0 0 .01">
      <geom name="target" material="target" type="sphere" size=".05"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="shoulder" joint="shoulder"/>
    <motor name="wrist" joint="wrist"/>
  </actuator>
</mujoco>
"""


comptime dm_reacher_xml = merge_mjcf(
    dm_skybox_xml, dm_visual_xml, dm_materials_xml, _reacher_body
)

comptime pmr = parse_xml(dm_reacher_xml)

# obs = position (qpos, 2) + to_target (2) + velocity (qvel, 2) = 6
comptime DMReacherModel = ModelDefFromXML[
    xml=dm_reacher_xml,
    nbody=pmr.NBODY, njoint=pmr.NJOINT, nq=pmr.NQ, nv=pmr.NV,
    ngeom=pmr.NGEOM, nact=pmr.NACT, ntex=pmr.NTEX, nmat=pmr.NMAT,
    nlight=pmr.NLIGHT, ncam=pmr.NCAM, nsite=pmr.NSITE,
    max_contacts=1,
    obs_dim_override=6,
    timestep=pmr.TIMESTEP,
]

# Body indices in worldbody DFS order (0 = world). `target` is appended last so
# the arm chain keeps the reference's own 1..3.
comptime ARM_BODY_IDX: Int = 1
comptime HAND_BODY_IDX: Int = 2
comptime FINGER_BODY_IDX: Int = 3
comptime TARGET_BODY_IDX: Int = 4

# Geom indices in OUR ordering (XML text order) — see the header note; these
# are NOT MuJoCo's, and the parity test pins both.
comptime FINGER_GEOM_IDX: Int = 8
comptime TARGET_GEOM_IDX: Int = 9

# `named.model.geom_size['finger', 0]`, the second half of the reward radius.
# Fixed by the XML in both tasks (only the TARGET radius varies).
comptime FINGER_SIZE: Float64 = 0.01

# The target's z, held constant by `initialize_episode` (it writes only x/y).
comptime TARGET_Z: Float64 = 0.01

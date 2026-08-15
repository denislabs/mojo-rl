"""`dm_control` `reacher` model — port of `dm_control/suite/reacher.xml`.

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
`.015` for `hard`. It never feeds a contact, so the REWARD side of it is a
config comptime (`DMReacherConfig.TARGET_SIZE`) rather than a per-episode model
write.

⚠ THAT IS NOT THE WHOLE STORY, and treating it as such was a bug. The radius is
also what the target is DRAWN at, and the renderer reads geom sizes at COMPILE
TIME (`MODEL_DEF.render_body_geoms`) — so one shared XML at `.05` drew `hard`'s
1.5 cm disc as a 5 cm ball, 3.3x too big. Hence two model defs,
`DMReacherModel` and `DMReacherHardModel`, differing in exactly that one
attribute. Nothing physical distinguishes them; the parity rollout is
parameterized over both precisely because it cannot tell them apart.

GEOM ORDER, as always: ours is XML text order, MuJoCo's is sorted by body id.
Here MuJoCo puts `target` (a world geom) at 6, ahead of arm/hand/finger; ours
puts it at 9, behind them. The parity test pins both.
"""

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.parser.xml_parser import merge_mjcf

from ..common_xml import dm_visual_xml, dm_skybox_xml, dm_materials_xml
from mojo_rl.envs.dm_control.reacher.reacher_dims import DM_REACHER_DIMS


comptime _reacher_body_pre = """
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
"""
"""The body, up to the TARGET GEOM — the one line the two tasks differ in.
`_reacher_body_post` resumes at the line after it."""


# ⚠ THE WHOLE LINE IS DUPLICATED, not spliced around the number. Splicing means
# ending the preceding chunk mid-attribute, right after `size=`, so the opening
# quote of the value has to reach the string through an escape — and an escaped
# quote sitting immediately before a triple-quote terminator reads as either the
# escape or the end of the string depending on how you squint. Two adjacent
# lines that differ in one number cannot drift unnoticed; a clever splice can.
comptime _target_geom_easy = """      <geom name="target" material="target" type="sphere" size=".05"/>
"""
comptime _target_geom_hard = """      <geom name="target" material="target" type="sphere" size=".015"/>
"""


comptime _reacher_body_post = """    </body>
  </worldbody>

  <actuator>
    <motor name="shoulder" joint="shoulder"/>
    <motor name="wrist" joint="wrist"/>
  </actuator>
</mujoco>
"""


comptime _reacher_body = (
    _reacher_body_pre + _target_geom_easy + _reacher_body_post
)
comptime _reacher_body_hard = (
    _reacher_body_pre + _target_geom_hard + _reacher_body_post
)


comptime dm_reacher_xml = merge_mjcf(
    dm_skybox_xml, dm_visual_xml, dm_materials_xml, _reacher_body
)
comptime dm_reacher_hard_xml = merge_mjcf(
    dm_skybox_xml, dm_visual_xml, dm_materials_xml, _reacher_body_hard
)

# ⚠ PARSED FROM THE EASY XML AND REUSED FOR BOTH. The two differ in one
# attribute VALUE, so every count `parse_xml` returns — nbody, ngeom, nq, nv,
# the timestep — is identical by construction. Parsing twice would only pay a
# second comptime XML parse to learn the same numbers.
comptime pmr = DM_REACHER_DIMS

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

comptime DMReacherHardModel = ModelDefFromXML[
    xml=dm_reacher_hard_xml,
    nbody=pmr.NBODY, njoint=pmr.NJOINT, nq=pmr.NQ, nv=pmr.NV,
    ngeom=pmr.NGEOM, nact=pmr.NACT, ntex=pmr.NTEX, nmat=pmr.NMAT,
    nlight=pmr.NLIGHT, ncam=pmr.NCAM, nsite=pmr.NSITE,
    max_contacts=1,
    obs_dim_override=6,
    timestep=pmr.TIMESTEP,
]
"""`hard`'s model — identical but for the target's `.015` radius.

⚠ A SECOND MODEL FOR A PURELY VISUAL DIFFERENCE, which is worth stating plainly
because it costs a second comptime model def. The target is INERT (contact is
disabled model-wide) and the reward measures against `DMReacherConfig`'s
`TARGET_SIZE`, so nothing physical reads this radius — but the RENDERER does,
and it reads it at COMPILE TIME (`MODEL_DEF.render_body_geoms`). There is no
runtime geom-size path to write instead, which is why `hard` used to draw
`easy`'s 5 cm ball: 3.3x too big, and the only visible symptom.

The alternative — dm_control's own `initialize_episode` writing
`geom_size['target', 0]` per episode — is not available to us for the same
reason: our `custom_reset_model_cpu` hook writes the HOST model lists, which
the comptime renderer never consults."""

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

"""`dm_control` `fish` model — port of `dm_control/suite/fish.xml`.

Verbatim apart from the `<include>` lines and ONE deliberate substitution, the
same gap-G4 one reacher, finger and swimmer needed.

THE SUBSTITUTION — THE TARGET BECOMES A MOCAP BODY. The reference declares it
as a static worldbody geom

    <geom name="target" type="sphere" pos="0 .4 .1" size=".04" material="target"/>

and `Swim.initialize_episode` then rewrites `model.geom_pos['target', 'xyz']`
every episode. `fields.Model` is one SHARED, UNBATCHED tensor set, so a model
write is a write for every env in the batch; `d.mocap_pos` is per-env state.
The geom rides its body at the body origin, so `geom_xpos['target']` is exactly
`d.xpos[TARGET_BODY_IDX]`.

`Upright.initialize_episode` also sets `geom_rgba['target', 3] = 0` to hide the
target — pure rendering, and nothing in that task reads the target at all.

WHAT MAKES THIS DOMAIN NEW (gap G3, and the reason it came after swimmer):

  * `<position>` ACTUATORS. All five are position servos, not torque motors:
    `force = kp*(ctrl - length) - kv*velocity`. The engine grew them for this
    model; see `ModelDefFromXML.apply_actions`. Note the force reads `qpos`,
    so it is recomputed every PHYSICS SUBSTEP rather than once per control
    step — `Phyics3dEnv.step` changed for that, and a `<motor>` model is
    bit-identical either way.

  * A TENDON TRANSMISSION. `fins_flap` actuates a fixed TENDON, not a joint:
    `length = -.5*finleft_roll + .5*finright_roll`, and its force is
    distributed back over both DOFs by the same coefficients. This is the
    only actuator in the port that touches more than one DOF.

  * A TENDON SPRING. `<fixed name="fins_sym" stiffness="1e-4">` is a passive
    spring pulling the two fin rolls toward symmetry. Same magnitude as the
    `kp` values around it, so it is not a rounding term.

  * `<flag constraint="disable"/>` on top of `gravity="disable"`. The whole
    constraint solver is off: no contacts, no joint limits (every joint is
    `limited="false"` anyway, despite carrying a `range`), no equality.
    The parser already reproduces both.

  * `density="5000"` — the fluid path again, the same one swimmer gated.
    Fish is neutrally buoyant-ish and gravity is off, so drag plus the fin
    servos are the entire dynamics.

⚠ `mass="0"` on the target geom is ours, as in swimmer: the reference's geom
is static-in-world so its density-derived mass never reaches the dynamics, but
on a mocap body it would reach `mj_fluid`'s per-body loop. It does not actually
make `body_mass[target]` zero — a body with no mass-CONTRIBUTING geom keeps the
staging default — and it does not need to: a mocap body has no DOF, so the
wrench projects to nothing and FK pins its velocity to zero.

⚠ `torso_massive` carries `group="4"`, and it is the ONLY geom on the torso
with mass. MuJoCo's `inertiagrouprange` default is "0 5" so it counts; a
narrower range would silently leave the torso massless.

GEOM ORDER, as always: ours is XML text order, MuJoCo's is sorted by body id
(which puts the world-attached `target` at index 1 there and last here). The
parity test pins the two indices this port reads by NAME on the reference side.

Reference: references/dm_control-main/dm_control/suite/fish.py + .xml
"""

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.parser.xml_parser import merge_mjcf

from ..common_xml import dm_visual_xml, dm_materials_xml


# fish.xml includes visual.xml and materials.xml but declares its OWN skybox
# inline rather than including `common/skybox.xml` — kept that way.
comptime _fish_body = """
<mujoco model="fish">
  <asset>
      <texture name="skybox" type="skybox" builtin="gradient" rgb1=".4 .6 .8" rgb2="0 0 0" width="800" height="800" mark="random" markrgb="1 1 1"/>
  </asset>

  <option timestep="0.004" density="5000">
    <flag gravity="disable" constraint="disable"/>
  </option>

  <default>
    <general ctrllimited="true"/>
    <default class="fish">
      <joint type="hinge" limited="false" range="-60 60" damping="2e-5" solreflimit=".1 1" solimplimit="0 .8 .1"/>
      <geom material="self"/>
    </default>
  </default>

  <worldbody>
    <camera name="tracking_top" pos="0 0 1" xyaxes="1 0 0 0 1 0" mode="trackcom"/>
    <camera name="tracking_x" pos="-.3 0 .2" xyaxes="0 -1 0 0.342 0 0.940" fovy="60" mode="trackcom"/>
    <camera name="tracking_y" pos="0 -.3 .2" xyaxes="1 0 0 0 0.342 0.940" fovy="60" mode="trackcom"/>
    <camera name="fixed_top" pos="0 0 5.5" fovy="10"/>
    <geom name="ground" type="plane" size=".5 .5 .1" material="grid"/>
    <body name="torso" pos="0 0 .1" childclass="fish">
      <light name="light" diffuse=".6 .6 .6" pos="0 0 0.5" dir="0 0 -1" specular=".3 .3 .3" mode="track"/>
      <joint name="root" type="free" damping="0" limited="false"/>
      <site name="torso" size=".01" rgba="0 0 0 0"/>
      <geom name="eye" type="ellipsoid" pos="0 .055 .015" size=".008 .012 .008" euler="-10 0 0" material="eye" mass="0"/>
      <camera name="eye" pos="0 .06 .02" xyaxes="1 0 0 0 0 1"/>
      <geom name="mouth" type="capsule" fromto="0 .079 0 0 .07 0" size=".005" material="effector" mass="0"/>
      <geom name="lower_mouth" type="capsule" fromto="0 .079 -.004 0 .07 -.003" size=".0045" material="effector" mass="0"/>
      <geom name="torso" type="ellipsoid" size=".01 .08 .04" mass="0"/>
      <geom name="back_fin" type="ellipsoid" size=".001 .03 .015" pos="0 -.03 .03" material="effector" mass="0"/>
      <geom name="torso_massive" type="box" size=".002 .06 .03" group="4"/>
      <body name="tail1" pos="0 -.09 0">
        <joint name="tail1" axis="0 0 1" pos="0 .01 0"/>
        <joint name="tail_twist" axis="0 1 0" pos="0 .01 0" range="-30 30"/>
        <geom name="tail1" type="ellipsoid" size=".001 .008 .016"/>
        <body name="tail2" pos="0 -.028 0">
          <joint name="tail2" axis="0 0 1" pos="0 .02 0" stiffness="8e-5"/>
          <geom name="tail2" type="ellipsoid" size=".001 .018 .035"/>
        </body>
      </body>
      <body name="finright" pos=".01 0 0">
        <joint name="finright_roll" axis="0 1 0"/>
        <joint name="finright_pitch" axis="1 0 0" pos="0 .005 0"/>
        <geom name="finright" type="ellipsoid" pos=".015 0 0" size=".02 .015 .001"  />
      </body>
      <body name="finleft" pos="-.01 0 0">
        <joint name="finleft_roll" axis="0 1 0"/>
        <joint name="finleft_pitch" axis="1 0 0" pos="0 .005 0"/>
        <geom name="finleft" type="ellipsoid"  pos="-.015 0 0" size=".02 .015 .001"/>
      </body>
    </body>
    <body name="target" mocap="true" pos="0 .4 .1">
      <geom name="target" type="sphere" size=".04" material="target" mass="0"/>
    </body>
  </worldbody>

  <tendon>
    <fixed name="fins_flap">
      <joint joint="finleft_roll"  coef="-.5"/>
      <joint joint="finright_roll" coef=".5"/>
    </fixed>
    <fixed name="fins_sym" stiffness="1e-4">
      <joint joint="finleft_roll"  coef=".5"/>
      <joint joint="finright_roll" coef=".5"/>
    </fixed>
  </tendon>

  <actuator>
    <position name="tail"           joint="tail1"           ctrlrange="-1 1"    kp="5e-4"/>
    <position name="tail_twist"     joint="tail_twist"      ctrlrange="-1 1"    kp="1e-4"/>
    <position name="fins_flap"      tendon="fins_flap"      ctrlrange="-1 1"    kp="3e-4"/>
    <position name="finleft_pitch"  joint="finleft_pitch"   ctrlrange="-1 1"    kp="1e-4"/>
    <position name="finright_pitch" joint="finright_pitch"  ctrlrange="-1 1"    kp="1e-4"/>
  </actuator>

  <sensor>
    <velocimeter name="velocimeter" site="torso"/>
    <gyro name="gyro" site="torso"/>
  </sensor>
</mujoco>
"""


comptime dm_fish_xml = merge_mjcf(dm_visual_xml, dm_materials_xml, _fish_body)

comptime pf = parse_xml(dm_fish_xml)


# `<sensor>` is neither an accumulator in `merge_mjcf` nor read by the parser,
# so both sensors are dropped on the way in. That costs nothing here: fish
# declares `velocimeter`/`gyro` and `Physics.torso_velocity()` reads them, but
# NEITHER task's `get_observation` calls it — the only velocity in the
# observation is `physics.velocity()`, i.e. the raw `qvel`.

# obs (upright) = joint_angles (7) + upright (1) + velocity (13) = 21
comptime DMFishUprightModel = ModelDefFromXML[
    xml=dm_fish_xml,
    nbody = pf.NBODY, njoint = pf.NJOINT, nq = pf.NQ, nv = pf.NV,
    ngeom = pf.NGEOM, nact = pf.NACT, ntex = pf.NTEX, nmat = pf.NMAT,
    nlight = pf.NLIGHT, ncam = pf.NCAM, nsite = pf.NSITE,
    max_contacts=1,
    # ⚠ REQUIRED, not optional. `max_tendon` sizes `_acd`'s tendon arrays;
    # unset it defaults to 0, which rounds up to one slot and silently drops
    # every tendon past the first. fish has two — `fins_flap` (an actuator
    # transmission) and `fins_sym` (a passive spring) — and ran with one from
    # cc7021d0 until this line was added. `ModelDefFromXML` now refuses to
    # build a model that under-declares it.
    max_tendon = pf.NTENDON,
    obs_dim_override=21,
    timestep = pf.TIMESTEP,
]

# obs (swim) = joint_angles (7) + upright (1) + target (3) + velocity (13) = 24
comptime DMFishSwimModel = ModelDefFromXML[
    xml=dm_fish_xml,
    nbody = pf.NBODY, njoint = pf.NJOINT, nq = pf.NQ, nv = pf.NV,
    ngeom = pf.NGEOM, nact = pf.NACT, ntex = pf.NTEX, nmat = pf.NMAT,
    nlight = pf.NLIGHT, ncam = pf.NCAM, nsite = pf.NSITE,
    max_contacts=1,
    # See the note on DMFishUprightModel above.
    max_tendon = pf.NTENDON,
    obs_dim_override=24,
    timestep = pf.TIMESTEP,
]


# ── Indices ──────────────────────────────────────────────────────────────────
#
#   body 0 world | 1 torso | 2 tail1 | 3 tail2 | 4 finright | 5 finleft
#        6 target (mocap, ours)
comptime TORSO_BODY_IDX: Int = 1
comptime TARGET_BODY_IDX: Int = 6

# Geoms in XML text order: ground, then the torso body's eye/mouth/lower_mouth/
# torso/back_fin/torso_massive, then tail1, tail2, finright, finleft, target.
comptime MOUTH_GEOM_IDX: Int = 2
comptime TARGET_GEOM_IDX: Int = 11

# `_JOINTS` (tail1, tail_twist, tail2, finright_roll, finright_pitch,
# finleft_roll, finleft_pitch) occupy qpos 7..13 contiguously, right after the
# free root's 7. `Physics.joint_angles()` is exactly that slice.
comptime N_ROOT_QPOS: Int = 7
comptime FREE_QUAT_ADR: Int = 3  # qpos[3:7] = (w, x, y, z), MuJoCo layout

# `radii = geom_size[['mouth', 'target'], 0].sum()` in `Swim.get_reward`:
# the mouth capsule's radius plus the target sphere's.
comptime MOUTH_RADIUS: Float64 = 0.005
comptime TARGET_RADIUS: Float64 = 0.04

# `initialize_episode` joint spread and target box (Swim only).
comptime JOINT_INIT_SPREAD: Float64 = 0.2
comptime TARGET_BOX_XY: Float64 = 0.4
comptime TARGET_Z_MIN: Float64 = 0.1
comptime TARGET_Z_MAX: Float64 = 0.3

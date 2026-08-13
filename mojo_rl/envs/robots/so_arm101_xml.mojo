"""SO-ARM101 (5 DOF + gripper) — model definition, `new_calib` variant.

Reference: `references/SO-ARM100-main/Simulation/SO101/`, vendored from
https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101 — the
model is in The Robot Studio's OWN repository, not in Menagerie and not in
`johnsutor/so101-nexus`. Generated upstream by `onshape-to-robot`.

Measured against MuJoCo 3.10.0: nbody 8, njnt 6 (all hinge), nq/nv/nu 6,
ngeom 31 (30 mesh, 1 plane), nsite 2, neq/ntendon/npair/nexclude/nkey 0,
cone pyramidal, impratio 1.

`so101_new_calib` is upstream's default: each joint's virtual zero is the
MIDDLE of its range. `so101_old_calib` puts zero at the fully-extended
horizontal pose. Only `new` is ported; `so_arm_bake.py` takes a `calib`
argument if the other is ever wanted.

RELATIONSHIP TO SO-100 — same robot, and NOT interchangeable. Same topology
(nbody 8 / njnt 6 / nq 6 / nu 6), and the two long links are BIT-IDENTICAL
(shoulder->elbow 0.116 m, elbow->wrist 0.135 m). But the base mount is 4 cm
shorter, and the inertials differ: **moving mass 0.485 kg here vs 0.609 kg**
for SO-100, with the moving jaw 68% apart. Under a stiff position servo a
policy will not notice; for torque-level sim-to-real it is the difference that
matters. ⚠ `qpos` is NOT portable between the two — SO-100 uses per-joint axes
`(0,1,0)/(1,0,0)/(0,0,1)`, SO-101 puts every axis on `(0,0,1)` and absorbs the
rest into body quaternions. Any cross-model comparison needs an explicit joint
mapping, not an array copy. See `docs/SO_ARM101_PORT_ASSESSMENT.md` §3.

⚠⚠ THE `fullinertia` DEVIATION IS THE ONE THAT MATTERS HERE. All seven bodies
spell their inertia as `<inertial ... fullinertia="ixx iyy izz ixy ixz iyz"/>`,
which `full_parser.mojo` RAISES on — honestly, and on both parser paths, since
`ModelDefFromXML.init_fields` goes through `parse_xml_full`. Rather than block
this port on a parser feature that belongs to ToddlerBot's phase 1a
(`docs/TODDLERBOT_PORT_PLAN.md` §4.5), `tests/robots/so_arm_bake.py`
diagonalises each tensor WITH MUJOCO and emits `quat` + `diaginertia` at 17
significant digits — the dog mesh-inertia bake precedent.

⚠ Gate the QUATERNION, not just the moments. A wrong `iquat` with a correct
`diaginertia` leaves total mass and every scalar moment right while silently
rotating each body's inertia frame. `so_arm_ref.py` diffs `body_iquat` AND
`body_inertia` per body at tolerance 0.0 for exactly that reason — §32.9 of
`docs/DM_CONTROL_PORT_PHASE2.md` is the precedent, where eigenvalues were
already right on 6 of 9 meshes while the frame was a different valid one.

⚠ WHEN `fullinertia` LANDS, switch `so_arm_bake.py` to emit the raw spelling
and keep the baked values as a REGRESSION FIXTURE — 7 near-symmetric robot
links are a much better probe of `mjuu_eig3`'s tie-breaking than a synthetic
tensor, and near-symmetric is exactly where those details bite.

⚠ THE TASK BODY IS INLINED BY THE BAKE, NOT BY `merge_mjcf` — and the reason
first recorded here was WRONG. That call did mangle this model (`<default>`
vanished; MuJoCo rejected it with "unknown default class name 'sts3215'"), but
NOT because the defaults are nested. `_extract_section_inner` depth-counts raw
text without stripping comments, and the comment the bake inserted contained
the literal `<default>` — that alone deleted the section. Measured: the same
fixture with an angle-bracket-free comment merges fine, and a CLEAN nested
model merges fine too. ⚠ Do not inherit "merge_mjcf cannot do nested defaults"
from this file; it can. Direct emission is kept anyway, as one less dependency
on a function with three recorded silent section drops. Full analysis:
`docs/PHYSICS3D_PARSER_GAPS_2026_08_13.md` §3.

⚠⚠ COLLISION IS 10x SO-100'S AND BUYS NOTHING PHYSICAL. Ten collidable meshes
totalling **26 198 convex-hull vertices** (raw 136 832) against SO-100's
2 456 — because upstream uses the RAW VISUAL MESHES as collision geometry.
Upstream's own README records that this behaved badly enough that they deleted
the base collision meshes outright. `mesh_max_poly/polyvert/edge` are linear
(2V/6V/8V) so this is compile time and support-function cost, not memory. If it
proves unworkable, the documented third option is grafting SO-100's
hand-authored collision onto these kinematics — label it a deviation if taken.
"""

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType


# --- BEGIN GENERATED XML (tests/robots/so_arm_bake.py) ---
comptime SO_ARM101_ROBOT_XML = """<?xml version="1.0" ?>
<!-- Generated using onshape-to-robot -->
<!-- Onshape https://cad.onshape.com/documents/7715cc284bb430fe6dab4ffd/w/4fd0791b683777b02f8d975a/e/826c553ede3b7592eb9ca800 -->
<mujoco model="so101_new_calib">
  <compiler angle="radian" autolimits="true"/>
  <default>
    <default class="so101_new_calib">
      <joint damping="1" frictionloss="0.1" armature="0.005"/>
      <position kp="50"/>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom group="3"/>
      </default>
    </default>
    <!-- merged from the reference's SECOND top-level default block; see so_arm_bake.py -->
    <default class="sts3215">
      <geom contype="0" conaffinity="0"/>
      <joint damping="0.60" frictionloss="0.052" armature="0.028"/>
      <!-- These gains are not a 1-to-1 mapping of the servo gains used in
           Lerobot. These position gains and forces were calculated according to
           https://github.com/Gregory119/RBE501-RL-arm-project/blob/main/gymnasium_env/README.md,
           assuming that the servo proportional gain is set to 16. -->
      <position kp="998.22" kv="2.731" forcerange="-2.94 2.94"/>
    </default>
    <default class="backlash">
      <!-- +/- 0.5° of backlash -->
      <joint damping="0.01" frictionloss="0" armature="0.01" limited="true" range="-0.008726646259971648 0.008726646259971648"/>
    </default>
  </default>
  <worldbody>
    <!-- Link base -->
    <body name="base" pos="0 0 0" quat="1 0 0 0" childclass="so101_new_calib">
      <inertial pos="0.0137179 -5.19711e-05 0.0334843" quat="0.42051918053114173 0.5587624782837298 0.5719799752087036 0.42870388338326826" mass="0.147" diaginertia="0.00013612687775945722 0.00013180727904032427 0.00011323284320021846"/><!-- DEVIATION: fullinertia diagonalised, see so_arm_bake.py -->
      <!-- Part base_motor_holder_so101_v1 -->
      <geom type="mesh" class="visual" pos="-0.00636471 -9.94414e-05 -0.0024" quat="0.5 0.5 0.5 0.5" mesh="base_motor_holder_so101_v1" material="base_motor_holder_so101_v1_material"/>
      <!-- Part base_so101_v2 -->
      <geom type="mesh" class="visual" pos="-0.00636471 -8.97657e-09 -0.0024" quat="0.5 0.5 0.5 0.5" mesh="base_so101_v2" material="base_so101_v2_material"/>
      <!-- Part sts3215_03a_v1 -->
      <geom type="mesh" class="visual" pos="0.0263353 -8.97657e-09 0.0437" quat="1 -2.85511e-16 -9.64433e-17 6.12908e-16" mesh="sts3215_03a_v1" material="sts3215_03a_v1_material"/>
      <!-- Part waveshare_mounting_plate_so101_v2 -->
      <geom type="mesh" class="visual" pos="-0.0309827 -0.000199441 0.0474" quat="0.5 0.5 0.5 0.5" mesh="waveshare_mounting_plate_so101_v2" material="waveshare_mounting_plate_so101_v2_material"/>
      <!-- Frame baseframe -->
      <site group="3" name="baseframe" pos="8.67362e-19 9.55596e-18 3.46945e-18" quat="1 -8.17396e-19 3.78392e-17 2.22045e-16"/>
      <!-- Link shoulder -->
      <body name="shoulder" pos="0.0388353 -8.97657e-09 0.0624" quat="3.56167e-16 1.22818e-15 -1 -4.14635e-16">
        <!-- Joint from base to shoulder -->
        <joint axis="0 0 1" name="shoulder_pan" type="hinge" range="-1.9198621771937616 1.9198621771937634" class="sts3215"/>
        <inertial pos="-0.0307604 -1.66727e-05 -0.0252713" quat="0.9998645689124817 0.0014956329102355124 0.00970839538046921 0.013204316521176531" mass="0.100006" diaginertia="8.378355122236958e-05 8.103880894259923e-05 2.3955239835031162e-05"/><!-- DEVIATION: fullinertia diagonalised, see so_arm_bake.py -->
        <!-- Part sts3215_03a_v1_2 -->
        <geom type="mesh" class="visual" pos="-0.0303992 0.000422241 -0.0417" quat="0.5 0.5 0.5 -0.5" mesh="sts3215_03a_v1" material="sts3215_03a_v1_material"/>
        <geom type="mesh" class="collision" pos="-0.0303992 0.000422241 -0.0417" quat="0.5 0.5 0.5 -0.5" mesh="sts3215_03a_v1" material="sts3215_03a_v1_material"/>
        <!-- Part motor_holder_so101_base_v1 -->
        <geom type="mesh" class="visual" pos="-0.0675992 -0.000177759 0.0158499" quat="0.5 0.5 -0.5 0.5" mesh="motor_holder_so101_base_v1" material="motor_holder_so101_base_v1_material"/>
        <geom type="mesh" class="collision" pos="-0.0675992 -0.000177759 0.0158499" quat="0.5 0.5 -0.5 0.5" mesh="motor_holder_so101_base_v1" material="motor_holder_so101_base_v1_material"/>
        <!-- Part rotation_pitch_so101_v1 -->
        <geom type="mesh" class="visual" pos="0.0122008 2.22413e-05 0.0464" quat="0.707107 -0.707107 -0 8.3163e-34" mesh="rotation_pitch_so101_v1" material="rotation_pitch_so101_v1_material"/>
        <geom type="mesh" class="collision" pos="0.0122008 2.22413e-05 0.0464" quat="0.707107 -0.707107 -0 8.3163e-34" mesh="rotation_pitch_so101_v1" material="rotation_pitch_so101_v1_material"/>
        <!-- Link upper_arm -->
        <body name="upper_arm" pos="-0.0303992 -0.0182778 -0.0542" quat="0.5 -0.5 -0.5 -0.5">
          <!-- Joint from shoulder to upper_arm -->
          <joint axis="0 0 1" name="shoulder_lift" type="hinge" range="-1.7453292519943224 1.7453292519943366" class="sts3215"/>
          <inertial pos="-0.0898471 -0.00838224 0.0184089" quat="0.45319470400751516 0.4540628989723458 0.5418911076086945 0.5429507081964211" mass="0.103" diaginertia="0.00015087316030509306 0.000142486983402127 3.724505629278006e-05"/><!-- DEVIATION: fullinertia diagonalised, see so_arm_bake.py -->
          <!-- Part sts3215_03a_v1_3 -->
          <geom type="mesh" class="visual" pos="-0.11257 -0.0155 0.0187" quat="4.56308e-16 -0.707107 0.707107 -1.37383e-16" mesh="sts3215_03a_v1" material="sts3215_03a_v1_material"/>
          <geom type="mesh" class="collision" pos="-0.11257 -0.0155 0.0187" quat="4.56308e-16 -0.707107 0.707107 -1.37383e-16" mesh="sts3215_03a_v1" material="sts3215_03a_v1_material"/>
          <!-- Part upper_arm_so101_v1 -->
          <geom type="mesh" class="visual" pos="-0.065085 0.012 0.0182" quat="0 1 0 0" mesh="upper_arm_so101_v1" material="upper_arm_so101_v1_material"/>
          <geom type="mesh" class="collision" pos="-0.065085 0.012 0.0182" quat="0 1 0 0" mesh="upper_arm_so101_v1" material="upper_arm_so101_v1_material"/>
          <!-- Link lower_arm -->
          <body name="lower_arm" pos="-0.11257 -0.028 1.73763e-16" quat="0.707107 -5.98613e-17 -2.58051e-17 0.707107">
            <!-- Joint from upper_arm to lower_arm -->
            <!-- Note: 5-degree calibration offset applied to joint range -->
            <joint axis="0 0 1" name="elbow_flex" type="hinge" range="-1.69 1.69" class="sts3215"/>
            <inertial pos="-0.0980701 0.00324376 0.0182831" quat="0.5107292891945808 0.5170056864986088 0.48796635480240474 0.4834765246294353" mass="0.104" diaginertia="0.0001602616995565295 0.00014530364353121692 2.8312456912253552e-05"/><!-- DEVIATION: fullinertia diagonalised, see so_arm_bake.py -->
            <!-- Part under_arm_so101_v1 -->
            <geom type="mesh" class="visual" pos="-0.0648499 -0.032 0.0182" quat="0 1 0 0" mesh="under_arm_so101_v1" material="under_arm_so101_v1_material"/>
            <geom type="mesh" class="collision" pos="-0.0648499 -0.032 0.0182" quat="0 1 0 0" mesh="under_arm_so101_v1" material="under_arm_so101_v1_material"/>
            <!-- Part motor_holder_so101_wrist_v1 -->
            <geom type="mesh" class="visual" pos="-0.0648499 -0.032 0.018" quat="3.92687e-16 -1 -1.9186e-15 -6.38378e-16" mesh="motor_holder_so101_wrist_v1" material="motor_holder_so101_wrist_v1_material"/>
            <geom type="mesh" class="collision" pos="-0.0648499 -0.032 0.018" quat="3.92687e-16 -1 -1.9186e-15 -6.38378e-16" mesh="motor_holder_so101_wrist_v1" material="motor_holder_so101_wrist_v1_material"/>
            <!-- Part sts3215_03a_v1_4 -->
            <geom type="mesh" class="visual" pos="-0.1224 0.0052 0.0187" quat="7.21645e-16 1.56949e-15 1 -3.33067e-16" mesh="sts3215_03a_v1" material="sts3215_03a_v1_material"/>
            <geom type="mesh" class="collision" pos="-0.1224 0.0052 0.0187" quat="7.21645e-16 1.56949e-15 1 -3.33067e-16" mesh="sts3215_03a_v1" material="sts3215_03a_v1_material"/>
            <!-- Link wrist -->
            <body name="wrist" pos="-0.1349 0.0052 3.62355e-17" quat="0.707107 9.58722e-16 -7.51313e-16 -0.707107">
              <!-- Joint from lower_arm to wrist -->
              <joint axis="0 0 1" name="wrist_flex" type="hinge" range="-1.6580628494556928 1.6580627293335335" class="sts3215"/>
              <inertial pos="-0.000103312 -0.0386143 0.0281156" quat="0.9671433911724339 0.25422629244203554 0.0016222629261781133 -0.00014630673012857546" mass="0.079" diaginertia="3.682647934724819e-05 2.744737720897244e-05 1.8943443443779328e-05"/><!-- DEVIATION: fullinertia diagonalised, see so_arm_bake.py -->
              <!-- Part sts3215_03a_no_horn_v1 -->
              <geom type="mesh" class="visual" pos="8.32667e-17 -0.0424 0.0306" quat="0.5 0.5 0.5 -0.5" mesh="sts3215_03a_no_horn_v1" material="sts3215_03a_no_horn_v1_material"/>
              <geom type="mesh" class="collision" pos="8.32667e-17 -0.0424 0.0306" quat="0.5 0.5 0.5 -0.5" mesh="sts3215_03a_no_horn_v1" material="sts3215_03a_no_horn_v1_material"/>
              <!-- Part wrist_roll_pitch_so101_v2 -->
              <geom type="mesh" class="visual" pos="0 -0.028 0.0181" quat="0.5 -0.5 -0.5 -0.5" mesh="wrist_roll_pitch_so101_v2" material="wrist_roll_pitch_so101_v2_material"/>
              <geom type="mesh" class="collision" pos="0 -0.028 0.0181" quat="0.5 -0.5 -0.5 -0.5" mesh="wrist_roll_pitch_so101_v2" material="wrist_roll_pitch_so101_v2_material"/>
              <!-- Link gripper -->
              <body name="gripper" pos="5.55112e-17 -0.0611 0.0181" quat="0.0172091 -0.0172091 0.706897 0.706897">
                <!-- Joint from wrist to gripper -->
                <joint axis="0 0 1" name="wrist_roll" type="hinge" range="-2.7438472969992493 2.841206309382605" class="sts3215"/>
                <inertial pos="0.000213627 0.000245138 -0.025187" quat="0.6007173837668219 0.35599307158417354 0.35862994993567826 0.6195095776770088" mass="0.087" diaginertia="4.337349303844789e-05 3.772288516824971e-05 2.428392179330235e-05"/><!-- DEVIATION: fullinertia diagonalised, see so_arm_bake.py -->
                <!-- Part sts3215_03a_v1_5 -->
                <geom type="mesh" class="visual" pos="0.0077 0.0001 -0.0234" quat="0.707107 -0.707107 1.66015e-15 6.45094e-15" mesh="sts3215_03a_v1" material="sts3215_03a_v1_material"/>
                <geom type="mesh" class="collision" pos="0.0077 0.0001 -0.0234" quat="0.707107 -0.707107 1.66015e-15 6.45094e-15" mesh="sts3215_03a_v1" material="sts3215_03a_v1_material"/>
                <!-- Part wrist_roll_follower_so101_v1 -->
                <geom type="mesh" class="visual" pos="8.32667e-17 -0.000218214 0.000949706" quat="0 1 0 0" mesh="wrist_roll_follower_so101_v1" material="wrist_roll_follower_so101_v1_material"/>
                <geom type="mesh" class="collision" pos="8.32667e-17 -0.000218214 0.000949706" quat="0 1 0 0" mesh="wrist_roll_follower_so101_v1" material="wrist_roll_follower_so101_v1_material"/>
                <!-- Frame gripperframe -->
                <site group="3" name="gripperframe" pos="-0.0079 -0.000218121 -0.0981274" quat="0.707107 -0 0.707107 -2.37788e-17"/>
                <!-- Link moving_jaw_so101_v1 -->
                <body name="moving_jaw_so101_v1" pos="0.0202 0.0188 -0.0234" quat="0.707107 0.707107 -1.85362e-08 1.85362e-08">
                  <!-- Joint from gripper to moving_jaw_so101_v1 -->
                  <joint axis="0 0 1" name="gripper" type="hinge" range="-0.17453297762778586 1.7453291995659765" class="sts3215"/>
                  <inertial pos="-0.00157495 -0.0300244 0.0192755" quat="0.6952650469411806 0.7179650313611695 -0.02456179937117522 -0.023009699174129962" mass="0.012" diaginertia="6.635823974947964e-06 5.290920121455255e-06 1.865225903596768e-06"/><!-- DEVIATION: fullinertia diagonalised, see so_arm_bake.py -->
                  <!-- Part moving_jaw_so101_v1 -->
                  <geom type="mesh" class="visual" pos="-5.55112e-17 -5.55112e-17 0.0189" quat="1 -0 3.00524e-16 -2.00834e-17" mesh="moving_jaw_so101_v1" material="moving_jaw_so101_v1_material"/>
                  <geom type="mesh" class="collision" pos="-5.55112e-17 -5.55112e-17 0.0189" quat="1 -0 3.00524e-16 -2.00834e-17" mesh="moving_jaw_so101_v1" material="moving_jaw_so101_v1_material"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    <!-- from SO101/scene.xml -->
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="groundplane"/>
  </worldbody>
  <asset>
    <mesh name="waveshare_mounting_plate_so101_v2" file="mojo_rl/envs/robots/assets/so_arm101/waveshare_mounting_plate_so101_v2.stl"/>
    <mesh name="sts3215_03a_v1" file="mojo_rl/envs/robots/assets/so_arm101/sts3215_03a_v1.stl"/>
    <mesh name="motor_holder_so101_base_v1" file="mojo_rl/envs/robots/assets/so_arm101/motor_holder_so101_base_v1.stl"/>
    <mesh name="wrist_roll_follower_so101_v1" file="mojo_rl/envs/robots/assets/so_arm101/wrist_roll_follower_so101_v1.stl"/>
    <mesh name="moving_jaw_so101_v1" file="mojo_rl/envs/robots/assets/so_arm101/moving_jaw_so101_v1.stl"/>
    <mesh name="base_motor_holder_so101_v1" file="mojo_rl/envs/robots/assets/so_arm101/base_motor_holder_so101_v1.stl"/>
    <mesh name="upper_arm_so101_v1" file="mojo_rl/envs/robots/assets/so_arm101/upper_arm_so101_v1.stl"/>
    <mesh name="wrist_roll_pitch_so101_v2" file="mojo_rl/envs/robots/assets/so_arm101/wrist_roll_pitch_so101_v2.stl"/>
    <mesh name="under_arm_so101_v1" file="mojo_rl/envs/robots/assets/so_arm101/under_arm_so101_v1.stl"/>
    <mesh name="rotation_pitch_so101_v1" file="mojo_rl/envs/robots/assets/so_arm101/rotation_pitch_so101_v1.stl"/>
    <mesh name="motor_holder_so101_wrist_v1" file="mojo_rl/envs/robots/assets/so_arm101/motor_holder_so101_wrist_v1.stl"/>
    <mesh name="sts3215_03a_no_horn_v1" file="mojo_rl/envs/robots/assets/so_arm101/sts3215_03a_no_horn_v1.stl"/>
    <mesh name="base_so101_v2" file="mojo_rl/envs/robots/assets/so_arm101/base_so101_v2.stl"/>
    <material name="base_motor_holder_so101_v1_material" rgba="1 0.82 0.12 1"/>
    <material name="base_so101_v2_material" rgba="1 0.82 0.12 1"/>
    <material name="sts3215_03a_v1_material" rgba="0.1 0.1 0.1 1"/>
    <material name="waveshare_mounting_plate_so101_v2_material" rgba="1 0.82 0.12 1"/>
    <material name="motor_holder_so101_base_v1_material" rgba="1 0.82 0.12 1"/>
    <material name="rotation_pitch_so101_v1_material" rgba="1 0.82 0.12 1"/>
    <material name="upper_arm_so101_v1_material" rgba="1 0.82 0.12 1"/>
    <material name="under_arm_so101_v1_material" rgba="1 0.82 0.12 1"/>
    <material name="motor_holder_so101_wrist_v1_material" rgba="1 0.82 0.12 1"/>
    <material name="sts3215_03a_no_horn_v1_material" rgba="0.1 0.1 0.1 1.0"/>
    <material name="wrist_roll_pitch_so101_v2_material" rgba="1 0.82 0.12 1"/>
    <material name="wrist_roll_follower_so101_v1_material" rgba="1 0.82 0.12 1"/>
    <material name="moving_jaw_so101_v1_material" rgba="1 0.82 0.12 1"/>
    <!-- from SO101/scene.xml -->
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <actuator>
    <position class="sts3215" name="shoulder_pan" joint="shoulder_pan" forcerange="-3.35 3.35" ctrlrange="-1.91986 1.91986"/>
    <position class="sts3215" name="shoulder_lift" joint="shoulder_lift" forcerange="-3.35 3.35" ctrlrange="-1.74533 1.74533"/>
    <position class="sts3215" name="elbow_flex" joint="elbow_flex" forcerange="-3.35 3.35" ctrlrange="-1.69 1.69"/>
    <position class="sts3215" name="wrist_flex" joint="wrist_flex" forcerange="-3.35 3.35" ctrlrange="-1.65806 1.65806"/>
    <position class="sts3215" name="wrist_roll" joint="wrist_roll" forcerange="-3.35 3.35" ctrlrange="-2.74385 2.84121"/>
    <position class="sts3215" name="gripper" joint="gripper" forcerange="-3.35 3.35" ctrlrange="-0.17453 1.74533"/>
  </actuator>
  <equality/>
</mujoco>
"""

comptime SO_ARM101_XML = """<?xml version="1.0" ?>
<!-- Generated using onshape-to-robot -->
<!-- Onshape https://cad.onshape.com/documents/7715cc284bb430fe6dab4ffd/w/4fd0791b683777b02f8d975a/e/826c553ede3b7592eb9ca800 -->
<mujoco model="so101_new_calib">
  <compiler angle="radian" autolimits="true"/>
  <default>
    <default class="so101_new_calib">
      <joint damping="1" frictionloss="0.1" armature="0.005"/>
      <position kp="50"/>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom group="3"/>
      </default>
    </default>
    <!-- merged from the reference's SECOND top-level default block; see so_arm_bake.py -->
    <default class="sts3215">
      <geom contype="0" conaffinity="0"/>
      <joint damping="0.60" frictionloss="0.052" armature="0.028"/>
      <!-- These gains are not a 1-to-1 mapping of the servo gains used in
           Lerobot. These position gains and forces were calculated according to
           https://github.com/Gregory119/RBE501-RL-arm-project/blob/main/gymnasium_env/README.md,
           assuming that the servo proportional gain is set to 16. -->
      <position kp="998.22" kv="2.731" forcerange="-2.94 2.94"/>
    </default>
    <default class="backlash">
      <!-- +/- 0.5° of backlash -->
      <joint damping="0.01" frictionloss="0" armature="0.01" limited="true" range="-0.008726646259971648 0.008726646259971648"/>
    </default>
  </default>
  <worldbody>
    <!-- Link base -->
    <body name="base" pos="0 0 0" quat="1 0 0 0" childclass="so101_new_calib">
      <inertial pos="0.0137179 -5.19711e-05 0.0334843" quat="0.42051918053114173 0.5587624782837298 0.5719799752087036 0.42870388338326826" mass="0.147" diaginertia="0.00013612687775945722 0.00013180727904032427 0.00011323284320021846"/><!-- DEVIATION: fullinertia diagonalised, see so_arm_bake.py -->
      <!-- Part base_motor_holder_so101_v1 -->
      <geom type="mesh" class="visual" pos="-0.00636471 -9.94414e-05 -0.0024" quat="0.5 0.5 0.5 0.5" mesh="base_motor_holder_so101_v1" material="base_motor_holder_so101_v1_material"/>
      <!-- Part base_so101_v2 -->
      <geom type="mesh" class="visual" pos="-0.00636471 -8.97657e-09 -0.0024" quat="0.5 0.5 0.5 0.5" mesh="base_so101_v2" material="base_so101_v2_material"/>
      <!-- Part sts3215_03a_v1 -->
      <geom type="mesh" class="visual" pos="0.0263353 -8.97657e-09 0.0437" quat="1 -2.85511e-16 -9.64433e-17 6.12908e-16" mesh="sts3215_03a_v1" material="sts3215_03a_v1_material"/>
      <!-- Part waveshare_mounting_plate_so101_v2 -->
      <geom type="mesh" class="visual" pos="-0.0309827 -0.000199441 0.0474" quat="0.5 0.5 0.5 0.5" mesh="waveshare_mounting_plate_so101_v2" material="waveshare_mounting_plate_so101_v2_material"/>
      <!-- Frame baseframe -->
      <site group="3" name="baseframe" pos="8.67362e-19 9.55596e-18 3.46945e-18" quat="1 -8.17396e-19 3.78392e-17 2.22045e-16"/>
      <!-- Link shoulder -->
      <body name="shoulder" pos="0.0388353 -8.97657e-09 0.0624" quat="3.56167e-16 1.22818e-15 -1 -4.14635e-16">
        <!-- Joint from base to shoulder -->
        <joint axis="0 0 1" name="shoulder_pan" type="hinge" range="-1.9198621771937616 1.9198621771937634" class="sts3215"/>
        <inertial pos="-0.0307604 -1.66727e-05 -0.0252713" quat="0.9998645689124817 0.0014956329102355124 0.00970839538046921 0.013204316521176531" mass="0.100006" diaginertia="8.378355122236958e-05 8.103880894259923e-05 2.3955239835031162e-05"/><!-- DEVIATION: fullinertia diagonalised, see so_arm_bake.py -->
        <!-- Part sts3215_03a_v1_2 -->
        <geom type="mesh" class="visual" pos="-0.0303992 0.000422241 -0.0417" quat="0.5 0.5 0.5 -0.5" mesh="sts3215_03a_v1" material="sts3215_03a_v1_material"/>
        <geom type="mesh" class="collision" pos="-0.0303992 0.000422241 -0.0417" quat="0.5 0.5 0.5 -0.5" mesh="sts3215_03a_v1" material="sts3215_03a_v1_material"/>
        <!-- Part motor_holder_so101_base_v1 -->
        <geom type="mesh" class="visual" pos="-0.0675992 -0.000177759 0.0158499" quat="0.5 0.5 -0.5 0.5" mesh="motor_holder_so101_base_v1" material="motor_holder_so101_base_v1_material"/>
        <geom type="mesh" class="collision" pos="-0.0675992 -0.000177759 0.0158499" quat="0.5 0.5 -0.5 0.5" mesh="motor_holder_so101_base_v1" material="motor_holder_so101_base_v1_material"/>
        <!-- Part rotation_pitch_so101_v1 -->
        <geom type="mesh" class="visual" pos="0.0122008 2.22413e-05 0.0464" quat="0.707107 -0.707107 -0 8.3163e-34" mesh="rotation_pitch_so101_v1" material="rotation_pitch_so101_v1_material"/>
        <geom type="mesh" class="collision" pos="0.0122008 2.22413e-05 0.0464" quat="0.707107 -0.707107 -0 8.3163e-34" mesh="rotation_pitch_so101_v1" material="rotation_pitch_so101_v1_material"/>
        <!-- Link upper_arm -->
        <body name="upper_arm" pos="-0.0303992 -0.0182778 -0.0542" quat="0.5 -0.5 -0.5 -0.5">
          <!-- Joint from shoulder to upper_arm -->
          <joint axis="0 0 1" name="shoulder_lift" type="hinge" range="-1.7453292519943224 1.7453292519943366" class="sts3215"/>
          <inertial pos="-0.0898471 -0.00838224 0.0184089" quat="0.45319470400751516 0.4540628989723458 0.5418911076086945 0.5429507081964211" mass="0.103" diaginertia="0.00015087316030509306 0.000142486983402127 3.724505629278006e-05"/><!-- DEVIATION: fullinertia diagonalised, see so_arm_bake.py -->
          <!-- Part sts3215_03a_v1_3 -->
          <geom type="mesh" class="visual" pos="-0.11257 -0.0155 0.0187" quat="4.56308e-16 -0.707107 0.707107 -1.37383e-16" mesh="sts3215_03a_v1" material="sts3215_03a_v1_material"/>
          <geom type="mesh" class="collision" pos="-0.11257 -0.0155 0.0187" quat="4.56308e-16 -0.707107 0.707107 -1.37383e-16" mesh="sts3215_03a_v1" material="sts3215_03a_v1_material"/>
          <!-- Part upper_arm_so101_v1 -->
          <geom type="mesh" class="visual" pos="-0.065085 0.012 0.0182" quat="0 1 0 0" mesh="upper_arm_so101_v1" material="upper_arm_so101_v1_material"/>
          <geom type="mesh" class="collision" pos="-0.065085 0.012 0.0182" quat="0 1 0 0" mesh="upper_arm_so101_v1" material="upper_arm_so101_v1_material"/>
          <!-- Link lower_arm -->
          <body name="lower_arm" pos="-0.11257 -0.028 1.73763e-16" quat="0.707107 -5.98613e-17 -2.58051e-17 0.707107">
            <!-- Joint from upper_arm to lower_arm -->
            <!-- Note: 5-degree calibration offset applied to joint range -->
            <joint axis="0 0 1" name="elbow_flex" type="hinge" range="-1.69 1.69" class="sts3215"/>
            <inertial pos="-0.0980701 0.00324376 0.0182831" quat="0.5107292891945808 0.5170056864986088 0.48796635480240474 0.4834765246294353" mass="0.104" diaginertia="0.0001602616995565295 0.00014530364353121692 2.8312456912253552e-05"/><!-- DEVIATION: fullinertia diagonalised, see so_arm_bake.py -->
            <!-- Part under_arm_so101_v1 -->
            <geom type="mesh" class="visual" pos="-0.0648499 -0.032 0.0182" quat="0 1 0 0" mesh="under_arm_so101_v1" material="under_arm_so101_v1_material"/>
            <geom type="mesh" class="collision" pos="-0.0648499 -0.032 0.0182" quat="0 1 0 0" mesh="under_arm_so101_v1" material="under_arm_so101_v1_material"/>
            <!-- Part motor_holder_so101_wrist_v1 -->
            <geom type="mesh" class="visual" pos="-0.0648499 -0.032 0.018" quat="3.92687e-16 -1 -1.9186e-15 -6.38378e-16" mesh="motor_holder_so101_wrist_v1" material="motor_holder_so101_wrist_v1_material"/>
            <geom type="mesh" class="collision" pos="-0.0648499 -0.032 0.018" quat="3.92687e-16 -1 -1.9186e-15 -6.38378e-16" mesh="motor_holder_so101_wrist_v1" material="motor_holder_so101_wrist_v1_material"/>
            <!-- Part sts3215_03a_v1_4 -->
            <geom type="mesh" class="visual" pos="-0.1224 0.0052 0.0187" quat="7.21645e-16 1.56949e-15 1 -3.33067e-16" mesh="sts3215_03a_v1" material="sts3215_03a_v1_material"/>
            <geom type="mesh" class="collision" pos="-0.1224 0.0052 0.0187" quat="7.21645e-16 1.56949e-15 1 -3.33067e-16" mesh="sts3215_03a_v1" material="sts3215_03a_v1_material"/>
            <!-- Link wrist -->
            <body name="wrist" pos="-0.1349 0.0052 3.62355e-17" quat="0.707107 9.58722e-16 -7.51313e-16 -0.707107">
              <!-- Joint from lower_arm to wrist -->
              <joint axis="0 0 1" name="wrist_flex" type="hinge" range="-1.6580628494556928 1.6580627293335335" class="sts3215"/>
              <inertial pos="-0.000103312 -0.0386143 0.0281156" quat="0.9671433911724339 0.25422629244203554 0.0016222629261781133 -0.00014630673012857546" mass="0.079" diaginertia="3.682647934724819e-05 2.744737720897244e-05 1.8943443443779328e-05"/><!-- DEVIATION: fullinertia diagonalised, see so_arm_bake.py -->
              <!-- Part sts3215_03a_no_horn_v1 -->
              <geom type="mesh" class="visual" pos="8.32667e-17 -0.0424 0.0306" quat="0.5 0.5 0.5 -0.5" mesh="sts3215_03a_no_horn_v1" material="sts3215_03a_no_horn_v1_material"/>
              <geom type="mesh" class="collision" pos="8.32667e-17 -0.0424 0.0306" quat="0.5 0.5 0.5 -0.5" mesh="sts3215_03a_no_horn_v1" material="sts3215_03a_no_horn_v1_material"/>
              <!-- Part wrist_roll_pitch_so101_v2 -->
              <geom type="mesh" class="visual" pos="0 -0.028 0.0181" quat="0.5 -0.5 -0.5 -0.5" mesh="wrist_roll_pitch_so101_v2" material="wrist_roll_pitch_so101_v2_material"/>
              <geom type="mesh" class="collision" pos="0 -0.028 0.0181" quat="0.5 -0.5 -0.5 -0.5" mesh="wrist_roll_pitch_so101_v2" material="wrist_roll_pitch_so101_v2_material"/>
              <!-- Link gripper -->
              <body name="gripper" pos="5.55112e-17 -0.0611 0.0181" quat="0.0172091 -0.0172091 0.706897 0.706897">
                <!-- Joint from wrist to gripper -->
                <joint axis="0 0 1" name="wrist_roll" type="hinge" range="-2.7438472969992493 2.841206309382605" class="sts3215"/>
                <inertial pos="0.000213627 0.000245138 -0.025187" quat="0.6007173837668219 0.35599307158417354 0.35862994993567826 0.6195095776770088" mass="0.087" diaginertia="4.337349303844789e-05 3.772288516824971e-05 2.428392179330235e-05"/><!-- DEVIATION: fullinertia diagonalised, see so_arm_bake.py -->
                <!-- Part sts3215_03a_v1_5 -->
                <geom type="mesh" class="visual" pos="0.0077 0.0001 -0.0234" quat="0.707107 -0.707107 1.66015e-15 6.45094e-15" mesh="sts3215_03a_v1" material="sts3215_03a_v1_material"/>
                <geom type="mesh" class="collision" pos="0.0077 0.0001 -0.0234" quat="0.707107 -0.707107 1.66015e-15 6.45094e-15" mesh="sts3215_03a_v1" material="sts3215_03a_v1_material"/>
                <!-- Part wrist_roll_follower_so101_v1 -->
                <geom type="mesh" class="visual" pos="8.32667e-17 -0.000218214 0.000949706" quat="0 1 0 0" mesh="wrist_roll_follower_so101_v1" material="wrist_roll_follower_so101_v1_material"/>
                <geom type="mesh" class="collision" pos="8.32667e-17 -0.000218214 0.000949706" quat="0 1 0 0" mesh="wrist_roll_follower_so101_v1" material="wrist_roll_follower_so101_v1_material"/>
                <!-- Frame gripperframe -->
                <site group="3" name="gripperframe" pos="-0.0079 -0.000218121 -0.0981274" quat="0.707107 -0 0.707107 -2.37788e-17"/>
                <!-- Link moving_jaw_so101_v1 -->
                <body name="moving_jaw_so101_v1" pos="0.0202 0.0188 -0.0234" quat="0.707107 0.707107 -1.85362e-08 1.85362e-08">
                  <!-- Joint from gripper to moving_jaw_so101_v1 -->
                  <joint axis="0 0 1" name="gripper" type="hinge" range="-0.17453297762778586 1.7453291995659765" class="sts3215"/>
                  <inertial pos="-0.00157495 -0.0300244 0.0192755" quat="0.6952650469411806 0.7179650313611695 -0.02456179937117522 -0.023009699174129962" mass="0.012" diaginertia="6.635823974947964e-06 5.290920121455255e-06 1.865225903596768e-06"/><!-- DEVIATION: fullinertia diagonalised, see so_arm_bake.py -->
                  <!-- Part moving_jaw_so101_v1 -->
                  <geom type="mesh" class="visual" pos="-5.55112e-17 -5.55112e-17 0.0189" quat="1 -0 3.00524e-16 -2.00834e-17" mesh="moving_jaw_so101_v1" material="moving_jaw_so101_v1_material"/>
                  <geom type="mesh" class="collision" pos="-5.55112e-17 -5.55112e-17 0.0189" quat="1 -0 3.00524e-16 -2.00834e-17" mesh="moving_jaw_so101_v1" material="moving_jaw_so101_v1_material"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    <!-- from SO101/scene.xml -->
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="groundplane"/>
    <body name="target" mocap="true" pos="0.25 0 0.2">
      <geom name="target" type="sphere" size="0.012" rgba="0.9 0.1 0.1 0.6"
            contype="0" conaffinity="0" group="1"/>
    </body>
  </worldbody>
  <asset>
    <mesh name="waveshare_mounting_plate_so101_v2" file="mojo_rl/envs/robots/assets/so_arm101/waveshare_mounting_plate_so101_v2.stl"/>
    <mesh name="sts3215_03a_v1" file="mojo_rl/envs/robots/assets/so_arm101/sts3215_03a_v1.stl"/>
    <mesh name="motor_holder_so101_base_v1" file="mojo_rl/envs/robots/assets/so_arm101/motor_holder_so101_base_v1.stl"/>
    <mesh name="wrist_roll_follower_so101_v1" file="mojo_rl/envs/robots/assets/so_arm101/wrist_roll_follower_so101_v1.stl"/>
    <mesh name="moving_jaw_so101_v1" file="mojo_rl/envs/robots/assets/so_arm101/moving_jaw_so101_v1.stl"/>
    <mesh name="base_motor_holder_so101_v1" file="mojo_rl/envs/robots/assets/so_arm101/base_motor_holder_so101_v1.stl"/>
    <mesh name="upper_arm_so101_v1" file="mojo_rl/envs/robots/assets/so_arm101/upper_arm_so101_v1.stl"/>
    <mesh name="wrist_roll_pitch_so101_v2" file="mojo_rl/envs/robots/assets/so_arm101/wrist_roll_pitch_so101_v2.stl"/>
    <mesh name="under_arm_so101_v1" file="mojo_rl/envs/robots/assets/so_arm101/under_arm_so101_v1.stl"/>
    <mesh name="rotation_pitch_so101_v1" file="mojo_rl/envs/robots/assets/so_arm101/rotation_pitch_so101_v1.stl"/>
    <mesh name="motor_holder_so101_wrist_v1" file="mojo_rl/envs/robots/assets/so_arm101/motor_holder_so101_wrist_v1.stl"/>
    <mesh name="sts3215_03a_no_horn_v1" file="mojo_rl/envs/robots/assets/so_arm101/sts3215_03a_no_horn_v1.stl"/>
    <mesh name="base_so101_v2" file="mojo_rl/envs/robots/assets/so_arm101/base_so101_v2.stl"/>
    <material name="base_motor_holder_so101_v1_material" rgba="1 0.82 0.12 1"/>
    <material name="base_so101_v2_material" rgba="1 0.82 0.12 1"/>
    <material name="sts3215_03a_v1_material" rgba="0.1 0.1 0.1 1"/>
    <material name="waveshare_mounting_plate_so101_v2_material" rgba="1 0.82 0.12 1"/>
    <material name="motor_holder_so101_base_v1_material" rgba="1 0.82 0.12 1"/>
    <material name="rotation_pitch_so101_v1_material" rgba="1 0.82 0.12 1"/>
    <material name="upper_arm_so101_v1_material" rgba="1 0.82 0.12 1"/>
    <material name="under_arm_so101_v1_material" rgba="1 0.82 0.12 1"/>
    <material name="motor_holder_so101_wrist_v1_material" rgba="1 0.82 0.12 1"/>
    <material name="sts3215_03a_no_horn_v1_material" rgba="0.1 0.1 0.1 1.0"/>
    <material name="wrist_roll_pitch_so101_v2_material" rgba="1 0.82 0.12 1"/>
    <material name="wrist_roll_follower_so101_v1_material" rgba="1 0.82 0.12 1"/>
    <material name="moving_jaw_so101_v1_material" rgba="1 0.82 0.12 1"/>
    <!-- from SO101/scene.xml -->
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <actuator>
    <position class="sts3215" name="shoulder_pan" joint="shoulder_pan" forcerange="-3.35 3.35" ctrlrange="-1.91986 1.91986"/>
    <position class="sts3215" name="shoulder_lift" joint="shoulder_lift" forcerange="-3.35 3.35" ctrlrange="-1.74533 1.74533"/>
    <position class="sts3215" name="elbow_flex" joint="elbow_flex" forcerange="-3.35 3.35" ctrlrange="-1.69 1.69"/>
    <position class="sts3215" name="wrist_flex" joint="wrist_flex" forcerange="-3.35 3.35" ctrlrange="-1.65806 1.65806"/>
    <position class="sts3215" name="wrist_roll" joint="wrist_roll" forcerange="-3.35 3.35" ctrlrange="-2.74385 2.84121"/>
    <position class="sts3215" name="gripper" joint="gripper" forcerange="-3.35 3.35" ctrlrange="-0.17453 1.74533"/>
  </actuator>
  <equality/>
</mujoco>
"""
# --- END GENERATED XML ---


comptime _pm = parse_xml(SO_ARM101_XML)

comptime SoArm101Model = ModelDefFromXML[
    xml=SO_ARM101_XML,
    nbody=_pm.NBODY,
    njoint=_pm.NJOINT,
    nq=_pm.NQ,
    nv=_pm.NV,
    ngeom=_pm.NGEOM,
    nact=_pm.NACT,
    ntex=_pm.NTEX,
    nmat=_pm.NMAT,
    nlight=_pm.NLIGHT,
    ncam=_pm.NCAM,
    nsite=_pm.NSITE,
    neq=_pm.NEQ,
    # ⚠⚠ `nexclude` AND `npair` DEFAULT TO 0, AND THE DROP IS SILENT. Omitting
    # `nexclude` here left SO-100's `<exclude body1="Base"
    # body2="Rotation_Pitch"/>` unbuilt — `parse_xml` reported NEXCLUDE 1 while
    # the MODEL carried 0, so the two adjacent base geoms would have collided
    # with each other forever. Caught by printing the model's counts against
    # `parse_xml`'s, which is why `test_so_arm10x_vs_mujoco` asserts BOTH.
    nexclude=_pm.NEXCLUDE,
    npair=_pm.NPAIR,
    timestep=_pm.TIMESTEP,
    # ⚠ PYRAMIDAL, unlike SO-100. Upstream sets no `<option cone>`, so this is
    # MuJoCo's default and `opt.cone` is gated at layer 1. Do not "align" the
    # two arms by making this elliptic — that would be a silent model change.
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=16,
    obs_dim_override=SO_ARM101_OBS_DIM,
    action_dim_override=6,
]

# Body indices, worldbody DFS order with world at 0.
comptime BASE_BODY_IDX: Int = 1
comptime SHOULDER_BODY_IDX: Int = 2
comptime UPPER_ARM_BODY_IDX: Int = 3
comptime LOWER_ARM_BODY_IDX: Int = 4
comptime WRIST_BODY_IDX: Int = 5
comptime GRIPPER_BODY_IDX: Int = 6
comptime MOVING_JAW_BODY_IDX: Int = 7
comptime TARGET_BODY_IDX: Int = 8

# qpos/qvel addresses. Six hinges in XML order.
comptime SHOULDER_PAN_ADR: Int = 0
comptime SHOULDER_LIFT_ADR: Int = 1
comptime ELBOW_FLEX_ADR: Int = 2
comptime WRIST_FLEX_ADR: Int = 3
comptime WRIST_ROLL_ADR: Int = 4
comptime GRIPPER_ADR: Int = 5

# Same layout as SO-100's, so one config shape serves both.
comptime SO_ARM101_OBS_DIM: Int = 21

# ⚠⚠ SIZED FROM **OUR** HULL: `fields_build` needs **32 934** vertices where
# MuJoCo's `mesh_graph` totals 26 198 — ours keeps 26% more. 33 280 is that
# rounded to a multiple of 512. A budget copied from `mjModel` raises at env
# construction; see `so_arm100_xml.mojo` for how to read the exact figure.
#
# ⚠ This is 13x SO-100's 2 551 and is the one place the two models genuinely
# diverge in cost — see the module docstring.
comptime SO_ARM101_NMESH_VERTS: Int = 33280

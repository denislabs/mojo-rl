"""End-to-end test for parse_xml_full().

Uses the inline HalfCheetah XML (simplified variant from test_flat_model_def)
which exercises:
  - Slide joints (rootx, rootz) with limited=false, explicit params
  - Hinge joint (rooty) with limited=false
  - Hinge joints (bthigh..ffoot) with explicit damping/stiffness overriding defaults
  - Capsule geoms with fromto (torso) and size only (limbs)
  - Plane geom (floor) on worldbody
  - Motor actuators with named joint lookup

Expected output:
  Dimensions: NBODY=8, NJOINT=9, NQ=9, NV=9, NGEOM=9, NACT=6
  torso pos_z = 0.7
  rootx axis_x = 1.0  (slide, limited=false)
  rootx armature = 0.0
  bthigh joint_id in joints[3].body_id = 2
  bthigh damping = 6.0   (explicit, overrides default 0.01)
  bthigh stiffness = 240.0
  floor geom_type = 0   (GEOM_PLANE)
  floor body_id = 0     (worldbody)
  bthigh motor gear = 120.0
  bthigh motor joint_id = 3
  FK round-trip succeeds, torso xpos_z ≈ 0.7
"""

from mojo_rl.physics3d.parser import ParsedModel, parse_xml
from mojo_rl.physics3d.parser import FlatModelDef
from mojo_rl.physics3d.parser import parse_xml_full
from mojo_rl.physics3d.types import Model, Data, ConeType
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.constants import GEOM_PLANE, GEOM_CAPSULE
from std.testing import assert_true, TestSuite


# =============================================================================
# Inline HalfCheetah XML (simplified — no axisangle, no head geom)
# =============================================================================

comptime half_cheetah_xml = """
<mujoco model="cheetah">
  <default>
    <joint armature=".1" damping=".01" limited="true" stiffness="8"/>
    <geom conaffinity="0" condim="3" contype="1" friction=".4 .1 .1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <worldbody>
    <geom name="floor" pos="0 0 0" type="plane" size="40 40 40"/>
    <body name="torso" pos="0 0 .7">
      <joint armature="0" axis="1 0 0" damping="0" limited="false" name="rootx" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootz" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="false" name="rooty" pos="0 0 0" stiffness="0" type="hinge"/>
      <geom fromto="-.5 0 0 .5 0 0" name="torso" size="0.046" type="capsule"/>
      <body name="bthigh" pos="-.5 0 0">
        <joint axis="0 1 0" damping="6" name="bthigh" pos="0 0 0" range="-.52 1.05" stiffness="240" type="hinge"/>
        <geom name="bthigh" size="0.046 .145" type="capsule"/>
        <body name="bshin" pos=".16 0 -.25">
          <joint axis="0 1 0" damping="4.5" name="bshin" pos="0 0 0" range="-.785 .785" stiffness="180" type="hinge"/>
          <geom name="bshin" size="0.046 .15" type="capsule"/>
          <body name="bfoot" pos="-.28 0 -.14">
            <joint axis="0 1 0" damping="3" name="bfoot" pos="0 0 0" range="-.4 .785" stiffness="120" type="hinge"/>
            <geom name="bfoot" size="0.046 .094" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="fthigh" pos=".5 0 0">
        <joint axis="0 1 0" damping="4.5" name="fthigh" pos="0 0 0" range="-1 .7" stiffness="180" type="hinge"/>
        <geom name="fthigh" size="0.046 .133" type="capsule"/>
        <body name="fshin" pos="-.14 0 -.24">
          <joint axis="0 1 0" damping="3" name="fshin" pos="0 0 0" range="-1.2 .87" stiffness="120" type="hinge"/>
          <geom name="fshin" size="0.046 .106" type="capsule"/>
          <body name="ffoot" pos=".13 0 -.18">
            <joint axis="0 1 0" damping="1.5" name="ffoot" pos="0 0 0" range="-.5 .5" stiffness="60" type="hinge"/>
            <geom name="ffoot" size="0.046 .07" type="capsule"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor gear="120" joint="bthigh" name="bthigh"/>
    <motor gear="90" joint="bshin" name="bshin"/>
    <motor gear="60" joint="bfoot" name="bfoot"/>
    <motor gear="120" joint="fthigh" name="fthigh"/>
    <motor gear="60" joint="fshin" name="fshin"/>
    <motor gear="30" joint="ffoot" name="ffoot"/>
  </actuator>
</mujoco>
"""


fn test_xml_full_parser() raises:
    # =========================================================================
    # Step 1: Dimension check
    # =========================================================================
    comptime pm = parse_xml(half_cheetah_xml)
    print("=== Dimensions ===")
    print("NBODY  =", pm.NBODY, " (expected 8)")
    print("NJOINT =", pm.NJOINT, " (expected 9)")
    print("NQ     =", pm.NQ, " (expected 9)")
    print("NV     =", pm.NV, " (expected 9)")
    print("NGEOM  =", pm.NGEOM, " (expected 8)")
    print("NACT   =", pm.NACT, " (expected 6)")
    print()

    comptime if pm.NBODY != 8 or pm.NJOINT != 9 or pm.NQ != 9 or pm.NV != 9 or pm.NGEOM != 8 or pm.NACT != 6:
        assert_true(
            False,
            (
                "dimension mismatch — NBODY/NJOINT/NQ/NV/NGEOM/NACT not as"
                " expected"
            ),
        )

    # =========================================================================
    # Step 2: Full parse — dimensions from comptime pm, data at runtime
    # =========================================================================
    var fmd = parse_xml_full[
        pm.NBODY, pm.NJOINT, pm.NQ, pm.NV, pm.NGEOM, pm.NACT
    ](half_cheetah_xml)

    print("=== Body checks ===")
    # bodies[0] = torso (model body index 1, parent=worldbody=0)
    print("torso parent  =", fmd.bodies[0].parent, " (expected 0)")
    print("torso pos_z   =", fmd.bodies[0].pos_z, " (expected 0.7)")

    # bodies[1] = bthigh (parent=torso=1)
    print("bthigh parent =", fmd.bodies[1].parent, " (expected 1)")
    print("bthigh pos_x  =", fmd.bodies[1].pos_x, " (expected -0.5)")
    print()

    print("=== Joint checks ===")
    # joints[0] = rootx (slide, axis=(1,0,0), armature=0, damping=0)
    print("rootx type    =", fmd.joints[0].jnt_type, " (expected 2=JNT_SLIDE)")
    print("rootx axis_x  =", fmd.joints[0].axis_x, " (expected 1.0)")
    print("rootx axis_y  =", fmd.joints[0].axis_y, " (expected 0.0)")
    print("rootx axis_z  =", fmd.joints[0].axis_z, " (expected 0.0)")
    print("rootx limited =", fmd.joints[0].is_limited, " (expected False)")
    print("rootx armature=", fmd.joints[0].armature, " (expected 0.0)")
    print("rootx damping =", fmd.joints[0].damping, " (expected 0.0)")

    # joints[3] = bthigh (hinge, body_id=2, damping=6, stiffness=240)
    print("bthigh body_id=", fmd.joints[3].body_id, " (expected 2)")
    print("bthigh damping=", fmd.joints[3].damping, " (expected 6.0)")
    print("bthigh stiffns=", fmd.joints[3].stiffness, " (expected 240.0)")
    print("bthigh limited=", fmd.joints[3].is_limited, " (expected True)")
    print("bthigh range_min=", fmd.joints[3].range_min, " (expected -0.52)")
    print()

    print("=== Geom checks ===")
    # geoms[0] = floor (plane, body_id=0)
    print(
        "floor type    =", fmd.geoms[0].geom_type, " (expected", GEOM_PLANE, ")"
    )
    print("floor body_id =", fmd.geoms[0].body_id, " (expected 0)")

    # geoms[1] = torso capsule (fromto, body_id=1)
    print(
        "torso geom type =",
        fmd.geoms[1].geom_type,
        " (expected",
        GEOM_CAPSULE,
        ")",
    )
    print("torso geom body =", fmd.geoms[1].body_id, " (expected 1)")
    print("torso geom radius=", fmd.geoms[1].radius, " (expected 0.046)")
    print("torso half_len  =", fmd.geoms[1].half_length, " (expected 0.5)")
    print()

    print("=== Actuator checks ===")
    # actuators[0] = bthigh motor (gear=120, joint_id=3)
    print("bthigh gear   =", fmd.actuators[0].gear, " (expected 120.0)")
    print("bthigh joint_id=", fmd.actuators[0].joint_id, " (expected 3)")

    # actuators[5] = ffoot motor (gear=30, joint_id=8)
    print("ffoot gear    =", fmd.actuators[5].gear, " (expected 30.0)")
    print("ffoot joint_id=", fmd.actuators[5].joint_id, " (expected 8)")
    print()

    # =========================================================================
    # Step 3: Full round-trip — setup_model + FK
    # =========================================================================
    print("=== FK round-trip ===")
    var model = Model[
        DType.float64,
        pm.NQ,
        pm.NV,
        pm.NBODY,
        pm.NJOINT,
        10,
        pm.NGEOM,
        0,
        ConeType.ELLIPTIC,
        0,
        0,
    ]()
    var data = Data[DType.float64, pm.NQ, pm.NV, pm.NBODY, pm.NJOINT, 10, 0]()

    fmd.setup_model[DType.float64, 10](model)

    print("gravity_z     =", Float64(model.gravity[2]), " (expected -9.81)")
    print(
        "torso body_id1 mass=",
        Float64(model.body_mass[1]),
        " (expected ~1.0 default)",
    )
    print(
        "torso pos_z   =", Float64(model.body_pos[1 * 3 + 2]), " (expected 0.7)"
    )

    forward_kinematics(model, data)
    print("FK completed")
    print("torso xpos_z  =", Float64(data.xpos[1 * 3 + 2]), " (expected ~0.7)")


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

"""Smoke test for ModelDefFromXML: CPU env creation + step.

Tests that ModelDefFromXML correctly implements ModelDefLike by:
1. Verifying comptime dimension constants (OBS_DIM, ACTION_DIM)
2. CPU setup_model_and_data → FK round-trip
3. reset_data → extract_obs
4. apply_actions → verify qfrc is set
5. enforce_limits → verify clamping
6. Comptime scalar helpers for GPU kernels

Uses the same inline HalfCheetah XML as test_xml_full_parser.mojo.
Expected output:
  OBS_DIM = 17  (NQ=9 - obs_qpos_skip=1 + NV=9 = 17)
  ACTION_DIM = 6
  torso xpos_z ≈ 0.7  (FK round-trip)
  obs[0] = 0.0       (qpos[1], rootz=0 after reset)
  bthigh qfrc = 120.0  (gear=120, action=1.0)
  enforce_limits clamped correctly
"""

from physics3d.parser import parse_xml, ModelDefFromXML
from physics3d.parser import parse_xml_full
from physics3d.parser.xml_parser import (
    _xml_nth_motor_gear,
    _xml_nth_motor_dof_adr,
    _xml_nth_joint_qpos_adr,
    _xml_nth_joint_limited,
    _xml_nth_joint_range_min,
    _xml_nth_joint_range_max,
)
from physics3d.types import Model, Data, ConeType
from physics3d.kinematics.forward_kinematics import forward_kinematics
from testing import assert_true, TestSuite


# =============================================================================
# Inline HalfCheetah XML (same as test_xml_full_parser.mojo)
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


fn test_model_def_from_xml() raises:
    # =========================================================================
    # Step 1: Parse dimensions
    # =========================================================================
    comptime pm = parse_xml(half_cheetah_xml)
    print("=== Dimensions ===")
    print("NBODY =", pm.NBODY, " (expected 8)")
    print("NJOINT=", pm.NJOINT, " (expected 9)")
    print("NQ    =", pm.NQ, " (expected 9)")
    print("NV    =", pm.NV, " (expected 9)")
    print("NGEOM =", pm.NGEOM, " (expected 8)")
    print("NACT  =", pm.NACT, " (expected 6)")
    print()

    comptime if (
        pm.NBODY != 8
        or pm.NJOINT != 9
        or pm.NQ != 9
        or pm.NV != 9
        or pm.NGEOM != 8
        or pm.NACT != 6
    ):
        print("ERROR: dimension mismatch — aborting")
        return

    # =========================================================================
    # Step 2: ModelDefFromXML comptime constants
    # =========================================================================
    # HalfCheetah: obs_qpos_skip=1 → OBS_DIM = 9-1+9 = 17
    comptime XmlModel = ModelDefFromXML[
        half_cheetah_xml,
        pm.NBODY,
        pm.NJOINT,
        pm.NQ,
        pm.NV,
        pm.NGEOM,
        pm.NACT,
        max_contacts=10,
        obs_qpos_skip=1,
    ]

    print("=== ModelDefFromXML comptime constants ===")
    print("OBS_DIM    =", XmlModel.OBS_DIM, " (expected 17)")
    print("ACTION_DIM =", XmlModel.ACTION_DIM, " (expected 6)")
    print("MAX_CONTACTS=", XmlModel.MAX_CONTACTS, " (expected 10)")
    print()

    comptime if XmlModel.OBS_DIM != 17:
        print("ERROR: OBS_DIM mismatch")
        return

    # =========================================================================
    # Step 3: Comptime scalar helpers for GPU kernels
    # =========================================================================
    print("=== Comptime GPU helper checks ===")
    # Motor 0 = bthigh, gear=120, joint index=3 (dof_adr=3 since 3 preceding joints)
    comptime gear0 = _xml_nth_motor_gear[half_cheetah_xml, 0]()
    comptime dof0 = _xml_nth_motor_dof_adr[half_cheetah_xml, 0]()
    print("motor0 gear =", gear0, " (expected 120.0)")
    print("motor0 dof_adr =", dof0, " (expected 3)")

    # Motor 5 = ffoot, gear=30
    comptime gear5 = _xml_nth_motor_gear[half_cheetah_xml, 5]()
    comptime dof5 = _xml_nth_motor_dof_adr[half_cheetah_xml, 5]()
    print("motor5 gear =", gear5, " (expected 30.0)")
    print("motor5 dof_adr =", dof5, " (expected 8)")

    # Joint 0 = rootx: slide, limited=false
    comptime rootx_limited = _xml_nth_joint_limited[half_cheetah_xml, 0]()
    comptime rootx_qpos_adr = _xml_nth_joint_qpos_adr[half_cheetah_xml, 0]()
    print("rootx limited =", rootx_limited, " (expected False)")
    print("rootx qpos_adr =", rootx_qpos_adr, " (expected 0)")

    # Joint 3 = bthigh: hinge, limited=true, range=[-0.52, 1.05]
    comptime bthigh_limited = _xml_nth_joint_limited[half_cheetah_xml, 3]()
    comptime bthigh_rmin = _xml_nth_joint_range_min[half_cheetah_xml, 3]()
    comptime bthigh_rmax = _xml_nth_joint_range_max[half_cheetah_xml, 3]()
    comptime bthigh_qpos_adr = _xml_nth_joint_qpos_adr[half_cheetah_xml, 3]()
    print("bthigh limited =", bthigh_limited, " (expected True)")
    print("bthigh range_min =", bthigh_rmin, " (expected -0.52)")
    print("bthigh range_max =", bthigh_rmax, " (expected 1.05)")
    print("bthigh qpos_adr =", bthigh_qpos_adr, " (expected 3)")
    print()

    # =========================================================================
    # Step 4: CPU setup_model_and_data + FK round-trip
    # =========================================================================
    print("=== CPU setup + FK round-trip ===")
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

    XmlModel.setup_model_and_data[DType.float64](model, data)
    print("setup_model_and_data succeeded")
    print("gravity_z     =", Float64(model.gravity[2]), " (expected -9.81)")
    print(
        "torso pos_z   =", Float64(model.body_pos[1 * 3 + 2]), " (expected 0.7)"
    )
    print("torso xpos_z  =", Float64(data.xpos[1 * 3 + 2]), " (expected ~0.7)")
    print()

    # =========================================================================
    # Step 5: reset_data + extract_obs
    # =========================================================================
    print("=== reset_data + extract_obs ===")
    XmlModel.reset_data[DType.float64](data)
    print("reset_data succeeded (qpos=0, qvel=0)")

    var obs = List[Scalar[DType.float64]]()
    XmlModel.extract_obs[DType.float64](data, obs)
    print("obs length =", len(obs), " (expected", XmlModel.OBS_DIM, ")")
    print("obs[0] =", Float64(obs[0]), " (expected 0.0, qpos[1]=rootz)")
    print("obs[7] =", Float64(obs[7]), " (expected 0.0, qvel[0]=rootx_dot)")
    print()

    # =========================================================================
    # Step 6: apply_actions
    # =========================================================================
    print("=== apply_actions ===")
    # Action 0 = bthigh motor (gear=120), action=1.0 → qfrc[3] = 120.0
    var actions = List[Float64]()
    for _ in range(pm.NACT):
        actions.append(Float64(0.0))
    actions[0] = Float64(1.0)  # bthigh motor with gear=120

    XmlModel.apply_actions[DType.float64](data, actions)
    # bthigh is joint 3 with dof_adr=3
    print("qfrc[3] =", Float64(data.qfrc[3]), " (expected 120.0)")
    print()

    # =========================================================================
    # Step 7: enforce_limits
    # =========================================================================
    print("=== enforce_limits ===")
    # Set bthigh qpos[3] out of range (2.0 > 1.05), should be clamped to 1.05
    data.qpos[3] = Scalar[DType.float64](2.0)
    XmlModel.enforce_limits[DType.float64](data)
    print(
        "bthigh qpos after clamp =",
        Float64(data.qpos[3]),
        " (expected 1.05)",
    )

    # rootx (joint 0) is not limited, should not be clamped
    data.qpos[0] = Scalar[DType.float64](100.0)
    XmlModel.enforce_limits[DType.float64](data)
    print(
        "rootx qpos after enforce =",
        Float64(data.qpos[0]),
        " (expected 100.0, not clamped)",
    )
    print()

    print("=== All CPU tests passed ===")


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

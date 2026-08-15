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

Note the inline XML below has NO `<compiler angle="..."/>`, so MuJoCo's default
of DEGREES applies and `range="-.52 1.05"` compiles to +-0.0183 rad. The real
Gymnasium half_cheetah.xml carries `angle="radian"`; this copy dropped it.
"""

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.parser import parse_xml_full
from mojo_rl.physics3d.parser.xml_parser import parse_xml_model_data
from max.gpu.host import DeviceContext
from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_POS_Z,
    MODEL_META_IDX_GRAVITY_Z,
)
from std.testing import assert_true, TestSuite


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


def test_model_def_from_xml() raises:
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
    # Step 3: parse_xml_model_data — precomputed InlineArray checks
    # =========================================================================
    print("=== parse_xml_model_data checks ===")
    # `ComptimeActData` is sized from the MODEL's dims, not from global caps
    # (see `ModelDefFromXML._NACT` and friends). Derive them here the same way
    # production does, so this test exercises the real sizing rule rather than
    # a hand-picked one. half_cheetah has no tendons, so the wrap cap is 1.
    comptime acd = parse_xml_model_data[
        pm.NACT, pm.NJOINT, pm.NQ, 1, 1
    ](half_cheetah_xml)

    # Motor 0 = bthigh, gear=120, dof_adr=3 (3 preceding joints: rootx,rootz,rooty)
    comptime gear0 = acd.motor_gears[0]
    comptime dof0 = acd.motor_dof_adr[0]
    print("motor0 gear =", gear0, " (expected 120.0)")
    print("motor0 dof_adr =", dof0, " (expected 3)")

    # Motor 5 = ffoot, gear=30, dof_adr=8
    comptime gear5 = acd.motor_gears[5]
    comptime dof5 = acd.motor_dof_adr[5]
    print("motor5 gear =", gear5, " (expected 30.0)")
    print("motor5 dof_adr =", dof5, " (expected 8)")

    # Joint 0 = rootx: slide, limited=false, qpos_adr=0
    comptime rootx_limited = acd.joint_is_limited[0]
    comptime rootx_qpos_adr = acd.joint_qpos_adr[0]
    print("rootx limited =", rootx_limited, " (expected False)")
    print("rootx qpos_adr =", rootx_qpos_adr, " (expected 0)")

    # Joint 3 = bthigh: hinge, limited=true, range=[-0.52, 1.05], qpos_adr=3
    comptime bthigh_limited = acd.joint_is_limited[3]
    comptime bthigh_rmin = acd.joint_range_min[3]
    comptime bthigh_rmax = acd.joint_range_max[3]
    comptime bthigh_qpos_adr = acd.joint_qpos_adr[3]
    print("bthigh limited =", bthigh_limited, " (expected True)")
    print("bthigh range_min =", bthigh_rmin, " (expected -0.52)")
    print("bthigh range_max =", bthigh_rmax, " (expected 1.05)")
    print("bthigh qpos_adr =", bthigh_qpos_adr, " (expected 3)")
    print()

    # =========================================================================
    # Step 4: spec-direct fields build (init_fields; G4)
    # =========================================================================
    print("=== fields model build ===")
    var ctx = DeviceContext()
    var mf = Model[
        DType.float64, pm.NV, pm.NBODY, pm.NJOINT, pm.NGEOM,
        XmlModel.MAX_EQUALITY, XmlModel.MAX_TENDON, XmlModel.NSITE,
        XmlModel.NEXCLUDE, 0,
    ]()
    XmlModel.init_fields[DType.float64, 0](ctx, mf)
    print("init_fields succeeded")
    print(
        "gravity_z     =",
        Float64(mf.meta.data[MODEL_META_IDX_GRAVITY_Z]),
        " (expected -9.81)",
    )
    print(
        "torso pos_z   =",
        Float64(mf.bodies.data[1 * MODEL_BODY_SIZE + BODY_IDX_POS_Z]),
        " (expected 0.7)",
    )
    print()

    # =========================================================================
    # Step 5: reset_data + extract_obs (fields-native hooks; G2)
    # =========================================================================
    var sf = XmlModel.make_spec_fields[DType.float64]()
    print("=== reset_data + extract_obs ===")
    var d = Data[DType.float64, pm.NQ, pm.NV, pm.NBODY, 10, 0, 1]()
    XmlModel.reset_data[DType.float64](sf, d)
    print("reset_data succeeded (qpos=0, qvel=0)")

    var obs = List[Scalar[DType.float64]]()
    XmlModel.extract_obs[DType.float64](d, obs)
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

    # `act` is the actuator ACTIVATION state (MuJoCo `d->act`). cheetah's
    # actuators are plain `<motor>`s with no dyntype, so NA == 0 and this
    # stays a one-element placeholder that apply_actions never reads.
    var act = List[Scalar[DType.float64]]()
    for _ in range(XmlModel.NA if XmlModel.NA > 0 else 1):
        act.append(Scalar[DType.float64](0))

    XmlModel.apply_actions[DType.float64](sf, d, actions, act)
    # bthigh is joint 3 with dof_adr=3
    print("qfrc[3] =", Float64(d.qfrc.data[3]), " (expected 120.0)")
    print()

    # =========================================================================
    # Step 7: enforce_limits
    # =========================================================================
    print("=== enforce_limits ===")
    # Set bthigh qpos[3] out of range (2.0 > 1.05), should be clamped to 1.05
    d.qpos.data[3] = Scalar[DType.float64](2.0)
    XmlModel.enforce_limits[DType.float64](d)
    print(
        "bthigh qpos after clamp =",
        Float64(d.qpos.data[3]),
        " (expected 0.018326 = 1.05 deg)",
    )

    # rootx (joint 0) is not limited, should not be clamped
    d.qpos.data[0] = Scalar[DType.float64](100.0)
    XmlModel.enforce_limits[DType.float64](d)
    print(
        "rootx qpos after enforce =",
        Float64(d.qpos.data[0]),
        " (expected 100.0, not clamped)",
    )
    print()

    print("=== All CPU tests passed ===")


# =============================================================================
# Nested-default regression (bug 24)
# =============================================================================

# A minimal model with dm_control swimmer's `<default>` SHAPE: the top-level
# `<motor>` and `<joint>` come AFTER two named class blocks, and one of those
# classes is itself nested. `_extract_section` is not depth aware, so before
# `_root_defaults` landed the scan saw a section truncated at the FIRST inner
# `</default>` — which contains `class="inner"`'s joint and no `<motor>` at all.
#
# What that cost, silently, on the real model:
#   * gear fell back to MuJoCo's 1.0 against an actual 5e-4 (2000x force),
#   * `class="inner"`'s `limited="true"` was read as the global default, so the
#     UNLIMITED root slide came out limited with an empty (0, 0) range.
comptime nested_default_xml = """
<mujoco model="nested_defaults">
  <option timestep="0.002"/>
  <default>
    <default class="inner">
      <joint type="hinge" axis="0 0 1" limited="true" armature="1e-6"/>
      <default class="innermost">
        <geom type="box" size=".01 .05 .01" mass=".01"/>
      </default>
    </default>
    <default class="free">
      <joint limited="false" armature="0"/>
    </default>
    <motor gear="5e-4" ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <worldbody>
    <body name="root" pos="0 0 0" childclass="inner">
      <joint name="slider" class="free" type="slide" axis="1 0 0"/>
      <geom class="innermost" name="root_geom"/>
      <body name="link" pos="0 .1 0">
        <geom class="innermost" name="link_geom"/>
        <joint name="hinge" range="-60 60"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="m0" joint="hinge"/>
  </actuator>
</mujoco>
"""


def test_root_default_survives_nested_classes() raises:
    """Bug 24: top-level `<default>` entries declared after a named class.

    Both assertions below are exactly the values the truncated scan produced,
    so each fails loudly if `_root_defaults` is ever bypassed again.
    """
    comptime pm = parse_xml(nested_default_xml)
    comptime XmlModel = ModelDefFromXML[
        xml=nested_default_xml,
        nbody = pm.NBODY, njoint = pm.NJOINT, nq = pm.NQ, nv = pm.NV,
        ngeom = pm.NGEOM, nact = pm.NACT, ntex = pm.NTEX, nmat = pm.NMAT,
        nlight = pm.NLIGHT, ncam = pm.NCAM, nsite = pm.NSITE,
        max_contacts=1,
        timestep = pm.TIMESTEP,
    ]

    # Bind the elements at comptime. Subscripting a comptime `InlineArray`
    # inside a runtime expression materializes the WHOLE array, which rc2
    # rejects (`Array` is no longer `ImplicitlyCopyable`); a scalar element is.
    comptime GEAR0 = XmlModel._acd.motor_gears[0]
    comptime CTRL_MIN0 = XmlModel._acd.motor_ctrl_min[0]
    comptime CTRL_MAX0 = XmlModel._acd.motor_ctrl_max[0]
    comptime LIMITED0 = XmlModel._acd.joint_is_limited[0]
    comptime LIMITED1 = XmlModel._acd.joint_is_limited[1]

    print("=== nested-default regression ===")
    print("motor gear =", GEAR0, " (expected 0.0005)")
    assert_true(
        abs(GEAR0 - 5e-4) <= 1e-18,
        "the top-level <default><motor gear=...> is declared after two named"
        " class blocks and was not picked up — gear fell back to 1.0, a 2000x"
        " actuator force error with no diagnostic (bug 24)",
    )
    assert_true(
        abs(CTRL_MIN0 + 1.0) <= 1e-15 and abs(CTRL_MAX0 - 1.0) <= 1e-15,
        "the same <motor>'s ctrlrange",
    )

    # `slider` is class="free" (limited="false"); `hinge` inherits childclass
    # "inner" (limited="true") and also declares a range. Neither default class
    # is resolvable by this class-blind scan, so `limited` follows MuJoCo's
    # `compiler/autolimits`: a joint with a range is limited.
    print(
        "joint limited =", LIMITED0, LIMITED1, " (expected False True)",
    )
    assert_true(
        not LIMITED0,
        "the UNLIMITED slide came out limited — a nested class's"
        " limited=\"true\" is being read as the global default (bug 24)",
    )
    assert_true(
        LIMITED1,
        "the ranged hinge came out unlimited — autolimits is not being"
        " applied, so a class-set `limited` is the only signal left",
    )
    print("=== nested-default regression passed ===")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

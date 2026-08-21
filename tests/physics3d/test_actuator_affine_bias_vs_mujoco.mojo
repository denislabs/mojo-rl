"""`biasprm[0]` and `biasprm[1]` — the two thirds of the bias we discarded.

    pixi run mojo run -I . tests/physics3d/test_actuator_affine_bias_vs_mujoco.mojo

WHAT WAS MISSING. MuJoCo's scalar actuator force (`mj_fwdActuation`,
`engine_forward.c:508-628`) is

    gain  = gainprm[0]                                  (gaintype FIXED)
    bias  = biasprm[0] + biasprm[1]*length + biasprm[2]*velocity   (AFFINE)
    force = gain * u + bias

The actuator record carried `gainprm[0]` (as `kp`) and `-biasprm[2]` (as
`kv`) and had NO SLOT for the other two, so `apply_actions` reconstructed the
bias from the actuator's KIND instead:

    POSITION   force = kp*(u - length) - kv*vel      i.e. biasprm[1] == -kp
    VELOCITY   force = kp*u            - kv*vel      i.e. biasprm[1] == 0

Both are true of a `<position>` or `<velocity>` ELEMENT by construction —
MuJoCo compiles them to exactly those biasprm — and neither is true of a
`<general>` that writes gain and bias independently.

⚠⚠ FIVE MENAGERIE SCENES DO. franka_emika_panda, robotiq_2f85,
robotiq_2f85_v4, stanford_tidybot and ufactory_xarm7 each drive a gripper
through one actuator whose `ctrlrange` is remapped to travel units. Panda's:

    <general tendon="split" ctrlrange="0 255" forcerange="-100 100"
             gainprm="0.01568627451 0 0" biasprm="0 -100 -10"/>

a gain of 1/64 against a position feedback of **100**. At its keyframe that is
`-100 * 0.04 = -4 N`, which the `split` tendon's two 0.5 coefficients put on
each finger dof as **-2.0**. We answered **-0.00031373** — the same 2.000e+00
error the sweep's `|d qfrc_actuator|` column had been printing for panda while
every other scene in the tree read 0.

⚠ THE PARSER ALREADY KNEW AND ONLY HALF OF IT SAID SO. `_fill_actuators`
records `bad_actuator_code = 3` for "biasprm[1] not in {-gain, 0}"; the
COMPTIME path raises on that field and the RUNTIME path reads it NOWHERE. So
`ModelDefFromXML` would refuse panda outright while `parse_xml_full` loaded it
silently with the wrong law. A diagnostic only one of two parsers consults is
not a diagnostic. Codes 2 and 3 are retired now — both shapes are modelled.

⚠ `biastype` DEFAULTS TO **none**, AND PANDA GETS "affine" FROM ITS CLASS.
Writing `biasprm` on a `<general>` does nothing at all unless `biastype` is
also set; panda's `<default class="panda"><general dyntype="none"
biastype="affine" .../></default>` is what turns it on. The first version of
the fixture below omitted it and MuJoCo answered `gain*ctrl` with no bias —
the `<default>`-chain trap, met from the other side.

⚠ AND THE GROUPING CHANGED ~20 MODELS THAT WERE ALREADY "RIGHT".
`kp*(u - length) - kv*vel` and `kp*u + (b0 + b1*length + b2*vel)` are equal in
exact arithmetic and not in float64. Adopting the reference's association took
kuka_iiwa_14 from 4.547e-13 to **0.000e+00**, boston_dynamics_spot from
1.137e-13 to 0, apptronik_apollo from 2.842e-14 to 0, unitree_g1, aloha,
booster_t1, arx_l5, berkeley_humanoid, both toddlerbots and trossen_vx300s
likewise. Every actuator in `mujoco_menagerie-main` now matches MuJoCo's
`qfrc_actuator` EXACTLY, apart from the two known-missing element types
(`<adhesion>` on flybody, `<plugin>` on shadow_dexee).

⚠ THE LAW NOW HAS ONE COPY. `dynamics.actuation.actuator_scalar_force`; the
GPU kernel inlines the same expression next to a pointer back to it. It had
five spellings before this commit — the CPU pass, the GPU kernel, and three
loops in `pose_transmission` — which is the `ctrllimited` accident (a rule
written inline twice, 322 actuators unclamped) waiting to happen again.
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
    read_model_source,
)
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.fields.dynamics_scratch import DynamicsScratch
from mojo_rl.physics3d.dynamics.actuation import apply_actions_fields
from mojo_rl.physics3d.gpu.constants import (
    KEY_IDX_NQPOS, KEY_META_SIZE, KEY_IDX_NQVEL,
    MODEL_ACTUATOR_SIZE, ACT_IDX_BIAS0, ACT_IDX_BIAS1, ACT_IDX_KP,
)

comptime DT = DType.float64

comptime PANDA = String(
    "references/mujoco_menagerie-main/franka_emika_panda/scene.xml"
)

# ── the fixture: three actuators, three bias shapes, one file ────────────
# `ja` is the shape that used to be unmodelled — gain and `biasprm[1]` are
# INDEPENDENT (1/64 against 100). `jb` and `jc` are the two shapes the old
# law hard-coded, kept as CONTROLS: if either moves, the change broke the
# common case rather than fixing the rare one.
#
# ⚠ THE KEYFRAME IS NOT OPTIONAL. Every bias term multiplies `length` or
# `velocity`, so at `qpos0 = qvel = 0` the whole bias vanishes and all three
# rows would agree for the wrong reason.
comptime XML_AFF = String(
    """<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="a" pos="0 0 1">
      <joint name="ja" type="slide" axis="1 0 0"/>
      <geom type="box" size=".05 .05 .05" mass="1"/>
    </body>
    <body name="b" pos="0 1 1">
      <joint name="jb" type="hinge" axis="0 1 0"/>
      <geom type="box" size=".05 .05 .05" mass="1"/>
    </body>
    <body name="c" pos="0 2 1">
      <joint name="jc" type="hinge" axis="0 1 0"/>
      <geom type="box" size=".05 .05 .05" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <general joint="ja" biastype="affine" gainprm="0.01568627451 0 0"
             biasprm="0 -100 -10" ctrlrange="0 255" forcerange="-1000 1000"/>
    <position joint="jb" kp="50" kv="3" ctrlrange="-3 3"/>
    <velocity joint="jc" kv="7" ctrlrange="-3 3"/>
  </actuator>
  <keyframe>
    <key qpos="0.03 -0.21 0.44" qvel="0.7 -1.3 2.1"/>
  </keyframe>
</mujoco>"""
)


def _aff_ctrl() -> List[Float64]:
    return [120.0, 0.85, -1.4]


def _mj_aff() -> List[Float64]:
    """MuJoCo 3.10.0 `qfrc_actuator` on `XML_AFF` from keyframe 0."""
    return [-8.1176470588, 56.9, -24.5]


# ⚠ WHAT THE OLD LAW ANSWERED for `ja`: `kp*(u - length) - kv*vel`
# = 0.01568627451*(120 - 0.03) - 10*0.7 = -5.1181176470353. The gate below
# asserts we are NOT within 1e-6 of it, so a revert cannot pass by accident.
comptime AFF_OLD_LAW_JA = -5.1181176470353


def _panda_ctrl() -> List[Float64]:
    """Seven arm servos plus the gripper at 200 of its 0..255 range."""
    return [0.20, -0.35, 0.15, -2.0, 0.10, 1.9, 0.40, 200.0]


def _mj_panda() -> List[Float64]:
    """MuJoCo `qfrc_actuator` at panda's keyframe under `_panda_ctrl()`."""
    return [
        87.0, -87.0, 87.0, -87.0, 12.0, 12.0, 12.0,
        -0.431372549, -0.431372549,
    ]


def _qfrc_from_key(
    xml: String, base: String, ctrl: List[Float64]
) raises -> List[Float64]:
    """`d.qfrc` after `apply_actions_fields`, started from keyframe 0."""
    var fmd = parse_xml_full(expand_mjcf(xml, base), base)
    var dims = dims_from_flat(fmd, max_contacts=128, nmesh_verts=65536)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var d = Data[DT, DynDims, 1](dims)
    assert_true(
        dims.get_nkey() > 0,
        "this fixture must carry a keyframe — every bias term multiplies"
        " `length` or `velocity`, so at qpos0 the gate is vacuous",
    )
    for i in range(nq):
        d.qpos.data[i] = sf.qpos0.data[i]
    var nqp = Int(Float64(sf.key_meta.data[KEY_IDX_NQPOS]))
    for i in range(min(nqp, nq)):
        d.qpos.data[i] = sf.key_qpos.data[i]
    var nqv = Int(Float64(sf.key_meta.data[KEY_IDX_NQVEL]))
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](0)
    for i in range(min(nqv, nv)):
        d.qvel.data[i] = sf.key_qvel.data[i]
    var nact = dims.get_nact()
    var act = List[Scalar[DT]](
        length=nact if nact > 0 else 1, fill=Scalar[DT](0)
    )
    for i in range(nv):
        d.qfrc.data[i] = Scalar[DT](0)
    apply_actions_fields[DT](sf, d, ctrl, act, fmd.timestep)
    var out = List[Float64]()
    for i in range(nv):
        out.append(Float64(d.qfrc.data[i]))
    return out^


def _worst(got: List[Float64], want: List[Float64]) -> Float64:
    var w = 0.0
    for i in range(len(want)):
        var e = abs(got[i] - want[i])
        if e > w:
            w = e
    return w


def test_general_bias_is_independent_of_the_gain() raises:
    """gain 1/64 against a position feedback of 100, and its two controls."""
    print("=== <general biastype=affine> with biasprm[1] != -gainprm[0] ===")
    var want = _mj_aff()
    var got = _qfrc_from_key(XML_AFF, String(""), _aff_ctrl())
    assert_true(
        len(got) == 3,
        "the fixture must build three dofs; got " + String(len(got)),
    )
    for i in range(3):
        print("  dof", i, " ours", got[i], " mj", want[i])
    var worst = _worst(got, want)
    print("  worst |d(qfrc)| =", worst)

    # ⚠ THE TWO CONTROLS FIRST — a `<position>` and a `<velocity>`, the
    # shapes the old law got right. If either has moved, the general form
    # broke the common case rather than fixing the rare one.
    assert_true(
        abs(got[1] - want[1]) < 1e-12 and abs(got[2] - want[2]) < 1e-12,
        "a CONTROL actuator moved: <position> reads " + String(got[1])
        + " (want " + String(want[1]) + ") and <velocity> reads "
        + String(got[2]) + " (want " + String(want[2]) + "). The affine law"
        " must reproduce both element shapes exactly.",
    )
    assert_true(
        abs(got[0] - want[0]) < 1e-12,
        "the <general> actuator reads " + String(got[0]) + " against"
        " MuJoCo's " + String(want[0]) + ".",
    )
    # ⚠ AND IT MUST NOT BE THE OLD ANSWER. Asserting closeness to MuJoCo
    # alone would still pass if someone widened the tolerance; this names
    # the wrong number explicitly.
    assert_true(
        abs(got[0] - AFF_OLD_LAW_JA) > 1e-6,
        "the <general> actuator answered " + String(got[0]) + ", which is"
        " the OLD law `kp*(u - length) - kv*vel` = "
        + String(AFF_OLD_LAW_JA) + ". `biasprm[1]` is being taken as `-kp`"
        " again.",
    )
    print("  PASS")


def test_bias_slots_reach_the_record() raises:
    """`ACT_IDX_BIAS0/1` carry MuJoCo's values, not a reconstruction."""
    print("=== biasprm[0] / biasprm[1] in the actuator record ===")
    var fmd = parse_xml_full(expand_mjcf(XML_AFF, String("")), String(""))
    var dims = dims_from_flat(fmd, max_contacts=8, nmesh_verts=0)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    assert_true(dims.get_nact() == 3, "three actuators expected")
    var want_b1: List[Float64] = [-100.0, -50.0, 0.0]
    for i in range(3):
        var o = i * MODEL_ACTUATOR_SIZE
        var b0 = Float64(sf.actuators.data[o + ACT_IDX_BIAS0])
        var b1 = Float64(sf.actuators.data[o + ACT_IDX_BIAS1])
        var kp = Float64(sf.actuators.data[o + ACT_IDX_KP])
        print("  actuator", i, " gain", kp, " biasprm[0]", b0,
              " biasprm[1]", b1, " (want", want_b1[i], ")")
        assert_true(
            b0 == 0.0,
            "biasprm[0] on actuator " + String(i) + " is " + String(b0)
            + "; every actuator in this tree has 0 there.",
        )
        assert_true(
            abs(b1 - want_b1[i]) < 1e-12,
            "biasprm[1] on actuator " + String(i) + " is " + String(b1)
            + " against MuJoCo's " + String(want_b1[i]) + ". A value equal"
            " to -gain on actuator 0 means the slot is being DERIVED from"
            " the gain, which is the whole defect.",
        )
    print("  PASS")


def test_panda_gripper_pulls_two_newton_metres() raises:
    """The real model, where the tendon's two 0.5 coefficients split -4 N."""
    print("=== franka_emika_panda, qfrc_actuator at its keyframe ===")
    var src = read_model_source(PANDA)
    var want = _mj_panda()
    var got = _qfrc_from_key(src[0], src[1], _panda_ctrl())
    assert_true(
        len(got) == 9,
        "panda has 9 dofs; got " + String(len(got)),
    )
    for i in range(9):
        print("  dof", i, " ours", got[i], " mj", want[i])
    var worst = _worst(got, want)
    print("  worst |d(qfrc_actuator)| =", worst)
    # ⚠ VACUITY. The seven arm servos SATURATE their forcerange under this
    # control, so they agree whatever the bias law is. The two that
    # discriminate are the finger dofs — assert on them by name.
    assert_true(
        abs(got[7] - want[7]) < 1e-12 and abs(got[8] - want[8]) < 1e-12,
        "the gripper dofs read (" + String(got[7]) + ", " + String(got[8])
        + ") against MuJoCo's " + String(want[7]) + " each. A value near"
        " -1.2e-05 is `gainprm[0]` standing in for `biasprm[1]`; at the"
        " sweep's own control that error was exactly 2.000e+00.",
    )
    assert_true(
        worst < 1e-12,
        "panda is " + String(worst) + " from MuJoCo's qfrc_actuator.",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

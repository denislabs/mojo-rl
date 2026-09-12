"""Five reader defects, each against a live 3.12 `MjModel` of the SAME xml.

    pixi run mojo run -I . tests/physics3d/test_parser_scalar_and_default_chain_vs_mujoco.mojo

None of these five changes what a legal, fully-spelled model does. Every one
of them changes what a PARTIALLY spelled one does, which is why they survived:
the gates in this tree all state every component of every vector, so they
agreed with the defect by construction.

AUD-15 — `_parse_float` skipped every character it did not recognise, so a
two-token scalar was read as ONE number with the space deleted:
`stiffness="7 9"` became 79. Not a rounding — an order of magnitude, times
ten, with no message. ⚠ AND THE PREMISE THE AUDIT WROTE IT ON WAS WRONG:
`stiffness="7 9"` is not a legacy 3.7 spelling MuJoCo refuses, it is a 3.12
POLYNOMIAL spring. `mjNPOLY = 2` (mjmodel.h:44); the MJCF tables give
`stiffness`/`damping` `mjNPOLY+1` slots on joints and on both tendon kinds
(mjcf_read_table.inc:336,344,535,536,557,558); `mj_springdamper` evaluates
`force = -x * mju_polyForce(linear, poly, x, mjNPOLY, odd)`
(engine_passive.c:654-740). So 3.12 ACCEPTS the model and runs a nonlinear
spring whose linear coefficient is 7 — which is what `m.jnt_stiffness` reports
below. Taking the first token is therefore not a repair of a malformed
attribute, it is MuJoCo's own linear term; the poly terms we still do not have
are counted and printed at load as AUD-54.

AUD-16 — a one-value `<equality><joint polycoef="0.5"/>` zeroed `polycoef[1]`.
MuJoCo copies the tokens present and leaves the rest at the element defaults,
which `mjs_defaultEquality` set to `[0, 1, 0, 0, 0]` (user_init.c). A stated
prefix is an OFFSET on a joint that still tracks its partner; zeroing slot 1
decoupled them and pinned q1 to a constant instead — a constraint that looks
active and is a different one.

AUD-17 — a one-value tendon `solreflimit` was dropped entirely by a
`len >= 2` gate. `_solref_into` — this parser's own rule, measured on the
runtime, already used by the joint, geom, pair and equality readers — keeps
the component the attribute supplies and leaves the other alone. This site was
the one that had the rule written out by hand, and it had drifted.

AUD-07 — `<equality><tendon>` consulted the element and its named class and
stopped. The connect/weld/joint branch of the same function consults the root
`<default><equality>` as well. A model that states its equality solparams once
at the root — the cassie and apollo spelling — got the built-in defaults for
its tendon equalities and the stated ones for everything else, in one model.

AUD-06 — `fields_build` wrote the literals 0.5 and 2.0 into `eq_solimp[3]`
and `eq_solimp[4]` although the parser had read both from the element, the
class and the root default. Those two are the width and the power of the
impedance ramp.

⚠ THE ORACLE IS THE SAME XML STRING ON BOTH SIDES. Every fixture below is
handed verbatim to `mujoco.MjModel.from_xml_string` and to `parse_xml_full`,
so there is no transcribed constant to go stale when the runtime moves.
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.fields import Model, DynDims
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat,
    build_model_runtime,
)
from mojo_rl.physics3d.gpu.constants import (
    MODEL_EQ_SIZE,
    EQ_IDX_SOLIMP_3,
    EQ_IDX_SOLIMP_4,
)

comptime DT = DType.float64
comptime TOL = 1e-15


def _close(
    a: Float64, b: Float64, label: String, mut n: Int
) raises:
    """Compare one value and COUNT it.

    ⚠ NON-VACUITY IS THE DEFAULT FAILURE HERE. Five tests that each compared
    zero numbers would print five OKs, so every test below carries its own
    counter and asserts a floor on it before returning.
    """
    n += 1
    assert_true(
        abs(a - b) <= TOL,
        label + ": ours " + String(a) + ", MuJoCo " + String(b)
        + " (|d| = " + String(abs(a - b)) + ")",
    )


# ── AUD-15 ─────────────────────────────────────────────────────────────────
# A hinge whose spring and damper are stated as POLYNOMIALS. MuJoCo's linear
# coefficients are 7 and 3; the concatenation the old reader produced was 79
# and 34.
comptime POLY_XML = String(
    """<mujoco model="poly_scalar">
  <compiler angle="radian"/>
  <worldbody>
    <body name="b" pos="0 0 1">
      <joint name="j" type="hinge" axis="0 1 0" stiffness="7 9" damping="3 4"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" density="1000"/>
    </body>
  </worldbody>
</mujoco>
"""
)

# ── AUD-16 ─────────────────────────────────────────────────────────────────
comptime POLYCOEF_XML = String(
    """<mujoco model="polycoef_prefix">
  <compiler angle="radian"/>
  <worldbody>
    <body name="a" pos="0 0 1">
      <joint name="ja" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" density="1000"/>
    </body>
    <body name="b" pos="0 0 2">
      <joint name="jb" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" density="1000"/>
    </body>
  </worldbody>
  <equality>
    <joint joint1="ja" joint2="jb" polycoef="0.5"/>
  </equality>
</mujoco>
"""
)

# ── AUD-17 ─────────────────────────────────────────────────────────────────
comptime TENDON_SOLREF_XML = String(
    """<mujoco model="tendon_one_value_solref">
  <compiler angle="radian"/>
  <worldbody>
    <body name="a" pos="0 0 1">
      <joint name="ja" type="slide" axis="0 0 1"/>
      <geom type="sphere" size="0.05" density="1000"/>
    </body>
  </worldbody>
  <tendon>
    <fixed name="t" limited="true" range="-1 1" solreflimit="0.005">
      <joint joint="ja" coef="1"/>
    </fixed>
  </tendon>
</mujoco>
"""
)

# ── AUD-07 ─────────────────────────────────────────────────────────────────
# The solparams are stated ONCE, on the root class, and the only consumer is
# a tendon equality.
comptime TENDON_EQ_ROOT_XML = String(
    """<mujoco model="tendon_eq_root_default">
  <compiler angle="radian"/>
  <default>
    <equality solref="0.005 1" solimp="0.95 0.99 0.002 0.6 3"/>
  </default>
  <worldbody>
    <body name="a" pos="0 0 1">
      <joint name="ja" type="slide" axis="0 0 1"/>
      <geom type="sphere" size="0.05" density="1000"/>
    </body>
  </worldbody>
  <tendon>
    <fixed name="t"><joint joint="ja" coef="1"/></fixed>
  </tendon>
  <equality>
    <tendon tendon1="t"/>
  </equality>
</mujoco>
"""
)

# ── AUD-06 ─────────────────────────────────────────────────────────────────
# solimp[3] = 0.8 and solimp[4] = 3 are both away from the 0.5 / 2 that
# `fields_build` used to write, so the test cannot pass by agreeing with the
# hardcode.
comptime EQ_SOLIMP_XML = String(
    """<mujoco model="eq_solimp_width_power">
  <compiler angle="radian"/>
  <worldbody>
    <body name="a" pos="0 0 1"><freejoint/><geom size="0.05" density="1000"/></body>
    <body name="b" pos="0 0 2"><freejoint/><geom size="0.05" density="1000"/></body>
  </worldbody>
  <equality>
    <connect body1="a" body2="b" anchor="0 0 0"
             solimp="0.9 0.95 0.001 0.8 3"/>
  </equality>
</mujoco>
"""
)


def _mj(xml: String) raises -> PythonObject:
    var mujoco = Python.import_module("mujoco")
    return mujoco.MjModel.from_xml_string(xml)


def test_a_polynomial_scalar_reads_its_linear_coefficient() raises:
    """AUD-15. Two tokens are two coefficients, never one number."""
    print("=== AUD-15: <joint stiffness='7 9' damping='3 4'> ===")
    var n = 0
    var mm = _mj(POLY_XML)
    var mj_k = Float64(py=mm.jnt_stiffness[0])
    var mj_b = Float64(py=mm.dof_damping[0])
    # ⚠ THE PREMISE, CHECKED RATHER THAN ASSUMED: 3.12 must ACCEPT this model
    # (it did not raise above) and must report the FIRST token, not the last
    # and not the concatenation. If a later release changes either, this
    # line is where it surfaces.
    print("  MuJoCo jnt_stiffness[0] =", mj_k, " dof_damping[0] =", mj_b)
    assert_true(
        abs(mj_k - 7.0) <= TOL and abs(mj_b - 3.0) <= TOL,
        "3.12 no longer reads the linear coefficient of a polynomial"
        " stiffness/damping as the first token: got " + String(mj_k) + " / "
        + String(mj_b) + ". The rule this file pins has moved.",
    )

    var fmd = parse_xml_full(POLY_XML, String("."))
    assert_true(len(fmd.joints) == 1, "expected one joint")
    var j = fmd.joints[0]
    print("  ours   stiffness    =", j.stiffness, " damping      =", j.damping)
    _close(j.stiffness, mj_k, "joint stiffness", n)
    _close(j.damping, mj_b, "joint damping", n)

    # The negative control is the value the defect produced, named so the
    # assertion says what it is rather than "not 79".
    assert_true(
        abs(j.stiffness - 79.0) > 1.0 and abs(j.damping - 34.0) > 1.0,
        "the two tokens were concatenated again (79 / 34): `_parse_float` is"
        " reading past the end of the first token",
    )

    # AND THE POLY TERM IS COUNTED, not silently dropped. One row, because
    # the two attributes are summed into a single AUD-15 count.
    print("  silent_attrs after the scan =", fmd.silent_attrs)
    var saw = False
    for i in range(len(fmd.silent_attr_ids)):
        if fmd.silent_attr_ids[i] == String("AUD-15"):
            saw = True
    assert_true(
        saw,
        "the polynomial spelling was read and NOT counted — the higher"
        " coefficients are gone and nothing said so",
    )
    print("  values compared against MuJoCo:", n)
    assert_true(
        n == 2,
        "expected 2 comparisons in this test, made " + String(n)
        + " — the test body changed and its non-vacuity floor did not",
    )


def test_a_one_value_polycoef_keeps_the_unit_slope() raises:
    """AUD-16. A stated prefix leaves the rest at the element defaults."""
    print("=== AUD-16: <joint polycoef='0.5'> ===")
    var n = 0
    var mm = _mj(POLYCOEF_XML)
    var fmd = parse_xml_full(POLYCOEF_XML, String("."))
    assert_true(len(fmd.equalities) == 1, "expected one equality")
    var e = fmd.equalities[0]
    var ours: List[Float64] = [
        e.anchor_a_x, e.anchor_a_y, e.anchor_a_z, e.anchor_b_x, e.anchor_b_y,
    ]
    for i in range(5):
        var refv = Float64(py=mm.eq_data[0][i])
        print("  polycoef[", i, "]  ours", ours[i], " MuJoCo", refv)
        _close(ours[i], refv, String("polycoef[") + String(i) + "]", n)
    # ⚠ THE FAILURE MODE NAMED: slot 1 is the SLOPE. Zero there is not a
    # smaller constraint, it is a different one — q1 pinned to 0.5 instead of
    # tracking q2 with a 0.5 offset.
    assert_true(
        abs(e.anchor_a_y - 1.0) <= TOL,
        "polycoef[1] is " + String(e.anchor_a_y) + ", not 1: the two joints"
        " are DECOUPLED and q1 is pinned to a constant",
    )
    print("  values compared against MuJoCo:", n)
    assert_true(
        n == 5,
        "expected 5 comparisons in this test, made " + String(n)
        + " — the test body changed and its non-vacuity floor did not",
    )


def test_an_explicit_zero_slope_still_reaches_the_record() raises:
    """The other half of AUD-16: the decoupled constraint stays reachable.

    ⚠ A FIX THAT MADE `polycoef[1]` UNCONDITIONALLY 1 WOULD PASS THE TEST
    ABOVE. Stating the zero explicitly must still produce a zero, or the
    reader has traded one wrong answer for another.
    """
    print("=== AUD-16 control: polycoef='0.5 0' is still (0.5, 0) ===")
    var n = 0
    var xml = String(POLYCOEF_XML).replace(
        String("polycoef=\"0.5\""), String("polycoef=\"0.5 0\"")
    )
    assert_true(
        xml != POLYCOEF_XML, "the fixture edit did not apply; test is vacuous"
    )
    var mm = _mj(xml)
    var fmd = parse_xml_full(xml, String("."))
    var e = fmd.equalities[0]
    print("  ours (", e.anchor_a_x, ",", e.anchor_a_y, ")  MuJoCo (",
          Float64(py=mm.eq_data[0][0]), ",", Float64(py=mm.eq_data[0][1]), ")")
    _close(e.anchor_a_x, Float64(py=mm.eq_data[0][0]), "polycoef[0] explicit", n)
    _close(e.anchor_a_y, Float64(py=mm.eq_data[0][1]), "polycoef[1] explicit", n)
    assert_true(
        abs(e.anchor_a_y) <= TOL,
        "an explicitly stated polycoef[1]=0 came back as "
        + String(e.anchor_a_y),
    )
    print("  values compared against MuJoCo:", n)
    assert_true(
        n == 2,
        "expected 2 comparisons in this test, made " + String(n)
        + " — the test body changed and its non-vacuity floor did not",
    )


def test_a_one_value_tendon_solreflimit_keeps_the_dampratio() raises:
    """AUD-17."""
    print("=== AUD-17: <fixed solreflimit='0.005'> ===")
    var n = 0
    var mm = _mj(TENDON_SOLREF_XML)
    var fmd = parse_xml_full(TENDON_SOLREF_XML, String("."))
    assert_true(len(fmd.tendons) == 1, "expected one tendon")
    var t = fmd.tendons[0]
    var r0 = Float64(py=mm.tendon_solref_lim[0][0])
    var r1 = Float64(py=mm.tendon_solref_lim[0][1])
    print("  ours (", t.solref_lim_0, ",", t.solref_lim_1,
          ")  MuJoCo (", r0, ",", r1, ")")
    _close(t.solref_lim_0, r0, "tendon solreflimit[0]", n)
    _close(t.solref_lim_1, r1, "tendon solreflimit[1]", n)
    assert_true(
        abs(t.solref_lim_0 - 0.02) > 1e-6,
        "the tendon kept the 0.02 default: the one-value attribute was"
        " dropped whole, and the limit runs a time constant four times too"
        " long",
    )
    print("  values compared against MuJoCo:", n)
    assert_true(
        n == 2,
        "expected 2 comparisons in this test, made " + String(n)
        + " — the test body changed and its non-vacuity floor did not",
    )


def test_the_tendon_equality_reads_the_root_default() raises:
    """AUD-07."""
    print("=== AUD-07: <default><equality> reaches <equality><tendon> ===")
    var n = 0
    var mm = _mj(TENDON_EQ_ROOT_XML)
    var fmd = parse_xml_full(TENDON_EQ_ROOT_XML, String("."))
    assert_true(len(fmd.tendons) == 1, "expected one tendon")
    var t = fmd.tendons[0]
    assert_true(
        t.is_equality == 1,
        "the <equality><tendon> row did not reach the tendon record at all",
    )
    _close(t.solref_eq_0, Float64(py=mm.eq_solref[0][0]), "eq solref[0]", n)
    _close(t.solref_eq_1, Float64(py=mm.eq_solref[0][1]), "eq solref[1]", n)
    var ours: List[Float64] = [
        t.solimp_eq_0, t.solimp_eq_1, t.solimp_eq_2, t.solimp_eq_3,
        t.solimp_eq_4,
    ]
    for i in range(5):
        var refv = Float64(py=mm.eq_solimp[0][i])
        print("  solimp[", i, "]  ours", ours[i], " MuJoCo", refv)
        _close(ours[i], refv, String("eq solimp[") + String(i) + "]", n)
    print("  solref  ours (", t.solref_eq_0, ",", t.solref_eq_1, ")")
    assert_true(
        abs(t.solref_eq_0 - 0.02) > 1e-6,
        "the tendon equality kept MuJoCo's built-in 0.02: the root"
        " <default><equality> was not consulted",
    )
    print("  values compared against MuJoCo:", n)
    assert_true(
        n == 7,
        "expected 7 comparisons in this test, made " + String(n)
        + " — the test body changed and its non-vacuity floor did not",
    )


def test_the_equality_solimp_width_and_power_survive_the_build() raises:
    """AUD-06, read off the BUILT model rather than the parse.

    ⚠ THE PARSE WAS ALWAYS RIGHT HERE. `_fill_equality_solparams` read all
    five components; `fields_build` then wrote literals into two of them. So
    a test that stopped at `FlatModelDef` would have passed before the fix.
    """
    print("=== AUD-06: eq_solimp[3], eq_solimp[4] reach the model ===")
    var n = 0
    var mm = _mj(EQ_SOLIMP_XML)
    var fmd = parse_xml_full(EQ_SOLIMP_XML, String("."))
    var dims = dims_from_flat(fmd, max_contacts=8, nmesh_verts=64)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)

    var o = 0 * MODEL_EQ_SIZE
    var w = Float64(m.equality.data[o + EQ_IDX_SOLIMP_3])
    var p = Float64(m.equality.data[o + EQ_IDX_SOLIMP_4])
    var mj_w = Float64(py=mm.eq_solimp[0][3])
    var mj_p = Float64(py=mm.eq_solimp[0][4])
    print("  width  ours", w, " MuJoCo", mj_w)
    print("  power  ours", p, " MuJoCo", mj_p)
    _close(w, mj_w, "eq_solimp[3] (width)", n)
    _close(p, mj_p, "eq_solimp[4] (power)", n)
    assert_true(
        abs(w - 0.5) > 1e-9 and abs(p - 2.0) > 1e-9,
        "the built model still carries the 0.5 / 2.0 literals, so the"
        " parsed width and power never left FlatModelDef",
    )
    print("  values compared against MuJoCo:", n)
    assert_true(
        n == 2,
        "expected 2 comparisons in this test, made " + String(n)
        + " — the test body changed and its non-vacuity floor did not",
    )


def main() raises:
    var suite = TestSuite()
    suite.test[test_a_polynomial_scalar_reads_its_linear_coefficient]()
    suite.test[test_a_one_value_polycoef_keeps_the_unit_slope]()
    suite.test[test_an_explicit_zero_slope_still_reaches_the_record]()
    suite.test[test_a_one_value_tendon_solreflimit_keeps_the_dampratio]()
    suite.test[test_the_tendon_equality_reads_the_root_default]()
    suite.test[test_the_equality_solimp_width_and_power_survive_the_build]()
    suite^.run()

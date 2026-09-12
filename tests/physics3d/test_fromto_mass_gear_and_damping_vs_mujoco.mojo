"""Five more reader/builder defects, against a live 3.12 `MjModel`.

    pixi run mojo run -I . tests/physics3d/test_fromto_mass_gear_and_damping_vs_mujoco.mojo

AUD-10 — `fromto` on a BOX or an ELLIPSOID reshapes the solid, it does not
only pose it. `mjCGeom::Compile` (user_objects.cc:4004-4013) puts half the
segment length in `size[1]`, then for those two types shifts it up:
`size[2] = size[1]; size[1] = size[0]`. A fromto box states ONE size token, so
reading half_y and half_z off `size` alone left both at ZERO — a flat,
zero-thickness box that collides with nothing and has no inertia. The site
reader in the same file already had the rule; the geom reader did not.

AUD-11 — `<compiler settotalmass>` was applied only when `inertiafromgeom` was
active, on a comment claiming that was MuJoCo's legacy ordering. There is no
such ordering: `mjCModel::Compile` (user_model.cc:5455) runs
`if (compiler.settotalmass > 0) mj_setTotalmass(...)` last and unconditionally.
A model that writes its inertias by hand AND asks for a total mass kept the
hand-written total.

AUD-18 — a 6-vector `gear` stated in a `<default>` class collapsed to its
first token, because `DefaultsData` carried `motor_gear` as one Float64. For a
`site=` transmission the six are a wrench in the site frame, and both
Menagerie quadrotors state `gear="0 0 1 0 0 -.0201"` exactly this way: the
first token is 0, i.e. an actuator that applies no force at all.

AUD-20 — a `<general class="c">` whose class `c` was written with
`<position kp kv>` became a gain-1 torque motor. `mjs_setToPosition`
(user_api.cc:1273) writes `gaintype = FIXED`, `biastype = AFFINE`,
`gainprm[0] = kp` and `biasprm[1] = -kp` UNCONDITIONALLY; we recorded only
gainprm[0] and biasprm[2], so the element inherited an EMPTY biastype and the
whole servo disappeared.

AUD-42 — joint damping was applied under `damp > 0`, so a NEGATIVE damping was
silently dropped. ⚠ MuJoCo USES TWO DIFFERENT TESTS AND THE FIX HAS TO CARRY
BOTH: `mj_EulerSkip` scans for any dof with `dof_damping[i] > 0` — strictly
positive — and integrates EXPLICITLY if it finds none; having found one, the
`qH += h*diag(B)` loop that follows runs over every dof with no test at all.
So a model with one positive and one negative damping gets the negative one in
the implicit matrix, and a model with ONLY negative damping gets no implicit
matrix at all. Three legs below, one per branch.

⚠ THE ORACLE IS THE SAME XML STRING ON BOTH SIDES throughout.
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.fields import Model, Data, DynDims
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat,
    build_model_runtime,
    spec_fields_runtime,
)
from mojo_rl.physics3d.gpu.constants import MODEL_BODY_SIZE, BODY_IDX_MASS
from mojo_rl.physics3d.studio.stepping import StudioIntegPyr

comptime DT = DType.float64
comptime TOL = 1e-14


def _close(
    a: Float64, b: Float64, label: String, tol: Float64, mut n: Int
) raises:
    """Compare one value and COUNT it — every test below prints its count."""
    n += 1
    assert_true(
        abs(a - b) <= tol,
        label + ": ours " + String(a) + ", MuJoCo " + String(b)
        + " (|d| = " + String(abs(a - b)) + ", tol " + String(tol) + ")",
    )


def _mj(xml: String) raises -> PythonObject:
    var mujoco = Python.import_module("mujoco")
    return mujoco.MjModel.from_xml_string(xml)


# ── AUD-10 ─────────────────────────────────────────────────────────────────
# A 1 m box of stated half-width 0.1 and a 0.6 m ellipsoid of stated
# semi-axis 0.07. MuJoCo: size (0.1, 0.1, 0.5) and (0.07, 0.07, 0.3).
comptime FROMTO_XML = String(
    """<mujoco model="fromto_shapes">
  <compiler angle="radian"/>
  <worldbody>
    <body name="a" pos="0 0 3"><freejoint/>
      <geom name="bx" type="box" size="0.1" fromto="0 0 0 0 0 1" density="500"/>
    </body>
    <body name="b" pos="1 0 3"><freejoint/>
      <geom name="el" type="ellipsoid" size="0.07" fromto="0 0 0 0 0 0.6" density="500"/>
    </body>
  </worldbody>
</mujoco>
"""
)

# ── AUD-11 ─────────────────────────────────────────────────────────────────
# ⚠ `inertiafromgeom="false"` IS THE POINT. Both bodies state an `<inertial>`,
# so the geoms contribute nothing and the masses are 3 and 5 — which
# settotalmass must rescale to 5.25 and 8.75.
comptime TOTALMASS_XML = String(
    """<mujoco model="settotalmass_explicit_inertia">
  <compiler angle="radian" inertiafromgeom="false" settotalmass="14"/>
  <worldbody>
    <body name="a" pos="0 0 1"><freejoint/>
      <inertial pos="0 0 0" mass="3" diaginertia="0.1 0.2 0.3"/>
      <geom type="box" size="0.1 0.1 0.1" density="1000"/>
    </body>
    <body name="b" pos="0 0 2"><freejoint/>
      <inertial pos="0 0 0" mass="5" diaginertia="0.4 0.5 0.6"/>
      <geom type="box" size="0.1 0.1 0.1" density="1000"/>
    </body>
  </worldbody>
</mujoco>
"""
)

# ── AUD-18 ─────────────────────────────────────────────────────────────────
comptime CLASS_GEAR_XML = String(
    """<mujoco model="class_gear">
  <compiler angle="radian"/>
  <default>
    <default class="prop">
      <motor gear="0 0 1 0 0 -0.0201" ctrlrange="0 10"/>
    </default>
  </default>
  <worldbody>
    <body name="a" pos="0 0 1"><freejoint/>
      <geom type="box" size="0.1 0.1 0.02" density="500"/>
      <site name="s" pos="0 0 0"/>
    </body>
  </worldbody>
  <actuator>
    <motor class="prop" name="p" site="s"/>
  </actuator>
</mujoco>
"""
)

# ── AUD-20 ─────────────────────────────────────────────────────────────────
comptime CLASS_SERVO_XML = String(
    """<mujoco model="class_servo">
  <compiler angle="radian"/>
  <default>
    <default class="srv"><position kp="100" kv="9"/></default>
    <default class="vel"><velocity kv="7"/></default>
  </default>
  <worldbody>
    <body name="b" pos="0 0 1">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" density="1000"/>
    </body>
    <body name="c" pos="0 0 2">
      <joint name="k" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" density="1000"/>
    </body>
  </worldbody>
  <actuator>
    <general name="a" class="srv" joint="j"/>
    <general name="v" class="vel" joint="k"/>
  </actuator>
</mujoco>
"""
)

# ── AUD-42 ─────────────────────────────────────────────────────────────────
# `%s` is filled per leg: mixed, negative-only, positive-only.
comptime DAMP_XML_FMT = String(
    """<mujoco model="neg_damp">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="Euler"/>
  <worldbody>
    <body name="b" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0" damping="D1"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.04" density="1000"/>
      <body name="c" pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" damping="D2"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.04" density="1000"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
)
comptime DAMP_STEPS = 20
# Set from the measurement: the three legs agree with MuJoCo to ~1e-14 on
# qvel after 20 steps. A bound at 1e-12 is two orders of slack on that and
# still four orders below the 1e-8 gap between the legs themselves.
comptime DAMP_TOL = 1e-12


def _damp_xml(d1: String, d2: String) -> String:
    return DAMP_XML_FMT.replace(String("D1"), d1).replace(String("D2"), d2)


def test_fromto_reshapes_a_box_and_an_ellipsoid() raises:
    """AUD-10."""
    print("=== AUD-10: fromto box / ellipsoid half-extents ===")
    var n = 0
    var mm = _mj(FROMTO_XML)
    var fmd = parse_xml_full(FROMTO_XML, String("."))
    assert_true(len(fmd.geoms) == 2, "expected two geoms")
    for g in range(2):
        var gd = fmd.geoms[g]
        var ours: List[Float64] = [gd.half_x, gd.half_y, gd.half_z]
        for k in range(3):
            var refv = Float64(py=mm.geom_size[g][k])
            print("  geom", g, " size[", k, "]  ours", ours[k],
                  " MuJoCo", refv)
            _close(ours[k], refv,
                   String("geom ") + String(g) + " size[" + String(k) + "]",
                   TOL, n)
        # The pose is the half that already worked; comparing it keeps a
        # reshape that also moved the solid from passing.
        var op: List[Float64] = [gd.pos_x, gd.pos_y, gd.pos_z]
        for k in range(3):
            _close(op[k], Float64(py=mm.geom_pos[g][k]),
                   String("geom ") + String(g) + " pos[" + String(k) + "]",
                   TOL, n)
    # ⚠ THE CONSEQUENCE, NOT JUST THE FIELD. A zero half-extent is a solid of
    # zero volume, so the defect showed up as a MASSLESS body — which is what
    # actually breaks a rollout.
    var dims = dims_from_flat(fmd, max_contacts=8, nmesh_verts=64)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    for b in range(1, 3):
        var mass = Float64(m.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_MASS])
        var refm = Float64(py=mm.body_mass[b])
        print("  body", b, " mass  ours", mass, " MuJoCo", refm)
        _close(mass, refm, String("body ") + String(b) + " mass", 1e-12, n)
        assert_true(
            mass > 1e-6,
            "body " + String(b) + " came out massless: its fromto solid still"
            " has a zero half-extent",
        )
    print("  values compared against MuJoCo:", n)
    assert_true(n == 14, "expected 14 comparisons, made " + String(n))


def test_settotalmass_applies_without_inertiafromgeom() raises:
    """AUD-11."""
    print("=== AUD-11: settotalmass with inertiafromgeom='false' ===")
    var n = 0
    var mm = _mj(TOTALMASS_XML)
    var fmd = parse_xml_full(TOTALMASS_XML, String("."))
    var dims = dims_from_flat(fmd, max_contacts=8, nmesh_verts=64)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var total = Float64(0)
    for b in range(1, 3):
        var mass = Float64(m.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_MASS])
        var refm = Float64(py=mm.body_mass[b])
        print("  body", b, " mass  ours", mass, " MuJoCo", refm)
        _close(mass, refm, String("body ") + String(b) + " mass", 1e-13, n)
        total += mass
    print("  total  ours", total, " (settotalmass asked for 14)")
    assert_true(
        abs(total - 14.0) < 1e-12,
        "the masses sum to " + String(total) + ", not 14: settotalmass did"
        " not run. The unscaled total is 8, which is the value the defect"
        " produced.",
    )
    assert_true(
        abs(total - 8.0) > 1.0,
        "the masses still sum to the hand-written 8",
    )
    print("  values compared against MuJoCo:", n)
    assert_true(n == 2, "expected 2 comparisons, made " + String(n))


def test_a_class_level_gear_keeps_all_six_components() raises:
    """AUD-18."""
    print("=== AUD-18: <default><motor gear='0 0 1 0 0 -0.0201'> ===")
    var n = 0
    var mm = _mj(CLASS_GEAR_XML)
    var fmd = parse_xml_full(CLASS_GEAR_XML, String("."))
    assert_true(len(fmd.actuators) == 1, "expected one actuator")
    var a = fmd.actuators[0]
    var ours: List[Float64] = [
        a.gear, a.gear1, a.gear2, a.gear3, a.gear4, a.gear5,
    ]
    for k in range(6):
        var refv = Float64(py=mm.actuator_gear[0][k])
        print("  gear[", k, "]  ours", ours[k], " MuJoCo", refv)
        _close(ours[k], refv, String("gear[") + String(k) + "]", TOL, n)
    # ⚠ THE NEGATIVE CONTROL IS THE WHOLE POINT HERE. The defect gave
    # gear = (0, 0, 0, 0, 0, 0) — the first token and five zeros — which is a
    # dead actuator, and the first token AGREES with MuJoCo. Only the tail
    # separates the two.
    assert_true(
        abs(a.gear2 - 1.0) < TOL and abs(a.gear5 + 0.0201) < TOL,
        "gear[2] and gear[5] are still zero: the class's six-vector was read"
        " as one token and this actuator applies no force",
    )
    print("  values compared against MuJoCo:", n)
    assert_true(n == 6, "expected 6 comparisons, made " + String(n))


def test_a_general_element_inherits_a_servo_class() raises:
    """AUD-20."""
    print("=== AUD-20: <general class='srv'> where srv is <position> ===")
    var n = 0
    var mm = _mj(CLASS_SERVO_XML)
    var fmd = parse_xml_full(CLASS_SERVO_XML, String("."))
    assert_true(len(fmd.actuators) == 2, "expected two actuators")
    for i in range(2):
        var a = fmd.actuators[i]
        # `kp` IS gainprm[0], `kv` IS -biasprm[2], `bias1` IS biasprm[1].
        # Comparing the three separately is what tells a position servo from
        # a velocity one: they differ ONLY in biasprm[1].
        print("  act", i, " kp", a.kp, " kv", a.kv, " bias0", a.bias0,
              " bias1", a.bias1)
        _close(a.kp, Float64(py=mm.actuator_gainprm[i][0]),
               String("act ") + String(i) + " gainprm[0]", TOL, n)
        _close(a.bias0, Float64(py=mm.actuator_biasprm[i][0]),
               String("act ") + String(i) + " biasprm[0]", TOL, n)
        _close(a.bias1, Float64(py=mm.actuator_biasprm[i][1]),
               String("act ") + String(i) + " biasprm[1]", TOL, n)
        _close(-a.kv, Float64(py=mm.actuator_biasprm[i][2]),
               String("act ") + String(i) + " biasprm[2]", TOL, n)
    # The defect made both of these a gain-1 torque motor: kp 1, everything
    # else 0. Named so the failure says which.
    assert_true(
        abs(fmd.actuators[0].kp - 100.0) < TOL,
        "actuator 0 has gain " + String(fmd.actuators[0].kp) + ", not 100:"
        " the class's <position kp> never reached the <general> element",
    )
    assert_true(
        abs(fmd.actuators[1].bias1) < TOL
        and abs(fmd.actuators[0].bias1 + 100.0) < TOL,
        "biasprm[1] does not separate the position servo from the velocity"
        " one: got " + String(fmd.actuators[0].bias1) + " and "
        + String(fmd.actuators[1].bias1),
    )
    print("  values compared against MuJoCo:", n)
    assert_true(n == 8, "expected 8 comparisons, made " + String(n))


def _ours_damped(xml: String) raises -> List[Float64]:
    var fmd = parse_xml_full(xml, String("."))
    var dims = dims_from_flat(fmd, max_contacts=8, nmesh_verts=64)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var d = Data[DT, DynDims, 1](dims)
    for i in range(nq):
        d.qpos.data[i] = sf.qpos0.data[i]
    d.qvel.data[0] = Scalar[DT](0.7)
    d.qvel.data[1] = Scalar[DT](-0.4)
    var integ = StudioIntegPyr(dims)
    for _ in range(DAMP_STEPS):
        integ.step["cpu"](d, m)
    var out = List[Float64]()
    for i in range(nq):
        out.append(Float64(d.qpos.data[i]))
    for i in range(nv):
        out.append(Float64(d.qvel.data[i]))
    return out^


def _mj_damped(xml: String) raises -> List[Float64]:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(xml)
    var d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qvel[0] = Float64(0.7)
    d.qvel[1] = Float64(-0.4)
    for _ in range(DAMP_STEPS):
        mujoco.mj_step(m, d)
    var out = List[Float64]()
    for i in range(2):
        out.append(Float64(py=d.qpos[i]))
    for i in range(2):
        out.append(Float64(py=d.qvel[i]))
    return out^


def test_negative_joint_damping_reaches_both_euler_tests() raises:
    """AUD-42, three legs — one per branch of MuJoCo's two tests."""
    print("=== AUD-42: negative damping, three legs ===")
    var n = 0
    var legs: List[String] = [
        String("mixed  (+0.5, -0.02)"),
        String("neg    ( 0.0, -0.02)"),
        String("pos    (+0.5, +0.02)"),
    ]
    var d1s: List[String] = [String("0.5"), String("0"), String("0.5")]
    var d2s: List[String] = [String("-0.02"), String("-0.02"), String("0.02")]
    var mixed_qvel1 = Float64(0)
    var neg_qvel1 = Float64(0)
    for k in range(3):
        var xml = _damp_xml(d1s[k], d2s[k])
        var ours = _ours_damped(xml)
        var refs = _mj_damped(xml)
        print("  ", legs[k])
        for i in range(4):
            var nm = String("qpos") if i < 2 else String("qvel")
            print("     ", nm, i % 2, "  ours", ours[i], " MuJoCo", refs[i])
            _close(ours[i], refs[i],
                   legs[k] + " " + nm + String(i % 2), DAMP_TOL, n)
        if k == 0:
            mixed_qvel1 = refs[3]
        elif k == 1:
            neg_qvel1 = refs[3]
    # ⚠⚠ NON-VACUITY, AND IT IS THE ONE THAT MATTERS HERE. The three legs
    # must actually DISAGREE; if MuJoCo's own two tests collapsed into one,
    # every leg above would pass with the defect still in place.
    var split = abs(mixed_qvel1 - neg_qvel1)
    print("  MuJoCo's own mixed-vs-negative-only gap on qvel1:", split)
    assert_true(
        split > 1e-6,
        "MuJoCo gives the same answer whether or not a POSITIVE damping is"
        " present alongside the negative one, so this file cannot see the"
        " difference between its two tests: gap " + String(split),
    )
    print("  values compared against MuJoCo:", n)
    assert_true(n == 12, "expected 12 comparisons, made " + String(n))


def main() raises:
    var suite = TestSuite()
    suite.test[test_fromto_reshapes_a_box_and_an_ellipsoid]()
    suite.test[test_settotalmass_applies_without_inertiafromgeom]()
    suite.test[test_a_class_level_gear_keeps_all_six_components]()
    suite.test[test_a_general_element_inherits_a_servo_class]()
    suite.test[test_negative_joint_damping_reaches_both_euler_tests]()
    suite^.run()

"""`<option wind>` — the fluid's own velocity (AUD-27), against 3.12.

    pixi run mojo run -I . tests/physics3d/test_wind_vs_mujoco.mojo

WIND IS NOT A FORCE. It is the velocity OF THE FLUID, and both of MuJoCo's
fluid models rotate it into the body's inertial frame and subtract it from the
body's linear velocity before any drag term is evaluated
(`engine_passive.c:1167-1174` for the inertia-box model, the identical block
at `:1239-1248` for the ellipsoid one). Every viscous and pressure term
downstream is then a function of the velocity RELATIVE TO THE FLUID.

Unread, the whole model was evaluated on the velocity relative to the WORLD,
so a body at rest in a moving fluid felt nothing at all — `qfrc_passive` zero
where MuJoCo reads 0.108 N on the fixture below. That is the shape of the
defect: not a small numeric drift but a force that is simply absent, and it is
silent because a model without wind gives the same answer either way.

⚠ THE ZERO-WIND LEG IS THE CONTROL AND IT IS NOT OPTIONAL. Subtracting a
rotated vector inside the fluid loop is exactly the kind of change that can
perturb the no-wind path too — a sign, a frame, a term applied to the angular
half. The first test below runs the same fixture with `wind="0 0 0"` and holds
it to the same bound, so a fix that moved the untouched case would fail here
rather than in a swimmer rollout three weeks later.

⚠ AND THE THREE LEGS MUST DISAGREE WITH EACH OTHER, which the last assertion
checks. If our engine ignored `wind` again, all three legs would still match
each other perfectly and only differ from MuJoCo — so the file prints the
spread between MuJoCo's own legs beside the per-leg residual.

The fixture has zero gravity: every number below is drag and nothing else.
Body `a` is an axis-aligned box; body `b` is a capsule at a 45-degree quat, so
the rotation into the inertial frame is not the identity and a wind subtracted
in the wrong frame cannot pass.
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from noeira.physics3d.fields import Model, Data, DynDims
from noeira.physics3d.parser.full_parser import parse_xml_full
from noeira.physics3d.parser.runtime_load import (
    dims_from_flat,
    build_model_runtime,
    spec_fields_runtime,
)
from noeira.physics3d.studio.stepping import StudioIntegPyr

comptime DT = DType.float64
comptime STEPS = 30

# Set from the measurement, not inherited: the three legs land at ~1e-15 on
# qvel after 30 steps. 1e-11 is four orders of slack on that and still seven
# orders below the ~5e-3 spread between the legs themselves.
comptime TOL = 1e-11

comptime WIND_XML = String(
    """<mujoco model="wind">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0" density="1.2" viscosity="0.00002"
          integrator="Euler" wind="WINDVEC"/>
  <worldbody>
    <body name="a" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.2 0.1 0.05" density="400"/>
    </body>
    <body name="b" pos="2 0 1" quat="0.9238795325112867 0 0.3826834323650898 0">
      <freejoint/>
      <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.05" density="400"/>
    </body>
  </worldbody>
</mujoco>
"""
)


def _xml(wind: String) -> String:
    return WIND_XML.replace(String("WINDVEC"), wind)


def _seed(mut v: List[Float64]):
    """One seed for every leg, so the legs differ only by the wind."""
    v[0] = 0.5
    v[4] = 1.1
    v[6] = -0.3
    v[10] = 0.8


def _ours(xml: String) raises -> List[Float64]:
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
    var seed = List[Float64]()
    for _ in range(nv):
        seed.append(0.0)
    _seed(seed)
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](seed[i])
    var integ = StudioIntegPyr(dims)
    for _ in range(STEPS):
        integ.step["cpu"](d, m)
    var out = List[Float64]()
    for i in range(nv):
        out.append(Float64(d.qvel.data[i]))
    return out^


def _mj(xml: String) raises -> List[Float64]:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(xml)
    var d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    var nv = Int(py=m.nv)
    var seed = List[Float64]()
    for _ in range(nv):
        seed.append(0.0)
    _seed(seed)
    for i in range(nv):
        d.qvel[i] = seed[i]
    for _ in range(STEPS):
        mujoco.mj_step(m, d)
    var out = List[Float64]()
    for i in range(nv):
        out.append(Float64(py=d.qvel[i]))
    return out^


def _leg(wind: String, label: String) raises -> List[Float64]:
    """One wind setting: compare ours against MuJoCo, return MuJoCo's qvel."""
    var xml = _xml(wind)
    var ours = _ours(xml)
    var refs = _mj(xml)
    assert_true(
        len(ours) == 12 and len(refs) == 12,
        label + ": expected 12 dofs, got " + String(len(ours)) + " / "
        + String(len(refs)),
    )
    var worst = Float64(0)
    var worst_i = 0
    for i in range(12):
        var e = abs(ours[i] - refs[i])
        if e > worst:
            worst = e
            worst_i = i
    print("  ", label, " worst |d(qvel)| =", worst, " at dof", worst_i)
    print("      ours[0,4,6,10] =", ours[0], ours[4], ours[6], ours[10])
    print("      MuJoCo        =", refs[0], refs[4], refs[6], refs[10])
    assert_true(
        worst <= TOL,
        label + ": worst |d(qvel)| = " + String(worst) + " at dof "
        + String(worst_i) + ", bound " + String(TOL),
    )
    return refs^


def test_zero_wind_is_unmoved() raises:
    """The control: the case the fix must NOT have touched."""
    print("=== wind='0 0 0' — the untouched path ===")
    _ = _leg(String("0 0 0"), String("zero wind"))


def test_an_axial_wind_matches_mujoco() raises:
    print("=== wind='3 0 0' ===")
    _ = _leg(String("3 0 0"), String("axial wind"))


def test_a_three_axis_wind_matches_mujoco() raises:
    """⚠ THE FRAME TEST. Body `b` sits at a 45-degree quat, so a wind
    subtracted in the world frame, in the body frame rather than the
    INERTIAL frame, or with its sign flipped all give different answers on
    this leg and the same answer on an axis-aligned one.
    """
    print("=== wind='-1.5 2.25 0.75' ===")
    _ = _leg(String("-1.5 2.25 0.75"), String("three-axis wind"))


def test_the_three_legs_are_actually_different() raises:
    """⚠⚠ NON-VACUITY. Three legs that agree with each other test nothing:
    an engine that ignored `wind` entirely would produce identical qvel for
    all three, and each leg's comparison would fail only because of MuJoCo,
    not because of the wind. Print the spread.
    """
    print("=== non-vacuity: MuJoCo's own three legs disagree ===")
    var a = _mj(_xml(String("0 0 0")))
    var b = _mj(_xml(String("3 0 0")))
    var c = _mj(_xml(String("-1.5 2.25 0.75")))
    var ab = Float64(0)
    var ac = Float64(0)
    for i in range(12):
        var e1 = abs(a[i] - b[i])
        if e1 > ab:
            ab = e1
        var e2 = abs(a[i] - c[i])
        if e2 > ac:
            ac = e2
    print("  |zero - axial|      =", ab)
    print("  |zero - three-axis|  =", ac)
    assert_true(
        ab > 1e3 * TOL and ac > 1e3 * TOL,
        "MuJoCo gives (nearly) the same rollout with and without wind on this"
        " fixture, so the three tests above cannot see whether wind is read:"
        " gaps " + String(ab) + " and " + String(ac),
    )
    # And ours must move too — the same spread, computed on our side.
    var oa = _ours(_xml(String("0 0 0")))
    var ob = _ours(_xml(String("3 0 0")))
    var oab = Float64(0)
    for i in range(12):
        var e = abs(oa[i] - ob[i])
        if e > oab:
            oab = e
    print("  ours |zero - axial| =", oab)
    assert_true(
        abs(oab - ab) <= TOL,
        "our wind response has a different SIZE from MuJoCo's: ours "
        + String(oab) + ", MuJoCo " + String(ab),
    )


def main() raises:
    var suite = TestSuite()
    suite.test[test_zero_wind_is_unmoved]()
    suite.test[test_an_axial_wind_matches_mujoco]()
    suite.test[test_a_three_axis_wind_matches_mujoco]()
    suite.test[test_the_three_legs_are_actually_different]()
    suite^.run()

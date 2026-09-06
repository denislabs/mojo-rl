"""A `<fixed>` tendon's `limited range` produces a constraint row — vs MuJoCo 3.10.0.

WHY THIS EXISTS. `build_tendon_limit_rows` had `else: continue` for every
tendon that was not `<spatial>`, so a fixed tendon's `range` never became a
row on any solver path, while `build_tendon_equality_rows` right below it
handled both kinds. MuJoCo's `mj_instantiateLimit` does not distinguish the
kinds. ToddlerBot's waist is coupled through two fixed tendons with
`range="-0.001 0.001"`; the missing rows are why its `waist_yaw` walked
4e-02 from MuJoCo in 100 steps (PERFORMANCE.md §13.29) — found by stepping
from MuJoCo's own state and watching the one-step error jump from 1e-13 to
3e-02 on exactly the step MuJoCo's `limTen` row appears.

WHAT IT GATES. Two hinges coupled by `fixed` tendon `coef=(1,-1)` with a
tight range, driven APART by seeded velocities so the limit engages within
the first few steps. `|d(qvel)|` after 1, 5 and 30 steps, ours vs MuJoCo,
runtime path, pyramidal (tendon limits are pyramidal-only here).

MEASURED on the pre-fix tree (ef2ee73b): 1 step 0.0 (the range is not yet
reached — both sides have no row), 5 steps 2.063 (MuJoCo's row is pushing
back, ours does not exist). After the fix: rounding level at all three.

⚠ THE TENDON MUST BE VIOLATED BY THE FIFTH STEP or the row is never built
on either side and the gate is blind — `test_the_limit_is_engaged` asserts
that MuJoCo itself carries a `limTen` row after step 5.

Run: pixi run mojo run -I . tests/physics3d/test_fixed_tendon_limit_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
)
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.studio.stepping import StudioIntegPyr


comptime DT = DType.float64

comptime FIXED = """
<mujoco model="fixed_tendon_limit">
  <option timestep="0.002" gravity="0 0 -9.81" cone="pyramidal"/>
  <worldbody>
    <body name="a" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0" damping="0.01"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.02" mass="0.3"/>
    </body>
    <body name="b" pos="0.5 0 1">
      <joint name="j2" type="hinge" axis="0 1 0" damping="0.01"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.02" mass="0.3"/>
    </body>
  </worldbody>
  <tendon>
    <fixed name="couple" limited="true" range="-0.02 0.02"
           solreflimit="0.02 1" solimplimit="0.9 0.95 0.001 0.5 2">
      <joint joint="j1" coef="1"/>
      <joint joint="j2" coef="-1"/>
    </fixed>
  </tendon>
</mujoco>
"""

comptime TOL: Float64 = 1e-9
comptime SEED_J1: Float64 = 4.0
comptime SEED_J2: Float64 = -4.0


def _ours(nstep: Int) raises -> List[Float64]:
    var fmd = parse_xml_full(materialize[FIXED](), String("."))
    var dims = dims_from_flat(fmd, max_contacts=8, nmesh_verts=64)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var d = Data[DT, DynDims, 1](dims)
    for i in range(nq):
        d.qpos.data[i] = sf.qpos0.data[i]
    d.qvel.data[0] = Scalar[DT](SEED_J1)
    d.qvel.data[1] = Scalar[DT](SEED_J2)
    var integ = StudioIntegPyr(dims)
    for _ in range(nstep):
        integ.step["cpu"](d, m)
    var out = List[Float64]()
    for i in range(nv):
        out.append(Float64(d.qvel.data[i]))
    return out^


def _mj(nstep: Int) raises -> Tuple[List[Float64], Int]:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[FIXED]())
    var d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qvel[0] = SEED_J1
    d.qvel[1] = SEED_J2
    var n_lim = 0
    for _ in range(nstep):
        mujoco.mj_step(m, d)
    # count limTen rows at the post-step state
    mujoco.mj_forward(m, d)
    var nefc = Int(py=d.nefc)
    for i in range(nefc):
        if Int(py=d.efc_type[i]) == 4:
            n_lim += 1
    var qv = d.qvel.flatten().tolist()
    var out = List[Float64]()
    for i in range(2):
        out.append(Float64(py=qv[i]))
    return (out^, n_lim)


def _maxdiff(a: List[Float64], b: List[Float64]) -> Float64:
    var w = 0.0
    for i in range(min(len(a), len(b))):
        var e = abs(a[i] - b[i])
        if e > w:
            w = e
    return w


def test_the_limit_is_engaged() raises:
    print("=== fixed tendon limit: MuJoCo carries the row ===")
    var r = _mj(5)
    print("  limTen rows after 5 steps:", r[1])
    assert_true(r[1] >= 1, "the fixture never violates its tendon range — gate is blind")
    print("  PASS")


def test_fixed_tendon_limit_rows_match_mujoco() raises:
    print("=== fixed tendon limit rows vs MuJoCo ===")
    for n in [1, 5, 30]:
        var e = _maxdiff(_ours(n), _mj(n)[0])
        print("  ", n, "step(s): |d(qvel)| =", e)
        assert_true(
            e <= TOL,
            "fixed tendon limit after " + String(n) + " step(s) disagrees with"
            " MuJoCo by " + String(e) + " (tol " + String(TOL) + ") — the"
            " <fixed> branch of build_tendon_limit_rows is missing or wrong",
        )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

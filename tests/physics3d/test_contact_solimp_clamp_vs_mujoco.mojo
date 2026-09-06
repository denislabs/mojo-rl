"""A contact `solimp` with dmax = 1 is clamped to mjMAXIMP on EVERY row — vs MuJoCo 3.10.0.

WHY THIS EXISTS. MuJoCo clamps `solimp[0]`, `[1]` and `[3]` to
[mjMINIMP, mjMAXIMP] = [0.0001, 0.9999] before deriving the impedance AND
the spring/damper (`engine_core_constraint.c:2044`). Ours clamped them
before the impedance, and `solref_spring_damper` — which derives
`B = 2/(dmax*timeconst)` — said in its docstring that "the caller clamps".
The normal-row caller did; the two friction-row callers in `contact_solve`
(pyramidal and elliptic) passed the contact's raw `solimp[1]`. A dmax of
exactly 1 therefore reached the friction damper unclamped: B off by 1e-4,
which is invisible at rest (the damper term is B*vel) and 9e-6 in qvel per
step once the foot moves. anymal_c's feet carry `solimp="0.015 1 0.03"`
under `priority="1"`, so the unmixed 1 came through; every priority-footed
quadruped on the fifty-step board sat at 1e-6 for it (anymal_b/c, spot,
a1, go1, go2 — PERFORMANCE.md §13.34). Found by ablation from MuJoCo's own
state: writing the XML's 1 as 0.9999 took the per-step error from 9e-6 to
4e-14. The clamp now lives inside `solref_spring_damper`.

WHAT IT GATES. A sphere with `priority="1" solimp="0.015 1 0.03"` sliding
on a plane under a seeded horizontal velocity, so the friction rows carry a
velocity term. `|d(qvel)|` after 1, 5 and 30 steps, ours vs MuJoCo, runtime
path, pyramidal. A non-vacuity arm asserts MuJoCo's contact rows carry
force at steps 5 and 30 (the contact is soft at first touch, dmin = 0.015). MEASURED on the pre-fix tree: 6.7e-06 after ONE step
(tol 1e-10) — after the fix, 4.5e-19 / 1.6e-18 / 3.6e-15.

Run: pixi run mojo run -I . tests/physics3d/test_contact_solimp_clamp_vs_mujoco.mojo
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

# ⚠ dmax = 1 EXACTLY is the point of the fixture. `priority="1"` makes the
# sphere's parameters win unmixed (an equal-priority mix with the plane's
# default 0.95 would land at 0.975 and never touch the clamp).
comptime SLIDE = """
<mujoco model="solimp_clamp">
  <option timestep="0.002" gravity="0 0 -9.81" cone="pyramidal"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="ball" pos="0 0 0.049">
      <joint type="free"/>
      <geom type="sphere" size="0.05" mass="0.2" priority="1"
            solimp="0.015 1 0.03" friction="0.8 0.02 0.01" condim="3"/>
    </body>
  </worldbody>
</mujoco>
"""

comptime TOL: Float64 = 1e-10
comptime SEED_VX: Float64 = 1.5
comptime SEED_WY: Float64 = 4.0


def _ours(nstep: Int) raises -> List[Float64]:
    var fmd = parse_xml_full(materialize[SLIDE](), String("."))
    var dims = dims_from_flat(fmd, max_contacts=8, nmesh_verts=64)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var d = Data[DT, DynDims, 1](dims)
    for i in range(nq):
        d.qpos.data[i] = sf.qpos0.data[i]
    d.qvel.data[0] = Scalar[DT](SEED_VX)
    d.qvel.data[4] = Scalar[DT](SEED_WY)
    var integ = StudioIntegPyr(dims)
    for _ in range(nstep):
        integ.step["cpu"](d, m)
    var out = List[Float64]()
    for i in range(nv):
        out.append(Float64(d.qvel.data[i]))
    return out^


def _mj(nstep: Int) raises -> Tuple[List[Float64], Float64]:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[SLIDE]())
    var d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qvel[0] = SEED_VX
    d.qvel[4] = SEED_WY
    for _ in range(nstep):
        mujoco.mj_step(m, d)
    # the friction rows' force at the post-step state (pyramidal: every
    # row of a contact carries the cone's edges; sum |force| over them)
    mujoco.mj_forward(m, d)
    var fsum = 0.0
    for i in range(Int(py=d.nefc)):
        fsum += abs(Float64(py=d.efc_force[i]))
    var qv = d.qvel.flatten().tolist()
    var out = List[Float64]()
    for i in range(Int(py=m.nv)):
        out.append(Float64(py=qv[i]))
    return (out^, fsum)


def _maxdiff(a: List[Float64], b: List[Float64]) -> Float64:
    var w = 0.0
    for i in range(min(len(a), len(b))):
        var e = abs(a[i] - b[i])
        if e > w:
            w = e
    return w


def test_the_contact_is_live() raises:
    print("=== solimp clamp: MuJoCo's contact rows carry force while sliding ===")
    # dmin = 0.015 makes the contact SOFT at first touch, so the load builds
    # as the sphere sinks: ~0.2 N of row force at step 5, ~3.4 N at step 30.
    var r5 = _mj(5)
    var r30 = _mj(30)
    print("  sum |efc_force| after 5 steps:", r5[1], " after 30:", r30[1])
    assert_true(r5[1] > 0.1 and r30[1] > 1.0,
                "the sphere is not in a loaded contact — gate is blind")
    print("  PASS")


def test_friction_damper_matches_mujoco() raises:
    print("=== contact solimp dmax=1: friction damper vs MuJoCo ===")
    for n in [1, 5, 30]:
        var e = _maxdiff(_ours(n), _mj(n)[0])
        print("  ", n, "step(s): |d(qvel)| =", e)
        assert_true(
            e <= TOL,
            "sliding contact after " + String(n) + " step(s) disagrees with"
            " MuJoCo by " + String(e) + " (tol " + String(TOL) + ") — a"
            " solimp dmax reached a spring/damper unclamped",
        )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

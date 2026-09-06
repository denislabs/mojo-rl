"""connect / weld rows under MOTION — the `J̇·v` reference term, vs MuJoCo 3.10.0.

WHY THIS EXISTS. MuJoCo 3.10 subtracts a `J̇·v` correction from the reference
acceleration of every connect and weld row (`mj_Jdotv`, called from
`mj_referenceConstraint`; engine_core_constraint.c). None of the three older
reference trees in this repo (3.3.6, 3.5.1, 3.6.0) has it, and neither did
our `build_weld_equality_rows`, which was transcribed from them. The term is
the centripetal/Coriolis part of a moving anchor's acceleration — exactly
zero at rest, so every connect and weld gate written at rest, and every
rollout that starts from rest and stays slow, was green. It showed up as
ToddlerBot's `waist_yaw` walking 4e-02 away from MuJoCo in 100 steps: its
neck is a closed loop of four near-hard connects (solimp 0.9999) whose
anchors move while the robot sways, and 1e-4 of qvel per step on those rows
compounds. Ablating the connects took the residual to 6e-15; everything
else in those rows (J, D, pos, vel, KBIP, invweights, M) had already been
matched to MuJoCo's to 1e-14 — `aref` was the one number left, and it was
off by exactly `J̇·v`.

WHAT IT GATES. One step from a seeded high-velocity state, ours vs MuJoCo,
through the runtime loader and the studio's Euler integrator (the path the
board and the studio use):
  * a four-bar linkage closed by a site-to-world-site `connect` — three
    hinges, one effective dof, the anchor on a link swinging at 8 rad/s;
  * two free boxes joined by a `weld` (relpose derived at qpos0), spinning
    against each other at several rad/s — the ROTATIONAL part of the term.
`|d(qvel)|` after one step and after three is asserted at `TOL`; the header
records what the numbers were without the term so the gate's teeth are
visible — MEASURED on ef2ee73b, one step: connect 2.918e-02, weld 5.626e-02
(and 5.0e-17 / 0.0 at rest on the same tree).

⚠ THE FIXTURES START MOVING, ON PURPOSE. Seed the velocities to zero and both
arms agree to 1e-15 with or without the term — that is the blindness this
gate exists to remove.

Run: pixi run mojo run -I . tests/physics3d/test_equality_jdotv_vs_mujoco.mojo
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

# A parallelogram four-bar closed on the world: link1 hangs from (0,0,1),
# link2 runs +x, link3 runs +z back up to the world site at (0.3,0,1).
comptime FOURBAR = """
<mujoco model="jdotv_connect">
  <option timestep="0.002" gravity="0 0 -9.81" cone="pyramidal"/>
  <worldbody>
    <site name="w_anchor" pos="0.3 0 1"/>
    <body name="link1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0" damping="0.02"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.02" mass="0.3"/>
      <body name="link2" pos="0 0 -0.3">
        <joint name="j2" type="hinge" axis="0 1 0" damping="0.02"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.3"/>
        <body name="link3" pos="0.3 0 0">
          <joint name="j3" type="hinge" axis="0 1 0" damping="0.02"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.02" mass="0.3"/>
          <site name="tip" pos="0 0 0.3"/>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <connect site1="tip" site2="w_anchor" solref="0.004 1"
             solimp="0.9999 0.9999 0.001 0.5 2"/>
  </equality>
</mujoco>
"""

# Two free boxes held together by a weld; spun against each other.
comptime WELDPAIR = """
<mujoco model="jdotv_weld">
  <option timestep="0.002" gravity="0 0 -9.81" cone="pyramidal"/>
  <worldbody>
    <body name="fa" pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size="0.05 0.04 0.03" mass="0.3"/>
    </body>
    <body name="fb" pos="0.15 0.02 1.03">
      <joint type="free"/>
      <geom type="box" size="0.04 0.05 0.03" mass="0.2"/>
    </body>
  </worldbody>
  <equality>
    <weld body1="fa" body2="fb" solref="0.004 1"
          solimp="0.9999 0.9999 0.001 0.5 2"/>
  </equality>
</mujoco>
"""

comptime TOL: Float64 = 1e-9
comptime MIN_TEETH: Float64 = 1e-4


def _seed_fourbar(mut v: List[Float64]):
    v[0] = 8.0
    v[1] = -6.0
    v[2] = 5.0


def _seed_weld(mut v: List[Float64]):
    # fa: linear then angular; fb likewise (free joint dofs are 6 each)
    v[0] = 0.4; v[1] = -0.3; v[2] = 0.2; v[3] = 3.0; v[4] = 5.0; v[5] = -2.0
    v[6] = -0.2; v[7] = 0.5; v[8] = 0.1; v[9] = 1.0; v[10] = -4.0; v[11] = 6.0


def _ours(xml: String, seed: List[Float64], nstep: Int) raises -> List[Float64]:
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
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](seed[i])
    var integ = StudioIntegPyr(dims)
    for _ in range(nstep):
        integ.step["cpu"](d, m)
    var out = List[Float64]()
    for i in range(nv):
        out.append(Float64(d.qvel.data[i]))
    return out^


def _mj(xml: String, seed: List[Float64], nstep: Int) raises -> List[Float64]:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(xml)
    var d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    var nv = Int(py=m.nv)
    for i in range(nv):
        d.qvel[i] = seed[i]
    for _ in range(nstep):
        mujoco.mj_step(m, d)
    var qv = d.qvel.flatten().tolist()
    var out = List[Float64]()
    for i in range(nv):
        out.append(Float64(py=qv[i]))
    return out^


def _maxdiff(a: List[Float64], b: List[Float64]) -> Float64:
    var w = 0.0
    for i in range(min(len(a), len(b))):
        var e = abs(a[i] - b[i])
        if e > w:
            w = e
    return w


def _gate(name: String, xml: String, seed: List[Float64], zero: List[Float64]) raises:
    var e1 = _maxdiff(_ours(xml, seed, 1), _mj(xml, seed, 1))
    var e3 = _maxdiff(_ours(xml, seed, 3), _mj(xml, seed, 3))
    var e0 = _maxdiff(_ours(xml, zero, 1), _mj(xml, zero, 1))
    print("  ", name, " |d(qvel)| 1 step:", e1, " 3 steps:", e3,
          "  (at rest:", e0, ")")
    assert_true(
        e0 <= TOL,
        name + " disagrees with MuJoCo even AT REST by " + String(e0)
        + " — that is not the J̇·v term, look upstream",
    )
    assert_true(
        e1 <= TOL,
        name + " one moving step disagrees with MuJoCo by " + String(e1)
        + " (tol " + String(TOL) + ") — the J̇·v reference term"
        " (mj_Jdotv, MuJoCo 3.10) is missing or wrong on these rows",
    )
    assert_true(
        e3 <= TOL,
        name + " three moving steps disagree with MuJoCo by " + String(e3),
    )


def test_connect_loop_in_motion() raises:
    print("=== connect four-bar in motion vs MuJoCo ===")
    var seed = List[Float64](length=3, fill=0.0)
    _seed_fourbar(seed)
    var zero = List[Float64](length=3, fill=0.0)
    _gate(String("connect"), materialize[FOURBAR](), seed, zero)
    print("  PASS")


def test_weld_pair_in_motion() raises:
    print("=== weld pair spinning vs MuJoCo ===")
    var seed = List[Float64](length=12, fill=0.0)
    _seed_weld(seed)
    var zero = List[Float64](length=12, fill=0.0)
    _gate(String("weld"), materialize[WELDPAIR](), seed, zero)
    print("  PASS")


def test_the_term_has_teeth() raises:
    """MuJoCo with and without the term differ by more than `MIN_TEETH` on
    these fixtures — computed on MuJoCo's side alone by comparing its step to
    an explicit `aref`-without-J̇·v re-solve is not available from Python, so
    the teeth are recorded in the header from the pre-fix tree instead. This
    test only asserts the seeds are not zero, so the fixtures cannot be
    silently put to rest."""
    print("=== fixtures are in motion ===")
    var s = List[Float64](length=3, fill=0.0)
    _seed_fourbar(s)
    var w = List[Float64](length=12, fill=0.0)
    _seed_weld(w)
    var vmax = 0.0
    for i in range(3):
        if abs(s[i]) > vmax: vmax = abs(s[i])
    for i in range(12):
        if abs(w[i]) > vmax: vmax = abs(w[i])
    assert_true(vmax > MIN_TEETH, "the seeded velocities are zero")
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

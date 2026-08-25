"""`<option noslip_iterations>` reaches the solver on the RUNTIME path.

WHAT THIS GATES, AND WHY IT IS NOT THE OTHER THREE NOSLIP TESTS. Until
2026-08-25 the iteration count reached `solve_newton` ONLY as its compile-time
`NOSLIP_ITER`, threaded from `ModelDefFromXML.NOSLIP_ITER` — which
`xml_parser._scan_noslip_iterations` fills when a model is baked into Mojo
source. `test_noslip_vs_mujoco`, `test_noslip_elliptic_vs_mujoco` and
`test_noslip_blocked_kernel` all take that door, and all three were green.

Every caller that loads an MJCF at RUN time takes the other one:
`parse_xml_full` -> `FlatModelDef` -> `build_model_runtime`. `FlatModelDef` had
no field for the attribute and `_parse_option` did not read it, so the count
was 0 no matter what the file said — the studio, and every fidelity harness
that mirrors it, stepped `mj_solNoSlip`-requesting models without the pass.
Three green gates and a whole class of callers with the feature off: the shape
of `feedback_a_gate_that_shares_its_reference_implementation_is_blind`.

⚠⚠ THE PLUMBING ASSERTIONS BELOW ARE NOT THE GATE. `fmd.noslip_iterations ==
5` and a meta slot holding 5 were BOTH true of `noslip_tolerance` for months
while nothing ran the pass those numbers configure. The gate is
`test_the_pass_actually_runs_on_the_runtime_path`, which steps the same state
twice through the runtime loader — once from an XML that asks for the pass and
once from the SAME XML with the attribute deleted — and requires the two to
disagree, with MuJoCo as the referee for which one is right.

MEASURED (MuJoCo 3.10.0, this fixture, one step from the shared settled state
with the slam velocity seeded):

    MuJoCo noslip 5 vs MuJoCo, attribute absent   |d(qvel)| = 6.7618e-02
    ours ON  vs MuJoCo noslip 5                   |d(qvel)| = 4.6391e-13
    ours OFF vs MuJoCo noslip 5                   |d(qvel)| = 6.7618e-02

⚠ READ THE THIRD LINE AGAINST THE FIRST: they agree to TWELVE digits. The OFF
arm is not merely "worse", it is wrong by EXACTLY the amount the pass is worth
— which is what "the count never reached the solver" looks like, and what this
file exists to keep from coming back. Both arms are asserted so that "the pass
runs" cannot be satisfied by two equally wrong answers.

⚠ ELLIPTIC. `noslip_elliptic` and `noslip_pyramidal` are different algorithms
over different row layouts and the caller picks by cone, so this covers one of
the two branches. The pyramidal branch's runtime behaviour is covered by
`robot_soccer_kit` on the Menagerie board, where the pass IS the whole
residual — but the board is not a committed test, which is why the sharper
elliptic fixture is the one written down here.

⚠ THE FIXTURE MUST BE IN CONTACT AT STEP 0 OR IT GATES NOTHING. The chain is
placed with its capsule axes exactly one radius above the plane and slammed
downward at 40 m/s; `test_the_fixture_is_in_contact` asserts the contact count
rather than trusting the geometry.
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, assert_equal, TestSuite

from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
)
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.studio.stepping import (
    StudioIntegEll, studio_cone_of, studio_integrator_of,
)
from mojo_rl.physics3d.types import ConeType, IntegratorType
from mojo_rl.physics3d.gpu.constants import (
    MODEL_META_IDX_NOSLIP_ITERATIONS, META_IDX_NUM_CONTACTS,
)


comptime DT = DType.float64

# ⚠ `cone="elliptic"` AND NO `integrator=`, so `studio_integrator_of` returns
# EULER and the arm exercised is `StudioIntegEll`. Both are asserted below
# rather than assumed — a fixture that silently moved to another integrator
# would still "pass" a parity check against a MuJoCo told the same thing.
comptime CHAIN_ON = """
<mujoco model="noslip_runtime">
  <option timestep="0.002" gravity="0 0 -9.81" cone="elliptic"
          noslip_iterations="5" noslip_tolerance="0"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1" condim="3"
          friction="1 .005 .0001"/>
    <body name="l1" pos="0 0 .05">
      <joint type="free" name="root"/>
      <geom name="g1" type="capsule" fromto="0 0 0 .3 0 0" size=".05"
            condim="3" friction=".7 .05 .05"/>
      <body name="l2" pos=".3 0 0">
        <joint type="hinge" name="j2" axis="0 1 0" range="-60 60"
               limited="true" frictionloss="0.05"/>
        <geom name="g2" type="capsule" fromto="0 0 0 .3 0 0" size=".05"
              condim="3" friction=".7 .05 .05"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# ⚠⚠ IDENTICAL EXCEPT THE ATTRIBUTE IS GONE — not `noslip_iterations="0"`.
# Absent is what the defect looked like from the solver's side, and it is also
# the only spelling that proves `_parse_option`'s DEFAULT is 0 rather than
# whatever the previous parse left behind.
comptime CHAIN_OFF = """
<mujoco model="noslip_runtime">
  <option timestep="0.002" gravity="0 0 -9.81" cone="elliptic"
          noslip_tolerance="0"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1" condim="3"
          friction="1 .005 .0001"/>
    <body name="l1" pos="0 0 .05">
      <joint type="free" name="root"/>
      <geom name="g1" type="capsule" fromto="0 0 0 .3 0 0" size=".05"
            condim="3" friction=".7 .05 .05"/>
      <body name="l2" pos=".3 0 0">
        <joint type="hinge" name="j2" axis="0 1 0" range="-60 60"
               limited="true" frictionloss="0.05"/>
        <geom name="g2" type="capsule" fromto="0 0 0 .3 0 0" size=".05"
              condim="3" friction=".7 .05 .05"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# The slam. `SEED_VZ` is why the fixture discriminates: a converged solve with
# nothing sliding has nothing for `mj_solNoSlip` to remove, which is exactly
# what `test_noslip_is_inert_on_a_converged_solve` measures (8.9e-16 there).
comptime N_SETTLE: Int = 400
comptime SEED_VX: Float64 = 3.0
comptime SEED_VY: Float64 = 1.0
comptime SEED_VZ: Float64 = -40.0
comptime SEED_WZ: Float64 = 3.0
comptime SEED_J2: Float64 = 4.0

# Ours-with-the-pass against MuJoCo-with-the-pass. Measured 4.6391e-13 of qvel
# on a step where `|qvel|` is ~4e+1 — round-off, not a solver allowance. Both
# sides are float64 running the same arithmetic from the same state.
comptime TOL_ON: Float64 = 1e-11
# And the OFF arm must miss by at least this. It misses by 6.7618e-02 when the
# pass genuinely does not run, so this sits nearly three orders below that —
# far enough not to be brittle, and eight orders above TOL_ON so the two
# assertions cannot both be satisfied by the same answer.
comptime MIN_OFF_MISS: Float64 = 1e-4


@fieldwise_init
struct _Ours(Movable):
    """One runtime-path rollout: `qvel` after N steps, and the contact count."""

    var qvel: List[Float64]
    var ncon: Int
    var noslip_meta: Int
    var cone: Int
    var integ: Int


def _ours(xml: String, qpos: List[Float64], nstep: Int) raises -> _Ours:
    """Load `xml` THE WAY THE STUDIO DOES and step it from the seeded slam.

    ⚠ `parse_xml_full` -> `dims_from_flat` -> `build_model_runtime` is the
    whole point: nothing here is a compile-time model, so `NOSLIP_ITER` cannot
    carry the count and the meta slot is the only route it has.
    """
    var fmd = parse_xml_full(xml, String("."))
    var dims = dims_from_flat(fmd, max_contacts=32, nmesh_verts=1024)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)

    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var d = Data[DT, DynDims, 1](dims)
    # ⚠⚠ THE SETTLED POSE, NOT `qpos0` — SHARED WITH THE REFERENCE. From
    # `qpos0` the capsules sit EXACTLY one radius above the plane, i.e. at
    # dist 0, and a 40 m/s slam onto a zero-depth contact is decided by
    # whichever engine's broadphase admits the pair first: the first version
    # of this file had 2 contacts to MuJoCo's 4 and read 26 of qvel apart with
    # the wiring working perfectly. Settling first puts both capsules in
    # unambiguous contact so the fixture is about the SWEEP.
    for i in range(min(nq, len(qpos))):
        d.qpos.data[i] = Scalar[DT](qpos[i])
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](0)
    # free joint: vx vy vz wx wy wz, then the hinge.
    d.qvel.data[0] = Scalar[DT](SEED_VX)
    d.qvel.data[1] = Scalar[DT](SEED_VY)
    d.qvel.data[2] = Scalar[DT](SEED_VZ)
    d.qvel.data[5] = Scalar[DT](SEED_WZ)
    d.qvel.data[6] = Scalar[DT](SEED_J2)

    var ell = StudioIntegEll(dims)
    for _ in range(nstep):
        ell.step["cpu"](d, m)

    var qv = List[Float64]()
    for i in range(nv):
        qv.append(Float64(d.qvel.data[i]))
    return _Ours(
        qvel=qv^,
        ncon=Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS])),
        noslip_meta=Int(
            Float64(m.meta.data[MODEL_META_IDX_NOSLIP_ITERATIONS])
        ),
        cone=studio_cone_of(fmd),
        integ=studio_integrator_of(fmd),
    )


def _mj(
    xml: String, qpos: List[Float64], nstep: Int, nv: Int
) raises -> List[Float64]:
    """MuJoCo from the same XML and the same seeded state.

    ⚠ `nv` IS PASSED IN rather than read off `m.nv`. A `PythonObject` is not
    `Intable`, and going through a float would make the loop bound depend on a
    conversion that has nothing to do with what is being measured.
    """
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(xml)
    var d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    for i in range(len(qpos)):
        d.qpos[i] = qpos[i]
    d.qvel[0] = SEED_VX
    d.qvel[1] = SEED_VY
    d.qvel[2] = SEED_VZ
    d.qvel[5] = SEED_WZ
    d.qvel[6] = SEED_J2
    for _ in range(nstep):
        mujoco.mj_step(m, d)
    # ⚠ `.tolist()` FIRST, THEN `Float64(py=...)`. A raw `d.qvel[i]` is a
    # numpy scalar wrapped in a `PythonObject`, which is neither `Floatable`
    # nor `Intable` on the Mojo side.
    var qv = d.qvel.flatten().tolist()
    var out = List[Float64]()
    for i in range(nv):
        out.append(Float64(py=qv[i]))
    return out^


def _settled() raises -> List[Float64]:
    """Drop the chain from `qpos0` and let it come to rest; return that qpos.

    ⚠ SETTLED ONCE AND SHARED BY ALL FOUR ROLLOUTS. Settling each arm
    separately would put 400 steps of divergence inside the number the gate
    reads — the mistake `test_noslip_elliptic_vs_mujoco`'s header records
    having made (a 3.2e-1 "effect" that was really 1.4e-8).

    ⚠ SETTLED FROM THE **OFF** XML so the pass under test has no hand in the
    state it is measured on.
    """
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[CHAIN_OFF]())
    var d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    for _ in range(N_SETTLE):
        mujoco.mj_step(m, d)
    var qp = d.qpos.flatten().tolist()
    var out = List[Float64]()
    for i in range(len(qp)):
        out.append(Float64(py=qp[i]))
    return out^


def _maxdiff(a: List[Float64], b: List[Float64]) -> Float64:
    var w = 0.0
    for i in range(min(len(a), len(b))):
        var e = abs(a[i] - b[i])
        if e > w:
            w = e
    return w


def test_the_option_reaches_the_runtime_model() raises:
    """`_parse_option` reads it, `FlatModelDef` carries it, meta holds it.

    ⚠ PLUMBING, NOT THE GATE — see the module header. Kept because it names
    WHICH of the three hops broke when the real gate below goes red.
    """
    print("=== noslip on the runtime path: the option is carried ===")
    var on = parse_xml_full(materialize[CHAIN_ON](), String("."))
    var off = parse_xml_full(materialize[CHAIN_OFF](), String("."))
    print("  fmd.noslip_iterations  ON =", on.noslip_iterations,
          "  OFF =", off.noslip_iterations)
    assert_equal(
        on.noslip_iterations, 5,
        "`_parse_option` did not read <option noslip_iterations='5'> — the"
        " runtime path has no other route for the count",
    )
    assert_equal(
        off.noslip_iterations, 0,
        "an XML with NO noslip_iterations did not default to 0",
    )
    var q0 = _settled()
    var ro = _ours(materialize[CHAIN_ON](), q0, 1)
    var rf = _ours(materialize[CHAIN_OFF](), q0, 1)
    print("  meta[NOSLIP_ITERATIONS] ON =", ro.noslip_meta,
          "  OFF =", rf.noslip_meta)
    assert_equal(
        ro.noslip_meta, 5,
        "`build_model_runtime` did not put the count in"
        " MODEL_META_IDX_NOSLIP_ITERATIONS; the solver reads it from there"
        " and nowhere else on this path",
    )
    assert_equal(rf.noslip_meta, 0, "the OFF arm's meta slot is not 0")


def test_the_fixture_is_in_contact() raises:
    """The chain hits the floor on step 1, and takes the elliptic Euler arm.

    Asserted rather than assumed: a fixture that floats gates nothing, and one
    that quietly changed integrator would still agree with a MuJoCo told the
    same thing.
    """
    print("=== noslip on the runtime path: the fixture ===")
    var r = _ours(materialize[CHAIN_ON](), _settled(), 1)
    print("  ncon =", r.ncon, "  cone =", r.cone, "  integrator =", r.integ)
    assert_true(
        r.ncon > 0,
        "the chain produced NO contacts on step 1 — with nothing touching,"
        " `mj_solNoSlip` has no friction rows to sweep and both arms below"
        " would agree for the wrong reason (ncon " + String(r.ncon) + ")",
    )
    assert_equal(
        r.cone, ConeType.ELLIPTIC,
        "the fixture is not on the elliptic cone, so it is not exercising"
        " `noslip_elliptic`",
    )
    assert_equal(
        r.integ, IntegratorType.EULER,
        "the fixture is not on the Euler arm the numbers were measured with",
    )


def test_the_pass_actually_runs_on_the_runtime_path() raises:
    """THE GATE. Ours matches MuJoCo with the pass; without it, ours misses.

    Both directions are required. Matching alone would be satisfiable by an
    engine that runs no pass against a reference that also does not; missing
    alone would be satisfiable by any divergence at all.
    """
    print("=== mj_solNoSlip on the RUNTIME path ===")
    var q0 = _settled()
    var ours_on = _ours(materialize[CHAIN_ON](), q0, 1)
    var ours_off = _ours(materialize[CHAIN_OFF](), q0, 1)
    var nv = len(ours_on.qvel)
    var mj_on = _mj(materialize[CHAIN_ON](), q0, 1, nv)
    var mj_off = _mj(materialize[CHAIN_OFF](), q0, 1, nv)

    var mj_effect = _maxdiff(mj_on, mj_off)
    var d_on = _maxdiff(ours_on.qvel, mj_on)
    var d_off = _maxdiff(ours_off.qvel, mj_on)
    print("  MuJoCo: the pass is worth      |d(qvel)| =", mj_effect)
    print("  ours ON  vs MuJoCo(noslip 5)   |d(qvel)| =", d_on)
    print("  ours OFF vs MuJoCo(noslip 5)   |d(qvel)| =", d_off)

    # Scope check first: if the reference itself does not move, the two
    # assertions below are about nothing.
    assert_true(
        mj_effect > MIN_OFF_MISS,
        "MuJoCo's OWN answer barely moves when the pass is removed"
        " (|d(qvel)| " + String(mj_effect) + ") — the fixture has stopped"
        " exercising mj_solNoSlip and nothing below is a test of it",
    )
    assert_true(
        d_on <= TOL_ON,
        "ours with <option noslip_iterations='5'> is " + String(d_on)
        + " from MuJoCo's answer (budget " + String(TOL_ON) + "). The OFF arm"
        " misses by " + String(d_off) + "; if the two are EQUAL the count is"
        " not reaching the solver on the runtime path at all — check"
        " MODEL_META_IDX_NOSLIP_ITERATIONS, not the sweep.",
    )
    assert_true(
        d_off >= MIN_OFF_MISS,
        "the arm with NO noslip_iterations agrees with MuJoCo to "
        + String(d_off) + ", so the ON arm's agreement proves nothing — this"
        " control is what tells a running pass from an absent one",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

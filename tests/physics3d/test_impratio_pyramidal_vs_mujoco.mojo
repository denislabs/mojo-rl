"""`<option impratio>` on the PYRAMIDAL cone vs MuJoCo — a box slammed into the
floor while sliding.

`impratio` is the ratio of frictional to normal constraint impedance.
`mj_makeImpedance` (`engine_core_constraint.c:2223`) applies it in TWO steps:

    R[i+1]   = R[i] / impratio
    con->mu  = friction[0] * sqrt(R[i+1]/R[i])      = friction[0]/sqrt(impratio)

and then, for a PYRAMIDAL contact, writes one common `R` over all `2*(dim-1)`
rows of that contact built from the REGULARIZED coefficient:

    Rpy = 2 * con->mu * con->mu * R[i]

⚠ THE PYRAMID JACOBIAN KEEPS THE RAW COEFFICIENT. `mj_instantiateContact`
builds the edges as `jac ± friction[k-1] * jac_k`
(`engine_core_constraint.c:1686`) — it has to, because `con->mu` does not
exist until `mj_makeImpedance` runs afterwards. So `friction[0]` and
`con->mu` are BOTH live in this contact and they are different numbers; using
the wrong one in either place is a silent factor of `impratio`.

WHY THIS FILE EXISTS RATHER THAN A BOARD ROW

`contact_solve.mojo` used the RAW `friction[0]` in `Rpy`, on both the Newton
and the PGS builders, making every pyramidal `R` a factor of `impratio` too
large and the edge constraints that much too soft. It is an EXACT no-op at
`impratio = 1`, and nothing in the tree exercised any other value on this
cone:

  * 27 Menagerie models set `<option impratio>`; all but two are
    `cone="elliptic"`, and the elliptic builder already regularized correctly.
  * of the two, `franka_emika_panda` has `ncon 0` at the board's pose, and
    `anybotics_anymal_c` is `cone="elliptic"` in `anymal_c.xml` — its
    `cone="pyramidal"` lives in `anymal_c_mjx.xml`, which the board filters
    out.

So no board row could move and none did. A defect that only one unshipped
COMBINATION of two options can reach needs a gate that FORCES the combination;
waiting for a model to ship it is not a plan. See
`feedback_a_latent_bug_filed_is_not_a_latent_bug_fixed`.

⚠⚠ THE SHARPEST READING IS THE RESPONSE, NOT THE ERROR. Before the fix this
fixture reported

    ours   |qvel(ir=10) - qvel(ir=1)|  =  0.0
    MuJoCo |qvel(ir=10) - qvel(ir=1)|  =  0.0638

— our pyramidal path did not respond to `impratio` AT ALL. The two `mu`s
cancelled: `Rpy` used the raw `friction[0]` and nothing else on this cone reads
the option, so changing it moved nothing. That is why assertion (3) exists;
the endpoint error (0.202 of qvel on step 1) says something is wrong, but the
0.0 says exactly what.

Measured, `max|d(qvel)|` against MuJoCo on the first step of this fixture, and
on `hello_robot_stretch_3` forced pyramidal with the noslip pass off at
`<option iterations="200">` (both engines converged; MuJoCo settles in 3-4) as
an independent second reading:

    impratio                    1          10         100
    this fixture, step-1     3.55e-15   2.02e-01      -        before
    this fixture, step-1     3.55e-15   3.55e-15      -        after
    stretch_3, |d qpos|      1.24e-15   1.25e-02   1.38e-02    before

THE NON-VACUITY PROOF IS STRUCTURAL, NOT A TOLERANCE

A parity number at `impratio=10` alone cannot tell "we regularize correctly"
from "this fixture does not care about `impratio`" — and the second is the
likely one, because the whole defect hid for exactly that reason. So the file
asserts three things, and the middle one is the one that keeps it honest:

  1. ours matches MuJoCo at `impratio = 10`;
  2. MuJoCo's OWN answer moves a lot between `impratio` 1 and 10, so the
     fixture is genuinely sensitive to the quantity under test;
  3. OUR answer moves by the same amount MuJoCo's does — the response to
     `impratio`, not just the endpoint, is the reference's.

(2) is what a no-op fixture fails. (3) is what a fixture that is sensitive for
some OTHER reason fails.

TOLERANCE. Both sides are float64 running the same arithmetic from the same
state, so the budget is round-off, not a solver allowance. Set from the values
the test itself prints, with headroom.

Run with:
    pixi run mojo run -I . tests/physics3d/test_impratio_pyramidal_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.fields import Model, Data, Dims
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.gpu.constants import MODEL_META_IDX_IMPRATIO
from max.gpu.host import DeviceContext
from mojo_rl.physics3d.fields.spec_fields import SpecFields
from mojo_rl.physics3d.model.model_dims import ModelDims


comptime DTYPE = DType.float64

# ⚠ THE TWO XMLs DIFFER IN ONE ATTRIBUTE. `impratio` reaches the engine as a
# RUNTIME model-meta value but the XML is a comptime parameter of
# `ModelDefFromXML`, so varying it means two model defs. Everything else —
# geometry, friction, solver, timestep, cone — is character-identical, which is
# what makes the deltas below attributable.
comptime BOX_XML_IR1 = """
<mujoco model="slambox_ir1">
  <option timestep="0.002" gravity="0 0 -9.81" cone="pyramidal"
          impratio="1" solver="Newton" iterations="100"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1" condim="3"
          friction="1 .005 .0001"/>
    <body name="box" pos="0 0 .3">
      <joint type="free" name="root"/>
      <geom name="g1" type="box" size=".15 .1 .05" condim="3"
            friction=".8 .05 .05"/>
    </body>
  </worldbody>
</mujoco>
"""

comptime BOX_XML_IR10 = """
<mujoco model="slambox_ir10">
  <option timestep="0.002" gravity="0 0 -9.81" cone="pyramidal"
          impratio="10" solver="Newton" iterations="100"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1" condim="3"
          friction="1 .005 .0001"/>
    <body name="box" pos="0 0 .3">
      <joint type="free" name="root"/>
      <geom name="g1" type="box" size=".15 .1 .05" condim="3"
            friction=".8 .05 .05"/>
    </body>
  </worldbody>
</mujoco>
"""

comptime p1 = parse_xml(BOX_XML_IR1)
comptime p10 = parse_xml(BOX_XML_IR10)

comptime M1 = ModelDefFromXML[
    xml=BOX_XML_IR1,
    nbody=p1.NBODY, njoint=p1.NJOINT, nq=p1.NQ, nv=p1.NV,
    ngeom=p1.NGEOM, nact=p1.NACT, ntex=p1.NTEX, nmat=p1.NMAT,
    nlight=p1.NLIGHT, ncam=p1.NCAM, nsite=p1.NSITE,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=32,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=p1.TIMESTEP,
    max_condim=p1.MAX_CONDIM,
    noslip_iter=0,
]

comptime M10 = ModelDefFromXML[
    xml=BOX_XML_IR10,
    nbody=p10.NBODY, njoint=p10.NJOINT, nq=p10.NQ, nv=p10.NV,
    ngeom=p10.NGEOM, nact=p10.NACT, ntex=p10.NTEX, nmat=p10.NMAT,
    nlight=p10.NLIGHT, ncam=p10.NCAM, nsite=p10.NSITE,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=32,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=p10.TIMESTEP,
    max_condim=p10.MAX_CONDIM,
    noslip_iter=0,
]

comptime N_SETTLE: Int = 300
comptime N_STEPS: Int = 20

# The seeded velocity. As in `test_noslip_elliptic_vs_mujoco`, the ingredient
# that makes a friction quantity first-order is a HARD NORMAL IMPULSE WHILE
# SLIDING — `impratio` scales the pyramid edge stiffness, and on a gently
# resting box the edges are nowhere near saturated and the whole option is
# inert. `SEED_VZ` is that impulse; `SEED_VX`/`SEED_VY` keep both tangential
# directions live and away from the pyramid apex; `SEED_WZ` makes the four
# corner contacts' slip directions disagree so their edges are genuinely
# coupled rather than four copies of one problem.
comptime SEED_VX: Float64 = 3.0
comptime SEED_VY: Float64 = 1.0
comptime SEED_VZ: Float64 = -40.0
comptime SEED_WZ: Float64 = 3.0

# (2) above: MuJoCo's own answer must move at least this much between
# `impratio` 1 and 10, or the fixture is not testing `impratio`.
comptime MIN_IMPRATIO_EFFECT: Float64 = 1e-3
# (1) and (3): round-off budgets, both reported by the test.
comptime TOL_STEP1: Float64 = 1e-10
comptime TOL_V: Float64 = 1e-6
comptime TOL_Q: Float64 = 1e-7
# (3) is a RELATIVE agreement on the response, since the response itself is a
# large number.
comptime TOL_RESPONSE_REL: Float64 = 1e-6


def _mj(xml: String) raises -> PythonObject:
    """MuJoCo from the same XML string the engine was built from."""
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(xml)
    return Python.tuple(mujoco, m, mujoco.MjData(m))


def _settled_qpos() raises -> PythonObject:
    """Drop the box and let it come to rest; return that `qpos`.

    ⚠ SETTLED ONCE AND SHARED by every configuration below. Settling each
    variant with its own `impratio` would let 300 steps of divergence be read
    as the effect of the option — the confound that
    `test_noslip_elliptic_vs_mujoco` documents having fallen into. Settling
    also avoids the degenerate all-zero free-joint quaternion a fresh `Data`
    carries.
    """
    var h = _mj(materialize[BOX_XML_IR1]())
    h[0].mj_resetData(h[1], h[2])
    for _ in range(N_SETTLE):
        h[0].mj_step(h[1], h[2])
    return h[2].qpos.copy()


def _load(md: PythonObject, q0: PythonObject) raises:
    """Put MuJoCo's `Data` at the shared settled pose with the seeded slam."""
    md.qpos[:] = q0
    md.qvel[:] = 0
    md.qvel[0] = SEED_VX
    md.qvel[1] = SEED_VY
    md.qvel[2] = SEED_VZ
    md.qvel[5] = SEED_WZ


def test_impratio_reaches_the_model() raises:
    """`<option impratio>` arrives as model META, and the cone is pyramidal.

    A default silently substituted for `impratio` is exactly the failure this
    file is about, and it would leave every other assertion green: at
    `impratio = 1` the regularization is the identity, so a build that dropped
    the attribute would agree with the `impratio=1` MuJoCo and disagree only
    with the `impratio=10` one — which is what the old code did.
    """
    print("--- impratio: the option reaches the model ---")
    var ctx = DeviceContext()

    comptime MD1_3 = Dims[
        nq=M1.NQ, nv=M1.NV, nbody=M1.NBODY, njoint=M1.NJOINT,
        ngeom=M1.NGEOM, nsite=M1.NSITE, max_contacts=M1.MAX_CONTACTS,
        nequality=M1.MAX_EQUALITY, ntendon=M1.MAX_TENDON,
        nexclude=M1.NEXCLUDE, nmesh_verts=0, npair=M1.NPAIR,
        nact=M1.NACT, nten=M1.NTEN_F, nkey=M1.NKEY,
    ]
    comptime MD10_3 = Dims[
        nq=M10.NQ, nv=M10.NV, nbody=M10.NBODY, njoint=M10.NJOINT,
        ngeom=M10.NGEOM, nsite=M10.NSITE, max_contacts=M10.MAX_CONTACTS,
        nequality=M10.MAX_EQUALITY, ntendon=M10.MAX_TENDON,
        nexclude=M10.NEXCLUDE, nmesh_verts=0, npair=M10.NPAIR,
        nact=M10.NACT, nten=M10.NTEN_F, nkey=M10.NKEY,
    ]

    var mf1 = Model[DTYPE, MD1_3]()
    M1.init_fields[DTYPE](ctx, mf1)
    var mf10 = Model[DTYPE, MD10_3]()
    M10.init_fields[DTYPE](ctx, mf10)

    var ir1 = Float64(mf1.meta.data[MODEL_META_IDX_IMPRATIO])
    var ir10 = Float64(mf10.meta.data[MODEL_META_IDX_IMPRATIO])
    print("  cone =", M1.CONE_TYPE, " impratio(1) =", ir1,
          " impratio(10) =", ir10)
    assert_true(
        Bool(M1.CONE_TYPE == ConeType.PYRAMIDAL)
        and Bool(M10.CONE_TYPE == ConeType.PYRAMIDAL),
        String(
            "the fixture must be PYRAMIDAL — the elliptic builder regularizes"
            " correctly and would pass this file without exercising the"
            " defect; got "
        )
        + String(M1.CONE_TYPE),
    )
    assert_true(
        abs(ir1 - 1.0) < 1e-12 and abs(ir10 - 10.0) < 1e-12,
        String(
            "<option impratio> did not reach model META: expected 1 and 10,"
            " got "
        )
        + String(ir1)
        + " and "
        + String(ir10),
    )
    print("  PASS")


def _run[
    MD: ModelDefLike
](xml: String, mut worst_q: Float64, mut worst_v: Float64,
  mut first_v: Float64, mut contact_steps: Int,
  mut our_v: List[Float64], mut mj_v: List[Float64]) raises:
    """Step ours and MuJoCo together from the shared settled state.

    Fills `our_v`/`mj_v` with the final `qvel` of each so the CALLER can form
    the impratio RESPONSE (assertions 2 and 3) — the deltas cannot be taken
    inside, because they are between two different model defs.
    """
    var h = _mj(xml)
    var mujoco = h[0]
    var m = h[1]
    var md = h[2]
    var q0 = _settled_qpos()
    _load(md, q0)

    comptime MD_3 = Dims[
        nq=MD.NQ, nv=MD.NV, nbody=MD.NBODY, njoint=MD.NJOINT,
        ngeom=MD.NGEOM, nsite=MD.NSITE, max_contacts=MD.MAX_CONTACTS,
        nequality=MD.MAX_EQUALITY, ntendon=MD.MAX_TENDON,
        nexclude=MD.NEXCLUDE, nmesh_verts=0, npair=MD.NPAIR,
        nact=MD.NACT, nten=MD.NTEN_F, nkey=MD.NKEY,
    ]
    var ctx = DeviceContext()
    var sf = SpecFields[DTYPE, ModelDims[MD]]()
    MD.init_spec_fields[DTYPE](ctx, sf)
    var mf = Model[DTYPE, MD_3]()
    MD.init_fields[DTYPE](ctx, mf)
    var d = Data[DTYPE, MD_3, 1]()
    MD.reset_data[DTYPE](sf, d)

    var sq = md.qpos.flatten().tolist()
    var sv = md.qvel.flatten().tolist()
    for i in range(MD.NQ):
        d.qpos.data[i] = Scalar[DTYPE](Float64(py=sq[i]))
    for i in range(MD.NV):
        d.qvel.data[i] = Scalar[DTYPE](Float64(py=sv[i]))
    forward_kinematics["cpu"](d, mf)

    var integ = EulerIntegrator[
        DTYPE, MD_3, MD.CONE_TYPE, 1, SOLVER="newton",
        MAX_CONDIM=MD.MAX_CONDIM, NOSLIP_ITER=MD.NOSLIP_ITER
    ]()

    worst_q = 0.0
    worst_v = 0.0
    first_v = 0.0
    contact_steps = 0
    for _s in range(N_STEPS):
        for i in range(MD.NV):
            d.qfrc.data[i] = Scalar[DTYPE](0)
        integ.step["cpu"](d, mf)
        mujoco.mj_step(m, md)
        if Int(py=md.ncon) > 0:
            contact_steps += 1
        var mq = md.qpos.flatten().tolist()
        var mv = md.qvel.flatten().tolist()
        for i in range(MD.NQ):
            var e = abs(Float64(d.qpos.data[i]) - Float64(py=mq[i]))
            if e > worst_q:
                worst_q = e
        for i in range(MD.NV):
            var e = abs(Float64(d.qvel.data[i]) - Float64(py=mv[i]))
            if e > worst_v:
                worst_v = e
            if _s == 0 and e > first_v:
                first_v = e
    var fv = md.qvel.flatten().tolist()
    our_v = List[Float64]()
    mj_v = List[Float64]()
    for i in range(MD.NV):
        our_v.append(Float64(d.qvel.data[i]))
        mj_v.append(Float64(py=fv[i]))


def test_pyramidal_impratio_matches_mujoco() raises:
    """The three assertions of the module docstring, in one run.

    ⚠ ASSERTION 2 IS THE LOAD-BEARING ONE. `impratio` was wrong for as long as
    it was because every gate in the tree used its default, where the
    regularization is the identity. A fixture that is insensitive to `impratio`
    reproduces that blind spot exactly while printing a green parity number, so
    the reference's OWN sensitivity is measured here and asserted.
    """
    print("--- impratio on the PYRAMIDAL cone: ours vs MuJoCo ---")

    var q1 = 0.0
    var v1 = 0.0
    var f1 = 0.0
    var c1 = 0
    var ours1 = List[Float64]()
    var mjs1 = List[Float64]()
    _run[M1](
        materialize[BOX_XML_IR1](), q1, v1, f1, c1, ours1, mjs1
    )

    var q10 = 0.0
    var v10 = 0.0
    var f10 = 0.0
    var c10 = 0
    var ours10 = List[Float64]()
    var mjs10 = List[Float64]()
    _run[M10](
        materialize[BOX_XML_IR10](), q10, v10, f10, c10, ours10, mjs10
    )

    print("  impratio=1   contact steps", c1, "/", N_STEPS,
          "  step-1 |d qvel|", f1, "  worst |d qvel|", v1,
          "  worst |d qpos|", q1)
    print("  impratio=10  contact steps", c10, "/", N_STEPS,
          "  step-1 |d qvel|", f10, "  worst |d qvel|", v10,
          "  worst |d qpos|", q10)

    # The fixture has to be in contact or nothing here means anything.
    assert_true(
        c1 > 0 and c10 > 0,
        String(
            "VACUOUS: the box never touched the floor — no pyramidal rows"
            " were built, so `impratio` could not have been read. contact"
            " steps "
        )
        + String(c1)
        + " and "
        + String(c10),
    )

    # (2) MuJoCo's own response to impratio, and ours.
    var mj_response = 0.0
    var our_response = 0.0
    for i in range(len(mjs1)):
        var e = abs(mjs10[i] - mjs1[i])
        if e > mj_response:
            mj_response = e
        var o = abs(ours10[i] - ours1[i])
        if o > our_response:
            our_response = o
    print("  MuJoCo's own |qvel(ir=10) - qvel(ir=1)| :", mj_response)
    print("  ours           |qvel(ir=10) - qvel(ir=1)| :", our_response)
    assert_true(
        mj_response > MIN_IMPRATIO_EFFECT,
        String(
            "VACUOUS FIXTURE: MuJoCo's own answer barely moves between"
            " impratio 1 and 10 ("
        )
        + String(mj_response)
        + "), so this state does not test `impratio` and a green parity"
        " number below would prove nothing. Harden the slam (SEED_VZ) or the"
        " slide until the reference itself responds.",
    )

    # (1) ours matches MuJoCo at impratio = 10 — the regularized branch.
    assert_true(
        f10 < TOL_STEP1,
        String("impratio=10 step-1 |d qvel| ") + String(f10)
        + " exceeds " + String(TOL_STEP1)
        + " — READ THIS ONE FIRST: nothing has been amplified yet at step 1,"
        " so a systematic error in `Rpy` cannot hide under it.",
    )
    assert_true(
        v10 < TOL_V and q10 < TOL_Q,
        String("impratio=10 worst |d qvel| ") + String(v10)
        + " |d qpos| " + String(q10) + " exceed "
        + String(TOL_V) + " / " + String(TOL_Q),
    )
    # And at impratio = 1, where the regularization is the identity — this leg
    # was green before the fix and must stay green after it.
    assert_true(
        f1 < TOL_STEP1 and v1 < TOL_V and q1 < TOL_Q,
        String("impratio=1 (the identity case) regressed: step-1 ")
        + String(f1) + " worst v " + String(v1) + " worst q " + String(q1),
    )

    # (3) the RESPONSE is the reference's, not just the endpoint.
    var rel = abs(our_response - mj_response) / mj_response
    print("  relative disagreement in the response :", rel)
    assert_true(
        rel < TOL_RESPONSE_REL,
        String("our response to impratio differs from MuJoCo's by ")
        + String(rel)
        + " relative — the endpoint may match for an unrelated reason while"
        " the SENSITIVITY to `impratio` is still wrong.",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

"""AUD-40 — the ELLIPTIC line search runs MuJoCo's bracket, and this counts it.

    pixi run mojo run -I . tests/physics3d/test_elliptic_linesearch_evals_vs_mujoco.mojo

`PrimalSearch` (engine_solver.c:1692) has three phases: one Newton step on the
line, a one-sided Newton pursuit until the derivative changes sign, then a
BRACKETED search over three candidates — a Newton step off each bracket end
plus the midpoint — accepting the cheapest of those under `gtol`. The
pyramidal helper carried all three; the elliptic per-env leg BISECTED in phase
3 (AUD-40), because bisection needs only a derivative and the elliptic
evaluator had no cost function.

⚠⚠ WHY THIS FILE COUNTS EVALUATIONS INSTEAD OF COMPARING `qacc`. The two
algorithms converge to the same root of the same derivative and stop at the
same tolerance, so on every elliptic model in this tree they land on the same
`alpha` to ~1e-16 and every existing gate passes either way:

    test_noslip_elliptic_vs_mujoco   worst |d qpos|  9.282614408974432e-07  bisection
                                                     9.282614412478574e-07  bracket

That is the 10th significant digit. A gate over the ANSWER is blind here. What
differs is the WORK, and the line search is the solver's hottest inner loop:

    the same 168 searches, same 49/89/30 phase split
      bisection   1207 evaluations   mean 7.18
      bracket      701 evaluations   mean 4.17        <- 42% fewer

and on this file's own 12-step fixture, against the reference itself:

      MuJoCo        70      bracket   70 (equal)      bisection   106

So the engine publishes its count on `d.meta[META_IDX_LS_EVAL]` — MuJoCo's
`sum_i d->solver[i].neval` — and this file gates it against MuJoCo's own.

⚠ THE FIXTURE IS THE SLAMCHAIN FROM `test_noslip_elliptic_vs_mujoco`, chosen
there by measurement: a three-capsule chain SLAMMED into the floor at 40 m/s
while sliding. A gently resting contact does not bracket at all — every search
converges in phase 1 or 2 and phase 3 never runs, which is exactly what
`test_elliptic_condim46_vs_mujoco` does (0 of its searches reach phase 3, so
it is blind to this change). The hard normal impulse against a saturated
friction cone is what makes the line piecewise enough to need a bracket.

⚠ BOUNDS ARE SET FROM MEASUREMENT, and they are RATIOS to MuJoCo's own count
rather than absolute numbers, so the file does not go red on an unrelated
change to the contact count.
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from noeira.physics3d.parser import parse_xml, ModelDefFromXML
from noeira.physics3d.model.model_def import ModelDefLike
from noeira.physics3d.model.model_dims import ModelDims
from noeira.physics3d.types import ConeType
from noeira.physics3d.fields import Model, Data, Dims
from noeira.physics3d.fields.spec_fields import SpecFields
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.integrator.euler import EulerIntegrator
from noeira.physics3d.gpu.constants import (
    META_IDX_LS_EVAL, META_IDX_NEWTON_ITER, META_IDX_SOLVER_ACC_ITER,
    META_IDX_SOLVER_ACC_LSEV, META_IDX_SOLVER_ACC_NCON,
    META_IDX_SOLVER_ACC_CAPPED, META_IDX_NUM_CONTACTS,
    MODEL_META_IDX_SOLVER_ITERATIONS,
)

comptime DTYPE = DType.float64
comptime N_SETTLE = 600
comptime N_STEPS = 12

comptime CHAIN_XML = """
<mujoco model="slamchain ls">
  <option timestep="0.002" gravity="0 0 -9.81" cone="elliptic"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1" condim="3"
          friction="1 .005 .0001"/>
    <body name="l1" pos="0 0 .2">
      <joint type="free" name="root"/>
      <geom name="g1" type="capsule" fromto="0 0 0 .3 0 0" size=".05"
            condim="3" friction=".7 .05 .05"/>
      <body name="l2" pos=".3 0 0">
        <joint type="hinge" name="j2" axis="0 1 0" range="-60 60"
               limited="true" frictionloss="0.05"/>
        <geom name="g2" type="capsule" fromto="0 0 0 .3 0 0" size=".05"
              condim="3" friction=".7 .05 .05"/>
        <body name="l3" pos=".3 0 0">
          <joint type="hinge" name="j3" axis="0 1 0" range="-60 60"
                 limited="true" frictionloss="0.05"/>
          <geom name="g3" type="capsule" fromto="0 0 0 .3 0 0" size=".05"
                condim="3" friction=".7 .05 .05"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

comptime cp = parse_xml(CHAIN_XML)
comptime CM = ModelDefFromXML[
    xml=CHAIN_XML,
    nbody=cp.NBODY, njoint=cp.NJOINT, nq=cp.NQ, nv=cp.NV,
    ngeom=cp.NGEOM, nact=cp.NACT, ntex=cp.NTEX, nmat=cp.NMAT,
    nlight=cp.NLIGHT, ncam=cp.NCAM, nsite=cp.NSITE,
    max_tendon=cp.NTENDON,
    cone_type=ConeType.ELLIPTIC,
    max_contacts=16,
    max_condim=cp.MAX_CONDIM,
    obs_dim_override=1, obs_qpos_skip=0, timestep=cp.TIMESTEP,
]

comptime CMD = ModelDims[CM]


def _mj_slammed() raises -> Tuple[PythonObject, PythonObject]:
    """MuJoCo settled on the floor, then given the slam + slide velocity."""
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(PythonObject(String(CHAIN_XML)))
    var d = mujoco.MjData(m)
    for _ in range(N_SETTLE):
        mujoco.mj_step(m, d)
    for i in range(CM.NV):
        d.qvel[i] = 0.0
    d.qvel[0] = 6.0     # slide
    d.qvel[2] = -40.0   # slam
    d.qvel[4] = 3.0
    mujoco.mj_forward(m, d)
    return (m^, d^)


def _run() raises -> Tuple[Int, Int, Float64, Int, Int, Int]:
    """Step ours and MuJoCo together from the same slammed state.

    Returns (our line-search evaluations, MuJoCo's, worst |d qvel|, contacting
    steps, our Newton iterations, MuJoCo's `solver_niter`).

    Also gates the four running sums the solver keeps on `d.meta` against
    the per-step words it publishes, here rather than in a test of their
    own: this is the one fixture that steps the elliptic leg with the
    reference beside it.
    """
    var mujoco = Python.import_module("mujoco")
    var pair = _mj_slammed()
    var m = pair[0]
    var md = pair[1]

    var ctx = DeviceContext()
    var sf = SpecFields[DTYPE, CMD]()
    CM.init_spec_fields[DTYPE](ctx, sf)
    var mf = Model[DTYPE, CMD]()
    CM.init_fields[DTYPE](ctx, mf)
    var d = Data[DTYPE, CMD, 1]()
    CM.reset_data[DTYPE](sf, d)

    var sq = md.qpos.flatten().tolist()
    var sv = md.qvel.flatten().tolist()
    for i in range(CM.NQ):
        d.qpos.data[i] = Scalar[DTYPE](Float64(py=sq[i]))
    for i in range(CM.NV):
        d.qvel.data[i] = Scalar[DTYPE](Float64(py=sv[i]))
    forward_kinematics["cpu"](d, mf)

    var integ = EulerIntegrator[
        DTYPE, CMD, CM.CONE_TYPE, 1, SOLVER="newton",
        MAX_CONDIM=CM.MAX_CONDIM,
    ]()

    var ours = 0
    var theirs = 0
    var worst_v = 0.0
    var contact_steps = 0
    var our_iters = 0
    var their_iters = 0
    var ncon_sum = 0
    var capped = 0
    # The solver's cap: the model's `<option iterations>`, which the parser
    # writes as MuJoCo's default 100 when the XML (this one) sets none.
    var cap = Int(Float64(mf.meta.data[MODEL_META_IDX_SOLVER_ITERATIONS]))
    assert_true(
        cap > 0,
        "the model carries no solver iteration cap — `fields_build` writes"
        " MuJoCo's default there; the `capped` word below would be untestable",
    )
    for _s in range(N_STEPS):
        for i in range(CM.NV):
            d.qfrc.data[i] = Scalar[DTYPE](0)
        integ.step["cpu"](d, mf)
        mujoco.mj_step(m, md)

        var lsev = Int(Float64(d.meta.data[META_IDX_LS_EVAL]))
        var iters = Int(Float64(d.meta.data[META_IDX_NEWTON_ITER]))
        ours += lsev
        our_iters += iters
        # ⚠ THE COUNT THE SOLVE WAS HANDED, which on this fixture (16 <
        # cap) is the collision count itself.
        ncon_sum += Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS]))
        if iters > 0 and iters >= cap:
            capped += 1
        var niter = Int(py=md.solver_niter[0])
        their_iters += niter
        for k in range(niter):
            theirs += Int(py=md.solver[k].neval)
        if Int(py=md.ncon) > 0:
            contact_steps += 1

        var mv = md.qvel.flatten().tolist()
        for i in range(CM.NV):
            var e = abs(Float64(d.qvel.data[i]) - Float64(py=mv[i]))
            if e > worst_v:
                worst_v = e
    # The running sums are the per-step words added up — exactly, the
    # words are small integers in a float64 `meta` here.
    var acc_it = Int(Float64(d.meta.data[META_IDX_SOLVER_ACC_ITER]))
    var acc_ls = Int(Float64(d.meta.data[META_IDX_SOLVER_ACC_LSEV]))
    var acc_nc = Int(Float64(d.meta.data[META_IDX_SOLVER_ACC_NCON]))
    var acc_cap = Int(Float64(d.meta.data[META_IDX_SOLVER_ACC_CAPPED]))
    assert_true(
        acc_it == our_iters and acc_ls == ours and acc_nc == ncon_sum
        and acc_cap == capped,
        "the solver's running sums on `meta` (iters " + String(acc_it)
        + ", ls " + String(acc_ls) + ", ncon " + String(acc_nc) + ", capped "
        + String(acc_cap) + ") are not the per-step words added up ("
        + String(our_iters) + ", " + String(ours) + ", " + String(ncon_sum)
        + ", " + String(capped) + ") — `_publish_solver_counters` is not"
        " accumulating what it publishes",
    )
    return (ours, theirs, worst_v, contact_steps, our_iters, their_iters)


def test_the_fixture_is_in_contact_and_brackets() raises:
    """⚠ RUN FIRST, AND BOTH HALVES MATTER.

    A rollout with no contacts has no cone rows, so the line is a plain
    quadratic and EVERY search converges on its first Newton step — the count
    below would be exactly two per solve and this file would be measuring
    nothing. And MuJoCo's own count must exceed that floor too, or the
    reference is not bracketing either and there is nothing to be faithful
    to.
    """
    print("=== the slammed chain is in contact, and the search works ===")
    var mujoco = Python.import_module("mujoco")
    print("  mujoco", String(mujoco.__version__))
    var r = _run()
    print("  contacting steps:", r[3], "/", N_STEPS)
    print("  MuJoCo line-search evaluations:", r[1])
    assert_true(
        r[3] == N_STEPS,
        "only " + String(r[3]) + " of " + String(N_STEPS) + " steps are in"
        " contact — without a saturated friction cone the line search never"
        " brackets and this file measures nothing",
    )
    assert_true(
        r[1] > 2 * N_STEPS,
        "MuJoCo spent only " + String(r[1]) + " line-search evaluations over "
        + String(N_STEPS) + " steps, i.e. about the two-per-solve floor. Its"
        " own search is not bracketing here, so ours has nothing to match",
    )


def test_our_line_search_costs_what_mujocos_costs() raises:
    """The measurement AUD-40 is about, and it came out EXACT.

    Measured on this fixture, 12 steps:

        MuJoCo                                     70 evaluations
        ours, three-candidate bracket              70          <- equal
        ours, the bisection this replaced         106          1.51x

    So the assertion is EQUALITY, not a ratio: running MuJoCo's algorithm on
    MuJoCo's line means making MuJoCo's decisions, and every one of them is a
    decision about whether to evaluate again. A bound of "within 2x" would
    have passed the bisection too.

    ⚠ EQUALITY IS THE RIGHT BOUND HERE AND WOULD NOT BE FOR `qacc`. The count
    is an INTEGER produced by a sequence of branches on `|deriv| < gtol`; it
    is either the same sequence or a different one. The float it is derived
    from is not being compared.
    """
    print("=== our line-search evaluations vs MuJoCo's ===")
    var r = _run()
    var ours = r[0]
    var theirs = r[1]
    var ratio = Float64(ours) / Float64(theirs)
    print("  ours:", ours, " MuJoCo:", theirs, " ratio:", ratio)
    assert_true(
        ours > 0,
        "our count is 0 — `META_IDX_LS_EVAL` is not being written. It is"
        " published by the ELLIPTIC per-env leg only; a pyramidal model"
        " reads an unwritten slot",
    )
    assert_true(
        ours == theirs,
        "our line search spent " + String(ours) + " evaluations where MuJoCo"
        " spent " + String(theirs) + " (ratio " + String(ratio) + "). The"
        " bisection this replaced measured 106 against the same 70. A"
        " difference here means the two searches made different decisions,"
        " not that one is slower",
    )


def test_the_answer_did_not_pay_for_the_speed() raises:
    """Fewer evaluations must not mean a looser answer.

    The line search is allowed to stop early only on `|deriv| < gtol`; a
    bracket that returned a stale endpoint would show up here as a qvel
    divergence, not as a count.
    """
    print("=== the answer still tracks MuJoCo ===")
    var r = _run()
    print("  worst |d qvel| over", N_STEPS, "steps =", r[2])
    assert_true(
        r[2] <= 1e-4,
        "worst |d qvel| is " + String(r[2]) + " — this is a 40 m/s impact so"
        " the rollout amplifies, but a line search returning stale endpoints"
        " diverges much faster than round-off",
    )


def test_our_iteration_count_is_mujocos() raises:
    """`META_IDX_NEWTON_ITER` against MuJoCo's `solver_niter`, summed over
    the rollout. The evaluation count above is EQUAL to MuJoCo's, and it is
    summed per iteration, so the iteration counts cannot differ without the
    evaluation counts differing too — this is the same fact read off the
    word §13.53's measurement depends on, and it is what makes that word
    trustworthy on the LIBERO box.
    """
    print("=== our Newton iterations vs MuJoCo's solver_niter ===")
    var r = _run()
    print("  ours:", r[4], " MuJoCo:", r[5])
    assert_true(
        r[4] > 0,
        "our iteration count is 0 over a contacting rollout —"
        " `META_IDX_NEWTON_ITER` is not being written",
    )
    assert_true(
        r[4] == r[5],
        "our solves ran " + String(r[4]) + " Newton iterations where MuJoCo"
        " ran " + String(r[5]) + " over the same " + String(N_STEPS)
        + " steps — the counter or the loop's exit tests moved",
    )


def main() raises:
    var suite = TestSuite()
    suite.test[test_the_fixture_is_in_contact_and_brackets]()
    suite.test[test_our_line_search_costs_what_mujocos_costs]()
    suite.test[test_our_iteration_count_is_mujocos]()
    suite.test[test_the_answer_did_not_pay_for_the_speed]()
    suite^.run()

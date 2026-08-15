"""`mj_solNoSlip` on the ELLIPTIC cone vs MuJoCo — a 3-capsule chain slammed
into the floor while sliding.

The pass is a friction-only Gauss-Seidel sweep run AFTER the primal solve with
the normal forces held fixed. The elliptic branch is a different algorithm from
the pyramidal one — one normal row plus `dim-1` tangential rows per contact,
and a QCQP over the friction ellipsoid rather than a closed-form 1-D minimum.
See `mojo_rl/physics3d/solver/noslip.mojo`.

⚠⚠ HOW THE FIXTURE WAS CHOSEN, INCLUDING THE MEASUREMENT THAT WAS WRONG

Every number below uses a SHARED-STATE protocol: settle ONCE, copy that exact
`qpos` into both models, seed the same `qvel`, then toggle only
`noslip_iterations`. The first version of this file did not — it settled each
model separately with its own setting, so 400 steps of divergence were being
attributed to the pass. That reported 3.2e-1 for a plain sliding chain. The
real number for that state is **1.4e-8**: the pass is inert there, and a gate
built on it would have proved nothing while looking convincing. See
`feedback_flag_toggle_attribution_is_confounded`.

Re-measured honestly, `max|d(qacc)|` on step 1 with `noslip_iterations` 5 -> 0:

    single ball, sliding + spinning        2.3e-12   inert
    single box on a plane, sliding         5.4e-11   inert
    two-box stack, both sliding            9.9e-08   inert
    3-capsule chain, sliding               1.4e-08   inert
    single box, SLAMMED at -40 m/s         3.1e+01   bites
    3-capsule chain, SLAMMED at -40 m/s    9.8e+01   BITES        <- this file

So the ingredient is not contact COUNT, it is a hard normal impulse while
sliding: that is what saturates the friction cone hard enough for the primal
optimum and noslip's (which drops the `R` regulariser — `flg_subR`) to differ.
On a gently resting contact the two agree to round-off, which is why every
small fixture is inert and why `test_noslip_vs_mujoco` (pyramidal) had to
admit the same thing about its own ball.

The chain rather than the single box because it also loads the sweep's FIRST
loop — the dry-friction dof rows, `nf=2` here — and puts joint-limit rows in
the `improvement` sum without the sweep touching them. Both are code paths the
box would leave dark. It also bites 3x harder.

For scale, the model this exists for: `reach_site_features` at 55 contacts
moves by 7.4e+2 of qacc on step 1, against a `|qacc|` of 1.7e+4 — 4.2%.

`test_noslip_is_first_order_here` re-asserts the 9.8e+1 in-process, so if the
fixture ever stops discriminating this file fails instead of passing quietly.

THE NON-VACUITY PROOF IS STRUCTURAL, NOT A TOLERANCE

Two model defs are built from the SAME XML differing only in `noslip_iter`, and
the test asserts BOTH directions:

  * with the pass, ours matches MuJoCo;
  * WITHOUT it, ours does not.

A parity number alone cannot tell "the sweep is right" from "the sweep never
ran" — see `feedback_confirm_the_code_under_test_actually_runs`, where a green
gate was measuring code that was not in the path. The second assertion is what
rules that out, and it is why the fixture had to be one where the pass is
first-order in the first place.

WHAT THE STATE IS FOR

noslip only ever moves FRICTION rows, so the seeded `qvel` has to load them:

  * `qvel[2] = -40` — the slam. This is the one that matters (see above);
    without it every other component together leaves the pass inert.
  * `qvel[0] = 3.0`, `qvel[1] = 1.0` — sliding, so both tangential rows of
    every contact are live and away from the cone apex.
  * `qvel[5] = 3.0` — yaw, so the six contacts' slip directions DISAGREE and
    their friction forces are genuinely coupled rather than six copies of one
    problem.
  * `qvel[6] = 4.0`, `qvel[7] = -4.0` — the hinges moving, which loads their
    `frictionloss` rows.

⚠ `noslip_tolerance="0"` is dm_control's manipulation setting and is carried
here so the fixture matches what the real models declare. It is NOT
load-bearing for this gate: measured against MuJoCo's 1e-6 default on this
state, on the plain sliding state, on `reach_site_features`, and at 5/20/50
iterations, the largest difference found was 8.9e-10. The attribute is parsed
for fidelity, not because a divergence was observed — the honest statement,
after an earlier draft of this file claimed 4.2e-2 from the confounded
protocol above.

TOLERANCE. Both sides are float64 running the same arithmetic from the same
state, so the budget is round-off, not a solver allowance. Set from the value
the test itself prints, with headroom.

Run with:
    pixi run mojo run -I . tests/physics3d/test_noslip_elliptic_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.fields import Model, Data
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.gpu.constants import MODEL_META_IDX_NOSLIP_TOLERANCE
from max.gpu.host import DeviceContext


comptime DTYPE = DType.float64

comptime CHAIN_XML = """
<mujoco model="slamchain">
  <option timestep="0.002" gravity="0 0 -9.81" cone="elliptic"
          noslip_iterations="5" noslip_tolerance="0"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1" condim="3"
          friction="1 .005 .0001"/>
    <body name="l1" pos="0 0 .3">
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

comptime pc = parse_xml(CHAIN_XML)

comptime M = ModelDefFromXML[
    xml=CHAIN_XML,
    nbody=pc.NBODY, njoint=pc.NJOINT, nq=pc.NQ, nv=pc.NV,
    ngeom=pc.NGEOM, nact=pc.NACT, ntex=pc.NTEX, nmat=pc.NMAT,
    nlight=pc.NLIGHT, ncam=pc.NCAM, nsite=pc.NSITE,
    cone_type=ConeType.ELLIPTIC,
    max_contacts=32,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=pc.TIMESTEP,
    max_condim=pc.MAX_CONDIM,
    noslip_iter=pc.NOSLIP_ITER,
]

# ⚠ IDENTICAL TO `M` EXCEPT `noslip_iter=0`. This is the control: it is what
# the engine did before the pass existed on the elliptic path, and the test
# requires it to DISAGREE with MuJoCo. Without it, a parity number cannot
# distinguish a correct sweep from one that never ran.
comptime M_OFF = ModelDefFromXML[
    xml=CHAIN_XML,
    nbody=pc.NBODY, njoint=pc.NJOINT, nq=pc.NQ, nv=pc.NV,
    ngeom=pc.NGEOM, nact=pc.NACT, ntex=pc.NTEX, nmat=pc.NMAT,
    nlight=pc.NLIGHT, ncam=pc.NCAM, nsite=pc.NSITE,
    cone_type=ConeType.ELLIPTIC,
    max_contacts=32,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=pc.TIMESTEP,
    max_condim=pc.MAX_CONDIM,
    noslip_iter=0,
]

comptime N_SETTLE: Int = 400
comptime N_STEPS: Int = 30

# The seeded velocity. `SEED_VZ` is the slam and is the reason this fixture
# discriminates at all — see the module docstring.
comptime SEED_VX: Float64 = 3.0
comptime SEED_VY: Float64 = 1.0
comptime SEED_VZ: Float64 = -40.0
comptime SEED_WZ: Float64 = 3.0
comptime SEED_J2: Float64 = 4.0
comptime SEED_J3: Float64 = -4.0

# Measured MuJoCo-vs-MuJoCo on this state, shared-state protocol:
#   worst |d(qacc)| = 9.79e+1, |d(qvel)| = 4.33e-1 over 30 steps.
# The gate below asserts the pass is worth at least this much, with room.
comptime MIN_NOSLIP_EFFECT: Float64 = 1.0
# `M_OFF` must miss MuJoCo by at least this. It misses by ~4e-1 of qvel if the
# pass is genuinely absent, so this is two orders of margin below that.
comptime MIN_CONTROL_MISS: Float64 = 1e-3
# Ours-vs-MuJoCo budgets with the pass on, both reported by the test.
#
# STEP 1 is the real gate: measured 2.5e-13 of qvel, i.e. round-off on a state
# where `|qacc|` is ~3e+3. Nothing has been amplified yet at that point, so a
# systematic error in the sweep cannot hide under it — the OFF control misses
# by 1.96e-1 at the same step, twelve orders away.
comptime TOL_STEP1: Float64 = 1e-11
# WORST-OVER-30 is looser ON PURPOSE and is not a solver allowance: this state
# is a 40 m/s impact, so the step-1 round-off is amplified over the rollout.
# Measured 1.1e-8. It is kept as a second assertion only to catch a divergence
# that starts small and grows — the message on it says which number to read
# first.
comptime TOL_V: Float64 = 1e-7
comptime TOL_Q: Float64 = 1e-8


def _mj(noslip: Int = -1, tol: Float64 = -1.0) raises -> PythonObject:
    """MuJoCo from the same XML.

    `noslip >= 0` overrides the iteration count; `tol >= 0` overrides
    `noslip_tolerance`. Both exist to measure MuJoCo against itself.
    """
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[CHAIN_XML]())
    if noslip >= 0:
        m.opt.noslip_iterations = noslip
    if tol >= 0.0:
        m.opt.noslip_tolerance = tol
    return Python.tuple(mujoco, m, mujoco.MjData(m))


def _settled_qpos(mujoco: PythonObject) raises -> PythonObject:
    """Drop the chain and let it come to rest; return that `qpos`.

    ⚠ SETTLED ONCE AND SHARED, never re-settled per configuration. Settling
    each variant separately is what made the first version of this file report
    a 3.2e-1 effect that was really 1.4e-8 — 400 steps of chaotic divergence
    read as the thing under test. Every comparison in this file starts from
    THIS array.

    Settling rather than starting from a fresh `Data` also avoids the
    degenerate all-zero free-joint quaternion a zeroed `Data` carries, and puts
    all three capsules on the floor so the contact set is the one the fixture
    is about.
    """
    var h = _mj()
    mujoco.mj_resetData(h[1], h[2])
    for _ in range(N_SETTLE):
        mujoco.mj_step(h[1], h[2])
    return h[2].qpos.copy()


def _load(md: PythonObject, q0: PythonObject) raises:
    """Put MuJoCo's `Data` at the shared settled pose with the seeded slam."""
    md.qpos[:] = q0
    md.qvel[:] = 0
    md.qvel[0] = SEED_VX
    md.qvel[1] = SEED_VY
    md.qvel[2] = SEED_VZ
    md.qvel[5] = SEED_WZ
    md.qvel[6] = SEED_J2
    md.qvel[7] = SEED_J3


def test_noslip_option_reaches_the_model() raises:
    """`noslip_iterations` and `noslip_tolerance` both arrive, on both paths.

    `noslip_iterations` comes from the COMPTIME counter (`parse_xml`) and
    `noslip_tolerance` from the RUNTIME parser via model META — two different
    parsers, so agreeing that one option exists proves nothing about the other.
    Both are checked because a default silently substituted for either changes
    what runs: 0 iterations removes the pass entirely.
    """
    print("--- elliptic noslip: the options are parsed ---")
    print("  NOSLIP_ITER =", M.NOSLIP_ITER, " CONE =", M.CONE_TYPE,
          " MAX_CONDIM =", M.MAX_CONDIM)
    assert_true(
        M.NOSLIP_ITER == 5,
        "parse_xml did not pick up noslip_iterations=5 — the pass would be"
        " compiled out and every comparison below would be no-op vs no-op",
    )
    assert_true(
        M.CONE_TYPE == ConeType.ELLIPTIC,
        "this model def is not on the ELLIPTIC path, so it exercises the"
        " pyramidal branch that another file already gates",
    )
    assert_true(
        M_OFF.NOSLIP_ITER == 0,
        "the control model def has the pass ON, so it is not a control",
    )

    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0, M.NPAIR,
    ]()
    M.init_fields[DTYPE, 0](ctx, mf)
    var tol = Float64(mf.meta.data[MODEL_META_IDX_NOSLIP_TOLERANCE])
    print("  noslip_tolerance in META =", tol, " (XML says 0)")
    # ⚠ The check is `== 0.0`, and 0 is the VALUE, not "unset". A consumer that
    # read a 0 here as absent and substituted MuJoCo's 1e-6 default would be
    # running a different model from the one declared, even though no fixture
    # available today can tell the two apart (largest measured gap 8.9e-10).
    assert_true(
        tol == 0.0,
        "<option noslip_tolerance=\"0\"> did not reach model META — the solver"
        " is running MuJoCo's 1e-6 default instead of the value the model"
        " declares",
    )


def test_noslip_is_first_order_here() raises:
    """MuJoCo vs MuJoCo from the SHARED state — they must DISAGREE.

    This is the fixture's licence to exist. It is a measurement of MuJoCo, not
    of us, and it says the pass changes the answer here at first order — so the
    parity test below can actually fail. The pyramidal file's equivalent
    asserts the opposite and says so.

    ⚠ Both legs start from `_settled_qpos()`, ONE array. Re-settling per leg is
    what produced this file's original, wrong 3.2e-1.
    """
    print("--- elliptic noslip: scope check (a fact about MuJoCo) ---")
    var np = Python.import_module("numpy")
    var mujoco = Python.import_module("mujoco")
    var q0 = _settled_qpos(mujoco)

    var on = _mj(5)
    var off = _mj(0)
    mujoco.mj_resetData(on[1], on[2])
    mujoco.mj_resetData(off[1], off[2])
    _load(on[2], q0)
    _load(off[2], q0)
    mujoco.mj_step(on[1], on[2])
    mujoco.mj_step(off[1], off[2])

    var da = Float64(py=np.abs(np.subtract(on[2].qacc, off[2].qacc)).max())
    print("  ncon =", on[2].ncon, " nefc =", on[2].nefc, " nf =", on[2].nf)
    print("  max |d(qacc)| between noslip 5 and 0 =", da, " (measured 9.8e+1)")
    assert_true(
        da > MIN_NOSLIP_EFFECT,
        "noslip barely changes MuJoCo's answer on this state — the fixture has"
        " stopped discriminating, so the parity test below would pass whether"
        " or not the sweep runs and must not be trusted. Check the seeded slam"
        " (qvel[2]) first: without it this state is inert at 1.4e-8",
    )
    assert_true(
        Int(py=on[2].nf) > 0,
        "no dry-friction rows are live, so the sweep's FIRST loop is never"
        " entered and only the contact block is being gated",
    )


def _rollout[
    MD: ModelDefLike
](mujoco: PythonObject, m: PythonObject, md: PythonObject) raises -> Tuple[
    Float64, Float64, Int, Float64
]:
    """Step `MD` and MuJoCo together from MuJoCo's current state.

    Returns `(worst |d qpos|, worst |d qvel|, contacting steps, |d qvel| after
    step 1)`. Both sides start from the SAME state so the comparison is about
    the solve, not about how each side reached the pose.

    ⚠ THE STEP-1 NUMBER IS THE DIAGNOSTIC ONE. Everything after it is a
    divergent rollout — this state is a 40 m/s impact, so any difference is
    amplified step over step and the worst-over-30 number cannot distinguish
    "round-off growing" from "wrong by a little". Step 1 has nothing to
    amplify, so a systematic error in the sweep shows up there at full size.
    """
    var sf = MD.make_spec_fields[DTYPE]()
    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, MD.NV, MD.NBODY, MD.NJOINT, MD.NGEOM, MD.MAX_EQUALITY,
        MD.MAX_TENDON, MD.NSITE, MD.NEXCLUDE, 0, MD.NPAIR,
    ]()
    MD.init_fields[DTYPE, 0](ctx, mf)
    var d = Data[DTYPE, MD.NQ, MD.NV, MD.NBODY, MD.MAX_CONTACTS, MD.NSITE, 1]()
    MD.reset_data[DTYPE](sf, d)

    var sq = md.qpos.flatten().tolist()
    var sv = md.qvel.flatten().tolist()
    for i in range(MD.NQ):
        d.qpos.data[i] = Scalar[DTYPE](Float64(py=sq[i]))
    for i in range(MD.NV):
        d.qvel.data[i] = Scalar[DTYPE](Float64(py=sv[i]))
    forward_kinematics["cpu"](d, mf)

    var integ = EulerIntegrator[
        DTYPE, MD.NQ, MD.NV, MD.NBODY, MD.NJOINT, MD.MAX_CONTACTS, MD.NGEOM,
        MD.MAX_EQUALITY, MD.MAX_TENDON, MD.NSITE, MD.NEXCLUDE, 0,
        MD.CONE_TYPE, 1, SOLVER="newton",
        MAX_CONDIM=MD.MAX_CONDIM, NOSLIP_ITER=MD.NOSLIP_ITER,
        NPAIR=MD.NPAIR,
    ]()

    var worst_q = 0.0
    var worst_v = 0.0
    var contact_steps = 0
    var first_v = 0.0
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
    return (worst_q, worst_v, contact_steps, first_v)


def test_noslip_elliptic_matches_mujoco() raises:
    """Ours against MuJoCo with the pass ON — and, as a control, with it OFF.

    Both legs run the identical rollout from the identical shared settled
    state against the identical MuJoCo (`noslip_iterations=5` both times). The
    ONLY difference is our own `NOSLIP_ITER`. The ON leg must match, the OFF
    leg must not.
    """
    print("--- elliptic noslip: ours vs MuJoCo ---")
    var mujoco = Python.import_module("mujoco")
    var q0 = _settled_qpos(mujoco)

    var on = _mj()
    mujoco.mj_resetData(on[1], on[2])
    _load(on[2], q0)
    var r_on = _rollout[M](mujoco, on[1], on[2])
    print("  ON : contact on", r_on[2], "/", N_STEPS, "steps")
    print("  ON : step-1 |d(qvel)| =", r_on[3], "   <- the diagnostic one")
    print("  ON : worst |d(qpos)| =", r_on[0], "  worst |d(qvel)| =", r_on[1])

    var off = _mj()
    mujoco.mj_resetData(off[1], off[2])
    _load(off[2], q0)
    var r_off = _rollout[M_OFF](mujoco, off[1], off[2])
    print("  OFF: step-1 |d(qvel)| =", r_off[3])
    print("  OFF: worst |d(qpos)| =", r_off[0],
          "  worst |d(qvel)| =", r_off[1])

    assert_true(
        r_on[2] > N_STEPS - 5,
        "the chain leaves the floor during the rollout, so most steps gate the"
        " smooth dynamics rather than the friction sweep this file is about",
    )
    assert_true(
        r_on[3] < TOL_STEP1,
        "our elliptic solve disagrees with MuJoCo on the FIRST step, where"
        " nothing has been amplified yet — the sweep is running but computing"
        " something other than mj_solNoSlip's elliptic branch",
    )
    assert_true(
        r_on[1] < TOL_V and r_on[0] < TOL_Q,
        "our elliptic rollout diverges from MuJoCo with noslip ON over 30"
        " steps. If the step-1 assertion above passed, this is round-off"
        " amplified by a 40 m/s impact rather than a wrong sweep — check the"
        " printed step-1 number before hunting in the solver",
    )
    # ⚠ THE ASSERTION THAT MAKES THE ONE ABOVE MEAN SOMETHING. Identical code
    # and identical state, `noslip_iter=0`: if this ALSO matched, the pass
    # would not be in the path and the gate above would be measuring nothing.
    assert_true(
        r_off[1] > MIN_CONTROL_MISS,
        "the rollout matches MuJoCo just as well WITHOUT the noslip pass, so"
        " the pass is not in the code path being measured and the parity"
        " assertion above proves nothing — stash the wiring and re-run before"
        " believing either number",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

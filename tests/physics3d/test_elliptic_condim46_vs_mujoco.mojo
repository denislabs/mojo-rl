"""The ELLIPTIC cone at condim 4 and 6 vs MuJoCo — a spinning, sliding ball.

MuJoCo's elliptic contact of dimension `dim` is ONE normal row followed by
`dim-1` tangential ones, paired with `con->friction[t]`:

    t = 0, 1   SLIDE along the two tangents          condim 3
    t = 2      TORSION about the contact normal      condim 4
    t = 3, 4   ROLLING about the two tangents        condim 6

Our elliptic solver carried exactly two of them until 2026-08-13 — `Jt1_c` and
`Jt2_c`, one isotropic `mu`, one shared `D_f` — so a `condim="4"` or
`condim="6"` geom under `cone="elliptic"` silently lost its torsional and
rolling rows. `MAX_CONDIM` was already threaded all the way to the friction
builder and consumed only by the PYRAMIDAL branch. See
`mojo_rl/physics3d/solver/elliptic_cone.mojo`.

WHY THIS MATTERS RATHER THAN BEING A COMPLETENESS ITEM. Every dm_control
manipulation model declares `cone="elliptic"`, and Jaco's hand pads are
`condim="4"`: `reach_site_features` has 3 condim-4 contacts of 55 at qpos0.
This file's fixture is a single sphere because that isolates the rows —
one contact, so `nefc` IS the contact dimension and nothing else can explain
a difference.

WHAT THE STATE IS FOR

The torsional row only carries force when there is angular velocity ABOUT THE
CONTACT NORMAL, and the rolling rows only when there is angular velocity about
the tangents. A resting ball gates nothing. So:

  * `qvel[5] = 15` — spin about z, i.e. about the contact normal. This is what
    the torsional row resists, and it is what makes condim 4 differ from 3.
  * `qvel[0] = 1.5`, `qvel[1] = 0.5` — sliding, so both SLIDE rows are live and
    away from the cone apex; without it the cone is degenerate and the
    torsional row would be the only friction in the system.
  * `qvel[2] = -5` — a downward impulse. It keeps the ball on the floor for all
    30 steps (a spinning ball with no `vz` is launched by friction and only
    contacts 14) and it is ALSO what makes the noslip sweep bite here, so the
    new `mju_QCQP3` / `mju_QCQP` dispatches are exercised rather than merely
    compiled.

`friction=".8 .3 .05"` against the floor's `"1 .005 .0001"` mixes (MuJoCo takes
the max per direction) to `slide 1, torsion 0.3, roll 0.05`. The three
coefficients are deliberately far apart: `R[j]*friction[j]^2` is constant in
MuJoCo, so the torsional row is `(1/0.3)^2 = 11x` stiffer than the slide rows
and the rolling rows `(1/0.05)^2 = 400x`. A port that shared one `D` across
the tangential rows — which is what ours did — gets those ratios wrong by
exactly those factors.

MEASURED, MuJoCo AGAINST ITSELF, from the shared settled pose (step 1,
`max|d(qacc)|` against a `|qacc|` of ~3e+3):

    condim 4 vs condim 3     1.58e+03     the TORSIONAL row
    condim 6 vs condim 4     1.13e+03     the ROLLING rows
    noslip 5 vs 0 at condim 4   2.98e+01   exercises mju_QCQP3
    noslip 5 vs 0 at condim 6   3.65e+02   exercises mju_QCQP

⚠ SHARED-STATE PROTOCOL, as in `test_noslip_elliptic_vs_mujoco`. The pose is
settled ONCE and copied into every arm; nothing is re-settled per
configuration. Settling each variant under its own setting measures the
divergence of two independent warm-ups and attributes it to the toggle — that
mistake reported a 3.2e-1 effect that was really 1.4e-8 in the noslip file. See
`feedback_ab_arms_must_share_the_warmup_state`.

THE NON-VACUITY PROOF IS STRUCTURAL, NOT A TOLERANCE

Each leg builds TWO model defs from the SAME XML differing only in
`max_condim`, and asserts both directions:

  * at the model's own condim, ours matches MuJoCo;
  * CLAMPED one step down, ours does not.

The clamped arm is exactly what the engine did before this change — the
producer clamps each contact's `condim` to `MAX_CONDIM` — so it is not a
synthetic control, it is the old behaviour. A parity number alone cannot tell
"the torsional row is right" from "the fixture does not need one"; see
`feedback_confirm_the_code_under_test_actually_runs`.

The condim-6 leg's control is `max_condim=4`, NOT 3. Controlling against 3
would pass if only the torsional row worked, since the total miss would still
be large. Against 4 the ONLY difference is the two rolling rows.

TOLERANCE. Both sides are float64 running the same arithmetic from the same
state, so the budget is round-off, not a solver allowance. Step 1 is the
diagnostic number — nothing has been amplified yet — and worst-over-30 is a
secondary check on a rollout that is genuinely divergent.

Run with:
    pixi run mojo run -I . tests/physics3d/test_elliptic_condim46_vs_mujoco.mojo
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
from max.gpu.host import DeviceContext


comptime DTYPE = DType.float64

# ⚠ THE TWO XMLs DIFFER IN ONE CHARACTER — the ball's `condim`. Everything
# else, including the floor, the option block and the friction triple, is
# identical, so a difference between the two legs cannot come from anywhere
# else. They are separate literals rather than one template because a comptime
# model def needs a literal XML (`feedback_comptime_string_store_use_runtime_xml`).
comptime XML4 = """
<mujoco model="spinner4">
  <option timestep="0.002" gravity="0 0 -9.81" cone="elliptic"
          noslip_iterations="5" noslip_tolerance="0"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1" condim="3"
          friction="1 .005 .0001"/>
    <body name="ball" pos="0 0 .12">
      <joint type="free" name="root"/>
      <geom name="g" type="sphere" size=".1" condim="4"
            friction=".8 .3 .05"/>
    </body>
  </worldbody>
</mujoco>
"""

comptime XML6 = """
<mujoco model="spinner6">
  <option timestep="0.002" gravity="0 0 -9.81" cone="elliptic"
          noslip_iterations="5" noslip_tolerance="0"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1" condim="3"
          friction="1 .005 .0001"/>
    <body name="ball" pos="0 0 .12">
      <joint type="free" name="root"/>
      <geom name="g" type="sphere" size=".1" condim="6"
            friction=".8 .3 .05"/>
    </body>
  </worldbody>
</mujoco>
"""

comptime p4 = parse_xml(XML4)
comptime p6 = parse_xml(XML6)

comptime M4 = ModelDefFromXML[
    xml=XML4,
    nbody=p4.NBODY, njoint=p4.NJOINT, nq=p4.NQ, nv=p4.NV,
    ngeom=p4.NGEOM, nact=p4.NACT, ntex=p4.NTEX, nmat=p4.NMAT,
    nlight=p4.NLIGHT, ncam=p4.NCAM, nsite=p4.NSITE,
    cone_type=ConeType.ELLIPTIC,
    max_contacts=16,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=p4.TIMESTEP,
    max_condim=p4.MAX_CONDIM,
    noslip_iter=p4.NOSLIP_ITER,
]

# The control for the condim-4 leg: identical XML, `max_condim` CLAMPED to 3.
# `_precompute_contact_friction` clamps each contact's own `condim` to
# `MAX_CONDIM`, so this contact keeps its two slide rows and loses its
# torsional one — which is precisely what every elliptic model got before this
# change, not a synthetic degradation.
comptime M4_FLAT = ModelDefFromXML[
    xml=XML4,
    nbody=p4.NBODY, njoint=p4.NJOINT, nq=p4.NQ, nv=p4.NV,
    ngeom=p4.NGEOM, nact=p4.NACT, ntex=p4.NTEX, nmat=p4.NMAT,
    nlight=p4.NLIGHT, ncam=p4.NCAM, nsite=p4.NSITE,
    cone_type=ConeType.ELLIPTIC,
    max_contacts=16,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=p4.TIMESTEP,
    max_condim=3,
    noslip_iter=p4.NOSLIP_ITER,
]

comptime M6 = ModelDefFromXML[
    xml=XML6,
    nbody=p6.NBODY, njoint=p6.NJOINT, nq=p6.NQ, nv=p6.NV,
    ngeom=p6.NGEOM, nact=p6.NACT, ntex=p6.NTEX, nmat=p6.NMAT,
    nlight=p6.NLIGHT, ncam=p6.NCAM, nsite=p6.NSITE,
    cone_type=ConeType.ELLIPTIC,
    max_contacts=16,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=p6.TIMESTEP,
    max_condim=p6.MAX_CONDIM,
    noslip_iter=p6.NOSLIP_ITER,
]

# ⚠ CLAMPED TO 4, NOT 3. The condim-6 leg is about the two ROLLING rows, so its
# control must already have the torsional one. Against `max_condim=3` this
# would pass on the strength of the torsional row alone.
comptime M6_FLAT = ModelDefFromXML[
    xml=XML6,
    nbody=p6.NBODY, njoint=p6.NJOINT, nq=p6.NQ, nv=p6.NV,
    ngeom=p6.NGEOM, nact=p6.NACT, ntex=p6.NTEX, nmat=p6.NMAT,
    nlight=p6.NLIGHT, ncam=p6.NCAM, nsite=p6.NSITE,
    cone_type=ConeType.ELLIPTIC,
    max_contacts=16,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=p6.TIMESTEP,
    max_condim=4,
    noslip_iter=p6.NOSLIP_ITER,
]

comptime N_SETTLE: Int = 300
comptime N_STEPS: Int = 30

comptime SEED_VX: Float64 = 1.5
comptime SEED_VY: Float64 = 0.5
comptime SEED_VZ: Float64 = -5.0
comptime SEED_WZ: Float64 = 15.0

# MuJoCo against itself on this state: 1.58e+3 for the torsional row, 1.13e+3
# for the rolling pair. Asserted with three orders of room, so this fails when
# the fixture stops discriminating rather than when it merely drifts.
comptime MIN_CONDIM_EFFECT: Float64 = 1.0
# A clamped arm misses by ~1e+1 of qvel on step 1 if the rows are genuinely
# absent; this is orders below that.
comptime MIN_CONTROL_MISS: Float64 = 1e-3
# Ours-vs-MuJoCo with the rows present. STEP 1 is the gate — see the module
# docstring. Both numbers are printed by the test.
comptime TOL_STEP1: Float64 = 1e-10
comptime TOL_V: Float64 = 1e-6
comptime TOL_Q: Float64 = 1e-7


def _mj[XML: StaticString](condim: Int = -1) raises -> PythonObject:
    """MuJoCo from `XML`; `condim >= 0` overrides the BALL's contact dimension.

    The override is on `geom_condim[1]` (geom 0 is the floor) rather than on
    the XML text, so the two arms of the MuJoCo-vs-MuJoCo comparison come from
    one parse and cannot differ in anything else.
    """
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[XML]())
    if condim >= 0:
        m.geom_condim[1] = condim
    return Python.tuple(mujoco, m, mujoco.MjData(m))


def _settled_qpos(mujoco: PythonObject) raises -> PythonObject:
    """Drop the ball and let it rest; return that `qpos`, ONCE, shared.

    Settled from the condim-4 model, but nothing about the settle depends on
    condim: the ball falls straight down with no spin and no slide, so the
    torsional and rolling rows carry no force on the way. Every arm below —
    including both condim-6 ones — starts from THIS array.
    """
    var h = _mj[XML4]()
    mujoco.mj_resetData(h[1], h[2])
    for _ in range(N_SETTLE):
        mujoco.mj_step(h[1], h[2])
    return h[2].qpos.copy()


def _load(md: PythonObject, q0: PythonObject) raises:
    """Shared settled pose + the spin/slide/impact that loads the rows."""
    md.qpos[:] = q0
    md.qvel[:] = 0
    md.qvel[0] = SEED_VX
    md.qvel[1] = SEED_VY
    md.qvel[2] = SEED_VZ
    md.qvel[5] = SEED_WZ


def test_condim_reaches_the_model_defs() raises:
    """The four model defs carry the `MAX_CONDIM` this file assumes.

    `MAX_CONDIM` comes from `parse_xml`'s COMPTIME scan of the XML text, which
    is a different parser from the runtime one that fills the geom records. If
    the scan missed the `condim="6"` the treatment arm would silently be a
    control and both legs would compare a clamped solver against a clamped
    solver.
    """
    print("--- elliptic condim 4/6: the model defs ---")
    print("  M4  MAX_CONDIM =", M4.MAX_CONDIM,
          "  M4_FLAT =", M4_FLAT.MAX_CONDIM)
    print("  M6  MAX_CONDIM =", M6.MAX_CONDIM,
          "  M6_FLAT =", M6_FLAT.MAX_CONDIM)
    print("  NOSLIP_ITER =", M4.NOSLIP_ITER, " CONE =", M4.CONE_TYPE)
    assert_true(
        M4.MAX_CONDIM == 4,
        "parse_xml did not see condim=\"4\" — the treatment arm is clamped to"
        " the same 3 as its control and this leg proves nothing",
    )
    assert_true(
        M6.MAX_CONDIM == 6,
        "parse_xml did not see condim=\"6\" — the rolling rows are absent from"
        " the treatment arm too",
    )
    assert_true(
        M4_FLAT.MAX_CONDIM == 3 and M6_FLAT.MAX_CONDIM == 4,
        "a control arm is not clamped, so it is not a control",
    )
    assert_true(
        M4.CONE_TYPE == ConeType.ELLIPTIC and M4.NOSLIP_ITER == 5,
        "this fixture is not on the ELLIPTIC path with the noslip pass on, so"
        " it does not exercise what it claims to",
    )


def test_condim_rows_are_first_order_here() raises:
    """MuJoCo vs MuJoCo — the torsional and rolling rows must CHANGE the answer.

    This is the fixture's licence to exist, and it is a measurement of MuJoCo,
    not of us. Both arms come from one parse with only `geom_condim` moved, and
    both start from the ONE shared settled pose.
    """
    print("--- elliptic condim 4/6: scope check (a fact about MuJoCo) ---")
    var np = Python.import_module("numpy")
    var mujoco = Python.import_module("mujoco")
    var q0 = _settled_qpos(mujoco)

    var h3 = _mj[XML6](3)
    var h4 = _mj[XML6](4)
    var h6 = _mj[XML6](6)
    mujoco.mj_resetData(h3[1], h3[2])
    mujoco.mj_resetData(h4[1], h4[2])
    mujoco.mj_resetData(h6[1], h6[2])
    _load(h3[2], q0)
    _load(h4[2], q0)
    _load(h6[2], q0)
    mujoco.mj_step(h3[1], h3[2])
    mujoco.mj_step(h4[1], h4[2])
    mujoco.mj_step(h6[1], h6[2])

    print("  nefc: condim3 =", h3[2].nefc, " condim4 =", h4[2].nefc,
          " condim6 =", h6[2].nefc, " (ncon =", h6[2].ncon, ")")
    var d43 = Float64(py=np.abs(np.subtract(h4[2].qacc, h3[2].qacc)).max())
    var d64 = Float64(py=np.abs(np.subtract(h6[2].qacc, h4[2].qacc)).max())
    print("  max |d(qacc)| condim 4 vs 3 =", d43, " (measured 1.58e+3)")
    print("  max |d(qacc)| condim 6 vs 4 =", d64, " (measured 1.13e+3)")

    # One contact, so `nefc` IS the contact dimension. If this ever stops
    # holding, the two "row" numbers above are measuring something else.
    assert_true(
        Int(py=h6[2].ncon) == 1
        and Int(py=h3[2].nefc) == 3
        and Int(py=h4[2].nefc) == 4
        and Int(py=h6[2].nefc) == 6,
        "the fixture is no longer one contact whose nefc equals its condim, so"
        " a difference below cannot be attributed to the tangential rows",
    )
    assert_true(
        d43 > MIN_CONDIM_EFFECT,
        "the TORSIONAL row barely changes MuJoCo's answer on this state — the"
        " condim-4 leg would pass whether or not we build that row. Check the"
        " spin (qvel[5]) first: torsion is about the contact NORMAL and a"
        " ball that is not spinning loads it with nothing",
    )
    assert_true(
        d64 > MIN_CONDIM_EFFECT,
        "the ROLLING rows barely change MuJoCo's answer, so the condim-6 leg"
        " is not gating them",
    )


def _rollout[
    MD: ModelDefLike
](mujoco: PythonObject, m: PythonObject, md: PythonObject) raises -> Tuple[
    Float64, Float64, Int, Float64
]:
    """Step `MD` and MuJoCo together from MuJoCo's current state.

    Returns `(worst |d qpos|, worst |d qvel|, contacting steps, |d qvel| after
    step 1)`.

    ⚠ THE STEP-1 NUMBER IS THE DIAGNOSTIC ONE. Everything after it is a
    divergent rollout of a spinning impact, so the worst-over-30 number cannot
    separate "round-off growing" from "wrong by a little". Step 1 has nothing
    to amplify.
    """
    var sf = MD.make_spec_fields[DTYPE]()
    var ctx = DeviceContext()
    var mf = Model[DTYPE, Dims[nv=MD.NV, nbody=MD.NBODY, njoint=MD.NJOINT, ngeom=MD.NGEOM, nequality=MD.MAX_EQUALITY, ntendon=MD.MAX_TENDON, nsite=MD.NSITE, nexclude=MD.NEXCLUDE, nmesh_verts=0, npair=MD.NPAIR]]()
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


def _leg[
    XML: StaticString, FULL: ModelDefLike, FLAT: ModelDefLike
](label: String) raises:
    """One condim leg: the full-row arm must match MuJoCo, the clamped one must
    not. Both run the identical rollout against the identical MuJoCo (which
    always uses the model's real condim) from the identical shared pose.
    """
    var mujoco = Python.import_module("mujoco")
    var q0 = _settled_qpos(mujoco)

    var full = _mj[XML]()
    mujoco.mj_resetData(full[1], full[2])
    _load(full[2], q0)
    var r_full = _rollout[FULL](mujoco, full[1], full[2])

    var flat = _mj[XML]()
    mujoco.mj_resetData(flat[1], flat[2])
    _load(flat[2], q0)
    var r_flat = _rollout[FLAT](mujoco, flat[1], flat[2])

    print("---", label, "---")
    print("  contact on", r_full[2], "/", N_STEPS, "steps")
    print("  MAX_CONDIM", FULL.MAX_CONDIM, ": step-1 |d(qvel)| =", r_full[3],
          "   <- the diagnostic one")
    print("  MAX_CONDIM", FULL.MAX_CONDIM, ": worst |d(qpos)| =", r_full[0],
          "  worst |d(qvel)| =", r_full[1])
    print("  MAX_CONDIM", FLAT.MAX_CONDIM, " (clamped control): step-1"
          " |d(qvel)| =", r_flat[3], " worst |d(qvel)| =", r_flat[1])

    assert_true(
        r_full[2] > N_STEPS - 3,
        "the ball leaves the floor during the rollout, so most steps gate the"
        " smooth dynamics rather than the friction rows this file is about."
        " The downward seed (qvel[2]) is what keeps it down",
    )
    assert_true(
        r_full[3] < TOL_STEP1,
        "our elliptic solve disagrees with MuJoCo on the FIRST step, where"
        " nothing has been amplified yet. The tangential rows are being built"
        " but are not MuJoCo's — check the per-row `R` first"
        " (`R[j]*friction[j]^2` is constant), since a shared D across rows"
        " reproduces exactly this",
    )
    assert_true(
        r_full[1] < TOL_V and r_full[0] < TOL_Q,
        "our rollout diverges from MuJoCo over 30 steps with the rows present."
        " If the step-1 assertion passed, this is round-off amplified by a"
        " spinning impact rather than a wrong row — read the printed step-1"
        " number before hunting in the solver",
    )
    # ⚠ THE ASSERTION THAT MAKES THE ONE ABOVE MEAN SOMETHING. Identical code,
    # identical state, one fewer tangential row.
    assert_true(
        r_flat[3] > MIN_CONTROL_MISS,
        "the rollout matches MuJoCo just as well with the extra tangential"
        " row(s) CLAMPED AWAY, so those rows are not what the parity above is"
        " measuring and the fixture does not need them. Re-check the scope"
        " numbers printed by test_condim_rows_are_first_order_here",
    )


def test_condim4_torsional_row_matches_mujoco() raises:
    """condim 4: the TORSIONAL row. Control is `max_condim=3`."""
    _leg[XML4, M4, M4_FLAT]("elliptic condim 4 (torsion): ours vs MuJoCo")


def test_condim6_rolling_rows_match_mujoco() raises:
    """condim 6: the ROLLING rows. Control is `max_condim=4`, so the torsional
    row is present in BOTH arms and only the rolling pair is under test."""
    _leg[XML6, M6, M6_FLAT]("elliptic condim 6 (rolling): ours vs MuJoCo")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

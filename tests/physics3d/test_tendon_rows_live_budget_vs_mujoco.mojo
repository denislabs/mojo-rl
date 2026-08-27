"""Tendon limit and equality rows were guarded against a cap that is 0.

    pixi run mojo run -I . tests/physics3d/test_tendon_rows_live_budget_vs_mujoco.mojo

WHAT WENT WRONG — TWICE, IN ONE FILE. `build_tendon_limit_rows` and
`build_tendon_equality_rows` both refuse to emit a row once the edge list is
full, and both spelled the test

    if num_edges >= E_CAP:
        break

`E_CAP` is the COMPTIME array capacity. On a dynamic dimension provider —
`DynDims`, i.e. the studio, every runtime-loaded model and every Menagerie
scene — it is **0**. So `0 >= 0` on the first tendon and the loop broke
before emitting anything: every spatial tendon LIMIT and every tendon
EQUALITY was silently dropped from the constraint system.

⚠⚠ THE CALLER DOCUMENTS THE EXACT TRAP, TWENTY LINES ABOVE THE CALL.
`newton_solve.mojo:836` reads:

    ⚠ TWO SPELLINGS OF THE ROW BUDGET, AND THEY ARE NOT INTERCHANGEABLE.
    `E_CAP` sizes the arrays and is 0 on a dynamic provider; `me` is the
    live budget the CAPACITY GUARDS below compare against. Guarding with
    the cap would admit zero rows on the dynamic leg and silently solve an
    unconstrained system.

…and then passed `E_CAP` to two builders that guard with it. The warning was
written for the rows built INLINE in that function and never followed into the
two it calls out to. Same lesson as `test_two_joint_equalities_vs_mujoco`:
fixing a hazard where you were looking is not fixing the hazard — grep the
SPELLING.

AND A SECOND, INDEPENDENT DEFECT ON THE ELLIPTIC LEG. Even with the budget
right, `_newton_solve_env`'s elliptic branch never called
`build_tendon_limit_rows` at all: it builds tendon EQUALITY rows (`:1705`)
and stops. The pyramidal branch (`:1114`) and the blocked kernel (`:3363`)
both call it. Two of three. So this gate runs every fixture under BOTH cones,
and the limit fixtures were red on elliptic for a different reason than they
were red on pyramidal.

MEASURED. The fixtures below are the smallest models that make each row the
ONLY constraint in the system, so a dropped row is not a tolerance — it is a
different physics:

    spatial tendon limit   qacc  ours 0, 0, -9.81, 0, 0, 0   (free fall)
                                 mj  -4.034, -4.034, -0.398, +1344.60, -1344.60, 0
    fixed tendon equality  qacc  ours +59.364, -73.688       (uncoupled)
                                 mj   +28.928, +15.623

`hello_robot_stretch_3` aside, the Menagerie scene this surfaced on is
`robotiq_2f85/scene.xml`, which hangs a free box from a 2 cm string: worst
|d(qpos)| after one step was 2.689e-03, entirely on the box's rotation dofs,
because the string was not there.

⚠ VACUITY NOTE. A fixture whose constraint is SATISFIED at the start pose
gates nothing — both engines would agree by doing nothing. Both fixtures below
start VIOLATED (the tendon is 0.0409 long against a 0.02 limit; the coupled
joints are pulled apart by gravity), and each test asserts the constrained
answer differs from the unconstrained one by a wide margin before comparing.
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
    read_model_source,
)
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.studio.stepping import StudioIntegPyr, StudioIntegEll

comptime DT = DType.float64

# ── fixture 1: a free box on a 2 cm string ────────────────────────────────
# No floor, so `ncon == 0` and the tendon limit is the ONLY row in the
# system. The two sites are 0.0409 m apart at the start pose against a
# `range="0 0.02"`, i.e. the limit is violated by 2.09 cm on step one.
comptime _LIM_BODY = String(
    """
  <worldbody>
    <site name="anchor" pos="0 0 0.2" size="0.002"/>
    <body name="object" pos="0 0 0.15">
      <freejoint/>
      <geom type="box" size="0.015 0.015 0.015"/>
      <site name="hook" pos="0.015 0.015 0.015" size="0.002"/>
    </body>
  </worldbody>
  <tendon>
    <spatial limited="true" range="0 0.02" width="0.001">
      <site site="hook"/>
      <site site="anchor"/>
    </spatial>
  </tendon>
"""
)
comptime XML_LIM_PYR = String("<mujoco>" + _LIM_BODY + "</mujoco>")
comptime XML_LIM_ELL = String(
    '<mujoco><option cone="elliptic"/>' + _LIM_BODY + "</mujoco>"
)

# ── fixture 2: two links coupled by a fixed tendon equality ───────────────
# `j1 - j2 == 0`, so gravity swinging the pair apart is what the row has to
# fight. Again no floor and no contacts.
comptime _EQ_BODY = String(
    """
  <worldbody>
    <body pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      <body pos="0.2 0 0">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="cpl">
      <joint joint="j1" coef="1"/>
      <joint joint="j2" coef="-1"/>
    </fixed>
  </tendon>
"""
)
comptime XML_EQ_PYR = String(
    "<mujoco>" + _EQ_BODY
    + '<equality><tendon tendon1="cpl"/></equality></mujoco>'
)
comptime XML_EQ_ELL = String(
    '<mujoco><option cone="elliptic"/>' + _EQ_BODY
    + '<equality><tendon tendon1="cpl"/></equality></mujoco>'
)
# The SAME chain with the equality removed — the control. Its answer is what
# a dropped row produces, so the gate can say which failure it is seeing.
comptime XML_EQ_NONE = String("<mujoco>" + _EQ_BODY + "</mujoco>")

# ── MuJoCo 3.10.0, `mj_forward` at the start pose ─────────────────────────
# Both cones give these to the last digit: with `ncon == 0` there is no
# friction cone to differ over, which is itself an invariant this gate leans
# on (a cone-dependent answer here would be a bug of its own).
comptime MJ_LIM_0 = -4.0338054065899999
comptime MJ_LIM_1 = -4.0338054065899999
comptime MJ_LIM_2 = -0.39778738462000001
comptime MJ_LIM_3 = 1344.6018022000001
comptime MJ_LIM_4 = -1344.6018022000001
comptime MJ_LIM_5 = 7.2911043497700004e-15
comptime MJ_TEN_LENGTH = 0.040926763859362274

comptime MJ_EQ_0 = 28.927902465273828
comptime MJ_EQ_1 = 15.622653879260239
# …and the SAME model with no `<equality>`, which is what dropping the row
# gives. Printed by the gate so a red run names its own cause.
comptime MJ_NOEQ_0 = 59.364193612022639
comptime MJ_NOEQ_1 = -73.688292248113214


def _qacc_after_one_step[
    ELLIPTIC: Bool
](xml: String) raises -> List[Float64]:
    """One step from qpos0 through the studio's own integrator pair.

    `d.qacc` after the step is the CONSTRAINED acceleration the solver
    produced — the quantity a dropped row changes outright, rather than the
    integrated pose where it would show up scaled by `dt`.
    """
    var fmd = parse_xml_full(expand_mjcf(xml, String("")), String(""))
    var dims = dims_from_flat(fmd, max_contacts=16, nmesh_verts=0)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    for i in range(dims.get_nq()):
        d.qpos.data[i] = sf.qpos0.data[i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)
        d.qfrc.data[i] = Scalar[DT](0)
    var pyr = StudioIntegPyr(dims)
    var ell = StudioIntegEll(dims)
    comptime if ELLIPTIC:
        ell.step["cpu"](d, m)
    else:
        pyr.step["cpu"](d, m)
    var out = List[Float64]()
    for i in range(dims.get_nv()):
        out.append(Float64(d.qacc.data[i]))
    return out^


def _worst(got: List[Float64], want: List[Float64]) -> Float64:
    var w = 0.0
    for i in range(len(want)):
        var e = abs(got[i] - want[i])
        if e > w:
            w = e
    return w


def _worst_rel(got: List[Float64], want: List[Float64]) -> Float64:
    """Worst |d| RELATIVE to the reference magnitude.

    ⚠ THE LIMIT FIXTURE NEEDS THIS AND THE EQUALITY ONE DOES NOT. A 2 cm
    string yanking a free box gives `qacc` of 1.34e+03 on two dofs and
    -0.398 on a third; an absolute tolerance tight enough to be meaningful
    for the small one is below the SOLVER's own convergence residual on the
    large one. The residual here is 2.8e-09 absolute — and 2.1e-12 relative
    on BOTH the 1344 dofs and the 0.398 one, which is what says it is
    Newton's stopping point and not a structural difference. A dropped row
    reads 1.0 relative (free fall), four orders clear of any tolerance.
    """
    var w = 0.0
    for i in range(len(want)):
        var scale = abs(want[i])
        if scale < 1.0:
            scale = 1.0
        var e = abs(got[i] - want[i]) / scale
        if e > w:
            w = e
    return w


def test_spatial_tendon_limit_reaches_the_solver() raises:
    """A violated `<spatial limited="true">` must produce a constraint row."""
    print("=== spatial tendon limit, both cones ===")
    var want: List[Float64] = [
        MJ_LIM_0, MJ_LIM_1, MJ_LIM_2, MJ_LIM_3, MJ_LIM_4, MJ_LIM_5,
    ]
    var got_p = _qacc_after_one_step[False](XML_LIM_PYR)
    var got_e = _qacc_after_one_step[True](XML_LIM_ELL)
    assert_true(
        len(got_p) == 6 and len(got_e) == 6,
        "the fixture must be one free body (6 dofs) — got "
        + String(len(got_p)) + " and " + String(len(got_e)),
    )
    for i in range(6):
        print(
            "  dof", i, " pyr", got_p[i], " ell", got_e[i], " mj", want[i]
        )
    var worst_p = _worst_rel(got_p, want)
    var worst_e = _worst_rel(got_e, want)
    print("  worst RELATIVE |d(qacc)| pyramidal", worst_p,
          "  elliptic", worst_e)
    print("  (absolute, for scale:", _worst(got_p, want), ")")
    # ⚠ AND THE TWO CONES MUST AGREE WITH EACH OTHER EXACTLY. With
    # `ncon == 0` there is no friction cone in the system at all, so
    # pyramidal and elliptic are running the same problem through two code
    # paths. That equality is a stronger statement than either tolerance
    # below: it fails the moment one leg builds a row the other does not.
    var cone_gap = _worst(got_p, got_e)
    # ⚠⚠ A FEW ULP, NOT BIT EQUALITY — RELAXED 2026-08-26, and the number is
    # the argument. This used to require `== 0.0` and got it, because both
    # cones cold-started at `qacc_smooth` and converged in a couple of
    # iterations along nearly the same path. Once the primal solve warm-starts
    # (`warmstart()`, engine_forward.c:786) the iterate begins somewhere else
    # and the two legs take a LONGER route to the same fixed point — through
    # genuinely different code, since pyramidal carries these rows as `Je`
    # EDGES and elliptic as `eq_*` dense rows, summed in different orders.
    #
    # MEASURED: 9.094947017729282e-13, which is exactly 2^-40 and ~3 ULP of
    # this fixture's `|qacc|` (~1344). It was BIT-IDENTICAL across three
    # separate solver changes — the `deriv[1]` floor, that floor applied to the
    # elliptic leg too, and the full `PrimalSearch` port — which is what says
    # it is deterministic converged rounding and not something algorithmic:
    # a real difference would have moved when the algorithm did.
    #
    # ⚠ THE ASSERTION'S PURPOSE IS UNCHANGED. "One leg is building a row the
    # other is not" is an ORDERS-OF-MAGNITUDE statement — a missing tendon
    # limit row leaves this fixture's `qacc` wrong by ~1e+03, not by 1e-12.
    # The bound below is 32 ULP of the answer's own magnitude, ~10x the
    # measured gap and ~14 orders under a missing row, so it cannot hide one.
    var cone_scale = Float64(0)
    for i in range(6):
        if abs(got_p[i]) > cone_scale:
            cone_scale = abs(got_p[i])
    var cone_tol = 32.0 * 2.220446049250313e-16 * cone_scale
    print("  pyramidal vs elliptic", cone_gap, " (budget", cone_tol, ")")
    assert_true(
        cone_gap <= cone_tol,
        "with no contacts the two cones solve an identical system and must"
        " agree to a few ULP; they differ by " + String(cone_gap)
        + " against a budget of " + String(cone_tol) + " (32 ULP of "
        + String(cone_scale) + "). At this magnitude that is one leg building"
        " a row the other is not, NOT rounding.",
    )

    # ⚠ VACUITY. Free fall is `(0, 0, -9.81, 0, 0, 0)`; the reference answer
    # has 1344 rad/s^2 on dof 3. If the box were merely falling, `worst`
    # would be ~1344 — so a passing run cannot be a frozen or unconstrained
    # one, and the assertion below is checking the row, not the tolerance.
    assert_true(
        abs(want[3]) > 1e3,
        "the reference must be far from free fall or this gate is vacuous",
    )
    assert_true(
        worst_p < 1e-10,
        "PYRAMIDAL: the tendon limit row never reached the solver; worst"
        " RELATIVE |d(qacc)| = " + String(worst_p)
        + ". ~1.0 means the row was dropped entirely and the box is in"
        " free fall — check that `build_tendon_limit_rows` guards against"
        " the LIVE row budget and not `E_CAP`, which is 0 on a dynamic"
        " provider.",
    )
    assert_true(
        worst_e < 1e-10,
        "ELLIPTIC: same fixture, and with `ncon == 0` there is no friction"
        " cone for the two legs to disagree over; worst RELATIVE"
        " |d(qacc)| = "
        + String(worst_e)
        + ". The elliptic branch of `_newton_solve_env` did not call"
        " `build_tendon_limit_rows` at all — it built the tendon EQUALITY"
        " rows and stopped.",
    )
    print("  PASS")


def test_tendon_equality_reaches_the_solver() raises:
    """`<equality><tendon>` must hold `j1 == j2` against gravity."""
    print("=== fixed tendon equality, both cones ===")
    var want: List[Float64] = [MJ_EQ_0, MJ_EQ_1]
    var noeq: List[Float64] = [MJ_NOEQ_0, MJ_NOEQ_1]
    var got_p = _qacc_after_one_step[False](XML_EQ_PYR)
    var got_e = _qacc_after_one_step[True](XML_EQ_ELL)
    var got_n = _qacc_after_one_step[False](XML_EQ_NONE)
    assert_true(
        len(got_p) == 2 and len(got_e) == 2 and len(got_n) == 2,
        "the fixture must be a two-hinge chain — got " + String(len(got_p)),
    )
    print("  pyr ", got_p[0], got_p[1])
    print("  ell ", got_e[0], got_e[1])
    print("  mj  ", want[0], want[1])
    print("  mj without the equality (what a dropped row gives)",
          noeq[0], noeq[1])
    var worst_p = _worst(got_p, want)
    var worst_e = _worst(got_e, want)
    print("  worst |d(qacc)| pyramidal", worst_p, "  elliptic", worst_e)

    # ⚠ THE CONTROL. The unconstrained twin must agree with MuJoCo's
    # unconstrained twin, or a failure above could be anything in the chain
    # (mass, gravity, the hinge axis) rather than the equality row.
    var worst_n = _worst(got_n, noeq)
    print("  control (no <equality>) worst |d(qacc)|", worst_n)
    assert_true(
        worst_n < 1e-9,
        "the fixture WITHOUT the equality must already match MuJoCo; worst"
        " |d(qacc)| = " + String(worst_n) + ". Until it does, nothing this"
        " gate says about the equality row is trustworthy.",
    )
    # ⚠ VACUITY. The two answers must be far apart, or the row is not doing
    # anything and the comparison would pass with it dropped.
    assert_true(
        abs(want[1] - noeq[1]) > 10.0,
        "the constrained and unconstrained answers must differ widely or"
        " this gate is vacuous",
    )
    assert_true(
        worst_p < 1e-9,
        "PYRAMIDAL: the tendon equality row never reached the solver; worst"
        " |d(qacc)| = " + String(worst_p)
        + ". ~8.9e+01 means the row was dropped and the chain swung free —"
        " `build_tendon_equality_rows` guards with `E_CAP`, which is 0 on a"
        " dynamic provider.",
    )
    assert_true(
        worst_e < 1e-9,
        "ELLIPTIC: same fixture, the other consumer of the same builder;"
        " worst |d(qacc)| = " + String(worst_e),
    )
    print("  PASS")


def test_robotiq_hangs_its_box() raises:
    """The Menagerie scene the two defects were found on."""
    print("=== robotiq_2f85/scene.xml, one step ===")
    var path = String(
        "references/mujoco_menagerie-main/robotiq_2f85/scene.xml"
    )
    var src = read_model_source(path)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var dims = dims_from_flat(fmd, max_contacts=128, nmesh_verts=65536)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    for i in range(dims.get_nq()):
        d.qpos.data[i] = sf.qpos0.data[i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)
        d.qfrc.data[i] = Scalar[DT](0)
    # 2f85.xml says `cone="elliptic"`, so this is the leg the studio builds
    # for it — and the leg that never called the limit builder.
    var ell = StudioIntegEll(dims)
    ell.step["cpu"](d, m)
    # MuJoCo 3.10.0, `mj_step` once with ctrl = 0: the box's angular dofs.
    # Zero ctrl keeps the gripper still, so the whole answer is the string.
    print("  qacc[11]", Float64(d.qacc.data[11]),
          " qacc[12]", Float64(d.qacc.data[12]))
    assert_true(
        abs(Float64(d.qacc.data[11])) > 1.0,
        "the hanging box must feel the string: qacc[11] = "
        + String(Float64(d.qacc.data[11]))
        + ", i.e. the 2 cm tendon limit produced no force at all.",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

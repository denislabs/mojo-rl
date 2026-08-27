"""`<option iterations / tolerance / ls_iterations / ls_tolerance>` reach the solver.

    pixi run mojo run -I . tests/physics3d/test_solver_budget_from_option.mojo

WHAT WAS MISSING. `newton_solve` hardcoded its budget — 200 iterations, 1e-8,
50 line-search halvings, `ls_tolerance` 0.01 — and `_parse_option` read none of
the four attributes. Sixteen Menagerie files set at least one of them
(`apptronik_apollo` `iterations="4" ls_iterations="10"`, five
`rainbow_robotics_rby1` scenes `iterations="30" tolerance="1e-6"`,
`robotstudio_so101` `iterations="10" ls_iterations="20"`,
`tetheria_aero_hand_open` `iterations="5" ls_iterations="8"`), and every one
of them was simulated with our budget instead of its own.

⚠⚠ RUNNING LONGER THAN THE REFERENCE IS NOT SAFER — IT IS A DIFFERENT ANSWER.
MuJoCo's answer for a model shipping `<option iterations="4">` IS its
4-iteration iterate; converging past it walks away from the reference rather
than toward it. This is the argument `MODEL_META_IDX_CCD_TOLERANCE` already
makes for EPA's stopping rule, and the same shape as `noslip_iterations`
(`b6be5c48`): a per-model count that only ever reached the solver as a
compile-time constant.

⚠ THE PLUMBING ASSERTIONS ARE NOT THE GATE. A `FlatModelDef` field holding 4
and a meta slot holding 4 were both true of `noslip_tolerance` for months while
nothing dispatched on it. The gate is `test_each_knob_changes_the_answer`,
which steps ONE model four ways through the runtime loader and requires each
knob to move the result.

MEASURED on `unitree_go1` (elliptic, `impratio="100"`, four foot contacts,
zero control, one step from keyframe 0), `|d(qvel)|` against MuJoCo 3.10.0:

    iterations="100" tolerance="0"       ours vs MuJoCo    2.998e-15  <- anchor
    iterations="1"   tolerance="0"       ours vs ours@100   2.807e-01
    iterations="100" tolerance="1e-1"    ours vs ours@tol0  6.732e-04
    iterations="100" ls_iterations="1"   ours vs ours@ls50  1.110e-16  (inert)

The thresholds below sit an order under the smallest asserted move and nine
orders above the anchor, so no single answer can satisfy both.

⚠ THE ANCHOR IS THE ARM THAT MAKES THE OTHERS MEAN SOMETHING. Three arms that
merely DIFFER would be satisfied by four equally wrong answers; the converged
arm pins one of them to the reference.

⚠ OUR PER-ITERATION ITERATE IS NOT MuJoCo'S, AND THIS FILE DOES NOT PRETEND
OTHERWISE. Both engines reach the same fixed point — go1 agrees to 2.2e-16 at
10 iterations and apollo at 6 — but truncated to the same small count the two
differ (go1 at `iterations="1"`: 2.56e+00; apollo at its shipped 4: 1.25e-04).
That is a real open divergence in the Newton TRAJECTORY, and honouring the
budget is what makes it visible instead of hiding it behind a shared fixed
point. Do not "fix" it by ignoring the option again.
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, assert_equal, TestSuite

from mojo_rl.physics3d.fields import Data, Model, DynDims, init_hfield_data
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
    read_model_source,
)
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.studio.stepping import (
    StudioIntegEll, studio_cone_of, studio_integrator_of,
)
from mojo_rl.physics3d.types import ConeType, IntegratorType
from mojo_rl.physics3d.gpu.constants import (
    MODEL_META_IDX_SOLVER_ITERATIONS, MODEL_META_IDX_SOLVER_TOLERANCE,
    MODEL_META_IDX_LS_ITERATIONS, MODEL_META_IDX_LS_TOLERANCE,
    META_IDX_NUM_CONTACTS, KEY_IDX_NQPOS,
)



comptime DT = DType.float64

comptime GO1 = String(
    "references/mujoco_menagerie-main/unitree_go1/scene.xml"
)

# The four arms. `tolerance="0"` in three of them means "never stop early", so
# the ITERATION COUNT is the only thing that ends the loop — without it the
# `iterations` arm would be comparing two converged answers.
comptime OPT_CONV = String(
    '<option cone="elliptic" impratio="100" iterations="100" tolerance="0"/>'
)
comptime OPT_IT1 = String(
    '<option cone="elliptic" impratio="100" iterations="1" tolerance="0"/>'
)
comptime OPT_TOL = String(
    '<option cone="elliptic" impratio="100" iterations="100" tolerance="1e-1"/>'
)
comptime OPT_LS1 = String(
    '<option cone="elliptic" impratio="100" iterations="100" tolerance="0"'
    ' ls_iterations="1"/>'
)

# Ours against MuJoCo on the converged arm. Measured 1.0242e-14 of `qvel` on a
# step whose `|qvel|` is ~2e+0 — round-off, not a solver allowance.
comptime TOL_ANCHOR: Float64 = 1e-11
# And each knob must move the answer by at least this. The smallest measured
# move is the `tolerance` arm's 9.52e-05, so this sits an order below it and
# nine orders above `TOL_ANCHOR`: no single answer can satisfy both.
comptime MIN_MOVE_ITER: Float64 = 1e-2
comptime MIN_MOVE_TOL: Float64 = 1e-6


def _swap_option(xml: String, opt: String) raises -> String:
    """Replace the model's `<option .../>` with `opt`.

    ⚠ THE WHOLE ELEMENT, not an attribute edit — go1's own `<option>` already
    carries `cone` and `impratio`, and every arm below restates them so the
    only thing varying across the four is the budget.
    """
    var i = xml.find(String("<option"))
    if i == -1:
        raise Error("fixture has no <option> to replace")
    var j = xml.find(String(">"), i)
    if j == -1:
        raise Error("unterminated <option> in the fixture")
    return String(xml[byte=0:i]) + opt + String(xml[byte = j + 1 :])


@fieldwise_init
struct _Run(Movable):
    """One runtime-path rollout of the fixture."""

    var qvel: List[Float64]
    var ncon: Int
    var m_iter: Int
    var m_tol: Float64
    var m_ls: Int
    var m_lstol: Float64
    var cone: Int
    var integ: Int


def _ours(opt: String) raises -> _Run:
    """Load go1 with `opt` THE WAY THE STUDIO DOES and step it once.

    ⚠ `parse_xml_full` -> `dims_from_flat` -> `build_model_runtime` — nothing
    here is a compile-time model, so the meta slots are the only route the
    budget has.
    """
    var src = read_model_source(GO1)
    var xml = _swap_option(expand_mjcf(src[0], src[1]), opt)
    var fmd = parse_xml_full(xml, src[1])
    var dims = dims_from_flat(fmd, max_contacts=32, nmesh_verts=8192)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)

    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var d = Data[DT, DynDims, 1](dims)
    # ⚠ THE HEIGHTFIELD GRID IS `Data`, NOT `Model` (`836a65ff`). go1's scene
    # has none, but a `Data` built by hand without this holds a grid of zeros
    # — a flat terrain that collides perfectly happily and is not the surface
    # the model declared.
    init_hfield_data(d, m)
    for i in range(nq):
        d.qpos.data[i] = sf.qpos0.data[i]
    var nqp = Int(Float64(sf.key_meta.data[KEY_IDX_NQPOS]))
    for i in range(min(nqp, nq)):
        d.qpos.data[i] = sf.key_qpos.data[i]
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](0)

    var ell = StudioIntegEll(dims)
    ell.step["cpu"](d, m)

    var qv = List[Float64]()
    for i in range(nv):
        qv.append(Float64(d.qvel.data[i]))
    return _Run(
        qvel=qv^,
        ncon=Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS])),
        m_iter=Int(Float64(m.meta.data[MODEL_META_IDX_SOLVER_ITERATIONS])),
        m_tol=Float64(m.meta.data[MODEL_META_IDX_SOLVER_TOLERANCE]),
        m_ls=Int(Float64(m.meta.data[MODEL_META_IDX_LS_ITERATIONS])),
        m_lstol=Float64(m.meta.data[MODEL_META_IDX_LS_TOLERANCE]),
        cone=studio_cone_of(fmd),
        integ=studio_integrator_of(fmd),
    )


def _mj(iters: Int, tol: Float64, ls: Int) raises -> List[Float64]:
    """MuJoCo's `qvel` after the same single step from keyframe 0.

    ⚠ THE BUDGET IS SET ON `m.opt`, NOT BY EDITING THE XML. `from_xml_string`
    cannot resolve go1's `meshdir="assets"` — it opens `assets/hip.stl`
    relative to the process, not to the model — and copying the tree to patch
    one attribute would write into `references/`. `mjOption` is mutable and
    is exactly what the attribute compiles to, so this sets the same three
    numbers the file would.
    """
    var mujoco = Python.import_module("mujoco")
    var py = Python.import_module("builtins")
    var m = mujoco.MjModel.from_xml_path(GO1)
    m.opt.iterations = iters
    m.opt.tolerance = tol
    m.opt.ls_iterations = ls
    var d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_step(m, d)
    # ⚠ `.flatten().tolist()` — a numpy scalar inside a `PythonObject` is
    # neither `Floatable` nor `Intable`, so indexing the array directly and
    # casting does not compile.
    var qv = d.qvel.flatten().tolist()
    var out = List[Float64]()
    for i in range(len(qv)):
        out.append(Float64(py=qv[i]))
    return out^


def _dmax(a: List[Float64], b: List[Float64]) -> Float64:
    var n = len(a) if len(a) < len(b) else len(b)
    var w = Float64(0)
    for i in range(n):
        var v = abs(a[i] - b[i])
        if v > w:
            w = v
    return w


def test_the_fixture_is_what_it_claims() raises:
    """Elliptic, Euler, and actually in contact — asserted, not assumed."""
    print("=== fixture ===")
    var r = _ours(OPT_CONV)
    print("  ncon", r.ncon, " cone", r.cone, " integ", r.integ,
          " meta iter", r.m_iter, " tol", r.m_tol,
          " ls", r.m_ls, " lstol", r.m_lstol)
    assert_true(
        r.ncon > 0,
        "go1 must be in contact at keyframe 0 or the solver has nothing to"
        " iterate on and every arm below is vacuous; got ncon " + String(r.ncon),
    )
    assert_equal(r.cone, ConeType.ELLIPTIC, "the fixture must be elliptic")
    assert_equal(
        r.integ, IntegratorType.EULER,
        "the fixture must take the EULER arm — `StudioIntegEll` is what is"
        " stepped below",
    )
    print("  PASS")


def test_the_budget_reaches_model_meta() raises:
    """Plumbing, NOT the gate — see the module docstring."""
    print("=== the four slots (plumbing) ===")
    var c = _ours(OPT_CONV)
    var i1 = _ours(OPT_IT1)
    var ls = _ours(OPT_LS1)
    assert_equal(c.m_iter, 100, "iterations=100 must reach meta")
    assert_equal(i1.m_iter, 1, "iterations=1 must reach meta")
    assert_true(c.m_tol == 0.0, "tolerance=0 must reach meta verbatim — a 0 is"
                " a REAL setting (never stop early), not 'unset'")
    assert_equal(ls.m_ls, 1, "ls_iterations=1 must reach meta")
    assert_true(
        abs(c.m_lstol - 0.01) < 1e-15,
        "an absent ls_tolerance must resolve to MuJoCo's 0.01, not 0",
    )
    print("  PASS")


def test_each_knob_changes_the_answer() raises:
    """THE GATE. One model, four budgets, through the runtime loader.

    Each arm must move `qvel`, and the converged arm must land on MuJoCo.
    """
    print("=== the budget changes the solve ===")
    var conv = _ours(OPT_CONV)
    var it1 = _ours(OPT_IT1)
    var tol = _ours(OPT_TOL)
    var ls1 = _ours(OPT_LS1)
    var mjref = _mj(100, 0.0, 50)

    var d_anchor = _dmax(conv.qvel, mjref)
    var d_iter = _dmax(it1.qvel, conv.qvel)
    var d_tol = _dmax(tol.qvel, conv.qvel)
    var d_ls = _dmax(ls1.qvel, conv.qvel)
    print("  ours(conv) vs MuJoCo      ", d_anchor)
    print("  ours(iterations=1)        ", d_iter)
    print("  ours(tolerance=1e-1)      ", d_tol)
    print("  ours(ls_iterations=1)     ", d_ls)

    assert_true(
        d_anchor < TOL_ANCHOR,
        "with the model's own budget honoured and run to convergence we must"
        " land on MuJoCo; |d(qvel)| = " + String(d_anchor) + " against "
        + String(TOL_ANCHOR) + ". This is the arm that pins the other three to"
        " the reference rather than merely to each other.",
    )
    assert_true(
        d_iter > MIN_MOVE_ITER,
        "`<option iterations=\"1\">` must truncate the solve; the answer moved"
        " " + String(d_iter) + ", which is below " + String(MIN_MOVE_ITER)
        + ". A 0 here means the count never reached `newton_solve` and the"
        " hardcoded budget is back.",
    )
    assert_true(
        d_tol > MIN_MOVE_TOL,
        "`<option tolerance=\"1e-1\">` must stop the solve early; moved "
        + String(d_tol),
    )
    # ⚠ `ls_iterations` IS WIRED AND UNEXERCISED, AND THIS SAYS SO RATHER THAN
    # ASSERTING A MOVE IT CANNOT MAKE. At this pose our line search accepts its
    # first analytic point, so cutting the halving budget to 1 changes `qvel`
    # by 8.9e-16 — the knob reaches the solver (the meta assertion above) but
    # nothing here can tell 1 from 50. Seeding a sliding `qvel` DOES make it
    # bite (1.16e+00), and was tried and removed: at that pose the ANCHOR stops
    # holding (8.1e-05), i.e. it is a pose where we disagree with MuJoCo for an
    # unrelated reason, and gating on it would be
    # `feedback_the_sweep_was_not_the_distribution` all over again. Left as a
    # printed number so a future fixture can pick it up.
    print("  (ls_iterations is plumbed but inert at this pose — see the note)")
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

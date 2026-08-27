"""`warmstart()` — the primal solve starts at the cheaper of two accelerations.

    pixi run mojo run -I . tests/physics3d/test_newton_warmstart_vs_mujoco.mojo

WHAT WAS MISSING. `mj_fwdConstraint` calls `warmstart()`
(engine_forward.c:786) before every solve. Unless `<flag warmstart="disable"/>`
is set it starts the iterate at `qacc_warmstart` — the PREVIOUS `mj_forward`'s
constrained acceleration — falls back to `qacc_smooth` only if the warm start's
primal cost is HIGHER, and `mj_forward` ends by saving the answer
(engine_forward.c:1087). This engine had no `qacc_warmstart` at all and cold-
started from `qacc_smooth` at all three Newton init sites.

⚠⚠ BOTH ENGINES SHARE THE FIXED POINT, SO A CONVERGED SOLVE CANNOT SEE THIS.
That is what let it sit: `unitree_go1` at `iterations="200"` gives the same
answer to 2.2e-16 with the feature on and off. It is a model that TRUNCATES
its solve that is stepping a different algorithm — measured in MuJoCo itself
on go1 at `iterations="2"`:

    N=1   |d(qvel)| warm ON vs OFF   4.70e-03
    N=2                              1.79e+00     <- the CARRY, not the reset
    N=3                              4.25e+00
    N=200 iterations, any N          2.2e-16      <- indifferent, as it must be

⚠ THE GATE HAS TO BE A ROLLOUT. At N=1 from a reset `qacc_warmstart` is zero,
so a single-step fixture tests only the cost comparison against a zero
candidate and NOTHING of the carry — which is the whole feature. Every arm
below steps at least twice.

⚠ IT IS ALSO A PERFORMANCE FEATURE, and the perf claim is the same mechanism:
go1's converged solve takes 28 Newton iterations on step 1 and 6 by step 5
BECAUSE the previous answer is carried. A cold solve pays the full count every
step.

⚠ `mjSOL_PGS` IS DELIBERATELY NOT COVERED. MuJoCo's PGS branch runs a
DIFFERENT test — a cost in `efc_force` space, zeroing the forces when it comes
out positive — and no model in either reference tree selects PGS or CG (every
`<option solver=>` in Menagerie and dm_control says Newton). Sharing one code
path between the two would be inventing an algorithm neither engine has.
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, assert_equal, assert_false, TestSuite

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
    MODEL_META_IDX_WARMSTART_DISABLED, META_IDX_NUM_CONTACTS, KEY_IDX_NQPOS,
)


comptime DT = DType.float64

comptime GO1 = String(
    "references/mujoco_menagerie-main/unitree_go1/scene.xml"
)

# go1's own `<option>` restated in every arm, so the ONLY thing that varies is
# the budget and the flag. `tolerance="0"` means "never stop early", which is
# what makes `iterations` the sole terminator of the truncated arms.
comptime OPT_T2 = String(
    '<option cone="elliptic" impratio="100" iterations="2" tolerance="0"/>'
)
comptime OPT_T2_OFF = String(
    '<option cone="elliptic" impratio="100" iterations="2" tolerance="0">'
    '<flag warmstart="disable"/></option>'
)
comptime OPT_CONV = String(
    '<option cone="elliptic" impratio="100" iterations="200" tolerance="0"/>'
)
comptime OPT_CONV_OFF = String(
    '<option cone="elliptic" impratio="100" iterations="200" tolerance="0">'
    '<flag warmstart="disable"/></option>'
)

comptime NSTEP: Int = 2

# How far the two arms must move apart at `iterations="2"`, N=2. MuJoCo's own
# separation is 1.79e+00; this sits two orders under it.
comptime MIN_ARM_SPLIT: Float64 = 1e-2
# And how tightly the two arms must agree once converged. MuJoCo's is 3.3e-16;
# ours carries a rollout's round-off on top.
comptime TOL_CONVERGED: Float64 = 1e-9
# And how tightly our carried acceleration must match MuJoCo's after ONE
# converged step from the shared keyframe.
comptime TOL_CARRY: Float64 = 1e-9


def _swap_option(xml: String, opt: String) raises -> String:
    """Replace the model's whole `<option .../>` element with `opt`."""
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
    var warmstart: List[Float64]
    var ncon: Int
    var ws_disabled: Int
    var cone: Int
    var integ: Int


def _ours(opt: String, nstep: Int, seed_ws: Float64 = 0.0) raises -> _Run:
    """Load go1 with `opt` THE WAY THE STUDIO DOES and step it `nstep` times.

    `seed_ws` plants a hostile `qacc_warmstart` before the FIRST step — see
    `test_a_hostile_warm_start_is_discarded`.
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
    init_hfield_data(d, m)
    for i in range(nq):
        d.qpos.data[i] = sf.qpos0.data[i]
    var nqp = Int(Float64(sf.key_meta.data[KEY_IDX_NQPOS]))
    for i in range(min(nqp, nq)):
        d.qpos.data[i] = sf.key_qpos.data[i]
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](0)
    if seed_ws != 0.0:
        for i in range(nv):
            d.qacc_warmstart.data[i] = Scalar[DT](
                seed_ws * Float64(1 + (i % 7)) * (-1.0 if i % 2 == 1 else 1.0)
            )

    var ell = StudioIntegEll(dims)
    for _ in range(nstep):
        ell.step["cpu"](d, m)

    var qv = List[Float64]()
    for i in range(nv):
        qv.append(Float64(d.qvel.data[i]))
    var ws = List[Float64]()
    for i in range(nv):
        ws.append(Float64(d.qacc_warmstart.data[i]))
    return _Run(
        qvel=qv^,
        warmstart=ws^,
        ncon=Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS])),
        ws_disabled=Int(
            Float64(m.meta.data[MODEL_META_IDX_WARMSTART_DISABLED])
        ),
        cone=studio_cone_of(fmd),
        integ=studio_integrator_of(fmd),
    )


@fieldwise_init
struct _MjRun(Movable):
    var qvel: List[Float64]
    var warmstart: List[Float64]


def _mj(iters: Int, disable_ws: Bool, nstep: Int) raises -> _MjRun:
    """MuJoCo's rollout with the same budget and the same flag.

    ⚠ THE FLAG IS SET ON `m.opt.disableflags`, NOT BY EDITING THE XML.
    `from_xml_string` cannot resolve go1's `meshdir="assets"`, and copying the
    tree to add one element would write into `references/`. `mjDSBL_WARMSTART`
    is exactly what `<flag warmstart="disable"/>` compiles to.
    """
    var mujoco = Python.import_module("mujoco")
    var py = Python.import_module("builtins")
    var m = mujoco.MjModel.from_xml_path(GO1)
    m.opt.iterations = iters
    m.opt.tolerance = Float64(0)
    if disable_ws:
        # ⚠ THE `|` STAYS IN PYTHON. `PythonObject` is neither `Intable` nor
        # `Floatable`, so casting either operand to a Mojo `Int` first does
        # not compile; `py.int(...)` keeps both sides Python ints and the
        # result is assigned straight back.
        m.opt.disableflags = py.int(m.opt.disableflags).__or__(
            py.int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
        )
    var d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(nstep):
        mujoco.mj_step(m, d)
    var qv = d.qvel.flatten().tolist()
    var out = List[Float64]()
    for i in range(len(qv)):
        out.append(Float64(py=qv[i]))
    var wl = d.qacc_warmstart.flatten().tolist()
    var ws = List[Float64]()
    for i in range(len(wl)):
        ws.append(Float64(py=wl[i]))
    return _MjRun(qvel=out^, warmstart=ws^)


def _dmax(a: List[Float64], b: List[Float64]) -> Float64:
    var n = len(a) if len(a) < len(b) else len(b)
    var w = Float64(0)
    for i in range(n):
        var v = abs(a[i] - b[i])
        if v > w:
            w = v
    return w


def _amax(a: List[Float64]) -> Float64:
    var w = Float64(0)
    for i in range(len(a)):
        if abs(a[i]) > w:
            w = abs(a[i])
    return w


def test_the_fixture_is_what_it_claims() raises:
    """Elliptic, Euler, in contact, and the flag actually reaching the meta."""
    print("=== fixture ===")
    var on = _ours(OPT_T2, NSTEP)
    var off = _ours(OPT_T2_OFF, NSTEP)
    print("  ncon", on.ncon, " cone", on.cone, " integ", on.integ)
    print("  meta warmstart_disabled: default", on.ws_disabled,
          " with the flag", off.ws_disabled)
    assert_true(
        on.ncon > 0,
        "go1 must be in contact at keyframe 0 or the solver has nothing to"
        " iterate on and every arm below is vacuous; got ncon "
        + String(on.ncon),
    )
    assert_equal(on.cone, ConeType.ELLIPTIC, "the fixture must be elliptic")
    assert_equal(
        on.integ, IntegratorType.EULER, "the fixture must be Euler"
    )
    assert_equal(
        on.ws_disabled, 0,
        "MuJoCo's default is warm start ENABLED, so an <option> without the"
        " flag must leave the slot at 0",
    )
    assert_equal(
        off.ws_disabled, 1,
        "<flag warmstart=\"disable\"/> must reach"
        " MODEL_META_IDX_WARMSTART_DISABLED — if this is 0 the parser is not"
        " reading the flag and every other arm here is comparing two identical"
        " runs",
    )


def test_the_carry_is_the_references_carry() raises:
    """`d.qacc_warmstart` after ONE converged step IS MuJoCo's, and is not zero.

    ⚠ ONE STEP, NOT THE ROLLOUT. Both engines leave the keyframe in the same
    state, so after a single converged step this compares the SOLVE and
    nothing else. Two steps in, our accelerations have already separated by
    ~0.7% — a contact acceleration under `impratio="100"` is a very stiff
    function of a penetration depth that differs in the 9th digit — and a
    tolerance loose enough to pass that would be loose enough to pass a field
    nobody wrote.

    ⚠ THE NON-ZERO CHECK IS HALF THE TEST. An engine that never wrote the
    field would leave it at `mj_resetData`'s zero, and a purely relative
    comparison would call that agreement.
    """
    print("=== the carry ===")
    var ours = _ours(OPT_CONV, 1)
    var mj = _mj(200, False, 1)
    var mag = _amax(mj.warmstart)
    var err = _dmax(ours.warmstart, mj.warmstart)
    print("  |qacc_warmstart|max  ours", _amax(ours.warmstart),
          " MuJoCo", mag, "   |d| ", err)
    assert_true(
        mag > 1.0,
        "the fixture must leave a substantial acceleration behind or this"
        " arm proves nothing; MuJoCo's |qacc_warmstart|max is " + String(mag),
    )
    assert_true(
        _amax(ours.warmstart) > 1.0,
        "d.qacc_warmstart is still (near) zero after a step —"
        " `save_qacc_warmstart` is not running, so the solver is pricing a"
        " candidate that never carries anything",
    )
    assert_true(
        err < TOL_CARRY * mag,
        "our carried acceleration differs from MuJoCo's by " + String(err)
        + " on a magnitude of " + String(mag),
    )


def test_a_truncated_solve_is_moved_by_the_warm_start() raises:
    """At `iterations="2"` the flag moves the answer — in both engines.

    ⚠ THIS ARM DELIBERATELY DOES NOT CROSS-COMPARE OURS TO MuJoCo. At a
    truncated budget the two engines' Newton ITERATES differ for a reason that
    has nothing to do with the warm start — the open divergence the sibling
    `test_solver_budget_from_option` documents (go1 at `iterations="1"`:
    2.56e+00). Measured here: with the warm start on we land 1.14e-01 from
    MuJoCo's warm arm and with it off 2.24e+00 from MuJoCo's cold arm, both
    dominated by that. Asserting on either number would be asserting on the
    open bug, and would go red the day it is fixed.

    What this arm IS for: the flag has to be LOAD-BEARING. A parsed flag that
    nothing dispatches on is this tree's single most repeated defect, and two
    identical answers is exactly how it looks. The reference's own split is
    printed beside ours so a shrinking one is visible.
    """
    print("=== truncated: iterations=2, N=" + String(NSTEP) + " ===")
    var o_on = _ours(OPT_T2, NSTEP)
    var o_off = _ours(OPT_T2_OFF, NSTEP)
    var m_on = _mj(2, False, NSTEP)
    var m_off = _mj(2, True, NSTEP)
    var split_mj = _dmax(m_on.qvel, m_off.qvel)
    var split_ours = _dmax(o_on.qvel, o_off.qvel)
    print("  arm split   MuJoCo", split_mj, "   ours", split_ours)
    print("  (open truncated-iterate divergence, NOT asserted:"
          " |ours-MuJoCo| warm", _dmax(o_on.qvel, m_on.qvel),
          " cold", _dmax(o_off.qvel, m_off.qvel), ")")
    assert_true(
        split_mj > MIN_ARM_SPLIT,
        "the REFERENCE's two arms barely differ (" + String(split_mj)
        + ") — the fixture is not truncated enough to test anything",
    )
    assert_true(
        split_ours > MIN_ARM_SPLIT,
        "our two arms give the same answer (" + String(split_ours)
        + "), so the flag is parsed and nothing dispatches on it",
    )


def test_a_converged_solve_is_indifferent() raises:
    """The feature moves the TRAJECTORY, never the fixed point.

    A warm start that changed a converged answer would not be a warm start —
    it would be a different optimisation problem.
    """
    print("=== converged: iterations=200 ===")
    var on = _ours(OPT_CONV, NSTEP)
    var off = _ours(OPT_CONV_OFF, NSTEP)
    var e = _dmax(on.qvel, off.qvel)
    print("  |warm ON - warm OFF| converged:", e)
    assert_true(
        e < TOL_CONVERGED,
        "a converged solve moved by " + String(e) + " when the warm start was"
        " switched on — the two starts are not reaching the same optimum",
    )


def test_both_branches_of_the_cost_comparison_are_live() raises:
    """A worse candidate is DISCARDED and a better one is KEPT.

    ⚠⚠ THE COMPARISON IS THE ALGORITHM. An unconditional
    `qacc = qacc_warmstart` would pass any test that only asked whether the
    field is read; MuJoCo prices both candidates and keeps the cheaper, and a
    warm start that is worse than the cold one is thrown away.

    Both branches are reachable from a single step of this fixture:

      * plant a large acceleration -> its Gauss term alone dwarfs the cold
        cost, so it MUST be rejected and the step must land exactly where the
        `warmstart="disable"` arm lands. EXACTLY — the reject path restores
        `Ma` from `qfrc_smooth`, which IS `M * qacc_smooth` bit for bit, so
        anything but 0.0 here is residue from the trial.
      * leave it at the reset's zero -> on this fixture ZERO IS CHEAPER than
        `qacc_smooth` (as it is on `apptronik_apollo`, where MuJoCo's own two
        costs are 1.129e+06 against 1.734e+06), so it must be KEPT and the
        step must NOT land there.

    ⚠ ONE STEP. From the second step on, the warm-start arm carries its own
    previous answer and the comparison against the cold arm stops being about
    the first decision.

    ⚠ THE SEED MAGNITUDE IS DELIBERATELY SWEPT. A difference that scales with
    it would be contamination — the trial state leaking into the solve. A
    difference that is IDENTICAL at 1e+02 and 1e+08 is a discrete branch,
    which is what a rejection is. That sweep is what showed the first version
    of this test was asserting the wrong thing.
    """
    print("=== both branches ===")
    var cold = _ours(OPT_T2_OFF, 1)
    var kept = _ours(OPT_T2, 1)
    print("  |kept(zero candidate) - cold| :", _dmax(kept.qvel, cold.qvel))
    assert_true(
        _dmax(kept.qvel, cold.qvel) > 0.0,
        "a zero warm start gave the same step as the cold arm — either the"
        " candidate is never kept, or the flag does not reach the solver",
    )
    for e10 in range(3):
        var mag = 100.0
        if e10 == 1:
            mag = 1.0e5
        elif e10 == 2:
            mag = 1.0e8
        var seeded = _ours(OPT_T2, 1, seed_ws=mag)
        var e = _dmax(seeded.qvel, cold.qvel)
        print("  seed", mag, " |seeded - cold| :", e)
        assert_equal(
            e, 0.0,
            "a " + String(mag) + " acceleration planted in qacc_warmstart"
            " moved the step by " + String(e) + " away from the cold arm —"
            " the solver is COPYING the candidate instead of pricing it, or"
            " the reject path is leaving trial state behind",
        )


def test_a_fresh_data_starts_from_zero() raises:
    """`mj_resetData` zeroes `qacc_warmstart`, and so must construction.

    ⚠ A LOOP-BOUND-SHAPED TRAP IN REVERSE: this field is not a bound, but it
    IS read on the first solve of every hand-built `Data` in this tree — the
    board harness, the GPU env specs, every fixture that skips the parser.
    """
    print("=== a fresh Data ===")
    var src = read_model_source(GO1)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var dims = dims_from_flat(fmd, max_contacts=32, nmesh_verts=8192)
    var d = Data[DT, DynDims, 1](dims)
    var w = Float64(0)
    for i in range(dims.get_nv()):
        if abs(Float64(d.qacc_warmstart.data[i])) > w:
            w = abs(Float64(d.qacc_warmstart.data[i]))
    print("  |qacc_warmstart|max on a fresh Data:", w)
    assert_equal(w, 0.0, "a fresh Data must start from a zero warm start")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

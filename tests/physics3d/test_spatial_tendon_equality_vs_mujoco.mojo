"""`<equality><tendon>` on a SPATIAL tendon — mjEQ_TENDON, site-routed.

THE DEFECT THIS GATES. Until 2026-08-12 a spatial equality tendon was not
constrained anywhere in the engine. `build_tendon_equality_rows` skipped
SPATIAL on the written grounds that "the `_tendon_env` post-pass still covers
it"; `_tendon_env` had no spatial branch at all — it computed a FIXED tendon's
length from joint coefficients, and a spatial tendon has `num_joints == 0`, so
it built a row with a ZERO Jacobian, converged it, and applied
`qacc += M^-1 J^T dlambda == 0`. Four comments across the solvers asserted the
handoff and cited each other in a ring. Nobody ever ran the pair, because no
model in the tree puts an equality on a spatial tendon (checked: quadruped,
manipulator and stacker each have one, all three FIXED).

Measured on this fixture against the code as it stood:

    ours (pre-fix)   -3.1470479999999923
    MuJoCo free fall -3.147048              <- bit-for-bit, the constraint was absent
    MuJoCo           -0.000367181842

⚠ THE FIRST VERSION OF THIS PROBE STEPPED WITH `CONTACTS=False`, on which NO
constraint solver runs at all (`EulerIntegrator.step` takes the `solve_limits`
branch). It reported free fall — the right answer, for a reason that had
nothing to do with the defect, and it would have gone on reporting free fall
after the fix. Every test below therefore steps with `CONTACTS=True`, and
`test_the_fixture_is_not_vacuous` pins the discrimination from MuJoCo's side.

WHY THIS FIXTURE. The bob's ONLY support is the equality: one slide dof, no
floor, no contacts, no limits, no weld, no spring. Nothing else in the engine
can produce an upward force, so "held" and "not held" are three orders of
magnitude apart rather than a tolerance argument. The residual sag is real
physics — MuJoCo's soft equality obeys `pos = R*lambda/(K*imp)` — and it is the
sag we have to reproduce, not zero.

`aux` is a second, independent body: a hinge with `ref` and a FIXED tendon on
it, present only so `tendon_length0` is gated on BOTH kinds with a NONZERO
answer. Nothing couples it to the bob.

Run with:
    pixi run mojo run -I . tests/physics3d/test_spatial_tendon_equality_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.parser.xml_parser import merge_mjcf
from mojo_rl.physics3d.fields import Model, Data, Dims
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.gpu.constants import (
    MODEL_TENDON_SIZE,
    TENDON_IDX_LENGTH_REF,
)

comptime DTYPE = DType.float64
comptime NSTEPS = 400

# MuJoCo 3.10.0 on this model. Recomputed in-test; these are here so a reader
# can see the scale of the discrimination without running anything.
comptime MJ_HELD = -0.000367181842
comptime MJ_FREE_FALL = -3.147048

comptime _RAW = """
<mujoco model="spatial_eq">
  <option timestep="0.002" gravity="0 0 -9.81" solver="Newton"/>
  <worldbody>
    <site name="anchor" pos="0 0 1" size="0.01"/>
    <body name="bob" pos="0 0 0.8">
      <joint name="slide_z" type="slide" axis="0 0 1"/>
      <geom name="g_bob" type="sphere" size="0.05" mass="1"/>
      <site name="s_bob" pos="0 0 0" size="0.01"/>
    </body>
    <body name="aux" pos="1 0 1">
      <joint name="aux_hinge" type="hinge" axis="0 1 0" ref="0.25"/>
      <geom name="g_aux" type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="rope">
      <site site="anchor"/>
      <site site="s_bob"/>
    </spatial>
    <fixed name="aux_fixed">
      <joint joint="aux_hinge" coef="1.0"/>
    </fixed>
  </tendon>
  <equality>
    <tendon tendon1="rope"/>
  </equality>
</mujoco>
"""

comptime XML = merge_mjcf(_RAW)
comptime pm = parse_xml(XML)


def _model_def[cone: Int]() -> ModelDefFromXML[
    xml=XML,
    nbody = pm.NBODY,
    njoint = pm.NJOINT,
    nq = pm.NQ,
    nv = pm.NV,
    ngeom = pm.NGEOM,
    nact = pm.NACT,
    ntex = pm.NTEX,
    nmat = pm.NMAT,
    nlight = pm.NLIGHT,
    ncam = pm.NCAM,
    nsite = pm.NSITE,
    max_tendon = pm.NTENDON,
    cone_type=cone,
    max_contacts=4,
    max_condim = pm.MAX_CONDIM,
    neq = pm.NEQ,
    nexclude = pm.NEXCLUDE,
    timestep = pm.TIMESTEP,
]:
    return {}


comptime M_ELL = _model_def[ConeType.ELLIPTIC]()
comptime MD_2 = Dims[
    nq=M_ELL.NQ,
    nv=M_ELL.NV,
    nbody=M_ELL.NBODY,
    njoint=M_ELL.NJOINT,
    ngeom=M_ELL.NGEOM,
    nsite=M_ELL.NSITE,
    max_contacts=M_ELL.MAX_CONTACTS,
    nequality=M_ELL.MAX_EQUALITY,
    ntendon=M_ELL.MAX_TENDON,
    nexclude=M_ELL.NEXCLUDE,
    nmesh_verts=0,
    npair=M_ELL.NPAIR,
    nact=M_ELL.NACT,
    nten=M_ELL.NTEN_F,
    nkey=M_ELL.NKEY,
]
comptime M_PYR = _model_def[ConeType.PYRAMIDAL]()


def _roll[M: ModelDefFromXML, SOLVER: StaticString]() raises -> Float64:
    """Step the fixture `NSTEPS` and return the bob's slide coordinate.

    ⚠ `CONTACTS=True` is load-bearing: the constraint seam only runs on that
    branch, so with `CONTACTS=False` this returns free fall no matter what the
    solvers do. See the module docstring.
    """
    comptime MD = Dims[
        nq=M.NQ,
        nv=M.NV,
        nbody=M.NBODY,
        njoint=M.NJOINT,
        ngeom=M.NGEOM,
        nsite=M.NSITE,
        max_contacts=M.MAX_CONTACTS,
        nequality=M.MAX_EQUALITY,
        ntendon=M.MAX_TENDON,
        nexclude=M.NEXCLUDE,
        nmesh_verts=0,
        npair=M.NPAIR,
        nact=M.NACT,
        nten=M.NTEN_F,
        nkey=M.NKEY,
    ]
    var sf = M.make_spec_fields[DTYPE]()
    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD]()
    M.init_fields[DTYPE](ctx, mf)

    var d = Data[DTYPE, MD, 1]()
    M.reset_data[DTYPE](sf, d)
    forward_kinematics["cpu"](d, mf)

    var integ = EulerIntegrator[DTYPE, MD, M.CONE_TYPE, 1, SOLVER=SOLVER, MAX_CONDIM = M.MAX_CONDIM, NOSLIP_ITER = M.NOSLIP_ITER]()
    for _ in range(NSTEPS):
        integ.step["cpu", CONTACTS=True](d, mf)
    return Float64(d.qpos.data[0])


def _mujoco_roll(disable_equality: Bool) raises -> Float64:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[XML]())
    if disable_equality:
        m.opt.disableflags = (
            Int(py=m.opt.disableflags)
            | Int(py=mujoco.mjtDisableBit.mjDSBL_EQUALITY)
        )
    var dat = mujoco.MjData(m)
    for _ in range(NSTEPS):
        mujoco.mj_step(m, dat)
    return Float64(py=dat.qpos[0])


def test_the_fixture_is_not_vacuous() raises:
    """MuJoCo itself must hold the bob, and free fall must be far away.

    Without this the file could pass while the constraint does nothing — which
    is exactly the state the engine was in, and exactly what the first version
    of this probe failed to notice.
    """
    print("--- spatial tendon equality: the fixture discriminates ---")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[XML]())
    var dat = mujoco.MjData(m)
    mujoco.mj_forward(m, dat)
    var nefc = Int(py=dat.nefc)
    print("  MuJoCo nefc =", nefc, " ncon =", Int(py=dat.ncon))
    assert_true(
        nefc == 1,
        "expected exactly ONE constraint row (the tendon equality) — got "
        + String(nefc)
        + ". More rows and the attribution below is not clean; none and this"
        " file gates nothing.",
    )
    var want_eq = Int(py=mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    assert_true(
        Int(py=dat.efc_type[0]) == want_eq,
        "the row is not mjCNSTR_EQUALITY — the fixture no longer expresses a"
        " tendon equality",
    )

    var held = _mujoco_roll(False)
    var freefall = _mujoco_roll(True)
    print("  MuJoCo held      =", held)
    print("  MuJoCo free fall =", freefall)
    assert_true(
        abs(freefall - held) > 1.0,
        "held and free fall are less than a metre apart — the fixture has"
        " stopped discriminating and a broken engine could pass",
    )
    assert_true(
        abs(held - MJ_HELD) < 1e-9,
        "MuJoCo's own answer moved from the recorded "
        + String(MJ_HELD)
        + " to "
        + String(held)
        + " — the runtime changed under this gate; re-derive before trusting"
        " the comparisons below.",
    )


def test_tendon_length0_matches_mujoco() raises:
    """`length_ref` is MuJoCo's `tendon_length0`, for BOTH tendon kinds.

    Nothing assigned it before 2026-08-12: `TendonData.length_ref` defaulted to
    0.0 and no parser ever wrote it, so every equality tendon was solved
    against a target of zero. It was invisible because `m.tendon_length0` is
    0.0 for all four equality tendons in the tree. Both entries here are
    NONZERO on purpose — a gate whose expected value is the old default would
    pass with the bug present.
    """
    print("--- spatial tendon equality: tendon_length0 ---")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[XML]())

    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD_2]()
    M_ELL.init_fields[DTYPE](ctx, mf)

    var worst = Float64(0)
    for t in range(M_ELL.MAX_TENDON):
        var ours = Float64(
            mf.tendons.data[t * MODEL_TENDON_SIZE + TENDON_IDX_LENGTH_REF]
        )
        var theirs = Float64(py=m.tendon_length0[t])
        print("  tendon", t, " ours", ours, " MuJoCo", theirs)
        assert_true(
            abs(theirs) > 1e-6,
            "tendon "
            + String(t)
            + " has length0 == 0, so it cannot distinguish the computed value"
            " from the old hardcoded default — the fixture has decayed",
        )
        var e = abs(ours - theirs)
        if e > worst:
            worst = e
    print("  worst |d| =", worst)
    assert_true(
        worst < 1e-12,
        "tendon_length0 disagrees with MuJoCo by " + String(worst),
    )


def _check(name: String, ours: Float64, held: Float64, freefall: Float64) raises:
    print("  ", name, " ours", ours, " MuJoCo", held, " |d|", abs(ours - held))
    assert_true(
        abs(ours - freefall) > 1.0,
        name
        + ": the bob is in FREE FALL — the spatial equality row is not"
        " reaching this solver. That is the original defect: the row builder"
        " skipped SPATIAL and the post-pass it deferred to had no spatial"
        " branch.",
    )
    assert_true(
        abs(ours - held) < 1e-9,
        name + ": disagrees with MuJoCo by " + String(abs(ours - held)),
    )


def test_spatial_equality_holds_on_every_solver_path() raises:
    """The row builder (elliptic + pyramidal Newton) and the post-pass (PGS).

    All three carried the defect and all three were changed, so all three are
    gated. The PGS path is the one that still solves these AFTER the contact
    solve — structurally wrong for the reason defect 29a documents, but it is
    the design those solvers have, and a spatial tendon must at least be
    present there rather than silently dropped.
    """
    print("--- spatial tendon equality: vs MuJoCo, per solver path ---")
    var held = _mujoco_roll(False)
    var freefall = _mujoco_roll(True)

    _check("newton/elliptic ", _roll[M_ELL, "newton"](), held, freefall)
    _check("newton/pyramidal", _roll[M_PYR, "newton"](), held, freefall)
    _check("pgs post-pass   ", _roll[M_PYR, "pgs"](), held, freefall)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

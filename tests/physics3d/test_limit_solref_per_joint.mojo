"""Joint-limit solref/solimp are PER-JOINT — defect 22, gated vs MuJoCo.

`fields_build` broadcasts JOINT 0's `solreflimit`/`solimplimit` into model meta,
and `constraints/limits.mojo` used to read them from there, so every limit row
in the model took joint 0's parameters. On dog that meant reading the model
defaults off the FREE ROOT — the one joint that can never own a limit row — and
every limit came out 3.68x too soft (K 2770.08 against MuJoCo's efc_KBIP
10203.04).

⚠⚠ WHY THIS FILE EXISTS RATHER THAN A CHECK IN THE EXISTING LIMITS GATE.
`test_humanoid_limits_fields_vs_mujoco` PASSES with the bug present, and always
would: humanoid's joints all carry the same limit params, so joint 0's values
ARE the correct ones and the broadcast is invisible. A gate for a
"wrong source for a parameter" defect has to use a model where the wrong source
gives a DIFFERENT answer, or it gates nothing. See
`feedback_sweep_model_must_express_defect`.

THE FIXTURE (`limit_solref_ref.py`, shared with the reference side):

    j0  hinge, UNLIMITED   -> global default solreflimit [0.02 1]
    j1  hinge, LIMITED at +-30 deg, driven to 45 deg with velocity into the stop
        solreflimit "0.04 1", solimplimit "0.9 0.99 0.01"

    K = 1/(dmax^2 * timeconst^2 * dampratio^2)
        joint 0's params:  1/(0.95^2 * 0.02^2) = 2770.08     <- the bug used this
        j1's own params:   1/(0.99^2 * 0.04^2) =  637.69     <- MuJoCo uses this
    4.3x apart, and MuJoCo's efc_KBIP confirms 637.69 / 50.505.

⚠ j0 IS UNLIMITED ON PURPOSE — it mirrors dog's free root. A "fix" that changed
the broadcast to use the first LIMITED joint would still be wrong in general and
must still fail here.
⚠ AND j1 IS NOT JOINT 0, or the broadcast would be accidentally right.

⚠ CONTACTS=False IS THE PATH UNDER TEST, and that is deliberate rather than
convenient. `solve_limits` is reached only from the `else` branch of
`if CONTACTS` (euler.mojo:632, implicit.mojo:552); with contacts live, Newton
and PGS build their own limit rows and already read per-joint values. That split
is exactly why dog's probe shows stage 5 exact at 2.8e-11 while stage 3 was off
by 86.9 — the two stages differ in precisely this parameter.

Run with:
    pixi run mojo run -I . tests/physics3d/test_limit_solref_per_joint.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.fields import Model, Data, Dims
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.types import ConeType

comptime DTYPE = DType.float64
comptime TEST_PATH = "tests/physics3d"

# ⚠ KEPT BYTE-FOR-BYTE IN STEP WITH `limit_solref_ref.py`'s `XML`, and the
# first test asserts exactly that. Two copies of a fixture that drift apart
# would compare two different models and report a physics defect.
comptime LIMIT_XML = """<mujoco model="limit_solref">
  <compiler angle="degree"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="b0" pos="0 0 1">
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" density="1000"/>
      <body name="b1" pos="0.2 0 0">
        <joint name="j1" type="hinge" axis="0 1 0" limited="true"
               range="-30 30" solreflimit="0.04 1"
               solimplimit="0.9 0.99 0.01"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" density="1000"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

comptime lp = parse_xml(LIMIT_XML)
comptime LM = ModelDefFromXML[
    xml=LIMIT_XML,
    nbody=lp.NBODY, njoint=lp.NJOINT, nq=lp.NQ, nv=lp.NV,
    ngeom=lp.NGEOM, nact=lp.NACT, ntex=lp.NTEX, nmat=lp.NMAT,
    nlight=lp.NLIGHT, ncam=lp.NCAM, nsite=lp.NSITE,
    max_tendon=lp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=8,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=lp.TIMESTEP,
]

# ── Defect 23 fixture: solreflimit BELOW 2*timestep ──────────────────────────
# Same model, solreflimit 0.0025 against timestep 0.005 (2*dt = 0.01). MuJoCo
# raises it to 0.01 when it builds the row ("integrator safety",
# engine_core_constraint.c:2028); unclamped the stiffness is 16x MuJoCo's.
#
# ⚠ THE MODEL TABLES STILL SAY 0.0025 — the clamp happens at ROW BUILD, not at
# compile. A model-constant gate sees nothing here; only the solved row does.
comptime LIMIT_XML_CLAMPED = """<mujoco model="limit_solref">
  <compiler angle="degree"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="b0" pos="0 0 1">
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" density="1000"/>
      <body name="b1" pos="0.2 0 0">
        <joint name="j1" type="hinge" axis="0 1 0" limited="true"
               range="-30 30" solreflimit="0.0025 1"
               solimplimit="0.9 0.99 0.01"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" density="1000"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

comptime cp = parse_xml(LIMIT_XML_CLAMPED)
comptime CM = ModelDefFromXML[
    xml=LIMIT_XML_CLAMPED,
    nbody=cp.NBODY, njoint=cp.NJOINT, nq=cp.NQ, nv=cp.NV,
    ngeom=cp.NGEOM, nact=cp.NACT, ntex=cp.NTEX, nmat=cp.NMAT,
    nlight=cp.NLIGHT, ncam=cp.NCAM, nsite=cp.NSITE,
    max_tendon=cp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=8,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=cp.TIMESTEP,
]

comptime NQ = LM.NQ
comptime NV = LM.NV

# j1 at 45 deg against a +-30 deg range, moving further into the stop so the
# damping term B is live too — a pure position violation would gate K and leave
# B untested.
comptime Q0 = 0.3
comptime Q1 = 0.785398163397448
comptime V0 = 0.0
comptime V1 = 1.5


def _ref() raises -> Tuple[PythonObject, PythonObject, PythonObject]:
    var sys = Python.import_module("sys")
    sys.path.insert(0, TEST_PATH)
    var mujoco = Python.import_module("mujoco")
    var refmod = Python.import_module("limit_solref_ref")
    var m = refmod.model()
    # Contacts off on BOTH sides: this gates the limit rows, and the two links
    # would otherwise self-collide and swamp the comparison.
    m.opt.disableflags = (
        Int(py=m.opt.disableflags)
        | Int(py=mujoco.mjtDisableBit.mjDSBL_CONTACT)
    )
    return (mujoco, m, refmod)


def test_fixture_matches_the_reference_xml() raises:
    """The two copies of the model must be the same model."""
    var h = _ref()
    var refmod = h[2]
    var py_xml = String(py=refmod.XML)
    assert_true(
        py_xml == String(LIMIT_XML),
        "the XML in this file and in limit_solref_ref.py have DRIFTED. They"
        " must stay byte-identical or this file compares two different models"
        " and reports the difference as a physics defect.",
    )


def test_limit_solref_is_read_per_joint() raises:
    """qacc under CONTACTS=False, ours vs MuJoCo, on a non-uniform model."""
    var sf = LM.make_spec_fields[DTYPE]()
    print("--- joint-limit solref/solimp per joint (defect 22) ---")
    var h = _ref()
    var mujoco = h[0]
    var m = h[1]
    var np = Python.import_module("numpy")

    var dat = mujoco.MjData(m)
    dat.qpos[0] = Q0
    dat.qpos[1] = Q1
    dat.qvel[0] = V0
    dat.qvel[1] = V1
    mujoco.mj_forward(m, dat)

    var nefc = Int(py=dat.nefc)
    print("  MuJoCo nefc =", nefc, " ncon =", Int(py=dat.ncon))
    # NON-VACUITY, and it is the whole point: with no active limit row every
    # solref would be unused and this file would pass with the bug present.
    assert_true(
        nefc == 1,
        "expected exactly ONE active limit row (j1 past its stop) — got "
        + String(nefc) + ". With no limit row this gate is vacuous, and with"
        " more than one the attribution below is not clean.",
    )
    assert_true(
        Int(py=dat.efc_type[0]) == 3,
        "the active row is not a LIMIT_JOINT row (mjCNSTR_LIMIT_JOINT = 3)",
    )
    var K_mj = Float64(py=dat.efc_KBIP[0][0])
    print("  MuJoCo efc_KBIP[0] K =", K_mj)
    # THE DISCRIMINATION: joint 0's params would give 2770.08 here. If MuJoCo
    # itself reported that, the fixture would have stopped expressing the
    # defect and the rest of this file would be meaningless.
    assert_true(
        abs(K_mj - 637.69003163) < 1e-4,
        "MuJoCo's limit stiffness is not j1's 637.69 — the fixture no longer"
        " expresses the defect. Joint 0's params would give 2770.08; if K has"
        " moved to that, solreflimit is being ignored on the reference side"
        " too and this gate proves nothing.",
    )

    # --- our side ---------------------------------------------------------
    var ctx = DeviceContext()
    var mf = Model[DTYPE, Dims[nv=LM.NV, nbody=LM.NBODY, njoint=LM.NJOINT, ngeom=LM.NGEOM, nequality=LM.MAX_EQUALITY, ntendon=LM.MAX_TENDON, nsite=LM.NSITE, nexclude=LM.NEXCLUDE, nmesh_verts=0]]()
    LM.init_fields[DTYPE, 0](ctx, mf)
    var d = Data[DTYPE, LM.NQ, LM.NV, LM.NBODY, LM.MAX_CONTACTS, LM.NSITE, 1]()
    LM.reset_data[DTYPE](sf, d)
    d.qpos.data[0] = Scalar[DTYPE](Q0)
    d.qpos.data[1] = Scalar[DTYPE](Q1)
    d.qvel.data[0] = Scalar[DTYPE](V0)
    d.qvel.data[1] = Scalar[DTYPE](V1)
    for i in range(NV):
        d.qfrc.data[i] = Scalar[DTYPE](0)
    forward_kinematics["cpu"](d, mf)

    var integ = EulerIntegrator[
        DTYPE, LM.NQ, LM.NV, LM.NBODY, LM.NJOINT, LM.MAX_CONTACTS, LM.NGEOM,
        LM.MAX_EQUALITY, LM.MAX_TENDON, LM.NSITE, LM.NEXCLUDE, 0,
        LM.CONE_TYPE, 1, SOLVER="newton",
        MAX_CONDIM=LM.MAX_CONDIM, NOSLIP_ITER=LM.NOSLIP_ITER,
    ]()
    # ⚠ CONTACTS=False reaches `solve_limits`, the path this defect lived on.
    integ.step["cpu", CONTACTS=False](d, mf)

    var worst = Float64(0)
    var worst_i = -1
    for i in range(NV):
        var ours = Float64(integ.scratch.qacc_constrained.data[i])
        var theirs = Float64(py=dat.qacc[i])
        var e = abs(ours - theirs)
        print("   dof", i, " ours", ours, " MuJoCo", theirs, " |d|", e)
        if e > worst:
            worst = e
            worst_i = i
    print("  worst |d(qacc)| =", worst, " at dof", worst_i)

    # The magnitude has to be real, or a near-zero qacc would pass trivially.
    var mag = Float64(py=np.max(np.abs(dat.qacc)))
    print("  max|qacc| =", mag)
    assert_true(
        mag > 10.0,
        "the reference acceleration is ~0, so this comparison is vacuous",
    )

    assert_true(
        worst < 1e-9,
        "qacc disagrees with MuJoCo on a model whose LIMITED joint does not"
        " share joint 0's solreflimit. That is defect 22: `solve_limits` must"
        " read JOINT_IDX_SOLREF_LIMIT_*/SOLIMP_LIMIT_* from the joint that OWNS"
        " each row, not the MODEL_META_IDX_* slots that `fields_build` fills"
        " from joint 0. Measured before the fix: joint 0's params give"
        " K = 2770.08 where MuJoCo uses j1's 637.69, 4.3x apart.",
    )


def test_refsafe_clamp_raises_timeconst_to_two_timesteps() raises:
    """solref[0] >= 2*timestep at row build — defect 23, gated vs MuJoCo.

    ⚠ THIS TEST EXISTS BECAUSE THE CLAMP HAD NO OBSERVABLE EFFECT ANYWHERE
    ELSE. Its only live site in the ported suite is quadruped's four equality
    rows, and quadruped gates 11/11 both with and without it — an equality
    constraint enforces the same kinematic condition at either stiffness, so
    the converged solution barely moves. A change with no failing test is
    indistinguishable from a change that is not wired up, which is why this
    fixture is built to make the clamp the ONLY thing that differs.

        declared solreflimit 0.0025, timestep 0.005, 2*timestep 0.01
        MuJoCo   K = 1/(0.99^2 * 0.01^2)   =  10203.04   (efc_KBIP, measured)
        unclamped  1/(0.99^2 * 0.0025^2)   = 163248.65   16x too stiff
    """
    var sf = CM.make_spec_fields[DTYPE]()
    print("--- REFSAFE clamp: solref[0] >= 2*timestep (defect 23) ---")
    var sys = Python.import_module("sys")
    sys.path.insert(0, TEST_PATH)
    var mujoco = Python.import_module("mujoco")
    var refmod = Python.import_module("limit_solref_ref")
    var m = refmod.model_clamped()
    m.opt.disableflags = (
        Int(py=m.opt.disableflags)
        | Int(py=mujoco.mjtDisableBit.mjDSBL_CONTACT)
    )
    var dat = mujoco.MjData(m)
    dat.qpos[0] = Q0
    dat.qpos[1] = Q1
    dat.qvel[0] = V0
    dat.qvel[1] = V1
    mujoco.mj_forward(m, dat)

    assert_true(
        Int(py=dat.nefc) == 1,
        "expected exactly one active limit row; the fixture is not exercising"
        " the clamp",
    )
    # THE FIXTURE MUST BE IN THE CLAMP REGION, or this gates nothing: MuJoCo's
    # own K has to be the CLAMPED value, not the declared one.
    var K_mj = Float64(py=dat.efc_KBIP[0][0])
    print("  declared solreflimit 0.0025, dt 0.005 -> MuJoCo K =", K_mj)
    assert_true(
        abs(K_mj - 10203.0405) < 1e-3,
        "MuJoCo's K is not the CLAMPED 10203.04. If it reads 163248.65 the"
        " reference stopped clamping and this test gates nothing; if it reads"
        " something else the fixture drifted out of the clamp region.",
    )

    var ctx = DeviceContext()
    var mf = Model[DTYPE, Dims[nv=CM.NV, nbody=CM.NBODY, njoint=CM.NJOINT, ngeom=CM.NGEOM, nequality=CM.MAX_EQUALITY, ntendon=CM.MAX_TENDON, nsite=CM.NSITE, nexclude=CM.NEXCLUDE, nmesh_verts=0]]()
    CM.init_fields[DTYPE, 0](ctx, mf)
    var d = Data[DTYPE, CM.NQ, CM.NV, CM.NBODY, CM.MAX_CONTACTS, CM.NSITE, 1]()
    CM.reset_data[DTYPE](sf, d)
    d.qpos.data[0] = Scalar[DTYPE](Q0)
    d.qpos.data[1] = Scalar[DTYPE](Q1)
    d.qvel.data[0] = Scalar[DTYPE](V0)
    d.qvel.data[1] = Scalar[DTYPE](V1)
    for i in range(CM.NV):
        d.qfrc.data[i] = Scalar[DTYPE](0)
    forward_kinematics["cpu"](d, mf)

    var integ = EulerIntegrator[
        DTYPE, CM.NQ, CM.NV, CM.NBODY, CM.NJOINT, CM.MAX_CONTACTS, CM.NGEOM,
        CM.MAX_EQUALITY, CM.MAX_TENDON, CM.NSITE, CM.NEXCLUDE, 0,
        CM.CONE_TYPE, 1, SOLVER="newton",
        MAX_CONDIM=CM.MAX_CONDIM, NOSLIP_ITER=CM.NOSLIP_ITER,
    ]()
    integ.step["cpu", CONTACTS=False](d, mf)

    var worst = Float64(0)
    for i in range(CM.NV):
        var ours = Float64(integ.scratch.qacc_constrained.data[i])
        var theirs = Float64(py=dat.qacc[i])
        var e = abs(ours - theirs)
        print("   dof", i, " ours", ours, " MuJoCo", theirs, " |d|", e)
        if e > worst:
            worst = e
    print("  worst |d(qacc)| =", worst)

    assert_true(
        worst < 1e-9,
        "qacc disagrees with MuJoCo on a model whose solreflimit is BELOW"
        " 2*timestep. That is defect 23: `solref_spring_damper` must raise"
        " `ref_tc` to 2*timestep for the STANDARD format (ref_tc > 0) before"
        " computing K and B, as engine_core_constraint.c:2028 does. The direct"
        " (negative) form is exempt. Declared 0.0025 against dt 0.005 gives"
        " K = 163248.65 unclamped where MuJoCo uses 10203.04 — 16x.",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

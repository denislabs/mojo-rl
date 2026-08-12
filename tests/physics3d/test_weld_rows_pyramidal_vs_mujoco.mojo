"""Connect/weld equality as ROWS on the PYRAMIDAL and BLOCKED Newton paths.

Defect 29a: solving weld/connect as a post-pass after the contact solve
rewrites the dofs the contacts had just balanced, so the contact force is
computed as if the coupling were absent. On sawyer that left the welded
object 77.6 mm from where MuJoCo rests it; building the rows INSIDE the
elliptic Newton system (`d22144ee`) brought it to 0.087 mm.

That conversion was ELLIPTIC-ONLY. The pyramidal per-env path and the
cooperative blocked kernel kept the post-pass until 2026-08-12, which is what
this file gates.

THE FIXTURE IS DEFECT 29a'S SHAPE, MINIMISED. `arm` hangs in the air held
only by a soft weld to the world; `obj` (10 kg, against the arm's 0.5 kg)
rests on its top face. The obj's weight reaches the weld THROUGH the contact,
so contact rows and weld rows act on the same dofs and have to be solved
together. Measured from MuJoCo: removing `obj` raises the arm by 9.79e-2 m, so
~98 mm of the arm's position is load carried across that coupling — the same
scale as the original sawyer defect, and the part a post-pass gets wrong.

⚠ A CONTACT MUST BE LIVE OR THIS FILE GATES NOTHING. With no contact the weld
is the only constraint, a post-pass converges to the same answer as a row
(that is exactly why the defect hid: "a quadruped in free flight matched
MuJoCo to 1e-7"), and every assertion below would pass with the bug present.
`test_the_fixture_is_not_vacuous` pins this from MuJoCo's side.

⚠ ALSO CHECK THE ARM CLEARS THE FLOOR. At heavier loads it lands on the plane,
ncon goes 4 -> 8, and the arm's position stops being set by the weld/contact
balance at all.

COVERAGE LIMIT, stated rather than implied: the blocked leg here compares
blocked-CPU against per-env-CPU, so it gates the ROW BUILDING on that kernel.
The cooperative GPU publication of `kind_e_sh` across threads is covered by
`test_newton_blocked_tendon_fields` (same mechanism, tendon rows) and not
re-tested here.

Run with:
    pixi run mojo run -I . tests/physics3d/test_weld_rows_pyramidal_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.parser.xml_parser import merge_mjcf
from mojo_rl.physics3d.fields import (
    Model,
    Data,
    DynamicsScratch,
    ContactScratch,
)
from mojo_rl.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from mojo_rl.physics3d.integrator.euler import (
    EulerIntegrator,
    _armature_env,
    _fnet_passive_env,
    _qacc_writeback_env,
)
from mojo_rl.physics3d.dynamics.subtree_com import compute_subtree_com
from mojo_rl.physics3d.dynamics.cdof import compute_cdof
from mojo_rl.physics3d.dynamics.mass_matrix import compute_mass_matrix
from mojo_rl.physics3d.dynamics.ldl import (
    ldl_factor,
    ldl_solve,
    compute_m_inv,
)
from mojo_rl.physics3d.dynamics.rne import compute_bias_forces_rne
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.solver.newton_solve import (
    solve_newton,
    solve_newton_blocked,
)
from mojo_rl.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    META_IDX_NUM_CONTACTS,
    METADATA_SIZE,
)
from mojo_rl.physics3d.types import ConeType
from layout import Layout

comptime DTYPE = DType.float64
comptime NSTEPS = 600

comptime _RAW = """
<mujoco model="weld_contact">
  <option timestep="0.002" gravity="0 0 -9.81" solver="Newton" cone="pyramidal"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body name="arm" pos="0 0 0.3">
      <freejoint/>
      <geom name="g_arm" type="box" size="0.15 0.15 0.05" mass="0.5"/>
    </body>
    <body name="obj" pos="0 0 0.42">
      <freejoint/>
      <geom name="g_obj" type="box" size="0.05 0.05 0.05" mass="10"/>
    </body>
  </worldbody>
  <equality>
    <weld name="hold" body1="arm" relpose="0 0 -0.3 1 0 0 0" solref="0.1 1"/>
  </equality>
</mujoco>
"""

comptime XML = merge_mjcf(_RAW)
comptime pm = parse_xml(XML)
comptime M = ModelDefFromXML[
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
    cone_type = ConeType.PYRAMIDAL,
    max_contacts=16,
    max_condim = pm.MAX_CONDIM,
    neq = pm.NEQ,
    max_equality = pm.NEQ,
    nexclude = pm.NEXCLUDE,
    timestep = pm.TIMESTEP,
]

# qpos = [arm(7), obj(7)]; z is index 2 of each free joint.
comptime ARM_Z = 2
comptime OBJ_Z = 9


def _mujoco_roll(drop_obj: Bool) raises -> Tuple[Float64, Float64, Int]:
    """Roll MuJoCo `NSTEPS` and return (arm_z, obj_z, ncon).

    `drop_obj` removes the obj body entirely — the control that says how much
    of the arm's position is load carried through the contact. ⚠ NOT "move the
    obj far away": at z = 3 it falls straight back onto the arm's footprint
    with more energy, which reads as a control and is not one.
    """
    var mujoco = Python.import_module("mujoco")
    var src = materialize[XML]()
    if drop_obj:
        var body_start = src.find('<body name="obj"')
        var body_end = src.find("</body>", body_start)
        src = String(src[byte=0:body_start]) + String(
            src[byte = body_end + 7 : src.byte_length()]
        )
    var m = mujoco.MjModel.from_xml_string(src)
    var dat = mujoco.MjData(m)
    for _ in range(NSTEPS):
        mujoco.mj_step(m, dat)
    var oz = Float64(0)
    if not drop_obj:
        oz = Float64(py=dat.qpos[OBJ_Z])
    return (Float64(py=dat.qpos[ARM_Z]), oz, Int(py=dat.ncon))


def test_the_fixture_is_not_vacuous() raises:
    """A weld with no live contact converges the same either way."""
    print("--- weld rows: the fixture couples a contact to the weld ---")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[XML]())
    var dat = mujoco.MjData(m)
    for _ in range(NSTEPS):
        mujoco.mj_step(m, dat)

    var ncon = Int(py=dat.ncon)
    print("  MuJoCo ncon =", ncon, " nefc =", Int(py=dat.nefc))
    assert_true(
        ncon == 4,
        "expected the obj's 4-point face contact on the arm and nothing else;"
        " got ncon = "
        + String(ncon)
        + ". At 0 the weld is unopposed and a post-pass would pass this file;"
        " at 8 the arm has landed on the floor and its position is no longer"
        " set by the weld/contact balance.",
    )
    var n_eq = 0
    for i in range(Int(py=dat.nefc)):
        if Int(py=dat.efc_type[i]) == Int(
            py=mujoco.mjtConstraint.mjCNSTR_EQUALITY
        ):
            n_eq += 1
    print("  MuJoCo equality rows =", n_eq)
    assert_true(n_eq == 6, "the weld is not contributing 6 rows")

    var held = _mujoco_roll(False)
    var unloaded = _mujoco_roll(True)
    var load_sag = unloaded[0] - held[0]
    print("  arm_z loaded  =", held[0])
    print("  arm_z unloaded=", unloaded[0])
    print("  load carried across the coupling =", load_sag, "m")
    assert_true(
        load_sag > 0.01,
        "less than 10 mm of the arm's position comes from the obj's load —"
        " the coupling this file exists to measure has gone weak, and a"
        " post-passed weld would be within tolerance",
    )
    # The arm must be clear of the plane, or the floor is holding it up.
    assert_true(
        held[0] - 0.05 > 0.02,
        "the arm's underside is within 20 mm of the floor — check ncon above;"
        " the fixture is drifting toward the regime where the plane, not the"
        " weld, sets the answer",
    )


def test_pyramidal_weld_matches_mujoco() raises:
    """Per-env PYRAMIDAL Newton, full rollout, against MuJoCo."""
    print("--- weld rows: per-env pyramidal vs MuJoCo ---")
    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DTYPE, 0](ctx, mf)

    var d = Data[DTYPE, M.NQ, M.NV, M.NBODY, M.MAX_CONTACTS, M.NSITE, 1]()
    M.reset_data[DTYPE](d)
    forward_kinematics["cpu"](d, mf)

    var integ = EulerIntegrator[
        DTYPE, M.NQ, M.NV, M.NBODY, M.NJOINT, M.MAX_CONTACTS, M.NGEOM,
        M.MAX_EQUALITY, M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
        M.CONE_TYPE, 1, SOLVER="newton",
        MAX_CONDIM = M.MAX_CONDIM, NOSLIP_ITER = M.NOSLIP_ITER,
    ]()
    for _ in range(NSTEPS):
        integ.step["cpu", CONTACTS=True](d, mf)

    var mj = _mujoco_roll(False)
    var arm = Float64(d.qpos.data[ARM_Z])
    var obj = Float64(d.qpos.data[OBJ_Z])
    print("  arm_z ours", arm, " MuJoCo", mj[0], " |d|", abs(arm - mj[0]))
    print("  obj_z ours", obj, " MuJoCo", mj[1], " |d|", abs(obj - mj[1]))
    assert_true(
        abs(arm - mj[0]) < 1e-6 and abs(obj - mj[1]) < 1e-6,
        "the pyramidal path disagrees with MuJoCo on a weld carrying a"
        " contact. As a POST-PASS this is the defect-29a failure: the contact"
        " force is computed as if the weld were absent.",
    )


def _prep(
    mut d: Data[DTYPE, M.NQ, M.NV, M.NBODY, M.MAX_CONTACTS, M.NSITE, 1],
    mut mf: Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ],
    mut sc: DynamicsScratch[DTYPE, M.NV, M.NBODY, 1],
) raises:
    """Smooth dynamics + detection, up to the constraint seam.

    Mirrors `EulerIntegrator.step`'s order so the two solvers below are handed
    the state the integrator would hand them — the same helper shape
    `test_newton_blocked_tendon_fields` uses, CPU-only.
    """
    forward_kinematics["cpu"](d, mf, None)
    compute_body_velocities["cpu"](d, mf, None)
    compute_subtree_com["cpu"](d, mf, None)
    compute_cdof["cpu"](d, mf, sc, None)
    compute_mass_matrix["cpu"](d, mf, sc, None)

    comptime L_JOINT = Layout.row_major(M.NJOINT, MODEL_JOINT_SIZE)
    comptime L_M = Layout.row_major(1, M.NV * M.NV)
    comptime L_NV = Layout.row_major(1, M.NV)
    comptime L_QPOS = Layout.row_major(1, M.NQ)

    var joints_v = mf.joints.lt["cpu", L_JOINT]()
    _armature_env[DTYPE, M.NV, M.NJOINT, 1](
        0, joints_v, sc.M.lt["cpu", L_M]()
    )
    ldl_factor["cpu", DTYPE, M.NV, M.NBODY, 1](sc, None)
    compute_m_inv["cpu", DTYPE, M.NV, M.NBODY, 1](sc, None)
    compute_bias_forces_rne["cpu"](d, mf, sc, None)
    _fnet_passive_env[DTYPE, M.NQ, M.NV, M.NJOINT, 1](
        0,
        d.qpos.lt["cpu", L_QPOS](),
        d.qvel.lt["cpu", L_NV](),
        d.qfrc.lt["cpu", L_NV](),
        joints_v,
        sc.bias.lt["cpu", L_NV](),
        sc.fnet.lt["cpu", L_NV](),
    )
    ldl_solve["cpu", DTYPE, M.NV, M.NBODY, 1](sc, None)
    _qacc_writeback_env[DTYPE, M.NV, 1](
        0,
        sc.qacc_ws.lt["cpu", L_NV](),
        d.qacc.lt["cpu", L_NV](),
        sc.qacc_constrained.lt["cpu", L_NV](),
    )
    detect_contacts["cpu"](d, mf, None)


def test_blocked_kernel_builds_the_same_weld_rows() raises:
    """Blocked cooperative kernel vs the per-env pyramidal solver, same state.

    Both build their rows from `build_weld_equality_rows`; if the blocked
    kernel had kept the post-pass — or sized `ME` without `6*NEQUALITY`, so
    the rows fell off the end of the edge list — these would diverge.
    """
    print("--- weld rows: blocked kernel vs per-env ---")
    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DTYPE, 0](ctx, mf)

    # Settle into the COUPLED state first. Comparing two solvers at the reset
    # pose would compare them with no contact live, which is the regime where
    # a post-pass and a row agree.
    var d = Data[DTYPE, M.NQ, M.NV, M.NBODY, M.MAX_CONTACTS, M.NSITE, 1]()
    M.reset_data[DTYPE](d)
    forward_kinematics["cpu"](d, mf)
    var integ = EulerIntegrator[
        DTYPE, M.NQ, M.NV, M.NBODY, M.NJOINT, M.MAX_CONTACTS, M.NGEOM,
        M.MAX_EQUALITY, M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
        M.CONE_TYPE, 1, SOLVER="newton",
        MAX_CONDIM = M.MAX_CONDIM, NOSLIP_ITER = M.NOSLIP_ITER,
    ]()
    for _ in range(NSTEPS):
        integ.step["cpu", CONTACTS=True](d, mf)

    var db = Data[DTYPE, M.NQ, M.NV, M.NBODY, M.MAX_CONTACTS, M.NSITE, 1]()
    var dp = Data[DTYPE, M.NQ, M.NV, M.NBODY, M.MAX_CONTACTS, M.NSITE, 1]()
    M.reset_data[DTYPE](db)
    M.reset_data[DTYPE](dp)
    for i in range(M.NQ):
        db.qpos.data[i] = d.qpos.data[i]
        dp.qpos.data[i] = d.qpos.data[i]
    for i in range(M.NV):
        db.qvel.data[i] = d.qvel.data[i]
        dp.qvel.data[i] = d.qvel.data[i]

    var sb = DynamicsScratch[DTYPE, M.NV, M.NBODY, 1]()
    var sp = DynamicsScratch[DTYPE, M.NV, M.NBODY, 1]()
    var cb = ContactScratch[DTYPE, M.NV, M.MAX_CONTACTS, 1]()
    var cp = ContactScratch[DTYPE, M.NV, M.MAX_CONTACTS, 1]()
    _prep(db, mf, sb)
    _prep(dp, mf, sp)

    var ncon = Int(db.meta.data[META_IDX_NUM_CONTACTS])
    print("  contacts in the prepared state:", ncon)
    assert_true(
        ncon > 0,
        "no contacts in the prepared state — the two solvers are being"
        " compared in the regime where a post-passed weld and a weld row"
        " agree, so this comparison would pass with the bug present",
    )

    solve_newton_blocked[
        "cpu", DTYPE, M.NQ, M.NV, M.NBODY, M.NJOINT, M.MAX_CONTACTS, M.NGEOM,
        M.MAX_EQUALITY, M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
        ConeType.PYRAMIDAL, 1,
    ](db, mf, sb, cb, None)
    solve_newton[
        "cpu", DTYPE, M.NQ, M.NV, M.NBODY, M.NJOINT, M.MAX_CONTACTS, M.NGEOM,
        M.MAX_EQUALITY, M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
        ConeType.PYRAMIDAL, 1,
    ](dp, mf, sp, cp, None)

    var worst = Float64(0)
    for i in range(M.NV):
        var e = abs(
            Float64(sb.qacc_constrained.data[i])
            - Float64(sp.qacc_constrained.data[i])
        )
        if e > worst:
            worst = e
    print("  worst |d(qacc)| blocked vs per-env =", worst)
    assert_true(
        worst < 1e-9,
        "the blocked kernel and the per-env pyramidal solver disagree on the"
        " weld rows (|d| " + String(worst) + ")",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

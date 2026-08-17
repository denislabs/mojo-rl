"""Behavioral gate: ACTIVE joint limits on the fields path vs LIVE MuJoCo,
on a tendon-carrying Humanoid (meta NTENDON=2 + records injected).

Why this gate exists: the legacy joint-limit builder
(detect_and_solve_limits_gpu, constraints/constraint_builder_gpu.mojo:984)
computes its dof_invweight0 model offset with NTENDON/NSITE defaulted to 0,
so on models whose slab carries tendon records (Humanoid MAX_TENDON=2) it
reads UNRELATED slots. The fields port (constraints/limits.mojo)
reads the correct per-field tensor. Decision: fix-forward — the fields
behavior is declared intended, and this gate validates it against MuJoCo
(the ground truth) instead of against the legacy misread.

Setup: Humanoid at float64, FLOATING (torso z=3 -> zero contacts), right
elbow pushed past its lower range (-1.62 rad vs -pi/2) with a closing
velocity -> exactly ONE active limit row. No XML model carries tendon
records (tendon XML parsing was removed from the parser), so the two
hip-knee tendon records + the meta NTENDON count are injected DIRECTLY into
the per-field tendon tensor after init_fields (no slab, no load_from_slab),
mirroring tests/physics3d/test_equality_tendon_fields.mojo Part A (fields
side only — this gate has no legacy leg). The MuJoCo reference gets the
matching physics via an <equality><tendon> block appended to the Gymnasium
humanoid.xml (its <fixed> tendons are dynamically inert on their own; our
tendon record is a bilateral length constraint ten_length - length_ref = 0,
which is MuJoCo's mjEQ_TENDON with a single tendon:
residual = L - tendon_length0 - polycoef[0], with tendon_length0 = 0 at
humanoid qpos0). length_ref = polycoef[0] = -0.15 = the tendon length at
the test pose, so the tendon rows start at zero residual and stay small —
they must be present (that is what shifts the legacy limit builder's
misread) but the row under test is the elbow LIMIT.

Both sides step RK4 (fields RK4Integrator[SOLVER="newton"], CPU,
BATCH=1, matching the legacy humanoid gate's RK4+Newton, vs mujoco.mj_step
with opt.integrator=1, solver=2 Newton, cone=1 — the exact option set of
tests/physics3d/test_humanoid_full_step_vs_mujoco.mojo), and qpos/qvel are
compared with that gate's tolerances (abs OR rel per entry), over 1 and 10
steps. This is a BEHAVIORAL gate, not bit-exact.

Non-vacuity: (a) MuJoCo nefc must be exactly 3 every step (2 always-active
tendon-equality rows + 1 elbow limit row) and ncon == 0 — the only
inequality row is the joint limit under test; (b) our own violated-limit
scan over the fields joint records must find exactly the elbow row (like
test_newton_solve_fields' violated-limit count); (c) dof_invweight0 at the
elbow dof must be > 0 so R_lim actually flows through the address under
test rather than the m_inv fallback; (d) a rerun with the elbow range
widened to "unlimited" (rmin < -1e9 -> the fields builder skips the joint)
must change qvel after one step.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_humanoid_limits_fields_vs_mujoco.mojo
"""

from std.testing import assert_true, TestSuite
from std.python import Python
from std.math import abs
from std.collections import InlineArray
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.fields import Data, Model, Dims
from mojo_rl.physics3d.integrator.rk4 import RK4Integrator
from mojo_rl.physics3d.joint_types import JNT_HINGE, JNT_SLIDE
from mojo_rl.physics3d.gpu.constants import (
    MODEL_TENDON_SIZE,
    MODEL_META_IDX_NTENDON,
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    META_IDX_NUM_CONTACTS,
    TENDON_IDX_NUM_JOINTS,
    TENDON_IDX_IS_EQUALITY,
    TENDON_IDX_JOINT_0,
    TENDON_IDX_JOINT_1,
    TENDON_IDX_JOINT_2,
    TENDON_IDX_JOINT_3,
    TENDON_IDX_COEF_0,
    TENDON_IDX_COEF_1,
    TENDON_IDX_COEF_2,
    TENDON_IDX_COEF_3,
    TENDON_IDX_LENGTH_REF,
    TENDON_IDX_SOLREF_0,
    TENDON_IDX_SOLREF_1,
    TENDON_IDX_SOLIMP_0,
    TENDON_IDX_SOLIMP_1,
    TENDON_IDX_SOLIMP_2,
    TENDON_IDX_SOLIMP_3,
    TENDON_IDX_SOLIMP_4,
    METADATA_SIZE,
)
from mojo_rl.envs.humanoid.humanoid_xml import HumanoidModel

comptime DTYPE = DType.float64
comptime NQ = HumanoidModel.NQ  # 24
comptime NV = HumanoidModel.NV  # 23
comptime NBODY = HumanoidModel.NBODY  # 14
comptime NJOINT = HumanoidModel.NJOINT  # 18
comptime NGEOM = HumanoidModel.NGEOM  # 18
comptime MC = HumanoidModel.MAX_CONTACTS  # 50
comptime NEQ = HumanoidModel.MAX_EQUALITY  # 0
comptime NTEN = HumanoidModel.MAX_TENDON  # 2
comptime NSITE = HumanoidModel.NSITE  # 0
comptime NEXCL = HumanoidModel.nexclude  # 0
comptime MD = Dims[
    nq=NQ,
    nv=NV,
    nbody=NBODY,
    njoint=NJOINT,
    ngeom=NGEOM,
    nsite=NSITE,
    max_contacts=MC,
    nequality=NEQ,
    ntendon=NTEN,
    nexclude=NEXCL,
    nmesh_verts=0,
    npair=HumanoidModel.NPAIR,
    nact=HumanoidModel.NACT,
    nten=HumanoidModel.NTEN_F,
    nkey=HumanoidModel.NKEY,
]
comptime CONE = HumanoidModel.CONE_TYPE

# Tendon length at the test pose (hip_y=0, knee=-0.15; L = -hip_y + knee):
# length_ref here == polycoef[0] in the MuJoCo equality block below.
comptime TENDON_LENGTH_REF: Float64 = -0.15

# Elbow limit test point: right_elbow range is [-90, 50] deg =
# [-1.5708, 0.8727] rad; -1.62 violates the lower bound by ~0.049 rad.
comptime ELBOW_QPOS_ADR = 20
comptime ELBOW_QPOS: Float64 = -1.62
comptime ELBOW_QVEL: Float64 = -0.3  # closing (deeper into violation)

# Same tolerance style as test_humanoid_full_step_vs_mujoco.mojo.
comptime QPOS_ABS_TOL: Float64 = 1e-3
comptime QPOS_REL_TOL: Float64 = 1e-2
comptime QVEL_ABS_TOL: Float64 = 1e-2
comptime QVEL_REL_TOL: Float64 = 1e-2


def _build_model(
    ctx: DeviceContext,
) raises -> Model[DTYPE, MD]:
    """Offset-free init_fields build, then inject meta NTENDON=2 + the two
    hip-knee tendon records DIRECTLY into the per-field tendon tensor
    (record layout t_i * MODEL_TENDON_SIZE + TENDON_IDX_*) exactly as
    test_equality_tendon_fields Part A does — the parser never emits tendon
    records, so injection is the only way any model carries them. No slab,
    no load_from_slab."""
    var mf = Model[DTYPE, MD]()
    HumanoidModel.init_fields[DTYPE](ctx, mf)

    # right = r_hip_y (joint 6) + r_knee (joint 7); left = l_hip_y (joint 10)
    # + l_knee (joint 11); coef -1 * hip_y + 1 * knee, MuJoCo-default
    # solref/solimp.
    mf.meta.data[MODEL_META_IDX_NTENDON] = Scalar[DTYPE](2)
    for t_i in range(2):
        var t_off = t_i * MODEL_TENDON_SIZE
        var j0 = 6 if t_i == 0 else 10
        # `_tendon_env` imposes a BILATERAL EQUALITY, and since
        # 2026-07-31 it only acts on records that say so. That gate
        # exists because `fields_build` now populates `ntendon`
        # honestly, and humanoid's <fixed> tendons are NOT constrained
        # by MuJoCo — without it, every humanoid hip-knee pair would be
        # welded. This test's whole subject IS the equality path, so it
        # opts in explicitly.
        mf.tendons.data[t_off + TENDON_IDX_IS_EQUALITY] = Scalar[DTYPE](1)
        mf.tendons.data[t_off + TENDON_IDX_NUM_JOINTS] = Scalar[DTYPE](2)
        mf.tendons.data[t_off + TENDON_IDX_JOINT_0] = Scalar[DTYPE](j0)
        mf.tendons.data[t_off + TENDON_IDX_JOINT_1] = Scalar[DTYPE](j0 + 1)
        mf.tendons.data[t_off + TENDON_IDX_JOINT_2] = Scalar[DTYPE](-1)
        mf.tendons.data[t_off + TENDON_IDX_JOINT_3] = Scalar[DTYPE](-1)
        mf.tendons.data[t_off + TENDON_IDX_COEF_0] = Scalar[DTYPE](-1)
        mf.tendons.data[t_off + TENDON_IDX_COEF_1] = Scalar[DTYPE](1)
        mf.tendons.data[t_off + TENDON_IDX_COEF_2] = Scalar[DTYPE](0)
        mf.tendons.data[t_off + TENDON_IDX_COEF_3] = Scalar[DTYPE](0)
        mf.tendons.data[t_off + TENDON_IDX_LENGTH_REF] = Scalar[DTYPE](
            TENDON_LENGTH_REF
        )
        mf.tendons.data[t_off + TENDON_IDX_SOLREF_0] = Scalar[DTYPE](0.02)
        mf.tendons.data[t_off + TENDON_IDX_SOLREF_1] = Scalar[DTYPE](1)
        mf.tendons.data[t_off + TENDON_IDX_SOLIMP_0] = Scalar[DTYPE](0.9)
        mf.tendons.data[t_off + TENDON_IDX_SOLIMP_1] = Scalar[DTYPE](0.95)
        mf.tendons.data[t_off + TENDON_IDX_SOLIMP_2] = Scalar[DTYPE](0.001)
        mf.tendons.data[t_off + TENDON_IDX_SOLIMP_3] = Scalar[DTYPE](0.5)
        mf.tendons.data[t_off + TENDON_IDX_SOLIMP_4] = Scalar[DTYPE](2)
    mf.tendons.upload(ctx)
    mf.meta.upload(ctx)
    return mf^


def _find_elbow_joint(
    mf: Model[DTYPE, MD],
) raises -> Tuple[Int, Int]:
    """Locate the right-elbow joint record (qpos_adr == 20) and return
    (joint index, dof_adr); assert its range is the expected [-90, 50] deg."""
    var elbow_j = -1
    var elbow_dof = -1
    for j in range(NJOINT):
        var qadr = Int(mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_QPOS_ADR])
        var jtype = Int(mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_TYPE])
        if qadr == ELBOW_QPOS_ADR and jtype == JNT_HINGE:
            elbow_j = j
            elbow_dof = Int(
                mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_DOF_ADR]
            )
    assert_true(elbow_j >= 0, "right elbow joint record not found")
    var rmin = Float64(
        mf.joints.data[elbow_j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MIN]
    )
    var rmax = Float64(
        mf.joints.data[elbow_j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MAX]
    )
    assert_true(
        abs(rmin - (-1.5707963)) < 1e-3 and abs(rmax - 0.8726646) < 1e-3,
        "elbow range is not the expected [-90, 50] deg",
    )
    return (elbow_j, elbow_dof)


def _pose_qpos() -> InlineArray[Float64, NQ]:
    """Floating (z=3, identity quat), knees at -0.15 (inside their
    [-160, -2] deg range; 0 would violate it), right elbow past its lower
    limit; every other joint at 0 sits strictly inside its range."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[2] = 3.0  # torso z — contact-free
    qpos[3] = 1.0  # identity quaternion (w first)
    qpos[13] = -0.15  # right knee
    qpos[17] = -0.15  # left knee
    qpos[ELBOW_QPOS_ADR] = ELBOW_QPOS  # right elbow — VIOLATED
    return qpos^


def _pose_qvel() -> InlineArray[Float64, NV]:
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[19] = ELBOW_QVEL  # right elbow dof, closing into the limit
    return qvel^


comptime MJ_EQUALITY_BLOCK = """    <equality>
        <tendon tendon1="right_hipknee" polycoef="-0.15 0 0 0 0" solref="0.02 1" solimp="0.9 0.95 0.001 0.5 2"/>
        <tendon tendon1="left_hipknee" polycoef="-0.15 0 0 0 0" solref="0.02 1" solimp="0.9 0.95 0.001 0.5 2"/>
    </equality>
</mujoco>"""


def _compare_vs_mujoco(num_steps: Int) raises:
    print(
        "--- fields RK4+Newton active elbow limit vs MuJoCo,",
        num_steps,
        "steps ---",
    )
    var ctx = DeviceContext()
    var mf = _build_model(ctx)
    var elbow = _find_elbow_joint(mf)
    var elbow_dof = elbow[1]

    # (c) the address under test must carry a real value (no m_inv fallback).
    var dw_elbow = Float64(mf.dof_invweight0.data[elbow_dof])
    print("  dof_invweight0[elbow dof", elbow_dof, "] =", dw_elbow)
    assert_true(
        dw_elbow > 1e-10,
        "dof_invweight0[elbow] is zero — the R_lim path under test is dead",
    )

    var qpos_init = _pose_qpos()
    var qvel_init = _pose_qvel()

    # (b) violated-limit scan (fields records): exactly the elbow row.
    var n_violated = 0
    var violated_is_elbow = False
    for j in range(NJOINT):
        var jtype = Int(mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_TYPE])
        if jtype != JNT_HINGE and jtype != JNT_SLIDE:
            continue
        var rmin = Float64(
            mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MIN]
        )
        var rmax = Float64(
            mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MAX]
        )
        if rmin < -1e9 or rmax > 1e9:
            continue
        var qadr = Int(
            mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_QPOS_ADR]
        )
        var pos = qpos_init[qadr]
        if pos - rmin < 0.0 or rmax - pos < 0.0:
            n_violated += 1
            if qadr == ELBOW_QPOS_ADR:
                violated_is_elbow = True
    print("  violated limit rows at init pose:", n_violated)
    assert_true(
        n_violated == 1 and violated_is_elbow,
        "expected exactly one violated limit (the right elbow)",
    )

    # ── Fields path (f64, CPU, BATCH=1, RK4 + Newton like the legacy gate).
    var d = Data[DTYPE, MD, 1]()
    for i in range(NQ):
        d.qpos.data[i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        d.qvel.data[i] = Scalar[DTYPE](qvel_init[i])
    var integ = RK4Integrator[DTYPE, MD, CONE, BATCH=1, SOLVER="newton"]()
    for _ in range(num_steps):
        for i in range(NV):
            d.qfrc.data[i] = Scalar[DTYPE](0)
        integ.step["cpu"](d, mf)
    var our_ncon = Int(d.meta.data[META_IDX_NUM_CONTACTS])
    assert_true(our_ncon == 0, "fields path detected contacts — pose not free")

    # ── Live MuJoCo reference: Gymnasium humanoid.xml (whose <tendon>
    # section defines the same two fixed tendons) + the equality block that
    # matches the injected records.
    var mujoco = Python.import_module("mujoco")
    var builtins = Python.import_module("builtins")
    var xml_path = (
        "./references/Gymnasium-main/gymnasium/envs/mujoco/assets/humanoid.xml"
    )
    var fh = builtins.open(xml_path)
    var xml_text = fh.read()
    _ = fh.close()
    var xml_new = xml_text.replace("</mujoco>", MJ_EQUALITY_BLOCK)
    var mj_model = mujoco.MjModel.from_xml_string(xml_new)
    mj_model.opt.integrator = 1  # RK4
    mj_model.opt.solver = 2  # Newton
    mj_model.opt.cone = 1  # Elliptic
    var mj_data = mujoco.MjData(mj_model)
    for i in range(NQ):
        mj_data.qpos[i] = qpos_init[i]
    for i in range(NV):
        mj_data.qvel[i] = qvel_init[i]
    for step in range(num_steps):
        mujoco.mj_step(mj_model, mj_data)
        # (a) constraint accounting: 2 tendon-equality rows (always active)
        # + exactly 1 joint-limit row (the elbow); zero contacts.
        var ncon = Int(py=mj_data.ncon)
        var nefc = Int(py=mj_data.nefc)
        if step == 0 or step == num_steps - 1:
            print("  mj step", step, ": ncon =", ncon, " nefc =", nefc)
        assert_true(ncon == 0, "MuJoCo found contacts — pose not contact-free")
        assert_true(
            nefc == 3,
            "MuJoCo nefc != 3 (2 tendon eq rows + 1 elbow limit row)",
        )

    var mj_qpos = mj_data.qpos.flatten().tolist()
    var mj_qvel = mj_data.qvel.flatten().tolist()

    # ── Compare (abs OR rel per entry, humanoid-gate style).
    var all_pass = True
    var qpos_max_abs: Float64 = 0.0
    var qvel_max_abs: Float64 = 0.0
    for i in range(NQ):
        var ours = Float64(d.qpos.data[i])
        var mj = Float64(py=mj_qpos[i])
        var abs_err = abs(ours - mj)
        var rel_err: Float64 = 0.0
        if abs(mj) > 1e-10:
            rel_err = abs_err / abs(mj)
        if abs_err > qpos_max_abs:
            qpos_max_abs = abs_err
        if not (abs_err < QPOS_ABS_TOL or rel_err < QPOS_REL_TOL):
            print(
                "  FAIL qpos[", i, "] ours=", ours, " mj=", mj,
                " abs=", abs_err, " rel=", rel_err,
            )
            all_pass = False
    for i in range(NV):
        var ours = Float64(d.qvel.data[i])
        var mj = Float64(py=mj_qvel[i])
        var abs_err = abs(ours - mj)
        var rel_err: Float64 = 0.0
        if abs(mj) > 1e-10:
            rel_err = abs_err / abs(mj)
        if abs_err > qvel_max_abs:
            qvel_max_abs = abs_err
        if not (abs_err < QVEL_ABS_TOL or rel_err < QVEL_REL_TOL):
            print(
                "  FAIL qvel[", i, "] ours=", ours, " mj=", mj,
                " abs=", abs_err, " rel=", rel_err,
            )
            all_pass = False
    print(
        "  elbow qpos: ours=", Float64(d.qpos.data[ELBOW_QPOS_ADR]),
        " mj=", Float64(py=mj_qpos[ELBOW_QPOS_ADR]),
        "  elbow qvel: ours=", Float64(d.qvel.data[19]),
        " mj=", Float64(py=mj_qvel[19]),
    )
    print(
        "  worst |qpos err| =", qpos_max_abs,
        " worst |qvel err| =", qvel_max_abs,
    )
    assert_true(all_pass, "fields active-limit step diverged from MuJoCo")


def test_active_elbow_limit_vs_mujoco_1_step() raises:
    _compare_vs_mujoco(1)


def test_active_elbow_limit_vs_mujoco_10_steps() raises:
    _compare_vs_mujoco(10)


def test_limit_off_rerun_differs() raises:
    """(d) Same pose, elbow range widened to unlimited (rmin < -1e9 -> the
    fields limits builder skips the joint): one step must differ in qvel —
    proves the limit row genuinely shaped the gated trajectory."""
    print("--- non-vacuity: elbow limit on vs off, 1 fields step ---")
    var ctx = DeviceContext()
    var mf = _build_model(ctx)
    var mf_off = _build_model(ctx)
    var elbow = _find_elbow_joint(mf)
    var elbow_j = elbow[0]
    mf_off.joints.data[
        elbow_j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MIN
    ] = Scalar[DTYPE](-1e10)

    var qpos_init = _pose_qpos()
    var qvel_init = _pose_qvel()
    var d_on = Data[DTYPE, MD, 1]()
    var d_off = Data[DTYPE, MD, 1]()
    for i in range(NQ):
        d_on.qpos.data[i] = Scalar[DTYPE](qpos_init[i])
        d_off.qpos.data[i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        d_on.qvel.data[i] = Scalar[DTYPE](qvel_init[i])
        d_off.qvel.data[i] = Scalar[DTYPE](qvel_init[i])
        d_on.qfrc.data[i] = Scalar[DTYPE](0)
        d_off.qfrc.data[i] = Scalar[DTYPE](0)
    var integ_on = RK4Integrator[DTYPE, MD, CONE, BATCH=1, SOLVER="newton"]()
    var integ_off = RK4Integrator[DTYPE, MD, CONE, BATCH=1, SOLVER="newton"]()
    integ_on.step["cpu"](d_on, mf)
    integ_off.step["cpu"](d_off, mf_off)
    var ndiff = 0
    var max_diff: Float64 = 0.0
    for i in range(NV):
        var diff = abs(Float64(d_on.qvel.data[i]) - Float64(d_off.qvel.data[i]))
        if diff > 1e-12:
            ndiff += 1
        if diff > max_diff:
            max_diff = diff
    print(
        "  limit-on vs limit-off qvel entries differing:", ndiff,
        " max |diff| =", max_diff,
    )
    assert_true(
        ndiff > 0 and max_diff > 1e-6,
        "limit-off rerun identical to limit-on — the limit row was vacuous",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

"""Differential gate: the fields engine vs LIVE MuJoCo with constraints ACTIVE.

Why this gate exists: until it was written, NOTHING compared the engine to
MuJoCo with a constraint row active. test_euler_fields_vs_mujoco asserts
`nefc == 0` on its free-flight cases and pins its active-limit case to a
frozen golden of our OWN output, so a systematic error in the constraint
response could not be seen — and one was there. `dof_invweight0` (MuJoCo's
`mj_diagApprox` for joint limits, engine_core_constraint.c:1121) was built at
the ENV RESET pose instead of qpos0, and skipped the free/ball dof-group
averaging of `mj_setConst` (engine_setconst.c:199-209). The result was a
~1% multiplicative error on every joint-limit force, invisible in free flight.

Three parts:
  A. INVERSE WEIGHTS — mf.dof_invweight0 / mf.body_invweight0 vs MuJoCo's
     m.dof_invweight0 / m.body_invweight0. This is the direct root-cause
     guard: it fails on the exact quantity that was wrong, not on a
     downstream symptom.
  B. JOINT LIMITS active, contact-free — ant parked past its ankle limits at
     three violation depths. The pre-fix error was multiplicative (constant
     ~1% relative across a 300x depth range), so sweeping depth is what
     distinguishes a scale error from noise.
  C. CONTACTS active — hopper dropped onto the floor.

Both sides step Euler (fields EulerIntegrator[SOLVER="newton"], CPU, BATCH=1,
vs mujoco.mj_step with opt.integrator=0). Non-vacuity is asserted throughout:
MuJoCo's nefc/ncon must be > 0 on the steps being compared, otherwise the
budgets below are gating free flight and prove nothing.

Run: pixi run mojo run -I . tests/physics3d/test_constraints_vs_mujoco.mojo
"""

from std.testing import assert_true, TestSuite
from std.python import Python, PythonObject
from std.math import abs
from std.collections import InlineArray
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.envs.ant.ant_xml import AntModel, ant_xml
from mojo_rl.envs.hopper.hopper_xml import HopperModel, hopper_xml
from mojo_rl.envs.humanoid.humanoid_xml import HumanoidModel, humanoid_xml

comptime DTYPE = DType.float64

# Inverse weights are a closed-form model constant on both sides; they agree
# to round-off, not to a behavioral budget.
comptime IW_REL_TOL: Float64 = 1e-9
# Constrained lockstep budgets. Pre-fix the limit case ran 9.4e-6 (0.001 rad
# overshoot) to 2.8e-3 (0.3 rad); these sit far below that and above the
# ~1e-10 the fixed engine actually achieves.
comptime LIMIT_QVEL_TOL: Float64 = 1e-8
comptime LIMIT_QPOS_TOL: Float64 = 1e-9
comptime CONTACT_QVEL_TOL: Float64 = 1e-7
comptime CONTACT_QPOS_TOL: Float64 = 1e-8


def _check_invweights(
    label: String,
    xml: StaticString,
    dof_iw: List[Float64],
    body_iw: List[Float64],
    nv: Int,
    nbody: Int,
) raises:
    """Compare our build-time inverse weights against MuJoCo's own."""
    var mujoco = Python.import_module("mujoco")
    var mj_model = mujoco.MjModel.from_xml_string(String(xml))

    var mj_dof = mj_model.dof_invweight0.flatten().tolist()
    var worst_dof = Float64(0)
    var worst_dof_i = 0
    for i in range(nv):
        var mref = Float64(py=mj_dof[i])
        var rel = abs(dof_iw[i] - mref) / (1.0 + abs(mref))
        if rel > worst_dof:
            worst_dof = rel
            worst_dof_i = i
    print(
        "  [", label, "] dof_invweight0 worst rel err", worst_dof,
        "at dof", worst_dof_i,
        " (ours", dof_iw[worst_dof_i], "vs mj", Float64(py=mj_dof[worst_dof_i]),
        ")",
    )

    var mj_body = mj_model.body_invweight0.flatten().tolist()
    var worst_body = Float64(0)
    var worst_body_i = 0
    for i in range(2 * nbody):
        var mref = Float64(py=mj_body[i])
        var rel = abs(body_iw[i] - mref) / (1.0 + abs(mref))
        if rel > worst_body:
            worst_body = rel
            worst_body_i = i
    print(
        "  [", label, "] body_invweight0 worst rel err", worst_body,
        "at index", worst_body_i,
        " (ours", body_iw[worst_body_i],
        "vs mj", Float64(py=mj_body[worst_body_i]), ")",
    )

    assert_true(
        worst_dof < IW_REL_TOL,
        String(label)
        + ": dof_invweight0 disagrees with MuJoCo — every joint-limit and"
        " tendon-equality force is scaled wrong",
    )
    assert_true(
        worst_body < IW_REL_TOL,
        String(label)
        + ": body_invweight0 disagrees with MuJoCo — every contact force is"
        " scaled wrong",
    )


def test_invweight0_vs_mujoco() raises:
    """Root-cause guard: build-time inverse weights must equal MuJoCo's.

    Ant is the discriminating model: it carries a <custom> init_qpos that
    bends its ankles, so a build at the reset pose instead of qpos0 shows up
    here (0.75% on the hinges) and nowhere else. Humanoid isolates the
    free-joint dof-group averaging (its reset pose IS qpos0, so only the
    averaging can differ). Hopper has neither and must stay exact.
    """
    var ctx = DeviceContext()

    comptime A = AntModel
    var mfa = Model[
        DTYPE, A.NV, A.NBODY, A.NJOINT, A.NGEOM, A.MAX_EQUALITY,
        A.MAX_TENDON, A.NSITE, A.NEXCLUDE, 0,
    ]()
    A.init_fields[DTYPE, 0](ctx, mfa)
    var dwa = List[Float64]()
    for i in range(A.NV):
        dwa.append(Float64(mfa.dof_invweight0.data[i]))
    var bwa = List[Float64]()
    for i in range(2 * A.NBODY):
        bwa.append(Float64(mfa.body_invweight0.data[i]))
    _check_invweights("ant", ant_xml, dwa, bwa, A.NV, A.NBODY)

    comptime U = HumanoidModel
    var mfu = Model[
        DTYPE, U.NV, U.NBODY, U.NJOINT, U.NGEOM, U.MAX_EQUALITY,
        U.MAX_TENDON, U.NSITE, U.NEXCLUDE, 0,
    ]()
    U.init_fields[DTYPE, 0](ctx, mfu)
    var dwu = List[Float64]()
    for i in range(U.NV):
        dwu.append(Float64(mfu.dof_invweight0.data[i]))
    var bwu = List[Float64]()
    for i in range(2 * U.NBODY):
        bwu.append(Float64(mfu.body_invweight0.data[i]))
    _check_invweights("humanoid", humanoid_xml, dwu, bwu, U.NV, U.NBODY)

    comptime H = HopperModel
    var mfh = Model[
        DTYPE, H.NV, H.NBODY, H.NJOINT, H.NGEOM, H.MAX_EQUALITY,
        H.MAX_TENDON, H.NSITE, H.NEXCLUDE, 0,
    ]()
    H.init_fields[DTYPE, 0](ctx, mfh)
    var dwh = List[Float64]()
    for i in range(H.NV):
        dwh.append(Float64(mfh.dof_invweight0.data[i]))
    var bwh = List[Float64]()
    for i in range(2 * H.NBODY):
        bwh.append(Float64(mfh.body_invweight0.data[i]))
    _check_invweights("hopper", hopper_xml, dwh, bwh, H.NV, H.NBODY)


def _ant_limits(num_steps: Int, overshoot: Float64) raises:
    """Lockstep ant vs MuJoCo with every limited ankle parked `overshoot`
    radians past its own upper limit, torso high enough for zero contacts."""
    comptime M = AntModel
    var mujoco = Python.import_module("mujoco")
    var mj_model = mujoco.MjModel.from_xml_string(String(ant_xml))
    mj_model.opt.integrator = 0
    var mj_data = mujoco.MjData(mj_model)

    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DTYPE, 0](ctx, mf)
    var d = Data[DTYPE, M.NQ, M.NV, M.NBODY, M.MAX_CONTACTS, M.NSITE, 1]()

    var q = InlineArray[Float64, M.NQ](fill=0.0)
    q[2] = 2.0  # torso high — contact-free, so limits are the ONLY rows
    q[3] = 0.9659258262890683  # tilted root: a wrong free-dof weight shows up
    q[5] = 0.25881904510252074
    var jr = mj_model.jnt_range.tolist()
    var ja = mj_model.jnt_qposadr.tolist()
    for j in range(Int(py=mj_model.njnt)):
        var adr = Int(py=ja[j])
        if adr < 7:
            continue
        var lo = Float64(py=jr[j][0])
        var hi = Float64(py=jr[j][1])
        if hi <= lo:
            continue
        q[adr] = hi + overshoot
    for i in range(M.NQ):
        d.qpos.data[i] = Scalar[DTYPE](q[i])
        mj_data.qpos[i] = q[i]
    var v = InlineArray[Float64, M.NV](fill=0.0)
    v[3] = 2.0
    v[4] = 1.0
    v[5] = 0.5
    for i in range(M.NV):
        d.qvel.data[i] = Scalar[DTYPE](v[i])
        mj_data.qvel[i] = v[i]

    var integ = EulerIntegrator[
        DTYPE, M.NQ, M.NV, M.NBODY, M.NJOINT, M.MAX_CONTACTS, M.NGEOM,
        M.MAX_EQUALITY, M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
        M.CONE_TYPE, 1, SOLVER="newton",
    ]()

    var worst_q = Float64(0)
    var worst_v = Float64(0)
    var active_steps = 0
    var max_nefc = 0
    for _s in range(num_steps):
        for i in range(M.NV):
            d.qfrc.data[i] = Scalar[DTYPE](0)
        integ.step["cpu"](d, mf)
        mujoco.mj_step(mj_model, mj_data)
        var nefc = Int(py=mj_data.nefc)
        if nefc > 0:
            active_steps += 1
        if nefc > max_nefc:
            max_nefc = nefc
        assert_true(
            Int(py=mj_data.ncon) == 0,
            "ant limit case made contact — the case is no longer limit-only",
        )
        var mq = mj_data.qpos.flatten().tolist()
        var mv = mj_data.qvel.flatten().tolist()
        for i in range(M.NQ):
            var e = abs(Float64(d.qpos.data[i]) - Float64(py=mq[i]))
            if e > worst_q:
                worst_q = e
        for i in range(M.NV):
            var e = abs(Float64(d.qvel.data[i]) - Float64(py=mv[i]))
            if e > worst_v:
                worst_v = e
    print(
        "  overshoot", overshoot, "rad: max |d qpos| =", worst_q,
        " max |d qvel| =", worst_v, " (steps with an active row =",
        active_steps, "/", num_steps, ", max nefc =", max_nefc, ")",
    )
    # The limits push the ankles back inside their range, so the rows go
    # inactive partway through the window — the requirement is that they were
    # active while it mattered, not that they stayed active.
    assert_true(
        active_steps >= 3 and max_nefc >= 4,
        "too few steps had an active limit row — this case gates free flight",
    )
    assert_true(worst_q < LIMIT_QPOS_TOL, "ant qpos diverged under active limits")
    assert_true(worst_v < LIMIT_QVEL_TOL, "ant qvel diverged under active limits")


def test_joint_limits_vs_mujoco() raises:
    """Active joint limits, swept over violation depth.

    The pre-fix error was a pure SCALE error on the constraint force: ~1%
    relative at every depth. Sweeping 0.001 -> 0.3 rad (a 300x range) is what
    separates that signature from ordinary integration noise, which would not
    hold a constant ratio.
    """
    _ant_limits(8, 0.001)
    _ant_limits(8, 0.02)
    _ant_limits(8, 0.3)


def test_contacts_vs_mujoco() raises:
    """Active contacts: hopper dropped onto the floor, lockstep vs MuJoCo."""
    comptime M = HopperModel
    var mujoco = Python.import_module("mujoco")
    var mj_model = mujoco.MjModel.from_xml_string(String(hopper_xml))
    mj_model.opt.integrator = 0
    var mj_data = mujoco.MjData(mj_model)

    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DTYPE, 0](ctx, mf)
    var d = Data[DTYPE, M.NQ, M.NV, M.NBODY, M.MAX_CONTACTS, M.NSITE, 1]()

    # z=0.95 puts the foot on the floor immediately; from the 1.25 rest height
    # it never lands inside the step budget and the case goes vacuous.
    var q = InlineArray[Float64, M.NQ](fill=0.0)
    q[1] = 0.95
    for i in range(M.NQ):
        d.qpos.data[i] = Scalar[DTYPE](q[i])
        mj_data.qpos[i] = q[i]
    for i in range(M.NV):
        d.qvel.data[i] = Scalar[DTYPE](0)
        mj_data.qvel[i] = 0.0

    var integ = EulerIntegrator[
        DTYPE, M.NQ, M.NV, M.NBODY, M.NJOINT, M.MAX_CONTACTS, M.NGEOM,
        M.MAX_EQUALITY, M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
        M.CONE_TYPE, 1, SOLVER="newton",
    ]()

    var worst_q = Float64(0)
    var worst_v = Float64(0)
    var contact_steps = 0
    var max_ncon = 0
    for _s in range(60):
        for i in range(M.NV):
            d.qfrc.data[i] = Scalar[DTYPE](0)
        integ.step["cpu"](d, mf)
        mujoco.mj_step(mj_model, mj_data)
        var ncon = Int(py=mj_data.ncon)
        if ncon > 0:
            contact_steps += 1
        if ncon > max_ncon:
            max_ncon = ncon
        var mq = mj_data.qpos.flatten().tolist()
        var mv = mj_data.qvel.flatten().tolist()
        for i in range(M.NQ):
            var e = abs(Float64(d.qpos.data[i]) - Float64(py=mq[i]))
            if e > worst_q:
                worst_q = e
        for i in range(M.NV):
            var e = abs(Float64(d.qvel.data[i]) - Float64(py=mv[i]))
            if e > worst_v:
                worst_v = e
    print(
        "  hopper contacts: max |d qpos| =", worst_q, " max |d qvel| =",
        worst_v, " (contact steps =", contact_steps, ", max ncon =",
        max_ncon, ")",
    )
    # The hopper lands, compresses and rebounds, so contact is intermittent
    # across the window rather than continuous.
    assert_true(
        contact_steps >= 20 and max_ncon >= 3,
        "hopper never established sustained contact — the case is vacuous",
    )
    assert_true(worst_q < CONTACT_QPOS_TOL, "hopper qpos diverged under contact")
    assert_true(worst_v < CONTACT_QVEL_TOL, "hopper qvel diverged under contact")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

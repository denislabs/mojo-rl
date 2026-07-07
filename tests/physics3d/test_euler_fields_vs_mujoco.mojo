"""P2 pilot behavioral gate: the per-field Euler step vs LIVE MuJoCo.

Mirrors tests/physics3d/test_free_joint_euler_vs_mujoco.mojo (tumbling
free-flight Ant, contact-free, float64, opt.integrator=0) but drives the
NEW per-field path: `EulerIntegratorFields.step["cpu"]` over
DataFields/ModelFields (model bridged through the existing
copy_model_to_buffer flattening). Same tolerances as the legacy gate.
Also cross-checks against the legacy CPU EulerIntegrator.step trajectory.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_euler_fields_vs_mujoco.mojo
"""

from std.testing import assert_true, TestSuite
from std.python import Python
from std.math import abs
from std.collections import InlineArray
from std.gpu.host import DeviceContext

from mojo_rl.physics3d.types import Model, Data
from mojo_rl.physics3d.integrator.euler_integrator import EulerIntegrator
from mojo_rl.physics3d.integrator.euler_fields import EulerIntegratorFields
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.physics3d.fields import DataFields, ModelFields
from mojo_rl.physics3d.gpu.buffer_utils import copy_model_to_buffer
from mojo_rl.physics3d.gpu.constants import model_size_with_invweight
from mojo_rl.envs.ant.ant_xml import AntModel

comptime DTYPE = DType.float64
comptime NQ = AntModel.NQ  # 15
comptime NV = AntModel.NV  # 14
comptime NBODY = AntModel.NBODY
comptime NJOINT = AntModel.NJOINT
comptime NGEOM = AntModel.NGEOM
comptime MAX_CONTACTS = AntModel.MAX_CONTACTS
comptime NSITE = AntModel.NSITE
comptime NEQ = AntModel.MAX_EQUALITY
comptime NTEN = AntModel.MAX_TENDON
comptime MS = model_size_with_invweight[
    NBODY, NJOINT, NV, NGEOM, NEQ, NTEN, NSITE
]()

# Same budgets as the legacy free-joint gate.
comptime QPOS_ABS_TOL_1: Float64 = 1e-4
comptime QVEL_ABS_TOL_1: Float64 = 5e-3
comptime QPOS_ABS_TOL_10: Float64 = 1e-3
comptime QVEL_ABS_TOL_10: Float64 = 5e-3


def _tumbling_qpos() -> InlineArray[Float64, NQ]:
    """Contact-free AND limit-free: torso high up, legs mid-range.

    The legacy free-joint gate uses legs at qpos=0, which VIOLATES the ant
    ankle ranges (±(30..70) deg) — fine there because both sides solve limit
    constraints, but this gate compares UNCONSTRAINED dynamics, so all
    joints must sit strictly inside their ranges (verified via MuJoCo
    nefc == 0 below)."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[2] = 2.0  # z — contact-free
    qpos[3] = 0.9659258262890683  # qw (30 deg tilt about y)
    qpos[5] = 0.25881904510252074  # qy
    # legs: hips 0 (inside ±30 deg); ankles mid-range (~51.6 deg, signs per
    # ant.xml: ankle_1/4 in (30,70), ankle_2/3 in (-70,-30))
    qpos[8] = 0.9  # ankle_1
    qpos[10] = -0.9  # ankle_2
    qpos[12] = -0.9  # ankle_3
    qpos[14] = 0.9  # ankle_4
    return qpos


def _tumbling_qvel() -> InlineArray[Float64, NV]:
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[3] = 2.0
    qvel[4] = 1.0
    qvel[5] = 0.5
    return qvel


def _make_model_fields(
    ctx: DeviceContext,
    model: Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, NEQ,
        AntModel.CONE_TYPE, NTEN, NSITE,
    ],
) raises -> ModelFields[DTYPE, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE]:
    """Bridge CPU Model -> ModelFields via the existing flattening."""
    var hb = ctx.enqueue_create_host_buffer[DTYPE](MS)
    ctx.synchronize()
    copy_model_to_buffer(model, hb)
    var flat = List[Scalar[DTYPE]](length=MS, fill=Scalar[DTYPE](0))
    for i in range(MS):
        flat[i] = hb[i]
    var mf = ModelFields[DTYPE, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE]()
    mf.load_from_slab(flat)
    return mf^


def _compare_vs_mujoco(
    num_steps: Int, qpos_tol: Float64, qvel_tol: Float64
) raises:
    print(
        "--- fields-CPU Euler vs MuJoCo, tumbling Ant,", num_steps, "steps ---"
    )
    var qpos_init = _tumbling_qpos()
    var qvel_init = _tumbling_qvel()
    var ctx = DeviceContext()

    # Model setup + bridge to fields.
    var model = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, NEQ,
        AntModel.CONE_TYPE, NTEN, NSITE,
    ]()
    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE]()
    AntModel.setup_model_and_data(model, data)
    var mf = _make_model_fields(ctx, model)

    # Fields path (f64, CPU target, BATCH=1).
    var d = DataFields[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1]()
    for i in range(NQ):
        d.qpos.data[i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        d.qvel.data[i] = Scalar[DTYPE](qvel_init[i])
    var integ = EulerIntegratorFields[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, NEQ, NTEN, NSITE,
        0, 0, 1,
    ]()
    for _ in range(num_steps):
        for i in range(NV):
            d.qfrc.data[i] = Scalar[DTYPE](0)
        integ.step["cpu"](d, mf)

    # Legacy CPU path (reference cross-check).
    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        data.qvel[i] = Scalar[DTYPE](qvel_init[i])
    for _ in range(num_steps):
        for i in range(NV):
            data.qfrc[i] = Scalar[DTYPE](0)
        EulerIntegrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](model, data)

    # Live MuJoCo reference.
    var mujoco = Python.import_module("mujoco")
    var xml_path = (
        "./references/Gymnasium-main/gymnasium/envs/mujoco/assets/ant.xml"
    )
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.integrator = 0
    mj_model.opt.solver = 2
    var mj_data = mujoco.MjData(mj_model)
    for i in range(NQ):
        mj_data.qpos[i] = qpos_init[i]
    for i in range(NV):
        mj_data.qvel[i] = qvel_init[i]
    for _ in range(num_steps):
        mujoco.mj_step(mj_model, mj_data)
        # nefc == 0 => NO active constraints (contacts NOR joint limits) —
        # the unconstrained comparison is only valid under this.
        assert_true(
            Int(py=mj_data.nefc) == 0,
            "expected constraint-free config (no contacts, no active limits)",
        )

    var mj_qpos = mj_data.qpos.flatten().tolist()
    var mj_qvel = mj_data.qvel.flatten().tolist()
    var max_qpos_err: Float64 = 0.0
    var max_qvel_err: Float64 = 0.0
    var max_qpos_vs_legacy: Float64 = 0.0
    for i in range(NQ):
        var e = abs(Float64(d.qpos.data[i]) - Float64(py=mj_qpos[i]))
        if e > max_qpos_err:
            max_qpos_err = e
        var el = abs(Float64(d.qpos.data[i]) - Float64(data.qpos[i]))
        if el > max_qpos_vs_legacy:
            max_qpos_vs_legacy = el
    for i in range(NV):
        var e = abs(Float64(d.qvel.data[i]) - Float64(py=mj_qvel[i]))
        if e > max_qvel_err:
            max_qvel_err = e
    print(
        "  fields quat=(", Float64(d.qpos.data[3]), Float64(d.qpos.data[4]),
        Float64(d.qpos.data[5]), Float64(d.qpos.data[6]),
        ")  mj=(", Float64(py=mj_qpos[3]), Float64(py=mj_qpos[4]),
        Float64(py=mj_qpos[5]), Float64(py=mj_qpos[6]), ")",
    )
    print(
        "  max |qpos err| vs mj =", max_qpos_err,
        " max |qvel err| vs mj =", max_qvel_err,
        " max |qpos| vs legacy-CPU =", max_qpos_vs_legacy,
    )
    assert_true(
        max_qpos_err < qpos_tol, "fields Euler qpos diverged from MuJoCo"
    )
    assert_true(
        max_qvel_err < qvel_tol, "fields Euler qvel diverged from MuJoCo"
    )


def test_fields_euler_vs_mujoco_1_step() raises:
    _compare_vs_mujoco(1, QPOS_ABS_TOL_1, QVEL_ABS_TOL_1)


def test_fields_euler_vs_mujoco_10_steps() raises:
    _compare_vs_mujoco(10, QPOS_ABS_TOL_10, QVEL_ABS_TOL_10)


def test_fields_euler_active_limits_vs_legacy_cpu() raises:
    """The legacy gate's pose (ankles at qpos=0, VIOLATING their ranges) —
    active-limit dynamics through the fields path vs the legacy CPU step.

    MuJoCo is not the reference here: the repo's limit model (acceleration-
    level PGS with impedance) matches MuJoCo's to solver tolerance, not
    trajectory-exactly. The gate is fields vs legacy-CPU, which run the
    same limit formulation."""
    print("--- fields-CPU active limits vs legacy-CPU, tumbling Ant ---")
    var qpos_init = _tumbling_qpos()
    for k in range(4):
        qpos_init[8 + 2 * k] = 0.0  # ankles back to 0 -> limits ACTIVE
    var qvel_init = _tumbling_qvel()
    var ctx = DeviceContext()

    var model = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, NEQ,
        AntModel.CONE_TYPE, NTEN, NSITE,
    ]()
    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE]()
    AntModel.setup_model_and_data(model, data)
    var mf = _make_model_fields(ctx, model)

    var d = DataFields[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1]()
    for i in range(NQ):
        d.qpos.data[i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        d.qvel.data[i] = Scalar[DTYPE](qvel_init[i])
    var integ = EulerIntegratorFields[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, NEQ, NTEN, NSITE,
        0, 0, 1,
    ]()
    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        data.qvel[i] = Scalar[DTYPE](qvel_init[i])

    for _ in range(10):
        for i in range(NV):
            d.qfrc.data[i] = Scalar[DTYPE](0)
            data.qfrc[i] = Scalar[DTYPE](0)
        integ.step["cpu"](d, mf)
        EulerIntegrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](model, data)

    var worst_qp = Float64(0)
    var worst_qv = Float64(0)
    for i in range(NQ):
        var e = abs(Float64(d.qpos.data[i]) - Float64(data.qpos[i]))
        if e > worst_qp:
            worst_qp = e
    for i in range(NV):
        var e = abs(Float64(d.qvel.data[i]) - Float64(data.qvel[i]))
        if e > worst_qv:
            worst_qv = e
    print(
        "  after 10 steps: max |qpos| vs legacy-CPU =", worst_qp,
        " max |qvel| =", worst_qv,
    )
    assert_true(
        worst_qp < 1e-3 and worst_qv < 5e-2,
        "fields active-limit dynamics diverged from legacy CPU",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

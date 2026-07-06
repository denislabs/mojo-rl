"""Free-joint quaternion integration: Euler/ImplicitFast/Implicit gates.

Regression gate for a real bug: Euler, ImplicitFast and Implicit read the
free-joint qpos quaternion X-FIRST (qx at qpos_adr+3) while the storage is
MuJoCo w-first ([tx,ty,tz,qw,qx,qy,qz]) — as FK and RK4 correctly assume.
Any floating-base body integrated by those three integrators rotated wrongly
(permuted components read AND written). The gap survived because every
free-joint env uses RK4, and no free-joint parity test ran under Euler.

Two gates:
  1. Euler CPU step vs MuJoCo (opt.integrator=0) on tumbling free-flight Ant
     configs (contact-free, so solver differences don't matter).
  2. ImplicitFast + Implicit vs our own RK4 over ONE small step: quaternion
     components must agree to O(dt^2) ~ 1e-4 (integration-order difference);
     the permutation bug produces O(1) divergence.

Run with:
    pixi run mojo run -I . tests/physics3d/test_free_joint_euler_vs_mujoco.mojo
"""

from std.testing import assert_true, TestSuite
from std.python import Python, PythonObject
from std.math import abs
from std.collections import InlineArray

from mojo_rl.physics3d.types import Model, Data, ConeType
from mojo_rl.physics3d.integrator.euler_integrator import EulerIntegrator
from mojo_rl.physics3d.integrator.implicit_fast_integrator import (
    ImplicitFastIntegrator,
)
from mojo_rl.physics3d.integrator.implicit_integrator import ImplicitIntegrator
from mojo_rl.physics3d.integrator.rk4_integrator import RK4Integrator
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.envs.ant.ant_xml import AntModel
from mojo_rl.envs.ant.ant_config import AntConfig

comptime DTYPE = DType.float64
comptime NQ = AntModel.NQ  # 15
comptime NV = AntModel.NV  # 14
comptime NBODY = AntModel.NBODY
comptime NJOINT = AntModel.NJOINT
comptime NGEOM = AntModel.NGEOM
comptime MAX_CONTACTS = AntModel.MAX_CONTACTS
comptime ACTION_DIM = AntConfig.ACTION_DIM  # 8

# Gates BOTH free-joint bug classes: the quat component-order bug (O(1)
# quat error) and the RNE cdof_dot ordering bug (cvel updated inside the
# free-rotation loop instead of MuJoCo's all-3-from-pre-rotation-cvel —
# ~1%/step spurious gyroscopic bias). Measured with both fixed:
# 1 step -> qpos 8.7e-6 / qvel 8.7e-4; 10 steps -> qpos 8.3e-5 / qvel 8e-4.
# Either regression pushes these past 1e-2.
comptime QPOS_ABS_TOL_1: Float64 = 1e-4
comptime QVEL_ABS_TOL_1: Float64 = 5e-3
comptime QPOS_ABS_TOL_10: Float64 = 1e-3
comptime QVEL_ABS_TOL_10: Float64 = 5e-3
# One dt=0.01 step: 1st-order vs 4th-order quat difference is O((w*dt)^2).
comptime XORDER_QUAT_TOL: Float64 = 5e-3


def _set_state(
    mut data: Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, AntModel.NSITE
    ],
    qpos: InlineArray[Float64, NQ],
    qvel: InlineArray[Float64, NV],
):
    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos[i])
    for i in range(NV):
        data.qvel[i] = Scalar[DTYPE](qvel[i])


def _tumbling_qpos() -> InlineArray[Float64, NQ]:
    """Torso high above ground (no contacts), tilted 30 deg about y."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[2] = 2.0  # z — contact-free for the short horizons used here
    # 30 deg tilt about y: (qw, qx, qy, qz) = (cos15, 0, sin15, 0)
    qpos[3] = 0.9659258262890683  # qw
    qpos[5] = 0.25881904510252074  # qy
    return qpos


def _tumbling_qvel() -> InlineArray[Float64, NV]:
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[3] = 2.0  # wx
    qvel[4] = 1.0  # wy
    qvel[5] = 0.5  # wz
    return qvel


def _compare_euler_vs_mujoco(
    num_steps: Int, qpos_tol: Float64, qvel_tol: Float64
) raises:
    print("--- Euler vs MuJoCo, tumbling free flight,", num_steps, "steps ---")
    var qpos_init = _tumbling_qpos()
    var qvel_init = _tumbling_qvel()

    var model = Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        AntModel.MAX_EQUALITY,
        AntModel.CONE_TYPE,
        AntModel.MAX_TENDON,
        AntModel.NSITE,
    ]()
    var data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, AntModel.NSITE
    ]()
    AntModel.setup_model_and_data(model, data)
    _set_state(data, qpos_init, qvel_init)

    for _ in range(num_steps):
        for i in range(NV):
            data.qfrc[i] = Scalar[DTYPE](0)
        EulerIntegrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](model, data)

    var mujoco = Python.import_module("mujoco")
    var xml_path = (
        "./references/Gymnasium-main/gymnasium/envs/mujoco/assets/ant.xml"
    )
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.integrator = 0  # mjINT_EULER
    mj_model.opt.solver = 2  # mjSOL_NEWTON
    var mj_data = mujoco.MjData(mj_model)
    for i in range(NQ):
        mj_data.qpos[i] = qpos_init[i]
    for i in range(NV):
        mj_data.qvel[i] = qvel_init[i]
    for _ in range(num_steps):
        mujoco.mj_step(mj_model, mj_data)
    assert_true(
        Int(py=mj_data.ncon) == 0, "expected contact-free config in MuJoCo"
    )

    var mj_qpos = mj_data.qpos.flatten().tolist()
    var mj_qvel = mj_data.qvel.flatten().tolist()
    var max_qpos_err: Float64 = 0.0
    var max_qvel_err: Float64 = 0.0
    for i in range(NQ):
        var e = abs(Float64(data.qpos[i]) - Float64(py=mj_qpos[i]))
        if e > max_qpos_err:
            max_qpos_err = e
    for i in range(NV):
        var e = abs(Float64(data.qvel[i]) - Float64(py=mj_qvel[i]))
        if e > max_qvel_err:
            max_qvel_err = e
    print(
        "  quat ours=(",
        Float64(data.qpos[3]),
        Float64(data.qpos[4]),
        Float64(data.qpos[5]),
        Float64(data.qpos[6]),
        ")  mj=(",
        Float64(py=mj_qpos[3]),
        Float64(py=mj_qpos[4]),
        Float64(py=mj_qpos[5]),
        Float64(py=mj_qpos[6]),
        ")",
    )
    print("  max |qpos err| =", max_qpos_err, " max |qvel err| =", max_qvel_err)
    assert_true(
        max_qpos_err < qpos_tol,
        "Euler free-joint qpos diverged from MuJoCo (quat order bug?)",
    )
    assert_true(
        max_qvel_err < qvel_tol,
        "Euler free-joint qvel diverged from MuJoCo",
    )


def test_euler_vs_mujoco_1_step() raises:
    _compare_euler_vs_mujoco(1, QPOS_ABS_TOL_1, QVEL_ABS_TOL_1)


def test_euler_vs_mujoco_10_steps() raises:
    _compare_euler_vs_mujoco(10, QPOS_ABS_TOL_10, QVEL_ABS_TOL_10)


def _one_step_quat[
    WHICH: Int
]() raises -> InlineArray[Float64, 4]:
    """One dt step of the chosen integrator on the tumbling config; returns
    the free-joint quaternion (w, x, y, z) from qpos[3..6]."""
    var qpos_init = _tumbling_qpos()
    var qvel_init = _tumbling_qvel()

    var model = Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        AntModel.MAX_EQUALITY,
        AntModel.CONE_TYPE,
        AntModel.MAX_TENDON,
        AntModel.NSITE,
    ]()
    var data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, AntModel.NSITE
    ]()
    AntModel.setup_model_and_data(model, data)
    _set_state(data, qpos_init, qvel_init)
    for i in range(NV):
        data.qfrc[i] = Scalar[DTYPE](0)

    comptime if WHICH == 0:
        RK4Integrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](model, data)
    elif WHICH == 1:
        ImplicitFastIntegrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](
            model, data
        )
    else:
        ImplicitIntegrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](model, data)

    var q = InlineArray[Float64, 4](fill=0.0)
    for k in range(4):
        q[k] = Float64(data.qpos[3 + k])
    return q


def _check_vs_rk4(name: String, q: InlineArray[Float64, 4],
                  q_rk4: InlineArray[Float64, 4]) raises:
    var max_err: Float64 = 0.0
    for k in range(4):
        var e = abs(q[k] - q_rk4[k])
        if e > max_err:
            max_err = e
    print(
        "  ", name, " quat=(", q[0], q[1], q[2], q[3],
        ")  rk4=(", q_rk4[0], q_rk4[1], q_rk4[2], q_rk4[3],
        ")  max_err=", max_err,
    )
    assert_true(
        max_err < XORDER_QUAT_TOL,
        name
        + ": free-joint quat diverged O(1) from RK4 after one step"
        + " (component-order bug?)",
    )


def test_implicit_variants_quat_vs_rk4() raises:
    """ImplicitFast/Implicit one-step quat must agree with RK4 to O(dt^2)."""
    print("--- ImplicitFast/Implicit one-step quat vs RK4 ---")
    var q_rk4 = _one_step_quat[0]()
    var q_if = _one_step_quat[1]()
    var q_im = _one_step_quat[2]()
    _check_vs_rk4("ImplicitFast", q_if, q_rk4)
    _check_vs_rk4("Implicit", q_im, q_rk4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

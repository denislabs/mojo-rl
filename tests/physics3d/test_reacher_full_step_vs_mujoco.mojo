"""Test Full Physics Step: Mojo Engine vs MuJoCo for Reacher.

Compares qpos/qvel after running physics steps in both engines from
identical initial states with identical actions applied.

Reacher is a 2-link planar arm with no contacts (all contype=0).
Uses RK4 integrator (timestep=0.01) and Newton solver.

Run with:
    cd mojo-rl && pixi run mojo run -I . tests/physics3d/test_reacher_full_step_vs_mujoco.mojo
"""

from std.testing import assert_true, TestSuite
from std.python import Python, PythonObject
from std.math import abs
from std.collections import InlineArray

from mojo_rl.physics3d.types import Model, Data, ConeType
from mojo_rl.physics3d.integrator.rk4_integrator import RK4Integrator
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.envs.reacher.reacher_xml import ReacherModel


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64
comptime NQ = ReacherModel.NQ  # 4
comptime NV = ReacherModel.NV  # 4
comptime NBODY = ReacherModel.NBODY  # 5
comptime NJOINT = ReacherModel.NJOINT  # 4
comptime NGEOM = ReacherModel.NGEOM
comptime MAX_CONTACTS = ReacherModel.MAX_CONTACTS  # 5
comptime ACTION_DIM = ReacherModel.ACTION_DIM  # 2

# Tolerances
comptime QPOS_ABS_TOL: Float64 = 1e-6
comptime QPOS_REL_TOL: Float64 = 1e-6
comptime QVEL_ABS_TOL: Float64 = 1e-6
comptime QVEL_REL_TOL: Float64 = 1e-6


# =============================================================================
# Comparison helper
# =============================================================================


def compare_step(
    test_name: String,
    qpos_init: InlineArray[Float64, NQ],
    qvel_init: InlineArray[Float64, NV],
    actions: InlineArray[Float64, ACTION_DIM],
    num_steps: Int = 1,
) raises:
    """Run num_steps physics steps in both engines, compare final qpos/qvel."""
    print("--- Test:", test_name, "---")
    print("  Steps:", num_steps)

    # === Our engine ===
    var model = Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        ReacherModel.MAX_EQUALITY,
        ReacherModel.CONE_TYPE,
        ReacherModel.MAX_TENDON,
        ReacherModel.NSITE,
    ]()
    var data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, ReacherModel.NSITE
    ]()
    ReacherModel.setup_model_and_data(model, data)

    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        data.qvel[i] = Scalar[DTYPE](qvel_init[i])

    var action_list = List[Float64]()
    for i in range(ACTION_DIM):
        action_list.append(actions[i])

    for _ in range(num_steps):
        for i in range(NV):
            data.qfrc[i] = Scalar[DTYPE](0)
        ReacherModel.apply_actions(data, action_list)
        RK4Integrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](model, data)

    # === MuJoCo reference ===
    var mujoco = Python.import_module("mujoco")

    var xml_path = "./references/Gymnasium-main/gymnasium/envs/mujoco/assets/reacher.xml"
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    var mj_data = mujoco.MjData(mj_model)

    for i in range(NQ):
        mj_data.qpos[i] = qpos_init[i]
    for i in range(NV):
        mj_data.qvel[i] = qvel_init[i]

    for i in range(ACTION_DIM):
        mj_data.ctrl[i] = actions[i]

    for _ in range(num_steps):
        mujoco.mj_step(mj_model, mj_data)

    # === Compare ===
    var mj_qpos = mj_data.qpos.flatten().tolist()
    var mj_qvel = mj_data.qvel.flatten().tolist()

    var all_pass = True
    var max_abs: Float64 = 0.0
    var max_rel: Float64 = 0.0

    for i in range(NQ):
        var our_val = Float64(data.qpos[i])
        var mj_val = Float64(py=mj_qpos[i])
        var abs_err = abs(our_val - mj_val)
        var ref_mag = abs(mj_val)
        var rel_err: Float64 = 0.0
        if ref_mag > 1e-10:
            rel_err = abs_err / ref_mag
        if abs_err > max_abs:
            max_abs = abs_err
        if rel_err > max_rel:
            max_rel = rel_err
        var ok = abs_err < QPOS_ABS_TOL or rel_err < QPOS_REL_TOL
        if not ok:
            print(
                "  FAIL qpos[", i, "] ours=", our_val, " mj=", mj_val,
                " abs=", abs_err, " rel=", rel_err,
            )
            all_pass = False

    for i in range(NV):
        var our_val = Float64(data.qvel[i])
        var mj_val = Float64(py=mj_qvel[i])
        var abs_err = abs(our_val - mj_val)
        var ref_mag = abs(mj_val)
        var rel_err: Float64 = 0.0
        if ref_mag > 1e-10:
            rel_err = abs_err / ref_mag
        if abs_err > max_abs:
            max_abs = abs_err
        if rel_err > max_rel:
            max_rel = rel_err
        var ok = abs_err < QVEL_ABS_TOL or rel_err < QVEL_REL_TOL
        if not ok:
            print(
                "  FAIL qvel[", i, "] ours=", our_val, " mj=", mj_val,
                " abs=", abs_err, " rel=", rel_err,
            )
            all_pass = False

    if all_pass:
        print("  ALL OK  max_abs=", max_abs, " max_rel=", max_rel)
    else:
        print("  Our qpos:", end="")
        for i in range(NQ):
            print(" ", Float64(data.qpos[i]), end="")
        print()
        print("  MJ  qpos:", end="")
        for i in range(NQ):
            print(" ", Float64(py=mj_qpos[i]), end="")
        print()
        print("  Our qvel:", end="")
        for i in range(NV):
            print(" ", Float64(data.qvel[i]), end="")
        print()
        print("  MJ  qvel:", end="")
        for i in range(NV):
            print(" ", Float64(py=mj_qvel[i]), end="")
        print()

    assert_true(all_pass, "compare_step failed for: " + test_name)


# =============================================================================
# Test cases
# =============================================================================


def test_zero_state_no_action() raises:
    """From rest at zero angles with no actions — gravity-only dynamics."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step("Zero state, no action", qpos, qvel, actions)


def test_zero_state_with_actions() raises:
    """From rest with actions applied — tests actuator forces."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.5  # joint0 torque
    actions[1] = -0.3  # joint1 torque
    compare_step("Zero state, with actions", qpos, qvel, actions)


def test_nonzero_angles() raises:
    """Starting from non-zero joint angles with velocity."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 0.5  # joint0 angle
    qpos[1] = -0.8  # joint1 angle
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 1.0
    qvel[1] = -0.5
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.3
    actions[1] = 0.7
    compare_step("Non-zero angles + velocity + actions", qpos, qvel, actions)


def test_target_offset() raises:
    """With target slide joints at non-zero positions."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 0.2  # joint0
    qpos[1] = -0.4  # joint1
    qpos[2] = 0.1  # target_x
    qpos[3] = -0.05  # target_y
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = -0.8
    actions[1] = 0.6
    compare_step("Target offset + actions", qpos, qvel, actions)


def test_multiple_steps() raises:
    """10 steps with actions — accumulates integration drift."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 0.3
    qpos[1] = -0.5
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 0.5
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.4
    actions[1] = -0.2
    compare_step("10 steps with actions", qpos, qvel, actions, num_steps=10)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

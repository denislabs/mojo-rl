"""Test Full Physics Step: Mojo Engine vs MuJoCo for Pusher.

Compares qpos/qvel after running physics steps in both engines from
identical initial states with identical actions applied.

Pusher is a 7-DOF arm pushing a cylinder on a table. Zero gravity.
Uses Euler integrator (timestep=0.01) and Newton solver.

Run with:
    cd mojo-rl && pixi run mojo run -I . tests/physics3d/test_pusher_full_step_vs_mujoco.mojo
"""

from std.testing import assert_true, TestSuite
from std.python import Python, PythonObject
from std.math import abs
from std.collections import InlineArray

from mojo_rl.physics3d.types import Model, Data, ConeType
from mojo_rl.physics3d.integrator.euler_integrator import EulerIntegrator
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.envs.pusher.pusher_xml import PusherModel


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64
comptime NQ = PusherModel.NQ  # 11
comptime NV = PusherModel.NV  # 11
comptime NBODY = PusherModel.NBODY  # 13
comptime NJOINT = PusherModel.NJOINT  # 11
comptime NGEOM = PusherModel.NGEOM
comptime MAX_CONTACTS = PusherModel.MAX_CONTACTS  # 20
comptime ACTION_DIM = PusherModel.ACTION_DIM  # 7

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
        PusherModel.MAX_EQUALITY,
        PusherModel.CONE_TYPE,
        PusherModel.MAX_TENDON,
        PusherModel.NSITE,
    ]()
    var data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, PusherModel.NSITE
    ]()
    PusherModel.setup_model_and_data(model, data)

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
        PusherModel.apply_actions(data, action_list)
        EulerIntegrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](model, data)

    # === MuJoCo reference ===
    var mujoco = Python.import_module("mujoco")

    var xml_path = "./references/Gymnasium-main/gymnasium/envs/mujoco/assets/pusher.xml"
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.solver = 2  # mjSOL_NEWTON
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
    """All joints at zero with no actions — zero-gravity, only damping."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step("Zero state, no action", qpos, qvel, actions)


def test_zero_state_with_actions() raises:
    """All joints at zero with actions — tests actuator mapping."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.5  # shoulder pan
    actions[1] = -0.3  # shoulder lift
    actions[2] = 0.2  # upper arm roll
    actions[3] = -0.4  # elbow flex
    actions[4] = 0.1  # forearm roll
    actions[5] = 0.3  # wrist flex
    actions[6] = -0.2  # wrist roll
    compare_step("Zero state, with actions", qpos, qvel, actions)


def test_nonzero_joints() raises:
    """Starting from non-zero joint positions with velocities."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 0.3  # shoulder pan
    qpos[1] = -0.5  # shoulder lift
    qpos[2] = 0.1  # upper arm roll
    qpos[3] = -0.8  # elbow flex
    qpos[4] = 0.2  # forearm roll
    qpos[5] = -0.3  # wrist flex
    qpos[6] = 0.15  # wrist roll
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 0.5
    qvel[3] = -1.0
    qvel[5] = 0.3
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = -0.6
    actions[3] = 0.8
    compare_step("Non-zero joints + velocity + actions", qpos, qvel, actions)


def test_with_object_position() raises:
    """Object and goal at non-zero positions (slide joints)."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 0.2  # shoulder pan
    qpos[3] = -0.5  # elbow flex
    qpos[7] = 0.1  # object slide y
    qpos[8] = -0.05  # object slide x
    qpos[9] = 0.15  # goal slide y
    qpos[10] = -0.1  # goal slide x
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.4
    actions[1] = -0.3
    actions[3] = 0.6
    compare_step("Object + goal positions", qpos, qvel, actions)


def test_multiple_steps() raises:
    """10 steps with actions — accumulates integration drift."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 0.1
    qpos[1] = -0.2
    qpos[3] = -0.4
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 0.3
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.3
    actions[3] = -0.5
    actions[5] = 0.2
    compare_step("10 steps with actions", qpos, qvel, actions, num_steps=10)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

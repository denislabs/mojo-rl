"""Test Swimmer WITH fluid dynamics: Mojo Engine vs MuJoCo.

Compares qpos/qvel with fluid forces ENABLED on both sides
(density=4000, viscosity=0.1). This validates our fluid force
implementation against MuJoCo's inertia-box fluid model.

Runs progressively more steps to find where divergence starts.

Run with:
    cd mojo-rl && pixi run mojo run -I . tests/physics3d/test_swimmer_fluid_vs_mujoco.mojo
"""

from std.testing import assert_true, TestSuite
from std.python import Python, PythonObject
from std.math import abs, isnan
from std.collections import InlineArray

from mojo_rl.physics3d.types import Model, Data, ConeType
from mojo_rl.physics3d.integrator.rk4_integrator import RK4Integrator
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.envs.swimmer.swimmer_xml import SwimmerModel


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64
comptime NQ = SwimmerModel.NQ  # 5
comptime NV = SwimmerModel.NV  # 5
comptime NBODY = SwimmerModel.NBODY  # 4
comptime NJOINT = SwimmerModel.NJOINT  # 5
comptime NGEOM = SwimmerModel.NGEOM  # 3
comptime MAX_CONTACTS = SwimmerModel.MAX_CONTACTS  # 5
comptime ACTION_DIM = SwimmerModel.ACTION_DIM  # 2


# =============================================================================
# Comparison helper (with fluid enabled on MuJoCo side)
# =============================================================================


def compare_fluid_step(
    test_name: String,
    qpos_init: InlineArray[Float64, NQ],
    qvel_init: InlineArray[Float64, NV],
    actions: InlineArray[Float64, ACTION_DIM],
    num_steps: Int = 1,
    qpos_abs_tol: Float64 = 1e-3,
    qvel_abs_tol: Float64 = 1e-2,
    print_every: Int = 0,
) raises -> Bool:
    """Run num_steps physics steps in both engines WITH fluid forces, compare."""
    print("--- Test:", test_name, "(", num_steps, "steps) ---")

    # === Our engine (RK4 + Newton + fluid forces) ===
    var model = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
        SwimmerModel.MAX_EQUALITY, SwimmerModel.CONE_TYPE,
        SwimmerModel.MAX_TENDON, SwimmerModel.NSITE,
    ]()
    var data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, SwimmerModel.NSITE
    ]()
    SwimmerModel.setup_model_and_data(model, data)

    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        data.qvel[i] = Scalar[DTYPE](qvel_init[i])

    var action_list = List[Float64]()
    for i in range(ACTION_DIM):
        action_list.append(actions[i])

    # === MuJoCo reference (RK4 WITH fluid: density=4000, viscosity=0.1) ===
    var mujoco = Python.import_module("mujoco")

    var xml_path = "./references/Gymnasium-main/gymnasium/envs/mujoco/assets/swimmer.xml"
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.integrator = 1  # mjINT_RK4
    mj_model.opt.solver = 2  # mjSOL_NEWTON
    mj_model.opt.cone = 1  # mjCONE_ELLIPTIC
    # KEEP fluid dynamics enabled (density=4000, viscosity=0.1 from XML)
    var mj_data = mujoco.MjData(mj_model)

    for i in range(NQ):
        mj_data.qpos[i] = qpos_init[i]
    for i in range(NV):
        mj_data.qvel[i] = qvel_init[i]
    for i in range(ACTION_DIM):
        mj_data.ctrl[i] = actions[i]

    # Step both engines, optionally printing intermediate state
    for step in range(num_steps):
        # Our engine
        for i in range(NV):
            data.qfrc[i] = Scalar[DTYPE](0)
        SwimmerModel.apply_actions(data, action_list)
        RK4Integrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](model, data)

        # MuJoCo
        mujoco.mj_step(mj_model, mj_data)

        # Check for NaN/Inf in our engine
        var has_nan = False
        for i in range(NQ):
            if isnan(Float64(data.qpos[i])):
                has_nan = True
        for i in range(NV):
            if isnan(Float64(data.qvel[i])):
                has_nan = True

        if has_nan:
            print("  *** NaN detected at step", step + 1, "***")
            print("  Our qpos:", end="")
            for i in range(NQ):
                print(" ", Float64(data.qpos[i]), end="")
            print()
            print("  Our qvel:", end="")
            for i in range(NV):
                print(" ", Float64(data.qvel[i]), end="")
            print()
            return False

        # Print intermediate comparison
        if print_every > 0 and ((step + 1) % print_every == 0 or step == 0):
            var mj_qpos_s = mj_data.qpos.flatten().tolist()
            var mj_qvel_s = mj_data.qvel.flatten().tolist()

            var max_qpos_err: Float64 = 0.0
            var max_qvel_err: Float64 = 0.0
            for i in range(NQ):
                var err = abs(Float64(data.qpos[i]) - Float64(py=mj_qpos_s[i]))
                if err > max_qpos_err:
                    max_qpos_err = err
            for i in range(NV):
                var err = abs(Float64(data.qvel[i]) - Float64(py=mj_qvel_s[i]))
                if err > max_qvel_err:
                    max_qvel_err = err

            print(
                "  Step", step + 1,
                " qpos_err=", max_qpos_err,
                " qvel_err=", max_qvel_err,
            )

            # Print full state at select steps
            if (step + 1) <= 5 or (step + 1) % (print_every * 5) == 0:
                print("    Our qpos:", end="")
                for i in range(NQ):
                    print(" ", Float64(data.qpos[i]), end="")
                print()
                print("    MJ  qpos:", end="")
                for i in range(NQ):
                    print(" ", Float64(py=mj_qpos_s[i]), end="")
                print()
                print("    Our qvel:", end="")
                for i in range(NV):
                    print(" ", Float64(data.qvel[i]), end="")
                print()
                print("    MJ  qvel:", end="")
                for i in range(NV):
                    print(" ", Float64(py=mj_qvel_s[i]), end="")
                print()

    # === Final comparison ===
    var mj_qpos = mj_data.qpos.flatten().tolist()
    var mj_qvel = mj_data.qvel.flatten().tolist()

    var qpos_max_abs: Float64 = 0.0
    var qvel_max_abs: Float64 = 0.0

    for i in range(NQ):
        var err = abs(Float64(data.qpos[i]) - Float64(py=mj_qpos[i]))
        if err > qpos_max_abs:
            qpos_max_abs = err
    for i in range(NV):
        var err = abs(Float64(data.qvel[i]) - Float64(py=mj_qvel[i]))
        if err > qvel_max_abs:
            qvel_max_abs = err

    var passed = qpos_max_abs < qpos_abs_tol and qvel_max_abs < qvel_abs_tol

    if passed:
        print("  PASS  qpos_max_err=", qpos_max_abs, " qvel_max_err=", qvel_max_abs)
    else:
        print("  FAIL  qpos_max_err=", qpos_max_abs, " qvel_max_err=", qvel_max_abs)

    print("  Final Our qpos:", end="")
    for i in range(NQ):
        print(" ", Float64(data.qpos[i]), end="")
    print()
    print("  Final MJ  qpos:", end="")
    for i in range(NQ):
        print(" ", Float64(py=mj_qpos[i]), end="")
    print()
    print("  Final Our qvel:", end="")
    for i in range(NV):
        print(" ", Float64(data.qvel[i]), end="")
    print()
    print("  Final MJ  qvel:", end="")
    for i in range(NV):
        print(" ", Float64(py=mj_qvel[i]), end="")
    print()

    return passed


# =============================================================================
# Test cases — with fluid dynamics enabled
# =============================================================================


def test_fluid_1_step_no_action() raises:
    """1 step, no action — pure fluid drag on stationary body."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    var ok = compare_fluid_step(
        "Fluid: 1 step, no action", qpos, qvel, actions,
        num_steps=1, qpos_abs_tol=1e-6, qvel_abs_tol=1e-5,
    )
    assert_true(ok, "test_fluid_1_step_no_action")


def test_fluid_1_step_with_actions() raises:
    """1 step with actions — fluid drag + motor torques."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.5
    actions[1] = -0.3
    var ok = compare_fluid_step(
        "Fluid: 1 step, actions (0.5, -0.3)", qpos, qvel, actions,
        num_steps=1, qpos_abs_tol=1e-3, qvel_abs_tol=1e-2,
    )
    assert_true(ok, "test_fluid_1_step_with_actions")


def test_fluid_1_step_moving() raises:
    """1 step with initial velocity — drag should slow it down."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 2.0  # x-velocity
    qvel[3] = 1.0  # motor1 angular velocity
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.3
    var ok = compare_fluid_step(
        "Fluid: 1 step, moving (vx=2, w3=1)", qpos, qvel, actions,
        num_steps=1, qpos_abs_tol=1e-2, qvel_abs_tol=0.5,
    )
    assert_true(ok, "test_fluid_1_step_moving")


def test_fluid_10_steps() raises:
    """10 steps with actions — check drift over short horizon."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.5
    actions[1] = -0.3
    var ok = compare_fluid_step(
        "Fluid: 10 steps, actions (0.5, -0.3)", qpos, qvel, actions,
        num_steps=10, print_every=5, qpos_abs_tol=0.1, qvel_abs_tol=2.0,
    )
    assert_true(ok, "test_fluid_10_steps")


def test_fluid_100_steps() raises:
    """100 steps — medium horizon, check for divergence."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.8
    actions[1] = -0.6
    var ok = compare_fluid_step(
        "Fluid: 100 steps, strong actions (0.8, -0.6)", qpos, qvel, actions,
        num_steps=100, print_every=10, qpos_abs_tol=5.0, qvel_abs_tol=10.0,
    )
    assert_true(ok, "test_fluid_100_steps")


def test_fluid_1000_steps() raises:
    """1000 steps (= 1 full episode) — check stability over full episode."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 1.0   # max action
    actions[1] = -1.0  # max action
    var ok = compare_fluid_step(
        "Fluid: 1000 steps, MAX actions (1.0, -1.0)", qpos, qvel, actions,
        num_steps=1000, print_every=100, qpos_abs_tol=50.0, qvel_abs_tol=50.0,
    )
    assert_true(ok, "test_fluid_1000_steps")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

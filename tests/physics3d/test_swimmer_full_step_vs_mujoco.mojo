"""Test Full Physics Step (no contacts): Mojo Engine vs MuJoCo for Swimmer.

Compares qpos/qvel after running physics steps in both engines from
identical initial states with identical actions applied.

The Swimmer has contype=0 on all geoms so there are no contacts — this is
the first test of pure dynamics without any contact solver involvement.
All differences come from integration and actuator mapping alone.

Key notes:
  - Integrator: RK4 (opt.integrator=1 = mjINT_RK4)
  - Solver: Newton (opt.solver=2 = mjSOL_NEWTON, irrelevant — no contacts)
  - Cone: Elliptic (opt.cone=1, default, irrelevant — no contacts)
  - Fluid dynamics: The original XML has viscosity=0.1, density=4000.
    We disable these in the MuJoCo comparison (opt.viscosity=0, opt.density=0)
    since our engine does not implement fluid drag/buoyancy. This makes the
    comparison test pure rigid-body dynamics, which our engine does support.

NQ=5, NV=5, ACTION_DIM=2 (motor1_rot gear=150, motor2_rot gear=150).

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_swimmer_full_step_vs_mujoco.mojo
"""

from std.testing import assert_true, TestSuite
from std.python import Python, PythonObject
from std.math import abs
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
comptime NGEOM = SwimmerModel.NGEOM
comptime MAX_CONTACTS = SwimmerModel.MAX_CONTACTS  # 5
comptime ACTION_DIM = SwimmerModel.ACTION_DIM  # 2

# Tolerances — no contacts, so should match tightly
comptime QPOS_ABS_TOL: Float64 = 1e-3
comptime QPOS_REL_TOL: Float64 = 1e-2
comptime QVEL_ABS_TOL: Float64 = 1e-2
comptime QVEL_REL_TOL: Float64 = 1e-2


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
    print("--- Test:", test_name, "(", num_steps, "steps) ---")

    # === Our engine (RK4 + Newton) ===
    var model = Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        SwimmerModel.MAX_EQUALITY,
        SwimmerModel.CONE_TYPE,
        SwimmerModel.MAX_TENDON,
        SwimmerModel.NSITE,
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

    for _ in range(num_steps):
        for i in range(NV):
            data.qfrc[i] = Scalar[DTYPE](0)
        SwimmerModel.apply_actions(data, action_list)
        RK4Integrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](model, data)

    print("  Our contacts:", Int(data.num_contacts))

    # === MuJoCo reference (RK4, no fluid) ===
    var mujoco = Python.import_module("mujoco")

    var xml_path = "./references/Gymnasium-main/gymnasium/envs/mujoco/assets/swimmer.xml"
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.integrator = 1  # mjINT_RK4
    mj_model.opt.solver = 2  # mjSOL_NEWTON (irrelevant — no contacts)
    mj_model.opt.cone = 1  # mjCONE_ELLIPTIC (irrelevant — no contacts)
    # Disable fluid dynamics: our engine has no viscosity or buoyancy.
    # Without this, MuJoCo applies viscous drag and buoyancy that we can't match.
    mj_model.opt.viscosity = 0.0
    mj_model.opt.density = 0.0
    var mj_data = mujoco.MjData(mj_model)

    for i in range(NQ):
        mj_data.qpos[i] = qpos_init[i]
    for i in range(NV):
        mj_data.qvel[i] = qvel_init[i]
    for i in range(ACTION_DIM):
        mj_data.ctrl[i] = actions[i]

    for _ in range(num_steps):
        mujoco.mj_step(mj_model, mj_data)

    var mj_ncon = Int(py=mj_data.ncon)
    print("  MJ  contacts:", mj_ncon)

    # === Compare qpos ===
    var mj_qpos = mj_data.qpos.flatten().tolist()
    var mj_qvel = mj_data.qvel.flatten().tolist()

    var qpos_pass = True
    var qpos_max_abs: Float64 = 0.0
    var qpos_max_rel: Float64 = 0.0
    var qpos_fails = 0

    for i in range(NQ):
        var our_val = Float64(data.qpos[i])
        var mj_val = Float64(py=mj_qpos[i])
        var abs_err = abs(our_val - mj_val)
        var ref_mag = abs(mj_val)
        var rel_err: Float64 = 0.0
        if ref_mag > 1e-10:
            rel_err = abs_err / ref_mag

        if abs_err > qpos_max_abs:
            qpos_max_abs = abs_err
        if rel_err > qpos_max_rel:
            qpos_max_rel = rel_err

        var ok = abs_err < QPOS_ABS_TOL or rel_err < QPOS_REL_TOL
        if not ok:
            if qpos_fails < 5:
                print(
                    "  FAIL qpos[",
                    i,
                    "]",
                    " ours=",
                    our_val,
                    " mj=",
                    mj_val,
                    " abs=",
                    abs_err,
                    " rel=",
                    rel_err,
                )
            qpos_fails += 1
            qpos_pass = False

    # === Compare qvel ===
    var qvel_pass = True
    var qvel_max_abs: Float64 = 0.0
    var qvel_max_rel: Float64 = 0.0
    var qvel_fails = 0

    for i in range(NV):
        var our_val = Float64(data.qvel[i])
        var mj_val = Float64(py=mj_qvel[i])
        var abs_err = abs(our_val - mj_val)
        var ref_mag = abs(mj_val)
        var rel_err: Float64 = 0.0
        if ref_mag > 1e-10:
            rel_err = abs_err / ref_mag

        if abs_err > qvel_max_abs:
            qvel_max_abs = abs_err
        if rel_err > qvel_max_rel:
            qvel_max_rel = rel_err

        var ok = abs_err < QVEL_ABS_TOL or rel_err < QVEL_REL_TOL
        if not ok:
            if qvel_fails < 5:
                print(
                    "  FAIL qvel[",
                    i,
                    "]",
                    " ours=",
                    our_val,
                    " mj=",
                    mj_val,
                    " abs=",
                    abs_err,
                    " rel=",
                    rel_err,
                )
            qvel_fails += 1
            qvel_pass = False

    var all_pass = qpos_pass and qvel_pass

    if all_pass:
        print(
            "  ALL OK  qpos_max_abs=",
            qpos_max_abs,
            " qpos_max_rel=",
            qpos_max_rel,
            " qvel_max_abs=",
            qvel_max_abs,
            " qvel_max_rel=",
            qvel_max_rel,
        )
    else:
        print(
            "  FAILED  qpos:",
            qpos_fails,
            "fails (max_abs=",
            qpos_max_abs,
            " max_rel=",
            qpos_max_rel,
            ")",
            " qvel:",
            qvel_fails,
            "fails (max_abs=",
            qvel_max_abs,
            " max_rel=",
            qvel_max_rel,
            ")",
        )

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
# Test cases — no contacts since all geoms have contype=0
# =============================================================================


def test_zero_state_zero_action() raises:
    """All-zero qpos/qvel with no actions — pure gravity effect.
    The swimmer is planar (z=0), gravity acts but joints are zero."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step("Zero state, zero actions", qpos, qvel, actions)


def test_bent_joints_no_action() raises:
    """Non-zero joint angles, zero velocity, no actions.
    Tests dynamics starting from a curved configuration."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[3] = 0.5  # motor1_rot bent ~28.6 deg
    qpos[4] = -0.5  # motor2_rot bent in opposite direction
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step("Bent joints, no action", qpos, qvel, actions)


def test_with_motor_actions() raises:
    """Straight swimmer with both motors at max action.
    Tests actuator gear (150) application through RK4 stages."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 1.0  # motor1_rot at max
    actions[1] = -1.0  # motor2_rot at min
    compare_step("Max motor actions (1.0, -1.0)", qpos, qvel, actions)


def test_already_moving_with_actions() raises:
    """Swimmer already undulating (nonzero joint velocities) + actions.
    Tests velocity-dependent forces (Coriolis/centripetal) in the swimmer chain.
    """
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[2] = 0.3  # free_body_rot (torso rotated)
    qpos[3] = 0.4  # motor1_rot
    qpos[4] = -0.4  # motor2_rot
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 0.5  # slider1 velocity
    qvel[1] = 0.3  # slider2 velocity
    qvel[2] = 0.2  # free_body_rot velocity (spinning)
    qvel[3] = 1.0  # motor1_rot velocity
    qvel[4] = -1.0  # motor2_rot velocity
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.5
    actions[1] = -0.5
    compare_step("Moving swimmer + actions", qpos, qvel, actions)


def test_10_steps_undulating() raises:
    """10 steps of undulation — tests multi-step drift in the planar body chain.
    """
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.8
    actions[1] = -0.8
    compare_step(
        "10 steps undulating (max actions)", qpos, qvel, actions, num_steps=10
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

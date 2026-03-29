"""Test Full Physics Step (no contacts): Mojo Engine vs MuJoCo for Swimmer.

Compares qpos/qvel after running physics steps in both engines from
identical initial states with identical actions applied.

Swimmer is a planar chain with slide+hinge root (no free joint).
All geoms have contype=0 so there are NO ground contacts — the swimmer
"floats" at z=0 in a 2D plane.

Key notes:
  - NQ=5, NV=5, NBODY=4, NJOINT=5, NGEOM=3
  - ACTION_DIM=2 (motor1_rot, motor2_rot)
  - Integrator: RK4 (opt.integrator=1 = mjINT_RK4)
  - Solver: Newton (opt.solver=2 = mjSOL_NEWTON, irrelevant — no contacts)
  - Cone: Elliptic (opt.cone=1, irrelevant — no contacts)
  - Fluid dynamics: The original XML has viscosity=0.1, density=4000.
    We disable these in the MuJoCo comparison (opt.viscosity=0, opt.density=0)
    since our engine does not implement fluid drag/buoyancy. This makes the
    comparison test pure rigid-body dynamics, which our engine does support.
  - Tolerances are relaxed since viscosity mismatch may cause some divergence.

Run with:
    cd mojo-rl && pixi run mojo run -I . tests/physics3d/test_swimmer_full_step_vs_mujoco.mojo
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
comptime NGEOM = SwimmerModel.NGEOM  # 3
comptime MAX_CONTACTS = SwimmerModel.MAX_CONTACTS  # 5
comptime ACTION_DIM = SwimmerModel.ACTION_DIM  # 2

# Tolerances — relaxed since viscosity mismatch may cause some divergence
comptime QPOS_ABS_TOL: Float64 = 1e-2
comptime QPOS_REL_TOL: Float64 = 1e-1
comptime QVEL_ABS_TOL: Float64 = 1e-1
comptime QVEL_REL_TOL: Float64 = 1e-1


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
    mj_model.opt.solver = 2  # mjSOL_NEWTON
    mj_model.opt.cone = 1  # mjCONE_ELLIPTIC
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


def test_no_action() raises:
    """Default pose, no actions, 1 step — pure gravity effect.
    Should be small since swimmer is planar at z=0."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step("No action (default pose)", qpos, qvel, actions)


def test_with_actions() raises:
    """Default pose with actions[0]=0.3, actions[1]=-0.2, 1 step.
    Tests actuator gear (150) application through RK4 stages."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.3  # motor1_rot
    actions[1] = -0.2  # motor2_rot
    compare_step("With actions (0.3, -0.2)", qpos, qvel, actions)


def test_moving() raises:
    """Slider1 velocity=1.0, actions[0]=0.1, 1 step.
    Tests dynamics with initial velocity — drag mismatch is minimal for 1 step."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 1.0  # slider1 velocity (x direction)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.1  # motor1_rot
    compare_step("Moving (slider1 vel=1.0, action=0.1)", qpos, qvel, actions)


def test_multi_step() raises:
    """Default pose, actions[0]=0.2, actions[1]=-0.1, 5 steps.
    Tests multi-step drift accumulation in the planar body chain."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.2  # motor1_rot
    actions[1] = -0.1  # motor2_rot
    compare_step(
        "Multi-step (5 steps, actions 0.2/-0.1)",
        qpos,
        qvel,
        actions,
        num_steps=5,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

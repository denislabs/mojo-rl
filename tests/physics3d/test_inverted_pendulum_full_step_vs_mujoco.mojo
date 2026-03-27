"""Test Full Physics Step: Mojo Engine vs MuJoCo for InvertedPendulum.

Compares qpos/qvel after running physics steps in both engines from
identical initial states with identical actions applied.

InvertedPendulum is the simplest MuJoCo model: a cart on a rail with a pole.
NQ=2 (slider + hinge), NV=2, ACTION_DIM=1 (motor on slider).
Uses RK4 integrator to match the MuJoCo XML default.

No ground contacts expected (cart slides on rail).

Run with:
    cd mojo-rl && pixi run mojo run -I . tests/physics3d/test_inverted_pendulum_full_step_vs_mujoco.mojo
"""

from std.testing import assert_true, TestSuite
from std.python import Python, PythonObject
from std.math import abs
from std.collections import InlineArray

from mojo_rl.physics3d.types import Model, Data, ConeType
from mojo_rl.physics3d.integrator.rk4_integrator import RK4Integrator
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.envs.inverted_pendulum.inverted_pendulum_xml import InvertedPendulumModel


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64
comptime NQ = InvertedPendulumModel.NQ  # 2
comptime NV = InvertedPendulumModel.NV  # 2
comptime NBODY = InvertedPendulumModel.NBODY  # 2
comptime NJOINT = InvertedPendulumModel.NJOINT  # 2
comptime NGEOM = InvertedPendulumModel.NGEOM  # 3
comptime MAX_CONTACTS = InvertedPendulumModel.MAX_CONTACTS  # 5
comptime ACTION_DIM = 1

# Tight tolerances — simple 2-DOF dynamics
comptime QPOS_ABS_TOL: Float64 = 1e-4
comptime QPOS_REL_TOL: Float64 = 1e-3
comptime QVEL_ABS_TOL: Float64 = 1e-3
comptime QVEL_REL_TOL: Float64 = 1e-3


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
        InvertedPendulumModel.MAX_EQUALITY,
        InvertedPendulumModel.CONE_TYPE,
        InvertedPendulumModel.MAX_TENDON,
        InvertedPendulumModel.NSITE,
    ]()
    var data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, InvertedPendulumModel.NSITE
    ]()
    InvertedPendulumModel.setup_model_and_data(model, data)

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
        InvertedPendulumModel.apply_actions(data, action_list)
        RK4Integrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](model, data)

    print("data.num_contacts:", data.num_contacts)

    # === MuJoCo reference ===
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")

    var xml_path = "./references/Gymnasium-main/gymnasium/envs/mujoco/assets/inverted_pendulum.xml"
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.integrator = 1  # mjINT_RK4
    mj_model.opt.solver = 2  # mjSOL_NEWTON
    mj_model.opt.cone = 1  # mjCONE_ELLIPTIC
    var mj_data = mujoco.MjData(mj_model)

    for i in range(NQ):
        mj_data.qpos[i] = qpos_init[i]
    for i in range(NV):
        mj_data.qvel[i] = qvel_init[i]

    for i in range(ACTION_DIM):
        mj_data.ctrl[i] = actions[i]

    for _ in range(num_steps):
        mujoco.mj_step(mj_model, mj_data)

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

    # Print values
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

    print("  Our contacts:", Int(data.num_contacts))
    var mj_ncon = Int(py=mj_data.ncon)
    print("  MJ  contacts:", mj_ncon)
    assert_true(all_pass, "compare_step failed for: " + test_name)


# =============================================================================
# Test cases
# =============================================================================


def test_upright_no_action() raises:
    """Pole upright, no action — gravity pulls the pole down."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step("Upright, no action", qpos, qvel, actions)


def test_upright_with_push() raises:
    """Pole upright, push cart right with action=0.5."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.5
    compare_step("Upright, push right", qpos, qvel, actions)


def test_tilted() raises:
    """Pole tilted 0.1 rad, no action — gravity torque on pole."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.1  # hinge angle = 0.1 rad
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step("Tilted pole, no action", qpos, qvel, actions)


def test_tilted_with_action() raises:
    """Cart offset + tilted pole + action to push cart left."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 0.2  # cart at x=0.2
    qpos[1] = 0.1  # pole tilted 0.1 rad
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = -0.3
    compare_step("Tilted with action", qpos, qvel, actions)


def test_multi_step() raises:
    """10 steps with slight tilt and small action — accumulates drift."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.05  # slight tilt
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.1
    compare_step("Multi-step (10)", qpos, qvel, actions, num_steps=10)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

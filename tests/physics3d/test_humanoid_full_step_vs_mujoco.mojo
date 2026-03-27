"""Test Humanoid full step against MuJoCo reference.

Compares our RK4+Newton engine output against MuJoCo for the Humanoid model
(free joint + 17 hinges, 3D bipedal, with tendons).

  - Integrator: RK4 (opt.integrator=1 = mjINT_RK4)
  - Solver: Newton (opt.solver=2)
  - Cone: Elliptic (opt.cone=1)

Run with:
    cd mojo-rl && pixi run mojo run -I . tests/physics3d/test_humanoid_full_step_vs_mujoco.mojo
"""

from std.testing import assert_true, TestSuite
from std.python import Python, PythonObject
from std.math import abs
from std.collections import InlineArray

from mojo_rl.physics3d.types import Model, Data, ConeType
from mojo_rl.physics3d.integrator.rk4_integrator import RK4Integrator
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.envs.humanoid.humanoid_xml import HumanoidModel
from mojo_rl.envs.humanoid.humanoid_config import HumanoidConfig


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64
comptime NQ = HumanoidModel.NQ  # 24
comptime NV = HumanoidModel.NV  # 23
comptime NBODY = HumanoidModel.NBODY  # 14
comptime NJOINT = HumanoidModel.NJOINT  # 18
comptime NGEOM = HumanoidModel.NGEOM  # 18
comptime MAX_CONTACTS = HumanoidModel.MAX_CONTACTS  # 50
comptime ACTION_DIM = HumanoidModel.ACTION_DIM  # 17

# Tolerances
comptime QPOS_ABS_TOL: Float64 = 1e-3
comptime QPOS_REL_TOL: Float64 = 1e-2
comptime QVEL_ABS_TOL: Float64 = 1e-2
comptime QVEL_REL_TOL: Float64 = 1e-2


# =============================================================================
# Comparison function
# =============================================================================


def compare_step(
    test_name: String,
    qpos_init: InlineArray[Float64, NQ],
    qvel_init: InlineArray[Float64, NV],
    actions: InlineArray[Float64, ACTION_DIM],
    num_steps: Int = 1,
) raises:
    print("--- Test:", test_name, "(", num_steps, "steps) ---")

    # === Our engine (RK4 + Newton) ===
    var model = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
        HumanoidModel.MAX_EQUALITY, HumanoidModel.CONE_TYPE,
        HumanoidModel.MAX_TENDON, HumanoidModel.NSITE,
    ]()
    var data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HumanoidModel.NSITE,
    ]()
    HumanoidModel.setup_model_and_data(model, data)

    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        data.qvel[i] = Scalar[DTYPE](qvel_init[i])

    var action_list = List[Float64](capacity=ACTION_DIM)
    for i in range(ACTION_DIM):
        action_list.append(actions[i])

    for _ in range(num_steps):
        for i in range(NV):
            data.qfrc[i] = Scalar[DTYPE](0)
        HumanoidModel.apply_actions(data, action_list)
        RK4Integrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](model, data)

    # === MuJoCo reference (RK4) ===
    var mujoco = Python.import_module("mujoco")
    var xml_path = "./references/Gymnasium-main/gymnasium/envs/mujoco/assets/humanoid.xml"
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    var mj_data = mujoco.MjData(mj_model)

    mj_model.opt.integrator = 1  # RK4
    mj_model.opt.solver = 2      # Newton
    mj_model.opt.cone = 1        # Elliptic

    for i in range(NQ):
        mj_data.qpos[i] = qpos_init[i]
    for i in range(NV):
        mj_data.qvel[i] = qvel_init[i]
    for i in range(ACTION_DIM):
        mj_data.ctrl[i] = actions[i]

    for _ in range(num_steps):
        mujoco.mj_step(mj_model, mj_data)

    # === Extract results ===
    var mj_ncon = Int(py=mj_data.ncon)
    var mj_qpos = mj_data.qpos.flatten().tolist()
    var mj_qvel = mj_data.qvel.flatten().tolist()

    print("  Our contacts:", data.num_contacts)
    print("  MJ  contacts:", mj_ncon)

    # === Compare ===
    var all_pass = True
    var qpos_fails = 0
    var qvel_fails = 0
    var qpos_max_abs: Float64 = 0.0
    var qpos_max_rel: Float64 = 0.0
    var qvel_max_abs: Float64 = 0.0
    var qvel_max_rel: Float64 = 0.0

    # qpos comparison
    for i in range(NQ):
        var ours = Float64(data.qpos[i])
        var mj = Float64(py=mj_qpos[i])
        var abs_err = abs(ours - mj)
        var rel_err: Float64 = 0.0
        if abs(mj) > 1e-10:
            rel_err = abs_err / abs(mj)
        if abs_err > qpos_max_abs:
            qpos_max_abs = abs_err
        if rel_err > qpos_max_rel:
            qpos_max_rel = rel_err
        var ok = abs_err < QPOS_ABS_TOL or rel_err < QPOS_REL_TOL
        if not ok:
            if qpos_fails < 3:
                print("  FAIL qpos[", i, "]  ours=", ours, " mj=", mj, " abs=", abs_err, " rel=", rel_err)
            qpos_fails += 1
            all_pass = False

    # qvel comparison
    for i in range(NV):
        var ours = Float64(data.qvel[i])
        var mj = Float64(py=mj_qvel[i])
        var abs_err = abs(ours - mj)
        var rel_err: Float64 = 0.0
        if abs(mj) > 1e-10:
            rel_err = abs_err / abs(mj)
        if abs_err > qvel_max_abs:
            qvel_max_abs = abs_err
        if rel_err > qvel_max_rel:
            qvel_max_rel = rel_err
        var ok = abs_err < QVEL_ABS_TOL or rel_err < QVEL_REL_TOL
        if not ok:
            if qvel_fails < 3:
                print("  FAIL qvel[", i, "]  ours=", ours, " mj=", mj, " abs=", abs_err, " rel=", rel_err)
            qvel_fails += 1
            all_pass = False

    if all_pass:
        print("  ALL OK  qpos_max_abs=", qpos_max_abs, " qpos_max_rel=", qpos_max_rel,
              " qvel_max_abs=", qvel_max_abs, " qvel_max_rel=", qvel_max_rel)
    else:
        print("  FAILED  qpos:", qpos_fails, "fails (max_abs=", qpos_max_abs,
              " max_rel=", qpos_max_rel, ")  qvel:", qvel_fails, "fails (max_abs=", qvel_max_abs,
              " max_rel=", qvel_max_rel, ")")

    # Print state vectors
    var our_qpos_str = String("")
    var mj_qpos_str = String("")
    for i in range(NQ):
        our_qpos_str += "  " + String(Float64(data.qpos[i]))
        mj_qpos_str += "  " + String(Float64(py=mj_qpos[i]))
    print("  Our qpos:", our_qpos_str)
    print("  MJ  qpos:", mj_qpos_str)

    var our_qvel_str = String("")
    var mj_qvel_str = String("")
    for i in range(min(6, NV)):
        our_qvel_str += "  " + String(Float64(data.qvel[i]))
        mj_qvel_str += "  " + String(Float64(py=mj_qvel[i]))
    print("  Our qvel[0:6]:", our_qvel_str)
    print("  MJ  qvel[0:6]:", mj_qvel_str)

    assert_true(all_pass, "compare_step failed for: " + test_name)


# =============================================================================
# Test cases
# =============================================================================


def test_free_fall() raises:
    """Free fall from height, no contacts, no actions. Tests pure dynamics."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[0] = 0.0  # x
    qpos[1] = 0.0  # y
    qpos[2] = 3.0  # z (high, no ground contact)
    qpos[3] = 1.0  # qw (identity)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step("Free fall, no actions", qpos, qvel, actions)


def test_free_fall_with_actions() raises:
    """Free fall with motor actions. Tests actuator force application."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[2] = 3.0
    qpos[3] = 1.0
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.3   # abdomen_y
    actions[1] = -0.2  # abdomen_z
    actions[5] = 0.4   # right_hip_y (strongest motor, gear=300)
    actions[9] = -0.4  # left_hip_y
    compare_step("Free fall with actions", qpos, qvel, actions)


def test_standing_no_action() raises:
    """Default standing pose, no actions. Tests ground contact handling."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[2] = 1.4   # standing height
    qpos[3] = 1.0   # qw
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step("Standing, no action", qpos, qvel, actions)


def test_standing_with_actions() raises:
    """Standing pose with moderate actions. Tests contacts + actuation."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[2] = 1.4
    qpos[3] = 1.0
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.2   # abdomen_y
    actions[5] = 0.3   # right_hip_y
    actions[6] = -0.2  # right_knee
    actions[9] = -0.3  # left_hip_y
    actions[10] = 0.2  # left_knee
    compare_step("Standing with actions", qpos, qvel, actions)


def test_falling_10_steps() raises:
    """Free fall for 10 steps. Tests drift accumulation."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[2] = 3.0
    qpos[3] = 1.0
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[5] = 0.2
    actions[9] = -0.2
    compare_step("Free fall, 10 steps", qpos, qvel, actions, num_steps=10)


def test_ground_contact_10_steps() raises:
    """Standing with contacts for 10 steps. Tests sustained contact stability."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[2] = 1.4
    qpos[3] = 1.0
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step("Ground contact, 10 steps", qpos, qvel, actions, num_steps=10)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

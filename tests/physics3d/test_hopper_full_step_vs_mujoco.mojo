"""Test Full Physics Step (no contacts): Mojo Engine vs MuJoCo for Hopper.

Compares qpos/qvel after running physics steps in both engines from
identical initial states with identical actions applied.

Hopper uses ELLIPTIC cone (default), providing coverage for non-pyramidal cone.
Tests use Euler integrator for consistency with MuJoCo comparison.

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_hopper_full_step_vs_mujoco.mojo
"""

from std.testing import assert_true, TestSuite
from std.python import Python, PythonObject
from std.math import abs
from std.collections import InlineArray

from physics3d.types import Model, Data, ConeType
from physics3d.integrator.euler_integrator import EulerIntegrator
from physics3d.solver import NewtonSolver
from envs.hopper.hopper_xml import HopperModel
from envs.hopper.hopper_config import HopperConfig


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64
comptime NQ = HopperModel.NQ  # 6
comptime NV = HopperModel.NV  # 6
comptime NBODY = HopperModel.NBODY  # 5
comptime NJOINT = HopperModel.NJOINT  # 6
comptime NGEOM = HopperModel.NGEOM  # 5
comptime MAX_CONTACTS = HopperConfig.MAX_CONTACTS  # 20
comptime ACTION_DIM = HopperConfig.ACTION_DIM  # 3

# Tolerances
comptime QPOS_ABS_TOL: Float64 = 1e-3
comptime QPOS_REL_TOL: Float64 = 1e-2
comptime QVEL_ABS_TOL: Float64 = 1e-2
comptime QVEL_REL_TOL: Float64 = 1e-2


# =============================================================================
# Comparison helper
# =============================================================================


fn compare_step(
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
        HopperModel.MAX_EQUALITY,
        HopperModel.CONE_TYPE,
        HopperModel.MAX_TENDON,
        HopperModel.NSITE,
    ]()
    var data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HopperModel.NSITE
    ]()
    HopperModel.setup_model_and_data(model, data)

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
        HopperModel.apply_actions(data, action_list)
        EulerIntegrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](model, data)

    print("data.num_contacts:", data.num_contacts)

    # === MuJoCo reference ===
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")

    var xml_path = "../Gymnasium-main/gymnasium/envs/mujoco/assets/hopper.xml"
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.integrator = 0  # mjINT_EULER
    mj_model.opt.solver = 2  # mjSOL_NEWTON
    mj_model.opt.cone = 1  # mjCONE_ELLIPTIC (matches HopperModel)
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
# Test cases — freefall / high enough to avoid contacts
# =============================================================================


fn test_freefall() raises:
    """Free fall from default height — no contacts expected.
    Hopper torso is at body_pos z=1.25, so qpos rootz=0 is high enough."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.5  # rootz extra offset => torso at 1.75, no contacts
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step("Free fall (no contacts)", qpos, qvel, actions)


fn test_standing_zero_action() raises:
    """Standing at default height — may have ground contact."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step("Standing, zero action", qpos, qvel, actions)


fn test_standing_with_action() raises:
    """Standing with moderate actions applied."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.5  # thigh
    actions[1] = -0.3  # leg
    actions[2] = 0.2  # foot
    compare_step("Standing, moderate action", qpos, qvel, actions)


fn test_moving_with_action() raises:
    """Robot already moving with velocity + actions."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[2] = 0.1  # rooty slight pitch
    qpos[3] = -0.3  # thigh_joint
    qpos[5] = 0.2  # foot_joint
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 2.0  # rootx vel (running)
    qvel[2] = 0.5  # rooty vel
    qvel[3] = -1.0  # thigh vel
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 1.0  # max thigh
    actions[1] = -0.5  # leg
    actions[2] = 0.3  # foot
    compare_step("Moving with actions", qpos, qvel, actions)


fn test_freefall_10_steps() raises:
    """Free fall 10 steps — accumulates any per-step drift."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 0.5  # high enough for 10 steps of free fall
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step("Free fall (10 steps)", qpos, qvel, actions, num_steps=10)


fn test_standing_10_steps() raises:
    """Standing 10 steps with actions — tests solver stability."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.5  # thigh
    actions[1] = -0.3  # leg
    compare_step(
        "Standing with action (10 steps)", qpos, qvel, actions, num_steps=10
    )


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

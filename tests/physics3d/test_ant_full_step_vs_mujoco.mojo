"""Test Full Physics Step: Mojo Engine vs MuJoCo for Ant.

Compares qpos/qvel after running physics steps in both engines from
identical initial states with identical actions applied.

The Ant is the first test with a 3D free joint (NQ=15, NV=14), symmetric
4-leg body tree, and 8 actuators. This exercises full 3D rigid body dynamics
including quaternion integration, unlike the planar HalfCheetah/Hopper tests.

Ant uses:
  - Integrator: RK4 (opt.integrator=1 = mjINT_RK4)
  - Solver: Newton (opt.solver=2 = mjSOL_NEWTON)
  - Cone: Elliptic (opt.cone=1 = mjCONE_ELLIPTIC, default)
  - FRAME_SKIP=5 (MuJoCo comparison runs 1 step at a time)

Note: The Ant's default init_qpos places the torso at z=0.55 above the ground.
Free-flight configs start at z=2.0 to ensure no contacts for 1-step tests.

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_ant_full_step_vs_mujoco.mojo
"""

from std.testing import assert_true, TestSuite
from std.python import Python, PythonObject
from std.math import abs
from std.collections import InlineArray

from mojo_rl.physics3d.types import Model, Data, ConeType
from mojo_rl.physics3d.integrator.rk4_integrator import RK4Integrator
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.envs.ant.ant_xml import AntModel
from mojo_rl.envs.ant.ant_config import AntConfig


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64
comptime NQ = AntModel.NQ  # 15
comptime NV = AntModel.NV  # 14
comptime NBODY = AntModel.NBODY
comptime NJOINT = AntModel.NJOINT
comptime NGEOM = AntModel.NGEOM
comptime MAX_CONTACTS = AntModel.MAX_CONTACTS  # 40
comptime ACTION_DIM = AntConfig.ACTION_DIM  # 8

# Tolerances — RK4 is 4th-order; expect tight match for no-contact cases.
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
        AntModel.MAX_EQUALITY,
        AntModel.CONE_TYPE,
        AntModel.MAX_TENDON,
        AntModel.NSITE,
    ]()
    var data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, AntModel.NSITE
    ]()
    AntModel.setup_model_and_data(model, data)

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
        AntModel.apply_actions(data, action_list)
        RK4Integrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](model, data)

    print("  Our contacts:", Int(data.num_contacts))

    # === MuJoCo reference (RK4) ===
    var mujoco = Python.import_module("mujoco")

    var xml_path = "../Gymnasium-main/gymnasium/envs/mujoco/assets/ant.xml"
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

    # Print full state for inspection
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


def test_free_fall() raises:
    """Free fall from high altitude — no contacts, pure 3D rigid-body dynamics.
    Torso starts at z=2.0 with identity quaternion and all joints at 0."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[2] = 2.0  # z high enough to avoid ground contact for 1 step
    qpos[3] = 1.0  # qw (identity quaternion — upright torso)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step("Free fall from z=2.0 (no contacts)", qpos, qvel, actions)


def test_free_fall_with_actions() raises:
    """Free fall with all 8 motors firing — exercises actuator mapping
    through the RK4 stages for a 3D free-body system."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[2] = 2.0  # high enough for no contacts
    qpos[3] = 1.0  # qw
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    # Moderate symmetric hip actions + ankle actions
    actions[0] = 0.5  # hip_4
    actions[1] = -0.3  # ankle_4
    actions[2] = 0.5  # hip_1
    actions[3] = -0.3  # ankle_1
    actions[4] = -0.5  # hip_2
    actions[5] = 0.3  # ankle_2
    actions[6] = -0.5  # hip_3
    actions[7] = 0.3  # ankle_3
    compare_step("Free fall with moderate actions", qpos, qvel, actions)


def test_default_pose_no_action() raises:
    """Default Ant init_qpos (z=0.55), zero velocity, no actions.
    Tests standing pose — may have ground contacts."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    # Exact init_qpos from XML
    qpos[0] = 0.0
    qpos[1] = 0.0
    qpos[2] = 0.55
    qpos[3] = 1.0
    qpos[4] = 0.0
    qpos[5] = 0.0
    qpos[6] = 0.0
    qpos[7] = 0.0
    qpos[8] = 1.0
    qpos[9] = 0.0
    qpos[10] = -1.0
    qpos[11] = 0.0
    qpos[12] = -1.0
    qpos[13] = 0.0
    qpos[14] = 1.0
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step("Default init_qpos, no action", qpos, qvel, actions)


def test_default_pose_with_actions() raises:
    """Default Ant pose with full motor actions — exercises combined
    contact + actuation through RK4 stages."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[2] = 0.55
    qpos[3] = 1.0
    qpos[8] = 1.0
    qpos[10] = -1.0
    qpos[12] = -1.0
    qpos[14] = 1.0
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.8
    actions[1] = 0.8
    actions[2] = 0.8
    actions[3] = 0.8
    actions[4] = -0.8
    actions[5] = -0.8
    actions[6] = -0.8
    actions[7] = -0.8
    compare_step("Default pose, max symmetric actions", qpos, qvel, actions)


def test_moving_with_velocity() raises:
    """Ant already translating and rotating — tests free-joint velocity
    integration (linear + angular velocity through quaternion update)."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[2] = 1.5  # elevated to avoid immediate contacts
    qpos[3] = 1.0  # qw
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 1.0  # vx (translational)
    qvel[1] = 0.5  # vy
    qvel[2] = -0.2  # vz (falling slowly)
    qvel[3] = 0.3  # wx (angular velocity about x)
    qvel[4] = 0.1  # wy
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.3
    actions[2] = 0.3
    compare_step("Moving torso + velocity + actions", qpos, qvel, actions)


def test_free_fall_10_steps() raises:
    """Free fall 10 steps — accumulates any per-step integration drift
    in the quaternion and free-joint position."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[2] = 5.0  # start high enough to stay airborne for 10 steps
    qpos[3] = 1.0  # qw
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step("Free fall 10 steps", qpos, qvel, actions, num_steps=10)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

"""Test Full Physics Step: Mojo Engine vs MuJoCo for Walker2d.

Compares qpos/qvel after running physics steps in both engines from
identical initial states with identical actions applied.

Walker2d uses RK4 integrator (timestep=0.002) and has a similar structure
to Hopper (slide+hinge root, no free joint) but with two legs (6 actuators).

Run with:
    cd mojo-rl && pixi run mojo run -I . tests/physics3d/test_walker2d_full_step_vs_mujoco.mojo
"""

from std.testing import assert_true, TestSuite
from std.python import Python, PythonObject
from std.math import abs
from std.collections import InlineArray

from mojo_rl.physics3d.types import Model, Data, ConeType
from mojo_rl.physics3d.integrator.rk4_integrator import RK4Integrator
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64
comptime NQ = Walker2dModel.NQ  # 9
comptime NV = Walker2dModel.NV  # 9
comptime NBODY = Walker2dModel.NBODY  # 8
comptime NJOINT = Walker2dModel.NJOINT  # 9
comptime NGEOM = Walker2dModel.NGEOM  # 8
comptime MAX_CONTACTS = Walker2dModel.MAX_CONTACTS  # 20
comptime ACTION_DIM = Walker2dModel.ACTION_DIM  # 6

# Tolerances
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
        Walker2dModel.MAX_EQUALITY,
        Walker2dModel.CONE_TYPE,
        Walker2dModel.MAX_TENDON,
        Walker2dModel.NSITE,
    ]()
    var data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, Walker2dModel.NSITE
    ]()
    Walker2dModel.setup_model_and_data(model, data)

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
        Walker2dModel.apply_actions(data, action_list)
        RK4Integrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](model, data)

    print("data.num_contacts:", data.num_contacts)

    # === MuJoCo reference ===
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")

    var xml_path = "./references/Gymnasium-main/gymnasium/envs/mujoco/assets/walker2d.xml"
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.integrator = 1  # mjINT_RK4
    mj_model.opt.solver = 2  # mjSOL_NEWTON
    mj_model.opt.cone = 0  # mjCONE_PYRAMIDAL (matches Walker2dModel default)
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


def test_free_fall() raises:
    """Free fall from high position — no contacts expected.
    Walker2d torso is at body_pos z=1.25, rootz ref=1.25.
    Setting qpos rootz=3.0 places torso very high."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 3.0  # rootz high => no contacts
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step("Free fall (no contacts)", qpos, qvel, actions)


def test_free_fall_with_actions() raises:
    """Free fall from high position with actions applied."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 3.0  # rootz high => no contacts
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.3  # thigh_joint
    actions[3] = -0.3  # thigh_left_joint
    compare_step("Free fall with actions", qpos, qvel, actions)


def test_standing_no_action() raises:
    """Standing with asymmetric joint angles to break L/R symmetry.
    qpos=0 puts both legs at identical positions with identical joint limits,
    creating an ill-conditioned constraint problem where MuJoCo breaks symmetry
    via solver ordering. Use slightly different L/R angles instead."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    # Asymmetric joint angles to avoid ill-conditioned symmetric problem
    qpos[3] = -0.1  # thigh_joint (right)
    qpos[4] = -0.2  # leg_joint (right)
    qpos[6] = -0.15  # thigh_left_joint
    qpos[7] = -0.25  # leg_left_joint
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step("Standing, asymmetric joints", qpos, qvel, actions)


def test_standing_with_actions() raises:
    """Standing with asymmetric joints and moderate actions."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[3] = -0.1  # thigh_joint (right)
    qpos[4] = -0.2  # leg_joint (right)
    qpos[6] = -0.15  # thigh_left_joint
    qpos[7] = -0.25  # leg_left_joint
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.5  # thigh_joint
    actions[1] = -0.3  # leg_joint
    actions[2] = 0.2  # foot_joint
    actions[3] = -0.4  # thigh_left_joint
    actions[4] = 0.3  # leg_left_joint
    actions[5] = -0.1  # foot_left_joint
    compare_step("Standing, asymmetric + actions", qpos, qvel, actions)


def test_falling_10_steps() raises:
    """Free fall 10 steps with small actions — accumulates drift."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 3.0  # rootz high => no contacts for 10 steps
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.1  # small thigh
    actions[2] = -0.1  # small foot
    actions[4] = 0.1  # small leg_left
    compare_step("Free fall (10 steps)", qpos, qvel, actions, num_steps=10)


def test_ground_contact_10_steps() raises:
    """10 steps with asymmetric joints — tests contact solver stability."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[3] = -0.1  # thigh_joint (right)
    qpos[4] = -0.2  # leg_joint (right)
    qpos[6] = -0.15  # thigh_left_joint
    qpos[7] = -0.25  # leg_left_joint
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step(
        "Ground contact, asymmetric (10 steps)",
        qpos,
        qvel,
        actions,
        num_steps=10,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

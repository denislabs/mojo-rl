"""Test Full Physics Step: Mojo Engine vs MuJoCo Reference.

Compares qpos/qvel after running 1 physics step in both engines from
identical initial states with identical actions applied.

This is the ultimate integration test — it validates the entire pipeline:
  FK → contacts → constraint building → solver → integration

To match our engine settings (forced in MuJoCo via opt):
  - Integrator: Euler (opt.integrator=0)
  - Solver: Newton (opt.solver=2)
  - Cone: elliptic (opt.cone=1)
  - timestep: 0.01
  - gravity: -9.81

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_full_step_vs_mujoco.mojo
"""

from python import Python, PythonObject
from math import abs
from collections import InlineArray

from physics3d.types import Model, Data, _max_one, ConeType
from physics3d.integrator.euler_integrator import EulerIntegrator
from physics3d.solver import NewtonSolver
from physics3d.kinematics.forward_kinematics import forward_kinematics
from physics3d.dynamics.mass_matrix import compute_body_invweight0
from envs.half_cheetah.half_cheetah_def import (
    HalfCheetahModel,
    HalfCheetahBodies,
    HalfCheetahJoints,
    HalfCheetahGeoms,
    HalfCheetahActuators,
    HalfCheetahParams,
)


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64
comptime NQ = HalfCheetahModel.NQ  # 9
comptime NV = HalfCheetahModel.NV  # 9
comptime NBODY = HalfCheetahModel.NBODY  # 8
comptime NJOINT = HalfCheetahModel.NJOINT  # 10
comptime NGEOM = HalfCheetahModel.NGEOM  # 9
comptime MAX_CONTACTS = HalfCheetahParams[DTYPE].MAX_CONTACTS  # 20
comptime ACTION_DIM = HalfCheetahParams[DTYPE].ACTION_DIM  # 6

# Tolerances
# Single-step: we expect reasonable agreement but solver convergence paths
# may differ (our solver vs MuJoCo's). Use relative tolerance primarily.
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
) raises -> Bool:
    """Run num_steps physics steps in both engines, compare final qpos/qvel."""
    print("--- Test:", test_name, "---")
    print("  Steps:", num_steps)

    # === Our engine ===
    var model = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, 0, ConeType.ELLIPTIC
    ](
        gravity_z=Scalar[DTYPE](-9.81),
        timestep=Scalar[DTYPE](0.01),
    )

    HalfCheetahBodies.setup_model(model)
    HalfCheetahJoints.setup_model(model)
    HalfCheetahGeoms.setup_model(model)

    # Set solver params to match MuJoCo defaults
    comptime P = HalfCheetahParams[DTYPE]
    model.solref_contact[0] = P.SOLREF_CONTACT_0
    model.solref_contact[1] = P.SOLREF_CONTACT_1
    model.solimp_contact[0] = P.SOLIMP_CONTACT_0
    model.solimp_contact[1] = P.SOLIMP_CONTACT_1
    model.solimp_contact[2] = P.SOLIMP_CONTACT_2
    model.solref_limit[0] = P.SOLREF_LIMIT_0
    model.solref_limit[1] = P.SOLREF_LIMIT_1
    model.solimp_limit[0] = P.SOLIMP_LIMIT_0
    model.solimp_limit[1] = P.SOLIMP_LIMIT_1
    model.solimp_limit[2] = P.SOLIMP_LIMIT_2

    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()

    # Compute body_invweight0 at REFERENCE pose (MuJoCo mj_setConst does this once at init)
    forward_kinematics(model, data)
    compute_body_invweight0[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
    ](model, data)

    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        data.qvel[i] = Scalar[DTYPE](qvel_init[i])

    # Apply actions via actuators (sets data.qfrc)
    var action_list = List[Float64]()
    for i in range(ACTION_DIM):
        action_list.append(actions[i])

    for _ in range(num_steps):
        # Zero qfrc before applying actions (actions are applied each step)
        for i in range(NV):
            data.qfrc[i] = Scalar[DTYPE](0)

        HalfCheetahActuators.apply_actions[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS
        ](data, action_list)

        EulerIntegrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](
            model, data
        )

    print("data.num_contacts:", data.num_contacts)

    # === MuJoCo reference ===
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")

    var xml_path = (
        "../Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml"
    )
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    # Match our engine settings: Euler integrator, Newton solver, elliptic cone
    mj_model.opt.integrator = 0  # mjINT_EULER
    mj_model.opt.solver = 2  # mjSOL_NEWTON
    mj_model.opt.cone = 1  # mjCONE_ELLIPTIC
    var mj_data = mujoco.MjData(mj_model)

    for i in range(NQ):
        mj_data.qpos[i] = qpos_init[i]
    for i in range(NV):
        mj_data.qvel[i] = qvel_init[i]

    # Set ctrl (actuator inputs)
    for i in range(ACTION_DIM):
        mj_data.ctrl[i] = actions[i]

    # Run steps
    for step in range(num_steps):
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

    # Print values for inspection
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

    # Also print contacts detected
    print("  Our contacts:", Int(data.num_contacts))

    # Get MuJoCo contact count
    var mj_ncon = Int(py=mj_data.ncon)
    print("  MJ  contacts:", mj_ncon)

    return all_pass


# =============================================================================
# Test cases
# =============================================================================


fn test_freefall() raises -> Bool:
    """Free fall from default height — no contacts expected."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 1.5  # rootz high enough to avoid ground contact
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    return compare_step("Free fall (no contacts)", qpos, qvel, actions)


fn test_standing_zero_action() raises -> Bool:
    """Standing at default height — contacts with ground, no actions."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    # MuJoCo half_cheetah default: rootz body_pos = 0.7, but our body_pos
    # already encodes this. qpos rootz=0 means at body_pos height.
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    return compare_step("Standing, zero action", qpos, qvel, actions)


fn test_standing_with_action() raises -> Bool:
    """Standing with moderate actions applied."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.5  # bthigh
    actions[1] = -0.3  # bshin
    actions[2] = 0.2  # bfoot
    actions[3] = 0.5  # fthigh
    actions[4] = -0.3  # fshin
    actions[5] = 0.1  # ffoot
    return compare_step("Standing, moderate action", qpos, qvel, actions)


fn test_moving_with_action() raises -> Bool:
    """Robot already moving with velocity + actions."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[2] = 0.1  # rooty = 0.1 (slight pitch)
    qpos[3] = -0.3  # bthigh
    qpos[6] = 0.4  # fthigh
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 2.0  # rootx vel (running)
    qvel[2] = 0.5  # rooty vel (pitching)
    qvel[3] = -1.0  # bthigh vel
    qvel[6] = 1.2  # fthigh vel
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 1.0  # max bthigh
    actions[1] = -0.5  # bshin
    actions[3] = 1.0  # max fthigh
    actions[4] = -0.5  # fshin
    return compare_step("Moving with actions", qpos, qvel, actions)


fn test_freefall_10_steps() raises -> Bool:
    """Free fall 10 steps — accumulates any per-step drift."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 1.5  # high enough for 10 steps of free fall
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    return compare_step(
        "Free fall (10 steps)", qpos, qvel, actions, num_steps=10
    )


fn test_standing_10_steps() raises -> Bool:
    """Standing 10 steps with actions — tests solver stability."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.5  # bthigh
    actions[3] = 0.5  # fthigh
    return compare_step(
        "Standing with action (10 steps)", qpos, qvel, actions, num_steps=10
    )


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    print("=" * 60)
    print("Full Step Validation: Mojo Engine vs MuJoCo Reference")
    print("=" * 60)
    print("Model: HalfCheetah (NQ=9, NV=9)")
    print("Integrator: Euler (opt.integrator=0)")
    print("Solver: Newton (opt.solver=2)")
    print("Cone: elliptic (opt.cone=1)")
    print("Precision: float64")
    print("Tolerances: qpos abs=", QPOS_ABS_TOL, " rel=", QPOS_REL_TOL)
    print("            qvel abs=", QVEL_ABS_TOL, " rel=", QVEL_REL_TOL)
    print()

    var num_pass = 0
    var num_fail = 0

    if test_freefall():
        num_pass += 1
    else:
        num_fail += 1
    print()

    if test_standing_zero_action():
        num_pass += 1
    else:
        num_fail += 1
    print()

    if test_standing_with_action():
        num_pass += 1
    else:
        num_fail += 1
    print()

    if test_moving_with_action():
        num_pass += 1
    else:
        num_fail += 1
    print()

    if test_freefall_10_steps():
        num_pass += 1
    else:
        num_fail += 1
    print()

    if test_standing_10_steps():
        num_pass += 1
    else:
        num_fail += 1
    print()

    print("=" * 60)
    print(
        "Results:",
        num_pass,
        "passed,",
        num_fail,
        "failed out of",
        num_pass + num_fail,
    )
    if num_fail == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

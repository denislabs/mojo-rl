"""Test Full Physics Step with Ground Contacts: Mojo Engine vs MuJoCo.

Separated from test_full_step_vs_mujoco.mojo because having too many
test functions in one file causes Mojo compiler stack overflow with
the heavily-generic constraint solver code.

Tests scenarios where the robot makes ground contact, exercising the
full constraint solver pipeline (contact detection + Jacobians + solver).
Includes multi-step and high-velocity impact tests to diagnose solver
accuracy under fast foot strikes.

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_full_step_contact_vs_mujoco.mojo
"""

from python import Python, PythonObject
from math import abs
from collections import InlineArray

from physics3d.types import Model, Data, ConeType
from physics3d.integrator.euler_integrator import EulerIntegrator
from physics3d.solver import NewtonSolver
from envs.half_cheetah.half_cheetah_xml import HalfCheetahModel
from envs.half_cheetah.half_cheetah_config import HalfCheetahConfig


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64
comptime NQ = HalfCheetahModel.NQ  # 9
comptime NV = HalfCheetahModel.NV  # 9
comptime NBODY = HalfCheetahModel.NBODY  # 7
comptime NJOINT = HalfCheetahModel.NJOINT  # 9
comptime NGEOM = HalfCheetahModel.NGEOM  # 9
comptime MAX_CONTACTS = HalfCheetahConfig.MAX_CONTACTS  # 20
comptime ACTION_DIM = HalfCheetahConfig.ACTION_DIM  # 6

# Tolerances — relaxed for contact scenarios. Remaining ~5-10% error is from
# contact geometry differences (our contact detection gives slightly different
# contact positions/normals than MuJoCo's), NOT from solver accuracy.
# D values match MuJoCo exactly; solver converges fully.
comptime QPOS_ABS_TOL: Float64 = 2e-4
comptime QPOS_REL_TOL: Float64 = 2e-4
comptime QVEL_ABS_TOL: Float64 = 2e-4
comptime QVEL_REL_TOL: Float64 = 2e-4

# Looser tolerances for multi-step tests — errors accumulate over steps.
# These are diagnostic (we want to SEE how errors grow, not hide them).
comptime MULTI_QPOS_ABS_TOL: Float64 = 5e-3
comptime MULTI_QPOS_REL_TOL: Float64 = 5e-3
comptime MULTI_QVEL_ABS_TOL: Float64 = 5e-3
comptime MULTI_QVEL_REL_TOL: Float64 = 5e-3


# =============================================================================
# Comparison helpers
# =============================================================================


fn compare_step(
    test_name: String,
    qpos_init: InlineArray[Float64, NQ],
    qvel_init: InlineArray[Float64, NV],
    actions: InlineArray[Float64, ACTION_DIM],
    num_steps: Int = 1,
    pos_abs_tol: Float64 = QPOS_ABS_TOL,
    pos_rel_tol: Float64 = QPOS_REL_TOL,
    vel_abs_tol: Float64 = QVEL_ABS_TOL,
    vel_rel_tol: Float64 = QVEL_REL_TOL,
) raises -> Bool:
    """Run num_steps physics steps in both engines, compare final qpos/qvel."""
    print("--- Test:", test_name, "---")
    print("  Steps:", num_steps)

    # === Our engine ===
    var model = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, 0, HalfCheetahModel.CONE_TYPE
    ](
    )
    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()
    HalfCheetahModel.setup_model_and_data(model, data)

    # Now set test configuration
    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        data.qvel[i] = Scalar[DTYPE](qvel_init[i])

    # Apply actions via actuators (sets data.qfrc)
    var action_list = List[Float64]()
    for i in range(ACTION_DIM):
        action_list.append(actions[i])

    for _ in range(num_steps):
        for i in range(NV):
            data.qfrc[i] = Scalar[DTYPE](0)

        HalfCheetahModel.apply_actions(data, action_list)

        EulerIntegrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](
            model, data
        )

    # === MuJoCo reference ===
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")

    var xml_path = (
        "../Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml"
    )
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    # Match our pyramidal cone setting
    mj_model.opt.cone = 0  # mjCONE_PYRAMIDAL (matches HalfCheetahModel)
    mj_model.opt.solver = 2  # mjSOL_NEWTON to match our NewtonSolver
    mj_model.opt.integrator = 0  # mjINT_EULER to match our EulerIntegrator
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

        var ok = abs_err < pos_abs_tol or rel_err < pos_rel_tol
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

        var ok = abs_err < vel_abs_tol or rel_err < vel_rel_tol
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

    # Print contact details for diagnosis
    var our_ncon = Int(data.num_contacts)
    if our_ncon > 0:
        print("  --- Contact details ---")
        for c in range(our_ncon):
            print(
                "  Our contact[",
                c,
                "]: body_a=",
                Int(data.contacts[c].body_a),
                " body_b=",
                Int(data.contacts[c].body_b),
                " pos=(",
                Float64(data.contacts[c].pos_x),
                ",",
                Float64(data.contacts[c].pos_y),
                ",",
                Float64(data.contacts[c].pos_z),
                ")",
                " dist=",
                Float64(data.contacts[c].dist),
                " force_n=",
                Float64(data.contacts[c].force_n),
            )

    if mj_ncon > 0:
        var mj_contacts = mj_data.contact
        for c in range(mj_ncon):
            var mj_c = mj_contacts[c]
            var mj_dist = Float64(py=mj_c.dist)
            var mj_pos = mj_c.pos.flatten().tolist()
            var mj_geom = mj_c.geom.flatten().tolist()
            print(
                "  MJ  contact[",
                c,
                "]: geom=(",
                Int(py=mj_geom[0]),
                ",",
                Int(py=mj_geom[1]),
                ")",
                " pos=(",
                Float64(py=mj_pos[0]),
                ",",
                Float64(py=mj_pos[1]),
                ",",
                Float64(py=mj_pos[2]),
                ")",
                " dist=",
                mj_dist,
            )

    # Also compare qfrc_constraint (net constraint force in joint space)
    var mj_qfrc = mj_data.qfrc_constraint.flatten().tolist()
    print("  Our qfrc_constraint: N/A (not stored separately)")
    print("  MJ  qfrc_constraint:", end="")
    for i in range(NV):
        print(" ", Float64(py=mj_qfrc[i]), end="")
    print()

    # --- Compare constraint parameters (run mj_forward on FRESH data) ---
    var mj_data2 = mujoco.MjData(mj_model)
    for i in range(NQ):
        mj_data2.qpos[i] = qpos_init[i]
    for i in range(NV):
        mj_data2.qvel[i] = qvel_init[i]
    for i in range(ACTION_DIM):
        mj_data2.ctrl[i] = actions[i]
    # mj_step1 does: position kinematics, collision, constraint setup
    mujoco.mj_step1(mj_model, mj_data2)

    var mj_nefc = Int(py=mj_data2.nefc)
    print("  --- MuJoCo constraint params (nefc=", mj_nefc, ") ---")
    # Print solimp/solref used by MuJoCo
    var mj_solref = mj_model.opt.o_solref.flatten().tolist()
    var mj_solimp = mj_model.opt.o_solimp.flatten().tolist()
    print("    solref:", Float64(py=mj_solref[0]), Float64(py=mj_solref[1]))
    print(
        "    solimp:",
        Float64(py=mj_solimp[0]),
        Float64(py=mj_solimp[1]),
        Float64(py=mj_solimp[2]),
        Float64(py=mj_solimp[3]),
        Float64(py=mj_solimp[4]),
    )
    if mj_nefc > 0:
        var mj_efc_b = mj_data2.efc_b.flatten().tolist()
        var mj_efc_D = mj_data2.efc_D.flatten().tolist()
        var mj_efc_R = mj_data2.efc_R.flatten().tolist()
        var mj_efc_aref = mj_data2.efc_aref.flatten().tolist()
        var mj_efc_type = mj_data2.efc_type.flatten().tolist()
        # Also print KBIP
        var mj_efc_KBIP = mj_data2.efc_KBIP.flatten().tolist()
        for r in range(mj_nefc):
            if r < 15:  # Limit output
                var kbip_off = r * 4
                print(
                    "    row",
                    r,
                    " type=",
                    Int(py=mj_efc_type[r]),
                    " D=",
                    Float64(py=mj_efc_D[r]),
                    " R=",
                    Float64(py=mj_efc_R[r]),
                    " aref=",
                    Float64(py=mj_efc_aref[r]),
                    " KBIP=[",
                    Float64(py=mj_efc_KBIP[kbip_off]),
                    Float64(py=mj_efc_KBIP[kbip_off + 1]),
                    Float64(py=mj_efc_KBIP[kbip_off + 2]),
                    Float64(py=mj_efc_KBIP[kbip_off + 3]),
                    "]",
                )
    return all_pass


# =============================================================================
# Test cases — all involve ground contact
# =============================================================================


fn test_ground_contact() raises -> Bool:
    """Robot low enough to have ground contact (feet touching)."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.45  # rootz — pushes robot down
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    return compare_step("Ground contact (low rootz)", qpos, qvel, actions)


fn test_ground_contact_with_action() raises -> Bool:
    """Robot on ground with actions — full constraint solver test."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.45  # rootz — pushes robot down
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.8  # bthigh
    actions[1] = -0.5  # bshin
    actions[2] = 0.3  # bfoot
    actions[3] = 0.8  # fthigh
    actions[4] = -0.5  # fshin
    actions[5] = 0.3  # ffoot
    return compare_step("Ground contact with action", qpos, qvel, actions)


# Multi-step tests — call compare_step multiple times from the same initial
# conditions. Each call re-runs from scratch to N steps and prints final
# error. Calling in sequence shows how error grows with step count.
# (Avoids creating a second Model+Data on the same stack frame.)


fn test_multi_step_accumulation() raises -> Bool:
    """Error growth: run 1,5,10,50 steps from the same start state."""
    print("--- Test: Multi-step error accumulation ---")
    print("  Same initial conditions, increasing number of steps")
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.45
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.8
    actions[1] = -0.5
    actions[2] = 0.3
    actions[3] = 0.8
    actions[4] = -0.5
    actions[5] = 0.3
    # Each call re-runs from the same qpos/qvel, increasing num_steps
    var p1 = compare_step("  N=1 ", qpos, qvel, actions, 1)
    var p5 = compare_step("  N=5 ", qpos, qvel, actions, 5,
        MULTI_QPOS_ABS_TOL, MULTI_QPOS_REL_TOL, MULTI_QVEL_ABS_TOL, MULTI_QVEL_REL_TOL)
    var p10 = compare_step("  N=10", qpos, qvel, actions, 10,
        MULTI_QPOS_ABS_TOL, MULTI_QPOS_REL_TOL, MULTI_QVEL_ABS_TOL, MULTI_QVEL_REL_TOL)
    var p50 = compare_step("  N=50", qpos, qvel, actions, 50,
        MULTI_QPOS_ABS_TOL, MULTI_QPOS_REL_TOL, MULTI_QVEL_ABS_TOL, MULTI_QVEL_REL_TOL)
    return p1 and p5 and p10 and p50


fn test_fast_downward_impact() raises -> Bool:
    """Robot falling fast (qvel[1]=-3 m/s) — high-velocity impact at 1 step."""
    print("--- Test: Fast downward impact (v_z=-3 m/s) ---")
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.35  # rootz — feet very close to ground
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[1] = -3.0   # 3 m/s downward
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    var p1 = compare_step("  v_z=-3 N=1 ", qpos, qvel, actions, 1)
    var p5 = compare_step("  v_z=-3 N=5 ", qpos, qvel, actions, 5,
        MULTI_QPOS_ABS_TOL, MULTI_QPOS_REL_TOL, MULTI_QVEL_ABS_TOL, MULTI_QVEL_REL_TOL)
    var p10 = compare_step("  v_z=-3 N=10", qpos, qvel, actions, 10,
        MULTI_QPOS_ABS_TOL, MULTI_QPOS_REL_TOL, MULTI_QVEL_ABS_TOL, MULTI_QVEL_REL_TOL)
    return p1 and p5 and p10


fn test_very_fast_impact() raises -> Bool:
    """Very high velocity impact (qvel[1]=-6 m/s) — worst-case penetration."""
    print("--- Test: Very fast impact (v_z=-6 m/s) ---")
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.3   # rootz — feet near ground
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[1] = -6.0   # 6 m/s downward
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    var p1 = compare_step("  v_z=-6 N=1 ", qpos, qvel, actions, 1)
    var p5 = compare_step("  v_z=-6 N=5 ", qpos, qvel, actions, 5,
        MULTI_QPOS_ABS_TOL, MULTI_QPOS_REL_TOL, MULTI_QVEL_ABS_TOL, MULTI_QVEL_REL_TOL)
    var p10 = compare_step("  v_z=-6 N=10", qpos, qvel, actions, 10,
        MULTI_QPOS_ABS_TOL, MULTI_QPOS_REL_TOL, MULTI_QVEL_ABS_TOL, MULTI_QVEL_REL_TOL)
    return p1 and p5 and p10


fn test_running_gait_impact() raises -> Bool:
    """Running gait velocities — forward motion + downward foot strike."""
    print("--- Test: Running gait impact (v_forward=3, v_z=-2) ---")
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.35
    qpos[2] = -0.2    # rooty: slight forward lean
    qpos[3] = 0.4     # bthigh
    qpos[4] = -0.7    # bshin
    qpos[6] = -0.3    # fthigh
    qpos[7] = 0.6     # fshin
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 3.0     # rootx: 3 m/s forward
    qvel[1] = -2.0    # rootz: 2 m/s downward
    qvel[3] = 5.0     # bthigh angular velocity
    qvel[4] = -8.0    # bshin
    qvel[6] = -4.0    # fthigh
    qvel[7] = 7.0     # fshin
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.8
    actions[1] = -0.8
    actions[3] = 0.8
    actions[4] = -0.8
    var p1 = compare_step("  running N=1 ", qpos, qvel, actions, 1)
    var p5 = compare_step("  running N=5 ", qpos, qvel, actions, 5,
        MULTI_QPOS_ABS_TOL, MULTI_QPOS_REL_TOL, MULTI_QVEL_ABS_TOL, MULTI_QVEL_REL_TOL)
    var p20 = compare_step("  running N=20", qpos, qvel, actions, 20,
        MULTI_QPOS_ABS_TOL, MULTI_QPOS_REL_TOL, MULTI_QVEL_ABS_TOL, MULTI_QVEL_REL_TOL)
    return p1 and p5 and p20


fn test_fthigh_at_limit_impact() raises -> Bool:
    """fthigh at upper limit (0.7) + fast downward impact.

    This reproduces the policy rollout bug: fthigh pinned at range_max=0.7
    while the foot strikes the ground at high velocity.  Previously,
    enforce_limits zeroed qvel_fthigh, which destroyed the B*v_n damping
    term in the constraint bias and corrupted contact force computation
    through M^{-1} coupling.
    """
    print("--- Test: fthigh at limit (0.7) + fast downward impact ---")
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.35  # rootz — feet close to ground
    qpos[6] = 0.7    # fthigh at upper range_max limit
    qpos[7] = 0.4    # fshin bent
    qpos[8] = -0.2   # ffoot
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[1] = -3.0   # 3 m/s downward (foot strike)
    qvel[6] = 2.0    # fthigh velocity pushing toward limit
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[3] = 1.0  # fthigh: max torque toward limit
    actions[4] = -0.5
    actions[5] = 0.3
    var p1 = compare_step("  fthigh_lim N=1 ", qpos, qvel, actions, 1)
    var p5 = compare_step("  fthigh_lim N=5 ", qpos, qvel, actions, 5,
        MULTI_QPOS_ABS_TOL, MULTI_QPOS_REL_TOL, MULTI_QVEL_ABS_TOL, MULTI_QVEL_REL_TOL)
    var p10 = compare_step("  fthigh_lim N=10", qpos, qvel, actions, 10,
        MULTI_QPOS_ABS_TOL, MULTI_QPOS_REL_TOL, MULTI_QVEL_ABS_TOL, MULTI_QVEL_REL_TOL)
    return p1 and p5 and p10


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    print("=" * 70)
    print("Full Step with Contacts: Mojo Engine vs MuJoCo Reference")
    print("=" * 70)
    print("Model: HalfCheetah (NQ=9, NV=9)")
    print("Integrator: Euler (MuJoCo default)")
    print("Solver: Newton")
    print("Cone: pyramidal (both engines)")
    print("Precision: float64")
    print(
        "Single-step tolerances: qpos/qvel abs=",
        QPOS_ABS_TOL,
        " rel=",
        QPOS_REL_TOL,
    )
    print(
        "Multi-step  tolerances: qpos/qvel abs=",
        MULTI_QPOS_ABS_TOL,
        " rel=",
        MULTI_QPOS_REL_TOL,
    )
    print()

    var num_pass = 0
    var num_fail = 0

    # --- Original single-step tests ---
    print("### Single-step baseline tests ###")
    print()

    if test_ground_contact():
        num_pass += 1
    else:
        num_fail += 1
    print()

    if test_ground_contact_with_action():
        num_pass += 1
    else:
        num_fail += 1
    print()

    # --- Multi-step accumulation tests ---
    print("### Multi-step error accumulation (same start, more steps) ###")
    print()

    if test_multi_step_accumulation():
        num_pass += 1
    else:
        num_fail += 1
    print()

    # --- High-velocity impact tests ---
    print("### High-velocity impact tests ###")
    print()

    if test_fast_downward_impact():
        num_pass += 1
    else:
        num_fail += 1
    print()

    if test_very_fast_impact():
        num_pass += 1
    else:
        num_fail += 1
    print()

    if test_running_gait_impact():
        num_pass += 1
    else:
        num_fail += 1
    print()

    # --- Joint-limit + contact coupling test ---
    print("### Joint limit + contact coupling test ###")
    print()

    if test_fthigh_at_limit_impact():
        num_pass += 1
    else:
        num_fail += 1
    print()

    print("=" * 70)
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
        print("SOME TESTS FAILED — check error magnitudes above for solver quality")
    print("=" * 70)

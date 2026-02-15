"""Test Full Physics Step with Ground Contacts: Mojo Engine vs MuJoCo.

Separated from test_full_step_vs_mujoco.mojo because having too many
test functions in one file causes Mojo compiler stack overflow with
the heavily-generic constraint solver code.

Tests scenarios where the robot makes ground contact, exercising the
full constraint solver pipeline (contact detection + Jacobians + solver).

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_full_step_contact_vs_mujoco.mojo
"""

from python import Python, PythonObject
from math import abs
from collections import InlineArray

from physics3d.types import Model, Data, _max_one
from physics3d.integrator.euler_integrator import EulerIntegrator
from physics3d.solver.newton_solver import NewtonSolver
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
comptime NBODY = HalfCheetahModel.NBODY  # 7
comptime NJOINT = HalfCheetahModel.NJOINT  # 9
comptime NGEOM = HalfCheetahModel.NGEOM  # 9
comptime MAX_CONTACTS = HalfCheetahParams[DTYPE].MAX_CONTACTS  # 20
comptime ACTION_DIM = HalfCheetahParams[DTYPE].ACTION_DIM  # 6

# Tolerances — relaxed for contact scenarios since solver convergence
# paths will differ between our engine and MuJoCo
comptime QPOS_ABS_TOL: Float64 = 5e-3
comptime QPOS_REL_TOL: Float64 = 5e-2
comptime QVEL_ABS_TOL: Float64 = 5e-2
comptime QVEL_REL_TOL: Float64 = 5e-2


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
    var model = Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM](
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

    # Use elliptic cone — avoids pyramidal redundancy (40 edges for 9 DoFs)
    # Set mj_model.opt.cone = 1 on the MuJoCo side to match
    model.cone_type = 1

    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()
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

        HalfCheetahActuators.apply_actions[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS
        ](data, action_list)

        EulerIntegrator[SOLVER=NewtonSolver].step[
            NGEOM=NGEOM
        ](model, data)

    # === MuJoCo reference ===
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")

    var xml_path = (
        "../Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml"
    )
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    # Match our elliptic cone setting
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

        var ok = abs_err < QPOS_ABS_TOL or rel_err < QPOS_REL_TOL
        if not ok:
            if qpos_fails < 5:
                print(
                    "  FAIL qpos[", i, "]",
                    " ours=", our_val,
                    " mj=", mj_val,
                    " abs=", abs_err,
                    " rel=", rel_err,
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

        var ok = abs_err < QVEL_ABS_TOL or rel_err < QVEL_REL_TOL
        if not ok:
            if qvel_fails < 5:
                print(
                    "  FAIL qvel[", i, "]",
                    " ours=", our_val,
                    " mj=", mj_val,
                    " abs=", abs_err,
                    " rel=", rel_err,
                )
            qvel_fails += 1
            qvel_pass = False

    var all_pass = qpos_pass and qvel_pass

    if all_pass:
        print(
            "  ALL OK  qpos_max_abs=", qpos_max_abs,
            " qpos_max_rel=", qpos_max_rel,
            " qvel_max_abs=", qvel_max_abs,
            " qvel_max_rel=", qvel_max_rel,
        )
    else:
        print(
            "  FAILED  qpos:", qpos_fails, "fails (max_abs=", qpos_max_abs,
            " max_rel=", qpos_max_rel, ")",
            " qvel:", qvel_fails, "fails (max_abs=", qvel_max_abs,
            " max_rel=", qvel_max_rel, ")",
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
                "  Our contact[", c, "]: body_a=", Int(data.contacts[c].body_a),
                " body_b=", Int(data.contacts[c].body_b),
                " pos=(", Float64(data.contacts[c].pos_x),
                ",", Float64(data.contacts[c].pos_y),
                ",", Float64(data.contacts[c].pos_z), ")",
                " dist=", Float64(data.contacts[c].dist),
                " force_n=", Float64(data.contacts[c].force_n),
            )

    if mj_ncon > 0:
        var mj_contacts = mj_data.contact
        for c in range(mj_ncon):
            var mj_c = mj_contacts[c]
            var mj_dist = Float64(py=mj_c.dist)
            var mj_pos = mj_c.pos.flatten().tolist()
            var mj_geom = mj_c.geom.flatten().tolist()
            print(
                "  MJ  contact[", c, "]: geom=(",
                Int(py=mj_geom[0]), ",", Int(py=mj_geom[1]), ")",
                " pos=(", Float64(py=mj_pos[0]),
                ",", Float64(py=mj_pos[1]),
                ",", Float64(py=mj_pos[2]), ")",
                " dist=", mj_dist,
            )

    # Also compare qfrc_constraint (net constraint force in joint space)
    var mj_qfrc = mj_data.qfrc_constraint.flatten().tolist()
    print("  Our qfrc_constraint: N/A (not stored separately)")
    print("  MJ  qfrc_constraint:", end="")
    for i in range(NV):
        print(" ", Float64(py=mj_qfrc[i]), end="")
    print()

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
    actions[0] = 0.8   # bthigh
    actions[1] = -0.5  # bshin
    actions[2] = 0.3   # bfoot
    actions[3] = 0.8   # fthigh
    actions[4] = -0.5  # fshin
    actions[5] = 0.3   # ffoot
    return compare_step("Ground contact with action", qpos, qvel, actions)


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    print("=" * 60)
    print("Full Step with Contacts: Mojo Engine vs MuJoCo Reference")
    print("=" * 60)
    print("Model: HalfCheetah (NQ=9, NV=9)")
    print("Integrator: Euler (MuJoCo default)")
    print("Solver: Newton (MuJoCo default)")
    print("Cone: elliptic (both engines)")
    print("Precision: float64")
    print("Tolerances: qpos abs=", QPOS_ABS_TOL, " rel=", QPOS_REL_TOL)
    print("            qvel abs=", QVEL_ABS_TOL, " rel=", QVEL_REL_TOL)
    print()

    var num_pass = 0
    var num_fail = 0

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

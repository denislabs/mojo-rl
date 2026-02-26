"""Test Solver Forces against MuJoCo reference for Hopper.

Compares our NewtonSolver output (qacc, qfrc_constraint) against MuJoCo's
after mj_forward() for the Hopper model at configurations with ground contacts.

Hopper uses ELLIPTIC cone (default), condim=1 (frictionless).

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_hopper_solver_forces_vs_mujoco.mojo
"""

from testing import assert_true, TestSuite
from python import Python, PythonObject
from math import abs, sqrt
from collections import InlineArray

from physics3d.types import Model, Data, _max_one, ConeType
from physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from physics3d.dynamics.jacobian import compute_cdof, compute_composite_inertia
from physics3d.dynamics.bias_forces import compute_bias_forces_rne
from physics3d.dynamics.mass_matrix import (
    compute_mass_matrix_full,
    ldl_factor,
    ldl_solve,
    compute_M_inv_from_ldl,
)
from physics3d.collision.contact_detection import detect_contacts
from physics3d.constraints.constraint_builder import build_constraints, writeback_forces
from physics3d.constraints.constraint_data import (
    ConstraintData,
    CNSTR_NORMAL,
    CNSTR_FRICTION_T1,
    CNSTR_FRICTION_T2,
    CNSTR_LIMIT,
)
from physics3d.solver import NewtonSolver
from physics3d.solver.primal_common import (
    compute_total_cost_with_D,
    primal_D,
)
from physics3d.joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
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
comptime MAX_EQUALITY = 0

comptime V_SIZE = _max_one[NV]()
comptime M_SIZE = _max_one[NV * NV]()
comptime CDOF_SIZE = _max_one[NV * 6]()
comptime CRB_SIZE = _max_one[NBODY * 10]()
comptime MAX_ROWS = 11 * MAX_CONTACTS + 2 * NJOINT

# Tolerances
comptime QACC_ABS_TOL: Float64 = 5e-2
comptime QACC_REL_TOL: Float64 = 2e-1
comptime QFRC_ABS_TOL: Float64 = 5e-2
comptime QFRC_REL_TOL: Float64 = 2e-1


# =============================================================================
# Helper: compare a vector
# =============================================================================


fn compare_vector(
    label: String,
    our_vals: InlineArray[Float64, NV],
    mj_vals: InlineArray[Float64, NV],
    abs_tol: Float64,
    rel_tol: Float64,
) raises:
    """Compare NV-length vectors, assert all pass."""
    var all_ok = True
    var max_abs: Float64 = 0.0
    var max_rel: Float64 = 0.0
    var fail_count = 0

    for i in range(NV):
        var abs_err = abs(our_vals[i] - mj_vals[i])
        var ref_mag = abs(mj_vals[i])
        var rel_err: Float64 = 0.0
        if ref_mag > 1e-10:
            rel_err = abs_err / ref_mag
        if abs_err > max_abs:
            max_abs = abs_err
        if rel_err > max_rel:
            max_rel = rel_err
        var ok = abs_err < abs_tol or rel_err < rel_tol
        if not ok:
            print(
                "    FAIL",
                label + "[" + String(i) + "]",
                " ours=",
                our_vals[i],
                " mj=",
                mj_vals[i],
                " abs=",
                abs_err,
                " rel=",
                rel_err,
            )
            fail_count += 1
            all_ok = False

    if all_ok:
        print(
            "  ", label, " ALL OK  max_abs=", max_abs, " max_rel=", max_rel,
        )
    else:
        print(
            "  ", label, " FAILED", fail_count, " elements  max_abs=",
            max_abs, " max_rel=", max_rel,
        )
    assert_true(all_ok, label + " mismatch")


# =============================================================================
# Comparison helper
# =============================================================================


fn compare_solver_forces(
    test_name: String,
    qpos_values: InlineArray[Float64, NQ],
    qvel_values: InlineArray[Float64, NV],
) raises:
    """Run full pipeline + solver in both engines, compare forces and qacc."""
    print("--- Test:", test_name, "---")

    # === Our engine ===
    var model = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, 0, HopperModel.CONE_TYPE
    ](
    )
    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()
    HopperModel.setup_model_and_data(model, data)

    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_values[i])
    for i in range(NV):
        data.qvel[i] = Scalar[DTYPE](qvel_values[i])

    # 1. FK + body velocities + cdof
    forward_kinematics(model, data)
    compute_body_velocities(model, data)

    var cdof = InlineArray[Scalar[DTYPE], CDOF_SIZE](uninitialized=True)
    compute_cdof[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CDOF_SIZE](
        model, data, cdof
    )

    # 2. Contact detection
    detect_contacts[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM](
        model, data
    )

    # 3. Composite inertia + Mass matrix
    var crb = InlineArray[Scalar[DTYPE], CRB_SIZE](uninitialized=True)
    for i in range(CRB_SIZE):
        crb[i] = Scalar[DTYPE](0)
    compute_composite_inertia(model, data, crb)

    var M = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    for i in range(M_SIZE):
        M[i] = Scalar[DTYPE](0)
    compute_mass_matrix_full(model, data, cdof, crb, M)

    # 4. Add armature to M diagonal
    var dt = Scalar[DTYPE](0.002)  # Hopper timestep
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var dof_adr = joint.dof_adr
        var arm = joint.armature
        if joint.jnt_type == JNT_FREE:
            for d in range(6):
                M[(dof_adr + d) * NV + (dof_adr + d)] += arm
        elif joint.jnt_type == JNT_BALL:
            for d in range(3):
                M[(dof_adr + d) * NV + (dof_adr + d)] += arm
        else:
            M[dof_adr * NV + dof_adr] += arm

    # 5. LDL factorize + M_inv
    var L = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    var D_ldl = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    ldl_factor[DTYPE, NV, M_SIZE, V_SIZE](M, L, D_ldl)

    var M_inv = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    for i in range(M_SIZE):
        M_inv[i] = Scalar[DTYPE](0)
    compute_M_inv_from_ldl[DTYPE, NV, M_SIZE, V_SIZE](L, D_ldl, M_inv)

    # 6. Bias forces
    var bias = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(V_SIZE):
        bias[i] = Scalar[DTYPE](0)
    compute_bias_forces_rne[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE, CDOF_SIZE
    ](model, data, cdof, bias)

    # 7. f_net = qfrc - bias
    var f_net = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        f_net[i] = data.qfrc[i] - bias[i]

    # 8. Apply passive forces (damping + stiffness + frictionloss)
    for j in range(model.num_joints):
        var joint_d = model.joints[j]
        var dof_adr_d = joint_d.dof_adr
        var damp_d = joint_d.damping
        if damp_d > Scalar[DTYPE](0):
            if joint_d.jnt_type == JNT_FREE:
                for d in range(6):
                    f_net[dof_adr_d + d] -= damp_d * data.qvel[dof_adr_d + d]
            elif joint_d.jnt_type == JNT_BALL:
                for d in range(3):
                    f_net[dof_adr_d + d] -= damp_d * data.qvel[dof_adr_d + d]
            else:
                f_net[dof_adr_d] -= damp_d * data.qvel[dof_adr_d]

    for j in range(model.num_joints):
        var joint = model.joints[j]
        var dof_adr = joint.dof_adr
        var qpos_adr = joint.qpos_adr
        var stiff = joint.stiffness
        var sref = joint.springref
        var floss = joint.frictionloss
        if stiff > Scalar[DTYPE](0):
            if joint.jnt_type == JNT_FREE:
                for d in range(6):
                    f_net[dof_adr + d] -= stiff * (
                        data.qpos[qpos_adr + d] - sref
                    )
            elif joint.jnt_type == JNT_BALL:
                for d in range(3):
                    f_net[dof_adr + d] -= stiff * (
                        data.qpos[qpos_adr + d] - sref
                    )
            else:
                f_net[dof_adr] -= stiff * (data.qpos[qpos_adr] - sref)
        if floss > Scalar[DTYPE](0):
            comptime VEL_THRESH: Scalar[DTYPE] = 1e-4
            if joint.jnt_type == JNT_FREE:
                for d in range(6):
                    var v = data.qvel[dof_adr + d]
                    if v > VEL_THRESH:
                        f_net[dof_adr + d] -= floss
                    elif v < -VEL_THRESH:
                        f_net[dof_adr + d] += floss
            elif joint.jnt_type == JNT_BALL:
                for d in range(3):
                    var v = data.qvel[dof_adr + d]
                    if v > VEL_THRESH:
                        f_net[dof_adr + d] -= floss
                    elif v < -VEL_THRESH:
                        f_net[dof_adr + d] += floss
            else:
                var v = data.qvel[dof_adr]
                if v > VEL_THRESH:
                    f_net[dof_adr] -= floss
                elif v < -VEL_THRESH:
                    f_net[dof_adr] += floss

    # 9. qacc = M^{-1} * f_net via LDL solve
    var qacc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        qacc[i] = Scalar[DTYPE](0)
    ldl_solve[DTYPE, NV, M_SIZE, V_SIZE](L, D_ldl, f_net, qacc)

    # 10. Build constraints
    var constraints = ConstraintData[DTYPE, MAX_ROWS, NV]()
    build_constraints[CONE_TYPE=HopperModel.CONE_TYPE](
        model, data, cdof, M_inv, qacc, dt, constraints
    )

    # 11. Fill M_hat and qfrc_smooth for primal solver
    for i in range(NV * NV):
        constraints.M_hat[i] = M[i]
    for i in range(NV):
        constraints.qfrc_smooth[i] = f_net[i]

    var our_ncon = data.num_contacts
    print(
        "  Our: contacts=", our_ncon,
        " rows=", constraints.num_rows,
        " (N:", constraints.num_normals, " F:", constraints.num_friction, " L:", constraints.num_limits, ")",
    )

    # Save qacc0
    var qacc0 = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        qacc0[i] = qacc[i]

    # 12. Solve constraints
    NewtonSolver.solve[CONE_TYPE=HopperModel.CONE_TYPE](
        model, data, M_inv, constraints, qacc, dt
    )

    # 13. Compute our qfrc_constraint = J^T * lambda
    var our_qfrc = InlineArray[Float64, NV](fill=0.0)
    for r in range(constraints.num_rows):
        var lam = Float64(constraints.rows[r].lambda_val)
        for i in range(NV):
            our_qfrc[i] += lam * Float64(constraints.J[r * NV + i])

    var our_qacc = InlineArray[Float64, NV](fill=0.0)
    for i in range(NV):
        our_qacc[i] = Float64(qacc[i])

    # === MuJoCo reference ===
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")

    var xml_path = (
        "../Gymnasium-main/gymnasium/envs/mujoco/assets/hopper.xml"
    )
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.cone = 1       # elliptic (matches HopperModel)
    mj_model.opt.solver = 2     # Newton
    mj_model.opt.integrator = 0 # Euler

    var mj_data = mujoco.MjData(mj_model)

    for i in range(NQ):
        mj_data.qpos[i] = qpos_values[i]
    for i in range(NV):
        mj_data.qvel[i] = qvel_values[i]

    mujoco.mj_forward(mj_model, mj_data)

    var mj_nefc = Int(py=mj_data.nefc)
    var mj_ncon = Int(py=mj_data.ncon)
    print("  MJ:  contacts=", mj_ncon, " rows=", mj_nefc)

    # Extract MuJoCo qacc and qfrc_constraint
    var mj_qacc_flat = mj_data.qacc.flatten().tolist()
    var mj_qfrc_flat = mj_data.qfrc_constraint.flatten().tolist()

    var mj_qacc = InlineArray[Float64, NV](fill=0.0)
    var mj_qfrc = InlineArray[Float64, NV](fill=0.0)
    for i in range(NV):
        mj_qacc[i] = Float64(py=mj_qacc_flat[i])
        mj_qfrc[i] = Float64(py=mj_qfrc_flat[i])

    # === Cost comparison ===
    comptime MR = _max_one[MAX_ROWS]()
    var D_vals_cmp = InlineArray[Scalar[DTYPE], MR](fill=Scalar[DTYPE](0))
    for r_cmp in range(constraints.num_rows):
        D_vals_cmp[r_cmp] = primal_D(
            constraints.rows[r_cmp].inv_K_imp,
            constraints.rows[r_cmp].K,
        )

    var our_cost = compute_total_cost_with_D[DTYPE, MAX_ROWS, NV, V_SIZE, MR](
        constraints, D_vals_cmp, qacc, qacc0, f_net, M,
    )
    var mj_qacc_typed = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        mj_qacc_typed[i] = Scalar[DTYPE](mj_qacc[i])
    var mj_cost = compute_total_cost_with_D[DTYPE, MAX_ROWS, NV, V_SIZE, MR](
        constraints, D_vals_cmp, mj_qacc_typed, qacc0, f_net, M,
    )

    print("  --- Cost comparison ---")
    print("    Our cost:    ", Float64(our_cost))
    print("    MuJoCo cost: ", Float64(mj_cost))

    print()
    print("  --- qfrc_constraint (NV) ---")
    compare_vector(
        "qfrc_constraint", our_qfrc, mj_qfrc, QFRC_ABS_TOL, QFRC_REL_TOL
    )

    print("  Our qfrc:", end="")
    for i in range(NV):
        print(" ", our_qfrc[i], end="")
    print()
    print("  MJ  qfrc:", end="")
    for i in range(NV):
        print(" ", mj_qfrc[i], end="")
    print()

    print()
    print("  --- qacc (NV) ---")
    compare_vector(
        "qacc", our_qacc, mj_qacc, QACC_ABS_TOL, QACC_REL_TOL
    )

    print("  Our qacc:", end="")
    for i in range(NV):
        print(" ", our_qacc[i], end="")
    print()
    print("  MJ  qacc:", end="")
    for i in range(NV):
        print(" ", mj_qacc[i], end="")
    print()

    # Per-row forces (informational)
    print()
    print("  --- Per-row forces ---")
    for r in range(constraints.num_rows):
        var ct = constraints.rows[r].constraint_type
        var ctype_str: String
        if ct == CNSTR_NORMAL:
            ctype_str = "N "
        elif ct == CNSTR_FRICTION_T1:
            ctype_str = "T1"
        elif ct == CNSTR_FRICTION_T2:
            ctype_str = "T2"
        elif ct == CNSTR_LIMIT:
            ctype_str = "L "
        else:
            ctype_str = "? "
        print(
            "    our[", r, "] type=", ctype_str,
            " lambda=", Float64(constraints.rows[r].lambda_val),
            " K=", Float64(constraints.rows[r].K),
            " bias=", Float64(constraints.rows[r].bias),
        )

    var mj_efc_force_flat = mj_data.efc_force.flatten().tolist()
    var mj_types_flat = mj_data.efc_type.flatten().tolist()
    var mj_efc_D_flat = mj_data.efc_D.flatten().tolist()
    var mj_efc_aref_flat = mj_data.efc_aref.flatten().tolist()
    var mj_efc_J_flat = mj_data.efc_J.flatten().tolist()
    for r in range(mj_nefc):
        var t = Int(py=mj_types_flat[r])
        var tstr: String
        if t == 7:
            tstr = "CE"  # contact elliptic
        elif t == 3:
            tstr = "LI"  # limit
        else:
            tstr = String(t)
        # Print J row
        var j_str = String("")
        for i in range(NV):
            j_str += " " + String(Float64(py=mj_efc_J_flat[r * NV + i]))
        print(
            "    mj [", r, "] type=", tstr,
            " force=", Float64(py=mj_efc_force_flat[r]),
            " D=", Float64(py=mj_efc_D_flat[r]),
            " aref=", Float64(py=mj_efc_aref_flat[r]),
            " J=[", j_str, "]",
        )
    print()
    print("  --- Our Jacobian rows for active constraints ---")
    for r in range(constraints.num_rows):
        var lam = Float64(constraints.rows[r].lambda_val)
        if abs(lam) < 1e-6 and Float64(constraints.rows[r].K) < 1e-5:
            continue
        var j_str = String("")
        for i in range(NV):
            j_str += " " + String(Float64(constraints.J[r * NV + i]))
        print("    our[", r, "] lambda=", lam, " J=[", j_str, "]")

    print()
    print("  ALL OK")


# =============================================================================
# Test cases
# =============================================================================


fn test_low_pose_static() raises:
    """Low pose — foot on ground, zero velocity."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.8  # rootz low
    var qvel = InlineArray[Float64, NV](fill=0.0)
    compare_solver_forces("Low pose static (rootz=-0.8)", qpos, qvel)


fn test_low_pose_moving() raises:
    """Low pose with velocity."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.8
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 2.0   # moving forward
    qvel[1] = -0.5  # moving down
    qvel[3] = -1.0  # thigh rotating
    compare_solver_forces("Low pose moving", qpos, qvel)


fn test_very_low_pose() raises:
    """Very low — deeper penetration."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -1.0  # very low
    var qvel = InlineArray[Float64, NV](fill=0.0)
    compare_solver_forces("Very low pose (rootz=-1.0)", qpos, qvel)


fn test_bent_joints() raises:
    """Bent joints — different contact geometry."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.8
    qpos[3] = -0.5   # thigh bent
    qpos[4] = 0.5    # leg extended
    qpos[5] = -0.3   # foot bent
    var qvel = InlineArray[Float64, NV](fill=0.0)
    compare_solver_forces("Bent joints", qpos, qvel)


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

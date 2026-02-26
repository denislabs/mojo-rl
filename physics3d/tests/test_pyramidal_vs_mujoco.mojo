"""Test Pyramidal Cone Solver Forces against MuJoCo reference.

Compares our NewtonSolver with ConeType.PYRAMIDAL against MuJoCo's
Newton solver with opt.cone=0 (pyramidal) for the HalfCheetah model
at configurations with ground contacts.

Pyramidal cone encodes friction as edge constraints:
  J_edge± = J_normal ± mu * J_tangent
with positivity constraint lambda >= 0 on each edge.

MuJoCo's Newton/CG solver treats pyramidal edges as simple inequality
constraints (same as limits), which is what our primal solver does too.

What we compare:
  1. qfrc_constraint (NV): J^T * lambda vs mj_data.qfrc_constraint
  2. qacc (NV): constrained acceleration vs mj_data.qacc
  3. Row counts: verify same number of constraint rows
  4. Per-row efc_force (informational, printed for analysis)
  5. Cost comparison at both solutions

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_pyramidal_vs_mujoco.mojo
"""

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
    CNSTR_PYRAMID_EDGE,
)
from physics3d.solver import NewtonSolver
from physics3d.solver.primal_common import (
    compute_total_cost_with_D,
    primal_D,
    compute_jar,
    constraint_update_with_D,
    PRIMAL_SATISFIED,
    PRIMAL_QUADRATIC,
    PRIMAL_CONE,
)
from physics3d.joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
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
comptime MAX_EQUALITY = 0

comptime V_SIZE = _max_one[NV]()
comptime M_SIZE = _max_one[NV * NV]()
comptime CDOF_SIZE = _max_one[NV * 6]()
comptime CRB_SIZE = _max_one[NBODY * 10]()
comptime MAX_ROWS = 11 * MAX_CONTACTS + 2 * NJOINT

# Tolerances — same as elliptic test initially
comptime QACC_ABS_TOL: Float64 = 5e-2
comptime QACC_REL_TOL: Float64 = 2e-1
comptime QFRC_ABS_TOL: Float64 = 5e-2
comptime QFRC_REL_TOL: Float64 = 2e-1


# =============================================================================
# Helper: compare vectors
# =============================================================================


fn compare_vector(
    label: String,
    our_vals: InlineArray[Float64, NV],
    mj_vals: InlineArray[Float64, NV],
    abs_tol: Float64,
    rel_tol: Float64,
) raises -> Bool:
    """Compare NV-length vectors, return True if all pass."""
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
    return all_ok


fn compare_scalar(
    label: String,
    our_val: Float64,
    mj_val: Float64,
    abs_tol: Float64,
    rel_tol: Float64,
) raises -> Bool:
    """Compare a single scalar value, return True if passes."""
    var abs_err = abs(our_val - mj_val)
    var ref_mag = abs(mj_val)
    var rel_err: Float64 = 0.0
    if ref_mag > 1e-10:
        rel_err = abs_err / ref_mag
    var ok = abs_err < abs_tol or rel_err < rel_tol
    if ok:
        print(
            "    OK  ", label, " ours=", our_val, " mj=", mj_val,
            " abs=", abs_err, " rel=", rel_err,
        )
    else:
        print(
            "    FAIL", label, " ours=", our_val, " mj=", mj_val,
            " abs=", abs_err, " rel=", rel_err,
        )
    return ok


# =============================================================================
# Comparison helper
# =============================================================================


fn compare_pyramidal_solver(
    test_name: String,
    qpos_values: InlineArray[Float64, NQ],
    qvel_values: InlineArray[Float64, NV],
) raises -> Bool:
    """Run full pipeline + solver with PYRAMIDAL cone in both engines, compare."""
    print("--- Test:", test_name, "---")

    # === Our engine (PYRAMIDAL) ===
    var model = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, 0, ConeType.PYRAMIDAL
    ](
    )
    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()
    HalfCheetahModel.setup_model_and_data(model, data)

    # Set test configuration
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

    # 4. Add armature only to M diagonal
    var dt = Scalar[DTYPE](0.01)
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

    # 10. Build constraints (PYRAMIDAL cone)
    var constraints = ConstraintData[DTYPE, MAX_ROWS, NV]()
    build_constraints[CONE_TYPE=ConeType.PYRAMIDAL](
        model, data, cdof, M_inv, qacc, dt, constraints
    )

    # 11. Fill M_hat and qfrc_smooth for primal solver
    for i in range(NV * NV):
        constraints.M_hat[i] = M[i]
    for i in range(NV):
        constraints.qfrc_smooth[i] = f_net[i]

    var our_ncon = data.num_contacts
    var our_nnorm = constraints.num_normals  # For pyramidal, this includes ALL edge rows
    var our_nfric = constraints.num_friction  # Should be 0 for pyramidal
    var our_nlim = constraints.num_limits
    print(
        "  Our: contacts=", our_ncon,
        " rows=", constraints.num_rows,
        " (edges:", our_nnorm, " F:", our_nfric, " L:", our_nlim, ")",
    )


    # Print row details
    for r in range(constraints.num_rows):
        var ct = constraints.rows[r].constraint_type
        var ctype_str: String
        if ct == CNSTR_PYRAMID_EDGE:
            var src_c = constraints.rows[r].source_contact_idx
            var src_dof = constraints.rows[r].source_dof
            var td = src_dof // 2
            var sign_idx = src_dof % 2
            var sign_str = "+" if sign_idx == 0 else "-"
            ctype_str = "PE(c" + String(src_c) + ",t" + String(td) + sign_str + ")"
        elif ct == CNSTR_LIMIT:
            ctype_str = "L "
        elif ct == CNSTR_NORMAL:
            ctype_str = "N "
        else:
            ctype_str = "? "
        print(
            "    row[", r, "] type=", ctype_str,
            " K=", Float64(constraints.rows[r].K),
            " bias=", Float64(constraints.rows[r].bias),
            " inv_K_imp=", Float64(constraints.rows[r].inv_K_imp),
        )

    # Save qacc0 before solver modifies it
    var qacc0 = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        qacc0[i] = qacc[i]

    # 12. Solve constraints (modifies qacc in-place)
    NewtonSolver.solve[CONE_TYPE=ConeType.PYRAMIDAL](
        model, data, M_inv, constraints, qacc, dt
    )

    # 13. Compute our qfrc_constraint = J^T * lambda
    var our_qfrc = InlineArray[Float64, NV](fill=0.0)
    for r in range(constraints.num_rows):
        var lam = Float64(constraints.rows[r].lambda_val)
        for i in range(NV):
            our_qfrc[i] += lam * Float64(constraints.J[r * NV + i])

    # Collect our qacc as Float64
    var our_qacc = InlineArray[Float64, NV](fill=0.0)
    for i in range(NV):
        our_qacc[i] = Float64(qacc[i])

    # === MuJoCo reference (PYRAMIDAL cone) ===
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")

    var xml_path = (
        "../Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml"
    )
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    # Match our solver: PYRAMIDAL cone, Newton solver, Euler integrator
    mj_model.opt.cone = 0       # pyramidal (mjCONE_PYRAMIDAL)
    mj_model.opt.solver = 2     # Newton
    mj_model.opt.integrator = 0 # Euler

    var mj_data = mujoco.MjData(mj_model)

    for i in range(NQ):
        mj_data.qpos[i] = qpos_values[i]
    for i in range(NV):
        mj_data.qvel[i] = qvel_values[i]

    # Run full step (includes solver)
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

    # Extract MuJoCo per-row forces and types
    var mj_efc_force_flat = mj_data.efc_force.flatten().tolist()
    var mj_types_flat = mj_data.efc_type.flatten().tolist()
    var mj_efc_D_flat = mj_data.efc_D.flatten().tolist()
    var mj_efc_R_flat = mj_data.efc_R.flatten().tolist()
    var mj_efc_b_flat = mj_data.efc_b.flatten().tolist()

    # Count MuJoCo row types
    var mj_pyramid_count = 0
    var mj_limit_count = 0
    for r in range(mj_nefc):
        var t = Int(py=mj_types_flat[r])
        if t == 6:  # mjCNSTR_CONTACT_PYRAMIDAL
            mj_pyramid_count += 1
        elif t == 3:  # mjCNSTR_LIMIT_JOINT
            mj_limit_count += 1
    print(
        "  MJ:  pyramid_rows=", mj_pyramid_count,
        " limit_rows=", mj_limit_count,
    )

    # Print MuJoCo row details
    for r in range(mj_nefc):
        var t = Int(py=mj_types_flat[r])
        var tstr: String
        if t == 6:
            tstr = "PY"  # pyramidal
        elif t == 3:
            tstr = "LI"  # limit
        else:
            tstr = String(t)
        print(
            "    mj [", r, "] type=", tstr,
            " force=", Float64(py=mj_efc_force_flat[r]),
            " D=", Float64(py=mj_efc_D_flat[r]),
            " R=", Float64(py=mj_efc_R_flat[r]),
            " b=", Float64(py=mj_efc_b_flat[r]),
        )

    # Print our D/R for comparison
    print("  --- D/R comparison ---")
    for r in range(constraints.num_rows):
        var our_D = Float64(primal_D(constraints.rows[r].inv_K_imp, constraints.rows[r].K))
        var our_R = Float64(Scalar[DTYPE](1.0) / constraints.rows[r].inv_K_imp - constraints.rows[r].K)
        print(
            "    our[", r, "]  D=", our_D,
            " R=", our_R,
            " K=", Float64(constraints.rows[r].K),
        )

    # === Cost comparison ===
    comptime MR = _max_one[MAX_ROWS]()

    var D_vals_cmp = InlineArray[Scalar[DTYPE], MR](fill=Scalar[DTYPE](0))
    for r_cmp in range(constraints.num_rows):
        D_vals_cmp[r_cmp] = primal_D(
            constraints.rows[r_cmp].inv_K_imp,
            constraints.rows[r_cmp].K,
        )

    # Cost at our solution
    var our_cost = compute_total_cost_with_D[DTYPE, MAX_ROWS, NV, V_SIZE, MR](
        constraints, D_vals_cmp, qacc, qacc0, f_net, M,
    )

    # Cost at MuJoCo's solution
    var mj_qacc_typed = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        mj_qacc_typed[i] = Scalar[DTYPE](mj_qacc[i])
    var mj_cost = compute_total_cost_with_D[DTYPE, MAX_ROWS, NV, V_SIZE, MR](
        constraints, D_vals_cmp, mj_qacc_typed, qacc0, f_net, M,
    )

    print("  --- Cost comparison ---")
    print("    Our cost:    ", Float64(our_cost))
    print("    MuJoCo cost: ", Float64(mj_cost))
    if our_cost < mj_cost:
        print("    OURS IS LOWER (or cost functions differ)")
    elif our_cost > mj_cost:
        print("    MUJOCO IS LOWER → our solver is suboptimal")
    else:
        print("    EQUAL costs")

    # === Comparisons ===
    var all_pass = True

    # 1. qfrc_constraint
    print()
    print("  --- Comparison 1: qfrc_constraint (NV) ---")
    if not compare_vector(
        "qfrc_constraint", our_qfrc, mj_qfrc, QFRC_ABS_TOL, QFRC_REL_TOL
    ):
        all_pass = False

    print("  Our qfrc:", end="")
    for i in range(NV):
        print(" ", our_qfrc[i], end="")
    print()
    print("  MJ  qfrc:", end="")
    for i in range(NV):
        print(" ", mj_qfrc[i], end="")
    print()

    # 2. qacc
    print()
    print("  --- Comparison 2: qacc (NV) ---")
    if not compare_vector(
        "qacc", our_qacc, mj_qacc, QACC_ABS_TOL, QACC_REL_TOL
    ):
        all_pass = False

    print("  Our qacc:", end="")
    for i in range(NV):
        print(" ", our_qacc[i], end="")
    print()
    print("  MJ  qacc:", end="")
    for i in range(NV):
        print(" ", mj_qacc[i], end="")
    print()

    # 3. Row count check
    print()
    print("  --- Row count check ---")
    # For pyramidal: each contact should have 2*(condim-1) = 4 edge rows (condim=3)
    # Plus limit rows
    var expected_pyramid_rows = mj_ncon * 4  # condim=3 → 4 edges per contact
    print("    Our edge rows:", our_nnorm, " expected:", expected_pyramid_rows)
    print("    Our limit rows:", our_nlim, " MJ limit rows:", mj_limit_count)
    if our_nnorm != mj_pyramid_count:
        print("    WARNING: pyramid row count mismatch! ours=", our_nnorm, " mj=", mj_pyramid_count)

    # Print per-row lambda values
    print()
    print("  --- Per-row forces ---")
    for r in range(constraints.num_rows):
        var ct = constraints.rows[r].constraint_type
        var lam = Float64(constraints.rows[r].lambda_val)
        if ct == CNSTR_PYRAMID_EDGE:
            print("    our edge[", r, "] lambda=", lam)
        elif ct == CNSTR_LIMIT:
            print("    our limit[", r, "] lambda=", lam)
        else:
            print("    our ?[", r, "] lambda=", lam)

    print()
    if all_pass:
        print("  ALL OK")
    else:
        print("  FAILED")
    return all_pass


# =============================================================================
# Test cases (same 4 configs as elliptic solver test)
# =============================================================================


fn test_low_pose_static() raises -> Bool:
    """Low pose (rootz=-0.3), zero velocity."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.3  # rootz low
    var qvel = InlineArray[Float64, NV](fill=0.0)
    return compare_pyramidal_solver("Low pose static (rootz=-0.3)", qpos, qvel)


fn test_low_pose_moving() raises -> Bool:
    """Low pose with velocity — tests friction damping."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.3  # rootz low
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 2.0   # moving forward
    qvel[1] = -0.5  # moving down
    qvel[3] = -1.0  # bthigh rotating
    return compare_pyramidal_solver("Low pose moving", qpos, qvel)


fn test_very_low_pose() raises -> Bool:
    """Very low pose (rootz=-0.45) — deeper penetration, larger forces."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.45  # rootz very low
    var qvel = InlineArray[Float64, NV](fill=0.0)
    return compare_pyramidal_solver("Very low pose (rootz=-0.45)", qpos, qvel)


fn test_bent_legs() raises -> Bool:
    """Bent legs — different contact geometry + joint limits active."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.3
    qpos[3] = -0.5   # bthigh bent
    qpos[4] = 0.8    # bshin extended
    qpos[6] = 0.5    # fthigh bent
    qpos[7] = -0.8   # fshin extended
    var qvel = InlineArray[Float64, NV](fill=0.0)
    return compare_pyramidal_solver("Bent legs", qpos, qvel)


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    print("=" * 60)
    print("Pyramidal Solver Forces: Mojo Engine vs MuJoCo")
    print("=" * 60)
    print("Model: HalfCheetah (NV=", NV, ")")
    print("MuJoCo: cone=pyramidal (0), solver=Newton")
    print("Our solver: NewtonSolver (cone=PYRAMIDAL)")
    print(
        "Tolerances: qacc abs=", QACC_ABS_TOL,
        " qfrc abs=", QFRC_ABS_TOL,
    )
    print()

    var num_pass = 0
    var num_fail = 0

    if test_low_pose_static():
        num_pass += 1
    else:
        num_fail += 1
    print()

    if test_low_pose_moving():
        num_pass += 1
    else:
        num_fail += 1
    print()

    if test_very_low_pose():
        num_pass += 1
    else:
        num_fail += 1
    print()

    if test_bent_legs():
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

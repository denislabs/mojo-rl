"""Test Solver Forces against MuJoCo reference.

Compares our NewtonSolver output (qacc, qfrc_constraint, per-row forces)
against MuJoCo's after mj_step() for the HalfCheetah model at configurations
with ground contacts.

This is "Test 5: Solver Forces" from TEST.md. All pre-solver stages pass
(FK, mass matrix, bias forces, qacc0, contacts, Jacobians, constraint params).
This test isolates the solver itself as a source of divergence.

What we compare (5 levels):
  1. qfrc_constraint (NV): J^T * lambda vs mj_data.qfrc_constraint
  2. qacc (NV): constrained acceleration vs mj_data.qacc
  3. Total normal force: sum of normal lambdas vs sum of mj efc_force[normals]
  4. Per-contact normal force (informational)
  5. Per-row efc_force (informational)

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_solver_forces_vs_mujoco.mojo
"""

from python import Python, PythonObject
from math import abs, sqrt
from collections import InlineArray
from testing import assert_true, TestSuite

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

# Tolerances
comptime QACC_ABS_TOL: Float64 = 5e-2
comptime QACC_REL_TOL: Float64 = 2e-1
comptime QFRC_ABS_TOL: Float64 = 5e-2
comptime QFRC_REL_TOL: Float64 = 2e-1
comptime TOTAL_FORCE_ABS_TOL: Float64 = 5e-2
comptime TOTAL_FORCE_REL_TOL: Float64 = 1e-1
comptime PER_CONTACT_ABS_TOL: Float64 = 1e-1
comptime PER_CONTACT_REL_TOL: Float64 = 3e-1


# =============================================================================
# Helper: compare a vector
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


fn compare_solver_forces(
    test_name: String,
    qpos_values: InlineArray[Float64, NQ],
    qvel_values: InlineArray[Float64, NV],
) raises:
    """Run full pipeline + solver in both engines, compare forces and qacc."""
    print("--- Test:", test_name, "---")

    # === Our engine ===
    var model = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, HalfCheetahModel.MAX_EQUALITY, HalfCheetahModel.CONE_TYPE, HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE
    ]()
    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
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

    # 4. Add armature only to M diagonal (MuJoCo solver uses M+arm, NOT M+arm+dt*damp)
    # Damping is applied post-solver during Euler integration (see MuJoCo engine_forward.c:944-973)
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
    # Matches euler_integrator.mojo lines 238-305
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

    # 9. qacc = M^{-1} * f_net via LDL solve (unconstrained acceleration)
    var qacc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        qacc[i] = Scalar[DTYPE](0)
    ldl_solve[DTYPE, NV, M_SIZE, V_SIZE](L, D_ldl, f_net, qacc)

    # Diagnostic: verify M*qacc = f_net (LDL sanity check)
    var residual_max: Scalar[DTYPE] = 0
    for i in range(NV):
        var Mq_i: Scalar[DTYPE] = 0
        for j in range(NV):
            Mq_i += M[i * NV + j] * qacc[j]
        var res = abs(Mq_i - f_net[i])
        if res > residual_max:
            residual_max = res

    # 10. Build constraints (passing qacc as unconstrained acceleration)
    var constraints = ConstraintData[DTYPE, MAX_ROWS, NV]()
    build_constraints[CONE_TYPE=HalfCheetahModel.CONE_TYPE](
        model, data, cdof, M_inv, qacc, dt, constraints
    )

    # 11. Fill M_hat and qfrc_smooth for primal solver
    for i in range(NV * NV):
        constraints.M_hat[i] = M[i]
    for i in range(NV):
        constraints.qfrc_smooth[i] = f_net[i]

    var our_ncon = data.num_contacts
    var our_nnorm = constraints.num_normals
    var our_nfric = constraints.num_friction
    var our_nlim = constraints.num_limits
    print(
        "  Our: contacts=", our_ncon,
        " rows=", constraints.num_rows,
        " (N:", our_nnorm, " F:", our_nfric, " L:", our_nlim, ")",
    )

    # Save qacc0 before solver modifies it
    var qacc0 = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        qacc0[i] = qacc[i]

    # 12. Solve constraints (modifies qacc in-place)
    NewtonSolver.solve[CONE_TYPE=HalfCheetahModel.CONE_TYPE](
        model, data, M_inv, constraints, qacc, dt
    )

    # 13. Write forces back to data.contacts (required for pyramidal where
    #     per-contact force = sum of edge lambdas, not a single normal row)
    writeback_forces(constraints, data)

    # Compute our qfrc_constraint = J^T * lambda
    var our_qfrc = InlineArray[Float64, NV](fill=0.0)
    for r in range(constraints.num_rows):
        var lam = Float64(constraints.rows[r].lambda_val)
        for i in range(NV):
            our_qfrc[i] += lam * Float64(constraints.J[r * NV + i])

    # Collect our qacc as Float64
    var our_qacc = InlineArray[Float64, NV](fill=0.0)
    for i in range(NV):
        our_qacc[i] = Float64(qacc[i])

    # Collect per-contact normal forces from data.contacts (post-writeback)
    # For pyramidal: writeback aggregates edge lambdas into force_n.
    # Loop over actual contact count (not row count) to avoid overflow.
    var our_normal_forces = InlineArray[Float64, MAX_CONTACTS](fill=0.0)
    var our_total_normal: Float64 = 0.0
    for c in range(our_ncon):
        our_normal_forces[c] = Float64(data.contacts[c].force_n)
        our_total_normal += our_normal_forces[c]

    # === MuJoCo reference via Python ===
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")

    var xml_path = (
        "../Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml"
    )
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    # Match our solver: pyramidal cone, Newton solver, Euler integrator
    mj_model.opt.cone = 0       # pyramidal (matches HalfCheetahModel)
    mj_model.opt.solver = 2     # Newton
    mj_model.opt.integrator = 0 # Euler (match our M_hat = M + arm + dt*damp)

    var mj_data = mujoco.MjData(mj_model)

    for i in range(NQ):
        mj_data.qpos[i] = qpos_values[i]
    for i in range(NV):
        mj_data.qvel[i] = qvel_values[i]

    # Use mj_forward to populate all quantities (includes solver)
    mujoco.mj_forward(mj_model, mj_data)

    # Get dense M via mj_fullM
    var mj_M_dense = np.zeros(NV * NV).reshape(NV, NV)
    mujoco.mj_fullM(mj_model, mj_M_dense, mj_data.qM)

    # Compare M diagonals (M + armature, no damping — matches MuJoCo solver M)
    print("  M diagonal comparison (M+armature vs mj_fullM):")
    var max_M_diff: Float64 = 0
    for i in range(NV):
        var mj_Mii = Float64(py=mj_M_dense[i][i])
        var our_Mii = Float64(M[i * NV + i])
        var diff = abs(mj_Mii - our_Mii)
        if diff > max_M_diff:
            max_M_diff = diff
        if diff > 0.001:
            print("    M[", i, ",", i, "] ours=", our_Mii, " mj=", mj_Mii, " diff=", diff)
    if max_M_diff < 0.001:
        print("    ALL M diag MATCH (max_diff=", max_M_diff, ")")

    # Compare full M off-diagonals
    print("  M off-diagonal comparison:")
    var max_offdiag_diff: Float64 = 0
    for i in range(NV):
        for j in range(NV):
            if i == j:
                continue
            var mj_Mij = Float64(py=mj_M_dense[i][j])
            var our_Mij = Float64(M[i * NV + j])
            var diff_ij = abs(mj_Mij - our_Mij)
            if diff_ij > max_offdiag_diff:
                max_offdiag_diff = diff_ij
            if diff_ij > 0.001:
                print("    M[", i, ",", j, "] ours=", our_Mij, " mj=", mj_Mij, " diff=", diff_ij)
    if max_offdiag_diff < 0.001:
        print("    ALL off-diag MATCH (max_diff=", max_offdiag_diff, ")")

    print("  LDL solve residual max |M*qacc - f_net| =", Float64(residual_max))

    # Compare qfrc_bias and qfrc_passive
    var mj_bias_flat = mj_data.qfrc_bias.flatten().tolist()
    var mj_passive_flat = mj_data.qfrc_passive.flatten().tolist()
    var mj_smooth_flat = mj_data.qfrc_smooth.flatten().tolist()
    print("  qfrc comparison (bias = C*v + gravity):")
    for i in range(NV):
        var mj_bias = Float64(py=mj_bias_flat[i])
        var our_bias = Float64(bias[i])
        var diff_b = abs(mj_bias - our_bias)
        if diff_b > 0.01:
            print("    bias[", i, "] ours=", our_bias, " mj=", mj_bias, " diff=", diff_b)
    print("  MJ qfrc_smooth:", end="")
    for i in range(NV):
        print(" ", Float64(py=mj_smooth_flat[i]), end="")
    print()
    print("  Our f_net:", end="")
    for i in range(NV):
        print(" ", Float64(f_net[i]), end="")
    print()

    # Compute MuJoCo's qacc_smooth = Minv * qfrc_smooth for comparison
    # Our qacc_smooth = Minv * f_net
    var mj_qacc_smooth = np.zeros(NV)
    var mj_qfrc_smooth_arr = np.array(mj_data.qfrc_smooth.flatten())
    # mj_solveM(model, data, x, y) — x = M^{-1} * y using qLD factorization
    # The reshaping to (1, NV) is required by the API: x and y must be [m, n] arrays
    var mj_qacc_smooth_2d = mj_qacc_smooth.reshape(1, NV)
    var mj_qfrc_smooth_2d = mj_qfrc_smooth_arr.reshape(1, NV)
    mujoco.mj_solveM(mj_model, mj_data, mj_qacc_smooth_2d, mj_qfrc_smooth_2d)
    print("  MJ qacc_smooth:", end="")
    var mj_qacc_smooth_flat = mj_qacc_smooth_2d.flatten().tolist()
    for i in range(NV):
        print(" ", Float64(py=mj_qacc_smooth_flat[i]), end="")
    print()
    print("  Our qacc0:", end="")
    for i in range(NV):
        print(" ", Float64(qacc0[i]), end="")
    print()

    # Numpy verification: compute M^{-1} * f_net using numpy
    var np_M = np.zeros(NV * NV).reshape(NV, NV)
    for i in range(NV):
        for j in range(NV):
            np_M[i][j] = Float64(M[i * NV + j])
    var np_f = np.zeros(NV)
    for i in range(NV):
        np_f[i] = Float64(f_net[i])
    var np_qacc = np.linalg.solve(np_M, np_f)
    var cond = np.linalg.cond(np_M)
    print("  M condition number:", Float64(py=cond))
    var D_vals = np.linalg.eigvalsh(np_M)
    print("  M eigenvalues:", end="")
    var D_list = D_vals.tolist()
    for i in range(NV):
        print(" ", Float64(py=D_list[i]), end="")
    print()
    print("  Numpy qacc0:", end="")
    var np_qacc_flat = np_qacc.flatten().tolist()
    for i in range(NV):
        print(" ", Float64(py=np_qacc_flat[i]), end="")
    print()

    # Verify M*our_qacc0 = f_net using numpy
    var np_our_qacc = np.zeros(NV)
    for i in range(NV):
        np_our_qacc[i] = Float64(qacc0[i])
    var np_Mq = np_M.__matmul__(np_our_qacc)
    var np_residual = np_Mq.__sub__(np_f)
    print("  Numpy M*our_qacc - f_net:", end="")
    var res_list = np_residual.tolist()
    for i in range(NV):
        print(" ", Float64(py=res_list[i]), end="")
    print()

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

    # MuJoCo pyramidal: type 6 for all contact rows (edge constraints)
    # Each contact has 2*(condim-1) = 4 edge rows for condim=3
    var mj_contact_start = -1
    for r in range(mj_nefc):
        var t = Int(py=mj_types_flat[r])
        if t == 6:  # mjCNSTR_CONTACT_PYRAMIDAL
            mj_contact_start = r
            break

    # Also count limit rows
    var mj_limit_count = 0
    for r in range(mj_nefc):
        var t = Int(py=mj_types_flat[r])
        if t == 3:  # mjCNSTR_LIMIT_JOINT
            mj_limit_count += 1

    var mj_normal_forces = InlineArray[Float64, MAX_CONTACTS](fill=0.0)
    mj_total_normal = 0.0
    if mj_contact_start >= 0:
        for c in range(mj_ncon):
            var mj_r = mj_contact_start + c * 4  # first edge row for contact c
            # Pyramidal: each contact has 4 edge rows, normal force = sum of all edge lambdas
            for edge_idx in range(4):
                if mj_r + edge_idx < mj_nefc:
                    mj_normal_forces[c] += Float64(py=mj_efc_force_flat[mj_r + edge_idx])
            mj_total_normal += mj_normal_forces[c]
    print(
        "  MJ:  contact_rows (pyramidal)=", mj_ncon * 4,
        " limit_rows=", mj_limit_count,
    )

    # === Cost comparison: evaluate our cost at both solutions ===
    # This tells us whether the cost functions differ or the solver is suboptimal
    comptime MR = _max_one[MAX_ROWS]()

    # Compute D values (same as solver does)
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
        print("    OURS IS LOWER → cost functions differ or MuJoCo not fully converged")
    elif our_cost > mj_cost:
        print("    MUJOCO IS LOWER → our solver is suboptimal")
    else:
        print("    EQUAL costs")

    # Also compare D values with MuJoCo's efc_D
    var mj_efc_D = mj_data.efc_D.flatten().tolist()
    var mj_efc_R = mj_data.efc_R.flatten().tolist()
    print("  --- D value comparison ---")
    var max_D_diff: Float64 = 0
    # MuJoCo interleaves [limit... | contact_n, contact_t1, contact_t2, ...]
    # Our layout: [normals... | frictions... | limits...]
    # Map our row → MuJoCo row for comparison
    var friction_start_cmp = constraints.num_normals
    for n_cmp in range(constraints.num_normals):
        # MuJoCo: contacts start at mj_contact_start, each has 3 rows [n,t1,t2]
        if mj_contact_start >= 0:
            var mj_n_row = mj_contact_start + n_cmp * 3
            if mj_n_row < mj_nefc:
                var mj_D_n = Float64(py=mj_efc_D[mj_n_row])
                var our_D_n = Float64(D_vals_cmp[n_cmp])
                var d_diff = abs(mj_D_n - our_D_n)
                if d_diff > max_D_diff:
                    max_D_diff = d_diff
                if d_diff > 0.01:
                    print("    D[normal", n_cmp, "] ours=", our_D_n, " mj=", mj_D_n, " diff=", d_diff)
    if max_D_diff < 0.01:
        print("    ALL D values match (max_diff=", max_D_diff, ")")
    else:
        print("    D values DIFFER (max_diff=", max_D_diff, ")")

    # Also show MuJoCo efc_D and efc_R for reference
    print("  MuJoCo efc_D:", end="")
    for r_cmp in range(mj_nefc):
        print(" ", Float64(py=mj_efc_D[r_cmp]), end="")
    print()
    print("  Our D_vals:", end="")
    for r_cmp in range(constraints.num_rows):
        print(" ", Float64(D_vals_cmp[r_cmp]), end="")
    print()

    # === Comparisons ===
    var all_pass = True

    # 1. qfrc_constraint (NV) — most robust comparison
    print()
    print("  --- Comparison 1: qfrc_constraint (NV) ---")
    if not compare_vector(
        "qfrc_constraint", our_qfrc, mj_qfrc, QFRC_ABS_TOL, QFRC_REL_TOL
    ):
        all_pass = False

    # Print values
    print("  Our qfrc:", end="")
    for i in range(NV):
        print(" ", our_qfrc[i], end="")
    print()
    print("  MJ  qfrc:", end="")
    for i in range(NV):
        print(" ", mj_qfrc[i], end="")
    print()

    # 2. qacc (NV) — constrained acceleration
    print()
    print("  --- Comparison 2: qacc (NV) ---")
    if not compare_vector(
        "qacc", our_qacc, mj_qacc, QACC_ABS_TOL, QACC_REL_TOL
    ):
        all_pass = False

    # Print values
    print("  Our qacc:", end="")
    for i in range(NV):
        print(" ", our_qacc[i], end="")
    print()
    print("  MJ  qacc:", end="")
    for i in range(NV):
        print(" ", mj_qacc[i], end="")
    print()

    # 3. Total normal force
    print()
    print("  --- Comparison 3: Total normal force ---")
    if not compare_scalar(
        "total_normal", our_total_normal, mj_total_normal,
        TOTAL_FORCE_ABS_TOL, TOTAL_FORCE_REL_TOL,
    ):
        all_pass = False

    # 4. Per-contact normal force (informational)
    print()
    print("  --- Comparison 4: Per-contact normal forces (informational) ---")
    var min_ncon = our_ncon if our_ncon < mj_ncon else mj_ncon
    for c in range(min_ncon):
        var abs_err = abs(our_normal_forces[c] - mj_normal_forces[c])
        var ref_mag = abs(mj_normal_forces[c])
        var rel_err: Float64 = 0.0
        if ref_mag > 1e-10:
            rel_err = abs_err / ref_mag
        var ok = abs_err < PER_CONTACT_ABS_TOL or rel_err < PER_CONTACT_REL_TOL
        var status = "OK  " if ok else "FAIL"
        print(
            "    [", status, "] contact", c,
            " ours=", our_normal_forces[c],
            " mj=", mj_normal_forces[c],
            " abs=", abs_err, " rel=", rel_err,
        )

    # 5. Per-row efc_force (informational)
    print()
    print("  --- Comparison 5: Per-row forces (informational) ---")
    # Print our rows
    for r in range(constraints.num_rows):
        var ctype_str: String
        var ct = constraints.rows[r].constraint_type
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

    # Print MuJoCo rows
    for r in range(mj_nefc):
        var t = Int(py=mj_types_flat[r])
        var tstr: String
        if t == 7:
            tstr = "CP"  # contact pyramidal
        elif t == 3:
            tstr = "LI"  # limit
        else:
            tstr = String(t)
        print(
            "    mj [", r, "] type=", tstr,
            " force=", Float64(py=mj_efc_force_flat[r]),
        )

    print()
    if all_pass:
        print("  ALL OK")
    else:
        print("  FAILED")

    assert_true(all_pass, "Solver forces mismatch for: " + test_name)


# =============================================================================
# Test cases (same 4 configs as constraint_params test)
# =============================================================================


fn test_low_pose_static() raises:
    """Low pose (rootz=-0.3), zero velocity."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.3  # rootz low
    var qvel = InlineArray[Float64, NV](fill=0.0)
    compare_solver_forces("Low pose static (rootz=-0.3)", qpos, qvel)


fn test_low_pose_moving() raises:
    """Low pose with velocity — tests friction damping."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.3  # rootz low
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 2.0   # moving forward
    qvel[1] = -0.5  # moving down
    qvel[3] = -1.0  # bthigh rotating
    compare_solver_forces("Low pose moving", qpos, qvel)


fn test_very_low_pose() raises:
    """Very low pose (rootz=-0.45) — deeper penetration, larger forces."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.45  # rootz very low
    var qvel = InlineArray[Float64, NV](fill=0.0)
    compare_solver_forces("Very low pose (rootz=-0.45)", qpos, qvel)


fn test_bent_legs() raises:
    """Bent legs — different contact geometry + joint limits active."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.3
    qpos[3] = -0.5   # bthigh bent
    qpos[4] = 0.8    # bshin extended
    qpos[6] = 0.5    # fthigh bent
    qpos[7] = -0.8   # fshin extended
    var qvel = InlineArray[Float64, NV](fill=0.0)
    compare_solver_forces("Bent legs", qpos, qvel)


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

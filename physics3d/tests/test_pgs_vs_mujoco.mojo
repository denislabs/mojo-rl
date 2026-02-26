"""Test PGS Solver Forces against MuJoCo reference.

Compares our PGSSolver output (qacc, qfrc_constraint, per-row forces)
against MuJoCo's PGS solver after mj_step() for the HalfCheetah model at
configurations with ground contacts.

MuJoCo's PGS is a dual solver (force space), and so is ours. This test
validates that our dual PGS produces comparable results to MuJoCo's PGS.

Note: No primal cost comparison is done here since PGS is dual (operates
in force/lambda space, not qacc space).

What we compare (4 levels):
  1. qfrc_constraint (NV): J^T * lambda vs mj_data.qfrc_constraint
  2. qacc (NV): constrained acceleration vs mj_data.qacc
  3. Total normal force: sum of normal lambdas vs sum of mj efc_force[normals]
  4. Per-contact normal force (informational)

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_pgs_vs_mujoco.mojo
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
from physics3d.solver import PGSSolver
from physics3d.joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from envs.half_cheetah.half_cheetah_xml import HalfCheetahModel
from envs.half_cheetah.half_cheetah_config import HalfCheetahConfig


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64
comptime NQ = HalfCheetahModel.NQ
comptime NV = HalfCheetahModel.NV
comptime NBODY = HalfCheetahModel.NBODY
comptime NJOINT = HalfCheetahModel.NJOINT
comptime NGEOM = HalfCheetahModel.NGEOM
comptime MAX_CONTACTS = HalfCheetahConfig.MAX_CONTACTS
comptime MAX_EQUALITY = 0

comptime V_SIZE = _max_one[NV]()
comptime M_SIZE = _max_one[NV * NV]()
comptime CDOF_SIZE = _max_one[NV * 6]()
comptime CRB_SIZE = _max_one[NBODY * 10]()
comptime MAX_ROWS = 11 * MAX_CONTACTS + 2 * NJOINT

# Tolerances — PGS converges less precisely than Newton/CG, so use looser bounds
comptime QACC_ABS_TOL: Float64 = 1e-1
comptime QACC_REL_TOL: Float64 = 3e-1
comptime QFRC_ABS_TOL: Float64 = 1e-1
comptime QFRC_REL_TOL: Float64 = 3e-1
comptime TOTAL_FORCE_ABS_TOL: Float64 = 1e-1
comptime TOTAL_FORCE_REL_TOL: Float64 = 2e-1
comptime PER_CONTACT_ABS_TOL: Float64 = 2e-1
comptime PER_CONTACT_REL_TOL: Float64 = 5e-1


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
    """Run full pipeline + PGS solver in both engines, compare forces and qacc."""
    print("--- Test:", test_name, "---")

    # === Our engine ===
    var model = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, 0, HalfCheetahModel.CONE_TYPE
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

    # 9. qacc = M^{-1} * f_net via LDL solve (unconstrained acceleration)
    var qacc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        qacc[i] = Scalar[DTYPE](0)
    ldl_solve[DTYPE, NV, M_SIZE, V_SIZE](L, D_ldl, f_net, qacc)

    # 10. Build constraints
    # PGS is dual — it does NOT use M_hat or qfrc_smooth
    var constraints = ConstraintData[DTYPE, MAX_ROWS, NV]()
    build_constraints[CONE_TYPE=HalfCheetahModel.CONE_TYPE](
        model, data, cdof, M_inv, qacc, dt, constraints
    )

    var our_ncon = data.num_contacts
    var our_nnorm = constraints.num_normals
    var our_nfric = constraints.num_friction
    var our_nlim = constraints.num_limits
    print(
        "  Our: contacts=", our_ncon,
        " rows=", constraints.num_rows,
        " (N:", our_nnorm, " F:", our_nfric, " L:", our_nlim, ")",
    )

    # 11. Solve constraints with PGSSolver (modifies qacc in-place)
    PGSSolver.solve[CONE_TYPE=HalfCheetahModel.CONE_TYPE](
        model, data, M_inv, constraints, qacc, dt
    )

    # 12. Compute our qfrc_constraint = J^T * lambda
    var our_qfrc = InlineArray[Float64, NV](fill=0.0)
    for r in range(constraints.num_rows):
        var lam = Float64(constraints.rows[r].lambda_val)
        for i in range(NV):
            our_qfrc[i] += lam * Float64(constraints.J[r * NV + i])

    # Collect our qacc as Float64
    var our_qacc = InlineArray[Float64, NV](fill=0.0)
    for i in range(NV):
        our_qacc[i] = Float64(qacc[i])

    # Collect per-contact forces (pyramidal: 4 edge rows per contact)
    var our_normal_forces = InlineArray[Float64, MAX_CONTACTS](fill=0.0)
    var our_total_normal: Float64 = 0.0
    var ROWS_PER_CON = 4  # condim=3 pyramidal
    for c in range(Int(data.num_contacts)):
        var total_c: Float64 = 0.0
        for e in range(ROWS_PER_CON):
            total_c += Float64(constraints.rows[c * ROWS_PER_CON + e].lambda_val)
        our_normal_forces[c] = total_c
        our_total_normal += total_c

    # === MuJoCo reference via Python ===
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")

    var xml_path = (
        "../Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml"
    )
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    # Match our solver: pyramidal cone, PGS solver, Euler integrator
    mj_model.opt.cone = 0       # pyramidal (matches HalfCheetahModel)
    mj_model.opt.solver = 0     # PGS
    mj_model.opt.integrator = 0 # Euler

    var mj_data = mujoco.MjData(mj_model)

    for i in range(NQ):
        mj_data.qpos[i] = qpos_values[i]
    for i in range(NV):
        mj_data.qvel[i] = qvel_values[i]

    # Use mj_forward to populate all quantities (includes solver)
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

    # MuJoCo pyramidal interleaves 4 edge rows per contact (efc_type=6)
    var mj_contact_start = -1
    for r in range(mj_nefc):
        var t = Int(py=mj_types_flat[r])
        if t == 6:  # mjCNSTR_CONTACT_PYRAMIDAL
            mj_contact_start = r
            break

    var ROWS_PER_MJ_CON = 4  # condim=3 pyramidal
    var mj_normal_forces = InlineArray[Float64, MAX_CONTACTS](fill=0.0)
    mj_total_normal = 0.0
    if mj_contact_start >= 0:
        for c in range(mj_ncon):
            var total_c: Float64 = 0.0
            for e in range(ROWS_PER_MJ_CON):
                var mj_r = mj_contact_start + c * ROWS_PER_MJ_CON + e
                if mj_r < mj_nefc:
                    total_c += Float64(py=mj_efc_force_flat[mj_r])
            mj_normal_forces[c] = total_c
            mj_total_normal += total_c

    # === Comparisons ===
    var all_pass = True

    # 1. qfrc_constraint (NV)
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

    # 2. qacc (NV)
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

    # 5. Per-row forces (informational)
    print()
    print("  --- Per-row forces (informational) ---")
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
        )

    for r in range(mj_nefc):
        var t = Int(py=mj_types_flat[r])
        var tstr: String
        if t == 6:
            tstr = "CP"  # Contact Pyramidal
        elif t == 3:
            tstr = "LI"
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
# Test cases (same 4 configs)
# =============================================================================


fn test_low_pose_static() raises:
    """Low pose (rootz=-0.3), zero velocity."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.3
    var qvel = InlineArray[Float64, NV](fill=0.0)
    compare_solver_forces("Low pose static (rootz=-0.3)", qpos, qvel)


fn test_low_pose_moving() raises:
    """Low pose with velocity — tests friction damping."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.3
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 2.0
    qvel[1] = -0.5
    qvel[3] = -1.0
    compare_solver_forces("Low pose moving", qpos, qvel)


fn test_very_low_pose() raises:
    """Very low pose (rootz=-0.45) — deeper penetration, larger forces."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.45
    var qvel = InlineArray[Float64, NV](fill=0.0)
    compare_solver_forces("Very low pose (rootz=-0.45)", qpos, qvel)


fn test_bent_legs() raises:
    """Bent legs — different contact geometry + joint limits active."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.3
    qpos[3] = -0.5
    qpos[4] = 0.8
    qpos[6] = 0.5
    qpos[7] = -0.8
    var qvel = InlineArray[Float64, NV](fill=0.0)
    compare_solver_forces("Bent legs", qpos, qvel)


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

"""Test Constraint Parameters against MuJoCo reference.

Compares our constraint parameters (K, R, bias, impedance) against MuJoCo's
efc_D, efc_R, efc_aref, efc_KBIP for the HalfCheetah model at configurations
with ground contacts.

Constraint parameters control how "stiff" each constraint is:
  - K (efc_D): Effective mass = J @ M_inv @ J^T (Delassus diagonal)
  - R (efc_R): Regularizer = (1-imp)/imp * K (softens constraint)
  - aref (efc_aref): Reference acceleration the constraint tries to achieve
  - imp: Impedance from solimp smoothstep (0 = soft, 1 = rigid)
  - bias: Our RHS = -aref

MuJoCo reference: efc_D, efc_R, efc_aref, efc_b, efc_KBIP (after mj_step1)

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_constraint_params_vs_mujoco.mojo
"""

from std.python import Python, PythonObject
from std.math import abs, sqrt
from std.collections import InlineArray
from std.testing import assert_true, TestSuite

from physics3d.types import Model, Data, _max_one, ConeType
from physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from physics3d.dynamics.jacobian import compute_cdof, compute_composite_inertia
from physics3d.dynamics.mass_matrix import (
    compute_mass_matrix_full,
    ldl_factor,
    ldl_solve,
    compute_M_inv_from_ldl,
)
from physics3d.collision.contact_detection import detect_contacts
from physics3d.constraints.constraint_builder import build_constraints
from physics3d.constraints.constraint_data import ConstraintData
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
comptime AREF_ABS_TOL: Float64 = 1e-2  # Reference acceleration
comptime AREF_REL_TOL: Float64 = 1e-3
comptime IMP_ABS_TOL: Float64 = 1e-3  # Impedance
comptime DR_ABS_TOL: Float64 = 1e-2  # D/R (now using body_invweight0)
comptime DR_REL_TOL: Float64 = 1e-2
# Minimum K for friction tangent — below this, direction is degenerate (skip)
comptime FRIC_K_MIN: Float64 = 1e-6


# =============================================================================
# Helper: compare a scalar parameter
# =============================================================================


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
    if not ok:
        print(
            "    FAIL",
            label,
            " ours=",
            our_val,
            " mj=",
            mj_val,
            " abs=",
            abs_err,
            " rel=",
            rel_err,
        )
    return ok


# =============================================================================
# Comparison helper
# =============================================================================


@no_inline
fn compare_constraint_params(
    test_name: String,
    qpos_values: InlineArray[Float64, NQ],
    qvel_values: InlineArray[Float64, NV],
) raises:
    """Compute constraints in both engines with identical state, compare params.
    """
    print("--- Test:", test_name, "---")

    # === Our engine ===
    var model = Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        HalfCheetahModel.MAX_EQUALITY,
        HalfCheetahModel.CONE_TYPE,
        HalfCheetahModel.MAX_TENDON,
        HalfCheetahModel.NSITE,
    ]()
    var data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE
    ]()
    HalfCheetahModel.setup_model_and_data(model, data)

    # Now set the test configuration
    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_values[i])
    for i in range(NV):
        data.qvel[i] = Scalar[DTYPE](qvel_values[i])

    # 1. FK + body velocities + cdof
    forward_kinematics(model, data)
    compute_body_velocities(model, data)

    var cdof = List[Scalar[DTYPE]](capacity=CDOF_SIZE)
    for _ in range(CDOF_SIZE):
        cdof.append(Scalar[DTYPE](0))
    compute_cdof(model, data, cdof)

    # 2. Contact detection
    detect_contacts[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM](
        model, data
    )

    # 3. Mass matrix + armature + dt*D -> LDL -> M_inv
    var crb = List[Scalar[DTYPE]](capacity=CRB_SIZE)
    for _ in range(CRB_SIZE):
        crb.append(Scalar[DTYPE](0))
    compute_composite_inertia(model, data, crb)

    var M = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        M.append(Scalar[DTYPE](0))
    compute_mass_matrix_full(model, data, cdof, crb, M)

    var dt = Scalar[DTYPE](0.01)
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var dof_adr = joint.dof_adr
        var arm = joint.armature
        var damp = joint.damping
        if joint.jnt_type == JNT_FREE:
            for d in range(6):
                M[(dof_adr + d) * NV + (dof_adr + d)] += arm + dt * damp
        elif joint.jnt_type == JNT_BALL:
            for d in range(3):
                M[(dof_adr + d) * NV + (dof_adr + d)] += arm + dt * damp
        else:
            M[dof_adr * NV + dof_adr] += arm + dt * damp

    var L = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        L.append(Scalar[DTYPE](0))
    var D_ldl = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        D_ldl.append(Scalar[DTYPE](0))
    ldl_factor[DTYPE, NV](M, L, D_ldl)

    var M_inv = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        M_inv.append(Scalar[DTYPE](0))
    compute_M_inv_from_ldl[DTYPE, NV](L, D_ldl, M_inv)

    # 4. Build constraints
    var constraints = ConstraintData[DTYPE, MAX_ROWS, NV]()
    build_constraints(model, data, cdof, M_inv, dt, constraints)

    var our_ncon = data.num_contacts
    var our_nnorm = constraints.num_normals
    var our_nfric = constraints.num_friction
    var our_nlim = constraints.num_limits
    print(
        "  Our: contacts=",
        our_ncon,
        " rows=",
        constraints.num_rows,
        " (N:",
        our_nnorm,
        " F:",
        our_nfric,
        " L:",
        our_nlim,
        ")",
    )

    # === MuJoCo reference via Python ===
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")

    var xml_path = (
        "../Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml"
    )
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    # Set elliptic cone to match our engine
    mj_model.opt.cone = 0  # mjCONE_PYRAMIDAL (matches HalfCheetahModel)

    var mj_data = mujoco.MjData(mj_model)

    for i in range(NQ):
        mj_data.qpos[i] = qpos_values[i]
    for i in range(NV):
        mj_data.qvel[i] = qvel_values[i]

    # mj_step1 runs FK + collision + constraint setup (no solver)
    mujoco.mj_step1(mj_model, mj_data)

    var mj_nefc = Int(py=mj_data.nefc)
    var mj_ncon = Int(py=mj_data.ncon)
    print("  MJ:  contacts=", mj_ncon, " rows=", mj_nefc)

    if mj_nefc == 0 and constraints.num_rows == 0:
        print("  ALL OK  (no constraints)")
        return

    # Get MuJoCo efc arrays
    var mj_D = mj_data.efc_D.flatten().tolist()
    var mj_R = mj_data.efc_R.flatten().tolist()
    var mj_aref = mj_data.efc_aref.flatten().tolist()
    var mj_b = mj_data.efc_b.flatten().tolist()
    var mj_types = mj_data.efc_type.flatten().tolist()
    # efc_KBIP: (nefc, 4) — [K_spring, B_damp, imp, pos] per row
    var mj_KBIP = mj_data.efc_KBIP.reshape(mj_nefc, 4)

    # Count MuJoCo contact vs limit rows
    var mj_contact_rows = 0
    var mj_limit_rows = 0
    for r in range(mj_nefc):
        var t = Int(py=mj_types[r])
        if t == 7:  # mjCNSTR_CONTACT_ELLIPTIC
            mj_contact_rows += 1
        elif t == 3:  # mjCNSTR_LIMIT_JOINT
            mj_limit_rows += 1
    print(
        "  MJ:  contact_rows=",
        mj_contact_rows,
        " limit_rows=",
        mj_limit_rows,
    )

    # Build MuJoCo row index maps
    var mj_contact_row_indices = InlineArray[Int, MAX_ROWS](fill=-1)
    var mj_idx = 0
    for r in range(mj_nefc):
        var t = Int(py=mj_types[r])
        if t == 7:
            mj_contact_row_indices[mj_idx] = r
            mj_idx += 1

    var mj_limit_row_indices = InlineArray[Int, MAX_ROWS](fill=-1)
    var ml_idx = 0
    for r in range(mj_nefc):
        var t = Int(py=mj_types[r])
        if t == 3:
            mj_limit_row_indices[ml_idx] = r
            ml_idx += 1

    # === Compare ===
    # We compare imp (impedance) and aref (reference acceleration) which
    # validate the constraint setup (K_spring, B_damp, solimp smoothstep).
    # We skip D (Delassus diagonal) and R (regularizer) because:
    #   - Our engine: D = J @ M_inv @ J^T (exact Delassus diagonal)
    #   - MuJoCo: D = imp/((1-imp)*invweight0) (body-level approximation)
    # These are fundamentally different approaches, both valid.
    var all_pass = True
    var total_checks = 0
    var pass_checks = 0

    # --- Compare per-contact parameters ---
    # Row mapping (elliptic, condim=3):
    #   MuJoCo interleaves: [n0, t1_0, t2_0, n1, t1_1, t2_1, ...]
    #   Ours groups: [n0,...,nN, t1_0, t2_0, t1_1, t2_1, ...]

    for c in range(our_ncon):
        if c * 3 + 2 >= mj_contact_rows:
            print("  WARN: contact", c, "out of MJ range")
            break

        print("  Contact", c, ":")

        # --- Normal row ---
        var our_norm_idx = c
        var mj_norm_idx = mj_contact_row_indices[c * 3]

        var our_K_n = Float64(constraints.rows[our_norm_idx].K)
        var mj_D_n = Float64(py=mj_D[mj_norm_idx])
        var mj_R_n = Float64(py=mj_R[mj_norm_idx])

        # Compute our impedance: recover from diagApprox and inv_K_imp
        var our_inv_K_imp_n = Float64(constraints.rows[our_norm_idx].inv_K_imp)
        var our_R_n = 1.0 / our_inv_K_imp_n - our_K_n
        var our_diagApprox_n = Float64(
            constraints.rows[our_norm_idx].diagApprox
        )
        var our_imp_n: Float64
        if our_diagApprox_n > 1e-12:
            our_imp_n = our_diagApprox_n / (our_diagApprox_n + our_R_n)
        else:
            our_imp_n = our_inv_K_imp_n * our_K_n
        # Our D = 1/R
        var our_D_n = 1.0 / our_R_n if our_R_n > 1e-12 else 0.0

        # impedance from KBIP
        var mj_imp_n = Float64(py=mj_KBIP[mj_norm_idx][2])
        total_checks += 1
        if compare_scalar(
            "imp(normal)", our_imp_n, mj_imp_n, IMP_ABS_TOL, 0.01
        ):
            pass_checks += 1
        else:
            all_pass = False

        # aref: our bias = -aref
        var our_aref_n = -Float64(constraints.rows[our_norm_idx].bias)
        var mj_aref_n = Float64(py=mj_aref[mj_norm_idx])
        total_checks += 1
        if compare_scalar(
            "aref(normal)", our_aref_n, mj_aref_n, AREF_ABS_TOL, AREF_REL_TOL
        ):
            pass_checks += 1
        else:
            all_pass = False

        # D (Delassus / primal stiffness)
        total_checks += 1
        if compare_scalar("D(normal)", our_D_n, mj_D_n, DR_ABS_TOL, DR_REL_TOL):
            pass_checks += 1
        else:
            all_pass = False

        # R (regularizer)
        total_checks += 1
        if compare_scalar("R(normal)", our_R_n, mj_R_n, DR_ABS_TOL, DR_REL_TOL):
            pass_checks += 1
        else:
            all_pass = False

        # Print summary for normal
        print(
            "    normal: K=",
            our_K_n,
            " imp=",
            our_imp_n,
            " aref=",
            our_aref_n,
            " D=",
            our_D_n,
            " R=",
            our_R_n,
        )
        print(
            "    mj:     D=",
            mj_D_n,
            " imp=",
            mj_imp_n,
            " aref=",
            mj_aref_n,
            " R=",
            mj_R_n,
        )

        # --- Friction rows ---
        var our_t1_idx = our_nnorm + c * 2
        var our_t2_idx = our_nnorm + c * 2 + 1
        var mj_t1_idx = mj_contact_row_indices[c * 3 + 1]
        var mj_t2_idx = mj_contact_row_indices[c * 3 + 2]

        var our_K_t1 = Float64(constraints.rows[our_t1_idx].K)
        var our_K_t2 = Float64(constraints.rows[our_t2_idx].K)

        # Match tangent directions by closest K value (basis can be swapped)
        var mj_D_t1 = Float64(py=mj_D[mj_t1_idx])
        var mj_D_t2 = Float64(py=mj_D[mj_t2_idx])
        var err_direct = abs(our_K_t1 - mj_D_t1) + abs(our_K_t2 - mj_D_t2)
        var err_swap = abs(our_K_t1 - mj_D_t2) + abs(our_K_t2 - mj_D_t1)
        var swap = err_swap < err_direct
        var matched_mj_t1 = mj_t1_idx if not swap else mj_t2_idx
        var matched_mj_t2 = mj_t2_idx if not swap else mj_t1_idx

        # Skip degenerate friction tangent (K < threshold)
        if our_K_t1 > FRIC_K_MIN:
            var our_aref_t1 = -Float64(constraints.rows[our_t1_idx].bias)
            var mj_aref_t1 = Float64(py=mj_aref[matched_mj_t1])
            total_checks += 1
            if compare_scalar(
                "aref(fric_t1)",
                our_aref_t1,
                mj_aref_t1,
                AREF_ABS_TOL,
                AREF_REL_TOL,
            ):
                pass_checks += 1
            else:
                all_pass = False

            # D/R for friction
            var our_R_t1 = (
                1.0 / Float64(constraints.rows[our_t1_idx].inv_K_imp) - our_K_t1
            )
            var mj_R_t1 = Float64(py=mj_R[matched_mj_t1])
            total_checks += 1
            if compare_scalar(
                "R(fric_t1)", our_R_t1, mj_R_t1, DR_ABS_TOL, DR_REL_TOL
            ):
                pass_checks += 1
            else:
                all_pass = False

            print(
                "    fric_t1: K=",
                our_K_t1,
                " aref=",
                our_aref_t1,
                " R=",
                our_R_t1,
                " mj_R=",
                mj_R_t1,
            )
        else:
            print("    fric_t1: SKIP (degenerate K=", our_K_t1, ")")

    # --- Compare limit rows ---
    if our_nlim > 0 or mj_limit_rows > 0:
        print("  Limits (ours:", our_nlim, " mj:", mj_limit_rows, "):")

        if our_nlim != mj_limit_rows:
            print(
                "  FAIL: limit count mismatch! ours=",
                our_nlim,
                " mj=",
                mj_limit_rows,
            )
            all_pass = False
        else:
            for ll in range(our_nlim):
                var our_lim_idx = our_nnorm + our_nfric + ll
                var mj_lim_idx = mj_limit_row_indices[ll]

                # impedance
                var our_K_lim = Float64(constraints.rows[our_lim_idx].K)
                var our_inv_K_imp_lim = Float64(
                    constraints.rows[our_lim_idx].inv_K_imp
                )
                var our_R_lim = 1.0 / our_inv_K_imp_lim - our_K_lim
                var our_diagApprox_lim = Float64(
                    constraints.rows[our_lim_idx].diagApprox
                )
                var our_imp_lim: Float64
                if our_diagApprox_lim > 1e-12:
                    our_imp_lim = our_diagApprox_lim / (
                        our_diagApprox_lim + our_R_lim
                    )
                else:
                    our_imp_lim = our_inv_K_imp_lim * our_K_lim
                var our_D_lim = 1.0 / our_R_lim if our_R_lim > 1e-12 else 0.0
                var mj_imp_lim = Float64(py=mj_KBIP[mj_lim_idx][2])
                total_checks += 1
                if compare_scalar(
                    "imp(limit_" + String(ll) + ")",
                    our_imp_lim,
                    mj_imp_lim,
                    IMP_ABS_TOL,
                    0.01,
                ):
                    pass_checks += 1
                else:
                    all_pass = False

                # aref
                var our_aref_lim = -Float64(constraints.rows[our_lim_idx].bias)
                var mj_aref_lim = Float64(py=mj_aref[mj_lim_idx])
                total_checks += 1
                if compare_scalar(
                    "aref(limit_" + String(ll) + ")",
                    our_aref_lim,
                    mj_aref_lim,
                    AREF_ABS_TOL,
                    AREF_REL_TOL,
                ):
                    pass_checks += 1
                else:
                    all_pass = False

                # D and R for limits
                var mj_D_lim = Float64(py=mj_D[mj_lim_idx])
                var mj_R_lim = Float64(py=mj_R[mj_lim_idx])
                total_checks += 1
                if compare_scalar(
                    "D(limit_" + String(ll) + ")",
                    our_D_lim,
                    mj_D_lim,
                    DR_ABS_TOL,
                    DR_REL_TOL,
                ):
                    pass_checks += 1
                else:
                    all_pass = False
                total_checks += 1
                if compare_scalar(
                    "R(limit_" + String(ll) + ")",
                    our_R_lim,
                    mj_R_lim,
                    DR_ABS_TOL,
                    DR_REL_TOL,
                ):
                    pass_checks += 1
                else:
                    all_pass = False

                print(
                    "    limit_" + String(ll) + ": K=",
                    our_K_lim,
                    " imp=",
                    our_imp_lim,
                    " aref=",
                    our_aref_lim,
                    " D=",
                    our_D_lim,
                    " R=",
                    our_R_lim,
                )
                print(
                    "    mj:       D=",
                    mj_D_lim,
                    " imp=",
                    mj_imp_lim,
                    " R=",
                    mj_R_lim,
                    " aref=",
                    mj_aref_lim,
                )

    print()
    print(
        "  Checks:",
        pass_checks,
        "/",
        total_checks,
        "passed",
    )
    if all_pass:
        print("  ALL OK")
    else:
        print("  FAILED")

    assert_true(all_pass, "Constraint params mismatch for: " + test_name)


# =============================================================================
# Test cases
# =============================================================================


fn test_low_pose_static() raises:
    """Low pose (rootz=-0.3), zero velocity."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.3  # rootz low
    var qvel = InlineArray[Float64, NV](fill=0.0)
    compare_constraint_params("Low pose static (rootz=-0.3)", qpos, qvel)


fn test_low_pose_moving() raises:
    """Low pose with velocity — bias should include velocity damping terms."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.3  # rootz low
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 2.0  # moving forward
    qvel[1] = -0.5  # moving down
    qvel[3] = -1.0  # bthigh rotating
    compare_constraint_params("Low pose moving", qpos, qvel)


fn test_very_low_pose() raises:
    """Very low pose (rootz=-0.45) — deeper penetration, different impedance."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.45  # rootz very low
    var qvel = InlineArray[Float64, NV](fill=0.0)
    compare_constraint_params("Very low pose (rootz=-0.45)", qpos, qvel)


fn test_bent_legs() raises:
    """Bent legs — different contact geometry + joint limits active."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.3
    qpos[3] = -0.5  # bthigh bent
    qpos[4] = 0.8  # bshin extended
    qpos[6] = 0.5  # fthigh bent
    qpos[7] = -0.8  # fshin extended
    var qvel = InlineArray[Float64, NV](fill=0.0)
    compare_constraint_params("Bent legs", qpos, qvel)


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

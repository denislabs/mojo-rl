"""Test Constraint Jacobians against MuJoCo reference.

Compares our constraint Jacobian rows (J matrix) against MuJoCo's efc_J
for the HalfCheetah model at configurations with ground contacts.

The constraint Jacobian maps joint velocities to constraint-space velocities:
    constraint_vel = J @ qvel
Each row of J corresponds to one constraint (normal, friction_t1, friction_t2, etc.)

MuJoCo reference: mj_data.efc_J (nefc x NV matrix, after mj_step1)
  - Must set model.opt.cone = 1 (elliptic) to match our parameterization
  - MuJoCo interleaves per-contact: [normal_0, t1_0, t2_0, normal_1, t1_1, t2_1, ...]
  - Our layout: [normal_0, ..., normal_N, t1_0, t2_0, t1_1, t2_1, ...]
  - MuJoCo efc_type: 7 = mjCNSTR_CONTACT_ELLIPTIC, 3 = mjCNSTR_LIMIT_JOINT

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_jacobian_vs_mujoco.mojo
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
from envs.half_cheetah.half_cheetah_def import (
    HalfCheetahModel,
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
comptime MAX_EQUALITY = 0

comptime V_SIZE = _max_one[NV]()
comptime M_SIZE = _max_one[NV * NV]()
comptime CDOF_SIZE = _max_one[NV * 6]()
comptime CRB_SIZE = _max_one[NBODY * 10]()
comptime MAX_ROWS = 11 * MAX_CONTACTS + 2 * NJOINT

# Tolerances
comptime ABS_TOL: Float64 = 1e-4
comptime REL_TOL: Float64 = 1e-3


# =============================================================================
# Helper: compare a single J row
# =============================================================================


fn compare_J_row(
    label: String,
    our_row_idx: Int,
    mj_J_row: PythonObject,
    constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
    sign: Float64 = 1.0,
) raises -> Bool:
    """Compare one Jacobian row element-by-element.

    sign: 1.0 for direct comparison, -1.0 to compare our row against -mj_row
    (tangent directions can be negated and still valid).
    """
    var max_abs_err: Float64 = 0.0
    var max_rel_err: Float64 = 0.0
    var fail_count = 0

    for col in range(NV):
        var our_val = Float64(constraints.J[our_row_idx * NV + col])
        var mj_val = sign * Float64(py=mj_J_row[col])
        var abs_err = abs(our_val - mj_val)
        var ref_mag = abs(mj_val)
        var rel_err: Float64 = 0.0
        if ref_mag > 1e-10:
            rel_err = abs_err / ref_mag

        if abs_err > max_abs_err:
            max_abs_err = abs_err
        if rel_err > max_rel_err:
            max_rel_err = rel_err

        var ok = abs_err < ABS_TOL or rel_err < REL_TOL
        if not ok:
            if fail_count < 3:
                print(
                    "    FAIL",
                    label,
                    "[",
                    col,
                    "]",
                    " ours=",
                    our_val,
                    " mj=",
                    mj_val,
                    " abs=",
                    abs_err,
                )
            fail_count += 1

    if fail_count == 0:
        print(
            "    ",
            label,
            " OK  max_abs=",
            max_abs_err,
            " max_rel=",
            max_rel_err,
        )
    else:
        print(
            "    ",
            label,
            " FAILED",
            fail_count,
            "/",
            NV,
            " max_abs=",
            max_abs_err,
        )
        # Print full rows for debugging
        print("      Our:", end="")
        for col in range(NV):
            print(" ", Float64(constraints.J[our_row_idx * NV + col]), end="")
        print()
        var mj_list = mj_J_row.tolist()
        print("      MJ :", end="")
        for col in range(NV):
            print(" ", sign * Float64(py=mj_list[col]), end="")
        print()

    return fail_count == 0


# =============================================================================
# Comparison helper
# =============================================================================


fn compare_jacobians(
    test_name: String,
    qpos_values: InlineArray[Float64, NQ],
    qvel_values: InlineArray[Float64, NV],
) raises -> Bool:
    """Compute constraint Jacobians in both engines with identical state, compare.
    """
    print("--- Test:", test_name, "---")

    # === Our engine ===
    var model = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, 0, HalfCheetahModel.CONE_TYPE
    ](
    )
    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()
    HalfCheetahModel.setup_model_and_data(model, data)
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

    # 3. Mass matrix + armature + dt*D → LDL → M_inv
    var crb = InlineArray[Scalar[DTYPE], CRB_SIZE](uninitialized=True)
    for i in range(CRB_SIZE):
        crb[i] = Scalar[DTYPE](0)
    compute_composite_inertia(model, data, crb)

    var M = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    for i in range(M_SIZE):
        M[i] = Scalar[DTYPE](0)
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

    var L = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    var D_ldl = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    ldl_factor[DTYPE, NV, M_SIZE, V_SIZE](M, L, D_ldl)

    var M_inv = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    for i in range(M_SIZE):
        M_inv[i] = Scalar[DTYPE](0)
    compute_M_inv_from_ldl[DTYPE, NV, M_SIZE, V_SIZE](L, D_ldl, M_inv)

    # 4. Build constraints
    var qvel_arr = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        qvel_arr[i] = Scalar[DTYPE](qvel_values[i])

    var constraints = ConstraintData[DTYPE, MAX_ROWS, NV]()
    build_constraints(model, data, cdof, M_inv, qvel_arr, dt, constraints)

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
        return True

    # Get MuJoCo efc_J (nefc x NV) and efc_type
    var mj_J = mj_data.efc_J.reshape(mj_nefc, NV)
    var mj_types = mj_data.efc_type.flatten().tolist()

    # MuJoCo elliptic layout (condim=3): interleaved per contact
    #   [normal_0, t1_0, t2_0, normal_1, t1_1, t2_1, ...]
    #   efc_type=7 for all contact rows
    # Our layout: grouped
    #   [normal_0, ..., normal_N, t1_0, t2_0, t1_1, t2_1, ...]

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

    # Verify row counts match
    var expected_contact_rows = our_ncon * 3  # normal + t1 + t2
    var all_pass = True

    if mj_contact_rows != expected_contact_rows:
        print(
            "  WARN: MJ contact rows=",
            mj_contact_rows,
            " expected=",
            expected_contact_rows,
        )

    if our_nnorm + our_nfric != mj_contact_rows:
        print(
            "  FAIL: total contact rows mismatch! ours=",
            our_nnorm + our_nfric,
            " mj=",
            mj_contact_rows,
        )
        all_pass = False

    # === Compare per-contact Jacobian rows ===
    # Match contacts by order (same detection order verified in contact test)
    # MuJoCo: contact c → rows [c*3, c*3+1, c*3+2] = [normal, t1, t2]
    # But MuJoCo may have limit rows mixed in... find contact-only row indices
    var mj_contact_row_indices = InlineArray[Int, MAX_ROWS](fill=-1)
    var mj_idx = 0
    for r in range(mj_nefc):
        var t = Int(py=mj_types[r])
        if t == 7:  # contact pyramidal
            mj_contact_row_indices[mj_idx] = r
            mj_idx += 1

    for c in range(our_ncon):
        print("  Contact", c, ":")

        # --- Normal row ---
        # Our: row index = c
        # MuJoCo: row index = mj_contact_row_indices[c * 3]
        var mj_normal_idx = mj_contact_row_indices[c * 3]
        var our_normal_idx = c
        if not compare_J_row(
            "Normal", our_normal_idx, mj_J[mj_normal_idx], constraints
        ):
            all_pass = False

        # --- Friction rows ---
        # Tangent basis can differ between engines (both valid, just different
        # parameterization of the tangent plane). Instead of matching t1↔t1,
        # find the best match: compare our t1 against both MJ t1 and MJ t2,
        # and vice versa.
        var mj_t1_idx = mj_contact_row_indices[c * 3 + 1]
        var mj_t2_idx = mj_contact_row_indices[c * 3 + 2]
        var our_t1_idx = our_nnorm + c * 2
        var our_t2_idx = our_nnorm + c * 2 + 1

        # Compute error for both possible matchings:
        #   Matching A: our_t1↔mj_t1, our_t2↔mj_t2
        #   Matching B: our_t1↔mj_t2, our_t2↔mj_t1
        var err_A1: Float64 = 0.0
        var err_A2: Float64 = 0.0
        var err_B1: Float64 = 0.0
        var err_B2: Float64 = 0.0

        var mj_t1_list = mj_J[mj_t1_idx].flatten().tolist()
        var mj_t2_list = mj_J[mj_t2_idx].flatten().tolist()

        for col in range(NV):
            var our_v1 = Float64(constraints.J[our_t1_idx * NV + col])
            var our_v2 = Float64(constraints.J[our_t2_idx * NV + col])
            var mj_v1 = Float64(py=mj_t1_list[col])
            var mj_v2 = Float64(py=mj_t2_list[col])
            err_A1 += abs(our_v1 - mj_v1)
            err_A2 += abs(our_v2 - mj_v2)
            err_B1 += abs(our_v1 - mj_v2)
            err_B2 += abs(our_v2 - mj_v1)

        var err_A = err_A1 + err_A2
        var err_B = err_B1 + err_B2

        # Also check sign-flipped matching (tangent can be negated)
        var err_C1: Float64 = 0.0
        var err_C2: Float64 = 0.0
        var err_D1: Float64 = 0.0
        var err_D2: Float64 = 0.0
        for col in range(NV):
            var our_v1 = Float64(constraints.J[our_t1_idx * NV + col])
            var our_v2 = Float64(constraints.J[our_t2_idx * NV + col])
            var mj_v1 = Float64(py=mj_t1_list[col])
            var mj_v2 = Float64(py=mj_t2_list[col])
            err_C1 += abs(our_v1 + mj_v1)  # sign flipped
            err_C2 += abs(our_v2 + mj_v2)
            err_D1 += abs(our_v1 + mj_v2)  # swapped + sign flipped
            err_D2 += abs(our_v2 + mj_v1)
        var err_C = err_C1 + err_C2
        var err_D = err_D1 + err_D2

        # Pick best matching
        var best_err = err_A
        var best_label = String("A(t1↔t1)")
        if err_B < best_err:
            best_err = err_B
            best_label = "B(t1↔t2)"
        if err_C < best_err:
            best_err = err_C
            best_label = "C(t1↔-t1)"
        if err_D < best_err:
            best_err = err_D
            best_label = "D(t1↔-t2)"

        # Determine sign and swap for best matching
        var swap = best_label == "B(t1↔t2)" or best_label == "D(t1↔-t2)"
        var negate = best_label == "C(t1↔-t1)" or best_label == "D(t1↔-t2)"
        var sign: Float64 = -1.0 if negate else 1.0

        if not swap:
            if not compare_J_row(
                "Fric_t1", our_t1_idx, mj_J[mj_t1_idx], constraints, sign
            ):
                all_pass = False
            if not compare_J_row(
                "Fric_t2", our_t2_idx, mj_J[mj_t2_idx], constraints, sign
            ):
                all_pass = False
        else:
            # Swapped: our_t1 matches MJ_t2 and our_t2 matches MJ_t1
            if not compare_J_row(
                "Fric_t1(↔mj_t2)",
                our_t1_idx,
                mj_J[mj_t2_idx],
                constraints,
                sign,
            ):
                all_pass = False
            if not compare_J_row(
                "Fric_t2(↔mj_t1)",
                our_t2_idx,
                mj_J[mj_t1_idx],
                constraints,
                sign,
            ):
                all_pass = False

        print("    (tangent matching:", best_label, " err=", best_err, ")")

    # === Compare limit rows (if any) ===
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
            # Get MuJoCo limit row indices
            var mj_limit_row_indices = InlineArray[Int, MAX_ROWS](fill=-1)
            var ml_idx = 0
            for r in range(mj_nefc):
                var t = Int(py=mj_types[r])
                if t == 3:  # mjCNSTR_LIMIT_JOINT
                    mj_limit_row_indices[ml_idx] = r
                    ml_idx += 1

            for ll in range(our_nlim):
                var our_lim_idx = our_nnorm + our_nfric + ll
                var mj_lim_idx = mj_limit_row_indices[ll]
                if not compare_J_row(
                    "Limit", our_lim_idx, mj_J[mj_lim_idx], constraints
                ):
                    all_pass = False

    print()
    if all_pass:
        print("  ALL OK")
    else:
        print("  FAILED")

    return all_pass


# =============================================================================
# Test cases
# =============================================================================


fn test_low_pose_static() raises -> Bool:
    """Low pose (rootz=-0.3), zero velocity — basic contact Jacobians."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.3  # rootz low
    var qvel = InlineArray[Float64, NV](fill=0.0)
    return compare_jacobians("Low pose static (rootz=-0.3)", qpos, qvel)


fn test_low_pose_moving() raises -> Bool:
    """Low pose with velocity — Jacobians should be same (velocity-independent).
    """
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.3  # rootz low
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 2.0  # moving forward
    qvel[1] = -0.5  # moving down
    qvel[3] = -1.0  # bthigh rotating
    return compare_jacobians("Low pose moving", qpos, qvel)


fn test_very_low_pose() raises -> Bool:
    """Very low pose (rootz=-0.45) — more contacts."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.45  # rootz very low
    var qvel = InlineArray[Float64, NV](fill=0.0)
    return compare_jacobians("Very low pose (rootz=-0.45)", qpos, qvel)


fn test_bent_legs() raises -> Bool:
    """Bent legs — different contact geometry + joint limits active."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.3
    qpos[3] = -0.5  # bthigh bent
    qpos[4] = 0.8  # bshin extended
    qpos[6] = 0.5  # fthigh bent
    qpos[7] = -0.8  # fshin extended
    var qvel = InlineArray[Float64, NV](fill=0.0)
    return compare_jacobians("Bent legs", qpos, qvel)


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    print("=" * 60)
    print("Constraint Jacobians: Mojo Engine vs MuJoCo")
    print("=" * 60)
    print("Model: HalfCheetah (NV=", NV, ")")
    print("MuJoCo cone: elliptic (to match our engine)")
    print("Tolerances: abs=", ABS_TOL, " rel=", REL_TOL)
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

"""Test Constraint Jacobians against MuJoCo reference.

Compares our constraint Jacobian rows (J matrix) against MuJoCo's efc_J
for the HalfCheetah model at configurations with ground contacts.

The constraint Jacobian maps joint velocities to constraint-space velocities:
    constraint_vel = J @ qvel
Each row of J corresponds to one constraint (normal, friction_t1, friction_t2, etc.)

MuJoCo reference: mj_data.efc_J (nefc x NV matrix, after mj_step1)
  - Must set model.opt.cone = 0 (pyramidal) to match HalfCheetahModel
  - MuJoCo pyramidal interleaves per-contact: [edge0_0, edge1_0, edge2_0, edge3_0, ...]
  - Our layout: all edge rows stored as "normals" (num_friction=0 for pyramidal)
  - MuJoCo efc_type: 6 = mjCNSTR_CONTACT_PYRAMIDAL, 3 = mjCNSTR_LIMIT_JOINT

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_jacobian_vs_mujoco.mojo
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
) raises:
    """Compute constraint Jacobians in both engines with identical state, compare.
    """
    print("--- Test:", test_name, "---")

    # === Our engine ===
    var model = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, HalfCheetahModel.MAX_EQUALITY, HalfCheetahModel.CONE_TYPE, HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE
    ]()
    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
    HalfCheetahModel.setup_model_and_data(model, data)
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

    # 3. Mass matrix + armature + dt*D → LDL → M_inv
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

    # Get MuJoCo efc_J (nefc x NV) and efc_type
    var mj_J = mj_data.efc_J.reshape(mj_nefc, NV)
    var mj_types = mj_data.efc_type.flatten().tolist()

    # MuJoCo pyramidal layout (condim=3): interleaved per contact
    #   [edge0_0, edge1_0, edge2_0, edge3_0, edge0_1, edge1_1, ...]
    #   efc_type=6 (mjCNSTR_CONTACT_PYRAMIDAL) for all contact rows
    # Our layout: all edge rows stored as "normals" (num_friction=0)
    #   [edge0_0, edge1_0, edge2_0, edge3_0, edge0_1, edge1_1, ...]

    # Count MuJoCo contact vs limit rows
    var mj_contact_rows = 0
    var mj_limit_rows = 0
    for r in range(mj_nefc):
        var t = Int(py=mj_types[r])
        if t == 6:  # mjCNSTR_CONTACT_PYRAMIDAL
            mj_contact_rows += 1
        elif t == 3:  # mjCNSTR_LIMIT_JOINT
            mj_limit_rows += 1
    print(
        "  MJ:  contact_rows=",
        mj_contact_rows,
        " limit_rows=",
        mj_limit_rows,
    )

    # Verify row counts match (condim=3 pyramidal: 4 edge rows per contact)
    var ROWS_PER_CONTACT = 4
    var expected_contact_rows = our_ncon * ROWS_PER_CONTACT
    var all_pass = True

    if mj_contact_rows != expected_contact_rows:
        print(
            "  WARN: MJ contact rows=",
            mj_contact_rows,
            " expected=",
            expected_contact_rows,
        )

    # our_nnorm contains all edge rows (num_friction=0 for pyramidal)
    if our_nnorm != mj_contact_rows:
        print(
            "  FAIL: total contact rows mismatch! ours=",
            our_nnorm,
            " mj=",
            mj_contact_rows,
        )
        all_pass = False

    # === Compare per-contact Jacobian rows ===
    # Match contacts by order (same detection order verified in contact test)
    # MuJoCo pyramidal: contact c → rows [c*4, c*4+1, c*4+2, c*4+3]
    # Our pyramidal: same ordering of edge rows stored in "normal" slots
    var mj_contact_row_indices = InlineArray[Int, MAX_ROWS](fill=-1)
    var mj_idx = 0
    for r in range(mj_nefc):
        var t = Int(py=mj_types[r])
        if t == 6:  # mjCNSTR_CONTACT_PYRAMIDAL
            mj_contact_row_indices[mj_idx] = r
            mj_idx += 1

    for c in range(our_ncon):
        print("  Contact", c, ":")

        # Compare all 4 edge rows directly
        for e in range(ROWS_PER_CONTACT):
            var our_row_idx = c * ROWS_PER_CONTACT + e
            var mj_row_idx = mj_contact_row_indices[c * ROWS_PER_CONTACT + e]
            var label = String("Edge") + String(e)
            if not compare_J_row(
                label, our_row_idx, mj_J[mj_row_idx], constraints
            ):
                all_pass = False

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

    assert_true(all_pass, "Jacobian mismatch for: " + test_name)


# =============================================================================
# Test cases
# =============================================================================


fn test_low_pose_static() raises:
    """Low pose (rootz=-0.3), zero velocity — basic contact Jacobians."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.3  # rootz low
    var qvel = InlineArray[Float64, NV](fill=0.0)
    compare_jacobians("Low pose static (rootz=-0.3)", qpos, qvel)


fn test_low_pose_moving() raises:
    """Low pose with velocity — Jacobians should be same (velocity-independent).
    """
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.3  # rootz low
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 2.0  # moving forward
    qvel[1] = -0.5  # moving down
    qvel[3] = -1.0  # bthigh rotating
    compare_jacobians("Low pose moving", qpos, qvel)


fn test_very_low_pose() raises:
    """Very low pose (rootz=-0.45) — more contacts."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.45  # rootz very low
    var qvel = InlineArray[Float64, NV](fill=0.0)
    compare_jacobians("Very low pose (rootz=-0.45)", qpos, qvel)


fn test_bent_legs() raises:
    """Bent legs — different contact geometry + joint limits active."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = -0.3
    qpos[3] = -0.5  # bthigh bent
    qpos[4] = 0.8  # bshin extended
    qpos[6] = 0.5  # fthigh bent
    qpos[7] = -0.8  # fshin extended
    var qvel = InlineArray[Float64, NV](fill=0.0)
    compare_jacobians("Bent legs", qpos, qvel)


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

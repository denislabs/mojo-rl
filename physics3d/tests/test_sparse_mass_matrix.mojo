"""Test sparse vs dense mass matrix for correctness.

Verifies that SparseMassMatrix (CSR format, MuJoCo-compatible) produces
identical results to compute_mass_matrix_full (dense) for the HalfCheetah
model. Also verifies that ldl_solve_sparse gives the same solution as
ldl_solve (dense).

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_sparse_mass_matrix.mojo
"""

from std.math import abs
from std.collections import InlineArray

from physics3d.types import Model, Data, _max_one
from physics3d.kinematics.forward_kinematics import forward_kinematics
from physics3d.dynamics.jacobian import compute_cdof, compute_composite_inertia
from physics3d.dynamics.mass_matrix import (
    compute_mass_matrix_full,
    ldl_factor,
    ldl_solve,
    SparseMassMatrix,
    build_sparse_pattern,
    count_sparse_nnz,
    compute_mass_matrix_sparse,
    ldl_factor_sparse,
    ldl_solve_sparse,
    sparse_to_dense,
)
from envs.half_cheetah.half_cheetah_xml import HalfCheetahModel
from envs.half_cheetah.half_cheetah_config import HalfCheetahConfig
from testing import assert_true, TestSuite


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64
comptime NQ = HalfCheetahModel.NQ  # 9
comptime NV = HalfCheetahModel.NV  # 9
comptime NBODY = HalfCheetahModel.NBODY
comptime NJOINT = HalfCheetahModel.NJOINT
comptime NGEOM = HalfCheetahModel.NGEOM
comptime MAX_CONTACTS = HalfCheetahConfig.MAX_CONTACTS

comptime M_SIZE = _max_one[NV * NV]()
comptime CDOF_SIZE = _max_one[NV * 6]()
comptime CRB_SIZE = _max_one[NBODY * 10]()
comptime V_SIZE = _max_one[NV]()

# Safe upper bound for NM: full lower triangle of an NV×NV matrix.
# For a kinematic chain like HalfCheetah this equals the actual non-zeros.
comptime NM = NV * (NV + 1) / 2  # = 45 for NV=9

comptime TOL: Float64 = 1e-12


# =============================================================================
# Test: sparse pattern and count_sparse_nnz
# =============================================================================


fn check_sparse_pattern(
    model: Model[
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
    ]
) raises:
    """Verify: actual_nnz == count_sparse_nnz, all positions in lower triangle,
    diagonal present in every row."""
    print("--- Test: Sparsity pattern ---")

    var sM = SparseMassMatrix[DTYPE, NV, NM]()
    build_sparse_pattern[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NM, NGEOM](
        model, sM
    )

    var nnz = count_sparse_nnz[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM
    ](model)

    print("  actual_nnz =", sM.actual_nnz, "  count_sparse_nnz =", nnz)

    if sM.actual_nnz != nnz:
        print("  FAIL: actual_nnz != count_sparse_nnz")
        assert_true(
            False, "Sparse pattern test failed: actual_nnz != count_sparse_nnz"
        )

    # Every row must contain its diagonal as the last element
    for i in range(NV):
        var dp = sM.diag_pos(i)
        if sM.col_ind[dp] != i:
            print(
                "  FAIL: row",
                i,
                "diagonal wrong (col_ind =",
                sM.col_ind[dp],
                ")",
            )
            assert_true(
                False,
                "Sparse pattern test failed: row "
                + String(i)
                + " diagonal wrong",
            )

    # All column indices must be in lower triangle (col <= row)
    for i in range(NV):
        var adr_i = sM.row_adr[i]
        for t in range(sM.row_nnz[i]):
            if sM.col_ind[adr_i + t] > i:
                print(
                    "  FAIL: row",
                    i,
                    "has col_ind",
                    sM.col_ind[adr_i + t],
                    "> i",
                )
                assert_true(
                    False,
                    "Sparse pattern test failed: row "
                    + String(i)
                    + " has col_ind > i (not lower triangle)",
                )

    print("  PASS  actual_nnz =", sM.actual_nnz)


# =============================================================================
# Test: sparse values == dense values
# =============================================================================


fn check_sparse_values_match_dense(
    model: Model[
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
    ],
    test_name: String,
    qpos_vals: InlineArray[Float64, NQ],
) raises:
    """For given qpos: compute sparse M, expand to dense, compare with
    compute_mass_matrix_full (reference)."""
    print("--- Test: Sparse == Dense values  [", test_name, "] ---")

    var data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE
    ]()
    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_vals[i])
    forward_kinematics(model, data)

    var cdof = List[Scalar[DTYPE]](capacity=CDOF_SIZE)
    for _ in range(CDOF_SIZE):
        cdof.append(Scalar[DTYPE](0))
    compute_cdof(model, data, cdof)
    var crb = List[Scalar[DTYPE]](capacity=CRB_SIZE)
    for _ in range(CRB_SIZE):
        crb.append(Scalar[DTYPE](0))
    compute_composite_inertia(model, data, crb)

    # Dense reference
    var M_dense = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        M_dense.append(Scalar[DTYPE](0))
    compute_mass_matrix_full(model, data, cdof, crb, M_dense)

    # Sparse
    var sM = SparseMassMatrix[DTYPE, NV, NM]()
    build_sparse_pattern[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NM, NGEOM](
        model, sM
    )
    compute_mass_matrix_sparse[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NM,
        CDOF_SIZE,
        CRB_SIZE,
        NGEOM,
    ](model, data, cdof, crb, sM)

    # Expand sparse to dense
    var M_from_sparse = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        M_from_sparse.append(Scalar[DTYPE](0))
    sparse_to_dense[DTYPE, NV, NM](sM, M_from_sparse)

    # Compare lower triangle (upper is symmetric copy)
    var max_err: Float64 = 0.0
    var fail_count = 0
    for i in range(NV):
        for j in range(i + 1):
            var d = Float64(M_dense[i * NV + j])
            var s = Float64(M_from_sparse[i * NV + j])
            var err = abs(d - s)
            if err > max_err:
                max_err = err
            if err > TOL:
                fail_count += 1
                if fail_count <= 5:
                    print(
                        "  FAIL M[",
                        i,
                        ",",
                        j,
                        "]  dense =",
                        d,
                        "  sparse =",
                        s,
                        "  err =",
                        err,
                    )

    if fail_count == 0:
        print("  PASS  max_err =", max_err)
    else:
        print("  FAIL", fail_count, "entries  max_err =", max_err)
        assert_true(
            False,
            "Sparse values match dense test failed ["
            + test_name
            + "]: "
            + String(fail_count)
            + " entries exceed tolerance, max_err = "
            + String(max_err),
        )


# =============================================================================
# Test: sparse LDL solve == dense LDL solve
# =============================================================================


fn check_sparse_solve_matches_dense(
    model: Model[
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
    ],
    test_name: String,
    qpos_vals: InlineArray[Float64, NQ],
) raises:
    """Verify ldl_solve_sparse(b) == ldl_solve(b) for multiple rhs vectors."""
    print(
        "--- Test: Sparse LDL solve == Dense LDL solve  [", test_name, "] ---"
    )

    var data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE
    ]()
    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_vals[i])
    forward_kinematics(model, data)

    var cdof = List[Scalar[DTYPE]](capacity=CDOF_SIZE)
    for _ in range(CDOF_SIZE):
        cdof.append(Scalar[DTYPE](0))
    compute_cdof(model, data, cdof)
    var crb = List[Scalar[DTYPE]](capacity=CRB_SIZE)
    for _ in range(CRB_SIZE):
        crb.append(Scalar[DTYPE](0))
    compute_composite_inertia(model, data, crb)

    # Build dense M with armature
    var M = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        M.append(Scalar[DTYPE](0))
    compute_mass_matrix_full(model, data, cdof, crb, M)
    for j in range(model.num_joints):
        var dof = model.joints[j].dof_adr
        M[dof * NV + dof] += model.joints[j].armature

    # Dense LDL
    var L = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        L.append(Scalar[DTYPE](0))
    var D = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        D.append(Scalar[DTYPE](0))
    ldl_factor[DTYPE, NV](M, L, D)

    # Build sparse M with armature
    var sM = SparseMassMatrix[DTYPE, NV, NM]()
    build_sparse_pattern[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NM, NGEOM](
        model, sM
    )
    compute_mass_matrix_sparse[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NM,
        CDOF_SIZE,
        CRB_SIZE,
        NGEOM,
    ](model, data, cdof, crb, sM)
    for j in range(model.num_joints):
        var dof = model.joints[j].dof_adr
        sM.values[sM.diag_pos(dof)] += model.joints[j].armature
    ldl_factor_sparse[DTYPE, NV, NM](sM)

    # Test with 3 rhs vectors
    var max_err: Float64 = 0.0
    var fail_count = 0

    for rhs_case in range(3):
        var b = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            b.append(Scalar[DTYPE](0))
        for i in range(NV):
            if rhs_case == 0:
                b[i] = Scalar[DTYPE](Float64(i + 1))
            elif rhs_case == 1:
                b[i] = Scalar[DTYPE](Float64(i * i) - Scalar[DTYPE](NV))
            else:
                b[i] = Scalar[DTYPE](0)
        if rhs_case == 2:
            b[NV // 2] = Scalar[DTYPE](1)  # unit vector

        var x_dense = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            x_dense.append(Scalar[DTYPE](0))
        var x_sparse = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            x_sparse.append(Scalar[DTYPE](0))

        ldl_solve[DTYPE, NV](L, D, b, x_dense)
        ldl_solve_sparse[DTYPE, NV, NM](sM, b, x_sparse)

        for i in range(NV):
            var err = abs(Float64(x_dense[i]) - Float64(x_sparse[i]))
            if err > max_err:
                max_err = err
            if err > 1e-9:
                fail_count += 1
                if fail_count <= 5:
                    print(
                        "  FAIL rhs",
                        rhs_case,
                        "x[",
                        i,
                        "]",
                        "  dense =",
                        Float64(x_dense[i]),
                        "  sparse =",
                        Float64(x_sparse[i]),
                        "  err =",
                        err,
                    )

    if fail_count == 0:
        print("  PASS  max_err =", max_err)
    else:
        print("  FAIL", fail_count, "entries  max_err =", max_err)
        assert_true(
            False,
            "Sparse LDL solve matches dense test failed ["
            + test_name
            + "]: "
            + String(fail_count)
            + " entries exceed tolerance, max_err = "
            + String(max_err),
        )


# =============================================================================
# Test: residual ||M * x_sparse - b|| is small
# =============================================================================


fn check_sparse_solve_residual(
    model: Model[
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
    ],
    test_name: String,
    qpos_vals: InlineArray[Float64, NQ],
) raises:
    """Verify M * ldl_solve_sparse(b) ≈ b (small residual)."""
    print("--- Test: Sparse solve residual  [", test_name, "] ---")

    var data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE
    ]()
    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_vals[i])
    forward_kinematics(model, data)

    var cdof = List[Scalar[DTYPE]](capacity=CDOF_SIZE)
    for _ in range(CDOF_SIZE):
        cdof.append(Scalar[DTYPE](0))
    compute_cdof(model, data, cdof)
    var crb = List[Scalar[DTYPE]](capacity=CRB_SIZE)
    for _ in range(CRB_SIZE):
        crb.append(Scalar[DTYPE](0))
    compute_composite_inertia(model, data, crb)

    var sM = SparseMassMatrix[DTYPE, NV, NM]()
    build_sparse_pattern[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NM, NGEOM](
        model, sM
    )
    compute_mass_matrix_sparse[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NM,
        CDOF_SIZE,
        CRB_SIZE,
        NGEOM,
    ](model, data, cdof, crb, sM)
    for j in range(model.num_joints):
        var dof = model.joints[j].dof_adr
        sM.values[sM.diag_pos(dof)] += model.joints[j].armature

    # Save M before factorization (expand to dense for residual check)
    var M_dense = List[Scalar[DTYPE]](capacity=M_SIZE)
    for _ in range(M_SIZE):
        M_dense.append(Scalar[DTYPE](0))
    sparse_to_dense[DTYPE, NV, NM](sM, M_dense)

    ldl_factor_sparse[DTYPE, NV, NM](sM)

    var b = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        b.append(Scalar[DTYPE](0))
    for i in range(NV):
        b[i] = Scalar[DTYPE](Float64(i + 1))

    var x = List[Scalar[DTYPE]](capacity=V_SIZE)
    for _ in range(V_SIZE):
        x.append(Scalar[DTYPE](0))
    ldl_solve_sparse[DTYPE, NV, NM](sM, b, x)

    # Compute residual r = M * x - b
    var max_res: Float64 = 0.0
    for i in range(NV):
        var Mx_i = Scalar[DTYPE](0)
        for j in range(NV):
            Mx_i += M_dense[i * NV + j] * x[j]
        var res = abs(Float64(Mx_i - b[i]))
        if res > max_res:
            max_res = res

    if max_res < 1e-9:
        print("  PASS  max_residual =", max_res)
    else:
        print("  FAIL  max_residual =", max_res)
        assert_true(
            False,
            "Sparse solve residual test failed ["
            + test_name
            + "]: max_residual = "
            + String(max_res)
            + " exceeds 1e-9",
        )


# =============================================================================
# Top-level test functions (called by test framework)
# =============================================================================


fn test_sparse_mass_matrix_all() raises:
    """Run all sparse mass matrix tests."""
    print("=" * 60)
    print("Sparse Mass Matrix Validation (CSR vs Dense)")
    print("Model: HalfCheetah  NV =", NV, "  NM =", NM)
    print("=" * 60)
    print()

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
    var _setup_data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE
    ]()
    HalfCheetahModel.setup_model_and_data[DTYPE](model, _setup_data)

    # --- Pattern test ---
    check_sparse_pattern(model)
    print()

    # --- Sparse values == dense values ---
    var qpos_default = InlineArray[Float64, NQ](fill=0.0)
    qpos_default[1] = 0.7

    check_sparse_values_match_dense(model, "default qpos", qpos_default)
    print()

    var qpos_joints = InlineArray[Float64, NQ](fill=0.0)
    qpos_joints[0] = 1.0
    qpos_joints[1] = 0.7
    qpos_joints[2] = 0.3
    qpos_joints[3] = -0.4
    qpos_joints[4] = 0.5
    qpos_joints[5] = -0.2
    qpos_joints[6] = 0.6
    qpos_joints[7] = -0.8
    qpos_joints[8] = 0.3

    check_sparse_values_match_dense(model, "non-zero joints", qpos_joints)
    print()

    # --- Sparse LDL solve == Dense LDL solve ---
    check_sparse_solve_matches_dense(model, "default qpos", qpos_default)
    print()

    check_sparse_solve_matches_dense(model, "non-zero joints", qpos_joints)
    print()

    # --- Residual test ---
    check_sparse_solve_residual(model, "default qpos", qpos_default)
    print()

    check_sparse_solve_residual(model, "non-zero joints", qpos_joints)
    print()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

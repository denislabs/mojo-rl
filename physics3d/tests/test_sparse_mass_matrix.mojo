"""Test sparse vs dense mass matrix for correctness.

Verifies that SparseMassMatrix (CSR format, MuJoCo-compatible) produces
identical results to compute_mass_matrix_full (dense) for the HalfCheetah
model. Also verifies that ldl_solve_sparse gives the same solution as
ldl_solve (dense).

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_sparse_mass_matrix.mojo
"""

from math import abs
from collections import InlineArray

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


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64
comptime NQ = HalfCheetahModel.NQ   # 9
comptime NV = HalfCheetahModel.NV   # 9
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


fn test_sparse_pattern(
    model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, HalfCheetahModel.MAX_EQUALITY, HalfCheetahModel.CONE_TYPE, HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE]
) -> Bool:
    """Verify: actual_nnz == count_sparse_nnz, all positions in lower triangle,
    diagonal present in every row."""
    print("--- Test: Sparsity pattern ---")

    var sM = SparseMassMatrix[DTYPE, NV, NM]()
    build_sparse_pattern[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NM, NGEOM
    ](model, sM)

    var nnz = count_sparse_nnz[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM
    ](model)

    print("  actual_nnz =", sM.actual_nnz, "  count_sparse_nnz =", nnz)

    var ok = True

    if sM.actual_nnz != nnz:
        print("  FAIL: actual_nnz != count_sparse_nnz")
        ok = False

    # Every row must contain its diagonal as the last element
    for i in range(NV):
        var dp = sM.diag_pos(i)
        if sM.col_ind[dp] != i:
            print("  FAIL: row", i, "diagonal wrong (col_ind =", sM.col_ind[dp], ")")
            ok = False

    # All column indices must be in lower triangle (col <= row)
    for i in range(NV):
        var adr_i = sM.row_adr[i]
        for t in range(sM.row_nnz[i]):
            if sM.col_ind[adr_i + t] > i:
                print("  FAIL: row", i, "has col_ind", sM.col_ind[adr_i + t], "> i")
                ok = False

    if ok:
        print("  PASS  actual_nnz =", sM.actual_nnz)
    return ok


# =============================================================================
# Test: sparse values == dense values
# =============================================================================


fn test_sparse_values_match_dense(
    model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, HalfCheetahModel.MAX_EQUALITY, HalfCheetahModel.CONE_TYPE, HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE],
    test_name: String,
    qpos_vals: InlineArray[Float64, NQ],
) -> Bool:
    """For given qpos: compute sparse M, expand to dense, compare with
    compute_mass_matrix_full (reference)."""
    print("--- Test: Sparse == Dense values  [", test_name, "] ---")

    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_vals[i])
    forward_kinematics(model, data)

    var cdof = InlineArray[Scalar[DTYPE], CDOF_SIZE](uninitialized=True)
    compute_cdof[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CDOF_SIZE](
        model, data, cdof
    )
    var crb = InlineArray[Scalar[DTYPE], CRB_SIZE](fill=Scalar[DTYPE](0))
    compute_composite_inertia[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CRB_SIZE
    ](model, data, crb)

    # Dense reference
    var M_dense = InlineArray[Scalar[DTYPE], M_SIZE](fill=Scalar[DTYPE](0))
    compute_mass_matrix_full[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        M_SIZE, CDOF_SIZE, CRB_SIZE, NGEOM,
    ](model, data, cdof, crb, M_dense)

    # Sparse
    var sM = SparseMassMatrix[DTYPE, NV, NM]()
    build_sparse_pattern[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NM, NGEOM
    ](model, sM)
    compute_mass_matrix_sparse[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        NM, CDOF_SIZE, CRB_SIZE, NGEOM,
    ](model, data, cdof, crb, sM)

    # Expand sparse to dense
    var M_from_sparse = InlineArray[Scalar[DTYPE], M_SIZE](fill=Scalar[DTYPE](0))
    sparse_to_dense[DTYPE, NV, NM, M_SIZE](sM, M_from_sparse)

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
                        "  FAIL M[", i, ",", j, "]  dense =", d,
                        "  sparse =", s, "  err =", err,
                    )

    var ok = fail_count == 0
    if ok:
        print("  PASS  max_err =", max_err)
    else:
        print("  FAIL", fail_count, "entries  max_err =", max_err)
    return ok


# =============================================================================
# Test: sparse LDL solve == dense LDL solve
# =============================================================================


fn test_sparse_solve_matches_dense(
    model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, HalfCheetahModel.MAX_EQUALITY, HalfCheetahModel.CONE_TYPE, HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE],
    test_name: String,
    qpos_vals: InlineArray[Float64, NQ],
) -> Bool:
    """Verify ldl_solve_sparse(b) == ldl_solve(b) for multiple rhs vectors."""
    print("--- Test: Sparse LDL solve == Dense LDL solve  [", test_name, "] ---")

    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_vals[i])
    forward_kinematics(model, data)

    var cdof = InlineArray[Scalar[DTYPE], CDOF_SIZE](uninitialized=True)
    compute_cdof[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CDOF_SIZE](
        model, data, cdof
    )
    var crb = InlineArray[Scalar[DTYPE], CRB_SIZE](fill=Scalar[DTYPE](0))
    compute_composite_inertia[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CRB_SIZE
    ](model, data, crb)

    # Build dense M with armature
    var M = InlineArray[Scalar[DTYPE], M_SIZE](fill=Scalar[DTYPE](0))
    compute_mass_matrix_full[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        M_SIZE, CDOF_SIZE, CRB_SIZE, NGEOM,
    ](model, data, cdof, crb, M)
    for j in range(model.num_joints):
        var dof = model.joints[j].dof_adr
        M[dof * NV + dof] += model.joints[j].armature

    # Dense LDL
    var L = InlineArray[Scalar[DTYPE], M_SIZE](fill=Scalar[DTYPE](0))
    var D = InlineArray[Scalar[DTYPE], V_SIZE](fill=Scalar[DTYPE](0))
    ldl_factor[DTYPE, NV, M_SIZE, V_SIZE](M, L, D)

    # Build sparse M with armature
    var sM = SparseMassMatrix[DTYPE, NV, NM]()
    build_sparse_pattern[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NM, NGEOM
    ](model, sM)
    compute_mass_matrix_sparse[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        NM, CDOF_SIZE, CRB_SIZE, NGEOM,
    ](model, data, cdof, crb, sM)
    for j in range(model.num_joints):
        var dof = model.joints[j].dof_adr
        sM.values[sM.diag_pos(dof)] += model.joints[j].armature
    ldl_factor_sparse[DTYPE, NV, NM](sM)

    # Test with 3 rhs vectors
    var max_err: Float64 = 0.0
    var fail_count = 0

    for rhs_case in range(3):
        var b = InlineArray[Scalar[DTYPE], V_SIZE](fill=Scalar[DTYPE](0))
        for i in range(NV):
            if rhs_case == 0:
                b[i] = Scalar[DTYPE](Float64(i + 1))
            elif rhs_case == 1:
                b[i] = Scalar[DTYPE](Float64(i * i) - Scalar[DTYPE](NV))
            else:
                b[i] = Scalar[DTYPE](0)
        if rhs_case == 2:
            b[NV // 2] = Scalar[DTYPE](1)  # unit vector

        var x_dense = InlineArray[Scalar[DTYPE], V_SIZE](fill=Scalar[DTYPE](0))
        var x_sparse = InlineArray[Scalar[DTYPE], V_SIZE](fill=Scalar[DTYPE](0))

        ldl_solve[DTYPE, NV, M_SIZE, V_SIZE](L, D, b, x_dense)
        ldl_solve_sparse[DTYPE, NV, NM, V_SIZE](sM, b, x_sparse)

        for i in range(NV):
            var err = abs(Float64(x_dense[i]) - Float64(x_sparse[i]))
            if err > max_err:
                max_err = err
            if err > 1e-9:
                fail_count += 1
                if fail_count <= 5:
                    print(
                        "  FAIL rhs", rhs_case, "x[", i, "]",
                        "  dense =", Float64(x_dense[i]),
                        "  sparse =", Float64(x_sparse[i]),
                        "  err =", err,
                    )

    var ok = fail_count == 0
    if ok:
        print("  PASS  max_err =", max_err)
    else:
        print("  FAIL", fail_count, "entries  max_err =", max_err)
    return ok


# =============================================================================
# Test: residual ||M * x_sparse - b|| is small
# =============================================================================


fn test_sparse_solve_residual(
    model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, HalfCheetahModel.MAX_EQUALITY, HalfCheetahModel.CONE_TYPE, HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE],
    test_name: String,
    qpos_vals: InlineArray[Float64, NQ],
) -> Bool:
    """Verify M * ldl_solve_sparse(b) ≈ b (small residual)."""
    print("--- Test: Sparse solve residual  [", test_name, "] ---")

    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](qpos_vals[i])
    forward_kinematics(model, data)

    var cdof = InlineArray[Scalar[DTYPE], CDOF_SIZE](uninitialized=True)
    compute_cdof[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CDOF_SIZE](
        model, data, cdof
    )
    var crb = InlineArray[Scalar[DTYPE], CRB_SIZE](fill=Scalar[DTYPE](0))
    compute_composite_inertia[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CRB_SIZE
    ](model, data, crb)

    var sM = SparseMassMatrix[DTYPE, NV, NM]()
    build_sparse_pattern[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NM, NGEOM
    ](model, sM)
    compute_mass_matrix_sparse[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        NM, CDOF_SIZE, CRB_SIZE, NGEOM,
    ](model, data, cdof, crb, sM)
    for j in range(model.num_joints):
        var dof = model.joints[j].dof_adr
        sM.values[sM.diag_pos(dof)] += model.joints[j].armature

    # Save M before factorization (expand to dense for residual check)
    var M_dense = InlineArray[Scalar[DTYPE], M_SIZE](fill=Scalar[DTYPE](0))
    sparse_to_dense[DTYPE, NV, NM, M_SIZE](sM, M_dense)

    ldl_factor_sparse[DTYPE, NV, NM](sM)

    var b = InlineArray[Scalar[DTYPE], V_SIZE](fill=Scalar[DTYPE](0))
    for i in range(NV):
        b[i] = Scalar[DTYPE](Float64(i + 1))

    var x = InlineArray[Scalar[DTYPE], V_SIZE](fill=Scalar[DTYPE](0))
    ldl_solve_sparse[DTYPE, NV, NM, V_SIZE](sM, b, x)

    # Compute residual r = M * x - b
    var max_res: Float64 = 0.0
    for i in range(NV):
        var Mx_i = Scalar[DTYPE](0)
        for j in range(NV):
            Mx_i += M_dense[i * NV + j] * x[j]
        var res = abs(Float64(Mx_i - b[i]))
        if res > max_res:
            max_res = res

    var ok = max_res < 1e-9
    if ok:
        print("  PASS  max_residual =", max_res)
    else:
        print("  FAIL  max_residual =", max_res)
    return ok


# =============================================================================
# Main
# =============================================================================


fn main():
    print("=" * 60)
    print("Sparse Mass Matrix Validation (CSR vs Dense)")
    print("Model: HalfCheetah  NV =", NV, "  NM =", NM)
    print("=" * 60)
    print()

    var model = Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, HalfCheetahModel.MAX_EQUALITY, HalfCheetahModel.CONE_TYPE, HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE]()
    var _setup_data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE]()
    HalfCheetahModel.setup_model_and_data[DTYPE](model, _setup_data)

    var num_pass = 0
    var num_fail = 0

    # --- Pattern test ---
    if test_sparse_pattern(model):
        num_pass += 1
    else:
        num_fail += 1
    print()

    # --- Sparse values == dense values ---
    var qpos_default = InlineArray[Float64, NQ](fill=0.0)
    qpos_default[1] = 0.7

    if test_sparse_values_match_dense(model, "default qpos", qpos_default):
        num_pass += 1
    else:
        num_fail += 1
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

    if test_sparse_values_match_dense(model, "non-zero joints", qpos_joints):
        num_pass += 1
    else:
        num_fail += 1
    print()

    # --- Sparse LDL solve == Dense LDL solve ---
    if test_sparse_solve_matches_dense(model, "default qpos", qpos_default):
        num_pass += 1
    else:
        num_fail += 1
    print()

    if test_sparse_solve_matches_dense(model, "non-zero joints", qpos_joints):
        num_pass += 1
    else:
        num_fail += 1
    print()

    # --- Residual test ---
    if test_sparse_solve_residual(model, "default qpos", qpos_default):
        num_pass += 1
    else:
        num_fail += 1
    print()

    if test_sparse_solve_residual(model, "non-zero joints", qpos_joints):
        num_pass += 1
    else:
        num_fail += 1
    print()

    print("=" * 60)
    print(
        "Results:",
        num_pass, "passed,",
        num_fail, "failed out of",
        num_pass + num_fail,
    )
    if num_fail == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

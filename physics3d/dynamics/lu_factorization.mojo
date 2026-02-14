"""Dense LU factorization with partial pivoting for small NV matrices.

Used by ImplicitIntegrator where M_hat = M - dt*qDeriv is non-symmetric
(due to the RNE velocity derivative), so LDL factorization cannot be used.

For the typical NV sizes in robotics (NV <= 30), dense LU is trivially fast.
"""

from ..types import _max_one


fn lu_factor[
    DTYPE: DType,
    NV: Int,
    M_SIZE: Int,
    V_SIZE: Int,
](
    mut A: InlineArray[Scalar[DTYPE], M_SIZE],
    mut piv: InlineArray[Int, V_SIZE],
):
    """In-place LU factorization with partial pivoting of NV×NV matrix A.

    Computes P*A = L*U where:
    - L is unit lower triangular (stored below diagonal of A)
    - U is upper triangular (stored on and above diagonal of A)
    - P is a permutation (stored in piv)

    Args:
        A: Input/output NV×NV matrix (row-major). On exit, contains L and U.
        piv: Output pivot indices. piv[i] = row that was swapped with row i.
    """
    # Initialize pivots to identity
    for i in range(NV):
        piv[i] = i

    for k in range(NV):
        # Find pivot: row with largest absolute value in column k, rows k..NV-1
        var max_val = abs(A[k * NV + k])
        var max_row = k
        for i in range(k + 1, NV):
            var val = abs(A[i * NV + k])
            if val > max_val:
                max_val = val
                max_row = i

        # Swap rows k and max_row in A and piv
        if max_row != k:
            piv[k] = max_row
            for j in range(NV):
                var tmp = A[k * NV + j]
                A[k * NV + j] = A[max_row * NV + j]
                A[max_row * NV + j] = tmp

        # Check for near-zero pivot (regularize if needed)
        var pivot = A[k * NV + k]
        if abs(pivot) < Scalar[DTYPE](1e-30):
            A[k * NV + k] = Scalar[DTYPE](1e-30)
            pivot = A[k * NV + k]

        # Compute multipliers and update trailing submatrix
        var inv_pivot = Scalar[DTYPE](1) / pivot
        for i in range(k + 1, NV):
            A[i * NV + k] = A[i * NV + k] * inv_pivot  # L[i,k]
            var lik = A[i * NV + k]
            for j in range(k + 1, NV):
                A[i * NV + j] = A[i * NV + j] - lik * A[k * NV + j]


fn lu_solve[
    DTYPE: DType,
    NV: Int,
    M_SIZE: Int,
    V_SIZE: Int,
](
    A: InlineArray[Scalar[DTYPE], M_SIZE],
    piv: InlineArray[Int, V_SIZE],
    b: InlineArray[Scalar[DTYPE], V_SIZE],
    mut x: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """Solve A * x = b using precomputed LU factors with pivoting.

    Solves P*A*x = P*b → L*U*x = P*b in three steps:
    1. Apply permutation: Pb = P * b
    2. Forward substitution: L * y = Pb
    3. Backward substitution: U * x = y

    Args:
        A: LU factors from lu_factor (NV×NV, row-major).
        piv: Pivot indices from lu_factor.
        b: Right-hand side vector (NV elements).
        x: Output solution vector (NV elements).
    """
    # Apply permutation and forward substitution: L * y = P * b
    # Combined: apply swaps as we go
    var y = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        y[i] = b[i]

    # Apply permutation
    for i in range(NV):
        if piv[i] != i:
            var tmp = y[i]
            y[i] = y[piv[i]]
            y[piv[i]] = tmp

    # Forward substitution: L * y = Pb (L has unit diagonal)
    for i in range(NV):
        for j in range(i):
            y[i] = y[i] - A[i * NV + j] * y[j]

    # Backward substitution: U * x = y
    for i in range(NV - 1, -1, -1):
        var s = y[i]
        for j in range(i + 1, NV):
            s = s - A[i * NV + j] * x[j]
        x[i] = s / A[i * NV + i]


fn compute_M_inv_from_lu[
    DTYPE: DType,
    NV: Int,
    M_SIZE: Int,
    V_SIZE: Int,
](
    A: InlineArray[Scalar[DTYPE], M_SIZE],
    piv: InlineArray[Int, V_SIZE],
    mut M_inv: InlineArray[Scalar[DTYPE], M_SIZE],
):
    """Compute M^-1 from LU factors by solving M * M_inv[:,j] = e_j.

    Args:
        A: LU factors from lu_factor (NV×NV, row-major).
        piv: Pivot indices from lu_factor.
        M_inv: Output inverse matrix (NV×NV, row-major).
    """
    for i in range(M_SIZE):
        M_inv[i] = Scalar[DTYPE](0)

    var e = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var col = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

    for j in range(NV):
        # Set up unit vector e_j
        for i in range(NV):
            e[i] = Scalar[DTYPE](0)
        e[j] = Scalar[DTYPE](1)

        # Solve A * col = e_j
        lu_solve[DTYPE, NV, M_SIZE, V_SIZE](A, piv, e, col)

        # Store column j of M_inv
        for i in range(NV):
            M_inv[i * NV + j] = col[i]

"""Dense Cholesky utilities for small NV×NV matrices.

Used by primal Newton solver for Hessian factorization and solve.
These operate on InlineArrays for register-friendly small matrices
(NV is typically 6-30 for robotics models).

Functions:
- chol_factor: In-place Cholesky L*L^T = H (lower triangular) [CPU, uses List]
- chol_solve: Solve H*x = b given Cholesky factor L [CPU, uses List]
- chol_factor_inline: Same as chol_factor but uses InlineArray [GPU-compatible]
- chol_solve_inline: Same as chol_solve but uses InlineArray [GPU-compatible]
- chol_rank1_update: Rank-1 update H ← H ± v*v^T with Cholesky factor update
"""

from std.math import sqrt
from ..types import _max_one


@always_inline
def chol_factor[
    DTYPE: DType,
    NV: Int,
    M_SIZE: Int,
](H: List[Scalar[DTYPE]], mut L: List[Scalar[DTYPE]],):
    """In-place Cholesky factorization: L*L^T = H (lower triangular).

    H must be symmetric positive definite. L is output lower triangular.
    Both are NV×NV row-major in M_SIZE arrays.
    """
    # Zero L
    for i in range(NV * NV):
        L[i] = Scalar[DTYPE](0)

    for i in range(NV):
        for j in range(i + 1):
            var s: Scalar[DTYPE] = 0
            for k in range(j):
                s += L[i * NV + k] * L[j * NV + k]
            if i == j:
                var diag = H[i * NV + i] - s
                if diag < Scalar[DTYPE](1e-14):
                    diag = Scalar[DTYPE](1e-14)
                L[i * NV + j] = sqrt(diag)
            else:
                L[i * NV + j] = (H[i * NV + j] - s) / L[j * NV + j]


@always_inline
def chol_solve[
    DTYPE: DType,
    NV: Int,
    M_SIZE: Int,
    V_SIZE: Int,
](L: List[Scalar[DTYPE]], b: List[Scalar[DTYPE]], mut x: List[Scalar[DTYPE]],):
    """Solve H*x = b given Cholesky factor L (where H = L*L^T).

    Two-phase: forward substitution L*y = b, then back substitution L^T*x = y.
    """
    # Forward substitution: L*y = b
    var y = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        var s: Scalar[DTYPE] = 0
        for j in range(i):
            s += L[i * NV + j] * y[j]
        y[i] = (b[i] - s) / L[i * NV + i]

    # Back substitution: L^T*x = y
    for i_rev in range(NV):
        var i = NV - 1 - i_rev
        var s: Scalar[DTYPE] = 0
        for j in range(i + 1, NV):
            s += L[j * NV + i] * x[j]
        x[i] = (y[i] - s) / L[i * NV + i]


@always_inline
def chol_factor_inline[
    DTYPE: DType,
    NV: Int,
    M_SIZE: Int,
](
    H: InlineArray[Scalar[DTYPE], M_SIZE],
    mut L: InlineArray[Scalar[DTYPE], M_SIZE],
):
    """In-place Cholesky factorization: L*L^T = H (lower triangular), GPU-compatible.

    Same algorithm as chol_factor but operates on InlineArrays so it can be
    used inside @always_inline GPU kernels without heap allocation.
    H must be symmetric positive definite. L is output lower triangular.
    Both are NV×NV row-major stored in M_SIZE (= max(1, NV*NV)) arrays.
    """
    for i in range(NV * NV):
        L[i] = Scalar[DTYPE](0)

    for i in range(NV):
        for j in range(i + 1):
            var s: Scalar[DTYPE] = 0
            for k in range(j):
                s += L[i * NV + k] * L[j * NV + k]
            if i == j:
                var diag = H[i * NV + i] - s
                if diag < Scalar[DTYPE](1e-14):
                    diag = Scalar[DTYPE](1e-14)
                L[i * NV + j] = sqrt(diag)
            else:
                L[i * NV + j] = (H[i * NV + j] - s) / L[j * NV + j]


@always_inline
def chol_solve_inline[
    DTYPE: DType,
    NV: Int,
    M_SIZE: Int,
    V_SIZE: Int,
](
    L: InlineArray[Scalar[DTYPE], M_SIZE],
    b: InlineArray[Scalar[DTYPE], V_SIZE],
    mut x: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """Solve H*x = b given Cholesky factor L (where H = L*L^T), GPU-compatible.

    Same algorithm as chol_solve but operates on InlineArrays so it can be
    used inside @always_inline GPU kernels without heap allocation.
    L is NV×NV in M_SIZE array, b/x are NV in V_SIZE arrays.
    Two-phase: forward substitution L*y = b, then back substitution L^T*x = y.
    """
    # Forward substitution: L*y = b
    var y = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        var s: Scalar[DTYPE] = 0
        for j in range(i):
            s += L[i * NV + j] * y[j]
        y[i] = (b[i] - s) / L[i * NV + i]

    # Back substitution: L^T*x = y
    for i_rev in range(NV):
        var i = NV - 1 - i_rev
        var s: Scalar[DTYPE] = 0
        for j in range(i + 1, NV):
            s += L[j * NV + i] * x[j]
        x[i] = (y[i] - s) / L[i * NV + i]


@always_inline
def chol_rank1_update[
    DTYPE: DType,
    NV: Int,
    M_SIZE: Int,
    V_SIZE: Int,
](
    mut L: InlineArray[Scalar[DTYPE], M_SIZE],
    v: InlineArray[Scalar[DTYPE], V_SIZE],
    sign: Scalar[DTYPE],
):
    """Rank-1 Cholesky update: H ← H + sign * v * v^T.

    sign = +1 for update (adding), sign = -1 for downdate (removing).
    Modifies L in-place. Uses the standard rank-1 Cholesky update algorithm.

    For downdate (sign=-1), the result may not be PD if v is too large.
    In that case, diagonal elements are clamped to a small positive value.
    """
    # Work on a copy of v that gets modified
    var w = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        w[i] = v[i]

    for i in range(NV):
        var L_ii = L[i * NV + i]
        var w_i = w[i]

        var r_sq = L_ii * L_ii + sign * w_i * w_i
        if r_sq < Scalar[DTYPE](1e-14):
            r_sq = Scalar[DTYPE](1e-14)
        var r = sqrt(r_sq)

        var c = r / L_ii
        var s_val = w_i / L_ii

        L[i * NV + i] = r

        # Update remaining elements in column i
        for j in range(i + 1, NV):
            L[j * NV + i] = (L[j * NV + i] + sign * s_val * w[j]) / c
            w[j] = c * w[j] - s_val * L[j * NV + i]

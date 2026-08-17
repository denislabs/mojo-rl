"""Dense Cholesky utilities for small NV×NV matrices.

Used by primal Newton solver for Hessian factorization and solve.
These operate on `Scratch` for register-friendly small matrices
(NV is typically 6-30 for robotics models).

Functions:
- chol_factor: In-place Cholesky L*L^T = H (lower triangular) [CPU, uses List]
- chol_solve: Solve H*x = b given Cholesky factor L [CPU, uses List]
- chol_factor_inline: Same as chol_factor but uses Scratch [GPU-compatible]
- chol_solve_inline: Same as chol_solve but uses Scratch [GPU-compatible]
- chol_rank1_update: Rank-1 update H ← H ± v*v^T with Cholesky factor update

## 2b.2: `nv` is a RUNTIME argument here, and that is the whole point

Every `NV` below was a comptime parameter, and every call site bound it to
`D.CAP_NV`. Inside this file NV is used only two ways — as a loop bound and as
the row stride of `L[i * NV + j]` — and it is a *cap* at the call site. On the
static leg `CAP_NV == NV`, so the two agree and no gate that runs today can
tell them apart; on a dynamic provider the cap is not the model's NV and every
one of these routines would factor the wrong matrix, silently, because every
offset it produces still lands inside the array.

So NV became the runtime `nv` and the comptime parameters that remain
(`M_CAP`, `V_CAP`) size containers and nothing else. See
`fields/scratch.mojo` for why a cap is 0 rather than -1 on the dynamic leg.
"""

from std.math import sqrt
from ..fields.scratch import Scratch


@always_inline
def chol_factor[
    DTYPE: DType,
](H: List[Scalar[DTYPE]], mut L: List[Scalar[DTYPE]], nv: Int) -> Bool:
    """In-place Cholesky factorization: L*L^T = H (lower triangular).

    H must be symmetric positive definite. L is output lower triangular.
    Both are nv×nv row-major.

    Returns True if successful, False if rank-deficient (diagonal < threshold).
    When False, L still contains a usable factorization (with clamped diagonals),
    but the caller should add regularization and retry.
    """
    var rank_ok = True

    # Zero L
    for i in range(nv * nv):
        L[i] = Scalar[DTYPE](0)

    for i in range(nv):
        for j in range(i + 1):
            var s: Scalar[DTYPE] = 0
            for k in range(j):
                s += L[i * nv + k] * L[j * nv + k]
            if i == j:
                var diag = H[i * nv + i] - s
                if diag < Scalar[DTYPE](1e-10):
                    rank_ok = False
                    diag = Scalar[DTYPE](1e-10)
                L[i * nv + j] = sqrt(diag)
            else:
                L[i * nv + j] = (H[i * nv + j] - s) / L[j * nv + j]

    return rank_ok


@always_inline
def chol_solve[
    DTYPE: DType,
    V_CAP: Int,
](
    L: List[Scalar[DTYPE]],
    b: List[Scalar[DTYPE]],
    mut x: List[Scalar[DTYPE]],
    nv: Int,
):
    """Solve H*x = b given Cholesky factor L (where H = L*L^T).

    Two-phase: forward substitution L*y = b, then back substitution L^T*x = y.
    """
    # Forward substitution: L*y = b
    var y = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
    for i in range(nv):
        var s: Scalar[DTYPE] = 0
        for j in range(i):
            s += L[i * nv + j] * y[j]
        y[i] = (b[i] - s) / L[i * nv + i]

    # Back substitution: L^T*x = y
    for i_rev in range(nv):
        var i = nv - 1 - i_rev
        var s: Scalar[DTYPE] = 0
        for j in range(i + 1, nv):
            s += L[j * nv + i] * x[j]
        x[i] = (y[i] - s) / L[i * nv + i]


@always_inline
def chol_factor_inline[
    DTYPE: DType,
    M_CAP: Int,
](
    H: Scratch[Scalar[DTYPE], M_CAP],
    mut L: Scratch[Scalar[DTYPE], M_CAP],
    nv: Int,
) -> Bool:
    """In-place Cholesky factorization: L*L^T = H (lower triangular), GPU-compatible.

    Returns True if successful, False if rank-deficient.
    """
    var rank_ok = True

    for i in range(nv * nv):
        L[i] = Scalar[DTYPE](0)

    for i in range(nv):
        for j in range(i + 1):
            var s: Scalar[DTYPE] = 0
            for k in range(j):
                s += L[i * nv + k] * L[j * nv + k]
            if i == j:
                var diag = H[i * nv + i] - s
                if diag < Scalar[DTYPE](1e-10):
                    rank_ok = False
                    diag = Scalar[DTYPE](1e-10)
                L[i * nv + j] = sqrt(diag)
            else:
                L[i * nv + j] = (H[i * nv + j] - s) / L[j * nv + j]

    return rank_ok


@always_inline
def chol_solve_inline[
    DTYPE: DType,
    M_CAP: Int,
    V_CAP: Int,
](
    L: Scratch[Scalar[DTYPE], M_CAP],
    b: Scratch[Scalar[DTYPE], V_CAP],
    mut x: Scratch[Scalar[DTYPE], V_CAP],
    nv: Int,
):
    """Solve H*x = b given Cholesky factor L (where H = L*L^T), GPU-compatible.

    Same algorithm as chol_solve but operates on `Scratch` so it can be
    used inside @always_inline GPU kernels without heap allocation.
    L is nv×nv in an M_CAP array, b/x are nv in V_CAP arrays.
    Two-phase: forward substitution L*y = b, then back substitution L^T*x = y.
    """
    # Forward substitution: L*y = b
    var y = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
    for i in range(nv):
        var s: Scalar[DTYPE] = 0
        for j in range(i):
            s += L[i * nv + j] * y[j]
        y[i] = (b[i] - s) / L[i * nv + i]

    # Back substitution: L^T*x = y
    for i_rev in range(nv):
        var i = nv - 1 - i_rev
        var s: Scalar[DTYPE] = 0
        for j in range(i + 1, nv):
            s += L[j * nv + i] * x[j]
        x[i] = (y[i] - s) / L[i * nv + i]


@always_inline
def chol_rank1_update[
    DTYPE: DType,
    M_CAP: Int,
    V_CAP: Int,
](
    mut L: Scratch[Scalar[DTYPE], M_CAP],
    v: Scratch[Scalar[DTYPE], V_CAP],
    sign: Scalar[DTYPE],
    nv: Int,
):
    """Rank-1 Cholesky update: H ← H + sign * v * v^T.

    sign = +1 for update (adding), sign = -1 for downdate (removing).
    Modifies L in-place. Uses the standard rank-1 Cholesky update algorithm.

    For downdate (sign=-1), the result may not be PD if v is too large.
    In that case, diagonal elements are clamped to a small positive value.
    """
    # Work on a copy of v that gets modified
    var w = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
    for i in range(nv):
        w[i] = v[i]

    for i in range(nv):
        var L_ii = L[i * nv + i]
        var w_i = w[i]

        var r_sq = L_ii * L_ii + sign * w_i * w_i
        if r_sq < Scalar[DTYPE](1e-14):
            r_sq = Scalar[DTYPE](1e-14)
        var r = sqrt(r_sq)

        var c = r / L_ii
        var s_val = w_i / L_ii

        L[i * nv + i] = r

        # Update remaining elements in column i
        for j in range(i + 1, nv):
            L[j * nv + i] = (L[j * nv + i] + sign * s_val * w[j]) / c
            w[j] = c * w[j] - s_val * L[j * nv + i]

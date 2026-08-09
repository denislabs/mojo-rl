"""Dense LU factorization + solve over per-field tensors (migration P2 /
Stage-I, single-source). Per-field ports of `lu_factor_gpu`,
`lu_solve_workspace_gpu`, `compute_M_inv_from_lu_gpu`
(dynamics/lu_factorization.mojo) — arithmetic verbatim.

Used by the fields `ImplicitIntegrator`, where
`M_hat = M + armature - dt*qDeriv` is NON-symmetric (the dense RNE velocity
derivative) so LDL cannot be used. Reuses the same `DynamicsScratch` tensors
as the LDL path — `factor` reads `scratch.M`, writes the LU factors into
`scratch.L` and the pivot indices (as floats) into `scratch.D`; `solve`
reads `scratch.L`/`scratch.D`/`scratch.fnet`, writes `scratch.qacc_ws`.
Partial pivoting; near-zero pivots regularized to 1e-30 (verbatim legacy).

Serial per-env kernels only — LU is inherently sequential in the elimination
index k (like LDL's columns). A cooperative `_mt` schedule is a later NVIDIA
perf lever, not a correctness requirement.
"""

from std.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from ..fields import DynamicsScratch

comptime LU_TPB: Int = 64


@always_inline
def _ensure_positive[N: Int]() -> Int:
    return N if N > 0 else 1


@always_inline
def _lu_factor_env[
    DTYPE: DType,
    NV: Int,
    BATCH: Int,
](
    env: Int,
    M: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    L: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    D: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
):
    """In-place LU factorization with partial pivoting for one env (verbatim
    from `lu_factor_gpu`). Copies M into the L slot and factorizes there;
    pivot indices (as floats) land in the D slot."""
    # Copy M -> L (LU overwrites in-place in the L slot)
    for i in range(NV * NV):
        L[env, i] = M[env, i]

    # Initialize pivots to identity
    for i in range(NV):
        D[env, i] = Scalar[DTYPE](i)

    for k in range(NV):
        # Find pivot: row with largest |value| in column k, rows k..NV-1
        var max_val = abs(rebind[Scalar[DTYPE]](L[env, k * NV + k]))
        var max_row = k
        for i in range(k + 1, NV):
            var val = abs(rebind[Scalar[DTYPE]](L[env, i * NV + k]))
            if val > max_val:
                max_val = val
                max_row = i

        # Swap rows k and max_row
        if max_row != k:
            D[env, k] = Scalar[DTYPE](max_row)
            for j in range(NV):
                var tmp = L[env, k * NV + j]
                L[env, k * NV + j] = L[env, max_row * NV + j]
                L[env, max_row * NV + j] = tmp

        # Regularize near-zero pivot
        var pivot = rebind[Scalar[DTYPE]](L[env, k * NV + k])
        if abs(pivot) < Scalar[DTYPE](1e-30):
            L[env, k * NV + k] = Scalar[DTYPE](1e-30)
            pivot = Scalar[DTYPE](1e-30)

        # Compute multipliers + update trailing submatrix
        var inv_pivot = Scalar[DTYPE](1) / pivot
        for i in range(k + 1, NV):
            var lik = (
                rebind[Scalar[DTYPE]](L[env, i * NV + k]) * inv_pivot
            )
            L[env, i * NV + k] = lik
            for j in range(k + 1, NV):
                var cur = rebind[Scalar[DTYPE]](L[env, i * NV + j])
                var ukj = rebind[Scalar[DTYPE]](L[env, k * NV + j])
                L[env, i * NV + j] = cur - lik * ukj


@always_inline
def _lu_solve_env[
    DTYPE: DType,
    NV: Int,
    BATCH: Int,
](
    env: Int,
    L: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    D: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    b: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    x: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
):
    """Solve A*x = b using the LU factors + pivots for one env (verbatim from
    `lu_solve_workspace_gpu`). L holds the in-place LU, D holds the pivots."""
    comptime V_SIZE = _ensure_positive[NV]()

    # Copy b, then apply the row permutation from the pivots
    var y = InlineArray[L.element_type, V_SIZE](uninitialized=True)
    for i in range(NV):
        y[i] = b[env, i]
    for i in range(NV):
        var piv_i = Int(rebind[Scalar[DTYPE]](D[env, i]))
        if piv_i != i:
            var tmp = y[i]
            y[i] = y[piv_i]
            y[piv_i] = tmp

    # Forward substitution: L * y = Pb (L has unit diagonal)
    for i in range(NV):
        for j in range(i):
            y[i] = y[i] - L[env, i * NV + j] * y[j]

    # Backward substitution: U * x = y
    for i in range(NV - 1, -1, -1):
        var s = y[i]
        for j in range(i + 1, NV):
            s = s - L[env, i * NV + j] * x[env, j]
        var diag = rebind[Scalar[DTYPE]](L[env, i * NV + i])
        if abs(diag) > Scalar[DTYPE](1e-30):
            x[env, i] = s / diag
        else:
            x[env, i] = Scalar[DTYPE](0)


@always_inline
def _lu_m_inv_col_env[
    DTYPE: DType,
    NV: Int,
    BATCH: Int,
](
    env: Int,
    j_col: Int,
    L: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    D: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    m_inv: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
):
    """One column of M^-1 from LU factors (solve A*col = e_j). Verbatim from
    `compute_M_inv_from_lu_gpu`'s column body."""
    comptime V_SIZE = _ensure_positive[NV]()
    var e = InlineArray[L.element_type, V_SIZE](uninitialized=True)
    var col = InlineArray[L.element_type, V_SIZE](uninitialized=True)

    for i in range(NV):
        e[i] = 0
    e[j_col] = 1

    # Apply permutation
    for i in range(NV):
        var piv_i = Int(rebind[Scalar[DTYPE]](D[env, i]))
        if piv_i != i:
            var tmp = e[i]
            e[i] = e[piv_i]
            e[piv_i] = tmp

    # Forward substitution: L * y = Pe (L has unit diagonal)
    var y = InlineArray[L.element_type, V_SIZE](uninitialized=True)
    for i in range(NV):
        var s = e[i]
        for k in range(i):
            s = s - L[env, i * NV + k] * y[k]
        y[i] = s

    # Backward substitution: U * col = y
    for i in range(NV - 1, -1, -1):
        var s = y[i]
        for k in range(i + 1, NV):
            s = s - L[env, i * NV + k] * col[k]
        var diag = rebind[Scalar[DTYPE]](L[env, i * NV + i])
        if abs(diag) > Scalar[DTYPE](1e-30):
            col[i] = s / diag
        else:
            col[i] = 0

    for i in range(NV):
        m_inv[env, i * NV + j_col] = col[i]


@always_inline
def _lu_m_inv_env[
    DTYPE: DType,
    NV: Int,
    BATCH: Int,
](
    env: Int,
    L: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    D: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    m_inv: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
):
    """Full dense M^-1 via per-column LU solves."""
    for j_col in range(NV):
        _lu_m_inv_col_env[DTYPE, NV, BATCH](env, j_col, L, D, m_inv)


# ── launchable kernels (serial: one thread per env) ───────────────────────
def _lu_factor_fields_kernel[
    DTYPE: DType,
    NV: Int,
    BATCH: Int,
](
    M: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    L: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    D: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _lu_factor_env[DTYPE, NV, BATCH](env, M, L, D)


def _lu_solve_fields_kernel[
    DTYPE: DType,
    NV: Int,
    BATCH: Int,
](
    L: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    D: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    b: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    x: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _lu_solve_env[DTYPE, NV, BATCH](env, L, D, b, x)


def _lu_m_inv_fields_kernel[
    DTYPE: DType,
    NV: Int,
    BATCH: Int,
](
    L: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    D: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    m_inv: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _lu_m_inv_env[DTYPE, NV, BATCH](env, L, D, m_inv)


# ── single-body dispatch wrappers (mirror ldl) ─────────────────────
def lu_factor[
    target: StaticString,
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    BATCH: Int = 1,
](
    mut scratch: DynamicsScratch[DTYPE, NV, NBODY, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """M -> LU factors in L + pivots in D (owned scratch), both targets."""
    comptime L_M = Layout.row_major(BATCH, NV * NV)
    comptime L_NV = Layout.row_major(BATCH, NV)

    comptime if target == "cpu":
        var M_v = scratch.M.lt["cpu", L_M]()
        var L_v = scratch.L.lt["cpu", L_M]()
        var D_v = scratch.D.lt["cpu", L_NV]()
        for e in range(BATCH):
            _lu_factor_env[DTYPE, NV, BATCH](e, M_v, L_v, D_v)
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + LU_TPB - 1) // LU_TPB
        c.enqueue_function[_lu_factor_fields_kernel[DTYPE, NV, BATCH]](
            scratch.M.lt["gpu", L_M](),
            scratch.L.lt["gpu", L_M](),
            scratch.D.lt["gpu", L_NV](),
            grid_dim=(BLOCKS,),
            block_dim=(LU_TPB,),
        )


def lu_solve[
    target: StaticString,
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    BATCH: Int = 1,
](
    mut scratch: DynamicsScratch[DTYPE, NV, NBODY, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """`qacc_ws = A^-1 fnet` via LU factors in L + pivots in D, both targets."""
    comptime L_M = Layout.row_major(BATCH, NV * NV)
    comptime L_NV = Layout.row_major(BATCH, NV)

    comptime if target == "cpu":
        var L_v = scratch.L.lt["cpu", L_M]()
        var D_v = scratch.D.lt["cpu", L_NV]()
        var b_v = scratch.fnet.lt["cpu", L_NV]()
        var x_v = scratch.qacc_ws.lt["cpu", L_NV]()
        for e in range(BATCH):
            _lu_solve_env[DTYPE, NV, BATCH](e, L_v, D_v, b_v, x_v)
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + LU_TPB - 1) // LU_TPB
        c.enqueue_function[_lu_solve_fields_kernel[DTYPE, NV, BATCH]](
            scratch.L.lt["gpu", L_M](),
            scratch.D.lt["gpu", L_NV](),
            scratch.fnet.lt["gpu", L_NV](),
            scratch.qacc_ws.lt["gpu", L_NV](),
            grid_dim=(BLOCKS,),
            block_dim=(LU_TPB,),
        )


def compute_m_inv_from_lu[
    target: StaticString,
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    BATCH: Int = 1,
](
    mut scratch: DynamicsScratch[DTYPE, NV, NBODY, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """LU factors (L) + pivots (D) -> m_inv (owned scratch), both targets."""
    comptime L_M = Layout.row_major(BATCH, NV * NV)
    comptime L_NV = Layout.row_major(BATCH, NV)

    comptime if target == "cpu":
        var L_v = scratch.L.lt["cpu", L_M]()
        var D_v = scratch.D.lt["cpu", L_NV]()
        var mi_v = scratch.m_inv.lt["cpu", L_M]()
        for e in range(BATCH):
            _lu_m_inv_env[DTYPE, NV, BATCH](e, L_v, D_v, mi_v)
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + LU_TPB - 1) // LU_TPB
        c.enqueue_function[_lu_m_inv_fields_kernel[DTYPE, NV, BATCH]](
            scratch.L.lt["gpu", L_M](),
            scratch.D.lt["gpu", L_NV](),
            scratch.m_inv.lt["gpu", L_M](),
            grid_dim=(BLOCKS,),
            block_dim=(LU_TPB,),
        )

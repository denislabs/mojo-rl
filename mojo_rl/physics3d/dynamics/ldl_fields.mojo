"""LDL factorization + solve over per-field tensors (migration P2,
single-source). Per-field ports of `ldl_factor_gpu` and
`ldl_solve_workspace_gpu` (dynamics/mass_matrix.mojo) — arithmetic verbatim.
Pure scratch math: factor reads `scratch.M`, writes `scratch.L`/`scratch.D`
(3 operands); solve reads `scratch.L`/`scratch.D`/`scratch.fnet`, writes
`scratch.qacc_ws` (4 operands)."""

from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from ..fields import DynamicsScratch

comptime LDL_TPB: Int = 64


@always_inline
def _ensure_positive[N: Int]() -> Int:
    return N if N > 0 else 1


@always_inline
def _ldl_factor_env_fields[
    DTYPE: DType,
    NV: Int,
    BATCH: Int,
](
    env: Int,
    M: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    L: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    D: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
):
    """LDL factorization for one env (verbatim from ldl_factor_gpu)."""
    for i in range(NV * NV):
        L[env, i] = 0
    for i in range(NV):
        D[env, i] = 0
        L[env, i * NV + i] = 1

    for j in range(NV):
        var d_j = M[env, j * NV + j]
        for k in range(j):
            d_j = d_j - L[env, j * NV + k] * L[env, j * NV + k] * D[env, k]
        D[env, j] = d_j

        if d_j > 1e-14 or d_j < -1e-14:
            for i in range(j + 1, NV):
                var l_ij = M[env, i * NV + j]
                for k in range(j):
                    l_ij = (
                        l_ij
                        - L[env, i * NV + k] * L[env, j * NV + k] * D[env, k]
                    )
                L[env, i * NV + j] = l_ij / d_j


@always_inline
def _ldl_solve_env_fields[
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
    """LDL solve x = M^-1 b for one env (verbatim from
    ldl_solve_workspace_gpu)."""
    comptime V_SIZE = _ensure_positive[NV]()
    var y = InlineArray[L.element_type, V_SIZE](uninitialized=True)
    for i in range(NV):
        var s = b[env, i]
        for j in range(i):
            s = s - L[env, i * NV + j] * y[j]
        y[i] = s

    var z = InlineArray[L.element_type, V_SIZE](uninitialized=True)
    for i in range(NV):
        var d_i = D[env, i]
        if d_i > 1e-14 or d_i < -1e-14:
            z[i] = y[i] / d_i
        else:
            z[i] = 0

    for i in range(NV - 1, -1, -1):
        var s = z[i]
        for j in range(i + 1, NV):
            s = s - L[env, j * NV + i] * x[env, j]
        x[env, i] = s


def _ldl_factor_fields_kernel[
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
    _ldl_factor_env_fields[DTYPE, NV, BATCH](env, M, L, D)


# ── Cooperative (_mt) kernel — schedule from the legacy `ldl_factor_gpu_mt`
# (dynamics/mass_matrix.mojo): columns are sequential, but within column j
# the off-diagonal entries L[i,j] (i>j) are independent — thread tid handles
# rows i = j+1+tid, j+1+tid+n, ... Each thread recomputes d_j locally (same
# reads, same k-ascending reduction -> identical to the serial D[j]); tid 0
# commits D[j]. One barrier per column. Expressions are copied verbatim from
# `_ldl_factor_env_fields` (the legacy _mt duplicates them the same way) ->
# bit-exact. Grid is exact (one block per env) -> no valid_env guards.
def _ldl_factor_fields_mt_kernel[
    DTYPE: DType,
    NV: Int,
    BATCH: Int,
    N_THREADS: Int,
](
    M: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    L: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    D: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
):
    var env = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    # Distributed init: L = 0, D = 0, unit diagonal.
    for i in range(tid, NV * NV, N_THREADS):
        L[env, i] = 0
    for i in range(tid, NV, N_THREADS):
        D[env, i] = 0
        L[env, i * NV + i] = 1
    barrier()

    for j in range(NV):
        # d_j: identical reduction to serial (k ascending). All threads
        # compute it; only tid 0 commits to D.
        var d_j = M[env, j * NV + j]
        for k in range(j):
            d_j = d_j - L[env, j * NV + k] * L[env, j * NV + k] * D[env, k]
        if tid == 0:
            D[env, j] = d_j

        if d_j > 1e-14 or d_j < -1e-14:
            for i in range(j + 1 + tid, NV, N_THREADS):
                var l_ij = M[env, i * NV + j]
                for k in range(j):
                    l_ij = (
                        l_ij
                        - L[env, i * NV + k] * L[env, j * NV + k] * D[env, k]
                    )
                L[env, i * NV + j] = l_ij / d_j
        # Column j complete (L[*,j], D[j]) before column j+1 reads it.
        barrier()


def _ldl_solve_fields_kernel[
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
    _ldl_solve_env_fields[DTYPE, NV, BATCH](env, L, D, b, x)


def ldl_factor_fields[
    target: StaticString,
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    BATCH: Int = 1,
    PARALLEL: Bool = False,
](
    mut scratch: DynamicsScratch[DTYPE, NV, NBODY, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """M -> L, D (owned scratch), both targets, one body. PARALLEL=True
    (GPU only): cooperative per-column kernel, bit-exact vs serial. CPU
    ignores PARALLEL."""
    comptime L_M = Layout.row_major(BATCH, NV * NV)
    comptime L_NV = Layout.row_major(BATCH, NV)

    comptime if target == "cpu":
        var M_v = scratch.M.lt["cpu", L_M]()
        var L_v = scratch.L.lt["cpu", L_M]()
        var D_v = scratch.D.lt["cpu", L_NV]()
        for e in range(BATCH):
            _ldl_factor_env_fields[DTYPE, NV, BATCH](e, M_v, L_v, D_v)
    elif PARALLEL:
        var c = ctx.value()
        comptime MT_T = NV
        c.enqueue_function[
            _ldl_factor_fields_mt_kernel[DTYPE, NV, BATCH, MT_T]
        ](
            scratch.M.lt["gpu", L_M](),
            scratch.L.lt["gpu", L_M](),
            scratch.D.lt["gpu", L_NV](),
            grid_dim=(BATCH,),
            block_dim=(MT_T,),
        )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + LDL_TPB - 1) // LDL_TPB
        c.enqueue_function[_ldl_factor_fields_kernel[DTYPE, NV, BATCH]](
            scratch.M.lt["gpu", L_M](),
            scratch.L.lt["gpu", L_M](),
            scratch.D.lt["gpu", L_NV](),
            grid_dim=(BLOCKS,),
            block_dim=(LDL_TPB,),
        )


def ldl_solve_fields[
    target: StaticString,
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    BATCH: Int = 1,
](
    mut scratch: DynamicsScratch[DTYPE, NV, NBODY, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """qacc_ws = M^-1 fnet via L/D (owned scratch), both targets."""
    comptime L_M = Layout.row_major(BATCH, NV * NV)
    comptime L_NV = Layout.row_major(BATCH, NV)

    comptime if target == "cpu":
        var L_v = scratch.L.lt["cpu", L_M]()
        var D_v = scratch.D.lt["cpu", L_NV]()
        var b_v = scratch.fnet.lt["cpu", L_NV]()
        var x_v = scratch.qacc_ws.lt["cpu", L_NV]()
        for e in range(BATCH):
            _ldl_solve_env_fields[DTYPE, NV, BATCH](e, L_v, D_v, b_v, x_v)
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + LDL_TPB - 1) // LDL_TPB
        c.enqueue_function[_ldl_solve_fields_kernel[DTYPE, NV, BATCH]](
            scratch.L.lt["gpu", L_M](),
            scratch.D.lt["gpu", L_NV](),
            scratch.fnet.lt["gpu", L_NV](),
            scratch.qacc_ws.lt["gpu", L_NV](),
            grid_dim=(BLOCKS,),
            block_dim=(LDL_TPB,),
        )


# ── M^-1 from LDL factors (per-field port of compute_M_inv_from_ldl_gpu) ──
@always_inline
def _m_inv_col_env_fields[
    DTYPE: DType,
    NV: Int,
    BATCH: Int,
](
    env: Int,
    j: Int,
    L: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    D: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    m_inv: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
):
    """One column j of M^-1 (triangular solve on e_j). Extracted verbatim
    from the `_m_inv_env_fields` column loop so serial and _mt schedules
    share identical arithmetic."""
    comptime V_SIZE = _ensure_positive[NV]()
    var e = InlineArray[L.element_type, V_SIZE](uninitialized=True)
    var col = InlineArray[L.element_type, V_SIZE](uninitialized=True)

    for i in range(NV):
        e[i] = 0
    e[j] = 1

    var y = InlineArray[L.element_type, V_SIZE](uninitialized=True)
    for i in range(NV):
        var s = e[i]
        for k in range(i):
            s = s - L[env, i * NV + k] * y[k]
        y[i] = s

    var z = InlineArray[L.element_type, V_SIZE](uninitialized=True)
    for i in range(NV):
        var d_i = D[env, i]
        if d_i > 1e-14 or d_i < -1e-14:
            z[i] = y[i] / d_i
        else:
            z[i] = 0

    for i in range(NV - 1, -1, -1):
        var s = z[i]
        for k in range(i + 1, NV):
            s = s - L[env, k * NV + i] * col[k]
        col[i] = s

    for i in range(NV):
        m_inv[env, i * NV + j] = col[i]


@always_inline
def _m_inv_env_fields[
    DTYPE: DType,
    NV: Int,
    BATCH: Int,
](
    env: Int,
    L: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    D: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    m_inv: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
):
    """Full dense M^-1 via per-column LDL solves (arithmetic verbatim;
    column body now lives in the shared `_m_inv_col_env_fields` helper —
    pure refactor)."""
    for j in range(NV):
        _m_inv_col_env_fields[DTYPE, NV, BATCH](env, j, L, D, m_inv)


def _m_inv_fields_kernel[
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
    _m_inv_env_fields[DTYPE, NV, BATCH](env, L, D, m_inv)


# ── Cooperative (_mt) kernel — schedule from the legacy
# `compute_M_inv_from_ldl_gpu_mt`: each column j of M^-1 is an independent
# triangular solve, so thread tid handles columns j % N_THREADS == tid.
# Per-column arithmetic is the SAME `_m_inv_col_env_fields` helper as the
# serial kernel -> bit-exact. No barriers needed (columns independent; the
# LDL factors are inputs from a previous launch).
def _m_inv_fields_mt_kernel[
    DTYPE: DType,
    NV: Int,
    BATCH: Int,
    N_THREADS: Int,
](
    L: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    D: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    m_inv: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
):
    var env = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    for j in range(tid, NV, N_THREADS):
        _m_inv_col_env_fields[DTYPE, NV, BATCH](env, j, L, D, m_inv)


def compute_m_inv_fields[
    target: StaticString,
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    BATCH: Int = 1,
    PARALLEL: Bool = False,
](
    mut scratch: DynamicsScratch[DTYPE, NV, NBODY, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """L, D -> m_inv (owned scratch), both targets, one body. PARALLEL=True
    (GPU only): cooperative column-striped kernel, bit-exact vs serial. CPU
    ignores PARALLEL."""
    comptime L_M = Layout.row_major(BATCH, NV * NV)
    comptime L_NV = Layout.row_major(BATCH, NV)

    comptime if target == "cpu":
        var L_v = scratch.L.lt["cpu", L_M]()
        var D_v = scratch.D.lt["cpu", L_NV]()
        var mi_v = scratch.m_inv.lt["cpu", L_M]()
        for e in range(BATCH):
            _m_inv_env_fields[DTYPE, NV, BATCH](e, L_v, D_v, mi_v)
    elif PARALLEL:
        var c = ctx.value()
        comptime MT_T = NV
        c.enqueue_function[_m_inv_fields_mt_kernel[DTYPE, NV, BATCH, MT_T]](
            scratch.L.lt["gpu", L_M](),
            scratch.D.lt["gpu", L_NV](),
            scratch.m_inv.lt["gpu", L_M](),
            grid_dim=(BATCH,),
            block_dim=(MT_T,),
        )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + LDL_TPB - 1) // LDL_TPB
        c.enqueue_function[_m_inv_fields_kernel[DTYPE, NV, BATCH]](
            scratch.L.lt["gpu", L_M](),
            scratch.D.lt["gpu", L_NV](),
            scratch.m_inv.lt["gpu", L_M](),
            grid_dim=(BLOCKS,),
            block_dim=(LDL_TPB,),
        )

"""LDL factorization + solve over per-field tensors (migration P2,
single-source). Per-field ports of `ldl_factor_gpu` and
`ldl_solve_workspace_gpu` (dynamics/mass_matrix.mojo) — arithmetic verbatim.
Pure scratch math: factor reads `scratch.M`, writes `scratch.L`/`scratch.D`
(3 operands); solve reads `scratch.L`/`scratch.D`/`scratch.fnet`, writes
`scratch.qacc_ws` (4 operands).

⚠⚠ `trees` IS THREADED THROUGH AND NOT YET READ. It is `Model.trees` —
`(dof_adr, dof_num, kind)` per kinematic tree, i.e. M's DIAGONAL BLOCKS. The
factorisations below are dense in `nv` and the blocks are what makes them
`sum(size^3)` instead of `nv^3`
(`docs/BLOCK_DIAGONAL_MASS_MATRIX_IMPLEMENTATION.md`, P2). This commit lands
the OPERAND ONLY — a wide, mechanical signature change across seven engine
sites and five tests — so that the arithmetic change lands on a tree where
nothing else is moving. See
`feedback_a_concurrent_commit_swept_my_in_flight_edits`.

⚠ THE TABLE TERMINATES ITSELF: rows past `ntree` are `(0, 0, 0)`, so
`dof_num == 0` ends the walk and a `Model` built without the parser — which
leaves the whole table zeroed — reads as NO BLOCKS. That case must fall back
to a single whole-`nv` block, not to zero work."""

from std.gpu import thread_idx, block_idx, block_dim
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from ..fields import (
    DynamicsScratch,
    Model,
    Dims,
    DimsLike,
    AsStatic,
    Scratch,
    cap,
    DYN2,
    rl2,
)
from ..gpu.constants import MODEL_TREE_SIZE

comptime LDL_TPB: Int = 64


@always_inline
def _ldl_factor_env[
    DTYPE: DType,
    DIMS: DimsLike,
    LM: Layout,
    LNV: Layout,
    LTREE: Layout,
](
    env: Int,
    dims: DIMS,
    M: LayoutTensor[DTYPE, LM, MutAnyOrigin],
    L: LayoutTensor[DTYPE, LM, MutAnyOrigin],
    D: LayoutTensor[DTYPE, LNV, MutAnyOrigin],
    trees: LayoutTensor[DTYPE, LTREE, MutAnyOrigin],
):
    """LDL factorization for one env (verbatim from ldl_factor_gpu).

    ⚠ `DIMS`, NOT `D`, FOR THE SAME REASON `DynamicsScratch` USES IT: `D` is
    already taken here by the LDL diagonal, and the two collide in one scope.

    `LM` / `LNV` are the leg-polymorphism parameters of §12.4 — the GPU
    kernel infers them as `Layout.row_major(BATCH, NV*NV)` and the dynamic
    CPU path as `DYN2`, and this body is compiled for each. Every index is
    hand-computed (`L[env, i*nv + j]`), so nothing in the arithmetic knows
    which it got."""
    var nv = dims.get_nv()
    for i in range(nv * nv):
        L[env, i] = 0
    for i in range(nv):
        D[env, i] = 0
        L[env, i * nv + i] = 1

    for j in range(nv):
        var d_j = M[env, j * nv + j]
        for k in range(j):
            d_j = d_j - L[env, j * nv + k] * L[env, j * nv + k] * D[env, k]
        D[env, j] = d_j

        if d_j > 1e-14 or d_j < -1e-14:
            for i in range(j + 1, nv):
                var l_ij = M[env, i * nv + j]
                for k in range(j):
                    l_ij = (
                        l_ij
                        - L[env, i * nv + k] * L[env, j * nv + k] * D[env, k]
                    )
                L[env, i * nv + j] = l_ij / d_j


@always_inline
def _ldl_solve_env[
    DTYPE: DType,
    DIMS: DimsLike,
    LM: Layout,
    LNV: Layout,
](
    env: Int,
    dims: DIMS,
    L: LayoutTensor[DTYPE, LM, MutAnyOrigin],
    D: LayoutTensor[DTYPE, LNV, MutAnyOrigin],
    b: LayoutTensor[DTYPE, LNV, MutAnyOrigin],
    x: LayoutTensor[DTYPE, LNV, MutAnyOrigin],
):
    """LDL solve x = M^-1 b for one env (verbatim from
    ldl_solve_workspace_gpu).

    ⚠ THE SCRATCH CONTAINER IS CHOSEN BY THE LEG, THE LOOPS ARE ALWAYS THE
    RUNTIME DIM. `cap[DIMS.NV]()` is `NV` on a static provider — so `Scratch`
    is the `InlineArray[.., NV]` that ships today, to the byte — and 0 on a
    dynamic one, which selects the heap `List`. §10.7 measured that a
    fixed-cap stack array indexed by a runtime bound is 1.13-1.18x WORSE than
    the heap, so there is deliberately no third option here."""
    var nv = dims.get_nv()
    comptime V_CAP = cap[DIMS.NV]()
    var y = Scratch[L.element_type, V_CAP](nv, uninitialized=0)
    for i in range(nv):
        var s = b[env, i]
        for j in range(i):
            s = s - L[env, i * nv + j] * y[j]
        y[i] = s

    var z = Scratch[L.element_type, V_CAP](nv, uninitialized=0)
    for i in range(nv):
        var d_i = D[env, i]
        if d_i > 1e-14 or d_i < -1e-14:
            z[i] = y[i] / d_i
        else:
            z[i] = 0

    for i in range(nv - 1, -1, -1):
        var s = z[i]
        for j in range(i + 1, nv):
            s = s - L[env, j * nv + i] * x[env, j]
        x[env, i] = s


def _ldl_factor_fields_kernel[
    DTYPE: DType,
    NV: Int,
    BATCH: Int,
](
    M: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    L: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    D: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    trees: LayoutTensor[DTYPE, Layout.row_major(NV, MODEL_TREE_SIZE), MutAnyOrigin],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _ldl_factor_env(env, Dims[nv=NV](), M, L, D, trees)


# ── Cooperative (_mt) kernel — schedule from the legacy `ldl_factor_gpu_mt`
# (dynamics/mass_matrix.mojo): columns are sequential, but within column j
# the off-diagonal entries L[i,j] (i>j) are independent — thread tid handles
# rows i = j+1+tid, j+1+tid+n, ... Each thread recomputes d_j locally (same
# reads, same k-ascending reduction -> identical to the serial D[j]); tid 0
# commits D[j]. One barrier per column. Expressions are copied verbatim from
# `_ldl_factor_env` (the legacy _mt duplicates them the same way) ->
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
    trees: LayoutTensor[DTYPE, Layout.row_major(NV, MODEL_TREE_SIZE), MutAnyOrigin],
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
    _ldl_solve_env(env, Dims[nv=NV](), L, D, b, x)


def ldl_factor[
    target: StaticString,
    DTYPE: DType,
    D: DimsLike,
    BATCH: Int = 1,
    PARALLEL: Bool = False,
](
    # ⚠ `mut`, MATCHING `compute_cdof` / `compute_mass_matrix`. Not because
    # anything here writes the model — nothing does — but because `TensorImpl.lt`
    # and `.lt_dyn` are mutating methods, so an immutable borrow cannot bind a
    # record as a kernel operand at all.
    mut m: Model[DTYPE, D],
    mut scratch: DynamicsScratch[DTYPE, D, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """M -> L, D (owned scratch), both targets, one body. PARALLEL=True
    (GPU only): cooperative per-column kernel, bit-exact vs serial. CPU
    ignores PARALLEL."""
    comptime L_M = Layout.row_major(BATCH, D.NV * D.NV)
    comptime L_NV = Layout.row_major(BATCH, D.NV)
    comptime L_TREE = Layout.row_major(D.NV, MODEL_TREE_SIZE)

    comptime if target == "cpu":
        var dm = scratch.dims
        var rl_M = rl2(BATCH, dm.get_nv() * dm.get_nv())
        var rl_NV = rl2(BATCH, dm.get_nv())
        var rl_TREE = rl2(dm.get_nv(), MODEL_TREE_SIZE)
        var M_v = scratch.M.lt_dyn["cpu", DYN2](rl_M)
        var L_v = scratch.L.lt_dyn["cpu", DYN2](rl_M)
        var D_v = scratch.D.lt_dyn["cpu", DYN2](rl_NV)
        var T_v = m.trees.lt_dyn["cpu", DYN2](rl_TREE)
        for e in range(BATCH):
            _ldl_factor_env(e, dm, M_v, L_v, D_v, T_v)
    elif PARALLEL:
        var c = ctx.value()
        comptime MT_T = D.NV
        c.enqueue_function[
            _ldl_factor_fields_mt_kernel[DTYPE, D.NV, BATCH, MT_T]
        ](
            scratch.M.lt["gpu", L_M](),
            scratch.L.lt["gpu", L_M](),
            scratch.D.lt["gpu", L_NV](),
            m.trees.lt["gpu", L_TREE](),
            grid_dim=(BATCH,),
            block_dim=(MT_T,),
        )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + LDL_TPB - 1) // LDL_TPB
        c.enqueue_function[_ldl_factor_fields_kernel[DTYPE, D.NV, BATCH]](
            scratch.M.lt["gpu", L_M](),
            scratch.L.lt["gpu", L_M](),
            scratch.D.lt["gpu", L_NV](),
            m.trees.lt["gpu", L_TREE](),
            grid_dim=(BLOCKS,),
            block_dim=(LDL_TPB,),
        )


def ldl_solve[target: StaticString, DTYPE: DType, D: DimsLike, BATCH: Int = 1](
    mut scratch: DynamicsScratch[DTYPE, D, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """`qacc_ws = M^-1 fnet` via L/D (owned scratch), both targets."""
    comptime L_M = Layout.row_major(BATCH, D.NV * D.NV)
    comptime L_NV = Layout.row_major(BATCH, D.NV)

    comptime if target == "cpu":
        var dm = scratch.dims
        var rl_M = rl2(BATCH, dm.get_nv() * dm.get_nv())
        var rl_NV = rl2(BATCH, dm.get_nv())
        var L_v = scratch.L.lt_dyn["cpu", DYN2](rl_M)
        var D_v = scratch.D.lt_dyn["cpu", DYN2](rl_NV)
        var b_v = scratch.fnet.lt_dyn["cpu", DYN2](rl_NV)
        var x_v = scratch.qacc_ws.lt_dyn["cpu", DYN2](rl_NV)
        for e in range(BATCH):
            _ldl_solve_env(e, dm, L_v, D_v, b_v, x_v)
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + LDL_TPB - 1) // LDL_TPB
        c.enqueue_function[_ldl_solve_fields_kernel[DTYPE, D.NV, BATCH]](
            scratch.L.lt["gpu", L_M](),
            scratch.D.lt["gpu", L_NV](),
            scratch.fnet.lt["gpu", L_NV](),
            scratch.qacc_ws.lt["gpu", L_NV](),
            grid_dim=(BLOCKS,),
            block_dim=(LDL_TPB,),
        )


# ── M^-1 from LDL factors (per-field port of compute_M_inv_from_ldl_gpu) ──
@always_inline
def _m_inv_col_env[
    DTYPE: DType,
    DIMS: DimsLike,
    LM: Layout,
    LNV: Layout,
    LTREE: Layout,
](
    env: Int,
    j: Int,
    dims: DIMS,
    L: LayoutTensor[DTYPE, LM, MutAnyOrigin],
    D: LayoutTensor[DTYPE, LNV, MutAnyOrigin],
    m_inv: LayoutTensor[DTYPE, LM, MutAnyOrigin],
    trees: LayoutTensor[DTYPE, LTREE, MutAnyOrigin],
):
    """One column j of M^-1 (triangular solve on e_j). Extracted verbatim
    from the `_m_inv_env` column loop so serial and _mt schedules
    share identical arithmetic."""
    var nv = dims.get_nv()
    comptime V_CAP = cap[DIMS.NV]()
    var e = Scratch[L.element_type, V_CAP](nv, uninitialized=0)
    var col = Scratch[L.element_type, V_CAP](nv, uninitialized=0)

    for i in range(nv):
        e[i] = 0
    e[j] = 1

    var y = Scratch[L.element_type, V_CAP](nv, uninitialized=0)
    for i in range(nv):
        var s = e[i]
        for k in range(i):
            s = s - L[env, i * nv + k] * y[k]
        y[i] = s

    var z = Scratch[L.element_type, V_CAP](nv, uninitialized=0)
    for i in range(nv):
        var d_i = D[env, i]
        if d_i > 1e-14 or d_i < -1e-14:
            z[i] = y[i] / d_i
        else:
            z[i] = 0

    for i in range(nv - 1, -1, -1):
        var s = z[i]
        for k in range(i + 1, nv):
            s = s - L[env, k * nv + i] * col[k]
        col[i] = s

    for i in range(nv):
        m_inv[env, i * nv + j] = col[i]


@always_inline
def _m_inv_env[
    DTYPE: DType,
    DIMS: DimsLike,
    LM: Layout,
    LNV: Layout,
    LTREE: Layout,
](
    env: Int,
    dims: DIMS,
    L: LayoutTensor[DTYPE, LM, MutAnyOrigin],
    D: LayoutTensor[DTYPE, LNV, MutAnyOrigin],
    m_inv: LayoutTensor[DTYPE, LM, MutAnyOrigin],
    trees: LayoutTensor[DTYPE, LTREE, MutAnyOrigin],
):
    """Full dense M^-1 via per-column LDL solves (arithmetic verbatim;
    column body now lives in the shared `_m_inv_col_env` helper —
    pure refactor)."""
    for j in range(dims.get_nv()):
        _m_inv_col_env(env, j, dims, L, D, m_inv, trees)


def _m_inv_fields_kernel[
    DTYPE: DType,
    NV: Int,
    BATCH: Int,
](
    L: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    D: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    m_inv: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    trees: LayoutTensor[DTYPE, Layout.row_major(NV, MODEL_TREE_SIZE), MutAnyOrigin],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _m_inv_env(env, Dims[nv=NV](), L, D, m_inv, trees)


# ── Cooperative (_mt) kernel — schedule from the legacy
# `compute_M_inv_from_ldl_gpu_mt`: each column j of M^-1 is an independent
# triangular solve, so thread tid handles columns j % N_THREADS == tid.
# Per-column arithmetic is the SAME `_m_inv_col_env` helper as the
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
    trees: LayoutTensor[DTYPE, Layout.row_major(NV, MODEL_TREE_SIZE), MutAnyOrigin],
):
    var env = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    for j in range(tid, NV, N_THREADS):
        _m_inv_col_env(env, j, Dims[nv=NV](), L, D, m_inv, trees)


def compute_m_inv[
    target: StaticString,
    DTYPE: DType,
    D: DimsLike,
    BATCH: Int = 1,
    PARALLEL: Bool = False,
](
    # ⚠ `mut`, MATCHING `compute_cdof` / `compute_mass_matrix`. Not because
    # anything here writes the model — nothing does — but because `TensorImpl.lt`
    # and `.lt_dyn` are mutating methods, so an immutable borrow cannot bind a
    # record as a kernel operand at all.
    mut m: Model[DTYPE, D],
    mut scratch: DynamicsScratch[DTYPE, D, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """L, D -> m_inv (owned scratch), both targets, one body. PARALLEL=True
    (GPU only): cooperative column-striped kernel, bit-exact vs serial. CPU
    ignores PARALLEL."""
    comptime L_M = Layout.row_major(BATCH, D.NV * D.NV)
    comptime L_NV = Layout.row_major(BATCH, D.NV)
    comptime L_TREE = Layout.row_major(D.NV, MODEL_TREE_SIZE)

    comptime if target == "cpu":
        var dm = scratch.dims
        var rl_M = rl2(BATCH, dm.get_nv() * dm.get_nv())
        var rl_NV = rl2(BATCH, dm.get_nv())
        var rl_TREE = rl2(dm.get_nv(), MODEL_TREE_SIZE)
        var L_v = scratch.L.lt_dyn["cpu", DYN2](rl_M)
        var D_v = scratch.D.lt_dyn["cpu", DYN2](rl_NV)
        var mi_v = scratch.m_inv.lt_dyn["cpu", DYN2](rl_M)
        var T_v = m.trees.lt_dyn["cpu", DYN2](rl_TREE)
        for e in range(BATCH):
            _m_inv_env(e, dm, L_v, D_v, mi_v, T_v)
    elif PARALLEL:
        var c = ctx.value()
        comptime MT_T = D.NV
        c.enqueue_function[_m_inv_fields_mt_kernel[DTYPE, D.NV, BATCH, MT_T]](
            scratch.L.lt["gpu", L_M](),
            scratch.D.lt["gpu", L_NV](),
            scratch.m_inv.lt["gpu", L_M](),
            m.trees.lt["gpu", L_TREE](),
            grid_dim=(BATCH,),
            block_dim=(MT_T,),
        )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + LDL_TPB - 1) // LDL_TPB
        c.enqueue_function[_m_inv_fields_kernel[DTYPE, D.NV, BATCH]](
            scratch.L.lt["gpu", L_M](),
            scratch.D.lt["gpu", L_NV](),
            scratch.m_inv.lt["gpu", L_M](),
            m.trees.lt["gpu", L_TREE](),
            grid_dim=(BLOCKS,),
            block_dim=(LDL_TPB,),
        )

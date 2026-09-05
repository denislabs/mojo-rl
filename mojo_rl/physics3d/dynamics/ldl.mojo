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

from std.sys import simd_width_of
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
    DYN1,
    DYN2,
    rl1,
    rl2,
)
from ..gpu.constants import (
    MODEL_TREE_SIZE,
    TREE_IDX_DOF_ADR,
    TREE_IDX_DOF_NUM,
    MODEL_META_IDX_NTREE,
)


@always_inline
def _dof_block[
    DTYPE: DType, LT: Layout
](trees: LayoutTensor[DTYPE, LT, MutAnyOrigin], nv: Int, j: Int) -> Tuple[Int, Int]:
    """The half-open dof range of the kinematic tree containing dof `j`.

    ⚠⚠ THE ONE SPELLING OF THE RULE, and it is called at BLOCK BOUNDARIES
    ONLY. Every loop below advances `j` sequentially, so calling this when
    `j >= b1` costs `ntree` walks of `ntree` steps — ~100 operations at k=9 —
    against the ~400 the restriction leaves behind. Calling it per column
    instead would be `nv*ntree` = 600 and would cost more than it saves.

    ⚠ A DEGENERATE TABLE IS ONE BLOCK, NEVER ZERO. `trees` is zeroed by
    `Model.__init__` and only the parser fills it, so `dof_num == 0` both
    terminates the table and marks "no table" — a `Model` built without the
    parser must factor the whole `nv`, exactly as it does today.
    """
    var adr = 0
    for t in range(nv):
        var num = Int(trees[t * MODEL_TREE_SIZE + TREE_IDX_DOF_NUM])
        if num <= 0:
            break
        adr = Int(trees[t * MODEL_TREE_SIZE + TREE_IDX_DOF_ADR])
        if j >= adr and j < adr + num:
            return (adr, adr + num)
    return (0, nv)

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

    # ⚠⚠ THE BLOCK OF COLUMN j, ADVANCED IN LOCKSTEP. `M`'s cross-tree entries
    # are STRUCTURALLY exactly zero — the treewalk CRBA zeroes `M` and writes
    # only ancestor pairs (`mass_matrix.mojo:592-600`), and the dense path
    # accumulates only over bodies in BOTH subtrees (`:219-223`), which no
    # cross-tree pair has. Between CRBA and here only `_armature_env` runs, and
    # it touches the diagonal. So every term these ranges drop is a product
    # with an exact `0.0`, and a sequential accumulation that drops exact zeros
    # returns the identical bit pattern: this is bit-exact on EVERY model, not
    # just single-tree ones.
    var b0 = 0
    var b1 = 0
    for j in range(nv):
        if j >= b1:
            var bb = _dof_block(trees, nv, j)
            b0 = bb[0]
            b1 = bb[1]
        var d_j = M[env, j * nv + j]
        for k in range(b0, j):
            d_j = d_j - L[env, j * nv + k] * L[env, j * nv + k] * D[env, k]
        D[env, j] = d_j

        if d_j > 1e-14 or d_j < -1e-14:
            for i in range(j + 1, b1):
                var l_ij = M[env, i * nv + j]
                for k in range(b0, j):
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
    LTREE: Layout,
](
    env: Int,
    dims: DIMS,
    L: LayoutTensor[DTYPE, LM, MutAnyOrigin],
    D: LayoutTensor[DTYPE, LNV, MutAnyOrigin],
    b: LayoutTensor[DTYPE, LNV, MutAnyOrigin],
    x: LayoutTensor[DTYPE, LNV, MutAnyOrigin],
    trees: LayoutTensor[DTYPE, LTREE, MutAnyOrigin],
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
    # ⚠⚠ THE BLOCK OF ROW i, LOOKED UP AT BOUNDARIES ONLY. `L` is block
    # diagonal because `M` is — its cross-tree entries are STRUCTURALLY exactly
    # `0.0` (see `_ldl_factor_env`) — so the segments are independent triangular
    # systems and every term these ranges drop is a product with an exact zero.
    # Bit-exact on every model, single-tree or not.
    #
    # ⚠ THE TEST IS `i < b0 or i >= b1`, NOT `i >= b1`, because the third loop
    # below runs DESCENDING. One spelling that works in both directions beats
    # two that each work in one.
    var y = Scratch[L.element_type, V_CAP](nv, uninitialized=0)
    var b0 = 1
    var b1 = 0
    for i in range(nv):
        if i < b0 or i >= b1:
            var bb = _dof_block(trees, nv, i)
            b0 = bb[0]
            b1 = bb[1]
        var s = b[env, i]
        for j in range(b0, i):
            s = s - L[env, i * nv + j] * y[j]
        y[i] = s

    var z = Scratch[L.element_type, V_CAP](nv, uninitialized=0)
    for i in range(nv):
        var d_i = D[env, i]
        if d_i > 1e-14 or d_i < -1e-14:
            z[i] = y[i] / d_i
        else:
            z[i] = 0

    # ⚠ `x` IS STILL WRITTEN FOR EVERY i — only the inner sum is restricted, so
    # there is no staleness to guard against here, unlike `m_inv`.
    var c0 = 1
    var c1 = 0
    for i in range(nv - 1, -1, -1):
        if i < c0 or i >= c1:
            var cc = _dof_block(trees, nv, i)
            c0 = cc[0]
            c1 = cc[1]
        var s = z[i]
        for j in range(i + 1, c1):
            s = s - L[env, j * nv + i] * x[env, j]
        x[env, i] = s


# =============================================================================
# The TREE-ORDERED LDL — MuJoCo's `mj_factorI` / `mj_solveLD`, CPU path
# =============================================================================
#
# ⚠⚠ A DIFFERENT FACTORISATION FROM THE ONE ABOVE, IN THE SAME BUFFERS.
# `_ldl_factor_env` eliminates forward and produces `M = L D Lᵀ`; on a
# kinematic tree that FILLS IN between siblings (eliminating a root couples
# every child), so `L` is dense within a tree and every solve is O(nv²) — and
# `compute_m_inv`, nv such solves, is O(nv³): 48% of dog's step after
# everything else was fixed (PERFORMANCE.md §13.10).
#
# MuJoCo eliminates from the LAST dof backwards (`engine_core_smooth.c:1973`)
# and gets `M = Lᵀ D L` with `L` unit-lower on M's OWN sparsity: row `k` is
# nonzero at `k`'s ancestors and nowhere else, no fill. Every solve is then
# O(nC) (`:2113`) and the inverse O(nv² · depth). These three routines are
# that algorithm, on our dense `[nv*nv]` storage, walking `Model.dof_parentid`
# instead of a sparse index.
#
# ⚠ THE CPU DISPATCHERS BELOW SELECT THEM when `MODEL_META_IDX_NTREE > 0`
# (the parser ran and the table is real) and keep the dense trio otherwise
# and on every GPU leg. So `scratch.L` holds `Lᵀ D L`'s L on the CPU and
# `L D Lᵀ`'s L on a GPU: a factor is only ever read by the solve that shares
# its dispatcher, and a test that compares `L` ACROSS the two conventions is
# comparing two different matrices. Compare solves.
#
# Numerically the two are different roundings of the same inverse; nothing
# here is bit-exact against the dense trio, and the gates are MuJoCo's:
# `dof_invweight0` at qpos0 (`test_constraints_vs_mujoco`, ~1e-15) and the
# trajectory gates.


@always_inline
def _ldl_factor_tree_env[
    DTYPE: DType,
    DIMS: DimsLike,
    LM: Layout,
    LNV: Layout,
    LP: Layout,
](
    env: Int,
    dims: DIMS,
    M: LayoutTensor[DTYPE, LM, MutAnyOrigin],
    L: LayoutTensor[DTYPE, LM, MutAnyOrigin],
    D: LayoutTensor[DTYPE, LNV, MutAnyOrigin],
    dofp: LayoutTensor[DTYPE, LP, MutAnyOrigin],
):
    """`M = Lᵀ D L`, `mj_factorI` on dense storage. `L[env, k*nv + j]` for
    `j` an ancestor of `k`; unit diagonal stored; everything else zero.

    ⚠ THE PARENT TABLE IS READ INTO INTEGERS ONCE. The chain walk is the
    whole cost of these routines — there is almost no arithmetic left — and a
    float->int conversion on every hop was a large part of it."""
    var nv = dims.get_nv()
    comptime V_CAP = cap[DIMS.NV]()
    comptime A_CAP = V_CAP * V_CAP
    var par = Scratch[Int, V_CAP](nv, uninitialized=0)
    for i in range(nv):
        par[i] = Int(dofp[i])
    # ⚠ FACTORED IN A COMPACT ROW LAYOUT, THEN SCATTERED. Row `k` of `Lc`
    # holds `L[k, anc]` for `anc` the ancestors of `k` ROOT-FIRST in slots
    # `0 .. dep[k]`, and the diagonal in slot `dep[k]` — `mj_factorI`'s `qLD`
    # shape. Because an ancestor's list is a PREFIX of its descendant's, the
    # inner update `L[i, {i} ∪ anc(i)] += coef · L[k, same]` for `i ∈ anc(k)`
    # is one CONTIGUOUS axpy of `dep[i] + 1` entries — `W` lanes wide — where
    # the dense-storage form gathered `nv`-strided entries one dependent hop
    # at a time (~8k of them on dog, twice a step). The dense `L` the solves
    # read is rebuilt from `Lc` at the end. Each element update is the same
    # single `a + coef·b`, so the result is identical up to the compiler's
    # multiply-add contraction (PERFORMANCE.md §13.23).
    # Root-first ancestor lists in one ascending pass: a dof's list is its
    # parent's list plus the parent, and a parent always has the smaller
    # index — no chain walk, which at small `nv` (the RK4 gym models factor
    # four times a step) was the visible cost of this routine.
    var dep = Scratch[Int, V_CAP](nv, uninitialized=0)
    var anc = Scratch[Int, A_CAP](nv * nv, uninitialized=0)
    for k in range(nv):
        var pk = par[k]
        if pk < 0:
            dep[k] = 0
        else:
            var dp = dep[pk]
            var rk = k * nv
            var rp = pk * nv
            for a in range(dp):
                anc[rk + a] = anc[rp + a]
            anc[rk + dp] = pk
            dep[k] = dp + 1
    var nn = nv * nv
    var Lp = L.ptr + env * nn
    var Mp = M.ptr + env * nn
    var Lc = Scratch[Scalar[DTYPE], A_CAP](nn, uninitialized=0)
    for k in range(nv):
        var rk = k * nv
        for a in range(dep[k]):
            Lc[rk + a] = Mp[rk + anc[rk + a]]
        Lc[rk + dep[k]] = Mp[rk + k]
    var cp = Lc.unsafe_ptr()
    comptime W = 2 * simd_width_of[DTYPE]()
    for k in range(nv - 1, -1, -1):
        var rk = k * nv
        var dk = Lc[rk + dep[k]]
        var inv_d = Scalar[DTYPE](0)
        if dk > 1e-14 or dk < -1e-14:
            inv_d = Scalar[DTYPE](1) / dk
        for a in range(dep[k]):
            var i = anc[rk + a]
            var ri = i * nv
            var coef = -(Lc[rk + a] * inv_d)
            var n_up = a + 1
            var cv = SIMD[DTYPE, W](coef)
            var q = 0
            while q + W <= n_up:
                cp.store(
                    ri + q,
                    cp.load[width=W](ri + q) + cv * cp.load[width=W](rk + q),
                )
                q += W
            while q < n_up:
                cp[ri + q] = cp[ri + q] + coef * cp[rk + q]
                q += 1
        D[env, k] = dk
        for a in range(dep[k]):
            Lc[rk + a] = Lc[rk + a] * inv_d
        Lc[rk + dep[k]] = 1
    # Dense `L` for the solves: zero, then scatter the compact rows.
    var q = 0
    while q + W <= nn:
        Lp.store(q, SIMD[DTYPE, W](0))
        q += W
    while q < nn:
        Lp[q] = 0
        q += 1
    for k in range(nv):
        var rk = k * nv
        for a in range(dep[k]):
            Lp[rk + anc[rk + a]] = Lc[rk + a]
        Lp[rk + k] = 1


@always_inline
def _ldl_solve_tree_env[
    DTYPE: DType,
    DIMS: DimsLike,
    LM: Layout,
    LNV: Layout,
    LP: Layout,
](
    env: Int,
    dims: DIMS,
    L: LayoutTensor[DTYPE, LM, MutAnyOrigin],
    D: LayoutTensor[DTYPE, LNV, MutAnyOrigin],
    b: LayoutTensor[DTYPE, LNV, MutAnyOrigin],
    x: LayoutTensor[DTYPE, LNV, MutAnyOrigin],
    dofp: LayoutTensor[DTYPE, LP, MutAnyOrigin],
):
    """`x = (Lᵀ D L)⁻¹ b`, `mj_solveLD`: scatter up the tree, divide, gather."""
    var nv = dims.get_nv()
    comptime V_CAP = cap[DIMS.NV]()
    var par = Scratch[Int, V_CAP](nv, uninitialized=0)
    for i in range(nv):
        par[i] = Int(dofp[i])
    var y = Scratch[L.element_type, V_CAP](nv, uninitialized=0)
    var Lp = L.ptr + env * nv * nv
    for i in range(nv):
        y[i] = b[env, i]
    for i in range(nv - 1, -1, -1):
        var yi = y[i]
        if yi != 0:
            var ri = i * nv
            var j = par[i]
            while j >= 0:
                y[j] = y[j] - Lp[ri + j] * yi
                j = par[j]
    for i in range(nv):
        var d_i = D[env, i]
        if d_i > 1e-14 or d_i < -1e-14:
            y[i] = y[i] / d_i
        else:
            y[i] = 0
    for i in range(nv):
        var ri = i * nv
        var s = y[i]
        var j = par[i]
        while j >= 0:
            s = s - Lp[ri + j] * y[j]
            j = par[j]
        y[i] = s
        x[env, i] = s


@always_inline
def _m_inv_tree_env[
    DTYPE: DType,
    DIMS: DimsLike,
    LM: Layout,
    LNV: Layout,
    LP: Layout,
](
    env: Int,
    dims: DIMS,
    L: LayoutTensor[DTYPE, LM, MutAnyOrigin],
    D: LayoutTensor[DTYPE, LNV, MutAnyOrigin],
    m_inv: LayoutTensor[DTYPE, LM, MutAnyOrigin],
    dofp: LayoutTensor[DTYPE, LP, MutAnyOrigin],
):
    """Dense `M⁻¹`, one tree solve per column: O(nv² · depth) in all."""
    var nv = dims.get_nv()
    comptime V_CAP = cap[DIMS.NV]()
    var par = Scratch[Int, V_CAP](nv, uninitialized=0)
    for i in range(nv):
        par[i] = Int(dofp[i])
    var y = Scratch[L.element_type, V_CAP](nv, uninitialized=0)
    for c in range(nv):
        for i in range(nv):
            y[i] = 0
        y[c] = 1
        # Only dofs at or below `c` can be nonzero after the scatter.
        for i in range(c, -1, -1):
            var yi = y[i]
            if yi != 0:
                var ri = i * nv
                var j = par[i]
                while j >= 0:
                    y[j] = y[j] - L[env, ri + j] * yi
                    j = par[j]
        for i in range(nv):
            var d_i = D[env, i]
            if d_i > 1e-14 or d_i < -1e-14:
                y[i] = y[i] / d_i
            else:
                y[i] = 0
        for i in range(nv):
            var ri = i * nv
            var s = y[i]
            var j = par[i]
            while j >= 0:
                s = s - L[env, ri + j] * y[j]
                j = par[j]
            y[i] = s
            m_inv[env, ri + c] = s


def _ldl_factor_fields_kernel[
    DTYPE: DType,
    NV: Int,
    BATCH: Int,
](
    M: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    L: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    D: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    trees: LayoutTensor[
        DTYPE, Layout.row_major(NV * MODEL_TREE_SIZE), MutAnyOrigin
    ],
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
    trees: LayoutTensor[
        DTYPE, Layout.row_major(NV * MODEL_TREE_SIZE), MutAnyOrigin
    ],
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

    # The block of column j, in lockstep — see `_ldl_factor_env` for why the
    # restriction is bit-exact. Every thread walks it identically, so the
    # per-column `barrier()` schedule below is untouched.
    var b0 = 0
    var b1 = 0
    for j in range(NV):
        if j >= b1:
            var bb = _dof_block(trees, NV, j)
            b0 = bb[0]
            b1 = bb[1]
        # d_j: identical reduction to serial (k ascending). All threads
        # compute it; only tid 0 commits to D.
        var d_j = M[env, j * NV + j]
        for k in range(b0, j):
            d_j = d_j - L[env, j * NV + k] * L[env, j * NV + k] * D[env, k]
        if tid == 0:
            D[env, j] = d_j

        if d_j > 1e-14 or d_j < -1e-14:
            for i in range(j + 1 + tid, b1, N_THREADS):
                var l_ij = M[env, i * NV + j]
                for k in range(b0, j):
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
    trees: LayoutTensor[
        DTYPE, Layout.row_major(NV * MODEL_TREE_SIZE), MutAnyOrigin
    ],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _ldl_solve_env(env, Dims[nv=NV](), L, D, b, x, trees)


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
    # ⚠ FLAT — `[t*MODEL_TREE_SIZE + col]`, the single spelling shared with
    # `newton_blocks` and `newton_solve`. A 2-D `LayoutTensor` given ONE index
    # returns a ROW, so the two forms are not interchangeable.
    comptime L_TREE = Layout.row_major(D.NV * MODEL_TREE_SIZE)

    comptime if target == "cpu":
        var dm = scratch.dims
        var rl_M = rl2(BATCH, dm.get_nv() * dm.get_nv())
        var rl_NV = rl2(BATCH, dm.get_nv())
        var rl_TREE = rl1(dm.get_nv() * MODEL_TREE_SIZE)
        var M_v = scratch.M.lt_dyn["cpu", DYN2](rl_M)
        var L_v = scratch.L.lt_dyn["cpu", DYN2](rl_M)
        var D_v = scratch.D.lt_dyn["cpu", DYN2](rl_NV)
        var T_v = m.trees.lt_dyn["cpu", DYN1](rl_TREE)
        # Tree-ordered when the parser built the table, dense otherwise —
        # see the note above `_ldl_factor_tree_env`.
        var use_tree = Int(m.meta.data[MODEL_META_IDX_NTREE]) > 0
        var P_v = m.dof_parentid.lt_dyn["cpu", DYN1](
            rl1(dm.get_nv() if dm.get_nv() > 0 else 1)
        )
        for e in range(BATCH):
            if use_tree:
                _ldl_factor_tree_env(e, dm, M_v, L_v, D_v, P_v)
            else:
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
    mut m: Model[DTYPE, D],
    mut scratch: DynamicsScratch[DTYPE, D, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """`qacc_ws = M^-1 fnet` via L/D (owned scratch), both targets."""
    comptime L_M = Layout.row_major(BATCH, D.NV * D.NV)
    comptime L_NV = Layout.row_major(BATCH, D.NV)
    comptime L_TREE = Layout.row_major(D.NV * MODEL_TREE_SIZE)

    comptime if target == "cpu":
        var dm = scratch.dims
        var rl_M = rl2(BATCH, dm.get_nv() * dm.get_nv())
        var rl_NV = rl2(BATCH, dm.get_nv())
        var rl_TREE = rl1(dm.get_nv() * MODEL_TREE_SIZE)
        var L_v = scratch.L.lt_dyn["cpu", DYN2](rl_M)
        var D_v = scratch.D.lt_dyn["cpu", DYN2](rl_NV)
        var b_v = scratch.fnet.lt_dyn["cpu", DYN2](rl_NV)
        var x_v = scratch.qacc_ws.lt_dyn["cpu", DYN2](rl_NV)
        var T_v = m.trees.lt_dyn["cpu", DYN1](rl_TREE)
        var use_tree = Int(m.meta.data[MODEL_META_IDX_NTREE]) > 0
        var P_v = m.dof_parentid.lt_dyn["cpu", DYN1](
            rl1(dm.get_nv() if dm.get_nv() > 0 else 1)
        )
        for e in range(BATCH):
            if use_tree:
                _ldl_solve_tree_env(e, dm, L_v, D_v, b_v, x_v, P_v)
            else:
                _ldl_solve_env(e, dm, L_v, D_v, b_v, x_v, T_v)
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + LDL_TPB - 1) // LDL_TPB
        c.enqueue_function[_ldl_solve_fields_kernel[DTYPE, D.NV, BATCH]](
            scratch.L.lt["gpu", L_M](),
            scratch.D.lt["gpu", L_NV](),
            scratch.fnet.lt["gpu", L_NV](),
            scratch.qacc_ws.lt["gpu", L_NV](),
            m.trees.lt["gpu", L_TREE](),
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

    # ⚠⚠ COLUMN j OF `M^-1` IS ZERO OUTSIDE j's TREE, AND TODAY'S CODE ALREADY
    # COMPUTES IT AS EXACTLY ZERO. Forward substitution gives `y[i] = 0` for
    # every i before the block by induction (`e` is 0 there and `L*0 = 0`);
    # back-substitution then gives `col[i] = 0` because `L[k, i]` with k in the
    # block and i outside it is a cross-tree entry, which `M` — and so `L` —
    # holds as an exact `0.0`. So restricting the three loops to the block is
    # bit-exact, and the explicit zeroing below reproduces what the dense
    # version wrote.
    var bb = _dof_block(trees, nv, j)
    var b0 = bb[0]
    var b1 = bb[1]

    # ⚠ `col` MUST BE ZEROED OVER THE WHOLE `nv`, and `Scratch(uninitialized=0)`
    # DOES NOT ZERO — the sibling `e` below is cleared by an explicit loop for
    # the same reason. The dense version wrote every row of the column, so
    # nothing downstream had to care; a block-restricted write that skipped the
    # rest would leave `m_inv` holding the PREVIOUS STEP's values there, since
    # it is reused scratch.
    for i in range(nv):
        e[i] = 0
        col[i] = 0
    e[j] = 1

    var y = Scratch[L.element_type, V_CAP](nv, uninitialized=0)
    for i in range(b0, b1):
        var s = e[i]
        for k in range(b0, i):
            s = s - L[env, i * nv + k] * y[k]
        y[i] = s

    var z = Scratch[L.element_type, V_CAP](nv, uninitialized=0)
    for i in range(b0, b1):
        var d_i = D[env, i]
        if d_i > 1e-14 or d_i < -1e-14:
            z[i] = y[i] / d_i
        else:
            z[i] = 0

    for i in range(b1 - 1, b0 - 1, -1):
        var s = z[i]
        for k in range(i + 1, b1):
            s = s - L[env, k * nv + i] * col[k]
        col[i] = s

    # The FULL column, so the off-block zeros land in `m_inv` explicitly.
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
    trees: LayoutTensor[
        DTYPE, Layout.row_major(NV * MODEL_TREE_SIZE), MutAnyOrigin
    ],
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
    trees: LayoutTensor[
        DTYPE, Layout.row_major(NV * MODEL_TREE_SIZE), MutAnyOrigin
    ],
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
    # ⚠ FLAT — `[t*MODEL_TREE_SIZE + col]`, the single spelling shared with
    # `newton_blocks` and `newton_solve`. A 2-D `LayoutTensor` given ONE index
    # returns a ROW, so the two forms are not interchangeable.
    comptime L_TREE = Layout.row_major(D.NV * MODEL_TREE_SIZE)

    comptime if target == "cpu":
        var dm = scratch.dims
        var rl_M = rl2(BATCH, dm.get_nv() * dm.get_nv())
        var rl_NV = rl2(BATCH, dm.get_nv())
        var rl_TREE = rl1(dm.get_nv() * MODEL_TREE_SIZE)
        var L_v = scratch.L.lt_dyn["cpu", DYN2](rl_M)
        var D_v = scratch.D.lt_dyn["cpu", DYN2](rl_NV)
        var mi_v = scratch.m_inv.lt_dyn["cpu", DYN2](rl_M)
        var T_v = m.trees.lt_dyn["cpu", DYN1](rl_TREE)
        var use_tree = Int(m.meta.data[MODEL_META_IDX_NTREE]) > 0
        var P_v = m.dof_parentid.lt_dyn["cpu", DYN1](
            rl1(dm.get_nv() if dm.get_nv() > 0 else 1)
        )
        for e in range(BATCH):
            if use_tree:
                _m_inv_tree_env(e, dm, L_v, D_v, mi_v, P_v)
            else:
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

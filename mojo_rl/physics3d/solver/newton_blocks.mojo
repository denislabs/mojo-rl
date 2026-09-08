"""The Newton Hessian's DIAGONAL BLOCKS — `H = M + sum D*J^T J`, segmented.

PN2a. This module computes the partition and NOTHING USES IT YET.

⚠⚠ WHY THIS IS NOT JUST `Model.trees`. `M`'s blocks are the kinematic trees and
that is a MODEL-TIME fact (`gpu/constants.MODEL_TREE_SIZE`). `H`'s are not: a
constraint row couples every tree its Jacobian touches, and which rows exist is
a RUNTIME property of the step. A contact between the arm and a prop merges two
trees that `Model.trees` lists apart.

WHAT IT IS WORTH, MEASURED. P0 on `so101_park_k9` (RTX 5090):

    newton  33.3 of 47.6 ms/step   70% of GPU time, 78% of the parked-slot cost
    nv = 60, ncon = 0, nefc = 6 — six FRICTION_DOF rows, all on dofs 0..5
    trees containing a constraint row: [0]  of 10

Nine of the ten trees carry no row at all, so their blocks of `H` are their
blocks of `M` — which P1's classifier already calls COMPACT, i.e. diagonal —
and they are being folded into one dense 60x60 Cholesky. One 6^3 plus nine
diagonals is 270 operations against 216,000.

⚠ SEGMENTS, NOT COMPONENTS, AND THE DIFFERENCE IS DELIBERATE. A connected
component can be non-contiguous in dof space — the arm (tree 0) gripping the
fourth prop (tree 3) is the component {0, 3}. Factoring a non-contiguous index
set needs a permutation, an indirection in the innermost loop, and a second
addressing scheme to get wrong. Instead a row's trees are merged as a SPAN:
`{0, 3}` becomes the segment `trees[0..3]`, which sweeps up trees 1 and 2 as
well. That is:

  * always CONTIGUOUS, so the Cholesky change is a loop bound and nothing else;
  * always a SUPERSET of the true coupling, so it can never drop a nonzero —
    the extra entries it factors are exact zeros;
  * still the whole win where it matters: at k=9 an arm holding one prop gives
    one segment of at most 24 dofs and six untouched blocks of 6, not 60.

⚠ IT USES `Je`'s SPARSITY, NOT THE ROW STATES. A row's state flips between
iterations (`SROW_QUADRATIC` or not), so a partition derived from the ACTIVE
set would have to be rebuilt every iteration and would change under the
factorisation. `Je` is built once, before the loop; keying on it gives one
partition for the whole solve that is a superset of every iteration's coupling.

⚠ A DEGENERATE TABLE MEANS ONE SEGMENT, NEVER ZERO. `ntree == 0` (a `Model`
built without the parser leaves `trees` zeroed) and any table that does not
tile `[0, nv)` exactly both fall back to a single segment spanning every dof —
which is today's behaviour, bit for bit.
"""

from ..fields.scratch import Scratch
from layout import Layout, LayoutTensor
from max.gpu.memory import AddressSpace

from ..gpu.constants import (
    MODEL_TREE_SIZE,
    TREE_IDX_DOF_ADR,
    TREE_IDX_DOF_NUM,
)


@always_inline
def build_dof_segments[
    DTYPE: DType,
    LT: Layout,
    LJ: Layout,
    LS: Layout,
    # ⚠ THE OPERANDS LIVE IN DIFFERENT ADDRESS SPACES. In the blocked kernel
    # `Je` is SHARED or GLOBAL depending on whether it fit (`JE_AS`), the
    # segment arrays are threadgroup memory, and `trees` is a plain model
    # tensor. Defaults keep a CPU caller — and the gate — writing none of this.
    T_AS: AddressSpace = AddressSpace.GENERIC,
    J_AS: AddressSpace = AddressSpace.GENERIC,
    S_AS: AddressSpace = AddressSpace.GENERIC,
](
    nv: Int,
    ntree: Int,
    num_edges: Int,
    trees: LayoutTensor[DTYPE, LT, MutAnyOrigin, address_space=T_AS],
    Je: LayoutTensor[DTYPE, LJ, MutAnyOrigin, address_space=J_AS],
    seg_start: LayoutTensor[DTYPE, LS, MutAnyOrigin, address_space=S_AS],
    seg_end: LayoutTensor[DTYPE, LS, MutAnyOrigin, address_space=S_AS],
) -> Int:
    """Per-dof segment bounds for `H`. Returns the segment count.

    `seg_start[i]` / `seg_end[i]` are the half-open dof range of the segment
    containing dof `i`, so a Cholesky restricts to `[seg_start[j], j)` and
    `[j+1, seg_end[j])` and changes nothing else.

    ⚠ EVERY OPERAND IS FLAT. `trees` is `[t*MODEL_TREE_SIZE + col]` and `Je` is
    `[e*nv + i]` — matching `Je_sh` in the blocked kernel. A 2-D `LayoutTensor`
    given ONE index returns a ROW rather than an element, which is a mismatch
    this tree has already paid for once (`fields/model.mojo`'s `L_CAM` note).
    """
    return build_dof_segments_p[
        DTYPE, T_AS=T_AS, J_AS=J_AS, S_AS=S_AS
    ](nv, ntree, num_edges, trees.ptr, Je.ptr, seg_start.ptr, seg_end.ptr)


@always_inline
def seg_one_segment_p[
    SO: MutOrigin,
    EO: MutOrigin, //,
    DTYPE: DType,
    S_AS: AddressSpace = AddressSpace.GENERIC,
](
    nv: Int,
    seg_start: Pointer[Scalar[DTYPE], SO, address_space=S_AS],
    seg_end: Pointer[Scalar[DTYPE], EO, address_space=S_AS],
) -> Int:
    """The degenerate partition: one segment spanning every dof. Returns 1."""
    for i in range(nv):
        seg_start[i] = Scalar[DTYPE](0)
        seg_end[i] = Scalar[DTYPE](nv)
    return 1


@always_inline
def seg_phase_trees_p[
    TO: MutOrigin,
    SO: MutOrigin,
    EO: MutOrigin, //,
    DTYPE: DType,
    T_AS: AddressSpace = AddressSpace.GENERIC,
    S_AS: AddressSpace = AddressSpace.GENERIC,
](
    nv: Int,
    ntree: Int,
    trees: Pointer[Scalar[DTYPE], TO, address_space=T_AS],
    seg_start: Pointer[Scalar[DTYPE], SO, address_space=S_AS],
    seg_end: Pointer[Scalar[DTYPE], EO, address_space=S_AS],
) -> Int:
    """PHASE A of the segment build: validate the tree table, write the
    tree of every dof into `seg_start[i]` and clear the per-tree merge flag
    `seg_end[t]` for `t < nt`. Returns `nt`, the number of trees, or 0 when
    the table is degenerate — the caller then writes ONE segment.

    ⚠ THE THREE PHASES ARE ONE RULE (2026-09-08). `build_dof_segments_p`
    below is A, then B (`seg_edge_span_dense_p` + `seg_span_mark_p`) once
    per edge, then C (`seg_assemble_p`); the blocked Newton kernel runs A on
    thread 0, B on ONE THREAD PER EDGE and C on thread 0, around two
    barriers. B was `num_edges * nv` dependent global loads of the spilled
    `Je` on one thread — the largest once-per-solve term the serial probe
    found on Apple (~5 ms/step at k=9, the same size as the whole per-block
    solve); per edge it is `nv` loads on each of `num_edges` threads. The
    marks are set-valued (every writer writes 1), so the order edges are
    marked in, or concurrently, cannot change the partition."""
    if ntree <= 0 or nv <= 0:
        return 0
    var covered = 0
    var nt = 0
    for t in range(ntree):
        var adr = Int(trees[t * MODEL_TREE_SIZE + TREE_IDX_DOF_ADR])
        var num = Int(trees[t * MODEL_TREE_SIZE + TREE_IDX_DOF_NUM])
        if num <= 0:
            break
        if adr != covered or adr + num > nv:
            return 0
        for i in range(adr, adr + num):
            seg_start[i] = Scalar[DTYPE](t)
        covered = adr + num
        nt = t + 1
    if covered != nv or nt <= 0:
        return 0
    for t in range(nt):
        seg_end[t] = Scalar[DTYPE](0)
    return nt


@always_inline
def seg_edge_span_dense_p[
    JO: MutOrigin,
    SO: MutOrigin, //,
    DTYPE: DType,
    J_AS: AddressSpace = AddressSpace.GENERIC,
    S_AS: AddressSpace = AddressSpace.GENERIC,
](
    e: Int,
    nv: Int,
    Je: Pointer[Scalar[DTYPE], JO, address_space=J_AS],
    seg_start: Pointer[Scalar[DTYPE], SO, address_space=S_AS],
) -> Tuple[Int, Int]:
    """PHASE B, one edge: the lowest and highest tree (as `seg_start` numbers
    them after phase A) that row `e` of the dense `Je` touches; `(-1, -1)`
    for an all-zero row. Reads only."""
    var lo = -1
    var hi = -1
    for i in range(nv):
        if Je[e * nv + i] != 0:
            var t = Int(seg_start[i])
            if lo < 0 or t < lo:
                lo = t
            if t > hi:
                hi = t
    return (lo, hi)


@always_inline
def seg_span_mark_p[
    EO: MutOrigin, //,
    DTYPE: DType,
    S_AS: AddressSpace = AddressSpace.GENERIC,
](
    lo: Int,
    hi: Int,
    seg_end: Pointer[Scalar[DTYPE], EO, address_space=S_AS],
):
    """PHASE B, the mark: trees `lo..hi-1` merge with their successor. A
    negative `lo` (an all-zero row) marks nothing."""
    if lo < 0:
        return
    for t in range(lo, hi):
        seg_end[t] = Scalar[DTYPE](1)


@always_inline
def seg_assemble_p[
    TO: MutOrigin,
    SO: MutOrigin,
    EO: MutOrigin, //,
    DTYPE: DType,
    T_AS: AddressSpace = AddressSpace.GENERIC,
    S_AS: AddressSpace = AddressSpace.GENERIC,
](
    nv: Int,
    nt: Int,
    trees: Pointer[Scalar[DTYPE], TO, address_space=T_AS],
    seg_start: Pointer[Scalar[DTYPE], SO, address_space=S_AS],
    seg_end: Pointer[Scalar[DTYPE], EO, address_space=S_AS],
) -> Int:
    """PHASE C: walk the trees from the last, gather each run of merged
    trees into one contiguous segment and write its `[d0, d1)` into every
    dof's `seg_start`/`seg_end`. Returns the segment count."""
    var nseg = 0
    var t1 = nt - 1
    while t1 >= 0:
        var t0 = t1
        while t0 - 1 >= 0 and Int(seg_end[t0 - 1]) == 1:
            t0 -= 1
        var d0 = Int(trees[t0 * MODEL_TREE_SIZE + TREE_IDX_DOF_ADR])
        var d1 = Int(trees[t1 * MODEL_TREE_SIZE + TREE_IDX_DOF_ADR]) + Int(
            trees[t1 * MODEL_TREE_SIZE + TREE_IDX_DOF_NUM]
        )
        for i in range(d0, d1):
            seg_start[i] = Scalar[DTYPE](d0)
            seg_end[i] = Scalar[DTYPE](d1)
        nseg += 1
        t1 = t0 - 1
    return nseg


@always_inline
def build_dof_segments_p[
    TO: MutOrigin,
    JO: MutOrigin,
    SO: MutOrigin,
    EO: MutOrigin, //,
    DTYPE: DType,
    T_AS: AddressSpace = AddressSpace.GENERIC,
    J_AS: AddressSpace = AddressSpace.GENERIC,
    S_AS: AddressSpace = AddressSpace.GENERIC,
    # `SPARSE`: the caller already holds each row's nonzero dof list
    # (`je_n[e]` entries at `je_ix[e*nv ..]`, ascending) — the CPU Newton
    # does — so a row's tree range is its first and last entry, not a scan
    # of all `nv` (PERFORMANCE.md §13.24: this second scan was half of the
    # Newton's `setup` on dog). Same `lo`/`hi`, same segments, bit-exact.
    SPARSE: Bool = False,
    N_CAP: Int = 1,
    IX_CAP: Int = 1,
](
    nv: Int,
    ntree: Int,
    num_edges: Int,
    trees: Pointer[Scalar[DTYPE], TO, address_space=T_AS],
    Je: Pointer[Scalar[DTYPE], JO, address_space=J_AS],
    seg_start: Pointer[Scalar[DTYPE], SO, address_space=S_AS],
    seg_end: Pointer[Scalar[DTYPE], EO, address_space=S_AS],
    je_n: Scratch[Int, N_CAP] = Scratch[Int, N_CAP](1, fill=0),
    je_ix: Scratch[Int, IX_CAP] = Scratch[Int, IX_CAP](1, fill=0),
) -> Int:
    """Pointer form of `build_dof_segments` — THE body; the `LayoutTensor`
    spelling above owns no arithmetic and delegates here.

    It exists so the per-env CPU solver, whose rows live in a `Scratch`, can
    share the one implementation with the blocked kernel, whose rows live in
    threadgroup memory — the same split `chol_solve_seg` / `chol_solve_seg_p`
    already makes, for the same reason: a rule written twice drifts.
    """

    var nt = seg_phase_trees_p[DTYPE, T_AS=T_AS, S_AS=S_AS](
        nv, ntree, trees, seg_start, seg_end
    )
    if nt <= 0:
        return seg_one_segment_p[DTYPE, S_AS=S_AS](nv, seg_start, seg_end)
    for e in range(num_edges):
        var lo = -1
        var hi = -1
        comptime if SPARSE:
            var n_e = je_n[e]
            if n_e > 0:
                lo = Int(seg_start[je_ix[e * nv]])
                hi = Int(seg_start[je_ix[e * nv + n_e - 1]])
        else:
            var span = seg_edge_span_dense_p[DTYPE, J_AS=J_AS, S_AS=S_AS](
                e, nv, Je, seg_start
            )
            lo = span[0]
            hi = span[1]
        seg_span_mark_p[DTYPE, S_AS=S_AS](lo, hi, seg_end)
    return seg_assemble_p[DTYPE, T_AS=T_AS, S_AS=S_AS](
        nv, nt, trees, seg_start, seg_end
    )

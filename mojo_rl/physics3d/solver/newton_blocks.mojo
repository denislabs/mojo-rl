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

from layout import Layout, LayoutTensor

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
](
    nv: Int,
    ntree: Int,
    num_edges: Int,
    trees: LayoutTensor[DTYPE, LT, MutAnyOrigin],
    Je: LayoutTensor[DTYPE, LJ, MutAnyOrigin],
    seg_start: LayoutTensor[DTYPE, LS, MutAnyOrigin],
    seg_end: LayoutTensor[DTYPE, LS, MutAnyOrigin],
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

    @parameter
    @always_inline
    def one_segment() -> Int:
        for i in range(nv):
            seg_start[i] = Scalar[DTYPE](0)
            seg_end[i] = Scalar[DTYPE](nv)
        return 1

    if ntree <= 0 or nv <= 0:
        return one_segment()

    # ── tree id per dof, parked in `seg_start` ───────────────────────────
    #
    # ⚠ AND VALIDATED WHILE BUILDING. The table must tile `[0, nv)` exactly:
    # a gap would leave a dof with no tree and an overlap would give it two,
    # and either way a segment bound computed from it is meaningless. Rather
    # than trust it, walk it and fall back on anything unexpected.
    var covered = 0
    var nt = 0
    for t in range(ntree):
        var adr = Int(trees[t * MODEL_TREE_SIZE + TREE_IDX_DOF_ADR])
        var num = Int(trees[t * MODEL_TREE_SIZE + TREE_IDX_DOF_NUM])
        # Self-terminating: rows past `ntree` are (0, 0, 0).
        if num <= 0:
            break
        if adr != covered or adr + num > nv:
            return one_segment()
        for i in range(adr, adr + num):
            seg_start[i] = Scalar[DTYPE](t)
        covered = adr + num
        nt = t + 1
    if covered != nv or nt <= 0:
        return one_segment()

    # ── merge flags, parked in `seg_end`: does tree t join tree t+1? ──────
    for t in range(nt):
        seg_end[t] = Scalar[DTYPE](0)
    for e in range(num_edges):
        var lo = -1
        var hi = -1
        for i in range(nv):
            if Je[e * nv + i] != 0:
                var t = Int(seg_start[i])
                if lo < 0 or t < lo:
                    lo = t
                if t > hi:
                    hi = t
        # A row that touches nothing couples nothing. Not a defect: a limit
        # row whose Jacobian is a single dof still has lo == hi.
        if lo < 0:
            continue
        for t in range(lo, hi):
            seg_end[t] = Scalar[DTYPE](1)

    # ── runs of merged trees -> per-dof bounds, WALKED BACKWARDS ─────────
    #
    # ⚠⚠ REVERSE ORDER IS A CORRECTNESS REQUIREMENT, NOT A STYLE CHOICE.
    # `seg_end[0 .. nt)` currently holds the merge flags indexed by TREE,
    # while the writes below are indexed by DOF — and dof indices start at 0
    # too. Forwards, the very first run (trees 0..0, dofs 0..5 on the park
    # scene) writes `seg_end[0..5] = 6` and destroys the flags for trees 1..5
    # before they are read: every later tree then reads `6 != 1` and is
    # silently treated as unmerged. The bug produces a plausible partition —
    # it would even be RIGHT on any scene with no coupling — which is exactly
    # the kind that survives a weak gate.
    #
    # Backwards it cannot happen. A run ending at tree `t1` starts at tree
    # `t0` and writes dofs from `d0` upwards, and every tree holds at least
    # one dof, so `d0 >= t0`. The flags still to be read live at indices
    # `<= t0 - 2`, which is strictly below anything this run writes.
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

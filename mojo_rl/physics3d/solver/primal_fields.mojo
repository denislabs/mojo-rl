"""Shared primal-solver fragments over InlineArray working sets (Stage-S
refactor). These are the reusable leaf computations extracted VERBATIM from
the inlined fields-Newton kernel (`solver/newton_solve_fields.mojo`) so the
CG primal solver (`cg_solve_fields`) can reuse them instead of copying the
Newton kernel body.

All helpers are `@always_inline` and operate on the per-env InlineArray
working set the primal solvers already build (Je/De/bias_e/qacc/...), so
inlining them at the Newton call sites is codegen- (and thus bit-) identical
to the previous inline code — the Newton golden gates re-validate this after
each extraction.

PYRAMIDAL cone only for now (the pyramidal primal path is what the Walker2D
Newton gates and the HalfCheetah CG gate exercise; the elliptic path is
extracted separately once it has fields-gate coverage)."""


@always_inline
def pyramidal_edge_forces[
    DTYPE: DType, NV: Int, ME: Int, V_SIZE: Int
](
    num_edges: Int,
    Je: InlineArray[Scalar[DTYPE], ME * V_SIZE],
    De: InlineArray[Scalar[DTYPE], ME],
    bias_e: InlineArray[Scalar[DTYPE], ME],
    qacc: InlineArray[Scalar[DTYPE], V_SIZE],
    mut jar: InlineArray[Scalar[DTYPE], ME],
    mut force: InlineArray[Scalar[DTYPE], ME],
    mut qfrc: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """Pyramidal edge constraint forces given the current qacc (verbatim from
    the fields-Newton inline body):

        qfrc = 0
        for each active edge e:
            jar[e]   = bias_e[e] + Je[e]·qacc
            force[e] = -De[e]*jar[e]   if jar[e] < 0 else 0   (unilateral)
            qfrc    += Je[e] * force[e]

    Je is row-major [ME, NV] (stride NV; array sized ME*V_SIZE with
    V_SIZE = max(NV,1)). Writes jar/force (per-edge) and qfrc (per-dof)."""
    for i in range(NV):
        qfrc[i] = Scalar[DTYPE](0)
    for e_idx in range(num_edges):
        jar[e_idx] = bias_e[e_idx]
        for i in range(NV):
            jar[e_idx] += Je[e_idx * NV + i] * qacc[i]
        if jar[e_idx] >= Scalar[DTYPE](0):
            force[e_idx] = Scalar[DTYPE](0)
        else:
            force[e_idx] = -De[e_idx] * jar[e_idx]
        for i in range(NV):
            qfrc[i] += Je[e_idx * NV + i] * force[e_idx]

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


@always_inline
def pyramidal_linesearch[
    DTYPE: DType,
    NV: Int,
    ME: Int,
    V_SIZE: Int,
    LINESEARCH_ITER: Int,
    PRIMAL_MINVAL: Float64,
](
    num_edges: Int,
    Je: InlineArray[Scalar[DTYPE], ME * V_SIZE],
    De: InlineArray[Scalar[DTYPE], ME],
    search: InlineArray[Scalar[DTYPE], V_SIZE],
    Mv: InlineArray[Scalar[DTYPE], V_SIZE],
    Ma: InlineArray[Scalar[DTYPE], V_SIZE],
    f_smooth: InlineArray[Scalar[DTYPE], V_SIZE],
    qacc: InlineArray[Scalar[DTYPE], V_SIZE],
    qacc_smooth: InlineArray[Scalar[DTYPE], V_SIZE],
    jar: InlineArray[Scalar[DTYPE], ME],
) -> Scalar[DTYPE]:
    """Analytical Newton/CG line-search for the pyramidal primal cost (verbatim
    from the fields-Newton inline body, matching CPU `primal_linesearch_with_D`).

    Given a search direction and its images (Mv = M·search, and Jv_e = Je·search
    computed internally), returns the step length `alpha` that minimizes the
    primal Gauss + edge-constraint cost along the line, with analytical initial
    step `alpha = -d1/d2` at alpha=0 then cost-based halving. Direction-agnostic:
    the Newton chol step and the CG conjugate step both feed the same math, so
    both primal solvers share this leaf.

    Inputs are the per-edge residual `jar[e] = bias_e[e] + Je[e]·qacc` at the
    current qacc, the smooth force/accel state (Ma/f_smooth/qacc/qacc_smooth),
    and the direction images. Returns 0 when the direction is not a descent
    direction (p0_d1 >= 0)."""
    # Precompute Jv_e = Je · search for each edge.
    var Jv_e = InlineArray[Scalar[DTYPE], ME](uninitialized=True)
    for e_idx in range(num_edges):
        Jv_e[e_idx] = Scalar[DTYPE](0)
        for i in range(NV):
            Jv_e[e_idx] += Je[e_idx * NV + i] * search[i]

    var gauss_a: Scalar[DTYPE] = 0
    var gauss_b: Scalar[DTYPE] = 0
    for i in range(NV):
        gauss_a += Mv[i] * search[i]
        gauss_b += (Ma[i] - f_smooth[i]) * search[i]

    # Evaluate d1, d2 at alpha=0
    var p0_d1 = gauss_b
    var p0_d2 = gauss_a
    for e_idx in range(num_edges):
        if jar[e_idx] < Scalar[DTYPE](0):
            p0_d1 += De[e_idx] * jar[e_idx] * Jv_e[e_idx]
            p0_d2 += De[e_idx] * Jv_e[e_idx] * Jv_e[e_idx]
    if p0_d2 < Scalar[DTYPE](PRIMAL_MINVAL):
        p0_d2 = Scalar[DTYPE](PRIMAL_MINVAL)

    var alpha: Scalar[DTYPE] = 0
    if p0_d1 < Scalar[DTYPE](0):
        # Analytical initial alpha, then cost-based halving
        alpha = -p0_d1 / p0_d2

        # Compute old cost for acceptance check
        # Gauss cost = 0.5*(Ma-f_smooth)·(qacc-qacc_smooth)
        var old_cost: Scalar[DTYPE] = 0
        for i in range(NV):
            old_cost += (
                Scalar[DTYPE](0.5)
                * (Ma[i] - f_smooth[i])
                * (qacc[i] - qacc_smooth[i])
            )
        for e_idx in range(num_edges):
            if jar[e_idx] < Scalar[DTYPE](0):
                old_cost += (
                    Scalar[DTYPE](0.5) * De[e_idx] * jar[e_idx] * jar[e_idx]
                )

        # Try alpha, halve if cost doesn't decrease
        for _ in range(LINESEARCH_ITER):
            var trial_cost: Scalar[DTYPE] = 0
            for i in range(NV):
                var qa_t = qacc[i] + alpha * search[i]
                var Ma_t = Ma[i] + alpha * Mv[i]
                trial_cost += (
                    Scalar[DTYPE](0.5)
                    * (Ma_t - f_smooth[i])
                    * (qa_t - qacc_smooth[i])
                )
            for e_idx in range(num_edges):
                var jar_t = jar[e_idx] + alpha * Jv_e[e_idx]
                if jar_t < Scalar[DTYPE](0):
                    trial_cost += (
                        Scalar[DTYPE](0.5) * De[e_idx] * jar_t * jar_t
                    )
            if trial_cost <= old_cost:
                break
            alpha *= Scalar[DTYPE](0.5)

    return alpha

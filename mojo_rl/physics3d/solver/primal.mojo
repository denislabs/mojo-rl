"""Shared primal-solver fragments over `Scratch` working sets (Stage-S
refactor). These are the reusable leaf computations extracted VERBATIM from
the inlined fields-Newton kernel (`solver/newton_solve.mojo`).

All helpers are `@always_inline` and operate on the per-env `Scratch`
working set the primal solvers already build (Je/De/bias_e/qacc/...), so
inlining them at the Newton call sites is codegen- (and thus bit-) identical
to the previous inline code — the Newton golden gates re-validate this after
each extraction.

PYRAMIDAL cone only (the pyramidal primal path is what the Walker2D Newton
gates exercise; the elliptic path is inlined in `newton_solve.mojo`).

ROW KINDS. The row list is no longer homogeneous. Contact edges and joint
limits are ONE-SIDED (`SROW_LIMIT`: inactive once jar >= 0), while
dry-friction dof rows are BOX-clamped to +-frictionloss (`SROW_FRICTION`),
which has a LINEAR regime where the force is constant and the row contributes
NOTHING to the Hessian. Classifying with `force > 0` — as this file did while
every row was one-sided — silently mis-handles a box row: a saturated negative
row has force > 0 and would wrongly add curvature. Callers therefore read the
per-row `state_e` that `pyramidal_edge_forces` now writes."""

from ..fields.scratch import Scratch
from ..constraints.scalar_rows import (
    scalar_row_state,
    scalar_row_force,
    scalar_row_cost,
    SROW_QUADRATIC,
)


@always_inline
def pyramidal_edge_forces[
    DTYPE: DType, E_CAP: Int, V_CAP: Int
](
    num_edges: Int,
    Je: Scratch[Scalar[DTYPE], E_CAP * V_CAP],
    De: Scratch[Scalar[DTYPE], E_CAP],
    bias_e: Scratch[Scalar[DTYPE], E_CAP],
    kind_e: Scratch[Int, E_CAP],
    R_e: Scratch[Scalar[DTYPE], E_CAP],
    floss_e: Scratch[Scalar[DTYPE], E_CAP],
    qacc: Scratch[Scalar[DTYPE], V_CAP],
    mut jar: Scratch[Scalar[DTYPE], E_CAP],
    mut force: Scratch[Scalar[DTYPE], E_CAP],
    mut state_e: Scratch[Int, E_CAP],
    mut qfrc: Scratch[Scalar[DTYPE], V_CAP],
    nv: Int,
):
    """Primal row forces given the current qacc.

        qfrc = 0
        for each row e:
            jar[e]   = bias_e[e] + Je[e]·qacc
            state[e] = branch(kind_e[e], jar[e], R_e[e], floss_e[e])
            force[e] = f(state[e])       one-sided or box, see scalar_rows
            qfrc    += Je[e] * force[e]

    Je is row-major [num_edges, nv], STRIDE `nv` — the live dof count, not
    `V_CAP`. The array is sized `E_CAP * V_CAP`, and on the static leg the cap
    and the stride are the same integer, which is why a mix-up here survives
    every gate in the tree; see `tests/physics3d/test_cholesky_both_legs.mojo`.
    Writes jar/force/state_e (per-row) and qfrc (per-dof)."""
    for i in range(nv):
        qfrc[i] = Scalar[DTYPE](0)
    for e_idx in range(num_edges):
        jar[e_idx] = bias_e[e_idx]
        for i in range(nv):
            jar[e_idx] += Je[e_idx * nv + i] * qacc[i]
        var st = scalar_row_state[DTYPE](
            kind_e[e_idx], jar[e_idx], R_e[e_idx], floss_e[e_idx]
        )
        state_e[e_idx] = st
        force[e_idx] = scalar_row_force[DTYPE](
            st, jar[e_idx], De[e_idx], floss_e[e_idx]
        )
        for i in range(nv):
            qfrc[i] += Je[e_idx * nv + i] * force[e_idx]


@always_inline
def pyramidal_linesearch[
    DTYPE: DType,
    E_CAP: Int,
    V_CAP: Int,
    LINESEARCH_ITER: Int,
    PRIMAL_MINVAL: Float64,
](
    num_edges: Int,
    Je: Scratch[Scalar[DTYPE], E_CAP * V_CAP],
    De: Scratch[Scalar[DTYPE], E_CAP],
    kind_e: Scratch[Int, E_CAP],
    R_e: Scratch[Scalar[DTYPE], E_CAP],
    floss_e: Scratch[Scalar[DTYPE], E_CAP],
    search: Scratch[Scalar[DTYPE], V_CAP],
    Mv: Scratch[Scalar[DTYPE], V_CAP],
    Ma: Scratch[Scalar[DTYPE], V_CAP],
    f_smooth: Scratch[Scalar[DTYPE], V_CAP],
    qacc: Scratch[Scalar[DTYPE], V_CAP],
    qacc_smooth: Scratch[Scalar[DTYPE], V_CAP],
    jar: Scratch[Scalar[DTYPE], E_CAP],
    nv: Int,
) -> Scalar[DTYPE]:
    """Analytical Newton/CG line-search for the pyramidal primal cost (matching
    CPU `primal_linesearch_with_D`).

    Given a search direction and its images (Mv = M·search, and Jv_e = Je·search
    computed internally), returns the step length `alpha` that minimizes the
    primal Gauss + row-constraint cost along the line, with analytical initial
    step `alpha = -d1/d2` at alpha=0 then cost-based halving. Direction-agnostic:
    the Newton chol step and the CG conjugate step both feed the same math.

    Inputs are the per-row residual `jar[e] = bias_e[e] + Je[e]·qacc` at the
    current qacc, the smooth force/accel state (Ma/f_smooth/qacc/qacc_smooth),
    and the direction images. Returns 0 when the direction is not a descent
    direction (p0_d1 >= 0)."""
    # Precompute Jv_e = Je · search for each row.
    var Jv_e = Scratch[Scalar[DTYPE], E_CAP](
        num_edges, uninitialized=Scalar[DTYPE](0)
    )
    for e_idx in range(num_edges):
        Jv_e[e_idx] = Scalar[DTYPE](0)
        for i in range(nv):
            Jv_e[e_idx] += Je[e_idx * nv + i] * search[i]

    var gauss_a: Scalar[DTYPE] = 0
    var gauss_b: Scalar[DTYPE] = 0
    for i in range(nv):
        gauss_a += Mv[i] * search[i]
        gauss_b += (Ma[i] - f_smooth[i]) * search[i]

    # Evaluate d1, d2 at alpha=0. d(cost)/dalpha = -f*Jv in every state; the
    # second derivative is D*Jv^2 only where the row is quadratic.
    var p0_d1 = gauss_b
    var p0_d2 = gauss_a
    for e_idx in range(num_edges):
        var st = scalar_row_state[DTYPE](
            kind_e[e_idx], jar[e_idx], R_e[e_idx], floss_e[e_idx]
        )
        var f = scalar_row_force[DTYPE](
            st, jar[e_idx], De[e_idx], floss_e[e_idx]
        )
        p0_d1 += -f * Jv_e[e_idx]
        if st == SROW_QUADRATIC:
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
        for i in range(nv):
            old_cost += (
                Scalar[DTYPE](0.5)
                * (Ma[i] - f_smooth[i])
                * (qacc[i] - qacc_smooth[i])
            )
        for e_idx in range(num_edges):
            var st = scalar_row_state[DTYPE](
                kind_e[e_idx], jar[e_idx], R_e[e_idx], floss_e[e_idx]
            )
            old_cost += scalar_row_cost[DTYPE](
                st, jar[e_idx], De[e_idx], R_e[e_idx], floss_e[e_idx]
            )

        # Try alpha, halve if cost doesn't decrease
        for _ in range(LINESEARCH_ITER):
            var trial_cost: Scalar[DTYPE] = 0
            for i in range(nv):
                var qa_t = qacc[i] + alpha * search[i]
                var Ma_t = Ma[i] + alpha * Mv[i]
                trial_cost += (
                    Scalar[DTYPE](0.5)
                    * (Ma_t - f_smooth[i])
                    * (qa_t - qacc_smooth[i])
                )
            for e_idx in range(num_edges):
                var jar_t = jar[e_idx] + alpha * Jv_e[e_idx]
                var st_t = scalar_row_state[DTYPE](
                    kind_e[e_idx], jar_t, R_e[e_idx], floss_e[e_idx]
                )
                trial_cost += scalar_row_cost[DTYPE](
                    st_t, jar_t, De[e_idx], R_e[e_idx], floss_e[e_idx]
                )
            if trial_cost <= old_cost:
                break
            alpha *= Scalar[DTYPE](0.5)

    return alpha

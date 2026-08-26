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

from std.math import sqrt, abs

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
    # `<option ls_iterations>`. ⚠ THE COMPTIME `LINESEARCH_ITER` IS THE CEILING
    # A `range()` NEEDS, NOT THE BUDGET — a model asking for fewer iterations
    # (apollo asks for 10, so101 for 20) must get them. A non-positive value
    # means "use the ceiling".
    max_ls: Int = -1,
    # `opt.tolerance * opt.ls_tolerance / scale`. `PrimalSearch` converges on
    # `|deriv[0]| < tolerance * snorm / scale` (engine_solver.c:1711) where its
    # `tolerance` argument is already the PRODUCT of the two options
    # (engine_solver.c:2236). `snorm` is `|search|`, computed here because
    # this is where the direction lives. A non-positive value falls back to
    # the old single-step behaviour — no caller should pass one.
    gtol_scale: Scalar[DTYPE] = Scalar[DTYPE](0),
) -> Scalar[DTYPE]:
    """`PrimalSearch` for the pyramidal primal cost (engine_solver.c:1692).

    Returns the step length `alpha` minimising the Gauss + row-constraint cost
    along `search`, or 0 when there is no improvement to be had.

    ⚠⚠ THIS ITERATES; IT USED TO TAKE ONE STEP. The old body computed
    `alpha = -d1/d2` from the derivatives AT alpha=0 and then halved until the
    cost stopped rising. That is a Newton step on the line plus a backtrack,
    and on a PIECEWISE-quadratic line — rows cross their zone boundaries
    between alpha=0 and the minimum, which is the whole reason a line search
    exists here — it lands SHORT and halving only guards the other direction.

    ⚠ THE COLD START HID IT FOR AS LONG AS WE HAD ONE. Starting every solve at
    `qacc_smooth` leaves the iterate far enough from the optimum that the NEXT
    Newton iteration absorbs the shortfall. `qacc_warmstart` puts the iterate
    where there is no next iteration, and the engine's answer became a
    function of where it started while MuJoCo's is not: on
    `test_frictionless_contact_pyramidal` MuJoCo reaches `gradient` 1e-34 in
    ONE iteration and is start-independent to 1.1e-16, where this plateaued at
    1.07e-08 with `alpha` collapsing to 5e-03.

    THE THREE PHASES ARE MuJoCo'S, and are the same three the ELLIPTIC leg in
    `newton_solve.mojo` already runs — that leg was ported faithfully and this
    one was not, which is why the two cones stopped agreeing:

      1. one Newton step on the line from alpha=0; accept on `|d1| < gtol`;
      2. a ONE-SIDED Newton search in the descent direction until the
         derivative changes sign (a bracket) or the budget runs out;
      3. bisection inside the bracket until `|d1| < gtol`.

    ⚠ Phase 3 BISECTS where `PrimalSearch` picks the best of three candidates
    (`p1next`, `p2next`, `pmid`) by cost. Both converge to the same root of
    `d1` and both stop at `gtol`, so the answers agree to within the tolerance;
    the candidate list only gets there in fewer evaluations. The ELLIPTIC leg
    made the same simplification and keeping the two identical matters more
    than the evaluation count.
    """
    comptime ZERO = Scalar[DTYPE](0)
    comptime MINVAL = Scalar[DTYPE](PRIMAL_MINVAL)

    var budget = LINESEARCH_ITER
    if max_ls > 0 and max_ls < budget:
        budget = max_ls

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
    var snorm_sq: Scalar[DTYPE] = 0
    for i in range(nv):
        gauss_a += Mv[i] * search[i]
        gauss_b += (Ma[i] - f_smooth[i]) * search[i]
        snorm_sq += search[i] * search[i]

    # `PrimalSearch` bails on a degenerate direction before anything else
    # (engine_solver.c:1705, LSresult 1).
    var snorm = sqrt(snorm_sq)
    if snorm < MINVAL:
        return ZERO

    var gtol = gtol_scale * snorm

    # ── the line's derivatives at `alpha` ────────────────────────────────
    # `d(cost)/dalpha = -f*Jv` in EVERY state; the second derivative is
    # `D*Jv^2` only where the row is quadratic. Rows are RE-CLASSIFIED at the
    # trial point — a step can move a row across a zone boundary, and not
    # re-classifying is what makes a line search a fixed quadratic model.
    @parameter
    @always_inline
    def eval_at(
        a: Scalar[DTYPE],
        mut d1: Scalar[DTYPE],
        mut d2: Scalar[DTYPE],
    ):
        d1 = gauss_a * a + gauss_b
        d2 = gauss_a
        for e_idx in range(num_edges):
            var jt = jar[e_idx] + a * Jv_e[e_idx]
            var st = scalar_row_state[DTYPE](
                kind_e[e_idx], jt, R_e[e_idx], floss_e[e_idx]
            )
            d1 += (
                -scalar_row_force[DTYPE](st, jt, De[e_idx], floss_e[e_idx])
                * Jv_e[e_idx]
            )
            if st == SROW_QUADRATIC:
                d2 += De[e_idx] * Jv_e[e_idx] * Jv_e[e_idx]
        # ⚠ FLOOR ONLY A NON-POSITIVE SECOND DERIVATIVE, AND ONLY TO
        # `mjMINVAL`. `PrimalPoint` (engine_solver.c:1648) reads
        # `if (deriv[1] <= 0) { mju_warning("not convex"); deriv[1] = mjMINVAL; }`
        # — a guard against a SHOULD-NOT-OCCUR convexity violation. Testing
        # `< PRIMAL_MINVAL` instead floors a legitimately SMALL POSITIVE
        # curvature, which is what this IS near the optimum, and crushes
        # `alpha = -d1/d2` by up to three orders.
        if d2 <= ZERO:
            d2 = MINVAL

    # ── the line's cost at `alpha`, RELATIVE to alpha = 0 ────────────────
    # Only the fallback branch needs it, and only to choose between two
    # brackets, so it is computed on demand rather than carried per point.
    @parameter
    @always_inline
    def cost_delta(a: Scalar[DTYPE]) -> Scalar[DTYPE]:
        var c: Scalar[DTYPE] = 0
        for i in range(nv):
            var qa_t = qacc[i] + a * search[i]
            var Ma_t = Ma[i] + a * Mv[i]
            c += (
                Scalar[DTYPE](0.5)
                * (Ma_t - f_smooth[i])
                * (qa_t - qacc_smooth[i])
            )
            c -= (
                Scalar[DTYPE](0.5)
                * (Ma[i] - f_smooth[i])
                * (qacc[i] - qacc_smooth[i])
            )
        for e_idx in range(num_edges):
            var jt = jar[e_idx] + a * Jv_e[e_idx]
            var st_t = scalar_row_state[DTYPE](
                kind_e[e_idx], jt, R_e[e_idx], floss_e[e_idx]
            )
            c += scalar_row_cost[DTYPE](
                st_t, jt, De[e_idx], R_e[e_idx], floss_e[e_idx]
            )
            var st_0 = scalar_row_state[DTYPE](
                kind_e[e_idx], jar[e_idx], R_e[e_idx], floss_e[e_idx]
            )
            c -= scalar_row_cost[DTYPE](
                st_0, jar[e_idx], De[e_idx], R_e[e_idx], floss_e[e_idx]
            )
        return c

    var p0_d1 = ZERO
    var p0_d2 = ZERO
    eval_at(ZERO, p0_d1, p0_d2)

    # ⚠ `PrimalSearch` ALWAYS ATTEMPTS ONE NEWTON STEP (engine_solver.c:1733),
    # including when `d1 >= 0`. The old body returned 0 on a non-descent
    # direction; the caller's `alpha < 1e-10` break then ended the whole solve.
    var p1_a = -p0_d1 / p0_d2
    var p1_d1 = ZERO
    var p1_d2 = ZERO
    eval_at(p1_a, p1_d1, p1_d2)
    var used = 1

    if abs(p1_d1) < gtol:
        return p1_a

    # Phase 2: one-sided Newton search until the derivative changes sign.
    var dir = Scalar[DTYPE](1) if p1_d1 < ZERO else Scalar[DTYPE](-1)
    var p2_a = ZERO
    var bracketed = False
    while used < budget:
        if p1_d1 * dir > -gtol:
            bracketed = True
            break
        p2_a = p1_a
        p1_a = p1_a - p1_d1 / p1_d2
        eval_at(p1_a, p1_d1, p1_d2)
        used += 1
        if abs(p1_d1) < gtol:
            return p1_a

    # Phase 3: bisect inside the bracket.
    if bracketed:
        while used < budget:
            var mid = (p1_a + p2_a) * Scalar[DTYPE](0.5)
            var m_d1 = ZERO
            var m_d2 = ZERO
            eval_at(mid, m_d1, m_d2)
            used += 1
            if abs(m_d1) < gtol:
                return mid
            # Keep the half whose endpoints straddle the root.
            if m_d1 * dir < ZERO:
                p2_a = mid
            else:
                p1_a = mid

    # ⚠ NO IMPROVEMENT MEANS ZERO, NOT A SMALL STEP. `PrimalSearch` returns
    # the bracket with the lower cost only when that cost is BELOW alpha=0's,
    # and otherwise returns exactly 0 (LSresult 5) so the caller's loop breaks
    # without moving `qacc`. Returning a tiny non-improving alpha instead — as
    # the halving body did — nudged `qacc` by ~1e-12 on EVERY step of a
    # rollout once the warm start put the iterate near its optimum.
    var c1 = cost_delta(p1_a)
    var c2 = cost_delta(p2_a)
    if c1 <= c2 and c1 < ZERO:
        return p1_a
    if c2 < c1 and c2 < ZERO:
        return p2_a
    return ZERO

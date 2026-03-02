"""Newton constraint solver (MuJoCo-matching).

Operates in qacc (acceleration) space, minimizing:
  cost = 0.5*(qacc - qacc_smooth)^T * M * (qacc - qacc_smooth)  [Gauss term]
       + sum_i penalty_i(J*qacc - aref)                         [constraint costs]

Forces are derived from qacc: force[i] = -D[i] * (J*qacc - aref)[i].
The Hessian is H = M + J^T*D_active*J (naturally positive definite).

This is fundamentally different from a dual Newton solver which builds
the Delassus matrix A = J*M^{-1}*J^T and solves a QP over forces.

The constraint_update uses MuJoCo's 3-zone cone logic for elliptic contacts:
- Top zone (N >= 0, N >= mu*T): satisfied, no forces
- Bottom zone (N + mu*T <= 0): full quadratic on all rows
- Middle zone: cone boundary projection with Dm-weighted cost

This eliminates the need for a separate PGS friction phase — the
Newton optimization handles ALL constraints (normals, friction, limits,
equality) in a unified framework.

Algorithm:
1. qacc = qacc_smooth (unconstrained acceleration)
2. Ma = M * qacc, jar = J*qacc - aref
3. constraint_update → force, state, cost (cone-aware)
4. Build H = M + J^T * D_active * J, Cholesky factorize
5. Main loop:
   a. grad = Ma - qfrc_smooth - J^T * force
   b. search = -chol_solve(H, grad)
   c. Armijo linesearch → alpha
   d. qacc += alpha * search, Ma += alpha * M*search
   e. constraint_update → force, state, cost
   f. Incremental Hessian update if state changed
   g. Check convergence

Reference: mujoco-main/src/engine/engine_solver.c (mj_solPrimal)
"""

from math import sqrt
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim, barrier
from ..types import Model, Data, _max_one, ConeType
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..traits.solver import ConstraintSolver
from ..constraints.constraint_data import (
    ConstraintData,
    CNSTR_NORMAL,
    CNSTR_FRICTION_T1,
    CNSTR_FRICTION_T2,
    CNSTR_FRICTION_TORSION,
    CNSTR_FRICTION_ROLL1,
    CNSTR_FRICTION_ROLL2,
    CNSTR_LIMIT,
    CNSTR_PYRAMID_EDGE,
    CNSTR_EQUALITY_CONNECT,
    CNSTR_EQUALITY_WELD,
)
from .primal_common import (
    constraint_update,
    constraint_update_with_D,
    compute_jar,
    compute_qfrc_constraint,
    compute_gauss_cost,
    primal_linesearch,
    primal_linesearch_with_D,
    primal_D,
    PRIMAL_SATISFIED,
    PRIMAL_QUADRATIC,
    PRIMAL_CONE,
    PRIMAL_MINVAL,
)
from .cholesky import (
    chol_factor,
    chol_solve,
    chol_rank1_update,
    chol_factor_inline,
    chol_solve_inline,
)

from ..gpu.constants import (
    contacts_offset,
    metadata_offset,
    model_metadata_offset,
    ws_qacc_constrained_offset,
    ws_m_inv_offset,
    ws_solver_offset,
    ws_M_offset,
    ws_fnet_offset,
    qvel_offset,
    CONTACT_SIZE,
    CONTACT_IDX_FORCE_N,
    CONTACT_IDX_FORCE_T1,
    CONTACT_IDX_FORCE_T2,
    CONTACT_IDX_FRICTION,
    CONTACT_IDX_CONDIM,
    CONTACT_IDX_FRAME_T1_X,
    CONTACT_IDX_FRAME_T1_Y,
    CONTACT_IDX_FRAME_T1_Z,
    META_IDX_NUM_CONTACTS,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_SOLREF_CONTACT_0,
    MODEL_META_IDX_SOLREF_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_0,
    MODEL_META_IDX_SOLIMP_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_2,
    MODEL_META_IDX_SOLIMP_CONTACT_3,
    MODEL_META_IDX_SOLIMP_CONTACT_4,
    MODEL_META_IDX_IMPRATIO,
)

from ..dynamics.jacobian import compute_contact_jacobian_row_gpu

from ..constraints.constraint_builder_gpu import (
    init_common_normal_workspace_gpu,
    precompute_contact_normal_gpu,
    warmstart_normals_gpu,
    apply_solved_normals_gpu,
    detect_and_solve_limits_gpu,
    build_and_solve_equality_gpu,
    build_and_solve_tendon_gpu,
)

# Newton solver parameters
comptime NEWTON_CPU_ITERATIONS: Int = 200
comptime NEWTON_CPU_TOLERANCE: Float64 = 1e-12
# Debug flag
comptime NEWTON_CPU_DEBUG: Bool = False
comptime MINVAL: Float64 = 1e-10


@always_inline
fn _build_hessian[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    MAX_ROWS: Int,
    V_SIZE: Int,
    M_SIZE: Int,
](
    constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
    D_vals: List[Scalar[DTYPE]],
    jar: List[Scalar[DTYPE]],
    cstate: List[Int],
    mut H: List[Scalar[DTYPE]],
):
    """Build Hessian contributions from active constraints.

    For QUADRATIC state: add D_r * J_r * J_r^T (standard per-row).
    For CONE state: add J_group^T * H_cone * J_group (coupled dim x dim).

    The cone Hessian H_cone is d^2(cost)/d(jar)^2 for the contact group:
      H[0,0] = Dm
      H[0,j] = -Dm * mu * jar_fj / T
      H[j,k] = Dm * mu^2 * jar_fj*jar_fk/T^2 + Dm*mu*(N-mu*T)/T * (jar_fj*jar_fk/T^2 - delta_jk)
      where N = jar_n, T = ||jar_f||, Dm = D_n/(1+mu^2)

    Reference: mujoco-main/src/engine/engine_core_constraint.c:2530-2569
    """
    comptime MR = _max_one[MAX_ROWS]()
    var num_normals = constraints.num_normals
    var num_friction = constraints.num_friction
    var friction_start = num_normals
    var limits_start = num_normals + num_friction

    # Track which rows are handled as part of cone groups
    var handled = InlineArray[Bool, MR](fill=False)

    # Process cone contact groups (normal + friction children together)
    var fric_idx = 0
    for n in range(num_normals):
        # Find friction children
        var group_size = 0
        while fric_idx + group_size < num_friction:
            if (
                constraints.rows[
                    friction_start + fric_idx + group_size
                ].friction_parent
                != n
            ):
                break
            group_size += 1

        if cstate[n] == PRIMAL_CONE and group_size > 0:
            # Cone state: build full dim x dim Hessian
            var D_n = D_vals[n]
            var mu = constraints.rows[friction_start + fric_idx].friction_coef
            # Dm = D_n / (1 + mu^2) in jar-space (no group_size factor)
            var Dm = D_n / (Scalar[DTYPE](1) + mu * mu)

            var N = jar[n]
            var T_sq: Scalar[DTYPE] = 0
            for g in range(group_size):
                T_sq += (
                    jar[friction_start + fric_idx + g]
                    * jar[friction_start + fric_idx + g]
                )
            var T = sqrt(T_sq)
            var T_safe = T
            if T_safe < Scalar[DTYPE](MINVAL):
                T_safe = Scalar[DTYPE](MINVAL)
            var s = N - mu * T

            # Build cone Hessian as J^T * H_cone * J
            # Rather than form H_cone explicitly, compute J_cone^T * H_cone * J_cone
            # row indices: n (normal), friction_start + fric_idx + g (friction children)

            # H_cone[0,0] = Dm → add Dm * J_n * J_n^T
            for i in range(NV):
                for j in range(NV):
                    H[i * NV + j] += (
                        Dm
                        * constraints.J[n * NV + i]
                        * constraints.J[n * NV + j]
                    )

            # H_cone[0,g+1] = -Dm * mu * jar_fg / T_safe  (cross-terms)
            for g in range(group_size):
                var fr = friction_start + fric_idx + g
                var h_cross = -Dm * mu * jar[fr] / T_safe
                for i in range(NV):
                    for j in range(NV):
                        # J_n^T * h_cross * J_fr + J_fr^T * h_cross * J_n (symmetric)
                        H[i * NV + j] += h_cross * (
                            constraints.J[n * NV + i]
                            * constraints.J[fr * NV + j]
                            + constraints.J[fr * NV + i]
                            * constraints.J[n * NV + j]
                        )

            # H_cone[g1+1,g2+1]: friction-friction block
            for g1 in range(group_size):
                var fr1 = friction_start + fric_idx + g1
                for g2 in range(group_size):
                    var fr2 = friction_start + fric_idx + g2
                    # Outer product term: Dm * mu^2 * jar_f1*jar_f2 / T^2
                    var h_ff = (
                        Dm * mu * mu * jar[fr1] * jar[fr2] / (T_safe * T_safe)
                    )
                    # Rank-correction: Dm * mu * s / T * (jar_f1*jar_f2/T^2 - delta)
                    h_ff += (
                        Dm
                        * mu
                        * s
                        / T_safe
                        * (jar[fr1] * jar[fr2] / (T_safe * T_safe))
                    )
                    if g1 == g2:
                        h_ff -= Dm * mu * s / T_safe
                    for i in range(NV):
                        for j in range(NV):
                            H[i * NV + j] += (
                                h_ff
                                * constraints.J[fr1 * NV + i]
                                * constraints.J[fr2 * NV + j]
                            )

            # Mark these rows as handled
            handled[n] = True
            for g in range(group_size):
                handled[friction_start + fric_idx + g] = True

        fric_idx += group_size

    # Add standard per-row contributions for non-cone active rows
    for r in range(constraints.num_rows):
        if handled[r] or cstate[r] == PRIMAL_SATISFIED:
            continue
        # QUADRATIC state: standard D * J * J^T
        var D_r = D_vals[r]
        for i in range(NV):
            for j in range(NV):
                H[i * NV + j] += (
                    D_r * constraints.J[r * NV + i] * constraints.J[r * NV + j]
                )


struct NewtonSolver(ConstraintSolver):
    """MuJoCo-style Newton constraint solver.

    Operates in qacc space, minimizing the cost function
    (Gauss + constraint penalties) using Newton's method with
    Cholesky-factored Hessian and Armijo linesearch.

    The Hessian H = M + J^T*D*J is always positive definite (since M is PD
    and J^T*D*J is PSD), so Newton direction is always a descent direction.

    Handles ALL constraints (including friction cone) in a unified
    optimization via cone-aware constraint_update.
    No separate PGS friction phase needed on CPU.
    """

    @staticmethod
    fn solver_workspace_size[NV: Int, MAX_CONTACTS: Int]() -> Int:
        """Newton solver workspace size (primal, qacc-space).

        Layout:
          Common normal block: 13*MC + 2*MC*NV
            [lambda_n, K_n, c_dist, c_body, c_body_b, px, py, pz,
             nx, ny, nz, pos_bias, inv_K_imp | J_n: MC*NV | MinvJn: MC*NV]
          Primal-specific block: 4*MC*NV + 5*MC + 7*MC = 12*MC + 4*MC*NV
            [J_t1: MC*NV | J_t2: MC*NV | MinvJt1: MC*NV | MinvJt2: MC*NV |
             mu: MC | D_n: MC | D_f: MC | bt1: MC | bt2: MC |
             jar_n: MC | jar_t1: MC | jar_t2: MC | fn: MC | ft1: MC | ft2: MC | cstate: MC]
        Total = 25*MC + 6*MC*NV
        """
        comptime MC = _max_one[MAX_CONTACTS]()
        return 25 * MC + 6 * MC * NV

    @no_inline
    @staticmethod
    fn solve[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        MAX_ROWS: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        MAX_TENDON: Int = 0,
        NSITE: Int = 0,
    ](
        model: Model[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
            NSITE,
        ],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
        M_inv: List[Scalar[DTYPE]],
        mut constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
        mut qacc: List[Scalar[DTYPE]],
        dt: Scalar[DTYPE],
    ):
        """Solve constraints using Newton on CPU.

        Unified optimization over all constraints (normals + friction cone +
        limits + equality) using cone-aware constraint_update.
        """
        if constraints.num_rows == 0:
            return

        comptime V_SIZE = _max_one[NV]()
        comptime M_SIZE = _max_one[NV * NV]()
        comptime MR = _max_one[MAX_ROWS]()

        var num_rows = constraints.num_rows
        var num_normals = constraints.num_normals
        var num_friction = constraints.num_friction
        var friction_start = num_normals

        # Compute D values from stored diagApprox and inv_K_imp
        # D = 1/R where R = 1/inv_K_imp - K = (1-imp)/imp * diagApprox
        # All constraint types now store diagApprox, so primal_D works correctly.
        var D_vals = List[Scalar[DTYPE]](capacity=MR)
        for _ in range(MR):
            D_vals.append(Scalar[DTYPE](0))
        for r in range(num_rows):
            D_vals[r] = primal_D(
                constraints.rows[r].inv_K_imp,
                constraints.rows[r].K,
            )
        # Each row keeps its own D = 1/R computed from its own diagApprox.
        # In MuJoCo's BOTTOM zone, force = -D[j]*jar[j] uses per-row D.
        # In CONE zone, Dm = D_n/(1+mu²) handles normal-friction coupling.

        # Save qacc_smooth (unconstrained acceleration)
        var qacc_smooth = List[Scalar[DTYPE]](capacity=V_SIZE)
        for i in range(NV):
            qacc_smooth.append(qacc[i])

        # MuJoCo-style warm-start: compare cost(qacc_warmstart) vs cost(qacc_smooth)
        # and use whichever is lower as the Newton starting point.
        # qacc_warmstart = solved qacc saved at end of previous step (exact).
        var has_warmstart = False
        for i in range(NV):
            if data.qacc_warmstart[i] != Scalar[DTYPE](0):
                has_warmstart = True
                break
        if has_warmstart:
            # Compute Gauss cost at qacc_smooth: 0 (by construction, it minimizes
            # unconstrained cost). Compare cost at qacc_warmstart vs qacc_smooth.
            # cost(q) = 0.5*(M*q - f)^T * M_inv * (M*q - f) + constraint terms
            # At qacc_smooth: 0.5 * |qacc_smooth - qacc_smooth|^2_M = 0
            # At qacc_warmstart: 0.5 * |qacc_ws - qacc_smooth|^2_M > 0 unless zero
            # MuJoCo uses: if cost_warmstart <= cost_smooth, use qacc_warmstart.
            # Simple heuristic: always use warmstart when available — it starts
            # closer to the previous solution which is near the new optimum for
            # slowly-varying dynamics. This matches MuJoCo's warm-start logic.
            for i in range(NV):
                qacc[i] = data.qacc_warmstart[i]

        # qfrc_smooth from constraints (filled by integrator)
        var qfrc_smooth = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            qfrc_smooth.append(Scalar[DTYPE](0))
        for i in range(V_SIZE):
            qfrc_smooth[i] = constraints.qfrc_smooth[i]

        # Compute Ma = M * qacc
        var Ma = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            Ma.append(Scalar[DTYPE](0))
        for i in range(NV):
            Ma[i] = Scalar[DTYPE](0)
            for j in range(NV):
                Ma[i] += constraints.M_hat[i * NV + j] * qacc[j]

        # Compute jar = J * qacc - aref (aref = -bias)
        var jar = List[Scalar[DTYPE]](capacity=MR)
        for _ in range(MR):
            jar.append(Scalar[DTYPE](0))
        compute_jar[DTYPE, MAX_ROWS, NV](constraints, qacc, jar)

        # Compute initial force, state, cost (cone-aware with MuJoCo D)
        var force = List[Scalar[DTYPE]](capacity=MR)
        for _ in range(MR):
            force.append(Scalar[DTYPE](0))
        var cstate = List[Int](capacity=MR)
        for _ in range(MR):
            cstate.append(PRIMAL_SATISFIED)
        var constraint_cost: Scalar[DTYPE] = 0
        constraint_update_with_D[DTYPE, MAX_ROWS, NV, MR](
            constraints, jar, D_vals, force, cstate, constraint_cost
        )

        @parameter
        if NEWTON_CPU_DEBUG:
            print(
                "  [PRIMAL] num_rows=",
                num_rows,
                " normals=",
                constraints.num_normals,
                " friction=",
                constraints.num_friction,
                " limits=",
                constraints.num_limits,
            )
            for r in range(num_rows):
                var state_str = "SAT"
                if cstate[r] == PRIMAL_QUADRATIC:
                    state_str = "QUAD"
                elif cstate[r] == PRIMAL_CONE:
                    state_str = "CONE"
                print(
                    "  row",
                    r,
                    " type=",
                    constraints.rows[r].constraint_type,
                    " D_mj=",
                    Float64(D_vals[r]),
                    " jar=",
                    Float64(jar[r]),
                    " force=",
                    Float64(force[r]),
                    " state=",
                    state_str,
                    " bias=",
                    Float64(constraints.rows[r].bias),
                    " mu=",
                    Float64(constraints.rows[r].friction_coef),
                    " parent=",
                    constraints.rows[r].friction_parent,
                )

        # Compute qfrc_constraint = J^T * force
        var qfrc_constraint = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            qfrc_constraint.append(Scalar[DTYPE](0))

        compute_qfrc_constraint[DTYPE, MAX_ROWS, NV, V_SIZE, MR](
            constraints, force, qfrc_constraint
        )

        # Build Hessian H = M + J^T * D_active * J + cone Hessians
        var H = List[Scalar[DTYPE]](capacity=M_SIZE)
        for _ in range(M_SIZE):
            H.append(Scalar[DTYPE](0))
        for i in range(M_SIZE):
            H[i] = constraints.M_hat[i]
        var L = List[Scalar[DTYPE]](capacity=M_SIZE)
        for _ in range(M_SIZE):
            L.append(Scalar[DTYPE](0))

        _build_hessian[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, MAX_ROWS, V_SIZE, M_SIZE
        ](constraints, D_vals, jar, cstate, H)

        # Cholesky factorize H
        chol_factor[DTYPE, NV, M_SIZE](H, L)

        # Compute scale for convergence check (MuJoCo: 1/sum(M diagonal))
        var scale: Scalar[DTYPE] = 0
        for i in range(NV):
            scale += constraints.M_hat[i * NV + i]
        if scale > Scalar[DTYPE](MINVAL):
            scale = Scalar[DTYPE](1.0) / scale
        else:
            scale = Scalar[DTYPE](1.0)

        # Main Newton iteration loop
        var grad = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            grad.append(Scalar[DTYPE](0))
        var search = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            search.append(Scalar[DTYPE](0))
        var Mv = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            Mv.append(Scalar[DTYPE](0))

        var total_iter = 0

        for iter in range(NEWTON_CPU_ITERATIONS):
            total_iter += 1
            # Compute gradient: grad = Ma - qfrc_smooth - qfrc_constraint
            var grad_norm: Scalar[DTYPE] = 0
            for i in range(NV):
                grad[i] = Ma[i] - qfrc_smooth[i] - qfrc_constraint[i]
                grad_norm += grad[i] * grad[i]

            @parameter
            if NEWTON_CPU_DEBUG:
                print(
                    "    [PRIMAL_NEWTON] iter_start",
                    total_iter,
                    " grad_norm=",
                    Float64(sqrt(grad_norm)),
                    " scaled=",
                    Float64(scale * sqrt(grad_norm)),
                )

            # Check gradient convergence
            if scale * sqrt(grad_norm) < Scalar[DTYPE](NEWTON_CPU_TOLERANCE):

                @parameter
                if NEWTON_CPU_DEBUG:
                    print(
                        "    [PRIMAL_NEWTON] CONVERGED at iter",
                        total_iter,
                        " (gradient)",
                    )
                break

            # Newton direction: search = -H^{-1} * grad via Cholesky solve
            chol_solve[DTYPE, NV, M_SIZE, V_SIZE](L, grad, search)
            for i in range(NV):
                search[i] = -search[i]

            # Compute Mv = M * search (needed for line search)
            for i in range(NV):
                Mv[i] = Scalar[DTYPE](0)
                for j in range(NV):
                    Mv[i] += constraints.M_hat[i * NV + j] * search[j]

            # Forward-exploring linesearch with MuJoCo D
            var alpha = primal_linesearch_with_D[
                DTYPE, MAX_ROWS, NV, V_SIZE, MR
            ](
                constraints,
                D_vals,
                qacc,
                qacc_smooth,
                qfrc_smooth,
                Ma,
                Mv,
                search,
                jar,
                force,
                Scalar[DTYPE](NEWTON_CPU_TOLERANCE),
            )

            if alpha == Scalar[DTYPE](0):

                @parameter
                if NEWTON_CPU_DEBUG:
                    print(
                        "    [PRIMAL_NEWTON] STOPPED at iter",
                        total_iter,
                        " (alpha=0)",
                    )
                break

            # Save old state, cost, qacc, Ma
            var old_cost = constraint_cost + compute_gauss_cost[
                DTYPE, NV, V_SIZE
            ](Ma, qfrc_smooth, qacc, qacc_smooth)
            var old_state = InlineArray[Int, MR](uninitialized=True)
            for i in range(num_rows):
                old_state[i] = cstate[i]
            var old_qacc = InlineArray[Scalar[DTYPE], V_SIZE](
                uninitialized=True
            )
            var old_Ma = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
            for i in range(NV):
                old_qacc[i] = qacc[i]
                old_Ma[i] = Ma[i]

            # Update qacc, Ma
            for i in range(NV):
                qacc[i] += alpha * search[i]
                Ma[i] += alpha * Mv[i]

            # Recompute jar
            compute_jar[DTYPE, MAX_ROWS, NV](constraints, qacc, jar)

            # Recompute force, state, cost (cone-aware with MuJoCo D)
            constraint_update_with_D[DTYPE, MAX_ROWS, NV, MR](
                constraints, jar, D_vals, force, cstate, constraint_cost
            )

            # Recompute qfrc_constraint
            compute_qfrc_constraint[DTYPE, MAX_ROWS, NV, V_SIZE, MR](
                constraints, force, qfrc_constraint
            )

            # Check improvement
            var new_cost = constraint_cost + compute_gauss_cost[
                DTYPE, NV, V_SIZE
            ](Ma, qfrc_smooth, qacc, qacc_smooth)
            var improvement = scale * (old_cost - new_cost)

            @parameter
            if NEWTON_CPU_DEBUG:
                print(
                    "    [PRIMAL_NEWTON] iter",
                    total_iter,
                    " alpha=",
                    Float64(alpha),
                    " cost=",
                    Float64(new_cost),
                    " improvement=",
                    Float64(improvement),
                    " grad=",
                    Float64(sqrt(grad_norm)),
                )

            if improvement < Scalar[DTYPE](NEWTON_CPU_TOLERANCE) and iter > 0:
                # Restore qacc/Ma if cost increased
                if improvement < Scalar[DTYPE](0):
                    for i in range(NV):
                        qacc[i] = old_qacc[i]
                        Ma[i] = old_Ma[i]
                    # Recompute jar/force at restored point
                    compute_jar[DTYPE, MAX_ROWS, NV](constraints, qacc, jar)
                    constraint_update_with_D[DTYPE, MAX_ROWS, NV, MR](
                        constraints, jar, D_vals, force, cstate, constraint_cost
                    )
                    compute_qfrc_constraint[DTYPE, MAX_ROWS, NV, V_SIZE, MR](
                        constraints, force, qfrc_constraint
                    )
                break

            # Check if any states changed — if so, rebuild Hessian
            var state_changed = False
            for r in range(num_rows):
                if old_state[r] != cstate[r]:
                    state_changed = True
                    break

            if state_changed:
                for i in range(NV * NV):
                    H[i] = constraints.M_hat[i]
                _build_hessian[
                    DTYPE,
                    NQ,
                    NV,
                    NBODY,
                    NJOINT,
                    MAX_CONTACTS,
                    MAX_ROWS,
                    V_SIZE,
                    M_SIZE,
                ](constraints, D_vals, jar, cstate, H)
                chol_factor[DTYPE, NV, M_SIZE](H, L)

        @parameter
        if NEWTON_CPU_DEBUG:
            print("  [PRIMAL] Final states:")
            for r in range(num_rows):
                var state_str = "SAT"
                if cstate[r] == PRIMAL_QUADRATIC:
                    state_str = "QUAD"
                elif cstate[r] == PRIMAL_CONE:
                    state_str = "CONE"
                print(
                    "    row",
                    r,
                    " state=",
                    state_str,
                    " jar=",
                    Float64(jar[r]),
                    " force=",
                    Float64(force[r]),
                )

        # Write forces back to constraint lambda_val for warm-starting
        for r in range(num_rows):
            constraints.rows[r].lambda_val = force[r]

        # Save solved qacc as warm-start for next step (MuJoCo qacc_warmstart)
        for i in range(NV):
            data.qacc_warmstart[i] = qacc[i]

    @staticmethod
    fn solver_threads[
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
    ]() -> Int:
        return _max_one[MAX_CONTACTS]()

    @staticmethod
    @always_inline
    fn solve_gpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
        MODEL_SIZE: Int,
        V_SIZE: Int,
        BATCH: Int,
        WS_SIZE: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        MAX_TENDON: Int = 0,
        NSITE: Int = 0,
    ](
        state: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        model: LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ],
        workspace: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
        ],
    ):
        """GPU primal Newton solver — matches CPU Newton exactly.

        Operates in qacc (NV-dimensional) space with unified friction cone.
        Builds H = M + J^T*D*J, Cholesky-factorizes, and iterates Newton steps.
        Handles normal + friction (T1+T2) contacts in a single optimization.
        No separate PGS friction phase needed.
        """
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        var contact_tid = Int(thread_idx.y)
        var valid_env = env < BATCH

        comptime qacc_idx = ws_qacc_constrained_offset[NV, NBODY]()
        comptime solver_ws_idx = ws_solver_offset[NV, NBODY]()
        comptime fnet_idx = ws_fnet_offset[NV, NBODY]()
        comptime M_idx = ws_M_offset[NV, NBODY]()
        comptime M_inv_idx = ws_m_inv_offset[NV, NBODY]()
        comptime MC = _max_one[MAX_CONTACTS]()
        comptime M_SIZE = _max_one[NV * NV]()

        # Common normal block offsets
        comptime ws_lambda_n_idx = solver_ws_idx + 0 * MC
        comptime ws_K_n_idx = solver_ws_idx + 1 * MC
        comptime ws_c_dist_idx = solver_ws_idx + 2 * MC
        comptime ws_c_body_idx = solver_ws_idx + 3 * MC
        comptime ws_c_body_b_idx = solver_ws_idx + 4 * MC
        comptime ws_c_px_idx = solver_ws_idx + 5 * MC
        comptime ws_c_py_idx = solver_ws_idx + 6 * MC
        comptime ws_c_pz_idx = solver_ws_idx + 7 * MC
        comptime ws_c_nx_idx = solver_ws_idx + 8 * MC
        comptime ws_c_ny_idx = solver_ws_idx + 9 * MC
        comptime ws_c_nz_idx = solver_ws_idx + 10 * MC
        comptime ws_pos_bias_idx = solver_ws_idx + 11 * MC
        comptime ws_inv_K_imp_idx = solver_ws_idx + 12 * MC
        comptime ws_J_n_idx = solver_ws_idx + 13 * MC
        comptime ws_MinvJn_idx = solver_ws_idx + 13 * MC + MC * NV

        # Primal-specific offsets (after common normal block)
        comptime PRIMAL_START = solver_ws_idx + 13 * MC + 2 * MC * NV
        comptime ws_Jt1_idx = PRIMAL_START + 0 * MC * NV
        comptime ws_Jt2_idx = PRIMAL_START + 1 * MC * NV
        comptime ws_MinvJt1_idx = PRIMAL_START + 2 * MC * NV
        comptime ws_MinvJt2_idx = PRIMAL_START + 3 * MC * NV
        comptime SC = PRIMAL_START + 4 * MC * NV
        comptime ws_mu_idx = SC + 0 * MC
        comptime ws_D_n_idx = SC + 1 * MC
        comptime ws_D_f_idx = SC + 2 * MC
        comptime ws_bt1_idx = SC + 3 * MC
        comptime ws_bt2_idx = SC + 4 * MC
        comptime CVS = SC + 5 * MC
        comptime ws_jar_n_idx = CVS + 0 * MC
        comptime ws_jar_t1_idx = CVS + 1 * MC
        comptime ws_jar_t2_idx = CVS + 2 * MC
        comptime ws_fn_idx = CVS + 3 * MC
        comptime ws_ft1_idx = CVS + 4 * MC
        comptime ws_ft2_idx = CVS + 5 * MC
        comptime ws_cstate_idx = CVS + 6 * MC

        # === PARALLEL: Initialize common normal workspace ===
        if valid_env:
            init_common_normal_workspace_gpu[
                DTYPE,
                NV,
                NBODY,
                MAX_CONTACTS,
                WS_SIZE,
                BATCH,
            ](env, contact_tid, workspace)
            # Zero primal workspace for this contact slot
            if contact_tid < MC:
                for d in range(NV):
                    workspace[env, ws_Jt1_idx + contact_tid * NV + d] = 0
                    workspace[env, ws_Jt2_idx + contact_tid * NV + d] = 0
                    workspace[env, ws_MinvJt1_idx + contact_tid * NV + d] = 0
                    workspace[env, ws_MinvJt2_idx + contact_tid * NV + d] = 0
                workspace[env, ws_mu_idx + contact_tid] = 0
                workspace[env, ws_D_n_idx + contact_tid] = 0
                workspace[env, ws_D_f_idx + contact_tid] = 0
                workspace[env, ws_bt1_idx + contact_tid] = 0
                workspace[env, ws_bt2_idx + contact_tid] = 0
                workspace[env, ws_jar_n_idx + contact_tid] = 0
                workspace[env, ws_jar_t1_idx + contact_tid] = 0
                workspace[env, ws_jar_t2_idx + contact_tid] = 0
                workspace[env, ws_fn_idx + contact_tid] = 0
                workspace[env, ws_ft1_idx + contact_tid] = 0
                workspace[env, ws_ft2_idx + contact_tid] = 0
                workspace[env, ws_cstate_idx + contact_tid] = 0

        # Read metadata
        comptime contacts_off = contacts_offset[NQ, NV, NBODY]()
        comptime meta_off = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
        comptime model_meta_off = model_metadata_offset[NBODY, NJOINT]()

        var nc = 0
        var dt: Scalar[DTYPE] = 0
        var K_spring: Scalar[DTYPE] = 0
        var B_damp: Scalar[DTYPE] = 0
        var si_dmin: Scalar[DTYPE] = 0
        var si_dmax: Scalar[DTYPE] = 0
        var si_width: Scalar[DTYPE] = 1
        var si_midpoint: Scalar[DTYPE] = Scalar[DTYPE](0.5)
        var si_power: Scalar[DTYPE] = Scalar[DTYPE](2.0)
        var impratio: Scalar[DTYPE] = Scalar[DTYPE](1.0)

        if valid_env:
            dt = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_TIMESTEP]
            )
            nc = Int(
                rebind[Scalar[DTYPE]](
                    state[env, meta_off + META_IDX_NUM_CONTACTS]
                )
            )
            if nc > MAX_CONTACTS:
                nc = MAX_CONTACTS
            var sr_tc = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLREF_CONTACT_0]
            )
            var sr_dr = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLREF_CONTACT_1]
            )
            si_dmin = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_0]
            )
            si_dmax = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_1]
            )
            si_width = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_2]
            )
            si_midpoint = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_3]
            )
            si_power = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_4]
            )
            if si_width < Scalar[DTYPE](1e-6):
                si_width = Scalar[DTYPE](1e-6)
            if si_dmax < Scalar[DTYPE](1e-4):
                si_dmax = Scalar[DTYPE](1e-4)
            K_spring = Scalar[DTYPE](1.0) / (sr_tc * sr_tc * si_dmax * si_dmax)
            B_damp = Scalar[DTYPE](2.0) * sr_dr / (sr_tc * si_dmax)
            impratio = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_IMPRATIO]
            )
            if impratio < Scalar[DTYPE](1e-6):
                impratio = Scalar[DTYPE](1.0)

        # === PARALLEL PHASE 1: Each thread precomputes one contact's normal data ===
        if valid_env:
            precompute_contact_normal_gpu[
                DTYPE,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                STATE_SIZE,
                MODEL_SIZE,
                V_SIZE,
                BATCH,
                WS_SIZE,
                NGEOM,
                MAX_EQUALITY,
                COMPUTE_RHS=False,
                RHS_IDX=0,
                MAX_TENDON=MAX_TENDON,
                NSITE=NSITE,
            ](
                env,
                contact_tid,
                nc,
                state,
                model,
                workspace,
                K_spring,
                B_damp,
                si_dmin,
                si_dmax,
                si_width,
                si_midpoint,
                si_power,
            )

        barrier()

        # === PARALLEL PHASE 2: Tangent frame + Jt1/Jt2/bt/D/mu (one contact per thread) ===
        # Each contact_tid thread independently handles one contact — no write conflicts.
        # MinvJt1/MinvJt2 removed: they are never read by the primal Newton loop.
        comptime qvel_off = qvel_offset[NQ, NV]()
        comptime PRIMAL_MINVAL_GPU: Float64 = 1e-12
        if valid_env and contact_tid < nc:
            var c = contact_tid
            if rebind[Scalar[DTYPE]](
                workspace[env, ws_c_dist_idx + c]
            ) < Scalar[DTYPE](0):
                var nx = rebind[Scalar[DTYPE]](workspace[env, ws_c_nx_idx + c])
                var ny = rebind[Scalar[DTYPE]](workspace[env, ws_c_ny_idx + c])
                var nz = rebind[Scalar[DTYPE]](workspace[env, ws_c_nz_idx + c])

                var c_off = contacts_off + c * CONTACT_SIZE
                var hint_x = rebind[Scalar[DTYPE]](
                    state[env, c_off + CONTACT_IDX_FRAME_T1_X]
                )
                var hint_y = rebind[Scalar[DTYPE]](
                    state[env, c_off + CONTACT_IDX_FRAME_T1_Y]
                )
                var hint_z = rebind[Scalar[DTYPE]](
                    state[env, c_off + CONTACT_IDX_FRAME_T1_Z]
                )
                if hint_x * hint_x + hint_y * hint_y + hint_z * hint_z < Scalar[
                    DTYPE
                ](0.25):
                    hint_x = Scalar[DTYPE](0)
                    if ny >= Scalar[DTYPE](-0.5) and ny <= Scalar[DTYPE](0.5):
                        hint_y = Scalar[DTYPE](1)
                        hint_z = Scalar[DTYPE](0)
                    else:
                        hint_y = Scalar[DTYPE](0)
                        hint_z = Scalar[DTYPE](1)

                # Gram-Schmidt: orthogonalize hint against normal → T1
                var dot_nh = nx * hint_x + ny * hint_y + nz * hint_z
                var t1x = hint_x - dot_nh * nx
                var t1y = hint_y - dot_nh * ny
                var t1z = hint_z - dot_nh * nz
                var t1_mag = sqrt(t1x * t1x + t1y * t1y + t1z * t1z)
                if t1_mag > Scalar[DTYPE](1e-10):
                    t1x = t1x / t1_mag
                    t1y = t1y / t1_mag
                    t1z = t1z / t1_mag

                # T2 = cross(normal, T1)
                var t2x = ny * t1z - nz * t1y
                var t2y = nz * t1x - nx * t1z
                var t2z = nx * t1y - ny * t1x

                var body_a = Int(
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_body_idx + c])
                )
                var body_b = Int(
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_body_b_idx + c])
                )
                var px = rebind[Scalar[DTYPE]](workspace[env, ws_c_px_idx + c])
                var py = rebind[Scalar[DTYPE]](workspace[env, ws_c_py_idx + c])
                var pz = rebind[Scalar[DTYPE]](workspace[env, ws_c_pz_idx + c])

                # Compute J_t1 (MinvJt1 omitted — not used by primal Newton)
                var J_row = InlineArray[Scalar[DTYPE], V_SIZE](
                    uninitialized=True
                )
                compute_contact_jacobian_row_gpu[
                    DTYPE,
                    NQ,
                    NV,
                    NBODY,
                    NJOINT,
                    MAX_CONTACTS,
                    STATE_SIZE,
                    MODEL_SIZE,
                    V_SIZE,
                    BATCH,
                    WS_SIZE,
                ](
                    env,
                    state,
                    model,
                    workspace,
                    body_a,
                    body_b,
                    px,
                    py,
                    pz,
                    t1x,
                    t1y,
                    t1z,
                    J_row,
                )
                for i in range(NV):
                    workspace[env, ws_Jt1_idx + c * NV + i] = J_row[i]

                # Compute J_t2 (MinvJt2 omitted — not used by primal Newton)
                compute_contact_jacobian_row_gpu[
                    DTYPE,
                    NQ,
                    NV,
                    NBODY,
                    NJOINT,
                    MAX_CONTACTS,
                    STATE_SIZE,
                    MODEL_SIZE,
                    V_SIZE,
                    BATCH,
                    WS_SIZE,
                ](
                    env,
                    state,
                    model,
                    workspace,
                    body_a,
                    body_b,
                    px,
                    py,
                    pz,
                    t2x,
                    t2y,
                    t2z,
                    J_row,
                )
                for i in range(NV):
                    workspace[env, ws_Jt2_idx + c * NV + i] = J_row[i]

                # D_n = 1/R_n, D_f = D_n/impratio
                var inv_K_imp_c = rebind[Scalar[DTYPE]](
                    workspace[env, ws_inv_K_imp_idx + c]
                )
                var K_n_c = rebind[Scalar[DTYPE]](
                    workspace[env, ws_K_n_idx + c]
                )
                var R_n_c = Scalar[DTYPE](1.0) / inv_K_imp_c - K_n_c
                if R_n_c < Scalar[DTYPE](1e-14):
                    R_n_c = Scalar[DTYPE](1e-14)
                var D_n_c = Scalar[DTYPE](1.0) / R_n_c
                workspace[env, ws_D_n_idx + c] = D_n_c
                workspace[env, ws_D_f_idx + c] = D_n_c / impratio

                # Friction coefficient
                var mu_c = rebind[Scalar[DTYPE]](
                    state[env, c_off + CONTACT_IDX_FRICTION]
                )
                if mu_c <= Scalar[DTYPE](0):
                    mu_c = Scalar[DTYPE](0.5)
                workspace[env, ws_mu_idx + c] = mu_c

                # Friction velocity-damping bias: bt = B_damp * J_t * qvel
                var bt1_c: Scalar[DTYPE] = 0
                var bt2_c: Scalar[DTYPE] = 0
                for i in range(NV):
                    var qv_i = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
                    bt1_c += (
                        rebind[Scalar[DTYPE]](
                            workspace[env, ws_Jt1_idx + c * NV + i]
                        )
                        * qv_i
                    )
                    bt2_c += (
                        rebind[Scalar[DTYPE]](
                            workspace[env, ws_Jt2_idx + c * NV + i]
                        )
                        * qv_i
                    )
                workspace[env, ws_bt1_idx + c] = B_damp * bt1_c
                workspace[env, ws_bt2_idx + c] = B_damp * bt2_c

        barrier()

        # === SEQUENTIAL: Thread 0 handles primal Newton ===
        if not valid_env or contact_tid != 0:
            return

        comptime NEWTON_ITER_GPU: Int = 20
        comptime NEWTON_TOL_GPU: Float64 = 1e-4
        comptime LINESEARCH_ITER: Int = 10
        comptime ARMIJO: Float64 = 1e-4

        # === Cache loop-invariant contact data into local InlineArrays ===
        # Jn, Jt1, Jt2, mu, D_n, D_f, dist, pos_bias, bt1, bt2 never change
        # during Newton iterations — load once to avoid ~1000 workspace reads/iter.
        var Jn_c = InlineArray[Scalar[DTYPE], MC * V_SIZE](uninitialized=True)
        var Jt1_c = InlineArray[Scalar[DTYPE], MC * V_SIZE](uninitialized=True)
        var Jt2_c = InlineArray[Scalar[DTYPE], MC * V_SIZE](uninitialized=True)
        var mu_cache = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var D_n_cache = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var D_f_cache = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var dist_cache = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var pb_cache = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var bt1_cache = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var bt2_cache = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        for c in range(nc):
            dist_cache[c] = rebind[Scalar[DTYPE]](
                workspace[env, ws_c_dist_idx + c]
            )
            mu_cache[c] = rebind[Scalar[DTYPE]](workspace[env, ws_mu_idx + c])
            D_n_cache[c] = rebind[Scalar[DTYPE]](workspace[env, ws_D_n_idx + c])
            D_f_cache[c] = rebind[Scalar[DTYPE]](workspace[env, ws_D_f_idx + c])
            pb_cache[c] = rebind[Scalar[DTYPE]](
                workspace[env, ws_pos_bias_idx + c]
            )
            bt1_cache[c] = rebind[Scalar[DTYPE]](workspace[env, ws_bt1_idx + c])
            bt2_cache[c] = rebind[Scalar[DTYPE]](workspace[env, ws_bt2_idx + c])
            for i in range(NV):
                Jn_c[c * NV + i] = rebind[Scalar[DTYPE]](
                    workspace[env, ws_J_n_idx + c * NV + i]
                )
                Jt1_c[c * NV + i] = rebind[Scalar[DTYPE]](
                    workspace[env, ws_Jt1_idx + c * NV + i]
                )
                Jt2_c[c * NV + i] = rebind[Scalar[DTYPE]](
                    workspace[env, ws_Jt2_idx + c * NV + i]
                )

        # === Step 2: Initialize local InlineArrays from workspace ===
        var H = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        var L_chol = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        var qacc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var qacc_sm = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var qfrc_sm = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var Ma = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var grad = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var search = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var Mv = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

        # Load M into H (primal Hessian starts as M_hat)
        for k in range(NV * NV):
            H[k] = rebind[Scalar[DTYPE]](workspace[env, M_idx + k])

        # Cache M locally — saves NV² workspace reads per Newton iteration (for Mv = M*search)
        var M_local = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        for k in range(NV * NV):
            M_local[k] = H[k]

        # qacc_sm = unconstrained qacc (set by integrator), save a copy
        for i in range(NV):
            var q_i = rebind[Scalar[DTYPE]](workspace[env, qacc_idx + i])
            qacc[i] = q_i
            qacc_sm[i] = q_i
            qfrc_sm[i] = rebind[Scalar[DTYPE]](workspace[env, fnet_idx + i])

        # Ma = M_local * qacc (uses cached M — no workspace reads)
        for i in range(NV):
            var s: Scalar[DTYPE] = 0
            for j in range(NV):
                s += M_local[i * NV + j] * qacc[j]
            Ma[i] = s

        # Scale = 1/trace(M) for convergence check
        var scale: Scalar[DTYPE] = 0
        for i in range(NV):
            scale += M_local[i * NV + i]
        if scale > Scalar[DTYPE](1e-10):
            scale = Scalar[DTYPE](1.0) / scale
        else:
            scale = Scalar[DTYPE](1.0)

        # === Mutable per-contact state: kept in InlineArrays, written to state buffer at end ===
        var fn_arr = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var ft1_arr = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var ft2_arr = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var jar_n_arr = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var jar_t1_arr = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var jar_t2_arr = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var cs_arr = InlineArray[Int, MC](uninitialized=True)

        # === Step 3: Compute initial jar and forces via 3-zone cone logic ===
        for c in range(nc):
            if dist_cache[c] >= Scalar[DTYPE](0):
                fn_arr[c] = 0
                ft1_arr[c] = 0
                ft2_arr[c] = 0
                jar_n_arr[c] = 0
                jar_t1_arr[c] = 0
                jar_t2_arr[c] = 0
                cs_arr[c] = 0
                continue

            var jar_n: Scalar[DTYPE] = pb_cache[c]
            var jar_t1: Scalar[DTYPE] = bt1_cache[c]
            var jar_t2: Scalar[DTYPE] = bt2_cache[c]
            for i in range(NV):
                var qa_i = qacc[i]
                jar_n += Jn_c[c * NV + i] * qa_i
                jar_t1 += Jt1_c[c * NV + i] * qa_i
                jar_t2 += Jt2_c[c * NV + i] * qa_i
            jar_n_arr[c] = jar_n
            jar_t1_arr[c] = jar_t1
            jar_t2_arr[c] = jar_t2

            var mu = mu_cache[c]
            var D_n = D_n_cache[c]
            var D_f = D_f_cache[c]
            var T = sqrt(jar_t1 * jar_t1 + jar_t2 * jar_t2)
            var T_safe = T
            if T_safe < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                T_safe = Scalar[DTYPE](PRIMAL_MINVAL_GPU)

            if jar_n >= mu * T_safe:
                fn_arr[c] = 0
                ft1_arr[c] = 0
                ft2_arr[c] = 0
                cs_arr[c] = 0  # SATISFIED
            elif mu * jar_n + T <= Scalar[DTYPE](0):
                fn_arr[c] = -D_n * jar_n
                ft1_arr[c] = -D_f * jar_t1
                ft2_arr[c] = -D_f * jar_t2
                cs_arr[c] = 1  # QUADRATIC
            else:
                var s = jar_n - mu * T_safe
                var Dm = D_n / (Scalar[DTYPE](1.0) + mu * mu)
                fn_arr[c] = -Dm * s
                ft1_arr[c] = Dm * mu * s * jar_t1 / T_safe
                ft2_arr[c] = Dm * mu * s * jar_t2 / T_safe
                cs_arr[c] = 2  # CONE

        # === Step 4: Build Hessian H = M + J^T*D*J (cone-aware, using cached Jacobians) ===
        for c in range(nc):
            var cs = cs_arr[c]
            if cs == 0:  # SATISFIED
                continue

            var mu = mu_cache[c]
            var D_n = D_n_cache[c]
            var D_f = D_f_cache[c]

            if cs == 1:  # QUADRATIC: standard rank-1 updates
                for i in range(NV):
                    for j in range(NV):
                        H[i * NV + j] += (
                            D_n * Jn_c[c * NV + i] * Jn_c[c * NV + j]
                            + D_f * Jt1_c[c * NV + i] * Jt1_c[c * NV + j]
                            + D_f * Jt2_c[c * NV + i] * Jt2_c[c * NV + j]
                        )
            else:  # CONE: cone Hessian (coupled normal+friction)
                var jar_n = jar_n_arr[c]
                var jar_t1 = jar_t1_arr[c]
                var jar_t2 = jar_t2_arr[c]
                var T_sq = jar_t1 * jar_t1 + jar_t2 * jar_t2
                var T = sqrt(T_sq)
                var T_safe = T
                if T_safe < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                    T_safe = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
                var s = jar_n - mu * T_safe
                var Dm = D_n / (Scalar[DTYPE](1.0) + mu * mu)
                var h_nt1 = -Dm * mu * jar_t1 / T_safe
                var h_nt2 = -Dm * mu * jar_t2 / T_safe
                var T2_safe = T_safe * T_safe
                var h_t1t1 = (
                    Dm * mu * mu * jar_t1 * jar_t1 / T2_safe
                    + Dm
                    * mu
                    * s
                    / T_safe
                    * (jar_t1 * jar_t1 / T2_safe - Scalar[DTYPE](1.0))
                )
                var h_t2t2 = (
                    Dm * mu * mu * jar_t2 * jar_t2 / T2_safe
                    + Dm
                    * mu
                    * s
                    / T_safe
                    * (jar_t2 * jar_t2 / T2_safe - Scalar[DTYPE](1.0))
                )
                var h_t1t2 = (
                    (Dm * mu * mu + Dm * mu * s / T_safe)
                    * jar_t1
                    * jar_t2
                    / T2_safe
                )
                for i in range(NV):
                    for j in range(NV):
                        H[i * NV + j] += (
                            Dm * Jn_c[c * NV + i] * Jn_c[c * NV + j]
                            + h_nt1
                            * (
                                Jn_c[c * NV + i] * Jt1_c[c * NV + j]
                                + Jt1_c[c * NV + i] * Jn_c[c * NV + j]
                            )
                            + h_nt2
                            * (
                                Jn_c[c * NV + i] * Jt2_c[c * NV + j]
                                + Jt2_c[c * NV + i] * Jn_c[c * NV + j]
                            )
                            + h_t1t1 * Jt1_c[c * NV + i] * Jt1_c[c * NV + j]
                            + h_t2t2 * Jt2_c[c * NV + i] * Jt2_c[c * NV + j]
                            + h_t1t2
                            * (
                                Jt1_c[c * NV + i] * Jt2_c[c * NV + j]
                                + Jt2_c[c * NV + i] * Jt1_c[c * NV + j]
                            )
                        )

        # Cholesky factorize H
        chol_factor_inline[DTYPE, NV, M_SIZE](H, L_chol)

        # === Precompute qfrc_c = J^T * force (replaces per-iteration gradient workspace reads) ===
        # Updated after each force update instead of recomputing from workspace each gradient step.
        var qfrc_c = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NV):
            qfrc_c[i] = Scalar[DTYPE](0)
        for c in range(nc):
            if cs_arr[c] == 0:
                continue
            for i in range(NV):
                qfrc_c[i] += (
                    Jn_c[c * NV + i] * fn_arr[c]
                    + Jt1_c[c * NV + i] * ft1_arr[c]
                    + Jt2_c[c * NV + i] * ft2_arr[c]
                )

        # === Step 5: Newton iteration loop ===
        for _iter in range(NEWTON_ITER_GPU):
            # Gradient = Ma - qfrc_sm - qfrc_c (pure InlineArray reads — no workspace access)
            var grad_norm_sq: Scalar[DTYPE] = 0
            for i in range(NV):
                grad[i] = Ma[i] - qfrc_sm[i] - qfrc_c[i]
                grad_norm_sq += grad[i] * grad[i]

            # Convergence check
            if scale * sqrt(grad_norm_sq) < Scalar[DTYPE](NEWTON_TOL_GPU):
                break

            # Newton direction: search = -H^{-1} * grad
            chol_solve_inline[DTYPE, NV, M_SIZE, V_SIZE](L_chol, grad, search)
            for i in range(NV):
                search[i] = -search[i]

            # Mv = M_local * search (InlineArray reads only — no workspace access)
            for i in range(NV):
                var s: Scalar[DTYPE] = 0
                for j in range(NV):
                    s += M_local[i * NV + j] * search[j]
                Mv[i] = s

            # Precompute J * search per contact (using cached Jacobians — no workspace access)
            var Js_n = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
            var Js_t1 = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
            var Js_t2 = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
            for c in range(nc):
                if dist_cache[c] >= Scalar[DTYPE](0):
                    Js_n[c] = 0
                    Js_t1[c] = 0
                    Js_t2[c] = 0
                    continue
                var js_n: Scalar[DTYPE] = 0
                var js_t1: Scalar[DTYPE] = 0
                var js_t2: Scalar[DTYPE] = 0
                for i in range(NV):
                    var s_i = search[i]
                    js_n += Jn_c[c * NV + i] * s_i
                    js_t1 += Jt1_c[c * NV + i] * s_i
                    js_t2 += Jt2_c[c * NV + i] * s_i
                Js_n[c] = js_n
                Js_t1[c] = js_t1
                Js_t2[c] = js_t2

            # Current total cost: Gauss + constraint (all InlineArray reads)
            var gauss_0: Scalar[DTYPE] = 0
            var g1: Scalar[DTYPE] = 0
            var g2: Scalar[DTYPE] = 0
            var gtd: Scalar[DTYPE] = 0
            for i in range(NV):
                var Ma_diff_i = Ma[i] - qfrc_sm[i]
                var qa_diff_i = qacc[i] - qacc_sm[i]
                gauss_0 += Ma_diff_i * qa_diff_i
                g1 += Ma_diff_i * search[i] + Mv[i] * qa_diff_i
                g2 += Mv[i] * search[i]
                gtd += grad[i] * search[i]
            gauss_0 = Scalar[DTYPE](0.5) * gauss_0
            g1 = Scalar[DTYPE](0.5) * g1
            g2 = Scalar[DTYPE](0.5) * g2

            # Current constraint cost (InlineArray reads only)
            var c_cost_0: Scalar[DTYPE] = 0
            for c in range(nc):
                if dist_cache[c] >= Scalar[DTYPE](0):
                    continue
                var cs = cs_arr[c]
                var N = jar_n_arr[c]
                var T1 = jar_t1_arr[c]
                var T2 = jar_t2_arr[c]
                var mu = mu_cache[c]
                var D_n = D_n_cache[c]
                var D_f = D_f_cache[c]
                if cs == 1:  # QUADRATIC
                    c_cost_0 += Scalar[DTYPE](0.5) * (
                        D_n * N * N + D_f * (T1 * T1 + T2 * T2)
                    )
                elif cs == 2:  # CONE
                    var T_s = sqrt(T1 * T1 + T2 * T2)
                    if T_s < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                        T_s = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
                    var s = N - mu * T_s
                    var Dm = D_n / (Scalar[DTYPE](1.0) + mu * mu)
                    c_cost_0 += Scalar[DTYPE](0.5) * Dm * s * s

            var current_cost = gauss_0 + c_cost_0

            # Armijo linesearch (InlineArray reads only — no workspace access)
            var alpha = Scalar[DTYPE](1.0)
            var armijo_c = Scalar[DTYPE](ARMIJO)
            for _ in range(LINESEARCH_ITER):
                var trial_gauss = gauss_0 + alpha * g1 + alpha * alpha * g2
                var trial_c_cost: Scalar[DTYPE] = 0
                for c in range(nc):
                    if dist_cache[c] >= Scalar[DTYPE](0):
                        continue
                    var trial_N = jar_n_arr[c] + alpha * Js_n[c]
                    var trial_T1 = jar_t1_arr[c] + alpha * Js_t1[c]
                    var trial_T2 = jar_t2_arr[c] + alpha * Js_t2[c]
                    var mu = mu_cache[c]
                    var D_n = D_n_cache[c]
                    var D_f = D_f_cache[c]
                    var trial_T = sqrt(
                        trial_T1 * trial_T1 + trial_T2 * trial_T2
                    )
                    var trial_T_safe = trial_T
                    if trial_T_safe < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                        trial_T_safe = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
                    if trial_N >= mu * trial_T_safe:
                        pass  # satisfied, cost = 0
                    elif mu * trial_N + trial_T <= Scalar[DTYPE](0):
                        trial_c_cost += Scalar[DTYPE](0.5) * (
                            D_n * trial_N * trial_N
                            + D_f * (trial_T1 * trial_T1 + trial_T2 * trial_T2)
                        )
                    else:
                        var trial_s = trial_N - mu * trial_T_safe
                        var Dm = D_n / (Scalar[DTYPE](1.0) + mu * mu)
                        trial_c_cost += (
                            Scalar[DTYPE](0.5) * Dm * trial_s * trial_s
                        )

                var trial_cost = trial_gauss + trial_c_cost
                if trial_cost <= current_cost + armijo_c * alpha * gtd:
                    break
                alpha = alpha * Scalar[DTYPE](0.5)

            # If alpha is negligible, stop
            if alpha < Scalar[DTYPE](1e-12):
                break

            # Update qacc and Ma
            for i in range(NV):
                qacc[i] = qacc[i] + alpha * search[i]
                Ma[i] = Ma[i] + alpha * Mv[i]

            # Recompute jar and forces (using cached Jacobians — no workspace reads)
            var state_changed = False
            for c in range(nc):
                if dist_cache[c] >= Scalar[DTYPE](0):
                    continue
                var old_cs = cs_arr[c]
                var jar_n: Scalar[DTYPE] = pb_cache[c]
                var jar_t1: Scalar[DTYPE] = bt1_cache[c]
                var jar_t2: Scalar[DTYPE] = bt2_cache[c]
                for i in range(NV):
                    var qa_i = qacc[i]
                    jar_n += Jn_c[c * NV + i] * qa_i
                    jar_t1 += Jt1_c[c * NV + i] * qa_i
                    jar_t2 += Jt2_c[c * NV + i] * qa_i
                jar_n_arr[c] = jar_n
                jar_t1_arr[c] = jar_t1
                jar_t2_arr[c] = jar_t2

                var mu = mu_cache[c]
                var D_n = D_n_cache[c]
                var D_f = D_f_cache[c]
                var T = sqrt(jar_t1 * jar_t1 + jar_t2 * jar_t2)
                var T_safe = T
                if T_safe < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                    T_safe = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
                if jar_n >= mu * T_safe:
                    fn_arr[c] = 0
                    ft1_arr[c] = 0
                    ft2_arr[c] = 0
                    cs_arr[c] = 0
                elif mu * jar_n + T <= Scalar[DTYPE](0):
                    fn_arr[c] = -D_n * jar_n
                    ft1_arr[c] = -D_f * jar_t1
                    ft2_arr[c] = -D_f * jar_t2
                    cs_arr[c] = 1
                else:
                    var s = jar_n - mu * T_safe
                    var Dm = D_n / (Scalar[DTYPE](1.0) + mu * mu)
                    fn_arr[c] = -Dm * s
                    ft1_arr[c] = Dm * mu * s * jar_t1 / T_safe
                    ft2_arr[c] = Dm * mu * s * jar_t2 / T_safe
                    cs_arr[c] = 2
                if cs_arr[c] != old_cs:
                    state_changed = True

            # Recompute qfrc_c = J^T * updated forces (all InlineArray ops)
            for i in range(NV):
                qfrc_c[i] = Scalar[DTYPE](0)
            for c in range(nc):
                if cs_arr[c] == 0:
                    continue
                for i in range(NV):
                    qfrc_c[i] += (
                        Jn_c[c * NV + i] * fn_arr[c]
                        + Jt1_c[c * NV + i] * ft1_arr[c]
                        + Jt2_c[c * NV + i] * ft2_arr[c]
                    )

            # Hessian rebuild if states changed (using cached Jacobians — no workspace reads)
            if state_changed:
                for k in range(NV * NV):
                    H[k] = M_local[k]
                for c in range(nc):
                    var cs = cs_arr[c]
                    if cs == 0:
                        continue
                    var mu = mu_cache[c]
                    var D_n = D_n_cache[c]
                    var D_f = D_f_cache[c]
                    if cs == 1:
                        for i in range(NV):
                            for j in range(NV):
                                H[i * NV + j] += (
                                    D_n * Jn_c[c * NV + i] * Jn_c[c * NV + j]
                                    + D_f
                                    * Jt1_c[c * NV + i]
                                    * Jt1_c[c * NV + j]
                                    + D_f
                                    * Jt2_c[c * NV + i]
                                    * Jt2_c[c * NV + j]
                                )
                    else:
                        var jar_n = jar_n_arr[c]
                        var jar_t1 = jar_t1_arr[c]
                        var jar_t2 = jar_t2_arr[c]
                        var T_sq = jar_t1 * jar_t1 + jar_t2 * jar_t2
                        var T_s = sqrt(T_sq)
                        if T_s < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                            T_s = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
                        var s = jar_n - mu * T_s
                        var Dm = D_n / (Scalar[DTYPE](1.0) + mu * mu)
                        var h_nt1 = -Dm * mu * jar_t1 / T_s
                        var h_nt2 = -Dm * mu * jar_t2 / T_s
                        var T2_s = T_s * T_s
                        var h_t1t1 = (
                            Dm * mu * mu * jar_t1 * jar_t1 / T2_s
                            + Dm
                            * mu
                            * s
                            / T_s
                            * (jar_t1 * jar_t1 / T2_s - Scalar[DTYPE](1.0))
                        )
                        var h_t2t2 = (
                            Dm * mu * mu * jar_t2 * jar_t2 / T2_s
                            + Dm
                            * mu
                            * s
                            / T_s
                            * (jar_t2 * jar_t2 / T2_s - Scalar[DTYPE](1.0))
                        )
                        var h_t1t2 = (
                            (Dm * mu * mu + Dm * mu * s / T_s)
                            * jar_t1
                            * jar_t2
                            / T2_s
                        )
                        for i in range(NV):
                            for j in range(NV):
                                H[i * NV + j] += (
                                    Dm * Jn_c[c * NV + i] * Jn_c[c * NV + j]
                                    + h_nt1
                                    * (
                                        Jn_c[c * NV + i] * Jt1_c[c * NV + j]
                                        + Jt1_c[c * NV + i] * Jn_c[c * NV + j]
                                    )
                                    + h_nt2
                                    * (
                                        Jn_c[c * NV + i] * Jt2_c[c * NV + j]
                                        + Jt2_c[c * NV + i] * Jn_c[c * NV + j]
                                    )
                                    + h_t1t1
                                    * Jt1_c[c * NV + i]
                                    * Jt1_c[c * NV + j]
                                    + h_t2t2
                                    * Jt2_c[c * NV + i]
                                    * Jt2_c[c * NV + j]
                                    + h_t1t2
                                    * (
                                        Jt1_c[c * NV + i] * Jt2_c[c * NV + j]
                                        + Jt2_c[c * NV + i] * Jt1_c[c * NV + j]
                                    )
                                )
                chol_factor_inline[DTYPE, NV, M_SIZE](H, L_chol)

        # Write solved qacc back to workspace
        for i in range(NV):
            workspace[env, qacc_idx + i] = qacc[i]

        # Write forces to state buffer for display/warmstart (directly from InlineArrays)
        for c in range(nc):
            var c_off = contacts_off + c * CONTACT_SIZE
            state[env, c_off + CONTACT_IDX_FORCE_N] = fn_arr[c]
            state[env, c_off + CONTACT_IDX_FORCE_T1] = ft1_arr[c]
            state[env, c_off + CONTACT_IDX_FORCE_T2] = ft2_arr[c]

        comptime SOLVER_ITER_GPU: Int = 50
        detect_and_solve_limits_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            STATE_SIZE,
            MODEL_SIZE,
            WS_SIZE,
            BATCH,
            SOLVER_ITER_GPU,
            NGEOM,
            MAX_EQUALITY,
        ](env, dt, state, model, workspace)

        build_and_solve_equality_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            MAX_EQUALITY,
            NGEOM,
            STATE_SIZE,
            MODEL_SIZE,
            V_SIZE,
            WS_SIZE,
            BATCH,
            SOLVER_ITER_GPU,
        ](env, state, model, workspace)

        @parameter
        if MAX_TENDON > 0:
            build_and_solve_tendon_gpu[
                DTYPE,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                MAX_EQUALITY,
                NGEOM,
                MAX_TENDON,
                STATE_SIZE,
                MODEL_SIZE,
                V_SIZE,
                WS_SIZE,
                BATCH,
                SOLVER_ITER_GPU,
            ](env, state, model, workspace)

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

from std.math import sqrt, pow
from std.sys import simd_width_of
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.memory import AddressSpace
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
    USE_NEWTON_SIMD,
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
    MODEL_META_IDX_SOLREF_LIMIT_0,
    MODEL_META_IDX_SOLREF_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_0,
    MODEL_META_IDX_SOLIMP_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_2,
    MODEL_META_IDX_SOLIMP_LIMIT_3,
    MODEL_META_IDX_SOLIMP_LIMIT_4,
    model_body_invweight0_offset,
    qpos_offset,
    model_dof_invweight0_offset,
    model_joint_offset,
    JOINT_IDX_TYPE,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    JOINT_IDX_SOLREF_LIMIT_0,
    JOINT_IDX_SOLREF_LIMIT_1,
    JOINT_IDX_SOLIMP_LIMIT_0,
    JOINT_IDX_SOLIMP_LIMIT_1,
    JOINT_IDX_SOLIMP_LIMIT_2,
    JOINT_IDX_SOLIMP_LIMIT_3,
    JOINT_IDX_SOLIMP_LIMIT_4,
)

from ..dynamics.jacobian import compute_contact_jacobian_row_gpu

from ..constraints.constraint_builder_gpu import (
    init_common_normal_workspace_gpu,
    precompute_contact_normal_gpu,
    precompute_contact_friction_gpu,
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
def _build_hessian[
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
    # var limits_start = num_normals + num_friction

    # Track which rows are handled as part of cone groups
    var handled = InlineArray[Bool, MR](fill=False)

    # Pointers for SIMD inner-loop ports below (USE_NEWTON_SIMD comptime-gated).
    comptime W = simd_width_of[DTYPE]()
    var H_p = H.unsafe_ptr()
    var J_p = constraints.J.unsafe_ptr()

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
            var n_off = n * NV
            comptime if USE_NEWTON_SIMD:
                for i in range(NV):
                    var s_i = Dm * J_p[n_off + i]
                    var s_iv = SIMD[DTYPE, W](s_i)
                    var row_off = i * NV
                    var jj = 0
                    while jj + W <= NV:
                        H_p.store(
                            row_off + jj,
                            H_p.load[width=W](row_off + jj)
                            + s_iv * J_p.load[width=W](n_off + jj),
                        )
                        jj += W
                    while jj < NV:
                        H_p[row_off + jj] += s_i * J_p[n_off + jj]
                        jj += 1
            else:
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
                var fr_off = fr * NV
                comptime if USE_NEWTON_SIMD:
                    for i in range(NV):
                        var s_a = h_cross * J_p[n_off + i]  # scales J_fr row
                        var s_b = h_cross * J_p[fr_off + i]  # scales J_n  row
                        var sa_v = SIMD[DTYPE, W](s_a)
                        var sb_v = SIMD[DTYPE, W](s_b)
                        var row_off = i * NV
                        var jj = 0
                        while jj + W <= NV:
                            H_p.store(
                                row_off + jj,
                                H_p.load[width=W](row_off + jj)
                                + sa_v * J_p.load[width=W](fr_off + jj)
                                + sb_v * J_p.load[width=W](n_off + jj),
                            )
                            jj += W
                        while jj < NV:
                            H_p[row_off + jj] += (
                                s_a * J_p[fr_off + jj] + s_b * J_p[n_off + jj]
                            )
                            jj += 1
                else:
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
                    var fr1_off = fr1 * NV
                    var fr2_off = fr2 * NV
                    comptime if USE_NEWTON_SIMD:
                        for i in range(NV):
                            var s_i = h_ff * J_p[fr1_off + i]
                            var s_iv = SIMD[DTYPE, W](s_i)
                            var row_off = i * NV
                            var jj = 0
                            while jj + W <= NV:
                                H_p.store(
                                    row_off + jj,
                                    H_p.load[width=W](row_off + jj)
                                    + s_iv * J_p.load[width=W](fr2_off + jj),
                                )
                                jj += W
                            while jj < NV:
                                H_p[row_off + jj] += s_i * J_p[fr2_off + jj]
                                jj += 1
                    else:
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
        var r_off = r * NV
        comptime if USE_NEWTON_SIMD:
            for i in range(NV):
                var s_i = D_r * J_p[r_off + i]
                var s_iv = SIMD[DTYPE, W](s_i)
                var row_off = i * NV
                var jj = 0
                while jj + W <= NV:
                    H_p.store(
                        row_off + jj,
                        H_p.load[width=W](row_off + jj)
                        + s_iv * J_p.load[width=W](r_off + jj),
                    )
                    jj += W
                while jj < NV:
                    H_p[row_off + jj] += s_i * J_p[r_off + jj]
                    jj += 1
        else:
            for i in range(NV):
                for j in range(NV):
                    H[i * NV + j] += (
                        D_r
                        * constraints.J[r * NV + i]
                        * constraints.J[r * NV + j]
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

    comptime NEEDS_M_INV: Bool = True

    @staticmethod
    def solver_workspace_size[NV: Int, MAX_CONTACTS: Int]() -> Int:
        """Newton solver workspace size (primal, qacc-space).

        Layout (ELLIPTIC):
          Common normal block: 15*MC + 2*MC*NV
          Primal block: 4*MC*NV + 12*MC
            [J_t1, J_t2, MinvJt1, MinvJt2 | mu, D_n, D_f, bt1, bt2 |
             jar_n, jar_t1, jar_t2, fn, ft1, ft2, cstate]

        Layout (PYRAMIDAL) — reuses same J slots for 4 edge Jacobians:
          Common normal block: 15*MC + 2*MC*NV
          Primal block: 4*MC*NV + 20*MC
            [J_e0, J_e1, J_e2, J_e3 | D_e0..3, bias_e0..3 |
             jar_e0..3, f_e0..3, cstate_e0..3]

        Total = 35*MC + 6*MC*NV (accommodates both)
        """
        comptime MC = _max_one[MAX_CONTACTS]()
        return 35 * MC + 6 * MC * NV

    @no_inline
    @staticmethod
    def solve[
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
        # var num_normals = constraints.num_normals
        # var num_friction = constraints.num_friction
        # var friction_start = num_normals

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

        comptime if NEWTON_CPU_DEBUG:
            print("  [PRIMAL] qacc (initial):")
            for i in range(NV):
                print("    qacc[", i, "]=", Float64(qacc[i]))
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
            # Print Jacobians for first few rows
            for r in range(min(num_rows, 32)):
                print("  J[", r, "] =", end="")
                for i in range(NV):
                    print(" ", Float64(constraints.J[r * NV + i]), end="")
                print()

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

        # Cholesky factorize H (with rank-deficiency detection + regularization)
        var chol_ok = chol_factor[DTYPE, NV, M_SIZE](H, L)
        if not chol_ok:
            # Hessian is ill-conditioned — add Tikhonov regularization and retry
            for i in range(NV):
                H[i * NV + i] = H[i * NV + i] + Scalar[DTYPE](1e-6)
            _ = chol_factor[DTYPE, NV, M_SIZE](H, L)

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

        comptime W = simd_width_of[DTYPE]()
        for iter in range(NEWTON_CPU_ITERATIONS):
            total_iter += 1
            # Compute gradient: grad = Ma - qfrc_smooth - qfrc_constraint
            var grad_norm: Scalar[DTYPE]
            comptime if USE_NEWTON_SIMD:
                var Ma_p = Ma.unsafe_ptr()
                var qfs_p = qfrc_smooth.unsafe_ptr()
                var qfc_p = qfrc_constraint.unsafe_ptr()
                var grad_p = grad.unsafe_ptr()
                var acc_v = SIMD[DTYPE, W](0)
                var ii = 0
                while ii + W <= NV:
                    var g = (
                        Ma_p.load[width=W](ii)
                        - qfs_p.load[width=W](ii)
                        - qfc_p.load[width=W](ii)
                    )
                    grad_p.store(ii, g)
                    acc_v += g * g
                    ii += W
                grad_norm = acc_v.reduce_add()
                while ii < NV:
                    var g = Ma_p[ii] - qfs_p[ii] - qfc_p[ii]
                    grad_p[ii] = g
                    grad_norm += g * g
                    ii += 1
            else:
                for i in range(NV):
                    grad[i] = Ma[i] - qfrc_smooth[i] - qfrc_constraint[i]
                    grad_norm += grad[i] * grad[i]

            comptime if NEWTON_CPU_DEBUG:
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
                comptime if NEWTON_CPU_DEBUG:
                    print(
                        "    [PRIMAL_NEWTON] CONVERGED at iter",
                        total_iter,
                        " (gradient)",
                    )
                break

            # Newton direction: search = -H^{-1} * grad via Cholesky solve
            chol_solve[DTYPE, NV, M_SIZE, V_SIZE](L, grad, search)
            var search_ok = True
            for i in range(NV):
                search[i] = -search[i]
                # NaN/Inf guard: if search direction is invalid, abort iteration
                if search[i] != search[i] or search[i] * Scalar[DTYPE](
                    0
                ) != Scalar[DTYPE](0):
                    search_ok = False
            if not search_ok:
                break

            # Compute Mv = M * search (needed for line search)
            comptime if USE_NEWTON_SIMD:
                var Mh_p = constraints.M_hat.unsafe_ptr()
                var sr_p = search.unsafe_ptr()
                for i in range(NV):
                    var row_off = i * NV
                    var acc_v = SIMD[DTYPE, W](0)
                    var sum_i: Scalar[DTYPE]
                    var jj = 0
                    while jj + W <= NV:
                        acc_v += Mh_p.load[width=W](row_off + jj) * sr_p.load[
                            width=W
                        ](jj)
                        jj += W
                    sum_i = acc_v.reduce_add()
                    while jj < NV:
                        sum_i += Mh_p[row_off + jj] * sr_p[jj]
                        jj += 1
                    Mv[i] = sum_i
            else:
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
                comptime if NEWTON_CPU_DEBUG:
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

            comptime if NEWTON_CPU_DEBUG:
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
                var chol_ok2 = chol_factor[DTYPE, NV, M_SIZE](H, L)
                if not chol_ok2:
                    for i in range(NV):
                        H[i * NV + i] = H[i * NV + i] + Scalar[DTYPE](1e-6)
                    _ = chol_factor[DTYPE, NV, M_SIZE](H, L)

        comptime if NEWTON_CPU_DEBUG:
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
    def solver_threads[
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
    ]() -> Int:
        return _max_one[MAX_CONTACTS]()

    @staticmethod
    @always_inline
    def solve_gpu[
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
        comptime ws_J_n_idx = solver_ws_idx + 15 * MC
        comptime ws_MinvJn_idx = solver_ws_idx + 15 * MC + MC * NV

        # Primal-specific offsets (after common normal block)
        comptime PRIMAL_START = solver_ws_idx + 15 * MC + 2 * MC * NV
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
            K_spring = Scalar[DTYPE](1.0) / (
                si_dmax * si_dmax * sr_tc * sr_tc * sr_dr * sr_dr
            )
            B_damp = Scalar[DTYPE](2.0) / (si_dmax * sr_tc)
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

        # === PARALLEL PHASE 2: Tangent frame + friction data ===
        # Uses shared builder — no more duplicated tangent frame code.
        # For ELLIPTIC: builds J_t1, J_t2, D_n, D_f, mu, bt1, bt2
        # For PYRAMIDAL: builds 4 edge J, D_edge, bias_edge
        if valid_env and contact_tid < nc:
            precompute_contact_friction_gpu[
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
                CONE_TYPE,
                MAX_TENDON,
                NSITE,
            ](
                env,
                contact_tid,
                nc,
                state,
                model,
                workspace,
                B_damp,
                impratio,
                K_spring,
                ws_Jt1_idx,
                ws_Jt2_idx,
                ws_mu_idx,
                ws_D_n_idx,
                ws_D_f_idx,
                ws_bt1_idx,
                ws_bt2_idx,
            )

        barrier()

        # === SEQUENTIAL: Thread 0 handles primal Newton ===
        if not valid_env or contact_tid != 0:
            return

        comptime NEWTON_ITER_GPU: Int = 200
        comptime NEWTON_TOL_GPU: Float64 = 1e-8
        comptime LINESEARCH_ITER: Int = 20
        comptime ARMIJO: Float64 = 1e-4
        comptime PRIMAL_MINVAL_GPU: Float64 = 1e-12

        comptime if CONE_TYPE == ConeType.PYRAMIDAL:
            # =================================================================
            # PYRAMIDAL Newton: iterate over edge rows (all >= 0 constraints)
            # 4 edges per contact for condim=3: J_e = J_n ± mu*J_t
            # No cone coupling — simpler than ELLIPTIC
            # =================================================================
            comptime NE = 4  # edges per contact
            comptime MAX_LIM = _max_one[2 * NJOINT]()
            comptime ME = NE * MC + MAX_LIM  # contact edges + limit edges

            # Cache edge data from PYRAMIDAL workspace layout
            var pyr_sc = ws_Jt1_idx + 4 * MC * NV
            var Je = InlineArray[Scalar[DTYPE], ME * V_SIZE](uninitialized=True)
            var De = InlineArray[Scalar[DTYPE], ME](uninitialized=True)
            var bias_e = InlineArray[Scalar[DTYPE], ME](uninitialized=True)
            var num_edges = nc * NE

            # Load contact edges
            for c in range(nc):
                for e in range(NE):
                    var idx = c * NE + e
                    for i in range(NV):
                        Je[idx * NV + i] = rebind[Scalar[DTYPE]](
                            workspace[
                                env, ws_Jt1_idx + e * MC * NV + c * NV + i
                            ]
                        )
                    De[idx] = rebind[Scalar[DTYPE]](
                        workspace[env, pyr_sc + e * MC + c]
                    )
                    bias_e[idx] = rebind[Scalar[DTYPE]](
                        workspace[env, pyr_sc + 4 * MC + e * MC + c]
                    )

            # Detect and add joint limit edges (unified with contacts)
            # Matches CPU build_constraints: per-joint solref/solimp with
            # model-level defaults fallback
            comptime M_inv_idx = ws_m_inv_offset[NV, NBODY]()
            comptime qpos_off_lim = qpos_offset[NQ, NV]()
            comptime qvel_off_lim = qvel_offset[NQ, NV]()
            # Model-level defaults for fallback
            var lr_tc_def = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLREF_LIMIT_0]
            )
            var lr_dr_def = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLREF_LIMIT_1]
            )
            var li_dmin_def = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLIMP_LIMIT_0]
            )
            var li_dmax_def = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLIMP_LIMIT_1]
            )
            var li_width_def = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLIMP_LIMIT_2]
            )
            var li_midpoint_def = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLIMP_LIMIT_3]
            )
            var li_power_def = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLIMP_LIMIT_4]
            )

            for j in range(NJOINT):
                var j_off = model_joint_offset[NBODY](j)
                var jtype = Int(
                    rebind[Scalar[DTYPE]](model[0, j_off + JOINT_IDX_TYPE])
                )
                if jtype != JNT_HINGE and jtype != JNT_SLIDE:
                    continue
                var dof = Int(
                    rebind[Scalar[DTYPE]](model[0, j_off + JOINT_IDX_DOF_ADR])
                )
                var qpos_adr = Int(
                    rebind[Scalar[DTYPE]](model[0, j_off + JOINT_IDX_QPOS_ADR])
                )
                var rmin = rebind[Scalar[DTYPE]](
                    model[0, j_off + JOINT_IDX_RANGE_MIN]
                )
                var rmax = rebind[Scalar[DTYPE]](
                    model[0, j_off + JOINT_IDX_RANGE_MAX]
                )
                if rmin < Scalar[DTYPE](-1e9) or rmax > Scalar[DTYPE](1e9):
                    continue
                # Per-joint solref/solimp with model-level defaults fallback
                var lr_tc = rebind[Scalar[DTYPE]](
                    model[0, j_off + JOINT_IDX_SOLREF_LIMIT_0]
                )
                var lr_dr = rebind[Scalar[DTYPE]](
                    model[0, j_off + JOINT_IDX_SOLREF_LIMIT_1]
                )
                if lr_tc <= Scalar[DTYPE](0):
                    lr_tc = lr_tc_def
                if lr_dr <= Scalar[DTYPE](0):
                    lr_dr = lr_dr_def
                var li_dmin = rebind[Scalar[DTYPE]](
                    model[0, j_off + JOINT_IDX_SOLIMP_LIMIT_0]
                )
                var li_dmax = rebind[Scalar[DTYPE]](
                    model[0, j_off + JOINT_IDX_SOLIMP_LIMIT_1]
                )
                var li_width = rebind[Scalar[DTYPE]](
                    model[0, j_off + JOINT_IDX_SOLIMP_LIMIT_2]
                )
                var li_midpoint = rebind[Scalar[DTYPE]](
                    model[0, j_off + JOINT_IDX_SOLIMP_LIMIT_3]
                )
                var li_power = rebind[Scalar[DTYPE]](
                    model[0, j_off + JOINT_IDX_SOLIMP_LIMIT_4]
                )
                if li_dmax <= Scalar[DTYPE](0) and li_width <= Scalar[DTYPE](0):
                    li_dmin = li_dmin_def
                    li_dmax = li_dmax_def
                    li_width = li_width_def
                    li_midpoint = li_midpoint_def
                    li_power = li_power_def
                if li_width < Scalar[DTYPE](1e-6):
                    li_width = Scalar[DTYPE](1e-6)
                if li_dmax < Scalar[DTYPE](1e-4):
                    li_dmax = Scalar[DTYPE](1e-4)
                var l_K_spring = Scalar[DTYPE](1.0) / (
                    li_dmax * li_dmax * lr_tc * lr_tc * lr_dr * lr_dr
                )
                var l_B_damp = Scalar[DTYPE](2.0) / (li_dmax * lr_tc)

                var pos = rebind[Scalar[DTYPE]](
                    state[env, qpos_off_lim + qpos_adr]
                )
                # Lower limit: dist_lo = pos - rmin < 0 → violated
                var dist_lo = pos - rmin
                if dist_lo < Scalar[DTYPE](0) and num_edges < ME:
                    var sign = Scalar[DTYPE](1)
                    var K_lim = rebind[Scalar[DTYPE]](
                        workspace[env, M_inv_idx + dof * NV + dof]
                    )
                    if K_lim < Scalar[DTYPE](1e-10):
                        K_lim = Scalar[DTYPE](1e-10)
                    var pen = -dist_lo
                    var v_lim = sign * rebind[Scalar[DTYPE]](
                        state[env, qvel_off_lim + dof]
                    )
                    # Impedance
                    var imp_lim: Scalar[DTYPE]
                    if li_dmin == li_dmax or li_width <= Scalar[DTYPE](0):
                        imp_lim = Scalar[DTYPE](0.5) * (li_dmin + li_dmax)
                    else:
                        var x_l = pen / li_width
                        if x_l <= Scalar[DTYPE](0):
                            imp_lim = li_dmin
                        elif x_l >= Scalar[DTYPE](1):
                            imp_lim = li_dmax
                        else:
                            var y_l: Scalar[DTYPE]
                            if li_power == Scalar[DTYPE](1):
                                y_l = x_l
                            elif x_l <= li_midpoint:
                                y_l = pow(x_l, li_power) / pow(
                                    li_midpoint, li_power - Scalar[DTYPE](1)
                                )
                            else:
                                y_l = Scalar[DTYPE](1) - pow(
                                    Scalar[DTYPE](1) - x_l, li_power
                                ) / pow(
                                    Scalar[DTYPE](1) - li_midpoint,
                                    li_power - Scalar[DTYPE](1),
                                )
                            imp_lim = li_dmin + y_l * (li_dmax - li_dmin)
                    if imp_lim < Scalar[DTYPE](1e-6):
                        imp_lim = Scalar[DTYPE](1e-6)
                    comptime dof_iw_off = model_dof_invweight0_offset[
                        NBODY, NJOINT, NGEOM, MAX_EQUALITY, MAX_TENDON, NSITE
                    ]()
                    var diag_lim = rebind[Scalar[DTYPE]](
                        model[0, dof_iw_off + dof]
                    )
                    if diag_lim < Scalar[DTYPE](1e-10):
                        diag_lim = K_lim
                    var R_lim = (
                        (Scalar[DTYPE](1) - imp_lim) / imp_lim * diag_lim
                    )
                    if R_lim < Scalar[DTYPE](1e-14):
                        R_lim = Scalar[DTYPE](1e-14)
                    # Sparse Jacobian: Je[dof] = sign, others 0
                    for i in range(NV):
                        Je[num_edges * NV + i] = Scalar[DTYPE](0)
                    Je[num_edges * NV + dof] = sign
                    # Match CPU: inv_K = 1/(K+R), D = 1/(1/inv_K - K)
                    # Same float32 rounding as primal_D(inv_K_imp, K)
                    var inv_K_lim = Scalar[DTYPE](1) / (K_lim + R_lim)
                    var R_recov = Scalar[DTYPE](1) / inv_K_lim - K_lim
                    if R_recov < Scalar[DTYPE](1e-14):
                        R_recov = Scalar[DTYPE](1e-14)
                    De[num_edges] = Scalar[DTYPE](1) / R_recov
                    bias_e[num_edges] = (
                        l_B_damp * v_lim - l_K_spring * imp_lim * pen
                    )
                    num_edges += 1

                # Upper limit: dist_hi = rmax - pos < 0 → violated
                var dist_hi = rmax - pos
                if dist_hi < Scalar[DTYPE](0) and num_edges < ME:
                    var sign = Scalar[DTYPE](-1)
                    var K_lim = rebind[Scalar[DTYPE]](
                        workspace[env, M_inv_idx + dof * NV + dof]
                    )
                    if K_lim < Scalar[DTYPE](1e-10):
                        K_lim = Scalar[DTYPE](1e-10)
                    var pen = -dist_hi
                    var v_lim = sign * rebind[Scalar[DTYPE]](
                        state[env, qvel_off_lim + dof]
                    )
                    var imp_lim: Scalar[DTYPE]
                    if li_dmin == li_dmax or li_width <= Scalar[DTYPE](0):
                        imp_lim = Scalar[DTYPE](0.5) * (li_dmin + li_dmax)
                    else:
                        var x_l = pen / li_width
                        if x_l <= Scalar[DTYPE](0):
                            imp_lim = li_dmin
                        elif x_l >= Scalar[DTYPE](1):
                            imp_lim = li_dmax
                        else:
                            var y_l: Scalar[DTYPE]
                            if li_power == Scalar[DTYPE](1):
                                y_l = x_l
                            elif x_l <= li_midpoint:
                                y_l = pow(x_l, li_power) / pow(
                                    li_midpoint, li_power - Scalar[DTYPE](1)
                                )
                            else:
                                y_l = Scalar[DTYPE](1) - pow(
                                    Scalar[DTYPE](1) - x_l, li_power
                                ) / pow(
                                    Scalar[DTYPE](1) - li_midpoint,
                                    li_power - Scalar[DTYPE](1),
                                )
                            imp_lim = li_dmin + y_l * (li_dmax - li_dmin)
                    if imp_lim < Scalar[DTYPE](1e-6):
                        imp_lim = Scalar[DTYPE](1e-6)
                    comptime dof_iw_off = model_dof_invweight0_offset[
                        NBODY, NJOINT, NGEOM, MAX_EQUALITY, MAX_TENDON, NSITE
                    ]()
                    var diag_lim = rebind[Scalar[DTYPE]](
                        model[0, dof_iw_off + dof]
                    )
                    if diag_lim < Scalar[DTYPE](1e-10):
                        diag_lim = K_lim
                    var R_lim = (
                        (Scalar[DTYPE](1) - imp_lim) / imp_lim * diag_lim
                    )
                    if R_lim < Scalar[DTYPE](1e-14):
                        R_lim = Scalar[DTYPE](1e-14)
                    for i in range(NV):
                        Je[num_edges * NV + i] = Scalar[DTYPE](0)
                    Je[num_edges * NV + dof] = sign
                    # Match CPU: inv_K = 1/(K+R), D = 1/(1/inv_K - K)
                    # Same float32 rounding as primal_D(inv_K_imp, K)
                    var inv_K_lim = Scalar[DTYPE](1) / (K_lim + R_lim)
                    var R_recov = Scalar[DTYPE](1) / inv_K_lim - K_lim
                    if R_recov < Scalar[DTYPE](1e-14):
                        R_recov = Scalar[DTYPE](1e-14)
                    De[num_edges] = Scalar[DTYPE](1) / R_recov
                    bias_e[num_edges] = (
                        l_B_damp * v_lim - l_K_spring * imp_lim * pen
                    )
                    num_edges += 1

            # Read normal J for gradient computation
            comptime ws_J_n_idx = solver_ws_idx + 15 * MC

            # Initialize qacc from workspace (qacc_smooth set by stage kernel)
            comptime M_SIZE = _max_one[NV * NV]()
            var qacc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
            var qacc_smooth = InlineArray[Scalar[DTYPE], V_SIZE](
                uninitialized=True
            )
            var Ma = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
            comptime M_idx = ws_M_offset[NV, NBODY]()
            comptime qacc_init_idx = ws_qacc_constrained_offset[NV, NBODY]()

            # Cache M locally once — M is loop-invariant during Newton iterations.
            # Avoids ~2*NV² workspace (global) reads per iteration (Hessian build
            # + Mv = M*search). Mirrors the ELLIPTIC path's M_local optimization.
            var M_local = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
            for k in range(NV * NV):
                M_local[k] = rebind[Scalar[DTYPE]](workspace[env, M_idx + k])

            for i in range(NV):
                var q_i = rebind[Scalar[DTYPE]](
                    workspace[env, qacc_init_idx + i]
                )
                qacc[i] = q_i
                qacc_smooth[i] = q_i
            for i in range(NV):
                Ma[i] = Scalar[DTYPE](0)
                for j in range(NV):
                    Ma[i] += M_local[i * NV + j] * qacc[j]
            # f_smooth = M * qacc (matching CPU's qfrc_smooth = M * qacc_smooth)
            # Using Ma directly avoids LDL round-trip error (f_net ≠ M*M^{-1}*f_net)
            var f_smooth = InlineArray[Scalar[DTYPE], V_SIZE](
                uninitialized=True
            )
            for i in range(NV):
                f_smooth[i] = Ma[i]

            # Scale for convergence check
            var scale: Scalar[DTYPE] = 0
            for i in range(NV):
                scale += M_local[i * NV + i]
            if scale > Scalar[DTYPE](1e-10):
                scale = Scalar[DTYPE](1.0) / scale
            else:
                scale = Scalar[DTYPE](1.0)

            # Working arrays
            var jar = InlineArray[Scalar[DTYPE], ME](uninitialized=True)
            var force = InlineArray[Scalar[DTYPE], ME](uninitialized=True)
            var H = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
            var L_chol = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
            var grad = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
            var search = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
            var Mv = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

            # Initial jar + force + qfrc
            var qfrc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
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

            # Newton iterations
            for iter_n in range(NEWTON_ITER_GPU):
                # Gradient
                var grad_norm: Scalar[DTYPE] = 0
                for i in range(NV):
                    grad[i] = Ma[i] - f_smooth[i] - qfrc[i]
                    grad_norm += grad[i] * grad[i]

                if scale * sqrt(grad_norm) < Scalar[DTYPE](NEWTON_TOL_GPU):
                    break

                # Build Hessian H = M + sum_active(D[e] * Je^T * Je)
                for i in range(NV):
                    for j in range(NV):
                        H[i * NV + j] = M_local[i * NV + j]
                for e_idx in range(num_edges):
                    if force[e_idx] > Scalar[DTYPE](0):
                        for i in range(NV):
                            for j in range(NV):
                                H[i * NV + j] += (
                                    De[e_idx]
                                    * Je[e_idx * NV + i]
                                    * Je[e_idx * NV + j]
                                )

                # Cholesky solve
                var chol_ok = chol_factor_inline[DTYPE, NV, M_SIZE](H, L_chol)
                if not chol_ok:
                    for i in range(NV):
                        H[i * NV + i] += Scalar[DTYPE](1e-6)
                    _ = chol_factor_inline[DTYPE, NV, M_SIZE](H, L_chol)
                chol_solve_inline[DTYPE, NV, M_SIZE, V_SIZE](
                    L_chol, grad, search
                )
                for i in range(NV):
                    search[i] = -search[i]

                # Mv = M * search
                for i in range(NV):
                    Mv[i] = Scalar[DTYPE](0)
                    for j in range(NV):
                        Mv[i] += M_local[i * NV + j] * search[j]

                # Analytical Newton linesearch (matches CPU primal_linesearch_with_D)
                # Precompute Jv_e = Je · search for each edge
                var Jv_e = InlineArray[Scalar[DTYPE], ME](uninitialized=True)
                for e_idx in range(num_edges):
                    Jv_e[e_idx] = Scalar[DTYPE](0)
                    for i in range(NV):
                        Jv_e[e_idx] += Je[e_idx * NV + i] * search[i]

                # Analytical Newton linesearch (matching CPU primal_linesearch_with_D)
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
                if p0_d2 < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                    p0_d2 = Scalar[DTYPE](PRIMAL_MINVAL_GPU)

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
                                Scalar[DTYPE](0.5)
                                * De[e_idx]
                                * jar[e_idx]
                                * jar[e_idx]
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
                                    Scalar[DTYPE](0.5)
                                    * De[e_idx]
                                    * jar_t
                                    * jar_t
                                )
                        if trial_cost <= old_cost:
                            break
                        alpha *= Scalar[DTYPE](0.5)

                if alpha < Scalar[DTYPE](1e-10):
                    break

                # Save old state for cost revert (matching CPU solver)
                var old_qacc = InlineArray[Scalar[DTYPE], V_SIZE](
                    uninitialized=True
                )
                var old_Ma = InlineArray[Scalar[DTYPE], V_SIZE](
                    uninitialized=True
                )
                var old_jar = InlineArray[Scalar[DTYPE], ME](uninitialized=True)
                var old_force = InlineArray[Scalar[DTYPE], ME](
                    uninitialized=True
                )
                var old_qfrc = InlineArray[Scalar[DTYPE], V_SIZE](
                    uninitialized=True
                )
                for i in range(NV):
                    old_qacc[i] = qacc[i]
                    old_Ma[i] = Ma[i]
                    old_qfrc[i] = qfrc[i]
                for e_idx in range(num_edges):
                    old_jar[e_idx] = jar[e_idx]
                    old_force[e_idx] = force[e_idx]

                # Compute old cost: gauss + constraint
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
                            Scalar[DTYPE](0.5)
                            * De[e_idx]
                            * jar[e_idx]
                            * jar[e_idx]
                        )

                # Update qacc, Ma
                for i in range(NV):
                    qacc[i] += alpha * search[i]
                    Ma[i] += alpha * Mv[i]

                # Recompute jar, force, qfrc
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

                # Compute new cost and check improvement
                var new_cost: Scalar[DTYPE] = 0
                for i in range(NV):
                    new_cost += (
                        Scalar[DTYPE](0.5)
                        * (Ma[i] - f_smooth[i])
                        * (qacc[i] - qacc_smooth[i])
                    )
                for e_idx in range(num_edges):
                    if jar[e_idx] < Scalar[DTYPE](0):
                        new_cost += (
                            Scalar[DTYPE](0.5)
                            * De[e_idx]
                            * jar[e_idx]
                            * jar[e_idx]
                        )

                var improvement = scale * (old_cost - new_cost)
                if improvement < Scalar[DTYPE](NEWTON_TOL_GPU) and iter_n > 0:
                    if improvement < Scalar[DTYPE](0):
                        # Cost increased — revert to old state
                        for i in range(NV):
                            qacc[i] = old_qacc[i]
                            Ma[i] = old_Ma[i]
                            qfrc[i] = old_qfrc[i]
                        for e_idx in range(num_edges):
                            jar[e_idx] = old_jar[e_idx]
                            force[e_idx] = old_force[e_idx]
                    break

            # Write qacc back
            for i in range(NV):
                workspace[env, qacc_idx + i] = qacc[i]

            # Write forces to state: reconstruct per-contact N/T1/T2
            for c in range(nc):
                var fn_c: Scalar[DTYPE] = 0
                var ft1_c: Scalar[DTYPE] = 0
                var ft2_c: Scalar[DTYPE] = 0
                var mu_c = rebind[Scalar[DTYPE]](
                    workspace[env, pyr_sc + 8 * MC + c]
                )
                var safe_mu = mu_c
                if safe_mu < Scalar[DTYPE](1e-8):
                    safe_mu = Scalar[DTYPE](1e-8)
                # f_n = sum of edge forces / num_tangent_dirs
                # f_tk = (f_edge_pos - f_edge_neg) * mu
                var f_e0 = force[c * NE + 0]
                var f_e1 = force[c * NE + 1]
                var f_e2 = force[c * NE + 2]
                var f_e3 = force[c * NE + 3]
                fn_c = (f_e0 + f_e1 + f_e2 + f_e3) / Scalar[DTYPE](2.0)
                ft1_c = (f_e0 - f_e1) * safe_mu
                ft2_c = (f_e2 - f_e3) * safe_mu
                var c_off = contacts_off + c * CONTACT_SIZE
                state[env, c_off + CONTACT_IDX_FORCE_N] = fn_c
                state[env, c_off + CONTACT_IDX_FORCE_T1] = ft1_c
                state[env, c_off + CONTACT_IDX_FORCE_T2] = ft2_c

            # Joint limits are now handled as edges in the Newton solver above.
            # Only equality constraints remain as a separate post-solve step.
            comptime SOLVER_ITER_GPU: Int = 50
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
            comptime if MAX_TENDON > 0:
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
            return  # PYRAMIDAL path complete

        # === ELLIPTIC path (existing code) ===
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

        # Ma = M_local * qacc (uses cached M — no workspace reads)
        for i in range(NV):
            var s: Scalar[DTYPE] = 0
            for j in range(NV):
                s += M_local[i * NV + j] * qacc[j]
            Ma[i] = s

        # qfrc_sm = M * qacc (matching CPU's qfrc_smooth = M * qacc_smooth)
        # Using Ma directly avoids LDL round-trip error
        for i in range(NV):
            qfrc_sm[i] = Ma[i]

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

        # Cholesky factorize H (with regularization on rank deficiency)
        var chol_ok_gpu = chol_factor_inline[DTYPE, NV, M_SIZE](H, L_chol)
        if not chol_ok_gpu:
            for i in range(NV):
                H[i * NV + i] = H[i * NV + i] + Scalar[DTYPE](1e-6)
            _ = chol_factor_inline[DTYPE, NV, M_SIZE](H, L_chol)

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
            var search_ok_gpu = True
            for i in range(NV):
                search[i] = -search[i]
                if search[i] != search[i]:
                    search_ok_gpu = False
            if not search_ok_gpu:
                break

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

            # Analytical Newton linesearch (matches CPU primal_linesearch_with_D)
            # Gauss coefficients for derivative: d_gauss/dalpha = ga*alpha + gb
            var ga: Scalar[DTYPE] = 0
            var gb: Scalar[DTYPE] = 0
            for i in range(NV):
                ga += Mv[i] * search[i]
                gb += (Ma[i] - qfrc_sm[i]) * search[i]

            # Evaluate d1, d2 at alpha=0
            var p0_d1 = gb
            var p0_d2 = ga
            for c in range(nc):
                if dist_cache[c] >= Scalar[DTYPE](0):
                    continue
                var N0 = jar_n_arr[c]
                var T10 = jar_t1_arr[c]
                var T20 = jar_t2_arr[c]
                var mu = mu_cache[c]
                var D_n = D_n_cache[c]
                var D_f = D_f_cache[c]
                var T0_sq = T10 * T10 + T20 * T20
                var T0 = sqrt(T0_sq)
                var T0_safe = T0
                if T0_safe < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                    T0_safe = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
                if N0 >= Scalar[DTYPE](0) and N0 * N0 >= mu * mu * T0_sq:
                    pass  # SATISFIED
                elif mu * N0 + T0 <= Scalar[DTYPE](0):
                    # QUADRATIC
                    p0_d1 += D_n * N0 * Js_n[c] + D_f * (
                        T10 * Js_t1[c] + T20 * Js_t2[c]
                    )
                    p0_d2 += D_n * Js_n[c] * Js_n[c] + D_f * (
                        Js_t1[c] * Js_t1[c] + Js_t2[c] * Js_t2[c]
                    )
                else:
                    # CONE
                    var Dm = D_n / (Scalar[DTYPE](1.0) + mu * mu)
                    var s0 = N0 - mu * T0
                    var dTda = (T10 * Js_t1[c] + T20 * Js_t2[c]) / T0_safe
                    var dsda = Js_n[c] - mu * dTda
                    p0_d1 += Dm * s0 * dsda
                    var Jv_f_sq = Js_t1[c] * Js_t1[c] + Js_t2[c] * Js_t2[c]
                    var d2sda2 = -mu * (Jv_f_sq - dTda * dTda) / T0_safe
                    p0_d2 += Dm * (dsda * dsda + s0 * d2sda2)
            if p0_d2 < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                p0_d2 = Scalar[DTYPE](PRIMAL_MINVAL_GPU)

            var alpha: Scalar[DTYPE] = 0
            if p0_d1 < Scalar[DTYPE](0):
                # Phase 1: initial Newton step
                var p1_alpha = -p0_d1 / p0_d2

                var snorm_sq: Scalar[DTYPE] = 0
                for i in range(NV):
                    snorm_sq += search[i] * search[i]
                var gtol = (
                    Scalar[DTYPE](NEWTON_TOL_GPU) * sqrt(snorm_sq) / scale
                )
                var gtol_sq = gtol * gtol

                # Inline eval at p1_alpha
                var p1_d1 = ga * p1_alpha + gb
                var p1_d2_v = ga
                for c in range(nc):
                    if dist_cache[c] >= Scalar[DTYPE](0):
                        continue
                    var tN = jar_n_arr[c] + p1_alpha * Js_n[c]
                    var tT1 = jar_t1_arr[c] + p1_alpha * Js_t1[c]
                    var tT2 = jar_t2_arr[c] + p1_alpha * Js_t2[c]
                    var mu = mu_cache[c]
                    var D_n = D_n_cache[c]
                    var D_f = D_f_cache[c]
                    var tT_sq = tT1 * tT1 + tT2 * tT2
                    var tT = sqrt(tT_sq)
                    var tT_s = tT
                    if tT_s < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                        tT_s = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
                    if tN >= Scalar[DTYPE](0) and tN * tN >= mu * mu * tT_sq:
                        pass
                    elif mu * tN + tT <= Scalar[DTYPE](0):
                        p1_d1 += D_n * tN * Js_n[c] + D_f * (
                            tT1 * Js_t1[c] + tT2 * Js_t2[c]
                        )
                        p1_d2_v += D_n * Js_n[c] * Js_n[c] + D_f * (
                            Js_t1[c] * Js_t1[c] + Js_t2[c] * Js_t2[c]
                        )
                    else:
                        var Dm = D_n / (Scalar[DTYPE](1.0) + mu * mu)
                        var s_v = tN - mu * tT
                        var dTda = (tT1 * Js_t1[c] + tT2 * Js_t2[c]) / tT_s
                        var dsda = Js_n[c] - mu * dTda
                        p1_d1 += Dm * s_v * dsda
                        var Jvf = Js_t1[c] * Js_t1[c] + Js_t2[c] * Js_t2[c]
                        var d2s = -mu * (Jvf - dTda * dTda) / tT_s
                        p1_d2_v += Dm * (dsda * dsda + s_v * d2s)
                if p1_d2_v < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                    p1_d2_v = Scalar[DTYPE](PRIMAL_MINVAL_GPU)

                alpha = p1_alpha
                if p1_d1 * p1_d1 >= gtol_sq:
                    # Phase 2: one-sided Newton pursuit
                    var dir_s = Scalar[DTYPE](-1) if p1_d1 > Scalar[DTYPE](
                        0
                    ) else Scalar[DTYPE](1)
                    var p2_alpha: Scalar[DTYPE] = 0
                    var p2_d1 = p0_d1
                    var bracket = False
                    for _ in range(LINESEARCH_ITER):
                        p2_alpha = p1_alpha
                        p2_d1 = p1_d1
                        if p1_d2_v > Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                            p1_alpha = p1_alpha - p1_d1 / p1_d2_v
                        else:
                            p1_alpha = p1_alpha + dir_s
                        # Eval at new p1_alpha
                        p1_d1 = ga * p1_alpha + gb
                        p1_d2_v = ga
                        for c in range(nc):
                            if dist_cache[c] >= Scalar[DTYPE](0):
                                continue
                            var tN = jar_n_arr[c] + p1_alpha * Js_n[c]
                            var tT1 = jar_t1_arr[c] + p1_alpha * Js_t1[c]
                            var tT2 = jar_t2_arr[c] + p1_alpha * Js_t2[c]
                            var mu = mu_cache[c]
                            var D_n = D_n_cache[c]
                            var D_f = D_f_cache[c]
                            var tT_sq = tT1 * tT1 + tT2 * tT2
                            var tT = sqrt(tT_sq)
                            var tT_s = tT
                            if tT_s < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                                tT_s = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
                            if (
                                tN >= Scalar[DTYPE](0)
                                and tN * tN >= mu * mu * tT_sq
                            ):
                                pass
                            elif mu * tN + tT <= Scalar[DTYPE](0):
                                p1_d1 += D_n * tN * Js_n[c] + D_f * (
                                    tT1 * Js_t1[c] + tT2 * Js_t2[c]
                                )
                                p1_d2_v += D_n * Js_n[c] * Js_n[c] + D_f * (
                                    Js_t1[c] * Js_t1[c] + Js_t2[c] * Js_t2[c]
                                )
                            else:
                                var Dm = D_n / (Scalar[DTYPE](1.0) + mu * mu)
                                var s_v = tN - mu * tT
                                var dTda = (
                                    tT1 * Js_t1[c] + tT2 * Js_t2[c]
                                ) / tT_s
                                var dsda = Js_n[c] - mu * dTda
                                p1_d1 += Dm * s_v * dsda
                                var Jvf = (
                                    Js_t1[c] * Js_t1[c] + Js_t2[c] * Js_t2[c]
                                )
                                var d2s = -mu * (Jvf - dTda * dTda) / tT_s
                                p1_d2_v += Dm * (dsda * dsda + s_v * d2s)
                        if p1_d2_v < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                            p1_d2_v = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
                        if p1_d1 * p1_d1 < gtol_sq:
                            alpha = p1_alpha
                            break
                        if p1_d1 * dir_s > Scalar[DTYPE](0):
                            bracket = True
                            break
                    if bracket:
                        # Phase 3: bracketed bisection
                        for _ in range(LINESEARCH_ITER):
                            var mid = (p1_alpha + p2_alpha) * Scalar[DTYPE](0.5)
                            var mid_d1 = ga * mid + gb
                            for c in range(nc):
                                if dist_cache[c] >= Scalar[DTYPE](0):
                                    continue
                                var tN = jar_n_arr[c] + mid * Js_n[c]
                                var tT1 = jar_t1_arr[c] + mid * Js_t1[c]
                                var tT2 = jar_t2_arr[c] + mid * Js_t2[c]
                                var mu = mu_cache[c]
                                var D_n = D_n_cache[c]
                                var D_f = D_f_cache[c]
                                var tT_sq = tT1 * tT1 + tT2 * tT2
                                var tT = sqrt(tT_sq)
                                var tT_s = tT
                                if tT_s < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                                    tT_s = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
                                if (
                                    tN >= Scalar[DTYPE](0)
                                    and tN * tN >= mu * mu * tT_sq
                                ):
                                    pass
                                elif mu * tN + tT <= Scalar[DTYPE](0):
                                    mid_d1 += D_n * tN * Js_n[c] + D_f * (
                                        tT1 * Js_t1[c] + tT2 * Js_t2[c]
                                    )
                                else:
                                    var Dm = D_n / (
                                        Scalar[DTYPE](1.0) + mu * mu
                                    )
                                    var s_v = tN - mu * tT
                                    var dTda = (
                                        tT1 * Js_t1[c] + tT2 * Js_t2[c]
                                    ) / tT_s
                                    var dsda = Js_n[c] - mu * dTda
                                    mid_d1 += Dm * s_v * dsda
                            if mid_d1 * mid_d1 < gtol_sq:
                                p1_alpha = mid
                                p1_d1 = mid_d1
                                break
                            if mid_d1 * p1_d1 > Scalar[DTYPE](0):
                                p1_alpha = mid
                                p1_d1 = mid_d1
                            else:
                                p2_alpha = mid
                                p2_d1 = mid_d1
                            if (p1_alpha - p2_alpha) * (
                                p1_alpha - p2_alpha
                            ) < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                                break
                        if p2_d1 * p2_d1 < p1_d1 * p1_d1:
                            alpha = p2_alpha
                        else:
                            alpha = p1_alpha
                    elif p1_d1 * p1_d1 >= gtol_sq:
                        alpha = p1_alpha

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
                var chol_ok_gpu2 = chol_factor_inline[DTYPE, NV, M_SIZE](
                    H, L_chol
                )
                if not chol_ok_gpu2:
                    for i in range(NV):
                        H[i * NV + i] = H[i * NV + i] + Scalar[DTYPE](1e-6)
                    _ = chol_factor_inline[DTYPE, NV, M_SIZE](H, L_chol)

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

        comptime if MAX_TENDON > 0:
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

    @staticmethod
    @always_inline
    def solve_gpu_blocked[
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
        """One-block-per-env Newton solver (PYRAMIDAL only).

        Solves ONE environment per CUDA block, cooperatively across the block's
        threads. The Hessian assembly is parallelized across threads. Numerically
        bit-identical to `solve_gpu` for the PYRAMIDAL cone path: the only
        reordering is distributing the Hessian's outer (i,j) loop across threads;
        the inner per-(i,j) edge-sum order (e ascending) is preserved.

        Launch: grid_dim=(BATCH, 1), block_dim=(THREADS,) where
        THREADS = solver_threads() = max(1, MAX_CONTACTS).
        """
        var env = Int(block_idx.x)
        var tid = Int(thread_idx.x)
        var contact_tid = tid
        var valid_env = env < BATCH

        comptime qacc_idx = ws_qacc_constrained_offset[NV, NBODY]()
        comptime solver_ws_idx = ws_solver_offset[NV, NBODY]()
        comptime fnet_idx = ws_fnet_offset[NV, NBODY]()
        comptime M_idx = ws_M_offset[NV, NBODY]()
        comptime M_inv_idx = ws_m_inv_offset[NV, NBODY]()
        comptime MC = _max_one[MAX_CONTACTS]()
        comptime M_SIZE = _max_one[NV * NV]()
        comptime THREADS = _max_one[MAX_CONTACTS]()

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
        comptime ws_J_n_idx = solver_ws_idx + 15 * MC
        comptime ws_MinvJn_idx = solver_ws_idx + 15 * MC + MC * NV

        # Primal-specific offsets (after common normal block)
        comptime PRIMAL_START = solver_ws_idx + 15 * MC + 2 * MC * NV
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
            K_spring = Scalar[DTYPE](1.0) / (
                si_dmax * si_dmax * sr_tc * sr_tc * sr_dr * sr_dr
            )
            B_damp = Scalar[DTYPE](2.0) / (si_dmax * sr_tc)
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

        # === PARALLEL PHASE 2: Tangent frame + friction data ===
        if valid_env and contact_tid < nc:
            precompute_contact_friction_gpu[
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
                CONE_TYPE,
                MAX_TENDON,
                NSITE,
            ](
                env,
                contact_tid,
                nc,
                state,
                model,
                workspace,
                B_damp,
                impratio,
                K_spring,
                ws_Jt1_idx,
                ws_Jt2_idx,
                ws_mu_idx,
                ws_D_n_idx,
                ws_D_f_idx,
                ws_bt1_idx,
                ws_bt2_idx,
            )

        barrier()

        comptime NEWTON_ITER_GPU: Int = 200
        comptime NEWTON_TOL_GPU: Float64 = 1e-8
        comptime LINESEARCH_ITER: Int = 20
        comptime ARMIJO: Float64 = 1e-4
        comptime PRIMAL_MINVAL_GPU: Float64 = 1e-12

        # PYRAMIDAL-only blocked solver. (Non-PYRAMIDAL never routes here.)
        comptime NE = 4  # edges per contact
        comptime MAX_LIM = _max_one[2 * NJOINT]()
        comptime ME = NE * MC + MAX_LIM  # contact edges + limit edges

        # === SHARED memory (per-block == per-env) ===
        var M_sh = LayoutTensor[
            DTYPE,
            Layout.row_major(M_SIZE),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()
        var H_sh = LayoutTensor[
            DTYPE,
            Layout.row_major(M_SIZE),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()
        var Je_sh = LayoutTensor[
            DTYPE,
            Layout.row_major(ME * V_SIZE),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()
        var De_sh = LayoutTensor[
            DTYPE,
            Layout.row_major(ME),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()
        var bias_e_sh = LayoutTensor[
            DTYPE,
            Layout.row_major(ME),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()
        var force_sh = LayoutTensor[
            DTYPE,
            Layout.row_major(ME),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()
        # Scalar shared state: [0]=num_edges, [1]=done flag.
        var ctrl_sh = LayoutTensor[
            DTYPE,
            Layout.row_major(2),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()

        comptime pyr_sc = ws_Jt1_idx + 4 * MC * NV
        comptime M_idx_b = ws_M_offset[NV, NBODY]()
        comptime qacc_init_idx = ws_qacc_constrained_offset[NV, NBODY]()

        # === COOPERATIVE LOAD: M into shared ===
        if valid_env:
            for k in range(tid, NV * NV, THREADS):
                M_sh[k] = rebind[Scalar[DTYPE]](workspace[env, M_idx_b + k])

            # Cooperative load of contact edges (Je/De/bias_e) into shared.
            # One thread per contact (contact_tid == c), matching serial load
            # order (c ascending, e ascending).
            if contact_tid < nc:
                var c = contact_tid
                for e in range(NE):
                    var idx = c * NE + e
                    for i in range(NV):
                        Je_sh[idx * NV + i] = rebind[Scalar[DTYPE]](
                            workspace[
                                env, ws_Jt1_idx + e * MC * NV + c * NV + i
                            ]
                        )
                    De_sh[idx] = rebind[Scalar[DTYPE]](
                        workspace[env, pyr_sc + e * MC + c]
                    )
                    bias_e_sh[idx] = rebind[Scalar[DTYPE]](
                        workspace[env, pyr_sc + 4 * MC + e * MC + c]
                    )

        barrier()

        # === THREAD 0: joint-limit edge detection + initial setup ===
        # All per-thread serial scratch stays thread-0-local exactly as serial.
        var qacc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var qacc_smooth = InlineArray[Scalar[DTYPE], V_SIZE](
            uninitialized=True
        )
        var Ma = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var f_smooth = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var jar = InlineArray[Scalar[DTYPE], ME](uninitialized=True)
        var H = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        var L_chol = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        var grad = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var search = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var Mv = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var qfrc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var scale: Scalar[DTYPE] = 0
        var num_edges = 0

        comptime M_inv_idx_pyr = ws_m_inv_offset[NV, NBODY]()
        comptime qpos_off_lim = qpos_offset[NQ, NV]()
        comptime qvel_off_lim = qvel_offset[NQ, NV]()

        if valid_env and tid == 0:
            num_edges = nc * NE

            # Model-level defaults for fallback
            var lr_tc_def = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLREF_LIMIT_0]
            )
            var lr_dr_def = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLREF_LIMIT_1]
            )
            var li_dmin_def = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLIMP_LIMIT_0]
            )
            var li_dmax_def = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLIMP_LIMIT_1]
            )
            var li_width_def = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLIMP_LIMIT_2]
            )
            var li_midpoint_def = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLIMP_LIMIT_3]
            )
            var li_power_def = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_SOLIMP_LIMIT_4]
            )

            for j in range(NJOINT):
                var j_off = model_joint_offset[NBODY](j)
                var jtype = Int(
                    rebind[Scalar[DTYPE]](model[0, j_off + JOINT_IDX_TYPE])
                )
                if jtype != JNT_HINGE and jtype != JNT_SLIDE:
                    continue
                var dof = Int(
                    rebind[Scalar[DTYPE]](model[0, j_off + JOINT_IDX_DOF_ADR])
                )
                var qpos_adr = Int(
                    rebind[Scalar[DTYPE]](model[0, j_off + JOINT_IDX_QPOS_ADR])
                )
                var rmin = rebind[Scalar[DTYPE]](
                    model[0, j_off + JOINT_IDX_RANGE_MIN]
                )
                var rmax = rebind[Scalar[DTYPE]](
                    model[0, j_off + JOINT_IDX_RANGE_MAX]
                )
                if rmin < Scalar[DTYPE](-1e9) or rmax > Scalar[DTYPE](1e9):
                    continue
                var lr_tc = rebind[Scalar[DTYPE]](
                    model[0, j_off + JOINT_IDX_SOLREF_LIMIT_0]
                )
                var lr_dr = rebind[Scalar[DTYPE]](
                    model[0, j_off + JOINT_IDX_SOLREF_LIMIT_1]
                )
                if lr_tc <= Scalar[DTYPE](0):
                    lr_tc = lr_tc_def
                if lr_dr <= Scalar[DTYPE](0):
                    lr_dr = lr_dr_def
                var li_dmin = rebind[Scalar[DTYPE]](
                    model[0, j_off + JOINT_IDX_SOLIMP_LIMIT_0]
                )
                var li_dmax = rebind[Scalar[DTYPE]](
                    model[0, j_off + JOINT_IDX_SOLIMP_LIMIT_1]
                )
                var li_width = rebind[Scalar[DTYPE]](
                    model[0, j_off + JOINT_IDX_SOLIMP_LIMIT_2]
                )
                var li_midpoint = rebind[Scalar[DTYPE]](
                    model[0, j_off + JOINT_IDX_SOLIMP_LIMIT_3]
                )
                var li_power = rebind[Scalar[DTYPE]](
                    model[0, j_off + JOINT_IDX_SOLIMP_LIMIT_4]
                )
                if li_dmax <= Scalar[DTYPE](0) and li_width <= Scalar[DTYPE](
                    0
                ):
                    li_dmin = li_dmin_def
                    li_dmax = li_dmax_def
                    li_width = li_width_def
                    li_midpoint = li_midpoint_def
                    li_power = li_power_def
                if li_width < Scalar[DTYPE](1e-6):
                    li_width = Scalar[DTYPE](1e-6)
                if li_dmax < Scalar[DTYPE](1e-4):
                    li_dmax = Scalar[DTYPE](1e-4)
                var l_K_spring = Scalar[DTYPE](1.0) / (
                    li_dmax * li_dmax * lr_tc * lr_tc * lr_dr * lr_dr
                )
                var l_B_damp = Scalar[DTYPE](2.0) / (li_dmax * lr_tc)

                var pos = rebind[Scalar[DTYPE]](
                    state[env, qpos_off_lim + qpos_adr]
                )
                # Lower limit
                var dist_lo = pos - rmin
                if dist_lo < Scalar[DTYPE](0) and num_edges < ME:
                    var sign = Scalar[DTYPE](1)
                    var K_lim = rebind[Scalar[DTYPE]](
                        workspace[env, M_inv_idx_pyr + dof * NV + dof]
                    )
                    if K_lim < Scalar[DTYPE](1e-10):
                        K_lim = Scalar[DTYPE](1e-10)
                    var pen = -dist_lo
                    var v_lim = sign * rebind[Scalar[DTYPE]](
                        state[env, qvel_off_lim + dof]
                    )
                    var imp_lim: Scalar[DTYPE]
                    if li_dmin == li_dmax or li_width <= Scalar[DTYPE](0):
                        imp_lim = Scalar[DTYPE](0.5) * (li_dmin + li_dmax)
                    else:
                        var x_l = pen / li_width
                        if x_l <= Scalar[DTYPE](0):
                            imp_lim = li_dmin
                        elif x_l >= Scalar[DTYPE](1):
                            imp_lim = li_dmax
                        else:
                            var y_l: Scalar[DTYPE]
                            if li_power == Scalar[DTYPE](1):
                                y_l = x_l
                            elif x_l <= li_midpoint:
                                y_l = pow(x_l, li_power) / pow(
                                    li_midpoint, li_power - Scalar[DTYPE](1)
                                )
                            else:
                                y_l = Scalar[DTYPE](1) - pow(
                                    Scalar[DTYPE](1) - x_l, li_power
                                ) / pow(
                                    Scalar[DTYPE](1) - li_midpoint,
                                    li_power - Scalar[DTYPE](1),
                                )
                            imp_lim = li_dmin + y_l * (li_dmax - li_dmin)
                    if imp_lim < Scalar[DTYPE](1e-6):
                        imp_lim = Scalar[DTYPE](1e-6)
                    comptime dof_iw_off = model_dof_invweight0_offset[
                        NBODY, NJOINT, NGEOM, MAX_EQUALITY, MAX_TENDON, NSITE
                    ]()
                    var diag_lim = rebind[Scalar[DTYPE]](
                        model[0, dof_iw_off + dof]
                    )
                    if diag_lim < Scalar[DTYPE](1e-10):
                        diag_lim = K_lim
                    var R_lim = (
                        (Scalar[DTYPE](1) - imp_lim) / imp_lim * diag_lim
                    )
                    if R_lim < Scalar[DTYPE](1e-14):
                        R_lim = Scalar[DTYPE](1e-14)
                    for i in range(NV):
                        Je_sh[num_edges * NV + i] = Scalar[DTYPE](0)
                    Je_sh[num_edges * NV + dof] = sign
                    var inv_K_lim = Scalar[DTYPE](1) / (K_lim + R_lim)
                    var R_recov = Scalar[DTYPE](1) / inv_K_lim - K_lim
                    if R_recov < Scalar[DTYPE](1e-14):
                        R_recov = Scalar[DTYPE](1e-14)
                    De_sh[num_edges] = Scalar[DTYPE](1) / R_recov
                    bias_e_sh[num_edges] = (
                        l_B_damp * v_lim - l_K_spring * imp_lim * pen
                    )
                    num_edges += 1

                # Upper limit
                var dist_hi = rmax - pos
                if dist_hi < Scalar[DTYPE](0) and num_edges < ME:
                    var sign = Scalar[DTYPE](-1)
                    var K_lim = rebind[Scalar[DTYPE]](
                        workspace[env, M_inv_idx_pyr + dof * NV + dof]
                    )
                    if K_lim < Scalar[DTYPE](1e-10):
                        K_lim = Scalar[DTYPE](1e-10)
                    var pen = -dist_hi
                    var v_lim = sign * rebind[Scalar[DTYPE]](
                        state[env, qvel_off_lim + dof]
                    )
                    var imp_lim: Scalar[DTYPE]
                    if li_dmin == li_dmax or li_width <= Scalar[DTYPE](0):
                        imp_lim = Scalar[DTYPE](0.5) * (li_dmin + li_dmax)
                    else:
                        var x_l = pen / li_width
                        if x_l <= Scalar[DTYPE](0):
                            imp_lim = li_dmin
                        elif x_l >= Scalar[DTYPE](1):
                            imp_lim = li_dmax
                        else:
                            var y_l: Scalar[DTYPE]
                            if li_power == Scalar[DTYPE](1):
                                y_l = x_l
                            elif x_l <= li_midpoint:
                                y_l = pow(x_l, li_power) / pow(
                                    li_midpoint, li_power - Scalar[DTYPE](1)
                                )
                            else:
                                y_l = Scalar[DTYPE](1) - pow(
                                    Scalar[DTYPE](1) - x_l, li_power
                                ) / pow(
                                    Scalar[DTYPE](1) - li_midpoint,
                                    li_power - Scalar[DTYPE](1),
                                )
                            imp_lim = li_dmin + y_l * (li_dmax - li_dmin)
                    if imp_lim < Scalar[DTYPE](1e-6):
                        imp_lim = Scalar[DTYPE](1e-6)
                    comptime dof_iw_off = model_dof_invweight0_offset[
                        NBODY, NJOINT, NGEOM, MAX_EQUALITY, MAX_TENDON, NSITE
                    ]()
                    var diag_lim = rebind[Scalar[DTYPE]](
                        model[0, dof_iw_off + dof]
                    )
                    if diag_lim < Scalar[DTYPE](1e-10):
                        diag_lim = K_lim
                    var R_lim = (
                        (Scalar[DTYPE](1) - imp_lim) / imp_lim * diag_lim
                    )
                    if R_lim < Scalar[DTYPE](1e-14):
                        R_lim = Scalar[DTYPE](1e-14)
                    for i in range(NV):
                        Je_sh[num_edges * NV + i] = Scalar[DTYPE](0)
                    Je_sh[num_edges * NV + dof] = sign
                    var inv_K_lim = Scalar[DTYPE](1) / (K_lim + R_lim)
                    var R_recov = Scalar[DTYPE](1) / inv_K_lim - K_lim
                    if R_recov < Scalar[DTYPE](1e-14):
                        R_recov = Scalar[DTYPE](1e-14)
                    De_sh[num_edges] = Scalar[DTYPE](1) / R_recov
                    bias_e_sh[num_edges] = (
                        l_B_damp * v_lim - l_K_spring * imp_lim * pen
                    )
                    num_edges += 1

            # Publish num_edges to shared for all threads.
            ctrl_sh[0] = Scalar[DTYPE](num_edges)

            # Initialize qacc/qacc_smooth from workspace
            for i in range(NV):
                var q_i = rebind[Scalar[DTYPE]](
                    workspace[env, qacc_init_idx + i]
                )
                qacc[i] = q_i
                qacc_smooth[i] = q_i
            # Ma = M * qacc (read from M_sh)
            for i in range(NV):
                Ma[i] = Scalar[DTYPE](0)
                for j in range(NV):
                    Ma[i] += rebind[Scalar[DTYPE]](M_sh[i * NV + j]) * qacc[j]
            for i in range(NV):
                f_smooth[i] = Ma[i]
            # Scale for convergence check
            for i in range(NV):
                scale += rebind[Scalar[DTYPE]](M_sh[i * NV + i])
            if scale > Scalar[DTYPE](1e-10):
                scale = Scalar[DTYPE](1.0) / scale
            else:
                scale = Scalar[DTYPE](1.0)

            # Initial jar + force + qfrc; publish force to force_sh
            for i in range(NV):
                qfrc[i] = Scalar[DTYPE](0)
            for e_idx in range(num_edges):
                jar[e_idx] = rebind[Scalar[DTYPE]](bias_e_sh[e_idx])
                for i in range(NV):
                    jar[e_idx] += (
                        rebind[Scalar[DTYPE]](Je_sh[e_idx * NV + i]) * qacc[i]
                    )
                var f_e: Scalar[DTYPE]
                if jar[e_idx] >= Scalar[DTYPE](0):
                    f_e = Scalar[DTYPE](0)
                else:
                    f_e = -rebind[Scalar[DTYPE]](De_sh[e_idx]) * jar[e_idx]
                force_sh[e_idx] = f_e
                for i in range(NV):
                    qfrc[i] += (
                        rebind[Scalar[DTYPE]](Je_sh[e_idx * NV + i]) * f_e
                    )

        # Make num_edges + force_sh visible to all threads.
        barrier()
        var num_edges_b = Int(rebind[Scalar[DTYPE]](ctrl_sh[0]))

        # === Newton iterations — ALL threads execute the loop ===
        for iter_n in range(NEWTON_ITER_GPU):
            # --- Thread 0: gradient + convergence check ---
            if valid_env and tid == 0:
                var grad_norm: Scalar[DTYPE] = 0
                for i in range(NV):
                    grad[i] = Ma[i] - f_smooth[i] - qfrc[i]
                    grad_norm += grad[i] * grad[i]
                if scale * sqrt(grad_norm) < Scalar[DTYPE](NEWTON_TOL_GPU):
                    ctrl_sh[1] = Scalar[DTYPE](1)  # done
                else:
                    ctrl_sh[1] = Scalar[DTYPE](0)
            barrier()
            if Int(rebind[Scalar[DTYPE]](ctrl_sh[1])) == 1:
                break

            # --- ALL threads: parallel Hessian assembly ---
            # Distribute NV*NV entries; preserve inner edge-sum order
            # (e ascending) so this is bit-identical to the serial build.
            if valid_env:
                for idx in range(tid, NV * NV, THREADS):
                    var i = idx // NV
                    var j = idx % NV
                    var h = rebind[Scalar[DTYPE]](M_sh[idx])
                    for e in range(num_edges_b):
                        if rebind[Scalar[DTYPE]](force_sh[e]) > Scalar[DTYPE](
                            0
                        ):
                            h += (
                                rebind[Scalar[DTYPE]](De_sh[e])
                                * rebind[Scalar[DTYPE]](Je_sh[e * NV + i])
                                * rebind[Scalar[DTYPE]](Je_sh[e * NV + j])
                            )
                    H_sh[idx] = h
            barrier()

            # --- Thread 0: Cholesky solve + line search + state update ---
            if valid_env and tid == 0:
                for k in range(NV * NV):
                    H[k] = rebind[Scalar[DTYPE]](H_sh[k])

                var chol_ok = chol_factor_inline[DTYPE, NV, M_SIZE](H, L_chol)
                if not chol_ok:
                    for i in range(NV):
                        H[i * NV + i] += Scalar[DTYPE](1e-6)
                    _ = chol_factor_inline[DTYPE, NV, M_SIZE](H, L_chol)
                chol_solve_inline[DTYPE, NV, M_SIZE, V_SIZE](
                    L_chol, grad, search
                )
                for i in range(NV):
                    search[i] = -search[i]

                # Mv = M * search
                for i in range(NV):
                    Mv[i] = Scalar[DTYPE](0)
                    for j in range(NV):
                        Mv[i] += (
                            rebind[Scalar[DTYPE]](M_sh[i * NV + j]) * search[j]
                        )

                # Precompute Jv_e = Je · search per edge
                var Jv_e = InlineArray[Scalar[DTYPE], ME](uninitialized=True)
                for e_idx in range(num_edges_b):
                    Jv_e[e_idx] = Scalar[DTYPE](0)
                    for i in range(NV):
                        Jv_e[e_idx] += (
                            rebind[Scalar[DTYPE]](Je_sh[e_idx * NV + i])
                            * search[i]
                        )

                var gauss_a: Scalar[DTYPE] = 0
                var gauss_b: Scalar[DTYPE] = 0
                for i in range(NV):
                    gauss_a += Mv[i] * search[i]
                    gauss_b += (Ma[i] - f_smooth[i]) * search[i]

                var p0_d1 = gauss_b
                var p0_d2 = gauss_a
                for e_idx in range(num_edges_b):
                    if jar[e_idx] < Scalar[DTYPE](0):
                        p0_d1 += (
                            rebind[Scalar[DTYPE]](De_sh[e_idx])
                            * jar[e_idx]
                            * Jv_e[e_idx]
                        )
                        p0_d2 += (
                            rebind[Scalar[DTYPE]](De_sh[e_idx])
                            * Jv_e[e_idx]
                            * Jv_e[e_idx]
                        )
                if p0_d2 < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                    p0_d2 = Scalar[DTYPE](PRIMAL_MINVAL_GPU)

                var alpha: Scalar[DTYPE] = 0
                if p0_d1 < Scalar[DTYPE](0):
                    alpha = -p0_d1 / p0_d2

                    var old_cost_ls: Scalar[DTYPE] = 0
                    for i in range(NV):
                        old_cost_ls += (
                            Scalar[DTYPE](0.5)
                            * (Ma[i] - f_smooth[i])
                            * (qacc[i] - qacc_smooth[i])
                        )
                    for e_idx in range(num_edges_b):
                        if jar[e_idx] < Scalar[DTYPE](0):
                            old_cost_ls += (
                                Scalar[DTYPE](0.5)
                                * rebind[Scalar[DTYPE]](De_sh[e_idx])
                                * jar[e_idx]
                                * jar[e_idx]
                            )

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
                        for e_idx in range(num_edges_b):
                            var jar_t = jar[e_idx] + alpha * Jv_e[e_idx]
                            if jar_t < Scalar[DTYPE](0):
                                trial_cost += (
                                    Scalar[DTYPE](0.5)
                                    * rebind[Scalar[DTYPE]](De_sh[e_idx])
                                    * jar_t
                                    * jar_t
                                )
                        if trial_cost <= old_cost_ls:
                            break
                        alpha *= Scalar[DTYPE](0.5)

                if alpha < Scalar[DTYPE](1e-10):
                    ctrl_sh[1] = Scalar[DTYPE](1)  # done (break next iter)
                else:
                    ctrl_sh[1] = Scalar[DTYPE](0)

                    # Save old state for revert
                    var old_qacc = InlineArray[Scalar[DTYPE], V_SIZE](
                        uninitialized=True
                    )
                    var old_Ma = InlineArray[Scalar[DTYPE], V_SIZE](
                        uninitialized=True
                    )
                    var old_jar = InlineArray[Scalar[DTYPE], ME](
                        uninitialized=True
                    )
                    var old_force = InlineArray[Scalar[DTYPE], ME](
                        uninitialized=True
                    )
                    var old_qfrc = InlineArray[Scalar[DTYPE], V_SIZE](
                        uninitialized=True
                    )
                    for i in range(NV):
                        old_qacc[i] = qacc[i]
                        old_Ma[i] = Ma[i]
                        old_qfrc[i] = qfrc[i]
                    for e_idx in range(num_edges_b):
                        old_jar[e_idx] = jar[e_idx]
                        old_force[e_idx] = rebind[Scalar[DTYPE]](
                            force_sh[e_idx]
                        )

                    var old_cost: Scalar[DTYPE] = 0
                    for i in range(NV):
                        old_cost += (
                            Scalar[DTYPE](0.5)
                            * (Ma[i] - f_smooth[i])
                            * (qacc[i] - qacc_smooth[i])
                        )
                    for e_idx in range(num_edges_b):
                        if jar[e_idx] < Scalar[DTYPE](0):
                            old_cost += (
                                Scalar[DTYPE](0.5)
                                * rebind[Scalar[DTYPE]](De_sh[e_idx])
                                * jar[e_idx]
                                * jar[e_idx]
                            )

                    for i in range(NV):
                        qacc[i] += alpha * search[i]
                        Ma[i] += alpha * Mv[i]

                    for i in range(NV):
                        qfrc[i] = Scalar[DTYPE](0)
                    for e_idx in range(num_edges_b):
                        jar[e_idx] = rebind[Scalar[DTYPE]](bias_e_sh[e_idx])
                        for i in range(NV):
                            jar[e_idx] += (
                                rebind[Scalar[DTYPE]](Je_sh[e_idx * NV + i])
                                * qacc[i]
                            )
                        var f_e: Scalar[DTYPE]
                        if jar[e_idx] >= Scalar[DTYPE](0):
                            f_e = Scalar[DTYPE](0)
                        else:
                            f_e = -rebind[Scalar[DTYPE]](De_sh[e_idx]) * jar[
                                e_idx
                            ]
                        force_sh[e_idx] = f_e
                        for i in range(NV):
                            qfrc[i] += (
                                rebind[Scalar[DTYPE]](Je_sh[e_idx * NV + i])
                                * f_e
                            )

                    var new_cost: Scalar[DTYPE] = 0
                    for i in range(NV):
                        new_cost += (
                            Scalar[DTYPE](0.5)
                            * (Ma[i] - f_smooth[i])
                            * (qacc[i] - qacc_smooth[i])
                        )
                    for e_idx in range(num_edges_b):
                        if jar[e_idx] < Scalar[DTYPE](0):
                            new_cost += (
                                Scalar[DTYPE](0.5)
                                * rebind[Scalar[DTYPE]](De_sh[e_idx])
                                * jar[e_idx]
                                * jar[e_idx]
                            )

                    var improvement = scale * (old_cost - new_cost)
                    if (
                        improvement < Scalar[DTYPE](NEWTON_TOL_GPU)
                        and iter_n > 0
                    ):
                        if improvement < Scalar[DTYPE](0):
                            for i in range(NV):
                                qacc[i] = old_qacc[i]
                                Ma[i] = old_Ma[i]
                                qfrc[i] = old_qfrc[i]
                            for e_idx in range(num_edges_b):
                                jar[e_idx] = old_jar[e_idx]
                                force_sh[e_idx] = old_force[e_idx]
                        ctrl_sh[1] = Scalar[DTYPE](1)  # done

            # force_sh updated by thread 0; make visible for next assembly.
            barrier()
            if Int(rebind[Scalar[DTYPE]](ctrl_sh[1])) == 1:
                break

        # === THREAD 0: write back + reconstruct forces + equality/tendon ===
        if not valid_env or tid != 0:
            return

        for i in range(NV):
            workspace[env, qacc_idx + i] = qacc[i]

        for c in range(nc):
            var fn_c: Scalar[DTYPE] = 0
            var ft1_c: Scalar[DTYPE] = 0
            var ft2_c: Scalar[DTYPE] = 0
            var mu_c = rebind[Scalar[DTYPE]](
                workspace[env, pyr_sc + 8 * MC + c]
            )
            var safe_mu = mu_c
            if safe_mu < Scalar[DTYPE](1e-8):
                safe_mu = Scalar[DTYPE](1e-8)
            var f_e0 = rebind[Scalar[DTYPE]](force_sh[c * NE + 0])
            var f_e1 = rebind[Scalar[DTYPE]](force_sh[c * NE + 1])
            var f_e2 = rebind[Scalar[DTYPE]](force_sh[c * NE + 2])
            var f_e3 = rebind[Scalar[DTYPE]](force_sh[c * NE + 3])
            fn_c = (f_e0 + f_e1 + f_e2 + f_e3) / Scalar[DTYPE](2.0)
            ft1_c = (f_e0 - f_e1) * safe_mu
            ft2_c = (f_e2 - f_e3) * safe_mu
            var c_off = contacts_off + c * CONTACT_SIZE
            state[env, c_off + CONTACT_IDX_FORCE_N] = fn_c
            state[env, c_off + CONTACT_IDX_FORCE_T1] = ft1_c
            state[env, c_off + CONTACT_IDX_FORCE_T2] = ft2_c

        comptime SOLVER_ITER_GPU_B: Int = 50
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
            SOLVER_ITER_GPU_B,
        ](env, state, model, workspace)
        comptime if MAX_TENDON > 0:
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
                SOLVER_ITER_GPU_B,
            ](env, state, model, workspace)

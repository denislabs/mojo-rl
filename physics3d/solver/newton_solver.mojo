"""Newton constraint solver for Generalized Coordinates engine.

Implements MuJoCo-style projected Newton method for contact solving:
1. Form the Delassus matrix: A[c1,c2] = J[c1] * M^-1 * J[c2]^T
2. Minimize 0.5*lambda^T*A*lambda + b^T*lambda subject to lambda >= 0
3. Use Newton steps with active-set identification and line search

The Newton solver has quadratic convergence rate for the active set
and is the most accurate solver for stiff contact problems. It is
more expensive per iteration than PGS or CG but converges in fewer steps.

Friction is solved with PGS iterations (same as MuJoCo's approach).

Reference: MuJoCo Technical Notes, Section on Newton solver.
"""

from math import sqrt
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim, barrier
from ..types import Model, Data, _max_one
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..traits.solver import ConstraintSolver
from ..dynamics.jacobian import compute_contact_jacobian_row
from ..constraints.constraint_data import (
    ConstraintData,
    CNSTR_NORMAL,
    CNSTR_FRICTION_T1,
    CNSTR_FRICTION_T2,
    CNSTR_LIMIT,
)

# Import shared friction solver (GPU only now — CPU friction uses ConstraintData)
from .friction_solver import _solve_friction_pgs_gpu

from ..gpu.constants import (
    contacts_offset,
    metadata_offset,
    model_metadata_offset,
    ws_qacc_constrained_offset,
    ws_m_inv_offset,
    ws_solver_offset,
    CONTACT_SIZE,
    CONTACT_IDX_FORCE_N,
    META_IDX_NUM_CONTACTS,
    MODEL_META_IDX_FRICTION,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_SOLREF_CONTACT_0,
    MODEL_META_IDX_SOLREF_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_0,
    MODEL_META_IDX_SOLIMP_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_2,
)

from ..constraints.constraint_builder_gpu import (
    init_common_normal_workspace_gpu,
    precompute_contact_normal_gpu,
    warmstart_normals_gpu,
    apply_solved_normals_gpu,
    detect_and_solve_limits_gpu,
)


# Newton solver parameters
comptime NEWTON_ITERATIONS: Int = 15
comptime NEWTON_TOLERANCE: Float64 = 1e-8
comptime LINESEARCH_ITERATIONS: Int = 10
comptime LINESEARCH_BETA: Float64 = 0.5  # Step shrink factor
comptime LINESEARCH_ARMIJO: Float64 = 1e-4  # Armijo sufficient decrease
# Coupled PGS iterations (normals + friction + limits together, MuJoCo-style)
comptime COUPLED_PGS_ITERATIONS: Int = 50
# Debug flag — set to True to print Newton QP convergence info
comptime NEWTON_DEBUG: Bool = False
# Minimum K for friction tangent rows — below this, direction is degenerate
comptime FRICTION_K_MIN: Float64 = 1e-6


struct NewtonSolver(ConstraintSolver):
    """Projected Newton constraint solver for GC engine.

    Solves the normal constraint QP using Newton's method with
    active-set identification and Armijo line search.

    Advantages over PGS/CG:
    - Quadratic convergence for well-conditioned problems
    - Most accurate for stiff contacts
    - Reliable convergence with line search

    Disadvantages:
    - Most expensive per iteration (solves linear system)
    - Requires forming and solving the reduced Hessian
    """

    @staticmethod
    fn solver_workspace_size[NV: Int, MAX_CONTACTS: Int]() -> Int:
        """Newton solver workspace: 28*MC + 6*MC*NV + MC*MC floats.

        Layout (offsets relative to solver workspace start):
          [0..13*MC+2*MC*NV)                            Common normal block
          [13*MC+2*MC*NV..14*MC+2*MC*NV)                rhs
          [14*MC+2*MC*NV..14*MC+2*MC*NV+MC*MC)          A (Delassus matrix)
          [14*MC+2*MC*NV+MC*MC..15*MC+2*MC*NV+MC*MC)    grad
          [15*MC+2*MC*NV+MC*MC..16*MC+2*MC*NV+MC*MC)    d (Newton direction)
          [16*MC+2*MC*NV+MC*MC..17*MC+2*MC*NV+MC*MC)    lambda_trial
          [17*MC+2*MC*NV+MC*MC..18*MC+2*MC*NV+MC*MC)    free_map (Float)
          [18*MC+2*MC*NV+MC*MC..28*MC+6*MC*NV+MC*MC)    Friction (10*MC + 4*MC*NV)
        """
        comptime MC = _max_one[MAX_CONTACTS]()
        return 28 * MC + 6 * MC * NV + MC * MC

    @staticmethod
    fn solve[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        MAX_ROWS: Int,
        V_SIZE: Int,
        M_SIZE: Int,
        NGEOM: Int = 0,
    ](
        model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        M_inv: InlineArray[Scalar[DTYPE], M_SIZE],
        mut constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
        mut qacc: InlineArray[Scalar[DTYPE], V_SIZE],
        dt: Scalar[DTYPE],
    ):
        """Solve constraints using Projected Newton on CPU (acceleration-level).

        Iterates over pre-built ConstraintData:
        1. Build Delassus matrix from normal constraint rows
        2. Projected Newton with Armijo line search for normals
        3. PGS for joint limit constraints
        4. PGS for friction (with Coulomb cone clamping)
        """
        if constraints.num_rows == 0:
            return

        var num_normals = constraints.num_normals
        var num_friction = constraints.num_friction
        var num_limits = constraints.num_limits
        var friction_start = num_normals
        var limits_start = num_normals + num_friction

        comptime MR = _max_one[MAX_ROWS]()
        comptime A_SIZE = _max_one[MAX_ROWS * MAX_ROWS]()

        # RHS for Newton: rhs[r] = J[r] · qacc_unconstrained + bias[r]
        # IMPORTANT: Compute ALL rhs from unconstrained qacc BEFORE warm-start
        var rhs = InlineArray[Scalar[DTYPE], MR](uninitialized=True)
        for i in range(MR):
            rhs[i] = Scalar[DTYPE](0)

        for r in range(num_normals):
            var a_n: Scalar[DTYPE] = 0
            for i in range(NV):
                a_n += constraints.J[r * NV + i] * qacc[i]
            rhs[r] = a_n + constraints.rows[r].bias

        # Apply warm-start (after rhs is fully computed)
        for r in range(num_normals):
            if constraints.rows[r].lambda_val > Scalar[DTYPE](0):
                for i in range(NV):
                    qacc[i] += (
                        constraints.MinvJT[r * NV + i]
                        * constraints.rows[r].lambda_val
                    )

        # Build Delassus matrix A[c1,c2] = J[c1] . MinvJT[c2]
        # Then add regularizer R to diagonal: AR[c,c] = K + R where R = K/imp - K
        var A = InlineArray[Scalar[DTYPE], A_SIZE](uninitialized=True)
        for i in range(A_SIZE):
            A[i] = Scalar[DTYPE](0)
        for c1 in range(num_normals):
            for c2 in range(num_normals):
                var a_val: Scalar[DTYPE] = 0
                for i in range(NV):
                    a_val += (
                        constraints.J[c1 * NV + i]
                        * constraints.MinvJT[c2 * NV + i]
                    )
                A[c1 * num_normals + c2] = a_val

        # Add MuJoCo regularizer R to diagonal: AR[c,c] = K/imp
        for c in range(num_normals):
            var R = Scalar[DTYPE](1.0) / constraints.rows[c].inv_K_imp - constraints.rows[c].K
            A[c * num_normals + c] += R

        # =====================================================================
        # Phase 2: Projected Newton for normal constraints
        # Minimize: f(x) = 0.5 * x^T * A * x + rhs^T * x subject to x >= 0
        # =====================================================================
        var grad = InlineArray[Scalar[DTYPE], MR](uninitialized=True)
        var d = InlineArray[Scalar[DTYPE], MR](uninitialized=True)
        var lambda_trial = InlineArray[Scalar[DTYPE], MR](uninitialized=True)

        for i in range(MR):
            grad[i] = Scalar[DTYPE](0)
            d[i] = Scalar[DTYPE](0)
            lambda_trial[i] = Scalar[DTYPE](0)

        for _ in range(NEWTON_ITERATIONS):
            # Compute gradient: g = A * lambda + rhs
            for c in range(num_normals):
                var g: Scalar[DTYPE] = rhs[c]
                for c2 in range(num_normals):
                    g += (
                        A[c * num_normals + c2]
                        * constraints.rows[c2].lambda_val
                    )
                grad[c] = g

            # Projected gradient norm
            var grad_norm: Scalar[DTYPE] = 0
            for c in range(num_normals):
                if constraints.rows[c].lambda_val > Scalar[DTYPE](0) or grad[
                    c
                ] < Scalar[DTYPE](0):
                    grad_norm += grad[c] * grad[c]

            if grad_norm < Scalar[DTYPE](NEWTON_TOLERANCE):
                break

            # Identify free set
            var free_count = 0
            var free_map = InlineArray[Int, MR](uninitialized=True)
            for i in range(MR):
                free_map[i] = -1

            for c in range(num_normals):
                if constraints.rows[c].lambda_val > Scalar[DTYPE](0) or grad[
                    c
                ] < Scalar[DTYPE](0):
                    free_map[c] = free_count
                    free_count += 1

            if free_count == 0:
                break

            # Jacobi initial guess + Gauss-Seidel refinement
            for c in range(num_normals):
                d[c] = Scalar[DTYPE](0)

            for c in range(num_normals):
                if free_map[c] < 0:
                    continue
                if A[c * num_normals + c] > Scalar[DTYPE](1e-14):
                    d[c] = -grad[c] / A[c * num_normals + c]

            for _ in range(5):
                for c in range(num_normals):
                    if free_map[c] < 0:
                        continue
                    var sum_off_diag: Scalar[DTYPE] = 0
                    for c2 in range(num_normals):
                        if c2 == c:
                            continue
                        if free_map[c2] < 0:
                            continue
                        sum_off_diag += A[c * num_normals + c2] * d[c2]
                    d[c] = (-grad[c] - sum_off_diag) / A[c * num_normals + c]

            # Line search with Armijo condition
            var f_current: Scalar[DTYPE] = 0
            for c in range(num_normals):
                f_current += rhs[c] * constraints.rows[c].lambda_val
                for c2 in range(num_normals):
                    f_current += (
                        Scalar[DTYPE](0.5)
                        * constraints.rows[c].lambda_val
                        * A[c * num_normals + c2]
                        * constraints.rows[c2].lambda_val
                    )

            var gtd: Scalar[DTYPE] = 0
            for c in range(num_normals):
                if free_map[c] < 0:
                    continue
                gtd += grad[c] * d[c]

            var step = Scalar[DTYPE](1.0)
            var armijo = Scalar[DTYPE](LINESEARCH_ARMIJO)
            var beta = Scalar[DTYPE](LINESEARCH_BETA)

            for _ in range(LINESEARCH_ITERATIONS):
                for c in range(num_normals):
                    lambda_trial[c] = constraints.rows[c].lambda_val
                    if free_map[c] >= 0:
                        lambda_trial[c] = (
                            constraints.rows[c].lambda_val + step * d[c]
                        )
                    if lambda_trial[c] < Scalar[DTYPE](0):
                        lambda_trial[c] = Scalar[DTYPE](0)

                var f_trial: Scalar[DTYPE] = 0
                for c in range(num_normals):
                    f_trial += rhs[c] * lambda_trial[c]
                    for c2 in range(num_normals):
                        f_trial += (
                            Scalar[DTYPE](0.5)
                            * lambda_trial[c]
                            * A[c * num_normals + c2]
                            * lambda_trial[c2]
                        )

                if f_trial <= f_current + armijo * step * gtd:
                    break

                step = step * beta

            for c in range(num_normals):
                constraints.rows[c].lambda_val = lambda_trial[c]

        # DEBUG: Print Newton QP convergence (before friction/limits modify qacc)
        @parameter
        if NEWTON_DEBUG:
            # Recompute final gradient
            var final_grad_norm: Scalar[DTYPE] = 0
            for c in range(num_normals):
                var g: Scalar[DTYPE] = rhs[c]
                for c2 in range(num_normals):
                    g += A[c * num_normals + c2] * constraints.rows[c2].lambda_val
                if constraints.rows[c].lambda_val > Scalar[DTYPE](0) or g < Scalar[DTYPE](0):
                    final_grad_norm += g * g
                print(
                    "    [NEWTON] row",
                    c,
                    ": lambda=",
                    Float64(constraints.rows[c].lambda_val),
                    " QP_grad=",
                    Float64(g),
                )
            print("    [NEWTON] final projected_grad_norm=", Float64(final_grad_norm))

        # Apply solved forces: remove warm-start, apply final
        for c in range(num_normals):
            var warm = data.contacts[
                constraints.rows[c].source_contact_idx
            ].force_n
            if warm > Scalar[DTYPE](0):
                for i in range(NV):
                    qacc[i] -= constraints.MinvJT[c * NV + i] * warm

        for c in range(num_normals):
            if constraints.rows[c].lambda_val > Scalar[DTYPE](0):
                for i in range(NV):
                    qacc[i] += (
                        constraints.MinvJT[c * NV + i]
                        * constraints.rows[c].lambda_val
                    )

        # =====================================================================
        # Phase 3: Coupled PGS (normals + friction + limits together)
        # MuJoCo-style: iterate over ALL constraints in each pass so that
        # normal and friction forces naturally couple.
        # =====================================================================
        if num_friction == 0 and num_limits == 0:
            return

        # Apply friction warm-start (skip degenerate tangent rows)
        for r_off in range(num_friction):
            var r = friction_start + r_off
            if constraints.rows[r].K < Scalar[DTYPE](FRICTION_K_MIN):
                constraints.rows[r].lambda_val = Scalar[DTYPE](0)
                continue
            if constraints.rows[r].lambda_val != Scalar[DTYPE](0):
                for i in range(NV):
                    qacc[i] += (
                        constraints.MinvJT[r * NV + i]
                        * constraints.rows[r].lambda_val
                    )

        # Coupled PGS iterations
        for _ in range(COUPLED_PGS_ITERATIONS):
            # --- Normal constraints (PGS update, clamp >= 0) ---
            for r in range(num_normals):
                var a_n: Scalar[DTYPE] = 0
                for i in range(NV):
                    a_n += constraints.J[r * NV + i] * qacc[i]
                var R_n = Scalar[DTYPE](1.0) / constraints.rows[r].inv_K_imp - constraints.rows[r].K
                var residual = a_n + constraints.rows[r].bias + R_n * constraints.rows[r].lambda_val
                var delta = -residual * constraints.rows[r].inv_K_imp
                var old_lambda = constraints.rows[r].lambda_val
                constraints.rows[r].lambda_val = constraints.rows[r].lambda_val + delta
                if constraints.rows[r].lambda_val < Scalar[DTYPE](0):
                    constraints.rows[r].lambda_val = Scalar[DTYPE](0)
                var actual = constraints.rows[r].lambda_val - old_lambda
                for i in range(NV):
                    qacc[i] += constraints.MinvJT[r * NV + i] * actual

            # --- Friction constraints (paired t1/t2 with Coulomb cone) ---
            var pair_idx = 0
            while pair_idx < num_friction:
                var r_t1 = friction_start + pair_idx
                var r_t2 = friction_start + pair_idx + 1
                var normal_row = constraints.rows[r_t1].friction_parent
                var mu = constraints.rows[r_t1].friction_coef
                var lambda_n = constraints.rows[normal_row].lambda_val

                if lambda_n <= Scalar[DTYPE](0):
                    pair_idx += 2
                    continue

                var max_friction = mu * lambda_n

                # Tangent 1
                var old_t1 = constraints.rows[r_t1].lambda_val
                if constraints.rows[r_t1].K >= Scalar[DTYPE](FRICTION_K_MIN):
                    var a_t1: Scalar[DTYPE] = 0
                    for i in range(NV):
                        a_t1 += constraints.J[r_t1 * NV + i] * qacc[i]
                    constraints.rows[r_t1].lambda_val = old_t1 - a_t1 / constraints.rows[r_t1].K

                # Tangent 2
                var old_t2 = constraints.rows[r_t2].lambda_val
                if constraints.rows[r_t2].K >= Scalar[DTYPE](FRICTION_K_MIN):
                    var a_t2: Scalar[DTYPE] = 0
                    for i in range(NV):
                        a_t2 += constraints.J[r_t2 * NV + i] * qacc[i]
                    constraints.rows[r_t2].lambda_val = old_t2 - a_t2 / constraints.rows[r_t2].K

                # Coulomb cone clamping
                var t_mag = sqrt(
                    constraints.rows[r_t1].lambda_val
                    * constraints.rows[r_t1].lambda_val
                    + constraints.rows[r_t2].lambda_val
                    * constraints.rows[r_t2].lambda_val
                )
                if t_mag > max_friction:
                    var scale = max_friction / t_mag
                    constraints.rows[r_t1].lambda_val = constraints.rows[r_t1].lambda_val * scale
                    constraints.rows[r_t2].lambda_val = constraints.rows[r_t2].lambda_val * scale

                var actual_t1 = constraints.rows[r_t1].lambda_val - old_t1
                var actual_t2 = constraints.rows[r_t2].lambda_val - old_t2

                for i in range(NV):
                    qacc[i] += (
                        constraints.MinvJT[r_t1 * NV + i] * actual_t1
                        + constraints.MinvJT[r_t2 * NV + i] * actual_t2
                    )

                pair_idx += 2

            # --- Joint limit constraints ---
            for r_off in range(num_limits):
                var r = limits_start + r_off
                var dof = constraints.rows[r].source_dof
                var sign = constraints.rows[r].limit_sign
                var a_limit = sign * qacc[dof]
                var R_lim = Scalar[DTYPE](1.0) / constraints.rows[r].inv_K_imp - constraints.rows[r].K
                var residual = a_limit + constraints.rows[r].bias + R_lim * constraints.rows[r].lambda_val
                var delta = -residual * constraints.rows[r].inv_K_imp
                var old_lambda = constraints.rows[r].lambda_val
                constraints.rows[r].lambda_val = constraints.rows[r].lambda_val + delta
                if constraints.rows[r].lambda_val < Scalar[DTYPE](0):
                    constraints.rows[r].lambda_val = Scalar[DTYPE](0)
                var actual = constraints.rows[r].lambda_val - old_lambda
                for i in range(NV):
                    qacc[i] += constraints.MinvJT[r * NV + i] * actual

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
        """Solve contact constraints using Projected Newton on GPU.

        Uses thread_x for environment index, thread_y for contact index.
        Phase 1 and Delassus build are parallelized across contacts.
        Newton iterations are sequential on thread_y==0.
        All threads must hit all barriers (no early returns between them).
        """

        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        var contact_tid = Int(thread_idx.y)
        var valid_env = env < BATCH

        comptime qacc_idx = ws_qacc_constrained_offset[NV, NBODY]()
        comptime solver_ws_idx = ws_solver_offset[NV, NBODY]()
        comptime MC = _max_one[MAX_CONTACTS]()

        # Common normal block offsets
        comptime ws_lambda_n_idx = solver_ws_idx + 0 * MC
        comptime ws_K_n_idx = solver_ws_idx + 1 * MC
        comptime ws_c_dist_idx = solver_ws_idx + 2 * MC
        comptime ws_J_n_idx = solver_ws_idx + 13 * MC
        comptime ws_MinvJn_idx = solver_ws_idx + 13 * MC + MC * NV

        # Newton-specific offsets (after common normal block)
        comptime NW_START = solver_ws_idx + 13 * MC + 2 * MC * NV
        comptime ws_rhs_idx = NW_START + 0 * MC
        comptime ws_A_idx = NW_START + 1 * MC
        comptime ws_grad_idx = NW_START + MC + MC * MC
        comptime ws_d_idx = NW_START + 2 * MC + MC * MC
        comptime ws_ltrial_idx = NW_START + 3 * MC + MC * MC
        comptime ws_fmap_idx = NW_START + 4 * MC + MC * MC

        # === PARALLEL: Initialize workspace ===
        if valid_env:
            init_common_normal_workspace_gpu[
                DTYPE,
                NV,
                NBODY,
                MAX_CONTACTS,
                WS_SIZE,
                BATCH,
            ](env, contact_tid, workspace)
            # Init Newton-specific
            workspace[env, ws_rhs_idx + contact_tid] = 0
            workspace[env, ws_grad_idx + contact_tid] = 0
            workspace[env, ws_d_idx + contact_tid] = 0
            workspace[env, ws_ltrial_idx + contact_tid] = 0
            workspace[env, ws_fmap_idx + contact_tid] = -1
            for c2 in range(MC):
                workspace[env, ws_A_idx + contact_tid * MAX_CONTACTS + c2] = 0

        # Read metadata
        comptime contacts_off = contacts_offset[NQ, NV, NBODY]()
        comptime meta_off = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
        comptime model_meta_off = model_metadata_offset[NBODY, NJOINT]()

        var nc = 0
        var dt: Scalar[DTYPE] = 0
        var friction_coef: Scalar[DTYPE] = 0
        var K_spring: Scalar[DTYPE] = 0
        var B_damp: Scalar[DTYPE] = 0
        var si_dmin: Scalar[DTYPE] = 0
        var si_dmax: Scalar[DTYPE] = 0
        var si_width: Scalar[DTYPE] = 1

        if valid_env:
            dt = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_TIMESTEP]
            )
            nc = Int(
                rebind[Scalar[DTYPE]](
                    state[env, meta_off + META_IDX_NUM_CONTACTS]
                )
            )
            friction_coef = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_FRICTION]
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
            if si_width < Scalar[DTYPE](1e-6):
                si_width = Scalar[DTYPE](1e-6)
            if si_dmax < Scalar[DTYPE](1e-4):
                si_dmax = Scalar[DTYPE](1e-4)
            K_spring = Scalar[DTYPE](1.0) / (si_dmax * si_dmax * sr_tc * sr_tc * sr_dr * sr_dr)
            B_damp = Scalar[DTYPE](2.0) / (si_dmax * sr_tc)

        # === PARALLEL PHASE 1: Each thread precomputes one contact ===
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
                COMPUTE_RHS=True,
                RHS_IDX=ws_rhs_idx,
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
            )

        barrier()

        # === PARALLEL DELASSUS BUILD: Each thread computes one row of A ===
        # Add regularizer R to diagonal: AR[c,c] = K + R = K/imp
        if valid_env and contact_tid < nc:
            if workspace[env, ws_c_dist_idx + contact_tid] < Scalar[DTYPE](0):
                for c2 in range(nc):
                    if workspace[env, ws_c_dist_idx + c2] >= Scalar[DTYPE](0):
                        continue
                    var a_val: workspace.element_type = 0
                    for i in range(NV):
                        a_val += (
                            workspace[env, ws_J_n_idx + contact_tid * NV + i]
                            * workspace[env, ws_MinvJn_idx + c2 * NV + i]
                        )
                    workspace[
                        env, ws_A_idx + contact_tid * MAX_CONTACTS + c2
                    ] = a_val
                # Add MuJoCo regularizer R to diagonal
                comptime ws_inv_K_imp_idx = solver_ws_idx + 12 * MC
                var R_c = Scalar[DTYPE](1.0) / workspace[env, ws_inv_K_imp_idx + contact_tid] - workspace[env, ws_K_n_idx + contact_tid]
                workspace[env, ws_A_idx + contact_tid * MAX_CONTACTS + contact_tid] += R_c

        barrier()

        # === SEQUENTIAL: Thread 0 handles warm-start, Newton iterations, limits, friction ===
        if not valid_env or contact_tid != 0:
            return

        warmstart_normals_gpu[
            DTYPE,
            NV,
            NBODY,
            MAX_CONTACTS,
            WS_SIZE,
            BATCH,
        ](env, nc, workspace)

        # Phase 2: Projected Newton iterations
        for _ in range(NEWTON_ITERATIONS):
            # Compute gradient: g = A * lambda + rhs
            var grad_norm: workspace.element_type = 0
            for c in range(nc):
                if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                    workspace[env, ws_grad_idx + c] = Scalar[DTYPE](0)
                    continue
                var g: workspace.element_type = workspace[env, ws_rhs_idx + c]
                for c2 in range(nc):
                    if workspace[env, ws_c_dist_idx + c2] >= Scalar[DTYPE](0):
                        continue
                    g += (
                        workspace[env, ws_A_idx + c * MAX_CONTACTS + c2]
                        * workspace[env, ws_lambda_n_idx + c2]
                    )
                workspace[env, ws_grad_idx + c] = g

            # Projected gradient norm
            grad_norm = 0
            for c in range(nc):
                if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                    continue
                if workspace[env, ws_lambda_n_idx + c] > Scalar[DTYPE](0) or (
                    workspace[env, ws_grad_idx + c]
                ) < Scalar[DTYPE](0):
                    grad_norm += (
                        workspace[env, ws_grad_idx + c]
                        * workspace[env, ws_grad_idx + c]
                    )

            if grad_norm < Scalar[DTYPE](NEWTON_TOLERANCE):
                break

            # Identify free set
            var free_count = 0
            for c in range(nc):
                workspace[env, ws_fmap_idx + c] = Scalar[DTYPE](-1)
                if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                    continue
                if workspace[env, ws_lambda_n_idx + c] > Scalar[DTYPE](0) or (
                    workspace[env, ws_grad_idx + c]
                ) < Scalar[DTYPE](0):
                    workspace[env, ws_fmap_idx + c] = Scalar[DTYPE](free_count)
                    free_count += 1

            if free_count == 0:
                break

            # Solve reduced system with Jacobi + GS refinement
            for c in range(nc):
                workspace[env, ws_d_idx + c] = Scalar[DTYPE](0)

            for c in range(nc):
                if workspace[env, ws_fmap_idx + c] < Scalar[DTYPE](0):
                    continue
                var AR_diag = workspace[env, ws_A_idx + c * MAX_CONTACTS + c]
                if AR_diag > Scalar[DTYPE](1e-14):
                    workspace[env, ws_d_idx + c] = (
                        -(workspace[env, ws_grad_idx + c])
                        / AR_diag
                    )

            for _ in range(5):
                for c in range(nc):
                    if workspace[env, ws_fmap_idx + c] < Scalar[DTYPE](0):
                        continue
                    var sum_off_diag: workspace.element_type = 0
                    for c2 in range(nc):
                        if c2 == c:
                            continue
                        if workspace[env, ws_fmap_idx + c2] < Scalar[DTYPE](0):
                            continue
                        sum_off_diag += workspace[
                            env, ws_A_idx + c * MAX_CONTACTS + c2
                        ] * (workspace[env, ws_d_idx + c2])
                    workspace[env, ws_d_idx + c] = (
                        -(workspace[env, ws_grad_idx + c]) - sum_off_diag
                    ) / (workspace[env, ws_A_idx + c * MAX_CONTACTS + c])

            # Line search with Armijo condition
            var f_current: workspace.element_type = 0
            for c in range(nc):
                if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                    continue
                f_current += (
                    workspace[env, ws_rhs_idx + c]
                    * workspace[env, ws_lambda_n_idx + c]
                )
                for c2 in range(nc):
                    if workspace[env, ws_c_dist_idx + c2] >= Scalar[DTYPE](0):
                        continue
                    f_current += (
                        Scalar[DTYPE](0.5)
                        * workspace[env, ws_lambda_n_idx + c]
                        * workspace[env, ws_A_idx + c * MAX_CONTACTS + c2]
                        * workspace[env, ws_lambda_n_idx + c2]
                    )

            var gtd: workspace.element_type = 0
            for c in range(nc):
                if workspace[env, ws_fmap_idx + c] < Scalar[DTYPE](0):
                    continue
                gtd += (workspace[env, ws_grad_idx + c]) * (
                    workspace[env, ws_d_idx + c]
                )

            var step = Scalar[DTYPE](1.0)
            var armijo = Scalar[DTYPE](LINESEARCH_ARMIJO)
            var beta = Scalar[DTYPE](LINESEARCH_BETA)

            for _ in range(LINESEARCH_ITERATIONS):
                for c in range(nc):
                    workspace[env, ws_ltrial_idx + c] = workspace[
                        env, ws_lambda_n_idx + c
                    ]
                    if workspace[env, ws_fmap_idx + c] >= Scalar[DTYPE](0):
                        workspace[env, ws_ltrial_idx + c] = workspace[
                            env, ws_lambda_n_idx + c
                        ] + step * (workspace[env, ws_d_idx + c])
                    if workspace[env, ws_ltrial_idx + c] < Scalar[DTYPE](0):
                        workspace[env, ws_ltrial_idx + c] = Scalar[DTYPE](0)

                var f_trial: workspace.element_type = 0
                for c in range(nc):
                    if workspace[env, ws_c_dist_idx + c] >= Scalar[DTYPE](0):
                        continue
                    f_trial += (
                        workspace[env, ws_rhs_idx + c]
                        * workspace[env, ws_ltrial_idx + c]
                    )
                    for c2 in range(nc):
                        if workspace[env, ws_c_dist_idx + c2] >= Scalar[DTYPE](
                            0
                        ):
                            continue
                        f_trial += (
                            Scalar[DTYPE](0.5)
                            * workspace[env, ws_ltrial_idx + c]
                            * workspace[env, ws_A_idx + c * MAX_CONTACTS + c2]
                            * workspace[env, ws_ltrial_idx + c2]
                        )

                if f_trial <= f_current + armijo * step * gtd:
                    break

                step = step * beta

            for c in range(nc):
                workspace[env, ws_lambda_n_idx + c] = workspace[
                    env, ws_ltrial_idx + c
                ]

        # Apply solved normals (remove warm-start, apply final)
        apply_solved_normals_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            MAX_CONTACTS,
            STATE_SIZE,
            WS_SIZE,
            BATCH,
        ](env, nc, state, workspace)

        # Joint limits
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
            NEWTON_ITERATIONS,
        ](env, dt, state, model, workspace)

        # Friction via PGS
        _solve_friction_pgs_gpu[
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
            FRICTION_WS_OFFSET = 18 * MC + 2 * MC * NV + MC * MC,
        ](
            env,
            state,
            model,
            workspace,
            nc,
            friction_coef,
            contacts_off,
        )

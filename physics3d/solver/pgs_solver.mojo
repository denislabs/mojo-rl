"""Projected Gauss-Seidel (PGS) constraint solver for Generalized Coordinates engine.

Implements MuJoCo-style constraint-based contact solving in joint space.
The solver receives pre-built ConstraintData from the constraint builder
and iterates to find impulses satisfying all constraints.

Key features:
- Unilateral normal constraints (lambda_n >= 0)
- Coulomb friction cone clamping
- MuJoCo solref/solimp impedance model for position stabilization
- Warm-starting from previous timestep impulses
- Joint limit constraints

Reference: MuJoCo's constraint solver + existing Cartesian PGS in pgs_solver.mojo
"""

from math import sqrt
from layout import LayoutTensor, Layout
from gpu import thread_idx, block_idx, block_dim, barrier
from ..types import Model, Data, _max_one
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..traits.solver import ConstraintSolver
from ..dynamics.jacobian import (
    compute_contact_jacobian_row,
    compute_contact_jacobian_row_gpu,
    compute_angular_jacobian_row_gpu,
)
from ..constraints.constraint_data import (
    ConstraintData,
    CNSTR_NORMAL,
    CNSTR_FRICTION_T1,
    CNSTR_FRICTION_T2,
    CNSTR_FRICTION_TORSION,
    CNSTR_FRICTION_ROLL1,
    CNSTR_FRICTION_ROLL2,
    CNSTR_LIMIT,
)
from .qcqp import qcqp2, qcqp3, qcqp5
from ..gpu.constants import (
    contacts_offset,
    metadata_offset,
    model_metadata_offset,
    ws_m_inv_offset,
    ws_solver_offset,
    ws_qacc_constrained_offset,
    CONTACT_SIZE,
    CONTACT_IDX_FORCE_N,
    CONTACT_IDX_FORCE_T1,
    CONTACT_IDX_FORCE_T2,
    CONTACT_IDX_FORCE_TORSION,
    CONTACT_IDX_FORCE_ROLL1,
    CONTACT_IDX_FORCE_ROLL2,
    CONTACT_IDX_FRICTION,
    CONTACT_IDX_FRICTION_SPIN,
    CONTACT_IDX_FRICTION_ROLL,
    CONTACT_IDX_CONDIM,
    META_IDX_NUM_CONTACTS,
    MODEL_META_IDX_FRICTION,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_SOLREF_CONTACT_0,
    MODEL_META_IDX_SOLREF_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_0,
    MODEL_META_IDX_SOLIMP_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_2,
    MODEL_META_IDX_IMPRATIO,
    qvel_offset,
)
from ..constraints.constraint_builder_gpu import (
    init_common_normal_workspace_gpu,
    precompute_contact_normal_gpu,
    warmstart_normals_gpu,
    detect_and_solve_limits_gpu,
    build_and_solve_equality_gpu,
)

# PGS solver parameters
comptime PGS_ITERATIONS: Int = 30
# Minimum K for friction tangent rows — below this, direction is degenerate
comptime FRICTION_K_MIN: Float64 = 1e-6


struct PGSSolver(ConstraintSolver):
    """PGS constraint solver for Generalized Coordinates engine.

    Modifies the predicted (unconstrained) velocity in-place to satisfy
    contact constraints (non-penetration + Coulomb friction) and joint limits.
    """

    @staticmethod
    fn solver_workspace_size[NV: Int, MAX_CONTACTS: Int]() -> Int:
        """PGS solver workspace: 79*MC + 12*MC*NV floats.

        Layout (offsets relative to solver workspace start):
          Common normal block (13*MC + 2*MC*NV):
          [0..13*MC+2*MC*NV)             See constraint_builder_gpu.mojo
          Friction block (66*MC + 10*MC*NV):
          [13*MC+2*MC*NV)                lambda_f[5*MC], K_f[5*MC], dir_f[15*MC],
                                         fric_coef[5*MC], condim[MC], R_f[5*MC],
                                         bias_f[5*MC],
                                         J_f[5*MC*NV], MinvJ_f[5*MC*NV],
                                         lambda_edge_neg[5*MC], C_nt[5*MC],
                                         K_edge_pos[5*MC], K_edge_neg[5*MC],
                                         R_edge[5*MC]
        """
        comptime MC = _max_one[MAX_CONTACTS]()
        return 79 * MC + 12 * MC * NV

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
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
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
        ],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        M_inv: InlineArray[Scalar[DTYPE], M_SIZE],
        mut constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
        mut qacc: InlineArray[Scalar[DTYPE], V_SIZE],
        dt: Scalar[DTYPE],
    ):
        """Solve constraints using PGS on CPU (acceleration-level).

        Iterates over pre-built ConstraintData:
        1. Apply warm-start forces for normals and friction
        2. PGS iterations for normal constraints (with aref)
        3. PGS iterations for joint limit constraints (with aref)
        4. PGS iterations for friction (with Coulomb cone clamping)
        """
        if constraints.num_rows == 0:
            return

        var num_normals = constraints.num_normals
        var num_friction = constraints.num_friction
        var num_limits = constraints.num_limits
        var num_equality = constraints.num_equality
        var friction_start = num_normals
        var limits_start = num_normals + num_friction
        var equality_start = limits_start + num_limits

        # =====================================================================
        # Phase 1: Apply warm-start forces (normals)
        # =====================================================================
        for r in range(num_normals):
            if constraints.rows[r].lambda_val > Scalar[DTYPE](0):
                for i in range(NV):
                    qacc[i] += (
                        constraints.MinvJT[r * NV + i]
                        * constraints.rows[r].lambda_val
                    )

        # =====================================================================
        # Phase 2: Coupled PGS (normals + friction + limits together)
        # MuJoCo-style: iterate over ALL constraints in each pass.
        # =====================================================================

        # Apply friction warm-start before coupled iterations
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
        for _ in range(PGS_ITERATIONS):
            # --- Normal constraints ---
            for r in range(num_normals):
                var a: Scalar[DTYPE] = 0
                for i in range(NV):
                    a += constraints.J[r * NV + i] * qacc[i]
                var R_n = Scalar[DTYPE](1.0) / constraints.rows[r].inv_K_imp - constraints.rows[r].K
                var residual = a + constraints.rows[r].bias + R_n * constraints.rows[r].lambda_val
                var delta = -residual * constraints.rows[r].inv_K_imp
                var old_lambda = constraints.rows[r].lambda_val
                constraints.rows[r].lambda_val = (
                    constraints.rows[r].lambda_val + delta
                )
                if constraints.rows[r].lambda_val < Scalar[DTYPE](0):
                    constraints.rows[r].lambda_val = Scalar[DTYPE](0)
                var actual_delta = constraints.rows[r].lambda_val - old_lambda
                for i in range(NV):
                    qacc[i] += constraints.MinvJT[r * NV + i] * actual_delta

            # --- Friction constraints (contact-group with elliptic cone) ---
            var fric_idx = 0
            while fric_idx < num_friction:
                var r_start = friction_start + fric_idx
                var normal_row = constraints.rows[r_start].friction_parent
                var lambda_n = constraints.rows[normal_row].lambda_val

                # Count group size (consecutive rows with same friction_parent)
                var group_size = 1
                while fric_idx + group_size < num_friction:
                    if (
                        constraints.rows[
                            friction_start + fric_idx + group_size
                        ].friction_parent
                        != normal_row
                    ):
                        break
                    group_size += 1

                if lambda_n <= Scalar[DTYPE](0):
                    # Zero friction when normal force is zero (cone constraint)
                    for g in range(group_size):
                        var r = r_start + g
                        var old_f = constraints.rows[r].lambda_val
                        if old_f != Scalar[DTYPE](0):
                            constraints.rows[r].lambda_val = Scalar[DTYPE](0)
                            for i in range(NV):
                                qacc[i] -= (
                                    constraints.MinvJT[r * NV + i] * old_f
                                )
                    fric_idx += group_size
                    continue

                # Save old values for all rows in group
                var old_vals = InlineArray[Scalar[DTYPE], 5](
                    fill=Scalar[DTYPE](0)
                )
                for g in range(group_size):
                    old_vals[g] = constraints.rows[r_start + g].lambda_val

                # GS update for each row in group
                for g in range(group_size):
                    var r = r_start + g
                    if constraints.rows[r].K >= Scalar[DTYPE](FRICTION_K_MIN):
                        var a_f: Scalar[DTYPE] = 0
                        for i in range(NV):
                            a_f += constraints.J[r * NV + i] * qacc[i]
                        var R_f = Scalar[DTYPE](1.0) / constraints.rows[r].inv_K_imp - constraints.rows[r].K
                        var residual_f = (
                            a_f
                            + constraints.rows[r].bias
                            + R_f * constraints.rows[r].lambda_val
                        )
                        var delta_f = (
                            -residual_f * constraints.rows[r].inv_K_imp
                        )
                        constraints.rows[r].lambda_val = (
                            constraints.rows[r].lambda_val + delta_f
                        )

                # QCQP elliptic cone projection
                if group_size == 2:
                    var f1 = constraints.rows[r_start].lambda_val
                    var f2 = constraints.rows[r_start + 1].lambda_val
                    qcqp2[DTYPE](
                        f1,
                        f2,
                        constraints.rows[r_start].friction_coef,
                        lambda_n,
                    )
                    constraints.rows[r_start].lambda_val = f1
                    constraints.rows[r_start + 1].lambda_val = f2
                elif group_size == 3:
                    var f1 = constraints.rows[r_start].lambda_val
                    var f2 = constraints.rows[r_start + 1].lambda_val
                    var f3 = constraints.rows[r_start + 2].lambda_val
                    qcqp3[DTYPE](
                        f1,
                        f2,
                        f3,
                        constraints.rows[r_start].friction_coef,
                        constraints.rows[r_start + 1].friction_coef,
                        constraints.rows[r_start + 2].friction_coef,
                        lambda_n,
                    )
                    constraints.rows[r_start].lambda_val = f1
                    constraints.rows[r_start + 1].lambda_val = f2
                    constraints.rows[r_start + 2].lambda_val = f3
                elif group_size == 5:
                    var f1 = constraints.rows[r_start].lambda_val
                    var f2 = constraints.rows[r_start + 1].lambda_val
                    var f3 = constraints.rows[r_start + 2].lambda_val
                    var f4 = constraints.rows[r_start + 3].lambda_val
                    var f5 = constraints.rows[r_start + 4].lambda_val
                    qcqp5[DTYPE](
                        f1,
                        f2,
                        f3,
                        f4,
                        f5,
                        constraints.rows[r_start].friction_coef,
                        constraints.rows[r_start + 1].friction_coef,
                        constraints.rows[r_start + 2].friction_coef,
                        constraints.rows[r_start + 3].friction_coef,
                        constraints.rows[r_start + 4].friction_coef,
                        lambda_n,
                    )
                    constraints.rows[r_start].lambda_val = f1
                    constraints.rows[r_start + 1].lambda_val = f2
                    constraints.rows[r_start + 2].lambda_val = f3
                    constraints.rows[r_start + 3].lambda_val = f4
                    constraints.rows[r_start + 4].lambda_val = f5

                # Apply delta to qacc
                for g in range(group_size):
                    var r = r_start + g
                    var actual = constraints.rows[r].lambda_val - old_vals[g]
                    for i in range(NV):
                        qacc[i] += constraints.MinvJT[r * NV + i] * actual

                fric_idx += group_size

            # --- Joint limit constraints ---
            for r_off in range(num_limits):
                var r = limits_start + r_off
                var dof = constraints.rows[r].source_dof
                var sign = constraints.rows[r].limit_sign
                var a_limit = sign * qacc[dof]
                var R_lim = Scalar[DTYPE](1.0) / constraints.rows[r].inv_K_imp - constraints.rows[r].K
                var residual = (
                    a_limit
                    + constraints.rows[r].bias
                    + R_lim * constraints.rows[r].lambda_val
                )
                var delta = -residual * constraints.rows[r].inv_K_imp
                var old_lambda = constraints.rows[r].lambda_val
                constraints.rows[r].lambda_val = (
                    constraints.rows[r].lambda_val + delta
                )
                if constraints.rows[r].lambda_val < Scalar[DTYPE](0):
                    constraints.rows[r].lambda_val = Scalar[DTYPE](0)
                var actual = constraints.rows[r].lambda_val - old_lambda
                for i in range(NV):
                    qacc[i] += constraints.MinvJT[r * NV + i] * actual

            # --- Equality constraints (bilateral, NO clamping) ---
            for r_off in range(num_equality):
                var r = equality_start + r_off
                var a_eq: Scalar[DTYPE] = 0
                for i in range(NV):
                    a_eq += constraints.J[r * NV + i] * qacc[i]
                var R_eq = Scalar[DTYPE](1.0) / constraints.rows[r].inv_K_imp - constraints.rows[r].K
                var residual = (
                    a_eq
                    + constraints.rows[r].bias
                    + R_eq * constraints.rows[r].lambda_val
                )
                var delta = -residual * constraints.rows[r].inv_K_imp
                var old_lambda = constraints.rows[r].lambda_val
                constraints.rows[r].lambda_val = (
                    constraints.rows[r].lambda_val + delta
                )
                # Bilateral: no clamping (force can push or pull)
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
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
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
        """Solve contact constraints using PGS on GPU with 2D threading.

        Uses thread_x for environment index, thread_y for contact index.
        Precompute phases (Phase 1, Phase 3) are parallelized across contacts.
        PGS iterations are sequential on thread_y==0 (Gauss-Seidel dependency).
        All threads must hit all barriers (no early returns).
        """

        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        var contact_tid = Int(thread_idx.y)
        var valid_env = env < BATCH

        comptime qacc_idx = ws_qacc_constrained_offset[NV, NBODY]()
        comptime M_inv_idx = ws_m_inv_offset[NV, NBODY]()
        comptime solver_idx = ws_solver_offset[NV, NBODY]()
        comptime MC = _max_one[MAX_CONTACTS]()

        # Common normal block offsets (for PGS normal iterations)
        comptime ws_lambda_n = solver_idx + 0 * MC
        comptime ws_K_n = solver_idx + 1 * MC
        comptime ws_c_dist = solver_idx + 2 * MC
        comptime ws_c_body = solver_idx + 3 * MC
        comptime ws_c_body_b = solver_idx + 4 * MC
        comptime ws_c_px = solver_idx + 5 * MC
        comptime ws_c_py = solver_idx + 6 * MC
        comptime ws_c_pz = solver_idx + 7 * MC
        comptime ws_c_nx = solver_idx + 8 * MC
        comptime ws_c_ny = solver_idx + 9 * MC
        comptime ws_c_nz = solver_idx + 10 * MC
        comptime ws_pos_bias = solver_idx + 11 * MC
        comptime ws_inv_K_imp = solver_idx + 12 * MC
        comptime ws_J_n = solver_idx + 13 * MC
        comptime ws_MinvJn = solver_idx + 13 * MC + MC * NV

        # Friction workspace offsets (66*MC + 10*MC*NV, same layout as friction_solver.mojo)
        comptime fws = solver_idx + 13 * MC + 2 * MC * NV
        comptime ws_lf = fws + 0 * MC  # lambda_f[5*MC]
        comptime ws_kf = fws + 5 * MC  # K_f[5*MC]
        comptime ws_df = fws + 10 * MC  # dir_f[15*MC]
        comptime ws_fc = fws + 25 * MC  # fric_coef[5*MC]
        comptime ws_cd = fws + 30 * MC  # condim[MC]
        comptime ws_rf = fws + 31 * MC  # R_f[5*MC] (friction regularizer)
        comptime ws_bf = fws + 36 * MC  # bias_f[5*MC] (velocity damping bias)
        comptime ws_jf = fws + 41 * MC  # J_f[5*MC*NV]
        comptime ws_mj = fws + 41 * MC + 5 * MC * NV  # MinvJ_f[5*MC*NV]
        # Pyramidal-only workspace offsets
        comptime ws_le_neg = fws + 41 * MC + 10 * MC * NV  # lambda_edge_neg[5*MC]
        comptime ws_cnt = ws_le_neg + 5 * MC  # C_nt[5*MC]
        comptime ws_kep = ws_cnt + 5 * MC  # K_edge_pos[5*MC]
        comptime ws_ken = ws_kep + 5 * MC  # K_edge_neg[5*MC]
        comptime ws_re = ws_ken + 5 * MC  # R_edge[5*MC]

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
            # Init friction workspace for this contact slot
            for d in range(5):
                workspace[env, ws_lf + d * MC + contact_tid] = 0
                workspace[env, ws_kf + d * MC + contact_tid] = 1
                workspace[env, ws_fc + d * MC + contact_tid] = 0
                workspace[env, ws_rf + d * MC + contact_tid] = 0
                workspace[env, ws_bf + d * MC + contact_tid] = 0
                # Pyramidal workspace
                workspace[env, ws_le_neg + d * MC + contact_tid] = 0
                workspace[env, ws_cnt + d * MC + contact_tid] = 0
                workspace[env, ws_kep + d * MC + contact_tid] = 1
                workspace[env, ws_ken + d * MC + contact_tid] = 1
                workspace[env, ws_re + d * MC + contact_tid] = 0
                for axis in range(3):
                    workspace[
                        env, ws_df + (d * 3 + axis) * MC + contact_tid
                    ] = 0
            workspace[env, ws_cd + contact_tid] = 3  # default condim=3

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
            K_spring = Scalar[DTYPE](1.0) / (
                si_dmax * si_dmax * sr_tc * sr_tc * sr_dr * sr_dr
            )
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

        # === SEQUENTIAL: Warm start + PGS normal + joint limits (thread 0) ===
        if valid_env and contact_tid == 0:
            warmstart_normals_gpu[
                DTYPE,
                NV,
                NBODY,
                MAX_CONTACTS,
                WS_SIZE,
                BATCH,
            ](env, nc, workspace)

            # PGS normal iterations (acceleration-level)
            for _ in range(PGS_ITERATIONS):
                var max_delta: workspace.element_type = 0
                for c in range(nc):
                    if workspace[env, ws_c_dist + c] >= Scalar[DTYPE](0):
                        continue
                    var a_n: workspace.element_type = 0
                    for i in range(NV):
                        a_n += (
                            workspace[env, ws_J_n + c * NV + i]
                            * workspace[env, qacc_idx + i]
                        )
                    var R_n = Scalar[DTYPE](1.0) / rebind[Scalar[DTYPE]](workspace[env, ws_inv_K_imp + c]) - rebind[Scalar[DTYPE]](workspace[env, ws_K_n + c])
                    var residual = (
                        a_n
                        + workspace[env, ws_pos_bias + c]
                        + R_n * workspace[env, ws_lambda_n + c]
                    )
                    var delta = -residual * workspace[env, ws_inv_K_imp + c]
                    var old_lambda = workspace[env, ws_lambda_n + c]
                    workspace[env, ws_lambda_n + c] = (
                        workspace[env, ws_lambda_n + c] + delta
                    )
                    if workspace[env, ws_lambda_n + c] < Scalar[DTYPE](0):
                        workspace[env, ws_lambda_n + c] = Scalar[DTYPE](0)
                    var actual_delta = (
                        workspace[env, ws_lambda_n + c] - old_lambda
                    )
                    var abs_delta = abs(actual_delta)
                    if abs_delta > max_delta:
                        max_delta = abs_delta
                    for i in range(NV):
                        workspace[env, qacc_idx + i] += (
                            workspace[env, ws_MinvJn + c * NV + i]
                            * actual_delta
                        )
                if max_delta < Scalar[DTYPE](1e-4):
                    break

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
                PGS_ITERATIONS,
            ](env, dt, state, model, workspace)

            # Equality constraints
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
                PGS_ITERATIONS,
            ](env, state, model, workspace)

        barrier()

        # === PARALLEL PHASE 3: Each thread precomputes friction for one contact ===
        var J_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(V_SIZE):
            J_row[i] = 0

        if valid_env and contact_tid < nc:
            var c = contact_tid
            if workspace[env, ws_lambda_n + c] > 0:
                var c_off = contacts_off + c * CONTACT_SIZE
                var nx = rebind[Scalar[DTYPE]](workspace[env, ws_c_nx + c])
                var ny = rebind[Scalar[DTYPE]](workspace[env, ws_c_ny + c])
                var nz = rebind[Scalar[DTYPE]](workspace[env, ws_c_nz + c])

                # Read per-contact friction params
                var mu_slide = rebind[Scalar[DTYPE]](
                    state[env, c_off + CONTACT_IDX_FRICTION]
                )
                if mu_slide <= Scalar[DTYPE](0):
                    mu_slide = friction_coef
                var mu_spin = rebind[Scalar[DTYPE]](
                    state[env, c_off + CONTACT_IDX_FRICTION_SPIN]
                )
                var mu_roll = rebind[Scalar[DTYPE]](
                    state[env, c_off + CONTACT_IDX_FRICTION_ROLL]
                )
                var condim = Int(
                    rebind[Scalar[DTYPE]](
                        state[env, c_off + CONTACT_IDX_CONDIM]
                    )
                )
                if condim < 1:
                    condim = 3
                workspace[env, ws_cd + c] = Scalar[DTYPE](condim)

                if condim > 1:
                    # Tangent basis
                    var t1x: Scalar[DTYPE]
                    var t1y: Scalar[DTYPE]
                    var t1z: Scalar[DTYPE]
                    if abs(nx) < Scalar[DTYPE](0.9):
                        t1x = Scalar[DTYPE](0)
                        t1y = -nz
                        t1z = ny
                    else:
                        t1x = nz
                        t1y = Scalar[DTYPE](0)
                        t1z = -nx
                    var t1_mag = sqrt(t1x * t1x + t1y * t1y + t1z * t1z)
                    if t1_mag > Scalar[DTYPE](1e-10):
                        t1x = t1x / t1_mag
                        t1y = t1y / t1_mag
                        t1z = t1z / t1_mag
                    var t2x = ny * t1z - nz * t1y
                    var t2y = nz * t1x - nx * t1z
                    var t2z = nx * t1y - ny * t1x

                    # Store directions and friction coefficients
                    workspace[env, ws_df + (0 * 3 + 0) * MC + c] = t1x
                    workspace[env, ws_df + (0 * 3 + 1) * MC + c] = t1y
                    workspace[env, ws_df + (0 * 3 + 2) * MC + c] = t1z
                    workspace[env, ws_df + (1 * 3 + 0) * MC + c] = t2x
                    workspace[env, ws_df + (1 * 3 + 1) * MC + c] = t2y
                    workspace[env, ws_df + (1 * 3 + 2) * MC + c] = t2z
                    workspace[env, ws_fc + 0 * MC + c] = mu_slide
                    workspace[env, ws_fc + 1 * MC + c] = mu_slide

                    var num_fric = 2
                    if condim >= 4:
                        num_fric = 3
                        workspace[env, ws_df + (2 * 3 + 0) * MC + c] = nx
                        workspace[env, ws_df + (2 * 3 + 1) * MC + c] = ny
                        workspace[env, ws_df + (2 * 3 + 2) * MC + c] = nz
                        workspace[env, ws_fc + 2 * MC + c] = mu_spin
                    if condim >= 6:
                        num_fric = 5
                        workspace[env, ws_df + (3 * 3 + 0) * MC + c] = t1x
                        workspace[env, ws_df + (3 * 3 + 1) * MC + c] = t1y
                        workspace[env, ws_df + (3 * 3 + 2) * MC + c] = t1z
                        workspace[env, ws_df + (4 * 3 + 0) * MC + c] = t2x
                        workspace[env, ws_df + (4 * 3 + 1) * MC + c] = t2y
                        workspace[env, ws_df + (4 * 3 + 2) * MC + c] = t2z
                        workspace[env, ws_fc + 3 * MC + c] = mu_roll
                        workspace[env, ws_fc + 4 * MC + c] = mu_roll

                    var body_a = Int(workspace[env, ws_c_body + c])
                    var body_b = Int(workspace[env, ws_c_body_b + c])
                    var px = rebind[Scalar[DTYPE]](workspace[env, ws_c_px + c])
                    var py = rebind[Scalar[DTYPE]](workspace[env, ws_c_py + c])
                    var pz = rebind[Scalar[DTYPE]](workspace[env, ws_c_pz + c])

                    # Compute J, MinvJ, K for each friction direction
                    for d in range(num_fric):
                        var dx = rebind[Scalar[DTYPE]](
                            workspace[env, ws_df + (d * 3 + 0) * MC + c]
                        )
                        var dy = rebind[Scalar[DTYPE]](
                            workspace[env, ws_df + (d * 3 + 1) * MC + c]
                        )
                        var dz = rebind[Scalar[DTYPE]](
                            workspace[env, ws_df + (d * 3 + 2) * MC + c]
                        )

                        if d < 2:
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
                                dx,
                                dy,
                                dz,
                                J_row,
                            )
                        else:
                            compute_angular_jacobian_row_gpu[
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
                                dx,
                                dy,
                                dz,
                                J_row,
                            )

                        var k_d: workspace.element_type = 0
                        for i in range(NV):
                            workspace[
                                env, ws_jf + d * MC * NV + c * NV + i
                            ] = J_row[i]
                            var mi_j_sum: workspace.element_type = 0
                            for j_idx in range(NV):
                                mi_j_sum += (
                                    workspace[env, M_inv_idx + i * NV + j_idx]
                                    * J_row[j_idx]
                                )
                            workspace[
                                env, ws_mj + d * MC * NV + c * NV + i
                            ] = mi_j_sum
                            k_d += J_row[i] * mi_j_sum
                        if k_d < Scalar[DTYPE](1e-10):
                            k_d = Scalar[DTYPE](1e-10)
                        workspace[env, ws_kf + d * MC + c] = k_d

                    # Compute friction regularizer R_f from parent normal's impedance
                    var impratio_pgs = rebind[Scalar[DTYPE]](
                        model[0, model_meta_off + MODEL_META_IDX_IMPRATIO]
                    )
                    if impratio_pgs < Scalar[DTYPE](1e-6):
                        impratio_pgs = Scalar[DTYPE](1.0)
                    var imp_n_pgs = rebind[Scalar[DTYPE]](
                        workspace[env, ws_inv_K_imp + c]
                    ) * rebind[Scalar[DTYPE]](workspace[env, ws_K_n + c])
                    var R_base_pgs = (
                        (Scalar[DTYPE](1.0) - imp_n_pgs)
                        / imp_n_pgs
                        * rebind[Scalar[DTYPE]](workspace[env, ws_K_n + c])
                        / impratio_pgs
                    )
                    for d in range(num_fric):
                        var R_d_pgs = R_base_pgs
                        if d >= 2:
                            var mu_d_pgs = rebind[Scalar[DTYPE]](
                                workspace[env, ws_fc + d * MC + c]
                            )
                            if mu_d_pgs > Scalar[DTYPE](1e-12):
                                R_d_pgs = (
                                    R_base_pgs
                                    * mu_slide
                                    * mu_slide
                                    / (mu_d_pgs * mu_d_pgs)
                                )
                        workspace[env, ws_rf + d * MC + c] = R_d_pgs

                    # Compute velocity damping bias for friction rows
                    comptime qvel_off = qvel_offset[NQ, NV]()
                    for d in range(num_fric):
                        var v_t: workspace.element_type = 0
                        for i in range(NV):
                            v_t += rebind[Scalar[DTYPE]](
                                workspace[env, ws_jf + d * MC * NV + c * NV + i]
                            ) * rebind[Scalar[DTYPE]](state[env, qvel_off + i])
                        workspace[env, ws_bf + d * MC + c] = B_damp * rebind[
                            Scalar[DTYPE]
                        ](v_t)

                    @parameter
                    if CONE_TYPE == ConeType.PYRAMIDAL:
                        # Pyramidal precomputation: C_nt, K_edge_pos/neg, R_edge
                        var R_n_val = (
                            (Scalar[DTYPE](1.0) - imp_n_pgs)
                            / imp_n_pgs
                            * rebind[Scalar[DTYPE]](workspace[env, ws_K_n + c])
                        )
                        for d in range(num_fric):
                            var mu_d_p = rebind[Scalar[DTYPE]](
                                workspace[env, ws_fc + d * MC + c]
                            )
                            # Cross-term: C_nt[d][c] = Σ_i J_n[c*NV+i] * MinvJ_f[d*MC*NV+c*NV+i]
                            var c_nt_val: workspace.element_type = 0
                            for i in range(NV):
                                c_nt_val += rebind[Scalar[DTYPE]](
                                    workspace[env, ws_J_n + c * NV + i]
                                ) * rebind[Scalar[DTYPE]](
                                    workspace[
                                        env, ws_mj + d * MC * NV + c * NV + i
                                    ]
                                )
                            workspace[env, ws_cnt + d * MC + c] = c_nt_val
                            var K_n_c = rebind[Scalar[DTYPE]](
                                workspace[env, ws_K_n + c]
                            )
                            var K_f_d = rebind[Scalar[DTYPE]](
                                workspace[env, ws_kf + d * MC + c]
                            )
                            workspace[env, ws_kep + d * MC + c] = (
                                K_n_c
                                + Scalar[DTYPE](2.0) * mu_d_p * c_nt_val
                                + mu_d_p * mu_d_p * K_f_d
                            )
                            workspace[env, ws_ken + d * MC + c] = (
                                K_n_c
                                - Scalar[DTYPE](2.0) * mu_d_p * c_nt_val
                                + mu_d_p * mu_d_p * K_f_d
                            )
                            workspace[env, ws_re + d * MC + c] = (
                                Scalar[DTYPE](2.0) * mu_d_p * mu_d_p * R_n_val
                            )
                        # No warm-start for pyramidal
                        for d in range(num_fric):
                            workspace[env, ws_lf + d * MC + c] = Scalar[DTYPE](
                                0
                            )
                            workspace[env, ws_le_neg + d * MC + c] = Scalar[
                                DTYPE
                            ](0)
                    else:
                        # Warm-start friction impulses (elliptic only)
                        var warm_idx = InlineArray[Int, 5](uninitialized=True)
                        warm_idx[0] = CONTACT_IDX_FORCE_T1
                        warm_idx[1] = CONTACT_IDX_FORCE_T2
                        warm_idx[2] = CONTACT_IDX_FORCE_TORSION
                        warm_idx[3] = CONTACT_IDX_FORCE_ROLL1
                        warm_idx[4] = CONTACT_IDX_FORCE_ROLL2
                        for d in range(num_fric):
                            workspace[env, ws_lf + d * MC + c] = rebind[
                                Scalar[DTYPE]
                            ](state[env, c_off + warm_idx[d]])

        # All threads must hit this barrier
        barrier()

        # === SEQUENTIAL: Coupled PGS (normals + friction) + impulse store (thread 0) ===
        if valid_env and contact_tid == 0:
            # Coupled PGS iterations (normals + friction together, MuJoCo-style)
            for _ in range(PGS_ITERATIONS):
                # --- Normal constraints PGS update ---
                for c in range(nc):
                    if workspace[env, ws_c_dist + c] >= Scalar[DTYPE](0):
                        continue
                    var a_n: workspace.element_type = 0
                    for i in range(NV):
                        a_n += (
                            workspace[env, ws_J_n + c * NV + i]
                            * workspace[env, qacc_idx + i]
                        )
                    var R_n = Scalar[DTYPE](1.0) / rebind[Scalar[DTYPE]](workspace[env, ws_inv_K_imp + c]) - rebind[Scalar[DTYPE]](workspace[env, ws_K_n + c])
                    var residual = (
                        a_n
                        + workspace[env, ws_pos_bias + c]
                        + R_n * workspace[env, ws_lambda_n + c]
                    )
                    var delta = -residual * workspace[env, ws_inv_K_imp + c]
                    var old_lambda = workspace[env, ws_lambda_n + c]
                    workspace[env, ws_lambda_n + c] = (
                        workspace[env, ws_lambda_n + c] + delta
                    )
                    if workspace[env, ws_lambda_n + c] < Scalar[DTYPE](0):
                        workspace[env, ws_lambda_n + c] = Scalar[DTYPE](0)
                    var actual_n = workspace[env, ws_lambda_n + c] - old_lambda
                    for i in range(NV):
                        workspace[env, qacc_idx + i] += (
                            workspace[env, ws_MinvJn + c * NV + i] * actual_n
                        )

                # --- Friction constraints PGS update ---
                for c in range(nc):
                    if workspace[env, ws_lambda_n + c] <= Scalar[DTYPE](0):
                        # Zero friction when normal force is zero
                        var condim_z = Int(workspace[env, ws_cd + c])
                        var num_fric_z = 2
                        if condim_z >= 4:
                            num_fric_z = 3
                        if condim_z >= 6:
                            num_fric_z = 5
                        for d in range(num_fric_z):

                            @parameter
                            if CONE_TYPE == ConeType.PYRAMIDAL:
                                var mu_d = rebind[Scalar[DTYPE]](
                                    workspace[env, ws_fc + d * MC + c]
                                )
                                var old_pos = rebind[Scalar[DTYPE]](
                                    workspace[env, ws_lf + d * MC + c]
                                )
                                var old_neg_v = rebind[Scalar[DTYPE]](
                                    workspace[env, ws_le_neg + d * MC + c]
                                )
                                if old_pos != Scalar[DTYPE](
                                    0
                                ) or old_neg_v != Scalar[DTYPE](0):
                                    workspace[env, ws_lf + d * MC + c] = Scalar[
                                        DTYPE
                                    ](0)
                                    workspace[
                                        env, ws_le_neg + d * MC + c
                                    ] = Scalar[DTYPE](0)
                                    for i in range(NV):
                                        var minvjn_i = rebind[Scalar[DTYPE]](
                                            workspace[
                                                env, ws_MinvJn + c * NV + i
                                            ]
                                        )
                                        var minvjf_i = rebind[Scalar[DTYPE]](
                                            workspace[
                                                env,
                                                ws_mj
                                                + d * MC * NV
                                                + c * NV
                                                + i,
                                            ]
                                        )
                                        workspace[env, qacc_idx + i] -= (
                                            minvjn_i + mu_d * minvjf_i
                                        ) * old_pos
                                        workspace[env, qacc_idx + i] -= (
                                            minvjn_i - mu_d * minvjf_i
                                        ) * old_neg_v
                            else:
                                var old_f = rebind[Scalar[DTYPE]](
                                    workspace[env, ws_lf + d * MC + c]
                                )
                                if old_f != Scalar[DTYPE](0):
                                    workspace[env, ws_lf + d * MC + c] = Scalar[
                                        DTYPE
                                    ](0)
                                    for i in range(NV):
                                        workspace[env, qacc_idx + i] -= (
                                            workspace[
                                                env,
                                                ws_mj
                                                + d * MC * NV
                                                + c * NV
                                                + i,
                                            ]
                                            * old_f
                                        )
                        continue
                    var condim = Int(workspace[env, ws_cd + c])
                    if condim == 1:
                        continue

                    var num_fric = 2
                    if condim >= 4:
                        num_fric = 3
                    if condim >= 6:
                        num_fric = 5

                    var lambda_n = rebind[Scalar[DTYPE]](
                        workspace[env, ws_lambda_n + c]
                    )

                    @parameter
                    if CONE_TYPE == ConeType.PYRAMIDAL:
                        # === PYRAMIDAL CONE: Edge constraints with λ ≥ 0 ===
                        var bias_n = rebind[Scalar[DTYPE]](
                            workspace[env, ws_pos_bias + c]
                        )

                        for d in range(num_fric):
                            var mu_d = rebind[Scalar[DTYPE]](
                                workspace[env, ws_fc + d * MC + c]
                            )
                            if mu_d <= Scalar[DTYPE](1e-12):
                                continue

                            var a_n_val: workspace.element_type = 0
                            var a_f_val: workspace.element_type = 0
                            for i in range(NV):
                                var qi = rebind[Scalar[DTYPE]](
                                    workspace[env, qacc_idx + i]
                                )
                                a_n_val += (
                                    rebind[Scalar[DTYPE]](
                                        workspace[env, ws_J_n + c * NV + i]
                                    )
                                    * qi
                                )
                                a_f_val += (
                                    rebind[Scalar[DTYPE]](
                                        workspace[
                                            env,
                                            ws_jf + d * MC * NV + c * NV + i,
                                        ]
                                    )
                                    * qi
                                )

                            var R_e = rebind[Scalar[DTYPE]](
                                workspace[env, ws_re + d * MC + c]
                            )

                            # Positive edge (+)
                            var a_edge_pos = a_n_val + mu_d * a_f_val
                            var K_ep = rebind[Scalar[DTYPE]](
                                workspace[env, ws_kep + d * MC + c]
                            )
                            var residual_pos = (
                                a_edge_pos
                                + bias_n
                                + R_e
                                * rebind[Scalar[DTYPE]](
                                    workspace[env, ws_lf + d * MC + c]
                                )
                            )
                            var delta_pos = -residual_pos / (K_ep + R_e)
                            var new_lp = (
                                rebind[Scalar[DTYPE]](
                                    workspace[env, ws_lf + d * MC + c]
                                )
                                + delta_pos
                            )
                            if new_lp < Scalar[DTYPE](0):
                                new_lp = Scalar[DTYPE](0)
                            var actual_pos = new_lp - rebind[Scalar[DTYPE]](
                                workspace[env, ws_lf + d * MC + c]
                            )
                            workspace[env, ws_lf + d * MC + c] = new_lp
                            if actual_pos != Scalar[DTYPE](0):
                                for i in range(NV):
                                    workspace[env, qacc_idx + i] += (
                                        rebind[Scalar[DTYPE]](
                                            workspace[
                                                env, ws_MinvJn + c * NV + i
                                            ]
                                        )
                                        + mu_d
                                        * rebind[Scalar[DTYPE]](
                                            workspace[
                                                env,
                                                ws_mj
                                                + d * MC * NV
                                                + c * NV
                                                + i,
                                            ]
                                        )
                                    ) * actual_pos

                            # Recompute after positive edge
                            a_n_val = 0
                            a_f_val = 0
                            for i in range(NV):
                                var qi = rebind[Scalar[DTYPE]](
                                    workspace[env, qacc_idx + i]
                                )
                                a_n_val += (
                                    rebind[Scalar[DTYPE]](
                                        workspace[env, ws_J_n + c * NV + i]
                                    )
                                    * qi
                                )
                                a_f_val += (
                                    rebind[Scalar[DTYPE]](
                                        workspace[
                                            env,
                                            ws_jf + d * MC * NV + c * NV + i,
                                        ]
                                    )
                                    * qi
                                )

                            # Negative edge (-)
                            var a_edge_neg = a_n_val - mu_d * a_f_val
                            var K_en = rebind[Scalar[DTYPE]](
                                workspace[env, ws_ken + d * MC + c]
                            )
                            var residual_neg = (
                                a_edge_neg
                                + bias_n
                                + R_e
                                * rebind[Scalar[DTYPE]](
                                    workspace[env, ws_le_neg + d * MC + c]
                                )
                            )
                            var delta_neg = -residual_neg / (K_en + R_e)
                            var new_ln = (
                                rebind[Scalar[DTYPE]](
                                    workspace[env, ws_le_neg + d * MC + c]
                                )
                                + delta_neg
                            )
                            if new_ln < Scalar[DTYPE](0):
                                new_ln = Scalar[DTYPE](0)
                            var actual_neg = new_ln - rebind[Scalar[DTYPE]](
                                workspace[env, ws_le_neg + d * MC + c]
                            )
                            workspace[env, ws_le_neg + d * MC + c] = new_ln
                            if actual_neg != Scalar[DTYPE](0):
                                for i in range(NV):
                                    workspace[env, qacc_idx + i] += (
                                        rebind[Scalar[DTYPE]](
                                            workspace[
                                                env, ws_MinvJn + c * NV + i
                                            ]
                                        )
                                        - mu_d
                                        * rebind[Scalar[DTYPE]](
                                            workspace[
                                                env,
                                                ws_mj
                                                + d * MC * NV
                                                + c * NV
                                                + i,
                                            ]
                                        )
                                    ) * actual_neg
                    else:
                        # === ELLIPTIC CONE: QCQP projection (default) ===
                        var old_vals = InlineArray[Scalar[DTYPE], 5](
                            fill=Scalar[DTYPE](0)
                        )
                        for d in range(num_fric):
                            old_vals[d] = rebind[Scalar[DTYPE]](
                                workspace[env, ws_lf + d * MC + c]
                            )

                        # GS update for each friction direction (regularized)
                        for d in range(num_fric):
                            if workspace[env, ws_kf + d * MC + c] >= Scalar[
                                DTYPE
                            ](FRICTION_K_MIN):
                                var a_f: workspace.element_type = 0
                                for i in range(NV):
                                    a_f += (
                                        workspace[
                                            env,
                                            ws_jf + d * MC * NV + c * NV + i,
                                        ]
                                        * workspace[env, qacc_idx + i]
                                    )
                                var R_f_d = workspace[env, ws_rf + d * MC + c]
                                var residual_f = (
                                    a_f
                                    + workspace[env, ws_bf + d * MC + c]
                                    + R_f_d * workspace[env, ws_lf + d * MC + c]
                                )
                                var inv_AR = Scalar[DTYPE](1.0) / (
                                    workspace[env, ws_kf + d * MC + c] + R_f_d
                                )
                                var delta_f = -residual_f * inv_AR
                                workspace[env, ws_lf + d * MC + c] = (
                                    workspace[env, ws_lf + d * MC + c] + delta_f
                                )

                        # QCQP elliptic cone projection
                        if num_fric == 2:
                            var f1 = rebind[Scalar[DTYPE]](
                                workspace[env, ws_lf + 0 * MC + c]
                            )
                            var f2 = rebind[Scalar[DTYPE]](
                                workspace[env, ws_lf + 1 * MC + c]
                            )
                            qcqp2[DTYPE](
                                f1,
                                f2,
                                rebind[Scalar[DTYPE]](
                                    workspace[env, ws_fc + 0 * MC + c]
                                ),
                                lambda_n,
                            )
                            workspace[env, ws_lf + 0 * MC + c] = f1
                            workspace[env, ws_lf + 1 * MC + c] = f2
                        elif num_fric == 3:
                            var f1 = rebind[Scalar[DTYPE]](
                                workspace[env, ws_lf + 0 * MC + c]
                            )
                            var f2 = rebind[Scalar[DTYPE]](
                                workspace[env, ws_lf + 1 * MC + c]
                            )
                            var f3 = rebind[Scalar[DTYPE]](
                                workspace[env, ws_lf + 2 * MC + c]
                            )
                            qcqp3[DTYPE](
                                f1,
                                f2,
                                f3,
                                rebind[Scalar[DTYPE]](
                                    workspace[env, ws_fc + 0 * MC + c]
                                ),
                                rebind[Scalar[DTYPE]](
                                    workspace[env, ws_fc + 1 * MC + c]
                                ),
                                rebind[Scalar[DTYPE]](
                                    workspace[env, ws_fc + 2 * MC + c]
                                ),
                                lambda_n,
                            )
                            workspace[env, ws_lf + 0 * MC + c] = f1
                            workspace[env, ws_lf + 1 * MC + c] = f2
                            workspace[env, ws_lf + 2 * MC + c] = f3
                        elif num_fric == 5:
                            var f1 = rebind[Scalar[DTYPE]](
                                workspace[env, ws_lf + 0 * MC + c]
                            )
                            var f2 = rebind[Scalar[DTYPE]](
                                workspace[env, ws_lf + 1 * MC + c]
                            )
                            var f3 = rebind[Scalar[DTYPE]](
                                workspace[env, ws_lf + 2 * MC + c]
                            )
                            var f4 = rebind[Scalar[DTYPE]](
                                workspace[env, ws_lf + 3 * MC + c]
                            )
                            var f5 = rebind[Scalar[DTYPE]](
                                workspace[env, ws_lf + 4 * MC + c]
                            )
                            qcqp5[DTYPE](
                                f1,
                                f2,
                                f3,
                                f4,
                                f5,
                                rebind[Scalar[DTYPE]](
                                    workspace[env, ws_fc + 0 * MC + c]
                                ),
                                rebind[Scalar[DTYPE]](
                                    workspace[env, ws_fc + 1 * MC + c]
                                ),
                                rebind[Scalar[DTYPE]](
                                    workspace[env, ws_fc + 2 * MC + c]
                                ),
                                rebind[Scalar[DTYPE]](
                                    workspace[env, ws_fc + 3 * MC + c]
                                ),
                                rebind[Scalar[DTYPE]](
                                    workspace[env, ws_fc + 4 * MC + c]
                                ),
                                lambda_n,
                            )
                            workspace[env, ws_lf + 0 * MC + c] = f1
                            workspace[env, ws_lf + 1 * MC + c] = f2
                            workspace[env, ws_lf + 2 * MC + c] = f3
                            workspace[env, ws_lf + 3 * MC + c] = f4
                            workspace[env, ws_lf + 4 * MC + c] = f5

                        # Apply delta to qacc
                        for d in range(num_fric):
                            var actual = (
                                workspace[env, ws_lf + d * MC + c] - old_vals[d]
                            )
                            if actual != Scalar[DTYPE](0):
                                for i in range(NV):
                                    workspace[env, qacc_idx + i] += (
                                        workspace[
                                            env,
                                            ws_mj + d * MC * NV + c * NV + i,
                                        ]
                                        * actual
                                    )

            # Store impulses back to state buffer for warm-starting
            @parameter
            if CONE_TYPE == ConeType.PYRAMIDAL:
                # Pyramidal: force_n includes edge contributions
                for c in range(nc):
                    var c_off = contacts_off + c * CONTACT_SIZE
                    var condim = Int(workspace[env, ws_cd + c])
                    var num_fric = 2
                    if condim >= 4:
                        num_fric = 3
                    if condim >= 6:
                        num_fric = 5
                    var total_n = rebind[Scalar[DTYPE]](
                        workspace[env, ws_lambda_n + c]
                    )
                    for d in range(num_fric):
                        total_n += rebind[Scalar[DTYPE]](
                            workspace[env, ws_lf + d * MC + c]
                        )
                        total_n += rebind[Scalar[DTYPE]](
                            workspace[env, ws_le_neg + d * MC + c]
                        )
                    state[env, c_off + CONTACT_IDX_FORCE_N] = total_n
                    var mu_0 = rebind[Scalar[DTYPE]](
                        workspace[env, ws_fc + 0 * MC + c]
                    )
                    state[env, c_off + CONTACT_IDX_FORCE_T1] = mu_0 * (
                        rebind[Scalar[DTYPE]](
                            workspace[env, ws_lf + 0 * MC + c]
                        )
                        - rebind[Scalar[DTYPE]](
                            workspace[env, ws_le_neg + 0 * MC + c]
                        )
                    )
                    var mu_1 = rebind[Scalar[DTYPE]](
                        workspace[env, ws_fc + 1 * MC + c]
                    )
                    state[env, c_off + CONTACT_IDX_FORCE_T2] = mu_1 * (
                        rebind[Scalar[DTYPE]](
                            workspace[env, ws_lf + 1 * MC + c]
                        )
                        - rebind[Scalar[DTYPE]](
                            workspace[env, ws_le_neg + 1 * MC + c]
                        )
                    )
                    if condim >= 4:
                        var mu_2 = rebind[Scalar[DTYPE]](
                            workspace[env, ws_fc + 2 * MC + c]
                        )
                        state[env, c_off + CONTACT_IDX_FORCE_TORSION] = mu_2 * (
                            rebind[Scalar[DTYPE]](
                                workspace[env, ws_lf + 2 * MC + c]
                            )
                            - rebind[Scalar[DTYPE]](
                                workspace[env, ws_le_neg + 2 * MC + c]
                            )
                        )
                    if condim >= 6:
                        var mu_3 = rebind[Scalar[DTYPE]](
                            workspace[env, ws_fc + 3 * MC + c]
                        )
                        state[env, c_off + CONTACT_IDX_FORCE_ROLL1] = mu_3 * (
                            rebind[Scalar[DTYPE]](
                                workspace[env, ws_lf + 3 * MC + c]
                            )
                            - rebind[Scalar[DTYPE]](
                                workspace[env, ws_le_neg + 3 * MC + c]
                            )
                        )
                        var mu_4 = rebind[Scalar[DTYPE]](
                            workspace[env, ws_fc + 4 * MC + c]
                        )
                        state[env, c_off + CONTACT_IDX_FORCE_ROLL2] = mu_4 * (
                            rebind[Scalar[DTYPE]](
                                workspace[env, ws_lf + 4 * MC + c]
                            )
                            - rebind[Scalar[DTYPE]](
                                workspace[env, ws_le_neg + 4 * MC + c]
                            )
                        )
            else:
                # Elliptic: direct force writeback
                for c in range(nc):
                    var c_off = contacts_off + c * CONTACT_SIZE
                    state[env, c_off + CONTACT_IDX_FORCE_N] = workspace[
                        env, ws_lambda_n + c
                    ]
                    state[env, c_off + CONTACT_IDX_FORCE_T1] = workspace[
                        env, ws_lf + 0 * MC + c
                    ]
                    state[env, c_off + CONTACT_IDX_FORCE_T2] = workspace[
                        env, ws_lf + 1 * MC + c
                    ]
                    var condim = Int(workspace[env, ws_cd + c])
                    if condim >= 4:
                        state[
                            env, c_off + CONTACT_IDX_FORCE_TORSION
                        ] = workspace[env, ws_lf + 2 * MC + c]
                    if condim >= 6:
                        state[env, c_off + CONTACT_IDX_FORCE_ROLL1] = workspace[
                            env, ws_lf + 3 * MC + c
                        ]
                        state[env, c_off + CONTACT_IDX_FORCE_ROLL2] = workspace[
                            env, ws_lf + 4 * MC + c
                        ]

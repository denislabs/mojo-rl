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
)
from ..constraints.constraint_data import (
    ConstraintData,
    CNSTR_NORMAL,
    CNSTR_FRICTION_T1,
    CNSTR_FRICTION_T2,
    CNSTR_LIMIT,
)
from ..gpu.constants import (
    contacts_offset,
    metadata_offset,
    model_metadata_offset,
    ws_m_inv_offset,
    ws_solver_offset,
    ws_qvel_pred_offset,
    CONTACT_SIZE,
    CONTACT_IDX_IMPULSE_N,
    CONTACT_IDX_IMPULSE_T1,
    CONTACT_IDX_IMPULSE_T2,
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
    detect_and_solve_limits_gpu,
)

# PGS solver parameters
comptime PGS_ITERATIONS: Int = 30


struct PGSSolver(ConstraintSolver):
    """PGS constraint solver for Generalized Coordinates engine.

    Modifies the predicted (unconstrained) velocity in-place to satisfy
    contact constraints (non-penetration + Coulomb friction) and joint limits.
    """

    @staticmethod
    fn solver_workspace_size[NV: Int, MAX_CONTACTS: Int]() -> Int:
        """PGS solver workspace: 23 * MC + 6 * MC * NV floats.

        Layout (offsets relative to solver workspace start):
          Common normal block (13*MC + 2*MC*NV):
          [0..13*MC+2*MC*NV)             See constraint_builder_gpu.mojo
          PGS friction block (10*MC + 4*MC*NV):
          [13*MC+2*MC*NV+0*MC)           lambda_t1  Tangent 1 impulse
          [13*MC+2*MC*NV+1*MC)           lambda_t2  Tangent 2 impulse
          [13*MC+2*MC*NV+2*MC)           K_t1       Tangent 1 effective mass
          [13*MC+2*MC*NV+3*MC)           K_t2       Tangent 2 effective mass
          [13*MC+2*MC*NV+4*MC..+10*MC)   t1xyz/t2xyz Tangent directions
          [13*MC+2*MC*NV+10*MC)          J_t1, J_t2, MinvJt1, MinvJt2 (4*MC*NV)
        """
        comptime MC = _max_one[MAX_CONTACTS]()
        return 23 * MC + 6 * MC * NV

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
    ](
        model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        M_inv: InlineArray[Scalar[DTYPE], M_SIZE],
        mut constraints: ConstraintData[DTYPE, MAX_ROWS, NV],
        mut qvel: InlineArray[Scalar[DTYPE], V_SIZE],
        dt: Scalar[DTYPE],
    ):
        """Solve constraints using PGS on CPU.

        Iterates over pre-built ConstraintData:
        1. Apply warm-start impulses for normals and friction
        2. PGS iterations for normal constraints (with impedance)
        3. PGS iterations for joint limit constraints (with impedance)
        4. PGS iterations for friction (with Coulomb cone clamping)
        """
        if constraints.num_rows == 0:
            return

        var num_normals = constraints.num_normals
        var num_friction = constraints.num_friction
        var num_limits = constraints.num_limits
        var friction_start = num_normals
        var limits_start = num_normals + num_friction

        # =====================================================================
        # Phase 1: Apply warm-start impulses (normals)
        # =====================================================================
        for r in range(num_normals):
            if constraints.rows[r].lambda_val > Scalar[DTYPE](0):
                for i in range(NV):
                    qvel[i] += (
                        constraints.MinvJT[r * NV + i]
                        * constraints.rows[r].lambda_val
                    )

        # =====================================================================
        # Phase 2: PGS normal iterations
        # =====================================================================
        for _ in range(PGS_ITERATIONS):
            for r in range(num_normals):
                # Compute constraint velocity: v = J . qvel
                var v: Scalar[DTYPE] = 0
                for i in range(NV):
                    v += constraints.J[r * NV + i] * qvel[i]

                # PGS update with impedance: delta = -(v + bias) * inv_K_imp
                var delta = (
                    -(v + constraints.rows[r].bias)
                    * constraints.rows[r].inv_K_imp
                )
                var old_lambda = constraints.rows[r].lambda_val
                constraints.rows[r].lambda_val = (
                    constraints.rows[r].lambda_val + delta
                )

                # Unilateral clamp: lambda >= 0
                if constraints.rows[r].lambda_val < Scalar[DTYPE](0):
                    constraints.rows[r].lambda_val = Scalar[DTYPE](0)

                var actual_delta = constraints.rows[r].lambda_val - old_lambda

                # Apply velocity correction: qvel += MinvJT * delta
                for i in range(NV):
                    qvel[i] += constraints.MinvJT[r * NV + i] * actual_delta

        # =====================================================================
        # Phase 2b: PGS joint limit iterations
        # =====================================================================
        if num_limits > 0:
            for _ in range(PGS_ITERATIONS):
                for r_off in range(num_limits):
                    var r = limits_start + r_off
                    var dof = constraints.rows[r].source_dof
                    var sign = constraints.rows[r].limit_sign
                    var v_limit = sign * qvel[dof]

                    var delta = (
                        -(v_limit + constraints.rows[r].bias)
                        * constraints.rows[r].inv_K_imp
                    )
                    var old_lambda = constraints.rows[r].lambda_val
                    constraints.rows[r].lambda_val = (
                        constraints.rows[r].lambda_val + delta
                    )

                    if constraints.rows[r].lambda_val < Scalar[DTYPE](0):
                        constraints.rows[r].lambda_val = Scalar[DTYPE](0)

                    var actual = constraints.rows[r].lambda_val - old_lambda

                    # Apply: qvel += MinvJT * actual
                    for i in range(NV):
                        qvel[i] += constraints.MinvJT[r * NV + i] * actual

        # =====================================================================
        # Phase 3: Friction PGS with Coulomb cone
        # =====================================================================
        if num_friction == 0:
            return

        # Apply friction warm-start
        for r_off in range(num_friction):
            var r = friction_start + r_off
            if constraints.rows[r].lambda_val != Scalar[DTYPE](0):
                for i in range(NV):
                    qvel[i] += (
                        constraints.MinvJT[r * NV + i]
                        * constraints.rows[r].lambda_val
                    )

        # Friction PGS iterations (t1 and t2 are consecutive pairs)
        for _ in range(PGS_ITERATIONS):
            # Process friction pairs (t1, t2 are consecutive)
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
                var v_t1: Scalar[DTYPE] = 0
                for i in range(NV):
                    v_t1 += constraints.J[r_t1 * NV + i] * qvel[i]
                var delta_t1 = -v_t1 / constraints.rows[r_t1].K
                var old_t1 = constraints.rows[r_t1].lambda_val
                constraints.rows[r_t1].lambda_val = (
                    constraints.rows[r_t1].lambda_val + delta_t1
                )

                # Tangent 2
                var v_t2: Scalar[DTYPE] = 0
                for i in range(NV):
                    v_t2 += constraints.J[r_t2 * NV + i] * qvel[i]
                var delta_t2 = -v_t2 / constraints.rows[r_t2].K
                var old_t2 = constraints.rows[r_t2].lambda_val
                constraints.rows[r_t2].lambda_val = (
                    constraints.rows[r_t2].lambda_val + delta_t2
                )

                # Coulomb cone clamping: |lambda_t| <= mu * lambda_n
                var t_mag = sqrt(
                    constraints.rows[r_t1].lambda_val
                    * constraints.rows[r_t1].lambda_val
                    + constraints.rows[r_t2].lambda_val
                    * constraints.rows[r_t2].lambda_val
                )
                if t_mag > max_friction:
                    var scale = max_friction / t_mag
                    constraints.rows[r_t1].lambda_val = (
                        constraints.rows[r_t1].lambda_val * scale
                    )
                    constraints.rows[r_t2].lambda_val = (
                        constraints.rows[r_t2].lambda_val * scale
                    )

                var actual_t1 = constraints.rows[r_t1].lambda_val - old_t1
                var actual_t2 = constraints.rows[r_t2].lambda_val - old_t2

                for i in range(NV):
                    qvel[i] += (
                        constraints.MinvJT[r_t1 * NV + i] * actual_t1
                        + constraints.MinvJT[r_t2 * NV + i] * actual_t2
                    )

                pair_idx += 2

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
        """Solve contact constraints using PGS on GPU with 2D threading.

        Uses thread_x for environment index, thread_y for contact index.
        Precompute phases (Phase 1, Phase 3) are parallelized across contacts.
        PGS iterations are sequential on thread_y==0 (Gauss-Seidel dependency).
        All threads must hit all barriers (no early returns).
        """

        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        var contact_tid = Int(thread_idx.y)
        var valid_env = env < BATCH

        comptime qvel_idx = ws_qvel_pred_offset[NV, NBODY]()
        comptime M_inv_idx = ws_m_inv_offset[NV, NBODY]()
        comptime solver_idx = ws_solver_offset[NV, NBODY]()
        comptime MC = _max_one[MAX_CONTACTS]()

        # Common normal block offsets (for PGS normal iterations)
        comptime ws_lambda_n = solver_idx + 0 * MC
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

        # PGS friction block offsets (after common normal block)
        comptime FRIC = solver_idx + 13 * MC + 2 * MC * NV
        comptime ws_lambda_t1 = FRIC + 0 * MC
        comptime ws_lambda_t2 = FRIC + 1 * MC
        comptime ws_K_t1 = FRIC + 2 * MC
        comptime ws_K_t2 = FRIC + 3 * MC
        comptime ws_t1x = FRIC + 4 * MC
        comptime ws_t1y = FRIC + 5 * MC
        comptime ws_t1z = FRIC + 6 * MC
        comptime ws_t2x = FRIC + 7 * MC
        comptime ws_t2y = FRIC + 8 * MC
        comptime ws_t2z = FRIC + 9 * MC
        comptime ws_J_t1 = FRIC + 10 * MC
        comptime ws_J_t2 = FRIC + 10 * MC + MC * NV
        comptime ws_MinvJt1 = FRIC + 10 * MC + 2 * MC * NV
        comptime ws_MinvJt2 = FRIC + 10 * MC + 3 * MC * NV

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
            # Init PGS friction workspace
            workspace[env, ws_lambda_t1 + contact_tid] = 0
            workspace[env, ws_lambda_t2 + contact_tid] = 0
            workspace[env, ws_K_t1 + contact_tid] = 1
            workspace[env, ws_K_t2 + contact_tid] = 1
            workspace[env, ws_t1x + contact_tid] = 0
            workspace[env, ws_t1y + contact_tid] = 0
            workspace[env, ws_t1z + contact_tid] = 0
            workspace[env, ws_t2x + contact_tid] = 0
            workspace[env, ws_t2y + contact_tid] = 0
            workspace[env, ws_t2z + contact_tid] = 0

        # Read metadata
        comptime contacts_off = contacts_offset[NQ, NV, NBODY]()
        comptime meta_off = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
        comptime model_meta_off = model_metadata_offset[NBODY, NJOINT]()

        var nc = 0
        var dt: Scalar[DTYPE] = 0
        var friction_coef: Scalar[DTYPE] = 0
        var inv_tc_dr: Scalar[DTYPE] = 0
        var b_vel_coef: Scalar[DTYPE] = 0
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
            inv_tc_dr = Scalar[DTYPE](1.0) / (sr_tc * sr_dr)
            b_vel_coef = Scalar[DTYPE](2.0) * sr_dr * dt / (si_dmax * sr_tc)

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
                inv_tc_dr,
                b_vel_coef,
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

            # PGS normal iterations
            var vel_factor = Scalar[DTYPE](1.0) - b_vel_coef
            for _ in range(PGS_ITERATIONS):
                var max_delta: workspace.element_type = 0
                for c in range(nc):
                    if workspace[env, ws_c_dist + c] >= Scalar[DTYPE](0):
                        continue
                    var v_n: workspace.element_type = 0
                    for i in range(NV):
                        v_n += (
                            workspace[env, ws_J_n + c * NV + i]
                            * workspace[env, qvel_idx + i]
                        )
                    var delta = (
                        -(v_n * vel_factor - workspace[env, ws_pos_bias + c])
                        * workspace[env, ws_inv_K_imp + c]
                    )
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
                        workspace[env, qvel_idx + i] += (
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

        barrier()

        # === PARALLEL PHASE 3: Each thread precomputes tangent for one contact ===
        var J_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(V_SIZE):
            J_row[i] = 0

        if valid_env and contact_tid < nc:
            var c = contact_tid
            if workspace[env, ws_lambda_n + c] > 0:
                var nx = workspace[env, ws_c_nx + c]
                var ny = workspace[env, ws_c_ny + c]
                var nz = workspace[env, ws_c_nz + c]

                # Tangent basis
                if abs(nx) < 0.9:
                    workspace[env, ws_t1x + c] = 0
                    workspace[env, ws_t1y + c] = -nz
                    workspace[env, ws_t1z + c] = ny
                else:
                    workspace[env, ws_t1x + c] = nz
                    workspace[env, ws_t1y + c] = 0
                    workspace[env, ws_t1z + c] = -nx

                var t1_mag = sqrt(
                    workspace[env, ws_t1x + c] * workspace[env, ws_t1x + c]
                    + workspace[env, ws_t1y + c] * workspace[env, ws_t1y + c]
                    + workspace[env, ws_t1z + c] * workspace[env, ws_t1z + c]
                )
                if t1_mag > Scalar[DTYPE](1e-10):
                    workspace[env, ws_t1x + c] /= t1_mag
                    workspace[env, ws_t1y + c] /= t1_mag
                    workspace[env, ws_t1z + c] /= t1_mag

                workspace[env, ws_t2x + c] = (
                    ny * workspace[env, ws_t1z + c]
                    - nz * workspace[env, ws_t1y + c]
                )
                workspace[env, ws_t2y + c] = (
                    nz * workspace[env, ws_t1x + c]
                    - nx * workspace[env, ws_t1z + c]
                )
                workspace[env, ws_t2z + c] = (
                    nx * workspace[env, ws_t1y + c]
                    - ny * workspace[env, ws_t1x + c]
                )

                # Compute J_t1, K_t1, MinvJt1
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
                    Int(workspace[env, ws_c_body + c]),
                    Int(workspace[env, ws_c_body_b + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_px + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_py + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_pz + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_t1x + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_t1y + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_t1z + c]),
                    J_row,
                )
                var k1: workspace.element_type = 0
                for i in range(NV):
                    workspace[env, ws_J_t1 + c * NV + i] = J_row[i]
                    var mi_j_sum: workspace.element_type = 0
                    for j_idx in range(NV):
                        mi_j_sum += (
                            workspace[env, M_inv_idx + i * NV + j_idx]
                            * J_row[j_idx]
                        )
                    workspace[env, ws_MinvJt1 + c * NV + i] = mi_j_sum
                    k1 += J_row[i] * mi_j_sum
                if k1 < Scalar[DTYPE](1e-10):
                    k1 = Scalar[DTYPE](1e-10)
                workspace[env, ws_K_t1 + c] = k1

                # Compute J_t2, K_t2, MinvJt2
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
                    Int(workspace[env, ws_c_body + c]),
                    Int(workspace[env, ws_c_body_b + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_px + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_py + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_pz + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_t2x + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_t2y + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_t2z + c]),
                    J_row,
                )
                var k2: workspace.element_type = 0
                for i in range(NV):
                    workspace[env, ws_J_t2 + c * NV + i] = J_row[i]
                    var mi_j_sum: workspace.element_type = 0
                    for j_idx in range(NV):
                        mi_j_sum += (
                            workspace[env, M_inv_idx + i * NV + j_idx]
                            * J_row[j_idx]
                        )
                    workspace[env, ws_MinvJt2 + c * NV + i] = mi_j_sum
                    k2 += J_row[i] * mi_j_sum
                if k2 < Scalar[DTYPE](1e-10):
                    k2 = Scalar[DTYPE](1e-10)
                workspace[env, ws_K_t2 + c] = k2

                # Store warm start tangent impulses
                var c_off = contacts_off + c * CONTACT_SIZE
                workspace[env, ws_lambda_t1 + c] = rebind[Scalar[DTYPE]](
                    state[env, c_off + CONTACT_IDX_IMPULSE_T1]
                )
                workspace[env, ws_lambda_t2 + c] = rebind[Scalar[DTYPE]](
                    state[env, c_off + CONTACT_IDX_IMPULSE_T2]
                )

        # All threads must hit this barrier
        barrier()

        # === SEQUENTIAL: Friction PGS + impulse store (thread 0 only) ===
        if valid_env and contact_tid == 0:
            # Friction PGS iterations
            for _ in range(PGS_ITERATIONS):
                var max_fric_delta: workspace.element_type = 0
                for c in range(nc):
                    if workspace[env, ws_lambda_n + c] <= Scalar[DTYPE](0):
                        continue

                    var max_friction = (
                        friction_coef * workspace[env, ws_lambda_n + c]
                    )

                    var v_t1: workspace.element_type = 0
                    for i in range(NV):
                        v_t1 += (
                            workspace[env, ws_J_t1 + c * NV + i]
                            * workspace[env, qvel_idx + i]
                        )
                    var delta_t1 = -v_t1 / workspace[env, ws_K_t1 + c]
                    var old_t1 = workspace[env, ws_lambda_t1 + c]
                    workspace[env, ws_lambda_t1 + c] = (
                        workspace[env, ws_lambda_t1 + c] + delta_t1
                    )

                    var v_t2: workspace.element_type = 0
                    for i in range(NV):
                        v_t2 += (
                            workspace[env, ws_J_t2 + c * NV + i]
                            * workspace[env, qvel_idx + i]
                        )
                    var delta_t2 = -v_t2 / workspace[env, ws_K_t2 + c]
                    var old_t2 = workspace[env, ws_lambda_t2 + c]
                    workspace[env, ws_lambda_t2 + c] = (
                        workspace[env, ws_lambda_t2 + c] + delta_t2
                    )

                    # Coulomb cone clamping
                    var t_mag = sqrt(
                        workspace[env, ws_lambda_t1 + c]
                        * workspace[env, ws_lambda_t1 + c]
                        + workspace[env, ws_lambda_t2 + c]
                        * workspace[env, ws_lambda_t2 + c]
                    )
                    if t_mag > max_friction:
                        var scale = max_friction / t_mag
                        workspace[env, ws_lambda_t1 + c] = (
                            workspace[env, ws_lambda_t1 + c] * scale
                        )
                        workspace[env, ws_lambda_t2 + c] = (
                            workspace[env, ws_lambda_t2 + c] * scale
                        )

                    var actual_t1 = workspace[env, ws_lambda_t1 + c] - old_t1
                    var actual_t2 = workspace[env, ws_lambda_t2 + c] - old_t2

                    var abs_t1 = abs(actual_t1)
                    var abs_t2 = abs(actual_t2)
                    if abs_t1 > max_fric_delta:
                        max_fric_delta = abs_t1
                    if abs_t2 > max_fric_delta:
                        max_fric_delta = abs_t2

                    for i in range(NV):
                        workspace[env, qvel_idx + i] += (
                            workspace[env, ws_MinvJt1 + c * NV + i] * actual_t1
                            + workspace[env, ws_MinvJt2 + c * NV + i]
                            * actual_t2
                        )
                if max_fric_delta < Scalar[DTYPE](1e-4):
                    break

            # Store impulses back to state buffer for warm-starting
            for c in range(nc):
                var c_off = contacts_off + c * CONTACT_SIZE
                state[env, c_off + CONTACT_IDX_IMPULSE_N] = workspace[
                    env, ws_lambda_n + c
                ]
                state[env, c_off + CONTACT_IDX_IMPULSE_T1] = workspace[
                    env, ws_lambda_t1 + c
                ]
                state[env, c_off + CONTACT_IDX_IMPULSE_T2] = workspace[
                    env, ws_lambda_t2 + c
                ]

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
from .constraint_data import (
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
    model_joint_offset,
    ws_m_inv_offset,
    ws_solver_offset,
    ws_cdof_offset,
    ws_qvel_pred_offset,
    CONTACT_SIZE,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_POS_X,
    CONTACT_IDX_POS_Y,
    CONTACT_IDX_POS_Z,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_DIST,
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
    MODEL_META_IDX_SOLREF_LIMIT_0,
    MODEL_META_IDX_SOLREF_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_0,
    MODEL_META_IDX_SOLIMP_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_2,
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    MAX_POS_CORRECTION_VEL,
)
from ..joint_types import (
    JNT_HINGE,
    JNT_SLIDE,
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
          Scalars (21 * MC):
          [0*MC..1*MC)                   lambda_n   Normal impulse accumulators
          [1*MC..2*MC)                   K_n        Effective mass (diagonal)
          [2*MC..3*MC)                   c_dist     Contact distance
          [3*MC..4*MC)                   c_body     Body A index (as Float)
          [4*MC..5*MC)                   c_body_b   Body B index (as Float)
          [5*MC..6*MC)                   c_px       Contact position X
          [6*MC..7*MC)                   c_py       Contact position Y
          [7*MC..8*MC)                   c_pz       Contact position Z
          [8*MC..9*MC)                   c_nx       Contact normal X
          [9*MC..10*MC)                  c_ny       Contact normal Y
          [10*MC..11*MC)                 c_nz       Contact normal Z
          [11*MC..12*MC)                 lambda_t1  Tangent 1 impulse
          [12*MC..13*MC)                 lambda_t2  Tangent 2 impulse
          [13*MC..14*MC)                 K_t1       Tangent 1 effective mass
          [14*MC..15*MC)                 K_t2       Tangent 2 effective mass
          [15*MC..16*MC)                 t1x        Tangent 1 direction X
          [16*MC..17*MC)                 t1y        Tangent 1 direction Y
          [17*MC..18*MC)                 t1z        Tangent 1 direction Z
          [18*MC..19*MC)                 t2x        Tangent 2 direction X
          [19*MC..20*MC)                 t2y        Tangent 2 direction Y
          [20*MC..21*MC)                 t2z        Tangent 2 direction Z
          Jacobians (3 * MC * NV):
          [21*MC..21*MC+MC*NV)           J_n        Normal Jacobian (MC x NV)
          [21*MC+MC*NV..21*MC+2*MC*NV)   J_t1       Tangent 1 Jacobian (MC x NV)
          [21*MC+2*MC*NV..21*MC+3*MC*NV) J_t2       Tangent 2 Jacobian (MC x NV)
          Precomputed M_inv @ J^T (3 * MC * NV):
          [21*MC+3*MC*NV..21*MC+4*MC*NV) MinvJn     M_inv @ J_n^T (MC x NV)
          [21*MC+4*MC*NV..21*MC+5*MC*NV) MinvJt1    M_inv @ J_t1^T (MC x NV)
          [21*MC+5*MC*NV..21*MC+6*MC*NV) MinvJt2    M_inv @ J_t2^T (MC x NV)
          Precomputed impedance (2 * MC):
          [21*MC+6*MC*NV..22*MC+6*MC*NV) pos_bias   imp*pen*inv_tc_dr per contact
          [22*MC+6*MC*NV..23*MC+6*MC*NV) inv_K_imp  imp/K_n per contact
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
                var delta = -(v + constraints.rows[r].bias) * constraints.rows[r].inv_K_imp
                var old_lambda = constraints.rows[r].lambda_val
                constraints.rows[r].lambda_val = constraints.rows[r].lambda_val + delta

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

                    var delta = -(v_limit + constraints.rows[r].bias) * constraints.rows[r].inv_K_imp
                    var old_lambda = constraints.rows[r].lambda_val
                    constraints.rows[r].lambda_val = constraints.rows[r].lambda_val + delta

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
                constraints.rows[r_t1].lambda_val = constraints.rows[r_t1].lambda_val + delta_t1

                # Tangent 2
                var v_t2: Scalar[DTYPE] = 0
                for i in range(NV):
                    v_t2 += constraints.J[r_t2 * NV + i] * qvel[i]
                var delta_t2 = -v_t2 / constraints.rows[r_t2].K
                var old_t2 = constraints.rows[r_t2].lambda_val
                constraints.rows[r_t2].lambda_val = constraints.rows[r_t2].lambda_val + delta_t2

                # Coulomb cone clamping: |lambda_t| <= mu * lambda_n
                var t_mag = sqrt(
                    constraints.rows[r_t1].lambda_val * constraints.rows[r_t1].lambda_val
                    + constraints.rows[r_t2].lambda_val * constraints.rows[r_t2].lambda_val
                )
                if t_mag > max_friction:
                    var scale = max_friction / t_mag
                    constraints.rows[r_t1].lambda_val = constraints.rows[r_t1].lambda_val * scale
                    constraints.rows[r_t2].lambda_val = constraints.rows[r_t2].lambda_val * scale

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

        NOTE: GPU path is unchanged — still uses inline setup. GPU refactor
        is deferred to a follow-up.
        """

        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        var contact_tid = Int(thread_idx.y)
        var valid_env = env < BATCH

        # Workspace pointers
        comptime qvel_idx = ws_qvel_pred_offset[NV, NBODY]()
        comptime M_inv_idx = ws_m_inv_offset[NV, NBODY]()
        comptime solver_idx = ws_solver_offset[NV, NBODY]()

        comptime MC = _max_one[MAX_CONTACTS]()

        # Solver workspace layout: 23 * MC + 6 * MC * NV floats
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
        comptime ws_lambda_t1 = solver_idx + 11 * MC
        comptime ws_lambda_t2 = solver_idx + 12 * MC
        comptime ws_K_t1 = solver_idx + 13 * MC
        comptime ws_K_t2 = solver_idx + 14 * MC
        comptime ws_t1x = solver_idx + 15 * MC
        comptime ws_t1y = solver_idx + 16 * MC
        comptime ws_t1z = solver_idx + 17 * MC
        comptime ws_t2x = solver_idx + 18 * MC
        comptime ws_t2y = solver_idx + 19 * MC
        comptime ws_t2z = solver_idx + 20 * MC
        comptime ws_J_n = solver_idx + 21 * MC
        comptime ws_J_t1 = solver_idx + 21 * MC + MC * NV
        comptime ws_J_t2 = solver_idx + 21 * MC + 2 * MC * NV
        comptime ws_MinvJn = solver_idx + 21 * MC + 3 * MC * NV
        comptime ws_MinvJt1 = solver_idx + 21 * MC + 4 * MC * NV
        comptime ws_MinvJt2 = solver_idx + 21 * MC + 5 * MC * NV
        comptime ws_pos_bias = solver_idx + 21 * MC + 6 * MC * NV
        comptime ws_inv_K_imp = solver_idx + 22 * MC + 6 * MC * NV

        # === PARALLEL: Initialize workspace (each thread handles one slot) ===
        if valid_env:
            workspace[env, ws_lambda_n + contact_tid] = 0
            workspace[env, ws_K_n + contact_tid] = 1
            workspace[env, ws_c_dist + contact_tid] = 0
            workspace[env, ws_c_body + contact_tid] = 0
            workspace[env, ws_c_body_b + contact_tid] = -1
            workspace[env, ws_c_px + contact_tid] = 0
            workspace[env, ws_c_py + contact_tid] = 0
            workspace[env, ws_c_pz + contact_tid] = 0
            workspace[env, ws_c_nx + contact_tid] = 0
            workspace[env, ws_c_ny + contact_tid] = 0
            workspace[env, ws_c_nz + contact_tid] = 1
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

        # All threads read metadata independently
        var contacts_off = contacts_offset[NQ, NV, NBODY]()
        var meta_off = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
        var model_meta_off = model_metadata_offset[NBODY, NJOINT]()

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

            # Read solref/solimp for impedance precompute
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
        var J_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(V_SIZE):
            J_row[i] = 0

        if valid_env and contact_tid < nc:
            var c = contact_tid
            var c_off = contacts_off + c * CONTACT_SIZE
            var body = Int(
                rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_BODY_A])
            )
            var body_b = Int(
                rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_BODY_B])
            )
            var dist = rebind[Scalar[DTYPE]](
                state[env, c_off + CONTACT_IDX_DIST]
            )

            workspace[env, ws_c_dist + c] = dist
            workspace[env, ws_c_body + c] = Scalar[DTYPE](body)
            workspace[env, ws_c_body_b + c] = Scalar[DTYPE](body_b)

            if dist < Scalar[DTYPE](0):
                workspace[env, ws_c_px + c] = state[
                    env, c_off + CONTACT_IDX_POS_X
                ]
                workspace[env, ws_c_py + c] = state[
                    env, c_off + CONTACT_IDX_POS_Y
                ]
                workspace[env, ws_c_pz + c] = state[
                    env, c_off + CONTACT_IDX_POS_Z
                ]
                workspace[env, ws_c_nx + c] = state[env, c_off + CONTACT_IDX_NX]
                workspace[env, ws_c_ny + c] = state[env, c_off + CONTACT_IDX_NY]
                workspace[env, ws_c_nz + c] = state[env, c_off + CONTACT_IDX_NZ]

                # Compute normal Jacobian
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
                    body,
                    body_b,
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_px + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_py + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_pz + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_nx + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_ny + c]),
                    rebind[Scalar[DTYPE]](workspace[env, ws_c_nz + c]),
                    J_row,
                )

                # Store J_n, compute MinvJn and K_n
                var k: workspace.element_type = 0
                for i in range(NV):
                    workspace[env, ws_J_n + c * NV + i] = J_row[i]
                    var mi_j_sum: workspace.element_type = 0
                    for j_idx in range(NV):
                        mi_j_sum += (
                            workspace[env, M_inv_idx + i * NV + j_idx]
                            * J_row[j_idx]
                        )
                    workspace[env, ws_MinvJn + c * NV + i] = mi_j_sum
                    k += J_row[i] * mi_j_sum
                if k < Scalar[DTYPE](1e-10):
                    k = Scalar[DTYPE](1e-10)
                workspace[env, ws_K_n + c] = k

                # Precompute impedance coefficients
                var penetration = -dist
                var x = penetration / si_width
                if x > Scalar[DTYPE](1.0):
                    x = Scalar[DTYPE](1.0)
                var imp = si_dmin + (
                    Scalar[DTYPE](3.0) * x * x - Scalar[DTYPE](2.0) * x * x * x
                ) * (si_dmax - si_dmin)
                if imp < Scalar[DTYPE](0.2):
                    imp = Scalar[DTYPE](0.2)
                var pos_correction = imp * penetration * inv_tc_dr
                if pos_correction > Scalar[DTYPE](MAX_POS_CORRECTION_VEL):
                    pos_correction = Scalar[DTYPE](MAX_POS_CORRECTION_VEL)
                workspace[env, ws_pos_bias + c] = pos_correction
                workspace[env, ws_inv_K_imp + c] = imp / k

                # Store warm start lambda (applied by thread 0 after barrier)
                workspace[env, ws_lambda_n + c] = state[
                    env, c_off + CONTACT_IDX_IMPULSE_N
                ]

        # All threads must hit this barrier
        barrier()

        # === SEQUENTIAL: Warm start + PGS normal + joint limits (thread 0) ===
        if valid_env and contact_tid == 0:
            # Apply warm start: qvel += MinvJn * lambda
            for c in range(nc):
                if workspace[env, ws_c_dist + c] >= Scalar[DTYPE](0):
                    continue
                if workspace[env, ws_lambda_n + c] > Scalar[DTYPE](0):
                    for i in range(NV):
                        workspace[env, qvel_idx + i] += (
                            workspace[env, ws_MinvJn + c * NV + i]
                            * workspace[env, ws_lambda_n + c]
                        )

            # Phase 2: PGS normal iterations
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

            # Phase 2b: Joint limit constraints
            comptime MAX_LIMITS = _max_one[2 * NJOINT]()
            var limit_dof = InlineArray[Int, MAX_LIMITS](uninitialized=True)
            var limit_sign = InlineArray[Scalar[DTYPE], MAX_LIMITS](
                uninitialized=True
            )
            var limit_dist_arr = InlineArray[Scalar[DTYPE], MAX_LIMITS](
                uninitialized=True
            )
            var K_limit = InlineArray[Scalar[DTYPE], MAX_LIMITS](
                uninitialized=True
            )
            var lambda_limit = InlineArray[Scalar[DTYPE], MAX_LIMITS](
                uninitialized=True
            )
            for i in range(MAX_LIMITS):
                limit_dof[i] = 0
                limit_sign[i] = Scalar[DTYPE](0)
                limit_dist_arr[i] = Scalar[DTYPE](0)
                K_limit[i] = Scalar[DTYPE](1)
                lambda_limit[i] = Scalar[DTYPE](0)

            var num_limits = 0
            var qpos_off = 0
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
                var pos = rebind[Scalar[DTYPE]](state[env, qpos_off + qpos_adr])
                var dist_lo = pos - rmin
                if dist_lo < Scalar[DTYPE](0.01) and num_limits < MAX_LIMITS:
                    limit_dof[num_limits] = dof
                    limit_sign[num_limits] = Scalar[DTYPE](1)
                    limit_dist_arr[num_limits] = dist_lo
                    K_limit[num_limits] = rebind[Scalar[DTYPE]](
                        workspace[env, M_inv_idx + dof * NV + dof]
                    )
                    if K_limit[num_limits] < Scalar[DTYPE](1e-10):
                        K_limit[num_limits] = Scalar[DTYPE](1e-10)
                    num_limits += 1
                var dist_hi = rmax - pos
                if dist_hi < Scalar[DTYPE](0.01) and num_limits < MAX_LIMITS:
                    limit_dof[num_limits] = dof
                    limit_sign[num_limits] = Scalar[DTYPE](-1)
                    limit_dist_arr[num_limits] = dist_hi
                    K_limit[num_limits] = rebind[Scalar[DTYPE]](
                        workspace[env, M_inv_idx + dof * NV + dof]
                    )
                    if K_limit[num_limits] < Scalar[DTYPE](1e-10):
                        K_limit[num_limits] = Scalar[DTYPE](1e-10)
                    num_limits += 1

            if num_limits > 0:
                var lr_tc = rebind[Scalar[DTYPE]](
                    model[0, model_meta_off + MODEL_META_IDX_SOLREF_LIMIT_0]
                )
                var lr_dr = rebind[Scalar[DTYPE]](
                    model[0, model_meta_off + MODEL_META_IDX_SOLREF_LIMIT_1]
                )
                var li_dmin = rebind[Scalar[DTYPE]](
                    model[0, model_meta_off + MODEL_META_IDX_SOLIMP_LIMIT_0]
                )
                var li_dmax = rebind[Scalar[DTYPE]](
                    model[0, model_meta_off + MODEL_META_IDX_SOLIMP_LIMIT_1]
                )
                var li_width = rebind[Scalar[DTYPE]](
                    model[0, model_meta_off + MODEL_META_IDX_SOLIMP_LIMIT_2]
                )
                if li_width < Scalar[DTYPE](1e-6):
                    li_width = Scalar[DTYPE](1e-6)
                if li_dmax < Scalar[DTYPE](1e-4):
                    li_dmax = Scalar[DTYPE](1e-4)
                var l_inv_tc_dr = Scalar[DTYPE](1.0) / (lr_tc * lr_dr)
                var l_b_vel_coef = (
                    Scalar[DTYPE](2.0) * lr_dr * dt / (li_dmax * lr_tc)
                )
                var l_vel_factor = Scalar[DTYPE](1.0) - l_b_vel_coef

                var lim_pos_bias = InlineArray[Scalar[DTYPE], MAX_LIMITS](
                    uninitialized=True
                )
                var lim_inv_K_imp = InlineArray[Scalar[DTYPE], MAX_LIMITS](
                    uninitialized=True
                )
                comptime MINVJ_LIM_SIZE = _max_one[2 * NJOINT * NV]()
                var lim_MinvJ = InlineArray[Scalar[DTYPE], MINVJ_LIM_SIZE](
                    uninitialized=True
                )
                for l in range(num_limits):
                    var penetration = -limit_dist_arr[l]
                    if penetration < Scalar[DTYPE](0):
                        penetration = Scalar[DTYPE](0)
                    var x_lim = penetration / li_width
                    if x_lim > Scalar[DTYPE](1.0):
                        x_lim = Scalar[DTYPE](1.0)
                    var imp_lim = li_dmin + (
                        Scalar[DTYPE](3.0) * x_lim * x_lim
                        - Scalar[DTYPE](2.0) * x_lim * x_lim * x_lim
                    ) * (li_dmax - li_dmin)
                    if imp_lim < Scalar[DTYPE](0.2):
                        imp_lim = Scalar[DTYPE](0.2)
                    var lim_pos_corr = imp_lim * penetration * l_inv_tc_dr
                    if lim_pos_corr > Scalar[DTYPE](MAX_POS_CORRECTION_VEL):
                        lim_pos_corr = Scalar[DTYPE](MAX_POS_CORRECTION_VEL)
                    lim_pos_bias[l] = lim_pos_corr
                    lim_inv_K_imp[l] = imp_lim / K_limit[l]
                    var ldof = limit_dof[l]
                    var lsign = limit_sign[l]
                    for i in range(NV):
                        lim_MinvJ[l * NV + i] = (
                            rebind[Scalar[DTYPE]](
                                workspace[env, M_inv_idx + i * NV + ldof]
                            )
                            * lsign
                        )

                for _ in range(PGS_ITERATIONS):
                    var max_lim_delta: Scalar[DTYPE] = 0
                    for l in range(num_limits):
                        var v_limit = (
                            limit_sign[l]
                            * workspace[env, qvel_idx + limit_dof[l]]
                        )
                        var delta_l = (
                            -(v_limit * l_vel_factor - lim_pos_bias[l])
                            * lim_inv_K_imp[l]
                        )
                        var old_lam = lambda_limit[l]
                        lambda_limit[l] = lambda_limit[l] + rebind[
                            Scalar[DTYPE]
                        ](delta_l)
                        if lambda_limit[l] < Scalar[DTYPE](0):
                            lambda_limit[l] = Scalar[DTYPE](0)
                        var actual_l = lambda_limit[l] - old_lam
                        var abs_l = abs(actual_l)
                        if abs_l > max_lim_delta:
                            max_lim_delta = abs_l
                        for i in range(NV):
                            workspace[env, qvel_idx + i] += (
                                lim_MinvJ[l * NV + i] * actual_l
                            )
                    if max_lim_delta < Scalar[DTYPE](1e-4):
                        break

        # All threads must hit this barrier
        barrier()

        # === PARALLEL PHASE 3: Each thread precomputes tangent for one contact ===
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

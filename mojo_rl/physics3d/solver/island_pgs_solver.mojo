"""IslandPGSSolver — ConstraintSolver wrapper with per-island early termination.

Drop-in replacement for PGSSolver that adds island-aware convergence.  When
the constraint system decomposes into multiple independent sub-problems
(separate robots, non-interacting objects) each island can converge and be
frozen independently, reducing total PGS iterations significantly.

Usage
-----
    from mojo_rl.physics3d.solver import IslandPGSSolver
    from mojo_rl.physics3d.integrator import EulerIntegrator

    # Instead of EulerIntegrator[PGSSolver]:
    alias MyIntegrator = EulerIntegrator[IslandPGSSolver]

Single-island systems (the common case) fall back directly to PGSSolver.solve()
with no overhead beyond the O(MAX_ROWS*NV) island-detection scan.

GPU behaviour
-------------
Island detection runs fully on-GPU in solve_gpu().  Each environment uses
body-based union-find to partition contacts into islands.  Islands that
converge early (max |Δλ| < ISLAND_CONVERGE_EPS) are frozen and skipped in
subsequent PGS iterations, reducing total work for multi-body scenes.
"""

from std.math import sqrt, abs
from layout import (
    Layout,
    LayoutTensor,
)
from std.gpu import thread_idx, block_idx, block_dim, barrier
from ..types import Model, Data, ConeType, _max_one
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..constraints.constraint_data import ConstraintData
from ..traits.solver import ConstraintSolver
from ..dynamics.jacobian import (
    compute_contact_jacobian_row_gpu,
    compute_angular_jacobian_row_gpu,
)
from .pgs_solver import PGSSolver
from .qcqp import qcqp2, qcqp3, qcqp5, mj_qcqp2, mj_qcqp3, mj_qcqp5, cost_change
from .island_solver import solve_with_islands, ISLAND_CONVERGE_EPS
from .island_detection import MAX_ISLANDS
from std.gpu.host import DeviceContext
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
    qvel_offset,
)
from ..constraints.constraint_builder_gpu import (
    init_common_normal_workspace_gpu,
    precompute_contact_normal_gpu,
    warmstart_normals_gpu,
    detect_and_solve_limits_gpu,
    build_and_solve_equality_gpu,
    build_and_solve_tendon_gpu,
)

# PGS solver parameters (mirror pgs_solver.mojo)
comptime PGS_ITERATIONS: Int = 100
comptime FRICTION_K_MIN: Float64 = 1e-6


struct IslandPGSSolver(ConstraintSolver):
    """PGS solver with per-island early termination.

    Implements the ConstraintSolver trait so it can be used anywhere
    PGSSolver is accepted (EulerIntegrator, RK4Integrator, etc.).

    CPU path: detect_islands() partitions constraint rows; islands that
    converge early are frozen and skipped in subsequent PGS iterations.

    GPU path: body union-find partitions contacts into islands on-GPU.
    Islands that converge early (max |Δλ| < ISLAND_CONVERGE_EPS) are frozen
    and skipped in subsequent iterations.
    """

    comptime NEEDS_M_INV: Bool = True

    @staticmethod
    fn solver_workspace_size[NV: Int, MAX_CONTACTS: Int]() -> Int:
        """Same workspace footprint as PGSSolver."""
        return PGSSolver.solver_workspace_size[NV, MAX_CONTACTS]()

    @staticmethod
    fn solver_threads[
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
    ]() -> Int:
        """Same thread count as PGSSolver."""
        return PGSSolver.solver_threads[NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()

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
        """Solve constraints with island-aware PGS on CPU.

        Detects constraint islands and applies per-island early termination.
        Falls back to PGSSolver.solve() when there is only one island.
        """
        solve_with_islands[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            MAX_ROWS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
            NSITE,
        ](model, data, M_inv, constraints, qacc, dt)

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
        """Solve contact constraints using island-aware PGS on GPU.

        Identical to PGSSolver.solve_gpu with the following additions:
        - Body union-find partitions contacts into islands (Phase 1, thread 0)
        - Phase 1 normal PGS loop skips converged islands
        - Phase 3 coupled PGS loop skips converged islands
        - Each phase tracks per-island max |Δλ| and freezes converged islands

        Uses thread_x for environment index, thread_y for contact index.
        Precompute phases are parallelized across contacts.
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

        # Friction workspace offsets
        comptime fws = solver_idx + 13 * MC + 2 * MC * NV
        comptime ws_lf = fws + 0 * MC
        comptime ws_kf = fws + 5 * MC
        comptime ws_df = fws + 10 * MC
        comptime ws_fc = fws + 25 * MC
        comptime ws_cd = fws + 30 * MC
        comptime ws_rf = fws + 31 * MC
        comptime ws_bf = fws + 36 * MC
        comptime ws_jf = fws + 41 * MC
        comptime ws_mj = fws + 41 * MC + 5 * MC * NV
        # Pyramidal-only workspace offsets
        comptime ws_le_neg = fws + 41 * MC + 10 * MC * NV
        comptime ws_cnt = ws_le_neg + 5 * MC
        comptime ws_kep = ws_cnt + 5 * MC
        comptime ws_ken = ws_kep + 5 * MC
        comptime ws_re = ws_ken + 5 * MC

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
        var K_spring: Scalar[DTYPE] = 0
        var B_damp: Scalar[DTYPE] = 0
        var si_dmin: Scalar[DTYPE] = 0
        var si_dmax: Scalar[DTYPE] = 0
        var si_width: Scalar[DTYPE] = 1
        var si_midpoint: Scalar[DTYPE] = Scalar[DTYPE](0.5)
        var si_power: Scalar[DTYPE] = Scalar[DTYPE](2.0)

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

        # Island tracking arrays (register-local, thread 0 populates them in Phase 1)
        comptime MC_safe = _max_one[MC]()
        var contact_island = InlineArray[Int, _max_one[MC_safe]()](fill=-1)
        var island_converged = InlineArray[Int, MAX_ISLANDS](fill=0)
        var num_islands = 0
        var num_converged = 0

        # === SEQUENTIAL: Warm start + island detection + PGS normal + joint limits (thread 0) ===
        if valid_env and contact_tid == 0:
            # ---- Body union-find: assign each contact to an island ----
            var uf_parent = InlineArray[Int, _max_one[NBODY]()](
                uninitialized=True
            )
            for b in range(NBODY):
                uf_parent[b] = b

            for c in range(nc):
                if workspace[env, ws_c_dist + c] >= Scalar[DTYPE](0):
                    continue
                var ba = Int(workspace[env, ws_c_body + c])
                var bb = Int(workspace[env, ws_c_body_b + c])
                # path-halving find for ba
                var ra = ba
                while uf_parent[ra] != ra:
                    var inner = uf_parent[ra]
                    var gp = uf_parent[inner]
                    uf_parent[ra] = gp
                    ra = gp
                # path-halving find for bb
                var rb = bb
                while uf_parent[rb] != rb:
                    var inner = uf_parent[rb]
                    var gp = uf_parent[inner]
                    uf_parent[rb] = gp
                    rb = gp
                if ra != rb:
                    uf_parent[rb] = ra

            var root_island = InlineArray[Int, _max_one[NBODY]()](fill=-1)
            for c in range(nc):
                if workspace[env, ws_c_dist + c] >= Scalar[DTYPE](0):
                    continue
                var ba = Int(workspace[env, ws_c_body + c])
                var root_b = ba
                while uf_parent[root_b] != root_b:
                    root_b = uf_parent[root_b]
                if root_island[root_b] < 0:
                    var iid = num_islands
                    if iid >= MAX_ISLANDS:
                        iid = MAX_ISLANDS - 1
                    root_island[root_b] = iid
                    num_islands += 1
                var ciid = root_island[root_b]
                if ciid >= MAX_ISLANDS:
                    ciid = MAX_ISLANDS - 1
                contact_island[c] = ciid

            if num_islands > MAX_ISLANDS:
                num_islands = MAX_ISLANDS

            warmstart_normals_gpu[
                DTYPE,
                NV,
                NBODY,
                MAX_CONTACTS,
                WS_SIZE,
                BATCH,
            ](env, nc, workspace)

            # PGS normal iterations (acceleration-level) with per-island early termination
            var eps_gpu = Scalar[DTYPE](ISLAND_CONVERGE_EPS)
            for _ in range(PGS_ITERATIONS):
                if num_converged >= num_islands:
                    break
                var island_max_delta_n = InlineArray[
                    Scalar[DTYPE], MAX_ISLANDS
                ](fill=Scalar[DTYPE](0))
                for c in range(nc):
                    if workspace[env, ws_c_dist + c] >= Scalar[DTYPE](0):
                        continue
                    var iid = contact_island[c]
                    if iid >= 0 and island_converged[iid] == 1:
                        continue
                    var a_n: workspace.element_type = 0
                    for i in range(NV):
                        a_n += (
                            workspace[env, ws_J_n + c * NV + i]
                            * workspace[env, qacc_idx + i]
                        )
                    var R_n = Scalar[DTYPE](1.0) / rebind[Scalar[DTYPE]](
                        workspace[env, ws_inv_K_imp + c]
                    ) - rebind[Scalar[DTYPE]](workspace[env, ws_K_n + c])
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
                    var abs_delta = abs(rebind[Scalar[DTYPE]](actual_delta))
                    if iid >= 0 and abs_delta > island_max_delta_n[iid]:
                        island_max_delta_n[iid] = abs_delta
                    for i in range(NV):
                        workspace[env, qacc_idx + i] += (
                            workspace[env, ws_MinvJn + c * NV + i]
                            * actual_delta
                        )
                for iid in range(num_islands):
                    if island_converged[iid] == 0:
                        if island_max_delta_n[iid] < eps_gpu:
                            island_converged[iid] = 1
                            num_converged += 1

            # Reset island convergence flags for Phase 3 coupled loop
            for iid in range(num_islands):
                island_converged[iid] = 0
            num_converged = 0

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
                NGEOM,
                MAX_EQUALITY,
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

            # Tendon equality constraints
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
                    mu_slide = Scalar[DTYPE](0.5)  # fallback
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
                    # Tangent basis (MuJoCo mju_makeFrame with capsule axis hint)
                    var hint_x = rebind[Scalar[DTYPE]](
                        state[env, c_off + CONTACT_IDX_FRAME_T1_X]
                    )
                    var hint_y = rebind[Scalar[DTYPE]](
                        state[env, c_off + CONTACT_IDX_FRAME_T1_Y]
                    )
                    var hint_z = rebind[Scalar[DTYPE]](
                        state[env, c_off + CONTACT_IDX_FRAME_T1_Z]
                    )
                    var hint_len_sq = (
                        hint_x * hint_x + hint_y * hint_y + hint_z * hint_z
                    )

                    # If no hint (non-capsule), use MuJoCo default
                    if hint_len_sq < Scalar[DTYPE](0.25):
                        hint_x = Scalar[DTYPE](0)
                        if ny < Scalar[DTYPE](0.5) and ny > Scalar[DTYPE](-0.5):
                            hint_y = Scalar[DTYPE](1)
                            hint_z = Scalar[DTYPE](0)
                        else:
                            hint_y = Scalar[DTYPE](0)
                            hint_z = Scalar[DTYPE](1)

                    # Gram-Schmidt: orthogonalize hint against normal
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

                    comptime if CONE_TYPE == ConeType.PYRAMIDAL:
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
            # with per-island early termination
            var eps_gpu2 = Scalar[DTYPE](ISLAND_CONVERGE_EPS)
            for _ in range(PGS_ITERATIONS):
                if num_converged >= num_islands:
                    break
                var island_max_delta_c = InlineArray[
                    Scalar[DTYPE], MAX_ISLANDS
                ](fill=Scalar[DTYPE](0))
                # --- Normal constraints PGS update ---
                for c in range(nc):
                    if workspace[env, ws_c_dist + c] >= Scalar[DTYPE](0):
                        continue
                    var iid = contact_island[c]
                    if iid >= 0 and island_converged[iid] == 1:
                        continue
                    var a_n: workspace.element_type = 0
                    for i in range(NV):
                        a_n += (
                            workspace[env, ws_J_n + c * NV + i]
                            * workspace[env, qacc_idx + i]
                        )
                    var R_n = Scalar[DTYPE](1.0) / rebind[Scalar[DTYPE]](
                        workspace[env, ws_inv_K_imp + c]
                    ) - rebind[Scalar[DTYPE]](workspace[env, ws_K_n + c])
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
                    var abs_n = abs(rebind[Scalar[DTYPE]](actual_n))
                    if iid >= 0 and abs_n > island_max_delta_c[iid]:
                        island_max_delta_c[iid] = abs_n
                    for i in range(NV):
                        workspace[env, qacc_idx + i] += (
                            workspace[env, ws_MinvJn + c * NV + i] * actual_n
                        )

                # --- Friction constraints PGS update ---
                for c in range(nc):
                    var iid = contact_island[c]
                    if iid >= 0 and island_converged[iid] == 1:
                        continue
                    if workspace[env, ws_lambda_n + c] <= Scalar[DTYPE](0):
                        # Zero friction when normal force is zero
                        var condim_z = Int(workspace[env, ws_cd + c])
                        var num_fric_z = 2
                        if condim_z >= 4:
                            num_fric_z = 3
                        if condim_z >= 6:
                            num_fric_z = 5
                        for d in range(num_fric_z):
                            comptime if CONE_TYPE == ConeType.PYRAMIDAL:
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

                    comptime if CONE_TYPE == ConeType.PYRAMIDAL:
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
                        # === ELLIPTIC CONE: MuJoCo-style block update ===
                        # Ray update + QCQP with AR submatrix + costChange
                        var dim = 1 + num_fric

                        # Build block AR matrix on-the-fly from J/MinvJ
                        var AR = InlineArray[Scalar[DTYPE], 36](
                            fill=Scalar[DTYPE](0)
                        )
                        var R_n_val = Scalar[DTYPE](1.0) / rebind[
                            Scalar[DTYPE]
                        ](workspace[env, ws_inv_K_imp + c]) - rebind[
                            Scalar[DTYPE]
                        ](
                            workspace[env, ws_K_n + c]
                        )
                        AR[0] = (
                            rebind[Scalar[DTYPE]](workspace[env, ws_K_n + c])
                            + R_n_val
                        )

                        for d1 in range(num_fric):
                            # Normal-friction cross: J_n @ MinvJ_f[d1]
                            var cross: Scalar[DTYPE] = 0
                            for i in range(NV):
                                cross += rebind[Scalar[DTYPE]](
                                    workspace[env, ws_J_n + c * NV + i]
                                ) * rebind[Scalar[DTYPE]](
                                    workspace[
                                        env, ws_mj + d1 * MC * NV + c * NV + i
                                    ]
                                )
                            AR[(d1 + 1)] = cross
                            AR[(d1 + 1) * dim] = cross

                            for d2 in range(num_fric):
                                var ff: Scalar[DTYPE] = 0
                                for i in range(NV):
                                    ff += rebind[Scalar[DTYPE]](
                                        workspace[
                                            env,
                                            ws_jf + d1 * MC * NV + c * NV + i,
                                        ]
                                    ) * rebind[Scalar[DTYPE]](
                                        workspace[
                                            env,
                                            ws_mj + d2 * MC * NV + c * NV + i,
                                        ]
                                    )
                                if d1 == d2:
                                    ff += rebind[Scalar[DTYPE]](
                                        workspace[env, ws_rf + d1 * MC + c]
                                    )
                                AR[(d1 + 1) * dim + (d2 + 1)] = ff

                        # Compute block residual
                        var block_res = InlineArray[Scalar[DTYPE], 6](
                            fill=Scalar[DTYPE](0)
                        )
                        var a_n_res: Scalar[DTYPE] = 0
                        for i in range(NV):
                            a_n_res += rebind[Scalar[DTYPE]](
                                workspace[env, ws_J_n + c * NV + i]
                            ) * rebind[Scalar[DTYPE]](
                                workspace[env, qacc_idx + i]
                            )
                        block_res[0] = (
                            a_n_res
                            + rebind[Scalar[DTYPE]](
                                workspace[env, ws_pos_bias + c]
                            )
                            + R_n_val
                            * rebind[Scalar[DTYPE]](
                                workspace[env, ws_lambda_n + c]
                            )
                        )
                        for d in range(num_fric):
                            var a_f_res: Scalar[DTYPE] = 0
                            for i in range(NV):
                                a_f_res += rebind[Scalar[DTYPE]](
                                    workspace[
                                        env, ws_jf + d * MC * NV + c * NV + i
                                    ]
                                ) * rebind[Scalar[DTYPE]](
                                    workspace[env, qacc_idx + i]
                                )
                            var R_f_d = rebind[Scalar[DTYPE]](
                                workspace[env, ws_rf + d * MC + c]
                            )
                            block_res[1 + d] = (
                                a_f_res
                                + rebind[Scalar[DTYPE]](
                                    workspace[env, ws_bf + d * MC + c]
                                )
                                + R_f_d
                                * rebind[Scalar[DTYPE]](
                                    workspace[env, ws_lf + d * MC + c]
                                )
                            )

                        # Save old forces
                        var oldforce = InlineArray[Scalar[DTYPE], 6](
                            fill=Scalar[DTYPE](0)
                        )
                        oldforce[0] = rebind[Scalar[DTYPE]](
                            workspace[env, ws_lambda_n + c]
                        )
                        for d in range(num_fric):
                            oldforce[1 + d] = rebind[Scalar[DTYPE]](
                                workspace[env, ws_lf + d * MC + c]
                            )

                        var ARinv0: Scalar[DTYPE] = 0
                        if AR[0] > Scalar[DTYPE](1e-10):
                            ARinv0 = Scalar[DTYPE](1.0) / AR[0]

                        # --- Ray update ---
                        if rebind[Scalar[DTYPE]](
                            workspace[env, ws_lambda_n + c]
                        ) < Scalar[DTYPE](1e-10):
                            workspace[env, ws_lambda_n + c] = (
                                rebind[Scalar[DTYPE]](
                                    workspace[env, ws_lambda_n + c]
                                )
                                - block_res[0] * ARinv0
                            )
                            if workspace[env, ws_lambda_n + c] < Scalar[DTYPE](
                                0
                            ):
                                workspace[env, ws_lambda_n + c] = Scalar[DTYPE](
                                    0
                                )
                            for d in range(num_fric):
                                workspace[env, ws_lf + d * MC + c] = Scalar[
                                    DTYPE
                                ](0)
                        else:
                            var v = InlineArray[Scalar[DTYPE], 6](
                                fill=Scalar[DTYPE](0)
                            )
                            v[0] = rebind[Scalar[DTYPE]](
                                workspace[env, ws_lambda_n + c]
                            )
                            for d in range(num_fric):
                                v[1 + d] = rebind[Scalar[DTYPE]](
                                    workspace[env, ws_lf + d * MC + c]
                                )
                            var denom: Scalar[DTYPE] = 0
                            for bi in range(dim):
                                for bj in range(dim):
                                    denom += v[bi] * AR[bi * dim + bj] * v[bj]
                            if denom >= Scalar[DTYPE](1e-10):
                                var vdotr: Scalar[DTYPE] = 0
                                for bi in range(dim):
                                    vdotr += v[bi] * block_res[bi]
                                var x = -vdotr / denom
                                if rebind[Scalar[DTYPE]](
                                    workspace[env, ws_lambda_n + c]
                                ) + x * v[0] < Scalar[DTYPE](0):
                                    x = (
                                        -rebind[Scalar[DTYPE]](
                                            workspace[env, ws_lambda_n + c]
                                        )
                                        / v[0]
                                    )
                                workspace[env, ws_lambda_n + c] = (
                                    rebind[Scalar[DTYPE]](
                                        workspace[env, ws_lambda_n + c]
                                    )
                                    + x * v[0]
                                )
                                for d in range(num_fric):
                                    workspace[env, ws_lf + d * MC + c] = (
                                        rebind[Scalar[DTYPE]](
                                            workspace[env, ws_lf + d * MC + c]
                                        )
                                        + x * v[1 + d]
                                    )

                        # --- QCQP friction update ---
                        var fn_val = rebind[Scalar[DTYPE]](
                            workspace[env, ws_lambda_n + c]
                        )
                        if fn_val >= Scalar[DTYPE](1e-10) and num_fric > 0:
                            var Ac = InlineArray[Scalar[DTYPE], 25](
                                fill=Scalar[DTYPE](0)
                            )
                            var bc_arr = InlineArray[Scalar[DTYPE], 5](
                                fill=Scalar[DTYPE](0)
                            )
                            for j in range(num_fric):
                                for j2 in range(num_fric):
                                    Ac[j * num_fric + j2] = AR[
                                        (1 + j) * dim + (1 + j2)
                                    ]
                                bc_arr[j] = block_res[1 + j]
                                for j2 in range(num_fric):
                                    bc_arr[j] -= (
                                        Ac[j * num_fric + j2] * oldforce[1 + j2]
                                    )
                                bc_arr[j] += AR[(1 + j) * dim + 0] * (
                                    fn_val - oldforce[0]
                                )

                            var mu_arr = InlineArray[Scalar[DTYPE], 5](
                                fill=Scalar[DTYPE](0)
                            )
                            for d in range(num_fric):
                                mu_arr[d] = rebind[Scalar[DTYPE]](
                                    workspace[env, ws_fc + d * MC + c]
                                )

                            var flg_active = False
                            if num_fric == 2:
                                var A2 = InlineArray[Scalar[DTYPE], 4](
                                    fill=Scalar[DTYPE](0)
                                )
                                var b2 = InlineArray[Scalar[DTYPE], 2](
                                    fill=Scalar[DTYPE](0)
                                )
                                var d2 = InlineArray[Scalar[DTYPE], 2](
                                    fill=Scalar[DTYPE](0)
                                )
                                for ii in range(2):
                                    b2[ii] = bc_arr[ii]
                                    d2[ii] = mu_arr[ii]
                                    for jj in range(2):
                                        A2[ii * 2 + jj] = Ac[ii * num_fric + jj]
                                var r0: Scalar[DTYPE] = 0
                                var r1: Scalar[DTYPE] = 0
                                flg_active = mj_qcqp2[DTYPE](
                                    r0, r1, A2, b2, d2, fn_val
                                )
                                workspace[env, ws_lf + 0 * MC + c] = r0
                                workspace[env, ws_lf + 1 * MC + c] = r1
                            elif num_fric == 3:
                                var A3 = InlineArray[Scalar[DTYPE], 9](
                                    fill=Scalar[DTYPE](0)
                                )
                                var b3 = InlineArray[Scalar[DTYPE], 3](
                                    fill=Scalar[DTYPE](0)
                                )
                                var d3 = InlineArray[Scalar[DTYPE], 3](
                                    fill=Scalar[DTYPE](0)
                                )
                                for ii in range(3):
                                    b3[ii] = bc_arr[ii]
                                    d3[ii] = mu_arr[ii]
                                    for jj in range(3):
                                        A3[ii * 3 + jj] = Ac[ii * num_fric + jj]
                                var r0: Scalar[DTYPE] = 0
                                var r1: Scalar[DTYPE] = 0
                                var r2: Scalar[DTYPE] = 0
                                flg_active = mj_qcqp3[DTYPE](
                                    r0, r1, r2, A3, b3, d3, fn_val
                                )
                                workspace[env, ws_lf + 0 * MC + c] = r0
                                workspace[env, ws_lf + 1 * MC + c] = r1
                                workspace[env, ws_lf + 2 * MC + c] = r2
                            elif num_fric == 5:
                                var A5 = InlineArray[Scalar[DTYPE], 25](
                                    fill=Scalar[DTYPE](0)
                                )
                                var b5 = InlineArray[Scalar[DTYPE], 5](
                                    fill=Scalar[DTYPE](0)
                                )
                                var d5 = InlineArray[Scalar[DTYPE], 5](
                                    fill=Scalar[DTYPE](0)
                                )
                                for ii in range(5):
                                    b5[ii] = bc_arr[ii]
                                    d5[ii] = mu_arr[ii]
                                    for jj in range(5):
                                        A5[ii * 5 + jj] = Ac[ii * num_fric + jj]
                                var res5 = InlineArray[Scalar[DTYPE], 5](
                                    fill=Scalar[DTYPE](0)
                                )
                                flg_active = mj_qcqp5[DTYPE](
                                    res5, A5, b5, d5, fn_val
                                )
                                for d in range(5):
                                    workspace[env, ws_lf + d * MC + c] = res5[d]

                            # Rescale to exact ellipsoid if constrained
                            if flg_active:
                                var s: Scalar[DTYPE] = 0
                                for d in range(num_fric):
                                    var fv = rebind[Scalar[DTYPE]](
                                        workspace[env, ws_lf + d * MC + c]
                                    )
                                    var mu_d = mu_arr[d]
                                    if mu_d > Scalar[DTYPE](1e-10):
                                        s += fv * fv / (mu_d * mu_d)
                                if s > Scalar[DTYPE](1e-10):
                                    var scale = sqrt(fn_val * fn_val / s)
                                    for d in range(num_fric):
                                        workspace[env, ws_lf + d * MC + c] = (
                                            rebind[Scalar[DTYPE]](
                                                workspace[
                                                    env, ws_lf + d * MC + c
                                                ]
                                            )
                                            * scale
                                        )

                        # --- Cost descent check ---
                        var cost_val: Scalar[DTYPE] = 0
                        for bi in range(dim):
                            var new_i: Scalar[DTYPE]
                            var old_i: Scalar[DTYPE]
                            if bi == 0:
                                new_i = rebind[Scalar[DTYPE]](
                                    workspace[env, ws_lambda_n + c]
                                )
                                old_i = oldforce[0]
                            else:
                                new_i = rebind[Scalar[DTYPE]](
                                    workspace[env, ws_lf + (bi - 1) * MC + c]
                                )
                                old_i = oldforce[bi]
                            var delta_i = new_i - old_i
                            cost_val += delta_i * block_res[bi]
                            for bj in range(dim):
                                var new_j: Scalar[DTYPE]
                                var old_j: Scalar[DTYPE]
                                if bj == 0:
                                    new_j = rebind[Scalar[DTYPE]](
                                        workspace[env, ws_lambda_n + c]
                                    )
                                    old_j = oldforce[0]
                                else:
                                    new_j = rebind[Scalar[DTYPE]](
                                        workspace[
                                            env, ws_lf + (bj - 1) * MC + c
                                        ]
                                    )
                                    old_j = oldforce[bj]
                                var delta_j = new_j - old_j
                                cost_val += (
                                    Scalar[DTYPE](0.5)
                                    * delta_i
                                    * AR[bi * dim + bj]
                                    * delta_j
                                )

                        if cost_val > Scalar[DTYPE](1e-10):
                            # Revert
                            workspace[env, ws_lambda_n + c] = oldforce[0]
                            for d in range(num_fric):
                                workspace[env, ws_lf + d * MC + c] = oldforce[
                                    1 + d
                                ]

                        # Apply delta to qacc
                        var actual_n2 = (
                            rebind[Scalar[DTYPE]](
                                workspace[env, ws_lambda_n + c]
                            )
                            - oldforce[0]
                        )
                        if actual_n2 != Scalar[DTYPE](0):
                            for i in range(NV):
                                workspace[env, qacc_idx + i] += (
                                    workspace[env, ws_MinvJn + c * NV + i]
                                    * actual_n2
                                )
                        for d in range(num_fric):
                            var actual_f = (
                                rebind[Scalar[DTYPE]](
                                    workspace[env, ws_lf + d * MC + c]
                                )
                                - oldforce[1 + d]
                            )
                            if actual_f != Scalar[DTYPE](0):
                                for i in range(NV):
                                    workspace[env, qacc_idx + i] += (
                                        workspace[
                                            env,
                                            ws_mj + d * MC * NV + c * NV + i,
                                        ]
                                        * actual_f
                                    )

                # Per-island convergence check for coupled loop
                for iid in range(num_islands):
                    if island_converged[iid] == 0:
                        if island_max_delta_c[iid] < eps_gpu2:
                            island_converged[iid] = 1
                            num_converged += 1

            # Store impulses back to state buffer for warm-starting
            comptime if CONE_TYPE == ConeType.PYRAMIDAL:
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

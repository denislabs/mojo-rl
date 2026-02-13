from layout import Layout, LayoutTensor
from math import sqrt
from ..types import _max_one
from ..dynamics.jacobian import (
    compute_contact_jacobian_row_gpu,
)
from ..gpu.constants import (
    ws_qacc_constrained_offset,
    ws_m_inv_offset,
    ws_solver_offset,
    CONTACT_SIZE,
    CONTACT_IDX_FORCE_N,
    CONTACT_IDX_FORCE_T1,
    CONTACT_IDX_FORCE_T2,
)

# Coupled PGS iterations (normals + friction together, MuJoCo-style)
comptime COUPLED_PGS_ITERATIONS_GPU: Int = 50
# Minimum K for friction tangent rows — below this, direction is degenerate
comptime FRICTION_K_MIN: Float64 = 1e-6

# =============================================================================
# Shared friction solver (PGS) - GPU
# (CPU friction is now handled inline via ConstraintData in each solver)
# =============================================================================


@always_inline
fn _solve_friction_pgs_gpu[
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
    FRICTION_WS_OFFSET: Int,
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    workspace: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ],
    # Offsets into workspace (absolute from env row start)
    nc: Int,
    friction_coef: Scalar[DTYPE],
    contacts_off: Int,
):
    """Friction solver using PGS on GPU (shared by CG and Newton solvers).

    Derives all writable pointers from workspace.ptr (preserves mutable origin).
    Contact data is read-only, friction data is read-write.
    """

    comptime qacc_idx = ws_qacc_constrained_offset[NV, NBODY]()

    comptime contact_ws_off = ws_solver_offset[NV, NBODY]()
    comptime friction_ws_off = contact_ws_off + FRICTION_WS_OFFSET
    comptime MC = _max_one[MAX_CONTACTS]()

    # Derive ALL pointers from workspace.ptr for mutable writes

    comptime M_inv = ws_m_inv_offset[NV, NBODY]()

    # Contact block (read-only)
    comptime ws_lambda_n = contact_ws_off + 0 * MC
    comptime ws_c_body = contact_ws_off + 3 * MC
    comptime ws_c_body_b = contact_ws_off + 4 * MC
    comptime ws_c_px = contact_ws_off + 5 * MC
    comptime ws_c_py = contact_ws_off + 6 * MC
    comptime ws_c_pz = contact_ws_off + 7 * MC
    comptime ws_c_nx = contact_ws_off + 8 * MC
    comptime ws_c_ny = contact_ws_off + 9 * MC
    comptime ws_c_nz = contact_ws_off + 10 * MC

    # Friction block (read-write)
    comptime lt1 = friction_ws_off + 0 * MC
    comptime lt2 = friction_ws_off + 1 * MC
    comptime kt1 = friction_ws_off + 2 * MC
    comptime kt2 = friction_ws_off + 3 * MC
    comptime _t1x = friction_ws_off + 4 * MC
    comptime _t1y = friction_ws_off + 5 * MC
    comptime _t1z = friction_ws_off + 6 * MC
    comptime _t2x = friction_ws_off + 7 * MC
    comptime _t2y = friction_ws_off + 8 * MC
    comptime _t2z = friction_ws_off + 9 * MC
    # Cached Jacobians and precomputed M_inv @ J^T (4 * MC * NV)
    comptime ws_J_t1 = friction_ws_off + 10 * MC
    comptime ws_J_t2 = friction_ws_off + 10 * MC + MC * NV
    comptime ws_MinvJt1 = friction_ws_off + 10 * MC + 2 * MC * NV
    comptime ws_MinvJt2 = friction_ws_off + 10 * MC + 3 * MC * NV

    # Initialize friction workspace
    for i in range(MC):
        workspace[env, lt1 + i] = Scalar[DTYPE](0)
        workspace[env, lt2 + i] = Scalar[DTYPE](0)
        workspace[env, kt1 + i] = Scalar[DTYPE](1)
        workspace[env, kt2 + i] = Scalar[DTYPE](1)
        workspace[env, _t1x + i] = Scalar[DTYPE](0)
        workspace[env, _t1y + i] = Scalar[DTYPE](0)
        workspace[env, _t1z + i] = Scalar[DTYPE](0)
        workspace[env, _t2x + i] = Scalar[DTYPE](0)
        workspace[env, _t2y + i] = Scalar[DTYPE](0)
        workspace[env, _t2z + i] = Scalar[DTYPE](0)

    var J_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

    # Precompute tangent basis and K_t for active contacts
    for c in range(nc):
        if workspace[env, ws_lambda_n + c] <= Scalar[DTYPE](0):
            continue

        var nx = workspace[env, ws_c_nx + c]
        var ny = workspace[env, ws_c_ny + c]
        var nz = workspace[env, ws_c_nz + c]

        if abs(nx) < Scalar[DTYPE](0.9):
            workspace[env, _t1x + c] = Scalar[DTYPE](0)
            workspace[env, _t1y + c] = -nz
            workspace[env, _t1z + c] = ny
        else:
            workspace[env, _t1x + c] = nz
            workspace[env, _t1y + c] = Scalar[DTYPE](0)
            workspace[env, _t1z + c] = -nx

        var t1_mag = sqrt(
            workspace[env, _t1x + c] * workspace[env, _t1x + c]
            + workspace[env, _t1y + c] * workspace[env, _t1y + c]
            + workspace[env, _t1z + c] * workspace[env, _t1z + c]
        )
        if t1_mag > Scalar[DTYPE](1e-10):
            workspace[env, _t1x + c] = workspace[env, _t1x + c] / t1_mag
            workspace[env, _t1y + c] = workspace[env, _t1y + c] / t1_mag
            workspace[env, _t1z + c] = workspace[env, _t1z + c] / t1_mag

        workspace[env, _t2x + c] = (
            ny * workspace[env, _t1z + c] - nz * workspace[env, _t1y + c]
        )
        workspace[env, _t2y + c] = (
            nz * workspace[env, _t1x + c] - nx * workspace[env, _t1z + c]
        )
        workspace[env, _t2z + c] = (
            nx * workspace[env, _t1y + c] - ny * workspace[env, _t1x + c]
        )

        # Compute K_t1, cache J_t1 and MinvJt1
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
            rebind[Scalar[DTYPE]](workspace[env, _t1x + c]),
            rebind[Scalar[DTYPE]](workspace[env, _t1y + c]),
            rebind[Scalar[DTYPE]](workspace[env, _t1z + c]),
            J_row,
        )

        var k1: workspace.element_type = 0
        for i in range(NV):
            workspace[env, ws_J_t1 + c * NV + i] = J_row[i]
            var mi_j_sum: workspace.element_type = 0
            for j_idx in range(NV):
                mi_j_sum += (
                    workspace[env, M_inv + i * NV + j_idx] * J_row[j_idx]
                )
            workspace[env, ws_MinvJt1 + c * NV + i] = mi_j_sum
            k1 += J_row[i] * mi_j_sum
        if k1 < Scalar[DTYPE](1e-10):
            k1 = Scalar[DTYPE](1e-10)
        workspace[env, kt1 + c] = k1

        # Compute K_t2, cache J_t2 and MinvJt2
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
            rebind[Scalar[DTYPE]](workspace[env, _t2x + c]),
            rebind[Scalar[DTYPE]](workspace[env, _t2y + c]),
            rebind[Scalar[DTYPE]](workspace[env, _t2z + c]),
            J_row,
        )
        var k2: workspace.element_type = 0
        for i in range(NV):
            workspace[env, ws_J_t2 + c * NV + i] = J_row[i]
            var mi_j_sum: workspace.element_type = 0
            for j_idx in range(NV):
                mi_j_sum += (
                    workspace[env, M_inv + i * NV + j_idx] * J_row[j_idx]
                )
            workspace[env, ws_MinvJt2 + c * NV + i] = mi_j_sum
            k2 += J_row[i] * mi_j_sum
        if k2 < Scalar[DTYPE](1e-10):
            k2 = Scalar[DTYPE](1e-10)
        workspace[env, kt2 + c] = k2

        # Warm start tangent impulses (skip degenerate directions)
        var c_off = contacts_off + c * CONTACT_SIZE
        if k1 >= Scalar[DTYPE](FRICTION_K_MIN):
            workspace[env, lt1 + c] = rebind[Scalar[DTYPE]](
                state[env, c_off + CONTACT_IDX_FORCE_T1]
            )
        else:
            workspace[env, lt1 + c] = Scalar[DTYPE](0)
        if k2 >= Scalar[DTYPE](FRICTION_K_MIN):
            workspace[env, lt2 + c] = rebind[Scalar[DTYPE]](
                state[env, c_off + CONTACT_IDX_FORCE_T2]
            )
        else:
            workspace[env, lt2 + c] = Scalar[DTYPE](0)

    # Normal constraint workspace offsets (already in common block at contact_ws_off)
    comptime ws_K_n = contact_ws_off + 1 * MC
    comptime ws_c_dist = contact_ws_off + 2 * MC
    comptime ws_pos_bias = contact_ws_off + 11 * MC
    comptime ws_inv_K_imp = contact_ws_off + 12 * MC
    comptime ws_J_n = contact_ws_off + 13 * MC
    comptime ws_MinvJn = contact_ws_off + 13 * MC + MC * NV

    # Coupled PGS iterations (normals + friction together, MuJoCo-style)
    for _ in range(COUPLED_PGS_ITERATIONS_GPU):
        # --- Normal constraints PGS update ---
        for c in range(nc):
            if workspace[env, ws_c_dist + c] >= Scalar[DTYPE](0):
                continue
            # Compute normal acceleration: a_n = J_n . qacc
            var a_n: workspace.element_type = 0
            for i in range(NV):
                a_n += workspace[env, ws_J_n + c * NV + i] * workspace[env, qacc_idx + i]
            # R = 1/inv_K_imp - K
            var R_n = Scalar[DTYPE](1.0) / workspace[env, ws_inv_K_imp + c] - workspace[env, ws_K_n + c]
            var residual = a_n + workspace[env, ws_pos_bias + c] + R_n * workspace[env, ws_lambda_n + c]
            var delta = -residual * workspace[env, ws_inv_K_imp + c]
            var old_lambda_n = workspace[env, ws_lambda_n + c]
            workspace[env, ws_lambda_n + c] = workspace[env, ws_lambda_n + c] + delta
            if workspace[env, ws_lambda_n + c] < Scalar[DTYPE](0):
                workspace[env, ws_lambda_n + c] = Scalar[DTYPE](0)
            var actual_n = workspace[env, ws_lambda_n + c] - old_lambda_n
            for i in range(NV):
                workspace[env, qacc_idx + i] += workspace[env, ws_MinvJn + c * NV + i] * actual_n

        # --- Friction constraints PGS update (with Coulomb cone) ---
        for c in range(nc):
            if workspace[env, ws_lambda_n + c] <= Scalar[DTYPE](0):
                continue

            var max_friction = friction_coef * workspace[env, ws_lambda_n + c]

            # Tangent 1
            var old_t1 = workspace[env, lt1 + c]
            if workspace[env, kt1 + c] >= Scalar[DTYPE](FRICTION_K_MIN):
                var v_t1: workspace.element_type = 0
                for i in range(NV):
                    v_t1 += workspace[env, ws_J_t1 + c * NV + i] * workspace[env, qacc_idx + i]
                workspace[env, lt1 + c] = workspace[env, lt1 + c] - v_t1 / workspace[env, kt1 + c]

            # Tangent 2
            var old_t2 = workspace[env, lt2 + c]
            if workspace[env, kt2 + c] >= Scalar[DTYPE](FRICTION_K_MIN):
                var v_t2: workspace.element_type = 0
                for i in range(NV):
                    v_t2 += workspace[env, ws_J_t2 + c * NV + i] * workspace[env, qacc_idx + i]
                workspace[env, lt2 + c] = workspace[env, lt2 + c] - v_t2 / workspace[env, kt2 + c]

            # Coulomb cone clamping
            var t_mag = sqrt(
                workspace[env, lt1 + c] * workspace[env, lt1 + c]
                + workspace[env, lt2 + c] * workspace[env, lt2 + c]
            )
            if t_mag > max_friction:
                var scale = max_friction / t_mag
                workspace[env, lt1 + c] = workspace[env, lt1 + c] * scale
                workspace[env, lt2 + c] = workspace[env, lt2 + c] * scale

            var actual_t1 = workspace[env, lt1 + c] - old_t1
            var actual_t2 = workspace[env, lt2 + c] - old_t2

            for i in range(NV):
                workspace[env, qacc_idx + i] += (
                    workspace[env, ws_MinvJt1 + c * NV + i] * actual_t1
                    + workspace[env, ws_MinvJt2 + c * NV + i] * actual_t2
                )

    # Store impulses back for warm-starting
    for c in range(nc):
        var c_off = contacts_off + c * CONTACT_SIZE
        state[env, c_off + CONTACT_IDX_FORCE_N] = workspace[
            env, ws_lambda_n + c
        ]
        state[env, c_off + CONTACT_IDX_FORCE_T1] = workspace[env, lt1 + c]
        state[env, c_off + CONTACT_IDX_FORCE_T2] = workspace[env, lt2 + c]

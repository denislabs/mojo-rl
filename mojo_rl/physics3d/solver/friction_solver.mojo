"""GPU friction solver with variable condim (1/3/4/6) and elliptic/pyramidal cone.

Shared friction PGS solver for GPU, used by CG and Newton solvers.
Supports up to 5 friction directions per contact:
  slot 0 = tangent 1 (slide)
  slot 1 = tangent 2 (slide)
  slot 2 = torsion (angular, along normal)
  slot 3 = roll 1 (angular, along tangent 1)
  slot 4 = roll 2 (angular, along tangent 2)

Two cone models:
  CONE_TYPE=ConeType.ELLIPTIC (elliptic, default): QCQP projection onto elliptic cone
  CONE_TYPE=ConeType.PYRAMIDAL (pyramidal): Edge constraints J_edge = J_n ± μ*J_t, λ≥0

Workspace layout (66*MC + 10*MC*NV per solver):
  lambda_f[5*MC]      - friction impulses per direction (elliptic) / lambda_edge_pos (pyramidal)
  K_f[5*MC]           - effective masses per direction
  dir_f[15*MC]        - direction vectors (5 dirs × 3 components)
  fric_coef[5*MC]     - per-direction friction coefficients
  condim_c[MC]        - per-contact condim
  R_f[5*MC]           - per-direction regularizer values
  bias_f[5*MC]        - per-direction velocity damping bias
  J_f[5*MC*NV]        - Jacobians per direction
  MinvJ_f[5*MC*NV]    - M_inv @ J^T per direction
  --- pyramidal-only (25*MC): ---
  lambda_edge_neg[5*MC] - negative edge lambdas
  C_nt[5*MC]            - cross-term J_n @ MinvJ_f
  K_edge_pos[5*MC]      - K_n + 2μ·C_nt + μ²·K_f
  K_edge_neg[5*MC]      - K_n - 2μ·C_nt + μ²·K_f
  R_edge[5*MC]          - 2·μ²·R_n
"""

from layout import Layout, LayoutTensor
from std.math import sqrt
from ..types import _max_one, ConeType
from ..dynamics.jacobian import (
    compute_contact_jacobian_row_gpu,
    compute_angular_jacobian_row_gpu,
)
from .qcqp import qcqp2, qcqp3, qcqp5, mj_qcqp2, mj_qcqp3, mj_qcqp5
from ..gpu.constants import (
    ws_qacc_constrained_offset,
    ws_m_inv_offset,
    ws_solver_offset,
    model_metadata_offset,
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
    MODEL_META_IDX_IMPRATIO,
    MODEL_META_IDX_SOLREF_CONTACT_0,
    MODEL_META_IDX_SOLREF_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_1,
    qvel_offset,
)

# Coupled PGS iterations (normals + friction together, MuJoCo-style)
comptime COUPLED_PGS_ITERATIONS_GPU: Int = 50
# Minimum K for friction tangent rows — below this, direction is degenerate
comptime FRICTION_K_MIN: Float64 = 1e-6


def friction_workspace_size[MC: Int, NV: Int]() -> Int:
    """Size of friction workspace block: 66*MC + 10*MC*NV.

    Includes 25*MC extra for pyramidal cone (lambda_edge_neg, C_nt,
    K_edge_pos, K_edge_neg, R_edge — 5*MC each).
    """
    return 66 * MC + 10 * MC * NV


# =============================================================================
# Shared friction solver (PGS) - GPU
# (CPU friction is now handled inline via ConstraintData in each solver)
# =============================================================================


@always_inline
def _solve_friction_pgs_gpu[
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
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    MAX_TENDON: Int = 0,
    NSITE: Int = 0,
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    workspace: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ],
    nc: Int,
    contacts_off: Int,
):
    """Friction solver using PGS on GPU with variable condim.

    Supports condim 1 (no friction), 3 (t1+t2), 4 (+torsion), 6 (+roll1+roll2).
    Two cone models:
      cone_type=1 (elliptic): QCQP projection onto elliptic friction cone
      cone_type=0 (pyramidal): Edge constraints J_edge = J_n ± μ*J_t, λ≥0
    """

    comptime qacc_idx = ws_qacc_constrained_offset[NV, NBODY]()
    comptime contact_ws_off = ws_solver_offset[NV, NBODY]()
    comptime fws = contact_ws_off + FRICTION_WS_OFFSET
    comptime MC = _max_one[MAX_CONTACTS]()
    comptime M_inv = ws_m_inv_offset[NV, NBODY]()

    # Contact block (read-only, from common normal workspace)
    comptime ws_lambda_n = contact_ws_off + 0 * MC
    comptime ws_K_n = contact_ws_off + 1 * MC
    comptime ws_c_body = contact_ws_off + 3 * MC
    comptime ws_c_body_b = contact_ws_off + 4 * MC
    comptime ws_c_px = contact_ws_off + 5 * MC
    comptime ws_c_py = contact_ws_off + 6 * MC
    comptime ws_c_pz = contact_ws_off + 7 * MC
    comptime ws_c_nx = contact_ws_off + 8 * MC
    comptime ws_c_ny = contact_ws_off + 9 * MC
    comptime ws_c_nz = contact_ws_off + 10 * MC
    comptime ws_pos_bias = contact_ws_off + 11 * MC
    comptime ws_inv_K_imp = contact_ws_off + 12 * MC
    comptime ws_J_n = contact_ws_off + 15 * MC
    comptime ws_MinvJn = contact_ws_off + 15 * MC + MC * NV

    # Friction workspace offsets (relative to fws)
    # lambda_f[d][c] = fws + d*MC + c
    comptime lf = fws + 0 * MC  # 5 * MC
    # K_f[d][c] = fws + 5*MC + d*MC + c
    comptime kf = fws + 5 * MC  # 5 * MC
    # dir_f[d][axis][c] = fws + 10*MC + (d*3+axis)*MC + c
    comptime df = fws + 10 * MC  # 15 * MC
    # fric_coef[d][c] = fws + 25*MC + d*MC + c
    comptime fc = fws + 25 * MC  # 5 * MC
    # condim_c[c] = fws + 30*MC + c
    comptime cd = fws + 30 * MC  # MC
    # R_f[d][c] = fws + 31*MC + d*MC + c  (friction regularizer per direction)
    comptime rf = fws + 31 * MC  # 5 * MC
    # bias_f[d][c] = fws + 36*MC + d*MC + c  (velocity damping bias per direction)
    comptime bf = fws + 36 * MC  # 5 * MC
    # J_f[d][c*NV+i] = fws + 41*MC + d*MC*NV + c*NV + i
    comptime jf = fws + 41 * MC  # 5 * MC * NV
    # MinvJ_f[d][c*NV+i] = fws + 41*MC + 5*MC*NV + d*MC*NV + c*NV + i
    comptime mj = fws + 41 * MC + 5 * MC * NV  # 5 * MC * NV

    # Pyramidal-only workspace offsets (25*MC after J_f/MinvJ_f)
    comptime le_neg = fws + 41 * MC + 10 * MC * NV  # lambda_edge_neg[5*MC]
    comptime cnt = le_neg + 5 * MC  # C_nt[5*MC] (cross-term)
    comptime kep = cnt + 5 * MC  # K_edge_pos[5*MC]
    comptime ken = kep + 5 * MC  # K_edge_neg[5*MC]
    comptime re = ken + 5 * MC  # R_edge[5*MC]

    # Read cone_type from model metadata
    comptime model_meta_off_ct = model_metadata_offset[NBODY, NJOINT]()

    # Initialize friction workspace
    for c in range(MC):
        for d in range(5):
            workspace[env, lf + d * MC + c] = Scalar[DTYPE](0)
            workspace[env, kf + d * MC + c] = Scalar[DTYPE](1)
            workspace[env, fc + d * MC + c] = Scalar[DTYPE](0)
            workspace[env, rf + d * MC + c] = Scalar[DTYPE](0)
            workspace[env, bf + d * MC + c] = Scalar[DTYPE](0)
            # Pyramidal workspace
            workspace[env, le_neg + d * MC + c] = Scalar[DTYPE](0)
            workspace[env, cnt + d * MC + c] = Scalar[DTYPE](0)
            workspace[env, kep + d * MC + c] = Scalar[DTYPE](1)
            workspace[env, ken + d * MC + c] = Scalar[DTYPE](1)
            workspace[env, re + d * MC + c] = Scalar[DTYPE](0)
            for axis in range(3):
                workspace[env, df + (d * 3 + axis) * MC + c] = Scalar[DTYPE](0)
        workspace[env, cd + c] = Scalar[DTYPE](3)  # default condim=3

    var J_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

    # Precompute friction directions, Jacobians, and K for active contacts
    for c in range(nc):
        if workspace[env, ws_lambda_n + c] <= Scalar[DTYPE](0):
            continue

        var c_off = contacts_off + c * CONTACT_SIZE

        # Read per-contact friction params from state buffer
        var mu_slide = rebind[Scalar[DTYPE]](
            state[env, c_off + CONTACT_IDX_FRICTION]
        )
        if mu_slide <= Scalar[DTYPE](0):
            mu_slide = Scalar[DTYPE](
                0.5
            )  # fallback (contacts always have friction from geom specs)
        var mu_spin = rebind[Scalar[DTYPE]](
            state[env, c_off + CONTACT_IDX_FRICTION_SPIN]
        )
        var mu_roll = rebind[Scalar[DTYPE]](
            state[env, c_off + CONTACT_IDX_FRICTION_ROLL]
        )
        var condim = Int(
            rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_CONDIM])
        )
        if condim < 1:
            condim = 3
        workspace[env, cd + c] = Scalar[DTYPE](condim)

        if condim == 1:
            continue  # No friction

        var nx = rebind[Scalar[DTYPE]](workspace[env, ws_c_nx + c])
        var ny = rebind[Scalar[DTYPE]](workspace[env, ws_c_ny + c])
        var nz = rebind[Scalar[DTYPE]](workspace[env, ws_c_nz + c])

        # Compute tangent basis (MuJoCo mju_makeFrame with capsule axis hint)
        var hint_x = rebind[Scalar[DTYPE]](
            state[env, c_off + CONTACT_IDX_FRAME_T1_X]
        )
        var hint_y = rebind[Scalar[DTYPE]](
            state[env, c_off + CONTACT_IDX_FRAME_T1_Y]
        )
        var hint_z = rebind[Scalar[DTYPE]](
            state[env, c_off + CONTACT_IDX_FRAME_T1_Z]
        )
        var hint_len_sq = hint_x * hint_x + hint_y * hint_y + hint_z * hint_z

        # If no hint (non-capsule), use MuJoCo default
        if hint_len_sq < Scalar[DTYPE](0.25):
            var abs_nx = abs(nx)
            var abs_ny = abs(ny)
            var abs_nz = abs(nz)
            if abs_nx <= abs_ny and abs_nx <= abs_nz:
                hint_x = Scalar[DTYPE](1); hint_y = Scalar[DTYPE](0); hint_z = Scalar[DTYPE](0)
            elif abs_ny <= abs_nz:
                hint_x = Scalar[DTYPE](0); hint_y = Scalar[DTYPE](1); hint_z = Scalar[DTYPE](0)
            else:
                hint_x = Scalar[DTYPE](0); hint_y = Scalar[DTYPE](0); hint_z = Scalar[DTYPE](1)

        # Gram-Schmidt: orthogonalize hint against normal
        var dot_nh = nx * hint_x + ny * hint_y + nz * hint_z
        var t1x = hint_x - dot_nh * nx
        var t1y = hint_y - dot_nh * ny
        var t1z = hint_z - dot_nh * nz
        var t1_mag = sqrt(t1x * t1x + t1y * t1y + t1z * t1z)
        if t1_mag < Scalar[DTYPE](1e-10):
            # Hint parallel to normal — fall back to least-aligned axis
            var abs_nx = abs(nx)
            var abs_ny = abs(ny)
            var abs_nz = abs(nz)
            if abs_nx <= abs_ny and abs_nx <= abs_nz:
                hint_x = Scalar[DTYPE](1); hint_y = Scalar[DTYPE](0); hint_z = Scalar[DTYPE](0)
            elif abs_ny <= abs_nz:
                hint_x = Scalar[DTYPE](0); hint_y = Scalar[DTYPE](1); hint_z = Scalar[DTYPE](0)
            else:
                hint_x = Scalar[DTYPE](0); hint_y = Scalar[DTYPE](0); hint_z = Scalar[DTYPE](1)
            dot_nh = nx * hint_x + ny * hint_y + nz * hint_z
            t1x = hint_x - dot_nh * nx
            t1y = hint_y - dot_nh * ny
            t1z = hint_z - dot_nh * nz
            t1_mag = sqrt(t1x * t1x + t1y * t1y + t1z * t1z)
        if t1_mag > Scalar[DTYPE](1e-10):
            t1x = t1x / t1_mag
            t1y = t1y / t1_mag
            t1z = t1z / t1_mag

        # T2 = cross(normal, T1)
        var t2x = ny * t1z - nz * t1y
        var t2y = nz * t1x - nx * t1z
        var t2z = nx * t1y - ny * t1x

        # Store tangent directions
        workspace[env, df + (0 * 3 + 0) * MC + c] = t1x
        workspace[env, df + (0 * 3 + 1) * MC + c] = t1y
        workspace[env, df + (0 * 3 + 2) * MC + c] = t1z
        workspace[env, df + (1 * 3 + 0) * MC + c] = t2x
        workspace[env, df + (1 * 3 + 1) * MC + c] = t2y
        workspace[env, df + (1 * 3 + 2) * MC + c] = t2z

        # Store friction coefficients
        workspace[env, fc + 0 * MC + c] = mu_slide
        workspace[env, fc + 1 * MC + c] = mu_slide

        # Number of friction directions based on condim
        var num_fric = 2  # condim=3: t1+t2
        if condim >= 4:
            num_fric = 3  # +torsion
            # Torsion direction = normal
            workspace[env, df + (2 * 3 + 0) * MC + c] = nx
            workspace[env, df + (2 * 3 + 1) * MC + c] = ny
            workspace[env, df + (2 * 3 + 2) * MC + c] = nz
            workspace[env, fc + 2 * MC + c] = mu_spin
        if condim >= 6:
            num_fric = 5  # +roll1+roll2
            # Roll 1 direction = t1
            workspace[env, df + (3 * 3 + 0) * MC + c] = t1x
            workspace[env, df + (3 * 3 + 1) * MC + c] = t1y
            workspace[env, df + (3 * 3 + 2) * MC + c] = t1z
            # Roll 2 direction = t2
            workspace[env, df + (4 * 3 + 0) * MC + c] = t2x
            workspace[env, df + (4 * 3 + 1) * MC + c] = t2y
            workspace[env, df + (4 * 3 + 2) * MC + c] = t2z
            workspace[env, fc + 3 * MC + c] = mu_roll
            workspace[env, fc + 4 * MC + c] = mu_roll

        var body_a = Int(workspace[env, ws_c_body + c])
        var body_b = Int(workspace[env, ws_c_body_b + c])
        var px = rebind[Scalar[DTYPE]](workspace[env, ws_c_px + c])
        var py = rebind[Scalar[DTYPE]](workspace[env, ws_c_py + c])
        var pz = rebind[Scalar[DTYPE]](workspace[env, ws_c_pz + c])

        # Compute Jacobian, MinvJ, and K for each friction direction
        for d in range(num_fric):
            var dx = rebind[Scalar[DTYPE]](
                workspace[env, df + (d * 3 + 0) * MC + c]
            )
            var dy = rebind[Scalar[DTYPE]](
                workspace[env, df + (d * 3 + 1) * MC + c]
            )
            var dz = rebind[Scalar[DTYPE]](
                workspace[env, df + (d * 3 + 2) * MC + c]
            )

            if d < 2:
                # Slide friction: full spatial Jacobian
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
                # Torsion/rolling: angular-only Jacobian
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
                workspace[env, jf + d * MC * NV + c * NV + i] = J_row[i]
                var mi_j_sum: workspace.element_type = 0
                for j_idx in range(NV):
                    mi_j_sum += (
                        workspace[env, M_inv + i * NV + j_idx] * J_row[j_idx]
                    )
                workspace[env, mj + d * MC * NV + c * NV + i] = mi_j_sum
                k_d += J_row[i] * mi_j_sum
            if k_d < Scalar[DTYPE](1e-10):
                k_d = Scalar[DTYPE](1e-10)
            workspace[env, kf + d * MC + c] = k_d

        # Compute friction regularizer R_f from parent normal's impedance
        comptime model_meta_off = model_metadata_offset[NBODY, NJOINT]()
        var impratio = rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_IMPRATIO]
        )
        if impratio < Scalar[DTYPE](1e-6):
            impratio = Scalar[DTYPE](1.0)
        # Compute R_base = R_n / impratio directly from stored imp and diag_n
        # (avoids lossy float32 round-trip through inv_K_imp)
        comptime ws_imp_n_slot = contact_ws_off + 13 * MC
        comptime ws_diag_n_slot = contact_ws_off + 14 * MC
        var imp_n = rebind[Scalar[DTYPE]](workspace[env, ws_imp_n_slot + c])
        var diag_n_val = rebind[Scalar[DTYPE]](
            workspace[env, ws_diag_n_slot + c]
        )
        var R_base = (
            (Scalar[DTYPE](1.0) - imp_n)
            / imp_n
            * diag_n_val
            / impratio
        )
        for d in range(num_fric):
            var R_d = R_base
            # For condim > 3, scale by mu_slide^2/mu_dir^2
            if d >= 2:
                var mu_d = rebind[Scalar[DTYPE]](
                    workspace[env, fc + d * MC + c]
                )
                if mu_d > Scalar[DTYPE](1e-12):
                    R_d = R_base * mu_slide * mu_slide / (mu_d * mu_d)
            workspace[env, rf + d * MC + c] = R_d

        # Compute velocity damping bias for friction rows
        # bias_f = B_damp * imp_n * v_tangential (matches MuJoCo friction aref)
        var sr_tc = rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_SOLREF_CONTACT_0]
        )
        var sr_dr = rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_SOLREF_CONTACT_1]
        )
        var si_dmax = rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_SOLIMP_CONTACT_1]
        )
        if si_dmax < Scalar[DTYPE](1e-4):
            si_dmax = Scalar[DTYPE](1e-4)
        var B_damp = Scalar[DTYPE](2.0) * sr_dr / (sr_tc * si_dmax)
        comptime qvel_off = qvel_offset[NQ, NV]()
        for d in range(num_fric):
            var v_t: workspace.element_type = 0
            for i in range(NV):
                v_t += rebind[Scalar[DTYPE]](
                    workspace[env, jf + d * MC * NV + c * NV + i]
                ) * rebind[Scalar[DTYPE]](state[env, qvel_off + i])
            workspace[env, bf + d * MC + c] = B_damp * rebind[Scalar[DTYPE]](
                v_t
            )

        # Pyramidal precomputation: cross-term C_nt, K_edge_pos/neg, R_edge
        comptime if CONE_TYPE == ConeType.PYRAMIDAL:
            # R_n computed directly from stored imp and diag_n
            var R_n_val = (
                (Scalar[DTYPE](1.0) - imp_n)
                / imp_n
                * diag_n_val
            )
            for d in range(num_fric):
                var mu_d = rebind[Scalar[DTYPE]](
                    workspace[env, fc + d * MC + c]
                )
                # Cross-term: C_nt[d][c] = Σ_i J_n[c*NV+i] * MinvJ_f[d*MC*NV+c*NV+i]
                var c_nt_val: workspace.element_type = 0
                for i in range(NV):
                    c_nt_val += rebind[Scalar[DTYPE]](
                        workspace[env, ws_J_n + c * NV + i]
                    ) * rebind[Scalar[DTYPE]](
                        workspace[env, mj + d * MC * NV + c * NV + i]
                    )
                workspace[env, cnt + d * MC + c] = c_nt_val
                # K_edge_pos = K_n + 2*mu*C_nt + mu^2*K_f
                var K_n_c = rebind[Scalar[DTYPE]](workspace[env, ws_K_n + c])
                var K_f_d = rebind[Scalar[DTYPE]](
                    workspace[env, kf + d * MC + c]
                )
                workspace[env, kep + d * MC + c] = (
                    K_n_c
                    + Scalar[DTYPE](2.0) * mu_d * c_nt_val
                    + mu_d * mu_d * K_f_d
                )
                # K_edge_neg = K_n - 2*mu*C_nt + mu^2*K_f
                workspace[env, ken + d * MC + c] = (
                    K_n_c
                    - Scalar[DTYPE](2.0) * mu_d * c_nt_val
                    + mu_d * mu_d * K_f_d
                )
                # R_edge = 2*mu^2*R_n
                workspace[env, re + d * MC + c] = (
                    Scalar[DTYPE](2.0) * mu_d * mu_d * R_n_val
                )
            # No warm-start for pyramidal (edge lambdas start at 0)
            for d in range(num_fric):
                workspace[env, lf + d * MC + c] = Scalar[DTYPE](0)
                workspace[env, le_neg + d * MC + c] = Scalar[DTYPE](0)
        else:
            # Warm-start friction impulses (elliptic only)
            var warm_force_idx = InlineArray[Int, 5](uninitialized=True)
            warm_force_idx[0] = CONTACT_IDX_FORCE_T1
            warm_force_idx[1] = CONTACT_IDX_FORCE_T2
            warm_force_idx[2] = CONTACT_IDX_FORCE_TORSION
            warm_force_idx[3] = CONTACT_IDX_FORCE_ROLL1
            warm_force_idx[4] = CONTACT_IDX_FORCE_ROLL2
            for d in range(num_fric):
                if workspace[env, kf + d * MC + c] >= Scalar[DTYPE](
                    FRICTION_K_MIN
                ):
                    workspace[env, lf + d * MC + c] = rebind[Scalar[DTYPE]](
                        state[env, c_off + warm_force_idx[d]]
                    )
                else:
                    workspace[env, lf + d * MC + c] = Scalar[DTYPE](0)

    comptime ws_c_dist = contact_ws_off + 2 * MC

    # Coupled PGS iterations (normals + friction together, MuJoCo-style)
    for _ in range(COUPLED_PGS_ITERATIONS_GPU):
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
            # Compute R_n directly from stored imp and diag_n
            var imp_c = rebind[Scalar[DTYPE]](
                workspace[env, ws_imp_n_slot + c]
            )
            var diag_c = rebind[Scalar[DTYPE]](
                workspace[env, ws_diag_n_slot + c]
            )
            var R_n = (Scalar[DTYPE](1.0) - imp_c) / imp_c * diag_c
            var residual = (
                a_n
                + workspace[env, ws_pos_bias + c]
                + R_n * workspace[env, ws_lambda_n + c]
            )
            var delta = -residual * workspace[env, ws_inv_K_imp + c]
            var old_lambda_n = workspace[env, ws_lambda_n + c]
            workspace[env, ws_lambda_n + c] = (
                workspace[env, ws_lambda_n + c] + delta
            )
            if workspace[env, ws_lambda_n + c] < Scalar[DTYPE](0):
                workspace[env, ws_lambda_n + c] = Scalar[DTYPE](0)
            var actual_n = workspace[env, ws_lambda_n + c] - old_lambda_n
            for i in range(NV):
                workspace[env, qacc_idx + i] += (
                    workspace[env, ws_MinvJn + c * NV + i] * actual_n
                )

        # --- Friction constraints PGS update ---
        for c in range(nc):
            if workspace[env, ws_lambda_n + c] <= Scalar[DTYPE](0):
                # Zero friction when normal force is zero (cone constraint)
                var condim_z = Int(workspace[env, cd + c])
                var num_fric_z = 2
                if condim_z >= 4:
                    num_fric_z = 3
                if condim_z >= 6:
                    num_fric_z = 5
                for d in range(num_fric_z):
                    comptime if CONE_TYPE == ConeType.PYRAMIDAL:
                        # Pyramidal: undo edge forces (MinvJ_n ± mu*MinvJ_f)
                        var mu_d = rebind[Scalar[DTYPE]](
                            workspace[env, fc + d * MC + c]
                        )
                        var old_pos = rebind[Scalar[DTYPE]](
                            workspace[env, lf + d * MC + c]
                        )
                        var old_neg_v = rebind[Scalar[DTYPE]](
                            workspace[env, le_neg + d * MC + c]
                        )
                        if old_pos != Scalar[DTYPE](0) or old_neg_v != Scalar[
                            DTYPE
                        ](0):
                            workspace[env, lf + d * MC + c] = Scalar[DTYPE](0)
                            workspace[env, le_neg + d * MC + c] = Scalar[DTYPE](
                                0
                            )
                            for i in range(NV):
                                var minvjn_i = rebind[Scalar[DTYPE]](
                                    workspace[env, ws_MinvJn + c * NV + i]
                                )
                                var minvjf_i = rebind[Scalar[DTYPE]](
                                    workspace[
                                        env, mj + d * MC * NV + c * NV + i
                                    ]
                                )
                                workspace[env, qacc_idx + i] -= (
                                    minvjn_i + mu_d * minvjf_i
                                ) * old_pos
                                workspace[env, qacc_idx + i] -= (
                                    minvjn_i - mu_d * minvjf_i
                                ) * old_neg_v
                    else:
                        # Elliptic: undo via MinvJ_f only
                        var old_f = rebind[Scalar[DTYPE]](
                            workspace[env, lf + d * MC + c]
                        )
                        if old_f != Scalar[DTYPE](0):
                            workspace[env, lf + d * MC + c] = Scalar[DTYPE](0)
                            for i in range(NV):
                                workspace[env, qacc_idx + i] -= (
                                    workspace[
                                        env, mj + d * MC * NV + c * NV + i
                                    ]
                                    * old_f
                                )
                continue
            var condim = Int(workspace[env, cd + c])
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
                # Save old values for delta computation
                var old_pos = InlineArray[Scalar[DTYPE], 5](
                    fill=Scalar[DTYPE](0)
                )
                var old_neg = InlineArray[Scalar[DTYPE], 5](
                    fill=Scalar[DTYPE](0)
                )
                for d in range(num_fric):
                    old_pos[d] = rebind[Scalar[DTYPE]](
                        workspace[env, lf + d * MC + c]
                    )
                    old_neg[d] = rebind[Scalar[DTYPE]](
                        workspace[env, le_neg + d * MC + c]
                    )

                var bias_n = rebind[Scalar[DTYPE]](
                    workspace[env, ws_pos_bias + c]
                )

                for d in range(num_fric):
                    var mu_d = rebind[Scalar[DTYPE]](
                        workspace[env, fc + d * MC + c]
                    )
                    if mu_d <= Scalar[DTYPE](1e-12):
                        continue

                    # Compute a_n = J_n @ qacc, a_f = J_f @ qacc
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
                                workspace[env, jf + d * MC * NV + c * NV + i]
                            )
                            * qi
                        )

                    var R_e = rebind[Scalar[DTYPE]](
                        workspace[env, re + d * MC + c]
                    )

                    # Per-edge bias: bias_edge = bias_n ± mu * B_damp * v_t
                    # bf[d*MC+c] already stores B_damp * v_t (friction velocity damping)
                    var bf_d = rebind[Scalar[DTYPE]](
                        workspace[env, bf + d * MC + c]
                    )

                    # Positive edge (+): a_edge = a_n + mu * a_f
                    var a_edge_pos = a_n_val + mu_d * a_f_val
                    var K_ep = rebind[Scalar[DTYPE]](
                        workspace[env, kep + d * MC + c]
                    )
                    var residual_pos = (
                        a_edge_pos
                        + bias_n
                        + mu_d * bf_d
                        + R_e
                        * rebind[Scalar[DTYPE]](workspace[env, lf + d * MC + c])
                    )
                    var delta_pos = -residual_pos / (K_ep + R_e)
                    var new_lp = (
                        rebind[Scalar[DTYPE]](workspace[env, lf + d * MC + c])
                        + delta_pos
                    )
                    if new_lp < Scalar[DTYPE](0):
                        new_lp = Scalar[DTYPE](0)
                    var actual_pos = new_lp - rebind[Scalar[DTYPE]](
                        workspace[env, lf + d * MC + c]
                    )
                    workspace[env, lf + d * MC + c] = new_lp
                    # Apply delta via MinvJ_n + mu * MinvJ_f
                    if actual_pos != Scalar[DTYPE](0):
                        for i in range(NV):
                            workspace[env, qacc_idx + i] += (
                                rebind[Scalar[DTYPE]](
                                    workspace[env, ws_MinvJn + c * NV + i]
                                )
                                + mu_d
                                * rebind[Scalar[DTYPE]](
                                    workspace[
                                        env, mj + d * MC * NV + c * NV + i
                                    ]
                                )
                            ) * actual_pos

                    # Recompute a_n, a_f after positive edge update
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
                                workspace[env, jf + d * MC * NV + c * NV + i]
                            )
                            * qi
                        )

                    # Negative edge (-): a_edge = a_n - mu * a_f
                    var a_edge_neg = a_n_val - mu_d * a_f_val
                    var K_en = rebind[Scalar[DTYPE]](
                        workspace[env, ken + d * MC + c]
                    )
                    var residual_neg = (
                        a_edge_neg
                        + bias_n
                        - mu_d * bf_d
                        + R_e
                        * rebind[Scalar[DTYPE]](
                            workspace[env, le_neg + d * MC + c]
                        )
                    )
                    var delta_neg = -residual_neg / (K_en + R_e)
                    var new_ln = (
                        rebind[Scalar[DTYPE]](
                            workspace[env, le_neg + d * MC + c]
                        )
                        + delta_neg
                    )
                    if new_ln < Scalar[DTYPE](0):
                        new_ln = Scalar[DTYPE](0)
                    var actual_neg = new_ln - rebind[Scalar[DTYPE]](
                        workspace[env, le_neg + d * MC + c]
                    )
                    workspace[env, le_neg + d * MC + c] = new_ln
                    # Apply delta via MinvJ_n - mu * MinvJ_f
                    if actual_neg != Scalar[DTYPE](0):
                        for i in range(NV):
                            workspace[env, qacc_idx + i] += (
                                rebind[Scalar[DTYPE]](
                                    workspace[env, ws_MinvJn + c * NV + i]
                                )
                                - mu_d
                                * rebind[Scalar[DTYPE]](
                                    workspace[
                                        env, mj + d * MC * NV + c * NV + i
                                    ]
                                )
                            ) * actual_neg
            else:
                # === ELLIPTIC CONE: MuJoCo-style block update ===
                # Ray update + QCQP with AR submatrix + costChange
                var dim = 1 + num_fric

                # Build block AR matrix on-the-fly from J/MinvJ
                # AR[0,0] = K_n + R_n (normal-normal)
                # AR[0,d+1] = AR[d+1,0] = J_n @ MinvJ_f (normal-friction cross)
                # AR[d1+1,d2+1] = J_f[d1] @ MinvJ_f[d2] + R_f*delta(d1,d2)
                var AR = InlineArray[Scalar[DTYPE], 36](fill=Scalar[DTYPE](0))
                # Compute R_n directly from stored imp and diag_n
                var imp_c2 = rebind[Scalar[DTYPE]](
                    workspace[env, ws_imp_n_slot + c]
                )
                var diag_c2 = rebind[Scalar[DTYPE]](
                    workspace[env, ws_diag_n_slot + c]
                )
                var R_n_val = (Scalar[DTYPE](1.0) - imp_c2) / imp_c2 * diag_c2
                AR[0] = (
                    rebind[Scalar[DTYPE]](workspace[env, ws_K_n + c]) + R_n_val
                )

                for d1 in range(num_fric):
                    # Normal-friction cross: J_n @ MinvJ_f[d1]
                    var cross: Scalar[DTYPE] = 0
                    for i in range(NV):
                        cross += rebind[Scalar[DTYPE]](
                            workspace[env, ws_J_n + c * NV + i]
                        ) * rebind[Scalar[DTYPE]](
                            workspace[env, mj + d1 * MC * NV + c * NV + i]
                        )
                    AR[(d1 + 1)] = cross  # AR[0, d1+1]
                    AR[(d1 + 1) * dim] = cross  # AR[d1+1, 0]

                    for d2 in range(num_fric):
                        # Friction-friction: J_f[d1] @ MinvJ_f[d2]
                        var ff: Scalar[DTYPE] = 0
                        for i in range(NV):
                            ff += rebind[Scalar[DTYPE]](
                                workspace[env, jf + d1 * MC * NV + c * NV + i]
                            ) * rebind[Scalar[DTYPE]](
                                workspace[env, mj + d2 * MC * NV + c * NV + i]
                            )
                        if d1 == d2:
                            ff += rebind[Scalar[DTYPE]](
                                workspace[env, rf + d1 * MC + c]
                            )
                        AR[(d1 + 1) * dim + (d2 + 1)] = ff

                # Compute block residual
                var block_res = InlineArray[Scalar[DTYPE], 6](
                    fill=Scalar[DTYPE](0)
                )
                # Normal residual
                var a_n_res: Scalar[DTYPE] = 0
                for i in range(NV):
                    a_n_res += rebind[Scalar[DTYPE]](
                        workspace[env, ws_J_n + c * NV + i]
                    ) * rebind[Scalar[DTYPE]](workspace[env, qacc_idx + i])
                block_res[0] = (
                    a_n_res
                    + rebind[Scalar[DTYPE]](workspace[env, ws_pos_bias + c])
                    + R_n_val
                    * rebind[Scalar[DTYPE]](workspace[env, ws_lambda_n + c])
                )
                # Friction residuals
                for d in range(num_fric):
                    var a_f_res: Scalar[DTYPE] = 0
                    for i in range(NV):
                        a_f_res += rebind[Scalar[DTYPE]](
                            workspace[env, jf + d * MC * NV + c * NV + i]
                        ) * rebind[Scalar[DTYPE]](workspace[env, qacc_idx + i])
                    var R_f_d = rebind[Scalar[DTYPE]](
                        workspace[env, rf + d * MC + c]
                    )
                    block_res[1 + d] = (
                        a_f_res
                        + rebind[Scalar[DTYPE]](workspace[env, bf + d * MC + c])
                        + R_f_d
                        * rebind[Scalar[DTYPE]](workspace[env, lf + d * MC + c])
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
                        workspace[env, lf + d * MC + c]
                    )

                var ARinv0: Scalar[DTYPE] = 0
                if AR[0] > Scalar[DTYPE](1e-10):
                    ARinv0 = Scalar[DTYPE](1.0) / AR[0]

                # --- Ray update ---
                if rebind[Scalar[DTYPE]](
                    workspace[env, ws_lambda_n + c]
                ) < Scalar[DTYPE](1e-10):
                    # Normal force too small: scalar update, zero friction
                    workspace[env, ws_lambda_n + c] = (
                        rebind[Scalar[DTYPE]](workspace[env, ws_lambda_n + c])
                        - block_res[0] * ARinv0
                    )
                    if workspace[env, ws_lambda_n + c] < Scalar[DTYPE](0):
                        workspace[env, ws_lambda_n + c] = Scalar[DTYPE](0)
                    for d in range(num_fric):
                        workspace[env, lf + d * MC + c] = Scalar[DTYPE](0)
                else:
                    var v = InlineArray[Scalar[DTYPE], 6](fill=Scalar[DTYPE](0))
                    v[0] = rebind[Scalar[DTYPE]](
                        workspace[env, ws_lambda_n + c]
                    )
                    for d in range(num_fric):
                        v[1 + d] = rebind[Scalar[DTYPE]](
                            workspace[env, lf + d * MC + c]
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
                            workspace[env, lf + d * MC + c] = (
                                rebind[Scalar[DTYPE]](
                                    workspace[env, lf + d * MC + c]
                                )
                                + x * v[1 + d]
                            )

                # --- QCQP friction update ---
                var fn_val = rebind[Scalar[DTYPE]](
                    workspace[env, ws_lambda_n + c]
                )
                if fn_val >= Scalar[DTYPE](1e-10) and num_fric > 0:
                    # Build friction sub-block Ac and adjusted bias bc
                    var Ac = InlineArray[Scalar[DTYPE], 25](
                        fill=Scalar[DTYPE](0)
                    )
                    var bc_arr = InlineArray[Scalar[DTYPE], 5](
                        fill=Scalar[DTYPE](0)
                    )
                    for j in range(num_fric):
                        for j2 in range(num_fric):
                            Ac[j * num_fric + j2] = AR[(1 + j) * dim + (1 + j2)]
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
                            workspace[env, fc + d * MC + c]
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
                        flg_active = mj_qcqp2[DTYPE](r0, r1, A2, b2, d2, fn_val)
                        workspace[env, lf + 0 * MC + c] = r0
                        workspace[env, lf + 1 * MC + c] = r1
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
                        workspace[env, lf + 0 * MC + c] = r0
                        workspace[env, lf + 1 * MC + c] = r1
                        workspace[env, lf + 2 * MC + c] = r2
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
                        flg_active = mj_qcqp5[DTYPE](res5, A5, b5, d5, fn_val)
                        for d in range(5):
                            workspace[env, lf + d * MC + c] = res5[d]

                    # Rescale to exact ellipsoid if constrained
                    if flg_active:
                        var s: Scalar[DTYPE] = 0
                        for d in range(num_fric):
                            var fv = rebind[Scalar[DTYPE]](
                                workspace[env, lf + d * MC + c]
                            )
                            var mu_d = mu_arr[d]
                            if mu_d > Scalar[DTYPE](1e-10):
                                s += fv * fv / (mu_d * mu_d)
                        if s > Scalar[DTYPE](1e-10):
                            var scale = sqrt(fn_val * fn_val / s)
                            for d in range(num_fric):
                                workspace[env, lf + d * MC + c] = (
                                    rebind[Scalar[DTYPE]](
                                        workspace[env, lf + d * MC + c]
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
                            workspace[env, lf + (bi - 1) * MC + c]
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
                                workspace[env, lf + (bj - 1) * MC + c]
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
                        workspace[env, lf + d * MC + c] = oldforce[1 + d]

                # Apply delta to qacc
                var actual_n = (
                    rebind[Scalar[DTYPE]](workspace[env, ws_lambda_n + c])
                    - oldforce[0]
                )
                if actual_n != Scalar[DTYPE](0):
                    for i in range(NV):
                        workspace[env, qacc_idx + i] += (
                            workspace[env, ws_MinvJn + c * NV + i] * actual_n
                        )
                for d in range(num_fric):
                    var actual_f = (
                        rebind[Scalar[DTYPE]](workspace[env, lf + d * MC + c])
                        - oldforce[1 + d]
                    )
                    if actual_f != Scalar[DTYPE](0):
                        for i in range(NV):
                            workspace[env, qacc_idx + i] += (
                                workspace[env, mj + d * MC * NV + c * NV + i]
                                * actual_f
                            )

    # Store impulses back for warm-starting
    comptime if CONE_TYPE == ConeType.PYRAMIDAL:
        # Pyramidal: force_n includes edge contributions, tangent = mu*(pos-neg)
        for c in range(nc):
            var c_off = contacts_off + c * CONTACT_SIZE
            var condim = Int(workspace[env, cd + c])
            var num_fric = 2
            if condim >= 4:
                num_fric = 3
            if condim >= 6:
                num_fric = 5
            # Normal force = lambda_n + sum of all edge lambdas
            var total_n = rebind[Scalar[DTYPE]](workspace[env, ws_lambda_n + c])
            for d in range(num_fric):
                total_n += rebind[Scalar[DTYPE]](
                    workspace[env, lf + d * MC + c]
                )
                total_n += rebind[Scalar[DTYPE]](
                    workspace[env, le_neg + d * MC + c]
                )
            state[env, c_off + CONTACT_IDX_FORCE_N] = total_n
            # Tangent forces = mu * (lambda_pos - lambda_neg)
            var mu_0 = rebind[Scalar[DTYPE]](workspace[env, fc + 0 * MC + c])
            state[env, c_off + CONTACT_IDX_FORCE_T1] = mu_0 * (
                rebind[Scalar[DTYPE]](workspace[env, lf + 0 * MC + c])
                - rebind[Scalar[DTYPE]](workspace[env, le_neg + 0 * MC + c])
            )
            var mu_1 = rebind[Scalar[DTYPE]](workspace[env, fc + 1 * MC + c])
            state[env, c_off + CONTACT_IDX_FORCE_T2] = mu_1 * (
                rebind[Scalar[DTYPE]](workspace[env, lf + 1 * MC + c])
                - rebind[Scalar[DTYPE]](workspace[env, le_neg + 1 * MC + c])
            )
            if condim >= 4:
                var mu_2 = rebind[Scalar[DTYPE]](
                    workspace[env, fc + 2 * MC + c]
                )
                state[env, c_off + CONTACT_IDX_FORCE_TORSION] = mu_2 * (
                    rebind[Scalar[DTYPE]](workspace[env, lf + 2 * MC + c])
                    - rebind[Scalar[DTYPE]](workspace[env, le_neg + 2 * MC + c])
                )
            if condim >= 6:
                var mu_3 = rebind[Scalar[DTYPE]](
                    workspace[env, fc + 3 * MC + c]
                )
                state[env, c_off + CONTACT_IDX_FORCE_ROLL1] = mu_3 * (
                    rebind[Scalar[DTYPE]](workspace[env, lf + 3 * MC + c])
                    - rebind[Scalar[DTYPE]](workspace[env, le_neg + 3 * MC + c])
                )
                var mu_4 = rebind[Scalar[DTYPE]](
                    workspace[env, fc + 4 * MC + c]
                )
                state[env, c_off + CONTACT_IDX_FORCE_ROLL2] = mu_4 * (
                    rebind[Scalar[DTYPE]](workspace[env, lf + 4 * MC + c])
                    - rebind[Scalar[DTYPE]](workspace[env, le_neg + 4 * MC + c])
                )
    else:
        # Elliptic: direct force writeback
        for c in range(nc):
            var c_off = contacts_off + c * CONTACT_SIZE
            state[env, c_off + CONTACT_IDX_FORCE_N] = workspace[
                env, ws_lambda_n + c
            ]
            state[env, c_off + CONTACT_IDX_FORCE_T1] = workspace[
                env, lf + 0 * MC + c
            ]
            state[env, c_off + CONTACT_IDX_FORCE_T2] = workspace[
                env, lf + 1 * MC + c
            ]
            var condim = Int(workspace[env, cd + c])
            if condim >= 4:
                state[env, c_off + CONTACT_IDX_FORCE_TORSION] = workspace[
                    env, lf + 2 * MC + c
                ]
            if condim >= 6:
                state[env, c_off + CONTACT_IDX_FORCE_ROLL1] = workspace[
                    env, lf + 3 * MC + c
                ]
                state[env, c_off + CONTACT_IDX_FORCE_ROLL2] = workspace[
                    env, lf + 4 * MC + c
                ]

"""GPU constraint builder - shared constraint setup for all GPU solvers.

Extracts duplicated contact setup, impedance computation, joint limit detection,
and limit solving from PGS, CG, and Newton GPU solvers. Each solver keeps its
unique iteration logic (PGS sweeps, CG iterations, Newton + line search).

All functions are @always_inline (required for Metal compiler).

Common normal workspace block layout (at solver_idx):
  [0*MC..1*MC)                  lambda_n      Normal impulses
  [1*MC..2*MC)                  K_n           Effective mass
  [2*MC..3*MC)                  c_dist        Contact distance
  [3*MC..4*MC)                  c_body        Body A index
  [4*MC..5*MC)                  c_body_b      Body B index
  [5*MC..8*MC)                  c_px/py/pz    Contact position
  [8*MC..11*MC)                 c_nx/ny/nz    Contact normal
  [11*MC..12*MC)                pos_bias      Impedance position correction
  [12*MC..13*MC)                inv_K_imp     imp/K ratio
  [13*MC..13*MC+MC*NV)          J_n           Normal Jacobian
  [13*MC+MC*NV..13*MC+2*MC*NV)  MinvJn        M_inv @ J_n^T

  COMMON_NORMAL_SIZE = 13*MC + 2*MC*NV
"""

from math import sqrt
from layout import LayoutTensor, Layout
from ..types import _max_one
from ..joint_types import JNT_HINGE, JNT_SLIDE
from ..dynamics.jacobian import compute_contact_jacobian_row_gpu
from ..gpu.constants import (
    contacts_offset,
    metadata_offset,
    model_metadata_offset,
    model_joint_offset,
    qvel_offset,
    ws_qacc_constrained_offset,
    ws_m_inv_offset,
    ws_solver_offset,
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
    CONTACT_IDX_FORCE_N,
    META_IDX_NUM_CONTACTS,
    MODEL_META_IDX_SOLREF_LIMIT_0,
    MODEL_META_IDX_SOLREF_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_0,
    MODEL_META_IDX_SOLIMP_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_2,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
)


fn common_normal_size[MC: Int, NV: Int]() -> Int:
    """Size of the common normal workspace block."""
    return 13 * MC + 2 * MC * NV


# =============================================================================
# 1. init_common_normal_workspace_gpu
# =============================================================================


@always_inline
fn init_common_normal_workspace_gpu[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    MAX_CONTACTS: Int,
    WS_SIZE: Int,
    BATCH: Int,
](
    env: Int,
    contact_tid: Int,
    workspace: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ],
):
    """Zero-initialize common normal workspace fields for one contact slot.

    Called in PARALLEL phase (one thread per contact slot).
    """
    comptime si = ws_solver_offset[NV, NBODY]()
    comptime MC = _max_one[MAX_CONTACTS]()

    workspace[env, si + 0 * MC + contact_tid] = 0  # lambda_n
    workspace[env, si + 1 * MC + contact_tid] = 1  # K_n
    workspace[env, si + 2 * MC + contact_tid] = 0  # c_dist
    workspace[env, si + 3 * MC + contact_tid] = 0  # c_body
    workspace[env, si + 4 * MC + contact_tid] = -1  # c_body_b
    workspace[env, si + 5 * MC + contact_tid] = 0  # c_px
    workspace[env, si + 6 * MC + contact_tid] = 0  # c_py
    workspace[env, si + 7 * MC + contact_tid] = 0  # c_pz
    workspace[env, si + 8 * MC + contact_tid] = 0  # c_nx
    workspace[env, si + 9 * MC + contact_tid] = 0  # c_ny
    workspace[env, si + 10 * MC + contact_tid] = 1  # c_nz
    workspace[env, si + 11 * MC + contact_tid] = 0  # pos_bias
    workspace[env, si + 12 * MC + contact_tid] = 0  # inv_K_imp
    # Zero J_n and MinvJn for this slot
    for i in range(NV):
        workspace[env, si + 13 * MC + contact_tid * NV + i] = 0
        workspace[env, si + 13 * MC + MC * NV + contact_tid * NV + i] = 0


# =============================================================================
# 2. precompute_contact_normal_gpu
# =============================================================================


@always_inline
fn precompute_contact_normal_gpu[
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
    COMPUTE_RHS: Bool = False,
    RHS_IDX: Int = 0,
](
    env: Int,
    contact_tid: Int,
    nc: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    workspace: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ],
    K_spring: Scalar[DTYPE],
    B_damp: Scalar[DTYPE],
    si_dmin: Scalar[DTYPE],
    si_dmax: Scalar[DTYPE],
    si_width: Scalar[DTYPE],
):
    """Precompute one contact's normal constraint data (parallel, one thread per contact).

    Reads contact from state buffer, computes J_n via Jacobian, MinvJn, K_n,
    acceleration-level aref (pos_bias, inv_K), and stores warm-start lambda_n.

    When COMPUTE_RHS is True, also computes a_n and stores rhs = a_n + bias
    at workspace offset RHS_IDX (used by CG and Newton solvers).
    """
    comptime contacts_off = contacts_offset[NQ, NV, NBODY]()
    comptime si = ws_solver_offset[NV, NBODY]()
    comptime M_inv_idx = ws_m_inv_offset[NV, NBODY]()
    comptime MC = _max_one[MAX_CONTACTS]()

    # Common block offsets
    comptime ws_lambda_n = si + 0 * MC
    comptime ws_K_n = si + 1 * MC
    comptime ws_c_dist = si + 2 * MC
    comptime ws_c_body = si + 3 * MC
    comptime ws_c_body_b = si + 4 * MC
    comptime ws_c_px = si + 5 * MC
    comptime ws_c_py = si + 6 * MC
    comptime ws_c_pz = si + 7 * MC
    comptime ws_c_nx = si + 8 * MC
    comptime ws_c_ny = si + 9 * MC
    comptime ws_c_nz = si + 10 * MC
    comptime ws_pos_bias = si + 11 * MC
    comptime ws_inv_K_imp = si + 12 * MC
    comptime ws_J_n = si + 13 * MC
    comptime ws_MinvJn = si + 13 * MC + MC * NV

    var J_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(V_SIZE):
        J_row[i] = 0

    if contact_tid < nc:
        var c = contact_tid
        var c_off = contacts_off + c * CONTACT_SIZE
        var body = Int(
            rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_BODY_A])
        )
        var body_b = Int(
            rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_BODY_B])
        )
        var dist = rebind[Scalar[DTYPE]](state[env, c_off + CONTACT_IDX_DIST])

        workspace[env, ws_c_dist + c] = dist
        workspace[env, ws_c_body + c] = Scalar[DTYPE](body)
        workspace[env, ws_c_body_b + c] = Scalar[DTYPE](body_b)

        if dist < Scalar[DTYPE](0):
            workspace[env, ws_c_px + c] = state[env, c_off + CONTACT_IDX_POS_X]
            workspace[env, ws_c_py + c] = state[env, c_off + CONTACT_IDX_POS_Y]
            workspace[env, ws_c_pz + c] = state[env, c_off + CONTACT_IDX_POS_Z]
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
            var v_n: workspace.element_type = 0
            var a_n: workspace.element_type = 0
            comptime qvel_off = qvel_offset[NQ, NV]()
            comptime qacc_idx = ws_qacc_constrained_offset[NV, NBODY]()

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
                # Use current VELOCITY for damping in aref (MuJoCo: efc_vel = J*qvel)
                v_n += J_row[i] * rebind[Scalar[DTYPE]](state[env, qvel_off + i])
                # Constraint-space acceleration (for solver RHS)
                a_n += J_row[i] * workspace[env, qacc_idx + i]

            if k < Scalar[DTYPE](1e-10):
                k = Scalar[DTYPE](1e-10)
            workspace[env, ws_K_n + c] = k

            # Acceleration-level aref: Hermite smoothstep impedance
            var penetration = -dist
            var x = penetration / si_width
            if x > Scalar[DTYPE](1.0):
                x = Scalar[DTYPE](1.0)
            var imp = si_dmin + (
                Scalar[DTYPE](3.0) * x * x - Scalar[DTYPE](2.0) * x * x * x
            ) * (si_dmax - si_dmin)
            # Impedance floor: 0.2 ensures firm contact from first touch
            if imp < Scalar[DTYPE](0.2):
                imp = Scalar[DTYPE](0.2)
            # aref = K*imp*pen - B*v_n (B term without imp for stronger damping)
            # Solver uses: delta = -(a_n + bias + R*lambda) * inv_K
            var bias = -K_spring * imp * penetration + B_damp * v_n
            workspace[env, ws_pos_bias + c] = bias
            # MuJoCo: AR[i,i] = K + (1-imp)/imp * K = K/imp, so inv = imp/K
            workspace[env, ws_inv_K_imp + c] = imp / k

            @parameter
            if COMPUTE_RHS:
                workspace[env, RHS_IDX + c] = a_n + bias

            # Warm-start lambda
            workspace[env, ws_lambda_n + c] = state[
                env, c_off + CONTACT_IDX_FORCE_N
            ]


# =============================================================================
# 3. warmstart_normals_gpu
# =============================================================================


@always_inline
fn warmstart_normals_gpu[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    MAX_CONTACTS: Int,
    WS_SIZE: Int,
    BATCH: Int,
](
    env: Int,
    nc: Int,
    workspace: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ],
):
    """Apply warm-start normal impulses to predicted velocity (sequential, thread 0).
    """
    comptime si = ws_solver_offset[NV, NBODY]()
    comptime qacc_idx = ws_qacc_constrained_offset[NV, NBODY]()
    comptime MC = _max_one[MAX_CONTACTS]()
    comptime ws_lambda_n = si + 0 * MC
    comptime ws_c_dist = si + 2 * MC
    comptime ws_MinvJn = si + 13 * MC + MC * NV

    for c in range(nc):
        if workspace[env, ws_c_dist + c] >= Scalar[DTYPE](0):
            continue
        if workspace[env, ws_lambda_n + c] > Scalar[DTYPE](0):
            for i in range(NV):
                workspace[env, qacc_idx + i] += (
                    workspace[env, ws_MinvJn + c * NV + i]
                    * workspace[env, ws_lambda_n + c]
                )


# =============================================================================
# 4. apply_solved_normals_gpu
# =============================================================================


@always_inline
fn apply_solved_normals_gpu[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    MAX_CONTACTS: Int,
    STATE_SIZE: Int,
    WS_SIZE: Int,
    BATCH: Int,
](
    env: Int,
    nc: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    workspace: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ],
):
    """Remove warm-start and apply final solved normal impulses (sequential, thread 0).

    Used by CG and Newton solvers after their iterative solve phase.
    """
    comptime contacts_off = contacts_offset[NQ, NV, NBODY]()
    comptime si = ws_solver_offset[NV, NBODY]()
    comptime qacc_idx = ws_qacc_constrained_offset[NV, NBODY]()
    comptime MC = _max_one[MAX_CONTACTS]()
    comptime ws_lambda_n = si + 0 * MC
    comptime ws_c_dist = si + 2 * MC
    comptime ws_MinvJn = si + 13 * MC + MC * NV

    # Remove warm-start impulses
    for c in range(nc):
        if workspace[env, ws_c_dist + c] >= Scalar[DTYPE](0):
            continue
        var c_off = contacts_off + c * CONTACT_SIZE
        var warm = rebind[Scalar[DTYPE]](
            state[env, c_off + CONTACT_IDX_FORCE_N]
        )
        if warm > Scalar[DTYPE](0):
            for i in range(NV):
                workspace[env, qacc_idx + i] -= rebind[Scalar[DTYPE]](
                    workspace[env, ws_MinvJn + c * NV + i] * warm
                )

    # Apply final solved impulses
    for c in range(nc):
        if workspace[env, ws_c_dist + c] >= Scalar[DTYPE](0):
            continue
        if workspace[env, ws_lambda_n + c] > Scalar[DTYPE](0):
            for i in range(NV):
                workspace[env, qacc_idx + i] += rebind[Scalar[DTYPE]](
                    workspace[env, ws_MinvJn + c * NV + i]
                    * workspace[env, ws_lambda_n + c]
                )


# =============================================================================
# 5. detect_and_solve_limits_gpu
# =============================================================================


@always_inline
fn detect_and_solve_limits_gpu[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    STATE_SIZE: Int,
    MODEL_SIZE: Int,
    WS_SIZE: Int,
    BATCH: Int,
    NUM_ITERATIONS: Int,
](
    env: Int,
    dt: Scalar[DTYPE],
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    workspace: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ],
):
    """Detect active joint limits and solve them using PGS with impedance.

    Called in SEQUENTIAL phase (thread 0 only). Allocates limit InlineArrays
    internally, detects which joints are at their limits, precomputes impedance,
    and runs PGS iterations.
    """
    comptime qacc_idx = ws_qacc_constrained_offset[NV, NBODY]()
    comptime M_inv_idx = ws_m_inv_offset[NV, NBODY]()
    comptime model_meta_off = model_metadata_offset[NBODY, NJOINT]()
    comptime MAX_LIMITS = _max_one[2 * NJOINT]()

    # Detect active joint limits
    var limit_dof = InlineArray[Int, MAX_LIMITS](uninitialized=True)
    var limit_sign = InlineArray[Scalar[DTYPE], MAX_LIMITS](uninitialized=True)
    var limit_dist_arr = InlineArray[Scalar[DTYPE], MAX_LIMITS](
        uninitialized=True
    )
    var K_limit = InlineArray[Scalar[DTYPE], MAX_LIMITS](uninitialized=True)
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
    for j in range(NJOINT):
        var j_off = model_joint_offset[NBODY](j)
        var jtype = Int(rebind[Scalar[DTYPE]](model[0, j_off + JOINT_IDX_TYPE]))
        if jtype != JNT_HINGE and jtype != JNT_SLIDE:
            continue
        var dof = Int(
            rebind[Scalar[DTYPE]](model[0, j_off + JOINT_IDX_DOF_ADR])
        )
        var qpos_adr = Int(
            rebind[Scalar[DTYPE]](model[0, j_off + JOINT_IDX_QPOS_ADR])
        )
        var rmin = rebind[Scalar[DTYPE]](model[0, j_off + JOINT_IDX_RANGE_MIN])
        var rmax = rebind[Scalar[DTYPE]](model[0, j_off + JOINT_IDX_RANGE_MAX])
        if rmin < Scalar[DTYPE](-1e9) or rmax > Scalar[DTYPE](1e9):
            continue
        var pos = rebind[Scalar[DTYPE]](state[env, qpos_adr])
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

    if num_limits == 0:
        return

    # Read limit solref/solimp
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
    # Acceleration-level coefficients for limits
    var l_K_spring = Scalar[DTYPE](1.0) / (
        li_dmax * li_dmax * lr_tc * lr_tc * lr_dr * lr_dr
    )
    var l_B_damp = Scalar[DTYPE](2.0) / (li_dmax * lr_tc)

    # Precompute impedance and MinvJ for limits
    var lim_bias = InlineArray[Scalar[DTYPE], MAX_LIMITS](uninitialized=True)
    var lim_inv_K = InlineArray[Scalar[DTYPE], MAX_LIMITS](uninitialized=True)
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
        # Impedance floor: 0.2 ensures firm limit correction from first touch
        if imp_lim < Scalar[DTYPE](0.2):
            imp_lim = Scalar[DTYPE](0.2)
        # Use current VELOCITY for damping (MuJoCo: aref = K*d*pen - B*d*v)
        comptime qvel_off_lim = qvel_offset[NQ, NV]()
        var v_limit = limit_sign[l] * rebind[Scalar[DTYPE]](
            state[env, qvel_off_lim + limit_dof[l]]
        )
        lim_bias[l] = rebind[Scalar[DTYPE]](
            -l_K_spring * imp_lim * penetration + l_B_damp * v_limit
        )
        # MuJoCo: AR = K + (1-imp)/imp * K = K/imp, so inv = imp/K
        lim_inv_K[l] = imp_lim / K_limit[l]
        var ldof = limit_dof[l]
        var lsign = limit_sign[l]
        for i in range(NV):
            lim_MinvJ[l * NV + i] = (
                rebind[Scalar[DTYPE]](workspace[env, M_inv_idx + i * NV + ldof])
                * lsign
            )

    # PGS iterations for limits (acceleration-level)
    for _ in range(NUM_ITERATIONS):
        var max_lim_delta: Scalar[DTYPE] = 0
        for l in range(num_limits):
            var a_limit = (
                limit_sign[l] * workspace[env, qacc_idx + limit_dof[l]]
            )
            # MuJoCo regularizer: R = K/imp - K = 1/inv_K - K
            var R_lim = Scalar[DTYPE](1.0) / lim_inv_K[l] - K_limit[l]
            var residual_l = a_limit + lim_bias[l] + R_lim * lambda_limit[l]
            var delta_l = -residual_l * lim_inv_K[l]
            var old_lam = lambda_limit[l]
            lambda_limit[l] = lambda_limit[l] + rebind[Scalar[DTYPE]](delta_l)
            if lambda_limit[l] < Scalar[DTYPE](0):
                lambda_limit[l] = Scalar[DTYPE](0)
            var actual_l = lambda_limit[l] - old_lam
            var abs_l = abs(actual_l)
            if abs_l > max_lim_delta:
                max_lim_delta = abs_l
            for i in range(NV):
                workspace[env, qacc_idx + i] += lim_MinvJ[l * NV + i] * actual_l
        if max_lim_delta < Scalar[DTYPE](1e-4):
            break

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

from math import sqrt, pow
from layout import LayoutTensor, Layout
from ..types import _max_one, EQ_CONNECT, EQ_WELD
from ..joint_types import JNT_HINGE, JNT_SLIDE
from ..dynamics.jacobian import (
    compute_contact_jacobian_row_gpu,
    compute_angular_jacobian_row_gpu,
)
from ..kinematics.quat_math import quat_mul, quat_conjugate, quat_rotate
from ..gpu.constants import (
    contacts_offset,
    metadata_offset,
    model_metadata_offset,
    model_joint_offset,
    qpos_offset,
    qvel_offset,
    xpos_offset,
    xquat_offset,
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
    MODEL_META_IDX_SOLIMP_LIMIT_3,
    MODEL_META_IDX_SOLIMP_LIMIT_4,
    MODEL_META_IDX_NEQUALITY,
    MODEL_META_IDX_NTENDON,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    MODEL_EQ_SIZE,
    EQ_IDX_TYPE,
    EQ_IDX_BODY_A,
    EQ_IDX_BODY_B,
    EQ_IDX_ANCHOR_AX,
    EQ_IDX_ANCHOR_AY,
    EQ_IDX_ANCHOR_AZ,
    EQ_IDX_ANCHOR_BX,
    EQ_IDX_ANCHOR_BY,
    EQ_IDX_ANCHOR_BZ,
    EQ_IDX_RELPOSE_X,
    EQ_IDX_RELPOSE_Y,
    EQ_IDX_RELPOSE_Z,
    EQ_IDX_RELPOSE_W,
    EQ_IDX_SOLREF_0,
    EQ_IDX_SOLREF_1,
    EQ_IDX_SOLIMP_0,
    EQ_IDX_SOLIMP_1,
    EQ_IDX_SOLIMP_2,
    EQ_IDX_SOLIMP_3,
    EQ_IDX_SOLIMP_4,
    model_equality_offset,
    model_body_invweight0_offset,
    model_dof_invweight0_offset,
    MODEL_TENDON_SIZE,
    TENDON_IDX_NUM_JOINTS,
    TENDON_IDX_JOINT_0,
    TENDON_IDX_JOINT_1,
    TENDON_IDX_JOINT_2,
    TENDON_IDX_JOINT_3,
    TENDON_IDX_COEF_0,
    TENDON_IDX_COEF_1,
    TENDON_IDX_COEF_2,
    TENDON_IDX_COEF_3,
    TENDON_IDX_LENGTH_REF,
    TENDON_IDX_SOLREF_0,
    TENDON_IDX_SOLREF_1,
    TENDON_IDX_SOLIMP_0,
    TENDON_IDX_SOLIMP_1,
    TENDON_IDX_SOLIMP_2,
    TENDON_IDX_SOLIMP_3,
    TENDON_IDX_SOLIMP_4,
    model_tendon_offset,
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
    NGEOM: Int = 0,
    MAX_EQUALITY: Int = 0,
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
    si_midpoint: Scalar[DTYPE],
    si_power: Scalar[DTYPE],
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
                v_n += J_row[i] * rebind[Scalar[DTYPE]](
                    state[env, qvel_off + i]
                )
                # Constraint-space acceleration (for solver RHS)
                a_n += J_row[i] * workspace[env, qacc_idx + i]

            if k < Scalar[DTYPE](1e-10):
                k = Scalar[DTYPE](1e-10)
            workspace[env, ws_K_n + c] = k

            # Acceleration-level aref: MuJoCo piecewise power impedance
            var penetration = -dist
            var imp: Scalar[DTYPE]
            if si_dmin == si_dmax or si_width <= Scalar[DTYPE](0):
                imp = Scalar[DTYPE](0.5) * (si_dmin + si_dmax)
            else:
                var x = penetration / si_width
                var y: Scalar[DTYPE]
                if x <= Scalar[DTYPE](0):
                    y = Scalar[DTYPE](0)
                elif x >= Scalar[DTYPE](1):
                    y = Scalar[DTYPE](1)
                elif si_power == Scalar[DTYPE](1):
                    y = x
                elif x <= si_midpoint:
                    var a = Scalar[DTYPE](1) / pow(si_midpoint, si_power - Scalar[DTYPE](1))
                    y = a * pow(x, si_power)
                else:
                    var b = Scalar[DTYPE](1) / pow(Scalar[DTYPE](1) - si_midpoint, si_power - Scalar[DTYPE](1))
                    y = Scalar[DTYPE](1) - b * pow(Scalar[DTYPE](1) - x, si_power)
                imp = si_dmin + y * (si_dmax - si_dmin)
            # Impedance floor prevents zero-force contacts at surface
            if imp < Scalar[DTYPE](1e-6):
                imp = Scalar[DTYPE](1e-6)
            # MuJoCo: aref = -B*vel - K*imp*pos, bias = -aref = B*vel + K*imp*pen
            # bias = -aref = -(K*imp*pen - B*v_n) = -K*imp*pen + B*v_n
            var bias = -K_spring * imp * penetration + B_damp * v_n
            workspace[env, ws_pos_bias + c] = bias
            # MuJoCo: R = (1-imp)/imp * diagApprox, inv_K_imp = 1/(K + R)
            # diagApprox = body_invweight0[2*body_a] + body_invweight0[2*body_b]
            comptime bw_off = model_body_invweight0_offset[
                NBODY, NJOINT, NGEOM, MAX_EQUALITY
            ]()
            var diag_n: Scalar[DTYPE] = 0
            if body > 0 and body < NBODY:
                diag_n += rebind[Scalar[DTYPE]](model[0, bw_off + body * 2])
            if body_b > 0 and body_b < NBODY:
                diag_n += rebind[Scalar[DTYPE]](model[0, bw_off + body_b * 2])
            if diag_n < Scalar[DTYPE](1e-10):
                diag_n = rebind[Scalar[DTYPE]](k)  # Fallback to exact K
            var R_n = (Scalar[DTYPE](1.0) - imp) / imp * diag_n
            workspace[env, ws_inv_K_imp + c] = Scalar[DTYPE](1.0) / (rebind[Scalar[DTYPE]](k) + R_n)

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
    NGEOM: Int = 0,
    MAX_EQUALITY: Int = 0,
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
    var li_midpoint = rebind[Scalar[DTYPE]](
        model[0, model_meta_off + MODEL_META_IDX_SOLIMP_LIMIT_3]
    )
    var li_power = rebind[Scalar[DTYPE]](
        model[0, model_meta_off + MODEL_META_IDX_SOLIMP_LIMIT_4]
    )
    if li_width < Scalar[DTYPE](1e-6):
        li_width = Scalar[DTYPE](1e-6)
    if li_dmax < Scalar[DTYPE](1e-4):
        li_dmax = Scalar[DTYPE](1e-4)
    # Acceleration-level coefficients for limits
    # MuJoCo formula: K = 1/(tc² * dr²), B = 2*dr/tc
    var l_K_spring = Scalar[DTYPE](1.0) / (
        lr_tc * lr_tc * li_dmax * li_dmax
    )
    var l_B_damp = Scalar[DTYPE](2.0) * lr_dr / (lr_tc * li_dmax)

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
        var imp_lim: Scalar[DTYPE]
        if li_dmin == li_dmax or li_width <= Scalar[DTYPE](0):
            imp_lim = Scalar[DTYPE](0.5) * (li_dmin + li_dmax)
        else:
            var x_lim = penetration / li_width
            var y_lim: Scalar[DTYPE]
            if x_lim <= Scalar[DTYPE](0):
                y_lim = Scalar[DTYPE](0)
            elif x_lim >= Scalar[DTYPE](1):
                y_lim = Scalar[DTYPE](1)
            elif li_power == Scalar[DTYPE](1):
                y_lim = x_lim
            elif x_lim <= li_midpoint:
                var a = Scalar[DTYPE](1) / pow(li_midpoint, li_power - Scalar[DTYPE](1))
                y_lim = a * pow(x_lim, li_power)
            else:
                var b = Scalar[DTYPE](1) / pow(Scalar[DTYPE](1) - li_midpoint, li_power - Scalar[DTYPE](1))
                y_lim = Scalar[DTYPE](1) - b * pow(Scalar[DTYPE](1) - x_lim, li_power)
            imp_lim = li_dmin + y_lim * (li_dmax - li_dmin)
        # MuJoCo uses mjMINIMP ~1e-6
        if imp_lim < Scalar[DTYPE](1e-6):
            imp_lim = Scalar[DTYPE](1e-6)
        # aref = K*imp*pen - B*v, bias = -aref
        comptime qvel_off_lim = qvel_offset[NQ, NV]()
        var v_limit = limit_sign[l] * rebind[Scalar[DTYPE]](
            state[env, qvel_off_lim + limit_dof[l]]
        )
        lim_bias[l] = rebind[Scalar[DTYPE]](
            -l_K_spring * imp_lim * penetration + l_B_damp * v_limit
        )
        # MuJoCo: R = (1-imp)/imp * dof_invweight0[dof], inv_K = 1/(K + R)
        comptime dw_off = model_dof_invweight0_offset[
            NBODY, NJOINT, NGEOM, MAX_EQUALITY
        ]()
        var diag_lim = rebind[Scalar[DTYPE]](model[0, dw_off + limit_dof[l]])
        if diag_lim < Scalar[DTYPE](1e-10):
            diag_lim = K_limit[l]  # Fallback
        var R_lim = (Scalar[DTYPE](1.0) - imp_lim) / imp_lim * diag_lim
        lim_inv_K[l] = Scalar[DTYPE](1.0) / (K_limit[l] + R_lim)
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


# =============================================================================
# 6. build_and_solve_equality_gpu
# =============================================================================


@always_inline
fn build_and_solve_equality_gpu[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    MAX_EQUALITY: Int,
    NGEOM: Int,
    STATE_SIZE: Int,
    MODEL_SIZE: Int,
    V_SIZE: Int,
    WS_SIZE: Int,
    BATCH: Int,
    NUM_ITERATIONS: Int,
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    workspace: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ],
):
    """Build and solve equality constraints (connect + weld) on GPU.

    Called in SEQUENTIAL phase (thread 0 only). Reads equality constraint
    definitions from model buffer, computes world anchors, Jacobians,
    impedance, and runs bilateral PGS iterations.

    Similar pattern to detect_and_solve_limits_gpu but for bilateral
    equality constraints (no lambda >= 0 clamping).
    """

    @parameter
    if MAX_EQUALITY == 0:
        return

    comptime qacc_idx = ws_qacc_constrained_offset[NV, NBODY]()
    comptime M_inv_idx = ws_m_inv_offset[NV, NBODY]()
    comptime model_meta_off = model_metadata_offset[NBODY, NJOINT]()
    comptime xpos_off = xpos_offset[NQ, NV, NBODY]()
    comptime xquat_off = xquat_offset[NQ, NV, NBODY]()
    comptime qvel_off = qvel_offset[NQ, NV]()

    # Read number of equality constraints from model metadata
    var neq = Int(
        rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_NEQUALITY]
        )
    )
    if neq == 0:
        return
    if neq > MAX_EQUALITY:
        neq = MAX_EQUALITY

    # Max rows: 6 per constraint (3 connect + 3 weld orientation)
    comptime MAX_EQ_ROWS = _max_one[6 * MAX_EQUALITY]()
    comptime MINVJ_EQ_SIZE = _max_one[6 * MAX_EQUALITY * NV]()

    var eq_K = InlineArray[Scalar[DTYPE], MAX_EQ_ROWS](fill=Scalar[DTYPE](1))
    var eq_bias = InlineArray[Scalar[DTYPE], MAX_EQ_ROWS](fill=Scalar[DTYPE](0))
    var eq_inv_K_imp = InlineArray[Scalar[DTYPE], MAX_EQ_ROWS](
        fill=Scalar[DTYPE](0)
    )
    var eq_lambda = InlineArray[Scalar[DTYPE], MAX_EQ_ROWS](
        fill=Scalar[DTYPE](0)
    )
    var eq_J = InlineArray[Scalar[DTYPE], MINVJ_EQ_SIZE](fill=Scalar[DTYPE](0))
    var eq_MinvJ = InlineArray[Scalar[DTYPE], MINVJ_EQ_SIZE](
        fill=Scalar[DTYPE](0)
    )

    var J_row = InlineArray[Scalar[DTYPE], V_SIZE](fill=Scalar[DTYPE](0))

    var num_eq_rows = 0

    # Build rows for each equality constraint
    for eq_i in range(neq):
        var eq_off = model_equality_offset[NBODY, NJOINT, NGEOM](eq_i)
        var eq_type = Int(rebind[Scalar[DTYPE]](model[0, eq_off + EQ_IDX_TYPE]))
        var body_a = Int(
            rebind[Scalar[DTYPE]](model[0, eq_off + EQ_IDX_BODY_A])
        )
        var body_b = Int(
            rebind[Scalar[DTYPE]](model[0, eq_off + EQ_IDX_BODY_B])
        )

        # Read anchors
        var anc_ax = rebind[Scalar[DTYPE]](model[0, eq_off + EQ_IDX_ANCHOR_AX])
        var anc_ay = rebind[Scalar[DTYPE]](model[0, eq_off + EQ_IDX_ANCHOR_AY])
        var anc_az = rebind[Scalar[DTYPE]](model[0, eq_off + EQ_IDX_ANCHOR_AZ])
        var anc_bx = rebind[Scalar[DTYPE]](model[0, eq_off + EQ_IDX_ANCHOR_BX])
        var anc_by = rebind[Scalar[DTYPE]](model[0, eq_off + EQ_IDX_ANCHOR_BY])
        var anc_bz = rebind[Scalar[DTYPE]](model[0, eq_off + EQ_IDX_ANCHOR_BZ])

        # Read solref/solimp
        var sr_tc = rebind[Scalar[DTYPE]](model[0, eq_off + EQ_IDX_SOLREF_0])
        var sr_dr = rebind[Scalar[DTYPE]](model[0, eq_off + EQ_IDX_SOLREF_1])
        var si_dmin = rebind[Scalar[DTYPE]](model[0, eq_off + EQ_IDX_SOLIMP_0])
        var si_dmax = rebind[Scalar[DTYPE]](model[0, eq_off + EQ_IDX_SOLIMP_1])
        var si_width = rebind[Scalar[DTYPE]](model[0, eq_off + EQ_IDX_SOLIMP_2])
        var si_midpoint = rebind[Scalar[DTYPE]](
            model[0, eq_off + EQ_IDX_SOLIMP_3]
        )
        var si_power = rebind[Scalar[DTYPE]](model[0, eq_off + EQ_IDX_SOLIMP_4])
        if si_width < Scalar[DTYPE](1e-6):
            si_width = Scalar[DTYPE](1e-6)
        if si_dmax < Scalar[DTYPE](1e-4):
            si_dmax = Scalar[DTYPE](1e-4)
        var eq_K_spring = Scalar[DTYPE](1.0) / (
            sr_tc * sr_tc * si_dmax * si_dmax
        )
        var eq_B_damp = Scalar[DTYPE](2.0) * sr_dr / (sr_tc * si_dmax)

        # Compute world anchor A: xpos[body_a] + quat_rotate(xquat[body_a], anchor_a)
        var xpos_a_x = rebind[Scalar[DTYPE]](
            state[env, xpos_off + body_a * 3 + 0]
        )
        var xpos_a_y = rebind[Scalar[DTYPE]](
            state[env, xpos_off + body_a * 3 + 1]
        )
        var xpos_a_z = rebind[Scalar[DTYPE]](
            state[env, xpos_off + body_a * 3 + 2]
        )
        var xquat_a_x = rebind[Scalar[DTYPE]](
            state[env, xquat_off + body_a * 4 + 0]
        )
        var xquat_a_y = rebind[Scalar[DTYPE]](
            state[env, xquat_off + body_a * 4 + 1]
        )
        var xquat_a_z = rebind[Scalar[DTYPE]](
            state[env, xquat_off + body_a * 4 + 2]
        )
        var xquat_a_w = rebind[Scalar[DTYPE]](
            state[env, xquat_off + body_a * 4 + 3]
        )
        var rot_a = quat_rotate[DTYPE](
            xquat_a_x, xquat_a_y, xquat_a_z, xquat_a_w, anc_ax, anc_ay, anc_az
        )
        var world_ax = xpos_a_x + rot_a[0]
        var world_ay = xpos_a_y + rot_a[1]
        var world_az = xpos_a_z + rot_a[2]

        # Compute world anchor B
        var world_bx: Scalar[DTYPE]
        var world_by: Scalar[DTYPE]
        var world_bz: Scalar[DTYPE]
        if body_b > 0:
            var xpos_b_x = rebind[Scalar[DTYPE]](
                state[env, xpos_off + body_b * 3 + 0]
            )
            var xpos_b_y = rebind[Scalar[DTYPE]](
                state[env, xpos_off + body_b * 3 + 1]
            )
            var xpos_b_z = rebind[Scalar[DTYPE]](
                state[env, xpos_off + body_b * 3 + 2]
            )
            var xquat_b_x = rebind[Scalar[DTYPE]](
                state[env, xquat_off + body_b * 4 + 0]
            )
            var xquat_b_y = rebind[Scalar[DTYPE]](
                state[env, xquat_off + body_b * 4 + 1]
            )
            var xquat_b_z = rebind[Scalar[DTYPE]](
                state[env, xquat_off + body_b * 4 + 2]
            )
            var xquat_b_w = rebind[Scalar[DTYPE]](
                state[env, xquat_off + body_b * 4 + 3]
            )
            var rot_b = quat_rotate[DTYPE](
                xquat_b_x,
                xquat_b_y,
                xquat_b_z,
                xquat_b_w,
                anc_bx,
                anc_by,
                anc_bz,
            )
            world_bx = xpos_b_x + rot_b[0]
            world_by = xpos_b_y + rot_b[1]
            world_bz = xpos_b_z + rot_b[2]
        else:
            world_bx = anc_bx
            world_by = anc_by
            world_bz = anc_bz

        var pos_err_x = world_ax - world_bx
        var pos_err_y = world_ay - world_by
        var pos_err_z = world_az - world_bz

        # --- 3 position rows (connect + weld) ---
        var dirs = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
        dirs[0] = Scalar[DTYPE](1)  # x-axis: (1,0,0)
        dirs[4] = Scalar[DTYPE](1)  # y-axis: (0,1,0)
        dirs[8] = Scalar[DTYPE](1)  # z-axis: (0,0,1)

        var pos_errs = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
        pos_errs[0] = pos_err_x
        pos_errs[1] = pos_err_y
        pos_errs[2] = pos_err_z

        for d in range(3):
            if num_eq_rows >= MAX_EQ_ROWS:
                break
            var dx = dirs[d * 3 + 0]
            var dy = dirs[d * 3 + 1]
            var dz = dirs[d * 3 + 2]

            # Compute Jacobian
            for i in range(V_SIZE):
                J_row[i] = 0
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
                world_ax,
                world_ay,
                world_az,
                dx,
                dy,
                dz,
                J_row,
            )

            # Compute K = J @ M_inv @ J^T, store J and MinvJ
            var k: Scalar[DTYPE] = 0
            var v_n: Scalar[DTYPE] = 0
            for i in range(NV):
                eq_J[num_eq_rows * NV + i] = J_row[i]
                var mi_j_sum: Scalar[DTYPE] = 0
                for j_idx in range(NV):
                    mi_j_sum += (
                        rebind[Scalar[DTYPE]](
                            workspace[env, M_inv_idx + i * NV + j_idx]
                        )
                        * J_row[j_idx]
                    )
                eq_MinvJ[num_eq_rows * NV + i] = mi_j_sum
                k += J_row[i] * mi_j_sum
                v_n += J_row[i] * rebind[Scalar[DTYPE]](
                    state[env, qvel_off + i]
                )

            if k < Scalar[DTYPE](1e-10):
                k = Scalar[DTYPE](1e-10)
            eq_K[num_eq_rows] = k

            # Impedance: MuJoCo piecewise power formula
            var err_d = pos_errs[d]
            var penetration = abs(err_d)
            var imp: Scalar[DTYPE]
            if si_dmin == si_dmax or si_width <= Scalar[DTYPE](0):
                imp = Scalar[DTYPE](0.5) * (si_dmin + si_dmax)
            else:
                var x = penetration / si_width
                var y: Scalar[DTYPE]
                if x <= Scalar[DTYPE](0):
                    y = Scalar[DTYPE](0)
                elif x >= Scalar[DTYPE](1):
                    y = Scalar[DTYPE](1)
                elif si_power == Scalar[DTYPE](1):
                    y = x
                elif x <= si_midpoint:
                    var a = Scalar[DTYPE](1) / pow(si_midpoint, si_power - Scalar[DTYPE](1))
                    y = a * pow(x, si_power)
                else:
                    var b = Scalar[DTYPE](1) / pow(Scalar[DTYPE](1) - si_midpoint, si_power - Scalar[DTYPE](1))
                    y = Scalar[DTYPE](1) - b * pow(Scalar[DTYPE](1) - x, si_power)
                imp = si_dmin + y * (si_dmax - si_dmin)
            if imp < Scalar[DTYPE](1e-6):
                imp = Scalar[DTYPE](1e-6)

            # bias = -aref (bilateral: sign depends on error direction)
            var bias = -eq_K_spring * imp * penetration + eq_B_damp * v_n
            if err_d < Scalar[DTYPE](0):
                bias = -bias
            eq_bias[num_eq_rows] = bias
            # MuJoCo: R = (1-imp)/imp * diagApprox (translation weights)
            comptime eq_bw_off = model_body_invweight0_offset[
                NBODY, NJOINT, NGEOM, MAX_EQUALITY
            ]()
            var diag_eq: Scalar[DTYPE] = 0
            if body_a > 0 and body_a < NBODY:
                diag_eq += rebind[Scalar[DTYPE]](
                    model[0, eq_bw_off + body_a * 2]
                )
            if body_b > 0 and body_b < NBODY:
                diag_eq += rebind[Scalar[DTYPE]](
                    model[0, eq_bw_off + body_b * 2]
                )
            if diag_eq < Scalar[DTYPE](1e-10):
                diag_eq = rebind[Scalar[DTYPE]](k)
            var R_eq = (Scalar[DTYPE](1.0) - imp) / imp * diag_eq
            eq_inv_K_imp[num_eq_rows] = Scalar[DTYPE](1.0) / (rebind[Scalar[DTYPE]](k) + R_eq)

            num_eq_rows += 1

        # --- 3 orientation rows (weld only) ---
        if eq_type == EQ_WELD:
            # Read relpose
            var rp_x = rebind[Scalar[DTYPE]](
                model[0, eq_off + EQ_IDX_RELPOSE_X]
            )
            var rp_y = rebind[Scalar[DTYPE]](
                model[0, eq_off + EQ_IDX_RELPOSE_Y]
            )
            var rp_z = rebind[Scalar[DTYPE]](
                model[0, eq_off + EQ_IDX_RELPOSE_Z]
            )
            var rp_w = rebind[Scalar[DTYPE]](
                model[0, eq_off + EQ_IDX_RELPOSE_W]
            )

            # Compute orientation error: 0.5 * imag(conj(quat_b) * quat_a * relpose)
            var qa_x = xquat_a_x
            var qa_y = xquat_a_y
            var qa_z = xquat_a_z
            var qa_w = xquat_a_w

            var qb_x: Scalar[DTYPE]
            var qb_y: Scalar[DTYPE]
            var qb_z: Scalar[DTYPE]
            var qb_w: Scalar[DTYPE]
            if body_b > 0:
                qb_x = rebind[Scalar[DTYPE]](
                    state[env, xquat_off + body_b * 4 + 0]
                )
                qb_y = rebind[Scalar[DTYPE]](
                    state[env, xquat_off + body_b * 4 + 1]
                )
                qb_z = rebind[Scalar[DTYPE]](
                    state[env, xquat_off + body_b * 4 + 2]
                )
                qb_w = rebind[Scalar[DTYPE]](
                    state[env, xquat_off + body_b * 4 + 3]
                )
            else:
                qb_x = Scalar[DTYPE](0)
                qb_y = Scalar[DTYPE](0)
                qb_z = Scalar[DTYPE](0)
                qb_w = Scalar[DTYPE](1)

            # conj(qb) * qa
            var cqb = quat_conjugate[DTYPE](qb_x, qb_y, qb_z, qb_w)
            var temp = quat_mul[DTYPE](
                cqb[0], cqb[1], cqb[2], cqb[3], qa_x, qa_y, qa_z, qa_w
            )
            # * relpose
            var err_q = quat_mul[DTYPE](
                temp[0], temp[1], temp[2], temp[3], rp_x, rp_y, rp_z, rp_w
            )
            # 0.5 * imaginary part
            var rot_errs = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
            rot_errs[0] = Scalar[DTYPE](0.5) * err_q[0]
            rot_errs[1] = Scalar[DTYPE](0.5) * err_q[1]
            rot_errs[2] = Scalar[DTYPE](0.5) * err_q[2]

            for d in range(3):
                if num_eq_rows >= MAX_EQ_ROWS:
                    break
                var dx = dirs[d * 3 + 0]
                var dy = dirs[d * 3 + 1]
                var dz = dirs[d * 3 + 2]

                # Angular Jacobian
                for i in range(V_SIZE):
                    J_row[i] = 0
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

                # K, store J and MinvJ
                var k: Scalar[DTYPE] = 0
                var v_n: Scalar[DTYPE] = 0
                for i in range(NV):
                    eq_J[num_eq_rows * NV + i] = J_row[i]
                    var mi_j_sum: Scalar[DTYPE] = 0
                    for j_idx in range(NV):
                        mi_j_sum += (
                            rebind[Scalar[DTYPE]](
                                workspace[env, M_inv_idx + i * NV + j_idx]
                            )
                            * J_row[j_idx]
                        )
                    eq_MinvJ[num_eq_rows * NV + i] = mi_j_sum
                    k += J_row[i] * mi_j_sum
                    v_n += J_row[i] * rebind[Scalar[DTYPE]](
                        state[env, qvel_off + i]
                    )

                if k < Scalar[DTYPE](1e-10):
                    k = Scalar[DTYPE](1e-10)
                eq_K[num_eq_rows] = k

                # Impedance for orientation: MuJoCo piecewise power formula
                var err_d = rot_errs[d]
                var penetration = abs(err_d)
                var imp: Scalar[DTYPE]
                if si_dmin == si_dmax or si_width <= Scalar[DTYPE](0):
                    imp = Scalar[DTYPE](0.5) * (si_dmin + si_dmax)
                else:
                    var x = penetration / si_width
                    var y: Scalar[DTYPE]
                    if x <= Scalar[DTYPE](0):
                        y = Scalar[DTYPE](0)
                    elif x >= Scalar[DTYPE](1):
                        y = Scalar[DTYPE](1)
                    elif si_power == Scalar[DTYPE](1):
                        y = x
                    elif x <= si_midpoint:
                        var a = Scalar[DTYPE](1) / pow(si_midpoint, si_power - Scalar[DTYPE](1))
                        y = a * pow(x, si_power)
                    else:
                        var b = Scalar[DTYPE](1) / pow(Scalar[DTYPE](1) - si_midpoint, si_power - Scalar[DTYPE](1))
                        y = Scalar[DTYPE](1) - b * pow(Scalar[DTYPE](1) - x, si_power)
                    imp = si_dmin + y * (si_dmax - si_dmin)
                if imp < Scalar[DTYPE](1e-6):
                    imp = Scalar[DTYPE](1e-6)

                var bias = -eq_K_spring * imp * penetration + eq_B_damp * v_n
                if err_d < Scalar[DTYPE](0):
                    bias = -bias
                eq_bias[num_eq_rows] = bias
                # MuJoCo: R = (1-imp)/imp * diagApprox (rotation weights)
                comptime eq_rot_bw_off = model_body_invweight0_offset[
                    NBODY, NJOINT, NGEOM, MAX_EQUALITY
                ]()
                var diag_rot: Scalar[DTYPE] = 0
                if body_a > 0 and body_a < NBODY:
                    diag_rot += rebind[Scalar[DTYPE]](
                        model[0, eq_rot_bw_off + body_a * 2 + 1]
                    )
                if body_b > 0 and body_b < NBODY:
                    diag_rot += rebind[Scalar[DTYPE]](
                        model[0, eq_rot_bw_off + body_b * 2 + 1]
                    )
                if diag_rot < Scalar[DTYPE](1e-10):
                    diag_rot = rebind[Scalar[DTYPE]](k)
                var R_rot = (Scalar[DTYPE](1.0) - imp) / imp * diag_rot
                eq_inv_K_imp[num_eq_rows] = Scalar[DTYPE](1.0) / (rebind[Scalar[DTYPE]](k) + R_rot)

                num_eq_rows += 1

    if num_eq_rows == 0:
        return

    # Bilateral PGS iterations (no clamping)
    for _ in range(NUM_ITERATIONS):
        var max_delta: Scalar[DTYPE] = 0
        for r in range(num_eq_rows):
            # a_eq = J @ qacc
            var a_eq: Scalar[DTYPE] = 0
            for i in range(NV):
                a_eq += eq_J[r * NV + i] * rebind[Scalar[DTYPE]](
                    workspace[env, qacc_idx + i]
                )

            var R_eq = Scalar[DTYPE](1.0) / eq_inv_K_imp[r] - eq_K[r]
            var residual = a_eq + eq_bias[r] + R_eq * eq_lambda[r]
            var delta = -residual * eq_inv_K_imp[r]
            var old_lambda = eq_lambda[r]
            eq_lambda[r] = eq_lambda[r] + delta
            # Bilateral: no clamping (force can push or pull)
            var actual = eq_lambda[r] - old_lambda
            var abs_d = abs(actual)
            if abs_d > max_delta:
                max_delta = abs_d
            # qacc += MinvJ * delta
            for i in range(NV):
                workspace[env, qacc_idx + i] = (
                    rebind[Scalar[DTYPE]](workspace[env, qacc_idx + i])
                    + eq_MinvJ[r * NV + i] * actual
                )

        if max_delta < Scalar[DTYPE](1e-4):
            break


# =============================================================================
# 7. build_and_solve_tendon_gpu
# =============================================================================


@always_inline
fn build_and_solve_tendon_gpu[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    MAX_EQUALITY: Int,
    NGEOM: Int,
    MAX_TENDON: Int,
    STATE_SIZE: Int,
    MODEL_SIZE: Int,
    V_SIZE: Int,
    WS_SIZE: Int,
    BATCH: Int,
    NUM_ITERATIONS: Int,
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    workspace: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ],
):
    """Build and solve fixed tendon equality constraints on GPU.

    Called in SEQUENTIAL phase (thread 0 only). Reads tendon definitions
    from model buffer, computes trivial Jacobians (J[dof_adr] = coef),
    impedance, and runs bilateral PGS iterations.

    A fixed tendon is: ten_length = Σ(coef_i * qpos[joint_qposadr_i]).
    Equality constraint: ten_length - length_ref = 0.
    """

    @parameter
    if MAX_TENDON == 0:
        return

    comptime qacc_idx = ws_qacc_constrained_offset[NV, NBODY]()
    comptime M_inv_idx = ws_m_inv_offset[NV, NBODY]()
    comptime model_meta_off = model_metadata_offset[NBODY, NJOINT]()
    comptime qpos_off = qpos_offset[NQ, NV]()
    comptime qvel_off = qvel_offset[NQ, NV]()

    # Read number of tendons from model metadata
    var nten = Int(
        rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_NTENDON]
        )
    )
    if nten == 0:
        return
    if nten > MAX_TENDON:
        nten = MAX_TENDON

    # One bilateral row per tendon
    comptime MAX_TEN_ROWS = _max_one[MAX_TENDON]()
    comptime MINVJ_TEN_SIZE = _max_one[MAX_TENDON * NV]()

    var ten_K = InlineArray[Scalar[DTYPE], MAX_TEN_ROWS](fill=Scalar[DTYPE](1))
    var ten_bias = InlineArray[Scalar[DTYPE], MAX_TEN_ROWS](
        fill=Scalar[DTYPE](0)
    )
    var ten_inv_K_imp = InlineArray[Scalar[DTYPE], MAX_TEN_ROWS](
        fill=Scalar[DTYPE](0)
    )
    var ten_lambda = InlineArray[Scalar[DTYPE], MAX_TEN_ROWS](
        fill=Scalar[DTYPE](0)
    )
    var ten_J = InlineArray[Scalar[DTYPE], MINVJ_TEN_SIZE](
        fill=Scalar[DTYPE](0)
    )
    var ten_MinvJ = InlineArray[Scalar[DTYPE], MINVJ_TEN_SIZE](
        fill=Scalar[DTYPE](0)
    )

    var num_ten_rows = 0

    for t_i in range(nten):
        if num_ten_rows >= MAX_TEN_ROWS:
            break

        var t_off = model_tendon_offset[NBODY, NJOINT, NGEOM, MAX_EQUALITY](
            t_i
        )
        var num_joints = Int(
            rebind[Scalar[DTYPE]](model[0, t_off + TENDON_IDX_NUM_JOINTS])
        )
        var length_ref = rebind[Scalar[DTYPE]](
            model[0, t_off + TENDON_IDX_LENGTH_REF]
        )

        # Read joint indices and coefficients (up to 4)
        var joint_idxs = InlineArray[Int, 4](fill=-1)
        var coefs = InlineArray[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
        joint_idxs[0] = Int(
            rebind[Scalar[DTYPE]](model[0, t_off + TENDON_IDX_JOINT_0])
        )
        joint_idxs[1] = Int(
            rebind[Scalar[DTYPE]](model[0, t_off + TENDON_IDX_JOINT_1])
        )
        joint_idxs[2] = Int(
            rebind[Scalar[DTYPE]](model[0, t_off + TENDON_IDX_JOINT_2])
        )
        joint_idxs[3] = Int(
            rebind[Scalar[DTYPE]](model[0, t_off + TENDON_IDX_JOINT_3])
        )
        coefs[0] = rebind[Scalar[DTYPE]](model[0, t_off + TENDON_IDX_COEF_0])
        coefs[1] = rebind[Scalar[DTYPE]](model[0, t_off + TENDON_IDX_COEF_1])
        coefs[2] = rebind[Scalar[DTYPE]](model[0, t_off + TENDON_IDX_COEF_2])
        coefs[3] = rebind[Scalar[DTYPE]](model[0, t_off + TENDON_IDX_COEF_3])

        # Compute tendon length and velocity, build trivial Jacobian
        var ten_length: Scalar[DTYPE] = 0
        var ten_vel: Scalar[DTYPE] = 0
        var r = num_ten_rows

        for ji in range(4):
            if ji >= num_joints:
                break
            var jnt_idx = joint_idxs[ji]
            if jnt_idx < 0 or jnt_idx >= NJOINT:
                continue
            # Read joint's qpos_adr and dof_adr from model buffer
            var j_off = model_joint_offset[NBODY](jnt_idx)
            var qpos_adr = Int(
                rebind[Scalar[DTYPE]](model[0, j_off + JOINT_IDX_QPOS_ADR])
            )
            var dof_adr = Int(
                rebind[Scalar[DTYPE]](model[0, j_off + JOINT_IDX_DOF_ADR])
            )
            var c = coefs[ji]
            ten_length += c * rebind[Scalar[DTYPE]](
                state[env, qpos_off + qpos_adr]
            )
            ten_vel += c * rebind[Scalar[DTYPE]](
                state[env, qvel_off + dof_adr]
            )
            # Trivial Jacobian: J[dof_adr] = coef
            ten_J[r * NV + dof_adr] = c

        # Tendon position error (bilateral)
        var pos_err = ten_length - length_ref

        # Compute K = J @ M_inv @ J^T and MinvJ
        var k: Scalar[DTYPE] = 0
        for i in range(NV):
            var mi_j_sum: Scalar[DTYPE] = 0
            for j_idx in range(NV):
                mi_j_sum += (
                    rebind[Scalar[DTYPE]](
                        workspace[env, M_inv_idx + i * NV + j_idx]
                    )
                    * ten_J[r * NV + j_idx]
                )
            ten_MinvJ[r * NV + i] = mi_j_sum
            k += ten_J[r * NV + i] * mi_j_sum

        if k < Scalar[DTYPE](1e-10):
            k = Scalar[DTYPE](1e-10)
        ten_K[r] = k

        # Read solref/solimp
        var sr_tc = rebind[Scalar[DTYPE]](model[0, t_off + TENDON_IDX_SOLREF_0])
        var sr_dr = rebind[Scalar[DTYPE]](model[0, t_off + TENDON_IDX_SOLREF_1])
        var si_dmin = rebind[Scalar[DTYPE]](
            model[0, t_off + TENDON_IDX_SOLIMP_0]
        )
        var si_dmax = rebind[Scalar[DTYPE]](
            model[0, t_off + TENDON_IDX_SOLIMP_1]
        )
        var si_width = rebind[Scalar[DTYPE]](
            model[0, t_off + TENDON_IDX_SOLIMP_2]
        )
        var si_midpoint = rebind[Scalar[DTYPE]](
            model[0, t_off + TENDON_IDX_SOLIMP_3]
        )
        var si_power = rebind[Scalar[DTYPE]](
            model[0, t_off + TENDON_IDX_SOLIMP_4]
        )
        if si_width < Scalar[DTYPE](1e-6):
            si_width = Scalar[DTYPE](1e-6)
        if si_dmax < Scalar[DTYPE](1e-4):
            si_dmax = Scalar[DTYPE](1e-4)
        var t_K_spring = Scalar[DTYPE](1.0) / (
            sr_tc * sr_tc * si_dmax * si_dmax
        )
        var t_B_damp = Scalar[DTYPE](2.0) * sr_dr / (sr_tc * si_dmax)

        # Impedance: MuJoCo piecewise power formula on |pos_err|
        var penetration = abs(pos_err)
        var imp: Scalar[DTYPE]
        if si_dmin == si_dmax or si_width <= Scalar[DTYPE](0):
            imp = Scalar[DTYPE](0.5) * (si_dmin + si_dmax)
        else:
            var x = penetration / si_width
            var y: Scalar[DTYPE]
            if x <= Scalar[DTYPE](0):
                y = Scalar[DTYPE](0)
            elif x >= Scalar[DTYPE](1):
                y = Scalar[DTYPE](1)
            elif si_power == Scalar[DTYPE](1):
                y = x
            elif x <= si_midpoint:
                var a = Scalar[DTYPE](1) / pow(si_midpoint, si_power - Scalar[DTYPE](1))
                y = a * pow(x, si_power)
            else:
                var b = Scalar[DTYPE](1) / pow(Scalar[DTYPE](1) - si_midpoint, si_power - Scalar[DTYPE](1))
                y = Scalar[DTYPE](1) - b * pow(Scalar[DTYPE](1) - x, si_power)
            imp = si_dmin + y * (si_dmax - si_dmin)
        if imp < Scalar[DTYPE](1e-6):
            imp = Scalar[DTYPE](1e-6)

        # bias = -aref (bilateral: sign depends on error direction)
        var bias = -t_K_spring * imp * penetration + t_B_damp * ten_vel
        if pos_err < Scalar[DTYPE](0):
            bias = -bias
        ten_bias[r] = bias

        # R = (1-imp)/imp * diagApprox (sum of dof_invweight0 for tendon joints)
        comptime dw_off = model_dof_invweight0_offset[
            NBODY, NJOINT, NGEOM, MAX_EQUALITY
        ]()
        var diag_ten: Scalar[DTYPE] = 0
        for ji in range(4):
            if ji >= num_joints:
                break
            var jnt_idx = joint_idxs[ji]
            if jnt_idx < 0 or jnt_idx >= NJOINT:
                continue
            var j_off = model_joint_offset[NBODY](jnt_idx)
            var dof_adr = Int(
                rebind[Scalar[DTYPE]](model[0, j_off + JOINT_IDX_DOF_ADR])
            )
            diag_ten += rebind[Scalar[DTYPE]](model[0, dw_off + dof_adr])
        if diag_ten < Scalar[DTYPE](1e-10):
            diag_ten = k  # Fallback to exact K
        var R_ten = (Scalar[DTYPE](1.0) - imp) / imp * diag_ten
        ten_inv_K_imp[r] = Scalar[DTYPE](1.0) / (k + R_ten)

        num_ten_rows += 1

    if num_ten_rows == 0:
        return

    # Bilateral PGS iterations (no clamping — bilateral constraint)
    for _ in range(NUM_ITERATIONS):
        var max_delta: Scalar[DTYPE] = 0
        for r in range(num_ten_rows):
            # a_ten = J @ qacc
            var a_ten: Scalar[DTYPE] = 0
            for i in range(NV):
                a_ten += ten_J[r * NV + i] * rebind[Scalar[DTYPE]](
                    workspace[env, qacc_idx + i]
                )

            var R_ten = Scalar[DTYPE](1.0) / ten_inv_K_imp[r] - ten_K[r]
            var residual = a_ten + ten_bias[r] + R_ten * ten_lambda[r]
            var delta = -residual * ten_inv_K_imp[r]
            var old_lambda = ten_lambda[r]
            ten_lambda[r] = ten_lambda[r] + delta
            # Bilateral: no clamping
            var actual = ten_lambda[r] - old_lambda
            var abs_d = abs(actual)
            if abs_d > max_delta:
                max_delta = abs_d
            # qacc += MinvJ * delta
            for i in range(NV):
                workspace[env, qacc_idx + i] = (
                    rebind[Scalar[DTYPE]](workspace[env, qacc_idx + i])
                    + ten_MinvJ[r * NV + i] * actual
                )

        if max_delta < Scalar[DTYPE](1e-4):
            break

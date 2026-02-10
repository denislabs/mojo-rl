from layout import Layout, LayoutTensor
from math import sqrt
from ..dynamics.jacobian import (
    compute_contact_jacobian_row,
    compute_contact_jacobian_row_gpu,
)
from ..gpu.constants import (
    contacts_offset,
    metadata_offset,
    model_metadata_offset,
    model_joint_offset,
    ws_cdof_offset,
    ws_qvel_pred_offset,
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
)
from ..joint_types import (
    JNT_HINGE,
    JNT_SLIDE,
)

from ..gpu.constants import (
    CONTACT_SIZE,
    CONTACT_IDX_IMPULSE_N,
    CONTACT_IDX_IMPULSE_T1,
    CONTACT_IDX_IMPULSE_T2,
)

# Friction uses PGS iterations
comptime FRICTION_PGS_ITERATIONS: Int = 30

# =============================================================================
# Shared friction solver (PGS) - CPU
# =============================================================================


fn _solve_friction_pgs_cpu[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    V_SIZE: Int,
    M_SIZE: Int,
    CDOF_SIZE: Int,
](
    model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    cdof: InlineArray[Scalar[DTYPE], CDOF_SIZE],
    M_inv: InlineArray[Scalar[DTYPE], M_SIZE],
    J_n: InlineArray[Scalar[DTYPE], _max_one[MAX_CONTACTS * NV]()],
    lambda_n: InlineArray[Scalar[DTYPE], _max_one[MAX_CONTACTS]()],
    contact_dist: InlineArray[Scalar[DTYPE], _max_one[MAX_CONTACTS]()],
    contact_body_b: InlineArray[Int, _max_one[MAX_CONTACTS]()],
    nc: Int,
    mut qvel: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """Friction solver using PGS (shared by CG and Newton solvers)."""
    var friction_coef = model.friction
    comptime MC = _max_one[MAX_CONTACTS]()
    comptime JT_SIZE = _max_one[MAX_CONTACTS * NV]()

    var J_t1_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var J_t2_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

    var lambda_t1 = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    var lambda_t2 = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    for i in range(MC):
        lambda_t1[i] = Scalar[DTYPE](0)
        lambda_t2[i] = Scalar[DTYPE](0)

    var J_t1_all = InlineArray[Scalar[DTYPE], JT_SIZE](uninitialized=True)
    var J_t2_all = InlineArray[Scalar[DTYPE], JT_SIZE](uninitialized=True)
    var K_t1 = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    var K_t2 = InlineArray[Scalar[DTYPE], MC](uninitialized=True)

    for i in range(JT_SIZE):
        J_t1_all[i] = Scalar[DTYPE](0)
        J_t2_all[i] = Scalar[DTYPE](0)
    for i in range(MC):
        K_t1[i] = Scalar[DTYPE](1)
        K_t2[i] = Scalar[DTYPE](1)

    for c in range(nc):
        if lambda_n[c] <= Scalar[DTYPE](0):
            continue

        var contact = data.contacts[c]
        var nx = contact.normal_x
        var ny = contact.normal_y
        var nz = contact.normal_z

        # Compute tangent basis
        var t1_x: Scalar[DTYPE]
        var t1_y: Scalar[DTYPE]
        var t1_z: Scalar[DTYPE]

        if abs(nx) < Scalar[DTYPE](0.9):
            t1_x = Scalar[DTYPE](0)
            t1_y = -nz
            t1_z = ny
        else:
            t1_x = nz
            t1_y = Scalar[DTYPE](0)
            t1_z = -nx

        var t1_mag = sqrt(t1_x * t1_x + t1_y * t1_y + t1_z * t1_z)
        if t1_mag > Scalar[DTYPE](1e-10):
            t1_x = t1_x / t1_mag
            t1_y = t1_y / t1_mag
            t1_z = t1_z / t1_mag

        var t2_x = ny * t1_z - nz * t1_y
        var t2_y = nz * t1_x - nx * t1_z
        var t2_z = nx * t1_y - ny * t1_x

        # Compute tangent Jacobian rows
        compute_contact_jacobian_row[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE, CDOF_SIZE
        ](
            model,
            data,
            cdof,
            contact.body_a,
            contact_body_b[c],
            contact.pos_x,
            contact.pos_y,
            contact.pos_z,
            t1_x,
            t1_y,
            t1_z,
            J_t1_row,
        )
        compute_contact_jacobian_row[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE, CDOF_SIZE
        ](
            model,
            data,
            cdof,
            contact.body_a,
            contact_body_b[c],
            contact.pos_x,
            contact.pos_y,
            contact.pos_z,
            t2_x,
            t2_y,
            t2_z,
            J_t2_row,
        )

        var k1: Scalar[DTYPE] = 0
        var k2: Scalar[DTYPE] = 0
        for i in range(NV):
            J_t1_all[c * NV + i] = J_t1_row[i]
            J_t2_all[c * NV + i] = J_t2_row[i]
            var mi_j_sum1: Scalar[DTYPE] = 0
            var mi_j_sum2: Scalar[DTYPE] = 0
            for j_idx in range(NV):
                mi_j_sum1 += M_inv[i * NV + j_idx] * J_t1_row[j_idx]
                mi_j_sum2 += M_inv[i * NV + j_idx] * J_t2_row[j_idx]
            k1 += J_t1_row[i] * mi_j_sum1
            k2 += J_t2_row[i] * mi_j_sum2

        if k1 < Scalar[DTYPE](1e-10):
            k1 = Scalar[DTYPE](1e-10)
        if k2 < Scalar[DTYPE](1e-10):
            k2 = Scalar[DTYPE](1e-10)
        K_t1[c] = k1
        K_t2[c] = k2

        # Warm start tangent impulses
        lambda_t1[c] = contact.impulse_t1
        lambda_t2[c] = contact.impulse_t2

    # Apply tangent warm start
    for c in range(nc):
        if lambda_n[c] <= Scalar[DTYPE](0):
            continue
        if lambda_t1[c] != Scalar[DTYPE](0) or lambda_t2[c] != Scalar[DTYPE](0):
            for i in range(NV):
                var mi_j_sum1: Scalar[DTYPE] = 0
                var mi_j_sum2: Scalar[DTYPE] = 0
                for j_idx in range(NV):
                    mi_j_sum1 += (
                        M_inv[i * NV + j_idx] * J_t1_all[c * NV + j_idx]
                    )
                    mi_j_sum2 += (
                        M_inv[i * NV + j_idx] * J_t2_all[c * NV + j_idx]
                    )
                qvel[i] += mi_j_sum1 * lambda_t1[c] + mi_j_sum2 * lambda_t2[c]

    # Friction PGS iterations
    for _ in range(FRICTION_PGS_ITERATIONS):
        for c in range(nc):
            if lambda_n[c] <= Scalar[DTYPE](0):
                continue

            var max_friction = friction_coef * lambda_n[c]

            var v_t1: Scalar[DTYPE] = 0
            for i in range(NV):
                v_t1 += J_t1_all[c * NV + i] * qvel[i]
            var delta_t1 = -v_t1 / K_t1[c]
            var old_t1 = lambda_t1[c]
            lambda_t1[c] = lambda_t1[c] + delta_t1

            var v_t2: Scalar[DTYPE] = 0
            for i in range(NV):
                v_t2 += J_t2_all[c * NV + i] * qvel[i]
            var delta_t2 = -v_t2 / K_t2[c]
            var old_t2 = lambda_t2[c]
            lambda_t2[c] = lambda_t2[c] + delta_t2

            # Coulomb cone clamping
            var t_mag = sqrt(
                lambda_t1[c] * lambda_t1[c] + lambda_t2[c] * lambda_t2[c]
            )
            if t_mag > max_friction:
                var scale = max_friction / t_mag
                lambda_t1[c] = lambda_t1[c] * scale
                lambda_t2[c] = lambda_t2[c] * scale

            var actual_delta_t1 = lambda_t1[c] - old_t1
            var actual_delta_t2 = lambda_t2[c] - old_t2

            for i in range(NV):
                var mi_j_sum1: Scalar[DTYPE] = 0
                var mi_j_sum2: Scalar[DTYPE] = 0
                for j_idx in range(NV):
                    mi_j_sum1 += (
                        M_inv[i * NV + j_idx] * J_t1_all[c * NV + j_idx]
                    )
                    mi_j_sum2 += (
                        M_inv[i * NV + j_idx] * J_t2_all[c * NV + j_idx]
                    )
                qvel[i] += (
                    mi_j_sum1 * actual_delta_t1 + mi_j_sum2 * actual_delta_t2
                )

    # Store impulses back for warm-starting next step
    for c in range(nc):
        data.contacts[c].impulse_n = lambda_n[c]
        data.contacts[c].impulse_t1 = lambda_t1[c]
        data.contacts[c].impulse_t2 = lambda_t2[c]


# =============================================================================
# Shared friction solver (PGS) - GPU
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

    comptime qvel_idx = ws_qvel_pred_offset[NV, NBODY]()

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

        # Warm start tangent impulses
        var c_off = contacts_off + c * CONTACT_SIZE
        workspace[env, lt1 + c] = rebind[Scalar[DTYPE]](
            state[env, c_off + CONTACT_IDX_IMPULSE_T1]
        )
        workspace[env, lt2 + c] = rebind[Scalar[DTYPE]](
            state[env, c_off + CONTACT_IDX_IMPULSE_T2]
        )

    # Friction PGS iterations (using cached J_t1/J_t2 and MinvJt1/MinvJt2)
    for _ in range(FRICTION_PGS_ITERATIONS):
        var max_delta: workspace.element_type = 0
        for c in range(nc):
            if workspace[env, ws_lambda_n + c] <= Scalar[DTYPE](0):
                continue

            var max_friction = friction_coef * workspace[env, ws_lambda_n + c]

            # Tangent 1: v_t1 = J_t1[c] . qvel
            var v_t1: workspace.element_type = 0
            for i in range(NV):
                v_t1 += workspace[env, ws_J_t1 + c * NV + i] * workspace[env, qvel_idx + i]

            var delta_t1 = -v_t1 / workspace[env, kt1 + c]
            var old_t1 = workspace[env, lt1 + c]
            workspace[env, lt1 + c] = workspace[env, lt1 + c] + delta_t1

            # Tangent 2: v_t2 = J_t2[c] . qvel
            var v_t2: workspace.element_type = 0
            for i in range(NV):
                v_t2 += workspace[env, ws_J_t2 + c * NV + i] * workspace[env, qvel_idx + i]

            var delta_t2 = -v_t2 / workspace[env, kt2 + c]
            var old_t2 = workspace[env, lt2 + c]
            workspace[env, lt2 + c] = workspace[env, lt2 + c] + delta_t2

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

            # Track max delta for early exit
            var abs_t1 = abs(actual_t1)
            var abs_t2 = abs(actual_t2)
            if abs_t1 > max_delta:
                max_delta = abs_t1
            if abs_t2 > max_delta:
                max_delta = abs_t2

            # Apply velocity correction: qvel += MinvJt1 * actual_t1 + MinvJt2 * actual_t2
            for i in range(NV):
                workspace[env, qvel_idx + i] += (
                    workspace[env, ws_MinvJt1 + c * NV + i] * actual_t1
                    + workspace[env, ws_MinvJt2 + c * NV + i] * actual_t2
                )

        # Early exit if converged
        if max_delta < Scalar[DTYPE](1e-4):
            break

    # Store impulses back for warm-starting
    for c in range(nc):
        var c_off = contacts_off + c * CONTACT_SIZE
        state[env, c_off + CONTACT_IDX_IMPULSE_N] = workspace[
            env, ws_lambda_n + c
        ]
        state[env, c_off + CONTACT_IDX_IMPULSE_T1] = workspace[env, lt1 + c]
        state[env, c_off + CONTACT_IDX_IMPULSE_T2] = workspace[env, lt2 + c]

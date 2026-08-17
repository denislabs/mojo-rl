"""Island-aware PGS contact solve over per-field tensors (Stage S-ISL,
single-source).

Per-field port of `IslandPGSSolver.solve_gpu` (solver/island_pgs_solver.mojo).
IslandPGS = plain PGS + body union-find island partition + per-island early
termination: once an island's max |Δλ| falls below ISLAND_CONVERGE_EPS the
island is frozen (its contacts are skipped in subsequent PGS iterations),
cutting total iterations for multi-body / multi-island scenes.

This is a VERBATIM copy of the fields PGS solver `_contact_solve_env`
(constraints/contact_solve.mojo — itself the golden-validated fields
port of PGSSolver.solve_gpu) with only the island machinery inserted at the
legacy positions (island_pgs_solver.mojo:388-512, 890-1574):
  A. island tracking state (contact_island / island_converged / counts),
  B. body union-find partition (path-halving find + union) before warmstart,
  C. per-island freeze in the NORMAL PGS loop (skip converged islands, track
     per-island max |Δλ_n|, freeze when < eps; reset flags for the coupled
     phase),
  D. per-island freeze in the COUPLED PGS loop (same guards; the delta metric
     is the normal-update |Δλ_n|, matching the legacy island coupled loop).
The per-contact PGS/QCQP arithmetic (both ELLIPTIC and PYRAMIDAL cones) and
the limits/equality/tendon tail are byte-identical to the fields PGS solver —
only freeze guards + delta tracking + island termination are added. It
reuses the shared contact_solve helpers (init/normal-precompute/
warmstart/jacobian rows) directly.

Serialized per env (one thread per env on GPU) like the other fields solvers.
Same signature family as `solve_contacts` so callers can swap solvers.
"""

from std.math import sqrt, pow, abs
from std.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from ..types import _max_one, ConeType
from ..joint_types import JNT_FREE, JNT_BALL
from ..constraints.qcqp import mj_qcqp2, mj_qcqp3, mj_qcqp5
from ..constraints.limits import _limits_env
from ..constraints.friction_dof import _friction_env
from ..constraints.equality_tendon import (
    _equality_env,
    _tendon_env,
)
from ..dynamics.jac_contact_row import _contact_jacobian_row
from ..constraints.contact_solve import (
    _angular_jacobian_row,
    _init_common_normal_ws,
    _precompute_contact_normal,
    _warmstart_normals,
)
# Island constants (relocated here at the P6 legacy sunset; formerly imported
# from the deleted legacy `island_detection` / `island_solver`).
from ..constraints.constraint_data import solref_spring_damper

comptime MAX_ISLANDS: Int = 64
comptime ISLAND_CONVERGE_EPS: Float64 = 1e-6
from ..fields import Data, Model, DynamicsScratch, ContactScratch, Dims, DimsLike
from ..gpu.constants import (
    MODEL_META_IDX_TIMESTEP,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_EQ_SIZE,
    MODEL_TENDON_SIZE,
    MODEL_SITE_SIZE,
    METADATA_SIZE,
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
    CONTACT_IDX_INCLUDEMARGIN,
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
    MODEL_META_IDX_NJOINT,
    MODEL_META_IDX_SOLREF_CONTACT_0,
    MODEL_META_IDX_SOLREF_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_0,
    MODEL_META_IDX_SOLIMP_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_2,
    MODEL_META_IDX_SOLIMP_CONTACT_3,
    MODEL_META_IDX_SOLIMP_CONTACT_4,
    MODEL_META_IDX_IMPRATIO,
    BODY_IDX_PARENT,
    BODY_IDX_ROOTID,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_DOF_ADR,
)
from ..collision.contact_frame import contact_tangent_frame

comptime CS_TPB: Int = 64

# PGS solver parameters (replicated from solver/pgs_solver.mojo:83)
comptime PGS_ITERATIONS: Int = 100
# Minimum K for friction tangent rows — below this, direction is degenerate
comptime FRICTION_K_MIN: Float64 = 1e-6


# === BODY (copied verbatim from contact_solve _contact_solve_env
# + _contact_solve_fields_kernel + solve_contacts, renamed, with island
# insertions A-D) ===
@always_inline
def _island_pgs_solve_env[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int,
    NEQUALITY: Int,
    NTENDON: Int,
    NSITE: Int,
    CONE_TYPE: Int,
    BATCH: Int,
    SOLVER_WS: Int,
](
    env: Int,
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    xpos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    subtree_com: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    contacts: LayoutTensor[
        DTYPE,
        Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE),
        MutAnyOrigin,
    ],
    smeta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, METADATA_SIZE), MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    mmeta: LayoutTensor[
        DTYPE, Layout.row_major(MODEL_META_SIZE), MutAnyOrigin
    ],
    equality: LayoutTensor[
        DTYPE, Layout.row_major(NEQUALITY, MODEL_EQ_SIZE), MutAnyOrigin
    ],
    tendons: LayoutTensor[
        DTYPE, Layout.row_major(NTENDON, MODEL_TENDON_SIZE), MutAnyOrigin
    ],
    sites: LayoutTensor[
        DTYPE, Layout.row_major(NSITE, MODEL_SITE_SIZE), MutAnyOrigin
    ],
    body_invweight0: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, 2), MutAnyOrigin
    ],
    dof_invweight0: LayoutTensor[DTYPE, Layout.row_major(NV), MutAnyOrigin],
    cdof: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * 6), MutAnyOrigin],
    m_inv: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin
    ],
    qacc_constrained: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin
    ],
    solver: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, SOLVER_WS), MutAnyOrigin
    ],
):
    """Full PGS contact solve for one env (verbatim from PGSSolver.solve_gpu,
    serialized per env — see module docstring; limits/equality/tendon run at
    the legacy position via their per-env fields ports)."""
    comptime MC = _max_one[MAX_CONTACTS]()
    comptime V_SIZE = _max_one[NV]()

    # Common normal block offsets (for PGS normal iterations)
    comptime ws_lambda_n = 0 * MC
    comptime ws_K_n = 1 * MC
    comptime ws_c_dist = 2 * MC
    comptime ws_c_body = 3 * MC
    comptime ws_c_body_b = 4 * MC
    comptime ws_c_px = 5 * MC
    comptime ws_c_py = 6 * MC
    comptime ws_c_pz = 7 * MC
    comptime ws_c_nx = 8 * MC
    comptime ws_c_ny = 9 * MC
    comptime ws_c_nz = 10 * MC
    comptime ws_pos_bias = 11 * MC
    comptime ws_inv_K_imp = 12 * MC
    comptime ws_J_n = 15 * MC
    comptime ws_MinvJn = 15 * MC + MC * NV

    # Friction workspace offsets (66*MC + 10*MC*NV, same layout as friction_solver.mojo)
    comptime fws = 15 * MC + 2 * MC * NV
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

    # === Initialize workspace (legacy: parallel, one thread per slot) ===
    for contact_tid in range(MC):
        _init_common_normal_ws[
            DTYPE, NV, MAX_CONTACTS, BATCH, SOLVER_WS
        ](env, contact_tid, solver)
        # Init friction workspace for this contact slot
        for d in range(5):
            solver[env, ws_lf + d * MC + contact_tid] = 0
            solver[env, ws_kf + d * MC + contact_tid] = 1
            solver[env, ws_fc + d * MC + contact_tid] = 0
            solver[env, ws_rf + d * MC + contact_tid] = 0
            solver[env, ws_bf + d * MC + contact_tid] = 0
            # Pyramidal workspace
            solver[env, ws_le_neg + d * MC + contact_tid] = 0
            solver[env, ws_cnt + d * MC + contact_tid] = 0
            solver[env, ws_kep + d * MC + contact_tid] = 1
            solver[env, ws_ken + d * MC + contact_tid] = 1
            solver[env, ws_re + d * MC + contact_tid] = 0
            for axis in range(3):
                solver[env, ws_df + (d * 3 + axis) * MC + contact_tid] = 0
        solver[env, ws_cd + contact_tid] = 3  # default condim=3

    # Read metadata (legacy `dt` read dropped — only the excluded limits
    # call consumed it)
    var nc = 0
    var K_spring: Scalar[DTYPE] = 0
    var B_damp: Scalar[DTYPE] = 0
    var si_dmin: Scalar[DTYPE] = 0
    var si_dmax: Scalar[DTYPE] = 0
    var si_width: Scalar[DTYPE] = 1
    var si_midpoint: Scalar[DTYPE] = Scalar[DTYPE](0.5)
    var si_power: Scalar[DTYPE] = Scalar[DTYPE](2.0)

    nc = Int(rebind[Scalar[DTYPE]](smeta[env, META_IDX_NUM_CONTACTS]))
    if nc > MAX_CONTACTS:
        nc = MAX_CONTACTS
    var sr_tc = rebind[Scalar[DTYPE]](
        mmeta[MODEL_META_IDX_SOLREF_CONTACT_0]
    )
    var sr_dr = rebind[Scalar[DTYPE]](
        mmeta[MODEL_META_IDX_SOLREF_CONTACT_1]
    )
    si_dmin = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLIMP_CONTACT_0])
    si_dmax = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLIMP_CONTACT_1])
    si_width = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLIMP_CONTACT_2])
    si_midpoint = rebind[Scalar[DTYPE]](
        mmeta[MODEL_META_IDX_SOLIMP_CONTACT_3]
    )
    si_power = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLIMP_CONTACT_4])
    if si_width < Scalar[DTYPE](1e-6):
        si_width = Scalar[DTYPE](1e-6)
    # MuJoCo clamps BOTH ends of solimp to [mjMINIMP, mjMAXIMP] before
    # interpolating (engine_core_constraint.c:1284-1287). The dmin floor is
    # the one that bites: R = (1-imp)/imp * diagApprox, so dmin=0 asks for an
    # infinitely soft contact at first touch. dm_control's finger is the first
    # model here to set it (`solimp="0 0.9 0.01"`); everything before used the
    # 0.9 default, which is why clamping only dmax survived.
    comptime MJ_MINIMP = Scalar[DTYPE](0.0001)
    comptime MJ_MAXIMP = Scalar[DTYPE](0.9999)
    if si_dmin < MJ_MINIMP:
        si_dmin = MJ_MINIMP
    elif si_dmin > MJ_MAXIMP:
        si_dmin = MJ_MAXIMP
    if si_dmax < MJ_MINIMP:
        si_dmax = MJ_MINIMP
    elif si_dmax > MJ_MAXIMP:
        si_dmax = MJ_MAXIMP
    if si_power < Scalar[DTYPE](1):
        si_power = Scalar[DTYPE](1)
    # K = 1/(dmax^2 * timeconst^2 * dampratio^2), B = 2/(dmax * timeconst)
    # (engine_core_constraint.c:1432,1440) — the dampratio belongs SQUARED
    # in K and not at all in B. Identical at dampratio=1 (every model in
    # the repo), but the other three solvers already use the MuJoCo form.
    # solref -> (K, B), including MuJoCo's DIRECT form for a NEGATIVE
    # solref. See `constraints/constraint_data.solref_spring_damper` — the
    # formula lived in twelve copy-pasted sites until 2026-08-03.
    (K_spring, B_damp) = solref_spring_damper[DTYPE](
        sr_tc, sr_dr, si_dmax,
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP]),
    )

    # === PHASE 1: normal precompute (legacy: parallel, one thread per
    # contact slot; internal `contact_tid < nc` guard kept in the helper) ===
    for contact_tid in range(MC):
        _precompute_contact_normal[
            DTYPE, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE, BATCH, SOLVER_WS
        ](
            env,
            contact_tid,
            nc,
            qvel,
            subtree_com,
            contacts,
            joints,
            bodies,
            mmeta,
            body_invweight0,
            cdof,
            m_inv,
            qacc_constrained,
            solver,
            K_spring,
            B_damp,
            si_dmin,
            si_dmax,
            si_width,
            si_midpoint,
            si_power,
        )

    # === Island tracking state (INSERTION A; legacy island_pgs_solver:388) ===
    var contact_island = InlineArray[Int, _max_one[MC]()](fill=-1)
    var island_converged = InlineArray[Int, MAX_ISLANDS](fill=0)
    var num_islands = 0
    var num_converged = 0

    # === SEQUENTIAL: island detection + Warm start + PGS normal (thread 0) ===
    # ---- Body union-find: assign each contact to an island (INSERTION B;
    # legacy island_pgs_solver:397-446) ----
    var uf_parent = InlineArray[Int, _max_one[NBODY]()](uninitialized=True)
    for b in range(NBODY):
        uf_parent[b] = b

    for c in range(nc):
        if solver[env, ws_c_dist + c] >= Scalar[DTYPE](0):
            continue
        var ba = Int(solver[env, ws_c_body + c])
        var bb = Int(solver[env, ws_c_body_b + c])
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
        if solver[env, ws_c_dist + c] >= Scalar[DTYPE](0):
            continue
        var ba = Int(solver[env, ws_c_body + c])
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

    _warmstart_normals[DTYPE, NV, MAX_CONTACTS, BATCH, SOLVER_WS](
        env, nc, qacc_constrained, solver
    )

    # PGS normal iterations (acceleration-level) with per-island early
    # termination (INSERTION C; legacy island_pgs_solver:457-512)
    var eps_gpu = Scalar[DTYPE](ISLAND_CONVERGE_EPS)
    for _ in range(PGS_ITERATIONS):
        if num_converged >= num_islands:
            break
        var island_max_delta_n = InlineArray[Scalar[DTYPE], MAX_ISLANDS](
            fill=Scalar[DTYPE](0)
        )
        for c in range(nc):
            if solver[env, ws_c_dist + c] >= Scalar[DTYPE](0):
                continue
            var iid = contact_island[c]
            if iid >= 0 and island_converged[iid] == 1:
                continue
            var a_n: solver.element_type = 0
            for i in range(NV):
                a_n += (
                    solver[env, ws_J_n + c * NV + i]
                    * qacc_constrained[env, i]
                )
            var R_n = Scalar[DTYPE](1.0) / rebind[Scalar[DTYPE]](
                solver[env, ws_inv_K_imp + c]
            ) - rebind[Scalar[DTYPE]](solver[env, ws_K_n + c])
            var residual = (
                a_n
                + solver[env, ws_pos_bias + c]
                + R_n * solver[env, ws_lambda_n + c]
            )
            var delta = -residual * solver[env, ws_inv_K_imp + c]
            var old_lambda = solver[env, ws_lambda_n + c]
            solver[env, ws_lambda_n + c] = (
                solver[env, ws_lambda_n + c] + delta
            )
            if solver[env, ws_lambda_n + c] < Scalar[DTYPE](0):
                solver[env, ws_lambda_n + c] = Scalar[DTYPE](0)
            var actual_delta = solver[env, ws_lambda_n + c] - old_lambda
            var abs_delta = abs(rebind[Scalar[DTYPE]](actual_delta))
            if iid >= 0 and abs_delta > island_max_delta_n[iid]:
                island_max_delta_n[iid] = abs_delta
            for i in range(NV):
                qacc_constrained[env, i] += (
                    solver[env, ws_MinvJn + c * NV + i] * actual_delta
                )
        for iid in range(num_islands):
            if island_converged[iid] == 0:
                if island_max_delta_n[iid] < eps_gpu:
                    island_converged[iid] = 1
                    num_converged += 1

    # Reset island convergence flags for the coupled phase (legacy 509-512)
    for iid in range(num_islands):
        island_converged[iid] = 0
    num_converged = 0

    # Joint limits — legacy position (between the normal PGS and the
    # friction phase), legacy iteration count (PGS_ITERATIONS, not the
    # Newton path's 50).
    _limits_env[DTYPE, NQ, NV, NJOINT, BATCH, PGS_ITERATIONS](
        env, qpos, qvel, joints, mmeta, dof_invweight0, m_inv,
        qacc_constrained,
    )
    # Dry-friction dof rows (MuJoCo mjCNSTR_FRICTION_DOF), solved
    # beside the limit rows. No-op for a model with no frictionloss.
    _friction_env[DTYPE, NQ, NV, NJOINT, BATCH, PGS_ITERATIONS](
        env, qvel, joints,
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP]),
        dof_invweight0, m_inv, qacc_constrained
    )

    # Equality constraints — legacy position (right after limits; the legacy
    # call is unconditional with a comptime gate inside the builder, which
    # this call-site gate matches bit-identically for NEQUALITY == 0).
    comptime if NEQUALITY > 0:
        _equality_env[
            DTYPE, NQ, NV, NBODY, NJOINT, NEQUALITY, V_SIZE,
            BATCH, PGS_ITERATIONS,
        ](
            env, qpos, qvel, xpos, xquat, subtree_com, joints, bodies, mmeta,
            equality, body_invweight0, dof_invweight0, cdof,
            m_inv, qacc_constrained,
        )

    # Tendon equality constraints — legacy call-site gate
    # (`comptime if MAX_TENDON > 0` in PGSSolver.solve_gpu).
    comptime if NTENDON > 0:
        _tendon_env[
            DTYPE, NQ, NV, NBODY, NJOINT, NTENDON, NSITE, BATCH,
            PGS_ITERATIONS,
        ](
            env, qpos, qvel, joints, mmeta, tendons, sites, bodies,
            subtree_com, cdof, xpos, xquat, m_inv, qacc_constrained,
        )

    # === PHASE 3: friction precompute (legacy: parallel, guarded
    # `contact_tid < nc`) ===
    for contact_tid in range(nc):
        var J_row = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(V_SIZE):
            J_row[i] = 0

        var c = contact_tid
        if solver[env, ws_lambda_n + c] > 0:
            var c_off = c * CONTACT_SIZE
            var nx = rebind[Scalar[DTYPE]](solver[env, ws_c_nx + c])
            var ny = rebind[Scalar[DTYPE]](solver[env, ws_c_ny + c])
            var nz = rebind[Scalar[DTYPE]](solver[env, ws_c_nz + c])

            # Read per-contact friction params
            var mu_slide = rebind[Scalar[DTYPE]](
                contacts[env, c_off + CONTACT_IDX_FRICTION]
            )
            if mu_slide <= Scalar[DTYPE](0):
                mu_slide = Scalar[DTYPE](0.5)  # fallback
            var mu_spin = rebind[Scalar[DTYPE]](
                contacts[env, c_off + CONTACT_IDX_FRICTION_SPIN]
            )
            var mu_roll = rebind[Scalar[DTYPE]](
                contacts[env, c_off + CONTACT_IDX_FRICTION_ROLL]
            )
            var condim = Int(
                rebind[Scalar[DTYPE]](
                    contacts[env, c_off + CONTACT_IDX_CONDIM]
                )
            )
            if condim < 1:
                condim = 3
            solver[env, ws_cd + c] = Scalar[DTYPE](condim)

            if condim > 1:
                # Tangent basis (MuJoCo mju_makeFrame with capsule axis hint)
                var hint_x = rebind[Scalar[DTYPE]](
                    contacts[env, c_off + CONTACT_IDX_FRAME_T1_X]
                )
                var hint_y = rebind[Scalar[DTYPE]](
                    contacts[env, c_off + CONTACT_IDX_FRAME_T1_Y]
                )
                var hint_z = rebind[Scalar[DTYPE]](
                    contacts[env, c_off + CONTACT_IDX_FRAME_T1_Z]
                )
                var frame = contact_tangent_frame[DTYPE](
                    nx, ny, nz, hint_x, hint_y, hint_z
                )
                var t1x = frame[0]
                var t1y = frame[1]
                var t1z = frame[2]
                var t2x = frame[3]
                var t2y = frame[4]
                var t2z = frame[5]

                # Store directions and friction coefficients
                solver[env, ws_df + (0 * 3 + 0) * MC + c] = t1x
                solver[env, ws_df + (0 * 3 + 1) * MC + c] = t1y
                solver[env, ws_df + (0 * 3 + 2) * MC + c] = t1z
                solver[env, ws_df + (1 * 3 + 0) * MC + c] = t2x
                solver[env, ws_df + (1 * 3 + 1) * MC + c] = t2y
                solver[env, ws_df + (1 * 3 + 2) * MC + c] = t2z
                solver[env, ws_fc + 0 * MC + c] = mu_slide
                solver[env, ws_fc + 1 * MC + c] = mu_slide

                var num_fric = 2
                if condim >= 4:
                    num_fric = 3
                    solver[env, ws_df + (2 * 3 + 0) * MC + c] = nx
                    solver[env, ws_df + (2 * 3 + 1) * MC + c] = ny
                    solver[env, ws_df + (2 * 3 + 2) * MC + c] = nz
                    solver[env, ws_fc + 2 * MC + c] = mu_spin
                if condim >= 6:
                    num_fric = 5
                    solver[env, ws_df + (3 * 3 + 0) * MC + c] = t1x
                    solver[env, ws_df + (3 * 3 + 1) * MC + c] = t1y
                    solver[env, ws_df + (3 * 3 + 2) * MC + c] = t1z
                    solver[env, ws_df + (4 * 3 + 0) * MC + c] = t2x
                    solver[env, ws_df + (4 * 3 + 1) * MC + c] = t2y
                    solver[env, ws_df + (4 * 3 + 2) * MC + c] = t2z
                    solver[env, ws_fc + 3 * MC + c] = mu_roll
                    solver[env, ws_fc + 4 * MC + c] = mu_roll

                var body_a = Int(solver[env, ws_c_body + c])
                var body_b = Int(solver[env, ws_c_body_b + c])
                var px = rebind[Scalar[DTYPE]](solver[env, ws_c_px + c])
                var py = rebind[Scalar[DTYPE]](solver[env, ws_c_py + c])
                var pz = rebind[Scalar[DTYPE]](solver[env, ws_c_pz + c])

                # Compute J, MinvJ, K for each friction direction
                for d in range(num_fric):
                    var dx = rebind[Scalar[DTYPE]](
                        solver[env, ws_df + (d * 3 + 0) * MC + c]
                    )
                    var dy = rebind[Scalar[DTYPE]](
                        solver[env, ws_df + (d * 3 + 1) * MC + c]
                    )
                    var dz = rebind[Scalar[DTYPE]](
                        solver[env, ws_df + (d * 3 + 2) * MC + c]
                    )

                    if d < 2:
                        _contact_jacobian_row[
                            DTYPE, V_SIZE](
                            env,
                            subtree_com,
                            joints,
                            bodies,
                            mmeta,
                            cdof,
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
                        _angular_jacobian_row[
                            DTYPE, NV, NBODY, NJOINT, V_SIZE, BATCH
                        ](
                            env,
                            joints,
                            bodies,
                            mmeta,
                            cdof,
                            body_a,
                            body_b,
                            dx,
                            dy,
                            dz,
                            J_row,
                        )

                    var k_d: solver.element_type = 0
                    for i in range(NV):
                        solver[env, ws_jf + d * MC * NV + c * NV + i] = J_row[
                            i
                        ]
                        var mi_j_sum: solver.element_type = 0
                        for j_idx in range(NV):
                            mi_j_sum += (
                                m_inv[env, i * NV + j_idx] * J_row[j_idx]
                            )
                        solver[
                            env, ws_mj + d * MC * NV + c * NV + i
                        ] = mi_j_sum
                        k_d += J_row[i] * mi_j_sum
                    if k_d < Scalar[DTYPE](1e-10):
                        k_d = Scalar[DTYPE](1e-10)
                    solver[env, ws_kf + d * MC + c] = k_d

                # Compute friction regularizer R_f from parent normal's impedance
                var impratio_pgs = rebind[Scalar[DTYPE]](
                    mmeta[MODEL_META_IDX_IMPRATIO]
                )
                if impratio_pgs < Scalar[DTYPE](1e-6):
                    impratio_pgs = Scalar[DTYPE](1.0)
                var imp_n_pgs = rebind[Scalar[DTYPE]](
                    solver[env, ws_inv_K_imp + c]
                ) * rebind[Scalar[DTYPE]](solver[env, ws_K_n + c])
                var R_base_pgs = (
                    (Scalar[DTYPE](1.0) - imp_n_pgs)
                    / imp_n_pgs
                    * rebind[Scalar[DTYPE]](solver[env, ws_K_n + c])
                    / impratio_pgs
                )
                for d in range(num_fric):
                    var R_d_pgs = R_base_pgs
                    if d >= 2:
                        var mu_d_pgs = rebind[Scalar[DTYPE]](
                            solver[env, ws_fc + d * MC + c]
                        )
                        if mu_d_pgs > Scalar[DTYPE](1e-12):
                            R_d_pgs = (
                                R_base_pgs
                                * mu_slide
                                * mu_slide
                                / (mu_d_pgs * mu_d_pgs)
                            )
                    solver[env, ws_rf + d * MC + c] = R_d_pgs

                # Compute velocity damping bias for friction rows
                for d in range(num_fric):
                    var v_t: solver.element_type = 0
                    for i in range(NV):
                        v_t += rebind[Scalar[DTYPE]](
                            solver[env, ws_jf + d * MC * NV + c * NV + i]
                        ) * rebind[Scalar[DTYPE]](qvel[env, i])
                    solver[env, ws_bf + d * MC + c] = B_damp * rebind[
                        Scalar[DTYPE]
                    ](v_t)

                comptime if CONE_TYPE == ConeType.PYRAMIDAL:
                    # Pyramidal precomputation: C_nt, K_edge_pos/neg, R_edge
                    var R_n_val = (
                        (Scalar[DTYPE](1.0) - imp_n_pgs)
                        / imp_n_pgs
                        * rebind[Scalar[DTYPE]](solver[env, ws_K_n + c])
                    )
                    for d in range(num_fric):
                        var mu_d_p = rebind[Scalar[DTYPE]](
                            solver[env, ws_fc + d * MC + c]
                        )
                        # Cross-term: C_nt[d][c] = Σ_i J_n[c*NV+i] * MinvJ_f[d*MC*NV+c*NV+i]
                        var c_nt_val: solver.element_type = 0
                        for i in range(NV):
                            c_nt_val += rebind[Scalar[DTYPE]](
                                solver[env, ws_J_n + c * NV + i]
                            ) * rebind[Scalar[DTYPE]](
                                solver[env, ws_mj + d * MC * NV + c * NV + i]
                            )
                        solver[env, ws_cnt + d * MC + c] = c_nt_val
                        var K_n_c = rebind[Scalar[DTYPE]](
                            solver[env, ws_K_n + c]
                        )
                        var K_f_d = rebind[Scalar[DTYPE]](
                            solver[env, ws_kf + d * MC + c]
                        )
                        solver[env, ws_kep + d * MC + c] = (
                            K_n_c
                            + Scalar[DTYPE](2.0) * mu_d_p * c_nt_val
                            + mu_d_p * mu_d_p * K_f_d
                        )
                        solver[env, ws_ken + d * MC + c] = (
                            K_n_c
                            - Scalar[DTYPE](2.0) * mu_d_p * c_nt_val
                            + mu_d_p * mu_d_p * K_f_d
                        )
                        solver[env, ws_re + d * MC + c] = (
                            Scalar[DTYPE](2.0) * mu_d_p * mu_d_p * R_n_val
                        )
                    # No warm-start for pyramidal
                    for d in range(num_fric):
                        solver[env, ws_lf + d * MC + c] = Scalar[DTYPE](0)
                        solver[env, ws_le_neg + d * MC + c] = Scalar[DTYPE](0)
                else:
                    # Warm-start friction impulses (elliptic only)
                    var warm_idx = InlineArray[Int, 5](uninitialized=True)
                    warm_idx[0] = CONTACT_IDX_FORCE_T1
                    warm_idx[1] = CONTACT_IDX_FORCE_T2
                    warm_idx[2] = CONTACT_IDX_FORCE_TORSION
                    warm_idx[3] = CONTACT_IDX_FORCE_ROLL1
                    warm_idx[4] = CONTACT_IDX_FORCE_ROLL2
                    for d in range(num_fric):
                        solver[env, ws_lf + d * MC + c] = rebind[
                            Scalar[DTYPE]
                        ](contacts[env, c_off + warm_idx[d]])

    # === SEQUENTIAL: Coupled PGS (normals + friction) + impulse store
    # (legacy: thread 0) ===
    # Coupled PGS iterations (normals + friction together, MuJoCo-style) with
    # per-island early termination (INSERTION D; legacy island_pgs_solver:
    # 890-1574; delta metric = normal-update |Δλ_n|)
    var eps_gpu2 = Scalar[DTYPE](ISLAND_CONVERGE_EPS)
    for _ in range(PGS_ITERATIONS):
        if num_converged >= num_islands:
            break
        var island_max_delta_c = InlineArray[Scalar[DTYPE], MAX_ISLANDS](
            fill=Scalar[DTYPE](0)
        )
        # --- Normal constraints PGS update ---
        for c in range(nc):
            if solver[env, ws_c_dist + c] >= Scalar[DTYPE](0):
                continue
            var iid = contact_island[c]
            if iid >= 0 and island_converged[iid] == 1:
                continue
            var a_n: solver.element_type = 0
            for i in range(NV):
                a_n += (
                    solver[env, ws_J_n + c * NV + i]
                    * qacc_constrained[env, i]
                )
            var R_n = Scalar[DTYPE](1.0) / rebind[Scalar[DTYPE]](
                solver[env, ws_inv_K_imp + c]
            ) - rebind[Scalar[DTYPE]](solver[env, ws_K_n + c])
            var residual = (
                a_n
                + solver[env, ws_pos_bias + c]
                + R_n * solver[env, ws_lambda_n + c]
            )
            var delta = -residual * solver[env, ws_inv_K_imp + c]
            var old_lambda = solver[env, ws_lambda_n + c]
            solver[env, ws_lambda_n + c] = (
                solver[env, ws_lambda_n + c] + delta
            )
            if solver[env, ws_lambda_n + c] < Scalar[DTYPE](0):
                solver[env, ws_lambda_n + c] = Scalar[DTYPE](0)
            var actual_n = solver[env, ws_lambda_n + c] - old_lambda
            var abs_n = abs(rebind[Scalar[DTYPE]](actual_n))
            if iid >= 0 and abs_n > island_max_delta_c[iid]:
                island_max_delta_c[iid] = abs_n
            for i in range(NV):
                qacc_constrained[env, i] += (
                    solver[env, ws_MinvJn + c * NV + i] * actual_n
                )

        # --- Friction constraints PGS update ---
        for c in range(nc):
            var iid = contact_island[c]
            if iid >= 0 and island_converged[iid] == 1:
                continue
            if solver[env, ws_lambda_n + c] <= Scalar[DTYPE](0):
                # Zero friction when normal force is zero
                var condim_z = Int(solver[env, ws_cd + c])
                var num_fric_z = 2
                if condim_z >= 4:
                    num_fric_z = 3
                if condim_z >= 6:
                    num_fric_z = 5
                for d in range(num_fric_z):
                    comptime if CONE_TYPE == ConeType.PYRAMIDAL:
                        var mu_d = rebind[Scalar[DTYPE]](
                            solver[env, ws_fc + d * MC + c]
                        )
                        var old_pos = rebind[Scalar[DTYPE]](
                            solver[env, ws_lf + d * MC + c]
                        )
                        var old_neg_v = rebind[Scalar[DTYPE]](
                            solver[env, ws_le_neg + d * MC + c]
                        )
                        if old_pos != Scalar[DTYPE](
                            0
                        ) or old_neg_v != Scalar[DTYPE](0):
                            solver[env, ws_lf + d * MC + c] = Scalar[DTYPE](0)
                            solver[env, ws_le_neg + d * MC + c] = Scalar[
                                DTYPE
                            ](0)
                            for i in range(NV):
                                var minvjn_i = rebind[Scalar[DTYPE]](
                                    solver[env, ws_MinvJn + c * NV + i]
                                )
                                var minvjf_i = rebind[Scalar[DTYPE]](
                                    solver[
                                        env, ws_mj + d * MC * NV + c * NV + i
                                    ]
                                )
                                qacc_constrained[env, i] -= (
                                    minvjn_i + mu_d * minvjf_i
                                ) * old_pos
                                qacc_constrained[env, i] -= (
                                    minvjn_i - mu_d * minvjf_i
                                ) * old_neg_v
                    else:
                        var old_f = rebind[Scalar[DTYPE]](
                            solver[env, ws_lf + d * MC + c]
                        )
                        if old_f != Scalar[DTYPE](0):
                            solver[env, ws_lf + d * MC + c] = Scalar[DTYPE](0)
                            for i in range(NV):
                                qacc_constrained[env, i] -= (
                                    solver[
                                        env, ws_mj + d * MC * NV + c * NV + i
                                    ]
                                    * old_f
                                )
                continue
            var condim = Int(solver[env, ws_cd + c])
            if condim == 1:
                continue

            var num_fric = 2
            if condim >= 4:
                num_fric = 3
            if condim >= 6:
                num_fric = 5

            var lambda_n = rebind[Scalar[DTYPE]](
                solver[env, ws_lambda_n + c]
            )

            comptime if CONE_TYPE == ConeType.PYRAMIDAL:
                # === PYRAMIDAL CONE: Edge constraints with λ ≥ 0 ===
                var bias_n = rebind[Scalar[DTYPE]](
                    solver[env, ws_pos_bias + c]
                )

                for d in range(num_fric):
                    var mu_d = rebind[Scalar[DTYPE]](
                        solver[env, ws_fc + d * MC + c]
                    )
                    if mu_d <= Scalar[DTYPE](1e-12):
                        continue

                    var a_n_val: solver.element_type = 0
                    var a_f_val: solver.element_type = 0
                    for i in range(NV):
                        var qi = rebind[Scalar[DTYPE]](
                            qacc_constrained[env, i]
                        )
                        a_n_val += (
                            rebind[Scalar[DTYPE]](
                                solver[env, ws_J_n + c * NV + i]
                            )
                            * qi
                        )
                        a_f_val += (
                            rebind[Scalar[DTYPE]](
                                solver[env, ws_jf + d * MC * NV + c * NV + i]
                            )
                            * qi
                        )

                    var R_e = rebind[Scalar[DTYPE]](
                        solver[env, ws_re + d * MC + c]
                    )

                    # Positive edge (+)
                    var a_edge_pos = a_n_val + mu_d * a_f_val
                    var K_ep = rebind[Scalar[DTYPE]](
                        solver[env, ws_kep + d * MC + c]
                    )
                    var residual_pos = (
                        a_edge_pos
                        + bias_n
                        + R_e
                        * rebind[Scalar[DTYPE]](
                            solver[env, ws_lf + d * MC + c]
                        )
                    )
                    var delta_pos = -residual_pos / (K_ep + R_e)
                    var new_lp = (
                        rebind[Scalar[DTYPE]](
                            solver[env, ws_lf + d * MC + c]
                        )
                        + delta_pos
                    )
                    if new_lp < Scalar[DTYPE](0):
                        new_lp = Scalar[DTYPE](0)
                    var actual_pos = new_lp - rebind[Scalar[DTYPE]](
                        solver[env, ws_lf + d * MC + c]
                    )
                    solver[env, ws_lf + d * MC + c] = new_lp
                    if actual_pos != Scalar[DTYPE](0):
                        for i in range(NV):
                            qacc_constrained[env, i] += (
                                rebind[Scalar[DTYPE]](
                                    solver[env, ws_MinvJn + c * NV + i]
                                )
                                + mu_d
                                * rebind[Scalar[DTYPE]](
                                    solver[
                                        env,
                                        ws_mj + d * MC * NV + c * NV + i,
                                    ]
                                )
                            ) * actual_pos

                    # Recompute after positive edge
                    a_n_val = 0
                    a_f_val = 0
                    for i in range(NV):
                        var qi = rebind[Scalar[DTYPE]](
                            qacc_constrained[env, i]
                        )
                        a_n_val += (
                            rebind[Scalar[DTYPE]](
                                solver[env, ws_J_n + c * NV + i]
                            )
                            * qi
                        )
                        a_f_val += (
                            rebind[Scalar[DTYPE]](
                                solver[env, ws_jf + d * MC * NV + c * NV + i]
                            )
                            * qi
                        )

                    # Negative edge (-)
                    var a_edge_neg = a_n_val - mu_d * a_f_val
                    var K_en = rebind[Scalar[DTYPE]](
                        solver[env, ws_ken + d * MC + c]
                    )
                    var residual_neg = (
                        a_edge_neg
                        + bias_n
                        + R_e
                        * rebind[Scalar[DTYPE]](
                            solver[env, ws_le_neg + d * MC + c]
                        )
                    )
                    var delta_neg = -residual_neg / (K_en + R_e)
                    var new_ln = (
                        rebind[Scalar[DTYPE]](
                            solver[env, ws_le_neg + d * MC + c]
                        )
                        + delta_neg
                    )
                    if new_ln < Scalar[DTYPE](0):
                        new_ln = Scalar[DTYPE](0)
                    var actual_neg = new_ln - rebind[Scalar[DTYPE]](
                        solver[env, ws_le_neg + d * MC + c]
                    )
                    solver[env, ws_le_neg + d * MC + c] = new_ln
                    if actual_neg != Scalar[DTYPE](0):
                        for i in range(NV):
                            qacc_constrained[env, i] += (
                                rebind[Scalar[DTYPE]](
                                    solver[env, ws_MinvJn + c * NV + i]
                                )
                                - mu_d
                                * rebind[Scalar[DTYPE]](
                                    solver[
                                        env,
                                        ws_mj + d * MC * NV + c * NV + i,
                                    ]
                                )
                            ) * actual_neg
                _ = lambda_n
            else:
                # === ELLIPTIC CONE: MuJoCo-style block update ===
                # Ray update + QCQP with AR submatrix + costChange
                var dim = 1 + num_fric

                # Build block AR matrix on-the-fly from J/MinvJ
                var AR = InlineArray[Scalar[DTYPE], 36](
                    fill=Scalar[DTYPE](0)
                )
                # Compute R_n directly from stored imp and diag_n
                comptime ws_imp_n_pgs = 13 * MC
                comptime ws_diag_n_pgs = 14 * MC
                var imp_pgs = rebind[Scalar[DTYPE]](
                    solver[env, ws_imp_n_pgs + c]
                )
                var diag_pgs = rebind[Scalar[DTYPE]](
                    solver[env, ws_diag_n_pgs + c]
                )
                var R_n_val = (
                    (Scalar[DTYPE](1.0) - imp_pgs) / imp_pgs * diag_pgs
                )
                AR[0] = (
                    rebind[Scalar[DTYPE]](solver[env, ws_K_n + c]) + R_n_val
                )

                for d1 in range(num_fric):
                    # Normal-friction cross: J_n @ MinvJ_f[d1]
                    var cross: Scalar[DTYPE] = 0
                    for i in range(NV):
                        cross += rebind[Scalar[DTYPE]](
                            solver[env, ws_J_n + c * NV + i]
                        ) * rebind[Scalar[DTYPE]](
                            solver[env, ws_mj + d1 * MC * NV + c * NV + i]
                        )
                    AR[(d1 + 1)] = cross
                    AR[(d1 + 1) * dim] = cross

                    for d2 in range(num_fric):
                        var ff: Scalar[DTYPE] = 0
                        for i in range(NV):
                            ff += rebind[Scalar[DTYPE]](
                                solver[
                                    env, ws_jf + d1 * MC * NV + c * NV + i
                                ]
                            ) * rebind[Scalar[DTYPE]](
                                solver[
                                    env, ws_mj + d2 * MC * NV + c * NV + i
                                ]
                            )
                        if d1 == d2:
                            ff += rebind[Scalar[DTYPE]](
                                solver[env, ws_rf + d1 * MC + c]
                            )
                        AR[(d1 + 1) * dim + (d2 + 1)] = ff

                # Compute block residual
                var block_res = InlineArray[Scalar[DTYPE], 6](
                    fill=Scalar[DTYPE](0)
                )
                var a_n_res: Scalar[DTYPE] = 0
                for i in range(NV):
                    a_n_res += rebind[Scalar[DTYPE]](
                        solver[env, ws_J_n + c * NV + i]
                    ) * rebind[Scalar[DTYPE]](qacc_constrained[env, i])
                block_res[0] = (
                    a_n_res
                    + rebind[Scalar[DTYPE]](solver[env, ws_pos_bias + c])
                    + R_n_val
                    * rebind[Scalar[DTYPE]](solver[env, ws_lambda_n + c])
                )
                for d in range(num_fric):
                    var a_f_res: Scalar[DTYPE] = 0
                    for i in range(NV):
                        a_f_res += rebind[Scalar[DTYPE]](
                            solver[env, ws_jf + d * MC * NV + c * NV + i]
                        ) * rebind[Scalar[DTYPE]](qacc_constrained[env, i])
                    var R_f_d = rebind[Scalar[DTYPE]](
                        solver[env, ws_rf + d * MC + c]
                    )
                    block_res[1 + d] = (
                        a_f_res
                        + rebind[Scalar[DTYPE]](solver[env, ws_bf + d * MC + c])
                        + R_f_d
                        * rebind[Scalar[DTYPE]](
                            solver[env, ws_lf + d * MC + c]
                        )
                    )

                # Save old forces
                var oldforce = InlineArray[Scalar[DTYPE], 6](
                    fill=Scalar[DTYPE](0)
                )
                oldforce[0] = rebind[Scalar[DTYPE]](
                    solver[env, ws_lambda_n + c]
                )
                for d in range(num_fric):
                    oldforce[1 + d] = rebind[Scalar[DTYPE]](
                        solver[env, ws_lf + d * MC + c]
                    )

                var ARinv0: Scalar[DTYPE] = 0
                if AR[0] > Scalar[DTYPE](1e-10):
                    ARinv0 = Scalar[DTYPE](1.0) / AR[0]

                # --- Ray update ---
                if rebind[Scalar[DTYPE]](
                    solver[env, ws_lambda_n + c]
                ) < Scalar[DTYPE](1e-10):
                    solver[env, ws_lambda_n + c] = (
                        rebind[Scalar[DTYPE]](solver[env, ws_lambda_n + c])
                        - block_res[0] * ARinv0
                    )
                    if solver[env, ws_lambda_n + c] < Scalar[DTYPE](0):
                        solver[env, ws_lambda_n + c] = Scalar[DTYPE](0)
                    for d in range(num_fric):
                        solver[env, ws_lf + d * MC + c] = Scalar[DTYPE](0)
                else:
                    var v = InlineArray[Scalar[DTYPE], 6](
                        fill=Scalar[DTYPE](0)
                    )
                    v[0] = rebind[Scalar[DTYPE]](
                        solver[env, ws_lambda_n + c]
                    )
                    for d in range(num_fric):
                        v[1 + d] = rebind[Scalar[DTYPE]](
                            solver[env, ws_lf + d * MC + c]
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
                            solver[env, ws_lambda_n + c]
                        ) + x * v[0] < Scalar[DTYPE](0):
                            x = (
                                -rebind[Scalar[DTYPE]](
                                    solver[env, ws_lambda_n + c]
                                )
                                / v[0]
                            )
                        solver[env, ws_lambda_n + c] = (
                            rebind[Scalar[DTYPE]](
                                solver[env, ws_lambda_n + c]
                            )
                            + x * v[0]
                        )
                        for d in range(num_fric):
                            solver[env, ws_lf + d * MC + c] = (
                                rebind[Scalar[DTYPE]](
                                    solver[env, ws_lf + d * MC + c]
                                )
                                + x * v[1 + d]
                            )

                # --- QCQP friction update ---
                var fn_val = rebind[Scalar[DTYPE]](
                    solver[env, ws_lambda_n + c]
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
                            solver[env, ws_fc + d * MC + c]
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
                        solver[env, ws_lf + 0 * MC + c] = r0
                        solver[env, ws_lf + 1 * MC + c] = r1
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
                        solver[env, ws_lf + 0 * MC + c] = r0
                        solver[env, ws_lf + 1 * MC + c] = r1
                        solver[env, ws_lf + 2 * MC + c] = r2
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
                            solver[env, ws_lf + d * MC + c] = res5[d]

                    # Rescale to exact ellipsoid if constrained
                    if flg_active:
                        var s: Scalar[DTYPE] = 0
                        for d in range(num_fric):
                            var fv = rebind[Scalar[DTYPE]](
                                solver[env, ws_lf + d * MC + c]
                            )
                            var mu_d = mu_arr[d]
                            if mu_d > Scalar[DTYPE](1e-10):
                                s += fv * fv / (mu_d * mu_d)
                        if s > Scalar[DTYPE](1e-10):
                            var scale = sqrt(fn_val * fn_val / s)
                            for d in range(num_fric):
                                solver[env, ws_lf + d * MC + c] = (
                                    rebind[Scalar[DTYPE]](
                                        solver[env, ws_lf + d * MC + c]
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
                            solver[env, ws_lambda_n + c]
                        )
                        old_i = oldforce[0]
                    else:
                        new_i = rebind[Scalar[DTYPE]](
                            solver[env, ws_lf + (bi - 1) * MC + c]
                        )
                        old_i = oldforce[bi]
                    var delta_i = new_i - old_i
                    cost_val += delta_i * block_res[bi]
                    for bj in range(dim):
                        var new_j: Scalar[DTYPE]
                        var old_j: Scalar[DTYPE]
                        if bj == 0:
                            new_j = rebind[Scalar[DTYPE]](
                                solver[env, ws_lambda_n + c]
                            )
                            old_j = oldforce[0]
                        else:
                            new_j = rebind[Scalar[DTYPE]](
                                solver[env, ws_lf + (bj - 1) * MC + c]
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
                    solver[env, ws_lambda_n + c] = oldforce[0]
                    for d in range(num_fric):
                        solver[env, ws_lf + d * MC + c] = oldforce[1 + d]

                # Apply delta to qacc
                var actual_n = (
                    rebind[Scalar[DTYPE]](solver[env, ws_lambda_n + c])
                    - oldforce[0]
                )
                if actual_n != Scalar[DTYPE](0):
                    for i in range(NV):
                        qacc_constrained[env, i] += (
                            solver[env, ws_MinvJn + c * NV + i] * actual_n
                        )
                for d in range(num_fric):
                    var actual_f = (
                        rebind[Scalar[DTYPE]](
                            solver[env, ws_lf + d * MC + c]
                        )
                        - oldforce[1 + d]
                    )
                    if actual_f != Scalar[DTYPE](0):
                        for i in range(NV):
                            qacc_constrained[env, i] += (
                                solver[env, ws_mj + d * MC * NV + c * NV + i]
                                * actual_f
                            )
                _ = lambda_n

        # Per-island freeze for the coupled phase (legacy 1570-1574)
        for iid in range(num_islands):
            if island_converged[iid] == 0:
                if island_max_delta_c[iid] < eps_gpu2:
                    island_converged[iid] = 1
                    num_converged += 1

    # Store impulses back to contact records for warm-starting
    comptime if CONE_TYPE == ConeType.PYRAMIDAL:
        # Pyramidal: force_n includes edge contributions
        for c in range(nc):
            var c_off = c * CONTACT_SIZE
            var condim = Int(solver[env, ws_cd + c])
            var num_fric = 2
            if condim >= 4:
                num_fric = 3
            if condim >= 6:
                num_fric = 5
            var total_n = rebind[Scalar[DTYPE]](
                solver[env, ws_lambda_n + c]
            )
            for d in range(num_fric):
                total_n += rebind[Scalar[DTYPE]](
                    solver[env, ws_lf + d * MC + c]
                )
                total_n += rebind[Scalar[DTYPE]](
                    solver[env, ws_le_neg + d * MC + c]
                )
            contacts[env, c_off + CONTACT_IDX_FORCE_N] = total_n
            var mu_0 = rebind[Scalar[DTYPE]](
                solver[env, ws_fc + 0 * MC + c]
            )
            contacts[env, c_off + CONTACT_IDX_FORCE_T1] = mu_0 * (
                rebind[Scalar[DTYPE]](solver[env, ws_lf + 0 * MC + c])
                - rebind[Scalar[DTYPE]](solver[env, ws_le_neg + 0 * MC + c])
            )
            var mu_1 = rebind[Scalar[DTYPE]](
                solver[env, ws_fc + 1 * MC + c]
            )
            contacts[env, c_off + CONTACT_IDX_FORCE_T2] = mu_1 * (
                rebind[Scalar[DTYPE]](solver[env, ws_lf + 1 * MC + c])
                - rebind[Scalar[DTYPE]](solver[env, ws_le_neg + 1 * MC + c])
            )
            if condim >= 4:
                var mu_2 = rebind[Scalar[DTYPE]](
                    solver[env, ws_fc + 2 * MC + c]
                )
                contacts[env, c_off + CONTACT_IDX_FORCE_TORSION] = mu_2 * (
                    rebind[Scalar[DTYPE]](solver[env, ws_lf + 2 * MC + c])
                    - rebind[Scalar[DTYPE]](
                        solver[env, ws_le_neg + 2 * MC + c]
                    )
                )
            if condim >= 6:
                var mu_3 = rebind[Scalar[DTYPE]](
                    solver[env, ws_fc + 3 * MC + c]
                )
                contacts[env, c_off + CONTACT_IDX_FORCE_ROLL1] = mu_3 * (
                    rebind[Scalar[DTYPE]](solver[env, ws_lf + 3 * MC + c])
                    - rebind[Scalar[DTYPE]](
                        solver[env, ws_le_neg + 3 * MC + c]
                    )
                )
                var mu_4 = rebind[Scalar[DTYPE]](
                    solver[env, ws_fc + 4 * MC + c]
                )
                contacts[env, c_off + CONTACT_IDX_FORCE_ROLL2] = mu_4 * (
                    rebind[Scalar[DTYPE]](solver[env, ws_lf + 4 * MC + c])
                    - rebind[Scalar[DTYPE]](
                        solver[env, ws_le_neg + 4 * MC + c]
                    )
                )
    else:
        # Elliptic: direct force writeback
        for c in range(nc):
            var c_off = c * CONTACT_SIZE
            contacts[env, c_off + CONTACT_IDX_FORCE_N] = solver[
                env, ws_lambda_n + c
            ]
            contacts[env, c_off + CONTACT_IDX_FORCE_T1] = solver[
                env, ws_lf + 0 * MC + c
            ]
            contacts[env, c_off + CONTACT_IDX_FORCE_T2] = solver[
                env, ws_lf + 1 * MC + c
            ]
            var condim = Int(solver[env, ws_cd + c])
            if condim >= 4:
                contacts[env, c_off + CONTACT_IDX_FORCE_TORSION] = solver[
                    env, ws_lf + 2 * MC + c
                ]
            if condim >= 6:
                contacts[env, c_off + CONTACT_IDX_FORCE_ROLL1] = solver[
                    env, ws_lf + 3 * MC + c
                ]
                contacts[env, c_off + CONTACT_IDX_FORCE_ROLL2] = solver[
                    env, ws_lf + 4 * MC + c
                ]


def _island_pgs_solve_fields_kernel[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int,
    NEQUALITY: Int,
    NTENDON: Int,
    NSITE: Int,
    CONE_TYPE: Int,
    BATCH: Int,
    SOLVER_WS: Int,
](
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    xpos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    subtree_com: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    contacts: LayoutTensor[
        DTYPE,
        Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE),
        MutAnyOrigin,
    ],
    smeta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, METADATA_SIZE), MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    mmeta: LayoutTensor[
        DTYPE, Layout.row_major(MODEL_META_SIZE), MutAnyOrigin
    ],
    equality: LayoutTensor[
        DTYPE, Layout.row_major(NEQUALITY, MODEL_EQ_SIZE), MutAnyOrigin
    ],
    tendons: LayoutTensor[
        DTYPE, Layout.row_major(NTENDON, MODEL_TENDON_SIZE), MutAnyOrigin
    ],
    sites: LayoutTensor[
        DTYPE, Layout.row_major(NSITE, MODEL_SITE_SIZE), MutAnyOrigin
    ],
    body_invweight0: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, 2), MutAnyOrigin
    ],
    dof_invweight0: LayoutTensor[DTYPE, Layout.row_major(NV), MutAnyOrigin],
    cdof: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * 6), MutAnyOrigin],
    m_inv: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin
    ],
    qacc_constrained: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin
    ],
    solver: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, SOLVER_WS), MutAnyOrigin
    ],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _island_pgs_solve_env[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        NEQUALITY,
        NTENDON,
        NSITE,
        CONE_TYPE,
        BATCH,
        SOLVER_WS,
    ](
        env, qpos, qvel, xpos, xquat, subtree_com, contacts, smeta, joints,
        bodies, mmeta, equality, tendons, sites, body_invweight0,
        dof_invweight0, cdof, m_inv, qacc_constrained, solver,
    )


def solve_island_pgs[

    target: StaticString,
    DTYPE: DType,
    D: DimsLike,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    BATCH: Int = 1,
    # Appended, not grouped with NEXCLUDE — see `fields.Model`.
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    mut scratch: DynamicsScratch[DTYPE, D, BATCH],
    mut cscratch: ContactScratch[DTYPE, D, BATCH, _],
    ctx: Optional[DeviceContext] = None,
) raises:
    """PGS contact solve into `scratch.qacc_constrained` (+ solved forces
    back into `d.contacts` for warm-starting), both targets, one body.
    Joint limits, equality constraints, and fixed tendons run INSIDE at the
    legacy position (between the normal and friction phases)."""
    comptime MC = _max_one[D.MAX_CONTACTS]()
    comptime SOLVER_WS = 81 * MC + 12 * MC * D.NV

    comptime L_NV = Layout.row_major(BATCH, D.NV)
    comptime L_B3 = Layout.row_major(BATCH, D.NBODY * 3)
    comptime L_B4 = Layout.row_major(BATCH, D.NBODY * 4)
    comptime L_CON = Layout.row_major(BATCH, D.MAX_CONTACTS * CONTACT_SIZE)
    comptime L_SMETA = Layout.row_major(BATCH, METADATA_SIZE)
    comptime L_JOINT = Layout.row_major(D.NJOINT, MODEL_JOINT_SIZE)
    comptime L_BODY = Layout.row_major(D.NBODY, MODEL_BODY_SIZE)
    comptime L_MMETA = Layout.row_major(MODEL_META_SIZE)
    comptime L_EQ = Layout.row_major(D.NEQUALITY, MODEL_EQ_SIZE)
    comptime L_TEN = Layout.row_major(D.NTENDON, MODEL_TENDON_SIZE)
    comptime L_SITE = Layout.row_major(D.NSITE, MODEL_SITE_SIZE)
    comptime L_BW = Layout.row_major(D.NBODY, 2)
    comptime L_CDOF = Layout.row_major(BATCH, D.NV * 6)
    comptime L_M = Layout.row_major(BATCH, D.NV * D.NV)
    comptime L_SOLVER = Layout.row_major(BATCH, SOLVER_WS)

    comptime L_QPOS = Layout.row_major(BATCH, D.NQ)
    comptime L_DW = Layout.row_major(D.NV)

    comptime if target == "cpu":
        var qpos_v = d.qpos.lt["cpu", L_QPOS]()
        var qvel_v = d.qvel.lt["cpu", L_NV]()
        var xpos_v = d.xpos.lt["cpu", L_B3]()
        var xquat_v = d.xquat.lt["cpu", L_B4]()
        var stcom_v = d.subtree_com.lt["cpu", L_B3]()
        var con_v = d.contacts.lt["cpu", L_CON]()
        var smeta_v = d.meta.lt["cpu", L_SMETA]()
        var joints_v = m.joints.lt["cpu", L_JOINT]()
        var bodies_v = m.bodies.lt["cpu", L_BODY]()
        var mmeta_v = m.meta.lt["cpu", L_MMETA]()
        var eq_v = m.equality.lt["cpu", L_EQ]()
        var ten_v = m.tendons.lt["cpu", L_TEN]()
        var site_v = m.sites.lt["cpu", L_SITE]()
        var bw_v = m.body_invweight0.lt["cpu", L_BW]()
        var dw_v = m.dof_invweight0.lt["cpu", L_DW]()
        var cdof_v = scratch.cdof.lt["cpu", L_CDOF]()
        var mi_v = scratch.m_inv.lt["cpu", L_M]()
        var qc_v = scratch.qacc_constrained.lt["cpu", L_NV]()
        var sol_v = cscratch.solver.lt["cpu", L_SOLVER]()
        for e in range(BATCH):
            _island_pgs_solve_env[
                DTYPE,
                D.NQ,
                D.NV,
                D.NBODY,
                D.NJOINT,
                D.MAX_CONTACTS,
                D.NGEOM,
                D.NEQUALITY,
                D.NTENDON,
                D.NSITE,
                CONE_TYPE,
                BATCH,
                SOLVER_WS,
            ](
                e, qpos_v, qvel_v, xpos_v, xquat_v, stcom_v, con_v, smeta_v,
                joints_v, bodies_v, mmeta_v, eq_v, ten_v, site_v, bw_v, dw_v,
                cdof_v, mi_v, qc_v, sol_v,
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + CS_TPB - 1) // CS_TPB
        c.enqueue_function[
            _island_pgs_solve_fields_kernel[
                DTYPE,
                D.NQ,
                D.NV,
                D.NBODY,
                D.NJOINT,
                D.MAX_CONTACTS,
                D.NGEOM,
                D.NEQUALITY,
                D.NTENDON,
                D.NSITE,
                CONE_TYPE,
                BATCH,
                SOLVER_WS,
            ]
        ](
            d.qpos.lt["gpu", L_QPOS](),
            d.qvel.lt["gpu", L_NV](),
            d.xpos.lt["gpu", L_B3](),
            d.xquat.lt["gpu", L_B4](),
            d.subtree_com.lt["gpu", L_B3](),
            d.contacts.lt["gpu", L_CON](),
            d.meta.lt["gpu", L_SMETA](),
            m.joints.lt["gpu", L_JOINT](),
            m.bodies.lt["gpu", L_BODY](),
            m.meta.lt["gpu", L_MMETA](),
            m.equality.lt["gpu", L_EQ](),
            m.tendons.lt["gpu", L_TEN](),
            m.sites.lt["gpu", L_SITE](),
            m.body_invweight0.lt["gpu", L_BW](),
            m.dof_invweight0.lt["gpu", L_DW](),
            scratch.cdof.lt["gpu", L_CDOF](),
            scratch.m_inv.lt["gpu", L_M](),
            scratch.qacc_constrained.lt["gpu", L_NV](),
            cscratch.solver.lt["gpu", L_SOLVER](),
            grid_dim=(BLOCKS,),
            block_dim=(CS_TPB,),
        )

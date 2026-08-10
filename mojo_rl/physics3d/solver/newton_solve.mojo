"""Newton contact solve over per-field tensors (migration P4, single-source).

Per-field port of `NewtonSolver.solve_gpu` (solver/newton_solver.mojo:1127)
— arithmetic, iteration order, constants (NEWTON_ITER_GPU=200,
NEWTON_TOL_GPU=1e-8, LINESEARCH_ITER=20, PRIMAL_MINVAL_GPU=1e-12) and
branch structure verbatim. Standalone solver entry only — NOT wired into
the fields integrators (later slice).

Structural transformation (the only deviation, identical to
constraints/contact_solve.mojo): the legacy kernel is 2D-threaded
(thread_y = contact slot) with barriers; this port SERIALIZES it per env.
The init + normal-precompute parallel phases become
`for contact_tid in range(MC)` loops (matching the legacy launch with
block_dim.y = MC and the helpers' internal `contact_tid < nc` guards); the
friction phase becomes `for contact_tid in range(nc)` (its legacy launch
guard). All phases write disjoint per-contact slots, so serialization is
value-identical. The entire Newton core after the legacy
`if not valid_env or contact_tid != 0: return` gate already runs
single-thread and is ported as-is.

Setup phases reuse the already-ported shared constraint-builder helpers
from contact_solve.mojo (`_init_common_normal_ws`,
`_precompute_contact_normal`, `_precompute_contact_friction`
— the latter two are the shared CG/Newton builders, verbatim ports of
`precompute_contact_normal_gpu` / `precompute_contact_friction_gpu`).

Cone-dependent tails at the exact legacy positions with the legacy
iteration count (SOLVER_ITER_GPU=50):
- ELLIPTIC: after the Newton core, `_limits_env` (port of
  `detect_and_solve_limits_gpu`) then `_equality_env` /
  `_tendon_env`.
- PYRAMIDAL: joint limits are edge rows INSIDE the Newton optimization
  (part of the verbatim core); equality/tendon after.
Equality/tendon are call-site gated `comptime if NEQUALITY > 0` /
`NTENDON > 0` (the legacy calls are unconditional with a comptime gate
inside the builder — bit-identical for zero counts, same as the PGS port).
Excluded: the legacy `dt` metadata read, whose only consumer was the
(unused-arg) limits call.

Workspace: the legacy Newton scratch is 35*MC + 6*MC*NV floats based at
`ws_solver_offset`. This port keeps the exact layout as row-relative
offsets into the fields `ContactScratch.solver` tensor, which is sized for
PGS (81*MC + 12*MC*NV) — strictly larger, so Newton uses a PREFIX of it
(no new scratch struct).

Operands (20): the 19 of `solve_contacts` + `M` (the Newton core
reads the mass matrix for the Gauss term / Hessian; legacy `ws_M_offset`).
The legacy `ws_fnet_offset` comptime was declared but never read — dropped.
"""

from std.math import sqrt, pow, abs
from std.gpu import thread_idx, block_idx, block_dim
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from std.sys.info import size_of
from max.gpu.host import DeviceContext
from std.sys import has_nvidia_gpu_accelerator
from layout import Layout, LayoutTensor

from ..types import _max_one, ConeType
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_FREE, JNT_BALL
from .cholesky import chol_factor_inline, chol_solve_inline
from .noslip import noslip_pyramidal

# `mjModel.opt.noslip_tolerance`, MuJoCo's default. dm_control's dog does not
# override it, and no ported model sets it, so it is a constant here rather
# than another parameter threaded through five call sites. The moment a model
# DOES set it, this must become one — the sweep stops on a different iteration
# otherwise.
comptime NOSLIP_TOLERANCE: Float64 = 1e-6
from .primal import pyramidal_edge_forces, pyramidal_linesearch
from ..constraints.contact_solve import (
    _init_common_normal_ws,
    _precompute_contact_normal,
    _precompute_contact_friction,
)
from ..constraints.limits import _limits_env
from ..constraints.friction_dof import _friction_env
from ..constraints.tendon_limit import (
    build_tendon_limit_rows,
    build_tendon_equality_rows,
)
from ..constraints.scalar_rows import (
    build_scalar_rows,
    max_scalar_rows,
    scalar_row_state,
    scalar_row_force,
    scalar_row_cost,
    SROW_QUADRATIC,
    SROW_LIMIT,
    SROW_FRICTION,
    SROW_EQ_BILATERAL,
    DOF_SOLREF_TIMECONST,
    DOF_SOLIMP_DMIN,
    DOF_SOLIMP_DMAX,
)
from ..constraints.equality_tendon import (
    _equality_env,
    _tendon_env,
)
from ..fields import Data, Model, DynamicsScratch, ContactScratch
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
    CONTACT_IDX_CONDIM,
    CONTACT_IDX_FORCE_N,
    CONTACT_IDX_FORCE_T1,
    CONTACT_IDX_FORCE_T2,
    META_IDX_NUM_CONTACTS,
    MODEL_META_IDX_MEANINERTIA,
    MODEL_META_IDX_SOLREF_CONTACT_0,
    MODEL_META_IDX_SOLREF_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_0,
    MODEL_META_IDX_SOLIMP_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_2,
    MODEL_META_IDX_SOLIMP_CONTACT_3,
    MODEL_META_IDX_SOLIMP_CONTACT_4,
    MODEL_META_IDX_IMPRATIO,
    MODEL_META_IDX_SOLREF_LIMIT_0,
    MODEL_META_IDX_SOLREF_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_0,
    MODEL_META_IDX_SOLIMP_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_2,
    MODEL_META_IDX_SOLIMP_LIMIT_3,
    MODEL_META_IDX_SOLIMP_LIMIT_4,
    JOINT_IDX_TYPE,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    JOINT_IDX_FRICTIONLOSS,
    JOINT_IDX_SOLREF_LIMIT_0,
    JOINT_IDX_SOLREF_LIMIT_1,
    JOINT_IDX_SOLIMP_LIMIT_0,
    JOINT_IDX_SOLIMP_LIMIT_1,
    JOINT_IDX_SOLIMP_LIMIT_2,
    JOINT_IDX_SOLIMP_LIMIT_3,
    JOINT_IDX_SOLIMP_LIMIT_4,
)

# One env per BLOCK (not 64 threads/block). The per-env Newton solve stack-
# allocates a large local frame (~ ME*NV + 3*MC*NV + several NV*NV, ~60KB for
# humanoid). With a wide block every thread — including the idle ones past
# BATCH — reserves that frame, and CUDA reserves it for max residency across
# the device, which OOMs at humanoid scale (Metal doesn't pre-reserve). One
# thread per block keeps the reservation to the envs actually running.
from ..constraints.constraint_data import refsafe_timeconst, solref_spring_damper

comptime NS_TPB: Int = 1


# =============================================================================
# Cooperative (one-env-per-block) helpers — verbatim ports of the shared-memory
# helpers in newton_solver.mojo (chol_factor_coop_gpu:174, matvec_mv_jve_coop:484,
# recompute_jfq_coop:546). They operate purely on SHARED-memory LayoutTensors, so
# the port is a straight copy — no slab/field addressing appears in them. @no_inline
# keeps their nested loops OUT of the giant blocked kernel (Mojo inline-explosion
# guard). Used only by the PYRAMIDAL blocked path.
# =============================================================================


@no_inline
def _chol_factor_coop[
    DTYPE: DType,
    NV: Int,
    M_SIZE: Int,
](
    tid: Int,
    n_threads: Int,
    H_sh: LayoutTensor[
        DTYPE,
        Layout.row_major(M_SIZE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    L_sh: LayoutTensor[
        DTYPE,
        Layout.row_major(M_SIZE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    ctrl_sh: LayoutTensor[
        DTYPE,
        Layout.row_major(3),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
):
    """Cooperative column-parallel Cholesky of shared H_sh -> L_sh (verbatim
    from chol_factor_coop_gpu). Bit-identical to chol_factor_inline."""
    for _attempt in range(2):
        if tid == 0:
            ctrl_sh[2] = Scalar[DTYPE](0)
        for k in range(tid, NV * NV, n_threads):
            L_sh[k] = Scalar[DTYPE](0)
        barrier()
        for j in range(NV):
            if tid == 0:
                var s_d: Scalar[DTYPE] = 0
                for k in range(j):
                    var ljk = rebind[Scalar[DTYPE]](L_sh[j * NV + k])
                    s_d += ljk * ljk
                var diag = rebind[Scalar[DTYPE]](H_sh[j * NV + j]) - s_d
                if diag < Scalar[DTYPE](1e-10):
                    ctrl_sh[2] = Scalar[DTYPE](1)
                    diag = Scalar[DTYPE](1e-10)
                L_sh[j * NV + j] = sqrt(diag)
            barrier()
            var ljj = rebind[Scalar[DTYPE]](L_sh[j * NV + j])
            for i in range(j + 1 + tid, NV, n_threads):
                var s: Scalar[DTYPE] = 0
                for k in range(j):
                    s += rebind[Scalar[DTYPE]](L_sh[i * NV + k]) * rebind[
                        Scalar[DTYPE]
                    ](L_sh[j * NV + k])
                L_sh[i * NV + j] = (
                    rebind[Scalar[DTYPE]](H_sh[i * NV + j]) - s
                ) / ljj
            barrier()
        if Int(rebind[Scalar[DTYPE]](ctrl_sh[2])) == 0:
            break
        # Rank-deficient: add 1e-6 to the H diagonal and refactor once.
        if tid == 0:
            for i in range(NV):
                H_sh[i * NV + i] += Scalar[DTYPE](1e-6)
        barrier()


@no_inline
def _matvec_mv_jve_coop[
    DTYPE: DType,
    NV: Int,
    V_SIZE: Int,
    M_SIZE: Int,
    ME: Int,
    JE_AS: AddressSpace = AddressSpace.SHARED,
](
    tid: Int,
    n_threads: Int,
    num_edges: Int,
    M_sh: LayoutTensor[
        DTYPE,
        Layout.row_major(M_SIZE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    # ⚠ `Je` is the ONE array whose address space varies — see JE_IN_SHARED at
    # the allocation site. Everything else stays in threadgroup memory.
    Je_sh: LayoutTensor[
        DTYPE,
        Layout.row_major(ME * V_SIZE),
        MutAnyOrigin,
        address_space=JE_AS,
    ],
    search_sh: LayoutTensor[
        DTYPE,
        Layout.row_major(V_SIZE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    Mv_sh: LayoutTensor[
        DTYPE,
        Layout.row_major(V_SIZE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    Jv_e_sh: LayoutTensor[
        DTYPE,
        Layout.row_major(ME),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
):
    """Cooperative Mv = M·search and Jv_e = Je·search (verbatim from
    matvec_mv_jve_coop). Ascending inner sums → bit-identical."""
    for i in range(tid, NV, n_threads):
        var s: Scalar[DTYPE] = 0
        for j in range(NV):
            s += rebind[Scalar[DTYPE]](M_sh[i * NV + j]) * rebind[
                Scalar[DTYPE]
            ](search_sh[j])
        Mv_sh[i] = s
    for e in range(tid, num_edges, n_threads):
        var s: Scalar[DTYPE] = 0
        for i in range(NV):
            s += rebind[Scalar[DTYPE]](Je_sh[e * NV + i]) * rebind[
                Scalar[DTYPE]
            ](search_sh[i])
        Jv_e_sh[e] = s


@no_inline
def _recompute_jfq_coop[
    DTYPE: DType,
    NV: Int,
    V_SIZE: Int,
    ME: Int,
    JE_AS: AddressSpace = AddressSpace.SHARED,
](
    tid: Int,
    n_threads: Int,
    num_edges: Int,
    # ⚠ address space varies — see JE_IN_SHARED at the allocation site.
    Je_sh: LayoutTensor[
        DTYPE, Layout.row_major(ME * V_SIZE), MutAnyOrigin,
        address_space=JE_AS,
    ],
    De_sh: LayoutTensor[
        DTYPE, Layout.row_major(ME), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    bias_e_sh: LayoutTensor[
        DTYPE, Layout.row_major(ME), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    kind_e_sh: LayoutTensor[
        DTYPE, Layout.row_major(ME), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    R_e_sh: LayoutTensor[
        DTYPE, Layout.row_major(ME), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    floss_e_sh: LayoutTensor[
        DTYPE, Layout.row_major(ME), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    state_e_sh: LayoutTensor[
        DTYPE, Layout.row_major(ME), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    qacc_sh: LayoutTensor[
        DTYPE, Layout.row_major(V_SIZE), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    jar_sh: LayoutTensor[
        DTYPE, Layout.row_major(ME), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    force_sh: LayoutTensor[
        DTYPE, Layout.row_major(ME), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
    qfrc_sh: LayoutTensor[
        DTYPE, Layout.row_major(V_SIZE), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ],
):
    """Cooperative jar/force/qfrc recompute (verbatim from recompute_jfq_coop).
    Two phases separated by a barrier; ascending inner sums → bit-identical."""
    for e in range(tid, num_edges, n_threads):
        var j = rebind[Scalar[DTYPE]](bias_e_sh[e])
        for i in range(NV):
            j += rebind[Scalar[DTYPE]](Je_sh[e * NV + i]) * rebind[
                Scalar[DTYPE]
            ](qacc_sh[i])
        jar_sh[e] = j
        var st = scalar_row_state[DTYPE](
            Int(rebind[Scalar[DTYPE]](kind_e_sh[e])),
            j,
            rebind[Scalar[DTYPE]](R_e_sh[e]),
            rebind[Scalar[DTYPE]](floss_e_sh[e]),
        )
        state_e_sh[e] = Scalar[DTYPE](st)
        force_sh[e] = scalar_row_force[DTYPE](
            st, j, rebind[Scalar[DTYPE]](De_sh[e]),
            rebind[Scalar[DTYPE]](floss_e_sh[e]),
        )
    barrier()
    for i in range(tid, NV, n_threads):
        var q: Scalar[DTYPE] = 0
        for e in range(num_edges):
            q += rebind[Scalar[DTYPE]](Je_sh[e * NV + i]) * rebind[
                Scalar[DTYPE]
            ](force_sh[e])
        qfrc_sh[i] = q


# =============================================================================
# Newton contact solve — single-source per-env body (port of
# NewtonSolver.solve_gpu)
# =============================================================================


@always_inline
def _newton_solve_env[
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
    MAX_CONDIM: Int = 3,
    NOSLIP_ITER: Int = 0,
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
    M: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
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
    """Full primal Newton contact solve for one env (verbatim from
    NewtonSolver.solve_gpu, serialized per env — see module docstring)."""
    comptime MC = _max_one[MAX_CONTACTS]()
    comptime V_SIZE = _max_one[NV]()
    comptime M_SIZE = _max_one[NV * NV]()

    # Common normal block offsets (row-relative; the legacy `solver_ws_idx`
    # base is gone)
    comptime ws_c_dist_idx = 2 * MC
    comptime ws_pos_bias_idx = 11 * MC
    comptime ws_J_n_idx = 15 * MC

    # Primal-specific offsets (after common normal block)
    comptime PRIMAL_START = 15 * MC + 2 * MC * NV
    comptime ws_Jt1_idx = PRIMAL_START + 0 * MC * NV
    comptime ws_Jt2_idx = PRIMAL_START + 1 * MC * NV
    comptime ws_MinvJt1_idx = PRIMAL_START + 2 * MC * NV
    comptime ws_MinvJt2_idx = PRIMAL_START + 3 * MC * NV
    comptime SC = PRIMAL_START + 4 * MC * NV
    comptime ws_mu_idx = SC + 0 * MC
    comptime ws_D_n_idx = SC + 1 * MC
    comptime ws_D_f_idx = SC + 2 * MC
    comptime ws_bt1_idx = SC + 3 * MC
    comptime ws_bt2_idx = SC + 4 * MC
    comptime CVS = SC + 5 * MC
    comptime ws_jar_n_idx = CVS + 0 * MC
    comptime ws_jar_t1_idx = CVS + 1 * MC
    comptime ws_jar_t2_idx = CVS + 2 * MC
    comptime ws_fn_idx = CVS + 3 * MC
    comptime ws_ft1_idx = CVS + 4 * MC
    comptime ws_ft2_idx = CVS + 5 * MC
    comptime ws_cstate_idx = CVS + 6 * MC

    # === Initialize workspace (legacy: parallel, one thread per slot; the
    # legacy `contact_tid < MC` guard is vacuous with block_dim.y = MC) ===
    for contact_tid in range(MC):
        _init_common_normal_ws[
            DTYPE, NV, MAX_CONTACTS, BATCH, SOLVER_WS
        ](env, contact_tid, solver)
        # Zero primal workspace for this contact slot
        for d in range(NV):
            solver[env, ws_Jt1_idx + contact_tid * NV + d] = 0
            solver[env, ws_Jt2_idx + contact_tid * NV + d] = 0
            solver[env, ws_MinvJt1_idx + contact_tid * NV + d] = 0
            solver[env, ws_MinvJt2_idx + contact_tid * NV + d] = 0
        solver[env, ws_mu_idx + contact_tid] = 0
        solver[env, ws_D_n_idx + contact_tid] = 0
        solver[env, ws_D_f_idx + contact_tid] = 0
        solver[env, ws_bt1_idx + contact_tid] = 0
        solver[env, ws_bt2_idx + contact_tid] = 0
        solver[env, ws_jar_n_idx + contact_tid] = 0
        solver[env, ws_jar_t1_idx + contact_tid] = 0
        solver[env, ws_jar_t2_idx + contact_tid] = 0
        solver[env, ws_fn_idx + contact_tid] = 0
        solver[env, ws_ft1_idx + contact_tid] = 0
        solver[env, ws_ft2_idx + contact_tid] = 0
        solver[env, ws_cstate_idx + contact_tid] = 0

    # Read metadata (legacy `dt` read dropped — only the unused-arg limits
    # call consumed it)
    var nc = 0
    var K_spring: Scalar[DTYPE] = 0
    var B_damp: Scalar[DTYPE] = 0
    var si_dmin: Scalar[DTYPE] = 0
    var si_dmax: Scalar[DTYPE] = 0
    var si_width: Scalar[DTYPE] = 1
    var si_midpoint: Scalar[DTYPE] = Scalar[DTYPE](0.5)
    var si_power: Scalar[DTYPE] = Scalar[DTYPE](2.0)
    var impratio: Scalar[DTYPE] = Scalar[DTYPE](1.0)

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
    # solref -> (K, B), including MuJoCo's DIRECT form for a NEGATIVE
    # solref. See `constraints/constraint_data.solref_spring_damper` — the
    # formula lived in twelve copy-pasted sites until 2026-08-03.
    (K_spring, B_damp) = solref_spring_damper[DTYPE](
        sr_tc, sr_dr, si_dmax,
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP]),
    )
    impratio = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_IMPRATIO])
    if impratio < Scalar[DTYPE](1e-6):
        impratio = Scalar[DTYPE](1.0)

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

    # === PHASE 2: Tangent frame + friction data (legacy launch guard
    # `contact_tid < nc`) ===
    for contact_tid in range(nc):
        _precompute_contact_friction[
            DTYPE,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            V_SIZE,
            BATCH,
            SOLVER_WS,
            CONE_TYPE,
            MAX_CONDIM,
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
            cdof,
            solver,
            B_damp,
            impratio,
            K_spring,
            ws_Jt1_idx,
            ws_Jt2_idx,
            ws_mu_idx,
            ws_D_n_idx,
            ws_D_f_idx,
            ws_bt1_idx,
            ws_bt2_idx,
        )

    # === SEQUENTIAL: primal Newton (legacy: thread 0) ===
    comptime NEWTON_ITER_GPU: Int = 200
    comptime NEWTON_TOL_GPU: Float64 = 1e-8
    comptime LINESEARCH_ITER: Int = 20
    comptime ARMIJO: Float64 = 1e-4
    comptime PRIMAL_MINVAL_GPU: Float64 = 1e-12

    comptime if CONE_TYPE == ConeType.PYRAMIDAL:
        # =================================================================
        # PYRAMIDAL Newton: iterate over edge rows (all >= 0 constraints)
        # 4 edges per contact for condim=3: J_e = J_n ± mu*J_t
        # No cone coupling — simpler than ELLIPTIC
        # =================================================================
        # Edges per contact = 2*(dim-1): 4 at condim 3, 6 at 4, 10 at 6.
        # Slots are sized for the model's worst condim; the builder zeros the
        # tail per contact, so a condim-3 contact here still spans 4 edges.
        comptime NE = 2 * (MAX_CONDIM - 1)
        comptime MAX_LIM = _max_one[2 * NJOINT]()
        comptime MAX_FRIC = V_SIZE  # one friction row per dof
        comptime MAX_TLIM = 2 * NTENDON  # lo + hi per tendon
        comptime MAX_TEQ = NTENDON  # one bilateral row per equality tendon
        # contact + limit + dry-friction + tendon-limit + tendon-equality rows
        comptime ME = NE * MC + MAX_LIM + MAX_FRIC + MAX_TLIM + MAX_TEQ

        # Cache edge data from PYRAMIDAL workspace layout
        var pyr_sc = ws_Jt1_idx + NE * MC * NV
        var Je = InlineArray[Scalar[DTYPE], ME * V_SIZE](uninitialized=True)
        var De = InlineArray[Scalar[DTYPE], ME](uninitialized=True)
        var bias_e = InlineArray[Scalar[DTYPE], ME](uninitialized=True)
        # Row kind + box data. Contact edges and joint limits are ONE-SIDED;
        # only dry-friction dof rows are box-clamped, and R/floss are read
        # solely on that branch, so the one-sided rows leave them at 0.
        var kind_e = InlineArray[Int, ME](fill=SROW_LIMIT)
        var R_e = InlineArray[Scalar[DTYPE], ME](fill=Scalar[DTYPE](0))
        var floss_e = InlineArray[Scalar[DTYPE], ME](fill=Scalar[DTYPE](0))
        var state_e = InlineArray[Int, ME](fill=0)
        var num_edges = nc * NE

        # Load contact edges
        for c in range(nc):
            for e in range(NE):
                var idx = c * NE + e
                for i in range(NV):
                    Je[idx * NV + i] = rebind[Scalar[DTYPE]](
                        solver[env, ws_Jt1_idx + e * MC * NV + c * NV + i]
                    )
                De[idx] = rebind[Scalar[DTYPE]](
                    solver[env, pyr_sc + e * MC + c]
                )
                bias_e[idx] = rebind[Scalar[DTYPE]](
                    solver[env, pyr_sc + NE * MC + e * MC + c]
                )

        # Detect and add joint limit edges (unified with contacts)
        # Matches CPU build_constraints: per-joint solref/solimp with
        # model-level defaults fallback
        # Model-level defaults for fallback
        var lr_tc_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLREF_LIMIT_0]
        )
        var lr_dr_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLREF_LIMIT_1]
        )
        var li_dmin_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLIMP_LIMIT_0]
        )
        var li_dmax_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLIMP_LIMIT_1]
        )
        var li_width_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLIMP_LIMIT_2]
        )
        var li_midpoint_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLIMP_LIMIT_3]
        )
        var li_power_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLIMP_LIMIT_4]
        )

        for j in range(NJOINT):
            var jtype = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE])
            )
            if jtype != JNT_HINGE and jtype != JNT_SLIDE:
                continue
            var dof = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR])
            )
            var qpos_adr = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS_ADR])
            )
            var rmin = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_RANGE_MIN]
            )
            var rmax = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_RANGE_MAX]
            )
            if rmin < Scalar[DTYPE](-1e9) or rmax > Scalar[DTYPE](1e9):
                continue
            # Per-joint solref/solimp with model-level defaults fallback
            var lr_tc = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLREF_LIMIT_0]
            )
            var lr_dr = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLREF_LIMIT_1]
            )
            if lr_tc <= Scalar[DTYPE](0):
                lr_tc = lr_tc_def
            if lr_dr <= Scalar[DTYPE](0):
                lr_dr = lr_dr_def
            var li_dmin = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLIMP_LIMIT_0]
            )
            var li_dmax = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLIMP_LIMIT_1]
            )
            var li_width = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLIMP_LIMIT_2]
            )
            var li_midpoint = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLIMP_LIMIT_3]
            )
            var li_power = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLIMP_LIMIT_4]
            )
            if li_dmax <= Scalar[DTYPE](0) and li_width <= Scalar[DTYPE](0):
                li_dmin = li_dmin_def
                li_dmax = li_dmax_def
                li_width = li_width_def
                li_midpoint = li_midpoint_def
                li_power = li_power_def
            if li_width < Scalar[DTYPE](1e-6):
                li_width = Scalar[DTYPE](1e-6)
            # Clamp BOTH ends to [mjMINIMP, mjMAXIMP] as MuJoCo does before
            # interpolating (engine_core_constraint.c:1284-1287); see the same fix
            # on the contact path above.
            comptime MJL_MINIMP = Scalar[DTYPE](0.0001)
            comptime MJL_MAXIMP = Scalar[DTYPE](0.9999)
            if li_dmin < MJL_MINIMP:
                li_dmin = MJL_MINIMP
            elif li_dmin > MJL_MAXIMP:
                li_dmin = MJL_MAXIMP
            if li_dmax < MJL_MINIMP:
                li_dmax = MJL_MINIMP
            elif li_dmax > MJL_MAXIMP:
                li_dmax = MJL_MAXIMP
            if li_power < Scalar[DTYPE](1):
                li_power = Scalar[DTYPE](1)
            # solref -> (K, B), including MuJoCo's DIRECT form for a NEGATIVE
            # solref. See `constraints/constraint_data.solref_spring_damper` — the
            # formula lived in twelve copy-pasted sites until 2026-08-03.
            var (l_K_spring, l_B_damp) = solref_spring_damper[DTYPE](
                lr_tc, lr_dr, li_dmax,
                rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP]),
            )

            var pos = rebind[Scalar[DTYPE]](qpos[env, qpos_adr])
            # Lower limit: dist_lo = pos - rmin < 0 → violated
            var dist_lo = pos - rmin
            if dist_lo < Scalar[DTYPE](0) and num_edges < ME:
                var sign = Scalar[DTYPE](1)
                var K_lim = rebind[Scalar[DTYPE]](
                    m_inv[env, dof * NV + dof]
                )
                if K_lim < Scalar[DTYPE](1e-10):
                    K_lim = Scalar[DTYPE](1e-10)
                var pen = -dist_lo
                var v_lim = sign * rebind[Scalar[DTYPE]](qvel[env, dof])
                # Impedance
                var imp_lim: Scalar[DTYPE]
                if li_dmin == li_dmax or li_width <= Scalar[DTYPE](0):
                    imp_lim = Scalar[DTYPE](0.5) * (li_dmin + li_dmax)
                else:
                    var x_l = pen / li_width
                    if x_l <= Scalar[DTYPE](0):
                        imp_lim = li_dmin
                    elif x_l >= Scalar[DTYPE](1):
                        imp_lim = li_dmax
                    else:
                        var y_l: Scalar[DTYPE]
                        if li_power == Scalar[DTYPE](1):
                            y_l = x_l
                        elif x_l <= li_midpoint:
                            y_l = pow(x_l, li_power) / pow(
                                li_midpoint, li_power - Scalar[DTYPE](1)
                            )
                        else:
                            y_l = Scalar[DTYPE](1) - pow(
                                Scalar[DTYPE](1) - x_l, li_power
                            ) / pow(
                                Scalar[DTYPE](1) - li_midpoint,
                                li_power - Scalar[DTYPE](1),
                            )
                        imp_lim = li_dmin + y_l * (li_dmax - li_dmin)
                if imp_lim < Scalar[DTYPE](1e-6):
                    imp_lim = Scalar[DTYPE](1e-6)
                var diag_lim = rebind[Scalar[DTYPE]](dof_invweight0[dof])
                if diag_lim < Scalar[DTYPE](1e-10):
                    diag_lim = K_lim
                var R_lim = (
                    (Scalar[DTYPE](1) - imp_lim) / imp_lim * diag_lim
                )
                if R_lim < Scalar[DTYPE](1e-14):
                    R_lim = Scalar[DTYPE](1e-14)
                # Sparse Jacobian: Je[dof] = sign, others 0
                for i in range(NV):
                    Je[num_edges * NV + i] = Scalar[DTYPE](0)
                Je[num_edges * NV + dof] = sign
                # Match CPU: inv_K = 1/(K+R), D = 1/(1/inv_K - K)
                # Same float32 rounding as primal_D(inv_K_imp, K)
                var inv_K_lim = Scalar[DTYPE](1) / (K_lim + R_lim)
                var R_recov = Scalar[DTYPE](1) / inv_K_lim - K_lim
                if R_recov < Scalar[DTYPE](1e-14):
                    R_recov = Scalar[DTYPE](1e-14)
                De[num_edges] = Scalar[DTYPE](1) / R_recov
                bias_e[num_edges] = (
                    l_B_damp * v_lim - l_K_spring * imp_lim * pen
                )
                num_edges += 1

            # Upper limit: dist_hi = rmax - pos < 0 → violated
            var dist_hi = rmax - pos
            if dist_hi < Scalar[DTYPE](0) and num_edges < ME:
                var sign = Scalar[DTYPE](-1)
                var K_lim = rebind[Scalar[DTYPE]](
                    m_inv[env, dof * NV + dof]
                )
                if K_lim < Scalar[DTYPE](1e-10):
                    K_lim = Scalar[DTYPE](1e-10)
                var pen = -dist_hi
                var v_lim = sign * rebind[Scalar[DTYPE]](qvel[env, dof])
                var imp_lim: Scalar[DTYPE]
                if li_dmin == li_dmax or li_width <= Scalar[DTYPE](0):
                    imp_lim = Scalar[DTYPE](0.5) * (li_dmin + li_dmax)
                else:
                    var x_l = pen / li_width
                    if x_l <= Scalar[DTYPE](0):
                        imp_lim = li_dmin
                    elif x_l >= Scalar[DTYPE](1):
                        imp_lim = li_dmax
                    else:
                        var y_l: Scalar[DTYPE]
                        if li_power == Scalar[DTYPE](1):
                            y_l = x_l
                        elif x_l <= li_midpoint:
                            y_l = pow(x_l, li_power) / pow(
                                li_midpoint, li_power - Scalar[DTYPE](1)
                            )
                        else:
                            y_l = Scalar[DTYPE](1) - pow(
                                Scalar[DTYPE](1) - x_l, li_power
                            ) / pow(
                                Scalar[DTYPE](1) - li_midpoint,
                                li_power - Scalar[DTYPE](1),
                            )
                        imp_lim = li_dmin + y_l * (li_dmax - li_dmin)
                if imp_lim < Scalar[DTYPE](1e-6):
                    imp_lim = Scalar[DTYPE](1e-6)
                var diag_lim = rebind[Scalar[DTYPE]](dof_invweight0[dof])
                if diag_lim < Scalar[DTYPE](1e-10):
                    diag_lim = K_lim
                var R_lim = (
                    (Scalar[DTYPE](1) - imp_lim) / imp_lim * diag_lim
                )
                if R_lim < Scalar[DTYPE](1e-14):
                    R_lim = Scalar[DTYPE](1e-14)
                for i in range(NV):
                    Je[num_edges * NV + i] = Scalar[DTYPE](0)
                Je[num_edges * NV + dof] = sign
                # Match CPU: inv_K = 1/(K+R), D = 1/(1/inv_K - K)
                # Same float32 rounding as primal_D(inv_K_imp, K)
                var inv_K_lim = Scalar[DTYPE](1) / (K_lim + R_lim)
                var R_recov = Scalar[DTYPE](1) / inv_K_lim - K_lim
                if R_recov < Scalar[DTYPE](1e-14):
                    R_recov = Scalar[DTYPE](1e-14)
                De[num_edges] = Scalar[DTYPE](1) / R_recov
                bias_e[num_edges] = (
                    l_B_damp * v_lim - l_K_spring * imp_lim * pen
                )
                num_edges += 1

        # Tendon limit rows (MuJoCo mjCNSTR_LIMIT_TENDON). Dense J, one row
        # per violated side — see constraints/tendon_limit.mojo for why this
        # is a row here rather than a post-pass.
        comptime if NTENDON > 0:
            build_tendon_limit_rows[
                DTYPE, NV, NBODY, NJOINT, NSITE, NTENDON, V_SIZE, ME, BATCH
            ](
                env, qvel, tendons, sites, bodies, joints, mmeta,
                subtree_com, cdof, xpos, xquat, m_inv,
                Je, De, bias_e, num_edges,
            )

        # Fixed-tendon equality rows (MuJoCo mjEQ_TENDON). BILATERAL — always
        # active, never clamped. These used to be a post-solve Gauss-Seidel
        # pass; with contacts live that split cost a standing quadruped two
        # thirds of its ground reaction force. See
        # constraints/tendon_limit.build_tendon_equality_rows.
        comptime if NTENDON > 0:
            build_tendon_equality_rows[
                DTYPE, NQ, NV, NJOINT, NTENDON, V_SIZE, ME, BATCH
            ](
                env, qpos, qvel, tendons, joints, mmeta, m_inv,
                Je, De, bias_e, kind_e, num_edges,
            )

        # Dry-friction dof rows (MuJoCo mjCNSTR_FRICTION_DOF). These were
        # MISSING from the pyramidal path entirely — `_friction_env` was only
        # ever called on the elliptic branch, so a pyramidal model with
        # `frictionloss` silently had no dry friction at all. They are box
        # rows, clamped to +-frictionloss, hence kind_e = SROW_FRICTION.
        var f_imp = Scalar[DTYPE](DOF_SOLIMP_DMIN)
        var f_dmax = Scalar[DTYPE](DOF_SOLIMP_DMAX)
        # REFSAFE applies to the hardcoded friction default too — see
        # `refsafe_timeconst`.
        var f_tc_p = refsafe_timeconst[DTYPE](
            Scalar[DTYPE](DOF_SOLREF_TIMECONST),
            rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP]),
        )
        var f_B = Scalar[DTYPE](2.0) / (f_dmax * f_tc_p)
        for j in range(NJOINT):
            var floss = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_FRICTIONLOSS]
            )
            if floss <= Scalar[DTYPE](0):
                continue
            var jt = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
            var dof_adr = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR])
            )
            var nd = 1
            if jt == JNT_FREE:
                nd = 6
            elif jt == JNT_BALL:
                nd = 3
            for k in range(nd):
                if num_edges >= ME:
                    break
                var dof = dof_adr + k
                var K_d = rebind[Scalar[DTYPE]](m_inv[env, dof * NV + dof])
                if K_d < Scalar[DTYPE](1e-10):
                    K_d = Scalar[DTYPE](1e-10)
                var diag_f = rebind[Scalar[DTYPE]](dof_invweight0[dof])
                if diag_f < Scalar[DTYPE](1e-10):
                    diag_f = K_d
                var R_f = (Scalar[DTYPE](1) - f_imp) / f_imp * diag_f
                if R_f < Scalar[DTYPE](1e-14):
                    R_f = Scalar[DTYPE](1e-14)
                for i in range(NV):
                    Je[num_edges * NV + i] = Scalar[DTYPE](0)
                Je[num_edges * NV + dof] = Scalar[DTYPE](1)
                De[num_edges] = Scalar[DTYPE](1) / R_f
                R_e[num_edges] = R_f
                floss_e[num_edges] = floss
                kind_e[num_edges] = SROW_FRICTION
                bias_e[num_edges] = f_B * rebind[Scalar[DTYPE]](
                    qvel[env, dof]
                )
                num_edges += 1

        # Initialize qacc from workspace (qacc_smooth set by stage kernel)
        var qacc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var qacc_smooth = InlineArray[Scalar[DTYPE], V_SIZE](
            uninitialized=True
        )
        var Ma = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

        # Cache M locally once — M is loop-invariant during Newton iterations.
        # Avoids ~2*NV² workspace (global) reads per iteration (Hessian build
        # + Mv = M*search). Mirrors the ELLIPTIC path's M_local optimization.
        var M_local = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        for k in range(NV * NV):
            M_local[k] = rebind[Scalar[DTYPE]](M[env, k])

        for i in range(NV):
            var q_i = rebind[Scalar[DTYPE]](qacc_constrained[env, i])
            qacc[i] = q_i
            qacc_smooth[i] = q_i
        for i in range(NV):
            Ma[i] = Scalar[DTYPE](0)
            for j in range(NV):
                Ma[i] += M_local[i * NV + j] * qacc[j]
        # f_smooth = M * qacc (matching CPU's qfrc_smooth = M * qacc_smooth)
        # Using Ma directly avoids LDL round-trip error (f_net ≠ M*M^{-1}*f_net)
        var f_smooth = InlineArray[Scalar[DTYPE], V_SIZE](
            uninitialized=True
        )
        for i in range(NV):
            f_smooth[i] = Ma[i]

        # ⚠ MuJoCo's CONVERGENCE SCALE IS A MODEL CONSTANT, NOT A POSE ONE.
        # `mj_solPrimal` uses `1 / (stat.meaninertia * max(1, nv))`
        # (`engine_solver.c:1863`), and `stat.meaninertia` is the mean of the
        # mass-matrix diagonal evaluated ONCE at qpos0 in `mj_setConst`. This
        # summed `M[i][i]` at the CURRENT pose instead — the same formula
        # (`sum(diag M)` at qpos0 IS `meaninertia * nv`; measured on dog,
        # 35.635564 both ways) evaluated at the wrong point.
        #
        # It scales BOTH exit tests, `improvement < tol` and `gradient < tol`,
        # so a pose-dependent scale makes the effective tolerance wander with
        # the configuration. Measured on dog at its settled pose: 34.107946
        # against 35.635564, i.e. a tolerance 1.045x looser than MuJoCo's.
        # Unbounded in general — a model that folds up moves its diagonal a lot
        # further than 4.5%.
        #
        # ⚠ THIS IS NOT A FIX FOR THE OPEN DOG RESIDUAL and must not be read as
        # one: tightening `NEWTON_TOL_GPU` to 1e-14 leaves our answer identical
        # to the last digit, so the exit threshold is not what is holding it.
        # This is a fidelity correction on its own merits.
        #
        # `meaninertia` reached the model meta with `mj_solNoSlip`, which needs
        # it for the same reason.
        # ⚠ STAY IN `DTYPE`. Computing this in Float64 makes the enclosing
        # kernel return a double and Metal rejects the module outright
        # ("returns unsupported type 'double'"), which is a BUILD failure on
        # every GPU model, not a dog-only one.
        var scale_d = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_MEANINERTIA]
        ) * Scalar[DTYPE](NV if NV > 1 else 1)
        var scale = (
            Scalar[DTYPE](1) / scale_d
            if scale_d > Scalar[DTYPE](1e-10)
            else Scalar[DTYPE](1)
        )

        # Working arrays
        var jar = InlineArray[Scalar[DTYPE], ME](uninitialized=True)
        var force = InlineArray[Scalar[DTYPE], ME](uninitialized=True)
        var H = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        var L_chol = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        var grad = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var search = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        var Mv = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

        # Initial jar + force + qfrc
        var qfrc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        pyramidal_edge_forces[DTYPE, NV, ME, V_SIZE](
            num_edges, Je, De, bias_e, kind_e, R_e, floss_e,
            qacc, jar, force, state_e, qfrc
        )

        # Newton iterations
        for iter_n in range(NEWTON_ITER_GPU):
            # Gradient
            var grad_norm: Scalar[DTYPE] = 0
            for i in range(NV):
                grad[i] = Ma[i] - f_smooth[i] - qfrc[i]
                grad_norm += grad[i] * grad[i]

            if scale * sqrt(grad_norm) < Scalar[DTYPE](NEWTON_TOL_GPU):
                break

            # Build Hessian H = M + sum_active(D[e] * Je^T * Je)
            for i in range(NV):
                for j in range(NV):
                    H[i * NV + j] = M_local[i * NV + j]
            for e_idx in range(num_edges):
                if state_e[e_idx] == SROW_QUADRATIC:
                    for i in range(NV):
                        for j in range(NV):
                            H[i * NV + j] += (
                                De[e_idx]
                                * Je[e_idx * NV + i]
                                * Je[e_idx * NV + j]
                            )

            # Cholesky solve
            var chol_ok = chol_factor_inline[DTYPE, NV, M_SIZE](H, L_chol)
            if not chol_ok:
                for i in range(NV):
                    H[i * NV + i] += Scalar[DTYPE](1e-6)
                _ = chol_factor_inline[DTYPE, NV, M_SIZE](H, L_chol)
            chol_solve_inline[DTYPE, NV, M_SIZE, V_SIZE](
                L_chol, grad, search
            )
            for i in range(NV):
                search[i] = -search[i]

            # Mv = M * search
            for i in range(NV):
                Mv[i] = Scalar[DTYPE](0)
                for j in range(NV):
                    Mv[i] += M_local[i * NV + j] * search[j]

            # Analytical Newton linesearch (matches CPU primal_linesearch_with_D)
            var alpha = pyramidal_linesearch[
                DTYPE, NV, ME, V_SIZE, LINESEARCH_ITER, PRIMAL_MINVAL_GPU
            ](
                num_edges, Je, De, kind_e, R_e, floss_e, search, Mv, Ma,
                f_smooth, qacc, qacc_smooth, jar,
            )

            if alpha < Scalar[DTYPE](1e-10):
                break

            # Save old state for cost revert (matching CPU solver)
            var old_qacc = InlineArray[Scalar[DTYPE], V_SIZE](
                uninitialized=True
            )
            var old_Ma = InlineArray[Scalar[DTYPE], V_SIZE](
                uninitialized=True
            )
            var old_jar = InlineArray[Scalar[DTYPE], ME](uninitialized=True)
            var old_force = InlineArray[Scalar[DTYPE], ME](
                uninitialized=True
            )
            var old_qfrc = InlineArray[Scalar[DTYPE], V_SIZE](
                uninitialized=True
            )
            for i in range(NV):
                old_qacc[i] = qacc[i]
                old_Ma[i] = Ma[i]
                old_qfrc[i] = qfrc[i]
            for e_idx in range(num_edges):
                old_jar[e_idx] = jar[e_idx]
                old_force[e_idx] = force[e_idx]

            # Compute old cost: gauss + constraint
            var old_cost: Scalar[DTYPE] = 0
            for i in range(NV):
                old_cost += (
                    Scalar[DTYPE](0.5)
                    * (Ma[i] - f_smooth[i])
                    * (qacc[i] - qacc_smooth[i])
                )
            for e_idx in range(num_edges):
                old_cost += scalar_row_cost[DTYPE](
                    state_e[e_idx], jar[e_idx], De[e_idx], R_e[e_idx],
                    floss_e[e_idx],
                )

            # Update qacc, Ma
            for i in range(NV):
                qacc[i] += alpha * search[i]
                Ma[i] += alpha * Mv[i]

            # Recompute jar, force, qfrc
            pyramidal_edge_forces[DTYPE, NV, ME, V_SIZE](
                num_edges, Je, De, bias_e, kind_e, R_e, floss_e,
                qacc, jar, force, state_e, qfrc
            )

            # Compute new cost and check improvement
            var new_cost: Scalar[DTYPE] = 0
            for i in range(NV):
                new_cost += (
                    Scalar[DTYPE](0.5)
                    * (Ma[i] - f_smooth[i])
                    * (qacc[i] - qacc_smooth[i])
                )
            for e_idx in range(num_edges):
                new_cost += scalar_row_cost[DTYPE](
                    state_e[e_idx], jar[e_idx], De[e_idx], R_e[e_idx],
                    floss_e[e_idx],
                )

            var improvement = scale * (old_cost - new_cost)
            if improvement < Scalar[DTYPE](NEWTON_TOL_GPU) and iter_n > 0:
                if improvement < Scalar[DTYPE](0):
                    # Cost increased — revert to old state
                    for i in range(NV):
                        qacc[i] = old_qacc[i]
                        Ma[i] = old_Ma[i]
                        qfrc[i] = old_qfrc[i]
                    for e_idx in range(num_edges):
                        jar[e_idx] = old_jar[e_idx]
                        force[e_idx] = old_force[e_idx]
                break

        # ── mj_solNoSlip ───────────────────────────────────────────────────
        # A friction-only Gauss-Seidel sweep with the NORMAL forces frozen,
        # run after the primal solve. Off unless the model asks for it
        # (`<option noslip_iterations>`); dm_control's dog is the only in-scope
        # model that does, and there it is first-order — 2.9e-2 of qvel on the
        # first contacting step — not a rounding refinement.
        #
        # PYRAMIDAL path only, and that is not an oversight: this is the
        # pyramidal branch of the solver, and `noslip.mojo` implements the
        # matching branch of MuJoCo's routine. The elliptic path below does
        # NOT call it, so an elliptic model with `noslip_iterations` set gets
        # the pass silently skipped — which is exactly why `ModelDefFromXML`
        # makes `noslip_iter > 0` a build error unless the model opts in.
        comptime if NOSLIP_ITER > 0:
            noslip_pyramidal[
                DTYPE, NV, ME, V_SIZE, MC, MAX_CONTACTS, MAX_CONDIM,
                BATCH, NOSLIP_ITER,
            ](
                env,
                nc,
                num_edges,
                contacts,
                m_inv,
                Je,
                bias_e,
                kind_e,
                R_e,
                floss_e,
                qacc_smooth,
                # `scale` = 1 / (meaninertia * max(1, nv)) and `tolerance` =
                # opt.noslip_tolerance. Both must be MuJoCo's or the sweep
                # stops on a different iteration — see the note on
                # MODEL_META_IDX_MEANINERTIA.
                1.0
                / (
                    Float64(
                        rebind[Scalar[DTYPE]](
                            mmeta[MODEL_META_IDX_MEANINERTIA]
                        )
                    )
                    * Float64(NV if NV > 1 else 1)
                ),
                NOSLIP_TOLERANCE,
                qacc,
                jar,
                force,
                qfrc,
            )

        # Write qacc back
        for i in range(NV):
            qacc_constrained[env, i] = qacc[i]

        # Write forces to state: reconstruct per-contact N/T1/T2
        for c in range(nc):
            var fn_c: Scalar[DTYPE] = 0
            var ft1_c: Scalar[DTYPE] = 0
            var ft2_c: Scalar[DTYPE] = 0
            var mu_c = rebind[Scalar[DTYPE]](
                solver[env, pyr_sc + 2 * NE * MC + c]
            )
            var safe_mu = mu_c
            if safe_mu < Scalar[DTYPE](1e-8):
                safe_mu = Scalar[DTYPE](1e-8)
            # f_n = sum of edge forces / num_tangent_dirs
            # f_tk = (f_edge_pos - f_edge_neg) * mu
            var f_e0 = force[c * NE + 0]
            var f_e1 = force[c * NE + 1]
            var f_e2 = force[c * NE + 2]
            var f_e3 = force[c * NE + 3]
            # `mju_decodePyramid`: the normal force is the SUM of the four edge
            # forces, NOT half of it. Both engines build each edge as
            # `Jn +- mu*Jt` with a FULL Jn (engine_core_constraint.c:1003), so
            # halving it made every pyramidal contact RECORD read half true
            # while qacc stayed correct — the solver works in edge forces and
            # only this write-back was wrong. Its two consumers are cfrc_ext
            # (hence Ant's contact_cost, a squared norm that had been costing a
            # quarter of what it should) and the quadruped force/torque
            # sensors. Fixed 2026-07-31.
            fn_c = f_e0 + f_e1 + f_e2 + f_e3
            var c_off = c * CONTACT_SIZE
            # ⚠ A FRICTIONLESS CONTACT HAS NO TANGENTIAL FORCE, and this
            # decode cannot know that from the edge forces alone. At condim 1
            # only edge 0 is live and edges 1..3 are zero, so `(f_e0 - f_e1)`
            # is `f_e0` and the record picks up a spurious `mu * f_n` of
            # friction. Measured on dog before this guard: `ft1/f_n = 0.9002`
            # on all three of its frictionless contacts — exactly the model's
            # default `friction="0.9"` — against MuJoCo's 0.
            #
            # It only became reachable when condim-1 contacts started producing
            # a row at all (see `_precompute_contact_friction`); before that
            # every edge force was zero and this read 0 for the right value by
            # accident, alongside an `f_n` of 0 that was simply wrong.
            #
            # `qacc` is NOT affected — that row's Jacobian is the pure normal,
            # so the solve stays frictionless. The damage is confined to the
            # record's consumers: `cfrc_ext` (hence contact-cost reward terms)
            # and the force/touch sensors, which is the fourth instance of this
            # write-back failure mode in this file's history.
            var dim_c = Int(
                rebind[Scalar[DTYPE]](contacts[env, c_off + CONTACT_IDX_CONDIM])
            )
            if dim_c > 1:
                ft1_c = (f_e0 - f_e1) * safe_mu
                ft2_c = (f_e2 - f_e3) * safe_mu
            contacts[env, c_off + CONTACT_IDX_FORCE_N] = fn_c
            contacts[env, c_off + CONTACT_IDX_FORCE_T1] = ft1_c
            contacts[env, c_off + CONTACT_IDX_FORCE_T2] = ft2_c

        # Joint limits are now handled as edges in the Newton solver above.
        # Only equality constraints remain as a separate post-solve step.
        comptime SOLVER_ITER_GPU: Int = 50
        comptime if NEQUALITY > 0:
            _equality_env[
                DTYPE, NV, NBODY, NJOINT, NEQUALITY, NTENDON, NSITE, V_SIZE,
                BATCH, SOLVER_ITER_GPU,
            ](
                env, qvel, xpos, xquat, subtree_com, joints, bodies, mmeta,
                equality, tendons, sites, body_invweight0, dof_invweight0,
                cdof, m_inv, qacc_constrained,
            )
        # SKIP_FIXED: the fixed-tendon equalities are rows of the system
        # solved above. Only SPATIAL equality tendons (none in the tree, but
        # the builder skips them by design) are left for this pass.
        comptime if NTENDON > 0:
            _tendon_env[
                DTYPE, NQ, NV, NBODY, NJOINT, NTENDON, NSITE, BATCH,
                SOLVER_ITER_GPU, SKIP_FIXED=True,
            ](
                env, qpos, qvel, joints, mmeta, tendons, sites,
                body_invweight0, dof_invweight0, m_inv, qacc_constrained,
            )
        return  # PYRAMIDAL path complete

    # === ELLIPTIC path (existing code) ===
    # === Cache loop-invariant contact data into local InlineArrays ===
    # Jn, Jt1, Jt2, mu, D_n, D_f, dist, pos_bias, bt1, bt2 never change
    # during Newton iterations — load once to avoid ~1000 workspace reads/iter.
    var Jn_c = InlineArray[Scalar[DTYPE], MC * V_SIZE](uninitialized=True)
    var Jt1_c = InlineArray[Scalar[DTYPE], MC * V_SIZE](uninitialized=True)
    var Jt2_c = InlineArray[Scalar[DTYPE], MC * V_SIZE](uninitialized=True)
    var mu_cache = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    var D_n_cache = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    var D_f_cache = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    var dist_cache = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    var pb_cache = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    var bt1_cache = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    var bt2_cache = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    for c in range(nc):
        dist_cache[c] = rebind[Scalar[DTYPE]](
            solver[env, ws_c_dist_idx + c]
        )
        mu_cache[c] = rebind[Scalar[DTYPE]](solver[env, ws_mu_idx + c])
        D_n_cache[c] = rebind[Scalar[DTYPE]](solver[env, ws_D_n_idx + c])
        D_f_cache[c] = rebind[Scalar[DTYPE]](solver[env, ws_D_f_idx + c])
        pb_cache[c] = rebind[Scalar[DTYPE]](
            solver[env, ws_pos_bias_idx + c]
        )
        bt1_cache[c] = rebind[Scalar[DTYPE]](solver[env, ws_bt1_idx + c])
        bt2_cache[c] = rebind[Scalar[DTYPE]](solver[env, ws_bt2_idx + c])
        for i in range(NV):
            Jn_c[c * NV + i] = rebind[Scalar[DTYPE]](
                solver[env, ws_J_n_idx + c * NV + i]
            )
            Jt1_c[c * NV + i] = rebind[Scalar[DTYPE]](
                solver[env, ws_Jt1_idx + c * NV + i]
            )
            Jt2_c[c * NV + i] = rebind[Scalar[DTYPE]](
                solver[env, ws_Jt2_idx + c * NV + i]
            )

    # === Scalar rows: joint limits + dry-friction dofs ===
    # These used to be PGS post-passes that ran AFTER this solve, so the
    # contact rows were solved as if they did not exist. They are rows of the
    # same system — see constraints/scalar_rows.mojo for the measurement that
    # established this. J = sign * e_dof, so only (dof, sign) is stored.
    comptime MAXS = max_scalar_rows[NV, NJOINT]()
    var sr_dof = InlineArray[Int, MAXS](fill=0)
    var sr_kind = InlineArray[Int, MAXS](fill=0)
    var sr_sign = InlineArray[Scalar[DTYPE], MAXS](fill=Scalar[DTYPE](0))
    var sr_D = InlineArray[Scalar[DTYPE], MAXS](fill=Scalar[DTYPE](0))
    var sr_R = InlineArray[Scalar[DTYPE], MAXS](fill=Scalar[DTYPE](0))
    var sr_bias = InlineArray[Scalar[DTYPE], MAXS](fill=Scalar[DTYPE](0))
    var sr_floss = InlineArray[Scalar[DTYPE], MAXS](fill=Scalar[DTYPE](0))
    var ns = build_scalar_rows[DTYPE, NQ, NV, NJOINT, BATCH, MAXS](
        env, qpos, qvel, joints, mmeta, dof_invweight0, m_inv,
        sr_dof, sr_kind, sr_sign, sr_D, sr_R, sr_bias, sr_floss,
    )
    var sr_jar = InlineArray[Scalar[DTYPE], MAXS](fill=Scalar[DTYPE](0))
    var sr_f = InlineArray[Scalar[DTYPE], MAXS](fill=Scalar[DTYPE](0))
    var sr_st = InlineArray[Int, MAXS](fill=0)
    var sr_Js = InlineArray[Scalar[DTYPE], MAXS](fill=Scalar[DTYPE](0))

    # === Fixed-tendon EQUALITY rows (dense J) ===
    # These ran as a post-pass on this path until 2026-08-01, which is the same
    # defect `build_scalar_rows` above exists to fix, one constraint type over.
    # Measured on dm_control manipulator `bring_ball`, where the `coupling`
    # tendon holds thumb == finger while the grasped ball's two contacts break
    # that symmetry: the post-pass left qacc[thumb]/qacc[finger] at
    # +60.12/-58.11 where MuJoCo has -832.98/-839.37 — equal and OPPOSITE, i.e.
    # the row was not enforced against the contact solve at all. Poses where the
    # contacts are SYMMETRIC (a closed empty hand, 18 contact rows) were exact
    # even then, because the equality had nothing to correct — which is why this
    # needed a domain with an object in the hand to surface.
    #
    # A scalar row is stored as `(dof, sign)` to keep the elliptic core's local
    # memory at O(rows); an equality row needs a full NV Jacobian. That is the
    # cost this deferral was avoiding, and it is `NTENDON * NV` floats — 22 for
    # manipulator, 88 for quadruped — next to the contact block's `MC * NV * 6`.
    #
    # The row is built by the SAME function the pyramidal edge list uses, so
    # both cones get bit-identical (J, D, bias).
    comptime MAXEQ = _max_one[NTENDON]()
    var eq_J = InlineArray[Scalar[DTYPE], MAXEQ * V_SIZE](
        fill=Scalar[DTYPE](0)
    )
    var eq_D = InlineArray[Scalar[DTYPE], MAXEQ](fill=Scalar[DTYPE](0))
    var eq_bias = InlineArray[Scalar[DTYPE], MAXEQ](fill=Scalar[DTYPE](0))
    var eq_kind = InlineArray[Int, MAXEQ](fill=0)
    var eq_jar = InlineArray[Scalar[DTYPE], MAXEQ](fill=Scalar[DTYPE](0))
    var eq_f = InlineArray[Scalar[DTYPE], MAXEQ](fill=Scalar[DTYPE](0))
    var eq_Js = InlineArray[Scalar[DTYPE], MAXEQ](fill=Scalar[DTYPE](0))
    var neq_rows = 0
    comptime if NTENDON > 0:
        build_tendon_equality_rows[
            DTYPE, NQ, NV, NJOINT, NTENDON, V_SIZE, MAXEQ, BATCH
        ](
            env, qpos, qvel, tendons, joints, mmeta, m_inv,
            eq_J, eq_D, eq_bias, eq_kind, neq_rows,
        )

    # === Step 2: Initialize local InlineArrays from workspace ===
    var H = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    var L_chol = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    var qacc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var qacc_sm = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var qfrc_sm = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var Ma = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var grad = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var search = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var Mv = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

    # Load M into H (primal Hessian starts as M_hat)
    for k in range(NV * NV):
        H[k] = rebind[Scalar[DTYPE]](M[env, k])

    # Cache M locally — saves NV² workspace reads per Newton iteration (for Mv = M*search)
    var M_local = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    for k in range(NV * NV):
        M_local[k] = H[k]

    # qacc_sm = unconstrained qacc (set by integrator), save a copy
    for i in range(NV):
        var q_i = rebind[Scalar[DTYPE]](qacc_constrained[env, i])
        qacc[i] = q_i
        qacc_sm[i] = q_i

    # Ma = M_local * qacc (uses cached M — no workspace reads)
    for i in range(NV):
        var s: Scalar[DTYPE] = 0
        for j in range(NV):
            s += M_local[i * NV + j] * qacc[j]
        Ma[i] = s

    # qfrc_sm = M * qacc (matching CPU's qfrc_smooth = M * qacc_smooth)
    # Using Ma directly avoids LDL round-trip error
    for i in range(NV):
        qfrc_sm[i] = Ma[i]

    # Same model-constant scale as the PYRAMIDAL path; see the note there.
    # `mj_solPrimal` is shared by both cones in MuJoCo, so the ELLIPTIC leg
    # took the identical pose-dependent-trace deviation and is corrected with
    # it rather than left as the odd one out.
    var scale_de = rebind[Scalar[DTYPE]](
        mmeta[MODEL_META_IDX_MEANINERTIA]
    ) * Scalar[DTYPE](NV if NV > 1 else 1)
    var scale = (
        Scalar[DTYPE](1) / scale_de
        if scale_de > Scalar[DTYPE](1e-10)
        else Scalar[DTYPE](1)
    )

    # === Mutable per-contact state: kept in InlineArrays, written to state buffer at end ===
    var fn_arr = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    var ft1_arr = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    var ft2_arr = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    var jar_n_arr = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    var jar_t1_arr = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    var jar_t2_arr = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
    var cs_arr = InlineArray[Int, MC](uninitialized=True)

    # === Step 3: Compute initial jar and forces via 3-zone cone logic ===
    for c in range(nc):
        if dist_cache[c] >= Scalar[DTYPE](0):
            fn_arr[c] = 0
            ft1_arr[c] = 0
            ft2_arr[c] = 0
            jar_n_arr[c] = 0
            jar_t1_arr[c] = 0
            jar_t2_arr[c] = 0
            cs_arr[c] = 0
            continue

        var jar_n: Scalar[DTYPE] = pb_cache[c]
        var jar_t1: Scalar[DTYPE] = bt1_cache[c]
        var jar_t2: Scalar[DTYPE] = bt2_cache[c]
        for i in range(NV):
            var qa_i = qacc[i]
            jar_n += Jn_c[c * NV + i] * qa_i
            jar_t1 += Jt1_c[c * NV + i] * qa_i
            jar_t2 += Jt2_c[c * NV + i] * qa_i
        jar_n_arr[c] = jar_n
        jar_t1_arr[c] = jar_t1
        jar_t2_arr[c] = jar_t2

        var mu = mu_cache[c]
        var D_n = D_n_cache[c]
        var D_f = D_f_cache[c]
        var T = sqrt(jar_t1 * jar_t1 + jar_t2 * jar_t2)
        var T_safe = T
        if T_safe < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
            T_safe = Scalar[DTYPE](PRIMAL_MINVAL_GPU)

        if jar_n >= mu * T_safe:
            fn_arr[c] = 0
            ft1_arr[c] = 0
            ft2_arr[c] = 0
            cs_arr[c] = 0  # SATISFIED
        elif mu * jar_n + T <= Scalar[DTYPE](0):
            fn_arr[c] = -D_n * jar_n
            ft1_arr[c] = -D_f * jar_t1
            ft2_arr[c] = -D_f * jar_t2
            cs_arr[c] = 1  # QUADRATIC
        else:
            var s = jar_n - mu * T_safe
            var Dm = D_n / (Scalar[DTYPE](1.0) + mu * mu)
            fn_arr[c] = -Dm * s
            ft1_arr[c] = Dm * mu * s * jar_t1 / T_safe
            ft2_arr[c] = Dm * mu * s * jar_t2 / T_safe
            cs_arr[c] = 2  # CONE

    # Scalar rows: same 3-zone logic, one dof each.
    for s in range(ns):
        var jar_s = sr_bias[s] + sr_sign[s] * qacc[sr_dof[s]]
        sr_jar[s] = jar_s
        var st = scalar_row_state[DTYPE](
            sr_kind[s], jar_s, sr_R[s], sr_floss[s]
        )
        sr_st[s] = st
        sr_f[s] = scalar_row_force[DTYPE](st, jar_s, sr_D[s], sr_floss[s])

    # Equality rows: BILATERAL, so unconditionally QUADRATIC — no state, and
    # `f = -D*jar` always.
    for e in range(neq_rows):
        var jar_e = eq_bias[e]
        for d in range(NV):
            jar_e += eq_J[e * NV + d] * qacc[d]
        eq_jar[e] = jar_e
        eq_f[e] = -eq_D[e] * jar_e

    # === Step 4: Build Hessian H = M + J^T*D*J (cone-aware, using cached Jacobians) ===
    # Scalar rows contribute D only on their own dof (J = sign*e_dof, so
    # J^T*J = e_dof*e_dof^T — the sign squares away).
    for s in range(ns):
        if sr_st[s] == SROW_QUADRATIC:
            var d = sr_dof[s]
            H[d * NV + d] += sr_D[s]
    # Equality rows have a DENSE J, so their contribution is a full rank-1
    # outer product rather than a diagonal bump.
    for e in range(neq_rows):
        for a in range(NV):
            var Ja = eq_J[e * NV + a]
            if Ja == Scalar[DTYPE](0):
                continue
            for b in range(NV):
                H[a * NV + b] += eq_D[e] * Ja * eq_J[e * NV + b]
    for c in range(nc):
        var cs = cs_arr[c]
        if cs == 0:  # SATISFIED
            continue

        var mu = mu_cache[c]
        var D_n = D_n_cache[c]
        var D_f = D_f_cache[c]

        if cs == 1:  # QUADRATIC: standard rank-1 updates
            for i in range(NV):
                for j in range(NV):
                    H[i * NV + j] += (
                        D_n * Jn_c[c * NV + i] * Jn_c[c * NV + j]
                        + D_f * Jt1_c[c * NV + i] * Jt1_c[c * NV + j]
                        + D_f * Jt2_c[c * NV + i] * Jt2_c[c * NV + j]
                    )
        else:  # CONE: cone Hessian (coupled normal+friction)
            var jar_n = jar_n_arr[c]
            var jar_t1 = jar_t1_arr[c]
            var jar_t2 = jar_t2_arr[c]
            var T_sq = jar_t1 * jar_t1 + jar_t2 * jar_t2
            var T = sqrt(T_sq)
            var T_safe = T
            if T_safe < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                T_safe = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
            var s = jar_n - mu * T_safe
            var Dm = D_n / (Scalar[DTYPE](1.0) + mu * mu)
            var h_nt1 = -Dm * mu * jar_t1 / T_safe
            var h_nt2 = -Dm * mu * jar_t2 / T_safe
            var T2_safe = T_safe * T_safe
            var h_t1t1 = (
                Dm * mu * mu * jar_t1 * jar_t1 / T2_safe
                + Dm
                * mu
                * s
                / T_safe
                * (jar_t1 * jar_t1 / T2_safe - Scalar[DTYPE](1.0))
            )
            var h_t2t2 = (
                Dm * mu * mu * jar_t2 * jar_t2 / T2_safe
                + Dm
                * mu
                * s
                / T_safe
                * (jar_t2 * jar_t2 / T2_safe - Scalar[DTYPE](1.0))
            )
            var h_t1t2 = (
                (Dm * mu * mu + Dm * mu * s / T_safe)
                * jar_t1
                * jar_t2
                / T2_safe
            )
            for i in range(NV):
                for j in range(NV):
                    H[i * NV + j] += (
                        Dm * Jn_c[c * NV + i] * Jn_c[c * NV + j]
                        + h_nt1
                        * (
                            Jn_c[c * NV + i] * Jt1_c[c * NV + j]
                            + Jt1_c[c * NV + i] * Jn_c[c * NV + j]
                        )
                        + h_nt2
                        * (
                            Jn_c[c * NV + i] * Jt2_c[c * NV + j]
                            + Jt2_c[c * NV + i] * Jn_c[c * NV + j]
                        )
                        + h_t1t1 * Jt1_c[c * NV + i] * Jt1_c[c * NV + j]
                        + h_t2t2 * Jt2_c[c * NV + i] * Jt2_c[c * NV + j]
                        + h_t1t2
                        * (
                            Jt1_c[c * NV + i] * Jt2_c[c * NV + j]
                            + Jt2_c[c * NV + i] * Jt1_c[c * NV + j]
                        )
                    )

    # Cholesky factorize H (with regularization on rank deficiency)
    var chol_ok_gpu = chol_factor_inline[DTYPE, NV, M_SIZE](H, L_chol)
    if not chol_ok_gpu:
        for i in range(NV):
            H[i * NV + i] = H[i * NV + i] + Scalar[DTYPE](1e-6)
        _ = chol_factor_inline[DTYPE, NV, M_SIZE](H, L_chol)

    # === Precompute qfrc_c = J^T * force (replaces per-iteration gradient workspace reads) ===
    # Updated after each force update instead of recomputing from workspace each gradient step.
    var qfrc_c = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        qfrc_c[i] = Scalar[DTYPE](0)
    for c in range(nc):
        if cs_arr[c] == 0:
            continue
        for i in range(NV):
            qfrc_c[i] += (
                Jn_c[c * NV + i] * fn_arr[c]
                + Jt1_c[c * NV + i] * ft1_arr[c]
                + Jt2_c[c * NV + i] * ft2_arr[c]
            )
    for s in range(ns):
        qfrc_c[sr_dof[s]] += sr_sign[s] * sr_f[s]
    for e in range(neq_rows):
        for d in range(NV):
            qfrc_c[d] += eq_J[e * NV + d] * eq_f[e]

    # === Step 5: Newton iteration loop ===
    for _iter in range(NEWTON_ITER_GPU):
        # Gradient = Ma - qfrc_sm - qfrc_c (pure InlineArray reads — no workspace access)
        var grad_norm_sq: Scalar[DTYPE] = 0
        for i in range(NV):
            grad[i] = Ma[i] - qfrc_sm[i] - qfrc_c[i]
            grad_norm_sq += grad[i] * grad[i]

        # Convergence check
        if scale * sqrt(grad_norm_sq) < Scalar[DTYPE](NEWTON_TOL_GPU):
            break

        # Newton direction: search = -H^{-1} * grad
        chol_solve_inline[DTYPE, NV, M_SIZE, V_SIZE](L_chol, grad, search)
        var search_ok_gpu = True
        for i in range(NV):
            search[i] = -search[i]
            if search[i] != search[i]:
                search_ok_gpu = False
        if not search_ok_gpu:
            break

        # Mv = M_local * search (InlineArray reads only — no workspace access)
        for i in range(NV):
            var s: Scalar[DTYPE] = 0
            for j in range(NV):
                s += M_local[i * NV + j] * search[j]
            Mv[i] = s

        # Precompute J * search per contact (using cached Jacobians — no workspace access)
        var Js_n = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var Js_t1 = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        var Js_t2 = InlineArray[Scalar[DTYPE], MC](uninitialized=True)
        for c in range(nc):
            if dist_cache[c] >= Scalar[DTYPE](0):
                Js_n[c] = 0
                Js_t1[c] = 0
                Js_t2[c] = 0
                continue
            var js_n: Scalar[DTYPE] = 0
            var js_t1: Scalar[DTYPE] = 0
            var js_t2: Scalar[DTYPE] = 0
            for i in range(NV):
                var s_i = search[i]
                js_n += Jn_c[c * NV + i] * s_i
                js_t1 += Jt1_c[c * NV + i] * s_i
                js_t2 += Jt2_c[c * NV + i] * s_i
            Js_n[c] = js_n
            Js_t1[c] = js_t1
            Js_t2[c] = js_t2
        for s in range(ns):
            sr_Js[s] = sr_sign[s] * search[sr_dof[s]]
        for e in range(neq_rows):
            var jv = Scalar[DTYPE](0)
            for d in range(NV):
                jv += eq_J[e * NV + d] * search[d]
            eq_Js[e] = jv

        # Analytical Newton linesearch (matches CPU primal_linesearch_with_D)
        # Gauss coefficients for derivative: d_gauss/dalpha = ga*alpha + gb
        var ga: Scalar[DTYPE] = 0
        var gb: Scalar[DTYPE] = 0
        for i in range(NV):
            ga += Mv[i] * search[i]
            gb += (Ma[i] - qfrc_sm[i]) * search[i]

        # Evaluate d1, d2 at alpha=0
        var p0_d1 = gb
        var p0_d2 = ga
        for c in range(nc):
            if dist_cache[c] >= Scalar[DTYPE](0):
                continue
            var N0 = jar_n_arr[c]
            var T10 = jar_t1_arr[c]
            var T20 = jar_t2_arr[c]
            var mu = mu_cache[c]
            var D_n = D_n_cache[c]
            var D_f = D_f_cache[c]
            var T0_sq = T10 * T10 + T20 * T20
            var T0 = sqrt(T0_sq)
            var T0_safe = T0
            if T0_safe < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                T0_safe = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
            if N0 >= Scalar[DTYPE](0) and N0 * N0 >= mu * mu * T0_sq:
                pass  # SATISFIED
            elif mu * N0 + T0 <= Scalar[DTYPE](0):
                # QUADRATIC
                p0_d1 += D_n * N0 * Js_n[c] + D_f * (
                    T10 * Js_t1[c] + T20 * Js_t2[c]
                )
                p0_d2 += D_n * Js_n[c] * Js_n[c] + D_f * (
                    Js_t1[c] * Js_t1[c] + Js_t2[c] * Js_t2[c]
                )
            else:
                # CONE
                var Dm = D_n / (Scalar[DTYPE](1.0) + mu * mu)
                var s0 = N0 - mu * T0
                var dTda = (T10 * Js_t1[c] + T20 * Js_t2[c]) / T0_safe
                var dsda = Js_n[c] - mu * dTda
                p0_d1 += Dm * s0 * dsda
                var Jv_f_sq = Js_t1[c] * Js_t1[c] + Js_t2[c] * Js_t2[c]
                var d2sda2 = -mu * (Jv_f_sq - dTda * dTda) / T0_safe
                p0_d2 += Dm * (dsda * dsda + s0 * d2sda2)
        # Scalar rows. d(cost)/dalpha = -f*Jv in EVERY state, and the second
        # derivative is D*Jv^2 only where the row is quadratic.
        for s in range(ns):
            p0_d1 += -sr_f[s] * sr_Js[s]
            if sr_st[s] == SROW_QUADRATIC:
                p0_d2 += sr_D[s] * sr_Js[s] * sr_Js[s]
        for e in range(neq_rows):
            p0_d1 += -eq_f[e] * eq_Js[e]
            p0_d2 += eq_D[e] * eq_Js[e] * eq_Js[e]
        if p0_d2 < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
            p0_d2 = Scalar[DTYPE](PRIMAL_MINVAL_GPU)

        var alpha: Scalar[DTYPE] = 0
        if p0_d1 < Scalar[DTYPE](0):
            # Phase 1: initial Newton step
            var p1_alpha = -p0_d1 / p0_d2

            var snorm_sq: Scalar[DTYPE] = 0
            for i in range(NV):
                snorm_sq += search[i] * search[i]
            var gtol = (
                Scalar[DTYPE](NEWTON_TOL_GPU) * sqrt(snorm_sq) / scale
            )
            var gtol_sq = gtol * gtol

            # Inline eval at p1_alpha
            var p1_d1 = ga * p1_alpha + gb
            var p1_d2_v = ga
            for c in range(nc):
                if dist_cache[c] >= Scalar[DTYPE](0):
                    continue
                var tN = jar_n_arr[c] + p1_alpha * Js_n[c]
                var tT1 = jar_t1_arr[c] + p1_alpha * Js_t1[c]
                var tT2 = jar_t2_arr[c] + p1_alpha * Js_t2[c]
                var mu = mu_cache[c]
                var D_n = D_n_cache[c]
                var D_f = D_f_cache[c]
                var tT_sq = tT1 * tT1 + tT2 * tT2
                var tT = sqrt(tT_sq)
                var tT_s = tT
                if tT_s < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                    tT_s = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
                if tN >= Scalar[DTYPE](0) and tN * tN >= mu * mu * tT_sq:
                    pass
                elif mu * tN + tT <= Scalar[DTYPE](0):
                    p1_d1 += D_n * tN * Js_n[c] + D_f * (
                        tT1 * Js_t1[c] + tT2 * Js_t2[c]
                    )
                    p1_d2_v += D_n * Js_n[c] * Js_n[c] + D_f * (
                        Js_t1[c] * Js_t1[c] + Js_t2[c] * Js_t2[c]
                    )
                else:
                    var Dm = D_n / (Scalar[DTYPE](1.0) + mu * mu)
                    var s_v = tN - mu * tT
                    var dTda = (tT1 * Js_t1[c] + tT2 * Js_t2[c]) / tT_s
                    var dsda = Js_n[c] - mu * dTda
                    p1_d1 += Dm * s_v * dsda
                    var Jvf = Js_t1[c] * Js_t1[c] + Js_t2[c] * Js_t2[c]
                    var d2s = -mu * (Jvf - dTda * dTda) / tT_s
                    p1_d2_v += Dm * (dsda * dsda + s_v * d2s)
            for s in range(ns):
                var tj = sr_jar[s] + p1_alpha * sr_Js[s]
                var tst = scalar_row_state[DTYPE](
                    sr_kind[s], tj, sr_R[s], sr_floss[s]
                )
                p1_d1 += (
                    -scalar_row_force[DTYPE](tst, tj, sr_D[s], sr_floss[s])
                    * sr_Js[s]
                )
                if tst == SROW_QUADRATIC:
                    p1_d2_v += sr_D[s] * sr_Js[s] * sr_Js[s]
            for e in range(neq_rows):
                var tje = eq_jar[e] + p1_alpha * eq_Js[e]
                p1_d1 += eq_D[e] * tje * eq_Js[e]
                p1_d2_v += eq_D[e] * eq_Js[e] * eq_Js[e]
            if p1_d2_v < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                p1_d2_v = Scalar[DTYPE](PRIMAL_MINVAL_GPU)

            alpha = p1_alpha
            if p1_d1 * p1_d1 >= gtol_sq:
                # Phase 2: one-sided Newton pursuit
                var dir_s = Scalar[DTYPE](-1) if p1_d1 > Scalar[DTYPE](
                    0
                ) else Scalar[DTYPE](1)
                var p2_alpha: Scalar[DTYPE] = 0
                var p2_d1 = p0_d1
                var bracket = False
                for _ in range(LINESEARCH_ITER):
                    p2_alpha = p1_alpha
                    p2_d1 = p1_d1
                    if p1_d2_v > Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                        p1_alpha = p1_alpha - p1_d1 / p1_d2_v
                    else:
                        p1_alpha = p1_alpha + dir_s
                    # Eval at new p1_alpha
                    p1_d1 = ga * p1_alpha + gb
                    p1_d2_v = ga
                    for c in range(nc):
                        if dist_cache[c] >= Scalar[DTYPE](0):
                            continue
                        var tN = jar_n_arr[c] + p1_alpha * Js_n[c]
                        var tT1 = jar_t1_arr[c] + p1_alpha * Js_t1[c]
                        var tT2 = jar_t2_arr[c] + p1_alpha * Js_t2[c]
                        var mu = mu_cache[c]
                        var D_n = D_n_cache[c]
                        var D_f = D_f_cache[c]
                        var tT_sq = tT1 * tT1 + tT2 * tT2
                        var tT = sqrt(tT_sq)
                        var tT_s = tT
                        if tT_s < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                            tT_s = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
                        if (
                            tN >= Scalar[DTYPE](0)
                            and tN * tN >= mu * mu * tT_sq
                        ):
                            pass
                        elif mu * tN + tT <= Scalar[DTYPE](0):
                            p1_d1 += D_n * tN * Js_n[c] + D_f * (
                                tT1 * Js_t1[c] + tT2 * Js_t2[c]
                            )
                            p1_d2_v += D_n * Js_n[c] * Js_n[c] + D_f * (
                                Js_t1[c] * Js_t1[c] + Js_t2[c] * Js_t2[c]
                            )
                        else:
                            var Dm = D_n / (Scalar[DTYPE](1.0) + mu * mu)
                            var s_v = tN - mu * tT
                            var dTda = (
                                tT1 * Js_t1[c] + tT2 * Js_t2[c]
                            ) / tT_s
                            var dsda = Js_n[c] - mu * dTda
                            p1_d1 += Dm * s_v * dsda
                            var Jvf = (
                                Js_t1[c] * Js_t1[c] + Js_t2[c] * Js_t2[c]
                            )
                            var d2s = -mu * (Jvf - dTda * dTda) / tT_s
                            p1_d2_v += Dm * (dsda * dsda + s_v * d2s)
                    for s in range(ns):
                        var tj = sr_jar[s] + p1_alpha * sr_Js[s]
                        var tst = scalar_row_state[DTYPE](
                            sr_kind[s], tj, sr_R[s], sr_floss[s]
                        )
                        p1_d1 += (
                            -scalar_row_force[DTYPE](
                                tst, tj, sr_D[s], sr_floss[s]
                            )
                            * sr_Js[s]
                        )
                        if tst == SROW_QUADRATIC:
                            p1_d2_v += sr_D[s] * sr_Js[s] * sr_Js[s]
                    for e in range(neq_rows):
                        var tje = eq_jar[e] + p1_alpha * eq_Js[e]
                        p1_d1 += eq_D[e] * tje * eq_Js[e]
                        p1_d2_v += eq_D[e] * eq_Js[e] * eq_Js[e]
                    if p1_d2_v < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                        p1_d2_v = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
                    if p1_d1 * p1_d1 < gtol_sq:
                        alpha = p1_alpha
                        break
                    if p1_d1 * dir_s > Scalar[DTYPE](0):
                        bracket = True
                        break
                if bracket:
                    # Phase 3: bracketed bisection
                    for _ in range(LINESEARCH_ITER):
                        var mid = (p1_alpha + p2_alpha) * Scalar[DTYPE](0.5)
                        var mid_d1 = ga * mid + gb
                        for c in range(nc):
                            if dist_cache[c] >= Scalar[DTYPE](0):
                                continue
                            var tN = jar_n_arr[c] + mid * Js_n[c]
                            var tT1 = jar_t1_arr[c] + mid * Js_t1[c]
                            var tT2 = jar_t2_arr[c] + mid * Js_t2[c]
                            var mu = mu_cache[c]
                            var D_n = D_n_cache[c]
                            var D_f = D_f_cache[c]
                            var tT_sq = tT1 * tT1 + tT2 * tT2
                            var tT = sqrt(tT_sq)
                            var tT_s = tT
                            if tT_s < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                                tT_s = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
                            if (
                                tN >= Scalar[DTYPE](0)
                                and tN * tN >= mu * mu * tT_sq
                            ):
                                pass
                            elif mu * tN + tT <= Scalar[DTYPE](0):
                                mid_d1 += D_n * tN * Js_n[c] + D_f * (
                                    tT1 * Js_t1[c] + tT2 * Js_t2[c]
                                )
                            else:
                                var Dm = D_n / (
                                    Scalar[DTYPE](1.0) + mu * mu
                                )
                                var s_v = tN - mu * tT
                                var dTda = (
                                    tT1 * Js_t1[c] + tT2 * Js_t2[c]
                                ) / tT_s
                                var dsda = Js_n[c] - mu * dTda
                                mid_d1 += Dm * s_v * dsda
                        for s in range(ns):
                            var tj = sr_jar[s] + mid * sr_Js[s]
                            var tst = scalar_row_state[DTYPE](
                                sr_kind[s], tj, sr_R[s], sr_floss[s]
                            )
                            mid_d1 += (
                                -scalar_row_force[DTYPE](
                                    tst, tj, sr_D[s], sr_floss[s]
                                )
                                * sr_Js[s]
                            )
                        for e in range(neq_rows):
                            var tje = eq_jar[e] + mid * eq_Js[e]
                            mid_d1 += eq_D[e] * tje * eq_Js[e]
                        if mid_d1 * mid_d1 < gtol_sq:
                            p1_alpha = mid
                            p1_d1 = mid_d1
                            break
                        if mid_d1 * p1_d1 > Scalar[DTYPE](0):
                            p1_alpha = mid
                            p1_d1 = mid_d1
                        else:
                            p2_alpha = mid
                            p2_d1 = mid_d1
                        if (p1_alpha - p2_alpha) * (
                            p1_alpha - p2_alpha
                        ) < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                            break
                    if p2_d1 * p2_d1 < p1_d1 * p1_d1:
                        alpha = p2_alpha
                    else:
                        alpha = p1_alpha
                elif p1_d1 * p1_d1 >= gtol_sq:
                    alpha = p1_alpha

        # If alpha is negligible, stop
        if alpha < Scalar[DTYPE](1e-12):
            break

        # Update qacc and Ma
        for i in range(NV):
            qacc[i] = qacc[i] + alpha * search[i]
            Ma[i] = Ma[i] + alpha * Mv[i]

        # Recompute jar and forces (using cached Jacobians — no workspace reads)
        var state_changed = False
        for c in range(nc):
            if dist_cache[c] >= Scalar[DTYPE](0):
                continue
            var old_cs = cs_arr[c]
            var jar_n: Scalar[DTYPE] = pb_cache[c]
            var jar_t1: Scalar[DTYPE] = bt1_cache[c]
            var jar_t2: Scalar[DTYPE] = bt2_cache[c]
            for i in range(NV):
                var qa_i = qacc[i]
                jar_n += Jn_c[c * NV + i] * qa_i
                jar_t1 += Jt1_c[c * NV + i] * qa_i
                jar_t2 += Jt2_c[c * NV + i] * qa_i
            jar_n_arr[c] = jar_n
            jar_t1_arr[c] = jar_t1
            jar_t2_arr[c] = jar_t2

            var mu = mu_cache[c]
            var D_n = D_n_cache[c]
            var D_f = D_f_cache[c]
            var T = sqrt(jar_t1 * jar_t1 + jar_t2 * jar_t2)
            var T_safe = T
            if T_safe < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                T_safe = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
            if jar_n >= mu * T_safe:
                fn_arr[c] = 0
                ft1_arr[c] = 0
                ft2_arr[c] = 0
                cs_arr[c] = 0
            elif mu * jar_n + T <= Scalar[DTYPE](0):
                fn_arr[c] = -D_n * jar_n
                ft1_arr[c] = -D_f * jar_t1
                ft2_arr[c] = -D_f * jar_t2
                cs_arr[c] = 1
            else:
                var s = jar_n - mu * T_safe
                var Dm = D_n / (Scalar[DTYPE](1.0) + mu * mu)
                fn_arr[c] = -Dm * s
                ft1_arr[c] = Dm * mu * s * jar_t1 / T_safe
                ft2_arr[c] = Dm * mu * s * jar_t2 / T_safe
                cs_arr[c] = 2
            if cs_arr[c] != old_cs:
                state_changed = True
        for s in range(ns):
            var old_st = sr_st[s]
            var jar_s = sr_bias[s] + sr_sign[s] * qacc[sr_dof[s]]
            sr_jar[s] = jar_s
            var st = scalar_row_state[DTYPE](
                sr_kind[s], jar_s, sr_R[s], sr_floss[s]
            )
            sr_st[s] = st
            sr_f[s] = scalar_row_force[DTYPE](st, jar_s, sr_D[s], sr_floss[s])
            if st != old_st:
                state_changed = True
        # Bilateral: no state, so nothing can flip `state_changed` — but jar
        # and f still track qacc every iteration.
        for e in range(neq_rows):
            var jar_e = eq_bias[e]
            for d in range(NV):
                jar_e += eq_J[e * NV + d] * qacc[d]
            eq_jar[e] = jar_e
            eq_f[e] = -eq_D[e] * jar_e

        # Recompute qfrc_c = J^T * updated forces (all InlineArray ops)
        for i in range(NV):
            qfrc_c[i] = Scalar[DTYPE](0)
        for c in range(nc):
            if cs_arr[c] == 0:
                continue
            for i in range(NV):
                qfrc_c[i] += (
                    Jn_c[c * NV + i] * fn_arr[c]
                    + Jt1_c[c * NV + i] * ft1_arr[c]
                    + Jt2_c[c * NV + i] * ft2_arr[c]
                )
        for s in range(ns):
            qfrc_c[sr_dof[s]] += sr_sign[s] * sr_f[s]
        for e in range(neq_rows):
            for d in range(NV):
                qfrc_c[d] += eq_J[e * NV + d] * eq_f[e]

        # Hessian rebuild if states changed (using cached Jacobians — no workspace reads)
        if state_changed:
            for k in range(NV * NV):
                H[k] = M_local[k]
            for s in range(ns):
                if sr_st[s] == SROW_QUADRATIC:
                    var d = sr_dof[s]
                    H[d * NV + d] += sr_D[s]
            # Equality rows are always QUADRATIC, so their outer product is
            # always in the Hessian — no state to test.
            for e in range(neq_rows):
                for a in range(NV):
                    var Ja = eq_J[e * NV + a]
                    if Ja == Scalar[DTYPE](0):
                        continue
                    for b in range(NV):
                        H[a * NV + b] += eq_D[e] * Ja * eq_J[e * NV + b]
            for c in range(nc):
                var cs = cs_arr[c]
                if cs == 0:
                    continue
                var mu = mu_cache[c]
                var D_n = D_n_cache[c]
                var D_f = D_f_cache[c]
                if cs == 1:
                    for i in range(NV):
                        for j in range(NV):
                            H[i * NV + j] += (
                                D_n * Jn_c[c * NV + i] * Jn_c[c * NV + j]
                                + D_f
                                * Jt1_c[c * NV + i]
                                * Jt1_c[c * NV + j]
                                + D_f
                                * Jt2_c[c * NV + i]
                                * Jt2_c[c * NV + j]
                            )
                else:
                    var jar_n = jar_n_arr[c]
                    var jar_t1 = jar_t1_arr[c]
                    var jar_t2 = jar_t2_arr[c]
                    var T_sq = jar_t1 * jar_t1 + jar_t2 * jar_t2
                    var T_s = sqrt(T_sq)
                    if T_s < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                        T_s = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
                    var s = jar_n - mu * T_s
                    var Dm = D_n / (Scalar[DTYPE](1.0) + mu * mu)
                    var h_nt1 = -Dm * mu * jar_t1 / T_s
                    var h_nt2 = -Dm * mu * jar_t2 / T_s
                    var T2_s = T_s * T_s
                    var h_t1t1 = (
                        Dm * mu * mu * jar_t1 * jar_t1 / T2_s
                        + Dm
                        * mu
                        * s
                        / T_s
                        * (jar_t1 * jar_t1 / T2_s - Scalar[DTYPE](1.0))
                    )
                    var h_t2t2 = (
                        Dm * mu * mu * jar_t2 * jar_t2 / T2_s
                        + Dm
                        * mu
                        * s
                        / T_s
                        * (jar_t2 * jar_t2 / T2_s - Scalar[DTYPE](1.0))
                    )
                    var h_t1t2 = (
                        (Dm * mu * mu + Dm * mu * s / T_s)
                        * jar_t1
                        * jar_t2
                        / T2_s
                    )
                    for i in range(NV):
                        for j in range(NV):
                            H[i * NV + j] += (
                                Dm * Jn_c[c * NV + i] * Jn_c[c * NV + j]
                                + h_nt1
                                * (
                                    Jn_c[c * NV + i] * Jt1_c[c * NV + j]
                                    + Jt1_c[c * NV + i] * Jn_c[c * NV + j]
                                )
                                + h_nt2
                                * (
                                    Jn_c[c * NV + i] * Jt2_c[c * NV + j]
                                    + Jt2_c[c * NV + i] * Jn_c[c * NV + j]
                                )
                                + h_t1t1
                                * Jt1_c[c * NV + i]
                                * Jt1_c[c * NV + j]
                                + h_t2t2
                                * Jt2_c[c * NV + i]
                                * Jt2_c[c * NV + j]
                                + h_t1t2
                                * (
                                    Jt1_c[c * NV + i] * Jt2_c[c * NV + j]
                                    + Jt2_c[c * NV + i] * Jt1_c[c * NV + j]
                                )
                            )
            var chol_ok_gpu2 = chol_factor_inline[DTYPE, NV, M_SIZE](
                H, L_chol
            )
            if not chol_ok_gpu2:
                for i in range(NV):
                    H[i * NV + i] = H[i * NV + i] + Scalar[DTYPE](1e-6)
                _ = chol_factor_inline[DTYPE, NV, M_SIZE](H, L_chol)

    # Write solved qacc back to workspace
    for i in range(NV):
        qacc_constrained[env, i] = qacc[i]

    # Write forces to state buffer for display/warmstart (directly from InlineArrays)
    for c in range(nc):
        var c_off = c * CONTACT_SIZE
        contacts[env, c_off + CONTACT_IDX_FORCE_N] = fn_arr[c]
        contacts[env, c_off + CONTACT_IDX_FORCE_T1] = ft1_arr[c]
        contacts[env, c_off + CONTACT_IDX_FORCE_T2] = ft2_arr[c]

    # Joint limits, dry-friction dofs AND fixed-tendon equalities are rows of
    # the Newton system above (`build_scalar_rows` /
    # `build_tendon_equality_rows`), not post-passes — solving them after the
    # contacts is what made the contact force wrong. `_equality_env` (connect /
    # weld) and SPATIAL tendon equalities still run as post-passes; neither has
    # a model in the tree that puts them on the same dofs as a live contact,
    # and both need a dense body Jacobian rather than a joint-coefficient one.
    comptime SOLVER_ITER_GPU: Int = 50

    comptime if NEQUALITY > 0:
        _equality_env[
            DTYPE, NV, NBODY, NJOINT, NEQUALITY, NTENDON, NSITE, V_SIZE,
            BATCH, SOLVER_ITER_GPU,
        ](
            env, qvel, xpos, xquat, subtree_com, joints, bodies, mmeta,
            equality, tendons, sites, body_invweight0, dof_invweight0, cdof,
            m_inv, qacc_constrained,
        )

    comptime if NTENDON > 0:
        # SKIP_FIXED: the fixed-tendon equalities are rows of the system above,
        # so re-applying them here would double the constraint force. Spatial
        # equality tendons are never row-built and stay this pass's job, which
        # is why the guard tests the KIND rather than skipping wholesale.
        _tendon_env[
            DTYPE, NQ, NV, NBODY, NJOINT, NTENDON, NSITE, BATCH,
            SOLVER_ITER_GPU, SKIP_FIXED=True,
        ](
            env, qpos, qvel, joints, mmeta, tendons, sites, body_invweight0,
            dof_invweight0, m_inv, qacc_constrained,
        )


def _newton_solve_fields_kernel[
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
    MAX_CONDIM: Int = 3,
    NOSLIP_ITER: Int = 0,
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
    M: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
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
    _newton_solve_env[
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
        MAX_CONDIM,
        NOSLIP_ITER,
    ](
        env, qpos, qvel, xpos, xquat, subtree_com, contacts, smeta, joints,
        bodies, mmeta, equality, tendons, sites, body_invweight0,
        dof_invweight0, cdof, M, m_inv, qacc_constrained, solver,
    )


def solve_newton[
    target: StaticString,
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int = 0,
    NEQUALITY: Int = 0,
    NTENDON: Int = 0,
    NSITE: Int = 0,
    NEXCLUDE: Int = 0,
    NMESH_VERTS: Int = 0,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    BATCH: Int = 1,
    MAX_CONDIM: Int = 3,
    NOSLIP_ITER: Int = 0,
](
    mut d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH],
    mut m: Model[
        DTYPE,
        NV,
        NBODY,
        NJOINT,
        NGEOM,
        NEQUALITY,
        NTENDON,
        NSITE,
        NEXCLUDE,
        NMESH_VERTS,
    ],
    mut scratch: DynamicsScratch[DTYPE, NV, NBODY, BATCH],
    mut cscratch: ContactScratch[DTYPE, NV, MAX_CONTACTS, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Primal Newton contact solve into `scratch.qacc_constrained` (+ solved
    forces back into `d.contacts` for warm-starting/display), both targets,
    one body. Standalone entry — same signature family as
    `solve_contacts` so callers can swap solvers later.

    ELLIPTIC: joint limits, equality constraints, and fixed tendons run
    INSIDE at the legacy position (after the Newton core, 50 iterations).
    PYRAMIDAL: limits are edge rows inside the Newton optimization;
    equality/tendon after.

    Newton uses a PREFIX (35*MC + 6*MC*NV) of the PGS-sized
    `cscratch.solver` tensor (81*MC + 12*MC*NV) — no separate scratch.
    """
    comptime MC = _max_one[MAX_CONTACTS]()
    comptime SOLVER_WS = 81 * MC + 12 * MC * NV

    comptime L_NV = Layout.row_major(BATCH, NV)
    comptime L_B3 = Layout.row_major(BATCH, NBODY * 3)
    comptime L_B4 = Layout.row_major(BATCH, NBODY * 4)
    comptime L_CON = Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE)
    comptime L_SMETA = Layout.row_major(BATCH, METADATA_SIZE)
    comptime L_JOINT = Layout.row_major(NJOINT, MODEL_JOINT_SIZE)
    comptime L_BODY = Layout.row_major(NBODY, MODEL_BODY_SIZE)
    comptime L_MMETA = Layout.row_major(MODEL_META_SIZE)
    comptime L_EQ = Layout.row_major(NEQUALITY, MODEL_EQ_SIZE)
    comptime L_TEN = Layout.row_major(NTENDON, MODEL_TENDON_SIZE)
    comptime L_SITE = Layout.row_major(NSITE, MODEL_SITE_SIZE)
    comptime L_BW = Layout.row_major(NBODY, 2)
    comptime L_CDOF = Layout.row_major(BATCH, NV * 6)
    comptime L_M = Layout.row_major(BATCH, NV * NV)
    comptime L_SOLVER = Layout.row_major(BATCH, SOLVER_WS)

    comptime L_QPOS = Layout.row_major(BATCH, NQ)
    comptime L_DW = Layout.row_major(NV)

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
        var M_v = scratch.M.lt["cpu", L_M]()
        var mi_v = scratch.m_inv.lt["cpu", L_M]()
        var qc_v = scratch.qacc_constrained.lt["cpu", L_NV]()
        var sol_v = cscratch.solver.lt["cpu", L_SOLVER]()
        for e in range(BATCH):
            _newton_solve_env[
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
                MAX_CONDIM,
                NOSLIP_ITER,
            ](
                e, qpos_v, qvel_v, xpos_v, xquat_v, stcom_v, con_v, smeta_v,
                joints_v, bodies_v, mmeta_v, eq_v, ten_v, site_v, bw_v, dw_v,
                cdof_v, M_v, mi_v, qc_v, sol_v,
            )
    else:
        # GPU. PYRAMIDAL (the production default cone) on NVIDIA uses the
        # one-env-per-block cooperative solver: the big Newton matrices live in
        # SHARED memory + the device workspace instead of a ~60KB per-thread
        # local frame, which fixes the humanoid-scale local-memory OOM. That
        # kernel's threadgroup memory exceeds Metal's 32 KB limit, so Metal —
        # and the ELLIPTIC cone on any device — keep the one-thread-per-env
        # kernel (which only OOMs on NVIDIA, where PYRAMIDAL never takes it).
        var used_blocked = False
        comptime if CONE_TYPE == ConeType.PYRAMIDAL:
            if has_nvidia_gpu_accelerator():
                solve_newton_blocked[
                    "gpu", DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
                    NEQUALITY, NTENDON, NSITE, NEXCLUDE, NMESH_VERTS, CONE_TYPE,
                    BATCH,
                ](d, m, scratch, cscratch, ctx)
                used_blocked = True
        if not used_blocked:
            var c = ctx.value()
            comptime BLOCKS = (BATCH + NS_TPB - 1) // NS_TPB
            c.enqueue_function[
                _newton_solve_fields_kernel[
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
                    MAX_CONDIM,
                    NOSLIP_ITER,
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
                scratch.M.lt["gpu", L_M](),
                scratch.m_inv.lt["gpu", L_M](),
                scratch.qacc_constrained.lt["gpu", L_NV](),
                cscratch.solver.lt["gpu", L_SOLVER](),
                grid_dim=(BLOCKS,),
                block_dim=(NS_TPB,),
            )


# =============================================================================
# PYRAMIDAL blocked Newton solve — ONE ENV PER BLOCK, cooperative across
# MAX_CONTACTS threads (fields port of NewtonSolver.solve_gpu_blocked,
# newton_solver.mojo:2748). The big Newton matrices live in SHARED memory + the
# device `solver` workspace instead of a per-thread local frame, so the
# per-thread local reservation stays tiny — this is what avoids the humanoid-
# scale OOM the one-thread-per-env kernel hits on NVIDIA. Arithmetic, iteration
# order, constants and cooperative thread distribution are VERBATIM from the
# legacy; only slab addressing → Data/Model/scratch tensors changes.
# SOLVE_COOP_NEWTON / SOLVE_COOP_RECOMPUTE are both True in the legacy production
# default, so only those cooperative code paths are ported (the tid-0 serial
# "oracle" branches are dead in production and dropped).
# =============================================================================


def _newton_blocked_fields_kernel[
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
    MAX_CONDIM: Int = 3,
    NOSLIP_ITER: Int = 0,
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
    M: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
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
    var env = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var contact_tid = tid
    var valid_env = env < BATCH

    comptime MC = _max_one[MAX_CONTACTS]()
    comptime V_SIZE = _max_one[NV]()
    comptime M_SIZE = _max_one[NV * NV]()
    comptime THREADS = _max_one[MAX_CONTACTS]()

    # Common normal block offsets (row-relative; the legacy `solver_ws_idx`
    # base is 0 in the fields solver tensor)
    comptime ws_J_n_idx = 15 * MC

    # Primal-specific offsets (after common normal block)
    comptime PRIMAL_START = 15 * MC + 2 * MC * NV
    comptime ws_Jt1_idx = PRIMAL_START + 0 * MC * NV
    comptime ws_Jt2_idx = PRIMAL_START + 1 * MC * NV
    comptime ws_MinvJt1_idx = PRIMAL_START + 2 * MC * NV
    comptime ws_MinvJt2_idx = PRIMAL_START + 3 * MC * NV
    comptime SC = PRIMAL_START + 4 * MC * NV
    comptime ws_mu_idx = SC + 0 * MC
    comptime ws_D_n_idx = SC + 1 * MC
    comptime ws_D_f_idx = SC + 2 * MC
    comptime ws_bt1_idx = SC + 3 * MC
    comptime ws_bt2_idx = SC + 4 * MC
    comptime CVS = SC + 5 * MC
    comptime ws_jar_n_idx = CVS + 0 * MC
    comptime ws_jar_t1_idx = CVS + 1 * MC
    comptime ws_jar_t2_idx = CVS + 2 * MC
    comptime ws_fn_idx = CVS + 3 * MC
    comptime ws_ft1_idx = CVS + 4 * MC
    comptime ws_ft2_idx = CVS + 5 * MC
    comptime ws_cstate_idx = CVS + 6 * MC
    comptime pyr_sc = ws_Jt1_idx + 2 * (MAX_CONDIM - 1) * MC * NV

    # === PARALLEL: Initialize common normal workspace (one thread/contact) ===
    if valid_env:
        _init_common_normal_ws[
            DTYPE, NV, MAX_CONTACTS, BATCH, SOLVER_WS
        ](env, contact_tid, solver)
        if contact_tid < MC:
            for d in range(NV):
                solver[env, ws_Jt1_idx + contact_tid * NV + d] = 0
                solver[env, ws_Jt2_idx + contact_tid * NV + d] = 0
                solver[env, ws_MinvJt1_idx + contact_tid * NV + d] = 0
                solver[env, ws_MinvJt2_idx + contact_tid * NV + d] = 0
            solver[env, ws_mu_idx + contact_tid] = 0
            solver[env, ws_D_n_idx + contact_tid] = 0
            solver[env, ws_D_f_idx + contact_tid] = 0
            solver[env, ws_bt1_idx + contact_tid] = 0
            solver[env, ws_bt2_idx + contact_tid] = 0
            solver[env, ws_jar_n_idx + contact_tid] = 0
            solver[env, ws_jar_t1_idx + contact_tid] = 0
            solver[env, ws_jar_t2_idx + contact_tid] = 0
            solver[env, ws_fn_idx + contact_tid] = 0
            solver[env, ws_ft1_idx + contact_tid] = 0
            solver[env, ws_ft2_idx + contact_tid] = 0
            solver[env, ws_cstate_idx + contact_tid] = 0

    # === Read metadata (all threads; legacy `dt` read dropped — unused) ===
    var nc = 0
    var K_spring: Scalar[DTYPE] = 0
    var B_damp: Scalar[DTYPE] = 0
    var si_dmin: Scalar[DTYPE] = 0
    var si_dmax: Scalar[DTYPE] = 0
    var si_width: Scalar[DTYPE] = 1
    var si_midpoint: Scalar[DTYPE] = Scalar[DTYPE](0.5)
    var si_power: Scalar[DTYPE] = Scalar[DTYPE](2.0)
    var impratio: Scalar[DTYPE] = Scalar[DTYPE](1.0)

    if valid_env:
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
        # solref -> (K, B), including MuJoCo's DIRECT form for a NEGATIVE
        # solref. See `constraints/constraint_data.solref_spring_damper` — the
        # formula lived in twelve copy-pasted sites until 2026-08-03.
        (K_spring, B_damp) = solref_spring_damper[DTYPE](
            sr_tc, sr_dr, si_dmax,
            rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP]),
        )
        impratio = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_IMPRATIO])
        if impratio < Scalar[DTYPE](1e-6):
            impratio = Scalar[DTYPE](1.0)

    # === PARALLEL PHASE 1: each thread precomputes one contact's normal data ==
    if valid_env:
        _precompute_contact_normal[
            DTYPE, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE, BATCH, SOLVER_WS
        ](
            env, contact_tid, nc, qvel, subtree_com, contacts, joints, bodies,
            mmeta, body_invweight0, cdof, m_inv, qacc_constrained, solver,
            K_spring, B_damp, si_dmin, si_dmax, si_width, si_midpoint,
            si_power,
        )

    barrier()

    # === PARALLEL PHASE 2: tangent frame + friction data ===
    if valid_env and contact_tid < nc:
        _precompute_contact_friction[
            DTYPE, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE, BATCH, SOLVER_WS,
            CONE_TYPE, MAX_CONDIM,
        ](
            env, contact_tid, nc, qvel, subtree_com, contacts, joints, bodies,
            mmeta, cdof, solver, B_damp, impratio, K_spring, ws_Jt1_idx,
            ws_Jt2_idx, ws_mu_idx, ws_D_n_idx, ws_D_f_idx, ws_bt1_idx,
            ws_bt2_idx,
        )

    barrier()

    comptime NEWTON_ITER_GPU: Int = 200
    comptime NEWTON_TOL_GPU: Float64 = 1e-8
    comptime LINESEARCH_ITER: Int = 20
    comptime PRIMAL_MINVAL_GPU: Float64 = 1e-12

    # PYRAMIDAL-only blocked solver. (Non-PYRAMIDAL never routes here.)
    # 2*(dim-1) edges per contact; see the per-env path for the layout note.
    comptime NE = 2 * (MAX_CONDIM - 1)
    comptime MAX_LIM = _max_one[2 * NJOINT]()
    comptime MAX_FRIC = V_SIZE  # one dry-friction row per dof
    comptime MAX_TLIM = 2 * NTENDON  # lo + hi per tendon
    # contact edges + joint limits + dry friction + tendon limits.
    #
    # ⚠ The last two were MISSING here until 2026-07-31, so on NVIDIA +
    # PYRAMIDAL — the only configuration that takes this kernel — a model with
    # `frictionloss` had NO dry friction and a model with a limited tendon had
    # NO string, both silently. `frictionloss` rows landed in the per-env
    # pyramidal path with 04a7c508 and were simply never mirrored here.
    #
    # ⚠ ME drives `Je_sh`, which is `ME * V_SIZE` DOUBLES of THREADGROUP
    # memory and is the dominant shared-memory term. Growing ME by
    # `V_SIZE + 3*NTENDON` grows Je_sh by `(V_SIZE + 3*NTENDON) * V_SIZE`. On a
    # large model that can push the block over the device's shared-memory
    # limit, which shows up as a LAUNCH FAILURE (loud), not a wrong answer.
    comptime MAX_TEQ = NTENDON  # one bilateral row per equality tendon
    comptime ME = NE * MC + MAX_LIM + MAX_FRIC + MAX_TLIM + MAX_TEQ

    # ── Je: shared when it fits, spilled to global when it does not ───────
    #
    # ⚠⚠ MEASURED FAILURE THIS GUARD EXISTS FOR (humanoid_CMU, 2026-08-10):
    #     ptxas error : Entry function ... uses too much shared data
    #                   (0x2975c bytes, 0x18c00 max)
    # i.e. 169,820 B requested against a 101,376 B limit. `Je_sh` is
    # `ME * V_SIZE` scalars and dominates everything else combined:
    #
    #     humanoid      NV=27  MC=32  ME~150   Je ~16 KB   -> fits
    #     humanoid_CMU  NV=62  MC=64  ME=432   Je ~107 KB  -> over the limit
    #                                                        BY ITSELF
    #
    # ⚠ HALVING max_contacts DOES NOT FIX IT — 64->32 gives ME=304, Je 75 KB,
    # total ~116 KB, still over. It would cost real contact fidelity and still
    # not compile, so do not reach for that lever.
    #
    # Spilling ONLY Je leaves ~61 KB in threadgroup memory, which fits with
    # room to spare. The spill is GATED because Je is read across up to
    # NEWTON_ITER_GPU (200) iterations: putting it in global memory costs
    # bandwidth on EVERY model taking this kernel, and only the oversized ones
    # need to pay. Models that fit keep the fast path unchanged, bit for bit.
    #
    # ⚠ THE THRESHOLD IS A COMPILE-TIME GUESS AT A RUNTIME LIMIT. Shared
    # memory per block is device-specific (99 KB on this box; 227 KB on an
    # H100), and the kernel is compiled without knowing the target. 64 KB is
    # deliberately conservative — the widely-supported opt-in floor — so a
    # model that fits everywhere keeps shared, and anything near the edge
    # spills rather than failing to compile on the smallest plausible device.
    comptime _JE_ELEMS = ME * V_SIZE
    comptime _JE_BYTES = _JE_ELEMS * size_of[Scalar[DTYPE]]()
    comptime JE_SHARED_BUDGET = 64 * 1024
    comptime JE_FITS_SHARED = _JE_BYTES <= JE_SHARED_BUDGET

    # Where a spilled Je lives: the TAIL of the per-env solver workspace.
    # `cscratch.solver` is sized for PGS (81*MC + 12*MC*NV) and Newton only
    # ever uses a prefix, so the tail is already allocated and unused:
    #
    #     Newton high-water  27*MC + 6*MC*NV   (ws_cstate_idx + MC)
    #     SOLVER_WS          81*MC + 12*MC*NV
    #     free tail          54*MC + 6*MC*NV
    #
    # humanoid_CMU: 27,264 free vs 26,784 needed — it fits, with 1.8% to
    # spare. ⚠⚠ THAT MARGIN IS TOO THIN TO TRUST BY INSPECTION, which is why
    # `JE_FITS_WS` is a real comptime test and not a comment: ME grows with
    # tendons and joint limits, and overrunning this region would silently
    # corrupt a neighbouring solver's workspace rather than fail. A model that
    # outgrows the tail falls back to shared (see the three-way gate below)
    # rather than spilling into memory it does not own.
    comptime JE_WS_START = 27 * MC + 6 * MC * NV
    comptime JE_FITS_WS = JE_WS_START + _JE_ELEMS <= SOLVER_WS

    # ⚠⚠ SPILL ONLY WHEN THE SPILL ACTUALLY FITS. A model can fit NEITHER,
    # and the third case is not hypothetical — it is dm_control's dog:
    #
    #     model         Je shared   workspace tail   verdict
    #     humanoid       20 KB          —            stays shared (unchanged)
    #     quadruped      13 KB          —            stays shared (unchanged)
    #     humanoid_CMU  104 KB      26784 <= 27264   SPILLS, fits
    #     dog           151 KB      38789 >  12672   fits NEITHER
    #     dog_fetch     178 KB      45815 >  15792   fits NEITHER
    #
    # ⚠ A `comptime assert` here would be WRONG: it fires during compilation
    # on EVERY target, so rejecting the fits-neither case would turn dog's
    # currently-passing Apple build into a build FAILURE. Falling back to
    # shared leaves those models exactly as they are today — no better, no
    # worse — and confines this change to the models it can actually help.
    #
    # ⚠⚠ FORWARD-LOOKING CONSTRAINT ON DOG — NOT A CURRENT BREAK. dog is
    # PYRAMIDAL, so it WOULD instantiate this kernel, and its Je wants 151 KB
    # of threadgroup memory against NVIDIA's ~99 KB — the same ptxas error
    # humanoid_CMU hit. It does not bite today only because dog has NO
    # GPU-batched env (no `Phyics3dBatchedEnv` alias, and no GPU-vs-CPU test),
    # so the GPU path is never instantiated for it.
    #
    # The moment dog gets a batched env, this kernel stops compiling on
    # NVIDIA and the workspace tail cannot absorb the spill either. That needs
    # SOLVER_WS widened in `fields/contact_scratch.mojo` — roughly 26k more
    # scalars per env for dog, 33k for dog_fetch — which is deliberately NOT
    # done here: it costs memory for every model and should be paid when dog
    # actually needs it, with a parity gate to prove it.
    comptime JE_IN_SHARED = JE_FITS_SHARED or not JE_FITS_WS
    comptime JE_AS = (
        AddressSpace.SHARED if JE_IN_SHARED else AddressSpace.GENERIC
    )

    # Sized to 1 when spilling so the threadgroup allocation disappears.
    comptime JE_SH_ELEMS = _JE_ELEMS if JE_IN_SHARED else 1

    # === SHARED memory (per-block == per-env) ===
    var M_sh = LayoutTensor[
        DTYPE, Layout.row_major(M_SIZE), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var H_sh = LayoutTensor[
        DTYPE, Layout.row_major(M_SIZE), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    # ⚠ BOTH BRANCHES ARE TYPE-CHECKED even though only one is emitted
    # (measured: an ill-typed untaken `comptime if` branch fails the build).
    # `address_space_cast[JE_AS]()` on each side is what makes them agree —
    # without it the SHARED build rejects the GENERIC branch and vice versa.
    var _je_backing = LayoutTensor[
        DTYPE, Layout.row_major(JE_SH_ELEMS), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var _je_ptr: Pointer[
        Scalar[DTYPE], MutAnyOrigin, address_space=JE_AS
    ]
    comptime if JE_IN_SHARED:
        _je_ptr = _je_backing.ptr.unsafe_address_space_cast[JE_AS]()
    else:
        _je_ptr = (
            solver.ptr.unsafe_offset(env * SOLVER_WS + JE_WS_START)
        ).unsafe_address_space_cast[JE_AS]()
    var Je_sh = LayoutTensor[
        DTYPE, Layout.row_major(ME * V_SIZE), MutAnyOrigin,
        address_space=JE_AS,
    ](_je_ptr)
    var De_sh = LayoutTensor[
        DTYPE, Layout.row_major(ME), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var bias_e_sh = LayoutTensor[
        DTYPE, Layout.row_major(ME), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var force_sh = LayoutTensor[
        DTYPE, Layout.row_major(ME), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    # Row kind + box data (written once by thread 0) and the per-iteration row
    # STATE (written by thread 0 with the forces, read by every thread for the
    # Hessian). The state cannot be re-derived from `force_sh` alone: a
    # saturated box row has force > 0 yet contributes NO curvature, which is
    # exactly the misclassification `primal.mojo` carried until 04a7c508.
    var kind_e_sh = LayoutTensor[
        DTYPE, Layout.row_major(ME), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var R_e_sh = LayoutTensor[
        DTYPE, Layout.row_major(ME), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var floss_e_sh = LayoutTensor[
        DTYPE, Layout.row_major(ME), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var state_e_sh = LayoutTensor[
        DTYPE, Layout.row_major(ME), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var L_sh = LayoutTensor[
        DTYPE, Layout.row_major(M_SIZE), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var search_sh = LayoutTensor[
        DTYPE, Layout.row_major(V_SIZE), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var Mv_sh = LayoutTensor[
        DTYPE, Layout.row_major(V_SIZE), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var Jv_e_sh = LayoutTensor[
        DTYPE, Layout.row_major(ME), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var qacc_sh = LayoutTensor[
        DTYPE, Layout.row_major(V_SIZE), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var jar_sh = LayoutTensor[
        DTYPE, Layout.row_major(ME), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var qfrc_sh = LayoutTensor[
        DTYPE, Layout.row_major(V_SIZE), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    # Scalar shared state: [0]=num_edges, [1]=done flag, [2]=Cholesky
    # rank-deficient flag.
    var ctrl_sh = LayoutTensor[
        DTYPE, Layout.row_major(3), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # === COOPERATIVE LOAD: M into shared ===
    if valid_env:
        for k in range(tid, NV * NV, THREADS):
            M_sh[k] = rebind[Scalar[DTYPE]](M[env, k])

        # Cooperative load of contact edges (Je/De/bias_e) into shared. One
        # thread per contact (contact_tid == c), matching serial load order
        # (c ascending, e ascending).
        if contact_tid < nc:
            var c = contact_tid
            for e in range(NE):
                var idx = c * NE + e
                for i in range(NV):
                    Je_sh[idx * NV + i] = rebind[Scalar[DTYPE]](
                        solver[env, ws_Jt1_idx + e * MC * NV + c * NV + i]
                    )
                De_sh[idx] = rebind[Scalar[DTYPE]](
                    solver[env, pyr_sc + e * MC + c]
                )
                bias_e_sh[idx] = rebind[Scalar[DTYPE]](
                    solver[env, pyr_sc + NE * MC + e * MC + c]
                )

    barrier()

    # === THREAD 0: joint-limit edge detection + initial setup ===
    var qacc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var qacc_smooth = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var Ma = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var f_smooth = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var jar = InlineArray[Scalar[DTYPE], ME](uninitialized=True)
    var grad = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var search = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var Mv = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var Jv_e = InlineArray[Scalar[DTYPE], ME](uninitialized=True)
    var qfrc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var old_qacc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var old_Ma = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var old_jar = InlineArray[Scalar[DTYPE], ME](uninitialized=True)
    var old_force = InlineArray[Scalar[DTYPE], ME](uninitialized=True)
    var old_qfrc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var old_cost: Scalar[DTYPE] = 0
    var scale: Scalar[DTYPE] = 0
    var num_edges = 0

    if valid_env and tid == 0:
        # Contact edges and joint limits are ONE-SIDED, so they leave
        # kind = SROW_LIMIT and R/floss = 0; only the dry-friction rows below
        # override. Must be cleared first — shared memory is uninitialised.
        for e in range(ME):
            kind_e_sh[e] = Scalar[DTYPE](SROW_LIMIT)
            R_e_sh[e] = Scalar[DTYPE](0)
            floss_e_sh[e] = Scalar[DTYPE](0)
            state_e_sh[e] = Scalar[DTYPE](0)
        num_edges = nc * NE

        # Model-level defaults for fallback
        var lr_tc_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLREF_LIMIT_0]
        )
        var lr_dr_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLREF_LIMIT_1]
        )
        var li_dmin_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLIMP_LIMIT_0]
        )
        var li_dmax_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLIMP_LIMIT_1]
        )
        var li_width_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLIMP_LIMIT_2]
        )
        var li_midpoint_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLIMP_LIMIT_3]
        )
        var li_power_def = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_SOLIMP_LIMIT_4]
        )

        for j in range(NJOINT):
            var jtype = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
            if jtype != JNT_HINGE and jtype != JNT_SLIDE:
                continue
            var dof = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR])
            )
            var qpos_adr = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS_ADR])
            )
            var rmin = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_RANGE_MIN])
            var rmax = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_RANGE_MAX])
            if rmin < Scalar[DTYPE](-1e9) or rmax > Scalar[DTYPE](1e9):
                continue
            var lr_tc = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLREF_LIMIT_0]
            )
            var lr_dr = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLREF_LIMIT_1]
            )
            if lr_tc <= Scalar[DTYPE](0):
                lr_tc = lr_tc_def
            if lr_dr <= Scalar[DTYPE](0):
                lr_dr = lr_dr_def
            var li_dmin = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLIMP_LIMIT_0]
            )
            var li_dmax = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLIMP_LIMIT_1]
            )
            var li_width = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLIMP_LIMIT_2]
            )
            var li_midpoint = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLIMP_LIMIT_3]
            )
            var li_power = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_SOLIMP_LIMIT_4]
            )
            if li_dmax <= Scalar[DTYPE](0) and li_width <= Scalar[DTYPE](0):
                li_dmin = li_dmin_def
                li_dmax = li_dmax_def
                li_width = li_width_def
                li_midpoint = li_midpoint_def
                li_power = li_power_def
            if li_width < Scalar[DTYPE](1e-6):
                li_width = Scalar[DTYPE](1e-6)
            # Clamp BOTH ends to [mjMINIMP, mjMAXIMP] as MuJoCo does before
            # interpolating (engine_core_constraint.c:1284-1287); see the same fix
            # on the contact path above.
            comptime MJL_MINIMP = Scalar[DTYPE](0.0001)
            comptime MJL_MAXIMP = Scalar[DTYPE](0.9999)
            if li_dmin < MJL_MINIMP:
                li_dmin = MJL_MINIMP
            elif li_dmin > MJL_MAXIMP:
                li_dmin = MJL_MAXIMP
            if li_dmax < MJL_MINIMP:
                li_dmax = MJL_MINIMP
            elif li_dmax > MJL_MAXIMP:
                li_dmax = MJL_MAXIMP
            if li_power < Scalar[DTYPE](1):
                li_power = Scalar[DTYPE](1)
            # solref -> (K, B), including MuJoCo's DIRECT form for a NEGATIVE
            # solref. See `constraints/constraint_data.solref_spring_damper` — the
            # formula lived in twelve copy-pasted sites until 2026-08-03.
            var (l_K_spring, l_B_damp) = solref_spring_damper[DTYPE](
                lr_tc, lr_dr, li_dmax,
                rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP]),
            )

            var pos = rebind[Scalar[DTYPE]](qpos[env, qpos_adr])
            # Lower limit
            var dist_lo = pos - rmin
            if dist_lo < Scalar[DTYPE](0) and num_edges < ME:
                var sign = Scalar[DTYPE](1)
                var K_lim = rebind[Scalar[DTYPE]](m_inv[env, dof * NV + dof])
                if K_lim < Scalar[DTYPE](1e-10):
                    K_lim = Scalar[DTYPE](1e-10)
                var pen = -dist_lo
                var v_lim = sign * rebind[Scalar[DTYPE]](qvel[env, dof])
                var imp_lim: Scalar[DTYPE]
                if li_dmin == li_dmax or li_width <= Scalar[DTYPE](0):
                    imp_lim = Scalar[DTYPE](0.5) * (li_dmin + li_dmax)
                else:
                    var x_l = pen / li_width
                    if x_l <= Scalar[DTYPE](0):
                        imp_lim = li_dmin
                    elif x_l >= Scalar[DTYPE](1):
                        imp_lim = li_dmax
                    else:
                        var y_l: Scalar[DTYPE]
                        if li_power == Scalar[DTYPE](1):
                            y_l = x_l
                        elif x_l <= li_midpoint:
                            y_l = pow(x_l, li_power) / pow(
                                li_midpoint, li_power - Scalar[DTYPE](1)
                            )
                        else:
                            y_l = Scalar[DTYPE](1) - pow(
                                Scalar[DTYPE](1) - x_l, li_power
                            ) / pow(
                                Scalar[DTYPE](1) - li_midpoint,
                                li_power - Scalar[DTYPE](1),
                            )
                        imp_lim = li_dmin + y_l * (li_dmax - li_dmin)
                if imp_lim < Scalar[DTYPE](1e-6):
                    imp_lim = Scalar[DTYPE](1e-6)
                var diag_lim = rebind[Scalar[DTYPE]](dof_invweight0[dof])
                if diag_lim < Scalar[DTYPE](1e-10):
                    diag_lim = K_lim
                var R_lim = (
                    (Scalar[DTYPE](1) - imp_lim) / imp_lim * diag_lim
                )
                if R_lim < Scalar[DTYPE](1e-14):
                    R_lim = Scalar[DTYPE](1e-14)
                for i in range(NV):
                    Je_sh[num_edges * NV + i] = Scalar[DTYPE](0)
                Je_sh[num_edges * NV + dof] = sign
                var inv_K_lim = Scalar[DTYPE](1) / (K_lim + R_lim)
                var R_recov = Scalar[DTYPE](1) / inv_K_lim - K_lim
                if R_recov < Scalar[DTYPE](1e-14):
                    R_recov = Scalar[DTYPE](1e-14)
                De_sh[num_edges] = Scalar[DTYPE](1) / R_recov
                bias_e_sh[num_edges] = (
                    l_B_damp * v_lim - l_K_spring * imp_lim * pen
                )
                num_edges += 1

            # Upper limit
            var dist_hi = rmax - pos
            if dist_hi < Scalar[DTYPE](0) and num_edges < ME:
                var sign = Scalar[DTYPE](-1)
                var K_lim = rebind[Scalar[DTYPE]](m_inv[env, dof * NV + dof])
                if K_lim < Scalar[DTYPE](1e-10):
                    K_lim = Scalar[DTYPE](1e-10)
                var pen = -dist_hi
                var v_lim = sign * rebind[Scalar[DTYPE]](qvel[env, dof])
                var imp_lim: Scalar[DTYPE]
                if li_dmin == li_dmax or li_width <= Scalar[DTYPE](0):
                    imp_lim = Scalar[DTYPE](0.5) * (li_dmin + li_dmax)
                else:
                    var x_l = pen / li_width
                    if x_l <= Scalar[DTYPE](0):
                        imp_lim = li_dmin
                    elif x_l >= Scalar[DTYPE](1):
                        imp_lim = li_dmax
                    else:
                        var y_l: Scalar[DTYPE]
                        if li_power == Scalar[DTYPE](1):
                            y_l = x_l
                        elif x_l <= li_midpoint:
                            y_l = pow(x_l, li_power) / pow(
                                li_midpoint, li_power - Scalar[DTYPE](1)
                            )
                        else:
                            y_l = Scalar[DTYPE](1) - pow(
                                Scalar[DTYPE](1) - x_l, li_power
                            ) / pow(
                                Scalar[DTYPE](1) - li_midpoint,
                                li_power - Scalar[DTYPE](1),
                            )
                        imp_lim = li_dmin + y_l * (li_dmax - li_dmin)
                if imp_lim < Scalar[DTYPE](1e-6):
                    imp_lim = Scalar[DTYPE](1e-6)
                var diag_lim = rebind[Scalar[DTYPE]](dof_invweight0[dof])
                if diag_lim < Scalar[DTYPE](1e-10):
                    diag_lim = K_lim
                var R_lim = (
                    (Scalar[DTYPE](1) - imp_lim) / imp_lim * diag_lim
                )
                if R_lim < Scalar[DTYPE](1e-14):
                    R_lim = Scalar[DTYPE](1e-14)
                for i in range(NV):
                    Je_sh[num_edges * NV + i] = Scalar[DTYPE](0)
                Je_sh[num_edges * NV + dof] = sign
                var inv_K_lim = Scalar[DTYPE](1) / (K_lim + R_lim)
                var R_recov = Scalar[DTYPE](1) / inv_K_lim - K_lim
                if R_recov < Scalar[DTYPE](1e-14):
                    R_recov = Scalar[DTYPE](1e-14)
                De_sh[num_edges] = Scalar[DTYPE](1) / R_recov
                bias_e_sh[num_edges] = (
                    l_B_damp * v_lim - l_K_spring * imp_lim * pen
                )
                num_edges += 1

        # Tendon limit rows (mjCNSTR_LIMIT_TENDON). Dense J — the same builder
        # the per-env pyramidal path uses, so the two cones cannot drift.
        comptime if NTENDON > 0:
            # Staging buffers sized MAX_TLIM, NOT ME: these are per-thread
            # LOCAL memory, and `ME * V_SIZE` doubles would be tens of KB —
            # precisely the local-memory OOM this cooperative kernel exists to
            # avoid. The builder fills from index 0, so tendon capacity is all
            # it can ever need.
            var t_je = InlineArray[Scalar[DTYPE], MAX_TLIM * V_SIZE](
                fill=Scalar[DTYPE](0)
            )
            var t_de = InlineArray[Scalar[DTYPE], MAX_TLIM](
                fill=Scalar[DTYPE](0)
            )
            var t_bias = InlineArray[Scalar[DTYPE], MAX_TLIM](
                fill=Scalar[DTYPE](0)
            )
            var t_n = 0
            build_tendon_limit_rows[
                DTYPE, NV, NBODY, NJOINT, NSITE, NTENDON, V_SIZE, MAX_TLIM,
                BATCH,
            ](
                env, qvel, tendons, sites, bodies, joints, mmeta,
                subtree_com, cdof, xpos, xquat, m_inv,
                t_je, t_de, t_bias, t_n,
            )
            for r in range(t_n):
                if num_edges >= ME:
                    break
                for i in range(NV):
                    Je_sh[num_edges * NV + i] = t_je[r * NV + i]
                De_sh[num_edges] = t_de[r]
                bias_e_sh[num_edges] = t_bias[r]
                num_edges += 1

            # Fixed-tendon equality rows — same staging, and the same reason
            # they are rows: see the CPU pyramidal path.
            var q_je = InlineArray[Scalar[DTYPE], MAX_TEQ * V_SIZE](
                fill=Scalar[DTYPE](0)
            )
            var q_de = InlineArray[Scalar[DTYPE], MAX_TEQ](
                fill=Scalar[DTYPE](0)
            )
            var q_bias = InlineArray[Scalar[DTYPE], MAX_TEQ](
                fill=Scalar[DTYPE](0)
            )
            var q_kind = InlineArray[Int, MAX_TEQ](fill=SROW_EQ_BILATERAL)
            var q_n = 0
            build_tendon_equality_rows[
                DTYPE, NQ, NV, NJOINT, NTENDON, V_SIZE, MAX_TEQ, BATCH
            ](
                env, qpos, qvel, tendons, joints, mmeta, m_inv,
                q_je, q_de, q_bias, q_kind, q_n,
            )
            for r in range(q_n):
                if num_edges >= ME:
                    break
                for i in range(NV):
                    Je_sh[num_edges * NV + i] = q_je[r * NV + i]
                De_sh[num_edges] = q_de[r]
                bias_e_sh[num_edges] = q_bias[r]
                kind_e_sh[num_edges] = Scalar[DTYPE](q_kind[r])
                num_edges += 1

        # Dry-friction dof rows (mjCNSTR_FRICTION_DOF). BOX rows, clamped to
        # +-frictionloss, so they are the reason this kernel needs row states
        # at all. Arithmetic identical to the per-env pyramidal builder.
        var f_imp = Scalar[DTYPE](DOF_SOLIMP_DMIN)
        var f_dmax = Scalar[DTYPE](DOF_SOLIMP_DMAX)
        # REFSAFE applies to the hardcoded friction default too — see
        # `refsafe_timeconst`.
        var f_tc_p = refsafe_timeconst[DTYPE](
            Scalar[DTYPE](DOF_SOLREF_TIMECONST),
            rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP]),
        )
        var f_B = Scalar[DTYPE](2.0) / (f_dmax * f_tc_p)
        for j in range(NJOINT):
            var floss = rebind[Scalar[DTYPE]](
                joints[j, JOINT_IDX_FRICTIONLOSS]
            )
            if floss <= Scalar[DTYPE](0):
                continue
            var jt = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
            var dof_adr = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR])
            )
            var nd = 1
            if jt == JNT_FREE:
                nd = 6
            elif jt == JNT_BALL:
                nd = 3
            for k in range(nd):
                if num_edges >= ME:
                    break
                var dof = dof_adr + k
                var K_d = rebind[Scalar[DTYPE]](m_inv[env, dof * NV + dof])
                if K_d < Scalar[DTYPE](1e-10):
                    K_d = Scalar[DTYPE](1e-10)
                var diag_f = rebind[Scalar[DTYPE]](dof_invweight0[dof])
                if diag_f < Scalar[DTYPE](1e-10):
                    diag_f = K_d
                var R_f = (Scalar[DTYPE](1) - f_imp) / f_imp * diag_f
                if R_f < Scalar[DTYPE](1e-14):
                    R_f = Scalar[DTYPE](1e-14)
                for i in range(NV):
                    Je_sh[num_edges * NV + i] = Scalar[DTYPE](0)
                Je_sh[num_edges * NV + dof] = Scalar[DTYPE](1)
                De_sh[num_edges] = Scalar[DTYPE](1) / R_f
                R_e_sh[num_edges] = R_f
                floss_e_sh[num_edges] = floss
                kind_e_sh[num_edges] = Scalar[DTYPE](SROW_FRICTION)
                bias_e_sh[num_edges] = f_B * rebind[Scalar[DTYPE]](
                    qvel[env, dof]
                )
                num_edges += 1

        # Publish num_edges to shared for all threads.
        ctrl_sh[0] = Scalar[DTYPE](num_edges)

        # Initialize qacc/qacc_smooth from workspace
        for i in range(NV):
            var q_i = rebind[Scalar[DTYPE]](qacc_constrained[env, i])
            qacc[i] = q_i
            qacc_smooth[i] = q_i
        # Ma = M * qacc (read from M_sh)
        for i in range(NV):
            Ma[i] = Scalar[DTYPE](0)
            for j in range(NV):
                Ma[i] += rebind[Scalar[DTYPE]](M_sh[i * NV + j]) * qacc[j]
        for i in range(NV):
            f_smooth[i] = Ma[i]
        # Same model-constant scale as the per-env path above; see the note
        # there for why a pose-dependent trace(M) is wrong and why this is NOT
        # a fix for the open dog residual.
        var scale_db = rebind[Scalar[DTYPE]](
            mmeta[MODEL_META_IDX_MEANINERTIA]
        ) * Scalar[DTYPE](NV if NV > 1 else 1)
        scale = (
            Scalar[DTYPE](1) / scale_db
            if scale_db > Scalar[DTYPE](1e-10)
            else Scalar[DTYPE](1)
        )

        # Initial jar + force + qfrc; publish force to force_sh
        for i in range(NV):
            qfrc[i] = Scalar[DTYPE](0)
        for e_idx in range(num_edges):
            jar[e_idx] = rebind[Scalar[DTYPE]](bias_e_sh[e_idx])
            for i in range(NV):
                jar[e_idx] += (
                    rebind[Scalar[DTYPE]](Je_sh[e_idx * NV + i]) * qacc[i]
                )
            var st_e = scalar_row_state[DTYPE](
                Int(rebind[Scalar[DTYPE]](kind_e_sh[e_idx])),
                jar[e_idx],
                rebind[Scalar[DTYPE]](R_e_sh[e_idx]),
                rebind[Scalar[DTYPE]](floss_e_sh[e_idx]),
            )
            state_e_sh[e_idx] = Scalar[DTYPE](st_e)
            var f_e = scalar_row_force[DTYPE](
                st_e,
                jar[e_idx],
                rebind[Scalar[DTYPE]](De_sh[e_idx]),
                rebind[Scalar[DTYPE]](floss_e_sh[e_idx]),
            )
            force_sh[e_idx] = f_e
            for i in range(NV):
                qfrc[i] += rebind[Scalar[DTYPE]](Je_sh[e_idx * NV + i]) * f_e

    # Make num_edges + force_sh visible to all threads.
    barrier()
    var num_edges_b = Int(rebind[Scalar[DTYPE]](ctrl_sh[0]))

    # === Newton iterations — ALL threads execute the loop ===
    for iter_n in range(NEWTON_ITER_GPU):
        # --- Thread 0: gradient + convergence check ---
        if valid_env and tid == 0:
            var grad_norm: Scalar[DTYPE] = 0
            for i in range(NV):
                grad[i] = Ma[i] - f_smooth[i] - qfrc[i]
                grad_norm += grad[i] * grad[i]
            if scale * sqrt(grad_norm) < Scalar[DTYPE](NEWTON_TOL_GPU):
                ctrl_sh[1] = Scalar[DTYPE](1)  # done
            else:
                ctrl_sh[1] = Scalar[DTYPE](0)
        barrier()
        if Int(rebind[Scalar[DTYPE]](ctrl_sh[1])) == 1:
            break

        # --- ALL threads: parallel Hessian assembly (inner edge-sum ascending
        # → bit-identical to the serial build) ---
        if valid_env:
            for idx in range(tid, NV * NV, THREADS):
                var i = idx // NV
                var j = idx % NV
                var h = rebind[Scalar[DTYPE]](M_sh[idx])
                for e in range(num_edges_b):
                    if (
                        Int(rebind[Scalar[DTYPE]](state_e_sh[e]))
                        == SROW_QUADRATIC
                    ):
                        h += (
                            rebind[Scalar[DTYPE]](De_sh[e])
                            * rebind[Scalar[DTYPE]](Je_sh[e * NV + i])
                            * rebind[Scalar[DTYPE]](Je_sh[e * NV + j])
                        )
                H_sh[idx] = h
        barrier()

        # --- Cooperative Cholesky factor of H into L_sh ---
        _chol_factor_coop[DTYPE, NV, M_SIZE](
            tid, THREADS, H_sh, L_sh, ctrl_sh
        )

        # --- Thread 0: Cholesky solve + negate search + publish ---
        if valid_env and tid == 0:
            var L_chol = InlineArray[Scalar[DTYPE], M_SIZE](
                uninitialized=True
            )
            for k in range(NV * NV):
                L_chol[k] = rebind[Scalar[DTYPE]](L_sh[k])
            chol_solve_inline[DTYPE, NV, M_SIZE, V_SIZE](L_chol, grad, search)
            for i in range(NV):
                search[i] = -search[i]
            # Publish search; Mv/Jv_e computed cooperatively below.
            for i in range(NV):
                search_sh[i] = search[i]

        # --- Cooperative Mv = M·search and Jv_e = Je·search ---
        barrier()
        _matvec_mv_jve_coop[DTYPE, NV, V_SIZE, M_SIZE, ME, JE_AS](
            tid, THREADS, num_edges_b, M_sh, Je_sh, search_sh, Mv_sh, Jv_e_sh
        )
        barrier()
        if valid_env and tid == 0:
            for i in range(NV):
                Mv[i] = rebind[Scalar[DTYPE]](Mv_sh[i])
            for e_idx in range(num_edges_b):
                Jv_e[e_idx] = rebind[Scalar[DTYPE]](Jv_e_sh[e_idx])

        # --- Thread 0: gauss / p0 / line search / update / cost ---
        if valid_env and tid == 0:
            var gauss_a: Scalar[DTYPE] = 0
            var gauss_b: Scalar[DTYPE] = 0
            for i in range(NV):
                gauss_a += Mv[i] * search[i]
                gauss_b += (Ma[i] - f_smooth[i]) * search[i]

            var p0_d1 = gauss_b
            var p0_d2 = gauss_a
            for e_idx in range(num_edges_b):
                # d1 gets -f*Jv in EVERY active state (a saturated box row
                # still pushes); d2 gets curvature only where the cost is
                # quadratic. Collapsing these two into one `jar < 0` test is
                # what makes box rows wrong.
                var st_p = Int(rebind[Scalar[DTYPE]](state_e_sh[e_idx]))
                var f_p = scalar_row_force[DTYPE](
                    st_p,
                    jar[e_idx],
                    rebind[Scalar[DTYPE]](De_sh[e_idx]),
                    rebind[Scalar[DTYPE]](floss_e_sh[e_idx]),
                )
                p0_d1 += -f_p * Jv_e[e_idx]
                if st_p == SROW_QUADRATIC:
                    p0_d2 += (
                        rebind[Scalar[DTYPE]](De_sh[e_idx])
                        * Jv_e[e_idx]
                        * Jv_e[e_idx]
                    )
            if p0_d2 < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                p0_d2 = Scalar[DTYPE](PRIMAL_MINVAL_GPU)

            var alpha: Scalar[DTYPE] = 0
            if p0_d1 < Scalar[DTYPE](0):
                alpha = -p0_d1 / p0_d2

                var old_cost_ls: Scalar[DTYPE] = 0
                for i in range(NV):
                    old_cost_ls += (
                        Scalar[DTYPE](0.5)
                        * (Ma[i] - f_smooth[i])
                        * (qacc[i] - qacc_smooth[i])
                    )
                for e_idx in range(num_edges_b):
                    old_cost_ls += scalar_row_cost[DTYPE](
                        Int(rebind[Scalar[DTYPE]](state_e_sh[e_idx])),
                        jar[e_idx],
                        rebind[Scalar[DTYPE]](De_sh[e_idx]),
                        rebind[Scalar[DTYPE]](R_e_sh[e_idx]),
                        rebind[Scalar[DTYPE]](floss_e_sh[e_idx]),
                    )

                for _ in range(LINESEARCH_ITER):
                    var trial_cost: Scalar[DTYPE] = 0
                    for i in range(NV):
                        var qa_t = qacc[i] + alpha * search[i]
                        var Ma_t = Ma[i] + alpha * Mv[i]
                        trial_cost += (
                            Scalar[DTYPE](0.5)
                            * (Ma_t - f_smooth[i])
                            * (qa_t - qacc_smooth[i])
                        )
                    for e_idx in range(num_edges_b):
                        var jar_t = jar[e_idx] + alpha * Jv_e[e_idx]
                        # Re-classify at the TRIAL point: a step can move a row
                        # across a zone boundary, which is the whole reason the
                        # line search exists.
                        var st_t = scalar_row_state[DTYPE](
                            Int(rebind[Scalar[DTYPE]](kind_e_sh[e_idx])),
                            jar_t,
                            rebind[Scalar[DTYPE]](R_e_sh[e_idx]),
                            rebind[Scalar[DTYPE]](floss_e_sh[e_idx]),
                        )
                        trial_cost += scalar_row_cost[DTYPE](
                            st_t,
                            jar_t,
                            rebind[Scalar[DTYPE]](De_sh[e_idx]),
                            rebind[Scalar[DTYPE]](R_e_sh[e_idx]),
                            rebind[Scalar[DTYPE]](floss_e_sh[e_idx]),
                        )
                    if trial_cost <= old_cost_ls:
                        break
                    alpha *= Scalar[DTYPE](0.5)

            if alpha < Scalar[DTYPE](1e-10):
                ctrl_sh[1] = Scalar[DTYPE](1)  # done (break next iter)
            else:
                ctrl_sh[1] = Scalar[DTYPE](0)

                # Save old state for revert.
                for i in range(NV):
                    old_qacc[i] = qacc[i]
                    old_Ma[i] = Ma[i]
                    old_qfrc[i] = qfrc[i]
                for e_idx in range(num_edges_b):
                    old_jar[e_idx] = jar[e_idx]
                    old_force[e_idx] = rebind[Scalar[DTYPE]](force_sh[e_idx])

                old_cost = Scalar[DTYPE](0)
                for i in range(NV):
                    old_cost += (
                        Scalar[DTYPE](0.5)
                        * (Ma[i] - f_smooth[i])
                        * (qacc[i] - qacc_smooth[i])
                    )
                for e_idx in range(num_edges_b):
                    old_cost += scalar_row_cost[DTYPE](
                        Int(rebind[Scalar[DTYPE]](state_e_sh[e_idx])),
                        jar[e_idx],
                        rebind[Scalar[DTYPE]](De_sh[e_idx]),
                        rebind[Scalar[DTYPE]](R_e_sh[e_idx]),
                        rebind[Scalar[DTYPE]](floss_e_sh[e_idx]),
                    )

                for i in range(NV):
                    qacc[i] += alpha * search[i]
                    Ma[i] += alpha * Mv[i]

            # Publish qacc unconditionally. When alpha<1e-10 qacc is unchanged,
            # so the cooperative recompute reproduces identical jar/force/qfrc.
            for i in range(NV):
                qacc_sh[i] = qacc[i]

        # Cooperative jar/force/qfrc recompute, then tid 0 reads back and
        # finishes the accept/revert.
        barrier()
        _recompute_jfq_coop[DTYPE, NV, V_SIZE, ME, JE_AS](
            tid, THREADS, num_edges_b, Je_sh, De_sh, bias_e_sh,
            kind_e_sh, R_e_sh, floss_e_sh, state_e_sh, qacc_sh,
            jar_sh, force_sh, qfrc_sh,
        )
        barrier()
        if valid_env and tid == 0:
            for e_idx in range(num_edges_b):
                jar[e_idx] = rebind[Scalar[DTYPE]](jar_sh[e_idx])
            for i in range(NV):
                qfrc[i] = rebind[Scalar[DTYPE]](qfrc_sh[i])
            if Int(rebind[Scalar[DTYPE]](ctrl_sh[1])) == 0:
                var new_cost: Scalar[DTYPE] = 0
                for i in range(NV):
                    new_cost += (
                        Scalar[DTYPE](0.5)
                        * (Ma[i] - f_smooth[i])
                        * (qacc[i] - qacc_smooth[i])
                    )
                for e_idx in range(num_edges_b):
                    new_cost += scalar_row_cost[DTYPE](
                        Int(rebind[Scalar[DTYPE]](state_e_sh[e_idx])),
                        jar[e_idx],
                        rebind[Scalar[DTYPE]](De_sh[e_idx]),
                        rebind[Scalar[DTYPE]](R_e_sh[e_idx]),
                        rebind[Scalar[DTYPE]](floss_e_sh[e_idx]),
                    )

                var improvement = scale * (old_cost - new_cost)
                if (
                    improvement < Scalar[DTYPE](NEWTON_TOL_GPU)
                    and iter_n > 0
                ):
                    if improvement < Scalar[DTYPE](0):
                        for i in range(NV):
                            qacc[i] = old_qacc[i]
                            Ma[i] = old_Ma[i]
                            qfrc[i] = old_qfrc[i]
                        for e_idx in range(num_edges_b):
                            jar[e_idx] = old_jar[e_idx]
                            force_sh[e_idx] = old_force[e_idx]
                    ctrl_sh[1] = Scalar[DTYPE](1)  # done

        # force_sh updated; make visible for next assembly.
        barrier()
        if Int(rebind[Scalar[DTYPE]](ctrl_sh[1])) == 1:
            break

    # === THREAD 0: write back + reconstruct forces + equality/tendon ===
    if not valid_env or tid != 0:
        return

    for i in range(NV):
        qacc_constrained[env, i] = qacc[i]

    for c in range(nc):
        var fn_c: Scalar[DTYPE] = 0
        var ft1_c: Scalar[DTYPE] = 0
        var ft2_c: Scalar[DTYPE] = 0
        var mu_c = rebind[Scalar[DTYPE]](solver[env, pyr_sc + 2 * NE * MC + c])
        var safe_mu = mu_c
        if safe_mu < Scalar[DTYPE](1e-8):
            safe_mu = Scalar[DTYPE](1e-8)
        var f_e0 = rebind[Scalar[DTYPE]](force_sh[c * NE + 0])
        var f_e1 = rebind[Scalar[DTYPE]](force_sh[c * NE + 1])
        var f_e2 = rebind[Scalar[DTYPE]](force_sh[c * NE + 2])
        var f_e3 = rebind[Scalar[DTYPE]](force_sh[c * NE + 3])
        # `mju_decodePyramid`: the normal force is the SUM of the four edge
        # forces, NOT half of it. Both engines build each edge as
        # `Jn +- mu*Jt` with a FULL Jn (engine_core_constraint.c:1003), so
        # halving it made every pyramidal contact RECORD read half true
        # while qacc stayed correct — the solver works in edge forces and
        # only this write-back was wrong. Its two consumers are cfrc_ext
        # (hence Ant's contact_cost, a squared norm that had been costing a
        # quarter of what it should) and the quadruped force/torque
        # sensors. Fixed 2026-07-31.
        fn_c = f_e0 + f_e1 + f_e2 + f_e3
        var c_off = c * CONTACT_SIZE
        # Frictionless contacts carry no tangential force — see the identical
        # guard in the per-env path above for the measurement and the reason
        # `qacc` is unaffected while the sensors are not.
        var dim_c = Int(
            rebind[Scalar[DTYPE]](contacts[env, c_off + CONTACT_IDX_CONDIM])
        )
        if dim_c > 1:
            ft1_c = (f_e0 - f_e1) * safe_mu
            ft2_c = (f_e2 - f_e3) * safe_mu
        contacts[env, c_off + CONTACT_IDX_FORCE_N] = fn_c
        contacts[env, c_off + CONTACT_IDX_FORCE_T1] = ft1_c
        contacts[env, c_off + CONTACT_IDX_FORCE_T2] = ft2_c

    # Joint limits and fixed-tendon equalities are handled as edges above;
    # weld/connect equalities (and any SPATIAL equality tendon) remain a
    # separate post-solve step. Mirrors the PYRAMIDAL per-env path's gating,
    # including SKIP_FIXED — without it the tendon rows would be applied
    # twice, once inside the solve and once after.
    comptime SOLVER_ITER_GPU: Int = 50
    comptime if NEQUALITY > 0:
        _equality_env[
            DTYPE, NV, NBODY, NJOINT, NEQUALITY, NTENDON, NSITE, V_SIZE,
            BATCH, SOLVER_ITER_GPU,
        ](
            env, qvel, xpos, xquat, subtree_com, joints, bodies, mmeta,
            equality, tendons, sites, body_invweight0, dof_invweight0, cdof,
            m_inv, qacc_constrained,
        )
    comptime if NTENDON > 0:
        _tendon_env[
            DTYPE, NQ, NV, NBODY, NJOINT, NTENDON, NSITE, BATCH,
            SOLVER_ITER_GPU, SKIP_FIXED=True,
        ](
            env, qpos, qvel, joints, mmeta, tendons, sites, body_invweight0,
            dof_invweight0, m_inv, qacc_constrained,
        )


def solve_newton_blocked[
    target: StaticString,
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int = 0,
    NEQUALITY: Int = 0,
    NTENDON: Int = 0,
    NSITE: Int = 0,
    NEXCLUDE: Int = 0,
    NMESH_VERTS: Int = 0,
    CONE_TYPE: Int = ConeType.PYRAMIDAL,
    BATCH: Int = 1,
    MAX_CONDIM: Int = 3,
    NOSLIP_ITER: Int = 0,
](
    mut d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH],
    mut m: Model[
        DTYPE,
        NV,
        NBODY,
        NJOINT,
        NGEOM,
        NEQUALITY,
        NTENDON,
        NSITE,
        NEXCLUDE,
        NMESH_VERTS,
    ],
    mut scratch: DynamicsScratch[DTYPE, NV, NBODY, BATCH],
    mut cscratch: ContactScratch[DTYPE, NV, MAX_CONTACTS, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """PYRAMIDAL-only ONE-ENV-PER-BLOCK Newton contact solve (fields port of
    NewtonSolver.solve_gpu_blocked). Cooperative across MAX_CONTACTS threads,
    big matrices in shared memory — the OOM-safe path at humanoid scale.

    Writes into `scratch.qacc_constrained` (+ solved forces into `d.contacts`).
    Same signature family as `solve_newton`. Only the GPU (blocked)
    launch is meaningful; the CPU branch falls back to the single-source per-env
    body (`_newton_solve_env`, identical PYRAMIDAL math) for parity.
    """
    comptime MC = _max_one[MAX_CONTACTS]()
    comptime SOLVER_WS = 81 * MC + 12 * MC * NV

    comptime L_NV = Layout.row_major(BATCH, NV)
    comptime L_B3 = Layout.row_major(BATCH, NBODY * 3)
    comptime L_B4 = Layout.row_major(BATCH, NBODY * 4)
    comptime L_CON = Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE)
    comptime L_SMETA = Layout.row_major(BATCH, METADATA_SIZE)
    comptime L_JOINT = Layout.row_major(NJOINT, MODEL_JOINT_SIZE)
    comptime L_BODY = Layout.row_major(NBODY, MODEL_BODY_SIZE)
    comptime L_MMETA = Layout.row_major(MODEL_META_SIZE)
    comptime L_EQ = Layout.row_major(NEQUALITY, MODEL_EQ_SIZE)
    comptime L_TEN = Layout.row_major(NTENDON, MODEL_TENDON_SIZE)
    comptime L_SITE = Layout.row_major(NSITE, MODEL_SITE_SIZE)
    comptime L_BW = Layout.row_major(NBODY, 2)
    comptime L_CDOF = Layout.row_major(BATCH, NV * 6)
    comptime L_M = Layout.row_major(BATCH, NV * NV)
    comptime L_SOLVER = Layout.row_major(BATCH, SOLVER_WS)

    comptime L_QPOS = Layout.row_major(BATCH, NQ)
    comptime L_DW = Layout.row_major(NV)

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
        var M_v = scratch.M.lt["cpu", L_M]()
        var mi_v = scratch.m_inv.lt["cpu", L_M]()
        var qc_v = scratch.qacc_constrained.lt["cpu", L_NV]()
        var sol_v = cscratch.solver.lt["cpu", L_SOLVER]()
        for e in range(BATCH):
            _newton_solve_env[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, NEQUALITY,
                NTENDON, NSITE, CONE_TYPE, BATCH, SOLVER_WS,
 MAX_CONDIM,
 NOSLIP_ITER,
            ](
                e, qpos_v, qvel_v, xpos_v, xquat_v, stcom_v, con_v, smeta_v,
                joints_v, bodies_v, mmeta_v, eq_v, ten_v, site_v, bw_v, dw_v,
                cdof_v, M_v, mi_v, qc_v, sol_v,
            )
    else:
        var c = ctx.value()
        c.enqueue_function[
            _newton_blocked_fields_kernel[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, NEQUALITY,
                NTENDON, NSITE, CONE_TYPE, BATCH, SOLVER_WS,
 MAX_CONDIM,
 NOSLIP_ITER,
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
            scratch.M.lt["gpu", L_M](),
            scratch.m_inv.lt["gpu", L_M](),
            scratch.qacc_constrained.lt["gpu", L_NV](),
            cscratch.solver.lt["gpu", L_SOLVER](),
            grid_dim=(BATCH,),
            block_dim=(MC,),
        )

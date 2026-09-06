"""Stateful Euler integrator over per-field tensors (migration P2 pilot).

`EulerIntegrator` is the stateful replacement for the stateless
`EulerIntegrator.step_gpu` + caller-provided workspace slab: the struct OWNS
its `DynamicsScratch` (allocated once, reused every step) and sequences the
single-source stage ports into a full contact-free step:

    FK -> subtree_com -> cdof -> CRBA -> +armature -> LDL factor -> RNE ->
    fnet assembly (qfrc - bias - damping - stiffness - frictionloss) ->
    LDL solve -> qacc writeback -> finalize (implicit-damping re-solve +
    velocity/position integration + quaternion renormalize)

Assembly/finalize arithmetic is verbatim from the legacy Euler
`step_kernel` (:744) / `step_finalize_kernel` (:2140). Body velocities
(xvel/xangvel, consumed by env obs and future fluid forces) run right after
FK, matching legacy step order. Deliberately NOT yet ported (raise on use):
fluid forces (density/viscosity > 0), contacts, limits, constraint solving
— the P4 scope.

vs legacy: 9 small per-stage kernel launches instead of 2 fused monoliths —
each stage is independently gated; fusion is a later NVIDIA perf lever."""

from std.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from ..kinematics.quat_math import quat_integrate, quat_normalize
from ..kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from ..dynamics.subtree_com import compute_subtree_com
from ..dynamics.cdof import compute_cdof
from ..dynamics.mass_matrix import compute_mass_matrix
from std.time import perf_counter_ns
from ..fields.scratch import Scratch, cap
from ..dynamics.ldl import (
    _dof_ancestors,
    ldl_factor,
    ldl_solve,
    compute_m_inv,
    _ldl_factor_env,
    _ldl_solve_env,
    _ldl_factor_tree_env,
    _ldl_solve_tree_env,
)
from ..constraints.limits import solve_limits
from ..constraints.friction_dof import solve_friction
from ..constraints.contact_solve import solve_contacts
from ..solver.newton_solve import solve_newton
from ..solver.warmstart import save_qacc_warmstart
from ..solver.je_budget import je_ws_size
from ..solver.cg_solve import solve_cg
from ..solver.island_pgs_solve import solve_island_pgs
from ..collision.broadphase_sap import detect_contacts_auto
from ..types import ConeType
from ..dynamics.rne import compute_bias_forces_rne
from ..dynamics.rne_post import compute_rne_post
from ..dynamics.fluid_forces import compute_fluid_forces
from ..dynamics.gravcomp import compute_gravcomp_forces
from ..joint_types import JNT_FREE, JNT_BALL, JNT_HINGE, JNT_SLIDE
from ..fields import (
    Data,
    Model,
    DynamicsScratch,
    ContactScratch,
    Dims,
    DimsLike,
    AsStatic,
    DYN1,
    DYN2,
    rl1,
    rl2,
)
from ..gpu.constants import (
    MJ_MAXVAL,
    MODEL_JOINT_SIZE,
    MODEL_TREE_SIZE,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_EULERDAMP_DISABLED,
    MODEL_META_IDX_DENSITY,
    MODEL_META_IDX_VISCOSITY,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_ARMATURE,
    JOINT_IDX_DAMPING,
    JOINT_IDX_STIFFNESS,
    JOINT_IDX_SPRINGREF,
    JOINT_IDX_FRICTIONLOSS,
    MODEL_META_IDX_NTREE,
)

comptime EU_TPB: Int = 64


# ── armature: M diagonal += armature (verbatim step_kernel 6b) ────────────

# ⚠ A STAGE PROBE FOR THE CPU EULER STEP, off and free by default (the twin of
# `newton_solve._CPU_PROBE`). On, `step["cpu"]` prints one `[eprobe]` line per
# step with nanoseconds per stage: FK, body velocities, subtree COM, cdof,
# CRBA, armature, LDL factor, M^-1, RNE, fnet assembly, LDL solve, writeback,
# collision, the constraint solve, warm-start save, finalize. Fold with awk.
comptime _EULER_PROBE: Bool = False


@always_inline
def _armature_env[
    DTYPE: DType,
    D: DimsLike,
    L_JOINTS: Layout,
    L_M: Layout](
    env: Int,
    dims: D,
    joints: LayoutTensor[
        DTYPE, L_JOINTS, MutAnyOrigin
    ],
    M: LayoutTensor[DTYPE, L_M, MutAnyOrigin],
):
    var nv = dims.get_nv()
    var njoint = dims.get_njoint()
    for j in range(njoint):
        var jnt_type = Int(joints[j, JOINT_IDX_TYPE])
        var dof_adr = Int(joints[j, JOINT_IDX_DOF_ADR])
        var arm = joints[j, JOINT_IDX_ARMATURE]
        var diag_add = arm
        if jnt_type == JNT_FREE:
            for d in range(6):
                M[env, (dof_adr + d) * nv + (dof_adr + d)] += diag_add
        elif jnt_type == JNT_BALL:
            for d in range(3):
                M[env, (dof_adr + d) * nv + (dof_adr + d)] += diag_add
        else:
            M[env, dof_adr * nv + dof_adr] += diag_add


# ── fnet assembly: qfrc - bias - damping - stiffness - friction ───────────
# (verbatim step_kernel 9 + 8b; fluid 8c NOT ported — guarded at step())
@always_inline
def _fnet_passive_env[
    DTYPE: DType,
    D: DimsLike,
    L_QPOS: Layout,
    L_QVEL: Layout,
    L_JOINTS: Layout](
    env: Int,
    dims: D,
    qpos: LayoutTensor[DTYPE, L_QPOS, MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, L_QVEL, MutAnyOrigin],
    qfrc: LayoutTensor[DTYPE, L_QVEL, MutAnyOrigin],
    joints: LayoutTensor[
        DTYPE, L_JOINTS, MutAnyOrigin
    ],
    bias: LayoutTensor[DTYPE, L_QVEL, MutAnyOrigin],
    fnet: LayoutTensor[DTYPE, L_QVEL, MutAnyOrigin],
):
    var nv = dims.get_nv()
    var njoint = dims.get_njoint()
    # f_net = qfrc - bias
    for i in range(nv):
        var qfrc_v = rebind[Scalar[DTYPE]](qfrc[env, i])
        var bias_val = rebind[Scalar[DTYPE]](bias[env, i])
        fnet[env, i] = qfrc_v - bias_val

    # Damping: f -= damping * qvel (explicit part)
    for j in range(njoint):
        var jnt_type_d = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var dof_adr_d = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR])
        )
        var damp_d = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DAMPING])
        if damp_d > Scalar[DTYPE](0):
            var nd = 1
            if jnt_type_d == JNT_FREE:
                nd = 6
            elif jnt_type_d == JNT_BALL:
                nd = 3
            for d in range(nd):
                var v = rebind[Scalar[DTYPE]](qvel[env, dof_adr_d + d])
                var cur = rebind[Scalar[DTYPE]](fnet[env, dof_adr_d + d])
                fnet[env, dof_adr_d + d] = cur - damp_d * v

    # Stiffness + frictionloss
    for j in range(njoint):
        var jnt_type = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var dof_adr = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR]))
        var qpos_adr = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS_ADR])
        )
        var stiff = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_STIFFNESS])
        var sref = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_SPRINGREF])
        var floss = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_FRICTIONLOSS])
        if stiff > Scalar[DTYPE](0):
            var nd = 1
            if jnt_type == JNT_FREE:
                nd = 6
            elif jnt_type == JNT_BALL:
                nd = 3
            for d in range(nd):
                var qpos_d = rebind[Scalar[DTYPE]](qpos[env, qpos_adr + d])
                var cur = rebind[Scalar[DTYPE]](fnet[env, dof_adr + d])
                fnet[env, dof_adr + d] = cur - stiff * (qpos_d - sref)
        # frictionloss is NOT a passive force. It used to be applied here as an
        # explicit Coulomb force with a 1e-4 velocity deadband, which cannot
        # arrest motion — it overshoots zero and settles into a period-2 limit
        # cycle, so a joint that should stop dead spins forever (dm_control's
        # finger spinner: MuJoCo 1e-17 rad/s, ours a bit-constant +-0.0329).
        # MuJoCo solves it as a CONSTRAINT ROW (`mjCNSTR_FRICTION_DOF`) whose
        # force is bounded by frictionloss rather than fixed at it; that lives
        # in `constraints/friction_dof.mojo` and runs beside the limit rows in
        # every solver. `floss` is read there, not here.
        _ = floss


# ── qacc writeback: state qacc + qacc_constrained = qacc_ws ───────────────
@always_inline
def _qacc_writeback_env[
    DTYPE: DType,
    D: DimsLike,
    L_QACC_WS: Layout](
    env: Int,
    dims: D,
    qacc_ws: LayoutTensor[DTYPE, L_QACC_WS, MutAnyOrigin],
    qacc: LayoutTensor[DTYPE, L_QACC_WS, MutAnyOrigin],
    qacc_constrained: LayoutTensor[
        DTYPE, L_QACC_WS, MutAnyOrigin
    ],
):
    var nv = dims.get_nv()
    for i in range(nv):
        var qacc_val = rebind[Scalar[DTYPE]](qacc_ws[env, i])
        qacc[env, i] = qacc_val
        qacc_constrained[env, i] = qacc_val


# ── the integrate half of finalize, shared by both damping branches ───────
#
# ⚠ EXTRACTED, NOT COPIED. `mj_EulerSkip` has two branches — explicit when
# `mjDSBL_EULERDAMP` is set (or no dof is damped) and implicit otherwise —
# and they differ ONLY in how `qacc_final` is obtained. Writing the velocity
# and position update once means the two cannot drift; this tree has been
# bitten by a rule written inline twice before.
@always_inline
def _finalize_integrate_env[
    DTYPE: DType,
    DIMS: DimsLike,
    L_QPOS: Layout,
    L_QVEL: Layout,
    L_JOINTS: Layout,
](
    env: Int,
    dt: Scalar[DTYPE],
    dims: DIMS,
    qpos: LayoutTensor[DTYPE, L_QPOS, MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, L_QVEL, MutAnyOrigin],
    qacc: LayoutTensor[DTYPE, L_QVEL, MutAnyOrigin],
    joints: LayoutTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    qacc_ws: LayoutTensor[DTYPE, L_QVEL, MutAnyOrigin],
):
    var nv = dims.get_nv()
    var njoint = dims.get_njoint()
    # Step 5: v_new = v_old + dt * qacc_final.
    #
    # ⚠ THE BOUND IS `mjMAXVAL`, NOT A STABILITY BUDGET. MuJoCo's only
    # velocity guard is `mj_checkVel`, which WARNS and resets the state
    # when a dof goes NaN/inf or past 1e10; it never rescales one. A
    # tighter saturation here is a silent physics change that only shows
    # up on the models fast enough to reach it.
    for i in range(nv):
        var old_qvel = rebind[Scalar[DTYPE]](qvel[env, i])
        var qacc_final = rebind[Scalar[DTYPE]](qacc_ws[env, i])
        qacc[env, i] = qacc_final
        var qvel_new = old_qvel + qacc_final * dt
        var qvel_max = Scalar[DTYPE](MJ_MAXVAL)
        if qvel_new != qvel_new:  # NaN guard
            qvel_new = Scalar[DTYPE](0.0)
        elif qvel_new > qvel_max:
            qvel_new = qvel_max
        elif qvel_new < -qvel_max:
            qvel_new = -qvel_max
        qvel[env, i] = qvel_new

    # Integrate position (quaternion-aware for FREE; BALL not handled, as
    # in the legacy finalize)
    for j in range(njoint):
        var jnt_type = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var jnt_qpos_adr = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS_ADR])
        )
        var jnt_dof_adr = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR])
        )

        if jnt_type == JNT_FREE:
            for d in range(3):
                var qp = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr + d])
                var qv = rebind[Scalar[DTYPE]](qvel[env, jnt_dof_adr + d])
                qpos[env, jnt_qpos_adr + d] = qp + qv * dt
            # FREE-joint qpos stores [tx,ty,tz, qw,qx,qy,qz] — w FIRST at +3
            var qw = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr + 3])
            var qx = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr + 4])
            var qy = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr + 5])
            var qz = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr + 6])
            var wx = rebind[Scalar[DTYPE]](qvel[env, jnt_dof_adr + 3])
            var wy = rebind[Scalar[DTYPE]](qvel[env, jnt_dof_adr + 4])
            var wz = rebind[Scalar[DTYPE]](qvel[env, jnt_dof_adr + 5])
            var result = quat_integrate(qx, qy, qz, qw, wx, wy, wz, dt)
            var norm = quat_normalize(
                result[0], result[1], result[2], result[3]
            )
            qpos[env, jnt_qpos_adr + 3] = norm[3]  # qw
            qpos[env, jnt_qpos_adr + 4] = norm[0]  # qx
            qpos[env, jnt_qpos_adr + 5] = norm[1]  # qy
            qpos[env, jnt_qpos_adr + 6] = norm[2]  # qz

        # ⚠⚠ THIS BRANCH DID NOT EXIST, so a `<joint type="ball">` NEVER
        # MOVED. Its three DOFs accumulated velocity that nothing applied to
        # `qpos`, while the quaternion stayed at whatever the reset left —
        # a joint that is free in the mass matrix and frozen on screen.
        # `kinematics/integrate_pos.mojo` has carried the correct body since
        # it was written and has no callers; the integrators each roll their
        # own qpos loop and only FREE and HINGE/SLIDE were ever transcribed.
        #
        # ⚠ MuJoCo FALLS THROUGH from FREE into BALL (`mj_integratePos`) —
        # the free joint's rotation IS this update on shifted addresses, which
        # is why the two are the same four lines and must stay that way.
        # qpos holds the quaternion w FIRST; `quat_math` takes and returns
        # (x, y, z, w).
        elif jnt_type == JNT_BALL:
            var bqw = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr + 0])
            var bqx = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr + 1])
            var bqy = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr + 2])
            var bqz = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr + 3])
            var bwx = rebind[Scalar[DTYPE]](qvel[env, jnt_dof_adr + 0])
            var bwy = rebind[Scalar[DTYPE]](qvel[env, jnt_dof_adr + 1])
            var bwz = rebind[Scalar[DTYPE]](qvel[env, jnt_dof_adr + 2])
            var bres = quat_integrate(bqx, bqy, bqz, bqw, bwx, bwy, bwz, dt)
            var bnorm = quat_normalize(bres[0], bres[1], bres[2], bres[3])
            qpos[env, jnt_qpos_adr + 0] = bnorm[3]  # qw
            qpos[env, jnt_qpos_adr + 1] = bnorm[0]  # qx
            qpos[env, jnt_qpos_adr + 2] = bnorm[1]  # qy
            qpos[env, jnt_qpos_adr + 3] = bnorm[2]  # qz

        elif jnt_type == JNT_HINGE or jnt_type == JNT_SLIDE:
            var qp = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr])
            var qv = rebind[Scalar[DTYPE]](qvel[env, jnt_dof_adr])
            qpos[env, jnt_qpos_adr] = qp + qv * dt


# ── finalize: implicit-damping re-solve + integrate (verbatim :2140) ──────


@always_inline
def _finalize_damping_env[
    DTYPE: DType,
    DIMS: DimsLike,
    L_JOINTS: Layout,
    L_M: Layout,
](
    env: Int,
    dt: Scalar[DTYPE],
    dims: DIMS,
    joints: LayoutTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    M: LayoutTensor[DTYPE, L_M, MutAnyOrigin],
):
    """`M_hat = M + dt * diag(damping)`, in place — `mj_Euler`'s implicit
    damping (engine_forward.c, `eulerdamp`). ONE spelling, shared by the dense
    and the tree-ordered finalize."""
    var nv = dims.get_nv()
    for j in range(dims.get_njoint()):
        var jnt_type = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var dof_adr = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR]))
        var damp = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DAMPING])
        if damp > Scalar[DTYPE](0):
            var nd = 1
            if jnt_type == JNT_FREE:
                nd = 6
            elif jnt_type == JNT_BALL:
                nd = 3
            for d in range(nd):
                M[env, (dof_adr + d) * nv + (dof_adr + d)] += dt * damp


@always_inline
def _finalize_tree_env[
    DTYPE: DType,
    DIMS: DimsLike,
    L_QPOS: Layout,
    L_QVEL: Layout,
    L_JOINTS: Layout,
    L_M: Layout,
    L_P: Layout,
](
    env: Int,
    dt: Scalar[DTYPE],
    dims: DIMS,
    qpos: LayoutTensor[DTYPE, L_QPOS, MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, L_QVEL, MutAnyOrigin],
    qacc: LayoutTensor[DTYPE, L_QVEL, MutAnyOrigin],
    joints: LayoutTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    M: LayoutTensor[DTYPE, L_M, MutAnyOrigin],
    L: LayoutTensor[DTYPE, L_M, MutAnyOrigin],
    D: LayoutTensor[DTYPE, L_QVEL, MutAnyOrigin],
    fnet: LayoutTensor[DTYPE, L_QVEL, MutAnyOrigin],
    qacc_ws: LayoutTensor[DTYPE, L_QVEL, MutAnyOrigin],
    qacc_constrained: LayoutTensor[DTYPE, L_QVEL, MutAnyOrigin],
    dofp: LayoutTensor[DTYPE, L_P, MutAnyOrigin],
    eulerdamp_off: Bool,
):
    """`_finalize_env` on the tree-ordered LDL (CPU, `NTREE > 0`).

    The dense twin below was the one caller of the forward-elimination LDL
    that §13.10 did not switch — it calls `_ldl_factor_env` directly, not
    through the dispatcher — and formed `M * qacc` as a dense nv^2 matvec:
    34 us of humanoid_CMU's 208 and 71 of dog's 663 against MuJoCo's
    `mj_Euler` at 4.6 / 6.8 (PERFORMANCE.md §13.14, item 3). Here the
    right-hand side walks M's tree sparsity (row i is nonzero at i's
    ancestors, plus the symmetric entries — `mj_mulM`'s shape) and the
    factorisation and solve are `mj_factorI` / `mj_solveLD` on our storage.
    Not bit-exact against the dense twin: a different rounding of the same
    `M_hat^-1 (M qacc)`, gated by the Euler-model MuJoCo tests."""
    var nv = dims.get_nv()
    if eulerdamp_off:
        for i in range(nv):
            qacc_ws[env, i] = rebind[Scalar[DTYPE]](qacc_constrained[env, i])
        _finalize_integrate_env(
            env, dt, dims, qpos, qvel, qacc, joints, qacc_ws
        )
        return
    comptime V_CAP = cap[DIMS.NV]()
    comptime A_CAP = V_CAP * V_CAP
    var par = Scratch[Int, V_CAP](nv, uninitialized=0)
    for i in range(nv):
        par[i] = Int(dofp[i])
    # rhs = M * qacc_constrained over M's sparsity: the diagonal, then each
    # (i, ancestor j) entry once for both rows it belongs to. Ancestor lists
    # and pointers in place of the `par` chase through `LayoutTensor`
    # (PERFORMANCE.md §13.26); the accumulation order per entry — diagonal,
    # own ancestors parent-first, then descendants ascending — is the walk's,
    # so this is the same arithmetic.
    var dep = Scratch[Int, V_CAP](nv, uninitialized=0)
    var anc = Scratch[Int, A_CAP](nv * nv, uninitialized=0)
    _dof_ancestors[V_CAP, A_CAP](nv, par, dep, anc)
    var Mp = M.ptr + env * nv * nv
    var fq = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
    var qc = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
    var fp = fq.unsafe_ptr()
    var qp = qc.unsafe_ptr()
    for i in range(nv):
        qp[i] = rebind[Scalar[DTYPE]](qacc_constrained[env, i])
        fp[i] = Mp[i * nv + i] * qp[i]
    for i in range(nv):
        var qi = qp[i]
        var ri = i * nv
        var acc = fp[i]
        for a in range(dep[i] - 1, -1, -1):
            var j = anc[ri + a]
            var mij = Mp[ri + j]
            acc = acc + mij * qp[j]
            fp[j] = fp[j] + mij * qi
        fp[i] = acc
    for i in range(nv):
        fnet[env, i] = fp[i]
    _finalize_damping_env[DTYPE](env, dt, dims, joints, M)
    _ldl_factor_tree_env(env, dims, M, L, D, dofp)
    _ldl_solve_tree_env(env, dims, L, D, fnet, qacc_ws, dofp)
    _finalize_integrate_env(
        env, dt, dims, qpos, qvel, qacc, joints, qacc_ws
    )


@always_inline
def _finalize_env[
    DTYPE: DType,
    DIMS: DimsLike,
    L_QPOS: Layout,
    L_QVEL: Layout,
    L_JOINTS: Layout,
    L_M: Layout,
    L_TREE: Layout](
    env: Int,
    dt: Scalar[DTYPE],
    dims: DIMS,
    qpos: LayoutTensor[DTYPE, L_QPOS, MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, L_QVEL, MutAnyOrigin],
    qacc: LayoutTensor[DTYPE, L_QVEL, MutAnyOrigin],
    joints: LayoutTensor[
        DTYPE, L_JOINTS, MutAnyOrigin
    ],
    M: LayoutTensor[DTYPE, L_M, MutAnyOrigin],
    L: LayoutTensor[DTYPE, L_M, MutAnyOrigin],
    D: LayoutTensor[DTYPE, L_QVEL, MutAnyOrigin],
    fnet: LayoutTensor[DTYPE, L_QVEL, MutAnyOrigin],
    qacc_ws: LayoutTensor[DTYPE, L_QVEL, MutAnyOrigin],
    qacc_constrained: LayoutTensor[
        DTYPE, L_QVEL, MutAnyOrigin
    ],
    # ⚠ M's DIAGONAL BLOCKS, for the `M_hat` re-factorisation below. Threaded
    # and not yet read — see `dynamics/ldl.mojo`'s header. ⚠⚠ NOTE THIS IS A
    # SECOND DENSE LDL PER STEP, distinct from the `ldl_factor` in `step`:
    # `M_hat = M + dt*diag(damping)` adds to the DIAGONAL only, so it has the
    # same block structure and the same O(nv^3) cost.
    trees: LayoutTensor[DTYPE, L_TREE, MutAnyOrigin],
    # ⚠⚠ `<option><flag eulerdamp="disable"/></option>` — mjDSBL_EULERDAMP.
    # `mj_EulerSkip` puts its ENTIRE damping scan behind this flag and, when
    # it is set, integrates velocity explicitly: `qvel += h * qacc`, with the
    # `(M + h*diag(B))` solve below skipped, not merely fed a zero B. The two
    # are far apart whenever `h*B` is comparable to `M`'s diagonal — on
    # tetheria (`h = 0.01`, `dof_damping = 0.1`, `M_ii ~ 1.6e-03`) the solve
    # returns 61.5% of the explicit velocity, which was that model's whole
    # remaining residual.
    eulerdamp_off: Bool,
):
    var nv = dims.get_nv()
    var njoint = dims.get_njoint()
    if eulerdamp_off:
        # Explicit velocity integration — `mj_EulerSkip`'s `if (!dof_damping)`
        # branch, which copies `d->qacc` and never touches `qH`.
        for i in range(nv):
            qacc_ws[env, i] = rebind[Scalar[DTYPE]](qacc_constrained[env, i])
        _finalize_integrate_env(
            env, dt, dims, qpos, qvel, qacc, joints, qacc_ws
        )
        return
    # Step 1: rhs = M * qacc_constrained (into fnet)
    for i in range(nv):
        var sum = Scalar[DTYPE](0)
        for j in range(nv):
            var M_ij = rebind[Scalar[DTYPE]](M[env, i * nv + j])
            var qacc_j = rebind[Scalar[DTYPE]](qacc_constrained[env, j])
            sum += M_ij * qacc_j
        fnet[env, i] = sum

    # Step 2: M_hat = M + dt*D (damping to diagonal)
    _finalize_damping_env[DTYPE](env, dt, dims, joints, M)

    # Step 3+4: re-factor M_hat, solve qacc_final = M_hat^{-1} * rhs
    _ldl_factor_env(env, dims, M, L, D, trees)
    _ldl_solve_env(env, dims, L, D, fnet, qacc_ws, trees)

    _finalize_integrate_env(
        env, dt, dims, qpos, qvel, qacc, joints, qacc_ws
    )


# ── launchable kernels ────────────────────────────────────────────────────
def _armature_kernel[
    DTYPE: DType, NV: Int, NJOINT: Int, BATCH: Int
](
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    M: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _armature_env[DTYPE](env, Dims[nv=NV, njoint=NJOINT](), joints, M)


def _fnet_passive_kernel[
    DTYPE: DType, NQ: Int, NV: Int, NJOINT: Int, BATCH: Int
](
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    qfrc: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    bias: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    fnet: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _fnet_passive_env[DTYPE](
        env, Dims[nq=NQ, nv=NV, njoint=NJOINT](), qpos, qvel, qfrc, joints, bias, fnet
    )


def _qacc_writeback_kernel[
    DTYPE: DType, NV: Int, BATCH: Int
](
    qacc_ws: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    qacc: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    qacc_constrained: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin
    ],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _qacc_writeback_env[DTYPE](
        env, Dims[nv=NV](), qacc_ws, qacc, qacc_constrained
    )


def _finalize_kernel[
    DTYPE: DType, NQ: Int, NV: Int, NJOINT: Int, BATCH: Int
](
    dt: Scalar[DTYPE],
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    qacc: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    M: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    L: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    D: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    fnet: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    qacc_ws: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    qacc_constrained: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin
    ],
    trees: LayoutTensor[
        DTYPE, Layout.row_major(NV * MODEL_TREE_SIZE), MutAnyOrigin
    ],
    # ⚠ A SCALAR, NOT A `Bool`, so the launch does not need a second
    # argument kind — every other model constant reaching a kernel here
    # arrives the same way.
    eulerdamp_off: Scalar[DTYPE],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _finalize_env[DTYPE](
        env, dt, Dims[nq=NQ, nv=NV, njoint=NJOINT](), qpos, qvel, qacc, joints, M, L, D, fnet, qacc_ws,
        qacc_constrained, trees, eulerdamp_off != Scalar[DTYPE](0),
    )


# ── the stateful integrator ───────────────────────────────────────────────
struct EulerIntegrator[
    DTYPE: DType,
    D: DimsLike,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    BATCH: Int = 1,
    SOLVER: StaticString = "pgs",
    PARALLEL_GPU: Bool = False,
    CRBA_TREEWALK: Bool = False,
    RNE_POST: Bool = False,
    MAX_CONDIM: Int = 3,
    NOSLIP_ITER: Int = 0,
](Movable):
    """Owns its scratch; steps contact-free dynamics on either target. See
    module docstring for what is deliberately not yet ported.
    PARALLEL_GPU=True: the GPU FK / body-velocity / cdof / CRBA /
    LDL-factor / M^-1 / RNE stages run their cooperative within-env (_mt)
    kernels (bit-exact vs serial; other stages stay serial). CPU ignores
    it. CRBA_TREEWALK=True: the CRBA runs the tree-walk algorithm
    (O(NV·depth)) instead of the dense O(NV²·NBODY) one —
    float-tolerance-equal, NOT bit-exact vs dense. ⚠ IT APPLIES ON BOTH
    TARGETS NOW; it used to be GPU-only, which left every CPU caller on the
    dense kernel — 12.6× slower on Sawyer (NV=15, NBODY=34)."""

    var scratch: DynamicsScratch[Self.DTYPE, Self.D, Self.BATCH]
    # Blocked-Newton Jacobian spill size — 0 unless `Je` overflows threadgroup
    # memory. Computed HERE (not by the caller) because this struct already
    # carries every dimension it depends on, and via `je_budget` so the buffer
    # and the kernel that indexes it cannot drift apart.
    comptime JE_WS = je_ws_size[
        Self.DTYPE, Self.D.NV, Self.D.NJOINT, Self.D.NTENDON, Self.D.NEQUALITY,
        Self.D.MAX_CONTACTS, Self.MAX_CONDIM,
    ]()

    var cscratch: ContactScratch[Self.DTYPE, Self.D, Self.BATCH, Self.JE_WS]

    def __init__(out self) raises:
        """Dimensions from the comptime provider; raises on a dynamic one.

        ⚠ THE DIMS OVERLOAD BELOW IS WHAT A RUNTIME-LOADED MODEL NEEDS. The
        `step` body has been dimension-agnostic since 3a — it reads `d.dims`
        and builds `RuntimeLayout`s — so the ONLY thing that stood between
        this integrator and a `DynDims` model was this constructor, which
        allocates its scratch through the nullary path and therefore through
        `comptime_value()`. Same dual-constructor shape as `Model`, `Data`,
        `SpecFields` and both scratches (3a/3b).
        """
        self = Self(Self.D.comptime_value())

    def __init__(out self, dims: Self.D) raises:
        """Dimensions passed in, and ALLOCATED FROM — the runtime path."""
        self.scratch = DynamicsScratch[Self.DTYPE, Self.D, Self.BATCH](dims)
        self.cscratch = ContactScratch[
            Self.DTYPE, Self.D, Self.BATCH, Self.JE_WS
        ](dims)

    def prepare_gpu(mut self, ctx: DeviceContext) raises:
        """Allocate device buffers for the scratch (once, before stepping)."""
        self.scratch.upload_all(ctx)
        self.cscratch.upload_all(ctx)

    def step[
        target: StaticString, CONTACTS: Bool = True
    ](
        mut self,
        mut d: Data[Self.DTYPE, Self.D, Self.BATCH],
        mut m: Model[Self.DTYPE, Self.D],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """One full contact-free Euler step."""
        var dt = m.meta.data[MODEL_META_IDX_TIMESTEP]
        # `<option><flag eulerdamp="disable"/></option>`. Read HERE rather
        # than inside `_finalize_env` so both targets get it from the same
        # place and the GPU kernel takes it as a launch argument.
        var eulerdamp_off = (
            m.meta.data[MODEL_META_IDX_EULERDAMP_DISABLED]
            != Scalar[Self.DTYPE](0)
        )

        var _e_fk: Int = 0
        var _e_bodyvel: Int = 0
        var _e_subtree: Int = 0
        var _e_cdof: Int = 0
        var _e_crba: Int = 0
        var _e_armature: Int = 0
        var _e_ldlf: Int = 0
        var _e_minv: Int = 0
        var _e_rne: Int = 0
        var _e_fnet: Int = 0
        var _e_ldls: Int = 0
        var _e_wb: Int = 0
        var _e_coll: Int = 0
        var _e_newton: Int = 0
        var _e_warm: Int = 0
        var _e_fin: Int = 0
        var _e_last: Int = 0
        comptime if _EULER_PROBE:
            _e_last = Int(perf_counter_ns())
        forward_kinematics[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](d, m, ctx)
        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_fk += _e_now - _e_last
            _e_last = _e_now
        compute_body_velocities[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](d, m, ctx)
        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_bodyvel += _e_now - _e_last
            _e_last = _e_now
        compute_subtree_com[target, Self.DTYPE, BATCH=Self.BATCH](d, m, ctx)
        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_subtree += _e_now - _e_last
            _e_last = _e_now
        compute_cdof[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](d, m, self.scratch, ctx)
        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_cdof += _e_now - _e_last
            _e_last = _e_now
        compute_mass_matrix[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU, TREEWALK = Self.CRBA_TREEWALK](d, m, self.scratch, ctx)
        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_crba += _e_now - _e_last
            _e_last = _e_now

        comptime L_JOINT = Layout.row_major(Self.D.NJOINT, MODEL_JOINT_SIZE)
        comptime L_M = Layout.row_major(Self.BATCH, Self.D.NV * Self.D.NV)
        comptime L_TREE = Layout.row_major(Self.D.NV * MODEL_TREE_SIZE)
        comptime L_NV = Layout.row_major(Self.BATCH, Self.D.NV)
        comptime L_QPOS = Layout.row_major(Self.BATCH, Self.D.NQ)
        comptime BLOCKS = (Self.BATCH + EU_TPB - 1) // EU_TPB

        comptime if target == "cpu":
            var dm = d.dims
            var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
            var rl_M = rl2(Self.BATCH, dm.get_nv() * dm.get_nv())
            var joints_v = m.joints.lt_dyn["cpu", DYN2](rl_JOINT)
            var M_v = self.scratch.M.lt_dyn["cpu", DYN2](rl_M)
            for e in range(Self.BATCH):
                _armature_env[
                    Self.DTYPE](e, dm, joints_v, M_v)
        else:
            ctx.value().enqueue_function[
                _armature_kernel[Self.DTYPE, Self.D.NV, Self.D.NJOINT, Self.BATCH]
            ](
                m.joints.lt["gpu", L_JOINT](),
                self.scratch.M.lt["gpu", L_M](),
                grid_dim=(BLOCKS,),
                block_dim=(EU_TPB,),
            )

        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_armature += _e_now - _e_last
            _e_last = _e_now
        ldl_factor[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](m, self.scratch, ctx)
        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_ldlf += _e_now - _e_last
            _e_last = _e_now
        # ⚠ `M^-1` IS FORMED ONLY FOR A READER. MuJoCo's Newton never builds
        # it: `mj_diagApprox` prices rows with the model-time `*_invweight0`,
        # and `mj_projectConstraint` solves against `qLD`, only for a dual
        # solver or noslip. Ours formed the full dense inverse every step —
        # O(nv^3), measured at 24-46% of every step past 20 dofs
        # (`PERFORMANCE.md` §13) — to feed a diagonal round-trip the row
        # builders no longer do. What still reads it: PGS/CG/island (their
        # whole `A R` machinery), `noslip_pyramidal` (M^-1 J^T per row), and
        # the connect/weld equality rows (`w_MinvJ`). The first two are
        # comptime facts; the last is a model count, read at RUNTIME so the
        # studio's dynamic leg makes the same decision as the static one.
        # ⚠ `CONTACTS` IS PART OF THE PREDICATE. With `CONTACTS=False` the seam
        # below runs the standalone `solve_limits` / `solve_friction` stages,
        # which are Gauss-Seidel over `J M^-1 J^T` and READ the inverse — the
        # bisect that found this was `test_limit_solref_per_joint`, whose
        # limit row went inert (153 of qacc) the moment the matrix was stale.
        comptime if CONTACTS and Self.SOLVER == "newton":
            # The Newton reads `M^-1` for equality rows, and its noslip does
            # unless the CPU tree path solves against the LDL factor instead
            # (`noslip._minv_apply`; the predicate is the LDL dispatcher's).
            var need_minv = d.dims.get_nequality() > 0
            comptime if Self.NOSLIP_ITER > 0:
                var tree_cpu = False
                comptime if target == "cpu":
                    tree_cpu = Int(m.meta.data[MODEL_META_IDX_NTREE]) > 0
                if not tree_cpu:
                    need_minv = True
            if need_minv:
                compute_m_inv[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](m, self.scratch, ctx)
        else:
            compute_m_inv[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](m, self.scratch, ctx)
        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_minv += _e_now - _e_last
            _e_last = _e_now
        compute_bias_forces_rne[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](d, m, self.scratch, ctx)
        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_rne += _e_now - _e_last
            _e_last = _e_now

        comptime if target == "cpu":
            var dm = d.dims
            var rl_QPOS = rl2(Self.BATCH, dm.get_nq())
            var rl_NV = rl2(Self.BATCH, dm.get_nv())
            var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
            var qpos_v = d.qpos.lt_dyn["cpu", DYN2](rl_QPOS)
            var qvel_v = d.qvel.lt_dyn["cpu", DYN2](rl_NV)
            var qfrc_v = d.qfrc.lt_dyn["cpu", DYN2](rl_NV)
            var joints_v2 = m.joints.lt_dyn["cpu", DYN2](rl_JOINT)
            var bias_v = self.scratch.bias.lt_dyn["cpu", DYN2](rl_NV)
            var fnet_v = self.scratch.fnet.lt_dyn["cpu", DYN2](rl_NV)
            for e in range(Self.BATCH):
                _fnet_passive_env[
                    Self.DTYPE](e, dm, qpos_v, qvel_v, qfrc_v, joints_v2, bias_v, fnet_v)
        else:
            ctx.value().enqueue_function[
                _fnet_passive_kernel[
                    Self.DTYPE, Self.D.NQ, Self.D.NV, Self.D.NJOINT, Self.BATCH
                ]
            ](
                d.qpos.lt["gpu", L_QPOS](),
                d.qvel.lt["gpu", L_NV](),
                d.qfrc.lt["gpu", L_NV](),
                m.joints.lt["gpu", L_JOINT](),
                self.scratch.bias.lt["gpu", L_NV](),
                self.scratch.fnet.lt["gpu", L_NV](),
                grid_dim=(BLOCKS,),
                block_dim=(EU_TPB,),
            )

        # 8c. Fluid drag into fnet (no-op unless meta density/viscosity > 0).
        compute_fluid_forces[target, Self.DTYPE, BATCH=Self.BATCH](d, m, self.scratch, ctx)
        # ⚠ AFTER the fluid call, not before: `mj_passive` adds `qfrc_fluid`
        # into `qfrc_passive` and only then `qfrc_gravcomp`
        # (engine_passive.c:1000-1022). No-op unless the model declares
        # `<body gravcomp>`.
        compute_gravcomp_forces[target, Self.DTYPE, BATCH=Self.BATCH](d, m, self.scratch, ctx)

        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_fnet += _e_now - _e_last
            _e_last = _e_now
        ldl_solve[target, Self.DTYPE, BATCH=Self.BATCH](m, self.scratch, ctx)
        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_ldls += _e_now - _e_last
            _e_last = _e_now

        comptime if target == "cpu":
            var dm = d.dims
            var rl_NV = rl2(Self.BATCH, dm.get_nv())
            var qacc_ws_v = self.scratch.qacc_ws.lt_dyn["cpu", DYN2](rl_NV)
            var qacc_v = d.qacc.lt_dyn["cpu", DYN2](rl_NV)
            var qacc_c_v = self.scratch.qacc_constrained.lt_dyn["cpu", DYN2](rl_NV)
            for e in range(Self.BATCH):
                _qacc_writeback_env[Self.DTYPE](
                    e, dm, qacc_ws_v, qacc_v, qacc_c_v
                )
        else:
            ctx.value().enqueue_function[
                _qacc_writeback_kernel[Self.DTYPE, Self.D.NV, Self.BATCH]
            ](
                self.scratch.qacc_ws.lt["gpu", L_NV](),
                d.qacc.lt["gpu", L_NV](),
                self.scratch.qacc_constrained.lt["gpu", L_NV](),
                grid_dim=(BLOCKS,),
                block_dim=(EU_TPB,),
            )

        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_wb += _e_now - _e_last
            _e_last = _e_now
        # Constraint seam (order matches the legacy PGS solver): contact
        # detection -> contact PGS -> joint limits, all updating
        # scratch.qacc_constrained. Equality/tendons join here later.
        comptime if CONTACTS:
            # Contact solve runs limits INSIDE (legacy PGS position: between
            # the normal and friction phases, PGS_ITERATIONS iterations) —
            # the standalone limits stage below is for CONTACTS=False only.
            # Auto broadphase = legacy production (SAP for NGEOM >= 16,
            # O(N^2) otherwise; same routing as the legacy step kernel's
            # detect_contacts_auto_gpu call).
            detect_contacts_auto[target, Self.DTYPE, BATCH=Self.BATCH](d, m, ctx)
            comptime if _EULER_PROBE:
                var _e_now = Int(perf_counter_ns())
                _e_coll += _e_now - _e_last
                _e_last = _e_now
            comptime assert (
                Self.SOLVER == "pgs"
                or Self.SOLVER == "newton"
                or Self.SOLVER == "cg"
                or Self.SOLVER == "island"
            ), (
                "EulerIntegrator: SOLVER must be 'pgs', 'newton', 'cg',"
                " or 'island'"
            )
            comptime if Self.SOLVER == "newton":
                solve_newton[target, Self.DTYPE, CONE_TYPE=Self.CONE_TYPE, BATCH=Self.BATCH, MAX_CONDIM=Self.MAX_CONDIM, NOSLIP_ITER=Self.NOSLIP_ITER, JE_WS=Self.JE_WS](d, m, self.scratch, self.cscratch, ctx)
            else:
                comptime if Self.SOLVER == "cg":
                    solve_cg[target, Self.DTYPE, CONE_TYPE=Self.CONE_TYPE, BATCH=Self.BATCH](d, m, self.scratch, self.cscratch, ctx)
                else:
                    comptime if Self.SOLVER == "island":
                        solve_island_pgs[target, Self.DTYPE, CONE_TYPE=Self.CONE_TYPE, BATCH=Self.BATCH](d, m, self.scratch, self.cscratch, ctx)
                    else:
                        solve_contacts[target, Self.DTYPE, CONE_TYPE=Self.CONE_TYPE, BATCH=Self.BATCH](d, m, self.scratch, self.cscratch, ctx)
        else:
            solve_limits[target, Self.DTYPE, BATCH=Self.BATCH](d, m, self.scratch, ctx)
            # Dry-friction dof rows. With CONTACTS=False no solver runs, so
            # this is the only place they can be applied; with contacts the
            # solvers call `_friction_env` themselves beside their limit rows.
            solve_friction[target, Self.DTYPE, BATCH=Self.BATCH](d, m, self.scratch, ctx)

        # `qacc_warmstart = qacc` — the tail of `mj_forward`
        # (engine_forward.c:1087). ⚠ HERE AND NOT AFTER THE INTEGRATOR: MuJoCo
        # saves the CONSTRAINT SOLVER's acceleration, and anything the
        # integrator does to `qacc` afterwards never reaches `qacc_warmstart`.
        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_newton += _e_now - _e_last
            _e_last = _e_now
        save_qacc_warmstart[target, Self.DTYPE, BATCH=Self.BATCH](
            d, self.scratch, ctx
        )

        # `mj_sensorAcc` sits exactly here in MuJoCo: after fwdConstraint,
        # before the integrator. Every input the stage needs (FK products,
        # solved contact forces, scratch.qacc_constrained) is valid at this
        # point and stale one line later — `_finalize_env` overwrites
        # d.qacc with the implicit-damping re-solve and moves qpos/qvel on.
        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_warm += _e_now - _e_last
            _e_last = _e_now
        comptime if Self.RNE_POST:
            compute_rne_post[target, Self.DTYPE, BATCH=Self.BATCH](d, m, self.scratch, ctx)

            # ⚠ AND THE FK PRODUCTS THAT GO WITH THEM — defect 19.
            #
            # `cacc`/`cfrc_int` are only half of an acceleration-stage sensor:
            # the other half is the site pose they are transported to and
            # rotated into. Those live in `site_xpos`/`xquat`, which
            # `Phyics3dEnv._fields_fk` moves to the POST-integration state
            # after the substep loop (correctly — the position/velocity-stage
            # observation dims need it, and dm_control's `mj_step1` does the
            # same). Reading them at observation time therefore mixed
            # pre-integration `cacc` with post-integration geometry.
            #
            # MuJoCo never has this problem because it evaluates the stage
            # HERE and stores the finished sensor value. We cannot do that
            # generically — the sensor set is per-CONFIG, not per-engine — so
            # the inputs are frozen instead, at the same instant and under the
            # same `RNE_POST` gate that writes `cacc`. A model without the
            # stage pays nothing.
            comptime N_SITE_ACC = Self.BATCH * Self.D.NSITE * 3
            comptime N_QUAT_ACC = Self.BATCH * Self.D.NBODY * 4
            comptime if target == "cpu":
                comptime if N_SITE_ACC > 0:
                    for i in range(N_SITE_ACC):
                        d.site_xpos_acc.data[i] = d.site_xpos.data[i]
                for i in range(N_QUAT_ACC):
                    d.xquat_acc.data[i] = d.xquat.data[i]
            else:
                # Device-to-device, both buffers owned and the same length by
                # construction. Mirrored on GPU rather than skipped: a snapshot
                # that only exists on one target is the same silent-divergence
                # shape this whole fix exists to remove.
                # ⚠ `as_unsafe_any_origin()` — `copy_from_device` wants
                # `MutAnyOrigin` and the buffer's pointer carries
                # `origin_of(dev._value)`. Same spelling as
                # `dreamerv3/param_sync.mojo`, which is the established caller.
                #
                # ⚠ COMPILES BUT IS NOT RUNTIME-GATED: no model with
                # `RNE_POST` runs on GPU today (dog and quadruped are both
                # CPU-only, because the batched facade carries no `act`). It is
                # written rather than skipped so the device path cannot silently
                # diverge the day one does, and it is flagged here rather than
                # left to look tested.
                var c = ctx.value()
                comptime if N_SITE_ACC > 0:
                    d.site_xpos_acc.copy_from_device(
                        c,
                        d.site_xpos.dev.value().unsafe_ptr().as_unsafe_any_origin(),
                        N_SITE_ACC,
                    )
                d.xquat_acc.copy_from_device(
                    c,
                    d.xquat.dev.value().unsafe_ptr().as_unsafe_any_origin(),
                    N_QUAT_ACC,
                )

        comptime if target == "cpu":
            var dm = d.dims
            var rl_QPOS = rl2(Self.BATCH, dm.get_nq())
            var rl_NV = rl2(Self.BATCH, dm.get_nv())
            var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
            var rl_M = rl2(Self.BATCH, dm.get_nv() * dm.get_nv())
            var qpos_v3 = d.qpos.lt_dyn["cpu", DYN2](rl_QPOS)
            var qvel_v3 = d.qvel.lt_dyn["cpu", DYN2](rl_NV)
            var qacc_v3 = d.qacc.lt_dyn["cpu", DYN2](rl_NV)
            var joints_v3 = m.joints.lt_dyn["cpu", DYN2](rl_JOINT)
            var M_v3 = self.scratch.M.lt_dyn["cpu", DYN2](rl_M)
            var L_v3 = self.scratch.L.lt_dyn["cpu", DYN2](rl_M)
            var D_v3 = self.scratch.D.lt_dyn["cpu", DYN2](rl_NV)
            var fnet_v3 = self.scratch.fnet.lt_dyn["cpu", DYN2](rl_NV)
            var qacc_ws_v3 = self.scratch.qacc_ws.lt_dyn["cpu", DYN2](rl_NV)
            var qacc_c_v3 = self.scratch.qacc_constrained.lt_dyn["cpu", DYN2](rl_NV)
            var rl_TREE = rl1(dm.get_nv() * MODEL_TREE_SIZE)
            var trees_v3 = m.trees.lt_dyn["cpu", DYN1](rl_TREE)
            # Tree-ordered when the parser built the tables (the LDL
            # dispatchers' own rule), dense otherwise.
            var use_tree3 = Int(m.meta.data[MODEL_META_IDX_NTREE]) > 0
            var P_v3 = m.dof_parentid.lt_dyn["cpu", DYN1](
                rl1(dm.get_nv() if dm.get_nv() > 0 else 1)
            )
            for e in range(Self.BATCH):
                if use_tree3:
                    _finalize_tree_env[Self.DTYPE](
                        e, dt, dm, qpos_v3, qvel_v3, qacc_v3, joints_v3, M_v3,
                        L_v3, D_v3, fnet_v3, qacc_ws_v3, qacc_c_v3, P_v3,
                        eulerdamp_off,
                    )
                else:
                    _finalize_env[
                        Self.DTYPE](
                        e, dt, dm, qpos_v3, qvel_v3, qacc_v3, joints_v3, M_v3, L_v3,
                        D_v3, fnet_v3, qacc_ws_v3, qacc_c_v3, trees_v3,
                        eulerdamp_off,
                    )
            comptime if _EULER_PROBE:
                var _e_now = Int(perf_counter_ns())
                _e_fin += _e_now - _e_last
                _e_last = _e_now
            comptime if _EULER_PROBE:
                print("[eprobe]", "fk", _e_fk, "bodyvel", _e_bodyvel, "subtree", _e_subtree, "cdof", _e_cdof, "crba", _e_crba, "armature", _e_armature, "ldlf", _e_ldlf, "minv", _e_minv, "rne", _e_rne, "fnet", _e_fnet, "ldls", _e_ldls, "wb", _e_wb, "coll", _e_coll, "newton", _e_newton, "warm", _e_warm, "fin", _e_fin)
        else:
            ctx.value().enqueue_function[
                _finalize_kernel[
                    Self.DTYPE, Self.D.NQ, Self.D.NV, Self.D.NJOINT, Self.BATCH
                ]
            ](
                dt,
                d.qpos.lt["gpu", L_QPOS](),
                d.qvel.lt["gpu", L_NV](),
                d.qacc.lt["gpu", L_NV](),
                m.joints.lt["gpu", L_JOINT](),
                self.scratch.M.lt["gpu", L_M](),
                self.scratch.L.lt["gpu", L_M](),
                self.scratch.D.lt["gpu", L_NV](),
                self.scratch.fnet.lt["gpu", L_NV](),
                self.scratch.qacc_ws.lt["gpu", L_NV](),
                self.scratch.qacc_constrained.lt["gpu", L_NV](),
                m.trees.lt["gpu", L_TREE](),
                Scalar[Self.DTYPE](1) if eulerdamp_off
                else Scalar[Self.DTYPE](0),
                grid_dim=(BLOCKS,),
                block_dim=(EU_TPB,),
            )

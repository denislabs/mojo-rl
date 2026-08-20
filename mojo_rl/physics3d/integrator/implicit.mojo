"""Stateful full-Implicit integrator over per-field tensors (Stage-I).

`ImplicitIntegrator` is the fields-native port of the legacy
`ImplicitIntegrator`: like Euler/ImplicitFast up through the mass matrix, but
it forms the FULL non-symmetric

    M_hat = M + armature - dt * qDeriv

where `qDeriv = d(qfrc_bias)/d(qvel)` includes BOTH the passive damping
diagonal (`qDeriv[i,i] = -damping[i]`) AND the dense RNE velocity derivative
(Coriolis/centrifugal, `qderiv`). Because `M_hat` is non-symmetric it
uses LU (`lu`), not LDL. Damping is also explicit in the force
(`fnet -= damping*qvel`), exactly like the legacy step — that is the standard
implicit linearization, not double counting.

Pipeline (contact-free path shown; the constraint seam mirrors euler):
    FK -> body vel -> subtree_com -> cdof -> CRBA -> +armature ->
    qDeriv (damping diag + RNE deriv) -> M_hat = M - dt*qDeriv ->
    LU factor -> M^-1 (for constraints) -> RNE bias -> fnet assembly ->
    LU solve -> qacc writeback -> [constraint seam] ->
    finalize (v += dt*qacc ; integrate qpos, quat-aware)

Unlike Euler/ImplicitFast there is NO post-constraint `dt*D` re-solve: the
implicit terms already live in `M_hat` (this matches the legacy CPU Implicit
step; the legacy GPU path re-used ImplicitFast's finalize, an asymmetry this
single-source port drops in favour of CPU==GPU consistency).

Reuses euler' tested per-stage helpers (armature / fnet / qacc
writeback); the M_hat-forming, damping-diagonal, and implicit-finalize
kernels are new here. Deliberately NOT ported yet (raise on use): fluid
forces (density/viscosity > 0)."""

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
from ..dynamics.rne import compute_bias_forces_rne
from ..dynamics.fluid_forces import compute_fluid_forces
from ..dynamics.lu import (
    lu_factor,
    lu_solve,
    compute_m_inv_from_lu,
)
from ..dynamics.qderiv import compute_rne_vel_derivative
from ..constraints.limits import solve_limits
from ..constraints.contact_solve import solve_contacts
from ..solver.newton_solve import solve_newton
from ..solver.je_budget import je_ws_size
from ..solver.cg_solve import solve_cg
from ..solver.island_pgs_solve import solve_island_pgs
from ..collision.broadphase_sap import detect_contacts_auto
from ..types import ConeType
from ..joint_types import JNT_FREE, JNT_BALL, JNT_HINGE, JNT_SLIDE
from ..fields import (
    AsStatic,
    Dims,
    Dims,
    Dims,
    DimsLike,
    DimsLike,
    DimsLike,
    Data,
    Model,
    DynamicsScratch,
    ContactScratch,
    ImplicitScratch,
    Dims,
    DimsLike,
    DYN2,
    rl2,
)
from .euler import (
    _armature_env,
    _armature_kernel,
    _fnet_passive_env,
    _fnet_passive_kernel,
    _qacc_writeback_env,
    _qacc_writeback_kernel,
)
from ..gpu.constants import (
    MODEL_JOINT_SIZE,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_DENSITY,
    MODEL_META_IDX_VISCOSITY,
    MODEL_META_IDX_NJOINT,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_DAMPING,
    METADATA_SIZE,
    META_IDX_ACTDAMP_LIVE,
)

comptime IM_TPB: Int = 64


# ── qDeriv damping diagonal init: zero, then qDeriv[i,i] = -damping[i] ─────
@always_inline
def _qderiv_damping_env[
    DTYPE: DType,
    D: DimsLike,
    L_JOINTS: Layout,
    L_ACTD: Layout,
    L_ACTDL: Layout,
    L_META: Layout,
    L_QDERIV: Layout](
    env: Int,
    dims: D,
    joints: LayoutTensor[
        DTYPE, L_JOINTS, MutAnyOrigin
    ],
    njoint: Int,
    actd: LayoutTensor[DTYPE, L_ACTD, MutAnyOrigin],
    actd_live: LayoutTensor[DTYPE, L_ACTDL, MutAnyOrigin],
    meta: LayoutTensor[DTYPE, L_META, MutAnyOrigin],
    qderiv: LayoutTensor[DTYPE, L_QDERIV, MutAnyOrigin],
):
    var nv = dims.get_nv()
    for i in range(nv * nv):
        qderiv[env, i] = 0
    for j in range(njoint):
        var jnt_type = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var dof_adr = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR]))
        var damp = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DAMPING])
        var nd = 1
        if jnt_type == JNT_FREE:
            nd = 6
        elif jnt_type == JNT_BALL:
            nd = 3
        for d in range(nd):
            var dof = dof_adr + d
            qderiv[env, dof * nv + dof] = -damp

    # ── d qfrc_actuator / d qvel — `mjd_actuator_vel` ────────────────────
    #
    # ⚠⚠ THE TERM WITHOUT WHICH THIS INTEGRATOR IS NOT IMPLICIT FOR SERVOS.
    # MuJoCo's `mjd_smooth_vel` is actuator + passive + (optional) RNE; only
    # the last two were here, so a model whose damping is ENTIRELY actuator
    # `kv` — spot's `dof_damping` is 0 — got an M_hat identical to M and an
    # integrator that was implicit in name only.
    #
    # `dof_actdamp` is `sum_a kv_a * trn_a^2`, the diagonal of
    # `J^T diag(kv) J`, banked at build time because for a JOINT transmission
    # it is constant in qpos and exact. Subtracted, because qDeriv holds
    # d(force)/d(vel) and the servo term is `-kv*vel`.
    # ⚠⚠ THE LIVE ARRAY WHEN IT WAS FILLED, THE MODEL'S WHEN IT WAS NOT.
    # MuJoCo's `mjd_actuator_vel` SKIPS an actuator whose force is clamped by
    # its `forcerange` — a saturated servo's force is pinned at the bound and
    # no longer depends on velocity — and whether it is saturated changes
    # every step, so no model-time array can carry it. `apply_actions_fields`
    # writes `d.dof_actdamp` and raises `META_IDX_ACTDAMP_LIVE`; a step taken
    # with no actuation call at all leaves the flag down, and then the
    # model-time value IS right because nothing can be saturated.
    #
    # Measured on rby1, whose 24 servos are `forcerange="-270 270"` and
    # saturate at qpos0: MuJoCo's `qDeriv` diagonal is -5 there (joint damping
    # alone) against our -405, while its two `<velocity>` wheels — no
    # forcerange, never clamped — read -4005 in both.
    var live = rebind[Scalar[DTYPE]](
        meta[env, META_IDX_ACTDAMP_LIVE]
    ) != Scalar[DTYPE](0)
    for i in range(nv):
        var a = (
            rebind[Scalar[DTYPE]](actd_live[env, i]) if live
            else rebind[Scalar[DTYPE]](actd[i, 0])
        )
        qderiv[env, i * nv + i] -= a


# ── M_hat: M -= dt * qDeriv (full, non-symmetric) ─────────────────────────
@always_inline
def _msub_qderiv_env[
    DTYPE: DType,
    D: DimsLike,
    L_M: Layout](
    env: Int,
    dt: Scalar[DTYPE],
    dims: D,
    M: LayoutTensor[DTYPE, L_M, MutAnyOrigin],
    qderiv: LayoutTensor[DTYPE, L_M, MutAnyOrigin],
):
    var nv = dims.get_nv()
    for i in range(nv * nv):
        var cur = rebind[Scalar[DTYPE]](M[env, i])
        var qd = rebind[Scalar[DTYPE]](qderiv[env, i])
        M[env, i] = cur - dt * qd


# ── rhs = M * qacc_constrained, and adopting the re-solved acceleration ───
@always_inline
def _mrhs_env[
    DTYPE: DType,
    D: DimsLike,
    L_M: Layout,
    L_NV: Layout](
    env: Int,
    dims: D,
    M: LayoutTensor[DTYPE, L_M, MutAnyOrigin],
    qacc_c: LayoutTensor[DTYPE, L_NV, MutAnyOrigin],
    fnet: LayoutTensor[DTYPE, L_NV, MutAnyOrigin],
):
    """`fnet = M * qacc_constrained` — the force MuJoCo re-solves against.

    ⚠⚠ THIS MUST RUN WHILE `M` IS STILL THE PLAIN MASS MATRIX, before
    `_msub_qderiv_env` turns it into M_hat in place. `qfrc_smooth +
    qfrc_constraint` is exactly `M * qacc_constrained`: the constraint solver
    reports an ACCELERATION, and multiplying it back by the same `M` it was
    solved against recovers the total force without needing the solver to
    hand out `qfrc_constraint` separately.
    """
    var nv = dims.get_nv()
    for i in range(nv):
        var acc = Scalar[DTYPE](0)
        for j in range(nv):
            acc += rebind[Scalar[DTYPE]](M[env, i * nv + j]) * rebind[
                Scalar[DTYPE]
            ](qacc_c[env, j])
        fnet[env, i] = acc


@always_inline
def _adopt_qacc_env[
    DTYPE: DType,
    D: DimsLike,
    L_NV: Layout](
    env: Int,
    dims: D,
    qacc_ws: LayoutTensor[DTYPE, L_NV, MutAnyOrigin],
    qacc_c: LayoutTensor[DTYPE, L_NV, MutAnyOrigin],
):
    """`qacc_constrained <- qacc_ws`, the M_hat re-solve's answer."""
    var nv = dims.get_nv()
    for i in range(nv):
        qacc_c[env, i] = rebind[Scalar[DTYPE]](qacc_ws[env, i])


# ── implicit finalize: v += dt*qacc ; integrate qpos (quat-aware) ─────────
@always_inline
def _implicit_finalize_env[
    DTYPE: DType,
    D: DimsLike,
    L_QPOS: Layout,
    L_QVEL: Layout,
    L_JOINTS: Layout](
    env: Int,
    dt: Scalar[DTYPE],
    dims: D,
    qpos: LayoutTensor[DTYPE, L_QPOS, MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, L_QVEL, MutAnyOrigin],
    qacc: LayoutTensor[DTYPE, L_QVEL, MutAnyOrigin],
    joints: LayoutTensor[
        DTYPE, L_JOINTS, MutAnyOrigin
    ],
    qacc_constrained: LayoutTensor[
        DTYPE, L_QVEL, MutAnyOrigin
    ],
):
    var nv = dims.get_nv()
    var njoint = dims.get_njoint()
    # Velocity update straight from the (constrained) implicit qacc — NO
    # dt*D re-solve (M_hat already carries the implicit terms).
    for i in range(nv):
        var qacc_final = rebind[Scalar[DTYPE]](qacc_constrained[env, i])
        qacc[env, i] = qacc_final
        var qvel_new = rebind[Scalar[DTYPE]](qvel[env, i]) + qacc_final * dt
        var qvel_max = Scalar[DTYPE](100.0)
        if qvel_new != qvel_new:  # NaN guard
            qvel_new = Scalar[DTYPE](0.0)
        elif qvel_new > qvel_max:
            qvel_new = qvel_max
        elif qvel_new < -qvel_max:
            qvel_new = -qvel_max
        qvel[env, i] = qvel_new

    # Position integration (verbatim from euler finalize).
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
            var qw = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr + 3])
            var qx = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr + 4])
            var qy = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr + 5])
            var qz = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr + 6])
            var wx = rebind[Scalar[DTYPE]](qvel[env, jnt_dof_adr + 3])
            var wy = rebind[Scalar[DTYPE]](qvel[env, jnt_dof_adr + 4])
            var wz = rebind[Scalar[DTYPE]](qvel[env, jnt_dof_adr + 5])
            var result = quat_integrate(qx, qy, qz, qw, wx, wy, wz, dt)
            var norm = quat_normalize(result[0], result[1], result[2], result[3])
            qpos[env, jnt_qpos_adr + 3] = norm[3]
            qpos[env, jnt_qpos_adr + 4] = norm[0]
            qpos[env, jnt_qpos_adr + 5] = norm[1]
            qpos[env, jnt_qpos_adr + 6] = norm[2]
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
            qpos[env, jnt_qpos_adr + 0] = bnorm[3]
            qpos[env, jnt_qpos_adr + 1] = bnorm[0]
            qpos[env, jnt_qpos_adr + 2] = bnorm[1]
            qpos[env, jnt_qpos_adr + 3] = bnorm[2]

        elif jnt_type == JNT_HINGE or jnt_type == JNT_SLIDE:
            var qp = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr])
            var qv = rebind[Scalar[DTYPE]](qvel[env, jnt_dof_adr])
            qpos[env, jnt_qpos_adr] = qp + qv * dt


# ── launchable kernels ────────────────────────────────────────────────────
def _qderiv_damping_kernel[
    DTYPE: DType, NV: Int, NJOINT: Int, BATCH: Int
](
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    njoint_arg: Int64,
    actd: LayoutTensor[DTYPE, Layout.row_major(NV, 1), MutAnyOrigin],
    actd_live: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin
    ],
    meta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, METADATA_SIZE), MutAnyOrigin
    ],
    qderiv: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
):
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var njoint = Int(njoint_arg)
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _qderiv_damping_env[DTYPE](
        env, Dims[nv=NV, njoint=NJOINT](), joints, njoint, actd, actd_live,
        meta, qderiv
    )


def _msub_qderiv_kernel[
    DTYPE: DType, NV: Int, BATCH: Int
](
    dt: Scalar[DTYPE],
    M: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    qderiv: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _msub_qderiv_env[DTYPE](env, dt, Dims[nv=NV](), M, qderiv)


def _mrhs_kernel[
    DTYPE: DType, NV: Int, BATCH: Int
](
    M: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    qacc_c: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    fnet: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _mrhs_env[DTYPE](env, Dims[nv=NV](), M, qacc_c, fnet)


def _adopt_qacc_kernel[
    DTYPE: DType, NV: Int, BATCH: Int
](
    qacc_ws: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    qacc_c: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _adopt_qacc_env[DTYPE](env, Dims[nv=NV](), qacc_ws, qacc_c)


def _implicit_finalize_kernel[
    DTYPE: DType, NQ: Int, NV: Int, NJOINT: Int, BATCH: Int
](
    dt: Scalar[DTYPE],
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    qacc: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    qacc_constrained: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin
    ],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _implicit_finalize_env[DTYPE](
        env, dt, Dims[nq=NQ, nv=NV, njoint=NJOINT](), qpos, qvel, qacc, joints, qacc_constrained
    )


# ── the stateful integrator ───────────────────────────────────────────────
struct ImplicitIntegrator[
    DTYPE: DType,
    D: DimsLike,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    BATCH: Int = 1,
    SOLVER: StaticString = "pgs",
    PARALLEL_GPU: Bool = False,
    CRBA_TREEWALK: Bool = False,
    SKIP_RNE_DERIV: Bool = False,
    # ⚠⚠ THIS DID NOT EXIST, AND ITS ABSENCE WAS SILENT. `solve_newton`
    # defaults it to 3 and `_contact_solve_env` CLAMPS each contact's own
    # condim down to it, so every model stepped by an implicit integrator
    # solved its contacts at condim 3 whatever the file declared — torsional
    # and rolling friction dropped, with no diagnostic. `EulerIntegrator` has
    # carried the parameter since the elliptic cone was generalised; this twin
    # was never given it. Measured on apptronik_apollo, whose soles are
    # `<pair condim="6">`: worst |d(qpos)| against MuJoCo after ONE step
    # 1.856e-03, against 5.551e-17 for the same model with its pairs edited
    # down to condim 3 — i.e. the entire divergence was the dropped rows.
    MAX_CONDIM: Int = 3,
](Movable):
    """Owns its scratch (dynamics + contact + implicit); steps full-implicit
    dynamics on either target. See module docstring for the algorithm and
    what is not yet ported. PARALLEL_GPU / CRBA_TREEWALK behave as in
    EulerIntegrator for the shared FK/CRBA/RNE stages; the LU + qDeriv
    stages are serial per-env."""

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

    var iscratch: ImplicitScratch[Self.DTYPE, Self.D, Self.BATCH]

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
        self.iscratch = ImplicitScratch[Self.DTYPE, Self.D, Self.BATCH](dims)

    def prepare_gpu(mut self, ctx: DeviceContext) raises:
        self.scratch.upload_all(ctx)
        self.cscratch.upload_all(ctx)
        self.iscratch.upload_all(ctx)

    def step[
        target: StaticString, CONTACTS: Bool = True
    ](
        mut self,
        mut d: Data[Self.DTYPE, Self.D, Self.BATCH],
        mut m: Model[Self.DTYPE, Self.D],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """One full implicit step."""
        var dt = m.meta.data[MODEL_META_IDX_TIMESTEP]
        var njoint = Int(m.meta.data[MODEL_META_IDX_NJOINT])

        # ── kinematics + composite inertia + mass matrix (as euler) ──────
        forward_kinematics[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](d, m, ctx)
        compute_body_velocities[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](d, m, ctx)
        compute_subtree_com[target, Self.DTYPE, BATCH=Self.BATCH](d, m, ctx)
        compute_cdof[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](d, m, self.scratch, ctx)
        compute_mass_matrix[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU, TREEWALK = Self.CRBA_TREEWALK](d, m, self.scratch, ctx)

        comptime L_JOINT = Layout.row_major(Self.D.NJOINT, MODEL_JOINT_SIZE)
        comptime L_ACTD = Layout.row_major(Self.D.NV, 1)
        comptime L_M = Layout.row_major(Self.BATCH, Self.D.NV * Self.D.NV)
        comptime L_NV = Layout.row_major(Self.BATCH, Self.D.NV)
        comptime L_QPOS = Layout.row_major(Self.BATCH, Self.D.NQ)
        comptime BLOCKS = (Self.BATCH + IM_TPB - 1) // IM_TPB

        # ── armature: M diag += armature ─────────────────────────────────
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
                block_dim=(IM_TPB,),
            )


        # ── LU factor the PLAIN M (+ M^-1 for the constraint solver) ────
        #
        # ⚠⚠ PLAIN M, NOT M_hat, AND THE ORDER IS THE WHOLE POINT. MuJoCo's
        # `mj_step` is `mj_forward` then `mj_implicit`: the constraint rows are
        # built AND SOLVED against the plain mass matrix, and only then does
        # the integrator form `M_hat = M - dt*qDeriv` and RE-SOLVE
        # `qacc = M_hat^-1 (qfrc_smooth + qfrc_constraint)`
        # (`engine_forward.c:1983` then `:2003`).
        #
        # This used to form M_hat first and hand `M_hat^-1` to the solver, so
        # every constraint row was solved against a mass matrix the reference
        # never uses. With no active rows the two orderings agree exactly —
        # which is why spot's implicitfast first step matched to 2.851622 —
        # and they diverge as soon as a row carries force. Measured on
        # sharpa_wave, whose 22 dof-friction rows are live from step 0: the
        # thumb's acceleration came out -1.86249 against MuJoCo's -1.56997,
        # and the one-dof algebra says exactly that: with R = 29.23 recovered
        # from MuJoCo's own efc_force, `a = a0*R/(K+R)` gives -1.902 at
        # K = 1/M and -1.8629 at K = 1/M_hat.
        lu_factor[target, Self.DTYPE, BATCH=Self.BATCH](self.scratch, ctx)
        compute_m_inv_from_lu[target, Self.DTYPE, BATCH=Self.BATCH](self.scratch, ctx)

        # ── RNE bias forces ──────────────────────────────────────────────
        compute_bias_forces_rne[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](d, m, self.scratch, ctx)

        # ── fnet = qfrc - bias - damping*qvel - stiffness - friction ─────
        comptime if target == "cpu":
            var dm = d.dims
            var rl_QPOS = rl2(Self.BATCH, dm.get_nq())
            var rl_NV = rl2(Self.BATCH, dm.get_nv())
            var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
            var qpos_v = d.qpos.lt_dyn["cpu", DYN2](rl_QPOS)
            var qvel_v = d.qvel.lt_dyn["cpu", DYN2](rl_NV)
            var qfrc_v = d.qfrc.lt_dyn["cpu", DYN2](rl_NV)
            var joints_v = m.joints.lt_dyn["cpu", DYN2](rl_JOINT)
            var bias_v = self.scratch.bias.lt_dyn["cpu", DYN2](rl_NV)
            var fnet_v = self.scratch.fnet.lt_dyn["cpu", DYN2](rl_NV)
            for e in range(Self.BATCH):
                _fnet_passive_env[
                    Self.DTYPE](e, dm, qpos_v, qvel_v, qfrc_v, joints_v, bias_v, fnet_v)
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
                block_dim=(IM_TPB,),
            )

        # Fluid drag into fnet (no-op unless meta density/viscosity > 0).
        compute_fluid_forces[target, Self.DTYPE, BATCH=Self.BATCH](d, m, self.scratch, ctx)

        # ── LU solve: qacc_ws = M^-1 fnet (the SMOOTH acceleration) ─────
        lu_solve[target, Self.DTYPE, BATCH=Self.BATCH](self.scratch, ctx)

        # ── qacc writeback: qacc + qacc_constrained = qacc_ws ────────────
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
                block_dim=(IM_TPB,),
            )

        # ── constraint seam (mirrors euler; uses M^-1 of the PLAIN M) ───
        comptime if CONTACTS:
            detect_contacts_auto[target, Self.DTYPE, BATCH=Self.BATCH](d, m, ctx)
            comptime assert (
                Self.SOLVER == "pgs"
                or Self.SOLVER == "newton"
                or Self.SOLVER == "cg"
                or Self.SOLVER == "island"
            ), (
                "ImplicitIntegrator: SOLVER must be 'pgs', 'newton',"
                " 'cg', or 'island'"
            )
            comptime if Self.SOLVER == "newton":
                solve_newton[target, Self.DTYPE, CONE_TYPE=Self.CONE_TYPE, BATCH=Self.BATCH, MAX_CONDIM=Self.MAX_CONDIM, JE_WS=Self.JE_WS](d, m, self.scratch, self.cscratch, ctx)
            else:
                comptime if Self.SOLVER == "cg":
                    solve_cg[target, Self.DTYPE, CONE_TYPE=Self.CONE_TYPE, BATCH=Self.BATCH](d, m, self.scratch, self.cscratch, ctx)
                else:
                    comptime if Self.SOLVER == "island":
                        solve_island_pgs[target, Self.DTYPE, CONE_TYPE=Self.CONE_TYPE, BATCH=Self.BATCH](d, m, self.scratch, self.cscratch, ctx)
                    else:
                        # ⚠ `solve_cg`, `solve_island_pgs` and `solve_contacts`
                        # DO NOT TAKE `MAX_CONDIM` — they have no such
                        # parameter, and `solve_contacts` calls
                        # `_contact_solve_env` (which does) without one. So
                        # those three are condim-3-only, on BOTH integrators;
                        # `EulerIntegrator` forwards the parameter to
                        # `solve_newton` alone for the same reason. The studio
                        # only ever builds `newton`, which is why this is the
                        # call that had to change first.
                        solve_contacts[target, Self.DTYPE, CONE_TYPE=Self.CONE_TYPE, BATCH=Self.BATCH](d, m, self.scratch, self.cscratch, ctx)
        else:
            solve_limits[target, Self.DTYPE, BATCH=Self.BATCH](d, m, self.scratch, ctx)

        # ── the implicit re-solve: qacc = M_hat^-1 * (M * qacc_constrained)
        #
        # `M * qacc_constrained` IS `qfrc_smooth + qfrc_constraint` — the
        # solver hands back an acceleration, and multiplying by the same M it
        # was solved against recovers the force MuJoCo re-solves with. It has
        # to happen while `M` is still plain, which is why it comes before the
        # qDeriv block below rather than after it.
        comptime if target == "cpu":
            var dm_r = d.dims
            var rl_M_r = rl2(Self.BATCH, dm_r.get_nv() * dm_r.get_nv())
            var rl_NV_r = rl2(Self.BATCH, dm_r.get_nv())
            var M_r = self.scratch.M.lt_dyn["cpu", DYN2](rl_M_r)
            var qc_r = self.scratch.qacc_constrained.lt_dyn["cpu", DYN2](rl_NV_r)
            var fnet_r = self.scratch.fnet.lt_dyn["cpu", DYN2](rl_NV_r)
            for e in range(Self.BATCH):
                _mrhs_env[Self.DTYPE](e, dm_r, M_r, qc_r, fnet_r)
        else:
            ctx.value().enqueue_function[
                _mrhs_kernel[Self.DTYPE, Self.D.NV, Self.BATCH]
            ](
                self.scratch.M.lt["gpu", L_M](),
                self.scratch.qacc_constrained.lt["gpu", L_NV](),
                self.scratch.fnet.lt["gpu", L_NV](),
                grid_dim=(BLOCKS,),
                block_dim=(IM_TPB,),
            )

        # ── qDeriv = damping diagonal, then subtract RNE velocity deriv ──
        comptime if target == "cpu":
            var dm = d.dims
            var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
            var rl_M = rl2(Self.BATCH, dm.get_nv() * dm.get_nv())
            var joints_v = m.joints.lt_dyn["cpu", DYN2](rl_JOINT)
            var qd_v = self.iscratch.qderiv.lt_dyn["cpu", DYN2](rl_M)
            var rl_ACTD = rl2(dm.get_nv(), 1)
            var actd_v = m.dof_actdamp.lt_dyn["cpu", DYN2](rl_ACTD)
            var rl_ACTDL = rl2(Self.BATCH, dm.get_nv())
            var actdl_v = d.dof_actdamp.lt_dyn["cpu", DYN2](rl_ACTDL)
            var rl_META = rl2(Self.BATCH, METADATA_SIZE)
            var meta_v = d.meta.lt_dyn["cpu", DYN2](rl_META)
            for e in range(Self.BATCH):
                _qderiv_damping_env[
                    Self.DTYPE](e, dm, joints_v, njoint, actd_v, actdl_v,
                                meta_v, qd_v)
        else:
            ctx.value().enqueue_function[
                _qderiv_damping_kernel[
                    Self.DTYPE, Self.D.NV, Self.D.NJOINT, Self.BATCH
                ]
            ](
                m.joints.lt["gpu", L_JOINT](),
                Int64(njoint),
                m.dof_actdamp.lt["gpu", L_ACTD](),
                d.dof_actdamp.lt["gpu", Layout.row_major(
                    Self.BATCH, Self.D.NV)](),
                d.meta.lt["gpu", Layout.row_major(
                    Self.BATCH, METADATA_SIZE)](),
                self.iscratch.qderiv.lt["gpu", L_M](),
                grid_dim=(BLOCKS,),
                block_dim=(IM_TPB,),
            )
        # ⚠⚠ THIS ONE FLAG IS THE DIFFERENCE BETWEEN `implicit` AND
        # `implicitfast`, and it is MuJoCo's own: `mj_implicitSkip` calls
        # `mjd_smooth_vel(m, d, flg_bias)` with 1 for `implicit` and 0 for
        # `implicitfast` (`engine_forward.c:1794` vs `:1806`). `flg_bias`
        # gates exactly this term — the dense RNE velocity derivative
        # (Coriolis/centrifugal).
        #
        # Skipping it is not only cheaper. Without it qDeriv is SYMMETRIC, so
        # MuJoCo factorises `implicitfast` with its ordinary Cholesky and
        # keeps LU for `implicit`. We use LU for both: correct either way, and
        # a second factorisation path is a second thing to keep in step for a
        # speed difference the studio does not need. Noted so the choice reads
        # as a decision rather than an oversight.
        comptime if not Self.SKIP_RNE_DERIV:
            compute_rne_vel_derivative[target, Self.DTYPE, Self.BATCH](d, m, self.scratch, self.iscratch, ctx)

        # ── M_hat = M - dt * qDeriv ──────────────────────────────────────
        comptime if target == "cpu":
            var dm = d.dims
            var rl_M = rl2(Self.BATCH, dm.get_nv() * dm.get_nv())
            var M_v = self.scratch.M.lt_dyn["cpu", DYN2](rl_M)
            var qd_v = self.iscratch.qderiv.lt_dyn["cpu", DYN2](rl_M)
            for e in range(Self.BATCH):
                _msub_qderiv_env[Self.DTYPE](
                    e, dt, dm, M_v, qd_v
                )
        else:
            ctx.value().enqueue_function[
                _msub_qderiv_kernel[Self.DTYPE, Self.D.NV, Self.BATCH]
            ](
                dt,
                self.scratch.M.lt["gpu", L_M](),
                self.iscratch.qderiv.lt["gpu", L_M](),
                grid_dim=(BLOCKS,),
                block_dim=(IM_TPB,),
            )

        # M_hat is formed; factor it and re-solve. `compute_m_inv_from_lu` is
        # deliberately NOT re-run: nothing downstream of here reads `m_inv`,
        # and M_hat^-1 is not what the constraint rows were solved against.
        lu_factor[target, Self.DTYPE, BATCH=Self.BATCH](self.scratch, ctx)
        lu_solve[target, Self.DTYPE, BATCH=Self.BATCH](self.scratch, ctx)

        comptime if target == "cpu":
            var dm_a = d.dims
            var rl_NV_a = rl2(Self.BATCH, dm_a.get_nv())
            var qws_a = self.scratch.qacc_ws.lt_dyn["cpu", DYN2](rl_NV_a)
            var qc_a = self.scratch.qacc_constrained.lt_dyn["cpu", DYN2](rl_NV_a)
            for e in range(Self.BATCH):
                _adopt_qacc_env[Self.DTYPE](e, dm_a, qws_a, qc_a)
        else:
            ctx.value().enqueue_function[
                _adopt_qacc_kernel[Self.DTYPE, Self.D.NV, Self.BATCH]
            ](
                self.scratch.qacc_ws.lt["gpu", L_NV](),
                self.scratch.qacc_constrained.lt["gpu", L_NV](),
                grid_dim=(BLOCKS,),
                block_dim=(IM_TPB,),
            )

        # ── implicit finalize: v += dt*qacc ; integrate qpos ─────────────
        comptime if target == "cpu":
            var dm = d.dims
            var rl_QPOS = rl2(Self.BATCH, dm.get_nq())
            var rl_NV = rl2(Self.BATCH, dm.get_nv())
            var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
            var qpos_v = d.qpos.lt_dyn["cpu", DYN2](rl_QPOS)
            var qvel_v = d.qvel.lt_dyn["cpu", DYN2](rl_NV)
            var qacc_v = d.qacc.lt_dyn["cpu", DYN2](rl_NV)
            var joints_v = m.joints.lt_dyn["cpu", DYN2](rl_JOINT)
            var qacc_c_v = self.scratch.qacc_constrained.lt_dyn["cpu", DYN2](rl_NV)
            for e in range(Self.BATCH):
                _implicit_finalize_env[
                    Self.DTYPE](e, dt, dm, qpos_v, qvel_v, qacc_v, joints_v, qacc_c_v)
        else:
            ctx.value().enqueue_function[
                _implicit_finalize_kernel[
                    Self.DTYPE, Self.D.NQ, Self.D.NV, Self.D.NJOINT, Self.BATCH
                ]
            ](
                dt,
                d.qpos.lt["gpu", L_QPOS](),
                d.qvel.lt["gpu", L_NV](),
                d.qacc.lt["gpu", L_NV](),
                m.joints.lt["gpu", L_JOINT](),
                self.scratch.qacc_constrained.lt["gpu", L_NV](),
                grid_dim=(BLOCKS,),
                block_dim=(IM_TPB,),
            )

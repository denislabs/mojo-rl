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
    Data,
    Model,
    DynamicsScratch,
    ContactScratch,
    ImplicitScratch,
    Dims,
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
)

comptime IM_TPB: Int = 64


# ── qDeriv damping diagonal init: zero, then qDeriv[i,i] = -damping[i] ─────
@always_inline
def _qderiv_damping_env[
    DTYPE: DType, NV: Int, NJOINT: Int, BATCH: Int
](
    env: Int,
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    njoint: Int,
    qderiv: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
):
    for i in range(NV * NV):
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
            qderiv[env, dof * NV + dof] = -damp


# ── M_hat: M -= dt * qDeriv (full, non-symmetric) ─────────────────────────
@always_inline
def _msub_qderiv_env[
    DTYPE: DType, NV: Int, BATCH: Int
](
    env: Int,
    dt: Scalar[DTYPE],
    M: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
    qderiv: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
):
    for i in range(NV * NV):
        var cur = rebind[Scalar[DTYPE]](M[env, i])
        var qd = rebind[Scalar[DTYPE]](qderiv[env, i])
        M[env, i] = cur - dt * qd


# ── implicit finalize: v += dt*qacc ; integrate qpos (quat-aware) ─────────
@always_inline
def _implicit_finalize_env[
    DTYPE: DType, NQ: Int, NV: Int, NJOINT: Int, BATCH: Int
](
    env: Int,
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
    # Velocity update straight from the (constrained) implicit qacc — NO
    # dt*D re-solve (M_hat already carries the implicit terms).
    for i in range(NV):
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
    for j in range(NJOINT):
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
    qderiv: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * NV), MutAnyOrigin],
):
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var njoint = Int(njoint_arg)
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _qderiv_damping_env[DTYPE, NV, NJOINT, BATCH](
        env, joints, njoint, qderiv
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
    _msub_qderiv_env[DTYPE, NV, BATCH](env, dt, M, qderiv)


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
    _implicit_finalize_env[DTYPE, NQ, NV, NJOINT, BATCH](
        env, dt, qpos, qvel, qacc, joints, qacc_constrained
    )


# ── the stateful integrator ───────────────────────────────────────────────
struct ImplicitIntegrator[
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
    SOLVER: StaticString = "pgs",
    PARALLEL_GPU: Bool = False,
    CRBA_TREEWALK: Bool = False,
    # ⚠ APPENDED, NOT GROUPED WITH `NEXCLUDE`. These are all `Int`, so a
    # mid-list insertion silently shifts every positional instantiation
    # in `mojo_rl/envs/` — see the same note on `fields.Model`.
    NPAIR: Int = 0,
](Movable):
    """Owns its scratch (dynamics + contact + implicit); steps full-implicit
    dynamics on either target. See module docstring for the algorithm and
    what is not yet ported. PARALLEL_GPU / CRBA_TREEWALK behave as in
    EulerIntegrator for the shared FK/CRBA/RNE stages; the LU + qDeriv
    stages are serial per-env."""

    var scratch: DynamicsScratch[Self.DTYPE, Dims[nv=Self.NV, nbody=Self.NBODY], Self.BATCH]
    # Blocked-Newton Jacobian spill size — 0 unless `Je` overflows threadgroup
    # memory. Computed HERE (not by the caller) because this struct already
    # carries every dimension it depends on, and via `je_budget` so the buffer
    # and the kernel that indexes it cannot drift apart.
    comptime JE_WS = je_ws_size[
        Self.DTYPE, Self.NV, Self.NJOINT, Self.NTENDON, Self.NEQUALITY,
        Self.MAX_CONTACTS, 3,
    ]()

    var cscratch: ContactScratch[Self.DTYPE, Dims[nv=Self.NV, max_contacts=Self.MAX_CONTACTS], Self.BATCH, Self.JE_WS]
    # Scaffolding, same shape as `RK4Integrator.D` — one alias, both spellings
    # below. It disappears when this integrator takes a real `D` in the sweep.
    comptime D = Dims[nv = Self.NV, nbody = Self.NBODY]

    var iscratch: ImplicitScratch[Self.DTYPE, Self.D, Self.BATCH]

    def __init__(out self) raises:
        self.scratch = DynamicsScratch[Self.DTYPE, Dims[nv=Self.NV, nbody=Self.NBODY], Self.BATCH]()
        self.cscratch = ContactScratch[Self.DTYPE, Dims[nv=Self.NV, max_contacts=Self.MAX_CONTACTS], Self.BATCH, Self.JE_WS]()
        self.iscratch = ImplicitScratch[Self.DTYPE, Self.D, Self.BATCH]()

    def prepare_gpu(mut self, ctx: DeviceContext) raises:
        self.scratch.upload_all(ctx)
        self.cscratch.upload_all(ctx)
        self.iscratch.upload_all(ctx)

    def step[
        target: StaticString, CONTACTS: Bool = True
    ](
        mut self,
        mut d: Data[
            Self.DTYPE, Self.NQ, Self.NV, Self.NBODY, Self.MAX_CONTACTS,
            Self.NSITE, Self.BATCH,
        ],
        mut m: Model[
            Self.DTYPE, Self.NV, Self.NBODY, Self.NJOINT, Self.NGEOM,
            Self.NEQUALITY, Self.NTENDON, Self.NSITE, Self.NEXCLUDE,
            Self.NMESH_VERTS,
            Self.NPAIR,
        ],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """One full implicit step."""
        var dt = m.meta.data[MODEL_META_IDX_TIMESTEP]
        var njoint = Int(m.meta.data[MODEL_META_IDX_NJOINT])

        # ── kinematics + composite inertia + mass matrix (as euler) ──────
        forward_kinematics[
            target, Self.DTYPE, Self.NQ, Self.NV, Self.NBODY, Self.NJOINT,
            Self.MAX_CONTACTS, Self.NGEOM, Self.NEQUALITY, Self.NTENDON,
            Self.NSITE, Self.NEXCLUDE, Self.NMESH_VERTS, Self.BATCH,
            PARALLEL = Self.PARALLEL_GPU,
        ](d, m, ctx)
        compute_body_velocities[
            target, Self.DTYPE, Self.NQ, Self.NV, Self.NBODY, Self.NJOINT,
            Self.MAX_CONTACTS, Self.NGEOM, Self.NEQUALITY, Self.NTENDON,
            Self.NSITE, Self.NEXCLUDE, Self.NMESH_VERTS, Self.BATCH,
            PARALLEL = Self.PARALLEL_GPU,
        ](d, m, ctx)
        compute_subtree_com[
            target, Self.DTYPE, Self.NQ, Self.NV, Self.NBODY, Self.NJOINT,
            Self.MAX_CONTACTS, Self.NGEOM, Self.NEQUALITY, Self.NTENDON,
            Self.NSITE, Self.NEXCLUDE, Self.NMESH_VERTS, Self.BATCH,
        ](d, m, ctx)
        compute_cdof[
            target, Self.DTYPE, Self.NQ, Self.NV, Self.NBODY, Self.NJOINT,
            Self.MAX_CONTACTS, Self.NGEOM, Self.NEQUALITY, Self.NTENDON,
            Self.NSITE, Self.NEXCLUDE, Self.NMESH_VERTS, Self.BATCH,
            PARALLEL = Self.PARALLEL_GPU,
        ](d, m, self.scratch, ctx)
        compute_mass_matrix[
            target, Self.DTYPE, Self.NQ, Self.NV, Self.NBODY, Self.NJOINT,
            Self.MAX_CONTACTS, Self.NGEOM, Self.NEQUALITY, Self.NTENDON,
            Self.NSITE, Self.NEXCLUDE, Self.NMESH_VERTS, Self.BATCH,
            PARALLEL = Self.PARALLEL_GPU,
            TREEWALK = Self.CRBA_TREEWALK,
        ](d, m, self.scratch, ctx)

        comptime L_JOINT = Layout.row_major(Self.NJOINT, MODEL_JOINT_SIZE)
        comptime L_M = Layout.row_major(Self.BATCH, Self.NV * Self.NV)
        comptime L_NV = Layout.row_major(Self.BATCH, Self.NV)
        comptime L_QPOS = Layout.row_major(Self.BATCH, Self.NQ)
        comptime BLOCKS = (Self.BATCH + IM_TPB - 1) // IM_TPB

        # ── armature: M diag += armature ─────────────────────────────────
        comptime if target == "cpu":
            var joints_v = m.joints.lt["cpu", L_JOINT]()
            var M_v = self.scratch.M.lt["cpu", L_M]()
            for e in range(Self.BATCH):
                _armature_env[
                    Self.DTYPE, Self.NV, Self.NJOINT, Self.BATCH
                ](e, joints_v, M_v)
        else:
            ctx.value().enqueue_function[
                _armature_kernel[Self.DTYPE, Self.NV, Self.NJOINT, Self.BATCH]
            ](
                m.joints.lt["gpu", L_JOINT](),
                self.scratch.M.lt["gpu", L_M](),
                grid_dim=(BLOCKS,),
                block_dim=(IM_TPB,),
            )

        # ── qDeriv = damping diagonal, then subtract RNE velocity deriv ──
        comptime if target == "cpu":
            var joints_v = m.joints.lt["cpu", L_JOINT]()
            var qd_v = self.iscratch.qderiv.lt["cpu", L_M]()
            for e in range(Self.BATCH):
                _qderiv_damping_env[
                    Self.DTYPE, Self.NV, Self.NJOINT, Self.BATCH
                ](e, joints_v, njoint, qd_v)
        else:
            ctx.value().enqueue_function[
                _qderiv_damping_kernel[
                    Self.DTYPE, Self.NV, Self.NJOINT, Self.BATCH
                ]
            ](
                m.joints.lt["gpu", L_JOINT](),
                Int64(njoint),
                self.iscratch.qderiv.lt["gpu", L_M](),
                grid_dim=(BLOCKS,),
                block_dim=(IM_TPB,),
            )
        compute_rne_vel_derivative[
            target, Self.DTYPE, Self.NQ, Self.NV, Self.NBODY, Self.NJOINT,
            Self.MAX_CONTACTS, Self.NGEOM, Self.NEQUALITY, Self.NTENDON,
            Self.NSITE, Self.NEXCLUDE, Self.NMESH_VERTS, Self.BATCH,
        ](d, m, self.scratch, self.iscratch, ctx)

        # ── M_hat = M - dt * qDeriv ──────────────────────────────────────
        comptime if target == "cpu":
            var M_v = self.scratch.M.lt["cpu", L_M]()
            var qd_v = self.iscratch.qderiv.lt["cpu", L_M]()
            for e in range(Self.BATCH):
                _msub_qderiv_env[Self.DTYPE, Self.NV, Self.BATCH](
                    e, dt, M_v, qd_v
                )
        else:
            ctx.value().enqueue_function[
                _msub_qderiv_kernel[Self.DTYPE, Self.NV, Self.BATCH]
            ](
                dt,
                self.scratch.M.lt["gpu", L_M](),
                self.iscratch.qderiv.lt["gpu", L_M](),
                grid_dim=(BLOCKS,),
                block_dim=(IM_TPB,),
            )

        # ── LU factor M_hat (+ M^-1 for the constraint solver) ───────────
        lu_factor[
            target, Self.DTYPE, Self.NV, Self.NBODY, Self.BATCH
        ](self.scratch, ctx)
        compute_m_inv_from_lu[
            target, Self.DTYPE, Self.NV, Self.NBODY, Self.BATCH
        ](self.scratch, ctx)

        # ── RNE bias forces ──────────────────────────────────────────────
        compute_bias_forces_rne[
            target, Self.DTYPE, Self.NQ, Self.NV, Self.NBODY, Self.NJOINT,
            Self.MAX_CONTACTS, Self.NGEOM, Self.NEQUALITY, Self.NTENDON,
            Self.NSITE, Self.NEXCLUDE, Self.NMESH_VERTS, Self.BATCH,
            PARALLEL = Self.PARALLEL_GPU,
        ](d, m, self.scratch, ctx)

        # ── fnet = qfrc - bias - damping*qvel - stiffness - friction ─────
        comptime if target == "cpu":
            var qpos_v = d.qpos.lt["cpu", L_QPOS]()
            var qvel_v = d.qvel.lt["cpu", L_NV]()
            var qfrc_v = d.qfrc.lt["cpu", L_NV]()
            var joints_v = m.joints.lt["cpu", L_JOINT]()
            var bias_v = self.scratch.bias.lt["cpu", L_NV]()
            var fnet_v = self.scratch.fnet.lt["cpu", L_NV]()
            for e in range(Self.BATCH):
                _fnet_passive_env[
                    Self.DTYPE, Self.NQ, Self.NV, Self.NJOINT, Self.BATCH
                ](e, qpos_v, qvel_v, qfrc_v, joints_v, bias_v, fnet_v)
        else:
            ctx.value().enqueue_function[
                _fnet_passive_kernel[
                    Self.DTYPE, Self.NQ, Self.NV, Self.NJOINT, Self.BATCH
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
        compute_fluid_forces[
            target, Self.DTYPE, Self.NQ, Self.NV, Self.NBODY, Self.NJOINT,
            Self.MAX_CONTACTS, Self.NGEOM, Self.NEQUALITY, Self.NTENDON,
            Self.NSITE, Self.NEXCLUDE, Self.NMESH_VERTS, Self.BATCH,
        ](d, m, self.scratch, ctx)

        # ── LU solve: qacc_ws = M_hat^-1 fnet ────────────────────────────
        lu_solve[
            target, Self.DTYPE, Self.NV, Self.NBODY, Self.BATCH
        ](self.scratch, ctx)

        # ── qacc writeback: qacc + qacc_constrained = qacc_ws ────────────
        comptime if target == "cpu":
            var qacc_ws_v = self.scratch.qacc_ws.lt["cpu", L_NV]()
            var qacc_v = d.qacc.lt["cpu", L_NV]()
            var qacc_c_v = self.scratch.qacc_constrained.lt["cpu", L_NV]()
            for e in range(Self.BATCH):
                _qacc_writeback_env[Self.DTYPE, Self.NV, Self.BATCH](
                    e, qacc_ws_v, qacc_v, qacc_c_v
                )
        else:
            ctx.value().enqueue_function[
                _qacc_writeback_kernel[Self.DTYPE, Self.NV, Self.BATCH]
            ](
                self.scratch.qacc_ws.lt["gpu", L_NV](),
                d.qacc.lt["gpu", L_NV](),
                self.scratch.qacc_constrained.lt["gpu", L_NV](),
                grid_dim=(BLOCKS,),
                block_dim=(IM_TPB,),
            )

        # ── constraint seam (mirrors euler; uses M^-1 of M_hat) ──────────
        comptime if CONTACTS:
            detect_contacts_auto[
                target, Self.DTYPE, Self.NQ, Self.NV, Self.NBODY,
                Self.NJOINT, Self.MAX_CONTACTS, Self.NGEOM, Self.NEQUALITY,
                Self.NTENDON, Self.NSITE, Self.NEXCLUDE, Self.NMESH_VERTS,
                Self.BATCH,
            ](d, m, ctx)
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
                solve_newton[
                    target, Self.DTYPE, Self.NQ, Self.NV, Self.NBODY,
                    Self.NJOINT, Self.MAX_CONTACTS, Self.NGEOM,
                    Self.NEQUALITY, Self.NTENDON, Self.NSITE, Self.NEXCLUDE,
                    Self.NMESH_VERTS, Self.CONE_TYPE, Self.BATCH,
                ](d, m, self.scratch, self.cscratch, ctx)
            else:
                comptime if Self.SOLVER == "cg":
                    solve_cg[
                        target, Self.DTYPE, Self.NQ, Self.NV, Self.NBODY,
                        Self.NJOINT, Self.MAX_CONTACTS, Self.NGEOM,
                        Self.NEQUALITY, Self.NTENDON, Self.NSITE,
                        Self.NEXCLUDE, Self.NMESH_VERTS, Self.CONE_TYPE,
                        Self.BATCH,
                    ](d, m, self.scratch, self.cscratch, ctx)
                else:
                    comptime if Self.SOLVER == "island":
                        solve_island_pgs[
                            target, Self.DTYPE, Self.NQ, Self.NV, Self.NBODY,
                            Self.NJOINT, Self.MAX_CONTACTS, Self.NGEOM,
                            Self.NEQUALITY, Self.NTENDON, Self.NSITE,
                            Self.NEXCLUDE, Self.NMESH_VERTS, Self.CONE_TYPE,
                            Self.BATCH,
                        ](d, m, self.scratch, self.cscratch, ctx)
                    else:
                        solve_contacts[
                            target, Self.DTYPE, Self.NQ, Self.NV, Self.NBODY,
                            Self.NJOINT, Self.MAX_CONTACTS, Self.NGEOM,
                            Self.NEQUALITY, Self.NTENDON, Self.NSITE,
                            Self.NEXCLUDE, Self.NMESH_VERTS, Self.CONE_TYPE,
                            Self.BATCH,
                        ](d, m, self.scratch, self.cscratch, ctx)
        else:
            solve_limits[
                target, Self.DTYPE, Self.NQ, Self.NV, Self.NBODY,
                Self.NJOINT, Self.MAX_CONTACTS, Self.NGEOM, Self.NEQUALITY,
                Self.NTENDON, Self.NSITE, Self.NEXCLUDE, Self.NMESH_VERTS,
                Self.BATCH,
            ](d, m, self.scratch, ctx)

        # ── implicit finalize: v += dt*qacc ; integrate qpos ─────────────
        comptime if target == "cpu":
            var qpos_v = d.qpos.lt["cpu", L_QPOS]()
            var qvel_v = d.qvel.lt["cpu", L_NV]()
            var qacc_v = d.qacc.lt["cpu", L_NV]()
            var joints_v = m.joints.lt["cpu", L_JOINT]()
            var qacc_c_v = self.scratch.qacc_constrained.lt["cpu", L_NV]()
            for e in range(Self.BATCH):
                _implicit_finalize_env[
                    Self.DTYPE, Self.NQ, Self.NV, Self.NJOINT, Self.BATCH
                ](e, dt, qpos_v, qvel_v, qacc_v, joints_v, qacc_c_v)
        else:
            ctx.value().enqueue_function[
                _implicit_finalize_kernel[
                    Self.DTYPE, Self.NQ, Self.NV, Self.NJOINT, Self.BATCH
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

"""Stateful RK4 integrator over per-field tensors (migration P2).

`RK4IntegratorFields` is the stateful replacement for the stateless
`RK4Integrator.step_gpu` + caller-provided workspace slab: the struct OWNS
its `DynamicsScratch` + `Rk4Scratch` and sequences the single-source
per-stage functions into a full contact-free RK4 step (MuJoCo
mj_RungeKutta tableau, exactly as the legacy `rk4_stage_kernel` /
`rk4_combine_kernel` compute it):

    save (q0, v0)                                     [stage-0 setup]
    for stage s in 0..3:
        s==1: A[0]=qacc_c; q = pos(q0, v0,  dt/2); v = v0 + dt/2*A[0]
        s==2: A[1]=qacc_c; C1 = v0 + dt/2*A[0];
              q = pos(q0, C1, dt/2); v = v0 + dt/2*A[1]
        s==3: A[2]=qacc_c; C2 = v0 + dt/2*A[1];
              q = pos(q0, C2, dt);   v = v0 + dt*A[2]
        dynamics chain: FK -> body velocities -> subtree_com -> cdof ->
            CRBA -> +armature (NO implicit damping for RK4) -> LDL factor
            (+ M_inv) -> RNE -> fnet (qfrc - bias - damping - stiffness -
            frictionloss, all EXPLICIT) -> LDL solve -> qacc writeback
            (d.qacc + qacc_constrained)
    combine: qacc = (A[0] + 2*A[1] + 2*A[2] + A[3])/6 with A[3] read from
        qacc_constrained; C[3] = v0 + dt*A[2];
        v_comb = (v0 + 2*C1 + 2*C2 + C[3])/6 (NaN guard + clamp, stored in
        the A0 slot exactly like legacy); qvel = v0 + qacc*dt (NaN guard +
        clamp); qpos = pos(q0, v_comb, dt) quaternion-aware.

Staging/combine arithmetic is verbatim from the legacy `rk4_stage_kernel`
(:1232) / `rk4_combine_kernel` (:2140) / `_integrate_pos_gpu` (:151);
armature/fnet/writeback env bodies are shared with `euler_fields` (the
legacy RK4 stage steps 6b/9/9b are byte-identical to the Euler ones —
verified against both sources). Unlike Euler's finalize, RK4 does NO
implicit-damping re-solve: damping is explicit in fnet, and the combine
integrates directly.

Deliberately NOT yet ported (raise / absent by design):
- fluid forces (density/viscosity > 0) — raise on use, like Euler fields.
- contacts AND limits: legacy RK4 handles both inside the per-stage
  constraint-SOLVER launch (`step_gpu` = 4 x (stage + solver) + combine);
  the contact-free legacy path (stage kernels only, no solver launches)
  never touches limits — so unlike `EulerIntegratorFields` there is no
  CONTACTS param and no standalone limits stage here. Contacts-in-RK4
  (solver seam per stage) is a later slice.
"""

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from ..kinematics.quat_math import quat_integrate, quat_normalize
from ..kinematics.forward_kinematics_fields import (
    forward_kinematics_fields,
    compute_body_velocities_fields,
)
from ..dynamics.subtree_com_fields import compute_subtree_com_fields
from ..dynamics.cdof_fields import compute_cdof_fields
from ..dynamics.mass_matrix_fields import compute_mass_matrix_fields
from ..dynamics.ldl_fields import (
    ldl_factor_fields,
    ldl_solve_fields,
    compute_m_inv_fields,
)
from ..types import ConeType
from ..dynamics.rne_fields import compute_bias_forces_rne_fields
from ..constraints.contact_solve_fields import solve_contacts_fields
from ..solver.newton_solve_fields import solve_newton_fields
from ..solver.cg_solve_fields import solve_cg_fields
from ..solver.island_pgs_solve_fields import solve_island_pgs_fields
from ..collision.broadphase_sap_fields import detect_contacts_auto_fields
from ..joint_types import JNT_FREE, JNT_BALL, JNT_HINGE, JNT_SLIDE
from ..fields import DataFields, ModelFields, DynamicsScratch, ContactScratch, Rk4Scratch
from ..gpu.constants import (
    MODEL_JOINT_SIZE,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_DENSITY,
    MODEL_META_IDX_VISCOSITY,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
)
from .euler_fields import (
    EU_TPB,
    _armature_env_fields,
    _armature_kernel,
    _fnet_passive_env_fields,
    _fnet_passive_kernel,
    _qacc_writeback_env_fields,
    _qacc_writeback_kernel,
)


# ── position integration: qpos = q0 + vel * dt, quaternion-aware ──────────
# (verbatim legacy _integrate_pos_gpu :151; num_joints -> comptime NJOINT,
#  ws rk4 regions -> per-field q0/vel tensors, state qpos -> qpos tensor)
@always_inline
def _rk4_integrate_pos_env_fields[
    DTYPE: DType, NQ: Int, NV: Int, NJOINT: Int, BATCH: Int
](
    env: Int,
    dt: Scalar[DTYPE],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    q0: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    vel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
):
    for j in range(NJOINT):
        var jnt_type = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var qpos_adr = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS_ADR])
        )
        var dof_adr = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR]))

        if jnt_type == JNT_FREE:
            # Position: simple addition
            for d in range(3):
                var q0_d = rebind[Scalar[DTYPE]](q0[env, qpos_adr + d])
                var v_d = rebind[Scalar[DTYPE]](vel[env, dof_adr + d])
                qpos[env, qpos_adr + d] = q0_d + v_d * dt
            # Quaternion: exponential map integration.
            # MuJoCo qpos layout: [tx, ty, tz, qw, qx, qy, qz]
            # Our internal convention: (x, y, z, w)
            var qw = rebind[Scalar[DTYPE]](q0[env, qpos_adr + 3])
            var qx = rebind[Scalar[DTYPE]](q0[env, qpos_adr + 4])
            var qy = rebind[Scalar[DTYPE]](q0[env, qpos_adr + 5])
            var qz = rebind[Scalar[DTYPE]](q0[env, qpos_adr + 6])
            var wx = rebind[Scalar[DTYPE]](vel[env, dof_adr + 3])
            var wy = rebind[Scalar[DTYPE]](vel[env, dof_adr + 4])
            var wz = rebind[Scalar[DTYPE]](vel[env, dof_adr + 5])
            var result = quat_integrate(qx, qy, qz, qw, wx, wy, wz, dt)
            var norm = quat_normalize(
                result[0], result[1], result[2], result[3]
            )
            # Write back in MuJoCo qpos layout: [qw, qx, qy, qz]
            qpos[env, qpos_adr + 3] = norm[3]  # qw
            qpos[env, qpos_adr + 4] = norm[0]  # qx
            qpos[env, qpos_adr + 5] = norm[1]  # qy
            qpos[env, qpos_adr + 6] = norm[2]  # qz

        elif jnt_type == JNT_HINGE or jnt_type == JNT_SLIDE:
            var q0_val = rebind[Scalar[DTYPE]](q0[env, qpos_adr])
            var v_val = rebind[Scalar[DTYPE]](vel[env, dof_adr])
            qpos[env, qpos_adr] = q0_val + v_val * dt


# ── per-stage setup (verbatim rk4_stage_kernel pre-stage block :1316) ─────
@always_inline
def _rk4_stage_setup_env_fields[
    DTYPE: DType, NQ: Int, NV: Int, NJOINT: Int, BATCH: Int, STAGE: Int
](
    env: Int,
    dt: Scalar[DTYPE],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    qacc_constrained: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin
    ],
    q0: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    v0: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    A0: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    A1: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    A2: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    C1: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    C2: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
):
    var half_dt = dt * Scalar[DTYPE](0.5)

    comptime if STAGE == 0:
        # Save initial state to scratch
        for i in range(NQ):
            q0[env, i] = qpos[env, i]
        for i in range(NV):
            v0[env, i] = qvel[env, i]
    elif STAGE == 1:
        # Save A[0] from qacc_constrained
        for i in range(NV):
            A0[env, i] = qacc_constrained[env, i]
        # Set intermediate state: qpos = q0 + dt/2 * v0 (C[0] = v0)
        # qvel = v0 + dt/2 * A[0]
        _rk4_integrate_pos_env_fields[DTYPE, NQ, NV, NJOINT, BATCH](
            env, half_dt, joints, q0, v0, qpos
        )
        for i in range(NV):
            var v0_i = rebind[Scalar[DTYPE]](v0[env, i])
            var a0_i = rebind[Scalar[DTYPE]](A0[env, i])
            qvel[env, i] = v0_i + half_dt * a0_i
    elif STAGE == 2:
        # Save A[1] from qacc_constrained
        for i in range(NV):
            A1[env, i] = qacc_constrained[env, i]
        # C[1] = v0 + dt/2 * A[0] — save for combine
        for i in range(NV):
            var v0_i = rebind[Scalar[DTYPE]](v0[env, i])
            var a0_i = rebind[Scalar[DTYPE]](A0[env, i])
            C1[env, i] = v0_i + half_dt * a0_i
        # Set intermediate state: qpos = q0 + dt/2 * C[1]
        # qvel = v0 + dt/2 * A[1]
        _rk4_integrate_pos_env_fields[DTYPE, NQ, NV, NJOINT, BATCH](
            env, half_dt, joints, q0, C1, qpos
        )
        for i in range(NV):
            var v0_i = rebind[Scalar[DTYPE]](v0[env, i])
            var a1_i = rebind[Scalar[DTYPE]](A1[env, i])
            qvel[env, i] = v0_i + half_dt * a1_i
    elif STAGE == 3:
        # Save A[2] from qacc_constrained
        for i in range(NV):
            A2[env, i] = qacc_constrained[env, i]
        # C[2] = v0 + dt/2 * A[1] — save for combine
        for i in range(NV):
            var v0_i = rebind[Scalar[DTYPE]](v0[env, i])
            var a1_i = rebind[Scalar[DTYPE]](A1[env, i])
            C2[env, i] = v0_i + half_dt * a1_i
        # Set intermediate state: qpos = q0 + dt * C[2]
        # qvel = v0 + dt * A[2]
        _rk4_integrate_pos_env_fields[DTYPE, NQ, NV, NJOINT, BATCH](
            env, dt, joints, q0, C2, qpos
        )
        for i in range(NV):
            var v0_i = rebind[Scalar[DTYPE]](v0[env, i])
            var a2_i = rebind[Scalar[DTYPE]](A2[env, i])
            qvel[env, i] = v0_i + dt * a2_i


# ── combine: RK4 weights + integrate (verbatim rk4_combine_kernel :2140) ──
@always_inline
def _rk4_combine_env_fields[
    DTYPE: DType, NQ: Int, NV: Int, NJOINT: Int, BATCH: Int
](
    env: Int,
    dt: Scalar[DTYPE],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    qacc: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    qacc_constrained: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin
    ],
    q0: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    v0: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    A0: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    A1: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    A2: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    C1: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    C2: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
):
    comptime ONE_SIXTH = Scalar[DTYPE](1.0 / 6.0)
    comptime ONE_THIRD = Scalar[DTYPE](1.0 / 3.0)

    # Read A[3] from qacc_constrained (stage 3 just ran)
    # Compute qacc_combined = (A[0] + 2*A[1] + 2*A[2] + A[3]) / 6
    # Compute C[3] = v0 + dt * A[2]
    # v_combined = (C[0] + 2*C[1] + 2*C[2] + C[3]) / 6  where C[0] = v0

    # First pass: compute qacc_combined, v_combined, update qvel/qacc.
    # Store v_combined in the A0 slot (no longer needed after this) —
    # exactly like the legacy combine kernel.
    for i in range(NV):
        var a0_i = rebind[Scalar[DTYPE]](A0[env, i])
        var a1_i = rebind[Scalar[DTYPE]](A1[env, i])
        var a2_i = rebind[Scalar[DTYPE]](A2[env, i])
        var a3_i = rebind[Scalar[DTYPE]](qacc_constrained[env, i])
        var v0_i = rebind[Scalar[DTYPE]](v0[env, i])
        var c1_i = rebind[Scalar[DTYPE]](C1[env, i])
        var c2_i = rebind[Scalar[DTYPE]](C2[env, i])

        # Combined acceleration
        var qacc_i = (
            ONE_SIXTH * a0_i
            + ONE_THIRD * a1_i
            + ONE_THIRD * a2_i
            + ONE_SIXTH * a3_i
        )

        # C[3] = v0 + dt * A[2]
        var c3_i = v0_i + dt * a2_i

        # Combined velocity — stored in A0 for position integration.
        # NaN guard + clamp: if any stage produced NaN qacc, c1/c2/c3 are
        # NaN; clamp v_combined to prevent NaN qpos integration.
        var v_combined_i = (
            ONE_SIXTH * v0_i
            + ONE_THIRD * c1_i
            + ONE_THIRD * c2_i
            + ONE_SIXTH * c3_i
        )
        var vpos_max = Scalar[DTYPE](100.0)
        if v_combined_i != v_combined_i:  # NaN guard: no position change
            v_combined_i = Scalar[DTYPE](0.0)
        elif v_combined_i > vpos_max:
            v_combined_i = vpos_max
        elif v_combined_i < -vpos_max:
            v_combined_i = -vpos_max
        A0[env, i] = v_combined_i

        # Integrate: qvel = v0 + qacc * dt (NaN guard + velocity clamp)
        var qvel_new = v0_i + qacc_i * dt
        var qvel_max = Scalar[DTYPE](100.0)
        if qvel_new != qvel_new:  # NaN guard: reset to zero
            qvel_new = Scalar[DTYPE](0.0)
        elif qvel_new > qvel_max:
            qvel_new = qvel_max
        elif qvel_new < -qvel_max:
            qvel_new = -qvel_max
        qvel[env, i] = qvel_new
        qacc[env, i] = qacc_i

    # Second pass: integrate position using v_combined (quaternion-aware);
    # v_combined lives in the A0 slot.
    _rk4_integrate_pos_env_fields[DTYPE, NQ, NV, NJOINT, BATCH](
        env, dt, joints, q0, A0, qpos
    )


# ── launchable kernels ────────────────────────────────────────────────────
def _rk4_stage_setup_kernel[
    DTYPE: DType, NQ: Int, NV: Int, NJOINT: Int, BATCH: Int, STAGE: Int
](
    dt: Scalar[DTYPE],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    qacc_constrained: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin
    ],
    q0: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    v0: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    A0: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    A1: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    A2: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    C1: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    C2: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _rk4_stage_setup_env_fields[DTYPE, NQ, NV, NJOINT, BATCH, STAGE](
        env, dt, joints, qpos, qvel, qacc_constrained,
        q0, v0, A0, A1, A2, C1, C2,
    )


def _rk4_combine_kernel_fields[
    DTYPE: DType, NQ: Int, NV: Int, NJOINT: Int, BATCH: Int
](
    dt: Scalar[DTYPE],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    qpos: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    qacc: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    qacc_constrained: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin
    ],
    q0: LayoutTensor[DTYPE, Layout.row_major(BATCH, NQ), MutAnyOrigin],
    v0: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    A0: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    A1: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    A2: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    C1: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    C2: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _rk4_combine_env_fields[DTYPE, NQ, NV, NJOINT, BATCH](
        env, dt, joints, qpos, qvel, qacc, qacc_constrained,
        q0, v0, A0, A1, A2, C1, C2,
    )


# ── the stateful integrator ───────────────────────────────────────────────
struct RK4IntegratorFields[
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
](Movable):
    """Owns its scratch; steps RK4 dynamics on either target. With
    CONTACTS=True (default), each stage is followed by contact detection +
    the PGS contact solve (joint limits inside, legacy position) — matching
    the legacy per-stage solver launch. CONTACTS=False = unconstrained
    stages (the original contact-free pilot gates).
    PARALLEL_GPU=True: the GPU FK / body-velocity / cdof / CRBA /
    LDL-factor / M^-1 / RNE stages run their cooperative within-env (_mt)
    kernels (bit-exact vs serial; other stages stay serial). CPU ignores
    it. CRBA_TREEWALK=True (requires PARALLEL_GPU): the GPU CRBA runs the
    legacy-production tree-walk algorithm (O(NV·depth)) instead of the
    dense one — float-tolerance-equal, NOT bit-exact vs dense; mirrors the
    legacy USE_TREEWALK_MM selection (rk4_integrator.mojo:1540). CPU stays
    dense, like legacy."""

    var scratch: DynamicsScratch[Self.DTYPE, Self.NV, Self.NBODY, Self.BATCH]
    var rk4: Rk4Scratch[Self.DTYPE, Self.NQ, Self.NV, Self.BATCH]
    var cscratch: ContactScratch[
        Self.DTYPE, Self.NV, Self.MAX_CONTACTS, Self.BATCH
    ]

    def __init__(out self) raises:
        comptime assert Self.PARALLEL_GPU or (not Self.CRBA_TREEWALK), (
            "RK4IntegratorFields: CRBA_TREEWALK requires PARALLEL_GPU (the"
            " tree-walk CRBA is inherently cooperative)"
        )
        self.scratch = DynamicsScratch[
            Self.DTYPE, Self.NV, Self.NBODY, Self.BATCH
        ]()
        self.rk4 = Rk4Scratch[Self.DTYPE, Self.NQ, Self.NV, Self.BATCH]()
        self.cscratch = ContactScratch[
            Self.DTYPE, Self.NV, Self.MAX_CONTACTS, Self.BATCH
        ]()

    def prepare_gpu(mut self, ctx: DeviceContext) raises:
        """Allocate device buffers for the scratch (once, before stepping)."""
        self.scratch.upload_all(ctx)
        self.rk4.upload_all(ctx)
        self.cscratch.upload_all(ctx)

    def _stage_dynamics[
        target: StaticString
    ](
        mut self,
        mut d: DataFields[
            Self.DTYPE, Self.NQ, Self.NV, Self.NBODY, Self.MAX_CONTACTS,
            Self.NSITE, Self.BATCH,
        ],
        mut m: ModelFields[
            Self.DTYPE, Self.NV, Self.NBODY, Self.NJOINT, Self.NGEOM,
            Self.NEQUALITY, Self.NTENDON, Self.NSITE, Self.NEXCLUDE,
            Self.NMESH_VERTS,
        ],
        ctx: Optional[DeviceContext],
    ) raises:
        """One RK4 stage's forward-dynamics chain (same order as the legacy
        rk4_stage_kernel: FK -> body velocities -> subtree_com -> cdof ->
        CRBA -> +armature (no implicit damping) -> LDL factor + M_inv ->
        RNE -> fnet passive (all explicit) -> LDL solve -> qacc writeback).
        Legacy also runs contact DETECTION here; contact-free slice skips
        it (detection writes only contact state, never qpos/qvel/qacc)."""
        forward_kinematics_fields[
            target, Self.DTYPE, Self.NQ, Self.NV, Self.NBODY, Self.NJOINT,
            Self.MAX_CONTACTS, Self.NGEOM, Self.NEQUALITY, Self.NTENDON,
            Self.NSITE, Self.NEXCLUDE, Self.NMESH_VERTS, Self.BATCH,
            PARALLEL = Self.PARALLEL_GPU,
        ](d, m, ctx)
        compute_body_velocities_fields[
            target, Self.DTYPE, Self.NQ, Self.NV, Self.NBODY, Self.NJOINT,
            Self.MAX_CONTACTS, Self.NGEOM, Self.NEQUALITY, Self.NTENDON,
            Self.NSITE, Self.NEXCLUDE, Self.NMESH_VERTS, Self.BATCH,
            PARALLEL = Self.PARALLEL_GPU,
        ](d, m, ctx)
        compute_subtree_com_fields[
            target, Self.DTYPE, Self.NQ, Self.NV, Self.NBODY, Self.NJOINT,
            Self.MAX_CONTACTS, Self.NGEOM, Self.NEQUALITY, Self.NTENDON,
            Self.NSITE, Self.NEXCLUDE, Self.NMESH_VERTS, Self.BATCH,
        ](d, m, ctx)
        compute_cdof_fields[
            target, Self.DTYPE, Self.NQ, Self.NV, Self.NBODY, Self.NJOINT,
            Self.MAX_CONTACTS, Self.NGEOM, Self.NEQUALITY, Self.NTENDON,
            Self.NSITE, Self.NEXCLUDE, Self.NMESH_VERTS, Self.BATCH,
            PARALLEL = Self.PARALLEL_GPU,
        ](d, m, self.scratch, ctx)
        compute_mass_matrix_fields[
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
        comptime BLOCKS = (Self.BATCH + EU_TPB - 1) // EU_TPB

        # 6b. Armature ONLY (no implicit damping for RK4) — shared env body
        # with Euler (legacy step 6b is byte-identical in both kernels).
        comptime if target == "cpu":
            var joints_v = m.joints.lt["cpu", L_JOINT]()
            var M_v = self.scratch.M.lt["cpu", L_M]()
            for e in range(Self.BATCH):
                _armature_env_fields[
                    Self.DTYPE, Self.NV, Self.NJOINT, Self.BATCH
                ](e, joints_v, M_v)
        else:
            ctx.value().enqueue_function[
                _armature_kernel[Self.DTYPE, Self.NV, Self.NJOINT, Self.BATCH]
            ](
                m.joints.lt["gpu", L_JOINT](),
                self.scratch.M.lt["gpu", L_M](),
                grid_dim=(BLOCKS,),
                block_dim=(EU_TPB,),
            )

        ldl_factor_fields[
            target, Self.DTYPE, Self.NV, Self.NBODY, Self.BATCH,
            PARALLEL = Self.PARALLEL_GPU,
        ](self.scratch, ctx)
        compute_m_inv_fields[
            target, Self.DTYPE, Self.NV, Self.NBODY, Self.BATCH,
            PARALLEL = Self.PARALLEL_GPU,
        ](self.scratch, ctx)
        compute_bias_forces_rne_fields[
            target, Self.DTYPE, Self.NQ, Self.NV, Self.NBODY, Self.NJOINT,
            Self.MAX_CONTACTS, Self.NGEOM, Self.NEQUALITY, Self.NTENDON,
            Self.NSITE, Self.NEXCLUDE, Self.NMESH_VERTS, Self.BATCH,
            PARALLEL = Self.PARALLEL_GPU,
        ](d, m, self.scratch, ctx)

        # 9 + 9b. fnet = qfrc - bias - damping - stiffness - frictionloss
        # (shared env body with Euler; legacy RK4 steps 9/9b are
        # byte-identical to Euler's).
        comptime if target == "cpu":
            var qpos_v = d.qpos.lt["cpu", L_QPOS]()
            var qvel_v = d.qvel.lt["cpu", L_NV]()
            var qfrc_v = d.qfrc.lt["cpu", L_NV]()
            var joints_v2 = m.joints.lt["cpu", L_JOINT]()
            var bias_v = self.scratch.bias.lt["cpu", L_NV]()
            var fnet_v = self.scratch.fnet.lt["cpu", L_NV]()
            for e in range(Self.BATCH):
                _fnet_passive_env_fields[
                    Self.DTYPE, Self.NQ, Self.NV, Self.NJOINT, Self.BATCH
                ](e, qpos_v, qvel_v, qfrc_v, joints_v2, bias_v, fnet_v)
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
                block_dim=(EU_TPB,),
            )

        ldl_solve_fields[
            target, Self.DTYPE, Self.NV, Self.NBODY, Self.BATCH
        ](self.scratch, ctx)

        comptime if target == "cpu":
            var qacc_ws_v = self.scratch.qacc_ws.lt["cpu", L_NV]()
            var qacc_v = d.qacc.lt["cpu", L_NV]()
            var qacc_c_v = self.scratch.qacc_constrained.lt["cpu", L_NV]()
            for e in range(Self.BATCH):
                _qacc_writeback_env_fields[Self.DTYPE, Self.NV, Self.BATCH](
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
                block_dim=(EU_TPB,),
            )

    def _stage_setup[
        target: StaticString, STAGE: Int
    ](
        mut self,
        dt: Scalar[Self.DTYPE],
        mut d: DataFields[
            Self.DTYPE, Self.NQ, Self.NV, Self.NBODY, Self.MAX_CONTACTS,
            Self.NSITE, Self.BATCH,
        ],
        mut m: ModelFields[
            Self.DTYPE, Self.NV, Self.NBODY, Self.NJOINT, Self.NGEOM,
            Self.NEQUALITY, Self.NTENDON, Self.NSITE, Self.NEXCLUDE,
            Self.NMESH_VERTS,
        ],
        ctx: Optional[DeviceContext],
    ) raises:
        comptime L_JOINT = Layout.row_major(Self.NJOINT, MODEL_JOINT_SIZE)
        comptime L_NV = Layout.row_major(Self.BATCH, Self.NV)
        comptime L_QPOS = Layout.row_major(Self.BATCH, Self.NQ)
        comptime BLOCKS = (Self.BATCH + EU_TPB - 1) // EU_TPB

        comptime if target == "cpu":
            var joints_v = m.joints.lt["cpu", L_JOINT]()
            var qpos_v = d.qpos.lt["cpu", L_QPOS]()
            var qvel_v = d.qvel.lt["cpu", L_NV]()
            var qacc_c_v = self.scratch.qacc_constrained.lt["cpu", L_NV]()
            var q0_v = self.rk4.q0.lt["cpu", L_QPOS]()
            var v0_v = self.rk4.v0.lt["cpu", L_NV]()
            var A0_v = self.rk4.A0.lt["cpu", L_NV]()
            var A1_v = self.rk4.A1.lt["cpu", L_NV]()
            var A2_v = self.rk4.A2.lt["cpu", L_NV]()
            var C1_v = self.rk4.C1.lt["cpu", L_NV]()
            var C2_v = self.rk4.C2.lt["cpu", L_NV]()
            for e in range(Self.BATCH):
                _rk4_stage_setup_env_fields[
                    Self.DTYPE, Self.NQ, Self.NV, Self.NJOINT, Self.BATCH,
                    STAGE,
                ](
                    e, dt, joints_v, qpos_v, qvel_v, qacc_c_v,
                    q0_v, v0_v, A0_v, A1_v, A2_v, C1_v, C2_v,
                )
        else:
            ctx.value().enqueue_function[
                _rk4_stage_setup_kernel[
                    Self.DTYPE, Self.NQ, Self.NV, Self.NJOINT, Self.BATCH,
                    STAGE,
                ]
            ](
                dt,
                m.joints.lt["gpu", L_JOINT](),
                d.qpos.lt["gpu", L_QPOS](),
                d.qvel.lt["gpu", L_NV](),
                self.scratch.qacc_constrained.lt["gpu", L_NV](),
                self.rk4.q0.lt["gpu", L_QPOS](),
                self.rk4.v0.lt["gpu", L_NV](),
                self.rk4.A0.lt["gpu", L_NV](),
                self.rk4.A1.lt["gpu", L_NV](),
                self.rk4.A2.lt["gpu", L_NV](),
                self.rk4.C1.lt["gpu", L_NV](),
                self.rk4.C2.lt["gpu", L_NV](),
                grid_dim=(BLOCKS,),
                block_dim=(EU_TPB,),
            )

    def step[
        target: StaticString, CONTACTS: Bool = True
    ](
        mut self,
        mut d: DataFields[
            Self.DTYPE, Self.NQ, Self.NV, Self.NBODY, Self.MAX_CONTACTS,
            Self.NSITE, Self.BATCH,
        ],
        mut m: ModelFields[
            Self.DTYPE, Self.NV, Self.NBODY, Self.NJOINT, Self.NGEOM,
            Self.NEQUALITY, Self.NTENDON, Self.NSITE, Self.NEXCLUDE,
            Self.NMESH_VERTS,
        ],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """One full RK4 step (4 stages [+ per-stage contact/limit solve] +
        combine)."""
        # Fluid forces are not ported yet — refuse rather than silently
        # diverge from the legacy step.
        if (
            m.meta.data[MODEL_META_IDX_DENSITY] != 0
            or m.meta.data[MODEL_META_IDX_VISCOSITY] != 0
        ):
            raise Error(
                "RK4IntegratorFields: fluid forces (density/viscosity) not"
                " ported yet"
            )
        var dt = m.meta.data[MODEL_META_IDX_TIMESTEP]

        comptime for s in range(4):
            self._stage_setup[target, s](dt, d, m, ctx)
            self._stage_dynamics[target](d, m, ctx)
            # Per-stage constraint solve (legacy: solver launch after every
            # stage kernel; corrects qacc_constrained before the next
            # stage's A[k] snapshot / the combine).
            comptime if CONTACTS:
                # Auto broadphase = legacy production (the legacy stage
                # kernel calls detect_contacts_auto_gpu): SAP for
                # NGEOM >= 16, O(N^2) otherwise — routing is bit-identical
                # for every existing gate model (all NGEOM < 16).
                detect_contacts_auto_fields[
                    target, Self.DTYPE, Self.NQ, Self.NV, Self.NBODY,
                    Self.NJOINT, Self.MAX_CONTACTS, Self.NGEOM,
                    Self.NEQUALITY, Self.NTENDON, Self.NSITE, Self.NEXCLUDE,
                    Self.NMESH_VERTS, Self.BATCH,
                ](d, m, ctx)
                comptime assert (
                    Self.SOLVER == "pgs"
                    or Self.SOLVER == "newton"
                    or Self.SOLVER == "cg"
                    or Self.SOLVER == "island"
                ), (
                    "RK4IntegratorFields: SOLVER must be 'pgs', 'newton',"
                    " 'cg', or 'island'"
                )
                comptime if Self.SOLVER == "newton":
                    solve_newton_fields[
                        target, Self.DTYPE, Self.NQ, Self.NV, Self.NBODY,
                        Self.NJOINT, Self.MAX_CONTACTS, Self.NGEOM,
                        Self.NEQUALITY, Self.NTENDON, Self.NSITE,
                        Self.NEXCLUDE, Self.NMESH_VERTS, Self.CONE_TYPE,
                        Self.BATCH,
                    ](d, m, self.scratch, self.cscratch, ctx)
                else:
                    comptime if Self.SOLVER == "cg":
                        solve_cg_fields[
                            target, Self.DTYPE, Self.NQ, Self.NV, Self.NBODY,
                            Self.NJOINT, Self.MAX_CONTACTS, Self.NGEOM,
                            Self.NEQUALITY, Self.NTENDON, Self.NSITE,
                            Self.NEXCLUDE, Self.NMESH_VERTS, Self.CONE_TYPE,
                            Self.BATCH,
                        ](d, m, self.scratch, self.cscratch, ctx)
                    else:
                        comptime if Self.SOLVER == "island":
                            solve_island_pgs_fields[
                                target, Self.DTYPE, Self.NQ, Self.NV,
                                Self.NBODY, Self.NJOINT, Self.MAX_CONTACTS,
                                Self.NGEOM, Self.NEQUALITY, Self.NTENDON,
                                Self.NSITE, Self.NEXCLUDE, Self.NMESH_VERTS,
                                Self.CONE_TYPE, Self.BATCH,
                            ](d, m, self.scratch, self.cscratch, ctx)
                        else:
                            solve_contacts_fields[
                                target, Self.DTYPE, Self.NQ, Self.NV,
                                Self.NBODY, Self.NJOINT, Self.MAX_CONTACTS,
                                Self.NGEOM, Self.NEQUALITY, Self.NTENDON,
                                Self.NSITE, Self.NEXCLUDE, Self.NMESH_VERTS,
                                Self.CONE_TYPE, Self.BATCH,
                            ](d, m, self.scratch, self.cscratch, ctx)

        comptime L_JOINT = Layout.row_major(Self.NJOINT, MODEL_JOINT_SIZE)
        comptime L_NV = Layout.row_major(Self.BATCH, Self.NV)
        comptime L_QPOS = Layout.row_major(Self.BATCH, Self.NQ)
        comptime BLOCKS = (Self.BATCH + EU_TPB - 1) // EU_TPB

        comptime if target == "cpu":
            var joints_v = m.joints.lt["cpu", L_JOINT]()
            var qpos_v = d.qpos.lt["cpu", L_QPOS]()
            var qvel_v = d.qvel.lt["cpu", L_NV]()
            var qacc_v = d.qacc.lt["cpu", L_NV]()
            var qacc_c_v = self.scratch.qacc_constrained.lt["cpu", L_NV]()
            var q0_v = self.rk4.q0.lt["cpu", L_QPOS]()
            var v0_v = self.rk4.v0.lt["cpu", L_NV]()
            var A0_v = self.rk4.A0.lt["cpu", L_NV]()
            var A1_v = self.rk4.A1.lt["cpu", L_NV]()
            var A2_v = self.rk4.A2.lt["cpu", L_NV]()
            var C1_v = self.rk4.C1.lt["cpu", L_NV]()
            var C2_v = self.rk4.C2.lt["cpu", L_NV]()
            for e in range(Self.BATCH):
                _rk4_combine_env_fields[
                    Self.DTYPE, Self.NQ, Self.NV, Self.NJOINT, Self.BATCH
                ](
                    e, dt, joints_v, qpos_v, qvel_v, qacc_v, qacc_c_v,
                    q0_v, v0_v, A0_v, A1_v, A2_v, C1_v, C2_v,
                )
        else:
            ctx.value().enqueue_function[
                _rk4_combine_kernel_fields[
                    Self.DTYPE, Self.NQ, Self.NV, Self.NJOINT, Self.BATCH
                ]
            ](
                dt,
                m.joints.lt["gpu", L_JOINT](),
                d.qpos.lt["gpu", L_QPOS](),
                d.qvel.lt["gpu", L_NV](),
                d.qacc.lt["gpu", L_NV](),
                self.scratch.qacc_constrained.lt["gpu", L_NV](),
                self.rk4.q0.lt["gpu", L_QPOS](),
                self.rk4.v0.lt["gpu", L_NV](),
                self.rk4.A0.lt["gpu", L_NV](),
                self.rk4.A1.lt["gpu", L_NV](),
                self.rk4.A2.lt["gpu", L_NV](),
                self.rk4.C1.lt["gpu", L_NV](),
                self.rk4.C2.lt["gpu", L_NV](),
                grid_dim=(BLOCKS,),
                block_dim=(EU_TPB,),
            )

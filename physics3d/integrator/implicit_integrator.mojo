"""Full implicit integrator matching MuJoCo's implicit integration scheme.

Like ImplicitFastIntegrator, but uses the FULL qDeriv matrix including
the RNE velocity derivative (d(bias)/d(qvel)), not just passive force
derivatives. This makes qDeriv a dense, non-symmetric NV×NV matrix,
requiring LU factorization instead of LDL.

  M_hat = M + armature - dt * qDeriv

where qDeriv includes:
  - Passive force derivatives: qDeriv[i,i] = -damping[i]  (diagonal)
  - RNE velocity derivative: d(Coriolis + centrifugal)/d(qvel)  (dense, non-symmetric)

For systems with significant gyroscopic effects (rapidly spinning objects,
robot arms at high speed), this provides better numerical stability by
accounting for how bias forces change with velocity.

CPU only. GPU deferred (matches MuJoCo Warp which also hasn't implemented this).

Pipeline matches ImplicitFastIntegrator except:
  - Step 5b: Full qDeriv computation (RNE velocity derivative)
  - Step 6: LU factorization instead of LDL
  - Step 7: M_inv via LU solve instead of LDL

Reference: MuJoCo engine_forward.c:1117-1137 (implicit path)
Reference: MuJoCo engine_derivative.c:596-705 (mjd_rne_vel)
"""

from math import sqrt
from gpu.host import DeviceContext, DeviceBuffer
from gpu import thread_idx, block_idx, block_dim
from layout import LayoutTensor, Layout

from ..types import Model, Data, _max_one
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
    forward_kinematics_gpu,
    compute_body_velocities_gpu,
)
from ..kinematics.quat_math import quat_normalize, quat_integrate, quat_rotate
from ..dynamics.mass_matrix import (
    compute_mass_matrix,
    compute_mass_matrix_full,
    compute_mass_matrix_full_gpu,
    ldl_factor,
    ldl_factor_gpu,
    ldl_solve,
    ldl_solve_gpu,
    ldl_solve_workspace_gpu,
    compute_M_inv_from_ldl,
    compute_M_inv_from_ldl_gpu,
    solve_linear_diagonal,
)
from ..dynamics.bias_forces import (
    compute_bias_forces,
    compute_bias_forces_rne,
    compute_bias_forces_rne_gpu,
)
from ..dynamics.jacobian import (
    compute_cdof,
    compute_cdof_gpu,
    compute_composite_inertia,
    compute_composite_inertia_gpu,
)
from ..dynamics.velocity_derivatives import compute_rne_vel_derivative
from ..dynamics.lu_factorization import (
    lu_factor,
    lu_solve,
    compute_M_inv_from_lu,
)
from ..collision.contact_detection import (
    detect_contacts,
    detect_contacts_gpu,
    normalize_qpos_quaternions,
    normalize_qpos_quaternions_gpu,
)
from ..solver.pgs_solver import PGSSolver
from ..constraints.constraint_data import (
    ConstraintData,
    CNSTR_NORMAL,
    CNSTR_FRICTION_T1,
    CNSTR_FRICTION_T2,
    CNSTR_LIMIT,
)
from ..constraints.constraint_builder import (
    build_constraints,
    writeback_forces,
)
from ..traits.integrator import Integrator
from ..traits.solver import ConstraintSolver
from ..gpu.constants import (
    TPB,
    state_size,
    model_size,
    model_metadata_offset,
    model_joint_offset,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_ARMATURE,
    JOINT_IDX_STIFFNESS,
    JOINT_IDX_DAMPING,
    JOINT_IDX_SPRINGREF,
    JOINT_IDX_FRICTIONLOSS,
    MODEL_META_IDX_TIMESTEP,
    integrator_workspace_size,
    ws_M_offset,
    ws_bias_offset,
    ws_fnet_offset,
    ws_qacc_ws_offset,
    ws_qacc_constrained_offset,
    ws_m_inv_offset,
)


struct ImplicitIntegrator[SOLVER: ConstraintSolver](Integrator):
    """Full implicit integrator with RNE velocity derivative.

    Uses M_hat = M + armature - dt * qDeriv where qDeriv includes both
    passive force derivatives AND the full RNE velocity derivative
    (d(Coriolis + centrifugal)/d(qvel)).

    Since qDeriv is non-symmetric, uses LU factorization instead of LDL.

    Parametrized by SOLVER type (PGSSolver, CGSolver, or NewtonSolver).

    Usage:
        # PGS:
        alias PGSImplicit = ImplicitIntegrator[PGSSolver]

        # Newton (most accurate):
        alias NewtonImplicit = ImplicitIntegrator[NewtonSolver]
    """

    # =========================================================================
    # CPU Methods
    # =========================================================================

    @staticmethod
    fn step[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
    ](
        model: Model[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, MAX_EQUALITY
        ],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        verbose: Bool = False,
    ) where DTYPE.is_floating_point():
        """Execute one simulation step with full implicit integration.

        Uses the full RNE velocity derivative for qDeriv and LU factorization
        for the non-symmetric M_hat = M + armature - dt * qDeriv.

        Args:
            model: Static model configuration.
            data: Mutable simulation state.
            verbose: Whether to print debug information.
        """
        var dt = model.timestep
        comptime M_SIZE = _max_one[NV * NV]()
        comptime V_SIZE = _max_one[NV]()
        comptime CDOF_SIZE = _max_one[NV * 6]()
        comptime CRB_SIZE = _max_one[NBODY * 10]()

        # 1. Forward kinematics
        forward_kinematics(model, data)
        compute_body_velocities(model, data)

        # 2. Collision detection
        detect_contacts[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM](
            model, data
        )

        if verbose:
            print("  [FK] body positions:")
            for b in range(NBODY):
                print(
                    "    body",
                    b,
                    "(",
                    model.get_body_name(b),
                    "): x=",
                    Float64(data.xpos[b * 3]),
                    " y=",
                    Float64(data.xpos[b * 3 + 1]),
                    " z=",
                    Float64(data.xpos[b * 3 + 2]),
                )
            print("  [FK] contacts:", data.num_contacts)

        # 3. Compute cdof (spatial motion axes per DOF)
        var cdof = InlineArray[Scalar[DTYPE], CDOF_SIZE](uninitialized=True)
        compute_cdof[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CDOF_SIZE](
            model, data, cdof
        )

        # 4. Compute composite rigid body inertia
        var crb = InlineArray[Scalar[DTYPE], CRB_SIZE](uninitialized=True)
        for i in range(CRB_SIZE):
            crb[i] = Scalar[DTYPE](0)
        compute_composite_inertia[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CRB_SIZE
        ](model, data, crb)

        # 5. Compute full mass matrix using CRBA
        var M = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        for i in range(M_SIZE):
            M[i] = Scalar[DTYPE](0)
        compute_mass_matrix_full[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            M_SIZE,
            CDOF_SIZE,
            CRB_SIZE,
        ](model, data, cdof, crb, M)

        # 5b. Compute FULL qDeriv and modify mass matrix
        # M_hat = M + armature - dt * qDeriv
        #
        # qDeriv has two parts:
        # (a) Passive force derivative: qDeriv[i,i] = -damping[i]  (diagonal)
        # (b) RNE velocity derivative: d(bias)/d(qvel)  (dense, non-symmetric)
        #
        # Initialize qDeriv with passive damping (diagonal)
        var qDeriv = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        for i in range(M_SIZE):
            qDeriv[i] = Scalar[DTYPE](0)

        # (a) Passive damping: qDeriv[i,i] = -damping[i]
        for j in range(model.num_joints):
            var joint = model.joints[j]
            var dof_adr = joint.dof_adr
            var damp = joint.damping
            if joint.jnt_type == JNT_FREE:
                for d in range(6):
                    qDeriv[(dof_adr + d) * NV + (dof_adr + d)] = -damp
            elif joint.jnt_type == JNT_BALL:
                for d in range(3):
                    qDeriv[(dof_adr + d) * NV + (dof_adr + d)] = -damp
            else:
                qDeriv[dof_adr * NV + dof_adr] = -damp

        # (b) RNE velocity derivative: subtract d(bias)/d(qvel) from qDeriv
        compute_rne_vel_derivative[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            M_SIZE,
            CDOF_SIZE,
        ](model, data, cdof, qDeriv)

        # Now form M_hat = M + armature - dt * qDeriv
        # Add armature to diagonal
        for j in range(model.num_joints):
            var joint = model.joints[j]
            var dof_adr = joint.dof_adr
            var arm = joint.armature
            if joint.jnt_type == JNT_FREE:
                for d in range(6):
                    M[(dof_adr + d) * NV + (dof_adr + d)] = (
                        M[(dof_adr + d) * NV + (dof_adr + d)] + arm
                    )
            elif joint.jnt_type == JNT_BALL:
                for d in range(3):
                    M[(dof_adr + d) * NV + (dof_adr + d)] = (
                        M[(dof_adr + d) * NV + (dof_adr + d)] + arm
                    )
            else:
                M[dof_adr * NV + dof_adr] = M[dof_adr * NV + dof_adr] + arm

        # Subtract dt * qDeriv from M (full matrix, not just diagonal)
        for i in range(NV):
            for j_col in range(NV):
                M[i * NV + j_col] = (
                    M[i * NV + j_col] - dt * qDeriv[i * NV + j_col]
                )

        if verbose:
            print("  [IMPLICIT] M_hat diagonal:", end="")
            for i in range(NV):
                print(" ", Float64(M[i * NV + i]), end="")
            print("")
            # Check if qDeriv has off-diagonal terms
            var max_offdiag: Float64 = 0
            for i in range(NV):
                for j_col in range(NV):
                    if i != j_col:
                        var val = abs(Float64(qDeriv[i * NV + j_col]))
                        if val > max_offdiag:
                            max_offdiag = val
            print(
                "  [IMPLICIT] qDeriv max off-diagonal:",
                max_offdiag,
            )

        # 6. LU factorize M_hat (non-symmetric) and solve for qacc
        # M_hat is stored in M after the modifications above
        var M_lu = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        for i in range(M_SIZE):
            M_lu[i] = M[i]  # Copy for LU (in-place)

        var piv = InlineArray[Int, V_SIZE](uninitialized=True)
        for i in range(NV):
            piv[i] = i
        lu_factor[DTYPE, NV, M_SIZE, V_SIZE](M_lu, piv)

        var bias = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(V_SIZE):
            bias[i] = Scalar[DTYPE](0)
        compute_bias_forces_rne[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE, CDOF_SIZE
        ](model, data, cdof, bias)

        var f_net = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NV):
            f_net[i] = data.qfrc[i] - bias[i]

        # 6b. Apply passive joint forces: damping + stiffness + frictionloss
        for j in range(model.num_joints):
            var joint_d = model.joints[j]
            var dof_adr_d = joint_d.dof_adr
            var damp_d = joint_d.damping
            if damp_d > Scalar[DTYPE](0):
                if joint_d.jnt_type == JNT_FREE:
                    for d in range(6):
                        f_net[dof_adr_d + d] = (
                            f_net[dof_adr_d + d]
                            - damp_d * data.qvel[dof_adr_d + d]
                        )
                elif joint_d.jnt_type == JNT_BALL:
                    for d in range(3):
                        f_net[dof_adr_d + d] = (
                            f_net[dof_adr_d + d]
                            - damp_d * data.qvel[dof_adr_d + d]
                        )
                else:
                    f_net[dof_adr_d] = (
                        f_net[dof_adr_d] - damp_d * data.qvel[dof_adr_d]
                    )

        # Stiffness: f -= stiffness * (qpos - springref)
        # Frictionloss: f -= frictionloss * sign(qvel)
        for j in range(model.num_joints):
            var joint = model.joints[j]
            var dof_adr = joint.dof_adr
            var qpos_adr = joint.qpos_adr
            var stiff = joint.stiffness
            var sref = joint.springref
            var floss = joint.frictionloss
            if stiff > Scalar[DTYPE](0):
                if joint.jnt_type == JNT_FREE:
                    for d in range(6):
                        f_net[dof_adr + d] = f_net[dof_adr + d] - stiff * (
                            data.qpos[qpos_adr + d] - sref
                        )
                elif joint.jnt_type == JNT_BALL:
                    for d in range(3):
                        f_net[dof_adr + d] = f_net[dof_adr + d] - stiff * (
                            data.qpos[qpos_adr + d] - sref
                        )
                else:
                    f_net[dof_adr] = f_net[dof_adr] - stiff * (
                        data.qpos[qpos_adr] - sref
                    )
            if floss > Scalar[DTYPE](0):
                comptime VEL_THRESH: Scalar[DTYPE] = 1e-4
                if joint.jnt_type == JNT_FREE:
                    for d in range(6):
                        var v = data.qvel[dof_adr + d]
                        if v > VEL_THRESH:
                            f_net[dof_adr + d] = f_net[dof_adr + d] - floss
                        elif v < -VEL_THRESH:
                            f_net[dof_adr + d] = f_net[dof_adr + d] + floss
                elif joint.jnt_type == JNT_BALL:
                    for d in range(3):
                        var v = data.qvel[dof_adr + d]
                        if v > VEL_THRESH:
                            f_net[dof_adr + d] = f_net[dof_adr + d] - floss
                        elif v < -VEL_THRESH:
                            f_net[dof_adr + d] = f_net[dof_adr + d] + floss
                else:
                    var v = data.qvel[dof_adr]
                    if v > VEL_THRESH:
                        f_net[dof_adr] = f_net[dof_adr] - floss
                    elif v < -VEL_THRESH:
                        f_net[dof_adr] = f_net[dof_adr] + floss

        # qacc = M_hat^-1 * f_net via LU solve
        var qacc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NV):
            qacc[i] = Scalar[DTYPE](0)
        lu_solve[DTYPE, NV, M_SIZE, V_SIZE](M_lu, piv, f_net, qacc)

        # 7. Compute full M_inv from LU factors for constraint solver
        var M_inv = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        for i in range(M_SIZE):
            M_inv[i] = Scalar[DTYPE](0)
        compute_M_inv_from_lu[DTYPE, NV, M_SIZE, V_SIZE](M_lu, piv, M_inv)

        if verbose:
            print("  [PRE-SOLVER]")
            print("    qacc_unconstrained:", end="")
            for i in range(NV):
                print(" ", Float64(qacc[i]), end="")
            print("")
            print("    f_net:", end="")
            for i in range(NV):
                print(" ", Float64(f_net[i]), end="")
            print("")

        # 8. Build constraints and solve (modifies qacc in-place)
        comptime MAX_ROWS = 11 * MAX_CONTACTS + 2 * NJOINT + 6 * MAX_EQUALITY
        var constraints = ConstraintData[DTYPE, MAX_ROWS, NV]()
        build_constraints[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            MAX_ROWS,
            V_SIZE,
            M_SIZE,
            CDOF_SIZE,
            NGEOM,
            MAX_EQUALITY,
        ](model, data, cdof, M_inv, qacc, dt, constraints)

        if verbose:
            print(
                "  [CONSTRAINTS] num_rows:",
                constraints.num_rows,
                "normals:",
                constraints.num_normals,
                "friction:",
                constraints.num_friction,
                "limits:",
                constraints.num_limits,
            )

        Self.SOLVER.solve[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            MAX_ROWS,
            V_SIZE,
            M_SIZE,
        ](model, data, M_inv, constraints, qacc, dt)

        if verbose:
            print("    qacc after solve:", end="")
            for i in range(NV):
                print(" ", Float64(qacc[i]), end="")
            print("")

        writeback_forces[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            MAX_ROWS,
        ](constraints, data)

        # 9. Integrate: qvel = old_qvel + constrained_qacc * dt
        comptime MAX_QVEL: Scalar[DTYPE] = 100.0
        for i in range(NV):
            data.qacc[i] = qacc[i]
            data.qvel[i] = data.qvel[i] + qacc[i] * dt

        # 9b. Clamp velocities
        for i in range(NV):
            if data.qvel[i] > MAX_QVEL:
                data.qvel[i] = MAX_QVEL
            elif data.qvel[i] < -MAX_QVEL:
                data.qvel[i] = -MAX_QVEL

        for i in range(NQ):
            if i < NV:
                data.qpos[i] = data.qpos[i] + data.qvel[i] * dt

        # 10. Normalize quaternions
        normalize_qpos_quaternions(model, data)

        if verbose:
            print("  [POST-INTEGRATION]")
            print("    qvel_new:", end="")
            for i in range(NV):
                print(" ", Float64(data.qvel[i]), end="")
            print("")

    @staticmethod
    fn simulate[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
    ](
        model: Model[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, MAX_EQUALITY
        ],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        num_steps: Int,
    ) where DTYPE.is_floating_point():
        """Run simulation for multiple steps on CPU."""
        for _ in range(num_steps):
            Self.step[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, MAX_EQUALITY
            ](model, data)

    # =========================================================================
    # GPU Methods — Deferred (fall back to ImplicitFast on GPU)
    # =========================================================================
    # The full implicit integrator's RNE velocity derivative is CPU-only.
    # GPU support uses the same kernels as ImplicitFastIntegrator.
    # This matches MuJoCo Warp which also hasn't implemented the full
    # implicit integrator on GPU.

    @always_inline
    @staticmethod
    fn step_kernel[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
        MODEL_SIZE: Int,
        BATCH: Int,
        WS_SIZE: Int,
        NGEOM: Int = 0,
    ](
        state: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        model: LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ],
        workspace: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
        ],
    ):
        """GPU step kernel — uses ImplicitFast (no RNE vel derivative on GPU).

        The full implicit integrator is CPU-only. On GPU, this falls back to
        the implicit-fast kernel (diagonal qDeriv only).
        """
        # Import and delegate to ImplicitFastIntegrator's step kernel
        from .implicit_fast_integrator import ImplicitFastIntegrator

        ImplicitFastIntegrator[Self.SOLVER].step_kernel[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            STATE_SIZE,
            MODEL_SIZE,
            BATCH,
            WS_SIZE,
            NGEOM,
        ](state, model, workspace)

    @always_inline
    @staticmethod
    fn step_finalize_kernel[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
        MODEL_SIZE: Int,
        BATCH: Int,
        WS_SIZE: Int,
    ](
        state: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        model: LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ],
        workspace: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
        ],
    ):
        """GPU finalize kernel — delegates to ImplicitFast."""
        from .implicit_fast_integrator import ImplicitFastIntegrator

        ImplicitFastIntegrator[Self.SOLVER].step_finalize_kernel[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            STATE_SIZE,
            MODEL_SIZE,
            BATCH,
            WS_SIZE,
        ](state, model, workspace)

    @staticmethod
    fn step_gpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        BATCH: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[DTYPE],
        mut model_buf: DeviceBuffer[DTYPE],
        mut workspace_buf: DeviceBuffer[DTYPE],
        dt: Scalar[DTYPE],
        gravity_z: Scalar[DTYPE],
        ground_z: Scalar[DTYPE],
    ) raises:
        """GPU step — delegates to ImplicitFast (no RNE vel derivative on GPU)."""
        from .implicit_fast_integrator import ImplicitFastIntegrator

        ImplicitFastIntegrator[Self.SOLVER].step_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            BATCH,
            NGEOM,
            MAX_EQUALITY,
        ](
            ctx,
            state_buf,
            model_buf,
            workspace_buf,
            dt,
            gravity_z,
            ground_z,
        )

    @staticmethod
    fn simulate_gpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        BATCH: Int,
        NGEOM: Int = 0,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[DTYPE],
        mut model_buf: DeviceBuffer[DTYPE],
        mut workspace_buf: DeviceBuffer[DTYPE],
        num_steps: Int,
        dt: Scalar[DTYPE],
        gravity_z: Scalar[DTYPE],
        ground_z: Scalar[DTYPE],
    ) raises:
        """Run simulation for multiple steps on GPU."""
        for _ in range(num_steps):
            Self.step_gpu[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, BATCH, NGEOM
            ](
                ctx,
                state_buf,
                model_buf,
                workspace_buf,
                dt,
                gravity_z,
                ground_z,
            )

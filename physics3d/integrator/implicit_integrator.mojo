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
from .implicit_fast_integrator import ImplicitFastIntegrator
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
from ..dynamics.velocity_derivatives import (
    compute_rne_vel_derivative,
    compute_rne_vel_derivative_gpu,
)
from ..dynamics.lu_factorization import (
    lu_factor,
    lu_solve,
    compute_M_inv_from_lu,
    lu_factor_gpu,
    lu_solve_workspace_gpu,
    compute_M_inv_from_lu_gpu,
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
    model_size_with_invweight,
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
    ws_L_offset,
    ws_bias_offset,
    ws_fnet_offset,
    ws_qacc_ws_offset,
    ws_qacc_constrained_offset,
    ws_m_inv_offset,
    ws_solver_offset,
    implicit_extra_workspace_size,
    ws_implicit_qderiv_offset,
)


struct ImplicitIntegrator[SOLVER: ConstraintSolver](Integrator):
    """Full implicit integrator with RNE velocity derivative.

    Uses M_hat = M + armature - dt * qDeriv where qDeriv includes both
    passive force derivatives AND the full RNE velocity derivative
    (d(Coriolis + centrifugal)/d(qvel)).

    Since qDeriv is non-symmetric, uses LU factorization instead of LDL.

    Parametrized by SOLVER type (PGSSolver, NewtonSolver, or CGSolver).

    Usage:
        # PGS:
        alias PGSImplicit = ImplicitIntegrator[PGSSolver]

        # Newton (most accurate, matches MuJoCo):
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
        CONE_TYPE: Int = ConeType.ELLIPTIC,
    ](
        model: Model[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
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
        detect_contacts(model, data)

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
        compute_cdof(model, data, cdof)

        # 4. Compute composite rigid body inertia
        var crb = InlineArray[Scalar[DTYPE], CRB_SIZE](uninitialized=True)
        for i in range(CRB_SIZE):
            crb[i] = Scalar[DTYPE](0)
        compute_composite_inertia(model, data, crb)

        # 5. Compute full mass matrix using CRBA
        var M = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        for i in range(M_SIZE):
            M[i] = Scalar[DTYPE](0)
        compute_mass_matrix_full(model, data, cdof, crb, M)

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
        compute_rne_vel_derivative(model, data, cdof, qDeriv)

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
        build_constraints[CONE_TYPE=CONE_TYPE,](
            model, data, cdof, M_inv, qacc, dt, constraints
        )

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

        Self.SOLVER.solve[CONE_TYPE=CONE_TYPE](
            model, data, M_inv, constraints, qacc, dt
        )

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

        # 9. Integrate: semi-implicit (symplectic) Euler
        # MuJoCo 3.3.6 uses symplectic Euler for ALL integrators:
        #   qvel += dt * qacc
        #   qpos += dt * qvel_new

        # Velocity update
        for i in range(NV):
            data.qacc[i] = qacc[i]
            data.qvel[i] = data.qvel[i] + qacc[i] * dt

        # Position update: symplectic Euler (use new velocity)
        for i in range(NQ):
            if i < NV:
                data.qpos[i] = data.qpos[i] + dt * data.qvel[i]

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
        CONE_TYPE: Int = ConeType.ELLIPTIC,
    ](
        model: Model[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
        ],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        num_steps: Int,
    ) where DTYPE.is_floating_point():
        """Run simulation for multiple steps on CPU."""
        for _ in range(num_steps):
            Self.step[
                DTYPE,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                NGEOM,
                MAX_EQUALITY,
                CONE_TYPE,
            ](model, data)

    # =========================================================================
    # GPU Methods — Full implicit with RNE velocity derivative
    # =========================================================================

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
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
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
        """GPU step kernel with full implicit integration.

        Same pipeline as ImplicitFast steps 1-6, then:
        - Step 6b: Full qDeriv via RNE velocity derivative
        - Step 6c: M_hat = M + armature - dt * qDeriv (full matrix)
        - Step 7: LU factorization + M_inv computation
        - Steps 8-10: Bias forces, passive forces, LU solve
        """
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        comptime M_idx = ws_M_offset[NV, NBODY]()
        comptime L_idx = ws_L_offset[NV, NBODY]()
        comptime bias_idx = ws_bias_offset[NV, NBODY]()
        comptime fnet_idx = ws_fnet_offset[NV, NBODY]()
        comptime qacc_ws_idx = ws_qacc_ws_offset[NV, NBODY]()
        comptime qacc_constrained_idx = ws_qacc_constrained_offset[NV, NBODY]()

        # Implicit extra workspace starts after solver workspace
        comptime implicit_base = ws_solver_offset[
            NV, NBODY
        ]() + Self.SOLVER.solver_workspace_size[NV, MAX_CONTACTS]()
        comptime qd_off = ws_implicit_qderiv_offset(implicit_base)

        # 1. Forward kinematics
        forward_kinematics_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            STATE_SIZE,
            MODEL_SIZE,
            BATCH,
        ](env, state, model)

        # 2. Compute body velocities
        compute_body_velocities_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            STATE_SIZE,
            MODEL_SIZE,
            BATCH,
        ](env, state, model)

        # 3. Detect contacts
        detect_contacts_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            STATE_SIZE,
            MODEL_SIZE,
            BATCH,
            NGEOM,
        ](env, state, model)

        # 4. Compute cdof
        compute_cdof_gpu[
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
        ](env, state, model, workspace)

        # 5. Compute composite rigid body inertia
        compute_composite_inertia_gpu[
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
        ](env, state, model, workspace)

        # 6. Compute full mass matrix
        compute_mass_matrix_full_gpu[
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
        ](env, state, model, workspace)

        # 6b. Compute full qDeriv via RNE velocity derivative
        # Initialize qDeriv to zero
        for i in range(NV * NV):
            workspace[env, qd_off + i] = Scalar[DTYPE](0)

        # Set passive damping diagonal: qDeriv[i,i] = -damping[i]
        var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
        var dt = rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_TIMESTEP]
        )

        for j in range(NJOINT):
            var joint_off = model_joint_offset[NBODY](j)
            var jnt_type = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
            )
            var dof_adr = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR])
            )
            var damp = rebind[Scalar[DTYPE]](
                model[0, joint_off + JOINT_IDX_DAMPING]
            )
            if jnt_type == JNT_FREE:
                for d in range(6):
                    workspace[
                        env, qd_off + (dof_adr + d) * NV + (dof_adr + d)
                    ] = -damp
            elif jnt_type == JNT_BALL:
                for d in range(3):
                    workspace[
                        env, qd_off + (dof_adr + d) * NV + (dof_adr + d)
                    ] = -damp
            else:
                workspace[env, qd_off + dof_adr * NV + dof_adr] = -damp

        # Compute RNE velocity derivative (subtracts from qDeriv)
        compute_rne_vel_derivative_gpu[
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
        ](env, state, model, workspace, implicit_base)

        # 6c. Form M_hat = M + armature - dt * qDeriv
        for j in range(NJOINT):
            var joint_off = model_joint_offset[NBODY](j)
            var jnt_type = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
            )
            var dof_adr = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR])
            )
            var arm = rebind[Scalar[DTYPE]](
                model[0, joint_off + JOINT_IDX_ARMATURE]
            )
            if jnt_type == JNT_FREE:
                for d in range(6):
                    var idx = M_idx + (dof_adr + d) * NV + (dof_adr + d)
                    workspace[env, idx] = (
                        rebind[Scalar[DTYPE]](workspace[env, idx]) + arm
                    )
            elif jnt_type == JNT_BALL:
                for d in range(3):
                    var idx = M_idx + (dof_adr + d) * NV + (dof_adr + d)
                    workspace[env, idx] = (
                        rebind[Scalar[DTYPE]](workspace[env, idx]) + arm
                    )
            else:
                var idx = M_idx + dof_adr * NV + dof_adr
                workspace[env, idx] = (
                    rebind[Scalar[DTYPE]](workspace[env, idx]) + arm
                )

        # Subtract dt * qDeriv from M (full matrix)
        for i in range(NV):
            for j_col in range(NV):
                var m_idx_ij = M_idx + i * NV + j_col
                var qd_val = rebind[Scalar[DTYPE]](
                    workspace[env, qd_off + i * NV + j_col]
                )
                workspace[env, m_idx_ij] = (
                    rebind[Scalar[DTYPE]](workspace[env, m_idx_ij])
                    - dt * qd_val
                )

        # 7. LU factorize M_hat and compute M_inv
        lu_factor_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](env, workspace)
        compute_M_inv_from_lu_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](
            env, workspace
        )

        # 8. Compute bias forces
        compute_bias_forces_rne_gpu[
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
        ](env, state, model, workspace)

        # 9. Compute f_net = qfrc - bias
        var qvel_off = qvel_offset[NQ, NV]()
        var qacc_off = qacc_offset[NQ, NV]()
        var qfrc_off = qfrc_offset[NQ, NV]()

        for i in range(NV):
            var qfrc = rebind[Scalar[DTYPE]](state[env, qfrc_off + i])
            var bias_val = rebind[Scalar[DTYPE]](workspace[env, bias_idx + i])
            workspace[env, fnet_idx + i] = qfrc - bias_val

        # 9b. Apply passive joint forces: damping + stiffness + frictionloss
        for j in range(NJOINT):
            var joint_off = model_joint_offset[NBODY](j)
            var jnt_type = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
            )
            var dof_adr = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR])
            )
            var damp_d = rebind[Scalar[DTYPE]](
                model[0, joint_off + JOINT_IDX_DAMPING]
            )
            if damp_d > Scalar[DTYPE](0):
                if jnt_type == JNT_FREE:
                    for d in range(6):
                        var v = rebind[Scalar[DTYPE]](
                            state[env, qvel_off + dof_adr + d]
                        )
                        var cur = rebind[Scalar[DTYPE]](
                            workspace[env, fnet_idx + dof_adr + d]
                        )
                        workspace[env, fnet_idx + dof_adr + d] = (
                            cur - damp_d * v
                        )
                elif jnt_type == JNT_BALL:
                    for d in range(3):
                        var v = rebind[Scalar[DTYPE]](
                            state[env, qvel_off + dof_adr + d]
                        )
                        var cur = rebind[Scalar[DTYPE]](
                            workspace[env, fnet_idx + dof_adr + d]
                        )
                        workspace[env, fnet_idx + dof_adr + d] = (
                            cur - damp_d * v
                        )
                else:
                    var v = rebind[Scalar[DTYPE]](
                        state[env, qvel_off + dof_adr]
                    )
                    var cur = rebind[Scalar[DTYPE]](
                        workspace[env, fnet_idx + dof_adr]
                    )
                    workspace[env, fnet_idx + dof_adr] = cur - damp_d * v

        # Stiffness + frictionloss
        var qpos_off = qpos_offset[NQ, NV]()
        for j in range(NJOINT):
            var joint_off = model_joint_offset[NBODY](j)
            var jnt_type = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
            )
            var dof_adr = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR])
            )
            var qpos_adr = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_QPOS_ADR])
            )
            var stiff = rebind[Scalar[DTYPE]](
                model[0, joint_off + JOINT_IDX_STIFFNESS]
            )
            var sref = rebind[Scalar[DTYPE]](
                model[0, joint_off + JOINT_IDX_SPRINGREF]
            )
            var floss = rebind[Scalar[DTYPE]](
                model[0, joint_off + JOINT_IDX_FRICTIONLOSS]
            )
            if stiff > Scalar[DTYPE](0):
                if jnt_type == JNT_FREE:
                    for d in range(6):
                        var qpos_d = rebind[Scalar[DTYPE]](
                            state[env, qpos_off + qpos_adr + d]
                        )
                        var cur = rebind[Scalar[DTYPE]](
                            workspace[env, fnet_idx + dof_adr + d]
                        )
                        workspace[env, fnet_idx + dof_adr + d] = cur - stiff * (
                            qpos_d - sref
                        )
                elif jnt_type == JNT_BALL:
                    for d in range(3):
                        var qpos_d = rebind[Scalar[DTYPE]](
                            state[env, qpos_off + qpos_adr + d]
                        )
                        var cur = rebind[Scalar[DTYPE]](
                            workspace[env, fnet_idx + dof_adr + d]
                        )
                        workspace[env, fnet_idx + dof_adr + d] = cur - stiff * (
                            qpos_d - sref
                        )
                else:
                    var qpos_d = rebind[Scalar[DTYPE]](
                        state[env, qpos_off + qpos_adr]
                    )
                    var cur = rebind[Scalar[DTYPE]](
                        workspace[env, fnet_idx + dof_adr]
                    )
                    workspace[env, fnet_idx + dof_adr] = cur - stiff * (
                        qpos_d - sref
                    )
            if floss > Scalar[DTYPE](0):
                comptime VEL_THRESH: Scalar[DTYPE] = 1e-4
                if jnt_type == JNT_FREE:
                    for d in range(6):
                        var v = rebind[Scalar[DTYPE]](
                            state[env, qvel_off + dof_adr + d]
                        )
                        var cur = rebind[Scalar[DTYPE]](
                            workspace[env, fnet_idx + dof_adr + d]
                        )
                        if v > VEL_THRESH:
                            workspace[env, fnet_idx + dof_adr + d] = cur - floss
                        elif v < -VEL_THRESH:
                            workspace[env, fnet_idx + dof_adr + d] = cur + floss
                elif jnt_type == JNT_BALL:
                    for d in range(3):
                        var v = rebind[Scalar[DTYPE]](
                            state[env, qvel_off + dof_adr + d]
                        )
                        var cur = rebind[Scalar[DTYPE]](
                            workspace[env, fnet_idx + dof_adr + d]
                        )
                        if v > VEL_THRESH:
                            workspace[env, fnet_idx + dof_adr + d] = cur - floss
                        elif v < -VEL_THRESH:
                            workspace[env, fnet_idx + dof_adr + d] = cur + floss
                else:
                    var v = rebind[Scalar[DTYPE]](
                        state[env, qvel_off + dof_adr]
                    )
                    var cur = rebind[Scalar[DTYPE]](
                        workspace[env, fnet_idx + dof_adr]
                    )
                    if v > VEL_THRESH:
                        workspace[env, fnet_idx + dof_adr] = cur - floss
                    elif v < -VEL_THRESH:
                        workspace[env, fnet_idx + dof_adr] = cur + floss

        # 10. LU solve for qacc
        lu_solve_workspace_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](env, workspace)

        for i in range(NV):
            var qacc_val = rebind[Scalar[DTYPE]](
                workspace[env, qacc_ws_idx + i]
            )
            state[env, qacc_off + i] = qacc_val

        # Write to constrained qacc slot for solver
        for i in range(NV):
            var qacc_val = rebind[Scalar[DTYPE]](
                workspace[env, qacc_ws_idx + i]
            )
            workspace[env, qacc_constrained_idx + i] = qacc_val

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
        """GPU finalize kernel — delegates to ImplicitFast (identical logic)."""

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
        CONE_TYPE: Int = ConeType.ELLIPTIC,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[DTYPE],
        mut model_buf: DeviceBuffer[DTYPE],
        mut workspace_buf: DeviceBuffer[DTYPE],
        dt: Scalar[DTYPE],
        gravity_z: Scalar[DTYPE],
        ground_z: Scalar[DTYPE],
    ) raises:
        """Perform one full implicit physics step on GPU."""
        comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS]()
        comptime MODEL_SIZE = model_size_with_invweight[
            NBODY, NJOINT, NV, NGEOM, NEQUALITY=MAX_EQUALITY
        ]()
        # Workspace = integrator_temps + M_inv + solver_ws + implicit_extra
        comptime SOLVER_WS = Self.SOLVER.solver_workspace_size[
            NV, MAX_CONTACTS
        ]()
        comptime WS_SIZE = (
            integrator_workspace_size[NV, NBODY]()
            + NV * NV
            + SOLVER_WS
            + implicit_extra_workspace_size[NV, NBODY]()
        )

        comptime THREADS = Self.SOLVER.solver_threads[
            NQ, NV, NBODY, NJOINT, MAX_CONTACTS
        ]()
        comptime ENV_BLOCKS = (BATCH + TPB - 1) // TPB
        comptime SOLVER_THREADS_BLOCKS = (THREADS + THREADS - 1) // THREADS
        comptime SOLVER_ENV_TPB = TPB // THREADS
        comptime SOLVER_ENV_BLOCKS = (
            BATCH + SOLVER_ENV_TPB - 1
        ) // SOLVER_ENV_TPB

        var state = LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ](state_buf.unsafe_ptr())

        var model_lt = LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ](model_buf.unsafe_ptr())

        var workspace = LayoutTensor[
            DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
        ](workspace_buf.unsafe_ptr())

        comptime kernel_wrapper = Self.step_kernel[
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
        ]

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            state,
            model_lt,
            workspace,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        comptime V_SIZE = _max_one[NV]()

        comptime solver_wrapper = Self.SOLVER.solve_gpu[
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
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
        ]

        ctx.enqueue_function[solver_wrapper, solver_wrapper](
            state,
            model_lt,
            workspace,
            grid_dim=(SOLVER_ENV_BLOCKS, SOLVER_THREADS_BLOCKS),
            block_dim=(SOLVER_ENV_TPB, THREADS),
        )

        comptime finalize_kernel_wrapper = Self.step_finalize_kernel[
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
        ]

        ctx.enqueue_function[finalize_kernel_wrapper, finalize_kernel_wrapper](
            state,
            model_lt,
            workspace,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
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
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
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
                DTYPE,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                BATCH,
                NGEOM,
                MAX_EQUALITY,
                CONE_TYPE,
            ](
                ctx,
                state_buf,
                model_buf,
                workspace_buf,
                dt,
                gravity_z,
                ground_z,
            )

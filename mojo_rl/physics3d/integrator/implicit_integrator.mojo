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

from std.math import sqrt, abs
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import thread_idx, block_idx, block_dim, barrier
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
from ..kinematics.quat_math import quat_normalize, quat_integrate, quat_rotate, gpu_quat_rotate
from ..dynamics.mass_matrix import (
    compute_mass_matrix,
    compute_mass_matrix_full,
    compute_mass_matrix_full_gpu,
    compute_mass_matrix_full_gpu_mt,
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
    compute_subtree_com,
    compute_cdof,
    compute_subtree_com_gpu,
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
    normalize_qpos_quaternions,
    normalize_qpos_quaternions_gpu,
)
from ..collision.broadphase_sap import (
    detect_contacts_auto,
    detect_contacts_auto_gpu,
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
from ..dynamics.cfrc_ext import compute_cfrc_ext
from ..dynamics.fluid_forces import compute_fluid_forces
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
    MODEL_META_IDX_DENSITY,
    MODEL_META_IDX_VISCOSITY,
    model_body_offset,
    JOINT_IDX_BODY_ID,
    BODY_IDX_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    BODY_IDX_PARENT,
    BODY_IDX_ROOTID,
    subtree_com_offset,
    xquat_offset,
    xvel_offset,
    xangvel_offset,
    xipos_offset,
    ws_cdof_offset,
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
    def step[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        MAX_TENDON: Int = 0,
        NSITE: Int = 0,
        NM: Int = 0,
        SPARSE: Bool = False,
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
            MAX_TENDON,
            NSITE,
        ],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
        verbose: Bool = False,
    ):
        """Execute one simulation step with full implicit integration.

        Uses the full RNE velocity derivative for qDeriv and LU factorization
        for the non-symmetric M_hat = M + armature - dt * qDeriv.

        Args:
            model: Static model configuration.
            data: Mutable simulation state.
            verbose: Whether to print debug information.
        """
        comptime assert (
            DTYPE.is_floating_point()
        ), "DTYPE must be floating point"
        var dt = model.timestep
        comptime M_SIZE = _max_one[NV * NV]()
        comptime V_SIZE = _max_one[NV]()
        comptime CDOF_SIZE = _max_one[NV * 6]()
        comptime CRB_SIZE = _max_one[NBODY * 10]()

        # 1. Forward kinematics
        forward_kinematics(model, data)
        compute_body_velocities(model, data)

        # 2. Collision detection
        detect_contacts_auto(model, data)

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

        # 3a. Compute subtree CoM (MuJoCo mj_comPos)
        var stcom_tmp = List[Scalar[DTYPE]](capacity=NBODY * 3)
        for _ in range(NBODY * 3):
            stcom_tmp.append(Scalar[DTYPE](0))
        compute_subtree_com(model, data, stcom_tmp)
        for sc_i in range(NBODY * 3):
            data.subtree_com[sc_i] = stcom_tmp[sc_i]
        data.has_subtree_com = True

        # 3b. Compute cdof (spatial motion axes per DOF)
        var cdof = List[Scalar[DTYPE]](capacity=CDOF_SIZE)
        for _ in range(CDOF_SIZE):
            cdof.append(Scalar[DTYPE](0))
        compute_cdof(model, data, cdof, stcom_tmp)

        # 4. Compute composite rigid body inertia
        var crb = List[Scalar[DTYPE]](capacity=CRB_SIZE)
        for _ in range(CRB_SIZE):
            crb.append(Scalar[DTYPE](0))
        for i in range(CRB_SIZE):
            crb[i] = Scalar[DTYPE](0)
        compute_composite_inertia(model, data, crb)

        # 5. Compute full mass matrix using CRBA
        var M = List[Scalar[DTYPE]](capacity=M_SIZE)
        for _ in range(M_SIZE):
            M.append(Scalar[DTYPE](0))
        compute_mass_matrix_full(model, data, cdof, crb, M)

        # 5b. Compute FULL qDeriv and modify mass matrix
        # M_hat = M + armature - dt * qDeriv
        #
        # qDeriv has two parts:
        # (a) Passive force derivative: qDeriv[i,i] = -damping[i]  (diagonal)
        # (b) RNE velocity derivative: d(bias)/d(qvel)  (dense, non-symmetric)
        #
        # Initialize qDeriv with passive damping (diagonal)
        var qDeriv = List[Scalar[DTYPE]](capacity=M_SIZE)
        for _ in range(M_SIZE):
            qDeriv.append(Scalar[DTYPE](0))

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
        # Note: compute_rne_vel_derivative internally converts cdof to body-origin
        # convention. The conversion is validated with xipos-based cdof (no subtree_com).
        # Use a separate cdof without subtree_com for the RNE derivative to match
        # the validated convention and produce correct qDeriv values.
        var cdof_rne = List[Scalar[DTYPE]](capacity=CDOF_SIZE)
        for _ in range(CDOF_SIZE):
            cdof_rne.append(Scalar[DTYPE](0))
        compute_cdof(model, data, cdof_rne)
        compute_rne_vel_derivative(model, data, cdof_rne, qDeriv)

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
        var M_lu = List[Scalar[DTYPE]](capacity=M_SIZE)
        for i in range(M_SIZE):
            M_lu.append(M[i])  # Copy for LU (in-place)

        var piv = List[Int](capacity=V_SIZE)
        for i in range(NV):
            piv.append(i)
        lu_factor[DTYPE, NV, M_SIZE, V_SIZE](M_lu, piv)

        var bias = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            bias.append(Scalar[DTYPE](0))
        compute_bias_forces_rne[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
        ](model, data, cdof, bias)

        var f_net = List[Scalar[DTYPE]](capacity=V_SIZE)
        for i in range(NV):
            f_net.append(data.qfrc[i] - bias[i])

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

        # 6c. Fluid forces: viscous + pressure drag (disabled when density=viscosity=0)
        compute_fluid_forces[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
        ](model, data, cdof, f_net, stcom_tmp)

        # qacc = M_hat^-1 * f_net via LU solve
        var qacc = List[Scalar[DTYPE]](capacity=V_SIZE)
        for _ in range(V_SIZE):
            qacc.append(Scalar[DTYPE](0))
        lu_solve[DTYPE, NV, M_SIZE, V_SIZE](M_lu, piv, f_net, qacc)

        # 7. Compute full M_inv from LU factors for constraint solver
        var M_inv = List[Scalar[DTYPE]](capacity=M_SIZE)
        for _ in range(M_SIZE):
            M_inv.append(Scalar[DTYPE](0))
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
        comptime MAX_ROWS = 11 * MAX_CONTACTS + 2 * NJOINT + 6 * MAX_EQUALITY + MAX_TENDON
        var constraints = ConstraintData[DTYPE, MAX_ROWS, NV]()
        build_constraints[CONE_TYPE=CONE_TYPE, MAX_TENDON=MAX_TENDON](
            model, data, cdof, M_inv, dt, constraints
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

        # Compute cfrc_ext: contact forces per body in subtree CoM frame
        compute_cfrc_ext[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
        ](model, data)

        # 9. Integrate: semi-implicit (symplectic) Euler
        # MuJoCo 3.3.6 uses symplectic Euler for ALL integrators:
        #   qvel += dt * qacc
        #   qpos += dt * qvel_new

        # Velocity update
        for i in range(NV):
            data.qacc[i] = qacc[i]
            data.qvel[i] = data.qvel[i] + qacc[i] * dt

        # Position update: symplectic Euler (quaternion-aware)
        for j in range(model.num_joints):
            var joint = model.joints[j]
            var qpos_adr = joint.qpos_adr
            var dof_adr = joint.dof_adr

            if joint.jnt_type == JNT_FREE:
                for d in range(3):
                    data.qpos[qpos_adr + d] = (
                        data.qpos[qpos_adr + d] + data.qvel[dof_adr + d] * dt
                    )
                var qx = data.qpos[qpos_adr + 3]
                var qy = data.qpos[qpos_adr + 4]
                var qz = data.qpos[qpos_adr + 5]
                var qw = data.qpos[qpos_adr + 6]
                var wx = data.qvel[dof_adr + 3]
                var wy = data.qvel[dof_adr + 4]
                var wz = data.qvel[dof_adr + 5]
                var result = quat_integrate(qx, qy, qz, qw, wx, wy, wz, dt)
                var norm = quat_normalize(
                    result[0], result[1], result[2], result[3]
                )
                data.qpos[qpos_adr + 3] = norm[0]
                data.qpos[qpos_adr + 4] = norm[1]
                data.qpos[qpos_adr + 5] = norm[2]
                data.qpos[qpos_adr + 6] = norm[3]

            elif joint.jnt_type == JNT_HINGE or joint.jnt_type == JNT_SLIDE:
                data.qpos[qpos_adr] = (
                    data.qpos[qpos_adr] + data.qvel[dof_adr] * dt
                )

        # 10. Normalize quaternions (handles remaining cases like BALL)
        normalize_qpos_quaternions(model, data)

        if verbose:
            print("  [POST-INTEGRATION]")
            print("    qvel_new:", end="")
            for i in range(NV):
                print(" ", Float64(data.qvel[i]), end="")
            print("")

    @staticmethod
    def simulate[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int = 0,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        MAX_TENDON: Int = 0,
        NSITE: Int = 0,
        NM: Int = 0,
        SPARSE: Bool = False,
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
            MAX_TENDON,
            NSITE,
        ],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
        num_steps: Int,
    ):
        """Run simulation for multiple steps on CPU."""
        comptime assert (
            DTYPE.is_floating_point()
        ), "DTYPE must be floating point"
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
                MAX_TENDON,
                NSITE,
                NM,
                SPARSE,
            ](model, data)

    # =========================================================================
    # GPU Methods — Full implicit with RNE velocity derivative
    # =========================================================================

    @always_inline
    @staticmethod
    def step_kernel[
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
        MAX_TENDON: Int = 0,
        NSITE: Int = 0,
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
        detect_contacts_auto_gpu[
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
            MAX_EQUALITY,
            MAX_TENDON,
            NSITE,
        ](env, state, model)

        # 3a. Compute subtree_com
        compute_subtree_com_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
            STATE_SIZE, MODEL_SIZE, BATCH,
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

        # 7. LU factorize M_hat, conditionally compute M_inv
        lu_factor_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](env, workspace)
        comptime if Self.SOLVER.NEEDS_M_INV:
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

        # 9c. Fluid forces: inertia-box viscous + pressure drag (GPU)
        var model_meta_off_fl = model_metadata_offset[NBODY, NJOINT]()
        var rho_fl = rebind[Scalar[DTYPE]](
            model[0, model_meta_off_fl + MODEL_META_IDX_DENSITY]
        )
        var mu_fl = rebind[Scalar[DTYPE]](
            model[0, model_meta_off_fl + MODEL_META_IDX_VISCOSITY]
        )
        if rho_fl > Scalar[DTYPE](0) or mu_fl > Scalar[DTYPE](0):
            comptime PI_FL: Scalar[DTYPE] = 3.14159265358979323846
            comptime xquat_off_fl = xquat_offset[NQ, NV, NBODY]()
            comptime xvel_off_fl = xvel_offset[NQ, NV, NBODY]()
            comptime xangvel_off_fl = xangvel_offset[NQ, NV, NBODY]()
            comptime xipos_off_fl = xipos_offset[NQ, NV, NBODY]()
            comptime cdof_off_fl = ws_cdof_offset()

            for b in range(1, NBODY):
                var body_off_b = model_body_offset(b)
                var mass_b = rebind[Scalar[DTYPE]](model[0, body_off_b + BODY_IDX_MASS])
                if mass_b <= Scalar[DTYPE](1e-10):
                    continue
                var Ixx = rebind[Scalar[DTYPE]](model[0, body_off_b + BODY_IDX_IXX])
                var Iyy = rebind[Scalar[DTYPE]](model[0, body_off_b + BODY_IDX_IYY])
                var Izz = rebind[Scalar[DTYPE]](model[0, body_off_b + BODY_IDX_IZZ])
                var bx2 = Scalar[DTYPE](6) * (Iyy + Izz - Ixx) / mass_b
                var by2 = Scalar[DTYPE](6) * (Ixx + Izz - Iyy) / mass_b
                var bz2 = Scalar[DTYPE](6) * (Ixx + Iyy - Izz) / mass_b
                var bx = sqrt(max(bx2, Scalar[DTYPE](0)))
                var by = sqrt(max(by2, Scalar[DTYPE](0)))
                var bz = sqrt(max(bz2, Scalar[DTYPE](0)))
                var vx_w = rebind[Scalar[DTYPE]](state[env, xvel_off_fl + b * 3 + 0])
                var vy_w = rebind[Scalar[DTYPE]](state[env, xvel_off_fl + b * 3 + 1])
                var vz_w = rebind[Scalar[DTYPE]](state[env, xvel_off_fl + b * 3 + 2])
                var wx_w = rebind[Scalar[DTYPE]](state[env, xangvel_off_fl + b * 3 + 0])
                var wy_w = rebind[Scalar[DTYPE]](state[env, xangvel_off_fl + b * 3 + 1])
                var wz_w = rebind[Scalar[DTYPE]](state[env, xangvel_off_fl + b * 3 + 2])
                var qx_b = rebind[Scalar[DTYPE]](state[env, xquat_off_fl + b * 4 + 0])
                var qy_b = rebind[Scalar[DTYPE]](state[env, xquat_off_fl + b * 4 + 1])
                var qz_b = rebind[Scalar[DTYPE]](state[env, xquat_off_fl + b * 4 + 2])
                var qw_b = rebind[Scalar[DTYPE]](state[env, xquat_off_fl + b * 4 + 3])
                var vloc_b = gpu_quat_rotate[DTYPE](-qx_b, -qy_b, -qz_b, qw_b, vx_w, vy_w, vz_w)
                var wloc_b = gpu_quat_rotate[DTYPE](-qx_b, -qy_b, -qz_b, qw_b, wx_w, wy_w, wz_w)
                var vx = vloc_b[0]; var vy = vloc_b[1]; var vz = vloc_b[2]
                var wx = wloc_b[0]; var wy = wloc_b[1]; var wz = wloc_b[2]
                var diam = (bx + by + bz) / Scalar[DTYPE](3)
                var lfx = Scalar[DTYPE](0); var lfy = Scalar[DTYPE](0); var lfz = Scalar[DTYPE](0)
                var ltx = Scalar[DTYPE](0); var lty = Scalar[DTYPE](0); var ltz = Scalar[DTYPE](0)
                if mu_fl > Scalar[DTYPE](0):
                    var visc_lin = Scalar[DTYPE](3) * PI_FL * diam * mu_fl
                    lfx = lfx - visc_lin * vx; lfy = lfy - visc_lin * vy; lfz = lfz - visc_lin * vz
                    var d3 = diam * diam * diam
                    var visc_ang = PI_FL * d3 * mu_fl
                    ltx = ltx - visc_ang * wx; lty = lty - visc_ang * wy; ltz = ltz - visc_ang * wz
                if rho_fl > Scalar[DTYPE](0):
                    var half_rho = Scalar[DTYPE](0.5) * rho_fl
                    lfx = lfx - half_rho * by * bz * abs(vx) * vx
                    lfy = lfy - half_rho * bx * bz * abs(vy) * vy
                    lfz = lfz - half_rho * bx * by * abs(vz) * vz
                    var bx4 = bx * bx * bx * bx; var by4 = by * by * by * by; var bz4 = bz * bz * bz * bz
                    ltx = ltx - rho_fl * bx * (by4 + bz4) * abs(wx) * wx / Scalar[DTYPE](64)
                    lty = lty - rho_fl * by * (bx4 + bz4) * abs(wy) * wy / Scalar[DTYPE](64)
                    ltz = ltz - rho_fl * bz * (bx4 + by4) * abs(wz) * wz / Scalar[DTYPE](64)
                var fw_b = gpu_quat_rotate[DTYPE](qx_b, qy_b, qz_b, qw_b, lfx, lfy, lfz)
                var tw_b = gpu_quat_rotate[DTYPE](qx_b, qy_b, qz_b, qw_b, ltx, lty, ltz)
                var fx_w = fw_b[0]; var fy_w = fw_b[1]; var fz_w = fw_b[2]
                var tx_w = tw_b[0]; var ty_w = tw_b[1]; var tz_w = tw_b[2]
                # Transport wrench to subtree_com[rootid] (cdof reference point)
                comptime stcom_off_fl = subtree_com_offset[NQ, NV, NBODY, MAX_CONTACTS]()
                var px_b = rebind[Scalar[DTYPE]](state[env, xipos_off_fl + b * 3 + 0])
                var py_b = rebind[Scalar[DTYPE]](state[env, xipos_off_fl + b * 3 + 1])
                var pz_b = rebind[Scalar[DTYPE]](state[env, xipos_off_fl + b * 3 + 2])
                var rootid_b = Int(rebind[Scalar[DTYPE]](model[0, body_off_b + BODY_IDX_ROOTID]))
                var dx_b = px_b - rebind[Scalar[DTYPE]](state[env, stcom_off_fl + rootid_b * 3 + 0])
                var dy_b = py_b - rebind[Scalar[DTYPE]](state[env, stcom_off_fl + rootid_b * 3 + 1])
                var dz_b = pz_b - rebind[Scalar[DTYPE]](state[env, stcom_off_fl + rootid_b * 3 + 2])
                var tau_ox = tx_w + dy_b * fz_w - dz_b * fy_w
                var tau_oy = ty_w + dz_b * fx_w - dx_b * fz_w
                var tau_oz = tz_w + dx_b * fy_w - dy_b * fx_w
                var anc = b
                while anc > 0:
                    for j2 in range(NJOINT):
                        var jo2 = model_joint_offset[NBODY](j2)
                        var bid2 = Int(rebind[Scalar[DTYPE]](model[0, jo2 + JOINT_IDX_BODY_ID]))
                        if bid2 != anc:
                            continue
                        var jt2 = Int(rebind[Scalar[DTYPE]](model[0, jo2 + JOINT_IDX_TYPE]))
                        var da2 = Int(rebind[Scalar[DTYPE]](model[0, jo2 + JOINT_IDX_DOF_ADR]))
                        var nd2 = 1
                        if jt2 == JNT_FREE:
                            nd2 = 6
                        elif jt2 == JNT_BALL:
                            nd2 = 3
                        for d2 in range(nd2):
                            var di2 = da2 + d2
                            var ca0 = rebind[Scalar[DTYPE]](workspace[env, cdof_off_fl + di2 * 6 + 0])
                            var ca1 = rebind[Scalar[DTYPE]](workspace[env, cdof_off_fl + di2 * 6 + 1])
                            var ca2 = rebind[Scalar[DTYPE]](workspace[env, cdof_off_fl + di2 * 6 + 2])
                            var cl0 = rebind[Scalar[DTYPE]](workspace[env, cdof_off_fl + di2 * 6 + 3])
                            var cl1 = rebind[Scalar[DTYPE]](workspace[env, cdof_off_fl + di2 * 6 + 4])
                            var cl2 = rebind[Scalar[DTYPE]](workspace[env, cdof_off_fl + di2 * 6 + 5])
                            var cur2 = rebind[Scalar[DTYPE]](workspace[env, fnet_idx + di2])
                            workspace[env, fnet_idx + di2] = (
                                cur2 + cl0 * fx_w + cl1 * fy_w + cl2 * fz_w
                                + ca0 * tau_ox + ca1 * tau_oy + ca2 * tau_oz
                            )
                    var anc_off = model_body_offset(anc)
                    anc = Int(rebind[Scalar[DTYPE]](model[0, anc_off + BODY_IDX_PARENT]))

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
    def step_kernel_mt[
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
        MAX_TENDON: Int = 0,
        NSITE: Int = 0,
        STEP_THREADS: Int = 1,
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
        """Multi-threaded full implicit step kernel (pre-solver).

        Uses 2D blocks (envs, STEP_THREADS) to parallelize mass matrix
        and M_hat formation across STEP_THREADS threads per environment.
        """

        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        var tid = Int(thread_idx.y)
        if env >= BATCH:
            # Invalid envs must still hit all barriers to avoid deadlock
            pass
        var valid_env = env < BATCH

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

        # =====================================================================
        # SERIAL PHASE 1: FK, body velocities, contacts, subtree_com, cdof, CRB
        # =====================================================================
        if tid == 0 and valid_env:
            forward_kinematics_gpu[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
                STATE_SIZE, MODEL_SIZE, BATCH,
            ](env, state, model)

            compute_body_velocities_gpu[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
                STATE_SIZE, MODEL_SIZE, BATCH,
            ](env, state, model)

            detect_contacts_auto_gpu[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
                STATE_SIZE, MODEL_SIZE, BATCH, NGEOM,
                MAX_EQUALITY, MAX_TENDON, NSITE,
            ](env, state, model)

            compute_subtree_com_gpu[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
                STATE_SIZE, MODEL_SIZE, BATCH,
            ](env, state, model)

            compute_cdof_gpu[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
                STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
            ](env, state, model, workspace)

            compute_composite_inertia_gpu[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
                STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
            ](env, state, model, workspace)

        barrier()

        # =====================================================================
        # PARALLEL PHASE 1: Mass matrix computation
        # =====================================================================
        if valid_env:
            compute_mass_matrix_full_gpu_mt[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
                STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
            ](env, tid, STEP_THREADS, state, model, workspace)

        barrier()

        # =====================================================================
        # SERIAL PHASE 2: qDeriv, RNE vel derivative, armature
        # =====================================================================
        if tid == 0 and valid_env:
            # Initialize qDeriv to zero
            for i in range(NV * NV):
                workspace[env, qd_off + i] = Scalar[DTYPE](0)

            # Set passive damping diagonal
            var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
            var dt = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_TIMESTEP]
            )

            for j in range(NJOINT):
                var joint_off = model_joint_offset[NBODY](j)
                var jnt_type = Int(
                    rebind[Scalar[DTYPE]](
                        model[0, joint_off + JOINT_IDX_TYPE]
                    )
                )
                var dof_adr = Int(
                    rebind[Scalar[DTYPE]](
                        model[0, joint_off + JOINT_IDX_DOF_ADR]
                    )
                )
                var damp = rebind[Scalar[DTYPE]](
                    model[0, joint_off + JOINT_IDX_DAMPING]
                )
                if jnt_type == JNT_FREE:
                    for d in range(6):
                        workspace[
                            env,
                            qd_off + (dof_adr + d) * NV + (dof_adr + d),
                        ] = -damp
                elif jnt_type == JNT_BALL:
                    for d in range(3):
                        workspace[
                            env,
                            qd_off + (dof_adr + d) * NV + (dof_adr + d),
                        ] = -damp
                else:
                    workspace[env, qd_off + dof_adr * NV + dof_adr] = -damp

            # RNE velocity derivative
            compute_rne_vel_derivative_gpu[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
                STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE, NGEOM,
            ](env, state, model, workspace, implicit_base)

            # Armature to M diagonal
            for j in range(NJOINT):
                var joint_off = model_joint_offset[NBODY](j)
                var jnt_type = Int(
                    rebind[Scalar[DTYPE]](
                        model[0, joint_off + JOINT_IDX_TYPE]
                    )
                )
                var dof_adr = Int(
                    rebind[Scalar[DTYPE]](
                        model[0, joint_off + JOINT_IDX_DOF_ADR]
                    )
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

        barrier()

        # =====================================================================
        # PARALLEL PHASE 2: M_hat -= dt * qDeriv (row-strided)
        # =====================================================================
        if valid_env:
            var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
            var dt = rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_TIMESTEP]
            )
            for i in range(tid, NV, STEP_THREADS):
                for j_col in range(NV):
                    var m_idx_ij = M_idx + i * NV + j_col
                    var qd_val = rebind[Scalar[DTYPE]](
                        workspace[env, qd_off + i * NV + j_col]
                    )
                    workspace[env, m_idx_ij] = (
                        rebind[Scalar[DTYPE]](workspace[env, m_idx_ij])
                        - dt * qd_val
                    )

        barrier()

        # =====================================================================
        # SERIAL PHASE 3: LU factor, M_inv, bias forces
        # =====================================================================
        if tid == 0 and valid_env:
            lu_factor_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](env, workspace)
            comptime if Self.SOLVER.NEEDS_M_INV:
                compute_M_inv_from_lu_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](
                    env, workspace
                )

            compute_bias_forces_rne_gpu[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
                STATE_SIZE, MODEL_SIZE, BATCH, WS_SIZE,
            ](env, state, model, workspace)

        barrier()

        # =====================================================================
        # PARALLEL PHASE 3: f_net = qfrc - bias (strided)
        # =====================================================================
        if valid_env:
            var qfrc_off = qfrc_offset[NQ, NV]()
            for i in range(tid, NV, STEP_THREADS):
                var qfrc = rebind[Scalar[DTYPE]](state[env, qfrc_off + i])
                var bias_val = rebind[Scalar[DTYPE]](
                    workspace[env, bias_idx + i]
                )
                workspace[env, fnet_idx + i] = qfrc - bias_val

        barrier()

        # =====================================================================
        # SERIAL PHASE 4: Passive forces, LU solve, write qacc
        # =====================================================================
        if tid == 0 and valid_env:
            var qvel_off = qvel_offset[NQ, NV]()
            var qacc_off = qacc_offset[NQ, NV]()
            var qfrc_off = qfrc_offset[NQ, NV]()

            # Passive forces: damping
            for j in range(NJOINT):
                var joint_off = model_joint_offset[NBODY](j)
                var jnt_type = Int(
                    rebind[Scalar[DTYPE]](
                        model[0, joint_off + JOINT_IDX_TYPE]
                    )
                )
                var dof_adr = Int(
                    rebind[Scalar[DTYPE]](
                        model[0, joint_off + JOINT_IDX_DOF_ADR]
                    )
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
                    rebind[Scalar[DTYPE]](
                        model[0, joint_off + JOINT_IDX_TYPE]
                    )
                )
                var dof_adr = Int(
                    rebind[Scalar[DTYPE]](
                        model[0, joint_off + JOINT_IDX_DOF_ADR]
                    )
                )
                var qpos_adr = Int(
                    rebind[Scalar[DTYPE]](
                        model[0, joint_off + JOINT_IDX_QPOS_ADR]
                    )
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
                            workspace[env, fnet_idx + dof_adr + d] = (
                                cur - stiff * (qpos_d - sref)
                            )
                    elif jnt_type == JNT_BALL:
                        for d in range(3):
                            var qpos_d = rebind[Scalar[DTYPE]](
                                state[env, qpos_off + qpos_adr + d]
                            )
                            var cur = rebind[Scalar[DTYPE]](
                                workspace[env, fnet_idx + dof_adr + d]
                            )
                            workspace[env, fnet_idx + dof_adr + d] = (
                                cur - stiff * (qpos_d - sref)
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
                                workspace[
                                    env, fnet_idx + dof_adr + d
                                ] = cur - floss
                            elif v < -VEL_THRESH:
                                workspace[
                                    env, fnet_idx + dof_adr + d
                                ] = cur + floss
                    elif jnt_type == JNT_BALL:
                        for d in range(3):
                            var v = rebind[Scalar[DTYPE]](
                                state[env, qvel_off + dof_adr + d]
                            )
                            var cur = rebind[Scalar[DTYPE]](
                                workspace[env, fnet_idx + dof_adr + d]
                            )
                            if v > VEL_THRESH:
                                workspace[
                                    env, fnet_idx + dof_adr + d
                                ] = cur - floss
                            elif v < -VEL_THRESH:
                                workspace[
                                    env, fnet_idx + dof_adr + d
                                ] = cur + floss
                    else:
                        var v = rebind[Scalar[DTYPE]](
                            state[env, qvel_off + dof_adr]
                        )
                        var cur = rebind[Scalar[DTYPE]](
                            workspace[env, fnet_idx + dof_adr]
                        )
                        if v > VEL_THRESH:
                            workspace[
                                env, fnet_idx + dof_adr
                            ] = cur - floss
                        elif v < -VEL_THRESH:
                            workspace[
                                env, fnet_idx + dof_adr
                            ] = cur + floss

            # 9c. Fluid forces: inertia-box viscous + pressure drag (GPU)
            var model_meta_off_fl = model_metadata_offset[NBODY, NJOINT]()
            var rho_fl = rebind[Scalar[DTYPE]](
                model[0, model_meta_off_fl + MODEL_META_IDX_DENSITY]
            )
            var mu_fl = rebind[Scalar[DTYPE]](
                model[0, model_meta_off_fl + MODEL_META_IDX_VISCOSITY]
            )
            if rho_fl > Scalar[DTYPE](0) or mu_fl > Scalar[DTYPE](0):
                comptime PI_FL: Scalar[DTYPE] = 3.14159265358979323846
                comptime xquat_off_fl = xquat_offset[NQ, NV, NBODY]()
                comptime xvel_off_fl = xvel_offset[NQ, NV, NBODY]()
                comptime xangvel_off_fl = xangvel_offset[NQ, NV, NBODY]()
                comptime xipos_off_fl = xipos_offset[NQ, NV, NBODY]()
                comptime cdof_off_fl = ws_cdof_offset()

                for b in range(1, NBODY):
                    var body_off_b = model_body_offset(b)
                    var mass_b = rebind[Scalar[DTYPE]](model[0, body_off_b + BODY_IDX_MASS])
                    if mass_b <= Scalar[DTYPE](1e-10):
                        continue
                    var Ixx = rebind[Scalar[DTYPE]](model[0, body_off_b + BODY_IDX_IXX])
                    var Iyy = rebind[Scalar[DTYPE]](model[0, body_off_b + BODY_IDX_IYY])
                    var Izz = rebind[Scalar[DTYPE]](model[0, body_off_b + BODY_IDX_IZZ])
                    var bx2 = Scalar[DTYPE](6) * (Iyy + Izz - Ixx) / mass_b
                    var by2 = Scalar[DTYPE](6) * (Ixx + Izz - Iyy) / mass_b
                    var bz2 = Scalar[DTYPE](6) * (Ixx + Iyy - Izz) / mass_b
                    var bx = sqrt(max(bx2, Scalar[DTYPE](0)))
                    var by = sqrt(max(by2, Scalar[DTYPE](0)))
                    var bz = sqrt(max(bz2, Scalar[DTYPE](0)))
                    var vx_w = rebind[Scalar[DTYPE]](state[env, xvel_off_fl + b * 3 + 0])
                    var vy_w = rebind[Scalar[DTYPE]](state[env, xvel_off_fl + b * 3 + 1])
                    var vz_w = rebind[Scalar[DTYPE]](state[env, xvel_off_fl + b * 3 + 2])
                    var wx_w = rebind[Scalar[DTYPE]](state[env, xangvel_off_fl + b * 3 + 0])
                    var wy_w = rebind[Scalar[DTYPE]](state[env, xangvel_off_fl + b * 3 + 1])
                    var wz_w = rebind[Scalar[DTYPE]](state[env, xangvel_off_fl + b * 3 + 2])
                    var qx_b = rebind[Scalar[DTYPE]](state[env, xquat_off_fl + b * 4 + 0])
                    var qy_b = rebind[Scalar[DTYPE]](state[env, xquat_off_fl + b * 4 + 1])
                    var qz_b = rebind[Scalar[DTYPE]](state[env, xquat_off_fl + b * 4 + 2])
                    var qw_b = rebind[Scalar[DTYPE]](state[env, xquat_off_fl + b * 4 + 3])
                    var vloc_b = gpu_quat_rotate[DTYPE](-qx_b, -qy_b, -qz_b, qw_b, vx_w, vy_w, vz_w)
                    var wloc_b = gpu_quat_rotate[DTYPE](-qx_b, -qy_b, -qz_b, qw_b, wx_w, wy_w, wz_w)
                    var vx = vloc_b[0]; var vy = vloc_b[1]; var vz = vloc_b[2]
                    var wx = wloc_b[0]; var wy = wloc_b[1]; var wz = wloc_b[2]
                    var diam = (bx + by + bz) / Scalar[DTYPE](3)
                    var lfx = Scalar[DTYPE](0); var lfy = Scalar[DTYPE](0); var lfz = Scalar[DTYPE](0)
                    var ltx = Scalar[DTYPE](0); var lty = Scalar[DTYPE](0); var ltz = Scalar[DTYPE](0)
                    if mu_fl > Scalar[DTYPE](0):
                        var visc_lin = Scalar[DTYPE](3) * PI_FL * diam * mu_fl
                        lfx = lfx - visc_lin * vx; lfy = lfy - visc_lin * vy; lfz = lfz - visc_lin * vz
                        var d3 = diam * diam * diam
                        var visc_ang = PI_FL * d3 * mu_fl
                        ltx = ltx - visc_ang * wx; lty = lty - visc_ang * wy; ltz = ltz - visc_ang * wz
                    if rho_fl > Scalar[DTYPE](0):
                        var half_rho = Scalar[DTYPE](0.5) * rho_fl
                        lfx = lfx - half_rho * by * bz * abs(vx) * vx
                        lfy = lfy - half_rho * bx * bz * abs(vy) * vy
                        lfz = lfz - half_rho * bx * by * abs(vz) * vz
                        var bx4 = bx * bx * bx * bx; var by4 = by * by * by * by; var bz4 = bz * bz * bz * bz
                        ltx = ltx - rho_fl * bx * (by4 + bz4) * abs(wx) * wx / Scalar[DTYPE](64)
                        lty = lty - rho_fl * by * (bx4 + bz4) * abs(wy) * wy / Scalar[DTYPE](64)
                        ltz = ltz - rho_fl * bz * (bx4 + by4) * abs(wz) * wz / Scalar[DTYPE](64)
                    var fw_b = gpu_quat_rotate[DTYPE](qx_b, qy_b, qz_b, qw_b, lfx, lfy, lfz)
                    var tw_b = gpu_quat_rotate[DTYPE](qx_b, qy_b, qz_b, qw_b, ltx, lty, ltz)
                    var fx_w = fw_b[0]; var fy_w = fw_b[1]; var fz_w = fw_b[2]
                    var tx_w = tw_b[0]; var ty_w = tw_b[1]; var tz_w = tw_b[2]
                    # Transport wrench to subtree_com[rootid] (cdof reference point)
                    comptime stcom_off_fl = subtree_com_offset[NQ, NV, NBODY, MAX_CONTACTS]()
                    var px_b = rebind[Scalar[DTYPE]](state[env, xipos_off_fl + b * 3 + 0])
                    var py_b = rebind[Scalar[DTYPE]](state[env, xipos_off_fl + b * 3 + 1])
                    var pz_b = rebind[Scalar[DTYPE]](state[env, xipos_off_fl + b * 3 + 2])
                    var rootid_b = Int(rebind[Scalar[DTYPE]](model[0, body_off_b + BODY_IDX_ROOTID]))
                    var dx_b = px_b - rebind[Scalar[DTYPE]](state[env, stcom_off_fl + rootid_b * 3 + 0])
                    var dy_b = py_b - rebind[Scalar[DTYPE]](state[env, stcom_off_fl + rootid_b * 3 + 1])
                    var dz_b = pz_b - rebind[Scalar[DTYPE]](state[env, stcom_off_fl + rootid_b * 3 + 2])
                    var tau_ox = tx_w + dy_b * fz_w - dz_b * fy_w
                    var tau_oy = ty_w + dz_b * fx_w - dx_b * fz_w
                    var tau_oz = tz_w + dx_b * fy_w - dy_b * fx_w
                    var anc = b
                    while anc > 0:
                        for j2 in range(NJOINT):
                            var jo2 = model_joint_offset[NBODY](j2)
                            var bid2 = Int(rebind[Scalar[DTYPE]](model[0, jo2 + JOINT_IDX_BODY_ID]))
                            if bid2 != anc:
                                continue
                            var jt2 = Int(rebind[Scalar[DTYPE]](model[0, jo2 + JOINT_IDX_TYPE]))
                            var da2 = Int(rebind[Scalar[DTYPE]](model[0, jo2 + JOINT_IDX_DOF_ADR]))
                            var nd2 = 1
                            if jt2 == JNT_FREE:
                                nd2 = 6
                            elif jt2 == JNT_BALL:
                                nd2 = 3
                            for d2 in range(nd2):
                                var di2 = da2 + d2
                                var ca0 = rebind[Scalar[DTYPE]](workspace[env, cdof_off_fl + di2 * 6 + 0])
                                var ca1 = rebind[Scalar[DTYPE]](workspace[env, cdof_off_fl + di2 * 6 + 1])
                                var ca2 = rebind[Scalar[DTYPE]](workspace[env, cdof_off_fl + di2 * 6 + 2])
                                var cl0 = rebind[Scalar[DTYPE]](workspace[env, cdof_off_fl + di2 * 6 + 3])
                                var cl1 = rebind[Scalar[DTYPE]](workspace[env, cdof_off_fl + di2 * 6 + 4])
                                var cl2 = rebind[Scalar[DTYPE]](workspace[env, cdof_off_fl + di2 * 6 + 5])
                                var cur2 = rebind[Scalar[DTYPE]](workspace[env, fnet_idx + di2])
                                workspace[env, fnet_idx + di2] = (
                                    cur2 + cl0 * fx_w + cl1 * fy_w + cl2 * fz_w
                                    + ca0 * tau_ox + ca1 * tau_oy + ca2 * tau_oz
                                )
                        var anc_off = model_body_offset(anc)
                        anc = Int(rebind[Scalar[DTYPE]](model[0, anc_off + BODY_IDX_PARENT]))

            # LU solve
            lu_solve_workspace_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](
                env, workspace
            )

            # Write qacc to state and constrained slot
            for i in range(NV):
                var qacc_val = rebind[Scalar[DTYPE]](
                    workspace[env, qacc_ws_idx + i]
                )
                state[env, qacc_off + i] = qacc_val
                workspace[env, qacc_constrained_idx + i] = qacc_val

    @always_inline
    @staticmethod
    def step_finalize_kernel[
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
    def step_gpu[
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
        MAX_TENDON: Int = 0,
        NSITE: Int = 0,
        NM: Int = 0,
        SPARSE: Bool = False,
        STEP_THREADS: Int = 1,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[DTYPE],
        mut model_buf: DeviceBuffer[DTYPE],
        mut workspace_buf: DeviceBuffer[DTYPE],
    ) raises:
        """Perform one full implicit physics step on GPU."""
        comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS, NSITE]()
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
        ](state_buf)

        var model_lt = LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ](model_buf)

        var workspace = LayoutTensor[
            DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
        ](workspace_buf)

        comptime if STEP_THREADS > 1:
            comptime STEP_ENV_TPB = TPB // STEP_THREADS
            comptime STEP_ENV_BLOCKS = (
                BATCH + STEP_ENV_TPB - 1
            ) // STEP_ENV_TPB

            comptime mt_kernel_wrapper = Self.step_kernel_mt[
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
                MAX_EQUALITY,
                CONE_TYPE,
                MAX_TENDON,
                NSITE,
                STEP_THREADS,
            ]

            ctx.enqueue_function[mt_kernel_wrapper, mt_kernel_wrapper](
                state,
                model_lt,
                workspace,
                grid_dim=(STEP_ENV_BLOCKS, 1),
                block_dim=(STEP_ENV_TPB, STEP_THREADS),
            )
        else:
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
            MAX_TENDON,
            NSITE,
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
    def simulate_gpu[
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
        MAX_TENDON: Int = 0,
        NSITE: Int = 0,
        NM: Int = 0,
        SPARSE: Bool = False,
        STEP_THREADS: Int = 1,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[DTYPE],
        mut model_buf: DeviceBuffer[DTYPE],
        mut workspace_buf: DeviceBuffer[DTYPE],
        num_steps: Int,
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
                MAX_TENDON,
                NSITE,
                NM,
                SPARSE,
            ](
                ctx,
                state_buf,
                model_buf,
                workspace_buf,
            )

"""Implicit-fast integrator matching MuJoCo's implicitfast integration scheme.

Like EulerIntegrator, but uses the qDeriv (velocity derivative) formulation
for mass matrix modification:

  M_hat = M + armature - dt * qDeriv

where qDeriv = d(passive_forces + actuator_forces) / d(qvel).

Currently (without actuators): qDeriv[i,i] = -damping[i], so:
  M_hat = M + armature + dt * damping  (same as Euler)

When actuators are added, qDeriv will also include:
  - Actuator velocity derivatives (gainprm[2], biasprm[2])
  - Tendon damping derivatives

This produces identical results to EulerIntegrator for passive systems,
but has the architecture to correctly handle actuator dynamics.

Pipeline matches EulerIntegrator exactly (see euler_integrator.mojo).
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
from ..collision.contact_detection import (
    detect_contacts,
    detect_contacts_gpu,
    normalize_qpos_quaternions,
    normalize_qpos_quaternions_gpu,
)
from ..solver.pgs_solver import PGSSolver
from ..constraints.constraint_data import ConstraintData
from ..constraints.constraint_builder import (
    build_constraints,
    writeback_impulses,
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
    ws_qvel_pred_offset,
    ws_m_inv_offset,
)


struct ImplicitFastIntegrator[SOLVER: ConstraintSolver](Integrator):
    """Implicit-fast integrator with qDeriv-based mass matrix modification.

    Uses M_hat = M + armature - dt * qDeriv where qDeriv = d(forces)/d(qvel).
    Currently qDeriv only includes passive damping; extensible for actuators.

    Parametrized by SOLVER type (PGSSolver, CGSolver, or NewtonSolver).

    Usage:
        # PGS (default):
        alias PGSImplicitFast = ImplicitFastIntegrator[PGSSolver]

        # Newton (most accurate):
        alias NewtonImplicitFast = ImplicitFastIntegrator[NewtonSolver]
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
    ](
        model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    ) where DTYPE.is_floating_point():
        """Execute one simulation step with implicit-fast integration.

        Args:
            model: Static model configuration.
            data: Mutable simulation state.
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
        detect_contacts[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM](model, data)

        # 3. Compute cdof (spatial motion axes per DOF) - needed for full M
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

        # 5b. Compute qDeriv and modify mass matrix
        # M_hat = M + armature - dt * qDeriv
        # qDeriv[i,i] = d(force_i)/d(qvel_i)
        # For passive damping: qDeriv[i,i] = -damping[i]
        # So: M_hat[i,i] += armature[i] - dt * (-damping[i])
        #                  = armature[i] + dt * damping[i]
        # Future: qDeriv will also include actuator velocity derivatives
        for j in range(model.num_joints):
            var joint = model.joints[j]
            var dof_adr = joint.dof_adr
            var arm = joint.armature
            var damp = joint.damping
            # qDeriv_diag = -damp (damping force derivative w.r.t. velocity)
            # M_hat += arm - dt * qDeriv_diag = arm + dt * damp
            var diag_add = arm + dt * damp
            if joint.jnt_type == JNT_FREE:
                for d in range(6):
                    M[(dof_adr + d) * NV + (dof_adr + d)] = (
                        M[(dof_adr + d) * NV + (dof_adr + d)] + diag_add
                    )
            elif joint.jnt_type == JNT_BALL:
                for d in range(3):
                    M[(dof_adr + d) * NV + (dof_adr + d)] = (
                        M[(dof_adr + d) * NV + (dof_adr + d)] + diag_add
                    )
            else:
                M[dof_adr * NV + dof_adr] = M[dof_adr * NV + dof_adr] + diag_add

        # 6. LDL factorize M and solve for qacc
        var L = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        var D = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        ldl_factor[DTYPE, NV, M_SIZE, V_SIZE](M, L, D)

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
        # Damping force: f -= damping * qvel (explicit part)
        # The implicit part (dt*damping added to M via qDeriv) handles the
        # NEW velocity component. Both are needed for MuJoCo-matching behavior.
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

        # qacc = M^-1 * f_net via LDL solve
        var qacc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NV):
            qacc[i] = Scalar[DTYPE](0)
        ldl_solve[DTYPE, NV, M_SIZE, V_SIZE](L, D, f_net, qacc)

        # 7. Compute full M_inv from LDL factors for constraint solver
        var M_inv = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        for i in range(M_SIZE):
            M_inv[i] = Scalar[DTYPE](0)
        compute_M_inv_from_ldl[DTYPE, NV, M_SIZE, V_SIZE](L, D, M_inv)

        # 8. Predict velocity: qvel_pred = qvel + qacc * dt
        comptime MAX_QVEL: Scalar[DTYPE] = 10.0
        var qvel_pred = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NV):
            qvel_pred[i] = data.qvel[i] + qacc[i] * dt
            # Clamp predicted velocity before solver to prevent runaway
            if qvel_pred[i] > MAX_QVEL:
                qvel_pred[i] = MAX_QVEL
            elif qvel_pred[i] < -MAX_QVEL:
                qvel_pred[i] = -MAX_QVEL

        # 9. Build constraints and solve (modifies qvel_pred in-place)
        comptime MAX_ROWS = 3 * MAX_CONTACTS + 2 * NJOINT
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
        ](model, data, cdof, M_inv, qvel_pred, dt, constraints)

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
        ](model, data, M_inv, constraints, qvel_pred, dt)

        writeback_impulses[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            MAX_ROWS,
        ](constraints, data)

        # 9. Write back constrained velocity and integrate position
        for i in range(NV):
            # qacc = (constrained_vel - old_vel) / dt
            data.qacc[i] = (qvel_pred[i] - data.qvel[i]) / dt
            data.qvel[i] = qvel_pred[i]

        # 9b. Clamp velocities to prevent divergence
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

        # 11. Joint limits now enforced as constraints inside the solver
        # (no post-step clamping needed)

    @staticmethod
    fn simulate[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        NGEOM: Int = 0,
    ](
        model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        num_steps: Int,
    ) where DTYPE.is_floating_point():
        """Run simulation for multiple steps on CPU."""
        for _ in range(num_steps):
            Self.step[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM](model, data)

    # =========================================================================
    # GPU Methods
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
        """Complete implicit-fast physics step kernel (pre-solver).

        Pipeline:
        1-6. Same as EulerIntegrator (FK, contacts, CRBA, mass matrix)
        6b. qDeriv-based mass matrix modification
        7-10. LDL factorize, bias forces, f_net with damping, predict velocity
        """

        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        comptime V_SIZE = _max_one[NV]()
        comptime M_idx = ws_M_offset[NV, NBODY]()
        comptime bias_idx = ws_bias_offset[NV, NBODY]()
        comptime fnet_idx = ws_fnet_offset[NV, NBODY]()
        comptime qacc_ws_idx = ws_qacc_ws_offset[NV, NBODY]()
        comptime qvel_pred_idx = ws_qvel_pred_offset[NV, NBODY]()
        comptime m_inv_idx = ws_m_inv_offset[NV, NBODY]()

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
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
            STATE_SIZE, MODEL_SIZE, BATCH, NGEOM,
        ](env, state, model)

        # 4. Compute cdof (writes to workspace at ws_cdof_offset)
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

        # 5. Compute composite rigid body inertia (writes to workspace at ws_crb_offset)
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

        # 6. Compute full mass matrix using CRBA (reads cdof/crb, writes M in workspace)
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

        # 6b. Compute qDeriv and modify mass matrix
        # M_hat = M + armature - dt * qDeriv
        # qDeriv[i,i] = -damping[i] (passive only; extensible for actuators)
        # Result: M_hat[i,i] += armature[i] + dt * damping[i]
        var model_meta_off_arm = model_metadata_offset[NBODY, NJOINT]()
        var dt_arm = model[0, model_meta_off_arm + MODEL_META_IDX_TIMESTEP]

        for j in range(NJOINT):
            var joint_off = model_joint_offset[NBODY](j)
            var jnt_type = Int(model[0, joint_off + JOINT_IDX_TYPE])
            var dof_adr = Int(model[0, joint_off + JOINT_IDX_DOF_ADR])
            var arm = model[0, joint_off + JOINT_IDX_ARMATURE]
            var damp = model[0, joint_off + JOINT_IDX_DAMPING]
            # qDeriv_diag = -damp => M_hat += arm - dt * (-damp) = arm + dt * damp
            var diag_add = arm + dt_arm * damp
            if jnt_type == JNT_FREE:
                for d in range(6):
                    var idx = M_idx + (dof_adr + d) * NV + (dof_adr + d)
                    workspace[env, idx] += diag_add
            elif jnt_type == JNT_BALL:
                for d in range(3):
                    var idx = M_idx + (dof_adr + d) * NV + (dof_adr + d)
                    workspace[env, idx] += diag_add
            else:
                var idx = M_idx + dof_adr * NV + dof_adr
                workspace[env, idx] += diag_add

        # 7. LDL factorize (reads M, writes L/D in workspace)
        ldl_factor_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](env, workspace)

        # Compute M_inv in workspace (reads L/D, writes M_inv in workspace)
        compute_M_inv_from_ldl_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](
            env, workspace
        )

        # 8. Compute bias forces (reads cdof from workspace, writes bias to workspace)
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

        # 9. Compute unconstrained acceleration via LDL solve
        var qvel_off = qvel_offset[NQ, NV]()
        var qacc_off = qacc_offset[NQ, NV]()
        var qfrc_off = qfrc_offset[NQ, NV]()
        var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
        var dt = rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_TIMESTEP]
        )

        # f_net = qfrc - bias (write to workspace fnet region)
        for i in range(NV):
            var qfrc = rebind[Scalar[DTYPE]](state[env, qfrc_off + i])
            var bias_val = rebind[Scalar[DTYPE]](workspace[env, bias_idx + i])
            workspace[env, fnet_idx + i] = qfrc - bias_val

        # 8b. Apply passive joint forces: damping + stiffness + frictionloss
        # Damping force: f -= damping * qvel (explicit part)
        for j in range(NJOINT):
            var joint_off_d = model_joint_offset[NBODY](j)
            var jnt_type_d = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off_d + JOINT_IDX_TYPE])
            )
            var dof_adr_d = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off_d + JOINT_IDX_DOF_ADR])
            )
            var damp_d = rebind[Scalar[DTYPE]](
                model[0, joint_off_d + JOINT_IDX_DAMPING]
            )
            if damp_d > Scalar[DTYPE](0):
                if jnt_type_d == JNT_FREE:
                    for d in range(6):
                        var v = rebind[Scalar[DTYPE]](
                            state[env, qvel_off + dof_adr_d + d]
                        )
                        var cur = rebind[Scalar[DTYPE]](
                            workspace[env, fnet_idx + dof_adr_d + d]
                        )
                        workspace[env, fnet_idx + dof_adr_d + d] = (
                            cur - damp_d * v
                        )
                elif jnt_type_d == JNT_BALL:
                    for d in range(3):
                        var v = rebind[Scalar[DTYPE]](
                            state[env, qvel_off + dof_adr_d + d]
                        )
                        var cur = rebind[Scalar[DTYPE]](
                            workspace[env, fnet_idx + dof_adr_d + d]
                        )
                        workspace[env, fnet_idx + dof_adr_d + d] = (
                            cur - damp_d * v
                        )
                else:
                    var v = rebind[Scalar[DTYPE]](
                        state[env, qvel_off + dof_adr_d]
                    )
                    var cur = rebind[Scalar[DTYPE]](
                        workspace[env, fnet_idx + dof_adr_d]
                    )
                    workspace[env, fnet_idx + dof_adr_d] = cur - damp_d * v

        # Stiffness + frictionloss
        var qpos_off_stiff = qpos_offset[NQ, NV]()
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
            # Stiffness: f -= stiffness * (qpos - springref)
            if stiff > Scalar[DTYPE](0):
                if jnt_type == JNT_FREE:
                    for d in range(6):
                        var qpos_d = rebind[Scalar[DTYPE]](
                            state[env, qpos_off_stiff + qpos_adr + d]
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
                            state[env, qpos_off_stiff + qpos_adr + d]
                        )
                        var cur = rebind[Scalar[DTYPE]](
                            workspace[env, fnet_idx + dof_adr + d]
                        )
                        workspace[env, fnet_idx + dof_adr + d] = cur - stiff * (
                            qpos_d - sref
                        )
                else:
                    var qpos_d = rebind[Scalar[DTYPE]](
                        state[env, qpos_off_stiff + qpos_adr]
                    )
                    var cur = rebind[Scalar[DTYPE]](
                        workspace[env, fnet_idx + dof_adr]
                    )
                    workspace[env, fnet_idx + dof_adr] = cur - stiff * (
                        qpos_d - sref
                    )
            # Frictionloss: f -= frictionloss * sign(qvel)
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

        # LDL solve: reads L, D, f_net from workspace, writes qacc to workspace
        ldl_solve_workspace_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](
            env, workspace
        )

        for i in range(NV):
            var qacc_val = rebind[Scalar[DTYPE]](
                workspace[env, qacc_ws_idx + i]
            )
            state[env, qacc_off + i] = qacc_val

        # 10. Predict velocity (write to workspace qvel_pred region)
        comptime MAX_QVEL: Scalar[DTYPE] = 10.0
        for i in range(NV):
            var qvel = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
            var qacc_val = rebind[Scalar[DTYPE]](
                workspace[env, qacc_ws_idx + i]
            )
            var vpred = qvel + qacc_val * dt
            # Clamp predicted velocity before solver to prevent runaway
            if vpred > MAX_QVEL:
                vpred = MAX_QVEL
            elif vpred < -MAX_QVEL:
                vpred = -MAX_QVEL
            workspace[env, qvel_pred_idx + i] = vpred

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
        """Finalize physics step: write back velocity, integrate position.

        Pipeline:
        9. Write back constrained velocity, integrate position
        10. Normalize quaternions
        11. Enforce joint limits
        """

        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return

        var qvel_off = qvel_offset[NQ, NV]()
        var qacc_off = qacc_offset[NQ, NV]()
        var qvel_pred_idx = ws_qvel_pred_offset[NV, NBODY]()
        var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
        var dt = rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_TIMESTEP]
        )
        # 9. Write back constrained velocity and update qacc
        var qpos_off = qpos_offset[NQ, NV]()
        for i in range(NV):
            # qacc = (constrained_vel - old_vel) / dt
            var old_qvel = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
            var constrained_vel = rebind[Scalar[DTYPE]](
                workspace[env, qvel_pred_idx + i]
            )
            state[env, qacc_off + i] = (constrained_vel - old_qvel) / dt
            state[env, qvel_off + i] = constrained_vel

        # 9b. Clamp velocities to prevent divergence
        comptime MAX_QVEL: Scalar[DTYPE] = 10.0
        for i in range(NV):
            var v = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
            if v > MAX_QVEL:
                state[env, qvel_off + i] = MAX_QVEL
            elif v < -MAX_QVEL:
                state[env, qvel_off + i] = -MAX_QVEL

        for i in range(NQ):
            if i < NV:
                var qpos = rebind[Scalar[DTYPE]](state[env, qpos_off + i])
                var qvel = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
                state[env, qpos_off + i] = qpos + qvel * dt

        # 10. Normalize quaternions
        normalize_qpos_quaternions_gpu[
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

        # 11. Joint limits now enforced as constraints inside the solver
        # (no post-step clamping needed)

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
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[DTYPE],
        mut model_buf: DeviceBuffer[DTYPE],
        mut workspace_buf: DeviceBuffer[DTYPE],
        dt: Scalar[DTYPE],
        gravity_z: Scalar[DTYPE],
        ground_z: Scalar[DTYPE],
    ) raises:
        """Perform one physics simulation step on GPU with implicit-fast integration.

        Uses the parametrized SOLVER for contact constraint resolution.
        """
        comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS]()
        comptime MODEL_SIZE = model_size[NBODY, NJOINT, NGEOM]()
        comptime WS_SIZE = integrator_workspace_size[
            NV, NBODY
        ]() + NV * NV + Self.SOLVER.solver_workspace_size[NV, MAX_CONTACTS]()

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

        var model = LayoutTensor[
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
            model,
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
        ]

        ctx.enqueue_function[solver_wrapper, solver_wrapper](
            state,
            model,
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
            model,
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
            Self.step_gpu[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, BATCH, NGEOM](
                ctx,
                state_buf,
                model_buf,
                workspace_buf,
                dt,
                gravity_z,
                ground_z,
            )

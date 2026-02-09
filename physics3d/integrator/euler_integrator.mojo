"""Constraint-based GC integrator with configurable contact solver.

Supports three solver types (mirroring MuJoCo's solver options):
- PGS (Projected Gauss-Seidel): Fast, reliable, default choice
- CG (Conjugate Gradient): Faster convergence for well-conditioned problems
- Newton: Quadratic convergence, most accurate for stiff contacts

Pipeline:
1. Forward kinematics (qpos -> xpos, xquat)
2. Compute body velocities (qvel -> xvel, xangvel)
3. Detect ground contacts
4. Compute cdof (spatial motion axes per DOF)
5. Compute composite rigid body inertia (CRBA)
6. Compute full mass matrix M(q) using CRBA
7. LDL factorize M, compute M_inv
8. Compute bias forces
9. Compute unconstrained acceleration: qacc = M^-1 * (qfrc - bias) via LDL solve
10. Predict velocity: qvel_pred = qvel + qacc * dt
11. Constraint solve (PGS/CG/Newton): modify qvel_pred using full M_inv
12. qpos += qvel_pred * dt
13. Normalize quaternions, enforce joint limits

This produces bounded, physically correct contact forces instead of
unbounded spring forces that can launch bodies into the sky.
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
    detect_ground_contacts,
    detect_ground_contacts_gpu,
    detect_body_body_contacts,
    detect_body_body_contacts_gpu,
    normalize_qpos_quaternions,
    normalize_qpos_quaternions_gpu,
)
from ..solver.pgs_solver import PGSSolver
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
    MODEL_META_IDX_TIMESTEP,
    integrator_workspace_size,
    ws_M_offset,
    ws_bias_offset,
    ws_fnet_offset,
    ws_qacc_ws_offset,
    ws_qvel_pred_offset,
    ws_m_inv_offset,
)


struct EulerIntegrator[SOLVER: ConstraintSolver](Integrator):
    """GC integrator with configurable constraint-based contact solving.

    Parametrized by SOLVER type (PGSSolver, CGSolver, or NewtonSolver).
    Uses the specified solver for contact constraints instead of penalty springs.

    Usage:
        # PGS (default, backward-compatible):
        alias PGSIntegrator = EulerIntegrator[PGSSolver]

        # Conjugate Gradient:
        alias CGIntegrator = EulerIntegrator[CGSolver]

        # Newton:
        alias NewtonIntegrator = EulerIntegrator[NewtonSolver]
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
    ](
        model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    ):
        """Execute one simulation step with constraint-based contacts.

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
        detect_ground_contacts(model, data)
        detect_body_body_contacts(model, data)

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

        # 5b. Add armature + implicit damping to mass matrix diagonal
        # MuJoCo implicitfast: M_eff[i,i] += armature[i] + dt * damping[i]
        # Implicit damping: instead of explicit f -= D*qvel, we add dt*D
        # to the mass matrix. This provides unconditional stability for damping
        # and correctly damps the NEW velocity (semi-implicit treatment).
        for j in range(model.num_joints):
            var joint = model.joints[j]
            var dof_adr = joint.dof_adr
            var arm = joint.armature
            var damp = joint.damping
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

        # 6b. Apply passive joint forces: stiffness only
        # Damping is handled implicitly via M_eff (step 5b).
        # Stiffness: f -= stiffness * (qpos - springref), springref=0
        for j in range(model.num_joints):
            var joint = model.joints[j]
            var dof_adr = joint.dof_adr
            var qpos_adr = joint.qpos_adr
            var stiff = joint.stiffness
            if stiff > Scalar[DTYPE](0):
                if joint.jnt_type == JNT_FREE:
                    for d in range(6):
                        # For free joints: stiffness on position DOFs
                        f_net[dof_adr + d] = (
                            f_net[dof_adr + d] - stiff * data.qpos[qpos_adr + d]
                        )
                elif joint.jnt_type == JNT_BALL:
                    for d in range(3):
                        f_net[dof_adr + d] = (
                            f_net[dof_adr + d] - stiff * data.qpos[qpos_adr + d]
                        )
                else:
                    # Hinge/slide: f = -stiffness * qpos
                    f_net[dof_adr] = (
                        f_net[dof_adr] - stiff * data.qpos[qpos_adr]
                    )

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
        var qvel_pred = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NV):
            qvel_pred[i] = data.qvel[i] + qacc[i] * dt

        # 9. Constraint solve (modifies qvel_pred in-place)
        Self.SOLVER.solve[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            V_SIZE,
            M_SIZE,
            CDOF_SIZE,
        ](model, data, M_inv, cdof, qvel_pred, dt)

        # 9. Write back constrained velocity and integrate position
        for i in range(NV):
            # qacc = (constrained_vel - old_vel) / dt
            data.qacc[i] = (qvel_pred[i] - data.qvel[i]) / dt
            data.qvel[i] = qvel_pred[i]

        # 9b. Clamp velocities to prevent divergence
        # MuJoCo uses ~10-50 depending on model; 20 is reasonable for walking robots
        comptime MAX_QVEL: Scalar[DTYPE] = 20.0
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
    ](
        model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        num_steps: Int,
    ):
        """Run simulation for multiple steps on CPU."""
        for _ in range(num_steps):
            Self.step(model, data)

    # =========================================================================
    # GPU Methods
    # =========================================================================

    # @always_inline
    # @staticmethod
    # fn step_constraint_kernel[
    #     DTYPE: DType,
    #     NQ: Int,
    #     NV: Int,
    #     NBODY: Int,
    #     NJOINT: Int,
    #     MAX_CONTACTS: Int,
    #     STATE_SIZE: Int,
    #     MODEL_SIZE: Int,
    #     BATCH: Int,
    #     WS_SIZE: Int,
    # ](
    #     env: Int,
    #     state: LayoutTensor[
    #         DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    #     ],
    #     model: LayoutTensor[
    #         DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
    #     ],
    #     workspace: LayoutTensor[
    #         DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    #     ],
    # ):
    #     """Complete GC physics step with configurable constraint solver.

    #     Pipeline:
    #     1. Forward kinematics (qpos -> xpos, xquat)
    #     2. Compute body velocities (qvel -> xvel, xangvel)
    #     3. Detect ground contacts
    #     4. Compute cdof (spatial motion axes per DOF)
    #     5. Compute composite rigid body inertia (CRBA)
    #     6. Compute full mass matrix M(q)
    #     7. LDL factorize M, compute M_inv
    #     8. Compute bias forces
    #     9. Compute unconstrained acceleration via LDL solve
    #     10. Predict velocity
    #     11. Constraint solve using SOLVER with full M_inv
    #     12. Write back constrained velocity, integrate position
    #     13. Normalize quaternions
    #     14. Enforce joint limits
    #     """
    #     comptime V_SIZE = _max_one[NV]()
    #     comptime M_idx = ws_M_offset[NV, NBODY]()
    #     comptime bias_idx = ws_bias_offset[NV, NBODY]()
    #     comptime fnet_idx = ws_fnet_offset[NV, NBODY]()
    #     comptime qacc_ws_idx = ws_qacc_ws_offset[NV, NBODY]()
    #     comptime qvel_pred_idx = ws_qvel_pred_offset[NV, NBODY]()
    #     comptime m_inv_idx = ws_m_inv_offset[NV, NBODY]()

    #     # 1. Forward kinematics
    #     forward_kinematics_gpu[
    #         DTYPE,
    #         NQ,
    #         NV,
    #         NBODY,
    #         NJOINT,
    #         MAX_CONTACTS,
    #         STATE_SIZE,
    #         MODEL_SIZE,
    #         BATCH,
    #     ](env, state, model)

    #     # 2. Compute body velocities
    #     compute_body_velocities_gpu[
    #         DTYPE,
    #         NQ,
    #         NV,
    #         NBODY,
    #         NJOINT,
    #         MAX_CONTACTS,
    #         STATE_SIZE,
    #         MODEL_SIZE,
    #         BATCH,
    #     ](env, state, model)

    #     # 3. Detect ground contacts + body-body contacts
    #     detect_ground_contacts_gpu[
    #         DTYPE,
    #         NQ,
    #         NV,
    #         NBODY,
    #         NJOINT,
    #         MAX_CONTACTS,
    #         STATE_SIZE,
    #         MODEL_SIZE,
    #         BATCH,
    #     ](env, state, model)
    #     detect_body_body_contacts_gpu[
    #         DTYPE,
    #         NQ,
    #         NV,
    #         NBODY,
    #         NJOINT,
    #         MAX_CONTACTS,
    #         STATE_SIZE,
    #         MODEL_SIZE,
    #         BATCH,
    #     ](env, state, model)

    #     # 4. Compute cdof (writes to workspace at ws_cdof_offset)
    #     compute_cdof_gpu[
    #         DTYPE,
    #         NQ,
    #         NV,
    #         NBODY,
    #         NJOINT,
    #         MAX_CONTACTS,
    #         STATE_SIZE,
    #         MODEL_SIZE,
    #         BATCH,
    #         WS_SIZE,
    #     ](env, state, model, workspace)

    #     # 5. Compute composite rigid body inertia (writes to workspace at ws_crb_offset)
    #     compute_composite_inertia_gpu[
    #         DTYPE,
    #         NQ,
    #         NV,
    #         NBODY,
    #         NJOINT,
    #         MAX_CONTACTS,
    #         STATE_SIZE,
    #         MODEL_SIZE,
    #         BATCH,
    #         WS_SIZE,
    #     ](env, state, model, workspace)

    #     # 6. Compute full mass matrix using CRBA (reads cdof/crb, writes M in workspace)
    #     compute_mass_matrix_full_gpu[
    #         DTYPE,
    #         NQ,
    #         NV,
    #         NBODY,
    #         NJOINT,
    #         MAX_CONTACTS,
    #         STATE_SIZE,
    #         MODEL_SIZE,
    #         BATCH,
    #         WS_SIZE,
    #     ](env, state, model, workspace)

    #     # 6b. Add armature + implicit damping to mass matrix diagonal
    #     # MuJoCo implicitfast: M_eff[i,i] += armature[i] + dt * damping[i]
    #     var model_meta_off_arm = model_metadata_offset[NBODY, NJOINT]()
    #     var dt_arm = rebind[Scalar[DTYPE]](
    #         model[0, model_meta_off_arm + MODEL_META_IDX_TIMESTEP]
    #     )
    #     for j in range(NJOINT):
    #         var joint_off = model_joint_offset[NBODY](j)
    #         var jnt_type = Int(
    #             rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
    #         )
    #         var dof_adr = Int(
    #             rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR])
    #         )
    #         var arm = rebind[Scalar[DTYPE]](
    #             model[0, joint_off + JOINT_IDX_ARMATURE]
    #         )
    #         var damp = rebind[Scalar[DTYPE]](
    #             model[0, joint_off + JOINT_IDX_DAMPING]
    #         )
    #         var diag_add = arm + dt_arm * damp
    #         if jnt_type == JNT_FREE:
    #             for d in range(6):
    #                 var idx = M_idx + (dof_adr + d) * NV + (dof_adr + d)
    #                 workspace[env, idx] = workspace[env, idx] + diag_add
    #         elif jnt_type == JNT_BALL:
    #             for d in range(3):
    #                 var idx = M_idx + (dof_adr + d) * NV + (dof_adr + d)
    #                 workspace[env, idx] = workspace[env, idx] + diag_add
    #         else:
    #             var idx = M_idx + dof_adr * NV + dof_adr
    #             workspace[env, idx] = workspace[env, idx] + diag_add

    #     # 7. LDL factorize (reads M, writes L/D in workspace)
    #     ldl_factor_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](env, workspace)

    #     # Compute M_inv in workspace (reads L/D, writes M_inv in workspace)
    #     compute_M_inv_from_ldl_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](
    #         env, workspace
    #     )

    #     # 8. Compute bias forces (reads cdof from workspace, writes bias to workspace)
    #     compute_bias_forces_rne_gpu[
    #         DTYPE,
    #         NQ,
    #         NV,
    #         NBODY,
    #         NJOINT,
    #         MAX_CONTACTS,
    #         STATE_SIZE,
    #         MODEL_SIZE,
    #         BATCH,
    #         WS_SIZE,
    #     ](env, state, model, workspace)

    #     # 9. Compute unconstrained acceleration via LDL solve
    #     var qvel_off = qvel_offset[NQ, NV]()
    #     var qacc_off = qacc_offset[NQ, NV]()
    #     var qfrc_off = qfrc_offset[NQ, NV]()
    #     var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
    #     var dt = rebind[Scalar[DTYPE]](
    #         model[0, model_meta_off + MODEL_META_IDX_TIMESTEP]
    #     )

    #     # f_net = qfrc - bias (write to workspace fnet region)
    #     for i in range(NV):
    #         var qfrc = rebind[Scalar[DTYPE]](state[env, qfrc_off + i])
    #         var bias_val = rebind[Scalar[DTYPE]](workspace[env, bias_idx + i])
    #         workspace[env, fnet_idx + i] = qfrc - bias_val

    #     # 8b. Apply passive joint forces: stiffness only
    #     # Damping is handled implicitly via M_eff (step 6b).
    #     var qpos_off_stiff = qpos_offset[NQ, NV]()
    #     for j in range(NJOINT):
    #         var joint_off = model_joint_offset[NBODY](j)
    #         var jnt_type = Int(
    #             rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
    #         )
    #         var dof_adr = Int(
    #             rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR])
    #         )
    #         var qpos_adr = Int(
    #             rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_QPOS_ADR])
    #         )
    #         var stiff = rebind[Scalar[DTYPE]](
    #             model[0, joint_off + JOINT_IDX_STIFFNESS]
    #         )
    #         if stiff > Scalar[DTYPE](0):
    #             if jnt_type == JNT_FREE:
    #                 for d in range(6):
    #                     var qpos_d = rebind[Scalar[DTYPE]](
    #                         state[env, qpos_off_stiff + qpos_adr + d]
    #                     )
    #                     var cur = rebind[Scalar[DTYPE]](
    #                         workspace[env, fnet_idx + dof_adr + d]
    #                     )
    #                     workspace[env, fnet_idx + dof_adr + d] = (
    #                         cur - stiff * qpos_d
    #                     )
    #             elif jnt_type == JNT_BALL:
    #                 for d in range(3):
    #                     var qpos_d = rebind[Scalar[DTYPE]](
    #                         state[env, qpos_off_stiff + qpos_adr + d]
    #                     )
    #                     var cur = rebind[Scalar[DTYPE]](
    #                         workspace[env, fnet_idx + dof_adr + d]
    #                     )
    #                     workspace[env, fnet_idx + dof_adr + d] = (
    #                         cur - stiff * qpos_d
    #                     )
    #             else:
    #                 # Hinge/slide: f = -stiffness * qpos
    #                 var qpos_d = rebind[Scalar[DTYPE]](
    #                     state[env, qpos_off_stiff + qpos_adr]
    #                 )
    #                 var cur = rebind[Scalar[DTYPE]](
    #                     workspace[env, fnet_idx + dof_adr]
    #                 )
    #                 workspace[env, fnet_idx + dof_adr] = cur - stiff * qpos_d

    #     # LDL solve: reads L, D, f_net from workspace, writes qacc to workspace
    #     ldl_solve_workspace_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](
    #         env, workspace
    #     )

    #     for i in range(NV):
    #         var qacc_val = rebind[Scalar[DTYPE]](
    #             workspace[env, qacc_ws_idx + i]
    #         )
    #         state[env, qacc_off + i] = qacc_val

    #     # 10. Predict velocity (write to workspace qvel_pred region)
    #     for i in range(NV):
    #         var qvel = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
    #         var qacc_val = rebind[Scalar[DTYPE]](
    #             workspace[env, qacc_ws_idx + i]
    #         )
    #         workspace[env, qvel_pred_idx + i] = qvel + qacc_val * dt

    #     # 11. Constraint solve using parametrized solver with full M_inv (in workspace)
    #     Self.SOLVER.solve_gpu[
    #         DTYPE,
    #         NQ,
    #         NV,
    #         NBODY,
    #         NJOINT,
    #         MAX_CONTACTS,
    #         STATE_SIZE,
    #         MODEL_SIZE,
    #         V_SIZE,
    #         BATCH,
    #         WS_SIZE,
    #     ](state, model, workspace)

    #     # 9. Write back constrained velocity and update qacc
    #     var qpos_off = qpos_offset[NQ, NV]()
    #     for i in range(NV):
    #         # qacc = (constrained_vel - old_vel) / dt
    #         var old_qvel = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
    #         var constrained_vel = rebind[Scalar[DTYPE]](
    #             workspace[env, qvel_pred_idx + i]
    #         )
    #         state[env, qacc_off + i] = (constrained_vel - old_qvel) / dt
    #         state[env, qvel_off + i] = constrained_vel

    #     # 9b. Clamp velocities to prevent divergence
    #     # MuJoCo uses ~10-50 depending on model; 20 is reasonable for walking robots
    #     comptime MAX_QVEL: Scalar[DTYPE] = 20.0
    #     for i in range(NV):
    #         var v = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
    #         if v > MAX_QVEL:
    #             state[env, qvel_off + i] = MAX_QVEL
    #         elif v < -MAX_QVEL:
    #             state[env, qvel_off + i] = -MAX_QVEL

    #     for i in range(NQ):
    #         if i < NV:
    #             var qpos = rebind[Scalar[DTYPE]](state[env, qpos_off + i])
    #             var qvel = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
    #             state[env, qpos_off + i] = qpos + qvel * dt

    #     # 10. Normalize quaternions
    #     normalize_qpos_quaternions_gpu[
    #         DTYPE,
    #         NQ,
    #         NV,
    #         NBODY,
    #         NJOINT,
    #         MAX_CONTACTS,
    #         STATE_SIZE,
    #         MODEL_SIZE,
    #         BATCH,
    #     ](env, state, model)

    #     # 11. Joint limits now enforced as constraints inside the solver
    #     # (no post-step clamping needed)

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
        """Complete GC physics step with configurable constraint solver.

        Pipeline:
        1. Forward kinematics (qpos -> xpos, xquat)
        2. Compute body velocities (qvel -> xvel, xangvel)
        3. Detect ground contacts
        4. Compute cdof (spatial motion axes per DOF)
        5. Compute composite rigid body inertia (CRBA)
        6. Compute full mass matrix M(q)
        7. LDL factorize M, compute M_inv
        8. Compute bias forces
        9. Compute unconstrained acceleration via LDL solve
        10. Predict velocity
        11. Constraint solve using SOLVER with full M_inv
        12. Write back constrained velocity, integrate position
        13. Normalize quaternions
        14. Enforce joint limits
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

        # 3. Detect ground contacts + body-body contacts
        detect_ground_contacts_gpu[
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
        detect_body_body_contacts_gpu[
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

        # 6b. Add armature + implicit damping to mass matrix diagonal
        # MuJoCo implicitfast: M_eff[i,i] += armature[i] + dt * damping[i]
        var model_meta_off_arm = model_metadata_offset[NBODY, NJOINT]()
        var dt_arm = rebind[Scalar[DTYPE]](
            model[0, model_meta_off_arm + MODEL_META_IDX_TIMESTEP]
        )
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
            var damp = rebind[Scalar[DTYPE]](
                model[0, joint_off + JOINT_IDX_DAMPING]
            )
            var diag_add = arm + dt_arm * damp
            if jnt_type == JNT_FREE:
                for d in range(6):
                    var idx = M_idx + (dof_adr + d) * NV + (dof_adr + d)
                    workspace[env, idx] = workspace[env, idx] + diag_add
            elif jnt_type == JNT_BALL:
                for d in range(3):
                    var idx = M_idx + (dof_adr + d) * NV + (dof_adr + d)
                    workspace[env, idx] = workspace[env, idx] + diag_add
            else:
                var idx = M_idx + dof_adr * NV + dof_adr
                workspace[env, idx] = workspace[env, idx] + diag_add

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

        # 8b. Apply passive joint forces: stiffness only
        # Damping is handled implicitly via M_eff (step 6b).
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
            if stiff > Scalar[DTYPE](0):
                if jnt_type == JNT_FREE:
                    for d in range(6):
                        var qpos_d = rebind[Scalar[DTYPE]](
                            state[env, qpos_off_stiff + qpos_adr + d]
                        )
                        var cur = rebind[Scalar[DTYPE]](
                            workspace[env, fnet_idx + dof_adr + d]
                        )
                        workspace[env, fnet_idx + dof_adr + d] = (
                            cur - stiff * qpos_d
                        )
                elif jnt_type == JNT_BALL:
                    for d in range(3):
                        var qpos_d = rebind[Scalar[DTYPE]](
                            state[env, qpos_off_stiff + qpos_adr + d]
                        )
                        var cur = rebind[Scalar[DTYPE]](
                            workspace[env, fnet_idx + dof_adr + d]
                        )
                        workspace[env, fnet_idx + dof_adr + d] = (
                            cur - stiff * qpos_d
                        )
                else:
                    # Hinge/slide: f = -stiffness * qpos
                    var qpos_d = rebind[Scalar[DTYPE]](
                        state[env, qpos_off_stiff + qpos_adr]
                    )
                    var cur = rebind[Scalar[DTYPE]](
                        workspace[env, fnet_idx + dof_adr]
                    )
                    workspace[env, fnet_idx + dof_adr] = cur - stiff * qpos_d

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
        for i in range(NV):
            var qvel = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
            var qacc_val = rebind[Scalar[DTYPE]](
                workspace[env, qacc_ws_idx + i]
            )
            workspace[env, qvel_pred_idx + i] = qvel + qacc_val * dt

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
        """Complete GC physics step with configurable constraint solver.

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
        # MuJoCo uses ~10-50 depending on model; 20 is reasonable for walking robots
        comptime MAX_QVEL: Scalar[DTYPE] = 20.0
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
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[DTYPE],
        mut model_buf: DeviceBuffer[DTYPE],
        mut workspace_buf: DeviceBuffer[DTYPE],
        dt: Scalar[DTYPE],
        gravity_z: Scalar[DTYPE],
        ground_z: Scalar[DTYPE],
    ) raises:
        """Perform one physics simulation step on GPU with constraint solving.

        Uses the parametrized SOLVER for contact constraint resolution.
        """
        comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS]()
        comptime MODEL_SIZE = model_size[NBODY, NJOINT]()
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
            Self.step_gpu[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, BATCH](
                ctx,
                state_buf,
                model_buf,
                workspace_buf,
                dt,
                gravity_z,
                ground_z,
            )


# Backward-compatible alias: uses PGS solver by default
comptime DefaultIntegrator = EulerIntegrator[PGSSolver]

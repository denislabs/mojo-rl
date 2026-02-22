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

from math import sqrt, abs
from gpu.host import DeviceContext, DeviceBuffer
from gpu import thread_idx, block_idx, block_dim
from layout import LayoutTensor, Layout

from ..types import Model, Data, _max_one, ConeType
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
    forward_kinematics_gpu,
    compute_body_velocities_gpu,
)
from ..kinematics.quat_math import quat_normalize, quat_integrate, quat_rotate
from ..kinematics.quat_math import gpu_quat_rotate
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
    model_body_offset,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    xquat_offset,
    xvel_offset,
    xangvel_offset,
    xipos_offset,
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
    JOINT_IDX_BODY_ID,
    BODY_IDX_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    BODY_IDX_PARENT,
    MODEL_BODY_SIZE,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_NJOINT,
    MODEL_META_IDX_DENSITY,
    MODEL_META_IDX_VISCOSITY,
    integrator_workspace_size,
    ws_cdof_offset,
    ws_M_offset,
    ws_bias_offset,
    ws_fnet_offset,
    ws_qacc_ws_offset,
    ws_qacc_constrained_offset,
    ws_m_inv_offset,
)


struct EulerIntegrator[SOLVER: ConstraintSolver](Integrator):
    """GC integrator with configurable constraint-based contact solving.

    Parametrized by SOLVER type (PGSSolver, NewtonSolver, or CGSolver).
    Uses the specified solver for contact constraints instead of penalty springs.

    Usage:
        # PGS (default):
        alias PGSIntegrator = EulerIntegrator[PGSSolver]

        # Newton (most accurate, matches MuJoCo):
        alias NewtonIntegrator = EulerIntegrator[NewtonSolver]

        # Conjugate Gradient:
        alias CGIntegrator = EulerIntegrator[CGSolver]
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
        MAX_TENDON: Int = 0,
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
        ],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        verbose: Bool = False,
    ) where DTYPE.is_floating_point():
        """Execute one simulation step with constraint-based contacts.

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

        # 3. Compute cdof (spatial motion axes per DOF) - needed for full M
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

        # 5b. Add armature to mass matrix diagonal
        # MuJoCo Euler: M_solver = M + armature (damping is purely explicit via f -= D*v)
        # This differs from ImplicitFast which uses M_hat = M + arm + dt*D
        for j in range(model.num_joints):
            var joint = model.joints[j]
            var dof_adr = joint.dof_adr
            var arm = joint.armature
            var diag_add = arm
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
        compute_bias_forces_rne(model, data, cdof, bias)

        var f_net = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NV):
            f_net[i] = data.qfrc[i] - bias[i]

        # 6b. Apply passive joint forces: damping + stiffness + frictionloss
        # Damping force: f -= damping * qvel (explicit part)
        # The implicit part (dt*damping added to M) handles the NEW velocity component.
        # Both are needed: MuJoCo Euler uses M_hat = M + arm + dt*diag(damping)
        # AND f_net -= damping * qvel.
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
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, MAX_EQUALITY, CONE_TYPE, MAX_TENDON
        ](model, data, cdof, f_net)

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

        # 8. Build constraints and solve (modifies qacc in-place)
        comptime MAX_ROWS = 11 * MAX_CONTACTS + 2 * NJOINT + 6 * MAX_EQUALITY + MAX_TENDON
        var constraints = ConstraintData[DTYPE, MAX_ROWS, NV]()
        build_constraints[CONE_TYPE=CONE_TYPE, MAX_TENDON=MAX_TENDON](
            model, data, cdof, M_inv, qacc, dt, constraints
        )

        # Fill M_hat and qfrc_smooth for primal solvers
        for i in range(NV * NV):
            constraints.M_hat[i] = M[i]
        for i in range(NV):
            constraints.qfrc_smooth[i] = f_net[i]

        Self.SOLVER.solve[CONE_TYPE=CONE_TYPE](
            model, data, M_inv, constraints, qacc, dt
        )

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
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, MAX_EQUALITY, CONE_TYPE, MAX_TENDON
        ](model, data)

        # 9. Integrate with implicit velocity damping (MuJoCo 3.x Euler)
        # Formula: (M + dt*D) * v_new = M * v_euler + dt * D * v_old
        # where v_euler = v_old + dt * qacc (qacc includes explicit damping).
        # The dt*D*v_old term cancels the explicit damping in qacc, making
        # all damping purely implicit. Without it, damping is double-counted.

        # Step 1: v_euler = v_old + dt * qacc
        var v_euler = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NV):
            data.qacc[i] = qacc[i]
            v_euler[i] = data.qvel[i] + qacc[i] * dt

        # Step 2: rhs = M * v_euler (M still has armature only)
        var rhs = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NV):
            var sum = Scalar[DTYPE](0)
            for j in range(NV):
                sum += M[i * NV + j] * v_euler[j]
            rhs[i] = sum

        # Step 2b: Add dt * D * v_old to rhs (cancels explicit damping in qacc)
        for j in range(model.num_joints):
            var joint = model.joints[j]
            var dof_adr = joint.dof_adr
            var damp = joint.damping
            if damp > Scalar[DTYPE](0):
                if joint.jnt_type == JNT_FREE:
                    for d in range(6):
                        rhs[dof_adr + d] += dt * damp * data.qvel[dof_adr + d]
                elif joint.jnt_type == JNT_BALL:
                    for d in range(3):
                        rhs[dof_adr + d] += dt * damp * data.qvel[dof_adr + d]
                else:
                    rhs[dof_adr] += dt * damp * data.qvel[dof_adr]

        # Step 3: Add dt*damping to M diagonal → M_hat = M + arm + dt*D
        for j in range(model.num_joints):
            var joint = model.joints[j]
            var dof_adr = joint.dof_adr
            var damp = joint.damping
            if damp > Scalar[DTYPE](0):
                if joint.jnt_type == JNT_FREE:
                    for d in range(6):
                        M[(dof_adr + d) * NV + (dof_adr + d)] += dt * damp
                elif joint.jnt_type == JNT_BALL:
                    for d in range(3):
                        M[(dof_adr + d) * NV + (dof_adr + d)] += dt * damp
                else:
                    M[dof_adr * NV + dof_adr] += dt * damp

        # Step 4: Re-factor M_hat via LDL
        ldl_factor[DTYPE, NV, M_SIZE, V_SIZE](M, L, D)

        # Step 5: Solve M_hat * v_new = rhs
        var v_new = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NV):
            v_new[i] = Scalar[DTYPE](0)
        ldl_solve[DTYPE, NV, M_SIZE, V_SIZE](L, D, rhs, v_new)

        for i in range(NV):
            data.qvel[i] = v_new[i]

        # Integrate position: qpos += qvel * dt (quaternion-aware)
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
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        MAX_TENDON: Int = 0,
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
                MAX_TENDON,
            ](model, data)

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
        comptime qacc_constrained_idx = ws_qacc_constrained_offset[NV, NBODY]()
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

        # 6b. Add armature to mass matrix diagonal
        # MuJoCo Euler: M_solver = M + armature (damping is purely explicit via f -= D*v)
        for j in range(NJOINT):
            var joint_off = model_joint_offset[NBODY](j)
            var jnt_type = Int(model[0, joint_off + JOINT_IDX_TYPE])
            var dof_adr = Int(model[0, joint_off + JOINT_IDX_DOF_ADR])
            var arm = model[0, joint_off + JOINT_IDX_ARMATURE]
            var diag_add = arm
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

        # 8c. Fluid forces: inertia-box viscous + pressure drag (GPU)
        # Enabled when density > 0 or viscosity > 0 (stored in model metadata).
        var model_meta_off_fl = model_metadata_offset[NBODY, NJOINT]()
        var rho_fl = rebind[Scalar[DTYPE]](
            model[0, model_meta_off_fl + MODEL_META_IDX_DENSITY]
        )
        var mu_fl = rebind[Scalar[DTYPE]](
            model[0, model_meta_off_fl + MODEL_META_IDX_VISCOSITY]
        )
        if rho_fl > Scalar[DTYPE](0) or mu_fl > Scalar[DTYPE](0):
            comptime PI_FL: Scalar[DTYPE] = 3.14159265358979323846
            comptime xquat_off = xquat_offset[NQ, NV, NBODY]()
            comptime xvel_off = xvel_offset[NQ, NV, NBODY]()
            comptime xangvel_off = xangvel_offset[NQ, NV, NBODY]()
            comptime xipos_off = xipos_offset[NQ, NV, NBODY]()
            comptime cdof_off = ws_cdof_offset()

            for b in range(1, NBODY):
                var body_off_b = model_body_offset(b)
                var mass_b = rebind[Scalar[DTYPE]](
                    model[0, body_off_b + BODY_IDX_MASS]
                )
                if mass_b <= Scalar[DTYPE](1e-10):
                    continue

                # Box from diagonal inertia
                var Ixx = rebind[Scalar[DTYPE]](
                    model[0, body_off_b + BODY_IDX_IXX]
                )
                var Iyy = rebind[Scalar[DTYPE]](
                    model[0, body_off_b + BODY_IDX_IYY]
                )
                var Izz = rebind[Scalar[DTYPE]](
                    model[0, body_off_b + BODY_IDX_IZZ]
                )
                var bx2 = Scalar[DTYPE](6) * (Iyy + Izz - Ixx) / mass_b
                var by2 = Scalar[DTYPE](6) * (Ixx + Izz - Iyy) / mass_b
                var bz2 = Scalar[DTYPE](6) * (Ixx + Iyy - Izz) / mass_b
                var bx = sqrt(max(bx2, Scalar[DTYPE](0)))
                var by = sqrt(max(by2, Scalar[DTYPE](0)))
                var bz = sqrt(max(bz2, Scalar[DTYPE](0)))

                # World-frame body velocity (at body origin)
                var vx_w = rebind[Scalar[DTYPE]](state[env, xvel_off + b * 3 + 0])
                var vy_w = rebind[Scalar[DTYPE]](state[env, xvel_off + b * 3 + 1])
                var vz_w = rebind[Scalar[DTYPE]](state[env, xvel_off + b * 3 + 2])
                var wx_w = rebind[Scalar[DTYPE]](state[env, xangvel_off + b * 3 + 0])
                var wy_w = rebind[Scalar[DTYPE]](state[env, xangvel_off + b * 3 + 1])
                var wz_w = rebind[Scalar[DTYPE]](state[env, xangvel_off + b * 3 + 2])

                # Rotate to body local frame (conjugate of xquat)
                var qx_b = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 0])
                var qy_b = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 1])
                var qz_b = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 2])
                var qw_b = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 3])

                var vloc_b = gpu_quat_rotate[DTYPE](-qx_b, -qy_b, -qz_b, qw_b, vx_w, vy_w, vz_w)
                var wloc_b = gpu_quat_rotate[DTYPE](-qx_b, -qy_b, -qz_b, qw_b, wx_w, wy_w, wz_w)
                var vx = vloc_b[0]; var vy = vloc_b[1]; var vz = vloc_b[2]
                var wx = wloc_b[0]; var wy = wloc_b[1]; var wz = wloc_b[2]

                var diam = (bx + by + bz) / Scalar[DTYPE](3)

                var lfx = Scalar[DTYPE](0)
                var lfy = Scalar[DTYPE](0)
                var lfz = Scalar[DTYPE](0)
                var ltx = Scalar[DTYPE](0)
                var lty = Scalar[DTYPE](0)
                var ltz = Scalar[DTYPE](0)

                if mu_fl > Scalar[DTYPE](0):
                    var visc_lin = Scalar[DTYPE](3) * PI_FL * diam * mu_fl
                    lfx = lfx - visc_lin * vx
                    lfy = lfy - visc_lin * vy
                    lfz = lfz - visc_lin * vz
                    var d3 = diam * diam * diam
                    var visc_ang = PI_FL * d3 * mu_fl
                    ltx = ltx - visc_ang * wx
                    lty = lty - visc_ang * wy
                    ltz = ltz - visc_ang * wz

                if rho_fl > Scalar[DTYPE](0):
                    var half_rho = Scalar[DTYPE](0.5) * rho_fl
                    lfx = lfx - half_rho * by * bz * abs(vx) * vx
                    lfy = lfy - half_rho * bx * bz * abs(vy) * vy
                    lfz = lfz - half_rho * bx * by * abs(vz) * vz
                    var bx4 = bx * bx * bx * bx
                    var by4 = by * by * by * by
                    var bz4 = bz * bz * bz * bz
                    ltx = ltx - rho_fl * bx * (by4 + bz4) * abs(wx) * wx / Scalar[DTYPE](64)
                    lty = lty - rho_fl * by * (bx4 + bz4) * abs(wy) * wy / Scalar[DTYPE](64)
                    ltz = ltz - rho_fl * bz * (bx4 + by4) * abs(wz) * wz / Scalar[DTYPE](64)

                # Rotate forces to world frame
                var fw_b = gpu_quat_rotate[DTYPE](qx_b, qy_b, qz_b, qw_b, lfx, lfy, lfz)
                var tw_b = gpu_quat_rotate[DTYPE](qx_b, qy_b, qz_b, qw_b, ltx, lty, ltz)
                var fx_w = fw_b[0]; var fy_w = fw_b[1]; var fz_w = fw_b[2]
                var tx_w = tw_b[0]; var ty_w = tw_b[1]; var tz_w = tw_b[2]

                # Apply wrench at xipos via Jacobian transpose (kinematic tree walk)
                var px_b = rebind[Scalar[DTYPE]](state[env, xipos_off + b * 3 + 0])
                var py_b = rebind[Scalar[DTYPE]](state[env, xipos_off + b * 3 + 1])
                var pz_b = rebind[Scalar[DTYPE]](state[env, xipos_off + b * 3 + 2])
                var tau_ox = tx_w + py_b * fz_w - pz_b * fy_w
                var tau_oy = ty_w + pz_b * fx_w - px_b * fz_w
                var tau_oz = tz_w + px_b * fy_w - py_b * fx_w

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
                            var ca0 = rebind[Scalar[DTYPE]](workspace[env, cdof_off + di2 * 6 + 0])
                            var ca1 = rebind[Scalar[DTYPE]](workspace[env, cdof_off + di2 * 6 + 1])
                            var ca2 = rebind[Scalar[DTYPE]](workspace[env, cdof_off + di2 * 6 + 2])
                            var cl0 = rebind[Scalar[DTYPE]](workspace[env, cdof_off + di2 * 6 + 3])
                            var cl1 = rebind[Scalar[DTYPE]](workspace[env, cdof_off + di2 * 6 + 4])
                            var cl2 = rebind[Scalar[DTYPE]](workspace[env, cdof_off + di2 * 6 + 5])
                            var cur2 = rebind[Scalar[DTYPE]](workspace[env, fnet_idx + di2])
                            workspace[env, fnet_idx + di2] = (
                                cur2
                                + cl0 * fx_w + cl1 * fy_w + cl2 * fz_w
                                + ca0 * tau_ox + ca1 * tau_oy + ca2 * tau_oz
                            )
                    var anc_off = model_body_offset(anc)
                    anc = Int(rebind[Scalar[DTYPE]](model[0, anc_off + BODY_IDX_PARENT]))

        # LDL solve: reads L, D, f_net from workspace, writes qacc to workspace
        ldl_solve_workspace_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](
            env, workspace
        )

        for i in range(NV):
            var qacc_val = rebind[Scalar[DTYPE]](
                workspace[env, qacc_ws_idx + i]
            )
            state[env, qacc_off + i] = qacc_val

        # 10. Write unconstrained qacc to workspace for constraint solver
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
        var qacc_constrained_idx = ws_qacc_constrained_offset[NV, NBODY]()
        comptime M_idx = ws_M_offset[NV, NBODY]()
        comptime fnet_idx = ws_fnet_offset[NV, NBODY]()
        comptime qacc_ws_idx = ws_qacc_ws_offset[NV, NBODY]()
        var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
        var dt = rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_TIMESTEP]
        )

        # 9. Integrate with implicit velocity damping (MuJoCo 3.x Euler)
        # Equivalent formula: v_new = v_old + dt * (M+dt*D)^{-1} * M * qacc_constrained
        # Same result as: (M+dt*D)*v_new = M*(v_old+dt*qacc) + dt*D*v_old
        # but using the compact form (identical to ImplicitFastIntegrator approach).

        var qpos_off = qpos_offset[NQ, NV]()

        # Step 1: rhs = M * qacc_constrained (store in fnet workspace)
        for i in range(NV):
            var sum = Scalar[DTYPE](0)
            for j in range(NV):
                var M_ij = rebind[Scalar[DTYPE]](
                    workspace[env, M_idx + i * NV + j]
                )
                var qacc_j = rebind[Scalar[DTYPE]](
                    workspace[env, qacc_constrained_idx + j]
                )
                sum += M_ij * qacc_j
            workspace[env, fnet_idx + i] = sum

        # Step 2: M_hat = M + dt*D (add damping to M diagonal)
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
            if damp > Scalar[DTYPE](0):
                if jnt_type == JNT_FREE:
                    for d in range(6):
                        var idx = M_idx + (dof_adr + d) * NV + (dof_adr + d)
                        workspace[env, idx] += dt * damp
                elif jnt_type == JNT_BALL:
                    for d in range(3):
                        var idx = M_idx + (dof_adr + d) * NV + (dof_adr + d)
                        workspace[env, idx] += dt * damp
                else:
                    var idx = M_idx + dof_adr * NV + dof_adr
                    workspace[env, idx] += dt * damp

        # Step 3: Re-factor M_hat via LDL (reuses L/D workspace regions)
        ldl_factor_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](env, workspace)

        # Step 4: qacc_final = M_hat^{-1} * rhs (reads fnet, writes qacc_ws)
        ldl_solve_workspace_gpu[DTYPE, NV, NBODY, BATCH, WS_SIZE](
            env, workspace
        )

        # Step 5: v_new = v_old + dt * qacc_final
        for i in range(NV):
            var old_qvel = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
            var qacc_final = rebind[Scalar[DTYPE]](workspace[env, qacc_ws_idx + i])
            state[env, qacc_off + i] = qacc_final
            state[env, qvel_off + i] = old_qvel + qacc_final * dt

        # Integrate position: qpos += qvel * dt (quaternion-aware for free joints)
        # (reuse model_meta_off from line 890)
        var num_joints = Int(
            rebind[Scalar[DTYPE]](
                model[0, model_meta_off + MODEL_META_IDX_NJOINT]
            )
        )

        for j in range(num_joints):
            var joint_off = model_joint_offset[NBODY](j)
            var jnt_type = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
            )
            var jnt_qpos_adr = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_QPOS_ADR])
            )
            var jnt_dof_adr = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR])
            )

            if jnt_type == JNT_FREE:
                # Position: simple addition
                for d in range(3):
                    var qp = rebind[Scalar[DTYPE]](
                        state[env, qpos_off + jnt_qpos_adr + d]
                    )
                    var qv = rebind[Scalar[DTYPE]](
                        state[env, qvel_off + jnt_dof_adr + d]
                    )
                    state[env, qpos_off + jnt_qpos_adr + d] = qp + qv * dt
                # Quaternion: exponential map integration
                var qx = rebind[Scalar[DTYPE]](
                    state[env, qpos_off + jnt_qpos_adr + 3]
                )
                var qy = rebind[Scalar[DTYPE]](
                    state[env, qpos_off + jnt_qpos_adr + 4]
                )
                var qz = rebind[Scalar[DTYPE]](
                    state[env, qpos_off + jnt_qpos_adr + 5]
                )
                var qw = rebind[Scalar[DTYPE]](
                    state[env, qpos_off + jnt_qpos_adr + 6]
                )
                var wx = rebind[Scalar[DTYPE]](
                    state[env, qvel_off + jnt_dof_adr + 3]
                )
                var wy = rebind[Scalar[DTYPE]](
                    state[env, qvel_off + jnt_dof_adr + 4]
                )
                var wz = rebind[Scalar[DTYPE]](
                    state[env, qvel_off + jnt_dof_adr + 5]
                )
                var result = quat_integrate(qx, qy, qz, qw, wx, wy, wz, dt)
                state[env, qpos_off + jnt_qpos_adr + 3] = result[0]
                state[env, qpos_off + jnt_qpos_adr + 4] = result[1]
                state[env, qpos_off + jnt_qpos_adr + 5] = result[2]
                state[env, qpos_off + jnt_qpos_adr + 6] = result[3]

            elif jnt_type == JNT_HINGE or jnt_type == JNT_SLIDE:
                var qp = rebind[Scalar[DTYPE]](
                    state[env, qpos_off + jnt_qpos_adr]
                )
                var qv = rebind[Scalar[DTYPE]](
                    state[env, qvel_off + jnt_dof_adr]
                )
                state[env, qpos_off + jnt_qpos_adr] = qp + qv * dt

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
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        MAX_TENDON: Int = 0,
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[DTYPE],
        mut model_buf: DeviceBuffer[DTYPE],
        mut workspace_buf: DeviceBuffer[DTYPE],
    ) raises:
        """Perform one physics simulation step on GPU with constraint solving.

        Uses the parametrized SOLVER for contact constraint resolution.
        """
        comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS]()
        comptime MODEL_SIZE = model_size_with_invweight[
            NBODY, NJOINT, NV, NGEOM, NEQUALITY=MAX_EQUALITY
        ]()
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
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
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
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        MAX_TENDON: Int = 0,
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
            ](
                ctx,
                state_buf,
                model_buf,
                workspace_buf,
            )


# Backward-compatible alias: uses PGS solver by default
comptime EulerDefaultIntegrator = EulerIntegrator[PGSSolver]

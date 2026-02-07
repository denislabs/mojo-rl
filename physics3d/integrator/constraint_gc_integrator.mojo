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

from ..types import ModelGC, DataGC, _max_one
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from ..kinematics.quat_math import quat_normalize, quat_integrate, quat_rotate
from ..dynamics.mass_matrix import (
    compute_mass_matrix,
    compute_mass_matrix_full,
    ldl_factor,
    ldl_solve,
    compute_M_inv_from_ldl,
    solve_linear_diagonal,
)
from ..dynamics.bias_forces import compute_bias_forces, compute_bias_forces_rne
from ..dynamics.jacobian import compute_cdof, compute_composite_inertia
from ..solver.semi_implicit_euler_solver import (
    detect_ground_contacts,
    detect_body_body_contacts_gc,
    normalize_qpos_quaternions,
    enforce_joint_limits,
)
from ..solver.gc_pgs_solver import GcPGSSolver
from ..traits.integrator import GcIntegrator
from ..traits.gc_solver import GcConstraintSolver
from ..gpu.constants import (
    TPB,
    gc_state_size,
    gc_model_size,
)
from ..gpu.gc_kernels import (
    step_gc_constraint_kernel,
    step_gc_constraint_kernel_with_solver,
)


struct ConstraintGcIntegratorWith[SOLVER: GcConstraintSolver](GcIntegrator):
    """GC integrator with configurable constraint-based contact solving.

    Parametrized by SOLVER type (GcPGSSolver, GcCGSolver, or GcNewtonSolver).
    Uses the specified solver for contact constraints instead of penalty springs.

    Usage:
        # PGS (default, backward-compatible):
        alias PGSIntegrator = ConstraintGcIntegratorWith[GcPGSSolver]

        # Conjugate Gradient:
        alias CGIntegrator = ConstraintGcIntegratorWith[GcCGSolver]

        # Newton:
        alias NewtonIntegrator = ConstraintGcIntegratorWith[GcNewtonSolver]
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
        model: ModelGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        mut data: DataGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
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
        detect_body_body_contacts_gc(model, data)

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
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, M_SIZE, CDOF_SIZE, CRB_SIZE
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
                    M[(dof_adr + d) * NV + (dof_adr + d)] = M[(dof_adr + d) * NV + (dof_adr + d)] + diag_add
            elif joint.jnt_type == JNT_BALL:
                for d in range(3):
                    M[(dof_adr + d) * NV + (dof_adr + d)] = M[(dof_adr + d) * NV + (dof_adr + d)] + diag_add
            else:
                M[dof_adr * NV + dof_adr] = M[dof_adr * NV + dof_adr] + diag_add

        # 6. LDL factorize M and solve for qacc
        var L = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        var D = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        ldl_factor[DTYPE, NV, M_SIZE, V_SIZE](M, L, D)

        var bias = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(V_SIZE):
            bias[i] = Scalar[DTYPE](0)
        compute_bias_forces_rne[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE, CDOF_SIZE](
            model, data, cdof, bias
        )

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
                        f_net[dof_adr + d] = f_net[dof_adr + d] - stiff * data.qpos[qpos_adr + d]
                elif joint.jnt_type == JNT_BALL:
                    for d in range(3):
                        f_net[dof_adr + d] = f_net[dof_adr + d] - stiff * data.qpos[qpos_adr + d]
                else:
                    # Hinge/slide: f = -stiffness * qpos
                    f_net[dof_adr] = f_net[dof_adr] - stiff * data.qpos[qpos_adr]

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
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE, M_SIZE, CDOF_SIZE
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

        # 11. Enforce joint limits
        enforce_joint_limits(model, data)

    @staticmethod
    fn simulate[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
    ](
        model: ModelGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        mut data: DataGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        num_steps: Int,
    ):
        """Run simulation for multiple steps on CPU."""
        for _ in range(num_steps):
            Self.step(model, data)

    # =========================================================================
    # GPU Methods
    # =========================================================================

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
        dt: Scalar[DTYPE],
        gravity_z: Scalar[DTYPE],
        ground_z: Scalar[DTYPE],
    ) raises:
        """Perform one physics simulation step on GPU with constraint solving.

        Uses the parametrized SOLVER for contact constraint resolution.
        """
        comptime STATE_SIZE = gc_state_size[NQ, NV, NBODY, MAX_CONTACTS]()
        comptime MODEL_SIZE = gc_model_size[NBODY, NJOINT]()
        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        var state = LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ](state_buf.unsafe_ptr())

        var model = LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ](model_buf.unsafe_ptr())

        @always_inline
        fn kernel_wrapper(
            state: LayoutTensor[
                DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
            ],
            model: LayoutTensor[
                DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH:
                return

            step_gc_constraint_kernel_with_solver[
                DTYPE,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                STATE_SIZE,
                MODEL_SIZE,
                BATCH,
                Self.SOLVER,
            ](env, state, model)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            state,
            model,
            grid_dim=(BLOCKS,),
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
        num_steps: Int,
        dt: Scalar[DTYPE],
        gravity_z: Scalar[DTYPE],
        ground_z: Scalar[DTYPE],
    ) raises:
        """Run simulation for multiple steps on GPU."""
        for _ in range(num_steps):
            Self.step_gpu[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, BATCH](
                ctx, state_buf, model_buf, dt, gravity_z, ground_z
            )


# Backward-compatible alias: uses PGS solver by default
comptime ConstraintGcIntegrator = ConstraintGcIntegratorWith[GcPGSSolver]

"""Constraint-based GC integrator with MuJoCo-style contact solving.

Replaces penalty-spring contacts with Projected Gauss-Seidel (PGS) constraint
solving. The new pipeline:

1. Forward kinematics (qpos -> xpos, xquat)
2. Compute body velocities (qvel -> xvel, xangvel)
3. Detect ground contacts
4. Compute mass matrix diagonal
5. Compute bias forces
6. Compute cdof (spatial motion axes per DOF)
7. Compute unconstrained acceleration: qacc = M^-1 * (qfrc - bias)
8. Predict velocity: qvel_pred = qvel + qacc * dt
9. PGS constraint solve: modify qvel_pred to satisfy contacts
10. qpos += qvel_pred * dt
11. Normalize quaternions, enforce joint limits

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
from ..dynamics.mass_matrix import compute_mass_matrix, solve_linear_diagonal
from ..dynamics.bias_forces import compute_bias_forces
from ..dynamics.jacobian import compute_cdof
from ..solver.semi_implicit_euler_solver import (
    detect_ground_contacts,
    normalize_qpos_quaternions,
    enforce_joint_limits,
)
from ..solver.gc_pgs_solver import GcPGSSolver
from ..traits.integrator import GcIntegrator
from ..gpu.constants import (
    TPB,
    gc_state_size,
    gc_model_size,
)
from ..gpu.gc_kernels import step_gc_constraint_kernel


struct ConstraintGcIntegrator(GcIntegrator):
    """GC integrator with MuJoCo-style constraint-based contact solving.

    Uses PGS to solve contact constraints instead of penalty springs.
    This produces bounded contact forces and prevents bodies from being
    launched by deep penetration.
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

        # 1. Forward kinematics
        forward_kinematics(model, data)
        compute_body_velocities(model, data)

        # 2. Collision detection
        detect_ground_contacts(model, data)

        # 3. Compute dynamics
        var M = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
        for i in range(M_SIZE):
            M[i] = Scalar[DTYPE](0)
        compute_mass_matrix[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, M_SIZE](
            model, data, M
        )

        var bias = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(V_SIZE):
            bias[i] = Scalar[DTYPE](0)
        compute_bias_forces[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE](
            model, data, bias
        )

        # 4. Extract diagonal of mass matrix
        var M_diag = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NV):
            M_diag[i] = M[i * NV + i]

        # 5. Compute cdof (spatial motion axes per DOF)
        var cdof = InlineArray[Scalar[DTYPE], CDOF_SIZE](uninitialized=True)
        compute_cdof[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CDOF_SIZE](
            model, data, cdof
        )

        # 6. Compute unconstrained acceleration: qacc = M^-1 * (qfrc - bias)
        var f_net = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NV):
            f_net[i] = data.qfrc[i] - bias[i]

        var qacc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NV):
            if M_diag[i] > Scalar[DTYPE](1e-10):
                qacc[i] = f_net[i] / M_diag[i]
            else:
                qacc[i] = Scalar[DTYPE](0)

        # 7. Predict velocity: qvel_pred = qvel + qacc * dt
        var qvel_pred = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
        for i in range(NV):
            qvel_pred[i] = data.qvel[i] + qacc[i] * dt

        # 8. PGS constraint solve (modifies qvel_pred in-place)
        GcPGSSolver.solve[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE, CDOF_SIZE
        ](model, data, M_diag, cdof, qvel_pred, dt)

        # 9. Write back constrained velocity and integrate position
        for i in range(NV):
            # qacc = (constrained_vel - old_vel) / dt
            data.qacc[i] = (qvel_pred[i] - data.qvel[i]) / dt
            data.qvel[i] = qvel_pred[i]

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

        Uses step_gc_constraint_kernel which replaces penalty springs with PGS.
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

            step_gc_constraint_kernel[
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

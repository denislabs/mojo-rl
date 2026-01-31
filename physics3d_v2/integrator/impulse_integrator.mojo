"""Impulse-based physics integrator (Bullet/Box2D style).

Uses Split Impulse approach for stable stacking:
- Velocity constraints handle collision response
- Position constraints directly correct penetration
- Resting contact detection prevents drift
- Joint constraints for articulated bodies

Reference: Erin Catto's GDC presentations on constraint solving.

Supports both CPU (step/simulate) and GPU (step_gpu/simulate_gpu) execution.
"""

from gpu.host import DeviceContext, DeviceBuffer
from gpu import thread_idx, block_idx, block_dim
from layout import LayoutTensor, Layout
from ..types import Model, Data
from ..traits import Integrator
from ..collision import CollisionDetector
from ..solver import (
    solve_velocity_constraints,
    solve_position_constraints,
    solve_resting_contacts,
    apply_gravity_gpu,
    solve_velocity_constraints_gpu,
    solve_position_constraints_gpu,
)
from ..joints import (
    solve_joint_velocity_constraints,
    solve_joint_position_constraints,
    solve_joint_velocity_constraints_gpu,
    solve_joint_position_constraints_gpu,
)
from ..gpu.constants import MODEL_BODY_SIZE, TPB, compute_state_size
from math import sqrt
from .integrate_positions import integrate_positions_kernel


struct ImpulseIntegrator(Integrator):
    """Impulse-based physics integrator.

    Uses Split Impulse method (similar to Bullet Physics / Box2D):
    - Velocity solver: Only handles velocity constraints (stopping/bouncing)
    - Position solver: Uses pseudo-velocities that don't affect real velocities

    This separation prevents position correction from adding energy to the system,
    which is critical for stable stacking.

    Supports joint constraints for articulated bodies (pendulums, chains, etc).

    Implements both CPU and GPU execution paths through the Integrator trait.
    """

    # =========================================================================
    # CPU Methods
    # =========================================================================

    @staticmethod
    fn step[
        DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int = 0
    ](
        model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS],
        mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS],
    ):
        """Perform one physics simulation step on CPU.

        Pipeline:
        1. Collision detection (pre-step)
        2. Apply gravity to velocities
        3. Solve velocity constraints (collision response)
        4. Solve joint velocity constraints (if any)
        5. Handle resting contacts (gravity cancellation)
        6. Integrate positions
        7. Integrate angular positions (quaternions)
        8. Collision detection (post-step)
        9. Solve position constraints (penetration correction)
        10. Solve joint position constraints (if any)

        Args:
            model: Static model configuration.
            data: Mutable simulation state.
        """
        var dt = model.timestep

        # 1. Collision detection (pre-step)
        CollisionDetector.detect_all_contacts(model, data)

        # 2. Apply gravity to velocities
        for i in range(NUM_BODIES):
            data.velocities[i * 3 + 2] += dt * model.gravity_z

        # 3. Solve velocity constraints (collision response with restitution)
        solve_velocity_constraints(model, data, iterations=30)

        # 4. Solve joint velocity constraints (if any)
        @parameter
        if MAX_JOINTS > 0:
            solve_joint_velocity_constraints(model, data, iterations=5)

        # 5. Handle resting contacts
        # For bodies at rest on support, clamp small downward velocities
        solve_resting_contacts(model, data)

        # 6. Integrate positions
        for i in range(NUM_BODIES * 3):
            data.positions[i] += dt * data.velocities[i]

        # 7. Integrate angular positions (quaternions)
        # q' = q + 0.5*dt*ω⊗q
        var half_dt = dt * Scalar[DTYPE](0.5)
        for i in range(NUM_BODIES):
            var wx = data.angular_velocities[i * 3 + 0]
            var wy = data.angular_velocities[i * 3 + 1]
            var wz = data.angular_velocities[i * 3 + 2]
            var qx = data.quaternions[i * 4 + 0]
            var qy = data.quaternions[i * 4 + 1]
            var qz = data.quaternions[i * 4 + 2]
            var qw = data.quaternions[i * 4 + 3]

            # Quaternion derivative: ω ⊗ q
            var qx_new = qx + half_dt * (wx * qw + wy * qz - wz * qy)
            var qy_new = qy + half_dt * (-wx * qz + wy * qw + wz * qx)
            var qz_new = qz + half_dt * (wx * qy - wy * qx + wz * qw)
            var qw_new = qw + half_dt * (-wx * qx - wy * qy - wz * qz)

            # Normalize
            var norm_sq = qx_new * qx_new + qy_new * qy_new + qz_new * qz_new + qw_new * qw_new
            if norm_sq > Scalar[DTYPE](1e-10):
                var inv_norm = Scalar[DTYPE](1.0) / sqrt(norm_sq)
                data.quaternions[i * 4 + 0] = qx_new * inv_norm
                data.quaternions[i * 4 + 1] = qy_new * inv_norm
                data.quaternions[i * 4 + 2] = qz_new * inv_norm
                data.quaternions[i * 4 + 3] = qw_new * inv_norm

        # 8. Collision detection (post-step)
        CollisionDetector.detect_all_contacts(model, data)

        # 9. Solve position constraints (direct penetration correction)
        # Use aggressive correction with many iterations for stable stacking
        for _ in range(15):
            solve_position_constraints(
                model,
                data,
                baumgarte=Scalar[DTYPE](1.0),
                slop=Scalar[DTYPE](0.00001),
            )
            # Re-detect to get updated penetration depths
            CollisionDetector.detect_all_contacts(model, data)

        # 10. Solve joint position constraints (if any)
        @parameter
        if MAX_JOINTS > 0:
            solve_joint_position_constraints(
                model, data, baumgarte=Scalar[DTYPE](0.2), iterations=5
            )

        # Final resting contact handling after position correction
        solve_resting_contacts(model, data)

    @staticmethod
    fn simulate[
        DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int = 0
    ](
        model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS],
        mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS],
        num_steps: Int,
    ):
        """Run simulation for multiple steps on CPU.

        Args:
            model: Static model configuration.
            data: Mutable simulation state (will be modified).
            num_steps: Number of simulation steps to run.
        """
        for _ in range(num_steps):
            Self.step(model, data)

    # =========================================================================
    # GPU Methods
    # =========================================================================

    # =========================================================================
    # Complete Physics Step (Impulse-based)
    # =========================================================================

    @always_inline
    @staticmethod
    fn step_impulse_kernel[
        DTYPE: DType,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
        MAX_JOINTS: Int = 0,
        STATE_SIZE: Int = 0,
        BATCH: Int = 1,
    ](
        env: Int,
        state: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        model: LayoutTensor[
            DTYPE, Layout.row_major(NUM_BODIES, MODEL_BODY_SIZE), MutAnyOrigin
        ],
        dt: Scalar[DTYPE],
        gravity_z: Scalar[DTYPE],
        ground_z: Scalar[DTYPE],
        restitution: Scalar[DTYPE],
    ):
        """Complete impulse-based physics step for one environment."""
        # 1. Collision detection
        CollisionDetector.detect_all_contacts_gpu[
            DTYPE, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BATCH
        ](env, state, model, ground_z)

        # 2. Apply gravity
        apply_gravity_gpu[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, STATE_SIZE, BATCH](
            env, state, dt, gravity_z
        )

        # 3. Solve velocity constraints
        solve_velocity_constraints_gpu[
            DTYPE, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BATCH
        ](env, state, model, restitution, 10)

        # 4. Solve joint velocity constraints
        @parameter
        if MAX_JOINTS > 0:
            solve_joint_velocity_constraints_gpu[
                DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, STATE_SIZE, BATCH
            ](env, state, model, 5)

        # 5. Integrate positions
        integrate_positions_kernel[
            DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, STATE_SIZE, BATCH
        ](env, state, dt)

        # 6. Post-step collision detection
        CollisionDetector.detect_all_contacts_gpu[
            DTYPE, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BATCH
        ](env, state, model, ground_z)

        # 7. Position correction
        solve_position_constraints_gpu[
            DTYPE, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BATCH
        ](env, state, model, Scalar[DTYPE](0.8), Scalar[DTYPE](0.001))

        # 8. Solve joint position constraints
        @parameter
        if MAX_JOINTS > 0:
            solve_joint_position_constraints_gpu[
                DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, STATE_SIZE, BATCH
            ](env, state, model, Scalar[DTYPE](0.2), 5)

    @staticmethod
    fn step_gpu[
        DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int = 0, BATCH: Int = 1
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[DTYPE],
        mut model_buf: DeviceBuffer[DTYPE],
        dt: Scalar[DTYPE],
        gravity_z: Scalar[DTYPE],
        ground_z: Scalar[DTYPE],
        restitution: Scalar[DTYPE],
        friction: Scalar[DTYPE],
    ) raises:
        """Perform one physics simulation step on GPU.

        Runs physics for all BATCH environments in parallel.
        Uses impulse-based constraint solving with Split Impulse method.

        Args:
            ctx: GPU device context.
            state_buf: Device buffer containing state for all environments.
            model_buf: Device buffer containing per-body model data.
            dt: Timestep.
            gravity_z: Z-component of gravity.
            ground_z: Ground plane height.
            restitution: Coefficient of restitution.
            friction: Friction coefficient (currently unused).
        """
        comptime STATE_SIZE = compute_state_size[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()
        comptime BLOCKS = (BATCH + TPB - 1) // TPB

        var state = LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ](state_buf.unsafe_ptr())

        var model = LayoutTensor[
            DTYPE, Layout.row_major(NUM_BODIES, MODEL_BODY_SIZE), MutAnyOrigin
        ](model_buf.unsafe_ptr())

        @always_inline
        fn kernel_wrapper(
            state: LayoutTensor[
                DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
            ],
            model: LayoutTensor[
                DTYPE,
                Layout.row_major(NUM_BODIES, MODEL_BODY_SIZE),
                MutAnyOrigin,
            ],
            dt: Scalar[DTYPE],
            gravity_z: Scalar[DTYPE],
            ground_z: Scalar[DTYPE],
            restitution: Scalar[DTYPE],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH:
                return

            Self.step_impulse_kernel[
                DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, STATE_SIZE, BATCH
            ](env, state, model, dt, gravity_z, ground_z, restitution)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            state,
            model,
            dt,
            gravity_z,
            ground_z,
            restitution,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn simulate_gpu[
        DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int = 0, BATCH: Int = 1
    ](
        ctx: DeviceContext,
        mut state_buf: DeviceBuffer[DTYPE],
        mut model_buf: DeviceBuffer[DTYPE],
        num_steps: Int,
        dt: Scalar[DTYPE],
        gravity_z: Scalar[DTYPE],
        ground_z: Scalar[DTYPE],
        restitution: Scalar[DTYPE],
        friction: Scalar[DTYPE],
    ) raises:
        """Run simulation for multiple steps on GPU.

        Args:
            ctx: GPU device context.
            state_buf: Device buffer containing state for all environments.
            model_buf: Device buffer containing per-body model data.
            num_steps: Number of simulation steps to run.
            dt: Timestep.
            gravity_z: Z-component of gravity.
            ground_z: Ground plane height.
            restitution: Coefficient of restitution.
            friction: Friction coefficient (currently unused).
        """
        for _ in range(num_steps):
            Self.step_gpu[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS, BATCH](
                ctx,
                state_buf,
                model_buf,
                dt,
                gravity_z,
                ground_z,
                restitution,
                friction,
            )

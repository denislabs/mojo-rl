"""LunarLanderV2 GPU environment using the physics_gpu architecture.

This implementation uses the existing physics components with strided methods:
- PhysicsStateStrided for accessing physics data in flat layout
- SemiImplicitEuler for velocity/position integration
- EdgeTerrainCollision for terrain collision detection
- ImpulseSolver for contact resolution
- RevoluteJointSolver for leg joints

The flat state layout is compatible with GPUDiscreteEnv trait.
All physics data is packed per-environment for efficient GPU access.

This follows the same patterns as lunar_lander_v2.mojo but adapted for
the GPUDiscreteEnv trait's flat [BATCH, STATE_SIZE] layout.
"""

from math import sqrt, cos, sin, pi
from layout import Layout, LayoutTensor
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer

from core.env_traits import GPUDiscreteEnv
from physics_gpu import (
    dtype,
    TPB,
    BODY_STATE_SIZE,
    SHAPE_MAX_SIZE,
    CONTACT_DATA_SIZE,
    JOINT_DATA_SIZE,
    IDX_X,
    IDX_Y,
    IDX_ANGLE,
    IDX_VX,
    IDX_VY,
    IDX_OMEGA,
    IDX_INV_MASS,
    IDX_INV_INERTIA,
    IDX_SHAPE,
    SHAPE_POLYGON,
    JOINT_REVOLUTE,
    JOINT_TYPE,
    JOINT_BODY_A,
    JOINT_BODY_B,
    JOINT_ANCHOR_AX,
    JOINT_ANCHOR_AY,
    JOINT_ANCHOR_BX,
    JOINT_ANCHOR_BY,
    JOINT_REF_ANGLE,
    JOINT_LOWER_LIMIT,
    JOINT_UPPER_LIMIT,
    JOINT_STIFFNESS,
    JOINT_DAMPING,
    JOINT_FLAGS,
    JOINT_FLAG_LIMIT_ENABLED,
    JOINT_FLAG_SPRING_ENABLED,
    PhysicsStateStrided,
)
from physics_gpu.integrators.euler import SemiImplicitEuler
from physics_gpu.collision.edge_terrain import EdgeTerrainCollision, MAX_TERRAIN_EDGES
from physics_gpu.solvers.impulse import ImpulseSolver
from physics_gpu.joints.revolute import RevoluteJointSolver


# =============================================================================
# Physics Constants - Matched to Gymnasium LunarLander-v3
# =============================================================================

comptime GRAVITY_X: Float64 = 0.0
comptime GRAVITY_Y: Float64 = -10.0
comptime DT: Float64 = 0.02  # 50 FPS
comptime VELOCITY_ITERATIONS: Int = 6
comptime POSITION_ITERATIONS: Int = 2

# Lander geometry (matching Gymnasium)
comptime SCALE: Float64 = 30.0
comptime LEG_AWAY: Float64 = 20.0 / SCALE
comptime LEG_DOWN: Float64 = 18.0 / SCALE
comptime LEG_W: Float64 = 2.0 / SCALE
comptime LEG_H: Float64 = 8.0 / SCALE
comptime LANDER_HALF_HEIGHT: Float64 = 17.0 / SCALE
comptime LANDER_HALF_WIDTH: Float64 = 10.0 / SCALE

# Lander mass/inertia
comptime LANDER_MASS: Float64 = 5.0
comptime LANDER_INERTIA: Float64 = 2.0
comptime LEG_MASS: Float64 = 0.2
comptime LEG_INERTIA: Float64 = 0.02

# Leg joint properties
comptime LEG_SPRING_STIFFNESS: Float64 = 400.0
comptime LEG_SPRING_DAMPING: Float64 = 40.0

# Engine power
comptime MAIN_ENGINE_POWER: Float64 = 13.0
comptime SIDE_ENGINE_POWER: Float64 = 0.6

# Reward constants
comptime CRASH_PENALTY: Float64 = -100.0
comptime LAND_REWARD: Float64 = 100.0
comptime MAIN_ENGINE_FUEL_COST: Float64 = 0.30
comptime SIDE_ENGINE_FUEL_COST: Float64 = 0.03

# Viewport
comptime VIEWPORT_W: Float64 = 600.0
comptime VIEWPORT_H: Float64 = 400.0
comptime W_UNITS: Float64 = VIEWPORT_W / SCALE
comptime H_UNITS: Float64 = VIEWPORT_H / SCALE
comptime HELIPAD_Y: Float64 = H_UNITS / 4.0
comptime HELIPAD_X: Float64 = W_UNITS / 2.0

# Physics constants
comptime FRICTION: Float64 = 0.1
comptime RESTITUTION: Float64 = 0.0
comptime BAUMGARTE: Float64 = 0.2
comptime SLOP: Float64 = 0.005

# Terrain
comptime TERRAIN_CHUNKS: Int = 11

# Body indices (matching lunar_lander_v2.mojo)
comptime BODY_LANDER: Int = 0
comptime BODY_LEFT_LEG: Int = 1
comptime BODY_RIGHT_LEG: Int = 2


# =============================================================================
# State Layout for GPUDiscreteEnv
# =============================================================================

# Counts
comptime NUM_BODIES: Int = 3
comptime NUM_SHAPES: Int = 3
comptime MAX_CONTACTS: Int = 8
comptime MAX_JOINTS: Int = 2
comptime OBS_DIM_VAL: Int = 8
comptime NUM_ACTIONS_VAL: Int = 4

# Buffer sizes
comptime BODIES_SIZE: Int = NUM_BODIES * BODY_STATE_SIZE  # 3 * 13 = 39
comptime FORCES_SIZE: Int = NUM_BODIES * 3  # 3 * 3 = 9
comptime JOINTS_SIZE: Int = MAX_JOINTS * JOINT_DATA_SIZE  # 2 * 17 = 34
comptime EDGES_SIZE: Int = MAX_TERRAIN_EDGES * 6  # 16 * 6 = 96
comptime METADATA_SIZE: Int = 4  # step_count, total_reward, prev_shaping, done

# Offsets within each environment's state
comptime OBS_OFFSET: Int = 0
comptime BODIES_OFFSET: Int = OBS_OFFSET + OBS_DIM_VAL  # 8
comptime FORCES_OFFSET: Int = BODIES_OFFSET + BODIES_SIZE  # 47
comptime JOINTS_OFFSET: Int = FORCES_OFFSET + FORCES_SIZE  # 56
comptime JOINT_COUNT_OFFSET: Int = JOINTS_OFFSET + JOINTS_SIZE  # 90
comptime EDGES_OFFSET: Int = JOINT_COUNT_OFFSET + 1  # 91
comptime EDGE_COUNT_OFFSET: Int = EDGES_OFFSET + EDGES_SIZE  # 187
comptime METADATA_OFFSET: Int = EDGE_COUNT_OFFSET + 1  # 188

# Metadata field indices
comptime META_STEP_COUNT: Int = 0
comptime META_TOTAL_REWARD: Int = 1
comptime META_PREV_SHAPING: Int = 2
comptime META_DONE: Int = 3

# Total state size per environment
comptime STATE_SIZE_VAL: Int = METADATA_OFFSET + METADATA_SIZE  # 192

# Type alias for PhysicsStateStrided with our layout
comptime PhysicsHelper = PhysicsStateStrided[
    NUM_BODIES,
    STATE_SIZE_VAL,
    BODIES_OFFSET,
    FORCES_OFFSET,
    JOINTS_OFFSET,
    JOINT_COUNT_OFFSET,
    EDGES_OFFSET,
    EDGE_COUNT_OFFSET,
    MAX_JOINTS,
]


# =============================================================================
# LunarLanderV2GPU Environment
# =============================================================================

struct LunarLanderV2GPU(GPUDiscreteEnv):
    """LunarLander environment with full physics using strided GPU methods.

    This environment uses the existing physics_gpu architecture:
    - PhysicsStateStrided for accessing physics data in flat layout
    - SemiImplicitEuler.integrate_velocities_gpu_strided
    - SemiImplicitEuler.integrate_positions_gpu_strided
    - EdgeTerrainCollision.detect_gpu_strided
    - ImpulseSolver.solve_velocity_gpu_strided / solve_position_gpu_strided
    - RevoluteJointSolver.solve_velocity_gpu_strided / solve_position_gpu_strided

    The structure follows lunar_lander_v2.mojo patterns but adapted for
    the GPUDiscreteEnv trait's flat state layout.
    """

    # Required trait aliases
    comptime STATE_SIZE: Int = STATE_SIZE_VAL
    comptime OBS_DIM: Int = OBS_DIM_VAL
    comptime NUM_ACTIONS: Int = NUM_ACTIONS_VAL

    # =========================================================================
    # CPU Kernels
    # =========================================================================

    @staticmethod
    fn step_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        actions: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), ImmutAnyOrigin],
        rewards: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
        dones: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
        rng_seed: Scalar[DType.uint64],
    ):
        """CPU step kernel using PhysicsStateStrided."""
        # Flatten for 1D access
        var flat_states = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE * STATE_SIZE), MutAnyOrigin
        ](states.ptr)

        for env in range(BATCH_SIZE):
            # Create physics helper for this environment
            var physics = PhysicsHelper(env)
            var env_base = env * STATE_SIZE_VAL

            # Check if already done
            if rebind[Scalar[dtype]](flat_states[env_base + METADATA_OFFSET + META_DONE]) > Scalar[dtype](0.5):
                rewards[env] = Scalar[dtype](0)
                dones[env] = Scalar[dtype](1)
                continue

            # Get action and apply engine forces
            var action = Int(actions[env])
            LunarLanderV2GPU._apply_engine_forces_cpu[BATCH_SIZE * STATE_SIZE](
                physics, flat_states, action
            )

            # Simple physics integration (CPU version is simplified)
            LunarLanderV2GPU._integrate_physics_cpu[BATCH_SIZE * STATE_SIZE](
                physics, flat_states
            )

            # Update observation and compute reward
            var result = LunarLanderV2GPU._finalize_step_cpu[BATCH_SIZE * STATE_SIZE](
                physics, flat_states, action
            )
            rewards[env] = result[0]
            dones[env] = result[1]

    @staticmethod
    fn reset_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
    ):
        """CPU reset kernel."""
        var flat_states = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE * STATE_SIZE), MutAnyOrigin
        ](states.ptr)

        for env in range(BATCH_SIZE):
            # Use env index as seed for deterministic but varied initial states
            LunarLanderV2GPU._reset_env_cpu[BATCH_SIZE * STATE_SIZE](
                flat_states, env, env * 12345
            )

    @staticmethod
    fn selective_reset_kernel[
        BATCH_SIZE: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL), MutAnyOrigin
        ],
        dones: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
        rng_seed: Int,
    ):
        """CPU selective reset kernel."""
        var flat_states = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE * STATE_SIZE_VAL), MutAnyOrigin
        ](states.ptr)

        for env in range(BATCH_SIZE):
            if rebind[Scalar[dtype]](dones[env]) > Scalar[dtype](0.5):
                LunarLanderV2GPU._reset_env_cpu[BATCH_SIZE * STATE_SIZE_VAL](
                    flat_states, env, rng_seed + env
                )

    # =========================================================================
    # GPU Kernels
    # =========================================================================

    @staticmethod
    fn step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        mut rewards_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """GPU step kernel using strided physics methods.

        This follows the same physics step sequence as lunar_lander_v2.mojo:
        1. Apply engine forces based on action
        2. Integrate velocities (SemiImplicitEuler)
        3. Detect terrain collisions (EdgeTerrainCollision)
        4. Solve velocity constraints (ImpulseSolver + RevoluteJointSolver)
        5. Integrate positions (SemiImplicitEuler)
        6. Solve position constraints
        7. Finalize (update observations, compute rewards, check termination)
        """
        # Allocate workspace buffers (contacts are temporary)
        var contacts_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_SIZE * MAX_CONTACTS * CONTACT_DATA_SIZE
        )
        var contact_counts_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
        var shapes_buf = ctx.enqueue_create_buffer[dtype](NUM_SHAPES * SHAPE_MAX_SIZE)
        var edge_counts_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
        var joint_counts_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)

        # Initialize shapes (once, shared across environments)
        LunarLanderV2GPU._init_shapes_gpu(ctx, shapes_buf)

        # Extract counts from state (edge_count, joint_count per env)
        LunarLanderV2GPU._extract_counts_gpu[BATCH_SIZE](
            ctx, states_buf, edge_counts_buf, joint_counts_buf
        )

        # 1. Apply engine forces based on action
        LunarLanderV2GPU._apply_forces_gpu[BATCH_SIZE](ctx, states_buf, actions_buf)

        # 2. Integrate velocities using existing physics architecture
        SemiImplicitEuler.integrate_velocities_gpu_strided[
            BATCH_SIZE, NUM_BODIES, STATE_SIZE_VAL, BODIES_OFFSET, FORCES_OFFSET
        ](ctx, states_buf, Scalar[dtype](GRAVITY_X), Scalar[dtype](GRAVITY_Y), Scalar[dtype](DT))

        # 3. Detect terrain collisions
        EdgeTerrainCollision.detect_gpu_strided[
            BATCH_SIZE, NUM_BODIES, NUM_SHAPES, MAX_CONTACTS, MAX_TERRAIN_EDGES,
            STATE_SIZE_VAL, BODIES_OFFSET, EDGES_OFFSET
        ](ctx, states_buf, shapes_buf, edge_counts_buf, contacts_buf, contact_counts_buf)

        # 4. Solve velocity constraints (contacts + joints)
        for _ in range(VELOCITY_ITERATIONS):
            ImpulseSolver.solve_velocity_gpu_strided[
                BATCH_SIZE, NUM_BODIES, MAX_CONTACTS, STATE_SIZE_VAL, BODIES_OFFSET
            ](ctx, states_buf, contacts_buf, contact_counts_buf,
              Scalar[dtype](FRICTION), Scalar[dtype](RESTITUTION))

            RevoluteJointSolver.solve_velocity_gpu_strided[
                BATCH_SIZE, NUM_BODIES, MAX_JOINTS, STATE_SIZE_VAL, BODIES_OFFSET, JOINTS_OFFSET
            ](ctx, states_buf, joint_counts_buf, Scalar[dtype](DT))

        # 5. Integrate positions
        SemiImplicitEuler.integrate_positions_gpu_strided[
            BATCH_SIZE, NUM_BODIES, STATE_SIZE_VAL, BODIES_OFFSET
        ](ctx, states_buf, Scalar[dtype](DT))

        # 6. Solve position constraints
        for _ in range(POSITION_ITERATIONS):
            ImpulseSolver.solve_position_gpu_strided[
                BATCH_SIZE, NUM_BODIES, MAX_CONTACTS, STATE_SIZE_VAL, BODIES_OFFSET
            ](ctx, states_buf, contacts_buf, contact_counts_buf,
              Scalar[dtype](BAUMGARTE), Scalar[dtype](SLOP))

            RevoluteJointSolver.solve_position_gpu_strided[
                BATCH_SIZE, NUM_BODIES, MAX_JOINTS, STATE_SIZE_VAL, BODIES_OFFSET, JOINTS_OFFSET
            ](ctx, states_buf, joint_counts_buf, Scalar[dtype](BAUMGARTE), Scalar[dtype](SLOP))

        # 7. Finalize step (update obs, compute rewards, check termination)
        LunarLanderV2GPU._finalize_step_gpu[BATCH_SIZE](
            ctx, states_buf, actions_buf, contact_counts_buf, rewards_buf, dones_buf
        )

    @staticmethod
    fn reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](ctx: DeviceContext, mut states_buf: DeviceBuffer[dtype]) raises:
        """GPU reset kernel."""
        # Create LayoutTensor from buffer
        var states = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE * STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn reset_wrapper(
            states: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE * STATE_SIZE), MutAnyOrigin
            ],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            # Use env index as seed for deterministic but varied initial states
            LunarLanderV2GPU._reset_env_gpu[BATCH_SIZE * STATE_SIZE](states, i, i * 12345)

        ctx.enqueue_function[reset_wrapper, reset_wrapper](
            states,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn selective_reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
        rng_seed: UInt32,
    ) raises:
        """GPU selective reset kernel."""
        var states = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE * STATE_SIZE), MutAnyOrigin
        ](states_buf.unsafe_ptr())

        var dones = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn selective_reset_wrapper(
            states: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE * STATE_SIZE), MutAnyOrigin
            ],
            dones: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
            seed: Scalar[dtype],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            if rebind[Scalar[dtype]](dones[i]) > Scalar[dtype](0.5):
                LunarLanderV2GPU._reset_env_gpu[BATCH_SIZE * STATE_SIZE](states, i, Int(seed) + i)

        ctx.enqueue_function[selective_reset_wrapper, selective_reset_wrapper](
            states,
            dones,
            Scalar[dtype](rng_seed),
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # Helper Functions - CPU
    # =========================================================================

    @staticmethod
    fn _apply_engine_forces_cpu[
        TOTAL_SIZE: Int
    ](
        physics: PhysicsHelper,
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
        action: Int,
    ):
        """Apply engine forces based on action (matching lunar_lander_v2.mojo)."""
        # Clear forces first
        physics.clear_forces(states)

        if action == 0:
            return  # No-op

        var angle = Float64(physics.get_body_angle(states, BODY_LANDER))
        var tip_x = sin(angle)
        var tip_y = cos(angle)
        var side_x = -tip_y
        var side_y = tip_x

        if action == 2:  # Main engine
            var fx = -tip_x * MAIN_ENGINE_POWER
            var fy = tip_y * MAIN_ENGINE_POWER
            physics.set_force(states, BODY_LANDER, fx, fy, 0.0)

        elif action == 1:  # Left engine
            var fx = side_x * SIDE_ENGINE_POWER
            var fy = -side_y * SIDE_ENGINE_POWER
            # Side engine creates torque
            physics.set_force(states, BODY_LANDER, fx, fy, SIDE_ENGINE_POWER * 0.5)

        elif action == 3:  # Right engine
            var fx = -side_x * SIDE_ENGINE_POWER
            var fy = side_y * SIDE_ENGINE_POWER
            physics.set_force(states, BODY_LANDER, fx, fy, -SIDE_ENGINE_POWER * 0.5)

    @staticmethod
    fn _integrate_physics_cpu[
        TOTAL_SIZE: Int
    ](
        physics: PhysicsHelper,
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
    ):
        """Simple physics integration for CPU fallback."""
        var dt = Scalar[dtype](DT)
        var gx = Scalar[dtype](GRAVITY_X)
        var gy = Scalar[dtype](GRAVITY_Y)

        for body in range(NUM_BODIES):
            var body_offset = physics.body_offset(body)
            var force_offset = physics.force_offset(body)

            var vx = rebind[Scalar[dtype]](states[body_offset + IDX_VX])
            var vy = rebind[Scalar[dtype]](states[body_offset + IDX_VY])
            var omega = rebind[Scalar[dtype]](states[body_offset + IDX_OMEGA])
            var inv_mass = rebind[Scalar[dtype]](states[body_offset + IDX_INV_MASS])
            var inv_inertia = rebind[Scalar[dtype]](states[body_offset + IDX_INV_INERTIA])

            var fx = rebind[Scalar[dtype]](states[force_offset + 0])
            var fy = rebind[Scalar[dtype]](states[force_offset + 1])
            var tau = rebind[Scalar[dtype]](states[force_offset + 2])

            # Integrate velocity
            vx = vx + (fx * inv_mass + gx) * dt
            vy = vy + (fy * inv_mass + gy) * dt
            omega = omega + tau * inv_inertia * dt

            states[body_offset + IDX_VX] = vx
            states[body_offset + IDX_VY] = vy
            states[body_offset + IDX_OMEGA] = omega

            # Integrate position
            var x = rebind[Scalar[dtype]](states[body_offset + IDX_X]) + vx * dt
            var y = rebind[Scalar[dtype]](states[body_offset + IDX_Y]) + vy * dt
            var angle = rebind[Scalar[dtype]](states[body_offset + IDX_ANGLE]) + omega * dt

            states[body_offset + IDX_X] = x
            states[body_offset + IDX_Y] = y
            states[body_offset + IDX_ANGLE] = angle

    @staticmethod
    fn _finalize_step_cpu[
        TOTAL_SIZE: Int
    ](
        physics: PhysicsHelper,
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
        action: Int,
    ) -> Tuple[Scalar[dtype], Scalar[dtype]]:
        """Update observation and compute reward (CPU version)."""
        var env_base = physics.env_base

        # Get lander state
        var x = Float64(physics.get_body_x(states, BODY_LANDER))
        var y = Float64(physics.get_body_y(states, BODY_LANDER))
        var vx = Float64(physics.get_body_vx(states, BODY_LANDER))
        var vy = Float64(physics.get_body_vy(states, BODY_LANDER))
        var angle = Float64(physics.get_body_angle(states, BODY_LANDER))
        var omega = Float64(physics.get_body_omega(states, BODY_LANDER))

        # Normalize observation (matching Gymnasium)
        var x_norm = (x - HELIPAD_X) / (W_UNITS / 2.0)
        var y_norm = (y - (HELIPAD_Y + LEG_DOWN / SCALE)) / (H_UNITS / 2.0)
        var vx_norm = vx * (W_UNITS / 2.0) / 50.0
        var vy_norm = vy * (H_UNITS / 2.0) / 50.0
        var angle_norm = angle
        var omega_norm = 20.0 * omega / 50.0

        # Compute leg contacts (simplified - check if y is near terrain)
        var left_contact: Float64 = 0.0
        var right_contact: Float64 = 0.0
        var left_y = Float64(physics.get_body_y[TOTAL_SIZE](states, BODY_LEFT_LEG))
        var right_y = Float64(physics.get_body_y[TOTAL_SIZE](states, BODY_RIGHT_LEG))
        if left_y <= HELIPAD_Y + 0.1:
            left_contact = 1.0
        if right_y <= HELIPAD_Y + 0.1:
            right_contact = 1.0

        # Update observation in state
        var obs_base = env_base + OBS_OFFSET
        states[obs_base + 0] = Scalar[dtype](x_norm)
        states[obs_base + 1] = Scalar[dtype](y_norm)
        states[obs_base + 2] = Scalar[dtype](vx_norm)
        states[obs_base + 3] = Scalar[dtype](vy_norm)
        states[obs_base + 4] = Scalar[dtype](angle_norm)
        states[obs_base + 5] = Scalar[dtype](omega_norm)
        states[obs_base + 6] = Scalar[dtype](left_contact)
        states[obs_base + 7] = Scalar[dtype](right_contact)

        # Compute shaping reward (matching Gymnasium)
        var dist = sqrt(x_norm * x_norm + y_norm * y_norm)
        var speed = sqrt(vx_norm * vx_norm + vy_norm * vy_norm)
        var abs_angle = angle if angle >= 0.0 else -angle  # manual abs
        var shaping = (
            -100.0 * dist
            - 100.0 * speed
            - 100.0 * abs_angle
            + 10.0 * left_contact
            + 10.0 * right_contact
        )

        var prev_shaping = Float64(rebind[Scalar[dtype]](states[env_base + METADATA_OFFSET + META_PREV_SHAPING]))
        var reward = shaping - prev_shaping
        states[env_base + METADATA_OFFSET + META_PREV_SHAPING] = Scalar[dtype](shaping)

        # Fuel costs
        if action == 2:
            reward = reward - MAIN_ENGINE_FUEL_COST
        elif action == 1 or action == 3:
            reward = reward - SIDE_ENGINE_FUEL_COST

        # Check termination
        var done: Float64 = 0.0

        # Out of bounds
        if x_norm >= 1.0 or x_norm <= -1.0:
            done = 1.0
            reward = CRASH_PENALTY

        # Below ground or too high
        if y < 0.0 or y > H_UNITS * 1.5:
            done = 1.0
            reward = CRASH_PENALTY

        # Successful landing
        var both_legs = left_contact > 0.5 and right_contact > 0.5
        var speed_val = sqrt(vx * vx + vy * vy)
        var abs_omega = omega if omega >= 0.0 else -omega
        if both_legs and speed_val < 0.1 and abs_omega < 0.1:
            done = 1.0
            reward = reward + LAND_REWARD

        # Max steps
        var step_count = Float64(rebind[Scalar[dtype]](states[env_base + METADATA_OFFSET + META_STEP_COUNT]))
        if step_count > 1000.0:
            done = 1.0

        # Update metadata
        states[env_base + METADATA_OFFSET + META_STEP_COUNT] = Scalar[dtype](step_count + 1.0)
        states[env_base + METADATA_OFFSET + META_DONE] = Scalar[dtype](done)

        return (Scalar[dtype](reward), Scalar[dtype](done))

    @staticmethod
    fn _reset_env_cpu[
        TOTAL_SIZE: Int
    ](
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
        env: Int,
        seed: Int,
    ):
        """Reset a single environment (CPU version)."""
        var physics = PhysicsHelper(env)
        var env_base = env * STATE_SIZE_VAL

        # Simple RNG
        var rng = UInt64(seed * 1664525 + 1013904223)

        # Generate terrain heights
        var n_edges = TERRAIN_CHUNKS - 1
        physics.set_edge_count(states, n_edges)

        var x_spacing = W_UNITS / Float64(TERRAIN_CHUNKS - 1)
        for edge in range(n_edges):
            rng = rng * 6364136223846793005 + 1442695040888963407
            var rand1 = Float64(rng >> 33) / Float64(2147483647)
            rng = rng * 6364136223846793005 + 1442695040888963407
            var rand2 = Float64(rng >> 33) / Float64(2147483647)

            var x0 = Float64(edge) * x_spacing
            var x1 = Float64(edge + 1) * x_spacing

            # Random height with helipad flat area
            var y0: Float64
            var y1: Float64
            if edge >= TERRAIN_CHUNKS // 2 - 2 and edge < TERRAIN_CHUNKS // 2 + 2:
                y0 = HELIPAD_Y
                y1 = HELIPAD_Y
            else:
                y0 = HELIPAD_Y + (rand1 - 0.5) * 2.0
                y1 = HELIPAD_Y + (rand2 - 0.5) * 2.0

            # Compute normal (pointing up)
            var dx = x1 - x0
            var dy = y1 - y0
            var length = sqrt(dx * dx + dy * dy)
            var nx = -dy / length
            var ny = dx / length
            if ny < 0:
                nx = -nx
                ny = -ny

            physics.set_edge(states, edge, x0, y0, x1, y1, nx, ny)

        # Initialize lander at top center with random velocity
        rng = rng * 6364136223846793005 + 1442695040888963407
        var init_vx = (Float64(rng >> 33) / Float64(2147483647) - 0.5) * 2.0
        rng = rng * 6364136223846793005 + 1442695040888963407
        var init_vy = (Float64(rng >> 33) / Float64(2147483647) - 0.5) * 2.0

        physics.set_body_position(states, BODY_LANDER, HELIPAD_X, H_UNITS)
        physics.set_body_angle(states, BODY_LANDER, 0.0)
        physics.set_body_velocity(states, BODY_LANDER, init_vx, init_vy, 0.0)
        physics.set_body_mass(states, BODY_LANDER, LANDER_MASS, LANDER_INERTIA)
        physics.set_body_shape(states, BODY_LANDER, 0)

        # Initialize legs
        var left_leg_x = HELIPAD_X - LEG_AWAY
        var left_leg_y = H_UNITS - (10.0 / SCALE) - LEG_DOWN
        physics.set_body_position(states, BODY_LEFT_LEG, left_leg_x, left_leg_y)
        physics.set_body_angle(states, BODY_LEFT_LEG, 0.0)
        physics.set_body_velocity(states, BODY_LEFT_LEG, init_vx, init_vy, 0.0)
        physics.set_body_mass(states, BODY_LEFT_LEG, LEG_MASS, LEG_INERTIA)
        physics.set_body_shape(states, BODY_LEFT_LEG, 1)

        var right_leg_x = HELIPAD_X + LEG_AWAY
        var right_leg_y = H_UNITS - (10.0 / SCALE) - LEG_DOWN
        physics.set_body_position(states, BODY_RIGHT_LEG, right_leg_x, right_leg_y)
        physics.set_body_angle(states, BODY_RIGHT_LEG, 0.0)
        physics.set_body_velocity(states, BODY_RIGHT_LEG, init_vx, init_vy, 0.0)
        physics.set_body_mass(states, BODY_RIGHT_LEG, LEG_MASS, LEG_INERTIA)
        physics.set_body_shape(states, BODY_RIGHT_LEG, 2)

        # Initialize joints (connecting legs to lander)
        physics.clear_joints(states)
        _ = physics.add_revolute_joint(
            states,
            body_a=BODY_LANDER,
            body_b=BODY_LEFT_LEG,
            anchor_ax=-LEG_AWAY,
            anchor_ay=-10.0 / SCALE,
            anchor_bx=0.0,
            anchor_by=LEG_H,
            stiffness=LEG_SPRING_STIFFNESS,
            damping=LEG_SPRING_DAMPING,
            lower_limit=0.4,
            upper_limit=0.9,
            enable_limit=True,
        )
        _ = physics.add_revolute_joint(
            states,
            body_a=BODY_LANDER,
            body_b=BODY_RIGHT_LEG,
            anchor_ax=LEG_AWAY,
            anchor_ay=-10.0 / SCALE,
            anchor_bx=0.0,
            anchor_by=LEG_H,
            stiffness=LEG_SPRING_STIFFNESS,
            damping=LEG_SPRING_DAMPING,
            lower_limit=-0.9,
            upper_limit=-0.4,
            enable_limit=True,
        )

        # Clear forces
        physics.clear_forces(states)

        # Initialize observation
        var x_norm = 0.0  # (HELIPAD_X - HELIPAD_X) / (W_UNITS / 2.0)
        var y_norm = (H_UNITS - (HELIPAD_Y + LEG_DOWN / SCALE)) / (H_UNITS / 2.0)
        var vx_norm = init_vx * (W_UNITS / 2.0) / 50.0
        var vy_norm = init_vy * (H_UNITS / 2.0) / 50.0

        var obs_base = env_base + OBS_OFFSET
        states[obs_base + 0] = Scalar[dtype](x_norm)
        states[obs_base + 1] = Scalar[dtype](y_norm)
        states[obs_base + 2] = Scalar[dtype](vx_norm)
        states[obs_base + 3] = Scalar[dtype](vy_norm)
        states[obs_base + 4] = Scalar[dtype](0)
        states[obs_base + 5] = Scalar[dtype](0)
        states[obs_base + 6] = Scalar[dtype](0)
        states[obs_base + 7] = Scalar[dtype](0)

        # Initialize metadata
        var meta_base = env_base + METADATA_OFFSET
        states[meta_base + META_STEP_COUNT] = Scalar[dtype](0)
        states[meta_base + META_TOTAL_REWARD] = Scalar[dtype](0)
        states[meta_base + META_PREV_SHAPING] = Scalar[dtype](0)
        states[meta_base + META_DONE] = Scalar[dtype](0)

    # =========================================================================
    # Helper Functions - GPU
    # =========================================================================

    @always_inline
    @staticmethod
    fn _reset_env_gpu[
        TOTAL_SIZE: Int,
    ](
        states: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
        env: Int,
        seed: Int,
    ):
        """Reset a single environment (GPU version)."""
        var env_base = env * STATE_SIZE_VAL

        # Simple RNG
        var rng = UInt64(seed * 1664525 + 1013904223)

        # Generate terrain
        var n_edges = TERRAIN_CHUNKS - 1
        states[env_base + EDGE_COUNT_OFFSET] = Scalar[dtype](n_edges)

        var x_spacing = W_UNITS / Float64(TERRAIN_CHUNKS - 1)
        for edge in range(n_edges):
            rng = rng * 6364136223846793005 + 1442695040888963407
            var rand1 = Float64(rng >> 33) / Float64(2147483647)
            rng = rng * 6364136223846793005 + 1442695040888963407
            var rand2 = Float64(rng >> 33) / Float64(2147483647)

            var x0 = Float64(edge) * x_spacing
            var x1 = Float64(edge + 1) * x_spacing
            var y0: Float64
            var y1: Float64

            if edge >= TERRAIN_CHUNKS // 2 - 2 and edge < TERRAIN_CHUNKS // 2 + 2:
                y0 = HELIPAD_Y
                y1 = HELIPAD_Y
            else:
                y0 = HELIPAD_Y + (rand1 - 0.5) * 2.0
                y1 = HELIPAD_Y + (rand2 - 0.5) * 2.0

            var dx = x1 - x0
            var dy = y1 - y0
            var length = sqrt(dx * dx + dy * dy)
            var nx = -dy / length
            var ny = dx / length
            if ny < 0:
                nx = -nx
                ny = -ny

            var edge_offset = env_base + EDGES_OFFSET + edge * 6
            states[edge_offset + 0] = Scalar[dtype](x0)
            states[edge_offset + 1] = Scalar[dtype](y0)
            states[edge_offset + 2] = Scalar[dtype](x1)
            states[edge_offset + 3] = Scalar[dtype](y1)
            states[edge_offset + 4] = Scalar[dtype](nx)
            states[edge_offset + 5] = Scalar[dtype](ny)

        # Initialize lander
        rng = rng * 6364136223846793005 + 1442695040888963407
        var init_vx = (Float64(rng >> 33) / Float64(2147483647) - 0.5) * 2.0
        rng = rng * 6364136223846793005 + 1442695040888963407
        var init_vy = (Float64(rng >> 33) / Float64(2147483647) - 0.5) * 2.0

        var lander_base = env_base + BODIES_OFFSET
        states[lander_base + IDX_X] = Scalar[dtype](HELIPAD_X)
        states[lander_base + IDX_Y] = Scalar[dtype](H_UNITS)
        states[lander_base + IDX_ANGLE] = Scalar[dtype](0)
        states[lander_base + IDX_VX] = Scalar[dtype](init_vx)
        states[lander_base + IDX_VY] = Scalar[dtype](init_vy)
        states[lander_base + IDX_OMEGA] = Scalar[dtype](0)
        states[lander_base + IDX_INV_MASS] = Scalar[dtype](1.0 / LANDER_MASS)
        states[lander_base + IDX_INV_INERTIA] = Scalar[dtype](1.0 / LANDER_INERTIA)
        states[lander_base + IDX_SHAPE] = Scalar[dtype](0)

        # Initialize legs
        for leg in range(2):
            var leg_base = env_base + BODIES_OFFSET + (leg + 1) * BODY_STATE_SIZE
            var leg_offset_x = LEG_AWAY if leg == 1 else -LEG_AWAY
            states[leg_base + IDX_X] = Scalar[dtype](HELIPAD_X + leg_offset_x)
            states[leg_base + IDX_Y] = Scalar[dtype](H_UNITS - 10.0 / SCALE - LEG_DOWN)
            states[leg_base + IDX_ANGLE] = Scalar[dtype](0)
            states[leg_base + IDX_VX] = Scalar[dtype](init_vx)
            states[leg_base + IDX_VY] = Scalar[dtype](init_vy)
            states[leg_base + IDX_OMEGA] = Scalar[dtype](0)
            states[leg_base + IDX_INV_MASS] = Scalar[dtype](1.0 / LEG_MASS)
            states[leg_base + IDX_INV_INERTIA] = Scalar[dtype](1.0 / LEG_INERTIA)
            states[leg_base + IDX_SHAPE] = Scalar[dtype](leg + 1)

        # Initialize joints
        states[env_base + JOINT_COUNT_OFFSET] = Scalar[dtype](2)
        for j in range(2):
            var joint_base = env_base + JOINTS_OFFSET + j * JOINT_DATA_SIZE
            var leg_offset_x = LEG_AWAY if j == 1 else -LEG_AWAY
            states[joint_base + JOINT_TYPE] = Scalar[dtype](JOINT_REVOLUTE)
            states[joint_base + JOINT_BODY_A] = Scalar[dtype](0)
            states[joint_base + JOINT_BODY_B] = Scalar[dtype](j + 1)
            states[joint_base + JOINT_ANCHOR_AX] = Scalar[dtype](leg_offset_x)
            states[joint_base + JOINT_ANCHOR_AY] = Scalar[dtype](-10.0 / SCALE)
            states[joint_base + JOINT_ANCHOR_BX] = Scalar[dtype](0)
            states[joint_base + JOINT_ANCHOR_BY] = Scalar[dtype](LEG_H)
            states[joint_base + JOINT_REF_ANGLE] = Scalar[dtype](0)
            states[joint_base + JOINT_LOWER_LIMIT] = Scalar[dtype](-0.9 if j == 1 else 0.4)
            states[joint_base + JOINT_UPPER_LIMIT] = Scalar[dtype](-0.4 if j == 1 else 0.9)
            states[joint_base + JOINT_STIFFNESS] = Scalar[dtype](LEG_SPRING_STIFFNESS)
            states[joint_base + JOINT_DAMPING] = Scalar[dtype](LEG_SPRING_DAMPING)
            states[joint_base + JOINT_FLAGS] = Scalar[dtype](JOINT_FLAG_LIMIT_ENABLED | JOINT_FLAG_SPRING_ENABLED)

        # Clear forces
        for body in range(NUM_BODIES):
            var force_base = env_base + FORCES_OFFSET + body * 3
            states[force_base + 0] = Scalar[dtype](0)
            states[force_base + 1] = Scalar[dtype](0)
            states[force_base + 2] = Scalar[dtype](0)

        # Initialize observation
        var y_norm = (H_UNITS - (HELIPAD_Y + LEG_DOWN / SCALE)) / (H_UNITS / 2.0)
        var vx_norm = init_vx * (W_UNITS / 2.0) / 50.0
        var vy_norm = init_vy * (H_UNITS / 2.0) / 50.0

        var obs_base = env_base + OBS_OFFSET
        states[obs_base + 0] = Scalar[dtype](0)
        states[obs_base + 1] = Scalar[dtype](y_norm)
        states[obs_base + 2] = Scalar[dtype](vx_norm)
        states[obs_base + 3] = Scalar[dtype](vy_norm)
        states[obs_base + 4] = Scalar[dtype](0)
        states[obs_base + 5] = Scalar[dtype](0)
        states[obs_base + 6] = Scalar[dtype](0)
        states[obs_base + 7] = Scalar[dtype](0)

        # Initialize metadata
        var meta_base = env_base + METADATA_OFFSET
        states[meta_base + META_STEP_COUNT] = Scalar[dtype](0)
        states[meta_base + META_TOTAL_REWARD] = Scalar[dtype](0)
        states[meta_base + META_PREV_SHAPING] = Scalar[dtype](0)
        states[meta_base + META_DONE] = Scalar[dtype](0)

    @staticmethod
    fn _init_shapes_gpu(
        ctx: DeviceContext,
        mut shapes_buf: DeviceBuffer[dtype],
    ) raises:
        """Initialize shape definitions (shared across all environments)."""
        var shapes = LayoutTensor[
            dtype, Layout.row_major(NUM_SHAPES * SHAPE_MAX_SIZE), MutAnyOrigin
        ](shapes_buf.unsafe_ptr())

        @always_inline
        fn init_shapes_wrapper(
            shapes: LayoutTensor[
                dtype, Layout.row_major(NUM_SHAPES * SHAPE_MAX_SIZE), MutAnyOrigin
            ],
        ):
            var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
            if tid > 0:
                return

            # Lander shape (6-vertex polygon matching Gymnasium)
            shapes[0] = Scalar[dtype](SHAPE_POLYGON)
            shapes[1] = Scalar[dtype](6)
            shapes[2] = Scalar[dtype](-14.0 / SCALE)
            shapes[3] = Scalar[dtype](17.0 / SCALE)
            shapes[4] = Scalar[dtype](-17.0 / SCALE)
            shapes[5] = Scalar[dtype](0.0)
            shapes[6] = Scalar[dtype](-17.0 / SCALE)
            shapes[7] = Scalar[dtype](-10.0 / SCALE)
            shapes[8] = Scalar[dtype](17.0 / SCALE)
            shapes[9] = Scalar[dtype](-10.0 / SCALE)
            shapes[10] = Scalar[dtype](17.0 / SCALE)
            shapes[11] = Scalar[dtype](0.0)
            shapes[12] = Scalar[dtype](14.0 / SCALE)
            shapes[13] = Scalar[dtype](17.0 / SCALE)

            # Leg shapes (rectangles)
            for leg in range(2):
                var base = (leg + 1) * SHAPE_MAX_SIZE
                shapes[base + 0] = Scalar[dtype](SHAPE_POLYGON)
                shapes[base + 1] = Scalar[dtype](4)
                shapes[base + 2] = Scalar[dtype](-LEG_W)
                shapes[base + 3] = Scalar[dtype](LEG_H)
                shapes[base + 4] = Scalar[dtype](-LEG_W)
                shapes[base + 5] = Scalar[dtype](-LEG_H)
                shapes[base + 6] = Scalar[dtype](LEG_W)
                shapes[base + 7] = Scalar[dtype](-LEG_H)
                shapes[base + 8] = Scalar[dtype](LEG_W)
                shapes[base + 9] = Scalar[dtype](LEG_H)

        ctx.enqueue_function[init_shapes_wrapper, init_shapes_wrapper](
            shapes, grid_dim=(1,), block_dim=(1,),
        )

    @staticmethod
    fn _extract_counts_gpu[
        BATCH_SIZE: Int,
    ](
        ctx: DeviceContext,
        states_buf: DeviceBuffer[dtype],
        mut edge_counts_buf: DeviceBuffer[dtype],
        mut joint_counts_buf: DeviceBuffer[dtype],
    ) raises:
        """Extract edge and joint counts from state."""
        var states = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE * STATE_SIZE_VAL), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var edge_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](edge_counts_buf.unsafe_ptr())
        var joint_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](joint_counts_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn extract_wrapper(
            states: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE * STATE_SIZE_VAL), MutAnyOrigin
            ],
            edge_counts: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
            joint_counts: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            edge_counts[i] = states[i * STATE_SIZE_VAL + EDGE_COUNT_OFFSET]
            joint_counts[i] = states[i * STATE_SIZE_VAL + JOINT_COUNT_OFFSET]

        ctx.enqueue_function[extract_wrapper, extract_wrapper](
            states, edge_counts, joint_counts,
            grid_dim=(BLOCKS,), block_dim=(TPB,),
        )

    @staticmethod
    fn _apply_forces_gpu[
        BATCH_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
    ) raises:
        """Apply engine forces based on actions."""
        var states = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE * STATE_SIZE_VAL), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var actions = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](actions_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn apply_wrapper(
            states: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE * STATE_SIZE_VAL), MutAnyOrigin
            ],
            actions: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return

            var env_base = i * STATE_SIZE_VAL
            var forces_base = env_base + FORCES_OFFSET
            var lander_base = env_base + BODIES_OFFSET

            # Clear forces
            for body in range(NUM_BODIES):
                var fb = forces_base + body * 3
                states[fb + 0] = Scalar[dtype](0)
                states[fb + 1] = Scalar[dtype](0)
                states[fb + 2] = Scalar[dtype](0)

            # Check if done
            if rebind[Scalar[dtype]](states[env_base + METADATA_OFFSET + META_DONE]) > Scalar[dtype](0.5):
                return

            var action = Int(actions[i])
            if action == 0:
                return

            var angle = states[lander_base + IDX_ANGLE]
            var tip_x = sin(angle)
            var tip_y = cos(angle)
            var side_x = -tip_y
            var side_y = tip_x

            if action == 2:  # Main engine
                states[forces_base + 0] = -tip_x * Scalar[dtype](MAIN_ENGINE_POWER)
                states[forces_base + 1] = tip_y * Scalar[dtype](MAIN_ENGINE_POWER)
            elif action == 1:  # Left engine
                states[forces_base + 0] = side_x * Scalar[dtype](SIDE_ENGINE_POWER)
                states[forces_base + 1] = -side_y * Scalar[dtype](SIDE_ENGINE_POWER)
                states[forces_base + 2] = Scalar[dtype](SIDE_ENGINE_POWER * 0.5)
            elif action == 3:  # Right engine
                states[forces_base + 0] = -side_x * Scalar[dtype](SIDE_ENGINE_POWER)
                states[forces_base + 1] = side_y * Scalar[dtype](SIDE_ENGINE_POWER)
                states[forces_base + 2] = Scalar[dtype](-SIDE_ENGINE_POWER * 0.5)

        ctx.enqueue_function[apply_wrapper, apply_wrapper](
            states, actions,
            grid_dim=(BLOCKS,), block_dim=(TPB,),
        )

    @staticmethod
    fn _finalize_step_gpu[
        BATCH_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        contact_counts_buf: DeviceBuffer[dtype],
        mut rewards_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
    ) raises:
        """Update observations and compute rewards."""
        var states = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE * STATE_SIZE_VAL), MutAnyOrigin
        ](states_buf.unsafe_ptr())
        var actions = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var contact_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](contact_counts_buf.unsafe_ptr())
        var rewards = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var dones = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones_buf.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn finalize_wrapper(
            states: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE * STATE_SIZE_VAL), MutAnyOrigin
            ],
            actions: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
            contact_counts: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
            rewards: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
            dones: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return

            var env_base = i * STATE_SIZE_VAL
            var lander_base = env_base + BODIES_OFFSET

            # Check if already done
            if rebind[Scalar[dtype]](states[env_base + METADATA_OFFSET + META_DONE]) > Scalar[dtype](0.5):
                rewards[i] = Scalar[dtype](0)
                dones[i] = Scalar[dtype](1)
                return

            # Get lander state
            var x = Float64(rebind[Scalar[dtype]](states[lander_base + IDX_X]))
            var y = Float64(rebind[Scalar[dtype]](states[lander_base + IDX_Y]))
            var vx = Float64(rebind[Scalar[dtype]](states[lander_base + IDX_VX]))
            var vy = Float64(rebind[Scalar[dtype]](states[lander_base + IDX_VY]))
            var angle = Float64(rebind[Scalar[dtype]](states[lander_base + IDX_ANGLE]))
            var omega = Float64(rebind[Scalar[dtype]](states[lander_base + IDX_OMEGA]))

            # Normalize observation
            var x_norm = (x - HELIPAD_X) / (W_UNITS / 2.0)
            var y_norm = (y - (HELIPAD_Y + LEG_DOWN / SCALE)) / (H_UNITS / 2.0)
            var vx_norm = vx * (W_UNITS / 2.0) / 50.0
            var vy_norm = vy * (H_UNITS / 2.0) / 50.0
            var omega_norm = omega * 20.0 / 50.0

            # Check leg contacts
            var left_contact: Float64 = 0.0
            var right_contact: Float64 = 0.0
            var left_y = Float64(rebind[Scalar[dtype]](states[env_base + BODIES_OFFSET + BODY_STATE_SIZE + IDX_Y]))
            var right_y = Float64(rebind[Scalar[dtype]](states[env_base + BODIES_OFFSET + 2 * BODY_STATE_SIZE + IDX_Y]))
            if left_y <= HELIPAD_Y + 0.1:
                left_contact = 1.0
            if right_y <= HELIPAD_Y + 0.1:
                right_contact = 1.0

            # Update observation
            var obs_base = env_base + OBS_OFFSET
            states[obs_base + 0] = Scalar[dtype](x_norm)
            states[obs_base + 1] = Scalar[dtype](y_norm)
            states[obs_base + 2] = Scalar[dtype](vx_norm)
            states[obs_base + 3] = Scalar[dtype](vy_norm)
            states[obs_base + 4] = Scalar[dtype](angle)
            states[obs_base + 5] = Scalar[dtype](omega_norm)
            states[obs_base + 6] = Scalar[dtype](left_contact)
            states[obs_base + 7] = Scalar[dtype](right_contact)

            # Compute shaping
            var dist = sqrt(x_norm * x_norm + y_norm * y_norm)
            var speed = sqrt(vx_norm * vx_norm + vy_norm * vy_norm)
            var abs_angle = angle if angle >= 0.0 else -angle
            var shaping = (
                -100.0 * dist
                - 100.0 * speed
                - 100.0 * abs_angle
                + 10.0 * left_contact
                + 10.0 * right_contact
            )

            var prev_shaping = Float64(rebind[Scalar[dtype]](states[env_base + METADATA_OFFSET + META_PREV_SHAPING]))
            var reward = shaping - prev_shaping
            states[env_base + METADATA_OFFSET + META_PREV_SHAPING] = Scalar[dtype](shaping)

            # Fuel costs
            var action = Int(actions[i])
            if action == 2:
                reward = reward - MAIN_ENGINE_FUEL_COST
            elif action == 1 or action == 3:
                reward = reward - SIDE_ENGINE_FUEL_COST

            # Check termination
            var done: Float64 = 0.0

            # Out of bounds
            if x_norm >= 1.0 or x_norm <= -1.0:
                done = 1.0
                reward = CRASH_PENALTY

            # Below ground or too high
            if y < 0.0 or y > H_UNITS * 1.5:
                done = 1.0
                reward = CRASH_PENALTY

            # Successful landing
            var both_legs = left_contact > 0.5 and right_contact > 0.5
            var speed_val = sqrt(vx * vx + vy * vy)
            var abs_omega = omega if omega >= 0.0 else -omega
            if both_legs and speed_val < 0.1 and abs_omega < 0.1:
                done = 1.0
                reward = reward + LAND_REWARD

            # Max steps
            var step_count = Float64(rebind[Scalar[dtype]](states[env_base + METADATA_OFFSET + META_STEP_COUNT]))
            if step_count > 1000.0:
                done = 1.0

            # Update metadata
            states[env_base + METADATA_OFFSET + META_STEP_COUNT] = Scalar[dtype](step_count + 1.0)
            states[env_base + METADATA_OFFSET + META_DONE] = Scalar[dtype](done)
            rewards[i] = Scalar[dtype](reward)
            dones[i] = Scalar[dtype](done)

        ctx.enqueue_function[finalize_wrapper, finalize_wrapper](
            states, actions, contact_counts, rewards, dones,
            grid_dim=(BLOCKS,), block_dim=(TPB,),
        )

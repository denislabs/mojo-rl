"""GPU CartPole Environment - Vectorized for Multiple Environments.

Implements a fully vectorized CartPole environment where a single kernel
call simulates NUM_ENVS environments in parallel.

State layout: [NUM_ENVS, STATE_SIZE] where STATE_SIZE = 4
  - state[env_idx, 0] = x (cart position)
  - state[env_idx, 1] = x_dot (cart velocity)
  - state[env_idx, 2] = theta (pole angle, radians, 0 = upright)
  - state[env_idx, 3] = theta_dot (pole angular velocity)

Actions layout: [NUM_ENVS, 1]
  - actions[env_idx, 0] = 0 (left) or 1 (right)

Step output layout: [NUM_ENVS, 6] = [state (4), reward (1), done (1)]

Physics matches Gymnasium CartPole-v1.
"""

from gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor
from math import cos, sin
from deep_rl.gpu import xorshift32, random_uniform, random_range

# =============================================================================
# Environment Dimensions (compile-time constants)
# =============================================================================

comptime OBS_DIM: Int = 4
comptime NUM_ACTIONS: Int = 2
comptime STATE_SIZE: Int = 4
comptime NUM_ENVS: Int = 1024
comptime THREADS_PER_BLOCK: Int = 256

# Compute grid dimensions
comptime BLOCKS_PER_GRID: Int = (
    NUM_ENVS + THREADS_PER_BLOCK - 1
) // THREADS_PER_BLOCK

comptime dtype = DType.float32

# Layout definitions
comptime state_layout = Layout.row_major(NUM_ENVS, STATE_SIZE)
comptime action_layout = Layout.row_major(NUM_ENVS, 1)
comptime rng_layout = Layout.row_major(NUM_ENVS, 1)
comptime reward_layout = Layout.row_major(NUM_ENVS, 1)
comptime done_layout = Layout.row_major(NUM_ENVS, 1)

# =============================================================================
# Physics Constants
# =============================================================================

comptime GRAVITY: Float32 = 9.8
comptime CART_MASS: Float32 = 1.0
comptime POLE_MASS: Float32 = 0.1
comptime TOTAL_MASS: Float32 = CART_MASS + POLE_MASS
comptime POLE_HALF_LENGTH: Float32 = 0.5
comptime POLE_MASS_LENGTH: Float32 = POLE_MASS * POLE_HALF_LENGTH
comptime FORCE_MAG: Float32 = 10.0
comptime TAU: Float32 = 0.02  # Time step

# Termination thresholds
comptime X_THRESHOLD: Float32 = 2.4
comptime THETA_THRESHOLD: Float32 = 0.2095  # ~12 degrees

# Initial state randomization range
comptime INIT_RANGE: Float32 = 0.05


# # =============================================================================
# # GPU Random Number Generator
# # =============================================================================


# @always_inline
# fn xorshift32(state: Scalar[DType.uint32]) -> Scalar[DType.uint32]:
#     """Simple xorshift PRNG - fast and GPU-friendly."""
#     var x = state
#     x ^= x << 13
#     x ^= x >> 17
#     x ^= x << 5
#     return x


# @always_inline
# fn random_uniform(
#     rng: Scalar[DType.uint32],
# ) -> Tuple[Scalar[dtype], Scalar[DType.uint32]]:
#     """Generate uniform random number in [0, 1) and return (value, new_rng)."""
#     new_rng = xorshift32(rng)
#     value = Scalar[dtype](new_rng) / Scalar[dtype](Scalar[DType.uint32].MAX)
#     return (value, new_rng)


# @always_inline
# fn random_range(
#     rng: Scalar[DType.uint32], low: Scalar[dtype], high: Scalar[dtype]
# ) -> Tuple[Scalar[dtype], Scalar[DType.uint32]]:
#     """Generate uniform random number in [low, high) and return (value, new_rng).
#     """
#     result = random_uniform(rng)
#     value = low + result[0] * (high - low)
#     return (value, result[1])


# =============================================================================
# GPU Kernels
# =============================================================================


fn step_kernel(
    states: LayoutTensor[dtype, state_layout, MutAnyOrigin],
    actions: LayoutTensor[DType.int32, action_layout, MutAnyOrigin],
    rewards: LayoutTensor[dtype, reward_layout, MutAnyOrigin],
    dones: LayoutTensor[DType.int32, done_layout, MutAnyOrigin],
):
    """Vectorized step kernel - each thread processes one environment.

    Args:
        states: [NUM_ENVS, STATE_SIZE] - states are updated in-place
        actions: [NUM_ENVS, 1] - action per environment (0=left, 1=right)
        rewards: [NUM_ENVS, 1] - output rewards
        dones: [NUM_ENVS, 1] - output done flags (1=done, 0=not done)
    """
    # Compute which environment this thread handles
    var env_idx = Int(block_dim.x * block_idx.x + thread_idx.x)

    # Bounds check
    if env_idx >= NUM_ENVS:
        return

    # Read current state for this environment (rebind for type safety)
    var x = rebind[Scalar[dtype]](states[env_idx, 0])
    var x_dot = rebind[Scalar[dtype]](states[env_idx, 1])
    var theta = rebind[Scalar[dtype]](states[env_idx, 2])
    var theta_dot = rebind[Scalar[dtype]](states[env_idx, 3])

    # Read action for this environment
    var action = Int(actions[env_idx, 0])

    # Compute force based on action
    var force = FORCE_MAG if action == 1 else -FORCE_MAG

    # Physics calculations (Euler integration matching Gymnasium)
    var cos_theta = cos(theta)
    var sin_theta = sin(theta)

    var temp = (
        force + POLE_MASS_LENGTH * theta_dot * theta_dot * sin_theta
    ) / TOTAL_MASS

    var theta_acc = (GRAVITY * sin_theta - cos_theta * temp) / (
        POLE_HALF_LENGTH
        * (
            Scalar[dtype](4.0 / 3.0)
            - POLE_MASS * cos_theta * cos_theta / TOTAL_MASS
        )
    )

    var x_acc = temp - POLE_MASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS

    # Euler integration
    var new_x = x + TAU * x_dot
    var new_x_dot = x_dot + TAU * x_acc
    var new_theta = theta + TAU * theta_dot
    var new_theta_dot = theta_dot + TAU * theta_acc

    # Write updated state back
    states[env_idx, 0] = new_x
    states[env_idx, 1] = new_x_dot
    states[env_idx, 2] = new_theta
    states[env_idx, 3] = new_theta_dot

    # Check termination conditions
    var done = (
        (new_x < -X_THRESHOLD)
        or (new_x > X_THRESHOLD)
        or (new_theta < -THETA_THRESHOLD)
        or (new_theta > THETA_THRESHOLD)
    )

    # Reward: +1 for staying alive, 0 if done
    var reward = Scalar[dtype](0.0) if done else Scalar[dtype](1.0)

    # Write outputs
    rewards[env_idx, 0] = reward
    dones[env_idx, 0] = 1 if done else 0


fn reset_kernel(
    states: LayoutTensor[dtype, state_layout, MutAnyOrigin],
    rng_states: LayoutTensor[DType.uint32, rng_layout, MutAnyOrigin],
):
    """Vectorized reset kernel - each thread resets one environment.

    Args:
        states: [NUM_ENVS, STATE_SIZE] - states to reset
        rng_states: [NUM_ENVS, 1] - per-environment RNG states (updated in-place)
    """
    # Compute which environment this thread handles
    var env_idx = Int(block_dim.x * block_idx.x + thread_idx.x)

    # Bounds check
    if env_idx >= NUM_ENVS:
        return

    # Get RNG state for this environment (rebind to scalar)
    var rng = rebind[Scalar[DType.uint32]](rng_states[env_idx, 0])

    # Generate random initial state values in [-INIT_RANGE, INIT_RANGE]
    var low = Scalar[dtype](-INIT_RANGE)
    var high = Scalar[dtype](INIT_RANGE)

    var r0 = random_range(rng, low, high)
    states[env_idx, 0] = r0[0]  # x
    rng = r0[1]

    var r1 = random_range(rng, low, high)
    states[env_idx, 1] = r1[0]  # x_dot
    rng = r1[1]

    var r2 = random_range(rng, low, high)
    states[env_idx, 2] = r2[0]  # theta
    rng = r2[1]

    var r3 = random_range(rng, low, high)
    states[env_idx, 3] = r3[0]  # theta_dot
    rng = r3[1]

    # Update RNG state
    rng_states[env_idx, 0] = rng


fn reset_where_done_kernel(
    states: LayoutTensor[dtype, state_layout, MutAnyOrigin],
    dones: LayoutTensor[DType.int32, done_layout, MutAnyOrigin],
    rng_states: LayoutTensor[DType.uint32, rng_layout, MutAnyOrigin],
):
    """Vectorized conditional reset - only reset environments where done=1.

    Args:
        states: [NUM_ENVS, STATE_SIZE] - states to conditionally reset
        dones: [NUM_ENVS, 1] - done flags (1=reset this env, 0=skip)
        rng_states: [NUM_ENVS, 1] - per-environment RNG states
    """
    # Compute which environment this thread handles
    var env_idx = Int(block_dim.x * block_idx.x + thread_idx.x)

    # Bounds check
    if env_idx >= NUM_ENVS:
        return

    # Only reset if this environment is done
    if dones[env_idx, 0] == 0:
        return

    # Get RNG state for this environment (rebind to scalar)
    var rng = rebind[Scalar[DType.uint32]](rng_states[env_idx, 0])

    # Generate random initial state values
    var low = Scalar[dtype](-INIT_RANGE)
    var high = Scalar[dtype](INIT_RANGE)

    var r0 = random_range(rng, low, high)
    states[env_idx, 0] = r0[0]
    rng = r0[1]

    var r1 = random_range(rng, low, high)
    states[env_idx, 1] = r1[0]
    rng = r1[1]

    var r2 = random_range(rng, low, high)
    states[env_idx, 2] = r2[0]
    rng = r2[1]

    var r3 = random_range(rng, low, high)
    states[env_idx, 3] = r3[0]
    rng = r3[1]

    # Update RNG state
    rng_states[env_idx, 0] = rng

    # Clear done flag after reset
    dones[env_idx, 0] = 0


# =============================================================================
# GPUCartPole Struct - Host-side interface
# =============================================================================


trait GPUDiscreteEnv:
    """Trait for GPU-compatible discrete action environments.

    Environments must define compile-time constants and inline methods
    for use in fused GPU kernels.
    """
    # Compile-time constants for environment dimensions
    comptime STATE_SIZE: Int
    comptime OBS_DIM: Int
    comptime NUM_ACTIONS: Int

    @staticmethod
    fn step_inline[size: Int](
        mut state: InlineArray[Scalar[dtype], size],
        action: Int,
    ) -> Tuple[Scalar[dtype], Bool]:
        """Perform one environment step. Returns (reward, done)."""
        ...

    @staticmethod
    fn reset_inline[size: Int](
        mut state: InlineArray[Scalar[dtype], size],
        mut rng: Scalar[DType.uint32],
    ):
        """Reset state to random initial values."""
        ...


struct GPUCartPole(GPUDiscreteEnv):
    """GPU-compatible vectorized CartPole environment.

    Manages NUM_ENVS environments in parallel. All operations are batched:
    - step() advances all environments by one timestep
    - reset() initializes all environments
    - reset_where_done() selectively resets terminated environments
    """

    # Implement GPUDiscreteEnv trait constants
    comptime STATE_SIZE: Int = 4  # [x, x_dot, theta, theta_dot]
    comptime OBS_DIM: Int = 4     # Same as state for CartPole
    comptime NUM_ACTIONS: Int = 2  # Left (0) or Right (1)

    # Usage example - see A2CAgent.train[GPUCartPole]() for full integration

    @staticmethod
    fn get_grid_dim() -> Int:
        """Return the number of blocks needed for NUM_ENVS."""
        return BLOCKS_PER_GRID

    @staticmethod
    fn get_block_dim() -> Int:
        """Return threads per block."""
        return THREADS_PER_BLOCK

    @staticmethod
    fn get_num_envs() -> Int:
        """Return the number of parallel environments."""
        return NUM_ENVS

    @staticmethod
    fn get_state_size() -> Int:
        """Return the state dimension (4 for CartPole)."""
        return STATE_SIZE

    @staticmethod
    fn get_num_actions() -> Int:
        """Return the number of discrete actions (2 for CartPole)."""
        return NUM_ACTIONS

    # =========================================================================
    # Inline methods for fused kernels (composability with A2C, PPO, etc.)
    # =========================================================================

    @staticmethod
    @always_inline
    fn step_inline[size: Int](
        mut state: InlineArray[Scalar[dtype], size],
        action: Int,
    ) -> Tuple[Scalar[dtype], Bool]:
        """Inline step for use in fused kernels. Returns (reward, done).

        State layout: [x, x_dot, theta, theta_dot]
        - x: cart position
        - x_dot: cart velocity
        - theta: pole angle
        - theta_dot: pole angular velocity
        """
        # Compute force based on action
        var force = FORCE_MAG if action == 1 else -FORCE_MAG

        # Physics calculations (Euler integration matching Gymnasium)
        var cos_theta = cos(state[2])
        var sin_theta = sin(state[2])

        var temp = (force + POLE_MASS_LENGTH * state[3] * state[3] * sin_theta) / TOTAL_MASS

        var theta_acc = (GRAVITY * sin_theta - cos_theta * temp) / (
            POLE_HALF_LENGTH
            * (
                Scalar[dtype](4.0 / 3.0)
                - POLE_MASS * cos_theta * cos_theta / TOTAL_MASS
            )
        )

        var x_acc = temp - POLE_MASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS

        # Euler integration - update state in-place
        state[0] = state[0] + TAU * state[1]
        state[1] = state[1] + TAU * x_acc
        state[2] = state[2] + TAU * state[3]
        state[3] = state[3] + TAU * theta_acc

        # Check termination conditions
        var done = (
            (state[0] < -X_THRESHOLD)
            or (state[0] > X_THRESHOLD)
            or (state[2] < -THETA_THRESHOLD)
            or (state[2] > THETA_THRESHOLD)
        )

        # Reward: +1 for staying alive, 0 if done
        var reward = Scalar[dtype](0.0) if done else Scalar[dtype](1.0)

        return (reward, done)

    @staticmethod
    @always_inline
    fn reset_inline[size: Int](
        mut state: InlineArray[Scalar[dtype], size],
        mut rng: Scalar[DType.uint32],
    ):
        """Inline reset for use in fused kernels.

        Resets state to random initial values and updates RNG state.
        """
        var low = Scalar[dtype](-INIT_RANGE)
        var high = Scalar[dtype](INIT_RANGE)

        var r0 = random_range(rng, low, high)
        state[0] = r0[0]
        rng = r0[1]

        var r1 = random_range(rng, low, high)
        state[1] = r1[0]
        rng = r1[1]

        var r2 = random_range(rng, low, high)
        state[2] = r2[0]
        rng = r2[1]

        var r3 = random_range(rng, low, high)
        state[3] = r3[0]
        rng = r3[1]

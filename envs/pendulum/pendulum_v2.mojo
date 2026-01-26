"""Pendulum V2 GPU environment.

Native Mojo implementation of Pendulum with GPU-accelerated batched simulation.

Physics matched to Gymnasium Pendulum-v1:
https://gymnasium.farama.org/environments/classic_control/pendulum/

A frictionless pendulum starts from a random position and the goal is to
swing it up and keep it balanced upright.

State observation: [cos(θ), sin(θ), θ_dot] (3D)
Action: torque in [-2.0, 2.0] (1D continuous)
Reward: -(θ² + 0.1*θ_dot² + 0.001*torque²)

Episode never terminates naturally (always runs for max_steps=200).
"""

from math import sqrt, cos, sin, pi
from layout import Layout, LayoutTensor
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer
from random import random_float64
from random.philox import Random as PhiloxRandom

from core import (
    BoxContinuousActionEnv,
    GPUContinuousEnv,
    BoxDiscreteActionEnv,
    DiscreteEnv,
    TileCoding,
    PolynomialFeatures,
)
from render import (
    RendererBase,
    Camera,
    SDL_Color,
    Vec2,
    sky_blue,
    black,
    light_gray,
    rgb,
)

from .state import PendulumV2State
from .action import PendulumV2Action
from .constants import PConstants, PendulumLayout

# Import global GPU constants
from physics2d import dtype, TPB


# =============================================================================
# PendulumV2 Environment
# =============================================================================


struct PendulumV2[DTYPE: DType](
    BoxContinuousActionEnv,
    BoxDiscreteActionEnv,
    Copyable,
    DiscreteEnv,
    GPUContinuousEnv,
    Movable,
):
    """Pendulum environment with GPU-accelerated batched simulation.

    This environment implements both CPU single-env and GPU batched interfaces.

    Features:
    - Simple pendulum physics (angle + angular velocity)
    - GPU-compatible state layout for batch training
    - Continuous action space: torque in [-2, 2]
    - Also supports discrete actions: 0 (left), 1 (none), 2 (right)

    Physics:
    - θ'' = (3g/2L) * sin(θ) + (3/mL²) * u
    - Euler integration with dt = 0.05

    Reward:
    - r = -(θ² + 0.1*θ_dot² + 0.001*u²)
    - Where θ is normalized to [-π, π]
    """

    # =========================================================================
    # Type Aliases and Constants
    # =========================================================================

    comptime dtype = Self.DTYPE
    comptime StateType = PendulumV2State[Self.DTYPE]
    comptime ActionType = PendulumV2Action[Self.DTYPE]

    # GPUContinuousEnv trait requirements
    comptime STATE_SIZE: Int = PConstants.STATE_SIZE  # 6
    comptime OBS_DIM: Int = PConstants.OBS_DIM  # 3
    comptime ACTION_DIM: Int = PConstants.ACTION_DIM  # 1

    # DiscreteEnv trait requirement
    comptime NUM_ACTIONS: Int = 3  # left, none, right

    # =========================================================================
    # Instance Variables (for CPU single-env mode)
    # =========================================================================

    # Physics constants
    var max_speed: Scalar[Self.dtype]
    var max_torque: Scalar[Self.dtype]
    var dt: Scalar[Self.dtype]
    var g: Scalar[Self.dtype]
    var m: Scalar[Self.dtype]
    var l: Scalar[Self.dtype]

    # Physics state
    var theta: Scalar[Self.dtype]
    var theta_dot: Scalar[Self.dtype]

    # Episode tracking
    var steps: Int
    var max_steps: Int
    var done: Bool
    var total_reward: Scalar[Self.dtype]
    var last_torque: Scalar[Self.dtype]

    # Discretization settings (for DiscreteEnv)
    var num_bins_angle: Int
    var num_bins_velocity: Int

    # =========================================================================
    # Constructors
    # =========================================================================

    fn __init__(
        out self, num_bins_angle: Int = 15, num_bins_velocity: Int = 15
    ):
        """Initialize Pendulum with default physics parameters.

        Args:
            num_bins_angle: Number of bins for angle discretization.
            num_bins_velocity: Number of bins for velocity discretization.
        """
        # Physics constants from Gymnasium
        self.max_speed = Scalar[Self.dtype](PConstants.MAX_SPEED)
        self.max_torque = Scalar[Self.dtype](PConstants.MAX_TORQUE)
        self.dt = Scalar[Self.dtype](PConstants.DT)
        self.g = Scalar[Self.dtype](PConstants.G)
        self.m = Scalar[Self.dtype](PConstants.M)
        self.l = Scalar[Self.dtype](PConstants.L)

        # State (θ=0 is pointing up, positive is clockwise)
        self.theta = Scalar[Self.dtype](pi)  # Start pointing down
        self.theta_dot = Scalar[Self.dtype](0.0)

        # Episode
        self.steps = 0
        self.max_steps = PConstants.MAX_STEPS
        self.done = False
        self.total_reward = Scalar[Self.dtype](0.0)
        self.last_torque = Scalar[Self.dtype](0.0)

        # Discretization settings
        self.num_bins_angle = num_bins_angle
        self.num_bins_velocity = num_bins_velocity

    fn __copyinit__(out self, existing: Self):
        """Copy constructor."""
        self.max_speed = existing.max_speed
        self.max_torque = existing.max_torque
        self.dt = existing.dt
        self.g = existing.g
        self.m = existing.m
        self.l = existing.l
        self.theta = existing.theta
        self.theta_dot = existing.theta_dot
        self.steps = existing.steps
        self.max_steps = existing.max_steps
        self.done = existing.done
        self.total_reward = existing.total_reward
        self.last_torque = existing.last_torque
        self.num_bins_angle = existing.num_bins_angle
        self.num_bins_velocity = existing.num_bins_velocity

    fn __moveinit__(out self, deinit existing: Self):
        """Move constructor."""
        self.max_speed = existing.max_speed
        self.max_torque = existing.max_torque
        self.dt = existing.dt
        self.g = existing.g
        self.m = existing.m
        self.l = existing.l
        self.theta = existing.theta
        self.theta_dot = existing.theta_dot
        self.steps = existing.steps
        self.max_steps = existing.max_steps
        self.done = existing.done
        self.total_reward = existing.total_reward
        self.last_torque = existing.last_torque
        self.num_bins_angle = existing.num_bins_angle
        self.num_bins_velocity = existing.num_bins_velocity

    # =========================================================================
    # GPU Batch Operations (Static Methods) - GPUContinuousEnv Trait
    # =========================================================================

    @staticmethod
    fn step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
        ACTION_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[dtype],
        actions: DeviceBuffer[dtype],
        mut rewards: DeviceBuffer[dtype],
        mut dones: DeviceBuffer[dtype],
        mut obs: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """Perform one environment step with continuous actions (GPUContinuousEnv trait).

        Pendulum physics:
        - θ'' = (3g/2L) * sin(θ) + (3/mL²) * u
        - Euler integration
        - Reward = -(θ² + 0.1*θ_dot² + 0.001*u²)

        Args:
            ctx: GPU device context.
            states: State buffer [BATCH_SIZE * STATE_SIZE].
            actions: Continuous actions buffer [BATCH_SIZE * ACTION_DIM].
            rewards: Rewards buffer (output) [BATCH_SIZE].
            dones: Done flags buffer (output) [BATCH_SIZE].
            obs: Observations buffer (output) [BATCH_SIZE * OBS_DIM].
            rng_seed: Optional random seed (unused for deterministic physics).
        """
        # Create tensor views
        var states_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states.unsafe_ptr())

        var actions_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ](actions.unsafe_ptr())

        var rewards_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](rewards.unsafe_ptr())

        var dones_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones.unsafe_ptr())

        var obs_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ](obs.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn step_wrapper(
            states: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
            ],
            actions: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
            ],
            rewards: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            dones: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            obs: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return

            PendulumV2[Self.dtype]._step_env_gpu[
                BATCH_SIZE, STATE_SIZE, OBS_DIM, ACTION_DIM
            ](states, actions, rewards, dones, obs, env)

        ctx.enqueue_function[step_wrapper, step_wrapper](
            states_tensor,
            actions_tensor,
            rewards_tensor,
            dones_tensor,
            obs_tensor,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """Reset all environments to random initial values (GPUContinuousEnv trait).

        Initial angle is uniformly random in [-π, π].
        Initial angular velocity is uniformly random in [-1, 1].

        Args:
            ctx: GPU device context.
            states: State buffer [BATCH_SIZE * STATE_SIZE].
            rng_seed: Random seed for initial state generation. Use different
                     values across calls for varied initial states.
        """
        var states_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn reset_wrapper(
            states: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
            ],
            seed: Scalar[dtype],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            # Combine seed with env index using prime multiplier for good distribution
            var combined_seed = Int(seed) * 2654435761 + env * 12345
            PendulumV2[Self.dtype]._reset_env_gpu[BATCH_SIZE, STATE_SIZE](
                states, env, combined_seed
            )

        ctx.enqueue_function[reset_wrapper, reset_wrapper](
            states_tensor,
            Scalar[dtype](rng_seed),
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn selective_reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[dtype],
        mut dones: DeviceBuffer[dtype],
        rng_seed: UInt64,
    ) raises:
        """Reset only done environments (GPUContinuousEnv trait).

        Args:
            ctx: GPU device context.
            states: State buffer [BATCH_SIZE * STATE_SIZE].
            dones: Done flags buffer [BATCH_SIZE].
            rng_seed: Random seed for initialization. Should be different each call
                     (e.g., training step counter) for varied initial states.
        """
        var states_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ](states.unsafe_ptr())

        var dones_tensor = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](dones.unsafe_ptr())

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        fn selective_reset_wrapper(
            states: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
            ],
            dones: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            seed: Scalar[dtype],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            # Only reset if done
            if rebind[Scalar[dtype]](dones[env]) > Scalar[dtype](0.5):
                # Combine seed with env index using prime multiplier for good distribution
                var combined_seed = Int(seed) * 2654435761 + env * 12345
                PendulumV2[Self.dtype]._reset_env_gpu[BATCH_SIZE, STATE_SIZE](
                    states, env, combined_seed
                )
                dones[env] = Scalar[dtype](0.0)

        ctx.enqueue_function[selective_reset_wrapper, selective_reset_wrapper](
            states_tensor,
            dones_tensor,
            Scalar[dtype](rng_seed),
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # GPU Helper Methods (Static, Inline)
    # =========================================================================

    @always_inline
    @staticmethod
    fn _step_env_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
        ACTION_DIM: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        actions: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), MutAnyOrigin
        ],
        rewards: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        dones: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
        obs: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ],
        env: Int,
    ):
        """Step a single environment (GPU-compatible inline function).

        Pendulum physics with Euler integration.

        State layout (8 floats):
            [0] cos(theta)      - observation
            [1] sin(theta)      - observation
            [2] theta_dot       - observation
            [3] theta           - raw angle for physics
            [4] step_count      - metadata
            [5] done            - metadata
            [6] total_reward    - metadata
            [7] last_torque     - metadata
        """
        # Physics constants (cast Float64 to dtype)
        var MAX_SPEED = Scalar[dtype](PConstants.MAX_SPEED)
        var MAX_TORQUE = Scalar[dtype](PConstants.MAX_TORQUE)
        var DT = Scalar[dtype](PConstants.DT)
        var G = Scalar[dtype](PConstants.G)
        var M = Scalar[dtype](PConstants.M)
        var L = Scalar[dtype](PConstants.L)
        var MAX_STEPS_VAL = Scalar[dtype](PConstants.MAX_STEPS)

        # Layout offsets - NEW LAYOUT with obs at offset 0
        comptime OBS_COS = PendulumLayout.OBS_COS_THETA  # 0
        comptime OBS_SIN = PendulumLayout.OBS_SIN_THETA  # 1
        comptime OBS_THETA_DOT = PendulumLayout.OBS_THETA_DOT  # 2
        comptime THETA_ABS = PendulumLayout.THETA_ABS  # 3
        comptime META_OFF = PendulumLayout.METADATA_OFFSET  # 4
        comptime META_STEP = PendulumLayout.META_STEP_COUNT
        comptime META_DONE = PendulumLayout.META_DONE
        comptime META_TOTAL_REWARD = PendulumLayout.META_TOTAL_REWARD
        comptime META_LAST_TORQUE = PendulumLayout.META_LAST_TORQUE

        # Read current state - theta from offset 3, theta_dot from obs offset 2
        var theta = rebind[Scalar[dtype]](states[env, THETA_ABS])
        var theta_dot = rebind[Scalar[dtype]](states[env, OBS_THETA_DOT])
        var step_count = rebind[Scalar[dtype]](
            states[env, META_OFF + META_STEP]
        )
        var total_reward = rebind[Scalar[dtype]](
            states[env, META_OFF + META_TOTAL_REWARD]
        )

        # Get action and scale from [-1, 1] to [-MAX_TORQUE, MAX_TORQUE]
        # PPO continuous outputs actions in [-1, 1] after tanh squashing
        var raw_action = rebind[Scalar[dtype]](actions[env, 0])
        var u = raw_action * MAX_TORQUE
        # Clamp just in case action is slightly out of bounds
        if u > MAX_TORQUE:
            u = MAX_TORQUE
        elif u < -MAX_TORQUE:
            u = -MAX_TORQUE

        # Physics: θ'' = (3g/2L) * sin(θ) + (3/mL²) * u
        # Use sin directly on dtype to avoid Float64 on GPU
        var sin_theta = sin(theta)
        var theta_acc = (
            Scalar[dtype](3.0) * G / (Scalar[dtype](2.0) * L)
        ) * sin_theta + (Scalar[dtype](3.0) / (M * L * L)) * u

        # Euler integration
        theta_dot = theta_dot + theta_acc * DT
        theta = theta + theta_dot * DT

        # Clip angular velocity
        if theta_dot > MAX_SPEED:
            theta_dot = MAX_SPEED
        elif theta_dot < -MAX_SPEED:
            theta_dot = -MAX_SPEED

        # Normalize angle to [-π, π] using dtype-native pi
        var PI = Scalar[dtype](3.14159265358979323846)
        var TWO_PI = PI * Scalar[dtype](2.0)
        while theta > PI:
            theta = theta - TWO_PI
        while theta < -PI:
            theta = theta + TWO_PI

        # Increment step
        step_count = step_count + Scalar[dtype](1.0)

        # Compute reward: -(θ² + 0.1*θ_dot² + 0.001*u²)
        var reward = -(
            theta * theta
            + Scalar[dtype](0.1) * theta_dot * theta_dot
            + Scalar[dtype](0.001) * u * u
        )
        total_reward = total_reward + reward

        # Check if done (pendulum never terminates early, only truncates)
        var is_done = Scalar[dtype](0.0)
        if step_count >= MAX_STEPS_VAL:
            is_done = Scalar[dtype](1.0)

        # Compute observation values
        var cos_theta = cos(theta)
        var sin_theta_val = sin(theta)

        # Write updated state (obs at offset 0, theta at offset 3, metadata at offset 4)
        states[env, OBS_COS] = cos_theta
        states[env, OBS_SIN] = sin_theta_val
        states[env, OBS_THETA_DOT] = theta_dot
        states[env, THETA_ABS] = theta
        states[env, META_OFF + META_STEP] = step_count
        states[env, META_OFF + META_DONE] = is_done
        states[env, META_OFF + META_TOTAL_REWARD] = total_reward
        states[env, META_OFF + META_LAST_TORQUE] = u

        # Write outputs
        rewards[env] = reward
        dones[env] = is_done

        # Write observation to separate obs buffer (for agent)
        obs[env, 0] = cos_theta
        obs[env, 1] = sin_theta_val
        obs[env, 2] = theta_dot

    @always_inline
    @staticmethod
    fn _reset_env_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        seed: Int,
    ):
        """Reset a single environment to random initial state (GPU-compatible).

        Initial angle: uniform random in [-π, π]
        Initial velocity: uniform random in [-1, 1]

        State layout (8 floats):
            [0] cos(theta)      - observation
            [1] sin(theta)      - observation
            [2] theta_dot       - observation
            [3] theta           - raw angle for physics
            [4] step_count      - metadata
            [5] done            - metadata
            [6] total_reward    - metadata
            [7] last_torque     - metadata
        """
        # Layout offsets from new layout
        comptime OBS_COS = PendulumLayout.OBS_COS_THETA  # 0
        comptime OBS_SIN = PendulumLayout.OBS_SIN_THETA  # 1
        comptime OBS_THETA_DOT = PendulumLayout.OBS_THETA_DOT  # 2
        comptime THETA_ABS = PendulumLayout.THETA_ABS  # 3
        comptime META_OFF = PendulumLayout.METADATA_OFFSET  # 4

        # Generate random initial state using Philox RNG
        var rng = PhiloxRandom(seed=seed, offset=0)
        var rand_vals = rng.step_uniform()

        # Random angle in [-π, π] using dtype-native pi to avoid Float64
        var PI = Scalar[dtype](3.14159265358979323846)
        var theta = (
            rand_vals[0] * Scalar[dtype](2.0) - Scalar[dtype](1.0)
        ) * PI

        # Random angular velocity in [-1, 1]
        var theta_dot = rand_vals[1] * Scalar[dtype](2.0) - Scalar[dtype](1.0)

        # Clear entire state
        for i in range(STATE_SIZE):
            states[env, i] = Scalar[dtype](0.0)

        # Write observation at offset 0 (CRITICAL for GPU training!)
        states[env, OBS_COS] = cos(theta)
        states[env, OBS_SIN] = sin(theta)
        states[env, OBS_THETA_DOT] = theta_dot

        # Write raw theta for physics updates
        states[env, THETA_ABS] = theta

        # Metadata is already zeroed

    # =========================================================================
    # CPU Single-Environment Methods - BoxContinuousActionEnv Trait
    # =========================================================================

    fn reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        """Reset environment and return initial observation as list."""
        # Random initial angle in [-π, π]
        self.theta = Scalar[Self.dtype]((random_float64() * 2.0 - 1.0) * pi)
        # Random initial angular velocity in [-1, 1]
        self.theta_dot = Scalar[Self.dtype](random_float64() * 2.0 - 1.0)

        self.steps = 0
        self.done = False
        self.total_reward = Scalar[Self.dtype](0.0)
        self.last_torque = Scalar[Self.dtype](0.0)

        return self.get_obs_list()

    fn get_obs_list(self) -> List[Scalar[Self.dtype]]:
        """Return current continuous observation as list."""
        var obs = List[Scalar[Self.dtype]](capacity=3)
        obs.append(Scalar[Self.dtype](cos(Float64(self.theta))))
        obs.append(Scalar[Self.dtype](sin(Float64(self.theta))))
        obs.append(self.theta_dot)
        return obs^

    fn step_continuous(
        mut self, action: Scalar[Self.dtype]
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Take 1D continuous action (torque) and return (obs, reward, done)."""
        var result = self._step_with_torque(action)
        return (self.get_obs_list(), result[1], result[2])

    fn step_continuous_vec[
        DTYPE_VEC: DType
    ](mut self, action: List[Scalar[DTYPE_VEC]]) -> Tuple[
        List[Scalar[DTYPE_VEC]], Scalar[DTYPE_VEC], Bool
    ]:
        """Take continuous action and return (obs, reward, done).

        Action is expected in [-1, 1] range (normalized).
        This method scales it by MAX_TORQUE to match GPU behavior.
        """
        var raw_action = Scalar[Self.dtype](action[0]) if len(
            action
        ) > 0 else Scalar[Self.dtype](0.0)
        # Scale action from [-1, 1] to [-MAX_TORQUE, MAX_TORQUE]
        # This matches the GPU step kernel behavior
        var torque = raw_action * self.max_torque
        var result = self._step_with_torque(torque)
        var obs = List[Scalar[DTYPE_VEC]](capacity=3)
        obs.append(Scalar[DTYPE_VEC](cos(Float64(self.theta))))
        obs.append(Scalar[DTYPE_VEC](sin(Float64(self.theta))))
        obs.append(Scalar[DTYPE_VEC](self.theta_dot))
        return (obs^, Scalar[DTYPE_VEC](result[1]), result[2])

    fn obs_dim(self) -> Int:
        """Return observation dimension (3)."""
        return 3

    fn action_dim(self) -> Int:
        """Return action dimension (1)."""
        return 1

    fn action_low(self) -> Scalar[Self.dtype]:
        """Return lower bound for action values."""
        return -self.max_torque

    fn action_high(self) -> Scalar[Self.dtype]:
        """Return upper bound for action values."""
        return self.max_torque

    # =========================================================================
    # CPU Single-Environment Methods - BoxDiscreteActionEnv Trait
    # =========================================================================

    fn step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Take discrete action and return (obs_list, reward, done)."""
        var torque = Scalar[Self.dtype](Float64(action - 1) * 2.0)
        var result = self._step_with_torque(torque)
        return (self.get_obs_list(), result[1], result[2])

    fn num_actions(self) -> Int:
        """Return number of discrete actions (3)."""
        return 3

    # =========================================================================
    # CPU Single-Environment Methods - DiscreteEnv Trait
    # =========================================================================

    fn reset(mut self) -> PendulumV2State[Self.dtype]:
        """Reset environment and return discretized state."""
        _ = self.reset_obs_list()
        return self.get_state()

    fn step(
        mut self, action: PendulumV2Action[Self.dtype]
    ) -> Tuple[PendulumV2State[Self.dtype], Scalar[Self.dtype], Bool]:
        """Take action and return (state, reward, done)."""
        var torque = action.torque
        var result = self._step_with_torque(torque)
        return (self.get_state(), result[1], result[2])

    fn get_state(self) -> PendulumV2State[Self.dtype]:
        """Return current observation state."""
        return PendulumV2State[Self.dtype].from_theta(
            self.theta, self.theta_dot
        )

    fn state_to_index(self, state: PendulumV2State[Self.dtype]) -> Int:
        """Convert state to index for tabular methods."""
        return self._discretize_obs()

    fn action_from_index(self, action_idx: Int) -> PendulumV2Action[Self.dtype]:
        """Create action from index."""
        return PendulumV2Action[Self.dtype].from_discrete(action_idx)

    fn num_states(self) -> Int:
        """Return total number of discrete states."""
        return self.num_bins_angle * self.num_bins_velocity

    # =========================================================================
    # Internal CPU Helpers
    # =========================================================================

    fn _step_with_torque(
        mut self, torque: Scalar[Self.dtype]
    ) -> Tuple[PendulumV2State[Self.dtype], Scalar[Self.dtype], Bool]:
        """Internal step function that accepts continuous torque."""
        # Clamp torque
        var u = torque
        if u > self.max_torque:
            u = self.max_torque
        elif u < -self.max_torque:
            u = -self.max_torque

        self.last_torque = u

        # Physics: θ'' = (3g/2L) * sin(θ) + (3/mL²) * u
        var sin_theta = Scalar[Self.dtype](sin(Float64(self.theta)))
        var theta_acc = (Scalar[Self.dtype](3.0) * self.g) / (
            Scalar[Self.dtype](2.0) * self.l
        ) * sin_theta + (
            Scalar[Self.dtype](3.0) / (self.m * self.l * self.l)
        ) * u

        # Euler integration
        self.theta_dot = self.theta_dot + theta_acc * self.dt
        self.theta = self.theta + self.theta_dot * self.dt

        # Clip angular velocity
        if self.theta_dot > self.max_speed:
            self.theta_dot = self.max_speed
        elif self.theta_dot < -self.max_speed:
            self.theta_dot = -self.max_speed

        # Normalize angle to [-π, π]
        self.theta = self._angle_normalize(self.theta)

        self.steps += 1

        # Compute reward: -(θ² + 0.1*θ_dot² + 0.001*u²)
        var reward = -(
            self.theta * self.theta
            + Scalar[Self.dtype](0.1) * self.theta_dot * self.theta_dot
            + Scalar[Self.dtype](0.001) * u * u
        )
        self.total_reward += reward

        # Pendulum never terminates early, only truncates at max_steps
        self.done = self.steps >= self.max_steps

        return (self.get_state(), reward, self.done)

    fn _angle_normalize(self, x: Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        """Normalize angle to [-π, π]."""
        var result = x
        var pi_val = Scalar[Self.dtype](pi)
        var two_pi = Scalar[Self.dtype](2.0 * pi)
        while result > pi_val:
            result -= two_pi
        while result < -pi_val:
            result += two_pi
        return result

    fn _discretize_obs(self) -> Int:
        """Discretize current continuous observation into a single state index.
        """

        fn bin_value(
            value: Float64, low: Float64, high: Float64, bins: Int
        ) -> Int:
            var normalized = (value - low) / (high - low)
            if normalized < 0.0:
                normalized = 0.0
            elif normalized > 1.0:
                normalized = 1.0
            return Int(normalized * Float64(bins - 1))

        var b_angle = bin_value(
            Float64(self.theta), -pi, pi, self.num_bins_angle
        )
        var b_vel = bin_value(
            Float64(self.theta_dot),
            Float64(-self.max_speed),
            Float64(self.max_speed),
            self.num_bins_velocity,
        )

        return b_angle * self.num_bins_velocity + b_vel

    fn is_done(self) -> Bool:
        """Check if episode is done."""
        return self.done

    # =========================================================================
    # Rendering
    # =========================================================================

    fn render(mut self, mut renderer: RendererBase):
        """Render the current state using SDL2.

        Args:
            renderer: External renderer to use for drawing.
        """
        if not renderer.begin_frame():
            return

        # Convert state variables to Float64 for rendering
        var theta_f64 = Float64(self.theta)
        var theta_dot_f64 = Float64(self.theta_dot)
        var last_torque_f64 = Float64(self.last_torque)

        # Colors
        var sky_color = sky_blue()
        var rod_color = rgb(139, 69, 19)  # Saddle brown
        var bob_color = rgb(255, 0, 0)  # Red
        var pivot_color = rgb(50, 50, 50)  # Dark gray
        var torque_color = rgb(0, 200, 0)  # Green

        # Clear screen with sky color
        renderer.clear_with_color(sky_color)

        # Create camera centered on screen (Y-flip for physics coords)
        var zoom = 100.0  # pixels per world unit
        var camera = renderer.make_camera(zoom, True)

        # World coordinates
        var pivot = Vec2(0.0, 0.0)  # Pivot at origin
        var rod_length_world = 1.5  # Rod length in world units
        var bob_radius_world = 0.2

        # Draw reference circle (the trajectory the bob follows)
        renderer.draw_circle_world(
            pivot, rod_length_world, camera, light_gray(), False
        )

        # Draw torque indicator (arc showing applied torque)
        if last_torque_f64 != 0.0:
            var torque_scale = abs(last_torque_f64) * 0.3
            var torque_direction = 1.0 if last_torque_f64 > 0 else -1.0
            var arc_end = Vec2(
                torque_direction * 0.3,
                0.3 + torque_scale,
            )
            renderer.draw_line_world(
                pivot + Vec2(0, 0.3),
                pivot + arc_end,
                camera,
                torque_color,
                4,
            )

        # Draw pendulum using helper
        # Note: theta=0 points up (negative Y in screen coords before flip)
        renderer.draw_pendulum(
            pivot,
            theta_f64 + pi,  # Adjust so 0 = down for the helper
            rod_length_world,
            bob_radius_world,
            camera,
            rod_color,
            bob_color,
            pivot_color,
            8,  # rod width
        )

        # Draw bob border
        var bob_pos = Vec2(
            pivot.x + rod_length_world * sin(theta_f64),
            pivot.y - rod_length_world * cos(theta_f64),
        )
        renderer.draw_circle_world(
            bob_pos, bob_radius_world, camera, black(), False
        )

        # Draw info text
        var info_lines = List[String]()
        info_lines.append("Step: " + String(self.steps))
        info_lines.append("Reward: " + String(Int(self.total_reward)))
        info_lines.append(
            "Angle: " + String(theta_f64 * 180.0 / pi)[:6] + " deg"
        )
        info_lines.append("Vel: " + String(theta_dot_f64)[:6])
        info_lines.append("Torque: " + String(last_torque_f64)[:5])
        renderer.draw_info_box(info_lines)

        # Update display
        renderer.flip()

    fn close(mut self):
        """Clean up resources (no-op since renderer is external)."""
        pass

    # =========================================================================
    # Static Factory Methods
    # =========================================================================

    @staticmethod
    fn make_tile_coding(
        num_tilings: Int = 8,
        tiles_per_dim: Int = 8,
    ) -> TileCoding:
        """Create tile coding configured for Pendulum environment.

        Pendulum observation: [cos(θ), sin(θ), θ_dot]

        Args:
            num_tilings: Number of tilings (default 8)
            tiles_per_dim: Tiles per dimension (default 8)

        Returns:
            TileCoding configured for Pendulum observation space
        """
        var tiles = List[Int]()
        tiles.append(tiles_per_dim)  # cos(θ)
        tiles.append(tiles_per_dim)  # sin(θ)
        tiles.append(tiles_per_dim)  # θ_dot

        # Observation bounds
        var state_low = List[Float64]()
        state_low.append(-1.0)  # cos(θ) min
        state_low.append(-1.0)  # sin(θ) min
        state_low.append(-8.0)  # θ_dot min

        var state_high = List[Float64]()
        state_high.append(1.0)  # cos(θ) max
        state_high.append(1.0)  # sin(θ) max
        state_high.append(8.0)  # θ_dot max

        return TileCoding(
            num_tilings=num_tilings,
            tiles_per_dim=tiles^,
            state_low=state_low^,
            state_high=state_high^,
        )

    @staticmethod
    fn make_poly_features(degree: Int = 2) -> PolynomialFeatures:
        """Create polynomial features for Pendulum (3D observation).

        Args:
            degree: Maximum polynomial degree

        Returns:
            PolynomialFeatures extractor configured for Pendulum
        """
        var state_low = List[Float64]()
        state_low.append(-1.0)  # cos(θ)
        state_low.append(-1.0)  # sin(θ)
        state_low.append(-8.0)  # θ_dot

        var state_high = List[Float64]()
        state_high.append(1.0)  # cos(θ)
        state_high.append(1.0)  # sin(θ)
        state_high.append(8.0)  # θ_dot

        return PolynomialFeatures(
            state_dim=3,
            degree=degree,
            include_bias=True,
            state_low=state_low^,
            state_high=state_high^,
        )

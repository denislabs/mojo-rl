"""Environment trait hierarchy for different state/action space types.

This module defines traits that categorize environments by their state and action
space types, enabling compile-time checking of algorithm-environment compatibility.

Trait Hierarchy:
    Env (base)
    ├── DiscreteStateEnv   - States can be converted to integer indices
    ├── ContinuousStateEnv - States are continuous observation vectors (List[Float64])
    ├── DiscreteActionEnv  - Actions are discrete integers
    └── ContinuousActionEnv - Actions are continuous vectors

Combined Traits:
    DiscreteEnv           = DiscreteStateEnv & DiscreteActionEnv
    BoxDiscreteActionEnv  = ContinuousStateEnv & DiscreteActionEnv
    BoxContinuousActionEnv = ContinuousStateEnv & ContinuousActionEnv

Environments implement combinations:
    GridWorld:   DiscreteEnv (tabular)
    CartPole:    DiscreteEnv + BoxDiscreteActionEnv (4D obs)
    MountainCar: DiscreteEnv + BoxDiscreteActionEnv (2D obs)
    Acrobot:     DiscreteEnv + BoxDiscreteActionEnv (6D obs)
    Pendulum:    DiscreteEnv + BoxContinuousActionEnv (3D obs, 1D action)

Algorithms specify requirements:
    Q-Learning (tabular): DiscreteEnv
    Tile-coded Q-Learning: BoxDiscreteActionEnv
    PPO:                   BoxDiscreteActionEnv (or BoxContinuousActionEnv)
    SAC/DDPG/TD3:          BoxContinuousActionEnv
"""

from .env import Env
from layout import LayoutTensor, Layout
from deep_rl import dtype
from gpu import DeviceContext, DeviceBuffer

# ============================================================================
# State Space Traits
# ============================================================================


trait DiscreteStateEnv(Env):
    """Environment with discrete states that can be indexed.

    Use this for tabular methods where states map to integer indices.
    The state space must be finite and enumerable.

    Examples: GridWorld, FrozenLake, Taxi, discretized CartPole
    """

    fn state_to_index(self, state: Self.StateType) -> Int:
        """Convert a state to an integer index for tabular methods."""
        ...

    fn num_states(self) -> Int:
        """Return the total number of discrete states."""
        ...


trait ContinuousStateEnv(Env):
    """Environment with continuous observation/state vectors.

    Use this for function approximation methods (tile coding, neural networks)
    where states are represented as continuous vectors.

    Observations are returned as List[Float64] for flexibility with any
    observation dimension. Environments may also provide SIMD-optimized
    methods internally for performance.

    Examples: CartPole (4D), MountainCar (2D), Acrobot (6D), MuJoCo environments.
    """

    fn get_obs_list(self) -> List[Float64]:
        """Return current continuous observation as a flexible list."""
        ...

    fn reset_obs_list(mut self) -> List[Float64]:
        """Reset environment and return initial continuous observation."""
        ...

    fn obs_dim(self) -> Int:
        """Return the dimension of the observation vector."""
        ...


# ============================================================================
# Action Space Traits
# ============================================================================


trait DiscreteActionEnv(Env):
    """Environment with discrete action space.

    Actions are represented as integer indices. Use this for environments
    where the agent chooses from a finite set of actions.

    Examples: CartPole (left/right), GridWorld (up/down/left/right)
    """

    fn action_from_index(self, action_idx: Int) -> Self.ActionType:
        """Create an action from an integer index."""
        ...

    fn num_actions(self) -> Int:
        """Return the number of discrete actions available."""
        ...


trait ContinuousActionEnv(Env):
    """Environment with continuous action space.

    Actions are continuous vectors. Use this for fine-grained control tasks
    where actions can take any value within bounds.

    Examples: Pendulum (torque), HalfCheetah (joint torques)
    """

    fn action_dim(self) -> Int:
        """Return the dimension of the action vector."""
        ...

    fn action_low(self) -> Float64:
        """Return the lower bound for action values.

        Note: Assumes symmetric bounds. For asymmetric bounds,
        environments should provide additional methods.
        """
        ...

    fn action_high(self) -> Float64:
        """Return the upper bound for action values."""
        ...


# ============================================================================
# Combined Traits (Common Combinations)
# ============================================================================


trait DiscreteEnv(DiscreteActionEnv, DiscreteStateEnv):
    """Environment with discrete states and actions suitable for tabular RL.

    Combines discrete state and discrete action spaces.
    Use with Q-Learning, SARSA, Monte Carlo, etc.

    This is the primary trait for tabular RL methods that require
    integer indices for states and actions.

    Examples: GridWorld, FrozenLake, Taxi, discretized CartPole
    """

    pass


trait BoxDiscreteActionEnv(ContinuousStateEnv, DiscreteActionEnv):
    """Environment with continuous observations (Box space) and discrete actions.

    Use with function approximation algorithms that handle continuous observations
    but discrete action selection:
    - Tile coding / Linear function approximation
    - Policy gradient methods (REINFORCE, Actor-Critic, PPO)
    - DQN

    Examples: CartPole (4D), MountainCar (2D), Acrobot (6D), LunarLander.
    """

    fn step_obs(mut self, action: Int) -> Tuple[List[Float64], Float64, Bool]:
        """Take discrete action and return (continuous_obs, reward, done).

        Convenience method for function approximation algorithms that
        work with raw observations and integer actions.
        """
        ...


trait BoxContinuousActionEnv(ContinuousActionEnv, ContinuousStateEnv):
    """Environment with continuous observations and continuous actions.

    Use with continuous control algorithms:
    - Policy gradient with Gaussian policies
    - DDPG, TD3, SAC

    Examples: Pendulum (3D obs, 1D action), HalfCheetah, Ant, Humanoid.
    """

    fn step_continuous(
        mut self, action: Float64
    ) -> Tuple[List[Float64], Float64, Bool]:
        """Take continuous action and return (continuous_obs, reward, done).

        Convenience method for continuous control algorithms that
        work with raw observations and continuous actions.

        Note: For multi-dimensional action spaces, environments should
        provide additional methods accepting action vectors.
        """
        ...


# ============================================================================
# GPU Environment Trait for Composable GPU RL. (Experimental)
# ============================================================================


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
    fn step_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        actions: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), ImmutAnyOrigin
        ],
        rewards: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        dones: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
    ):
        """Perform one environment step. Returns (reward, done).

        Note: MutAnyOrigin allows mutation without `mut` keyword.
        Actions use ImmutAnyOrigin since they are read-only.
        Actions should be 0.0, 1.0, 2.0, etc. (cast to Int inside kernel).
        Dones should be 0.0 (false) or 1.0 (true).
        """
        ...

    @staticmethod
    fn reset_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
    ):
        """Reset state to random initial values."""
        ...

    @staticmethod
    fn step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[dtype],
        actions: DeviceBuffer[dtype],
        mut rewards: DeviceBuffer[dtype],
        mut dones: DeviceBuffer[dtype],
    ) raises:
        """Perform one environment step. Returns (reward, done)."""
        ...

    @staticmethod
    fn reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](ctx: DeviceContext, mut states: DeviceBuffer[dtype],) raises:
        """Reset state to random initial values."""
        ...

    @staticmethod
    fn selective_reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[dtype],
        mut dones: DeviceBuffer[dtype],
        rng_seed: UInt32,
    ) raises:
        """Reset only done environments to random initial values.

        This enables efficient vectorized training where only completed
        episodes are reset while others continue running.
        """
        ...

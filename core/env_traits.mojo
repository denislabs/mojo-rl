"""Environment trait hierarchy for different state/action space types.

This module defines traits that categorize environments by their state and action
space types, enabling compile-time checking of algorithm-environment compatibility.

Trait Hierarchy:
    Env (base)
    ├── DiscreteStateEnv  - States can be converted to integer indices
    ├── ContinuousStateEnv - States are continuous observation vectors
    ├── DiscreteActionEnv  - Actions are discrete integers
    └── ContinuousActionEnv - Actions are continuous vectors

Environments implement combinations:
    GridWorld: DiscreteStateEnv & DiscreteActionEnv
    CartPole:  ContinuousStateEnv & DiscreteActionEnv
    Pendulum:  ContinuousStateEnv & ContinuousActionEnv

Algorithms specify requirements:
    Q-Learning: DiscreteStateEnv & DiscreteActionEnv
    PPO:        ContinuousStateEnv & DiscreteActionEnv (or ContinuousActionEnv)
    SAC:        ContinuousStateEnv & ContinuousActionEnv
"""

from .env import Env


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

    Examples: CartPole, MountainCar, Pendulum, MuJoCo environments
    """

    fn get_obs(self) -> SIMD[DType.float64, 4]:
        """Return current continuous observation vector.

        Note: Returns SIMD[4] for common 4D observations. Environments with
        different observation dimensions should provide additional methods.
        """
        ...

    fn reset_obs(mut self) -> SIMD[DType.float64, 4]:
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


trait DiscreteEnv(DiscreteStateEnv, DiscreteActionEnv):
    """Environment with discrete states and actions suitable for tabular RL.

    Combines discrete state and discrete action spaces.
    Use with Q-Learning, SARSA, Monte Carlo, etc.

    This is the primary trait for tabular RL methods that require
    integer indices for states and actions.

    Examples: GridWorld, FrozenLake, Taxi, discretized CartPole
    """

    pass


# Alias for backward compatibility / alternative naming
comptime TabularEnv = DiscreteEnv


trait ClassicControlEnv(ContinuousStateEnv, DiscreteActionEnv):
    """Environment with continuous observations and discrete actions.

    Common for classic control tasks. Use with:
    - Tile coding / Linear function approximation
    - Policy gradient methods (REINFORCE, Actor-Critic, PPO)
    - DQN

    Examples: CartPole, MountainCar, Acrobot, LunarLander
    """

    fn step_raw(mut self, action: Int) -> Tuple[SIMD[DType.float64, 4], Float64, Bool]:
        """Take action (as int) and return (continuous_obs, reward, done).

        Convenience method for function approximation algorithms that
        work with raw observations and integer actions.
        """
        ...


trait ContinuousControlEnv(ContinuousStateEnv, ContinuousActionEnv):
    """Environment with continuous observations and continuous actions.

    Use with continuous control algorithms:
    - Policy gradient with Gaussian policies
    - DDPG, TD3, SAC

    Examples: Pendulum, HalfCheetah, Ant, Humanoid
    """

    pass

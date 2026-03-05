"""PPO CPU state containers.

Separates heap-allocated rollout state (actor/critic networks + rollout
buffers) from algorithm logic (hyperparameters + update rules) in
DeepPPOAgent and DeepPPOContinuousAgent.

Mirrors the DDPG/TD3/SAC pattern where DDPGCPUState holds all mutable data
and DeepDDPGAgent holds only hyperparameters and update logic.

Two state structs:
    PPODiscreteState  — for DeepPPOAgent (discrete actions)
    PPOContinuousState — for DeepPPOContinuousAgent (continuous actions)

Both implement the corresponding OnPolicyDiscreteState / OnPolicyContinuousState
traits defined in deep_agents/core/onpolicy_train.mojo.

Usage:
    # Discrete PPO
    var state = agent.make_cpu_state()
    for _ in range(num_updates):
        agent.collect_rollout(state, env)
        agent.compute_advantages(state)
        var loss = agent.update_epochs(state)

    # or shorthand (backward compat)
    var metrics = agent.train(env, num_episodes=500)
"""

from layout import Layout, LayoutTensor
from nn.constants import dtype
from nn.model import Model
from nn.optimizer import Optimizer
from nn.training import NetworkState, Network
from nn.initializer import Xavier, Kaiming
from deep_agents.core.onpolicy_train import (
    OnPolicyDiscreteState,
    OnPolicyContinuousState,
)


# =============================================================================
# PPODiscreteState — CPU state container for discrete-action PPO
# =============================================================================


struct PPODiscreteState[
    ActorModel: Model,
    ActorOpt: Optimizer,
    CriticModel: Model,
    CriticOpt: Optimizer,
    obs_dim: Int,
    num_actions: Int,
    rollout_len: Int,
](Movable, OnPolicyDiscreteState):
    """CPU-resident state for discrete-action PPO training.

    Holds all heap-allocated data needed for one PPO training loop:
      - Actor and critic NetworkStates (weights + optimizer state)
      - Rollout buffers for obs, actions, rewards, values, log_probs, dones
      - Advantage and return buffers (filled by compute_advantages)
      - Current observation for between-rollout bootstrapping
      - Shuffled index buffer for minibatch sampling

    Created via agent.make_cpu_state() or PPODiscreteState[...]() directly.

    Parameters:
        ActorModel: Actor network model type (implements Model trait).
        ActorOpt: Actor optimizer type (implements Optimizer trait).
        CriticModel: Critic network model type.
        CriticOpt: Critic optimizer type.
        obs_dim: Observation space dimension.
        num_actions: Number of discrete actions.
        rollout_len: Steps per rollout (compile-time constant).
    """

    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.num_actions
    comptime ROLLOUT = Self.rollout_len
    comptime ActorNet = Network[Self.ActorModel, Self.ActorOpt]
    comptime CriticNet = Network[Self.CriticModel, Self.CriticOpt]

    # Networks: actor + critic (online weights + optimizer state)
    var actor: NetworkState[Self.ActorModel, Self.ActorOpt]
    var critic: NetworkState[Self.CriticModel, Self.CriticOpt]

    # Rollout buffers (heap-allocated Lists to avoid stack overflow)
    var buffer_obs: List[Scalar[dtype]]      # [rollout_len * obs_dim]
    var buffer_actions: List[Int]            # [rollout_len]
    var buffer_rewards: List[Scalar[dtype]]  # [rollout_len]
    var buffer_values: List[Scalar[dtype]]   # [rollout_len]
    var buffer_log_probs: List[Scalar[dtype]] # [rollout_len]
    var buffer_dones: List[Bool]             # [rollout_len]
    var buffer_idx: Int                      # current fill position

    # Computed by compute_advantages (filled after collect_rollout)
    var _advantages: List[Scalar[dtype]]     # [rollout_len]
    var _returns: List[Scalar[dtype]]        # [rollout_len]

    # Last observation after rollout ends (used to bootstrap value)
    var _current_obs: List[Scalar[dtype]]    # [obs_dim]
    var _env_initialized: Bool

    # Shuffled index buffer (reused across epochs in update_epochs)
    var _indices: List[Int]                  # [rollout_len]

    fn __init__(out self):
        """Allocate networks (Xavier init), rollout buffers, and scratch."""
        # Initialize actor and critic with Xavier
        self.actor = NetworkState[Self.ActorModel, Self.ActorOpt]()
        self.actor.initialize[Xavier]()
        self.critic = NetworkState[Self.CriticModel, Self.CriticOpt]()
        self.critic.initialize[Xavier]()

        # Allocate and zero-fill rollout buffers
        self.buffer_obs = List[Scalar[dtype]](
            capacity=Self.ROLLOUT * Self.OBS
        )
        for _ in range(Self.ROLLOUT * Self.OBS):
            self.buffer_obs.append(Scalar[dtype](0))

        self.buffer_actions = List[Int](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self.buffer_actions.append(0)

        self.buffer_rewards = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self.buffer_rewards.append(Scalar[dtype](0))

        self.buffer_values = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self.buffer_values.append(Scalar[dtype](0))

        self.buffer_log_probs = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self.buffer_log_probs.append(Scalar[dtype](0))

        self.buffer_dones = List[Bool](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self.buffer_dones.append(False)

        self.buffer_idx = 0

        # Advantage and return scratch
        self._advantages = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self._advantages.append(Scalar[dtype](0))

        self._returns = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self._returns.append(Scalar[dtype](0))

        # Current obs scratch
        self._current_obs = List[Scalar[dtype]](capacity=Self.OBS)
        for _ in range(Self.OBS):
            self._current_obs.append(Scalar[dtype](0))

        self._env_initialized = False

        # Index buffer for minibatch shuffling
        self._indices = List[Int](capacity=Self.ROLLOUT)
        for i in range(Self.ROLLOUT):
            self._indices.append(i)

    # -------------------------------------------------------------------------
    # OnPolicyDiscreteState trait methods
    # -------------------------------------------------------------------------

    fn store_step(
        mut self,
        obs: List[Scalar[dtype]],
        action: Int,
        reward: Float64,
        value: Scalar[dtype],
        log_prob: Scalar[dtype],
        done: Bool,
    ) -> None:
        """Store one step (obs, action, reward, value, log_prob, done) in the rollout buffer."""
        var idx = self.buffer_idx
        for i in range(Self.OBS):
            self.buffer_obs[idx * Self.OBS + i] = obs[i]
        self.buffer_actions[idx] = action
        self.buffer_rewards[idx] = Scalar[dtype](reward)
        self.buffer_values[idx] = value
        self.buffer_log_probs[idx] = log_prob
        self.buffer_dones[idx] = done
        self.buffer_idx += 1

    fn is_full(self) -> Bool:
        """Return True when rollout_len steps have been collected."""
        return self.buffer_idx >= Self.ROLLOUT

    fn clear(mut self) -> None:
        """Reset the rollout buffer write pointer."""
        self.buffer_idx = 0


# =============================================================================
# PPOContinuousState — CPU state container for continuous-action PPO
# =============================================================================


struct PPOContinuousState[
    ActorModel: Model,
    ActorOpt: Optimizer,
    CriticModel: Model,
    CriticOpt: Optimizer,
    obs_dim: Int,
    action_dim: Int,
    rollout_len: Int,
](Movable, OnPolicyContinuousState):
    """CPU-resident state for continuous-action PPO training.

    Same layout as PPODiscreteState but with continuous action buffers:
      buffer_actions: List[Scalar[dtype]] of length rollout_len * action_dim

    Parameters:
        ActorModel: Actor network model type (implements Model trait).
        ActorOpt: Actor optimizer type (implements Optimizer trait).
        CriticModel: Critic network model type.
        CriticOpt: Critic optimizer type.
        obs_dim: Observation space dimension.
        action_dim: Continuous action space dimension.
        rollout_len: Steps per rollout (compile-time constant).
    """

    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.action_dim
    comptime ROLLOUT = Self.rollout_len
    comptime ActorNet = Network[Self.ActorModel, Self.ActorOpt]
    comptime CriticNet = Network[Self.CriticModel, Self.CriticOpt]

    # Networks
    var actor: NetworkState[Self.ActorModel, Self.ActorOpt]
    var critic: NetworkState[Self.CriticModel, Self.CriticOpt]

    # Rollout buffers
    var buffer_obs: List[Scalar[dtype]]      # [rollout_len * obs_dim]
    var buffer_actions: List[Scalar[dtype]]  # [rollout_len * action_dim]
    var buffer_rewards: List[Scalar[dtype]]  # [rollout_len]
    var buffer_values: List[Scalar[dtype]]   # [rollout_len]
    var buffer_log_probs: List[Scalar[dtype]] # [rollout_len]
    var buffer_dones: List[Bool]             # [rollout_len]
    var buffer_idx: Int

    # Advantage and return scratch
    var _advantages: List[Scalar[dtype]]     # [rollout_len]
    var _returns: List[Scalar[dtype]]        # [rollout_len]

    # Current obs for bootstrapping
    var _current_obs: List[Scalar[dtype]]    # [obs_dim]
    var _env_initialized: Bool

    # Shuffled index buffer
    var _indices: List[Int]                  # [rollout_len]

    fn __init__(out self):
        """Allocate networks (Kaiming init for ReLU/Tanh), rollout buffers, scratch."""
        self.actor = NetworkState[Self.ActorModel, Self.ActorOpt]()
        self.actor.initialize[Kaiming]()
        self.critic = NetworkState[Self.CriticModel, Self.CriticOpt]()
        self.critic.initialize[Kaiming]()

        self.buffer_obs = List[Scalar[dtype]](
            capacity=Self.ROLLOUT * Self.OBS
        )
        for _ in range(Self.ROLLOUT * Self.OBS):
            self.buffer_obs.append(Scalar[dtype](0))

        self.buffer_actions = List[Scalar[dtype]](
            capacity=Self.ROLLOUT * Self.ACTIONS
        )
        for _ in range(Self.ROLLOUT * Self.ACTIONS):
            self.buffer_actions.append(Scalar[dtype](0))

        self.buffer_rewards = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self.buffer_rewards.append(Scalar[dtype](0))

        self.buffer_values = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self.buffer_values.append(Scalar[dtype](0))

        self.buffer_log_probs = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self.buffer_log_probs.append(Scalar[dtype](0))

        self.buffer_dones = List[Bool](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self.buffer_dones.append(False)

        self.buffer_idx = 0

        self._advantages = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self._advantages.append(Scalar[dtype](0))

        self._returns = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(Self.ROLLOUT):
            self._returns.append(Scalar[dtype](0))

        self._current_obs = List[Scalar[dtype]](capacity=Self.OBS)
        for _ in range(Self.OBS):
            self._current_obs.append(Scalar[dtype](0))

        self._env_initialized = False

        self._indices = List[Int](capacity=Self.ROLLOUT)
        for i in range(Self.ROLLOUT):
            self._indices.append(i)

    # -------------------------------------------------------------------------
    # OnPolicyContinuousState trait methods
    # -------------------------------------------------------------------------

    fn store_step(
        mut self,
        obs: List[Scalar[dtype]],
        action: List[Scalar[dtype]],
        reward: Float64,
        value: Scalar[dtype],
        log_prob: Scalar[dtype],
        done: Bool,
    ) -> None:
        """Store one step (obs, action, reward, value, log_prob, done) in the rollout buffer."""
        var idx = self.buffer_idx
        for i in range(Self.OBS):
            self.buffer_obs[idx * Self.OBS + i] = obs[i]
        for i in range(Self.ACTIONS):
            self.buffer_actions[idx * Self.ACTIONS + i] = action[i]
        self.buffer_rewards[idx] = Scalar[dtype](reward)
        self.buffer_values[idx] = value
        self.buffer_log_probs[idx] = log_prob
        self.buffer_dones[idx] = done
        self.buffer_idx += 1

    fn is_full(self) -> Bool:
        """Return True when rollout_len steps have been collected."""
        return self.buffer_idx >= Self.ROLLOUT

    fn clear(mut self) -> None:
        """Reset the rollout buffer write pointer."""
        self.buffer_idx = 0

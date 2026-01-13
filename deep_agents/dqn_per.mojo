"""DQN Agent with Prioritized Experience Replay using the new trait-based architecture.

This DQN+PER implementation uses:
- Network wrapper from deep_rl.training for stateless model + params management
- seq() composition for building Q-networks
- PrioritizedReplayBuffer from deep_rl.replay for priority-weighted sampling

Key differences from standard DQN:
- Samples transitions proportionally to TD error magnitude
- Uses importance sampling weights to correct for non-uniform sampling bias
- Updates priorities after each training step based on new TD errors

Features:
- Works with any BoxDiscreteActionEnv (continuous obs, discrete actions)
- Epsilon-greedy exploration with decay
- Target network with soft updates
- Double DQN support via compile-time parameter
- Beta annealing for importance sampling correction

Usage:
    from deep_agents.dqn_per import DQNPERAgent
    from envs import LunarLanderEnv

    var env = LunarLanderEnv()
    var agent = DQNPERAgent[8, 4, 128, 100000, 64]()

    var metrics = agent.train(env, num_episodes=500)

Reference: Schaul et al., "Prioritized Experience Replay" (2015)
"""

from math import exp
from random import random_float64, seed

from layout import Layout, LayoutTensor

from deep_rl.constants import dtype, TILE, TPB
from deep_rl.model import Linear, ReLU, seq
from deep_rl.optimizer import Adam
from deep_rl.initializer import Kaiming
from deep_rl.training import Network
from deep_rl.replay import PrioritizedReplayBuffer
from core import TrainingMetrics, BoxDiscreteActionEnv


# =============================================================================
# DQN + PER Agent
# =============================================================================


struct DQNPERAgent[
    obs_dim: Int,
    num_actions: Int,
    hidden_dim: Int = 128,
    buffer_capacity: Int = 20000,
    batch_size: Int = 64,
    double_dqn: Bool = True,
]:
    """DQN Agent with Prioritized Experience Replay using new trait-based architecture.

    PER samples transitions proportionally to their TD error magnitude, which
    helps the agent learn more efficiently from important experiences.

    Key features:
    - Priority-weighted sampling based on TD errors
    - Importance sampling weights correct for non-uniform sampling bias
    - Beta annealing from beta_start to 1.0 over training
    - Double DQN support (compile-time flag)

    Parameters:
        obs_dim: Dimension of observation space.
        num_actions: Number of discrete actions.
        hidden_dim: Hidden layer size (default: 128).
        buffer_capacity: Replay buffer capacity (default: 100000).
        batch_size: Training batch size (default: 64).
        double_dqn: If True, use Double DQN (default: True).
    """

    # Convenience aliases
    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.num_actions
    comptime HIDDEN = Self.hidden_dim
    comptime BATCH = Self.batch_size

    # Cache size for network
    # Q-network: Linear[obs, h] + ReLU[h] + Linear[h, h] + ReLU[h] + Linear[h, actions]
    # Cache: OBS + HIDDEN + HIDDEN + HIDDEN + HIDDEN
    comptime CACHE_SIZE: Int = Self.OBS + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN

    # Q-network: obs -> hidden (ReLU) -> hidden (ReLU) -> num_actions
    var q_network: Network[
        type_of(
            seq(
                Linear[Self.OBS, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, Self.ACTIONS](),
            )
        ),
        Adam,
        Kaiming,
    ]

    # Target Q-network
    var target_network: Network[
        type_of(
            seq(
                Linear[Self.OBS, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, Self.ACTIONS](),
            )
        ),
        Adam,
        Kaiming,
    ]

    # Prioritized Replay Buffer (action_dim=1 for discrete actions stored as scalar)
    var buffer: PrioritizedReplayBuffer[
        Self.buffer_capacity, Self.obs_dim, 1, dtype
    ]

    # Standard DQN hyperparameters
    var gamma: Float64
    var tau: Float64
    var lr: Float64
    var epsilon: Float64
    var epsilon_min: Float64
    var epsilon_decay: Float64

    # PER hyperparameters
    var beta: Float64  # Current IS exponent (annealed to 1.0)
    var beta_start: Float64  # Initial beta value
    var beta_frames: Int  # Frames to anneal beta over

    # Training state
    var total_steps: Int
    var train_step_count: Int

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        tau: Float64 = 0.005,
        lr: Float64 = 0.0005,
        epsilon: Float64 = 1.0,
        epsilon_min: Float64 = 0.01,
        epsilon_decay: Float64 = 0.995,
        alpha: Float64 = 0.6,
        beta_start: Float64 = 0.4,
        beta_frames: Int = 100000,
    ):
        """Initialize DQN+PER agent.

        Args:
            gamma: Discount factor (default: 0.99).
            tau: Soft update coefficient (default: 0.005).
            lr: Learning rate (default: 0.0005).
            epsilon: Initial exploration rate (default: 1.0).
            epsilon_min: Minimum exploration rate (default: 0.01).
            epsilon_decay: Epsilon decay per episode (default: 0.995).
            alpha: Priority exponent (0=uniform, 1=full prioritization) (default: 0.6).
            beta_start: Initial IS correction exponent (default: 0.4).
            beta_frames: Frames to anneal beta from beta_start to 1.0 (default: 100000).
        """
        # Build Q-network model
        var q_model = seq(
            Linear[Self.OBS, Self.HIDDEN](),
            ReLU[Self.HIDDEN](),
            Linear[Self.HIDDEN, Self.HIDDEN](),
            ReLU[Self.HIDDEN](),
            Linear[Self.HIDDEN, Self.ACTIONS](),
        )

        # Initialize networks
        self.q_network = Network(q_model, Adam(lr=lr), Kaiming())
        self.target_network = Network(q_model, Adam(lr=lr), Kaiming())

        # Copy weights to target network
        self.target_network.copy_params_from(self.q_network)

        # Initialize prioritized replay buffer
        self.buffer = PrioritizedReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, 1, dtype
        ](alpha=Scalar[dtype](alpha), beta=Scalar[dtype](beta_start))

        # Store hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # PER hyperparameters
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames

        # Training state
        self.total_steps = 0
        self.train_step_count = 0

    fn select_action(
        self,
        obs: InlineArray[Scalar[dtype], Self.OBS],
        training: Bool = True,
    ) -> Int:
        """Select action using epsilon-greedy policy.

        Args:
            obs: Current observation.
            training: If True, use epsilon-greedy; else use greedy.

        Returns:
            Selected action index.
        """
        # Epsilon-greedy exploration
        if training and random_float64() < self.epsilon:
            return Int(random_float64() * Float64(Self.ACTIONS)) % Self.ACTIONS

        # Greedy action: argmax Q(s, a)
        var q_values = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        self.q_network.forward[1](obs, q_values)

        var best_action = 0
        var best_q = q_values[0]
        for a in range(1, Self.ACTIONS):
            if q_values[a] > best_q:
                best_q = q_values[a]
                best_action = a

        return best_action

    fn store_transition(
        mut self,
        obs: InlineArray[Scalar[dtype], Self.OBS],
        action: Int,
        reward: Float64,
        next_obs: InlineArray[Scalar[dtype], Self.OBS],
        done: Bool,
    ):
        """Store transition in prioritized replay buffer with max priority."""
        var action_arr = InlineArray[Scalar[dtype], 1](fill=0)
        action_arr[0] = Scalar[dtype](action)
        self.buffer.add(obs, action_arr, Scalar[dtype](reward), next_obs, done)
        self.total_steps += 1

        # Anneal beta towards 1.0
        var progress = Float64(self.total_steps) / Float64(self.beta_frames)
        if progress > 1.0:
            progress = 1.0
        self.beta = self.beta_start + progress * (1.0 - self.beta_start)
        self.buffer.set_beta(Scalar[dtype](self.beta))

    fn train_step(mut self) -> Float64:
        """Perform one training step with PER.

        Returns:
            TD loss value.
        """
        if not self.buffer.is_ready[Self.BATCH]():
            return 0.0

        # =====================================================================
        # Phase 1: Sample batch with importance sampling weights
        # =====================================================================
        var batch_obs = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var batch_actions_arr = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        var batch_rewards = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        var batch_next_obs = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var batch_dones = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        var batch_weights = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        var batch_indices = InlineArray[Int, Self.BATCH](uninitialized=True)

        self.buffer.sample[Self.BATCH](
            batch_obs,
            batch_actions_arr,
            batch_rewards,
            batch_next_obs,
            batch_dones,
            batch_weights,
            batch_indices,
        )

        # =====================================================================
        # Phase 2: Compute TD targets
        # =====================================================================

        var max_next_q = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )

        @parameter
        if Self.double_dqn:
            # Double DQN: online network selects best action, target evaluates it
            var online_next_q = InlineArray[
                Scalar[dtype], Self.BATCH * Self.ACTIONS
            ](uninitialized=True)
            var target_next_q = InlineArray[
                Scalar[dtype], Self.BATCH * Self.ACTIONS
            ](uninitialized=True)

            self.q_network.forward[Self.BATCH](batch_next_obs, online_next_q)
            self.target_network.forward[Self.BATCH](
                batch_next_obs, target_next_q
            )

            for b in range(Self.BATCH):
                # Online selects best action
                var best_action = 0
                var best_online_q = online_next_q[b * Self.ACTIONS]
                for a in range(1, Self.ACTIONS):
                    var q = online_next_q[b * Self.ACTIONS + a]
                    if q > best_online_q:
                        best_online_q = q
                        best_action = a

                # Target evaluates that action
                max_next_q[b] = target_next_q[b * Self.ACTIONS + best_action]
        else:
            # Standard DQN: max_a Q_target(s', a)
            var next_q = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
                uninitialized=True
            )
            self.target_network.forward[Self.BATCH](batch_next_obs, next_q)

            for b in range(Self.BATCH):
                var max_q = next_q[b * Self.ACTIONS]
                for a in range(1, Self.ACTIONS):
                    var q = next_q[b * Self.ACTIONS + a]
                    if q > max_q:
                        max_q = q
                max_next_q[b] = max_q

        # Compute TD targets: y = r + gamma * max_next_q * (1 - done)
        var targets = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)
        for b in range(Self.BATCH):
            var done_mask = Scalar[dtype](1.0) - batch_dones[b]
            targets[b] = (
                batch_rewards[b]
                + Scalar[dtype](self.gamma) * max_next_q[b] * done_mask
            )

        # =====================================================================
        # Phase 3: Forward with cache
        # =====================================================================

        var q_values = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        var cache = InlineArray[Scalar[dtype], Self.BATCH * Self.CACHE_SIZE](
            uninitialized=True
        )
        self.q_network.forward_with_cache[Self.BATCH](
            batch_obs, q_values, cache
        )

        # =====================================================================
        # Phase 4: Compute weighted loss and gradients
        # TD errors are used for priority updates
        # IS weights correct for non-uniform sampling bias
        # =====================================================================

        var loss: Float64 = 0.0
        var dq = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](fill=0)
        var td_errors = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )

        for b in range(Self.BATCH):
            var action = Int(batch_actions_arr[b])
            var q_idx = b * Self.ACTIONS + action
            var td_error = q_values[q_idx] - targets[b]

            # Store TD error for priority update (absolute value)
            td_errors[b] = td_error

            # Weighted MSE loss (IS weight corrects for non-uniform sampling)
            var weight = batch_weights[b]
            var weighted_error = weight * td_error
            loss += Float64(weighted_error * weighted_error)

            # Gradient: d(weighted_loss)/d(q) = 2 * weight * (q - target) / batch_size
            # With IS weighting applied
            dq[q_idx] = (
                Scalar[dtype](2.0) * weighted_error / Scalar[dtype](Self.BATCH)
            )

        loss /= Float64(Self.BATCH)

        # =====================================================================
        # Phase 5: Backward pass and update
        # =====================================================================

        var grad_input = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )

        self.q_network.zero_grads()
        self.q_network.backward[Self.BATCH](dq, grad_input, cache)
        self.q_network.update()

        # =====================================================================
        # Phase 6: Update priorities based on TD errors
        # =====================================================================

        self.buffer.update_priorities[Self.BATCH](batch_indices, td_errors)

        # =====================================================================
        # Phase 7: Soft update target network
        # =====================================================================

        self.target_network.soft_update_from(self.q_network, self.tau)

        self.train_step_count += 1

        return loss

    fn decay_epsilon(mut self):
        """Decay exploration rate (call once per episode)."""
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    fn _list_to_inline(
        self, obs_list: List[Float64]
    ) -> InlineArray[Scalar[dtype], Self.OBS]:
        """Convert List[Float64] to InlineArray."""
        var obs = InlineArray[Scalar[dtype], Self.OBS](fill=0)
        for i in range(Self.OBS):
            if i < len(obs_list):
                obs[i] = Scalar[dtype](obs_list[i])
        return obs

    fn get_epsilon(self) -> Float64:
        """Get current exploration rate."""
        return self.epsilon

    fn get_beta(self) -> Float64:
        """Get current IS correction exponent."""
        return self.beta

    fn get_train_steps(self) -> Int:
        """Get total training steps performed."""
        return self.train_step_count

    fn train[
        E: BoxDiscreteActionEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int,
        max_steps_per_episode: Int = 1000,
        warmup_steps: Int = 1000,
        train_every: Int = 1,
        verbose: Bool = False,
        print_every: Int = 10,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train the DQN+PER agent on a discrete action environment.

        Args:
            env: The environment to train on (must implement BoxDiscreteActionEnv).
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
            warmup_steps: Number of random steps to fill replay buffer.
            train_every: Train every N steps.
            verbose: Whether to print progress.
            print_every: Print progress every N episodes if verbose.
            environment_name: Name of environment for metrics labeling.

        Returns:
            TrainingMetrics object with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="DQN + PER",
            environment_name=environment_name,
        )

        # =====================================================================
        # Warmup: fill replay buffer with random actions
        # =====================================================================
        var warmup_obs_list = env.reset_obs_list()
        var warmup_obs = self._list_to_inline(warmup_obs_list)
        var warmup_count = 0

        while warmup_count < warmup_steps:
            # Random action
            var action = (
                Int(random_float64() * Float64(Self.ACTIONS)) % Self.ACTIONS
            )

            # Step environment
            var result = env.step_obs(action)
            var next_obs_list = result[0].copy()
            var reward = result[1]
            var done = result[2]

            var next_obs = self._list_to_inline(next_obs_list)
            self.store_transition(warmup_obs, action, reward, next_obs, done)

            warmup_obs = next_obs
            warmup_count += 1

            if done:
                warmup_obs_list = env.reset_obs_list()
                warmup_obs = self._list_to_inline(warmup_obs_list)

        if verbose:
            print("Warmup complete:", warmup_count, "transitions collected")

        # =====================================================================
        # Training loop
        # =====================================================================
        var total_train_steps = 0

        for episode in range(num_episodes):
            var obs_list = env.reset_obs_list()
            var obs = self._list_to_inline(obs_list)
            var episode_reward: Float64 = 0.0
            var episode_steps = 0

            for step in range(max_steps_per_episode):
                # Select action with epsilon-greedy
                var action = self.select_action(obs, training=True)

                # Step environment
                var result = env.step_obs(action)
                var next_obs_list = result[0].copy()
                var reward = result[1]
                var done = result[2]

                var next_obs = self._list_to_inline(next_obs_list)

                # Store transition (with max priority initially)
                self.store_transition(obs, action, reward, next_obs, done)

                # Train every N steps
                if total_train_steps % train_every == 0:
                    _ = self.train_step()

                episode_reward += reward
                obs = next_obs
                total_train_steps += 1
                episode_steps += 1

                if done:
                    break

            # Decay exploration rate
            self.decay_epsilon()

            # Log metrics
            metrics.log_episode(
                episode, episode_reward, episode_steps, self.epsilon
            )

            # Print progress
            if verbose and (episode + 1) % print_every == 0:
                var avg_reward = metrics.mean_reward_last_n(print_every)
                print(
                    "Episode",
                    episode + 1,
                    "| Avg reward:",
                    String(avg_reward)[:7],
                    "| Epsilon:",
                    String(self.epsilon)[:5],
                    "| Beta:",
                    String(self.beta)[:5],
                    "| Steps:",
                    total_train_steps,
                )

        return metrics^

    fn evaluate[
        E: BoxDiscreteActionEnv
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps: Int = 1000,
        verbose: Bool = False,
        render: Bool = False,
    ) -> Float64:
        """Evaluate the agent using greedy policy (no exploration).

        Args:
            env: The environment to evaluate on.
            num_episodes: Number of evaluation episodes.
            max_steps: Maximum steps per episode.
            verbose: Whether to print per-episode results.
            render: Whether to render the environment.

        Returns:
            Average reward over evaluation episodes.
        """
        var total_reward: Float64 = 0.0

        for episode in range(num_episodes):
            var obs_list = env.reset_obs_list()
            var obs = self._list_to_inline(obs_list)
            var episode_reward: Float64 = 0.0
            var episode_steps = 0

            for step in range(max_steps):
                # Greedy action (no exploration)
                var action = self.select_action(obs, training=False)

                # Step environment
                var result = env.step_obs(action)
                var next_obs_list = result[0].copy()
                var reward = result[1]
                var done = result[2]

                if render:
                    env.render()

                episode_reward += reward
                obs = self._list_to_inline(next_obs_list)
                episode_steps += 1

                if done:
                    break

            total_reward += episode_reward

            if verbose:
                print(
                    "Eval Episode",
                    episode + 1,
                    "| Reward:",
                    String(episode_reward)[:10],
                    "| Steps:",
                    episode_steps,
                )

        return total_reward / Float64(num_episodes)

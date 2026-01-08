"""Deep Q-Network (DQN) Agent for discrete action spaces.

DQN is the foundational deep RL algorithm for discrete control.
It uses:
- Q-Network: obs -> Q-values for all actions (2-layer MLP with ReLU)
- Target Network with soft updates for stability
- Experience replay to break sample correlation
- Epsilon-greedy exploration with decay

Reference: Mnih et al. "Playing Atari with Deep Reinforcement Learning" (2013)
           Mnih et al. "Human-level control through deep RL" (2015, Nature)

Example usage:
    from deep_agents.cpu import DeepDQNAgent
    from envs import LunarLanderEnv

    var env = LunarLanderEnv()
    var agent = DeepDQNAgent[obs_dim=8, num_actions=4, hidden_dim=128]()

    # Simple training
    var metrics = agent.train(env, num_episodes=500)
    var eval_reward = agent.evaluate(env)
"""

from random import random_float64

from deep_rl.cpu import (
    LinearAdam,
    ReplayBuffer,
    relu,
    relu_grad,
    elementwise_mul,
    zeros,
)
from core import TrainingMetrics, BoxDiscreteActionEnv


# =============================================================================
# Q-Network for DQN
# =============================================================================


struct QNetwork[
    obs_dim: Int,
    num_actions: Int,
    hidden1_dim: Int = 256,
    hidden2_dim: Int = 256,
    dtype: DType = DType.float64,
]:
    """Q-Network that outputs Q-values for all discrete actions.

    Architecture: obs_dim -> hidden1 (relu) -> hidden2 (relu) -> num_actions (linear).
    Output is Q(s, a) for each action a in [0, num_actions).
    """

    var layer1: LinearAdam[Self.obs_dim, Self.hidden1_dim, Self.dtype]
    var layer2: LinearAdam[Self.hidden1_dim, Self.hidden2_dim, Self.dtype]
    var layer3: LinearAdam[Self.hidden2_dim, Self.num_actions, Self.dtype]

    fn __init__(out self):
        """Initialize Q-Network with Xavier initialization."""
        self.layer1 = LinearAdam[Self.obs_dim, Self.hidden1_dim, Self.dtype]()
        self.layer2 = LinearAdam[
            Self.hidden1_dim, Self.hidden2_dim, Self.dtype
        ]()
        self.layer3 = LinearAdam[
            Self.hidden2_dim, Self.num_actions, Self.dtype
        ]()

    fn forward[
        batch_size: Int
    ](
        mut self,
        obs: InlineArray[Scalar[Self.dtype], batch_size * Self.obs_dim],
    ) -> InlineArray[Scalar[Self.dtype], batch_size * Self.num_actions]:
        """Forward pass: obs -> Q-values for all actions.

        Returns Q(s, a) for each action a.
        """
        # Layer 1: linear + relu
        var h1_pre = self.layer1.forward[batch_size](obs)
        var h1 = relu[batch_size * Self.hidden1_dim, Self.dtype](h1_pre)

        # Layer 2: linear + relu
        var h2_pre = self.layer2.forward[batch_size](h1)
        var h2 = relu[batch_size * Self.hidden2_dim, Self.dtype](h2_pre)

        # Layer 3: linear (no activation - raw Q-values)
        var q_values = self.layer3.forward[batch_size](h2)

        return q_values^

    fn forward_with_cache[
        batch_size: Int
    ](
        mut self,
        obs: InlineArray[Scalar[Self.dtype], batch_size * Self.obs_dim],
        mut h1_out: InlineArray[
            Scalar[Self.dtype], batch_size * Self.hidden1_dim
        ],
        mut h2_out: InlineArray[
            Scalar[Self.dtype], batch_size * Self.hidden2_dim
        ],
    ) -> InlineArray[Scalar[Self.dtype], batch_size * Self.num_actions]:
        """Forward pass with cached activations for backward."""
        var h1_pre = self.layer1.forward[batch_size](obs)
        var h1 = relu[batch_size * Self.hidden1_dim, Self.dtype](h1_pre)

        var h2_pre = self.layer2.forward[batch_size](h1)
        var h2 = relu[batch_size * Self.hidden2_dim, Self.dtype](h2_pre)

        # Store caches
        for i in range(batch_size * Self.hidden1_dim):
            h1_out[i] = h1[i]
        for i in range(batch_size * Self.hidden2_dim):
            h2_out[i] = h2[i]

        var q_values = self.layer3.forward[batch_size](h2)

        return q_values^

    fn backward[
        batch_size: Int
    ](
        mut self,
        dq: InlineArray[Scalar[Self.dtype], batch_size * Self.num_actions],
        obs: InlineArray[Scalar[Self.dtype], batch_size * Self.obs_dim],
        h1: InlineArray[Scalar[Self.dtype], batch_size * Self.hidden1_dim],
        h2: InlineArray[Scalar[Self.dtype], batch_size * Self.hidden2_dim],
    ):
        """Backward pass through Q-Network.

        Args:
            dq: Gradient w.r.t. Q-values.
            obs: Original observations.
            h1, h2: Cached hidden activations.
        """
        # Backward through layer3
        var dh2 = self.layer3.backward[batch_size](dq, h2)

        # Backprop through relu2
        var relu_g2 = relu_grad[batch_size * Self.hidden2_dim, Self.dtype](h2)
        var dh2_pre = elementwise_mul[
            batch_size * Self.hidden2_dim, Self.dtype
        ](dh2, relu_g2)

        # Backward through layer2
        var dh1 = self.layer2.backward[batch_size](dh2_pre, h1)

        # Backprop through relu1
        var relu_g1 = relu_grad[batch_size * Self.hidden1_dim, Self.dtype](h1)
        var dh1_pre = elementwise_mul[
            batch_size * Self.hidden1_dim, Self.dtype
        ](dh1, relu_g1)

        # Backward through layer1
        _ = self.layer1.backward[batch_size](dh1_pre, obs)

    fn update_adam(
        mut self,
        lr: Scalar[Self.dtype] = 0.001,
        beta1: Scalar[Self.dtype] = 0.9,
        beta2: Scalar[Self.dtype] = 0.999,
    ):
        """Update all layers using Adam."""
        self.layer1.update_adam(lr, beta1, beta2)
        self.layer2.update_adam(lr, beta1, beta2)
        self.layer3.update_adam(lr, beta1, beta2)

    fn zero_grad(mut self):
        """Reset all gradients."""
        self.layer1.zero_grad()
        self.layer2.zero_grad()
        self.layer3.zero_grad()

    fn soft_update_from(mut self, source: Self, tau: Scalar[Self.dtype]):
        """Soft update from source network: theta = tau * source + (1-tau) * theta.
        """
        self.layer1.soft_update_from(source.layer1, tau)
        self.layer2.soft_update_from(source.layer2, tau)
        self.layer3.soft_update_from(source.layer3, tau)

    fn copy_from(mut self, source: Self):
        """Hard copy from source network."""
        self.layer1.copy_from(source.layer1)
        self.layer2.copy_from(source.layer2)
        self.layer3.copy_from(source.layer3)

    fn num_parameters(self) -> Int:
        """Total number of parameters."""
        return (
            self.layer1.num_parameters()
            + self.layer2.num_parameters()
            + self.layer3.num_parameters()
        )

    fn print_info(self, name: String = "QNetwork"):
        """Print network architecture."""
        print(name + ":")
        print(
            "  Architecture: "
            + String(Self.obs_dim)
            + " -> "
            + String(Self.hidden1_dim)
            + " (relu)"
            + " -> "
            + String(Self.hidden2_dim)
            + " (relu)"
            + " -> "
            + String(Self.num_actions)
        )
        print("  Parameters: " + String(self.num_parameters()))


# =============================================================================
# Deep DQN Agent
# =============================================================================


struct DeepDQNAgent[
    obs_dim: Int,
    num_actions: Int,
    hidden_dim: Int = 128,
    buffer_capacity: Int = 100000,
    batch_size: Int = 64,
    dtype: DType = DType.float64,
    double_dqn: Bool = True,  # Use Double DQN by default
]:
    """Deep DQN Agent for discrete action spaces using neural networks.

    This agent uses a 2-layer MLP Q-network with compile-time dimensions
    for maximum performance.

    Supports Double DQN (Van Hasselt et al., 2016) which reduces
    overestimation by using the online network to select actions
    and target network to evaluate them.

    Parameters:
        obs_dim: Observation dimension (e.g., 8 for LunarLander).
        num_actions: Number of discrete actions (e.g., 4 for LunarLander).
        hidden_dim: Hidden layer size for Q-network.
        buffer_capacity: Replay buffer capacity.
        batch_size: Training batch size.
        dtype: Data type for computations.
        double_dqn: If True, use Double DQN (recommended).
    """

    # Q-Networks
    var q_network: QNetwork[
        Self.obs_dim,
        Self.num_actions,
        Self.hidden_dim,
        Self.hidden_dim,
        Self.dtype,
    ]
    var target_network: QNetwork[
        Self.obs_dim,
        Self.num_actions,
        Self.hidden_dim,
        Self.hidden_dim,
        Self.dtype,
    ]

    # Replay buffer (action_dim=1 for discrete actions stored as single value)
    var buffer: ReplayBuffer[Self.buffer_capacity, Self.obs_dim, 1, Self.dtype]

    # Hyperparameters
    var gamma: Scalar[Self.dtype]  # Discount factor
    var tau: Scalar[Self.dtype]  # Soft update coefficient
    var lr: Scalar[Self.dtype]  # Learning rate
    var epsilon: Scalar[Self.dtype]  # Exploration rate
    var epsilon_min: Scalar[Self.dtype]  # Minimum epsilon
    var epsilon_decay: Scalar[Self.dtype]  # Epsilon decay per episode

    # Training state
    var total_steps: Int
    var total_episodes: Int

    fn __init__(
        out self,
        gamma: Scalar[Self.dtype] = 0.99,
        tau: Scalar[Self.dtype] = 0.005,
        lr: Scalar[Self.dtype] = 0.0005,
        epsilon: Scalar[Self.dtype] = 1.0,
        epsilon_min: Scalar[Self.dtype] = 0.01,
        epsilon_decay: Scalar[Self.dtype] = 0.995,
    ):
        """Initialize Deep DQN agent.

        Args:
            gamma: Discount factor for future rewards.
            tau: Soft update coefficient for target network.
            lr: Learning rate for Q-network.
            epsilon: Initial exploration rate.
            epsilon_min: Minimum exploration rate.
            epsilon_decay: Epsilon decay rate per episode.
        """
        self.q_network = QNetwork[
            Self.obs_dim,
            Self.num_actions,
            Self.hidden_dim,
            Self.hidden_dim,
            Self.dtype,
        ]()
        self.target_network = QNetwork[
            Self.obs_dim,
            Self.num_actions,
            Self.hidden_dim,
            Self.hidden_dim,
            Self.dtype,
        ]()

        # Initialize target network with same weights
        self.target_network.copy_from(self.q_network)

        self.buffer = ReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, 1, Self.dtype
        ]()

        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.total_steps = 0
        self.total_episodes = 0

    fn select_action(
        mut self,
        obs: InlineArray[Scalar[Self.dtype], Self.obs_dim],
        training: Bool = True,
    ) -> Int:
        """Select action using epsilon-greedy policy.

        Args:
            obs: Current observation.
            training: If True, use epsilon-greedy; if False, use greedy.

        Returns:
            Selected action index.
        """
        # Epsilon-greedy exploration
        if training and random_float64() < Float64(self.epsilon):
            # Random action
            return (
                Int(random_float64() * Float64(Self.num_actions))
                % Self.num_actions
            )

        # Greedy action: argmax Q(s, a)
        var q_values = self.q_network.forward[1](obs)

        # Find argmax
        var best_action = 0
        var best_q = q_values[0]
        for a in range(1, Self.num_actions):
            if q_values[a] > best_q:
                best_q = q_values[a]
                best_action = a

        return best_action

    fn store_transition(
        mut self,
        obs: InlineArray[Scalar[Self.dtype], Self.obs_dim],
        action: Int,
        reward: Scalar[Self.dtype],
        next_obs: InlineArray[Scalar[Self.dtype], Self.obs_dim],
        done: Bool,
    ):
        """Store a transition in the replay buffer."""
        # Store action as single-element array
        var action_arr = InlineArray[Scalar[Self.dtype], 1](fill=0)
        action_arr[0] = Scalar[Self.dtype](action)
        self.buffer.add(obs, action_arr, reward, next_obs, done)
        self.total_steps += 1

    fn train_step(mut self) -> Scalar[Self.dtype]:
        """Perform one training step. Returns loss."""
        if not self.buffer.is_ready[Self.batch_size]():
            return 0.0

        # Sample batch
        var batch_obs = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.obs_dim
        ](fill=0)
        var batch_actions = InlineArray[Scalar[Self.dtype], Self.batch_size](
            fill=0
        )
        var batch_rewards = InlineArray[Scalar[Self.dtype], Self.batch_size](
            fill=0
        )
        var batch_next_obs = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.obs_dim
        ](fill=0)
        var batch_dones = InlineArray[Scalar[Self.dtype], Self.batch_size](
            fill=0
        )

        # Sample from buffer (actions stored with dim=1)
        var batch_actions_arr = InlineArray[
            Scalar[Self.dtype], Self.batch_size
        ](fill=0)
        self.buffer.sample[Self.batch_size](
            batch_obs,
            batch_actions_arr,
            batch_rewards,
            batch_next_obs,
            batch_dones,
        )
        # Copy actions
        for i in range(Self.batch_size):
            batch_actions[i] = batch_actions_arr[i]

        # ========================================
        # Compute target Q-values
        # ========================================

        var max_next_q = InlineArray[Scalar[Self.dtype], Self.batch_size](
            fill=0
        )

        @parameter
        if Self.double_dqn:
            # Double DQN: Use online network to SELECT action, target to EVALUATE
            # This reduces overestimation bias
            var online_next_q = self.q_network.forward[Self.batch_size](
                batch_next_obs
            )
            var target_next_q = self.target_network.forward[Self.batch_size](
                batch_next_obs
            )

            for i in range(Self.batch_size):
                # Select best action using ONLINE network
                var best_action = 0
                var best_online_q = online_next_q[i * Self.num_actions]
                for a in range(1, Self.num_actions):
                    var q = online_next_q[i * Self.num_actions + a]
                    if q > best_online_q:
                        best_online_q = q
                        best_action = a

                # Evaluate that action using TARGET network
                max_next_q[i] = target_next_q[
                    i * Self.num_actions + best_action
                ]
        else:
            # Standard DQN: max_a Q_target(s', a)
            var next_q = self.target_network.forward[Self.batch_size](
                batch_next_obs
            )
            for i in range(Self.batch_size):
                var max_q = next_q[i * Self.num_actions]
                for a in range(1, Self.num_actions):
                    var q = next_q[i * Self.num_actions + a]
                    if q > max_q:
                        max_q = q
                max_next_q[i] = max_q

        # Target: y = r + gamma * (1 - done) * Q(s', a*)
        var target_values = InlineArray[Scalar[Self.dtype], Self.batch_size](
            fill=0
        )
        for i in range(Self.batch_size):
            target_values[i] = (
                batch_rewards[i]
                + self.gamma * (1.0 - batch_dones[i]) * max_next_q[i]
            )

        # ========================================
        # Compute current Q-values with caching
        # ========================================

        var h1_cache = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.hidden_dim
        ](fill=0)
        var h2_cache = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.hidden_dim
        ](fill=0)

        var current_q = self.q_network.forward_with_cache[Self.batch_size](
            batch_obs, h1_cache, h2_cache
        )

        # ========================================
        # Compute loss and gradients
        # ========================================

        # MSE loss on Q(s, a) for taken actions only
        var loss: Scalar[Self.dtype] = 0.0
        var dq = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.num_actions
        ](fill=0)
        var batch_size_scalar = Scalar[Self.dtype](Self.batch_size)

        for i in range(Self.batch_size):
            var action_idx = Int(batch_actions[i])
            var q_idx = i * Self.num_actions + action_idx
            var td_error = current_q[q_idx] - target_values[i]
            loss += td_error * td_error

            # Gradient only flows through the taken action's Q-value
            dq[q_idx] = 2.0 * td_error / batch_size_scalar

        loss /= batch_size_scalar

        # ========================================
        # Backward pass and update
        # ========================================

        self.q_network.zero_grad()
        self.q_network.backward[Self.batch_size](
            dq, batch_obs, h1_cache, h2_cache
        )
        self.q_network.update_adam(self.lr)

        # ========================================
        # Soft update target network
        # ========================================

        self.target_network.soft_update_from(self.q_network, self.tau)

        return loss

    fn decay_epsilon(mut self):
        """Decay exploration rate (call once per episode)."""
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    fn print_info(self):
        """Print agent information."""

        @parameter
        if Self.double_dqn:
            print("Deep Double DQN Agent:")
        else:
            print("Deep DQN Agent:")
        print("  Obs dim: " + String(Self.obs_dim))
        print("  Num actions: " + String(Self.num_actions))
        print("  Hidden dim: " + String(Self.hidden_dim))
        print("  Buffer capacity: " + String(Self.buffer_capacity))
        print("  Batch size: " + String(Self.batch_size))
        print("  Double DQN: " + String(Self.double_dqn))
        print("  Gamma: " + String(self.gamma)[:6])
        print("  Tau: " + String(self.tau)[:6])
        print("  LR: " + String(self.lr)[:8])
        print("  Epsilon: " + String(self.epsilon)[:5])
        print("  Epsilon min: " + String(self.epsilon_min)[:5])
        print("  Epsilon decay: " + String(self.epsilon_decay)[:6])
        self.q_network.print_info("  Q-Network")

    # ========================================================================
    # Training and Evaluation
    # ========================================================================

    fn train[
        E: BoxDiscreteActionEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int,
        max_steps_per_episode: Int = 1000,
        warmup_steps: Int = 1000,
        train_every: Int = 1,
        verbose: Bool = True,
        print_every: Int = 10,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train the Deep DQN agent on a discrete action environment.

        Args:
            env: Environment implementing BoxDiscreteActionEnv trait.
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
            warmup_steps: Steps of random actions before training starts.
            train_every: Train every N steps.
            verbose: Whether to print progress.
            print_every: Print progress every N episodes.
            environment_name: Name for logging.

        Returns:
            TrainingMetrics object with episode history.
        """
        var metrics = TrainingMetrics(
            algorithm_name="Deep DQN",
            environment_name=environment_name,
        )

        if verbose:
            print("=" * 60)
            print("Deep DQN Training on " + environment_name)
            print("=" * 60)
            self.print_info()
            print("-" * 60)

        # Warmup phase: collect random experiences
        if verbose:
            print(
                "Warmup: collecting "
                + String(warmup_steps)
                + " random steps..."
            )

        var warmup_done = 0
        while warmup_done < warmup_steps:
            var obs_list = env.reset_obs_list()
            var done = False

            while not done and warmup_done < warmup_steps:
                # Random action
                var action = (
                    Int(random_float64() * Float64(Self.num_actions))
                    % Self.num_actions
                )

                # Step environment
                var step_result = env.step_obs(action)
                var reward = step_result[1]
                done = step_result[2]

                # Convert observations
                var obs = _list_to_inline[Self.obs_dim, Self.dtype](obs_list)
                var next_obs = _list_to_inline[Self.obs_dim, Self.dtype](
                    step_result[0]
                )

                # Store transition
                self.store_transition(
                    obs, action, Scalar[Self.dtype](reward), next_obs, done
                )

                obs_list = env.get_obs_list()
                warmup_done += 1

        if verbose:
            print("Warmup complete. Buffer size: " + String(self.buffer.len()))
            print("-" * 60)

        # Training loop
        for episode in range(num_episodes):
            var obs_list = env.reset_obs_list()
            var episode_reward: Float64 = 0.0
            var steps = 0
            var done = False

            while not done and steps < max_steps_per_episode:
                # Convert observation
                var obs = _list_to_inline[Self.obs_dim, Self.dtype](obs_list)

                # Select action with epsilon-greedy
                var action = self.select_action(obs, training=True)

                # Step environment
                var step_result = env.step_obs(action)
                var reward = step_result[1]
                done = step_result[2]

                # Get next observation
                var next_obs = _list_to_inline[Self.obs_dim, Self.dtype](
                    step_result[0]
                )

                # Store transition
                self.store_transition(
                    obs, action, Scalar[Self.dtype](reward), next_obs, done
                )

                # Train agent
                if steps % train_every == 0:
                    _ = self.train_step()

                episode_reward += reward
                obs_list = env.get_obs_list()
                steps += 1

            # Log episode metrics
            metrics.log_episode(
                episode, episode_reward, steps, Float64(self.epsilon)
            )
            self.total_episodes += 1

            # Decay exploration
            self.decay_epsilon()

            # Print progress
            if verbose and (episode + 1) % print_every == 0:
                # Compute recent average reward
                var start_idx = max(0, len(metrics.episodes) - print_every)
                var sum_reward: Float64 = 0.0
                for j in range(start_idx, len(metrics.episodes)):
                    sum_reward += metrics.episodes[j].total_reward
                var avg_reward = sum_reward / Float64(
                    len(metrics.episodes) - start_idx
                )
                print(
                    "Episode "
                    + String(episode + 1)
                    + " | Avg Reward: "
                    + String(avg_reward)[:8]
                    + " | Steps: "
                    + String(steps)
                    + " | Epsilon: "
                    + String(self.epsilon)[:5]
                )

        if verbose:
            print("-" * 60)
            print("Training complete!")
            # Compute final average
            var start_idx = max(0, len(metrics.episodes) - 100)
            var sum_reward: Float64 = 0.0
            for j in range(start_idx, len(metrics.episodes)):
                sum_reward += metrics.episodes[j].total_reward
            var final_avg = sum_reward / Float64(
                len(metrics.episodes) - start_idx
            )
            print("Final avg reward (last 100): " + String(final_avg)[:8])

        return metrics^

    fn evaluate[
        E: BoxDiscreteActionEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps_per_episode: Int = 1000,
        verbose: Bool = False,
        render: Bool = False,
    ) -> Float64:
        """Evaluate the trained agent using greedy policy (no exploration).

        Args:
            env: Environment implementing BoxDiscreteActionEnv trait.
            num_episodes: Number of evaluation episodes.
            max_steps_per_episode: Maximum steps per episode.
            verbose: Whether to print per-episode results.
            render: Whether to render the environment.

        Returns:
            Average reward over evaluation episodes.
        """
        var total_reward: Float64 = 0.0

        for ep in range(num_episodes):
            var obs_list = env.reset_obs_list()
            var episode_reward: Float64 = 0.0
            var done = False
            var steps = 0

            while not done and steps < max_steps_per_episode:
                if render:
                    env.render()

                var obs = _list_to_inline[Self.obs_dim, Self.dtype](obs_list)

                # Use greedy action (no exploration)
                var action = self.select_action(obs, training=False)

                var step_result = env.step_obs(action)
                var reward = step_result[1]
                done = step_result[2]

                episode_reward += reward
                obs_list = step_result[0].copy()
                steps += 1

            total_reward += episode_reward

            if verbose:
                print(
                    "  Eval episode "
                    + String(ep + 1)
                    + ": "
                    + String(episode_reward)[:10]
                    + " (steps: "
                    + String(steps)
                    + ")"
                )

        return total_reward / Float64(num_episodes)


# ============================================================================
# Helper functions
# ============================================================================


fn _list_to_inline[
    size: Int, dtype: DType = DType.float64
](obs_list: List[Float64]) -> InlineArray[Scalar[dtype], size]:
    """Convert List[Float64] to InlineArray."""
    var obs = InlineArray[Scalar[dtype], size](fill=0)
    for i in range(size):
        if i < len(obs_list):
            obs[i] = Scalar[dtype](obs_list[i])
    return obs^


fn max(a: Int, b: Int) -> Int:
    """Return maximum of two integers."""
    return a if a > b else b

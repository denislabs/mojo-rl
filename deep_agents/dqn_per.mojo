"""Deep DQN Agent with Prioritized Experience Replay.

This agent extends DQN with PER for better sample efficiency,
especially beneficial for environments with sparse rewards like LunarLander.

Key differences from standard DQN:
- Samples transitions proportionally to TD error magnitude
- Uses importance sampling weights to correct for bias
- Updates priorities after each training step

Reference:
- Schaul et al., "Prioritized Experience Replay" (2015)
- Mnih et al., "Human-level control through deep RL" (2015)

Example usage:
    from deep_agents import DeepDQNPERAgent
    from envs import LunarLanderEnv

    var env = LunarLanderEnv()
    var agent = DeepDQNPERAgent[obs_dim=8, num_actions=4]()
    var metrics = agent.train(env, num_episodes=500)
"""

from random import random_float64

from deep_rl import (
    LinearAdam,
    PrioritizedReplayBuffer,
    relu,
    relu_grad,
    elementwise_mul,
    zeros,
)
from core import TrainingMetrics, BoxDiscreteActionEnv


# Import Q-Network from dqn module
from .dqn import QNetwork


struct DeepDQNPERAgent[
    obs_dim: Int,
    num_actions: Int,
    hidden_dim: Int = 128,
    buffer_capacity: Int = 100000,
    batch_size: Int = 64,
    dtype: DType = DType.float64,
    double_dqn: Bool = True,
]:
    """Deep DQN Agent with Prioritized Experience Replay.

    Uses sum-tree based PER for O(log n) priority sampling.
    Importance sampling weights correct for non-uniform sampling bias.

    Parameters:
        obs_dim: Observation dimension (e.g., 8 for LunarLander).
        num_actions: Number of discrete actions.
        hidden_dim: Hidden layer size for Q-network.
        buffer_capacity: Replay buffer capacity.
        batch_size: Training batch size.
        dtype: Data type for computations.
        double_dqn: If True, use Double DQN.
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

    # Prioritized Replay Buffer
    var buffer: PrioritizedReplayBuffer[Self.buffer_capacity, Self.obs_dim, 1, Self.dtype]

    # Hyperparameters
    var gamma: Scalar[Self.dtype]
    var tau: Scalar[Self.dtype]
    var lr: Scalar[Self.dtype]
    var epsilon: Scalar[Self.dtype]
    var epsilon_min: Scalar[Self.dtype]
    var epsilon_decay: Scalar[Self.dtype]

    # PER hyperparameters
    var beta: Scalar[Self.dtype]  # IS exponent (annealed to 1.0)
    var beta_start: Scalar[Self.dtype]
    var beta_frames: Int  # Frames to anneal beta

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
        alpha: Scalar[Self.dtype] = 0.6,
        beta_start: Scalar[Self.dtype] = 0.4,
        beta_frames: Int = 100000,
    ):
        """Initialize Deep DQN agent with PER.

        Args:
            gamma: Discount factor.
            tau: Soft update coefficient.
            lr: Learning rate.
            epsilon: Initial exploration rate.
            epsilon_min: Minimum exploration rate.
            epsilon_decay: Epsilon decay per episode.
            alpha: Priority exponent (0=uniform, 1=full prioritization).
            beta_start: Initial IS correction exponent.
            beta_frames: Frames to anneal beta from beta_start to 1.0.
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
        self.target_network.copy_from(self.q_network)

        self.buffer = PrioritizedReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, 1, Self.dtype
        ](alpha=alpha, beta=beta_start)

        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames

        self.total_steps = 0
        self.total_episodes = 0

    fn select_action(
        mut self,
        obs: InlineArray[Scalar[Self.dtype], Self.obs_dim],
        training: Bool = True,
    ) -> Int:
        """Select action using epsilon-greedy policy."""
        if training and random_float64() < Float64(self.epsilon):
            return Int(random_float64() * Float64(Self.num_actions)) % Self.num_actions

        var q_values = self.q_network.forward[1](obs)
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
        """Store transition with max priority (will be updated after training)."""
        var action_arr = InlineArray[Scalar[Self.dtype], 1](fill=0)
        action_arr[0] = Scalar[Self.dtype](action)
        self.buffer.add(obs, action_arr, reward, next_obs, done)
        self.total_steps += 1

        # Anneal beta towards 1.0
        var progress = Scalar[Self.dtype](
            _min(Float64(self.total_steps) / Float64(self.beta_frames), 1.0)
        )
        self.beta = self.beta_start + progress * (1.0 - self.beta_start)
        self.buffer.set_beta(self.beta)

    fn train_step(mut self) -> Scalar[Self.dtype]:
        """Perform one training step with PER. Returns loss."""
        if not self.buffer.is_ready[Self.batch_size]():
            return 0.0

        # Sample batch with IS weights
        var batch_obs = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.obs_dim
        ](fill=0)
        var batch_actions_arr = InlineArray[Scalar[Self.dtype], Self.batch_size](fill=0)
        var batch_rewards = InlineArray[Scalar[Self.dtype], Self.batch_size](fill=0)
        var batch_next_obs = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.obs_dim
        ](fill=0)
        var batch_dones = InlineArray[Scalar[Self.dtype], Self.batch_size](fill=0)
        var batch_weights = InlineArray[Scalar[Self.dtype], Self.batch_size](fill=0)
        var batch_indices = InlineArray[Int, Self.batch_size](fill=0)

        self.buffer.sample[Self.batch_size](
            batch_obs,
            batch_actions_arr,
            batch_rewards,
            batch_next_obs,
            batch_dones,
            batch_weights,
            batch_indices,
        )

        # Compute target Q-values
        var max_next_q = InlineArray[Scalar[Self.dtype], Self.batch_size](fill=0)

        @parameter
        if Self.double_dqn:
            var online_next_q = self.q_network.forward[Self.batch_size](batch_next_obs)
            var target_next_q = self.target_network.forward[Self.batch_size](batch_next_obs)

            for i in range(Self.batch_size):
                var best_action = 0
                var best_online_q = online_next_q[i * Self.num_actions]
                for a in range(1, Self.num_actions):
                    var q = online_next_q[i * Self.num_actions + a]
                    if q > best_online_q:
                        best_online_q = q
                        best_action = a
                max_next_q[i] = target_next_q[i * Self.num_actions + best_action]
        else:
            var next_q = self.target_network.forward[Self.batch_size](batch_next_obs)
            for i in range(Self.batch_size):
                var max_q = next_q[i * Self.num_actions]
                for a in range(1, Self.num_actions):
                    var q = next_q[i * Self.num_actions + a]
                    if q > max_q:
                        max_q = q
                max_next_q[i] = max_q

        # Target values
        var target_values = InlineArray[Scalar[Self.dtype], Self.batch_size](fill=0)
        for i in range(Self.batch_size):
            target_values[i] = (
                batch_rewards[i]
                + self.gamma * (1.0 - batch_dones[i]) * max_next_q[i]
            )

        # Forward pass with caching
        var h1_cache = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.hidden_dim
        ](fill=0)
        var h2_cache = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.hidden_dim
        ](fill=0)

        var current_q = self.q_network.forward_with_cache[Self.batch_size](
            batch_obs, h1_cache, h2_cache
        )

        # Compute TD errors and weighted loss
        var loss: Scalar[Self.dtype] = 0.0
        var dq = InlineArray[Scalar[Self.dtype], Self.batch_size * Self.num_actions](
            fill=0
        )
        var td_errors = InlineArray[Scalar[Self.dtype], Self.batch_size](fill=0)

        for i in range(Self.batch_size):
            var action = Int(batch_actions_arr[i])
            var q_pred = current_q[i * Self.num_actions + action]
            var td_error = target_values[i] - q_pred

            # Store TD error for priority update
            td_errors[i] = td_error

            # Weighted MSE loss (IS weight corrects for non-uniform sampling)
            var weighted_td = batch_weights[i] * td_error
            loss += weighted_td * weighted_td

            # Gradient: d(loss)/d(q) = -2 * weight * td_error
            dq[i * Self.num_actions + action] = -2.0 * weighted_td / Scalar[Self.dtype](
                Self.batch_size
            )

        loss = loss / Scalar[Self.dtype](Self.batch_size)

        # Backward pass
        self.q_network.zero_grad()
        self.q_network.backward[Self.batch_size](dq, batch_obs, h1_cache, h2_cache)
        self.q_network.update_adam(self.lr)

        # Update priorities based on TD errors
        self.buffer.update_priorities[Self.batch_size](batch_indices, td_errors)

        # Soft update target network
        self.target_network.soft_update_from(self.q_network, self.tau)

        return loss

    fn decay_epsilon(mut self):
        """Decay epsilon after each episode."""
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min
        self.total_episodes += 1

    fn train[E: BoxDiscreteActionEnv](
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
        """Train the DQN+PER agent on a discrete action environment.

        Args:
            env: Environment implementing BoxDiscreteActionEnv trait.
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
            warmup_steps: Steps of random actions before training.
            train_every: Train every N steps.
            verbose: Whether to print progress.
            print_every: Print progress every N episodes.
            environment_name: Name for logging.

        Returns:
            TrainingMetrics object with episode history.
        """
        var metrics = TrainingMetrics(
            algorithm_name="DQN + PER",
            environment_name=environment_name,
        )

        if verbose:
            print("=" * 60)
            print("DQN + PER Training on " + environment_name)
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
                var action = Int(random_float64() * Float64(Self.num_actions)) % Self.num_actions
                var step_result = env.step_obs(action)
                var reward = step_result[1]
                done = step_result[2]

                var obs = _list_to_inline[Self.obs_dim, Self.dtype](obs_list)
                var next_obs = _list_to_inline[Self.obs_dim, Self.dtype](step_result[0])

                self.store_transition(obs, action, Scalar[Self.dtype](reward), next_obs, done)
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
                var obs = _list_to_inline[Self.obs_dim, Self.dtype](obs_list)
                var action = self.select_action(obs, training=True)
                var step_result = env.step_obs(action)
                var reward = step_result[1]
                done = step_result[2]

                var next_obs = _list_to_inline[Self.obs_dim, Self.dtype](step_result[0])
                self.store_transition(obs, action, Scalar[Self.dtype](reward), next_obs, done)

                if steps % train_every == 0:
                    _ = self.train_step()

                episode_reward += reward
                obs_list = env.get_obs_list()
                steps += 1

            metrics.log_episode(episode, episode_reward, steps, Float64(self.epsilon))
            self.decay_epsilon()

            if verbose and (episode + 1) % print_every == 0:
                var start_idx = _max(0, len(metrics.episodes) - print_every)
                var sum_reward: Float64 = 0.0
                for j in range(start_idx, len(metrics.episodes)):
                    sum_reward += metrics.episodes[j].total_reward
                var avg_reward = sum_reward / Float64(len(metrics.episodes) - start_idx)
                print(
                    "Episode "
                    + String(episode + 1)
                    + " | Avg Reward: "
                    + String(avg_reward)[:8]
                    + " | Steps: "
                    + String(steps)
                    + " | Epsilon: "
                    + String(self.epsilon)[:5]
                    + " | Beta: "
                    + String(self.beta)[:5]
                )

        if verbose:
            print("-" * 60)
            print("Training complete!")
            var start_idx = _max(0, len(metrics.episodes) - 100)
            var sum_reward: Float64 = 0.0
            for j in range(start_idx, len(metrics.episodes)):
                sum_reward += metrics.episodes[j].total_reward
            var final_avg = sum_reward / Float64(len(metrics.episodes) - start_idx)
            print("Final avg reward (last 100): " + String(final_avg)[:8])

        return metrics^

    fn evaluate[E: BoxDiscreteActionEnv](
        mut self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps_per_episode: Int = 1000,
        verbose: Bool = False,
        render: Bool = False,
    ) -> Float64:
        """Evaluate the trained agent using greedy policy."""
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

    fn print_info(self):
        """Print agent configuration."""
        @parameter
        if Self.double_dqn:
            print("Deep Double DQN + PER Agent:")
        else:
            print("Deep DQN + PER Agent:")
        print("  Obs dim: " + String(Self.obs_dim))
        print("  Num actions: " + String(Self.num_actions))
        print("  Hidden dim: " + String(Self.hidden_dim))
        print("  Buffer capacity: " + String(Self.buffer_capacity))
        print("  Batch size: " + String(Self.batch_size))
        print("  Gamma: " + String(self.gamma)[:6])
        print("  Tau: " + String(self.tau)[:6])
        print("  LR: " + String(self.lr)[:8])
        print("  Epsilon: " + String(self.epsilon)[:5])
        print("  Beta (IS): " + String(self.beta)[:5])
        self.q_network.print_info("  Q-Network")


# Helper functions
fn _list_to_inline[
    size: Int, dtype: DType = DType.float64
](obs_list: List[Float64]) -> InlineArray[Scalar[dtype], size]:
    """Convert List[Float64] to InlineArray."""
    var obs = InlineArray[Scalar[dtype], size](fill=0)
    for i in range(size):
        if i < len(obs_list):
            obs[i] = Scalar[dtype](obs_list[i])
    return obs^


fn _max(a: Int, b: Int) -> Int:
    return a if a > b else b


fn _min(a: Float64, b: Float64) -> Float64:
    return a if a < b else b

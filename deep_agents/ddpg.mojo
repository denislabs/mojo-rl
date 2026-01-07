"""Deep Deterministic Policy Gradient (DDPG) Agent with Neural Networks.

DDPG is an off-policy actor-critic algorithm for continuous control.
It uses:
- Deterministic actor: obs -> action (2-layer MLP with tanh output)
- Q-function critic: (obs, action) -> Q-value (2-layer MLP)
- Target networks with soft updates
- Experience replay
- Gaussian exploration noise

This is the DEEP version using neural networks with compile-time dimensions.
For linear function approximation, see agents/ddpg.mojo (DDPGAgent).

Reference: Lillicrap et al. "Continuous control with deep reinforcement learning"

Example usage:
    from deeprl import DeepDDPGAgent
    from envs import PendulumEnv

    var env = PendulumEnv()
    var agent = DeepDDPGAgent[obs_dim=3, action_dim=1, hidden_dim=128](
        action_scale=2.0,  # Pendulum actions in [-2, 2]
    )

    # Simple training
    var metrics = agent.train(env, num_episodes=200)
    var eval_reward = agent.evaluate(env)
"""

from random import random_float64
from math import sqrt, log

from deep_rl import (
    Actor,
    Critic,
    ReplayBuffer,
    scale,
    elementwise_sub,
    elementwise_mul,
)
from core import TrainingMetrics, BoxContinuousActionEnv


fn gaussian_noise() -> Float64:
    """Generate Gaussian noise using Box-Muller transform."""
    var u1 = random_float64()
    var u2 = random_float64()
    # Avoid log(0)
    if u1 < 1e-10:
        u1 = 1e-10
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265359 * u2)


fn cos(x: Float64) -> Float64:
    """Cosine approximation using Taylor series."""
    var x2 = x * x
    var x4 = x2 * x2
    var x6 = x4 * x2
    return 1.0 - x2 / 2.0 + x4 / 24.0 - x6 / 720.0


struct DeepDDPGAgent[
    obs_dim: Int,
    action_dim: Int,
    hidden_dim: Int = 256,
    buffer_capacity: Int = 100000,
    batch_size: Int = 64,
    dtype: DType = DType.float64,
]:
    """Deep DDPG Agent for continuous control using neural networks.

    This agent uses 2-layer MLPs for both actor and critic networks,
    with compile-time dimensions for maximum performance.

    Parameters:
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        hidden_dim: Hidden layer size for networks.
        buffer_capacity: Replay buffer capacity.
        batch_size: Training batch size.
        dtype: Data type for computations.
    """

    # Networks
    var actor: Actor[
        Self.obs_dim,
        Self.action_dim,
        Self.hidden_dim,
        Self.hidden_dim,
        Self.dtype,
    ]
    var actor_target: Actor[
        Self.obs_dim,
        Self.action_dim,
        Self.hidden_dim,
        Self.hidden_dim,
        Self.dtype,
    ]
    var critic: Critic[
        Self.obs_dim,
        Self.action_dim,
        Self.hidden_dim,
        Self.hidden_dim,
        Self.dtype,
    ]
    var critic_target: Critic[
        Self.obs_dim,
        Self.action_dim,
        Self.hidden_dim,
        Self.hidden_dim,
        Self.dtype,
    ]

    # Replay buffer
    var buffer: ReplayBuffer[
        Self.buffer_capacity, Self.obs_dim, Self.action_dim, Self.dtype
    ]

    # Hyperparameters
    var gamma: Scalar[Self.dtype]  # Discount factor
    var tau: Scalar[Self.dtype]  # Soft update coefficient
    var actor_lr: Scalar[Self.dtype]  # Actor learning rate
    var critic_lr: Scalar[Self.dtype]  # Critic learning rate
    var noise_std: Scalar[Self.dtype]  # Exploration noise std
    var noise_std_min: Scalar[Self.dtype]  # Minimum noise after decay
    var noise_decay: Scalar[Self.dtype]  # Noise decay rate per episode
    var action_scale: Scalar[Self.dtype]  # Action scaling

    # Training state
    var total_steps: Int
    var total_episodes: Int

    fn __init__(
        out self,
        gamma: Scalar[Self.dtype] = 0.99,
        tau: Scalar[Self.dtype] = 0.005,
        actor_lr: Scalar[Self.dtype] = 0.001,
        critic_lr: Scalar[Self.dtype] = 0.001,
        noise_std: Scalar[Self.dtype] = 0.1,
        noise_std_min: Scalar[Self.dtype] = 0.01,
        noise_decay: Scalar[Self.dtype] = 0.995,
        action_scale: Scalar[Self.dtype] = 1.0,
    ):
        """Initialize Deep DDPG agent.

        Args:
            gamma: Discount factor for future rewards.
            tau: Soft update coefficient for target networks.
            actor_lr: Learning rate for actor network.
            critic_lr: Learning rate for critic network.
            noise_std: Initial standard deviation of exploration noise.
            noise_std_min: Minimum noise after decay.
            noise_decay: Noise decay rate per episode.
            action_scale: Maximum action magnitude.
        """
        self.actor = Actor[
            Self.obs_dim,
            Self.action_dim,
            Self.hidden_dim,
            Self.hidden_dim,
            Self.dtype,
        ](
            action_scale=action_scale,
            action_bias=0.0,
        )
        self.actor_target = Actor[
            Self.obs_dim,
            Self.action_dim,
            Self.hidden_dim,
            Self.hidden_dim,
            Self.dtype,
        ](
            action_scale=action_scale,
            action_bias=0.0,
        )
        self.critic = Critic[
            Self.obs_dim,
            Self.action_dim,
            Self.hidden_dim,
            Self.hidden_dim,
            Self.dtype,
        ]()
        self.critic_target = Critic[
            Self.obs_dim,
            Self.action_dim,
            Self.hidden_dim,
            Self.hidden_dim,
            Self.dtype,
        ]()

        # Initialize target networks with same weights
        self.actor_target.copy_from(self.actor)
        self.critic_target.copy_from(self.critic)

        self.buffer = ReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, Self.action_dim, Self.dtype
        ]()

        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.noise_std = noise_std
        self.noise_std_min = noise_std_min
        self.noise_decay = noise_decay
        self.action_scale = action_scale
        self.total_steps = 0
        self.total_episodes = 0

    fn select_action(
        mut self,
        obs: InlineArray[Scalar[Self.dtype], Self.obs_dim],
        add_noise: Bool = True,
    ) -> InlineArray[Scalar[Self.dtype], Self.action_dim]:
        """Select action using actor network with optional exploration noise."""
        # Reshape obs to batch of 1
        var obs_batch = InlineArray[Scalar[Self.dtype], Self.obs_dim](fill=0)
        for i in range(Self.obs_dim):
            obs_batch[i] = obs[i]

        # Get action from actor
        var action_batch = self.actor.forward[1](obs_batch)

        # Extract action and add noise
        var action = InlineArray[Scalar[Self.dtype], Self.action_dim](fill=0)
        for i in range(Self.action_dim):
            var a = action_batch[i]
            if add_noise:
                var noise = Scalar[Self.dtype](gaussian_noise())
                a += self.noise_std * noise
            # Clip to action bounds
            if a > self.action_scale:
                a = self.action_scale
            elif a < -self.action_scale:
                a = -self.action_scale
            action[i] = a

        return action^

    fn store_transition(
        mut self,
        obs: InlineArray[Scalar[Self.dtype], Self.obs_dim],
        action: InlineArray[Scalar[Self.dtype], Self.action_dim],
        reward: Scalar[Self.dtype],
        next_obs: InlineArray[Scalar[Self.dtype], Self.obs_dim],
        done: Bool,
    ):
        """Store a transition in the replay buffer."""
        self.buffer.add(obs, action, reward, next_obs, done)
        self.total_steps += 1

    fn train_step(mut self) -> Scalar[Self.dtype]:
        """Perform one training step. Returns critic loss."""
        if not self.buffer.is_ready[Self.batch_size]():
            return 0.0

        # Sample batch
        var batch_obs = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.obs_dim
        ](fill=0)
        var batch_actions = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.action_dim
        ](fill=0)
        var batch_rewards = InlineArray[Scalar[Self.dtype], Self.batch_size](
            fill=0
        )
        var batch_next_obs = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.obs_dim
        ](fill=0)
        var batch_dones = InlineArray[Scalar[Self.dtype], Self.batch_size](
            fill=0
        )

        self.buffer.sample[Self.batch_size](
            batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones
        )

        # ========================================
        # Critic update
        # ========================================

        # Compute target Q-values: y = r + gamma * (1 - done) * Q_target(s', pi_target(s'))
        var next_actions = self.actor_target.forward[Self.batch_size](
            batch_next_obs
        )
        var target_q = self.critic_target.forward[Self.batch_size](
            batch_next_obs, next_actions
        )

        var target_values = InlineArray[Scalar[Self.dtype], Self.batch_size](
            fill=0
        )
        for i in range(Self.batch_size):
            target_values[i] = (
                batch_rewards[i]
                + self.gamma * (1.0 - batch_dones[i]) * target_q[i]
            )

        # Current Q-values with caching for backward
        var x_cache = InlineArray[
            Scalar[Self.dtype],
            Self.batch_size * (Self.obs_dim + Self.action_dim),
        ](fill=0)
        var h1_cache = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.hidden_dim
        ](fill=0)
        var h2_cache = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.hidden_dim
        ](fill=0)

        var current_q = self.critic.forward_with_cache[Self.batch_size](
            batch_obs, batch_actions, x_cache, h1_cache, h2_cache
        )

        # Critic loss: MSE = mean((Q - target)^2)
        var critic_loss: Scalar[Self.dtype] = 0.0
        var dq = InlineArray[Scalar[Self.dtype], Self.batch_size](fill=0)
        var batch_size_scalar = Scalar[Self.dtype](Self.batch_size)
        for i in range(Self.batch_size):
            var td_error = current_q[i] - target_values[i]
            critic_loss += td_error * td_error
            dq[i] = 2.0 * td_error / batch_size_scalar  # Gradient of MSE

        critic_loss /= batch_size_scalar

        # Critic backward and update
        self.critic.zero_grad()
        _ = self.critic.backward[Self.batch_size](
            dq, x_cache, h1_cache, h2_cache
        )
        self.critic.update_adam(self.critic_lr)

        # ========================================
        # Actor update
        # ========================================

        # Get actions from current actor with caching
        var h1_actor = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.hidden_dim
        ](fill=0)
        var h2_actor = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.hidden_dim
        ](fill=0)
        var out_tanh = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.action_dim
        ](fill=0)

        var actor_actions = self.actor.forward_with_cache[Self.batch_size](
            batch_obs, h1_actor, h2_actor, out_tanh
        )

        # Get Q-values for actor's actions with caching
        var x_cache2 = InlineArray[
            Scalar[Self.dtype],
            Self.batch_size * (Self.obs_dim + Self.action_dim),
        ](fill=0)
        var h1_cache2 = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.hidden_dim
        ](fill=0)
        var h2_cache2 = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.hidden_dim
        ](fill=0)

        _ = self.critic.forward_with_cache[Self.batch_size](
            batch_obs, actor_actions, x_cache2, h1_cache2, h2_cache2
        )

        # Actor gradient: maximize Q -> dQ/daction = -1 (gradient ascent)
        var dq_actor = InlineArray[Scalar[Self.dtype], Self.batch_size](fill=0)
        var neg_one_over_batch = Scalar[Self.dtype](-1.0) / batch_size_scalar
        for i in range(Self.batch_size):
            dq_actor[i] = neg_one_over_batch  # Negative for gradient ascent

        # Get gradient w.r.t. actions from critic
        var dactions = self.critic.backward[Self.batch_size](
            dq_actor, x_cache2, h1_cache2, h2_cache2
        )

        # Backward through actor
        self.actor.zero_grad()
        self.actor.backward[Self.batch_size](
            dactions, batch_obs, h1_actor, h2_actor, out_tanh
        )
        self.actor.update_adam(self.actor_lr)

        # ========================================
        # Soft update target networks
        # ========================================

        self.actor_target.soft_update_from(self.actor, self.tau)
        self.critic_target.soft_update_from(self.critic, self.tau)

        return critic_loss

    fn print_info(self):
        """Print agent information."""
        print("Deep DDPG Agent:")
        print("  Obs dim: " + String(Self.obs_dim))
        print("  Action dim: " + String(Self.action_dim))
        print("  Hidden dim: " + String(Self.hidden_dim))
        print("  Buffer capacity: " + String(Self.buffer_capacity))
        print("  Batch size: " + String(Self.batch_size))
        print("  Gamma: " + String(self.gamma)[:6])
        print("  Tau: " + String(self.tau)[:6])
        print("  Actor LR: " + String(self.actor_lr)[:8])
        print("  Critic LR: " + String(self.critic_lr)[:8])
        print("  Noise std: " + String(self.noise_std)[:6])
        print("  Noise decay: " + String(self.noise_decay)[:6])
        self.actor.print_info("  Actor")
        self.critic.print_info("  Critic")

    fn decay_noise(mut self):
        """Decay exploration noise (call once per episode)."""
        self.noise_std *= self.noise_decay
        if self.noise_std < self.noise_std_min:
            self.noise_std = self.noise_std_min

    # ========================================================================
    # Training and Evaluation
    # ========================================================================

    fn train[
        E: BoxContinuousActionEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int,
        max_steps_per_episode: Int = 200,
        warmup_steps: Int = 1000,
        train_every: Int = 1,
        verbose: Bool = True,
        print_every: Int = 10,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train the Deep DDPG agent on a continuous control environment.

        Args:
            env: Environment implementing BoxContinuousActionEnv trait.
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
            algorithm_name="Deep DDPG",
            environment_name=environment_name,
        )

        if verbose:
            print("=" * 60)
            print("Deep DDPG Training on " + environment_name)
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
                var random_action = InlineArray[
                    Scalar[Self.dtype], Self.action_dim
                ](fill=0)
                for i in range(Self.action_dim):
                    var rand_val = Scalar[Self.dtype](
                        random_float64() * 2.0 - 1.0
                    )
                    random_action[i] = rand_val * self.action_scale

                # Step environment
                var step_result = env.step_continuous(Float64(random_action[0]))
                var reward = step_result[1]
                done = step_result[2]

                # Convert observations
                var obs = _list_to_inline[Self.obs_dim, Self.dtype](obs_list)
                var next_obs_list = env.get_obs_list()
                var next_obs = _list_to_inline[Self.obs_dim, Self.dtype](
                    next_obs_list
                )

                # Store transition
                self.store_transition(
                    obs,
                    random_action,
                    Scalar[Self.dtype](reward),
                    next_obs,
                    done,
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

                # Select action with exploration noise
                var action = self.select_action(obs, add_noise=True)

                # Step environment (extract first action for single-action envs)
                var step_result = env.step_continuous(Float64(action[0]))
                var reward = step_result[1]
                done = step_result[2]

                # Get next observation
                var next_obs_list = env.get_obs_list()
                var next_obs = _list_to_inline[Self.obs_dim, Self.dtype](
                    next_obs_list
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
                episode, episode_reward, steps, Float64(self.noise_std)
            )
            self.total_episodes += 1

            # Decay exploration noise
            self.decay_noise()

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
                    + " | Noise: "
                    + String(self.noise_std)[:5]
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
        E: BoxContinuousActionEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps_per_episode: Int = 200,
        verbose: Bool = False,
    ) -> Float64:
        """Evaluate the trained agent using deterministic policy (no noise).

        Args:
            env: Environment implementing BoxContinuousActionEnv trait.
            num_episodes: Number of evaluation episodes.
            max_steps_per_episode: Maximum steps per episode.
            verbose: Whether to print per-episode results.

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
                var obs = _list_to_inline[Self.obs_dim, Self.dtype](obs_list)

                # Use deterministic action (no noise)
                var action = self.select_action(obs, add_noise=False)

                var step_result = env.step_continuous(Float64(action[0]))
                var reward = step_result[1]
                done = step_result[2]

                episode_reward += reward
                obs_list = env.get_obs_list()
                steps += 1

            total_reward += episode_reward

            if verbose:
                print(
                    "  Eval episode "
                    + String(ep + 1)
                    + ": "
                    + String(episode_reward)[:10]
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

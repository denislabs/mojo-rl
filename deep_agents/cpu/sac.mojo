"""Deep Soft Actor-Critic (SAC) with Neural Networks.

SAC is a maximum entropy reinforcement learning algorithm that learns a
stochastic policy by maximizing both expected return and policy entropy.
This encourages exploration and leads to more robust policies.

Key components:
1. Stochastic actor: Gaussian policy with learned mean and log_std
2. Twin Q-networks: Like TD3, uses min(Q1, Q2) to reduce overestimation
3. Entropy bonus: Policy trained to maximize reward + alpha * entropy
4. Automatic temperature: alpha can be learned to maintain target entropy
5. No target actor: Uses current policy for next action sampling

This is the DEEP version using neural networks with compile-time dimensions.
For linear function approximation, see agents/sac.mojo (SACAgent).

References:
- Haarnoja et al. (2018): "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
- Haarnoja et al. (2018): "Soft Actor-Critic Algorithms and Applications"

Example usage:
    from deep_agents.cpu import DeepSACAgent
    from envs import PendulumEnv

    var env = PendulumEnv()
    var agent = DeepSACAgent[obs_dim=3, action_dim=1, hidden_dim=256](
        action_scale=2.0,  # Pendulum actions in [-2, 2]
        alpha=0.2,         # Entropy coefficient
        auto_alpha=True,   # Automatically tune alpha
    )

    # Simple training
    var metrics = agent.train(env, num_episodes=200)
    var eval_reward = agent.evaluate(env)
"""

from random import random_float64
from math import sqrt, log, exp

from deep_rl.cpu import (
    StochasticActor,
    Critic,
    ReplayBuffer,
    scale,
    elementwise_sub,
    elementwise_mul,
)
from core import TrainingMetrics, BoxContinuousActionEnv


@always_inline
fn _gaussian_noise_sac() -> Float64:
    """Generate Gaussian noise using Box-Muller transform."""
    var u1 = random_float64()
    var u2 = random_float64()
    # Avoid log(0)
    if u1 < 1e-10:
        u1 = 1e-10
    return sqrt(-2.0 * log(u1)) * _cos_sac(2.0 * 3.14159265359 * u2)


@always_inline
fn _cos_sac(x: Float64) -> Float64:
    """Cosine approximation using Taylor series."""
    var x2 = x * x
    var x4 = x2 * x2
    var x6 = x4 * x2
    return 1.0 - x2 / 2.0 + x4 / 24.0 - x6 / 720.0


struct DeepSACAgent[
    obs_dim: Int,
    action_dim: Int,
    hidden_dim: Int = 256,
    buffer_capacity: Int = 100000,
    batch_size: Int = 64,
    dtype: DType = DType.float64,
]:
    """Deep SAC Agent for continuous control using neural networks.

    This agent uses a stochastic actor (Gaussian policy with learned mean/std)
    and twin critics, with compile-time dimensions for maximum performance.

    Key SAC features:
    1. Maximum entropy objective: maximize reward + alpha * entropy
    2. Stochastic policy for better exploration
    3. Twin critics to reduce overestimation (like TD3)
    4. Optional automatic entropy temperature tuning
    5. No target actor network (unlike DDPG/TD3)

    Parameters:
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        hidden_dim: Hidden layer size for networks.
        buffer_capacity: Replay buffer capacity.
        batch_size: Training batch size.
        dtype: Data type for computations.
    """

    # Stochastic Actor network (no target - SAC uses current policy)
    var actor: StochasticActor[
        Self.obs_dim,
        Self.action_dim,
        Self.hidden_dim,
        Self.hidden_dim,
        Self.dtype,
    ]

    # Twin Critic networks (like TD3)
    var critic1: Critic[
        Self.obs_dim,
        Self.action_dim,
        Self.hidden_dim,
        Self.hidden_dim,
        Self.dtype,
    ]
    var critic1_target: Critic[
        Self.obs_dim,
        Self.action_dim,
        Self.hidden_dim,
        Self.hidden_dim,
        Self.dtype,
    ]
    var critic2: Critic[
        Self.obs_dim,
        Self.action_dim,
        Self.hidden_dim,
        Self.hidden_dim,
        Self.dtype,
    ]
    var critic2_target: Critic[
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
    var action_scale: Scalar[Self.dtype]  # Action scaling

    # SAC-specific hyperparameters
    var alpha: Scalar[Self.dtype]  # Entropy coefficient (temperature)
    var log_alpha: Scalar[Self.dtype]  # Log of alpha for gradient updates
    var target_entropy: Scalar[Self.dtype]  # Target entropy for auto-tuning
    var alpha_lr: Scalar[Self.dtype]  # Learning rate for alpha
    var auto_alpha: Bool  # Whether to automatically tune alpha

    # Training state
    var total_steps: Int
    var total_episodes: Int

    fn __init__(
        out self,
        gamma: Scalar[Self.dtype] = 0.99,
        tau: Scalar[Self.dtype] = 0.005,
        actor_lr: Scalar[Self.dtype] = 0.0003,
        critic_lr: Scalar[Self.dtype] = 0.0003,
        action_scale: Scalar[Self.dtype] = 1.0,
        alpha: Scalar[Self.dtype] = 0.2,
        auto_alpha: Bool = True,
        alpha_lr: Scalar[Self.dtype] = 0.0003,
        target_entropy: Scalar[Self.dtype] = -1.0,  # -dim(A) is common default
    ):
        """Initialize Deep SAC agent.

        Args:
            gamma: Discount factor for future rewards.
            tau: Soft update coefficient for target networks.
            actor_lr: Learning rate for actor network.
            critic_lr: Learning rate for critic networks.
            action_scale: Maximum action magnitude.
            alpha: Initial entropy coefficient (temperature).
            auto_alpha: Whether to automatically tune alpha.
            alpha_lr: Learning rate for alpha tuning.
            target_entropy: Target entropy for automatic tuning (default: -action_dim).
        """
        # Initialize stochastic actor
        self.actor = StochasticActor[
            Self.obs_dim,
            Self.action_dim,
            Self.hidden_dim,
            Self.hidden_dim,
            Self.dtype,
        ](
            action_scale=action_scale,
            action_bias=0.0,
        )

        # Initialize twin critics and targets
        self.critic1 = Critic[
            Self.obs_dim,
            Self.action_dim,
            Self.hidden_dim,
            Self.hidden_dim,
            Self.dtype,
        ]()
        self.critic1_target = Critic[
            Self.obs_dim,
            Self.action_dim,
            Self.hidden_dim,
            Self.hidden_dim,
            Self.dtype,
        ]()
        self.critic2 = Critic[
            Self.obs_dim,
            Self.action_dim,
            Self.hidden_dim,
            Self.hidden_dim,
            Self.dtype,
        ]()
        self.critic2_target = Critic[
            Self.obs_dim,
            Self.action_dim,
            Self.hidden_dim,
            Self.hidden_dim,
            Self.dtype,
        ]()

        # Initialize target networks with same weights
        self.critic1_target.copy_from(self.critic1)
        self.critic2_target.copy_from(self.critic2)

        self.buffer = ReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, Self.action_dim, Self.dtype
        ]()

        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.action_scale = action_scale

        # SAC entropy settings
        self.alpha = alpha
        self.log_alpha = Scalar[Self.dtype](log(Float64(alpha)))
        self.target_entropy = target_entropy
        self.alpha_lr = alpha_lr
        self.auto_alpha = auto_alpha

        self.total_steps = 0
        self.total_episodes = 0

    fn select_action(
        mut self,
        obs: InlineArray[Scalar[Self.dtype], Self.obs_dim],
        deterministic: Bool = False,
    ) -> InlineArray[Scalar[Self.dtype], Self.action_dim]:
        """Select action using stochastic policy.

        Args:
            obs: Observation array.
            deterministic: If True, use mean action (for evaluation).

        Returns:
            Action array.
        """
        # Generate noise for sampling
        var noise = InlineArray[Scalar[Self.dtype], Self.action_dim](fill=0)
        if not deterministic:
            for i in range(Self.action_dim):
                noise[i] = Scalar[Self.dtype](_gaussian_noise_sac())

        # Sample action from stochastic policy
        var actions_logprobs = self.actor.sample_action[1](obs, noise)
        var action = actions_logprobs[0]

        # Clip to action bounds
        var clipped_action = InlineArray[Scalar[Self.dtype], Self.action_dim](
            fill=0
        )
        for i in range(Self.action_dim):
            var a = action[i]
            if a > self.action_scale:
                a = self.action_scale
            elif a < -self.action_scale:
                a = -self.action_scale
            clipped_action[i] = a

        return clipped_action^

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
        """Perform one training step. Returns critic loss.

        SAC update procedure:
        1. Update both critics using soft Bellman residual
        2. Update actor to maximize Q - alpha * log_prob
        3. Update alpha to maintain target entropy (if auto_alpha)
        4. Soft update target critic networks
        """
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

        # Sample next actions from current policy (no target actor in SAC)
        var next_noise = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.action_dim
        ](fill=0)
        for i in range(Self.batch_size * Self.action_dim):
            next_noise[i] = Scalar[Self.dtype](_gaussian_noise_sac())

        var next_actions_logprobs = self.actor.sample_action[Self.batch_size](
            batch_next_obs, next_noise
        )
        var next_actions = next_actions_logprobs[0]
        var next_log_probs = next_actions_logprobs[1]

        # Compute sum of log probs per sample
        var next_log_prob_sum = InlineArray[
            Scalar[Self.dtype], Self.batch_size
        ](fill=0)
        for i in range(Self.batch_size):
            var sum_log_prob: Scalar[Self.dtype] = 0.0
            for j in range(Self.action_dim):
                sum_log_prob += next_log_probs[i * Self.action_dim + j]
            next_log_prob_sum[i] = sum_log_prob

        # Get Q-values from both target critics
        var target_q1 = self.critic1_target.forward[Self.batch_size](
            batch_next_obs, next_actions
        )
        var target_q2 = self.critic2_target.forward[Self.batch_size](
            batch_next_obs, next_actions
        )

        # Compute soft Bellman target using min of Q1, Q2 and entropy bonus
        var target_values = InlineArray[Scalar[Self.dtype], Self.batch_size](
            fill=0
        )
        for i in range(Self.batch_size):
            var min_q = target_q1[i]
            if target_q2[i] < min_q:
                min_q = target_q2[i]
            # Soft target: y = r + gamma * (min_Q - alpha * log_prob)
            target_values[i] = batch_rewards[i] + self.gamma * (
                1.0 - batch_dones[i]
            ) * (min_q - self.alpha * next_log_prob_sum[i])

        # Update critic 1
        var x_cache1 = InlineArray[
            Scalar[Self.dtype],
            Self.batch_size * (Self.obs_dim + Self.action_dim),
        ](fill=0)
        var h1_cache1 = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.hidden_dim
        ](fill=0)
        var h2_cache1 = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.hidden_dim
        ](fill=0)

        var current_q1 = self.critic1.forward_with_cache[Self.batch_size](
            batch_obs, batch_actions, x_cache1, h1_cache1, h2_cache1
        )

        var critic1_loss: Scalar[Self.dtype] = 0.0
        var dq1 = InlineArray[Scalar[Self.dtype], Self.batch_size](fill=0)
        var batch_size_scalar = Scalar[Self.dtype](Self.batch_size)
        for i in range(Self.batch_size):
            var td_error = current_q1[i] - target_values[i]
            critic1_loss += td_error * td_error
            dq1[i] = 2.0 * td_error / batch_size_scalar

        self.critic1.zero_grad()
        _ = self.critic1.backward[Self.batch_size](
            dq1, x_cache1, h1_cache1, h2_cache1
        )
        self.critic1.update_adam(self.critic_lr)

        # Update critic 2
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

        var current_q2 = self.critic2.forward_with_cache[Self.batch_size](
            batch_obs, batch_actions, x_cache2, h1_cache2, h2_cache2
        )

        var critic2_loss: Scalar[Self.dtype] = 0.0
        var dq2 = InlineArray[Scalar[Self.dtype], Self.batch_size](fill=0)
        for i in range(Self.batch_size):
            var td_error = current_q2[i] - target_values[i]
            critic2_loss += td_error * td_error
            dq2[i] = 2.0 * td_error / batch_size_scalar

        self.critic2.zero_grad()
        _ = self.critic2.backward[Self.batch_size](
            dq2, x_cache2, h1_cache2, h2_cache2
        )
        self.critic2.update_adam(self.critic_lr)

        var total_critic_loss = (critic1_loss + critic2_loss) / (
            2.0 * batch_size_scalar
        )

        # ========================================
        # Actor update
        # ========================================

        # Sample new actions with caching for backward
        var actor_noise = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.action_dim
        ](fill=0)
        for i in range(Self.batch_size * Self.action_dim):
            actor_noise[i] = Scalar[Self.dtype](_gaussian_noise_sac())

        var h1_actor = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.hidden_dim
        ](fill=0)
        var h2_actor = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.hidden_dim
        ](fill=0)
        var mean_actor = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.action_dim
        ](fill=0)
        var log_std_actor = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.action_dim
        ](fill=0)
        var tanh_actor = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.action_dim
        ](fill=0)

        var actor_actions_logprobs = self.actor.sample_action_with_cache[
            Self.batch_size
        ](
            batch_obs,
            actor_noise,
            h1_actor,
            h2_actor,
            mean_actor,
            log_std_actor,
            tanh_actor,
        )
        var actor_actions = actor_actions_logprobs[0]
        var actor_log_probs = actor_actions_logprobs[1]

        # Compute log prob sum for each sample
        var actor_log_prob_sum = InlineArray[
            Scalar[Self.dtype], Self.batch_size
        ](fill=0)
        for i in range(Self.batch_size):
            var sum_log_prob: Scalar[Self.dtype] = 0.0
            for j in range(Self.action_dim):
                sum_log_prob += actor_log_probs[i * Self.action_dim + j]
            actor_log_prob_sum[i] = sum_log_prob

        # Get Q-values for actor's actions from critic1
        var x_cache_actor = InlineArray[
            Scalar[Self.dtype],
            Self.batch_size * (Self.obs_dim + Self.action_dim),
        ](fill=0)
        var h1_cache_actor = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.hidden_dim
        ](fill=0)
        var h2_cache_actor = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.hidden_dim
        ](fill=0)

        _ = self.critic1.forward_with_cache[Self.batch_size](
            batch_obs,
            actor_actions,
            x_cache_actor,
            h1_cache_actor,
            h2_cache_actor,
        )

        # Actor loss: min (alpha * log_prob - Q(s, a))
        # Gradient for actor: ∂/∂θ (α * log π - Q) = α * ∂log π/∂θ - ∂Q/∂a * ∂a/∂θ
        # We combine these into a single gradient for actions

        # Get gradient w.r.t. actions from critic1: ∂Q/∂a
        var dq_actor = InlineArray[Scalar[Self.dtype], Self.batch_size](fill=0)
        var neg_one_over_batch = Scalar[Self.dtype](-1.0) / batch_size_scalar
        for i in range(Self.batch_size):
            dq_actor[i] = neg_one_over_batch

        var dactions = self.critic1.backward[Self.batch_size](
            dq_actor, x_cache_actor, h1_cache_actor, h2_cache_actor
        )

        # Combine with entropy gradient: d_loss = -∂Q/∂a + α * ∂log_π/∂a
        # For reparameterization, we need: ∂(Q - α*log_π)/∂θ
        # The log_prob gradient through action is handled in actor backward
        var d_actor_loss = InlineArray[
            Scalar[Self.dtype], Self.batch_size * Self.action_dim
        ](fill=0)

        for i in range(Self.batch_size):
            for j in range(Self.action_dim):
                var idx = i * Self.action_dim + j
                # Gradient for maximizing Q - α * log_π
                # = ∂Q/∂a - α * ∂log_π/∂a (through a)
                d_actor_loss[idx] = dactions[idx]

        # Backward through actor
        self.actor.zero_grad()
        self.actor.backward[Self.batch_size](
            d_actor_loss,
            batch_obs,
            actor_noise,
            h1_actor,
            h2_actor,
            mean_actor,
            log_std_actor,
            tanh_actor,
        )
        self.actor.update_adam(self.actor_lr)

        # ========================================
        # Alpha update (if auto_alpha)
        # ========================================

        if self.auto_alpha:
            # Alpha loss: J(α) = E[-α * (log π + target_entropy)]
            # Gradient: ∂J/∂α = -E[log π + target_entropy]
            var alpha_grad: Scalar[Self.dtype] = 0.0
            for i in range(Self.batch_size):
                alpha_grad += -(actor_log_prob_sum[i] + self.target_entropy)
            alpha_grad /= batch_size_scalar

            # Update log_alpha (not alpha directly for stability)
            self.log_alpha -= self.alpha_lr * alpha_grad

            # Clamp log_alpha to reasonable range
            if self.log_alpha < -10.0:
                self.log_alpha = -10.0
            elif self.log_alpha > 2.0:
                self.log_alpha = 2.0

            self.alpha = Scalar[Self.dtype](exp(Float64(self.log_alpha)))

        # ========================================
        # Soft update target networks
        # ========================================

        self.critic1_target.soft_update_from(self.critic1, self.tau)
        self.critic2_target.soft_update_from(self.critic2, self.tau)

        return total_critic_loss

    fn print_info(self):
        """Print agent information."""
        print("Deep SAC Agent:")
        print("  Obs dim: " + String(Self.obs_dim))
        print("  Action dim: " + String(Self.action_dim))
        print("  Hidden dim: " + String(Self.hidden_dim))
        print("  Buffer capacity: " + String(Self.buffer_capacity))
        print("  Batch size: " + String(Self.batch_size))
        print("  Gamma: " + String(self.gamma)[:6])
        print("  Tau: " + String(self.tau)[:6])
        print("  Actor LR: " + String(self.actor_lr)[:8])
        print("  Critic LR: " + String(self.critic_lr)[:8])
        print("  Alpha: " + String(self.alpha)[:6])
        print("  Auto alpha: " + String(self.auto_alpha))
        print("  Target entropy: " + String(self.target_entropy)[:6])
        self.actor.print_info("  Actor")
        self.critic1.print_info("  Critic1")
        self.critic2.print_info("  Critic2")

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
        """Train the Deep SAC agent on a continuous control environment.

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
            algorithm_name="Deep SAC",
            environment_name=environment_name,
        )

        if verbose:
            print("=" * 60)
            print("Deep SAC Training on " + environment_name)
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
                var obs = _list_to_inline_sac[Self.obs_dim, Self.dtype](
                    obs_list
                )
                var next_obs_list = env.get_obs_list()
                var next_obs = _list_to_inline_sac[Self.obs_dim, Self.dtype](
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
                var obs = _list_to_inline_sac[Self.obs_dim, Self.dtype](
                    obs_list
                )

                # Select action using stochastic policy
                var action = self.select_action(obs, deterministic=False)

                # Step environment (extract first action for single-action envs)
                var step_result = env.step_continuous(Float64(action[0]))
                var reward = step_result[1]
                done = step_result[2]

                # Get next observation
                var next_obs_list = env.get_obs_list()
                var next_obs = _list_to_inline_sac[Self.obs_dim, Self.dtype](
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
                episode, episode_reward, steps, Float64(self.alpha)
            )
            self.total_episodes += 1

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
                    + " | Alpha: "
                    + String(self.alpha)[:5]
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
            print("Final alpha: " + String(self.alpha)[:6])

        return metrics^

    fn evaluate[
        E: BoxContinuousActionEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps_per_episode: Int = 200,
        verbose: Bool = False,
        render: Bool = False,
    ) -> Float64:
        """Evaluate the trained agent using deterministic policy (mean action).

        Args:
            env: Environment implementing BoxContinuousActionEnv trait.
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
                var obs = _list_to_inline_sac[Self.obs_dim, Self.dtype](
                    obs_list
                )

                # Use deterministic action (mean, no sampling noise)
                var action = self.select_action(obs, deterministic=True)

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


fn _list_to_inline_sac[
    size: Int, dtype: DType = DType.float64
](obs_list: List[Float64]) -> InlineArray[Scalar[dtype], size]:
    """Convert List[Float64] to InlineArray."""
    var obs = InlineArray[Scalar[dtype], size](fill=0)
    for i in range(size):
        if i < len(obs_list):
            obs[i] = Scalar[dtype](obs_list[i])
    return obs^

"""Deep Deterministic Policy Gradient (DDPG) with Linear Function Approximation.

DDPG is an actor-critic algorithm for continuous action spaces. Unlike
stochastic policy gradient methods, DDPG learns a deterministic policy.

Key components:
1. Deterministic actor: μ(s) directly outputs the action
2. Q-function critic: Q(s, a) estimates action-value
3. Target networks: Stabilize learning via soft updates
4. Experience replay: Off-policy learning from past experiences
5. Exploration noise: Added during training for exploration

This implementation uses linear function approximation instead of neural networks:
- Actor: μ(s) = tanh(w_actor · φ(s)) * action_scale
- Critic: Q(s, a) = w_critic · [φ(s); a; a²]

References:
- Lillicrap et al. (2015): "Continuous control with deep reinforcement learning"
- Silver et al. (2014): "Deterministic Policy Gradient Algorithms"

Example usage:
    from core import PolynomialFeatures, ContinuousReplayBuffer
    from agents.ddpg import DDPGAgent
    from envs import PendulumEnv

    var env = PendulumEnv()
    var features = PendulumEnv.make_poly_features(degree=2)
    var buffer = ContinuousReplayBuffer(capacity=100000, feature_dim=features.get_num_features())

    var agent = DDPGAgent(
        num_state_features=features.get_num_features(),
        action_scale=2.0,  # Pendulum: [-2, 2]
        actor_lr=0.001,
        critic_lr=0.001,
    )

    # Training loop
    var obs = env.reset_obs()
    var state_features = features.get_features_simd4(obs)
    var action = agent.select_action_with_noise(state_features)
    var next_obs, reward, done = env.step_continuous(action)
    var next_features = features.get_features_simd4(next_obs)
    buffer.push(state_features, action, reward, next_features, done)

    if buffer.len() >= batch_size:
        var batch = buffer.sample(batch_size)
        agent.update(batch)
"""

from math import exp, tanh, sqrt
from random import random_float64
from core.continuous_replay_buffer import ContinuousTransition, ContinuousReplayBuffer
from core import PolynomialFeatures, TrainingMetrics, BoxContinuousActionEnv


struct DDPGAgent(Copyable, Movable):
    """DDPG agent with linear function approximation.

    Actor: Deterministic policy μ(s) = tanh(w_actor · φ(s)) * action_scale
    Critic: Q-function Q(s, a) = w_critic · [φ(s); a; a²]

    Uses target networks for stability and Gaussian noise for exploration.
    """

    # Actor weights: w_actor[feature] -> scalar output
    var actor_weights: List[Float64]
    var target_actor_weights: List[Float64]

    # Critic weights: w_critic[feature + action_features]
    # Input features: [φ(s); a; a²] (state features + action + action²)
    var critic_weights: List[Float64]
    var target_critic_weights: List[Float64]

    # Dimensions
    var num_state_features: Int
    var num_critic_features: Int  # num_state_features + 2 (for a and a²)

    # Hyperparameters
    var actor_lr: Float64
    var critic_lr: Float64
    var discount_factor: Float64
    var tau: Float64  # Soft update rate for target networks
    var noise_std: Float64  # Gaussian exploration noise standard deviation
    var action_scale: Float64  # Maximum action magnitude

    fn __init__(
        out self,
        num_state_features: Int,
        action_scale: Float64 = 2.0,
        actor_lr: Float64 = 0.001,
        critic_lr: Float64 = 0.001,
        discount_factor: Float64 = 0.99,
        tau: Float64 = 0.005,
        noise_std: Float64 = 0.1,
        init_std: Float64 = 0.01,
    ):
        """Initialize DDPG agent.

        Args:
            num_state_features: Dimensionality of state feature vectors
            action_scale: Maximum action magnitude (output clipped to [-action_scale, action_scale])
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            discount_factor: Discount factor γ
            tau: Soft update rate for target networks (typical: 0.001-0.01)
            noise_std: Standard deviation of Gaussian exploration noise (scaled by action_scale)
            init_std: Standard deviation for weight initialization
        """
        self.num_state_features = num_state_features
        self.num_critic_features = num_state_features + 2  # +2 for [a, a²]
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor
        self.tau = tau
        self.noise_std = noise_std
        self.action_scale = action_scale

        # Initialize actor weights with small random values
        self.actor_weights = List[Float64]()
        self.target_actor_weights = List[Float64]()
        for _ in range(num_state_features):
            var w = (random_float64() - 0.5) * 2.0 * init_std
            self.actor_weights.append(w)
            self.target_actor_weights.append(w)

        # Initialize critic weights
        self.critic_weights = List[Float64]()
        self.target_critic_weights = List[Float64]()
        for _ in range(self.num_critic_features):
            var w = (random_float64() - 0.5) * 2.0 * init_std
            self.critic_weights.append(w)
            self.target_critic_weights.append(w)

    fn __copyinit__(out self, existing: Self):
        self.num_state_features = existing.num_state_features
        self.num_critic_features = existing.num_critic_features
        self.actor_lr = existing.actor_lr
        self.critic_lr = existing.critic_lr
        self.discount_factor = existing.discount_factor
        self.tau = existing.tau
        self.noise_std = existing.noise_std
        self.action_scale = existing.action_scale

        self.actor_weights = List[Float64]()
        self.target_actor_weights = List[Float64]()
        for i in range(existing.num_state_features):
            self.actor_weights.append(existing.actor_weights[i])
            self.target_actor_weights.append(existing.target_actor_weights[i])

        self.critic_weights = List[Float64]()
        self.target_critic_weights = List[Float64]()
        for i in range(existing.num_critic_features):
            self.critic_weights.append(existing.critic_weights[i])
            self.target_critic_weights.append(existing.target_critic_weights[i])

    fn __moveinit__(out self, deinit existing: Self):
        self.num_state_features = existing.num_state_features
        self.num_critic_features = existing.num_critic_features
        self.actor_lr = existing.actor_lr
        self.critic_lr = existing.critic_lr
        self.discount_factor = existing.discount_factor
        self.tau = existing.tau
        self.noise_std = existing.noise_std
        self.action_scale = existing.action_scale
        self.actor_weights = existing.actor_weights^
        self.target_actor_weights = existing.target_actor_weights^
        self.critic_weights = existing.critic_weights^
        self.target_critic_weights = existing.target_critic_weights^

    # ========================================================================
    # Actor (Policy) Methods
    # ========================================================================

    fn _compute_actor_output(self, features: List[Float64], weights: List[Float64]) -> Float64:
        """Compute raw actor output (before tanh and scaling).

        Args:
            features: State feature vector φ(s)
            weights: Actor weights to use (online or target)

        Returns:
            Raw linear output w · φ(s)
        """
        var output: Float64 = 0.0
        var n = min(len(features), len(weights))
        for i in range(n):
            output += weights[i] * features[i]
        return output

    fn select_action(self, features: List[Float64]) -> Float64:
        """Select action using deterministic policy (no noise).

        μ(s) = tanh(w · φ(s)) * action_scale

        Args:
            features: State feature vector φ(s)

        Returns:
            Deterministic action in [-action_scale, action_scale]
        """
        var raw_output = self._compute_actor_output(features, self.actor_weights)
        return tanh(raw_output) * self.action_scale

    fn select_action_with_noise(self, features: List[Float64]) -> Float64:
        """Select action with Gaussian exploration noise.

        a = μ(s) + N(0, σ²)

        Args:
            features: State feature vector φ(s)

        Returns:
            Noisy action clipped to [-action_scale, action_scale]
        """
        var action = self.select_action(features)

        # Add Gaussian noise using Box-Muller transform
        var u1 = random_float64()
        var u2 = random_float64()
        # Avoid log(0)
        if u1 < 1e-10:
            u1 = 1e-10
        var noise = sqrt(-2.0 * _log(u1)) * _cos(2.0 * 3.141592653589793 * u2)
        noise *= self.noise_std * self.action_scale

        action += noise

        # Clip to action bounds
        if action > self.action_scale:
            action = self.action_scale
        elif action < -self.action_scale:
            action = -self.action_scale

        return action

    fn _select_action_target(self, features: List[Float64]) -> Float64:
        """Select action using target actor (for TD target computation)."""
        var raw_output = self._compute_actor_output(features, self.target_actor_weights)
        return tanh(raw_output) * self.action_scale

    # ========================================================================
    # Critic (Q-function) Methods
    # ========================================================================

    fn _build_critic_features(self, state_features: List[Float64], action: Float64) -> List[Float64]:
        """Build critic input features by concatenating state features with action features.

        Critic features = [φ(s); a; a²]

        Args:
            state_features: State feature vector φ(s)
            action: Action value

        Returns:
            Concatenated feature vector for critic
        """
        var critic_features = List[Float64]()

        # Add state features
        for i in range(len(state_features)):
            critic_features.append(state_features[i])

        # Add action features: [a, a²]
        var a_normalized = action / self.action_scale  # Normalize to [-1, 1]
        critic_features.append(a_normalized)
        critic_features.append(a_normalized * a_normalized)

        return critic_features^

    fn get_q_value(self, state_features: List[Float64], action: Float64) -> Float64:
        """Compute Q-value using online critic.

        Q(s, a) = w · [φ(s); a; a²]

        Args:
            state_features: State feature vector φ(s)
            action: Action value

        Returns:
            Q-value estimate
        """
        var critic_features = self._build_critic_features(state_features, action)
        var q_value: Float64 = 0.0
        var n = min(len(critic_features), len(self.critic_weights))
        for i in range(n):
            q_value += self.critic_weights[i] * critic_features[i]
        return q_value

    fn _get_q_value_target(self, state_features: List[Float64], action: Float64) -> Float64:
        """Compute Q-value using target critic."""
        var critic_features = self._build_critic_features(state_features, action)
        var q_value: Float64 = 0.0
        var n = min(len(critic_features), len(self.target_critic_weights))
        for i in range(n):
            q_value += self.target_critic_weights[i] * critic_features[i]
        return q_value

    # ========================================================================
    # Update Methods
    # ========================================================================

    fn update(mut self, batch: List[ContinuousTransition]):
        """Update actor and critic from a batch of transitions.

        1. Update critic using TD error
        2. Update actor using policy gradient
        3. Soft update target networks

        Args:
            batch: List of ContinuousTransition objects
        """
        if len(batch) == 0:
            return

        # Update critic
        self._update_critic(batch)

        # Update actor
        self._update_actor(batch)

        # Soft update target networks
        self._soft_update_targets()

    fn _update_critic(mut self, batch: List[ContinuousTransition]):
        """Update critic using TD error.

        Loss = (1/N) Σ (Q(s,a) - y)²
        where y = r + γ * Q_target(s', μ_target(s'))

        Uses semi-gradient update:
        w += α * (y - Q(s,a)) * ∇Q(s,a)
        """
        var batch_size = len(batch)
        var step_size = self.critic_lr / Float64(batch_size)

        for i in range(batch_size):
            var transition = batch[i]

            # Compute TD target: y = r + γ * Q_target(s', μ_target(s'))
            var target: Float64
            if transition.done:
                target = transition.reward
            else:
                var next_action = self._select_action_target(transition.next_state)
                var next_q = self._get_q_value_target(transition.next_state, next_action)
                target = transition.reward + self.discount_factor * next_q

            # Current Q-value and features
            var critic_features = self._build_critic_features(transition.state, transition.action)
            var current_q = self.get_q_value(transition.state, transition.action)

            # TD error
            var td_error = target - current_q

            # Update critic weights: w += α * δ * φ
            for j in range(len(critic_features)):
                if j < len(self.critic_weights):
                    self.critic_weights[j] += step_size * td_error * critic_features[j]

    fn _update_actor(mut self, batch: List[ContinuousTransition]):
        """Update actor using deterministic policy gradient.

        The policy gradient for DDPG is:
        ∇_θ J ≈ (1/N) Σ ∇_a Q(s, a)|_{a=μ(s)} * ∇_θ μ(s)

        For linear critic Q(s,a) = w · [φ(s); a; a²]:
        ∇_a Q(s, a) = w[state_dim] + 2 * w[state_dim+1] * a

        For linear actor μ(s) = tanh(w_actor · φ(s)) * scale:
        ∇_θ μ(s) = scale * (1 - tanh²(w_actor · φ(s))) * φ(s)
        """
        var batch_size = len(batch)
        var step_size = self.actor_lr / Float64(batch_size)

        for i in range(batch_size):
            var transition = batch[i]

            # Compute current action from actor
            var raw_output = self._compute_actor_output(transition.state, self.actor_weights)
            var action = tanh(raw_output) * self.action_scale

            # Compute ∇_a Q(s, a)
            # Q(s, a) = Σ w_i * φ_i(s) + w_{n} * a_norm + w_{n+1} * a_norm²
            # a_norm = a / action_scale
            # ∇_a Q = (1/scale) * (w_n + 2 * w_{n+1} * a_norm)
            var a_norm = action / self.action_scale
            var grad_a_q: Float64 = 0.0
            if self.num_state_features < len(self.critic_weights):
                grad_a_q = self.critic_weights[self.num_state_features] / self.action_scale
            if self.num_state_features + 1 < len(self.critic_weights):
                grad_a_q += 2.0 * self.critic_weights[self.num_state_features + 1] * a_norm / self.action_scale

            # Compute ∇_θ μ(s)
            # μ(s) = tanh(h) * scale, where h = w_actor · φ(s)
            # ∇_θ μ(s) = scale * sech²(h) * φ(s) = scale * (1 - tanh²(h)) * φ(s)
            var tanh_h = tanh(raw_output)
            var grad_mu_scale = self.action_scale * (1.0 - tanh_h * tanh_h)

            # Update actor weights using chain rule: ∇_θ J = ∇_a Q * ∇_θ μ
            for j in range(len(transition.state)):
                if j < len(self.actor_weights):
                    var grad_theta = grad_a_q * grad_mu_scale * transition.state[j]
                    self.actor_weights[j] += step_size * grad_theta

    fn _soft_update_targets(mut self):
        """Soft update target networks.

        θ_target = τ * θ + (1 - τ) * θ_target
        """
        # Update target actor
        for i in range(len(self.actor_weights)):
            self.target_actor_weights[i] = (
                self.tau * self.actor_weights[i]
                + (1.0 - self.tau) * self.target_actor_weights[i]
            )

        # Update target critic
        for i in range(len(self.critic_weights)):
            self.target_critic_weights[i] = (
                self.tau * self.critic_weights[i]
                + (1.0 - self.tau) * self.target_critic_weights[i]
            )

    fn reset(mut self):
        """Reset for new episode (no-op for DDPG)."""
        pass

    # ========================================================================
    # Training and Evaluation
    # ========================================================================

    fn train[E: BoxContinuousActionEnv](
        mut self,
        mut env: E,
        features: PolynomialFeatures,
        mut buffer: ContinuousReplayBuffer,
        num_episodes: Int,
        max_steps_per_episode: Int = 200,
        batch_size: Int = 64,
        min_buffer_size: Int = 1000,
        warmup_episodes: Int = 10,
        verbose: Bool = False,
        print_every: Int = 10,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train the DDPG agent on a continuous control environment.

        Args:
            env: Environment implementing ContinuousControlEnv trait.
            features: PolynomialFeatures extractor for state representation.
            buffer: ContinuousReplayBuffer for experience storage.
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
            batch_size: Minibatch size for updates.
            min_buffer_size: Minimum buffer size before starting updates.
            warmup_episodes: Episodes of random actions for initial buffer filling.
            verbose: Whether to print progress.
            print_every: Print progress every N episodes.
            environment_name: Name for logging.

        Returns:
            TrainingMetrics object with episode history.
        """
        var metrics = TrainingMetrics(
            algorithm_name="DDPG (Linear FA)",
            environment_name=environment_name,
        )

        if verbose:
            print("=" * 60)
            print("DDPG Training on", environment_name)
            print("=" * 60)
            print("State features:", self.num_state_features)
            print("Critic features:", self.num_critic_features)
            print("Action scale:", self.action_scale)
            print("Actor LR:", self.actor_lr, "Critic LR:", self.critic_lr)
            print("Tau:", self.tau, "Noise std:", self.noise_std)
            print("Warmup episodes:", warmup_episodes)
            print("Batch size:", batch_size)
            print("-" * 60)

        var total_steps = 0

        for episode in range(num_episodes):
            var obs_list = env.reset_obs_list()
            var obs = _list_to_simd4(obs_list)
            var episode_reward: Float64 = 0.0
            var steps = 0

            for step in range(max_steps_per_episode):
                # Extract features from observation
                var state_features = features.get_features_simd4(obs)

                # Select action (with or without noise)
                var action: Float64
                if episode < warmup_episodes:
                    # Random exploration during warmup
                    action = (random_float64() * 2.0 - 1.0) * self.action_scale
                else:
                    action = self.select_action_with_noise(state_features)

                # Take action in environment
                var result = env.step_continuous(action)
                var next_obs_list = result[0]
                var next_obs = _list_to_simd4(next_obs_list)
                var reward = result[1]
                var done = result[2]

                # Extract next state features
                var next_features = features.get_features_simd4(next_obs)

                # Store transition in replay buffer
                buffer.push(state_features, action, reward, next_features, done)

                # Update agent if we have enough samples
                if buffer.len() >= min_buffer_size and episode >= warmup_episodes:
                    var batch = buffer.sample(batch_size)
                    self.update(batch)

                episode_reward += reward
                steps += 1
                total_steps += 1
                obs = next_obs

                if done:
                    break

            # Log episode metrics
            metrics.log_episode(episode, episode_reward, steps, self.noise_std)

            # Print progress
            if verbose and (episode + 1) % print_every == 0:
                # Compute recent average reward
                var start_idx = max(0, len(metrics.episodes) - print_every)
                var sum_reward: Float64 = 0.0
                for j in range(start_idx, len(metrics.episodes)):
                    sum_reward += metrics.episodes[j].total_reward
                var avg_reward = sum_reward / Float64(len(metrics.episodes) - start_idx)
                print(
                    "Episode",
                    episode + 1,
                    "| Avg Reward:",
                    String(avg_reward)[:8],
                    "| Steps:",
                    steps,
                    "| Buffer:",
                    buffer.len(),
                )

        if verbose:
            print("-" * 60)
            print("Training complete!")
            # Compute final 100-episode average
            var start_idx = max(0, len(metrics.episodes) - 100)
            var sum_reward: Float64 = 0.0
            for j in range(start_idx, len(metrics.episodes)):
                sum_reward += metrics.episodes[j].total_reward
            var final_avg = sum_reward / Float64(len(metrics.episodes) - start_idx)
            print("Final avg reward:", String(final_avg)[:8])

        return metrics^

    fn evaluate[E: BoxContinuousActionEnv](
        self,
        mut env: E,
        features: PolynomialFeatures,
        num_episodes: Int = 10,
        max_steps_per_episode: Int = 200,
    ) -> Float64:
        """Evaluate the trained DDPG agent using deterministic policy.

        Args:
            env: Environment implementing ContinuousControlEnv trait.
            features: PolynomialFeatures extractor for state representation.
            num_episodes: Number of evaluation episodes.
            max_steps_per_episode: Maximum steps per episode.

        Returns:
            Average reward over evaluation episodes.
        """
        var total_reward: Float64 = 0.0

        for _ in range(num_episodes):
            var obs_list = env.reset_obs_list()
            var obs = _list_to_simd4(obs_list)
            var episode_reward: Float64 = 0.0

            for _ in range(max_steps_per_episode):
                var state_features = features.get_features_simd4(obs)

                # Use deterministic action (no noise)
                var action = self.select_action(state_features)

                var result = env.step_continuous(action)
                var next_obs_list = result[0]
                var next_obs = _list_to_simd4(next_obs_list)
                var reward = result[1]
                var done = result[2]

                episode_reward += reward
                obs = next_obs

                if done:
                    break

            total_reward += episode_reward

        return total_reward / Float64(num_episodes)


# ============================================================================
# Helper functions (math utilities)
# ============================================================================


fn _log(x: Float64) -> Float64:
    """Natural logarithm."""
    from math import log
    return log(x)


fn _cos(x: Float64) -> Float64:
    """Cosine function."""
    from math import cos
    return cos(x)


fn _gaussian_noise() -> Float64:
    """Generate Gaussian noise using Box-Muller transform."""
    var u1 = random_float64()
    var u2 = random_float64()
    if u1 < 1e-10:
        u1 = 1e-10
    return sqrt(-2.0 * _log(u1)) * _cos(2.0 * 3.141592653589793 * u2)


fn _list_to_simd4(obs: List[Float64]) -> SIMD[DType.float64, 4]:
    """Convert a List[Float64] to SIMD[DType.float64, 4].

    Pads with zeros if the list has fewer than 4 elements.
    """
    var result = SIMD[DType.float64, 4](0.0)
    var n = min(len(obs), 4)
    for i in range(n):
        result[i] = obs[i]
    return result

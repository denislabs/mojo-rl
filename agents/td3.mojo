"""Twin Delayed Deep Deterministic Policy Gradient (TD3) with Linear Function Approximation.

TD3 is an improved version of DDPG that addresses function approximation error
through three key innovations:

1. Twin Q-networks: Use two Q-functions and take the minimum to reduce overestimation
2. Delayed policy updates: Update the policy less frequently than the Q-functions
3. Target policy smoothing: Add noise to target actions to smooth the Q-function

This implementation uses linear function approximation instead of neural networks:
- Actor: μ(s) = tanh(w_actor · φ(s)) * action_scale
- Critics: Q1(s, a) = w_critic1 · [φ(s); a; a²], Q2(s, a) = w_critic2 · [φ(s); a; a²]

References:
- Fujimoto et al. (2018): "Addressing Function Approximation Error in Actor-Critic Methods"
- Original code: https://github.com/sfujim/TD3

Example usage:
    from core import PolynomialFeatures, ContinuousReplayBuffer
    from agents.td3 import TD3Agent
    from envs import PendulumEnv

    var env = PendulumEnv()
    var features = PendulumEnv.make_poly_features(degree=2)
    var buffer = ContinuousReplayBuffer(capacity=100000, feature_dim=features.get_num_features())

    var agent = TD3Agent(
        num_state_features=features.get_num_features(),
        action_scale=2.0,  # Pendulum: [-2, 2]
        actor_lr=0.001,
        critic_lr=0.001,
        policy_delay=2,  # Update actor every 2 critic updates
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
from core.continuous_replay_buffer import (
    ContinuousTransition,
    ContinuousReplayBuffer,
)
from core import PolynomialFeatures, TrainingMetrics, BoxContinuousActionEnv
from deep_rl.gpu.random import gaussian_noise


struct TD3Agent(Copyable, Movable):
    """TD3 agent with linear function approximation.

    Actor: Deterministic policy μ(s) = tanh(w_actor · φ(s)) * action_scale
    Critics: Twin Q-functions Q1(s, a), Q2(s, a) = w_critic · [φ(s); a; a²]

    Key improvements over DDPG:
    1. Twin Q-networks to reduce overestimation bias
    2. Delayed policy updates for more stable learning
    3. Target policy smoothing to prevent exploitation of Q-function errors
    """

    # Actor weights: w_actor[feature] -> scalar output
    var actor_weights: List[Float64]
    var target_actor_weights: List[Float64]

    # Twin Critic weights: w_critic[feature + action_features]
    # Input features: [φ(s); a; a²] (state features + action + action²)
    var critic1_weights: List[Float64]
    var target_critic1_weights: List[Float64]
    var critic2_weights: List[Float64]
    var target_critic2_weights: List[Float64]

    # Dimensions
    var num_state_features: Int
    var num_critic_features: Int  # num_state_features + 2 (for a and a²)

    # Hyperparameters
    var actor_lr: Float64
    var critic_lr: Float64
    var discount_factor: Float64
    var tau: Float64  # Soft update rate for target networks
    var noise_std: Float64  # Gaussian exploration noise standard deviation
    var noise_std_min: Float64  # Minimum noise after decay
    var noise_decay: Float64  # Noise decay rate per episode
    var action_scale: Float64  # Maximum action magnitude
    var reward_scale: Float64  # Reward scaling factor
    var updates_per_step: Int  # Number of gradient updates per env step

    # TD3-specific hyperparameters
    var policy_delay: Int  # Update actor every policy_delay critic updates
    var target_noise_std: Float64  # Noise added to target action
    var target_noise_clip: Float64  # Clip range for target noise

    # Internal state
    var update_count: Int  # Track number of updates for delayed policy

    # Pre-allocated storage for performance
    var _critic_features: List[Float64]

    fn __init__(
        out self,
        num_state_features: Int,
        action_scale: Float64 = 2.0,
        actor_lr: Float64 = 0.0003,
        critic_lr: Float64 = 0.001,
        discount_factor: Float64 = 0.99,
        tau: Float64 = 0.005,
        noise_std: Float64 = 0.2,
        noise_std_min: Float64 = 0.05,
        noise_decay: Float64 = 0.995,
        reward_scale: Float64 = 0.1,
        updates_per_step: Int = 1,
        policy_delay: Int = 2,
        target_noise_std: Float64 = 0.2,
        target_noise_clip: Float64 = 0.5,
        init_std: Float64 = 0.1,
    ):
        """Initialize TD3 agent.

        Args:
            num_state_features: Dimensionality of state feature vectors
            action_scale: Maximum action magnitude (output clipped to [-action_scale, action_scale])
            actor_lr: Actor learning rate (lower is more stable)
            critic_lr: Critic learning rate
            discount_factor: Discount factor γ
            tau: Soft update rate for target networks (typical: 0.001-0.01)
            noise_std: Initial standard deviation of Gaussian exploration noise
            noise_std_min: Minimum noise after decay
            noise_decay: Noise decay rate per episode (1.0 = no decay)
            reward_scale: Scale factor for rewards (helps with large reward ranges)
            updates_per_step: Number of gradient updates per environment step
            policy_delay: Update actor every policy_delay critic updates (default: 2)
            target_noise_std: Standard deviation of target policy smoothing noise
            target_noise_clip: Clip range for target noise (default: 0.5)
            init_std: Standard deviation for weight initialization
        """
        self.num_state_features = num_state_features
        self.num_critic_features = num_state_features + 2  # +2 for [a, a²]
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor
        self.tau = tau
        self.noise_std = noise_std
        self.noise_std_min = noise_std_min
        self.noise_decay = noise_decay
        self.action_scale = action_scale
        self.reward_scale = reward_scale
        self.updates_per_step = updates_per_step
        self.policy_delay = policy_delay
        self.target_noise_std = target_noise_std
        self.target_noise_clip = target_noise_clip
        self.update_count = 0

        # Pre-allocate critic features storage
        self._critic_features = List[Float64](capacity=self.num_critic_features)
        for _ in range(self.num_critic_features):
            self._critic_features.append(0.0)

        # Initialize actor weights with small random values
        self.actor_weights = List[Float64]()
        self.target_actor_weights = List[Float64]()
        for _ in range(num_state_features):
            var w = (random_float64() - 0.5) * 2.0 * init_std
            self.actor_weights.append(w)
            self.target_actor_weights.append(w)

        # Initialize twin critic weights
        self.critic1_weights = List[Float64]()
        self.target_critic1_weights = List[Float64]()
        self.critic2_weights = List[Float64]()
        self.target_critic2_weights = List[Float64]()
        for _ in range(self.num_critic_features):
            var w1 = (random_float64() - 0.5) * 2.0 * init_std
            var w2 = (random_float64() - 0.5) * 2.0 * init_std
            self.critic1_weights.append(w1)
            self.target_critic1_weights.append(w1)
            self.critic2_weights.append(w2)
            self.target_critic2_weights.append(w2)

    fn __copyinit__(out self, existing: Self):
        self.num_state_features = existing.num_state_features
        self.num_critic_features = existing.num_critic_features
        self.actor_lr = existing.actor_lr
        self.critic_lr = existing.critic_lr
        self.discount_factor = existing.discount_factor
        self.tau = existing.tau
        self.noise_std = existing.noise_std
        self.noise_std_min = existing.noise_std_min
        self.noise_decay = existing.noise_decay
        self.action_scale = existing.action_scale
        self.reward_scale = existing.reward_scale
        self.updates_per_step = existing.updates_per_step
        self.policy_delay = existing.policy_delay
        self.target_noise_std = existing.target_noise_std
        self.target_noise_clip = existing.target_noise_clip
        self.update_count = existing.update_count

        self.actor_weights = List[Float64]()
        self.target_actor_weights = List[Float64]()
        for i in range(existing.num_state_features):
            self.actor_weights.append(existing.actor_weights[i])
            self.target_actor_weights.append(existing.target_actor_weights[i])

        self.critic1_weights = List[Float64]()
        self.target_critic1_weights = List[Float64]()
        self.critic2_weights = List[Float64]()
        self.target_critic2_weights = List[Float64]()
        for i in range(existing.num_critic_features):
            self.critic1_weights.append(existing.critic1_weights[i])
            self.target_critic1_weights.append(
                existing.target_critic1_weights[i]
            )
            self.critic2_weights.append(existing.critic2_weights[i])
            self.target_critic2_weights.append(
                existing.target_critic2_weights[i]
            )

        # Pre-allocate critic features storage
        self._critic_features = List[Float64](capacity=self.num_critic_features)
        for _ in range(self.num_critic_features):
            self._critic_features.append(0.0)

    fn __moveinit__(out self, deinit existing: Self):
        self.num_state_features = existing.num_state_features
        self.num_critic_features = existing.num_critic_features
        self.actor_lr = existing.actor_lr
        self.critic_lr = existing.critic_lr
        self.discount_factor = existing.discount_factor
        self.tau = existing.tau
        self.noise_std = existing.noise_std
        self.noise_std_min = existing.noise_std_min
        self.noise_decay = existing.noise_decay
        self.action_scale = existing.action_scale
        self.reward_scale = existing.reward_scale
        self.updates_per_step = existing.updates_per_step
        self.policy_delay = existing.policy_delay
        self.target_noise_std = existing.target_noise_std
        self.target_noise_clip = existing.target_noise_clip
        self.update_count = existing.update_count
        self.actor_weights = existing.actor_weights^
        self.target_actor_weights = existing.target_actor_weights^
        self.critic1_weights = existing.critic1_weights^
        self.target_critic1_weights = existing.target_critic1_weights^
        self.critic2_weights = existing.critic2_weights^
        self.target_critic2_weights = existing.target_critic2_weights^
        self._critic_features = existing._critic_features^

    # ========================================================================
    # Actor (Policy) Methods
    # ========================================================================

    fn _compute_actor_output(
        self, features: List[Float64], weights: List[Float64]
    ) -> Float64:
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
        var raw_output = self._compute_actor_output(
            features, self.actor_weights
        )
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
        var noise = gaussian_noise() * self.noise_std * self.action_scale

        action += noise

        # Clip to action bounds
        if action > self.action_scale:
            action = self.action_scale
        elif action < -self.action_scale:
            action = -self.action_scale

        return action

    fn _select_action_target(self, features: List[Float64]) -> Float64:
        """Select action using target actor (for TD target computation)."""
        var raw_output = self._compute_actor_output(
            features, self.target_actor_weights
        )
        return tanh(raw_output) * self.action_scale

    fn _select_action_target_smoothed(self, features: List[Float64]) -> Float64:
        """Select action using target actor with smoothing noise.

        TD3 adds clipped noise to the target action to smooth the Q-function:
        a' = clip(μ_target(s') + clip(ε, -c, c), -action_scale, action_scale)
        where ε ~ N(0, σ_target²)

        Args:
            features: State feature vector φ(s')

        Returns:
            Smoothed target action
        """
        var action = self._select_action_target(features)

        # Add clipped Gaussian noise for target policy smoothing
        var noise = (
            gaussian_noise() * self.target_noise_std * self.action_scale
        )

        # Clip noise
        var clip_bound = self.target_noise_clip * self.action_scale
        if noise > clip_bound:
            noise = clip_bound
        elif noise < -clip_bound:
            noise = -clip_bound

        action += noise

        # Clip to action bounds
        if action > self.action_scale:
            action = self.action_scale
        elif action < -self.action_scale:
            action = -self.action_scale

        return action

    # ========================================================================
    # Critic (Q-function) Methods
    # ========================================================================

    fn _build_critic_features(
        self, state_features: List[Float64], action: Float64
    ) -> List[Float64]:
        """Build critic input features by concatenating state features with action features.

        Critic features = [φ(s); a; a²]

        Args:
            state_features: State feature vector φ(s)
            action: Action value

        Returns:
            Concatenated feature vector for critic
        """
        var critic_features = List[Float64](capacity=self.num_critic_features)

        # Add state features
        for i in range(len(state_features)):
            critic_features.append(state_features[i])

        # Add action features: [a, a²]
        var a_normalized = action / self.action_scale  # Normalize to [-1, 1]
        critic_features.append(a_normalized)
        critic_features.append(a_normalized * a_normalized)

        return critic_features^

    fn decay_noise(mut self):
        """Decay exploration noise (call once per episode)."""
        self.noise_std *= self.noise_decay
        if self.noise_std < self.noise_std_min:
            self.noise_std = self.noise_std_min

    fn get_q1_value(
        self, state_features: List[Float64], action: Float64
    ) -> Float64:
        """Compute Q-value using first online critic.

        Q1(s, a) = w · [φ(s); a; a²]

        Args:
            state_features: State feature vector φ(s)
            action: Action value

        Returns:
            Q1-value estimate
        """
        var critic_features = self._build_critic_features(
            state_features, action
        )
        var q_value: Float64 = 0.0
        var n = min(len(critic_features), len(self.critic1_weights))
        for i in range(n):
            q_value += self.critic1_weights[i] * critic_features[i]
        return q_value

    fn get_q2_value(
        self, state_features: List[Float64], action: Float64
    ) -> Float64:
        """Compute Q-value using second online critic.

        Q2(s, a) = w · [φ(s); a; a²]

        Args:
            state_features: State feature vector φ(s)
            action: Action value

        Returns:
            Q2-value estimate
        """
        var critic_features = self._build_critic_features(
            state_features, action
        )
        var q_value: Float64 = 0.0
        var n = min(len(critic_features), len(self.critic2_weights))
        for i in range(n):
            q_value += self.critic2_weights[i] * critic_features[i]
        return q_value

    fn _get_q1_value_target(
        self, state_features: List[Float64], action: Float64
    ) -> Float64:
        """Compute Q-value using first target critic."""
        var critic_features = self._build_critic_features(
            state_features, action
        )
        var q_value: Float64 = 0.0
        var n = min(len(critic_features), len(self.target_critic1_weights))
        for i in range(n):
            q_value += self.target_critic1_weights[i] * critic_features[i]
        return q_value

    fn _get_q2_value_target(
        self, state_features: List[Float64], action: Float64
    ) -> Float64:
        """Compute Q-value using second target critic."""
        var critic_features = self._build_critic_features(
            state_features, action
        )
        var q_value: Float64 = 0.0
        var n = min(len(critic_features), len(self.target_critic2_weights))
        for i in range(n):
            q_value += self.target_critic2_weights[i] * critic_features[i]
        return q_value

    fn get_min_q_value(
        self, state_features: List[Float64], action: Float64
    ) -> Float64:
        """Get minimum Q-value from twin critics (for conservative estimates).

        Args:
            state_features: State feature vector φ(s)
            action: Action value

        Returns:
            min(Q1(s,a), Q2(s,a))
        """
        var q1 = self.get_q1_value(state_features, action)
        var q2 = self.get_q2_value(state_features, action)
        if q1 < q2:
            return q1
        return q2

    # ========================================================================
    # Update Methods
    # ========================================================================

    fn update(mut self, batch: List[ContinuousTransition]):
        """Update critics and (maybe) actor from a batch of transitions.

        TD3 update procedure:
        1. Always update both critics using TD error with min target Q
        2. Every policy_delay updates:
           - Update actor using policy gradient from Q1
           - Soft update all target networks

        Args:
            batch: List of ContinuousTransition objects
        """
        if len(batch) == 0:
            return

        self.update_count += 1

        # Always update critics
        self._update_critics(batch)

        # Delayed policy update
        if self.update_count % self.policy_delay == 0:
            self._update_actor(batch)
            self._soft_update_targets()

    fn _update_critics(mut self, batch: List[ContinuousTransition]):
        """Update both critics using TD error.

        TD3 uses the minimum of target Q-values to compute the target:
        y = r + γ * min(Q1_target(s', ã'), Q2_target(s', ã'))
        where ã' = μ_target(s') + clipped_noise

        Both critics are updated with the same target.
        """
        var batch_size = len(batch)
        var step_size = self.critic_lr / Float64(batch_size)

        for i in range(batch_size):
            var transition = batch[i]

            # Compute TD target using min of target Q-values
            var target: Float64
            if transition.done:
                target = transition.reward
            else:
                # Target action with smoothing noise
                var next_action = self._select_action_target_smoothed(
                    transition.next_state
                )

                # Min of target Q-values (key TD3 innovation #1)
                var next_q1 = self._get_q1_value_target(
                    transition.next_state, next_action
                )
                var next_q2 = self._get_q2_value_target(
                    transition.next_state, next_action
                )
                var next_q = next_q1
                if next_q2 < next_q1:
                    next_q = next_q2

                target = transition.reward + self.discount_factor * next_q

            # Build critic features
            var critic_features = self._build_critic_features(
                transition.state, transition.action
            )

            # Update critic 1
            var current_q1 = self.get_q1_value(
                transition.state, transition.action
            )
            var td_error1 = target - current_q1
            for j in range(len(critic_features)):
                if j < len(self.critic1_weights):
                    self.critic1_weights[j] += (
                        step_size * td_error1 * critic_features[j]
                    )

            # Update critic 2
            var current_q2 = self.get_q2_value(
                transition.state, transition.action
            )
            var td_error2 = target - current_q2
            for j in range(len(critic_features)):
                if j < len(self.critic2_weights):
                    self.critic2_weights[j] += (
                        step_size * td_error2 * critic_features[j]
                    )

    fn _update_actor(mut self, batch: List[ContinuousTransition]):
        """Update actor using deterministic policy gradient.

        TD3 uses only Q1 for the policy gradient (not min):
        ∇_θ J ≈ (1/N) Σ ∇_a Q1(s, a)|_{a=μ(s)} * ∇_θ μ(s)

        For linear critic Q1(s,a) = w · [φ(s); a; a²]:
        ∇_a Q1(s, a) = w[state_dim] + 2 * w[state_dim+1] * a

        For linear actor μ(s) = tanh(w_actor · φ(s)) * scale:
        ∇_θ μ(s) = scale * (1 - tanh²(w_actor · φ(s))) * φ(s)
        """
        var batch_size = len(batch)
        var step_size = self.actor_lr / Float64(batch_size)

        for i in range(batch_size):
            var transition = batch[i]

            # Compute current action from actor
            var raw_output = self._compute_actor_output(
                transition.state, self.actor_weights
            )
            var action = tanh(raw_output) * self.action_scale

            # Compute ∇_a Q1(s, a) - using Q1 only (not min)
            var a_norm = action / self.action_scale
            var grad_a_q: Float64 = 0.0
            if self.num_state_features < len(self.critic1_weights):
                grad_a_q = (
                    self.critic1_weights[self.num_state_features]
                    / self.action_scale
                )
            if self.num_state_features + 1 < len(self.critic1_weights):
                grad_a_q += (
                    2.0
                    * self.critic1_weights[self.num_state_features + 1]
                    * a_norm
                    / self.action_scale
                )

            # Compute ∇_θ μ(s)
            var tanh_h = tanh(raw_output)
            var grad_mu_scale = self.action_scale * (1.0 - tanh_h * tanh_h)

            # Update actor weights using chain rule: ∇_θ J = ∇_a Q * ∇_θ μ
            for j in range(len(transition.state)):
                if j < len(self.actor_weights):
                    var grad_theta = (
                        grad_a_q * grad_mu_scale * transition.state[j]
                    )
                    self.actor_weights[j] += step_size * grad_theta

    fn _soft_update_targets(mut self):
        """Soft update all target networks.

        θ_target = τ * θ + (1 - τ) * θ_target
        """
        # Update target actor
        for i in range(len(self.actor_weights)):
            self.target_actor_weights[i] = (
                self.tau * self.actor_weights[i]
                + (1.0 - self.tau) * self.target_actor_weights[i]
            )

        # Update target critic 1
        for i in range(len(self.critic1_weights)):
            self.target_critic1_weights[i] = (
                self.tau * self.critic1_weights[i]
                + (1.0 - self.tau) * self.target_critic1_weights[i]
            )

        # Update target critic 2
        for i in range(len(self.critic2_weights)):
            self.target_critic2_weights[i] = (
                self.tau * self.critic2_weights[i]
                + (1.0 - self.tau) * self.target_critic2_weights[i]
            )

    fn reset(mut self):
        """Reset for new episode (resets update counter)."""
        pass

    # ========================================================================
    # Training and Evaluation
    # ========================================================================

    fn train[
        E: BoxContinuousActionEnv
    ](
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
        """Train the TD3 agent on a continuous control environment.

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
            algorithm_name="TD3 (Linear FA)",
            environment_name=environment_name,
        )

        if verbose:
            print("=" * 60)
            print("TD3 Training on", environment_name)
            print("=" * 60)
            print("State features:", self.num_state_features)
            print("Critic features:", self.num_critic_features)
            print("Action scale:", self.action_scale)
            print("Actor LR:", self.actor_lr, "Critic LR:", self.critic_lr)
            print("Tau:", self.tau, "Noise std:", self.noise_std)
            print("Policy delay:", self.policy_delay)
            print(
                "Target noise std:",
                self.target_noise_std,
                "clip:",
                self.target_noise_clip,
            )
            print("Warmup episodes:", warmup_episodes)
            print("Batch size:", batch_size)
            print("-" * 60)

        var total_steps = 0

        for episode in range(num_episodes):
            var obs_list = env.reset_obs_list()
            var obs = _list_to_simd4(obs_list)
            var episode_reward: Float64 = 0.0
            var steps = 0

            for _ in range(max_steps_per_episode):
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
                var next_obs_list = result[0].copy()
                var next_obs = _list_to_simd4(next_obs_list)
                var reward = result[1]
                var done = result[2]

                # Extract next state features
                var next_features = features.get_features_simd4(next_obs)

                # Store transition in replay buffer with scaled reward
                var scaled_reward = reward * self.reward_scale
                buffer.push(state_features, action, scaled_reward, next_features, done)

                # Update agent if we have enough samples
                if (
                    buffer.len() >= min_buffer_size
                    and episode >= warmup_episodes
                ):
                    # Multiple updates per step for faster learning
                    for _ in range(self.updates_per_step):
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

            # Decay exploration noise after warmup
            if episode >= warmup_episodes:
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
            var final_avg = sum_reward / Float64(
                len(metrics.episodes) - start_idx
            )
            print("Final avg reward:", String(final_avg)[:8])

        return metrics^

    fn evaluate[
        E: BoxContinuousActionEnv
    ](
        self,
        mut env: E,
        features: PolynomialFeatures,
        num_episodes: Int = 10,
        max_steps_per_episode: Int = 200,
        render: Bool = False,
    ) -> Float64:
        """Evaluate the trained TD3 agent using deterministic policy.

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
                if render:
                    env.render()

                var state_features = features.get_features_simd4(obs)

                # Use deterministic action (no noise)
                var action = self.select_action(state_features)

                var result = env.step_continuous(action)
                var next_obs_list = result[0].copy()
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


fn _list_to_simd4(obs: List[Float64]) -> SIMD[DType.float64, 4]:
    """Convert a List[Float64] to SIMD[DType.float64, 4].

    Pads with zeros if the list has fewer than 4 elements.
    """
    var result = SIMD[DType.float64, 4](0.0)
    var n = min(len(obs), 4)
    for i in range(n):
        result[i] = obs[i]
    return result

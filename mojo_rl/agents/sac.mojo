"""Soft Actor-Critic (SAC) with Linear Function Approximation.

SAC is a maximum entropy reinforcement learning algorithm that learns a
stochastic policy by maximizing both expected return and policy entropy.
This encourages exploration and leads to more robust policies.

Key components:
1. Stochastic actor: π(a|s) is a Gaussian with learned mean and std
2. Twin Q-networks: Like TD3, uses min(Q1, Q2) to reduce overestimation
3. Entropy bonus: Policy trained to maximize reward + α * entropy
4. Automatic temperature: α can be learned to maintain target entropy
5. No target actor: Uses current policy for next action sampling

This implementation uses linear function approximation:
- Actor mean: μ(s) = w_mean · φ(s)
- Actor log_std: log_σ(s) = w_logstd · φ(s) (clamped to [-20, 2])
- Critics: Q1(s, a) = w_critic1 · [φ(s); a; a²], Q2(s, a) = w_critic2 · [φ(s); a; a²]

References:
- Haarnoja et al. (2018): "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
- Haarnoja et al. (2018): "Soft Actor-Critic Algorithms and Applications"

Example usage:
    from mojo_rl.core import PolynomialFeatures, ContinuousReplayBuffer
    from mojo_rl.agents.sac import SACAgent
    from mojo_rl.envs import PendulumEnv

    var env = PendulumEnv()
    var features = PendulumEnv.make_poly_features(degree=2)
    var buffer = ContinuousReplayBuffer(capacity=100000, feature_dim=features.get_num_features())

    var agent = SACAgent(
        num_state_features=features.get_num_features(),
        action_scale=2.0,
        actor_lr=0.001,
        critic_lr=0.001,
        alpha=0.2,  # Entropy coefficient
    )

    # Training loop uses stochastic policy
    var action = agent.select_action(state_features)  # Samples from Gaussian
    # ... environment step ...
    agent.update(batch)
"""

from std.math import exp, tanh, sqrt, log
from std.random import random_float64
from mojo_rl.core.continuous_replay_buffer import (
    ContinuousTransition,
    ContinuousReplayBuffer,
)
from mojo_rl.core import (
    PolynomialFeatures,
    TrainingMetrics,
    BoxContinuousActionEnv,
    RenderableEnv,
)
from mojo_rl.nn.gpu.random import gaussian_noise


struct SACAgent(Copyable, Movable):
    """SAC agent with linear function approximation.

    Actor: Stochastic Gaussian policy π(a|s) = N(μ(s), σ(s)²)
           μ(s) = tanh(w_mean · φ(s)) * action_scale
           σ(s) = exp(clamp(w_logstd · φ(s), -20, 2))

    Critics: Twin Q-functions Q1(s, a), Q2(s, a) = w_critic · [φ(s); a; a²]

    Key SAC features:
    - Maximum entropy objective: maximize reward + α * entropy
    - Stochastic policy for better exploration
    - Twin critics to reduce overestimation (like TD3)
    - Optional automatic entropy temperature tuning
    """

    # Actor weights for mean: w_mean[feature] -> scalar mean
    var actor_mean_weights: List[Float64]
    # Actor weights for log std: w_logstd[feature] -> scalar log_std
    var actor_logstd_weights: List[Float64]

    # Twin Critic weights
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
    var action_scale: Float64  # Maximum action magnitude

    # SAC-specific hyperparameters
    var alpha: Float64  # Entropy coefficient (temperature)
    var target_entropy: Float64  # Target entropy for automatic tuning
    var alpha_lr: Float64  # Learning rate for alpha
    var log_alpha: Float64  # Log of alpha (for gradient-based learning)
    var auto_alpha: Bool  # Whether to automatically tune alpha

    # Constants for log_std clamping
    var log_std_min: Float64
    var log_std_max: Float64

    def __init__(
        out self,
        num_state_features: Int,
        action_scale: Float64 = 2.0,
        actor_lr: Float64 = 0.001,
        critic_lr: Float64 = 0.001,
        discount_factor: Float64 = 0.99,
        tau: Float64 = 0.005,
        alpha: Float64 = 0.2,
        auto_alpha: Bool = True,
        alpha_lr: Float64 = 0.001,
        target_entropy: Float64 = -1.0,  # -dim(A) is common default
        init_std: Float64 = 0.01,
    ):
        """Initialize SAC agent.

        Args:
            num_state_features: Dimensionality of state feature vectors.
            action_scale: Maximum action magnitude.
            actor_lr: Actor learning rate.
            critic_lr: Critic learning rate.
            discount_factor: Discount factor γ.
            tau: Soft update rate for target networks.
            alpha: Initial entropy coefficient (temperature).
            auto_alpha: Whether to automatically tune alpha.
            alpha_lr: Learning rate for alpha tuning.
            target_entropy: Target entropy for automatic tuning (default: -1 for 1D action).
            init_std: Standard deviation for weight initialization.
        """
        self.num_state_features = num_state_features
        self.num_critic_features = num_state_features + 2
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor
        self.tau = tau
        self.action_scale = action_scale
        self.alpha = alpha
        self.auto_alpha = auto_alpha
        self.alpha_lr = alpha_lr
        self.target_entropy = target_entropy
        self.log_alpha = log(alpha) if alpha > 0 else -2.0
        self.log_std_min = -20.0
        self.log_std_max = 2.0

        # Initialize actor mean weights
        self.actor_mean_weights = List[Float64]()
        for _ in range(num_state_features):
            var w = (random_float64() - 0.5) * 2.0 * init_std
            self.actor_mean_weights.append(w)

        # Initialize actor log_std weights (start with small values for moderate exploration)
        self.actor_logstd_weights = List[Float64]()
        for _ in range(num_state_features):
            var w = (
                random_float64() - 0.5
            ) * 2.0 * init_std - 1.0  # Bias toward lower variance
            self.actor_logstd_weights.append(w)

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

    def __init__(out self, *, copy: Self):
        self.num_state_features = copy.num_state_features
        self.num_critic_features = copy.num_critic_features
        self.actor_lr = copy.actor_lr
        self.critic_lr = copy.critic_lr
        self.discount_factor = copy.discount_factor
        self.tau = copy.tau
        self.action_scale = copy.action_scale
        self.alpha = copy.alpha
        self.auto_alpha = copy.auto_alpha
        self.alpha_lr = copy.alpha_lr
        self.target_entropy = copy.target_entropy
        self.log_alpha = copy.log_alpha
        self.log_std_min = copy.log_std_min
        self.log_std_max = copy.log_std_max

        self.actor_mean_weights = List[Float64]()
        self.actor_logstd_weights = List[Float64]()
        for i in range(copy.num_state_features):
            self.actor_mean_weights.append(copy.actor_mean_weights[i])
            self.actor_logstd_weights.append(copy.actor_logstd_weights[i])

        self.critic1_weights = List[Float64]()
        self.target_critic1_weights = List[Float64]()
        self.critic2_weights = List[Float64]()
        self.target_critic2_weights = List[Float64]()
        for i in range(copy.num_critic_features):
            self.critic1_weights.append(copy.critic1_weights[i])
            self.target_critic1_weights.append(copy.target_critic1_weights[i])
            self.critic2_weights.append(copy.critic2_weights[i])
            self.target_critic2_weights.append(copy.target_critic2_weights[i])

    def __init__(out self, *, deinit take: Self):
        self.num_state_features = take.num_state_features
        self.num_critic_features = take.num_critic_features
        self.actor_lr = take.actor_lr
        self.critic_lr = take.critic_lr
        self.discount_factor = take.discount_factor
        self.tau = take.tau
        self.action_scale = take.action_scale
        self.alpha = take.alpha
        self.auto_alpha = take.auto_alpha
        self.alpha_lr = take.alpha_lr
        self.target_entropy = take.target_entropy
        self.log_alpha = take.log_alpha
        self.log_std_min = take.log_std_min
        self.log_std_max = take.log_std_max
        self.actor_mean_weights = take.actor_mean_weights^
        self.actor_logstd_weights = take.actor_logstd_weights^
        self.critic1_weights = take.critic1_weights^
        self.target_critic1_weights = take.target_critic1_weights^
        self.critic2_weights = take.critic2_weights^
        self.target_critic2_weights = take.target_critic2_weights^

    # ========================================================================
    # Actor (Stochastic Policy) Methods
    # ========================================================================

    def _compute_mean_logstd(
        self, features: List[Float64]
    ) -> Tuple[Float64, Float64]:
        """Compute policy mean and log_std from features.

        Args:
            features: State feature vector φ(s)

        Returns:
            Tuple of (raw_mean, clamped_log_std)
        """
        var mean: Float64 = 0.0
        var log_std: Float64 = 0.0
        var n = min(len(features), self.num_state_features)

        for i in range(n):
            mean += self.actor_mean_weights[i] * features[i]
            log_std += self.actor_logstd_weights[i] * features[i]

        # Clamp log_std to prevent numerical issues
        if log_std < self.log_std_min:
            log_std = self.log_std_min
        elif log_std > self.log_std_max:
            log_std = self.log_std_max

        return (mean, log_std)

    def select_action(self, features: List[Float64]) -> Float64:
        """Sample action from stochastic policy.

        π(a|s) = tanh(μ(s) + σ(s) * ε) * action_scale, where ε ~ N(0, 1)

        Args:
            features: State feature vector φ(s).

        Returns:
            Sampled action in [-action_scale, action_scale].
        """
        var mean_logstd = self._compute_mean_logstd(features)
        var mean = mean_logstd[0]
        var log_std = mean_logstd[1]
        var std = exp(log_std)

        # Sample from Gaussian using reparameterization
        var noise = gaussian_noise()
        var raw_action = mean + std * noise

        # Apply tanh squashing and scale
        var action = tanh(raw_action) * self.action_scale

        return action

    def select_action_deterministic(self, features: List[Float64]) -> Float64:
        """Select deterministic action (mean of policy) for evaluation.

        Args:
            features: State feature vector φ(s).

        Returns:
            Deterministic action (tanh(mean) * scale).
        """
        var mean_logstd = self._compute_mean_logstd(features)
        var mean = mean_logstd[0]
        return tanh(mean) * self.action_scale

    def _sample_action_with_log_prob(
        self, features: List[Float64]
    ) -> Tuple[Float64, Float64]:
        """Sample action and compute log probability (for policy gradient).

        Uses reparameterization trick and computes log prob with tanh correction:
        log π(a|s) = log N(u; μ, σ²) - log(1 - tanh²(u))

        where u is the pre-squashed action.

        Args:
            features: State feature vector φ(s).

        Returns:
            Tuple of (action, log_probability).
        """
        var mean_logstd = self._compute_mean_logstd(features)
        var mean = mean_logstd[0]
        var log_std = mean_logstd[1]
        var std = exp(log_std)

        # Sample from Gaussian
        var noise = gaussian_noise()
        var raw_action = mean + std * noise

        # Compute log prob of Gaussian
        # log N(u; μ, σ²) = -0.5 * ((u - μ)/σ)² - log(σ) - 0.5 * log(2π)
        var log_prob = (
            -0.5 * noise * noise - log_std - 0.5 * 0.9189385332
        )  # 0.5 * log(2π)

        # Apply tanh squashing
        var tanh_action = tanh(raw_action)
        var action = tanh_action * self.action_scale

        # Correction for tanh squashing: subtract log|det(d tanh / d u)|
        # log(1 - tanh²(u)) with numerical stability
        var tanh_sq = tanh_action * tanh_action
        if tanh_sq > 0.999999:
            tanh_sq = 0.999999  # Numerical stability
        log_prob -= log(1.0 - tanh_sq + 1e-6)

        return (action, log_prob)

    # ========================================================================
    # Critic (Q-function) Methods
    # ========================================================================

    def _build_critic_features(
        self, state_features: List[Float64], action: Float64
    ) -> List[Float64]:
        """Build critic input features.

        Critic features = [φ(s); a_norm; a_norm²]
        """
        var critic_features = List[Float64]()

        for i in range(len(state_features)):
            critic_features.append(state_features[i])

        var a_normalized = action / self.action_scale
        critic_features.append(a_normalized)
        critic_features.append(a_normalized * a_normalized)

        return critic_features^

    def get_q1_value(
        self, state_features: List[Float64], action: Float64
    ) -> Float64:
        """Compute Q-value using first critic."""
        var critic_features = self._build_critic_features(
            state_features, action
        )
        var q_value: Float64 = 0.0
        var n = min(len(critic_features), len(self.critic1_weights))
        for i in range(n):
            q_value += self.critic1_weights[i] * critic_features[i]
        return q_value

    def get_q2_value(
        self, state_features: List[Float64], action: Float64
    ) -> Float64:
        """Compute Q-value using second critic."""
        var critic_features = self._build_critic_features(
            state_features, action
        )
        var q_value: Float64 = 0.0
        var n = min(len(critic_features), len(self.critic2_weights))
        for i in range(n):
            q_value += self.critic2_weights[i] * critic_features[i]
        return q_value

    def _get_q1_value_target(
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

    def _get_q2_value_target(
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

    def get_min_q_value(
        self, state_features: List[Float64], action: Float64
    ) -> Float64:
        """Get minimum Q-value from twin critics."""
        var q1 = self.get_q1_value(state_features, action)
        var q2 = self.get_q2_value(state_features, action)
        if q1 < q2:
            return q1
        return q2

    # ========================================================================
    # Update Methods
    # ========================================================================

    def update(mut self, batch: List[ContinuousTransition[DType.float64]]):
        """Update critics, actor, and optionally alpha from a batch.

        SAC update procedure:
        1. Update critics using soft Bellman residual
        2. Update actor to maximize Q - α * log π
        3. Update α to maintain target entropy (if auto_alpha)
        4. Soft update target networks

        Args:
            batch: List of ContinuousTransition objects.
        """
        if len(batch) == 0:
            return

        # Update critics
        self._update_critics(batch)

        # Update actor
        self._update_actor(batch)

        # Update alpha (if automatic tuning enabled)
        if self.auto_alpha:
            self._update_alpha(batch)

        # Soft update target networks
        self._soft_update_targets()

    def _update_critics(
        mut self, batch: List[ContinuousTransition[DType.float64]]
    ):
        """Update both critics using soft Bellman residual.

        Target: y = r + γ * (min(Q1_target, Q2_target)(s', a') - α * log π(a'|s'))
        where a' ~ π(·|s')
        """
        var batch_size = len(batch)
        var step_size = self.critic_lr / Float64(batch_size)

        for i in range(batch_size):
            var transition = batch[i]

            # Compute TD target
            var target: Float64
            if transition.done:
                target = transition.reward
            else:
                # Sample next action from current policy
                var next_action_logprob = self._sample_action_with_log_prob(
                    transition.next_state
                )
                var next_action = next_action_logprob[0]
                var next_log_prob = next_action_logprob[1]

                # Min of target Q-values
                var next_q1 = self._get_q1_value_target(
                    transition.next_state, next_action
                )
                var next_q2 = self._get_q2_value_target(
                    transition.next_state, next_action
                )
                var next_q = next_q1
                if next_q2 < next_q1:
                    next_q = next_q2

                # Soft Bellman target with entropy
                target = transition.reward + self.discount_factor * (
                    next_q - self.alpha * next_log_prob
                )

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

    def _update_actor(
        mut self, batch: List[ContinuousTransition[DType.float64]]
    ):
        """Update actor to maximize expected Q-value minus entropy.

        Objective: max E[Q(s, a) - α * log π(a|s)] where a ~ π(·|s)

        For linear actor with Gaussian policy and tanh squashing:
        ∇_θ J ≈ (1/N) Σ [∇_a (Q(s,a) - α * log π) * ∇_θ a]
        """
        var batch_size = len(batch)
        var step_size = self.actor_lr / Float64(batch_size)

        for i in range(batch_size):
            var transition = batch[i]

            # Get policy parameters
            var mean_logstd = self._compute_mean_logstd(transition.state)
            var mean = mean_logstd[0]
            var log_std = mean_logstd[1]
            var std = exp(log_std)

            # Sample action with reparameterization
            var noise = gaussian_noise()
            var raw_action = mean + std * noise
            var tanh_action = tanh(raw_action)
            var action = tanh_action * self.action_scale

            # Compute log probability
            var log_prob = -0.5 * noise * noise - log_std - 0.5 * 0.9189385332
            var tanh_sq = tanh_action * tanh_action
            if tanh_sq > 0.999999:
                tanh_sq = 0.999999
            log_prob -= log(1.0 - tanh_sq + 1e-6)

            # Use Q1 for actor update (like TD3)
            # Compute ∇_a Q1(s, a)
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

            # Gradient of action w.r.t. mean (through tanh)
            # a = tanh(mean + std * noise) * scale
            # ∂a/∂mean = scale * (1 - tanh²) = scale * (1 - tanh_sq)
            var grad_a_mean = self.action_scale * (1.0 - tanh_sq)

            # Gradient of log_prob w.r.t. mean
            # log_prob = -0.5 * ((raw - mean)/std)² - log_std - correction
            # ∂log_prob/∂mean = noise / std (since raw = mean + std*noise)
            # But with tanh correction, it's more complex. Approximate:
            var grad_logprob_mean = noise / std

            # Gradient of log_prob w.r.t. log_std
            # ∂log_prob/∂log_std = noise² - 1 (approximately, ignoring tanh correction)
            var grad_logprob_logstd = noise * noise - 1.0

            # Combined gradient for mean weights
            # We want to maximize Q - α * log_prob
            # ∇_θ (Q - α * log_prob) = ∇_a Q * ∂a/∂mean * ∂mean/∂θ - α * ∂log_prob/∂mean * ∂mean/∂θ
            for j in range(len(transition.state)):
                if j < len(self.actor_mean_weights):
                    var grad_mean = (
                        grad_a_q * grad_a_mean - self.alpha * grad_logprob_mean
                    )
                    self.actor_mean_weights[j] += (
                        step_size * grad_mean * transition.state[j]
                    )

            # Update log_std weights
            # Gradient for log_std: maximize entropy means minimize -log_prob
            # But we also want actions that maximize Q
            # Gradient through action: ∂a/∂log_std = scale * (1 - tanh²) * noise * std
            var grad_a_logstd = (
                self.action_scale * (1.0 - tanh_sq) * noise * std
            )

            for j in range(len(transition.state)):
                if j < len(self.actor_logstd_weights):
                    var grad_logstd = (
                        grad_a_q * grad_a_logstd
                        - self.alpha * grad_logprob_logstd
                    )
                    self.actor_logstd_weights[j] += (
                        step_size * grad_logstd * transition.state[j]
                    )

    def _update_alpha(
        mut self, batch: List[ContinuousTransition[DType.float64]]
    ):
        """Update entropy coefficient α to maintain target entropy.

        Objective: min α * E[-log π(a|s) - target_entropy]

        If entropy is too low (log_prob too high), increase α.
        If entropy is too high (log_prob too low), decrease α.
        """
        var batch_size = len(batch)
        var avg_log_prob: Float64 = 0.0

        for i in range(batch_size):
            var transition = batch[i]
            var action_logprob = self._sample_action_with_log_prob(
                transition.state
            )
            avg_log_prob += action_logprob[1]

        avg_log_prob /= Float64(batch_size)

        # Gradient: ∂L/∂log_α = -log_prob - target_entropy
        # We want to minimize α * (log_prob + target_entropy)
        var alpha_grad = -(avg_log_prob + self.target_entropy)
        self.log_alpha += self.alpha_lr * alpha_grad

        # Clamp log_alpha to reasonable range
        if self.log_alpha < -10.0:
            self.log_alpha = -10.0
        elif self.log_alpha > 2.0:
            self.log_alpha = 2.0

        self.alpha = exp(self.log_alpha)

    def _soft_update_targets(mut self):
        """Soft update target critic networks.

        θ_target = τ * θ + (1 - τ) * θ_target

        Note: SAC doesn't use a target actor.
        """
        for i in range(len(self.critic1_weights)):
            self.target_critic1_weights[i] = (
                self.tau * self.critic1_weights[i]
                + (1.0 - self.tau) * self.target_critic1_weights[i]
            )

        for i in range(len(self.critic2_weights)):
            self.target_critic2_weights[i] = (
                self.tau * self.critic2_weights[i]
                + (1.0 - self.tau) * self.target_critic2_weights[i]
            )

    def reset(mut self):
        """Reset for new episode (no-op for SAC)."""
        pass

    # ========================================================================
    # Training and Evaluation
    # ========================================================================

    def train[
        E: BoxContinuousActionEnv
    ](
        mut self,
        mut env: E,
        features: PolynomialFeatures[DType.float64],
        mut buffer: ContinuousReplayBuffer[DType.float64],
        num_episodes: Int,
        max_steps_per_episode: Int = 200,
        batch_size: Int = 64,
        min_buffer_size: Int = 1000,
        warmup_episodes: Int = 10,
        verbose: Bool = False,
        print_every: Int = 10,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train the SAC agent on a continuous control environment.

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
            algorithm_name="SAC (Linear FA)",
            environment_name=environment_name,
        )

        if verbose:
            print("=" * 60)
            print("SAC Training on", environment_name)
            print("=" * 60)
            print("State features:", self.num_state_features)
            print("Critic features:", self.num_critic_features)
            print("Action scale:", self.action_scale)
            print("Actor LR:", self.actor_lr, "Critic LR:", self.critic_lr)
            print("Tau:", self.tau, "Initial alpha:", self.alpha)
            print(
                "Auto alpha:",
                self.auto_alpha,
                "Target entropy:",
                self.target_entropy,
            )
            print("Warmup episodes:", warmup_episodes)
            print("Batch size:", batch_size)
            print("-" * 60)

        var total_steps = 0

        for episode in range(num_episodes):
            var obs_list = env.reset_obs_list()
            var obs = _list_to_simd4_f64(obs_list)
            var episode_reward: Float64 = 0.0
            var steps = 0

            for _ in range(max_steps_per_episode):
                var state_features = features.get_features_simd4(obs)

                # Select action
                var action: Float64
                if episode < warmup_episodes:
                    action = (random_float64() * 2.0 - 1.0) * self.action_scale
                else:
                    action = self.select_action(state_features)

                # Take action
                var result = env.step_continuous_vec[DType.float64]([action])
                var next_obs_list = result[0].copy()
                var next_obs = _list_to_simd4_f64(next_obs_list)
                var reward = Float64(result[1])
                var done = result[2]

                var next_features = features.get_features_simd4(next_obs)
                buffer.push(state_features, action, reward, next_features, done)

                # Update if enough samples
                if (
                    buffer.len() >= min_buffer_size
                    and episode >= warmup_episodes
                ):
                    var batch = buffer.sample(batch_size)
                    self.update(batch)

                episode_reward += reward
                steps += 1
                total_steps += 1
                obs = next_obs

                if done:
                    break

            metrics.log_episode(episode, episode_reward, steps, self.alpha)

            if verbose and (episode + 1) % print_every == 0:
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
                    "| Alpha:",
                    String(self.alpha)[:6],
                    "| Buffer:",
                    buffer.len(),
                )

        if verbose:
            print("-" * 60)
            print("Training complete!")
            var start_idx = max(0, len(metrics.episodes) - 100)
            var sum_reward: Float64 = 0.0
            for j in range(start_idx, len(metrics.episodes)):
                sum_reward += metrics.episodes[j].total_reward
            var final_avg = sum_reward / Float64(
                len(metrics.episodes) - start_idx
            )
            print("Final avg reward:", String(final_avg)[:8])
            print("Final alpha:", String(self.alpha)[:6])

        return metrics^

    def evaluate[
        E: BoxContinuousActionEnv & RenderableEnv
    ](
        self,
        mut env: E,
        features: PolynomialFeatures[DType.float64],
        num_episodes: Int = 10,
        max_steps_per_episode: Int = 200,
        render: Bool = False,
        frame_delay_ms: Int = 16,
    ) raises -> Float64:
        """Evaluate the trained SAC agent using deterministic policy.

        Args:
            env: Environment implementing ContinuousControlEnv trait.
            features: PolynomialFeatures extractor for state representation.
            num_episodes: Number of evaluation episodes.
            max_steps_per_episode: Maximum steps per episode.
            render: Whether to render the environment (default: False).
            frame_delay_ms: Delay between frames in milliseconds (default: 16).

        Returns:
            Average reward over evaluation episodes.
        """
        var total_reward: Float64 = 0.0
        var quit_requested = False

        if render:
            _ = env.init_renderer()

        for _ in range(num_episodes):
            if quit_requested:
                break

            var obs_list = env.reset_obs_list()
            var obs = _list_to_simd4_f64(obs_list)
            var episode_reward: Float64 = 0.0

            for _ in range(max_steps_per_episode):
                var state_features = features.get_features_simd4(obs)
                var action = self.select_action_deterministic(state_features)

                var result = env.step_continuous(action)
                var next_obs_list = result[0].copy()
                var next_obs = _list_to_simd4_f64(next_obs_list)
                var reward = Float64(result[1])
                var done = result[2]

                if render:
                    env.render_frame()
                    env.renderer_delay(frame_delay_ms)
                    if env.check_renderer_quit():
                        quit_requested = True
                        break

                episode_reward += reward
                obs = next_obs

                if done:
                    break

            total_reward += episode_reward

        if render:
            env.close_renderer()

        return total_reward / Float64(num_episodes)


# ============================================================================
# Helper functions
# ============================================================================


def _list_to_simd4[DTYPE: DType](obs: List[Scalar[DTYPE]]) -> SIMD[DTYPE, 4]:
    """Convert a List[Scalar[DTYPE]] to SIMD[DTYPE, 4].

    Pads with zeros if the list has fewer than 4 elements.
    """
    var result = SIMD[DTYPE, 4](0.0)
    var n = min(len(obs), 4)
    for i in range(n):
        result[i] = obs[i]
    return result


def _list_to_simd4_f64[
    DTYPE: DType
](obs: List[Scalar[DTYPE]]) -> SIMD[DType.float64, 4]:
    """Convert a List[Scalar[DTYPE]] to SIMD[DType.float64, 4].

    Pads with zeros if the list has fewer than 4 elements.
    Casts each element to Float64.
    """
    var result = SIMD[DType.float64, 4](0.0)
    var n = min(len(obs), 4)
    for i in range(n):
        result[i] = Float64(obs[i])
    return result

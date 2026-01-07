"""Linear Q-Learning Agent with Arbitrary Feature Vectors.

This module provides Q-Learning and SARSA agents that use linear function
approximation with arbitrary feature representations (polynomial, RBF, or custom).

Unlike tile coding which uses sparse binary features, these agents work with
dense real-valued feature vectors, allowing more flexible representations.

Key differences from TiledQLearningAgent:
- Works with any feature extractor, not just tile coding
- Dense feature vectors instead of sparse tile indices
- Supports gradient-based updates with arbitrary features
- More general but potentially less sample-efficient than tile coding

Example usage:
    from core.linear_fa import PolynomialFeatures, LinearWeights
    from agents.linear_qlearning import LinearQLearningAgent

    # Create polynomial feature extractor for CartPole
    var features = PolynomialFeatures(state_dim=4, degree=2)

    # Create agent
    var agent = LinearQLearningAgent(
        num_features=features.get_num_features(),
        num_actions=2,
        learning_rate=0.01,
    )

    # Training loop
    var phi = features.get_features_simd4(obs)
    var action = agent.select_action(phi)
    # ... environment step ...
    var phi_next = features.get_features_simd4(next_obs)
    agent.update(phi, action, reward, phi_next, done)

References:
- Sutton & Barto, Chapter 9.3: "Linear Methods"
- Sutton & Barto, Chapter 10.1: "Episodic Semi-gradient Control"
"""

from random import random_float64, random_si64
from core.linear_fa import LinearWeights
from core import BoxDiscreteActionEnv, TrainingMetrics, FeatureExtractor


struct LinearQLearningAgent:
    """Q-Learning agent with linear function approximation.

    Uses arbitrary feature vectors for state representation.
    Q(s, a) = w[a]^T * φ(s)

    Update rule (semi-gradient Q-learning):
        w[a] += α * (r + γ * max_a' Q(s', a') - Q(s, a)) * φ(s)
    """

    var weights: LinearWeights
    var num_actions: Int
    var num_features: Int
    var learning_rate: Float64
    var discount_factor: Float64
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64

    fn __init__(
        out self,
        num_features: Int,
        num_actions: Int,
        learning_rate: Float64 = 0.01,
        discount_factor: Float64 = 0.99,
        epsilon: Float64 = 1.0,
        epsilon_decay: Float64 = 0.995,
        epsilon_min: Float64 = 0.01,
        init_std: Float64 = 0.01,
    ):
        """Initialize linear Q-learning agent.

        Args:
            num_features: Dimensionality of feature vectors
            num_actions: Number of discrete actions
            learning_rate: Learning rate α (typically smaller than tabular, e.g., 0.01)
            discount_factor: Discount factor γ
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay per episode
            epsilon_min: Minimum epsilon
            init_std: Standard deviation for weight initialization
        """
        self.num_features = num_features
        self.num_actions = num_actions
        self.weights = LinearWeights(
            num_features=num_features,
            num_actions=num_actions,
            init_std=init_std,
        )
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    fn select_action(self, features: List[Float64]) -> Int:
        """Select action using epsilon-greedy policy.

        Args:
            features: Feature vector φ(s)

        Returns:
            Selected action index
        """
        if random_float64() < self.epsilon:
            # Explore: random action
            return Int(random_si64(0, self.num_actions - 1))
        else:
            # Exploit: best action
            return self.weights.get_best_action(features)

    fn get_best_action(self, features: List[Float64]) -> Int:
        """Get greedy action (no exploration).

        Args:
            features: Feature vector φ(s)

        Returns:
            Action with highest Q-value
        """
        return self.weights.get_best_action(features)

    fn get_value(self, features: List[Float64], action: Int) -> Float64:
        """Get Q-value for state-action pair.

        Args:
            features: Feature vector φ(s)
            action: Action index

        Returns:
            Q(s, a)
        """
        return self.weights.get_value(features, action)

    fn get_max_value(self, features: List[Float64]) -> Float64:
        """Get maximum Q-value over all actions.

        Args:
            features: Feature vector φ(s)

        Returns:
            max_a Q(s, a)
        """
        return self.weights.get_max_value(features)

    fn update(
        mut self,
        features: List[Float64],
        action: Int,
        reward: Float64,
        next_features: List[Float64],
        done: Bool,
    ):
        """Update weights using semi-gradient Q-learning.

        TD target: r + γ * max_a' Q(s', a') for non-terminal
                   r for terminal

        Update: w[a] += α * (target - Q(s,a)) * φ(s)

        Args:
            features: Feature vector for current state
            action: Action taken
            reward: Reward received
            next_features: Feature vector for next state
            done: Whether episode terminated
        """
        var target: Float64
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * self.get_max_value(next_features)

        self.weights.update(features, action, target, self.learning_rate)

    fn decay_epsilon(mut self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    fn get_epsilon(self) -> Float64:
        """Return current exploration rate."""
        return self.epsilon

    fn reset(mut self):
        """Reset for new episode (no-op for Q-learning)."""
        pass

    fn train[E: BoxDiscreteActionEnv, F: FeatureExtractor](
        mut self,
        mut env: E,
        features: F,
        num_episodes: Int,
        max_steps_per_episode: Int = 500,
        verbose: Bool = False,
        print_every: Int = 100,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train the linear Q-learning agent on a continuous-state environment.

        Args:
            env: Environment implementing BoxDiscreteActionEnv trait.
            features: Feature extractor implementing FeatureExtractor trait.
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
            verbose: Whether to print progress.
            print_every: Print frequency when verbose.
            environment_name: Name for logging.

        Returns:
            Training metrics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="Linear Q-Learning",
            environment_name=environment_name,
        )

        for episode in range(num_episodes):
            var obs = env.reset_obs()
            var total_reward: Float64 = 0.0
            var steps = 0

            for _ in range(max_steps_per_episode):
                var phi = features.get_features_simd4(obs)
                var action = self.select_action(phi)

                var result = env.step_raw(action)
                var next_obs = result[0]
                var reward = result[1]
                var done = result[2]

                var phi_next = features.get_features_simd4(next_obs)
                self.update(phi, action, reward, phi_next, done)

                total_reward += reward
                steps += 1
                obs = next_obs
                if done:
                    break

            self.decay_epsilon()
            metrics.log_episode(episode, total_reward, steps, self.epsilon)
            if verbose and (episode + 1) % print_every == 0:
                metrics.print_progress(episode, window=100)

        return metrics^

    fn evaluate[E: BoxDiscreteActionEnv, F: FeatureExtractor](
        self,
        mut env: E,
        features: F,
        num_episodes: Int = 100,
        max_steps_per_episode: Int = 500,
    ) -> Float64:
        """Evaluate the linear Q-learning agent using greedy policy.

        Args:
            env: Environment implementing BoxDiscreteActionEnv trait.
            features: Feature extractor implementing FeatureExtractor trait.
            num_episodes: Number of evaluation episodes.
            max_steps_per_episode: Maximum steps per episode.

        Returns:
            Average reward over evaluation episodes.
        """
        var total_reward: Float64 = 0.0

        for _ in range(num_episodes):
            var obs = env.reset_obs()
            var episode_reward: Float64 = 0.0

            for _ in range(max_steps_per_episode):
                var phi = features.get_features_simd4(obs)
                var action = self.get_best_action(phi)

                var result = env.step_raw(action)
                var next_obs = result[0]
                var reward = result[1]
                var done = result[2]

                episode_reward += reward
                obs = next_obs
                if done:
                    break

            total_reward += episode_reward

        return total_reward / Float64(num_episodes)


struct LinearSARSAAgent:
    """SARSA agent with linear function approximation.

    On-policy variant using actual next action:
        w[a] += α * (r + γ * Q(s', a') - Q(s, a)) * φ(s)
    """

    var weights: LinearWeights
    var num_actions: Int
    var num_features: Int
    var learning_rate: Float64
    var discount_factor: Float64
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64

    fn __init__(
        out self,
        num_features: Int,
        num_actions: Int,
        learning_rate: Float64 = 0.01,
        discount_factor: Float64 = 0.99,
        epsilon: Float64 = 1.0,
        epsilon_decay: Float64 = 0.995,
        epsilon_min: Float64 = 0.01,
        init_std: Float64 = 0.01,
    ):
        """Initialize linear SARSA agent."""
        self.num_features = num_features
        self.num_actions = num_actions
        self.weights = LinearWeights(
            num_features=num_features,
            num_actions=num_actions,
            init_std=init_std,
        )
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    fn select_action(self, features: List[Float64]) -> Int:
        """Select action using epsilon-greedy policy."""
        if random_float64() < self.epsilon:
            return Int(random_si64(0, self.num_actions - 1))
        else:
            return self.weights.get_best_action(features)

    fn get_best_action(self, features: List[Float64]) -> Int:
        """Get greedy action."""
        return self.weights.get_best_action(features)

    fn get_value(self, features: List[Float64], action: Int) -> Float64:
        """Get Q-value for state-action pair."""
        return self.weights.get_value(features, action)

    fn update(
        mut self,
        features: List[Float64],
        action: Int,
        reward: Float64,
        next_features: List[Float64],
        next_action: Int,
        done: Bool,
    ):
        """Update weights using semi-gradient SARSA.

        Args:
            features: Feature vector for current state
            action: Action taken
            reward: Reward received
            next_features: Feature vector for next state
            next_action: Actual action selected for next state
            done: Whether episode terminated
        """
        var target: Float64
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * self.get_value(next_features, next_action)

        self.weights.update(features, action, target, self.learning_rate)

    fn decay_epsilon(mut self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    fn get_epsilon(self) -> Float64:
        """Return current exploration rate."""
        return self.epsilon

    fn reset(mut self):
        """Reset for new episode."""
        pass

    fn train[E: BoxDiscreteActionEnv, F: FeatureExtractor](
        mut self,
        mut env: E,
        features: F,
        num_episodes: Int,
        max_steps_per_episode: Int = 500,
        verbose: Bool = False,
        print_every: Int = 100,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train the linear SARSA agent on a continuous-state environment.

        Args:
            env: Environment implementing BoxDiscreteActionEnv trait.
            features: Feature extractor implementing FeatureExtractor trait.
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
            verbose: Whether to print progress.
            print_every: Print frequency when verbose.
            environment_name: Name for logging.

        Returns:
            Training metrics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="Linear SARSA",
            environment_name=environment_name,
        )

        for episode in range(num_episodes):
            var obs = env.reset_obs()
            var action = self.select_action(features.get_features_simd4(obs))
            var total_reward: Float64 = 0.0
            var steps = 0

            for _ in range(max_steps_per_episode):
                var phi = features.get_features_simd4(obs)

                var result = env.step_raw(action)
                var next_obs = result[0]
                var reward = result[1]
                var done = result[2]

                var phi_next = features.get_features_simd4(next_obs)
                var next_action = self.select_action(phi_next)

                self.update(phi, action, reward, phi_next, next_action, done)

                total_reward += reward
                steps += 1
                obs = next_obs
                action = next_action
                if done:
                    break

            self.decay_epsilon()
            metrics.log_episode(episode, total_reward, steps, self.epsilon)
            if verbose and (episode + 1) % print_every == 0:
                metrics.print_progress(episode, window=100)

        return metrics^

    fn evaluate[E: BoxDiscreteActionEnv, F: FeatureExtractor](
        self,
        mut env: E,
        features: F,
        num_episodes: Int = 100,
        max_steps_per_episode: Int = 500,
    ) -> Float64:
        """Evaluate the linear SARSA agent using greedy policy.

        Args:
            env: Environment implementing BoxDiscreteActionEnv trait.
            features: Feature extractor implementing FeatureExtractor trait.
            num_episodes: Number of evaluation episodes.
            max_steps_per_episode: Maximum steps per episode.

        Returns:
            Average reward over evaluation episodes.
        """
        var total_reward: Float64 = 0.0

        for _ in range(num_episodes):
            var obs = env.reset_obs()
            var episode_reward: Float64 = 0.0

            for _ in range(max_steps_per_episode):
                var phi = features.get_features_simd4(obs)
                var action = self.get_best_action(phi)

                var result = env.step_raw(action)
                var next_obs = result[0]
                var reward = result[1]
                var done = result[2]

                episode_reward += reward
                obs = next_obs
                if done:
                    break

            total_reward += episode_reward

        return total_reward / Float64(num_episodes)


struct LinearSARSALambdaAgent:
    """SARSA(λ) agent with linear function approximation and eligibility traces.

    Combines linear function approximation with eligibility traces for
    faster credit assignment over long episodes.

    Uses accumulating traces:
        e += φ(s)  (for current state-action)
        e *= γλ    (decay all traces)
        w += αδe   (update all weights)
    """

    var weights: LinearWeights
    var traces: List[List[Float64]]  # Eligibility traces [action][feature]
    var num_actions: Int
    var num_features: Int
    var learning_rate: Float64
    var discount_factor: Float64
    var lambda_: Float64
    var epsilon: Float64
    var epsilon_decay: Float64
    var epsilon_min: Float64

    fn __init__(
        out self,
        num_features: Int,
        num_actions: Int,
        learning_rate: Float64 = 0.01,
        discount_factor: Float64 = 0.99,
        lambda_: Float64 = 0.9,
        epsilon: Float64 = 1.0,
        epsilon_decay: Float64 = 0.995,
        epsilon_min: Float64 = 0.01,
        init_std: Float64 = 0.01,
    ):
        """Initialize linear SARSA(λ) agent.

        Args:
            num_features: Dimensionality of feature vectors
            num_actions: Number of actions
            learning_rate: Learning rate α
            discount_factor: Discount γ
            lambda_: Eligibility trace decay (0 = SARSA, 1 = Monte Carlo)
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay per episode
            epsilon_min: Minimum epsilon
            init_std: Weight initialization std
        """
        self.num_features = num_features
        self.num_actions = num_actions
        self.weights = LinearWeights(
            num_features=num_features,
            num_actions=num_actions,
            init_std=init_std,
        )
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize eligibility traces to zero
        self.traces = List[List[Float64]]()
        for _ in range(num_actions):
            var action_traces = List[Float64]()
            for _ in range(num_features):
                action_traces.append(0.0)
            self.traces.append(action_traces^)

    fn select_action(self, features: List[Float64]) -> Int:
        """Select action using epsilon-greedy policy."""
        if random_float64() < self.epsilon:
            return Int(random_si64(0, self.num_actions - 1))
        else:
            return self.weights.get_best_action(features)

    fn get_best_action(self, features: List[Float64]) -> Int:
        """Get greedy action."""
        return self.weights.get_best_action(features)

    fn get_value(self, features: List[Float64], action: Int) -> Float64:
        """Get Q-value for state-action pair."""
        return self.weights.get_value(features, action)

    fn update(
        mut self,
        features: List[Float64],
        action: Int,
        reward: Float64,
        next_features: List[Float64],
        next_action: Int,
        done: Bool,
    ):
        """Update weights and traces using SARSA(λ).

        Steps:
        1. Compute TD error: δ = r + γ * Q(s', a') - Q(s, a)
        2. Accumulate traces: e[a] += φ(s)
        3. Update weights: w += α * δ * e
        4. Decay traces: e *= γ * λ
        """
        # Compute TD error
        var current_value = self.weights.get_value(features, action)
        var next_value: Float64 = 0.0
        if not done:
            next_value = self.weights.get_value(next_features, next_action)
        var td_error = reward + self.discount_factor * next_value - current_value

        # Accumulate traces for current state-action
        for i in range(self.num_features):
            self.traces[action][i] += features[i]

        # Update weights using traces
        for a in range(self.num_actions):
            for i in range(self.num_features):
                self.weights.weights[a][i] += self.learning_rate * td_error * self.traces[a][i]

        # Decay traces
        var decay = self.discount_factor * self.lambda_
        for a in range(self.num_actions):
            for i in range(self.num_features):
                self.traces[a][i] *= decay

    fn decay_epsilon(mut self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    fn get_epsilon(self) -> Float64:
        """Return current exploration rate."""
        return self.epsilon

    fn reset(mut self):
        """Reset eligibility traces for new episode."""
        for a in range(self.num_actions):
            for i in range(self.num_features):
                self.traces[a][i] = 0.0

    fn train[E: BoxDiscreteActionEnv, F: FeatureExtractor](
        mut self,
        mut env: E,
        features: F,
        num_episodes: Int,
        max_steps_per_episode: Int = 500,
        verbose: Bool = False,
        print_every: Int = 100,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train the linear SARSA(λ) agent on a continuous-state environment.

        Args:
            env: Environment implementing BoxDiscreteActionEnv trait.
            features: Feature extractor implementing FeatureExtractor trait.
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
            verbose: Whether to print progress.
            print_every: Print frequency when verbose.
            environment_name: Name for logging.

        Returns:
            Training metrics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="Linear SARSA(λ)",
            environment_name=environment_name,
        )

        for episode in range(num_episodes):
            self.reset()  # Reset eligibility traces
            var obs = env.reset_obs()
            var action = self.select_action(features.get_features_simd4(obs))
            var total_reward: Float64 = 0.0
            var steps = 0

            for _ in range(max_steps_per_episode):
                var phi = features.get_features_simd4(obs)

                var result = env.step_raw(action)
                var next_obs = result[0]
                var reward = result[1]
                var done = result[2]

                var phi_next = features.get_features_simd4(next_obs)
                var next_action = self.select_action(phi_next)

                self.update(phi, action, reward, phi_next, next_action, done)

                total_reward += reward
                steps += 1
                obs = next_obs
                action = next_action
                if done:
                    break

            self.decay_epsilon()
            metrics.log_episode(episode, total_reward, steps, self.epsilon)
            if verbose and (episode + 1) % print_every == 0:
                metrics.print_progress(episode, window=100)

        return metrics^

    fn evaluate[E: BoxDiscreteActionEnv, F: FeatureExtractor](
        self,
        mut env: E,
        features: F,
        num_episodes: Int = 100,
        max_steps_per_episode: Int = 500,
    ) -> Float64:
        """Evaluate the linear SARSA(λ) agent using greedy policy.

        Args:
            env: Environment implementing BoxDiscreteActionEnv trait.
            features: Feature extractor implementing FeatureExtractor trait.
            num_episodes: Number of evaluation episodes.
            max_steps_per_episode: Maximum steps per episode.

        Returns:
            Average reward over evaluation episodes.
        """
        var total_reward: Float64 = 0.0

        for _ in range(num_episodes):
            var obs = env.reset_obs()
            var episode_reward: Float64 = 0.0

            for _ in range(max_steps_per_episode):
                var phi = features.get_features_simd4(obs)
                var action = self.get_best_action(phi)

                var result = env.step_raw(action)
                var next_obs = result[0]
                var reward = result[1]
                var done = result[2]

                episode_reward += reward
                obs = next_obs
                if done:
                    break

            total_reward += episode_reward

        return total_reward / Float64(num_episodes)
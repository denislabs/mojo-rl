"""REINFORCE (Monte Carlo Policy Gradient) Agent.

REINFORCE is the simplest policy gradient algorithm. It learns a parameterized
policy directly by performing gradient ascent on the expected return.

The policy is represented using a softmax over linear features (tile coding):
    π(a|s) = exp(θ_a · φ(s)) / Σ_a' exp(θ_a' · φ(s))

where φ(s) are the active tiles for state s.

Update rule (at end of episode):
    θ += α * G_t * ∇log π(a_t|s_t)

For softmax policy with linear features:
    ∇log π(a|s) = φ(s) - Σ_a' π(a'|s) * φ(s) = φ(s) - E[φ(s)]

Since φ(s) is the same for all actions (tile coding), this simplifies to:
    ∇log π(a|s)[a] = 1 - π(a|s)  (for chosen action)
    ∇log π(a|s)[a'] = -π(a'|s)   (for other actions)

References:
- Sutton & Barto, Chapter 13: "Policy Gradient Methods"
- Williams (1992): "Simple Statistical Gradient-Following Algorithms"

Example usage:
    from core.tile_coding import make_cartpole_tile_coding
    from agents.reinforce import REINFORCEAgent

    var tc = make_cartpole_tile_coding(num_tilings=8, tiles_per_dim=8)
    var agent = REINFORCEAgent(
        tile_coding=tc,
        num_actions=2,
        learning_rate=0.001,
    )

    # Training loop
    var tiles = tc.get_tiles_simd4(obs)
    var action = agent.select_action(tiles)
    # ... environment step ...
    agent.store_transition(tiles, action, reward)
    if done:
        agent.update_from_episode()
"""

from math import exp, log
from random import random_float64
from core.tile_coding import TileCoding
from core import BoxDiscreteActionEnv, TrainingMetrics


struct REINFORCEAgent(Copyable, Movable, ImplicitlyCopyable):
    """REINFORCE agent with tile coding function approximation.

    Uses softmax policy over tile-coded features.
    Updates policy parameters at the end of each episode using
    Monte Carlo returns.
    """

    # Policy parameters: θ[action][tile]
    var theta: List[List[Float64]]
    var num_actions: Int
    var num_tiles: Int
    var num_tilings: Int
    var learning_rate: Float64
    var discount_factor: Float64

    # Episode storage for Monte Carlo update
    var episode_tiles: List[List[Int]]
    var episode_actions: List[Int]
    var episode_rewards: List[Float64]

    # Optional baseline for variance reduction
    var use_baseline: Bool
    var baseline_weights: List[Float64]  # V(s) approximation
    var baseline_lr: Float64

    fn __init__(
        out self,
        tile_coding: TileCoding,
        num_actions: Int,
        learning_rate: Float64 = 0.001,
        discount_factor: Float64 = 0.99,
        use_baseline: Bool = True,
        baseline_lr: Float64 = 0.01,
        init_value: Float64 = 0.0,
    ):
        """Initialize REINFORCE agent.

        Args:
            tile_coding: TileCoding instance defining the feature representation
            num_actions: Number of discrete actions
            learning_rate: Policy learning rate α (default 0.001)
            discount_factor: Discount factor γ (default 0.99)
            use_baseline: Whether to use a learned value baseline (default True)
            baseline_lr: Learning rate for baseline (default 0.01)
            init_value: Initial parameter value (default 0.0)
        """
        self.num_actions = num_actions
        self.num_tiles = tile_coding.get_num_tiles()
        self.num_tilings = tile_coding.get_num_tilings()
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.use_baseline = use_baseline
        self.baseline_lr = baseline_lr

        # Initialize policy parameters θ
        self.theta = List[List[Float64]]()
        for a in range(num_actions):
            var action_params = List[Float64]()
            for _ in range(self.num_tiles):
                action_params.append(init_value)
            self.theta.append(action_params^)

        # Initialize baseline (state value function approximation)
        self.baseline_weights = List[Float64]()
        for _ in range(self.num_tiles):
            self.baseline_weights.append(0.0)

        # Initialize episode storage
        self.episode_tiles = List[List[Int]]()
        self.episode_actions = List[Int]()
        self.episode_rewards = List[Float64]()

    fn __copyinit__(out self, existing: Self):
        self.num_actions = existing.num_actions
        self.num_tiles = existing.num_tiles
        self.num_tilings = existing.num_tilings
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.use_baseline = existing.use_baseline
        self.baseline_lr = existing.baseline_lr
        self.theta = List[List[Float64]]()
        for a in range(existing.num_actions):
            var action_params = List[Float64]()
            for t in range(existing.num_tiles):
                action_params.append(existing.theta[a][t])
            self.theta.append(action_params^)
        self.baseline_weights = List[Float64]()
        for t in range(existing.num_tiles):
            self.baseline_weights.append(existing.baseline_weights[t])
        self.episode_tiles = List[List[Int]]()
        self.episode_actions = List[Int]()
        self.episode_rewards = List[Float64]()

    fn __moveinit__(out self, deinit existing: Self):
        self.num_actions = existing.num_actions
        self.num_tiles = existing.num_tiles
        self.num_tilings = existing.num_tilings
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.use_baseline = existing.use_baseline
        self.baseline_lr = existing.baseline_lr
        self.theta = existing.theta^
        self.baseline_weights = existing.baseline_weights^
        self.episode_tiles = existing.episode_tiles^
        self.episode_actions = existing.episode_actions^
        self.episode_rewards = existing.episode_rewards^

    fn _get_action_preferences(self, tiles: List[Int]) -> List[Float64]:
        """Compute action preferences (logits) for given state.

        h(s, a) = θ_a · φ(s) = sum of θ[a][tile] for active tiles

        Args:
            tiles: Active tile indices from tile coding

        Returns:
            List of preferences, one per action
        """
        var preferences = List[Float64]()
        for a in range(self.num_actions):
            var pref: Float64 = 0.0
            for i in range(len(tiles)):
                pref += self.theta[a][tiles[i]]
            preferences.append(pref)
        return preferences^

    fn _softmax(self, preferences: List[Float64]) -> List[Float64]:
        """Compute softmax probabilities from preferences.

        π(a|s) = exp(h(s,a)) / Σ_a' exp(h(s,a'))

        Uses numerically stable softmax (subtract max).

        Args:
            preferences: Action preferences (logits)

        Returns:
            Action probabilities
        """
        # Find max for numerical stability
        var max_pref = preferences[0]
        for i in range(1, len(preferences)):
            if preferences[i] > max_pref:
                max_pref = preferences[i]

        # Compute exp(h - max) and sum
        var exp_prefs = List[Float64]()
        var sum_exp: Float64 = 0.0
        for i in range(len(preferences)):
            var e = exp(preferences[i] - max_pref)
            exp_prefs.append(e)
            sum_exp += e

        # Normalize
        var probs = List[Float64]()
        for i in range(len(exp_prefs)):
            probs.append(exp_prefs[i] / sum_exp)

        return probs^

    fn get_action_probs(self, tiles: List[Int]) -> List[Float64]:
        """Get action probabilities for given state.

        Args:
            tiles: Active tile indices

        Returns:
            Probability distribution over actions
        """
        var prefs = self._get_action_preferences(tiles)
        return self._softmax(prefs^)

    fn select_action(self, tiles: List[Int]) -> Int:
        """Sample action from policy π(a|s).

        Args:
            tiles: Active tile indices from TileCoding.get_tiles()

        Returns:
            Sampled action index
        """
        var probs = self.get_action_probs(tiles)

        # Sample from categorical distribution
        var rand = random_float64()
        var cumsum: Float64 = 0.0
        for a in range(self.num_actions):
            cumsum += probs[a]
            if rand < cumsum:
                return a

        # Fallback (shouldn't happen with proper probabilities)
        return self.num_actions - 1

    fn get_best_action(self, tiles: List[Int]) -> Int:
        """Get greedy action (highest probability).

        Args:
            tiles: Active tile indices

        Returns:
            Action with highest probability
        """
        var probs = self.get_action_probs(tiles)
        var best_action = 0
        var best_prob = probs[0]
        for a in range(1, self.num_actions):
            if probs[a] > best_prob:
                best_prob = probs[a]
                best_action = a
        return best_action

    fn _get_baseline_value(self, tiles: List[Int]) -> Float64:
        """Get baseline value estimate V(s).

        Args:
            tiles: Active tile indices

        Returns:
            Estimated state value
        """
        var value: Float64 = 0.0
        for i in range(len(tiles)):
            value += self.baseline_weights[tiles[i]]
        return value

    fn _get_baseline_value_idx(self, episode_idx: Int) -> Float64:
        """Get baseline value for episode step by index."""
        var value: Float64 = 0.0
        var num_tiles = len(self.episode_tiles[episode_idx])
        for i in range(num_tiles):
            var tile_idx = self.episode_tiles[episode_idx][i]
            value += self.baseline_weights[tile_idx]
        return value

    fn _get_action_probs_idx(self, episode_idx: Int) -> List[Float64]:
        """Get action probabilities for episode step by index."""
        var preferences = List[Float64]()
        var num_tiles = len(self.episode_tiles[episode_idx])
        for a in range(self.num_actions):
            var pref: Float64 = 0.0
            for i in range(num_tiles):
                var tile_idx = self.episode_tiles[episode_idx][i]
                pref += self.theta[a][tile_idx]
            preferences.append(pref)
        return self._softmax(preferences^)

    fn store_transition(
        mut self,
        tiles: List[Int],
        action: Int,
        reward: Float64,
    ):
        """Store transition for end-of-episode update.

        Args:
            tiles: Active tiles for current state
            action: Action taken
            reward: Reward received
        """
        # Copy tiles since we need to store them
        var tiles_copy = List[Int]()
        for i in range(len(tiles)):
            tiles_copy.append(tiles[i])

        self.episode_tiles.append(tiles_copy^)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

    fn update_from_episode(mut self):
        """Update policy parameters using REINFORCE.

        Called at end of episode. Computes returns and updates
        θ += α * G_t * ∇log π(a_t|s_t)
        """
        var num_steps = len(self.episode_tiles)
        if num_steps == 0:
            return

        # Compute returns (discounted sum of future rewards)
        var returns = List[Float64]()
        for _ in range(num_steps):
            returns.append(0.0)

        var g: Float64 = 0.0
        for t in range(num_steps - 1, -1, -1):
            g = self.episode_rewards[t] + self.discount_factor * g
            returns[t] = g

        # Update policy for each timestep
        var step_size = self.learning_rate / Float64(self.num_tilings)

        for t in range(num_steps):
            var action = self.episode_actions[t]
            var g_t = returns[t]
            var num_tiles_t = len(self.episode_tiles[t])

            # Optionally subtract baseline for variance reduction
            var advantage = g_t
            if self.use_baseline:
                var baseline = self._get_baseline_value_idx(t)
                advantage = g_t - baseline

                # Update baseline towards return
                var baseline_step = self.baseline_lr / Float64(self.num_tilings)
                var baseline_error = g_t - baseline
                for i in range(num_tiles_t):
                    var tile_idx = self.episode_tiles[t][i]
                    self.baseline_weights[tile_idx] += baseline_step * baseline_error

            # Compute action probabilities
            var probs = self._get_action_probs_idx(t)

            # Update θ using policy gradient
            # ∇log π(a|s) for softmax:
            #   For chosen action a: 1 - π(a|s)
            #   For other actions a': -π(a'|s)
            for a in range(self.num_actions):
                var grad: Float64
                if a == action:
                    grad = 1.0 - probs[a]
                else:
                    grad = -probs[a]

                # θ_a += α * G_t * ∇log π(a|s)
                for i in range(num_tiles_t):
                    var tile_idx = self.episode_tiles[t][i]
                    self.theta[a][tile_idx] += step_size * advantage * grad

        # Clear episode storage
        self.episode_tiles.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()

    fn reset(mut self):
        """Reset episode storage for new episode."""
        self.episode_tiles.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()

    fn get_policy_entropy(self, tiles: List[Int]) -> Float64:
        """Compute entropy of policy at given state.

        H(π) = -Σ_a π(a|s) log π(a|s)

        Higher entropy = more exploration.

        Args:
            tiles: Active tile indices

        Returns:
            Policy entropy (in nats)
        """
        var probs = self.get_action_probs(tiles)
        var entropy: Float64 = 0.0
        for a in range(self.num_actions):
            if probs[a] > 1e-10:  # Avoid log(0)
                entropy -= probs[a] * log(probs[a])
        return entropy

    fn train[E: BoxDiscreteActionEnv](
        mut self,
        mut env: E,
        tile_coding: TileCoding,
        num_episodes: Int,
        max_steps_per_episode: Int = 500,
        verbose: Bool = False,
        print_every: Int = 100,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train the agent on a continuous-state environment using REINFORCE.

        Args:
            env: The classic control environment to train on.
            tile_coding: TileCoding instance for feature extraction.
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
            verbose: Whether to print progress.
            print_every: Print progress every N episodes (if verbose).
            environment_name: Name of environment for metrics labeling.

        Returns:
            TrainingMetrics object with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="REINFORCE",
            environment_name=environment_name,
        )

        for episode in range(num_episodes):
            self.reset()  # Clear episode storage
            var obs = env.reset_obs()
            var total_reward: Float64 = 0.0
            var steps = 0

            for _ in range(max_steps_per_episode):
                var tiles = tile_coding.get_tiles_simd4(obs)
                var action = self.select_action(tiles)

                var result = env.step_raw(action)
                var next_obs = result[0]
                var reward = result[1]
                var done = result[2]

                self.store_transition(tiles, action, reward)

                total_reward += reward
                steps += 1
                obs = next_obs

                if done:
                    break

            # REINFORCE updates at end of episode
            self.update_from_episode()
            metrics.log_episode(episode, total_reward, steps, 0.0)

            if verbose and (episode + 1) % print_every == 0:
                metrics.print_progress(episode, window=100)

        return metrics^

    fn evaluate[E: BoxDiscreteActionEnv](
        self,
        mut env: E,
        tile_coding: TileCoding,
        num_episodes: Int = 10,
        max_steps: Int = 500,
        render: Bool = False,
    ) -> Float64:
        """Evaluate the agent on the environment.

        Args:
            env: The classic control environment to evaluate on.
            tile_coding: TileCoding instance for feature extraction.
            num_episodes: Number of evaluation episodes.
            max_steps: Maximum steps per episode.
            render: Whether to render the environment.

        Returns:
            Average reward across episodes.
        """
        var total_reward: Float64 = 0.0

        for _ in range(num_episodes):
            var obs = env.reset_obs()
            var episode_reward: Float64 = 0.0

            for _ in range(max_steps):
                if render:
                    env.render()

                var tiles = tile_coding.get_tiles_simd4(obs)
                var action = self.get_best_action(tiles)

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


struct REINFORCEWithEntropyAgent(Copyable, Movable, ImplicitlyCopyable):
    """REINFORCE with entropy regularization for better exploration.

    Adds entropy bonus to encourage exploration:
    J(θ) = E[Σ_t (G_t * log π(a_t|s_t) + β * H(π(·|s_t)))]

    where β is the entropy coefficient.
    """

    var theta: List[List[Float64]]
    var num_actions: Int
    var num_tiles: Int
    var num_tilings: Int
    var learning_rate: Float64
    var discount_factor: Float64
    var entropy_coef: Float64

    var episode_tiles: List[List[Int]]
    var episode_actions: List[Int]
    var episode_rewards: List[Float64]

    var use_baseline: Bool
    var baseline_weights: List[Float64]
    var baseline_lr: Float64

    fn __init__(
        out self,
        tile_coding: TileCoding,
        num_actions: Int,
        learning_rate: Float64 = 0.001,
        discount_factor: Float64 = 0.99,
        entropy_coef: Float64 = 0.01,
        use_baseline: Bool = True,
        baseline_lr: Float64 = 0.01,
        init_value: Float64 = 0.0,
    ):
        """Initialize REINFORCE with entropy regularization.

        Args:
            tile_coding: TileCoding instance
            num_actions: Number of discrete actions
            learning_rate: Policy learning rate
            discount_factor: Discount factor γ
            entropy_coef: Entropy bonus coefficient β
            use_baseline: Whether to use learned baseline
            baseline_lr: Baseline learning rate
            init_value: Initial parameter value
        """
        self.num_actions = num_actions
        self.num_tiles = tile_coding.get_num_tiles()
        self.num_tilings = tile_coding.get_num_tilings()
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.entropy_coef = entropy_coef
        self.use_baseline = use_baseline
        self.baseline_lr = baseline_lr

        self.theta = List[List[Float64]]()
        for a in range(num_actions):
            var action_params = List[Float64]()
            for _ in range(self.num_tiles):
                action_params.append(init_value)
            self.theta.append(action_params^)

        self.baseline_weights = List[Float64]()
        for _ in range(self.num_tiles):
            self.baseline_weights.append(0.0)

        self.episode_tiles = List[List[Int]]()
        self.episode_actions = List[Int]()
        self.episode_rewards = List[Float64]()

    fn __copyinit__(out self, existing: Self):
        self.num_actions = existing.num_actions
        self.num_tiles = existing.num_tiles
        self.num_tilings = existing.num_tilings
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.entropy_coef = existing.entropy_coef
        self.use_baseline = existing.use_baseline
        self.baseline_lr = existing.baseline_lr
        self.theta = List[List[Float64]]()
        for a in range(existing.num_actions):
            var action_params = List[Float64]()
            for t in range(existing.num_tiles):
                action_params.append(existing.theta[a][t])
            self.theta.append(action_params^)
        self.baseline_weights = List[Float64]()
        for t in range(existing.num_tiles):
            self.baseline_weights.append(existing.baseline_weights[t])
        self.episode_tiles = List[List[Int]]()
        self.episode_actions = List[Int]()
        self.episode_rewards = List[Float64]()

    fn __moveinit__(out self, deinit existing: Self):
        self.num_actions = existing.num_actions
        self.num_tiles = existing.num_tiles
        self.num_tilings = existing.num_tilings
        self.learning_rate = existing.learning_rate
        self.discount_factor = existing.discount_factor
        self.entropy_coef = existing.entropy_coef
        self.use_baseline = existing.use_baseline
        self.baseline_lr = existing.baseline_lr
        self.theta = existing.theta^
        self.baseline_weights = existing.baseline_weights^
        self.episode_tiles = existing.episode_tiles^
        self.episode_actions = existing.episode_actions^
        self.episode_rewards = existing.episode_rewards^

    fn _get_action_preferences(self, tiles: List[Int]) -> List[Float64]:
        """Compute action preferences."""
        var preferences = List[Float64]()
        for a in range(self.num_actions):
            var pref: Float64 = 0.0
            for i in range(len(tiles)):
                pref += self.theta[a][tiles[i]]
            preferences.append(pref)
        return preferences^

    fn _softmax(self, preferences: List[Float64]) -> List[Float64]:
        """Compute softmax probabilities."""
        var max_pref = preferences[0]
        for i in range(1, len(preferences)):
            if preferences[i] > max_pref:
                max_pref = preferences[i]

        var exp_prefs = List[Float64]()
        var sum_exp: Float64 = 0.0
        for i in range(len(preferences)):
            var e = exp(preferences[i] - max_pref)
            exp_prefs.append(e)
            sum_exp += e

        var probs = List[Float64]()
        for i in range(len(exp_prefs)):
            probs.append(exp_prefs[i] / sum_exp)
        return probs^

    fn get_action_probs(self, tiles: List[Int]) -> List[Float64]:
        """Get action probabilities."""
        var prefs = self._get_action_preferences(tiles)
        return self._softmax(prefs^)

    fn select_action(self, tiles: List[Int]) -> Int:
        """Sample action from policy."""
        var probs = self.get_action_probs(tiles)
        var rand = random_float64()
        var cumsum: Float64 = 0.0
        for a in range(self.num_actions):
            cumsum += probs[a]
            if rand < cumsum:
                return a
        return self.num_actions - 1

    fn get_best_action(self, tiles: List[Int]) -> Int:
        """Get greedy action."""
        var probs = self.get_action_probs(tiles)
        var best_action = 0
        var best_prob = probs[0]
        for a in range(1, self.num_actions):
            if probs[a] > best_prob:
                best_prob = probs[a]
                best_action = a
        return best_action

    fn _get_baseline_value(self, tiles: List[Int]) -> Float64:
        """Get baseline value estimate."""
        var value: Float64 = 0.0
        for i in range(len(tiles)):
            value += self.baseline_weights[tiles[i]]
        return value

    fn _get_baseline_value_idx(self, episode_idx: Int) -> Float64:
        """Get baseline value for episode step by index."""
        var value: Float64 = 0.0
        var num_tiles = len(self.episode_tiles[episode_idx])
        for i in range(num_tiles):
            var tile_idx = self.episode_tiles[episode_idx][i]
            value += self.baseline_weights[tile_idx]
        return value

    fn _get_action_probs_idx(self, episode_idx: Int) -> List[Float64]:
        """Get action probabilities for episode step by index."""
        var preferences = List[Float64]()
        var num_tiles = len(self.episode_tiles[episode_idx])
        for a in range(self.num_actions):
            var pref: Float64 = 0.0
            for i in range(num_tiles):
                var tile_idx = self.episode_tiles[episode_idx][i]
                pref += self.theta[a][tile_idx]
            preferences.append(pref)
        return self._softmax(preferences^)

    fn store_transition(
        mut self,
        tiles: List[Int],
        action: Int,
        reward: Float64,
    ):
        """Store transition."""
        var tiles_copy = List[Int]()
        for i in range(len(tiles)):
            tiles_copy.append(tiles[i])
        self.episode_tiles.append(tiles_copy^)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

    fn update_from_episode(mut self):
        """Update with entropy regularization."""
        var num_steps = len(self.episode_tiles)
        if num_steps == 0:
            return

        var returns = List[Float64]()
        for _ in range(num_steps):
            returns.append(0.0)

        var g: Float64 = 0.0
        for t in range(num_steps - 1, -1, -1):
            g = self.episode_rewards[t] + self.discount_factor * g
            returns[t] = g

        var step_size = self.learning_rate / Float64(self.num_tilings)

        for t in range(num_steps):
            var action = self.episode_actions[t]
            var g_t = returns[t]
            var num_tiles_t = len(self.episode_tiles[t])

            var advantage = g_t
            if self.use_baseline:
                var baseline = self._get_baseline_value_idx(t)
                advantage = g_t - baseline
                var baseline_step = self.baseline_lr / Float64(self.num_tilings)
                var baseline_error = g_t - baseline
                for i in range(num_tiles_t):
                    var tile_idx = self.episode_tiles[t][i]
                    self.baseline_weights[tile_idx] += baseline_step * baseline_error

            var probs = self._get_action_probs_idx(t)

            # Policy gradient with entropy bonus
            # ∇J = ∇log π(a|s) * A + β * ∇H(π)
            # ∇H(π) = -∇(Σ_a π log π) = -(1 + log π) for each action
            for a in range(self.num_actions):
                var policy_grad: Float64
                if a == action:
                    policy_grad = 1.0 - probs[a]
                else:
                    policy_grad = -probs[a]

                # Entropy gradient: ∂H/∂θ_a = -π(a)(1 + log π(a)) * ∂π/∂θ_a
                # For softmax: ∂π_a/∂θ_a = π_a(1 - π_a)
                # Simplified: encourages uniform distribution
                var entropy_grad: Float64 = 0.0
                if probs[a] > 1e-10:
                    entropy_grad = -probs[a] * (1.0 + log(probs[a])) * probs[a] * (1.0 - probs[a])

                var total_grad = advantage * policy_grad + self.entropy_coef * entropy_grad

                for i in range(num_tiles_t):
                    var tile_idx = self.episode_tiles[t][i]
                    self.theta[a][tile_idx] += step_size * total_grad

        self.episode_tiles.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()

    fn reset(mut self):
        """Reset episode storage."""
        self.episode_tiles.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()

    fn get_policy_entropy(self, tiles: List[Int]) -> Float64:
        """Compute policy entropy."""
        var probs = self.get_action_probs(tiles)
        var entropy: Float64 = 0.0
        for a in range(self.num_actions):
            if probs[a] > 1e-10:
                entropy -= probs[a] * log(probs[a])
        return entropy

    fn train[E: BoxDiscreteActionEnv](
        mut self,
        mut env: E,
        tile_coding: TileCoding,
        num_episodes: Int,
        max_steps_per_episode: Int = 500,
        verbose: Bool = False,
        print_every: Int = 100,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train the agent on a continuous-state environment.

        Args:
            env: The classic control environment to train on.
            tile_coding: TileCoding instance for feature extraction.
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
            verbose: Whether to print progress.
            print_every: Print progress every N episodes (if verbose).
            environment_name: Name of environment for metrics labeling.

        Returns:
            TrainingMetrics object with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="REINFORCE + Entropy",
            environment_name=environment_name,
        )

        for episode in range(num_episodes):
            self.reset()  # Clear episode storage
            var obs = env.reset_obs()
            var total_reward: Float64 = 0.0
            var steps = 0

            for _ in range(max_steps_per_episode):
                var tiles = tile_coding.get_tiles_simd4(obs)
                var action = self.select_action(tiles)

                var result = env.step_raw(action)
                var next_obs = result[0]
                var reward = result[1]
                var done = result[2]

                self.store_transition(tiles, action, reward)

                total_reward += reward
                steps += 1
                obs = next_obs

                if done:
                    break

            # REINFORCE updates at end of episode
            self.update_from_episode()
            metrics.log_episode(episode, total_reward, steps, 0.0)

            if verbose and (episode + 1) % print_every == 0:
                metrics.print_progress(episode, window=100)

        return metrics^

    fn evaluate[E: BoxDiscreteActionEnv](
        self,
        mut env: E,
        tile_coding: TileCoding,
        num_episodes: Int = 10,
        max_steps: Int = 500,
        render: Bool = False,
    ) -> Float64:
        """Evaluate the agent on the environment.

        Args:
            env: The classic control environment to evaluate on.
            tile_coding: TileCoding instance for feature extraction.
            num_episodes: Number of evaluation episodes.
            max_steps: Maximum steps per episode.
            render: Whether to render the environment.

        Returns:
            Average reward across episodes.
        """
        var total_reward: Float64 = 0.0

        for _ in range(num_episodes):
            var obs = env.reset_obs()
            var episode_reward: Float64 = 0.0

            for _ in range(max_steps):
                if render:
                    env.render()

                var tiles = tile_coding.get_tiles_simd4(obs)
                var action = self.get_best_action(tiles)

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

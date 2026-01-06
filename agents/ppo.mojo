"""Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE).

PPO is a state-of-the-art policy gradient algorithm that improves training
stability through clipped surrogate objectives. Combined with GAE for
advantage estimation, it provides a good balance of sample efficiency and
stability.

Key components:
1. **GAE (Generalized Advantage Estimation)**: Computes advantages using
   exponentially-weighted average of TD residuals:
   A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
   where δ_t = r_t + γV(s_{t+1}) - V(s_t)

2. **PPO Clipping**: Constrains policy updates to prevent large changes:
   L^CLIP = min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)
   where r(θ) = π_θ(a|s) / π_θ_old(a|s)

References:
- Schulman et al. (2017): "Proximal Policy Optimization Algorithms"
- Schulman et al. (2016): "High-Dimensional Continuous Control Using GAE"

Example usage:
    from core.tile_coding import make_cartpole_tile_coding
    from agents.ppo import PPOAgent

    var tc = make_cartpole_tile_coding(num_tilings=8, tiles_per_dim=8)
    var agent = PPOAgent(
        tile_coding=tc,
        num_actions=2,
        actor_lr=0.0003,
        critic_lr=0.001,
        clip_epsilon=0.2,
        gae_lambda=0.95,
    )

    # Collect rollout
    for step in range(rollout_length):
        var tiles = tc.get_tiles_simd4(obs)
        var action = agent.select_action(tiles)
        var log_prob = agent.get_log_prob(tiles, action)
        # ... environment step ...
        agent.store_transition(tiles, action, reward, log_prob)

    # Update at end of rollout
    agent.update(next_tiles, done)
"""

from math import exp, log
from random import random_float64
from core.tile_coding import TileCoding


fn compute_gae(
    rewards: List[Float64],
    values: List[Float64],
    next_value: Float64,
    done: Bool,
    discount_factor: Float64,
    gae_lambda: Float64,
) -> List[Float64]:
    """Compute Generalized Advantage Estimation.

    GAE provides a family of advantage estimators parameterized by λ:
    - λ=0: One-step TD advantage (low variance, high bias)
    - λ=1: Monte Carlo advantage (high variance, low bias)
    - 0<λ<1: Exponentially-weighted average of n-step advantages

    A^GAE_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}

    where δ_t = r_t + γV(s_{t+1}) - V(s_t)

    Args:
        rewards: List of rewards [r_0, r_1, ..., r_{T-1}]
        values: List of value estimates [V(s_0), V(s_1), ..., V(s_{T-1})]
        next_value: Bootstrap value V(s_T) (0 if terminal)
        done: Whether episode terminated
        discount_factor: Discount factor γ
        gae_lambda: GAE parameter λ

    Returns:
        List of advantages [A_0, A_1, ..., A_{T-1}]
    """
    var num_steps = len(rewards)
    var advantages = List[Float64]()

    # Initialize advantages list
    for _ in range(num_steps):
        advantages.append(0.0)

    # Bootstrap value for last step
    var last_value = next_value
    if done:
        last_value = 0.0

    # Compute GAE backwards
    var gae: Float64 = 0.0
    var gae_decay = discount_factor * gae_lambda

    for t in range(num_steps - 1, -1, -1):
        # Get next value
        var next_val: Float64
        if t == num_steps - 1:
            next_val = last_value
        else:
            next_val = values[t + 1]

        # TD residual: δ_t = r_t + γV(s_{t+1}) - V(s_t)
        var delta = rewards[t] + discount_factor * next_val - values[t]

        # GAE: A_t = δ_t + γλA_{t+1}
        gae = delta + gae_decay * gae
        advantages[t] = gae

    return advantages^


fn compute_returns_from_advantages(
    advantages: List[Float64],
    values: List[Float64],
) -> List[Float64]:
    """Compute returns from advantages and values.

    Returns = Advantages + Values

    Args:
        advantages: GAE advantages
        values: Value estimates

    Returns:
        Returns for value function training
    """
    var returns = List[Float64]()
    for t in range(len(advantages)):
        returns.append(advantages[t] + values[t])
    return returns^


struct PPOAgent(Copyable, Movable, ImplicitlyCopyable):
    """Proximal Policy Optimization agent with tile coding.

    Uses clipped surrogate objective for stable policy updates and
    GAE for advantage estimation.
    """

    # Actor (policy) parameters: θ[action][tile]
    var theta: List[List[Float64]]

    # Critic (value) parameters: w[tile]
    var critic_weights: List[Float64]

    var num_actions: Int
    var num_tiles: Int
    var num_tilings: Int
    var actor_lr: Float64
    var critic_lr: Float64
    var discount_factor: Float64
    var gae_lambda: Float64
    var clip_epsilon: Float64
    var entropy_coef: Float64
    var value_loss_coef: Float64
    var num_epochs: Int
    var normalize_advantages: Bool

    # Rollout buffer
    var buffer_tiles: List[List[Int]]
    var buffer_actions: List[Int]
    var buffer_rewards: List[Float64]
    var buffer_log_probs: List[Float64]  # Old log probabilities
    var buffer_values: List[Float64]      # Old value estimates

    fn __init__(
        out self,
        tile_coding: TileCoding,
        num_actions: Int,
        actor_lr: Float64 = 0.0003,
        critic_lr: Float64 = 0.001,
        discount_factor: Float64 = 0.99,
        gae_lambda: Float64 = 0.95,
        clip_epsilon: Float64 = 0.2,
        entropy_coef: Float64 = 0.01,
        value_loss_coef: Float64 = 0.5,
        num_epochs: Int = 4,
        normalize_advantages: Bool = True,
        init_value: Float64 = 0.0,
    ):
        """Initialize PPO agent.

        Args:
            tile_coding: TileCoding instance defining feature representation
            num_actions: Number of discrete actions
            actor_lr: Actor (policy) learning rate (default 0.0003)
            critic_lr: Critic (value) learning rate (default 0.001)
            discount_factor: Discount factor γ (default 0.99)
            gae_lambda: GAE parameter λ (default 0.95)
            clip_epsilon: PPO clipping parameter ε (default 0.2)
            entropy_coef: Entropy bonus coefficient (default 0.01)
            value_loss_coef: Value loss coefficient (default 0.5)
            num_epochs: Number of optimization epochs per update (default 4)
            normalize_advantages: Whether to normalize advantages (default True)
            init_value: Initial parameter value (default 0.0)
        """
        self.num_actions = num_actions
        self.num_tiles = tile_coding.get_num_tiles()
        self.num_tilings = tile_coding.get_num_tilings()
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.num_epochs = num_epochs
        self.normalize_advantages = normalize_advantages

        # Initialize actor parameters
        self.theta = List[List[Float64]]()
        for _ in range(num_actions):
            var action_params = List[Float64]()
            for _ in range(self.num_tiles):
                action_params.append(init_value)
            self.theta.append(action_params^)

        # Initialize critic parameters
        self.critic_weights = List[Float64]()
        for _ in range(self.num_tiles):
            self.critic_weights.append(init_value)

        # Initialize rollout buffer
        self.buffer_tiles = List[List[Int]]()
        self.buffer_actions = List[Int]()
        self.buffer_rewards = List[Float64]()
        self.buffer_log_probs = List[Float64]()
        self.buffer_values = List[Float64]()

    fn __copyinit__(out self, existing: Self):
        self.num_actions = existing.num_actions
        self.num_tiles = existing.num_tiles
        self.num_tilings = existing.num_tilings
        self.actor_lr = existing.actor_lr
        self.critic_lr = existing.critic_lr
        self.discount_factor = existing.discount_factor
        self.gae_lambda = existing.gae_lambda
        self.clip_epsilon = existing.clip_epsilon
        self.entropy_coef = existing.entropy_coef
        self.value_loss_coef = existing.value_loss_coef
        self.num_epochs = existing.num_epochs
        self.normalize_advantages = existing.normalize_advantages
        self.theta = List[List[Float64]]()
        for a in range(existing.num_actions):
            var action_params = List[Float64]()
            for t in range(existing.num_tiles):
                action_params.append(existing.theta[a][t])
            self.theta.append(action_params^)
        self.critic_weights = List[Float64]()
        for t in range(existing.num_tiles):
            self.critic_weights.append(existing.critic_weights[t])
        self.buffer_tiles = List[List[Int]]()
        self.buffer_actions = List[Int]()
        self.buffer_rewards = List[Float64]()
        self.buffer_log_probs = List[Float64]()
        self.buffer_values = List[Float64]()

    fn __moveinit__(out self, deinit existing: Self):
        self.num_actions = existing.num_actions
        self.num_tiles = existing.num_tiles
        self.num_tilings = existing.num_tilings
        self.actor_lr = existing.actor_lr
        self.critic_lr = existing.critic_lr
        self.discount_factor = existing.discount_factor
        self.gae_lambda = existing.gae_lambda
        self.clip_epsilon = existing.clip_epsilon
        self.entropy_coef = existing.entropy_coef
        self.value_loss_coef = existing.value_loss_coef
        self.num_epochs = existing.num_epochs
        self.normalize_advantages = existing.normalize_advantages
        self.theta = existing.theta^
        self.critic_weights = existing.critic_weights^
        self.buffer_tiles = existing.buffer_tiles^
        self.buffer_actions = existing.buffer_actions^
        self.buffer_rewards = existing.buffer_rewards^
        self.buffer_log_probs = existing.buffer_log_probs^
        self.buffer_values = existing.buffer_values^

    fn _get_action_preferences(self, tiles: List[Int]) -> List[Float64]:
        """Compute action preferences (logits).

        h(s, a) = θ_a · φ(s)
        """
        var preferences = List[Float64]()
        for a in range(self.num_actions):
            var pref: Float64 = 0.0
            for i in range(len(tiles)):
                pref += self.theta[a][tiles[i]]
            preferences.append(pref)
        return preferences^

    fn _softmax(self, preferences: List[Float64]) -> List[Float64]:
        """Compute softmax probabilities (numerically stable)."""
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
        """Get action probabilities π(·|s).

        Args:
            tiles: Active tile indices

        Returns:
            Probability distribution over actions
        """
        var prefs = self._get_action_preferences(tiles)
        return self._softmax(prefs^)

    fn get_value(self, tiles: List[Int]) -> Float64:
        """Get state value estimate V(s).

        Args:
            tiles: Active tile indices

        Returns:
            Estimated state value
        """
        var value: Float64 = 0.0
        for i in range(len(tiles)):
            value += self.critic_weights[tiles[i]]
        return value

    fn get_log_prob(self, tiles: List[Int], action: Int) -> Float64:
        """Get log probability of action under current policy.

        Args:
            tiles: Active tile indices
            action: Action index

        Returns:
            log π(a|s)
        """
        var probs = self.get_action_probs(tiles)
        if probs[action] > 1e-10:
            return log(probs[action])
        return -23.0  # log(1e-10) ≈ -23

    fn select_action(self, tiles: List[Int]) -> Int:
        """Sample action from policy π(a|s).

        Args:
            tiles: Active tile indices

        Returns:
            Sampled action index
        """
        var probs = self.get_action_probs(tiles)

        var rand = random_float64()
        var cumsum: Float64 = 0.0
        for a in range(self.num_actions):
            cumsum += probs[a]
            if rand < cumsum:
                return a
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

    fn store_transition(
        mut self,
        tiles: List[Int],
        action: Int,
        reward: Float64,
        log_prob: Float64,
        value: Float64,
    ):
        """Store transition in rollout buffer.

        Args:
            tiles: Active tiles for current state
            action: Action taken
            reward: Reward received
            log_prob: Log probability of action (from old policy)
            value: Value estimate (from old critic)
        """
        var tiles_copy = List[Int]()
        for i in range(len(tiles)):
            tiles_copy.append(tiles[i])

        self.buffer_tiles.append(tiles_copy^)
        self.buffer_actions.append(action)
        self.buffer_rewards.append(reward)
        self.buffer_log_probs.append(log_prob)
        self.buffer_values.append(value)

    fn _get_value_idx(self, buffer_idx: Int) -> Float64:
        """Get current value estimate for buffer step."""
        var value: Float64 = 0.0
        var num_tiles = len(self.buffer_tiles[buffer_idx])
        for i in range(num_tiles):
            var tile_idx = self.buffer_tiles[buffer_idx][i]
            value += self.critic_weights[tile_idx]
        return value

    fn _get_action_probs_idx(self, buffer_idx: Int) -> List[Float64]:
        """Get current action probabilities for buffer step."""
        var preferences = List[Float64]()
        var num_tiles = len(self.buffer_tiles[buffer_idx])
        for a in range(self.num_actions):
            var pref: Float64 = 0.0
            for i in range(num_tiles):
                var tile_idx = self.buffer_tiles[buffer_idx][i]
                pref += self.theta[a][tile_idx]
            preferences.append(pref)
        return self._softmax(preferences^)

    fn _get_log_prob_idx(self, buffer_idx: Int, action: Int) -> Float64:
        """Get current log probability for buffer step."""
        var probs = self._get_action_probs_idx(buffer_idx)
        if probs[action] > 1e-10:
            return log(probs[action])
        return -23.0

    fn update(
        mut self,
        next_tiles: List[Int],
        done: Bool,
    ):
        """Update policy and value function using PPO.

        Performs multiple epochs of optimization on collected rollout.

        Args:
            next_tiles: Tiles for next state (for bootstrap)
            done: Whether episode terminated
        """
        var buffer_len = len(self.buffer_tiles)
        if buffer_len == 0:
            return

        # Get bootstrap value
        var next_value: Float64 = 0.0
        if not done:
            next_value = self.get_value(next_tiles)

        # Compute GAE advantages
        var advantages = compute_gae(
            self.buffer_rewards,
            self.buffer_values,
            next_value,
            done,
            self.discount_factor,
            self.gae_lambda,
        )

        # Compute returns for value function
        var returns = compute_returns_from_advantages(advantages, self.buffer_values)

        # Normalize advantages (optional but recommended)
        if self.normalize_advantages and buffer_len > 1:
            var mean: Float64 = 0.0
            for t in range(buffer_len):
                mean += advantages[t]
            mean /= Float64(buffer_len)

            var variance: Float64 = 0.0
            for t in range(buffer_len):
                var diff = advantages[t] - mean
                variance += diff * diff
            variance /= Float64(buffer_len)
            var std = (variance + 1e-8) ** 0.5

            for t in range(buffer_len):
                advantages[t] = (advantages[t] - mean) / std

        # Multiple epochs of optimization
        var actor_step = self.actor_lr / Float64(self.num_tilings)
        var critic_step = self.critic_lr / Float64(self.num_tilings)

        for _ in range(self.num_epochs):
            for t in range(buffer_len):
                var action = self.buffer_actions[t]
                var old_log_prob = self.buffer_log_probs[t]
                var advantage = advantages[t]
                var return_t = returns[t]
                var num_tiles_t = len(self.buffer_tiles[t])

                # Current policy probability
                var new_log_prob = self._get_log_prob_idx(t, action)

                # Probability ratio r(θ) = π_θ(a|s) / π_θ_old(a|s)
                var ratio = exp(new_log_prob - old_log_prob)

                # Clipped surrogate objective
                var surr1 = ratio * advantage
                var clipped_ratio: Float64
                if advantage >= 0:
                    clipped_ratio = min(ratio, 1.0 + self.clip_epsilon)
                else:
                    clipped_ratio = max(ratio, 1.0 - self.clip_epsilon)
                var surr2 = clipped_ratio * advantage

                # Note: We don't use policy_loss directly since we compute gradients
                # manually below. The clipping check uses surr1 vs surr2.

                # Value function update (gradient of squared error)
                var current_value = self._get_value_idx(t)
                # Note: value_loss = (return_t - current_value)^2, gradient is 2*(current - return)

                # Get current probabilities for entropy and gradient
                var probs = self._get_action_probs_idx(t)

                # Entropy bonus (negative entropy because we minimize loss)
                var entropy: Float64 = 0.0
                for a in range(self.num_actions):
                    if probs[a] > 1e-10:
                        entropy -= probs[a] * log(probs[a])

                # Update critic (minimize value loss)
                var value_grad = 2.0 * (current_value - return_t)
                for i in range(num_tiles_t):
                    var tile_idx = self.buffer_tiles[t][i]
                    self.critic_weights[tile_idx] -= critic_step * self.value_loss_coef * value_grad

                # Update actor (minimize policy loss - entropy bonus)
                # Gradient of clipped objective is:
                # - If not clipped: advantage * ∇log π(a|s)
                # - If clipped: 0 (no gradient)
                var clipped = (ratio < 1.0 - self.clip_epsilon) or (ratio > 1.0 + self.clip_epsilon)

                if not clipped:
                    for a in range(self.num_actions):
                        # Policy gradient for softmax
                        var policy_grad: Float64
                        if a == action:
                            policy_grad = 1.0 - probs[a]
                        else:
                            policy_grad = -probs[a]

                        # Entropy gradient: encourages uniform distribution
                        var entropy_grad: Float64 = 0.0
                        if probs[a] > 1e-10:
                            # ∂H/∂θ_a for softmax
                            entropy_grad = -probs[a] * (1.0 + log(probs[a])) * probs[a] * (1.0 - probs[a])

                        # Combined gradient (advantage * policy grad - entropy_coef * entropy grad)
                        # We maximize: advantage * log_prob + entropy_coef * entropy
                        # So gradient is: advantage * policy_grad + entropy_coef * entropy_grad
                        var total_grad = advantage * policy_grad + self.entropy_coef * entropy_grad

                        for i in range(num_tiles_t):
                            var tile_idx = self.buffer_tiles[t][i]
                            self.theta[a][tile_idx] += actor_step * total_grad

        # Clear buffer
        self.buffer_tiles.clear()
        self.buffer_actions.clear()
        self.buffer_rewards.clear()
        self.buffer_log_probs.clear()
        self.buffer_values.clear()

    fn reset(mut self):
        """Reset buffer for new episode."""
        self.buffer_tiles.clear()
        self.buffer_actions.clear()
        self.buffer_rewards.clear()
        self.buffer_log_probs.clear()
        self.buffer_values.clear()

    fn get_policy_entropy(self, tiles: List[Int]) -> Float64:
        """Compute policy entropy at given state.

        Args:
            tiles: Active tile indices

        Returns:
            Policy entropy (in nats)
        """
        var probs = self.get_action_probs(tiles)
        var entropy: Float64 = 0.0
        for a in range(self.num_actions):
            if probs[a] > 1e-10:
                entropy -= probs[a] * log(probs[a])
        return entropy


struct PPOAgentWithMinibatch(Copyable, Movable, ImplicitlyCopyable):
    """PPO Agent with minibatch updates for larger rollouts.

    Extends PPOAgent with minibatch sampling during updates,
    which is more efficient for longer rollouts.
    """

    # Actor (policy) parameters
    var theta: List[List[Float64]]

    # Critic (value) parameters
    var critic_weights: List[Float64]

    var num_actions: Int
    var num_tiles: Int
    var num_tilings: Int
    var actor_lr: Float64
    var critic_lr: Float64
    var discount_factor: Float64
    var gae_lambda: Float64
    var clip_epsilon: Float64
    var entropy_coef: Float64
    var value_loss_coef: Float64
    var num_epochs: Int
    var minibatch_size: Int
    var normalize_advantages: Bool

    # Rollout buffer
    var buffer_tiles: List[List[Int]]
    var buffer_actions: List[Int]
    var buffer_rewards: List[Float64]
    var buffer_log_probs: List[Float64]
    var buffer_values: List[Float64]

    fn __init__(
        out self,
        tile_coding: TileCoding,
        num_actions: Int,
        actor_lr: Float64 = 0.0003,
        critic_lr: Float64 = 0.001,
        discount_factor: Float64 = 0.99,
        gae_lambda: Float64 = 0.95,
        clip_epsilon: Float64 = 0.2,
        entropy_coef: Float64 = 0.01,
        value_loss_coef: Float64 = 0.5,
        num_epochs: Int = 4,
        minibatch_size: Int = 64,
        normalize_advantages: Bool = True,
        init_value: Float64 = 0.0,
    ):
        """Initialize PPO agent with minibatch updates.

        Args:
            tile_coding: TileCoding instance
            num_actions: Number of discrete actions
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            discount_factor: Discount factor γ
            gae_lambda: GAE parameter λ
            clip_epsilon: PPO clipping parameter ε
            entropy_coef: Entropy bonus coefficient
            value_loss_coef: Value loss coefficient
            num_epochs: Number of optimization epochs
            minibatch_size: Size of minibatches for updates
            normalize_advantages: Whether to normalize advantages
            init_value: Initial parameter value
        """
        self.num_actions = num_actions
        self.num_tiles = tile_coding.get_num_tiles()
        self.num_tilings = tile_coding.get_num_tilings()
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.normalize_advantages = normalize_advantages

        # Initialize actor parameters
        self.theta = List[List[Float64]]()
        for _ in range(num_actions):
            var action_params = List[Float64]()
            for _ in range(self.num_tiles):
                action_params.append(init_value)
            self.theta.append(action_params^)

        # Initialize critic parameters
        self.critic_weights = List[Float64]()
        for _ in range(self.num_tiles):
            self.critic_weights.append(init_value)

        # Initialize rollout buffer
        self.buffer_tiles = List[List[Int]]()
        self.buffer_actions = List[Int]()
        self.buffer_rewards = List[Float64]()
        self.buffer_log_probs = List[Float64]()
        self.buffer_values = List[Float64]()

    fn __copyinit__(out self, existing: Self):
        self.num_actions = existing.num_actions
        self.num_tiles = existing.num_tiles
        self.num_tilings = existing.num_tilings
        self.actor_lr = existing.actor_lr
        self.critic_lr = existing.critic_lr
        self.discount_factor = existing.discount_factor
        self.gae_lambda = existing.gae_lambda
        self.clip_epsilon = existing.clip_epsilon
        self.entropy_coef = existing.entropy_coef
        self.value_loss_coef = existing.value_loss_coef
        self.num_epochs = existing.num_epochs
        self.minibatch_size = existing.minibatch_size
        self.normalize_advantages = existing.normalize_advantages
        self.theta = List[List[Float64]]()
        for a in range(existing.num_actions):
            var action_params = List[Float64]()
            for t in range(existing.num_tiles):
                action_params.append(existing.theta[a][t])
            self.theta.append(action_params^)
        self.critic_weights = List[Float64]()
        for t in range(existing.num_tiles):
            self.critic_weights.append(existing.critic_weights[t])
        self.buffer_tiles = List[List[Int]]()
        self.buffer_actions = List[Int]()
        self.buffer_rewards = List[Float64]()
        self.buffer_log_probs = List[Float64]()
        self.buffer_values = List[Float64]()

    fn __moveinit__(out self, deinit existing: Self):
        self.num_actions = existing.num_actions
        self.num_tiles = existing.num_tiles
        self.num_tilings = existing.num_tilings
        self.actor_lr = existing.actor_lr
        self.critic_lr = existing.critic_lr
        self.discount_factor = existing.discount_factor
        self.gae_lambda = existing.gae_lambda
        self.clip_epsilon = existing.clip_epsilon
        self.entropy_coef = existing.entropy_coef
        self.value_loss_coef = existing.value_loss_coef
        self.num_epochs = existing.num_epochs
        self.minibatch_size = existing.minibatch_size
        self.normalize_advantages = existing.normalize_advantages
        self.theta = existing.theta^
        self.critic_weights = existing.critic_weights^
        self.buffer_tiles = existing.buffer_tiles^
        self.buffer_actions = existing.buffer_actions^
        self.buffer_rewards = existing.buffer_rewards^
        self.buffer_log_probs = existing.buffer_log_probs^
        self.buffer_values = existing.buffer_values^

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

    fn get_value(self, tiles: List[Int]) -> Float64:
        """Get state value estimate."""
        var value: Float64 = 0.0
        for i in range(len(tiles)):
            value += self.critic_weights[tiles[i]]
        return value

    fn get_log_prob(self, tiles: List[Int], action: Int) -> Float64:
        """Get log probability of action."""
        var probs = self.get_action_probs(tiles)
        if probs[action] > 1e-10:
            return log(probs[action])
        return -23.0

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

    fn store_transition(
        mut self,
        tiles: List[Int],
        action: Int,
        reward: Float64,
        log_prob: Float64,
        value: Float64,
    ):
        """Store transition in buffer."""
        var tiles_copy = List[Int]()
        for i in range(len(tiles)):
            tiles_copy.append(tiles[i])

        self.buffer_tiles.append(tiles_copy^)
        self.buffer_actions.append(action)
        self.buffer_rewards.append(reward)
        self.buffer_log_probs.append(log_prob)
        self.buffer_values.append(value)

    fn _get_value_idx(self, buffer_idx: Int) -> Float64:
        """Get current value for buffer step."""
        var value: Float64 = 0.0
        var num_tiles = len(self.buffer_tiles[buffer_idx])
        for i in range(num_tiles):
            var tile_idx = self.buffer_tiles[buffer_idx][i]
            value += self.critic_weights[tile_idx]
        return value

    fn _get_action_probs_idx(self, buffer_idx: Int) -> List[Float64]:
        """Get current action probabilities for buffer step."""
        var preferences = List[Float64]()
        var num_tiles = len(self.buffer_tiles[buffer_idx])
        for a in range(self.num_actions):
            var pref: Float64 = 0.0
            for i in range(num_tiles):
                var tile_idx = self.buffer_tiles[buffer_idx][i]
                pref += self.theta[a][tile_idx]
            preferences.append(pref)
        return self._softmax(preferences^)

    fn _get_log_prob_idx(self, buffer_idx: Int, action: Int) -> Float64:
        """Get current log probability for buffer step."""
        var probs = self._get_action_probs_idx(buffer_idx)
        if probs[action] > 1e-10:
            return log(probs[action])
        return -23.0

    fn _generate_random_indices(self, n: Int) -> List[Int]:
        """Generate shuffled indices for minibatch sampling."""
        var indices = List[Int]()
        for i in range(n):
            indices.append(i)

        # Fisher-Yates shuffle
        for i in range(n - 1, 0, -1):
            var j = Int(random_float64() * Float64(i + 1))
            if j > i:
                j = i
            # Swap
            var temp = indices[i]
            indices[i] = indices[j]
            indices[j] = temp

        return indices^

    fn update(
        mut self,
        next_tiles: List[Int],
        done: Bool,
    ):
        """Update using minibatch PPO.

        Args:
            next_tiles: Tiles for next state
            done: Whether episode terminated
        """
        var buffer_len = len(self.buffer_tiles)
        if buffer_len == 0:
            return

        # Get bootstrap value
        var next_value: Float64 = 0.0
        if not done:
            next_value = self.get_value(next_tiles)

        # Compute GAE advantages
        var advantages = compute_gae(
            self.buffer_rewards,
            self.buffer_values,
            next_value,
            done,
            self.discount_factor,
            self.gae_lambda,
        )

        # Compute returns
        var returns = compute_returns_from_advantages(advantages, self.buffer_values)

        # Normalize advantages
        if self.normalize_advantages and buffer_len > 1:
            var mean: Float64 = 0.0
            for t in range(buffer_len):
                mean += advantages[t]
            mean /= Float64(buffer_len)

            var variance: Float64 = 0.0
            for t in range(buffer_len):
                var diff = advantages[t] - mean
                variance += diff * diff
            variance /= Float64(buffer_len)
            var std = (variance + 1e-8) ** 0.5

            for t in range(buffer_len):
                advantages[t] = (advantages[t] - mean) / std

        var actor_step = self.actor_lr / Float64(self.num_tilings)
        var critic_step = self.critic_lr / Float64(self.num_tilings)

        # Multiple epochs with minibatch updates
        for _ in range(self.num_epochs):
            var indices = self._generate_random_indices(buffer_len)

            var batch_start = 0
            while batch_start < buffer_len:
                var batch_end = min(batch_start + self.minibatch_size, buffer_len)

                # Process minibatch
                for b in range(batch_start, batch_end):
                    var t = indices[b]
                    var action = self.buffer_actions[t]
                    var old_log_prob = self.buffer_log_probs[t]
                    var advantage = advantages[t]
                    var return_t = returns[t]
                    var num_tiles_t = len(self.buffer_tiles[t])

                    # Current policy probability
                    var new_log_prob = self._get_log_prob_idx(t, action)

                    # Probability ratio
                    var ratio = exp(new_log_prob - old_log_prob)

                    # Clipping
                    var clipped = (ratio < 1.0 - self.clip_epsilon) or (ratio > 1.0 + self.clip_epsilon)

                    # Update critic
                    var current_value = self._get_value_idx(t)
                    var value_grad = 2.0 * (current_value - return_t)
                    for i in range(num_tiles_t):
                        var tile_idx = self.buffer_tiles[t][i]
                        self.critic_weights[tile_idx] -= critic_step * self.value_loss_coef * value_grad

                    # Update actor if not clipped
                    if not clipped:
                        var probs = self._get_action_probs_idx(t)

                        for a in range(self.num_actions):
                            var policy_grad: Float64
                            if a == action:
                                policy_grad = 1.0 - probs[a]
                            else:
                                policy_grad = -probs[a]

                            var entropy_grad: Float64 = 0.0
                            if probs[a] > 1e-10:
                                entropy_grad = -probs[a] * (1.0 + log(probs[a])) * probs[a] * (1.0 - probs[a])

                            var total_grad = advantage * policy_grad + self.entropy_coef * entropy_grad

                            for i in range(num_tiles_t):
                                var tile_idx = self.buffer_tiles[t][i]
                                self.theta[a][tile_idx] += actor_step * total_grad

                batch_start = batch_end

        # Clear buffer
        self.buffer_tiles.clear()
        self.buffer_actions.clear()
        self.buffer_rewards.clear()
        self.buffer_log_probs.clear()
        self.buffer_values.clear()

    fn reset(mut self):
        """Reset buffer."""
        self.buffer_tiles.clear()
        self.buffer_actions.clear()
        self.buffer_rewards.clear()
        self.buffer_log_probs.clear()
        self.buffer_values.clear()

    fn get_policy_entropy(self, tiles: List[Int]) -> Float64:
        """Compute policy entropy."""
        var probs = self.get_action_probs(tiles)
        var entropy: Float64 = 0.0
        for a in range(self.num_actions):
            if probs[a] > 1e-10:
                entropy -= probs[a] * log(probs[a])
        return entropy

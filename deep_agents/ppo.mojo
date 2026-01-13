"""Deep PPO (Proximal Policy Optimization) Agent using the new trait-based architecture.

This PPO implementation uses:
- Network wrapper from deep_rl.training for stateless model + params management
- seq() composition for building actor and critic networks
- Clipped surrogate objective for stable policy updates
- GAE (Generalized Advantage Estimation) for variance reduction

Key features:
- Works with any BoxDiscreteActionEnv (continuous obs, discrete actions)
- Clipped policy ratio for stable updates
- Multiple epochs of optimization per rollout
- Entropy bonus for exploration
- Advantage normalization

Architecture:
- Actor: obs -> hidden (ReLU) -> hidden (ReLU) -> num_actions (Softmax)
- Critic: obs -> hidden (ReLU) -> hidden (ReLU) -> 1 (value)

Usage:
    from deep_agents.ppo import DeepPPOAgent
    from envs import CartPoleNative

    var env = CartPoleNative()
    var agent = DeepPPOAgent[4, 2, 128]()

    var metrics = agent.train(env, num_episodes=1000)

Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
"""

from math import exp, log
from random import random_float64, seed

from layout import Layout, LayoutTensor

from deep_rl.constants import dtype, TILE, TPB
from deep_rl.model import Linear, ReLU, seq
from deep_rl.optimizer import Adam
from deep_rl.initializer import Xavier
from deep_rl.training import Network
from core import TrainingMetrics, BoxDiscreteActionEnv
from core.utils.gae import compute_gae_inline
from core.utils.softmax import softmax_inline, sample_from_probs_inline, argmax_probs_inline
from core.utils.normalization import normalize_inline
from core.utils.shuffle import shuffle_indices_inline


# =============================================================================
# Deep PPO Agent
# =============================================================================


struct DeepPPOAgent[
    obs_dim: Int,
    num_actions: Int,
    hidden_dim: Int = 128,
    rollout_len: Int = 2048,
]:
    """Deep Proximal Policy Optimization Agent using new trait-based architecture.

    Uses clipped surrogate objective for stable policy updates:
    L^CLIP = min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)
    where r(θ) = π_θ(a|s) / π_θ_old(a|s)

    Parameters:
        obs_dim: Dimension of observation space.
        num_actions: Number of discrete actions.
        hidden_dim: Hidden layer size (default: 128).
        rollout_len: Number of steps per rollout (default: 2048).
    """

    # Convenience aliases
    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.num_actions
    comptime HIDDEN = Self.hidden_dim
    comptime ROLLOUT = Self.rollout_len

    # Cache sizes
    comptime ACTOR_CACHE: Int = Self.OBS + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN
    comptime CRITIC_CACHE: Int = Self.OBS + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN

    # Actor network: obs -> hidden (ReLU) -> hidden (ReLU) -> action logits
    var actor: Network[
        type_of(
            seq(
                Linear[Self.OBS, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, Self.ACTIONS](),
            )
        ),
        Adam,
        Xavier,
    ]

    # Critic network: obs -> hidden (ReLU) -> hidden (ReLU) -> value
    var critic: Network[
        type_of(
            seq(
                Linear[Self.OBS, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, 1](),
            )
        ),
        Adam,
        Xavier,
    ]

    # Hyperparameters
    var gamma: Float64
    var gae_lambda: Float64
    var clip_epsilon: Float64
    var actor_lr: Float64
    var critic_lr: Float64
    var entropy_coef: Float64
    var value_loss_coef: Float64
    var num_epochs: Int
    var minibatch_size: Int
    var normalize_advantages: Bool

    # Rollout buffers
    var buffer_obs: InlineArray[Scalar[dtype], Self.ROLLOUT * Self.OBS]
    var buffer_actions: InlineArray[Int, Self.ROLLOUT]
    var buffer_rewards: InlineArray[Scalar[dtype], Self.ROLLOUT]
    var buffer_values: InlineArray[Scalar[dtype], Self.ROLLOUT]
    var buffer_log_probs: InlineArray[Scalar[dtype], Self.ROLLOUT]
    var buffer_dones: InlineArray[Bool, Self.ROLLOUT]
    var buffer_idx: Int

    # Training state
    var train_step_count: Int

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        gae_lambda: Float64 = 0.95,
        clip_epsilon: Float64 = 0.2,
        actor_lr: Float64 = 0.0003,
        critic_lr: Float64 = 0.001,
        entropy_coef: Float64 = 0.01,
        value_loss_coef: Float64 = 0.5,
        num_epochs: Int = 4,
        minibatch_size: Int = 64,
        normalize_advantages: Bool = True,
    ):
        """Initialize Deep PPO agent.

        Args:
            gamma: Discount factor (default: 0.99).
            gae_lambda: GAE lambda parameter (default: 0.95).
            clip_epsilon: PPO clipping parameter (default: 0.2).
            actor_lr: Actor learning rate (default: 0.0003).
            critic_lr: Critic learning rate (default: 0.001).
            entropy_coef: Entropy bonus coefficient (default: 0.01).
            value_loss_coef: Value loss coefficient (default: 0.5).
            num_epochs: Number of optimization epochs per update (default: 4).
            minibatch_size: Size of minibatches (default: 64).
            normalize_advantages: Whether to normalize advantages (default: True).
        """
        # Build actor and critic models
        var actor_model = seq(
            Linear[Self.OBS, Self.HIDDEN](),
            ReLU[Self.HIDDEN](),
            Linear[Self.HIDDEN, Self.HIDDEN](),
            ReLU[Self.HIDDEN](),
            Linear[Self.HIDDEN, Self.ACTIONS](),
        )

        var critic_model = seq(
            Linear[Self.OBS, Self.HIDDEN](),
            ReLU[Self.HIDDEN](),
            Linear[Self.HIDDEN, Self.HIDDEN](),
            ReLU[Self.HIDDEN](),
            Linear[Self.HIDDEN, 1](),
        )

        # Initialize networks
        self.actor = Network(actor_model, Adam(lr=actor_lr), Xavier())
        self.critic = Network(critic_model, Adam(lr=critic_lr), Xavier())

        # Store hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.normalize_advantages = normalize_advantages

        # Initialize buffers
        self.buffer_obs = InlineArray[Scalar[dtype], Self.ROLLOUT * Self.OBS](
            fill=0
        )
        self.buffer_actions = InlineArray[Int, Self.ROLLOUT](fill=0)
        self.buffer_rewards = InlineArray[Scalar[dtype], Self.ROLLOUT](fill=0)
        self.buffer_values = InlineArray[Scalar[dtype], Self.ROLLOUT](fill=0)
        self.buffer_log_probs = InlineArray[Scalar[dtype], Self.ROLLOUT](fill=0)
        self.buffer_dones = InlineArray[Bool, Self.ROLLOUT](fill=False)
        self.buffer_idx = 0

        # Training state
        self.train_step_count = 0

    fn _softmax(
        self,
        logits: InlineArray[Scalar[dtype], Self.ACTIONS],
    ) -> InlineArray[Scalar[dtype], Self.ACTIONS]:
        """Compute numerically stable softmax."""
        var probs = InlineArray[Scalar[dtype], Self.ACTIONS](fill=0)

        # Find max for numerical stability
        var max_logit = logits[0]
        for i in range(1, Self.ACTIONS):
            if logits[i] > max_logit:
                max_logit = logits[i]

        # Compute exp and sum
        var sum_exp = Scalar[dtype](0.0)
        for i in range(Self.ACTIONS):
            probs[i] = exp(logits[i] - max_logit)
            sum_exp += probs[i]

        # Normalize
        for i in range(Self.ACTIONS):
            probs[i] /= sum_exp

        return probs

    fn select_action(
        self,
        obs: InlineArray[Scalar[dtype], Self.OBS],
        training: Bool = True,
    ) -> Tuple[Int, Scalar[dtype], Scalar[dtype]]:
        """Select action from policy and compute log probability and value.

        Args:
            obs: Current observation.
            training: If True, sample action; else use greedy.

        Returns:
            Tuple of (action, log_prob, value).
        """
        # Forward actor to get logits
        var logits = InlineArray[Scalar[dtype], Self.ACTIONS](uninitialized=True)
        self.actor.forward[1](obs, logits)

        # Compute softmax probabilities
        var probs = self._softmax(logits)

        # Forward critic to get value
        var value_out = InlineArray[Scalar[dtype], 1](uninitialized=True)
        self.critic.forward[1](obs, value_out)
        var value = value_out[0]

        # Sample or select greedy action
        var action: Int
        if training:
            # Sample from categorical distribution
            var rand = Scalar[dtype](random_float64())
            var cumsum = Scalar[dtype](0.0)
            action = Self.ACTIONS - 1
            for a in range(Self.ACTIONS):
                cumsum += probs[a]
                if rand < cumsum:
                    action = a
                    break
        else:
            # Greedy action
            action = 0
            var best_prob = probs[0]
            for a in range(1, Self.ACTIONS):
                if probs[a] > best_prob:
                    best_prob = probs[a]
                    action = a

        # Compute log probability
        var log_prob = log(probs[action] + Scalar[dtype](1e-8))

        return (action, log_prob, value)

    fn store_transition(
        mut self,
        obs: InlineArray[Scalar[dtype], Self.OBS],
        action: Int,
        reward: Float64,
        log_prob: Scalar[dtype],
        value: Scalar[dtype],
        done: Bool,
    ):
        """Store transition in rollout buffer."""
        # Store observation
        for i in range(Self.OBS):
            self.buffer_obs[self.buffer_idx * Self.OBS + i] = obs[i]

        self.buffer_actions[self.buffer_idx] = action
        self.buffer_rewards[self.buffer_idx] = Scalar[dtype](reward)
        self.buffer_log_probs[self.buffer_idx] = log_prob
        self.buffer_values[self.buffer_idx] = value
        self.buffer_dones[self.buffer_idx] = done

        self.buffer_idx += 1

    fn _shuffle_indices(self, n: Int, mut indices: InlineArray[Int, Self.ROLLOUT]):
        """Generate shuffled indices using Fisher-Yates."""
        for i in range(n):
            indices[i] = i

        # Fisher-Yates shuffle
        for i in range(n - 1, 0, -1):
            var j = Int(random_float64() * Float64(i + 1))
            if j > i:
                j = i
            # Swap
            var temp = indices[i]
            indices[i] = indices[j]
            indices[j] = temp

    fn update(
        mut self,
        next_obs: InlineArray[Scalar[dtype], Self.OBS],
    ) -> Float64:
        """Update actor and critic using PPO with clipped objective.

        Args:
            next_obs: Next observation for bootstrapping.

        Returns:
            Total loss value.
        """
        if self.buffer_idx == 0:
            return 0.0

        var buffer_len = self.buffer_idx

        # Get bootstrap value
        var next_value_out = InlineArray[Scalar[dtype], 1](uninitialized=True)
        self.critic.forward[1](next_obs, next_value_out)
        var next_value = next_value_out[0]

        # Compute GAE advantages and returns
        var advantages = InlineArray[Scalar[dtype], Self.ROLLOUT](fill=0)
        var returns = InlineArray[Scalar[dtype], Self.ROLLOUT](fill=0)

        compute_gae_inline[dtype, Self.ROLLOUT](
            self.buffer_rewards,
            self.buffer_values,
            next_value,
            self.buffer_dones,
            self.gamma,
            self.gae_lambda,
            buffer_len,
            advantages,
            returns,
        )

        # Normalize advantages
        if self.normalize_advantages and buffer_len > 1:
            var adv_mean = Scalar[dtype](0.0)
            for t in range(buffer_len):
                adv_mean += advantages[t]
            adv_mean /= Scalar[dtype](buffer_len)

            var adv_var = Scalar[dtype](0.0)
            for t in range(buffer_len):
                var diff = advantages[t] - adv_mean
                adv_var += diff * diff
            adv_var /= Scalar[dtype](buffer_len)
            var adv_std = (adv_var + Scalar[dtype](1e-8)) ** 0.5

            for t in range(buffer_len):
                advantages[t] = (advantages[t] - adv_mean) / adv_std

        # =====================================================================
        # Multiple epochs of optimization
        # =====================================================================

        var total_loss = Scalar[dtype](0.0)
        var indices = InlineArray[Int, Self.ROLLOUT](fill=0)

        for epoch in range(self.num_epochs):
            # Shuffle indices for minibatch sampling
            self._shuffle_indices(buffer_len, indices)

            var batch_start = 0
            while batch_start < buffer_len:
                var batch_end = batch_start + self.minibatch_size
                if batch_end > buffer_len:
                    batch_end = buffer_len

                # Process minibatch
                for b in range(batch_start, batch_end):
                    var t = indices[b]

                    # Get observation for this timestep
                    var obs = InlineArray[Scalar[dtype], Self.OBS](fill=0)
                    for i in range(Self.OBS):
                        obs[i] = self.buffer_obs[t * Self.OBS + i]

                    var action = self.buffer_actions[t]
                    var old_log_prob = self.buffer_log_probs[t]
                    var advantage = advantages[t]
                    var return_t = returns[t]

                    # ==========================================================
                    # Actor forward and update
                    # ==========================================================
                    var logits = InlineArray[Scalar[dtype], Self.ACTIONS](
                        uninitialized=True
                    )
                    var actor_cache = InlineArray[
                        Scalar[dtype], Self.ACTOR_CACHE
                    ](uninitialized=True)
                    self.actor.forward_with_cache[1](obs, logits, actor_cache)

                    var probs = self._softmax(logits)
                    var new_log_prob = log(probs[action] + Scalar[dtype](1e-8))

                    # Probability ratio r(θ) = π_θ(a|s) / π_θ_old(a|s)
                    var ratio = exp(new_log_prob - old_log_prob)

                    # Clipped surrogate objective
                    var surr1 = ratio * advantage
                    var clipped_ratio: Scalar[dtype]
                    if advantage >= Scalar[dtype](0.0):
                        clipped_ratio = min(
                            ratio, Scalar[dtype](1.0 + self.clip_epsilon)
                        )
                    else:
                        clipped_ratio = max(
                            ratio, Scalar[dtype](1.0 - self.clip_epsilon)
                        )
                    var surr2 = clipped_ratio * advantage

                    # Policy loss: -min(surr1, surr2)
                    var policy_loss: Scalar[dtype]
                    if surr1 < surr2:
                        policy_loss = -surr1
                    else:
                        policy_loss = -surr2

                    # Entropy bonus
                    var entropy = Scalar[dtype](0.0)
                    for a in range(Self.ACTIONS):
                        if probs[a] > Scalar[dtype](1e-8):
                            entropy -= probs[a] * log(probs[a])

                    # Check if ratio is clipped
                    var is_clipped = (
                        ratio < Scalar[dtype](1.0 - self.clip_epsilon)
                    ) or (ratio > Scalar[dtype](1.0 + self.clip_epsilon))

                    # Actor gradient (only if not clipped)
                    var d_logits = InlineArray[Scalar[dtype], Self.ACTIONS](
                        fill=0
                    )
                    if not is_clipped:
                        for a in range(Self.ACTIONS):
                            var d_log_prob: Scalar[dtype]
                            if a == action:
                                d_log_prob = Scalar[dtype](1.0) - probs[a]
                            else:
                                d_log_prob = -probs[a]

                            # Entropy gradient
                            var d_entropy = -probs[a] * (
                                Scalar[dtype](1.0)
                                + log(probs[a] + Scalar[dtype](1e-8))
                            )

                            d_logits[a] = (
                                -advantage * ratio * d_log_prob
                                - Scalar[dtype](self.entropy_coef) * d_entropy
                            )

                    # Backward through actor
                    var actor_grad_input = InlineArray[Scalar[dtype], Self.OBS](
                        fill=0
                    )
                    self.actor.zero_grads()
                    self.actor.backward[1](
                        d_logits, actor_grad_input, actor_cache
                    )
                    self.actor.update()

                    # ==========================================================
                    # Critic forward and update
                    # ==========================================================
                    var value_out = InlineArray[Scalar[dtype], 1](
                        uninitialized=True
                    )
                    var critic_cache = InlineArray[
                        Scalar[dtype], Self.CRITIC_CACHE
                    ](uninitialized=True)
                    self.critic.forward_with_cache[1](obs, value_out, critic_cache)

                    var value = value_out[0]

                    # Value loss: (return - value)^2
                    var value_loss = (return_t - value) * (return_t - value)

                    # Critic gradient
                    var d_value = InlineArray[Scalar[dtype], 1](fill=0)
                    d_value[0] = (
                        Scalar[dtype](2.0)
                        * Scalar[dtype](self.value_loss_coef)
                        * (value - return_t)
                    )

                    # Backward through critic
                    var critic_grad_input = InlineArray[Scalar[dtype], Self.OBS](
                        fill=0
                    )
                    self.critic.zero_grads()
                    self.critic.backward[1](
                        d_value, critic_grad_input, critic_cache
                    )
                    self.critic.update()

                    total_loss += (
                        policy_loss
                        + Scalar[dtype](self.value_loss_coef) * value_loss
                        - Scalar[dtype](self.entropy_coef) * entropy
                    )

                batch_start = batch_end

        # Clear buffer
        self.buffer_idx = 0
        self.train_step_count += 1

        return Float64(
            total_loss / Scalar[dtype](self.num_epochs * buffer_len)
        )

    fn _list_to_inline(
        self, obs_list: List[Float64]
    ) -> InlineArray[Scalar[dtype], Self.OBS]:
        """Convert List[Float64] to InlineArray."""
        var obs = InlineArray[Scalar[dtype], Self.OBS](fill=0)
        for i in range(Self.OBS):
            if i < len(obs_list):
                obs[i] = Scalar[dtype](obs_list[i])
        return obs

    fn train[
        E: BoxDiscreteActionEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int,
        max_steps_per_episode: Int = 1000,
        verbose: Bool = False,
        print_every: Int = 10,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train the PPO agent on a discrete action environment.

        Args:
            env: The environment to train on.
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
            verbose: Whether to print progress.
            print_every: Print progress every N episodes if verbose.
            environment_name: Name of environment for metrics labeling.

        Returns:
            TrainingMetrics object with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="Deep PPO",
            environment_name=environment_name,
        )

        var total_steps = 0

        for episode in range(num_episodes):
            var obs_list = env.reset_obs_list()
            var obs = self._list_to_inline(obs_list)
            var episode_reward: Float64 = 0.0
            var episode_steps = 0

            for step in range(max_steps_per_episode):
                # Select action
                var action_result = self.select_action(obs, training=True)
                var action = action_result[0]
                var log_prob = action_result[1]
                var value = action_result[2]

                # Step environment
                var result = env.step_obs(action)
                var next_obs_list = result[0].copy()
                var reward = result[1]
                var done = result[2]

                var next_obs = self._list_to_inline(next_obs_list)

                # Store transition
                self.store_transition(obs, action, reward, log_prob, value, done)

                episode_reward += reward
                obs = next_obs
                total_steps += 1
                episode_steps += 1

                # Update at rollout boundary or episode end
                if self.buffer_idx >= Self.ROLLOUT or done:
                    _ = self.update(obs)

                if done:
                    break

            # Log metrics
            metrics.log_episode(episode, episode_reward, episode_steps, 0.0)

            # Print progress
            if verbose and (episode + 1) % print_every == 0:
                var avg_reward = metrics.mean_reward_last_n(print_every)
                print(
                    "Episode",
                    episode + 1,
                    "| Avg reward:",
                    String(avg_reward)[:7],
                    "| Steps:",
                    total_steps,
                )

        return metrics^

    fn evaluate[
        E: BoxDiscreteActionEnv
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps: Int = 1000,
        verbose: Bool = False,
        render: Bool = False,
    ) -> Float64:
        """Evaluate the agent using greedy policy.

        Args:
            env: The environment to evaluate on.
            num_episodes: Number of evaluation episodes.
            max_steps: Maximum steps per episode.
            verbose: Whether to print per-episode results.
            render: Whether to render the environment.

        Returns:
            Average reward over evaluation episodes.
        """
        var total_reward: Float64 = 0.0

        for episode in range(num_episodes):
            var obs_list = env.reset_obs_list()
            var obs = self._list_to_inline(obs_list)
            var episode_reward: Float64 = 0.0
            var episode_steps = 0

            for step in range(max_steps):
                # Greedy action
                var action_result = self.select_action(obs, training=False)
                var action = action_result[0]

                # Step environment
                var result = env.step_obs(action)
                var next_obs_list = result[0].copy()
                var reward = result[1]
                var done = result[2]

                if render:
                    env.render()

                episode_reward += reward
                obs = self._list_to_inline(next_obs_list)
                episode_steps += 1

                if done:
                    break

            total_reward += episode_reward

            if verbose:
                print(
                    "Eval Episode",
                    episode + 1,
                    "| Reward:",
                    String(episode_reward)[:10],
                    "| Steps:",
                    episode_steps,
                )

        return total_reward / Float64(num_episodes)

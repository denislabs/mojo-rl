"""Deep Dueling DQN Agent using the new trait-based deep learning architecture.

This Dueling DQN implementation uses:
- Network wrapper from deep_rl.training for stateless model + params management
- seq() composition for building network components
- ReplayBuffer from deep_rl.replay for experience replay

Dueling Architecture:
- Shared backbone: obs -> h1 (ReLU) -> h2 (ReLU)
- Value stream: h2 -> value_hidden (ReLU) -> V(s) [scalar]
- Advantage stream: h2 -> adv_hidden (ReLU) -> A(s,a) [num_actions]
- Q(s,a) = V(s) + (A(s,a) - mean(A))

This decomposition helps the network learn which states are valuable
without having to learn the effect of each action for every state.
Particularly useful when actions don't always affect the outcome.

Features:
- Works with any BoxDiscreteActionEnv (continuous obs, discrete actions)
- Epsilon-greedy exploration with decay
- Target network with soft updates
- Double DQN support (online selects, target evaluates)

Usage:
    from deep_agents.dueling_dqn import DuelingDQNAgent
    from envs import LunarLanderEnv

    var env = LunarLanderEnv()
    var agent = DuelingDQNAgent[8, 4, 128, 64, 100000, 64]()

    var metrics = agent.train(env, num_episodes=500)

Reference: Wang et al. "Dueling Network Architectures for Deep RL" (2016)
"""

from math import exp
from random import random_float64, seed

from layout import Layout, LayoutTensor

from deep_rl.constants import dtype, TILE, TPB
from deep_rl.model import Linear, ReLU, seq
from deep_rl.optimizer import Adam
from deep_rl.initializer import Kaiming
from deep_rl.training import Network
from deep_rl.replay import ReplayBuffer
from core import TrainingMetrics, BoxDiscreteActionEnv


# =============================================================================
# Deep Dueling DQN Agent
# =============================================================================


struct DuelingDQNAgent[
    obs_dim: Int,
    num_actions: Int,
    hidden_dim: Int = 128,
    stream_hidden_dim: Int = 64,
    buffer_capacity: Int = 100000,
    batch_size: Int = 64,
    double_dqn: Bool = True,
]:
    """Deep Dueling DQN Agent using the new trait-based architecture.

    Dueling DQN separates the Q-network into two streams:
    - Value stream V(s): Estimates how good is this state
    - Advantage stream A(s,a): Estimates relative action advantages

    Final Q-values: Q(s,a) = V(s) + (A(s,a) - mean(A))

    Parameters:
        obs_dim: Dimension of observation space.
        num_actions: Number of discrete actions.
        hidden_dim: Hidden layer size for shared backbone.
        stream_hidden_dim: Hidden layer size for value/advantage streams.
        buffer_capacity: Replay buffer capacity.
        batch_size: Training batch size.
        double_dqn: If True, use Double DQN target computation.
    """

    # Convenience aliases
    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.num_actions
    comptime HIDDEN = Self.hidden_dim
    comptime STREAM_H = Self.stream_hidden_dim
    comptime BATCH = Self.batch_size

    # Cache sizes for networks
    # Backbone: Linear[obs, h] + ReLU[h] + Linear[h, h] + ReLU[h]
    # Cache: OBS + HIDDEN + HIDDEN + HIDDEN
    comptime BACKBONE_CACHE_SIZE: Int = Self.OBS + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN

    # Value head: Linear[h, stream_h] + ReLU[stream_h] + Linear[stream_h, 1]
    # Cache: HIDDEN + STREAM_H + STREAM_H
    comptime VALUE_CACHE_SIZE: Int = Self.HIDDEN + Self.STREAM_H + Self.STREAM_H

    # Advantage head: Linear[h, stream_h] + ReLU[stream_h] + Linear[stream_h, num_actions]
    # Cache: HIDDEN + STREAM_H + STREAM_H
    comptime ADV_CACHE_SIZE: Int = Self.HIDDEN + Self.STREAM_H + Self.STREAM_H

    # =========================================================================
    # Online Networks
    # =========================================================================

    # Shared backbone: obs -> hidden (ReLU) -> hidden (ReLU)
    var backbone: Network[
        type_of(
            seq(
                Linear[Self.OBS, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
            )
        ),
        Adam,
        Kaiming,
    ]

    # Value head: hidden -> stream_hidden (ReLU) -> 1
    var value_head: Network[
        type_of(
            seq(
                Linear[Self.HIDDEN, Self.STREAM_H](),
                ReLU[Self.STREAM_H](),
                Linear[Self.STREAM_H, 1](),
            )
        ),
        Adam,
        Kaiming,
    ]

    # Advantage head: hidden -> stream_hidden (ReLU) -> num_actions
    var advantage_head: Network[
        type_of(
            seq(
                Linear[Self.HIDDEN, Self.STREAM_H](),
                ReLU[Self.STREAM_H](),
                Linear[Self.STREAM_H, Self.ACTIONS](),
            )
        ),
        Adam,
        Kaiming,
    ]

    # =========================================================================
    # Target Networks
    # =========================================================================

    var backbone_target: Network[
        type_of(
            seq(
                Linear[Self.OBS, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
                Linear[Self.HIDDEN, Self.HIDDEN](),
                ReLU[Self.HIDDEN](),
            )
        ),
        Adam,
        Kaiming,
    ]

    var value_head_target: Network[
        type_of(
            seq(
                Linear[Self.HIDDEN, Self.STREAM_H](),
                ReLU[Self.STREAM_H](),
                Linear[Self.STREAM_H, 1](),
            )
        ),
        Adam,
        Kaiming,
    ]

    var advantage_head_target: Network[
        type_of(
            seq(
                Linear[Self.HIDDEN, Self.STREAM_H](),
                ReLU[Self.STREAM_H](),
                Linear[Self.STREAM_H, Self.ACTIONS](),
            )
        ),
        Adam,
        Kaiming,
    ]

    # Replay buffer (action_dim=1 for discrete actions stored as scalar)
    var buffer: ReplayBuffer[Self.buffer_capacity, Self.obs_dim, 1, dtype]

    # Hyperparameters
    var gamma: Float64
    var tau: Float64
    var lr: Float64
    var epsilon: Float64
    var epsilon_min: Float64
    var epsilon_decay: Float64

    # Training state
    var total_steps: Int
    var train_step_count: Int

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        tau: Float64 = 0.005,
        lr: Float64 = 0.0005,
        epsilon: Float64 = 1.0,
        epsilon_min: Float64 = 0.01,
        epsilon_decay: Float64 = 0.995,
    ):
        """Initialize Deep Dueling DQN agent.

        Args:
            gamma: Discount factor (default: 0.99).
            tau: Soft update coefficient (default: 0.005).
            lr: Learning rate (default: 0.0005).
            epsilon: Initial exploration rate (default: 1.0).
            epsilon_min: Minimum exploration rate (default: 0.01).
            epsilon_decay: Epsilon decay per episode (default: 0.995).
        """
        # Build models
        var backbone_model = seq(
            Linear[Self.OBS, Self.HIDDEN](),
            ReLU[Self.HIDDEN](),
            Linear[Self.HIDDEN, Self.HIDDEN](),
            ReLU[Self.HIDDEN](),
        )

        var value_model = seq(
            Linear[Self.HIDDEN, Self.STREAM_H](),
            ReLU[Self.STREAM_H](),
            Linear[Self.STREAM_H, 1](),
        )

        var advantage_model = seq(
            Linear[Self.HIDDEN, Self.STREAM_H](),
            ReLU[Self.STREAM_H](),
            Linear[Self.STREAM_H, Self.ACTIONS](),
        )

        # Initialize online networks
        self.backbone = Network(backbone_model, Adam(lr=lr), Kaiming())
        self.value_head = Network(value_model, Adam(lr=lr), Kaiming())
        self.advantage_head = Network(advantage_model, Adam(lr=lr), Kaiming())

        # Initialize target networks
        self.backbone_target = Network(backbone_model, Adam(lr=lr), Kaiming())
        self.value_head_target = Network(value_model, Adam(lr=lr), Kaiming())
        self.advantage_head_target = Network(
            advantage_model, Adam(lr=lr), Kaiming()
        )

        # Copy weights to target networks
        self.backbone_target.copy_params_from(self.backbone)
        self.value_head_target.copy_params_from(self.value_head)
        self.advantage_head_target.copy_params_from(self.advantage_head)

        # Initialize replay buffer
        self.buffer = ReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, 1, dtype
        ]()

        # Store hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Training state
        self.total_steps = 0
        self.train_step_count = 0

    fn _dueling_forward[
        BATCH: Int
    ](
        self,
        obs: InlineArray[Scalar[dtype], BATCH * Self.OBS],
        mut q_values: InlineArray[Scalar[dtype], BATCH * Self.ACTIONS],
        use_target: Bool = False,
    ):
        """Forward pass through dueling network (online or target).

        Computes: Q(s,a) = V(s) + (A(s,a) - mean(A))
        """
        # Forward through backbone
        var h2 = InlineArray[Scalar[dtype], BATCH * Self.HIDDEN](
            uninitialized=True
        )
        if use_target:
            self.backbone_target.forward[BATCH](obs, h2)
        else:
            self.backbone.forward[BATCH](obs, h2)

        # Forward through value head
        var value = InlineArray[Scalar[dtype], BATCH](uninitialized=True)
        if use_target:
            self.value_head_target.forward[BATCH](h2, value)
        else:
            self.value_head.forward[BATCH](h2, value)

        # Forward through advantage head
        var advantage = InlineArray[Scalar[dtype], BATCH * Self.ACTIONS](
            uninitialized=True
        )
        if use_target:
            self.advantage_head_target.forward[BATCH](h2, advantage)
        else:
            self.advantage_head.forward[BATCH](h2, advantage)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A))
        for b in range(BATCH):
            # Compute mean advantage
            var mean_adv: Scalar[dtype] = 0.0
            for a in range(Self.ACTIONS):
                mean_adv += advantage[b * Self.ACTIONS + a]
            mean_adv /= Scalar[dtype](Self.ACTIONS)

            # Compute Q-values
            var v_s = value[b]
            for a in range(Self.ACTIONS):
                var adv_idx = b * Self.ACTIONS + a
                q_values[adv_idx] = v_s + (advantage[adv_idx] - mean_adv)

    fn select_action(
        self,
        obs: InlineArray[Scalar[dtype], Self.OBS],
        training: Bool = True,
    ) -> Int:
        """Select action using epsilon-greedy policy.

        Args:
            obs: Current observation.
            training: If True, use epsilon-greedy; else use greedy.

        Returns:
            Selected action index.
        """
        # Epsilon-greedy exploration
        if training and random_float64() < self.epsilon:
            return Int(random_float64() * Float64(Self.ACTIONS)) % Self.ACTIONS

        # Greedy action: argmax Q(s, a)
        var q_values = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        self._dueling_forward[1](obs, q_values, use_target=False)

        var best_action = 0
        var best_q = q_values[0]
        for a in range(1, Self.ACTIONS):
            if q_values[a] > best_q:
                best_q = q_values[a]
                best_action = a

        return best_action

    fn store_transition(
        mut self,
        obs: InlineArray[Scalar[dtype], Self.OBS],
        action: Int,
        reward: Float64,
        next_obs: InlineArray[Scalar[dtype], Self.OBS],
        done: Bool,
    ):
        """Store a transition in the replay buffer."""
        var action_arr = InlineArray[Scalar[dtype], 1](fill=0)
        action_arr[0] = Scalar[dtype](action)
        self.buffer.add(obs, action_arr, Scalar[dtype](reward), next_obs, done)
        self.total_steps += 1

    fn train_step(mut self) -> Float64:
        """Perform one training step.

        Returns:
            TD loss value.
        """
        if not self.buffer.is_ready[Self.BATCH]():
            return 0.0

        # =====================================================================
        # Phase 1: Sample batch from buffer
        # =====================================================================
        var batch_obs = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var batch_actions_arr = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        var batch_rewards = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        var batch_next_obs = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var batch_dones = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )

        self.buffer.sample[Self.BATCH](
            batch_obs,
            batch_actions_arr,
            batch_rewards,
            batch_next_obs,
            batch_dones,
        )

        # =====================================================================
        # Phase 2: Compute TD targets
        # =====================================================================

        var max_next_q = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )

        @parameter
        if Self.double_dqn:
            # Double DQN: online network selects best action, target evaluates it
            var online_next_q = InlineArray[
                Scalar[dtype], Self.BATCH * Self.ACTIONS
            ](uninitialized=True)
            var target_next_q = InlineArray[
                Scalar[dtype], Self.BATCH * Self.ACTIONS
            ](uninitialized=True)

            self._dueling_forward[Self.BATCH](
                batch_next_obs, online_next_q, use_target=False
            )
            self._dueling_forward[Self.BATCH](
                batch_next_obs, target_next_q, use_target=True
            )

            for b in range(Self.BATCH):
                # Online selects best action
                var best_action = 0
                var best_online_q = online_next_q[b * Self.ACTIONS]
                for a in range(1, Self.ACTIONS):
                    var q = online_next_q[b * Self.ACTIONS + a]
                    if q > best_online_q:
                        best_online_q = q
                        best_action = a

                # Target evaluates that action
                max_next_q[b] = target_next_q[b * Self.ACTIONS + best_action]
        else:
            # Standard DQN: max_a Q_target(s', a)
            var next_q = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
                uninitialized=True
            )
            self._dueling_forward[Self.BATCH](
                batch_next_obs, next_q, use_target=True
            )

            for b in range(Self.BATCH):
                var max_q = next_q[b * Self.ACTIONS]
                for a in range(1, Self.ACTIONS):
                    var q = next_q[b * Self.ACTIONS + a]
                    if q > max_q:
                        max_q = q
                max_next_q[b] = max_q

        # Compute TD targets: y = r + gamma * max_next_q * (1 - done)
        var targets = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)
        for b in range(Self.BATCH):
            var done_mask = Scalar[dtype](1.0) - batch_dones[b]
            targets[b] = (
                batch_rewards[b]
                + Scalar[dtype](self.gamma) * max_next_q[b] * done_mask
            )

        # =====================================================================
        # Phase 3: Forward with cache for backward pass
        # =====================================================================

        # Backbone forward with cache
        var h2 = InlineArray[Scalar[dtype], Self.BATCH * Self.HIDDEN](
            uninitialized=True
        )
        var backbone_cache = InlineArray[
            Scalar[dtype], Self.BATCH * Self.BACKBONE_CACHE_SIZE
        ](uninitialized=True)
        self.backbone.forward_with_cache[Self.BATCH](
            batch_obs, h2, backbone_cache
        )

        # Value head forward with cache
        var value = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)
        var value_cache = InlineArray[
            Scalar[dtype], Self.BATCH * Self.VALUE_CACHE_SIZE
        ](uninitialized=True)
        self.value_head.forward_with_cache[Self.BATCH](h2, value, value_cache)

        # Advantage head forward with cache
        var advantage = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        var adv_cache = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ADV_CACHE_SIZE
        ](uninitialized=True)
        self.advantage_head.forward_with_cache[Self.BATCH](
            h2, advantage, adv_cache
        )

        # Compute Q-values: Q(s,a) = V(s) + (A(s,a) - mean(A))
        var q_values = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        for b in range(Self.BATCH):
            var mean_adv: Scalar[dtype] = 0.0
            for a in range(Self.ACTIONS):
                mean_adv += advantage[b * Self.ACTIONS + a]
            mean_adv /= Scalar[dtype](Self.ACTIONS)

            var v_s = value[b]
            for a in range(Self.ACTIONS):
                var idx = b * Self.ACTIONS + a
                q_values[idx] = v_s + (advantage[idx] - mean_adv)

        # =====================================================================
        # Phase 4: Compute loss and gradients
        # =====================================================================

        var loss: Float64 = 0.0
        var dq = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](fill=0)

        for b in range(Self.BATCH):
            var action_idx = Int(batch_actions_arr[b])
            var q_idx = b * Self.ACTIONS + action_idx
            var td_error = q_values[q_idx] - targets[b]
            loss += Float64(td_error * td_error)
            # Only backprop through the taken action
            dq[q_idx] = (
                Scalar[dtype](2.0) * td_error / Scalar[dtype](Self.BATCH)
            )

        loss /= Float64(Self.BATCH)

        # =====================================================================
        # Phase 5: Backward through dueling network
        # =====================================================================

        # Compute dV and dA from dQ
        # Q(s,a) = V(s) + A(s,a) - mean(A)
        # dQ/dV = 1 for all actions that were taken
        # dQ/dA_i = 1 - 1/num_actions (if action i was taken) or -1/num_actions (otherwise)

        var dv = InlineArray[Scalar[dtype], Self.BATCH](fill=0)
        var da = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](fill=0)

        var one_over_n = Scalar[dtype](1.0) / Scalar[dtype](Self.ACTIONS)

        for b in range(Self.BATCH):
            # dV = sum of all dQ for this sample (V contributes to all Q values equally)
            var sum_dq: Scalar[dtype] = 0.0
            for a in range(Self.ACTIONS):
                sum_dq += dq[b * Self.ACTIONS + a]
            dv[b] = sum_dq

            # dA_i = dQ_i - (1/n) * sum(dQ_j)
            # Because Q_i = V + A_i - (1/n)*sum(A_j)
            # So dQ_i/dA_j = delta_ij - 1/n
            for a in range(Self.ACTIONS):
                var idx = b * Self.ACTIONS + a
                da[idx] = dq[idx] - one_over_n * sum_dq

        # Backward through value head
        var dh2_from_v = InlineArray[Scalar[dtype], Self.BATCH * Self.HIDDEN](
            uninitialized=True
        )
        self.value_head.zero_grads()
        self.value_head.backward[Self.BATCH](dv, dh2_from_v, value_cache)

        # Backward through advantage head
        var dh2_from_a = InlineArray[Scalar[dtype], Self.BATCH * Self.HIDDEN](
            uninitialized=True
        )
        self.advantage_head.zero_grads()
        self.advantage_head.backward[Self.BATCH](da, dh2_from_a, adv_cache)

        # Combine gradients from both streams
        var dh2 = InlineArray[Scalar[dtype], Self.BATCH * Self.HIDDEN](
            uninitialized=True
        )
        for i in range(Self.BATCH * Self.HIDDEN):
            dh2[i] = dh2_from_v[i] + dh2_from_a[i]

        # Backward through backbone
        var dobs = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        self.backbone.zero_grads()
        self.backbone.backward[Self.BATCH](dh2, dobs, backbone_cache)

        # Update all networks
        self.backbone.update()
        self.value_head.update()
        self.advantage_head.update()

        # =====================================================================
        # Phase 6: Soft update target networks
        # =====================================================================

        self.backbone_target.soft_update_from(self.backbone, self.tau)
        self.value_head_target.soft_update_from(self.value_head, self.tau)
        self.advantage_head_target.soft_update_from(
            self.advantage_head, self.tau
        )

        self.train_step_count += 1

        return loss

    fn decay_epsilon(mut self):
        """Decay exploration rate (call once per episode)."""
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    fn _list_to_inline(
        self, obs_list: List[Float64]
    ) -> InlineArray[Scalar[dtype], Self.OBS]:
        """Convert List[Float64] to InlineArray."""
        var obs = InlineArray[Scalar[dtype], Self.OBS](fill=0)
        for i in range(Self.OBS):
            if i < len(obs_list):
                obs[i] = Scalar[dtype](obs_list[i])
        return obs

    fn get_epsilon(self) -> Float64:
        """Get current exploration rate."""
        return self.epsilon

    fn get_train_steps(self) -> Int:
        """Get total training steps performed."""
        return self.train_step_count

    fn train[
        E: BoxDiscreteActionEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int,
        max_steps_per_episode: Int = 1000,
        warmup_steps: Int = 1000,
        train_every: Int = 1,
        verbose: Bool = False,
        print_every: Int = 10,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train the Dueling DQN agent on a discrete action environment.

        Args:
            env: The environment to train on (must implement BoxDiscreteActionEnv).
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
            warmup_steps: Number of random steps to fill replay buffer.
            train_every: Train every N steps.
            verbose: Whether to print progress.
            print_every: Print progress every N episodes if verbose.
            environment_name: Name of environment for metrics labeling.

        Returns:
            TrainingMetrics object with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="Deep Dueling DQN",
            environment_name=environment_name,
        )

        # =====================================================================
        # Warmup: fill replay buffer with random actions
        # =====================================================================
        var warmup_obs_list = env.reset_obs_list()
        var warmup_obs = self._list_to_inline(warmup_obs_list)
        var warmup_count = 0

        while warmup_count < warmup_steps:
            # Random action
            var action = (
                Int(random_float64() * Float64(Self.ACTIONS)) % Self.ACTIONS
            )

            # Step environment
            var result = env.step_obs(action)
            var next_obs_list = result[0].copy()
            var reward = result[1]
            var done = result[2]

            var next_obs = self._list_to_inline(next_obs_list)
            self.store_transition(warmup_obs, action, reward, next_obs, done)

            warmup_obs = next_obs
            warmup_count += 1

            if done:
                warmup_obs_list = env.reset_obs_list()
                warmup_obs = self._list_to_inline(warmup_obs_list)

        if verbose:
            print("Warmup complete:", warmup_count, "transitions collected")

        # =====================================================================
        # Training loop
        # =====================================================================
        var total_train_steps = 0

        for episode in range(num_episodes):
            var obs_list = env.reset_obs_list()
            var obs = self._list_to_inline(obs_list)
            var episode_reward: Float64 = 0.0
            var episode_steps = 0

            for step in range(max_steps_per_episode):
                # Select action with epsilon-greedy
                var action = self.select_action(obs, training=True)

                # Step environment
                var result = env.step_obs(action)
                var next_obs_list = result[0].copy()
                var reward = result[1]
                var done = result[2]

                var next_obs = self._list_to_inline(next_obs_list)

                # Store transition
                self.store_transition(obs, action, reward, next_obs, done)

                # Train every N steps
                if total_train_steps % train_every == 0:
                    _ = self.train_step()

                episode_reward += reward
                obs = next_obs
                total_train_steps += 1
                episode_steps += 1

                if done:
                    break

            # Decay exploration rate
            self.decay_epsilon()

            # Log metrics
            metrics.log_episode(
                episode, episode_reward, episode_steps, self.epsilon
            )

            # Print progress
            if verbose and (episode + 1) % print_every == 0:
                var avg_reward = metrics.mean_reward_last_n(print_every)
                print(
                    "Episode",
                    episode + 1,
                    "| Avg reward:",
                    String(avg_reward)[:7],
                    "| Epsilon:",
                    String(self.epsilon)[:5],
                    "| Steps:",
                    total_train_steps,
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
        """Evaluate the agent using greedy policy (no exploration).

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
                # Greedy action (no exploration)
                var action = self.select_action(obs, training=False)

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

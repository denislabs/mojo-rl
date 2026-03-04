"""Deep Dueling DQN Agent using the new trait-based deep learning architecture.

This Dueling DQN implementation uses:
- NetworkState for heap-allocated params + grads + optimizer state
- Network (all-static) for stateless forward/backward ops via LayoutTensor
- ReplayBuffer from nn.replay for experience replay
- compile-time lr (Adam LR baked in at compile time)

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

from nn.constants import dtype, TILE, TPB
from nn.model import Linear, LinearReLU, Sequential
from nn.optimizer import Adam
from nn.initializer import Kaiming
from nn.training import Network, NetworkState
from nn.replay import ReplayBuffer
from nn.utils import fill_inline
from core import TrainingMetrics, BoxDiscreteActionEnv, RenderableEnv


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
    lr: Float64 = 0.0005,
]:
    """Deep Dueling DQN Agent using NetworkState architecture.

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
        lr: Adam learning rate — compile-time (default: 0.0005).
    """

    # Convenience aliases
    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.num_actions
    comptime HIDDEN = Self.hidden_dim
    comptime STREAM_H = Self.stream_hidden_dim
    comptime BATCH = Self.batch_size

    # =========================================================================
    # Model type aliases
    # =========================================================================

    # Shared backbone: obs -> hidden (ReLU) -> hidden (ReLU)
    comptime BackboneModel = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
    ]
    comptime BackboneNet = Network[Self.BackboneModel, Adam[Self.lr]]

    # Value head: hidden -> stream_hidden (ReLU) -> 1
    comptime ValueModel = Sequential[
        LinearReLU[Self.HIDDEN, Self.STREAM_H],
        Linear[Self.STREAM_H, 1],
    ]
    comptime ValueNet = Network[Self.ValueModel, Adam[Self.lr]]

    # Advantage head: hidden -> stream_hidden (ReLU) -> num_actions
    comptime AdvModel = Sequential[
        LinearReLU[Self.HIDDEN, Self.STREAM_H],
        Linear[Self.STREAM_H, Self.ACTIONS],
    ]
    comptime AdvNet = Network[Self.AdvModel, Adam[Self.lr]]

    # Network states
    var backbone_online:  NetworkState[Self.BackboneModel, Adam[Self.lr]]
    var backbone_target:  NetworkState[Self.BackboneModel, Adam[Self.lr]]
    var value_online:     NetworkState[Self.ValueModel,    Adam[Self.lr]]
    var value_target:     NetworkState[Self.ValueModel,    Adam[Self.lr]]
    var adv_online:       NetworkState[Self.AdvModel,      Adam[Self.lr]]
    var adv_target:       NetworkState[Self.AdvModel,      Adam[Self.lr]]

    # Replay buffer (action_dim=1 for discrete actions stored as scalar)
    var buffer: ReplayBuffer[Self.buffer_capacity, Self.obs_dim, 1, dtype]

    # Hyperparameters
    var gamma: Float64
    var tau: Float64
    var epsilon: Float64
    var epsilon_min: Float64
    var epsilon_decay: Float64

    # Training state
    var total_steps: Int
    var train_step_count: Int

    # Auto-checkpoint settings
    var checkpoint_every: Int
    var checkpoint_path: String

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        tau: Float64 = 0.005,
        epsilon: Float64 = 1.0,
        epsilon_min: Float64 = 0.01,
        epsilon_decay: Float64 = 0.995,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
    ):
        """Initialize Deep Dueling DQN agent.

        Args:
            gamma: Discount factor (default: 0.99).
            tau: Soft update coefficient (default: 0.005).
            epsilon: Initial exploration rate (default: 1.0).
            epsilon_min: Minimum exploration rate (default: 0.01).
            epsilon_decay: Epsilon decay per episode (default: 0.995).
            checkpoint_every: Save checkpoint every N episodes (0 to disable).
            checkpoint_path: Path for auto-checkpointing.
        """
        # Initialize online networks
        self.backbone_online = NetworkState[Self.BackboneModel, Adam[Self.lr]]()
        self.backbone_online.initialize[Kaiming]()
        self.value_online = NetworkState[Self.ValueModel, Adam[Self.lr]]()
        self.value_online.initialize[Kaiming]()
        self.adv_online = NetworkState[Self.AdvModel, Adam[Self.lr]]()
        self.adv_online.initialize[Kaiming]()

        # Initialize target networks as copies of online
        self.backbone_target = NetworkState[Self.BackboneModel, Adam[Self.lr]](
            copy=self.backbone_online
        )
        self.value_target = NetworkState[Self.ValueModel, Adam[Self.lr]](
            copy=self.value_online
        )
        self.adv_target = NetworkState[Self.AdvModel, Adam[Self.lr]](
            copy=self.adv_online
        )

        self.buffer = ReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, 1, dtype
        ]()

        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.total_steps = 0
        self.train_step_count = 0
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path

    # =========================================================================
    # Dueling Forward Helpers
    # =========================================================================

    fn _dueling_forward_inline[
        BATCH_N: Int
    ](
        self,
        obs: InlineArray[Scalar[dtype], BATCH_N * Self.OBS],
        mut q_values: InlineArray[Scalar[dtype], BATCH_N * Self.ACTIONS],
        use_target: Bool = False,
    ):
        """Forward pass through dueling network: Q(s,a) = V(s) + (A(s,a) - mean(A)).

        Args:
            obs: Observations [BATCH_N * OBS].
            q_values: Output Q-values [BATCH_N * ACTIONS] (written in-place).
            use_target: If True, use target networks; else use online networks.
        """
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_N, Self.OBS), MutAnyOrigin
        ](obs.unsafe_ptr())

        # Backbone forward
        var h2_arr = InlineArray[Scalar[dtype], BATCH_N * Self.HIDDEN](
            uninitialized=True
        )
        var h2_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_N, Self.HIDDEN), MutAnyOrigin
        ](h2_arr.unsafe_ptr())

        if use_target:
            var pb = self.backbone_target.params_view()
            Self.BackboneNet.forward[BATCH_N](obs_t, h2_t, pb)
        else:
            var pb = self.backbone_online.params_view()
            Self.BackboneNet.forward[BATCH_N](obs_t, h2_t, pb)

        # Value head forward
        var v_arr = InlineArray[Scalar[dtype], BATCH_N](uninitialized=True)
        var v_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_N, 1), MutAnyOrigin
        ](v_arr.unsafe_ptr())

        if use_target:
            var pv = self.value_target.params_view()
            Self.ValueNet.forward[BATCH_N](h2_t, v_t, pv)
        else:
            var pv = self.value_online.params_view()
            Self.ValueNet.forward[BATCH_N](h2_t, v_t, pv)

        # Advantage head forward
        var adv_arr = InlineArray[Scalar[dtype], BATCH_N * Self.ACTIONS](
            uninitialized=True
        )
        var adv_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_N, Self.ACTIONS), MutAnyOrigin
        ](adv_arr.unsafe_ptr())

        if use_target:
            var pa = self.adv_target.params_view()
            Self.AdvNet.forward[BATCH_N](h2_t, adv_t, pa)
        else:
            var pa = self.adv_online.params_view()
            Self.AdvNet.forward[BATCH_N](h2_t, adv_t, pa)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A))
        for b in range(BATCH_N):
            var mean_adv: Scalar[dtype] = 0.0
            for a in range(Self.ACTIONS):
                mean_adv += adv_arr[b * Self.ACTIONS + a]
            mean_adv /= Scalar[dtype](Self.ACTIONS)

            var v_s = v_arr[b]
            for a in range(Self.ACTIONS):
                var idx = b * Self.ACTIONS + a
                q_values[idx] = v_s + (adv_arr[idx] - mean_adv)

    # =========================================================================
    # Action Selection
    # =========================================================================

    fn select_action(
        self,
        obs: InlineArray[Scalar[dtype], Self.OBS],
        greedy: Bool = False,
    ) -> Int:
        """Select action using epsilon-greedy policy.

        Args:
            obs: Current observation.
            greedy: If True, always select argmax (ignore epsilon).

        Returns:
            Selected action index.
        """
        if not greedy and random_float64() < self.epsilon:
            return Int(random_float64() * Float64(Self.ACTIONS)) % Self.ACTIONS

        var q_arr = InlineArray[Scalar[dtype], Self.ACTIONS](uninitialized=True)
        var obs_batch = InlineArray[Scalar[dtype], Self.OBS](uninitialized=True)
        for i in range(Self.OBS):
            obs_batch[i] = obs[i]
        self._dueling_forward_inline[1](obs_batch, q_arr, use_target=False)

        var best_action = 0
        var best_q = q_arr[0]
        for a in range(1, Self.ACTIONS):
            if q_arr[a] > best_q:
                best_q = q_arr[a]
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
        var action_arr = InlineArray[Scalar[dtype], 1](fill=Scalar[dtype](action))
        self.buffer.add(obs, action_arr, Scalar[dtype](reward), next_obs, done)
        self.total_steps += 1

    # =========================================================================
    # CPU Training Step
    # =========================================================================

    fn train_step(mut self) -> Float64:
        """Perform one training step.

        Returns:
            TD loss value (0 if buffer not ready).
        """
        if not self.buffer.is_ready[Self.BATCH]():
            return 0.0

        # --- Phase 1: Sample batch ---
        var batch_obs = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var batch_act1 = InlineArray[Scalar[dtype], Self.BATCH](
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

        var batch_act_flat = InlineArray[
            Scalar[dtype], Self.BATCH * 1
        ](uninitialized=True)
        self.buffer.sample[Self.BATCH](
            batch_obs, batch_act_flat, batch_rewards, batch_next_obs, batch_dones
        )
        for i in range(Self.BATCH):
            batch_act1[i] = batch_act_flat[i]

        # --- Phase 2: Compute TD targets ---
        var max_next_q = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)

        comptime if Self.double_dqn:
            var online_next_q = InlineArray[
                Scalar[dtype], Self.BATCH * Self.ACTIONS
            ](uninitialized=True)
            var target_next_q = InlineArray[
                Scalar[dtype], Self.BATCH * Self.ACTIONS
            ](uninitialized=True)
            self._dueling_forward_inline[Self.BATCH](
                batch_next_obs, online_next_q, use_target=False
            )
            self._dueling_forward_inline[Self.BATCH](
                batch_next_obs, target_next_q, use_target=True
            )
            for b in range(Self.BATCH):
                var best_action = 0
                var best_online_q = online_next_q[b * Self.ACTIONS]
                for a in range(1, Self.ACTIONS):
                    var q = online_next_q[b * Self.ACTIONS + a]
                    if q > best_online_q:
                        best_online_q = q
                        best_action = a
                max_next_q[b] = target_next_q[b * Self.ACTIONS + best_action]
        else:
            var next_q = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
                uninitialized=True
            )
            self._dueling_forward_inline[Self.BATCH](
                batch_next_obs, next_q, use_target=True
            )
            for b in range(Self.BATCH):
                var best_q = next_q[b * Self.ACTIONS]
                for a in range(1, Self.ACTIONS):
                    var q = next_q[b * Self.ACTIONS + a]
                    if q > best_q:
                        best_q = q
                max_next_q[b] = best_q

        var targets = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)
        for b in range(Self.BATCH):
            var done_mask = Scalar[dtype](1.0) - batch_dones[b]
            targets[b] = (
                batch_rewards[b]
                + Scalar[dtype](self.gamma) * max_next_q[b] * done_mask
            )

        # --- Phase 3: Forward with cache (online) ---
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](batch_obs.unsafe_ptr())

        # Backbone
        var h2_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.HIDDEN](
            uninitialized=True
        )
        var h2_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.HIDDEN), MutAnyOrigin
        ](h2_arr.unsafe_ptr())
        var backbone_cache_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.BackboneModel.CACHE_SIZE
        ](uninitialized=True)
        var backbone_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.BackboneModel.CACHE_SIZE),
            MutAnyOrigin,
        ](backbone_cache_arr.unsafe_ptr())
        var pb = self.backbone_online.params_view()
        Self.BackboneNet.forward_with_cache[Self.BATCH](obs_t, h2_t, pb, backbone_cache_t)

        # Value head
        var v_arr = InlineArray[Scalar[dtype], Self.BATCH](uninitialized=True)
        var v_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](v_arr.unsafe_ptr())
        var value_cache_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.ValueModel.CACHE_SIZE
        ](uninitialized=True)
        var value_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.ValueModel.CACHE_SIZE),
            MutAnyOrigin,
        ](value_cache_arr.unsafe_ptr())
        var pv = self.value_online.params_view()
        Self.ValueNet.forward_with_cache[Self.BATCH](h2_t, v_t, pv, value_cache_t)

        # Advantage head
        var adv_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        var adv_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](adv_arr.unsafe_ptr())
        var adv_cache_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.AdvModel.CACHE_SIZE
        ](uninitialized=True)
        var adv_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.AdvModel.CACHE_SIZE),
            MutAnyOrigin,
        ](adv_cache_arr.unsafe_ptr())
        var pa = self.adv_online.params_view()
        Self.AdvNet.forward_with_cache[Self.BATCH](h2_t, adv_t, pa, adv_cache_t)

        # Compute Q-values from V + A
        var q_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        for b in range(Self.BATCH):
            var mean_adv: Scalar[dtype] = 0.0
            for a in range(Self.ACTIONS):
                mean_adv += adv_arr[b * Self.ACTIONS + a]
            mean_adv /= Scalar[dtype](Self.ACTIONS)
            var v_s = v_arr[b]
            for a in range(Self.ACTIONS):
                var idx = b * Self.ACTIONS + a
                q_arr[idx] = v_s + (adv_arr[idx] - mean_adv)

        # --- Phase 4: Compute loss and gradients ---
        var loss: Float64 = 0.0
        var dq_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            fill=Scalar[dtype](0.0)
        )

        for b in range(Self.BATCH):
            var action_idx = Int(batch_act1[b])
            var q_idx = b * Self.ACTIONS + action_idx
            var td_error = q_arr[q_idx] - targets[b]
            loss += Float64(td_error * td_error)
            dq_arr[q_idx] = (
                Scalar[dtype](2.0) * td_error / Scalar[dtype](Self.BATCH)
            )
        loss /= Float64(Self.BATCH)

        # --- Phase 5: Backward through dueling network ---
        # dV = sum(dQ), dA_i = dQ_i - (1/n)*sum(dQ_j)
        var dv_arr = InlineArray[Scalar[dtype], Self.BATCH](
            fill=Scalar[dtype](0.0)
        )
        var da_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            fill=Scalar[dtype](0.0)
        )
        var one_over_n = Scalar[dtype](1.0) / Scalar[dtype](Self.ACTIONS)

        for b in range(Self.BATCH):
            var sum_dq: Scalar[dtype] = 0.0
            for a in range(Self.ACTIONS):
                sum_dq += dq_arr[b * Self.ACTIONS + a]
            dv_arr[b] = sum_dq
            for a in range(Self.ACTIONS):
                var idx = b * Self.ACTIONS + a
                da_arr[idx] = dq_arr[idx] - one_over_n * sum_dq

        # Backward value head: dh2_from_v
        var dv_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, 1), MutAnyOrigin
        ](dv_arr.unsafe_ptr())
        var dh2_from_v_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.HIDDEN](
            uninitialized=True
        )
        var dh2_from_v_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.HIDDEN), MutAnyOrigin
        ](dh2_from_v_arr.unsafe_ptr())
        var gv = self.value_online.grads_view()
        self.value_online.zero_grads()
        Self.ValueNet.backward[Self.BATCH](dv_t, dh2_from_v_t, pv, value_cache_t, gv)
        self.value_online.optimizer_step()

        # Backward advantage head: dh2_from_a
        var da_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](da_arr.unsafe_ptr())
        var dh2_from_a_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.HIDDEN](
            uninitialized=True
        )
        var dh2_from_a_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.HIDDEN), MutAnyOrigin
        ](dh2_from_a_arr.unsafe_ptr())
        var ga = self.adv_online.grads_view()
        self.adv_online.zero_grads()
        Self.AdvNet.backward[Self.BATCH](da_t, dh2_from_a_t, pa, adv_cache_t, ga)
        self.adv_online.optimizer_step()

        # Combine gradients from both streams for backbone backward
        var dh2_combined_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.HIDDEN](
            uninitialized=True
        )
        var dh2_combined_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.HIDDEN), MutAnyOrigin
        ](dh2_combined_arr.unsafe_ptr())
        for i in range(Self.BATCH * Self.HIDDEN):
            dh2_combined_arr[i] = dh2_from_v_arr[i] + dh2_from_a_arr[i]

        # Backward backbone
        var dobs_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var dobs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](dobs_arr.unsafe_ptr())
        var gb = self.backbone_online.grads_view()
        self.backbone_online.zero_grads()
        Self.BackboneNet.backward[Self.BATCH](dh2_combined_t, dobs_t, pb, backbone_cache_t, gb)
        self.backbone_online.optimizer_step()

        # --- Phase 6: Soft update target networks ---
        self.backbone_target.soft_update_from(self.backbone_online, self.tau)
        self.value_target.soft_update_from(self.value_online, self.tau)
        self.adv_target.soft_update_from(self.adv_online, self.tau)

        self.train_step_count += 1

        return loss

    fn decay_epsilon(mut self):
        """Decay exploration rate (call once per episode)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    fn get_epsilon(self) -> Float64:
        """Get current exploration rate."""
        return self.epsilon

    fn get_train_steps(self) -> Int:
        """Get total training steps performed."""
        return self.train_step_count

    # =========================================================================
    # High-level CPU Training and Evaluation
    # =========================================================================

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

        # Warmup
        var warmup_obs = InlineArray[Scalar[dtype], Self.OBS](uninitialized=True)
        var warmup_next = InlineArray[Scalar[dtype], Self.OBS](uninitialized=True)
        fill_inline(env.reset_obs_list(), warmup_obs)
        var warmup_count = 0
        while warmup_count < warmup_steps:
            var action = Int(random_float64() * Float64(Self.ACTIONS)) % Self.ACTIONS
            var result = env.step_obs(action)
            fill_inline(result[0], warmup_next)
            self.store_transition(warmup_obs, action, Float64(result[1]), warmup_next, result[2])
            fill_inline(result[0], warmup_obs)
            warmup_count += 1
            if result[2]:
                fill_inline(env.reset_obs_list(), warmup_obs)

        # Training loop
        var total_train_steps = 0
        var obs = InlineArray[Scalar[dtype], Self.OBS](uninitialized=True)
        var next_obs = InlineArray[Scalar[dtype], Self.OBS](uninitialized=True)

        for episode in range(num_episodes):
            fill_inline(env.reset_obs_list(), obs)
            var episode_reward: Float64 = 0.0
            var episode_steps = 0

            for _ in range(max_steps_per_episode):
                var action = self.select_action(obs)
                var result = env.step_obs(action)
                fill_inline(result[0], next_obs)
                var reward = Float64(result[1])
                var done = result[2]

                self.store_transition(obs, action, reward, next_obs, done)

                if total_train_steps % train_every == 0:
                    _ = self.train_step()

                episode_reward += reward
                fill_inline(result[0], obs)
                total_train_steps += 1
                episode_steps += 1

                if done:
                    break

            self.decay_epsilon()
            metrics.log_episode(
                episode,
                Scalar[DType.float64](episode_reward),
                episode_steps,
                self.epsilon,
            )

            if verbose and (episode + 1) % print_every == 0:
                var avg_reward = metrics.mean_reward_last_n(print_every)
                print(
                    "Episode "
                    + String(episode + 1)
                    + " | Avg reward: "
                    + String(avg_reward)[:7]
                    + " | Epsilon: "
                    + String(self.epsilon)[:5]
                    + " | Steps: "
                    + String(total_train_steps)
                )

            if self.checkpoint_every > 0 and len(self.checkpoint_path) > 0:
                if (episode + 1) % self.checkpoint_every == 0:
                    try:
                        self.save_checkpoint(self.checkpoint_path)
                    except:
                        pass

        return metrics^

    fn evaluate[
        E: BoxDiscreteActionEnv & RenderableEnv
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps: Int = 1000,
        verbose: Bool = False,
        render: Bool = False,
        frame_delay_ms: Int = 16,
        greedy: Bool = True,
    ) raises -> Float64:
        """Evaluate the agent using greedy policy.

        Args:
            env: The environment to evaluate on.
            num_episodes: Number of evaluation episodes.
            max_steps: Maximum steps per episode.
            verbose: Whether to print per-episode results.
            render: Whether to render the environment.
            frame_delay_ms: Delay between frames in milliseconds.
            greedy: Use pure greedy policy (default: True).

        Returns:
            Average reward over evaluation episodes.
        """
        var total_reward: Float64 = 0.0
        var quit_requested = False

        if render:
            _ = env.init_renderer()

        var obs = InlineArray[Scalar[dtype], Self.OBS](uninitialized=True)

        for episode in range(num_episodes):
            if quit_requested:
                break

            fill_inline(env.reset_obs_list(), obs)
            var episode_reward: Float64 = 0.0
            var episode_steps = 0

            for _ in range(max_steps):
                var action = self.select_action(obs, greedy=greedy)
                var result = env.step_obs(action)
                fill_inline(result[0], obs)
                episode_reward += Float64(result[1])
                episode_steps += 1

                if render:
                    env.render_frame()
                    env.renderer_delay(frame_delay_ms)
                    if env.check_renderer_quit():
                        quit_requested = True
                        break

                if result[2]:
                    break

            total_reward += episode_reward

            if verbose:
                print(
                    "Eval Episode "
                    + String(episode + 1)
                    + " | Reward: "
                    + String(episode_reward)[:10]
                    + " | Steps: "
                    + String(episode_steps)
                )

        if render:
            env.close_renderer()

        return total_reward / Float64(num_episodes)

    # =========================================================================
    # Checkpoint Save / Load
    # =========================================================================

    fn save_checkpoint(self, filepath: String) raises:
        """Save agent state to a checkpoint file.

        Args:
            filepath: Destination path for the checkpoint file.
        """
        from nn.checkpoint import (
            write_checkpoint_header,
            write_metadata_section,
            save_checkpoint_file,
        )
        var param_size = (
            Self.BackboneModel.PARAM_SIZE
            + Self.ValueModel.PARAM_SIZE
            + Self.AdvModel.PARAM_SIZE
        )
        var content = write_checkpoint_header("dueling_dqn", param_size, param_size)
        content += self.backbone_online.write_sections("backbone_online_")
        content += self.backbone_target.write_sections("backbone_target_")
        content += self.value_online.write_sections("value_online_")
        content += self.value_target.write_sections("value_target_")
        content += self.adv_online.write_sections("adv_online_")
        content += self.adv_target.write_sections("adv_target_")
        var metadata = List[String]()
        metadata.append("epsilon=" + String(self.epsilon))
        metadata.append("total_steps=" + String(self.total_steps))
        content += write_metadata_section(metadata)
        save_checkpoint_file(filepath, content)

    fn load_checkpoint(mut self, filepath: String) raises:
        """Load agent state from a checkpoint file.

        Args:
            filepath: Path to the checkpoint file.
        """
        from nn.checkpoint import (
            read_checkpoint_file,
            read_metadata_section,
            get_metadata_value,
        )
        var content = read_checkpoint_file(filepath)
        self.backbone_online.read_sections(content, "backbone_online_")
        self.backbone_target.read_sections(content, "backbone_target_")
        self.value_online.read_sections(content, "value_online_")
        self.value_target.read_sections(content, "value_target_")
        self.adv_online.read_sections(content, "adv_online_")
        self.adv_target.read_sections(content, "adv_target_")
        var metadata = read_metadata_section(content)
        var eps_str = get_metadata_value(metadata, "epsilon")
        if len(eps_str) > 0:
            self.epsilon = Float64(atof(eps_str))

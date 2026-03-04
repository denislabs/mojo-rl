"""DQN Agent with Prioritized Experience Replay using the new trait-based architecture.

This DQN+PER implementation uses:
- NetworkState for heap-allocated params + grads + optimizer state
- Network (all-static) for stateless forward/backward ops via LayoutTensor
- PrioritizedReplayBuffer from nn.replay for priority-weighted sampling
- compile-time lr (Adam LR baked in at compile time)

Key differences from standard DQN:
- Samples transitions proportionally to TD error magnitude
- Uses importance sampling weights to correct for non-uniform sampling bias
- Updates priorities after each training step based on new TD errors

Features:
- Works with any BoxDiscreteActionEnv (continuous obs, discrete actions)
- Epsilon-greedy exploration with decay
- Target network with soft updates
- Double DQN support via compile-time parameter
- Beta annealing for importance sampling correction

No GPU path: PrioritizedReplayBuffer uses a sum-tree that requires
serial CPU access for priority updates, making GPU training impractical.

Usage:
    from deep_agents.dqn_per import DQNPERAgent
    from envs import LunarLanderEnv

    var env = LunarLanderEnv()
    var agent = DQNPERAgent[8, 4, 128, 100000, 64]()

    var metrics = agent.train(env, num_episodes=500)

Reference: Schaul et al., "Prioritized Experience Replay" (2015)
"""

from std.math import exp
from std.random import random_float64, seed

from layout import Layout, LayoutTensor

from nn.constants import dtype, TILE, TPB
from nn.model import Linear, LinearReLU, Sequential
from nn.optimizer import Adam
from nn.initializer import Kaiming
from nn.training import Network, NetworkState
from nn.replay import PrioritizedReplayBuffer
from deep_agents.core import (
    fill_inline,
    obs_to_inline,
    OffPolicyAgent,
    run_offpolicy_discrete_train,
)
from core import TrainingMetrics, BoxDiscreteActionEnv, RenderableEnv


# =============================================================================
# DQN + PER Agent
# =============================================================================


struct DQNPERAgent[
    obs_dim: Int,
    num_actions: Int,
    hidden_dim: Int = 128,
    buffer_capacity: Int = 20000,
    batch_size: Int = 64,
    double_dqn: Bool = True,
    lr: Float64 = 0.0005,
](OffPolicyAgent):
    """DQN Agent with Prioritized Experience Replay using NetworkState architecture.

    PER samples transitions proportionally to their TD error magnitude, which
    helps the agent learn more efficiently from important experiences.

    Key features:
    - Priority-weighted sampling based on TD errors
    - Importance sampling weights correct for non-uniform sampling bias
    - Beta annealing from beta_start to 1.0 over training
    - Double DQN support (compile-time flag)
    - lr is compile-time (baked into Adam optimizer type)

    Parameters:
        obs_dim: Dimension of observation space.
        num_actions: Number of discrete actions.
        hidden_dim: Hidden layer size (default: 128).
        buffer_capacity: Replay buffer capacity (default: 20000).
        batch_size: Training batch size (default: 64).
        double_dqn: If True, use Double DQN (default: True).
        lr: Adam learning rate — compile-time (default: 0.0005).
    """

    # Convenience aliases
    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.num_actions
    comptime HIDDEN = Self.hidden_dim
    comptime BATCH = Self.batch_size

    # Q-network: obs -> hidden (ReLU) -> hidden (ReLU) -> num_actions
    comptime Q_Model = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, Self.ACTIONS],
    ]
    comptime Q_Network = Network[Self.Q_Model, Adam[Self.lr]]

    # Network states (heap-allocated params + grads + optimizer state)
    var q_network: NetworkState[Self.Q_Model, Adam[Self.lr]]
    var target_network: NetworkState[Self.Q_Model, Adam[Self.lr]]

    # Prioritized Replay Buffer (action_dim=1 for discrete actions stored as scalar)
    var buffer: PrioritizedReplayBuffer[
        Self.buffer_capacity, Self.obs_dim, 1, dtype
    ]

    # Standard DQN hyperparameters
    var gamma: Float64
    var tau: Float64
    var epsilon: Float64
    var epsilon_min: Float64
    var epsilon_decay: Float64

    # PER hyperparameters
    var beta: Float64  # Current IS exponent (annealed to 1.0)
    var beta_start: Float64  # Initial beta value
    var beta_frames: Int  # Steps to anneal beta over

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
        alpha: Float64 = 0.6,
        beta_start: Float64 = 0.4,
        beta_frames: Int = 100000,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
    ):
        """Initialize DQN+PER agent.

        Args:
            gamma: Discount factor (default: 0.99).
            tau: Soft update coefficient (default: 0.005).
            epsilon: Initial exploration rate (default: 1.0).
            epsilon_min: Minimum exploration rate (default: 0.01).
            epsilon_decay: Epsilon decay per episode (default: 0.995).
            alpha: Priority exponent (0=uniform, 1=full prioritization) (default: 0.6).
            beta_start: Initial IS correction exponent (default: 0.4).
            beta_frames: Steps to anneal beta from beta_start to 1.0 (default: 100000).
            checkpoint_every: Save checkpoint every N episodes (0 to disable).
            checkpoint_path: Path for auto-checkpointing.
        """
        self.q_network = NetworkState[Self.Q_Model, Adam[Self.lr]]()
        self.q_network.initialize[Kaiming]()
        self.target_network = NetworkState[Self.Q_Model, Adam[Self.lr]](
            copy=self.q_network
        )

        self.buffer = PrioritizedReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, 1, dtype
        ](alpha=Scalar[dtype](alpha), beta=Scalar[dtype](beta_start))

        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames

        self.total_steps = 0
        self.train_step_count = 0
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path

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

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs.unsafe_ptr())
        var q_arr = InlineArray[Scalar[dtype], Self.ACTIONS](uninitialized=True)
        var q_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](q_arr.unsafe_ptr())
        var p = self.q_network.params_view()
        Self.Q_Network.forward[1](obs_t, q_t, p)

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
        """Store transition in prioritized replay buffer with max priority."""
        var action_arr = InlineArray[Scalar[dtype], 1](
            fill=Scalar[dtype](action)
        )
        self.buffer.add(obs, action_arr, Scalar[dtype](reward), next_obs, done)
        self.total_steps += 1

        # Anneal beta towards 1.0
        var progress = Float64(self.total_steps) / Float64(self.beta_frames)
        if progress > 1.0:
            progress = 1.0
        self.beta = self.beta_start + progress * (1.0 - self.beta_start)
        self.buffer.set_beta(Scalar[dtype](self.beta))

    # =========================================================================
    # CPU Training Step
    # =========================================================================

    fn train_step(mut self) -> Float64:
        """Perform one training step with PER.

        Returns:
            TD loss value (0 if buffer not ready).
        """
        if not self.buffer.is_ready[Self.BATCH]():
            return 0.0

        # --- Phase 1: Sample batch with importance sampling weights ---
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
        var batch_weights = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )
        var batch_indices = InlineArray[Int, Self.BATCH](uninitialized=True)

        self.buffer.sample[Self.BATCH](
            batch_obs,
            batch_act1,
            batch_rewards,
            batch_next_obs,
            batch_dones,
            batch_weights,
            batch_indices,
        )

        # LayoutTensor views over sampled data
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](batch_obs.unsafe_ptr())
        var next_obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](batch_next_obs.unsafe_ptr())

        # --- Phase 2: Compute TD targets ---
        var p_target = self.target_network.params_view()
        var next_q_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        var next_q_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](next_q_arr.unsafe_ptr())
        Self.Q_Network.forward[Self.BATCH](next_obs_t, next_q_t, p_target)

        var p_online = self.q_network.params_view()
        var max_next_q = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )

        comptime if Self.double_dqn:
            var online_next_arr = InlineArray[
                Scalar[dtype], Self.BATCH * Self.ACTIONS
            ](uninitialized=True)
            var online_next_t = LayoutTensor[
                dtype,
                Layout.row_major(Self.BATCH, Self.ACTIONS),
                MutAnyOrigin,
            ](online_next_arr.unsafe_ptr())
            Self.Q_Network.forward[Self.BATCH](
                next_obs_t, online_next_t, p_online
            )
            for b in range(Self.BATCH):
                var best_action = 0
                var best_online_q = online_next_arr[b * Self.ACTIONS]
                for a in range(1, Self.ACTIONS):
                    var q = online_next_arr[b * Self.ACTIONS + a]
                    if q > best_online_q:
                        best_online_q = q
                        best_action = a
                max_next_q[b] = next_q_arr[b * Self.ACTIONS + best_action]
        else:
            for b in range(Self.BATCH):
                var best_q = next_q_arr[b * Self.ACTIONS]
                for a in range(1, Self.ACTIONS):
                    var q = next_q_arr[b * Self.ACTIONS + a]
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

        # --- Phase 3: Forward with cache ---
        var q_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            uninitialized=True
        )
        var q_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](q_arr.unsafe_ptr())
        var cache_arr = InlineArray[
            Scalar[dtype], Self.BATCH * Self.Q_Model.CACHE_SIZE
        ](uninitialized=True)
        var cache_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.BATCH, Self.Q_Model.CACHE_SIZE),
            MutAnyOrigin,
        ](cache_arr.unsafe_ptr())
        Self.Q_Network.forward_with_cache[Self.BATCH](
            obs_t, q_t, p_online, cache_t
        )

        # --- Phase 4: Compute weighted loss and gradients ---
        var loss: Float64 = 0.0
        var dq_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.ACTIONS](
            fill=Scalar[dtype](0.0)
        )
        var td_errors = InlineArray[Scalar[dtype], Self.BATCH](
            uninitialized=True
        )

        for b in range(Self.BATCH):
            var action = Int(batch_act1[b])
            var q_idx = b * Self.ACTIONS + action
            var td_error = q_arr[q_idx] - targets[b]
            td_errors[b] = td_error

            var weight = batch_weights[b]
            var weighted_error = weight * td_error
            loss += Float64(weighted_error * weighted_error)

            dq_arr[q_idx] = (
                Scalar[dtype](2.0) * weighted_error / Scalar[dtype](Self.BATCH)
            )
        loss /= Float64(Self.BATCH)

        # --- Phase 5: Backward pass and optimizer step ---
        var dq_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.ACTIONS), MutAnyOrigin
        ](dq_arr.unsafe_ptr())
        var grad_in_arr = InlineArray[Scalar[dtype], Self.BATCH * Self.OBS](
            uninitialized=True
        )
        var grad_in_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin
        ](grad_in_arr.unsafe_ptr())
        var g = self.q_network.grads_view()

        self.q_network.zero_grads()
        Self.Q_Network.backward[Self.BATCH](
            dq_t, grad_in_t, p_online, cache_t, g
        )
        self.q_network.optimizer_step()

        # --- Phase 6: Update priorities ---
        self.buffer.update_priorities[Self.BATCH](batch_indices, td_errors)

        # --- Phase 7: Soft update target ---
        self.target_network.soft_update_from(self.q_network, self.tau)
        self.train_step_count += 1

        return loss

    fn decay_epsilon(mut self):
        """Decay exploration rate (call once per episode)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    fn get_epsilon(self) -> Float64:
        """Get current exploration rate."""
        return self.epsilon

    fn get_beta(self) -> Float64:
        """Get current IS correction exponent."""
        return self.beta

    fn get_train_steps(self) -> Int:
        """Get total training steps performed."""
        return self.train_step_count

    # =========================================================================
    # OffPolicyAgent trait conformance
    # =========================================================================

    fn select_action_list[
        dt: DType
    ](mut self, obs: List[Scalar[dt]]) -> List[Scalar[dt]]:
        var obs_inline = obs_to_inline[Self.OBS, dt](obs)
        var action = self.select_action(obs_inline)
        var result = List[Scalar[dt]]()
        result.append(Scalar[dt](action))
        return result^

    fn store_list_transition[
        dt: DType
    ](
        mut self,
        obs: List[Scalar[dt]],
        action: List[Scalar[dt]],
        reward: Float64,
        next_obs: List[Scalar[dt]],
        done: Bool,
    ) -> None:
        var obs_inline = obs_to_inline[Self.OBS, dt](obs)
        var next_inline = obs_to_inline[Self.OBS, dt](next_obs)
        var action_int = Int(Float64(action[0]))
        self.store_transition(obs_inline, action_int, reward, next_inline, done)

    fn is_ready(self) -> Bool:
        return self.buffer.is_ready[Self.BATCH]()

    fn do_train_step(mut self) -> Float64:
        return self.train_step()

    fn decay_explore(mut self) -> None:
        self.decay_epsilon()

    fn get_explore_rate(self) -> Float64:
        return self.epsilon

    fn random_action_list[dt: DType](self) -> List[Scalar[dt]]:
        var result = List[Scalar[dt]]()
        result.append(
            Scalar[dt](
                Int(random_float64() * Float64(Self.ACTIONS)) % Self.ACTIONS
            )
        )
        return result^

    fn select_greedy_action_list(self, obs: List[Float64]) -> List[Float64]:
        var obs_inline = obs_to_inline[Self.OBS, DType.float64](obs)
        var action = self.select_action(obs_inline, greedy=True)
        var result = List[Float64]()
        result.append(Float64(action))
        return result^

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
        """Train the DQN+PER agent on a discrete action environment.

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
        return run_offpolicy_discrete_train(
            self,
            env,
            num_episodes,
            max_steps_per_episode,
            warmup_steps,
            train_every,
            verbose,
            print_every,
            environment_name,
            "DQN + PER",
        )

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

        var content = write_checkpoint_header(
            "dqn_per",
            Self.Q_Model.PARAM_SIZE,
            Self.Q_Model.PARAM_SIZE,
        )
        content += self.q_network.write_sections("online_")
        content += self.target_network.write_sections("target_")
        var metadata = List[String]()
        metadata.append("epsilon=" + String(self.epsilon))
        metadata.append("beta=" + String(self.beta))
        metadata.append("total_steps=" + String(self.total_steps))
        metadata.append("train_step_count=" + String(self.train_step_count))
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
        self.q_network.read_sections(content, "online_")
        self.target_network.read_sections(content, "target_")
        var metadata = read_metadata_section(content)
        var eps_str = get_metadata_value(metadata, "epsilon")
        if len(eps_str) > 0:
            self.epsilon = Float64(atof(eps_str))
        var beta_str = get_metadata_value(metadata, "beta")
        if len(beta_str) > 0:
            self.beta = Float64(atof(beta_str))

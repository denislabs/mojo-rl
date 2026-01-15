"""DQN Agent using the new trait-based deep learning architecture.

This DQN implementation uses:
- Network wrapper from deep_rl.training for stateless model + params management
- seq() composition for building Q-networks
- ReplayBuffer from deep_rl.replay for experience replay
- Double DQN support via compile-time parameter

Features:
- Works with any BoxDiscreteActionEnv (continuous obs, discrete actions)
- Epsilon-greedy exploration with decay
- Target network with soft updates
- Double DQN to reduce overestimation bias (optional)
- GPU support for batch training (forward/backward/update on GPU)

Usage:
    from deep_agents.dqn import DQNAgent
    from envs import CartPoleEnv

    var env = CartPoleEnv()
    var agent = DQNAgent[4, 2, 64, 10000, 32]()

    # CPU Training
    var metrics = agent.train(env, num_episodes=200)

    # GPU Training
    var ctx = DeviceContext()
    agent.init_gpu(ctx)
    var metrics_gpu = agent.train_gpu(ctx, env, num_episodes=200)
"""

from math import exp
from random import random_float64, seed

from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor

from deep_rl.constants import dtype, TILE, TPB
from deep_rl.model import Linear, ReLU, seq
from deep_rl.optimizer import Adam
from deep_rl.initializer import Kaiming
from deep_rl.training import Network
from deep_rl.replay import ReplayBuffer
from deep_rl.checkpoint import (
    write_checkpoint_header,
    write_float_section,
    write_metadata_section,
    parse_checkpoint_header,
    read_checkpoint_file,
    read_float_section,
    read_metadata_section,
    get_metadata_value,
    save_checkpoint_file,
    split_lines,
    find_section_start,
)
from deep_rl.gpu import (
    random_range,
    xorshift32,
    random_uniform,
    soft_update_kernel,
    zero_buffer_kernel,
    copy_buffer_kernel,
    accumulate_rewards_kernel,
    increment_steps_kernel,
    extract_completed_episodes_kernel,
    selective_reset_tracking_kernel,
    store_transitions_kernel,
    sample_indices_kernel,
    gather_batch_kernel,
)
from core import TrainingMetrics, BoxDiscreteActionEnv, GPUDiscreteEnv


# =============================================================================
# GPU Kernels for DQN Operations
# =============================================================================


@always_inline
fn dqn_td_target_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    NUM_ACTIONS: Int,
](
    # Outputs
    targets: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    # Inputs
    next_q_values: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
    rewards: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    dones: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    gamma: Scalar[dtype],
):
    """Compute TD targets for standard DQN: target = r + gamma * max_a Q(s', a) * (1 - done).
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH_SIZE:
        return

    # Find max Q-value for next state
    var max_q = next_q_values[b, 0]
    for a in range(1, NUM_ACTIONS):
        var q = next_q_values[b, a]
        if q > max_q:
            max_q = q

    # Compute TD target
    var done_mask = Scalar[dtype](1.0) - dones[b]
    targets[b] = rewards[b] + gamma * max_q * done_mask


@always_inline
fn dqn_double_td_target_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    NUM_ACTIONS: Int,
](
    # Outputs
    targets: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    # Inputs
    online_next_q: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
    target_next_q: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
    rewards: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    dones: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    gamma: Scalar[dtype],
):
    """Compute TD targets for Double DQN: target = r + gamma * Q_target(s', argmax_a Q_online(s', a)) * (1 - done).
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH_SIZE:
        return

    # Online network selects best action (argmax)
    var best_action = 0
    var best_q = online_next_q[b, 0]
    for a in range(1, NUM_ACTIONS):
        var q = online_next_q[b, a]
        if q > best_q:
            best_q = q
            best_action = a

    # Target network evaluates that action
    var target_q = target_next_q[b, best_action]

    # Compute TD target
    var done_mask = Scalar[dtype](1.0) - dones[b]
    targets[b] = rewards[b] + gamma * target_q * done_mask


@always_inline
fn dqn_grad_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    NUM_ACTIONS: Int,
](
    # Outputs
    grad_output: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
    loss_out: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
    # Inputs
    q_values: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
    targets: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    actions: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
):
    """Compute masked gradient for DQN loss. Only backprop through taken action.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH_SIZE:
        return

    var action = Int(actions[b])
    var q_pred = q_values[b, action]
    var td_error = q_pred - targets[b]

    # Masked gradient: only for taken action
    for a in range(NUM_ACTIONS):
        if a == action:
            grad_output[b, a] = (
                Scalar[dtype](2.0) * td_error / Scalar[dtype](BATCH_SIZE)
            )
        else:
            grad_output[b, a] = Scalar[dtype](0.0)


# =============================================================================
# DQN Agent
# =============================================================================


struct DQNAgent[
    obs_dim: Int,
    num_actions: Int,
    hidden_dim: Int = 64,
    buffer_capacity: Int = 10000,
    batch_size: Int = 256,
    n_envs: Int = 1024,
    double_dqn: Bool = True,
]:
    """Deep Q-Network agent using the new trait-based architecture.

    Parameters:
        obs_dim: Dimension of observation space.
        num_actions: Number of discrete actions.
        hidden_dim: Hidden layer size (default: 64).
        buffer_capacity: Replay buffer capacity (default: 10000).
        batch_size: Training batch size for gradient updates (default: 256).
        n_envs: Number of parallel environments for GPU training (default: 1024).
        double_dqn: Use Double DQN (default: True).

    Note on batch_size vs n_envs (for GPU training):
        - n_envs: How many environments run in parallel (affects data collection rate)
        - batch_size: How many samples used per gradient update (affects learning stability)
        These are decoupled to allow independent tuning.
    """

    # Q-network architecture: obs -> hidden (ReLU) -> hidden (ReLU) -> num_actions
    comptime HIDDEN = Self.hidden_dim
    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.num_actions

    # Compute network dimensions at compile time
    # Cache sizes for each layer:
    # - Linear[obs, hidden]: caches obs_dim
    # - ReLU[hidden]: caches hidden_dim
    # - Linear[hidden, hidden]: caches hidden_dim
    # - ReLU[hidden]: caches hidden_dim
    # - Linear[hidden, actions]: caches hidden_dim
    comptime NETWORK_CACHE_SIZE: Int = (
        Self.OBS + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN
    )

    # Param sizes for each layer:
    # - Linear[obs, hidden]: obs * hidden + hidden (W + b)
    # - ReLU[hidden]: 0
    # - Linear[hidden, hidden]: hidden * hidden + hidden
    # - ReLU[hidden]: 0
    # - Linear[hidden, actions]: hidden * actions + actions
    comptime NETWORK_PARAM_SIZE: Int = (
        Self.OBS * Self.HIDDEN
        + Self.HIDDEN
        + Self.HIDDEN * Self.HIDDEN
        + Self.HIDDEN
        + Self.HIDDEN * Self.ACTIONS
        + Self.ACTIONS
    )

    # Online and target networks
    var online_model: Network[
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
        Kaiming,
    ]
    var target_model: Network[
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
        Kaiming,
    ]

    # Replay buffer (action_dim=1 since we store discrete action as scalar)
    var buffer: ReplayBuffer[Self.buffer_capacity, Self.obs_dim, 1, dtype]

    # Hyperparameters
    var gamma: Float64  # Discount factor
    var tau: Float64  # Soft update rate
    var lr: Float64  # Learning rate (stored for reference)

    # Exploration
    var epsilon: Float64
    var epsilon_min: Float64
    var epsilon_decay: Float64

    # Training state
    var train_step_count: Int

    # Auto-checkpoint settings
    var checkpoint_every: Int  # Save checkpoint every N episodes (0 to disable)
    var checkpoint_path: String  # Path for auto-checkpointing

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        tau: Float64 = 0.005,
        lr: Float64 = 0.001,
        epsilon: Float64 = 1.0,
        epsilon_min: Float64 = 0.01,
        epsilon_decay: Float64 = 0.995,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
    ):
        """Initialize DQN agent.

        Args:
            gamma: Discount factor (default: 0.99).
            tau: Soft update rate for target network (default: 0.005).
            lr: Learning rate for Adam optimizer (default: 0.001).
            epsilon: Initial exploration rate (default: 1.0).
            epsilon_min: Minimum exploration rate (default: 0.01).
            epsilon_decay: Epsilon decay per episode (default: 0.995).
            checkpoint_every: Save checkpoint every N episodes (0 to disable).
            checkpoint_path: Path to save checkpoints (required if checkpoint_every > 0).
        """
        # Create Q-network model
        var q_model = seq(
            Linear[Self.OBS, Self.HIDDEN](),
            ReLU[Self.HIDDEN](),
            Linear[Self.HIDDEN, Self.HIDDEN](),
            ReLU[Self.HIDDEN](),
            Linear[Self.HIDDEN, Self.ACTIONS](),
        )

        # Initialize networks
        self.online_model = Network(q_model, Adam(lr=lr), Kaiming())
        self.target_model = Network(q_model, Adam(lr=lr), Kaiming())

        # Initialize target with online's weights
        self.target_model.copy_params_from(self.online_model)

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
        self.train_step_count = 0

        # Auto-checkpoint settings
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path

    fn select_action(self, obs: SIMD[DType.float64, Self.obs_dim]) -> Int:
        """Select action using epsilon-greedy policy.

        Args:
            obs: Current observation.

        Returns:
            Selected action index.
        """
        # Epsilon-greedy exploration
        if random_float64() < self.epsilon:
            return Int(random_float64() * Float64(Self.num_actions))

        # Greedy action: argmax_a Q(s, a)
        var obs_input = InlineArray[Scalar[dtype], Self.obs_dim](
            uninitialized=True
        )
        for i in range(Self.obs_dim):
            obs_input[i] = Scalar[dtype](obs[i])

        var q_values = InlineArray[Scalar[dtype], Self.num_actions](
            uninitialized=True
        )
        self.online_model.forward[1](obs_input, q_values)

        # Find argmax
        var best_action = 0
        var best_q = q_values[0]
        for a in range(1, Self.num_actions):
            if q_values[a] > best_q:
                best_q = q_values[a]
                best_action = a

        return best_action

    fn store_transition(
        mut self,
        obs: SIMD[DType.float64, Self.obs_dim],
        action: Int,
        reward: Float64,
        next_obs: SIMD[DType.float64, Self.obs_dim],
        done: Bool,
    ):
        """Store transition in replay buffer.

        Args:
            obs: Current observation.
            action: Action taken.
            reward: Reward received.
            next_obs: Next observation.
            done: Whether episode ended.
        """
        var obs_arr = InlineArray[Scalar[dtype], Self.obs_dim](
            uninitialized=True
        )
        var next_obs_arr = InlineArray[Scalar[dtype], Self.obs_dim](
            uninitialized=True
        )
        for i in range(Self.obs_dim):
            obs_arr[i] = Scalar[dtype](obs[i])
            next_obs_arr[i] = Scalar[dtype](next_obs[i])

        var action_arr = InlineArray[Scalar[dtype], 1](Scalar[dtype](action))

        self.buffer.add(
            obs_arr, action_arr, Scalar[dtype](reward), next_obs_arr, done
        )

    fn train_step(mut self) -> Float64:
        """Perform one training step.

        Returns:
            Loss value (0 if buffer not ready).
        """
        # Check if buffer has enough samples
        if not self.buffer.is_ready[Self.batch_size]():
            return 0.0

        # Sample batch from buffer
        var batch_obs = InlineArray[
            Scalar[dtype], Self.batch_size * Self.obs_dim
        ](uninitialized=True)
        var batch_actions = InlineArray[Scalar[dtype], Self.batch_size](
            uninitialized=True
        )
        var batch_rewards = InlineArray[Scalar[dtype], Self.batch_size](
            uninitialized=True
        )
        var batch_next_obs = InlineArray[
            Scalar[dtype], Self.batch_size * Self.obs_dim
        ](uninitialized=True)
        var batch_dones = InlineArray[Scalar[dtype], Self.batch_size](
            uninitialized=True
        )

        # Temporary for actions (action_dim=1)
        var batch_actions_tmp = InlineArray[Scalar[dtype], Self.batch_size * 1](
            uninitialized=True
        )

        self.buffer.sample[Self.batch_size](
            batch_obs,
            batch_actions_tmp,
            batch_rewards,
            batch_next_obs,
            batch_dones,
        )

        # Copy actions (action_dim=1 so just copy directly)
        for i in range(Self.batch_size):
            batch_actions[i] = batch_actions_tmp[i]

        # Forward pass on online network (with cache for backward)
        var q_values = InlineArray[
            Scalar[dtype], Self.batch_size * Self.num_actions
        ](uninitialized=True)
        var cache = InlineArray[
            Scalar[dtype], Self.batch_size * Self.NETWORK_CACHE_SIZE
        ](uninitialized=True)
        self.online_model.forward_with_cache[Self.batch_size](
            batch_obs, q_values, cache
        )

        # Forward pass on target network (no cache needed)
        var next_q_values = InlineArray[
            Scalar[dtype], Self.batch_size * Self.num_actions
        ](uninitialized=True)
        self.target_model.forward[Self.batch_size](
            batch_next_obs, next_q_values
        )

        # Compute TD targets
        var targets = InlineArray[Scalar[dtype], Self.batch_size](
            uninitialized=True
        )

        @parameter
        if Self.double_dqn:
            # Double DQN: online network selects action, target evaluates
            var online_next_q = InlineArray[
                Scalar[dtype], Self.batch_size * Self.num_actions
            ](uninitialized=True)
            self.online_model.forward[Self.batch_size](
                batch_next_obs, online_next_q
            )

            for b in range(Self.batch_size):
                # Online network selects best action
                var best_action = 0
                var best_q = online_next_q[b * Self.num_actions]
                for a in range(1, Self.num_actions):
                    var q = online_next_q[b * Self.num_actions + a]
                    if q > best_q:
                        best_q = q
                        best_action = a

                # Target network evaluates that action
                var next_q = next_q_values[b * Self.num_actions + best_action]

                # TD target: r + γ * Q_target(s', argmax_a Q_online(s', a)) * (1 - done)
                var done_mask = Scalar[dtype](1.0) - batch_dones[b]
                targets[b] = (
                    batch_rewards[b]
                    + Scalar[dtype](self.gamma) * next_q * done_mask
                )
        else:
            # Standard DQN: max_a Q_target(s', a)
            for b in range(Self.batch_size):
                var max_next_q = next_q_values[b * Self.num_actions]
                for a in range(1, Self.num_actions):
                    var q = next_q_values[b * Self.num_actions + a]
                    if q > max_next_q:
                        max_next_q = q

                # TD target: r + γ * max_a Q_target(s', a) * (1 - done)
                var done_mask = Scalar[dtype](1.0) - batch_dones[b]
                targets[b] = (
                    batch_rewards[b]
                    + Scalar[dtype](self.gamma) * max_next_q * done_mask
                )

        # Compute loss gradient: d(Q(s,a) - target)^2 / dQ = 2 * (Q(s,a) - target)
        # We only backprop through the action that was taken
        var grad_output = InlineArray[
            Scalar[dtype], Self.batch_size * Self.num_actions
        ](uninitialized=True)

        var total_loss: Float64 = 0.0

        for b in range(Self.batch_size):
            var action = Int(batch_actions[b])
            var q_pred = q_values[b * Self.num_actions + action]
            var td_error = q_pred - targets[b]

            # MSE loss
            total_loss += Float64(td_error * td_error)

            # Gradient: only for the taken action
            for a in range(Self.num_actions):
                if a == action:
                    # d/dQ (Q - target)^2 = 2 * (Q - target) / batch_size
                    grad_output[b * Self.num_actions + a] = (
                        Scalar[dtype](2.0)
                        * td_error
                        / Scalar[dtype](Self.batch_size)
                    )
                else:
                    grad_output[b * Self.num_actions + a] = Scalar[dtype](0.0)

        total_loss /= Float64(Self.batch_size)

        # Backward pass
        var grad_input = InlineArray[
            Scalar[dtype], Self.batch_size * Self.obs_dim
        ](uninitialized=True)

        self.online_model.zero_grads()
        self.online_model.backward[Self.batch_size](
            grad_output, grad_input, cache
        )

        # Update online network
        self.online_model.update()

        # Soft update target network
        self.target_model.soft_update_from(self.online_model, self.tau)

        self.train_step_count += 1

        return total_loss

    fn decay_epsilon(mut self):
        """Decay exploration rate (call at end of each episode)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    fn get_epsilon(self) -> Float64:
        """Get current exploration rate."""
        return self.epsilon

    fn get_train_steps(self) -> Int:
        """Get total training steps performed."""
        return self.train_step_count

    # =========================================================================
    # High-level training and evaluation methods
    # =========================================================================

    fn _list_to_simd(
        self, obs_list: List[Float64]
    ) -> SIMD[DType.float64, Self.obs_dim]:
        """Convert List[Float64] to SIMD for internal use."""
        var obs = SIMD[DType.float64, Self.obs_dim]()
        for i in range(Self.obs_dim):
            obs[i] = obs_list[i]
        return obs

    fn train[
        E: BoxDiscreteActionEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int,
        max_steps_per_episode: Int = 500,
        warmup_steps: Int = 1000,
        train_every: Int = 4,
        verbose: Bool = False,
        print_every: Int = 10,
        environment_name: String = "Environment",
    ) -> TrainingMetrics:
        """Train the DQN agent on a continuous-state environment.

        Args:
            env: The environment to train on (must implement BoxDiscreteActionEnv).
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode (default: 500).
            warmup_steps: Number of random steps to fill replay buffer (default: 1000).
            train_every: Train every N steps (default: 4).
            verbose: Whether to print progress (default: False).
            print_every: Print progress every N episodes if verbose (default: 10).
            environment_name: Name of environment for metrics labeling.

        Returns:
            TrainingMetrics object with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="DQN" if not Self.double_dqn else "Double DQN",
            environment_name=environment_name,
        )

        # =====================================================================
        # Warmup: fill replay buffer with random actions
        # =====================================================================
        var warmup_obs = self._list_to_simd(env.reset_obs_list())
        var warmup_count = 0

        while warmup_count < warmup_steps:
            # Random action
            var action = Int(random_float64() * Float64(Self.num_actions))
            var result = env.step_obs(action)

            var next_obs = self._list_to_simd(result[0])
            self.store_transition(
                warmup_obs, action, result[1], next_obs, result[2]
            )

            warmup_obs = next_obs
            warmup_count += 1

            if result[2]:  # done
                warmup_obs = self._list_to_simd(env.reset_obs_list())

        # =====================================================================
        # Training loop
        # =====================================================================
        var total_steps = 0

        for episode in range(num_episodes):
            var obs = self._list_to_simd(env.reset_obs_list())
            var episode_reward: Float64 = 0.0
            var episode_steps = 0

            for step in range(max_steps_per_episode):
                # Select action using epsilon-greedy
                var action = self.select_action(obs)

                # Step environment
                var result = env.step_obs(action)
                var next_obs = self._list_to_simd(result[0])
                var reward = result[1]
                var done = result[2]

                # Store transition
                self.store_transition(obs, action, reward, next_obs, done)

                # Train every N steps
                if total_steps % train_every == 0:
                    _ = self.train_step()

                episode_reward += reward
                obs = next_obs
                total_steps += 1
                episode_steps += 1

                if done:
                    break

            # Decay epsilon at end of episode
            self.decay_epsilon()

            # Log metrics
            metrics.log_episode(
                episode, episode_reward, episode_steps, self.epsilon
            )

            # Print progress
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
                    + String(total_steps)
                )

            # Auto-checkpoint
            if self.checkpoint_every > 0 and len(self.checkpoint_path) > 0:
                if (episode + 1) % self.checkpoint_every == 0:
                    try:
                        self.save_checkpoint(self.checkpoint_path)
                        if verbose:
                            print(
                                "Checkpoint saved at episode " + String(episode + 1)
                            )
                    except:
                        print(
                            "Warning: Failed to save checkpoint at episode "
                            + String(episode + 1)
                        )

        return metrics^

    fn evaluate[
        E: BoxDiscreteActionEnv
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps: Int = 500,
        render: Bool = False,
    ) -> Float64:
        """Evaluate the agent on the environment using greedy policy.

        Note: This uses the current epsilon value. For pure greedy evaluation,
        ensure epsilon is set to 0 or a very small value before calling.

        Args:
            env: The environment to evaluate on.
            num_episodes: Number of evaluation episodes (default: 10).
            max_steps: Maximum steps per episode (default: 500).
            render: Whether to render the environment (default: False).

        Returns:
            Average reward across episodes.
        """
        var total_reward: Float64 = 0.0

        for _ in range(num_episodes):
            var obs = self._list_to_simd(env.reset_obs_list())
            var episode_reward: Float64 = 0.0

            for _ in range(max_steps):
                if render:
                    env.render()

                # Select action (uses current epsilon - typically low after training)
                var action = self.select_action(obs)

                # Step environment
                var result = env.step_obs(action)
                episode_reward += result[1]
                obs = self._list_to_simd(result[0])

                if result[2]:  # done
                    break

            total_reward += episode_reward

        if render:
            env.close()

        return total_reward / Float64(num_episodes)

    fn select_actions_gpu_envs(
        self,
        ctx: DeviceContext,
        mut obs_buf: DeviceBuffer[dtype],
        mut q_buf: DeviceBuffer[dtype],
        mut actions_buf: DeviceBuffer[dtype],
        params_buf: DeviceBuffer[dtype],
        workspace_buf: DeviceBuffer[dtype],
        rng_seed: UInt32,
    ) raises:
        """Select actions for n_envs environments using GPU forward pass.

        This version is for vectorized GPU training where we have n_envs
        parallel environments (different from batch_size used for training).

        Args:
            ctx: GPU device context.
            obs_buf: Pre-allocated GPU buffer for observations [n_envs * obs_dim].
            q_buf: Pre-allocated GPU buffer for Q-values [n_envs * num_actions].
            actions_buf: Pre-allocated GPU buffer for actions [n_envs].
            params_buf: GPU buffer containing current params.
            workspace_buf: Pre-allocated workspace buffer [n_envs * WORKSPACE_PER_SAMPLE].
            rng_seed: Seed for random number generation (should vary per call).
        """
        var obs = LayoutTensor[
            dtype,
            Layout.row_major(Self.n_envs, Self.obs_dim),
            MutAnyOrigin,
        ](obs_buf.unsafe_ptr())
        var q = LayoutTensor[
            dtype,
            Layout.row_major(Self.n_envs, Self.num_actions),
            MutAnyOrigin,
        ](q_buf.unsafe_ptr())

        # Forward pass on GPU for n_envs environments (using pre-allocated workspace)
        self.online_model.forward_gpu_ws[Self.n_envs](
            ctx, obs_buf, q_buf, params_buf, workspace_buf
        )

        # Find argmax with epsilon-greedy
        var actions = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](actions_buf.unsafe_ptr())

        var seed_scalar = Scalar[DType.uint32](rng_seed)

        @parameter
        @always_inline
        fn argmax_kernel_wrapper_envs(
            epsilon: Scalar[dtype],
            q_vals: LayoutTensor[
                dtype,
                Layout.row_major(Self.n_envs, Self.num_actions),
                MutAnyOrigin,
            ],
            acts: LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ],
            base_seed: Scalar[DType.uint32],
        ):
            # Inline argmax with epsilon-greedy using GPU-compatible random
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= Self.n_envs:
                return

            # GPU-compatible random: seed varies with thread AND iteration
            var rng = xorshift32(
                Scalar[DType.uint32](b * 2654435761) + base_seed
            )
            var rand_result = random_uniform[dtype](rng)
            var rand_val = rand_result[0]
            rng = rand_result[1]

            if rand_val < epsilon:
                # Random action using second random draw
                var action_result = random_uniform[dtype](rng)
                acts[b] = Scalar[dtype](
                    Int(action_result[0] * Scalar[dtype](Self.num_actions))
                )
                return

            var best_q = q_vals[b, 0]
            var best_action = 0
            for a in range(1, Self.num_actions):
                var qv = q_vals[b, a]
                if qv > best_q:
                    best_q = qv
                    best_action = a

            acts[b] = Scalar[dtype](best_action)

        ctx.enqueue_function[
            argmax_kernel_wrapper_envs, argmax_kernel_wrapper_envs
        ](
            Scalar[dtype](self.epsilon),
            q,
            actions,
            seed_scalar,
            grid_dim=((Self.n_envs + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )

    fn evaluate_greedy[
        E: BoxDiscreteActionEnv
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps: Int = 500,
        render: Bool = False,
    ) -> Float64:
        """Evaluate the agent using pure greedy policy (epsilon=0).

        This performs evaluation without any exploration, always selecting
        the action with highest Q-value.

        Args:
            env: The environment to evaluate on.
            num_episodes: Number of evaluation episodes (default: 10).
            max_steps: Maximum steps per episode (default: 500).
            render: Whether to render the environment (default: False).

        Returns:
            Average reward across episodes.
        """
        var total_reward: Float64 = 0.0

        for _ in range(num_episodes):
            var obs = self._list_to_simd(env.reset_obs_list())
            var episode_reward: Float64 = 0.0

            for _ in range(max_steps):
                if render:
                    env.render()

                # Greedy action: argmax_a Q(s, a) - no epsilon
                var obs_input = InlineArray[Scalar[dtype], Self.obs_dim](
                    uninitialized=True
                )
                for i in range(Self.obs_dim):
                    obs_input[i] = Scalar[dtype](obs[i])

                var q_values = InlineArray[Scalar[dtype], Self.num_actions](
                    uninitialized=True
                )
                self.online_model.forward[1](obs_input, q_values)

                # Find argmax
                var best_action = 0
                var best_q = q_values[0]
                for a in range(1, Self.num_actions):
                    if q_values[a] > best_q:
                        best_q = q_values[a]
                        best_action = a

                # Step environment
                var result = env.step_obs(best_action)
                episode_reward += result[1]
                obs = self._list_to_simd(result[0])

                if result[2]:  # done
                    break

            total_reward += episode_reward

        if render:
            env.close()

        return total_reward / Float64(num_episodes)

    # =========================================================================
    # GPU Training Methods (Optimized with pre-allocated buffers)
    # =========================================================================

    fn train_step_gpu_online(
        mut self,
        ctx: DeviceContext,
        # Network buffers (pre-allocated)
        mut online_params_buf: DeviceBuffer[dtype],
        mut online_grads_buf: DeviceBuffer[dtype],
        mut online_state_buf: DeviceBuffer[dtype],
        mut target_params_buf: DeviceBuffer[dtype],
        # Batch buffers (pre-allocated) - uses current transition batch directly
        mut obs_buf: DeviceBuffer[dtype],  # Current observations (before step)
        mut next_obs_buf: DeviceBuffer[dtype],  # Next observations (after step)
        mut q_values_buf: DeviceBuffer[dtype],
        mut next_q_values_buf: DeviceBuffer[dtype],
        mut online_next_q_buf: DeviceBuffer[dtype],  # For Double DQN
        mut cache_buf: DeviceBuffer[dtype],
        mut grad_output_buf: DeviceBuffer[dtype],
        mut grad_input_buf: DeviceBuffer[dtype],
        mut targets_buf: DeviceBuffer[dtype],
        mut rewards_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
        mut actions_buf: DeviceBuffer[dtype],
        # Workspace buffer (pre-allocated to avoid GPU memory leak)
        workspace_buf: DeviceBuffer[dtype],
    ) raises -> Float64:
        """Online GPU training step using current batch directly (no replay buffer).

        This is used for vectorized GPU training where we train on the current
        batch of transitions from parallel environments, avoiding CPU-GPU transfers
        for replay buffer operations.

        All buffers are passed in to avoid allocation overhead.
        All operations run entirely on GPU.

        Note: In online mode, obs_buf contains the state BEFORE the step,
        and next_obs_buf should contain the state AFTER the step.
        The training loop is responsible for copying obs to next_obs before stepping.
        """
        # GPU Forward pass: online network with cache (using pre-allocated workspace)
        # obs_buf contains the previous observations (before step)
        self.online_model.forward_gpu_with_cache_ws[Self.batch_size](
            ctx, obs_buf, q_values_buf, online_params_buf, cache_buf, workspace_buf
        )

        # GPU Forward pass: target network (no cache, using pre-allocated workspace)
        self.target_model.forward_gpu_ws[Self.batch_size](
            ctx, next_obs_buf, next_q_values_buf, target_params_buf, workspace_buf
        )

        # GPU TD Target computation
        comptime BATCH_BLOCKS = (Self.batch_size + TPB - 1) // TPB
        var gamma_scalar = Scalar[dtype](self.gamma)

        # Create LayoutTensor views for kernels
        var targets_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](targets_buf.unsafe_ptr())
        var rewards_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var dones_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](dones_buf.unsafe_ptr())
        var next_q_tensor = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size, Self.num_actions),
            MutAnyOrigin,
        ](next_q_values_buf.unsafe_ptr())

        @parameter
        if Self.double_dqn:
            # For Double DQN: forward online network on next_obs (using workspace)
            self.online_model.forward_gpu_ws[Self.batch_size](
                ctx, next_obs_buf, online_next_q_buf, online_params_buf, workspace_buf
            )

            var online_next_tensor = LayoutTensor[
                dtype,
                Layout.row_major(Self.batch_size, Self.num_actions),
                MutAnyOrigin,
            ](online_next_q_buf.unsafe_ptr())

            # Double DQN TD target kernel
            @parameter
            @always_inline
            fn double_td_kernel_wrapper(
                targets_t: LayoutTensor[
                    dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
                ],
                online_next_t: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.batch_size, Self.num_actions),
                    MutAnyOrigin,
                ],
                target_next_t: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.batch_size, Self.num_actions),
                    MutAnyOrigin,
                ],
                rewards_t: LayoutTensor[
                    dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
                ],
                dones_t: LayoutTensor[
                    dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
                ],
                gamma: Scalar[dtype],
            ):
                dqn_double_td_target_kernel[
                    dtype, Self.batch_size, Self.num_actions
                ](
                    targets_t,
                    online_next_t,
                    target_next_t,
                    rewards_t,
                    dones_t,
                    gamma,
                )

            ctx.enqueue_function[
                double_td_kernel_wrapper, double_td_kernel_wrapper
            ](
                targets_tensor,
                online_next_tensor,
                next_q_tensor,
                rewards_tensor,
                dones_tensor,
                gamma_scalar,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
        else:
            # Standard DQN TD target kernel
            @parameter
            @always_inline
            fn td_kernel_wrapper(
                targets_t: LayoutTensor[
                    dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
                ],
                next_q_t: LayoutTensor[
                    dtype,
                    Layout.row_major(Self.batch_size, Self.num_actions),
                    MutAnyOrigin,
                ],
                rewards_t: LayoutTensor[
                    dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
                ],
                dones_t: LayoutTensor[
                    dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
                ],
                gamma: Scalar[dtype],
            ):
                dqn_td_target_kernel[dtype, Self.batch_size, Self.num_actions](
                    targets_t, next_q_t, rewards_t, dones_t, gamma
                )

            ctx.enqueue_function[td_kernel_wrapper, td_kernel_wrapper](
                targets_tensor,
                next_q_tensor,
                rewards_tensor,
                dones_tensor,
                gamma_scalar,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

        # GPU Gradient computation
        var q_tensor = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size, Self.num_actions),
            MutAnyOrigin,
        ](q_values_buf.unsafe_ptr())
        var grad_tensor = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size, Self.num_actions),
            MutAnyOrigin,
        ](grad_output_buf.unsafe_ptr())
        var actions_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](actions_buf.unsafe_ptr())

        # We don't use loss_out in the kernel for now (would need reduction)
        @parameter
        @always_inline
        fn grad_kernel_wrapper(
            grad_t: LayoutTensor[
                dtype,
                Layout.row_major(Self.batch_size, Self.num_actions),
                MutAnyOrigin,
            ],
            q_t: LayoutTensor[
                dtype,
                Layout.row_major(Self.batch_size, Self.num_actions),
                MutAnyOrigin,
            ],
            targets_t: LayoutTensor[
                dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
            actions_t: LayoutTensor[
                dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
        ):
            # Inline the gradient computation
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= Self.batch_size:
                return
            var action = Int(actions_t[b])
            var q_pred = q_t[b, action]
            var td_error = q_pred - targets_t[b]
            for a in range(Self.num_actions):
                if a == action:
                    grad_t[b, a] = (
                        Scalar[dtype](2.0)
                        * td_error
                        / Scalar[dtype](Self.batch_size)
                    )
                else:
                    grad_t[b, a] = Scalar[dtype](0.0)

        ctx.enqueue_function[grad_kernel_wrapper, grad_kernel_wrapper](
            grad_tensor,
            q_tensor,
            targets_tensor,
            actions_tensor,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # GPU Zero gradients
        comptime PARAM_BLOCKS = (Self.NETWORK_PARAM_SIZE + TPB - 1) // TPB
        var grads_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.NETWORK_PARAM_SIZE), MutAnyOrigin
        ](online_grads_buf.unsafe_ptr())

        @parameter
        @always_inline
        fn zero_kernel_wrapper(
            buf: LayoutTensor[
                dtype, Layout.row_major(Self.NETWORK_PARAM_SIZE), MutAnyOrigin
            ],
        ):
            zero_buffer_kernel[dtype, Self.NETWORK_PARAM_SIZE](buf)

        ctx.enqueue_function[zero_kernel_wrapper, zero_kernel_wrapper](
            grads_tensor,
            grid_dim=(PARAM_BLOCKS,),
            block_dim=(TPB,),
        )

        # GPU Backward pass (using pre-allocated workspace)
        self.online_model.backward_gpu_ws[Self.batch_size](
            ctx,
            grad_output_buf,
            grad_input_buf,
            online_params_buf,
            cache_buf,
            online_grads_buf,
            workspace_buf,
        )

        # GPU Optimizer update
        self.online_model.update_gpu(
            ctx, online_params_buf, online_grads_buf, online_state_buf
        )

        # GPU Soft update target network
        var tau_scalar = Scalar[dtype](self.tau)
        var online_params_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.NETWORK_PARAM_SIZE), MutAnyOrigin
        ](online_params_buf.unsafe_ptr())
        var target_params_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.NETWORK_PARAM_SIZE), MutAnyOrigin
        ](target_params_buf.unsafe_ptr())

        @parameter
        @always_inline
        fn soft_update_wrapper(
            target_t: LayoutTensor[
                dtype, Layout.row_major(Self.NETWORK_PARAM_SIZE), MutAnyOrigin
            ],
            source_t: LayoutTensor[
                dtype, Layout.row_major(Self.NETWORK_PARAM_SIZE), MutAnyOrigin
            ],
            tau: Scalar[dtype],
        ):
            soft_update_kernel[dtype, Self.NETWORK_PARAM_SIZE](
                target_t, source_t, tau
            )

        ctx.enqueue_function[soft_update_wrapper, soft_update_wrapper](
            target_params_tensor,
            online_params_tensor,
            tau_scalar,
            grid_dim=(PARAM_BLOCKS,),
            block_dim=(TPB,),
        )

        self.train_step_count += 1
        return 0.0  # Loss computation would need GPU reduction

    fn train_gpu[
        EnvType: GPUDiscreteEnv
    ](
        mut self,
        ctx: DeviceContext,
        mut env: EnvType,
        num_episodes: Int,
        max_steps_per_episode: Int = 500,
        warmup_steps: Int = 1000,
        train_every: Int = 4,
        sync_every: Int = 5,
        verbose: Bool = False,
        print_every: Int = 10,
        environment_name: String = "Environment",
    ) raises -> TrainingMetrics:
        """Train the DQN agent on GPU with optimized buffer management.

        All GPU buffers are pre-allocated once at the start of training.
        Action selection uses GPU forward pass (no CPU param sync needed).
        Environment interaction happens on CPU, all training operations on GPU.

        Args:
            ctx: GPU device context.
            env: The environment to train on (must implement BoxDiscreteActionEnv).
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode (default: 500).
            warmup_steps: Number of random steps to fill replay buffer (default: 1000).
            train_every: Train every N steps (default: 4).
            sync_every: Sync GPU params to CPU every N episodes for backup (default: 5).
            verbose: Whether to print progress (default: False).
            print_every: Print progress every N episodes if verbose (default: 10).
            environment_name: Name of environment for metrics labeling.

        Returns:
            TrainingMetrics object with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="DQN (GPU)" if not Self.double_dqn else "Double DQN (GPU)",
            environment_name=environment_name,
        )

        # =====================================================================
        # Pre-allocate ALL GPU buffers (done once!)
        # =====================================================================
        # Network parameters
        comptime PARAM_SIZE = Self.NETWORK_PARAM_SIZE
        comptime STATE_SIZE = PARAM_SIZE * 2  # Adam has 2 state values per param

        # Environment buffers (n_envs parallel environments)
        comptime ENV_OBS_SIZE = Self.n_envs * Self.obs_dim
        comptime ENV_Q_SIZE = Self.n_envs * Self.num_actions

        # Training buffers (batch_size samples for gradient updates)
        comptime BATCH_OBS_SIZE = Self.batch_size * Self.obs_dim
        comptime BATCH_Q_SIZE = Self.batch_size * Self.num_actions
        comptime BATCH_CACHE_SIZE = Self.batch_size * Self.NETWORK_CACHE_SIZE

        # Workspace sizes for forward passes (5-layer network = 4*HIDDEN intermediates)
        # This prevents GPU memory leaks from repeated internal buffer allocations
        comptime WORKSPACE_PER_SAMPLE = 4 * Self.HIDDEN
        comptime ENV_WORKSPACE_SIZE = Self.n_envs * WORKSPACE_PER_SAMPLE
        comptime BATCH_WORKSPACE_SIZE = Self.batch_size * WORKSPACE_PER_SAMPLE

        # Network parameter buffers
        var online_params_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)
        var online_grads_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)
        var online_state_buf = ctx.enqueue_create_buffer[dtype](STATE_SIZE)
        var target_params_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)

        # =====================================================================
        # Environment buffers (n_envs parallel environments)
        # =====================================================================
        var prev_obs_buf = ctx.enqueue_create_buffer[dtype](ENV_OBS_SIZE)
        var obs_buf = ctx.enqueue_create_buffer[dtype](ENV_OBS_SIZE)
        var obs = LayoutTensor[
            dtype,
            Layout.row_major(Self.n_envs, Self.obs_dim),
            MutAnyOrigin,
        ](obs_buf.unsafe_ptr())
        var env_q_values_buf = ctx.enqueue_create_buffer[dtype](ENV_Q_SIZE)
        var rewards_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var rewards = LayoutTensor[
            dtype,
            Layout.row_major(Self.n_envs),
            MutAnyOrigin,
        ](rewards_buf.unsafe_ptr())
        var dones_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var dones = LayoutTensor[
            dtype,
            Layout.row_major(Self.n_envs),
            MutAnyOrigin,
        ](dones_buf.unsafe_ptr())
        var actions_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var actions = LayoutTensor[
            dtype,
            Layout.row_major(Self.n_envs),
            MutAnyOrigin,
        ](actions_buf.unsafe_ptr())

        # Host buffers for CPU-GPU transfer (environment data)
        var obs_host = ctx.enqueue_create_host_buffer[dtype](ENV_OBS_SIZE)
        var rewards_host = ctx.enqueue_create_host_buffer[dtype](Self.n_envs)
        var dones_host = ctx.enqueue_create_host_buffer[dtype](Self.n_envs)
        var actions_host = ctx.enqueue_create_host_buffer[dtype](Self.n_envs)

        # =====================================================================
        # Training buffers (batch_size samples for gradient updates)
        # =====================================================================
        var q_values_buf = ctx.enqueue_create_buffer[dtype](BATCH_Q_SIZE)
        var next_q_values_buf = ctx.enqueue_create_buffer[dtype](BATCH_Q_SIZE)
        var online_next_q_buf = ctx.enqueue_create_buffer[dtype](BATCH_Q_SIZE)
        var cache_buf = ctx.enqueue_create_buffer[dtype](BATCH_CACHE_SIZE)
        var grad_output_buf = ctx.enqueue_create_buffer[dtype](BATCH_Q_SIZE)
        var grad_input_buf = ctx.enqueue_create_buffer[dtype](BATCH_OBS_SIZE)
        var targets_buf = ctx.enqueue_create_buffer[dtype](Self.batch_size)
        var next_obs_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH_OBS_SIZE
        )

        # =====================================================================
        # Pre-allocated workspace buffers (prevents GPU memory leak!)
        # =====================================================================
        var env_workspace_buf = ctx.enqueue_create_buffer[dtype](
            ENV_WORKSPACE_SIZE
        )
        var batch_workspace_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_WORKSPACE_SIZE
        )

        # =====================================================================
        # GPU Replay Buffer Storage (stateless - just buffers, no struct)
        # =====================================================================
        comptime RB_CAPACITY = Self.buffer_capacity
        comptime RB_OBS_SIZE = RB_CAPACITY * Self.obs_dim

        var rb_states_buf = ctx.enqueue_create_buffer[dtype](RB_OBS_SIZE)
        var rb_actions_buf = ctx.enqueue_create_buffer[dtype](RB_CAPACITY)
        var rb_rewards_buf = ctx.enqueue_create_buffer[dtype](RB_CAPACITY)
        var rb_next_states_buf = ctx.enqueue_create_buffer[dtype](RB_OBS_SIZE)
        var rb_dones_buf = ctx.enqueue_create_buffer[dtype](RB_CAPACITY)

        # Sample indices buffer (for random sampling)
        var sample_indices_buf = ctx.enqueue_create_buffer[DType.int32](
            Self.batch_size
        )

        # Sampled batch buffers (for training from replay buffer)
        var sampled_obs_buf = ctx.enqueue_create_buffer[dtype](BATCH_OBS_SIZE)
        var sampled_actions_buf = ctx.enqueue_create_buffer[dtype](
            Self.batch_size
        )
        var sampled_rewards_buf = ctx.enqueue_create_buffer[dtype](
            Self.batch_size
        )
        var sampled_next_obs_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_OBS_SIZE
        )
        var sampled_dones_buf = ctx.enqueue_create_buffer[dtype](
            Self.batch_size
        )

        # Replay buffer state (managed on CPU)
        var rb_write_idx: Int = 0
        var rb_size: Int = 0

        # Episode tracking buffers (GPU) - one per environment
        var episode_rewards_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var episode_steps_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var completed_rewards_buf = ctx.enqueue_create_buffer[dtype](
            Self.n_envs
        )
        var completed_steps_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var completed_mask_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)

        # Episode tracking host buffers (for reading back completed episodes)
        var completed_rewards_host = ctx.enqueue_create_host_buffer[dtype](
            Self.n_envs
        )
        var completed_steps_host = ctx.enqueue_create_host_buffer[dtype](
            Self.n_envs
        )
        var completed_mask_host = ctx.enqueue_create_host_buffer[dtype](
            Self.n_envs
        )

        # Action selection buffers (batch=1 for single observation)
        var action_obs_buf = ctx.enqueue_create_buffer[dtype](Self.obs_dim)
        var action_q_buf = ctx.enqueue_create_buffer[dtype](Self.num_actions)
        var action_obs_host = ctx.enqueue_create_host_buffer[dtype](
            Self.obs_dim
        )
        var action_q_host = ctx.enqueue_create_host_buffer[dtype](
            Self.num_actions
        )

        # Copy CPU params to GPU
        self.online_model.copy_params_to_device(ctx, online_params_buf)
        self.online_model.copy_state_to_device(ctx, online_state_buf)
        self.target_model.copy_params_to_device(ctx, target_params_buf)

        # =====================================================================
        # Initialize episode tracking buffers to zero
        # =====================================================================
        comptime ENV_BLOCKS = (Self.n_envs + TPB - 1) // TPB
        comptime BATCH_BLOCKS = (Self.batch_size + TPB - 1) // TPB

        var episode_rewards_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](episode_rewards_buf.unsafe_ptr())
        var episode_steps_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](episode_steps_buf.unsafe_ptr())

        @parameter
        @always_inline
        fn zero_episode_rewards(
            buf: LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ],
        ):
            zero_buffer_kernel[dtype, Self.n_envs](buf)

        ctx.enqueue_function[zero_episode_rewards, zero_episode_rewards](
            episode_rewards_tensor,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )
        ctx.enqueue_function[zero_episode_rewards, zero_episode_rewards](
            episode_steps_tensor,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # =====================================================================
        # Initial environment reset (all n_envs environments)
        # =====================================================================
        EnvType.reset_kernel_gpu[Self.n_envs, Self.obs_dim](ctx, obs_buf)

        if verbose:
            print("GPU buffers allocated. Starting training...")
            print("Running " + String(Self.n_envs) + " parallel environments")
            print("Training batch size: " + String(Self.batch_size))

        # =====================================================================
        # Vectorized Training loop
        # Runs n_envs environments in parallel, counting episodes as they complete
        # =====================================================================
        var total_steps = 0
        var completed_episodes = 0
        var last_print_episode = 0
        var last_checkpoint_episode = 0

        # Create tensor views for episode tracking kernels (n_envs environments)
        var rewards_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var dones_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](dones_buf.unsafe_ptr())
        var completed_rewards_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](completed_rewards_buf.unsafe_ptr())
        var completed_steps_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](completed_steps_buf.unsafe_ptr())
        var completed_mask_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](completed_mask_buf.unsafe_ptr())

        # Define kernel wrappers for episode tracking (n_envs)
        @parameter
        @always_inline
        fn accum_rewards_wrapper(
            ep_rewards: LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ],
            step_rewards: LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ],
        ):
            accumulate_rewards_kernel[dtype, Self.n_envs](
                ep_rewards, step_rewards
            )

        @parameter
        @always_inline
        fn incr_steps_wrapper(
            ep_steps: LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ],
        ):
            increment_steps_kernel[dtype, Self.n_envs](ep_steps)

        @parameter
        @always_inline
        fn extract_completed_wrapper(
            d: LayoutTensor[dtype, Layout.row_major(Self.n_envs), MutAnyOrigin],
            ep_r: LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ],
            ep_s: LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ],
            comp_r: LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ],
            comp_s: LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ],
            comp_m: LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ],
        ):
            extract_completed_episodes_kernel[dtype, Self.n_envs](
                d, ep_r, ep_s, comp_r, comp_s, comp_m
            )

        # Create tensor views for copy kernel (ENV_OBS_SIZE for environment data)
        var prev_obs_tensor = LayoutTensor[
            dtype, Layout.row_major(ENV_OBS_SIZE), MutAnyOrigin
        ](prev_obs_buf.unsafe_ptr())
        var obs_flat_tensor = LayoutTensor[
            dtype, Layout.row_major(ENV_OBS_SIZE), MutAnyOrigin
        ](obs_buf.unsafe_ptr())

        @parameter
        @always_inline
        fn copy_obs_wrapper(
            dst: LayoutTensor[
                dtype, Layout.row_major(ENV_OBS_SIZE), MutAnyOrigin
            ],
            src: LayoutTensor[
                dtype, Layout.row_major(ENV_OBS_SIZE), MutAnyOrigin
            ],
        ):
            copy_buffer_kernel[dtype, ENV_OBS_SIZE](dst, src)

        comptime OBS_BLOCKS = (ENV_OBS_SIZE + TPB - 1) // TPB

        # =====================================================================
        # Replay buffer tensor views and kernel wrappers
        # =====================================================================
        var rb_states_tensor = LayoutTensor[
            dtype, Layout.row_major(RB_CAPACITY, Self.obs_dim), MutAnyOrigin
        ](rb_states_buf.unsafe_ptr())
        var rb_actions_tensor = LayoutTensor[
            dtype, Layout.row_major(RB_CAPACITY), MutAnyOrigin
        ](rb_actions_buf.unsafe_ptr())
        var rb_rewards_tensor = LayoutTensor[
            dtype, Layout.row_major(RB_CAPACITY), MutAnyOrigin
        ](rb_rewards_buf.unsafe_ptr())
        var rb_next_states_tensor = LayoutTensor[
            dtype, Layout.row_major(RB_CAPACITY, Self.obs_dim), MutAnyOrigin
        ](rb_next_states_buf.unsafe_ptr())
        var rb_dones_tensor = LayoutTensor[
            dtype, Layout.row_major(RB_CAPACITY), MutAnyOrigin
        ](rb_dones_buf.unsafe_ptr())

        var sample_indices_tensor = LayoutTensor[
            DType.int32, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](sample_indices_buf.unsafe_ptr())

        var sampled_obs_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size, Self.obs_dim), MutAnyOrigin
        ](sampled_obs_buf.unsafe_ptr())
        var sampled_actions_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](sampled_actions_buf.unsafe_ptr())
        var sampled_rewards_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](sampled_rewards_buf.unsafe_ptr())
        var sampled_next_obs_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size, Self.obs_dim), MutAnyOrigin
        ](sampled_next_obs_buf.unsafe_ptr())
        var sampled_dones_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
        ](sampled_dones_buf.unsafe_ptr())

        # Prev obs as 2D tensor for store kernel (n_envs environments)
        var prev_obs_2d_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs, Self.obs_dim), MutAnyOrigin
        ](prev_obs_buf.unsafe_ptr())
        var obs_2d_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs, Self.obs_dim), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var actions_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](actions_buf.unsafe_ptr())

        @parameter
        @always_inline
        fn store_transitions_wrapper(
            states: LayoutTensor[
                dtype,
                Layout.row_major(Self.n_envs, Self.obs_dim),
                MutAnyOrigin,
            ],
            actions: LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ],
            rewards: LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ],
            next_states: LayoutTensor[
                dtype,
                Layout.row_major(Self.n_envs, Self.obs_dim),
                MutAnyOrigin,
            ],
            dones: LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ],
            buf_states: LayoutTensor[
                dtype, Layout.row_major(RB_CAPACITY, Self.obs_dim), MutAnyOrigin
            ],
            buf_actions: LayoutTensor[
                dtype, Layout.row_major(RB_CAPACITY), MutAnyOrigin
            ],
            buf_rewards: LayoutTensor[
                dtype, Layout.row_major(RB_CAPACITY), MutAnyOrigin
            ],
            buf_next_states: LayoutTensor[
                dtype, Layout.row_major(RB_CAPACITY, Self.obs_dim), MutAnyOrigin
            ],
            buf_dones: LayoutTensor[
                dtype, Layout.row_major(RB_CAPACITY), MutAnyOrigin
            ],
            write_idx: Scalar[DType.int32],
        ):
            store_transitions_kernel[
                dtype, Self.n_envs, Self.obs_dim, RB_CAPACITY
            ](
                states,
                actions,
                rewards,
                next_states,
                dones,
                buf_states,
                buf_actions,
                buf_rewards,
                buf_next_states,
                buf_dones,
                write_idx,
            )

        @parameter
        @always_inline
        fn sample_indices_wrapper(
            indices: LayoutTensor[
                DType.int32, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
            buffer_size: Scalar[DType.int32],
            rng_seed: Scalar[DType.uint32],
        ):
            sample_indices_kernel[dtype, Self.batch_size](
                indices, buffer_size, rng_seed
            )

        @parameter
        @always_inline
        fn gather_batch_wrapper(
            batch_states: LayoutTensor[
                dtype,
                Layout.row_major(Self.batch_size, Self.obs_dim),
                MutAnyOrigin,
            ],
            batch_actions: LayoutTensor[
                dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
            batch_rewards: LayoutTensor[
                dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
            batch_next_states: LayoutTensor[
                dtype,
                Layout.row_major(Self.batch_size, Self.obs_dim),
                MutAnyOrigin,
            ],
            batch_dones: LayoutTensor[
                dtype, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
            buf_states: LayoutTensor[
                dtype, Layout.row_major(RB_CAPACITY, Self.obs_dim), MutAnyOrigin
            ],
            buf_actions: LayoutTensor[
                dtype, Layout.row_major(RB_CAPACITY), MutAnyOrigin
            ],
            buf_rewards: LayoutTensor[
                dtype, Layout.row_major(RB_CAPACITY), MutAnyOrigin
            ],
            buf_next_states: LayoutTensor[
                dtype, Layout.row_major(RB_CAPACITY, Self.obs_dim), MutAnyOrigin
            ],
            buf_dones: LayoutTensor[
                dtype, Layout.row_major(RB_CAPACITY), MutAnyOrigin
            ],
            indices: LayoutTensor[
                DType.int32, Layout.row_major(Self.batch_size), MutAnyOrigin
            ],
        ):
            gather_batch_kernel[
                dtype, Self.batch_size, Self.obs_dim, RB_CAPACITY
            ](
                batch_states,
                batch_actions,
                batch_rewards,
                batch_next_states,
                batch_dones,
                buf_states,
                buf_actions,
                buf_rewards,
                buf_next_states,
                buf_dones,
                indices,
            )

        # =====================================================================
        # Warmup phase: fill replay buffer with random transitions
        # =====================================================================
        if verbose:
            print(
                "Warmup: collecting "
                + String(warmup_steps)
                + " random transitions..."
            )

        var saved_epsilon = self.epsilon
        self.epsilon = 1.0  # Force random actions during warmup

        var warmup_count = 0
        while warmup_count < warmup_steps:
            # Copy current obs to prev_obs
            ctx.enqueue_function[copy_obs_wrapper, copy_obs_wrapper](
                prev_obs_tensor,
                obs_flat_tensor,
                grid_dim=(OBS_BLOCKS,),
                block_dim=(TPB,),
            )

            # Select random actions (epsilon=1.0)
            self.select_actions_gpu_envs(
                ctx,
                obs_buf,
                env_q_values_buf,
                actions_buf,
                online_params_buf,
                env_workspace_buf,
                UInt32(warmup_count * 7919 + 12345),  # Varying seed
            )

            # Step environments
            EnvType.step_kernel_gpu[Self.n_envs, Self.obs_dim](
                ctx, obs_buf, actions_buf, rewards_buf, dones_buf
            )

            # Store transitions to replay buffer
            ctx.enqueue_function[
                store_transitions_wrapper, store_transitions_wrapper
            ](
                prev_obs_2d_tensor,
                actions_tensor,
                rewards_tensor,
                obs_2d_tensor,
                dones_tensor,
                rb_states_tensor,
                rb_actions_tensor,
                rb_rewards_tensor,
                rb_next_states_tensor,
                rb_dones_tensor,
                Scalar[DType.int32](rb_write_idx),
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )

            # Update replay buffer state
            rb_write_idx = (rb_write_idx + Self.n_envs) % RB_CAPACITY
            rb_size = min(rb_size + Self.n_envs, RB_CAPACITY)
            warmup_count += Self.n_envs

            # Reset done environments
            var rng_seed = UInt32(warmup_count * 7919 + 42)
            EnvType.selective_reset_kernel_gpu[Self.n_envs, Self.obs_dim](
                ctx, obs_buf, dones_buf, rng_seed
            )

        self.epsilon = saved_epsilon  # Restore epsilon

        # Reset ALL environments after warmup to start fresh episodes
        # (warmup may leave envs mid-episode, which would give incorrect episode rewards)
        EnvType.reset_kernel_gpu[Self.n_envs, Self.obs_dim](ctx, obs_buf)

        if verbose:
            print("Warmup complete. Replay buffer size: " + String(rb_size))

        # =====================================================================
        # Timing counters (for debugging performance)
        # =====================================================================
        from time import perf_counter_ns

        var time_action_select: UInt = 0
        var time_env_step: UInt = 0
        var time_store: UInt = 0
        var time_train: UInt = 0
        var time_episode_track: UInt = 0
        var iteration_count = 0

        # =====================================================================
        # Main Training Loop
        # =====================================================================
        while completed_episodes < num_episodes:
            var t0 = perf_counter_ns()

            # =================================================================
            # Copy current observations to prev_obs for training
            # =================================================================
            ctx.enqueue_function[copy_obs_wrapper, copy_obs_wrapper](
                prev_obs_tensor,
                obs_flat_tensor,
                grid_dim=(OBS_BLOCKS,),
                block_dim=(TPB,),
            )

            # =================================================================
            # Select actions using GPU forward pass (n_envs environments)
            # =================================================================
            self.select_actions_gpu_envs(
                ctx,
                obs_buf,
                env_q_values_buf,
                actions_buf,
                online_params_buf,
                env_workspace_buf,
                UInt32(
                    total_steps * 2654435761 + iteration_count * 7919
                ),  # Varying seed
            )
            ctx.synchronize()
            var t1 = perf_counter_ns()
            time_action_select += t1 - t0

            # =================================================================
            # Step all environments on GPU (n_envs environments)
            # After step, obs_buf contains next_obs
            # =================================================================
            EnvType.step_kernel_gpu[Self.n_envs, Self.obs_dim](
                ctx, obs_buf, actions_buf, rewards_buf, dones_buf
            )
            ctx.synchronize()
            var t2 = perf_counter_ns()
            time_env_step += t2 - t1

            # =================================================================
            # Accumulate rewards and increment steps on GPU (n_envs)
            # =================================================================
            ctx.enqueue_function[accum_rewards_wrapper, accum_rewards_wrapper](
                episode_rewards_tensor,
                rewards_tensor,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
            ctx.enqueue_function[incr_steps_wrapper, incr_steps_wrapper](
                episode_steps_tensor,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )

            total_steps += Self.n_envs  # Each step processes n_envs transitions

            # =================================================================
            # Store transitions to GPU replay buffer (n_envs transitions)
            # prev_obs = state before step, obs = next_state after step
            # =================================================================
            ctx.enqueue_function[
                store_transitions_wrapper, store_transitions_wrapper
            ](
                prev_obs_2d_tensor,
                actions_tensor,
                rewards_tensor,
                obs_2d_tensor,  # After step, obs contains next_obs
                dones_tensor,
                rb_states_tensor,
                rb_actions_tensor,
                rb_rewards_tensor,
                rb_next_states_tensor,
                rb_dones_tensor,
                Scalar[DType.int32](rb_write_idx),
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )

            # Update replay buffer state (CPU-side tracking)
            rb_write_idx = (rb_write_idx + Self.n_envs) % RB_CAPACITY
            rb_size = min(rb_size + Self.n_envs, RB_CAPACITY)
            ctx.synchronize()
            var t3 = perf_counter_ns()
            time_store += t3 - t2

            # Initialize t4 for timing (will be updated after training)
            var t4 = t3

            # =================================================================
            # Training: Train proportional to new data collected
            # With n_envs parallel environments, we collect n_envs transitions
            # per iteration. Train n_envs / (batch_size * train_every) times
            # to maintain a consistent training-to-data ratio.
            # =================================================================
            var should_train = (
                rb_size >= Self.batch_size
                and iteration_count % train_every == 0
            )

            if should_train:
                # Number of training steps per iteration
                # With n_envs=1024, batch_size=256, train_every=1: train 4 times
                # With n_envs=1024, batch_size=256, train_every=4: train 1 time
                var num_train_steps = max(
                    1, Self.n_envs // (Self.batch_size * train_every)
                )

                for train_idx in range(num_train_steps):
                    # Use better RNG seed mixing for each training step
                    var raw_seed = UInt32(
                        total_steps * 2654435761
                        + train_idx * 1013904223
                        + iteration_count * 7919
                    )
                    var rng_seed = Scalar[DType.uint32](
                        (raw_seed ^ (raw_seed >> 16)) * 2246822519
                    )
                    ctx.enqueue_function[
                        sample_indices_wrapper, sample_indices_wrapper
                    ](
                        sample_indices_tensor,
                        Scalar[DType.int32](rb_size),
                        rng_seed,
                        grid_dim=(BATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )

                    # Gather sampled transitions into batch tensors
                    ctx.enqueue_function[
                        gather_batch_wrapper, gather_batch_wrapper
                    ](
                        sampled_obs_tensor,
                        sampled_actions_tensor,
                        sampled_rewards_tensor,
                        sampled_next_obs_tensor,
                        sampled_dones_tensor,
                        rb_states_tensor,
                        rb_actions_tensor,
                        rb_rewards_tensor,
                        rb_next_states_tensor,
                        rb_dones_tensor,
                        sample_indices_tensor,
                        grid_dim=(BATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )

                    # Train on sampled batch
                    _ = self.train_step_gpu_online(
                        ctx,
                        online_params_buf,
                        online_grads_buf,
                        online_state_buf,
                        target_params_buf,
                        sampled_obs_buf,  # Sampled observations from replay buffer
                        sampled_next_obs_buf,  # Sampled next observations
                        q_values_buf,
                        next_q_values_buf,
                        online_next_q_buf,
                        cache_buf,
                        grad_output_buf,
                        grad_input_buf,
                        targets_buf,
                        sampled_rewards_buf,  # Sampled rewards
                        sampled_dones_buf,  # Sampled dones
                        sampled_actions_buf,  # Sampled actions
                        batch_workspace_buf,  # Pre-allocated workspace
                    )
                ctx.synchronize()
                t4 = perf_counter_ns()
                time_train += t4 - t3

            # =================================================================
            # Extract completed episodes and reset done environments (n_envs)
            # =================================================================
            ctx.enqueue_function[
                extract_completed_wrapper, extract_completed_wrapper
            ](
                dones_tensor,
                episode_rewards_tensor,
                episode_steps_tensor,
                completed_rewards_tensor,
                completed_steps_tensor,
                completed_mask_tensor,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )

            # Reset done environments (where dones > 0.5)
            # The reset_kernel will reset all envs but step_kernel already updates state
            # So we need a conditional reset - for now, copy dones to check and reset selectively
            ctx.enqueue_copy(completed_rewards_host, completed_rewards_buf)
            ctx.enqueue_copy(completed_steps_host, completed_steps_buf)
            ctx.enqueue_copy(completed_mask_host, completed_mask_buf)
            ctx.synchronize()

            # Process completed episodes on CPU and reset done envs on GPU
            var any_done = False
            for i in range(Self.n_envs):
                if completed_mask_host[i] > 0.5:
                    any_done = True
                    # Log completed episode
                    var ep_reward = Float64(completed_rewards_host[i])
                    var ep_steps = Int(completed_steps_host[i])
                    metrics.log_episode(
                        completed_episodes, ep_reward, ep_steps, self.epsilon
                    )
                    completed_episodes += 1

                    # Decay epsilon after each episode
                    self.decay_epsilon()

            # Selectively reset only done environments
            # Note: extract_completed_episodes_kernel already reset episode tracking for done envs
            if any_done:
                # Use varying seed based on total_steps for different random init each time
                var rng_seed = UInt32(
                    total_steps * 7919 + 42
                )  # Prime-based variation
                EnvType.selective_reset_kernel_gpu[Self.n_envs, Self.obs_dim](
                    ctx, obs_buf, dones_buf, rng_seed
                )

            var t5 = perf_counter_ns()
            time_episode_track += (
                t5 - t4
            )  # Episode tracking time (from after training)
            iteration_count += 1

            # =================================================================
            # Sync GPU params to CPU periodically
            # =================================================================
            if (
                completed_episodes > 0
                and (completed_episodes % sync_every == 0)
                and completed_episodes != last_print_episode
            ):
                self.online_model.copy_params_from_device(
                    ctx, online_params_buf
                )

            # =================================================================
            # Print progress (handle batch completions that may skip milestones)
            # =================================================================
            if verbose and completed_episodes > 0:
                # Calculate next print milestone
                var next_milestone = (
                    (last_print_episode // print_every) + 1
                ) * print_every
                if completed_episodes >= next_milestone:
                    last_print_episode = completed_episodes
                    var avg_reward = metrics.mean_reward_last_n(
                        min(print_every, completed_episodes)
                    )
                    print(
                        "Episode "
                        + String(completed_episodes)
                        + " | Avg reward: "
                        + String(avg_reward)[:7]
                        + " | Epsilon: "
                        + String(self.epsilon)[:5]
                        + " | Steps: "
                        + String(total_steps)
                    )

            # =================================================================
            # Auto-checkpoint (similar milestone tracking as printing)
            # =================================================================
            if (
                self.checkpoint_every > 0
                and len(self.checkpoint_path) > 0
                and completed_episodes > 0
            ):
                var next_ckpt_milestone = (
                    (last_checkpoint_episode // self.checkpoint_every) + 1
                ) * self.checkpoint_every
                if completed_episodes >= next_ckpt_milestone:
                    last_checkpoint_episode = completed_episodes
                    # Make sure params are synced to CPU before saving
                    self.online_model.copy_params_from_device(
                        ctx, online_params_buf
                    )
                    self.target_model.copy_params_from_device(
                        ctx, target_params_buf
                    )
                    self.save_checkpoint(self.checkpoint_path)
                    if verbose:
                        print(
                            "Checkpoint saved at episode "
                            + String(completed_episodes)
                        )

        # Copy GPU params back to CPU for evaluation
        self.online_model.copy_params_from_device(ctx, online_params_buf)
        self.target_model.copy_params_from_device(ctx, target_params_buf)

        # Print timing summary
        if verbose:
            var total_time = (
                time_action_select
                + time_env_step
                + time_store
                + time_train
                + time_episode_track
            )
            print()
            print("Timing breakdown (ms):")
            print(
                "  Action select: "
                + String(Float64(time_action_select) / 1e6)[:8]
            )
            print(
                "  Env step:      " + String(Float64(time_env_step) / 1e6)[:8]
            )
            print("  Store trans:   " + String(Float64(time_store) / 1e6)[:8])
            print("  Training:      " + String(Float64(time_train) / 1e6)[:8])
            print(
                "  Episode track: "
                + String(Float64(time_episode_track) / 1e6)[:8]
            )
            print("  Total:         " + String(Float64(total_time) / 1e6)[:8])
            print("  Iterations:    " + String(iteration_count))
            if iteration_count > 0:
                print(
                    "  Avg per iter:  "
                    + String(
                        Float64(total_time) / Float64(iteration_count) / 1e6
                    )[:8]
                    + " ms"
                )

        return metrics^

    # =========================================================================
    # Checkpoint Save/Load
    # =========================================================================

    fn save_checkpoint(self, filepath: String) raises:
        """Save DQN agent state to a single checkpoint file.

        Saves online and target network parameters, optimizer states,
        hyperparameters, and training counters. Replay buffer is NOT saved.

        Args:
            filepath: Path to the checkpoint file (e.g., "dqn_agent.ckpt").

        Example:
            agent.save_checkpoint("checkpoints/dqn_agent.ckpt")
        """
        comptime PARAM_SIZE = Self.NETWORK_PARAM_SIZE
        comptime STATE_SIZE = Self.NETWORK_PARAM_SIZE * Adam.STATE_PER_PARAM

        # Header
        var content = String("# mojo-rl checkpoint v1\n")
        content += "# type: dqn_agent\n"
        content += "# param_size: " + String(PARAM_SIZE) + "\n"
        content += "# state_size: " + String(STATE_SIZE) + "\n"
        content += "# dtype: float32\n"

        # Online network params (write directly to avoid type mismatch)
        content += "online_params:\n"
        for i in range(PARAM_SIZE):
            content += String(Float64(self.online_model.params[i])) + "\n"

        # Online optimizer state
        content += "online_optimizer_state:\n"
        for i in range(STATE_SIZE):
            content += String(Float64(self.online_model.optimizer_state[i])) + "\n"

        # Target network params
        content += "target_params:\n"
        for i in range(PARAM_SIZE):
            content += String(Float64(self.target_model.params[i])) + "\n"

        # Target optimizer state
        content += "target_optimizer_state:\n"
        for i in range(STATE_SIZE):
            content += String(Float64(self.target_model.optimizer_state[i])) + "\n"

        # Metadata: hyperparameters and training state
        content += "metadata:\n"
        content += "gamma=" + String(self.gamma) + "\n"
        content += "tau=" + String(self.tau) + "\n"
        content += "lr=" + String(self.lr) + "\n"
        content += "epsilon=" + String(self.epsilon) + "\n"
        content += "epsilon_min=" + String(self.epsilon_min) + "\n"
        content += "epsilon_decay=" + String(self.epsilon_decay) + "\n"
        content += "train_step_count=" + String(self.train_step_count) + "\n"

        save_checkpoint_file(filepath, content)

    fn load_checkpoint(mut self, filepath: String) raises:
        """Load DQN agent state from a single checkpoint file.

        Loads online and target network parameters, optimizer states,
        hyperparameters, and training counters. Replay buffer starts empty.

        Args:
            filepath: Path to the checkpoint file (e.g., "dqn_agent.ckpt").

        Example:
            agent.load_checkpoint("checkpoints/dqn_agent.ckpt")
        """
        comptime PARAM_SIZE = Self.NETWORK_PARAM_SIZE
        comptime STATE_SIZE = Self.NETWORK_PARAM_SIZE * Adam.STATE_PER_PARAM

        var content = read_checkpoint_file(filepath)

        # Parse file into lines
        var lines = split_lines(content)

        # Find and load each section
        var online_params_start = find_section_start(lines, "online_params:")
        var online_state_start = find_section_start(lines, "online_optimizer_state:")
        var target_params_start = find_section_start(lines, "target_params:")
        var target_state_start = find_section_start(lines, "target_optimizer_state:")

        # Load online network params
        if online_params_start >= 0:
            for i in range(PARAM_SIZE):
                var line = lines[online_params_start + i]
                self.online_model.params[i] = Scalar[dtype](atof(line))

        # Load online optimizer state
        if online_state_start >= 0:
            for i in range(STATE_SIZE):
                var line = lines[online_state_start + i]
                self.online_model.optimizer_state[i] = Scalar[dtype](atof(line))

        # Load target network params
        if target_params_start >= 0:
            for i in range(PARAM_SIZE):
                var line = lines[target_params_start + i]
                self.target_model.params[i] = Scalar[dtype](atof(line))

        # Load target optimizer state
        if target_state_start >= 0:
            for i in range(STATE_SIZE):
                var line = lines[target_state_start + i]
                self.target_model.optimizer_state[i] = Scalar[dtype](atof(line))

        # Load metadata
        var metadata = read_metadata_section(content)

        var gamma_str = get_metadata_value(metadata, "gamma")
        if len(gamma_str) > 0:
            self.gamma = atof(gamma_str)

        var tau_str = get_metadata_value(metadata, "tau")
        if len(tau_str) > 0:
            self.tau = atof(tau_str)

        var lr_str = get_metadata_value(metadata, "lr")
        if len(lr_str) > 0:
            self.lr = atof(lr_str)

        var epsilon_str = get_metadata_value(metadata, "epsilon")
        if len(epsilon_str) > 0:
            self.epsilon = atof(epsilon_str)

        var epsilon_min_str = get_metadata_value(metadata, "epsilon_min")
        if len(epsilon_min_str) > 0:
            self.epsilon_min = atof(epsilon_min_str)

        var epsilon_decay_str = get_metadata_value(metadata, "epsilon_decay")
        if len(epsilon_decay_str) > 0:
            self.epsilon_decay = atof(epsilon_decay_str)

        var train_step_str = get_metadata_value(metadata, "train_step_count")
        if len(train_step_str) > 0:
            self.train_step_count = Int(atol(train_step_str))

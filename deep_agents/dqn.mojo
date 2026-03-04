"""DQN Agent using the new trait-based deep learning architecture.

This DQN implementation uses:
- NetworkState for heap-allocated params + grads + optimizer state
- Network (all-static) for stateless forward/backward ops via LayoutTensor
- GPUNetworkState for device-side training (local to train_gpu)
- GPUReplayBuffer for GPU-side experience replay (local to train_gpu)
- Sequential composition for Q-networks

Features:
- Works with any BoxDiscreteActionEnv (continuous obs, discrete actions)
- Epsilon-greedy exploration with decay
- Target network with soft updates
- Double DQN to reduce overestimation bias (optional)
- GPU support for batch training (all network ops on GPU)
- lr is a compile-time parameter (Adam LR baked in at compile time)

Usage:
    from deep_agents.dqn import DQNAgent
    from envs import CartPoleNative

    var env = CartPoleNative()
    var agent = DQNAgent[4, 2, 64, 10000, 32]()

    # CPU Training
    var metrics = agent.train(env, num_episodes=200)

    # GPU Training
    var ctx = DeviceContext()
    var metrics_gpu = agent.train_gpu(ctx, env, num_episodes=200)

    # Evaluate
    var avg_reward = agent.evaluate(env, num_episodes=10, greedy=True)
"""

from math import exp
from random import random_float64, seed

from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor

from nn.constants import dtype, TILE, TPB
from nn.model import Linear, Sequential, LinearReLU
from nn.optimizer import Adam
from nn.initializer import Kaiming
from nn.training import Network, NetworkState, GPUNetworkState
from nn.utils import fill_inline
from nn.replay import ReplayBuffer, GPUReplayBuffer
from nn.checkpoint import (
    write_checkpoint_header,
    write_metadata_section,
    parse_checkpoint_header,
    read_checkpoint_file,
    read_metadata_section,
    get_metadata_value,
    save_checkpoint_file,
)
from nn.gpu import (
    random_range,
    xorshift32,
    random_uniform,
    zero_buffer_kernel,
    copy_buffer_kernel,
    accumulate_rewards_kernel,
    increment_steps_kernel,
    extract_completed_episodes_kernel,
    selective_reset_tracking_kernel,
)
from core import (
    TrainingMetrics,
    BoxDiscreteActionEnv,
    GPUDiscreteEnv,
    RenderableEnv,
)


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

    var max_q = next_q_values[b, 0]
    for a in range(1, NUM_ACTIONS):
        var q = next_q_values[b, a]
        if q > max_q:
            max_q = q

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

    var best_action = 0
    var best_q = online_next_q[b, 0]
    for a in range(1, NUM_ACTIONS):
        var q = online_next_q[b, a]
        if q > best_q:
            best_q = q
            best_action = a

    var target_q = target_next_q[b, best_action]
    var done_mask = Scalar[dtype](1.0) - dones[b]
    targets[b] = rewards[b] + gamma * target_q * done_mask


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
    lr: Float64 = 0.001,
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
        lr: Adam learning rate — compile-time (default: 0.001).

    Note on batch_size vs n_envs (GPU training):
        n_envs controls parallel data collection; batch_size controls update size.
    """

    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.num_actions
    comptime HIDDEN = Self.hidden_dim

    # Q-network: obs → hidden (ReLU) → hidden (ReLU) → num_actions
    comptime Q_Model = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, Self.ACTIONS],
    ]
    comptime Q_Network = Network[Self.Q_Model, Adam[Self.lr]]

    # Network states (heap-allocated params + grads + optimizer state)
    var online: NetworkState[Self.Q_Model, Adam[Self.lr]]
    var target: NetworkState[Self.Q_Model, Adam[Self.lr]]

    # CPU replay buffer (action_dim=1: store discrete action as scalar)
    var buffer: ReplayBuffer[Self.buffer_capacity, Self.obs_dim, 1, dtype]

    # Hyperparameters
    var gamma: Float64
    var tau: Float64

    # Exploration
    var epsilon: Float64
    var epsilon_min: Float64
    var epsilon_decay: Float64

    # Training state
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
        """Initialize DQN agent.

        Args:
            gamma: Discount factor (default: 0.99).
            tau: Soft update rate for target network (default: 0.005).
            epsilon: Initial exploration rate (default: 1.0).
            epsilon_min: Minimum exploration rate (default: 0.01).
            epsilon_decay: Epsilon decay per episode (default: 0.995).
            checkpoint_every: Save checkpoint every N episodes (0 to disable).
            checkpoint_path: Path for auto-checkpointing.
        """
        self.online = NetworkState[Self.Q_Model, Adam[Self.lr]]()
        self.online.initialize[Kaiming]()
        self.target = NetworkState[Self.Q_Model, Adam[Self.lr]](
            copy=self.online
        )

        self.buffer = ReplayBuffer[
            Self.buffer_capacity, Self.obs_dim, 1, dtype
        ]()

        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.train_step_count = 0
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path

    # =========================================================================
    # Action Selection
    # =========================================================================

    fn select_action(
        self,
        obs: InlineArray[Scalar[dtype], Self.obs_dim],
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
            return Int(random_float64() * Float64(Self.num_actions))

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs.unsafe_ptr())
        var q_arr = InlineArray[Scalar[dtype], Self.ACTIONS](uninitialized=True)
        var q_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](q_arr.unsafe_ptr())
        var p = self.online.params_view()
        Self.Q_Network.forward[1](obs_t, q_t, p)

        var best_action = 0
        var best_q = q_arr[0]
        for a in range(1, Self.num_actions):
            if q_arr[a] > best_q:
                best_q = q_arr[a]
                best_action = a

        return best_action

    fn store_transition(
        mut self,
        obs: InlineArray[Scalar[dtype], Self.obs_dim],
        action: Int,
        reward: Float64,
        next_obs: InlineArray[Scalar[dtype], Self.obs_dim],
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
        var action_arr = InlineArray[Scalar[dtype], 1](
            fill=Scalar[dtype](action)
        )
        self.buffer.add(obs, action_arr, Scalar[dtype](reward), next_obs, done)

    # =========================================================================
    # CPU Training Step
    # =========================================================================

    fn train_step(mut self) -> Float64:
        """Perform one CPU training step.

        Returns:
            Loss value (0 if buffer not ready).
        """
        if not self.buffer.is_ready[Self.batch_size]():
            return 0.0

        var batch_obs = InlineArray[
            Scalar[dtype], Self.batch_size * Self.obs_dim
        ](uninitialized=True)
        var batch_actions_tmp = InlineArray[
            Scalar[dtype], Self.batch_size
        ](uninitialized=True)
        var batch_rewards = InlineArray[Scalar[dtype], Self.batch_size](
            uninitialized=True
        )
        var batch_next_obs = InlineArray[
            Scalar[dtype], Self.batch_size * Self.obs_dim
        ](uninitialized=True)
        var batch_dones = InlineArray[Scalar[dtype], Self.batch_size](
            uninitialized=True
        )

        # action_dim=1 — use a length-1 row per sample then flatten
        var batch_act1 = InlineArray[
            Scalar[dtype], Self.batch_size * 1
        ](uninitialized=True)
        self.buffer.sample[Self.batch_size](
            batch_obs, batch_act1, batch_rewards, batch_next_obs, batch_dones
        )
        for i in range(Self.batch_size):
            batch_actions_tmp[i] = batch_act1[i]

        # LayoutTensor views over sampled data
        var obs_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size, Self.OBS),
            MutAnyOrigin,
        ](batch_obs.unsafe_ptr())
        var next_obs_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size, Self.OBS),
            MutAnyOrigin,
        ](batch_next_obs.unsafe_ptr())

        # Forward: online with cache
        var q_arr = InlineArray[
            Scalar[dtype], Self.batch_size * Self.ACTIONS
        ](uninitialized=True)
        var q_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size, Self.ACTIONS),
            MutAnyOrigin,
        ](q_arr.unsafe_ptr())
        var cache_arr = InlineArray[
            Scalar[dtype], Self.batch_size * Self.Q_Model.CACHE_SIZE
        ](uninitialized=True)
        var cache_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size, Self.Q_Model.CACHE_SIZE),
            MutAnyOrigin,
        ](cache_arr.unsafe_ptr())

        var p_online = self.online.params_view()
        Self.Q_Network.forward_with_cache[Self.batch_size](
            obs_t, q_t, p_online, cache_t
        )

        # Forward: target (no cache)
        var next_q_arr = InlineArray[
            Scalar[dtype], Self.batch_size * Self.ACTIONS
        ](uninitialized=True)
        var next_q_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size, Self.ACTIONS),
            MutAnyOrigin,
        ](next_q_arr.unsafe_ptr())
        var p_target = self.target.params_view()
        Self.Q_Network.forward[Self.batch_size](next_obs_t, next_q_t, p_target)

        # Compute TD targets
        var targets = InlineArray[Scalar[dtype], Self.batch_size](
            uninitialized=True
        )

        comptime if Self.double_dqn:
            var online_next_arr = InlineArray[
                Scalar[dtype], Self.batch_size * Self.ACTIONS
            ](uninitialized=True)
            var online_next_t = LayoutTensor[
                dtype,
                Layout.row_major(Self.batch_size, Self.ACTIONS),
                MutAnyOrigin,
            ](online_next_arr.unsafe_ptr())
            Self.Q_Network.forward[Self.batch_size](
                next_obs_t, online_next_t, p_online
            )

            for b in range(Self.batch_size):
                var best_action = 0
                var best_q = online_next_arr[b * Self.ACTIONS]
                for a in range(1, Self.ACTIONS):
                    var q = online_next_arr[b * Self.ACTIONS + a]
                    if q > best_q:
                        best_q = q
                        best_action = a

                var next_q = next_q_arr[b * Self.ACTIONS + best_action]
                var done_mask = Scalar[dtype](1.0) - batch_dones[b]
                targets[b] = (
                    batch_rewards[b]
                    + Scalar[dtype](self.gamma) * next_q * done_mask
                )
        else:
            for b in range(Self.batch_size):
                var max_next_q = next_q_arr[b * Self.ACTIONS]
                for a in range(1, Self.ACTIONS):
                    var q = next_q_arr[b * Self.ACTIONS + a]
                    if q > max_next_q:
                        max_next_q = q

                var done_mask = Scalar[dtype](1.0) - batch_dones[b]
                targets[b] = (
                    batch_rewards[b]
                    + Scalar[dtype](self.gamma) * max_next_q * done_mask
                )

        # Compute gradient (MSE, masked to taken action)
        var grad_arr = InlineArray[
            Scalar[dtype], Self.batch_size * Self.ACTIONS
        ](uninitialized=True)
        var total_loss: Float64 = 0.0

        for b in range(Self.batch_size):
            var action = Int(batch_actions_tmp[b])
            var q_pred = q_arr[b * Self.ACTIONS + action]
            var td_error = q_pred - targets[b]
            total_loss += Float64(td_error * td_error)

            for a in range(Self.ACTIONS):
                if a == action:
                    grad_arr[b * Self.ACTIONS + a] = (
                        Scalar[dtype](2.0)
                        * td_error
                        / Scalar[dtype](Self.batch_size)
                    )
                else:
                    grad_arr[b * Self.ACTIONS + a] = Scalar[dtype](0.0)

        total_loss /= Float64(Self.batch_size)

        # Backward
        var grad_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size, Self.ACTIONS),
            MutAnyOrigin,
        ](grad_arr.unsafe_ptr())
        var grad_in_arr = InlineArray[
            Scalar[dtype], Self.batch_size * Self.OBS
        ](uninitialized=True)
        var grad_in_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.batch_size, Self.OBS),
            MutAnyOrigin,
        ](grad_in_arr.unsafe_ptr())
        var g = self.online.grads_view()

        self.online.zero_grads()
        Self.Q_Network.backward[Self.batch_size](
            grad_t, grad_in_t, p_online, cache_t, g
        )
        self.online.optimizer_step()

        self.target.soft_update_from(self.online, self.tau)
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
    # High-level CPU training and evaluation
    # =========================================================================

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
            env: Environment (must implement BoxDiscreteActionEnv).
            num_episodes: Number of training episodes.
            max_steps_per_episode: Maximum steps per episode (default: 500).
            warmup_steps: Random steps to fill replay buffer (default: 1000).
            train_every: Train every N steps (default: 4).
            verbose: Print progress (default: False).
            print_every: Print every N episodes if verbose (default: 10).
            environment_name: Name for metrics labeling.

        Returns:
            TrainingMetrics with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="DQN" if not Self.double_dqn else "Double DQN",
            environment_name=environment_name,
        )

        # Warmup: fill buffer with random transitions
        var warmup_obs = InlineArray[Scalar[dtype], Self.obs_dim](
            uninitialized=True
        )
        var warmup_next = InlineArray[Scalar[dtype], Self.obs_dim](
            uninitialized=True
        )
        fill_inline(env.reset_obs_list(), warmup_obs)
        var warmup_count = 0
        while warmup_count < warmup_steps:
            var action = Int(random_float64() * Float64(Self.num_actions))
            var result = env.step_obs(action)
            fill_inline(result[0], warmup_next)
            self.store_transition(
                warmup_obs, action, Float64(result[1]), warmup_next, result[2]
            )
            fill_inline(result[0], warmup_obs)
            warmup_count += 1
            if result[2]:
                fill_inline(env.reset_obs_list(), warmup_obs)

        # Training loop
        var total_steps = 0
        var obs = InlineArray[Scalar[dtype], Self.obs_dim](uninitialized=True)
        var next_obs = InlineArray[Scalar[dtype], Self.obs_dim](
            uninitialized=True
        )
        for episode in range(num_episodes):
            fill_inline(env.reset_obs_list(), obs)
            var episode_reward: Float64 = 0.0
            var episode_steps = 0

            for step in range(max_steps_per_episode):
                var action = self.select_action(obs)
                var result = env.step_obs(action)
                fill_inline(result[0], next_obs)
                var reward = Float64(result[1])
                var done = result[2]

                self.store_transition(obs, action, reward, next_obs, done)

                if total_steps % train_every == 0:
                    _ = self.train_step()

                episode_reward += reward
                fill_inline(result[0], obs)
                total_steps += 1
                episode_steps += 1

                if done:
                    break

            self.decay_epsilon()
            metrics.log_episode(
                episode, episode_reward, episode_steps, self.epsilon
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
                    + String(total_steps)
                )

            if self.checkpoint_every > 0 and len(self.checkpoint_path) > 0:
                if (episode + 1) % self.checkpoint_every == 0:
                    try:
                        self.save_checkpoint(self.checkpoint_path)
                        if verbose:
                            print(
                                "Checkpoint saved at episode "
                                + String(episode + 1)
                            )
                    except:
                        print(
                            "Warning: Failed to save checkpoint at episode "
                            + String(episode + 1)
                        )

        return metrics^

    fn evaluate[
        E: BoxDiscreteActionEnv & RenderableEnv
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps_per_episode: Int = 500,
        verbose: Bool = False,
        render: Bool = False,
        frame_delay_ms: Int = 16,
        greedy: Bool = True,
    ) raises -> Float64:
        """Evaluate the agent on the environment.

        Args:
            env: Environment to evaluate on.
            num_episodes: Number of evaluation episodes (default: 10).
            max_steps_per_episode: Maximum steps per episode (default: 500).
            verbose: Print per-episode results (default: False).
            render: Render the environment (default: False).
            frame_delay_ms: Delay between frames in ms (default: 16).
            greedy: Use pure greedy policy (epsilon=0) when True (default: True).

        Returns:
            Average reward across episodes.
        """
        var total_reward: Float64 = 0.0
        var quit_requested = False

        if render:
            _ = env.init_renderer()

        var obs = InlineArray[Scalar[dtype], Self.obs_dim](uninitialized=True)
        for episode in range(num_episodes):
            if quit_requested:
                break

            fill_inline(env.reset_obs_list(), obs)
            var episode_reward: Float64 = 0.0
            var episode_steps = 0

            for _ in range(max_steps_per_episode):
                var action = self.select_action(obs, greedy)

                var result = env.step_obs(action)
                var done = result[2]

                if render:
                    env.render_frame()
                    env.renderer_delay(frame_delay_ms)
                    if env.check_renderer_quit():
                        quit_requested = True
                        break

                episode_reward += Float64(result[1])
                fill_inline(result[0], obs)
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

        if render:
            env.close_renderer()

        return total_reward / Float64(num_episodes)

    # =========================================================================
    # GPU Training Helpers
    # =========================================================================

    fn _select_actions_gpu[N: Int](
        self,
        ctx: DeviceContext,
        mut obs_buf: DeviceBuffer[dtype],
        mut q_buf: DeviceBuffer[dtype],
        mut actions_buf: DeviceBuffer[dtype],
        mut gpu_online: GPUNetworkState[Self.Q_Model, Adam[Self.lr]],
        workspace_buf: DeviceBuffer[dtype],
        rng_seed: UInt32,
    ) raises:
        """Select actions for N environments using a GPU forward pass.

        Args:
            ctx: GPU device context.
            obs_buf: Observations [N * obs_dim].
            q_buf: Q-value output buffer [N * num_actions].
            actions_buf: Action output buffer [N].
            gpu_online: Online network GPU state (params read-only).
            workspace_buf: Pre-allocated workspace [N * WORKSPACE_SIZE_PER_SAMPLE].
            rng_seed: RNG seed (should vary per call).
        """
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(N, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var q_t = LayoutTensor[
            dtype, Layout.row_major(N, Self.ACTIONS), MutAnyOrigin
        ](q_buf.unsafe_ptr())
        var p = gpu_online.params_view()
        Self.Q_Network.forward_gpu[N](ctx, obs_t, q_t, p, workspace_buf)

        var actions_t = LayoutTensor[
            dtype, Layout.row_major(N), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var seed_s = Scalar[DType.uint32](rng_seed)
        var epsilon_s = Scalar[dtype](self.epsilon)

        @always_inline
        fn argmax_wrapper(
            eps: Scalar[dtype],
            q_vals: LayoutTensor[
                dtype, Layout.row_major(N, Self.ACTIONS), MutAnyOrigin
            ],
            acts: LayoutTensor[dtype, Layout.row_major(N), MutAnyOrigin],
            base_seed: Scalar[DType.uint32],
        ):
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= N:
                return

            var rng = xorshift32(
                Scalar[DType.uint32](b * 2654435761) + base_seed
            )
            var rand_result = random_uniform[dtype](rng)
            var rand_val = rand_result[0]
            rng = rand_result[1]

            if rand_val < eps:
                var action_result = random_uniform[dtype](rng)
                acts[b] = Scalar[dtype](
                    Int(action_result[0] * Scalar[dtype](Self.ACTIONS))
                )
                return

            var best_q = q_vals[b, 0]
            var best_action = 0
            for a in range(1, Self.ACTIONS):
                var qv = q_vals[b, a]
                if qv > best_q:
                    best_q = qv
                    best_action = a

            acts[b] = Scalar[dtype](best_action)

        ctx.enqueue_function[argmax_wrapper, argmax_wrapper](
            epsilon_s,
            q_t,
            actions_t,
            seed_s,
            grid_dim=((N + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )

    fn _train_step_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_online: GPUNetworkState[Self.Q_Model, Adam[Self.lr]],
        mut gpu_target: GPUNetworkState[Self.Q_Model, Adam[Self.lr]],
        sampled_obs_buf: DeviceBuffer[dtype],
        sampled_next_obs_buf: DeviceBuffer[dtype],
        sampled_actions_buf: DeviceBuffer[dtype],
        sampled_rewards_buf: DeviceBuffer[dtype],
        sampled_dones_buf: DeviceBuffer[dtype],
        mut q_values_buf: DeviceBuffer[dtype],
        mut next_q_values_buf: DeviceBuffer[dtype],
        mut online_next_q_buf: DeviceBuffer[dtype],
        mut cache_buf: DeviceBuffer[dtype],
        mut grad_output_buf: DeviceBuffer[dtype],
        mut grad_input_buf: DeviceBuffer[dtype],
        mut targets_buf: DeviceBuffer[dtype],
        workspace_buf: DeviceBuffer[dtype],
    ) raises:
        """One GPU training step: forward → TD targets → grad → backward → update → soft-update.

        Args:
            ctx: GPU device context.
            gpu_online: Online network (params updated in-place).
            gpu_target: Target network (soft-updated in-place).
            sampled_obs_buf: Sampled observations [batch_size * obs_dim].
            sampled_next_obs_buf: Sampled next observations [batch_size * obs_dim].
            sampled_actions_buf: Sampled actions [batch_size].
            sampled_rewards_buf: Sampled rewards [batch_size].
            sampled_dones_buf: Sampled done flags [batch_size].
            q_values_buf: Q-value scratch [batch_size * num_actions].
            next_q_values_buf: Next Q-value scratch [batch_size * num_actions].
            online_next_q_buf: Online next-state Q scratch [batch_size * num_actions].
            cache_buf: Forward-pass cache [batch_size * CACHE_SIZE].
            grad_output_buf: Gradient scratch [batch_size * num_actions].
            grad_input_buf: Input gradient scratch [batch_size * obs_dim].
            targets_buf: TD target scratch [batch_size].
            workspace_buf: Network workspace [batch_size * WORKSPACE_SIZE_PER_SAMPLE].
        """
        comptime BATCH = Self.batch_size
        comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB

        # LayoutTensor views for sampled batch
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin
        ](sampled_obs_buf.unsafe_ptr())
        var next_obs_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin
        ](sampled_next_obs_buf.unsafe_ptr())
        var q_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
        ](q_values_buf.unsafe_ptr())
        var next_q_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
        ](next_q_values_buf.unsafe_ptr())
        var cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.Q_Model.CACHE_SIZE),
            MutAnyOrigin,
        ](cache_buf.unsafe_ptr())
        var targets_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](targets_buf.unsafe_ptr())
        var rewards_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](sampled_rewards_buf.unsafe_ptr())
        var dones_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](sampled_dones_buf.unsafe_ptr())
        var actions_t = LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ](sampled_actions_buf.unsafe_ptr())
        var grad_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
        ](grad_output_buf.unsafe_ptr())
        var grad_in_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OBS), MutAnyOrigin
        ](grad_input_buf.unsafe_ptr())

        # Online forward with cache
        var p_online = gpu_online.params_view()
        var p_target = gpu_target.params_view()
        Self.Q_Network.forward_gpu_with_cache[BATCH](
            ctx, obs_t, q_t, p_online, cache_t, workspace_buf
        )

        # Target forward (no cache)
        Self.Q_Network.forward_gpu[BATCH](
            ctx, next_obs_t, next_q_t, p_target, workspace_buf
        )

        var gamma_s = Scalar[dtype](self.gamma)

        comptime if Self.double_dqn:
            var online_next_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
            ](online_next_q_buf.unsafe_ptr())
            Self.Q_Network.forward_gpu[BATCH](
                ctx, next_obs_t, online_next_t, p_online, workspace_buf
            )

            @always_inline
            fn double_td_wrapper(
                tgt: LayoutTensor[
                    dtype, Layout.row_major(BATCH), MutAnyOrigin
                ],
                onq: LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
                ],
                tnq: LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
                ],
                rew: LayoutTensor[
                    dtype, Layout.row_major(BATCH), MutAnyOrigin
                ],
                don: LayoutTensor[
                    dtype, Layout.row_major(BATCH), MutAnyOrigin
                ],
                g: Scalar[dtype],
            ):
                dqn_double_td_target_kernel[dtype, BATCH, Self.ACTIONS](
                    tgt, onq, tnq, rew, don, g
                )

            ctx.enqueue_function[double_td_wrapper, double_td_wrapper](
                targets_t,
                online_next_t,
                next_q_t,
                rewards_t,
                dones_t,
                gamma_s,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
        else:
            @always_inline
            fn td_wrapper(
                tgt: LayoutTensor[
                    dtype, Layout.row_major(BATCH), MutAnyOrigin
                ],
                nq: LayoutTensor[
                    dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
                ],
                rew: LayoutTensor[
                    dtype, Layout.row_major(BATCH), MutAnyOrigin
                ],
                don: LayoutTensor[
                    dtype, Layout.row_major(BATCH), MutAnyOrigin
                ],
                g: Scalar[dtype],
            ):
                dqn_td_target_kernel[dtype, BATCH, Self.ACTIONS](
                    tgt, nq, rew, don, g
                )

            ctx.enqueue_function[td_wrapper, td_wrapper](
                targets_t,
                next_q_t,
                rewards_t,
                dones_t,
                gamma_s,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

        # Gradient kernel (masked MSE grad)
        @always_inline
        fn grad_wrapper(
            grd: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
            ],
            qv: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.ACTIONS), MutAnyOrigin
            ],
            tgt: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
            act: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        ):
            var b = Int(block_dim.x * block_idx.x + thread_idx.x)
            if b >= BATCH:
                return
            var action = Int(act[b])
            var q_pred = qv[b, action]
            var td_error = q_pred - tgt[b]
            for a in range(Self.ACTIONS):
                if a == action:
                    grd[b, a] = (
                        Scalar[dtype](2.0)
                        * td_error
                        / Scalar[dtype](BATCH)
                    )
                else:
                    grd[b, a] = Scalar[dtype](0.0)

        ctx.enqueue_function[grad_wrapper, grad_wrapper](
            grad_t,
            q_t,
            targets_t,
            actions_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # Backward + optimizer step
        var g = gpu_online.grads_view()
        gpu_online.zero_grads(ctx)
        Self.Q_Network.backward_gpu[BATCH](
            ctx, grad_t, grad_in_t, p_online, cache_t, g, workspace_buf
        )
        gpu_online.optimizer_step(ctx)

        # Soft-update target on GPU
        gpu_target.soft_update_from_gpu(gpu_online, self.tau, ctx)

        self.train_step_count += 1

    # =========================================================================
    # GPU Training Loop
    # =========================================================================

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
        """Train the DQN agent on GPU with vectorized environments.

        GPU states and replay buffer are created locally (freed on exit).
        After training, params are synced back to self.online/self.target
        so evaluate() works immediately.

        Args:
            ctx: GPU device context.
            env: Environment (must implement GPUDiscreteEnv).
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode (default: 500).
            warmup_steps: Random steps to fill replay buffer (default: 1000).
            train_every: Train every N iterations (default: 4).
            sync_every: Sync GPU→CPU every N episodes for backup (default: 5).
            verbose: Print progress (default: False).
            print_every: Print every N episodes if verbose (default: 10).
            environment_name: Name for metrics labeling.

        Returns:
            TrainingMetrics with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name=(
                "DQN (GPU)" if not Self.double_dqn else "Double DQN (GPU)"
            ),
            environment_name=environment_name,
        )

        # =====================================================================
        # Create GPU network states from current CPU params
        # =====================================================================
        var gpu_online = GPUNetworkState[Self.Q_Model, Adam[Self.lr]](ctx)
        var gpu_target = GPUNetworkState[Self.Q_Model, Adam[Self.lr]](ctx)
        gpu_online.upload_from(self.online, ctx)
        gpu_target.upload_from(self.target, ctx)

        # =====================================================================
        # GPU Replay Buffer (local — freed on function exit)
        # =====================================================================
        var rb = GPUReplayBuffer[Self.buffer_capacity, Self.obs_dim](ctx)
        var indices_buf = ctx.enqueue_create_buffer[DType.int32](Self.batch_size)

        # =====================================================================
        # Pre-allocate environment and training buffers
        # =====================================================================
        comptime ENV_OBS_SIZE = Self.n_envs * Self.obs_dim
        comptime ENV_Q_SIZE = Self.n_envs * Self.num_actions
        comptime BATCH_OBS_SIZE = Self.batch_size * Self.obs_dim
        comptime BATCH_Q_SIZE = Self.batch_size * Self.num_actions
        comptime BATCH_CACHE_SIZE = Self.batch_size * Self.Q_Model.CACHE_SIZE
        comptime WS_PER_SAMPLE = Self.Q_Network.WORKSPACE_SIZE_PER_SAMPLE
        comptime ENV_WS_SIZE = Self.n_envs * WS_PER_SAMPLE
        comptime BATCH_WS_SIZE = Self.batch_size * WS_PER_SAMPLE

        # Environment buffers
        var prev_obs_buf = ctx.enqueue_create_buffer[dtype](ENV_OBS_SIZE)
        var obs_buf = ctx.enqueue_create_buffer[dtype](ENV_OBS_SIZE)
        var state_buf = ctx.enqueue_create_buffer[dtype](
            Self.n_envs * EnvType.STATE_SIZE
        )
        var env_q_buf = ctx.enqueue_create_buffer[dtype](ENV_Q_SIZE)
        var rewards_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var dones_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var actions_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)

        # Training buffers
        var q_values_buf = ctx.enqueue_create_buffer[dtype](BATCH_Q_SIZE)
        var next_q_values_buf = ctx.enqueue_create_buffer[dtype](BATCH_Q_SIZE)
        var online_next_q_buf = ctx.enqueue_create_buffer[dtype](BATCH_Q_SIZE)
        var cache_buf = ctx.enqueue_create_buffer[dtype](BATCH_CACHE_SIZE)
        var grad_output_buf = ctx.enqueue_create_buffer[dtype](BATCH_Q_SIZE)
        var grad_input_buf = ctx.enqueue_create_buffer[dtype](BATCH_OBS_SIZE)
        var targets_buf = ctx.enqueue_create_buffer[dtype](Self.batch_size)

        # Sampled batch buffers (output of GPUReplayBuffer.sample)
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
        var sampled_dones_buf = ctx.enqueue_create_buffer[dtype](Self.batch_size)

        # Workspace buffers
        var env_ws_buf = ctx.enqueue_create_buffer[dtype](ENV_WS_SIZE)
        var batch_ws_buf = ctx.enqueue_create_buffer[dtype](BATCH_WS_SIZE)

        # Host buffers for CPU↔GPU data transfer
        var obs_host = ctx.enqueue_create_host_buffer[dtype](ENV_OBS_SIZE)
        var rewards_host = ctx.enqueue_create_host_buffer[dtype](Self.n_envs)
        var dones_host = ctx.enqueue_create_host_buffer[dtype](Self.n_envs)
        var actions_host = ctx.enqueue_create_host_buffer[dtype](Self.n_envs)

        # Episode tracking buffers (GPU)
        var episode_rewards_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var episode_steps_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var completed_rewards_buf = ctx.enqueue_create_buffer[dtype](
            Self.n_envs
        )
        var completed_steps_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var completed_mask_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)

        # Host mirrors for episode tracking read-back
        var completed_rewards_host = ctx.enqueue_create_host_buffer[dtype](
            Self.n_envs
        )
        var completed_steps_host = ctx.enqueue_create_host_buffer[dtype](
            Self.n_envs
        )
        var completed_mask_host = ctx.enqueue_create_host_buffer[dtype](
            Self.n_envs
        )

        # =====================================================================
        # LayoutTensor views (created once — zero-copy pointer wrappers)
        # =====================================================================
        comptime ENV_BLOCKS = (Self.n_envs + TPB - 1) // TPB
        comptime BATCH_BLOCKS = (Self.batch_size + TPB - 1) // TPB
        comptime OBS_BLOCKS = (ENV_OBS_SIZE + TPB - 1) // TPB

        var episode_rewards_t = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](episode_rewards_buf.unsafe_ptr())
        var episode_steps_t = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](episode_steps_buf.unsafe_ptr())
        var completed_rewards_t = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](completed_rewards_buf.unsafe_ptr())
        var completed_steps_t = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](completed_steps_buf.unsafe_ptr())
        var completed_mask_t = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](completed_mask_buf.unsafe_ptr())
        var rewards_t = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var dones_t = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](dones_buf.unsafe_ptr())
        var prev_obs_t = LayoutTensor[
            dtype, Layout.row_major(ENV_OBS_SIZE), MutAnyOrigin
        ](prev_obs_buf.unsafe_ptr())
        var obs_flat_t = LayoutTensor[
            dtype, Layout.row_major(ENV_OBS_SIZE), MutAnyOrigin
        ](obs_buf.unsafe_ptr())

        # =====================================================================
        # Kernel wrappers (defined once outside the loop)
        # =====================================================================
        @always_inline
        fn zero_envs_wrapper(
            buf: LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ],
        ):
            zero_buffer_kernel[dtype, Self.n_envs](buf)

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

        @always_inline
        fn accum_rewards_wrapper(
            ep_r: LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ],
            step_r: LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ],
        ):
            accumulate_rewards_kernel[dtype, Self.n_envs](ep_r, step_r)

        @always_inline
        fn incr_steps_wrapper(
            ep_s: LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ],
        ):
            increment_steps_kernel[dtype, Self.n_envs](ep_s)

        @always_inline
        fn extract_wrapper(
            d: LayoutTensor[
                dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
            ],
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

        # =====================================================================
        # Initialization
        # =====================================================================
        ctx.enqueue_function[zero_envs_wrapper, zero_envs_wrapper](
            episode_rewards_t,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )
        ctx.enqueue_function[zero_envs_wrapper, zero_envs_wrapper](
            episode_steps_t,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        EnvType.reset_kernel_gpu[Self.n_envs, Self.obs_dim](ctx, obs_buf)

        if verbose:
            print("GPU buffers allocated. Starting training...")
            print("Running " + String(Self.n_envs) + " parallel environments")
            print("Training batch size: " + String(Self.batch_size))

        # =====================================================================
        # Warmup: fill replay buffer with random transitions
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
            ctx.enqueue_function[copy_obs_wrapper, copy_obs_wrapper](
                prev_obs_t,
                obs_flat_t,
                grid_dim=(OBS_BLOCKS,),
                block_dim=(TPB,),
            )

            self._select_actions_gpu[Self.n_envs](
                ctx,
                obs_buf,
                env_q_buf,
                actions_buf,
                gpu_online,
                env_ws_buf,
                UInt32(warmup_count * 7919 + 12345),
            )

            EnvType.step_kernel_gpu[
                Self.n_envs, EnvType.STATE_SIZE, Self.obs_dim
            ](ctx, state_buf, actions_buf, rewards_buf, dones_buf, obs_buf)

            rb.store[Self.n_envs](
                ctx,
                prev_obs_buf,
                actions_buf,
                rewards_buf,
                obs_buf,
                dones_buf,
            )

            warmup_count += Self.n_envs

            var rng_seed = UInt64(warmup_count * 7919 + 42)
            EnvType.selective_reset_kernel_gpu[Self.n_envs, Self.obs_dim](
                ctx, obs_buf, dones_buf, rng_seed
            )

        self.epsilon = saved_epsilon

        # Reset all envs after warmup (start fresh episodes)
        EnvType.reset_kernel_gpu[Self.n_envs, Self.obs_dim](
            ctx, obs_buf, UInt64(warmup_count)
        )

        if verbose:
            print("Warmup complete. Replay buffer size: " + String(rb.size))

        # =====================================================================
        # Timing counters
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
        var total_steps = 0
        var completed_episodes = 0
        var last_print_episode = 0
        var last_checkpoint_episode = 0

        while completed_episodes < num_episodes:
            var t0 = perf_counter_ns()

            # Copy current obs to prev_obs
            ctx.enqueue_function[copy_obs_wrapper, copy_obs_wrapper](
                prev_obs_t,
                obs_flat_t,
                grid_dim=(OBS_BLOCKS,),
                block_dim=(TPB,),
            )

            # Select actions
            self._select_actions_gpu[Self.n_envs](
                ctx,
                obs_buf,
                env_q_buf,
                actions_buf,
                gpu_online,
                env_ws_buf,
                UInt32(
                    total_steps * 2654435761 + iteration_count * 7919
                ),
            )
            ctx.synchronize()
            var t1 = perf_counter_ns()
            time_action_select += t1 - t0

            # Step environments (obs_buf now contains next_obs)
            EnvType.step_kernel_gpu[
                Self.n_envs, EnvType.STATE_SIZE, Self.obs_dim
            ](ctx, state_buf, actions_buf, rewards_buf, dones_buf, obs_buf)
            ctx.synchronize()
            var t2 = perf_counter_ns()
            time_env_step += t2 - t1

            # Accumulate rewards / steps
            ctx.enqueue_function[accum_rewards_wrapper, accum_rewards_wrapper](
                episode_rewards_t,
                rewards_t,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
            ctx.enqueue_function[incr_steps_wrapper, incr_steps_wrapper](
                episode_steps_t,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )

            total_steps += Self.n_envs

            # Store transitions (prev_obs → obs = next_obs)
            rb.store[Self.n_envs](
                ctx,
                prev_obs_buf,
                actions_buf,
                rewards_buf,
                obs_buf,
                dones_buf,
            )

            ctx.synchronize()
            var t3 = perf_counter_ns()
            time_store += t3 - t2

            var t4 = t3

            # Training
            if (
                rb.is_ready[Self.batch_size]()
                and iteration_count % train_every == 0
            ):
                var num_train_steps = max(
                    1, Self.n_envs // (Self.batch_size * train_every)
                )

                for train_idx in range(num_train_steps):
                    var raw_seed = UInt32(
                        total_steps * 2654435761
                        + train_idx * 1013904223
                        + iteration_count * 7919
                    )
                    var rng_seed = UInt32(
                        (raw_seed ^ (raw_seed >> 16)) * 2246822519
                    )

                    rb.sample[Self.batch_size](
                        ctx,
                        rng_seed,
                        sampled_obs_buf,
                        sampled_actions_buf,
                        sampled_rewards_buf,
                        sampled_next_obs_buf,
                        sampled_dones_buf,
                        indices_buf,
                    )

                    self._train_step_gpu(
                        ctx,
                        gpu_online,
                        gpu_target,
                        sampled_obs_buf,
                        sampled_next_obs_buf,
                        sampled_actions_buf,
                        sampled_rewards_buf,
                        sampled_dones_buf,
                        q_values_buf,
                        next_q_values_buf,
                        online_next_q_buf,
                        cache_buf,
                        grad_output_buf,
                        grad_input_buf,
                        targets_buf,
                        batch_ws_buf,
                    )

                ctx.synchronize()
                t4 = perf_counter_ns()
                time_train += t4 - t3

            # Extract completed episodes and reset done envs
            ctx.enqueue_function[extract_wrapper, extract_wrapper](
                dones_t,
                episode_rewards_t,
                episode_steps_t,
                completed_rewards_t,
                completed_steps_t,
                completed_mask_t,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )

            ctx.enqueue_copy(completed_rewards_host, completed_rewards_buf)
            ctx.enqueue_copy(completed_steps_host, completed_steps_buf)
            ctx.enqueue_copy(completed_mask_host, completed_mask_buf)
            ctx.synchronize()

            var any_done = False
            for i in range(Self.n_envs):
                if completed_mask_host[i] > 0.5:
                    any_done = True
                    var ep_reward = Float64(completed_rewards_host[i])
                    var ep_steps = Int(completed_steps_host[i])
                    metrics.log_episode(
                        completed_episodes, ep_reward, ep_steps, self.epsilon
                    )
                    completed_episodes += 1
                    self.decay_epsilon()

            if any_done:
                EnvType.selective_reset_kernel_gpu[Self.n_envs, Self.obs_dim](
                    ctx,
                    obs_buf,
                    dones_buf,
                    UInt64(total_steps * 7919 + 42),
                )

            var t5 = perf_counter_ns()
            time_episode_track += t5 - t4
            iteration_count += 1

            # Periodic CPU sync (cheap: online only)
            if (
                completed_episodes > 0
                and (completed_episodes % sync_every == 0)
                and completed_episodes != last_print_episode
            ):
                gpu_online.download_to(self.online, ctx)

            # Print progress
            if verbose and completed_episodes > 0:
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

            # Auto-checkpoint
            if (
                self.checkpoint_every > 0
                and len(self.checkpoint_path) > 0
                and completed_episodes > 0
            ):
                var next_ckpt = (
                    (last_checkpoint_episode // self.checkpoint_every) + 1
                ) * self.checkpoint_every
                if completed_episodes >= next_ckpt:
                    last_checkpoint_episode = completed_episodes
                    gpu_online.download_to(self.online, ctx)
                    gpu_target.download_to(self.target, ctx)
                    try:
                        self.save_checkpoint(self.checkpoint_path)
                        if verbose:
                            print(
                                "Checkpoint saved at episode "
                                + String(completed_episodes)
                            )
                    except:
                        print(
                            "Warning: Failed to save checkpoint at episode "
                            + String(completed_episodes)
                        )

        # Final sync → evaluate() works immediately after train_gpu()
        gpu_online.download_to(self.online, ctx)
        gpu_target.download_to(self.target, ctx)

        # Timing summary
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
    # Checkpoint Save / Load
    # =========================================================================

    fn save_checkpoint(self, filepath: String) raises:
        """Save DQN agent state to a checkpoint file.

        Saves online and target network params + optimizer states,
        plus epsilon and training counters.  Replay buffer is NOT saved.

        Args:
            filepath: Destination path (e.g. "dqn_agent.ckpt").
        """
        comptime PARAM_SIZE = Self.Q_Network.PARAM_SIZE
        comptime STATE_SIZE = PARAM_SIZE * Adam[Self.lr].STATE_PER_PARAM

        var content = write_checkpoint_header(
            "dqn_agent", PARAM_SIZE, STATE_SIZE
        )
        content += self.online.write_sections("online_")
        content += self.target.write_sections("target_")

        var metadata = List[String]()
        metadata.append("gamma=" + String(self.gamma))
        metadata.append("tau=" + String(self.tau))
        metadata.append("lr=" + String(Self.lr))
        metadata.append("epsilon=" + String(self.epsilon))
        metadata.append("epsilon_min=" + String(self.epsilon_min))
        metadata.append("epsilon_decay=" + String(self.epsilon_decay))
        metadata.append("train_step_count=" + String(self.train_step_count))
        content += write_metadata_section(metadata)

        save_checkpoint_file(filepath, content)

    fn load_checkpoint(mut self, filepath: String) raises:
        """Load DQN agent state from a checkpoint file.

        Args:
            filepath: Path to the checkpoint file.
        """
        var content = read_checkpoint_file(filepath)
        var header = parse_checkpoint_header(content)

        comptime PARAM_SIZE = Self.Q_Network.PARAM_SIZE
        if header.param_size != PARAM_SIZE:
            print(
                "Warning: checkpoint param_size ("
                + String(header.param_size)
                + ") != PARAM_SIZE ("
                + String(PARAM_SIZE)
                + ")"
            )

        self.online.read_sections(content, "online_")
        self.target.read_sections(content, "target_")

        var metadata = read_metadata_section(content)

        var gamma_str = get_metadata_value(metadata, "gamma")
        if len(gamma_str) > 0:
            self.gamma = atof(gamma_str)

        var tau_str = get_metadata_value(metadata, "tau")
        if len(tau_str) > 0:
            self.tau = atof(tau_str)

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

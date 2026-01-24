"""Deep PPO (Proximal Policy Optimization) Agent for Continuous Action Spaces.

This PPO implementation supports continuous action spaces using a Gaussian policy:
- Network wrapper from deep_rl.training for stateless model + params management
- seq() composition for building actor and critic networks
- StochasticActor for Gaussian policy with reparameterization trick
- Clipped surrogate objective for stable policy updates
- GAE (Generalized Advantage Estimation) for variance reduction

Key features:
- Works with any BoxContinuousActionEnv (continuous obs, continuous actions)
- Unbounded Gaussian policy (CleanRL-style) - actions clipped at environment boundary
- Clipped policy ratio for stable updates
- Multiple epochs of optimization per rollout
- Entropy bonus for exploration
- Advantage normalization

Architecture:
- Actor: obs -> hidden (ReLU) -> hidden (ReLU) -> StochasticActor (mean + log_std)
- Critic: obs -> hidden (ReLU) -> hidden (ReLU) -> 1 (value)

Usage:
    from deep_agents.ppo_continuous import DeepPPOContinuousAgent
    from envs import CarRacingEnv

    var env = CarRacingEnv(continuous=True)
    var agent = DeepPPOContinuousAgent[13, 3, 256]()

    # Hybrid GPU+CPU training
    with DeviceContext() as ctx:
        var metrics = agent.train_gpu_cpu_env(ctx, envs, num_episodes=1000)

Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
"""

from math import exp, log, sqrt, cos, tanh
from random import random_float64, seed
from time import perf_counter_ns
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor

from deep_rl.constants import dtype, TILE, TPB
from deep_rl.model import Linear, ReLU, LinearReLU, seq, StochasticActor
from deep_rl.optimizer import Adam
from deep_rl.initializer import Xavier, Kaiming
from deep_rl.training import Network
from deep_rl.checkpoint import (
    split_lines,
    find_section_start,
    save_checkpoint_file,
    read_checkpoint_file,
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
)
from deep_rl.gpu.random import gaussian_noise
from core import TrainingMetrics, BoxContinuousActionEnv, GPUContinuousEnv
from render import RendererBase
from memory import UnsafePointer
from core.utils.gae import compute_gae_inline
from core.utils.normalization import normalize_inline, RunningMeanStd
from core.utils.shuffle import shuffle_indices_inline
from .kernels import (
    _sample_continuous_actions_kernel,
    _store_continuous_pre_step_kernel,
    gradient_norm_kernel,
    gradient_clip_kernel,
    gradient_reduce_and_compute_scale_kernel,
    gradient_apply_scale_kernel,
    gradient_reduce_apply_fused_kernel,
    ppo_continuous_gather_minibatch_kernel,
    ppo_continuous_actor_grad_kernel,
    normalize_advantages_fused_kernel,
    ppo_critic_grad_kernel,
    ppo_critic_grad_clipped_kernel,
    normalize_advantages_kernel,
    _extract_obs_from_state_continuous_kernel,
    _store_post_step_kernel,
    clamp_log_std_params_kernel,
)

# =============================================================================
# Deep PPO Continuous Agent
# =============================================================================


struct DeepPPOContinuousAgent[
    obs_dim: Int,
    action_dim: Int,
    hidden_dim: Int = 256,
    rollout_len: Int = 128,
    n_envs: Int = 64,
    gpu_minibatch_size: Int = 256,
    clip_value: Bool = True,
]:
    """Deep Proximal Policy Optimization Agent for Continuous Action Spaces.

    Uses an unbounded Gaussian policy (CleanRL-style) - actions clipped at env boundary.
    Supports hybrid GPU+CPU training where neural networks run on GPU and
    environment physics (like CarRacing) run on CPU.

    Parameters:
        obs_dim: Dimension of observation space.
        action_dim: Dimension of continuous action space.
        hidden_dim: Hidden layer size (default: 256).
        rollout_len: Steps per rollout per environment (default: 128).
        n_envs: Number of parallel environments for training (default: 64).
        gpu_minibatch_size: Minibatch size for GPU training (default: 256).
        clip_value: Whether to clip value function updates (default: True).

    Note on hybrid training:
        - Neural network computations (forward/backward) run on GPU
        - Environment physics (e.g., CarRacing) run on CPU
        - This allows accurate physics while leveraging GPU acceleration
    """

    # Convenience aliases
    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.action_dim
    comptime HIDDEN = Self.hidden_dim
    comptime ROLLOUT = Self.rollout_len

    # Actor output: mean + log_std = 2 * action_dim
    comptime ACTOR_OUT = Self.action_dim * 2

    # Cache sizes
    # Actor: Linear[obs, h] + ReLU[h] + Linear[h, h] + ReLU[h] + StochasticActor[h, action]
    comptime ACTOR_CACHE: Int = Self.OBS + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN
    # Critic: Linear[obs, h] + ReLU[h] + Linear[h, h] + ReLU[h] + Linear[h, 1]
    comptime CRITIC_CACHE: Int = Self.OBS + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN

    # Network parameter sizes
    # Actor: Linear[obs, hidden] + ReLU + Linear[hidden, hidden] + ReLU + StochasticActor[hidden, action]
    # StochasticActor params: (hidden * action + action) for mean head + action for state-independent log_std
    comptime ACTOR_PARAM_SIZE: Int = (
        Self.OBS * Self.HIDDEN
        + Self.HIDDEN  # Linear 1
        + Self.HIDDEN * Self.HIDDEN
        + Self.HIDDEN  # Linear 2
        + (
            Self.HIDDEN * Self.ACTIONS + Self.ACTIONS + Self.ACTIONS
        )  # StochasticActor
    )
    # Critic: Linear[obs, hidden] + ReLU + Linear[hidden, hidden] + ReLU + Linear[hidden, 1]
    comptime CRITIC_PARAM_SIZE: Int = (
        Self.OBS * Self.HIDDEN
        + Self.HIDDEN  # Linear 1
        + Self.HIDDEN * Self.HIDDEN
        + Self.HIDDEN  # Linear 2
        + Self.HIDDEN * 1
        + 1  # Linear 3
    )

    # GPU-specific sizes
    comptime TOTAL_ROLLOUT_SIZE: Int = Self.n_envs * Self.rollout_len
    comptime GPU_MINIBATCH = Self.gpu_minibatch_size

    # Actor network: obs -> hidden (ReLU) -> hidden (ReLU) -> StochasticActor (mean, log_std)
    var actor: Network[
        type_of(
            seq(
                LinearReLU[Self.OBS, Self.HIDDEN](),
                LinearReLU[Self.HIDDEN, Self.HIDDEN](),
                StochasticActor[Self.HIDDEN, Self.ACTIONS](),
            )
        ),
        Adam,
        Kaiming,
    ]

    # Critic network: obs -> hidden (ReLU) -> hidden (ReLU) -> value
    var critic: Network[
        type_of(
            seq(
                LinearReLU[Self.OBS, Self.HIDDEN](),
                LinearReLU[Self.HIDDEN, Self.HIDDEN](),
                Linear[Self.HIDDEN, 1](),
            )
        ),
        Adam,
        Kaiming,
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
    var normalize_advantages: Bool

    # Advanced hyperparameters
    var target_kl: Float64
    var max_grad_norm: Float64
    var anneal_lr: Bool
    var anneal_entropy: Bool
    var target_total_steps: Int

    var norm_adv_per_minibatch: Bool

    # Action scaling (for environments with action bounds other than [-1, 1])
    var action_scale: Float64
    var action_bias: Float64

    # Training state
    var train_step_count: Int

    # Auto-checkpoint settings
    var checkpoint_every: Int
    var checkpoint_path: String

    # Reward normalization (CleanRL-style)
    var normalize_rewards: Bool
    var reward_rms: RunningMeanStd

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        gae_lambda: Float64 = 0.95,
        clip_epsilon: Float64 = 0.2,
        actor_lr: Float64 = 0.0003,
        critic_lr: Float64 = 0.001,
        entropy_coef: Float64 = 0.01,
        value_loss_coef: Float64 = 0.5,
        num_epochs: Int = 10,
        normalize_advantages: Bool = True,
        # Advanced hyperparameters
        target_kl: Float64 = 0.02,
        max_grad_norm: Float64 = 0.5,
        anneal_lr: Bool = True,
        anneal_entropy: Bool = False,
        target_total_steps: Int = 0,
        norm_adv_per_minibatch: Bool = True,
        # Action scaling
        action_scale: Float64 = 1.0,
        action_bias: Float64 = 0.0,
        # Checkpoint settings
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
        # Reward normalization (CleanRL-style)
        normalize_rewards: Bool = True,
        # Per-action mean biases for policy initialization (optional)
        # Use this for environments where default action != 0
        # e.g., CarRacing: [0, 2.0, -2.0] for steering=0, gas=high, brake=low
        action_mean_biases: List[Float64] = List[Float64](),
    ):
        """Initialize Deep PPO Continuous agent.

        Args:
            gamma: Discount factor (default: 0.99).
            gae_lambda: GAE lambda parameter (default: 0.95).
            clip_epsilon: PPO clipping parameter (default: 0.2).
            actor_lr: Actor learning rate (default: 0.0003).
            critic_lr: Critic learning rate (default: 0.001).
            entropy_coef: Entropy bonus coefficient (default: 0.01).
            value_loss_coef: Value loss coefficient (default: 0.5).
            num_epochs: Number of optimization epochs per update (default: 10).
            normalize_advantages: Whether to normalize advantages (default: True).
            target_kl: KL threshold for early stopping (default: 0.02).
            max_grad_norm: Gradient clipping threshold (default: 0.5).
            anneal_lr: Whether to linearly anneal learning rate (default: True).
            anneal_entropy: Whether to anneal entropy coefficient (default: False).
            target_total_steps: Target steps for annealing (0 = auto).
            norm_adv_per_minibatch: Normalize advantages per minibatch (default: True).
            action_scale: Scale for actions (default: 1.0).
            action_bias: Bias for actions (default: 0.0).
            checkpoint_every: Save checkpoint every N episodes (0 to disable).
            checkpoint_path: Path for auto-checkpointing.
        """
        self.actor = Network[
            type_of(
                seq(
                    LinearReLU[Self.OBS, Self.HIDDEN](),
                    LinearReLU[Self.HIDDEN, Self.HIDDEN](),
                    StochasticActor[Self.HIDDEN, Self.ACTIONS](),
                )
            ),
            Adam,
            Kaiming,
        ](
            seq(
                LinearReLU[Self.OBS, Self.HIDDEN](),
                LinearReLU[Self.HIDDEN, Self.HIDDEN](),
                StochasticActor[Self.HIDDEN, Self.ACTIONS](),
            ),
            Adam(lr=actor_lr),
            Kaiming(),
        )

        # Re-initialize StochasticActor with small weights for stable RL training
        # This is crucial: Kaiming init produces large initial means which breaks training
        comptime STOCHASTIC_ACTOR_OFFSET = (
            Self.OBS * Self.HIDDEN
            + Self.HIDDEN
            + Self.HIDDEN * Self.HIDDEN  # Linear 1
            + Self.HIDDEN  # Linear 2
        )
        # State-independent log_std: W_mean + b_mean + log_std
        comptime STOCHASTIC_ACTOR_SIZE = (
            Self.HIDDEN * Self.ACTIONS + Self.ACTIONS + Self.ACTIONS
        )
        var stochastic_actor_params = LayoutTensor[
            dtype, Layout.row_major(STOCHASTIC_ACTOR_SIZE), MutAnyOrigin
        ](self.actor.params.unsafe_ptr() + STOCHASTIC_ACTOR_OFFSET)

        # Use per-action mean biases if provided, otherwise use centered initialization
        if len(action_mean_biases) > 0:
            StochasticActor[
                Self.HIDDEN, Self.ACTIONS
            ].init_params_with_mean_bias(
                stochastic_actor_params,
                action_mean_biases,
                weight_scale=0.01,  # Small weights for stable learning
                log_std_init=-0.5,  # std ≈ 0.6 for exploration
            )
        else:
            StochasticActor[Self.HIDDEN, Self.ACTIONS].init_params_small(
                stochastic_actor_params,
                weight_scale=0.01,  # Small weights -> initial mean ≈ 0
                log_std_init=-0.5,  # std ≈ 0.6 for moderate exploration
            )

        self.critic = Network[
            type_of(
                seq(
                    LinearReLU[Self.OBS, Self.HIDDEN](),
                    LinearReLU[Self.HIDDEN, Self.HIDDEN](),
                    Linear[Self.HIDDEN, 1](),
                )
            ),
            Adam,
            Kaiming,
        ](
            seq(
                LinearReLU[Self.OBS, Self.HIDDEN](),
                LinearReLU[Self.HIDDEN, Self.HIDDEN](),
                Linear[Self.HIDDEN, 1](),
            ),
            Adam(lr=critic_lr),
            Kaiming(),
        )

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.num_epochs = num_epochs
        self.normalize_advantages = normalize_advantages

        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm
        self.anneal_lr = anneal_lr
        self.anneal_entropy = anneal_entropy
        self.target_total_steps = target_total_steps
        self.norm_adv_per_minibatch = norm_adv_per_minibatch

        self.action_scale = action_scale
        self.action_bias = action_bias

        self.train_step_count = 0

        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path

        self.normalize_rewards = normalize_rewards
        self.reward_rms = RunningMeanStd()

    # =========================================================================
    # Action Selection (for evaluation)
    # =========================================================================

    fn select_action(
        self,
        obs: InlineArray[Scalar[dtype], Self.OBS],
        training: Bool = True,
    ) -> Tuple[
        InlineArray[Scalar[dtype], Self.ACTIONS], Scalar[dtype], Scalar[dtype]
    ]:
        """Select continuous action from unbounded Gaussian policy (CleanRL-style).

        Args:
            obs: Current observation.
            training: If True, sample from Gaussian; else use mean (deterministic).

        Returns:
            Tuple of (actions, log_prob, value) where actions are unbounded
            (clipping to env bounds is done at environment step).
        """
        # Forward actor to get mean and log_std
        # StochasticActor outputs: [mean_0, ..., mean_n, log_std_0, ..., log_std_n]
        var actor_output = InlineArray[Scalar[dtype], Self.ACTOR_OUT](
            uninitialized=True
        )
        self.actor.forward[1](obs, actor_output)

        # Extract means and log_stds
        var means = InlineArray[Scalar[dtype], Self.ACTIONS](uninitialized=True)
        var log_stds = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        for j in range(Self.ACTIONS):
            means[j] = actor_output[j]
            log_stds[j] = actor_output[Self.ACTIONS + j]

        # Forward critic to get value
        var value_out = InlineArray[Scalar[dtype], 1](uninitialized=True)
        self.critic.forward[1](obs, value_out)
        var value = value_out[0]

        # Compute actions (unbounded Gaussian, no tanh squashing)
        var actions = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        var total_log_prob = Scalar[dtype](0.0)

        # Log_std bounds (must match GPU kernel)
        comptime LOG_STD_MIN: Scalar[dtype] = -5.0
        comptime LOG_STD_MAX: Scalar[dtype] = 2.0

        for j in range(Self.ACTIONS):
            var mean = means[j]
            var log_std = log_stds[j]
            # Clamp log_std to match GPU kernel
            if log_std < LOG_STD_MIN:
                log_std = LOG_STD_MIN
            elif log_std > LOG_STD_MAX:
                log_std = LOG_STD_MAX
            var std = exp(log_std)

            var action: Scalar[dtype]
            if training:
                # Sample from Gaussian using Box-Muller transform (same as GPU kernel)
                # Generate two uniform random numbers in (0, 1)
                var u1 = random_float64(0.0, 1.0)
                var u2 = random_float64(0.0, 1.0)
                # Avoid log(0) by ensuring u1 > 0
                if u1 < 1e-10:
                    u1 = 1e-10
                # Box-Muller transform for standard normal
                var mag = sqrt(-2.0 * log(u1))
                var noise = Scalar[dtype](mag * cos(u2 * 6.283185307179586))
                action = mean + std * noise

                # Simple Gaussian log probability (no tanh correction)
                var action_normalized = (action - mean) / (
                    std + Scalar[dtype](1e-8)
                )
                var log_prob_gaussian = (
                    -Scalar[dtype](0.5) * action_normalized * action_normalized
                    - log_std
                    - Scalar[dtype](0.9189385)  # -0.5 * log(2*pi)
                )
                total_log_prob += log_prob_gaussian
            else:
                # Deterministic: use mean
                action = mean

            # Unbounded action (clipping done at environment step)
            actions[j] = action

        return (actions^, total_log_prob, value)

    # =========================================================================
    # Checkpoint Save/Load
    # =========================================================================

    fn save_checkpoint(self, path: String) raises:
        """Save agent state to a checkpoint file."""
        var content = String()
        content += "[AGENT_TYPE]\n"
        content += "DeepPPOContinuousAgent\n"
        content += "[HYPERPARAMETERS]\n"
        content += "gamma=" + String(self.gamma) + "\n"
        content += "gae_lambda=" + String(self.gae_lambda) + "\n"
        content += "clip_epsilon=" + String(self.clip_epsilon) + "\n"
        content += "actor_lr=" + String(self.actor_lr) + "\n"
        content += "critic_lr=" + String(self.critic_lr) + "\n"
        content += "entropy_coef=" + String(self.entropy_coef) + "\n"
        content += "train_step_count=" + String(self.train_step_count) + "\n"

        content += "[ACTOR_PARAMS]\n"
        for i in range(len(self.actor.params)):
            content += String(self.actor.params[i]) + "\n"

        content += "[ACTOR_STATE]\n"
        for i in range(len(self.actor.optimizer_state)):
            content += String(self.actor.optimizer_state[i]) + "\n"

        content += "[CRITIC_PARAMS]\n"
        for i in range(len(self.critic.params)):
            content += String(self.critic.params[i]) + "\n"

        content += "[CRITIC_STATE]\n"
        for i in range(len(self.critic.optimizer_state)):
            content += String(self.critic.optimizer_state[i]) + "\n"

        save_checkpoint_file(path, content)

    fn load_checkpoint(mut self, path: String) raises:
        """Load agent state from a checkpoint file."""
        var content = read_checkpoint_file(path)
        if len(content) == 0:
            print("No checkpoint found at:", path)
            return

        var lines = split_lines(content)

        # Load actor params
        var actor_start = find_section_start(lines, "[ACTOR_PARAMS]")
        if actor_start >= 0:
            var idx = actor_start  # find_section_start already returns line after header
            for i in range(len(self.actor.params)):
                if idx < len(lines) and not lines[idx].startswith("["):
                    try:
                        self.actor.params[i] = Scalar[dtype](
                            Float32(atof(lines[idx]))
                        )
                    except:
                        pass
                    idx += 1

        # Load actor optimizer state
        var actor_state_start = find_section_start(lines, "[ACTOR_STATE]")
        if actor_state_start >= 0:
            var idx = actor_state_start  # find_section_start already returns line after header
            for i in range(len(self.actor.optimizer_state)):
                if idx < len(lines) and not lines[idx].startswith("["):
                    try:
                        self.actor.optimizer_state[i] = Scalar[dtype](
                            Float32(atof(lines[idx]))
                        )
                    except:
                        pass
                    idx += 1

        # Load critic params
        var critic_start = find_section_start(lines, "[CRITIC_PARAMS]")
        if critic_start >= 0:
            var idx = critic_start  # find_section_start already returns line after header
            for i in range(len(self.critic.params)):
                if idx < len(lines) and not lines[idx].startswith("["):
                    try:
                        self.critic.params[i] = Scalar[dtype](
                            Float32(atof(lines[idx]))
                        )
                    except:
                        pass
                    idx += 1

        # Load critic optimizer state
        var critic_state_start = find_section_start(lines, "[CRITIC_STATE]")
        if critic_state_start >= 0:
            var idx = critic_state_start  # find_section_start already returns line after header
            for i in range(len(self.critic.optimizer_state)):
                if idx < len(lines) and not lines[idx].startswith("["):
                    try:
                        self.critic.optimizer_state[i] = Scalar[dtype](
                            Float32(atof(lines[idx]))
                        )
                    except:
                        pass
                    idx += 1

        # Load train step count
        var hyper_start = find_section_start(lines, "[HYPERPARAMETERS]")
        if hyper_start >= 0:
            var idx = hyper_start + 1
            while idx < len(lines) and not lines[idx].startswith("["):
                if lines[idx].startswith("train_step_count="):
                    try:
                        self.train_step_count = Int(
                            atof(lines[idx][len("train_step_count=") :])
                        )
                    except:
                        pass
                idx += 1

        print("Checkpoint loaded from:", path)

    # =========================================================================
    # Evaluation Methods
    # =========================================================================

    fn evaluate[
        E: BoxContinuousActionEnv
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps: Int = 1000,
        verbose: Bool = False,
        stochastic: Bool = True,
        renderer: UnsafePointer[RendererBase, MutAnyOrigin] = UnsafePointer[
            RendererBase, MutAnyOrigin
        ](),
    ) -> Float64:
        """Evaluate the agent (CPU with optional rendering).

        Uses unbounded Gaussian policy (CleanRL-style). Actions are clipped
        to [-1, 1] at the environment boundary.

        Args:
            env: The environment to evaluate on.
            num_episodes: Number of evaluation episodes.
            max_steps: Maximum steps per episode.
            verbose: Whether to print per-episode results.
            stochastic: If True (default), sample from policy; if False, use mean.
            renderer: Optional pointer to renderer for visualization.

        Returns:
            Average reward over evaluation episodes.
        """
        var total_reward: Float64 = 0.0

        for episode in range(num_episodes):
            var obs_list = env.reset_obs_list()
            var obs = InlineArray[Scalar[dtype], Self.OBS](uninitialized=True)
            for i in range(Self.OBS):
                obs[i] = Scalar[dtype](obs_list[i])

            var episode_reward: Float64 = 0.0
            var episode_steps = 0

            for step in range(max_steps):
                # stochastic=True samples from policy, False uses mean
                var action_result = self.select_action(obs, training=stochastic)
                var actions = action_result[0].copy()

                # Convert actions to List for environment
                # Apply action scaling and clip to environment bounds
                var action_list = List[Scalar[dtype]]()
                for j in range(Self.ACTIONS):
                    var action_val = Float64(actions[j])
                    action_val = (
                        action_val * self.action_scale + self.action_bias
                    )
                    # Clip to [-1, 1] for environment (unbounded Gaussian may exceed)
                    if action_val > 1.0:
                        action_val = 1.0
                    elif action_val < -1.0:
                        action_val = -1.0
                    action_list.append(Scalar[dtype](action_val))

                # Step environment with multi-dimensional actions
                var result = env.step_continuous_vec[dtype](action_list)
                var next_obs_list = result[0].copy()
                var reward = result[1]
                var done = result[2]

                if renderer:
                    env.render(renderer[])

                episode_reward += Float64(reward)
                episode_steps += 1

                # Update observation
                for i in range(Self.OBS):
                    obs[i] = next_obs_list[i]

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

    fn evaluate_gpu[
        EnvType: GPUContinuousEnv
    ](
        self,
        ctx: DeviceContext,
        num_episodes: Int = 100,
        max_steps: Int = 1000,
        verbose: Bool = False,
        stochastic: Bool = True,
    ) raises -> Float64:
        """Evaluate the agent on GPU parallel environments.

        Uses unbounded Gaussian policy (CleanRL-style). Actions are clipped
        to environment bounds by the GPU environment kernel.

        Args:
            ctx: GPU device context.
            num_episodes: Target number of evaluation episodes.
            max_steps: Maximum steps per episode.
            verbose: Whether to print progress.
            stochastic: If True (default), sample from policy; if False, use mean.

        Returns:
            Average reward over completed episodes.
        """
        # =====================================================================
        # Buffer allocation
        # =====================================================================
        comptime ENV_OBS_SIZE = Self.n_envs * Self.OBS
        comptime ENV_STATE_SIZE = Self.n_envs * EnvType.STATE_SIZE
        comptime ENV_ACTION_SIZE = Self.n_envs * Self.ACTIONS

        # Environment state buffers
        var env_states_buf = ctx.enqueue_create_buffer[dtype](ENV_STATE_SIZE)
        var obs_buf = ctx.enqueue_create_buffer[dtype](ENV_OBS_SIZE)
        var rewards_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var dones_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)

        # Action buffers
        var actions_buf = ctx.enqueue_create_buffer[dtype](ENV_ACTION_SIZE)
        var actor_out_buf = ctx.enqueue_create_buffer[dtype](
            Self.n_envs * Self.ACTOR_OUT
        )

        # Network parameter buffers (copy from CPU)
        var actor_params_buf = ctx.enqueue_create_buffer[dtype](
            Self.ACTOR_PARAM_SIZE
        )
        ctx.enqueue_copy(actor_params_buf, self.actor.params.unsafe_ptr())

        # Workspace buffer for forward pass
        comptime WORKSPACE_PER_SAMPLE = 4 * Self.HIDDEN
        comptime ENV_WORKSPACE_SIZE = Self.n_envs * WORKSPACE_PER_SAMPLE
        var actor_workspace_buf = ctx.enqueue_create_buffer[dtype](
            ENV_WORKSPACE_SIZE
        )

        # Tracking arrays (on CPU)
        var episode_rewards = List[Float64]()
        var current_rewards = InlineArray[Float64, Self.n_envs](fill=0.0)
        var episodes_completed = 0

        # =====================================================================
        # Initialize environments
        # =====================================================================
        EnvType.reset_kernel_gpu[Self.n_envs, EnvType.STATE_SIZE](
            ctx, env_states_buf
        )

        # Extract initial observations (WITHOUT stepping - just read from state)
        comptime ENV_BLOCKS = (Self.n_envs + TPB - 1) // TPB
        var obs_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var states_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs, EnvType.STATE_SIZE), MutAnyOrigin
        ](env_states_buf.unsafe_ptr())

        comptime extract_obs_wrapper = _extract_obs_from_state_continuous_kernel[
            dtype, Self.n_envs, EnvType.STATE_SIZE, Self.OBS
        ]
        ctx.enqueue_function[extract_obs_wrapper, extract_obs_wrapper](
            obs_tensor,
            states_tensor,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )
        ctx.synchronize()

        if verbose:
            print(
                "Running GPU evaluation with", Self.n_envs, "parallel envs..."
            )

        # =====================================================================
        # Evaluation loop
        # =====================================================================
        # Note: ENV_BLOCKS already defined above for observation extraction

        # Buffers for stochastic sampling (log_probs needed for sampling kernel)
        var log_probs_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)

        # Kernel for deterministic action extraction (unbounded Gaussian - use mean directly)
        @always_inline
        fn extract_deterministic_actions(
            actions: LayoutTensor[
                dtype, Layout.row_major(Self.n_envs, Self.ACTIONS), MutAnyOrigin
            ],
            actor_out: LayoutTensor[
                dtype,
                Layout.row_major(Self.n_envs, Self.ACTOR_OUT),
                ImmutAnyOrigin,
            ],
        ):
            var idx = Int(block_idx.x) * TPB + Int(thread_idx.x)
            if idx >= Self.n_envs:
                return

            # Use mean directly (unbounded Gaussian, no tanh squashing)
            for j in range(Self.ACTIONS):
                actions[idx, j] = actor_out[idx, j]

        # Sampling kernel wrapper for stochastic evaluation
        comptime sample_actions_wrapper = _sample_continuous_actions_kernel[
            dtype, Self.n_envs, Self.ACTIONS
        ]

        var step = 0
        while episodes_completed < num_episodes and step < max_steps:
            # Forward actor to get mean and log_std
            self.actor.model.forward_gpu_no_cache_ws[Self.n_envs](
                ctx,
                actor_out_buf,
                obs_buf,
                actor_params_buf,
                actor_workspace_buf,
            )

            var actions_tensor = LayoutTensor[
                dtype, Layout.row_major(Self.n_envs, Self.ACTIONS), MutAnyOrigin
            ](actions_buf.unsafe_ptr())
            var actor_out_tensor = LayoutTensor[
                dtype,
                Layout.row_major(Self.n_envs, Self.ACTOR_OUT),
                MutAnyOrigin,
            ](actor_out_buf.unsafe_ptr())

            if stochastic:
                # Stochastic: sample from policy distribution (unbounded Gaussian)
                var log_probs_tensor = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](log_probs_buf.unsafe_ptr())

                ctx.enqueue_function[
                    sample_actions_wrapper, sample_actions_wrapper
                ](
                    actor_out_tensor,
                    actions_tensor,
                    log_probs_tensor,
                    Scalar[DType.uint32](step * 2654435761),
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )
            else:
                # Deterministic: use mean action
                var actor_out_immut = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs, Self.ACTOR_OUT),
                    ImmutAnyOrigin,
                ](actor_out_buf.unsafe_ptr())
                ctx.enqueue_function[
                    extract_deterministic_actions, extract_deterministic_actions
                ](
                    actions_tensor,
                    actor_out_immut,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )

            # Step all environments
            EnvType.step_kernel_gpu[
                Self.n_envs, EnvType.STATE_SIZE, Self.OBS, Self.ACTIONS
            ](
                ctx,
                env_states_buf,
                actions_buf,
                rewards_buf,
                dones_buf,
                obs_buf,
                UInt64(step),  # rng_seed
            )
            ctx.synchronize()

            # Copy rewards and dones to CPU
            var rewards_host = InlineArray[Scalar[dtype], Self.n_envs](
                uninitialized=True
            )
            var dones_host = InlineArray[Scalar[dtype], Self.n_envs](
                uninitialized=True
            )
            ctx.enqueue_copy(rewards_host.unsafe_ptr(), rewards_buf)
            ctx.enqueue_copy(dones_host.unsafe_ptr(), dones_buf)
            ctx.synchronize()

            # Track rewards and episode completion
            for i in range(Self.n_envs):
                current_rewards[i] += Float64(rewards_host[i])

                if dones_host[i] > 0:
                    episode_rewards.append(current_rewards[i])
                    current_rewards[i] = 0.0
                    episodes_completed += 1

                    if episodes_completed >= num_episodes:
                        break

            # Auto-reset done environments
            EnvType.selective_reset_kernel_gpu[Self.n_envs, EnvType.STATE_SIZE](
                ctx,
                env_states_buf,
                dones_buf,
                UInt64(step),
            )

            # Extract observations from reset environments (critical for correct forward pass)
            ctx.enqueue_function[extract_obs_wrapper, extract_obs_wrapper](
                obs_tensor,
                states_tensor,
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )

            step += 1

        # =====================================================================
        # Compute statistics
        # =====================================================================
        if len(episode_rewards) == 0:
            if verbose:
                print("Warning: No episodes completed!")
            return 0.0

        var total_reward: Float64 = 0.0
        var min_reward = episode_rewards[0]
        var max_reward = episode_rewards[0]

        for i in range(len(episode_rewards)):
            total_reward += episode_rewards[i]
            if episode_rewards[i] < min_reward:
                min_reward = episode_rewards[i]
            if episode_rewards[i] > max_reward:
                max_reward = episode_rewards[i]

        var avg_reward = total_reward / Float64(len(episode_rewards))

        if verbose:
            print(
                "----------------------------------------------------------------------"
            )
            print("GPU EVALUATION SUMMARY (Continuous Actions)")
            print(
                "----------------------------------------------------------------------"
            )
            print("Episodes completed:", len(episode_rewards))
            print("Average reward:", avg_reward)
            print("Min reward:", min_reward)
            print("Max reward:", max_reward)

        return avg_reward

    # =========================================================================
    # GPU Training with CPU Environments (Hybrid)
    # =========================================================================

    fn train_gpu_cpu_env[
        EnvType: BoxContinuousActionEnv & Copyable & Movable,
    ](
        mut self,
        ctx: DeviceContext,
        mut envs: List[EnvType],
        num_episodes: Int,
        verbose: Bool = False,
        print_every: Int = 10,
    ) raises -> TrainingMetrics:
        """Train PPO on GPU with CPU environments for continuous action spaces.

        This hybrid approach uses CPU environments for accurate physics simulation
        while leveraging GPU for neural network computations.

        Args:
            ctx: GPU device context.
            envs: List of n_envs CPU environments (must match Self.n_envs).
            num_episodes: Target number of episodes to complete.
            verbose: Whether to print progress.
            print_every: Print progress every N rollouts.

        Returns:
            TrainingMetrics with episode rewards and statistics.
        """
        # Validate environment count
        if len(envs) != Self.n_envs:
            print(
                "Error: Expected",
                Self.n_envs,
                "environments, got",
                len(envs),
            )
            return TrainingMetrics(
                algorithm_name="Deep PPO Continuous (GPU+CPU)",
                environment_name="Error",
            )

        var metrics = TrainingMetrics(
            algorithm_name="Deep PPO Continuous (GPU+CPU)",
            environment_name="CPU Environment",
        )

        # =====================================================================
        # Compile-time constants for buffer sizes
        # =====================================================================
        comptime ACTOR_PARAMS = Self.ACTOR_PARAM_SIZE
        comptime CRITIC_PARAMS = Self.CRITIC_PARAM_SIZE
        comptime ACTOR_STATE = ACTOR_PARAMS * 2  # Adam: 2 states per param
        comptime CRITIC_STATE = CRITIC_PARAMS * 2

        comptime ENV_OBS_SIZE = Self.n_envs * Self.OBS
        comptime ENV_ACTION_SIZE = Self.n_envs * Self.ACTIONS
        comptime ENV_ACTOR_OUT_SIZE = Self.n_envs * Self.ACTOR_OUT
        comptime ROLLOUT_TOTAL = Self.TOTAL_ROLLOUT_SIZE
        comptime ROLLOUT_OBS_SIZE = ROLLOUT_TOTAL * Self.OBS
        comptime ROLLOUT_ACTION_SIZE = ROLLOUT_TOTAL * Self.ACTIONS

        comptime MINIBATCH = Self.GPU_MINIBATCH
        comptime MINIBATCH_OBS_SIZE = MINIBATCH * Self.OBS
        comptime MINIBATCH_ACTION_SIZE = MINIBATCH * Self.ACTIONS
        comptime MINIBATCH_ACTOR_OUT_SIZE = MINIBATCH * Self.ACTOR_OUT
        comptime MINIBATCH_CACHE_ACTOR = MINIBATCH * Self.ACTOR_CACHE
        comptime MINIBATCH_CACHE_CRITIC = MINIBATCH * Self.CRITIC_CACHE

        comptime ENV_BLOCKS = (Self.n_envs + TPB - 1) // TPB
        comptime MINIBATCH_BLOCKS = (MINIBATCH + TPB - 1) // TPB
        comptime ROLLOUT_BLOCKS = (ROLLOUT_TOTAL + TPB - 1) // TPB

        # Workspace sizes for forward passes
        comptime WORKSPACE_PER_SAMPLE = 4 * Self.HIDDEN
        comptime ENV_WORKSPACE_SIZE = Self.n_envs * WORKSPACE_PER_SAMPLE
        comptime MINIBATCH_WORKSPACE_SIZE = MINIBATCH * WORKSPACE_PER_SAMPLE

        # =====================================================================
        # Network parameter buffers (GPU)
        # =====================================================================
        var actor_params_buf = ctx.enqueue_create_buffer[dtype](ACTOR_PARAMS)
        var actor_grads_buf = ctx.enqueue_create_buffer[dtype](ACTOR_PARAMS)
        var actor_state_buf = ctx.enqueue_create_buffer[dtype](ACTOR_STATE)

        var critic_params_buf = ctx.enqueue_create_buffer[dtype](CRITIC_PARAMS)
        var critic_grads_buf = ctx.enqueue_create_buffer[dtype](CRITIC_PARAMS)
        var critic_state_buf = ctx.enqueue_create_buffer[dtype](CRITIC_STATE)

        # Pre-allocated workspace buffers
        var actor_env_workspace_buf = ctx.enqueue_create_buffer[dtype](
            ENV_WORKSPACE_SIZE
        )
        var critic_env_workspace_buf = ctx.enqueue_create_buffer[dtype](
            ENV_WORKSPACE_SIZE
        )
        var actor_minibatch_workspace_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_WORKSPACE_SIZE
        )
        var critic_minibatch_workspace_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_WORKSPACE_SIZE
        )

        # =====================================================================
        # Environment buffers (GPU device + CPU host for transfers)
        # =====================================================================
        var obs_buf = ctx.enqueue_create_buffer[dtype](ENV_OBS_SIZE)
        var obs_host = ctx.enqueue_create_host_buffer[dtype](ENV_OBS_SIZE)
        var rewards_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var dones_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var actions_buf = ctx.enqueue_create_buffer[dtype](ENV_ACTION_SIZE)
        var actions_host = ctx.enqueue_create_host_buffer[dtype](
            ENV_ACTION_SIZE
        )
        var values_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var log_probs_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var actor_output_buf = ctx.enqueue_create_buffer[dtype](
            ENV_ACTOR_OUT_SIZE
        )

        # =====================================================================
        # Rollout buffers
        # =====================================================================
        var rollout_obs_buf = ctx.enqueue_create_buffer[dtype](ROLLOUT_OBS_SIZE)
        var rollout_actions_buf = ctx.enqueue_create_buffer[dtype](
            ROLLOUT_ACTION_SIZE
        )
        var rollout_rewards_buf = ctx.enqueue_create_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var rollout_values_buf = ctx.enqueue_create_buffer[dtype](ROLLOUT_TOTAL)
        var rollout_log_probs_buf = ctx.enqueue_create_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var rollout_dones_buf = ctx.enqueue_create_buffer[dtype](ROLLOUT_TOTAL)

        # Advantages and returns
        var advantages_buf = ctx.enqueue_create_buffer[dtype](ROLLOUT_TOTAL)
        var returns_buf = ctx.enqueue_create_buffer[dtype](ROLLOUT_TOTAL)

        # Host buffers for GAE computation
        var rollout_rewards_host = ctx.enqueue_create_host_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var rollout_values_host = ctx.enqueue_create_host_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var rollout_dones_host = ctx.enqueue_create_host_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var advantages_host = ctx.enqueue_create_host_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var returns_host = ctx.enqueue_create_host_buffer[dtype](ROLLOUT_TOTAL)
        var bootstrap_values_host = ctx.enqueue_create_host_buffer[dtype](
            Self.n_envs
        )

        # =====================================================================
        # Minibatch buffers (for training)
        # =====================================================================
        var mb_obs_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH_OBS_SIZE)
        var mb_actions_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_ACTION_SIZE
        )
        var mb_advantages_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var mb_returns_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var mb_old_log_probs_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var mb_old_values_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var mb_indices_buf = ctx.enqueue_create_buffer[DType.int32](MINIBATCH)
        var mb_indices_host = ctx.enqueue_create_host_buffer[DType.int32](
            MINIBATCH
        )

        # Training workspace
        var actor_output_mb_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_ACTOR_OUT_SIZE
        )
        var actor_cache_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_CACHE_ACTOR
        )
        var actor_grad_output_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_ACTOR_OUT_SIZE
        )
        var actor_grad_input_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_OBS_SIZE
        )

        var critic_values_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var critic_cache_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_CACHE_CRITIC
        )
        var critic_grad_output_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var critic_grad_input_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_OBS_SIZE
        )

        # KL divergence and gradient clipping buffers
        var kl_divergences_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var kl_divergences_host = ctx.enqueue_create_host_buffer[dtype](
            MINIBATCH
        )
        var mb_advantages_host = ctx.enqueue_create_host_buffer[dtype](
            MINIBATCH
        )

        comptime ACTOR_GRAD_BLOCKS = (ACTOR_PARAMS + TPB - 1) // TPB
        comptime CRITIC_GRAD_BLOCKS = (CRITIC_PARAMS + TPB - 1) // TPB
        var actor_grad_partial_sums_buf = ctx.enqueue_create_buffer[dtype](
            ACTOR_GRAD_BLOCKS
        )
        var critic_grad_partial_sums_buf = ctx.enqueue_create_buffer[dtype](
            CRITIC_GRAD_BLOCKS
        )
        var actor_grad_partial_sums_host = ctx.enqueue_create_host_buffer[
            dtype
        ](ACTOR_GRAD_BLOCKS)
        var critic_grad_partial_sums_host = ctx.enqueue_create_host_buffer[
            dtype
        ](CRITIC_GRAD_BLOCKS)

        # =====================================================================
        # Initialize network parameters on GPU
        # =====================================================================
        self.actor.copy_params_to_device(ctx, actor_params_buf)
        self.actor.copy_state_to_device(ctx, actor_state_buf)
        self.critic.copy_params_to_device(ctx, critic_params_buf)
        self.critic.copy_state_to_device(ctx, critic_state_buf)

        # =====================================================================
        # Create LayoutTensor views
        # =====================================================================
        var obs_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var rewards_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var dones_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](dones_buf.unsafe_ptr())
        var actions_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs, Self.ACTIONS), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var actor_output_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs, Self.ACTOR_OUT), MutAnyOrigin
        ](actor_output_buf.unsafe_ptr())
        var log_probs_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](log_probs_buf.unsafe_ptr())

        var mb_obs_tensor = LayoutTensor[
            dtype,
            Layout.row_major(MINIBATCH, Self.OBS),
            MutAnyOrigin,
        ](mb_obs_buf.unsafe_ptr())
        var mb_actions_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, Self.ACTIONS), MutAnyOrigin
        ](mb_actions_buf.unsafe_ptr())
        var mb_advantages_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_advantages_buf.unsafe_ptr())
        var mb_returns_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_returns_buf.unsafe_ptr())
        var mb_old_log_probs_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_old_log_probs_buf.unsafe_ptr())
        var mb_old_values_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_old_values_buf.unsafe_ptr())
        var rollout_obs_tensor = LayoutTensor[
            dtype,
            Layout.row_major(ROLLOUT_TOTAL, Self.OBS),
            MutAnyOrigin,
        ](rollout_obs_buf.unsafe_ptr())
        var rollout_actions_tensor = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL, Self.ACTIONS), MutAnyOrigin
        ](rollout_actions_buf.unsafe_ptr())
        var advantages_tensor = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](advantages_buf.unsafe_ptr())
        var returns_tensor = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](returns_buf.unsafe_ptr())
        var rollout_log_probs_tensor = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](rollout_log_probs_buf.unsafe_ptr())
        var rollout_values_tensor = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](rollout_values_buf.unsafe_ptr())
        var mb_indices_tensor = LayoutTensor[
            DType.int32, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_indices_buf.unsafe_ptr())

        var actor_output_mb_tensor = LayoutTensor[
            dtype,
            Layout.row_major(MINIBATCH, Self.ACTOR_OUT),
            MutAnyOrigin,
        ](actor_output_mb_buf.unsafe_ptr())
        var actor_grad_output_tensor = LayoutTensor[
            dtype,
            Layout.row_major(MINIBATCH, Self.ACTOR_OUT),
            MutAnyOrigin,
        ](actor_grad_output_buf.unsafe_ptr())
        var critic_values_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, 1), MutAnyOrigin
        ](critic_values_buf.unsafe_ptr())
        var critic_grad_output_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, 1), MutAnyOrigin
        ](critic_grad_output_buf.unsafe_ptr())

        var kl_divergences_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](kl_divergences_buf.unsafe_ptr())

        var actor_grads_tensor = LayoutTensor[
            dtype, Layout.row_major(ACTOR_PARAMS), MutAnyOrigin
        ](actor_grads_buf.unsafe_ptr())
        var critic_grads_tensor = LayoutTensor[
            dtype, Layout.row_major(CRITIC_PARAMS), MutAnyOrigin
        ](critic_grads_buf.unsafe_ptr())
        var actor_grad_partial_sums_tensor = LayoutTensor[
            dtype, Layout.row_major(ACTOR_GRAD_BLOCKS), MutAnyOrigin
        ](actor_grad_partial_sums_buf.unsafe_ptr())
        var critic_grad_partial_sums_tensor = LayoutTensor[
            dtype, Layout.row_major(CRITIC_GRAD_BLOCKS), MutAnyOrigin
        ](critic_grad_partial_sums_buf.unsafe_ptr())

        # =====================================================================
        # Initialize all CPU environments and copy observations to GPU
        # =====================================================================
        for i in range(Self.n_envs):
            var obs_list = envs[i].reset_obs_list()
            for d in range(Self.OBS):
                obs_host[i * Self.OBS + d] = Scalar[dtype](obs_list[d])

        ctx.enqueue_copy(obs_buf, obs_host)
        ctx.synchronize()

        # =====================================================================
        # Episode tracking (CPU-side)
        # =====================================================================
        var episode_rewards = List[Scalar[dtype]]()
        var episode_steps = List[Int]()
        for _ in range(Self.n_envs):
            episode_rewards.append(0.0)
            episode_steps.append(0)

        # =====================================================================
        # Training state
        # =====================================================================
        var completed_episodes = 0
        var total_steps = 0
        var rollout_count = 0

        # Annealing target
        var annealing_target_steps = self.target_total_steps
        if annealing_target_steps == 0:
            annealing_target_steps = (
                num_episodes * 500
            )  # Longer for continuous control

        var initial_actor_lr = self.actor_lr
        var initial_critic_lr = self.critic_lr
        var initial_entropy_coef = self.entropy_coef

        # Kernel wrappers
        comptime sample_actions_wrapper = _sample_continuous_actions_kernel[
            dtype, Self.n_envs, Self.ACTIONS
        ]
        comptime store_pre_step_wrapper = _store_continuous_pre_step_kernel[
            dtype, Self.n_envs, Self.OBS, Self.ACTIONS
        ]
        comptime gather_wrapper = ppo_continuous_gather_minibatch_kernel[
            dtype, MINIBATCH, Self.OBS, Self.ACTIONS, ROLLOUT_TOTAL
        ]
        comptime actor_grad_wrapper = ppo_continuous_actor_grad_kernel[
            dtype, MINIBATCH, Self.ACTIONS
        ]
        comptime critic_grad_wrapper = ppo_critic_grad_kernel[dtype, MINIBATCH]
        comptime critic_grad_clipped_wrapper = ppo_critic_grad_clipped_kernel[
            dtype, MINIBATCH
        ]
        comptime normalize_advantages_wrapper = normalize_advantages_kernel[
            dtype, MINIBATCH
        ]
        # Fused GPU-only normalization (no CPU roundtrip)
        comptime normalize_advantages_fused_wrapper = normalize_advantages_fused_kernel[
            dtype, MINIBATCH, TPB
        ]

        comptime actor_grad_norm_wrapper = gradient_norm_kernel[
            dtype, ACTOR_PARAMS, ACTOR_GRAD_BLOCKS, TPB
        ]
        comptime critic_grad_norm_wrapper = gradient_norm_kernel[
            dtype, CRITIC_PARAMS, CRITIC_GRAD_BLOCKS, TPB
        ]
        comptime actor_grad_clip_wrapper = gradient_clip_kernel[
            dtype, ACTOR_PARAMS
        ]
        comptime critic_grad_clip_wrapper = gradient_clip_kernel[
            dtype, CRITIC_PARAMS
        ]

        # Fully fused gradient clipping wrappers (2 kernels instead of 3)
        # Each block redundantly reduces partial_sums and applies scale
        comptime actor_reduce_apply_fused_wrapper = gradient_reduce_apply_fused_kernel[
            dtype, ACTOR_PARAMS, ACTOR_GRAD_BLOCKS, TPB
        ]
        comptime critic_reduce_apply_fused_wrapper = gradient_reduce_apply_fused_kernel[
            dtype, CRITIC_PARAMS, CRITIC_GRAD_BLOCKS, TPB
        ]

        # Timing accumulators
        var total_phase1_ns: UInt = 0
        var total_phase2_ns: UInt = 0
        var total_phase3_ns: UInt = 0

        # Phase 3 sub-timers
        var total_shuffle_ns: UInt = 0
        var total_indices_ns: UInt = 0
        var total_gather_ns: UInt = 0
        # Fine-grained actor timers
        var total_actor_forward_ns: UInt = 0
        var total_actor_grad_kernel_ns: UInt = 0
        var total_actor_backward_ns: UInt = 0
        var total_actor_grad_clip_ns: UInt = 0
        var total_actor_optim_ns: UInt = 0
        # Fine-grained critic timers
        var total_critic_forward_ns: UInt = 0
        var total_critic_grad_kernel_ns: UInt = 0
        var total_critic_backward_ns: UInt = 0
        var total_critic_grad_clip_ns: UInt = 0
        var total_critic_optim_ns: UInt = 0

        # =====================================================================
        # Main Training Loop
        # =====================================================================

        while completed_episodes < num_episodes:
            var rollout_start_episodes = completed_episodes
            rollout_count += 1

            # =================================================================
            # Phase 1: Collect rollout (CPU environments + GPU forward passes)
            # =================================================================
            var phase1_start = perf_counter_ns()

            for t in range(Self.rollout_len):
                var rng_seed = UInt32(total_steps * 2654435761 + t * 7919)

                # Forward actor on GPU to get mean and log_std
                self.actor.model.forward_gpu_no_cache_ws[Self.n_envs](
                    ctx,
                    actor_output_buf,
                    obs_buf,
                    actor_params_buf,
                    actor_env_workspace_buf,
                )

                # Forward critic on GPU to get values
                self.critic.model.forward_gpu_no_cache_ws[Self.n_envs](
                    ctx,
                    values_buf,
                    obs_buf,
                    critic_params_buf,
                    critic_env_workspace_buf,
                )
                ctx.synchronize()

                # Sample continuous actions on GPU (unbounded Gaussian)
                ctx.enqueue_function[
                    sample_actions_wrapper, sample_actions_wrapper
                ](
                    actor_output_tensor,
                    actions_tensor,
                    log_probs_tensor,
                    Scalar[DType.uint32](rng_seed),
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )
                ctx.synchronize()

                # Copy actions to CPU for environment stepping
                ctx.enqueue_copy(actions_host, actions_buf)
                ctx.synchronize()

                # Store pre-step data to rollout buffer (using actions directly)
                var t_offset = t * Self.n_envs

                var rollout_obs_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs, Self.OBS),
                    MutAnyOrigin,
                ](rollout_obs_buf.unsafe_ptr() + t_offset * Self.OBS)
                var rollout_actions_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs, Self.ACTIONS),
                    MutAnyOrigin,
                ](rollout_actions_buf.unsafe_ptr() + t_offset * Self.ACTIONS)
                var rollout_log_probs_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](rollout_log_probs_buf.unsafe_ptr() + t_offset)
                var rollout_values_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](rollout_values_buf.unsafe_ptr() + t_offset)

                var values_tensor_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](values_buf.unsafe_ptr())

                ctx.enqueue_function[
                    store_pre_step_wrapper, store_pre_step_wrapper
                ](
                    rollout_obs_t,
                    rollout_actions_t,
                    rollout_log_probs_t,
                    rollout_values_t,
                    obs_tensor,
                    actions_tensor,
                    log_probs_tensor,
                    values_tensor_t,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )
                ctx.synchronize()

                # =============================================================
                # Step CPU environments with continuous actions
                # =============================================================

                for i in range(Self.n_envs):
                    # Build action list from GPU buffer
                    var action_list = List[Scalar[dtype]]()
                    for a in range(Self.ACTIONS):
                        var action_val = Float64(
                            actions_host[i * Self.ACTIONS + a]
                        )
                        # Apply action scaling and clip to environment bounds
                        action_val = (
                            action_val * self.action_scale + self.action_bias
                        )
                        # Clip to [-1, 1] for environment (unbounded Gaussian may exceed)
                        if action_val > 1.0:
                            action_val = 1.0
                        elif action_val < -1.0:
                            action_val = -1.0
                        action_list.append(Scalar[dtype](action_val))

                    var result = envs[i].step_continuous_vec[dtype](
                        action_list^
                    )
                    var next_obs = result[0].copy()
                    var reward = result[1]
                    var done = result[2]

                    # Store reward and done in host buffer
                    rollout_rewards_host[t_offset + i] = Scalar[dtype](reward)
                    rollout_dones_host[t_offset + i] = Scalar[dtype](
                        1.0 if done else 0.0
                    )

                    # Update episode tracking
                    episode_rewards[i] += Scalar[dtype](reward)
                    episode_steps[i] += 1

                    # Handle episode completion
                    if done:
                        metrics.log_episode(
                            completed_episodes,
                            Float64(episode_rewards[i]),
                            episode_steps[i],
                            0.0,
                        )
                        completed_episodes += 1

                        # Auto-checkpoint
                        if (
                            self.checkpoint_every > 0
                            and self.checkpoint_path != ""
                            and completed_episodes % self.checkpoint_every == 0
                        ):
                            self.actor.copy_params_from_device(
                                ctx, actor_params_buf
                            )
                            self.actor.copy_state_from_device(
                                ctx, actor_state_buf
                            )
                            self.critic.copy_params_from_device(
                                ctx, critic_params_buf
                            )
                            self.critic.copy_state_from_device(
                                ctx, critic_state_buf
                            )
                            ctx.synchronize()
                            self.save_checkpoint(self.checkpoint_path)
                            if verbose:
                                print(
                                    "  [Checkpoint saved at episode",
                                    completed_episodes,
                                    "]",
                                )

                        # Reset episode tracking
                        episode_rewards[i] = 0.0
                        episode_steps[i] = 0

                        # Reset environment
                        var reset_obs = envs[i].reset_obs_list()
                        for d in range(Self.OBS):
                            obs_host[i * Self.OBS + d] = Scalar[dtype](
                                reset_obs[d]
                            )
                    else:
                        # Copy next observation
                        for d in range(Self.OBS):
                            obs_host[i * Self.OBS + d] = Scalar[dtype](
                                next_obs[d]
                            )

                total_steps += Self.n_envs

                # Copy updated observations to GPU for next iteration
                ctx.enqueue_copy(obs_buf, obs_host)
                ctx.synchronize()

            # Early exit if we've reached target episodes
            if completed_episodes >= num_episodes:
                break

            var phase1_end = perf_counter_ns()

            # =================================================================
            # Phase 2: Compute GAE advantages on CPU
            # =================================================================
            var phase2_start = perf_counter_ns()

            # Get bootstrap values from final observations
            self.critic.model.forward_gpu_no_cache_ws[Self.n_envs](
                ctx,
                values_buf,
                obs_buf,
                critic_params_buf,
                critic_env_workspace_buf,
            )

            ctx.enqueue_copy(bootstrap_values_host, values_buf)

            # Copy rollout values to CPU
            ctx.enqueue_copy(rollout_values_host, rollout_values_buf)
            ctx.synchronize()

            # Compute GAE for each environment
            for env_idx in range(Self.n_envs):
                var gae = Scalar[dtype](0.0)
                var gae_decay = Scalar[dtype](self.gamma * self.gae_lambda)
                var bootstrap_val = Scalar[dtype](
                    bootstrap_values_host[env_idx]
                )

                for t in range(Self.rollout_len - 1, -1, -1):
                    var idx = t * Self.n_envs + env_idx
                    var reward = rollout_rewards_host[idx]
                    var value = rollout_values_host[idx]
                    var done = rollout_dones_host[idx]

                    var next_val: Scalar[dtype]
                    if t == Self.rollout_len - 1:
                        next_val = bootstrap_val
                    else:
                        var next_idx = (t + 1) * Self.n_envs + env_idx
                        next_val = rollout_values_host[next_idx]

                    if done > Scalar[dtype](0.5):
                        next_val = Scalar[dtype](0.0)
                        gae = Scalar[dtype](0.0)

                    var delta = (
                        reward + Scalar[dtype](self.gamma) * next_val - value
                    )
                    gae = delta + gae_decay * gae

                    advantages_host[idx] = gae
                    returns_host[idx] = gae + value

            # Normalize advantages
            if self.normalize_advantages:
                var mean = Scalar[dtype](0.0)
                var var_sum = Scalar[dtype](0.0)
                for i in range(ROLLOUT_TOTAL):
                    mean += advantages_host[i]
                mean /= Scalar[dtype](ROLLOUT_TOTAL)
                for i in range(ROLLOUT_TOTAL):
                    var diff = advantages_host[i] - mean
                    var_sum += diff * diff

                var variance = var_sum / Scalar[dtype](ROLLOUT_TOTAL)
                var std = sqrt(Float64(variance) + 1e-8)
                for i in range(ROLLOUT_TOTAL):
                    advantages_host[i] = (advantages_host[i] - mean) / (
                        Scalar[dtype](std) + Scalar[dtype](1e-8)
                    )

            # Copy advantages, returns, rewards, dones to GPU
            ctx.enqueue_copy(advantages_buf, advantages_host)
            ctx.enqueue_copy(returns_buf, returns_host)
            ctx.enqueue_copy(rollout_rewards_buf, rollout_rewards_host)
            ctx.enqueue_copy(rollout_dones_buf, rollout_dones_host)
            ctx.synchronize()

            var phase2_end = perf_counter_ns()

            # =================================================================
            # Phase 3: Train actor and critic with minibatches (GPU)
            # =================================================================
            var phase3_start = perf_counter_ns()

            var progress = Float64(total_steps) / Float64(
                annealing_target_steps
            )
            if progress > 1.0:
                progress = 1.0

            var current_actor_lr = initial_actor_lr
            var current_critic_lr = initial_critic_lr
            var current_entropy_coef = initial_entropy_coef
            if self.anneal_lr:
                var lr_multiplier = 1.0 - progress
                current_actor_lr = initial_actor_lr * lr_multiplier
                current_critic_lr = initial_critic_lr * lr_multiplier
                self.actor.optimizer.lr = current_actor_lr
                self.critic.optimizer.lr = current_critic_lr

            if self.anneal_entropy:
                current_entropy_coef = initial_entropy_coef * (1.0 - progress)

            var kl_early_stop = False

            # Sub-timers for this rollout's phase 3
            var shuffle_time_ns: UInt = 0
            var indices_copy_time_ns: UInt = 0
            var gather_time_ns: UInt = 0
            # Fine-grained actor timers
            var actor_forward_ns: UInt = 0
            var actor_grad_kernel_ns: UInt = 0
            var actor_backward_ns: UInt = 0
            var actor_grad_clip_ns: UInt = 0
            var actor_optim_ns: UInt = 0
            # Fine-grained critic timers
            var critic_forward_ns: UInt = 0
            var critic_grad_kernel_ns: UInt = 0
            var critic_backward_ns: UInt = 0
            var critic_grad_clip_ns: UInt = 0
            var critic_optim_ns: UInt = 0

            for epoch in range(self.num_epochs):
                if kl_early_stop:
                    break

                # Generate shuffled indices
                var shuffle_start = perf_counter_ns()
                var indices_list = List[Int]()
                for i in range(ROLLOUT_TOTAL):
                    indices_list.append(i)

                for i in range(ROLLOUT_TOTAL - 1, 0, -1):
                    var j = Int(random_float64() * Float64(i + 1))
                    var temp = indices_list[i]
                    indices_list[i] = indices_list[j]
                    indices_list[j] = temp
                shuffle_time_ns += perf_counter_ns() - shuffle_start

                var num_minibatches = ROLLOUT_TOTAL // MINIBATCH
                for mb_idx in range(num_minibatches):
                    var start_idx = mb_idx * MINIBATCH

                    # Copy indices to host buffer
                    var indices_copy_start = perf_counter_ns()
                    for i in range(MINIBATCH):
                        mb_indices_host[i] = Int32(indices_list[start_idx + i])

                    ctx.enqueue_copy(mb_indices_buf, mb_indices_host)
                    indices_copy_time_ns += (
                        perf_counter_ns() - indices_copy_start
                    )

                    # Gather minibatch data
                    var gather_start = perf_counter_ns()
                    ctx.enqueue_function[gather_wrapper, gather_wrapper](
                        mb_obs_tensor,
                        mb_actions_tensor,
                        mb_advantages_tensor,
                        mb_returns_tensor,
                        mb_old_log_probs_tensor,
                        mb_old_values_tensor,
                        rollout_obs_tensor,
                        rollout_actions_tensor,
                        advantages_tensor,
                        returns_tensor,
                        rollout_log_probs_tensor,
                        rollout_values_tensor,
                        mb_indices_tensor,
                        MINIBATCH,
                        grid_dim=(MINIBATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    # No sync needed - GPU operations execute in order on same stream

                    # Per-minibatch advantage normalization (include in gather time)
                    # Uses fused GPU kernel - no CPU roundtrip needed
                    if self.norm_adv_per_minibatch:
                        ctx.enqueue_function[
                            normalize_advantages_fused_wrapper,
                            normalize_advantages_fused_wrapper,
                        ](
                            mb_advantages_tensor,
                            grid_dim=(1,),  # Single block for reduction
                            block_dim=(TPB,),
                        )
                        # No sync needed - next kernels will wait for this to complete
                    gather_time_ns += perf_counter_ns() - gather_start

                    # Train actor - forward pass
                    var actor_fwd_start = perf_counter_ns()
                    ctx.enqueue_memset(actor_grads_buf, 0)
                    self.actor.model.forward_gpu_ws[MINIBATCH](
                        ctx,
                        actor_output_mb_buf,
                        mb_obs_buf,
                        actor_params_buf,
                        actor_cache_buf,
                        actor_minibatch_workspace_buf,
                    )
                    # No sync needed - grad kernel uses forward output
                    actor_forward_ns += perf_counter_ns() - actor_fwd_start

                    # Actor grad kernel (ppo_continuous_actor_grad_kernel)
                    var actor_grad_start = perf_counter_ns()
                    ctx.enqueue_function[
                        actor_grad_wrapper, actor_grad_wrapper
                    ](
                        actor_grad_output_tensor,
                        kl_divergences_tensor,
                        actor_output_mb_tensor,
                        mb_old_log_probs_tensor,
                        mb_advantages_tensor,
                        mb_actions_tensor,
                        Scalar[dtype](self.clip_epsilon),
                        Scalar[dtype](current_entropy_coef),
                        MINIBATCH,
                        grid_dim=(MINIBATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    # Sync only if we need to check KL divergence on CPU
                    if self.target_kl > 0.0:
                        ctx.synchronize()
                    actor_grad_kernel_ns += perf_counter_ns() - actor_grad_start

                    # KL divergence early stopping
                    if self.target_kl > 0.0:
                        ctx.enqueue_copy(
                            kl_divergences_host, kl_divergences_buf
                        )
                        ctx.synchronize()

                        var kl_sum = Scalar[dtype](0.0)
                        for i in range(MINIBATCH):
                            kl_sum += kl_divergences_host[i]
                        var mean_kl = Float64(kl_sum) / Float64(MINIBATCH)

                        if mean_kl > self.target_kl:
                            kl_early_stop = True
                            if verbose:
                                print(
                                    "    KL early stop at epoch",
                                    epoch,
                                    "minibatch",
                                    mb_idx,
                                    "| KL:",
                                    String(mean_kl)[:7],
                                )
                            break

                    # Actor backward pass
                    var actor_bwd_start = perf_counter_ns()
                    self.actor.model.backward_gpu_ws[MINIBATCH](
                        ctx,
                        actor_grad_input_buf,
                        actor_grad_output_buf,
                        actor_params_buf,
                        actor_cache_buf,
                        actor_grads_buf,
                        actor_minibatch_workspace_buf,
                    )
                    # No sync needed - grad clip uses backward output
                    actor_backward_ns += perf_counter_ns() - actor_bwd_start

                    # Gradient clipping for actor (fully fused, 2 kernels)
                    var actor_clip_start = perf_counter_ns()
                    if self.max_grad_norm > 0.0:
                        # Step 1: Compute partial sums of squared gradients
                        ctx.enqueue_function[
                            actor_grad_norm_wrapper, actor_grad_norm_wrapper
                        ](
                            actor_grad_partial_sums_tensor,
                            actor_grads_tensor,
                            grid_dim=(ACTOR_GRAD_BLOCKS,),
                            block_dim=(TPB,),
                        )
                        # Step 2: Reduce partial sums AND apply scale (fused, multi-block)
                        ctx.enqueue_function[
                            actor_reduce_apply_fused_wrapper,
                            actor_reduce_apply_fused_wrapper,
                        ](
                            actor_grads_tensor,
                            actor_grad_partial_sums_tensor,
                            Scalar[dtype](self.max_grad_norm),
                            grid_dim=(ACTOR_GRAD_BLOCKS,),
                            block_dim=(TPB,),
                        )
                        # No sync needed - optimizer uses clipped grads
                    actor_grad_clip_ns += perf_counter_ns() - actor_clip_start

                    # Actor optimizer step
                    var actor_optim_start = perf_counter_ns()
                    self.actor.optimizer.step_gpu[ACTOR_PARAMS](
                        ctx,
                        actor_params_buf,
                        actor_grads_buf,
                        actor_state_buf,
                    )

                    # Clamp log_std params to prevent drift to extreme values
                    # log_std is at: STOCHASTIC_ACTOR_OFFSET + HIDDEN * ACTIONS + ACTIONS
                    comptime LOG_STD_OFFSET_IN_ACTOR = (
                        Self.OBS * Self.HIDDEN
                        + Self.HIDDEN  # Linear 1
                        + Self.HIDDEN * Self.HIDDEN
                        + Self.HIDDEN  # Linear 2
                        + Self.HIDDEN * Self.ACTIONS
                        + Self.ACTIONS  # mean head (W + b)
                    )
                    var actor_params_for_clamp = LayoutTensor[
                        dtype, Layout.row_major(ACTOR_PARAMS), MutAnyOrigin
                    ](actor_params_buf.unsafe_ptr())
                    comptime clamp_log_std_wrapper = clamp_log_std_params_kernel[
                        dtype, ACTOR_PARAMS, LOG_STD_OFFSET_IN_ACTOR, Self.ACTIONS
                    ]
                    ctx.enqueue_function[clamp_log_std_wrapper, clamp_log_std_wrapper](
                        actor_params_for_clamp,
                        grid_dim=(1,),  # Only ACTION_DIM params to clamp
                        block_dim=(Self.ACTIONS,),
                    )
                    # No sync needed - critic uses completely different buffers
                    actor_optim_ns += perf_counter_ns() - actor_optim_start

                    # Train critic - forward pass
                    var critic_fwd_start = perf_counter_ns()
                    ctx.enqueue_memset(critic_grads_buf, 0)
                    self.critic.model.forward_gpu_ws[MINIBATCH](
                        ctx,
                        critic_values_buf,
                        mb_obs_buf,
                        critic_params_buf,
                        critic_cache_buf,
                        critic_minibatch_workspace_buf,
                    )
                    # No sync needed - grad kernel uses forward output
                    critic_forward_ns += perf_counter_ns() - critic_fwd_start

                    # Critic grad kernel
                    var critic_grad_start = perf_counter_ns()

                    @parameter
                    if Self.clip_value:
                        ctx.enqueue_function[
                            critic_grad_clipped_wrapper,
                            critic_grad_clipped_wrapper,
                        ](
                            critic_grad_output_tensor,
                            critic_values_tensor,
                            mb_returns_tensor,
                            mb_old_values_tensor,
                            Scalar[dtype](self.clip_epsilon),
                            MINIBATCH,
                            grid_dim=(MINIBATCH_BLOCKS,),
                            block_dim=(TPB,),
                        )
                    else:
                        ctx.enqueue_function[
                            critic_grad_wrapper, critic_grad_wrapper
                        ](
                            critic_grad_output_tensor,
                            critic_values_tensor,
                            mb_returns_tensor,
                            MINIBATCH,
                            grid_dim=(MINIBATCH_BLOCKS,),
                            block_dim=(TPB,),
                        )
                    # No sync needed - backward uses grad output
                    critic_grad_kernel_ns += (
                        perf_counter_ns() - critic_grad_start
                    )

                    # Critic backward pass
                    var critic_bwd_start = perf_counter_ns()
                    self.critic.model.backward_gpu_ws[MINIBATCH](
                        ctx,
                        critic_grad_input_buf,
                        critic_grad_output_buf,
                        critic_params_buf,
                        critic_cache_buf,
                        critic_grads_buf,
                        critic_minibatch_workspace_buf,
                    )
                    # No sync needed - grad clip uses backward output
                    critic_backward_ns += perf_counter_ns() - critic_bwd_start

                    # Gradient clipping for critic (fully fused, 2 kernels)
                    var critic_clip_start = perf_counter_ns()
                    if self.max_grad_norm > 0.0:
                        # Step 1: Compute partial sums of squared gradients
                        ctx.enqueue_function[
                            critic_grad_norm_wrapper, critic_grad_norm_wrapper
                        ](
                            critic_grad_partial_sums_tensor,
                            critic_grads_tensor,
                            grid_dim=(CRITIC_GRAD_BLOCKS,),
                            block_dim=(TPB,),
                        )
                        # Step 2: Reduce partial sums AND apply scale (fused, multi-block)
                        ctx.enqueue_function[
                            critic_reduce_apply_fused_wrapper,
                            critic_reduce_apply_fused_wrapper,
                        ](
                            critic_grads_tensor,
                            critic_grad_partial_sums_tensor,
                            Scalar[dtype](self.max_grad_norm),
                            grid_dim=(CRITIC_GRAD_BLOCKS,),
                            block_dim=(TPB,),
                        )
                        # No sync needed - optimizer uses clipped grads
                    critic_grad_clip_ns += perf_counter_ns() - critic_clip_start

                    # Critic optimizer step
                    var critic_optim_start = perf_counter_ns()
                    self.critic.optimizer.step_gpu[CRITIC_PARAMS](
                        ctx,
                        critic_params_buf,
                        critic_grads_buf,
                        critic_state_buf,
                    )
                    # Sync at end of minibatch to ensure all GPU work completes
                    # before next iteration (params must be updated for next forward)
                    ctx.synchronize()
                    critic_optim_ns += perf_counter_ns() - critic_optim_start

            var phase3_end = perf_counter_ns()

            # Update timing accumulators
            total_phase1_ns += UInt(phase1_end - phase1_start)
            total_phase2_ns += UInt(phase2_end - phase2_start)
            total_phase3_ns += UInt(phase3_end - phase3_start)
            total_shuffle_ns += shuffle_time_ns
            total_indices_ns += indices_copy_time_ns
            total_gather_ns += gather_time_ns
            # Fine-grained actor timers
            total_actor_forward_ns += actor_forward_ns
            total_actor_grad_kernel_ns += actor_grad_kernel_ns
            total_actor_backward_ns += actor_backward_ns
            total_actor_grad_clip_ns += actor_grad_clip_ns
            total_actor_optim_ns += actor_optim_ns
            # Fine-grained critic timers
            total_critic_forward_ns += critic_forward_ns
            total_critic_grad_kernel_ns += critic_grad_kernel_ns
            total_critic_backward_ns += critic_backward_ns
            total_critic_grad_clip_ns += critic_grad_clip_ns
            total_critic_optim_ns += critic_optim_ns

            # Print progress
            if verbose and rollout_count % print_every == 0:
                var recent_mean = metrics.mean_reward_last_n(100)
                print(
                    "Rollout",
                    rollout_count,
                    "| Episodes:",
                    completed_episodes,
                    "| Mean reward (last 100):",
                    String(recent_mean)[:8],
                )

        # Copy final parameters back to CPU
        self.actor.copy_params_from_device(ctx, actor_params_buf)
        self.actor.copy_state_from_device(ctx, actor_state_buf)
        self.critic.copy_params_from_device(ctx, critic_params_buf)
        self.critic.copy_state_from_device(ctx, critic_state_buf)
        ctx.synchronize()

        self.train_step_count += total_steps

        # Final checkpoint save (ensures trained weights are always saved)
        if self.checkpoint_path != "":
            self.save_checkpoint(self.checkpoint_path)
            if verbose:
                print("  [Final checkpoint saved to", self.checkpoint_path, "]")

        if verbose:
            print("-" * 60)
            print("Training Complete!")
            print("Total episodes:", completed_episodes)
            print("Total steps:", total_steps)
            print("Total rollouts:", rollout_count)
            print("-" * 60)
            print("Timing breakdown:")
            print(
                "  Phase 1 (data collection):",
                String(Float64(total_phase1_ns) / 1e6)[:8],
                "ms",
            )
            print(
                "  Phase 2 (GAE computation):",
                String(Float64(total_phase2_ns) / 1e6)[:8],
                "ms",
            )
            print(
                "  Phase 3 (training):",
                String(Float64(total_phase3_ns) / 1e6)[:8],
                "ms",
            )
            print("-" * 60)

        return metrics^

    # =========================================================================
    # GPU Training with GPU Environments (Fully GPU)
    # =========================================================================

    fn train_gpu[
        EnvType: GPUContinuousEnv
    ](
        mut self,
        ctx: DeviceContext,
        num_episodes: Int,
        verbose: Bool = False,
        print_every: Int = 10,
    ) raises -> TrainingMetrics:
        """Train PPO on GPU with GPU-native continuous action environments.

        This fully GPU implementation runs both the neural networks AND the
        environment physics on GPU for maximum throughput.

        Args:
            ctx: GPU device context.
            num_episodes: Target number of episodes to complete.
            verbose: Whether to print progress.
            print_every: Print progress every N rollouts.

        Returns:
            TrainingMetrics with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="Deep PPO Continuous (GPU)",
            environment_name="GPU Environment",
        )

        # =====================================================================
        # Compile-time constants for buffer sizes
        # =====================================================================
        comptime ACTOR_PARAMS = Self.ACTOR_PARAM_SIZE
        comptime CRITIC_PARAMS = Self.CRITIC_PARAM_SIZE
        comptime ACTOR_STATE = ACTOR_PARAMS * 2  # Adam: 2 states per param
        comptime CRITIC_STATE = CRITIC_PARAMS * 2

        comptime ENV_OBS_SIZE = Self.n_envs * Self.OBS
        comptime ENV_ACTION_SIZE = Self.n_envs * Self.ACTIONS
        comptime ENV_ACTOR_OUT_SIZE = Self.n_envs * Self.ACTOR_OUT
        # Full environment state size
        comptime ENV_STATE_SIZE = Self.n_envs * EnvType.STATE_SIZE
        comptime ROLLOUT_TOTAL = Self.TOTAL_ROLLOUT_SIZE
        comptime ROLLOUT_OBS_SIZE = ROLLOUT_TOTAL * Self.OBS
        comptime ROLLOUT_ACTION_SIZE = ROLLOUT_TOTAL * Self.ACTIONS

        comptime MINIBATCH = Self.GPU_MINIBATCH
        comptime MINIBATCH_OBS_SIZE = MINIBATCH * Self.OBS
        comptime MINIBATCH_ACTION_SIZE = MINIBATCH * Self.ACTIONS
        comptime MINIBATCH_ACTOR_OUT_SIZE = MINIBATCH * Self.ACTOR_OUT
        comptime MINIBATCH_CACHE_ACTOR = MINIBATCH * Self.ACTOR_CACHE
        comptime MINIBATCH_CACHE_CRITIC = MINIBATCH * Self.CRITIC_CACHE

        comptime ENV_BLOCKS = (Self.n_envs + TPB - 1) // TPB
        comptime MINIBATCH_BLOCKS = (MINIBATCH + TPB - 1) // TPB
        comptime ROLLOUT_BLOCKS = (ROLLOUT_TOTAL + TPB - 1) // TPB

        # Workspace sizes for forward passes
        comptime WORKSPACE_PER_SAMPLE = 4 * Self.HIDDEN
        comptime ENV_WORKSPACE_SIZE = Self.n_envs * WORKSPACE_PER_SAMPLE
        comptime MINIBATCH_WORKSPACE_SIZE = MINIBATCH * WORKSPACE_PER_SAMPLE

        # =====================================================================
        # Network parameter buffers (GPU)
        # =====================================================================
        var actor_params_buf = ctx.enqueue_create_buffer[dtype](ACTOR_PARAMS)
        var actor_grads_buf = ctx.enqueue_create_buffer[dtype](ACTOR_PARAMS)
        var actor_state_buf = ctx.enqueue_create_buffer[dtype](ACTOR_STATE)

        var critic_params_buf = ctx.enqueue_create_buffer[dtype](CRITIC_PARAMS)
        var critic_grads_buf = ctx.enqueue_create_buffer[dtype](CRITIC_PARAMS)
        var critic_state_buf = ctx.enqueue_create_buffer[dtype](CRITIC_STATE)

        # Pre-allocated workspace buffers
        var actor_env_workspace_buf = ctx.enqueue_create_buffer[dtype](
            ENV_WORKSPACE_SIZE
        )
        var critic_env_workspace_buf = ctx.enqueue_create_buffer[dtype](
            ENV_WORKSPACE_SIZE
        )
        var actor_minibatch_workspace_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_WORKSPACE_SIZE
        )
        var critic_minibatch_workspace_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_WORKSPACE_SIZE
        )

        # =====================================================================
        # Environment buffers (n_envs parallel environments)
        # =====================================================================
        # Full state buffer for environment
        var states_buf = ctx.enqueue_create_buffer[dtype](ENV_STATE_SIZE)
        # Observation buffer for neural network input
        var obs_buf = ctx.enqueue_create_buffer[dtype](ENV_OBS_SIZE)
        var rewards_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var dones_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var actions_buf = ctx.enqueue_create_buffer[dtype](ENV_ACTION_SIZE)
        var values_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var log_probs_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var actor_output_buf = ctx.enqueue_create_buffer[dtype](
            ENV_ACTOR_OUT_SIZE
        )

        # Episode tracking buffers
        var episode_rewards_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var episode_steps_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var completed_rewards_buf = ctx.enqueue_create_buffer[dtype](
            Self.n_envs
        )
        var completed_steps_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var completed_mask_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)

        # Host buffers for episode tracking
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
        # Rollout buffers
        # =====================================================================
        var rollout_obs_buf = ctx.enqueue_create_buffer[dtype](ROLLOUT_OBS_SIZE)
        var rollout_actions_buf = ctx.enqueue_create_buffer[dtype](
            ROLLOUT_ACTION_SIZE
        )
        var rollout_rewards_buf = ctx.enqueue_create_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var rollout_values_buf = ctx.enqueue_create_buffer[dtype](ROLLOUT_TOTAL)
        var rollout_log_probs_buf = ctx.enqueue_create_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var rollout_dones_buf = ctx.enqueue_create_buffer[dtype](ROLLOUT_TOTAL)

        # Advantages and returns
        var advantages_buf = ctx.enqueue_create_buffer[dtype](ROLLOUT_TOTAL)
        var returns_buf = ctx.enqueue_create_buffer[dtype](ROLLOUT_TOTAL)

        # Host buffers for GAE computation
        var rollout_rewards_host = ctx.enqueue_create_host_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var rollout_values_host = ctx.enqueue_create_host_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var rollout_dones_host = ctx.enqueue_create_host_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var advantages_host = ctx.enqueue_create_host_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var returns_host = ctx.enqueue_create_host_buffer[dtype](ROLLOUT_TOTAL)
        var bootstrap_values_host = ctx.enqueue_create_host_buffer[dtype](
            Self.n_envs
        )

        # =====================================================================
        # Minibatch buffers (for training)
        # =====================================================================
        var mb_obs_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH_OBS_SIZE)
        var mb_actions_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_ACTION_SIZE
        )
        var mb_advantages_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var mb_returns_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var mb_old_log_probs_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var mb_old_values_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var mb_indices_buf = ctx.enqueue_create_buffer[DType.int32](MINIBATCH)
        var mb_indices_host = ctx.enqueue_create_host_buffer[DType.int32](
            MINIBATCH
        )

        # Training workspace
        var actor_output_mb_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_ACTOR_OUT_SIZE
        )
        var actor_cache_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_CACHE_ACTOR
        )
        var actor_grad_output_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_ACTOR_OUT_SIZE
        )
        var actor_grad_input_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_OBS_SIZE
        )

        var critic_values_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var critic_cache_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_CACHE_CRITIC
        )
        var critic_grad_output_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var critic_grad_input_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_OBS_SIZE
        )

        # KL divergence and gradient clipping buffers
        var kl_divergences_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var kl_divergences_host = ctx.enqueue_create_host_buffer[dtype](
            MINIBATCH
        )
        var mb_advantages_host = ctx.enqueue_create_host_buffer[dtype](
            MINIBATCH
        )

        comptime ACTOR_GRAD_BLOCKS = (ACTOR_PARAMS + TPB - 1) // TPB
        comptime CRITIC_GRAD_BLOCKS = (CRITIC_PARAMS + TPB - 1) // TPB
        var actor_grad_partial_sums_buf = ctx.enqueue_create_buffer[dtype](
            ACTOR_GRAD_BLOCKS
        )
        var critic_grad_partial_sums_buf = ctx.enqueue_create_buffer[dtype](
            CRITIC_GRAD_BLOCKS
        )
        var actor_grad_partial_sums_host = ctx.enqueue_create_host_buffer[
            dtype
        ](ACTOR_GRAD_BLOCKS)
        var critic_grad_partial_sums_host = ctx.enqueue_create_host_buffer[
            dtype
        ](CRITIC_GRAD_BLOCKS)

        # =====================================================================
        # Initialize network parameters on GPU
        # =====================================================================
        self.actor.copy_params_to_device(ctx, actor_params_buf)
        self.actor.copy_state_to_device(ctx, actor_state_buf)
        self.critic.copy_params_to_device(ctx, critic_params_buf)
        self.critic.copy_state_to_device(ctx, critic_state_buf)

        # =====================================================================
        # Create LayoutTensor views
        # =====================================================================
        var states_tensor = LayoutTensor[
            dtype,
            Layout.row_major(Self.n_envs, EnvType.STATE_SIZE),
            MutAnyOrigin,
        ](states_buf.unsafe_ptr())
        var obs_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var rewards_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](rewards_buf.unsafe_ptr())
        var dones_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](dones_buf.unsafe_ptr())
        var actions_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs, Self.ACTIONS), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var actor_output_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs, Self.ACTOR_OUT), MutAnyOrigin
        ](actor_output_buf.unsafe_ptr())
        var log_probs_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](log_probs_buf.unsafe_ptr())

        var episode_rewards_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](episode_rewards_buf.unsafe_ptr())
        var episode_steps_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](episode_steps_buf.unsafe_ptr())
        var completed_rewards_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](completed_rewards_buf.unsafe_ptr())
        var completed_steps_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](completed_steps_buf.unsafe_ptr())
        var completed_mask_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](completed_mask_buf.unsafe_ptr())

        var mb_obs_tensor = LayoutTensor[
            dtype,
            Layout.row_major(MINIBATCH, Self.OBS),
            MutAnyOrigin,
        ](mb_obs_buf.unsafe_ptr())
        var mb_actions_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, Self.ACTIONS), MutAnyOrigin
        ](mb_actions_buf.unsafe_ptr())
        var mb_advantages_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_advantages_buf.unsafe_ptr())
        var mb_returns_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_returns_buf.unsafe_ptr())
        var mb_old_log_probs_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_old_log_probs_buf.unsafe_ptr())
        var mb_old_values_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_old_values_buf.unsafe_ptr())
        var rollout_obs_tensor = LayoutTensor[
            dtype,
            Layout.row_major(ROLLOUT_TOTAL, Self.OBS),
            MutAnyOrigin,
        ](rollout_obs_buf.unsafe_ptr())
        var rollout_actions_tensor = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL, Self.ACTIONS), MutAnyOrigin
        ](rollout_actions_buf.unsafe_ptr())
        var advantages_tensor = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](advantages_buf.unsafe_ptr())
        var returns_tensor = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](returns_buf.unsafe_ptr())
        var rollout_log_probs_tensor = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](rollout_log_probs_buf.unsafe_ptr())
        var rollout_values_tensor = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](rollout_values_buf.unsafe_ptr())
        var mb_indices_tensor = LayoutTensor[
            DType.int32, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_indices_buf.unsafe_ptr())

        var actor_output_mb_tensor = LayoutTensor[
            dtype,
            Layout.row_major(MINIBATCH, Self.ACTOR_OUT),
            MutAnyOrigin,
        ](actor_output_mb_buf.unsafe_ptr())
        var actor_grad_output_tensor = LayoutTensor[
            dtype,
            Layout.row_major(MINIBATCH, Self.ACTOR_OUT),
            MutAnyOrigin,
        ](actor_grad_output_buf.unsafe_ptr())
        var critic_values_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, 1), MutAnyOrigin
        ](critic_values_buf.unsafe_ptr())
        var critic_grad_output_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, 1), MutAnyOrigin
        ](critic_grad_output_buf.unsafe_ptr())

        var kl_divergences_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](kl_divergences_buf.unsafe_ptr())

        var actor_grads_tensor = LayoutTensor[
            dtype, Layout.row_major(ACTOR_PARAMS), MutAnyOrigin
        ](actor_grads_buf.unsafe_ptr())
        var critic_grads_tensor = LayoutTensor[
            dtype, Layout.row_major(CRITIC_PARAMS), MutAnyOrigin
        ](critic_grads_buf.unsafe_ptr())
        var actor_grad_partial_sums_tensor = LayoutTensor[
            dtype, Layout.row_major(ACTOR_GRAD_BLOCKS), MutAnyOrigin
        ](actor_grad_partial_sums_buf.unsafe_ptr())
        var critic_grad_partial_sums_tensor = LayoutTensor[
            dtype, Layout.row_major(CRITIC_GRAD_BLOCKS), MutAnyOrigin
        ](critic_grad_partial_sums_buf.unsafe_ptr())

        # =====================================================================
        # Define kernel wrappers
        # =====================================================================
        comptime extract_obs_wrapper = _extract_obs_from_state_continuous_kernel[
            dtype, Self.n_envs, EnvType.STATE_SIZE, Self.OBS
        ]

        comptime accum_rewards_wrapper = accumulate_rewards_kernel[
            dtype, Self.n_envs
        ]
        comptime incr_steps_wrapper = increment_steps_kernel[dtype, Self.n_envs]
        comptime extract_completed_wrapper = extract_completed_episodes_kernel[
            dtype, Self.n_envs
        ]
        comptime reset_tracking_wrapper = selective_reset_tracking_kernel[
            dtype, Self.n_envs
        ]

        comptime sample_actions_wrapper = _sample_continuous_actions_kernel[
            dtype, Self.n_envs, Self.ACTIONS
        ]
        comptime store_pre_step_wrapper = _store_continuous_pre_step_kernel[
            dtype, Self.n_envs, Self.OBS, Self.ACTIONS
        ]
        comptime store_post_step_wrapper = _store_post_step_kernel[
            dtype, Self.n_envs
        ]
        comptime gather_wrapper = ppo_continuous_gather_minibatch_kernel[
            dtype, MINIBATCH, Self.OBS, Self.ACTIONS, ROLLOUT_TOTAL
        ]
        comptime actor_grad_wrapper = ppo_continuous_actor_grad_kernel[
            dtype, MINIBATCH, Self.ACTIONS
        ]
        comptime critic_grad_wrapper = ppo_critic_grad_kernel[dtype, MINIBATCH]
        comptime critic_grad_clipped_wrapper = ppo_critic_grad_clipped_kernel[
            dtype, MINIBATCH
        ]
        comptime normalize_advantages_wrapper = normalize_advantages_kernel[
            dtype, MINIBATCH
        ]
        # Fused GPU-only normalization (no CPU roundtrip)
        comptime normalize_advantages_fused_wrapper = normalize_advantages_fused_kernel[
            dtype, MINIBATCH, TPB
        ]

        comptime actor_grad_norm_wrapper = gradient_norm_kernel[
            dtype, ACTOR_PARAMS, ACTOR_GRAD_BLOCKS, TPB
        ]
        comptime critic_grad_norm_wrapper = gradient_norm_kernel[
            dtype, CRITIC_PARAMS, CRITIC_GRAD_BLOCKS, TPB
        ]
        comptime actor_grad_clip_wrapper = gradient_clip_kernel[
            dtype, ACTOR_PARAMS
        ]
        comptime critic_grad_clip_wrapper = gradient_clip_kernel[
            dtype, CRITIC_PARAMS
        ]

        # Fully fused gradient clipping wrappers (2 kernels instead of 3)
        # Each block redundantly reduces partial_sums and applies scale
        comptime actor_reduce_apply_fused_wrapper = gradient_reduce_apply_fused_kernel[
            dtype, ACTOR_PARAMS, ACTOR_GRAD_BLOCKS, TPB
        ]
        comptime critic_reduce_apply_fused_wrapper = gradient_reduce_apply_fused_kernel[
            dtype, CRITIC_PARAMS, CRITIC_GRAD_BLOCKS, TPB
        ]

        # =====================================================================
        # Initialize episode tracking and environments
        # =====================================================================
        ctx.enqueue_memset(episode_rewards_buf, 0)
        ctx.enqueue_memset(episode_steps_buf, 0)

        # Reset all environments
        EnvType.reset_kernel_gpu[Self.n_envs, EnvType.STATE_SIZE](
            ctx, states_buf
        )
        ctx.synchronize()

        # Extract observations from state buffer
        ctx.enqueue_function[extract_obs_wrapper, extract_obs_wrapper](
            obs_tensor,
            states_tensor,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )
        ctx.synchronize()

        # =====================================================================
        # Training state
        # =====================================================================
        var completed_episodes = 0
        var total_steps = 0
        var rollout_count = 0

        # Annealing target
        var annealing_target_steps = self.target_total_steps
        if annealing_target_steps == 0:
            annealing_target_steps = num_episodes * 500

        var initial_actor_lr = self.actor_lr
        var initial_critic_lr = self.critic_lr
        var initial_entropy_coef = self.entropy_coef

        # Timing accumulators
        var total_phase1_ns: UInt = 0
        var total_phase2_ns: UInt = 0
        var total_phase3_ns: UInt = 0

        # Phase 3 sub-timers
        var total_shuffle_ns: UInt = 0
        var total_indices_ns: UInt = 0
        var total_gather_ns: UInt = 0
        # Fine-grained actor timers
        var total_actor_forward_ns: UInt = 0
        var total_actor_grad_kernel_ns: UInt = 0
        var total_actor_backward_ns: UInt = 0
        var total_actor_grad_clip_ns: UInt = 0
        var total_actor_optim_ns: UInt = 0
        # Fine-grained critic timers
        var total_critic_forward_ns: UInt = 0
        var total_critic_grad_kernel_ns: UInt = 0
        var total_critic_backward_ns: UInt = 0
        var total_critic_grad_clip_ns: UInt = 0
        var total_critic_optim_ns: UInt = 0

        # =====================================================================
        # Main Training Loop
        # =====================================================================

        while completed_episodes < num_episodes:
            var rollout_start_episodes = completed_episodes
            rollout_count += 1

            # =================================================================
            # Phase 1: Collect rollout (rollout_len steps across n_envs envs)
            # =================================================================
            var phase1_start = perf_counter_ns()

            for t in range(Self.rollout_len):
                var rng_seed = UInt32(total_steps * 2654435761 + t * 7919)

                # Forward actor on GPU to get mean and log_std
                self.actor.model.forward_gpu_no_cache_ws[Self.n_envs](
                    ctx,
                    actor_output_buf,
                    obs_buf,
                    actor_params_buf,
                    actor_env_workspace_buf,
                )

                # Forward critic on GPU to get values
                self.critic.model.forward_gpu_no_cache_ws[Self.n_envs](
                    ctx,
                    values_buf,
                    obs_buf,
                    critic_params_buf,
                    critic_env_workspace_buf,
                )

                # Sample continuous actions on GPU (unbounded Gaussian)
                ctx.enqueue_function[
                    sample_actions_wrapper, sample_actions_wrapper
                ](
                    actor_output_tensor,
                    actions_tensor,
                    log_probs_tensor,
                    Scalar[DType.uint32](rng_seed),
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Store pre-step data to rollout buffer (using actions directly)
                var t_offset = t * Self.n_envs

                var rollout_obs_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs, Self.OBS),
                    MutAnyOrigin,
                ](rollout_obs_buf.unsafe_ptr() + t_offset * Self.OBS)
                var rollout_actions_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs, Self.ACTIONS),
                    MutAnyOrigin,
                ](rollout_actions_buf.unsafe_ptr() + t_offset * Self.ACTIONS)
                var rollout_log_probs_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](rollout_log_probs_buf.unsafe_ptr() + t_offset)
                var rollout_values_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](rollout_values_buf.unsafe_ptr() + t_offset)

                var values_tensor_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](values_buf.unsafe_ptr())

                ctx.enqueue_function[
                    store_pre_step_wrapper, store_pre_step_wrapper
                ](
                    rollout_obs_t,
                    rollout_actions_t,
                    rollout_log_probs_t,
                    rollout_values_t,
                    obs_tensor,
                    actions_tensor,
                    log_probs_tensor,
                    values_tensor_t,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Step all environments on GPU with continuous actions
                var env_step_seed = UInt64(total_steps * 1103515245 + t * 12345)
                EnvType.step_kernel_gpu[
                    Self.n_envs, EnvType.STATE_SIZE, Self.OBS, Self.ACTIONS
                ](
                    ctx,
                    states_buf,
                    actions_buf,
                    rewards_buf,
                    dones_buf,
                    obs_buf,
                    env_step_seed,
                )

                # Store rewards and dones
                var rollout_rewards_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](rollout_rewards_buf.unsafe_ptr() + t_offset)
                var rollout_dones_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](rollout_dones_buf.unsafe_ptr() + t_offset)

                ctx.enqueue_function[
                    store_post_step_wrapper, store_post_step_wrapper
                ](
                    rollout_rewards_t,
                    rollout_dones_t,
                    rewards_tensor,
                    dones_tensor,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Accumulate episode rewards and steps
                ctx.enqueue_function[
                    accum_rewards_wrapper, accum_rewards_wrapper
                ](
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

                total_steps += Self.n_envs

                # Extract completed episodes
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

                # Copy to CPU and process
                ctx.enqueue_copy(completed_rewards_host, completed_rewards_buf)
                ctx.enqueue_copy(completed_steps_host, completed_steps_buf)
                ctx.enqueue_copy(completed_mask_host, completed_mask_buf)
                ctx.synchronize()

                # Log completed episodes
                for i in range(Self.n_envs):
                    if Float64(completed_mask_host[i]) > 0.5:
                        var ep_reward = Float64(completed_rewards_host[i])
                        var ep_steps = Int(completed_steps_host[i])
                        metrics.log_episode(
                            completed_episodes, ep_reward, ep_steps, 0.0
                        )
                        completed_episodes += 1

                        # Auto-checkpoint
                        if (
                            self.checkpoint_every > 0
                            and self.checkpoint_path != ""
                            and completed_episodes % self.checkpoint_every == 0
                        ):
                            self.actor.copy_params_from_device(
                                ctx, actor_params_buf
                            )
                            self.actor.copy_state_from_device(
                                ctx, actor_state_buf
                            )
                            self.critic.copy_params_from_device(
                                ctx, critic_params_buf
                            )
                            self.critic.copy_state_from_device(
                                ctx, critic_state_buf
                            )
                            ctx.synchronize()
                            self.save_checkpoint(self.checkpoint_path)
                            if verbose:
                                print(
                                    "  [Checkpoint saved at episode",
                                    completed_episodes,
                                    "]",
                                )

                # Reset episode tracking for done environments
                ctx.enqueue_function[
                    reset_tracking_wrapper, reset_tracking_wrapper
                ](
                    dones_tensor,
                    episode_rewards_tensor,
                    episode_steps_tensor,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Auto-reset done environments
                EnvType.selective_reset_kernel_gpu[
                    Self.n_envs, EnvType.STATE_SIZE
                ](
                    ctx,
                    states_buf,
                    dones_buf,
                    UInt64(total_steps * 1013904223 + t * 2654435761),
                )

                # Extract observations from state buffer after selective reset
                ctx.enqueue_function[extract_obs_wrapper, extract_obs_wrapper](
                    obs_tensor,
                    states_tensor,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )

            # Early exit if we've reached target episodes
            if completed_episodes >= num_episodes:
                break

            var phase1_end = perf_counter_ns()

            # =================================================================
            # Phase 2: Compute GAE advantages on CPU
            # =================================================================
            var phase2_start = perf_counter_ns()

            # Get bootstrap values from final observations
            self.critic.model.forward_gpu_no_cache_ws[Self.n_envs](
                ctx,
                values_buf,
                obs_buf,
                critic_params_buf,
                critic_env_workspace_buf,
            )

            ctx.enqueue_copy(bootstrap_values_host, values_buf)

            # Copy rollout data to CPU
            ctx.enqueue_copy(rollout_rewards_host, rollout_rewards_buf)
            ctx.enqueue_copy(rollout_values_host, rollout_values_buf)
            ctx.enqueue_copy(rollout_dones_host, rollout_dones_buf)
            ctx.synchronize()

            # Reward normalization (CleanRL-style)
            # This prevents dense fuel penalties from dominating over sparse landing bonuses
            if self.normalize_rewards:
                # Update running statistics with the new rewards
                self.reward_rms.update(rollout_rewards_host, ROLLOUT_TOTAL)
                # Normalize rewards in-place
                self.reward_rms.normalize(rollout_rewards_host, ROLLOUT_TOTAL)

            # Compute GAE for each environment
            for env_idx in range(Self.n_envs):
                var gae = Scalar[dtype](0.0)
                var gae_decay = Scalar[dtype](self.gamma * self.gae_lambda)
                var bootstrap_val = Scalar[dtype](
                    bootstrap_values_host[env_idx]
                )

                for t in range(Self.rollout_len - 1, -1, -1):
                    var idx = t * Self.n_envs + env_idx
                    var reward = rollout_rewards_host[idx]
                    var value = rollout_values_host[idx]
                    var done = rollout_dones_host[idx]

                    var next_val: Scalar[dtype]
                    if t == Self.rollout_len - 1:
                        next_val = bootstrap_val
                    else:
                        var next_idx = (t + 1) * Self.n_envs + env_idx
                        next_val = rollout_values_host[next_idx]

                    if done > Scalar[dtype](0.5):
                        next_val = Scalar[dtype](0.0)
                        gae = Scalar[dtype](0.0)

                    var delta = (
                        reward + Scalar[dtype](self.gamma) * next_val - value
                    )
                    gae = delta + gae_decay * gae

                    advantages_host[idx] = gae
                    returns_host[idx] = gae + value

            # Normalize advantages
            if self.normalize_advantages:
                var mean = Scalar[dtype](0.0)
                var var_sum = Scalar[dtype](0.0)
                for i in range(ROLLOUT_TOTAL):
                    mean += advantages_host[i]
                mean /= Scalar[dtype](ROLLOUT_TOTAL)
                for i in range(ROLLOUT_TOTAL):
                    var diff = advantages_host[i] - mean
                    var_sum += diff * diff

                var variance = var_sum / Scalar[dtype](ROLLOUT_TOTAL)
                var std = sqrt(Float64(variance) + 1e-8)
                for i in range(ROLLOUT_TOTAL):
                    advantages_host[i] = (advantages_host[i] - mean) / (
                        Scalar[dtype](std) + Scalar[dtype](1e-8)
                    )

            # Copy advantages and returns to GPU
            ctx.enqueue_copy(advantages_buf, advantages_host)
            ctx.enqueue_copy(returns_buf, returns_host)
            ctx.synchronize()

            var phase2_end = perf_counter_ns()

            # =================================================================
            # Phase 3: Train actor and critic with minibatches (GPU)
            # =================================================================
            var phase3_start = perf_counter_ns()

            var progress = Float64(total_steps) / Float64(
                annealing_target_steps
            )
            if progress > 1.0:
                progress = 1.0

            var current_actor_lr = initial_actor_lr
            var current_critic_lr = initial_critic_lr
            var current_entropy_coef = initial_entropy_coef
            if self.anneal_lr:
                var lr_multiplier = 1.0 - progress
                current_actor_lr = initial_actor_lr * lr_multiplier
                current_critic_lr = initial_critic_lr * lr_multiplier
                self.actor.optimizer.lr = current_actor_lr
                self.critic.optimizer.lr = current_critic_lr

            if self.anneal_entropy:
                current_entropy_coef = initial_entropy_coef * (1.0 - progress)

            var kl_early_stop = False

            # Sub-timers for this rollout's phase 3
            var shuffle_time_ns: UInt = 0
            var indices_copy_time_ns: UInt = 0
            var gather_time_ns: UInt = 0
            # Fine-grained actor timers
            var actor_forward_ns: UInt = 0
            var actor_grad_kernel_ns: UInt = 0
            var actor_backward_ns: UInt = 0
            var actor_grad_clip_ns: UInt = 0
            var actor_optim_ns: UInt = 0
            # Fine-grained critic timers
            var critic_forward_ns: UInt = 0
            var critic_grad_kernel_ns: UInt = 0
            var critic_backward_ns: UInt = 0
            var critic_grad_clip_ns: UInt = 0
            var critic_optim_ns: UInt = 0

            for epoch in range(self.num_epochs):
                if kl_early_stop:
                    break

                # Generate shuffled indices
                var shuffle_start = perf_counter_ns()
                var indices_list = List[Int]()
                for i in range(ROLLOUT_TOTAL):
                    indices_list.append(i)

                for i in range(ROLLOUT_TOTAL - 1, 0, -1):
                    var j = Int(random_float64() * Float64(i + 1))
                    var temp = indices_list[i]
                    indices_list[i] = indices_list[j]
                    indices_list[j] = temp
                shuffle_time_ns += perf_counter_ns() - shuffle_start

                var num_minibatches = ROLLOUT_TOTAL // MINIBATCH
                for mb_idx in range(num_minibatches):
                    var start_idx = mb_idx * MINIBATCH

                    # Copy indices to host buffer
                    var indices_copy_start = perf_counter_ns()
                    for i in range(MINIBATCH):
                        mb_indices_host[i] = Int32(indices_list[start_idx + i])

                    ctx.enqueue_copy(mb_indices_buf, mb_indices_host)
                    indices_copy_time_ns += (
                        perf_counter_ns() - indices_copy_start
                    )

                    # Gather minibatch data
                    var gather_start = perf_counter_ns()
                    ctx.enqueue_function[gather_wrapper, gather_wrapper](
                        mb_obs_tensor,
                        mb_actions_tensor,
                        mb_advantages_tensor,
                        mb_returns_tensor,
                        mb_old_log_probs_tensor,
                        mb_old_values_tensor,
                        rollout_obs_tensor,
                        rollout_actions_tensor,
                        advantages_tensor,
                        returns_tensor,
                        rollout_log_probs_tensor,
                        rollout_values_tensor,
                        mb_indices_tensor,
                        MINIBATCH,
                        grid_dim=(MINIBATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    # No sync needed - GPU operations execute in order on same stream

                    # Per-minibatch advantage normalization (include in gather time)
                    # Uses fused GPU kernel - no CPU roundtrip needed
                    if self.norm_adv_per_minibatch:
                        ctx.enqueue_function[
                            normalize_advantages_fused_wrapper,
                            normalize_advantages_fused_wrapper,
                        ](
                            mb_advantages_tensor,
                            grid_dim=(1,),  # Single block for reduction
                            block_dim=(TPB,),
                        )
                        # No sync needed - next kernels will wait for this to complete
                    gather_time_ns += perf_counter_ns() - gather_start

                    # Train actor - forward pass
                    var actor_fwd_start = perf_counter_ns()
                    ctx.enqueue_memset(actor_grads_buf, 0)
                    self.actor.model.forward_gpu_ws[MINIBATCH](
                        ctx,
                        actor_output_mb_buf,
                        mb_obs_buf,
                        actor_params_buf,
                        actor_cache_buf,
                        actor_minibatch_workspace_buf,
                    )
                    # No sync needed - grad kernel uses forward output
                    actor_forward_ns += perf_counter_ns() - actor_fwd_start

                    # Actor grad kernel (ppo_continuous_actor_grad_kernel)
                    var actor_grad_start = perf_counter_ns()
                    ctx.enqueue_function[
                        actor_grad_wrapper, actor_grad_wrapper
                    ](
                        actor_grad_output_tensor,
                        kl_divergences_tensor,
                        actor_output_mb_tensor,
                        mb_old_log_probs_tensor,
                        mb_advantages_tensor,
                        mb_actions_tensor,
                        Scalar[dtype](self.clip_epsilon),
                        Scalar[dtype](current_entropy_coef),
                        MINIBATCH,
                        grid_dim=(MINIBATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    # Sync only if we need to check KL divergence on CPU
                    if self.target_kl > 0.0:
                        ctx.synchronize()
                    actor_grad_kernel_ns += perf_counter_ns() - actor_grad_start

                    # KL divergence early stopping
                    if self.target_kl > 0.0:
                        ctx.enqueue_copy(
                            kl_divergences_host, kl_divergences_buf
                        )
                        ctx.synchronize()

                        var kl_sum = Scalar[dtype](0.0)
                        for i in range(MINIBATCH):
                            kl_sum += kl_divergences_host[i]
                        var mean_kl = Float64(kl_sum) / Float64(MINIBATCH)

                        if mean_kl > self.target_kl:
                            kl_early_stop = True
                            if verbose:
                                print(
                                    "    KL early stop at epoch",
                                    epoch,
                                    "minibatch",
                                    mb_idx,
                                    "| KL:",
                                    String(mean_kl)[:7],
                                )
                            break

                    # Actor backward pass
                    var actor_bwd_start = perf_counter_ns()
                    self.actor.model.backward_gpu_ws[MINIBATCH](
                        ctx,
                        actor_grad_input_buf,
                        actor_grad_output_buf,
                        actor_params_buf,
                        actor_cache_buf,
                        actor_grads_buf,
                        actor_minibatch_workspace_buf,
                    )
                    # No sync needed - grad clip uses backward output
                    actor_backward_ns += perf_counter_ns() - actor_bwd_start

                    # Gradient clipping for actor (fully fused, 2 kernels)
                    var actor_clip_start = perf_counter_ns()
                    if self.max_grad_norm > 0.0:
                        # Step 1: Compute partial sums of squared gradients
                        ctx.enqueue_function[
                            actor_grad_norm_wrapper, actor_grad_norm_wrapper
                        ](
                            actor_grad_partial_sums_tensor,
                            actor_grads_tensor,
                            grid_dim=(ACTOR_GRAD_BLOCKS,),
                            block_dim=(TPB,),
                        )
                        # Step 2: Reduce partial sums AND apply scale (fused, multi-block)
                        ctx.enqueue_function[
                            actor_reduce_apply_fused_wrapper,
                            actor_reduce_apply_fused_wrapper,
                        ](
                            actor_grads_tensor,
                            actor_grad_partial_sums_tensor,
                            Scalar[dtype](self.max_grad_norm),
                            grid_dim=(ACTOR_GRAD_BLOCKS,),
                            block_dim=(TPB,),
                        )
                        # No sync needed - optimizer uses clipped grads
                    actor_grad_clip_ns += perf_counter_ns() - actor_clip_start

                    # Actor optimizer step
                    var actor_optim_start = perf_counter_ns()
                    self.actor.optimizer.step_gpu[ACTOR_PARAMS](
                        ctx,
                        actor_params_buf,
                        actor_grads_buf,
                        actor_state_buf,
                    )

                    # Clamp log_std params to prevent drift to extreme values
                    # log_std is at: STOCHASTIC_ACTOR_OFFSET + HIDDEN * ACTIONS + ACTIONS
                    comptime LOG_STD_OFFSET_IN_ACTOR = (
                        Self.OBS * Self.HIDDEN
                        + Self.HIDDEN  # Linear 1
                        + Self.HIDDEN * Self.HIDDEN
                        + Self.HIDDEN  # Linear 2
                        + Self.HIDDEN * Self.ACTIONS
                        + Self.ACTIONS  # mean head (W + b)
                    )
                    var actor_params_for_clamp = LayoutTensor[
                        dtype, Layout.row_major(ACTOR_PARAMS), MutAnyOrigin
                    ](actor_params_buf.unsafe_ptr())
                    comptime clamp_log_std_wrapper = clamp_log_std_params_kernel[
                        dtype, ACTOR_PARAMS, LOG_STD_OFFSET_IN_ACTOR, Self.ACTIONS
                    ]
                    ctx.enqueue_function[clamp_log_std_wrapper, clamp_log_std_wrapper](
                        actor_params_for_clamp,
                        grid_dim=(1,),  # Only ACTION_DIM params to clamp
                        block_dim=(Self.ACTIONS,),
                    )
                    # No sync needed - critic uses completely different buffers
                    actor_optim_ns += perf_counter_ns() - actor_optim_start

                    # Train critic - forward pass
                    var critic_fwd_start = perf_counter_ns()
                    ctx.enqueue_memset(critic_grads_buf, 0)
                    self.critic.model.forward_gpu_ws[MINIBATCH](
                        ctx,
                        critic_values_buf,
                        mb_obs_buf,
                        critic_params_buf,
                        critic_cache_buf,
                        critic_minibatch_workspace_buf,
                    )
                    # No sync needed - grad kernel uses forward output
                    critic_forward_ns += perf_counter_ns() - critic_fwd_start

                    # Critic grad kernel
                    var critic_grad_start = perf_counter_ns()

                    @parameter
                    if Self.clip_value:
                        ctx.enqueue_function[
                            critic_grad_clipped_wrapper,
                            critic_grad_clipped_wrapper,
                        ](
                            critic_grad_output_tensor,
                            critic_values_tensor,
                            mb_returns_tensor,
                            mb_old_values_tensor,
                            Scalar[dtype](self.clip_epsilon),
                            MINIBATCH,
                            grid_dim=(MINIBATCH_BLOCKS,),
                            block_dim=(TPB,),
                        )
                    else:
                        ctx.enqueue_function[
                            critic_grad_wrapper, critic_grad_wrapper
                        ](
                            critic_grad_output_tensor,
                            critic_values_tensor,
                            mb_returns_tensor,
                            MINIBATCH,
                            grid_dim=(MINIBATCH_BLOCKS,),
                            block_dim=(TPB,),
                        )
                    # No sync needed - backward uses grad output
                    critic_grad_kernel_ns += (
                        perf_counter_ns() - critic_grad_start
                    )

                    # Critic backward pass
                    var critic_bwd_start = perf_counter_ns()
                    self.critic.model.backward_gpu_ws[MINIBATCH](
                        ctx,
                        critic_grad_input_buf,
                        critic_grad_output_buf,
                        critic_params_buf,
                        critic_cache_buf,
                        critic_grads_buf,
                        critic_minibatch_workspace_buf,
                    )
                    # No sync needed - grad clip uses backward output
                    critic_backward_ns += perf_counter_ns() - critic_bwd_start

                    # Gradient clipping for critic (fully fused, 2 kernels)
                    var critic_clip_start = perf_counter_ns()
                    if self.max_grad_norm > 0.0:
                        # Step 1: Compute partial sums of squared gradients
                        ctx.enqueue_function[
                            critic_grad_norm_wrapper, critic_grad_norm_wrapper
                        ](
                            critic_grad_partial_sums_tensor,
                            critic_grads_tensor,
                            grid_dim=(CRITIC_GRAD_BLOCKS,),
                            block_dim=(TPB,),
                        )
                        # Step 2: Reduce partial sums AND apply scale (fused, multi-block)
                        ctx.enqueue_function[
                            critic_reduce_apply_fused_wrapper,
                            critic_reduce_apply_fused_wrapper,
                        ](
                            critic_grads_tensor,
                            critic_grad_partial_sums_tensor,
                            Scalar[dtype](self.max_grad_norm),
                            grid_dim=(CRITIC_GRAD_BLOCKS,),
                            block_dim=(TPB,),
                        )
                        # No sync needed - optimizer uses clipped grads
                    critic_grad_clip_ns += perf_counter_ns() - critic_clip_start

                    # Critic optimizer step
                    var critic_optim_start = perf_counter_ns()
                    self.critic.optimizer.step_gpu[CRITIC_PARAMS](
                        ctx,
                        critic_params_buf,
                        critic_grads_buf,
                        critic_state_buf,
                    )
                    # Sync at end of minibatch to ensure all GPU work completes
                    # before next iteration (params must be updated for next forward)
                    ctx.synchronize()
                    critic_optim_ns += perf_counter_ns() - critic_optim_start

            var phase3_end = perf_counter_ns()

            # Update timing accumulators
            total_phase1_ns += UInt(phase1_end - phase1_start)
            total_phase2_ns += UInt(phase2_end - phase2_start)
            total_phase3_ns += UInt(phase3_end - phase3_start)
            total_shuffle_ns += shuffle_time_ns
            total_indices_ns += indices_copy_time_ns
            total_gather_ns += gather_time_ns
            # Fine-grained actor timers
            total_actor_forward_ns += actor_forward_ns
            total_actor_grad_kernel_ns += actor_grad_kernel_ns
            total_actor_backward_ns += actor_backward_ns
            total_actor_grad_clip_ns += actor_grad_clip_ns
            total_actor_optim_ns += actor_optim_ns
            # Fine-grained critic timers
            total_critic_forward_ns += critic_forward_ns
            total_critic_grad_kernel_ns += critic_grad_kernel_ns
            total_critic_backward_ns += critic_backward_ns
            total_critic_grad_clip_ns += critic_grad_clip_ns
            total_critic_optim_ns += critic_optim_ns

            # Print progress
            if verbose and rollout_count % print_every == 0:
                var rollout_end_episodes = completed_episodes
                var episodes_this_rollout = (
                    rollout_end_episodes - rollout_start_episodes
                )
                var avg_reward = metrics.mean_reward_last_n(
                    min(100, completed_episodes)
                )
                print(
                    "Rollout",
                    rollout_count,
                    "| Episodes",
                    rollout_start_episodes + 1,
                    "-",
                    rollout_end_episodes,
                    "(+" + String(episodes_this_rollout) + ")",
                    "| Avg(100):",
                    String(avg_reward)[:7],
                    "| Steps:",
                    total_steps,
                )

        # Copy final parameters back to CPU
        self.actor.copy_params_from_device(ctx, actor_params_buf)
        self.actor.copy_state_from_device(ctx, actor_state_buf)
        self.critic.copy_params_from_device(ctx, critic_params_buf)
        self.critic.copy_state_from_device(ctx, critic_state_buf)
        ctx.synchronize()

        self.train_step_count += total_steps

        # Final checkpoint save (ensures trained weights are always saved)
        if self.checkpoint_path != "":
            self.save_checkpoint(self.checkpoint_path)
            if verbose:
                print("  [Final checkpoint saved to", self.checkpoint_path, "]")

        if verbose:
            var total_time_ns = (
                total_phase1_ns + total_phase2_ns + total_phase3_ns
            )
            var p1_pct = Float64(total_phase1_ns) / Float64(total_time_ns) * 100
            var p2_pct = Float64(total_phase2_ns) / Float64(total_time_ns) * 100
            var p3_pct = Float64(total_phase3_ns) / Float64(total_time_ns) * 100
            var total_actor_time_ns = (
                total_actor_forward_ns
                + total_actor_grad_kernel_ns
                + total_actor_backward_ns
                + total_actor_grad_clip_ns
                + total_actor_optim_ns
            )
            var total_critic_time_ns = (
                total_critic_forward_ns
                + total_critic_grad_kernel_ns
                + total_critic_backward_ns
                + total_critic_grad_clip_ns
                + total_critic_optim_ns
            )

            print()
            print("-" * 60)
            print(
                "Performance Summary (" + String(rollout_count) + " rollouts)"
            )
            print("-" * 60)
            print(
                "  Phase 1 (collect):  ",
                String(Float64(total_phase1_ns) / 1e9)[:6],
                "s (",
                String(p1_pct)[:4],
                "%)",
            )
            print(
                "  Phase 2 (GAE):      ",
                String(Float64(total_phase2_ns) / 1e9)[:6],
                "s (",
                String(p2_pct)[:4],
                "%)",
            )
            print(
                "  Phase 3 (train):    ",
                String(Float64(total_phase3_ns) / 1e9)[:6],
                "s (",
                String(p3_pct)[:4],
                "%)",
            )
            print()
            print("  Phase 3 breakdown:")
            print(
                "    Shuffle:            ",
                String(Float64(total_shuffle_ns) / 1e6)[:8],
                "ms",
            )
            print(
                "    Indices copy:       ",
                String(Float64(total_indices_ns) / 1e6)[:8],
                "ms",
            )
            print(
                "    Gather:             ",
                String(Float64(total_gather_ns) / 1e6)[:8],
                "ms",
            )
            print(
                "    Actor:              ",
                String(Float64(total_actor_time_ns) / 1e6)[:8],
                "ms",
            )
            print(
                "      forward:          ",
                String(Float64(total_actor_forward_ns) / 1e6)[:8],
                "ms",
            )
            print(
                "      grad kernel:      ",
                String(Float64(total_actor_grad_kernel_ns) / 1e6)[:8],
                "ms",
            )
            print(
                "      backward:         ",
                String(Float64(total_actor_backward_ns) / 1e6)[:8],
                "ms",
            )
            print(
                "      grad clip:        ",
                String(Float64(total_actor_grad_clip_ns) / 1e6)[:8],
                "ms",
            )
            print(
                "      optimizer:        ",
                String(Float64(total_actor_optim_ns) / 1e6)[:8],
                "ms",
            )
            print(
                "    Critic:             ",
                String(Float64(total_critic_time_ns) / 1e6)[:8],
                "ms",
            )
            print(
                "      forward:          ",
                String(Float64(total_critic_forward_ns) / 1e6)[:8],
                "ms",
            )
            print(
                "      grad kernel:      ",
                String(Float64(total_critic_grad_kernel_ns) / 1e6)[:8],
                "ms",
            )
            print(
                "      backward:         ",
                String(Float64(total_critic_backward_ns) / 1e6)[:8],
                "ms",
            )
            print(
                "      grad clip:        ",
                String(Float64(total_critic_grad_clip_ns) / 1e6)[:8],
                "ms",
            )
            print(
                "      optimizer:        ",
                String(Float64(total_critic_optim_ns) / 1e6)[:8],
                "ms",
            )
            print("-" * 60)
            print("Training Complete!")
            print("Total episodes:", completed_episodes)
            print("Total steps:", total_steps)
            print("-" * 60)

        return metrics^

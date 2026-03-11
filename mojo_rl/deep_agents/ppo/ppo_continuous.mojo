"""Deep PPO (Proximal Policy Optimization) Agent for Continuous Action Spaces.

This PPO implementation supports continuous action spaces using a Gaussian policy:
- Network wrapper from mojo_rl.nn.training for stateless model + params management
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

Architecture (CleanRL-style with Tanh activations):
- Actor: obs -> hidden (Tanh) -> hidden (Tanh) -> StochasticActor (mean + log_std)
- Critic: obs -> hidden (Tanh) -> hidden (Tanh) -> 1 (value)

Usage:
    from mojo_rl.deep_agents.ppo_continuous import DeepPPOContinuousAgent
    from mojo_rl.envs import CarRacingEnv

    var env = CarRacingEnv(continuous=True)
    var agent = DeepPPOContinuousAgent[13, 3, 256]()

    # Hybrid GPU+CPU training
    with DeviceContext() as ctx:
        var metrics = agent.train_gpu(ctx, env, num_episodes=1000)

Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
"""

from std.math import exp, log, sqrt, cos, tanh
from std.random import random_float64, seed
from std.time import perf_counter_ns
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype, TILE, TPB
from mojo_rl.nn import (
    Dense,
    DenseTanh,
    Sequential,
    StochasticActor,
)
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Xavier, Kaiming
from mojo_rl.nn.training import Network, NetworkState, GPUNetworkState
from mojo_rl.nn.checkpoint import (
    split_lines,
    find_section_start,
    save_checkpoint_file,
    read_checkpoint_file,
)
from mojo_rl.core import (
    TrainingMetrics,
    BoxDiscreteActionEnv,
    BoxContinuousActionEnv,
    GPUContinuousEnv,
    RenderableEnv,
    CurriculumScheduler,
    NoCurriculumScheduler,
)
from mojo_rl.render import Renderer2D
from std.memory import UnsafePointer
from mojo_rl.core.utils.gae import compute_gae_inline
from mojo_rl.core.utils.normalization import normalize_inline, RunningMeanStd
from mojo_rl.core.utils.shuffle import shuffle_indices_inline
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
    _store_post_step_kernel,
    clamp_log_std_params_kernel,
    add_obs_noise_kernel,
)
from mojo_rl.deep_agents.ppo.state import (
    PPOContinuousState,
    PPOContinuousGPUState,
)
from mojo_rl.deep_agents.core.onpolicy_train import (
    OnPolicyContinuousAgent,
    OnPolicyAgent,
    run_onpolicy_continuous_train,
)
from mojo_rl.deep_agents.core.onpolicy_helpers import (
    compute_gae_list,
    normalize_advantages_list,
    fisher_yates_shuffle,
)
from mojo_rl.deep_agents.core.gpu_onpolicy_train import (
    GPUOnPolicyContinuousAgent,
    run_onpolicy_continuous_train_gpu,
)
from mojo_rl.deep_agents.core.checkpoint_trait import Checkpointable

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
    actor_lr: Float64 = 0.0003,
    critic_lr: Float64 = 0.001,
](
    Checkpointable,
    GPUOnPolicyContinuousAgent,
    OnPolicyAgent,
    OnPolicyContinuousAgent,
):
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
        actor_lr: Learning rate for actor network (default: 0.0003).
        critic_lr: Learning rate for critic network (default: 0.001).

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

    # GPUOnPolicyContinuousAgent + OnPolicyAgent trait constants
    comptime OBS_DIM: Int = Self.obs_dim
    comptime ACTION_DIM: Int = Self.action_dim
    comptime ROLLOUT_LEN: Int = Self.rollout_len
    comptime MAX_N_ENVS: Int = Self.n_envs
    comptime GPUStateType = PPOContinuousGPUState[
        Self.ActorModel,
        Self.ActorOpt,
        Self.CriticModel,
        Self.CriticOpt,
        Self.OBS,
        Self.ACTIONS,
        Self.ROLLOUT,
        Self.n_envs,
        Self.GPU_MINIBATCH,
    ]

    # Actor model and network (stateless ops)
    comptime ActorModel = Sequential[
        DenseTanh[Self.OBS, Self.HIDDEN],
        DenseTanh[Self.HIDDEN, Self.HIDDEN],
        StochasticActor[Self.HIDDEN, Self.ACTIONS],
    ]
    comptime ActorNet = Network[Self.ActorModel, Adam[Self.actor_lr]]

    # Critic model and network (stateless ops)
    comptime CriticModel = Sequential[
        DenseTanh[Self.OBS, Self.HIDDEN],
        DenseTanh[Self.HIDDEN, Self.HIDDEN],
        Dense[Self.HIDDEN, 1],
    ]
    comptime CriticNet = Network[Self.CriticModel, Adam[Self.critic_lr]]

    # Compile-time state type (actor + critic networks + rollout buffers)
    comptime ActorOpt = Adam[Self.actor_lr]
    comptime CriticOpt = Adam[Self.critic_lr]
    comptime CPUStateType = PPOContinuousState[
        Self.ActorModel,
        Self.ActorOpt,
        Self.CriticModel,
        Self.CriticOpt,
        Self.OBS,
        Self.ACTIONS,
        Self.ROLLOUT,
    ]

    # CPU state (actor + critic networks + rollout buffers)
    var state: Self.CPUStateType

    # Hyperparameters
    var gamma: Float64
    var gae_lambda: Float64
    var clip_epsilon: Float64
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

    # Observation noise for robustness (domain randomization)
    var obs_noise_std: Float64

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        gae_lambda: Float64 = 0.95,
        clip_epsilon: Float64 = 0.2,
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
        # Observation noise for robustness (domain randomization)
        obs_noise_std: Float64 = 0.0,
    ):
        """Initialize Deep PPO Continuous agent.

        Args:
            gamma: Discount factor (default: 0.99).
            gae_lambda: GAE lambda parameter (default: 0.95).
            clip_epsilon: PPO clipping parameter (default: 0.2).
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
            normalize_rewards: Whether to normalize rewards (default: True).
            action_mean_biases: Per-action mean biases for policy initialization (optional).
            obs_noise_std: Std dev of Gaussian noise added to observations (default: 0.0).
        """
        # Initialize CPU state (actor + critic + rollout buffers, Kaiming init)
        self.state = Self.CPUStateType()

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
        ](self.state.actor.params + STOCHASTIC_ACTOR_OFFSET)

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

        # Initialize critic AFTER init_params_small to match old's RNG ordering:
        #   actor_kaiming → init_params_small → critic_kaiming
        # PPOContinuousState.__init__() allocates the critic but does NOT
        # initialize it (see state.mojo docstring). We initialize it here so
        # the critic consumes RNG at exactly the same positions as the old code.
        self.state.critic.initialize[Kaiming[]]()

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
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

        self.obs_noise_std = obs_noise_std

    # =========================================================================
    # Action Selection (for evaluation)
    # =========================================================================

    fn select_action(
        self,
        obs: LayoutTensor[dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin],
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
        var actor_output_tensor = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTOR_OUT), MutAnyOrigin
        ](actor_output.unsafe_ptr())
        var p_actor = self.state.actor.params_view()
        Self.ActorNet.forward[1](obs, actor_output_tensor, p_actor)

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
        var value_out_tensor = LayoutTensor[
            dtype, Layout.row_major(1, 1), MutAnyOrigin
        ](value_out.unsafe_ptr())
        var p_critic = self.state.critic.params_view()
        Self.CriticNet.forward[1](obs, value_out_tensor, p_critic)
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
    # OnPolicyContinuousAgent trait conformance
    # =========================================================================

    fn make_cpu_state(self) -> Self.CPUStateType:
        """Allocate a fresh PPOContinuousState with Kaiming-initialized networks.

        Applies the same init_params_small + critic_kaiming ordering as __init__
        so CPU training gets consistent initialization.
        """
        var s = Self.CPUStateType()
        comptime sa_offset = (
            Self.OBS * Self.HIDDEN
            + Self.HIDDEN
            + Self.HIDDEN * Self.HIDDEN
            + Self.HIDDEN
        )
        comptime sa_size = Self.HIDDEN * Self.ACTIONS + Self.ACTIONS + Self.ACTIONS
        var stochastic_actor_params = LayoutTensor[
            dtype, Layout.row_major(sa_size), MutAnyOrigin
        ](s.actor.params + sa_offset)
        StochasticActor[Self.HIDDEN, Self.ACTIONS].init_params_small(
            stochastic_actor_params,
            weight_scale=0.01,
            log_std_init=-0.5,
        )
        s.critic.initialize[Kaiming[]]()
        return s^

    fn collect_rollout[
        E: BoxContinuousActionEnv
    ](mut self, mut cpu_state: Self.CPUStateType, mut env: E) -> None:
        """Collect exactly ROLLOUT_LEN steps into cpu_state rollout buffers.

        Handles episode resets. After this call, cpu_state._current_obs holds
        the last obs for bootstrapping in compute_advantages.
        """
        if not cpu_state._env_initialized:
            var obs_list = env.reset_obs_list()
            for i in range(Self.OBS):
                cpu_state._current_obs[i] = Scalar[dtype](obs_list[i])
            cpu_state._env_initialized = True

        cpu_state.buffer_idx = 0

        for _ in range(Self.ROLLOUT):
            # Build obs InlineArray from cpu_state._current_obs
            var obs_arr = InlineArray[Scalar[dtype], Self.OBS](
                uninitialized=True
            )
            for i in range(Self.OBS):
                obs_arr[i] = cpu_state._current_obs[i]

            var obs_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
            ](obs_arr.unsafe_ptr())

            # Select action from cpu_state.actor
            var action_result = self.select_action(obs_t, training=True)
            var log_prob = action_result[1]
            var value = action_result[2]

            # Build action list for environment (scale from actor output range)
            var action_list = List[Float64](capacity=Self.ACTIONS)
            for i in range(Self.ACTIONS):
                var a = (
                    Float64(action_result[0][i]) * self.action_scale
                    + self.action_bias
                )
                action_list.append(a)

            # Step environment
            var result = env.step_continuous_vec(action_list)
            var reward = Float64(result[1])
            var done = result[2]

            # Store in buffer
            var idx = cpu_state.buffer_idx
            for i in range(Self.OBS):
                cpu_state.buffer_obs[
                    idx * Self.OBS + i
                ] = cpu_state._current_obs[i]
            for i in range(Self.ACTIONS):
                cpu_state.buffer_actions[
                    idx * Self.ACTIONS + i
                ] = action_result[0][i]
            cpu_state.buffer_rewards[idx] = Scalar[dtype](reward)
            cpu_state.buffer_values[idx] = value
            cpu_state.buffer_log_probs[idx] = log_prob
            cpu_state.buffer_dones[idx] = done
            cpu_state.buffer_idx += 1

            if done:
                var next_obs_list = env.reset_obs_list()
                for i in range(Self.OBS):
                    cpu_state._current_obs[i] = Scalar[dtype](next_obs_list[i])
            else:
                for i in range(Self.OBS):
                    cpu_state._current_obs[i] = Scalar[dtype](result[0][i])

    fn compute_advantages(mut self, mut cpu_state: Self.CPUStateType) -> None:
        """Compute GAE advantages and returns using cpu_state._current_obs as bootstrap.
        """
        var buffer_len = cpu_state.buffer_idx
        if buffer_len == 0:
            return

        # Bootstrap: run critic on last obs
        var next_obs_arr = InlineArray[Scalar[dtype], Self.OBS](
            uninitialized=True
        )
        for i in range(Self.OBS):
            next_obs_arr[i] = cpu_state._current_obs[i]
        var next_obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](next_obs_arr.unsafe_ptr())
        var next_val_data = InlineArray[Scalar[dtype], 1](uninitialized=True)
        var next_val_t = LayoutTensor[
            dtype, Layout.row_major(1, 1), MutAnyOrigin
        ](next_val_data.unsafe_ptr())
        var p_critic = cpu_state.critic.params_view()
        Self.CriticNet.forward[1](next_obs_t, next_val_t, p_critic)
        var next_value = rebind[Scalar[dtype]](next_val_t[0, 0])

        compute_gae_list[dtype](
            cpu_state.buffer_rewards,
            cpu_state.buffer_values,
            cpu_state.buffer_dones,
            next_value,
            buffer_len,
            self.gamma,
            self.gae_lambda,
            cpu_state._advantages,
            cpu_state._returns,
        )

        if self.normalize_advantages and buffer_len > 1:
            normalize_advantages_list[dtype](cpu_state._advantages, buffer_len)

    fn update_epochs(mut self, mut cpu_state: Self.CPUStateType) -> Float64:
        """Update actor/critic over num_epochs with minibatch PPO. Returns mean loss.
        """
        from std.math import exp as _exp, log as _log, sqrt as _sqrt

        var buffer_len = cpu_state.buffer_idx
        if buffer_len == 0:
            return 0.0

        # Reset index list for shuffling
        for i in range(buffer_len):
            cpu_state._indices[i] = i

        var total_loss = Scalar[dtype](0.0)

        for epoch in range(self.num_epochs):
            fisher_yates_shuffle(cpu_state._indices, buffer_len)

            var batch_start = 0
            comptime MB = 64  # CPU minibatch size
            while batch_start < buffer_len:
                var batch_end = batch_start + MB
                if batch_end > buffer_len:
                    batch_end = buffer_len
                var mb_size = batch_end - batch_start

                # Per-minibatch advantage normalization
                var mb_advantages = List[Scalar[dtype]](capacity=mb_size)
                for b in range(batch_start, batch_end):
                    mb_advantages.append(
                        cpu_state._advantages[cpu_state._indices[b]]
                    )

                if self.norm_adv_per_minibatch and mb_size > 1:
                    normalize_advantages_list[dtype](mb_advantages, mb_size)

                # Process each sample in minibatch
                for b in range(batch_start, batch_end):
                    var t = cpu_state._indices[b]
                    var mb_idx = b - batch_start

                    var obs_arr = InlineArray[Scalar[dtype], Self.OBS](
                        uninitialized=True
                    )
                    for i in range(Self.OBS):
                        obs_arr[i] = cpu_state.buffer_obs[t * Self.OBS + i]

                    var obs_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
                    ](obs_arr.unsafe_ptr())

                    var old_log_prob = cpu_state.buffer_log_probs[t]
                    var old_value = cpu_state.buffer_values[t]
                    var advantage = mb_advantages[mb_idx]
                    var return_t = cpu_state._returns[t]

                    # Actor forward (get mean + log_std)
                    var actor_out_arr = InlineArray[
                        Scalar[dtype], Self.ACTOR_OUT
                    ](uninitialized=True)
                    var actor_out_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.ACTOR_OUT), MutAnyOrigin
                    ](actor_out_arr.unsafe_ptr())
                    var p_actor = cpu_state.actor.params_view()
                    Self.ActorNet.forward[1](obs_t, actor_out_t, p_actor)

                    # Compute new log_prob from stored actions
                    comptime LOG_STD_MIN: Scalar[dtype] = -5.0
                    comptime LOG_STD_MAX: Scalar[dtype] = 2.0
                    var new_log_prob = Scalar[dtype](0.0)
                    for j in range(Self.ACTIONS):
                        var mean = actor_out_arr[j]
                        var log_std = actor_out_arr[Self.ACTIONS + j]
                        if log_std < LOG_STD_MIN:
                            log_std = LOG_STD_MIN
                        elif log_std > LOG_STD_MAX:
                            log_std = LOG_STD_MAX
                        var std = _exp(log_std)
                        var action_j = cpu_state.buffer_actions[
                            t * Self.ACTIONS + j
                        ]
                        var normalized = (action_j - mean) / (
                            std + Scalar[dtype](1e-8)
                        )
                        var lp = (
                            -Scalar[dtype](0.5) * normalized * normalized
                            - log_std
                            - Scalar[dtype](0.9189385)
                        )
                        new_log_prob += lp

                    var ratio = _exp(new_log_prob - old_log_prob)
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
                    var policy_loss = -min(surr1, surr2)

                    # Actor backward
                    var is_clipped = (
                        ratio < Scalar[dtype](1.0 - self.clip_epsilon)
                    ) or (ratio > Scalar[dtype](1.0 + self.clip_epsilon))
                    var d_actor_out = InlineArray[
                        Scalar[dtype], Self.ACTOR_OUT
                    ](fill=0)
                    if not is_clipped:
                        for j in range(Self.ACTIONS):
                            var log_std = actor_out_arr[Self.ACTIONS + j]
                            if log_std < LOG_STD_MIN:
                                log_std = LOG_STD_MIN
                            elif log_std > LOG_STD_MAX:
                                log_std = LOG_STD_MAX
                            var std = _exp(log_std)
                            var action_j = cpu_state.buffer_actions[
                                t * Self.ACTIONS + j
                            ]
                            var mean = actor_out_arr[j]
                            var normalized = (action_j - mean) / (
                                std + Scalar[dtype](1e-8)
                            )
                            # d(log_prob)/d(mean) = normalized / std
                            var d_lp_d_mean = normalized / (
                                std + Scalar[dtype](1e-8)
                            )
                            # d(log_prob)/d(log_std) = normalized^2 - 1
                            var d_lp_d_log_std = (
                                normalized * normalized - Scalar[dtype](1.0)
                            )
                            var d_policy_d_lp = -advantage * ratio
                            d_actor_out[
                                j
                            ] = d_policy_d_lp * d_lp_d_mean - Scalar[dtype](
                                self.entropy_coef
                            ) * (
                                -d_lp_d_mean
                            )
                            d_actor_out[
                                Self.ACTIONS + j
                            ] = d_policy_d_lp * d_lp_d_log_std - Scalar[dtype](
                                self.entropy_coef
                            ) * (
                                -d_lp_d_log_std
                            )

                    var actor_cache = List[Scalar[dtype]](
                        capacity=Self.ActorModel.CACHE_SIZE
                    )
                    for _ in range(Self.ActorModel.CACHE_SIZE):
                        actor_cache.append(Scalar[dtype](0))
                    var actor_cache_t = LayoutTensor[
                        dtype,
                        Layout.row_major(1, Self.ActorModel.CACHE_SIZE),
                        MutAnyOrigin,
                    ](actor_cache.unsafe_ptr())
                    Self.ActorNet.forward_with_cache[1](
                        obs_t, actor_out_t, p_actor, actor_cache_t
                    )

                    var d_actor_out_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.ACTOR_OUT), MutAnyOrigin
                    ](d_actor_out.unsafe_ptr())
                    var actor_grad_in = InlineArray[Scalar[dtype], Self.OBS](
                        fill=0
                    )
                    var actor_grad_in_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
                    ](actor_grad_in.unsafe_ptr())
                    var g_actor = cpu_state.actor.grads_view()
                    cpu_state.actor.zero_grads()
                    Self.ActorNet.backward[1](
                        d_actor_out_t,
                        actor_grad_in_t,
                        p_actor,
                        actor_cache_t,
                        g_actor,
                    )
                    cpu_state.actor.optimizer_step()

                    # Critic forward + update
                    var val_data = InlineArray[Scalar[dtype], 1](
                        uninitialized=True
                    )
                    var val_t = LayoutTensor[
                        dtype, Layout.row_major(1, 1), MutAnyOrigin
                    ](val_data.unsafe_ptr())
                    var critic_cache = List[Scalar[dtype]](
                        capacity=Self.CriticModel.CACHE_SIZE
                    )
                    for _ in range(Self.CriticModel.CACHE_SIZE):
                        critic_cache.append(Scalar[dtype](0))
                    var critic_cache_t = LayoutTensor[
                        dtype,
                        Layout.row_major(1, Self.CriticModel.CACHE_SIZE),
                        MutAnyOrigin,
                    ](critic_cache.unsafe_ptr())
                    var p_critic_u = cpu_state.critic.params_view()
                    Self.CriticNet.forward_with_cache[1](
                        obs_t, val_t, p_critic_u, critic_cache_t
                    )
                    var val = rebind[Scalar[dtype]](val_t[0, 0])
                    var value_loss = (return_t - val) * (return_t - val)

                    var d_val = InlineArray[Scalar[dtype], 1](fill=0)
                    d_val[0] = (
                        Scalar[dtype](2.0)
                        * Scalar[dtype](self.value_loss_coef)
                        * (val - return_t)
                    )

                    var d_val_t = LayoutTensor[
                        dtype, Layout.row_major(1, 1), MutAnyOrigin
                    ](d_val.unsafe_ptr())
                    var critic_grad_in = InlineArray[Scalar[dtype], Self.OBS](
                        fill=0
                    )
                    var d_in_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
                    ](critic_grad_in.unsafe_ptr())
                    var g_critic = cpu_state.critic.grads_view()
                    cpu_state.critic.zero_grads()
                    Self.CriticNet.backward[1](
                        d_val_t, d_in_t, p_critic_u, critic_cache_t, g_critic
                    )
                    cpu_state.critic.optimizer_step()

                    total_loss += (
                        policy_loss
                        + Scalar[dtype](self.value_loss_coef) * value_loss
                    )

                batch_start = batch_end

        cpu_state.buffer_idx = 0
        self.train_step_count += 1
        return Float64(total_loss / Scalar[dtype](self.num_epochs * buffer_len))

    fn select_greedy_action(
        self, cpu_state: Self.CPUStateType, obs: List[Float64]
    ) -> List[Float64]:
        """Select deterministic action (actor mean) for evaluation."""
        var obs_arr = InlineArray[Scalar[dtype], Self.OBS](uninitialized=True)
        for i in range(Self.OBS):
            obs_arr[i] = Scalar[dtype](obs[i])
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())
        var action_result = self.select_action(obs_t, training=False)
        var result = List[Float64](capacity=Self.ACTIONS)
        for i in range(Self.ACTIONS):
            result.append(
                Float64(action_result[0][i]) * self.action_scale
                + self.action_bias
            )
        return result^

    fn get_explore_rate(self) -> Float64:
        """Return entropy coefficient as exploration proxy."""
        return self.entropy_coef

    # =========================================================================
    # CPU Training (delegates to shared on-policy loop)
    # =========================================================================

    fn train[
        E: BoxContinuousActionEnv
    ](
        mut self,
        mut env: E,
        num_episodes: Int,
        verbose: Bool = False,
        print_every: Int = 10,
        environment_name: String = "Environment",
    ) raises -> TrainingMetrics:
        """Train the PPO continuous agent on a continuous action environment.

        Delegates to the shared on-policy continuous training loop.
        num_episodes is treated as num_updates (rollout-based, not episode-based).

        Args:
            env: The environment to train on.
            num_episodes: Number of rollout updates (num_updates).
            verbose: Whether to print progress.
            print_every: Print progress every N updates if verbose.
            environment_name: Name of environment for metrics labeling.

        Returns:
            TrainingMetrics with one entry per update (reward = policy loss).
        """
        var checkpoint_path = self.checkpoint_path
        var checkpoint_every = self.checkpoint_every
        return run_onpolicy_continuous_train(
            self,
            env,
            num_episodes,
            checkpoint_every,
            checkpoint_path,
            verbose,
            print_every,
            environment_name,
            "PPO Continuous (GPU)",
        )

    # =========================================================================
    # OnPolicyAgent trait methods (use self.state directly, no aliasing issue)
    # =========================================================================

    fn collect_rollout[E: BoxDiscreteActionEnv](mut self, mut env: E) -> None:
        """No-op: continuous PPO does not use discrete environments."""
        pass

    fn collect_rollout_continuous[
        E: BoxContinuousActionEnv
    ](mut self, mut env: E) -> None:
        """Collect ROLLOUT_LEN steps using self.state (OnPolicyAgent overload).
        """
        if not self.state._env_initialized:
            var obs_list = env.reset_obs_list()
            for i in range(Self.OBS):
                self.state._current_obs[i] = Scalar[dtype](obs_list[i])
            self.state._env_initialized = True

        self.state.buffer_idx = 0

        for _ in range(Self.ROLLOUT):
            var obs_arr = InlineArray[Scalar[dtype], Self.OBS](
                uninitialized=True
            )
            for i in range(Self.OBS):
                obs_arr[i] = self.state._current_obs[i]

            var obs_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
            ](obs_arr.unsafe_ptr())

            var action_result = self.select_action(obs_t, training=True)
            var log_prob = action_result[1]
            var value = action_result[2]

            var action_list = List[Float64](capacity=Self.ACTIONS)
            for i in range(Self.ACTIONS):
                var a = (
                    Float64(action_result[0][i]) * self.action_scale
                    + self.action_bias
                )
                action_list.append(a)

            var result = env.step_continuous_vec(action_list)
            var reward = Float64(result[1])
            var done = result[2]

            var idx = self.state.buffer_idx
            for i in range(Self.OBS):
                self.state.buffer_obs[
                    idx * Self.OBS + i
                ] = self.state._current_obs[i]
            for i in range(Self.ACTIONS):
                self.state.buffer_actions[
                    idx * Self.ACTIONS + i
                ] = action_result[0][i]
            self.state.buffer_rewards[idx] = Scalar[dtype](reward)
            self.state.buffer_values[idx] = value
            self.state.buffer_log_probs[idx] = log_prob
            self.state.buffer_dones[idx] = done
            self.state.buffer_idx += 1

            if done:
                var next_obs_list = env.reset_obs_list()
                for i in range(Self.OBS):
                    self.state._current_obs[i] = Scalar[dtype](next_obs_list[i])
            else:
                for i in range(Self.OBS):
                    self.state._current_obs[i] = Scalar[dtype](result[0][i])

    fn compute_advantages(mut self) -> None:
        """Compute GAE advantages using self.state (OnPolicyAgent overload)."""
        var buffer_len = self.state.buffer_idx
        if buffer_len == 0:
            return

        var next_obs_arr = InlineArray[Scalar[dtype], Self.OBS](
            uninitialized=True
        )
        for i in range(Self.OBS):
            next_obs_arr[i] = self.state._current_obs[i]
        var next_obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](next_obs_arr.unsafe_ptr())
        var next_val_data = InlineArray[Scalar[dtype], 1](uninitialized=True)
        var next_val_t = LayoutTensor[
            dtype, Layout.row_major(1, 1), MutAnyOrigin
        ](next_val_data.unsafe_ptr())
        var p_critic = self.state.critic.params_view()
        Self.CriticNet.forward[1](next_obs_t, next_val_t, p_critic)
        var next_value = rebind[Scalar[dtype]](next_val_t[0, 0])

        compute_gae_list[dtype](
            self.state.buffer_rewards,
            self.state.buffer_values,
            self.state.buffer_dones,
            next_value,
            buffer_len,
            self.gamma,
            self.gae_lambda,
            self.state._advantages,
            self.state._returns,
        )

        if self.normalize_advantages and buffer_len > 1:
            normalize_advantages_list[dtype](self.state._advantages, buffer_len)

    fn update_epochs(mut self) -> Float64:
        """Update actor/critic over num_epochs with minibatch PPO (OnPolicyAgent overload).
        """
        from std.math import exp as _exp, log as _log, sqrt as _sqrt

        var buffer_len = self.state.buffer_idx
        if buffer_len == 0:
            return 0.0

        for i in range(buffer_len):
            self.state._indices[i] = i

        var total_loss = Scalar[dtype](0.0)

        for epoch in range(self.num_epochs):
            fisher_yates_shuffle(self.state._indices, buffer_len)

            var batch_start = 0
            comptime MB = 64
            while batch_start < buffer_len:
                var batch_end = batch_start + MB
                if batch_end > buffer_len:
                    batch_end = buffer_len
                var mb_size = batch_end - batch_start

                var mb_advantages = List[Scalar[dtype]](capacity=mb_size)
                for b in range(batch_start, batch_end):
                    mb_advantages.append(
                        self.state._advantages[self.state._indices[b]]
                    )

                if self.norm_adv_per_minibatch and mb_size > 1:
                    normalize_advantages_list[dtype](mb_advantages, mb_size)

                for b in range(batch_start, batch_end):
                    var t = self.state._indices[b]
                    var mb_idx = b - batch_start

                    var obs_arr = InlineArray[Scalar[dtype], Self.OBS](
                        uninitialized=True
                    )
                    for i in range(Self.OBS):
                        obs_arr[i] = self.state.buffer_obs[t * Self.OBS + i]

                    var obs_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
                    ](obs_arr.unsafe_ptr())

                    var old_log_prob = self.state.buffer_log_probs[t]
                    var old_value = self.state.buffer_values[t]
                    var advantage = mb_advantages[mb_idx]
                    var return_t = self.state._returns[t]

                    var actor_out_arr = InlineArray[
                        Scalar[dtype], Self.ACTOR_OUT
                    ](uninitialized=True)
                    var actor_out_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.ACTOR_OUT), MutAnyOrigin
                    ](actor_out_arr.unsafe_ptr())
                    var p_actor = self.state.actor.params_view()
                    Self.ActorNet.forward[1](obs_t, actor_out_t, p_actor)

                    comptime LOG_STD_MIN: Scalar[dtype] = -5.0
                    comptime LOG_STD_MAX: Scalar[dtype] = 2.0
                    var new_log_prob = Scalar[dtype](0.0)
                    for j in range(Self.ACTIONS):
                        var mean = actor_out_arr[j]
                        var log_std = actor_out_arr[Self.ACTIONS + j]
                        if log_std < LOG_STD_MIN:
                            log_std = LOG_STD_MIN
                        elif log_std > LOG_STD_MAX:
                            log_std = LOG_STD_MAX
                        var std = _exp(log_std)
                        var action_j = self.state.buffer_actions[
                            t * Self.ACTIONS + j
                        ]
                        var normalized = (action_j - mean) / (
                            std + Scalar[dtype](1e-8)
                        )
                        var lp = (
                            -Scalar[dtype](0.5) * normalized * normalized
                            - log_std
                            - Scalar[dtype](0.9189385)
                        )
                        new_log_prob += lp

                    var ratio = _exp(new_log_prob - old_log_prob)
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
                    var policy_loss = -min(surr1, surr2)

                    var is_clipped = (
                        ratio < Scalar[dtype](1.0 - self.clip_epsilon)
                    ) or (ratio > Scalar[dtype](1.0 + self.clip_epsilon))
                    var d_actor_out = InlineArray[
                        Scalar[dtype], Self.ACTOR_OUT
                    ](fill=0)
                    if not is_clipped:
                        for j in range(Self.ACTIONS):
                            var log_std = actor_out_arr[Self.ACTIONS + j]
                            if log_std < LOG_STD_MIN:
                                log_std = LOG_STD_MIN
                            elif log_std > LOG_STD_MAX:
                                log_std = LOG_STD_MAX
                            var std = _exp(log_std)
                            var action_j = self.state.buffer_actions[
                                t * Self.ACTIONS + j
                            ]
                            var mean = actor_out_arr[j]
                            var normalized = (action_j - mean) / (
                                std + Scalar[dtype](1e-8)
                            )
                            var d_lp_d_mean = normalized / (
                                std + Scalar[dtype](1e-8)
                            )
                            var d_lp_d_log_std = (
                                normalized * normalized - Scalar[dtype](1.0)
                            )
                            var d_policy_d_lp = -advantage * ratio
                            d_actor_out[
                                j
                            ] = d_policy_d_lp * d_lp_d_mean - Scalar[dtype](
                                self.entropy_coef
                            ) * (
                                -d_lp_d_mean
                            )
                            d_actor_out[
                                Self.ACTIONS + j
                            ] = d_policy_d_lp * d_lp_d_log_std - Scalar[dtype](
                                self.entropy_coef
                            ) * (
                                -d_lp_d_log_std
                            )

                    var actor_cache = List[Scalar[dtype]](
                        capacity=Self.ActorModel.CACHE_SIZE
                    )
                    for _ in range(Self.ActorModel.CACHE_SIZE):
                        actor_cache.append(Scalar[dtype](0))
                    var actor_cache_t = LayoutTensor[
                        dtype,
                        Layout.row_major(1, Self.ActorModel.CACHE_SIZE),
                        MutAnyOrigin,
                    ](actor_cache.unsafe_ptr())
                    Self.ActorNet.forward_with_cache[1](
                        obs_t, actor_out_t, p_actor, actor_cache_t
                    )

                    var d_actor_out_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.ACTOR_OUT), MutAnyOrigin
                    ](d_actor_out.unsafe_ptr())
                    var actor_grad_in = InlineArray[Scalar[dtype], Self.OBS](
                        fill=0
                    )
                    var actor_grad_in_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
                    ](actor_grad_in.unsafe_ptr())
                    var g_actor = self.state.actor.grads_view()
                    self.state.actor.zero_grads()
                    Self.ActorNet.backward[1](
                        d_actor_out_t,
                        actor_grad_in_t,
                        p_actor,
                        actor_cache_t,
                        g_actor,
                    )
                    self.state.actor.optimizer_step()

                    var val_data = InlineArray[Scalar[dtype], 1](
                        uninitialized=True
                    )
                    var val_t = LayoutTensor[
                        dtype, Layout.row_major(1, 1), MutAnyOrigin
                    ](val_data.unsafe_ptr())
                    var critic_cache = List[Scalar[dtype]](
                        capacity=Self.CriticModel.CACHE_SIZE
                    )
                    for _ in range(Self.CriticModel.CACHE_SIZE):
                        critic_cache.append(Scalar[dtype](0))
                    var critic_cache_t = LayoutTensor[
                        dtype,
                        Layout.row_major(1, Self.CriticModel.CACHE_SIZE),
                        MutAnyOrigin,
                    ](critic_cache.unsafe_ptr())
                    var p_critic_u = self.state.critic.params_view()
                    Self.CriticNet.forward_with_cache[1](
                        obs_t, val_t, p_critic_u, critic_cache_t
                    )
                    var val = rebind[Scalar[dtype]](val_t[0, 0])
                    var value_loss = (return_t - val) * (return_t - val)

                    var d_val = InlineArray[Scalar[dtype], 1](fill=0)
                    d_val[0] = (
                        Scalar[dtype](2.0)
                        * Scalar[dtype](self.value_loss_coef)
                        * (val - return_t)
                    )

                    var d_val_t = LayoutTensor[
                        dtype, Layout.row_major(1, 1), MutAnyOrigin
                    ](d_val.unsafe_ptr())
                    var critic_grad_in = InlineArray[Scalar[dtype], Self.OBS](
                        fill=0
                    )
                    var d_in_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
                    ](critic_grad_in.unsafe_ptr())
                    var g_critic = self.state.critic.grads_view()
                    self.state.critic.zero_grads()
                    Self.CriticNet.backward[1](
                        d_val_t, d_in_t, p_critic_u, critic_cache_t, g_critic
                    )
                    self.state.critic.optimizer_step()

                    total_loss += (
                        policy_loss
                        + Scalar[dtype](self.value_loss_coef) * value_loss
                    )

                batch_start = batch_end

        self.state.buffer_idx = 0
        self.train_step_count += 1
        return Float64(total_loss / Scalar[dtype](self.num_epochs * buffer_len))

    fn select_greedy_action_list(self, obs: List[Float64]) -> List[Float64]:
        """Select deterministic action (actor mean) for evaluation (OnPolicyAgent overload).
        """
        var obs_arr = InlineArray[Scalar[dtype], Self.OBS](uninitialized=True)
        for i in range(Self.OBS):
            obs_arr[i] = Scalar[dtype](obs[i])
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())
        var action_result = self.select_action(obs_t, training=False)
        var result = List[Float64](capacity=Self.ACTIONS)
        for i in range(Self.ACTIONS):
            result.append(
                Float64(action_result[0][i]) * self.action_scale
                + self.action_bias
            )
        return result^

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
        content += "actor_lr=" + String(Float64(Self.actor_lr)) + "\n"
        content += "critic_lr=" + String(Float64(Self.critic_lr)) + "\n"
        content += "entropy_coef=" + String(self.entropy_coef) + "\n"
        content += "train_step_count=" + String(self.train_step_count) + "\n"

        content += "[ACTOR_PARAMS]\n"
        for i in range(Self.ActorModel.PARAM_SIZE):
            content += String((self.state.actor.params + i)[]) + "\n"

        content += "[ACTOR_STATE]\n"
        comptime ACTOR_STATE_SIZE = Self.ActorModel.PARAM_SIZE * Self.ActorOpt.STATE_PER_PARAM
        for i in range(ACTOR_STATE_SIZE):
            content += String((self.state.actor.optimizer_state + i)[]) + "\n"

        content += "[CRITIC_PARAMS]\n"
        for i in range(Self.CriticModel.PARAM_SIZE):
            content += String((self.state.critic.params + i)[]) + "\n"

        content += "[CRITIC_STATE]\n"
        comptime CRITIC_STATE_SIZE = Self.CriticModel.PARAM_SIZE * Self.CriticOpt.STATE_PER_PARAM
        for i in range(CRITIC_STATE_SIZE):
            content += String((self.state.critic.optimizer_state + i)[]) + "\n"

        save_checkpoint_file(path, content)

    fn load_checkpoint(mut self, path: String) raises:
        """Load agent state from a checkpoint file."""
        comptime ACTOR_STATE_SIZE = Self.ActorModel.PARAM_SIZE * Self.ActorOpt.STATE_PER_PARAM
        comptime CRITIC_STATE_SIZE = Self.CriticModel.PARAM_SIZE * Self.CriticOpt.STATE_PER_PARAM

        var content = read_checkpoint_file(path)
        if len(content) == 0:
            print("No checkpoint found at:", path)
            return

        var lines = split_lines(content)

        # Load actor params
        var actor_start = find_section_start(lines, "[ACTOR_PARAMS]")
        if actor_start >= 0:
            var idx = actor_start
            for i in range(Self.ActorModel.PARAM_SIZE):
                if idx < len(lines) and not lines[idx].startswith("["):
                    try:
                        (self.state.actor.params + i)[] = Scalar[dtype](
                            Float32(atof(lines[idx]))
                        )
                    except:
                        pass
                    idx += 1

        # Load actor optimizer state
        var actor_state_start = find_section_start(lines, "[ACTOR_STATE]")
        if actor_state_start >= 0:
            var idx = actor_state_start
            for i in range(ACTOR_STATE_SIZE):
                if idx < len(lines) and not lines[idx].startswith("["):
                    try:
                        (self.state.actor.optimizer_state + i)[] = Scalar[
                            dtype
                        ](Float32(atof(lines[idx])))
                    except:
                        pass
                    idx += 1

        # Load critic params
        var critic_start = find_section_start(lines, "[CRITIC_PARAMS]")
        if critic_start >= 0:
            var idx = critic_start
            for i in range(Self.CriticModel.PARAM_SIZE):
                if idx < len(lines) and not lines[idx].startswith("["):
                    try:
                        (self.state.critic.params + i)[] = Scalar[dtype](
                            Float32(atof(lines[idx]))
                        )
                    except:
                        pass
                    idx += 1

        # Load critic optimizer state
        var critic_state_start = find_section_start(lines, "[CRITIC_STATE]")
        if critic_state_start >= 0:
            var idx = critic_state_start
            for i in range(CRITIC_STATE_SIZE):
                if idx < len(lines) and not lines[idx].startswith("["):
                    try:
                        (self.state.critic.optimizer_state + i)[] = Scalar[
                            dtype
                        ](Float32(atof(lines[idx])))
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
        E: BoxContinuousActionEnv & RenderableEnv
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps: Int = 1000,
        verbose: Bool = False,
        debug: Bool = False,
        stochastic: Bool = True,
        render: Bool = False,
        frame_delay_ms: Int = 16,
    ) raises -> Float64:
        """Evaluate the agent with environment-owned rendering (RenderableEnv).

        This method uses the RenderableEnv trait for visualization, allowing
        environments to use their own renderer (2D or 3D) without the algorithm
        needing to know the renderer type.

        Uses unbounded Gaussian policy (CleanRL-style). Actions are clipped
        to [-1, 1] at the environment boundary.

        Args:
            env: The environment to evaluate on (must implement RenderableEnv).
            num_episodes: Number of evaluation episodes.
            max_steps: Maximum steps per episode.
            verbose: Whether to print per-episode results.
            debug: Whether to print debug information.
            stochastic: If True (default), sample from policy; if False, use mean.
            render: If True, render each frame using the environment's renderer.
            frame_delay_ms: Delay between frames in milliseconds (default: 16 ~60fps).

        Returns:
            Average reward over evaluation episodes.
        """
        var total_reward: Float64 = 0.0
        var quit_requested = False

        # Initialize renderer if needed
        if render:
            _ = env.init_renderer()

        for episode in range(num_episodes):
            if quit_requested:
                break

            var obs_list = env.reset_obs_list()
            var obs = InlineArray[Scalar[dtype], Self.OBS](uninitialized=True)
            for i in range(Self.OBS):
                obs[i] = Scalar[dtype](obs_list[i])
            var obs_tensor = LayoutTensor[
                dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
            ](obs.unsafe_ptr())

            var episode_reward: Float64 = 0.0
            var episode_steps = 0

            for _ in range(max_steps):
                # Render current state and handle input events first
                if render:
                    env.render_frame()
                    env.renderer_delay(frame_delay_ms)
                    if env.check_renderer_quit():
                        quit_requested = True
                        break
                    # Skip physics step while paused (unless Right arrow pressed)
                    if (
                        env.renderer_is_paused()
                        and not env.renderer_step_once()
                    ):
                        continue

                # stochastic=True samples from policy, False uses mean
                var action_result = self.select_action(
                    obs_tensor, training=stochastic
                )
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
                var result = env.step_continuous_vec[dtype](
                    action_list, verbose=debug
                )
                var next_obs_list = result[0].copy()
                var reward = result[1]
                var done = result[2]

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

        # Close renderer if initialized
        if render:
            env.close_renderer()

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
        var terminated_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)

        # Action buffers
        var actions_buf = ctx.enqueue_create_buffer[dtype](ENV_ACTION_SIZE)
        var actor_out_buf = ctx.enqueue_create_buffer[dtype](
            Self.n_envs * Self.ACTOR_OUT
        )

        # Network parameter buffers (copy from CPU)
        var actor_params_buf = ctx.enqueue_create_buffer[dtype](
            Self.ActorModel.PARAM_SIZE
        )
        ctx.enqueue_copy(actor_params_buf, self.state.actor.params)

        # Workspace buffer for forward pass
        comptime WORKSPACE_PER_SAMPLE = Self.ActorModel.WORKSPACE_SIZE_PER_SAMPLE
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
        # Pre-allocate step workspace BEFORE reset to avoid stack overflow
        # (reset compiles large Metal shaders that consume stack space)
        comptime EVAL_TOTAL_WS = EnvType.STEP_WS_SHARED + Self.n_envs * EnvType.STEP_WS_PER_ENV
        comptime EVAL_WS_ALLOC = EVAL_TOTAL_WS if EVAL_TOTAL_WS > 0 else 1
        var eval_ws_buf = ctx.enqueue_create_buffer[dtype](EVAL_WS_ALLOC)
        EnvType.init_step_workspace_gpu[Self.n_envs](ctx, eval_ws_buf)

        EnvType.reset_kernel_gpu[Self.n_envs, EnvType.STATE_SIZE](
            ctx, env_states_buf
        )

        # Extract initial observations using environment-specific kernel
        comptime ENV_BLOCKS = (Self.n_envs + TPB - 1) // TPB

        EnvType.extract_obs_kernel_gpu[
            Self.n_envs, EnvType.STATE_SIZE, Self.OBS
        ](ctx, env_states_buf, obs_buf)
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
            var eval_actor_out_tensor = LayoutTensor[
                dtype,
                Layout.row_major(Self.n_envs, Self.ActorModel.OUT_DIM),
                MutAnyOrigin,
            ](actor_out_buf.unsafe_ptr())
            var eval_obs_tensor = LayoutTensor[
                dtype,
                Layout.row_major(Self.n_envs, Self.ActorModel.IN_DIM),
                MutAnyOrigin,
            ](obs_buf.unsafe_ptr())
            var eval_actor_params_tensor = LayoutTensor[
                dtype,
                Layout.row_major(Self.ActorModel.PARAM_SIZE),
                MutAnyOrigin,
            ](actor_params_buf.unsafe_ptr())
            Self.ActorModel.forward_gpu_no_cache[Self.n_envs](
                ctx,
                eval_actor_out_tensor,
                eval_obs_tensor,
                eval_actor_params_tensor,
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
            comptime if EVAL_TOTAL_WS > 0:
                EnvType.step_kernel_gpu[
                    Self.n_envs, EnvType.STATE_SIZE, Self.OBS, Self.ACTIONS
                ](
                    ctx,
                    env_states_buf,
                    actions_buf,
                    rewards_buf,
                    dones_buf,
                    terminated_buf,
                    obs_buf,
                    UInt64(step),
                    List[Scalar[dtype]](),
                    eval_ws_buf.unsafe_ptr(),
                )
            else:
                EnvType.step_kernel_gpu[
                    Self.n_envs, EnvType.STATE_SIZE, Self.OBS, Self.ACTIONS
                ](
                    ctx,
                    env_states_buf,
                    actions_buf,
                    rewards_buf,
                    dones_buf,
                    terminated_buf,
                    obs_buf,
                    UInt64(step),
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

            # Auto-reset done environments (reuse model from workspace)
            EnvType.selective_reset_kernel_gpu[Self.n_envs, EnvType.STATE_SIZE](
                ctx,
                env_states_buf,
                dones_buf,
                UInt64(step),
                workspace_ptr=eval_ws_buf.unsafe_ptr(),
            )

            # Extract observations from reset environments using env-specific kernel
            EnvType.extract_obs_kernel_gpu[
                Self.n_envs, EnvType.STATE_SIZE, Self.OBS
            ](ctx, env_states_buf, obs_buf)

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
    # GPUOnPolicyContinuousAgent trait conformance
    # =========================================================================

    fn make_gpu_state(self, ctx: DeviceContext) raises -> Self.GPUStateType:
        """Allocate all GPU buffers for this agent."""
        return Self.GPUStateType(ctx)

    fn upload_to_gpu(
        self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        """Upload CPU network weights to GPU state."""
        gpu_state.gpu_actor.upload_from(self.state.actor, ctx)
        gpu_state.gpu_critic.upload_from(self.state.critic, ctx)

    fn download_from_gpu(
        mut self,
        mut gpu_state: Self.GPUStateType,
        ctx: DeviceContext,
    ) raises -> None:
        """Download trained GPU weights back to CPU network states."""
        gpu_state.gpu_actor.download_to(self.state.actor, ctx)
        gpu_state.gpu_critic.download_to(self.state.critic, ctx)
        ctx.synchronize()

    fn select_actions_with_meta_gpu[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        obs_buf: DeviceBuffer[dtype],
        mut actions_buf: DeviceBuffer[dtype],
        mut log_probs_buf: DeviceBuffer[dtype],
        mut values_buf: DeviceBuffer[dtype],
        rng_seed: UInt32 = 0,
    ) raises -> None:
        """Forward actor + critic on GPU and sample continuous actions."""
        comptime blocks = (N_ENVS + TPB - 1) // TPB

        var actor_params_t = gpu_state.gpu_actor.params_view()
        var critic_params_t = gpu_state.gpu_critic.params_view()

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        # Reuse actor_logits_buf (sized MB * ACTOR_OUT >= N_ENVS * ACTOR_OUT)
        var actor_out_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTOR_OUT), MutAnyOrigin
        ](gpu_state.actor_logits_buf.unsafe_ptr())
        var actions_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
        ](actions_buf.unsafe_ptr())
        var log_probs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
        ](log_probs_buf.unsafe_ptr())
        var values_t = LayoutTensor[
            dtype,
            Layout.row_major(N_ENVS, Self.CriticModel.OUT_DIM),
            MutAnyOrigin,
        ](values_buf.unsafe_ptr())

        Self.ActorModel.forward_gpu_no_cache[N_ENVS](
            ctx,
            actor_out_t,
            obs_t,
            actor_params_t,
            gpu_state.actor_env_workspace_buf,
        )
        Self.CriticModel.forward_gpu_no_cache[N_ENVS](
            ctx,
            values_t,
            obs_t,
            critic_params_t,
            gpu_state.critic_env_workspace_buf,
        )

        comptime sample_wrapper = _sample_continuous_actions_kernel[
            dtype, N_ENVS, Self.ACTIONS
        ]
        ctx.enqueue_function[sample_wrapper, sample_wrapper](
            actor_out_t,
            actions_t,
            log_probs_t,
            Scalar[DType.uint32](rng_seed),
            grid_dim=(blocks,),
            block_dim=(TPB,),
        )

    fn compute_advantages_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        final_obs_buf: DeviceBuffer[dtype],
    ) raises -> None:
        """Compute GAE advantages from the collected rollout (CPU-side)."""
        comptime ROLLOUT_TOTAL = Self.TOTAL_ROLLOUT_SIZE

        var critic_params_t = gpu_state.gpu_critic.params_view()
        var final_obs_t = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs, Self.OBS), MutAnyOrigin
        ](final_obs_buf.unsafe_ptr())
        var bootstrap_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.n_envs, Self.CriticModel.OUT_DIM),
            MutAnyOrigin,
        ](gpu_state.values_env_buf.unsafe_ptr())

        # Forward critic on final obs to get bootstrap values
        Self.CriticModel.forward_gpu_no_cache[Self.n_envs](
            ctx,
            bootstrap_t,
            final_obs_t,
            critic_params_t,
            gpu_state.critic_env_workspace_buf,
        )

        # Copy rollout data to host for GAE computation
        ctx.enqueue_copy(
            gpu_state.bootstrap_values_host, gpu_state.values_env_buf
        )
        ctx.enqueue_copy(
            gpu_state.rollout_rewards_host, gpu_state.rollout_rewards_buf
        )
        ctx.enqueue_copy(
            gpu_state.rollout_values_host, gpu_state.rollout_values_buf
        )
        ctx.enqueue_copy(
            gpu_state.rollout_dones_host, gpu_state.rollout_dones_buf
        )
        ctx.synchronize()

        # Reward normalization (CleanRL-style).
        if self.normalize_rewards:
            # --- Inline RunningMeanStd.update ---
            self.reward_rms.update(
                gpu_state.rollout_rewards_host, ROLLOUT_TOTAL
            )
            self.reward_rms.normalize(
                gpu_state.rollout_rewards_host, ROLLOUT_TOTAL
            )

        # GAE computation per environment
        for env_idx in range(Self.n_envs):
            var gae = Scalar[dtype](0.0)
            var gae_decay = Scalar[dtype](self.gamma * self.gae_lambda)
            var bootstrap_val = Scalar[dtype](
                gpu_state.bootstrap_values_host[env_idx]
            )

            for t in range(Self.ROLLOUT - 1, -1, -1):
                var idx = t * Self.n_envs + env_idx
                var reward = gpu_state.rollout_rewards_host[idx]
                var value = gpu_state.rollout_values_host[idx]
                var done = gpu_state.rollout_dones_host[idx]

                var next_val: Scalar[dtype]
                if t == Self.ROLLOUT - 1:
                    next_val = bootstrap_val
                else:
                    var next_idx = (t + 1) * Self.n_envs + env_idx
                    next_val = gpu_state.rollout_values_host[next_idx]

                if done > Scalar[dtype](0.5):
                    next_val = Scalar[dtype](0.0)
                    gae = Scalar[dtype](0.0)

                var delta = (
                    reward + Scalar[dtype](self.gamma) * next_val - value
                )
                gae = delta + gae_decay * gae
                gpu_state.advantages_host[idx] = gae
                gpu_state.returns_host[idx] = gae + value

        # Normalize advantages globally if requested
        if self.normalize_advantages:
            var mean = Scalar[dtype](0.0)
            var var_sum = Scalar[dtype](0.0)
            for i in range(ROLLOUT_TOTAL):
                mean += gpu_state.advantages_host[i]
            mean /= Scalar[dtype](ROLLOUT_TOTAL)
            for i in range(ROLLOUT_TOTAL):
                var diff = gpu_state.advantages_host[i] - mean
                var_sum += diff * diff
            var std = sqrt(
                var_sum / Scalar[dtype](ROLLOUT_TOTAL) + Scalar[dtype](1e-8)
            )
            for i in range(ROLLOUT_TOTAL):
                gpu_state.advantages_host[i] = (
                    gpu_state.advantages_host[i] - mean
                ) / (std + Scalar[dtype](1e-8))

        # Copy advantages and returns to GPU
        ctx.enqueue_copy(gpu_state.advantages_buf, gpu_state.advantages_host)
        ctx.enqueue_copy(gpu_state.returns_buf, gpu_state.returns_host)
        ctx.synchronize()

    fn update_epochs_gpu(
        mut self,
        ctx: DeviceContext,
        mut gpu_state: Self.GPUStateType,
        update_idx: Int,
    ) raises -> None:
        """Run PPO continuous multi-epoch minibatch updates on GPU."""
        comptime ROLLOUT_TOTAL = Self.TOTAL_ROLLOUT_SIZE
        comptime MINIBATCH = Self.GPU_MINIBATCH
        comptime MINIBATCH_BLOCKS = (MINIBATCH + TPB - 1) // TPB
        comptime ACTOR_PARAMS = Self.ActorModel.PARAM_SIZE
        comptime CRITIC_PARAMS = Self.CriticModel.PARAM_SIZE
        comptime ACTOR_GRAD_BLOCKS = (ACTOR_PARAMS + TPB - 1) // TPB
        comptime CRITIC_GRAD_BLOCKS = (CRITIC_PARAMS + TPB - 1) // TPB

        var actor_params_t = gpu_state.gpu_actor.params_view()
        var actor_grads_t = gpu_state.gpu_actor.grads_view()
        var critic_params_t = gpu_state.gpu_critic.params_view()
        var critic_grads_t = gpu_state.gpu_critic.grads_view()

        var mb_obs_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, Self.OBS), MutAnyOrigin
        ](gpu_state.mb_obs_buf.unsafe_ptr())
        var mb_actions_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.mb_actions_buf.unsafe_ptr())
        var mb_advantages_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](gpu_state.mb_advantages_buf.unsafe_ptr())
        var mb_returns_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](gpu_state.mb_returns_buf.unsafe_ptr())
        var mb_old_log_probs_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](gpu_state.mb_old_log_probs_buf.unsafe_ptr())
        var mb_old_values_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](gpu_state.mb_old_values_buf.unsafe_ptr())
        var mb_indices_t = LayoutTensor[
            DType.int32, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](gpu_state.mb_indices_buf.unsafe_ptr())

        var rollout_obs_t = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL, Self.OBS), MutAnyOrigin
        ](gpu_state.rollout_obs_buf.unsafe_ptr())
        var rollout_actions_t = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.rollout_actions_buf.unsafe_ptr())
        var advantages_t = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](gpu_state.advantages_buf.unsafe_ptr())
        var returns_t = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](gpu_state.returns_buf.unsafe_ptr())
        var rollout_log_probs_t = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](gpu_state.rollout_log_probs_buf.unsafe_ptr())
        var rollout_values_t = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
        ](gpu_state.rollout_values_buf.unsafe_ptr())

        var actor_output_mb_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, Self.ACTOR_OUT), MutAnyOrigin
        ](gpu_state.actor_logits_buf.unsafe_ptr())
        var actor_grad_output_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, Self.ACTOR_OUT), MutAnyOrigin
        ](gpu_state.actor_grad_output_buf.unsafe_ptr())
        var actor_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(MINIBATCH, Self.ActorModel.CACHE_SIZE),
            MutAnyOrigin,
        ](gpu_state.actor_cache_buf.unsafe_ptr())
        var actor_grad_input_t = LayoutTensor[
            dtype,
            Layout.row_major(MINIBATCH, Self.ActorModel.IN_DIM),
            MutAnyOrigin,
        ](gpu_state.actor_grad_input_buf.unsafe_ptr())

        var critic_values_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, 1), MutAnyOrigin
        ](gpu_state.critic_values_buf.unsafe_ptr())
        var critic_grad_output_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, 1), MutAnyOrigin
        ](gpu_state.critic_grad_output_buf.unsafe_ptr())
        var critic_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(MINIBATCH, Self.CriticModel.CACHE_SIZE),
            MutAnyOrigin,
        ](gpu_state.critic_cache_buf.unsafe_ptr())
        var critic_grad_input_t = LayoutTensor[
            dtype,
            Layout.row_major(MINIBATCH, Self.CriticModel.IN_DIM),
            MutAnyOrigin,
        ](gpu_state.critic_grad_input_buf.unsafe_ptr())

        var kl_divergences_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](gpu_state.kl_divergences_buf.unsafe_ptr())
        var actor_grad_partial_sums_t = LayoutTensor[
            dtype, Layout.row_major(ACTOR_GRAD_BLOCKS), MutAnyOrigin
        ](gpu_state.actor_grad_partial_sums_buf.unsafe_ptr())
        var critic_grad_partial_sums_t = LayoutTensor[
            dtype, Layout.row_major(CRITIC_GRAD_BLOCKS), MutAnyOrigin
        ](gpu_state.critic_grad_partial_sums_buf.unsafe_ptr())

        # LR annealing (linear decay, CleanRL-style)
        if self.anneal_lr and self.target_total_steps > 0:
            var total_updates = self.target_total_steps // Int(ROLLOUT_TOTAL)
            if total_updates > 0:
                var progress = Float64(update_idx) / Float64(total_updates)
                if progress > 1.0:
                    progress = 1.0
                var lr_scale = 1.0 - progress
                gpu_state.gpu_actor.set_lr_scale(lr_scale)
                gpu_state.gpu_critic.set_lr_scale(lr_scale)

        # Entropy annealing
        var current_entropy_coef = self.entropy_coef
        if self.anneal_entropy and self.target_total_steps > 0:
            var estimated_steps = update_idx * ROLLOUT_TOTAL
            var progress = Float64(estimated_steps) / Float64(
                self.target_total_steps
            )
            if progress > 1.0:
                progress = 1.0
            current_entropy_coef = self.entropy_coef * (1.0 - progress)

        # Kernel wrappers
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
        comptime normalize_advantages_fused_wrapper = normalize_advantages_fused_kernel[
            dtype, MINIBATCH, TPB
        ]
        comptime actor_grad_norm_wrapper = gradient_norm_kernel[
            dtype, ACTOR_PARAMS, ACTOR_GRAD_BLOCKS, TPB
        ]
        comptime critic_grad_norm_wrapper = gradient_norm_kernel[
            dtype, CRITIC_PARAMS, CRITIC_GRAD_BLOCKS, TPB
        ]
        comptime actor_reduce_apply_fused_wrapper = gradient_reduce_apply_fused_kernel[
            dtype, ACTOR_PARAMS, ACTOR_GRAD_BLOCKS, TPB
        ]
        comptime critic_reduce_apply_fused_wrapper = gradient_reduce_apply_fused_kernel[
            dtype, CRITIC_PARAMS, CRITIC_GRAD_BLOCKS, TPB
        ]
        comptime LOG_STD_OFFSET_IN_ACTOR = (
            Self.OBS * Self.HIDDEN
            + Self.HIDDEN
            + Self.HIDDEN * Self.HIDDEN
            + Self.HIDDEN
            + Self.HIDDEN * Self.ACTIONS
            + Self.ACTIONS
        )
        comptime clamp_log_std_wrapper = clamp_log_std_params_kernel[
            dtype,
            ACTOR_PARAMS,
            LOG_STD_OFFSET_IN_ACTOR,
            Self.ACTIONS,
        ]

        var kl_early_stop = False

        for epoch in range(self.num_epochs):
            if kl_early_stop:
                break

            # Generate shuffled indices
            var indices_list = List[Int]()
            for i in range(ROLLOUT_TOTAL):
                indices_list.append(i)
            for i in range(ROLLOUT_TOTAL - 1, 0, -1):
                var j = Int(random_float64() * Float64(i + 1))
                var temp = indices_list[i]
                indices_list[i] = indices_list[j]
                indices_list[j] = temp

            var num_minibatches = ROLLOUT_TOTAL // MINIBATCH
            for mb_idx in range(num_minibatches):
                var start_idx = mb_idx * MINIBATCH

                for i in range(MINIBATCH):
                    gpu_state.mb_indices_host[i] = Int32(
                        indices_list[start_idx + i]
                    )
                ctx.enqueue_copy(
                    gpu_state.mb_indices_buf, gpu_state.mb_indices_host
                )

                # Gather minibatch data
                ctx.enqueue_function[gather_wrapper, gather_wrapper](
                    mb_obs_t,
                    mb_actions_t,
                    mb_advantages_t,
                    mb_returns_t,
                    mb_old_log_probs_t,
                    mb_old_values_t,
                    rollout_obs_t,
                    rollout_actions_t,
                    advantages_t,
                    returns_t,
                    rollout_log_probs_t,
                    rollout_values_t,
                    mb_indices_t,
                    MINIBATCH,
                    grid_dim=(MINIBATCH_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Per-minibatch advantage normalization (fully GPU fused)
                if self.norm_adv_per_minibatch:
                    ctx.enqueue_function[
                        normalize_advantages_fused_wrapper,
                        normalize_advantages_fused_wrapper,
                    ](
                        mb_advantages_t,
                        grid_dim=(1,),
                        block_dim=(TPB,),
                    )

                # Actor forward
                gpu_state.gpu_actor.zero_grads(ctx)
                Self.ActorModel.forward_gpu[MINIBATCH](
                    ctx,
                    actor_output_mb_t,
                    mb_obs_t,
                    actor_params_t,
                    actor_cache_t,
                    gpu_state.actor_mb_workspace_buf,
                )

                # Actor grad kernel
                ctx.enqueue_function[actor_grad_wrapper, actor_grad_wrapper](
                    actor_grad_output_t,
                    kl_divergences_t,
                    actor_output_mb_t,
                    mb_old_log_probs_t,
                    mb_advantages_t,
                    mb_actions_t,
                    Scalar[dtype](self.clip_epsilon),
                    Scalar[dtype](current_entropy_coef),
                    MINIBATCH,
                    grid_dim=(MINIBATCH_BLOCKS,),
                    block_dim=(TPB,),
                )

                # KL early stopping
                if self.target_kl > 0.0:
                    ctx.synchronize()
                    ctx.enqueue_copy(
                        gpu_state.kl_divergences_host,
                        gpu_state.kl_divergences_buf,
                    )
                    ctx.synchronize()
                    var kl_sum = Scalar[dtype](0.0)
                    for i in range(MINIBATCH):
                        kl_sum += gpu_state.kl_divergences_host[i]
                    var mean_kl = Float64(kl_sum) / Float64(MINIBATCH)
                    if mean_kl > self.target_kl:
                        kl_early_stop = True
                        break

                # Actor backward
                Self.ActorModel.backward_gpu[MINIBATCH](
                    ctx,
                    actor_grad_input_t,
                    actor_grad_output_t,
                    actor_params_t,
                    actor_cache_t,
                    actor_grads_t,
                    gpu_state.actor_mb_workspace_buf,
                )

                # Gradient clipping for actor (fused, 2 kernels)
                if self.max_grad_norm > 0.0:
                    ctx.enqueue_function[
                        actor_grad_norm_wrapper, actor_grad_norm_wrapper
                    ](
                        actor_grad_partial_sums_t,
                        actor_grads_t,
                        grid_dim=(ACTOR_GRAD_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.enqueue_function[
                        actor_reduce_apply_fused_wrapper,
                        actor_reduce_apply_fused_wrapper,
                    ](
                        actor_grads_t,
                        actor_grad_partial_sums_t,
                        Scalar[dtype](self.max_grad_norm),
                        grid_dim=(ACTOR_GRAD_BLOCKS,),
                        block_dim=(TPB,),
                    )

                # Actor optimizer step + clamp log_std
                gpu_state.gpu_actor.optimizer_step(ctx)
                ctx.enqueue_function[
                    clamp_log_std_wrapper, clamp_log_std_wrapper
                ](
                    actor_params_t,
                    grid_dim=(1,),
                    block_dim=(Self.ACTIONS,),
                )

                # Critic forward
                gpu_state.gpu_critic.zero_grads(ctx)
                Self.CriticModel.forward_gpu[MINIBATCH](
                    ctx,
                    critic_values_t,
                    mb_obs_t,
                    critic_params_t,
                    critic_cache_t,
                    gpu_state.critic_mb_workspace_buf,
                )

                # Critic grad kernel
                comptime if Self.clip_value:
                    ctx.enqueue_function[
                        critic_grad_clipped_wrapper,
                        critic_grad_clipped_wrapper,
                    ](
                        critic_grad_output_t,
                        critic_values_t,
                        mb_returns_t,
                        mb_old_values_t,
                        Scalar[dtype](self.clip_epsilon),
                        Scalar[dtype](self.value_loss_coef),
                        MINIBATCH,
                        grid_dim=(MINIBATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )
                else:
                    ctx.enqueue_function[
                        critic_grad_wrapper, critic_grad_wrapper
                    ](
                        critic_grad_output_t,
                        critic_values_t,
                        mb_returns_t,
                        Scalar[dtype](self.value_loss_coef),
                        MINIBATCH,
                        grid_dim=(MINIBATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )

                # Critic backward
                Self.CriticModel.backward_gpu[MINIBATCH](
                    ctx,
                    critic_grad_input_t,
                    critic_grad_output_t,
                    critic_params_t,
                    critic_cache_t,
                    critic_grads_t,
                    gpu_state.critic_mb_workspace_buf,
                )

                # Gradient clipping for critic (fused, 2 kernels)
                if self.max_grad_norm > 0.0:
                    ctx.enqueue_function[
                        critic_grad_norm_wrapper, critic_grad_norm_wrapper
                    ](
                        critic_grad_partial_sums_t,
                        critic_grads_t,
                        grid_dim=(CRITIC_GRAD_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.enqueue_function[
                        critic_reduce_apply_fused_wrapper,
                        critic_reduce_apply_fused_wrapper,
                    ](
                        critic_grads_t,
                        critic_grad_partial_sums_t,
                        Scalar[dtype](self.max_grad_norm),
                        grid_dim=(CRITIC_GRAD_BLOCKS,),
                        block_dim=(TPB,),
                    )

                # Critic optimizer step
                gpu_state.gpu_critic.optimizer_step(ctx)
                ctx.synchronize()

        # Reset rollout step so next rollout collection starts from position 0
        gpu_state.rollout_step = 0
        self.train_step_count += 1

    # =========================================================================
    # GPU Training with GPU Environments (Fully GPU)
    # =========================================================================

    fn train_gpu[
        EnvType: GPUContinuousEnv,
        CurriculumType: CurriculumScheduler = NoCurriculumScheduler,
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
        var checkpoint_path = self.checkpoint_path
        var checkpoint_every = self.checkpoint_every
        return run_onpolicy_continuous_train_gpu[EnvType, Self, CurriculumType](
            self,
            ctx,
            num_updates=10_000_000,
            target_episodes=num_episodes,
            target_total_steps=self.target_total_steps,
            checkpoint_every=checkpoint_every,
            checkpoint_path=checkpoint_path,
            algorithm_name="PPO Continuous (GPU)",
            environment_name=EnvType.NAME,
            verbose=verbose,
            print_every=print_every,
        )

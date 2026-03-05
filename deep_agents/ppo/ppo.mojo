"""Deep PPO (Proximal Policy Optimization) Agent using the new trait-based architecture.

This PPO implementation uses:
- Network wrapper from nn.training for stateless model + params management
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

Features:
 - Per-minibatch advantage normalization (norm_adv_per_minibatch)
 - Value clipping for stable critic updates (clip_value)

Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
"""

from std.math import exp, log, sqrt
from std.random import random_float64, seed
from std.time import perf_counter_ns
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor

from nn.constants import dtype, TILE, TPB
from nn.model import Linear, ReLU, LinearReLU, Sequential
from nn.optimizer import Adam
from nn.initializer import Xavier
from nn.training import Network, NetworkState, GPUNetworkState
from nn.checkpoint import (
    split_lines,
    find_section_start,
    save_checkpoint_file,
    read_checkpoint_file,
)
from nn.gpu import (
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
from core import (
    TrainingMetrics,
    BoxDiscreteActionEnv,
    BoxContinuousActionEnv,
    GPUDiscreteEnv,
    RenderableEnv,
)
from core.utils.gae import compute_gae_inline
from core.utils.softmax import (
    softmax_inline,
    sample_from_probs_inline,
    argmax_probs_inline,
)
from core.utils.normalization import normalize_inline
from core.utils.shuffle import shuffle_indices_inline
from deep_agents.ppo.state import PPODiscreteState, PPODiscreteGPUState
from deep_agents.core.onpolicy_train import OnPolicyAgent, OnPolicyDiscreteAgent
from deep_agents.core.onpolicy_helpers import (
    compute_gae_list,
    normalize_advantages_list,
    fisher_yates_shuffle,
)
from deep_agents.core.gpu_onpolicy_train import (
    GPUOnPolicyDiscreteAgent,
    run_onpolicy_discrete_train_gpu,
)
from deep_agents.core.eval import (
    run_onpolicy_discrete_eval,
    run_onpolicy_continuous_eval,
)
from deep_agents.core.onpolicy_train import run_onpolicy_discrete_train

from .kernels import (
    ppo_actor_grad_kernel,
    ppo_actor_grad_with_kl_kernel,
    ppo_critic_grad_kernel,
    ppo_critic_grad_clipped_kernel,
    normalize_advantages_kernel,
    _store_post_step_kernel,
    _extract_obs_from_state_kernel,
    _store_pre_step_kernel,
    ppo_gather_minibatch_kernel,
    gradient_norm_kernel,
    gradient_clip_kernel,
    gradient_reduce_and_compute_scale_kernel,
    gradient_apply_scale_kernel,
    gradient_reduce_apply_fused_kernel,
)

# =============================================================================
# Deep PPO Agent
# =============================================================================


struct DeepPPOAgent[
    obs_dim: Int,
    num_actions: Int,
    hidden_dim: Int = 64,
    rollout_len: Int = 128,
    n_envs: Int = 1024,
    gpu_minibatch_size: Int = 256,
    actor_lr: Float64 = 0.0003,
    critic_lr: Float64 = 0.001,
](OnPolicyDiscreteAgent, OnPolicyAgent, GPUOnPolicyDiscreteAgent):
    """Deep Proximal Policy Optimization Agent using new trait-based architecture.

    Uses clipped surrogate objective for stable policy updates:
    L^CLIP = min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)
    where r(θ) = π_θ(a|s) / π_θ_old(a|s)

    Parameters:
        obs_dim: Observation dimension.
        num_actions: Action dimension.
        hidden_dim: Hidden layer size (default: 64).
        rollout_len: Steps per rollout per environment (default: 128 for GPU).
        n_envs: Number of parallel environments for GPU training.
        gpu_minibatch_size: Minibatch size for GPU training.
        actor_lr: Actor learning rate.
        critic_lr: Critic learning rate.

    Note on GPU training:
        - n_envs: Parallel environments on GPU (affects data collection rate)
        - rollout_len: Steps before training (total transitions = n_envs × rollout_len)
        - gpu_minibatch_size: Samples per gradient update
    """

    # Convenience aliases
    comptime OBS = Self.obs_dim
    comptime ACTIONS = Self.num_actions
    comptime HIDDEN = Self.hidden_dim
    comptime ROLLOUT = Self.rollout_len

    # GPU-specific sizes
    comptime TOTAL_ROLLOUT_SIZE: Int = Self.n_envs * Self.rollout_len
    comptime GPU_MINIBATCH = Self.gpu_minibatch_size

    # Actor model and network (stateless ops)
    comptime ActorModel = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, Self.ACTIONS],
    ]
    comptime ActorNet = Network[Self.ActorModel, Adam[Self.actor_lr]]

    # Critic model and network (stateless ops)
    comptime CriticModel = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, 1],
    ]
    comptime CriticNet = Network[Self.CriticModel, Adam[Self.critic_lr]]

    # GPUOnPolicyDiscreteAgent trait constants
    comptime OBS_DIM: Int = Self.obs_dim
    comptime NUM_ACTIONS: Int = Self.num_actions
    comptime ROLLOUT_LEN: Int = Self.rollout_len
    comptime MAX_N_ENVS: Int = Self.n_envs

    # Compile-time state type (actor + critic + rollout buffers)
    comptime ActorOpt = Adam[Self.actor_lr]
    comptime CriticOpt = Adam[Self.critic_lr]
    comptime CPUStateType = PPODiscreteState[
        Self.ActorModel,
        Self.ActorOpt,
        Self.CriticModel,
        Self.CriticOpt,
        Self.OBS,
        Self.ACTIONS,
        Self.ROLLOUT,
    ]
    comptime GPUStateType = PPODiscreteGPUState[
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

    # CPU state (actor + critic networks + rollout buffers)
    var state: Self.CPUStateType

    # Hyperparameters
    var gamma: Float64
    var gae_lambda: Float64
    var clip_epsilon: Float64
    var entropy_coef: Float64
    var value_loss_coef: Float64
    var num_epochs: Int
    var minibatch_size: Int
    var normalize_advantages: Bool

    # Advanced hyperparameters (environment-agnostic improvements)
    var target_kl: Float64  # KL threshold for early epoch stopping
    var max_grad_norm: Float64  # Gradient clipping threshold
    var anneal_lr: Bool  # Whether to linearly anneal learning rate
    var anneal_entropy: Bool  # Whether to anneal entropy coefficient
    var target_total_steps: Int  # Target steps for annealing (0 = auto-calculate)
    var clip_value: Bool  # Whether to clip value function updates
    var norm_adv_per_minibatch: Bool  # Normalize advantages per minibatch

    # Training state
    var train_step_count: Int

    # Auto-checkpoint settings
    var checkpoint_every: Int  # Save checkpoint every N episodes (0 to disable)
    var checkpoint_path: String  # Path for auto-checkpointing

    fn __init__(
        out self,
        gamma: Float64 = 0.99,
        gae_lambda: Float64 = 0.95,
        clip_epsilon: Float64 = 0.2,
        entropy_coef: Float64 = 0.01,
        value_loss_coef: Float64 = 0.5,
        num_epochs: Int = 4,
        minibatch_size: Int = 64,
        normalize_advantages: Bool = True,
        # Advanced hyperparameters
        target_kl: Float64 = 0.015,
        max_grad_norm: Float64 = 0.5,
        anneal_lr: Bool = True,
        anneal_entropy: Bool = False,
        target_total_steps: Int = 0,
        clip_value: Bool = True,
        norm_adv_per_minibatch: Bool = True,
        # Checkpoint settings
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
    ):
        """Initialize Deep PPO agent.

        Args:
            gamma: Discount factor (default: 0.99).
            gae_lambda: GAE lambda parameter (default: 0.95).
            clip_epsilon: PPO clipping parameter (default: 0.2).
            entropy_coef: Entropy bonus coefficient (default: 0.01).
            value_loss_coef: Value loss coefficient (default: 0.5).
            num_epochs: Number of optimization epochs per update (default: 4).
            minibatch_size: Size of minibatches (default: 64).
            normalize_advantages: Whether to normalize advantages (default: True).
            target_kl: KL divergence threshold for early epoch stopping (default: 0.015).
            max_grad_norm: Maximum gradient norm for clipping (default: 0.5).
            anneal_lr: Whether to linearly anneal learning rate (default: True).
            anneal_entropy: Whether to anneal entropy coefficient (default: False).
            target_total_steps: Target total steps for annealing, 0=auto (default: 0).
            clip_value: Whether to clip value function updates (default: True).
            norm_adv_per_minibatch: Normalize advantages per minibatch (default: True).
            checkpoint_every: Save checkpoint every N episodes (0 to disable).
            checkpoint_path: Path to save checkpoints.
        """
        # Initialize CPU state (actor + critic + rollout buffers)
        self.state = Self.CPUStateType()

        # Store hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.normalize_advantages = normalize_advantages

        # Store advanced hyperparameters
        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm
        self.anneal_lr = anneal_lr
        self.anneal_entropy = anneal_entropy
        self.target_total_steps = target_total_steps
        self.clip_value = clip_value
        self.norm_adv_per_minibatch = norm_adv_per_minibatch

        # Training state
        self.train_step_count = 0

        # Auto-checkpoint settings
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path

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
        var logits_data = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs.unsafe_ptr())
        var logits_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](logits_data.unsafe_ptr())
        var p_actor = self.state.actor.params_view()
        Self.ActorNet.forward[1](obs_t, logits_t, p_actor)

        var logits = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        for i in range(Self.ACTIONS):
            logits[i] = rebind[Scalar[dtype]](logits_t[0, i])

        # Compute softmax probabilities
        var probs = softmax_inline[dtype, Self.ACTIONS](logits)

        # Forward critic to get value
        var value_data = InlineArray[Scalar[dtype], 1](uninitialized=True)
        var value_t = LayoutTensor[dtype, Layout.row_major(1, 1), MutAnyOrigin](
            value_data.unsafe_ptr()
        )
        var p_critic = self.state.critic.params_view()
        Self.CriticNet.forward[1](obs_t, value_t, p_critic)
        var value = rebind[Scalar[dtype]](value_t[0, 0])

        # Sample or select greedy action
        var action: Int
        if training:
            action = sample_from_probs_inline[dtype, Self.ACTIONS](probs)
        else:
            action = argmax_probs_inline[dtype, Self.ACTIONS](probs)

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
            self.state.buffer_obs[self.state.buffer_idx * Self.OBS + i] = obs[i]

        self.state.buffer_actions[self.state.buffer_idx] = action
        self.state.buffer_rewards[self.state.buffer_idx] = Scalar[dtype](reward)
        self.state.buffer_log_probs[self.state.buffer_idx] = log_prob
        self.state.buffer_values[self.state.buffer_idx] = value
        self.state.buffer_dones[self.state.buffer_idx] = done

        self.state.buffer_idx += 1

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
        if self.state.buffer_idx == 0:
            return 0.0

        var buffer_len = self.state.buffer_idx

        # Get bootstrap value
        var next_obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](next_obs.unsafe_ptr())
        var next_val_data = InlineArray[Scalar[dtype], 1](uninitialized=True)
        var next_val_t = LayoutTensor[
            dtype, Layout.row_major(1, 1), MutAnyOrigin
        ](next_val_data.unsafe_ptr())
        var p_critic = self.state.critic.params_view()
        Self.CriticNet.forward[1](next_obs_t, next_val_t, p_critic)
        var next_value = rebind[Scalar[dtype]](next_val_t[0, 0])

        # Compute GAE advantages and returns (inline for List compatibility)
        var advantages = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        var returns = List[Scalar[dtype]](capacity=Self.ROLLOUT)
        for _ in range(buffer_len):
            advantages.append(Scalar[dtype](0))
            returns.append(Scalar[dtype](0))

        # GAE computation
        var gae = Scalar[dtype](0.0)
        var gae_decay = Scalar[dtype](self.gamma * self.gae_lambda)
        for t in range(buffer_len - 1, -1, -1):
            var next_val: Scalar[dtype]
            if t == buffer_len - 1:
                next_val = next_value
            else:
                next_val = self.state.buffer_values[t + 1]

            # Reset GAE at episode boundary
            if self.state.buffer_dones[t]:
                next_val = Scalar[dtype](0.0)
                gae = Scalar[dtype](0.0)

            # TD residual: δ = r + γV(s') - V(s)
            var delta = (
                self.state.buffer_rewards[t]
                + Scalar[dtype](self.gamma) * next_val
                - self.state.buffer_values[t]
            )

            # GAE accumulation: A = δ + γλA'
            gae = delta + gae_decay * gae

            advantages[t] = gae
            returns[t] = gae + self.state.buffer_values[t]

        # Normalize advantages
        if self.normalize_advantages and buffer_len > 1:
            var mean = Scalar[dtype](0.0)
            for i in range(buffer_len):
                mean += advantages[i]
            mean /= Scalar[dtype](buffer_len)

            var var_sum = Scalar[dtype](0.0)
            for i in range(buffer_len):
                var diff = advantages[i] - mean
                var_sum += diff * diff

            var std = sqrt(
                var_sum / Scalar[dtype](buffer_len) + Scalar[dtype](1e-8)
            )
            for i in range(buffer_len):
                advantages[i] = (advantages[i] - mean) / std

        # =====================================================================
        # Multiple epochs of optimization
        # =====================================================================

        var total_loss = Scalar[dtype](0.0)
        var indices = List[Int](capacity=buffer_len)
        for i in range(buffer_len):
            indices.append(i)

        for epoch in range(self.num_epochs):
            # Shuffle indices for minibatch sampling using Fisher-Yates
            for i in range(buffer_len - 1, 0, -1):
                var j = Int(random_float64() * Float64(i + 1))
                var temp = indices[i]
                indices[i] = indices[j]
                indices[j] = temp

            var batch_start = 0
            while batch_start < buffer_len:
                var batch_end = batch_start + self.minibatch_size
                if batch_end > buffer_len:
                    batch_end = buffer_len

                var mb_size = batch_end - batch_start

                # Per-minibatch advantage normalization
                var mb_advantages = List[Scalar[dtype]](capacity=mb_size)
                for b in range(batch_start, batch_end):
                    var t = indices[b]
                    mb_advantages.append(advantages[t])

                if self.norm_adv_per_minibatch and mb_size > 1:
                    var mb_mean = Scalar[dtype](0.0)
                    for i in range(mb_size):
                        mb_mean += mb_advantages[i]
                    mb_mean /= Scalar[dtype](mb_size)

                    var mb_var_sum = Scalar[dtype](0.0)
                    for i in range(mb_size):
                        var diff = mb_advantages[i] - mb_mean
                        mb_var_sum += diff * diff

                    var mb_std = sqrt(
                        mb_var_sum / Scalar[dtype](mb_size)
                        + Scalar[dtype](1e-8)
                    )
                    for i in range(mb_size):
                        mb_advantages[i] = (mb_advantages[i] - mb_mean) / mb_std

                # Process minibatch
                for b in range(batch_start, batch_end):
                    var t = indices[b]
                    var mb_idx = b - batch_start

                    # Get observation for this timestep
                    var obs = InlineArray[Scalar[dtype], Self.OBS](fill=0)
                    for i in range(Self.OBS):
                        obs[i] = self.state.buffer_obs[t * Self.OBS + i]

                    var action = self.state.buffer_actions[t]
                    var old_log_prob = self.state.buffer_log_probs[t]
                    var old_value = self.state.buffer_values[t]
                    var advantage = mb_advantages[mb_idx]
                    var return_t = returns[t]

                    # ==========================================================
                    # Actor forward and update
                    # ==========================================================
                    var logits_data = InlineArray[Scalar[dtype], Self.ACTIONS](
                        uninitialized=True
                    )
                    var obs_tensor = LayoutTensor[
                        dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
                    ](obs.unsafe_ptr())
                    var logits_tensor = LayoutTensor[
                        dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
                    ](logits_data.unsafe_ptr())
                    var p_actor = self.state.actor.params_view()
                    Self.ActorNet.forward[1](obs_tensor, logits_tensor, p_actor)

                    var logits = InlineArray[Scalar[dtype], Self.ACTIONS](
                        uninitialized=True
                    )
                    for i in range(Self.ACTIONS):
                        logits[i] = rebind[Scalar[dtype]](logits_tensor[0, i])

                    var probs = softmax_inline[dtype, Self.ACTIONS](logits)
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

                    # Backward through actor (heap-allocated cache to avoid stack overflow)
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
                        obs_tensor, logits_tensor, p_actor, actor_cache_t
                    )

                    var d_logits_tensor = LayoutTensor[
                        dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
                    ](d_logits.unsafe_ptr())
                    var actor_grad_input = InlineArray[Scalar[dtype], Self.OBS](
                        fill=0
                    )
                    var actor_grad_input_tensor = LayoutTensor[
                        dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
                    ](actor_grad_input.unsafe_ptr())
                    var g_actor = self.state.actor.grads_view()
                    self.state.actor.zero_grads()
                    Self.ActorNet.backward[1](
                        d_logits_tensor,
                        actor_grad_input_tensor,
                        p_actor,
                        actor_cache_t,
                        g_actor,
                    )
                    self.state.actor.optimizer_step()

                    # ==========================================================
                    # Critic forward and update
                    # ==========================================================
                    var value_data = InlineArray[Scalar[dtype], 1](
                        uninitialized=True
                    )
                    var value_out_t = LayoutTensor[
                        dtype, Layout.row_major(1, 1), MutAnyOrigin
                    ](value_data.unsafe_ptr())
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
                        obs_tensor, value_out_t, p_critic_u, critic_cache_t
                    )

                    var value = rebind[Scalar[dtype]](value_out_t[0, 0])

                    # Value loss: (return - value)^2
                    var value_loss = (return_t - value) * (return_t - value)

                    # Critic gradient (with optional value clipping)
                    var d_value = InlineArray[Scalar[dtype], 1](fill=0)
                    if self.clip_value:
                        # Clipped value function
                        var v_diff = value - old_value
                        var v_clipped: Scalar[dtype]
                        if v_diff > Scalar[dtype](self.clip_epsilon):
                            v_clipped = old_value + Scalar[dtype](
                                self.clip_epsilon
                            )
                        elif v_diff < -Scalar[dtype](self.clip_epsilon):
                            v_clipped = old_value - Scalar[dtype](
                                self.clip_epsilon
                            )
                        else:
                            v_clipped = value

                        # Unclipped and clipped losses
                        var loss_unclipped = (value - return_t) * (
                            value - return_t
                        )
                        var loss_clipped = (v_clipped - return_t) * (
                            v_clipped - return_t
                        )

                        # Use gradient of the larger loss (pessimistic)
                        if loss_unclipped > loss_clipped:
                            # Gradient of unclipped loss
                            d_value[0] = (
                                Scalar[dtype](2.0)
                                * Scalar[dtype](self.value_loss_coef)
                                * (value - return_t)
                            )
                        else:
                            # Gradient of clipped loss
                            if v_diff > Scalar[dtype](
                                self.clip_epsilon
                            ) or v_diff < -Scalar[dtype](self.clip_epsilon):
                                # v_clipped doesn't depend on value, gradient is 0
                                d_value[0] = Scalar[dtype](0.0)
                            else:
                                # v_clipped = value
                                d_value[0] = (
                                    Scalar[dtype](2.0)
                                    * Scalar[dtype](self.value_loss_coef)
                                    * (value - return_t)
                                )
                    else:
                        # Regular gradient
                        d_value[0] = (
                            Scalar[dtype](2.0)
                            * Scalar[dtype](self.value_loss_coef)
                            * (value - return_t)
                        )

                    # Backward through critic
                    var d_value_t = LayoutTensor[
                        dtype, Layout.row_major(1, 1), MutAnyOrigin
                    ](d_value.unsafe_ptr())
                    var critic_grad_input = InlineArray[
                        Scalar[dtype], Self.OBS
                    ](fill=0)
                    var d_in_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
                    ](critic_grad_input.unsafe_ptr())
                    var g_critic = self.state.critic.grads_view()
                    self.state.critic.zero_grads()
                    Self.CriticNet.backward[1](
                        d_value_t,
                        d_in_t,
                        p_critic_u,
                        critic_cache_t,
                        g_critic,
                    )
                    self.state.critic.optimizer_step()

                    total_loss += (
                        policy_loss
                        + Scalar[dtype](self.value_loss_coef) * value_loss
                        - Scalar[dtype](self.entropy_coef) * entropy
                    )

                batch_start = batch_end

        # Clear buffer
        self.state.buffer_idx = 0
        self.train_step_count += 1

        return Float64(total_loss / Scalar[dtype](self.num_epochs * buffer_len))

    fn _list_to_inline[
        dtype: DType
    ](self, obs_list: List[Scalar[dtype]]) -> InlineArray[
        Scalar[dtype], Self.OBS
    ]:
        """Convert List[Float64] to InlineArray."""
        var obs = InlineArray[Scalar[dtype], Self.OBS](fill=0)
        for i in range(Self.OBS):
            if i < len(obs_list):
                obs[i] = Scalar[dtype](obs_list[i])
        return obs^

    # =========================================================================
    # OnPolicyDiscreteAgent trait conformance
    # =========================================================================

    fn make_cpu_state(self) -> Self.CPUStateType:
        """Allocate a fresh PPODiscreteState with Xavier-initialized networks.
        """
        return Self.CPUStateType()

    fn collect_rollout[
        E: BoxDiscreteActionEnv
    ](mut self, mut cpu_state: Self.CPUStateType, mut env: E) -> None:
        """Collect exactly ROLLOUT_LEN steps into cpu_state rollout buffers.

        Handles episode resets internally. Stores obs, action, reward,
        value, log_prob, done in cpu_state buffers. After this call,
        cpu_state._current_obs holds the last observation for bootstrapping.
        """
        if not cpu_state._env_initialized:
            var obs_list = env.reset_obs_list()
            for i in range(Self.OBS):
                cpu_state._current_obs[i] = Scalar[dtype](obs_list[i])
            cpu_state._env_initialized = True

        cpu_state.buffer_idx = 0

        for _ in range(Self.ROLLOUT):
            var obs = InlineArray[Scalar[dtype], Self.OBS](uninitialized=True)
            for i in range(Self.OBS):
                obs[i] = cpu_state._current_obs[i]

            var action_result = self.select_action_state(
                cpu_state, obs, training=True
            )
            var action = action_result[0]
            var log_prob = action_result[1]
            var value = action_result[2]

            var result = env.step_obs(action)
            var reward = Float64(result[1])
            var done = result[2]

            # Store in buffer
            for i in range(Self.OBS):
                cpu_state.buffer_obs[cpu_state.buffer_idx * Self.OBS + i] = obs[
                    i
                ]
            cpu_state.buffer_actions[cpu_state.buffer_idx] = action
            cpu_state.buffer_rewards[cpu_state.buffer_idx] = Scalar[dtype](
                reward
            )
            cpu_state.buffer_values[cpu_state.buffer_idx] = value
            cpu_state.buffer_log_probs[cpu_state.buffer_idx] = log_prob
            cpu_state.buffer_dones[cpu_state.buffer_idx] = done
            cpu_state.buffer_idx += 1

            if done:
                var next_obs_list = env.reset_obs_list()
                for i in range(Self.OBS):
                    cpu_state._current_obs[i] = Scalar[dtype](next_obs_list[i])
            else:
                for i in range(Self.OBS):
                    cpu_state._current_obs[i] = Scalar[dtype](result[0][i])

    fn compute_advantages(mut self, mut cpu_state: Self.CPUStateType) -> None:
        """Compute GAE advantages and returns using cpu_state._current_obs to bootstrap.

        Fills cpu_state._advantages and cpu_state._returns in-place.
        Optionally normalizes advantages.
        """
        var buffer_len = cpu_state.buffer_idx
        if buffer_len == 0:
            return

        # Bootstrap value from the last obs
        var next_obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](cpu_state._current_obs.unsafe_ptr())
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
        var buffer_len = cpu_state.buffer_idx
        if buffer_len == 0:
            return 0.0

        # Prepare index list for shuffling
        for i in range(buffer_len):
            cpu_state._indices[i] = i

        var total_loss = Scalar[dtype](0.0)

        for epoch in range(self.num_epochs):
            # Shuffle indices (Fisher-Yates)
            fisher_yates_shuffle(cpu_state._indices, buffer_len)

            var batch_start = 0
            while batch_start < buffer_len:
                var batch_end = batch_start + self.minibatch_size
                if batch_end > buffer_len:
                    batch_end = buffer_len
                var mb_size = batch_end - batch_start

                # Per-minibatch advantage normalization
                var mb_advantages = List[Scalar[dtype]](capacity=mb_size)
                for b in range(batch_start, batch_end):
                    var t = cpu_state._indices[b]
                    mb_advantages.append(cpu_state._advantages[t])

                if self.norm_adv_per_minibatch and mb_size > 1:
                    normalize_advantages_list[dtype](mb_advantages, mb_size)

                # Process minibatch samples
                for b in range(batch_start, batch_end):
                    var t = cpu_state._indices[b]
                    var mb_idx = b - batch_start

                    var obs = InlineArray[Scalar[dtype], Self.OBS](fill=0)
                    for i in range(Self.OBS):
                        obs[i] = cpu_state.buffer_obs[t * Self.OBS + i]

                    var action = cpu_state.buffer_actions[t]
                    var old_log_prob = cpu_state.buffer_log_probs[t]
                    var old_value = cpu_state.buffer_values[t]
                    var advantage = mb_advantages[mb_idx]
                    var return_t = cpu_state._returns[t]

                    # Actor forward
                    var logits_data = InlineArray[Scalar[dtype], Self.ACTIONS](
                        uninitialized=True
                    )
                    var obs_tensor = LayoutTensor[
                        dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
                    ](obs.unsafe_ptr())
                    var logits_tensor = LayoutTensor[
                        dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
                    ](logits_data.unsafe_ptr())
                    var p_actor = cpu_state.actor.params_view()
                    Self.ActorNet.forward[1](obs_tensor, logits_tensor, p_actor)

                    var logits = InlineArray[Scalar[dtype], Self.ACTIONS](
                        uninitialized=True
                    )
                    for i in range(Self.ACTIONS):
                        logits[i] = rebind[Scalar[dtype]](logits_tensor[0, i])

                    var probs = softmax_inline[dtype, Self.ACTIONS](logits)
                    var new_log_prob = log(probs[action] + Scalar[dtype](1e-8))
                    var ratio = exp(new_log_prob - old_log_prob)

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

                    var policy_loss: Scalar[dtype]
                    if surr1 < surr2:
                        policy_loss = -surr1
                    else:
                        policy_loss = -surr2

                    var entropy = Scalar[dtype](0.0)
                    for a in range(Self.ACTIONS):
                        if probs[a] > Scalar[dtype](1e-8):
                            entropy -= probs[a] * log(probs[a])

                    var is_clipped = (
                        ratio < Scalar[dtype](1.0 - self.clip_epsilon)
                    ) or (ratio > Scalar[dtype](1.0 + self.clip_epsilon))

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
                            var d_entropy = -probs[a] * (
                                Scalar[dtype](1.0)
                                + log(probs[a] + Scalar[dtype](1e-8))
                            )
                            d_logits[a] = (
                                -advantage * ratio * d_log_prob
                                - Scalar[dtype](self.entropy_coef) * d_entropy
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
                        obs_tensor, logits_tensor, p_actor, actor_cache_t
                    )

                    var d_logits_tensor = LayoutTensor[
                        dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
                    ](d_logits.unsafe_ptr())
                    var actor_grad_input = InlineArray[Scalar[dtype], Self.OBS](
                        fill=0
                    )
                    var actor_grad_input_tensor = LayoutTensor[
                        dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
                    ](actor_grad_input.unsafe_ptr())
                    var g_actor = cpu_state.actor.grads_view()
                    cpu_state.actor.zero_grads()
                    Self.ActorNet.backward[1](
                        d_logits_tensor,
                        actor_grad_input_tensor,
                        p_actor,
                        actor_cache_t,
                        g_actor,
                    )
                    cpu_state.actor.optimizer_step()

                    # Critic forward + update
                    var value_data = InlineArray[Scalar[dtype], 1](
                        uninitialized=True
                    )
                    var value_out_t = LayoutTensor[
                        dtype, Layout.row_major(1, 1), MutAnyOrigin
                    ](value_data.unsafe_ptr())
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
                        obs_tensor, value_out_t, p_critic_u, critic_cache_t
                    )
                    var value = rebind[Scalar[dtype]](value_out_t[0, 0])
                    var value_loss = (return_t - value) * (return_t - value)

                    var d_value = InlineArray[Scalar[dtype], 1](fill=0)
                    if self.clip_value:
                        var v_diff = value - old_value
                        var v_clipped: Scalar[dtype]
                        if v_diff > Scalar[dtype](self.clip_epsilon):
                            v_clipped = old_value + Scalar[dtype](
                                self.clip_epsilon
                            )
                        elif v_diff < -Scalar[dtype](self.clip_epsilon):
                            v_clipped = old_value - Scalar[dtype](
                                self.clip_epsilon
                            )
                        else:
                            v_clipped = value
                        var loss_unclipped = (value - return_t) * (
                            value - return_t
                        )
                        var loss_clipped = (v_clipped - return_t) * (
                            v_clipped - return_t
                        )
                        if loss_unclipped > loss_clipped:
                            d_value[0] = (
                                Scalar[dtype](2.0)
                                * Scalar[dtype](self.value_loss_coef)
                                * (value - return_t)
                            )
                        else:
                            if v_diff > Scalar[dtype](
                                self.clip_epsilon
                            ) or v_diff < -Scalar[dtype](self.clip_epsilon):
                                d_value[0] = Scalar[dtype](0.0)
                            else:
                                d_value[0] = (
                                    Scalar[dtype](2.0)
                                    * Scalar[dtype](self.value_loss_coef)
                                    * (value - return_t)
                                )
                    else:
                        d_value[0] = (
                            Scalar[dtype](2.0)
                            * Scalar[dtype](self.value_loss_coef)
                            * (value - return_t)
                        )

                    var d_value_t = LayoutTensor[
                        dtype, Layout.row_major(1, 1), MutAnyOrigin
                    ](d_value.unsafe_ptr())
                    var critic_grad_input = InlineArray[
                        Scalar[dtype], Self.OBS
                    ](fill=0)
                    var d_in_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
                    ](critic_grad_input.unsafe_ptr())
                    var g_critic = cpu_state.critic.grads_view()
                    cpu_state.critic.zero_grads()
                    Self.CriticNet.backward[1](
                        d_value_t, d_in_t, p_critic_u, critic_cache_t, g_critic
                    )
                    cpu_state.critic.optimizer_step()

                    total_loss += (
                        policy_loss
                        + Scalar[dtype](self.value_loss_coef) * value_loss
                        - Scalar[dtype](self.entropy_coef) * entropy
                    )

                batch_start = batch_end

        cpu_state.buffer_idx = 0
        self.train_step_count += 1
        return Float64(total_loss / Scalar[dtype](self.num_epochs * buffer_len))

    fn select_greedy_action(
        self, cpu_state: Self.CPUStateType, obs: List[Float64]
    ) -> List[Float64]:
        """Select greedy action for evaluation using cpu_state actor."""
        var obs_arr = InlineArray[Scalar[dtype], Self.OBS](uninitialized=True)
        for i in range(Self.OBS):
            obs_arr[i] = Scalar[dtype](obs[i])
        var action_result = self.select_action_state(
            cpu_state, obs_arr, training=False
        )
        var result = List[Float64]()
        result.append(Float64(action_result[0]))
        return result^

    fn get_explore_rate(self) -> Float64:
        """Return entropy coefficient as exploration rate."""
        return self.entropy_coef

    # =========================================================================
    # OnPolicyAgent trait conformance
    # (Used by run_onpolicy_discrete_train simple overload — no aliasing)
    # =========================================================================

    fn collect_rollout[
        E: BoxDiscreteActionEnv
    ](mut self, mut env: E) -> None:
        """Collect ROLLOUT_LEN steps using self.state (OnPolicyAgent overload)."""
        if not self.state._env_initialized:
            var obs_list = env.reset_obs_list()
            for i in range(Self.OBS):
                self.state._current_obs[i] = Scalar[dtype](obs_list[i])
            self.state._env_initialized = True
        self.state.buffer_idx = 0

        for _ in range(Self.ROLLOUT):
            var obs = InlineArray[Scalar[dtype], Self.OBS](uninitialized=True)
            for i in range(Self.OBS):
                obs[i] = self.state._current_obs[i]

            # Inline action selection using self.state (avoids aliasing)
            var logits_data = InlineArray[Scalar[dtype], Self.ACTIONS](
                uninitialized=True
            )
            var obs_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
            ](obs.unsafe_ptr())
            var logits_t = LayoutTensor[
                dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
            ](logits_data.unsafe_ptr())
            var p_actor = self.state.actor.params_view()
            Self.ActorNet.forward[1](obs_t, logits_t, p_actor)
            var logits = InlineArray[Scalar[dtype], Self.ACTIONS](
                uninitialized=True
            )
            for i in range(Self.ACTIONS):
                logits[i] = rebind[Scalar[dtype]](logits_t[0, i])
            var probs = softmax_inline[dtype, Self.ACTIONS](logits)
            var value_data = InlineArray[Scalar[dtype], 1](uninitialized=True)
            var value_t = LayoutTensor[
                dtype, Layout.row_major(1, 1), MutAnyOrigin
            ](value_data.unsafe_ptr())
            var p_critic = self.state.critic.params_view()
            Self.CriticNet.forward[1](obs_t, value_t, p_critic)
            var value = rebind[Scalar[dtype]](value_t[0, 0])
            var action = sample_from_probs_inline[dtype, Self.ACTIONS](probs)
            var log_prob = log(probs[action] + Scalar[dtype](1e-8))

            var result = env.step_obs(action)
            var reward = Float64(result[1])
            var done = result[2]

            for i in range(Self.OBS):
                self.state.buffer_obs[
                    self.state.buffer_idx * Self.OBS + i
                ] = obs[i]
            self.state.buffer_actions[self.state.buffer_idx] = action
            self.state.buffer_rewards[self.state.buffer_idx] = Scalar[dtype](
                reward
            )
            self.state.buffer_values[self.state.buffer_idx] = value
            self.state.buffer_log_probs[self.state.buffer_idx] = log_prob
            self.state.buffer_dones[self.state.buffer_idx] = done
            self.state.buffer_idx += 1

            if done:
                var next_obs_list = env.reset_obs_list()
                for i in range(Self.OBS):
                    self.state._current_obs[i] = Scalar[dtype](
                        next_obs_list[i]
                    )
            else:
                for i in range(Self.OBS):
                    self.state._current_obs[i] = Scalar[dtype](result[0][i])

    fn collect_rollout_continuous[
        E: BoxContinuousActionEnv
    ](mut self, mut env: E) -> None:
        """No-op: discrete PPO does not use continuous environments."""
        pass

    fn compute_advantages(mut self) -> None:
        """Compute GAE advantages using self.state (OnPolicyAgent overload)."""
        var buffer_len = self.state.buffer_idx
        if buffer_len == 0:
            return

        var next_obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](self.state._current_obs.unsafe_ptr())
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
            normalize_advantages_list[dtype](
                self.state._advantages, buffer_len
            )

    fn update_epochs(mut self) -> Float64:
        """Update actor/critic over num_epochs (OnPolicyAgent overload)."""
        var buffer_len = self.state.buffer_idx
        if buffer_len == 0:
            return 0.0

        for i in range(buffer_len):
            self.state._indices[i] = i

        var total_loss = Scalar[dtype](0.0)

        for epoch in range(self.num_epochs):
            fisher_yates_shuffle(self.state._indices, buffer_len)

            var batch_start = 0
            while batch_start < buffer_len:
                var batch_end = batch_start + self.minibatch_size
                if batch_end > buffer_len:
                    batch_end = buffer_len
                var mb_size = batch_end - batch_start

                var mb_advantages = List[Scalar[dtype]](capacity=mb_size)
                for b in range(batch_start, batch_end):
                    var t = self.state._indices[b]
                    mb_advantages.append(self.state._advantages[t])

                if self.norm_adv_per_minibatch and mb_size > 1:
                    normalize_advantages_list[dtype](mb_advantages, mb_size)

                for b in range(batch_start, batch_end):
                    var t = self.state._indices[b]
                    var mb_idx = b - batch_start

                    var obs = InlineArray[Scalar[dtype], Self.OBS](fill=0)
                    for i in range(Self.OBS):
                        obs[i] = self.state.buffer_obs[t * Self.OBS + i]

                    var action = self.state.buffer_actions[t]
                    var old_log_prob = self.state.buffer_log_probs[t]
                    var old_value = self.state.buffer_values[t]
                    var advantage = mb_advantages[mb_idx]
                    var return_t = self.state._returns[t]

                    var logits_data = InlineArray[Scalar[dtype], Self.ACTIONS](
                        uninitialized=True
                    )
                    var obs_tensor = LayoutTensor[
                        dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
                    ](obs.unsafe_ptr())
                    var logits_tensor = LayoutTensor[
                        dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
                    ](logits_data.unsafe_ptr())
                    var p_actor = self.state.actor.params_view()
                    Self.ActorNet.forward[1](obs_tensor, logits_tensor, p_actor)

                    var logits = InlineArray[Scalar[dtype], Self.ACTIONS](
                        uninitialized=True
                    )
                    for i in range(Self.ACTIONS):
                        logits[i] = rebind[Scalar[dtype]](logits_tensor[0, i])

                    var probs = softmax_inline[dtype, Self.ACTIONS](logits)
                    var new_log_prob = log(probs[action] + Scalar[dtype](1e-8))
                    var ratio = exp(new_log_prob - old_log_prob)

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

                    var policy_loss: Scalar[dtype]
                    if surr1 < surr2:
                        policy_loss = -surr1
                    else:
                        policy_loss = -surr2

                    var entropy = Scalar[dtype](0.0)
                    for a in range(Self.ACTIONS):
                        if probs[a] > Scalar[dtype](1e-8):
                            entropy -= probs[a] * log(probs[a])

                    var is_clipped = (
                        ratio < Scalar[dtype](1.0 - self.clip_epsilon)
                    ) or (ratio > Scalar[dtype](1.0 + self.clip_epsilon))

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
                            var d_entropy = -probs[a] * (
                                Scalar[dtype](1.0)
                                + log(probs[a] + Scalar[dtype](1e-8))
                            )
                            d_logits[a] = (
                                -advantage * ratio * d_log_prob
                                - Scalar[dtype](self.entropy_coef) * d_entropy
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
                        obs_tensor, logits_tensor, p_actor, actor_cache_t
                    )

                    var d_logits_tensor = LayoutTensor[
                        dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
                    ](d_logits.unsafe_ptr())
                    var actor_grad_input = InlineArray[
                        Scalar[dtype], Self.OBS
                    ](fill=0)
                    var actor_grad_input_tensor = LayoutTensor[
                        dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
                    ](actor_grad_input.unsafe_ptr())
                    var g_actor = self.state.actor.grads_view()
                    self.state.actor.zero_grads()
                    Self.ActorNet.backward[1](
                        d_logits_tensor,
                        actor_grad_input_tensor,
                        p_actor,
                        actor_cache_t,
                        g_actor,
                    )
                    self.state.actor.optimizer_step()

                    var value_data = InlineArray[Scalar[dtype], 1](
                        uninitialized=True
                    )
                    var value_out_t = LayoutTensor[
                        dtype, Layout.row_major(1, 1), MutAnyOrigin
                    ](value_data.unsafe_ptr())
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
                        obs_tensor, value_out_t, p_critic_u, critic_cache_t
                    )
                    var value = rebind[Scalar[dtype]](value_out_t[0, 0])
                    var value_loss = (return_t - value) * (return_t - value)

                    var d_value = InlineArray[Scalar[dtype], 1](fill=0)
                    if self.clip_value:
                        var v_diff = value - old_value
                        var v_clipped: Scalar[dtype]
                        if v_diff > Scalar[dtype](self.clip_epsilon):
                            v_clipped = old_value + Scalar[dtype](
                                self.clip_epsilon
                            )
                        elif v_diff < -Scalar[dtype](self.clip_epsilon):
                            v_clipped = old_value - Scalar[dtype](
                                self.clip_epsilon
                            )
                        else:
                            v_clipped = value
                        var loss_unclipped = (value - return_t) * (
                            value - return_t
                        )
                        var loss_clipped = (v_clipped - return_t) * (
                            v_clipped - return_t
                        )
                        if loss_unclipped > loss_clipped:
                            d_value[0] = (
                                Scalar[dtype](2.0)
                                * Scalar[dtype](self.value_loss_coef)
                                * (value - return_t)
                            )
                        else:
                            if v_diff > Scalar[dtype](
                                self.clip_epsilon
                            ) or v_diff < -Scalar[dtype](self.clip_epsilon):
                                d_value[0] = Scalar[dtype](0.0)
                            else:
                                d_value[0] = (
                                    Scalar[dtype](2.0)
                                    * Scalar[dtype](self.value_loss_coef)
                                    * (value - return_t)
                                )
                    else:
                        d_value[0] = (
                            Scalar[dtype](2.0)
                            * Scalar[dtype](self.value_loss_coef)
                            * (value - return_t)
                        )

                    var d_value_t = LayoutTensor[
                        dtype, Layout.row_major(1, 1), MutAnyOrigin
                    ](d_value.unsafe_ptr())
                    var critic_grad_input = InlineArray[
                        Scalar[dtype], Self.OBS
                    ](fill=0)
                    var d_in_t = LayoutTensor[
                        dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
                    ](critic_grad_input.unsafe_ptr())
                    var g_critic = self.state.critic.grads_view()
                    self.state.critic.zero_grads()
                    Self.CriticNet.backward[1](
                        d_value_t, d_in_t, p_critic_u, critic_cache_t, g_critic
                    )
                    self.state.critic.optimizer_step()

                    total_loss += (
                        policy_loss
                        + Scalar[dtype](self.value_loss_coef) * value_loss
                        - Scalar[dtype](self.entropy_coef) * entropy
                    )

                batch_start = batch_end

        self.state.buffer_idx = 0
        self.train_step_count += 1
        return Float64(total_loss / Scalar[dtype](self.num_epochs * buffer_len))

    fn select_greedy_action_list(
        self, obs: List[Float64]
    ) -> List[Float64]:
        """Select greedy action using self.state (OnPolicyAgent overload)."""
        var obs_arr = InlineArray[Scalar[dtype], Self.OBS](uninitialized=True)
        for i in range(Self.OBS):
            obs_arr[i] = Scalar[dtype](obs[i])
        # Inline actor forward to avoid potential aliasing with self.state
        var logits_data = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs_arr.unsafe_ptr())
        var logits_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](logits_data.unsafe_ptr())
        var p_actor = self.state.actor.params_view()
        Self.ActorNet.forward[1](obs_t, logits_t, p_actor)
        var logits = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        for i in range(Self.ACTIONS):
            logits[i] = rebind[Scalar[dtype]](logits_t[0, i])
        var probs = softmax_inline[dtype, Self.ACTIONS](logits)
        var action = argmax_probs_inline[dtype, Self.ACTIONS](probs)
        var result = List[Float64]()
        result.append(Float64(action))
        return result^

    fn select_action_state(
        self,
        cpu_state: Self.CPUStateType,
        obs: InlineArray[Scalar[dtype], Self.OBS],
        training: Bool = True,
    ) -> Tuple[Int, Scalar[dtype], Scalar[dtype]]:
        """Select action using cpu_state actor/critic networks."""
        var logits_data = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        var obs_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.OBS), MutAnyOrigin
        ](obs.unsafe_ptr())
        var logits_t = LayoutTensor[
            dtype, Layout.row_major(1, Self.ACTIONS), MutAnyOrigin
        ](logits_data.unsafe_ptr())
        var p_actor = cpu_state.actor.params_view()
        Self.ActorNet.forward[1](obs_t, logits_t, p_actor)

        var logits = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        for i in range(Self.ACTIONS):
            logits[i] = rebind[Scalar[dtype]](logits_t[0, i])

        var probs = softmax_inline[dtype, Self.ACTIONS](logits)

        var value_data = InlineArray[Scalar[dtype], 1](uninitialized=True)
        var value_t = LayoutTensor[dtype, Layout.row_major(1, 1), MutAnyOrigin](
            value_data.unsafe_ptr()
        )
        var p_critic = cpu_state.critic.params_view()
        Self.CriticNet.forward[1](obs_t, value_t, p_critic)
        var value = rebind[Scalar[dtype]](value_t[0, 0])

        var action: Int
        if training:
            action = sample_from_probs_inline[dtype, Self.ACTIONS](probs)
        else:
            action = argmax_probs_inline[dtype, Self.ACTIONS](probs)

        var log_prob = log(probs[action] + Scalar[dtype](1e-8))
        return (action, log_prob, value)

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

        Delegates to the shared on-policy training loop.
        num_episodes is treated as num_updates (rollout-based, not episode-based).

        Args:
            env: The environment to train on.
            num_episodes: Number of rollout updates (num_updates).
            max_steps_per_episode: Ignored (PPO uses rollout_len, not max steps).
            verbose: Whether to print progress.
            print_every: Print progress every N updates if verbose.
            environment_name: Name of environment for metrics labeling.

        Returns:
            TrainingMetrics with one entry per update (reward = policy loss).
        """
        return run_onpolicy_discrete_train(
            self,
            env,
            num_episodes,
            verbose,
            print_every,
            environment_name,
            "Deep PPO",
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
    ) raises -> Float64:
        """Evaluate the agent using greedy policy.

        Args:
            env: The environment to evaluate on.
            num_episodes: Number of evaluation episodes.
            max_steps: Maximum steps per episode.
            verbose: Whether to print per-episode results.
            render: Whether to render the environment (default: False).
            frame_delay_ms: Delay between frames in milliseconds (default: 16).

        Returns:
            Average reward over evaluation episodes.
        """
        var total_reward: Float64 = 0.0
        var quit_requested = False

        if render:
            _ = env.init_renderer()

        for episode in range(num_episodes):
            if quit_requested:
                break

            var obs_list = env.reset_obs_list()
            var obs = InlineArray[Scalar[dtype], Self.OBS](uninitialized=True)
            for i in range(Self.OBS):
                obs[i] = Scalar[dtype](obs_list[i])
            var episode_reward: Float64 = 0.0
            var episode_steps = 0

            for step in range(max_steps):
                # Greedy action
                var action_result = self.select_action(obs, training=False)
                var action = action_result[0]

                # Step environment
                var result = env.step_obs(action)
                var next_obs_list = result[0].copy()
                var next_obs = InlineArray[Scalar[dtype], Self.OBS](
                    uninitialized=True
                )
                for i in range(Self.OBS):
                    next_obs[i] = Scalar[dtype](next_obs_list[i])
                var reward = result[1]
                var done = result[2]

                if render:
                    env.render_frame()
                    env.renderer_delay(frame_delay_ms)
                    if env.check_renderer_quit():
                        quit_requested = True
                        break

                episode_reward += Float64(reward)
                obs = next_obs^
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
    # GPUOnPolicyDiscreteAgent trait conformance
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
        """Forward actor + critic on GPU and sample actions."""
        comptime blocks = (N_ENVS + TPB - 1) // TPB

        var actor_params_t = gpu_state.gpu_actor.params_view()
        var critic_params_t = gpu_state.gpu_critic.params_view()

        var obs_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.OBS), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var logits_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.logits_buf.unsafe_ptr())
        var actions_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS), MutAnyOrigin
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
            logits_t,
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

        comptime sample_wrapper = _sample_actions_kernel[
            dtype, N_ENVS, Self.ACTIONS
        ]
        ctx.enqueue_function[sample_wrapper, sample_wrapper](
            logits_t,
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
        ctx.enqueue_copy(gpu_state.bootstrap_values_host, gpu_state.values_env_buf)
        ctx.enqueue_copy(gpu_state.rollout_rewards_host, gpu_state.rollout_rewards_buf)
        ctx.enqueue_copy(gpu_state.rollout_values_host, gpu_state.rollout_values_buf)
        ctx.enqueue_copy(gpu_state.rollout_dones_host, gpu_state.rollout_dones_buf)
        ctx.synchronize()

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
        """Run PPO multi-epoch minibatch updates on GPU."""
        comptime ROLLOUT_TOTAL = Self.TOTAL_ROLLOUT_SIZE
        comptime MINIBATCH = Self.GPU_MINIBATCH
        comptime MINIBATCH_BLOCKS = (MINIBATCH + TPB - 1) // TPB
        comptime ACTOR_PARAMS = Self.ActorModel.PARAM_SIZE
        comptime CRITIC_PARAMS = Self.CriticModel.PARAM_SIZE
        comptime ACTOR_GRAD_BLOCKS = (ACTOR_PARAMS + TPB - 1) // TPB
        comptime CRITIC_GRAD_BLOCKS = (CRITIC_PARAMS + TPB - 1) // TPB

        # LayoutTensor views over gpu_state rollout buffers
        var actor_params_t = gpu_state.gpu_actor.params_view()
        var actor_grads_t = gpu_state.gpu_actor.grads_view()
        var critic_params_t = gpu_state.gpu_critic.params_view()
        var critic_grads_t = gpu_state.gpu_critic.grads_view()

        var mb_obs_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, Self.OBS), MutAnyOrigin
        ](gpu_state.mb_obs_buf.unsafe_ptr())
        var mb_actions_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
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
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
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

        var actor_logits_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, Self.ACTIONS), MutAnyOrigin
        ](gpu_state.actor_logits_buf.unsafe_ptr())
        var actor_grad_output_t = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, Self.ACTIONS), MutAnyOrigin
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
        var actor_scale_t = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](gpu_state.actor_scale_buf.unsafe_ptr())
        var critic_scale_t = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](gpu_state.critic_scale_buf.unsafe_ptr())

        # Entropy annealing (based on estimated total steps)
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
        comptime gather_wrapper = ppo_gather_minibatch_kernel[
            dtype, MINIBATCH, Self.OBS, ROLLOUT_TOTAL
        ]
        comptime actor_grad_with_kl_wrapper = ppo_actor_grad_with_kl_kernel[
            dtype, MINIBATCH, Self.ACTIONS
        ]
        comptime critic_grad_wrapper = ppo_critic_grad_kernel[dtype, MINIBATCH]
        comptime critic_grad_clipped_wrapper = ppo_critic_grad_clipped_kernel[
            dtype, MINIBATCH
        ]
        comptime normalize_advantages_wrapper = normalize_advantages_kernel[
            dtype, MINIBATCH
        ]
        comptime actor_grad_norm_wrapper = gradient_norm_kernel[
            dtype, ACTOR_PARAMS, ACTOR_GRAD_BLOCKS, TPB
        ]
        comptime critic_grad_norm_wrapper = gradient_norm_kernel[
            dtype, CRITIC_PARAMS, CRITIC_GRAD_BLOCKS, TPB
        ]
        comptime actor_reduce_scale_wrapper = gradient_reduce_and_compute_scale_kernel[
            dtype, ACTOR_GRAD_BLOCKS, TPB
        ]
        comptime actor_apply_scale_wrapper = gradient_apply_scale_kernel[
            dtype, ACTOR_PARAMS
        ]
        comptime critic_reduce_scale_wrapper = gradient_reduce_and_compute_scale_kernel[
            dtype, CRITIC_GRAD_BLOCKS, TPB
        ]
        comptime critic_apply_scale_wrapper = gradient_apply_scale_kernel[
            dtype, CRITIC_PARAMS
        ]

        var kl_early_stop = False
        var num_minibatches = ROLLOUT_TOTAL // MINIBATCH

        for epoch in range(self.num_epochs):
            if kl_early_stop:
                break

            # Fisher-Yates shuffle on CPU
            var indices_list = List[Int]()
            for i in range(ROLLOUT_TOTAL):
                indices_list.append(i)
            for i in range(ROLLOUT_TOTAL - 1, 0, -1):
                var j = Int(random_float64() * Float64(i + 1))
                var temp = indices_list[i]
                indices_list[i] = indices_list[j]
                indices_list[j] = temp

            for mb_idx in range(num_minibatches):
                if kl_early_stop:
                    break
                var start_idx = mb_idx * MINIBATCH

                for i in range(MINIBATCH):
                    gpu_state.mb_indices_host[i] = Int32(
                        indices_list[start_idx + i]
                    )
                ctx.enqueue_copy(gpu_state.mb_indices_buf, gpu_state.mb_indices_host)

                # Gather minibatch from rollout
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
                ctx.synchronize()

                # Per-minibatch advantage normalization
                if self.norm_adv_per_minibatch:
                    ctx.enqueue_copy(
                        gpu_state.mb_advantages_host, gpu_state.mb_advantages_buf
                    )
                    ctx.synchronize()
                    var adv_mean = Scalar[dtype](0.0)
                    for i in range(MINIBATCH):
                        adv_mean += gpu_state.mb_advantages_host[i]
                    adv_mean /= Scalar[dtype](MINIBATCH)
                    var adv_var = Scalar[dtype](0.0)
                    for i in range(MINIBATCH):
                        var diff = gpu_state.mb_advantages_host[i] - adv_mean
                        adv_var += diff * diff
                    var adv_std = sqrt(
                        adv_var / Scalar[dtype](MINIBATCH) + Scalar[dtype](1e-8)
                    )
                    ctx.enqueue_function[
                        normalize_advantages_wrapper,
                        normalize_advantages_wrapper,
                    ](
                        mb_advantages_t,
                        adv_mean,
                        adv_std,
                        MINIBATCH,
                        grid_dim=(MINIBATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.synchronize()

                # ---- Train actor ----
                gpu_state.gpu_actor.zero_grads(ctx)
                Self.ActorModel.forward_gpu[MINIBATCH](
                    ctx,
                    actor_logits_t,
                    mb_obs_t,
                    actor_params_t,
                    actor_cache_t,
                    gpu_state.actor_mb_workspace_buf,
                )
                ctx.synchronize()

                ctx.enqueue_function[
                    actor_grad_with_kl_wrapper, actor_grad_with_kl_wrapper
                ](
                    actor_grad_output_t,
                    kl_divergences_t,
                    actor_logits_t,
                    mb_old_log_probs_t,
                    mb_advantages_t,
                    mb_actions_t,
                    Scalar[dtype](self.clip_epsilon),
                    Scalar[dtype](current_entropy_coef),
                    MINIBATCH,
                    grid_dim=(MINIBATCH_BLOCKS,),
                    block_dim=(TPB,),
                )
                ctx.synchronize()

                if self.target_kl > 0.0:
                    ctx.enqueue_copy(
                        gpu_state.kl_divergences_host, gpu_state.kl_divergences_buf
                    )
                    ctx.synchronize()
                    var kl_sum = Scalar[dtype](0.0)
                    for i in range(MINIBATCH):
                        kl_sum += gpu_state.kl_divergences_host[i]
                    if Float64(kl_sum) / Float64(MINIBATCH) > self.target_kl:
                        kl_early_stop = True
                        break

                Self.ActorModel.backward_gpu[MINIBATCH](
                    ctx,
                    actor_grad_input_t,
                    actor_grad_output_t,
                    actor_params_t,
                    actor_cache_t,
                    actor_grads_t,
                    gpu_state.actor_mb_workspace_buf,
                )

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
                        actor_reduce_scale_wrapper, actor_reduce_scale_wrapper
                    ](
                        actor_scale_t,
                        actor_grad_partial_sums_t,
                        Scalar[dtype](self.max_grad_norm),
                        grid_dim=(1,),
                        block_dim=(TPB,),
                    )
                    ctx.enqueue_function[
                        actor_apply_scale_wrapper, actor_apply_scale_wrapper
                    ](
                        actor_grads_t,
                        actor_scale_t,
                        grid_dim=(ACTOR_GRAD_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.synchronize()

                gpu_state.gpu_actor.optimizer_step(ctx)
                ctx.synchronize()

                # ---- Train critic ----
                gpu_state.gpu_critic.zero_grads(ctx)
                Self.CriticModel.forward_gpu[MINIBATCH](
                    ctx,
                    critic_values_t,
                    mb_obs_t,
                    critic_params_t,
                    critic_cache_t,
                    gpu_state.critic_mb_workspace_buf,
                )
                ctx.synchronize()

                if self.clip_value:
                    ctx.enqueue_function[
                        critic_grad_clipped_wrapper, critic_grad_clipped_wrapper
                    ](
                        critic_grad_output_t,
                        critic_values_t,
                        mb_old_values_t,
                        mb_returns_t,
                        Scalar[dtype](self.value_loss_coef),
                        Scalar[dtype](self.clip_epsilon),
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
                ctx.synchronize()

                Self.CriticModel.backward_gpu[MINIBATCH](
                    ctx,
                    critic_grad_input_t,
                    critic_grad_output_t,
                    critic_params_t,
                    critic_cache_t,
                    critic_grads_t,
                    gpu_state.critic_mb_workspace_buf,
                )

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
                        critic_reduce_scale_wrapper, critic_reduce_scale_wrapper
                    ](
                        critic_scale_t,
                        critic_grad_partial_sums_t,
                        Scalar[dtype](self.max_grad_norm),
                        grid_dim=(1,),
                        block_dim=(TPB,),
                    )
                    ctx.enqueue_function[
                        critic_apply_scale_wrapper, critic_apply_scale_wrapper
                    ](
                        critic_grads_t,
                        critic_scale_t,
                        grid_dim=(CRITIC_GRAD_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.synchronize()

                gpu_state.gpu_critic.optimizer_step(ctx)
                ctx.synchronize()

        # Reset rollout step so next rollout collection starts from position 0
        gpu_state.rollout_step = 0
        self.train_step_count += 1

    # =========================================================================
    # GPU Training
    # =========================================================================

    fn train_gpu[
        EnvType: GPUDiscreteEnv
    ](
        mut self,
        ctx: DeviceContext,
        num_updates: Int,
        verbose: Bool = False,
        print_every: Int = 10,
    ) raises -> TrainingMetrics:
        """Train PPO on GPU with parallel environments.

        Delegates to the shared GPU on-policy training loop.
        num_updates controls the number of rollout+update cycles.

        Args:
            ctx: GPU device context.
            num_updates: Number of rollout+update cycles.
            verbose: Whether to print progress.
            print_every: Print progress every N updates.

        Returns:
            TrainingMetrics with episode rewards and statistics.
        """
        return run_onpolicy_discrete_train_gpu[EnvType](
            self,
            ctx,
            num_updates,
            verbose=verbose,
            print_every=print_every,
        )

    # =========================================================================
    # Checkpoint Save/Load
    # =========================================================================

    fn save_checkpoint(self, filepath: String) raises:
        """Save agent state to a checkpoint file.

        Saves actor and critic networks and hyperparameters.

        Args:
            filepath: Path to save the checkpoint file.
        """
        var actor_param_size = Self.ActorModel.PARAM_SIZE
        var critic_param_size = Self.CriticModel.PARAM_SIZE
        var actor_state_size = (
            actor_param_size * Adam[Self.actor_lr].STATE_PER_PARAM
        )
        var critic_state_size = (
            critic_param_size * Adam[Self.critic_lr].STATE_PER_PARAM
        )

        var content = String("# mojo-rl checkpoint v1\n")
        content += "# type: ppo_agent\n"
        content += "# actor_param_size: " + String(actor_param_size) + "\n"
        content += "# critic_param_size: " + String(critic_param_size) + "\n"

        # Actor params
        content += "actor_params:\n"
        for i in range(actor_param_size):
            content += String(Float64(self.state.actor.params[i])) + "\n"

        content += "actor_optimizer_state:\n"
        for i in range(actor_state_size):
            content += (
                String(Float64(self.state.actor.optimizer_state[i])) + "\n"
            )

        # Critic params
        content += "critic_params:\n"
        for i in range(critic_param_size):
            content += String(Float64(self.state.critic.params[i])) + "\n"

        content += "critic_optimizer_state:\n"
        for i in range(critic_state_size):
            content += (
                String(Float64(self.state.critic.optimizer_state[i])) + "\n"
            )

        # Metadata
        content += "metadata:\n"
        content += "gamma=" + String(self.gamma) + "\n"
        content += "gae_lambda=" + String(self.gae_lambda) + "\n"
        content += "clip_epsilon=" + String(self.clip_epsilon) + "\n"
        content += "actor_lr=" + String(Float64(Self.actor_lr)) + "\n"
        content += "critic_lr=" + String(Float64(Self.critic_lr)) + "\n"
        content += "entropy_coef=" + String(self.entropy_coef) + "\n"
        content += "value_loss_coef=" + String(self.value_loss_coef) + "\n"
        content += "num_epochs=" + String(self.num_epochs) + "\n"
        content += "minibatch_size=" + String(self.minibatch_size) + "\n"
        content += (
            "normalize_advantages=" + String(self.normalize_advantages) + "\n"
        )
        content += "target_kl=" + String(self.target_kl) + "\n"
        content += "max_grad_norm=" + String(self.max_grad_norm) + "\n"
        content += "train_step_count=" + String(self.train_step_count) + "\n"

        save_checkpoint_file(filepath, content)

    fn load_checkpoint(mut self, filepath: String) raises:
        """Load agent state from a checkpoint file.

        Args:
            filepath: Path to the checkpoint file.
        """
        var actor_param_size = Self.ActorModel.PARAM_SIZE
        var critic_param_size = Self.CriticModel.PARAM_SIZE
        var actor_state_size = (
            actor_param_size * Adam[Self.actor_lr].STATE_PER_PARAM
        )
        var critic_state_size = (
            critic_param_size * Adam[Self.critic_lr].STATE_PER_PARAM
        )

        var content = read_checkpoint_file(filepath)
        var lines = split_lines(content)

        # Load actor params
        var actor_params_start = find_section_start(lines, "actor_params:")
        for i in range(actor_param_size):
            self.state.actor.params[i] = Scalar[dtype](
                atof(lines[actor_params_start + i])
            )

        var actor_state_start = find_section_start(
            lines, "actor_optimizer_state:"
        )
        for i in range(actor_state_size):
            self.state.actor.optimizer_state[i] = Scalar[dtype](
                atof(lines[actor_state_start + i])
            )

        # Load critic params
        var critic_params_start = find_section_start(lines, "critic_params:")
        for i in range(critic_param_size):
            self.state.critic.params[i] = Scalar[dtype](
                atof(lines[critic_params_start + i])
            )

        var critic_state_start = find_section_start(
            lines, "critic_optimizer_state:"
        )
        for i in range(critic_state_size):
            self.state.critic.optimizer_state[i] = Scalar[dtype](
                atof(lines[critic_state_start + i])
            )

        # Load metadata
        var metadata_start = find_section_start(lines, "metadata:")
        for i in range(metadata_start, len(lines)):
            var line = lines[i]
            if line.startswith("gamma="):
                self.gamma = atof(String(line[6:]))
            elif line.startswith("gae_lambda="):
                self.gae_lambda = atof(String(line[11:]))
            elif line.startswith("clip_epsilon="):
                self.clip_epsilon = atof(String(line[13:]))
            elif line.startswith("actor_lr="):
                pass  # compile-time parameter; ignore loaded value
            elif line.startswith("critic_lr="):
                pass  # compile-time parameter; ignore loaded value
            elif line.startswith("entropy_coef="):
                self.entropy_coef = atof(String(line[13:]))
            elif line.startswith("value_loss_coef="):
                self.value_loss_coef = atof(String(line[16:]))
            elif line.startswith("num_epochs="):
                self.num_epochs = Int(atol(String(line[11:])))
            elif line.startswith("minibatch_size="):
                self.minibatch_size = Int(atol(String(line[15:])))
            elif line.startswith("normalize_advantages="):
                self.normalize_advantages = String(line[21:]) == "True"
            elif line.startswith("target_kl="):
                self.target_kl = atof(String(line[10:]))
            elif line.startswith("max_grad_norm="):
                self.max_grad_norm = atof(String(line[14:]))
            elif line.startswith("train_step_count="):
                self.train_step_count = Int(atol(String(line[17:])))


# =============================================================================
# GPU Kernel: Sample actions from categorical distribution
# =============================================================================


@always_inline
fn _sample_actions_kernel[
    dtype: DType where dtype.is_floating_point(),
    N_ENVS: Int,
    NUM_ACTIONS: Int,
](
    logits: LayoutTensor[
        dtype, Layout.row_major(N_ENVS, NUM_ACTIONS), MutAnyOrigin
    ],
    actions: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    log_probs: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    seed: Scalar[DType.uint32],
):
    """Sample actions from categorical distribution and compute log probs."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= N_ENVS:
        return

    # Per-thread RNG
    var rng_state = UInt32(seed) ^ (UInt32(i) * 2654435761)
    rng_state = xorshift32(rng_state)

    # Compute softmax probabilities
    var max_logit = logits[i, 0]
    for a in range(1, NUM_ACTIONS):
        var l = logits[i, a]
        if l > max_logit:
            max_logit = l

    var sum_exp = (
        logits[i, 0] - logits[i, 0]
    )  # Initialize to zero with correct type
    for a in range(NUM_ACTIONS):
        var logit_val = logits[i, a] - max_logit
        sum_exp = sum_exp + exp(logit_val)

    # Sample action
    var rand_result = random_uniform[dtype](rng_state)
    var rand_val = rand_result[0]
    rng_state = rand_result[1]

    var cumsum_val = Scalar[dtype](0.0)
    var selected_action: actions.element_type = 0
    for a in range(NUM_ACTIONS):
        var logit_val = logits[i, a] - max_logit
        var prob = exp(logit_val) / sum_exp
        var prob_scalar = Scalar[dtype](prob[0])
        cumsum_val = cumsum_val + prob_scalar
        if rand_val < cumsum_val:
            selected_action = Scalar[dtype](a)
            break

    actions[i] = selected_action

    # Compute log probability
    var logit_sel = logits[i, selected_action] - max_logit
    var selected_prob_simd = exp(logit_sel) / sum_exp
    var selected_prob = Float32(selected_prob_simd[0])
    var eps = Float32(1e-8)
    var log_prob_val = log(selected_prob + eps)
    log_probs[i] = Scalar[dtype](log_prob_val)

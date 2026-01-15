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
from time import perf_counter_ns
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor

from deep_rl.constants import dtype, TILE, TPB
from deep_rl.model import Linear, ReLU, seq
from deep_rl.optimizer import Adam
from deep_rl.initializer import Xavier
from deep_rl.training import Network
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
from core import TrainingMetrics, BoxDiscreteActionEnv, GPUDiscreteEnv
from core.utils.gae import compute_gae_inline
from core.utils.softmax import (
    softmax_inline,
    sample_from_probs_inline,
    argmax_probs_inline,
)
from core.utils.normalization import normalize_inline
from core.utils.shuffle import shuffle_indices_inline


# =============================================================================
# GPU Kernels for PPO Operations
# =============================================================================


@always_inline
fn ppo_store_rollout_kernel[
    dtype: DType,
    N_ENVS: Int,
    OBS_DIM: Int,
](
    # Outputs - rollout buffer storage
    rollout_obs: LayoutTensor[
        dtype, Layout.row_major(N_ENVS, OBS_DIM), MutAnyOrigin
    ],
    rollout_actions: LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ],
    rollout_rewards: LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ],
    rollout_values: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    rollout_log_probs: LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ],
    rollout_dones: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Inputs - current step data
    obs: LayoutTensor[dtype, Layout.row_major(N_ENVS, OBS_DIM), MutAnyOrigin],
    actions: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    rewards: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    values: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    log_probs: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    dones: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
):
    """Store transition data for one timestep (n_envs transitions).

    This kernel stores data at timestep t. The rollout buffer tensors
    passed in should be views at offset t * n_envs.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= N_ENVS:
        return

    # Store observation
    for d in range(OBS_DIM):
        rollout_obs[i, d] = obs[i, d]

    rollout_actions[i] = actions[i]
    rollout_rewards[i] = rewards[i]
    rollout_values[i] = values[i]
    rollout_log_probs[i] = log_probs[i]
    rollout_dones[i] = dones[i]


@always_inline
fn ppo_gather_minibatch_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    OBS_DIM: Int,
    TOTAL_SIZE: Int,
](
    # Outputs - minibatch buffers
    mb_obs: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
    ],
    mb_actions: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    mb_advantages: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    mb_returns: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    mb_old_log_probs: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    # Inputs - rollout buffers and indices
    rollout_obs: LayoutTensor[
        dtype, Layout.row_major(TOTAL_SIZE, OBS_DIM), MutAnyOrigin
    ],
    rollout_actions: LayoutTensor[
        dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin
    ],
    advantages: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
    returns: LayoutTensor[dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin],
    rollout_log_probs: LayoutTensor[
        dtype, Layout.row_major(TOTAL_SIZE), MutAnyOrigin
    ],
    indices: LayoutTensor[
        DType.int32, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    batch_size: Int,
):
    """Gather samples from rollout buffer using shuffled indices."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= batch_size:
        return

    var src_idx = Int(indices[i])

    # Gather observation
    for d in range(OBS_DIM):
        mb_obs[i, d] = rollout_obs[src_idx, d]

    mb_actions[i] = rollout_actions[src_idx]
    mb_advantages[i] = advantages[src_idx]
    mb_returns[i] = returns[src_idx]
    mb_old_log_probs[i] = rollout_log_probs[src_idx]


@always_inline
fn ppo_actor_grad_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    NUM_ACTIONS: Int,
](
    # Outputs
    grad_logits: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
    # Inputs
    logits: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
    old_log_probs: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    advantages: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    actions: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    clip_epsilon: Scalar[dtype],
    entropy_coef: Scalar[dtype],
    batch_size: Int,
):
    """Compute gradient for PPO actor with clipped surrogate objective.

    Gradient is zero if ratio is clipped, otherwise:
    grad = -advantage * ratio * d_log_prob - entropy_coef * d_entropy
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= batch_size:
        return

    var action = Int(actions[b])
    var advantage = advantages[b]

    # Compute softmax probabilities
    var max_logit = logits[b, 0]
    for a in range(1, NUM_ACTIONS):
        if logits[b, a] > max_logit:
            max_logit = logits[b, a]

    var sum_exp = max_logit - max_logit  # Initialize to zero with correct type
    for a in range(NUM_ACTIONS):
        var l = logits[b, a]
        var logit_val = l - max_logit
        sum_exp = sum_exp + exp(logit_val)

    var probs = InlineArray[Scalar[dtype], NUM_ACTIONS](fill=Scalar[dtype](0.0))
    for a in range(NUM_ACTIONS):
        var l = logits[b, a]
        var logit_val = l - max_logit
        var prob_val = exp(logit_val) / sum_exp
        probs[a] = Scalar[dtype](prob_val[0])

    # Compute new log probability
    var log_eps = Float32(1e-8)
    var prob_for_log = Float32(probs[action]) + log_eps
    var new_log_prob = Scalar[dtype](log(prob_for_log))

    # Probability ratio
    var ratio = exp(new_log_prob - old_log_probs[b])

    # Check if clipped
    var is_clipped = (ratio < Scalar[dtype](1.0) - clip_epsilon) or (
        ratio > Scalar[dtype](1.0) + clip_epsilon
    )

    # Compute gradients
    for a in range(NUM_ACTIONS):
        if is_clipped:
            grad_logits[b, a] = Scalar[dtype](0.0)
        else:
            # d_log_prob / d_logits for softmax
            var d_log_prob: Scalar[dtype]
            if a == action:
                d_log_prob = Scalar[dtype](1.0) - probs[a]
            else:
                d_log_prob = -probs[a]

            # Entropy gradient: d(-p * log(p)) / d_logits
            var prob_for_log_ent = Float32(probs[a]) + Float32(1e-8)
            var log_prob_ent = Scalar[dtype](log(prob_for_log_ent))
            var d_entropy = -probs[a] * (Scalar[dtype](1.0) + log_prob_ent)

            # PPO gradient (negative because we maximize)
            grad_logits[b, a] = (
                -advantage * ratio * d_log_prob - entropy_coef * d_entropy
            ) / Scalar[dtype](BATCH_SIZE)


@always_inline
fn ppo_actor_grad_with_kl_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    NUM_ACTIONS: Int,
](
    # Outputs
    grad_logits: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
    kl_divergences: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    # Inputs
    logits: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
    old_log_probs: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
    ],
    advantages: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    actions: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    clip_epsilon: Scalar[dtype],
    entropy_coef: Scalar[dtype],
    batch_size: Int,
):
    """Compute gradient for PPO actor with clipped surrogate objective.

    Also computes approximate KL divergence for early stopping:
    KL ≈ old_log_prob - new_log_prob (approximation)
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= batch_size:
        return

    var action = Int(actions[b])
    var advantage = advantages[b]

    # Compute softmax probabilities
    var max_logit = logits[b, 0]
    for a in range(1, NUM_ACTIONS):
        if logits[b, a] > max_logit:
            max_logit = logits[b, a]

    var sum_exp = max_logit - max_logit  # Initialize to zero with correct type
    for a in range(NUM_ACTIONS):
        var l = logits[b, a]
        var logit_val = l - max_logit
        sum_exp = sum_exp + exp(logit_val)

    var probs = InlineArray[Scalar[dtype], NUM_ACTIONS](fill=Scalar[dtype](0.0))
    for a in range(NUM_ACTIONS):
        var l = logits[b, a]
        var logit_val = l - max_logit
        var prob_val = exp(logit_val) / sum_exp
        probs[a] = Scalar[dtype](prob_val[0])

    # Compute new log probability
    var log_eps = Float32(1e-8)
    var prob_for_log = Float32(probs[action]) + log_eps
    var new_log_prob = Scalar[dtype](log(prob_for_log))

    # Compute approximate KL divergence: old_log_prob - new_log_prob
    var kl = old_log_probs[b] - new_log_prob
    kl_divergences[b] = kl

    # Probability ratio
    var ratio = exp(new_log_prob - old_log_probs[b])

    # Check if clipped
    var is_clipped = (ratio < Scalar[dtype](1.0) - clip_epsilon) or (
        ratio > Scalar[dtype](1.0) + clip_epsilon
    )

    # Compute gradients
    for a in range(NUM_ACTIONS):
        if is_clipped:
            grad_logits[b, a] = Scalar[dtype](0.0)
        else:
            # d_log_prob / d_logits for softmax
            var d_log_prob: Scalar[dtype]
            if a == action:
                d_log_prob = Scalar[dtype](1.0) - probs[a]
            else:
                d_log_prob = -probs[a]

            # Entropy gradient: d(-p * log(p)) / d_logits
            var prob_for_log_ent = Float32(probs[a]) + Float32(1e-8)
            var log_prob_ent = Scalar[dtype](log(prob_for_log_ent))
            var d_entropy = -probs[a] * (Scalar[dtype](1.0) + log_prob_ent)

            # PPO gradient (negative because we maximize)
            grad_logits[b, a] = (
                -advantage * ratio * d_log_prob - entropy_coef * d_entropy
            ) / Scalar[dtype](BATCH_SIZE)


@always_inline
fn gradient_norm_kernel[
    dtype: DType,
    SIZE: Int,
    NUM_BLOCKS: Int,
    BLOCK_SIZE: Int,
](
    # Output
    partial_sums: LayoutTensor[dtype, Layout.row_major(NUM_BLOCKS), MutAnyOrigin],
    # Input
    grads: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
):
    """Compute partial squared sums for gradient norm calculation.

    Each block computes a partial sum of squared gradients.
    Final norm = sqrt(sum of all partial_sums).
    """
    var tid = Int(thread_idx.x)
    var bid = Int(block_idx.x)
    var i = bid * BLOCK_SIZE + tid

    # Shared memory for block reduction
    shared = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Load and square
    if i < SIZE:
        var g = grads[i]
        shared[tid] = g * g
    else:
        shared[tid] = Scalar[dtype](0.0)

    barrier()

    # Block reduction
    var stride = BLOCK_SIZE // 2
    while stride > 0:
        if tid < stride:
            shared[tid] = shared[tid] + shared[tid + stride]
        barrier()
        stride //= 2

    # Write result
    if tid == 0:
        partial_sums[bid] = shared[0]


@always_inline
fn gradient_clip_kernel[
    dtype: DType,
    SIZE: Int,
](
    # In/Out
    grads: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    # Input
    scale: Scalar[dtype],
):
    """Scale gradients by a factor (for gradient clipping).

    Called when grad_norm > max_grad_norm with scale = max_grad_norm / grad_norm.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= SIZE:
        return

    grads[i] = grads[i] * scale


@always_inline
fn ppo_critic_grad_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
](
    # Outputs
    grad_values: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, 1), MutAnyOrigin
    ],
    # Inputs
    values: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, 1), MutAnyOrigin],
    returns: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    value_loss_coef: Scalar[dtype],
    batch_size: Int,
):
    """Compute gradient for critic value loss: MSE(value, return)."""
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH_SIZE:
        return

    # Gradient of MSE loss: 2 * (value - return) / N
    grad_values[b, 0] = (
        Scalar[dtype](2.0)
        * value_loss_coef
        * (values[b, 0] - returns[b])
        / Scalar[dtype](BATCH_SIZE)
    )


# =============================================================================
# GPU Kernels: Store transition data
# =============================================================================


@always_inline
fn _store_pre_step_kernel[
    dtype: DType,
    N_ENVS: Int,
    OBS_DIM: Int,
](
    # Outputs - rollout buffer at timestep t
    r_obs: LayoutTensor[dtype, Layout.row_major(N_ENVS, OBS_DIM), MutAnyOrigin],
    r_actions: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    r_log_probs: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    r_values: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Inputs - current step data
    obs: LayoutTensor[dtype, Layout.row_major(N_ENVS, OBS_DIM), MutAnyOrigin],
    actions: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    log_probs: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    values: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
):
    """Store pre-step data (obs, action, log_prob, value) to rollout buffer."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= N_ENVS:
        return

    for d in range(OBS_DIM):
        r_obs[i, d] = obs[i, d]
    r_actions[i] = actions[i]
    r_log_probs[i] = log_probs[i]
    r_values[i] = values[i]


@always_inline
fn _store_post_step_kernel[
    dtype: DType,
    N_ENVS: Int,
](
    # Outputs - rollout buffer at timestep t
    r_rewards: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    r_dones: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Inputs - current step data
    rewards: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    dones: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
):
    """Store post-step data (rewards, dones) to rollout buffer."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= N_ENVS:
        return

    r_rewards[i] = rewards[i]
    r_dones[i] = dones[i]


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
]:
    """Deep Proximal Policy Optimization Agent using new trait-based architecture.

    Uses clipped surrogate objective for stable policy updates:
    L^CLIP = min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)
    where r(θ) = π_θ(a|s) / π_θ_old(a|s)

    Parameters:
        obs_dim: Dimension of observation space.
        num_actions: Number of discrete actions.
        hidden_dim: Hidden layer size (default: 64).
        rollout_len: Steps per rollout per environment (default: 128 for GPU).
        n_envs: Number of parallel environments for GPU training (default: 1024).
        gpu_minibatch_size: Minibatch size for GPU training (default: 256).

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

    # Cache sizes
    comptime ACTOR_CACHE: Int = Self.OBS + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN
    comptime CRITIC_CACHE: Int = Self.OBS + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN + Self.HIDDEN

    # Network parameter sizes (for GPU buffer allocation)
    # Actor: Linear[obs, hidden] + ReLU + Linear[hidden, hidden] + ReLU + Linear[hidden, actions]
    comptime ACTOR_PARAM_SIZE: Int = (
        Self.OBS * Self.HIDDEN
        + Self.HIDDEN  # Linear 1
        + Self.HIDDEN * Self.HIDDEN
        + Self.HIDDEN  # Linear 2
        + Self.HIDDEN * Self.ACTIONS
        + Self.ACTIONS  # Linear 3
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

    # Advanced hyperparameters (environment-agnostic improvements)
    var target_kl: Float64  # KL threshold for early epoch stopping
    var max_grad_norm: Float64  # Gradient clipping threshold
    var anneal_lr: Bool  # Whether to linearly anneal learning rate
    var anneal_entropy: Bool  # Whether to anneal entropy coefficient
    var target_total_steps: Int  # Target steps for annealing (0 = auto-calculate)

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
        # Advanced hyperparameters
        target_kl: Float64 = 0.015,
        max_grad_norm: Float64 = 0.5,
        anneal_lr: Bool = True,
        anneal_entropy: Bool = False,
        target_total_steps: Int = 0,
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
            target_kl: KL divergence threshold for early epoch stopping (default: 0.015).
            max_grad_norm: Maximum gradient norm for clipping (default: 0.5).
            anneal_lr: Whether to linearly anneal learning rate (default: True).
            anneal_entropy: Whether to anneal entropy coefficient (default: False).
            target_total_steps: Target total steps for annealing, 0=auto (default: 0).
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

        # Store advanced hyperparameters
        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm
        self.anneal_lr = anneal_lr
        self.anneal_entropy = anneal_entropy
        self.target_total_steps = target_total_steps

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
        var logits = InlineArray[Scalar[dtype], Self.ACTIONS](
            uninitialized=True
        )
        self.actor.forward[1](obs, logits)

        # Compute softmax probabilities
        var probs = softmax_inline[dtype, Self.ACTIONS](logits)

        # Forward critic to get value
        var value_out = InlineArray[Scalar[dtype], 1](uninitialized=True)
        self.critic.forward[1](obs, value_out)
        var value = value_out[0]

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
            self.buffer_obs[self.buffer_idx * Self.OBS + i] = obs[i]

        self.buffer_actions[self.buffer_idx] = action
        self.buffer_rewards[self.buffer_idx] = Scalar[dtype](reward)
        self.buffer_log_probs[self.buffer_idx] = log_prob
        self.buffer_values[self.buffer_idx] = value
        self.buffer_dones[self.buffer_idx] = done

        self.buffer_idx += 1

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
            normalize_inline[dtype, Self.ROLLOUT](buffer_len, advantages)

        # =====================================================================
        # Multiple epochs of optimization
        # =====================================================================

        var total_loss = Scalar[dtype](0.0)
        var indices = InlineArray[Int, Self.ROLLOUT](fill=0)

        for epoch in range(self.num_epochs):
            # Shuffle indices for minibatch sampling
            shuffle_indices_inline[Self.ROLLOUT](buffer_len, indices)

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
                    self.critic.forward_with_cache[1](
                        obs, value_out, critic_cache
                    )

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
                    var critic_grad_input = InlineArray[
                        Scalar[dtype], Self.OBS
                    ](fill=0)
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

        return Float64(total_loss / Scalar[dtype](self.num_epochs * buffer_len))

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
                self.store_transition(
                    obs, action, reward, log_prob, value, done
                )

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

    # =========================================================================
    # GPU Training
    # =========================================================================

    fn train_gpu[
        EnvType: GPUDiscreteEnv
    ](
        mut self,
        ctx: DeviceContext,
        num_episodes: Int,
        verbose: Bool = False,
        print_every: Int = 10,
    ) raises -> TrainingMetrics:
        """Train PPO on GPU with parallel environments.

        Args:
            ctx: GPU device context.
            num_episodes: Target number of episodes to complete.
            verbose: Whether to print progress.
            print_every: Print progress every N episodes.

        Returns:
            TrainingMetrics with episode rewards and statistics.
        """
        var metrics = TrainingMetrics(
            algorithm_name="Deep PPO (GPU)",
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
        comptime ROLLOUT_TOTAL = Self.TOTAL_ROLLOUT_SIZE
        comptime ROLLOUT_OBS_SIZE = ROLLOUT_TOTAL * Self.OBS

        comptime MINIBATCH = Self.GPU_MINIBATCH
        comptime MINIBATCH_OBS_SIZE = MINIBATCH * Self.OBS
        comptime MINIBATCH_LOGITS_SIZE = MINIBATCH * Self.ACTIONS
        comptime MINIBATCH_CACHE_ACTOR = MINIBATCH * Self.ACTOR_CACHE
        comptime MINIBATCH_CACHE_CRITIC = MINIBATCH * Self.CRITIC_CACHE

        comptime ENV_BLOCKS = (Self.n_envs + TPB - 1) // TPB
        comptime MINIBATCH_BLOCKS = (MINIBATCH + TPB - 1) // TPB
        comptime ROLLOUT_BLOCKS = (ROLLOUT_TOTAL + TPB - 1) // TPB

        # =====================================================================
        # Network parameter buffers
        # =====================================================================
        var actor_params_buf = ctx.enqueue_create_buffer[dtype](ACTOR_PARAMS)
        var actor_grads_buf = ctx.enqueue_create_buffer[dtype](ACTOR_PARAMS)
        var actor_state_buf = ctx.enqueue_create_buffer[dtype](ACTOR_STATE)

        var critic_params_buf = ctx.enqueue_create_buffer[dtype](CRITIC_PARAMS)
        var critic_grads_buf = ctx.enqueue_create_buffer[dtype](CRITIC_PARAMS)
        var critic_state_buf = ctx.enqueue_create_buffer[dtype](CRITIC_STATE)

        # =====================================================================
        # Environment buffers (n_envs parallel environments)
        # =====================================================================
        var obs_buf = ctx.enqueue_create_buffer[dtype](ENV_OBS_SIZE)
        var rewards_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var dones_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var actions_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var values_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var log_probs_buf = ctx.enqueue_create_buffer[dtype](Self.n_envs)
        var logits_buf = ctx.enqueue_create_buffer[dtype](
            Self.n_envs * Self.ACTIONS
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
        # Rollout buffers (store transitions for one rollout)
        # =====================================================================
        var rollout_obs_buf = ctx.enqueue_create_buffer[dtype](ROLLOUT_OBS_SIZE)
        var rollout_actions_buf = ctx.enqueue_create_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var rollout_rewards_buf = ctx.enqueue_create_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var rollout_values_buf = ctx.enqueue_create_buffer[dtype](ROLLOUT_TOTAL)
        var rollout_log_probs_buf = ctx.enqueue_create_buffer[dtype](
            ROLLOUT_TOTAL
        )
        var rollout_dones_buf = ctx.enqueue_create_buffer[dtype](ROLLOUT_TOTAL)

        # Advantages and returns (computed after rollout)
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
        var mb_actions_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var mb_advantages_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var mb_returns_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var mb_old_log_probs_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var mb_indices_buf = ctx.enqueue_create_buffer[DType.int32](MINIBATCH)
        var mb_indices_host = ctx.enqueue_create_host_buffer[DType.int32](
            MINIBATCH
        )

        # Training workspace
        var actor_logits_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_LOGITS_SIZE
        )
        var actor_cache_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_CACHE_ACTOR
        )
        var actor_grad_output_buf = ctx.enqueue_create_buffer[dtype](
            MINIBATCH_LOGITS_SIZE
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

        # =====================================================================
        # KL divergence and gradient clipping buffers
        # =====================================================================
        var kl_divergences_buf = ctx.enqueue_create_buffer[dtype](MINIBATCH)
        var kl_divergences_host = ctx.enqueue_create_host_buffer[dtype](MINIBATCH)

        # Gradient norm computation buffers
        comptime ACTOR_GRAD_BLOCKS = (ACTOR_PARAMS + TPB - 1) // TPB
        comptime CRITIC_GRAD_BLOCKS = (CRITIC_PARAMS + TPB - 1) // TPB
        var actor_grad_partial_sums_buf = ctx.enqueue_create_buffer[dtype](
            ACTOR_GRAD_BLOCKS
        )
        var critic_grad_partial_sums_buf = ctx.enqueue_create_buffer[dtype](
            CRITIC_GRAD_BLOCKS
        )
        var actor_grad_partial_sums_host = ctx.enqueue_create_host_buffer[dtype](
            ACTOR_GRAD_BLOCKS
        )
        var critic_grad_partial_sums_host = ctx.enqueue_create_host_buffer[dtype](
            CRITIC_GRAD_BLOCKS
        )

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
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](actions_buf.unsafe_ptr())

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
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
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
        var rollout_obs_tensor = LayoutTensor[
            dtype,
            Layout.row_major(ROLLOUT_TOTAL, Self.OBS),
            MutAnyOrigin,
        ](rollout_obs_buf.unsafe_ptr())
        var rollout_actions_tensor = LayoutTensor[
            dtype, Layout.row_major(ROLLOUT_TOTAL), MutAnyOrigin
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
        var mb_indices_tensor = LayoutTensor[
            DType.int32, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_indices_buf.unsafe_ptr())

        var actor_logits_tensor = LayoutTensor[
            dtype,
            Layout.row_major(MINIBATCH, Self.ACTIONS),
            MutAnyOrigin,
        ](actor_logits_buf.unsafe_ptr())
        var actor_grad_output_tensor = LayoutTensor[
            dtype,
            Layout.row_major(MINIBATCH, Self.ACTIONS),
            MutAnyOrigin,
        ](actor_grad_output_buf.unsafe_ptr())
        var actor_old_log_probs_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_old_log_probs_buf.unsafe_ptr())
        var actor_advantages_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_advantages_buf.unsafe_ptr())
        var actor_actions_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_actions_buf.unsafe_ptr())
        var critic_values_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, 1), MutAnyOrigin
        ](critic_values_buf.unsafe_ptr())
        var critic_grad_output_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH, 1), MutAnyOrigin
        ](critic_grad_output_buf.unsafe_ptr())
        var critic_returns_tensor = LayoutTensor[
            dtype, Layout.row_major(MINIBATCH), MutAnyOrigin
        ](mb_returns_buf.unsafe_ptr())

        var logits_tensor = LayoutTensor[
            dtype,
            Layout.row_major(Self.n_envs, Self.ACTIONS),
            MutAnyOrigin,
        ](logits_buf.unsafe_ptr())

        var log_probs_tensor = LayoutTensor[
            dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
        ](log_probs_buf.unsafe_ptr())

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

        # Initialize episode tracking to zero
        ctx.enqueue_memset(episode_rewards_buf, 0)
        ctx.enqueue_memset(episode_steps_buf, 0)

        # =====================================================================
        # Initialize all environments on GPU
        # =====================================================================
        EnvType.reset_kernel_gpu[Self.n_envs, Self.OBS](ctx, obs_buf)
        ctx.synchronize()

        # =====================================================================
        # Training state
        # =====================================================================
        var completed_episodes = 0
        var total_steps = 0
        var rollout_count = 0

        # Annealing: compute target total steps
        # If not set, estimate based on num_episodes * average episode length
        # Use ROLLOUT_TOTAL as rough estimate of steps per rollout batch
        var annealing_target_steps = self.target_total_steps
        if annealing_target_steps == 0:
            # Estimate: num_episodes * 200 steps average (conservative for most envs)
            annealing_target_steps = num_episodes * 200

        # Store initial learning rates for annealing
        var initial_actor_lr = self.actor_lr
        var initial_critic_lr = self.critic_lr
        var initial_entropy_coef = self.entropy_coef

        # Kernel wrappers
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

        # Define wrappers OUTSIDE the loop to avoid recompilation
        comptime store_post_step_wrapper = _store_post_step_kernel[
            dtype, Self.n_envs
        ]

        comptime store_pre_step_wrapper = _store_pre_step_kernel[
            dtype, Self.n_envs, Self.OBS
        ]

        # Phase 3 kernel wrappers - defined ONCE outside the training loop
        comptime gather_wrapper = ppo_gather_minibatch_kernel[
            dtype, MINIBATCH, Self.OBS, ROLLOUT_TOTAL
        ]
        comptime actor_grad_wrapper = ppo_actor_grad_kernel[
            dtype, MINIBATCH, Self.ACTIONS
        ]
        comptime actor_grad_with_kl_wrapper = ppo_actor_grad_with_kl_kernel[
            dtype, MINIBATCH, Self.ACTIONS
        ]
        comptime critic_grad_wrapper = ppo_critic_grad_kernel[dtype, MINIBATCH]

        # Gradient clipping kernel wrappers
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

        comptime sample_actions_wrapper = _sample_actions_kernel[
            dtype, Self.n_envs, Self.ACTIONS
        ]

        # =====================================================================
        # Main Training Loop
        # =====================================================================

        while completed_episodes < num_episodes:
            rollout_count += 1
            var rollout_start = perf_counter_ns()

            # =================================================================
            # Phase 1: Collect rollout (rollout_len steps across n_envs envs)
            # =================================================================
            var phase1_start = perf_counter_ns()

            for t in range(Self.rollout_len):
                # Select actions for all environments
                var rng_seed = UInt32(total_steps * 2654435761 + t * 7919)
                # Forward actor to get logits
                self.actor.model.forward_gpu_no_cache[Self.n_envs](
                    ctx, logits_buf, obs_buf, actor_params_buf
                )

                # Forward critic to get values
                self.critic.model.forward_gpu_no_cache[Self.n_envs](
                    ctx, values_buf, obs_buf, critic_params_buf
                )
                ctx.synchronize()

                # Sample actions and compute log probs on GPU

                ctx.enqueue_function[
                    sample_actions_wrapper, sample_actions_wrapper
                ](
                    logits_tensor,
                    actions_tensor,
                    log_probs_tensor,
                    Scalar[DType.uint32](rng_seed),
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )

                ctx.synchronize()

                # Store pre-step observation to rollout buffer using kernel
                var t_offset = t * Self.n_envs

                # Create views at the correct offset for this timestep
                var rollout_obs_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.n_envs, Self.OBS),
                    MutAnyOrigin,
                ](rollout_obs_buf.unsafe_ptr() + t_offset * Self.OBS)
                var rollout_actions_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](rollout_actions_buf.unsafe_ptr() + t_offset)
                var rollout_log_probs_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](rollout_log_probs_buf.unsafe_ptr() + t_offset)
                var rollout_values_t = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](rollout_values_buf.unsafe_ptr() + t_offset)

                var values_tensor = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](values_buf.unsafe_ptr())
                var log_probs_tensor = LayoutTensor[
                    dtype, Layout.row_major(Self.n_envs), MutAnyOrigin
                ](log_probs_buf.unsafe_ptr())

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
                    values_tensor,
                    grid_dim=(ENV_BLOCKS,),
                    block_dim=(TPB,),
                )

                # Step all environments
                EnvType.step_kernel_gpu[Self.n_envs, Self.OBS](
                    ctx, obs_buf, actions_buf, rewards_buf, dones_buf
                )
                ctx.synchronize()

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

                        if verbose and completed_episodes % print_every == 0:
                            var avg = metrics.mean_reward_last_n(print_every)
                            print(
                                "Episode",
                                completed_episodes,
                                "| Avg reward:",
                                String(avg)[:7],
                                "| Steps:",
                                total_steps,
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
                EnvType.selective_reset_kernel_gpu[Self.n_envs, Self.OBS](
                    ctx,
                    obs_buf,
                    dones_buf,
                    UInt32(total_steps * 1013904223 + t * 2654435761),
                )
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

            self.critic.model.forward_gpu_no_cache[Self.n_envs](
                ctx, values_buf, obs_buf, critic_params_buf
            )

            ctx.enqueue_copy(bootstrap_values_host, values_buf)

            # Copy rollout data to CPU
            ctx.enqueue_copy(rollout_rewards_host, rollout_rewards_buf)
            ctx.enqueue_copy(rollout_values_host, rollout_values_buf)
            ctx.enqueue_copy(rollout_dones_host, rollout_dones_buf)
            ctx.synchronize()

            # Compute GAE for each environment
            for env_idx in range(Self.n_envs):
                var gae = Scalar[dtype](0.0)
                var gae_decay = Scalar[dtype](self.gamma * self.gae_lambda)
                var bootstrap_val = Scalar[dtype](
                    bootstrap_values_host[env_idx]
                )

                # Iterate backwards through timesteps for this environment
                for t in range(Self.rollout_len - 1, -1, -1):
                    var idx = t * Self.n_envs + env_idx
                    var reward = rollout_rewards_host[idx]
                    var value = rollout_values_host[idx]
                    var done = rollout_dones_host[idx]

                    # Get next value
                    var next_val: Scalar[dtype]
                    if t == Self.rollout_len - 1:
                        next_val = bootstrap_val
                    else:
                        var next_idx = (t + 1) * Self.n_envs + env_idx
                        next_val = rollout_values_host[next_idx]

                    # Reset GAE at episode boundary
                    if done > Scalar[dtype](0.5):
                        next_val = Scalar[dtype](0.0)
                        gae = Scalar[dtype](0.0)

                    # TD residual
                    var delta = (
                        reward + Scalar[dtype](self.gamma) * next_val - value
                    )

                    # GAE accumulation
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
                from math import sqrt

                var variance = var_sum / Scalar[dtype](ROLLOUT_TOTAL)
                var std = sqrt(variance + Scalar[dtype](1e-8))
                for i in range(ROLLOUT_TOTAL):
                    advantages_host[i] = (advantages_host[i] - mean) / (
                        std + Scalar[dtype](1e-8)
                    )

            # Copy advantages and returns to GPU
            ctx.enqueue_copy(advantages_buf, advantages_host)
            ctx.enqueue_copy(returns_buf, returns_host)
            ctx.synchronize()

            var phase2_end = perf_counter_ns()

            # =================================================================
            # Phase 3: Train actor and critic with minibatches
            # =================================================================
            var phase3_start = perf_counter_ns()

            # Compute annealing progress (0.0 to 1.0)
            var progress = Float64(total_steps) / Float64(annealing_target_steps)
            if progress > 1.0:
                progress = 1.0

            # Apply learning rate annealing
            var current_actor_lr = initial_actor_lr
            var current_critic_lr = initial_critic_lr
            var current_entropy_coef = initial_entropy_coef
            if self.anneal_lr:
                var lr_multiplier = 1.0 - progress
                current_actor_lr = initial_actor_lr * lr_multiplier
                current_critic_lr = initial_critic_lr * lr_multiplier
                # Update optimizer learning rates
                self.actor.optimizer.lr = current_actor_lr
                self.critic.optimizer.lr = current_critic_lr

            # Apply entropy coefficient annealing
            if self.anneal_entropy:
                current_entropy_coef = initial_entropy_coef * (1.0 - progress)

            # Sub-timers for phase 3
            var shuffle_time_ns: UInt = 0
            var indices_copy_time_ns: UInt = 0
            var gather_time_ns: UInt = 0
            var actor_train_time_ns: UInt = 0
            var critic_train_time_ns: UInt = 0
            var sync_time_ns: UInt = 0

            # KL early stopping flag
            var kl_early_stop = False

            for epoch in range(self.num_epochs):
                # Check if we should early stop due to KL
                if kl_early_stop:
                    break
                # Generate shuffled indices on CPU
                var shuffle_start = perf_counter_ns()
                var indices_list = List[Int]()
                for i in range(ROLLOUT_TOTAL):
                    indices_list.append(i)

                # Fisher-Yates shuffle
                for i in range(ROLLOUT_TOTAL - 1, 0, -1):
                    var j = Int(random_float64() * Float64(i + 1))
                    var temp = indices_list[i]
                    indices_list[i] = indices_list[j]
                    indices_list[j] = temp
                shuffle_time_ns += perf_counter_ns() - shuffle_start

                # Process minibatches
                var num_minibatches = ROLLOUT_TOTAL // MINIBATCH
                for mb_idx in range(num_minibatches):
                    var start_idx = mb_idx * MINIBATCH

                    # Copy minibatch indices to host buffer
                    var indices_copy_start = perf_counter_ns()
                    for i in range(MINIBATCH):
                        mb_indices_host[i] = Int32(indices_list[start_idx + i])

                    # Copy indices to GPU
                    ctx.enqueue_copy(mb_indices_buf, mb_indices_host)
                    indices_copy_time_ns += (
                        perf_counter_ns() - indices_copy_start
                    )

                    # Gather minibatch data (inlined)
                    var gather_start = perf_counter_ns()

                    ctx.enqueue_function[gather_wrapper, gather_wrapper](
                        mb_obs_tensor,
                        mb_actions_tensor,
                        mb_advantages_tensor,
                        mb_returns_tensor,
                        mb_old_log_probs_tensor,
                        rollout_obs_tensor,
                        rollout_actions_tensor,
                        advantages_tensor,
                        returns_tensor,
                        rollout_log_probs_tensor,
                        mb_indices_tensor,
                        MINIBATCH,
                        grid_dim=(MINIBATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.synchronize()
                    gather_time_ns += perf_counter_ns() - gather_start

                    # Train actor (inlined)
                    var actor_start = perf_counter_ns()

                    # Zero actor gradients
                    ctx.enqueue_memset(actor_grads_buf, 0)

                    # Forward pass with cache
                    self.actor.model.forward_gpu[MINIBATCH](
                        ctx,
                        actor_logits_buf,
                        mb_obs_buf,
                        actor_params_buf,
                        actor_cache_buf,
                    )
                    ctx.synchronize()

                    # Compute PPO gradient with KL divergence tracking
                    ctx.enqueue_function[
                        actor_grad_with_kl_wrapper, actor_grad_with_kl_wrapper
                    ](
                        actor_grad_output_tensor,
                        kl_divergences_tensor,
                        actor_logits_tensor,
                        actor_old_log_probs_tensor,
                        actor_advantages_tensor,
                        actor_actions_tensor,
                        Scalar[dtype](self.clip_epsilon),
                        Scalar[dtype](current_entropy_coef),
                        MINIBATCH,
                        grid_dim=(MINIBATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.synchronize()

                    # Check KL divergence for early stopping
                    if self.target_kl > 0.0:
                        ctx.enqueue_copy(kl_divergences_host, kl_divergences_buf)
                        ctx.synchronize()

                        # Compute mean KL
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
                            break  # Break from minibatch loop

                    # Backward pass
                    self.actor.model.backward_gpu[MINIBATCH](
                        ctx,
                        actor_grad_input_buf,
                        actor_grad_output_buf,
                        actor_params_buf,
                        actor_cache_buf,
                        actor_grads_buf,
                    )

                    # Gradient clipping for actor
                    if self.max_grad_norm > 0.0:
                        # Compute gradient norm using block reduction
                        ctx.enqueue_function[
                            actor_grad_norm_wrapper, actor_grad_norm_wrapper
                        ](
                            actor_grad_partial_sums_tensor,
                            actor_grads_tensor,
                            grid_dim=(ACTOR_GRAD_BLOCKS,),
                            block_dim=(TPB,),
                        )
                        ctx.enqueue_copy(
                            actor_grad_partial_sums_host, actor_grad_partial_sums_buf
                        )
                        ctx.synchronize()

                        # Sum partial sums on CPU
                        var actor_grad_sq_sum = Scalar[dtype](0.0)
                        for i in range(ACTOR_GRAD_BLOCKS):
                            actor_grad_sq_sum += actor_grad_partial_sums_host[i]
                        from math import sqrt

                        var actor_grad_norm = Float64(sqrt(actor_grad_sq_sum))

                        # Clip if necessary
                        if actor_grad_norm > self.max_grad_norm:
                            var clip_scale = Scalar[dtype](
                                self.max_grad_norm / actor_grad_norm
                            )
                            ctx.enqueue_function[
                                actor_grad_clip_wrapper, actor_grad_clip_wrapper
                            ](
                                actor_grads_tensor,
                                clip_scale,
                                grid_dim=(ACTOR_GRAD_BLOCKS,),
                                block_dim=(TPB,),
                            )
                            ctx.synchronize()

                    # Update actor parameters
                    self.actor.optimizer.step_gpu[Self.ACTOR_PARAM_SIZE](
                        ctx, actor_params_buf, actor_grads_buf, actor_state_buf
                    )
                    ctx.synchronize()
                    actor_train_time_ns += perf_counter_ns() - actor_start

                    # Train critic (inlined)
                    var critic_start = perf_counter_ns()

                    # Zero critic gradients
                    ctx.enqueue_memset(critic_grads_buf, 0)

                    # Forward pass with cache
                    self.critic.model.forward_gpu[MINIBATCH](
                        ctx,
                        critic_values_buf,
                        mb_obs_buf,
                        critic_params_buf,
                        critic_cache_buf,
                    )
                    ctx.synchronize()

                    # Compute value loss gradient
                    ctx.enqueue_function[
                        critic_grad_wrapper, critic_grad_wrapper
                    ](
                        critic_grad_output_tensor,
                        critic_values_tensor,
                        critic_returns_tensor,
                        Scalar[dtype](self.value_loss_coef),
                        MINIBATCH,
                        grid_dim=(MINIBATCH_BLOCKS,),
                        block_dim=(TPB,),
                    )
                    ctx.synchronize()

                    # Backward pass
                    self.critic.model.backward_gpu[MINIBATCH](
                        ctx,
                        critic_grad_input_buf,
                        critic_grad_output_buf,
                        critic_params_buf,
                        critic_cache_buf,
                        critic_grads_buf,
                    )

                    # Gradient clipping for critic
                    if self.max_grad_norm > 0.0:
                        # Compute gradient norm using block reduction
                        ctx.enqueue_function[
                            critic_grad_norm_wrapper, critic_grad_norm_wrapper
                        ](
                            critic_grad_partial_sums_tensor,
                            critic_grads_tensor,
                            grid_dim=(CRITIC_GRAD_BLOCKS,),
                            block_dim=(TPB,),
                        )
                        ctx.enqueue_copy(
                            critic_grad_partial_sums_host, critic_grad_partial_sums_buf
                        )
                        ctx.synchronize()

                        # Sum partial sums on CPU
                        var critic_grad_sq_sum = Scalar[dtype](0.0)
                        for i in range(CRITIC_GRAD_BLOCKS):
                            critic_grad_sq_sum += critic_grad_partial_sums_host[i]
                        from math import sqrt

                        var critic_grad_norm = Float64(sqrt(critic_grad_sq_sum))

                        # Clip if necessary
                        if critic_grad_norm > self.max_grad_norm:
                            var clip_scale = Scalar[dtype](
                                self.max_grad_norm / critic_grad_norm
                            )
                            ctx.enqueue_function[
                                critic_grad_clip_wrapper, critic_grad_clip_wrapper
                            ](
                                critic_grads_tensor,
                                clip_scale,
                                grid_dim=(CRITIC_GRAD_BLOCKS,),
                                block_dim=(TPB,),
                            )
                            ctx.synchronize()

                    # Update critic parameters
                    self.critic.optimizer.step_gpu[Self.CRITIC_PARAM_SIZE](
                        ctx,
                        critic_params_buf,
                        critic_grads_buf,
                        critic_state_buf,
                    )
                    ctx.synchronize()
                    critic_train_time_ns += perf_counter_ns() - critic_start

            var phase3_end = perf_counter_ns()

            # Print phase 3 sub-timings
            if verbose:
                var shuffle_ms = Float64(shuffle_time_ns) / 1e6
                var indices_ms = Float64(indices_copy_time_ns) / 1e6
                var gather_ms = Float64(gather_time_ns) / 1e6
                var actor_ms = Float64(actor_train_time_ns) / 1e6
                var critic_ms = Float64(critic_train_time_ns) / 1e6
                print(
                    "    P3 breakdown: shuffle:",
                    String(shuffle_ms)[:5],
                    "ms | indices:",
                    String(indices_ms)[:5],
                    "ms | gather:",
                    String(gather_ms)[:5],
                    "ms | actor:",
                    String(actor_ms)[:5],
                    "ms | critic:",
                    String(critic_ms)[:5],
                    "ms",
                )

            # Print timing for this rollout
            if verbose:
                var p1_ms = Float64(phase1_end - phase1_start) / 1e6
                var p2_ms = Float64(phase2_end - phase2_start) / 1e6
                var p3_ms = Float64(phase3_end - phase3_start) / 1e6
                var total_ms = Float64(phase3_end - rollout_start) / 1e6
                print(
                    "  Rollout",
                    rollout_count,
                    "| P1(collect):",
                    String(p1_ms)[:6],
                    "ms | P2(GAE):",
                    String(p2_ms)[:6],
                    "ms | P3(train):",
                    String(p3_ms)[:6],
                    "ms | Total:",
                    String(total_ms)[:6],
                    "ms",
                )

        # =====================================================================
        # Copy final parameters back to CPU
        # =====================================================================
        self.actor.copy_params_from_device(ctx, actor_params_buf)
        self.actor.copy_state_from_device(ctx, actor_state_buf)
        self.critic.copy_params_from_device(ctx, critic_params_buf)
        self.critic.copy_state_from_device(ctx, critic_state_buf)
        ctx.synchronize()

        return metrics^


# =============================================================================
# GPU Kernel: Sample actions from categorical distribution
# =============================================================================


@always_inline
fn _sample_actions_kernel[
    dtype: DType,
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
    var selected_action = 0
    for a in range(NUM_ACTIONS):
        var logit_val = logits[i, a] - max_logit
        var prob = exp(logit_val) / sum_exp
        var prob_scalar = Scalar[dtype](prob[0])
        cumsum_val = cumsum_val + prob_scalar
        if rand_val < cumsum_val:
            selected_action = a
            break

    actions[i] = selected_action

    # Compute log probability
    var logit_sel = logits[i, selected_action] - max_logit
    var selected_prob_simd = exp(logit_sel) / sum_exp
    var selected_prob = Float32(selected_prob_simd[0])
    var eps = Float32(1e-8)
    var log_prob_val = log(selected_prob + eps)
    log_probs[i] = Scalar[dtype](log_prob_val)

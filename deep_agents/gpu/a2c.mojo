"""GPU A2C (Advantage Actor-Critic) with Shared Network.

Fully composable implementation for any GPU environment implementing GPUDiscreteEnv.
All dimensions (NUM_ENVS, Self.HIDDEN_DIM, ROLLOUT_LEN) are compile-time parameters.

Features:
- TrainingMetrics integration for result storage
- evaluate() method for testing trained policy on CPU environments
- Instance-based design matching CPU agents

Run with:
    pixi run -e apple mojo run examples/test_a2c_gpu.mojo
"""

from time import perf_counter_ns
from math import exp, log, sqrt, cos, sin
from random import random_float64, seed

from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from deep_rl.gpu import (
    xorshift32,
    random_uniform,
    random_range,
    reduce_kernel,
    adam_kernel,
    compute_gae_kernel,
    track_episodes_kernel,
    relu_inline,
    softmax2_inline,
    sample_from_probs2,
    relu_grad_inline,
)
from core import GPUDiscreteEnv, TrainingMetrics, BoxDiscreteActionEnv


# =============================================================================
# Constants (module-level)
# =============================================================================

comptime dtype = DType.float32
comptime TPB: Int = 256  # Threads per block for GPU kernels


# =============================================================================
# Reset All Kernel - uses EnvType.reset_inline for composability
# =============================================================================


fn reset_all_kernel[
    EnvType: GPUDiscreteEnv,
    NUM_ENVS: Int,
](
    states: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, EnvType.STATE_SIZE), MutAnyOrigin
    ],
    rng_states: LayoutTensor[
        DType.uint32, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin
    ],
):
    """Reset all environments using EnvType.reset_inline for composability."""
    var env_idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env_idx >= NUM_ENVS:
        return

    # Get RNG state
    var rng = rebind[Scalar[DType.uint32]](rng_states[env_idx, 0])

    # Load state into InlineArray
    var state = InlineArray[Scalar[dtype], EnvType.STATE_SIZE](
        fill=Scalar[dtype](0)
    )

    # Reset using environment's reset_inline
    EnvType.reset_inline[EnvType.STATE_SIZE](state, rng)

    # Write state back
    for i in range(EnvType.STATE_SIZE):
        states[env_idx, i] = state[i]
    rng_states[env_idx, 0] = rng


# =============================================================================
# Fused Rollout Collection Kernel - combines forward, step, store, reset
# Parameterized on environment type for composability
# =============================================================================


fn fused_rollout_kernel[
    EnvType: GPUDiscreteEnv,
    NUM_ENVS: Int,
    HIDDEN_DIM: Int,
    ROLLOUT_LEN: Int,
](
    # Environment state (no env instance needed - methods are static)
    states: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, EnvType.STATE_SIZE), MutAnyOrigin
    ],
    rng_states: LayoutTensor[
        DType.uint32, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin
    ],
    # Network weights
    W1: LayoutTensor[
        dtype,
        Layout.row_major(EnvType.OBS_DIM, HIDDEN_DIM),
        ImmutAnyOrigin,
    ],
    b1: LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin],
    W_actor: LayoutTensor[
        dtype,
        Layout.row_major(HIDDEN_DIM, EnvType.NUM_ACTIONS),
        ImmutAnyOrigin,
    ],
    b_actor: LayoutTensor[
        dtype, Layout.row_major(EnvType.NUM_ACTIONS), ImmutAnyOrigin
    ],
    W_critic: LayoutTensor[
        dtype, Layout.row_major(HIDDEN_DIM, 1), ImmutAnyOrigin
    ],
    b_critic: LayoutTensor[dtype, Layout.row_major(1), ImmutAnyOrigin],
    # Rollout storage
    rollout_obs: LayoutTensor[
        dtype,
        Layout.row_major(NUM_ENVS, ROLLOUT_LEN, EnvType.OBS_DIM),
        MutAnyOrigin,
    ],
    rollout_actions: LayoutTensor[
        DType.int32, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin
    ],
    rollout_rewards: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin
    ],
    rollout_dones: LayoutTensor[
        DType.int32, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin
    ],
    rollout_log_probs: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin
    ],
    rollout_values: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin
    ],
):
    """Fused kernel: forward pass + env step + store + reset for all ROLLOUT_LEN steps.

    Each thread processes one environment for the entire rollout.
    Reduces kernel launches from 128 (4 kernels × 32 steps) to 1.
    """
    var env_idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env_idx >= NUM_ENVS:
        return

    # Get RNG state for this env
    var rng = rebind[Scalar[DType.uint32]](rng_states[env_idx, 0])

    # Load current state into InlineArray (composable with any STATE_SIZE)
    var state = InlineArray[Scalar[dtype], EnvType.STATE_SIZE](
        fill=Scalar[dtype](0)
    )
    for i in range(EnvType.STATE_SIZE):
        state[i] = rebind[Scalar[dtype]](states[env_idx, i])

    # Process all steps
    for step in range(ROLLOUT_LEN):
        # === 1. Store current observation ===
        for i in range(EnvType.OBS_DIM):
            rollout_obs[env_idx, step, i] = state[i]

        # === 2. Forward pass: obs -> hidden -> action, value ===
        # Hidden layer: h = ReLU(obs @ W1 + b1)
        var h = InlineArray[Scalar[dtype], HIDDEN_DIM](fill=Scalar[dtype](0))
        for j in range(HIDDEN_DIM):
            var sum_val = rebind[Scalar[dtype]](b1[j])
            for i in range(EnvType.OBS_DIM):
                sum_val += state[i] * rebind[Scalar[dtype]](W1[i, j])
            h[j] = relu_inline(sum_val)

        # Actor: logits
        var logit0 = rebind[Scalar[dtype]](b_actor[0])
        var logit1 = rebind[Scalar[dtype]](b_actor[1])
        for k in range(HIDDEN_DIM):
            logit0 += h[k] * rebind[Scalar[dtype]](W_actor[k, 0])
            logit1 += h[k] * rebind[Scalar[dtype]](W_actor[k, 1])

        # Critic: value
        var value: Scalar[dtype] = rebind[Scalar[dtype]](b_critic[0])
        for k in range(HIDDEN_DIM):
            value += h[k] * rebind[Scalar[dtype]](W_critic[k, 0])

        # Softmax and sample action
        var probs = softmax2_inline(logit0, logit1)
        var u_result = random_uniform[dtype](rng)
        rng = u_result[1]
        var action = sample_from_probs2(probs[0], u_result[0])
        var log_prob = log(probs[action] + Scalar[dtype](1e-8))

        # Store action, log_prob, value
        rollout_actions[env_idx, step] = Int32(action)
        rollout_log_probs[env_idx, step] = log_prob
        rollout_values[env_idx, step] = value

        # === 3. Environment step (uses EnvType static methods for composability) ===
        var step_result = EnvType.step_inline[EnvType.STATE_SIZE](state, action)
        var reward = step_result[0]
        var done = step_result[1]

        # Store reward and done
        rollout_rewards[env_idx, step] = reward
        rollout_dones[env_idx, step] = 1 if done else 0

        # === 4. Reset if done (uses EnvType static method for composability) ===
        if done:
            EnvType.reset_inline[EnvType.STATE_SIZE](state, rng)

    # Write final state back
    for i in range(EnvType.STATE_SIZE):
        states[env_idx, i] = state[i]
    rng_states[env_idx, 0] = rng


# =============================================================================
# Get Values Kernel
# =============================================================================


fn get_values_kernel[
    EnvType: GPUDiscreteEnv,
    NUM_ENVS: Int,
    HIDDEN_DIM: Int,
](
    obs: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, EnvType.OBS_DIM), ImmutAnyOrigin
    ],
    W1: LayoutTensor[
        dtype,
        Layout.row_major(EnvType.OBS_DIM, HIDDEN_DIM),
        ImmutAnyOrigin,
    ],
    b1: LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin],
    W_critic: LayoutTensor[
        dtype, Layout.row_major(HIDDEN_DIM, 1), ImmutAnyOrigin
    ],
    b_critic: LayoutTensor[dtype, Layout.row_major(1), ImmutAnyOrigin],
    values: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin],
):
    """Get value estimates."""
    var env_idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env_idx >= NUM_ENVS:
        return

    var o = InlineArray[Scalar[dtype], EnvType.OBS_DIM](fill=Scalar[dtype](0))
    for i in range(EnvType.OBS_DIM):
        o[i] = rebind[Scalar[dtype]](obs[env_idx, i])

    var h = InlineArray[Scalar[dtype], HIDDEN_DIM](fill=Scalar[dtype](0))
    for j in range(HIDDEN_DIM):
        var sum_val = rebind[Scalar[dtype]](b1[j])
        for i in range(EnvType.OBS_DIM):
            sum_val += o[i] * rebind[Scalar[dtype]](W1[i, j])
        h[j] = relu_inline(sum_val)

    var value: Scalar[dtype] = rebind[Scalar[dtype]](b_critic[0])
    for k in range(HIDDEN_DIM):
        value += h[k] * rebind[Scalar[dtype]](W_critic[k, 0])

    values[env_idx, 0] = value


# =============================================================================
# Policy Gradient Kernel
# =============================================================================


fn policy_gradient_kernel[
    EnvType: GPUDiscreteEnv,
    NUM_ENVS: Int,
    HIDDEN_DIM: Int,
    ROLLOUT_LEN: Int,
](
    rollout_obs: LayoutTensor[
        dtype,
        Layout.row_major(NUM_ENVS, ROLLOUT_LEN, EnvType.OBS_DIM),
        ImmutAnyOrigin,
    ],
    rollout_actions: LayoutTensor[
        DType.int32, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), ImmutAnyOrigin
    ],
    rollout_advantages: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), ImmutAnyOrigin
    ],
    rollout_returns: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), ImmutAnyOrigin
    ],
    W1: LayoutTensor[
        dtype,
        Layout.row_major(EnvType.OBS_DIM, HIDDEN_DIM),
        ImmutAnyOrigin,
    ],
    b1: LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin],
    W_actor: LayoutTensor[
        dtype,
        Layout.row_major(HIDDEN_DIM, EnvType.NUM_ACTIONS),
        ImmutAnyOrigin,
    ],
    b_actor: LayoutTensor[
        dtype, Layout.row_major(EnvType.NUM_ACTIONS), ImmutAnyOrigin
    ],
    W_critic: LayoutTensor[
        dtype, Layout.row_major(HIDDEN_DIM, 1), ImmutAnyOrigin
    ],
    b_critic: LayoutTensor[dtype, Layout.row_major(1), ImmutAnyOrigin],
    grad_W1: LayoutTensor[
        dtype,
        Layout.row_major(NUM_ENVS, EnvType.OBS_DIM * HIDDEN_DIM),
        MutAnyOrigin,
    ],
    grad_b1: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin
    ],
    grad_W_actor: LayoutTensor[
        dtype,
        Layout.row_major(NUM_ENVS, HIDDEN_DIM * EnvType.NUM_ACTIONS),
        MutAnyOrigin,
    ],
    grad_b_actor: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, EnvType.NUM_ACTIONS), MutAnyOrigin
    ],
    grad_W_critic: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin
    ],
    grad_b_critic: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin
    ],
    entropy_coef: Scalar[dtype],
    value_coef: Scalar[dtype],
):
    """Compute policy gradients."""
    var env_idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env_idx >= NUM_ENVS:
        return

    # Local gradient accumulators
    var local_grad_W1 = InlineArray[
        Scalar[dtype], EnvType.OBS_DIM * HIDDEN_DIM
    ](fill=Scalar[dtype](0))
    var local_grad_b1 = InlineArray[Scalar[dtype], HIDDEN_DIM](
        fill=Scalar[dtype](0)
    )
    var local_grad_W_actor = InlineArray[
        Scalar[dtype], HIDDEN_DIM * EnvType.NUM_ACTIONS
    ](fill=Scalar[dtype](0))
    var local_grad_b_actor = InlineArray[Scalar[dtype], EnvType.NUM_ACTIONS](
        fill=Scalar[dtype](0)
    )
    var local_grad_W_critic = InlineArray[Scalar[dtype], HIDDEN_DIM](
        fill=Scalar[dtype](0)
    )
    var local_grad_b_critic: Scalar[dtype] = 0

    for t in range(ROLLOUT_LEN):
        var advantage = rebind[Scalar[dtype]](rollout_advantages[env_idx, t])
        var ret = rebind[Scalar[dtype]](rollout_returns[env_idx, t])
        var action = Int(rollout_actions[env_idx, t])

        # Load observation
        var o = InlineArray[Scalar[dtype], EnvType.OBS_DIM](
            fill=Scalar[dtype](0)
        )
        for i in range(EnvType.OBS_DIM):
            o[i] = rebind[Scalar[dtype]](rollout_obs[env_idx, t, i])

        # Forward: hidden (store pre-activation for backprop)
        var h = InlineArray[Scalar[dtype], HIDDEN_DIM](fill=Scalar[dtype](0))
        var h_pre = InlineArray[Scalar[dtype], HIDDEN_DIM](
            fill=Scalar[dtype](0)
        )
        for j in range(HIDDEN_DIM):
            var sum_val = rebind[Scalar[dtype]](b1[j])
            for i in range(EnvType.OBS_DIM):
                sum_val += o[i] * rebind[Scalar[dtype]](W1[i, j])
            h_pre[j] = sum_val
            h[j] = relu_inline(sum_val)

        # Forward: logits
        var logits = InlineArray[Scalar[dtype], EnvType.NUM_ACTIONS](
            fill=Scalar[dtype](0)
        )
        for j in range(EnvType.NUM_ACTIONS):
            var sum_val = rebind[Scalar[dtype]](b_actor[j])
            for k in range(HIDDEN_DIM):
                sum_val += h[k] * rebind[Scalar[dtype]](W_actor[k, j])
            logits[j] = sum_val

        # Forward: value
        var value: Scalar[dtype] = rebind[Scalar[dtype]](b_critic[0])
        for k in range(HIDDEN_DIM):
            value += h[k] * rebind[Scalar[dtype]](W_critic[k, 0])

        # Softmax and log probabilities for entropy
        var probs = softmax2_inline(logits[0], logits[1])
        var prob0 = probs[0]
        var prob1 = probs[1]
        var eps = Scalar[dtype](1e-8)
        var log_prob0 = log(prob0 + eps)
        var log_prob1 = log(prob1 + eps)
        var entropy = -(prob0 * log_prob0 + prob1 * log_prob1)

        # Policy gradient with entropy regularization
        # We minimize: -advantage * log(π(a|s)) - entropy_coef * H(π)
        # Gradient w.r.t. logit_j = (prob_j - one_hot_j) * advantage + entropy_coef * prob_j * (H + log(prob_j))
        var d_logit0 = (
            prob0 - (Scalar[dtype](1) if action == 0 else Scalar[dtype](0))
        ) * advantage + entropy_coef * prob0 * (entropy + log_prob0)
        var d_logit1 = (
            prob1 - (Scalar[dtype](1) if action == 1 else Scalar[dtype](0))
        ) * advantage + entropy_coef * prob1 * (entropy + log_prob1)

        # Value gradient
        var d_value = value_coef * (value - ret)

        # Backprop actor
        for k in range(HIDDEN_DIM):
            local_grad_W_actor[k * EnvType.NUM_ACTIONS + 0] += h[k] * d_logit0
            local_grad_W_actor[k * EnvType.NUM_ACTIONS + 1] += h[k] * d_logit1
        local_grad_b_actor[0] += d_logit0
        local_grad_b_actor[1] += d_logit1

        # Backprop critic
        for k in range(HIDDEN_DIM):
            local_grad_W_critic[k] += h[k] * d_value
        local_grad_b_critic += d_value

        # Backprop through shared hidden
        var dh = InlineArray[Scalar[dtype], HIDDEN_DIM](fill=Scalar[dtype](0))
        for k in range(HIDDEN_DIM):
            dh[k] = d_logit0 * rebind[Scalar[dtype]](W_actor[k, 0])
            dh[k] += d_logit1 * rebind[Scalar[dtype]](W_actor[k, 1])
            dh[k] += d_value * rebind[Scalar[dtype]](W_critic[k, 0])

        for k in range(HIDDEN_DIM):
            var dh_pre = dh[k] * relu_grad_inline(h_pre[k])
            for i in range(EnvType.OBS_DIM):
                local_grad_W1[i * HIDDEN_DIM + k] += o[i] * dh_pre
            local_grad_b1[k] += dh_pre

    # Write gradients
    for i in range(EnvType.OBS_DIM * HIDDEN_DIM):
        grad_W1[env_idx, i] = local_grad_W1[i]
    for k in range(HIDDEN_DIM):
        grad_b1[env_idx, k] = local_grad_b1[k]
    for i in range(HIDDEN_DIM * EnvType.NUM_ACTIONS):
        grad_W_actor[env_idx, i] = local_grad_W_actor[i]
    for j in range(EnvType.NUM_ACTIONS):
        grad_b_actor[env_idx, j] = local_grad_b_actor[j]
    for k in range(HIDDEN_DIM):
        grad_W_critic[env_idx, k] = local_grad_W_critic[k]
    grad_b_critic[env_idx, 0] = local_grad_b_critic


# =============================================================================
# Training
# =============================================================================


struct A2CAgent[
    HIDDEN_DIM: Int = 64,
]:
    """GPU A2C Agent with shared actor-critic network.

    This agent stores trained weights and provides:
    - train(): Train on GPU and return TrainingMetrics
    - evaluate(): Evaluate on CPU with BoxDiscreteActionEnv environments

    Parameters:
        HIDDEN_DIM: Hidden layer size (default: 64).

    Example:
        var agent = A2CAgent[HIDDEN_DIM=64]()
        var metrics = agent.train[CartPoleEnv](num_updates=100)
        var eval_reward = agent.evaluate(env, num_episodes=10)
    """

    # Stored network weights (on host for persistence)
    var W1: List[Float32]
    var b1: List[Float32]
    var W_actor: List[Float32]
    var b_actor: List[Float32]
    var W_critic: List[Float32]
    var b_critic: List[Float32]

    # Dimensions (set during training)
    var obs_dim: Int
    var num_actions: Int

    # Training state
    var trained: Bool
    var total_episodes: Int
    var total_steps: Int

    fn __init__(out self):
        """Initialize A2C agent with empty weights (initialized at training)."""
        self.W1 = List[Float32]()
        self.b1 = List[Float32]()
        self.W_actor = List[Float32]()
        self.b_actor = List[Float32]()
        self.W_critic = List[Float32]()
        self.b_critic = List[Float32]()

        self.obs_dim = 0
        self.num_actions = 0
        self.trained = False
        self.total_episodes = 0
        self.total_steps = 0

    fn select_action(self, obs: List[Float64]) -> Int:
        """Select action using the trained policy (deterministic argmax).

        Args:
            obs: Observation as list of Float64.

        Returns:
            Selected action index.
        """
        # Forward pass: hidden layer
        var h = List[Float32](capacity=Self.HIDDEN_DIM)
        for j in range(Self.HIDDEN_DIM):
            var sum_val = self.b1[j]
            for i in range(self.obs_dim):
                sum_val += Float32(obs[i]) * self.W1[i * Self.HIDDEN_DIM + j]
            # ReLU
            h.append(sum_val if sum_val > Float32(0) else Float32(0))

        # Actor output: logits
        var logits = List[Float32](capacity=self.num_actions)
        for j in range(self.num_actions):
            var sum_val = self.b_actor[j]
            for k in range(Self.HIDDEN_DIM):
                sum_val += h[k] * self.W_actor[k * self.num_actions + j]
            logits.append(sum_val)

        # Argmax for deterministic action
        var best_action = 0
        var best_logit = logits[0]
        for j in range(1, self.num_actions):
            if logits[j] > best_logit:
                best_logit = logits[j]
                best_action = j

        return best_action

    fn train[
        EnvType: GPUDiscreteEnv,
        NUM_ENVS: Int = 1024,
        ROLLOUT_LEN: Int = 128,
    ](
        mut self,
        num_updates: Int,
        lr: Float32 = 0.0003,
        gamma: Float32 = 0.99,
        gae_lambda: Float32 = 0.95,
        entropy_coef: Float32 = 0.01,
        value_coef: Float32 = 0.5,
        beta1: Float32 = 0.9,
        beta2: Float32 = 0.999,
        adam_eps: Float32 = 1e-8,
        verbose: Bool = True,
        environment_name: String = "GPUEnv",
    ) raises -> TrainingMetrics:
        """Train A2C with composable GPU environment using Adam optimizer.

        EnvType must implement GPUDiscreteEnv trait with step_inline/reset_inline.
        Creates GPU DeviceContext internally.

        Args:
            num_updates: Number of policy updates.
            lr: Learning rate (default 0.0003 for Adam).
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.
            entropy_coef: Entropy bonus coefficient.
            value_coef: Value loss coefficient.
            beta1: Adam first moment decay (default 0.9).
            beta2: Adam second moment decay (default 0.999).
            adam_eps: Adam epsilon for numerical stability.
            verbose: Print progress.
            environment_name: Name for metrics.

        Returns:
            TrainingMetrics with episode history.
        """

        # Initialize metrics
        var metrics = TrainingMetrics(
            algorithm_name="GPU A2C",
            environment_name=environment_name,
        )

        with DeviceContext() as ctx:
            # Parameter sizes derived from EnvType and training hyperparams
            comptime W1_SIZE = EnvType.OBS_DIM * Self.HIDDEN_DIM
            comptime B1_SIZE = Self.HIDDEN_DIM
            comptime W_ACTOR_SIZE = Self.HIDDEN_DIM * EnvType.NUM_ACTIONS
            comptime B_ACTOR_SIZE = EnvType.NUM_ACTIONS
            comptime W_CRITIC_SIZE = Self.HIDDEN_DIM
            comptime B_CRITIC_SIZE = 1

            # Allocate buffers using EnvType constants for composability
            var states_buf = ctx.enqueue_create_buffer[dtype](
                NUM_ENVS * EnvType.STATE_SIZE
            )
            var actions_buf = ctx.enqueue_create_buffer[DType.int32](NUM_ENVS)
            var rewards_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS)
            var dones_buf = ctx.enqueue_create_buffer[DType.int32](NUM_ENVS)
            var rng_buf = ctx.enqueue_create_buffer[DType.uint32](NUM_ENVS)

            var W1_buf = ctx.enqueue_create_buffer[dtype](
                EnvType.OBS_DIM * Self.HIDDEN_DIM
            )
            var b1_buf = ctx.enqueue_create_buffer[dtype](Self.HIDDEN_DIM)
            var W_actor_buf = ctx.enqueue_create_buffer[dtype](
                Self.HIDDEN_DIM * EnvType.NUM_ACTIONS
            )
            var b_actor_buf = ctx.enqueue_create_buffer[dtype](
                EnvType.NUM_ACTIONS
            )
            var W_critic_buf = ctx.enqueue_create_buffer[dtype](Self.HIDDEN_DIM)
            var b_critic_buf = ctx.enqueue_create_buffer[dtype](1)

            var log_probs_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS)
            var values_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS)
            var bootstrap_values_buf = ctx.enqueue_create_buffer[dtype](
                NUM_ENVS
            )

            var rollout_obs_buf = ctx.enqueue_create_buffer[dtype](
                NUM_ENVS * ROLLOUT_LEN * EnvType.OBS_DIM
            )
            var rollout_actions_buf = ctx.enqueue_create_buffer[DType.int32](
                NUM_ENVS * ROLLOUT_LEN
            )
            var rollout_rewards_buf = ctx.enqueue_create_buffer[dtype](
                NUM_ENVS * ROLLOUT_LEN
            )
            var rollout_dones_buf = ctx.enqueue_create_buffer[DType.int32](
                NUM_ENVS * ROLLOUT_LEN
            )
            var rollout_log_probs_buf = ctx.enqueue_create_buffer[dtype](
                NUM_ENVS * ROLLOUT_LEN
            )
            var rollout_values_buf = ctx.enqueue_create_buffer[dtype](
                NUM_ENVS * ROLLOUT_LEN
            )
            var rollout_advantages_buf = ctx.enqueue_create_buffer[dtype](
                NUM_ENVS * ROLLOUT_LEN
            )
            var rollout_returns_buf = ctx.enqueue_create_buffer[dtype](
                NUM_ENVS * ROLLOUT_LEN
            )

            # Episode tracking buffers
            var episode_rewards_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS)
            var completed_episodes_buf = ctx.enqueue_create_buffer[DType.int32](
                NUM_ENVS
            )
            var completed_rewards_buf = ctx.enqueue_create_buffer[dtype](
                NUM_ENVS * ROLLOUT_LEN
            )

            var grad_W1_buf = ctx.enqueue_create_buffer[dtype](
                NUM_ENVS * EnvType.OBS_DIM * Self.HIDDEN_DIM
            )
            var grad_b1_buf = ctx.enqueue_create_buffer[dtype](
                NUM_ENVS * Self.HIDDEN_DIM
            )
            var grad_W_actor_buf = ctx.enqueue_create_buffer[dtype](
                NUM_ENVS * Self.HIDDEN_DIM * EnvType.NUM_ACTIONS
            )
            var grad_b_actor_buf = ctx.enqueue_create_buffer[dtype](
                NUM_ENVS * EnvType.NUM_ACTIONS
            )
            var grad_W_critic_buf = ctx.enqueue_create_buffer[dtype](
                NUM_ENVS * Self.HIDDEN_DIM
            )
            var grad_b_critic_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS)

            var reduced_W1_buf = ctx.enqueue_create_buffer[dtype](W1_SIZE)
            var reduced_b1_buf = ctx.enqueue_create_buffer[dtype](B1_SIZE)
            var reduced_W_actor_buf = ctx.enqueue_create_buffer[dtype](
                W_ACTOR_SIZE
            )
            var reduced_b_actor_buf = ctx.enqueue_create_buffer[dtype](
                B_ACTOR_SIZE
            )
            var reduced_W_critic_buf = ctx.enqueue_create_buffer[dtype](
                W_CRITIC_SIZE
            )
            var reduced_b_critic_buf = ctx.enqueue_create_buffer[dtype](
                B_CRITIC_SIZE
            )

            # Adam moment buffers (m = first moment, v = second moment)
            var m_W1_buf = ctx.enqueue_create_buffer[dtype](W1_SIZE)
            var v_W1_buf = ctx.enqueue_create_buffer[dtype](W1_SIZE)
            var m_b1_buf = ctx.enqueue_create_buffer[dtype](B1_SIZE)
            var v_b1_buf = ctx.enqueue_create_buffer[dtype](B1_SIZE)
            var m_W_actor_buf = ctx.enqueue_create_buffer[dtype](W_ACTOR_SIZE)
            var v_W_actor_buf = ctx.enqueue_create_buffer[dtype](W_ACTOR_SIZE)
            var m_b_actor_buf = ctx.enqueue_create_buffer[dtype](B_ACTOR_SIZE)
            var v_b_actor_buf = ctx.enqueue_create_buffer[dtype](B_ACTOR_SIZE)
            var m_W_critic_buf = ctx.enqueue_create_buffer[dtype](W_CRITIC_SIZE)
            var v_W_critic_buf = ctx.enqueue_create_buffer[dtype](W_CRITIC_SIZE)
            var m_b_critic_buf = ctx.enqueue_create_buffer[dtype](B_CRITIC_SIZE)
            var v_b_critic_buf = ctx.enqueue_create_buffer[dtype](B_CRITIC_SIZE)

            # Initialize Adam moments to zero
            m_W1_buf.enqueue_fill(0)
            v_W1_buf.enqueue_fill(0)
            m_b1_buf.enqueue_fill(0)
            v_b1_buf.enqueue_fill(0)
            m_W_actor_buf.enqueue_fill(0)
            v_W_actor_buf.enqueue_fill(0)
            m_b_actor_buf.enqueue_fill(0)
            v_b_actor_buf.enqueue_fill(0)
            m_W_critic_buf.enqueue_fill(0)
            v_W_critic_buf.enqueue_fill(0)
            m_b_critic_buf.enqueue_fill(0)
            v_b_critic_buf.enqueue_fill(0)

            # Initialize RNG
            with rng_buf.map_to_host() as host:
                for i in range(NUM_ENVS):
                    host[i] = UInt32(i + 12345)

            # Store dimensions for later use (select_action, print_info)
            self.obs_dim = EnvType.OBS_DIM
            self.num_actions = EnvType.NUM_ACTIONS

            # Xavier initialization of weights
            self.W1 = List[Float32](capacity=W1_SIZE)
            self.b1 = List[Float32](capacity=B1_SIZE)
            self.W_actor = List[Float32](capacity=W_ACTOR_SIZE)
            self.b_actor = List[Float32](capacity=B_ACTOR_SIZE)
            self.W_critic = List[Float32](capacity=W_CRITIC_SIZE)
            self.b_critic = List[Float32](capacity=B_CRITIC_SIZE)

            var std1 = sqrt(2.0 / Float64(EnvType.OBS_DIM + Self.HIDDEN_DIM))
            for _ in range(W1_SIZE):
                self.W1.append(Float32((random_float64() - 0.5) * 2 * std1))
            for _ in range(B1_SIZE):
                self.b1.append(Float32(0))

            var std_actor = sqrt(
                2.0 / Float64(Self.HIDDEN_DIM + EnvType.NUM_ACTIONS)
            )
            for _ in range(W_ACTOR_SIZE):
                self.W_actor.append(
                    Float32((random_float64() - 0.5) * 2 * std_actor)
                )
            for _ in range(B_ACTOR_SIZE):
                self.b_actor.append(Float32(0))

            var std_critic = sqrt(2.0 / Float64(Self.HIDDEN_DIM + 1))
            for _ in range(W_CRITIC_SIZE):
                self.W_critic.append(
                    Float32((random_float64() - 0.5) * 2 * std_critic)
                )
            self.b_critic.append(Float32(0))

            # Copy weights to GPU
            with W1_buf.map_to_host() as host:
                for i in range(W1_SIZE):
                    host[i] = Scalar[dtype](self.W1[i])
            with b1_buf.map_to_host() as host:
                for i in range(B1_SIZE):
                    host[i] = Scalar[dtype](self.b1[i])
            with W_actor_buf.map_to_host() as host:
                for i in range(W_ACTOR_SIZE):
                    host[i] = Scalar[dtype](self.W_actor[i])
            with b_actor_buf.map_to_host() as host:
                for i in range(B_ACTOR_SIZE):
                    host[i] = Scalar[dtype](self.b_actor[i])
            with W_critic_buf.map_to_host() as host:
                for i in range(W_CRITIC_SIZE):
                    host[i] = Scalar[dtype](self.W_critic[i])
            with b_critic_buf.map_to_host() as host:
                host[0] = Scalar[dtype](self.b_critic[0])

            # Initialize episode tracking buffers
            episode_rewards_buf.enqueue_fill(0)
            completed_episodes_buf.enqueue_fill(0)

            # Create tensors with parameterized layouts (fully composable)
            var states = LayoutTensor[
                dtype,
                Layout.row_major(NUM_ENVS, EnvType.STATE_SIZE),
                MutAnyOrigin,
            ](states_buf)
            var actions = LayoutTensor[
                DType.int32, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin
            ](actions_buf)
            var rewards = LayoutTensor[
                dtype, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin
            ](rewards_buf)
            var dones = LayoutTensor[
                DType.int32, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin
            ](dones_buf)
            var rng_states = LayoutTensor[
                DType.uint32, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin
            ](rng_buf)

            var W1 = LayoutTensor[
                dtype,
                Layout.row_major(EnvType.OBS_DIM, Self.HIDDEN_DIM),
                ImmutAnyOrigin,
            ](W1_buf)
            var b1 = LayoutTensor[
                dtype, Layout.row_major(Self.HIDDEN_DIM), ImmutAnyOrigin
            ](b1_buf)
            var W_actor = LayoutTensor[
                dtype,
                Layout.row_major(Self.HIDDEN_DIM, EnvType.NUM_ACTIONS),
                ImmutAnyOrigin,
            ](W_actor_buf)
            var b_actor = LayoutTensor[
                dtype, Layout.row_major(EnvType.NUM_ACTIONS), ImmutAnyOrigin
            ](b_actor_buf)
            var W_critic = LayoutTensor[
                dtype, Layout.row_major(Self.HIDDEN_DIM, 1), ImmutAnyOrigin
            ](W_critic_buf)
            var b_critic = LayoutTensor[
                dtype, Layout.row_major(1), ImmutAnyOrigin
            ](b_critic_buf)

            var obs = LayoutTensor[
                dtype,
                Layout.row_major(NUM_ENVS, EnvType.OBS_DIM),
                ImmutAnyOrigin,
            ](states_buf)
            var log_probs = LayoutTensor[
                dtype, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin
            ](log_probs_buf)
            var values = LayoutTensor[
                dtype, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin
            ](values_buf)
            var bootstrap_values = LayoutTensor[
                dtype, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin
            ](bootstrap_values_buf)

            var rollout_obs = LayoutTensor[
                dtype,
                Layout.row_major(NUM_ENVS, ROLLOUT_LEN, EnvType.OBS_DIM),
                MutAnyOrigin,
            ](rollout_obs_buf)
            var rollout_actions = LayoutTensor[
                DType.int32,
                Layout.row_major(NUM_ENVS, ROLLOUT_LEN),
                MutAnyOrigin,
            ](rollout_actions_buf)
            var rollout_rewards = LayoutTensor[
                dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin
            ](rollout_rewards_buf)
            var rollout_dones = LayoutTensor[
                DType.int32,
                Layout.row_major(NUM_ENVS, ROLLOUT_LEN),
                MutAnyOrigin,
            ](rollout_dones_buf)
            var rollout_log_probs = LayoutTensor[
                dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin
            ](rollout_log_probs_buf)
            var rollout_values = LayoutTensor[
                dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin
            ](rollout_values_buf)
            var rollout_advantages = LayoutTensor[
                dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin
            ](rollout_advantages_buf)
            var rollout_returns = LayoutTensor[
                dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin
            ](rollout_returns_buf)

            var episode_rewards = LayoutTensor[
                dtype, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin
            ](episode_rewards_buf)
            var completed_episodes = LayoutTensor[
                DType.int32, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin
            ](completed_episodes_buf)
            var completed_rewards = LayoutTensor[
                dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin
            ](completed_rewards_buf)

            var grad_W1 = LayoutTensor[
                dtype,
                Layout.row_major(NUM_ENVS, EnvType.OBS_DIM * Self.HIDDEN_DIM),
                MutAnyOrigin,
            ](grad_W1_buf)
            var grad_b1 = LayoutTensor[
                dtype, Layout.row_major(NUM_ENVS, Self.HIDDEN_DIM), MutAnyOrigin
            ](grad_b1_buf)
            var grad_W_actor = LayoutTensor[
                dtype,
                Layout.row_major(
                    NUM_ENVS, Self.HIDDEN_DIM * EnvType.NUM_ACTIONS
                ),
                MutAnyOrigin,
            ](grad_W_actor_buf)
            var grad_b_actor = LayoutTensor[
                dtype,
                Layout.row_major(NUM_ENVS, EnvType.NUM_ACTIONS),
                MutAnyOrigin,
            ](grad_b_actor_buf)
            var grad_W_critic = LayoutTensor[
                dtype, Layout.row_major(NUM_ENVS, Self.HIDDEN_DIM), MutAnyOrigin
            ](grad_W_critic_buf)
            var grad_b_critic = LayoutTensor[
                dtype, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin
            ](grad_b_critic_buf)

            var reduced_W1 = LayoutTensor[
                dtype, Layout.row_major(W1_SIZE), MutAnyOrigin
            ](reduced_W1_buf)
            var reduced_b1 = LayoutTensor[
                dtype, Layout.row_major(B1_SIZE), MutAnyOrigin
            ](reduced_b1_buf)
            var reduced_W_actor = LayoutTensor[
                dtype, Layout.row_major(W_ACTOR_SIZE), MutAnyOrigin
            ](reduced_W_actor_buf)
            var reduced_b_actor = LayoutTensor[
                dtype, Layout.row_major(B_ACTOR_SIZE), MutAnyOrigin
            ](reduced_b_actor_buf)
            var reduced_W_critic = LayoutTensor[
                dtype, Layout.row_major(W_CRITIC_SIZE), MutAnyOrigin
            ](reduced_W_critic_buf)
            var reduced_b_critic = LayoutTensor[
                dtype, Layout.row_major(B_CRITIC_SIZE), MutAnyOrigin
            ](reduced_b_critic_buf)

            var W1_mut = LayoutTensor[
                dtype, Layout.row_major(W1_SIZE), MutAnyOrigin
            ](W1_buf)
            var b1_mut = LayoutTensor[
                dtype, Layout.row_major(B1_SIZE), MutAnyOrigin
            ](b1_buf)
            var W_actor_mut = LayoutTensor[
                dtype, Layout.row_major(W_ACTOR_SIZE), MutAnyOrigin
            ](W_actor_buf)
            var b_actor_mut = LayoutTensor[
                dtype, Layout.row_major(B_ACTOR_SIZE), MutAnyOrigin
            ](b_actor_buf)
            var W_critic_mut = LayoutTensor[
                dtype, Layout.row_major(W_CRITIC_SIZE), MutAnyOrigin
            ](W_critic_buf)
            var b_critic_mut = LayoutTensor[
                dtype, Layout.row_major(B_CRITIC_SIZE), MutAnyOrigin
            ](b_critic_buf)

            # Scalars
            var gamma_s = Scalar[dtype](gamma)
            var gae_lambda_s = Scalar[dtype](gae_lambda)
            var entropy_coef_s = Scalar[dtype](entropy_coef)
            var value_coef_s = Scalar[dtype](value_coef)
            var lr_s = Scalar[dtype](lr)

            # Adam hyperparameters
            var beta1_s = Scalar[dtype](beta1)
            var beta2_s = Scalar[dtype](beta2)
            var adam_eps_s = Scalar[dtype](adam_eps)

            # Create Adam moment tensors
            var m_W1 = LayoutTensor[
                dtype, Layout.row_major(W1_SIZE), MutAnyOrigin
            ](m_W1_buf)
            var v_W1 = LayoutTensor[
                dtype, Layout.row_major(W1_SIZE), MutAnyOrigin
            ](v_W1_buf)
            var m_b1 = LayoutTensor[
                dtype, Layout.row_major(B1_SIZE), MutAnyOrigin
            ](m_b1_buf)
            var v_b1 = LayoutTensor[
                dtype, Layout.row_major(B1_SIZE), MutAnyOrigin
            ](v_b1_buf)
            var m_W_actor = LayoutTensor[
                dtype, Layout.row_major(W_ACTOR_SIZE), MutAnyOrigin
            ](m_W_actor_buf)
            var v_W_actor = LayoutTensor[
                dtype, Layout.row_major(W_ACTOR_SIZE), MutAnyOrigin
            ](v_W_actor_buf)
            var m_b_actor = LayoutTensor[
                dtype, Layout.row_major(B_ACTOR_SIZE), MutAnyOrigin
            ](m_b_actor_buf)
            var v_b_actor = LayoutTensor[
                dtype, Layout.row_major(B_ACTOR_SIZE), MutAnyOrigin
            ](v_b_actor_buf)
            var m_W_critic = LayoutTensor[
                dtype, Layout.row_major(W_CRITIC_SIZE), MutAnyOrigin
            ](m_W_critic_buf)
            var v_W_critic = LayoutTensor[
                dtype, Layout.row_major(W_CRITIC_SIZE), MutAnyOrigin
            ](v_W_critic_buf)
            var m_b_critic = LayoutTensor[
                dtype, Layout.row_major(B_CRITIC_SIZE), MutAnyOrigin
            ](m_b_critic_buf)
            var v_b_critic = LayoutTensor[
                dtype, Layout.row_major(B_CRITIC_SIZE), MutAnyOrigin
            ](v_b_critic_buf)

            # Grid dimensions computed from parameterized NUM_ENVS
            comptime env_blocks = (NUM_ENVS + TPB - 1) // TPB
            comptime env_threads = TPB

            # Reset environments using composable reset_all_kernel
            ctx.enqueue_function_checked[
                reset_all_kernel[EnvType, NUM_ENVS],
                reset_all_kernel[EnvType, NUM_ENVS],
            ](
                states,
                rng_states,
                grid_dim=(env_blocks,),
                block_dim=(env_threads,),
            )
            ctx.synchronize()

            if verbose:
                print("=" * 60)
                print("A2C Training on GPU Environment (Adam Optimizer)")
                print("=" * 60)
                print("  Environments:", NUM_ENVS)
                print("  Rollout length:", ROLLOUT_LEN)
                print("  Hidden dim:", Self.HIDDEN_DIM)
                print("  Learning rate:", lr)
                print("  Adam beta1:", beta1)
                print("  Adam beta2:", beta2)
                print("  Adam epsilon:", adam_eps)
                print("  Entropy coef:", entropy_coef)
                print()

            var start_time = perf_counter_ns()
            var total_episodes_count: Int = 0
            var episode_counter: Int = 0

            # Training loop
            for update in range(num_updates):
                # Phase 1: Collect rollout (FUSED - single kernel for all steps)
                ctx.enqueue_function_checked[
                    fused_rollout_kernel[
                        EnvType, NUM_ENVS, Self.HIDDEN_DIM, ROLLOUT_LEN
                    ],
                    fused_rollout_kernel[
                        EnvType, NUM_ENVS, Self.HIDDEN_DIM, ROLLOUT_LEN
                    ],
                ](
                    states,
                    rng_states,
                    W1,
                    b1,
                    W_actor,
                    b_actor,
                    W_critic,
                    b_critic,
                    rollout_obs,
                    rollout_actions,
                    rollout_rewards,
                    rollout_dones,
                    rollout_log_probs,
                    rollout_values,
                    grid_dim=(env_blocks,),
                    block_dim=(env_threads,),
                )

                # Track episodes for metrics
                var rollout_rewards_i = LayoutTensor[
                    dtype,
                    Layout.row_major(NUM_ENVS, ROLLOUT_LEN),
                    ImmutAnyOrigin,
                ](rollout_rewards_buf)
                var rollout_dones_i = LayoutTensor[
                    DType.int32,
                    Layout.row_major(NUM_ENVS, ROLLOUT_LEN),
                    ImmutAnyOrigin,
                ](rollout_dones_buf)

                ctx.enqueue_function_checked[
                    track_episodes_kernel[NUM_ENVS, ROLLOUT_LEN],
                    track_episodes_kernel[NUM_ENVS, ROLLOUT_LEN],
                ](
                    rollout_rewards_i,
                    rollout_dones_i,
                    episode_rewards,
                    completed_episodes,
                    completed_rewards,
                    grid_dim=(env_blocks,),
                    block_dim=(env_threads,),
                )

                # Phase 2: Compute GAE
                ctx.enqueue_function_checked[
                    get_values_kernel[EnvType, NUM_ENVS, Self.HIDDEN_DIM],
                    get_values_kernel[EnvType, NUM_ENVS, Self.HIDDEN_DIM],
                ](
                    obs,
                    W1,
                    b1,
                    W_critic,
                    b_critic,
                    bootstrap_values,
                    grid_dim=(env_blocks,),
                    block_dim=(env_threads,),
                )

                var rollout_values_i = LayoutTensor[
                    dtype,
                    Layout.row_major(NUM_ENVS, ROLLOUT_LEN),
                    ImmutAnyOrigin,
                ](rollout_values_buf)
                var bootstrap_values_i = LayoutTensor[
                    dtype, Layout.row_major(NUM_ENVS, 1), ImmutAnyOrigin
                ](bootstrap_values_buf)

                ctx.enqueue_function_checked[
                    compute_gae_kernel[NUM_ENVS, ROLLOUT_LEN],
                    compute_gae_kernel[NUM_ENVS, ROLLOUT_LEN],
                ](
                    gamma_s,
                    gae_lambda_s,
                    rollout_rewards_i,
                    rollout_dones_i,
                    rollout_values_i,
                    bootstrap_values_i,
                    rollout_advantages,
                    rollout_returns,
                    grid_dim=(env_blocks,),
                    block_dim=(env_threads,),
                )

                # Phase 3: Policy gradients
                var rollout_obs_i = LayoutTensor[
                    dtype,
                    Layout.row_major(NUM_ENVS, ROLLOUT_LEN, EnvType.OBS_DIM),
                    ImmutAnyOrigin,
                ](rollout_obs_buf)
                var rollout_actions_i = LayoutTensor[
                    DType.int32,
                    Layout.row_major(NUM_ENVS, ROLLOUT_LEN),
                    ImmutAnyOrigin,
                ](rollout_actions_buf)
                var rollout_advantages_i = LayoutTensor[
                    dtype,
                    Layout.row_major(NUM_ENVS, ROLLOUT_LEN),
                    ImmutAnyOrigin,
                ](rollout_advantages_buf)
                var rollout_returns_i = LayoutTensor[
                    dtype,
                    Layout.row_major(NUM_ENVS, ROLLOUT_LEN),
                    ImmutAnyOrigin,
                ](rollout_returns_buf)

                ctx.enqueue_function_checked[
                    policy_gradient_kernel[
                        EnvType, NUM_ENVS, Self.HIDDEN_DIM, ROLLOUT_LEN
                    ],
                    policy_gradient_kernel[
                        EnvType, NUM_ENVS, Self.HIDDEN_DIM, ROLLOUT_LEN
                    ],
                ](
                    rollout_obs_i,
                    rollout_actions_i,
                    rollout_advantages_i,
                    rollout_returns_i,
                    W1,
                    b1,
                    W_actor,
                    b_actor,
                    W_critic,
                    b_critic,
                    grad_W1,
                    grad_b1,
                    grad_W_actor,
                    grad_b_actor,
                    grad_W_critic,
                    grad_b_critic,
                    entropy_coef_s,
                    value_coef_s,
                    grid_dim=(env_blocks,),
                    block_dim=(env_threads,),
                )

                # Phase 4: Reduce and update ALL parameters
                var grad_W1_i = LayoutTensor[
                    dtype, Layout.row_major(NUM_ENVS, W1_SIZE), ImmutAnyOrigin
                ](grad_W1_buf)
                var grad_b1_i = LayoutTensor[
                    dtype, Layout.row_major(NUM_ENVS, B1_SIZE), ImmutAnyOrigin
                ](grad_b1_buf)
                var grad_W_actor_i = LayoutTensor[
                    dtype,
                    Layout.row_major(NUM_ENVS, W_ACTOR_SIZE),
                    ImmutAnyOrigin,
                ](grad_W_actor_buf)
                var grad_b_actor_i = LayoutTensor[
                    dtype,
                    Layout.row_major(NUM_ENVS, B_ACTOR_SIZE),
                    ImmutAnyOrigin,
                ](grad_b_actor_buf)
                var grad_W_critic_i = LayoutTensor[
                    dtype,
                    Layout.row_major(NUM_ENVS, W_CRITIC_SIZE),
                    ImmutAnyOrigin,
                ](grad_W_critic_buf)
                var grad_b_critic_i = LayoutTensor[
                    dtype,
                    Layout.row_major(NUM_ENVS, B_CRITIC_SIZE),
                    ImmutAnyOrigin,
                ](grad_b_critic_buf)

                var reduced_W1_i = LayoutTensor[
                    dtype, Layout.row_major(W1_SIZE), ImmutAnyOrigin
                ](reduced_W1_buf)
                var reduced_b1_i = LayoutTensor[
                    dtype, Layout.row_major(B1_SIZE), ImmutAnyOrigin
                ](reduced_b1_buf)
                var reduced_W_actor_i = LayoutTensor[
                    dtype, Layout.row_major(W_ACTOR_SIZE), ImmutAnyOrigin
                ](reduced_W_actor_buf)
                var reduced_b_actor_i = LayoutTensor[
                    dtype, Layout.row_major(B_ACTOR_SIZE), ImmutAnyOrigin
                ](reduced_b_actor_buf)
                var reduced_W_critic_i = LayoutTensor[
                    dtype, Layout.row_major(W_CRITIC_SIZE), ImmutAnyOrigin
                ](reduced_W_critic_buf)
                var reduced_b_critic_i = LayoutTensor[
                    dtype, Layout.row_major(B_CRITIC_SIZE), ImmutAnyOrigin
                ](reduced_b_critic_buf)

                # Block counts for each parameter size (one thread per element)
                comptime blocks_W1 = (W1_SIZE + TPB - 1) // TPB
                comptime blocks_b1 = (B1_SIZE + TPB - 1) // TPB
                comptime blocks_W_actor = (W_ACTOR_SIZE + TPB - 1) // TPB
                comptime blocks_b_actor = (B_ACTOR_SIZE + TPB - 1) // TPB
                comptime blocks_W_critic = (W_CRITIC_SIZE + TPB - 1) // TPB
                comptime blocks_b_critic = (B_CRITIC_SIZE + TPB - 1) // TPB

                # Reduce all gradients using parameterized kernel
                ctx.enqueue_function_checked[
                    reduce_kernel[W1_SIZE, NUM_ENVS],
                    reduce_kernel[W1_SIZE, NUM_ENVS],
                ](
                    reduced_W1,
                    grad_W1_i,
                    grid_dim=(blocks_W1,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function_checked[
                    reduce_kernel[B1_SIZE, NUM_ENVS],
                    reduce_kernel[B1_SIZE, NUM_ENVS],
                ](
                    reduced_b1,
                    grad_b1_i,
                    grid_dim=(blocks_b1,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function_checked[
                    reduce_kernel[W_ACTOR_SIZE, NUM_ENVS],
                    reduce_kernel[W_ACTOR_SIZE, NUM_ENVS],
                ](
                    reduced_W_actor,
                    grad_W_actor_i,
                    grid_dim=(blocks_W_actor,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function_checked[
                    reduce_kernel[B_ACTOR_SIZE, NUM_ENVS],
                    reduce_kernel[B_ACTOR_SIZE, NUM_ENVS],
                ](
                    reduced_b_actor,
                    grad_b_actor_i,
                    grid_dim=(blocks_b_actor,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function_checked[
                    reduce_kernel[W_CRITIC_SIZE, NUM_ENVS],
                    reduce_kernel[W_CRITIC_SIZE, NUM_ENVS],
                ](
                    reduced_W_critic,
                    grad_W_critic_i,
                    grid_dim=(blocks_W_critic,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function_checked[
                    reduce_kernel[B_CRITIC_SIZE, NUM_ENVS],
                    reduce_kernel[B_CRITIC_SIZE, NUM_ENVS],
                ](
                    reduced_b_critic,
                    grad_b_critic_i,
                    grid_dim=(blocks_b_critic,),
                    block_dim=(TPB,),
                )

                # Compute Adam bias corrections (t = update + 1, 1-indexed)
                var t = update + 1
                var beta1_t = beta1_s
                var beta2_t = beta2_s
                for _ in range(1, t):
                    beta1_t *= beta1_s
                    beta2_t *= beta2_s
                var bias_correction1 = Scalar[dtype](1) - beta1_t
                var bias_correction2 = Scalar[dtype](1) - beta2_t

                # Adam update all parameters using parameterized kernel
                ctx.enqueue_function_checked[
                    adam_kernel[W1_SIZE], adam_kernel[W1_SIZE]
                ](
                    W1_mut,
                    reduced_W1_i,
                    m_W1,
                    v_W1,
                    lr_s,
                    beta1_s,
                    beta2_s,
                    adam_eps_s,
                    bias_correction1,
                    bias_correction2,
                    grid_dim=(blocks_W1,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function_checked[
                    adam_kernel[B1_SIZE], adam_kernel[B1_SIZE]
                ](
                    b1_mut,
                    reduced_b1_i,
                    m_b1,
                    v_b1,
                    lr_s,
                    beta1_s,
                    beta2_s,
                    adam_eps_s,
                    bias_correction1,
                    bias_correction2,
                    grid_dim=(blocks_b1,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function_checked[
                    adam_kernel[W_ACTOR_SIZE], adam_kernel[W_ACTOR_SIZE]
                ](
                    W_actor_mut,
                    reduced_W_actor_i,
                    m_W_actor,
                    v_W_actor,
                    lr_s,
                    beta1_s,
                    beta2_s,
                    adam_eps_s,
                    bias_correction1,
                    bias_correction2,
                    grid_dim=(blocks_W_actor,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function_checked[
                    adam_kernel[B_ACTOR_SIZE], adam_kernel[B_ACTOR_SIZE]
                ](
                    b_actor_mut,
                    reduced_b_actor_i,
                    m_b_actor,
                    v_b_actor,
                    lr_s,
                    beta1_s,
                    beta2_s,
                    adam_eps_s,
                    bias_correction1,
                    bias_correction2,
                    grid_dim=(blocks_b_actor,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function_checked[
                    adam_kernel[W_CRITIC_SIZE], adam_kernel[W_CRITIC_SIZE]
                ](
                    W_critic_mut,
                    reduced_W_critic_i,
                    m_W_critic,
                    v_W_critic,
                    lr_s,
                    beta1_s,
                    beta2_s,
                    adam_eps_s,
                    bias_correction1,
                    bias_correction2,
                    grid_dim=(blocks_W_critic,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function_checked[
                    adam_kernel[B_CRITIC_SIZE], adam_kernel[B_CRITIC_SIZE]
                ](
                    b_critic_mut,
                    reduced_b_critic_i,
                    m_b_critic,
                    v_b_critic,
                    lr_s,
                    beta1_s,
                    beta2_s,
                    adam_eps_s,
                    bias_correction1,
                    bias_correction2,
                    grid_dim=(blocks_b_critic,),
                    block_dim=(TPB,),
                )

                # Collect metrics every update (or periodically for efficiency)
                if (update + 1) % 10 == 0 or update == num_updates - 1:
                    ctx.synchronize()

                    # Collect completed episode rewards
                    with completed_episodes_buf.map_to_host() as ep_counts:
                        with completed_rewards_buf.map_to_host() as ep_rewards:
                            for env_idx in range(NUM_ENVS):
                                var num_completed = Int(ep_counts[env_idx])
                                for ep_idx in range(num_completed):
                                    var reward = Float64(
                                        ep_rewards[
                                            env_idx * ROLLOUT_LEN + ep_idx
                                        ]
                                    )
                                    metrics.log_episode(
                                        episode_counter, reward, 0, 0.0
                                    )
                                    episode_counter += 1
                                    total_episodes_count += 1

                    # Reset episode tracking buffers for next batch
                    completed_episodes_buf.enqueue_fill(0)

                # Logging - reduced frequency to minimize GPU sync overhead
                if verbose and (update + 1) % 25 == 0:
                    var steps = (update + 1) * NUM_ENVS * ROLLOUT_LEN
                    var avg_len = (
                        Float64(steps)
                        / Float64(total_episodes_count) if total_episodes_count
                        > 0 else 0.0
                    )

                    # Get recent average reward
                    var recent_avg: Float64 = 0.0
                    var recent_count = min(100, len(metrics.episodes))
                    if recent_count > 0:
                        var start_idx = len(metrics.episodes) - recent_count
                        for j in range(start_idx, len(metrics.episodes)):
                            recent_avg += metrics.episodes[j].total_reward
                        recent_avg /= Float64(recent_count)

                    print(
                        "Update",
                        update + 1,
                        "| Steps:",
                        steps,
                        "| Episodes:",
                        total_episodes_count,
                        "| Avg reward:",
                        Int(recent_avg),
                        "| Avg len:",
                        Int(avg_len),
                    )

            ctx.synchronize()

            # Copy trained weights back to host
            with W1_buf.map_to_host() as host:
                for i in range(W1_SIZE):
                    self.W1[i] = Float32(host[i])
            with b1_buf.map_to_host() as host:
                for i in range(B1_SIZE):
                    self.b1[i] = Float32(host[i])
            with W_actor_buf.map_to_host() as host:
                for i in range(W_ACTOR_SIZE):
                    self.W_actor[i] = Float32(host[i])
            with b_actor_buf.map_to_host() as host:
                for i in range(B_ACTOR_SIZE):
                    self.b_actor[i] = Float32(host[i])
            with W_critic_buf.map_to_host() as host:
                for i in range(W_CRITIC_SIZE):
                    self.W_critic[i] = Float32(host[i])
            with b_critic_buf.map_to_host() as host:
                self.b_critic[0] = Float32(host[0])

            var end_time = perf_counter_ns()

            var total_steps_count = num_updates * NUM_ENVS * ROLLOUT_LEN
            var elapsed = Float64(end_time - start_time) / 1e9
            var throughput = Float64(total_steps_count) / elapsed

            self.trained = True
            self.total_episodes = total_episodes_count
            self.total_steps = total_steps_count

            if verbose:
                print()
                print("Training complete!")
                print("  Total steps:", total_steps_count)
                print("  Total episodes:", total_episodes_count)
                print("  Time:", elapsed, "seconds")
                print("  Throughput:", Int(throughput), "steps/sec")

        return metrics^

    fn evaluate[
        E: BoxDiscreteActionEnv
    ](
        self,
        mut env: E,
        num_episodes: Int = 10,
        max_steps_per_episode: Int = 500,
        verbose: Bool = False,
        render: Bool = False,
    ) -> Float64:
        """Evaluate the trained agent on a CPU environment.

        Uses deterministic policy (argmax over logits).

        Args:
            env: Environment implementing BoxDiscreteActionEnv trait.
            num_episodes: Number of evaluation episodes.
            max_steps_per_episode: Maximum steps per episode.
            verbose: Print per-episode results.
            render: Render environment during evaluation.

        Returns:
            Average reward over evaluation episodes.
        """
        var total_reward: Float64 = 0.0

        for ep in range(num_episodes):
            var obs_list = env.reset_obs_list()
            var episode_reward: Float64 = 0.0
            var done = False
            var steps = 0

            while not done and steps < max_steps_per_episode:
                if render:
                    env.render()

                # Select action using deterministic policy
                var action = self.select_action(obs_list)

                # Step environment
                var step_result = env.step_obs(action)
                var reward = step_result[1]
                done = step_result[2]

                episode_reward += reward
                obs_list = env.get_obs_list()
                steps += 1

            total_reward += episode_reward

            if verbose:
                print(
                    "  Eval episode "
                    + String(ep + 1)
                    + ": "
                    + String(episode_reward)[:10]
                    + " steps: "
                    + String(steps)
                )

        return total_reward / Float64(num_episodes)

    fn print_info(self):
        """Print agent information."""
        print("GPU A2C Agent:")
        print("  Obs dim:", self.obs_dim)
        print("  Num actions:", self.num_actions)
        print("  Hidden dim:", Self.HIDDEN_DIM)
        print("  Trained:", self.trained)
        print("  Total episodes:", self.total_episodes)
        print("  Total steps:", self.total_steps)

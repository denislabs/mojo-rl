"""GPU A2C (Advantage Actor-Critic) with Shared Network.

Clean, composable implementation for vectorized GPU environments.
Uses module-level constants to avoid parameterization overhead.

Run with:
    pixi run -e apple mojo run deep_agents/gpu/a2c.mojo
"""

from time import perf_counter_ns
from math import exp, log, sqrt
from random import random_float64, seed

from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from deep_rl.gpu import xorshift32, random_uniform
from envs.cartpole_gpu import (
    step_kernel,
    reset_kernel,
    reset_where_done_kernel,
    BLOCKS_PER_GRID,
    THREADS_PER_BLOCK,
    state_layout,
    action_layout,
    rng_layout,
    reward_layout,
    done_layout,
)


# =============================================================================
# Constants (module-level, no parameterization)
# =============================================================================

comptime dtype = DType.float32
comptime NUM_ENVS: Int = 1024
comptime OBS_DIM: Int = 4
comptime NUM_ACTIONS: Int = 2
comptime STATE_SIZE: Int = 4
comptime HIDDEN_DIM: Int = 64
comptime ROLLOUT_LEN: Int = 32
comptime TPB: Int = 256


# =============================================================================
# Forward Pass Kernel: obs -> (action, log_prob, value)
# =============================================================================


fn forward_sample_kernel(
    obs: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, OBS_DIM), ImmutAnyOrigin
    ],
    W1: LayoutTensor[
        dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), ImmutAnyOrigin
    ],
    b1: LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin],
    W_actor: LayoutTensor[
        dtype, Layout.row_major(HIDDEN_DIM, NUM_ACTIONS), ImmutAnyOrigin
    ],
    b_actor: LayoutTensor[dtype, Layout.row_major(NUM_ACTIONS), ImmutAnyOrigin],
    W_critic: LayoutTensor[
        dtype, Layout.row_major(HIDDEN_DIM, 1), ImmutAnyOrigin
    ],
    b_critic: LayoutTensor[dtype, Layout.row_major(1), ImmutAnyOrigin],
    rng_states: LayoutTensor[
        DType.uint32, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin
    ],
    actions: LayoutTensor[
        DType.int32, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin
    ],
    log_probs: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin],
    values: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin],
):
    """Forward pass + action sampling."""
    var env_idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env_idx >= NUM_ENVS:
        return

    # Load observation
    var o = InlineArray[Scalar[dtype], OBS_DIM](fill=Scalar[dtype](0))
    for i in range(OBS_DIM):
        o[i] = rebind[Scalar[dtype]](obs[env_idx, i])

    # Hidden layer: h = ReLU(obs @ W1 + b1)
    var h = InlineArray[Scalar[dtype], HIDDEN_DIM](fill=Scalar[dtype](0))
    for j in range(HIDDEN_DIM):
        var sum_val = rebind[Scalar[dtype]](b1[j])
        for i in range(OBS_DIM):
            sum_val += o[i] * rebind[Scalar[dtype]](W1[i, j])
        h[j] = sum_val if sum_val > Scalar[dtype](0) else Scalar[dtype](0)

    # Actor: logits
    var logits = InlineArray[Scalar[dtype], NUM_ACTIONS](fill=Scalar[dtype](0))
    for j in range(NUM_ACTIONS):
        var sum_val = rebind[Scalar[dtype]](b_actor[j])
        for k in range(HIDDEN_DIM):
            sum_val += h[k] * rebind[Scalar[dtype]](W_actor[k, j])
        logits[j] = sum_val

    # Critic: value
    var value: Scalar[dtype] = rebind[Scalar[dtype]](b_critic[0])
    for k in range(HIDDEN_DIM):
        value += h[k] * rebind[Scalar[dtype]](W_critic[k, 0])

    # Softmax
    var max_logit = logits[0]
    if logits[1] > max_logit:
        max_logit = logits[1]

    var exp0 = exp(logits[0] - max_logit)
    var exp1 = exp(logits[1] - max_logit)
    var sum_exp = exp0 + exp1
    var prob0 = exp0 / sum_exp
    var prob1 = exp1 / sum_exp

    # Sample action
    var rng = rebind[Scalar[DType.uint32]](rng_states[env_idx, 0])
    var u_result = random_uniform[dtype](rng)
    rng = u_result[1]
    rng_states[env_idx, 0] = rng

    var action = 0 if u_result[0] < prob0 else 1
    var log_prob = log((prob0 if action == 0 else prob1) + Scalar[dtype](1e-8))

    actions[env_idx, 0] = Int32(action)
    log_probs[env_idx, 0] = log_prob
    values[env_idx, 0] = value


# =============================================================================
# Store Rollout Kernel
# =============================================================================


fn store_rollout_kernel(
    step_idx: Int,
    obs: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, OBS_DIM), ImmutAnyOrigin
    ],
    actions: LayoutTensor[
        DType.int32, Layout.row_major(NUM_ENVS, 1), ImmutAnyOrigin
    ],
    rewards: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, 1), ImmutAnyOrigin],
    dones: LayoutTensor[
        DType.int32, Layout.row_major(NUM_ENVS, 1), ImmutAnyOrigin
    ],
    log_probs: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, 1), ImmutAnyOrigin
    ],
    values: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, 1), ImmutAnyOrigin],
    rollout_obs: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN, OBS_DIM), MutAnyOrigin
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
    """Store one step of rollout data."""
    var env_idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env_idx >= NUM_ENVS:
        return

    for i in range(OBS_DIM):
        rollout_obs[env_idx, step_idx, i] = obs[env_idx, i]

    rollout_actions[env_idx, step_idx] = actions[env_idx, 0]
    rollout_rewards[env_idx, step_idx] = rewards[env_idx, 0]
    rollout_dones[env_idx, step_idx] = dones[env_idx, 0]
    rollout_log_probs[env_idx, step_idx] = log_probs[env_idx, 0]
    rollout_values[env_idx, step_idx] = values[env_idx, 0]


# =============================================================================
# Compute GAE Kernel
# =============================================================================


fn compute_gae_kernel(
    gamma: Scalar[dtype],
    gae_lambda: Scalar[dtype],
    rollout_rewards: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), ImmutAnyOrigin
    ],
    rollout_dones: LayoutTensor[
        DType.int32, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), ImmutAnyOrigin
    ],
    rollout_values: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), ImmutAnyOrigin
    ],
    bootstrap_values: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, 1), ImmutAnyOrigin
    ],
    rollout_advantages: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin
    ],
    rollout_returns: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin
    ],
):
    """Compute GAE advantages."""
    var env_idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env_idx >= NUM_ENVS:
        return

    var gae: Scalar[dtype] = 0
    var next_value = rebind[Scalar[dtype]](bootstrap_values[env_idx, 0])

    for t in range(ROLLOUT_LEN - 1, -1, -1):
        var reward = rebind[Scalar[dtype]](rollout_rewards[env_idx, t])
        var done = rollout_dones[env_idx, t]
        var value = rebind[Scalar[dtype]](rollout_values[env_idx, t])

        var not_done = Scalar[dtype](1.0) if done == 0 else Scalar[dtype](0.0)
        var delta = reward + gamma * next_value * not_done - value
        gae = delta + gamma * gae_lambda * not_done * gae

        rollout_advantages[env_idx, t] = gae
        rollout_returns[env_idx, t] = gae + value
        next_value = value


# =============================================================================
# Get Values Kernel
# =============================================================================


fn get_values_kernel(
    obs: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, OBS_DIM), ImmutAnyOrigin
    ],
    W1: LayoutTensor[
        dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), ImmutAnyOrigin
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

    var o = InlineArray[Scalar[dtype], OBS_DIM](fill=Scalar[dtype](0))
    for i in range(OBS_DIM):
        o[i] = rebind[Scalar[dtype]](obs[env_idx, i])

    var h = InlineArray[Scalar[dtype], HIDDEN_DIM](fill=Scalar[dtype](0))
    for j in range(HIDDEN_DIM):
        var sum_val = rebind[Scalar[dtype]](b1[j])
        for i in range(OBS_DIM):
            sum_val += o[i] * rebind[Scalar[dtype]](W1[i, j])
        h[j] = sum_val if sum_val > Scalar[dtype](0) else Scalar[dtype](0)

    var value: Scalar[dtype] = rebind[Scalar[dtype]](b_critic[0])
    for k in range(HIDDEN_DIM):
        value += h[k] * rebind[Scalar[dtype]](W_critic[k, 0])

    values[env_idx, 0] = value


# =============================================================================
# Policy Gradient Kernel
# =============================================================================


fn policy_gradient_kernel(
    rollout_obs: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN, OBS_DIM), ImmutAnyOrigin
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
        dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), ImmutAnyOrigin
    ],
    b1: LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin],
    W_actor: LayoutTensor[
        dtype, Layout.row_major(HIDDEN_DIM, NUM_ACTIONS), ImmutAnyOrigin
    ],
    b_actor: LayoutTensor[dtype, Layout.row_major(NUM_ACTIONS), ImmutAnyOrigin],
    W_critic: LayoutTensor[
        dtype, Layout.row_major(HIDDEN_DIM, 1), ImmutAnyOrigin
    ],
    b_critic: LayoutTensor[dtype, Layout.row_major(1), ImmutAnyOrigin],
    grad_W1: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, OBS_DIM * HIDDEN_DIM), MutAnyOrigin
    ],
    grad_b1: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin
    ],
    grad_W_actor: LayoutTensor[
        dtype,
        Layout.row_major(NUM_ENVS, HIDDEN_DIM * NUM_ACTIONS),
        MutAnyOrigin,
    ],
    grad_b_actor: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), MutAnyOrigin
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
    var local_grad_W1 = InlineArray[Scalar[dtype], OBS_DIM * HIDDEN_DIM](
        fill=Scalar[dtype](0)
    )
    var local_grad_b1 = InlineArray[Scalar[dtype], HIDDEN_DIM](
        fill=Scalar[dtype](0)
    )
    var local_grad_W_actor = InlineArray[
        Scalar[dtype], HIDDEN_DIM * NUM_ACTIONS
    ](fill=Scalar[dtype](0))
    var local_grad_b_actor = InlineArray[Scalar[dtype], NUM_ACTIONS](
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
        var o = InlineArray[Scalar[dtype], OBS_DIM](fill=Scalar[dtype](0))
        for i in range(OBS_DIM):
            o[i] = rebind[Scalar[dtype]](rollout_obs[env_idx, t, i])

        # Forward: hidden
        var h = InlineArray[Scalar[dtype], HIDDEN_DIM](fill=Scalar[dtype](0))
        var h_pre = InlineArray[Scalar[dtype], HIDDEN_DIM](
            fill=Scalar[dtype](0)
        )
        for j in range(HIDDEN_DIM):
            var sum_val = rebind[Scalar[dtype]](b1[j])
            for i in range(OBS_DIM):
                sum_val += o[i] * rebind[Scalar[dtype]](W1[i, j])
            h_pre[j] = sum_val
            h[j] = sum_val if sum_val > Scalar[dtype](0) else Scalar[dtype](0)

        # Forward: logits
        var logits = InlineArray[Scalar[dtype], NUM_ACTIONS](
            fill=Scalar[dtype](0)
        )
        for j in range(NUM_ACTIONS):
            var sum_val = rebind[Scalar[dtype]](b_actor[j])
            for k in range(HIDDEN_DIM):
                sum_val += h[k] * rebind[Scalar[dtype]](W_actor[k, j])
            logits[j] = sum_val

        # Forward: value
        var value: Scalar[dtype] = rebind[Scalar[dtype]](b_critic[0])
        for k in range(HIDDEN_DIM):
            value += h[k] * rebind[Scalar[dtype]](W_critic[k, 0])

        # Softmax
        var max_logit = logits[0]
        if logits[1] > max_logit:
            max_logit = logits[1]
        var exp0 = exp(logits[0] - max_logit)
        var exp1 = exp(logits[1] - max_logit)
        var sum_exp = exp0 + exp1
        var prob0 = exp0 / sum_exp
        var prob1 = exp1 / sum_exp

        # Policy gradient
        var d_logit0 = (
            (Scalar[dtype](1) if action == 0 else Scalar[dtype](0)) - prob0
        ) * advantage
        var d_logit1 = (
            (Scalar[dtype](1) if action == 1 else Scalar[dtype](0)) - prob1
        ) * advantage

        # Value gradient
        var d_value = value_coef * (value - ret)

        # Backprop actor
        for k in range(HIDDEN_DIM):
            local_grad_W_actor[k * NUM_ACTIONS + 0] += h[k] * d_logit0
            local_grad_W_actor[k * NUM_ACTIONS + 1] += h[k] * d_logit1
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
            var dh_pre = dh[k] if h_pre[k] > Scalar[dtype](0) else Scalar[
                dtype
            ](0)
            for i in range(OBS_DIM):
                local_grad_W1[i * HIDDEN_DIM + k] += o[i] * dh_pre
            local_grad_b1[k] += dh_pre

    # Write gradients
    for i in range(OBS_DIM * HIDDEN_DIM):
        grad_W1[env_idx, i] = local_grad_W1[i]
    for k in range(HIDDEN_DIM):
        grad_b1[env_idx, k] = local_grad_b1[k]
    for i in range(HIDDEN_DIM * NUM_ACTIONS):
        grad_W_actor[env_idx, i] = local_grad_W_actor[i]
    for j in range(NUM_ACTIONS):
        grad_b_actor[env_idx, j] = local_grad_b_actor[j]
    for k in range(HIDDEN_DIM):
        grad_W_critic[env_idx, k] = local_grad_W_critic[k]
    grad_b_critic[env_idx, 0] = local_grad_b_critic


# =============================================================================
# Reduce and SGD Kernels
# =============================================================================


fn reduce_add_kernel(
    reduced: LayoutTensor[
        dtype, Layout.row_major(OBS_DIM * HIDDEN_DIM), MutAnyOrigin
    ],
    per_env: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, OBS_DIM * HIDDEN_DIM), ImmutAnyOrigin
    ],
):
    """Reduce W1 gradients."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= OBS_DIM * HIDDEN_DIM:
        return
    var sum_val: Scalar[dtype] = 0
    for env in range(NUM_ENVS):
        sum_val += rebind[Scalar[dtype]](per_env[env, idx])
    reduced[idx] = sum_val / Scalar[dtype](NUM_ENVS)


fn sgd_kernel(
    weights: LayoutTensor[
        dtype, Layout.row_major(OBS_DIM * HIDDEN_DIM), MutAnyOrigin
    ],
    grads: LayoutTensor[
        dtype, Layout.row_major(OBS_DIM * HIDDEN_DIM), ImmutAnyOrigin
    ],
    lr: Scalar[dtype],
):
    """SGD for W1."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= OBS_DIM * HIDDEN_DIM:
        return
    weights[idx] = rebind[Scalar[dtype]](weights[idx]) - lr * rebind[
        Scalar[dtype]
    ](grads[idx])


# =============================================================================
# Training
# =============================================================================


fn train_a2c(
    ctx: DeviceContext,
    num_updates: Int,
    lr: Float32 = 0.0007,
    gamma: Float32 = 0.99,
    gae_lambda: Float32 = 0.95,
    entropy_coef: Float32 = 0.01,
    value_coef: Float32 = 0.5,
    verbose: Bool = True,
) raises -> Float64:
    """Train A2C on GPU CartPole."""

    # Allocate buffers
    var states_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * STATE_SIZE)
    var actions_buf = ctx.enqueue_create_buffer[DType.int32](NUM_ENVS)
    var rewards_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS)
    var dones_buf = ctx.enqueue_create_buffer[DType.int32](NUM_ENVS)
    var rng_buf = ctx.enqueue_create_buffer[DType.uint32](NUM_ENVS)

    var W1_buf = ctx.enqueue_create_buffer[dtype](OBS_DIM * HIDDEN_DIM)
    var b1_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
    var W_actor_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM * NUM_ACTIONS)
    var b_actor_buf = ctx.enqueue_create_buffer[dtype](NUM_ACTIONS)
    var W_critic_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
    var b_critic_buf = ctx.enqueue_create_buffer[dtype](1)

    var log_probs_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS)
    var values_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS)
    var bootstrap_values_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS)

    var rollout_obs_buf = ctx.enqueue_create_buffer[dtype](
        NUM_ENVS * ROLLOUT_LEN * OBS_DIM
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

    var grad_W1_buf = ctx.enqueue_create_buffer[dtype](
        NUM_ENVS * OBS_DIM * HIDDEN_DIM
    )
    var grad_b1_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * HIDDEN_DIM)
    var grad_W_actor_buf = ctx.enqueue_create_buffer[dtype](
        NUM_ENVS * HIDDEN_DIM * NUM_ACTIONS
    )
    var grad_b_actor_buf = ctx.enqueue_create_buffer[dtype](
        NUM_ENVS * NUM_ACTIONS
    )
    var grad_W_critic_buf = ctx.enqueue_create_buffer[dtype](
        NUM_ENVS * HIDDEN_DIM
    )
    var grad_b_critic_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS)

    var reduced_W1_buf = ctx.enqueue_create_buffer[dtype](OBS_DIM * HIDDEN_DIM)

    # Initialize
    with rng_buf.map_to_host() as host:
        for i in range(NUM_ENVS):
            host[i] = UInt32(i + 12345)

    var std1 = sqrt(2.0 / Float64(OBS_DIM + HIDDEN_DIM))
    with W1_buf.map_to_host() as host:
        for i in range(OBS_DIM * HIDDEN_DIM):
            host[i] = Scalar[dtype]((random_float64() - 0.5) * 2 * std1)
    b1_buf.enqueue_fill(0)

    var std_actor = sqrt(2.0 / Float64(HIDDEN_DIM + NUM_ACTIONS))
    with W_actor_buf.map_to_host() as host:
        for i in range(HIDDEN_DIM * NUM_ACTIONS):
            host[i] = Scalar[dtype]((random_float64() - 0.5) * 2 * std_actor)
    b_actor_buf.enqueue_fill(0)

    var std_critic = sqrt(2.0 / Float64(HIDDEN_DIM + 1))
    with W_critic_buf.map_to_host() as host:
        for i in range(HIDDEN_DIM):
            host[i] = Scalar[dtype]((random_float64() - 0.5) * 2 * std_critic)
    b_critic_buf.enqueue_fill(0)

    # Create tensors
    var states = LayoutTensor[dtype, state_layout, MutAnyOrigin](states_buf)
    var actions = LayoutTensor[DType.int32, action_layout, MutAnyOrigin](
        actions_buf
    )
    var rewards = LayoutTensor[dtype, reward_layout, MutAnyOrigin](rewards_buf)
    var dones = LayoutTensor[DType.int32, done_layout, MutAnyOrigin](dones_buf)
    var rng_states = LayoutTensor[DType.uint32, rng_layout, MutAnyOrigin](
        rng_buf
    )

    var W1 = LayoutTensor[
        dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), ImmutAnyOrigin
    ](W1_buf)
    var b1 = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin](
        b1_buf
    )
    var W_actor = LayoutTensor[
        dtype, Layout.row_major(HIDDEN_DIM, NUM_ACTIONS), ImmutAnyOrigin
    ](W_actor_buf)
    var b_actor = LayoutTensor[
        dtype, Layout.row_major(NUM_ACTIONS), ImmutAnyOrigin
    ](b_actor_buf)
    var W_critic = LayoutTensor[
        dtype, Layout.row_major(HIDDEN_DIM, 1), ImmutAnyOrigin
    ](W_critic_buf)
    var b_critic = LayoutTensor[dtype, Layout.row_major(1), ImmutAnyOrigin](
        b_critic_buf
    )

    var obs = LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, OBS_DIM), ImmutAnyOrigin
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
        dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN, OBS_DIM), MutAnyOrigin
    ](rollout_obs_buf)
    var rollout_actions = LayoutTensor[
        DType.int32, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin
    ](rollout_actions_buf)
    var rollout_rewards = LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin
    ](rollout_rewards_buf)
    var rollout_dones = LayoutTensor[
        DType.int32, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin
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

    var grad_W1 = LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, OBS_DIM * HIDDEN_DIM), MutAnyOrigin
    ](grad_W1_buf)
    var grad_b1 = LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin
    ](grad_b1_buf)
    var grad_W_actor = LayoutTensor[
        dtype,
        Layout.row_major(NUM_ENVS, HIDDEN_DIM * NUM_ACTIONS),
        MutAnyOrigin,
    ](grad_W_actor_buf)
    var grad_b_actor = LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), MutAnyOrigin
    ](grad_b_actor_buf)
    var grad_W_critic = LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin
    ](grad_W_critic_buf)
    var grad_b_critic = LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin
    ](grad_b_critic_buf)

    var reduced_W1 = LayoutTensor[
        dtype, Layout.row_major(OBS_DIM * HIDDEN_DIM), MutAnyOrigin
    ](reduced_W1_buf)
    var W1_mut = LayoutTensor[
        dtype, Layout.row_major(OBS_DIM * HIDDEN_DIM), MutAnyOrigin
    ](W1_buf)

    # Scalars
    var gamma_s = Scalar[dtype](gamma)
    var gae_lambda_s = Scalar[dtype](gae_lambda)
    var entropy_coef_s = Scalar[dtype](entropy_coef)
    var value_coef_s = Scalar[dtype](value_coef)
    var lr_s = Scalar[dtype](lr)

    comptime env_blocks = BLOCKS_PER_GRID
    comptime env_threads = THREADS_PER_BLOCK

    # Reset environments
    ctx.enqueue_function_checked[reset_kernel, reset_kernel](
        states,
        rng_states,
        grid_dim=(env_blocks,),
        block_dim=(env_threads,),
    )
    ctx.synchronize()

    if verbose:
        print("=" * 60)
        print("A2C Training on GPU CartPole")
        print("=" * 60)
        print("  Environments:", NUM_ENVS)
        print("  Rollout length:", ROLLOUT_LEN)
        print("  Hidden dim:", HIDDEN_DIM)
        print("  Learning rate:", lr)
        print()

    var start_time = perf_counter_ns()
    var total_episodes: Int = 0

    # Training loop
    for update in range(num_updates):
        # Phase 1: Collect rollout
        for step in range(ROLLOUT_LEN):
            ctx.enqueue_function_checked[
                forward_sample_kernel, forward_sample_kernel
            ](
                obs,
                W1,
                b1,
                W_actor,
                b_actor,
                W_critic,
                b_critic,
                rng_states,
                actions,
                log_probs,
                values,
                grid_dim=(env_blocks,),
                block_dim=(env_threads,),
            )

            # Create immutable views
            var obs_i = LayoutTensor[
                dtype, Layout.row_major(NUM_ENVS, OBS_DIM), ImmutAnyOrigin
            ](states_buf)
            var actions_i = LayoutTensor[
                DType.int32, Layout.row_major(NUM_ENVS, 1), ImmutAnyOrigin
            ](actions_buf)
            var log_probs_i = LayoutTensor[
                dtype, Layout.row_major(NUM_ENVS, 1), ImmutAnyOrigin
            ](log_probs_buf)
            var values_i = LayoutTensor[
                dtype, Layout.row_major(NUM_ENVS, 1), ImmutAnyOrigin
            ](values_buf)

            ctx.enqueue_function_checked[step_kernel, step_kernel](
                states,
                actions,
                rewards,
                dones,
                grid_dim=(env_blocks,),
                block_dim=(env_threads,),
            )

            var rewards_i = LayoutTensor[
                dtype, Layout.row_major(NUM_ENVS, 1), ImmutAnyOrigin
            ](rewards_buf)
            var dones_i = LayoutTensor[
                DType.int32, Layout.row_major(NUM_ENVS, 1), ImmutAnyOrigin
            ](dones_buf)

            ctx.enqueue_function_checked[
                store_rollout_kernel, store_rollout_kernel
            ](
                step,
                obs_i,
                actions_i,
                rewards_i,
                dones_i,
                log_probs_i,
                values_i,
                rollout_obs,
                rollout_actions,
                rollout_rewards,
                rollout_dones,
                rollout_log_probs,
                rollout_values,
                grid_dim=(env_blocks,),
                block_dim=(env_threads,),
            )

            ctx.enqueue_function_checked[
                reset_where_done_kernel, reset_where_done_kernel
            ](
                states,
                dones,
                rng_states,
                grid_dim=(env_blocks,),
                block_dim=(env_threads,),
            )

        # Phase 2: Compute GAE
        ctx.enqueue_function_checked[get_values_kernel, get_values_kernel](
            obs,
            W1,
            b1,
            W_critic,
            b_critic,
            bootstrap_values,
            grid_dim=(env_blocks,),
            block_dim=(env_threads,),
        )

        var rollout_rewards_i = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), ImmutAnyOrigin
        ](rollout_rewards_buf)
        var rollout_dones_i = LayoutTensor[
            DType.int32, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), ImmutAnyOrigin
        ](rollout_dones_buf)
        var rollout_values_i = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), ImmutAnyOrigin
        ](rollout_values_buf)
        var bootstrap_values_i = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, 1), ImmutAnyOrigin
        ](bootstrap_values_buf)

        ctx.enqueue_function_checked[compute_gae_kernel, compute_gae_kernel](
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
            Layout.row_major(NUM_ENVS, ROLLOUT_LEN, OBS_DIM),
            ImmutAnyOrigin,
        ](rollout_obs_buf)
        var rollout_actions_i = LayoutTensor[
            DType.int32, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), ImmutAnyOrigin
        ](rollout_actions_buf)
        var rollout_advantages_i = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), ImmutAnyOrigin
        ](rollout_advantages_buf)
        var rollout_returns_i = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), ImmutAnyOrigin
        ](rollout_returns_buf)

        ctx.enqueue_function_checked[
            policy_gradient_kernel, policy_gradient_kernel
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

        # Phase 4: Reduce and update (simplified - only W1 for now)
        var grad_W1_i = LayoutTensor[
            dtype,
            Layout.row_major(NUM_ENVS, OBS_DIM * HIDDEN_DIM),
            ImmutAnyOrigin,
        ](grad_W1_buf)
        var reduced_W1_i = LayoutTensor[
            dtype, Layout.row_major(OBS_DIM * HIDDEN_DIM), ImmutAnyOrigin
        ](reduced_W1_buf)

        comptime blocks_W1 = (OBS_DIM * HIDDEN_DIM + TPB - 1) // TPB
        ctx.enqueue_function_checked[reduce_add_kernel, reduce_add_kernel](
            reduced_W1,
            grad_W1_i,
            grid_dim=(blocks_W1,),
            block_dim=(TPB,),
        )
        ctx.enqueue_function_checked[sgd_kernel, sgd_kernel](
            W1_mut,
            reduced_W1_i,
            lr_s,
            grid_dim=(blocks_W1,),
            block_dim=(TPB,),
        )

        # Logging
        if verbose and (update + 1) % 20 == 0:
            ctx.synchronize()
            var ep_count: Int = 0
            with rollout_dones_buf.map_to_host() as host:
                for i in range(NUM_ENVS * ROLLOUT_LEN):
                    if host[i] != 0:
                        ep_count += 1
            total_episodes += ep_count
            var steps = (update + 1) * NUM_ENVS * ROLLOUT_LEN
            var avg_len = (
                Float64(steps) / Float64(total_episodes) if total_episodes
                > 0 else 0.0
            )
            print(
                "Update",
                update + 1,
                "| Steps:",
                steps,
                "| Episodes:",
                total_episodes,
                "| Avg len:",
                Int(avg_len),
            )

    ctx.synchronize()
    var end_time = perf_counter_ns()

    var total_steps = num_updates * NUM_ENVS * ROLLOUT_LEN
    var elapsed = Float64(end_time - start_time) / 1e9
    var throughput = Float64(total_steps) / elapsed

    if verbose:
        print()
        print("Training complete!")
        print("  Total steps:", total_steps)
        print("  Time:", elapsed, "seconds")
        print("  Throughput:", Int(throughput), "steps/sec")

    return throughput

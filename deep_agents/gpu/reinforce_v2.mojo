"""Generic GPU REINFORCE Algorithm v2 - Practical Composable Design.

This version avoids compile-time explosion by:
1. Taking dimensions as explicit compile-time Int parameters (not extracted from trait)
2. Taking environment step/reset functions as fn parameters
3. Using structural composition rather than trait polymorphism

Usage:
    from deep_rl.gpu.envs.cartpole import (
        cartpole_step, cartpole_reset, cartpole_get_obs,
        CARTPOLE_OBS_DIM, CARTPOLE_NUM_ACTIONS, CARTPOLE_STATE_SIZE
    )

    with DeviceContext() as ctx:
        train_reinforce[
            OBS_DIM = CARTPOLE_OBS_DIM,
            NUM_ACTIONS = CARTPOLE_NUM_ACTIONS,
            STATE_SIZE = CARTPOLE_STATE_SIZE,
            step_fn = cartpole_step,
            reset_fn = cartpole_reset,
            get_obs_fn = cartpole_get_obs,
        ](ctx, num_updates=100, lr=0.01, gamma=0.99)
"""

from time import perf_counter_ns
from math import exp, log, sqrt
from random import random_float64

from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from ..env_trait import xorshift32, gpu_random_uniform


# =============================================================================
# Result struct
# =============================================================================


@fieldwise_init
struct TrainResultV2(Copyable, Movable):
    """Result of REINFORCE training."""

    var total_steps: Int
    var total_episodes: Int
    var elapsed_seconds: Float64
    var steps_per_sec: Float64
    var final_avg_ep_length: Float64


# =============================================================================
# REINFORCE Training Kernel with explicit dimensions
# =============================================================================


fn reinforce_kernel_v2[
    dtype: DType,
    OBS_DIM: Int,
    NUM_ACTIONS: Int,
    STATE_SIZE: Int,
    HIDDEN_DIM: Int,
    NUM_ENVS: Int,
    STEPS_PER_KERNEL: Int,
    TPB: Int,
    # Environment functions as compile-time parameters
    step_fn: fn (
        mut state: InlineArray[Scalar[dtype], STATE_SIZE],
        action: Int,
        rng: Scalar[DType.uint32],
    ) capturing -> Tuple[Scalar[dtype], Bool, Scalar[DType.uint32]],
    reset_fn: fn (
        mut state: InlineArray[Scalar[dtype], STATE_SIZE],
        rng: Scalar[DType.uint32],
    ) capturing -> Scalar[DType.uint32],
    get_obs_fn: fn (
        state: InlineArray[Scalar[dtype], STATE_SIZE],
    ) capturing -> InlineArray[Scalar[dtype], OBS_DIM],
](
    # Environment state
    env_state: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS * STATE_SIZE), MutAnyOrigin
    ],
    rng_state: LayoutTensor[
        DType.uint32, Layout.row_major(NUM_ENVS), MutAnyOrigin
    ],
    episode_lengths: LayoutTensor[
        DType.int32, Layout.row_major(NUM_ENVS), MutAnyOrigin
    ],
    total_episodes: LayoutTensor[
        DType.int32, Layout.row_major(NUM_ENVS), MutAnyOrigin
    ],
    # Policy weights (read-only during forward pass)
    W1: LayoutTensor[
        dtype, Layout.row_major(OBS_DIM * HIDDEN_DIM), ImmutAnyOrigin
    ],
    b1: LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin],
    W2: LayoutTensor[
        dtype, Layout.row_major(HIDDEN_DIM * NUM_ACTIONS), ImmutAnyOrigin
    ],
    b2: LayoutTensor[dtype, Layout.row_major(NUM_ACTIONS), ImmutAnyOrigin],
    # Gradient accumulators
    grad_W1: LayoutTensor[
        dtype,
        Layout.row_major(NUM_ENVS * OBS_DIM * HIDDEN_DIM),
        MutAnyOrigin,
    ],
    grad_b1: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS * HIDDEN_DIM), MutAnyOrigin
    ],
    grad_W2: LayoutTensor[
        dtype,
        Layout.row_major(NUM_ENVS * HIDDEN_DIM * NUM_ACTIONS),
        MutAnyOrigin,
    ],
    grad_b2: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS * NUM_ACTIONS), MutAnyOrigin
    ],
    # Hyperparameters
    gamma: Scalar[dtype],
):
    """REINFORCE training kernel - runs STEPS_PER_KERNEL steps per env.

    Each GPU thread:
    1. Runs one environment instance
    2. Collects trajectory (obs, action, reward) for STEPS_PER_KERNEL steps
    3. Computes returns and policy gradients
    4. Accumulates gradients for later reduction
    """
    var env_idx = Int(block_dim.x * block_idx.x + thread_idx.x)

    if env_idx >= NUM_ENVS:
        return

    # Load environment state into register-based InlineArray
    var state = InlineArray[Scalar[dtype], STATE_SIZE](
        fill=Scalar[dtype](0)
    )
    for i in range(STATE_SIZE):
        state[i] = rebind[Scalar[dtype]](
            env_state[env_idx * STATE_SIZE + i]
        )

    var rng = rebind[Scalar[DType.uint32]](rng_state[env_idx])
    var ep_length = Int(episode_lengths[env_idx])
    var num_episodes = Int(total_episodes[env_idx])

    # Load policy weights into local arrays
    var w1 = InlineArray[Scalar[dtype], OBS_DIM * HIDDEN_DIM](
        fill=Scalar[dtype](0)
    )
    var bias1 = InlineArray[Scalar[dtype], HIDDEN_DIM](fill=Scalar[dtype](0))
    var w2 = InlineArray[Scalar[dtype], HIDDEN_DIM * NUM_ACTIONS](
        fill=Scalar[dtype](0)
    )
    var bias2 = InlineArray[Scalar[dtype], NUM_ACTIONS](
        fill=Scalar[dtype](0)
    )

    for i in range(OBS_DIM * HIDDEN_DIM):
        w1[i] = rebind[Scalar[dtype]](W1[i])
    for i in range(HIDDEN_DIM):
        bias1[i] = rebind[Scalar[dtype]](b1[i])
    for i in range(HIDDEN_DIM * NUM_ACTIONS):
        w2[i] = rebind[Scalar[dtype]](W2[i])
    for i in range(NUM_ACTIONS):
        bias2[i] = rebind[Scalar[dtype]](b2[i])

    # Trajectory storage
    var traj_obs = InlineArray[Scalar[dtype], STEPS_PER_KERNEL * OBS_DIM](
        fill=Scalar[dtype](0)
    )
    var traj_actions = InlineArray[Int, STEPS_PER_KERNEL](fill=0)
    var traj_rewards = InlineArray[Scalar[dtype], STEPS_PER_KERNEL](
        fill=Scalar[dtype](0)
    )
    var traj_dones = InlineArray[Bool, STEPS_PER_KERNEL](fill=False)
    var traj_length = 0

    # =================================================================
    # Run STEPS_PER_KERNEL environment steps
    # =================================================================

    for step in range(STEPS_PER_KERNEL):
        # Get observation from state using the env's get_obs function
        var obs = get_obs_fn(state)

        # Store observation in trajectory
        for i in range(OBS_DIM):
            traj_obs[step * OBS_DIM + i] = obs[i]

        # =============================================================
        # Forward pass: obs -> hidden -> action_probs
        # =============================================================

        # Hidden layer: h = ReLU(obs @ W1 + b1)
        var h = InlineArray[Scalar[dtype], HIDDEN_DIM](fill=Scalar[dtype](0))
        for j in range(HIDDEN_DIM):
            var sum_val = bias1[j]
            for i in range(OBS_DIM):
                sum_val += obs[i] * w1[i * HIDDEN_DIM + j]
            h[j] = sum_val if sum_val > Scalar[dtype](0) else Scalar[dtype](0)

        # Output layer: logits = h @ W2 + b2
        var logits = InlineArray[Scalar[dtype], NUM_ACTIONS](
            fill=Scalar[dtype](0)
        )
        for j in range(NUM_ACTIONS):
            var sum_val = bias2[j]
            for k in range(HIDDEN_DIM):
                sum_val += h[k] * w2[k * NUM_ACTIONS + j]
            logits[j] = sum_val

        # Softmax with numerical stability
        var max_logit = logits[0]
        for j in range(1, NUM_ACTIONS):
            if logits[j] > max_logit:
                max_logit = logits[j]

        var exp_logits = InlineArray[Scalar[dtype], NUM_ACTIONS](
            fill=Scalar[dtype](0)
        )
        var sum_exp: Scalar[dtype] = 0
        for j in range(NUM_ACTIONS):
            exp_logits[j] = exp(logits[j] - max_logit)
            sum_exp += exp_logits[j]

        var probs = InlineArray[Scalar[dtype], NUM_ACTIONS](
            fill=Scalar[dtype](0)
        )
        for j in range(NUM_ACTIONS):
            probs[j] = exp_logits[j] / sum_exp

        # =============================================================
        # Sample action from policy
        # =============================================================

        var u_result = gpu_random_uniform[dtype](rng)
        var u = u_result[0]
        rng = u_result[1]

        var action = 0
        var cumsum: Scalar[dtype] = 0
        for j in range(NUM_ACTIONS):
            cumsum += probs[j]
            if u < cumsum:
                action = j
                break
            action = j  # Last action if loop completes

        traj_actions[step] = action

        # =============================================================
        # Environment step using the env's step function
        # =============================================================

        var step_result = step_fn(state, action, rng)
        var reward = step_result[0]
        var done = step_result[1]
        rng = step_result[2]

        traj_rewards[step] = reward
        traj_dones[step] = done
        ep_length += 1
        traj_length = step + 1

        # =============================================================
        # Reset if done using the env's reset function
        # =============================================================

        if done:
            num_episodes += 1
            rng = reset_fn(state, rng)
            ep_length = 0

    # =================================================================
    # Compute returns and policy gradients (REINFORCE)
    # =================================================================

    # Compute discounted returns (backwards)
    var returns = InlineArray[Scalar[dtype], STEPS_PER_KERNEL](
        fill=Scalar[dtype](0)
    )
    var G: Scalar[dtype] = 0

    for step in range(traj_length - 1, -1, -1):
        if traj_dones[step]:
            G = Scalar[dtype](0)
        G = traj_rewards[step] + gamma * G
        returns[step] = G

    # Initialize local gradient accumulators
    var local_grad_W1 = InlineArray[Scalar[dtype], OBS_DIM * HIDDEN_DIM](
        fill=Scalar[dtype](0)
    )
    var local_grad_b1 = InlineArray[Scalar[dtype], HIDDEN_DIM](
        fill=Scalar[dtype](0)
    )
    var local_grad_W2 = InlineArray[
        Scalar[dtype], HIDDEN_DIM * NUM_ACTIONS
    ](fill=Scalar[dtype](0))
    var local_grad_b2 = InlineArray[Scalar[dtype], NUM_ACTIONS](
        fill=Scalar[dtype](0)
    )

    # Compute gradients for each step
    for step in range(traj_length):
        var ret = returns[step]
        var action = traj_actions[step]

        # Reload observation
        var obs = InlineArray[Scalar[dtype], OBS_DIM](fill=Scalar[dtype](0))
        for i in range(OBS_DIM):
            obs[i] = traj_obs[step * OBS_DIM + i]

        # Recompute forward pass
        var h = InlineArray[Scalar[dtype], HIDDEN_DIM](fill=Scalar[dtype](0))
        var h_pre = InlineArray[Scalar[dtype], HIDDEN_DIM](
            fill=Scalar[dtype](0)
        )
        for j in range(HIDDEN_DIM):
            var sum_val = bias1[j]
            for i in range(OBS_DIM):
                sum_val += obs[i] * w1[i * HIDDEN_DIM + j]
            h_pre[j] = sum_val
            h[j] = sum_val if sum_val > Scalar[dtype](0) else Scalar[dtype](0)

        var logits = InlineArray[Scalar[dtype], NUM_ACTIONS](
            fill=Scalar[dtype](0)
        )
        for j in range(NUM_ACTIONS):
            var sum_val = bias2[j]
            for k in range(HIDDEN_DIM):
                sum_val += h[k] * w2[k * NUM_ACTIONS + j]
            logits[j] = sum_val

        # Softmax
        var max_logit = logits[0]
        for j in range(1, NUM_ACTIONS):
            if logits[j] > max_logit:
                max_logit = logits[j]

        var exp_logits = InlineArray[Scalar[dtype], NUM_ACTIONS](
            fill=Scalar[dtype](0)
        )
        var sum_exp: Scalar[dtype] = 0
        for j in range(NUM_ACTIONS):
            exp_logits[j] = exp(logits[j] - max_logit)
            sum_exp += exp_logits[j]

        var probs = InlineArray[Scalar[dtype], NUM_ACTIONS](
            fill=Scalar[dtype](0)
        )
        for j in range(NUM_ACTIONS):
            probs[j] = exp_logits[j] / sum_exp

        # Gradient of log softmax: d/d_logit[a] log(softmax[action]) = 1[a == action] - softmax[a]
        var d_logits = InlineArray[Scalar[dtype], NUM_ACTIONS](
            fill=Scalar[dtype](0)
        )
        for j in range(NUM_ACTIONS):
            d_logits[j] = (
                Scalar[dtype](1.0) if j == action else Scalar[dtype](0.0)
            ) - probs[j]
            d_logits[j] = d_logits[j] * ret  # Scale by return

        # Gradient for W2 and b2
        for k in range(HIDDEN_DIM):
            for j in range(NUM_ACTIONS):
                local_grad_W2[k * NUM_ACTIONS + j] += h[k] * d_logits[j]

        for j in range(NUM_ACTIONS):
            local_grad_b2[j] += d_logits[j]

        # Backprop through hidden layer
        for k in range(HIDDEN_DIM):
            var dh: Scalar[dtype] = 0
            for j in range(NUM_ACTIONS):
                dh += d_logits[j] * w2[k * NUM_ACTIONS + j]
            var dh_pre = dh if h_pre[k] > Scalar[dtype](0) else Scalar[dtype](0)

            for i in range(OBS_DIM):
                local_grad_W1[i * HIDDEN_DIM + k] += obs[i] * dh_pre

            local_grad_b1[k] += dh_pre

    # =================================================================
    # Write back state and gradients
    # =================================================================

    for i in range(STATE_SIZE):
        env_state[env_idx * STATE_SIZE + i] = state[i]

    rng_state[env_idx] = rng
    episode_lengths[env_idx] = Int32(ep_length)
    total_episodes[env_idx] = Int32(num_episodes)

    # Per-env gradients
    for i in range(OBS_DIM * HIDDEN_DIM):
        grad_W1[env_idx * OBS_DIM * HIDDEN_DIM + i] = local_grad_W1[i]
    for i in range(HIDDEN_DIM):
        grad_b1[env_idx * HIDDEN_DIM + i] = local_grad_b1[i]
    for i in range(HIDDEN_DIM * NUM_ACTIONS):
        grad_W2[env_idx * HIDDEN_DIM * NUM_ACTIONS + i] = local_grad_W2[i]
    for i in range(NUM_ACTIONS):
        grad_b2[env_idx * NUM_ACTIONS + i] = local_grad_b2[i]


# =============================================================================
# Gradient Reduction Kernel
# =============================================================================


fn reduce_gradients_kernel[
    dtype: DType,
    NUM_ENVS: Int,
    SIZE: Int,
    TPB: Int,
](
    reduced: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    per_env: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS * SIZE), ImmutAnyOrigin
    ],
):
    """Sum gradients across environments and average."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)

    if idx >= SIZE:
        return

    var sum_val: Scalar[dtype] = 0
    for env in range(NUM_ENVS):
        sum_val += rebind[Scalar[dtype]](per_env[env * SIZE + idx])

    reduced[idx] = sum_val / Scalar[dtype](NUM_ENVS)


# =============================================================================
# SGD Update Kernel
# =============================================================================


fn sgd_update_kernel[
    dtype: DType,
    SIZE: Int,
    TPB: Int,
](
    weights: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    gradients: LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin],
    lr: Scalar[dtype],
):
    """SGD weight update (gradient ascent for policy optimization)."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)

    if idx >= SIZE:
        return

    var w = rebind[Scalar[dtype]](weights[idx])
    var g = rebind[Scalar[dtype]](gradients[idx])
    weights[idx] = w + lr * g

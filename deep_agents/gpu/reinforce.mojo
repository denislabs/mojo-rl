"""Generic GPU REINFORCE Algorithm.

This module provides a composable REINFORCE implementation that works with
any environment implementing the GPUEnv trait.

Key design:
1. Training kernel is parameterized by Env type (compile-time polymorphism)
2. Multi-step kernel pattern: many steps per kernel launch
3. Policy network: simple 2-layer MLP with softmax output
4. All computation on GPU with minimal CPU interaction

Usage:
    from deep_rl.gpu.envs.cartpole import GPUCartPole
    from deep_rl.gpu.algorithms.reinforce import train_reinforce

    with DeviceContext() as ctx:
        train_reinforce[GPUCartPole, HIDDEN_DIM=32, NUM_ENVS=1024](
            ctx, num_updates=100, lr=0.01, gamma=0.99
        )
"""

from time import perf_counter_ns
from math import exp, log, sqrt
from random import random_float64

from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from deep_rl.gpu import xorshift32, gpu_random_uniform
from core import GPUEnvDims


# =============================================================================
# REINFORCE Training Kernel (Generic over Env)
# =============================================================================


fn reinforce_kernel[
    dtype: DType,
    Env: GPUEnvDims,
    HIDDEN_DIM: Int,
    NUM_ENVS: Int,
    STEPS_PER_KERNEL: Int,
    TPB: Int,
](
    # Environment state
    env_state: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS * Env.STATE_SIZE), MutAnyOrigin
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
        dtype, Layout.row_major(Env.OBS_DIM * HIDDEN_DIM), ImmutAnyOrigin
    ],
    b1: LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin],
    W2: LayoutTensor[
        dtype, Layout.row_major(HIDDEN_DIM * Env.NUM_ACTIONS), ImmutAnyOrigin
    ],
    b2: LayoutTensor[dtype, Layout.row_major(Env.NUM_ACTIONS), ImmutAnyOrigin],
    # Gradient accumulators
    grad_W1: LayoutTensor[
        dtype,
        Layout.row_major(NUM_ENVS * Env.OBS_DIM * HIDDEN_DIM),
        MutAnyOrigin,
    ],
    grad_b1: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS * HIDDEN_DIM), MutAnyOrigin
    ],
    grad_W2: LayoutTensor[
        dtype,
        Layout.row_major(NUM_ENVS * HIDDEN_DIM * Env.NUM_ACTIONS),
        MutAnyOrigin,
    ],
    grad_b2: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS * Env.NUM_ACTIONS), MutAnyOrigin
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
    var state = InlineArray[Scalar[dtype], Env.STATE_SIZE](
        fill=Scalar[dtype](0)
    )
    for i in range(Env.STATE_SIZE):
        state[i] = rebind[Scalar[dtype]](
            env_state[env_idx * Env.STATE_SIZE + i]
        )

    var rng = rebind[Scalar[DType.uint32]](rng_state[env_idx])
    var ep_length = Int(episode_lengths[env_idx])
    var num_episodes = Int(total_episodes[env_idx])

    # Load policy weights into local arrays
    var w1 = InlineArray[Scalar[dtype], Env.OBS_DIM * HIDDEN_DIM](
        fill=Scalar[dtype](0)
    )
    var bias1 = InlineArray[Scalar[dtype], HIDDEN_DIM](fill=Scalar[dtype](0))
    var w2 = InlineArray[Scalar[dtype], HIDDEN_DIM * Env.NUM_ACTIONS](
        fill=Scalar[dtype](0)
    )
    var bias2 = InlineArray[Scalar[dtype], Env.NUM_ACTIONS](
        fill=Scalar[dtype](0)
    )

    for i in range(Env.OBS_DIM * HIDDEN_DIM):
        w1[i] = rebind[Scalar[dtype]](W1[i])
    for i in range(HIDDEN_DIM):
        bias1[i] = rebind[Scalar[dtype]](b1[i])
    for i in range(HIDDEN_DIM * Env.NUM_ACTIONS):
        w2[i] = rebind[Scalar[dtype]](W2[i])
    for i in range(Env.NUM_ACTIONS):
        bias2[i] = rebind[Scalar[dtype]](b2[i])

    # Trajectory storage
    var traj_obs = InlineArray[Scalar[dtype], STEPS_PER_KERNEL * Env.OBS_DIM](
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
        # Get observation from state
        var obs = Env.get_obs[dtype](state)

        # Store observation in trajectory
        for i in range(Env.OBS_DIM):
            traj_obs[step * Env.OBS_DIM + i] = obs[i]

        # =============================================================
        # Forward pass: obs -> hidden -> action_probs
        # =============================================================

        # Hidden layer: h = ReLU(obs @ W1 + b1)
        var h = InlineArray[Scalar[dtype], HIDDEN_DIM](fill=Scalar[dtype](0))
        for j in range(HIDDEN_DIM):
            var sum_val = bias1[j]
            for i in range(Env.OBS_DIM):
                sum_val += obs[i] * w1[i * HIDDEN_DIM + j]
            h[j] = sum_val if sum_val > Scalar[dtype](0) else Scalar[dtype](0)

        # Output layer: logits = h @ W2 + b2
        var logits = InlineArray[Scalar[dtype], Env.NUM_ACTIONS](
            fill=Scalar[dtype](0)
        )
        for j in range(Env.NUM_ACTIONS):
            var sum_val = bias2[j]
            for k in range(HIDDEN_DIM):
                sum_val += h[k] * w2[k * Env.NUM_ACTIONS + j]
            logits[j] = sum_val

        # Softmax with numerical stability
        var max_logit = logits[0]
        for j in range(1, Env.NUM_ACTIONS):
            if logits[j] > max_logit:
                max_logit = logits[j]

        var exp_logits = InlineArray[Scalar[dtype], Env.NUM_ACTIONS](
            fill=Scalar[dtype](0)
        )
        var sum_exp: Scalar[dtype] = 0
        for j in range(Env.NUM_ACTIONS):
            exp_logits[j] = exp(logits[j] - max_logit)
            sum_exp += exp_logits[j]

        var probs = InlineArray[Scalar[dtype], Env.NUM_ACTIONS](
            fill=Scalar[dtype](0)
        )
        for j in range(Env.NUM_ACTIONS):
            probs[j] = exp_logits[j] / sum_exp

        # =============================================================
        # Sample action from policy
        # =============================================================

        var u_result = gpu_random_uniform[dtype](rng)
        var u = u_result[0]
        rng = u_result[1]

        var action = 0
        var cumsum: Scalar[dtype] = 0
        for j in range(Env.NUM_ACTIONS):
            cumsum += probs[j]
            if u < cumsum:
                action = j
                break
            action = j  # Last action if loop completes

        traj_actions[step] = action

        # =============================================================
        # Environment step (using trait method)
        # =============================================================

        var step_result = Env.step[dtype](state, action, rng)
        var reward = step_result[0]
        var done = step_result[1]
        rng = step_result[2]

        traj_rewards[step] = reward
        traj_dones[step] = done
        ep_length += 1
        traj_length = step + 1

        # =============================================================
        # Reset if done
        # =============================================================

        if done:
            num_episodes += 1
            rng = Env.reset[dtype](state, rng)
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
    var local_grad_W1 = InlineArray[Scalar[dtype], Env.OBS_DIM * HIDDEN_DIM](
        fill=Scalar[dtype](0)
    )
    var local_grad_b1 = InlineArray[Scalar[dtype], HIDDEN_DIM](
        fill=Scalar[dtype](0)
    )
    var local_grad_W2 = InlineArray[
        Scalar[dtype], HIDDEN_DIM * Env.NUM_ACTIONS
    ](fill=Scalar[dtype](0))
    var local_grad_b2 = InlineArray[Scalar[dtype], Env.NUM_ACTIONS](
        fill=Scalar[dtype](0)
    )

    # Compute gradients for each step
    for step in range(traj_length):
        var ret = returns[step]
        var action = traj_actions[step]

        # Reload observation
        var obs = InlineArray[Scalar[dtype], Env.OBS_DIM](fill=Scalar[dtype](0))
        for i in range(Env.OBS_DIM):
            obs[i] = traj_obs[step * Env.OBS_DIM + i]

        # Recompute forward pass
        var h = InlineArray[Scalar[dtype], HIDDEN_DIM](fill=Scalar[dtype](0))
        var h_pre = InlineArray[Scalar[dtype], HIDDEN_DIM](
            fill=Scalar[dtype](0)
        )
        for j in range(HIDDEN_DIM):
            var sum_val = bias1[j]
            for i in range(Env.OBS_DIM):
                sum_val += obs[i] * w1[i * HIDDEN_DIM + j]
            h_pre[j] = sum_val
            h[j] = sum_val if sum_val > Scalar[dtype](0) else Scalar[dtype](0)

        var logits = InlineArray[Scalar[dtype], Env.NUM_ACTIONS](
            fill=Scalar[dtype](0)
        )
        for j in range(Env.NUM_ACTIONS):
            var sum_val = bias2[j]
            for k in range(HIDDEN_DIM):
                sum_val += h[k] * w2[k * Env.NUM_ACTIONS + j]
            logits[j] = sum_val

        # Softmax
        var max_logit = logits[0]
        for j in range(1, Env.NUM_ACTIONS):
            if logits[j] > max_logit:
                max_logit = logits[j]

        var exp_logits = InlineArray[Scalar[dtype], Env.NUM_ACTIONS](
            fill=Scalar[dtype](0)
        )
        var sum_exp: Scalar[dtype] = 0
        for j in range(Env.NUM_ACTIONS):
            exp_logits[j] = exp(logits[j] - max_logit)
            sum_exp += exp_logits[j]

        var probs = InlineArray[Scalar[dtype], Env.NUM_ACTIONS](
            fill=Scalar[dtype](0)
        )
        for j in range(Env.NUM_ACTIONS):
            probs[j] = exp_logits[j] / sum_exp

        # Gradient of log softmax: d/d_logit[a] log(softmax[action]) = 1[a == action] - softmax[a]
        var d_logits = InlineArray[Scalar[dtype], Env.NUM_ACTIONS](
            fill=Scalar[dtype](0)
        )
        for j in range(Env.NUM_ACTIONS):
            d_logits[j] = (
                Scalar[dtype](1.0) if j == action else Scalar[dtype](0.0)
            ) - probs[j]
            d_logits[j] = d_logits[j] * ret  # Scale by return

        # Gradient for W2 and b2
        for k in range(HIDDEN_DIM):
            for j in range(Env.NUM_ACTIONS):
                local_grad_W2[k * Env.NUM_ACTIONS + j] += h[k] * d_logits[j]

        for j in range(Env.NUM_ACTIONS):
            local_grad_b2[j] += d_logits[j]

        # Backprop through hidden layer
        for k in range(HIDDEN_DIM):
            var dh: Scalar[dtype] = 0
            for j in range(Env.NUM_ACTIONS):
                dh += d_logits[j] * w2[k * Env.NUM_ACTIONS + j]
            var dh_pre = dh if h_pre[k] > Scalar[dtype](0) else Scalar[dtype](0)

            for i in range(Env.OBS_DIM):
                local_grad_W1[i * HIDDEN_DIM + k] += obs[i] * dh_pre

            local_grad_b1[k] += dh_pre

    # =================================================================
    # Write back state and gradients
    # =================================================================

    for i in range(Env.STATE_SIZE):
        env_state[env_idx * Env.STATE_SIZE + i] = state[i]

    rng_state[env_idx] = rng
    episode_lengths[env_idx] = Int32(ep_length)
    total_episodes[env_idx] = Int32(num_episodes)

    # Per-env gradients
    for i in range(Env.OBS_DIM * HIDDEN_DIM):
        grad_W1[env_idx * Env.OBS_DIM * HIDDEN_DIM + i] = local_grad_W1[i]
    for i in range(HIDDEN_DIM):
        grad_b1[env_idx * HIDDEN_DIM + i] = local_grad_b1[i]
    for i in range(HIDDEN_DIM * Env.NUM_ACTIONS):
        grad_W2[env_idx * HIDDEN_DIM * Env.NUM_ACTIONS + i] = local_grad_W2[i]
    for i in range(Env.NUM_ACTIONS):
        grad_b2[env_idx * Env.NUM_ACTIONS + i] = local_grad_b2[i]


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


# =============================================================================
# Training Result
# =============================================================================


@fieldwise_init
struct TrainResult(Copyable, Movable):
    """Result of REINFORCE training."""

    var total_steps: Int
    var total_episodes: Int
    var elapsed_seconds: Float64
    var steps_per_sec: Float64
    var final_avg_ep_length: Float64


# =============================================================================
# Main Training Function (Generic over Env)
# =============================================================================


fn train_reinforce[
    Env: GPUEnvDims,  # Structural typing - assumes Env has OBS_DIM, NUM_ACTIONS, STATE_SIZE
    HIDDEN_DIM: Int = 32,
    NUM_ENVS: Int = 1024,
    STEPS_PER_KERNEL: Int = 200,
    TPB: Int = 256,
](
    ctx: DeviceContext,
    num_updates: Int,
    lr: Float32,
    gamma: Float32,
    verbose: Bool = True,
) raises -> TrainResult:
    """Train a policy with REINFORCE on any GPUEnv.

    Args:
        ctx: GPU device context
        num_updates: Number of training updates (each runs STEPS_PER_KERNEL steps)
        lr: Learning rate for SGD
        gamma: Discount factor
        verbose: Print training progress

    Returns:
        TrainResult with training statistics
    """
    comptime dtype = DType.float32

    # Compute sizes
    comptime OBS_DIM = Env.OBS_DIM
    comptime NUM_ACTIONS = Env.NUM_ACTIONS
    comptime STATE_SIZE = Env.STATE_SIZE

    comptime W1_SIZE = OBS_DIM * HIDDEN_DIM
    comptime W2_SIZE = HIDDEN_DIM * NUM_ACTIONS

    # Allocate environment state buffers
    var env_state = ctx.enqueue_create_buffer[dtype](NUM_ENVS * STATE_SIZE)
    var rng_state = ctx.enqueue_create_buffer[DType.uint32](NUM_ENVS)
    var episode_lengths = ctx.enqueue_create_buffer[DType.int32](NUM_ENVS)
    var total_episodes_buf = ctx.enqueue_create_buffer[DType.int32](NUM_ENVS)

    # Allocate policy weights
    var W1 = ctx.enqueue_create_buffer[dtype](W1_SIZE)
    var b1 = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
    var W2 = ctx.enqueue_create_buffer[dtype](W2_SIZE)
    var b2 = ctx.enqueue_create_buffer[dtype](NUM_ACTIONS)

    # Allocate per-env gradient accumulators
    var grad_W1_per_env = ctx.enqueue_create_buffer[dtype](NUM_ENVS * W1_SIZE)
    var grad_b1_per_env = ctx.enqueue_create_buffer[dtype](
        NUM_ENVS * HIDDEN_DIM
    )
    var grad_W2_per_env = ctx.enqueue_create_buffer[dtype](NUM_ENVS * W2_SIZE)
    var grad_b2_per_env = ctx.enqueue_create_buffer[dtype](
        NUM_ENVS * NUM_ACTIONS
    )

    # Allocate reduced gradients
    var grad_W1 = ctx.enqueue_create_buffer[dtype](W1_SIZE)
    var grad_b1 = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
    var grad_W2 = ctx.enqueue_create_buffer[dtype](W2_SIZE)
    var grad_b2 = ctx.enqueue_create_buffer[dtype](NUM_ACTIONS)

    # Initialize environment state (random)
    with env_state.map_to_host() as host:
        for i in range(NUM_ENVS * STATE_SIZE):
            host[i] = Scalar[dtype]((random_float64() - 0.5) * 0.1)

    with rng_state.map_to_host() as host:
        for i in range(NUM_ENVS):
            host[i] = UInt32(i + 12345)

    episode_lengths.enqueue_fill(0)
    total_episodes_buf.enqueue_fill(0)

    # Initialize weights with Xavier
    var std1 = sqrt(2.0 / Float64(OBS_DIM + HIDDEN_DIM))
    var std2 = sqrt(2.0 / Float64(HIDDEN_DIM + NUM_ACTIONS))

    with W1.map_to_host() as host:
        for i in range(W1_SIZE):
            host[i] = Scalar[dtype]((random_float64() - 0.5) * 2 * std1)
    b1.enqueue_fill(0)
    with W2.map_to_host() as host:
        for i in range(W2_SIZE):
            host[i] = Scalar[dtype]((random_float64() - 0.5) * 2 * std2)
    b2.enqueue_fill(0)

    ctx.synchronize()

    # Create tensors
    var env_t = LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS * STATE_SIZE), MutAnyOrigin
    ](env_state)
    var rng_t = LayoutTensor[
        DType.uint32, Layout.row_major(NUM_ENVS), MutAnyOrigin
    ](rng_state)
    var ep_len_t = LayoutTensor[
        DType.int32, Layout.row_major(NUM_ENVS), MutAnyOrigin
    ](episode_lengths)
    var tot_ep_t = LayoutTensor[
        DType.int32, Layout.row_major(NUM_ENVS), MutAnyOrigin
    ](total_episodes_buf)

    var W1_t = LayoutTensor[dtype, Layout.row_major(W1_SIZE), ImmutAnyOrigin](
        W1
    )
    var b1_t = LayoutTensor[
        dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin
    ](b1)
    var W2_t = LayoutTensor[dtype, Layout.row_major(W2_SIZE), ImmutAnyOrigin](
        W2
    )
    var b2_t = LayoutTensor[
        dtype, Layout.row_major(NUM_ACTIONS), ImmutAnyOrigin
    ](b2)

    var gW1_env_t = LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS * W1_SIZE), MutAnyOrigin
    ](grad_W1_per_env)
    var gb1_env_t = LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS * HIDDEN_DIM), MutAnyOrigin
    ](grad_b1_per_env)
    var gW2_env_t = LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS * W2_SIZE), MutAnyOrigin
    ](grad_W2_per_env)
    var gb2_env_t = LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS * NUM_ACTIONS), MutAnyOrigin
    ](grad_b2_per_env)

    var gW1_t = LayoutTensor[dtype, Layout.row_major(W1_SIZE), MutAnyOrigin](
        grad_W1
    )
    var gb1_t = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), MutAnyOrigin](
        grad_b1
    )
    var gW2_t = LayoutTensor[dtype, Layout.row_major(W2_SIZE), MutAnyOrigin](
        grad_W2
    )
    var gb2_t = LayoutTensor[
        dtype, Layout.row_major(NUM_ACTIONS), MutAnyOrigin
    ](grad_b2)

    var W1_mut = LayoutTensor[dtype, Layout.row_major(W1_SIZE), MutAnyOrigin](
        W1
    )
    var b1_mut = LayoutTensor[
        dtype, Layout.row_major(HIDDEN_DIM), MutAnyOrigin
    ](b1)
    var W2_mut = LayoutTensor[dtype, Layout.row_major(W2_SIZE), MutAnyOrigin](
        W2
    )
    var b2_mut = LayoutTensor[
        dtype, Layout.row_major(NUM_ACTIONS), MutAnyOrigin
    ](b2)

    var gamma_s = Scalar[dtype](gamma)
    var lr_s = Scalar[dtype](lr)

    comptime num_blocks = (NUM_ENVS + TPB - 1) // TPB
    comptime main_kernel = reinforce_kernel[
        dtype, Env, HIDDEN_DIM, NUM_ENVS, STEPS_PER_KERNEL, TPB
    ]

    comptime blocks_W1 = (W1_SIZE + TPB - 1) // TPB
    comptime blocks_b1 = (HIDDEN_DIM + TPB - 1) // TPB
    comptime blocks_W2 = (W2_SIZE + TPB - 1) // TPB
    comptime blocks_b2 = (NUM_ACTIONS + TPB - 1) // TPB

    if verbose:
        print("Training REINFORCE on GPU")
        print(
            "  Environment: "
            + String(Env.OBS_DIM)
            + "D obs, "
            + String(Env.NUM_ACTIONS)
            + " actions"
        )
        print("  Parallel envs: " + String(NUM_ENVS))
        print("  Steps per update: " + String(STEPS_PER_KERNEL))
        print("  Hidden dim: " + String(HIDDEN_DIM))
        print("  Learning rate: " + String(lr))
        print("  Gamma: " + String(gamma))
        print()

    var start_time = perf_counter_ns()
    var final_avg_ep_len: Float64 = 0

    for update in range(num_updates):
        # Zero gradients
        grad_W1_per_env.enqueue_fill(0)
        grad_b1_per_env.enqueue_fill(0)
        grad_W2_per_env.enqueue_fill(0)
        grad_b2_per_env.enqueue_fill(0)

        # Run main kernel
        ctx.enqueue_function_checked[main_kernel, main_kernel](
            env_t,
            rng_t,
            ep_len_t,
            tot_ep_t,
            W1_t,
            b1_t,
            W2_t,
            b2_t,
            gW1_env_t,
            gb1_env_t,
            gW2_env_t,
            gb2_env_t,
            gamma_s,
            grid_dim=(num_blocks,),
            block_dim=(TPB,),
        )

        # Reduce gradients
        var gW1_env_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS * W1_SIZE), ImmutAnyOrigin
        ](grad_W1_per_env)
        var gb1_env_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS * HIDDEN_DIM), ImmutAnyOrigin
        ](grad_b1_per_env)
        var gW2_env_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS * W2_SIZE), ImmutAnyOrigin
        ](grad_W2_per_env)
        var gb2_env_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS * NUM_ACTIONS), ImmutAnyOrigin
        ](grad_b2_per_env)

        comptime reduce_W1 = reduce_gradients_kernel[
            dtype, NUM_ENVS, W1_SIZE, TPB
        ]
        comptime reduce_b1 = reduce_gradients_kernel[
            dtype, NUM_ENVS, HIDDEN_DIM, TPB
        ]
        comptime reduce_W2 = reduce_gradients_kernel[
            dtype, NUM_ENVS, W2_SIZE, TPB
        ]
        comptime reduce_b2 = reduce_gradients_kernel[
            dtype, NUM_ENVS, NUM_ACTIONS, TPB
        ]

        ctx.enqueue_function_checked[reduce_W1, reduce_W1](
            gW1_t, gW1_env_immut, grid_dim=(blocks_W1,), block_dim=(TPB,)
        )
        ctx.enqueue_function_checked[reduce_b1, reduce_b1](
            gb1_t, gb1_env_immut, grid_dim=(blocks_b1,), block_dim=(TPB,)
        )
        ctx.enqueue_function_checked[reduce_W2, reduce_W2](
            gW2_t, gW2_env_immut, grid_dim=(blocks_W2,), block_dim=(TPB,)
        )
        ctx.enqueue_function_checked[reduce_b2, reduce_b2](
            gb2_t, gb2_env_immut, grid_dim=(blocks_b2,), block_dim=(TPB,)
        )

        # SGD update
        var gW1_immut = LayoutTensor[
            dtype, Layout.row_major(W1_SIZE), ImmutAnyOrigin
        ](grad_W1)
        var gb1_immut = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin
        ](grad_b1)
        var gW2_immut = LayoutTensor[
            dtype, Layout.row_major(W2_SIZE), ImmutAnyOrigin
        ](grad_W2)
        var gb2_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ACTIONS), ImmutAnyOrigin
        ](grad_b2)

        comptime sgd_W1 = sgd_update_kernel[dtype, W1_SIZE, TPB]
        comptime sgd_b1 = sgd_update_kernel[dtype, HIDDEN_DIM, TPB]
        comptime sgd_W2 = sgd_update_kernel[dtype, W2_SIZE, TPB]
        comptime sgd_b2 = sgd_update_kernel[dtype, NUM_ACTIONS, TPB]

        ctx.enqueue_function_checked[sgd_W1, sgd_W1](
            W1_mut, gW1_immut, lr_s, grid_dim=(blocks_W1,), block_dim=(TPB,)
        )
        ctx.enqueue_function_checked[sgd_b1, sgd_b1](
            b1_mut, gb1_immut, lr_s, grid_dim=(blocks_b1,), block_dim=(TPB,)
        )
        ctx.enqueue_function_checked[sgd_W2, sgd_W2](
            W2_mut, gW2_immut, lr_s, grid_dim=(blocks_W2,), block_dim=(TPB,)
        )
        ctx.enqueue_function_checked[sgd_b2, sgd_b2](
            b2_mut, gb2_immut, lr_s, grid_dim=(blocks_b2,), block_dim=(TPB,)
        )

        # Logging
        if verbose and (update + 1) % 10 == 0:
            ctx.synchronize()
            var total_eps: Int32 = 0
            with total_episodes_buf.map_to_host() as host:
                for i in range(NUM_ENVS):
                    total_eps += host[i]
            var steps_so_far = (update + 1) * NUM_ENVS * STEPS_PER_KERNEL
            var avg_ep_len = (
                Float64(steps_so_far) / Float64(total_eps) if total_eps
                > 0 else 0.0
            )
            final_avg_ep_len = avg_ep_len
            print(
                "Update "
                + String(update + 1)
                + " | Episodes: "
                + String(total_eps)
                + " | Avg ep len: "
                + String(avg_ep_len)[:5]
            )

    ctx.synchronize()
    var end_time = perf_counter_ns()

    var total_steps = num_updates * NUM_ENVS * STEPS_PER_KERNEL
    var elapsed_sec = Float64(end_time - start_time) / 1e9
    var steps_per_sec = Float64(total_steps) / elapsed_sec

    var total_eps: Int32 = 0
    with total_episodes_buf.map_to_host() as host:
        for i in range(NUM_ENVS):
            total_eps += host[i]

    if verbose:
        print()
        print("Training complete!")
        print("  Total steps: " + String(total_steps))
        print("  Total episodes: " + String(total_eps))
        print("  Time: " + String(elapsed_sec)[:6] + " seconds")
        print("  Throughput: " + String(Int(steps_per_sec)) + " steps/sec")

    return TrainResult(
        total_steps=total_steps,
        total_episodes=Int(total_eps),
        elapsed_seconds=elapsed_sec,
        steps_per_sec=steps_per_sec,
        final_avg_ep_length=final_avg_ep_len,
    )

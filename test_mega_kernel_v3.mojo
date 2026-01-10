"""Mega-kernel A2C v3: Full training with backward pass + weight updates.

Key insight: Each thread computes gradients for its environment.
Weight updates need atomic operations or CPU aggregation.

For simplicity, we:
1. Store per-env gradients in global memory
2. Aggregate and update weights on CPU (between kernel calls)

This is a hybrid approach - not pure GPU but much faster than the naive version.
"""

from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from math import exp, log, sqrt, cos, sin
from random import random_float64

comptime dtype = DType.float32

# =============================================================================
# Physics constants
# =============================================================================

comptime GRAVITY: Float32 = 9.8
comptime MASSCART: Float32 = 1.0
comptime MASSPOLE: Float32 = 0.1
comptime TOTAL_MASS: Float32 = MASSCART + MASSPOLE
comptime LENGTH: Float32 = 0.5
comptime POLEMASS_LENGTH: Float32 = MASSPOLE * LENGTH
comptime FORCE_MAG: Float32 = 10.0
comptime TAU: Float32 = 0.02
comptime X_THRESHOLD: Float32 = 2.4
comptime THETA_THRESHOLD: Float32 = 0.2095


# =============================================================================
# Inline sub-operations
# =============================================================================


@always_inline
fn inline_relu(x: Scalar[dtype]) -> Scalar[dtype]:
    return x if x > 0 else Scalar[dtype](0)


@always_inline
fn inline_relu_grad(x: Scalar[dtype]) -> Scalar[dtype]:
    return Scalar[dtype](1.0) if x > 0 else Scalar[dtype](0.0)


@always_inline
fn inline_lcg_random(seed: UInt32) -> Tuple[UInt32, Scalar[dtype]]:
    var new_seed = seed * 1103515245 + 12345
    var val = Scalar[dtype](Float32(new_seed & 0x7FFFFFFF) / Float32(0x7FFFFFFF))
    return (new_seed, val)


# =============================================================================
# Mega kernel v3: Full A2C training step
# =============================================================================


fn mega_a2c_train_kernel[
    NUM_ENVS: Int,
    OBS_DIM: Int,
    HIDDEN_DIM: Int,
    NUM_ACTIONS: Int,
    ROLLOUT_LEN: Int,
](
    # Weights (mutable for updates)
    W1: LayoutTensor[dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), MutAnyOrigin],
    b1: LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), MutAnyOrigin],
    W_actor: LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, NUM_ACTIONS), MutAnyOrigin],
    b_actor: LayoutTensor[dtype, Layout.row_major(NUM_ACTIONS), MutAnyOrigin],
    W_critic: LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, 1), MutAnyOrigin],
    b_critic: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
    # Per-env state
    env_states: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, 4), MutAnyOrigin],
    random_seeds: LayoutTensor[DType.uint32, Layout.row_major(NUM_ENVS), MutAnyOrigin],
    # Per-env gradient accumulators (accumulated, then CPU does weight update)
    grad_W1: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, OBS_DIM * HIDDEN_DIM), MutAnyOrigin],
    grad_b1: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin],
    grad_W_actor: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM * NUM_ACTIONS), MutAnyOrigin],
    grad_b_actor: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), MutAnyOrigin],
    grad_W_critic: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin],
    grad_b_critic: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin],
    # Outputs
    total_rewards: LayoutTensor[dtype, Layout.row_major(NUM_ENVS), MutAnyOrigin],
    episode_counts: LayoutTensor[DType.int32, Layout.row_major(NUM_ENVS), MutAnyOrigin],
    # Hyperparameters
    gamma: Scalar[dtype],
    gae_lambda: Scalar[dtype],
    entropy_coef: Scalar[dtype],
    value_coef: Scalar[dtype],
):
    """Full A2C training: rollout + GAE + backward + gradient accumulation."""
    var env_idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env_idx >= NUM_ENVS:
        return

    # Initialize gradient accumulators to zero
    for i in range(OBS_DIM * HIDDEN_DIM):
        grad_W1[env_idx, i] = Scalar[dtype](0.0)
    for i in range(HIDDEN_DIM):
        grad_b1[env_idx, i] = Scalar[dtype](0.0)
    for i in range(HIDDEN_DIM * NUM_ACTIONS):
        grad_W_actor[env_idx, i] = Scalar[dtype](0.0)
    for i in range(NUM_ACTIONS):
        grad_b_actor[env_idx, i] = Scalar[dtype](0.0)
    for i in range(HIDDEN_DIM):
        grad_W_critic[env_idx, i] = Scalar[dtype](0.0)
    grad_b_critic[env_idx, 0] = Scalar[dtype](0.0)

    # Load env state
    var x = rebind[Scalar[dtype]](env_states[env_idx, 0])
    var x_dot = rebind[Scalar[dtype]](env_states[env_idx, 1])
    var theta = rebind[Scalar[dtype]](env_states[env_idx, 2])
    var theta_dot = rebind[Scalar[dtype]](env_states[env_idx, 3])
    var rng = rebind[UInt32](random_seeds[env_idx])

    var total_reward: Scalar[dtype] = 0
    var episodes: Int = 0

    # Rollout storage (small, fits in registers/local memory)
    var rewards = InlineArray[Scalar[dtype], ROLLOUT_LEN](uninitialized=True)
    var values = InlineArray[Scalar[dtype], ROLLOUT_LEN](uninitialized=True)
    var log_probs = InlineArray[Scalar[dtype], ROLLOUT_LEN](uninitialized=True)
    var dones = InlineArray[Int, ROLLOUT_LEN](uninitialized=True)
    var actions = InlineArray[Int, ROLLOUT_LEN](uninitialized=True)

    # Store observations and hidden states for backprop
    var obs_history = InlineArray[InlineArray[Scalar[dtype], OBS_DIM], ROLLOUT_LEN](uninitialized=True)
    var hidden_history = InlineArray[InlineArray[Scalar[dtype], HIDDEN_DIM], ROLLOUT_LEN](uninitialized=True)
    var pre_act_history = InlineArray[InlineArray[Scalar[dtype], HIDDEN_DIM], ROLLOUT_LEN](uninitialized=True)

    # =========================================================================
    # Phase 1: Collect rollout with forward pass
    # =========================================================================
    for step in range(ROLLOUT_LEN):
        # Store observation
        obs_history[step][0] = x
        obs_history[step][1] = x_dot
        obs_history[step][2] = theta
        obs_history[step][3] = theta_dot

        # Forward pass - Layer 1
        for h in range(HIDDEN_DIM):
            var acc: Scalar[dtype] = rebind[Scalar[dtype]](b1[h])
            acc += x * rebind[Scalar[dtype]](W1[0, h])
            acc += x_dot * rebind[Scalar[dtype]](W1[1, h])
            acc += theta * rebind[Scalar[dtype]](W1[2, h])
            acc += theta_dot * rebind[Scalar[dtype]](W1[3, h])
            pre_act_history[step][h] = acc
            hidden_history[step][h] = inline_relu(acc)

        # Actor forward
        var logits = InlineArray[Scalar[dtype], NUM_ACTIONS](uninitialized=True)
        var max_logit: Scalar[dtype] = Scalar[dtype](-1e10)
        for a in range(NUM_ACTIONS):
            var acc: Scalar[dtype] = rebind[Scalar[dtype]](b_actor[a])
            for h in range(HIDDEN_DIM):
                acc += hidden_history[step][h] * rebind[Scalar[dtype]](W_actor[h, a])
            logits[a] = acc
            if acc > max_logit:
                max_logit = acc

        # Softmax
        var sum_exp: Scalar[dtype] = 0
        var probs = InlineArray[Scalar[dtype], NUM_ACTIONS](uninitialized=True)
        for a in range(NUM_ACTIONS):
            var e = exp(logits[a] - max_logit)
            probs[a] = e
            sum_exp += e
        for a in range(NUM_ACTIONS):
            probs[a] = probs[a] / sum_exp

        # Critic forward
        var value: Scalar[dtype] = rebind[Scalar[dtype]](b_critic[0])
        for h in range(HIDDEN_DIM):
            value += hidden_history[step][h] * rebind[Scalar[dtype]](W_critic[h, 0])
        values[step] = value

        # Sample action
        var rand_result = inline_lcg_random(rng)
        rng = rand_result[0]
        var action = 0
        if rand_result[1] >= probs[0]:
            action = 1
        actions[step] = action
        log_probs[step] = log(probs[action] + Scalar[dtype](1e-10))

        # Step environment
        var force = Scalar[dtype](FORCE_MAG) if action == 1 else Scalar[dtype](-FORCE_MAG)
        var costheta = cos(theta)
        var sintheta = sin(theta)
        var temp = (force + Scalar[dtype](POLEMASS_LENGTH) * theta_dot * theta_dot * sintheta) / Scalar[dtype](TOTAL_MASS)
        var theta_acc = (Scalar[dtype](GRAVITY) * sintheta - costheta * temp) / (
            Scalar[dtype](LENGTH) * (Scalar[dtype](4.0 / 3.0) - Scalar[dtype](MASSPOLE) * costheta * costheta / Scalar[dtype](TOTAL_MASS))
        )
        var x_acc = temp - Scalar[dtype](POLEMASS_LENGTH) * theta_acc * costheta / Scalar[dtype](TOTAL_MASS)

        x = x + Scalar[dtype](TAU) * x_dot
        x_dot = x_dot + Scalar[dtype](TAU) * x_acc
        theta = theta + Scalar[dtype](TAU) * theta_dot
        theta_dot = theta_dot + Scalar[dtype](TAU) * theta_acc

        var done = (
            x < Scalar[dtype](-X_THRESHOLD) or x > Scalar[dtype](X_THRESHOLD)
            or theta < Scalar[dtype](-THETA_THRESHOLD) or theta > Scalar[dtype](THETA_THRESHOLD)
        )

        rewards[step] = Scalar[dtype](1.0)
        dones[step] = 1 if done else 0
        total_reward += Scalar[dtype](1.0)

        if done:
            episodes += 1
            var r1 = inline_lcg_random(rng)
            rng = r1[0]
            x = r1[1] * Scalar[dtype](0.1) - Scalar[dtype](0.05)
            var r2 = inline_lcg_random(rng)
            rng = r2[0]
            x_dot = r2[1] * Scalar[dtype](0.1) - Scalar[dtype](0.05)
            var r3 = inline_lcg_random(rng)
            rng = r3[0]
            theta = r3[1] * Scalar[dtype](0.1) - Scalar[dtype](0.05)
            var r4 = inline_lcg_random(rng)
            rng = r4[0]
            theta_dot = r4[1] * Scalar[dtype](0.1) - Scalar[dtype](0.05)

    # =========================================================================
    # Phase 2: Compute bootstrap value
    # =========================================================================
    var bootstrap_hidden = InlineArray[Scalar[dtype], HIDDEN_DIM](uninitialized=True)
    for h in range(HIDDEN_DIM):
        var acc: Scalar[dtype] = rebind[Scalar[dtype]](b1[h])
        acc += x * rebind[Scalar[dtype]](W1[0, h])
        acc += x_dot * rebind[Scalar[dtype]](W1[1, h])
        acc += theta * rebind[Scalar[dtype]](W1[2, h])
        acc += theta_dot * rebind[Scalar[dtype]](W1[3, h])
        bootstrap_hidden[h] = inline_relu(acc)

    var bootstrap_value: Scalar[dtype] = rebind[Scalar[dtype]](b_critic[0])
    for h in range(HIDDEN_DIM):
        bootstrap_value += bootstrap_hidden[h] * rebind[Scalar[dtype]](W_critic[h, 0])

    # =========================================================================
    # Phase 3: Compute GAE advantages
    # =========================================================================
    var advantages = InlineArray[Scalar[dtype], ROLLOUT_LEN](uninitialized=True)
    var returns = InlineArray[Scalar[dtype], ROLLOUT_LEN](uninitialized=True)

    var gae: Scalar[dtype] = 0
    var next_value = bootstrap_value

    for t_rev in range(ROLLOUT_LEN):
        var t = ROLLOUT_LEN - 1 - t_rev
        var not_done = Scalar[dtype](1.0) if dones[t] == 0 else Scalar[dtype](0.0)
        var delta = rewards[t] + gamma * next_value * not_done - values[t]
        gae = delta + gamma * gae_lambda * not_done * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
        next_value = values[t]

    # =========================================================================
    # Phase 4: Backward pass - accumulate gradients
    # =========================================================================
    for step in range(ROLLOUT_LEN):
        var adv = advantages[step]
        var ret = returns[step]
        var action = actions[step]

        # Recompute forward pass for this step (could optimize by storing more)
        var probs = InlineArray[Scalar[dtype], NUM_ACTIONS](uninitialized=True)
        var logits = InlineArray[Scalar[dtype], NUM_ACTIONS](uninitialized=True)
        var max_logit: Scalar[dtype] = Scalar[dtype](-1e10)

        for a in range(NUM_ACTIONS):
            var acc: Scalar[dtype] = rebind[Scalar[dtype]](b_actor[a])
            for h in range(HIDDEN_DIM):
                acc += hidden_history[step][h] * rebind[Scalar[dtype]](W_actor[h, a])
            logits[a] = acc
            if acc > max_logit:
                max_logit = acc

        var sum_exp: Scalar[dtype] = 0
        for a in range(NUM_ACTIONS):
            var e = exp(logits[a] - max_logit)
            probs[a] = e
            sum_exp += e
        for a in range(NUM_ACTIONS):
            probs[a] = probs[a] / sum_exp

        # Policy gradient: d_logits[a] = (probs[a] - one_hot[a]) * advantage
        # Plus entropy bonus: d_logits[a] += entropy_coef * probs[a] * (log(probs[a]) + 1)
        var d_logits = InlineArray[Scalar[dtype], NUM_ACTIONS](uninitialized=True)
        for a in range(NUM_ACTIONS):
            var one_hot: Scalar[dtype] = Scalar[dtype](1.0) if a == action else Scalar[dtype](0.0)
            d_logits[a] = (probs[a] - one_hot) * adv
            # Entropy gradient
            d_logits[a] = d_logits[a] + entropy_coef * probs[a] * (log(probs[a] + Scalar[dtype](1e-10)) + Scalar[dtype](1.0))

        # Value loss gradient: d_value = 2 * value_coef * (value - return)
        var d_value = Scalar[dtype](2.0) * value_coef * (values[step] - ret)

        # Backprop through actor
        var d_hidden_actor = InlineArray[Scalar[dtype], HIDDEN_DIM](uninitialized=True)
        for h in range(HIDDEN_DIM):
            d_hidden_actor[h] = Scalar[dtype](0.0)

        for a in range(NUM_ACTIONS):
            # d_W_actor[h, a] += hidden[h] * d_logits[a]
            for h in range(HIDDEN_DIM):
                var idx = h * NUM_ACTIONS + a
                grad_W_actor[env_idx, idx] = rebind[Scalar[dtype]](grad_W_actor[env_idx, idx]) + hidden_history[step][h] * d_logits[a]
                d_hidden_actor[h] = d_hidden_actor[h] + rebind[Scalar[dtype]](W_actor[h, a]) * d_logits[a]
            # d_b_actor[a] += d_logits[a]
            grad_b_actor[env_idx, a] = rebind[Scalar[dtype]](grad_b_actor[env_idx, a]) + d_logits[a]

        # Backprop through critic
        var d_hidden_critic = InlineArray[Scalar[dtype], HIDDEN_DIM](uninitialized=True)
        for h in range(HIDDEN_DIM):
            # d_W_critic[h] += hidden[h] * d_value
            grad_W_critic[env_idx, h] = rebind[Scalar[dtype]](grad_W_critic[env_idx, h]) + hidden_history[step][h] * d_value
            d_hidden_critic[h] = rebind[Scalar[dtype]](W_critic[h, 0]) * d_value
        grad_b_critic[env_idx, 0] = rebind[Scalar[dtype]](grad_b_critic[env_idx, 0]) + d_value

        # Combine hidden gradients and backprop through ReLU
        var d_pre_relu = InlineArray[Scalar[dtype], HIDDEN_DIM](uninitialized=True)
        for h in range(HIDDEN_DIM):
            var d_hidden = d_hidden_actor[h] + d_hidden_critic[h]
            d_pre_relu[h] = d_hidden * inline_relu_grad(pre_act_history[step][h])

        # Backprop through first layer
        for h in range(HIDDEN_DIM):
            for d in range(OBS_DIM):
                var idx = d * HIDDEN_DIM + h
                grad_W1[env_idx, idx] = rebind[Scalar[dtype]](grad_W1[env_idx, idx]) + obs_history[step][d] * d_pre_relu[h]
            grad_b1[env_idx, h] = rebind[Scalar[dtype]](grad_b1[env_idx, h]) + d_pre_relu[h]

    # Save state
    env_states[env_idx, 0] = x
    env_states[env_idx, 1] = x_dot
    env_states[env_idx, 2] = theta
    env_states[env_idx, 3] = theta_dot
    random_seeds[env_idx] = rebind[Scalar[DType.uint32]](rng)
    total_rewards[env_idx] = total_reward
    episode_counts[env_idx] = Scalar[DType.int32](episodes)


# =============================================================================
# Test with CPU weight updates
# =============================================================================


fn main() raises:
    print("Testing mega-kernel v3: Full A2C Training")
    print("=" * 55)

    comptime NUM_ENVS = 64  # Reduced for faster testing
    comptime OBS_DIM = 4
    comptime HIDDEN_DIM = 32
    comptime NUM_ACTIONS = 2
    comptime ROLLOUT_LEN = 32  # Reduced for faster testing

    var gamma = Scalar[dtype](0.99)
    var gae_lambda = Scalar[dtype](0.95)
    var entropy_coef = Scalar[dtype](0.01)
    var value_coef = Scalar[dtype](0.5)
    var lr: Float64 = 0.0007  # Standard A2C learning rate

    with DeviceContext() as ctx:
        # Weight buffers
        var W1_buf = ctx.enqueue_create_buffer[dtype](OBS_DIM * HIDDEN_DIM)
        var b1_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
        var W_actor_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM * NUM_ACTIONS)
        var b_actor_buf = ctx.enqueue_create_buffer[dtype](NUM_ACTIONS)
        var W_critic_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM * 1)
        var b_critic_buf = ctx.enqueue_create_buffer[dtype](1)

        # Env state buffers
        var env_states_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * 4)
        var random_seeds_buf = ctx.enqueue_create_buffer[DType.uint32](NUM_ENVS)

        # Gradient buffers (per-env)
        var grad_W1_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * OBS_DIM * HIDDEN_DIM)
        var grad_b1_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * HIDDEN_DIM)
        var grad_W_actor_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * HIDDEN_DIM * NUM_ACTIONS)
        var grad_b_actor_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * NUM_ACTIONS)
        var grad_W_critic_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * HIDDEN_DIM)
        var grad_b_critic_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * 1)

        # Output buffers
        var total_rewards_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS)
        var episode_counts_buf = ctx.enqueue_create_buffer[DType.int32](NUM_ENVS)

        # Initialize weights (Xavier-ish)
        with W1_buf.map_to_host() as h:
            var scale = sqrt(2.0 / Float64(OBS_DIM))
            for i in range(OBS_DIM * HIDDEN_DIM):
                h[i] = Scalar[dtype]((random_float64() - 0.5) * scale)
        with b1_buf.map_to_host() as h:
            for i in range(HIDDEN_DIM):
                h[i] = Scalar[dtype](0.0)
        with W_actor_buf.map_to_host() as h:
            var scale = sqrt(2.0 / Float64(HIDDEN_DIM)) * 0.01
            for i in range(HIDDEN_DIM * NUM_ACTIONS):
                h[i] = Scalar[dtype]((random_float64() - 0.5) * scale)
        with b_actor_buf.map_to_host() as h:
            for i in range(NUM_ACTIONS):
                h[i] = Scalar[dtype](0.0)
        with W_critic_buf.map_to_host() as h:
            var scale = sqrt(2.0 / Float64(HIDDEN_DIM))
            for i in range(HIDDEN_DIM):
                h[i] = Scalar[dtype]((random_float64() - 0.5) * scale)
        with b_critic_buf.map_to_host() as h:
            h[0] = Scalar[dtype](0.0)

        # Initialize env states
        with env_states_buf.map_to_host() as h:
            for i in range(NUM_ENVS):
                h[i * 4 + 0] = Scalar[dtype]((random_float64() - 0.5) * 0.1)
                h[i * 4 + 1] = Scalar[dtype]((random_float64() - 0.5) * 0.1)
                h[i * 4 + 2] = Scalar[dtype]((random_float64() - 0.5) * 0.1)
                h[i * 4 + 3] = Scalar[dtype]((random_float64() - 0.5) * 0.1)

        # Initialize random seeds
        with random_seeds_buf.map_to_host() as h:
            for i in range(NUM_ENVS):
                h[i] = Scalar[DType.uint32](UInt32(i * 12345 + 1))

        # Create tensors
        var W1 = LayoutTensor[dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), MutAnyOrigin](W1_buf)
        var b1 = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), MutAnyOrigin](b1_buf)
        var W_actor = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, NUM_ACTIONS), MutAnyOrigin](W_actor_buf)
        var b_actor = LayoutTensor[dtype, Layout.row_major(NUM_ACTIONS), MutAnyOrigin](b_actor_buf)
        var W_critic = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, 1), MutAnyOrigin](W_critic_buf)
        var b_critic = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](b_critic_buf)
        var env_states = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, 4), MutAnyOrigin](env_states_buf)
        var random_seeds = LayoutTensor[DType.uint32, Layout.row_major(NUM_ENVS), MutAnyOrigin](random_seeds_buf)
        var grad_W1 = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, OBS_DIM * HIDDEN_DIM), MutAnyOrigin](grad_W1_buf)
        var grad_b1 = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin](grad_b1_buf)
        var grad_W_actor = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM * NUM_ACTIONS), MutAnyOrigin](grad_W_actor_buf)
        var grad_b_actor = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), MutAnyOrigin](grad_b_actor_buf)
        var grad_W_critic = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin](grad_W_critic_buf)
        var grad_b_critic = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin](grad_b_critic_buf)
        var total_rewards = LayoutTensor[dtype, Layout.row_major(NUM_ENVS), MutAnyOrigin](total_rewards_buf)
        var episode_counts = LayoutTensor[DType.int32, Layout.row_major(NUM_ENVS), MutAnyOrigin](episode_counts_buf)

        print("Training: NUM_ENVS=", NUM_ENVS, " ROLLOUT_LEN=", ROLLOUT_LEN)
        print()

        var total_episodes = 0
        var recent_rewards = List[Float64]()

        for update in range(500):
            # Run training kernel
            ctx.enqueue_function_checked[
                mega_a2c_train_kernel[NUM_ENVS, OBS_DIM, HIDDEN_DIM, NUM_ACTIONS, ROLLOUT_LEN],
                mega_a2c_train_kernel[NUM_ENVS, OBS_DIM, HIDDEN_DIM, NUM_ACTIONS, ROLLOUT_LEN],
            ](
                W1, b1, W_actor, b_actor, W_critic, b_critic,
                env_states, random_seeds,
                grad_W1, grad_b1, grad_W_actor, grad_b_actor, grad_W_critic, grad_b_critic,
                total_rewards, episode_counts,
                gamma, gae_lambda, entropy_coef, value_coef,
                grid_dim=(1, 1),
                block_dim=(NUM_ENVS, 1),
            )
            ctx.synchronize()

            # Aggregate gradients and update weights on CPU
            with W1_buf.map_to_host() as W1_h, b1_buf.map_to_host() as b1_h, W_actor_buf.map_to_host() as Wa_h, b_actor_buf.map_to_host() as ba_h, W_critic_buf.map_to_host() as Wc_h, b_critic_buf.map_to_host() as bc_h, grad_W1_buf.map_to_host() as gW1, grad_b1_buf.map_to_host() as gb1, grad_W_actor_buf.map_to_host() as gWa, grad_b_actor_buf.map_to_host() as gba, grad_W_critic_buf.map_to_host() as gWc, grad_b_critic_buf.map_to_host() as gbc:
                # Aggregate and apply gradients
                # Note: gradients already sum over ROLLOUT_LEN, so just average over envs
                var scale = lr / Float64(NUM_ENVS)

                for i in range(OBS_DIM * HIDDEN_DIM):
                    var grad_sum: Float64 = 0
                    for e in range(NUM_ENVS):
                        grad_sum += Float64(gW1[e * OBS_DIM * HIDDEN_DIM + i])
                    W1_h[i] = Scalar[dtype](Float64(W1_h[i]) - scale * grad_sum)

                for i in range(HIDDEN_DIM):
                    var grad_sum: Float64 = 0
                    for e in range(NUM_ENVS):
                        grad_sum += Float64(gb1[e * HIDDEN_DIM + i])
                    b1_h[i] = Scalar[dtype](Float64(b1_h[i]) - scale * grad_sum)

                for i in range(HIDDEN_DIM * NUM_ACTIONS):
                    var grad_sum: Float64 = 0
                    for e in range(NUM_ENVS):
                        grad_sum += Float64(gWa[e * HIDDEN_DIM * NUM_ACTIONS + i])
                    Wa_h[i] = Scalar[dtype](Float64(Wa_h[i]) - scale * grad_sum)

                for i in range(NUM_ACTIONS):
                    var grad_sum: Float64 = 0
                    for e in range(NUM_ENVS):
                        grad_sum += Float64(gba[e * NUM_ACTIONS + i])
                    ba_h[i] = Scalar[dtype](Float64(ba_h[i]) - scale * grad_sum)

                for i in range(HIDDEN_DIM):
                    var grad_sum: Float64 = 0
                    for e in range(NUM_ENVS):
                        grad_sum += Float64(gWc[e * HIDDEN_DIM + i])
                    Wc_h[i] = Scalar[dtype](Float64(Wc_h[i]) - scale * grad_sum)

                var grad_sum_bc: Float64 = 0
                for e in range(NUM_ENVS):
                    grad_sum_bc += Float64(gbc[e])
                bc_h[0] = Scalar[dtype](Float64(bc_h[0]) - scale * grad_sum_bc)

            # Track progress
            with total_rewards_buf.map_to_host() as r, episode_counts_buf.map_to_host() as e:
                var iter_episodes = 0
                for i in range(NUM_ENVS):
                    iter_episodes += Int(e[i])
                    if Int(e[i]) > 0:
                        var avg_ep_reward = Float64(r[i]) / Float64(e[i])
                        recent_rewards.append(avg_ep_reward)
                        if len(recent_rewards) > 100:
                            _ = recent_rewards.pop(0)
                total_episodes += iter_episodes

            if (update + 1) % 50 == 0:
                var avg_reward: Float64 = 0
                if len(recent_rewards) > 0:
                    for i in range(len(recent_rewards)):
                        avg_reward += recent_rewards[i]
                    avg_reward /= len(recent_rewards)
                print("Update", update + 1, "| Episodes:", total_episodes, "| Avg reward:", avg_reward)

        print()
        print("Training complete! Total episodes:", total_episodes)

        var final_avg: Float64 = 0
        if len(recent_rewards) > 0:
            for i in range(len(recent_rewards)):
                final_avg += recent_rewards[i]
            final_avg /= len(recent_rewards)
        print("Final avg episode reward:", final_avg)

        if final_avg >= 100:
            print("Good progress! Agent is learning.")
        elif final_avg >= 50:
            print("Some learning detected.")
        else:
            print("Training may need more iterations or hyperparameter tuning.")

    print("=" * 55)
    print("Mega-kernel v3 test completed!")

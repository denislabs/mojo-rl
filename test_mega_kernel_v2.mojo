"""Mega-kernel A2C v2: Rollout + GAE + Policy Gradient.

Progressive build:
- v1: Forward pass + environment stepping (DONE)
- v2: Add rollout storage + GAE computation (THIS FILE)
- v3: Add backward pass + weight updates
"""

from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from math import exp, log, sqrt, cos, sin

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
fn inline_lcg_random(seed: UInt32) -> Tuple[UInt32, Scalar[dtype]]:
    """LCG random: returns (new_seed, random_value in [0,1])."""
    var new_seed = seed * 1103515245 + 12345
    var val = Scalar[dtype](Float32(new_seed & 0x7FFFFFFF) / Float32(0x7FFFFFFF))
    return (new_seed, val)


# =============================================================================
# Mega kernel v2: Rollout + GAE + Policy Gradient (no weight update yet)
# =============================================================================


fn mega_a2c_kernel_v2[
    NUM_ENVS: Int,
    OBS_DIM: Int,
    HIDDEN_DIM: Int,
    NUM_ACTIONS: Int,
    ROLLOUT_LEN: Int,
](
    # Weights (read-only for now - will make mutable in v3)
    W1: LayoutTensor[dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), ImmutAnyOrigin],
    b1: LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin],
    W_actor: LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, NUM_ACTIONS), ImmutAnyOrigin],
    b_actor: LayoutTensor[dtype, Layout.row_major(NUM_ACTIONS), ImmutAnyOrigin],
    W_critic: LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, 1), ImmutAnyOrigin],
    b_critic: LayoutTensor[dtype, Layout.row_major(1), ImmutAnyOrigin],
    # Per-env state (mutable)
    env_states: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, 4), MutAnyOrigin],
    random_seeds: LayoutTensor[DType.uint32, Layout.row_major(NUM_ENVS), MutAnyOrigin],
    # Rollout storage (per-env, per-step) - stored in global memory
    rollout_log_probs: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin],
    rollout_values: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin],
    rollout_rewards: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin],
    rollout_dones: LayoutTensor[DType.int32, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin],
    # GAE outputs
    rollout_advantages: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin],
    rollout_returns: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin],
    # Outputs
    total_rewards: LayoutTensor[dtype, Layout.row_major(NUM_ENVS), MutAnyOrigin],
    episode_counts: LayoutTensor[DType.int32, Layout.row_major(NUM_ENVS), MutAnyOrigin],
    # Hyperparameters
    gamma: Scalar[dtype],
    gae_lambda: Scalar[dtype],
):
    """Collect rollout + compute GAE advantages."""
    var env_idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env_idx >= NUM_ENVS:
        return

    # Load env state
    var x = rebind[Scalar[dtype]](env_states[env_idx, 0])
    var x_dot = rebind[Scalar[dtype]](env_states[env_idx, 1])
    var theta = rebind[Scalar[dtype]](env_states[env_idx, 2])
    var theta_dot = rebind[Scalar[dtype]](env_states[env_idx, 3])
    var rng = rebind[UInt32](random_seeds[env_idx])

    var total_reward: Scalar[dtype] = 0
    var episodes: Int = 0

    # =========================================================================
    # Phase 1: Collect rollout
    # =========================================================================
    for step in range(ROLLOUT_LEN):
        # === Forward pass ===

        # Layer 1: obs @ W1 + b1 -> ReLU
        var hidden_vals = InlineArray[Scalar[dtype], HIDDEN_DIM](uninitialized=True)
        for h in range(HIDDEN_DIM):
            var acc: Scalar[dtype] = rebind[Scalar[dtype]](b1[h])
            acc += x * rebind[Scalar[dtype]](W1[0, h])
            acc += x_dot * rebind[Scalar[dtype]](W1[1, h])
            acc += theta * rebind[Scalar[dtype]](W1[2, h])
            acc += theta_dot * rebind[Scalar[dtype]](W1[3, h])
            hidden_vals[h] = inline_relu(acc)

        # Actor: hidden @ W_actor + b_actor -> softmax
        var logits = InlineArray[Scalar[dtype], NUM_ACTIONS](uninitialized=True)
        var max_logit: Scalar[dtype] = Scalar[dtype](-1e10)
        for a in range(NUM_ACTIONS):
            var acc: Scalar[dtype] = rebind[Scalar[dtype]](b_actor[a])
            for h in range(HIDDEN_DIM):
                acc += hidden_vals[h] * rebind[Scalar[dtype]](W_actor[h, a])
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

        # Critic: hidden @ W_critic + b_critic
        var value: Scalar[dtype] = rebind[Scalar[dtype]](b_critic[0])
        for h in range(HIDDEN_DIM):
            value += hidden_vals[h] * rebind[Scalar[dtype]](W_critic[h, 0])

        # Sample action
        var rand_result = inline_lcg_random(rng)
        rng = rand_result[0]
        var rand_val = rand_result[1]
        var action = 0
        if rand_val >= probs[0]:
            action = 1

        # Store rollout data
        var log_prob = log(probs[action] + Scalar[dtype](1e-10))
        rollout_log_probs[env_idx, step] = log_prob
        rollout_values[env_idx, step] = value

        # === Step environment ===
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
            x < Scalar[dtype](-X_THRESHOLD)
            or x > Scalar[dtype](X_THRESHOLD)
            or theta < Scalar[dtype](-THETA_THRESHOLD)
            or theta > Scalar[dtype](THETA_THRESHOLD)
        )

        var reward: Scalar[dtype] = Scalar[dtype](1.0)
        rollout_rewards[env_idx, step] = reward
        rollout_dones[env_idx, step] = 1 if done else 0
        total_reward += reward

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
    # Phase 2: Compute bootstrap value (for GAE)
    # =========================================================================
    var hidden_vals = InlineArray[Scalar[dtype], HIDDEN_DIM](uninitialized=True)
    for h in range(HIDDEN_DIM):
        var acc: Scalar[dtype] = rebind[Scalar[dtype]](b1[h])
        acc += x * rebind[Scalar[dtype]](W1[0, h])
        acc += x_dot * rebind[Scalar[dtype]](W1[1, h])
        acc += theta * rebind[Scalar[dtype]](W1[2, h])
        acc += theta_dot * rebind[Scalar[dtype]](W1[3, h])
        hidden_vals[h] = inline_relu(acc)

    var bootstrap_value: Scalar[dtype] = rebind[Scalar[dtype]](b_critic[0])
    for h in range(HIDDEN_DIM):
        bootstrap_value += hidden_vals[h] * rebind[Scalar[dtype]](W_critic[h, 0])

    # =========================================================================
    # Phase 3: Compute GAE advantages and returns
    # =========================================================================
    var gae: Scalar[dtype] = 0
    var next_value = bootstrap_value

    # Backward pass through rollout
    for t_rev in range(ROLLOUT_LEN):
        var t = ROLLOUT_LEN - 1 - t_rev  # Reverse index

        var reward = rebind[Scalar[dtype]](rollout_rewards[env_idx, t])
        var value = rebind[Scalar[dtype]](rollout_values[env_idx, t])
        var done = Int(rollout_dones[env_idx, t])

        var not_done: Scalar[dtype] = Scalar[dtype](1.0) if done == 0 else Scalar[dtype](0.0)
        var delta = reward + gamma * next_value * not_done - value
        gae = delta + gamma * gae_lambda * not_done * gae

        rollout_advantages[env_idx, t] = gae
        rollout_returns[env_idx, t] = gae + value
        next_value = value

    # Save state back
    env_states[env_idx, 0] = x
    env_states[env_idx, 1] = x_dot
    env_states[env_idx, 2] = theta
    env_states[env_idx, 3] = theta_dot
    random_seeds[env_idx] = rebind[Scalar[DType.uint32]](rng)
    total_rewards[env_idx] = total_reward
    episode_counts[env_idx] = Scalar[DType.int32](episodes)


# =============================================================================
# Test
# =============================================================================


fn main() raises:
    print("Testing mega-kernel v2: Rollout + GAE")
    print("=" * 55)

    comptime NUM_ENVS = 256
    comptime OBS_DIM = 4
    comptime HIDDEN_DIM = 32
    comptime NUM_ACTIONS = 2
    comptime ROLLOUT_LEN = 128

    var gamma = Scalar[dtype](0.99)
    var gae_lambda = Scalar[dtype](0.95)

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

        # Rollout buffers
        var rollout_log_probs_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * ROLLOUT_LEN)
        var rollout_values_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * ROLLOUT_LEN)
        var rollout_rewards_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * ROLLOUT_LEN)
        var rollout_dones_buf = ctx.enqueue_create_buffer[DType.int32](NUM_ENVS * ROLLOUT_LEN)
        var rollout_advantages_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * ROLLOUT_LEN)
        var rollout_returns_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * ROLLOUT_LEN)

        # Output buffers
        var total_rewards_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS)
        var episode_counts_buf = ctx.enqueue_create_buffer[DType.int32](NUM_ENVS)

        # Initialize weights
        with W1_buf.map_to_host() as h:
            for i in range(OBS_DIM * HIDDEN_DIM):
                h[i] = Scalar[dtype](0.2)
        with b1_buf.map_to_host() as h:
            for i in range(HIDDEN_DIM):
                h[i] = Scalar[dtype](0.0)
        with W_actor_buf.map_to_host() as h:
            for i in range(HIDDEN_DIM * NUM_ACTIONS):
                h[i] = Scalar[dtype](0.05)
        with b_actor_buf.map_to_host() as h:
            for i in range(NUM_ACTIONS):
                h[i] = Scalar[dtype](0.0)
        with W_critic_buf.map_to_host() as h:
            for i in range(HIDDEN_DIM):
                h[i] = Scalar[dtype](0.1)
        with b_critic_buf.map_to_host() as h:
            h[0] = Scalar[dtype](0.0)

        # Initialize env states
        with env_states_buf.map_to_host() as h:
            for i in range(NUM_ENVS * 4):
                h[i] = Scalar[dtype](0.01)

        # Initialize random seeds
        with random_seeds_buf.map_to_host() as h:
            for i in range(NUM_ENVS):
                h[i] = Scalar[DType.uint32](UInt32(i * 12345 + 1))

        # Create tensors
        var W1 = LayoutTensor[dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), ImmutAnyOrigin](W1_buf)
        var b1 = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin](b1_buf)
        var W_actor = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, NUM_ACTIONS), ImmutAnyOrigin](W_actor_buf)
        var b_actor = LayoutTensor[dtype, Layout.row_major(NUM_ACTIONS), ImmutAnyOrigin](b_actor_buf)
        var W_critic = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, 1), ImmutAnyOrigin](W_critic_buf)
        var b_critic = LayoutTensor[dtype, Layout.row_major(1), ImmutAnyOrigin](b_critic_buf)
        var env_states = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, 4), MutAnyOrigin](env_states_buf)
        var random_seeds = LayoutTensor[DType.uint32, Layout.row_major(NUM_ENVS), MutAnyOrigin](random_seeds_buf)
        var rollout_log_probs = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin](rollout_log_probs_buf)
        var rollout_values = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin](rollout_values_buf)
        var rollout_rewards = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin](rollout_rewards_buf)
        var rollout_dones = LayoutTensor[DType.int32, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin](rollout_dones_buf)
        var rollout_advantages = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin](rollout_advantages_buf)
        var rollout_returns = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin](rollout_returns_buf)
        var total_rewards = LayoutTensor[dtype, Layout.row_major(NUM_ENVS), MutAnyOrigin](total_rewards_buf)
        var episode_counts = LayoutTensor[DType.int32, Layout.row_major(NUM_ENVS), MutAnyOrigin](episode_counts_buf)

        print("Running", NUM_ENVS, "envs x", ROLLOUT_LEN, "steps =", NUM_ENVS * ROLLOUT_LEN, "steps per update")
        print()

        var total_steps = 0
        var total_episodes = 0

        for iteration in range(10):
            ctx.enqueue_function_checked[
                mega_a2c_kernel_v2[NUM_ENVS, OBS_DIM, HIDDEN_DIM, NUM_ACTIONS, ROLLOUT_LEN],
                mega_a2c_kernel_v2[NUM_ENVS, OBS_DIM, HIDDEN_DIM, NUM_ACTIONS, ROLLOUT_LEN],
            ](
                W1, b1, W_actor, b_actor, W_critic, b_critic,
                env_states, random_seeds,
                rollout_log_probs, rollout_values, rollout_rewards, rollout_dones,
                rollout_advantages, rollout_returns,
                total_rewards, episode_counts,
                gamma, gae_lambda,
                grid_dim=(1, 1),
                block_dim=(NUM_ENVS, 1),
            )
            ctx.synchronize()

            # Check results
            with total_rewards_buf.map_to_host() as r, episode_counts_buf.map_to_host() as e, rollout_advantages_buf.map_to_host() as adv:
                var iter_reward: Float64 = 0
                var iter_episodes: Int = 0
                for i in range(NUM_ENVS):
                    iter_reward += Float64(r[i])
                    iter_episodes += Int(e[i])

                total_steps += NUM_ENVS * ROLLOUT_LEN
                total_episodes += iter_episodes

                # Check GAE computed correctly (should have non-zero values)
                var adv_sample = Float64(adv[0])  # First env, first step
                var avg_ep_len = Float64(NUM_ENVS * ROLLOUT_LEN) / Float64(iter_episodes) if iter_episodes > 0 else 0.0

                print("  Update", iteration + 1, "| Episodes:", iter_episodes, "| Avg ep len:", avg_ep_len, "| Adv[0]:", adv_sample)

        print()
        print("Total steps:", total_steps)
        print("Total episodes:", total_episodes)

        # Verify GAE values look reasonable
        with rollout_advantages_buf.map_to_host() as adv, rollout_returns_buf.map_to_host() as ret:
            var adv_sum: Float64 = 0
            var ret_sum: Float64 = 0
            for i in range(min(NUM_ENVS * ROLLOUT_LEN, 1000)):
                adv_sum += Float64(adv[i])
                ret_sum += Float64(ret[i])
            print("Sample mean advantage:", adv_sum / 1000.0)
            print("Sample mean return:", ret_sum / 1000.0)

    print("=" * 55)
    print("Mega-kernel v2 test passed!")

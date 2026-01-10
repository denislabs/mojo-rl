"""Fused parallel kernel with 2D thread layout.

Thread organization:
- x dimension: environments (block_idx.x * block_dim.x + thread_idx.x)
- y dimension: hidden units (thread_idx.y)

This allows:
- Multiple envs per block (ENVS_PER_BLOCK)
- HIDDEN_DIM threads on y for parallel computation
- Total threads per block = ENVS_PER_BLOCK * HIDDEN_DIM (must be <= 1024)
"""

from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from math import exp, log, sqrt, cos, sin
from random import random_float64

comptime dtype = DType.float32

# =============================================================================
# Physics constants (CartPole)
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


@always_inline
fn inline_lcg_random(seed: UInt32) -> Tuple[UInt32, Scalar[dtype]]:
    var new_seed = seed * 1103515245 + 12345
    var val = Scalar[dtype](Float32(new_seed & 0x7FFFFFFF) / Float32(0x7FFFFFFF))
    return (new_seed, val)


# =============================================================================
# FUSED PARALLEL ROLLOUT KERNEL - 2D Thread Layout
# =============================================================================

fn fused_parallel_rollout_2d_kernel[
    NUM_ENVS: Int,
    OBS_DIM: Int,
    HIDDEN_DIM: Int,
    NUM_ACTIONS: Int,
    ROLLOUT_LEN: Int,
    ENVS_PER_BLOCK: Int,  # How many envs per block (on x dimension)
](
    # Environment states
    env_states: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, OBS_DIM), MutAnyOrigin],
    rng_states: LayoutTensor[DType.uint32, Layout.row_major(NUM_ENVS), MutAnyOrigin],
    # Network weights
    W1: LayoutTensor[dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), ImmutAnyOrigin],
    b1: LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin],
    W_actor: LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, NUM_ACTIONS), ImmutAnyOrigin],
    b_actor: LayoutTensor[dtype, Layout.row_major(NUM_ACTIONS), ImmutAnyOrigin],
    W_critic: LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, 1), ImmutAnyOrigin],
    b_critic: LayoutTensor[dtype, Layout.row_major(1), ImmutAnyOrigin],
    # Rollout storage
    rollout_obs: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN, OBS_DIM), MutAnyOrigin],
    rollout_actions: LayoutTensor[DType.int32, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin],
    rollout_rewards: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin],
    rollout_dones: LayoutTensor[DType.int32, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin],
    rollout_log_probs: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin],
    rollout_values: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin],
    # Output
    total_rewards: LayoutTensor[dtype, Layout.row_major(NUM_ENVS), MutAnyOrigin],
    episode_counts: LayoutTensor[DType.int32, Layout.row_major(NUM_ENVS), MutAnyOrigin],
):
    """Fused rollout with 2D thread layout.

    Standard 2D indexing convention (like mojo-gpu-puzzles):
    - y = rows (environments)
    - x = columns (hidden units)

    Grid: (1, NUM_ENVS // ENVS_PER_BLOCK)
    Block: (HIDDEN_DIM, ENVS_PER_BLOCK)
    """
    # Standard 2D indexing: y = rows (envs), x = cols (hidden)
    var local_env_idx = Int(thread_idx.y)   # local row
    var hidden_idx = Int(thread_idx.x)       # local col
    var global_env_idx = Int(block_dim.y * block_idx.y + thread_idx.y)
    var global_hidden_idx = Int(block_dim.x * block_idx.x + thread_idx.x)

    # Valid thread check (for guarding memory access, NOT barriers)
    var is_valid = global_env_idx < NUM_ENVS and global_hidden_idx < HIDDEN_DIM

    # Shared memory - organized per local env
    # [ENVS_PER_BLOCK, HIDDEN_DIM] for hidden activations
    var shared_hidden = LayoutTensor[
        dtype,
        Layout.row_major(ENVS_PER_BLOCK, HIDDEN_DIM),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # [ENVS_PER_BLOCK, NUM_ACTIONS] for logits
    var shared_logits = LayoutTensor[
        dtype,
        Layout.row_major(ENVS_PER_BLOCK, NUM_ACTIONS),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # [ENVS_PER_BLOCK, HIDDEN_DIM] for reduction scratch
    var shared_reduce = LayoutTensor[
        dtype,
        Layout.row_major(ENVS_PER_BLOCK, HIDDEN_DIM),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # [ENVS_PER_BLOCK, OBS_DIM] for observations
    var shared_obs = LayoutTensor[
        dtype,
        Layout.row_major(ENVS_PER_BLOCK, OBS_DIM),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # [ENVS_PER_BLOCK] for RNG states
    var shared_rng = LayoutTensor[
        DType.uint32,
        Layout.row_major(ENVS_PER_BLOCK),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # [ENVS_PER_BLOCK, 2] for stats (total_reward, episode_count)
    var shared_stats = LayoutTensor[
        dtype,
        Layout.row_major(ENVS_PER_BLOCK, 2),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Initialize shared state (only thread x=0 for each local env)
    if is_valid and hidden_idx == 0:
        for d in range(OBS_DIM):
            shared_obs[local_env_idx, d] = rebind[Scalar[dtype]](env_states[global_env_idx, d])
        shared_rng[local_env_idx] = rebind[Scalar[DType.uint32]](rng_states[global_env_idx])
        shared_stats[local_env_idx, 0] = Scalar[dtype](0)  # total_reward
        shared_stats[local_env_idx, 1] = Scalar[dtype](0)  # episode_count

        # DEBUG: Write to verify kernel executes
        total_rewards[global_env_idx] = Scalar[dtype](-1.0)

    barrier()

    # Run ROLLOUT_LEN steps
    for step in range(ROLLOUT_LEN):
        # Store observation (thread y=0 handles this for each local env)
        if hidden_idx < OBS_DIM:
            rollout_obs[global_env_idx, step, hidden_idx] = rebind[Scalar[dtype]](shared_obs[local_env_idx, hidden_idx])

        barrier()

        # =====================================================================
        # Forward pass - Layer 1: parallel across HIDDEN_DIM (y dimension)
        # Each thread (x, y) computes hidden[x, y]
        # =====================================================================
        var acc = rebind[Scalar[dtype]](b1[hidden_idx])
        for d in range(OBS_DIM):
            acc += rebind[Scalar[dtype]](shared_obs[local_env_idx, d]) * rebind[Scalar[dtype]](W1[d, hidden_idx])
        # ReLU
        shared_hidden[local_env_idx, hidden_idx] = acc if acc > Scalar[dtype](0) else Scalar[dtype](0)

        barrier()

        # =====================================================================
        # Actor logits - parallel reduction across y dimension
        # =====================================================================
        for action in range(NUM_ACTIONS):
            # Each thread computes: hidden[local_env, hidden_idx] * W_actor[hidden_idx, action]
            var partial = rebind[Scalar[dtype]](shared_hidden[local_env_idx, hidden_idx]) * rebind[Scalar[dtype]](W_actor[hidden_idx, action])
            shared_reduce[local_env_idx, hidden_idx] = partial

            barrier()

            # Tree reduction along y dimension
            var stride = HIDDEN_DIM // 2
            while stride > 0:
                if hidden_idx < stride:
                    shared_reduce[local_env_idx, hidden_idx] = (
                        rebind[Scalar[dtype]](shared_reduce[local_env_idx, hidden_idx]) +
                        rebind[Scalar[dtype]](shared_reduce[local_env_idx, hidden_idx + stride])
                    )
                barrier()
                stride = stride // 2

            # Thread y=0 adds bias and stores result
            if hidden_idx == 0:
                shared_logits[local_env_idx, action] = rebind[Scalar[dtype]](shared_reduce[local_env_idx, 0]) + rebind[Scalar[dtype]](b_actor[action])

            barrier()

        # =====================================================================
        # Softmax and sampling (thread y=0 for each env)
        # =====================================================================
        if hidden_idx == 0:
            var max_logit = rebind[Scalar[dtype]](shared_logits[local_env_idx, 0])
            for a in range(1, NUM_ACTIONS):
                var l = rebind[Scalar[dtype]](shared_logits[local_env_idx, a])
                if l > max_logit:
                    max_logit = l

            var sum_exp: Scalar[dtype] = 0
            for a in range(NUM_ACTIONS):
                var e = exp(rebind[Scalar[dtype]](shared_logits[local_env_idx, a]) - max_logit)
                shared_reduce[local_env_idx, a] = e  # Reuse scratch
                sum_exp += e

            var prob0 = rebind[Scalar[dtype]](shared_reduce[local_env_idx, 0]) / sum_exp
            var prob1 = rebind[Scalar[dtype]](shared_reduce[local_env_idx, 1]) / sum_exp

            # Sample action
            var rng = rebind[UInt32](shared_rng[local_env_idx])
            var rand_result = inline_lcg_random(rng)
            shared_rng[local_env_idx] = rebind[Scalar[DType.uint32]](rand_result[0])

            var selected_action = 0 if rand_result[1] < prob0 else 1
            var log_prob = log((prob0 if selected_action == 0 else prob1) + Scalar[dtype](1e-10))

            rollout_actions[global_env_idx, step] = Int32(selected_action)
            rollout_log_probs[global_env_idx, step] = log_prob

        barrier()

        # Read action (broadcast from thread y=0)
        var selected_action = Int(rollout_actions[global_env_idx, step])

        # =====================================================================
        # Critic value - parallel reduction
        # =====================================================================
        var partial_v = rebind[Scalar[dtype]](shared_hidden[local_env_idx, hidden_idx]) * rebind[Scalar[dtype]](W_critic[hidden_idx, 0])
        shared_reduce[local_env_idx, hidden_idx] = partial_v

        barrier()

        var stride_v = HIDDEN_DIM // 2
        while stride_v > 0:
            if hidden_idx < stride_v:
                shared_reduce[local_env_idx, hidden_idx] = (
                    rebind[Scalar[dtype]](shared_reduce[local_env_idx, hidden_idx]) +
                    rebind[Scalar[dtype]](shared_reduce[local_env_idx, hidden_idx + stride_v])
                )
            barrier()
            stride_v = stride_v // 2

        if hidden_idx == 0:
            rollout_values[global_env_idx, step] = rebind[Scalar[dtype]](shared_reduce[local_env_idx, 0]) + rebind[Scalar[dtype]](b_critic[0])

        barrier()

        # =====================================================================
        # Environment step (thread y=0 updates state)
        # =====================================================================
        if hidden_idx == 0:
            var x = rebind[Scalar[dtype]](shared_obs[local_env_idx, 0])
            var x_dot = rebind[Scalar[dtype]](shared_obs[local_env_idx, 1])
            var theta = rebind[Scalar[dtype]](shared_obs[local_env_idx, 2])
            var theta_dot = rebind[Scalar[dtype]](shared_obs[local_env_idx, 3])

            var force = Scalar[dtype](FORCE_MAG) if selected_action == 1 else Scalar[dtype](-FORCE_MAG)
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

            rollout_rewards[global_env_idx, step] = Scalar[dtype](1.0)
            rollout_dones[global_env_idx, step] = Int32(1) if done else Int32(0)

            shared_stats[local_env_idx, 0] = rebind[Scalar[dtype]](shared_stats[local_env_idx, 0]) + Scalar[dtype](1.0)

            if done:
                shared_stats[local_env_idx, 1] = rebind[Scalar[dtype]](shared_stats[local_env_idx, 1]) + Scalar[dtype](1.0)
                # Reset env
                var rng = rebind[UInt32](shared_rng[local_env_idx])
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
                shared_rng[local_env_idx] = rebind[Scalar[DType.uint32]](rng)

            # Update shared obs
            shared_obs[local_env_idx, 0] = x
            shared_obs[local_env_idx, 1] = x_dot
            shared_obs[local_env_idx, 2] = theta
            shared_obs[local_env_idx, 3] = theta_dot

        barrier()

    # Write final state back (thread y=0)
    if hidden_idx == 0:
        for d in range(OBS_DIM):
            env_states[global_env_idx, d] = rebind[Scalar[dtype]](shared_obs[local_env_idx, d])
        rng_states[global_env_idx] = rebind[Scalar[DType.uint32]](shared_rng[local_env_idx])
        total_rewards[global_env_idx] = rebind[Scalar[dtype]](shared_stats[local_env_idx, 0])
        episode_counts[global_env_idx] = Scalar[DType.int32](Int32(rebind[Scalar[dtype]](shared_stats[local_env_idx, 1])))


# =============================================================================
# Test
# =============================================================================

fn main() raises:
    print("Testing fused parallel kernel with 2D thread layout")
    print("=" * 60)

    comptime NUM_ENVS = 1024
    comptime OBS_DIM = 4
    comptime HIDDEN_DIM = 64   # x dimension (columns)
    comptime NUM_ACTIONS = 2
    comptime ROLLOUT_LEN = 128
    comptime ENVS_PER_BLOCK = 8  # y dimension (rows) - 8 for 512 threads max

    # Threads per block = HIDDEN_DIM * ENVS_PER_BLOCK = 64 * 8 = 512
    comptime THREADS_PER_BLOCK = HIDDEN_DIM * ENVS_PER_BLOCK
    comptime NUM_BLOCKS = NUM_ENVS // ENVS_PER_BLOCK

    print("Configuration:")
    print("  NUM_ENVS:", NUM_ENVS)
    print("  HIDDEN_DIM:", HIDDEN_DIM)
    print("  ENVS_PER_BLOCK:", ENVS_PER_BLOCK)
    print("  THREADS_PER_BLOCK:", THREADS_PER_BLOCK)
    print("  NUM_BLOCKS:", NUM_BLOCKS)
    print("  Block dim: (", HIDDEN_DIM, ",", ENVS_PER_BLOCK, ") = (x, y)")
    print()

    with DeviceContext() as ctx:
        # Allocate buffers
        var env_states_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * OBS_DIM)
        var rng_states_buf = ctx.enqueue_create_buffer[DType.uint32](NUM_ENVS)

        var W1_buf = ctx.enqueue_create_buffer[dtype](OBS_DIM * HIDDEN_DIM)
        var b1_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
        var W_actor_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM * NUM_ACTIONS)
        var b_actor_buf = ctx.enqueue_create_buffer[dtype](NUM_ACTIONS)
        var W_critic_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
        var b_critic_buf = ctx.enqueue_create_buffer[dtype](1)

        var rollout_obs_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * ROLLOUT_LEN * OBS_DIM)
        var rollout_actions_buf = ctx.enqueue_create_buffer[DType.int32](NUM_ENVS * ROLLOUT_LEN)
        var rollout_rewards_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * ROLLOUT_LEN)
        var rollout_dones_buf = ctx.enqueue_create_buffer[DType.int32](NUM_ENVS * ROLLOUT_LEN)
        var rollout_log_probs_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * ROLLOUT_LEN)
        var rollout_values_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * ROLLOUT_LEN)

        var total_rewards_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS)
        var episode_counts_buf = ctx.enqueue_create_buffer[DType.int32](NUM_ENVS)

        # Initialize weights
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
                h[i * OBS_DIM + 0] = Scalar[dtype]((random_float64() - 0.5) * 0.1)
                h[i * OBS_DIM + 1] = Scalar[dtype]((random_float64() - 0.5) * 0.1)
                h[i * OBS_DIM + 2] = Scalar[dtype]((random_float64() - 0.5) * 0.1)
                h[i * OBS_DIM + 3] = Scalar[dtype]((random_float64() - 0.5) * 0.1)

        with rng_states_buf.map_to_host() as h:
            for i in range(NUM_ENVS):
                h[i] = Scalar[DType.uint32](UInt32(i * 12345 + 1))

        # Create tensors
        var env_states = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, OBS_DIM), MutAnyOrigin](env_states_buf)
        var rng_states = LayoutTensor[DType.uint32, Layout.row_major(NUM_ENVS), MutAnyOrigin](rng_states_buf)
        var W1 = LayoutTensor[dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), ImmutAnyOrigin](W1_buf)
        var b1 = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin](b1_buf)
        var W_actor = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, NUM_ACTIONS), ImmutAnyOrigin](W_actor_buf)
        var b_actor = LayoutTensor[dtype, Layout.row_major(NUM_ACTIONS), ImmutAnyOrigin](b_actor_buf)
        var W_critic = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, 1), ImmutAnyOrigin](W_critic_buf)
        var b_critic = LayoutTensor[dtype, Layout.row_major(1), ImmutAnyOrigin](b_critic_buf)

        var rollout_obs = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN, OBS_DIM), MutAnyOrigin](rollout_obs_buf)
        var rollout_actions = LayoutTensor[DType.int32, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin](rollout_actions_buf)
        var rollout_rewards = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin](rollout_rewards_buf)
        var rollout_dones = LayoutTensor[DType.int32, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin](rollout_dones_buf)
        var rollout_log_probs = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin](rollout_log_probs_buf)
        var rollout_values = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin](rollout_values_buf)
        var total_rewards = LayoutTensor[dtype, Layout.row_major(NUM_ENVS), MutAnyOrigin](total_rewards_buf)
        var episode_counts = LayoutTensor[DType.int32, Layout.row_major(NUM_ENVS), MutAnyOrigin](episode_counts_buf)

        print("Running fused parallel 2D rollout kernel...")

        # Check initial state
        with env_states_buf.map_to_host() as s:
            print("Initial state env 0:", Float64(s[0]), Float64(s[1]), Float64(s[2]), Float64(s[3]))

        # Debug: check indexing
        print("NUM_BLOCKS:", NUM_BLOCKS, "ENVS_PER_BLOCK:", ENVS_PER_BLOCK)
        print("block 0, thread (0,0) -> global_env_idx = 0 * 16 + 0 = 0")
        print("block 0, thread (1,0) -> global_env_idx = 0 * 16 + 1 = 1")

        # Initialize total_rewards to 999 to check if kernel writes
        with total_rewards_buf.map_to_host() as tr:
            for i in range(NUM_ENVS):
                tr[i] = Scalar[dtype](999.0)
        print()

        from time import perf_counter_ns
        var start = perf_counter_ns()

        # Run 10 iterations
        for iteration in range(10):
            ctx.enqueue_function_checked[
                fused_parallel_rollout_2d_kernel[NUM_ENVS, OBS_DIM, HIDDEN_DIM, NUM_ACTIONS, ROLLOUT_LEN, ENVS_PER_BLOCK],
                fused_parallel_rollout_2d_kernel[NUM_ENVS, OBS_DIM, HIDDEN_DIM, NUM_ACTIONS, ROLLOUT_LEN, ENVS_PER_BLOCK],
            ](
                env_states, rng_states,
                W1, b1, W_actor, b_actor, W_critic, b_critic,
                rollout_obs, rollout_actions, rollout_rewards, rollout_dones,
                rollout_log_probs, rollout_values,
                total_rewards, episode_counts,
                # Standard convention: (x, y) = (columns, rows) = (hidden, envs)
                grid_dim=(1, NUM_BLOCKS),
                block_dim=(HIDDEN_DIM, ENVS_PER_BLOCK),
            )
            ctx.synchronize()

            with total_rewards_buf.map_to_host() as r, episode_counts_buf.map_to_host() as e, rollout_dones_buf.map_to_host() as d:
                var total_ep = 0
                var total_rew: Float64 = 0
                var total_dones = 0
                for i in range(NUM_ENVS):
                    total_ep += Int(e[i])
                    total_rew += Float64(r[i])
                for i in range(NUM_ENVS * ROLLOUT_LEN):
                    total_dones += Int(d[i])

                var avg_ep_len = total_rew / Float64(total_ep) if total_ep > 0 else 0.0
                print("  Iter", iteration + 1, "| Steps:", NUM_ENVS * ROLLOUT_LEN, "| Dones:", total_dones, "| Episodes:", total_ep, "| Avg len:", avg_ep_len)

            # Check state after iteration
            if iteration == 0:
                with env_states_buf.map_to_host() as s:
                    print("  State env 0 after iter 1:", Float64(s[0]), Float64(s[1]), Float64(s[2]), Float64(s[3]))
                with rollout_actions_buf.map_to_host() as a:
                    print("  First 10 actions env 0:", Int(a[0]), Int(a[1]), Int(a[2]), Int(a[3]), Int(a[4]), Int(a[5]), Int(a[6]), Int(a[7]), Int(a[8]), Int(a[9]))
                with rollout_rewards_buf.map_to_host() as rw:
                    print("  First 10 rewards env 0:", Float64(rw[0]), Float64(rw[1]), Float64(rw[2]))
                with total_rewards_buf.map_to_host() as tr:
                    print("  total_rewards[0:3]:", Float64(tr[0]), Float64(tr[1]), Float64(tr[2]), "(should be -1 if debug write worked)")

        var end = perf_counter_ns()
        var elapsed_ms = Float64(end - start) / 1e6
        var total_steps = 10 * NUM_ENVS * ROLLOUT_LEN
        var throughput = Float64(total_steps) / (Float64(end - start) / 1e9)

        print()
        print("Total steps:", total_steps)
        print("Time:", elapsed_ms, "ms")
        print("Throughput:", Int(throughput), "steps/sec")

    print("=" * 60)
    print("Fused parallel 2D kernel test completed!")

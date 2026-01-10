"""Test fused kernel with GPU parallelism INSIDE a mega-kernel.

Key idea: Block-per-Environment pattern
- Each thread BLOCK handles one environment
- Threads within the block COLLABORATE on matmul and reductions
- Shared memory holds intermediate results
- One mega-kernel does the entire forward + backward pass

This combines:
- Fused kernel (single launch like a2c.mojo)
- Parallel GPU operations (like a2c_native.mojo)
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


# =============================================================================
# Inline helper for LCG random
# =============================================================================

@always_inline
fn inline_lcg_random(seed: UInt32) -> Tuple[UInt32, Scalar[dtype]]:
    var new_seed = seed * 1103515245 + 12345
    var val = Scalar[dtype](Float32(new_seed & 0x7FFFFFFF) / Float32(0x7FFFFFFF))
    return (new_seed, val)


# =============================================================================
# FUSED PARALLEL FORWARD KERNEL
# Block-per-Environment: Each block handles one env, threads collaborate
# =============================================================================

fn fused_parallel_forward_kernel[
    NUM_ENVS: Int,
    OBS_DIM: Int,      # 4 for CartPole
    HIDDEN_DIM: Int,   # e.g., 64
    NUM_ACTIONS: Int,  # 2 for CartPole
](
    # Environment states (NUM_ENVS, OBS_DIM)
    obs: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, OBS_DIM), ImmutAnyOrigin],
    # Network weights
    W1: LayoutTensor[dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), ImmutAnyOrigin],
    b1: LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin],
    W_actor: LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, NUM_ACTIONS), ImmutAnyOrigin],
    b_actor: LayoutTensor[dtype, Layout.row_major(NUM_ACTIONS), ImmutAnyOrigin],
    W_critic: LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, 1), ImmutAnyOrigin],
    b_critic: LayoutTensor[dtype, Layout.row_major(1), ImmutAnyOrigin],
    # Outputs
    hidden_out: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin],
    logits_out: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), MutAnyOrigin],
    probs_out: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), MutAnyOrigin],
    values_out: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin],
):
    """Fused forward pass with parallel operations INSIDE one kernel.

    Grid: (NUM_ENVS,) - one block per environment
    Block: (HIDDEN_DIM,) - HIDDEN_DIM threads per block

    Each thread computes one hidden unit, then they collaborate for actor/critic.
    """
    var env_idx = Int(block_idx.x)  # Which environment this block handles
    var tid = Int(thread_idx.x)      # Thread ID within block (0 to HIDDEN_DIM-1)

    if env_idx >= NUM_ENVS:
        return

    # Shared memory for this block
    var shared_hidden = LayoutTensor[
        dtype,
        Layout.row_major(HIDDEN_DIM),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var shared_logits = LayoutTensor[
        dtype,
        Layout.row_major(NUM_ACTIONS),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var shared_reduce = LayoutTensor[
        dtype,
        Layout.row_major(HIDDEN_DIM),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # =========================================================================
    # LAYER 1: obs @ W1 + b1 -> ReLU -> hidden
    # Each thread computes one hidden unit (parallel across HIDDEN_DIM)
    # =========================================================================
    if tid < HIDDEN_DIM:
        var acc = rebind[Scalar[dtype]](b1[tid])
        # Dot product: sum over OBS_DIM
        for d in range(OBS_DIM):
            acc += rebind[Scalar[dtype]](obs[env_idx, d]) * rebind[Scalar[dtype]](W1[d, tid])
        # ReLU
        var h = acc if acc > Scalar[dtype](0) else Scalar[dtype](0)
        shared_hidden[tid] = h
        hidden_out[env_idx, tid] = h  # Store for backward pass

    barrier()

    # =========================================================================
    # ACTOR: hidden @ W_actor + b_actor -> logits
    # Use parallel reduction: each action's logit is a dot product of HIDDEN_DIM
    # Strategy: All threads compute partial sums, then reduce
    # =========================================================================

    # For each action, compute logit using parallel reduction
    for action in range(NUM_ACTIONS):
        # Each thread computes: hidden[tid] * W_actor[tid, action]
        var partial: Scalar[dtype] = 0
        if tid < HIDDEN_DIM:
            partial = rebind[Scalar[dtype]](shared_hidden[tid]) * rebind[Scalar[dtype]](W_actor[tid, action])
        shared_reduce[tid] = partial

        barrier()

        # Tree reduction to sum all partials
        var stride = HIDDEN_DIM // 2
        while stride > 0:
            if tid < stride:
                shared_reduce[tid] = rebind[Scalar[dtype]](shared_reduce[tid]) + rebind[Scalar[dtype]](shared_reduce[tid + stride])
            barrier()
            stride = stride // 2

        # Thread 0 adds bias and stores result
        if tid == 0:
            shared_logits[action] = rebind[Scalar[dtype]](shared_reduce[0]) + rebind[Scalar[dtype]](b_actor[action])

    barrier()

    # =========================================================================
    # SOFTMAX: Parallel reduction for max and sum
    # =========================================================================

    # Find max logit (using first NUM_ACTIONS threads, or reduce if large)
    var max_logit = rebind[Scalar[dtype]](shared_logits[0])
    if NUM_ACTIONS > 1:
        for a in range(1, NUM_ACTIONS):
            var l = rebind[Scalar[dtype]](shared_logits[a])
            if l > max_logit:
                max_logit = l

    # Compute exp(logit - max) and sum
    var sum_exp: Scalar[dtype] = 0
    if tid == 0:
        for a in range(NUM_ACTIONS):
            var e = exp(rebind[Scalar[dtype]](shared_logits[a]) - max_logit)
            shared_reduce[a] = e  # Store exp values temporarily
            sum_exp += e

        # Normalize to get probabilities
        for a in range(NUM_ACTIONS):
            var prob = rebind[Scalar[dtype]](shared_reduce[a]) / sum_exp
            probs_out[env_idx, a] = prob
            logits_out[env_idx, a] = rebind[Scalar[dtype]](shared_logits[a])

    barrier()

    # =========================================================================
    # CRITIC: hidden @ W_critic + b_critic -> value
    # Parallel reduction similar to actor
    # =========================================================================

    var partial_v: Scalar[dtype] = 0
    if tid < HIDDEN_DIM:
        partial_v = rebind[Scalar[dtype]](shared_hidden[tid]) * rebind[Scalar[dtype]](W_critic[tid, 0])
    shared_reduce[tid] = partial_v

    barrier()

    # Tree reduction
    var stride_v = HIDDEN_DIM // 2
    while stride_v > 0:
        if tid < stride_v:
            shared_reduce[tid] = rebind[Scalar[dtype]](shared_reduce[tid]) + rebind[Scalar[dtype]](shared_reduce[tid + stride_v])
        barrier()
        stride_v = stride_v // 2

    if tid == 0:
        values_out[env_idx, 0] = rebind[Scalar[dtype]](shared_reduce[0]) + rebind[Scalar[dtype]](b_critic[0])


# =============================================================================
# FUSED ROLLOUT KERNEL with parallel forward
# =============================================================================

fn fused_parallel_rollout_kernel[
    NUM_ENVS: Int,
    OBS_DIM: Int,
    HIDDEN_DIM: Int,
    NUM_ACTIONS: Int,
    ROLLOUT_LEN: Int,
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
    """Fused rollout with parallel forward pass.

    Grid: (NUM_ENVS,) - one block per environment
    Block: (HIDDEN_DIM,) - threads collaborate on forward pass
    """
    var env_idx = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    if env_idx >= NUM_ENVS:
        return

    # Shared memory
    var shared_hidden = LayoutTensor[
        dtype,
        Layout.row_major(HIDDEN_DIM),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var shared_logits = LayoutTensor[
        dtype,
        Layout.row_major(NUM_ACTIONS),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var shared_reduce = LayoutTensor[
        dtype,
        Layout.row_major(HIDDEN_DIM),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Shared env state (only thread 0 manages this)
    var shared_obs = LayoutTensor[
        dtype,
        Layout.row_major(OBS_DIM),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var shared_rng = LayoutTensor[
        DType.uint32,
        Layout.row_major(1),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var shared_stats = LayoutTensor[
        dtype,
        Layout.row_major(2),  # [total_reward, episode_count]
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Initialize shared state (thread 0 only)
    if tid == 0:
        for d in range(OBS_DIM):
            shared_obs[d] = rebind[Scalar[dtype]](env_states[env_idx, d])
        shared_rng[0] = rebind[Scalar[DType.uint32]](rng_states[env_idx])
        shared_stats[0] = Scalar[dtype](0)  # total_reward
        shared_stats[1] = Scalar[dtype](0)  # episode_count

    barrier()

    # Run ROLLOUT_LEN steps
    for step in range(ROLLOUT_LEN):
        # Store observation
        if tid < OBS_DIM:
            rollout_obs[env_idx, step, tid] = rebind[Scalar[dtype]](shared_obs[tid])

        barrier()

        # =====================================================================
        # Forward pass - Layer 1: parallel across HIDDEN_DIM threads
        # =====================================================================
        if tid < HIDDEN_DIM:
            var acc = rebind[Scalar[dtype]](b1[tid])
            for d in range(OBS_DIM):
                acc += rebind[Scalar[dtype]](shared_obs[d]) * rebind[Scalar[dtype]](W1[d, tid])
            # ReLU
            shared_hidden[tid] = acc if acc > Scalar[dtype](0) else Scalar[dtype](0)

        barrier()

        # =====================================================================
        # Actor logits - parallel reduction
        # =====================================================================
        for action in range(NUM_ACTIONS):
            var partial: Scalar[dtype] = 0
            if tid < HIDDEN_DIM:
                partial = rebind[Scalar[dtype]](shared_hidden[tid]) * rebind[Scalar[dtype]](W_actor[tid, action])
            shared_reduce[tid] = partial

            barrier()

            var stride = HIDDEN_DIM // 2
            while stride > 0:
                if tid < stride:
                    shared_reduce[tid] = rebind[Scalar[dtype]](shared_reduce[tid]) + rebind[Scalar[dtype]](shared_reduce[tid + stride])
                barrier()
                stride = stride // 2

            if tid == 0:
                shared_logits[action] = rebind[Scalar[dtype]](shared_reduce[0]) + rebind[Scalar[dtype]](b_actor[action])

        barrier()

        # =====================================================================
        # Softmax and sampling (thread 0)
        # =====================================================================
        var selected_action: Int = 0
        var log_prob: Scalar[dtype] = 0

        if tid == 0:
            var max_logit = rebind[Scalar[dtype]](shared_logits[0])
            for a in range(1, NUM_ACTIONS):
                var l = rebind[Scalar[dtype]](shared_logits[a])
                if l > max_logit:
                    max_logit = l

            var sum_exp: Scalar[dtype] = 0
            for a in range(NUM_ACTIONS):
                var e = exp(rebind[Scalar[dtype]](shared_logits[a]) - max_logit)
                shared_reduce[a] = e
                sum_exp += e

            # Normalize
            var prob0 = rebind[Scalar[dtype]](shared_reduce[0]) / sum_exp
            var prob1 = rebind[Scalar[dtype]](shared_reduce[1]) / sum_exp

            # Sample action
            var rng = rebind[UInt32](shared_rng[0])
            var rand_result = inline_lcg_random(rng)
            shared_rng[0] = rebind[Scalar[DType.uint32]](rand_result[0])

            selected_action = 0 if rand_result[1] < prob0 else 1
            log_prob = log((prob0 if selected_action == 0 else prob1) + Scalar[dtype](1e-10))

            rollout_actions[env_idx, step] = Int32(selected_action)
            rollout_log_probs[env_idx, step] = log_prob

        barrier()

        # Read action for all threads (from thread 0's decision stored in rollout)
        selected_action = Int(rollout_actions[env_idx, step])

        # =====================================================================
        # Critic value - parallel reduction
        # =====================================================================
        var partial_v: Scalar[dtype] = 0
        if tid < HIDDEN_DIM:
            partial_v = rebind[Scalar[dtype]](shared_hidden[tid]) * rebind[Scalar[dtype]](W_critic[tid, 0])
        shared_reduce[tid] = partial_v

        barrier()

        var stride_v = HIDDEN_DIM // 2
        while stride_v > 0:
            if tid < stride_v:
                shared_reduce[tid] = rebind[Scalar[dtype]](shared_reduce[tid]) + rebind[Scalar[dtype]](shared_reduce[tid + stride_v])
            barrier()
            stride_v = stride_v // 2

        if tid == 0:
            rollout_values[env_idx, step] = rebind[Scalar[dtype]](shared_reduce[0]) + rebind[Scalar[dtype]](b_critic[0])

        barrier()

        # =====================================================================
        # Environment step (thread 0 updates shared state)
        # =====================================================================
        if tid == 0:
            var x = rebind[Scalar[dtype]](shared_obs[0])
            var x_dot = rebind[Scalar[dtype]](shared_obs[1])
            var theta = rebind[Scalar[dtype]](shared_obs[2])
            var theta_dot = rebind[Scalar[dtype]](shared_obs[3])

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

            rollout_rewards[env_idx, step] = Scalar[dtype](1.0)
            rollout_dones[env_idx, step] = Int32(1) if done else Int32(0)

            shared_stats[0] = rebind[Scalar[dtype]](shared_stats[0]) + Scalar[dtype](1.0)

            if done:
                shared_stats[1] = rebind[Scalar[dtype]](shared_stats[1]) + Scalar[dtype](1.0)
                # Reset env
                var rng = rebind[UInt32](shared_rng[0])
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
                shared_rng[0] = rebind[Scalar[DType.uint32]](rng)

            # Update shared obs
            shared_obs[0] = x
            shared_obs[1] = x_dot
            shared_obs[2] = theta
            shared_obs[3] = theta_dot

        barrier()

    # Write final state back
    if tid == 0:
        for d in range(OBS_DIM):
            env_states[env_idx, d] = rebind[Scalar[dtype]](shared_obs[d])
        rng_states[env_idx] = rebind[Scalar[DType.uint32]](shared_rng[0])
        total_rewards[env_idx] = rebind[Scalar[dtype]](shared_stats[0])
        episode_counts[env_idx] = Scalar[DType.int32](Int32(rebind[Scalar[dtype]](shared_stats[1])))


# =============================================================================
# Test
# =============================================================================

fn main() raises:
    print("Testing fused kernel with parallel GPU operations")
    print("=" * 55)

    comptime NUM_ENVS = 256
    comptime OBS_DIM = 4
    comptime HIDDEN_DIM = 64  # Must be power of 2 for reduction
    comptime NUM_ACTIONS = 2
    comptime ROLLOUT_LEN = 128

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

        print("Running fused parallel rollout kernel...")
        print("  NUM_ENVS:", NUM_ENVS)
        print("  HIDDEN_DIM:", HIDDEN_DIM, "(threads per block)")
        print("  ROLLOUT_LEN:", ROLLOUT_LEN)
        print()

        from time import perf_counter_ns
        var start = perf_counter_ns()

        # Run 10 iterations
        for iteration in range(10):
            ctx.enqueue_function_checked[
                fused_parallel_rollout_kernel[NUM_ENVS, OBS_DIM, HIDDEN_DIM, NUM_ACTIONS, ROLLOUT_LEN],
                fused_parallel_rollout_kernel[NUM_ENVS, OBS_DIM, HIDDEN_DIM, NUM_ACTIONS, ROLLOUT_LEN],
            ](
                env_states, rng_states,
                W1, b1, W_actor, b_actor, W_critic, b_critic,
                rollout_obs, rollout_actions, rollout_rewards, rollout_dones,
                rollout_log_probs, rollout_values,
                total_rewards, episode_counts,
                grid_dim=(NUM_ENVS,),
                block_dim=(HIDDEN_DIM,),
            )
            ctx.synchronize()

            with total_rewards_buf.map_to_host() as r, episode_counts_buf.map_to_host() as e:
                var total_ep = 0
                var total_rew: Float64 = 0
                for i in range(NUM_ENVS):
                    total_ep += Int(e[i])
                    total_rew += Float64(r[i])

                var avg_ep_len = total_rew / Float64(total_ep) if total_ep > 0 else 0.0
                print("  Iter", iteration + 1, "| Steps:", NUM_ENVS * ROLLOUT_LEN, "| Episodes:", total_ep, "| Avg len:", avg_ep_len)

        var end = perf_counter_ns()
        var elapsed_ms = Float64(end - start) / 1e6
        var total_steps = 10 * NUM_ENVS * ROLLOUT_LEN
        var throughput = Float64(total_steps) / (Float64(end - start) / 1e9)

        print()
        print("Total steps:", total_steps)
        print("Time:", elapsed_ms, "ms")
        print("Throughput:", Int(throughput), "steps/sec")

    print("=" * 55)
    print("Fused parallel kernel test completed!")

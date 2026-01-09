"""Composable GPU RL Example v3 - Practical Pattern.

This example demonstrates the practical composable pattern for GPU RL:
1. Environment defines constants and inline functions (Section 1)
2. Algorithm kernel uses those constants directly (Section 2)
3. Training loop orchestrates everything (Section 3)

To create a new environment, copy Section 1 and modify the physics.
The algorithm (Section 2 & 3) remains unchanged!

Usage:
    pixi run -e apple mojo run examples/composable_gpu_rl_v3.mojo
"""

from time import perf_counter_ns
from math import exp, sqrt, cos, sin
from random import seed, random_float64

from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor


# =============================================================================
# SECTION 1: ENVIRONMENT DEFINITION (CARTPOLE)
# =============================================================================
# To create a new environment:
# 1. Copy this section to a new file or replace in place
# 2. Change the ENV_* constants to match your environment
# 3. Modify the step/reset/get_obs functions with your physics
# The algorithm code below stays the same!

# Environment dimensions (compile-time constants)
comptime ENV_OBS_DIM: Int = 4
comptime ENV_NUM_ACTIONS: Int = 2
comptime ENV_STATE_SIZE: Int = 4

# CartPole physics constants
comptime GRAVITY: Float64 = 9.8
comptime CART_MASS: Float64 = 1.0
comptime POLE_MASS: Float64 = 0.1
comptime TOTAL_MASS: Float64 = CART_MASS + POLE_MASS
comptime POLE_HALF_LENGTH: Float64 = 0.5
comptime POLE_MASS_LENGTH: Float64 = POLE_MASS * POLE_HALF_LENGTH
comptime FORCE_MAG: Float64 = 10.0
comptime TAU: Float64 = 0.02
comptime X_THRESHOLD: Float64 = 2.4
comptime THETA_THRESHOLD: Float64 = 0.2095
comptime INIT_RANGE: Float64 = 0.05


# GPU RNG utility
@always_inline
fn xorshift32(state: Scalar[DType.uint32]) -> Scalar[DType.uint32]:
    var x = state
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    return x


@always_inline
fn random_uniform[
    dtype: DType
](rng: Scalar[DType.uint32]) -> Tuple[Scalar[dtype], Scalar[DType.uint32]]:
    var new_rng = xorshift32(rng)
    var value = Scalar[dtype](new_rng) / Scalar[dtype](Scalar[DType.uint32].MAX)
    return (value, new_rng)


@always_inline
fn random_range[
    dtype: DType
](rng: Scalar[DType.uint32], low: Scalar[dtype], high: Scalar[dtype]) -> Tuple[
    Scalar[dtype], Scalar[DType.uint32]
]:
    var result = random_uniform[dtype](rng)
    var value = low + result[0] * (high - low)
    return (value, result[1])


@always_inline
fn env_step[
    dtype: DType
](
    mut state: InlineArray[Scalar[dtype], ENV_STATE_SIZE],
    action: Int,
    rng: Scalar[DType.uint32],
) -> Tuple[Scalar[dtype], Bool, Scalar[DType.uint32]]:
    """CartPole physics step."""
    var x = state[0]
    var x_dot = state[1]
    var theta = state[2]
    var theta_dot = state[3]

    var force_mag = Scalar[dtype](FORCE_MAG)
    var gravity = Scalar[dtype](GRAVITY)
    var total_mass = Scalar[dtype](TOTAL_MASS)
    var pole_half_length = Scalar[dtype](POLE_HALF_LENGTH)
    var pole_mass_length = Scalar[dtype](POLE_MASS_LENGTH)
    var pole_mass = Scalar[dtype](POLE_MASS)
    var tau = Scalar[dtype](TAU)
    var x_threshold = Scalar[dtype](X_THRESHOLD)
    var theta_threshold = Scalar[dtype](THETA_THRESHOLD)

    var force = force_mag if action == 1 else -force_mag

    var cos_theta = cos(theta)
    var sin_theta = sin(theta)

    var temp = (
        force + pole_mass_length * theta_dot * theta_dot * sin_theta
    ) / total_mass
    var theta_acc = (gravity * sin_theta - cos_theta * temp) / (
        pole_half_length
        * (
            Scalar[dtype](4.0 / 3.0)
            - pole_mass * cos_theta * cos_theta / total_mass
        )
    )
    var x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass

    x = x + tau * x_dot
    x_dot = x_dot + tau * x_acc
    theta = theta + tau * theta_dot
    theta_dot = theta_dot + tau * theta_acc

    state[0] = x
    state[1] = x_dot
    state[2] = theta
    state[3] = theta_dot

    var done = (
        (x < -x_threshold)
        or (x > x_threshold)
        or (theta < -theta_threshold)
        or (theta > theta_threshold)
    )

    var reward = Scalar[dtype](1.0) if not done else Scalar[dtype](0.0)
    return (reward, done, rng)


@always_inline
fn env_reset[
    dtype: DType
](
    mut state: InlineArray[Scalar[dtype], ENV_STATE_SIZE],
    rng: Scalar[DType.uint32],
) -> Scalar[DType.uint32]:
    """Reset CartPole to initial state."""
    var low = Scalar[dtype](-INIT_RANGE)
    var high = Scalar[dtype](INIT_RANGE)
    var current_rng = rng

    var r0 = random_range[dtype](current_rng, low, high)
    state[0] = r0[0]
    current_rng = r0[1]

    var r1 = random_range[dtype](current_rng, low, high)
    state[1] = r1[0]
    current_rng = r1[1]

    var r2 = random_range[dtype](current_rng, low, high)
    state[2] = r2[0]
    current_rng = r2[1]

    var r3 = random_range[dtype](current_rng, low, high)
    state[3] = r3[0]
    current_rng = r3[1]

    return current_rng


@always_inline
fn env_get_obs[
    dtype: DType
](
    state: InlineArray[Scalar[dtype], ENV_STATE_SIZE],
) -> InlineArray[
    Scalar[dtype], ENV_OBS_DIM
]:
    """Get observation (identity for CartPole)."""
    var obs = InlineArray[Scalar[dtype], ENV_OBS_DIM](fill=Scalar[dtype](0))
    obs[0] = state[0]
    obs[1] = state[1]
    obs[2] = state[2]
    obs[3] = state[3]
    return obs


# =============================================================================
# SECTION 2: GENERIC REINFORCE ALGORITHM
# =============================================================================
# This section is ENVIRONMENT AGNOSTIC - it only uses ENV_* constants
# and env_step/env_reset/env_get_obs functions defined above.


fn reinforce_kernel[
    dtype: DType,
    HIDDEN_DIM: Int,
    NUM_ENVS: Int,
    STEPS_PER_KERNEL: Int,
    TPB: Int,
](
    env_state: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS * ENV_STATE_SIZE), MutAnyOrigin
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
    W1: LayoutTensor[
        dtype, Layout.row_major(ENV_OBS_DIM * HIDDEN_DIM), ImmutAnyOrigin
    ],
    b1: LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin],
    W2: LayoutTensor[
        dtype, Layout.row_major(HIDDEN_DIM * ENV_NUM_ACTIONS), ImmutAnyOrigin
    ],
    b2: LayoutTensor[dtype, Layout.row_major(ENV_NUM_ACTIONS), ImmutAnyOrigin],
    grad_W1: LayoutTensor[
        dtype,
        Layout.row_major(NUM_ENVS * ENV_OBS_DIM * HIDDEN_DIM),
        MutAnyOrigin,
    ],
    grad_b1: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS * HIDDEN_DIM), MutAnyOrigin
    ],
    grad_W2: LayoutTensor[
        dtype,
        Layout.row_major(NUM_ENVS * HIDDEN_DIM * ENV_NUM_ACTIONS),
        MutAnyOrigin,
    ],
    grad_b2: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS * ENV_NUM_ACTIONS), MutAnyOrigin
    ],
    gamma: Scalar[dtype],
):
    """REINFORCE training kernel - environment functions are inlined."""
    var env_idx = Int(block_dim.x * block_idx.x + thread_idx.x)

    if env_idx >= NUM_ENVS:
        return

    # Load environment state
    var state = InlineArray[Scalar[dtype], ENV_STATE_SIZE](
        fill=Scalar[dtype](0)
    )
    for i in range(ENV_STATE_SIZE):
        state[i] = rebind[Scalar[dtype]](
            env_state[env_idx * ENV_STATE_SIZE + i]
        )

    var rng = rebind[Scalar[DType.uint32]](rng_state[env_idx])
    var ep_length = Int(episode_lengths[env_idx])
    var num_episodes = Int(total_episodes[env_idx])

    # Load policy weights
    var w1 = InlineArray[Scalar[dtype], ENV_OBS_DIM * HIDDEN_DIM](
        fill=Scalar[dtype](0)
    )
    var bias1 = InlineArray[Scalar[dtype], HIDDEN_DIM](fill=Scalar[dtype](0))
    var w2 = InlineArray[Scalar[dtype], HIDDEN_DIM * ENV_NUM_ACTIONS](
        fill=Scalar[dtype](0)
    )
    var bias2 = InlineArray[Scalar[dtype], ENV_NUM_ACTIONS](
        fill=Scalar[dtype](0)
    )

    for i in range(ENV_OBS_DIM * HIDDEN_DIM):
        w1[i] = rebind[Scalar[dtype]](W1[i])
    for i in range(HIDDEN_DIM):
        bias1[i] = rebind[Scalar[dtype]](b1[i])
    for i in range(HIDDEN_DIM * ENV_NUM_ACTIONS):
        w2[i] = rebind[Scalar[dtype]](W2[i])
    for i in range(ENV_NUM_ACTIONS):
        bias2[i] = rebind[Scalar[dtype]](b2[i])

    # Trajectory storage
    var traj_obs = InlineArray[Scalar[dtype], STEPS_PER_KERNEL * ENV_OBS_DIM](
        fill=Scalar[dtype](0)
    )
    var traj_actions = InlineArray[Int, STEPS_PER_KERNEL](fill=0)
    var traj_rewards = InlineArray[Scalar[dtype], STEPS_PER_KERNEL](
        fill=Scalar[dtype](0)
    )
    var traj_dones = InlineArray[Bool, STEPS_PER_KERNEL](fill=False)
    var traj_length = 0

    # Run STEPS_PER_KERNEL steps
    for step in range(STEPS_PER_KERNEL):
        # Get observation (uses env_get_obs from Section 1)
        var obs = env_get_obs[dtype](state)

        for i in range(ENV_OBS_DIM):
            traj_obs[step * ENV_OBS_DIM + i] = obs[i]

        # Forward pass
        var h = InlineArray[Scalar[dtype], HIDDEN_DIM](fill=Scalar[dtype](0))
        for j in range(HIDDEN_DIM):
            var sum_val = bias1[j]
            for i in range(ENV_OBS_DIM):
                sum_val += obs[i] * w1[i * HIDDEN_DIM + j]
            h[j] = sum_val if sum_val > Scalar[dtype](0) else Scalar[dtype](0)

        var logits = InlineArray[Scalar[dtype], ENV_NUM_ACTIONS](
            fill=Scalar[dtype](0)
        )
        for j in range(ENV_NUM_ACTIONS):
            var sum_val = bias2[j]
            for k in range(HIDDEN_DIM):
                sum_val += h[k] * w2[k * ENV_NUM_ACTIONS + j]
            logits[j] = sum_val

        # Softmax
        var max_logit = logits[0]
        for j in range(1, ENV_NUM_ACTIONS):
            if logits[j] > max_logit:
                max_logit = logits[j]

        var exp_logits = InlineArray[Scalar[dtype], ENV_NUM_ACTIONS](
            fill=Scalar[dtype](0)
        )
        var sum_exp: Scalar[dtype] = 0
        for j in range(ENV_NUM_ACTIONS):
            exp_logits[j] = exp(logits[j] - max_logit)
            sum_exp += exp_logits[j]

        var probs = InlineArray[Scalar[dtype], ENV_NUM_ACTIONS](
            fill=Scalar[dtype](0)
        )
        for j in range(ENV_NUM_ACTIONS):
            probs[j] = exp_logits[j] / sum_exp

        # Sample action
        var u_result = random_uniform[dtype](rng)
        var u = u_result[0]
        rng = u_result[1]

        var action = 0
        var cumsum: Scalar[dtype] = 0
        for j in range(ENV_NUM_ACTIONS):
            cumsum += probs[j]
            if u < cumsum:
                action = j
                break
            action = j

        traj_actions[step] = action

        # Environment step (uses env_step from Section 1)
        var step_result = env_step[dtype](state, action, rng)
        var reward = step_result[0]
        var done = step_result[1]
        rng = step_result[2]

        traj_rewards[step] = reward
        traj_dones[step] = done
        ep_length += 1
        traj_length = step + 1

        # Reset if done (uses env_reset from Section 1)
        if done:
            num_episodes += 1
            rng = env_reset[dtype](state, rng)
            ep_length = 0

    # Compute returns
    var returns = InlineArray[Scalar[dtype], STEPS_PER_KERNEL](
        fill=Scalar[dtype](0)
    )
    var G: Scalar[dtype] = 0

    for step in range(traj_length - 1, -1, -1):
        if traj_dones[step]:
            G = Scalar[dtype](0)
        G = traj_rewards[step] + gamma * G
        returns[step] = G

    # Compute gradients
    var local_grad_W1 = InlineArray[Scalar[dtype], ENV_OBS_DIM * HIDDEN_DIM](
        fill=Scalar[dtype](0)
    )
    var local_grad_b1 = InlineArray[Scalar[dtype], HIDDEN_DIM](
        fill=Scalar[dtype](0)
    )
    var local_grad_W2 = InlineArray[
        Scalar[dtype], HIDDEN_DIM * ENV_NUM_ACTIONS
    ](fill=Scalar[dtype](0))
    var local_grad_b2 = InlineArray[Scalar[dtype], ENV_NUM_ACTIONS](
        fill=Scalar[dtype](0)
    )

    for step in range(traj_length):
        var ret = returns[step]
        var action = traj_actions[step]

        var obs = InlineArray[Scalar[dtype], ENV_OBS_DIM](fill=Scalar[dtype](0))
        for i in range(ENV_OBS_DIM):
            obs[i] = traj_obs[step * ENV_OBS_DIM + i]

        var h = InlineArray[Scalar[dtype], HIDDEN_DIM](fill=Scalar[dtype](0))
        var h_pre = InlineArray[Scalar[dtype], HIDDEN_DIM](
            fill=Scalar[dtype](0)
        )
        for j in range(HIDDEN_DIM):
            var sum_val = bias1[j]
            for i in range(ENV_OBS_DIM):
                sum_val += obs[i] * w1[i * HIDDEN_DIM + j]
            h_pre[j] = sum_val
            h[j] = sum_val if sum_val > Scalar[dtype](0) else Scalar[dtype](0)

        var logits = InlineArray[Scalar[dtype], ENV_NUM_ACTIONS](
            fill=Scalar[dtype](0)
        )
        for j in range(ENV_NUM_ACTIONS):
            var sum_val = bias2[j]
            for k in range(HIDDEN_DIM):
                sum_val += h[k] * w2[k * ENV_NUM_ACTIONS + j]
            logits[j] = sum_val

        var max_logit = logits[0]
        for j in range(1, ENV_NUM_ACTIONS):
            if logits[j] > max_logit:
                max_logit = logits[j]

        var exp_logits = InlineArray[Scalar[dtype], ENV_NUM_ACTIONS](
            fill=Scalar[dtype](0)
        )
        var sum_exp: Scalar[dtype] = 0
        for j in range(ENV_NUM_ACTIONS):
            exp_logits[j] = exp(logits[j] - max_logit)
            sum_exp += exp_logits[j]

        var probs = InlineArray[Scalar[dtype], ENV_NUM_ACTIONS](
            fill=Scalar[dtype](0)
        )
        for j in range(ENV_NUM_ACTIONS):
            probs[j] = exp_logits[j] / sum_exp

        var d_logits = InlineArray[Scalar[dtype], ENV_NUM_ACTIONS](
            fill=Scalar[dtype](0)
        )
        for j in range(ENV_NUM_ACTIONS):
            d_logits[j] = (
                Scalar[dtype](1.0) if j == action else Scalar[dtype](0.0)
            ) - probs[j]
            d_logits[j] = d_logits[j] * ret

        for k in range(HIDDEN_DIM):
            for j in range(ENV_NUM_ACTIONS):
                local_grad_W2[k * ENV_NUM_ACTIONS + j] += h[k] * d_logits[j]

        for j in range(ENV_NUM_ACTIONS):
            local_grad_b2[j] += d_logits[j]

        for k in range(HIDDEN_DIM):
            var dh: Scalar[dtype] = 0
            for j in range(ENV_NUM_ACTIONS):
                dh += d_logits[j] * w2[k * ENV_NUM_ACTIONS + j]
            var dh_pre = dh if h_pre[k] > Scalar[dtype](0) else Scalar[dtype](0)

            for i in range(ENV_OBS_DIM):
                local_grad_W1[i * HIDDEN_DIM + k] += obs[i] * dh_pre

            local_grad_b1[k] += dh_pre

    # Write back
    for i in range(ENV_STATE_SIZE):
        env_state[env_idx * ENV_STATE_SIZE + i] = state[i]

    rng_state[env_idx] = rng
    episode_lengths[env_idx] = Int32(ep_length)
    total_episodes[env_idx] = Int32(num_episodes)

    for i in range(ENV_OBS_DIM * HIDDEN_DIM):
        grad_W1[env_idx * ENV_OBS_DIM * HIDDEN_DIM + i] = local_grad_W1[i]
    for i in range(HIDDEN_DIM):
        grad_b1[env_idx * HIDDEN_DIM + i] = local_grad_b1[i]
    for i in range(HIDDEN_DIM * ENV_NUM_ACTIONS):
        grad_W2[env_idx * HIDDEN_DIM * ENV_NUM_ACTIONS + i] = local_grad_W2[i]
    for i in range(ENV_NUM_ACTIONS):
        grad_b2[env_idx * ENV_NUM_ACTIONS + i] = local_grad_b2[i]


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
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SIZE:
        return

    var sum_val: Scalar[dtype] = 0
    for env in range(NUM_ENVS):
        sum_val += rebind[Scalar[dtype]](per_env[env * SIZE + idx])

    reduced[idx] = sum_val / Scalar[dtype](NUM_ENVS)


fn sgd_update_kernel[
    dtype: DType,
    SIZE: Int,
    TPB: Int,
](
    weights: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    gradients: LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin],
    lr: Scalar[dtype],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SIZE:
        return

    var w = rebind[Scalar[dtype]](weights[idx])
    var g = rebind[Scalar[dtype]](gradients[idx])
    weights[idx] = w + lr * g


# =============================================================================
# SECTION 3: TRAINING LOOP
# =============================================================================


def main():
    seed(42)

    print("=" * 70)
    print("Composable GPU RL v3 - CartPole + REINFORCE")
    print("=" * 70)
    print()
    print("This demonstrates the COMPOSABLE pattern:")
    print(
        "  - Section 1: Environment definition (constants + inline functions)"
    )
    print("  - Section 2: Generic REINFORCE algorithm (uses ENV_* constants)")
    print("  - Section 3: Training loop (environment-agnostic)")
    print()
    print("To use a different environment:")
    print("  1. Replace Section 1 with your environment's physics")
    print("  2. Update ENV_OBS_DIM, ENV_NUM_ACTIONS, ENV_STATE_SIZE")
    print("  3. Implement env_step, env_reset, env_get_obs")
    print("  4. Sections 2 and 3 work unchanged!")
    print()

    # Configuration
    comptime dtype = DType.float32
    comptime HIDDEN_DIM = 32
    comptime NUM_ENVS = 1024
    comptime STEPS_PER_KERNEL = 200
    comptime TPB = 256

    var num_updates = 100
    var lr = Float32(0.01)
    var gamma = Float32(0.99)

    comptime W1_SIZE = ENV_OBS_DIM * HIDDEN_DIM
    comptime W2_SIZE = HIDDEN_DIM * ENV_NUM_ACTIONS

    with DeviceContext() as ctx:
        print("Allocating GPU buffers...")

        # Allocate buffers
        var env_state = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * ENV_STATE_SIZE
        )
        var rng_state = ctx.enqueue_create_buffer[DType.uint32](NUM_ENVS)
        var episode_lengths = ctx.enqueue_create_buffer[DType.int32](NUM_ENVS)
        var total_episodes_buf = ctx.enqueue_create_buffer[DType.int32](
            NUM_ENVS
        )

        var W1 = ctx.enqueue_create_buffer[dtype](W1_SIZE)
        var b1 = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
        var W2 = ctx.enqueue_create_buffer[dtype](W2_SIZE)
        var b2 = ctx.enqueue_create_buffer[dtype](ENV_NUM_ACTIONS)

        var grad_W1_per_env = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * W1_SIZE
        )
        var grad_b1_per_env = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * HIDDEN_DIM
        )
        var grad_W2_per_env = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * W2_SIZE
        )
        var grad_b2_per_env = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * ENV_NUM_ACTIONS
        )

        var grad_W1 = ctx.enqueue_create_buffer[dtype](W1_SIZE)
        var grad_b1 = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
        var grad_W2 = ctx.enqueue_create_buffer[dtype](W2_SIZE)
        var grad_b2 = ctx.enqueue_create_buffer[dtype](ENV_NUM_ACTIONS)

        # Initialize
        with env_state.map_to_host() as host:
            for i in range(NUM_ENVS * ENV_STATE_SIZE):
                host[i] = Scalar[dtype]((random_float64() - 0.5) * 0.1)

        with rng_state.map_to_host() as host:
            for i in range(NUM_ENVS):
                host[i] = UInt32(i + 12345)

        episode_lengths.enqueue_fill(0)
        total_episodes_buf.enqueue_fill(0)

        var std1 = sqrt(2.0 / Float64(ENV_OBS_DIM + HIDDEN_DIM))
        var std2 = sqrt(2.0 / Float64(HIDDEN_DIM + ENV_NUM_ACTIONS))

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
            dtype, Layout.row_major(NUM_ENVS * ENV_STATE_SIZE), MutAnyOrigin
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

        var W1_t = LayoutTensor[
            dtype, Layout.row_major(W1_SIZE), ImmutAnyOrigin
        ](W1)
        var b1_t = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin
        ](b1)
        var W2_t = LayoutTensor[
            dtype, Layout.row_major(W2_SIZE), ImmutAnyOrigin
        ](W2)
        var b2_t = LayoutTensor[
            dtype, Layout.row_major(ENV_NUM_ACTIONS), ImmutAnyOrigin
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
            dtype, Layout.row_major(NUM_ENVS * ENV_NUM_ACTIONS), MutAnyOrigin
        ](grad_b2_per_env)

        var gW1_t = LayoutTensor[
            dtype, Layout.row_major(W1_SIZE), MutAnyOrigin
        ](grad_W1)
        var gb1_t = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM), MutAnyOrigin
        ](grad_b1)
        var gW2_t = LayoutTensor[
            dtype, Layout.row_major(W2_SIZE), MutAnyOrigin
        ](grad_W2)
        var gb2_t = LayoutTensor[
            dtype, Layout.row_major(ENV_NUM_ACTIONS), MutAnyOrigin
        ](grad_b2)

        var W1_mut = LayoutTensor[
            dtype, Layout.row_major(W1_SIZE), MutAnyOrigin
        ](W1)
        var b1_mut = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM), MutAnyOrigin
        ](b1)
        var W2_mut = LayoutTensor[
            dtype, Layout.row_major(W2_SIZE), MutAnyOrigin
        ](W2)
        var b2_mut = LayoutTensor[
            dtype, Layout.row_major(ENV_NUM_ACTIONS), MutAnyOrigin
        ](b2)

        var gamma_s = Scalar[dtype](gamma)
        var lr_s = Scalar[dtype](lr)

        comptime num_blocks = (NUM_ENVS + TPB - 1) // TPB
        comptime main_kernel = reinforce_kernel[
            dtype,
            HIDDEN_DIM,
            NUM_ENVS,
            STEPS_PER_KERNEL,
            TPB,
        ]

        comptime blocks_W1 = (W1_SIZE + TPB - 1) // TPB
        comptime blocks_b1 = (HIDDEN_DIM + TPB - 1) // TPB
        comptime blocks_W2 = (W2_SIZE + TPB - 1) // TPB
        comptime blocks_b2 = (ENV_NUM_ACTIONS + TPB - 1) // TPB

        print("Starting training...")
        print("  ENV_OBS_DIM: " + String(ENV_OBS_DIM))
        print("  ENV_NUM_ACTIONS: " + String(ENV_NUM_ACTIONS))
        print("  HIDDEN_DIM: " + String(HIDDEN_DIM))
        print("  NUM_ENVS: " + String(NUM_ENVS))
        print("  STEPS_PER_KERNEL: " + String(STEPS_PER_KERNEL))
        print()

        var start_time = perf_counter_ns()

        for update in range(num_updates):
            grad_W1_per_env.enqueue_fill(0)
            grad_b1_per_env.enqueue_fill(0)
            grad_W2_per_env.enqueue_fill(0)
            grad_b2_per_env.enqueue_fill(0)

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
                dtype,
                Layout.row_major(NUM_ENVS * ENV_NUM_ACTIONS),
                ImmutAnyOrigin,
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
                dtype, NUM_ENVS, ENV_NUM_ACTIONS, TPB
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
                dtype, Layout.row_major(ENV_NUM_ACTIONS), ImmutAnyOrigin
            ](grad_b2)

            comptime sgd_W1 = sgd_update_kernel[dtype, W1_SIZE, TPB]
            comptime sgd_b1 = sgd_update_kernel[dtype, HIDDEN_DIM, TPB]
            comptime sgd_W2 = sgd_update_kernel[dtype, W2_SIZE, TPB]
            comptime sgd_b2 = sgd_update_kernel[dtype, ENV_NUM_ACTIONS, TPB]

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

            if (update + 1) % 10 == 0:
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

        print()
        print("=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print("  Total steps: " + String(total_steps))
        print("  Total episodes: " + String(total_eps))
        print("  Time: " + String(elapsed_sec)[:6] + " seconds")
        print("  Throughput: " + String(Int(steps_per_sec)) + " steps/sec")
        print()
        print("COMPOSABLE DESIGN SUMMARY:")
        print("  This file is structured in 3 sections:")
        print(
            "  1. ENVIRONMENT: ENV_* constants + env_step/reset/get_obs"
            " functions"
        )
        print("  2. ALGORITHM: Generic REINFORCE kernel using ENV_* constants")
        print("  3. TRAINING: Training loop (environment-agnostic)")
        print()
        print("  To add a new environment (e.g., MountainCar, Pendulum):")
        print("  - Copy this file")
        print("  - Replace Section 1 with your environment's physics")
        print("  - Sections 2 and 3 work unchanged!")
        print("=" * 70)

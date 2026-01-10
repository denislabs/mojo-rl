"""Test 2D kernel with same tensor layout as fused kernel but minimal logic."""

from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from random import random_float64
from math import sqrt, exp, log

comptime dtype = DType.float32


@always_inline
fn inline_lcg_random(seed: UInt32) -> Tuple[UInt32, Scalar[dtype]]:
    var new_seed = seed * 1103515245 + 12345
    var val = Scalar[dtype](
        Float32(new_seed & 0x7FFFFFFF) / Float32(0x7FFFFFFF)
    )
    return (new_seed, val)


fn simple_2d_kernel[
    NUM_ENVS: Int,
    HIDDEN_DIM: Int,
    ENVS_PER_BLOCK: Int,
    OBS_DIM: Int,
    NUM_ACTIONS: Int,
    ROLLOUT_LEN: Int,
](
    # Many parameters like the real kernel
    env_states: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, OBS_DIM), MutAnyOrigin
    ],
    rng_states: LayoutTensor[
        DType.uint32, Layout.row_major(NUM_ENVS), MutAnyOrigin
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
    total_rewards: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS), MutAnyOrigin
    ],
    episode_counts: LayoutTensor[
        DType.int32, Layout.row_major(NUM_ENVS), MutAnyOrigin
    ],
):
    # Standard 2D indexing convention (like mojo-gpu-puzzles):
    # y = rows (environments), x = columns (hidden units)
    var local_env_idx = Int(thread_idx.y)  # local row
    var hidden_idx = Int(thread_idx.x)  # local col
    var global_env_idx = Int(block_dim.y * block_idx.y + thread_idx.y)
    var global_hidden_idx = Int(block_dim.x * block_idx.x + thread_idx.x)

    # Shared memory - allocated by ALL threads (outside guard)
    var shared_hidden = LayoutTensor[
        dtype,
        Layout.row_major(ENVS_PER_BLOCK, HIDDEN_DIM),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var shared_obs = LayoutTensor[
        dtype,
        Layout.row_major(ENVS_PER_BLOCK, OBS_DIM),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var shared_reduce = LayoutTensor[
        dtype,
        Layout.row_major(ENVS_PER_BLOCK, HIDDEN_DIM),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var shared_logits = LayoutTensor[
        dtype,
        Layout.row_major(ENVS_PER_BLOCK, NUM_ACTIONS),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var shared_rng = LayoutTensor[
        DType.uint32,
        Layout.row_major(ENVS_PER_BLOCK),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var shared_stats = LayoutTensor[
        dtype,
        Layout.row_major(ENVS_PER_BLOCK, 2),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Valid thread check (for guarding memory access, NOT barriers)
    var is_valid = global_env_idx < NUM_ENVS and global_hidden_idx < HIDDEN_DIM

    # Initialize - guard memory access, NOT barrier
    if is_valid and hidden_idx == 0:
        for d in range(OBS_DIM):
            shared_obs[local_env_idx, d] = rebind[Scalar[dtype]](
                env_states[global_env_idx, d]
            )
        shared_rng[local_env_idx] = rebind[Scalar[DType.uint32]](
            rng_states[global_env_idx]
        )

    barrier()  # ALL threads hit this

    # Run loop
    for step in range(ROLLOUT_LEN):
        # Forward pass - guard memory access only
        if is_valid:
            var acc = rebind[Scalar[dtype]](b1[hidden_idx])
            for d in range(OBS_DIM):
                acc += rebind[Scalar[dtype]](
                    shared_obs[local_env_idx, d]
                ) * rebind[Scalar[dtype]](W1[d, hidden_idx])
            shared_hidden[local_env_idx, hidden_idx] = acc if acc > Scalar[dtype](0) else Scalar[dtype](0)

        barrier()  # ALL threads hit this

        # Actor logits with parallel reduction
        for action in range(NUM_ACTIONS):
            if is_valid:
                var partial = rebind[Scalar[dtype]](
                    shared_hidden[local_env_idx, hidden_idx]
                ) * rebind[Scalar[dtype]](W_actor[hidden_idx, action])
                shared_reduce[local_env_idx, hidden_idx] = partial

            barrier()  # ALL threads hit this

            # Tree reduction
            var stride = HIDDEN_DIM // 2
            while stride > 0:
                if is_valid and hidden_idx < stride:
                    shared_reduce[local_env_idx, hidden_idx] = rebind[
                        Scalar[dtype]
                    ](shared_reduce[local_env_idx, hidden_idx]) + rebind[
                        Scalar[dtype]
                    ](shared_reduce[local_env_idx, hidden_idx + stride])
                barrier()  # ALL threads hit this
                stride = stride // 2

            if is_valid and hidden_idx == 0:
                shared_logits[local_env_idx, action] = rebind[
                    Scalar[dtype]
                ](shared_reduce[local_env_idx, 0]) + rebind[Scalar[dtype]](
                    b_actor[action]
                )

            barrier()  # ALL threads hit this

        # Variables for sampling
        var selected_action: Int = 0
        var log_prob: Scalar[dtype] = 0

        # Softmax and sampling (thread 0 per env)
        if is_valid and hidden_idx == 0:
            var max_logit = rebind[Scalar[dtype]](shared_logits[local_env_idx, 0])
            for a in range(1, NUM_ACTIONS):
                var l = rebind[Scalar[dtype]](shared_logits[local_env_idx, a])
                if l > max_logit:
                    max_logit = l

            var sum_exp: Scalar[dtype] = 0
            for a in range(NUM_ACTIONS):
                var e = exp(rebind[Scalar[dtype]](shared_logits[local_env_idx, a]) - max_logit)
                shared_reduce[local_env_idx, a] = e
                sum_exp += e

            var prob0 = rebind[Scalar[dtype]](shared_reduce[local_env_idx, 0]) / (sum_exp + Scalar[dtype](1e-10))
            var prob1 = rebind[Scalar[dtype]](shared_reduce[local_env_idx, 1]) / (sum_exp + Scalar[dtype](1e-10))

            var rng = rebind[UInt32](shared_rng[local_env_idx])
            var rand_result = inline_lcg_random(rng)
            shared_rng[local_env_idx] = rebind[Scalar[DType.uint32]](rand_result[0])

            selected_action = 0 if rand_result[1] < prob0 else 1
            log_prob = log((prob0 if selected_action == 0 else prob1) + Scalar[dtype](1e-10))

            rollout_actions[global_env_idx, step] = Int32(selected_action)
            rollout_log_probs[global_env_idx, step] = log_prob

        barrier()  # ALL threads hit this

        # Write rewards
        if is_valid and hidden_idx == 0:
            rollout_rewards[global_env_idx, step] = Scalar[dtype](1.0)

        barrier()  # ALL threads hit this

    # Final write
    if is_valid and hidden_idx == 0:
        total_rewards[global_env_idx] = Scalar[dtype](Float32(global_env_idx))


fn main() raises:
    print("Testing 2D write-only kernel with many parameters")
    print("=" * 50)

    comptime NUM_ENVS = 1024
    comptime HIDDEN_DIM = 64
    comptime ENVS_PER_BLOCK = 8  # 8 envs per block = 512 threads
    comptime NUM_BLOCKS = NUM_ENVS // ENVS_PER_BLOCK
    comptime OBS_DIM = 4
    comptime NUM_ACTIONS = 2
    comptime ROLLOUT_LEN = 128

    print("NUM_ENVS:", NUM_ENVS)
    print("Parameters: 16 tensors")
    print()

    with DeviceContext() as ctx:
        var env_states_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * OBS_DIM
        )
        var rng_states_buf = ctx.enqueue_create_buffer[DType.uint32](NUM_ENVS)
        var W1_buf = ctx.enqueue_create_buffer[dtype](OBS_DIM * HIDDEN_DIM)
        var b1_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
        var W_actor_buf = ctx.enqueue_create_buffer[dtype](
            HIDDEN_DIM * NUM_ACTIONS
        )
        var b_actor_buf = ctx.enqueue_create_buffer[dtype](NUM_ACTIONS)
        var W_critic_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
        var b_critic_buf = ctx.enqueue_create_buffer[dtype](1)
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
        var total_rewards_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS)
        var episode_counts_buf = ctx.enqueue_create_buffer[DType.int32](
            NUM_ENVS
        )

        with total_rewards_buf.map_to_host() as h:
            for i in range(NUM_ENVS):
                h[i] = Scalar[dtype](-999.0)

        var env_states = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, OBS_DIM), MutAnyOrigin
        ](env_states_buf)
        var rng_states = LayoutTensor[
            DType.uint32, Layout.row_major(NUM_ENVS), MutAnyOrigin
        ](rng_states_buf)
        var W1 = LayoutTensor[
            dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), ImmutAnyOrigin
        ](W1_buf)
        var b1 = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin
        ](b1_buf)
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
        var rollout_obs = LayoutTensor[
            dtype,
            Layout.row_major(NUM_ENVS, ROLLOUT_LEN, OBS_DIM),
            MutAnyOrigin,
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
        var total_rewards = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS), MutAnyOrigin
        ](total_rewards_buf)
        var episode_counts = LayoutTensor[
            DType.int32, Layout.row_major(NUM_ENVS), MutAnyOrigin
        ](episode_counts_buf)

        ctx.enqueue_function_checked[
            simple_2d_kernel[
                NUM_ENVS,
                HIDDEN_DIM,
                ENVS_PER_BLOCK,
                OBS_DIM,
                NUM_ACTIONS,
                ROLLOUT_LEN,
            ],
            simple_2d_kernel[
                NUM_ENVS,
                HIDDEN_DIM,
                ENVS_PER_BLOCK,
                OBS_DIM,
                NUM_ACTIONS,
                ROLLOUT_LEN,
            ],
        ](
            env_states,
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
            total_rewards,
            episode_counts,
            # Standard convention: (x, y) = (columns, rows) = (hidden, envs)
            grid_dim=(1, NUM_BLOCKS),
            block_dim=(HIDDEN_DIM, ENVS_PER_BLOCK),
        )
        ctx.synchronize()

        with total_rewards_buf.map_to_host() as h:
            print("Results (should be 0, 1, 2, ...):")
            for i in range(10):
                print("  [", i, "] =", Float64(h[i]))
            print("  ...")
            for i in range(NUM_ENVS - 3, NUM_ENVS):
                print("  [", i, "] =", Float64(h[i]))

    print("=" * 50)
    print("Done")

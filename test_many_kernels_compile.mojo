"""Test if many different kernel calls per iteration cause compile time issues."""

from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from deep_agents.gpu.a2c_native import (
    tiled_matmul_bias_relu_kernel,
    tiled_matmul_bias_kernel,
    tiled_matmul_bias_relu_save_kernel,
    parallel_softmax_kernel,
    policy_gradient_kernel,
    value_loss_gradient_kernel,
    tiled_matmul_transA_kernel,
    tiled_matmul_transB_kernel,
    relu_backward_kernel,
    elementwise_add_kernel,
    bias_gradient_parallel_kernel,
    sgd_update_2d_kernel,
    sgd_update_1d_kernel,
)
from bit import log2_ceil

comptime dtype = DType.float32


fn test_full_training_step() raises:
    """Simulate full training step with all 21 kernel calls."""
    print("Testing full training step simulation (21 kernels per step)...")

    comptime NUM_ENVS = 64
    comptime OBS_DIM = 4
    comptime HIDDEN_DIM = 64
    comptime NUM_ACTIONS = 2
    comptime TILE = 8
    comptime ROLLOUT_LEN = 4  # Small for compile test

    with DeviceContext() as ctx:
        # Allocate all buffers
        var obs_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * OBS_DIM)
        var W1_buf = ctx.enqueue_create_buffer[dtype](OBS_DIM * HIDDEN_DIM)
        var b1_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
        var pre_act1_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * HIDDEN_DIM
        )
        var hidden_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * HIDDEN_DIM)
        var W_actor_buf = ctx.enqueue_create_buffer[dtype](
            HIDDEN_DIM * NUM_ACTIONS
        )
        var b_actor_buf = ctx.enqueue_create_buffer[dtype](NUM_ACTIONS)
        var logits_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * NUM_ACTIONS
        )
        var probs_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * NUM_ACTIONS)
        var W_critic_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM * 1)
        var b_critic_buf = ctx.enqueue_create_buffer[dtype](1)
        var values_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * 1)

        var actions_buf = ctx.enqueue_create_buffer[DType.int32](NUM_ENVS)
        var advantages_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS)
        var returns_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS)

        var d_logits_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * NUM_ACTIONS
        )
        var d_values_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * 1)
        var d_hidden_actor_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * HIDDEN_DIM
        )
        var d_hidden_critic_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * HIDDEN_DIM
        )
        var d_hidden_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * HIDDEN_DIM
        )
        var d_pre_relu_buf = ctx.enqueue_create_buffer[dtype](
            NUM_ENVS * HIDDEN_DIM
        )
        var d_W1_buf = ctx.enqueue_create_buffer[dtype](OBS_DIM * HIDDEN_DIM)
        var d_b1_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
        var d_W_actor_buf = ctx.enqueue_create_buffer[dtype](
            HIDDEN_DIM * NUM_ACTIONS
        )
        var d_b_actor_buf = ctx.enqueue_create_buffer[dtype](NUM_ACTIONS)
        var d_W_critic_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM * 1)
        var d_b_critic_buf = ctx.enqueue_create_buffer[dtype](1)

        # Create all tensors
        var obs = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, OBS_DIM), MutAnyOrigin
        ](obs_buf)
        var obs_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, OBS_DIM), ImmutAnyOrigin
        ](obs_buf)
        var W1 = LayoutTensor[
            dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), MutAnyOrigin
        ](W1_buf)
        var W1_immut = LayoutTensor[
            dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), ImmutAnyOrigin
        ](W1_buf)
        var b1 = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM), MutAnyOrigin
        ](b1_buf)
        var b1_immut = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin
        ](b1_buf)
        var pre_act1 = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin
        ](pre_act1_buf)
        var pre_act1_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), ImmutAnyOrigin
        ](pre_act1_buf)
        var hidden = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin
        ](hidden_buf)
        var hidden_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), ImmutAnyOrigin
        ](hidden_buf)
        var W_actor = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, NUM_ACTIONS), MutAnyOrigin
        ](W_actor_buf)
        var W_actor_immut = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, NUM_ACTIONS), ImmutAnyOrigin
        ](W_actor_buf)
        var b_actor = LayoutTensor[
            dtype, Layout.row_major(NUM_ACTIONS), MutAnyOrigin
        ](b_actor_buf)
        var b_actor_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ACTIONS), ImmutAnyOrigin
        ](b_actor_buf)
        var logits = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), MutAnyOrigin
        ](logits_buf)
        var logits_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), ImmutAnyOrigin
        ](logits_buf)
        var probs = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), MutAnyOrigin
        ](probs_buf)
        var probs_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), ImmutAnyOrigin
        ](probs_buf)
        var W_critic = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, 1), MutAnyOrigin
        ](W_critic_buf)
        var W_critic_immut = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, 1), ImmutAnyOrigin
        ](W_critic_buf)
        var b_critic = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](
            b_critic_buf
        )
        var b_critic_immut = LayoutTensor[
            dtype, Layout.row_major(1), ImmutAnyOrigin
        ](b_critic_buf)
        var values = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin
        ](values_buf)
        var values_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, 1), ImmutAnyOrigin
        ](values_buf)

        var actions_t = LayoutTensor[
            DType.int32, Layout.row_major(NUM_ENVS), ImmutAnyOrigin
        ](actions_buf)
        var advantages_t = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS), ImmutAnyOrigin
        ](advantages_buf)
        var returns_t = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS), ImmutAnyOrigin
        ](returns_buf)

        var d_logits = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), MutAnyOrigin
        ](d_logits_buf)
        var d_logits_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), ImmutAnyOrigin
        ](d_logits_buf)
        var d_values = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin
        ](d_values_buf)
        var d_values_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, 1), ImmutAnyOrigin
        ](d_values_buf)
        var d_hidden_actor = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin
        ](d_hidden_actor_buf)
        var d_hidden_actor_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), ImmutAnyOrigin
        ](d_hidden_actor_buf)
        var d_hidden_critic = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin
        ](d_hidden_critic_buf)
        var d_hidden_critic_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), ImmutAnyOrigin
        ](d_hidden_critic_buf)
        var d_hidden = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin
        ](d_hidden_buf)
        var d_hidden_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), ImmutAnyOrigin
        ](d_hidden_buf)
        var d_pre_relu = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin
        ](d_pre_relu_buf)
        var d_pre_relu_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), ImmutAnyOrigin
        ](d_pre_relu_buf)
        var d_W1 = LayoutTensor[
            dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), MutAnyOrigin
        ](d_W1_buf)
        var d_W1_immut = LayoutTensor[
            dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), ImmutAnyOrigin
        ](d_W1_buf)
        var d_b1 = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM), MutAnyOrigin
        ](d_b1_buf)
        var d_b1_immut = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin
        ](d_b1_buf)
        var d_W_actor = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, NUM_ACTIONS), MutAnyOrigin
        ](d_W_actor_buf)
        var d_W_actor_immut = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, NUM_ACTIONS), ImmutAnyOrigin
        ](d_W_actor_buf)
        var d_b_actor = LayoutTensor[
            dtype, Layout.row_major(NUM_ACTIONS), MutAnyOrigin
        ](d_b_actor_buf)
        var d_b_actor_immut = LayoutTensor[
            dtype, Layout.row_major(NUM_ACTIONS), ImmutAnyOrigin
        ](d_b_actor_buf)
        var d_W_critic = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, 1), MutAnyOrigin
        ](d_W_critic_buf)
        var d_W_critic_immut = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, 1), ImmutAnyOrigin
        ](d_W_critic_buf)
        var d_b_critic = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](
            d_b_critic_buf
        )
        var d_b_critic_immut = LayoutTensor[
            dtype, Layout.row_major(1), ImmutAnyOrigin
        ](d_b_critic_buf)

        comptime grid_h_x = (HIDDEN_DIM + TILE - 1) // TILE
        comptime grid_h_y = (NUM_ENVS + TILE - 1) // TILE
        comptime grid_a_x = (NUM_ACTIONS + TILE - 1) // TILE
        comptime grid_a_y = (NUM_ENVS + TILE - 1) // TILE
        comptime grid_v_x = (1 + TILE - 1) // TILE
        comptime grid_v_y = (NUM_ENVS + TILE - 1) // TILE
        comptime SOFTMAX_BLOCK = 1 << log2_ceil(NUM_ACTIONS)
        comptime BLOCK_SIZE_BIAS = 1 << log2_ceil(NUM_ENVS)

        var entropy_coef = Scalar[dtype](0.01)
        var value_coef = Scalar[dtype](0.5)
        var lr = Scalar[dtype](0.0003)

        # Simulate 2 updates with ROLLOUT_LEN steps each
        for update in range(2):
            for step in range(ROLLOUT_LEN):
                # === Forward pass (4 kernels) ===
                ctx.enqueue_function_checked[
                    tiled_matmul_bias_relu_save_kernel[
                        NUM_ENVS, OBS_DIM, HIDDEN_DIM, TILE
                    ],
                    tiled_matmul_bias_relu_save_kernel[
                        NUM_ENVS, OBS_DIM, HIDDEN_DIM, TILE
                    ],
                ](
                    hidden,
                    pre_act1,
                    obs_immut,
                    W1_immut,
                    b1_immut,
                    grid_dim=(grid_h_x, grid_h_y),
                    block_dim=(TILE, TILE),
                )

                ctx.enqueue_function_checked[
                    tiled_matmul_bias_kernel[
                        NUM_ENVS, HIDDEN_DIM, NUM_ACTIONS, TILE
                    ],
                    tiled_matmul_bias_kernel[
                        NUM_ENVS, HIDDEN_DIM, NUM_ACTIONS, TILE
                    ],
                ](
                    logits,
                    hidden_immut,
                    W_actor_immut,
                    b_actor_immut,
                    grid_dim=(grid_a_x, grid_a_y),
                    block_dim=(TILE, TILE),
                )

                ctx.enqueue_function_checked[
                    parallel_softmax_kernel[NUM_ENVS, NUM_ACTIONS],
                    parallel_softmax_kernel[NUM_ENVS, NUM_ACTIONS],
                ](
                    probs,
                    logits_immut,
                    grid_dim=(1, NUM_ENVS),
                    block_dim=(SOFTMAX_BLOCK, 1),
                )

                ctx.enqueue_function_checked[
                    tiled_matmul_bias_kernel[NUM_ENVS, HIDDEN_DIM, 1, TILE],
                    tiled_matmul_bias_kernel[NUM_ENVS, HIDDEN_DIM, 1, TILE],
                ](
                    values,
                    hidden_immut,
                    W_critic_immut,
                    b_critic_immut,
                    grid_dim=(grid_v_x, grid_v_y),
                    block_dim=(TILE, TILE),
                )

                # === Backward pass (11 kernels) ===
                ctx.enqueue_function_checked[
                    policy_gradient_kernel[NUM_ENVS, NUM_ACTIONS],
                    policy_gradient_kernel[NUM_ENVS, NUM_ACTIONS],
                ](
                    d_logits,
                    probs_immut,
                    actions_t,
                    advantages_t,
                    entropy_coef,
                    grid_dim=(grid_a_x, grid_a_y),
                    block_dim=(TILE, TILE),
                )

                ctx.enqueue_function_checked[
                    value_loss_gradient_kernel[NUM_ENVS],
                    value_loss_gradient_kernel[NUM_ENVS],
                ](
                    d_values,
                    values_immut,
                    returns_t,
                    value_coef,
                    grid_dim=(1, grid_v_y),
                    block_dim=(1, TILE),
                )

                ctx.enqueue_function_checked[
                    tiled_matmul_transA_kernel[
                        HIDDEN_DIM, NUM_ENVS, NUM_ACTIONS, TILE
                    ],
                    tiled_matmul_transA_kernel[
                        HIDDEN_DIM, NUM_ENVS, NUM_ACTIONS, TILE
                    ],
                ](
                    d_W_actor,
                    hidden_immut,
                    d_logits_immut,
                    grid_dim=(grid_a_x, (HIDDEN_DIM + TILE - 1) // TILE),
                    block_dim=(TILE, TILE),
                )

                ctx.enqueue_function_checked[
                    bias_gradient_parallel_kernel[NUM_ENVS, NUM_ACTIONS],
                    bias_gradient_parallel_kernel[NUM_ENVS, NUM_ACTIONS],
                ](
                    d_b_actor,
                    d_logits_immut,
                    grid_dim=(NUM_ACTIONS, 1),
                    block_dim=(BLOCK_SIZE_BIAS, 1),
                )

                ctx.enqueue_function_checked[
                    tiled_matmul_transB_kernel[
                        NUM_ENVS, NUM_ACTIONS, HIDDEN_DIM, TILE
                    ],
                    tiled_matmul_transB_kernel[
                        NUM_ENVS, NUM_ACTIONS, HIDDEN_DIM, TILE
                    ],
                ](
                    d_hidden_actor,
                    d_logits_immut,
                    W_actor_immut,
                    grid_dim=(grid_h_x, grid_h_y),
                    block_dim=(TILE, TILE),
                )

                ctx.enqueue_function_checked[
                    tiled_matmul_transA_kernel[HIDDEN_DIM, NUM_ENVS, 1, TILE],
                    tiled_matmul_transA_kernel[HIDDEN_DIM, NUM_ENVS, 1, TILE],
                ](
                    d_W_critic,
                    hidden_immut,
                    d_values_immut,
                    grid_dim=(grid_v_x, (HIDDEN_DIM + TILE - 1) // TILE),
                    block_dim=(TILE, TILE),
                )

                ctx.enqueue_function_checked[
                    bias_gradient_parallel_kernel[NUM_ENVS, 1],
                    bias_gradient_parallel_kernel[NUM_ENVS, 1],
                ](
                    d_b_critic,
                    d_values_immut,
                    grid_dim=(1, 1),
                    block_dim=(BLOCK_SIZE_BIAS, 1),
                )

                ctx.enqueue_function_checked[
                    tiled_matmul_transB_kernel[NUM_ENVS, 1, HIDDEN_DIM, TILE],
                    tiled_matmul_transB_kernel[NUM_ENVS, 1, HIDDEN_DIM, TILE],
                ](
                    d_hidden_critic,
                    d_values_immut,
                    W_critic_immut,
                    grid_dim=(grid_h_x, grid_h_y),
                    block_dim=(TILE, TILE),
                )

                ctx.enqueue_function_checked[
                    elementwise_add_kernel[NUM_ENVS, HIDDEN_DIM, TILE],
                    elementwise_add_kernel[NUM_ENVS, HIDDEN_DIM, TILE],
                ](
                    d_hidden,
                    d_hidden_actor_immut,
                    d_hidden_critic_immut,
                    grid_dim=(grid_h_x, grid_h_y),
                    block_dim=(TILE, TILE),
                )

                ctx.enqueue_function_checked[
                    relu_backward_kernel[NUM_ENVS, HIDDEN_DIM, TILE],
                    relu_backward_kernel[NUM_ENVS, HIDDEN_DIM, TILE],
                ](
                    d_pre_relu,
                    d_hidden_immut,
                    pre_act1_immut,
                    grid_dim=(grid_h_x, grid_h_y),
                    block_dim=(TILE, TILE),
                )

                ctx.enqueue_function_checked[
                    tiled_matmul_transA_kernel[
                        OBS_DIM, NUM_ENVS, HIDDEN_DIM, TILE
                    ],
                    tiled_matmul_transA_kernel[
                        OBS_DIM, NUM_ENVS, HIDDEN_DIM, TILE
                    ],
                ](
                    d_W1,
                    obs_immut,
                    d_pre_relu_immut,
                    grid_dim=(grid_h_x, (OBS_DIM + TILE - 1) // TILE),
                    block_dim=(TILE, TILE),
                )

                ctx.enqueue_function_checked[
                    bias_gradient_parallel_kernel[NUM_ENVS, HIDDEN_DIM],
                    bias_gradient_parallel_kernel[NUM_ENVS, HIDDEN_DIM],
                ](
                    d_b1,
                    d_pre_relu_immut,
                    grid_dim=(HIDDEN_DIM, 1),
                    block_dim=(BLOCK_SIZE_BIAS, 1),
                )

                # === SGD updates (6 kernels) ===
                ctx.enqueue_function_checked[
                    sgd_update_2d_kernel[OBS_DIM, HIDDEN_DIM, TILE],
                    sgd_update_2d_kernel[OBS_DIM, HIDDEN_DIM, TILE],
                ](
                    W1,
                    d_W1_immut,
                    lr,
                    grid_dim=(grid_h_x, (OBS_DIM + TILE - 1) // TILE),
                    block_dim=(TILE, TILE),
                )

                ctx.enqueue_function_checked[
                    sgd_update_1d_kernel[HIDDEN_DIM],
                    sgd_update_1d_kernel[HIDDEN_DIM],
                ](
                    b1,
                    d_b1_immut,
                    lr,
                    grid_dim=(1, 1),
                    block_dim=(HIDDEN_DIM, 1),
                )

                ctx.enqueue_function_checked[
                    sgd_update_2d_kernel[HIDDEN_DIM, NUM_ACTIONS, TILE],
                    sgd_update_2d_kernel[HIDDEN_DIM, NUM_ACTIONS, TILE],
                ](
                    W_actor,
                    d_W_actor_immut,
                    lr,
                    grid_dim=(grid_a_x, (HIDDEN_DIM + TILE - 1) // TILE),
                    block_dim=(TILE, TILE),
                )

                ctx.enqueue_function_checked[
                    sgd_update_1d_kernel[NUM_ACTIONS],
                    sgd_update_1d_kernel[NUM_ACTIONS],
                ](
                    b_actor,
                    d_b_actor_immut,
                    lr,
                    grid_dim=(1, 1),
                    block_dim=(NUM_ACTIONS, 1),
                )

                ctx.enqueue_function_checked[
                    sgd_update_2d_kernel[HIDDEN_DIM, 1, TILE],
                    sgd_update_2d_kernel[HIDDEN_DIM, 1, TILE],
                ](
                    W_critic,
                    d_W_critic_immut,
                    lr,
                    grid_dim=(grid_v_x, (HIDDEN_DIM + TILE - 1) // TILE),
                    block_dim=(TILE, TILE),
                )

                ctx.enqueue_function_checked[
                    sgd_update_1d_kernel[1],
                    sgd_update_1d_kernel[1],
                ](
                    b_critic,
                    d_b_critic_immut,
                    lr,
                    grid_dim=(1, 1),
                    block_dim=(1, 1),
                )

            ctx.synchronize()

    print("  Full training step simulation OK")


fn main() raises:
    print("Testing many kernels per iteration")
    print("=" * 50)

    test_full_training_step()

    print("=" * 50)
    print("Test passed!")

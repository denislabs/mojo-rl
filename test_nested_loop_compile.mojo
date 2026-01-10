"""Test if nested loops with multiple kernel calls cause compile time issues."""

from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from deep_agents.gpu.a2c_native import (
    tiled_matmul_bias_relu_kernel,
    tiled_matmul_bias_kernel,
    parallel_softmax_kernel,
)
from bit import log2_ceil

comptime dtype = DType.float32


fn test_nested_loops() raises:
    """Nested loops with 4 kernel calls per inner iteration - like the rollout collection."""
    print("Testing nested loops (5 x 8 = 40 outer iterations, 4 kernels each)...")

    comptime NUM_ENVS = 64
    comptime OBS_DIM = 4
    comptime HIDDEN_DIM = 64
    comptime NUM_ACTIONS = 2
    comptime TILE = 8
    comptime ROLLOUT_LEN = 8  # Reduced from 32

    with DeviceContext() as ctx:
        var obs_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * OBS_DIM)
        var W1_buf = ctx.enqueue_create_buffer[dtype](OBS_DIM * HIDDEN_DIM)
        var b1_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
        var hidden_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * HIDDEN_DIM)
        var W_actor_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM * NUM_ACTIONS)
        var b_actor_buf = ctx.enqueue_create_buffer[dtype](NUM_ACTIONS)
        var logits_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * NUM_ACTIONS)
        var probs_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * NUM_ACTIONS)
        var W_critic_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM * 1)
        var b_critic_buf = ctx.enqueue_create_buffer[dtype](1)
        var values_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * 1)

        var hidden = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin](hidden_buf)
        var obs = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, OBS_DIM), ImmutAnyOrigin](obs_buf)
        var W1 = LayoutTensor[dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), ImmutAnyOrigin](W1_buf)
        var b1 = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin](b1_buf)
        var hidden_immut = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), ImmutAnyOrigin](hidden_buf)
        var W_actor = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, NUM_ACTIONS), ImmutAnyOrigin](W_actor_buf)
        var b_actor = LayoutTensor[dtype, Layout.row_major(NUM_ACTIONS), ImmutAnyOrigin](b_actor_buf)
        var logits = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), MutAnyOrigin](logits_buf)
        var logits_immut = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), ImmutAnyOrigin](logits_buf)
        var probs = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), MutAnyOrigin](probs_buf)
        var W_critic = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, 1), ImmutAnyOrigin](W_critic_buf)
        var b_critic = LayoutTensor[dtype, Layout.row_major(1), ImmutAnyOrigin](b_critic_buf)
        var values = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin](values_buf)

        comptime grid_h_x = (HIDDEN_DIM + TILE - 1) // TILE
        comptime grid_h_y = (NUM_ENVS + TILE - 1) // TILE
        comptime grid_a_x = (NUM_ACTIONS + TILE - 1) // TILE
        comptime grid_a_y = (NUM_ENVS + TILE - 1) // TILE
        comptime grid_v_x = (1 + TILE - 1) // TILE
        comptime grid_v_y = (NUM_ENVS + TILE - 1) // TILE
        comptime SOFTMAX_BLOCK = 1 << log2_ceil(NUM_ACTIONS)

        # Simulate the nested training loop (reduced counts)
        var max_updates = 5  # Reduced from 500

        for update in range(max_updates):
            for step in range(ROLLOUT_LEN):
                # 4 kernel calls per step (like rollout collection)
                ctx.enqueue_function_checked[
                    tiled_matmul_bias_relu_kernel[NUM_ENVS, OBS_DIM, HIDDEN_DIM, TILE],
                    tiled_matmul_bias_relu_kernel[NUM_ENVS, OBS_DIM, HIDDEN_DIM, TILE],
                ](
                    hidden, obs, W1, b1,
                    grid_dim=(grid_h_x, grid_h_y),
                    block_dim=(TILE, TILE),
                )

                ctx.enqueue_function_checked[
                    tiled_matmul_bias_kernel[NUM_ENVS, HIDDEN_DIM, NUM_ACTIONS, TILE],
                    tiled_matmul_bias_kernel[NUM_ENVS, HIDDEN_DIM, NUM_ACTIONS, TILE],
                ](
                    logits, hidden_immut, W_actor, b_actor,
                    grid_dim=(grid_a_x, grid_a_y),
                    block_dim=(TILE, TILE),
                )

                ctx.enqueue_function_checked[
                    parallel_softmax_kernel[NUM_ENVS, NUM_ACTIONS],
                    parallel_softmax_kernel[NUM_ENVS, NUM_ACTIONS],
                ](
                    probs, logits_immut,
                    grid_dim=(1, NUM_ENVS),
                    block_dim=(SOFTMAX_BLOCK, 1),
                )

                ctx.enqueue_function_checked[
                    tiled_matmul_bias_kernel[NUM_ENVS, HIDDEN_DIM, 1, TILE],
                    tiled_matmul_bias_kernel[NUM_ENVS, HIDDEN_DIM, 1, TILE],
                ](
                    values, hidden_immut, W_critic, b_critic,
                    grid_dim=(grid_v_x, grid_v_y),
                    block_dim=(TILE, TILE),
                )

            ctx.synchronize()

    print("  Nested loops OK")


fn main() raises:
    print("Testing nested loop compilation behavior")
    print("=" * 50)

    test_nested_loops()

    print("=" * 50)
    print("Test passed!")

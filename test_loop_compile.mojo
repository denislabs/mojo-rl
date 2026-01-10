"""Test if loops with enqueue_function_checked cause compile time issues."""

from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from deep_agents.gpu.a2c_native import tiled_matmul_bias_relu_kernel

comptime dtype = DType.float32


fn test_single_call() raises:
    """Just one kernel call - baseline."""
    print("Testing single call...")

    comptime NUM_ENVS = 64
    comptime OBS_DIM = 4
    comptime HIDDEN_DIM = 64
    comptime TILE = 8

    with DeviceContext() as ctx:
        var obs_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * OBS_DIM)
        var W1_buf = ctx.enqueue_create_buffer[dtype](OBS_DIM * HIDDEN_DIM)
        var b1_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
        var hidden_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * HIDDEN_DIM)

        var hidden = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin](hidden_buf)
        var obs = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, OBS_DIM), ImmutAnyOrigin](obs_buf)
        var W1 = LayoutTensor[dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), ImmutAnyOrigin](W1_buf)
        var b1 = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin](b1_buf)

        comptime grid_h_x = (HIDDEN_DIM + TILE - 1) // TILE
        comptime grid_h_y = (NUM_ENVS + TILE - 1) // TILE

        ctx.enqueue_function_checked[
            tiled_matmul_bias_relu_kernel[NUM_ENVS, OBS_DIM, HIDDEN_DIM, TILE],
            tiled_matmul_bias_relu_kernel[NUM_ENVS, OBS_DIM, HIDDEN_DIM, TILE],
        ](
            hidden, obs, W1, b1,
            grid_dim=(grid_h_x, grid_h_y),
            block_dim=(TILE, TILE),
        )
        ctx.synchronize()

    print("  Single call OK")


fn test_loop_2_iterations() raises:
    """2 iterations - should be fast."""
    print("Testing 2 iterations...")

    comptime NUM_ENVS = 64
    comptime OBS_DIM = 4
    comptime HIDDEN_DIM = 64
    comptime TILE = 8

    with DeviceContext() as ctx:
        var obs_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * OBS_DIM)
        var W1_buf = ctx.enqueue_create_buffer[dtype](OBS_DIM * HIDDEN_DIM)
        var b1_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
        var hidden_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * HIDDEN_DIM)

        var hidden = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin](hidden_buf)
        var obs = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, OBS_DIM), ImmutAnyOrigin](obs_buf)
        var W1 = LayoutTensor[dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), ImmutAnyOrigin](W1_buf)
        var b1 = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin](b1_buf)

        comptime grid_h_x = (HIDDEN_DIM + TILE - 1) // TILE
        comptime grid_h_y = (NUM_ENVS + TILE - 1) // TILE

        for i in range(2):
            ctx.enqueue_function_checked[
                tiled_matmul_bias_relu_kernel[NUM_ENVS, OBS_DIM, HIDDEN_DIM, TILE],
                tiled_matmul_bias_relu_kernel[NUM_ENVS, OBS_DIM, HIDDEN_DIM, TILE],
            ](
                hidden, obs, W1, b1,
                grid_dim=(grid_h_x, grid_h_y),
                block_dim=(TILE, TILE),
            )
        ctx.synchronize()

    print("  2 iterations OK")


fn test_loop_32_iterations() raises:
    """32 iterations (like ROLLOUT_LEN) - might be slow."""
    print("Testing 32 iterations...")

    comptime NUM_ENVS = 64
    comptime OBS_DIM = 4
    comptime HIDDEN_DIM = 64
    comptime TILE = 8

    with DeviceContext() as ctx:
        var obs_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * OBS_DIM)
        var W1_buf = ctx.enqueue_create_buffer[dtype](OBS_DIM * HIDDEN_DIM)
        var b1_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
        var hidden_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * HIDDEN_DIM)

        var hidden = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin](hidden_buf)
        var obs = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, OBS_DIM), ImmutAnyOrigin](obs_buf)
        var W1 = LayoutTensor[dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), ImmutAnyOrigin](W1_buf)
        var b1 = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin](b1_buf)

        comptime grid_h_x = (HIDDEN_DIM + TILE - 1) // TILE
        comptime grid_h_y = (NUM_ENVS + TILE - 1) // TILE

        for i in range(32):
            ctx.enqueue_function_checked[
                tiled_matmul_bias_relu_kernel[NUM_ENVS, OBS_DIM, HIDDEN_DIM, TILE],
                tiled_matmul_bias_relu_kernel[NUM_ENVS, OBS_DIM, HIDDEN_DIM, TILE],
            ](
                hidden, obs, W1, b1,
                grid_dim=(grid_h_x, grid_h_y),
                block_dim=(TILE, TILE),
            )
        ctx.synchronize()

    print("  32 iterations OK")


fn main() raises:
    print("Testing loop compilation behavior")
    print("=" * 40)

    test_single_call()
    test_loop_2_iterations()
    test_loop_32_iterations()

    print("=" * 40)
    print("All tests passed!")

"""Minimal test for 2D thread blocks."""

from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor

comptime dtype = DType.float32


fn test_2d_kernel[
    NUM_BLOCKS: Int,
    BLOCK_X: Int,
    BLOCK_Y: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(NUM_BLOCKS * BLOCK_X), MutAnyOrigin],
    output2: LayoutTensor[dtype, Layout.row_major(NUM_BLOCKS * BLOCK_X * BLOCK_Y), MutAnyOrigin],
):
    """Simple kernel: test thread indexing."""
    var local_x = Int(thread_idx.x)
    var local_y = Int(thread_idx.y)
    var block_x = Int(block_idx.x)

    # Linear thread ID within block
    var linear_tid = local_y * BLOCK_X + local_x
    var global_linear = block_x * BLOCK_X * BLOCK_Y + linear_tid

    # Write to output2 using linear index
    if global_linear < NUM_BLOCKS * BLOCK_X * BLOCK_Y:
        output2[global_linear] = Scalar[dtype](Float32(local_x * 1000 + local_y))

    # Also write to output (first thread per x position)
    var global_idx = block_x * BLOCK_X + local_x
    if local_y == 0:
        output[global_idx] = Scalar[dtype](Float32(local_x * 100 + local_y))


fn main() raises:
    print("Testing 2D thread blocks")
    print("=" * 40)

    comptime NUM_BLOCKS = 4
    comptime BLOCK_X = 8   # envs per block
    comptime BLOCK_Y = 16  # "hidden dim"
    comptime OUTPUT_SIZE = NUM_BLOCKS * BLOCK_X

    print("Grid: (", NUM_BLOCKS, ",)")
    print("Block: (", BLOCK_X, ",", BLOCK_Y, ")")
    print("Total threads per block:", BLOCK_X * BLOCK_Y)

    comptime TOTAL_THREADS = NUM_BLOCKS * BLOCK_X * BLOCK_Y

    with DeviceContext() as ctx:
        var output_buf = ctx.enqueue_create_buffer[dtype](OUTPUT_SIZE)
        var output2_buf = ctx.enqueue_create_buffer[dtype](TOTAL_THREADS)

        # Initialize to -1
        with output_buf.map_to_host() as h:
            for i in range(OUTPUT_SIZE):
                h[i] = Scalar[dtype](-1.0)
        with output2_buf.map_to_host() as h:
            for i in range(TOTAL_THREADS):
                h[i] = Scalar[dtype](-1.0)

        var output = LayoutTensor[dtype, Layout.row_major(OUTPUT_SIZE), MutAnyOrigin](output_buf)
        var output2 = LayoutTensor[dtype, Layout.row_major(TOTAL_THREADS), MutAnyOrigin](output2_buf)

        ctx.enqueue_function_checked[
            test_2d_kernel[NUM_BLOCKS, BLOCK_X, BLOCK_Y],
            test_2d_kernel[NUM_BLOCKS, BLOCK_X, BLOCK_Y],
        ](
            output,
            output2,
            grid_dim=(NUM_BLOCKS,),
            block_dim=(BLOCK_X, BLOCK_Y),
        )
        ctx.synchronize()

        print("Linear thread output (x*1000 + y):")
        with output2_buf.map_to_host() as h:
            for i in range(min(32, TOTAL_THREADS)):
                var v = Int(h[i])
                var x = v // 1000
                var y = v % 1000
                print("  linear[", i, "] = x:", x, "y:", y)

        print()
        print("Output where local_y == 0:")
        with output_buf.map_to_host() as h:
            for i in range(OUTPUT_SIZE):
                print("  [", i, "] =", Float64(h[i]))

    print("=" * 40)
    print("Done")

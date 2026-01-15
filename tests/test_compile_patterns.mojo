"""Test compilation patterns: compile-time vs runtime dimensions.

This test explores the p07 puzzle pattern where:
- Layout is a compile-time type parameter (not a dimension Int)
- Actual size is passed as runtime UInt parameter
- Bounds checking uses the runtime parameter

Pattern A: Dimension as compile-time Int (current approach) - recompiles per size
Pattern B: Layout as type param + size as runtime UInt (puzzle p07 pattern)

Run with:
    pixi run -e apple mojo run tests/test_compile_patterns.mojo
"""

from time import perf_counter_ns
from random import seed, random_float64

from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, DeviceBuffer
from layout import LayoutTensor, Layout

from deep_rl.constants import dtype

comptime TILE = 16

# Fixed dimensions for the test
comptime IN_DIM = 64
comptime OUT_DIM = 32


# =============================================================================
# Pattern A: Dimension as compile-time Int (current approach)
# The kernel has [BATCH: Int] which causes recompilation for each batch size
# =============================================================================

@always_inline
fn pattern_a_kernel[
    BATCH: Int,  # <-- This compile-time Int causes recompilation!
](
    output: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    input: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
    W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), MutAnyOrigin],
):
    """Compile-time BATCH: recompiled for every batch size."""
    var row = Int(block_idx.y * TILE + thread_idx.y)
    var col = Int(block_idx.x * TILE + thread_idx.x)

    if row < BATCH and col < OUT_DIM:  # BATCH is compile-time - baked in!
        var acc = Scalar[dtype](0)
        for k in range(IN_DIM):
            var inp_val = input[row, k]
            var w_val = W[k, col]
            acc += inp_val[0] * w_val[0]
        output[row, col] = acc


fn launch_pattern_a[BATCH: Int](
    ctx: DeviceContext,
    output_buf: DeviceBuffer[dtype],
    input_buf: DeviceBuffer[dtype],
    W_buf: DeviceBuffer[dtype],
) raises:
    """Launch Pattern A kernel."""
    var output = LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin](
        output_buf.unsafe_ptr()
    )
    var input = LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin](
        input_buf.unsafe_ptr()
    )
    var W = LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), MutAnyOrigin](
        W_buf.unsafe_ptr()
    )

    comptime grid_x = (OUT_DIM + TILE - 1) // TILE
    comptime grid_y = (BATCH + TILE - 1) // TILE

    @always_inline
    fn wrapper(
        output: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
        input: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
        W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), MutAnyOrigin],
    ):
        pattern_a_kernel[BATCH](output, input, W)

    ctx.enqueue_function[wrapper, wrapper](
        output, input, W,
        grid_dim=(grid_x, grid_y),
        block_dim=(TILE, TILE),
    )


# =============================================================================
# Pattern B: Layout as type param + size as runtime UInt (puzzle p07 pattern)
# The kernel has [out_layout: Layout] - the layout TYPE, not a dimension Int
# Size is passed as runtime UInt for bounds checking
# =============================================================================

@always_inline
fn pattern_b_kernel[
    out_layout: Layout,  # <-- Layout TYPE, not a dimension Int
    in_layout: Layout,
    W_layout: Layout,
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    input: LayoutTensor[dtype, in_layout, MutAnyOrigin],
    W: LayoutTensor[dtype, W_layout, MutAnyOrigin],
    batch: UInt,  # <-- Runtime parameter for bounds checking!
):
    """Layout as type param, batch as runtime UInt (puzzle p07 pattern)."""
    var row = Int(block_idx.y * TILE + thread_idx.y)
    var col = Int(block_idx.x * TILE + thread_idx.x)

    if row < Int(batch) and col < OUT_DIM:  # batch is runtime - flexible!
        var acc = Scalar[dtype](0)
        for k in range(IN_DIM):
            var inp_val = input[row, k]
            var w_val = W[k, col]
            acc += inp_val[0] * w_val[0]
        output[row, col] = acc


fn launch_pattern_b[
    BUFFER_SIZE: Int,  # Fixed buffer size (e.g., max batch you'll ever use)
](
    ctx: DeviceContext,
    output_buf: DeviceBuffer[dtype],
    input_buf: DeviceBuffer[dtype],
    W_buf: DeviceBuffer[dtype],
    actual_batch: Int,  # Runtime batch size
) raises:
    """Launch Pattern B kernel with runtime batch."""
    # Layouts are fixed at compile time based on BUFFER_SIZE
    comptime out_layout = Layout.row_major(BUFFER_SIZE, OUT_DIM)
    comptime in_layout = Layout.row_major(BUFFER_SIZE, IN_DIM)
    comptime W_layout = Layout.row_major(IN_DIM, OUT_DIM)

    var output = LayoutTensor[dtype, out_layout, MutAnyOrigin](
        output_buf.unsafe_ptr()
    )
    var input = LayoutTensor[dtype, in_layout, MutAnyOrigin](
        input_buf.unsafe_ptr()
    )
    var W = LayoutTensor[dtype, W_layout, MutAnyOrigin](
        W_buf.unsafe_ptr()
    )

    comptime grid_x = (OUT_DIM + TILE - 1) // TILE
    var grid_y = (actual_batch + TILE - 1) // TILE  # Runtime grid!

    # Compile the kernel ONCE with fixed layouts
    comptime kernel = pattern_b_kernel[out_layout, in_layout, W_layout]

    ctx.enqueue_function[kernel, kernel](
        output, input, W,
        UInt(actual_batch),  # Pass batch as runtime UInt
        grid_dim=(grid_x, grid_y),
        block_dim=(TILE, TILE),
    )


# =============================================================================
# Main test
# =============================================================================

fn main() raises:
    seed(42)
    print("=" * 70)
    print("Compilation Pattern Test: p07 Puzzle Pattern")
    print("=" * 70)
    print()
    print("IN_DIM:", IN_DIM, "| OUT_DIM:", OUT_DIM)
    print()

    # Test parameters
    comptime BUFFER_SIZE = 1024  # Fixed buffer size for layouts
    var test_batches = List[Int]()
    test_batches.append(64)
    test_batches.append(128)
    test_batches.append(256)
    test_batches.append(512)
    test_batches.append(1024)

    with DeviceContext() as ctx:
        # Allocate buffers for max batch size
        var output_buf = ctx.enqueue_create_buffer[dtype](BUFFER_SIZE * OUT_DIM)
        var input_buf = ctx.enqueue_create_buffer[dtype](BUFFER_SIZE * IN_DIM)
        var W_buf = ctx.enqueue_create_buffer[dtype](IN_DIM * OUT_DIM)

        # Initialize W with random values
        var W_host = ctx.enqueue_create_host_buffer[dtype](IN_DIM * OUT_DIM)
        for i in range(IN_DIM * OUT_DIM):
            W_host[i] = Scalar[dtype](random_float64() * 0.1)
        ctx.enqueue_copy(W_buf, W_host)

        # Initialize input
        var input_host = ctx.enqueue_create_host_buffer[dtype](BUFFER_SIZE * IN_DIM)
        for i in range(BUFFER_SIZE * IN_DIM):
            input_host[i] = Scalar[dtype](random_float64())
        ctx.enqueue_copy(input_buf, input_host)
        ctx.synchronize()

        print("-" * 70)
        print("Pattern A: [BATCH: Int] compile-time dimension (current approach)")
        print("  Each batch size triggers kernel recompilation")
        print("-" * 70)

        var warmup_iters = 10
        var bench_iters = 100

        # Warmup and benchmark batch=256 only
        print("\n  Testing batch=256 only (other sizes would each recompile)")
        for _ in range(warmup_iters):
            launch_pattern_a[256](ctx, output_buf, input_buf, W_buf)
        ctx.synchronize()

        var start_a = perf_counter_ns()
        for _ in range(bench_iters):
            launch_pattern_a[256](ctx, output_buf, input_buf, W_buf)
        ctx.synchronize()
        var time_a = Float64(perf_counter_ns() - start_a) / Float64(bench_iters) / 1e3
        print("  Batch 256: ", String(time_a)[:8], " μs/iter")

        print()
        print("-" * 70)
        print("Pattern B: [layout: Layout] + size: UInt (puzzle p07 pattern)")
        print("  Single kernel compilation, batch passed as runtime UInt")
        print("-" * 70)

        # Warmup - same kernel for all batch sizes!
        for _ in range(warmup_iters):
            launch_pattern_b[BUFFER_SIZE](ctx, output_buf, input_buf, W_buf, 256)
        ctx.synchronize()

        # Benchmark different batch sizes - SAME kernel!
        for batch_idx in range(len(test_batches)):
            var batch = test_batches[batch_idx]
            var start_b = perf_counter_ns()
            for _ in range(bench_iters):
                launch_pattern_b[BUFFER_SIZE](ctx, output_buf, input_buf, W_buf, batch)
            ctx.synchronize()
            var time_b = Float64(perf_counter_ns() - start_b) / Float64(bench_iters) / 1e3
            print("  Batch", batch, ":", String(time_b)[:8], " μs/iter")

        print()
        print("=" * 70)
        print("Summary: The p07 Pattern")
        print("=" * 70)
        print()
        print("BEFORE (recompiles for each batch size):")
        print("  fn kernel[BATCH: Int](")
        print("      t: LayoutTensor[..., Layout.row_major(BATCH, DIM), ...]")
        print("  ):")
        print("      if row < BATCH: ...  # compile-time constant")
        print()
        print("AFTER (single compilation, runtime batch):")
        print("  fn kernel[layout: Layout](")
        print("      t: LayoutTensor[..., layout, ...],")
        print("      batch: UInt,  # runtime parameter")
        print("  ):")
        print("      if row < Int(batch): ...  # runtime check")
        print()
        print("KEY INSIGHT:")
        print("  - Parameterize by Layout TYPE, not dimension Int")
        print("  - Pass actual dimensions as runtime UInt")
        print("  - Bounds checks use runtime values")
        print("  - Grid dimensions can be computed at runtime")

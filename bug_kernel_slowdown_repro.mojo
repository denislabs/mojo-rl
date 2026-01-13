"""Minimal reproduction of GPU kernel slowdown bug.

BUG: Kernel execution time increases progressively with number of launches.
Even the simplest kernel shows 10-14x slowdown over 10,000 launches.

Run: pixi run -e apple mojo run bug_kernel_slowdown_repro.mojo

Expected: Consistent timing across batches
Actual: Timing increases from ~700µs to ~10,000µs per kernel
"""

from time import perf_counter_ns
from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor

comptime dtype = DType.float32
comptime TPB = 256
comptime SIZE = 4096
comptime GRID = (SIZE + TPB - 1) // TPB


fn add_kernel(
    output: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    a: LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx < SIZE:
        output[idx] = a[idx] + b[idx]


fn main() raises:
    print("GPU Kernel Slowdown Bug Reproduction")
    print("=" * 50)

    var ctx = DeviceContext()

    var a_buf = ctx.enqueue_create_buffer[dtype](SIZE)
    var b_buf = ctx.enqueue_create_buffer[dtype](SIZE)
    var c_buf = ctx.enqueue_create_buffer[dtype](SIZE)
    ctx.enqueue_memset(a_buf, 1)
    ctx.enqueue_memset(b_buf, 2)
    ctx.synchronize()

    var a_t = LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin](a_buf)
    var b_t = LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin](b_buf)
    var c_t = LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin](c_buf)

    # Warmup
    for _ in range(100):
        ctx.enqueue_function[add_kernel, add_kernel](
            c_t, a_t, b_t, grid_dim=(GRID,), block_dim=(TPB,)
        )
    ctx.synchronize()

    print("10 batches of 1000 kernel launches:")
    print("-" * 50)

    for batch in range(15):
        var start = perf_counter_ns()
        for _ in range(1000):
            ctx.enqueue_function[add_kernel, add_kernel](
                c_t, a_t, b_t, grid_dim=(GRID,), block_dim=(TPB,)
            )
        ctx.synchronize()
        var end = perf_counter_ns()
        var us_per_iter = Float64(end - start) / 1_000 / 1000.0
        print("Batch", batch, ":", us_per_iter, "µs/kernel")

    print()
    print("BUG: Timing should be constant, but increases ~14x!")

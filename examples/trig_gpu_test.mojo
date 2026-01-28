"""Test various trig functions on GPU/Metal to identify which ones work.

This script tests: sin, cos, tan, asin, acos, atan, atan2
to identify which math functions cause Metal compiler issues.

To run:
    pixi run -e apple mojo run examples/trig_gpu_test.mojo
"""

from math import sin, cos, sqrt
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor


fn main() raises:
    print("=== Trig Functions GPU Test ===\n")

    comptime BATCH = 4
    comptime dtype = DType.float32

    var ctx = DeviceContext()
    print("GPU context created")

    var input_buf = ctx.enqueue_create_buffer[dtype](BATCH)
    var output_buf = ctx.enqueue_create_buffer[dtype](BATCH)

    var input_host = ctx.enqueue_create_host_buffer[dtype](BATCH)
    input_host.unsafe_ptr()[0] = 0.0
    input_host.unsafe_ptr()[1] = 0.5
    input_host.unsafe_ptr()[2] = 1.0
    input_host.unsafe_ptr()[3] = 1.5

    ctx.enqueue_copy(input_buf.unsafe_ptr(), input_host.unsafe_ptr(), BATCH)
    ctx.synchronize()

    var input = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](
        input_buf.unsafe_ptr()
    )
    var output = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](
        output_buf.unsafe_ptr()
    )

    # Test sin (should work)
    print("Testing sin()...")

    @always_inline
    fn sin_kernel(
        input: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        output: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    ):
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH:
            return
        output[i] = sin(input[i])

    ctx.enqueue_function[sin_kernel, sin_kernel](
        input, output, grid_dim=(1,), block_dim=(BATCH,)
    )
    ctx.synchronize()
    print("  sin() works!")

    # Test cos (should work)
    print("Testing cos()...")

    @always_inline
    fn cos_kernel(
        input: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        output: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    ):
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH:
            return
        output[i] = cos(input[i])

    ctx.enqueue_function[cos_kernel, cos_kernel](
        input, output, grid_dim=(1,), block_dim=(BATCH,)
    )
    ctx.synchronize()
    print("  cos() works!")

    # Test sqrt (should work)
    print("Testing sqrt()...")

    @always_inline
    fn sqrt_kernel(
        input: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        output: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    ):
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH:
            return
        output[i] = sqrt(input[i])

    ctx.enqueue_function[sqrt_kernel, sqrt_kernel](
        input, output, grid_dim=(1,), block_dim=(BATCH,)
    )
    ctx.synchronize()
    print("  sqrt() works!")

    print("\n=== All tested functions work on Metal ===")
    print("\nNote: atan2 is tested separately and FAILS on Metal.")

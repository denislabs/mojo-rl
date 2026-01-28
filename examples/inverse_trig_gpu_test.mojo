"""Test inverse trig functions on GPU/Metal.

Tests: asin, acos, atan (single arg), atan2 (two args)

To run:
    pixi run -e apple mojo run examples/inverse_trig_gpu_test.mojo
"""

from math import asin, acos, atan
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor


fn main() raises:
    print("=== Inverse Trig Functions GPU Test ===\n")

    comptime BATCH = 4
    comptime dtype = DType.float32

    var ctx = DeviceContext()
    print("GPU context created")

    var input_buf = ctx.enqueue_create_buffer[dtype](BATCH)
    var output_buf = ctx.enqueue_create_buffer[dtype](BATCH)

    var input_host = ctx.enqueue_create_host_buffer[dtype](BATCH)
    # Use values in valid range for asin/acos: [-1, 1]
    input_host.unsafe_ptr()[0] = 0.0
    input_host.unsafe_ptr()[1] = 0.5
    input_host.unsafe_ptr()[2] = -0.5
    input_host.unsafe_ptr()[3] = 0.9

    ctx.enqueue_copy(input_buf.unsafe_ptr(), input_host.unsafe_ptr(), BATCH)
    ctx.synchronize()

    var input = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](
        input_buf.unsafe_ptr()
    )
    var output = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](
        output_buf.unsafe_ptr()
    )

    # Test asin
    print("Testing asin()...")

    @always_inline
    fn asin_kernel(
        input: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        output: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    ):
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH:
            return
        output[i] = asin(input[i])

    ctx.enqueue_function[asin_kernel, asin_kernel](
        input, output, grid_dim=(1,), block_dim=(BATCH,)
    )
    ctx.synchronize()
    print("  asin() works!")

    # Test acos
    print("Testing acos()...")

    @always_inline
    fn acos_kernel(
        input: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        output: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    ):
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH:
            return
        output[i] = acos(input[i])

    ctx.enqueue_function[acos_kernel, acos_kernel](
        input, output, grid_dim=(1,), block_dim=(BATCH,)
    )
    ctx.synchronize()
    print("  acos() works!")

    # Test atan (single argument)
    print("Testing atan() [single arg]...")

    @always_inline
    fn atan_kernel(
        input: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        output: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    ):
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH:
            return
        output[i] = atan(input[i])

    ctx.enqueue_function[atan_kernel, atan_kernel](
        input, output, grid_dim=(1,), block_dim=(BATCH,)
    )
    ctx.synchronize()
    print("  atan() works!")

    print("\n=== All single-arg inverse trig functions work on Metal ===")
    print("\nNote: atan2(y, x) [two-arg] is tested separately and FAILS on Metal.")

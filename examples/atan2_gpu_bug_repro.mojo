"""Minimal reproduction of atan2 Metal compiler issue.

BUG SUMMARY:
Using atan2() in a GPU kernel causes the Metal compiler to fail with:
    "Metal Compiler failed to compile metallib. Please submit a bug report."

OBSERVATIONS:
- sin, cos, sqrt work on Metal GPU ✓
- asin, acos work on Metal GPU ✓
- atan (single arg) fails with clear error: "libm operations are only available on CPU targets"
- atan2 (two args) CRASHES the Metal compiler (no useful error message)

To reproduce:
    pixi run -e apple mojo run examples/atan2_gpu_bug_repro.mojo

Expected behavior: Should compute atan2 values on GPU
Actual behavior: Metal compiler crashes/fails

Environment:
    - macOS with Apple Silicon (Metal backend)
    - Mojo nightly
"""

from math import atan2
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor


fn main() raises:
    print("=== atan2 GPU Bug Reproduction ===\n")

    comptime BATCH = 4
    comptime dtype = DType.float32

    var ctx = DeviceContext()
    print("GPU context created")

    var input_y_buf = ctx.enqueue_create_buffer[dtype](BATCH)
    var input_x_buf = ctx.enqueue_create_buffer[dtype](BATCH)
    var output_buf = ctx.enqueue_create_buffer[dtype](BATCH)

    var input_y_host = ctx.enqueue_create_host_buffer[dtype](BATCH)
    var input_x_host = ctx.enqueue_create_host_buffer[dtype](BATCH)

    input_y_host.unsafe_ptr()[0] = 1.0
    input_x_host.unsafe_ptr()[0] = 1.0
    input_y_host.unsafe_ptr()[1] = 1.0
    input_x_host.unsafe_ptr()[1] = 0.0
    input_y_host.unsafe_ptr()[2] = 0.0
    input_x_host.unsafe_ptr()[2] = 1.0
    input_y_host.unsafe_ptr()[3] = -1.0
    input_x_host.unsafe_ptr()[3] = -1.0

    ctx.enqueue_copy(input_y_buf.unsafe_ptr(), input_y_host.unsafe_ptr(), BATCH)
    ctx.enqueue_copy(input_x_buf.unsafe_ptr(), input_x_host.unsafe_ptr(), BATCH)
    ctx.synchronize()
    print("Input buffers initialized")

    var input_y = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](
        input_y_buf.unsafe_ptr()
    )
    var input_x = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](
        input_x_buf.unsafe_ptr()
    )
    var output = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](
        output_buf.unsafe_ptr()
    )

    print("Launching GPU kernel with atan2...")
    print("(Expected: Metal compiler crash)\n")

    @always_inline
    fn atan2_kernel(
        input_y: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        input_x: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        output: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    ):
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH:
            return

        var y = input_y[i]
        var x = input_x[i]

        # BUG: This causes Metal compiler to crash
        var result = atan2(y, x)

        output[i] = result

    ctx.enqueue_function[atan2_kernel, atan2_kernel](
        input_y,
        input_x,
        output,
        grid_dim=(1,),
        block_dim=(BATCH,),
    )
    ctx.synchronize()

    print("Kernel completed (unexpected)!")

    var output_host = ctx.enqueue_create_host_buffer[dtype](BATCH)
    ctx.enqueue_copy(output_host.unsafe_ptr(), output_buf.unsafe_ptr(), BATCH)
    ctx.synchronize()

    print("\nResults:")
    print("  atan2(1, 1) =", Float64(output_host.unsafe_ptr()[0]))
    print("  atan2(1, 0) =", Float64(output_host.unsafe_ptr()[1]))
    print("  atan2(0, 1) =", Float64(output_host.unsafe_ptr()[2]))
    print("  atan2(-1, -1) =", Float64(output_host.unsafe_ptr()[3]))

    print("\n=== Test Complete ===")

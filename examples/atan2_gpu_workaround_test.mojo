"""Test GPU-compatible atan2 implementation.

This tests our polynomial approximation of atan2 that works on Metal GPU.

To run:
    pixi run -e apple mojo run examples/atan2_gpu_workaround_test.mojo
"""

from math import atan2 as atan2_cpu, pi
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor
from physics3d.math_gpu import atan2_gpu


fn main() raises:
    print("=== atan2_gpu Workaround Test ===\n")

    comptime BATCH = 8
    comptime dtype = DType.float32

    var ctx = DeviceContext()
    print("GPU context created")

    # Allocate buffers
    var input_y_buf = ctx.enqueue_create_buffer[dtype](BATCH)
    var input_x_buf = ctx.enqueue_create_buffer[dtype](BATCH)
    var output_buf = ctx.enqueue_create_buffer[dtype](BATCH)

    var input_y_host = ctx.enqueue_create_host_buffer[dtype](BATCH)
    var input_x_host = ctx.enqueue_create_host_buffer[dtype](BATCH)

    # Test cases covering all quadrants and special cases
    # (y, x) pairs
    input_y_host.unsafe_ptr()[0] = 1.0   # Q1: (1, 1) -> pi/4
    input_x_host.unsafe_ptr()[0] = 1.0
    input_y_host.unsafe_ptr()[1] = 1.0   # Q2: (1, -1) -> 3pi/4
    input_x_host.unsafe_ptr()[1] = -1.0
    input_y_host.unsafe_ptr()[2] = -1.0  # Q3: (-1, -1) -> -3pi/4
    input_x_host.unsafe_ptr()[2] = -1.0
    input_y_host.unsafe_ptr()[3] = -1.0  # Q4: (-1, 1) -> -pi/4
    input_x_host.unsafe_ptr()[3] = 1.0
    input_y_host.unsafe_ptr()[4] = 1.0   # +Y axis: (1, 0) -> pi/2
    input_x_host.unsafe_ptr()[4] = 0.0
    input_y_host.unsafe_ptr()[5] = -1.0  # -Y axis: (-1, 0) -> -pi/2
    input_x_host.unsafe_ptr()[5] = 0.0
    input_y_host.unsafe_ptr()[6] = 0.0   # +X axis: (0, 1) -> 0
    input_x_host.unsafe_ptr()[6] = 1.0
    input_y_host.unsafe_ptr()[7] = 0.0   # -X axis: (0, -1) -> pi
    input_x_host.unsafe_ptr()[7] = -1.0

    # Compute expected values on CPU
    print("Expected values (CPU atan2):")
    for i in range(BATCH):
        var y = Float64(input_y_host.unsafe_ptr()[i])
        var x = Float64(input_x_host.unsafe_ptr()[i])
        var expected = atan2_cpu(y, x)
        print("  atan2(", y, ",", x, ") =", expected)

    # Copy to device
    ctx.enqueue_copy(input_y_buf.unsafe_ptr(), input_y_host.unsafe_ptr(), BATCH)
    ctx.enqueue_copy(input_x_buf.unsafe_ptr(), input_x_host.unsafe_ptr(), BATCH)
    ctx.synchronize()

    var input_y = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](
        input_y_buf.unsafe_ptr()
    )
    var input_x = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](
        input_x_buf.unsafe_ptr()
    )
    var output = LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin](
        output_buf.unsafe_ptr()
    )

    print("\nLaunching GPU kernel with atan2_gpu...")

    @always_inline
    fn atan2_gpu_kernel(
        input_y: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        input_x: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
        output: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    ):
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH:
            return

        var y = rebind[Scalar[dtype]](input_y[i])
        var x = rebind[Scalar[dtype]](input_x[i])

        # Use our GPU-compatible implementation
        var result = atan2_gpu[dtype](y, x)

        output[i] = result

    ctx.enqueue_function[atan2_gpu_kernel, atan2_gpu_kernel](
        input_y,
        input_x,
        output,
        grid_dim=(1,),
        block_dim=(BATCH,),
    )
    ctx.synchronize()

    print("GPU kernel completed!")

    # Copy results back
    var output_host = ctx.enqueue_create_host_buffer[dtype](BATCH)
    ctx.enqueue_copy(output_host.unsafe_ptr(), output_buf.unsafe_ptr(), BATCH)
    ctx.synchronize()

    # Compare results
    print("\nGPU results vs CPU expected:")
    var max_error = Float64(0.0)
    for i in range(BATCH):
        var y = Float64(input_y_host.unsafe_ptr()[i])
        var x = Float64(input_x_host.unsafe_ptr()[i])
        var expected = atan2_cpu(y, x)
        var gpu_result = Float64(output_host.unsafe_ptr()[i])
        var error = expected - gpu_result
        if error < 0:
            error = -error
        if error > max_error:
            max_error = error
        print("  atan2(", y, ",", x, "): GPU=", gpu_result, " CPU=", expected, " err=", error)

    print("\nMax error:", max_error, "radians")
    if max_error < 0.01:
        print("PASS: Error within acceptable tolerance")
    else:
        print("WARNING: Error exceeds tolerance")

    print("\n=== Test Complete ===")

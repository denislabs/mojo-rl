"""Test GPU Compilation Time for forward_gpu and backward_gpu.

This tests compilation time for the Linear layer's GPU methods.

Run with:
    pixi run -e apple mojo run tests/test_compile_nn.mojo
"""

from time import perf_counter_ns
from random import seed, random_float64

from gpu.host import DeviceContext, DeviceBuffer

from deep_rl.constants import dtype
from deep_rl.model import Linear

# =============================================================================
# Constants
# =============================================================================

comptime BATCH_SIZE = 4096
comptime INPUT_DIM = 2
comptime HIDDEN_DIM = 4096

# Linear layer constants
comptime LinearLayer = Linear[INPUT_DIM, HIDDEN_DIM]
comptime PARAM_SIZE = LinearLayer.PARAM_SIZE
comptime CACHE_SIZE = LinearLayer.CACHE_SIZE


# =============================================================================
# Main
# =============================================================================


def main():
    seed(42)
    print("=" * 70)
    print("GPU forward_gpu Compilation Test")
    print("=" * 70)
    print()
    print("Linear layer: " + String(INPUT_DIM) + " -> " + String(HIDDEN_DIM))
    print("Batch size: " + String(BATCH_SIZE))
    print("PARAM_SIZE: " + String(PARAM_SIZE))
    print("CACHE_SIZE: " + String(CACHE_SIZE))
    print()

    # =========================================================================
    # Generate input data on host
    # =========================================================================

    var input_host = List[Scalar[dtype]](capacity=BATCH_SIZE * INPUT_DIM)
    var output_host = List[Scalar[dtype]](capacity=BATCH_SIZE * HIDDEN_DIM)
    var params_host = List[Scalar[dtype]](capacity=PARAM_SIZE)
    var cache_host = List[Scalar[dtype]](capacity=BATCH_SIZE * CACHE_SIZE)

    # Initialize input data
    for i in range(BATCH_SIZE * INPUT_DIM):
        var val = Scalar[dtype](random_float64() * 2 - 1)
        input_host.append(val)

    # Initialize output buffer (will be overwritten)
    for i in range(BATCH_SIZE * HIDDEN_DIM):
        output_host.append(Scalar[dtype](0))

    # Initialize params (random weights and biases)
    for i in range(PARAM_SIZE):
        var val = Scalar[dtype](random_float64() * 0.1)
        params_host.append(val)

    # Initialize cache buffer
    for i in range(BATCH_SIZE * CACHE_SIZE):
        cache_host.append(Scalar[dtype](0))

    print("Host data initialized")
    print()

    # =========================================================================
    # Run forward_gpu on GPU
    # =========================================================================

    print("Running forward_gpu on GPU...")
    print("-" * 70)

    var start_time = perf_counter_ns()

    with DeviceContext() as ctx:
        # Create DeviceBuffers from host data
        var input_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * INPUT_DIM)
        var output_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_SIZE * HIDDEN_DIM
        )
        var params_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)
        var cache_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_SIZE * CACHE_SIZE
        )

        # Copy host data to device
        ctx.enqueue_copy(input_buf, input_host.unsafe_ptr())
        ctx.enqueue_copy(params_buf, params_host.unsafe_ptr())

        # Call forward_gpu as a static method on the Linear type
        Linear[INPUT_DIM, HIDDEN_DIM].forward_gpu[BATCH_SIZE](
            ctx,
            output_buf,
            input_buf,
            params_buf,
            cache_buf,
        )

        # Synchronize to ensure kernel completes
        ctx.synchronize()

        var end_time = perf_counter_ns()
        var elapsed_ms = Float64(end_time - start_time) / 1e6

        print("-" * 70)
        print()
        print(
            "  Total time (including compilation): "
            + String(elapsed_ms)[:8]
            + " ms"
        )

        # Copy output back to verify
        ctx.enqueue_copy(output_host.unsafe_ptr(), output_buf)
        ctx.synchronize()

        print()
        print("Sample outputs (first 5):")
        for i in range(5):
            print("  output[" + String(i) + "] = " + String(output_host[i]))

        # =====================================================================
        # Run backward_gpu on GPU
        # =====================================================================

        print()
        print("Running backward_gpu on GPU...")
        print("-" * 70)

        # Create additional buffers for backward pass
        # grad_output: gradient from loss (simulated as all ones)
        var grad_output_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_SIZE * HIDDEN_DIM
        )
        # grad_input: gradient w.r.t. input (computed by backward)
        var grad_input_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_SIZE * INPUT_DIM
        )
        # grads: parameter gradients (dW and db)
        var grads_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)

        # Initialize grad_output with ones (simulating loss gradient)
        var grad_output_host = List[Scalar[dtype]](
            capacity=BATCH_SIZE * HIDDEN_DIM
        )
        for i in range(BATCH_SIZE * HIDDEN_DIM):
            grad_output_host.append(Scalar[dtype](1.0))

        # Initialize grads to zero
        var grads_host = List[Scalar[dtype]](capacity=PARAM_SIZE)
        for i in range(PARAM_SIZE):
            grads_host.append(Scalar[dtype](0))

        ctx.enqueue_copy(grad_output_buf, grad_output_host.unsafe_ptr())
        ctx.enqueue_copy(grads_buf, grads_host.unsafe_ptr())

        var backward_start = perf_counter_ns()

        # Call backward_gpu as a static method
        Linear[INPUT_DIM, HIDDEN_DIM].backward_gpu[BATCH_SIZE](
            ctx,
            grad_input_buf,
            grad_output_buf,
            params_buf,
            cache_buf,
            grads_buf,
        )

        ctx.synchronize()

        var backward_end = perf_counter_ns()
        var backward_ms = Float64(backward_end - backward_start) / 1e6

        print("-" * 70)
        print()
        print(
            "  Backward time (including compilation): "
            + String(backward_ms)[:8]
            + " ms"
        )

        # Copy grad_input back to verify
        var grad_input_host = List[Scalar[dtype]](
            capacity=BATCH_SIZE * INPUT_DIM
        )
        for i in range(BATCH_SIZE * INPUT_DIM):
            grad_input_host.append(Scalar[dtype](0))

        ctx.enqueue_copy(grad_input_host.unsafe_ptr(), grad_input_buf)
        ctx.synchronize()

        print()
        print("Sample grad_input (first 5):")
        for i in range(5):
            print(
                "  grad_input["
                + String(i)
                + "] = "
                + String(grad_input_host[i])
            )

    print()
    print("=" * 70)

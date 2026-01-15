"""Debug test for GPU Training - identifies which kernel causes crashes.

This test adds synchronization after each GPU operation to pinpoint failures.

Run with:
    pixi run -e apple mojo run tests/test_trainer_gpu_debug.mojo
"""

from time import perf_counter_ns
from random import seed, random_float64

from gpu.host import DeviceContext

from deep_rl.constants import dtype
from deep_rl.model import Linear, ReLU, Tanh, seq, Seq2
from deep_rl.loss import MSELoss
from deep_rl.optimizer import Adam, SGD
from deep_rl.initializer import Xavier


# =============================================================================
# Test Configuration - Start small and increase to find the limit
# =============================================================================

comptime BATCH_SIZE = 1024  # Original failing config
comptime INPUT_DIM = 2
comptime HIDDEN_DIM = 1024  # Original failing config
comptime OUTPUT_DIM = 1


def test_buffer_allocation():
    """Test 1: Verify GPU buffer allocation works."""
    print("Test 1: Buffer allocation...")

    comptime PARAM_SIZE = INPUT_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * OUTPUT_DIM + OUTPUT_DIM
    comptime CACHE_SIZE = BATCH_SIZE * (INPUT_DIM + HIDDEN_DIM + HIDDEN_DIM)
    comptime WORKSPACE_SIZE = BATCH_SIZE * (HIDDEN_DIM + HIDDEN_DIM)

    print("  PARAM_SIZE: " + String(PARAM_SIZE))
    print("  CACHE_SIZE: " + String(CACHE_SIZE))
    print("  WORKSPACE_SIZE: " + String(WORKSPACE_SIZE))

    with DeviceContext() as ctx:
        print("  Allocating input buffer...")
        var input_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * INPUT_DIM)
        ctx.synchronize()
        print("    OK")

        print("  Allocating output buffer...")
        var output_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * OUTPUT_DIM)
        ctx.synchronize()
        print("    OK")

        print("  Allocating params buffer...")
        var params_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)
        ctx.synchronize()
        print("    OK")

        print("  Allocating cache buffer...")
        var cache_buf = ctx.enqueue_create_buffer[dtype](CACHE_SIZE)
        ctx.synchronize()
        print("    OK")

        print("  Allocating workspace buffer...")
        var workspace_buf = ctx.enqueue_create_buffer[dtype](WORKSPACE_SIZE)
        ctx.synchronize()
        print("    OK")

    print("Test 1: PASSED")
    print()


def test_linear_forward():
    """Test 2: Test single Linear layer forward pass."""
    print("Test 2: Linear[" + String(INPUT_DIM) + ", " + String(HIDDEN_DIM) + "] forward...")

    with DeviceContext() as ctx:
        comptime IN_DIM = INPUT_DIM
        comptime OUT_DIM = HIDDEN_DIM
        comptime PARAM_SIZE = IN_DIM * OUT_DIM + OUT_DIM
        comptime CACHE_SIZE = BATCH_SIZE * IN_DIM

        var input_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * IN_DIM)
        var output_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * OUT_DIM)
        var params_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)
        var cache_buf = ctx.enqueue_create_buffer[dtype](CACHE_SIZE)

        # Initialize
        ctx.enqueue_memset(input_buf, 1)
        ctx.enqueue_memset(params_buf, 0.01)
        ctx.synchronize()
        print("  Buffers initialized")

        # Forward
        print("  Launching forward kernel...")
        Linear[IN_DIM, OUT_DIM].forward_gpu[BATCH_SIZE](
            ctx, output_buf, input_buf, params_buf, cache_buf
        )
        ctx.synchronize()
        print("    OK")

    print("Test 2: PASSED")
    print()


def test_tanh_forward():
    """Test 3: Test Tanh activation forward pass."""
    print("Test 3: Tanh[" + String(HIDDEN_DIM) + "] forward...")

    with DeviceContext() as ctx:
        comptime DIM = HIDDEN_DIM
        comptime CACHE_SIZE = BATCH_SIZE * DIM

        var input_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * DIM)
        var output_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * DIM)
        var params_buf = ctx.enqueue_create_buffer[dtype](1)  # Unused but required
        var cache_buf = ctx.enqueue_create_buffer[dtype](CACHE_SIZE)

        ctx.enqueue_memset(input_buf, 0.5)
        ctx.synchronize()
        print("  Buffers initialized")

        print("  Launching forward kernel...")
        Tanh[DIM].forward_gpu[BATCH_SIZE](
            ctx, output_buf, input_buf, params_buf, cache_buf
        )
        ctx.synchronize()
        print("    OK")

    print("Test 3: PASSED")
    print()


def test_linear_backward():
    """Test 4: Test Linear layer backward pass."""
    print("Test 4: Linear[" + String(HIDDEN_DIM) + ", " + String(OUTPUT_DIM) + "] backward...")

    with DeviceContext() as ctx:
        comptime IN_DIM = HIDDEN_DIM
        comptime OUT_DIM = OUTPUT_DIM
        comptime PARAM_SIZE = IN_DIM * OUT_DIM + OUT_DIM
        comptime CACHE_SIZE = BATCH_SIZE * IN_DIM

        var grad_input_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * IN_DIM)
        var grad_output_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * OUT_DIM)
        var params_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)
        var cache_buf = ctx.enqueue_create_buffer[dtype](CACHE_SIZE)
        var grads_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)

        ctx.enqueue_memset(grad_output_buf, 1)
        ctx.enqueue_memset(params_buf, 0.01)
        ctx.enqueue_memset(cache_buf, 0.5)
        ctx.enqueue_memset(grads_buf, 0)
        ctx.synchronize()
        print("  Buffers initialized")

        print("  Launching backward kernel...")
        Linear[IN_DIM, OUT_DIM].backward_gpu[BATCH_SIZE](
            ctx, grad_input_buf, grad_output_buf, params_buf, cache_buf, grads_buf
        )
        ctx.synchronize()
        print("    OK")

    print("Test 4: PASSED")
    print()


def test_full_model():
    """Test 5: Test full Sequential model forward/backward."""
    print("Test 5: Full model forward/backward...")

    # Define model type alias for static method calls
    comptime ModelType = Seq2[Seq2[Linear[INPUT_DIM, HIDDEN_DIM], Tanh[HIDDEN_DIM]], Linear[HIDDEN_DIM, OUTPUT_DIM]]

    var model = seq(
        Linear[INPUT_DIM, HIDDEN_DIM](),
        Tanh[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, OUTPUT_DIM](),
    )

    print("  Model created:")
    print("    IN_DIM: " + String(model.IN_DIM))
    print("    OUT_DIM: " + String(model.OUT_DIM))
    print("    PARAM_SIZE: " + String(model.PARAM_SIZE))
    print("    CACHE_SIZE: " + String(model.CACHE_SIZE))
    print("    WORKSPACE_SIZE_PER_SAMPLE: " + String(model.WORKSPACE_SIZE_PER_SAMPLE))

    comptime PARAM_SIZE = ModelType.PARAM_SIZE
    comptime CACHE_SIZE = BATCH_SIZE * ModelType.CACHE_SIZE
    comptime WORKSPACE_SIZE = BATCH_SIZE * ModelType.WORKSPACE_SIZE_PER_SAMPLE

    with DeviceContext() as ctx:
        var input_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * INPUT_DIM)
        var output_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * OUTPUT_DIM)
        var params_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)
        var cache_buf = ctx.enqueue_create_buffer[dtype](CACHE_SIZE)
        var workspace_buf = ctx.enqueue_create_buffer[dtype](
            WORKSPACE_SIZE if WORKSPACE_SIZE > 0 else 1
        )
        var grad_input_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * INPUT_DIM)
        var grad_output_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * OUTPUT_DIM)
        var grads_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)

        ctx.enqueue_memset(input_buf, 0.5)
        ctx.enqueue_memset(params_buf, 0.01)
        ctx.enqueue_memset(grads_buf, 0)
        ctx.enqueue_memset(grad_output_buf, 1)
        ctx.synchronize()
        print("  Buffers initialized")

        print("  Launching forward pass...")
        ModelType.forward_gpu_ws[BATCH_SIZE](
            ctx, output_buf, input_buf, params_buf, cache_buf, workspace_buf
        )
        ctx.synchronize()
        print("    Forward: OK")

        print("  Launching backward pass...")
        ModelType.backward_gpu_ws[BATCH_SIZE](
            ctx, grad_input_buf, grad_output_buf, params_buf, cache_buf, grads_buf, workspace_buf
        )
        ctx.synchronize()
        print("    Backward: OK")

    print("Test 5: PASSED")
    print()


def test_multiple_iterations():
    """Test 6: Multiple forward/backward iterations (like training loop)."""
    print("Test 6: Multiple iterations (10 epochs)...")

    comptime ModelType = Seq2[Seq2[Linear[INPUT_DIM, HIDDEN_DIM], Tanh[HIDDEN_DIM]], Linear[HIDDEN_DIM, OUTPUT_DIM]]
    comptime PARAM_SIZE = ModelType.PARAM_SIZE
    comptime CACHE_SIZE = BATCH_SIZE * ModelType.CACHE_SIZE
    comptime WORKSPACE_SIZE = BATCH_SIZE * ModelType.WORKSPACE_SIZE_PER_SAMPLE

    with DeviceContext() as ctx:
        var input_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * INPUT_DIM)
        var output_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * OUTPUT_DIM)
        var params_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)
        var cache_buf = ctx.enqueue_create_buffer[dtype](CACHE_SIZE)
        var workspace_buf = ctx.enqueue_create_buffer[dtype](
            WORKSPACE_SIZE if WORKSPACE_SIZE > 0 else 1
        )
        var grad_input_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * INPUT_DIM)
        var grad_output_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * OUTPUT_DIM)
        var grads_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)

        ctx.enqueue_memset(input_buf, 0.5)
        ctx.enqueue_memset(params_buf, 0.01)
        ctx.synchronize()
        print("  Buffers initialized")

        for epoch in range(10):
            # Zero grads
            ctx.enqueue_memset(grads_buf, 0)

            # Forward
            ModelType.forward_gpu_ws[BATCH_SIZE](
                ctx, output_buf, input_buf, params_buf, cache_buf, workspace_buf
            )

            # Simulate loss backward (just set grad_output to 1)
            ctx.enqueue_memset(grad_output_buf, 0.1)

            # Backward
            ModelType.backward_gpu_ws[BATCH_SIZE](
                ctx, grad_input_buf, grad_output_buf, params_buf, cache_buf, grads_buf, workspace_buf
            )

            # Sync every iteration to check for errors
            ctx.synchronize()
            print("  Epoch " + String(epoch) + ": OK")

    print("Test 6: PASSED")
    print()


def test_loss_gpu():
    """Test 7: Test MSELoss GPU operations."""
    print("Test 7: MSELoss GPU forward/backward...")

    with DeviceContext() as ctx:
        comptime SIZE = BATCH_SIZE * OUTPUT_DIM

        var output_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        var target_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        var loss_buf = ctx.enqueue_create_buffer[dtype](1)
        var grad_output_buf = ctx.enqueue_create_buffer[dtype](SIZE)

        ctx.enqueue_memset(output_buf, 0.5)
        ctx.enqueue_memset(target_buf, 0.3)
        ctx.synchronize()
        print("  Buffers initialized")

        print("  Launching loss forward...")
        MSELoss.forward_gpu[BATCH_SIZE, OUTPUT_DIM](ctx, loss_buf, output_buf, target_buf)
        ctx.synchronize()
        print("    OK")

        print("  Launching loss backward...")
        MSELoss.backward_gpu[BATCH_SIZE, OUTPUT_DIM](ctx, grad_output_buf, output_buf, target_buf)
        ctx.synchronize()
        print("    OK")

    print("Test 7: PASSED")
    print()


def test_optimizer_gpu():
    """Test 8: Test Adam optimizer GPU step."""
    print("Test 8: Adam optimizer GPU step...")

    comptime ModelType = Seq2[Seq2[Linear[INPUT_DIM, HIDDEN_DIM], Tanh[HIDDEN_DIM]], Linear[HIDDEN_DIM, OUTPUT_DIM]]
    comptime PARAM_SIZE = ModelType.PARAM_SIZE
    comptime STATE_SIZE = PARAM_SIZE * 2  # Adam has 2 states per param (m, v)

    with DeviceContext() as ctx:
        var params_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)
        var grads_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)
        var state_buf = ctx.enqueue_create_buffer[dtype](STATE_SIZE)

        ctx.enqueue_memset(params_buf, 0.01)
        ctx.enqueue_memset(grads_buf, 0.001)
        ctx.enqueue_memset(state_buf, 0)
        ctx.synchronize()
        print("  Buffers initialized")

        var optimizer = Adam(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8)

        print("  Launching optimizer step...")
        optimizer.step_gpu[PARAM_SIZE](ctx, params_buf, grads_buf, state_buf)
        ctx.synchronize()
        print("    OK")

    print("Test 8: PASSED")
    print()


def test_full_training_loop():
    """Test 9: Full training loop with loss and optimizer."""
    print("Test 9: Full training loop (10 epochs with loss + optimizer)...")

    comptime ModelType = Seq2[Seq2[Linear[INPUT_DIM, HIDDEN_DIM], Tanh[HIDDEN_DIM]], Linear[HIDDEN_DIM, OUTPUT_DIM]]
    comptime PARAM_SIZE = ModelType.PARAM_SIZE
    comptime CACHE_SIZE = BATCH_SIZE * ModelType.CACHE_SIZE
    comptime WORKSPACE_SIZE = BATCH_SIZE * ModelType.WORKSPACE_SIZE_PER_SAMPLE
    comptime STATE_SIZE = PARAM_SIZE * 2

    with DeviceContext() as ctx:
        var input_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * INPUT_DIM)
        var target_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * OUTPUT_DIM)
        var output_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * OUTPUT_DIM)
        var params_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)
        var grads_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)
        var cache_buf = ctx.enqueue_create_buffer[dtype](CACHE_SIZE)
        var workspace_buf = ctx.enqueue_create_buffer[dtype](
            WORKSPACE_SIZE if WORKSPACE_SIZE > 0 else 1
        )
        var grad_output_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * OUTPUT_DIM)
        var grad_input_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * INPUT_DIM)
        var state_buf = ctx.enqueue_create_buffer[dtype](STATE_SIZE)
        var loss_buf = ctx.enqueue_create_buffer[dtype](1)

        ctx.enqueue_memset(input_buf, 0.5)
        ctx.enqueue_memset(target_buf, 0.3)
        ctx.enqueue_memset(params_buf, 0.01)
        ctx.enqueue_memset(state_buf, 0)
        ctx.synchronize()
        print("  Buffers initialized")

        var optimizer = Adam(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8)

        for epoch in range(10):
            # Zero grads
            ctx.enqueue_memset(grads_buf, 0)

            # Forward
            ModelType.forward_gpu_ws[BATCH_SIZE](
                ctx, output_buf, input_buf, params_buf, cache_buf, workspace_buf
            )

            # Loss backward
            MSELoss.backward_gpu[BATCH_SIZE, OUTPUT_DIM](
                ctx, grad_output_buf, output_buf, target_buf
            )

            # Model backward
            ModelType.backward_gpu_ws[BATCH_SIZE](
                ctx, grad_input_buf, grad_output_buf, params_buf, cache_buf, grads_buf, workspace_buf
            )

            # Optimizer step
            optimizer.step_gpu[PARAM_SIZE](ctx, params_buf, grads_buf, state_buf)

            ctx.synchronize()
            print("  Epoch " + String(epoch) + ": OK")

    print("Test 9: PASSED")
    print()


def test_trainer_like():
    """Test 10: Mimics Trainer exactly with host buffer copies."""
    print("Test 10: Trainer-like test with host buffer copies...")

    comptime ModelType = Seq2[Seq2[Linear[INPUT_DIM, HIDDEN_DIM], Tanh[HIDDEN_DIM]], Linear[HIDDEN_DIM, OUTPUT_DIM]]
    comptime PARAM_SIZE = ModelType.PARAM_SIZE
    comptime IN_SIZE = BATCH_SIZE * INPUT_DIM
    comptime OUT_SIZE = BATCH_SIZE * OUTPUT_DIM
    comptime CACHE_SIZE = BATCH_SIZE * ModelType.CACHE_SIZE
    comptime WORKSPACE_SIZE = BATCH_SIZE * ModelType.WORKSPACE_SIZE_PER_SAMPLE
    comptime STATE_SIZE = PARAM_SIZE * 2

    # Create data on stack like Trainer does
    var input_data = InlineArray[Scalar[dtype], IN_SIZE](uninitialized=True)
    var target_data = InlineArray[Scalar[dtype], OUT_SIZE](uninitialized=True)
    for i in range(IN_SIZE):
        input_data[i] = 0.5
    for i in range(OUT_SIZE):
        target_data[i] = 0.3

    # Create params like Xavier initializer (simplified)
    var params = InlineArray[Scalar[dtype], PARAM_SIZE](uninitialized=True)
    for i in range(PARAM_SIZE):
        params[i] = Scalar[dtype](random_float64() * 0.1 - 0.05)

    var optimizer_state = InlineArray[Scalar[dtype], STATE_SIZE](uninitialized=True)
    for i in range(STATE_SIZE):
        optimizer_state[i] = 0

    with DeviceContext() as ctx:
        # Create host buffers and copy data (like Trainer)
        var input_host = ctx.enqueue_create_host_buffer[dtype](IN_SIZE)
        var target_host = ctx.enqueue_create_host_buffer[dtype](OUT_SIZE)
        var params_host = ctx.enqueue_create_host_buffer[dtype](PARAM_SIZE)
        var state_host = ctx.enqueue_create_host_buffer[dtype](STATE_SIZE)

        for i in range(IN_SIZE):
            input_host[i] = input_data[i]
        for i in range(OUT_SIZE):
            target_host[i] = target_data[i]
        for i in range(PARAM_SIZE):
            params_host[i] = params[i]
        for i in range(STATE_SIZE):
            state_host[i] = optimizer_state[i]

        print("  Host buffers populated")

        # Create device buffers
        var input_buf = ctx.enqueue_create_buffer[dtype](IN_SIZE)
        var target_buf = ctx.enqueue_create_buffer[dtype](OUT_SIZE)
        var output_buf = ctx.enqueue_create_buffer[dtype](OUT_SIZE)
        var params_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)
        var grads_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)
        var cache_buf = ctx.enqueue_create_buffer[dtype](CACHE_SIZE)
        var workspace_buf = ctx.enqueue_create_buffer[dtype](
            WORKSPACE_SIZE if WORKSPACE_SIZE > 0 else 1
        )
        var grad_output_buf = ctx.enqueue_create_buffer[dtype](OUT_SIZE)
        var grad_input_buf = ctx.enqueue_create_buffer[dtype](IN_SIZE)
        var state_buf = ctx.enqueue_create_buffer[dtype](STATE_SIZE)
        var loss_buf = ctx.enqueue_create_buffer[dtype](1)
        var loss_host = ctx.enqueue_create_host_buffer[dtype](1)

        # Copy to device (like Trainer)
        ctx.enqueue_copy(input_buf, input_host)
        ctx.enqueue_copy(target_buf, target_host)
        ctx.enqueue_copy(params_buf, params_host)
        ctx.enqueue_copy(state_buf, state_host)
        ctx.synchronize()
        print("  Device buffers initialized via copy")

        var optimizer = Adam(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8)

        for epoch in range(50):  # Run 50 epochs like first print interval
            ctx.enqueue_memset(grads_buf, 0)

            ModelType.forward_gpu_ws[BATCH_SIZE](
                ctx, output_buf, input_buf, params_buf, cache_buf, workspace_buf
            )

            MSELoss.backward_gpu[BATCH_SIZE, OUTPUT_DIM](
                ctx, grad_output_buf, output_buf, target_buf
            )

            ModelType.backward_gpu_ws[BATCH_SIZE](
                ctx, grad_input_buf, grad_output_buf, params_buf, cache_buf, grads_buf, workspace_buf
            )

            optimizer.step_gpu[PARAM_SIZE](ctx, params_buf, grads_buf, state_buf)

            # At epoch 0, also call loss forward (like Trainer does at print_every)
            if epoch == 0:
                MSELoss.forward_gpu[BATCH_SIZE, OUTPUT_DIM](
                    ctx, loss_buf, output_buf, target_buf
                )
                ctx.enqueue_copy(loss_host, loss_buf)
                ctx.synchronize()
                print("  Epoch 0 - Loss: " + String(Float64(loss_host[0])))

        # Final sync
        ctx.synchronize()
        print("  50 epochs completed")

    print("Test 10: PASSED")
    print()


def test_many_iterations_no_sync():
    """Test 11: Many iterations without sync (like real training)."""
    print("Test 11: Many iterations without sync (100 epochs)...")

    comptime ModelType = Seq2[Seq2[Linear[INPUT_DIM, HIDDEN_DIM], Tanh[HIDDEN_DIM]], Linear[HIDDEN_DIM, OUTPUT_DIM]]
    comptime PARAM_SIZE = ModelType.PARAM_SIZE
    comptime CACHE_SIZE = BATCH_SIZE * ModelType.CACHE_SIZE
    comptime WORKSPACE_SIZE = BATCH_SIZE * ModelType.WORKSPACE_SIZE_PER_SAMPLE

    with DeviceContext() as ctx:
        var input_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * INPUT_DIM)
        var output_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * OUTPUT_DIM)
        var params_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)
        var cache_buf = ctx.enqueue_create_buffer[dtype](CACHE_SIZE)
        var workspace_buf = ctx.enqueue_create_buffer[dtype](
            WORKSPACE_SIZE if WORKSPACE_SIZE > 0 else 1
        )
        var grad_input_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * INPUT_DIM)
        var grad_output_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * OUTPUT_DIM)
        var grads_buf = ctx.enqueue_create_buffer[dtype](PARAM_SIZE)

        ctx.enqueue_memset(input_buf, 0.5)
        ctx.enqueue_memset(params_buf, 0.01)
        ctx.enqueue_memset(grad_output_buf, 0.1)
        ctx.synchronize()
        print("  Buffers initialized")

        for epoch in range(100):
            # Zero grads
            ctx.enqueue_memset(grads_buf, 0)

            # Forward
            ModelType.forward_gpu_ws[BATCH_SIZE](
                ctx, output_buf, input_buf, params_buf, cache_buf, workspace_buf
            )

            # Backward
            ModelType.backward_gpu_ws[BATCH_SIZE](
                ctx, grad_input_buf, grad_output_buf, params_buf, cache_buf, grads_buf, workspace_buf
            )

            # Only sync every 10 epochs
            if epoch % 10 == 9:
                ctx.synchronize()
                print("  Epoch " + String(epoch) + ": OK")

        ctx.synchronize()

    print("Test 7: PASSED")
    print()


def main():
    seed(42)
    print("=" * 70)
    print("GPU Debug Test Suite")
    print("=" * 70)
    print("BATCH_SIZE: " + String(BATCH_SIZE))
    print("HIDDEN_DIM: " + String(HIDDEN_DIM))
    print()

    test_buffer_allocation()
    test_linear_forward()
    test_tanh_forward()
    test_linear_backward()
    test_full_model()
    test_multiple_iterations()
    test_loss_gpu()
    test_optimizer_gpu()
    test_full_training_loop()
    test_trainer_like()
    test_many_iterations_no_sync()

    print("=" * 70)
    print("All tests passed!")
    print("=" * 70)
    print()
    print("If all tests pass, try increasing BATCH_SIZE and HIDDEN_DIM")
    print("to find the failure point.")

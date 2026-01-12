"""Test for nn_gpu.mojo - GPU Neural Network Module.

This demonstrates training a simple MLP on GPU using kernels from nn_gpu.mojo.
We train on the same task as test_mlp.mojo: y = x1 * x2 (product function).

Run with:
    pixi run -e apple mojo run nn_gpu_test.mojo
"""

from time import perf_counter_ns
from random import seed

from layout import Layout, LayoutTensor
from gpu.host import DeviceContext

from deep_rl.nn_gpu import (
    dtype,
    TILE,
    TPB,
    linear_forward_kernel,
    linear_forward_relu_dual_kernel,
    linear_backward_dx_kernel,
    linear_backward_dx_relu_kernel,
    linear_backward_dW_db_kernel,
    adam_update_kernel,
    mse_loss_backward_kernel,
    mse_loss_kernel,
    xavier_init_kernel,
    generate_data_kernel,
)

# =============================================================================
# Constants
# =============================================================================

comptime BATCH_SIZE = 256
comptime INPUT_DIM = 2
comptime HIDDEN_DIM = 32
comptime OUTPUT_DIM = 1
comptime NUM_EPOCHS = 500

# Parameter sizes
comptime W1_SIZE = INPUT_DIM * HIDDEN_DIM
comptime B1_SIZE = HIDDEN_DIM
comptime W2_SIZE = HIDDEN_DIM * OUTPUT_DIM
comptime B2_SIZE = OUTPUT_DIM


# =============================================================================
# Main
# =============================================================================


def main():
    seed(42)
    print("=" * 60)
    print("GPU Neural Network Module Test (nn_gpu.mojo)")
    print("=" * 60)
    print()
    print(
        "Network: "
        + String(INPUT_DIM)
        + " -> "
        + String(HIDDEN_DIM)
        + " (ReLU) -> "
        + String(OUTPUT_DIM)
    )
    print("Task: Learn y = x1 * x2 (product function)")
    print("Batch size: " + String(BATCH_SIZE))
    print()

    with DeviceContext() as ctx:
        # =================================================================
        # Allocate GPU buffers
        # =================================================================

        print("Allocating buffers...")

        # Layer 1: input -> hidden
        var W1_buf = ctx.enqueue_create_buffer[dtype](W1_SIZE)
        var b1_buf = ctx.enqueue_create_buffer[dtype](B1_SIZE)
        var dW1_buf = ctx.enqueue_create_buffer[dtype](W1_SIZE)
        var db1_buf = ctx.enqueue_create_buffer[dtype](B1_SIZE)
        var m_W1_buf = ctx.enqueue_create_buffer[dtype](W1_SIZE)
        var v_W1_buf = ctx.enqueue_create_buffer[dtype](W1_SIZE)
        var m_b1_buf = ctx.enqueue_create_buffer[dtype](B1_SIZE)
        var v_b1_buf = ctx.enqueue_create_buffer[dtype](B1_SIZE)

        # Layer 2: hidden -> output
        var W2_buf = ctx.enqueue_create_buffer[dtype](W2_SIZE)
        var b2_buf = ctx.enqueue_create_buffer[dtype](B2_SIZE)
        var dW2_buf = ctx.enqueue_create_buffer[dtype](W2_SIZE)
        var db2_buf = ctx.enqueue_create_buffer[dtype](B2_SIZE)
        var m_W2_buf = ctx.enqueue_create_buffer[dtype](W2_SIZE)
        var v_W2_buf = ctx.enqueue_create_buffer[dtype](W2_SIZE)
        var m_b2_buf = ctx.enqueue_create_buffer[dtype](B2_SIZE)
        var v_b2_buf = ctx.enqueue_create_buffer[dtype](B2_SIZE)

        # Activations
        var h1_pre_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * HIDDEN_DIM)
        var h1_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * HIDDEN_DIM)
        var y_pred_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * OUTPUT_DIM)

        # Gradients
        var d_y_pred_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * OUTPUT_DIM)
        var d_h1_pre_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * HIDDEN_DIM)

        # Data
        var x_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * INPUT_DIM)
        var y_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * OUTPUT_DIM)
        var loss_buf = ctx.enqueue_create_buffer[dtype](1)
        var data_rng_buf = ctx.enqueue_create_buffer[DType.uint32](BATCH_SIZE)
        var rng_seed_buf = ctx.enqueue_create_buffer[DType.uint32](1)

        # =================================================================
        # Initialize weights
        # =================================================================

        print("Initializing weights...")

        # Zero initialize moments and biases
        m_W1_buf.enqueue_fill(0)
        v_W1_buf.enqueue_fill(0)
        m_b1_buf.enqueue_fill(0)
        v_b1_buf.enqueue_fill(0)
        m_W2_buf.enqueue_fill(0)
        v_W2_buf.enqueue_fill(0)
        m_b2_buf.enqueue_fill(0)
        v_b2_buf.enqueue_fill(0)
        b1_buf.enqueue_fill(0)
        b2_buf.enqueue_fill(0)

        # Xavier init for weights
        with rng_seed_buf.map_to_host() as host:
            host[0] = UInt32(12345)

        var rng_t = LayoutTensor[DType.uint32, Layout.row_major(1), MutAnyOrigin](
            rng_seed_buf
        )

        var W1_t_init = LayoutTensor[dtype, Layout.row_major(W1_SIZE), MutAnyOrigin](
            W1_buf
        )
        comptime W1_init_blocks = (W1_SIZE + TPB - 1) // TPB
        ctx.enqueue_function_checked[
            xavier_init_kernel[W1_SIZE, INPUT_DIM, HIDDEN_DIM],
            xavier_init_kernel[W1_SIZE, INPUT_DIM, HIDDEN_DIM],
        ](W1_t_init, rng_t, grid_dim=(W1_init_blocks,), block_dim=(TPB,))

        var W2_t_init = LayoutTensor[dtype, Layout.row_major(W2_SIZE), MutAnyOrigin](
            W2_buf
        )
        comptime W2_init_blocks = (W2_SIZE + TPB - 1) // TPB
        ctx.enqueue_function_checked[
            xavier_init_kernel[W2_SIZE, HIDDEN_DIM, OUTPUT_DIM],
            xavier_init_kernel[W2_SIZE, HIDDEN_DIM, OUTPUT_DIM],
        ](W2_t_init, rng_t, grid_dim=(W2_init_blocks,), block_dim=(TPB,))

        # Initialize data RNG seeds
        with data_rng_buf.map_to_host() as host:
            for i in range(BATCH_SIZE):
                host[i] = UInt32(i * 1099087573 + 42)

        ctx.synchronize()
        print("Weights initialized!")

        # =================================================================
        # Create tensor views
        # =================================================================

        var W1_t = LayoutTensor[
            dtype, Layout.row_major(INPUT_DIM, HIDDEN_DIM), MutAnyOrigin
        ](W1_buf)
        var b1_t = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), MutAnyOrigin](
            b1_buf
        )
        var dW1_t = LayoutTensor[
            dtype, Layout.row_major(INPUT_DIM, HIDDEN_DIM), MutAnyOrigin
        ](dW1_buf)
        var db1_t = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), MutAnyOrigin](
            db1_buf
        )
        var m_W1_t = LayoutTensor[dtype, Layout.row_major(W1_SIZE), MutAnyOrigin](
            m_W1_buf
        )
        var v_W1_t = LayoutTensor[dtype, Layout.row_major(W1_SIZE), MutAnyOrigin](
            v_W1_buf
        )
        var m_b1_t = LayoutTensor[dtype, Layout.row_major(B1_SIZE), MutAnyOrigin](
            m_b1_buf
        )
        var v_b1_t = LayoutTensor[dtype, Layout.row_major(B1_SIZE), MutAnyOrigin](
            v_b1_buf
        )

        var W2_t = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, OUTPUT_DIM), MutAnyOrigin
        ](W2_buf)
        var b2_t = LayoutTensor[dtype, Layout.row_major(OUTPUT_DIM), MutAnyOrigin](
            b2_buf
        )
        var dW2_t = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, OUTPUT_DIM), MutAnyOrigin
        ](dW2_buf)
        var db2_t = LayoutTensor[dtype, Layout.row_major(OUTPUT_DIM), MutAnyOrigin](
            db2_buf
        )
        var m_W2_t = LayoutTensor[dtype, Layout.row_major(W2_SIZE), MutAnyOrigin](
            m_W2_buf
        )
        var v_W2_t = LayoutTensor[dtype, Layout.row_major(W2_SIZE), MutAnyOrigin](
            v_W2_buf
        )
        var m_b2_t = LayoutTensor[dtype, Layout.row_major(B2_SIZE), MutAnyOrigin](
            m_b2_buf
        )
        var v_b2_t = LayoutTensor[dtype, Layout.row_major(B2_SIZE), MutAnyOrigin](
            v_b2_buf
        )

        var h1_pre_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, HIDDEN_DIM), MutAnyOrigin
        ](h1_pre_buf)
        var h1_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, HIDDEN_DIM), MutAnyOrigin
        ](h1_buf)
        var y_pred_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, OUTPUT_DIM), MutAnyOrigin
        ](y_pred_buf)

        var d_y_pred_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, OUTPUT_DIM), MutAnyOrigin
        ](d_y_pred_buf)
        var d_h1_pre_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, HIDDEN_DIM), MutAnyOrigin
        ](d_h1_pre_buf)

        var x_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, INPUT_DIM), MutAnyOrigin
        ](x_buf)
        var y_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, OUTPUT_DIM), MutAnyOrigin
        ](y_buf)
        var loss_t = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](loss_buf)
        var data_rng_t = LayoutTensor[
            DType.uint32, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](data_rng_buf)

        # =================================================================
        # Compile kernels
        # =================================================================

        print("Compiling kernels...")

        # Forward kernels
        var fwd_layer1_fn = ctx.compile_function_checked[
            linear_forward_relu_dual_kernel[BATCH_SIZE, INPUT_DIM, HIDDEN_DIM],
            linear_forward_relu_dual_kernel[BATCH_SIZE, INPUT_DIM, HIDDEN_DIM],
        ]()
        var fwd_layer2_fn = ctx.compile_function_checked[
            linear_forward_kernel[BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM],
            linear_forward_kernel[BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM],
        ]()

        # Loss kernels
        var mse_loss_fn = ctx.compile_function_checked[
            mse_loss_kernel[BATCH_SIZE, OUTPUT_DIM],
            mse_loss_kernel[BATCH_SIZE, OUTPUT_DIM],
        ]()
        var mse_backward_fn = ctx.compile_function_checked[
            mse_loss_backward_kernel[BATCH_SIZE, OUTPUT_DIM],
            mse_loss_backward_kernel[BATCH_SIZE, OUTPUT_DIM],
        ]()

        # Backward kernels
        var bwd_dW_db2_fn = ctx.compile_function_checked[
            linear_backward_dW_db_kernel[BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM],
            linear_backward_dW_db_kernel[BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM],
        ]()
        var bwd_dx_relu_fn = ctx.compile_function_checked[
            linear_backward_dx_relu_kernel[BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM],
            linear_backward_dx_relu_kernel[BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM],
        ]()
        var bwd_dW_db1_fn = ctx.compile_function_checked[
            linear_backward_dW_db_kernel[BATCH_SIZE, INPUT_DIM, HIDDEN_DIM],
            linear_backward_dW_db_kernel[BATCH_SIZE, INPUT_DIM, HIDDEN_DIM],
        ]()

        # Adam kernels
        var adam_W1_fn = ctx.compile_function_checked[
            adam_update_kernel[W1_SIZE],
            adam_update_kernel[W1_SIZE],
        ]()
        var adam_b1_fn = ctx.compile_function_checked[
            adam_update_kernel[B1_SIZE],
            adam_update_kernel[B1_SIZE],
        ]()
        var adam_W2_fn = ctx.compile_function_checked[
            adam_update_kernel[W2_SIZE],
            adam_update_kernel[W2_SIZE],
        ]()
        var adam_b2_fn = ctx.compile_function_checked[
            adam_update_kernel[B2_SIZE],
            adam_update_kernel[B2_SIZE],
        ]()

        # Data generation kernel
        var data_gen_fn = ctx.compile_function_checked[
            generate_data_kernel[BATCH_SIZE, INPUT_DIM, OUTPUT_DIM],
            generate_data_kernel[BATCH_SIZE, INPUT_DIM, OUTPUT_DIM],
        ]()

        print("Kernels compiled!")
        print()

        # =================================================================
        # Generate training data
        # =================================================================

        print("Generating training data...")
        comptime data_gen_blocks = (BATCH_SIZE + TPB - 1) // TPB
        ctx.enqueue_function_checked(
            data_gen_fn,
            x_t,
            y_t,
            data_rng_t,
            grid_dim=(data_gen_blocks,),
            block_dim=(TPB,),
        )
        ctx.synchronize()
        print("Data generated!")
        print()

        # =================================================================
        # Training hyperparameters
        # =================================================================

        var lr = Scalar[dtype](0.001)
        var beta1 = Scalar[dtype](0.9)
        var beta2 = Scalar[dtype](0.999)
        var eps = Scalar[dtype](1e-8)

        # Grid dimensions
        comptime grid_h1 = (
            (HIDDEN_DIM + TILE - 1) // TILE,
            (BATCH_SIZE + TILE - 1) // TILE,
        )
        comptime grid_out = (
            (OUTPUT_DIM + TILE - 1) // TILE,
            (BATCH_SIZE + TILE - 1) // TILE,
        )
        comptime grid_dW1 = (
            (HIDDEN_DIM + TILE - 1) // TILE,
            (INPUT_DIM + TILE - 1) // TILE,
        )
        comptime grid_dW2 = (
            (OUTPUT_DIM + TILE - 1) // TILE,
            (HIDDEN_DIM + TILE - 1) // TILE,
        )
        comptime grid_dx_h1 = (
            (HIDDEN_DIM + TILE - 1) // TILE,
            (BATCH_SIZE + TILE - 1) // TILE,
        )
        comptime block_2d = (TILE, TILE)
        comptime mse_blocks = (BATCH_SIZE * OUTPUT_DIM + TPB - 1) // TPB
        comptime adam_W1_blocks = (W1_SIZE + TPB - 1) // TPB
        comptime adam_b1_blocks = (B1_SIZE + TPB - 1) // TPB
        comptime adam_W2_blocks = (W2_SIZE + TPB - 1) // TPB
        comptime adam_b2_blocks = (B2_SIZE + TPB - 1) // TPB

        # =================================================================
        # Training loop
        # =================================================================

        print("Training...")
        print("-" * 60)

        var start_time = perf_counter_ns()
        var print_every = 50

        # Flat views for Adam
        var W1_flat = LayoutTensor[dtype, Layout.row_major(W1_SIZE), MutAnyOrigin](
            W1_buf
        )
        var dW1_flat = LayoutTensor[dtype, Layout.row_major(W1_SIZE), ImmutAnyOrigin](
            dW1_buf
        )
        var b1_flat = LayoutTensor[dtype, Layout.row_major(B1_SIZE), MutAnyOrigin](
            b1_buf
        )
        var db1_flat = LayoutTensor[dtype, Layout.row_major(B1_SIZE), ImmutAnyOrigin](
            db1_buf
        )
        var W2_flat = LayoutTensor[dtype, Layout.row_major(W2_SIZE), MutAnyOrigin](
            W2_buf
        )
        var dW2_flat = LayoutTensor[dtype, Layout.row_major(W2_SIZE), ImmutAnyOrigin](
            dW2_buf
        )
        var b2_flat = LayoutTensor[dtype, Layout.row_major(B2_SIZE), MutAnyOrigin](
            b2_buf
        )
        var db2_flat = LayoutTensor[dtype, Layout.row_major(B2_SIZE), ImmutAnyOrigin](
            db2_buf
        )

        for epoch in range(NUM_EPOCHS):
            # =========================================================
            # Forward pass
            # =========================================================

            # Layer 1: x -> h1_pre, h1 (fused linear + dual relu)
            ctx.enqueue_function_checked(
                fwd_layer1_fn,
                h1_pre_t,  # pre-ReLU output
                h1_t,  # post-ReLU output
                x_t,
                W1_t,
                b1_t,
                grid_dim=grid_h1,
                block_dim=block_2d,
            )

            # Layer 2: h1 -> y_pred
            ctx.enqueue_function_checked(
                fwd_layer2_fn,
                y_pred_t,
                h1_t,
                W2_t,
                b2_t,
                grid_dim=grid_out,
                block_dim=block_2d,
            )

            # =========================================================
            # Compute loss gradient
            # =========================================================

            ctx.enqueue_function_checked(
                mse_backward_fn,
                d_y_pred_t,
                y_pred_t,
                y_t,
                grid_dim=(mse_blocks,),
                block_dim=(TPB,),
            )

            # =========================================================
            # Backward pass
            # =========================================================

            # Layer 2 backward: dW2, db2
            ctx.enqueue_function_checked(
                bwd_dW_db2_fn,
                dW2_t,
                db2_t,
                h1_t,
                d_y_pred_t,
                grid_dim=grid_dW2,
                block_dim=block_2d,
            )

            # Fused: dx_h1 + relu backward
            ctx.enqueue_function_checked(
                bwd_dx_relu_fn,
                d_h1_pre_t,
                d_y_pred_t,
                W2_t,
                h1_pre_t,
                grid_dim=grid_dx_h1,
                block_dim=block_2d,
            )

            # Layer 1 backward: dW1, db1
            ctx.enqueue_function_checked(
                bwd_dW_db1_fn,
                dW1_t,
                db1_t,
                x_t,
                d_h1_pre_t,
                grid_dim=grid_dW1,
                block_dim=block_2d,
            )

            # =========================================================
            # Adam updates
            # =========================================================

            var t = Scalar[dtype](epoch + 1)
            var bc1 = Scalar[dtype](1) - beta1**t
            var bc2 = Scalar[dtype](1) - beta2**t

            ctx.enqueue_function_checked(
                adam_W1_fn,
                W1_flat,
                dW1_flat,
                m_W1_t,
                v_W1_t,
                lr,
                beta1,
                beta2,
                eps,
                bc1,
                bc2,
                grid_dim=(adam_W1_blocks,),
                block_dim=(TPB,),
            )

            ctx.enqueue_function_checked(
                adam_b1_fn,
                b1_flat,
                db1_flat,
                m_b1_t,
                v_b1_t,
                lr,
                beta1,
                beta2,
                eps,
                bc1,
                bc2,
                grid_dim=(adam_b1_blocks,),
                block_dim=(TPB,),
            )

            ctx.enqueue_function_checked(
                adam_W2_fn,
                W2_flat,
                dW2_flat,
                m_W2_t,
                v_W2_t,
                lr,
                beta1,
                beta2,
                eps,
                bc1,
                bc2,
                grid_dim=(adam_W2_blocks,),
                block_dim=(TPB,),
            )

            ctx.enqueue_function_checked(
                adam_b2_fn,
                b2_flat,
                db2_flat,
                m_b2_t,
                v_b2_t,
                lr,
                beta1,
                beta2,
                eps,
                bc1,
                bc2,
                grid_dim=(adam_b2_blocks,),
                block_dim=(TPB,),
            )

            # =========================================================
            # Print progress
            # =========================================================

            if (epoch + 1) % print_every == 0 or epoch == 0:
                ctx.enqueue_function_checked(
                    mse_loss_fn,
                    loss_t,
                    y_pred_t,
                    y_t,
                    grid_dim=(1,),
                    block_dim=(TPB,),
                )
                ctx.synchronize()

                with loss_buf.map_to_host() as host:
                    var loss_val = Float32(host[0])
                    print(
                        "Epoch "
                        + String(epoch + 1)
                        + "/"
                        + String(NUM_EPOCHS)
                        + " - Loss: "
                        + String(loss_val)
                    )

        var end_time = perf_counter_ns()
        var elapsed_ms = Float64(end_time - start_time) / 1e6

        ctx.synchronize()

        print("-" * 60)
        print()
        print("Training completed in " + String(elapsed_ms)[:8] + " ms")
        print(
            "Average time per epoch: "
            + String(elapsed_ms / Float64(NUM_EPOCHS))[:6]
            + " ms"
        )
        print(
            "Throughput: "
            + String(
                Int(
                    Float64(NUM_EPOCHS)
                    * Float64(BATCH_SIZE)
                    / (elapsed_ms / 1000.0)
                )
            )
            + " samples/sec"
        )

        # =================================================================
        # Final evaluation
        # =================================================================

        print()
        print("=" * 60)
        print("Final Evaluation")
        print("=" * 60)

        # Generate fresh test data
        ctx.enqueue_function_checked(
            data_gen_fn,
            x_t,
            y_t,
            data_rng_t,
            grid_dim=(data_gen_blocks,),
            block_dim=(TPB,),
        )

        # Forward pass
        ctx.enqueue_function_checked(
            fwd_layer1_fn,
            h1_pre_t,
            h1_t,
            x_t,
            W1_t,
            b1_t,
            grid_dim=grid_h1,
            block_dim=block_2d,
        )
        ctx.enqueue_function_checked(
            fwd_layer2_fn,
            y_pred_t,
            h1_t,
            W2_t,
            b2_t,
            grid_dim=grid_out,
            block_dim=block_2d,
        )

        # Compute final loss
        ctx.enqueue_function_checked(
            mse_loss_fn,
            loss_t,
            y_pred_t,
            y_t,
            grid_dim=(1,),
            block_dim=(TPB,),
        )
        ctx.synchronize()

        print()
        with loss_buf.map_to_host() as host:
            var final_loss = Float32(host[0])
            print("Test MSE Loss: " + String(final_loss))

        print()
        print("Sample predictions (x1, x2) -> predicted vs actual:")
        with x_buf.map_to_host() as x_host:
            with y_pred_buf.map_to_host() as pred_host:
                with y_buf.map_to_host() as target_host:
                    for i in range(5):
                        var x1 = Float32(x_host[i * INPUT_DIM + 0])
                        var x2 = Float32(x_host[i * INPUT_DIM + 1])
                        var pred = Float32(pred_host[i])
                        var target = Float32(target_host[i])
                        print(
                            "  ("
                            + String(x1)[:6]
                            + ", "
                            + String(x2)[:6]
                            + ") -> "
                            + String(pred)[:7]
                            + " vs "
                            + String(target)[:7]
                        )

        print()
        print("=" * 60)
        print("Test completed!")
        print("=" * 60)

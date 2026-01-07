"""Test deeprl package on XOR problem.

XOR is the classic test for neural networks - it requires a hidden layer
to learn because the problem is not linearly separable.

XOR truth table:
    (0, 0) -> 0
    (0, 1) -> 1
    (1, 0) -> 1
    (1, 1) -> 0

Run with:
    pixi run mojo run examples/test_deeprl_xor.mojo
"""

from deeprl import (
    MLP2,
    sigmoid,
    sigmoid_grad,
    elementwise_mul,
    elementwise_sub,
    scale,
    mean_all,
    print_matrix,
    zeros,
)


fn train_xor():
    """Train a 2-layer MLP on XOR problem."""
    print("=" * 60)
    print("Training MLP on XOR Problem (deeprl package)")
    print("=" * 60)

    # XOR dataset - compile-time dimensions
    # 4 samples, 2 features each
    comptime batch_size = 4
    comptime input_dim = 2
    comptime hidden_dim = 8
    comptime output_dim = 1

    # Input data: (0,0), (0,1), (1,0), (1,1)
    var X = InlineArray[Float64, batch_size * input_dim](fill=0.0)
    X[0] = 0.0
    X[1] = 0.0  # (0, 0)
    X[2] = 0.0
    X[3] = 1.0  # (0, 1)
    X[4] = 1.0
    X[5] = 0.0  # (1, 0)
    X[6] = 1.0
    X[7] = 1.0  # (1, 1)

    # Target data: 0, 1, 1, 0
    var Y = InlineArray[Float64, batch_size * output_dim](fill=0.0)
    Y[0] = 0.0
    Y[1] = 1.0
    Y[2] = 1.0
    Y[3] = 0.0

    print("\nTraining data:")
    print_matrix[batch_size, input_dim](X, "X")
    print_matrix[batch_size, output_dim](Y, "Y")

    # Create MLP: 2 -> 8 (tanh) -> 1
    var mlp = MLP2[input_dim, hidden_dim, output_dim]()
    mlp.print_info("XOR MLP")

    # Cache buffers for backward pass
    var h_cache = InlineArray[Float64, batch_size * hidden_dim](fill=0.0)
    var h_pre_cache = InlineArray[Float64, batch_size * hidden_dim](fill=0.0)

    # Training parameters
    var learning_rate: Float64 = 0.5
    comptime num_epochs = 2000
    comptime print_every = 200

    print("\nTraining...")
    print("Learning rate:", learning_rate)
    print("Epochs:", num_epochs)
    print("-" * 40)

    for epoch in range(num_epochs):
        # Zero gradients
        mlp.zero_grad()

        # Forward pass with cache for backward
        var out_pre = mlp.forward_with_cache[batch_size](X, h_cache, h_pre_cache)

        # Apply sigmoid to output for binary classification
        var pred = sigmoid[batch_size * output_dim](out_pre)

        # Compute MSE loss: L = mean((pred - Y)^2)
        var diff = elementwise_sub[batch_size * output_dim](pred, Y)
        var sq_diff = elementwise_mul[batch_size * output_dim](diff, diff)
        var loss = mean_all[batch_size * output_dim](sq_diff)

        # Backward pass
        # dL/dpred = 2 * (pred - Y) / n
        var dy_pred = scale[batch_size * output_dim](diff, 2.0 / Float64(batch_size))

        # Multiply by sigmoid gradient
        var sig_grad = sigmoid_grad[batch_size * output_dim](pred)
        var dy = elementwise_mul[batch_size * output_dim](dy_pred, sig_grad)

        # Backward through MLP
        _ = mlp.backward[batch_size](dy, X, h_cache)

        # Update weights
        mlp.update(learning_rate)

        # Print progress
        if (epoch + 1) % print_every == 0 or epoch == 0:
            print("Epoch", epoch + 1, "| Loss:", String(loss)[:10])

    # Final predictions
    print("\n" + "-" * 40)
    print("Final predictions:")

    var final_out = mlp.forward_with_cache[batch_size](X, h_cache, h_pre_cache)
    var final_pred = sigmoid[batch_size * output_dim](final_out)

    var correct = 0
    for i in range(batch_size):
        var x0 = X[i * input_dim]
        var x1 = X[i * input_dim + 1]
        var target = Y[i]
        var predicted = final_pred[i]
        var rounded: Float64 = 1.0 if predicted > 0.5 else 0.0

        print(
            "  (" + String(Int(x0)) + "," + String(Int(x1)) + ") -> " +
            "target: " + String(Int(target)) +
            ", pred: " + String(predicted)[:6] +
            ", rounded: " + String(Int(rounded))
        )

        if rounded == target:
            correct += 1

    print("\nAccuracy:", correct, "/ 4")
    if correct == 4:
        print("XOR SOLVED!")
    else:
        print("XOR not fully solved.")


fn test_linear():
    """Test basic linear layer operations."""
    print("=" * 60)
    print("Testing Linear Layer")
    print("=" * 60)

    from deeprl import Linear, print_matrix

    comptime in_features = 3
    comptime out_features = 2
    comptime batch_size = 2

    var layer = Linear[in_features, out_features]()
    layer.print_info("test_layer")

    # Create input
    var x = InlineArray[Float64, batch_size * in_features](fill=0.0)
    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0
    x[3] = 4.0
    x[4] = 5.0
    x[5] = 6.0

    print("\nInput:")
    print_matrix[batch_size, in_features](x, "x")

    # Forward pass
    var y = layer.forward[batch_size](x)
    print("\nOutput:")
    print_matrix[batch_size, out_features](y, "y = x @ W + b")

    print("\nLinear layer test passed!")


fn test_matmul():
    """Test matrix multiplication."""
    print("=" * 60)
    print("Testing Matrix Multiplication")
    print("=" * 60)

    from deeprl import matmul, print_matrix

    # A = [[1, 2], [3, 4]]
    # B = [[5, 6], [7, 8]]
    # C = A @ B = [[19, 22], [43, 50]]

    var A = InlineArray[Float64, 4](fill=0.0)
    A[0] = 1.0
    A[1] = 2.0
    A[2] = 3.0
    A[3] = 4.0

    var B = InlineArray[Float64, 4](fill=0.0)
    B[0] = 5.0
    B[1] = 6.0
    B[2] = 7.0
    B[3] = 8.0

    print_matrix[2, 2](A, "A")
    print_matrix[2, 2](B, "B")

    var C = matmul[2, 2, 2](A, B)
    print_matrix[2, 2](C, "A @ B")

    print("Expected: [[19, 22], [43, 50]]")

    var correct = (C[0] == 19.0 and C[1] == 22.0 and C[2] == 43.0 and C[3] == 50.0)
    if correct:
        print("Matmul test PASSED!")
    else:
        print("Matmul test FAILED!")


fn main() raises:
    print("Deep RL Neural Network Tests")
    print("=" * 60)
    print("")

    test_matmul()
    print("")
    test_linear()
    print("")
    train_xor()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

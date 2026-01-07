"""Linear (fully connected) layer with compile-time dimensions.

A linear layer computes: y = x @ W + b

Where:
    x: input tensor of shape (batch_size, in_features)
    W: weight matrix of shape (in_features, out_features)
    b: bias vector of shape (out_features,)
    y: output tensor of shape (batch_size, out_features)

All dimensions are compile-time known for maximum performance.
"""

from .tensor import (
    zeros,
    xavier_init,
)


struct Linear[in_features: Int, out_features: Int, dtype: DType = DType.float64]:
    """Linear layer with compile-time dimensions.

    Parameters:
        in_features: Number of input features.
        out_features: Number of output features.
        dtype: Data type for weights and computations.
    """

    # Weights and biases
    var W: InlineArray[Scalar[Self.dtype], Self.in_features * Self.out_features]
    var b: InlineArray[Scalar[Self.dtype], Self.out_features]

    # Gradients
    var dW: InlineArray[Scalar[Self.dtype], Self.in_features * Self.out_features]
    var db: InlineArray[Scalar[Self.dtype], Self.out_features]

    # Cache for backward pass
    var _batch_size: Int

    fn __init__(out self):
        """Initialize with Xavier/Glorot initialization."""
        self.W = xavier_init[Self.in_features * Self.out_features, Self.in_features, Self.out_features, Self.dtype]()
        self.b = zeros[Self.out_features, Self.dtype]()
        self.dW = zeros[Self.in_features * Self.out_features, Self.dtype]()
        self.db = zeros[Self.out_features, Self.dtype]()
        self._batch_size = 0

    fn forward[batch_size: Int](
        mut self,
        x: InlineArray[Scalar[Self.dtype], batch_size * Self.in_features],
    ) -> InlineArray[Scalar[Self.dtype], batch_size * Self.out_features]:
        """Forward pass: y = x @ W + b.

        Args:
            x: Input tensor of shape (batch_size, in_features).

        Returns:
            Output tensor of shape (batch_size, out_features).
        """
        # Store batch size for backward
        self._batch_size = batch_size

        # Compute y = x @ W + b (with broadcasting)
        var result = InlineArray[Scalar[Self.dtype], batch_size * Self.out_features](fill=0)

        for i in range(batch_size):
            for j in range(Self.out_features):
                var sum: Scalar[Self.dtype] = self.b[j]
                for k in range(Self.in_features):
                    sum += x[i * Self.in_features + k] * self.W[k * Self.out_features + j]
                result[i * Self.out_features + j] = sum

        return result^

    fn backward[batch_size: Int](
        mut self,
        dy: InlineArray[Scalar[Self.dtype], batch_size * Self.out_features],
        x: InlineArray[Scalar[Self.dtype], batch_size * Self.in_features],
    ) -> InlineArray[Scalar[Self.dtype], batch_size * Self.in_features]:
        """Backward pass: compute gradients and return input gradient.

        Args:
            dy: Upstream gradient of shape (batch_size, out_features).
            x: Cached input of shape (batch_size, in_features).

        Returns:
            Input gradient of shape (batch_size, in_features).
        """
        # Zero gradients
        for i in range(Self.in_features * Self.out_features):
            self.dW[i] = 0
        for i in range(Self.out_features):
            self.db[i] = 0

        # dW = x.T @ dy -> (in_features, out_features)
        for j in range(Self.in_features):
            for k in range(Self.out_features):
                var sum: Scalar[Self.dtype] = 0
                for i in range(batch_size):
                    sum += x[i * Self.in_features + j] * dy[i * Self.out_features + k]
                self.dW[j * Self.out_features + k] = sum

        # db = sum(dy, axis=0) -> (out_features,)
        for j in range(Self.out_features):
            var sum: Scalar[Self.dtype] = 0
            for i in range(batch_size):
                sum += dy[i * Self.out_features + j]
            self.db[j] = sum

        # dx = dy @ W.T -> (batch_size, in_features)
        var dx = InlineArray[Scalar[Self.dtype], batch_size * Self.in_features](fill=0)
        for i in range(batch_size):
            for j in range(Self.in_features):
                var sum: Scalar[Self.dtype] = 0
                for k in range(Self.out_features):
                    sum += dy[i * Self.out_features + k] * self.W[j * Self.out_features + k]
                dx[i * Self.in_features + j] = sum

        return dx^

    fn update(mut self, learning_rate: Scalar[Self.dtype]):
        """Update weights using SGD: W -= lr * dW, b -= lr * db."""
        for i in range(Self.in_features * Self.out_features):
            self.W[i] -= learning_rate * self.dW[i]
        for i in range(Self.out_features):
            self.b[i] -= learning_rate * self.db[i]

    fn zero_grad(mut self):
        """Reset gradients to zero."""
        for i in range(Self.in_features * Self.out_features):
            self.dW[i] = 0
        for i in range(Self.out_features):
            self.db[i] = 0

    fn num_parameters(self) -> Int:
        """Return total number of learnable parameters."""
        return Self.in_features * Self.out_features + Self.out_features

    fn print_info(self, name: String = "Linear"):
        """Print layer information."""
        print(
            name + ": " + String(Self.in_features) + " -> " + String(Self.out_features) +
            " (" + String(self.num_parameters()) + " params)"
        )

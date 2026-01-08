"""Multi-Layer Perceptron with compile-time architecture.

For deep RL, we typically use fixed architectures like:
- Actor: obs_dim -> 256 -> 256 -> action_dim
- Critic: obs_dim + action_dim -> 256 -> 256 -> 1

All dimensions are compile-time known for maximum performance.
"""

from .tensor import (
    tanh_activation,
    tanh_grad,
    elementwise_mul,
    zeros,
)
from .linear import Linear


# =============================================================================
# Two-Layer MLP (one hidden layer)
# =============================================================================


struct MLP2[
    input_dim: Int,
    hidden_dim: Int,
    output_dim: Int,
    dtype: DType = DType.float64,
]:
    """Two-layer MLP: input -> hidden (tanh) -> output.

    Architecture: input_dim -> hidden_dim -> output_dim.
    """

    var layer1: Linear[Self.input_dim, Self.hidden_dim, Self.dtype]
    var layer2: Linear[Self.hidden_dim, Self.output_dim, Self.dtype]

    fn __init__(out self):
        """Initialize MLP with Xavier initialization."""
        self.layer1 = Linear[Self.input_dim, Self.hidden_dim, Self.dtype]()
        self.layer2 = Linear[Self.hidden_dim, Self.output_dim, Self.dtype]()

    fn forward[batch_size: Int](
        mut self,
        x: InlineArray[Scalar[Self.dtype], batch_size * Self.input_dim],
    ) -> InlineArray[Scalar[Self.dtype], batch_size * Self.output_dim]:
        """Forward pass with tanh hidden activation.

        Args:
            x: Input of shape (batch_size, input_dim).

        Returns:
            Output of shape (batch_size, output_dim).
        """
        # Layer 1: linear + tanh
        var h_pre = self.layer1.forward[batch_size](x)
        var h = tanh_activation[batch_size * Self.hidden_dim, Self.dtype](h_pre)

        # Layer 2: linear (no activation for flexibility)
        var out = self.layer2.forward[batch_size](h)

        return out^

    fn forward_with_cache[batch_size: Int](
        mut self,
        x: InlineArray[Scalar[Self.dtype], batch_size * Self.input_dim],
        mut h_out: InlineArray[Scalar[Self.dtype], batch_size * Self.hidden_dim],
        mut h_pre_out: InlineArray[Scalar[Self.dtype], batch_size * Self.hidden_dim],
    ) -> InlineArray[Scalar[Self.dtype], batch_size * Self.output_dim]:
        """Forward pass storing intermediate activations for backward.

        Args:
            x: Input of shape (batch_size, input_dim).
            h_out: Output buffer for hidden activations (post-tanh).
            h_pre_out: Output buffer for pre-activation hidden.

        Returns:
            Output of shape (batch_size, output_dim).
        """
        # Layer 1: linear + tanh
        var h_pre = self.layer1.forward[batch_size](x)
        var h = tanh_activation[batch_size * Self.hidden_dim, Self.dtype](h_pre)

        # Store in output buffers
        for i in range(batch_size * Self.hidden_dim):
            h_out[i] = h[i]
            h_pre_out[i] = h_pre[i]

        # Layer 2: linear
        var out = self.layer2.forward[batch_size](h)

        return out^

    fn backward[batch_size: Int](
        mut self,
        dy: InlineArray[Scalar[Self.dtype], batch_size * Self.output_dim],
        x: InlineArray[Scalar[Self.dtype], batch_size * Self.input_dim],
        h: InlineArray[Scalar[Self.dtype], batch_size * Self.hidden_dim],
    ) -> InlineArray[Scalar[Self.dtype], batch_size * Self.input_dim]:
        """Backward pass through the MLP.

        Args:
            dy: Upstream gradient of shape (batch_size, output_dim).
            x: Original input of shape (batch_size, input_dim).
            h: Hidden activations (post-tanh) of shape (batch_size, hidden_dim).

        Returns:
            Input gradient of shape (batch_size, input_dim).
        """
        # Backward through layer2
        var dh = self.layer2.backward[batch_size](dy, h)

        # Backward through tanh: dh_pre = dh * tanh_grad(h)
        var tanh_g = tanh_grad[batch_size * Self.hidden_dim, Self.dtype](h)
        var dh_pre = elementwise_mul[batch_size * Self.hidden_dim, Self.dtype](dh, tanh_g)

        # Backward through layer1
        var dx = self.layer1.backward[batch_size](dh_pre, x)

        return dx^

    fn update(mut self, learning_rate: Scalar[Self.dtype]):
        """Update all weights using SGD."""
        self.layer1.update(learning_rate)
        self.layer2.update(learning_rate)

    fn zero_grad(mut self):
        """Reset all gradients to zero."""
        self.layer1.zero_grad()
        self.layer2.zero_grad()

    fn num_parameters(self) -> Int:
        """Return total number of learnable parameters."""
        return self.layer1.num_parameters() + self.layer2.num_parameters()

    fn print_info(self, name: String = "MLP2"):
        """Print network architecture."""
        print(name + ":")
        print("  Architecture: " + String(Self.input_dim) + " -> " + String(Self.hidden_dim) + " (tanh) -> " + String(Self.output_dim))
        self.layer1.print_info("  Layer 1")
        self.layer2.print_info("  Layer 2")
        print("  Total parameters: " + String(self.num_parameters()))


# =============================================================================
# Three-Layer MLP (two hidden layers) - common for RL
# =============================================================================


struct MLP3[
    input_dim: Int,
    hidden1_dim: Int,
    hidden2_dim: Int,
    output_dim: Int,
    dtype: DType = DType.float64,
]:
    """Three-layer MLP: input -> hidden1 (tanh) -> hidden2 (tanh) -> output.

    Architecture: input_dim -> hidden1_dim -> hidden2_dim -> output_dim.
    Common for RL actors and critics.
    """

    var layer1: Linear[Self.input_dim, Self.hidden1_dim, Self.dtype]
    var layer2: Linear[Self.hidden1_dim, Self.hidden2_dim, Self.dtype]
    var layer3: Linear[Self.hidden2_dim, Self.output_dim, Self.dtype]

    fn __init__(out self):
        """Initialize MLP with Xavier initialization."""
        self.layer1 = Linear[Self.input_dim, Self.hidden1_dim, Self.dtype]()
        self.layer2 = Linear[Self.hidden1_dim, Self.hidden2_dim, Self.dtype]()
        self.layer3 = Linear[Self.hidden2_dim, Self.output_dim, Self.dtype]()

    fn forward[batch_size: Int](
        mut self,
        x: InlineArray[Scalar[Self.dtype], batch_size * Self.input_dim],
    ) -> InlineArray[Scalar[Self.dtype], batch_size * Self.output_dim]:
        """Forward pass with tanh hidden activations."""
        # Layer 1: linear + tanh
        var h1_pre = self.layer1.forward[batch_size](x)
        var h1 = tanh_activation[batch_size * Self.hidden1_dim, Self.dtype](h1_pre)

        # Layer 2: linear + tanh
        var h2_pre = self.layer2.forward[batch_size](h1)
        var h2 = tanh_activation[batch_size * Self.hidden2_dim, Self.dtype](h2_pre)

        # Layer 3: linear (no activation)
        var out = self.layer3.forward[batch_size](h2)

        return out^

    fn forward_with_cache[batch_size: Int](
        mut self,
        x: InlineArray[Scalar[Self.dtype], batch_size * Self.input_dim],
        mut h1_out: InlineArray[Scalar[Self.dtype], batch_size * Self.hidden1_dim],
        mut h2_out: InlineArray[Scalar[Self.dtype], batch_size * Self.hidden2_dim],
    ) -> InlineArray[Scalar[Self.dtype], batch_size * Self.output_dim]:
        """Forward pass storing intermediate activations."""
        var h1_pre = self.layer1.forward[batch_size](x)
        var h1 = tanh_activation[batch_size * Self.hidden1_dim, Self.dtype](h1_pre)

        var h2_pre = self.layer2.forward[batch_size](h1)
        var h2 = tanh_activation[batch_size * Self.hidden2_dim, Self.dtype](h2_pre)

        # Store in output buffers
        for i in range(batch_size * Self.hidden1_dim):
            h1_out[i] = h1[i]
        for i in range(batch_size * Self.hidden2_dim):
            h2_out[i] = h2[i]

        var out = self.layer3.forward[batch_size](h2)

        return out^

    fn backward[batch_size: Int](
        mut self,
        dy: InlineArray[Scalar[Self.dtype], batch_size * Self.output_dim],
        x: InlineArray[Scalar[Self.dtype], batch_size * Self.input_dim],
        h1: InlineArray[Scalar[Self.dtype], batch_size * Self.hidden1_dim],
        h2: InlineArray[Scalar[Self.dtype], batch_size * Self.hidden2_dim],
    ) -> InlineArray[Scalar[Self.dtype], batch_size * Self.input_dim]:
        """Backward pass through the MLP."""
        # Backward through layer3
        var dh2 = self.layer3.backward[batch_size](dy, h2)

        # Backward through tanh2
        var tanh_g2 = tanh_grad[batch_size * Self.hidden2_dim, Self.dtype](h2)
        var dh2_pre = elementwise_mul[batch_size * Self.hidden2_dim, Self.dtype](dh2, tanh_g2)

        # Backward through layer2
        var dh1 = self.layer2.backward[batch_size](dh2_pre, h1)

        # Backward through tanh1
        var tanh_g1 = tanh_grad[batch_size * Self.hidden1_dim, Self.dtype](h1)
        var dh1_pre = elementwise_mul[batch_size * Self.hidden1_dim, Self.dtype](dh1, tanh_g1)

        # Backward through layer1
        var dx = self.layer1.backward[batch_size](dh1_pre, x)

        return dx^

    fn update(mut self, learning_rate: Scalar[Self.dtype]):
        """Update all weights using SGD."""
        self.layer1.update(learning_rate)
        self.layer2.update(learning_rate)
        self.layer3.update(learning_rate)

    fn zero_grad(mut self):
        """Reset all gradients to zero."""
        self.layer1.zero_grad()
        self.layer2.zero_grad()
        self.layer3.zero_grad()

    fn num_parameters(self) -> Int:
        """Return total number of learnable parameters."""
        return (
            self.layer1.num_parameters() +
            self.layer2.num_parameters() +
            self.layer3.num_parameters()
        )

    fn print_info(self, name: String = "MLP3"):
        """Print network architecture."""
        print(name + ":")
        print(
            "  Architecture: " + String(Self.input_dim) +
            " -> " + String(Self.hidden1_dim) + " (tanh)" +
            " -> " + String(Self.hidden2_dim) + " (tanh)" +
            " -> " + String(Self.output_dim)
        )
        self.layer1.print_info("  Layer 1")
        self.layer2.print_info("  Layer 2")
        self.layer3.print_info("  Layer 3")
        print("  Total parameters: " + String(self.num_parameters()))

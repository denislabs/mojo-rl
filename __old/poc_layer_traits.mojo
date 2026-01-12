"""POC: Trait-based Layer Design for GPU-compatible Neural Networks.

This POC demonstrates a trait-based layer API inspired by the GPUDiscreteEnv
pattern in cartpole.mojo and env_traits.mojo.

Key design elements:
1. Compile-time dimensions via `comptime` constants
2. Static inline methods for GPU kernel fusion
3. Trait interface defining the contract
4. Both CPU (instance methods) and GPU (static inline) support

Usage:
    pixi run mojo run deep_rl/poc_layer_traits.mojo           # CPU
    pixi run -e apple mojo run deep_rl/poc_layer_traits.mojo  # GPU
"""

from math import sqrt
from random import random_float64
from memory import UnsafePointer, memset_zero

# =============================================================================
# Compile-time dtype (shared across all layers)
# =============================================================================

comptime dtype = DType.float32


# =============================================================================
# Layer Trait - The Core Abstraction
# =============================================================================


trait GPULayer:
    """Trait for GPU-compatible neural network layers.

    Layers must define compile-time dimensions and provide static inline
    methods that can be called inside fused GPU kernels.

    This mirrors the GPUDiscreteEnv pattern from env_traits.mojo:
    - comptime constants for dimensions
    - @staticmethod @always_inline methods for GPU fusion
    """

    # Compile-time constants for layer dimensions
    comptime IN_DIM: Int
    comptime OUT_DIM: Int

    # Parameter size (for weight allocation)
    comptime PARAM_SIZE: Int  # IN_DIM * OUT_DIM + OUT_DIM (weights + bias)

    # Cache size (for backward pass - stores pre-activation or input)
    comptime CACHE_SIZE: Int

    @staticmethod
    @always_inline
    fn forward_inline[
        batch_size: Int
    ](
        input: UnsafePointer[Scalar[dtype], mut=False],       # [batch_size, IN_DIM]
        params: UnsafePointer[Scalar[dtype], mut=False],      # [PARAM_SIZE]
        output: UnsafePointer[Scalar[dtype]],      # [batch_size, OUT_DIM]
        cache: UnsafePointer[Scalar[dtype]],       # [CACHE_SIZE] or None
    ):
        """Forward pass - can be called inside GPU kernels."""
        ...

    @staticmethod
    @always_inline
    fn backward_inline[
        batch_size: Int
    ](
        grad_output: UnsafePointer[Scalar[dtype], mut=False], # [batch_size, OUT_DIM]
        cache: UnsafePointer[Scalar[dtype], mut=False],       # [CACHE_SIZE]
        params: UnsafePointer[Scalar[dtype], mut=False],      # [PARAM_SIZE]
        grad_input: UnsafePointer[Scalar[dtype]],  # [batch_size, IN_DIM]
        grad_params: UnsafePointer[Scalar[dtype]], # [PARAM_SIZE]
    ):
        """Backward pass - can be called inside GPU kernels."""
        ...


# =============================================================================
# Linear Layer Implementation
# =============================================================================


struct LinearLayer[in_dim: Int, out_dim: Int](GPULayer):
    """Linear layer y = x @ W + b implementing GPULayer trait.

    Parameters layout:
        params[0 : in_dim * out_dim] = weights (row-major)
        params[in_dim * out_dim : ] = bias

    Cache layout:
        cache[0 : batch_size * in_dim] = input (for weight gradients)
    """

    # Trait constants
    comptime IN_DIM: Int = Self.in_dim
    comptime OUT_DIM: Int = Self.out_dim
    comptime PARAM_SIZE: Int = Self.in_dim * Self.out_dim + Self.out_dim
    comptime CACHE_SIZE: Int = 0  # Cache size is batch_size * in_dim (dynamic)

    @staticmethod
    @always_inline
    fn forward_inline[
        batch_size: Int
    ](
        input: UnsafePointer[Scalar[dtype], mut=False],
        params: UnsafePointer[Scalar[dtype], mut=False],
        output: UnsafePointer[Scalar[dtype]],
        cache: UnsafePointer[Scalar[dtype]],
    ):
        """Forward output = input @ W + b."""
        comptime weight_size = Self.in_dim * Self.out_dim

        # For each sample in batch
        for b in range(batch_size):
            # For each output neuron
            for j in range(Self.out_dim):
                var acc: Scalar[dtype] = params[weight_size + j]  # Start with bias

                # Dot product: input[b] @ W[:, j]
                for i in range(Self.in_dim):
                    var input_val = input[b * Self.in_dim + i]
                    var weight_val = params[i * Self.out_dim + j]  # W[i, j]
                    acc += input_val * weight_val

                output[b * Self.out_dim + j] = acc

        # Cache input for backward pass (if cache provided)
        if cache:
            for i in range(batch_size * Self.in_dim):
                cache[i] = input[i]

    @staticmethod
    @always_inline
    fn backward_inline[
        batch_size: Int
    ](
        grad_output: UnsafePointer[Scalar[dtype], mut=False],
        cache: UnsafePointer[Scalar[dtype], mut=False],  # Contains cached input
        params: UnsafePointer[Scalar[dtype], mut=False],
        grad_input: UnsafePointer[Scalar[dtype]],
        grad_params: UnsafePointer[Scalar[dtype]],
    ):
        """Backward compute grad_input, grad_W, grad_b."""
        comptime weight_size = Self.in_dim * Self.out_dim

        # Zero gradients
        for i in range(Self.in_dim * Self.out_dim + Self.out_dim):
            grad_params[i] = 0

        for b in range(batch_size):
            # Gradient w.r.t. input: grad_input = grad_output @ W.T
            for i in range(Self.in_dim):
                var acc: Scalar[dtype] = 0
                for j in range(Self.out_dim):
                    var grad_out = grad_output[b * Self.out_dim + j]
                    var weight = params[i * Self.out_dim + j]  # W[i, j]
                    acc += grad_out * weight
                grad_input[b * Self.in_dim + i] = acc

            # Gradient w.r.t. weights: grad_W += input.T @ grad_output
            for i in range(Self.in_dim):
                var input_val = cache[b * Self.in_dim + i]
                for j in range(Self.out_dim):
                    var grad_out = grad_output[b * Self.out_dim + j]
                    grad_params[i * Self.out_dim + j] += input_val * grad_out

            # Gradient w.r.t. bias: grad_b += sum(grad_output, axis=0)
            for j in range(Self.out_dim):
                grad_params[weight_size + j] += grad_output[b * Self.out_dim + j]


# =============================================================================
# ReLU Activation (stateless - no parameters)
# =============================================================================


struct ReLULayer[dim: Int](GPULayer):
    """ReLU activation y = max(0, x) - stateless layer.

    Stateless layer - no learnable parameters.
    Cache stores pre-activation for backward pass.
    """

    comptime IN_DIM: Int = Self.dim
    comptime OUT_DIM: Int = Self.dim
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = 0  # Will be batch_size * dim (dynamic)

    @staticmethod
    @always_inline
    fn forward_inline[
        batch_size: Int
    ](
        input: UnsafePointer[Scalar[dtype], mut=False],
        params: UnsafePointer[Scalar[dtype], mut=False],  # Unused
        output: UnsafePointer[Scalar[dtype]],
        cache: UnsafePointer[Scalar[dtype]],
    ):
        """Forward output = max(0, input)."""
        for i in range(batch_size * Self.dim):
            var x = input[i]
            output[i] = x if x > 0 else 0

            # Cache pre-activation for backward
            if cache:
                cache[i] = x

    @staticmethod
    @always_inline
    fn backward_inline[
        batch_size: Int
    ](
        grad_output: UnsafePointer[Scalar[dtype], mut=False],
        cache: UnsafePointer[Scalar[dtype], mut=False],  # Pre-activation values
        params: UnsafePointer[Scalar[dtype], mut=False],  # Unused
        grad_input: UnsafePointer[Scalar[dtype]],
        grad_params: UnsafePointer[Scalar[dtype]],  # Unused
    ):
        """Backward grad_input = grad_output * (cache > 0)."""
        for i in range(batch_size * Self.dim):
            var x = cache[i]
            grad_input[i] = grad_output[i] if x > 0 else 0


# =============================================================================
# MLP using Trait-based Layers
# =============================================================================


struct MLP2Layers[input_dim: Int, hidden_dim: Int, output_dim: Int]:
    """Two-layer MLP using trait-based layers.

    Architecture: Linear -> ReLU -> Linear

    This demonstrates composing GPULayer implementations.
    """

    # Total parameter sizes
    comptime PARAM_SIZE_1: Int = Self.input_dim * Self.hidden_dim + Self.hidden_dim
    comptime PARAM_SIZE_2: Int = Self.hidden_dim * Self.output_dim + Self.output_dim
    comptime TOTAL_PARAMS: Int = Self.PARAM_SIZE_1 + Self.PARAM_SIZE_2

    # Parameters (contiguous buffer)
    var params: UnsafePointer[Scalar[dtype]]
    var grad_params: UnsafePointer[Scalar[dtype]]

    # Adam state
    var m: UnsafePointer[Scalar[dtype]]  # First moment
    var v: UnsafePointer[Scalar[dtype]]  # Second moment
    var t: Int  # Time step

    fn __init__(out self):
        """Initialize MLP with Xavier initialization."""
        self.params = UnsafePointer[Scalar[dtype]].alloc(Self.TOTAL_PARAMS)
        self.grad_params = UnsafePointer[Scalar[dtype]].alloc(Self.TOTAL_PARAMS)
        self.m = UnsafePointer[Scalar[dtype]].alloc(Self.TOTAL_PARAMS)
        self.v = UnsafePointer[Scalar[dtype]].alloc(Self.TOTAL_PARAMS)
        self.t = 0

        # Zero gradients and Adam state
        memset_zero(self.grad_params, Self.TOTAL_PARAMS)
        memset_zero(self.m, Self.TOTAL_PARAMS)
        memset_zero(self.v, Self.TOTAL_PARAMS)

        # Xavier init for layer 1 weights
        var std1 = sqrt(2.0 / Float64(Self.input_dim + Self.hidden_dim))
        for i in range(Self.input_dim * Self.hidden_dim):
            self.params[i] = Scalar[dtype]((random_float64() * 2 - 1) * std1)
        # Zero bias 1
        for i in range(Self.hidden_dim):
            self.params[Self.input_dim * Self.hidden_dim + i] = 0

        # Xavier init for layer 2 weights
        var std2 = sqrt(2.0 / Float64(Self.hidden_dim + Self.output_dim))
        var offset = Self.PARAM_SIZE_1
        for i in range(Self.hidden_dim * Self.output_dim):
            self.params[offset + i] = Scalar[dtype]((random_float64() * 2 - 1) * std2)
        # Zero bias 2
        for i in range(Self.output_dim):
            self.params[offset + Self.hidden_dim * Self.output_dim + i] = 0

    fn __del__(deinit self):
        self.params.free()
        self.grad_params.free()
        self.m.free()
        self.v.free()

    fn forward[
        batch_size: Int
    ](
        self,
        input: UnsafePointer[Scalar[dtype], mut=False],
        output: UnsafePointer[Scalar[dtype]],
        # Caches for backward
        cache_linear1: UnsafePointer[Scalar[dtype]],
        cache_relu: UnsafePointer[Scalar[dtype]],
        cache_linear2: UnsafePointer[Scalar[dtype]],
        # Intermediate buffers
        hidden: UnsafePointer[Scalar[dtype]],
        hidden_relu: UnsafePointer[Scalar[dtype]],
    ):
        """Forward pass through the MLP."""
        # Layer 1: Linear
        LinearLayer[Self.input_dim, Self.hidden_dim].forward_inline[batch_size](
            input, self.params, hidden, cache_linear1
        )

        # Activation: ReLU
        ReLULayer[Self.hidden_dim].forward_inline[batch_size](
            hidden, UnsafePointer[Scalar[dtype], mut=False](), hidden_relu, cache_relu
        )

        # Layer 2: Linear
        var params2 = (self.params + Self.PARAM_SIZE_1).bitcast[Scalar[dtype], mut=False]()
        LinearLayer[Self.hidden_dim, Self.output_dim].forward_inline[batch_size](
            hidden_relu, params2, output, cache_linear2
        )

    fn backward[
        batch_size: Int
    ](
        mut self,
        grad_output: UnsafePointer[Scalar[dtype], mut=False],
        # Caches from forward
        cache_linear1: UnsafePointer[Scalar[dtype], mut=False],
        cache_relu: UnsafePointer[Scalar[dtype], mut=False],
        cache_linear2: UnsafePointer[Scalar[dtype], mut=False],
        # Intermediate gradient buffers
        grad_hidden_relu: UnsafePointer[Scalar[dtype]],
        grad_hidden: UnsafePointer[Scalar[dtype]],
        grad_input: UnsafePointer[Scalar[dtype]],
    ):
        """Backward pass through the MLP."""
        # Layer 2 backward
        var params2 = (self.params + Self.PARAM_SIZE_1).bitcast[Scalar[dtype], mut=False]()
        LinearLayer[Self.hidden_dim, Self.output_dim].backward_inline[batch_size](
            grad_output,
            cache_linear2,
            params2,
            grad_hidden_relu,
            self.grad_params + Self.PARAM_SIZE_1,
        )

        # ReLU backward
        ReLULayer[Self.hidden_dim].backward_inline[batch_size](
            grad_hidden_relu,
            cache_relu,
            UnsafePointer[Scalar[dtype], mut=False](),
            grad_hidden,
            UnsafePointer[Scalar[dtype]](),
        )

        # Layer 1 backward
        LinearLayer[Self.input_dim, Self.hidden_dim].backward_inline[batch_size](
            grad_hidden,
            cache_linear1,
            self.params,
            grad_input,
            self.grad_params,
        )

    fn adam_step(mut self, lr: Float64 = 0.001, beta1: Float64 = 0.9, beta2: Float64 = 0.999, eps: Float64 = 1e-8):
        """Update parameters using Adam optimizer."""
        self.t += 1
        var t = Float64(self.t)

        for i in range(Self.TOTAL_PARAMS):
            var g = self.grad_params[i]

            # Update biased first moment
            self.m[i] = Scalar[dtype](beta1) * self.m[i] + Scalar[dtype](1 - beta1) * g

            # Update biased second moment
            self.v[i] = Scalar[dtype](beta2) * self.v[i] + Scalar[dtype](1 - beta2) * g * g

            # Bias correction
            var m_hat = Float64(self.m[i]) / (1 - beta1 ** t)
            var v_hat = Float64(self.v[i]) / (1 - beta2 ** t)

            # Update parameters
            self.params[i] -= Scalar[dtype](lr * m_hat / (sqrt(v_hat) + eps))

    fn zero_grad(mut self):
        """Zero all gradients."""
        memset_zero(self.grad_params, Self.TOTAL_PARAMS)


# =============================================================================
# MSE Loss
# =============================================================================


@always_inline
fn mse_loss[
    batch_size: Int, output_dim: Int
](
    predictions: UnsafePointer[Scalar[dtype], mut=False],
    targets: UnsafePointer[Scalar[dtype], mut=False],
    grad_output: UnsafePointer[Scalar[dtype]],
) -> Scalar[dtype]:
    """Compute MSE loss and gradient."""
    var loss: Scalar[dtype] = 0
    var scale = Scalar[dtype](2.0 / Float64(batch_size * output_dim))

    for i in range(batch_size * output_dim):
        var diff = predictions[i] - targets[i]
        loss += diff * diff
        grad_output[i] = scale * diff

    return loss / Scalar[dtype](batch_size * output_dim)


# =============================================================================
# CPU Training Demo
# =============================================================================


fn train_cpu():
    """CPU training demonstration."""
    print("=" * 70)
    print("POC: Trait-based Layer Design")
    print("=" * 70)
    print()

    # Hyperparameters
    comptime BATCH_SIZE = 64
    comptime INPUT_DIM = 2
    comptime HIDDEN_DIM = 32
    comptime OUTPUT_DIM = 1
    comptime NUM_EPOCHS = 500

    print("Network: " + String(INPUT_DIM) + " -> " + String(HIDDEN_DIM) + " (ReLU) -> " + String(OUTPUT_DIM))
    print("Batch: " + String(BATCH_SIZE) + ", Epochs: " + String(NUM_EPOCHS))
    print()

    # Create MLP
    var mlp = MLP2Layers[INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM]()

    # Allocate buffers
    var input = UnsafePointer[Scalar[dtype]].alloc(BATCH_SIZE * INPUT_DIM)
    var target = UnsafePointer[Scalar[dtype]].alloc(BATCH_SIZE * OUTPUT_DIM)
    var output = UnsafePointer[Scalar[dtype]].alloc(BATCH_SIZE * OUTPUT_DIM)
    var grad_output = UnsafePointer[Scalar[dtype]].alloc(BATCH_SIZE * OUTPUT_DIM)

    # Intermediate buffers
    var hidden = UnsafePointer[Scalar[dtype]].alloc(BATCH_SIZE * HIDDEN_DIM)
    var hidden_relu = UnsafePointer[Scalar[dtype]].alloc(BATCH_SIZE * HIDDEN_DIM)

    # Gradient buffers
    var grad_hidden_relu = UnsafePointer[Scalar[dtype]].alloc(BATCH_SIZE * HIDDEN_DIM)
    var grad_hidden = UnsafePointer[Scalar[dtype]].alloc(BATCH_SIZE * HIDDEN_DIM)
    var grad_input = UnsafePointer[Scalar[dtype]].alloc(BATCH_SIZE * INPUT_DIM)

    # Cache buffers
    var cache_linear1 = UnsafePointer[Scalar[dtype]].alloc(BATCH_SIZE * INPUT_DIM)
    var cache_relu = UnsafePointer[Scalar[dtype]].alloc(BATCH_SIZE * HIDDEN_DIM)
    var cache_linear2 = UnsafePointer[Scalar[dtype]].alloc(BATCH_SIZE * HIDDEN_DIM)

    print("Training (learning XOR-like function)...")
    print("-" * 70)

    for epoch in range(NUM_EPOCHS):
        # Generate batch: target = sin(x1 * x2) - simple nonlinear function
        for b in range(BATCH_SIZE):
            var x1 = Scalar[dtype]((random_float64() - 0.5) * 4)  # [-2, 2]
            var x2 = Scalar[dtype]((random_float64() - 0.5) * 4)
            input[b * INPUT_DIM + 0] = x1
            input[b * INPUT_DIM + 1] = x2
            # Target: XOR-like pattern
            var sign1: Scalar[dtype] = 1.0 if x1 > 0 else -1.0
            var sign2: Scalar[dtype] = 1.0 if x2 > 0 else -1.0
            target[b] = sign1 * sign2  # +1 if same sign, -1 if different

        # Forward pass
        mlp.forward[BATCH_SIZE](
            input, output,
            cache_linear1, cache_relu, cache_linear2,
            hidden, hidden_relu
        )

        # Compute loss and gradient
        var loss = mse_loss[BATCH_SIZE, OUTPUT_DIM](output, target, grad_output)

        # Backward pass
        mlp.zero_grad()
        mlp.backward[BATCH_SIZE](
            grad_output,
            cache_linear1, cache_relu, cache_linear2,
            grad_hidden_relu, grad_hidden, grad_input
        )

        # Update parameters
        mlp.adam_step(lr=0.01)

        # Print progress
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print("Epoch " + String(epoch + 1) + "/" + String(NUM_EPOCHS) + " - Loss: " + String(Float64(loss)))

    print("-" * 70)
    print()

    # Test predictions
    print("Test predictions (XOR-like pattern):")
    var test_input = UnsafePointer[Scalar[dtype]].alloc(INPUT_DIM)
    var test_output = UnsafePointer[Scalar[dtype]].alloc(OUTPUT_DIM)
    var test_cache1 = UnsafePointer[Scalar[dtype]].alloc(INPUT_DIM)
    var test_cache_relu = UnsafePointer[Scalar[dtype]].alloc(HIDDEN_DIM)
    var test_cache2 = UnsafePointer[Scalar[dtype]].alloc(HIDDEN_DIM)
    var test_hidden = UnsafePointer[Scalar[dtype]].alloc(HIDDEN_DIM)
    var test_hidden_relu = UnsafePointer[Scalar[dtype]].alloc(HIDDEN_DIM)

    # Test case 1: (1, 1) -> +1
    test_input[0] = Scalar[dtype](1.0)
    test_input[1] = Scalar[dtype](1.0)
    mlp.forward[1](test_input, test_output, test_cache1, test_cache_relu, test_cache2, test_hidden, test_hidden_relu)
    print("  (1.0, 1.0) -> " + String(Float64(test_output[0])) + " (expected: 1.0)")

    # Test case 2: (-1, -1) -> +1
    test_input[0] = Scalar[dtype](-1.0)
    test_input[1] = Scalar[dtype](-1.0)
    mlp.forward[1](test_input, test_output, test_cache1, test_cache_relu, test_cache2, test_hidden, test_hidden_relu)
    print("  (-1.0, -1.0) -> " + String(Float64(test_output[0])) + " (expected: 1.0)")

    # Test case 3: (1, -1) -> -1
    test_input[0] = Scalar[dtype](1.0)
    test_input[1] = Scalar[dtype](-1.0)
    mlp.forward[1](test_input, test_output, test_cache1, test_cache_relu, test_cache2, test_hidden, test_hidden_relu)
    print("  (1.0, -1.0) -> " + String(Float64(test_output[0])) + " (expected: -1.0)")

    # Test case 4: (-1, 1) -> -1
    test_input[0] = Scalar[dtype](-1.0)
    test_input[1] = Scalar[dtype](1.0)
    mlp.forward[1](test_input, test_output, test_cache1, test_cache_relu, test_cache2, test_hidden, test_hidden_relu)
    print("  (-1.0, 1.0) -> " + String(Float64(test_output[0])) + " (expected: -1.0)")

    # Cleanup
    test_input.free()
    test_output.free()
    test_cache1.free()
    test_cache_relu.free()
    test_cache2.free()
    test_hidden.free()
    test_hidden_relu.free()
    input.free()
    target.free()
    output.free()
    grad_output.free()
    hidden.free()
    hidden_relu.free()
    grad_hidden_relu.free()
    grad_hidden.free()
    grad_input.free()
    cache_linear1.free()
    cache_relu.free()
    cache_linear2.free()

    print()
    print("=" * 70)
    print("TRAIT-BASED DESIGN SUMMARY")
    print("=" * 70)
    print()
    print("GPULayer trait provides:")
    print("  - comptime IN_DIM, OUT_DIM, PARAM_SIZE, CACHE_SIZE")
    print("  - @staticmethod @always_inline fn forward_inline[batch_size](...)")
    print("  - @staticmethod @always_inline fn backward_inline[batch_size](...)")
    print()
    print("Benefits:")
    print("  1. Compile-time dimension checking")
    print("  2. Static inline methods fuse into GPU kernels")
    print("  3. Composable layers (MLP2Layers uses LinearLayer, ReLULayer)")
    print("  4. Same interface for CPU and GPU")
    print()
    print("Pattern mirrors GPUDiscreteEnv from cartpole.mojo:")
    print("  - comptime constants for dimensions")
    print("  - @staticmethod @always_inline for GPU fusion")
    print("  - Trait defines the contract")
    print()
    print("=" * 70)


fn main():
    train_cpu()

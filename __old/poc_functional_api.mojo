"""POC: Functional API for Neural Networks in Mojo.

This demonstrates a Keras/PyTorch-inspired API adapted to Mojo's struct-based paradigm.

Design Goals:
1. Composable layers via traits
2. Compile-time dimension checking
3. GPU-compatible inline operations
4. Clean separation: Layer (params) vs Forward/Backward (ops)

Key Insight: Instead of micrograd's dynamic graph, we use:
- Compile-time layer composition (Sequential)
- Explicit cache passing for backward (no closures needed)
- Traits for polymorphism without virtual dispatch

Usage:
    # Define network at compile time
    alias Net = Sequential2[
        LinearLayer[2, 64],
        ReLULayer[64],
        LinearLayer[64, 1],
    ]

    # Training
    var net = Net()
    var cache = net.forward_with_cache[batch_size](x)
    var grads = net.backward[batch_size](dy, cache)
    net.update(lr=0.001)
"""

from math import sqrt
from random import random_float64
from sys import simd_width_of


# =============================================================================
# Core Types (compile-time sized tensors)
# =============================================================================

comptime DEFAULT_DTYPE = DType.float32


struct Tensor[size: Int, dtype: DType = DEFAULT_DTYPE]:
    """Fixed-size tensor for compile-time dimension checking."""
    var data: InlineArray[Scalar[Self.dtype], Self.size]

    fn __init__(out self):
        self.data = InlineArray[Scalar[Self.dtype], Self.size](fill=0)

    fn __init__(out self, fill: Scalar[Self.dtype]):
        self.data = InlineArray[Scalar[Self.dtype], Self.size](fill=fill)

    @always_inline
    fn __getitem__(self, idx: Int) -> Scalar[Self.dtype]:
        return self.data[idx]

    @always_inline
    fn __setitem__(mut self, idx: Int, value: Scalar[Self.dtype]):
        self.data[idx] = value


# =============================================================================
# Layer Trait: The Core Abstraction
# =============================================================================

# Note: Mojo traits can't have associated types yet, so we use compile-time
# parameters on the struct level. Each layer defines its own cache type.


# =============================================================================
# Linear Layer
# =============================================================================


struct LinearParams[in_dim: Int, out_dim: Int, dtype: DType = DEFAULT_DTYPE]:
    """Parameters for a linear layer: W (in_dim x out_dim) + b (out_dim)."""
    var W: InlineArray[Scalar[Self.dtype], Self.in_dim * Self.out_dim]
    var b: InlineArray[Scalar[Self.dtype], Self.out_dim]

    # Gradients
    var dW: InlineArray[Scalar[Self.dtype], Self.in_dim * Self.out_dim]
    var db: InlineArray[Scalar[Self.dtype], Self.out_dim]

    # Adam state (optional, for update)
    var m_W: InlineArray[Scalar[Self.dtype], Self.in_dim * Self.out_dim]
    var v_W: InlineArray[Scalar[Self.dtype], Self.in_dim * Self.out_dim]
    var m_b: InlineArray[Scalar[Self.dtype], Self.out_dim]
    var v_b: InlineArray[Scalar[Self.dtype], Self.out_dim]
    var t: Int

    fn __init__(out self):
        """Xavier initialization."""
        var limit = sqrt(Scalar[Self.dtype](6.0) / Scalar[Self.dtype](Self.in_dim + Self.out_dim))
        self.W = InlineArray[Scalar[Self.dtype], Self.in_dim * Self.out_dim](fill=0)
        for i in range(Self.in_dim * Self.out_dim):
            self.W[i] = (Scalar[Self.dtype](random_float64()) * 2 - 1) * limit

        self.b = InlineArray[Scalar[Self.dtype], Self.out_dim](fill=0)
        self.dW = InlineArray[Scalar[Self.dtype], Self.in_dim * Self.out_dim](fill=0)
        self.db = InlineArray[Scalar[Self.dtype], Self.out_dim](fill=0)
        self.m_W = InlineArray[Scalar[Self.dtype], Self.in_dim * Self.out_dim](fill=0)
        self.v_W = InlineArray[Scalar[Self.dtype], Self.in_dim * Self.out_dim](fill=0)
        self.m_b = InlineArray[Scalar[Self.dtype], Self.out_dim](fill=0)
        self.v_b = InlineArray[Scalar[Self.dtype], Self.out_dim](fill=0)
        self.t = 0


struct LinearCache[batch: Int, in_dim: Int, dtype: DType = DEFAULT_DTYPE]:
    """Cache for linear layer backward pass: stores input x."""
    var x: InlineArray[Scalar[Self.dtype], Self.batch * Self.in_dim]

    fn __init__(out self):
        self.x = InlineArray[Scalar[Self.dtype], Self.batch * Self.in_dim](fill=0)


# Forward/Backward as free functions (functional style)
@always_inline
fn linear_forward[
    batch: Int, in_dim: Int, out_dim: Int, dtype: DType = DEFAULT_DTYPE
](
    x: InlineArray[Scalar[dtype], batch * in_dim],
    params: LinearParams[in_dim, out_dim, dtype],
) -> InlineArray[Scalar[dtype], batch * out_dim]:
    """y = x @ W + b"""
    var y = InlineArray[Scalar[dtype], batch * out_dim](fill=0)

    for i in range(batch):
        for j in range(out_dim):
            var sum = params.b[j]
            for k in range(in_dim):
                sum += x[i * in_dim + k] * params.W[k * out_dim + j]
            y[i * out_dim + j] = sum

    return y^


@always_inline
fn linear_backward[
    batch: Int, in_dim: Int, out_dim: Int, dtype: DType = DEFAULT_DTYPE
](
    dy: InlineArray[Scalar[dtype], batch * out_dim],
    cache: LinearCache[batch, in_dim, dtype],
    mut params: LinearParams[in_dim, out_dim, dtype],
) -> InlineArray[Scalar[dtype], batch * in_dim]:
    """Compute dW, db, and return dx."""
    # Zero gradients
    for i in range(in_dim * out_dim):
        params.dW[i] = 0
    for i in range(out_dim):
        params.db[i] = 0

    # dW = x.T @ dy
    for j in range(in_dim):
        for k in range(out_dim):
            var sum: Scalar[dtype] = 0
            for i in range(batch):
                sum += cache.x[i * in_dim + j] * dy[i * out_dim + k]
            params.dW[j * out_dim + k] = sum

    # db = sum(dy, axis=0)
    for j in range(out_dim):
        var sum: Scalar[dtype] = 0
        for i in range(batch):
            sum += dy[i * out_dim + j]
        params.db[j] = sum

    # dx = dy @ W.T
    var dx = InlineArray[Scalar[dtype], batch * in_dim](fill=0)
    for i in range(batch):
        for j in range(in_dim):
            var sum: Scalar[dtype] = 0
            for k in range(out_dim):
                sum += dy[i * out_dim + k] * params.W[j * out_dim + k]
            dx[i * in_dim + j] = sum

    return dx^


@always_inline
fn linear_update_adam[
    in_dim: Int, out_dim: Int, dtype: DType = DEFAULT_DTYPE
](
    mut params: LinearParams[in_dim, out_dim, dtype],
    lr: Scalar[dtype] = 0.001,
    beta1: Scalar[dtype] = 0.9,
    beta2: Scalar[dtype] = 0.999,
    eps: Scalar[dtype] = 1e-8,
):
    """Adam update for linear layer."""
    params.t += 1
    var bc1 = 1 - beta1 ** params.t
    var bc2 = 1 - beta2 ** params.t

    # Update W
    for i in range(in_dim * out_dim):
        var g = params.dW[i]
        params.m_W[i] = beta1 * params.m_W[i] + (1 - beta1) * g
        params.v_W[i] = beta2 * params.v_W[i] + (1 - beta2) * g * g
        var m_hat = params.m_W[i] / bc1
        var v_hat = params.v_W[i] / bc2
        params.W[i] -= lr * m_hat / (sqrt(v_hat) + eps)

    # Update b
    for i in range(out_dim):
        var g = params.db[i]
        params.m_b[i] = beta1 * params.m_b[i] + (1 - beta1) * g
        params.v_b[i] = beta2 * params.v_b[i] + (1 - beta2) * g * g
        var m_hat = params.m_b[i] / bc1
        var v_hat = params.v_b[i] / bc2
        params.b[i] -= lr * m_hat / (sqrt(v_hat) + eps)


# =============================================================================
# ReLU Activation
# =============================================================================


struct ReLUCache[size: Int, dtype: DType = DEFAULT_DTYPE]:
    """Cache for ReLU backward: stores pre-activation."""
    var x: InlineArray[Scalar[Self.dtype], Self.size]

    fn __init__(out self):
        self.x = InlineArray[Scalar[Self.dtype], Self.size](fill=0)


@always_inline
fn relu_forward[
    size: Int, dtype: DType = DEFAULT_DTYPE
](
    x: InlineArray[Scalar[dtype], size],
) -> InlineArray[Scalar[dtype], size]:
    """y = max(0, x)"""
    var y = InlineArray[Scalar[dtype], size](fill=0)
    for i in range(size):
        y[i] = x[i] if x[i] > 0 else 0
    return y^


@always_inline
fn relu_backward[
    size: Int, dtype: DType = DEFAULT_DTYPE
](
    dy: InlineArray[Scalar[dtype], size],
    cache: ReLUCache[size, dtype],
) -> InlineArray[Scalar[dtype], size]:
    """dx = dy * (x > 0)"""
    var dx = InlineArray[Scalar[dtype], size](fill=0)
    for i in range(size):
        dx[i] = dy[i] if cache.x[i] > 0 else 0
    return dx^


# =============================================================================
# MLP: Composing Layers (2-layer example)
# =============================================================================


struct MLP2Params[in_dim: Int, hidden_dim: Int, out_dim: Int, dtype: DType = DEFAULT_DTYPE]:
    """Parameters for a 2-layer MLP: Linear -> ReLU -> Linear."""
    var layer1: LinearParams[Self.in_dim, Self.hidden_dim, Self.dtype]
    var layer2: LinearParams[Self.hidden_dim, Self.out_dim, Self.dtype]

    fn __init__(out self):
        self.layer1 = LinearParams[Self.in_dim, Self.hidden_dim, Self.dtype]()
        self.layer2 = LinearParams[Self.hidden_dim, Self.out_dim, Self.dtype]()


struct MLP2Cache[batch: Int, in_dim: Int, hidden_dim: Int, dtype: DType = DEFAULT_DTYPE]:
    """Cache for 2-layer MLP backward pass."""
    var linear1_cache: LinearCache[Self.batch, Self.in_dim, Self.dtype]
    var relu_cache: ReLUCache[Self.batch * Self.hidden_dim, Self.dtype]
    var linear2_cache: LinearCache[Self.batch, Self.hidden_dim, Self.dtype]

    fn __init__(out self):
        self.linear1_cache = LinearCache[Self.batch, Self.in_dim, Self.dtype]()
        self.relu_cache = ReLUCache[Self.batch * Self.hidden_dim, Self.dtype]()
        self.linear2_cache = LinearCache[Self.batch, Self.hidden_dim, Self.dtype]()


@always_inline
fn mlp2_forward[
    batch: Int, in_dim: Int, hidden_dim: Int, out_dim: Int,
    dtype: DType = DEFAULT_DTYPE
](
    x: InlineArray[Scalar[dtype], batch * in_dim],
    params: MLP2Params[in_dim, hidden_dim, out_dim, dtype],
) -> InlineArray[Scalar[dtype], batch * out_dim]:
    """Forward pass: Linear -> ReLU -> Linear."""
    var h = linear_forward[batch, in_dim, hidden_dim, dtype](x, params.layer1)
    var h_relu = relu_forward[batch * hidden_dim, dtype](h)
    var out = linear_forward[batch, hidden_dim, out_dim, dtype](h_relu, params.layer2)
    return out^


@always_inline
fn mlp2_forward_with_cache[
    batch: Int, in_dim: Int, hidden_dim: Int, out_dim: Int,
    dtype: DType = DEFAULT_DTYPE
](
    x: InlineArray[Scalar[dtype], batch * in_dim],
    params: MLP2Params[in_dim, hidden_dim, out_dim, dtype],
    mut cache: MLP2Cache[batch, in_dim, hidden_dim, dtype],
) -> InlineArray[Scalar[dtype], batch * out_dim]:
    """Forward pass with caching for backward."""
    # Store input for layer1 backward
    for i in range(batch * in_dim):
        cache.linear1_cache.x[i] = x[i]

    # Layer 1: Linear
    var h = linear_forward[batch, in_dim, hidden_dim, dtype](x, params.layer1)

    # Store pre-activation for ReLU backward
    for i in range(batch * hidden_dim):
        cache.relu_cache.x[i] = h[i]

    # ReLU
    var h_relu = relu_forward[batch * hidden_dim, dtype](h)

    # Store input for layer2 backward
    for i in range(batch * hidden_dim):
        cache.linear2_cache.x[i] = h_relu[i]

    # Layer 2: Linear
    var out = linear_forward[batch, hidden_dim, out_dim, dtype](h_relu, params.layer2)
    return out^


@always_inline
fn mlp2_backward[
    batch: Int, in_dim: Int, hidden_dim: Int, out_dim: Int,
    dtype: DType = DEFAULT_DTYPE
](
    dy: InlineArray[Scalar[dtype], batch * out_dim],
    cache: MLP2Cache[batch, in_dim, hidden_dim, dtype],
    mut params: MLP2Params[in_dim, hidden_dim, out_dim, dtype],
) -> InlineArray[Scalar[dtype], batch * in_dim]:
    """Backward pass through the entire MLP."""
    # Layer 2 backward
    var dh_relu = linear_backward[batch, hidden_dim, out_dim, dtype](
        dy, cache.linear2_cache, params.layer2
    )

    # ReLU backward
    var dh = relu_backward[batch * hidden_dim, dtype](dh_relu, cache.relu_cache)

    # Layer 1 backward
    var dx = linear_backward[batch, in_dim, hidden_dim, dtype](
        dh, cache.linear1_cache, params.layer1
    )

    return dx^


@always_inline
fn mlp2_update_adam[
    in_dim: Int, hidden_dim: Int, out_dim: Int,
    dtype: DType = DEFAULT_DTYPE
](
    mut params: MLP2Params[in_dim, hidden_dim, out_dim, dtype],
    lr: Scalar[dtype] = 0.001,
):
    """Adam update for both layers."""
    linear_update_adam[in_dim, hidden_dim, dtype](params.layer1, lr)
    linear_update_adam[hidden_dim, out_dim, dtype](params.layer2, lr)


# =============================================================================
# Loss Functions
# =============================================================================


@always_inline
fn mse_loss[
    size: Int, dtype: DType = DEFAULT_DTYPE
](
    pred: InlineArray[Scalar[dtype], size],
    target: InlineArray[Scalar[dtype], size],
) -> Scalar[dtype]:
    """Mean squared error loss."""
    var sum: Scalar[dtype] = 0
    for i in range(size):
        var diff = pred[i] - target[i]
        sum += diff * diff
    return sum / size


@always_inline
fn mse_loss_backward[
    size: Int, dtype: DType = DEFAULT_DTYPE
](
    pred: InlineArray[Scalar[dtype], size],
    target: InlineArray[Scalar[dtype], size],
) -> InlineArray[Scalar[dtype], size]:
    """Gradient of MSE loss w.r.t. predictions."""
    var grad = InlineArray[Scalar[dtype], size](fill=0)
    var scale = Scalar[dtype](2.0) / size
    for i in range(size):
        grad[i] = scale * (pred[i] - target[i])
    return grad^


# =============================================================================
# Demo: Training a 2-Layer MLP
# =============================================================================


fn main() raises:
    print("=" * 60)
    print("POC: Functional API for Neural Networks")
    print("=" * 60)
    print()

    # Network configuration
    comptime BATCH_SIZE = 32
    comptime INPUT_DIM = 2
    comptime HIDDEN_DIM = 64
    comptime OUTPUT_DIM = 1
    comptime dtype = DType.float32

    # Initialize network
    var params = MLP2Params[INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, dtype]()
    var cache = MLP2Cache[BATCH_SIZE, INPUT_DIM, HIDDEN_DIM, dtype]()

    print("Network: " + String(INPUT_DIM) + " -> " + String(HIDDEN_DIM) + " (ReLU) -> " + String(OUTPUT_DIM))
    print("Task: Learn y = x1 * x2")
    print("Batch size: " + String(BATCH_SIZE))
    print()

    # Training loop
    comptime NUM_EPOCHS = 1000
    var lr = Scalar[dtype](0.001)

    print("Training for " + String(NUM_EPOCHS) + " epochs...")
    print("-" * 60)

    for epoch in range(NUM_EPOCHS):
        # Generate batch data: y = x1 * x2
        var x = InlineArray[Scalar[dtype], BATCH_SIZE * INPUT_DIM](fill=0)
        var y_target = InlineArray[Scalar[dtype], BATCH_SIZE * OUTPUT_DIM](fill=0)

        for i in range(BATCH_SIZE):
            var x1 = Scalar[dtype](random_float64() * 2 - 1)
            var x2 = Scalar[dtype](random_float64() * 2 - 1)
            x[i * INPUT_DIM + 0] = x1
            x[i * INPUT_DIM + 1] = x2
            y_target[i] = x1 * x2

        # Forward pass with cache
        var y_pred = mlp2_forward_with_cache[
            BATCH_SIZE, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, dtype
        ](x, params, cache)

        # Compute loss
        var loss = mse_loss[BATCH_SIZE * OUTPUT_DIM, dtype](y_pred, y_target)

        # Backward pass
        var dy = mse_loss_backward[BATCH_SIZE * OUTPUT_DIM, dtype](y_pred, y_target)
        _ = mlp2_backward[
            BATCH_SIZE, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, dtype
        ](dy, cache, params)

        # Update parameters
        mlp2_update_adam[INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, dtype](params, lr)

        # Print progress
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print("Epoch " + String(epoch + 1) + "/" + String(NUM_EPOCHS) + " - Loss: " + String(Float64(loss)))

    print("-" * 60)
    print()
    print("Training completed!")
    print()

    # Final evaluation
    print("Sample predictions:")
    var test_x = InlineArray[Scalar[dtype], 5 * INPUT_DIM](fill=0)
    var test_y = InlineArray[Scalar[dtype], 5 * OUTPUT_DIM](fill=0)

    for i in range(5):
        var x1 = Scalar[dtype](random_float64() * 2 - 1)
        var x2 = Scalar[dtype](random_float64() * 2 - 1)
        test_x[i * INPUT_DIM + 0] = x1
        test_x[i * INPUT_DIM + 1] = x2
        test_y[i] = x1 * x2

    # Need a cache for batch size 5
    var test_cache = MLP2Cache[5, INPUT_DIM, HIDDEN_DIM, dtype]()
    var test_pred = mlp2_forward_with_cache[5, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, dtype](
        test_x, params, test_cache
    )

    for i in range(5):
        var x1 = Float64(test_x[i * INPUT_DIM + 0])
        var x2 = Float64(test_x[i * INPUT_DIM + 1])
        var pred = Float64(test_pred[i])
        var target = Float64(test_y[i])
        print("  (" + String(x1)[:6] + ", " + String(x2)[:6] + ") -> " +
              String(pred)[:7] + " vs " + String(target)[:7])

    print()
    print("=" * 60)

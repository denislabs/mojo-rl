"""POC: Functional Layer API for GPU-Compatible Neural Networks.

This demonstrates a clean functional design with:
1. Pure functions for forward/backward (no methods)
2. Separate Params structs for learnable weights
3. Explicit cache passing (no closures/heap)
4. GPU-ready inline operations

Key insight: Free functions are simpler than static methods in Mojo
because they don't require Self. prefix everywhere.

Usage:
    # Define params
    var layer1 = LinearParams[2, 64]()
    var layer2 = LinearParams[64, 1]()

    # Forward
    var h = linear_forward[BATCH, 2, 64](x, layer1)
    var h_relu = relu_forward[BATCH * 64](h)
    var y = linear_forward[BATCH, 64, 1](h_relu, layer2)

    # Backward
    var dy = mse_backward(y, target)
    var dh_relu = linear_backward[BATCH, 64, 1](dy, h_relu, layer2)
    var dh = relu_backward[BATCH * 64](dh_relu, h)
    _ = linear_backward[BATCH, 2, 64](dh, x, layer1)

    # Update
    layer1.update_adam(lr=0.001)
    layer2.update_adam(lr=0.001)
"""

from math import sqrt
from random import random_float64, seed
from time import perf_counter_ns


comptime DEFAULT_DTYPE = DType.float32


# =============================================================================
# Linear Layer (Params + Functions)
# =============================================================================


struct LinearParams[in_dim: Int, out_dim: Int, dtype: DType = DEFAULT_DTYPE]:
    """Learnable parameters for Linear layer."""
    var W: InlineArray[Scalar[Self.dtype], Self.in_dim * Self.out_dim]
    var b: InlineArray[Scalar[Self.dtype], Self.out_dim]
    var dW: InlineArray[Scalar[Self.dtype], Self.in_dim * Self.out_dim]
    var db: InlineArray[Scalar[Self.dtype], Self.out_dim]
    # Adam state
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

    fn zero_grad(mut self):
        """Reset gradients to zero."""
        for i in range(Self.in_dim * Self.out_dim):
            self.dW[i] = 0
        for i in range(Self.out_dim):
            self.db[i] = 0

    fn update_adam(mut self, lr: Scalar[Self.dtype] = 0.001,
                   beta1: Scalar[Self.dtype] = 0.9,
                   beta2: Scalar[Self.dtype] = 0.999,
                   eps: Scalar[Self.dtype] = 1e-8):
        """Adam optimizer step."""
        self.t += 1
        var bc1 = 1 - beta1 ** self.t
        var bc2 = 1 - beta2 ** self.t

        for i in range(Self.in_dim * Self.out_dim):
            var g = self.dW[i]
            self.m_W[i] = beta1 * self.m_W[i] + (1 - beta1) * g
            self.v_W[i] = beta2 * self.v_W[i] + (1 - beta2) * g * g
            self.W[i] -= lr * (self.m_W[i] / bc1) / (sqrt(self.v_W[i] / bc2) + eps)

        for i in range(Self.out_dim):
            var g = self.db[i]
            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * g
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * g * g
            self.b[i] -= lr * (self.m_b[i] / bc1) / (sqrt(self.v_b[i] / bc2) + eps)


# Free functions for Linear forward/backward
@always_inline
fn linear_forward[
    batch: Int, in_dim: Int, out_dim: Int, dtype: DType = DEFAULT_DTYPE
](
    x: InlineArray[Scalar[dtype], batch * in_dim],
    params: LinearParams[in_dim, out_dim, dtype],
) -> InlineArray[Scalar[dtype], batch * out_dim]:
    """Linear forward: y = x @ W + b."""
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
    x: InlineArray[Scalar[dtype], batch * in_dim],
    mut params: LinearParams[in_dim, out_dim, dtype],
) -> InlineArray[Scalar[dtype], batch * in_dim]:
    """Linear backward: compute gradients and return dx."""
    params.zero_grad()

    # dW = x.T @ dy
    for j in range(in_dim):
        for k in range(out_dim):
            var sum: Scalar[dtype] = 0
            for i in range(batch):
                sum += x[i * in_dim + j] * dy[i * out_dim + k]
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


# =============================================================================
# ReLU Activation (Pure Functions)
# =============================================================================


@always_inline
fn relu_forward[size: Int, dtype: DType = DEFAULT_DTYPE](
    x: InlineArray[Scalar[dtype], size],
) -> InlineArray[Scalar[dtype], size]:
    """ReLU forward: y = max(0, x)."""
    var y = InlineArray[Scalar[dtype], size](fill=0)
    for i in range(size):
        y[i] = x[i] if x[i] > 0 else 0
    return y^


@always_inline
fn relu_backward[size: Int, dtype: DType = DEFAULT_DTYPE](
    dy: InlineArray[Scalar[dtype], size],
    x: InlineArray[Scalar[dtype], size],
) -> InlineArray[Scalar[dtype], size]:
    """ReLU backward: dx = dy * (x > 0)."""
    var dx = InlineArray[Scalar[dtype], size](fill=0)
    for i in range(size):
        dx[i] = dy[i] if x[i] > 0 else 0
    return dx^


# =============================================================================
# Loss Functions
# =============================================================================


@always_inline
fn mse_loss[size: Int, dtype: DType = DEFAULT_DTYPE](
    pred: InlineArray[Scalar[dtype], size],
    target: InlineArray[Scalar[dtype], size],
) -> Scalar[dtype]:
    """Mean squared error."""
    var sum: Scalar[dtype] = 0
    for i in range(size):
        var diff = pred[i] - target[i]
        sum += diff * diff
    return sum / size


@always_inline
fn mse_backward[size: Int, dtype: DType = DEFAULT_DTYPE](
    pred: InlineArray[Scalar[dtype], size],
    target: InlineArray[Scalar[dtype], size],
) -> InlineArray[Scalar[dtype], size]:
    """Gradient of MSE loss."""
    var grad = InlineArray[Scalar[dtype], size](fill=0)
    var scale = Scalar[dtype](2.0) / size
    for i in range(size):
        grad[i] = scale * (pred[i] - target[i])
    return grad^


# =============================================================================
# MLP2: Composed from primitives (wrapper for convenience)
# =============================================================================


struct MLP2[in_dim: Int, hidden_dim: Int, out_dim: Int, dtype: DType = DEFAULT_DTYPE]:
    """2-layer MLP built from composable primitives."""
    var layer1: LinearParams[Self.in_dim, Self.hidden_dim, Self.dtype]
    var layer2: LinearParams[Self.hidden_dim, Self.out_dim, Self.dtype]

    fn __init__(out self):
        self.layer1 = LinearParams[Self.in_dim, Self.hidden_dim, Self.dtype]()
        self.layer2 = LinearParams[Self.hidden_dim, Self.out_dim, Self.dtype]()

    @always_inline
    fn forward[batch: Int](
        self,
        x: InlineArray[Scalar[Self.dtype], batch * Self.in_dim],
    ) -> InlineArray[Scalar[Self.dtype], batch * Self.out_dim]:
        """Forward pass (no cache)."""
        var h = linear_forward[batch, Self.in_dim, Self.hidden_dim, Self.dtype](x, self.layer1)
        var h_relu = relu_forward[batch * Self.hidden_dim, Self.dtype](h)
        var out = linear_forward[batch, Self.hidden_dim, Self.out_dim, Self.dtype](h_relu, self.layer2)
        return out^

    @always_inline
    fn forward_cache[batch: Int](
        self,
        x: InlineArray[Scalar[Self.dtype], batch * Self.in_dim],
    ) -> Tuple[
        InlineArray[Scalar[Self.dtype], batch * Self.out_dim],
        InlineArray[Scalar[Self.dtype], batch * Self.in_dim],
        InlineArray[Scalar[Self.dtype], batch * Self.hidden_dim],
        InlineArray[Scalar[Self.dtype], batch * Self.hidden_dim],
    ]:
        """Forward with cache: returns (output, x, h_pre, h_post)."""
        var h_pre = linear_forward[batch, Self.in_dim, Self.hidden_dim, Self.dtype](x, self.layer1)
        var h = relu_forward[batch * Self.hidden_dim, Self.dtype](h_pre)
        var out = linear_forward[batch, Self.hidden_dim, Self.out_dim, Self.dtype](h, self.layer2)
        return (out, x, h_pre, h)

    @always_inline
    fn backward[batch: Int](
        mut self,
        dy: InlineArray[Scalar[Self.dtype], batch * Self.out_dim],
        x: InlineArray[Scalar[Self.dtype], batch * Self.in_dim],
        h_pre: InlineArray[Scalar[Self.dtype], batch * Self.hidden_dim],
        h: InlineArray[Scalar[Self.dtype], batch * Self.hidden_dim],
    ) -> InlineArray[Scalar[Self.dtype], batch * Self.in_dim]:
        """Backward through network."""
        var dh = linear_backward[batch, Self.hidden_dim, Self.out_dim, Self.dtype](dy, h, self.layer2)
        var dh_pre = relu_backward[batch * Self.hidden_dim, Self.dtype](dh, h_pre)
        var dx = linear_backward[batch, Self.in_dim, Self.hidden_dim, Self.dtype](dh_pre, x, self.layer1)
        return dx^

    fn update_adam(mut self, lr: Scalar[Self.dtype] = 0.001):
        """Update all parameters."""
        self.layer1.update_adam(lr)
        self.layer2.update_adam(lr)


# =============================================================================
# Demo
# =============================================================================


fn main() raises:
    seed(42)

    print("=" * 70)
    print("POC: Functional Layer API")
    print("=" * 70)
    print()

    comptime BATCH_SIZE = 1024
    comptime INPUT_DIM = 2
    comptime HIDDEN_DIM = 64
    comptime OUTPUT_DIM = 1
    comptime NUM_EPOCHS = 1000
    comptime dtype = DType.float32

    var net = MLP2[INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, dtype]()

    print("Network: " + String(INPUT_DIM) + " -> " + String(HIDDEN_DIM) + " (ReLU) -> " + String(OUTPUT_DIM))
    print("Task: Learn y = x1 * x2")
    print("Batch size: " + String(BATCH_SIZE))
    print()

    var lr = Scalar[dtype](0.001)
    var start_time = perf_counter_ns()

    print("Training...")
    print("-" * 70)

    for epoch in range(NUM_EPOCHS):
        var x = InlineArray[Scalar[dtype], BATCH_SIZE * INPUT_DIM](fill=0)
        var target = InlineArray[Scalar[dtype], BATCH_SIZE * OUTPUT_DIM](fill=0)

        for i in range(BATCH_SIZE):
            var x1 = Scalar[dtype](random_float64() * 2 - 1)
            var x2 = Scalar[dtype](random_float64() * 2 - 1)
            x[i * INPUT_DIM + 0] = x1
            x[i * INPUT_DIM + 1] = x2
            target[i] = x1 * x2

        var result = net.forward_cache[BATCH_SIZE](x)
        var y = result[0]
        var x_cache = result[1]
        var h_pre = result[2]
        var h = result[3]

        var loss = mse_loss[BATCH_SIZE * OUTPUT_DIM, dtype](y, target)
        var dy = mse_backward[BATCH_SIZE * OUTPUT_DIM, dtype](y, target)
        _ = net.backward[BATCH_SIZE](dy, x_cache, h_pre, h)
        net.update_adam(lr)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print("Epoch " + String(epoch + 1) + "/" + String(NUM_EPOCHS) + " - Loss: " + String(Float64(loss)))

    var elapsed_ms = Float64(perf_counter_ns() - start_time) / 1e6

    print("-" * 70)
    print()
    print("Training completed in " + String(elapsed_ms)[:8] + " ms")
    print("Average: " + String(elapsed_ms / Float64(NUM_EPOCHS))[:6] + " ms/epoch")
    print("Throughput: " + String(Int(Float64(NUM_EPOCHS * BATCH_SIZE) / (elapsed_ms / 1000.0))) + " samples/sec")
    print()

    # Evaluation
    print("Sample predictions:")
    var test_x = InlineArray[Scalar[dtype], 5 * INPUT_DIM](fill=0)
    var test_y = InlineArray[Scalar[dtype], 5](fill=0)

    for i in range(5):
        var x1 = Scalar[dtype](random_float64() * 2 - 1)
        var x2 = Scalar[dtype](random_float64() * 2 - 1)
        test_x[i * 2] = x1
        test_x[i * 2 + 1] = x2
        test_y[i] = x1 * x2

    var pred = net.forward[5](test_x)
    for i in range(5):
        print("  (" + String(Float64(test_x[i * 2]))[:6] + ", " +
              String(Float64(test_x[i * 2 + 1]))[:6] + ") -> " +
              String(Float64(pred[i]))[:7] + " vs " + String(Float64(test_y[i]))[:7])

    print()
    print("=" * 70)
    print()
    print("DESIGN SUMMARY")
    print("=" * 70)
    print()
    print("This POC demonstrates a functional API for neural networks:")
    print()
    print("1. SEPARATION OF CONCERNS:")
    print("   - LinearParams: weights, gradients, optimizer state (~60 LOC)")
    print("   - linear_forward/backward: pure functions (~40 LOC)")
    print("   - MLP2: composition wrapper (~50 LOC)")
    print()
    print("2. COMPOSABILITY:")
    print("   h = linear_forward(x, layer1)")
    print("   h = relu_forward(h)")
    print("   y = linear_forward(h, layer2)")
    print()
    print("3. GPU READINESS:")
    print("   All functions are @always_inline for kernel fusion")
    print("   No heap allocation, all InlineArray (stack)")
    print()
    print("4. COMPARE TO test_mlp.mojo:")
    print("   - test_mlp.mojo: ~1800 LOC monolithic GPU kernels")
    print("   - This design: ~200 LOC modular, composable")
    print("   - Add new layer: ~50 LOC (params + forward + backward)")
    print()
    print("NEXT STEPS FOR PRODUCTION:")
    print("   - Add SIMD optimization to free functions")
    print("   - Add GPU kernel wrappers that call inline functions")
    print("   - Add more layers: Conv2D, BatchNorm, Dropout")
    print("   - Generic Sequential[L1, L2, L3] using variadic params")
    print("=" * 70)

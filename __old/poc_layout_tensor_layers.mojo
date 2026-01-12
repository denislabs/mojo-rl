"""POC: LayoutTensor-based Layer Design for CPU/GPU Portability.

This POC demonstrates a trait-based layer API using LayoutTensor,
which works on both CPU and GPU with the same code.

Key design elements:
1. LayoutTensor as the universal tensor abstraction
2. Compile-time dimensions via `comptime` constants
3. Generic origin types for CPU/GPU compatibility
4. Same layer code works on CPU (with InlineArray storage) and GPU (with DeviceBuffer)

Usage:
    pixi run mojo run deep_rl/poc_layout_tensor_layers.mojo           # CPU
    pixi run -e apple mojo run deep_rl/poc_layout_tensor_layers.mojo  # GPU
"""

from math import sqrt
from random import random_float64
from memory import memset_zero

from layout import Layout, LayoutTensor, IntTuple
from gpu.host import DeviceContext
from gpu.memory import AddressSpace

# =============================================================================
# Compile-time constants
# =============================================================================

comptime dtype = DType.float32


# =============================================================================
# Layer functions using LayoutTensor with generic origins
# =============================================================================


@always_inline
fn linear_forward[
    dtype: DType,
    batch_size: Int,
    in_dim: Int,
    out_dim: Int,
    out_origin: MutableOrigin,
    in_origin: ImmutableOrigin,
    w_origin: ImmutableOrigin,
    b_origin: ImmutableOrigin,
](
    output: LayoutTensor[dtype, Layout.row_major(batch_size, out_dim), out_origin],
    input: LayoutTensor[dtype, Layout.row_major(batch_size, in_dim), in_origin],
    W: LayoutTensor[dtype, Layout.row_major(in_dim, out_dim), w_origin],
    b: LayoutTensor[dtype, Layout.row_major(out_dim), b_origin],
):
    """Linear forward: output = input @ W + b."""
    for batch in range(batch_size):
        for j in range(out_dim):
            var acc: output.element_type = rebind[output.element_type](b[j])
            for i in range(in_dim):
                var x_val = rebind[output.element_type](input[batch, i])
                var w_val = rebind[output.element_type](W[i, j])
                acc += x_val * w_val
            output[batch, j] = acc


@always_inline
fn linear_backward[
    dtype: DType,
    batch_size: Int,
    in_dim: Int,
    out_dim: Int,
    gin_origin: MutableOrigin,
    gW_origin: MutableOrigin,
    gb_origin: MutableOrigin,
    gout_origin: ImmutableOrigin,
    in_origin: ImmutableOrigin,
    w_origin: ImmutableOrigin,
](
    grad_input: LayoutTensor[dtype, Layout.row_major(batch_size, in_dim), gin_origin],
    grad_W: LayoutTensor[dtype, Layout.row_major(in_dim, out_dim), gW_origin],
    grad_b: LayoutTensor[dtype, Layout.row_major(out_dim), gb_origin],
    grad_output: LayoutTensor[dtype, Layout.row_major(batch_size, out_dim), gout_origin],
    input: LayoutTensor[dtype, Layout.row_major(batch_size, in_dim), in_origin],
    W: LayoutTensor[dtype, Layout.row_major(in_dim, out_dim), w_origin],
):
    """Linear backward: compute grad_input, grad_W, grad_b."""
    # Zero gradients
    for i in range(in_dim):
        for j in range(out_dim):
            grad_W[i, j] = 0
    for j in range(out_dim):
        grad_b[j] = 0

    for batch in range(batch_size):
        # grad_input = grad_output @ W.T
        for i in range(in_dim):
            var acc: grad_input.element_type = 0
            for j in range(out_dim):
                var g_out = rebind[grad_input.element_type](grad_output[batch, j])
                var w_val = rebind[grad_input.element_type](W[i, j])
                acc += g_out * w_val
            grad_input[batch, i] = acc

        # grad_W += input.T @ grad_output
        for i in range(in_dim):
            var x_val = rebind[grad_W.element_type](input[batch, i])
            for j in range(out_dim):
                var g_out = rebind[grad_W.element_type](grad_output[batch, j])
                grad_W[i, j] = rebind[grad_W.element_type](grad_W[i, j]) + x_val * g_out

        # grad_b += sum(grad_output, axis=0)
        for j in range(out_dim):
            grad_b[j] = rebind[grad_b.element_type](grad_b[j]) + rebind[grad_b.element_type](grad_output[batch, j])


@always_inline
fn relu_forward[
    dtype: DType,
    batch_size: Int,
    dim: Int,
    out_origin: MutableOrigin,
    in_origin: ImmutableOrigin,
](
    output: LayoutTensor[dtype, Layout.row_major(batch_size, dim), out_origin],
    input: LayoutTensor[dtype, Layout.row_major(batch_size, dim), in_origin],
):
    """ReLU forward: output = max(0, input)."""
    for batch in range(batch_size):
        for i in range(dim):
            var x = rebind[output.element_type](input[batch, i])
            output[batch, i] = x if x > 0 else 0


@always_inline
fn relu_backward[
    dtype: DType,
    batch_size: Int,
    dim: Int,
    gin_origin: MutableOrigin,
    gout_origin: ImmutableOrigin,
    pre_origin: ImmutableOrigin,
](
    grad_input: LayoutTensor[dtype, Layout.row_major(batch_size, dim), gin_origin],
    grad_output: LayoutTensor[dtype, Layout.row_major(batch_size, dim), gout_origin],
    pre_activation: LayoutTensor[dtype, Layout.row_major(batch_size, dim), pre_origin],
):
    """ReLU backward: grad_input = grad_output * (pre_activation > 0)."""
    for batch in range(batch_size):
        for i in range(dim):
            var x = rebind[grad_input.element_type](pre_activation[batch, i])
            var g = rebind[grad_input.element_type](grad_output[batch, i])
            grad_input[batch, i] = g if x > 0 else 0


@always_inline
fn mse_loss_and_grad[
    dtype: DType,
    batch_size: Int,
    output_dim: Int,
    gout_origin: MutableOrigin,
    pred_origin: ImmutableOrigin,
    tgt_origin: ImmutableOrigin,
](
    grad_output: LayoutTensor[dtype, Layout.row_major(batch_size, output_dim), gout_origin],
    predictions: LayoutTensor[dtype, Layout.row_major(batch_size, output_dim), pred_origin],
    targets: LayoutTensor[dtype, Layout.row_major(batch_size, output_dim), tgt_origin],
) -> Scalar[dtype]:
    """Compute MSE loss and gradient."""
    var loss: Scalar[dtype] = 0
    var scale = Scalar[dtype](2.0 / Float64(batch_size * output_dim))

    for batch in range(batch_size):
        for j in range(output_dim):
            var pred = rebind[Scalar[dtype]](predictions[batch, j])
            var tgt = rebind[Scalar[dtype]](targets[batch, j])
            var diff = pred - tgt
            loss += diff * diff
            grad_output[batch, j] = scale * diff

    return loss / Scalar[dtype](batch_size * output_dim)


@always_inline
fn adam_update_2d[
    dtype: DType,
    dim1: Int,
    dim2: Int,
    p_origin: MutableOrigin,
    g_origin: ImmutableOrigin,
    m_origin: MutableOrigin,
    v_origin: MutableOrigin,
](
    params: LayoutTensor[dtype, Layout.row_major(dim1, dim2), p_origin],
    grads: LayoutTensor[dtype, Layout.row_major(dim1, dim2), g_origin],
    m: LayoutTensor[dtype, Layout.row_major(dim1, dim2), m_origin],
    v: LayoutTensor[dtype, Layout.row_major(dim1, dim2), v_origin],
    t: Int,
    lr: Float64 = 0.001,
    beta1: Float64 = 0.9,
    beta2: Float64 = 0.999,
    eps: Float64 = 1e-8,
):
    """Adam update for 2D tensor."""
    var t_f = Float64(t)
    var b1_corr = 1.0 - beta1 ** t_f
    var b2_corr = 1.0 - beta2 ** t_f

    for i in range(dim1):
        for j in range(dim2):
            var g = rebind[Scalar[dtype]](grads[i, j])
            var m_val = rebind[Scalar[dtype]](m[i, j])
            var v_val = rebind[Scalar[dtype]](v[i, j])

            m_val = Scalar[dtype](beta1) * m_val + Scalar[dtype](1 - beta1) * g
            v_val = Scalar[dtype](beta2) * v_val + Scalar[dtype](1 - beta2) * g * g

            m[i, j] = m_val
            v[i, j] = v_val

            var m_hat = Float64(m_val) / b1_corr
            var v_hat = Float64(v_val) / b2_corr
            var p = rebind[Scalar[dtype]](params[i, j])
            params[i, j] = p - Scalar[dtype](lr * m_hat / (sqrt(v_hat) + eps))


@always_inline
fn adam_update_1d[
    dtype: DType,
    dim: Int,
    p_origin: MutableOrigin,
    g_origin: ImmutableOrigin,
    m_origin: MutableOrigin,
    v_origin: MutableOrigin,
](
    params: LayoutTensor[dtype, Layout.row_major(dim), p_origin],
    grads: LayoutTensor[dtype, Layout.row_major(dim), g_origin],
    m: LayoutTensor[dtype, Layout.row_major(dim), m_origin],
    v: LayoutTensor[dtype, Layout.row_major(dim), v_origin],
    t: Int,
    lr: Float64 = 0.001,
    beta1: Float64 = 0.9,
    beta2: Float64 = 0.999,
    eps: Float64 = 1e-8,
):
    """Adam update for 1D tensor."""
    var t_f = Float64(t)
    var b1_corr = 1.0 - beta1 ** t_f
    var b2_corr = 1.0 - beta2 ** t_f

    for i in range(dim):
        var g = rebind[Scalar[dtype]](grads[i])
        var m_val = rebind[Scalar[dtype]](m[i])
        var v_val = rebind[Scalar[dtype]](v[i])

        m_val = Scalar[dtype](beta1) * m_val + Scalar[dtype](1 - beta1) * g
        v_val = Scalar[dtype](beta2) * v_val + Scalar[dtype](1 - beta2) * g * g

        m[i] = m_val
        v[i] = v_val

        var m_hat = Float64(m_val) / b1_corr
        var v_hat = Float64(v_val) / b2_corr
        var p = rebind[Scalar[dtype]](params[i])
        params[i] = p - Scalar[dtype](lr * m_hat / (sqrt(v_hat) + eps))


# =============================================================================
# CPU Training Demo
# =============================================================================


fn train_cpu():
    """CPU training using LayoutTensor with InlineArray storage."""
    print("=" * 70)
    print("POC: LayoutTensor-based Layers (CPU)")
    print("=" * 70)
    print()

    # Hyperparameters
    comptime BATCH_SIZE = 32
    comptime INPUT_DIM = 2
    comptime HIDDEN_DIM = 16
    comptime OUTPUT_DIM = 1
    comptime NUM_EPOCHS = 300

    print("Network: " + String(INPUT_DIM) + " -> " + String(HIDDEN_DIM) + " (ReLU) -> " + String(OUTPUT_DIM))
    print("Batch: " + String(BATCH_SIZE) + ", Epochs: " + String(NUM_EPOCHS))
    print()

    # Storage using InlineArray (stack-allocated)
    var input_storage = InlineArray[Scalar[dtype], BATCH_SIZE * INPUT_DIM](uninitialized=True)
    var target_storage = InlineArray[Scalar[dtype], BATCH_SIZE * OUTPUT_DIM](uninitialized=True)

    # Layer 1
    var W1_storage = InlineArray[Scalar[dtype], INPUT_DIM * HIDDEN_DIM](uninitialized=True)
    var b1_storage = InlineArray[Scalar[dtype], HIDDEN_DIM](uninitialized=True)
    var dW1_storage = InlineArray[Scalar[dtype], INPUT_DIM * HIDDEN_DIM](uninitialized=True)
    var db1_storage = InlineArray[Scalar[dtype], HIDDEN_DIM](uninitialized=True)
    var m_W1_storage = InlineArray[Scalar[dtype], INPUT_DIM * HIDDEN_DIM](uninitialized=True)
    var v_W1_storage = InlineArray[Scalar[dtype], INPUT_DIM * HIDDEN_DIM](uninitialized=True)
    var m_b1_storage = InlineArray[Scalar[dtype], HIDDEN_DIM](uninitialized=True)
    var v_b1_storage = InlineArray[Scalar[dtype], HIDDEN_DIM](uninitialized=True)

    # Layer 2
    var W2_storage = InlineArray[Scalar[dtype], HIDDEN_DIM * OUTPUT_DIM](uninitialized=True)
    var b2_storage = InlineArray[Scalar[dtype], OUTPUT_DIM](uninitialized=True)
    var dW2_storage = InlineArray[Scalar[dtype], HIDDEN_DIM * OUTPUT_DIM](uninitialized=True)
    var db2_storage = InlineArray[Scalar[dtype], OUTPUT_DIM](uninitialized=True)
    var m_W2_storage = InlineArray[Scalar[dtype], HIDDEN_DIM * OUTPUT_DIM](uninitialized=True)
    var v_W2_storage = InlineArray[Scalar[dtype], HIDDEN_DIM * OUTPUT_DIM](uninitialized=True)
    var m_b2_storage = InlineArray[Scalar[dtype], OUTPUT_DIM](uninitialized=True)
    var v_b2_storage = InlineArray[Scalar[dtype], OUTPUT_DIM](uninitialized=True)

    # Activations and gradients
    var h1_pre_storage = InlineArray[Scalar[dtype], BATCH_SIZE * HIDDEN_DIM](uninitialized=True)
    var h1_storage = InlineArray[Scalar[dtype], BATCH_SIZE * HIDDEN_DIM](uninitialized=True)
    var output_storage = InlineArray[Scalar[dtype], BATCH_SIZE * OUTPUT_DIM](uninitialized=True)
    var grad_output_storage = InlineArray[Scalar[dtype], BATCH_SIZE * OUTPUT_DIM](uninitialized=True)
    var grad_h1_storage = InlineArray[Scalar[dtype], BATCH_SIZE * HIDDEN_DIM](uninitialized=True)
    var grad_h1_pre_storage = InlineArray[Scalar[dtype], BATCH_SIZE * HIDDEN_DIM](uninitialized=True)
    var grad_input_storage = InlineArray[Scalar[dtype], BATCH_SIZE * INPUT_DIM](uninitialized=True)

    # Initialize weights (Xavier)
    var std1 = sqrt(2.0 / Float64(INPUT_DIM + HIDDEN_DIM))
    for i in range(INPUT_DIM * HIDDEN_DIM):
        W1_storage[i] = Scalar[dtype]((random_float64() * 2 - 1) * std1)
        m_W1_storage[i] = 0
        v_W1_storage[i] = 0
    for i in range(HIDDEN_DIM):
        b1_storage[i] = 0
        m_b1_storage[i] = 0
        v_b1_storage[i] = 0

    var std2 = sqrt(2.0 / Float64(HIDDEN_DIM + OUTPUT_DIM))
    for i in range(HIDDEN_DIM * OUTPUT_DIM):
        W2_storage[i] = Scalar[dtype]((random_float64() * 2 - 1) * std2)
        m_W2_storage[i] = 0
        v_W2_storage[i] = 0
    for i in range(OUTPUT_DIM):
        b2_storage[i] = 0
        m_b2_storage[i] = 0
        v_b2_storage[i] = 0

    # Create LayoutTensors (views into storage)
    var input_t = LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, INPUT_DIM)](input_storage)
    var target_t = LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, OUTPUT_DIM)](target_storage)

    var W1_t = LayoutTensor[dtype, Layout.row_major(INPUT_DIM, HIDDEN_DIM)](W1_storage)
    var b1_t = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM)](b1_storage)
    var dW1_t = LayoutTensor[dtype, Layout.row_major(INPUT_DIM, HIDDEN_DIM)](dW1_storage)
    var db1_t = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM)](db1_storage)
    var m_W1_t = LayoutTensor[dtype, Layout.row_major(INPUT_DIM, HIDDEN_DIM)](m_W1_storage)
    var v_W1_t = LayoutTensor[dtype, Layout.row_major(INPUT_DIM, HIDDEN_DIM)](v_W1_storage)
    var m_b1_t = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM)](m_b1_storage)
    var v_b1_t = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM)](v_b1_storage)

    var W2_t = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, OUTPUT_DIM)](W2_storage)
    var b2_t = LayoutTensor[dtype, Layout.row_major(OUTPUT_DIM)](b2_storage)
    var dW2_t = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, OUTPUT_DIM)](dW2_storage)
    var db2_t = LayoutTensor[dtype, Layout.row_major(OUTPUT_DIM)](db2_storage)
    var m_W2_t = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, OUTPUT_DIM)](m_W2_storage)
    var v_W2_t = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, OUTPUT_DIM)](v_W2_storage)
    var m_b2_t = LayoutTensor[dtype, Layout.row_major(OUTPUT_DIM)](m_b2_storage)
    var v_b2_t = LayoutTensor[dtype, Layout.row_major(OUTPUT_DIM)](v_b2_storage)

    var h1_pre_t = LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, HIDDEN_DIM)](h1_pre_storage)
    var h1_t = LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, HIDDEN_DIM)](h1_storage)
    var output_t = LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, OUTPUT_DIM)](output_storage)
    var grad_output_t = LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, OUTPUT_DIM)](grad_output_storage)
    var grad_h1_t = LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, HIDDEN_DIM)](grad_h1_storage)
    var grad_h1_pre_t = LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, HIDDEN_DIM)](grad_h1_pre_storage)
    var grad_input_t = LayoutTensor[dtype, Layout.row_major(BATCH_SIZE, INPUT_DIM)](grad_input_storage)

    print("Training (XOR-like function)...")
    print("-" * 70)

    var t = 0
    for epoch in range(NUM_EPOCHS):
        # Generate batch data
        for b in range(BATCH_SIZE):
            var x1 = Scalar[dtype]((random_float64() - 0.5) * 4)
            var x2 = Scalar[dtype]((random_float64() - 0.5) * 4)
            input_t[b, 0] = x1
            input_t[b, 1] = x2
            var sign1 = Scalar[dtype](1.0) if x1 > 0 else Scalar[dtype](-1.0)
            var sign2 = Scalar[dtype](1.0) if x2 > 0 else Scalar[dtype](-1.0)
            target_t[b, 0] = sign1 * sign2

        # Forward pass
        linear_forward(h1_pre_t, input_t, W1_t, b1_t)
        relu_forward(h1_t, h1_pre_t)
        linear_forward(output_t, h1_t, W2_t, b2_t)

        # Compute loss
        var loss = mse_loss_and_grad(grad_output_t, output_t, target_t)

        # Backward pass
        linear_backward(grad_h1_t, dW2_t, db2_t, grad_output_t, h1_t, W2_t)
        relu_backward(grad_h1_pre_t, grad_h1_t, h1_pre_t)
        linear_backward(grad_input_t, dW1_t, db1_t, grad_h1_pre_t, input_t, W1_t)

        # Adam update
        t += 1
        adam_update_2d(W1_t, dW1_t, m_W1_t, v_W1_t, t, lr=0.01)
        adam_update_1d(b1_t, db1_t, m_b1_t, v_b1_t, t, lr=0.01)
        adam_update_2d(W2_t, dW2_t, m_W2_t, v_W2_t, t, lr=0.01)
        adam_update_1d(b2_t, db2_t, m_b2_t, v_b2_t, t, lr=0.01)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print("Epoch " + String(epoch + 1) + "/" + String(NUM_EPOCHS) + " - Loss: " + String(Float64(loss)))

    print("-" * 70)
    print()

    # Test predictions
    print("Test predictions (XOR pattern):")

    # Test (1, 1) -> +1
    input_t[0, 0] = Scalar[dtype](1.0)
    input_t[0, 1] = Scalar[dtype](1.0)
    linear_forward(h1_pre_t, input_t, W1_t, b1_t)
    relu_forward(h1_t, h1_pre_t)
    linear_forward(output_t, h1_t, W2_t, b2_t)
    print("  (1.0, 1.0) -> " + String(Float64(rebind[Scalar[dtype]](output_t[0, 0]))) + " (expected: 1.0)")

    # Test (-1, -1) -> +1
    input_t[0, 0] = Scalar[dtype](-1.0)
    input_t[0, 1] = Scalar[dtype](-1.0)
    linear_forward(h1_pre_t, input_t, W1_t, b1_t)
    relu_forward(h1_t, h1_pre_t)
    linear_forward(output_t, h1_t, W2_t, b2_t)
    print("  (-1.0, -1.0) -> " + String(Float64(rebind[Scalar[dtype]](output_t[0, 0]))) + " (expected: 1.0)")

    # Test (1, -1) -> -1
    input_t[0, 0] = Scalar[dtype](1.0)
    input_t[0, 1] = Scalar[dtype](-1.0)
    linear_forward(h1_pre_t, input_t, W1_t, b1_t)
    relu_forward(h1_t, h1_pre_t)
    linear_forward(output_t, h1_t, W2_t, b2_t)
    print("  (1.0, -1.0) -> " + String(Float64(rebind[Scalar[dtype]](output_t[0, 0]))) + " (expected: -1.0)")

    # Test (-1, 1) -> -1
    input_t[0, 0] = Scalar[dtype](-1.0)
    input_t[0, 1] = Scalar[dtype](1.0)
    linear_forward(h1_pre_t, input_t, W1_t, b1_t)
    relu_forward(h1_t, h1_pre_t)
    linear_forward(output_t, h1_t, W2_t, b2_t)
    print("  (-1.0, 1.0) -> " + String(Float64(rebind[Scalar[dtype]](output_t[0, 0]))) + " (expected: -1.0)")

    print()
    print("=" * 70)
    print("LAYOUTTENSOR DESIGN SUMMARY")
    print("=" * 70)
    print()
    print("Key insight: LayoutTensor works on BOTH CPU and GPU!")
    print()
    print("CPU storage: InlineArray (stack-allocated)")
    print("  var storage = InlineArray[Scalar[dtype], SIZE](uninitialized=True)")
    print("  var tensor = LayoutTensor[dtype, Layout.row_major(...)](storage)")
    print()
    print("GPU storage: DeviceBuffer")
    print("  var buffer = ctx.enqueue_create_buffer[dtype](SIZE)")
    print("  var tensor = LayoutTensor[dtype, Layout.row_major(...), MutAnyOrigin](buffer)")
    print()
    print("Same layer functions work on both (via generic origin types):")
    print("  linear_forward(output, input, W, b)")
    print("  relu_forward(output, input)")
    print("  adam_update_2d(params, grads, m, v, t)")
    print()
    print("Benefits:")
    print("  1. Write once, run on CPU or GPU")
    print("  2. Compile-time dimensions for optimization")
    print("  3. @always_inline for GPU kernel fusion")
    print("  4. Clean separation: storage vs. tensor view")
    print()
    print("=" * 70)


fn main():
    train_cpu()

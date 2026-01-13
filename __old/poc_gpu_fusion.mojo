"""POC: GPU Kernel Fusion with Functional Layer Design.

This demonstrates that a modular functional API is compatible with
high-performance GPU execution through kernel fusion.

Key insight: @always_inline helper functions get inlined into GPU kernels,
enabling the compiler to fuse operations without function call overhead.

Fusion patterns demonstrated:
1. relu_scalar() inlined into forward kernel
2. relu_backward_scalar() inlined into backward kernel
3. All Adam updates in single fused kernel

Run: pixi run -e apple mojo run deep_rl/poc_gpu_fusion.mojo
"""

from time import perf_counter_ns
from math import sqrt
from random import seed, random_float64

from layout import Layout, LayoutTensor
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace


# =============================================================================
# Constants
# =============================================================================

comptime dtype = DType.float32
comptime TILE = 16
comptime TPB = 256

comptime BATCH_SIZE = 1024
comptime INPUT_DIM = 2
comptime HIDDEN_DIM = 64
comptime OUTPUT_DIM = 1
comptime NUM_EPOCHS = 1000


# =============================================================================
# Inline Layer Primitives (for GPU kernel fusion)
# =============================================================================


@always_inline
fn relu_scalar[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Inline ReLU - fuses into kernel."""
    return x if x > 0 else 0


@always_inline
fn relu_backward_scalar[T: DType](dy: Scalar[T], x: Scalar[T]) -> Scalar[T]:
    """Inline ReLU backward - fuses into kernel."""
    return dy if x > 0 else 0


# =============================================================================
# Forward Kernel with Fused ReLU
# =============================================================================


fn forward_layer1_relu_kernel[
    BATCH: Int, IN_DIM: Int, OUT_DIM: Int
](
    output_pre: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ],
    output: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    x: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(OUT_DIM), ImmutAnyOrigin],
):
    """Fused forward: y = ReLU(x @ W + b). Outputs both pre and post ReLU."""
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var batch_idx = Int(block_idx.y) * TILE + local_row
    var out_idx = Int(block_idx.x) * TILE + local_col

    var x_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var W_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: output.element_type = 0
    if out_idx < OUT_DIM:
        acc = b[out_idx]

    comptime num_tiles = (IN_DIM + TILE - 1) // TILE
    for tile_idx in range(num_tiles):
        var in_idx = tile_idx * TILE + local_col
        if batch_idx < BATCH and in_idx < IN_DIM:
            x_shared[local_row, local_col] = x[batch_idx, in_idx]
        else:
            x_shared[local_row, local_col] = 0

        var w_row = tile_idx * TILE + local_row
        if w_row < IN_DIM and out_idx < OUT_DIM:
            W_shared[local_row, local_col] = W[w_row, out_idx]
        else:
            W_shared[local_row, local_col] = 0

        barrier()

        for k in range(TILE):
            acc += x_shared[local_row, k] * W_shared[k, local_col]

        barrier()

    if batch_idx < BATCH and out_idx < OUT_DIM:
        output_pre[batch_idx, out_idx] = acc
        # INLINE ReLU - this is the key fusion!
        var acc_scalar = rebind[Scalar[dtype]](acc)
        output[batch_idx, out_idx] = relu_scalar(acc_scalar)


fn forward_layer2_kernel[
    BATCH: Int, IN_DIM: Int, OUT_DIM: Int
](
    output: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    x: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(OUT_DIM), ImmutAnyOrigin],
):
    """Linear forward: y = x @ W + b."""
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var batch_idx = Int(block_idx.y) * TILE + local_row
    var out_idx = Int(block_idx.x) * TILE + local_col

    var x_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var W_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: output.element_type = 0
    if out_idx < OUT_DIM:
        acc = b[out_idx]

    comptime num_tiles = (IN_DIM + TILE - 1) // TILE
    for tile_idx in range(num_tiles):
        var in_idx = tile_idx * TILE + local_col
        if batch_idx < BATCH and in_idx < IN_DIM:
            x_shared[local_row, local_col] = x[batch_idx, in_idx]
        else:
            x_shared[local_row, local_col] = 0

        var w_row = tile_idx * TILE + local_row
        if w_row < IN_DIM and out_idx < OUT_DIM:
            W_shared[local_row, local_col] = W[w_row, out_idx]
        else:
            W_shared[local_row, local_col] = 0

        barrier()

        for k in range(TILE):
            acc += x_shared[local_row, k] * W_shared[k, local_col]

        barrier()

    if batch_idx < BATCH and out_idx < OUT_DIM:
        output[batch_idx, out_idx] = acc


# =============================================================================
# Backward Kernels with Fused Operations
# =============================================================================


fn backward_mse_dW_db_kernel[
    BATCH: Int, IN_DIM: Int, OUT_DIM: Int
](
    dW: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), MutAnyOrigin],
    db: LayoutTensor[dtype, Layout.row_major(OUT_DIM), MutAnyOrigin],
    d_output: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ],
    x: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    pred: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin],
    target: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin
    ],
):
    """Fused: MSE gradient + dW + db in one kernel."""
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var in_idx = Int(block_idx.y) * TILE + local_row
    var out_idx = Int(block_idx.x) * TILE + local_col

    var x_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var dy_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var dW_acc: dW.element_type = 0
    var db_acc: dW.element_type = 0
    var compute_db = block_idx.y == 0
    var mse_scale: dW.element_type = rebind[dW.element_type](
        Scalar[dtype](2.0) / Scalar[dtype](BATCH * OUT_DIM)
    )

    comptime num_tiles = (BATCH + TILE - 1) // TILE
    for tile_idx in range(num_tiles):
        var batch_idx = tile_idx * TILE + local_col
        if in_idx < IN_DIM and batch_idx < BATCH:
            x_shared[local_row, local_col] = x[batch_idx, in_idx]
        else:
            x_shared[local_row, local_col] = 0

        var dy_batch = tile_idx * TILE + local_row
        var dy_val: dy_shared.element_type = 0
        if dy_batch < BATCH and out_idx < OUT_DIM:
            var p = pred[dy_batch, out_idx]
            var t = target[dy_batch, out_idx]
            dy_val = mse_scale * (p - t)
            dy_shared[local_row, local_col] = dy_val

            if compute_db:
                d_output[dy_batch, out_idx] = dy_val
                db_acc += dy_val
        else:
            dy_shared[local_row, local_col] = 0

        barrier()

        for k in range(TILE):
            dW_acc += x_shared[local_row, k] * dy_shared[k, local_col]

        barrier()

    if in_idx < IN_DIM and out_idx < OUT_DIM:
        dW[in_idx, out_idx] = dW_acc

    if compute_db:
        dy_shared[local_row, local_col] = db_acc
        barrier()
        var stride = TILE // 2
        while stride > 0:
            if local_row < stride:
                dy_shared[local_row, local_col] += dy_shared[
                    local_row + stride, local_col
                ]
            barrier()
            stride //= 2
        if local_row == 0 and out_idx < OUT_DIM:
            db[out_idx] = dy_shared[0, local_col]


fn backward_dx_relu_dW_db_kernel[
    BATCH: Int, IN_DIM: Int, HIDDEN_DIM: Int, OUT_DIM: Int
](
    dW: LayoutTensor[dtype, Layout.row_major(IN_DIM, HIDDEN_DIM), MutAnyOrigin],
    db: LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), MutAnyOrigin],
    x: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    h_pre: LayoutTensor[
        dtype, Layout.row_major(BATCH, HIDDEN_DIM), ImmutAnyOrigin
    ],
    dy: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin],
    W2: LayoutTensor[
        dtype, Layout.row_major(HIDDEN_DIM, OUT_DIM), ImmutAnyOrigin
    ],
):
    """Fused: (dy @ W2.T) * relu_grad + dW1 + db1."""
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var in_idx = Int(block_idx.y) * TILE + local_row
    var hidden_idx = Int(block_idx.x) * TILE + local_col

    var x_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var dh_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var dW_acc: dW.element_type = 0
    var db_acc: dW.element_type = 0
    var compute_db = block_idx.y == 0

    comptime num_tiles = (BATCH + TILE - 1) // TILE
    for tile_idx in range(num_tiles):
        var batch_idx = tile_idx * TILE + local_col
        if in_idx < IN_DIM and batch_idx < BATCH:
            x_shared[local_row, local_col] = x[batch_idx, in_idx]
        else:
            x_shared[local_row, local_col] = 0

        var dh_batch = tile_idx * TILE + local_row
        var dh_val: dh_shared.element_type = 0
        if dh_batch < BATCH and hidden_idx < HIDDEN_DIM:
            # Compute dh = dy @ W2.T
            for k in range(OUT_DIM):
                dh_val += dy[dh_batch, k] * W2[hidden_idx, k]
            # INLINE ReLU backward - key fusion!
            var h_pre_val = rebind[Scalar[dtype]](h_pre[dh_batch, hidden_idx])
            var dh_scalar = rebind[Scalar[dtype]](dh_val)
            dh_val = rebind[dh_shared.element_type](
                relu_backward_scalar(dh_scalar, h_pre_val)
            )
            dh_shared[local_row, local_col] = dh_val
            if compute_db:
                db_acc += dh_val
        else:
            dh_shared[local_row, local_col] = 0

        barrier()

        for k in range(TILE):
            dW_acc += x_shared[local_row, k] * dh_shared[k, local_col]

        barrier()

    if in_idx < IN_DIM and hidden_idx < HIDDEN_DIM:
        dW[in_idx, hidden_idx] = dW_acc

    if compute_db:
        dh_shared[local_row, local_col] = db_acc
        barrier()
        var stride = TILE // 2
        while stride > 0:
            if local_row < stride:
                dh_shared[local_row, local_col] += dh_shared[
                    local_row + stride, local_col
                ]
            barrier()
            stride //= 2
        if local_row == 0 and hidden_idx < HIDDEN_DIM:
            db[hidden_idx] = dh_shared[0, local_col]


# =============================================================================
# Fused Adam Kernel
# =============================================================================


fn adam_fused_kernel[
    W1_SIZE: Int, B1_SIZE: Int, W2_SIZE: Int, B2_SIZE: Int
](
    W1: LayoutTensor[dtype, Layout.row_major(W1_SIZE), MutAnyOrigin],
    dW1: LayoutTensor[dtype, Layout.row_major(W1_SIZE), ImmutAnyOrigin],
    m_W1: LayoutTensor[dtype, Layout.row_major(W1_SIZE), MutAnyOrigin],
    v_W1: LayoutTensor[dtype, Layout.row_major(W1_SIZE), MutAnyOrigin],
    b1: LayoutTensor[dtype, Layout.row_major(B1_SIZE), MutAnyOrigin],
    db1: LayoutTensor[dtype, Layout.row_major(B1_SIZE), ImmutAnyOrigin],
    m_b1: LayoutTensor[dtype, Layout.row_major(B1_SIZE), MutAnyOrigin],
    v_b1: LayoutTensor[dtype, Layout.row_major(B1_SIZE), MutAnyOrigin],
    W2: LayoutTensor[dtype, Layout.row_major(W2_SIZE), MutAnyOrigin],
    dW2: LayoutTensor[dtype, Layout.row_major(W2_SIZE), ImmutAnyOrigin],
    m_W2: LayoutTensor[dtype, Layout.row_major(W2_SIZE), MutAnyOrigin],
    v_W2: LayoutTensor[dtype, Layout.row_major(W2_SIZE), MutAnyOrigin],
    b2: LayoutTensor[dtype, Layout.row_major(B2_SIZE), MutAnyOrigin],
    db2: LayoutTensor[dtype, Layout.row_major(B2_SIZE), ImmutAnyOrigin],
    m_b2: LayoutTensor[dtype, Layout.row_major(B2_SIZE), MutAnyOrigin],
    v_b2: LayoutTensor[dtype, Layout.row_major(B2_SIZE), MutAnyOrigin],
    lr: Scalar[dtype],
    beta1: Scalar[dtype],
    beta2: Scalar[dtype],
    eps: Scalar[dtype],
    bc1: Scalar[dtype],
    bc2: Scalar[dtype],
):
    """Fused Adam for all parameters."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    comptime TOTAL = W1_SIZE + B1_SIZE + W2_SIZE + B2_SIZE
    if idx >= TOTAL:
        return

    if idx < W1_SIZE:
        var g = rebind[Scalar[dtype]](dW1[idx])
        var m = rebind[Scalar[dtype]](m_W1[idx])
        var v = rebind[Scalar[dtype]](v_W1[idx])
        var m_new = beta1 * m + (1 - beta1) * g
        var v_new = beta2 * v + (1 - beta2) * g * g
        W1[idx] = rebind[Scalar[dtype]](W1[idx]) - lr * (m_new / bc1) / (
            sqrt(v_new / bc2) + eps
        )
        m_W1[idx] = m_new
        v_W1[idx] = v_new
    elif idx < W1_SIZE + B1_SIZE:
        var i = idx - W1_SIZE
        var g = rebind[Scalar[dtype]](db1[i])
        var m = rebind[Scalar[dtype]](m_b1[i])
        var v = rebind[Scalar[dtype]](v_b1[i])
        var m_new = beta1 * m + (1 - beta1) * g
        var v_new = beta2 * v + (1 - beta2) * g * g
        b1[i] = rebind[Scalar[dtype]](b1[i]) - lr * (m_new / bc1) / (
            sqrt(v_new / bc2) + eps
        )
        m_b1[i] = m_new
        v_b1[i] = v_new
    elif idx < W1_SIZE + B1_SIZE + W2_SIZE:
        var i = idx - W1_SIZE - B1_SIZE
        var g = rebind[Scalar[dtype]](dW2[i])
        var m = rebind[Scalar[dtype]](m_W2[i])
        var v = rebind[Scalar[dtype]](v_W2[i])
        var m_new = beta1 * m + (1 - beta1) * g
        var v_new = beta2 * v + (1 - beta2) * g * g
        W2[i] = rebind[Scalar[dtype]](W2[i]) - lr * (m_new / bc1) / (
            sqrt(v_new / bc2) + eps
        )
        m_W2[i] = m_new
        v_W2[i] = v_new
    else:
        var i = idx - W1_SIZE - B1_SIZE - W2_SIZE
        var g = rebind[Scalar[dtype]](db2[i])
        var m = rebind[Scalar[dtype]](m_b2[i])
        var v = rebind[Scalar[dtype]](v_b2[i])
        var m_new = beta1 * m + (1 - beta1) * g
        var v_new = beta2 * v + (1 - beta2) * g * g
        b2[i] = rebind[Scalar[dtype]](b2[i]) - lr * (m_new / bc1) / (
            sqrt(v_new / bc2) + eps
        )
        m_b2[i] = m_new
        v_b2[i] = v_new


# =============================================================================
# Utility Kernels
# =============================================================================


fn generate_data_kernel[
    BATCH: Int
](
    x: LayoutTensor[dtype, Layout.row_major(BATCH, INPUT_DIM), MutAnyOrigin],
    y: LayoutTensor[dtype, Layout.row_major(BATCH, OUTPUT_DIM), MutAnyOrigin],
    rng: LayoutTensor[DType.uint32, Layout.row_major(BATCH), MutAnyOrigin],
):
    """Generate y = x1 * x2 data on GPU."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH:
        return
    var state = rebind[Scalar[DType.uint32]](rng[idx])
    state ^= state << 13
    state ^= state >> 17
    state ^= state << 5
    var x1 = (
        Scalar[dtype](state) / Scalar[dtype](Scalar[DType.uint32].MAX) * 2 - 1
    )
    state ^= state << 13
    state ^= state >> 17
    state ^= state << 5
    var x2 = (
        Scalar[dtype](state) / Scalar[dtype](Scalar[DType.uint32].MAX) * 2 - 1
    )
    x[idx, 0] = x1
    x[idx, 1] = x2
    y[idx, 0] = x1 * x2
    rng[idx] = state


fn xavier_init_kernel[
    SIZE: Int, FAN_IN: Int, FAN_OUT: Int
](
    weights: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    rng_seed: LayoutTensor[DType.uint32, Layout.row_major(1), MutAnyOrigin],
):
    """Xavier init on GPU."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SIZE:
        return
    var state = rebind[Scalar[DType.uint32]](rng_seed[0]) + UInt32(
        idx * 1099087573
    )
    state ^= state << 13
    state ^= state >> 17
    state ^= state << 5
    var u = (
        Scalar[dtype](state) / Scalar[dtype](Scalar[DType.uint32].MAX) * 2 - 1
    )
    var scale = sqrt(Scalar[dtype](6.0) / Scalar[dtype](FAN_IN + FAN_OUT))
    weights[idx] = u * scale


fn mse_loss_kernel[
    BATCH: Int, OUT_DIM: Int
](
    loss: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
    pred: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin],
    target: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin
    ],
):
    """Compute MSE loss."""
    var local_i = thread_idx.x
    var sum: pred.element_type = 0
    var idx = Int(local_i)
    while idx < BATCH * OUT_DIM:
        var row = idx // OUT_DIM
        var col = idx % OUT_DIM
        var diff = pred[row, col] - target[row, col]
        sum += diff * diff
        idx += TPB

    var shared = LayoutTensor[
        dtype,
        Layout.row_major(TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    shared[Int(local_i)] = sum
    barrier()
    var stride = TPB // 2
    while stride > 0:
        if Int(local_i) < stride:
            shared[Int(local_i)] += shared[Int(local_i) + stride]
        barrier()
        stride //= 2
    if local_i == 0:
        loss[0] = rebind[loss.element_type](
            rebind[Scalar[dtype]](shared[0]) / Scalar[dtype](BATCH * OUT_DIM)
        )


# =============================================================================
# Main
# =============================================================================


def main():
    seed(42)
    print("=" * 70)
    print("POC: GPU Kernel Fusion with Functional Layer Design")
    print("=" * 70)
    print()
    print(
        "Network: "
        + String(INPUT_DIM)
        + " -> "
        + String(HIDDEN_DIM)
        + " (ReLU) -> "
        + String(OUTPUT_DIM)
    )
    print("Batch: " + String(BATCH_SIZE) + ", Epochs: " + String(NUM_EPOCHS))
    print()

    with DeviceContext() as ctx:
        comptime W1_SIZE = INPUT_DIM * HIDDEN_DIM
        comptime B1_SIZE = HIDDEN_DIM
        comptime W2_SIZE = HIDDEN_DIM * OUTPUT_DIM
        comptime B2_SIZE = OUTPUT_DIM

        # Buffers
        var W1_buf = ctx.enqueue_create_buffer[dtype](W1_SIZE)
        var b1_buf = ctx.enqueue_create_buffer[dtype](B1_SIZE)
        var W2_buf = ctx.enqueue_create_buffer[dtype](W2_SIZE)
        var b2_buf = ctx.enqueue_create_buffer[dtype](B2_SIZE)
        var dW1_buf = ctx.enqueue_create_buffer[dtype](W1_SIZE)
        var db1_buf = ctx.enqueue_create_buffer[dtype](B1_SIZE)
        var dW2_buf = ctx.enqueue_create_buffer[dtype](W2_SIZE)
        var db2_buf = ctx.enqueue_create_buffer[dtype](B2_SIZE)
        var m_W1_buf = ctx.enqueue_create_buffer[dtype](W1_SIZE)
        var v_W1_buf = ctx.enqueue_create_buffer[dtype](W1_SIZE)
        var m_b1_buf = ctx.enqueue_create_buffer[dtype](B1_SIZE)
        var v_b1_buf = ctx.enqueue_create_buffer[dtype](B1_SIZE)
        var m_W2_buf = ctx.enqueue_create_buffer[dtype](W2_SIZE)
        var v_W2_buf = ctx.enqueue_create_buffer[dtype](W2_SIZE)
        var m_b2_buf = ctx.enqueue_create_buffer[dtype](B2_SIZE)
        var v_b2_buf = ctx.enqueue_create_buffer[dtype](B2_SIZE)
        var h1_pre_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_SIZE * HIDDEN_DIM
        )
        var h1_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * HIDDEN_DIM)
        var y_pred_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_SIZE * OUTPUT_DIM
        )
        var dy_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * OUTPUT_DIM)
        var x_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * INPUT_DIM)
        var y_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * OUTPUT_DIM)
        var loss_buf = ctx.enqueue_create_buffer[dtype](1)
        var rng_buf = ctx.enqueue_create_buffer[DType.uint32](BATCH_SIZE)
        var rng_seed_buf = ctx.enqueue_create_buffer[DType.uint32](1)

        # Init buffers to zero
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

        with rng_seed_buf.map_to_host() as host:
            host[0] = UInt32(12345)
        with rng_buf.map_to_host() as host:
            for i in range(BATCH_SIZE):
                host[i] = UInt32(i * 1099087573 + 42)

        var rng_t = LayoutTensor[
            DType.uint32, Layout.row_major(1), MutAnyOrigin
        ](rng_seed_buf)
        var W1_init = LayoutTensor[
            dtype, Layout.row_major(W1_SIZE), MutAnyOrigin
        ](W1_buf)
        var W2_init = LayoutTensor[
            dtype, Layout.row_major(W2_SIZE), MutAnyOrigin
        ](W2_buf)

        ctx.enqueue_function[
            xavier_init_kernel[W1_SIZE, INPUT_DIM, HIDDEN_DIM],
            xavier_init_kernel[W1_SIZE, INPUT_DIM, HIDDEN_DIM],
        ](
            W1_init,
            rng_t,
            grid_dim=((W1_SIZE + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )
        ctx.enqueue_function[
            xavier_init_kernel[W2_SIZE, HIDDEN_DIM, OUTPUT_DIM],
            xavier_init_kernel[W2_SIZE, HIDDEN_DIM, OUTPUT_DIM],
        ](
            W2_init,
            rng_t,
            grid_dim=((W2_SIZE + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )
        ctx.synchronize()
        print("Initialized weights")

        # Tensors
        var W1_t = LayoutTensor[
            dtype, Layout.row_major(INPUT_DIM, HIDDEN_DIM), MutAnyOrigin
        ](W1_buf)
        var b1_t = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM), MutAnyOrigin
        ](b1_buf)
        var W2_t = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, OUTPUT_DIM), MutAnyOrigin
        ](W2_buf)
        var b2_t = LayoutTensor[
            dtype, Layout.row_major(OUTPUT_DIM), MutAnyOrigin
        ](b2_buf)
        var dW1_t = LayoutTensor[
            dtype, Layout.row_major(INPUT_DIM, HIDDEN_DIM), MutAnyOrigin
        ](dW1_buf)
        var db1_t = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM), MutAnyOrigin
        ](db1_buf)
        var dW2_t = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, OUTPUT_DIM), MutAnyOrigin
        ](dW2_buf)
        var db2_t = LayoutTensor[
            dtype, Layout.row_major(OUTPUT_DIM), MutAnyOrigin
        ](db2_buf)
        var h1_pre_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, HIDDEN_DIM), MutAnyOrigin
        ](h1_pre_buf)
        var h1_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, HIDDEN_DIM), MutAnyOrigin
        ](h1_buf)
        var y_pred_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, OUTPUT_DIM), MutAnyOrigin
        ](y_pred_buf)
        var dy_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, OUTPUT_DIM), MutAnyOrigin
        ](dy_buf)
        var x_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, INPUT_DIM), MutAnyOrigin
        ](x_buf)
        var y_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, OUTPUT_DIM), MutAnyOrigin
        ](y_buf)
        var loss_t = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](
            loss_buf
        )
        var rng_t2 = LayoutTensor[
            DType.uint32, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](rng_buf)

        # Flat tensors for Adam
        var W1_flat = LayoutTensor[
            dtype, Layout.row_major(W1_SIZE), MutAnyOrigin
        ](W1_buf)
        var dW1_flat = LayoutTensor[
            dtype, Layout.row_major(W1_SIZE), ImmutAnyOrigin
        ](dW1_buf)
        var m_W1_t = LayoutTensor[
            dtype, Layout.row_major(W1_SIZE), MutAnyOrigin
        ](m_W1_buf)
        var v_W1_t = LayoutTensor[
            dtype, Layout.row_major(W1_SIZE), MutAnyOrigin
        ](v_W1_buf)
        var b1_flat = LayoutTensor[
            dtype, Layout.row_major(B1_SIZE), MutAnyOrigin
        ](b1_buf)
        var db1_flat = LayoutTensor[
            dtype, Layout.row_major(B1_SIZE), ImmutAnyOrigin
        ](db1_buf)
        var m_b1_t = LayoutTensor[
            dtype, Layout.row_major(B1_SIZE), MutAnyOrigin
        ](m_b1_buf)
        var v_b1_t = LayoutTensor[
            dtype, Layout.row_major(B1_SIZE), MutAnyOrigin
        ](v_b1_buf)
        var W2_flat = LayoutTensor[
            dtype, Layout.row_major(W2_SIZE), MutAnyOrigin
        ](W2_buf)
        var dW2_flat = LayoutTensor[
            dtype, Layout.row_major(W2_SIZE), ImmutAnyOrigin
        ](dW2_buf)
        var m_W2_t = LayoutTensor[
            dtype, Layout.row_major(W2_SIZE), MutAnyOrigin
        ](m_W2_buf)
        var v_W2_t = LayoutTensor[
            dtype, Layout.row_major(W2_SIZE), MutAnyOrigin
        ](v_W2_buf)
        var b2_flat = LayoutTensor[
            dtype, Layout.row_major(B2_SIZE), MutAnyOrigin
        ](b2_buf)
        var db2_flat = LayoutTensor[
            dtype, Layout.row_major(B2_SIZE), ImmutAnyOrigin
        ](db2_buf)
        var m_b2_t = LayoutTensor[
            dtype, Layout.row_major(B2_SIZE), MutAnyOrigin
        ](m_b2_buf)
        var v_b2_t = LayoutTensor[
            dtype, Layout.row_major(B2_SIZE), MutAnyOrigin
        ](v_b2_buf)

        # Grids
        comptime grid_h1 = (
            (HIDDEN_DIM + TILE - 1) // TILE,
            (BATCH_SIZE + TILE - 1) // TILE,
        )
        comptime grid_out = (
            (OUTPUT_DIM + TILE - 1) // TILE,
            (BATCH_SIZE + TILE - 1) // TILE,
        )
        comptime grid_dW2 = (
            (OUTPUT_DIM + TILE - 1) // TILE,
            (HIDDEN_DIM + TILE - 1) // TILE,
        )
        comptime grid_dW1 = (
            (HIDDEN_DIM + TILE - 1) // TILE,
            (INPUT_DIM + TILE - 1) // TILE,
        )
        comptime block_2d = (TILE, TILE)

        var lr = Scalar[dtype](0.001)
        var beta1 = Scalar[dtype](0.9)
        var beta2 = Scalar[dtype](0.999)
        var eps = Scalar[dtype](1e-8)

        print("Compiling kernels...")
        var fwd1_fn = ctx.compile_function_checked[
            forward_layer1_relu_kernel[BATCH_SIZE, INPUT_DIM, HIDDEN_DIM],
            forward_layer1_relu_kernel[BATCH_SIZE, INPUT_DIM, HIDDEN_DIM],
        ]()
        var fwd2_fn = ctx.compile_function_checked[
            forward_layer2_kernel[BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM],
            forward_layer2_kernel[BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM],
        ]()
        var bwd2_fn = ctx.compile_function_checked[
            backward_mse_dW_db_kernel[BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM],
            backward_mse_dW_db_kernel[BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM],
        ]()
        var bwd1_fn = ctx.compile_function_checked[
            backward_dx_relu_dW_db_kernel[
                BATCH_SIZE, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM
            ],
            backward_dx_relu_dW_db_kernel[
                BATCH_SIZE, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM
            ],
        ]()
        var adam_fn = ctx.compile_function_checked[
            adam_fused_kernel[W1_SIZE, B1_SIZE, W2_SIZE, B2_SIZE],
            adam_fused_kernel[W1_SIZE, B1_SIZE, W2_SIZE, B2_SIZE],
        ]()
        var loss_fn = ctx.compile_function_checked[
            mse_loss_kernel[BATCH_SIZE, OUTPUT_DIM],
            mse_loss_kernel[BATCH_SIZE, OUTPUT_DIM],
        ]()
        var data_fn = ctx.compile_function_checked[
            generate_data_kernel[BATCH_SIZE],
            generate_data_kernel[BATCH_SIZE],
        ]()
        print("Compiled!")
        print()

        ctx.enqueue_function(
            data_fn,
            x_t,
            y_t,
            rng_t2,
            grid_dim=((BATCH_SIZE + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )

        print("Training...")
        print("-" * 70)
        var start_time = perf_counter_ns()

        for epoch in range(NUM_EPOCHS):
            var log = (epoch + 1) % 100 == 0 or epoch == 0

            # Forward (2 kernels)
            ctx.enqueue_function(
                fwd1_fn,
                h1_pre_t,
                h1_t,
                x_t,
                W1_t,
                b1_t,
                grid_dim=grid_h1,
                block_dim=block_2d,
            )
            ctx.enqueue_function(
                fwd2_fn,
                y_pred_t,
                h1_t,
                W2_t,
                b2_t,
                grid_dim=grid_out,
                block_dim=block_2d,
            )

            # Backward (2 kernels)
            ctx.enqueue_function(
                bwd2_fn,
                dW2_t,
                db2_t,
                dy_t,
                h1_t,
                y_pred_t,
                y_t,
                grid_dim=grid_dW2,
                block_dim=block_2d,
            )
            ctx.enqueue_function(
                bwd1_fn,
                dW1_t,
                db1_t,
                x_t,
                h1_pre_t,
                dy_t,
                W2_t,
                grid_dim=grid_dW1,
                block_dim=block_2d,
            )

            # Adam (1 kernel)
            var t = Scalar[dtype](epoch + 1)
            var bc1 = 1 - beta1**t
            var bc2 = 1 - beta2**t
            comptime TOTAL = W1_SIZE + B1_SIZE + W2_SIZE + B2_SIZE
            ctx.enqueue_function(
                adam_fn,
                W1_flat,
                dW1_flat,
                m_W1_t,
                v_W1_t,
                b1_flat,
                db1_flat,
                m_b1_t,
                v_b1_t,
                W2_flat,
                dW2_flat,
                m_W2_t,
                v_W2_t,
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
                grid_dim=((TOTAL + TPB - 1) // TPB,),
                block_dim=(TPB,),
            )

            if log:
                ctx.enqueue_function(
                    loss_fn,
                    loss_t,
                    y_pred_t,
                    y_t,
                    grid_dim=(1,),
                    block_dim=(TPB,),
                )
                ctx.synchronize()
                with loss_buf.map_to_host() as host:
                    print(
                        "Epoch "
                        + String(epoch + 1)
                        + "/"
                        + String(NUM_EPOCHS)
                        + " - Loss: "
                        + String(Float32(host[0]))
                    )

        ctx.synchronize()
        var elapsed_ms = Float64(perf_counter_ns() - start_time) / 1e6

        print("-" * 70)
        print()
        print("Completed in " + String(elapsed_ms)[:8] + " ms")
        print(
            "Per epoch: " + String(elapsed_ms / Float64(NUM_EPOCHS))[:6] + " ms"
        )
        print(
            "Throughput: "
            + String(
                Int(Float64(NUM_EPOCHS * BATCH_SIZE) / (elapsed_ms / 1000.0))
            )
            + " samples/sec"
        )

        print()
        print("=" * 70)
        print("FUSION SUMMARY")
        print("=" * 70)
        print()
        print("5 kernel launches per training step:")
        print("  1. forward_layer1_relu: Linear + ReLU (FUSED)")
        print("  2. forward_layer2: Linear")
        print("  3. backward_mse_dW_db: MSE_grad + dW2 + db2 (FUSED)")
        print(
            "  4. backward_dx_relu_dW_db: dx + ReLU_backward + dW1 + db1"
            " (FUSED)"
        )
        print("  5. adam_fused: All 4 param groups (FUSED)")
        print()
        print(
            "Key: relu_scalar() and relu_backward_scalar() are @always_inline"
        )
        print(
            "     -> Compiled directly into kernel code, no function call"
            " overhead"
        )
        print()
        print(
            "This demonstrates the functional layer design IS GPU-compatible!"
        )
        print("=" * 70)

"""GPU MLP Test - Training a simple neural network entirely on GPU.

This demonstrates:
- Forward pass using tiled matrix multiplication
- Backward pass computing gradients
- Adam optimizer updates
- Training loop with loss tracking
- Pre-generated data on GPU for maximum performance
- Detailed timing breakdown

We train a 2-layer MLP on a simple regression task: y = x1*x2 (XOR-like)

Run with:
    pixi run -e apple mojo run test_mlp.mojo
"""

from time import perf_counter_ns
from math import sqrt, exp
from random import seed, random_float64

from layout import Layout, LayoutTensor
from gpu import thread_idx, block_idx, block_dim, barrier, block
from gpu.host import DeviceContext
from gpu.memory import AddressSpace

# =============================================================================
# Constants
# =============================================================================

comptime dtype = DType.float32
comptime TILE = 16  # Tile size for matmul
comptime TPB = 256  # Threads per block for elementwise ops

# Network architecture - LARGER batch size for better GPU utilization
comptime BATCH_SIZE = 1024
comptime INPUT_DIM = 2
comptime HIDDEN_DIM = 64
comptime OUTPUT_DIM = 1

# Training parameters
comptime NUM_EPOCHS = 1000


# =============================================================================
# GPU Data Generation Kernel
# =============================================================================


fn generate_data_kernel[
    BATCH: Int,
](
    x_data: LayoutTensor[
        dtype, Layout.row_major(BATCH, INPUT_DIM), MutAnyOrigin
    ],
    y_data: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUTPUT_DIM), MutAnyOrigin
    ],
    rng_seeds: LayoutTensor[
        DType.uint32, Layout.row_major(BATCH), MutAnyOrigin
    ],
):
    """Generate XOR-like data on GPU: y = x1 * x2."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH:
        return

    # xorshift PRNG
    var state = rebind[Scalar[DType.uint32]](rng_seeds[idx])

    # Generate x1
    state ^= state << 13
    state ^= state >> 17
    state ^= state << 5
    var x1 = (
        Scalar[dtype](state) / Scalar[dtype](Scalar[DType.uint32].MAX) * 2 - 1
    )

    # Generate x2
    state ^= state << 13
    state ^= state >> 17
    state ^= state << 5
    var x2 = (
        Scalar[dtype](state) / Scalar[dtype](Scalar[DType.uint32].MAX) * 2 - 1
    )

    # Store inputs and target
    x_data[idx, 0] = x1
    x_data[idx, 1] = x2
    y_data[idx, 0] = x1 * x2

    # Update RNG state for next use
    rng_seeds[idx] = state


# =============================================================================
# Weight Initialization Kernel
# =============================================================================


fn xavier_init_kernel[
    SIZE: Int,
    FAN_IN: Int,
    FAN_OUT: Int,
](
    weights: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    rng_seed: LayoutTensor[DType.uint32, Layout.row_major(1), MutAnyOrigin],
):
    """Xavier/Glorot initialization."""
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


# =============================================================================
# Forward Pass Kernels
# =============================================================================


fn linear_forward_relu_kernel[
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    x: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(OUT_DIM), ImmutAnyOrigin],
):
    """Fused forward pass with ReLU: y = max(0, x @ W + b)."""
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var global_row = Int(block_idx.y) * TILE + local_row
    var global_col = Int(block_idx.x) * TILE + local_col

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
    if global_col < OUT_DIM:
        acc = b[global_col]

    comptime num_tiles = (IN_DIM + TILE - 1) // TILE

    for tile_idx in range(num_tiles):
        var x_col = tile_idx * TILE + local_col
        if global_row < BATCH and x_col < IN_DIM:
            x_shared[local_row, local_col] = x[global_row, x_col]
        else:
            x_shared[local_row, local_col] = 0

        var W_row = tile_idx * TILE + local_row
        if W_row < IN_DIM and global_col < OUT_DIM:
            W_shared[local_row, local_col] = W[W_row, global_col]
        else:
            W_shared[local_row, local_col] = 0

        barrier()

        for k in range(TILE):
            acc += x_shared[local_row, k] * W_shared[k, local_col]

        barrier()

    if global_row < BATCH and global_col < OUT_DIM:
        output[global_row, global_col] = max(acc, 0)


fn linear_forward_kernel[
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    x: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(OUT_DIM), ImmutAnyOrigin],
):
    """Forward pass without activation: y = x @ W + b."""
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var global_row = Int(block_idx.y) * TILE + local_row
    var global_col = Int(block_idx.x) * TILE + local_col

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
    if global_col < OUT_DIM:
        acc = b[global_col]

    comptime num_tiles = (IN_DIM + TILE - 1) // TILE

    for tile_idx in range(num_tiles):
        var x_col = tile_idx * TILE + local_col
        if global_row < BATCH and x_col < IN_DIM:
            x_shared[local_row, local_col] = x[global_row, x_col]
        else:
            x_shared[local_row, local_col] = 0

        var W_row = tile_idx * TILE + local_row
        if W_row < IN_DIM and global_col < OUT_DIM:
            W_shared[local_row, local_col] = W[W_row, global_col]
        else:
            W_shared[local_row, local_col] = 0

        barrier()

        for k in range(TILE):
            acc += x_shared[local_row, k] * W_shared[k, local_col]

        barrier()

    if global_row < BATCH and global_col < OUT_DIM:
        output[global_row, global_col] = acc


# =============================================================================
# Backward Pass Kernels
# =============================================================================


fn mse_loss_backward_kernel[
    BATCH: Int,
    OUT_DIM: Int,
](
    d_output: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ],
    predictions: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin
    ],
    targets: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin
    ],
):
    """Compute gradient of MSE loss."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * OUT_DIM:
        return

    var row = idx // OUT_DIM
    var col = idx % OUT_DIM
    var pred = predictions[row, col]
    var target = targets[row, col]
    d_output[row, col] = 2.0 * (pred - target) / (BATCH * OUT_DIM)


fn linear_backward_dx_kernel[
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
](
    dx: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
    dy: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin],
    W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
):
    """Input gradient: dx = dy @ W.T."""
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var global_row = Int(block_idx.y) * TILE + local_row
    var global_col = Int(block_idx.x) * TILE + local_col

    var dy_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var W_T_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: dx.element_type = 0
    comptime num_tiles = (OUT_DIM + TILE - 1) // TILE

    for tile_idx in range(num_tiles):
        var dy_col = tile_idx * TILE + local_col
        if global_row < BATCH and dy_col < OUT_DIM:
            dy_shared[local_row, local_col] = dy[global_row, dy_col]
        else:
            dy_shared[local_row, local_col] = 0

        var W_col = tile_idx * TILE + local_row
        if W_col < OUT_DIM and global_col < IN_DIM:
            W_T_shared[local_row, local_col] = W[global_col, W_col]
        else:
            W_T_shared[local_row, local_col] = 0

        barrier()

        for k in range(TILE):
            acc += dy_shared[local_row, k] * W_T_shared[k, local_col]

        barrier()

    if global_row < BATCH and global_col < IN_DIM:
        dx[global_row, global_col] = acc


fn linear_backward_dW_kernel[
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
](
    dW: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), MutAnyOrigin],
    x: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    dy: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin],
):
    """Weight gradient: dW = x.T @ dy."""
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var global_row = Int(block_idx.y) * TILE + local_row
    var global_col = Int(block_idx.x) * TILE + local_col

    var x_T_shared = LayoutTensor[
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

    var acc: dW.element_type = 0
    comptime num_tiles = (BATCH + TILE - 1) // TILE

    for tile_idx in range(num_tiles):
        var batch_idx = tile_idx * TILE + local_col
        if global_row < IN_DIM and batch_idx < BATCH:
            x_T_shared[local_row, local_col] = x[batch_idx, global_row]
        else:
            x_T_shared[local_row, local_col] = 0

        var dy_row = tile_idx * TILE + local_row
        if dy_row < BATCH and global_col < OUT_DIM:
            dy_shared[local_row, local_col] = dy[dy_row, global_col]
        else:
            dy_shared[local_row, local_col] = 0

        barrier()

        for k in range(TILE):
            acc += x_T_shared[local_row, k] * dy_shared[k, local_col]

        barrier()

    if global_row < IN_DIM and global_col < OUT_DIM:
        dW[global_row, global_col] = acc


fn linear_backward_db_kernel[
    BATCH: Int,
    OUT_DIM: Int,
](
    db: LayoutTensor[dtype, Layout.row_major(OUT_DIM), MutAnyOrigin],
    dy: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin],
):
    """Bias gradient: db = sum(dy, axis=0)."""
    var col = Int(block_idx.x)
    var local_i = thread_idx.x

    if col >= OUT_DIM:
        return

    var my_value: dy.element_type = 0
    var batch_idx = Int(local_i)
    while batch_idx < BATCH:
        my_value += rebind[dy.element_type](dy[batch_idx, col])
        batch_idx += TPB

    var total = block.sum[block_size=TPB, broadcast=False](val=my_value)

    if local_i == 0:
        db[col] = total[0]


fn relu_backward_kernel[
    BATCH: Int,
    DIM: Int,
](
    dx: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    dy: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), ImmutAnyOrigin],
    x: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), ImmutAnyOrigin],
):
    """ReLU backward: dx = dy * (x > 0)."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * DIM:
        return

    var row = idx // DIM
    var col = idx % DIM
    var x_val = x[row, col]
    var dy_val = dy[row, col]
    dx[row, col] = dy_val if x_val > 0 else 0


# =============================================================================
# Adam Optimizer Kernel
# =============================================================================


fn adam_kernel[
    SIZE: Int,
](
    weights: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    grads: LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin],
    m: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    v: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    lr: Scalar[dtype],
    beta1: Scalar[dtype],
    beta2: Scalar[dtype],
    eps: Scalar[dtype],
    bias_correction1: Scalar[dtype],
    bias_correction2: Scalar[dtype],
):
    """Adam optimizer update."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SIZE:
        return

    var g = grads[idx]
    var m_val = m[idx]
    var v_val = v[idx]

    var m_new = beta1 * m_val + (1 - beta1) * g
    var v_new = beta2 * v_val + (1 - beta2) * g * g

    var m_hat = m_new / bias_correction1
    var v_hat = v_new / bias_correction2

    weights[idx] = weights[idx] - lr * m_hat / (sqrt(v_hat) + eps)

    m[idx] = m_new
    v[idx] = v_new


# =============================================================================
# Loss Computation Kernel
# =============================================================================


fn mse_loss_kernel[
    BATCH: Int,
    OUT_DIM: Int,
](
    loss: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin],
    predictions: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin
    ],
    targets: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin
    ],
):
    """Compute MSE loss using block reduction."""
    var local_i = thread_idx.x

    var my_value: predictions.element_type = 0
    var idx = Int(local_i)
    while idx < BATCH * OUT_DIM:
        var row = idx // OUT_DIM
        var col = idx % OUT_DIM
        var diff = predictions[row, col] - targets[row, col]
        my_value += diff * diff
        idx += TPB

    var total = block.sum[block_size=TPB, broadcast=False](val=my_value)

    if local_i == 0:
        loss[0] = total[0] / (BATCH * OUT_DIM)


# =============================================================================
# Timing Helper
# =============================================================================


struct TimingStats:
    var data_select_ns: Int
    var forward_ns: Int
    var loss_ns: Int
    var backward_ns: Int
    var adam_ns: Int
    var sync_ns: Int
    var count: Int

    fn __init__(out self):
        self.data_select_ns = 0
        self.forward_ns = 0
        self.loss_ns = 0
        self.backward_ns = 0
        self.adam_ns = 0
        self.sync_ns = 0
        self.count = 0

    fn print_stats(self):
        var total = (
            self.data_select_ns
            + self.forward_ns
            + self.loss_ns
            + self.backward_ns
            + self.adam_ns
            + self.sync_ns
        )
        print("\nTiming breakdown (average per epoch):")
        print(
            "  Data select:  "
            + String(
                Float64(self.data_select_ns) / Float64(self.count) / 1000.0
            )[:8]
            + " us"
        )
        print(
            "  Forward pass: "
            + String(Float64(self.forward_ns) / Float64(self.count) / 1000.0)[
                :8
            ]
            + " us"
        )
        print(
            "  Loss compute: "
            + String(Float64(self.loss_ns) / Float64(self.count) / 1000.0)[:8]
            + " us"
        )
        print(
            "  Backward:     "
            + String(Float64(self.backward_ns) / Float64(self.count) / 1000.0)[
                :8
            ]
            + " us"
        )
        print(
            "  Adam update:  "
            + String(Float64(self.adam_ns) / Float64(self.count) / 1000.0)[:8]
            + " us"
        )
        print(
            "  Sync/logging: "
            + String(Float64(self.sync_ns) / Float64(self.count) / 1000.0)[:8]
            + " us"
        )
        print(
            "  Total:        "
            + String(Float64(total) / Float64(self.count) / 1000.0)[:8]
            + " us"
        )


# =============================================================================
# Main Training Loop
# =============================================================================


def main():
    seed(42)
    print("=" * 70)
    print("GPU MLP Training Test - Optimized Version")
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
    print("Task: Learn y = x1 * x2 (product function)")
    print("Batch size: " + String(BATCH_SIZE))
    print("Data generation: On GPU each epoch (fast)")
    print()

    with DeviceContext() as ctx:
        # =====================================================================
        # Allocate GPU buffers
        # =====================================================================

        # Layer sizes
        comptime W1_SIZE = INPUT_DIM * HIDDEN_DIM
        comptime B1_SIZE = HIDDEN_DIM
        comptime W2_SIZE = HIDDEN_DIM * OUTPUT_DIM
        comptime B2_SIZE = OUTPUT_DIM

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

        # Activations and gradients (single batch)
        var h1_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * HIDDEN_DIM)
        var h1_pre_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_SIZE * HIDDEN_DIM
        )
        var y_pred_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_SIZE * OUTPUT_DIM
        )

        var d_y_pred_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_SIZE * OUTPUT_DIM
        )
        var d_h1_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * HIDDEN_DIM)
        var d_h1_pre_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_SIZE * HIDDEN_DIM
        )

        var loss_buf = ctx.enqueue_create_buffer[dtype](1)
        var rng_seed_buf = ctx.enqueue_create_buffer[DType.uint32](1)

        # Training data (single batch, regenerated each epoch on GPU)
        var x_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * INPUT_DIM)
        var y_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * OUTPUT_DIM)
        var data_rng_buf = ctx.enqueue_create_buffer[DType.uint32](BATCH_SIZE)

        # =====================================================================
        # Initialize weights
        # =====================================================================

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

        var rng_t = LayoutTensor[
            DType.uint32, Layout.row_major(1), MutAnyOrigin
        ](rng_seed_buf)

        # Xavier initialization for weights
        var W1_t_init = LayoutTensor[
            dtype, Layout.row_major(W1_SIZE), MutAnyOrigin
        ](W1_buf)
        comptime W1_init_blocks = (W1_SIZE + TPB - 1) // TPB
        ctx.enqueue_function_checked[
            xavier_init_kernel[W1_SIZE, INPUT_DIM, HIDDEN_DIM],
            xavier_init_kernel[W1_SIZE, INPUT_DIM, HIDDEN_DIM],
        ](W1_t_init, rng_t, grid_dim=(W1_init_blocks,), block_dim=(TPB,))

        var W2_t_init = LayoutTensor[
            dtype, Layout.row_major(W2_SIZE), MutAnyOrigin
        ](W2_buf)
        comptime W2_init_blocks = (W2_SIZE + TPB - 1) // TPB
        ctx.enqueue_function_checked[
            xavier_init_kernel[W2_SIZE, HIDDEN_DIM, OUTPUT_DIM],
            xavier_init_kernel[W2_SIZE, HIDDEN_DIM, OUTPUT_DIM],
        ](W2_t_init, rng_t, grid_dim=(W2_init_blocks,), block_dim=(TPB,))

        ctx.synchronize()
        print("Weights initialized")

        # Initialize RNG seeds for data generation
        with data_rng_buf.map_to_host() as host:
            for i in range(BATCH_SIZE):
                host[i] = UInt32(i * 1099087573 + 42)

        # =====================================================================
        # Create tensors for training
        # =====================================================================

        var W1_t = LayoutTensor[
            dtype, Layout.row_major(INPUT_DIM, HIDDEN_DIM), MutAnyOrigin
        ](W1_buf)
        var b1_t = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM), MutAnyOrigin
        ](b1_buf)
        var dW1_t = LayoutTensor[
            dtype, Layout.row_major(INPUT_DIM, HIDDEN_DIM), MutAnyOrigin
        ](dW1_buf)
        var db1_t = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM), MutAnyOrigin
        ](db1_buf)
        var m_W1_t = LayoutTensor[
            dtype, Layout.row_major(W1_SIZE), MutAnyOrigin
        ](m_W1_buf)
        var v_W1_t = LayoutTensor[
            dtype, Layout.row_major(W1_SIZE), MutAnyOrigin
        ](v_W1_buf)
        var m_b1_t = LayoutTensor[
            dtype, Layout.row_major(B1_SIZE), MutAnyOrigin
        ](m_b1_buf)
        var v_b1_t = LayoutTensor[
            dtype, Layout.row_major(B1_SIZE), MutAnyOrigin
        ](v_b1_buf)

        var W2_t = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, OUTPUT_DIM), MutAnyOrigin
        ](W2_buf)
        var b2_t = LayoutTensor[
            dtype, Layout.row_major(OUTPUT_DIM), MutAnyOrigin
        ](b2_buf)
        var dW2_t = LayoutTensor[
            dtype, Layout.row_major(HIDDEN_DIM, OUTPUT_DIM), MutAnyOrigin
        ](dW2_buf)
        var db2_t = LayoutTensor[
            dtype, Layout.row_major(OUTPUT_DIM), MutAnyOrigin
        ](db2_buf)
        var m_W2_t = LayoutTensor[
            dtype, Layout.row_major(W2_SIZE), MutAnyOrigin
        ](m_W2_buf)
        var v_W2_t = LayoutTensor[
            dtype, Layout.row_major(W2_SIZE), MutAnyOrigin
        ](v_W2_buf)
        var m_b2_t = LayoutTensor[
            dtype, Layout.row_major(B2_SIZE), MutAnyOrigin
        ](m_b2_buf)
        var v_b2_t = LayoutTensor[
            dtype, Layout.row_major(B2_SIZE), MutAnyOrigin
        ](v_b2_buf)

        var h1_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, HIDDEN_DIM), MutAnyOrigin
        ](h1_buf)
        var h1_pre_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, HIDDEN_DIM), MutAnyOrigin
        ](h1_pre_buf)
        var y_pred_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, OUTPUT_DIM), MutAnyOrigin
        ](y_pred_buf)

        var d_y_pred_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, OUTPUT_DIM), MutAnyOrigin
        ](d_y_pred_buf)
        var d_h1_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, HIDDEN_DIM), MutAnyOrigin
        ](d_h1_buf)
        var d_h1_pre_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, HIDDEN_DIM), MutAnyOrigin
        ](d_h1_pre_buf)

        var loss_t = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](
            loss_buf
        )

        var x_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, INPUT_DIM), MutAnyOrigin
        ](x_buf)
        var y_t = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, OUTPUT_DIM), MutAnyOrigin
        ](y_buf)
        var data_rng_t = LayoutTensor[
            DType.uint32, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ](data_rng_buf)

        # =====================================================================
        # Training hyperparameters
        # =====================================================================

        var lr = Scalar[dtype](0.001)  # Try much smaller
        var beta1 = Scalar[dtype](0.9)
        var beta2 = Scalar[dtype](0.999)
        var eps = Scalar[dtype](1e-8)

        var print_every = 100

        # Grid/block dimensions
        comptime grid_h1 = (
            (BATCH_SIZE + TILE - 1) // TILE,
            (HIDDEN_DIM + TILE - 1) // TILE,
        )
        comptime grid_out = (
            (BATCH_SIZE + TILE - 1) // TILE,
            (OUTPUT_DIM + TILE - 1) // TILE,
        )
        comptime grid_dW1 = (
            (INPUT_DIM + TILE - 1) // TILE,
            (HIDDEN_DIM + TILE - 1) // TILE,
        )
        comptime grid_dW2 = (
            (HIDDEN_DIM + TILE - 1) // TILE,
            (OUTPUT_DIM + TILE - 1) // TILE,
        )
        comptime grid_dx_h1 = (
            (BATCH_SIZE + TILE - 1) // TILE,
            (HIDDEN_DIM + TILE - 1) // TILE,
        )
        comptime block_2d = (TILE, TILE)

        # =====================================================================
        # Pre-compile kernels
        # =====================================================================

        print()
        print("Compiling kernels...")

        var linear_forward_layer1 = ctx.compile_function_checked[
            linear_forward_kernel[BATCH_SIZE, INPUT_DIM, HIDDEN_DIM],
            linear_forward_kernel[BATCH_SIZE, INPUT_DIM, HIDDEN_DIM],
        ]()
        var linear_forward_relu_layer1 = ctx.compile_function_checked[
            linear_forward_relu_kernel[BATCH_SIZE, INPUT_DIM, HIDDEN_DIM],
            linear_forward_relu_kernel[BATCH_SIZE, INPUT_DIM, HIDDEN_DIM],
        ]()
        var linear_forward_layer2 = ctx.compile_function_checked[
            linear_forward_kernel[BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM],
            linear_forward_kernel[BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM],
        ]()
        var mse_loss_fn = ctx.compile_function_checked[
            mse_loss_kernel[BATCH_SIZE, OUTPUT_DIM],
            mse_loss_kernel[BATCH_SIZE, OUTPUT_DIM],
        ]()
        var mse_loss_backward_fn = ctx.compile_function_checked[
            mse_loss_backward_kernel[BATCH_SIZE, OUTPUT_DIM],
            mse_loss_backward_kernel[BATCH_SIZE, OUTPUT_DIM],
        ]()
        var backward_dW2_fn = ctx.compile_function_checked[
            linear_backward_dW_kernel[BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM],
            linear_backward_dW_kernel[BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM],
        ]()
        var backward_db2_fn = ctx.compile_function_checked[
            linear_backward_db_kernel[BATCH_SIZE, OUTPUT_DIM],
            linear_backward_db_kernel[BATCH_SIZE, OUTPUT_DIM],
        ]()
        var backward_dx_h1_fn = ctx.compile_function_checked[
            linear_backward_dx_kernel[BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM],
            linear_backward_dx_kernel[BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM],
        ]()
        var relu_backward_fn = ctx.compile_function_checked[
            relu_backward_kernel[BATCH_SIZE, HIDDEN_DIM],
            relu_backward_kernel[BATCH_SIZE, HIDDEN_DIM],
        ]()
        var backward_dW1_fn = ctx.compile_function_checked[
            linear_backward_dW_kernel[BATCH_SIZE, INPUT_DIM, HIDDEN_DIM],
            linear_backward_dW_kernel[BATCH_SIZE, INPUT_DIM, HIDDEN_DIM],
        ]()
        var backward_db1_fn = ctx.compile_function_checked[
            linear_backward_db_kernel[BATCH_SIZE, HIDDEN_DIM],
            linear_backward_db_kernel[BATCH_SIZE, HIDDEN_DIM],
        ]()
        var adam_W1_fn = ctx.compile_function_checked[
            adam_kernel[W1_SIZE], adam_kernel[W1_SIZE]
        ]()
        var adam_b1_fn = ctx.compile_function_checked[
            adam_kernel[B1_SIZE], adam_kernel[B1_SIZE]
        ]()
        var adam_W2_fn = ctx.compile_function_checked[
            adam_kernel[W2_SIZE], adam_kernel[W2_SIZE]
        ]()
        var adam_b2_fn = ctx.compile_function_checked[
            adam_kernel[B2_SIZE], adam_kernel[B2_SIZE]
        ]()

        print("Kernels compiled!")
        print()
        print("Training...")
        print("-" * 70)

        var stats = TimingStats()
        var start_time = perf_counter_ns()

        # Pre-compile data generation kernel
        comptime data_gen_blocks = (BATCH_SIZE + TPB - 1) // TPB
        var generate_data_fn = ctx.compile_function_checked[
            generate_data_kernel[BATCH_SIZE],
            generate_data_kernel[BATCH_SIZE],
        ]()

        for epoch in range(NUM_EPOCHS):
            var t0 = perf_counter_ns()

            # Generate data - but reset RNG seeds each epoch for SAME data
            # (Re-initialize RNG to get consistent data)
            if epoch == 0:
                ctx.enqueue_function_checked(
                    generate_data_fn,
                    x_t,
                    y_t,
                    data_rng_t,
                    grid_dim=(data_gen_blocks,),
                    block_dim=(TPB,),
                )
            # After first epoch, data stays the same (we don't regenerate)

            var t1 = perf_counter_ns()
            stats.data_select_ns += Int(t1 - t0)

            # =================================================================
            # Forward pass
            # =================================================================

            ctx.enqueue_function_checked(
                linear_forward_layer1,
                h1_pre_t,
                x_t,
                W1_t,
                b1_t,
                grid_dim=grid_h1,
                block_dim=block_2d,
            )
            ctx.enqueue_function_checked(
                linear_forward_relu_layer1,
                h1_t,
                x_t,
                W1_t,
                b1_t,
                grid_dim=grid_h1,
                block_dim=block_2d,
            )
            ctx.enqueue_function_checked(
                linear_forward_layer2,
                y_pred_t,
                h1_t,
                W2_t,
                b2_t,
                grid_dim=grid_out,
                block_dim=block_2d,
            )

            var t2 = perf_counter_ns()
            stats.forward_ns += Int(t2 - t1)

            # =================================================================
            # Compute loss
            # =================================================================

            ctx.enqueue_function_checked(
                mse_loss_fn,
                loss_t,
                y_pred_t,
                y_t,
                grid_dim=(1,),
                block_dim=(TPB,),
            )

            var t3 = perf_counter_ns()
            stats.loss_ns += Int(t3 - t2)

            # =================================================================
            # Backward pass
            # =================================================================

            comptime loss_grad_blocks = (
                BATCH_SIZE * OUTPUT_DIM + TPB - 1
            ) // TPB
            ctx.enqueue_function_checked(
                mse_loss_backward_fn,
                d_y_pred_t,
                y_pred_t,
                y_t,
                grid_dim=(loss_grad_blocks,),
                block_dim=(TPB,),
            )

            ctx.enqueue_function_checked(
                backward_dW2_fn,
                dW2_t,
                h1_t,
                d_y_pred_t,
                grid_dim=grid_dW2,
                block_dim=block_2d,
            )
            ctx.enqueue_function_checked(
                backward_db2_fn,
                db2_t,
                d_y_pred_t,
                grid_dim=(OUTPUT_DIM,),
                block_dim=(TPB,),
            )
            ctx.enqueue_function_checked(
                backward_dx_h1_fn,
                d_h1_t,
                d_y_pred_t,
                W2_t,
                grid_dim=grid_dx_h1,
                block_dim=block_2d,
            )

            comptime relu_grad_blocks = (
                BATCH_SIZE * HIDDEN_DIM + TPB - 1
            ) // TPB
            ctx.enqueue_function_checked(
                relu_backward_fn,
                d_h1_pre_t,
                d_h1_t,
                h1_pre_t,
                grid_dim=(relu_grad_blocks,),
                block_dim=(TPB,),
            )

            ctx.enqueue_function_checked(
                backward_dW1_fn,
                dW1_t,
                x_t,
                d_h1_pre_t,
                grid_dim=grid_dW1,
                block_dim=block_2d,
            )
            ctx.enqueue_function_checked(
                backward_db1_fn,
                db1_t,
                d_h1_pre_t,
                grid_dim=(HIDDEN_DIM,),
                block_dim=(TPB,),
            )

            var t4 = perf_counter_ns()
            stats.backward_ns += Int(t4 - t3)

            # =================================================================
            # Debug: Check gradients on first epoch
            # =================================================================

            # Save weights before update on first epoch
            var w1_before: Float32 = 0
            var w2_before: Float32 = 0
            if epoch == 0:
                ctx.synchronize()
                print("\n=== DEBUG: Gradient check ===")
                with d_y_pred_buf.map_to_host() as host:
                    var sum_abs: Float32 = 0
                    for i in range(min(BATCH_SIZE * OUTPUT_DIM, 10)):
                        sum_abs += abs(Float32(host[i]))
                    print("d_y_pred (first 10 sum): " + String(sum_abs))
                with dW2_buf.map_to_host() as host:
                    var sum_abs: Float32 = 0
                    for i in range(min(W2_SIZE, 10)):
                        sum_abs += abs(Float32(host[i]))
                    print("dW2 (first 10 sum): " + String(sum_abs))
                with dW1_buf.map_to_host() as host:
                    var sum_abs: Float32 = 0
                    for i in range(min(W1_SIZE, 10)):
                        sum_abs += abs(Float32(host[i]))
                    print("dW1 (first 10 sum): " + String(sum_abs))
                # Save W1[0] before update
                with W1_buf.map_to_host() as host:
                    w1_before = Float32(host[0])
                    print("W1[0] BEFORE update: " + String(w1_before))
                with W2_buf.map_to_host() as host:
                    w2_before = Float32(host[0])
                    print("W2[0] BEFORE update: " + String(w2_before))

            # =================================================================
            # Adam updates
            # =================================================================

            var t = Scalar[dtype](epoch + 1)
            var bc1 = Scalar[dtype](1) - beta1**t
            var bc2 = Scalar[dtype](1) - beta2**t

            comptime W1_blocks = (W1_SIZE + TPB - 1) // TPB
            var dW1_flat = LayoutTensor[
                dtype, Layout.row_major(W1_SIZE), ImmutAnyOrigin
            ](dW1_buf)
            var W1_flat = LayoutTensor[
                dtype, Layout.row_major(W1_SIZE), MutAnyOrigin
            ](W1_buf)
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
                grid_dim=(W1_blocks,),
                block_dim=(TPB,),
            )

            comptime b1_blocks = (B1_SIZE + TPB - 1) // TPB
            var db1_flat = LayoutTensor[
                dtype, Layout.row_major(B1_SIZE), ImmutAnyOrigin
            ](db1_buf)
            var b1_flat = LayoutTensor[
                dtype, Layout.row_major(B1_SIZE), MutAnyOrigin
            ](b1_buf)
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
                grid_dim=(b1_blocks,),
                block_dim=(TPB,),
            )

            comptime W2_blocks = (W2_SIZE + TPB - 1) // TPB
            var dW2_flat = LayoutTensor[
                dtype, Layout.row_major(W2_SIZE), ImmutAnyOrigin
            ](dW2_buf)
            var W2_flat = LayoutTensor[
                dtype, Layout.row_major(W2_SIZE), MutAnyOrigin
            ](W2_buf)
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
                grid_dim=(W2_blocks,),
                block_dim=(TPB,),
            )

            comptime b2_blocks = (B2_SIZE + TPB - 1) // TPB
            var db2_flat = LayoutTensor[
                dtype, Layout.row_major(B2_SIZE), ImmutAnyOrigin
            ](db2_buf)
            var b2_flat = LayoutTensor[
                dtype, Layout.row_major(B2_SIZE), MutAnyOrigin
            ](b2_buf)
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
                grid_dim=(b2_blocks,),
                block_dim=(TPB,),
            )

            var t5 = perf_counter_ns()
            stats.adam_ns += Int(t5 - t4)

            # Check weights after update on first epoch
            if epoch == 0:
                ctx.synchronize()
                with W1_buf.map_to_host() as host:
                    var w1_after = Float32(host[0])
                    print("W1[0] AFTER update: " + String(w1_after))
                    print("W1[0] CHANGE: " + String(w1_after - w1_before))
                with W2_buf.map_to_host() as host:
                    var w2_after = Float32(host[0])
                    print("W2[0] AFTER update: " + String(w2_after))
                    print("W2[0] CHANGE: " + String(w2_after - w2_before))
                print("=== END DEBUG ===\n")

            # =================================================================
            # Print progress (with sync)
            # =================================================================

            if (epoch + 1) % print_every == 0 or epoch == 0:
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

            var t6 = perf_counter_ns()
            stats.sync_ns += Int(t6 - t5)
            stats.count += 1

        var end_time = perf_counter_ns()
        var elapsed_ms = Float64(end_time - start_time) / 1e6

        ctx.synchronize()

        print("-" * 70)
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

        stats.print_stats()

        # =====================================================================
        # Final evaluation
        # =====================================================================

        print()
        print("=" * 70)
        print("Final Evaluation")
        print("=" * 70)

        # Generate fresh test data
        ctx.enqueue_function_checked(
            generate_data_fn,
            x_t,
            y_t,
            data_rng_t,
            grid_dim=(data_gen_blocks,),
            block_dim=(TPB,),
        )

        ctx.enqueue_function_checked(
            linear_forward_relu_layer1,
            h1_t,
            x_t,
            W1_t,
            b1_t,
            grid_dim=grid_h1,
            block_dim=block_2d,
        )
        ctx.enqueue_function_checked(
            linear_forward_layer2,
            y_pred_t,
            h1_t,
            W2_t,
            b2_t,
            grid_dim=grid_out,
            block_dim=block_2d,
        )
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
        print("=" * 70)

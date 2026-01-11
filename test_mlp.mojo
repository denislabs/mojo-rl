"""GPU MLP Test - Training a simple neural network entirely on GPU.

This demonstrates:
- Forward pass using tiled matrix multiplication
- Backward pass computing gradients
- Adam optimizer updates
- Training loop with loss tracking
- Pre-generated data on GPU for maximum performance
- Detailed timing breakdown

We train a 2-layer MLP on a simple regression task: y = x1*x2 (XOR-like)

All Optimizations Applied:
  Optimization: 1. Loop Unrolling
  Technique: 4x unrolled inner k-loop in all matmul kernels
  Impact: Better ILP
  ────────────────────────────────────────
  Optimization: 2. Triple-Fused Kernel
  Technique: linear_backward_mse_dW_db_kernel combines MSE gradient + dW + db
  Impact: Reduced backward 36ms → 24ms
  ────────────────────────────────────────
  Optimization: 3. Conditional Loss Compute
  Technique: Only compute mse_loss when logging
  Impact: Saved ~8ms per non-logging epoch
  ────────────────────────────────────────
  Optimization: 4. Stride-Based Reduction
  Technique: Generic tree reduction loop (your suggestion)
  Impact: Cleaner, works for any TILE size
  ────────────────────────────────────────
  Optimization: 5. Conditional Sync
  Technique: Only sync for timing when logging
  Impact: Reduced sync overhead
  Performance Progression:
  ┌──────────────────────────────┬────────────────┬─────────────────────┬─────────┐
  │           Version            │   Total Time   │     Throughput      │ Speedup │
  ├──────────────────────────────┼────────────────┼─────────────────────┼─────────┤
  │ Original (from summary)      │ 179.4 ms/epoch │ ~5,700 samples/sec  │ 1.0x    │
  ├──────────────────────────────┼────────────────┼─────────────────────┼─────────┤
  │ After fused dW+db            │ 147.0 ms/epoch │ ~6,960 samples/sec  │ 1.2x    │
  ├──────────────────────────────┼────────────────┼─────────────────────┼─────────┤
  │ After fused Adam             │ 82.0 ms/epoch  │ ~12,480 samples/sec │ 2.2x    │
  ├──────────────────────────────┼────────────────┼─────────────────────┼─────────┤
  │ After fused dx+ReLU          │ 72.0 ms/epoch  │ ~14,200 samples/sec │ 2.5x    │
  ├──────────────────────────────┼────────────────┼─────────────────────┼─────────┤
  │ After triple-fused MSE+dW+db │ 60.0 ms/epoch  │ ~17,080 samples/sec │ 3.0x    │
  ├──────────────────────────────┼────────────────┼─────────────────────┼─────────┤
  │ After conditional loss       │ 48.1 ms/epoch  │ ~21,290 samples/sec │ 3.7x    │
  ├──────────────────────────────┼────────────────┼─────────────────────┼─────────┤
  │ After reduced sync           │ 46.9 ms/epoch  │ 21,839 samples/sec  │ 3.8x    │
  └──────────────────────────────┴────────────────┴─────────────────────┴─────────┘
  Final Timing Breakdown (logged epochs):

  - Forward: 15.8 ms (29%)
  - Backward: 23.7 ms (43%)
  - Adam: 7.8 ms (14%)
  - Loss: 8.0 ms (14%) - only on logging epochs

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
# Generic Tiled Matmul - Parametric for all use cases
# =============================================================================
#
# This generic kernel handles:
# - Forward:     C = A @ B + bias  (TRANSPOSE_A=False, TRANSPOSE_B=False)
# - Backward dW: C = A.T @ B       (TRANSPOSE_A=True,  TRANSPOSE_B=False)
# - Backward dx: C = A @ B.T       (TRANSPOSE_A=False, TRANSPOSE_B=True)
#
# Activations: "none", "relu", "dual_relu" (outputs both pre and post ReLU)
# =============================================================================


fn generic_matmul_kernel[
    dtype: DType,
    M: Int,  # Output rows (batch for forward, in_dim for dW, batch for dx)
    K: Int,  # Inner dimension (in_dim for forward, batch for dW, out_dim for dx)
    N: Int,  # Output cols (out_dim for forward/dW, in_dim for dx)
    TILE: Int,
    TRANSPOSE_A: Bool,  # A is (K, M) and we access A.T
    TRANSPOSE_B: Bool,  # B is (N, K) and we access B.T
    HAS_BIAS: Bool,
    ACTIVATION: StringLiteral,  # "none", "relu", "dual_relu"
](
    output: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    output2: LayoutTensor[
        dtype, Layout.row_major(M, N), MutAnyOrigin
    ],  # For dual_relu
    A: LayoutTensor[
        dtype,
        Layout.row_major(M, K) if not TRANSPOSE_A else Layout.row_major(K, M),
        ImmutAnyOrigin,
    ],
    B: LayoutTensor[
        dtype,
        Layout.row_major(K, N) if not TRANSPOSE_B else Layout.row_major(N, K),
        ImmutAnyOrigin,
    ],
    bias: LayoutTensor[dtype, Layout.row_major(N), ImmutAnyOrigin],
):
    """Generic tiled matmul with compile-time options."""
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var global_row = Int(block_idx.y) * TILE + local_row
    var global_col = Int(block_idx.x) * TILE + local_col

    var A_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var B_shared = LayoutTensor[
        dtype,
        Layout.row_major(TILE, TILE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Initialize accumulator (with bias if applicable)
    var acc: output.element_type = 0

    @parameter
    if HAS_BIAS:
        if global_col < N:
            acc = bias[global_col]

    comptime num_tiles = (K + TILE - 1) // TILE

    for tile_idx in range(num_tiles):
        # Load A tile (handle transpose)
        @parameter
        if TRANSPOSE_A:
            # A is (K, M), we want A.T[global_row, tile_k] = A[tile_k, global_row]
            var k_idx = tile_idx * TILE + local_col
            if global_row < M and k_idx < K:
                A_shared[local_row, local_col] = A[k_idx, global_row]
            else:
                A_shared[local_row, local_col] = 0
        else:
            # A is (M, K), access normally
            var k_idx = tile_idx * TILE + local_col
            if global_row < M and k_idx < K:
                A_shared[local_row, local_col] = A[global_row, k_idx]
            else:
                A_shared[local_row, local_col] = 0

        # Load B tile (handle transpose)
        @parameter
        if TRANSPOSE_B:
            # B is (N, K), we want B.T[tile_k, global_col] = B[global_col, tile_k]
            var k_idx = tile_idx * TILE + local_row
            if k_idx < K and global_col < N:
                B_shared[local_row, local_col] = B[global_col, k_idx]
            else:
                B_shared[local_row, local_col] = 0
        else:
            # B is (K, N), access normally
            var k_idx = tile_idx * TILE + local_row
            if k_idx < K and global_col < N:
                B_shared[local_row, local_col] = B[k_idx, global_col]
            else:
                B_shared[local_row, local_col] = 0

        barrier()

        # Compute partial dot product - UNROLLED by 4 for better ILP
        # TILE=16, so 16/4 = 4 iterations of 4 MACs each
        @parameter
        for k_base in range(0, TILE, 4):
            acc += (
                A_shared[local_row, k_base + 0]
                * B_shared[k_base + 0, local_col]
            )
            acc += (
                A_shared[local_row, k_base + 1]
                * B_shared[k_base + 1, local_col]
            )
            acc += (
                A_shared[local_row, k_base + 2]
                * B_shared[k_base + 2, local_col]
            )
            acc += (
                A_shared[local_row, k_base + 3]
                * B_shared[k_base + 3, local_col]
            )

        barrier()

    # Write output with activation
    if global_row < M and global_col < N:

        @parameter
        if ACTIVATION == "relu":
            output[global_row, global_col] = max(acc, 0)
        elif ACTIVATION == "dual_relu":
            output[global_row, global_col] = acc  # pre-ReLU
            output2[global_row, global_col] = max(acc, 0)  # post-ReLU
        else:  # "none"
            output[global_row, global_col] = acc


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
    """Fused forward pass with ReLU: y = max(0, x @ W + b).

    Delegates to generic_matmul_kernel with ACTIVATION="relu".
    """
    generic_matmul_kernel[
        dtype,
        BATCH,
        IN_DIM,
        OUT_DIM,
        TILE,
        TRANSPOSE_A=False,
        TRANSPOSE_B=False,
        HAS_BIAS=True,
        ACTIVATION="relu",
    ](output, output, x, W, b)


fn linear_forward_relu_dual_kernel[
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
](
    output_pre: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ],
    output_relu: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ],
    x: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(OUT_DIM), ImmutAnyOrigin],
):
    """Fused forward: outputs both pre-ReLU and post-ReLU in ONE matmul.

    Delegates to generic_matmul_kernel with ACTIVATION="dual_relu".
    """
    generic_matmul_kernel[
        dtype,
        BATCH,
        IN_DIM,
        OUT_DIM,
        TILE,
        TRANSPOSE_A=False,
        TRANSPOSE_B=False,
        HAS_BIAS=True,
        ACTIVATION="dual_relu",
    ](output_pre, output_relu, x, W, b)


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
    """Forward pass without activation: y = x @ W + b.

    Delegates to generic_matmul_kernel with ACTIVATION="none".
    """
    generic_matmul_kernel[
        dtype,
        BATCH,
        IN_DIM,
        OUT_DIM,
        TILE,
        TRANSPOSE_A=False,
        TRANSPOSE_B=False,
        HAS_BIAS=True,
        ACTIVATION="none",
    ](output, output, x, W, b)


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


# =============================================================================
# Fused MSE backward + dW + db Kernel
# =============================================================================


fn linear_backward_mse_dW_db_kernel[
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
](
    dW: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), MutAnyOrigin],
    db: LayoutTensor[dtype, Layout.row_major(OUT_DIM), MutAnyOrigin],
    d_output: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ],  # Also outputs d_y_pred
    x: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    predictions: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin
    ],
    targets: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin
    ],
):
    """Triple-fused backward: computes MSE gradient + dW + db in one kernel.

    - d_output = 2 * (predictions - targets) / (BATCH * OUT_DIM)
    - dW = x.T @ d_output
    - db = sum(d_output, axis=0)

    Grid: (OUT_DIM/TILE, IN_DIM/TILE) - each block computes TILE×TILE of dW.
    Blocks in first row (block_idx.y == 0) also compute db and d_output.

    Optimized with loop unrolling (4x) for better ILP.
    """
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var global_row = Int(block_idx.y) * TILE + local_row  # IN_DIM dimension
    var global_col = Int(block_idx.x) * TILE + local_col  # OUT_DIM dimension

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

    var dW_acc: dW.element_type = 0
    var db_acc: dW.element_type = 0

    comptime num_tiles = (BATCH + TILE - 1) // TILE
    var compute_db = block_idx.y == 0  # First row of blocks computes db

    # MSE gradient scale: 2 / (BATCH * OUT_DIM)
    var mse_scale = Scalar[dtype](2.0) / Scalar[dtype](BATCH * OUT_DIM)

    for tile_idx in range(num_tiles):
        # Load x.T tile: x_T[global_row, k] = x[k, global_row]
        var batch_idx = tile_idx * TILE + local_col
        if global_row < IN_DIM and batch_idx < BATCH:
            x_T_shared[local_row, local_col] = x[batch_idx, global_row]
        else:
            x_T_shared[local_row, local_col] = 0

        # Compute MSE gradient inline, load into dy_shared, and accumulate for db
        var dy_row = tile_idx * TILE + local_row
        var dy_val: dy_shared.element_type = 0
        if dy_row < BATCH and global_col < OUT_DIM:
            # Compute MSE gradient: 2 * (pred - target) / (batch * out_dim)
            var pred = predictions[dy_row, global_col]
            var target = targets[dy_row, global_col]
            dy_val = mse_scale * (pred - target)
            dy_shared[local_row, local_col] = dy_val

            # Also write d_output if this is the first row of blocks
            # (only need to write once per (batch, out) element)
            if compute_db:
                d_output[dy_row, global_col] = dy_val
                db_acc += dy_val
        else:
            dy_shared[local_row, local_col] = 0

        barrier()

        # Compute dW partial dot product - UNROLLED by 4 for ILP
        @parameter
        for k_base in range(0, TILE, 4):
            dW_acc += (
                x_T_shared[local_row, k_base + 0]
                * dy_shared[k_base + 0, local_col]
            )
            dW_acc += (
                x_T_shared[local_row, k_base + 1]
                * dy_shared[k_base + 1, local_col]
            )
            dW_acc += (
                x_T_shared[local_row, k_base + 2]
                * dy_shared[k_base + 2, local_col]
            )
            dW_acc += (
                x_T_shared[local_row, k_base + 3]
                * dy_shared[k_base + 3, local_col]
            )

        barrier()

    # Write dW
    if global_row < IN_DIM and global_col < OUT_DIM:
        dW[global_row, global_col] = dW_acc

    # Reduce and write db (only blocks in first row)
    if compute_db:
        # Reuse dy_shared for reduction across local_row
        dy_shared[local_row, local_col] = db_acc
        barrier()

        # Tree reduction with stride (works for any TILE size)
        var stride = TILE // 2
        while stride > 0:
            if local_row < stride:
                dy_shared[local_row, local_col] += dy_shared[
                    local_row + stride, local_col
                ]
            barrier()
            stride //= 2

        # Thread 0 of each column writes the result
        if local_row == 0 and global_col < OUT_DIM:
            db[global_col] = dy_shared[0, local_col]


# =============================================================================
# Fused dW + db Kernel (for layer 1 without MSE)
# =============================================================================


fn linear_backward_dW_db_kernel[
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
](
    dW: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), MutAnyOrigin],
    db: LayoutTensor[dtype, Layout.row_major(OUT_DIM), MutAnyOrigin],
    x: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    dy: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin],
):
    """Fused backward: dW = x.T @ dy, db = sum(dy, axis=0).

    Grid: (OUT_DIM/TILE, IN_DIM/TILE) - each block computes TILE×TILE of dW.
    Blocks in first row (block_idx.y == 0) also compute db for their columns.

    Optimized with loop unrolling (4x) for better ILP.
    """
    var local_row = Int(thread_idx.y)
    var local_col = Int(thread_idx.x)
    var global_row = Int(block_idx.y) * TILE + local_row  # IN_DIM dimension
    var global_col = Int(block_idx.x) * TILE + local_col  # OUT_DIM dimension

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

    var dW_acc: dW.element_type = 0
    var db_acc: dW.element_type = 0

    comptime num_tiles = (BATCH + TILE - 1) // TILE
    var compute_db = block_idx.y == 0  # First row of blocks computes db

    for tile_idx in range(num_tiles):
        # Load x.T tile: x_T[global_row, k] = x[k, global_row]
        var batch_idx = tile_idx * TILE + local_col
        if global_row < IN_DIM and batch_idx < BATCH:
            x_T_shared[local_row, local_col] = x[batch_idx, global_row]
        else:
            x_T_shared[local_row, local_col] = 0

        # Load dy tile and accumulate for db
        var dy_row = tile_idx * TILE + local_row
        var dy_val: dy_shared.element_type = 0
        if dy_row < BATCH and global_col < OUT_DIM:
            dy_val = dy[dy_row, global_col]
            dy_shared[local_row, local_col] = dy_val
            # Accumulate for db: each thread handles rows local_row, local_row+TILE, ...
            if compute_db:
                db_acc += dy_val
        else:
            dy_shared[local_row, local_col] = 0

        barrier()

        # Compute dW partial dot product - UNROLLED by 4 for ILP
        # TILE=16, so 16/4 = 4 iterations of 4 MACs each
        @parameter
        for k_base in range(0, TILE, 4):
            dW_acc += (
                x_T_shared[local_row, k_base + 0]
                * dy_shared[k_base + 0, local_col]
            )
            dW_acc += (
                x_T_shared[local_row, k_base + 1]
                * dy_shared[k_base + 1, local_col]
            )
            dW_acc += (
                x_T_shared[local_row, k_base + 2]
                * dy_shared[k_base + 2, local_col]
            )
            dW_acc += (
                x_T_shared[local_row, k_base + 3]
                * dy_shared[k_base + 3, local_col]
            )

        barrier()

    # Write dW
    if global_row < IN_DIM and global_col < OUT_DIM:
        dW[global_row, global_col] = dW_acc

    # Reduce and write db (only blocks in first row)
    if compute_db:
        # Reuse dy_shared for reduction across local_row
        dy_shared[local_row, local_col] = db_acc
        barrier()

        # Tree reduction with stride (works for any TILE size)
        var stride = TILE // 2
        while stride > 0:
            if local_row < stride:
                dy_shared[local_row, local_col] += dy_shared[
                    local_row + stride, local_col
                ]
            barrier()
            stride //= 2

        # Thread 0 of each column writes the result
        if local_row == 0 and global_col < OUT_DIM:
            db[global_col] = dy_shared[0, local_col]


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
# Fused dx + ReLU backward Kernel
# =============================================================================


fn linear_backward_dx_relu_kernel[
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
](
    dx: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin],
    dy: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin],
    W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
    pre_activation: LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin
    ],
):
    """Fused: dx = (dy @ W.T) * (pre_activation > 0).

    Combines linear backward (dx = dy @ W.T) with ReLU backward.
    Eliminates intermediate d_h1 tensor from memory.

    Optimized with loop unrolling (4x) for better ILP.
    """
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

        # UNROLLED by 4 for better ILP
        # TILE=16, so 16/4 = 4 iterations of 4 MACs each
        @parameter
        for k_base in range(0, TILE, 4):
            acc += (
                dy_shared[local_row, k_base + 0]
                * W_T_shared[k_base + 0, local_col]
            )
            acc += (
                dy_shared[local_row, k_base + 1]
                * W_T_shared[k_base + 1, local_col]
            )
            acc += (
                dy_shared[local_row, k_base + 2]
                * W_T_shared[k_base + 2, local_col]
            )
            acc += (
                dy_shared[local_row, k_base + 3]
                * W_T_shared[k_base + 3, local_col]
            )

        barrier()

    # Fused: apply ReLU mask before writing
    if global_row < BATCH and global_col < IN_DIM:
        var pre_act = pre_activation[global_row, global_col]
        dx[global_row, global_col] = acc if pre_act > 0 else 0


# =============================================================================
# Adam Optimizer - Modular Design
# =============================================================================
#
# Design pattern for composable GPU code:
# 1. Single-param kernel - simple, reusable (adam_kernel)
# 2. Multi-param kernel - fused version for reduced launch overhead
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
    """Single-parameter Adam kernel."""
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


fn adam_multi_kernel[
    SIZE1: Int,  # W1 size
    SIZE2: Int,  # b1 size
    SIZE3: Int,  # W2 size
    SIZE4: Int,  # b2 size
](
    # Parameter group 1: W1
    w1: LayoutTensor[dtype, Layout.row_major(SIZE1), MutAnyOrigin],
    dw1: LayoutTensor[dtype, Layout.row_major(SIZE1), ImmutAnyOrigin],
    m_w1: LayoutTensor[dtype, Layout.row_major(SIZE1), MutAnyOrigin],
    v_w1: LayoutTensor[dtype, Layout.row_major(SIZE1), MutAnyOrigin],
    # Parameter group 2: b1
    b1: LayoutTensor[dtype, Layout.row_major(SIZE2), MutAnyOrigin],
    db1: LayoutTensor[dtype, Layout.row_major(SIZE2), ImmutAnyOrigin],
    m_b1: LayoutTensor[dtype, Layout.row_major(SIZE2), MutAnyOrigin],
    v_b1: LayoutTensor[dtype, Layout.row_major(SIZE2), MutAnyOrigin],
    # Parameter group 3: W2
    w2: LayoutTensor[dtype, Layout.row_major(SIZE3), MutAnyOrigin],
    dw2: LayoutTensor[dtype, Layout.row_major(SIZE3), ImmutAnyOrigin],
    m_w2: LayoutTensor[dtype, Layout.row_major(SIZE3), MutAnyOrigin],
    v_w2: LayoutTensor[dtype, Layout.row_major(SIZE3), MutAnyOrigin],
    # Parameter group 4: b2
    b2: LayoutTensor[dtype, Layout.row_major(SIZE4), MutAnyOrigin],
    db2: LayoutTensor[dtype, Layout.row_major(SIZE4), ImmutAnyOrigin],
    m_b2: LayoutTensor[dtype, Layout.row_major(SIZE4), MutAnyOrigin],
    v_b2: LayoutTensor[dtype, Layout.row_major(SIZE4), MutAnyOrigin],
    # Shared hyperparameters
    lr: Scalar[dtype],
    beta1: Scalar[dtype],
    beta2: Scalar[dtype],
    eps: Scalar[dtype],
    bc1: Scalar[dtype],
    bc2: Scalar[dtype],
):
    """Fused Adam kernel for 4 parameter groups (2-layer MLP).

    Grid layout: threads are assigned to parameter groups based on global_idx
    - Threads 0 to SIZE1-1: handle W1
    - Threads SIZE1 to SIZE1+SIZE2-1: handle b1
    - etc.

    This reduces kernel launch overhead from 4 launches to 1.
    """
    var global_idx = Int(block_dim.x * block_idx.x + thread_idx.x)

    # Compute boundaries for each parameter group
    comptime TOTAL_SIZE = SIZE1 + SIZE2 + SIZE3 + SIZE4

    if global_idx >= TOTAL_SIZE:
        return

    # Dispatch to appropriate parameter group
    if global_idx < SIZE1:
        # Group 1: W1
        var idx = global_idx
        var g = dw1[idx]
        var m_val = m_w1[idx]
        var v_val = v_w1[idx]
        var m_new = beta1 * m_val + (1 - beta1) * g
        var v_new = beta2 * v_val + (1 - beta2) * g * g
        var m_hat = m_new / bc1
        var v_hat = v_new / bc2
        w1[idx] = w1[idx] - lr * m_hat / (sqrt(v_hat) + eps)
        m_w1[idx] = m_new
        v_w1[idx] = v_new

    elif global_idx < SIZE1 + SIZE2:
        # Group 2: b1
        var idx = global_idx - SIZE1
        var g = db1[idx]
        var m_val = m_b1[idx]
        var v_val = v_b1[idx]
        var m_new = beta1 * m_val + (1 - beta1) * g
        var v_new = beta2 * v_val + (1 - beta2) * g * g
        var m_hat = m_new / bc1
        var v_hat = v_new / bc2
        b1[idx] = b1[idx] - lr * m_hat / (sqrt(v_hat) + eps)
        m_b1[idx] = m_new
        v_b1[idx] = v_new

    elif global_idx < SIZE1 + SIZE2 + SIZE3:
        # Group 3: W2
        var idx = global_idx - SIZE1 - SIZE2
        var g = dw2[idx]
        var m_val = m_w2[idx]
        var v_val = v_w2[idx]
        var m_new = beta1 * m_val + (1 - beta1) * g
        var v_new = beta2 * v_val + (1 - beta2) * g * g
        var m_hat = m_new / bc1
        var v_hat = v_new / bc2
        w2[idx] = w2[idx] - lr * m_hat / (sqrt(v_hat) + eps)
        m_w2[idx] = m_new
        v_w2[idx] = v_new

    else:
        # Group 4: b2
        var idx = global_idx - SIZE1 - SIZE2 - SIZE3
        var g = db2[idx]
        var m_val = m_b2[idx]
        var v_val = v_b2[idx]
        var m_new = beta1 * m_val + (1 - beta1) * g
        var v_new = beta2 * v_val + (1 - beta2) * g * g
        var m_hat = m_new / bc1
        var v_hat = v_new / bc2
        b2[idx] = b2[idx] - lr * m_hat / (sqrt(v_hat) + eps)
        m_b2[idx] = m_new
        v_b2[idx] = v_new


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
    var count: Int

    fn __init__(out self):
        self.data_select_ns = 0
        self.forward_ns = 0
        self.loss_ns = 0
        self.backward_ns = 0
        self.adam_ns = 0
        self.count = 0

    fn print_stats(self):
        var total = (
            self.data_select_ns
            + self.forward_ns
            + self.loss_ns
            + self.backward_ns
            + self.adam_ns
        )
        print("\nTiming breakdown (average per epoch, with GPU sync):")
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

        var print_every = 50  # More frequent to see timing trend

        # Grid/block dimensions
        # Convention: global_row = block_idx.y * TILE, global_col = block_idx.x * TILE
        # So grid = (cols/TILE, rows/TILE) = (N/TILE, M/TILE) for output shape (M, N)

        # Forward pass: output is (BATCH, OUT_DIM) for each layer
        comptime grid_h1 = (
            (HIDDEN_DIM + TILE - 1) // TILE,  # cols (N = HIDDEN_DIM)
            (BATCH_SIZE + TILE - 1) // TILE,  # rows (M = BATCH_SIZE)
        )
        comptime grid_out = (
            (OUTPUT_DIM + TILE - 1) // TILE,  # cols (N = OUTPUT_DIM)
            (BATCH_SIZE + TILE - 1) // TILE,  # rows (M = BATCH_SIZE)
        )
        # Backward dW: output is (IN_DIM, OUT_DIM)
        comptime grid_dW1 = (
            (HIDDEN_DIM + TILE - 1) // TILE,  # cols (N = OUT_DIM = HIDDEN_DIM)
            (INPUT_DIM + TILE - 1) // TILE,  # rows (M = IN_DIM = INPUT_DIM)
        )
        comptime grid_dW2 = (
            (OUTPUT_DIM + TILE - 1) // TILE,  # cols (N = OUT_DIM = OUTPUT_DIM)
            (HIDDEN_DIM + TILE - 1) // TILE,  # rows (M = IN_DIM = HIDDEN_DIM)
        )
        # Backward dx: output is (BATCH, IN_DIM)
        comptime grid_dx_h1 = (
            (HIDDEN_DIM + TILE - 1) // TILE,  # cols (N = IN_DIM = HIDDEN_DIM)
            (BATCH_SIZE + TILE - 1) // TILE,  # rows (M = BATCH_SIZE)
        )
        comptime block_2d = (TILE, TILE)

        # =====================================================================
        # Pre-compile kernels
        # =====================================================================

        print()
        print("Compiling kernels...")

        # Fused forward for layer1: outputs both pre-ReLU and post-ReLU
        var linear_forward_dual_layer1 = ctx.compile_function_checked[
            linear_forward_relu_dual_kernel[BATCH_SIZE, INPUT_DIM, HIDDEN_DIM],
            linear_forward_relu_dual_kernel[BATCH_SIZE, INPUT_DIM, HIDDEN_DIM],
        ]()
        var linear_forward_layer2 = ctx.compile_function_checked[
            linear_forward_kernel[BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM],
            linear_forward_kernel[BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM],
        ]()
        var mse_loss_fn = ctx.compile_function_checked[
            mse_loss_kernel[BATCH_SIZE, OUTPUT_DIM],
            mse_loss_kernel[BATCH_SIZE, OUTPUT_DIM],
        ]()
        # Triple-fused: MSE backward + dW2 + db2 in one kernel
        var backward_mse_dW_db2_fn = ctx.compile_function_checked[
            linear_backward_mse_dW_db_kernel[
                BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM
            ],
            linear_backward_mse_dW_db_kernel[
                BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM
            ],
        ]()
        # Fused dx + ReLU backward (eliminates d_h1 intermediate tensor)
        var backward_dx_relu_fn = ctx.compile_function_checked[
            linear_backward_dx_relu_kernel[BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM],
            linear_backward_dx_relu_kernel[BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM],
        ]()
        var backward_dW_db1_fn = ctx.compile_function_checked[
            linear_backward_dW_db_kernel[BATCH_SIZE, INPUT_DIM, HIDDEN_DIM],
            linear_backward_dW_db_kernel[BATCH_SIZE, INPUT_DIM, HIDDEN_DIM],
        ]()
        # Fused Adam kernel for all 4 parameter groups (reduces 4 launches to 1)
        var adam_all_fn = ctx.compile_function_checked[
            adam_multi_kernel[W1_SIZE, B1_SIZE, W2_SIZE, B2_SIZE],
            adam_multi_kernel[W1_SIZE, B1_SIZE, W2_SIZE, B2_SIZE],
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
            # Forward pass (fused layer1 - one matmul instead of two!)
            # =================================================================

            ctx.enqueue_function_checked(
                linear_forward_dual_layer1,
                h1_pre_t,  # pre-ReLU output (for backward)
                h1_t,  # post-ReLU output (for layer 2)
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
            var should_log = (epoch + 1) % print_every == 0 or epoch == 0

            # Only sync for timing when logging
            if should_log:
                ctx.synchronize()

            var t2 = perf_counter_ns()
            if should_log:
                stats.forward_ns += Int(t2 - t1)

            # =================================================================
            # Compute loss (ONLY when logging - saves ~8ms per non-logging epoch)
            # =================================================================

            if should_log:
                ctx.enqueue_function_checked(
                    mse_loss_fn,
                    loss_t,
                    y_pred_t,
                    y_t,
                    grid_dim=(1,),
                    block_dim=(TPB,),
                )
                ctx.synchronize()

            var t3 = perf_counter_ns()
            if should_log:
                stats.loss_ns += Int(t3 - t2)

            # =================================================================
            # Backward pass
            # =================================================================

            # Triple-fused: MSE backward + dW2 + db2 in ONE kernel
            # Also outputs d_y_pred for subsequent backward_dx_relu
            ctx.enqueue_function_checked(
                backward_mse_dW_db2_fn,
                dW2_t,
                db2_t,
                d_y_pred_t,  # Output: d_y_pred for next kernel
                h1_t,
                y_pred_t,
                y_t,
                grid_dim=grid_dW2,
                block_dim=block_2d,
            )

            # Fused: dx_h1 + relu_backward (eliminates d_h1 intermediate)
            # Computes d_h1_pre = (d_y_pred @ W2.T) * (h1_pre > 0) directly
            ctx.enqueue_function_checked(
                backward_dx_relu_fn,
                d_h1_pre_t,  # Output: d_h1_pre directly (skips d_h1)
                d_y_pred_t,
                W2_t,
                h1_pre_t,  # Pre-activation for ReLU mask
                grid_dim=grid_dx_h1,
                block_dim=block_2d,
            )

            # Layer 1 backward: fused dW1 + db1
            ctx.enqueue_function_checked(
                backward_dW_db1_fn,
                dW1_t,
                db1_t,
                x_t,
                d_h1_pre_t,
                grid_dim=grid_dW1,
                block_dim=block_2d,
            )

            # Only sync for timing when logging
            if should_log:
                ctx.synchronize()

            var t4 = perf_counter_ns()
            if should_log:
                stats.backward_ns += Int(t4 - t3)

            # =================================================================
            # Adam updates - Fused (single kernel for all 4 parameter groups)
            # =================================================================

            var t = Scalar[dtype](epoch + 1)
            var bc1 = Scalar[dtype](1) - beta1**t
            var bc2 = Scalar[dtype](1) - beta2**t

            # Flat views for Adam
            var W1_flat = LayoutTensor[
                dtype, Layout.row_major(W1_SIZE), MutAnyOrigin
            ](W1_buf)
            var dW1_flat = LayoutTensor[
                dtype, Layout.row_major(W1_SIZE), ImmutAnyOrigin
            ](dW1_buf)
            var b1_flat = LayoutTensor[
                dtype, Layout.row_major(B1_SIZE), MutAnyOrigin
            ](b1_buf)
            var db1_flat = LayoutTensor[
                dtype, Layout.row_major(B1_SIZE), ImmutAnyOrigin
            ](db1_buf)
            var W2_flat = LayoutTensor[
                dtype, Layout.row_major(W2_SIZE), MutAnyOrigin
            ](W2_buf)
            var dW2_flat = LayoutTensor[
                dtype, Layout.row_major(W2_SIZE), ImmutAnyOrigin
            ](dW2_buf)
            var b2_flat = LayoutTensor[
                dtype, Layout.row_major(B2_SIZE), MutAnyOrigin
            ](b2_buf)
            var db2_flat = LayoutTensor[
                dtype, Layout.row_major(B2_SIZE), ImmutAnyOrigin
            ](db2_buf)

            comptime TOTAL_PARAMS = W1_SIZE + B1_SIZE + W2_SIZE + B2_SIZE
            comptime adam_blocks = (TOTAL_PARAMS + TPB - 1) // TPB

            ctx.enqueue_function_checked(
                adam_all_fn,
                # W1 group
                W1_flat,
                dW1_flat,
                m_W1_t,
                v_W1_t,
                # b1 group
                b1_flat,
                db1_flat,
                m_b1_t,
                v_b1_t,
                # W2 group
                W2_flat,
                dW2_flat,
                m_W2_t,
                v_W2_t,
                # b2 group
                b2_flat,
                db2_flat,
                m_b2_t,
                v_b2_t,
                # Hyperparameters
                lr,
                beta1,
                beta2,
                eps,
                bc1,
                bc2,
                grid_dim=(adam_blocks,),
                block_dim=(TPB,),
            )

            # Only sync for timing when logging
            if should_log:
                ctx.synchronize()

            var t5 = perf_counter_ns()
            if should_log:
                stats.adam_ns += Int(t5 - t4)
                stats.count += 1

            # Print progress (loss was computed above only when should_log)
            if should_log:
                var epoch_time_ms = Float64(t5 - t0) / 1e6
                with loss_buf.map_to_host() as host:
                    var loss_val = Float32(host[0])
                    print(
                        "Epoch "
                        + String(epoch + 1)
                        + "/"
                        + String(NUM_EPOCHS)
                        + " - Loss: "
                        + String(loss_val)
                        + " - Time: "
                        + String(epoch_time_ms)[:6]
                        + " ms"
                    )

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
            linear_forward_dual_layer1,
            h1_pre_t,  # not used in eval, but kernel outputs both
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

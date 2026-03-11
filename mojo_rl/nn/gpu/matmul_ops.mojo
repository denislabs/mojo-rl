"""Optimized matmul building blocks for deep RL on Apple Silicon.

This module provides reusable, highly optimized matmul operations that can be
composed by the model layers (Linear, LinearTanh, LinearReLU, etc.).

Key features:
1. 8x8 tile sizes optimized for Apple Silicon simdgroups
2. Parametrized operations: bias, input caching, activation functions
3. Support for fused forward passes (matmul + bias + activation + caching)

Usage:
    # Simple matmul with bias
    matmul_bias_kernel[BATCH, M, N, K, TILE](output, input, W, bias)

    # Matmul with bias and input caching
    matmul_bias_cache_input_kernel[BATCH, M, N, K, TILE](output, input, W, bias, cache)

    # Fused matmul + tanh with caching
    matmul_bias_tanh_cached_kernel[BATCH, M, N, K, TILE](output, input, W, bias, cache)
"""

from std.math import tanh
from std.gpu import thread_idx, block_idx, barrier
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor

from ..constants import dtype

# Apple Silicon optimized tile size (matches 32-thread simdgroup)
comptime TILE_APPLE = 8


# =============================================================================
# Matmul + Bias (no caching, no activation) - for inference
# =============================================================================


@always_inline
fn matmul_bias_kernel[
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
    TILE: Int = TILE_APPLE,
](
    output: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    input: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
    bias: LayoutTensor[dtype, Layout.row_major(OUT_DIM), ImmutAnyOrigin],
):
    """Matmul with bias: output = input @ W + bias.

    Grid: ((OUT_DIM + TILE - 1) // TILE, (BATCH + TILE - 1) // TILE)
    Block: (TILE, TILE)
    """
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

    # Start with bias
    var acc: output.element_type = 0
    if global_col < OUT_DIM:
        acc = bias[global_col]

    for tile_idx in range((IN_DIM + TILE - 1) // TILE):
        var x_col = tile_idx * TILE + local_col
        var W_row = tile_idx * TILE + local_row

        if global_row < BATCH and x_col < IN_DIM:
            x_shared[local_row, local_col] = input[global_row, x_col]
        else:
            x_shared[local_row, local_col] = 0

        if W_row < IN_DIM and global_col < OUT_DIM:
            W_shared[local_row, local_col] = W[W_row, global_col]
        else:
            W_shared[local_row, local_col] = 0

        barrier()

        comptime for k in range(TILE):
            acc += rebind[output.element_type](x_shared[local_row, k]) * rebind[
                output.element_type
            ](W_shared[k, local_col])

        barrier()

    if global_row < BATCH and global_col < OUT_DIM:
        output[global_row, global_col] = acc


# =============================================================================
# Matmul + Bias + Input Caching (for training, no activation)
# =============================================================================


@always_inline
fn matmul_bias_cache_input_kernel[
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
    TILE: Int = TILE_APPLE,
](
    output: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    input: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
    bias: LayoutTensor[dtype, Layout.row_major(OUT_DIM), ImmutAnyOrigin],
    input_cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin
    ],
):
    """Matmul with bias and input caching: output = input @ W + bias.

    Caches input for backward pass (dW computation).

    Grid: ((OUT_DIM + TILE - 1) // TILE, (BATCH + TILE - 1) // TILE)
    Block: (TILE, TILE)
    """
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

    # Start with bias
    var acc: output.element_type = 0
    if global_col < OUT_DIM:
        acc = bias[global_col]

    for tile_idx in range((IN_DIM + TILE - 1) // TILE):
        var x_col = tile_idx * TILE + local_col
        var W_row = tile_idx * TILE + local_row

        # Load input tile and cache it
        if global_row < BATCH and x_col < IN_DIM:
            var x_val = input[global_row, x_col]
            x_shared[local_row, local_col] = x_val
            input_cache[global_row, x_col] = x_val  # Cache for backward
        else:
            x_shared[local_row, local_col] = 0

        if W_row < IN_DIM and global_col < OUT_DIM:
            W_shared[local_row, local_col] = W[W_row, global_col]
        else:
            W_shared[local_row, local_col] = 0

        barrier()

        comptime for k in range(TILE):
            acc += rebind[output.element_type](x_shared[local_row, k]) * rebind[
                output.element_type
            ](W_shared[k, local_col])

        barrier()

    if global_row < BATCH and global_col < OUT_DIM:
        output[global_row, global_col] = acc


# =============================================================================
# Fused Matmul + Bias + Tanh with Full Caching
# =============================================================================


@always_inline
fn matmul_bias_tanh_cached_kernel[
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
    CACHE_SIZE: Int,  # IN_DIM + OUT_DIM typically
    TILE: Int = TILE_APPLE,
](
    output: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    input: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
    bias: LayoutTensor[dtype, Layout.row_major(OUT_DIM), ImmutAnyOrigin],
    cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin
    ],
):
    """Fused matmul + bias + tanh with caching.

    output = tanh(input @ W + bias)

    Cache layout: [input (IN_DIM) | tanh_output (OUT_DIM)]
    - Input cached for dW computation
    - Tanh output cached for gradient: d/dx tanh(x) = 1 - tanh²(x)

    Grid: ((OUT_DIM + TILE - 1) // TILE, (BATCH + TILE - 1) // TILE)
    Block: (TILE, TILE)
    """
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

    # Start with bias
    var acc: output.element_type = 0
    if global_col < OUT_DIM:
        acc = bias[global_col]

    for tile_idx in range((IN_DIM + TILE - 1) // TILE):
        var x_col = tile_idx * TILE + local_col
        var W_row = tile_idx * TILE + local_row

        # Load input tile and cache it
        if global_row < BATCH and x_col < IN_DIM:
            var x_val = input[global_row, x_col]
            x_shared[local_row, local_col] = x_val
            cache[global_row, x_col] = x_val  # Cache input
        else:
            x_shared[local_row, local_col] = 0

        if W_row < IN_DIM and global_col < OUT_DIM:
            W_shared[local_row, local_col] = W[W_row, global_col]
        else:
            W_shared[local_row, local_col] = 0

        barrier()

        comptime for k in range(TILE):
            acc += rebind[output.element_type](x_shared[local_row, k]) * rebind[
                output.element_type
            ](W_shared[k, local_col])

        barrier()

    # Apply tanh and cache output
    if global_row < BATCH and global_col < OUT_DIM:
        var tanh_out = tanh(acc)
        cache[global_row, IN_DIM + global_col] = tanh_out  # Cache tanh output
        output[global_row, global_col] = tanh_out


@always_inline
fn matmul_bias_tanh_kernel[
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
    TILE: Int = TILE_APPLE,
](
    output: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    input: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
    bias: LayoutTensor[dtype, Layout.row_major(OUT_DIM), ImmutAnyOrigin],
):
    """Fused matmul + bias + tanh (no caching, for inference).

    output = tanh(input @ W + bias)

    Grid: ((OUT_DIM + TILE - 1) // TILE, (BATCH + TILE - 1) // TILE)
    Block: (TILE, TILE)
    """
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
        acc = bias[global_col]

    for tile_idx in range((IN_DIM + TILE - 1) // TILE):
        var x_col = tile_idx * TILE + local_col
        var W_row = tile_idx * TILE + local_row

        if global_row < BATCH and x_col < IN_DIM:
            x_shared[local_row, local_col] = input[global_row, x_col]
        else:
            x_shared[local_row, local_col] = 0

        if W_row < IN_DIM and global_col < OUT_DIM:
            W_shared[local_row, local_col] = W[W_row, global_col]
        else:
            W_shared[local_row, local_col] = 0

        barrier()

        comptime for k in range(TILE):
            acc += rebind[output.element_type](x_shared[local_row, k]) * rebind[
                output.element_type
            ](W_shared[k, local_col])

        barrier()

    if global_row < BATCH and global_col < OUT_DIM:
        output[global_row, global_col] = tanh(acc)


# =============================================================================
# Fused Matmul + Bias + ReLU with Caching
# =============================================================================


@always_inline
fn matmul_bias_relu_cached_kernel[
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
    CACHE_SIZE: Int,  # IN_DIM + OUT_DIM typically
    TILE: Int = TILE_APPLE,
](
    output: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    input: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
    bias: LayoutTensor[dtype, Layout.row_major(OUT_DIM), ImmutAnyOrigin],
    cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, CACHE_SIZE), MutAnyOrigin
    ],
):
    """Fused matmul + bias + ReLU with caching.

    output = max(0, input @ W + bias)

    Cache layout: [input (IN_DIM) | pre_activation (OUT_DIM)]
    - Input cached for dW computation
    - Pre-activation cached for ReLU gradient (mask)

    Grid: ((OUT_DIM + TILE - 1) // TILE, (BATCH + TILE - 1) // TILE)
    Block: (TILE, TILE)
    """
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
        acc = bias[global_col]

    for tile_idx in range((IN_DIM + TILE - 1) // TILE):
        var x_col = tile_idx * TILE + local_col
        var W_row = tile_idx * TILE + local_row

        if global_row < BATCH and x_col < IN_DIM:
            var x_val = input[global_row, x_col]
            x_shared[local_row, local_col] = x_val
            cache[global_row, x_col] = x_val  # Cache input
        else:
            x_shared[local_row, local_col] = 0

        if W_row < IN_DIM and global_col < OUT_DIM:
            W_shared[local_row, local_col] = W[W_row, global_col]
        else:
            W_shared[local_row, local_col] = 0

        barrier()

        comptime for k in range(TILE):
            acc += rebind[output.element_type](x_shared[local_row, k]) * rebind[
                output.element_type
            ](W_shared[k, local_col])

        barrier()

    # Apply ReLU and cache pre-activation
    if global_row < BATCH and global_col < OUT_DIM:
        cache[global_row, IN_DIM + global_col] = acc  # Cache pre-activation
        output[global_row, global_col] = acc if acc > 0 else 0


@always_inline
fn matmul_bias_relu_kernel[
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
    TILE: Int = TILE_APPLE,
](
    output: LayoutTensor[dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    input: LayoutTensor[dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
    bias: LayoutTensor[dtype, Layout.row_major(OUT_DIM), ImmutAnyOrigin],
):
    """Fused matmul + bias + ReLU (no caching, for inference).

    output = max(0, input @ W + bias)

    Grid: ((OUT_DIM + TILE - 1) // TILE, (BATCH + TILE - 1) // TILE)
    Block: (TILE, TILE)
    """
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
        acc = bias[global_col]

    for tile_idx in range((IN_DIM + TILE - 1) // TILE):
        var x_col = tile_idx * TILE + local_col
        var W_row = tile_idx * TILE + local_row

        if global_row < BATCH and x_col < IN_DIM:
            x_shared[local_row, local_col] = input[global_row, x_col]
        else:
            x_shared[local_row, local_col] = 0

        if W_row < IN_DIM and global_col < OUT_DIM:
            W_shared[local_row, local_col] = W[W_row, global_col]
        else:
            W_shared[local_row, local_col] = 0

        barrier()

        comptime for k in range(TILE):
            acc += rebind[output.element_type](x_shared[local_row, k]) * rebind[
                output.element_type
            ](W_shared[k, local_col])

        barrier()

    if global_row < BATCH and global_col < OUT_DIM:
        output[global_row, global_col] = acc if acc > 0 else 0


# =============================================================================
# Backward Gradient Computations
# =============================================================================


@always_inline
fn matmul_backward_dx_kernel[
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
    TILE: Int = TILE_APPLE,
](
    grad_input: LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin
    ],
    grad_output: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin
    ],
    W: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
):
    """Backward pass for input gradient: dx = dy @ W.T.

    Grid: ((IN_DIM + TILE - 1) // TILE, (BATCH + TILE - 1) // TILE)
    Block: (TILE, TILE)
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

    var acc: grad_input.element_type = 0

    for tile_idx in range((OUT_DIM + TILE - 1) // TILE):
        var dy_col = tile_idx * TILE + local_col
        var W_col = tile_idx * TILE + local_row

        if global_row < BATCH and dy_col < OUT_DIM:
            dy_shared[local_row, local_col] = grad_output[global_row, dy_col]
        else:
            dy_shared[local_row, local_col] = 0

        # Load W.T tile (transpose: W_T[i,j] = W[j,i])
        if W_col < OUT_DIM and global_col < IN_DIM:
            W_T_shared[local_row, local_col] = W[global_col, W_col]
        else:
            W_T_shared[local_row, local_col] = 0

        barrier()

        comptime for k in range(TILE):
            acc += rebind[grad_input.element_type](
                dy_shared[local_row, k]
            ) * rebind[grad_input.element_type](W_T_shared[k, local_col])

        barrier()

    if global_row < BATCH and global_col < IN_DIM:
        grad_input[global_row, global_col] = acc


@always_inline
fn matmul_backward_dW_kernel[
    BATCH: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
    TILE: Int = TILE_APPLE,
](
    dW: LayoutTensor[dtype, Layout.row_major(IN_DIM, OUT_DIM), MutAnyOrigin],
    input_cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin
    ],
    grad_output: LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), ImmutAnyOrigin
    ],
):
    """Backward pass for weight gradient: dW = x.T @ dy.

    Grid: ((OUT_DIM + TILE - 1) // TILE, (IN_DIM + TILE - 1) // TILE)
    Block: (TILE, TILE)
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

    var acc: dW.element_type = 0

    for tile_idx in range((BATCH + TILE - 1) // TILE):
        var batch_idx = tile_idx * TILE + local_col
        var dy_row = tile_idx * TILE + local_row

        # Load x.T tile (transpose: x_T[i,j] = x[j,i])
        if global_row < IN_DIM and batch_idx < BATCH:
            x_T_shared[local_row, local_col] = input_cache[
                batch_idx, global_row
            ]
        else:
            x_T_shared[local_row, local_col] = 0

        if dy_row < BATCH and global_col < OUT_DIM:
            dy_shared[local_row, local_col] = grad_output[dy_row, global_col]
        else:
            dy_shared[local_row, local_col] = 0

        barrier()

        comptime for k in range(TILE):
            acc += rebind[dW.element_type](x_T_shared[local_row, k]) * rebind[
                dW.element_type
            ](dy_shared[k, local_col])

        barrier()

    if global_row < IN_DIM and global_col < OUT_DIM:
        dW[global_row, global_col] = acc


# =============================================================================
# Helper: Grid dimensions
# =============================================================================


fn get_forward_grid[
    BATCH: Int, OUT_DIM: Int, TILE: Int = TILE_APPLE
]() -> Tuple[Int, Int]:
    """Returns (grid_x, grid_y) for forward kernels."""
    return ((OUT_DIM + TILE - 1) // TILE, (BATCH + TILE - 1) // TILE)


fn get_backward_dx_grid[
    BATCH: Int, IN_DIM: Int, TILE: Int = TILE_APPLE
]() -> Tuple[Int, Int]:
    """Returns (grid_x, grid_y) for dx backward kernel."""
    return ((IN_DIM + TILE - 1) // TILE, (BATCH + TILE - 1) // TILE)


fn get_backward_dW_grid[
    IN_DIM: Int, OUT_DIM: Int, TILE: Int = TILE_APPLE
]() -> Tuple[Int, Int]:
    """Returns (grid_x, grid_y) for dW backward kernel."""
    return ((OUT_DIM + TILE - 1) // TILE, (IN_DIM + TILE - 1) // TILE)

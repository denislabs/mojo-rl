"""GPU reduction operations for deep RL.

This module provides GPU-accelerated reduction operations using
block-wide primitives from Mojo GPU (Puzzle 27 patterns).

Operations:
- gpu_sum_kernel: sum reduction using block.sum()
- gpu_max_kernel: max reduction using block.max()
- gpu_mean_kernel: mean using block.sum()
- gpu_normalize_kernel: normalize by mean using block.sum() + block.broadcast()

These patterns replace 15+ lines of manual shared memory + barriers + tree
reduction with simple one-line block primitive calls.
"""

from gpu import thread_idx, block_idx, block_dim, barrier, block
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor


fn gpu_sum_kernel[
    dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    TPB: Int,
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    input: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
    size: Int,
):
    """GPU kernel for parallel sum reduction using block.sum().

    Replaces manual shared memory + barriers + tree reduction with one line!
    """
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    local_i = thread_idx.x

    # Each thread loads its element (0 if out of bounds)
    var my_value: Scalar[dtype] = 0
    if global_i < size:
        my_value = rebind[Scalar[dtype]](input[global_i])

    # The magic: block.sum() replaces 15+ lines of manual reduction!
    total = block.sum[block_size=TPB, broadcast=False](
        val=SIMD[dtype, 1](my_value)
    )

    # Only thread 0 writes the result
    if local_i == 0:
        output[0] = total[0]


fn gpu_max_kernel[
    dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    TPB: Int,
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    input: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
    size: Int,
):
    """GPU kernel for parallel max reduction using block.max()."""
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    local_i = thread_idx.x

    # Each thread loads its element (min value if out of bounds)
    var my_value: Scalar[dtype] = Scalar[dtype].MIN
    if global_i < size:
        my_value = rebind[Scalar[dtype]](input[global_i])

    # block.max() for parallel maximum
    max_val = block.max[block_size=TPB, broadcast=False](
        val=SIMD[dtype, 1](my_value)
    )

    # Only thread 0 writes the result
    if local_i == 0:
        output[0] = max_val[0]


fn gpu_mean_kernel[
    dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    TPB: Int,
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    input: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
    size: Int,
):
    """GPU kernel for mean using block.sum().

    Computes sum with block.sum(), then divides by size.
    """
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    local_i = thread_idx.x

    # Each thread loads its element
    var my_value: Scalar[dtype] = 0
    if global_i < size:
        my_value = rebind[Scalar[dtype]](input[global_i])

    # Sum all elements
    total = block.sum[block_size=TPB, broadcast=False](
        val=SIMD[dtype, 1](my_value)
    )

    # Thread 0 computes and writes mean
    if local_i == 0:
        output[0] = total[0] / Scalar[dtype](size)


fn gpu_normalize_kernel[
    dtype: DType,
    layout: Layout,
    TPB: Int,
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    size: Int,
):
    """GPU kernel for mean normalization using block.sum() + block.broadcast().

    Each element is divided by the mean: output[i] = input[i] / mean.
    Demonstrates block.sum() for reduction and block.broadcast() for sharing.
    """
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    local_i = thread_idx.x

    # Each thread loads its element
    var my_value: Scalar[dtype] = 0
    if global_i < size:
        my_value = rebind[Scalar[dtype]](input[global_i])

    # Step 1: Sum all elements using block.sum()
    total = block.sum[block_size=TPB, broadcast=False](
        val=SIMD[dtype, 1](my_value)
    )

    # Step 2: Thread 0 computes mean
    var mean_value: Scalar[dtype] = 1  # Default to avoid div by zero
    if local_i == 0:
        if total[0] > 0:
            mean_value = total[0] / Scalar[dtype](size)

    # Step 3: Broadcast mean to all threads
    broadcasted_mean = block.broadcast[dtype=dtype, width=1, block_size=TPB](
        val=SIMD[dtype, 1](mean_value), src_thread=UInt(0)
    )

    # Step 4: Each thread normalizes its element
    if global_i < size:
        output[global_i] = my_value / broadcasted_mean[0]

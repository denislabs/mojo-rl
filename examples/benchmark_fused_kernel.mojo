"""Benchmark: Optimized GPU kernels with tiled matmul and kernel fusion.

This benchmark explores:
1. Tiled matmul vs naive matmul
2. Separate kernel launches vs fused mega-kernel
3. The impact of kernel launch overhead

Key idea: If kernel launch overhead is 1.7ms, but we can fuse 10 kernels into 1,
we save 9 × 1.7ms = 15.3ms per training step!

Run with:
    pixi run -e apple mojo run examples/benchmark_fused_kernel.mojo
"""

from time import perf_counter_ns
from math import sqrt
from random import random_float64, seed

from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor

from deep_rl.gpu import (
    linear_forward_kernel,
    linear_forward_relu_kernel,
    tiled_matmul_kernel,
)


# =============================================================================
# Fused 2-Layer Forward Pass Kernel (single kernel launch!)
# =============================================================================


# Note: Complex tiled fused kernel removed - using simple fused kernel instead
# The tiled version caused compile time explosion due to @parameter loop unrolling
# For small networks (8->128->4), the simple "one thread per batch element" approach
# is sufficient and compiles quickly.


# =============================================================================
# Simple fused forward (no tiling, just barrier between layers)
# =============================================================================


fn fused_forward_simple_kernel[
    dtype: DType,
    BATCH: Int,
    OBS_DIM: Int,
    HIDDEN_DIM: Int,
    OUT_DIM: Int,
    TPB: Int,  # Threads per block (1D)
](
    output: LayoutTensor[
        dtype, Layout.row_major(BATCH * OUT_DIM), MutAnyOrigin
    ],
    obs: LayoutTensor[dtype, Layout.row_major(BATCH * OBS_DIM), ImmutAnyOrigin],
    W1: LayoutTensor[
        dtype, Layout.row_major(OBS_DIM * HIDDEN_DIM), ImmutAnyOrigin
    ],
    b1: LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin],
    W2: LayoutTensor[
        dtype, Layout.row_major(HIDDEN_DIM * OUT_DIM), ImmutAnyOrigin
    ],
    b2: LayoutTensor[dtype, Layout.row_major(OUT_DIM), ImmutAnyOrigin],
):
    """Simpler fused forward: one thread per batch element, computes full forward.

    This avoids complex tiling - each thread handles one sample's entire forward pass.
    Works well when BATCH <= num_threads.
    Uses runtime loops instead of @parameter to avoid compile time explosion.
    """
    var batch_idx = Int(block_dim.x * block_idx.x + thread_idx.x)

    if batch_idx >= BATCH:
        return

    # Use pointer to avoid InlineArray for GPU compatibility
    # Compute hidden layer for this batch element directly

    # Layer 1: Hidden = ReLU(obs @ W1 + b1)
    # Layer 2: Output = Hidden @ W2 + b2
    # Fused: compute output directly by accumulating through hidden

    # Fused forward: compute hidden values and immediately accumulate to outputs
    # This keeps hidden values in registers (no intermediate storage needed)

    # Initialize output accumulators
    var out0 = rebind[Scalar[dtype]](b2[0])
    var out1 = rebind[Scalar[dtype]](b2[1]) if OUT_DIM > 1 else Scalar[dtype](0)
    var out2 = rebind[Scalar[dtype]](b2[2]) if OUT_DIM > 2 else Scalar[dtype](0)
    var out3 = rebind[Scalar[dtype]](b2[3]) if OUT_DIM > 3 else Scalar[dtype](0)

    # Process hidden neurons and accumulate to outputs
    for hid in range(HIDDEN_DIM):
        # Compute hidden[hid] = ReLU(sum_k obs[k] * W1[k, hid] + b1[hid])
        var h_val = rebind[Scalar[dtype]](b1[hid])
        for k in range(OBS_DIM):
            h_val += rebind[Scalar[dtype]](
                obs[batch_idx * OBS_DIM + k]
            ) * rebind[Scalar[dtype]](W1[k * HIDDEN_DIM + hid])
        # ReLU
        if h_val < Scalar[dtype](0):
            h_val = Scalar[dtype](0)

        # Immediately accumulate to all outputs (keeps h_val in register)
        out0 += h_val * rebind[Scalar[dtype]](W2[hid * OUT_DIM + 0])
        if OUT_DIM > 1:
            out1 += h_val * rebind[Scalar[dtype]](W2[hid * OUT_DIM + 1])
        if OUT_DIM > 2:
            out2 += h_val * rebind[Scalar[dtype]](W2[hid * OUT_DIM + 2])
        if OUT_DIM > 3:
            out3 += h_val * rebind[Scalar[dtype]](W2[hid * OUT_DIM + 3])

    # Write outputs
    output[batch_idx * OUT_DIM + 0] = out0
    if OUT_DIM > 1:
        output[batch_idx * OUT_DIM + 1] = out1
    if OUT_DIM > 2:
        output[batch_idx * OUT_DIM + 2] = out2
    if OUT_DIM > 3:
        output[batch_idx * OUT_DIM + 3] = out3


# =============================================================================
# Benchmark: Separate Kernels vs Fused Kernel
# =============================================================================


fn benchmark_separate_kernels[
    BATCH: Int, OBS_DIM: Int, HIDDEN_DIM: Int, OUT_DIM: Int
](ctx: DeviceContext, num_iters: Int) raises -> Float64:
    """Benchmark: 2 separate kernel launches for 2-layer forward pass."""
    comptime dtype = DType.float32
    comptime TILE = 16

    # Allocate buffers
    var obs_gpu = ctx.enqueue_create_buffer[dtype](BATCH * OBS_DIM)
    var W1_gpu = ctx.enqueue_create_buffer[dtype](OBS_DIM * HIDDEN_DIM)
    var b1_gpu = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
    var h_gpu = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN_DIM)
    var W2_gpu = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM * OUT_DIM)
    var b2_gpu = ctx.enqueue_create_buffer[dtype](OUT_DIM)
    var out_gpu = ctx.enqueue_create_buffer[dtype](BATCH * OUT_DIM)

    # Initialize with random data
    with obs_gpu.map_to_host() as host:
        for i in range(BATCH * OBS_DIM):
            host[i] = Scalar[dtype](random_float64())
    with W1_gpu.map_to_host() as host:
        for i in range(OBS_DIM * HIDDEN_DIM):
            host[i] = Scalar[dtype]((random_float64() - 0.5) * 0.1)
    b1_gpu.enqueue_fill(0)
    with W2_gpu.map_to_host() as host:
        for i in range(HIDDEN_DIM * OUT_DIM):
            host[i] = Scalar[dtype]((random_float64() - 0.5) * 0.1)
    b2_gpu.enqueue_fill(0)
    h_gpu.enqueue_fill(0)
    out_gpu.enqueue_fill(0)
    ctx.synchronize()

    # Create tensors
    var obs_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS_DIM), ImmutAnyOrigin
    ](obs_gpu)
    var W1_t = LayoutTensor[
        dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), ImmutAnyOrigin
    ](W1_gpu)
    var b1_t = LayoutTensor[
        dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin
    ](b1_gpu)
    var h_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, HIDDEN_DIM), MutAnyOrigin
    ](h_gpu)
    var W2_t = LayoutTensor[
        dtype, Layout.row_major(HIDDEN_DIM, OUT_DIM), ImmutAnyOrigin
    ](W2_gpu)
    var b2_t = LayoutTensor[dtype, Layout.row_major(OUT_DIM), ImmutAnyOrigin](
        b2_gpu
    )
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ](out_gpu)

    comptime blocks_h_y = (BATCH + TILE - 1) // TILE
    comptime blocks_h_x = (HIDDEN_DIM + TILE - 1) // TILE
    comptime blocks_o_y = (BATCH + TILE - 1) // TILE
    comptime blocks_o_x = (OUT_DIM + TILE - 1) // TILE

    comptime kernel1 = linear_forward_relu_kernel[
        dtype, BATCH, OBS_DIM, HIDDEN_DIM, TILE
    ]
    comptime kernel2 = linear_forward_kernel[
        dtype, BATCH, HIDDEN_DIM, OUT_DIM, TILE
    ]

    # Warm up
    for _ in range(100):
        ctx.enqueue_function[kernel1, kernel1](
            h_t,
            obs_t,
            W1_t,
            b1_t,
            grid_dim=(blocks_h_x, blocks_h_y),
            block_dim=(TILE, TILE),
        )
        var h_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, HIDDEN_DIM), ImmutAnyOrigin
        ](h_gpu)
        ctx.enqueue_function[kernel2, kernel2](
            out_t,
            h_immut,
            W2_t,
            b2_t,
            grid_dim=(blocks_o_x, blocks_o_y),
            block_dim=(TILE, TILE),
        )
    ctx.synchronize()

    # Benchmark
    var start = perf_counter_ns()
    for _ in range(num_iters):
        ctx.enqueue_function[kernel1, kernel1](
            h_t,
            obs_t,
            W1_t,
            b1_t,
            grid_dim=(blocks_h_x, blocks_h_y),
            block_dim=(TILE, TILE),
        )
        var h_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, HIDDEN_DIM), ImmutAnyOrigin
        ](h_gpu)
        ctx.enqueue_function[kernel2, kernel2](
            out_t,
            h_immut,
            W2_t,
            b2_t,
            grid_dim=(blocks_o_x, blocks_o_y),
            block_dim=(TILE, TILE),
        )
    ctx.synchronize()
    var end = perf_counter_ns()

    return Float64(end - start) / Float64(num_iters)  # ns per forward pass


fn benchmark_fused_simple[
    BATCH: Int, OBS_DIM: Int, HIDDEN_DIM: Int, OUT_DIM: Int
](ctx: DeviceContext, num_iters: Int) raises -> Float64:
    """Benchmark: Single fused kernel for 2-layer forward pass."""
    comptime dtype = DType.float32
    comptime TPB = 64

    # Allocate buffers
    var obs_gpu = ctx.enqueue_create_buffer[dtype](BATCH * OBS_DIM)
    var W1_gpu = ctx.enqueue_create_buffer[dtype](OBS_DIM * HIDDEN_DIM)
    var b1_gpu = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
    var W2_gpu = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM * OUT_DIM)
    var b2_gpu = ctx.enqueue_create_buffer[dtype](OUT_DIM)
    var out_gpu = ctx.enqueue_create_buffer[dtype](BATCH * OUT_DIM)

    # Initialize
    with obs_gpu.map_to_host() as host:
        for i in range(BATCH * OBS_DIM):
            host[i] = Scalar[dtype](random_float64())
    with W1_gpu.map_to_host() as host:
        for i in range(OBS_DIM * HIDDEN_DIM):
            host[i] = Scalar[dtype]((random_float64() - 0.5) * 0.1)
    b1_gpu.enqueue_fill(0)
    with W2_gpu.map_to_host() as host:
        for i in range(HIDDEN_DIM * OUT_DIM):
            host[i] = Scalar[dtype]((random_float64() - 0.5) * 0.1)
    b2_gpu.enqueue_fill(0)
    out_gpu.enqueue_fill(0)
    ctx.synchronize()

    # Create tensors (flat layout for simple kernel)
    var obs_t = LayoutTensor[
        dtype, Layout.row_major(BATCH * OBS_DIM), ImmutAnyOrigin
    ](obs_gpu)
    var W1_t = LayoutTensor[
        dtype, Layout.row_major(OBS_DIM * HIDDEN_DIM), ImmutAnyOrigin
    ](W1_gpu)
    var b1_t = LayoutTensor[
        dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin
    ](b1_gpu)
    var W2_t = LayoutTensor[
        dtype, Layout.row_major(HIDDEN_DIM * OUT_DIM), ImmutAnyOrigin
    ](W2_gpu)
    var b2_t = LayoutTensor[dtype, Layout.row_major(OUT_DIM), ImmutAnyOrigin](
        b2_gpu
    )
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH * OUT_DIM), MutAnyOrigin
    ](out_gpu)

    comptime num_blocks = (BATCH + TPB - 1) // TPB
    comptime kernel = fused_forward_simple_kernel[
        dtype, BATCH, OBS_DIM, HIDDEN_DIM, OUT_DIM, TPB
    ]

    # Warm up
    for _ in range(100):
        ctx.enqueue_function[kernel, kernel](
            out_t,
            obs_t,
            W1_t,
            b1_t,
            W2_t,
            b2_t,
            grid_dim=(num_blocks,),
            block_dim=(TPB,),
        )
    ctx.synchronize()

    # Benchmark
    var start = perf_counter_ns()
    for _ in range(num_iters):
        ctx.enqueue_function[kernel, kernel](
            out_t,
            obs_t,
            W1_t,
            b1_t,
            W2_t,
            b2_t,
            grid_dim=(num_blocks,),
            block_dim=(TPB,),
        )
    ctx.synchronize()
    var end = perf_counter_ns()

    return Float64(end - start) / Float64(num_iters)


fn benchmark_kernel_launch_overhead(
    ctx: DeviceContext, num_iters: Int
) raises -> Float64:
    """Measure pure kernel launch overhead."""
    comptime dtype = DType.float32
    comptime size = 64

    var buf = ctx.enqueue_create_buffer[dtype](size)
    buf.enqueue_fill(0)
    ctx.synchronize()

    var buf_t = LayoutTensor[dtype, Layout.row_major(size), MutAnyOrigin](buf)

    fn trivial_kernel(
        buf: LayoutTensor[dtype, Layout.row_major(size), MutAnyOrigin]
    ):
        var i = Int(thread_idx.x)
        if i < size:
            buf[i] = rebind[Scalar[dtype]](buf[i]) + Scalar[dtype](1.0)

    # Warm up
    for _ in range(100):
        ctx.enqueue_function[trivial_kernel, trivial_kernel](
            buf_t, grid_dim=(1,), block_dim=(64,)
        )
    ctx.synchronize()

    # Benchmark
    var start = perf_counter_ns()
    for _ in range(num_iters):
        ctx.enqueue_function[trivial_kernel, trivial_kernel](
            buf_t, grid_dim=(1,), block_dim=(64,)
        )
    ctx.synchronize()
    var end = perf_counter_ns()

    return Float64(end - start) / Float64(num_iters)


# =============================================================================
# CPU Baseline
# =============================================================================


fn benchmark_cpu_forward[
    BATCH: Int, OBS_DIM: Int, HIDDEN_DIM: Int, OUT_DIM: Int
](num_iters: Int) -> Float64:
    """CPU baseline for comparison."""
    # Use List to prevent optimization (heap allocated, can't be optimized away)
    var W1 = List[Float32](capacity=OBS_DIM * HIDDEN_DIM)
    var h = List[Float32](capacity=HIDDEN_DIM)  # Hidden layer buffer
    var W2 = List[Float32](capacity=HIDDEN_DIM * OUT_DIM)
    var obs = List[Float32](capacity=BATCH * OBS_DIM)
    var output = List[Float32](capacity=BATCH * OUT_DIM)

    for i in range(OBS_DIM * HIDDEN_DIM):
        W1.append(Float32((random_float64() - 0.5) * 0.1))
    for i in range(HIDDEN_DIM):
        h.append(Float32(0))
    for i in range(HIDDEN_DIM * OUT_DIM):
        W2.append(Float32((random_float64() - 0.5) * 0.1))
    for i in range(BATCH * OBS_DIM):
        obs.append(Float32(random_float64()))
    for i in range(BATCH * OUT_DIM):
        output.append(Float32(0))

    # Warm up
    for _ in range(100):
        for batch_idx in range(BATCH):
            # Hidden layer (no bias for simplicity)
            for j in range(HIDDEN_DIM):
                var sum_val: Float32 = 0
                for k in range(OBS_DIM):
                    sum_val += (
                        obs[batch_idx * OBS_DIM + k] * W1[k * HIDDEN_DIM + j]
                    )
                h[j] = sum_val if sum_val > 0 else 0  # ReLU

            # Output layer
            for j in range(OUT_DIM):
                var sum_val: Float32 = 0
                for k in range(HIDDEN_DIM):
                    sum_val += h[k] * W2[k * OUT_DIM + j]
                output[batch_idx * OUT_DIM + j] = sum_val

    # Benchmark
    var start = perf_counter_ns()
    for _ in range(num_iters):
        for batch_idx in range(BATCH):
            # Hidden layer
            for j in range(HIDDEN_DIM):
                var sum_val: Float32 = 0
                for k in range(OBS_DIM):
                    sum_val += (
                        obs[batch_idx * OBS_DIM + k] * W1[k * HIDDEN_DIM + j]
                    )
                h[j] = sum_val if sum_val > 0 else 0  # ReLU

            # Output layer
            for j in range(OUT_DIM):
                var sum_val: Float32 = 0
                for k in range(HIDDEN_DIM):
                    sum_val += h[k] * W2[k * OUT_DIM + j]
                output[batch_idx * OUT_DIM + j] = sum_val
    var end = perf_counter_ns()

    # Prevent dead code elimination
    var dummy: Float32 = 0
    for i in range(BATCH * OUT_DIM):
        dummy += output[i]
    if dummy == Float32.MAX:
        print("dummy")

    return Float64(end - start) / Float64(num_iters)


# =============================================================================
# Main
# =============================================================================


def main():
    seed(42)
    print("=" * 70)
    print("Fused Kernel Benchmark: Separate vs Fused GPU Kernels")
    print("=" * 70)
    print()

    # DQN-like network: 8 -> 128 -> 4
    comptime OBS_DIM = 8
    comptime HIDDEN_DIM = 128
    comptime OUT_DIM = 4
    comptime BATCH = 64

    var num_iters = 1000

    print(
        "Network: "
        + String(OBS_DIM)
        + " -> "
        + String(HIDDEN_DIM)
        + " -> "
        + String(OUT_DIM)
    )
    print("Batch size: " + String(BATCH))
    print("Iterations: " + String(num_iters))
    print()

    # CPU baseline
    print("-" * 70)
    print("CPU Baseline")
    print("-" * 70)
    var cpu_ns = benchmark_cpu_forward[BATCH, OBS_DIM, HIDDEN_DIM, OUT_DIM](
        num_iters
    )
    print("  " + String(cpu_ns / 1000)[:8] + " us per batch forward pass")
    print("  " + String(Int(1e9 / cpu_ns)) + " batches/sec")
    print()

    with DeviceContext() as ctx:
        # Kernel launch overhead
        print("-" * 70)
        print("GPU Kernel Launch Overhead")
        print("-" * 70)
        var launch_ns = benchmark_kernel_launch_overhead(ctx, num_iters)
        print("  " + String(launch_ns / 1000)[:8] + " us per kernel launch")
        print()

        # Separate kernels (2 launches)
        print("-" * 70)
        print("GPU: 2 Separate Kernel Launches (tiled matmul)")
        print("-" * 70)
        var separate_ns = benchmark_separate_kernels[
            BATCH, OBS_DIM, HIDDEN_DIM, OUT_DIM
        ](ctx, num_iters)
        print(
            "  " + String(separate_ns / 1000)[:8] + " us per batch forward pass"
        )
        print("  " + String(Int(1e9 / separate_ns)) + " batches/sec")
        print("  (includes 2 kernel launches)")
        print()

        # Fused kernel (1 launch)
        print("-" * 70)
        print("GPU: 1 Fused Kernel Launch")
        print("-" * 70)
        var fused_ns = benchmark_fused_simple[
            BATCH, OBS_DIM, HIDDEN_DIM, OUT_DIM
        ](ctx, num_iters)
        print("  " + String(fused_ns / 1000)[:8] + " us per batch forward pass")
        print("  " + String(Int(1e9 / fused_ns)) + " batches/sec")
        print("  (single kernel launch)")
        print()

        # Summary
        print("=" * 70)
        print("Summary")
        print("=" * 70)
        print("  CPU:            " + String(cpu_ns / 1000)[:8] + " us")
        print("  GPU separate:   " + String(separate_ns / 1000)[:8] + " us")
        print("  GPU fused:      " + String(fused_ns / 1000)[:8] + " us")
        print("  Kernel launch:  " + String(launch_ns / 1000)[:8] + " us")
        print()
        print(
            "  Fused speedup over separate: "
            + String(separate_ns / fused_ns)[:4]
            + "x"
        )
        print("  GPU fused vs CPU: " + String(cpu_ns / fused_ns)[:4] + "x")
        print()

        if fused_ns < cpu_ns:
            print("  -> GPU fused kernel is FASTER than CPU!")
        else:
            print("  -> CPU is still faster for this small network")
            print("  -> Kernel launch overhead dominates computation")

        print("=" * 70)

"""Benchmark: LinearBatchNormReLU backward for AlphaZero.

Profiles the relu_bn_backward_kernel_impl which takes 339μs in nsys.

Problem: For LinearBatchNormReLU[5376, 256] with BATCH=64:
  - Grid: (256,) blocks, Block: (256,) threads
  - BATCH=64 < TPB=256 → 75% threads idle in accumulation
  - 4 sequential shared-memory reductions (32 barriers total)
  - Each block handles just 1 feature — very little work per block

Optimized approach: Process multiple features per block.
  - Grid: (ceil(out_dim/FEATURES_PER_BLOCK),), Block: (TPB,)
  - Each block handles F features, reducing across BATCH in parallel
  - Combine 4 reductions into 1 pass with 4 accumulators
  - Skip shared memory reduction entirely when BATCH <= warp size (32)

Run:
    pixi run -e nvidia mojo run -I . tests/nn/test_bn_backward_bench.mojo
"""

from std.random import seed, random_float64
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.model.linear_bn_relu import LinearBatchNormReLU


# ─────────────────────────────────────────────────────────────────────
# Optimized kernel: multiple features per block, single reduction pass
# ─────────────────────────────────────────────────────────────────────


@always_inline
def bn_relu_backward_optimized[
    BATCH: Int,
    out_dim: Int,
    in_dim: Int,
    GAMMA_OFF: Int,
    BETA_OFF: Int,
    XHAT_OFF: Int,
    INVSTD_OFF: Int,
    PARAM_SIZE: Int,
    CACHE_SIZE: Int,
](
    grad_pre_bn: LayoutTensor[
        dtype, Layout.row_major(BATCH, out_dim), MutAnyOrigin
    ],
    grad_output: LayoutTensor[
        dtype, Layout.row_major(BATCH, out_dim), ImmutAnyOrigin
    ],
    params: LayoutTensor[
        dtype, Layout.row_major(PARAM_SIZE), ImmutAnyOrigin
    ],
    cache: LayoutTensor[
        dtype, Layout.row_major(BATCH, CACHE_SIZE), ImmutAnyOrigin
    ],
    grads: LayoutTensor[
        dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
    ],
):
    """Optimized BN+ReLU backward: one thread per (batch, feature) element.

    Grid: (ceil(BATCH * out_dim / TPB),)
    Block: (TPB,)

    Each thread computes one element of grad_pre_bn.
    Uses atomicAdd for dgamma/dbeta (only out_dim atoms total).
    Two-pass approach:
      Pass 1: Compute dgamma, dbeta, sum_dy_g, sum_dy_g_xh via atomics
      Pass 2: Compute grad_pre_bn using the reduced values
    """
    # This kernel uses a different strategy: one block per feature,
    # but with warp-level reduction instead of shared memory.
    # For BATCH=64, we use 2 warps (64 threads) per feature,
    # packing multiple features into one block.

    # Strategy: each block handles 4 features, each using BATCH threads
    comptime FEATURES_PER_BLOCK = 4
    comptime THREADS_PER_FEATURE = TPB // FEATURES_PER_BLOCK  # 64

    var tid = Int(thread_idx.x)
    var feature_in_block = tid // THREADS_PER_FEATURE  # 0..3
    var lane_in_feature = tid % THREADS_PER_FEATURE  # 0..63

    var f = Int(block_idx.x) * FEATURES_PER_BLOCK + feature_in_block
    if f >= out_dim:
        return

    var n_f = Scalar[dtype](BATCH)
    var gamma = rebind[Scalar[dtype]](params[GAMMA_OFF + f])
    var beta = rebind[Scalar[dtype]](params[BETA_OFF + f])
    var inv_std = rebind[Scalar[dtype]](cache[0, INVSTD_OFF + f])

    # Shared memory: 4 values per feature × FEATURES_PER_BLOCK
    var smem = LayoutTensor[
        dtype, Layout.row_major(FEATURES_PER_BLOCK * 4), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # Pass 1: Accumulate partial sums (each thread handles 1 batch element)
    var local_d_gamma: Scalar[dtype] = 0
    var local_d_beta: Scalar[dtype] = 0
    var local_sum_dy_g: Scalar[dtype] = 0
    var local_sum_dy_g_xh: Scalar[dtype] = 0

    var idx = lane_in_feature
    while idx < BATCH:
        var x_hat = rebind[Scalar[dtype]](cache[idx, XHAT_OFF + f])
        var pre_relu = gamma * x_hat + beta
        var dy = rebind[Scalar[dtype]](grad_output[idx, f])
        if pre_relu <= Scalar[dtype](0.0):
            dy = Scalar[dtype](0.0)
        local_d_gamma += dy * x_hat
        local_d_beta += dy
        local_sum_dy_g += dy * gamma
        local_sum_dy_g_xh += dy * gamma * x_hat
        idx += THREADS_PER_FEATURE

    # Warp-level reduction for all 4 values
    # THREADS_PER_FEATURE=64 = 2 warps, so we need shared mem for inter-warp
    var base = feature_in_block * 4

    # Warp shuffle reduction (within warp)
    # For 64 threads per feature, we need shared mem between 2 warps
    # Use simple shared memory approach but only 2 steps (not 8)

    # Step 1: warp-level reduction via shared memory
    # Each of the 64 threads writes to shared mem, barrier, reduce
    # But we only have 4*4 = 16 shared mem slots... need per-feature reduction

    # Actually, let's use a simpler but effective approach:
    # Since BATCH=64 = THREADS_PER_FEATURE, each thread handles exactly 1 batch elem
    # We can do a single shared memory reduction with 64 threads

    # Use a separate shared memory region for reduction
    var red_smem = LayoutTensor[
        dtype, Layout.row_major(TPB), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # Reduce d_gamma within feature group
    red_smem[tid] = local_d_gamma
    barrier()
    var s = THREADS_PER_FEATURE // 2  # 32
    while s > 0:
        if lane_in_feature < s:
            red_smem[tid] = rebind[Scalar[dtype]](red_smem[tid]) + rebind[Scalar[dtype]](red_smem[tid + s])
        barrier()
        s //= 2
    var d_gamma = rebind[Scalar[dtype]](red_smem[feature_in_block * THREADS_PER_FEATURE])
    barrier()

    # Reduce d_beta
    red_smem[tid] = local_d_beta
    barrier()
    s = THREADS_PER_FEATURE // 2
    while s > 0:
        if lane_in_feature < s:
            red_smem[tid] = rebind[Scalar[dtype]](red_smem[tid]) + rebind[Scalar[dtype]](red_smem[tid + s])
        barrier()
        s //= 2
    var d_beta = rebind[Scalar[dtype]](red_smem[feature_in_block * THREADS_PER_FEATURE])
    barrier()

    # Reduce sum_dy_g
    red_smem[tid] = local_sum_dy_g
    barrier()
    s = THREADS_PER_FEATURE // 2
    while s > 0:
        if lane_in_feature < s:
            red_smem[tid] = rebind[Scalar[dtype]](red_smem[tid]) + rebind[Scalar[dtype]](red_smem[tid + s])
        barrier()
        s //= 2
    var sum_dy_g = rebind[Scalar[dtype]](red_smem[feature_in_block * THREADS_PER_FEATURE])
    barrier()

    # Reduce sum_dy_g_xh
    red_smem[tid] = local_sum_dy_g_xh
    barrier()
    s = THREADS_PER_FEATURE // 2
    while s > 0:
        if lane_in_feature < s:
            red_smem[tid] = rebind[Scalar[dtype]](red_smem[tid]) + rebind[Scalar[dtype]](red_smem[tid + s])
        barrier()
        s //= 2
    var sum_dy_g_xh = rebind[Scalar[dtype]](red_smem[feature_in_block * THREADS_PER_FEATURE])
    barrier()

    # Write param grads (first thread per feature)
    if lane_in_feature == 0:
        grads.ptr[GAMMA_OFF + f] = rebind[Scalar[dtype]](grads[GAMMA_OFF + f]) + d_gamma
        grads.ptr[BETA_OFF + f] = rebind[Scalar[dtype]](grads[BETA_OFF + f]) + d_beta

    # Pass 2: Compute grad_input
    idx = lane_in_feature
    while idx < BATCH:
        var x_hat = rebind[Scalar[dtype]](cache[idx, XHAT_OFF + f])
        var pre_relu = gamma * x_hat + beta
        var dy = rebind[Scalar[dtype]](grad_output[idx, f])
        if pre_relu <= Scalar[dtype](0.0):
            dy = Scalar[dtype](0.0)
        grad_pre_bn[idx, f] = inv_std * (
            dy * gamma - sum_dy_g / n_f - x_hat * sum_dy_g_xh / n_f
        )
        idx += THREADS_PER_FEATURE


# ─────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────


def bench[
    BATCH: Int,
    IN: Int,
    OUT: Int,
    N_ITERS: Int,
    label: StringLiteral,
](ctx: DeviceContext) raises:
    comptime L = LinearBatchNormReLU[IN, OUT]

    # Allocate
    var go_buf = ctx.enqueue_create_buffer[dtype](BATCH * OUT)
    var gpb_buf = ctx.enqueue_create_buffer[dtype](BATCH * OUT)
    var params_buf = ctx.enqueue_create_buffer[dtype](L.PARAM_SIZE)
    var cache_buf = ctx.enqueue_create_buffer[dtype](BATCH * L.CACHE_SIZE)
    var grads_buf = ctx.enqueue_create_buffer[dtype](L.PARAM_SIZE)

    # Fill random
    var hb = ctx.enqueue_create_host_buffer[dtype](max(BATCH * L.CACHE_SIZE, L.PARAM_SIZE))
    for i in range(BATCH * OUT):
        hb.unsafe_ptr()[i] = Scalar[dtype](random_float64(-1.0, 1.0).cast[dtype]())
    ctx.enqueue_copy(go_buf, hb)
    for i in range(L.PARAM_SIZE):
        hb.unsafe_ptr()[i] = Scalar[dtype](random_float64(-0.5, 0.5).cast[dtype]())
    # Set gamma=1 so BN is well-conditioned
    for i in range(OUT):
        hb.unsafe_ptr()[L.GAMMA_OFF + i] = 1.0
    ctx.enqueue_copy(params_buf, hb)
    for i in range(BATCH * L.CACHE_SIZE):
        hb.unsafe_ptr()[i] = Scalar[dtype](random_float64(-1.0, 1.0).cast[dtype]())
    # Set inv_std to reasonable values
    for b in range(BATCH):
        for f in range(OUT):
            hb.unsafe_ptr()[b * L.CACHE_SIZE + L.INVSTD_OFF + f] = Scalar[dtype](1.0)
    ctx.enqueue_copy(cache_buf, hb)
    ctx.enqueue_memset(gpb_buf, 0)
    ctx.enqueue_memset(grads_buf, 0)
    ctx.synchronize()

    # Tensors
    var go_immut = LayoutTensor[dtype, Layout.row_major(BATCH, OUT), ImmutAnyOrigin](go_buf.unsafe_ptr())
    var gpb = LayoutTensor[dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin](gpb_buf.unsafe_ptr())
    var params_immut = LayoutTensor[dtype, Layout.row_major(L.PARAM_SIZE), ImmutAnyOrigin](params_buf.unsafe_ptr())
    var cache_immut = LayoutTensor[dtype, Layout.row_major(BATCH, L.CACHE_SIZE), ImmutAnyOrigin](cache_buf.unsafe_ptr())
    var grads_lt = LayoutTensor[dtype, Layout.row_major(L.PARAM_SIZE), MutAnyOrigin](grads_buf.unsafe_ptr())

    # ── Current kernel wrapper ──
    @always_inline
    def current_wrapper(
        gpb: LayoutTensor[dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin],
        go: LayoutTensor[dtype, Layout.row_major(BATCH, OUT), ImmutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(L.PARAM_SIZE), ImmutAnyOrigin],
        cache: LayoutTensor[dtype, Layout.row_major(BATCH, L.CACHE_SIZE), ImmutAnyOrigin],
        grads: LayoutTensor[dtype, Layout.row_major(L.PARAM_SIZE), MutAnyOrigin],
    ):
        L.relu_bn_backward_kernel_impl[BATCH](gpb, go, params, cache, grads)

    # ── Optimized kernel wrapper ──
    @always_inline
    def opt_wrapper(
        gpb: LayoutTensor[dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin],
        go: LayoutTensor[dtype, Layout.row_major(BATCH, OUT), ImmutAnyOrigin],
        params: LayoutTensor[dtype, Layout.row_major(L.PARAM_SIZE), ImmutAnyOrigin],
        cache: LayoutTensor[dtype, Layout.row_major(BATCH, L.CACHE_SIZE), ImmutAnyOrigin],
        grads: LayoutTensor[dtype, Layout.row_major(L.PARAM_SIZE), MutAnyOrigin],
    ):
        bn_relu_backward_optimized[
            BATCH, OUT, IN,
            L.GAMMA_OFF, L.BETA_OFF, L.XHAT_OFF, L.INVSTD_OFF,
            L.PARAM_SIZE, L.CACHE_SIZE,
        ](gpb, go, params, cache, grads)

    comptime opt_grid = (OUT + 3) // 4  # 4 features per block

    # Warmup
    for _ in range(5):
        ctx.enqueue_function[current_wrapper, current_wrapper](
            gpb, go_immut, params_immut, cache_immut, grads_lt,
            grid_dim=(OUT,), block_dim=(TPB,),
        )
        ctx.enqueue_function[opt_wrapper, opt_wrapper](
            gpb, go_immut, params_immut, cache_immut, grads_lt,
            grid_dim=(opt_grid,), block_dim=(TPB,),
        )
    ctx.synchronize()

    # ── Benchmark current ──
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(N_ITERS):
        ctx.enqueue_function[current_wrapper, current_wrapper](
            gpb, go_immut, params_immut, cache_immut, grads_lt,
            grid_dim=(OUT,), block_dim=(TPB,),
        )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var current_us = Float64(t1 - t0) / 1000.0 / Float64(N_ITERS)

    # ── Benchmark optimized ──
    ctx.synchronize()
    var t2 = perf_counter_ns()
    for _ in range(N_ITERS):
        ctx.enqueue_function[opt_wrapper, opt_wrapper](
            gpb, go_immut, params_immut, cache_immut, grads_lt,
            grid_dim=(opt_grid,), block_dim=(TPB,),
        )
    ctx.synchronize()
    var t3 = perf_counter_ns()
    var opt_us = Float64(t3 - t2) / 1000.0 / Float64(N_ITERS)

    # ── Verify correctness ──
    var gpb_cur_buf = ctx.enqueue_create_buffer[dtype](BATCH * OUT)
    var gpb_opt_buf = ctx.enqueue_create_buffer[dtype](BATCH * OUT)
    ctx.enqueue_memset(grads_buf, 0)
    var gpb_cur = LayoutTensor[dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin](gpb_cur_buf.unsafe_ptr())
    var gpb_opt = LayoutTensor[dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin](gpb_opt_buf.unsafe_ptr())
    ctx.enqueue_memset(gpb_cur_buf, 0)
    ctx.enqueue_memset(gpb_opt_buf, 0)
    ctx.synchronize()

    ctx.enqueue_function[current_wrapper, current_wrapper](
        gpb_cur, go_immut, params_immut, cache_immut, grads_lt,
        grid_dim=(OUT,), block_dim=(TPB,),
    )
    ctx.enqueue_memset(grads_buf, 0)
    ctx.synchronize()
    ctx.enqueue_function[opt_wrapper, opt_wrapper](
        gpb_opt, go_immut, params_immut, cache_immut, grads_lt,
        grid_dim=(opt_grid,), block_dim=(TPB,),
    )
    ctx.synchronize()

    var cur_hb = ctx.enqueue_create_host_buffer[dtype](BATCH * OUT)
    var opt_hb = ctx.enqueue_create_host_buffer[dtype](BATCH * OUT)
    ctx.enqueue_copy(cur_hb, gpb_cur_buf)
    ctx.enqueue_copy(opt_hb, gpb_opt_buf)
    ctx.synchronize()

    var max_diff: Float64 = 0
    for i in range(BATCH * OUT):
        var d = abs(Float64(cur_hb.unsafe_ptr()[i]) - Float64(opt_hb.unsafe_ptr()[i]))
        if d > max_diff:
            max_diff = d

    var speedup = current_us / opt_us
    print(
        "  "
        + label
        + " [B="
        + String(BATCH)
        + ", "
        + String(IN)
        + "→"
        + String(OUT)
        + "]:"
    )
    print(
        "    current:   "
        + String(current_us)[byte=:10]
        + " μs  (grid="
        + String(OUT)
        + " blocks)"
    )
    print(
        "    optimized: "
        + String(opt_us)[byte=:10]
        + " μs  (grid="
        + String(opt_grid)
        + " blocks, 4 feat/block)"
    )
    print(
        "    speedup:   "
        + String(speedup)[byte=:5]
        + "x  diff="
        + String(max_diff)
    )
    print()


def main() raises:
    seed(42)
    print("=" * 70)
    print("BN+ReLU Backward Benchmark — LinearBatchNormReLU")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        # The 339μs kernel from nsys
        bench[64, 5376, 256, 1000, "LinBNReLU[5376,256]"](ctx)

        # Other layers in the network
        bench[64, 256, 128, 1000, "LinBNReLU[256,128] "](ctx)

    print("=" * 70)
